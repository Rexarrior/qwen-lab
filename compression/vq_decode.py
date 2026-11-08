"""Standalone VQ kit decoder: vq-d4-k256 projections + scalar-K16 embedding.

No encoder imports. Original/cache reads are forbidden in a separate process
via PRACTICAL_DENIED_PATHS and a filesystem audit hook.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def unpack(packed, bits, count):
    stride = 8 // bits
    assert bits in (2, 4) and len(packed) == count // stride
    codes = np.empty(count, np.uint8)
    for j in range(stride):
        codes[j::stride] = (packed >> (bits * j)) & ((1 << bits) - 1)
    return codes


def native(kit, item):
    kit = Path(kit)
    shape = item['shape']
    rows, nin = shape
    count = int(np.prod(shape))
    fmt = item['format']
    scales = np.fromfile(kit / item['scales'], '<f2').astype(np.float32) \
        .reshape(rows, -1)
    # Signed scales are legal: recovery training may flip a group's sign
    # (P7: 564/4.66M groups, mostly in the tied embedding head).
    assert scales.size == count // 128 and (scales != 0).all() \
        and np.isfinite(scales).all()
    b_dtype = '<f2' if item['book'].endswith('.f16') else '<f4'
    if fmt in ('vq-d4-k256', 'vq-d2-k256'):
        assert item['bits'] == 8
        d = item['d']
        assert item['k'] == 256 and nin % d == 0
        book = np.fromfile(kit / item['book'], b_dtype).astype(np.float32).reshape(item['k'], d)
        codes = np.fromfile(kit / item['codes'], np.uint8).reshape(rows, nin // d)
        rec = book[codes].reshape(rows, nin)
    elif fmt == 'scalar-k16':
        assert item['bits'] == 4
        book = np.fromfile(kit / item['book'], b_dtype).astype(np.float32)
        assert len(book) == 16
        codes = unpack(np.fromfile(kit / item['codes'], np.uint8), 4, count) \
            .reshape(rows, nin)
        rec = book[codes]
    else:
        raise AssertionError(f'unknown format {fmt}')
    out = np.empty((rows, nin), np.float32)
    for g in range(nin // 128):
        out[:, g * 128:(g + 1) * 128] = np.multiply(
            rec[:, g * 128:(g + 1) * 128], scales[:, g, None], dtype=np.float32)
    assert np.isfinite(out).all()
    return out


def denied_hook(roots):
    roots = [Path(p).resolve() for p in roots]

    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            p = Path(os.fsdecode(args[0])).resolve()
            if any(p == root or root in p.parents for root in roots):
                raise PermissionError('Source read forbidden')
    return audit


def decode(kit, output):
    start = time.perf_counter()
    kit = Path(kit).resolve()
    output = Path(output)
    denied = json.loads(os.environ['PRACTICAL_DENIED_PATHS'])
    assert denied
    sys.addaudithook(denied_hook(denied))
    try:
        open(denied[0], 'rb')
    except PermissionError:
        pass
    else:
        raise AssertionError('source denial inactive')
    recipe = json.loads((kit / 'recipe.json').read_text())
    for name, meta in recipe['manifest'].items():
        assert (kit / name).stat().st_size == meta['bytes'] \
            and sha(kit / name) == meta['sha256']
    tensors = {}
    with safe_open(kit / 'retained.safetensors', framework='pt', device='cpu') as f:
        for n in f.keys():
            tensors[n] = f.get_tensor(n)
    assert len(tensors) == 113 \
        and all(t.ndim == 1 and t.dtype == torch.float32 for t in tensors.values())
    assert sum(t.numel() for t in tensors.values()) == 65536 \
        and len(recipe['matrices']) == 197
    for item in recipe['matrices']:
        out = native(kit, item)
        assert hashlib.sha256(out.tobytes()).hexdigest() \
            == item['decoded_fp32_sha256']
        endpoint = out.astype('<f2')
        assert np.isfinite(endpoint).all()
        assert hashlib.sha256(endpoint.tobytes()).hexdigest() \
            == item['endpoint_f16_sha256']
        tensors[item['name']] = torch.from_numpy(endpoint)
    assert len(tensors) == 310 \
        and recipe['aliases'] == {'lm_head.weight': 'model.embed_tokens.weight'}
    save_file(tensors, output, metadata={'format': 'pt'})
    print(json.dumps({'event': 'independent_decoder', 'matrices': 197,
                      'norms': 113, 'all_hashes_pass': True,
                      'source_reads_forbidden': True,
                      'bytes': output.stat().st_size, 'sha256': sha(output),
                      'seconds': time.perf_counter() - start}), flush=True)


if __name__ == '__main__':
    decode(sys.argv[1], sys.argv[2])
