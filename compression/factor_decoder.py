"""Deployment decoder: NumPy only, no teacher, encoder, original factors or HF."""
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(values):
    assert values.ndim == 2 and values.shape[1] == 8
    assert np.all((values >= -8) & (values <= 7))
    n = (values.astype(np.int16) + 8).astype(np.uint8)
    return n[:, ::2] | (n[:, 1::2] << 4)


def unpack(values):
    assert values.dtype == np.uint8 and values.ndim == 2 and values.shape[1] == 4
    result=np.empty((len(values),8),dtype=np.int16)
    result[:,::2]=(values & 15).astype(np.int16)-8
    result[:,1::2]=(values >> 4).astype(np.int16)-8
    return result


def condition(family, role, layer, count):
    c = np.zeros((count, 10), dtype=np.float32)
    c[:, family] = 1
    c[:, 7+role] = 1
    c[:, 9] = layer/27
    return c


def decode_vector(codes, cond, params, film=None):
    x = np.concatenate((codes.astype(np.float32)/4, cond), axis=1)
    for i in range(3):
        x = x @ params[str(i)+'.weight'].astype(np.float32).T + params[str(i)+'.bias'].astype(np.float32)
        if i != 2:
            if film is not None:
                gain,bias=film
                x=x*gain[i].astype(np.float32)+bias[i].astype(np.float32)
            x = x/(1+np.exp(-np.clip(x, -80, 80)))
    assert np.isfinite(x).all()
    return x + codes.astype(np.float32)/2


def decode_kit(directory):
    root = Path(directory)
    meta = json.loads((root/'factor-codec.json').read_text())
    assert meta['format'] in ('OpenResearch-rank8-factor-codec-v1','OpenResearch-rank8-factor-codec-v2-film')
    assert meta['rank'] == 8 and meta['decoder_dimensions'] == [18, 64, 64, 8]
    for name, record in meta['payload_files'].items():
        assert (root/name).stat().st_size == record['bytes'] and digest(root/name) == record['sha256'], name
    packed = np.load(root/'factor-codes.npy', allow_pickle=False)
    assert packed.shape == (630784, 4)
    codes = unpack(packed)
    assert np.array_equal(pack(codes), packed), 'INT4 packing roundtrip'
    with np.load(root/'factor-decoder.npz', allow_pickle=False) as archive:
        params = {k:archive[k].copy() for k in archive.files}
    assert len(params) == 6 and all(p.dtype == np.float16 and np.isfinite(p).all() for p in params.values())
    for i,(a,b) in enumerate(zip([18,64,64],[64,64,8])):
        assert params[str(i)+'.weight'].shape == (b,a) and params[str(i)+'.bias'].shape == (b,)
    film=None
    if meta['format'].endswith('v2-film'):
        assert meta['decoder_film']['file']=='decoder-film.npz'
        with np.load(root/'decoder-film.npz',allow_pickle=False) as archive:
            assert set(archive.files)=={'gain','bias'}
            film=tuple(archive[k].copy() for k in ('gain','bias'))
        assert all(a.shape==(28,2,64) and a.dtype==np.float16 and np.isfinite(a).all() for a in film)
    factors = {}
    cursor = 0
    for row in meta['factors']:
        assert row['offset'] == cursor
        assert row['role'] in (0,1) and 0 <= row['family'] < 7 and 0 <= row['layer'] < 28
        assert np.isfinite(row['FP16_RMS']) and row['FP16_RMS'] > 0
        cond = condition(row['family'], row['role'], row['layer'], row['vectors'])
        values = []
        for start in range(0, row['vectors'], 4096):
            z = codes[cursor+start:cursor+min(start+4096, row['vectors'])]
            local_film=None if film is None else tuple(a[row['layer']] for a in film)
            value = decode_vector(z, cond[start:start+4096], params,local_film)*np.float32(row['FP16_RMS'])
            values.append(value.astype('<f2'))
        values = np.concatenate(values)
        array = values.T.copy() if row['role'] == 0 else values
        assert list(array.shape) == row['shape'] and np.isfinite(array).all()
        cursor += row['vectors']
        factors.setdefault(row['name'], [None, None])[row['role']] = array
    assert cursor == len(codes) and len(factors) == 196
    return {k:tuple(v) for k,v in factors.items()}, meta
