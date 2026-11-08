"""Generate the measured llama.cpp patch from P9 and P10 portable recipes."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def build_codebook_tables(candidates):
    """Collect all d=4 (256x4) and d=2 (256x2) codebooks across P9_vq and P10_vq."""
    books_d4 = []  # list of (256, 4) uint16 arrays
    books_d2 = []  # list of (256, 2) uint16 arrays
    book_map = {}  # (kind, matrix_name) -> ('d4' | 'd2', book_id)

    for cand in candidates:
        kind = cand['kind']
        kit = Path(cand['kit'])
        recipe = json.loads((kit / 'recipe.json').read_text())
        for item in recipe['matrices']:
            name = item['name']
            fmt = item['format']
            b_dtype = '<f2' if item['book'].endswith('.f16') else '<f4'
            raw_b = np.fromfile(kit / item['book'], b_dtype).astype(np.float16)
            if fmt == 'vq-d4-k256':
                b_u16 = raw_b.view(np.uint16).reshape(256, 4)
                bid = len(books_d4)
                books_d4.append(b_u16)
                book_map[(kind, name)] = ('d4', bid)
            elif fmt == 'vq-d2-k256':
                b_u16 = raw_b.view(np.uint16).reshape(256, 2)
                bid = len(books_d2)
                books_d2.append(b_u16)
                book_map[(kind, name)] = ('d2', bid)
            elif fmt == 'scalar-k16':
                assert len(raw_b) == 16
                # Exact representation of scalar-k16 packed byte u = a | (b << 4) as a 256x2 codebook!
                u = np.arange(256, dtype=np.uint16)
                b256 = np.stack([raw_b[u & 0xF], raw_b[u >> 4]], axis=1).astype(np.float16)
                b_u16 = b256.view(np.uint16).reshape(256, 2)
                bid = len(books_d2)
                books_d2.append(b_u16)
                book_map[(kind, name)] = ('d2', bid)
            else:
                raise ValueError(f'Unknown format: {fmt}')

    arr_d4 = np.concatenate([b.reshape(-1) for b in books_d4], axis=0)
    arr_d2 = np.concatenate([b.reshape(-1) for b in books_d2], axis=0)
    return arr_d4, arr_d2, book_map

def format_u16_table(name, arr):
    lines = [f'GGML_TABLE_BEGIN(uint16_t, {name}, {len(arr)})']
    chunk = 16
    for i in range(0, len(arr), chunk):
        vals = ', '.join(f'0x{int(v):04x}' for v in arr[i:i + chunk])
        lines.append(f'    {vals},')
    lines.append('GGML_TABLE_END()')
    return '\n'.join(lines)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('p9');p.add_argument('p10');p.add_argument('out')
    a=p.parse_args()
    arr4,arr2,_=build_codebook_tables([dict(kind='P9_vq',kit=a.p9),dict(kind='P10_vq',kit=a.p10)])
    text=Path(__file__).with_name('native-vq.patch.in').read_text()
    for n,arr in [('d4',arr4),('d2',arr2)]:
        text=text.replace('@@GENERATE_'+n+'@@','\n'.join('+'+s for s in format_u16_table('vq_books_'+n+'_u16',arr).splitlines()))
    data=text.encode()
    if hashlib.sha256(data).hexdigest() != "c5eea136dca0db3d3c02c8c38076716588a9a8bb83b6ceb89e198443986c2d4a":
        raise ValueError("Patch differs from studied runtime. Verify model kits and argument order: P9 then P10.")
    Path(a.out).write_bytes(data)
