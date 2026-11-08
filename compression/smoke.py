"""Small numerical checks; does not download models or run benchmarks."""
import numpy as np
import torch
from fit_projection import fit
from vq_core import decode_codes
from vq_recovery import VQBookEntry, VQBilinearEntry

def main():
    rng = np.random.default_rng(123)
    w = rng.normal(size=(32,128)).astype('f4')
    xs = rng.normal(size=(160,128)).astype('f4')
    q = fit(w, xs, 4)
    rec = decode_codes([q['codes']], [q['book']], q['scales'], 4)
    assert np.linalg.norm(w-rec) < np.linalg.norm(w)*0.65
    book = q['book'].astype('f4')
    scales = q['scales'].astype('f4')
    for typ in (VQBookEntry,VQBilinearEntry):
        e=typ('toy',book,q['codes'],scales,4)
        leaf=e.make_leaf_weight()
        upstream=torch.from_numpy(w)
        (leaf*upstream).sum().backward()
        if typ is VQBookEntry: e.accumulate_book_grad(leaf)
        else: e.accumulate_grads(leaf)
        refbook=torch.tensor(book,requires_grad=True)
        refscale=torch.tensor(scales,requires_grad=True)
        direct=refbook[q['codes'].astype('int64')].reshape(32,1,128)*refscale[:,:,None]
        (direct.reshape_as(upstream)*upstream).sum().backward()
        torch.testing.assert_close(e.book.grad,refbook.grad)
        if typ is VQBilinearEntry: torch.testing.assert_close(e.scales.grad, refscale.grad)
    print('PASS: codebook round trip and both recovery gradients')

if __name__=='__main__':main()
