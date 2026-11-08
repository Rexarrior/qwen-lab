"""Fit one Qwen projection from its weights and calibration inputs."""
import argparse
import numpy as np
import torch
from vq_core import VQCodec, h_block_list, feedback_vq

def scales_from_source(w):
    x=w.astype(np.float64).reshape(-1,128);s=np.sqrt(np.mean(x*x,axis=1));s[s==0]=1
    s=s.astype('<f2');assert (s>0).all() and np.isfinite(s).all();return s.reshape(w.shape[0],-1)

def inverse_factor(h):
    """U upper and U.T@U = inverse(H+.01mean(diag)I)."""
    t=torch.from_numpy(np.ascontiguousarray(h,dtype=np.float64));t=(t+t.T)*.5
    mean=float(torch.diagonal(t).mean());assert np.isfinite(mean) and mean>0,'unsupported Gram trace'
    lam=.01*mean;t.diagonal().add_(lam)
    chol=torch.linalg.cholesky(t);inv=torch.cholesky_inverse(chol)
    u=torch.linalg.cholesky(inv,upper=True)
    residual=float(torch.linalg.norm(u.T@u-inv)/torch.linalg.norm(inv))
    inverse_residual=float(torch.linalg.norm(t@inv-torch.eye(len(t),dtype=torch.float64))/len(t)**.5)
    assert residual<1e-8 and inverse_residual<1e-7
    eig=torch.linalg.eigvalsh(t)
    return u.numpy(),{'lambda':lam,'normalization':'mean xxT over all calibration positions, not NLL Hessian',
                      'minimum_eigenvalue':float(eig[0]),'maximum_eigenvalue':float(eig[-1]),'condition':float(eig[-1]/eig[0]),
                      'U_T_U_inverse_relative_residual':residual,'regularized_inverse_residual':inverse_residual,'factor_orientation':'upper'}

def fit(w, x, d, seed=123):
    if w.ndim != 2 or x.ndim != 2 or w.shape[1] != x.shape[1] or w.shape[1] % 128 or d not in (2,4):
        raise ValueError("Supply W[outputs, inputs] and X[examples, inputs], width divisible by 128, d=2 or 4")
    if w.size // d < 256 or not len(x):
        raise ValueError("Need at least 256 training blocks and nonempty calibration data")
    x = x.astype(np.float64)
    gram = x.T @ x / len(x)
    s = scales_from_source(w).astype(np.float64)
    norm = (w / np.repeat(s,128,axis=1)).reshape(w.shape[0], -1, d)
    codec = VQCodec(d, 256, 1, seed)
    blocks = h_block_list(gram,d)
    books = codec.init_books(norm)
    for _ in range(3):
        codes = codec.assign(norm,books,blocks)
        books,_ = codec.em_update(norm,codes,blocks,s*s,books)
    u, _ = inverse_factor(gram)
    codes,_ = feedback_vq(w,s,books,d,u,blocks)
    return dict(book=books[0].astype('<f2'), scales=s.astype('<f2'), codes=codes[0])

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('weights','calibration','out'): p.add_argument(name)
    p.add_argument('--d',type=int,choices=[2,4],default=4)
    a=p.parse_args()
    np.savez(a.out,**fit(np.load(a.weights,allow_pickle=False),np.load(a.calibration,allow_pickle=False),a.d))
