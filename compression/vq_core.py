"""Vector codebooks: weighted EM fit + GPTQ-style block error feedback.

Methodology: AQLM/GPTVQ-style input-adaptive vector quantization. Blocks of d
consecutive input columns share a per-matrix codebook (or M additive books).
Assignment uses per-position Mahalanobis quadratics from the exact calibration
Gram H = X X^T (no diagonal-only proxy). Books update by exact weighted EM
(normal equations per codeword). Codes are finally reassigned by a GPTQ-style
sequential block feedback with the damped-Gram Cholesky factor U (U^T U =
(H + lambda I)^-1), generalized from per-column to joint d-column blocks:
e = E @ inv(U[J, J]); current[:, after] -= e @ U[J, after].
"""
import numpy as np

TINY = np.finfo(np.float64).tiny
SCORE_BUDGET = 2 ** 23


def h_block_list(h, d):
    """(nin, nin) Gram -> (nin//d, d, d) per-position diagonal blocks."""
    nin = h.shape[0]
    assert nin % d == 0
    P = nin // d
    return np.stack([h[p * d:(p + 1) * d, p * d:(p + 1) * d] for p in range(P)])


def identity_blocks(P, d):
    return np.tile(np.eye(d, dtype=np.float64), (P, 1, 1))


def kmeanspp_init(blocks, k, rng):
    """k-means++ seeding on a block sample; returns (k, d)."""
    n = len(blocks)
    first = int(rng.integers(n))
    chosen = [blocks[first]]
    closest = np.sum((blocks - chosen[0]) ** 2, axis=1)
    for _ in range(1, k):
        total = float(closest.sum())
        if total <= 0:
            pick = int(rng.integers(n))
        else:
            pick = int(np.searchsorted(np.cumsum(closest), rng.random() * total, side='right'))
            pick = min(pick, n - 1)
        chosen.append(blocks[pick])
        closest = np.minimum(closest, np.sum((blocks - blocks[pick]) ** 2, axis=1))
    return np.stack(chosen).astype(np.float64)


def assign_all(u, book, hblocks):
    """argmin_k (u - c_k) hb (u - c_k)^T per row and position, vectorized.

    u: (rows, P, d) scaled units; book: (k, d); hblocks: (P, d, d).
    Returns (rows, P) uint8 codes; lowest index wins ties.
    """
    rows, P, d = u.shape
    k = len(book)
    vh = np.einsum('rpd,pde->rpe', u, hblocks, optimize=True)
    c_e = np.einsum('kd,pde,ke->pk', book, hblocks, book, optimize=True)
    codes = np.empty((rows, P), np.uint8)
    chunk = max(1, SCORE_BUDGET // max(1, P * k))
    for begin in range(0, rows, chunk):
        end = min(begin + chunk, rows)
        scores = -2.0 * np.einsum('rpe,ke->rpk', vh[begin:end], book, optimize=True) \
            + c_e[None, :, :]
        codes[begin:end] = np.argmin(scores, axis=2)
    return codes


def assign_position(v, book, hb):
    """Single-position assignment; v: (rows, d), book: (k, d), hb: (d, d)."""
    vh = v @ hb
    scores = -2.0 * (vh @ book.T) + np.einsum('kd,de,ke->k', book, hb, book)
    return np.argmin(scores, axis=1)


def group_of_position(P, d):
    return (np.arange(P) * d) // 128


def weighted_objective(u, rec, hblocks, s2, d):
    """sum_{r,p} s^2 (u - rec) hb (u - rec)^T, exact per scale group."""
    rows, P, _ = u.shape
    total = 0.0
    groups = group_of_position(P, d)
    for p in range(P):
        delta = u[:, p, :] - rec[:, p, :]
        quad = np.sum((delta @ hblocks[p]) * delta, axis=1)
        total += float(np.sum(s2[:, groups[p]] * quad))
    return total


def sse_objective(u, rec, s2, d):
    """Original-unit SSE sum_{r,p} s^2 |u - rec|^2 (w - s c)^2."""
    rows, P, _ = u.shape
    diff = u - rec
    per = np.sum(diff * diff, axis=2)
    groups = group_of_position(P, d)
    return float(np.sum(per * s2[:, groups]))


def em_update_single(u, codes, hblocks, s2, book_prev, d, ridge_rel=1e-9):
    """Exact weighted EM: minimize sum (w - s c_z) hb (w - s c_z)^T over the
    shared book. u: (rows, P, d); codes: (rows, P); s2: (rows, groups)."""
    rows, P, dd = u.shape
    k = len(book_prev)
    W2 = np.zeros((P, k))
    S = np.zeros((P, k, dd))
    groups = group_of_position(P, d)
    for p in range(P):
        w2 = s2[:, groups[p]]
        z = codes[:, p]
        W2[p] = np.bincount(z, weights=w2, minlength=k)
        for j in range(dd):
            S[p, :, j] = np.bincount(z, weights=w2 * u[:, p, j], minlength=k)
    A = np.einsum('pk,pde->kde', W2, hblocks, optimize=True)
    b = np.einsum('pkd,pde->ke', S, hblocks, optimize=True)
    diag = np.maximum(np.diagonal(A, axis1=1, axis2=2).sum(axis=1), TINY)
    empty = diag <= TINY
    ridge = ridge_rel * diag
    Areg = A + ridge[:, None, None] * np.eye(dd)[None, :, :]
    sol = np.linalg.solve(Areg, b[:, :, None])[:, :, 0]
    book = np.where(empty[:, None], book_prev, sol)
    return book, {'empty_codewords': int(empty.sum())}


class VQCodec:
    """Single or additive-M codebooks over d-column blocks, scaled units."""

    def __init__(self, d, k, m, seed):
        self.d = int(d)
        self.k = int(k)
        self.m = int(m)
        self.rng = np.random.default_rng(seed)

    def init_books(self, u):
        rows, P, d = u.shape
        target = 20000
        hb = identity_blocks(P, d)
        take = max(1, target // P)
        row_stride = max(1, (rows + take - 1) // take)
        sample = u[::row_stride]
        blocks = sample.reshape(-1, d)
        if self.m == 1:
            return [kmeanspp_init(blocks, self.k, self.rng)]
        book1 = kmeanspp_init(blocks, self.k, self.rng)
        z1 = assign_all(sample, book1, hb)
        resid = sample - book1[z1]
        book2 = kmeanspp_init(resid.reshape(-1, d), self.k, self.rng)
        return [book1, book2]

    def assign(self, u, books, hblocks, rounds=2):
        rows, P, d = u.shape
        if self.m == 1:
            return [assign_all(u, books[0], hblocks)]
        z2 = np.zeros((rows, P), np.uint8)
        for _ in range(max(1, rounds)):
            r1 = u - books[1][z2]
            z1 = assign_all(r1, books[0], hblocks)
            r2 = u - books[0][z1]
            z2 = assign_all(r2, books[1], hblocks)
        return [z1, z2]

    def em_update(self, u, codes, hblocks, s2, books):
        if self.m == 1:
            book, info = em_update_single(u, codes[0], hblocks, s2, books[0], self.d)
            return [book], info
        total_empty = 0
        new_books = []
        for m in range(self.m):
            other = 1 - m
            resid = u - books[other][codes[other]]
            book, info = em_update_single(resid, codes[m], hblocks, s2, books[m], self.d)
            new_books.append(book)
            total_empty += info['empty_codewords']
        return new_books, {'empty_codewords': total_empty}

    def scaled_reconstruction(self, codes, books):
        rec = books[0][codes[0]].astype(np.float64)
        for m in range(1, self.m):
            rec = rec + books[m][codes[m]]
        return rec


def feedback_vq(w, scales, books, d, ufac, hblocks, block=128,
                check=lambda: None, add_rounds=1):
    """GPTQ-style sequential joint-block feedback.

    w: (rows, nin) source weights F32; scales: (rows, groups) F64 from
    serialized FP16; ufac: upper Cholesky factor of the damped inverse Gram;
    hblocks: (P, d, d) position blocks for the assignment metric.
    Returns codes per book plus diagnostics.
    """
    rows, nin = w.shape
    assert nin % block == 0 and block % d == 0
    P = nin // d
    m = len(books)
    codes = [np.zeros((rows, P), np.uint8) for _ in range(m)]
    current = w.astype(np.float64).copy()
    quad = 0.0
    for begin in range(0, nin, block):
        check()
        end = begin + block
        tile = current[:, begin:end].copy()
        e_all = np.empty((rows, block))
        for sub in range(block // d):
            J0 = begin + sub * d
            J1 = J0 + d
            g = J0 // 128
            pos = J0 // d
            v = tile[:, J0 - begin:J1 - begin]
            u_v = v / scales[:, g, None]
            if m == 1:
                z = assign_position(u_v, books[0], hblocks[pos])
                codes[0][:, pos] = z
                q = books[0][z] * scales[:, g, None]
            else:
                z1 = codes[0][:, pos].copy()
                z2 = codes[1][:, pos].copy()
                for _ in range(add_rounds):
                    r = u_v - books[1][z2]
                    z1 = assign_position(r, books[0], hblocks[pos])
                    r2 = u_v - books[0][z1]
                    z2 = assign_position(r2, books[1], hblocks[pos])
                codes[0][:, pos] = z1
                codes[1][:, pos] = z2
                q = (books[0][z1] + books[1][z2]) * scales[:, g, None]
            E = v - q
            quad += float(np.sum(E * E))
            Usq = ufac[J0:J1, J0:J1]
            e = np.linalg.solve(Usq.T, E.T).T
            e_all[:, J0 - begin:J1 - begin] = e
            if J1 < end:
                tile[:, J1 - begin:] -= e @ ufac[J0:J1, J1:end]
        if end < nin:
            current[:, end:] -= e_all @ ufac[begin:end, end:]
    return codes, {'conditional_regularized_quadratic_increment_sum': quad,
                   'feedback_block': block, 'column_order': 'default',
                   'act_order': False, 'd': d, 'm': m}


def decode_codes(codes, books, scales, d):
    """Reconstruction in original units from codes/books/scales (F64 path)."""
    rows, P = codes[0].shape
    rec = books[0][codes[0]].astype(np.float64)
    for m in range(1, len(books)):
        rec = rec + books[m][codes[m]]
    rec = rec.reshape(rows, P * d)
    out = np.empty_like(rec)
    for g in range(rec.shape[1] // 128):
        out[:, g * 128:(g + 1) * 128] = rec[:, g * 128:(g + 1) * 128] * scales[:, g, None]
    return out


def exact_quadratic(delta, h):
    """tr(delta h delta^T) with h = X X^T unnormalized (exact ||delta X||_F^2)."""
    prod = delta @ h
    return float(np.sum(prod * delta))
