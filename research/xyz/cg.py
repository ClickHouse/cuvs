"""CAGRA graph-pruning / seed research helpers (cupy + ctypes shim over libcuvs)."""
import ctypes, os, time
import numpy as np
import cupy as cp

D = '/data/scratch/ann-data'
HERE = os.path.dirname(os.path.abspath(__file__))
_lib = None


def lib():
    global _lib
    if _lib is None:
        ctypes.CDLL('/data/scratch/cuvs-install/lib/libcuvs.so', mode=ctypes.RTLD_GLOBAL)
        _lib = ctypes.CDLL(os.path.join(HERE, 'libshim.so'))
        P, I, F = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        _lib.shim_index.restype = P
        _lib.shim_index.argtypes = [P, I, I, P, I]
        _lib.shim_free.argtypes = [P]
        _lib.shim_search.restype = ctypes.c_double
        _lib.shim_search.argtypes = [P, P, I, I, F, F, F, F, F, F, ctypes.c_ulonglong, P, P]
        _lib.shim_bf_knn.argtypes = [P, I, I, P, I, F, P, P]
        _lib.shim_optimize.argtypes = [P, I, I, P, I, F]
        _lib.shim_build.argtypes = [P, I, I, I, I, P]
        _lib.shim_ivfpq_knn.argtypes = [P, I, I, P, I]
        _lib.shim_nnd_knn.argtypes = [P, I, I, P, I]
        _lib.shim_select_k.argtypes = [P, I, I, F, P, P]
        if hasattr(_lib, 'cuvs_cagra_experimental_set_seeds'):
            _lib.cuvs_cagra_experimental_set_seeds.argtypes = [P, ctypes.c_uint32]
    return _lib


def ptr(a):
    return ctypes.c_void_p(a.data.ptr)


def load(name):
    p = f'{D}/{name}'
    base = np.load(f'{p}/base.npy')
    q = np.load(f'{p}/query.npy')
    gt = np.load(f'{p}/gt.npy')
    return base, q, gt


def knn_graph(name, X, K=128):
    f = f'{D}/{name}/knn{K}.npy'
    if os.path.exists(f):
        return np.load(f)
    n, dim = X.shape
    nb = cp.empty((n, K + 1), cp.int64)
    dd = cp.empty((n, K + 1), cp.float32)
    lib().shim_bf_knn(ptr(X), n, dim, ptr(X), n, K + 1, ptr(nb), ptr(dd))
    nb = nb.get()
    # drop self (usually col 0, but duplicates may reorder)
    out = np.empty((n, K), np.uint32)
    ar = np.arange(n)[:, None]
    m = nb != ar
    # keep first K non-self entries
    idx = np.argsort(~m, axis=1, kind='stable')[:, :K]
    out[:] = np.take_along_axis(nb, idx, 1)
    np.save(f, out)
    return out


# ----------------------------------------------------------------------------------------------
# Baseline rank-based detour counting prune (mirror of kern_fused_prune, constant degree).
# One warp per node; lane loops over kDB, binary search into A's (id,rank) sorted list.
_detour_src = r'''
extern "C" __global__ void detour_count(const unsigned* __restrict__ knn, long long n, int K,
                                        unsigned short* __restrict__ cnt_out, int mode)
{
  // mode 0: stock (count D->B for any kDB);  mode 1: require kDB < kAB (commented-out cond)
  extern __shared__ unsigned smem[];
  const int wid = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
  long long a = (long long)blockIdx.x * nw + wid;
  if (a >= n) return;
  unsigned* ids   = smem + wid * (3 * K);       // sorted ids
  unsigned* rk    = ids + K;                     // rank of sorted id
  unsigned* cnt   = rk + K;                      // counts by rank
  const unsigned* row = knn + a * K;
  // load + bitonic-free sort: simple rank computation (K<=256), O(K^2/32)
  for (int i = lane; i < K; i += 32) {
    unsigned v = row[i]; int pos = 0;
    for (int j = 0; j < K; j++) { unsigned w = row[j]; pos += (w < v) || (w == v && j < i); }
    ids[pos] = v; rk[pos] = i; cnt[i] = 0;
  }
  __syncwarp();
  for (int kAD = 0; kAD < K - 1; kAD++) {
    unsigned d = row[kAD];
    if (d >= n) continue;
    for (int kDB = lane; kDB < K; kDB += 32) {
      unsigned b = knn[(long long)d * K + kDB];
      int lo = 0, hi = K;
      while (lo < hi) { int mid = (lo + hi) >> 1; if (ids[mid] < b) lo = mid + 1; else hi = mid; }
      // duplicates: take the first occurrence with rank > kAD (matches 'break' after first hit)
      int best = 1 << 30;
      for (int p = lo; p < K && ids[p] == b; p++) if ((int)rk[p] > kAD && (int)rk[p] < best) best = rk[p];
      if (best < (1 << 30) && (mode == 0 || kDB < best)) atomicAdd(cnt + best, 1u);
    }
    __syncwarp();
  }
  for (int i = lane; i < K; i += 32) cnt_out[a * K + i] = (unsigned short)min(cnt[i], 65535u);
}
'''
_detour = cp.RawKernel(_detour_src, 'detour_count')


def detour_counts(knn_d, mode=0):
    n, K = knn_d.shape
    cnt = cp.empty((n, K), cp.uint16)
    nw = 8
    _detour(((n + nw - 1) // nw,), (32 * nw,), (knn_d, cp.int64(n), cp.int32(K), cnt, cp.int32(mode)),
            shared_mem=nw * 3 * K * 4)
    return cnt


def select_by_score(knn_d, score, d, row0=0, n_total=None, chunk=1_000_000):
    """Pick d entries with smallest (score, rank); returns [n, d] ids in that order. Processed in row chunks."""
    n, K = knn_d.shape
    if n > chunk:
        out = cp.empty((n, d), knn_d.dtype)
        for s0 in range(0, n, chunk):
            out[s0:s0 + chunk] = select_by_score(knn_d[s0:s0 + chunk], score[s0:s0 + chunk], d, row0 + s0, n_total or n, chunk)
        return out
    n_total = n_total or n
    sc = score.astype(cp.int64)
    sc = sc + K * (knn_d == cp.arange(row0, row0 + n, dtype=cp.uint32)[:, None])  # deprioritize self-edge (as cuVS)
    sc = cp.where(knn_d >= n_total, 1 << 20, sc)                       # invalid entries last
    key = sc * K + cp.arange(K, dtype=cp.int64)[None, :]
    o = cp.argsort(key, axis=1)[:, :d]
    return cp.take_along_axis(knn_d, o, 1)


# ----------------------------------------------------------------------------------------------
# Reverse-edge merge (mirror of kern_make_rev_graph_k + kern_merge_graph, constant degree, no MST)
_merge_src = r'''
extern "C" __global__ void make_rev(const unsigned* __restrict__ g, long long n, int d, int k,
                                    unsigned* rev, unsigned* rev_cnt)
{
  long long s = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (s >= n) return;
  unsigned t = g[s * d + k];
  if (t >= n) return;
  unsigned pos = atomicAdd(rev_cnt + t, 1u);
  if (pos < (unsigned)d) rev[(long long)t * d + pos] = (unsigned)s;
}
extern "C" __global__ void merge(unsigned* g, long long n, int d, const unsigned* rev,
                                 const unsigned* rev_cnt, int nprot)
{
  long long a = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (a >= n) return;
  unsigned buf[256];
  for (int i = 0; i < d; i++) buf[i] = g[a * d + i];
  int kr = min(rev_cnt[a], (unsigned)d);
  while (kr) {
    kr--;
    unsigned v = rev[a * d + kr];
    if (v >= n) continue;
    int pos = d;
    for (int i = 0; i < d; i++) if (buf[i] == v) { pos = i; break; }
    if (pos < nprot) continue;
    int ns = pos - nprot;
    if (pos >= d) ns = d - nprot - 1;
    for (int i = nprot + ns; i > nprot; i--) buf[i] = buf[i - 1];
    buf[nprot] = v;
  }
  for (int i = 0; i < d; i++) g[a * d + i] = buf[i];
}
'''
_mm = cp.RawModule(code=_merge_src)
_make_rev = _mm.get_function('make_rev')
_merge = _mm.get_function('merge')


def merge_reverse(pruned, nprot=None):
    g = pruned.copy()
    n, d = g.shape
    if nprot is None:
        nprot = d // 2
    rev = cp.full((n, d), 0xFFFFFFFF, cp.uint32)
    cnt = cp.zeros(n, cp.uint32)
    T = 256
    B = (n + T - 1) // T
    for k in range(d):
        _make_rev((B,), (T,), (g, cp.int64(n), cp.int32(d), cp.int32(k), rev, cnt))
    if nprot < d:
        _merge((B,), (T,), (g, cp.int64(n), cp.int32(d), rev, cnt, cp.int32(nprot)))
    return g


# ----------------------------------------------------------------------------------------------
# Distance-based pruning. For each node A: candidates c_0..c_{K-1} (sorted by distance).
def _chunk_dists(X, knn_d, s, e):
    """returns dAB [c,K] and DD [c,K,K] squared L2 within candidate lists."""
    V = X[knn_d[s:e].astype(cp.int64)]                    # c,K,dim
    A = X[s:e]                                            # c,dim
    nv = (V * V).sum(-1)                                  # c,K
    G = cp.matmul(V, V.transpose(0, 2, 1))                # c,K,K
    DD = nv[:, :, None] + nv[:, None, :] - 2 * G
    dAB = nv - 2 * cp.einsum('ckd,cd->ck', V, A) + (A * A).sum(-1)[:, None]
    return dAB, DD


def occlusion_counts(X, knn_d, alpha=1.0, chunk=None):
    """count[B] = #{D closer than B : alpha^2 * d(D,B)^2 < d(A,B)^2}  (distance analog of detours)."""
    n, K = knn_d.shape
    dim = X.shape[1]
    chunk = chunk or max(64, int(1.5e9 / (K * dim * 4)))
    out = cp.empty((n, K), cp.uint16)
    tri = cp.tri(K, K, -1, dtype=cp.bool_).T             # [D,B]: D<B
    a2 = alpha * alpha
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        dAB, DD = _chunk_dists(X, knn_d, s, e)
        m = (a2 * DD < dAB[:, None, :]) & tri[None]
        out[s:e] = m.sum(1).astype(cp.uint16)
    return out


def rng_greedy(X, knn_d, d, alpha=1.0, chunk=None, fill='rank'):
    """Vamana RobustPrune on the fixed candidate list; fill to fixed degree d."""
    n, K = knn_d.shape
    dim = X.shape[1]
    chunk = chunk or max(64, int(1.5e9 / (K * dim * 4)))
    out = cp.empty((n, d), cp.uint32)
    a2 = alpha * alpha
    ar = cp.arange(K)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        dAB, DD = _chunk_dists(X, knn_d, s, e)
        c = e - s
        kept = cp.zeros((c, K), cp.bool_)
        nk = cp.zeros(c, cp.int32)
        occ = cp.zeros((c, K), cp.int32)   # number of kept occluders (for fill ordering)
        for b in range(K):
            if b:
                o = (kept[:, :b] & (a2 * DD[:, :b, b] <= dAB[:, b:b + 1])).sum(1)
            else:
                o = cp.zeros(c, cp.int64)
            occ[:, b] = o
            take = (o == 0) & (nk < d)
            kept[:, b] = take
            nk += take
        # order: kept (by rank) then rest by (occluders, rank)
        key = cp.where(kept, 0, 1 + occ if fill == 'occ' else 1).astype(cp.int64) * K + ar[None]
        sel = cp.argsort(key, 1)[:, :d]
        out[s:e] = cp.take_along_axis(knn_d[s:e], sel, 1)
    return out


# ----------------------------------------------------------------------------------------------
class Index:
    def __init__(self, X, graph):
        self.X = X
        self.g = cp.ascontiguousarray(graph.astype(cp.uint32))
        self.h = lib().shim_index(ptr(X), X.shape[0], X.shape[1], ptr(self.g), self.g.shape[1])

    def __del__(self):
        try:
            lib().shim_free(self.h)
        except Exception:
            pass

    def search(self, Q, k=10, itopk=64, width=1, algo=0, max_iter=0, nrs=1, seeds=None, xor=0):
        nq = Q.shape[0]
        nb = cp.empty((nq, k), cp.uint32)
        dd = cp.empty((nq, k), cp.float32)
        L = lib()
        if seeds is not None:
            seeds = cp.ascontiguousarray(seeds.astype(cp.uint32))
            L.cuvs_cagra_experimental_set_seeds(ptr(seeds), seeds.shape[1])  # needs research build
        try:
            ms = L.shim_search(self.h, ptr(Q), nq, Q.shape[1], k, itopk, width, algo, max_iter, nrs, xor,
                               ptr(nb), ptr(dd))
        finally:
            if seeds is not None:
                L.cuvs_cagra_experimental_set_seeds(None, 0)
        return nb, dd, ms


def base_graph(name, X, d=64):
    import os
    f = f'{D}/{name}/strict_d{d}_nnd.npy'
    if not os.path.exists(f):
        knn = cp.asarray(np.load(f'{D}/{name}/knn128_nnd.npy'))
        np.save(f, merge_reverse(select_by_score(knn, detour_counts(knn, 1), d)).get())
    return cp.asarray(np.load(f))


def recall_per_query(nb, gt, k=10):
    nb = cp.asarray(nb)[:, :k].astype(cp.int64)
    g = cp.asarray(gt[:, :k]).astype(cp.int64)
    return (nb[:, :, None] == g[:, None, :]).any(2).sum(1) / k


def recall(nb, gt, k=10):
    nb = cp.asnumpy(nb)[:, :k]
    g = gt[:, :k]
    return np.mean([len(np.intersect1d(a, b)) for a, b in zip(nb, g)]) / k


def kmeans(X, C, iters=15, sample=200_000, seed=0):
    rs = cp.random.RandomState(seed)
    n = X.shape[0]
    S = X[rs.choice(n, min(sample, n), replace=False)]
    cent = S[rs.choice(S.shape[0], C, replace=False)].copy()
    for _ in range(iters):
        lab = assign(S, cent)
        sums = cp.zeros_like(cent)
        cp.add.at(sums, lab, S)
        cnt = cp.bincount(lab, minlength=C).astype(cp.float32)
        nz = cnt > 0
        cent[nz] = sums[nz] / cnt[nz, None]
    return cent


def assign(S, cent, B=None):
    B = B or max(1024, min(65536, int(4e8 // max(cent.shape[0], 1))))
    out = cp.empty(S.shape[0], cp.int64)
    cn = (cent * cent).sum(1)
    for s in range(0, S.shape[0], B):
        out[s:s + B] = cp.argmin(cn[None] - 2 * S[s:s + B] @ cent.T, 1)
    return out


def nearest_points(X, cent, B=65536):
    """for each centroid, id of nearest dataset point (medoid proxy)."""
    best = cp.full(cent.shape[0], cp.inf, cp.float32)
    bid = cp.zeros(cent.shape[0], cp.int64)
    xn = (X * X).sum(1)
    for s in range(0, X.shape[0], B):
        d = xn[s:s + B, None] - 2 * X[s:s + B] @ cent.T     # b,C
        j = cp.argmin(d, 0)
        v = d[j, cp.arange(cent.shape[0])]
        upd = v < best
        best = cp.where(upd, v, best)
        bid = cp.where(upd, j + s, bid)
    return bid.astype(cp.uint32)


class SeedSelector:
    """query -> top-m nearest entry points among a fixed pool (pool vectors = dataset rows)."""

    def __init__(self, X, pool_ids):
        self.ids = cp.asarray(pool_ids, cp.uint32)
        self.P = X[self.ids.astype(cp.int64)]
        self.pn = (self.P * self.P).sum(1)

    def __call__(self, Q, m):
        nq = Q.shape[0]
        nb = cp.empty((nq, m), cp.int64)
        dd = cp.empty((nq, m), cp.float32)
        lib().shim_bf_knn(ptr(self.P), self.P.shape[0], self.P.shape[1], ptr(Q), nq, m, ptr(nb), ptr(dd))
        return self.ids[nb]

    def slow(self, Q, m):
        d = self.pn[None] - 2 * Q @ self.P.T
        if m >= d.shape[1]:
            o = cp.argsort(d, 1)
        else:
            o = cp.argpartition(d, m, 1)[:, :m]
        return self.ids[o]
