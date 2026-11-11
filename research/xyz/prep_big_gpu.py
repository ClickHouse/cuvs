"""ground truth (top-100) + NN-descent kNN graph (K=128) for the 10M subsets."""
import sys, time, ctypes, numpy as np, cupy as cp, cg
name = sys.argv[1]
d = f'/data/scratch/ann-data/{name}'
if name.startswith('bigann'):
    base = np.load(f'{d}/base_u8.npy'); q = np.load(f'{d}/query_u8.npy')
else:
    base = np.load(f'{d}/base.npy'); q = np.load(f'{d}/query.npy')
n, D = base.shape
t = time.time()
Q = cp.asarray(q.astype(np.float32))
# exact GT in chunks of the base (float32 chunk on GPU), merging top-100
K = 100; best_d = cp.full((Q.shape[0], K), cp.inf, cp.float32); best_i = cp.zeros((Q.shape[0], K), cp.int64)
CH = 2_000_000
for s0 in range(0, n, CH):
    Xc = cp.asarray(base[s0:s0 + CH].astype(np.float32))
    nb = cp.empty((Q.shape[0], K), cp.int64); dd = cp.empty((Q.shape[0], K), cp.float32)
    cg.lib().shim_bf_knn(cg.ptr(Xc), Xc.shape[0], D, cg.ptr(Q), Q.shape[0], K, cg.ptr(nb), cg.ptr(dd))
    ad = cp.concatenate([best_d, dd], 1); ai = cp.concatenate([best_i, nb + s0], 1)
    o = cp.argsort(ad, 1)[:, :K]; best_d = cp.take_along_axis(ad, o, 1); best_i = cp.take_along_axis(ai, o, 1)
    del Xc; cp.get_default_memory_pool().free_all_blocks()
np.save(f'{d}/gt.npy', best_i.get().astype(np.int32))
print(name, 'gt done', round(time.time() - t, 1), 's', flush=True)
