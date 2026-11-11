import sys, time, numpy as np, cupy as cp, cg
name = sys.argv[1]; d = f'/data/scratch/ann-data/{name}'
base = np.load(f'{d}/base_u8.npy') if name.startswith('bigann') else np.load(f'{d}/base.npy')
n, D = base.shape
X = cp.asarray(base.astype(np.float32))
t = time.time()
import os
kf = f'{d}/knn128_nnd.npy'
if not os.path.exists(kf):
    out = np.empty((n, 128), np.uint32)
    cg.lib().shim_nnd_knn(cg.ptr(X), n, D, out.ctypes.data, 128)
    np.save(kf, out)
print(name, 'nnd knn', round(time.time() - t, 1), 's', flush=True)
knn = cp.asarray(np.load(kf))
for deg in (16, 24, 32, 48, 64):
    t = time.time()
    g = cp.empty((n, deg), cp.uint32)
    cg.lib().shim_optimize(cg.ptr(knn), n, 128, cg.ptr(g), deg, 0)
    np.save(f'{d}/cagra_d{deg}.npy', g.get())
    print(name, 'optimize deg', deg, round(time.time() - t, 1), 's', flush=True)
    del g; cp.get_default_memory_pool().free_all_blocks()
