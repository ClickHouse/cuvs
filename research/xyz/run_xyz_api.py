"""XYZ through the cuVS API (xyz::build / xyz::search), end to end, vs CAGRA-Q. 10k queries."""
import sys, os, json, time, ctypes, numpy as np, cupy as cp, cg
name = sys.argv[1]; d = f'{cg.D}/{name}'
if name.endswith('10M'):
    base = np.load(f'{d}/base_u8.npy') if name.startswith('bigann') else np.load(f'{d}/base.npy')
    q = np.load(f'{d}/query_u8.npy') if name.startswith('bigann') else np.load(f'{d}/query.npy')
    gt = np.load(f'{d}/gt.npy')
else:
    base, q, gt = cg.load(name)
X = cp.asarray(base.astype(np.float32)); Q = cp.asarray(q.astype(np.float32)); del base
n, D = X.shape
rep = max(1, 10000 // Q.shape[0])
if rep > 1: Q = cp.tile(Q, (rep, 1)); gt = np.tile(gt, (rep, 1))
nq = Q.shape[0]
L = cg.lib(); P_, I_, F_ = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
L.shim_xyz_build.restype = P_; L.shim_xyz_build.argtypes = [P_, I_, I_, F_, F_, F_, ctypes.POINTER(ctypes.c_double), F_]
ROT = int(os.environ.get('XYZ_ROT', '1'))   # 1 always, 0 never, 2 auto
L.shim_xyz_free.argtypes = [P_]; L.shim_xyz_memory.restype = ctypes.c_size_t; L.shim_xyz_memory.argtypes = [P_]
L.shim_xyz_search.restype = ctypes.c_double
L.shim_xyz_search.argtypes = [P_, P_, I_, I_, F_, F_, F_, ctypes.c_float, F_, P_, P_]
out = open(f'res_xyzapi_{name}{"" if ROT == 1 else f"_rot{ROT}"}.jsonl', 'a')
nb = cp.empty((nq, 10), cp.int64); dd = cp.empty((nq, 10), cp.float32)
for pack in ((1, 0) if ROT == 1 else (1,)):
    bms = ctypes.c_double(0)
    h = L.shim_xyz_build(cg.ptr(X), n, D, 64, 128, pack, ctypes.byref(bms), ROT)
    mem = L.shim_xyz_memory(h)
    print(f'{name} xyz build pack={pack}: {bms.value/1e3:.1f} s, memory {mem/n:.0f} B/pt (+float {4*D} B/pt for refine)', flush=True)
    for team in (0, 8):
        for eps in (0.0, 0.5, 1.0):
            for it in (32, 48, 64, 96, 128, 192, 256):
                L.shim_xyz_search(h, cg.ptr(Q), nq, D, 10, it, team, eps, 32, cg.ptr(nb), cg.ptr(dd))
                ms = min(L.shim_xyz_search(h, cg.ptr(Q), nq, D, 10, it, team, eps, 32, cg.ptr(nb), cg.ptr(dd)) for _ in range(3))
                r = cg.recall(nb, gt)
                row = dict(ds=name, index=f'XYZ api pack={pack} rot={ROT} team{team} eps={eps}', param=f'itopk={it}', recall=r, ms=ms,
                           qps=nq / ms * 1e3, bytes_per_pt=mem / n, build_s=bms.value / 1e3)
                out.write(json.dumps(row) + '\n'); out.flush()
                print(f"{row['index']:36s} {row['param']:>10s}: recall {r:.4f}  {ms:7.1f} ms  QPS {row['qps']:9.0f}", flush=True)
    L.shim_xyz_free(h); cp.get_default_memory_pool().free_all_blocks()
