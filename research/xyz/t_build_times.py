import sys, time, ctypes, numpy as np, cupy as cp, cg
name = sys.argv[1]
base, q, gt = cg.load(name); X = cp.asarray(base); n, D = X.shape
L = cg.lib(); P_, I_, F_ = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
L.shim_vpq_index.restype = P_; L.shim_vpq_index.argtypes = [P_, I_, I_, P_, I_, F_, F_]; L.shim_vpq_free.argtypes = [P_]
L.shim_ivfpq_build.restype = P_; L.shim_ivfpq_build.argtypes = [P_, I_, I_, F_, F_, F_, ctypes.POINTER(ctypes.c_double)]
L.shim_ivfpq_free.argtypes = [P_]
sync = cp.cuda.Device().synchronize
out = np.empty((n, 128), np.uint32)
sync(); t = time.time(); L.shim_nnd_knn(cg.ptr(X), n, D, out.ctypes.data, 128); nnd = time.time() - t
knn = cp.asarray(out)
sync(); t = time.time(); g = cp.empty((n, 64), cp.uint32); L.shim_optimize(cg.ptr(knn), n, 128, cg.ptr(g), 64, 0); sync(); opt = time.time() - t
sync(); t = time.time(); gs = cg.merge_reverse(cg.select_by_score(knn, cg.detour_counts(knn, 1), 64)); sync(); strict = time.time() - t
del knn, gs; cp.get_default_memory_pool().free_all_blocks()
sync(); t = time.time(); h = L.shim_vpq_index(cg.ptr(X), n, D, cg.ptr(g), 64, D // 2, 0); sync(); vpq = time.time() - t
L.shim_vpq_free(h)
sync(); t = time.time()
R, _ = cp.linalg.qr(cp.random.RandomState(0).standard_normal((D, D)).astype(cp.float32))
s = float(127 / cp.percentile(cp.abs(X[:200000] @ R), 99.9)); X8 = cp.empty((n, D), cp.int8)
for s0 in range(0, n, 200_000): X8[s0:s0 + 200_000] = cp.clip(cp.rint((X[s0:s0 + 200_000] @ R) * s), -127, 127).astype(cp.int8)
sync(); quant = time.time() - t; del X8, g; cp.get_default_memory_pool().free_all_blocks()
bms = ctypes.c_double(0); h = L.shim_ivfpq_build(cg.ptr(X), n, D, 1024, D // 2, 8, ctypes.byref(bms)); L.shim_ivfpq_free(h)
print(f'{name} nnd={nnd:.1f} stock_optimize={opt:.2f} strict={strict:.2f} vpq={vpq:.2f} int8={quant:.2f} ivfpq={bms.value/1e3:.1f}', flush=True)
