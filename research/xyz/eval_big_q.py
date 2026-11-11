"""10M: CAGRA-Q (VPQ codes + graph, refine) vs IVF-PQ (refine). Real GPU timings, end to end."""
import sys, json, time, ctypes, numpy as np, cupy as cp, cg
name = sys.argv[1]; d = f'/data/scratch/ann-data/{name}'
base = np.load(f'{d}/base_u8.npy') if name.startswith('bigann') else np.load(f'{d}/base.npy')
q = np.load(f'{d}/query_u8.npy') if name.startswith('bigann') else np.load(f'{d}/query.npy')
gt = np.load(f'{d}/gt.npy')
X = cp.asarray(base.astype(np.float32)); Q = cp.asarray(q.astype(np.float32)); n, D = X.shape; nq = Q.shape[0]
L = cg.lib(); P_, I_, F_ = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
L.shim_vpq_index.restype = P_; L.shim_vpq_index.argtypes = [P_, I_, I_, P_, I_, F_, F_]; L.shim_vpq_free.argtypes = [P_]
L.shim_vpq_search_only.restype = ctypes.c_double; L.shim_vpq_search_only.argtypes = [P_, P_, I_, I_, F_, F_, P_, P_]
L.shim_refine_only.restype = ctypes.c_double; L.shim_refine_only.argtypes = [P_, I_, P_, I_, I_, F_, F_, P_, P_, P_]
L.shim_ivfpq_build.restype = P_; L.shim_ivfpq_build.argtypes = [P_, I_, I_, F_, F_, F_, ctypes.POINTER(ctypes.c_double)]
L.shim_ivfpq_free.argtypes = [P_]
L.shim_ivfpq_search.restype = ctypes.c_double; L.shim_ivfpq_search.argtypes = [P_, P_, I_, I_, F_, F_, P_, P_, F_]
FP16 = 0 if name.startswith('bigann') else 1   # fp16 LUT/distances overflow on BigANN magnitudes
out = open(f'res_bigq_{name}.jsonl', 'a')
on = cp.empty((nq, 10), cp.int64); od = cp.empty((nq, 10), cp.float32)


def refine(cand64):
    return L.shim_refine_only(cg.ptr(X), n, cg.ptr(Q), nq, D, 10, cand64.shape[1], cg.ptr(cand64), cg.ptr(on), cg.ptr(od))


def emit(row):
    out.write(json.dumps(row) + '\n'); out.flush()
    print(f"{row['algo']:24s} {row['param']:>12s}: recall {row['recall']:.4f}  {row['ms']:7.1f} ms  QPS {nq / row['ms'] * 1e3:8.0f}", flush=True)


# ---------------- CAGRA-Q ----------------
for deg in ((32, 64) if 'ivfonly' not in sys.argv else ()):
    G = cp.asarray(np.load(f'{d}/cagra_d{deg}.npy'))
    cp.cuda.Device().synchronize(); t = time.time()
    h = L.shim_vpq_index(cg.ptr(X), n, D, cg.ptr(G), deg, D // 2, 0)
    vpq_s = time.time() - t
    print(f'CAGRA-Q deg {deg}: VPQ train+encode {vpq_s:.1f} s', flush=True)
    cand = 32
    tn = cp.empty((nq, cand), cp.uint32); td = cp.empty((nq, cand), cp.float32)
    for it in (32, 48, 64, 96, 128, 192, 256):
        L.shim_vpq_search_only(h, cg.ptr(Q), nq, D, cand, it, cg.ptr(tn), cg.ptr(td))
        best = 1e30
        for _ in range(3):
            ms1 = L.shim_vpq_search_only(h, cg.ptr(Q), nq, D, cand, it, cg.ptr(tn), cg.ptr(td))
            c64 = tn.astype(cp.int64); c64[tn == 0xFFFFFFFF] = -1
            ms2 = refine(c64)
            best = min(best, ms1 + ms2)
        emit(dict(ds=name, algo=f'CAGRA-Q deg{deg}', param=f'itopk={it}', recall=cg.recall(on, gt), ms=best,
                  vpq_build_s=vpq_s, bytes_per_pt=D // 2 + 4 * deg))
    L.shim_vpq_free(h); del G; cp.get_default_memory_pool().free_all_blocks()

# ---------------- IVF-PQ ----------------
bms = ctypes.c_double(0)
h = L.shim_ivfpq_build(cg.ptr(X), n, D, 16384, D // 2, 8, ctypes.byref(bms))
print(f'IVF-PQ build {bms.value / 1e3:.1f} s', flush=True)
for kc in (50, 100):
    nb = cp.empty((nq, kc), cp.int64); dd = cp.empty((nq, kc), cp.float32)
    for npb in (16, 32, 64, 128, 256, 512):
        L.shim_ivfpq_search(h, cg.ptr(Q), nq, D, kc, npb, cg.ptr(nb), cg.ptr(dd), FP16)
        best = 1e30
        for _ in range(3):
            ms1 = L.shim_ivfpq_search(h, cg.ptr(Q), nq, D, kc, npb, cg.ptr(nb), cg.ptr(dd), FP16)
            ms2 = refine(nb)
            best = min(best, ms1 + ms2)
        emit(dict(ds=name, algo=f'IVF-PQ refine{kc}' + ('' if FP16 else ' fp32'), param=f'probes={npb}', recall=cg.recall(on, gt), ms=best,
                  build_s=bms.value / 1e3, bytes_per_pt=D // 2 + 8))
L.shim_ivfpq_free(h)
