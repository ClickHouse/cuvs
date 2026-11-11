"""XYZ = patched CAGRA (strict graph + fast refine + ADSampling in the CAGRA kernel) on rotated int8
vectors, vs CAGRA-Q (stock graph, VPQ) — real cuVS search, 10k queries, rerank of 32 included."""
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
L.shim_index_i8.restype = P_; L.shim_index_i8.argtypes = [P_, I_, I_, P_, I_]; L.shim_free_i8.argtypes = [P_]
L.shim_search_i8.restype = ctypes.c_double; L.shim_search_i8.argtypes = [P_, P_, I_, I_, F_, F_, F_, P_, P_]
L.shim_vpq_index.restype = P_; L.shim_vpq_index.argtypes = [P_, I_, I_, P_, I_, F_, F_]; L.shim_vpq_free.argtypes = [P_]
L.shim_vpq_search_only.restype = ctypes.c_double; L.shim_vpq_search_only.argtypes = [P_, P_, I_, I_, F_, F_, P_, P_]
L.shim_refine_only.restype = ctypes.c_double; L.shim_refine_only.argtypes = [P_, I_, P_, I_, I_, F_, F_, P_, P_, P_]
XDEG = int(os.environ.get('XYZ_DEG', '64'))
U24 = os.environ.get('XYZ_U24') == '1'
out = open(f'res_xyzc_{name}{"" if XDEG == 64 else f"_d{XDEG}"}{"_u24" if U24 else ""}.jsonl', 'a')
on = cp.empty((nq, 10), cp.int64); od = cp.empty((nq, 10), cp.float32)


def refine32(c32):
    c64 = c32.astype(cp.int64); c64[c32 == 0xFFFFFFFF] = -1
    L.shim_refine_only(cg.ptr(X), n, cg.ptr(Q), nq, D, 10, 32, cg.ptr(c64), cg.ptr(on), cg.ptr(od))
    return on


def timed(fn, k=3):
    fn(); best = 1e30
    for _ in range(k):
        cp.cuda.Device().synchronize(); t0 = time.perf_counter(); fn(); cp.cuda.Device().synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def emit(row):
    row.update(ds=name, qps=nq / row['ms'] * 1e3)
    out.write(json.dumps(row) + '\n'); out.flush()
    print(f"{row['index']:40s} {row['param']:>10s}: recall {row['recall']:.4f}  {row['ms']:7.1f} ms  QPS {row['qps']:9.0f}  {row['bytes_per_pt']} B/pt", flush=True)

# ---- CAGRA-Q baseline (stock graph) ----
os.environ['CUVS_CAGRA_ADS_EPS'] = '0'
if os.environ.get('SKIP_BASE') != '1':
  pass
Gs = cp.asarray(np.load(f'{d}/cagra_d64.npy'))
h = L.shim_vpq_index(cg.ptr(X), n, D, cg.ptr(Gs), 64, D // 2 if D % 2 == 0 else D, 0)
tn = cp.empty((nq, 32), cp.uint32); td = cp.empty((nq, 32), cp.float32)
for it in (32, 48, 64, 96, 128, 192, 256):
    def f(): L.shim_vpq_search_only(h, cg.ptr(Q), nq, D, 32, it, cg.ptr(tn), cg.ptr(td)); return refine32(tn)
    ms = timed(f); emit(dict(index='CAGRA-Q (stock graph, VPQ)', param=f'itopk={it}', recall=cg.recall(f(), gt), ms=ms, bytes_per_pt=D // 2 + 256))
L.shim_vpq_free(h); del Gs; cp.get_default_memory_pool().free_all_blocks()

# ---- XYZ: patched CAGRA on rotated int8 ----
G = cp.asarray(np.load(f'{d}/xyz_strict_d{XDEG}.npy'))
R, _ = cp.linalg.qr(cp.random.RandomState(0).standard_normal((D, D)).astype(cp.float32))
samp = X[cp.asarray(np.random.default_rng(0).choice(n, min(n, 200_000), replace=False))] @ R
s = float(127 / cp.percentile(cp.abs(samp), 99.9)); del samp
Dp = (D + 15) // 16 * 16          # CAGRA needs 16-byte rows for int8; zero padding keeps L2 unchanged
X8 = cp.zeros((n, Dp), cp.int8)
for s0 in range(0, n, 1_000_000):
    X8[s0:s0 + 1_000_000, :D] = cp.clip(cp.rint((X[s0:s0 + 1_000_000] @ R) * s), -127, 127).astype(cp.int8)
Q8 = cp.zeros((nq, Dp), cp.int8); Q8[:, :D] = cp.clip(cp.rint((Q @ R) * s), -127, 127).astype(cp.int8)
if U24:   # pack neighbor ids into 3 bytes (little endian); invalid 0xFFFFFFFF -> 0xFFFFFF
    assert n < (1 << 24) - 1
    Gp = cp.ascontiguousarray(G.astype(cp.uint32).view(cp.uint8).reshape(n, XDEG, 4)[:, :, :3]).reshape(n, XDEG * 3)
    Gp = cp.ascontiguousarray(Gp).view(cp.uint32)                  # [n, 3*deg/4]
    os.environ['CUVS_CAGRA_GRAPH_U24'] = '1'
    h = L.shim_index_i8(cg.ptr(X8), n, Dp, cg.ptr(Gp), XDEG * 3 // 4)
    # NB: the CAGRA index keeps a non-owning view of a device graph -> Gp must stay alive
else:
    h = L.shim_index_i8(cg.ptr(X8), n, Dp, cg.ptr(G), XDEG)
for team in ((0, 8, 16) if not U24 else (0, 8)):
    for eps in ((0.0, 0.5, 0.75, 1.0, 1.5) if not U24 else (0.0, 0.5, 1.0)):
        os.environ['CUVS_CAGRA_ADS_EPS'] = str(eps)
        for it in (32, 48, 64, 96, 128, 192, 256):
            def f(): L.shim_search_i8(h, cg.ptr(Q8), nq, Dp, 32, it, team, cg.ptr(tn), cg.ptr(td)); return refine32(tn)
            ms = timed(f)
            emit(dict(index=f'XYZ d{XDEG}{"-u24" if U24 else ""} int8 team{team} ' + (f'ads eps={eps}' if eps > 0 else 'no-ads'), param=f'itopk={it}',
                      recall=cg.recall(f(), gt), ms=ms, bytes_per_pt=Dp + (3 if U24 else 4) * XDEG, team=team, eps=eps))
os.environ['CUVS_CAGRA_ADS_EPS'] = '0'
os.environ.pop('CUVS_CAGRA_GRAPH_U24', None)
L.shim_free_i8(h)
