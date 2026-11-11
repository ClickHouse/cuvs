"""graphs for XYZ (strict, deg 32/64) and the CAGRA baseline (stock optimize, deg 64), from the NN-descent knn."""
import sys, os, time, json, numpy as np, cupy as cp, cg
name = sys.argv[1]
d = f'{cg.D}/{name}'
knn_np = np.load(f'{d}/knn128_nnd.npy'); n = knn_np.shape[0]
times = {}
knn = cp.asarray(knn_np); del knn_np
t = time.time(); cnt = cg.detour_counts(knn, 1); times['detour_count_s'] = time.time() - t
for deg in (32, 64):
    f = f'{d}/xyz_strict_d{deg}.npy'
    if os.path.exists(f): continue
    t = time.time()
    sel = cg.select_by_score(knn, cnt, deg, chunk=250_000)
    g = cg.merge_reverse(sel); del sel
    np.save(f, g.get()); del g; cp.get_default_memory_pool().free_all_blocks()
    times[f'strict_d{deg}_s'] = time.time() - t + times['detour_count_s']
del cnt; cp.get_default_memory_pool().free_all_blocks()
f = f'{d}/cagra_d64.npy'
if not os.path.exists(f):
    t = time.time(); g = cp.empty((n, 64), cp.uint32)
    cg.lib().shim_optimize(cg.ptr(knn), n, 128, cg.ptr(g), 64, 0); np.save(f, g.get()); times['stock_d64_s'] = time.time() - t
json.dump(times, open(f'{d}/xyz_graph_times.json', 'w'))
print(name, times, flush=True)
