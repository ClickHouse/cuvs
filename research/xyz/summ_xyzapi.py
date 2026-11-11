import sys, json, numpy as np
name = sys.argv[1]
def qps_at(rs, t):
    rs = sorted(rs, key=lambda r: r['recall']); best = float('nan')
    for a, b in zip(rs, rs[1:]):
        if a['recall'] <= t <= b['recall']:
            w = (t - a['recall']) / max(b['recall'] - a['recall'], 1e-9)
            best = np.exp(np.log(a['qps']) * (1 - w) + np.log(b['qps']) * w)
    return best
api = [json.loads(l) for l in open(f'res_xyzapi_{name}.jsonl')]
cq = [json.loads(l) for l in open(f'res_xyzc_{name}.jsonl')]
cq = [r for r in cq if r['index'].startswith('CAGRA-Q')]
T = (0.90, 0.95, 0.98)
print(f'### {name}: end-to-end via cuVS API (query rotation included), QPS at recall@10')
print(f"  {'CAGRA-Q':24s} {cq[0]['bytes_per_pt']:5.0f} B/pt | " + ' | '.join(f'{t}: {qps_at(cq, t)/1e3:5.0f}k' for t in T))
for pack in (1, 0):
    rows = [r for r in api if f'pack={pack}' in r['index']]
    if not rows: continue
    fam = {}
    for r in rows: fam.setdefault(r['index'], []).append(r)
    cells = []
    for t in T:
        c = [(qps_at(v, t), k) for k, v in fam.items()]; c = [x for x in c if x[0] == x[0]]
        if not c: cells.append(f'{t}:   —   '); continue
        v, k = max(c); ref = qps_at(cq, t)
        cells.append(f'{t}: {v/1e3:5.0f}k' + (f' x{v/ref:.2f}' if ref == ref else ''))
    print(f"  {'XYZ ' + ('3-byte graph' if pack else '4-byte graph'):24s} {rows[0]['bytes_per_pt']:5.0f} B/pt | " + ' | '.join(cells) + f"   build {rows[0]['build_s']:.1f} s")
