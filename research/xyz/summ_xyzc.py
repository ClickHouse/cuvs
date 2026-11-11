import sys, json, numpy as np
name = sys.argv[1]
rows = [json.loads(l) for l in open(f'res_xyzc_{name}.jsonl')]
def qps_at(rs, t):
    rs = sorted(rs, key=lambda r: r['recall']); best = float('nan')
    for a, b in zip(rs, rs[1:]):
        if a['recall'] <= t <= b['recall']:
            w = (t - a['recall']) / max(b['recall'] - a['recall'], 1e-9)
            best = np.exp(np.log(a['qps']) * (1 - w) + np.log(b['qps']) * w)
    return best
fam = {}
for r in rows: fam.setdefault(r['index'], []).append(r)
T = (0.90, 0.95, 0.98)
cq = fam.get('CAGRA-Q (stock graph, VPQ)', [])
print(f'### {name}: QPS at recall@10 (10k queries)')
print('  CAGRA-Q          : ' + ' | '.join(f'{t}: {qps_at(cq, t)/1e3:6.0f}k' for t in T))
nod = {k: v for k, v in fam.items() if 'no-ads' in k}
ads = {k: v for k, v in fam.items() if 'ads eps' in k}
for label, group in (('XYZ no ADS (best)', nod), ('XYZ with ADS (best)', ads)):
    cells = []
    for t in T:
        c = [(qps_at(v, t), k) for k, v in group.items()]; c = [x for x in c if x[0] == x[0]]
        if not c: cells.append(f'{t}:     —   '); continue
        v, k = max(c); cells.append(f'{t}: {v/1e3:6.0f}k x{v/qps_at(cq, t):.2f} [{k.replace("XYZ int8 ", "")}]')
    print(f'  {label:17s}: ' + ' | '.join(cells))
