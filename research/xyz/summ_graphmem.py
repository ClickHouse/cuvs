"""XYZ graph-memory variants vs CAGRA-Q: best QPS at target recalls + bytes/point."""
import sys, json, os, numpy as np
name = sys.argv[1]
def load(suffix):
    f = f'res_xyzc_{name}{suffix}.jsonl'
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []
def qps_at(rs, t):
    rs = sorted(rs, key=lambda r: r['recall']); best = float('nan')
    for a, b in zip(rs, rs[1:]):
        if a['recall'] <= t <= b['recall']:
            w = (t - a['recall']) / max(b['recall'] - a['recall'], 1e-9)
            best = np.exp(np.log(a['qps']) * (1 - w) + np.log(b['qps']) * w)
    return best
base = load('')
cq = [r for r in base if r['index'].startswith('CAGRA-Q')]
T = (0.90, 0.95, 0.98)
print(f'### {name} (QPS at recall@10, 10k queries; xN vs CAGRA-Q)')
print(f"  {'CAGRA-Q':30s} {cq[0]['bytes_per_pt'] if cq else 0:5d} B/pt | " + ' | '.join(f'{t}: {qps_at(cq, t)/1e3:5.0f}k' for t in T))
for label, suf in (('XYZ deg64', ''), ('XYZ deg32', '_d32'), ('XYZ deg64 3-byte graph', '_u24'), ('XYZ deg32 3-byte graph', '_d32_u24')):
    rows = [r for r in load(suf) if r['index'].startswith('XYZ')]
    if not rows: continue
    fam = {}
    for r in rows: fam.setdefault(r['index'], []).append(r)
    cells = []
    for t in T:
        c = [(qps_at(v, t), k) for k, v in fam.items()]; c = [x for x in c if x[0] == x[0]]
        if not c: cells.append(f'{t}:   —   '); continue
        v, k = max(c); ref = qps_at(cq, t)
        cells.append(f'{t}: {v/1e3:5.0f}k' + (f' x{v/ref:.2f}' if ref == ref else ''))
    print(f"  {label:30s} {rows[0]['bytes_per_pt']:5d} B/pt | " + ' | '.join(cells))
