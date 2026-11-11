import sys, json, os, numpy as np
name = sys.argv[1]
def qps_at(rs, t):
    rs = sorted(rs, key=lambda r: r['recall']); best = float('nan')
    for a, b in zip(rs, rs[1:]):
        if a['recall'] <= t <= b['recall']:
            w = (t - a['recall']) / max(b['recall'] - a['recall'], 1e-9)
            best = np.exp(np.log(a['qps']) * (1 - w) + np.log(b['qps']) * w)
    return best
def best_fam(rows, t):
    fam = {}
    for r in rows: fam.setdefault(r['index'], []).append(r)
    c = [(qps_at(v, t), k) for k, v in fam.items()]; c = [x for x in c if x[0] == x[0]]
    return max(c) if c else (float('nan'), '')
rot1 = [json.loads(l) for l in open(f'res_xyzapi_{name}.jsonl') if 'pack=1' in l]
rot0 = [json.loads(l) for l in open(f'res_xyzapi_{name}_rot0.jsonl')]
cq = [json.loads(l) for l in open(f'res_xyzc_{name}.jsonl')]; cq = [r for r in cq if r['index'].startswith('CAGRA-Q')]
mx = lambda rows, it: max([r['recall'] for r in rows if r['param'] == f'itopk={it}'] or [float('nan')])
print(f'### {name}: rotation on vs off (XYZ via API, 3-byte graph, QPS at recall@10; xN vs CAGRA-Q)')
for t in (0.90, 0.95, 0.98):
    c = qps_at(cq, t); a, ka = best_fam(rot1, t); b, kb = best_fam(rot0, t)
    fmt = lambda v: '   —   ' if v != v else f'{v/1e3:5.0f}k' + (f' x{v/c:.2f}' if c == c else '')
    print(f'  @{t}: CAGRA-Q {c/1e3 if c == c else float("nan"):5.0f}k | rotated {fmt(a)} | not rotated {fmt(b)}')
print(f'  best recall at itopk=64: rotated {mx(rot1, 64):.4f}, not rotated {mx(rot0, 64):.4f};  itopk=32: {mx(rot1, 32):.4f} vs {mx(rot0, 32):.4f}')
