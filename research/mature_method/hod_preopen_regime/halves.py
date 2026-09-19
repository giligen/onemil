"""POST-HOC: the programme's own selection rule (same sign in BOTH halves of TRAIN) applied to
every cell that got anywhere near a bar. Counted in the multiplicity total."""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
for q in ('hod_break', 'hod_filter_stack', 'hod_preopen_regime'):
    sys.path.insert(0, f'{ROOT}/research/mature_method/{q}')
import score as S, score2 as S2, score3 as S3                    # noqa: E402

D = f'{ROOT}/research/mature_method/hod_preopen_regime'
p = S2.load_pop(); S.build_impute(p)
pre = {nm: S2.sig_set(p, **kw) for nm, kw in S2.BASES.items() if nm in ('B0', 'B2')}
pm, pdy = S3.idx_tape()
dfx = pd.read_csv(f'{D}/day_fields.csv', dtype={'day': str}).set_index('day')
for nm in pre:
    x = pre[nm].merge(pm, on=['day', 'break_m'], how='left'); x.index = pre[nm].index
    for c in dfx.columns:
        if c not in x.columns:
            x[c] = x.day.map(dfx[c])
    for c in pdy.columns:
        x[c] = x.day.map(pdy[c])
    pre[nm] = x

GATES = {
    'A3 spy_r5>0': lambda x: x.spy_r5_pct > 0,
    'A14 gap>0 AND r5>0': lambda x: (x.spy_gap_pct > 0) & (x.spy_r5_pct > 0),
    'A5 qqq_gap>0': lambda x: x.qqq_gap_pct > 0,
    'A13 spy_r5>0 AND qqq_r5>0': lambda x: (x.spy_r5_pct > 0) & (x.qqq_r5_pct > 0),
    'A1 spy_gap>0': lambda x: x.spy_gap_pct > 0,
    'F5M-7 gap-and-GO': lambda x: np.sign(x.spy_gap_pct) == np.sign(x.spy_r5_pct),
    'T SPY 1000 sgn (late arm)': lambda x: x.spy_ret_1000 > 0,
}
print('== HALF-CONSISTENCY (the pre-committed selection rule: same sign in BOTH TRAIN halves) ==')
print('| gate | base | H1-2025 gross sep (clustered t) | H2-2025 | VAL | both TRAIN halves same sign? |')
for nm, fn in GATES.items():
    for bn in ('B2', 'B0'):
        b = pre[bn]
        if nm.startswith('T SPY'):
            b = b[b.entry_m >= 601]
        out, sg = [], []
        for lab, lo, hi in (('H1-25', '2025-01-01', '2025-07-01'),
                            ('H2-25', '2025-07-01', '2026-01-01'),
                            ('VAL', '2026-01-01', '2026-06-01')):
            d = b[(b.day >= lo) & (b.day < hi)]
            m = fn(d).fillna(False)
            k, r = d[m], d[~m]
            if len(k) < 20 or len(r) < 20:
                out.append(f'{lab} n/a'); continue
            g = float(k.rr.mean() - r.rr.mean())
            vc = S3._cluster_var(k.net.values, k.day.values) + S3._cluster_var(r.net.values, r.day.values)
            t = (k.net.mean() - r.net.mean()) / np.sqrt(vc) if vc > 0 else np.nan
            out.append(f'{lab} {g:+.3f} (t {t:+.2f})')
            if lab != 'VAL':
                sg.append(g)
        ok = len(sg) == 2 and (sg[0] > 0) == (sg[1] > 0)
        print(f'| {nm:<26s} | {bn} | ' + ' | '.join(out) + f' | **{"YES" if ok else "NO"}** |')
