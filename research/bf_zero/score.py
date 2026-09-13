#!/usr/bin/env python3
"""Bull-flag from zero — scoring per DESIGN.md (pre-registered selection rule).

Step 1 (TRAIN only): every family-config × exit → trades/week, mean R, mean R ex-tail (R >= 3
removed), WR, profit factor, weeks green, worst week, max drawdown (R). Keep configs with
trades/week >= 5, mean R > 0, worst week >= -10R.
Step 2 (TRAIN only): for each kept config, hypothesis splits; a split is adopted if it raises
mean R by >= 0.05 while keeping >= 5 trades/week; at most three, greedy by lift.
Step 3: VALIDATE — mean R > 0 and >= half weeks green, else dropped (no re-tuning).
Step 4: TEST read once, week by week, tail removed.
Everything printed and written to score_tables.md, failures included.
"""
import json, os
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); D = os.environ.get('BFZ_DIR', 'research/bf_zero')
MIN_TPW = float(os.environ.get('BFZ_MIN_TPW', '5')); WORST_WK = -10.0; LIFT = 0.05; MIN_N = int(os.environ.get('BFZ_MIN_N', '200'))  # a split needs >= MIN_N TRAIN trades
c = pd.read_csv(f'{D}/candidates_full.csv', low_memory=False, dtype={'symbol': str}, keep_default_na=False)
for k in c.columns:
    if k not in ('day', 'symbol', 'fam', 'cfg', 'why_e1', 'why_e2', 'why_e3', 'why_e4', 'anchor'): c[k] = pd.to_numeric(c[k], errors='coerce')
c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)
WEEKS = {s: c[c.split == s].wk.nunique() for s in ('TRAIN', 'VAL', 'TEST')}
c['key'] = c.fam + ' ' + c.cfg
EXITS = ('rr_e1', 'rr_e2', 'rr_e3', 'rr_e4')
out = []


def stats(d, rr, split):
    r = d[rr].dropna(); nw = WEEKS[split]
    if len(r) == 0: return dict(n=0, tpw=0.0)
    w = d.groupby('wk')[rr].sum().reindex(sorted(c[c.split == split].wk.unique())).fillna(0)
    cum = w.cumsum(); mdd = float((cum - cum.cummax()).min())
    gp = r[r > 0].sum(); gl = -r[r < 0].sum()
    return dict(n=len(r), tpw=round(len(r) / nw, 1), meanR=round(r.mean(), 3), meanR_ex=round(r[r < 3].mean(), 3), WR=round((r > 0).mean() * 100, 1),
                PF=round(gp / gl, 2) if gl > 0 else np.inf, wk_green=f"{int((w > 0).sum())}/{nw}", wk_green_frac=float((w > 0).mean()),
                worst_wk=round(float(w.min()), 1), mdd_R=round(mdd, 1), sumR=round(float(r.sum()), 1))


# ---- Step 1: family-config x exit on TRAIN (all splits printed) ----
rows = []
for key, dk in c.groupby('key'):
    for rr in EXITS:
        for s in ('TRAIN', 'VAL', 'TEST'):
            rows.append(dict(key=key, exit=rr, split=s, **stats(dk[dk.split == s], rr, s)))
T1 = pd.DataFrame(rows)
tr = T1[T1.split == 'TRAIN']
keep = tr[(tr.tpw >= MIN_TPW) & (tr.meanR > 0) & (tr.worst_wk >= WORST_WK)]
print(f"candidates {len(c):,} | weeks TRAIN/VAL/TEST {WEEKS} | configs {c.key.nunique()} x exits {len(EXITS)}")
print(f"\n## STEP 1 — TRAIN screen (tpw >= {MIN_TPW}, meanR > 0, worst week >= {WORST_WK}R): {len(keep)} of {len(tr)} config-exits pass")
pd.set_option('display.width', 250)
print(tr.sort_values('meanR', ascending=False).drop(columns=['wk_green_frac']).head(40).to_string(index=False))
out.append(('STEP 1 — every family-config × exit, TRAIN (top 40 by mean R; full table in step1_train.csv)', tr.drop(columns=['wk_green_frac']).sort_values('meanR', ascending=False)))
T1.to_csv(f'{D}/step1_all_splits.csv', index=False); tr.to_csv(f'{D}/step1_train.csv', index=False)

# ---- hypothesis splits ----
def splits(d):
    q = lambda col, cuts: {f'{col} {a}-{b}': (d[col] >= a) & (d[col] < b) for a, b in zip(cuts[:-1], cuts[1:])}
    S = {}
    S.update(q('rv_profile', [0, 0.5, 1, 2, 5, 1e9])); S.update(q('rv_clock', [0, 1, 2, 4, 1e9]))
    S.update(q('dist_open_pct', [-1e9, 0, 5, 15, 30, 1e9])); S.update(q('retrace', [0, 0.33, 0.5]))
    S['above_vwap'] = d.above_vwap == 1; S['below_vwap'] = d.above_vwap == 0
    S.update(q('price', [1, 2, 5, 10, 20, 1e9])); S.update(q('r_pct', [0, 1, 3, 6, 1e9]))
    S.update(q('entry_m', [570, 630, 720, 930])); S['prev_range>=10'] = d.prev_range_pct >= 10; S['prev_range<10'] = d.prev_range_pct < 10
    S.update(q('gap_pct', [-1e9, 0, 5, 15, 1e9])); S.update(q('adv20', [0, 5e5, 2e6, 1e12]))
    S['spy5m_up'] = d.spy_5m_ret >= 0; S['spy5m_down'] = d.spy_5m_ret < 0
    S['wrapper'] = d.is_wrapper == 1; S['common'] = d.is_wrapper == 0; S['coh_by_t>=1'] = d.coh_by_t >= 1; S['alone'] = d.coh_by_t == 0
    S.update(q('bar_vol_x', [0, 1, 3, 1e9]))
    return S


# ---- Step 2/3/4 per kept config ----
results = []
for _, k in keep.sort_values('meanR', ascending=False).iterrows():
    key, rr = k.key, k['exit']; d = c[c.key == key].copy(); mask = pd.Series(True, index=d.index); chosen = []
    dtr = d[d.split == 'TRAIN']
    base = stats(dtr, rr, 'TRAIN')
    for step in range(3):
        best = None
        for name, m in splits(d).items():
            mm = mask & m; st = stats(dtr[mm[dtr.index]], rr, 'TRAIN')
            if st['n'] >= MIN_N and st['tpw'] >= MIN_TPW and st['meanR'] - base['meanR'] >= LIFT and (best is None or st['meanR'] > best[1]['meanR']):
                best = (name, st, m)
        if best is None: break
        chosen.append(best[0]); mask = mask & best[2]; base = best[1]
    va = stats(d[(d.split == 'VAL') & mask], rr, 'VAL'); te = stats(d[(d.split == 'TEST') & mask], rr, 'TEST'); trs = stats(dtr[mask[dtr.index]], rr, 'TRAIN')
    val_ok = va.get('n', 0) > 0 and va['meanR'] > 0 and va['wk_green_frac'] >= 0.5 and va['tpw'] >= MIN_TPW
    test_ok = val_ok and te.get('n', 0) > 0 and te['meanR'] > 0 and te['wk_green_frac'] >= 0.5 and te['tpw'] >= MIN_TPW
    verdict = 'CANDIDATE' if test_ok else ('val_pass_test_fail' if val_ok else 'val_fail')
    results.append(dict(key=key, exit=rr, splits=' & '.join(chosen) or '(none)', verdict=verdict,
                        **{f'TRAIN_{a}': trs.get(a) for a in ('tpw', 'meanR', 'meanR_ex', 'WR', 'wk_green', 'worst_wk')},
                        **{f'VAL_{a}': va.get(a) for a in ('tpw', 'meanR', 'meanR_ex', 'WR', 'wk_green', 'worst_wk')},
                        **{f'TEST_{a}': te.get(a) for a in ('tpw', 'meanR', 'meanR_ex', 'WR', 'wk_green', 'worst_wk', 'mdd_R', 'sumR')}))
    if test_ok:
        wk = d[(d.split == 'TEST') & mask].groupby('wk')[rr].agg(['sum', 'count']).round(1)
        results[-1]['TEST_weeks'] = wk.to_dict('index')
R = pd.DataFrame(results)
print(f"\n## STEPS 2-4 — {len(R)} TRAIN-kept config-exits; verdicts: {R.verdict.value_counts().to_dict() if len(R) else {}}")
if len(R): print(R.drop(columns=['TEST_weeks'], errors='ignore').to_string(index=False))
R.to_csv(f'{D}/results.csv', index=False)
with open(f'{D}/score_tables.md', 'w') as fh:
    fh.write('# bf_zero — score tables (auto-generated by score.py)\n\n')
    for title, t in out: fh.write(f'## {title}\n\n' + t.head(60).to_markdown(index=False) + '\n\n')
    fh.write('## STEPS 2-4 — kept configs with adopted splits, VAL and TEST\n\n' + (R.drop(columns=['TEST_weeks'], errors='ignore').to_markdown(index=False) if len(R) else '_none kept on TRAIN_') + '\n')
    for _, r in R[R.verdict == 'CANDIDATE'].iterrows():
        fh.write(f"\n### TEST week-by-week — {r.key} {r['exit']} [{r.splits}]\n\n" + pd.DataFrame(r.TEST_weeks).T.to_markdown() + '\n')
print('\nCANDIDATES:', R[R.verdict == 'CANDIDATE'][['key', 'exit', 'splits']].values.tolist() if len(R) else [])
