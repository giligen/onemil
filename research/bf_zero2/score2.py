#!/usr/bin/env python3
"""bf_zero2 scoring — the pre-registered rule (DESIGN.md), on the executable BOOK, net of costs.

Per family-config: the live exit (rr_e1c: +2R on a bar close, stop first, flat 15:55) is run through the ONE book rule
(trading.hod_break.run_book: 4 concurrent, 12/day, last entry 14:00, causal freeing, symbol tie-break) after charging
costs (half spread in, half out on non-target exits — 40 bps for price >= 5, 60 bps below; the 10 bps stop slip is in the
fill model). TRAIN screen: net mean R > 0, >= 5 trades/week, weekly net R >= +10. Hypothesis splits (signal-minute
features only) adopted greedily on TRAIN if the excluded bucket is worse and the kept book still clears the bar; at most
three. VAL: weekly net R >= +7 and >= 60% weeks green. TEST read once. Cells counted; the VAL bar rises 1 SE per 10
TRAIN-passing cells. Population stats for e1/e2/e3/e4 are printed for context only. Failures included.
"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book
D = 'research/bf_zero2'
MIN_TPW, TRAIN_WK, VAL_WK, VAL_GREEN, MIN_N, MAX_SPLITS = 5.0, 10.0, 7.0, 0.60, 200, 3
MAX_PER_DAY, MAX_CONC, LAST_ENTRY, PRICE_FLOOR = 12, 4, 840, 5.0
NUM = ['entry_m', 'exit_m_e1c', 'price', 'r_pct', 'dist_open_pct', 'rv_adv', 'bar_vol_x', 'above_vwap', 'gap_pct', 'prev_range_pct', 'adv20',
       'dist_20d_high_pct', 'spy_5m_ret', 'spy_range3', 'rr_e1', 'rr_e2', 'rr_e3', 'rr_e4', 'rr_e1c', 'rv_clock', 'rv_profile', 'is_wrapper', 'coh_by_t',
       'pm_vol', 'pm_high_pct', 'range_so_far_pct', 'bars_per_min', 'n_bars_at_entry']
c = pd.read_csv(f'{D}/candidates_full.csv', usecols=lambda k: k in set(['day', 'symbol', 'fam', 'cfg', 'why_e1c'] + NUM),
                dtype={'symbol': 'category', 'day': 'category', 'fam': 'category', 'cfg': 'category', 'why_e1c': 'category'},
                keep_default_na=False, na_values=[''], low_memory=True)
c['day'] = c.day.astype(str); c['symbol'] = c.symbol.astype(str)                     # the book needs plain strings; the rest stay categorical
for k in NUM:
    if k in c.columns: c[k] = pd.to_numeric(c[k], errors='coerce').astype('float32')
c = c[(c.price >= PRICE_FLOOR) & (c.entry_m <= LAST_ENTRY + 1) & (c.r_pct >= 1.0)].reset_index(drop=True)   # the live spec's min_r_pct
# CAUSAL FLOOR (the universe is "day range (high-low)/low >= 5%" — a HINDSIGHT filter). A pole/drive/VWAP entry (F1-F4)
# implies it by construction; for F5-F10 only an entry >= 5% above the open guarantees it (open >= low). Without this the
# first score2 run "found" quiet names that WILL have a 5% range after entry — the same look-ahead class as the cache
# population (9/16 01:30 UTC; that run is void, tables kept as score_tables_VOID_no_floor.md).
NEED = ~c.fam.isin(['F1', 'F2', 'F3', 'F4'])
print(f'causal floor: {int((NEED & (c.dist_open_pct < 5)).sum()):,} rows dropped of {int(NEED.sum()):,} (F5-F10)', flush=True)
c = c[~(NEED & ~(c.dist_open_pct >= 5))].reset_index(drop=True)
c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)
WEEKS = {s: c[c.split == s].wk.nunique() for s in ('TRAIN', 'VAL', 'TEST')}
c['key'] = c.fam.astype(str) + ' ' + c.cfg.astype(str)
spread = np.where(c.price >= 5, 0.40, 0.60)                              # % of price, the §8 study medians
half = 0.5 * spread / c.r_pct.clip(lower=0.05)                           # half a spread in R units
c['net_e1c'] = c.rr_e1c - half - np.where(c.why_e1c == 'target', 0.0, half)
print(f'rows {len(c):,} | keys {c.key.nunique()} | weeks {WEEKS}', flush=True)


def book_stats(d, split):
    """the executable book on the live exit, weekly net R"""
    nw = WEEKS[split]; wks = sorted(c[c.split == split].wk.unique())
    if not len(d): return dict(n=0, tpw=0.0, meanR=np.nan, wkR=np.nan, green=0.0, worst=np.nan, WR=np.nan)
    rows = [(r.day, int(r.entry_m), int(r.exit_m_e1c), r.symbol, r.net_e1c, r.wk) for r in d.itertuples()]
    taken = run_book(rows, MAX_PER_DAY, MAX_CONC)
    t = pd.DataFrame(taken, columns=['day', 'entry_m', 'exit_m', 'symbol', 'net', 'wk'])
    w = t.groupby('wk').net.sum().reindex(wks).fillna(0)
    return dict(n=len(t), tpw=round(len(t) / nw, 1), meanR=round(t.net.mean(), 3), wkR=round(float(w.mean()), 1), green=round(float((w > 0).mean()), 2),
                worst=round(float(w.min()), 1), WR=round(float((t.net > 0).mean() * 100), 1), sd=round(float(w.std()), 1))


def splits(d):
    q = lambda col, cuts: {f'{col} {a}-{b}': (d[col] >= a) & (d[col] < b) for a, b in zip(cuts[:-1], cuts[1:])} if col in d.columns else {}
    S = {}
    S.update(q('rv_profile', [0, 0.5, 1, 2, 5, 1e9])); S.update(q('rv_clock', [0, 1, 2, 4, 1e9]))
    S.update(q('dist_open_pct', [-1e9, 0, 2, 5, 10, 20, 1e9])); S.update(q('gap_pct', [-1e9, -2, 0, 2, 5, 10, 1e9]))
    S.update(q('price', [5, 10, 20, 50, 1e9])); S.update(q('r_pct', [0, 1, 2, 4, 1e9])); S.update(q('entry_m', [570, 600, 630, 720, 841]))
    S.update(q('pm_vol', [0, 1e4, 1e5, 1e6, 1e12])); S.update(q('pm_high_pct', [-1e9, 0, 5, 15, 1e9])); S.update(q('range_so_far_pct', [0, 3, 6, 12, 1e9]))
    S.update(q('adv20', [0, 5e5, 2e6, 1e13])); S.update(q('bars_per_min', [0, 0.5, 0.9, 1.01])); S.update(q('prev_range_pct', [0, 5, 10, 1e9]))
    if 'above_vwap' in d: S['above_vwap'] = d.above_vwap == 1; S['below_vwap'] = d.above_vwap == 0
    if 'is_wrapper' in d: S['wrapper'] = d.is_wrapper == 1; S['common'] = d.is_wrapper == 0
    if 'coh_by_t' in d: S['coh>=1'] = d.coh_by_t >= 1; S['alone'] = d.coh_by_t == 0
    if 'spy_5m_ret' in d: S['spy5m_up'] = d.spy_5m_ret >= 0; S['spy5m_down'] = d.spy_5m_ret < 0
    return S


rows = []; cells = 0
for key, dk in c.groupby('key'):
    for s in ('TRAIN', 'VAL', 'TEST'):
        st = book_stats(dk[dk.split == s], s); rows.append(dict(key=key, split=s, **st))
        for rr in ('rr_e1', 'rr_e2', 'rr_e3', 'rr_e4', 'rr_e1c'):
            rows[-1][f'pop_{rr}'] = round(float(dk[dk.split == s][rr].mean()), 3) if (dk.split == s).any() else np.nan
T1 = pd.DataFrame(rows); T1.to_csv(f'{D}/step1_book.csv', index=False)
tr = T1[T1.split == 'TRAIN']; cells = len(tr)
keep = tr[(tr.meanR > 0) & (tr.tpw >= MIN_TPW)].copy(); keep['hits_10R'] = keep.wkR >= TRAIN_WK   # owner 9/16: 10R is the target line, not a filter — every positive config is carried through
pd.set_option('display.width', 250)
print(f'\n## STEP 1 — every family-config on the 4/12 book, live exit, net of costs, TRAIN: {len(keep)} of {len(tr)} positive with tpw >= {MIN_TPW}; {int(keep.hits_10R.sum())} reach the +{TRAIN_WK}R/week target line')
print(tr.sort_values('wkR', ascending=False).head(40).to_string(index=False))
res = []
for _, k in keep.sort_values('wkR', ascending=False).iterrows():
    key = k.key; d = c[c.key == key]; mask = pd.Series(True, index=d.index); chosen = []; dtr = d[d.split == 'TRAIN']
    base = book_stats(dtr, 'TRAIN')
    for step in range(MAX_SPLITS):
        best = None
        for name, m in splits(d).items():
            cells += 1
            kept = dtr[(mask & m)[dtr.index]]; excl = dtr[(mask & ~m)[dtr.index]]
            if len(kept) < MIN_N or len(excl) < 50: continue
            st = book_stats(kept, 'TRAIN'); ex = book_stats(excl, 'TRAIN')
            if st['tpw'] >= MIN_TPW and st['wkR'] > 0 and st['meanR'] > base['meanR'] + 0.05 and ex['meanR'] < base['meanR'] and (best is None or st['wkR'] > best[1]['wkR']):
                best = (name, st, m)
        if best is None: break
        chosen.append(best[0]); mask = mask & best[2]; base = best[1]
    trs = book_stats(dtr[mask[dtr.index]], 'TRAIN'); va = book_stats(d[(d.split == 'VAL') & mask], 'VAL'); te = book_stats(d[(d.split == 'TEST') & mask], 'TEST')
    val_bar = VAL_WK + (len(keep) // 10) * (va.get('sd', 0) or 0) / np.sqrt(max(WEEKS['VAL'], 1))
    val_ok = va['n'] > 0 and va['wkR'] >= val_bar and va['green'] >= VAL_GREEN and va['tpw'] >= MIN_TPW
    verdict = 'CANDIDATE' if val_ok and te['n'] > 0 and te['wkR'] > 0 else ('val_pass_test_fail' if val_ok else 'val_fail')
    res.append(dict(key=key, splits=' & '.join(chosen) or '(none)', val_bar=round(val_bar, 1), verdict=verdict, hits_10R_train=bool(trs.get('wkR', 0) >= TRAIN_WK),
                    **{f'TRAIN_{a}': trs.get(a) for a in ('tpw', 'meanR', 'wkR', 'green', 'worst')}, **{f'VAL_{a}': va.get(a) for a in ('tpw', 'meanR', 'wkR', 'green', 'worst')},
                    **{f'TEST_{a}': te.get(a) for a in ('tpw', 'meanR', 'wkR', 'green', 'worst', 'WR')}))
R = pd.DataFrame(res)
print(f'\n## STEPS 2-4 — {len(R)} TRAIN-kept configs, {cells} cells looked at; verdicts: {R.verdict.value_counts().to_dict() if len(R) else {}}')
if len(R): print(R.to_string(index=False))
R.to_csv(f'{D}/results.csv', index=False)
miss = tr.sort_values('wkR', ascending=False).head(15)
with open(f'{D}/score_tables.md', 'w') as fh:
    fh.write(f'# bf_zero2 — score tables (score2.py; {cells} cells)\n\n## STEP 1 TRAIN book, every family-config (top 40 by weekly net R)\n\n' + tr.sort_values('wkR', ascending=False).head(40).to_markdown(index=False) + '\n\n')
    fh.write('## STEPS 2-4\n\n' + (R.to_markdown(index=False) if len(R) else '_nothing passed the TRAIN bar_') + '\n\n## CLOSEST MISSES (TRAIN, by weekly net R)\n\n' + miss.to_markdown(index=False) + '\n')
print('\nCANDIDATES:', R[R.verdict == 'CANDIDATE'][['key', 'splits']].values.tolist() if len(R) else [])
