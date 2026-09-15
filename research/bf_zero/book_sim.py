#!/usr/bin/env python3
"""bf_zero — the executable HOD-break book, simulated exactly as live would trade it.

Rows = F5 {"K": 5, "X": 0.04} candidates (one per symbol-day: the FIRST HOD break after >= 5 bars all
within 4% of the running high), causal floor entry >= 5% above the open. For each row the bars are
reloaded and the trade is re-walked under the LIVE fill model:
  entry: stop-limit at the HOD level, limit = level x (1 + CAP); fills at the limit if the entry bar's
         high reaches it, else no trade (no chase)
  stop:  consolidation low; fills at min(stop, bar open) x 0.999 (gap-through modeled)
  target: entry + 2R, resting limit, fills only when a bar CLOSES at/above it (no wick fills)
  flat 15:55
Then the book: first-come, per-day cap N_DAY, at most N_CONC open at once (exit minute known),
rv_profile in [RV_LO, RV_HI), r_pct >= RMIN. Outputs per split + week-by-week + TRAIN-only band check.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, f'{ROOT}/research/bf_zero')
import build_candidates as B
D = 'research/bf_zero'
CAP = float(os.environ.get('HB_CAP', '0.006')); N_DAY = int(os.environ.get('HB_NDAY', '8')); N_CONC = int(os.environ.get('HB_NCONC', '4'))
RV_LO, RV_HI, RMIN = 1.0, 5.0, 1.0
EOD = B.EOD_M

c = pd.read_csv(f'{D}/candidates_full.csv', usecols=['day', 'symbol', 'fam', 'cfg', 'entry_m', 'entry', 'stop', 'dist_open_pct', 'r_pct', 'rv_profile', 'rv_clock', 'price', 'adv20', 'is_wrapper'],
                dtype={'symbol': str, 'fam': 'category', 'cfg': 'category'}, keep_default_na=False, na_values=[''])
c = c[(c.fam == 'F5') & (c.cfg == '{"K": 5, "X": 0.04}')].drop(columns=['fam', 'cfg'])
for k in ('entry_m', 'entry', 'stop', 'dist_open_pct', 'r_pct', 'rv_profile', 'rv_clock', 'price', 'adv20', 'is_wrapper'): c[k] = pd.to_numeric(c[k], errors='coerce')
c = c[c.dist_open_pct >= 5].copy()
print('rows', len(c), flush=True)

rows = []
for n, (day, sub) in enumerate(c.groupby('day')):
    bars = B.load_bars(day, sub.symbol.tolist())
    for r in sub.itertuples():
        gg = bars.get(r.symbol)
        if gg is None: continue
        rth = gg[(gg.m >= B.OPEN_M) & (gg.m < 960)].reset_index(drop=True)
        o, h, l, cl = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c')); m = rth.m.values.astype(int)
        idx = np.flatnonzero(m == int(r.entry_m))
        if not len(idx): continue
        i = int(idx[0]); level = r.entry / (1 + B.SLIP); entry = level * (1 + CAP); stop = r.stop
        if h[i] < entry or stop >= entry: rows.append(dict(day=day, symbol=r.symbol, entry_m=int(m[i]), filled=0)); continue
        Rd = entry - stop; tgt = entry + 2 * Rd
        oo, hh, ll, cc, mm = o[i + 1:], h[i + 1:], l[i + 1:], cl[i + 1:], m[i + 1:]
        if len(oo) == 0: continue
        eod = B.first_true(mm >= EOD); eod = eod if eod is not None else len(oo) - 1
        s_idx = B.first_true(ll <= stop); t_idx = B.first_true(cc >= tgt)
        cand = [(k, w) for k, w in ((s_idx, 'stop'), (t_idx, 'target'), (eod, 'eod')) if k is not None]
        k, w = min(cand, key=lambda x: (x[0], 0 if x[1] == 'stop' else 1))
        px = min(stop, oo[k]) * 0.999 if w == 'stop' else (tgt if w == 'target' else oo[k])
        rows.append(dict(day=day, symbol=r.symbol, entry_m=int(m[i]), filled=1, exit_m=int(mm[k]), why=w, rr=(px - entry) / Rd,
                         r_pct=(entry - stop) / entry * 100, rv_profile=r.rv_profile, rv_clock=r.rv_clock, price=entry, adv20=r.adv20, is_wrapper=r.is_wrapper, dist_open_pct=r.dist_open_pct))
    if n % 50 == 0: print(f'{n} days {len(rows)} rows', flush=True)
T = pd.DataFrame(rows); T.to_csv(f'{D}/hodbreak_trades_cap{int(CAP * 1e4)}.csv', index=False)
T['split'] = np.where(T.day < '2026-01-01', 'TRAIN', np.where(T.day < '2026-06-01', 'VAL', 'TEST')); T['wk'] = pd.to_datetime(T.day).dt.to_period('W-FRI').astype(str)
NW = T.groupby('split').wk.nunique().to_dict()
F = T[(T.filled == 1) & (T.rv_profile >= RV_LO) & (T.rv_profile < RV_HI) & (T.r_pct >= RMIN)].copy()
print(f'\nfill rate at cap {CAP:.3%}: {T.filled.mean():.2%} | filtered signals {len(F)}', flush=True)

# TRAIN-only band check (was the rv band chosen on TRAIN alone the same?)
tr = T[(T.filled == 1) & (T.split == 'TRAIN')]
print('\nTRAIN-only: meanR by rv_profile bucket:', tr.groupby(pd.cut(tr.rv_profile, [0, 0.5, 1, 2, 5, 1e9]), observed=True).rr.agg(['mean', 'count']).round(3).to_dict('index'))
print('TRAIN-only: meanR by r_pct bucket:', tr.groupby(pd.cut(tr.r_pct, [0, 1, 2, 4, 100]), observed=True).rr.agg(['mean', 'count']).round(3).to_dict('index'))


def run_book(F):
    """first-come, per-day cap, concurrency cap — the ONE rule in trading.hod_break.run_book (causal freeing, symbol tie-break)."""
    from trading.hod_break import run_book as _rb
    return F.loc[[t[4] for t in _rb([(r.day, r.entry_m, r.exit_m, r.symbol, r.Index) for r in F.itertuples()], N_DAY, N_CONC)]]


bk = run_book(F)
print(f'\n## HOD-break book: cap {CAP:.2%} no-chase, close-fill target, first-come {N_DAY}/day, {N_CONC} concurrent, rv [{RV_LO},{RV_HI}), r_pct >= {RMIN}')
for s in ('TRAIN', 'VAL', 'TEST'):
    d = bk[bk.split == s]; w = d.groupby('wk').rr.sum().reindex(sorted(T[T.split == s].wk.unique())).fillna(0); nw = NW[s]
    print(f"  {s:5s} trades {len(d):5d} ({len(d) / nw:4.1f}/wk) meanR {d.rr.mean():+.3f} WR {(d.rr > 0).mean() * 100:4.1f} PF {d.rr[d.rr > 0].sum() / -d.rr[d.rr < 0].sum():.2f} | weekly R {w.mean():+.1f} sd {w.std():.1f} green {(w > 0).sum()}/{nw} worst {w.min():+.1f} | exits {d.why.value_counts().to_dict()}")
print('\nTEST week-by-week (R, n):'); d = bk[bk.split == 'TEST']; print(d.groupby('wk').rr.agg(['sum', 'count']).round(1).to_string())
print('\nby price band (all splits):', bk.groupby(pd.cut(bk.price, [1, 2, 5, 10, 20, 50, 1e6]), observed=True).rr.agg(['mean', 'count']).round(3).to_dict('index'))
print('wrapper vs common:', bk.groupby('is_wrapper').rr.agg(['mean', 'count']).round(3).to_dict('index'))
bk.to_csv(f'{D}/hodbreak_book_cap{int(CAP * 1e4)}.csv', index=False)
print('DONE', flush=True)
