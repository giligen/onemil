"""Cell 1,739 — score the M1 retest bid ONCE on the sealed TEST arms (1739_arms.csv). Reuses the 1,737 scorer's walkers
(run_bid / walk_long / stats / pools) UNCHANGED (importlib; H1737_THROUGH set from argv: 0 = strict one-tick, 0.001 = 10 bp).
Adds: capped book (8 fills/day, 4 concurrent, first-come by fill minute), 1739_orders[_thru10bp].csv, summary printout.
Usage: python3 1739_score.py [through]"""
import os, sys, sqlite3, importlib.util, math
THRU = sys.argv[1] if len(sys.argv) > 1 else '0'
os.environ['H1737_THROUGH'] = THRU
import numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location('s1737', '/home/ec2-user/onemil/research/hod_entry/1737_score.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
W = S.walker; ROOT = S.ROOT; OUT = S.OUT
tag = '' if float(THRU) == 0 else '_thru10bp'
arms = pd.read_csv(f'{OUT}/1739_arms.csv', dtype={'symbol': str, 'day': str})
arms = arms[arms.level.notna()].reset_index(drop=True)
sig = pd.read_parquet(f'{ROOT}/research/hod_exit_lab/signals.parquet')[['day', 'symbol', 'half_spread']].drop_duplicates(['day', 'symbol'])
arms = arms.merge(sig, on=['day', 'symbol'], how='left')
arms['matched_spread'] = arms.half_spread.notna()
arms['dec'] = pd.qcut(arms.level, 10, labels=False, duplicates='drop')
med = arms[arms.matched_spread].groupby('dec').half_spread.median()
arms.loc[~arms.matched_spread, 'half_spread'] = arms.loc[~arms.matched_spread, 'dec'].map(med)
b0 = pd.read_csv(f'{ROOT}/research/hod_exit_lab/b0_trades.csv').drop_duplicates(['day', 'symbol'])[['day', 'symbol', 'net_R']].rename(columns={'net_R': 'b0_net_R'})
arms = arms.merge(b0, on=['day', 'symbol'], how='left')
con = sqlite3.connect('file:' + ROOT + '/research/bf_zero/bars_sip.db?mode=ro', uri=True)
rng = np.random.default_rng(1739)
rows, pools, nobars, noC = [], {}, 0, 0
for ix, a in enumerate(arms.itertuples()):
    d = W.fetch_day_bars(con, a.symbol, a.day)
    if d is None: nobars += 1; continue
    A = S.arrays(d); h, l, c, m = A['h'], A['l'], A['c'], A['m']; n = len(m)
    idx = [k for k in range(n) if m[k] >= a.entry_m]
    if not idx: nobars += 1; continue
    j = idx[0]; lvl = a.level; hs = a.half_spread
    floor_px = lvl * (1 - S.FLOOR)
    stop_p = min(min(l[max(j - 1 - 4, 0):max(j - 1, 0) + 1]), floor_px)   # engine consolidation_low at the last closed bar j-1
    C = next((k for k in range(j, n) if m[k] < S.LAST_M and c[k] >= lvl), None)
    if C is None or C + 1 >= n: noC += 1; continue
    r = S.run_bid(A, C + 1, lvl, None, lvl * 1.02, stop_p, None, hs)
    rec = dict(day=a.day, symbol=a.symbol, entry_m=a.entry_m, level=lvl, split=a.split, half_spread=hs, matched_spread=a.matched_spread,
               b0_net_R=a.b0_net_R, c_m=m[C], stop=stop_p, filled=bool(r.get('filled')),
               fill_m=m[r['fill_i']] if r.get('filled') else np.nan, exit_m=m[r['exit_i']] if r.get('filled') else np.nan,
               **{k: r.get(k, np.nan) for k in ('R', 'r_pct', 'exit_px', 'why', 'net_R')})
    rows.append(rec)
    pools[len(rows) - 1] = S.pool_bid(A, S.sample_starts(rng, A, C + 2), c[C], lvl, None, lvl * 1.02, stop_p, None, hs)
    if ix % 500 == 0: W.log(f'{ix}/{len(arms)} rows {len(rows)}')
o = pd.DataFrame(rows); o.to_csv(f'{OUT}/1739_orders{tag}.csv', index=False)
fl = o[o.filled]
nweeks = pd.to_datetime(arms.day).dt.strftime('%G-%V').nunique()
st = S.stats(fl, nweeks)
ids = [i for i in o.index if i in pools and len(pools[i])]
L = np.array([len(pools[i]) for i in ids]); P = np.full((len(ids), L.max()), np.nan)
for r_, i in enumerate(ids): P[r_, :L[r_]] = pools[i]
prng = np.random.default_rng(17390)
pick = (prng.random((S.N_PLACEBO, len(ids))) * L).astype(int)
with np.errstate(all='ignore'): draws = np.nanmean(P[np.arange(len(ids)), pick], axis=1)
st.update(placebo_pctile=(draws < fl.net_R.mean()).mean() * 100, placebo_mean=np.nanmean(draws))
# winner-capped = mean with R capped at 3 ; ex-top-1%
r = np.sort(fl.net_R.values); k1 = max(int(math.ceil(0.01 * len(r))), 1); st['ex_top1'] = r[:-k1].mean()
# capped book: 8 fills/day, 4 concurrent, first-come by fill minute
cap = []
for day, g in fl.sort_values(['day', 'fill_m', 'symbol']).groupby('day'):
    acc = []
    for x in g.itertuples():
        if len(acc) >= 8: break
        if sum(1 for e in acc if e.exit_m > x.fill_m) >= 4: continue
        acc.append(x); cap.append(x.Index)
cp = fl.loc[cap]; cst = S.stats(cp, nweeks)
head = dict(n_arms=len(arms), n_orders=len(o), n_filled=len(fl), fill_share_orders=len(fl) / len(o), fill_share_arms=len(fl) / len(arms), nobars=nobars, noC=noC, nweeks=nweeks,
            b0_filled=fl.b0_net_R.mean(), b0_unfilled=o[~o.filled].b0_net_R.mean())
out = [f'THROUGH={THRU}', 'HEAD ' + str(head), 'UNCAPPED ' + str(st), 'CAPPED(8/day,4conc) ' + str(cst)]
pd.DataFrame([dict(book='uncapped', **head, **st), dict(book='capped', **head, **cst)]).to_csv(f'{OUT}/1739_summary{tag}.csv', index=False)
print('\n'.join(out), flush=True)
