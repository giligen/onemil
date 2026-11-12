"""Cell 1,740 — M1 retest bid on TRUE high-of-day levels (PREREG_1740). Reuses the 1,737 scorer's walkers UNCHANGED (run_bid / walk_long /
stats / pool_bid / sample_starts) by importlib; variants: strict (bid=level, fill low<level), thru10 (fill low<level*0.999, targets need
h>=target*1.001), under1 (bid=level*0.99, strict fill; secondary, no pass). Stop = engine consolidation_low at the last closed bar j-1,
raised to level*0.985 if closer (as 1739). Groups: TRAIN_H1, TRAIN_H2, TRAIN, VAL (or TEST when run on the TEST arms).
Capped book: 8 fills/day, 4 concurrent, first-come by fill minute. Placebo: 300 draws, same geometry from a random start after C+1.
Usage: python3 1740_score.py [arms_tag]   (arms_tag '' = TRAIN+VAL file 1740_arms.csv; '_test' = 1740_arms_test.csv)"""
import os, sys, sqlite3, importlib.util, math
import numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location('s1737', '/home/ec2-user/onemil/research/hod_entry/1737_score.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
W = S.walker; ROOT = S.ROOT; OUT = S.OUT
atag = sys.argv[1] if len(sys.argv) > 1 else ''
VARS = {'strict': (0.0, 1.0), 'thru10': (0.001, 1.0), 'under1': (0.0, 0.99)}   # name: (THRU, bid multiple of level)
W.wait_for_db_window()
arms = pd.read_csv(f'{OUT}/1740_arms{atag}.csv', dtype={'symbol': str, 'day': str})
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
rng = np.random.default_rng(1740)
rows, pools, nobars, noC = [], {}, 0, 0
for ix, a in enumerate(arms.itertuples()):
    d = W.fetch_day_bars(con, a.symbol, a.day)
    if d is None: nobars += 1; continue
    A = S.arrays(d); h, l, c, m = A['h'], A['l'], A['c'], A['m']; n = len(m)
    idx = [k for k in range(n) if m[k] >= a.entry_m]
    if not idx: nobars += 1; continue
    j = idx[0]; lvl = a.level; hs = a.half_spread
    floor_px = lvl * (1 - S.FLOOR)
    stop_p = min(min(l[max(j - 1 - 4, 0):max(j - 1, 0) + 1]), floor_px)
    C = next((k for k in range(j, n) if m[k] < S.LAST_M and c[k] >= lvl), None)
    if C is None or C + 1 >= n: noC += 1; continue
    starts = S.sample_starts(rng, A, C + 2)
    for v, (thru, mult) in VARS.items():
        S.THRU = thru
        bid = lvl * mult
        r = S.run_bid(A, C + 1, bid, None, lvl * 1.02, stop_p, None, hs)
        rows.append(dict(var=v, day=a.day, symbol=a.symbol, entry_m=a.entry_m, level=lvl, split=a.split, half=a.half, half_spread=hs,
                         matched_spread=a.matched_spread, b0_net_R=a.b0_net_R, c_m=m[C], bid=bid, stop=stop_p, filled=bool(r.get('filled')),
                         fill_m=m[r['fill_i']] if r.get('filled') else np.nan, exit_m=m[r['exit_i']] if r.get('filled') else np.nan,
                         **{k: r.get(k, np.nan) for k in ('R', 'r_pct', 'exit_px', 'why', 'net_R')}))
        pools[len(rows) - 1] = S.pool_bid(A, starts, c[C], bid, None, lvl * 1.02, stop_p, None, hs)
    S.THRU = 0.0
    if ix % 500 == 0: W.log(f'{ix}/{len(arms)} rows {len(rows)}')
o = pd.DataFrame(rows); o.to_csv(f'{OUT}/1740_orders{atag}.csv', index=False)
W.log('orders written; scoring')

def capped(fl):
    """First-come capped book: <= 8 fills/day, <= 4 concurrent (exit_m > new fill_m counts as open)."""
    keep = []
    for day, g in fl.sort_values(['day', 'fill_m', 'symbol']).groupby('day'):
        acc = []
        for x in g.itertuples():
            if len(acc) >= 8: break
            if sum(1 for e in acc if e.exit_m > x.fill_m) >= 4: continue
            acc.append(x); keep.append(x.Index)
    return fl.loc[keep]

def groups(df):
    """Scoring groups present in df."""
    g = {}
    if (df.split == 'TRAIN').any():
        g['TRAIN_H1'] = df[(df.split == 'TRAIN') & (df.half == 'H1')]; g['TRAIN_H2'] = df[(df.split == 'TRAIN') & (df.half == 'H2')]
        g['TRAIN'] = df[df.split == 'TRAIN']
    for s in ('VAL', 'TEST'):
        if (df.split == s).any(): g[s] = df[df.split == s]
    return g
prng = np.random.default_rng(17400)
out = []
for v in VARS:
    ov = o[o['var'] == v]
    for gname, sel in groups(ov).items():
        fl = sel[sel.filled]
        arms_g = groups(arms.assign(half=arms.half))[gname]
        nweeks = pd.to_datetime(arms_g.day).dt.strftime('%G-%V').nunique()
        st = S.stats(fl, nweeks) if len(fl) else dict(n=0)
        r = np.sort(fl.net_R.values); k1 = max(int(math.ceil(0.01 * len(r))), 1)
        st.update(var=v, group=gname, n_arms=len(arms_g), n_orders=len(sel), n_filled=len(fl), fill_share_orders=len(fl) / max(len(sel), 1),
                  fill_share_arms=len(fl) / max(len(arms_g), 1), nweeks=nweeks, ex_top1=r[:-k1].mean() if len(r) > k1 else np.nan,
                  b0_filled=fl.b0_net_R.mean(), b0_unfilled=sel[~sel.filled].b0_net_R.mean(), r_pct_median=fl.r_pct.median())
        ids = [i for i in sel.index if i in pools and len(pools[i])]
        if len(fl) > 5 and ids:
            L = np.array([len(pools[i]) for i in ids]); P = np.full((len(ids), L.max()), np.nan)
            for r_, i in enumerate(ids): P[r_, :L[r_]] = pools[i]
            pick = (prng.random((S.N_PLACEBO, len(ids))) * L).astype(int)
            with np.errstate(all='ignore'): draws = np.nanmean(P[np.arange(len(ids)), pick], axis=1)
            st.update(placebo_pctile=(draws < fl.net_R.mean()).mean() * 100, placebo_mean=np.nanmean(draws))
        if len(fl) > 5:
            cp = capped(fl); cst = S.stats(cp, nweeks)
            st.update({'cap_' + k: val for k, val in cst.items() if k in ('n', 'mean_R', 'day_t', 'fills_per_wk', 'ex_top5')})
        out.append(st)
        W.log(f"{v:6s} {gname:8s} orders={len(sel)} filled={len(fl)} mean={st.get('mean_R', float('nan')):.3f} day_t={st.get('day_t', float('nan')):.2f} "
              f"cap_n={st.get('cap_n')} cap_mean={st.get('cap_mean_R', float('nan')):.3f} pct={st.get('placebo_pctile', float('nan'))}")
pd.DataFrame(out).to_csv(f'{OUT}/1740_summary{atag}.csv', index=False)
W.log(f'DONE nobars={nobars} noC={noC}')
