"""Summaries for the Frame B refuter: rule variants, obtainability, ETB, tails, day concentration."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil')
import research.hod_entry.cell_1619 as C
d = pd.read_csv('/home/ec2-user/onemil/research/hod_entry/review/refute_1619_probe.csv')
f = pd.read_csv('/home/ec2-user/onemil/research/hod_entry/cell_1619_fills.csv')
for sp in ('TRAIN', 'VAL'):
    v = d[d.split == sp]
    print(f'=== {sp} n={len(v)}')
    for tag in ('c19', 'c20'):
        for var in ('b', 's', 'sp'):
            y = v[f'{tag}_{var}_Rf']
            print(f'{tag}_{var}: mean Rf {y.mean():+.4f} t {C.day_clustered_t(y.values, v.day.values):+.2f} '
                  f'pct {v[f"{tag}_{var}_pct"].mean():+.4f} exT5 {C.ex_top5_mean(y.values):+.4f} '
                  f'cover {(v[f"{tag}_{var}_why"]=="cover").mean():.3f}')
    b = f[(f.cell == 1619) & (f.split == sp) & (f.filled == True)]
    print('builder csv 1619 mean', round(b.net_Rf.mean(), 4), 'n', len(b))
    # obtainability
    print('first-hit print: odd-lot share', round((v.first_size < 100).mean(), 3),
          'median size', v.first_size.median(), '| median bps above limit', round(v.first_px_bps.median(), 1))
    print('round-lot vol above limit: zero share', round((v.vol_above_rl == 0).mean(), 3),
          '<100', round((v.vol_above_rl < 100).mean(), 3), '<500', round((v.vol_above_rl < 500).mean(), 3),
          'median', v.vol_above_rl.median())
    ok = v.vol_above_rl >= 500
    print('  subset rl vol>=500 n', ok.sum(), 'mean Rf', round(v[ok].c19_b_Rf.mean(), 4),
          '| <500 mean', round(v[~ok].c19_b_Rf.mean(), 4))
    # stops: excess over stop on the triggering print
    st = v[v.c19_sp_why == 'stop']
    print('stop trig excess bps: mean', round(st.c19_sp_stopx_bps.mean(), 1), 'p50', round(st.c19_sp_stopx_bps.median(), 1),
          'p90', round(st.c19_sp_stopx_bps.quantile(.9), 1), 'p99', round(st.c19_sp_stopx_bps.quantile(.99), 1),
          '| builder tail bps', round(C.SLIP_STOP_BPS[sp], 1))
    # ETB
    for e, g in v.groupby(v.etb.fillna(False)):
        print(f'  etb={e}: n {len(g)} mean Rf {g.c19_b_Rf.mean():+.4f} pct {g.c19_b_pct.mean():+.4f}')
    # tails and days
    y = v.c19_b_Rf.values
    srt = np.sort(y)[::-1]
    print('ex-top-1%', round(srt[int(len(y)*.01):].mean(), 4), 'top-5% share of +sum',
          round(srt[:int(len(y)*.05)].sum() / y[y > 0].sum(), 3), 'max', round(y.max(), 3), 'min', round(y.min(), 3))
    day = v.groupby('day').c19_b_Rf.sum()
    print('days', len(day), 'green-day share', round((day > 0).mean(), 3), 'top-5 days sum', round(day.nlargest(5).sum(), 2),
          'total', round(day.sum(), 2), 'worst-5 days', round(day.nsmallest(5).sum(), 2))
    # payoff geometry: win size in R, breakeven cover share
    w = v[v.c19_b_why == 'cover'].c19_b_Rf.mean(); l = v[v.c19_b_why == 'stop'].c19_b_Rf.mean()
    print('mean cover Rf', round(w, 3), 'mean stop Rf', round(l, 3), 'breakeven cover share', round(-l / (w - l), 3),
          'cover share needed for +0.15', round((0.15 - l) / (w - l), 3))
    # gross (no stop tail, no borrow) on builder walk: approx add back tail on stops
    gross = v.c19_b_Rf + np.where(v.c19_b_why == 'stop', C.SLIP_STOP_BPS[sp] / 1e4 / (v.R_pct / 100 if False else 0.006), 0)
    print('approx gross (stop tail added back) mean Rf', round(gross.mean(), 4))
    # mirror: units
    print('R_pct median', v.R_pct.median(), 'base long net% mean', round((v.outcome_R * v.R_pct).mean(), 4),
          'short pct mean', round(v.c19_b_pct.mean(), 4), 'sum short+long mean', round((v.outcome_R * v.R_pct + v.c19_b_pct).mean(), 4))
    # spread quartile on c19_b
    q = pd.qcut(v.half_entry / v.level, 4, labels=False)
    print('by half-spread quartile mean Rf', v.groupby(q).c19_b_Rf.mean().round(4).tolist())
