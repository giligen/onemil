#!/usr/bin/env python3
"""Refuter checks for PREREG_1617 frame C (cells 1,621/1,622). Reuses the builder's own machinery
(cell_1621.eligibility_and_entry_w, cell_1487 loaders) and perturbs one lens at a time:
  1. W=15 reproduction of RESULT_1487 (calibration + primary book) -- the parameterisation check
  2. entry-bar existence look-ahead: no_entry_bar rows entered at the first bar >= F+W+1
  3. window sparsity: eligible rows whose "no dip" rests on < W window bars
  4. kept cache-only share vs 19.5 %
  5. tails: ex-top-1 %, drop best 2 days, per-month sign
  6. run-up cost: (entry - level)/level bps, R''/R_base, entry-bar stop share
  7. cost decomposition: gross at mid (ask proxy removed) and zero exit cost
  8. the VAL all_eligible -2.13 R outlier
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from research.hod_entry import cell_1445 as c1445   # noqa: E402
from research.hod_entry import cell_1487 as c1487   # noqa: E402
from research.hod_entry import cell_1621 as c1621   # noqa: E402
from research.hod_entry import sip_rebuild as sr    # noqa: E402


def stats(x, days):
    """mean, day-clustered t, ex-top-5 %, ex-top-1 %, n."""
    x = pd.Series(x).reset_index(drop=True)
    days = pd.Series(days).reset_index(drop=True)
    if len(x) < 3:
        return dict(n=len(x))
    s = np.sort(x.values)
    k1 = max(1, int(round(0.01 * len(s))))
    return dict(n=len(x), mean=round(float(x.mean()), 4), t=round(float(c1445.day_clustered_t(x, days)), 2),
                ex5=round(float(c1445.ex_top5_mean(x)), 4), ex1=round(float(s[:-k1].mean()), 4))


def primary(sc):
    """Entered rows above the 0.5 % R'' floor."""
    e = sc[sc.entered.astype(bool)]
    return e[~e.below_floor.astype(bool)]


def main():
    base = c1487.load_base()
    print('base cols:', [c for c in base.columns if c in ('fill', 'stop', 'R', 'half_entry', 'exit_m', 'why')], flush=True)
    bars = c1487.load_bars(list(zip(base.symbol, base.day)))
    cmap = c1487.load_cacheonly_flags()
    bidx = base.set_index(['day', 'symbol'])

    # 1. W=15 reproduction
    sc15 = c1621.eligibility_and_entry_w(base, bars, 15, 1487)
    for h in ('TRAIN-H2', 'VAL'):
        d = sc15[sc15.holdout == h]
        p = primary(d)
        print(f'[1] W=15 {h}: eligible_share {d.eligible.mean():.4f} calib {d[d.eligible].base_outcome_R.mean():.4f} '
              f'primary {stats(p.net_R2, p.day)}', flush=True)

    for cid, W in c1621.WINDOWS.items():
        sc = c1621.eligibility_and_entry_w(base, bars, W, cid)
        # 2. entry-bar existence: enter the no_entry_bar rows at the first bar >= F+W+1
        extra = []
        for r in sc[sc.why == 'no_entry_bar'].itertuples():
            b = bars[(r.symbol, r.day)]
            br = base.loc[(base.day == r.day) & (base.symbol == r.symbol) & (base.fill_min == r.fill_min)].iloc[0]
            m0 = int(np.floor(r.fill_min)) + W + 1
            later = b[b.m >= m0].sort_values('m')
            if not len(later):
                continue
            # the skipped minutes must not have dipped (they have no bars, so no print dipped)
            entry = float(later.o.iloc[0]) + float(br.half_entry)
            stop = r.level - c1621.DIP_TICK
            R2 = entry - stop
            if R2 <= 0:
                continue
            ex_m, ex_p, why = sr.walk_path(entry, stop, entry + 2 * R2, later)
            raw = (ex_p - entry) / R2
            cost = 0 if why == 'target' else ex_p * (c1621.SLIP_STOP_BPS if why in ('stop', 'stop_bar') else c1621.EOD_BPS)[r.split] / 1e4 / R2
            extra.append(dict(day=r.day, holdout=r.holdout, net_R2=raw - cost, below_floor=R2 / entry < c1621.RFLOOR_PCT,
                              delay=int(later.m.iloc[0]) - m0, base_outcome_R=r.base_outcome_R))
        extra = pd.DataFrame(extra)
        for h in ('TRAIN-H2', 'VAL'):
            d = sc[sc.holdout == h]
            p = primary(d)
            print(f'\n=== cell {cid} W={W} {h} builder primary {stats(p.net_R2, p.day)}', flush=True)
            ne = d[d.why == 'no_entry_bar']
            print(f'[2] no_entry_bar n={len(ne)} base_outcome mean {ne.base_outcome_R.mean():.3f}', flush=True)
            if len(extra):
                ex = extra[(extra.holdout == h) & (~extra.below_floor.astype(bool))]
                comb = pd.concat([p[['day', 'net_R2']], ex[['day', 'net_R2']]])
                print(f'    added {len(ex)} (median delay {ex.delay.median() if len(ex) else np.nan} min, mean net {ex.net_R2.mean():.3f}); '
                      f'combined primary {stats(comb.net_R2, comb.day)}', flush=True)
            # 3. window sparsity
            nb = []
            for r in p.itertuples():
                b = bars[(r.symbol, r.day)]
                nb.append(int(((b.m > r.fill_min) & (b.m <= r.fill_min + W)).sum()))
            p = p.assign(nwin=nb)
            full = p[p.nwin == W]
            print(f'[3] window bars: share full {len(full)/len(p):.3f}; full-window book {stats(full.net_R2, full.day)}; '
                  f'sparse book {stats(p[p.nwin < W].net_R2, p[p.nwin < W].day)}', flush=True)
            # 4. cache-only share
            flags = pd.Series([cmap.get((r.day, r.symbol)) for r in p.itertuples()], dtype=float)
            print(f'[4] kept cache-only share {flags.mean():.4f} (coverage {flags.notna().mean():.3f}) vs 0.195+-0.05', flush=True)
            # 5. tails
            dm = p.groupby('day').net_R2.sum().sort_values(ascending=False)
            drop2 = p[~p.day.isin(dm.index[:2])]
            mon = p.assign(mo=p.day.str[:7]).groupby('mo').net_R2.mean()
            print(f'[5] drop best 2 days {stats(drop2.net_R2, drop2.day)}; months positive {int((mon > 0).sum())}/{len(mon)}; '
                  f'max net {p.net_R2.max():.3f} min {p.net_R2.min():.3f}; why {p.why.value_counts().to_dict()}', flush=True)
            # 6. run-up cost
            runup_bps = (p.entry - p.level) / p.level * 1e4
            bb = bidx.loc[list(zip(p.day, p.symbol))]
            ratio = p.R2.values / (bb['fill'].values - bb['stop'].values) if 'fill' in bb and 'stop' in bb else np.array([np.nan])
            stop_entry_bar = ((p.why.isin(['stop', 'stop_bar'])) & (p.exit_m2 == p.entry_bar_m)).mean()
            print(f'[6] run-up entry-level median {runup_bps.median():.1f} bps (p90 {runup_bps.quantile(.9):.1f}); '
                  f'median R2/R_base {np.nanmedian(ratio):.2f}; stopped in entry bar {stop_entry_bar:.3f}; '
                  f'target share {(p.why == "target").mean():.3f} (breakeven at 2R target w/o cost = 0.333)', flush=True)
            # 7. cost decomposition
            he = bb['half_entry'].values
            gross_mid = p.raw_R2.values + he / p.R2.values
            print(f'[7] raw (ask entry, no exit cost) {stats(p.raw_R2, p.day)}; gross at mid {stats(gross_mid, p.day)}; '
                  f'mean entry half-spread {np.mean(he / p.R2.values):.3f} R, mean exit cost {p.cost_R2.mean():.3f} R', flush=True)
            # 8. all_eligible outlier
            e = d[d.entered.astype(bool)]
            w = e.nsmallest(3, 'net_R2')[['day', 'symbol', 'entry', 'level', 'R2', 'exit', 'why', 'net_R2']]
            print(f'[8] worst all_eligible rows:\n{w.to_string()}', flush=True)


if __name__ == '__main__':
    main()
