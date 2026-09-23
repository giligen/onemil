"""Post-hoc diagnostics of cell 1,412's KEPT book (disclosed: not in PREREG_1412, no rule is selected here).

Per split (TRAIN, VAL, TEST): (1) day and week distribution of the unconstrained kept book (best/worst day, share of R
on the top 10 % of days, weekly P10/min, cadence bar via scripts/cadence_bar.py); (2) order within the day: the
first 4 kept signals of each day vs the later ones; (3) placebo — random-minute longs on the SAME name-days, only
at minutes where the tape is also risk-on (BR >= 0.6115 at the decision minute), identical C1 exit and cost.

Usage: python3 research/day_breadth/kept_diag.py
"""
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'research/hod_consol'))
sys.path.insert(0, str(HERE))
import run_consol as rc  # noqa: E402
import breadth as B  # noqa: E402
import test_1412 as T  # noqa: E402
from adversarial_read import stats  # noqa: E402

EDGE = T.EDGE


def placebo_riskon(kept, idx, brmap, rng):
    """Per kept trade: mean of up to 8 random-minute C1 longs on the same name-day at risk-on minutes."""
    vals = []
    for r in kept.itertuples():
        key = (r.day, r.symbol)
        if key not in idx.index:
            continue
        g = idx.loc[[key]]
        draws = []
        for _ in range(rc.N_DRAWS):
            for _try in range(10):
                m_r = rng.randint(rc.SCAN_START_M, rc.SCAN_END_M)
                if brmap.get((r.day, m_r - 1), 0) < EDGE:
                    continue
                prior = g[(g.m >= m_r - rc.BASE_WINDOW) & (g.m < m_r)]
                eb = g[g.m == m_r]
                if len(prior) < rc.BASE_WINDOW or eb.empty:
                    continue
                entry, stop = float(eb.iloc[0].o), float(prior.l.min())
                ok, R = rc._floor_ok(entry, stop)
                if not ok:
                    continue
                after = g[(g.m > m_r) & (g.m <= rc.EOD_M)]
                res = rc.fill_c1(entry, stop, R, after.itertuples()) if not after.empty else None
                if res is None:
                    continue
                _, net = rc.cost_net(entry, res[1], R, 2 * rc.PROXY_HALF_SPREAD_PCT * entry)
                if not pd.isna(net):
                    draws.append(net)
                break
        if draws:
            vals.append(float(np.mean(draws)))
    return vals


def cadence(k, split):
    p = HERE / f'kept_{split}.csv'
    k.assign(date=k.day, pnl_R=k.net_R)[['date', 'pnl_R']].to_csv(p, index=False)
    r = subprocess.run([sys.executable, str(ROOT / 'scripts/cadence_bar.py'), '--trades', str(p), '--split', 'ALL'],
                       capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()


def main():
    br_tv = pd.read_parquet(HERE / 'breadth.parquet')[['day', 'm', 'BR']]
    br_te = T.test_breadth()
    br = pd.concat([br_tv, br_te], ignore_index=True)
    brmap = dict(zip(zip(br.day, br.m), br.BR))
    sig = pd.read_parquet(ROOT / 'research/hod_consol/signals.parquet')
    paths = pd.read_parquet(ROOT / 'research/hod_consol/paths.parquet')
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    w = rc.walk(sig, idx, rc.fill_c1)
    w['BR'] = [brmap.get((d, int(m)), np.nan) for d, m in zip(w.day, w.signal_m)]
    kept_all = w[w.BR >= EDGE].copy()
    lines = ['# KEPT_DIAG.md — post-hoc diagnostics of cell 1,412 kept book (risk-on tape, all signals)', '']
    for split in ('TRAIN', 'VAL', 'TEST'):
        k = kept_all[kept_all.split == split].sort_values(['day', 'entry_m'])
        if k.empty:
            continue
        day_R = k.groupby('day').net_R.sum()
        wk_R = k.groupby('wk').net_R.sum()
        top = day_R.sort_values(ascending=False)
        n10 = max(1, int(round(0.1 * len(top))))
        k['order'] = k.groupby('day').cumcount()
        first4, later = k[k.order < 4], k[k.order >= 4]
        pl = placebo_riskon(k, idx, brmap, random.Random(1412 + len(split)))
        s = stats(k.net_R, k.day)
        lines += [f'## {split}: kept {len(k)} trades on {k.day.nunique()} days, mean {s["mean"]:+.3f} R '
                  f'(t {s["t_cluster"]:+.2f})', '',
                  f'- day R: best {top.iloc[0]:+.1f}, worst {top.iloc[-1]:+.1f}, median {day_R.median():+.2f}, '
                  f'green days {(day_R > 0).mean():.0%}; top 10 % of days = {top.iloc[:n10].sum() / day_R.sum():.0%} of R',
                  f'- week R: worst {wk_R.min():+.1f}, P10 {wk_R.quantile(0.1):+.1f}, green weeks {(wk_R > 0).mean():.0%} '
                  f'of {len(wk_R)}',
                  f'- order within the day: first 4 kept signals {first4.net_R.mean():+.3f} R (n {len(first4)}) vs '
                  f'later ones {later.net_R.mean():+.3f} R (n {len(later)})',
                  f'- placebo (random-minute long, same name-day, risk-on minutes, C1): {np.mean(pl):+.3f} R '
                  f'(n {len(pl)}) → signal − placebo {s["mean"] - np.mean(pl):+.3f} R', '',
                  '```', cadence(k, split), '```', '']
        print('\n'.join(lines[-11:]), flush=True)
    (HERE / 'KEPT_DIAG.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
