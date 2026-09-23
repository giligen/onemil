"""Score cells 1,413-1,414 (PREREG_2024.md) on the 2024H2 holdout with the frozen code.

One pass over y2024/bars.db: 09:30-09:35 volume floor -> universe, breadth flags per day-minute
(breadth.symbol_minute_flags), base signals (run_consol.find_signal / build_signal) with full-day paths. Then C1 walk
(run_consol.walk), BR at signal_m, kept = BR >= test_1412.EDGE, order_in_day among kept; placebo
(kept_diag.placebo_riskon); stats (adversarial_read.stats); weekly Sharpe / green share; slotted book; verdicts.
Writes REPORT_2024.md and trades_2024.csv.

Usage: python3 research/day_breadth/y2024/score.py
"""
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
HERE = Path(__file__).resolve().parent
for p in (ROOT / 'research/hod_consol', ROOT / 'research/day_breadth'):
    sys.path.insert(0, str(p))
import run_consol as rc  # noqa: E402
import breadth as B  # noqa: E402
import test_1412 as T  # noqa: E402
import kept_diag as K  # noqa: E402
from adversarial_read import stats  # noqa: E402

FLOOR_935 = 15_000


def one_pass():
    cands = pd.read_csv(HERE / 'candidates.csv', keep_default_na=False, na_values=[''])
    con = sqlite3.connect(HERE / 'bars.db')
    acc, sig_rows, frames = {}, [], []
    n_bars = n_floor = 0
    for s, d in zip(cands.symbol, cands.bar_date):
        bars, _ = rc.fetch_day_bars_dual(con, None, s, d)
        if bars is None or bars.empty:
            continue
        n_bars += 1
        if bars[(bars.m >= 570) & (bars.m <= 574)].v.sum() < FLOOR_935:
            continue
        n_floor += 1
        has, above, _, _ = B.symbol_minute_flags(bars)
        a = acc.setdefault(d, np.zeros((2, B.NM), dtype=np.int32))
        a[0] += has
        a[1] += above
        sigbar, _ = rc.find_signal(bars)
        if sigbar is None:
            continue
        sig, _ = rc.build_signal(bars, sigbar, s, d)
        if sig is None:
            continue
        sig_rows.append(sig)
        p = bars[['m', 'o', 'h', 'l', 'c']].copy()
        p['day'], p['symbol'] = d, s
        frames.append(p)
    con.close()
    print(f'candidates {len(cands)}, with RTH bars {n_bars} ({n_bars / len(cands):.1%}), '
          f'after the 09:35 floor {n_floor}, signals {len(sig_rows)}', flush=True)
    brmap = {}
    for d, a in acc.items():
        with np.errstate(divide='ignore', invalid='ignore'):
            br = a[1] / a[0]
        for i, v in enumerate(br):
            brmap[(d, B.M0 + i)] = v
    return sig_rows, pd.concat(frames, ignore_index=True), brmap, n_bars / len(cands)


def weekly(t, all_weeks):
    w = t.groupby('wk').net_R.sum().reindex(all_weeks, fill_value=0.0)
    decided = w[w.abs() >= 0.5]
    green = float((decided > 0).mean()) if len(decided) else float('nan')
    sharpe = float(w.mean() / w.std(ddof=1)) if w.std(ddof=1) > 0 else float('nan')
    return w, green, sharpe


def main():
    sig_rows, paths, brmap, avail = one_pass()
    sig = pd.DataFrame(sig_rows)
    sig['split'], sig['half'] = 'Y2024', None
    sig['wk'] = sig.day.map(rc.week_monday)
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    w = rc.walk(sig, idx, rc.fill_c1)
    w['BR'] = [brmap.get((d, int(m)), np.nan) for d, m in zip(w.day, w.signal_m)]
    w = w.dropna(subset=['BR']).sort_values(['day', 'entry_m'])
    w['kept'] = w.BR >= T.EDGE
    w.loc[w.kept, 'order'] = w[w.kept].groupby('day').cumcount()
    w.to_csv(HERE / 'trades_2024.csv', index=False)
    all_weeks = sorted({rc.week_monday(d) for d in pd.read_csv(HERE / 'candidates.csv').bar_date.unique()})
    n_weeks = len(all_weeks)
    lines = ['# REPORT_2024.md — cells 1,413–1,414 on the untouched 2024H2 holdout (frozen rule, frozen code)', '',
             f'Availability: {avail:.1%} of candidate symbol-days have Alpaca RTH bars '
             f'({"PASS" if avail >= 0.8 else "VOID — below the 80 % rail"}). Signals walked: {len(w)}; '
             f'kept (BR ≥ {T.EDGE}): {int(w.kept.sum())} on {w[w.kept].day.nunique()} days; weeks: {n_weeks}.', '']
    rest = w[~w.kept]
    for cid, book in (('1413', w[w.kept]), ('1414', w[w.kept & (w.order >= 4)])):
        s = stats(book.net_R, book.day)
        pl = K.placebo_riskon(book, idx, brmap, random.Random(int(cid)))
        pl_mean = float(np.mean(pl)) if pl else float('nan')
        wk, green, sharpe = weekly(book, all_weeks)
        keep = rc.simulate_slots(book)
        slotted = book.loc[keep.index[keep]]
        ss = stats(slotted.net_R, slotted.day)
        day_R = book.groupby('day').net_R.sum().sort_values(ascending=False)
        n10 = max(1, int(round(0.1 * len(day_R))))
        fills_wk = len(book) / n_weeks
        passed = (s['mean'] >= 0.10 and s['t_cluster'] >= 2 and s['mean'] - pl_mean >= 0.10 and sharpe >= 0.30
                  and green >= 0.55 and fills_wk >= 3
                  and (cid != '1413' or s['mean'] - rest.net_R.mean() >= 0.10))
        lines += [f'## Cell {cid} — {"all kept signals" if cid == "1413" else "kept, skipping each day’s first four"}', '',
                  f'- n {s["n"]} on {s["days"]} days · net {s["mean"]:+.3f} R · t {s["t_cluster"]:+.2f} · '
                  f'day-weighted {s["day_mean"]:+.3f}',
                  f'- placebo (random risk-on long, same name-day) {pl_mean:+.3f} R → setup − placebo {s["mean"] - pl_mean:+.3f} R',
                  (f'- rest (BR below the edge) {rest.net_R.mean():+.3f} R (n {len(rest)}) → kept − rest '
                   f'{s["mean"] - rest.net_R.mean():+.3f} R' if cid == '1413' else '- (rest comparison: cell 1,413 only)'),
                  f'- weekly: Sharpe {sharpe:+.2f}, green {green:.0%} of decided weeks, worst week {wk.min():+.1f} R, '
                  f'best week {wk.max():+.1f} R, fills/week {fills_wk:.1f}',
                  f'- days: worst {day_R.min():+.1f} R, best {day_R.max():+.1f} R, top 10 % of days = '
                  f'{day_R.iloc[:n10].sum() / day_R.sum():.0%} of R' if day_R.sum() else '- days: net zero',
                  f'- slotted (first 4 at once, 12/day): n {ss.get("n")} net {ss.get("mean", float("nan")):+.3f} R',
                  f'- **PASS = {passed}**', '']
    (HERE / 'REPORT_2024.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
