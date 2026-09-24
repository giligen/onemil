"""Score cells 1,418-1,419 (research/orb_2023/PREREG.md) and the pooled out-of-regime ORB estimate.

Per cell: entered fills, net R/fill (R = _sized_pnl / 375), trade-weighted day-clustered t, ex-top-5 %, calendar
halves (2023 vs 2024H1), fills/week, monthly $, worst month, no-fill share, and the frozen verdict
(EDGE / FLAT / NEGATIVE). Cell 1,418 is pooled with the 2024H2 production book (research/orb_2024/book_1415.csv).
Writes research/orb_2023/REPORT.md.

Usage: python3 research/orb_2023/score_2023.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'research/hod_consol'))
from adversarial_read import stats  # noqa: E402

R_USD = 375.0


def book(path):
    b = pd.read_csv(path, keep_default_na=False, na_values=[''])
    n_all = len(b)
    e = b[b.entered == 1].copy()
    e['R'] = e._sized_pnl / R_USD
    e['date'] = e.date.astype(str)
    return e, n_all


def verdict(s, h_2023, h_2024):
    if s['mean'] >= 0.10 and s['t_cluster'] >= 2 and s['ex5'] > 0 and h_2023 > 0 and h_2024 > 0:
        return 'EDGE'
    if s['mean'] <= -0.10:
        return 'NEGATIVE'
    if abs(s['mean']) < 0.10 and s['t_cluster'] < 2:
        return 'FLAT'
    return 'NEITHER (between the pre-registered bands)'


def main():
    lines = ['# REPORT — live ORB on 2023-01 .. 2024-06, point-in-time universe (cells 1,418–1,419)', '']
    weeks = None
    for cell in ('1418', '1419'):
        p = HERE / f'book_{cell}.csv'
        if not p.exists():
            lines.append(f'Cell {cell}: MISSING book ({p})')
            continue
        e, n_all = book(p)
        s = stats(e.R, e.date)
        k = max(1, int(round(0.05 * len(e))))
        s['ex5'] = float(e.R.drop(e.R.nlargest(k).index).mean()) if len(e) > 2 else float('nan')
        h23 = float(e[e.date < '2024-01-01'].R.mean())
        h24 = float(e[e.date >= '2024-01-01'].R.mean())
        weeks = pd.to_datetime(e.date).dt.to_period('W').nunique() if weeks is None else weeks
        m = e.groupby(e.date.str[:7])._sized_pnl.sum()
        v = verdict(s, h23, h24)
        lines += [f'## Cell {cell} ({"production seed gap ≥ 5 %, $3–30" if cell == "1418" else "addon_p30 gap 3–5 %, $30–50"})',
                  '', f'- fills {s["n"]} (of {n_all} picks, no-fill share {1 - s["n"] / max(1, n_all):.0%}) over '
                  f'{s["days"]} days, {s["n"] / 78:.1f} fills/week (78 weeks)',
                  f'- net R/fill {s["mean"]:+.3f}, t (day-clustered) {s["t_cluster"]:+.2f}, ex-top-5 % {s["ex5"]:+.3f}, '
                  f'total ${e._sized_pnl.sum():+,.0f}',
                  f'- halves: 2023 {h23:+.3f} R/fill, 2024H1 {h24:+.3f} R/fill',
                  f'- months: worst {m.min():+,.0f} ({m.idxmin()}), best {m.max():+,.0f} ({m.idxmax()}), '
                  f'green months {(m > 0).mean():.0%} of {len(m)}',
                  f'- **VERDICT (frozen): {v}**', '']
        if cell == '1418':
            b24, _ = book(ROOT / 'research/orb_2024/book_1415.csv')
            pool = pd.concat([e[['R', 'date']], b24[['R', 'date']]], ignore_index=True)
            ps = stats(pool.R, pool.date)
            se = ps['mean'] / ps['t_cluster'] if ps.get('t_cluster') else float('nan')
            lines += [f'- **Pooled out-of-regime ORB (2023-01 .. 2024-12, same code): n {ps["n"]}, {ps["mean"]:+.3f} R/fill '
                      f'± {se:.3f} (t {ps["t_cluster"]:+.2f})** — vs 2025 +0.272 (n 127)', '']
    (HERE / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
