"""Score cells 1,415-1,416 (PREREG.md) on the 2024H2 ORB book output, beside 2025 runB_true (cell 1,415 only).
Usage: python3 research/orb_2024/score.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_2024'
RISK = 375.0


def cell_stats(df: pd.DataFrame, label: str) -> dict:
    n_slots = len(df)
    ent = df[df['entered'] == 1].copy()
    ent['R'] = ent['_sized_pnl'] / RISK
    n = len(ent)
    no_fill_share = 1 - n / n_slots if n_slots else float('nan')
    if n == 0:
        return dict(label=label, n=0, n_slots=n_slots, no_fill_share=no_fill_share)
    total_usd = float(ent['_sized_pnl'].sum())
    net_R = float(ent['R'].mean())
    weeks = pd.to_datetime(ent['date']).dt.to_period('W').nunique()
    fills_per_week = n / weeks if weeks else float('nan')
    daily = ent.groupby('date')['R'].sum()
    t_cluster = (daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))
                 if len(daily) > 1 and daily.std(ddof=1) > 0 else float('nan'))
    k = max(1, int(np.ceil(0.05 * n)))
    ex_top5 = float(ent['R'].sort_values(ascending=False).iloc[k:].mean()) if n > k else float('nan')
    monthly = ent.groupby(ent['date'].astype(str).str[:7])['_sized_pnl'].sum()
    return dict(label=label, n=n, n_slots=n_slots, days=int(ent['date'].nunique()), weeks=int(weeks),
                fills_per_week=fills_per_week, net_R=net_R, t_cluster=t_cluster,
                total_usd=total_usd, ex_top5_R=ex_top5, no_fill_share=no_fill_share,
                best_month=float(monthly.max()) if len(monthly) else float('nan'),
                worst_month=float(monthly.min()) if len(monthly) else float('nan'))


def verdict(s: dict) -> str:
    if s['n'] == 0:
        return 'NO FILLS'
    if s['net_R'] <= -0.10:
        return 'RED FLAG'
    if s['net_R'] > 0 and s['total_usd'] > 0 and (pd.isna(s['ex_top5_R']) or s['ex_top5_R'] > 0):
        return 'SURVIVES'
    return 'neither'


def fmt(s: dict) -> str:
    if s['n'] == 0:
        return f"{s['label']}: NO ENTERED FILLS (slots {s['n_slots']}) -> {verdict(s)}"
    return (f"{s['label']}: n={s['n']} fills/wk={s['fills_per_week']:.2f} net_R/fill={s['net_R']:+.3f} "
            f"t={s['t_cluster']:+.2f} total=${s['total_usd']:,.0f} ex-top5%_R={s['ex_top5_R']:+.3f} "
            f"no-fill={s['no_fill_share']:.1%} best_mo=${s['best_month']:,.0f} "
            f"worst_mo=${s['worst_month']:,.0f} -> {verdict(s)}")


def main():
    # ORB_VERIFY_BOOK_SUFFIX (e.g. "_liveexit") points the scorer at book_{cid}{suffix}.csv and writes
    # REPORT{suffix}.md instead of the frozen REPORT.md. Verdict logic (verdict(), cell_stats()) is untouched.
    suffix = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('ORB_VERIFY_BOOK_SUFFIX', '')
    out_name = f'REPORT{suffix}.md' if suffix else 'REPORT.md'
    lines = ['# REPORT — cells 1,415-1,416, 2024H2 ORB holdout (PREREG.md)', '']
    for cid, path in (('1415', OUT / f'book_1415{suffix}.csv'), ('1416', OUT / f'book_1416{suffix}.csv')):
        if not path.exists():
            lines.append(f'## Cell {cid}: MISSING {path}\n')
            continue
        df = pd.read_csv(path)
        s = cell_stats(df, f'Cell {cid} (2024H2)')
        lines.append(f'## Cell {cid}\n\n- {fmt(s)}\n')
    b2025 = ROOT / 'research/orb_seed_wide/out/runB_true.csv'
    if b2025.exists():
        df25 = pd.read_csv(b2025)
        s25 = cell_stats(df25, 'runB_true 2025 (beside cell 1,415)')
        lines.append(f'## 2025 comparison (runB_true, production seed)\n\n- {fmt(s25)}\n')
    report = '\n'.join(lines)
    (OUT / out_name).write_text(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
