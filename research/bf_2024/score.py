"""Score bull-flag P1 on 2024H2 vs PREREG.md (cell 1,417) verdict. research/bf_2024/SPEC.md step 3.
Reads research/bf_2024/stage2_2024.csv (Stage-2 P1 trades), computes n, mean R, day-clustered t, total $,
ex-top-5% mean R, monthly $, puts the P1 2025-26 book beside it, applies the PREREG verdict rule, writes
research/bf_2024/REPORT.md. Never touches trading/, config.yaml, or any production cache.
"""
import math
import os
from pathlib import Path

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
STAGE2_CSV = ROOT / 'research/bf_2024/stage2_2024.csv'
P1_BOOK_CSV = ROOT / 'research/bf_frequency/runs/P1.csv'
REPORT = ROOT / 'research/bf_2024/REPORT.md'


def _r_multiple(df: pd.DataFrame) -> pd.Series:
    """R = pnl / (shares * |entry - stop|), the ramp's R basis (PREREG.md). Falls back to whatever
    R/pnl columns the Stage-2 CSV actually has, logging which path was used (no silent fallback)."""
    cols = set(df.columns)
    if {'pnl', 'shares', 'entry_price', 'stop_price'} <= cols:
        risk_per_share = (df['entry_price'] - df['stop_price']).abs()
        denom = (df['shares'].astype(float) * risk_per_share).replace(0, pd.NA)
        r = df['pnl'].astype(float) / denom
        print(f"[score] R via pnl/(shares*|entry-stop|), {r.isna().sum()} NA denom rows dropped")
        return r
    if 'r_multiple' in cols:
        print("[score] WARNING: using CSV's own r_multiple column (entry/stop/shares not all present)")
        return df['r_multiple'].astype(float)
    if {'pnl', 'risk_dollars'} <= cols:
        print("[score] WARNING: R via pnl/risk_dollars fallback (no entry/stop columns)")
        return df['pnl'].astype(float) / df['risk_dollars'].astype(float).replace(0, pd.NA)
    raise SystemExit(f"[score] ERROR: no usable R columns in {STAGE2_CSV}; have: {sorted(cols)}")


def _day_clustered_t(df: pd.DataFrame, date_col: str, val_col: str) -> float:
    """t-stat on day-clustered means (one obs per trading day), per CLAUDE.md's day-clustered SE rule."""
    by_day = df.groupby(date_col)[val_col].mean()
    n = len(by_day)
    if n < 2:
        return float('nan')
    se = by_day.std(ddof=1) / math.sqrt(n)
    return float('nan') if se == 0 else by_day.mean() / se


def main():
    if not STAGE2_CSV.exists():
        raise SystemExit(f"[score] ERROR: {STAGE2_CSV} missing — Stage 2 did not run or produced nothing")
    df = pd.read_csv(STAGE2_CSV)
    lines = ["# REPORT — bull-flag P1 on 2024H2 (PREREG.md cell 1,417)", ""]
    n = len(df)
    lines.append(f"Stage-2 P1 trades: n={n}")
    if n == 0:
        lines.append("\nZERO trades — cannot score mean R / t / ex-top-5% / monthly $. See chain.log and "
                      "SPEC.md's own-report seam/parity caveats before treating this as a verdict.")
        REPORT.write_text("\n".join(lines) + "\n")
        print(f"[score] WROTE {REPORT} (n=0, no verdict)")
        return

    date_col = 'date' if 'date' in df.columns else ('entry_date' if 'entry_date' in df.columns else None)
    df['R'] = _r_multiple(df)
    df = df.dropna(subset=['R'])
    mean_r = df['R'].mean()
    total_usd = df['pnl'].sum() if 'pnl' in df.columns else float('nan')
    t_stat = _day_clustered_t(df, date_col, 'R') if date_col else float('nan')

    k = max(1, int(round(len(df) * 0.05)))
    ex_top5 = df.sort_values('R', ascending=False).iloc[k:]
    ex_top5_mean_r = ex_top5['R'].mean() if len(ex_top5) else float('nan')

    lines.append(f"Mean R: {mean_r:.4f}")
    lines.append(f"Day-clustered t: {t_stat:.3f}" if date_col else "Day-clustered t: N/A (no date column)")
    lines.append(f"Total $: {total_usd:,.2f}")
    lines.append(f"Ex-top-5% mean R (dropped {k}): {ex_top5_mean_r:.4f}")

    if date_col:
        df['_month'] = df[date_col].astype(str).str.slice(0, 7)
        monthly = df.groupby('_month')['pnl'].sum() if 'pnl' in df.columns else None
        if monthly is not None:
            lines.append("\n## Monthly $")
            for m, v in monthly.items():
                lines.append(f"- {m}: {v:,.2f}")

    lines.append("\n## P1 2025-26 book (beside it)")
    if P1_BOOK_CSV.exists():
        p1 = pd.read_csv(P1_BOOK_CSV)
        p1_r_col = 'R' if 'R' in p1.columns else ('r_multiple' if 'r_multiple' in p1.columns else None)
        if p1_r_col:
            lines.append(f"P1 2025-26: n={len(p1)}, mean R={p1[p1_r_col].mean():.4f}")
        else:
            lines.append(f"P1 2025-26: n={len(p1)} (no R/r_multiple column found; columns={list(p1.columns)})")
    else:
        lines.append(f"WARNING: {P1_BOOK_CSV} not found — cannot show the 2025-26 book beside 2024H2")

    survives = (mean_r > 0) and (total_usd > 0) and (ex_top5_mean_r > 0)
    red_flag = mean_r <= -0.10
    verdict = "SURVIVES" if survives else ("RED FLAG — recommend pausing bull flag live pending review"
                                            if red_flag else "DOES NOT SURVIVE (not a red flag)")
    lines.append(f"\n## PREREG verdict: {verdict}")
    lines.append(f"(mean_r={mean_r:.4f} > 0: {mean_r > 0}; total_$={total_usd:,.2f} > 0: {total_usd > 0}; "
                  f"ex_top5_mean_r={ex_top5_mean_r:.4f} > 0: {ex_top5_mean_r > 0})")

    REPORT.write_text("\n".join(lines) + "\n")
    print(f"[score] WROTE {REPORT}: n={n} mean_r={mean_r:.4f} verdict={verdict}")


if __name__ == '__main__':
    main()
