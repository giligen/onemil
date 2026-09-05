"""Compare two ORB book CSVs (study_orb_pipeline_static_lock output) — 2026-09-05.

Usage: python3 compare_books.py BOOK_A.csv BOOK_B.csv [LABEL_A] [LABEL_B]

Prints, for each book: trades, entered trades (if the column exists),
sized P&L, WR, max drawdown on the daily sized-P&L curve, worst/best month,
negative months; then the monthly side-by-side and the (symbol, date) picks
that differ. Relative tool only — never a forecast.
"""
import sys

import pandas as pd


def load(path: str) -> pd.DataFrame:
    """Load a book CSV and normalise the columns this comparison needs."""
    b = pd.read_csv(path)
    b['date'] = pd.to_datetime(b['date']).dt.strftime('%Y-%m-%d')
    b['month'] = b['date'].str[:7]
    if '_sized_pnl' not in b.columns:
        b['_sized_pnl'] = b['pnl']
    if 'entered' not in b.columns:
        b['entered'] = 1
    return b


def max_drawdown(daily: pd.Series) -> float:
    """Max peak-to-trough drawdown of a cumulative daily P&L series."""
    cum = daily.cumsum()
    return float((cum - cum.cummax()).min())


def summary(b: pd.DataFrame, label: str) -> dict:
    """One-row book summary."""
    ent = b[b['entered'] == 1]
    daily = b.groupby('date')['_sized_pnl'].sum().sort_index()
    monthly = b.groupby('month')['_sized_pnl'].sum()
    return {
        'book': label, 'picks': len(b), 'entered': len(ent),
        'fill_rate%': round(len(ent) / max(len(b), 1) * 100, 1),
        'pnl': round(b['_sized_pnl'].sum()), 'wr%': round((ent['_sized_pnl'] > 0).mean() * 100, 1) if len(ent) else 0,
        'mdd': round(max_drawdown(daily)), 'worst_month': round(monthly.min()), 'best_month': round(monthly.max()),
        'neg_months': int((monthly < 0).sum()), 'months': len(monthly),
    }


def main() -> None:
    pa, pb = sys.argv[1], sys.argv[2]
    la = sys.argv[3] if len(sys.argv) > 3 else 'A'
    lb = sys.argv[4] if len(sys.argv) > 4 else 'B'
    a, b = load(pa), load(pb)
    print(pd.DataFrame([summary(a, la), summary(b, lb)]).to_string(index=False))
    ma = a.groupby('month')['_sized_pnl'].sum().rename(la)
    mb = b.groupby('month')['_sized_pnl'].sum().rename(lb)
    m = pd.concat([ma, mb], axis=1).fillna(0).round(0)
    m['delta'] = m[lb] - m[la]
    print("\nmonthly:\n" + m.to_string())
    ka = set(zip(a['symbol'], a['date']))
    kb = set(zip(b['symbol'], b['date']))
    only_b = b[[k not in ka for k in zip(b['symbol'], b['date'])]]
    only_a = a[[k not in kb for k in zip(a['symbol'], a['date'])]]
    cols = [c for c in ('date', 'symbol', 'entered', 'exit_reason', '_quintile', '_sized_pnl') if c in b.columns]
    print(f"\npicks only in {lb} ({len(only_b)}), sized P&L ${only_b['_sized_pnl'].sum():+,.0f}:")
    print(only_b.sort_values('date')[cols].to_string(index=False) if len(only_b) else '  (none)')
    print(f"\npicks only in {la} ({len(only_a)}), sized P&L ${only_a['_sized_pnl'].sum():+,.0f}:")
    print(only_a.sort_values('date')[cols].to_string(index=False) if len(only_a) else '  (none)')


if __name__ == '__main__':
    main()
