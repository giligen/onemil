"""Diff trades between baseline (pb=5) and B_pb10 (pb=10) caches.

Answers:
- How many trades are unique to each cell?
- Of trades present in both, do entry/exit/P&L differ (different pattern)?
- Which months drive the P&L delta?
- What's the worst-case scenario (most negative individual added trade)?
"""
import csv
import glob
import os
from collections import defaultdict

# All caches: 2025 per-month + 2026 single
BASELINE_CACHES = sorted(glob.glob('/tmp/oos2025_baseline_2025-*.csv')) + ['/tmp/sweep_baseline.csv']
B_PB10_CACHES = sorted(glob.glob('/tmp/oos2025_B_pb10_2025-*.csv')) + ['/tmp/sweep_B_pb10.csv']


def load_trades(paths):
    """Return dict[(symbol, date)] = row_dict."""
    trades = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['symbol'], row['date'])
                if key in trades:
                    # Multiple trades same symbol+date — distinguish by entry time
                    key = (row['symbol'], row['date'], row['entry_time_et'])
                trades[key] = row
    return trades


def total_pnl(rows):
    return sum(float(r.get('pnl') or 0) for r in rows)


def main():
    baseline = load_trades(BASELINE_CACHES)
    b_pb10 = load_trades(B_PB10_CACHES)

    print(f'Baseline (pb=5):  {len(baseline)} trades  ${total_pnl(baseline.values()):+,.2f}')
    print(f'B_pb10 (pb=10):   {len(b_pb10)} trades  ${total_pnl(b_pb10.values()):+,.2f}')
    print(f'Delta: {len(b_pb10) - len(baseline):+d} trades, '
          f'${total_pnl(b_pb10.values()) - total_pnl(baseline.values()):+,.2f}')
    print()

    common_keys = set(baseline) & set(b_pb10)
    only_baseline = set(baseline) - set(b_pb10)
    only_b_pb10 = set(b_pb10) - set(baseline)

    print(f'  Common (same symbol+date): {len(common_keys)}')
    print(f'  Only in baseline:          {len(only_baseline)}')
    print(f'  Only in B_pb10:            {len(only_b_pb10)}')
    print()

    # Common but different entry — pb=10 found a different pattern instance
    diff_entry = []
    for k in common_keys:
        b = baseline[k]
        x = b_pb10[k]
        if (b.get('entry_time_et') != x.get('entry_time_et')
                or b.get('entry_price') != x.get('entry_price')):
            diff_entry.append((k, b, x))
    same_entry = len(common_keys) - len(diff_entry)
    print(f'  Of common: {same_entry} identical entry, {len(diff_entry)} differ')

    if diff_entry:
        pnl_b_total = sum(float(b.get('pnl') or 0) for k, b, x in diff_entry)
        pnl_x_total = sum(float(x.get('pnl') or 0) for k, b, x in diff_entry)
        print(f'  Differ-entry P&L: baseline ${pnl_b_total:+,.2f} → B_pb10 ${pnl_x_total:+,.2f}  '
              f'(delta ${pnl_x_total - pnl_b_total:+,.2f})')

    # Trades unique to B_pb10
    if only_b_pb10:
        ub_rows = [b_pb10[k] for k in only_b_pb10]
        ub_pnl = sum(float(r.get('pnl') or 0) for r in ub_rows)
        ub_wins = sum(1 for r in ub_rows if float(r.get('pnl') or 0) > 0)
        print(f'  Added by B_pb10: {len(only_b_pb10)} trades  ${ub_pnl:+,.2f}  '
              f'WR {100*ub_wins/len(ub_rows):.0f}%')
        print('  Top 5 winners ADDED by B_pb10:')
        for r in sorted(ub_rows, key=lambda r: -float(r.get('pnl') or 0))[:5]:
            print(f'    {r["symbol"]:6} {r["date"]} entry ${r["entry_price"]:>6} → exit ${r["exit_price"]:>6} '
                  f'(${float(r.get("pnl") or 0):+,.2f}) {r.get("exit_reason","")}')
        print('  Top 5 losers ADDED by B_pb10:')
        for r in sorted(ub_rows, key=lambda r: float(r.get('pnl') or 0))[:5]:
            print(f'    {r["symbol"]:6} {r["date"]} entry ${r["entry_price"]:>6} → exit ${r["exit_price"]:>6} '
                  f'(${float(r.get("pnl") or 0):+,.2f}) {r.get("exit_reason","")}')

    if only_baseline:
        ob_rows = [baseline[k] for k in only_baseline]
        ob_pnl = sum(float(r.get('pnl') or 0) for r in ob_rows)
        ob_wins = sum(1 for r in ob_rows if float(r.get('pnl') or 0) > 0)
        print(f'  LOST by B_pb10:  {len(only_baseline)} trades  ${ob_pnl:+,.2f}  '
              f'WR {100*ob_wins/len(ob_rows):.0f}%')


if __name__ == '__main__':
    main()
