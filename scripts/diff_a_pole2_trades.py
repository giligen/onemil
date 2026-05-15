"""Diff trades between baseline (min_pole=3) and A_pole2 (min_pole=2) caches."""
import csv
import glob
import os


def load_trades(paths):
    trades = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['symbol'], row['date'])
                if key in trades:
                    key = (row['symbol'], row['date'], row.get('entry_time_et', ''))
                trades[key] = row
    return trades


def total_pnl(rows):
    return sum(float(r.get('pnl') or 0) for r in rows)


baseline = load_trades(sorted(glob.glob('/tmp/oos2025_baseline_2025-*.csv')))
a_pole2 = load_trades(sorted(glob.glob('/tmp/wide_A_pole2_2025-*.csv')))

print(f'Baseline (min_pole=3): {len(baseline)} trades  ${total_pnl(baseline.values()):+,.2f}')
print(f'A_pole2 (min_pole=2):  {len(a_pole2)} trades  ${total_pnl(a_pole2.values()):+,.2f}')
print()

common = set(baseline) & set(a_pole2)
only_b = set(baseline) - set(a_pole2)
only_a = set(a_pole2) - set(baseline)

print(f'Common keys:           {len(common)}')
print(f'Only in baseline:      {len(only_b)} (lost by A_pole2)')
print(f'Only in A_pole2:       {len(only_a)} (added by A_pole2)')
print()

# Different entry/exit for common
diff_common = []
for k in common:
    b, a = baseline[k], a_pole2[k]
    if b.get('entry_price') != a.get('entry_price') or b.get('exit_price') != a.get('exit_price'):
        diff_common.append((k, b, a))
print(f'Common but different entry/exit: {len(diff_common)}')
if diff_common:
    b_pnl = sum(float(b.get('pnl') or 0) for k, b, a in diff_common)
    a_pnl = sum(float(a.get('pnl') or 0) for k, b, a in diff_common)
    print(f'  baseline ${b_pnl:+,.2f} → A_pole2 ${a_pnl:+,.2f}  (delta ${a_pnl - b_pnl:+,.2f})')
print()

# Added trades — the new 2-candle-pole signals
added_rows = [a_pole2[k] for k in only_a]
added_pnl = sum(float(r.get('pnl') or 0) for r in added_rows)
added_wins = sum(1 for r in added_rows if float(r.get('pnl') or 0) > 0)
print(f'ADDED by A_pole2: {len(added_rows)} trades  ${added_pnl:+,.2f}  WR {100*added_wins/max(len(added_rows),1):.1f}%')
print(f'  avg P&L per added trade: ${added_pnl/max(len(added_rows),1):+,.2f}')

print('\n  Top 5 winners (newly accessible 2-candle setups):')
for r in sorted(added_rows, key=lambda r: -float(r.get('pnl') or 0))[:5]:
    print(f'    {r["symbol"]:<6} {r["date"]} {r.get("entry_time_et","")} entry ${r.get("entry_price","?"):>6} → ${r.get("exit_price","?"):>6} ({float(r.get("pnl") or 0):+,.0f}) {r.get("exit_reason","")}')
print('\n  Top 5 losers:')
for r in sorted(added_rows, key=lambda r: float(r.get('pnl') or 0))[:5]:
    print(f'    {r["symbol"]:<6} {r["date"]} {r.get("entry_time_et","")} entry ${r.get("entry_price","?"):>6} → ${r.get("exit_price","?"):>6} ({float(r.get("pnl") or 0):+,.0f}) {r.get("exit_reason","")}')

# Lost — trades that baseline had but A_pole2 doesn't
lost_rows = [baseline[k] for k in only_b]
if lost_rows:
    lost_pnl = sum(float(r.get('pnl') or 0) for r in lost_rows)
    print(f'\nLOST by A_pole2: {len(lost_rows)} trades  ${lost_pnl:+,.2f}')
