"""Walk-forward validation of candidate regime / conviction filters."""
import csv
import sqlite3
import statistics

TRADES_CSV = '/tmp/full_rebuild_a20_t10.csv'
CACHE_CSV = 'data/bull_flag_cache_e50_x30.csv'
DB_PATH = 'data/cache.db'

def load_trades():
    feat = {}
    with open(CACHE_CSV) as f:
        for r in csv.DictReader(f):
            feat[(r['symbol'], r['date'], r['entry_time_et'])] = r
    trades = []
    with open(TRADES_CSV) as f:
        for t in csv.DictReader(f):
            k = (t['symbol'], t['date'], t['entry_time_et'])
            merged = {**feat.get(k, {}), **t}
            merged['pnl'] = float(merged.get('pnl', 0))
            merged['entry_price'] = float(merged.get('entry_price', 0))
            merged['stop_loss'] = float(merged.get('stop_loss', 0))
            merged['shares'] = int(merged.get('shares', 0))
            merged['conviction_mult'] = float(merged.get('conviction_mult', 1.0) or 1.0)
            risk = merged['entry_price'] - merged['stop_loss']
            merged['r_mult'] = merged['pnl']/(risk*merged['shares']) if risk>0 and merged['shares']>0 else 0
            merged['win'] = 1 if merged['pnl'] > 0 else 0
            merged['month'] = merged['date'][:7]
            trades.append(merged)
    return trades

def get_spy_features():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""SELECT bar_date, open, high, low, close FROM daily_bars
                           WHERE symbol='SPY' AND bar_date >= '2024-10-01'
                           ORDER BY bar_date""").fetchall()
    conn.close()
    ranges = [(h-l)/c*100 for _,_,h,l,c in rows]
    closes = [r[4] for r in rows]
    dates = [r[0] for r in rows]
    features = {}
    for i, d in enumerate(dates):
        if i < 60: continue
        features[d] = {'spy_vol_5d': sum(ranges[i-5:i])/5}
    return features

def sim(trades, filt):
    kept = [t for t in trades if not filt(t)]
    pnl = sum(t['pnl'] for t in kept)
    wins = sum(t['win'] for t in kept)
    wr = wins/len(kept)*100 if kept else 0
    ordered = sorted(kept, key=lambda r: (r['date'], r['entry_time_et']))
    cum=0; peak=0; dd=0
    for r in ordered:
        cum += r['pnl']
        if cum > peak: peak = cum
        d = peak-cum
        if d>dd: dd=d
    return {'n': len(kept), 'pnl': pnl, 'wr': wr, 'dd': dd, 'blocked': len(trades)-len(kept)}

trades = load_trades()
spy = get_spy_features()
for t in trades:
    t.update(spy.get(t['date'], {'spy_vol_5d': 0}))

# =============================================================
# Walk-forward splits
# =============================================================
# Train on 2025 (Jan-Dec, 12 months), test on 2026 (Jan-Apr, 4 months)
train = [t for t in trades if t['month'] < '2026-01']
test = [t for t in trades if t['month'] >= '2026-01']

print("=" * 90)
print(f"WALK-FORWARD: train=2025 ({len(train)} tr), test=2026-Q1+Q2 ({len(test)} tr)")
print("=" * 90)

filters = [
    ('No filter', lambda t: False),
    ('[prod] vol>5', lambda t: t.get('spy_vol_5d', 0) > 5.0),
    ('vol>5 OR conv<1.2', lambda t: t.get('spy_vol_5d', 0) > 5 or t['conviction_mult'] < 1.2),
    ('vol>5 OR conv<1.3', lambda t: t.get('spy_vol_5d', 0) > 5 or t['conviction_mult'] < 1.3),
    ('conv<1.2', lambda t: t['conviction_mult'] < 1.2),
    ('conv<1.3', lambda t: t['conviction_mult'] < 1.3),
]

def tbl(label, trades_set, filters):
    print(f"\n--- {label} (n={len(trades_set)}) ---")
    print(f"{'Filter':<30} {'n':>5} {'Blocked':>7} {'P&L':>12} {'Δvs base':>10} {'DD':>10} {'WR':>6}")
    print('-'*85)
    base = sim(trades_set, lambda t: False)
    for name, fn in filters:
        r = sim(trades_set, fn)
        delta = r['pnl'] - base['pnl']
        marker = ' ★' if delta > 0 and name != 'No filter' else ''
        print(f"{name:<30} {r['n']:>5} {r['blocked']:>7} ${r['pnl']:>+10,.0f} ${delta:>+8,.0f} ${r['dd']:>8,.0f} {r['wr']:>4.1f}%{marker}")

tbl('TRAIN (2025)', train, filters)
tbl('TEST (2026 Q1+Q2)', test, filters)

# Month-by-month on test
print("\n--- Test set month-by-month with 'vol>5 OR conv<1.3' ---")
from collections import defaultdict
by_mo = defaultdict(list)
for t in test:
    by_mo[t['month']].append(t)
filt = lambda t: t.get('spy_vol_5d', 0) > 5 or t['conviction_mult'] < 1.3
for mo in sorted(by_mo):
    sub = by_mo[mo]
    r = sim(sub, filt)
    base_pnl = sum(t['pnl'] for t in sub)
    base_wr = sum(t['win'] for t in sub)/len(sub)*100 if sub else 0
    print(f"  {mo}: orig {len(sub):>2}tr {base_wr:>4.0f}% ${base_pnl:>+9,.0f}  →  filter {r['n']:>2}tr {r['wr']:>4.0f}% ${r['pnl']:>+9,.0f}  (Δ${r['pnl']-base_pnl:+,.0f})")

# Also try the reverse: train on 2026-H1, test on 2025-H2 (4 months)
print("\n" + "=" * 90)
print("REVERSE WALK: train=2025-H1 (Jan-Jun), test=2025-H2 (Jul-Dec)")
print("=" * 90)
train2 = [t for t in trades if '2025-01' <= t['month'] <= '2025-06']
test2 = [t for t in trades if '2025-07' <= t['month'] <= '2025-12']
tbl('TRAIN 2025-H1', train2, filters)
tbl('TEST 2025-H2', test2, filters)

# Split 3: train on Jan-Sep, test on Oct-Apr
print("\n" + "=" * 90)
print("SPLIT 3: train=2025 Jan-Sep (~9mo, 105 trades), test=2025 Oct–2026 Apr (~7mo)")
print("=" * 90)
train3 = [t for t in trades if t['month'] <= '2025-09']
test3 = [t for t in trades if t['month'] >= '2025-10']
tbl('TRAIN Jan-Sep 2025', train3, filters)
tbl('TEST Oct 2025 - Apr 2026', test3, filters)
