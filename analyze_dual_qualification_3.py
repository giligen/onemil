"""
Final targeted analysis:
- Gap-up extra trades (Path A) broken down by daily_range_pct
- What makes t15 extra good but t8 extra gap-ups bad?
- Monthly breakdown to check recency bias
"""
import sqlite3
import pandas as pd

DB_PATH = "data/cache.db"
GAP_THRESHOLD = 0.10


def compute_gap_pct_batch(df, conn):
    """Add gap_pct column using daily_bars open vs prev_close."""
    from datetime import date as date_type
    results = {}
    pairs = df[['symbol', 'date']].drop_duplicates()
    for _, row in pairs.iterrows():
        sym = row['symbol']
        dt = row['date'] if isinstance(row['date'], date_type) else row['date'].date()
        dt_str = str(dt)
        cur = conn.cursor()
        cur.execute(
            "SELECT open FROM daily_bars WHERE symbol = ? AND bar_date = ?",
            (sym, dt_str)
        )
        r = cur.fetchone()
        today_open = float(r[0]) if r else None
        cur.execute(
            "SELECT close FROM daily_bars WHERE symbol = ? AND bar_date < ? ORDER BY bar_date DESC LIMIT 1",
            (sym, dt_str)
        )
        r = cur.fetchone()
        prev_close = float(r[0]) if r else None
        if today_open and prev_close and prev_close > 0:
            results[(sym, dt_str)] = (today_open - prev_close) / prev_close
        else:
            results[(sym, dt_str)] = None
    gap_pcts = []
    for _, row in df.iterrows():
        dt = row['date'] if isinstance(row['date'], date_type) else row['date'].date()
        gap_pcts.append(results.get((row['symbol'], str(dt))))
    return gap_pcts


def stats(label, df):
    if df.empty:
        print(f"  {label}: 0 trades")
        return
    w = df[df['pnl'] > 0]
    l = df[df['pnl'] <= 0]
    wr = len(w) / len(df) * 100
    pf = abs(w['pnl'].sum() / l['pnl'].sum()) if l['pnl'].sum() != 0 else float('inf')
    print(f"  {label}: {len(df)} trades, WR={wr:.1f}%, PNL=${df['pnl'].sum():,.0f}, PF={pf:.2f}")


print("Loading caches...")
t20 = pd.read_csv('data/bull_flag_cache_e50_x30_t20.csv')
t15 = pd.read_csv('data/bull_flag_cache_e50_x30_t15.csv')
t8  = pd.read_csv('data/bull_flag_cache_e50_x30_t8.csv')
for df in (t20, t15, t8):
    df['date'] = pd.to_datetime(df['date']).dt.date

start_dt = t20['date'].min()
end_dt   = t20['date'].max()
t15 = t15[(t15['date'] >= start_dt) & (t15['date'] <= end_dt)].copy()
t8  = t8[(t8['date'] >= start_dt) & (t8['date'] <= end_dt)].copy()

keys_t20 = set(zip(t20['symbol'], t20['date'].astype(str), t20['entry_time_et']))
keys_t15 = set(zip(t15['symbol'], t15['date'].astype(str), t15['entry_time_et']))


def extra_vs_t20(df):
    mask = [(r['symbol'], str(r['date']), r['entry_time_et']) not in keys_t20
            for _, r in df.iterrows()]
    return df[mask]


extra_t15 = extra_vs_t20(t15)
extra_t8  = extra_vs_t20(t8)

conn = sqlite3.connect(DB_PATH)

print("Computing gap pcts for t15 extra...")
extra_t15 = extra_t15.copy()
extra_t15['gap_pct'] = compute_gap_pct_batch(extra_t15, conn)
extra_t15['is_gap_up'] = extra_t15['gap_pct'].apply(lambda g: g is not None and g >= GAP_THRESHOLD)

print("Computing gap pcts for t8 extra...")
extra_t8 = extra_t8.copy()
extra_t8['gap_pct'] = compute_gap_pct_batch(extra_t8, conn)
extra_t8['is_gap_up'] = extra_t8['gap_pct'].apply(lambda g: g is not None and g >= GAP_THRESHOLD)

conn.close()

gap_t15 = extra_t15[extra_t15['is_gap_up']]
gap_t8  = extra_t8[extra_t8['is_gap_up']]
nogap_t15 = extra_t15[~extra_t15['is_gap_up']]
nogap_t8  = extra_t8[~extra_t8['is_gap_up']]

print()
print("=== Gap>=10% extra trades broken down by daily_range_pct ===")
print("t15 EXTRA gap-ups (Path A candidates), by daily_range_pct band:")
for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 200)]:
    band = gap_t15[(gap_t15['daily_range_pct'] >= lo) & (gap_t15['daily_range_pct'] < hi)]
    if not band.empty:
        stats(f"  range {lo}-{hi}%", band)

print()
print("t8 EXTRA gap-ups (Path A candidates), by daily_range_pct band:")
for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 200)]:
    band = gap_t8[(gap_t8['daily_range_pct'] >= lo) & (gap_t8['daily_range_pct'] < hi)]
    if not band.empty:
        stats(f"  range {lo}-{hi}%", band)

print()
print("=== Key insight: t8 EXTRA gap-ups filtered to daily_range >= 20% ===")
gap_t8_high_range = gap_t8[gap_t8['daily_range_pct'] >= 20]
gap_t8_low_range  = gap_t8[gap_t8['daily_range_pct'] < 20]
stats("t8 EXTRA gap-up + daily_range>=20%", gap_t8_high_range)
stats("t8 EXTRA gap-up + daily_range<20%", gap_t8_low_range)

print()
print("=== Dual qual with additional daily_range filter on gap-up extras ===")
# If we add gap-up extras only when daily_range >= 20%
dual_filtered = pd.concat([t20, gap_t8_high_range], ignore_index=True)
stats("Dual qual (t20 + gap-up extra with range>=20%)", dual_filtered)

# Try 30%
gap_t8_30 = gap_t8[gap_t8['daily_range_pct'] >= 30]
dual_30 = pd.concat([t20, gap_t8_30], ignore_index=True)
stats("Dual qual (t20 + gap-up extra with range>=30%)", dual_30)

print()
print("=== Monthly breakdown: t20 vs gap_t8 extra performance ===")
t20['ym'] = t20['date'].apply(lambda d: str(d)[:7])
gap_t8['ym'] = gap_t8['date'].apply(lambda d: str(d)[:7])

months = sorted(set(list(t20['ym'].unique()) + list(gap_t8['ym'].unique())))
print(f"{'Month':<8} {'t20 trades':>10} {'t20 PNL':>12} {'gapup extra':>12} {'gapup PNL':>12}")
for m in months:
    t20m = t20[t20['ym'] == m]
    gm = gap_t8[gap_t8['ym'] == m]
    print(f"{m:<8} {len(t20m):>10} ${t20m['pnl'].sum():>10,.0f} {len(gm):>12} ${gm['pnl'].sum():>10,.0f}")

print()
print("=== How many gap-up extra trades WOULD have hit 15% or 20% intraday? ===")
# These stocks DID gap but their intraday range was < 20% — so current system missed them
gap_t8['range_bucket'] = pd.cut(
    gap_t8['daily_range_pct'],
    bins=[0, 10, 15, 20, 30, 50, 100, 1000],
    labels=['<10%', '10-15%', '15-20%', '20-30%', '30-50%', '50-100%', '>100%']
)
print("t8 EXTRA gap-up trades by intraday range:")
print(gap_t8.groupby('range_bucket', observed=True)['pnl'].agg(['count', 'sum', 'mean']).round(0))
