"""
Additional analysis: gap distribution in t20 and volume filter effects.
"""
import pandas as pd

# Load caches
t20 = pd.read_csv('data/bull_flag_cache_e50_x30_t20.csv')
t15 = pd.read_csv('data/bull_flag_cache_e50_x30_t15.csv')
t8  = pd.read_csv('data/bull_flag_cache_e50_x30_t8.csv')

# Normalize dates
t20['date'] = pd.to_datetime(t20['date']).dt.date
t15['date'] = pd.to_datetime(t15['date']).dt.date
t8['date']  = pd.to_datetime(t8['date']).dt.date

start_dt = t20['date'].min()
end_dt   = t20['date'].max()
t15 = t15[(t15['date'] >= start_dt) & (t15['date'] <= end_dt)].copy()
t8  = t8[(t8['date'] >= start_dt) & (t8['date'] <= end_dt)].copy()


def stats(label, df):
    if df.empty:
        print(f"  {label}: empty")
        return
    w = df[df['pnl'] > 0]
    l = df[df['pnl'] <= 0]
    wr = len(w) / len(df) * 100
    pf = abs(w['pnl'].sum() / l['pnl'].sum()) if l['pnl'].sum() != 0 else float('inf')
    print(f"  {label}: {len(df)} trades, WR={wr:.1f}%, PNL=${df['pnl'].sum():,.0f}, PF={pf:.2f}")


print("=== Volume filter effect on t20 baseline ===")
pass_vol = t20[t20['avg_volume_20d'] >= 200000]
fail_vol = t20[t20['avg_volume_20d'] < 200000]
stats("t20 ALL  (no vol filter)", t20)
stats("t20 pass 200K vol filter", pass_vol)
stats("t20 fail 200K vol filter", fail_vol)

print()
print("=== t20 gap distribution (qf_gap_pct column) ===")
print(t20['qf_gap_pct'].describe())
gap_t20    = t20[t20['qf_gap_pct'] >= 10]
nogap_t20  = t20[t20['qf_gap_pct'].fillna(0) < 10]
na_t20     = t20[t20['qf_gap_pct'].isna()]
stats("t20 gap>=10%", gap_t20)
stats("t20 gap<10% (incl NaN)", nogap_t20)
stats("t20 gap=NaN only", na_t20)

print()
print("=== t15 and t8 extra trades daily_range_pct ===")
keys_t20 = set(zip(t20['symbol'], t20['date'].astype(str), t20['entry_time_et']))


def extra_vs_t20(df):
    mask = [(r['symbol'], str(r['date']), r['entry_time_et']) not in keys_t20
            for _, r in df.iterrows()]
    return df[mask]


extra_t15 = extra_vs_t20(t15)
extra_t8  = extra_vs_t20(t8)

print("t15 EXTRA daily_range_pct stats:")
print(extra_t15['daily_range_pct'].describe())
print()
print("t8 EXTRA daily_range_pct stats:")
print(extra_t8['daily_range_pct'].describe())

print()
print("=== t15 EXTRA: breakdown by daily_range_pct bands ===")
for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 200)]:
    band = extra_t15[(extra_t15['daily_range_pct'] >= lo) & (extra_t15['daily_range_pct'] < hi)]
    if not band.empty:
        stats(f"  range {lo}-{hi}%", band)

print()
print("=== t8 EXTRA: breakdown by daily_range_pct bands ===")
for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 200)]:
    band = extra_t8[(extra_t8['daily_range_pct'] >= lo) & (extra_t8['daily_range_pct'] < hi)]
    if not band.empty:
        stats(f"  range {lo}-{hi}%", band)

print()
print("=== avg_volume_20d on extra trades ===")
print("t15 EXTRA avg_volume_20d:")
print(extra_t15['avg_volume_20d'].describe())
pass_vol_t15 = extra_t15[extra_t15['avg_volume_20d'] >= 200000]
fail_vol_t15 = extra_t15[extra_t15['avg_volume_20d'] < 200000]
stats("t15 EXTRA pass 200K vol", pass_vol_t15)
stats("t15 EXTRA fail 200K vol", fail_vol_t15)

print()
print("t8 EXTRA avg_volume_20d:")
print(extra_t8['avg_volume_20d'].describe())
pass_vol_t8 = extra_t8[extra_t8['avg_volume_20d'] >= 200000]
fail_vol_t8 = extra_t8[extra_t8['avg_volume_20d'] < 200000]
stats("t8 EXTRA pass 200K vol", pass_vol_t8)
stats("t8 EXTRA fail 200K vol", fail_vol_t8)
