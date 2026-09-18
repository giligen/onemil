#!/usr/bin/env python3
"""BF decay part 3: population drift into the three P1-gated buckets, R-column
cross-check against the independent regen-7 feature file, and the C1/C2 gate test."""
import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil')
OUT = '/home/ec2-user/onemil/research/bf_decay'
def log(*a): print(*a); sys.stdout.flush()

df = pd.read_csv('/home/ec2-user/onemil/data/bull_flag_cache_causal_full_20260905.csv',
                 keep_default_na=False, na_values=[''], dtype={'symbol': str})
df = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2026-08-31')].copy()
df['year'] = df['date'].str[:4]
for c in ['entry_price','stop_loss','pnl','shares','qf_pole_gain_pct','qf_vwap_dist_pct',
          'conviction_mult','macd_zone_mult','avg_volume_20d','qf_pole_bars']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['R'] = df['pnl']/((df['entry_price']-df['stop_loss'])*df['shares'])
df = df[np.isfinite(df['R'])].copy()

# ---- cross-check my R against the independently built feature file's R
fe = pd.read_csv(f'/home/ec2-user/onemil/research/bf_consistency/trade_features_regen7.csv',
                 keep_default_na=False, na_values=[''], dtype={'symbol': str})
m = df.merge(fe[['symbol','date','entry_time_et','R']], on=['symbol','date','entry_time_et'],
             suffixes=('','_fe'), how='inner')
d = (m['R']-m['R_fe']).abs()
log(f"R cross-check vs trade_features_regen7: matched {len(m)} rows, "
    f"max|diff|={d.max():.6f}, median|diff|={d.median():.6f}, "
    f">1e-6: {(d>1e-6).sum()}")

# ---- population drift into the three P1-gated buckets
log("\n=== POPULATION DRIFT (share of raw detections in each P1-gated bucket) ===")
buckets = {
    'entry price > $20':        df['entry_price'] > 20,
    'pole gain < 5%':           df['qf_pole_gain_pct'] < 5.0,
    'breakout at/below VWAP':   df['qf_vwap_dist_pct'] <= 0,
    'ANY of the three':         (df['entry_price']>20)|(df['qf_pole_gain_pct']<5.0)|(df['qf_vwap_dist_pct']<=0),
    'conviction < 1.8':         df['conviction_mult'].fillna(1.0) < 1.8,
    'ADV20 < 200K':             df['avg_volume_20d'].fillna(0) < 200000,
}
rows = []
for k, mask in buckets.items():
    a = df[df.year=='2025']; b = df[df.year=='2026']
    ma, mb = mask[df.year=='2025'], mask[df.year=='2026']
    # 2-proportion z
    p1_, p2_ = ma.mean(), mb.mean(); n1, n2 = len(a), len(b)
    pp = (ma.sum()+mb.sum())/(n1+n2)
    z = (p2_-p1_)/math.sqrt(pp*(1-pp)*(1/n1+1/n2))
    rows.append(dict(bucket=k, share25=p1_*100, share26=p2_*100, dpp=(p2_-p1_)*100, z=z,
                     R25_in=a[ma]['R'].mean(), R26_in=b[mb]['R'].mean()))
t = pd.DataFrame(rows)
log(t.to_string(index=False, float_format=lambda x: f'{x:7.2f}'))
t.to_csv(f'{OUT}/population_drift.csv', index=False)

log("\n=== DETECTION RATE ===")
for y, mo in (('2025',12), ('2026',8)):
    s = df[df.year==y]
    log(f"{y}: {len(s)} detections / {mo} months = {len(s)/mo:.1f} per month; "
        f"distinct symbols {s.symbol.nunique()}, distinct days {s.date.nunique()}")

log("\n=== quartile medians of the population ===")
for c in ['entry_price','qf_pole_gain_pct','qf_vwap_dist_pct','avg_volume_20d','conviction_mult']:
    a = df[df.year=='2025'][c].dropna(); b = df[df.year=='2026'][c].dropna()
    log(f"{c:22s} 2025 p25/p50/p75 = {a.quantile(.25):9.2f}/{a.median():9.2f}/{a.quantile(.75):9.2f} | "
        f"2026 = {b.quantile(.25):9.2f}/{b.median():9.2f}/{b.quantile(.75):9.2f}")

# ---- would a C1/C2-day gate on P1 have helped? (pre-registration candidate only)
p1 = pd.read_csv(f'{OUT}/p1_picks.csv', dtype={'symbol': str})
log("\n=== C1/C2 gate counterfactual on P1 picks (NOT a proposal — power check) ===")
for label, mask in [('A only', p1.regime=='A'), ('A+B', p1.regime.isin(['A','B'])),
                    ('all (as-is)', p1.regime.notna())]:
    s = p1[mask]
    for y in ('2025','2026'):
        v = s[s.date.str[:4]==y]
        log(f"  {label:12s} {y}: n={len(v):3d} R={v['R'].mean():+.3f} pnl=${v['pnl'].sum():>10,.0f}")
    log(f"  {label:12s} TOT : n={len(s):3d} R={s['R'].mean():+.3f} pnl=${s['pnl'].sum():>10,.0f}")
# power on the C1 2025-vs-2026 cell
c1a = p1[(p1.regime=='C1')&(p1.date.str[:4]=='2025')]['R'].values
c1b = p1[(p1.regime=='C1')&(p1.date.str[:4]=='2026')]['R'].values
se = math.sqrt(c1a.var(ddof=1)/len(c1a)+c1b.var(ddof=1)/len(c1b))
log(f"  C1 cell: 2025 n={len(c1a)} R={c1a.mean():+.3f} | 2026 n={len(c1b)} R={c1b.mean():+.3f} "
    f"| d={c1b.mean()-c1a.mean():+.3f} se={se:.3f} t={(c1b.mean()-c1a.mean())/se:+.2f}")
