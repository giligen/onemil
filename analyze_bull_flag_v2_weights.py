#!/usr/bin/env python3
"""Phase B — Weight discovery for the 3 candidate V2 conviction rules.

Bucket-analyze each candidate feature on chronological train/test split:
- TRAIN: H1'25 (Jan-Jun 2025)
- TEST:  H2'25 + Q1+Apr'26 (Jul'25 - Apr'26)

For each feature:
1. Sort trades into 4 quartiles by feature value (TRAIN-derived edges)
2. Compute per-bucket WR + avg P&L on TRAIN AND TEST
3. Identify thresholds where EV jumps cleanly (signal of where to put rule cutpoints)

Output: proposed weights for Rule 6 (daily_range), Rule 7 (vwap_dist),
Rule 8 (gap_fading) — to feed into Phase C walk-forward variant study.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/bull_flag_cache_e50_x30.csv')
df['date'] = pd.to_datetime(df['date'])

# Restrict to rows with full per-rule contribs (post-Phase A rebuild)
df = df[df['conv_raw_score'].notna()].reset_index(drop=True)
print(f"Loaded {len(df)} cache rows with full conv breakdown\n")

# Walk-forward split (H1'25 train, rest test)
ts, te = pd.Timestamp('2025-01-01'), pd.Timestamp('2025-06-30')
train = df[(df['date'] >= ts) & (df['date'] <= te)].copy()
test = df[df['date'] > te].copy()
print(f"Train: {len(train)} trades   Test: {len(test)} trades\n")


def bucket_table(d, feature, edges, label):
    """Print per-bucket WR + avg P&L using fixed edges."""
    print(f"  {label} bucket table for {feature}:")
    print(f"  {'bucket':<26} {'n':>4} {'WR':>5} {'avg P&L':>10} {'total':>11}")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        if i == 0:
            sub = d[(d[feature] >= lo) & (d[feature] <= hi)]
        else:
            sub = d[(d[feature] > lo) & (d[feature] <= hi)]
        if len(sub) == 0:
            continue
        wr = (sub['pnl'] > 0).mean() * 100
        avg = sub['pnl'].mean()
        tot = sub['pnl'].sum()
        lo_str = f"{lo:.1f}" if np.isfinite(lo) else "-inf"
        hi_str = f"{hi:.1f}" if np.isfinite(hi) else "+inf"
        print(f"  ({lo_str:>6} .. {hi_str:>6}]  "
              f"{len(sub):>4} {wr:>4.0f}% ${avg:>+9,.0f} ${tot:>+10,.0f}")
    print()


# =============================================================================
# Rule 6 — daily_range_pct (continuous)
# =============================================================================
print("=" * 68)
print("RULE 6: daily_range_pct")
print("=" * 68)

# Train quartile edges, then add explicit cutpoints to test rule shape
train_q = train['daily_range_pct'].quantile([0, 0.25, 0.5, 0.75, 1.0]).tolist()
print(f"Train quartile edges: {[f'{e:.1f}' for e in train_q]}")
edges = [-np.inf, train_q[1], train_q[2], train_q[3], np.inf]
bucket_table(train, 'daily_range_pct', edges, "TRAIN (quartile)")
bucket_table(test, 'daily_range_pct', edges, "TEST (same edges)")

# Test specific candidate cutpoints we'd use for the rule
print("Candidate rule cutpoints (continuous):")
candidate_edges = [-np.inf, 25, 30, 40, 60, np.inf]
bucket_table(train, 'daily_range_pct', candidate_edges, "TRAIN (25/30/40/60)")
bucket_table(test, 'daily_range_pct', candidate_edges, "TEST")


# =============================================================================
# Rule 7 — qf_vwap_dist_pct (continuous, distance from VWAP at fill bar)
# =============================================================================
print("=" * 68)
print("RULE 7: qf_vwap_dist_pct (or qf_fill_vwap_dist_pct)")
print("=" * 68)

# Choose the field with more data
for f in ['qf_vwap_dist_pct', 'qf_fill_vwap_dist_pct']:
    n_nonnull = df[f].notna().sum()
    print(f"  {f}: {n_nonnull}/{len(df)} non-null")
print()

f = 'qf_fill_vwap_dist_pct'  # fill-bar VWAP — more accurate for entry quality
train_f = train[train[f].notna()].copy()
test_f = test[test[f].notna()].copy()

if len(train_f) >= 20:
    train_q = train_f[f].quantile([0, 0.25, 0.5, 0.75, 1.0]).tolist()
    print(f"Train quartile edges for {f}: {[f'{e:.2f}' for e in train_q]}")
    edges = [-np.inf, train_q[1], train_q[2], train_q[3], np.inf]
    bucket_table(train_f, f, edges, "TRAIN (quartile)")
    bucket_table(test_f, f, edges, "TEST (same edges)")

    # Candidate cutpoints based on what looks like the EV inflection point
    print("Candidate cutpoints:")
    candidate_edges = [-np.inf, -2, 0, 2, 5, np.inf]
    bucket_table(train_f, f, candidate_edges, f"TRAIN (-2/0/2/5)")
    bucket_table(test_f, f, candidate_edges, f"TEST")


# =============================================================================
# Rule 8 — qf_gap_fading (boolean: gap reversed pre-entry)
# =============================================================================
print("=" * 68)
print("RULE 8: qf_gap_fading (boolean — gap reversed before entry)")
print("=" * 68)

f = 'qf_gap_fading'
# Coerce to boolean (csv encodes as True/False string)
def _b(x):
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return bool(x)
    if isinstance(x, str): return x.lower() in ('true', '1', 'yes')
    return False

train_b = train.copy()
test_b = test.copy()
train_b[f] = train_b[f].apply(_b)
test_b[f] = test_b[f].apply(_b)

for label, d in [('TRAIN', train_b), ('TEST', test_b)]:
    print(f"  {label}:")
    for v in (False, True):
        sub = d[d[f] == v]
        if len(sub) == 0: continue
        wr = (sub['pnl'] > 0).mean() * 100
        avg = sub['pnl'].mean()
        tot = sub['pnl'].sum()
        print(f"    gap_fading={v!s:<5}  n={len(sub):>3}  WR={wr:>4.0f}%  "
              f"avg ${avg:>+8,.0f}  total ${tot:>+9,.0f}")
    print()


# =============================================================================
# Recommendation — proposed weights
# =============================================================================
print("=" * 68)
print("PROPOSED V2 RULE WEIGHTS")
print("=" * 68)
print("""
Based on bucket analysis above, candidate weights for Phase C walk-forward:

Rule 6 — daily_range_pct (continuous):
  +0.3 if daily_range_pct >= 30%
  +0.1 if 25% <= daily_range_pct < 30%
  +0.0 if daily_range_pct < 25%

Rule 7 — qf_fill_vwap_dist_pct (continuous):
  Inspect bucket table above to pick X, Y; tentative:
  +0.2 if qf_fill_vwap_dist_pct >= X (top quartile)
  +0.1 if qf_fill_vwap_dist_pct >= Y (above median)
  +0.0 below

Rule 8 — qf_gap_fading (binary):
  -0.3 if qf_gap_fading == True (gap reversed pre-entry)
  +0.0 otherwise

Score range with all 3 added:
  Old max: 1.0 + 0.3+0.3+0.3+0.3+0.2 = +1.4  → score 2.4
  New max: + 0.3+0.2+0.0 = +0.5 more       → score 2.9 (still under 3.0 clamp)
  Old min: 1.0 - 0.3 - 0.5 = 0.2  → clamped 0.25
  New min: + (-0.3 from gap_fading) = -0.1 → still clamped 0.25
""")
