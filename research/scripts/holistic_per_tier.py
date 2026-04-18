#!/usr/bin/env python3
"""Per-tier realized-R decomposition (Phase A.1 of per-tier optimization).

Splits trades by intraday-change tier:
  - A (A-tier): intraday_change_at_entry >= 20%
  - E (Extras):  10% <= intraday_change_at_entry < 20%
  - edge:       intraday_change_at_entry < 10%   (sanity check only)

For each tier, computes:
  - Realized-R stats (mean, median, std, win_rate)
  - Per-rule β coefficient (realized-R(rule fires) - realized-R(doesn't))
  - Per-MACD-zone edge (1.0× bucket vs 1.5× bucket)
  - Per-bucket pnl_at_1x sum (capital-deployment potential at 1× sizing)

This tells us which rules and MACD zones actually HAVE edge in each tier,
so we can size accordingly in Phase A.2+.

Reads: /tmp/s1_looser/cache_base_{2025,q1}.csv (10%-frame, user's working
threshold).

Output: research/per_tier_decomp.md
"""
from __future__ import annotations

import csv
from collections import defaultdict
from typing import List, Dict, Tuple

# -------------------------------------------------------------------------
# Splits (matches prior audit)
# -------------------------------------------------------------------------

CACHES = {
    'TRAIN (2025 Jan-Jul)': ('/tmp/s1_looser/cache_base_2025.csv', '2025-01-01', '2025-07-31'),
    'VAL (2025 Aug-Dec)':   ('/tmp/s1_looser/cache_base_2025.csv', '2025-08-01', '2025-12-31'),
    'HOQ1 (2026 Q1)':       ('/tmp/s1_looser/cache_base_q1.csv',   '2026-01-01', '2026-03-31'),
}

MIN_CONV = 1.4

# -------------------------------------------------------------------------
# Tier classification (matches trading/two_tier_filter.py::classify_tier)
# -------------------------------------------------------------------------

def classify_tier(ic: float) -> str:
    """Map intraday_change_at_entry % to tier label."""
    if ic >= 20.0:
        return 'A'
    if ic >= 10.0:
        return 'E'
    return 'edge'


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

def _f(x, default=0.0):
    try:
        return float(x) if x not in (None, '', 'None') else default
    except (TypeError, ValueError):
        return default


def load_trades():
    """Load trades tagged with split, tier, and derived features."""
    all_trades = []
    loaded_paths = set()
    for split, (path, start, end) in CACHES.items():
        with open(path) as f:
            for r in csv.DictReader(f):
                if r['date'] < start or r['date'] > end:
                    continue
                try:
                    cm = _f(r.get('conviction_mult'), 1.0) or 1.0
                    if cm < MIN_CONV:
                        continue
                    entry = _f(r.get('entry_price'))
                    stop = _f(r.get('stop_loss'))
                    shares = int(_f(r.get('shares')))
                    risk_ps = entry - stop
                    if shares <= 0 or risk_ps <= 0:
                        continue
                    pnl = _f(r.get('pnl'))
                    macd = _f(r.get('macd_zone_mult'), 1.0) or 1.0
                    realized_R = pnl / (shares * risk_ps)
                    raw = _f(r.get('conv_raw_score'))
                    r1 = _f(r.get('conv_pole_gain'))
                    r2 = _f(r.get('conv_flag_tightness'))
                    r3 = _f(r.get('conv_vol_ratio'))
                    r4 = _f(r.get('conv_spy_regime'))
                    r5 = _f(r.get('conv_retracement'))
                    r7 = _f(r.get('conv_vwap_dist'))
                    r8 = _f(r.get('conv_gap_fading'))
                    r9 = raw - 1.0 - (r1 + r2 + r3 + r4 + r5 + r7 + r8) if raw > 0 else 0.0
                    if abs(r9) < 0.01:
                        r9 = 0.0
                    ic = _f(r.get('intraday_change_at_entry'))
                    all_trades.append({
                        'split': split, 'date': r['date'],
                        'symbol': r.get('symbol', ''),
                        'tier': classify_tier(ic),
                        'intraday_change': ic,
                        'pnl': pnl, 'shares': shares,
                        'risk_per_share': risk_ps,
                        'realized_R': realized_R,
                        'conv_mult': cm, 'macd_mult': macd,
                        'raw_score': raw,
                        'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4,
                        'r5': r5, 'r7': r7, 'r8': r8, 'r9': r9,
                        'pnl_at_1x': pnl / (cm * macd) if (cm * macd) > 0 else pnl,
                    })
                except (KeyError, ValueError):
                    continue
    return all_trades


# -------------------------------------------------------------------------
# Analysis
# -------------------------------------------------------------------------

def summarize(trades):
    n = len(trades)
    if n == 0:
        return {'n': 0, 'wr': 0.0, 'mean_R': 0.0, 'median_R': 0.0,
                'pnl_sum': 0.0, 'pnl_1x_sum': 0.0}
    wins = sum(1 for t in trades if t['pnl'] > 0)
    Rs = sorted(t['realized_R'] for t in trades)
    median = Rs[n // 2]
    return {
        'n': n, 'wr': wins / n * 100,
        'mean_R': sum(Rs) / n, 'median_R': median,
        'pnl_sum': sum(t['pnl'] for t in trades),
        'pnl_1x_sum': sum(t['pnl_at_1x'] for t in trades),
    }


def rule_beta(trades, rule_key: str, pos_branch: bool = True):
    """β = mean_R(rule fires) - mean_R(rule doesn't). Sign-controlled for
    bidirectional rules (e.g., r2 can be pos or neg)."""
    sign = 1 if pos_branch else -1
    fires = [t['realized_R'] for t in trades if (t[rule_key] * sign) > 0]
    nots = [t['realized_R'] for t in trades if (t[rule_key] * sign) <= 0]
    if not fires or not nots:
        return None
    return {
        'n_fires': len(fires),
        'mean_R_fires': sum(fires) / len(fires),
        'mean_R_nots': sum(nots) / len(nots),
        'beta': (sum(fires) / len(fires)) - (sum(nots) / len(nots)),
    }


def main():
    trades = load_trades()
    tiers = ['A', 'E', 'edge']
    splits = ['TRAIN (2025 Jan-Jul)', 'VAL (2025 Aug-Dec)', 'HOQ1 (2026 Q1)']

    print("# Per-tier realized-R decomposition (10%-frame, current shipping config)\n")
    print(f"Total trades loaded (conv>=1.4): {len(trades)}")
    print(f"Source caches: /tmp/s1_looser/cache_base_{{2025,q1}}.csv\n")
    print("Tier classification: A ≥ 20%, E in [10%, 20%), edge < 10%.\n")

    # === Overall per-tier summary ===
    print("## 1. Per-tier summary (all splits combined)\n")
    print("| Tier | n | WR | mean R | median R | Total PnL | pnl@1x |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for tier in tiers:
        ts = [t for t in trades if t['tier'] == tier]
        s = summarize(ts)
        if s['n'] == 0:
            continue
        print(f"| {tier} | {s['n']} | {s['wr']:.1f}% | {s['mean_R']:+.3f}R | "
              f"{s['median_R']:+.3f}R | ${s['pnl_sum']:+,.0f} | ${s['pnl_1x_sum']:+,.0f} |")

    # === Per-tier per-split ===
    print("\n## 2. Per-tier × split breakdown\n")
    print("| Tier | Split | n | WR | mean R | Total PnL |")
    print("|---|---|---:|---:|---:|---:|")
    for tier in tiers:
        for split in splits:
            ts = [t for t in trades if t['tier'] == tier and t['split'] == split]
            s = summarize(ts)
            if s['n'] == 0:
                continue
            print(f"| {tier} | {split.split(' ')[0]} | {s['n']} | "
                  f"{s['wr']:.1f}% | {s['mean_R']:+.3f}R | ${s['pnl_sum']:+,.0f} |")

    # === Per-tier MACD zone breakdown ===
    print("\n## 3. Per-tier × MACD zone (most important table)\n")
    print("MACD 1.0 = normal zone. 1.5 = strong (pos or neg). Current shipping "
          "multiplies strong bucket 1.5× sizing. This table shows whether "
          "the 1.5 bucket actually HAS edge per tier.\n")
    print("| Tier | MACD | n | WR | mean R | Total PnL | pnl@1x |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for tier in tiers:
        for macd_key, macd_filter in [('1.0 (normal)', lambda t: abs(t['macd_mult'] - 1.0) < 0.01),
                                       ('1.5 (strong)', lambda t: abs(t['macd_mult'] - 1.5) < 0.01)]:
            ts = [t for t in trades if t['tier'] == tier and macd_filter(t)]
            s = summarize(ts)
            if s['n'] == 0:
                continue
            print(f"| {tier} | {macd_key} | {s['n']} | {s['wr']:.1f}% | "
                  f"{s['mean_R']:+.3f}R | ${s['pnl_sum']:+,.0f} | ${s['pnl_1x_sum']:+,.0f} |")

    # === Per-tier per-rule β ===
    print("\n## 4. Per-tier rule β (realized-R lift when rule fires vs not)\n")
    print("Positive β = rule has edge IN THAT TIER. Negative β = rule is "
          "noise or counter-signal. Compare magnitudes: if β_A >> β_E, the "
          "rule is A-tier-specific.\n")
    rules = [
        ('r1 (pole gain)', 'r1', True),
        ('r2+ (flag tight)', 'r2', True),
        ('r2- (flag loose)', 'r2', False),
        ('r3 (vol ratio)', 'r3', True),
        ('r4+ (SPY good)', 'r4', True),
        ('r4- (SPY bad)', 'r4', False),
        ('r5 (retracement)', 'r5', True),
        ('r7 (vwap dist)', 'r7', True),
        ('r8 (gap fading)', 'r8', False),
        ('r9 (V-reversal)', 'r9', True),
    ]
    print("| Rule | A β | A n_fires | E β | E n_fires | edge β | edge n |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, key, pos in rules:
        row = [name]
        for tier in tiers:
            ts = [t for t in trades if t['tier'] == tier]
            res = rule_beta(ts, key, pos_branch=pos)
            if res is None:
                row.extend(['n/a', '—'])
            else:
                row.append(f"{res['beta']:+.3f}R")
                row.append(str(res['n_fires']))
        print(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | "
              f"{row[5]} | {row[6]} |")

    # === Per-tier × conviction decile ===
    print("\n## 5. Per-tier conviction-decile breakdown\n")
    print("Shows realized R by conv bucket within each tier. Helps identify "
          "whether high-conv trades generalize across tiers.\n")
    conv_buckets = [('<1.5', 1.4, 1.5), ('1.5-1.8', 1.5, 1.8),
                     ('1.8-2.2', 1.8, 2.2), ('2.2-2.6', 2.2, 2.6),
                     ('≥2.6', 2.6, 99)]
    print("| Tier | ConvBucket | n | WR | mean R | Total PnL |")
    print("|---|---|---:|---:|---:|---:|")
    for tier in tiers:
        for bucket_name, lo, hi in conv_buckets:
            ts = [t for t in trades if t['tier'] == tier
                  and lo <= t['conv_mult'] < hi]
            s = summarize(ts)
            if s['n'] == 0:
                continue
            print(f"| {tier} | {bucket_name} | {s['n']} | {s['wr']:.1f}% | "
                  f"{s['mean_R']:+.3f}R | ${s['pnl_sum']:+,.0f} |")

    # === Per-tier × rule pattern combo (top 15) ===
    print("\n## 6. Top rule patterns per tier\n")
    for tier in tiers:
        print(f"\n### Tier {tier}: Top-10 rule-firing patterns\n")
        ts = [t for t in trades if t['tier'] == tier]
        if not ts:
            continue
        patterns = defaultdict(list)
        for t in ts:
            pat = (
                '+' if t['r1'] > 0 else '0',
                '+' if t['r2'] > 0 else ('-' if t['r2'] < 0 else '0'),
                '+' if t['r3'] > 0 else '0',
                '+' if t['r4'] > 0 else ('-' if t['r4'] < 0 else '0'),
                '+' if t['r5'] > 0 else '0',
                '+' if t['r7'] > 0 else '0',
                '-' if t['r8'] < 0 else '0',
                '+' if t['r9'] > 0 else '0',
            )
            patterns[pat].append(t)
        top = sorted(patterns.items(), key=lambda kv: -len(kv[1]))[:10]
        print("| Pattern (r1,r2,r3,r4,r5,r7,r8,r9) | n | mean R | Total PnL |")
        print("|---|---:|---:|---:|")
        for pat, ts2 in top:
            s = summarize(ts2)
            print(f"| ({','.join(pat)}) | {s['n']} | {s['mean_R']:+.3f}R | "
                  f"${s['pnl_sum']:+,.0f} |")


if __name__ == "__main__":
    main()
