#!/usr/bin/env python3
"""Holistic multi-layer multiplier optimizer.

Zooms out from per-rule audit to the full scaling stack. Decomposes every
cached trade to its realized R-multiple (intrinsic edge, multiplier-invariant)
and pnl_at_1x (pnl normalized to 1x sizing). Then:

  Phase 1 — per-feature realized-R tables (rule firings, price/vol/MACD buckets,
            entry time, spy regime)
  Phase 2 — OLS-style rule β coefficients on realized_R, cross-split stability
  Phase 3 — 4D interaction matrix: (conv_bucket × macd × tier × time)
  Phase 4 — joint grid search over (rule weights, threshold, tier table,
            MACD zone mults, cap) on TRAIN+VAL only, rank by P&L, constrain
            trade count
  Phase 5 — one-shot HOLDOUT validation for top candidates

Splits match the audit: TRAIN 2025 Jan-Jul, VAL 2025 Aug-Dec, HOLDOUT Q1 +
April 1-17 2026.

All read-only: operates on cache CSVs; no config changes.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

SPLITS = [
    ('TRAIN', '/tmp/expVH_final/cache_V_2025.csv', '2025-01-01', '2025-07-31'),
    ('VAL',   '/tmp/expVH_final/cache_V_2025.csv', '2025-08-01', '2025-12-31'),
    ('HOQ1',  '/tmp/expVH_final/cache_V_q1.csv',   '2026-01-01', '2026-03-31'),
    ('HOAPR', '/tmp/april_cache/cache.csv',        '2026-04-01', '2026-04-17'),
]


def _f(x, default=0.0):
    try:
        return float(x) if x not in (None, '', 'None') else default
    except (TypeError, ValueError):
        return default


def _b(x):
    return str(x).lower() == 'true'


def price_bucket(p):
    if p < 5:   return 'P<5'
    if p < 10:  return 'P5-10'
    if p < 15:  return 'P10-15'
    if p < 23:  return 'P15-23'
    return 'P23+'


def vol_bucket(v):
    if v < 500_000:     return 'V<500K'
    if v <= 5_000_000:  return 'V500K-5M'
    return 'V5M+'


def time_bucket(t):
    if not t:
        return 'T?'
    try:
        h, m = int(t[:2]), int(t[3:5])
    except (ValueError, IndexError):
        return 'T?'
    mins = h * 60 + m
    if mins < 600:       return 'T09:30-10:00'
    if mins < 645:       return 'T10:00-10:45'
    if mins < 720:       return 'T10:45-12:00'
    return 'T12:00+'


def macd_bucket(m):
    if abs(m - 1.0) < 0.01:  return 'M1.0'
    if abs(m - 1.5) < 0.01:  return 'M1.5'
    return f'M{m:.2f}'


@dataclass
class Trade:
    split: str
    date: str
    symbol: str
    entry_price: float
    avg_vol: float
    shares: int
    risk_per_share: float
    pnl: float
    realized_R: float
    pnl_at_1x: float
    conv_mult: float
    macd_mult: float
    mult_cached: float
    raw_score: float
    # Rule contributions (signed values as stored in cache):
    r1: float = 0.0  # pole_gain
    r2: float = 0.0  # flag_tightness
    r3: float = 0.0  # vol_ratio
    r4: float = 0.0  # spy_regime
    r5: float = 0.0  # retracement
    r7: float = 0.0  # vwap_dist
    r8: float = 0.0  # gap_fading
    r9: float = 0.0  # v_reversal (derived from raw_score residual)
    # Feature values for custom scoring:
    pole_gain_pct: float = 0.0
    gap_pct: float = 0.0
    intraday_chg: float = 0.0
    qf_vwap_dist: float = 0.0
    qf_gap_fading: bool = False
    spy_3d: float = 0.0
    flag_tightness_pct: float = 0.0  # not in cache, leave 0
    vol_ratio: float = 0.0  # not directly in cache
    retracement_pct: float = 0.0  # not in cache
    entry_time: str = ''


def load_all() -> List[Trade]:
    out = []
    for split, path, start, end in SPLITS:
        with open(path) as f:
            for row in csv.DictReader(f):
                d = row['date']
                if d < start or d > end:
                    continue
                entry = _f(row.get('entry_price'))
                stop = _f(row.get('stop_loss'))
                shares = int(_f(row.get('shares')))
                pnl = _f(row.get('pnl'))
                risk_ps = entry - stop
                if shares <= 0 or risk_ps <= 0:
                    continue
                conv = _f(row.get('conviction_mult'), 1.0) or 1.0
                macd = _f(row.get('macd_zone_mult'), 1.0) or 1.0
                mult_cached = conv * macd
                pnl_at_1x = pnl / mult_cached if mult_cached > 0 else pnl
                realized_R = pnl / (shares * risk_ps) if shares * risk_ps > 0 else 0.0
                raw = _f(row.get('conv_raw_score'))
                r1 = _f(row.get('conv_pole_gain'))
                r2 = _f(row.get('conv_flag_tightness'))
                r3 = _f(row.get('conv_vol_ratio'))
                r4 = _f(row.get('conv_spy_regime'))
                r5 = _f(row.get('conv_retracement'))
                r7 = _f(row.get('conv_vwap_dist'))
                r8 = _f(row.get('conv_gap_fading'))
                # Rule 9 derived as residual of raw_score from base + explicit rules.
                r9 = raw - 1.0 - (r1 + r2 + r3 + r4 + r5 + r7 + r8) if raw > 0 else 0.0
                # Numerical noise → snap near-0 to 0
                if abs(r9) < 0.01:
                    r9 = 0.0
                out.append(Trade(
                    split=split, date=d, symbol=row.get('symbol', ''),
                    entry_price=entry, avg_vol=_f(row.get('avg_volume_20d')),
                    shares=shares, risk_per_share=risk_ps,
                    pnl=pnl, realized_R=realized_R, pnl_at_1x=pnl_at_1x,
                    conv_mult=conv, macd_mult=macd, mult_cached=mult_cached,
                    raw_score=raw,
                    r1=r1, r2=r2, r3=r3, r4=r4, r5=r5, r7=r7, r8=r8, r9=r9,
                    pole_gain_pct=_f(row.get('qf_pole_gain_pct')),
                    gap_pct=_f(row.get('qf_gap_pct')),
                    intraday_chg=_f(row.get('intraday_change_at_entry')),
                    qf_vwap_dist=_f(row.get('qf_vwap_dist_pct')),
                    qf_gap_fading=_b(row.get('qf_gap_fading')),
                    spy_3d=_f(row.get('spy_3d_range')),
                    entry_time=row.get('entry_time_et', ''),
                ))
    return out


# -------------------------------------------------------------------------
# Phase 1 — per-feature realized-R tables
# -------------------------------------------------------------------------

def group_stats(trades: List[Trade], key_fn):
    """Return {group_value: {n, mean_R, median_R, pnl_at_1x_sum, win_rate}}."""
    groups = defaultdict(list)
    for t in trades:
        groups[key_fn(t)].append(t)
    out = {}
    for k, ts in groups.items():
        if not ts:
            continue
        Rs = [t.realized_R for t in ts]
        pnls = [t.pnl for t in ts]
        p1x = [t.pnl_at_1x for t in ts]
        wins = sum(1 for t in ts if t.pnl > 0)
        Rs_sorted = sorted(Rs)
        median = Rs_sorted[len(Rs) // 2]
        out[k] = {
            'n': len(ts),
            'mean_R': sum(Rs) / len(Rs),
            'median_R': median,
            'pnl_sum': sum(pnls),
            'pnl_1x_sum': sum(p1x),
            'win_rate': wins / len(ts) * 100,
            'std_R': (sum((r - sum(Rs)/len(Rs))**2 for r in Rs) / max(1, len(Rs)-1))**0.5,
        }
    return out


def print_group_table(title: str, stats: Dict, sort_by: str = 'n',
                      cols=('n', 'mean_R', 'median_R', 'win_rate', 'pnl_1x_sum')):
    print(f"### {title}\n")
    hdr_cells = ['Bucket'] + list(cols)
    print('| ' + ' | '.join(hdr_cells) + ' |')
    print('|' + '|'.join(['---'] + [':---:'] * len(cols)) + '|')
    for k in sorted(stats.keys(), key=lambda x: -stats[x].get(sort_by, 0)):
        s = stats[k]
        row = [str(k)]
        for c in cols:
            v = s[c]
            if c == 'n':
                row.append(str(v))
            elif 'pnl' in c:
                row.append(f"${v:+,.0f}")
            elif 'rate' in c:
                row.append(f"{v:.1f}%")
            else:
                row.append(f"{v:+.3f}R")
        print('| ' + ' | '.join(row) + ' |')
    print()


def phase1(trades: List[Trade]):
    print("# Phase 1 — per-feature realized-R decomposition\n")
    print(f"Total trades loaded: {len(trades)}. Splits: "
          + ', '.join(f"{s}={sum(1 for t in trades if t.split == s)}"
                      for s in ['TRAIN', 'VAL', 'HOQ1', 'HOAPR']))
    print()
    # Rule firings (positive branch vs doesn't)
    print("## Per-rule firing vs realized R (all splits, current weights)\n")
    rules = [('r1 pole_gain', lambda t: t.r1 > 0),
             ('r2+ flag tight', lambda t: t.r2 > 0),
             ('r2- flag loose', lambda t: t.r2 < 0),
             ('r3 vol_ratio', lambda t: t.r3 > 0),
             ('r4+ spy pos', lambda t: t.r4 > 0),
             ('r4- spy neg', lambda t: t.r4 < 0),
             ('r5 retracement', lambda t: t.r5 > 0),
             ('r7 vwap_dist', lambda t: t.r7 > 0),
             ('r8 gap_fading', lambda t: t.r8 < 0),
             ('r9 v_reversal', lambda t: t.r9 > 0)]
    print("| Rule | Split | n_fires | R_fires | R_not | R_lift | sign stable? |")
    print("|---|---|---:|---:|---:|---:|---|")
    for rule_name, pred in rules:
        split_signs = []
        for split in ['TRAIN', 'VAL', 'HOQ1', 'HOAPR']:
            ts = [t for t in trades if t.split == split]
            fires = [t for t in ts if pred(t)]
            nots = [t for t in ts if not pred(t)]
            if not fires or not nots:
                split_signs.append(None)
                continue
            mf = sum(t.realized_R for t in fires) / len(fires)
            mn = sum(t.realized_R for t in nots) / len(nots)
            lift = mf - mn
            split_signs.append(1 if lift > 0 else -1)
            print(f"| {rule_name} | {split} | {len(fires)} | "
                  f"{mf:+.3f}R | {mn:+.3f}R | {lift:+.3f}R | |")
        signs = [s for s in split_signs if s is not None]
        consistency = "✓" if signs and all(s == signs[0] for s in signs) else "✗"
        print(f"| {rule_name} | **AGG** | | | | | **{consistency}** |")
    print()

    # Price x Vol buckets
    print("## Price × Vol bucket realized-R (all splits)\n")
    bkt_stats = group_stats(trades, lambda t: (price_bucket(t.entry_price), vol_bucket(t.avg_vol)))
    print_group_table("Price-Vol", bkt_stats, sort_by='n')

    # MACD bucket
    print("## MACD zone realized-R (all splits)\n")
    print_group_table("MACD", group_stats(trades, lambda t: macd_bucket(t.macd_mult)))

    # Entry time bucket
    print("## Entry-time bucket realized-R (all splits)\n")
    print_group_table("Time", group_stats(trades, lambda t: time_bucket(t.entry_time)))

    # Conviction bucket
    def conv_bkt(t):
        c = t.conv_mult
        if c < 1.2: return 'C<1.2'
        if c < 1.5: return 'C1.2-1.5'
        if c < 1.8: return 'C1.5-1.8'
        if c < 2.2: return 'C1.8-2.2'
        if c < 2.6: return 'C2.2-2.6'
        return 'C2.6+'
    print("## Conviction decile realized-R\n")
    print_group_table("ConvDecile", group_stats(trades, conv_bkt))


# -------------------------------------------------------------------------
# Phase 2 — Rule β regression (OLS approx)
# -------------------------------------------------------------------------

def simple_ols_beta(trades: List[Trade], rule_idx_fn, intercept_shift=0.0) -> Tuple[float, float]:
    """Simple 1-rule OLS: realized_R = β_0 + β_1 * is_rule_fires.

    Returns (β_1, |β_1| p-value proxy from t-stat). Uses simple pooled variance
    since stdlib only.
    """
    fires = [t.realized_R for t in trades if rule_idx_fn(t) > 0]
    nots = [t.realized_R for t in trades if rule_idx_fn(t) <= 0]
    if not fires or not nots:
        return 0.0, 1.0
    n1, n0 = len(fires), len(nots)
    m1 = sum(fires) / n1
    m0 = sum(nots) / n0
    beta = m1 - m0
    var1 = sum((r - m1)**2 for r in fires) / max(1, n1 - 1)
    var0 = sum((r - m0)**2 for r in nots) / max(1, n0 - 1)
    se = ((var1 / n1) + (var0 / n0))**0.5
    t_stat = beta / se if se > 0 else 0.0
    # crude normal-approx p-value (two-sided): exp(-t^2 / 2) ish; just return |t|
    return beta, t_stat


def phase2(trades: List[Trade]):
    print("# Phase 2 — rule β coefficients on realized_R\n")
    print("β = mean_R(rule fires) - mean_R(rule doesn't fire). Split-wise.\n")
    print("Compare to the **current weight** column: if β has OPPOSITE sign or "
          "very different magnitude than the weight, that rule is mis-calibrated.\n")
    rules = [
        ('r1 pole_gain', 0.30,  lambda t: t.r1 > 0),
        ('r2+ tight', 0.30,     lambda t: t.r2 > 0),
        ('r2- loose', -0.30,    lambda t: t.r2 < 0),  # note: pred = "rule fires negatively"
        ('r3 vol_ratio', 0.30,  lambda t: t.r3 > 0),
        ('r4+ spy good', 0.30,  lambda t: t.r4 > 0),
        ('r4- spy bad', -0.50,  lambda t: t.r4 < 0),
        ('r5 retrace', 0.20,    lambda t: t.r5 > 0),
        ('r7 vwap_dist', 0.20,  lambda t: t.r7 > 0),
        ('r8 gap_fading', -0.30, lambda t: t.r8 < 0),
        ('r9 v_reversal', 0.40,  lambda t: t.r9 > 0),
    ]
    print("| Rule | weight | β_TRAIN | β_VAL | β_HOQ1 | β_HOAPR | β_ALL | sign stable? |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for name, w, pred in rules:
        betas = {}
        for split in ['TRAIN', 'VAL', 'HOQ1', 'HOAPR']:
            ts = [t for t in trades if t.split == split]
            b, t = simple_ols_beta(ts, pred)
            betas[split] = b
        b_all, _ = simple_ols_beta(trades, pred)
        # Sign consistency (excluding zero splits):
        signs = [(1 if b > 0 else -1) for b in betas.values() if abs(b) > 1e-9]
        consistency = '✓ all' if len(signs) >= 3 and all(s == signs[0] for s in signs) else '✗'
        print(f"| {name} | {w:+.2f} | {betas['TRAIN']:+.3f}R | "
              f"{betas['VAL']:+.3f}R | {betas['HOQ1']:+.3f}R | "
              f"{betas['HOAPR']:+.3f}R | {b_all:+.3f}R | {consistency} |")

    # Interpret β as "right weight"
    print("\n## Suggested weight from β (normalized to β_max ≈ 0.3 conviction contribution)\n")
    print("Rule of thumb: contribution_weight ≈ β_median × k, where k is chosen so the "
          "largest positive β maps to +0.3 (matching current rule 4 magnitude).\n")
    # Compute median-β across splits
    suggested = {}
    for name, w, pred in rules:
        bs = []
        for split in ['TRAIN', 'VAL', 'HOQ1', 'HOAPR']:
            ts = [t for t in trades if t.split == split]
            b, _ = simple_ols_beta(ts, pred)
            if abs(b) > 1e-9:
                bs.append(b)
        median_b = 0.0 if not bs else sorted(bs)[len(bs) // 2]
        suggested[name] = (w, median_b)

    # Scale so largest |median_b| corresponds to 0.3
    max_b = max(abs(x[1]) for x in suggested.values())
    k = 0.3 / max_b if max_b > 0 else 1.0
    print("| Rule | current | β_median | suggested | Δ |")
    print("|---|---:|---:|---:|---:|")
    for name, (w, b) in suggested.items():
        sug = round(b * k, 2)
        delta = sug - w
        print(f"| {name} | {w:+.2f} | {b:+.3f}R | {sug:+.2f} | {delta:+.2f} |")


# -------------------------------------------------------------------------
# Phase 3 — multi-layer interaction matrix
# -------------------------------------------------------------------------

def phase3(trades: List[Trade]):
    print("\n# Phase 3 — multi-layer interaction matrix\n")
    print("For each (price × MACD × time) cell, show sample count, realized R, "
          "current avg multiplier, and capital misallocation score.\n")
    print("**Misallocation score** = (avg_R - overall_median_R) * n_trades. "
          "Positive → bucket is under-sized (should get MORE). "
          "Negative → over-sized (should get LESS).\n")

    overall_median_R = sorted([t.realized_R for t in trades])[len(trades) // 2]
    print(f"Overall median realized_R across all trades: {overall_median_R:+.3f}R\n")

    cells = defaultdict(list)
    for t in trades:
        key = (price_bucket(t.entry_price), macd_bucket(t.macd_mult),
               time_bucket(t.entry_time))
        cells[key].append(t)

    print("| Price × MACD × Time | n | avg_R | avg_mult_applied | realized_1x_pnl | misalloc score |")
    print("|---|---:|---:|---:|---:|---:|")
    ranked = []
    for key, ts in cells.items():
        if len(ts) < 10:
            continue
        avg_R = sum(t.realized_R for t in ts) / len(ts)
        avg_mult = sum(t.mult_cached for t in ts) / len(ts)
        pnl_1x_sum = sum(t.pnl_at_1x for t in ts)
        score = (avg_R - overall_median_R) * len(ts)
        ranked.append((key, len(ts), avg_R, avg_mult, pnl_1x_sum, score))
    ranked.sort(key=lambda x: -abs(x[5]))
    for key, n, avg_R, avg_mult, pnl_1x, score in ranked[:30]:
        print(f"| {' × '.join(key)} | {n} | {avg_R:+.3f}R | "
              f"{avg_mult:.2f}× | ${pnl_1x:+,.0f} | {score:+.1f} |")

    # Rule combo frequency + R
    print("\n## Top 15 rule-firing patterns (by count) with realized R\n")
    print("| Pattern (r1,r2,r3,r4,r5,r7,r8,r9) | n | avg_R | total pnl_1x |")
    print("|---|---:|---:|---:|")
    combo_cells = defaultdict(list)
    for t in trades:
        pat = (
            '+' if t.r1 > 0 else '0',
            '+' if t.r2 > 0 else ('-' if t.r2 < 0 else '0'),
            '+' if t.r3 > 0 else '0',
            '+' if t.r4 > 0 else ('-' if t.r4 < 0 else '0'),
            '+' if t.r5 > 0 else '0',
            '+' if t.r7 > 0 else '0',
            '-' if t.r8 < 0 else '0',
            '+' if t.r9 > 0 else '0',
        )
        combo_cells[pat].append(t)
    for pat, ts in sorted(combo_cells.items(), key=lambda x: -len(x[1]))[:15]:
        avg_R = sum(t.realized_R for t in ts) / len(ts)
        pnl1x = sum(t.pnl_at_1x for t in ts)
        print(f"| ({','.join(pat)}) | {len(ts)} | {avg_R:+.3f}R | ${pnl1x:+,.0f} |")


# -------------------------------------------------------------------------
# Phase 4 — joint parameter grid search
# -------------------------------------------------------------------------

@dataclass
class Params:
    """Candidate multiplier configuration — post-hoc recomputable."""
    # Rule weights (override baseline):
    w_r1: float = 0.3
    w_r2p: float = 0.3
    w_r2n: float = -0.3
    w_r3: float = 0.3
    w_r4p: float = 0.3
    w_r4n: float = -0.5
    w_r5: float = 0.2
    w_r7: float = 0.2
    w_r8: float = -0.3
    w_r9: float = 0.4
    # Threshold:
    min_threshold: float = 1.4
    # Cap:
    cap: float = 3.0
    # Tier table (list of dicts):
    tiers: list = field(default_factory=lambda: [
        {'name': 'T1', 'p_min': 10, 'p_max': 15, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
        {'name': 'T2', 'p_min': 15, 'p_max': 23, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
    ])
    # MACD zone multipliers:
    macd_normal: float = 1.0  # currently applied to macd_mult=1.0 bucket
    macd_strong: float = 1.5
    # Clamp range:
    clamp_lo: float = 0.25
    clamp_hi: float = 3.0


def recompute_conv(t: Trade, p: Params) -> float:
    """Recompute conviction_mult under alternate weights. Rule firing patterns
    are inherited from cache; weights are substituted."""
    raw = 1.0
    # r1: fired if cached r1 > 0
    if t.r1 > 0: raw += p.w_r1
    if t.r2 > 0: raw += p.w_r2p
    elif t.r2 < 0: raw += p.w_r2n
    if t.r3 > 0: raw += p.w_r3
    if t.r4 > 0: raw += p.w_r4p
    elif t.r4 < 0: raw += p.w_r4n
    if t.r5 > 0: raw += p.w_r5
    if t.r7 > 0: raw += p.w_r7
    if t.r8 < 0: raw += p.w_r8
    if t.r9 > 0: raw += p.w_r9
    return max(p.clamp_lo, min(p.clamp_hi, raw))


def compute_total_mult(t: Trade, p: Params) -> Tuple[float, bool]:
    """Return (total_multiplier, kept?). kept=False means threshold filter
    drops this trade. Mirrors the shipping BT stacking: conv×tier capped at
    cap; if no tier, apply MACD zone mult (using remapped macd_normal/strong)."""
    conv = recompute_conv(t, p)
    if conv < p.min_threshold:
        return 0.0, False
    tier_mult = 1.0
    matched = False
    for tier in p.tiers:
        if (tier['p_min'] <= t.entry_price < tier['p_max']
                and tier['v_min'] <= t.avg_vol <= tier['v_max']):
            tier_mult = tier['mult']
            matched = True
            break
    # Apply MACD when tier <= 1.0
    if matched and tier_mult > 1.0:
        total = min(p.cap, conv * tier_mult)
    else:
        # MACD zone — remap cached macd_mult (1.0 or 1.5) to params
        if abs(t.macd_mult - 1.5) < 0.01:
            macd = p.macd_strong
        else:
            macd = p.macd_normal
        total = min(p.cap, conv * tier_mult) * macd
    return total, True


def simulate(trades: List[Trade], p: Params) -> Dict:
    """Run a parameter tuple across trades; return aggregated stats per split."""
    by_split = defaultdict(list)
    for t in trades:
        tot, kept = compute_total_mult(t, p)
        if not kept:
            continue
        new_pnl = t.pnl_at_1x * tot
        by_split[t.split].append({'date': t.date, 'pnl': new_pnl, 'is_win': new_pnl > 0})
    out = {}
    for split in ['TRAIN', 'VAL', 'HOQ1', 'HOAPR']:
        ts = by_split[split]
        n = len(ts)
        if n == 0:
            out[split] = {'n': 0, 'pnl': 0.0, 'wr': 0.0, 'maxdd': 0.0}
            continue
        total = sum(t['pnl'] for t in ts)
        wins = sum(1 for t in ts if t['is_win'])
        running, peak, dd = 0.0, 0.0, 0.0
        for t in sorted(ts, key=lambda x: x['date']):
            running += t['pnl']
            peak = max(peak, running)
            dd = min(dd, running - peak)
        out[split] = {'n': n, 'pnl': total, 'wr': wins / n * 100, 'maxdd': dd}
    return out


def baseline_stats(trades) -> Dict:
    """Baseline = cached pnl (current shipping stack)."""
    by_split = defaultdict(list)
    for t in trades:
        if t.conv_mult < 1.4:  # current threshold
            continue
        # Apply tier post-hoc (mirrors batch_backtest.py):
        # current tiers:
        tiers = [
            {'p_min': 10, 'p_max': 15, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
            {'p_min': 15, 'p_max': 23, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
        ]
        conv, macd = t.conv_mult, t.macd_mult
        pnl = t.pnl
        for tier in tiers:
            if (tier['p_min'] <= t.entry_price < tier['p_max']
                    and tier['v_min'] <= t.avg_vol <= tier['v_max']):
                combined = min(3.0, conv * tier['mult'])
                denom = conv * macd
                if denom > 0:
                    scale = combined / denom
                    if abs(scale - 1.0) > 0.001:
                        pnl *= scale
                break
        by_split[t.split].append({'date': t.date, 'pnl': pnl})
    out = {}
    for split in ['TRAIN', 'VAL', 'HOQ1', 'HOAPR']:
        ts = by_split[split]
        if not ts:
            out[split] = {'n': 0, 'pnl': 0.0, 'wr': 0.0, 'maxdd': 0.0}
            continue
        total = sum(t['pnl'] for t in ts)
        wins = sum(1 for t in ts if t['pnl'] > 0)
        running, peak, dd = 0.0, 0.0, 0.0
        for t in sorted(ts, key=lambda x: x['date']):
            running += t['pnl']
            peak = max(peak, running)
            dd = min(dd, running - peak)
        out[split] = {'n': len(ts), 'pnl': total, 'wr': wins / len(ts) * 100, 'maxdd': dd}
    return out


def score_params(stats: Dict, baseline: Dict, min_trades_ratio=0.7,
                 max_trades_ratio=1.5) -> float:
    """Rank function: TRAIN+VAL P&L improvement, with penalty for extreme
    trade-count drift or negative delta on any split."""
    base_tv = baseline['TRAIN']['pnl'] + baseline['VAL']['pnl']
    p_tv = stats['TRAIN']['pnl'] + stats['VAL']['pnl']
    # Trade count constraint
    base_n = baseline['TRAIN']['n'] + baseline['VAL']['n']
    p_n = stats['TRAIN']['n'] + stats['VAL']['n']
    ratio = p_n / base_n if base_n > 0 else 0
    if ratio < min_trades_ratio or ratio > max_trades_ratio:
        return -1e9
    # Reward TRAIN+VAL gain; mild penalty for VAL regressing
    gain = p_tv - base_tv
    # Penalize if either TRAIN or VAL regresses badly
    if stats['TRAIN']['pnl'] < baseline['TRAIN']['pnl'] - 3000:
        gain -= 5000
    if stats['VAL']['pnl'] < baseline['VAL']['pnl'] - 3000:
        gain -= 5000
    return gain


def phase4(trades: List[Trade], max_candidates=15):
    print("\n# Phase 4 — joint parameter grid search\n")
    print("Grid search over rule weights × threshold × tier × MACD zone. "
          "Score = (TRAIN+VAL P&L gain vs baseline) with trade-count constraint "
          "[0.7×, 1.5×] and ≤$3K hit on either TRAIN or VAL.\n")
    print("Baseline reference (current config @ conv>=1.4, current tiers):")
    base = baseline_stats(trades)
    for split in ['TRAIN', 'VAL', 'HOQ1', 'HOAPR']:
        s = base[split]
        print(f"  {split}: n={s['n']} WR={s['wr']:.1f}% PnL=${s['pnl']:+,.0f} DD=${s['maxdd']:+,.0f}")

    # Grid definition:
    weight_ranges = {
        'w_r1':  [0.0, 0.1, 0.2, 0.3, 0.4],
        'w_r2p': [0.0, 0.15, 0.3],
        'w_r2n': [-0.3, -0.15, 0.0],
        'w_r3':  [0.0, 0.15, 0.3, 0.45],
        'w_r5':  [0.0, 0.1, 0.2],
        'w_r7':  [-0.2, 0.0, 0.2],
        'w_r9':  [0.4, 0.5, 0.6],
    }
    threshold_range = [0.8, 1.0, 1.2, 1.4, 1.6]
    cap_range = [3.0, 3.5]

    # Because the full grid is too large, we'll do a smart 2-phase search:
    # (A) coarse sweep on rule weights at fixed threshold & tiers,
    # (B) threshold sweep at the top-3 weight configs,
    # (C) tier-table refinement at the best (weights, threshold).

    # ------- Phase A: rule-weight coarse sweep (threshold=1.4, default tiers, cap=3.0) -------
    print("\n## A. Rule-weight coarse sweep (threshold=1.4, tiers=default)\n")
    from itertools import product
    candidates = []
    space = [(k, v) for k, v in weight_ranges.items()]
    keys = [x[0] for x in space]
    vals = [x[1] for x in space]
    total_configs = 1
    for v in vals:
        total_configs *= len(v)
    print(f"Total weight configs: {total_configs}")
    count = 0
    for combo in product(*vals):
        params = Params(**dict(zip(keys, combo)))
        s = simulate(trades, params)
        score = score_params(s, base)
        if score > -1e8:
            candidates.append((score, s, dict(zip(keys, combo))))
        count += 1
    candidates.sort(key=lambda x: -x[0])
    print(f"Top {max_candidates} weight configs by TRAIN+VAL gain:\n")
    print("| Rank | Config | T+V gain | TRAIN n/Pnl | VAL n/Pnl | HOLDOUT n/Pnl |")
    print("|---|---|---:|---|---|---|")
    for i, (score, s, cfg) in enumerate(candidates[:max_candidates]):
        cfg_str = ' '.join(f"{k[2:]}={v}" for k, v in cfg.items())
        h_n = s['HOQ1']['n'] + s['HOAPR']['n']
        h_p = s['HOQ1']['pnl'] + s['HOAPR']['pnl']
        print(f"| {i+1} | {cfg_str} | ${score:+,.0f} | "
              f"{s['TRAIN']['n']}/${s['TRAIN']['pnl']:+,.0f} | "
              f"{s['VAL']['n']}/${s['VAL']['pnl']:+,.0f} | "
              f"{h_n}/${h_p:+,.0f} |")

    # ------- Phase B: threshold sweep on top-3 weight configs -------
    print("\n## B. Threshold sweep on top-3 weight configs\n")
    b_candidates = []
    for rank, (_, _, cfg) in enumerate(candidates[:3]):
        for th in threshold_range:
            for cap in cap_range:
                params = Params(**cfg, min_threshold=th, cap=cap)
                s = simulate(trades, params)
                score = score_params(s, base)
                b_candidates.append((score, s, {**cfg, 'th': th, 'cap': cap}, rank))
    b_candidates.sort(key=lambda x: -x[0])
    print("Top 10 (weights+threshold+cap) configs:\n")
    print("| Rank | rank_A | th | cap | T+V gain | TRAIN pnl | VAL pnl | HOLDOUT pnl |")
    print("|---|---|---|---|---:|---:|---:|---:|")
    for i, (score, s, cfg, rank_A) in enumerate(b_candidates[:10]):
        h_p = s['HOQ1']['pnl'] + s['HOAPR']['pnl']
        print(f"| {i+1} | {rank_A+1} | {cfg['th']} | {cfg['cap']} | "
              f"${score:+,.0f} | ${s['TRAIN']['pnl']:+,.0f} | "
              f"${s['VAL']['pnl']:+,.0f} | ${h_p:+,.0f} |")

    # ------- Phase C: tier refinement on top candidate -------
    if not b_candidates:
        return
    best_cfg = b_candidates[0][2]
    print(f"\n## C. Tier-table refinement on top config (weights+th={best_cfg['th']})\n")

    tier_variants = [
        ('Current', [{'name': 'T1', 'p_min': 10, 'p_max': 15, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
                     {'name': 'T2', 'p_min': 15, 'p_max': 23, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0}]),
        ('+T3<$5 @1.5', [{'name': 'T1', 'p_min': 10, 'p_max': 15, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
                        {'name': 'T2', 'p_min': 15, 'p_max': 23, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
                        {'name': 'T3', 'p_min': 0, 'p_max': 5, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.5}]),
        ('+T3<$5 @2.0', [{'name': 'T1', 'p_min': 10, 'p_max': 15, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
                        {'name': 'T2', 'p_min': 15, 'p_max': 23, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
                        {'name': 'T3', 'p_min': 0, 'p_max': 5, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0}]),
        ('demote T1 + T3@2.0',
                       [{'name': 'T1', 'p_min': 10, 'p_max': 15, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
                        {'name': 'T2', 'p_min': 15, 'p_max': 23, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
                        {'name': 'T3', 'p_min': 0, 'p_max': 5, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0}]),
    ]
    print("| Variant | T+V gain | TRAIN pnl | VAL pnl | HOLDOUT pnl | Total gain |")
    print("|---|---:|---:|---:|---:|---:|")
    for tname, tiers in tier_variants:
        params_kwargs = {k: v for k, v in best_cfg.items() if k not in ('th', 'cap')}
        params = Params(**params_kwargs, min_threshold=best_cfg['th'],
                        cap=best_cfg['cap'], tiers=tiers)
        s = simulate(trades, params)
        score = score_params(s, base)
        h_p = s['HOQ1']['pnl'] + s['HOAPR']['pnl']
        total_gain = (s['TRAIN']['pnl'] + s['VAL']['pnl'] + h_p) - (
            base['TRAIN']['pnl'] + base['VAL']['pnl'] + base['HOQ1']['pnl'] + base['HOAPR']['pnl'])
        print(f"| {tname} | ${score:+,.0f} | ${s['TRAIN']['pnl']:+,.0f} | "
              f"${s['VAL']['pnl']:+,.0f} | ${h_p:+,.0f} | ${total_gain:+,.0f} |")


def main():
    trades = load_all()
    phase1(trades)
    phase2(trades)
    phase3(trades)
    phase4(trades)


if __name__ == "__main__":
    main()
