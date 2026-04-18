#!/usr/bin/env python3
"""Holistic optimizer Phase 4b — stability-aware joint search.

Extends holistic_optimizer.py Phase 4 with:
  - TRAIN-VAL stability scoring (gain% must be similar on both)
  - Expanded weight ranges (can INCREASE weights, not just decrease)
  - Tier table folded into the search (not just post-hoc refinement)
  - MACD zone multipliers varied (can shrink 1.0x bucket, bump 1.5x to 1.8x)
  - Finer threshold/cap grid
  - HOLDOUT evaluated ONE-SHOT on the leakage-clean winner
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from itertools import product
from typing import List, Dict, Tuple
from collections import defaultdict

from holistic_optimizer import (
    SPLITS, load_all, price_bucket, vol_bucket, Trade, Params,
    recompute_conv, baseline_stats, simulate,
)


def _score(stats, base, min_gain_pct=0.03, max_imbalance_pct=0.20,
           min_trades_ratio=0.7, max_trades_ratio=1.5):
    """Stability-aware scorer.

    Returns (primary_score, secondary_score, notes_dict).
    primary_score = T+V gain (negative if constraint violated)
    secondary_score = -|TRAIN gain% - VAL gain%| (stability tiebreaker)

    Constraints (violation → -1e9):
      1) trade count within [min, max] of baseline
      2) TRAIN gain% >= min_gain_pct AND VAL gain% >= min_gain_pct
      3) |TRAIN% - VAL%| <= max_imbalance_pct
    """
    b_train = base['TRAIN']['pnl']
    b_val = base['VAL']['pnl']
    p_train = stats['TRAIN']['pnl']
    p_val = stats['VAL']['pnl']
    if b_train <= 0 or b_val <= 0:
        return (-1e9, -1e9, {})

    train_gain_pct = (p_train - b_train) / b_train
    val_gain_pct = (p_val - b_val) / b_val
    tv_gain = (p_train + p_val) - (b_train + b_val)

    # Trade count constraint
    base_n = base['TRAIN']['n'] + base['VAL']['n']
    p_n = stats['TRAIN']['n'] + stats['VAL']['n']
    if base_n == 0:
        return (-1e9, -1e9, {})
    ratio = p_n / base_n
    if ratio < min_trades_ratio or ratio > max_trades_ratio:
        return (-1e9, -1e9, {'violated': 'trade_ratio', 'ratio': ratio})

    # Gain constraints
    if train_gain_pct < min_gain_pct:
        return (-1e9, -1e9, {'violated': 'train_gain', 'pct': train_gain_pct})
    if val_gain_pct < min_gain_pct:
        return (-1e9, -1e9, {'violated': 'val_gain', 'pct': val_gain_pct})

    # Imbalance constraint (TRAIN and VAL must both be in agreement)
    imbalance = abs(train_gain_pct - val_gain_pct)
    if imbalance > max_imbalance_pct:
        return (-1e9, -1e9, {'violated': 'imbalance', 'imb': imbalance})

    return (tv_gain, -imbalance, {
        'train_gain_pct': train_gain_pct,
        'val_gain_pct': val_gain_pct,
        'imbalance': imbalance,
        'ratio': ratio,
    })


# Tier candidate library
TIER_VARIANTS = {
    'current': [
        {'p_min': 10, 'p_max': 15, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
        {'p_min': 15, 'p_max': 23, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
    ],
    '+T3_small_1.5': [
        {'p_min': 10, 'p_max': 15, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
        {'p_min': 15, 'p_max': 23, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
        {'p_min': 0, 'p_max': 5,   'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.5},
    ],
    '+T3_small_2.0': [
        {'p_min': 10, 'p_max': 15, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
        {'p_min': 15, 'p_max': 23, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
        {'p_min': 0, 'p_max': 5,   'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
    ],
    'demote_T1_T3_1.5': [
        {'p_min': 10, 'p_max': 15, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
        {'p_min': 15, 'p_max': 23, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
        {'p_min': 0, 'p_max': 5,   'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.5},
    ],
    'demote_T1_T3_2.0': [
        {'p_min': 10, 'p_max': 15, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
        {'p_min': 15, 'p_max': 23, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
        {'p_min': 0, 'p_max': 5,   'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
    ],
    'rescue_T1b': [
        # rescue $10-15 <500K (orphan, +$14,704 in audit)
        {'p_min': 10, 'p_max': 15, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
        {'p_min': 10, 'p_max': 15, 'v_min': 0,       'v_max': 500_000,   'mult': 1.5},
        {'p_min': 15, 'p_max': 23, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
        {'p_min': 0,  'p_max': 5,  'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
    ],
    'no_tiers': [],  # disable all tiers (all trades default to 1.0x)
    'filter_large_mid': [
        # tier $15-23 @1.0 but filter out $23+ and <500K mid-caps (implicit via not scaling)
        {'p_min': 10, 'p_max': 15, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
        {'p_min': 15, 'p_max': 23, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
        {'p_min': 0,  'p_max': 5,  'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
        {'p_min': 10, 'p_max': 15, 'v_min': 0,       'v_max': 500_000,   'mult': 1.5},
    ],
}


def search_weights_first(trades, base, limit=30):
    """Phase 1 of v2 search: rule-weight coarse sweep at current tiers & th=1.4.
    Returns top-N by stability-scored T+V gain."""
    weight_ranges = {
        'w_r1':  [0.0, 0.15, 0.3, 0.4],
        'w_r2p': [0.15, 0.3, 0.4],
        'w_r2n': [-0.3, -0.15, 0.0],
        'w_r3':  [0.0, 0.15, 0.3, 0.45],
        'w_r5':  [0.0, 0.1, 0.2, 0.3],
        'w_r7':  [-0.2, 0.0, 0.2, 0.3],
        'w_r9':  [0.4, 0.5, 0.6, 0.7],
    }
    keys = list(weight_ranges.keys())
    vals = [weight_ranges[k] for k in keys]
    total = 1
    for v in vals:
        total *= len(v)
    print(f"Phase S1: scanning {total} weight configs @ th=1.4 tiers=current...")
    cand = []
    for combo in product(*vals):
        p = Params(**dict(zip(keys, combo)))
        s = simulate(trades, p)
        prim, sec, notes = _score(s, base)
        if prim > -1e8:
            cand.append((prim, sec, dict(zip(keys, combo)), s, notes))
    cand.sort(key=lambda x: (-x[0], -x[1]))
    return cand[:limit]


def search_stage2(trades, base, top_weights, limit=30):
    """Phase 2 of v2 search: threshold × cap × tier × MACD zone on top weight configs."""
    threshold_range = [0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6]
    cap_range = [2.5, 3.0, 3.5, 4.0]
    macd_pairs = [
        (1.0, 1.5),   # current
        (0.75, 1.5),  # shrink 1.0 bucket
        (0.5, 1.5),   # aggressive shrink
        (1.0, 1.8),   # bump 1.5 bucket
        (0.75, 1.8),  # both
        (1.0, 2.0),   # max bump
    ]
    tier_keys = list(TIER_VARIANTS.keys())

    print(f"Phase S2: on {len(top_weights)} weight configs × "
          f"{len(threshold_range)} thresholds × {len(cap_range)} caps × "
          f"{len(macd_pairs)} macd-pairs × {len(tier_keys)} tiers = "
          f"{len(top_weights)*len(threshold_range)*len(cap_range)*len(macd_pairs)*len(tier_keys)} configs")
    cand = []
    for _, _, w_cfg, _, _ in top_weights:
        for th, cap, (m_norm, m_strong), tk in product(
                threshold_range, cap_range, macd_pairs, tier_keys):
            p = Params(
                **w_cfg, min_threshold=th, cap=cap,
                macd_normal=m_norm, macd_strong=m_strong,
                tiers=TIER_VARIANTS[tk],
            )
            s = simulate(trades, p)
            prim, sec, notes = _score(s, base)
            if prim > -1e8:
                full_cfg = {**w_cfg, 'th': th, 'cap': cap,
                            'macd_norm': m_norm, 'macd_strong': m_strong,
                            'tier': tk}
                cand.append((prim, sec, full_cfg, s, notes))
    cand.sort(key=lambda x: (-x[0], -x[1]))
    return cand[:limit]


def print_candidate(rank, score, sec, cfg, stats, notes):
    h_n = stats['HOQ1']['n'] + stats['HOAPR']['n']
    h_p = stats['HOQ1']['pnl'] + stats['HOAPR']['pnl']
    tv = stats['TRAIN']['pnl'] + stats['VAL']['pnl']
    print(f"| {rank} | ${tv:+,.0f} | {notes.get('train_gain_pct', 0)*100:+.1f}% | "
          f"{notes.get('val_gain_pct', 0)*100:+.1f}% | "
          f"{notes.get('imbalance', 0)*100:.1f}% | "
          f"{stats['TRAIN']['n']} | {stats['VAL']['n']} | {h_n} | "
          f"${h_p:+,.0f} | "
          + ' '.join(f"{k}={v}" for k, v in cfg.items() if k != 'tier')
          + f" | t={cfg.get('tier', 'current')} |")


def main():
    trades = load_all()
    base = baseline_stats(trades)

    print("# Phase 4b — stability-aware joint grid search\n")
    print(f"Trades: TRAIN={sum(1 for t in trades if t.split=='TRAIN')}, "
          f"VAL={sum(1 for t in trades if t.split=='VAL')}, "
          f"HOQ1={sum(1 for t in trades if t.split=='HOQ1')}, "
          f"HOAPR={sum(1 for t in trades if t.split=='HOAPR')}")
    print("\nBaseline (current ship config @ conv>=1.4, current tiers):\n")
    for s in ['TRAIN', 'VAL', 'HOQ1', 'HOAPR']:
        st = base[s]
        print(f"  {s}: n={st['n']} WR={st['wr']:.1f}% PnL=${st['pnl']:+,.0f} "
              f"DD=${st['maxdd']:+,.0f}")

    print("\nConstraints: TRAIN gain ≥ +3%, VAL gain ≥ +3%, "
          "|TRAIN%-VAL%| ≤ 20pt, trade count 0.7x-1.5x of baseline.\n")

    # Stage 1: coarse weight sweep
    top_w = search_weights_first(trades, base, limit=30)
    print(f"\n## Stage 1 — Top {len(top_w)} weight configs (fixed th=1.4, tiers=current)\n")
    print("| # | T+V gain | TRAIN% | VAL% | |Δ| | TRn | VLn | HOn | HOLDOUT | weights | tier |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for i, (sc, sec, cfg, s, notes) in enumerate(top_w[:10]):
        print_candidate(i+1, sc, sec, cfg, s, notes)

    # Stage 2: add threshold, cap, macd, tier on top-10 weight configs
    top_full = search_stage2(trades, base, top_w[:10], limit=30)
    print(f"\n## Stage 2 — Top {len(top_full)} full configs (weights × th × cap × macd × tier)\n")
    print("| # | T+V gain | TRAIN% | VAL% | |Δ| | TRn | VLn | HOn | HOLDOUT | config | tier |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for i, (sc, sec, cfg, s, notes) in enumerate(top_full[:15]):
        print_candidate(i+1, sc, sec, cfg, s, notes)

    # HOLDOUT one-shot for top-5
    print(f"\n## Stage 3 — One-shot HOLDOUT validation for top-5 stable configs\n")
    print("Same top 5 from Stage 2, reporting HOLDOUT Q1 + April deltas vs baseline:\n")
    base_ho = base['HOQ1']['pnl'] + base['HOAPR']['pnl']
    base_all = sum(base[s]['pnl'] for s in ['TRAIN', 'VAL', 'HOQ1', 'HOAPR'])
    print("| # | config | TRAIN Δ | VAL Δ | HOQ1 Δ | HOAPR Δ | HOLDOUT Δ | Grand Δ | % of base |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, (sc, sec, cfg, s, notes) in enumerate(top_full[:5]):
        tr_d = s['TRAIN']['pnl'] - base['TRAIN']['pnl']
        vl_d = s['VAL']['pnl'] - base['VAL']['pnl']
        q1_d = s['HOQ1']['pnl'] - base['HOQ1']['pnl']
        ap_d = s['HOAPR']['pnl'] - base['HOAPR']['pnl']
        ho_d = q1_d + ap_d
        total_d = tr_d + vl_d + ho_d
        pct = total_d / base_all * 100
        cfg_str = ' '.join(f"{k[2:] if k.startswith('w_') else k}={v}"
                           for k, v in cfg.items() if k != 'tier')
        print(f"| {i+1} | {cfg_str} t={cfg.get('tier')} | "
              f"${tr_d:+,.0f} | ${vl_d:+,.0f} | ${q1_d:+,.0f} | ${ap_d:+,.0f} | "
              f"${ho_d:+,.0f} | ${total_d:+,.0f} | {pct:+.1f}% |")

    # Detailed inspection of #1
    if top_full:
        sc, sec, cfg, s, notes = top_full[0]
        print("\n## Recommended ship config (rank 1 of stability-scored search)\n")
        print(f"```")
        print("Rule weights:")
        print(f"  w_r1 (pole_gain):      {cfg.get('w_r1', 0.3)}  (baseline 0.3)")
        print(f"  w_r2+ (flag_tight):    {cfg.get('w_r2p', 0.3)}  (baseline 0.3)")
        print(f"  w_r2- (flag_loose):    {cfg.get('w_r2n', -0.3)}  (baseline -0.3)")
        print(f"  w_r3 (vol_ratio):      {cfg.get('w_r3', 0.3)}  (baseline 0.3)")
        print(f"  w_r5 (retracement):    {cfg.get('w_r5', 0.2)}  (baseline 0.2)")
        print(f"  w_r7 (vwap_dist):      {cfg.get('w_r7', 0.2)}  (baseline 0.2)")
        print(f"  w_r9 (v_reversal):     {cfg.get('w_r9', 0.4)}  (baseline 0.4)")
        print(f"min_threshold:           {cfg.get('th', 1.4)}  (baseline 1.4)")
        print(f"cap:                     {cfg.get('cap', 3.0)}  (baseline 3.0)")
        print(f"macd_normal:             {cfg.get('macd_norm', 1.0)}  (baseline 1.0)")
        print(f"macd_strong:             {cfg.get('macd_strong', 1.5)}  (baseline 1.5)")
        print(f"tier variant:            {cfg.get('tier', 'current')}")
        print(f"```\n")
        print(f"- TRAIN: {s['TRAIN']['n']} trades / ${s['TRAIN']['pnl']:+,.0f} "
              f"({notes.get('train_gain_pct', 0)*100:+.1f}%)")
        print(f"- VAL:   {s['VAL']['n']} trades / ${s['VAL']['pnl']:+,.0f} "
              f"({notes.get('val_gain_pct', 0)*100:+.1f}%)")
        print(f"- HOQ1:  {s['HOQ1']['n']} trades / ${s['HOQ1']['pnl']:+,.0f}")
        print(f"- HOAPR: {s['HOAPR']['n']} trades / ${s['HOAPR']['pnl']:+,.0f}")
        total = sum(s[x]['pnl'] for x in ['TRAIN', 'VAL', 'HOQ1', 'HOAPR'])
        print(f"- **Grand total PnL: ${total:+,.0f} vs baseline ${base_all:+,.0f} "
              f"(+${total - base_all:+,.0f}, +{(total-base_all)/base_all*100:.1f}%)**")


if __name__ == "__main__":
    main()
