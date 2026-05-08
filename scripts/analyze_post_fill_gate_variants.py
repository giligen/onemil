"""Post-fill gate variant comparison (IREZ post-mortem, 2026-05-08).

Reads a cache CSV produced by `BT_POST_FILL_GATE_DISABLE=1 batch_backtest.py
--build-cache` (every filled trade has its NATURAL exit + gate inputs at fill
time recorded). Replays multiple gate-variant logics and tabulates aggregate
P&L, win rate, max drawdown, and per-variant outcome diffs.

Variants tested:
  V0  control                — spy_3d<0.8 AND bk_ratio<1.0 -> kill (full close)
  V1  tightened thresholds   — spy_3d<0.5 AND bk_ratio<0.5 -> kill (full close)
  V2  V1 + momentum override — V1 unless intraday_change_at_entry >= 25 (skip kill)
  V3  V0 + scale-down (25%)  — kill keeps 25% of position (= 25% natural PnL)
  V4  gate disabled          — never kill (full natural PnL)

For trades the variant would KILL:
    pnl_killed = -slippage * shares * entry_price       (V0/V1/V2)
    pnl_killed = 0.25 * pnl_natural                     (V3)
For trades the variant would NOT kill:
    pnl_kept = pnl_natural

Walk-forward: train Jan-Sep 2025, test Oct 2025 - May 2026.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


# Same exit slippage constant the BT uses (BacktestRunner.exit_slippage_pct).
EXIT_SLIPPAGE_PCT = 0.003


def _f(s: str, default: float = 0.0) -> float:
    if s is None or s == '':
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


@dataclass
class Variant:
    name: str
    description: str
    # gate_fires(trade_dict) -> True if this variant would kill the trade
    gate_fires: Callable[[Dict], bool]
    # outcome_pnl(trade_dict) -> P&L this variant produces.
    # Default: natural P&L if not killed, slippage loss if killed.
    outcome_pnl: Callable[[Dict], float]


def _killed_pnl_full_close(t: Dict) -> float:
    """P&L when gate kills with full close: entry_price * shares * -slippage."""
    return -EXIT_SLIPPAGE_PCT * _f(t.get('entry_price', '0')) * _f(t.get('shares', '0'))


def _killed_pnl_scale_25(t: Dict) -> float:
    """P&L when gate kills with scale-down to 25%."""
    return 0.25 * _f(t.get('pnl', '0'))


def _gate_v0(t: Dict) -> bool:
    spy3d = t.get('spy_3d_at_fill', '')
    bk = t.get('bk_ratio_at_fill', '')
    if spy3d == '' or bk == '':
        return False  # missing inputs -> can't fire (defensive)
    return _f(spy3d) < 0.8 and _f(bk) < 1.0


def _gate_v1(t: Dict) -> bool:
    spy3d = t.get('spy_3d_at_fill', '')
    bk = t.get('bk_ratio_at_fill', '')
    if spy3d == '' or bk == '':
        return False
    return _f(spy3d) < 0.5 and _f(bk) < 0.5


def _gate_v2(t: Dict) -> bool:
    """V1 thresholds with momentum override: skip kill if obvious momentum."""
    if not _gate_v1(t):
        return False
    ic = t.get('intraday_change_at_entry', '')
    if ic == '':
        return True  # no override info, fall through to V1 kill
    return not (_f(ic) >= 25.0)


def _outcome_natural_or_killed(killed_pnl_fn):
    def fn(t: Dict, fires: bool) -> float:
        return killed_pnl_fn(t) if fires else _f(t.get('pnl', '0'))
    return fn


def variants() -> List[Variant]:
    """Return all variants to compare."""
    nat_full = _outcome_natural_or_killed(_killed_pnl_full_close)
    nat_scale = _outcome_natural_or_killed(_killed_pnl_scale_25)

    return [
        Variant('V0_control',
                'spy_3d<0.8 AND bk<1.0 -> full close (current production)',
                _gate_v0, lambda t: nat_full(t, _gate_v0(t))),
        Variant('V1_tight',
                'spy_3d<0.5 AND bk<0.5 -> full close (only severe context)',
                _gate_v1, lambda t: nat_full(t, _gate_v1(t))),
        Variant('V2_v1_override',
                'V1 with skip if intraday_change>=25 (obvious momentum)',
                _gate_v2, lambda t: nat_full(t, _gate_v2(t))),
        Variant('V3_scale25',
                'V0 thresholds but kill scales position to 25% (keep 25% PnL)',
                _gate_v0, lambda t: nat_scale(t, _gate_v0(t))),
        Variant('V4_off',
                'gate disabled (always natural exit)',
                lambda t: False, lambda t: _f(t.get('pnl', '0'))),
    ]


def _max_drawdown(pnls: List[float]) -> float:
    """Max equity-curve drawdown from a list of trade P&Ls."""
    eq = 0.0
    peak = 0.0
    mdd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
    return mdd


def _summarize(name: str, trades: List[Dict], variant: Variant) -> Dict:
    """Compute aggregate stats for `variant` applied to `trades`."""
    pnls = []
    n_kills = 0
    n_kill_winners = 0  # trades killed that would have been profitable
    n_kill_losers = 0   # trades killed that would have been losers
    saved_dollars = 0.0  # net diff vs natural
    for t in trades:
        fires = variant.gate_fires(t)
        outcome = variant.outcome_pnl(t)
        natural = _f(t.get('pnl', '0'))
        pnls.append(outcome)
        if fires:
            n_kills += 1
            if natural > 0:
                n_kill_winners += 1
            else:
                n_kill_losers += 1
            saved_dollars += outcome - natural

    total = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        'split': name,
        'variant': variant.name,
        'description': variant.description,
        'n_trades': len(trades),
        'total_pnl': round(total, 2),
        'n_wins': len(wins),
        'n_losses': len(losses),
        'win_rate_pct': round(100 * len(wins) / max(len(trades), 1), 1),
        'avg_win': round(sum(wins) / max(len(wins), 1), 2),
        'avg_loss': round(sum(losses) / max(len(losses), 1), 2),
        'max_drawdown': round(_max_drawdown(pnls), 2),
        'best_trade': round(max(pnls) if pnls else 0, 2),
        'worst_trade': round(min(pnls) if pnls else 0, 2),
        'n_kills': n_kills,
        'n_kill_winners_lost': n_kill_winners,
        'n_kill_losers_avoided': n_kill_losers,
        'gate_pnl_diff_vs_natural': round(saved_dollars, 2),
    }


def _load(cache_path: str) -> List[Dict]:
    """Load cache CSV rows. Sorts by date then entry_time so DD is sequential."""
    rows = []
    with open(cache_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: (r.get('date', ''), r.get('entry_time_et', '')))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True, help='Path to cache CSV')
    p.add_argument('--train-end', default='2025-09-30', help='Train split inclusive end date')
    p.add_argument('--test-start', default='2025-10-01', help='Test split inclusive start date')
    p.add_argument('--out', default='data/experiments/gate_variant_results.csv',
                   help='Output CSV with summary stats')
    p.add_argument('--diff-out', default='data/experiments/gate_variant_diffs.csv',
                   help='Output CSV with per-trade diffs (V0 vs V4)')
    args = p.parse_args()

    if not os.path.exists(args.cache):
        print(f"ERROR: cache not found: {args.cache}", file=sys.stderr)
        sys.exit(1)

    all_trades = _load(args.cache)
    print(f"Loaded {len(all_trades)} trades from {args.cache}")

    # Sanity: how many have populated gate inputs?
    n_with_inputs = sum(1 for t in all_trades
                        if t.get('spy_3d_at_fill', '') != ''
                        and t.get('bk_ratio_at_fill', '') != '')
    print(f"  {n_with_inputs} trades have gate inputs populated "
          f"({100*n_with_inputs/max(len(all_trades),1):.1f}%)")

    # Splits
    train = [t for t in all_trades if t.get('date', '') <= args.train_end]
    test = [t for t in all_trades if t.get('date', '') >= args.test_start]
    print(f"  Train (≤{args.train_end}): {len(train)} trades")
    print(f"  Test  (≥{args.test_start}): {len(test)} trades")
    print(f"  Full:  {len(all_trades)} trades")
    print()

    summaries = []
    for split_name, split_trades in (('train', train), ('test', test), ('full', all_trades)):
        for v in variants():
            summaries.append(_summarize(split_name, split_trades, v))

    # Pretty print
    print(f"{'split':<6} {'variant':<16} {'n':>4} {'total':>11} {'WR%':>5} "
          f"{'maxDD':>10} {'kills':>6} {'kill_W_lost':>11} {'kill_L_saved':>12} "
          f"{'$ vs natural':>14}")
    print('-' * 116)
    for s in summaries:
        print(f"{s['split']:<6} {s['variant']:<16} "
              f"{s['n_trades']:>4} {s['total_pnl']:>11,.0f} "
              f"{s['win_rate_pct']:>5.1f} {s['max_drawdown']:>10,.0f} "
              f"{s['n_kills']:>6} {s['n_kill_winners_lost']:>11} "
              f"{s['n_kill_losers_avoided']:>12} "
              f"{s['gate_pnl_diff_vs_natural']:>14,.0f}")

    # Write summaries CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    print(f"\nSummaries written to {args.out}")

    # Per-trade diff: V0 (control) vs V4 (off) — shows exactly which trades
    # the gate is currently killing and what we'd recover by lifting it.
    diffs = []
    for t in all_trades:
        v0_fires = _gate_v0(t)
        if not v0_fires:
            continue
        natural = _f(t.get('pnl', '0'))
        v0_outcome = _killed_pnl_full_close(t)
        diffs.append({
            'date': t.get('date'),
            'symbol': t.get('symbol'),
            'entry_time_et': t.get('entry_time_et'),
            'entry_price': t.get('entry_price'),
            'shares': t.get('shares'),
            'spy_3d_at_fill': t.get('spy_3d_at_fill'),
            'bk_ratio_at_fill': t.get('bk_ratio_at_fill'),
            'intraday_change_at_entry': t.get('intraday_change_at_entry'),
            'natural_pnl': round(natural, 2),
            'v0_pnl': round(v0_outcome, 2),
            'recovered_if_no_kill': round(natural - v0_outcome, 2),
            'v1_would_kill': _gate_v1(t),
            'v2_would_kill': _gate_v2(t),
        })
    diffs.sort(key=lambda d: -d['recovered_if_no_kill'])
    with open(args.diff_out, 'w', newline='') as f:
        if diffs:
            w = csv.DictWriter(f, fieldnames=list(diffs[0].keys()))
            w.writeheader()
            w.writerows(diffs)
    print(f"V0-vs-V4 per-trade diffs ({len(diffs)} kills) written to {args.diff_out}")

    # Distribution of "what we'd recover by lifting V0 kill" — winners vs losers
    if diffs:
        winners = [d for d in diffs if d['natural_pnl'] > 0]
        losers = [d for d in diffs if d['natural_pnl'] <= 0]
        print()
        print(f"V0 kills breakdown ({len(diffs)} total):")
        print(f"  Would have been WINNERS (gate killed alpha): {len(winners)}, "
              f"avg natural ${sum(d['natural_pnl'] for d in winners)/max(len(winners),1):,.0f}, "
              f"max ${max((d['natural_pnl'] for d in winners), default=0):,.0f}")
        print(f"  Would have been LOSERS (gate saved): {len(losers)}, "
              f"avg natural ${sum(d['natural_pnl'] for d in losers)/max(len(losers),1):,.0f}, "
              f"min ${min((d['natural_pnl'] for d in losers), default=0):,.0f}")
        net = sum(d['recovered_if_no_kill'] for d in diffs)
        print(f"  Net $ recovered if no kill: ${net:,.0f}")
        # Top 5 lost winners
        print()
        print(f"Top 5 winners the gate killed:")
        for d in winners[:5]:
            print(f"  {d['date']} {d['symbol']:6} natural ${d['natural_pnl']:>9,.0f} "
                  f"spy_3d={d['spy_3d_at_fill']:<6} bk={d['bk_ratio_at_fill']:<6} "
                  f"intraday={d['intraday_change_at_entry']}")
        # Worst 5 losers the gate saved (i.e., natural would have been worse than slippage)
        worst_saves = sorted([d for d in losers if d['natural_pnl'] < d['v0_pnl']],
                             key=lambda x: x['natural_pnl'])[:5]
        print()
        print(f"Top 5 losers the gate saved (natural worse than slippage):")
        for d in worst_saves:
            print(f"  {d['date']} {d['symbol']:6} natural ${d['natural_pnl']:>9,.0f} "
                  f"v0=${d['v0_pnl']:>7,.0f} spy_3d={d['spy_3d_at_fill']:<6} "
                  f"bk={d['bk_ratio_at_fill']:<6}")

    # Per-month V0 vs V4 breakdown
    print()
    print("Per-month total P&L (V0 control vs V4 disabled vs V2 v1+override):")
    print(f"{'month':<8} {'V0':>11} {'V2':>11} {'V4':>11} {'V4-V0':>11} {'V2-V0':>11}")
    by_month = defaultdict(list)
    for t in all_trades:
        by_month[t.get('date', '')[:7]].append(t)
    for m in sorted(by_month):
        ts = by_month[m]
        v0_pnl = sum(_outcome_natural_or_killed(_killed_pnl_full_close)(t, _gate_v0(t)) for t in ts)
        v2_pnl = sum(_outcome_natural_or_killed(_killed_pnl_full_close)(t, _gate_v2(t)) for t in ts)
        v4_pnl = sum(_f(t.get('pnl', '0')) for t in ts)
        print(f"{m:<8} {v0_pnl:>11,.0f} {v2_pnl:>11,.0f} {v4_pnl:>11,.0f} "
              f"{v4_pnl-v0_pnl:>+11,.0f} {v2_pnl-v0_pnl:>+11,.0f}")


if __name__ == '__main__':
    main()
