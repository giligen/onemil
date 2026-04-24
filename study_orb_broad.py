#!/usr/bin/env python3
"""ORB Phase-B study with broader, unbiased universe.

Reuses simulate_orb_trade from study_orb.py. Replaces the universe loader
with a broad query on daily_bars (gap>=5%, prev_vol>=500K, open $3-30) —
this eliminates the universe-selection bias of Phase A (which used only
bull-flag-qualified stocks).

Usage:
    python3 study_orb_broad.py

Outputs (timestamped):
    analysis_results/orb_study_broad_{YYYYMMDD_HHMM}.md
    analysis_results/orb_trades_broad_{YYYYMMDD_HHMM}.csv
"""
from __future__ import annotations

import os
import sqlite3
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database

# Reuse Phase A: simulator + variants + splits + ship criteria
from study_orb import (
    ENTRY_SLIP_BPS_DEFAULT, EXIT_SLIP_BPS_DEFAULT, POSITION_SIZE_USD,
    SPLITS, VARIANTS, SHIP_CRITERIA, OUT_DIR,
    OrbTrade, simulate_orb_trade, _bars_to_df,
    summarize_trades, filter_trades_by_date, walk_forward_stats,
    _verdict,
)

CACHE_DB = 'data/cache.db'
BULL_FLAG_CACHE = 'data/bull_flag_cache_e50_x30.csv'

# Broad universe filter
MIN_GAP_PCT = 5.0           # today.open vs yesterday.close
MIN_PREV_DAY_VOL = 500_000  # yesterday's volume floor
MIN_OPEN_PRICE = 3.0
MAX_OPEN_PRICE = 30.0
DATE_START = '2025-01-01'
DATE_END = '2026-04-30'


def load_broad_universe(
    db_path: str = CACHE_DB,
    include_provisional_today=None,
) -> Dict[str, List[str]]:
    """Query daily_bars for gap-up movers with liquidity.

    Criteria (all must hold):
      - Previous day's close > 0 (valid prev bar)
      - (today.open - prev.close) / prev.close * 100 >= MIN_GAP_PCT
      - Previous day's volume >= MIN_PREV_DAY_VOL
      - Today's open in [MIN_OPEN_PRICE, MAX_OPEN_PRICE]
      - AND we have intraday 1-min bars cached for the (symbol, date) pair
        (we can't backtest without bars; missing pairs are documented)

    `include_provisional_today` (a `datetime.date`) adds a second pass that
    pulls today's gap-up candidates from `daily_bars_provisional`, joined
    against prev-day's FINAL row in `daily_bars`. Used by mid-day BT runs
    with `--include-today-provisional` so today's trades are visible even
    though the main daily_bars table is (correctly) empty for today.

    Returns {date_str: [symbol, ...]}.
    """
    conn = sqlite3.connect(db_path)
    grouped: Dict[str, List[str]] = {}
    # Main pass: final daily_bars only.
    query = """
    WITH daily_ranked AS (
        SELECT symbol, bar_date, open,
               LAG(close) OVER (PARTITION BY symbol ORDER BY bar_date) AS prev_close,
               LAG(volume) OVER (PARTITION BY symbol ORDER BY bar_date) AS prev_vol
        FROM daily_bars
    ),
    qualified AS (
        SELECT symbol, bar_date FROM daily_ranked
        WHERE bar_date BETWEEN ? AND ?
          AND prev_close IS NOT NULL AND prev_close > 0
          AND (open - prev_close) / prev_close * 100 >= ?
          AND prev_vol >= ?
          AND open BETWEEN ? AND ?
    )
    SELECT q.symbol, q.bar_date
    FROM qualified q
    WHERE EXISTS (
        SELECT 1 FROM intraday_bars_1min i
        WHERE i.symbol = q.symbol AND i.bar_date = q.bar_date
    )
    ORDER BY q.bar_date, q.symbol
    """
    cur = conn.execute(
        query,
        (DATE_START, DATE_END, MIN_GAP_PCT, MIN_PREV_DAY_VOL,
         MIN_OPEN_PRICE, MAX_OPEN_PRICE),
    )
    for symbol, bar_date in cur.fetchall():
        grouped.setdefault(str(bar_date), []).append(symbol)

    # Mid-day overlay: use today's provisional row. Prev-close / prev-vol
    # come from the most-recent row in daily_bars strictly before today.
    # Intraday bars must also exist for today (written by the BT's provisional
    # intraday-fill step) — otherwise we can't extract features / simulate.
    if include_provisional_today is not None:
        today_str = str(include_provisional_today)
        q2 = """
        WITH prior AS (
            SELECT symbol, close AS prev_close, volume AS prev_vol,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY bar_date DESC) AS rn
            FROM daily_bars
            WHERE bar_date < ?
        )
        SELECT p.symbol
        FROM daily_bars_provisional p
        JOIN prior r ON r.symbol = p.symbol AND r.rn = 1
        WHERE p.bar_date = ?
          AND r.prev_close > 0
          AND (p.open - r.prev_close) / r.prev_close * 100 >= ?
          AND r.prev_vol >= ?
          AND p.open BETWEEN ? AND ?
          AND EXISTS (
              SELECT 1 FROM intraday_bars_1min i
              WHERE i.symbol = p.symbol AND i.bar_date = ?
          )
        ORDER BY p.symbol
        """
        cur = conn.execute(
            q2,
            (today_str, today_str,
             MIN_GAP_PCT, MIN_PREV_DAY_VOL,
             MIN_OPEN_PRICE, MAX_OPEN_PRICE,
             today_str),
        )
        for (symbol,) in cur.fetchall():
            grouped.setdefault(today_str, []).append(symbol)

    conn.close()
    return grouped


def run_variant_broad(
    variant: Tuple,
    universe: Dict[str, List[str]],
    bars_cache: Dict[Tuple[str, str], pd.DataFrame],
) -> List[OrbTrade]:
    """Mirror of study_orb.run_variant but operates on broad universe."""
    name, range_min, entry_mode, stop_mode, target_mult, time_stop, vol_conf, spread_gate = variant
    trades: List[OrbTrade] = []
    for date_str in sorted(universe.keys()):
        for symbol in universe[date_str]:
            bars_df = bars_cache.get((symbol, date_str))
            if bars_df is None or bars_df.empty:
                continue
            t = simulate_orb_trade(
                bars_df, symbol, date_str, name,
                range_minutes=range_min, entry_mode=entry_mode,
                stop_mode=stop_mode, target_mult=target_mult,
                time_stop_minutes=time_stop, require_vol_confirm=vol_conf,
                max_spread_bps=spread_gate,
            )
            if t.entered:
                trades.append(t)
    return trades


def bull_flag_daily_pnl() -> pd.Series:
    """Daily realised P&L from bull flag cache (same filter that produced $+372K)."""
    df = pd.read_csv(BULL_FLAG_CACHE)
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
    daily = df.groupby('date')['pnl'].sum()
    daily.index = pd.to_datetime(daily.index)
    return daily


def orthogonality(trades: List[OrbTrade]) -> Dict[str, float]:
    if not trades:
        return {'corr': float('nan'), 'unique_win_days_pct': 0.0,
                'unique_win_days': 0, 'total_win_days': 0,
                'combined_max_dd': 0.0}
    df = pd.DataFrame([{'date': t.date, 'pnl': t.pnl} for t in trades])
    orb_daily = df.groupby('date')['pnl'].sum()
    orb_daily.index = pd.to_datetime(orb_daily.index)
    bf_daily = bull_flag_daily_pnl()
    all_dates = sorted(set(orb_daily.index) | set(bf_daily.index))
    orb_aligned = orb_daily.reindex(all_dates, fill_value=0)
    bf_aligned = bf_daily.reindex(all_dates, fill_value=0)
    corr = float(orb_aligned.corr(bf_aligned))
    total_orb_wins = int((orb_aligned > 0).sum())
    unique_orb_wins = int(((orb_aligned > 0) & (bf_aligned <= 0)).sum())
    pct = unique_orb_wins / total_orb_wins if total_orb_wins > 0 else 0.0
    combined = orb_aligned + bf_aligned
    cum = combined.cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()
    return {
        'corr': corr, 'unique_win_days_pct': pct,
        'unique_win_days': unique_orb_wins, 'total_win_days': total_orb_wins,
        'combined_max_dd': float(dd),
    }


def main() -> None:
    t0 = datetime.now()
    print(f"[{t0.isoformat(timespec='seconds')}] Phase B: BROAD universe ORB study")
    print(f"  Filter: gap>={MIN_GAP_PCT}%, prev_vol>={MIN_PREV_DAY_VOL:,}, "
          f"open ${MIN_OPEN_PRICE}-${MAX_OPEN_PRICE}\n")

    print("Loading broad universe from daily_bars...")
    universe = load_broad_universe()
    n_dates = len(universe)
    n_pairs = sum(len(v) for v in universe.values())
    print(f"  Broad universe: {n_dates} dates, {n_pairs:,} (symbol, date) pairs "
          f"with intraday bars cached\n")

    # Bulk fetch bars
    print("Bulk-fetching 1-min bars...")
    db = Database(db_path=CACHE_DB)
    pair_list: List[Tuple[str, str]] = []
    for d, syms in universe.items():
        for s in syms:
            pair_list.append((s, d))
    raw_bars = db.get_intraday_bars_bulk(pair_list)
    print(f"  Retrieved bars for {len(raw_bars):,} pairs")
    db.close()

    bars_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    for k, v in raw_bars.items():
        bars_cache[k] = _bars_to_df(v)

    # Run variants
    print(f"\nRunning {len(VARIANTS)} variants on {len(bars_cache):,} bar sets...")
    all_trades_by_variant: Dict[str, List[OrbTrade]] = {}
    for i, variant in enumerate(VARIANTS, 1):
        name = variant[0]
        t_start = datetime.now()
        trades = run_variant_broad(variant, universe, bars_cache)
        all_trades_by_variant[name] = trades
        s = summarize_trades(trades)
        elapsed = (datetime.now() - t_start).total_seconds()
        print(f"  [{i:>2}/{len(VARIANTS)}] {name:<28} n={s['n']:>5}  "
              f"WR={s['wr']:>4.1f}%  P&L=${s['pnl']:>+10,.0f}  "
              f"avg=${s['avg_pnl']:>+6,.0f}  DD=${s['max_dd']:>+10,.0f}  "
              f"({elapsed:.1f}s)")

    # Walk-forward
    print("\nWalk-forward per variant (test ΔP&L vs ORB_5_vanilla baseline):")
    vanilla_wf = walk_forward_stats(all_trades_by_variant['ORB_5_vanilla'])
    variant_wf: Dict[str, Dict] = {}
    print(f"{'Variant':<28} {'A_test':>13} {'B_test':>13} {'C_test':>13} "
          f"{'mean Δ':>13} {'min Δ':>13}  Verdict")
    print('-' * 115)
    for name, trades in all_trades_by_variant.items():
        wf = walk_forward_stats(trades)
        variant_wf[name] = wf
        if name == 'ORB_5_vanilla':
            cells = [f"${wf[s[0]]['test']['pnl']:>+11,.0f}" for s in SPLITS]
            print(f"{'ORB_5_vanilla (baseline)':<28} {' '.join(cells)}  (baseline)")
            continue
        deltas = []
        cells = []
        for split_name, *_ in SPLITS:
            d = wf[split_name]['test']['pnl'] - vanilla_wf[split_name]['test']['pnl']
            deltas.append(d)
            cells.append(f"${d:>+11,.0f}")
        mean_d = sum(deltas) / len(deltas)
        min_d = min(deltas)
        v = _verdict(min_d, mean_d)
        print(f"{name:<28} {' '.join(cells)} ${mean_d:>+11,.0f} ${min_d:>+11,.0f}  {v}")

    # Winner selection
    print("\nSelecting winner by test mean P&L with min > 0 preferred...")
    candidates = []
    for name, wf in variant_wf.items():
        test_pnls = [wf[s[0]]['test']['pnl'] for s in SPLITS]
        candidates.append((name, sum(test_pnls)/len(test_pnls), min(test_pnls), test_pnls))
    # Rank: min > 0 first, then highest mean
    candidates.sort(key=lambda c: (c[2] > 0, c[1]), reverse=True)
    winner_name, w_mean, w_min, w_tpnls = candidates[0]
    print(f"  Winner: {winner_name}")
    print(f"    Test mean: ${w_mean:+,.0f}  min: ${w_min:+,.0f}  "
          f"per-split: [{', '.join(f'${p:+,.0f}' for p in w_tpnls)}]")

    # Orthogonality
    print("\nOrthogonality analysis on winner...")
    orth = orthogonality(all_trades_by_variant[winner_name])
    print(f"  Daily P&L corr with bull flag: {orth['corr']:+.3f}")
    print(f"  Unique winning days: {orth['unique_win_days']}/{orth['total_win_days']} "
          f"({orth['unique_win_days_pct']*100:.1f}%)")
    print(f"  Combined-book (ORB+BF) max DD: ${orth['combined_max_dd']:+,.0f}")

    # Ship gate
    winner_trades = all_trades_by_variant[winner_name]
    winner_summary = summarize_trades(winner_trades)
    winner_total = winner_summary['pnl']
    print("\n=== SHIP CRITERIA GATE (Phase B, broad universe) ===")
    results = {}
    v1 = winner_total >= SHIP_CRITERIA['min_total_test_pnl']
    results['total_15.5mo_pnl'] = (v1, f"${winner_total:+,.0f} vs >=${SHIP_CRITERIA['min_total_test_pnl']:+,.0f}")
    print(f"  [{'PASS' if v1 else 'FAIL'}] Total 15.5mo P&L: ${winner_total:+,.0f}")

    v2 = w_min > SHIP_CRITERIA['min_split_test_pnl']
    results['min_split_test_pnl'] = (v2, f"${w_min:+,.0f} vs > 0")
    print(f"  [{'PASS' if v2 else 'FAIL'}] Min split test P&L: ${w_min:+,.0f}")

    peak = winner_summary['peak'] or 1.0
    dd_pct = abs(winner_summary['max_dd']) / peak if peak > 0 else 0.0
    v3 = dd_pct < SHIP_CRITERIA['max_dd_pct_of_peak']
    results['max_dd_pct'] = (v3, f"{dd_pct*100:.1f}% vs < {SHIP_CRITERIA['max_dd_pct_of_peak']*100:.0f}%")
    print(f"  [{'PASS' if v3 else 'FAIL'}] Max DD / peak: {dd_pct*100:.1f}%")

    corr = orth['corr']
    v4 = corr < SHIP_CRITERIA['max_corr_with_bullflag']
    results['daily_pnl_corr'] = (v4, f"{corr:+.3f} vs < {SHIP_CRITERIA['max_corr_with_bullflag']}")
    print(f"  [{'PASS' if v4 else 'FAIL'}] Daily P&L corr: {corr:+.3f}")

    v5 = orth['unique_win_days_pct'] > SHIP_CRITERIA['min_unique_winning_days_pct']
    results['unique_win_days'] = (v5, f"{orth['unique_win_days_pct']*100:.1f}% vs > {SHIP_CRITERIA['min_unique_winning_days_pct']*100:.0f}%")
    print(f"  [{'PASS' if v5 else 'FAIL'}] Unique winning days: {orth['unique_win_days_pct']*100:.1f}%")

    all_pass = all(r[0] for r in results.values())
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT (Phase B broad): "
          f"{'✓ ALL PASS — real Strategy 3 candidate' if all_pass else '✗ REJECTED — edge does not survive broader universe'}")
    print(f"{'='*60}")

    # Write outputs
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    md_path = f"{OUT_DIR}/orb_study_broad_{ts}.md"
    csv_path = f"{OUT_DIR}/orb_trades_broad_{ts}.csv"

    flat = []
    for name, trades in all_trades_by_variant.items():
        for t in trades:
            flat.append(asdict(t))
    pd.DataFrame(flat).to_csv(csv_path, index=False)
    print(f"\nPer-trade CSV: {csv_path} ({len(flat):,} rows)")

    with open(md_path, 'w') as f:
        f.write(f"# ORB Phase-B Study (Broader Universe)\n\n")
        f.write(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n\n")
        f.write(f"**Purpose**: correct universe-selection bias of Phase A by replacing the "
                f"bull-flag-qualified subset (373 pairs) with all stocks meeting a "
                f"broader gap-up criterion.\n\n")
        f.write(f"**Universe criteria**: gap >= {MIN_GAP_PCT}% (prev close to open), "
                f"prev-day volume >= {MIN_PREV_DAY_VOL:,}, open price "
                f"${MIN_OPEN_PRICE}-${MAX_OPEN_PRICE}. Only pairs with intraday "
                f"bars cached are tested.\n\n")
        f.write(f"**Sample size**: {n_pairs:,} (symbol, date) pairs across "
                f"{n_dates} days ({DATE_START} to {DATE_END}).\n\n")
        f.write(f"**Slippage**: {ENTRY_SLIP_BPS_DEFAULT:.0f}/{EXIT_SLIP_BPS_DEFAULT:.0f}bps. "
                f"Position: ${POSITION_SIZE_USD:,}/trade.\n\n")

        f.write("## Per-variant summary (15.5mo, broad universe)\n\n")
        f.write("| Variant | n | WR | Total P&L | Avg | Max DD |\n|---|---:|---:|---:|---:|---:|\n")
        for name, trades in all_trades_by_variant.items():
            s = summarize_trades(trades)
            f.write(f"| {name} | {s['n']:,} | {s['wr']:.0f}% | "
                    f"${s['pnl']:+,.0f} | ${s['avg_pnl']:+,.0f} | "
                    f"${s['max_dd']:+,.0f} |\n")

        f.write("\n## Walk-forward (ΔP&L vs ORB_5_vanilla, test only)\n\n")
        f.write("| Variant | A test Δ | B test Δ | C test Δ | Mean Δ | Min Δ | Verdict |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for name, wf in variant_wf.items():
            if name == 'ORB_5_vanilla':
                continue
            deltas = []
            cells = []
            for split_name, *_ in SPLITS:
                d = wf[split_name]['test']['pnl'] - vanilla_wf[split_name]['test']['pnl']
                deltas.append(d)
                cells.append(f"${d:+,.0f}")
            mean_d = sum(deltas) / len(deltas)
            min_d = min(deltas)
            f.write(f"| {name} | {' | '.join(cells)} | ${mean_d:+,.0f} | "
                    f"${min_d:+,.0f} | {_verdict(min_d, mean_d)} |\n")

        f.write(f"\n## Winner: `{winner_name}`\n\n")
        f.write(f"- Test mean P&L: ${w_mean:+,.0f}\n")
        f.write(f"- Test min: ${w_min:+,.0f}\n")
        f.write(f"- Per-split: [{', '.join(f'${p:+,.0f}' for p in w_tpnls)}]\n\n")

        f.write("## Orthogonality vs bull flag\n\n")
        f.write(f"- Daily P&L correlation: **{corr:+.3f}**\n")
        f.write(f"- Unique winning days: **{orth['unique_win_days']}/{orth['total_win_days']} "
                f"({orth['unique_win_days_pct']*100:.1f}%)**\n")
        f.write(f"- Combined-book max DD: **${orth['combined_max_dd']:+,.0f}**\n\n")

        f.write("## Ship criteria\n\n")
        f.write("| Criterion | Result | Pass |\n|---|---|:-:|\n")
        for k, (passed, detail) in results.items():
            f.write(f"| {k} | {detail} | {'✓' if passed else '✗'} |\n")
        f.write(f"\n**Verdict: {'✓ ALL PASS — real Strategy 3 candidate' if all_pass else '✗ REJECTED'}**\n\n")

        f.write("## Limitations remaining (even after Phase B fix)\n\n")
        f.write(f"- Intraday coverage is ~60% of qualifying pairs — the 40% without cached "
                f"bars are silently dropped (small overall-sample bias, unlikely to flip sign).\n")
        f.write(f"- Flat {ENTRY_SLIP_BPS_DEFAULT}/{EXIT_SLIP_BPS_DEFAULT}bps slippage; "
                f"real 9:30-9:35 ET spreads are wider. The spread_gate_100 variant is our "
                f"proxy — it's restrictive but realistic for thin movers.\n")
        f.write("- No halt simulation.\n")
        f.write("- No max_concurrent simulated — on high-volatility days ORB could fire "
                f"5-10 setups simultaneously; live system would cap lower.\n")
        f.write("- Fixed $50K sizing — realistic but doesn't model conviction-based scaling.\n")

    print(f"Report: {md_path}")
    print(f"\nElapsed: {(datetime.now() - t0).total_seconds():.1f}s")


if __name__ == '__main__':
    main()
