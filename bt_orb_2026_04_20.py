"""BT-parity ORB simulation for 2026-04-20 — what SHOULD have happened.

Uses the shipped static_lock_1R exit, the live yaml filter / quintile /
adaptive mult params. Pulls 1-min bars directly from DB cache (already
populated from yesterday's live session) + daily bars for prior-day
features. Picks top-K at 9:35 ET cleanly (no restart-induced bugs).

Compares to actual live yesterday (which had 10+ bugs we've since fixed).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from persistence.database import Database
from data_sources.alpaca_client import AlpacaClient
from config import Config
from trading.orb_filter import FeatureParam, load_feature_params, composite_score, assign_quintile
from trading.orb_conviction import load_adaptive_mults, apply_adaptive_mult
from trading.orb_correlation import dedup_candidates
from trading.orb_planner import OrbTradePlanner


DATE = '2026-04-20'
FORCE_CLOSE_ET = dtime(15, 45)

# Yesterday's confirmed ORB universe from live logs (all passed gap>=5%, vol>=500K, $3-30)
# Plus the two late-arrivers (QBTZ, BATL) that also appeared in the 11.
UNIVERSE = [
    'ANNA', 'BMNZ', 'CRCD', 'SKYQ',   # the actual 4 picked
    'BATL', 'QBTZ',                     # later additions in actual
    'USGG',                             # in universe but deduped or ranked lower
    # Other symbols seen in the snapshot universe logs
    'ASTN', 'NBIG', 'ENVB', 'OPTX', 'PBM', 'LUNL',
]


def _load_bars(alpaca, symbol: str, date_str: str) -> pd.DataFrame:
    """Fetch historical 1-min bars from Alpaca for the given trading day."""
    start = datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc)  # 9:30 ET
    end = datetime(2026, 4, 20, 20, 0, tzinfo=timezone.utc)     # 16:00 ET
    try:
        df = alpaca.get_historical_1min_bars(symbol, start, end)
    except Exception as e:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.sort_values('timestamp').reset_index(drop=True)


def _build_range_data(bars: pd.DataFrame) -> Optional[dict]:
    """Compute 9:30-9:34 ET range features."""
    if bars.empty:
        return None
    # Market open = 13:30 UTC (EDT on 4/20)
    open_ts = datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc)
    end_ts = datetime(2026, 4, 20, 13, 35, tzinfo=timezone.utc)
    rb = bars[(bars['timestamp'] >= open_ts) & (bars['timestamp'] < end_ts)]
    if len(rb) < 5:
        return None
    return {
        'range_high': float(rb['high'].max()),
        'range_low': float(rb['low'].min()),
        'range_open': float(rb.iloc[0]['open']),
        'range_close': float(rb.iloc[-1]['close']),
        'range_size_pct': (float(rb['high'].max()) - float(rb['low'].min())) / float(rb.iloc[0]['open']) * 100.0,
        'range_total_volume': int(rb['volume'].sum()),
        'range_avg_bar_range_pct': float(((rb['high'] - rb['low']) / rb['close']).mean() * 100.0),
    }


def _get_prev_day_features(db, symbol: str, date_str: str) -> dict:
    """Fetch 20d daily bars for prior-day + 20d features."""
    target = pd.Timestamp(date_str)
    start = (target - pd.Timedelta(days=40)).date()
    end = (target - pd.Timedelta(days=1)).date()
    try:
        bulk = db.get_daily_bars_cached([symbol], str(start), str(end))
    except Exception:
        return {}
    daily = bulk.get(symbol) or []
    if not daily:
        return {}
    prev = daily[-1]
    window = daily[-20:]
    highs = [float(b.get('high', 0)) for b in window]
    return {
        'prev_day_bar': {
            'open': float(prev.get('open', 0)),
            'high': float(prev.get('high', 0)),
            'low': float(prev.get('low', 0)),
            'close': float(prev.get('close', 0)),
            'volume': int(prev.get('volume', 0)),
        },
        'daily_stats_20d': {'high_20d': max(highs) if highs else 0.0},
    }


def _compute_composite(rd: dict, prev: dict, stats_20d: dict, z_params: Dict[str, FeatureParam]) -> Optional[float]:
    """Build the 7-feature vector + composite score using the yaml z-params."""
    range_high = rd['range_high']
    range_low = rd['range_low']
    range_open = rd['range_open']
    prev_close = prev['close'] if prev else 0
    prev_high = prev['high'] if prev else 0
    prev_low = prev['low'] if prev else 0
    if range_open <= 0 or prev_close <= 0:
        return None
    gap_pct = (range_open - prev_close) / prev_close * 100.0
    price_vs_20d_high = (stats_20d.get('high_20d', 0) - range_high) / range_high * 100.0 if range_high > 0 else 0
    prev_range = prev_high - prev_low
    prev_day_close_position = (prev_close - prev_low) / prev_range if prev_range > 0 else 0.5
    range_close_position = (rd['range_close'] - range_low) / (range_high - range_low) if range_high > range_low else 0.5
    features = {
        'gap_pct': gap_pct,
        'range_total_volume': float(rd['range_total_volume']),
        'range_avg_bar_range_pct': rd['range_avg_bar_range_pct'],
        'range_size_pct': rd['range_size_pct'],
        'price_vs_20d_high_pct': price_vs_20d_high,
        'prev_day_close_position': prev_day_close_position,
        'range_close_position': range_close_position,
    }
    return composite_score(features, z_params)


def _simulate_static_lock(bars: pd.DataFrame, entry_price: float, range_high: float, range_low: float) -> tuple:
    """BT-parity exit: initial stop at range_low, lock at +1R after +1.5R touched.
    Hold to force_close (15:45 ET = 19:45 UTC) unless stop/lock fires.

    Returns (exit_price, exit_reason, exit_minute_from_open).
    """
    one_r = range_high - range_low
    arm_level = entry_price + 1.5 * one_r
    lock_stop = entry_price + 1.0 * one_r
    stop = range_low
    armed = False
    open_ts = datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc)
    # BT-parity: skip entry bar; find first bar AFTER entry
    entry_bar_ts = None
    for _, b in bars.iterrows():
        if b['timestamp'] >= datetime(2026, 4, 20, 13, 35, tzinfo=timezone.utc) and b['high'] >= range_high:
            entry_bar_ts = b['timestamp']
            break
    if entry_bar_ts is None:
        return None, 'no_trigger', None
    # Iterate post-entry bars
    post = bars[bars['timestamp'] > entry_bar_ts].reset_index(drop=True)
    force_close = datetime(2026, 4, 20, 19, 45, tzinfo=timezone.utc)
    for _, b in post.iterrows():
        if b['timestamp'] >= force_close:
            exit_min = int((b['timestamp'] - open_ts).total_seconds() / 60)
            return float(b['open']), 'force_close', exit_min
        if not armed and float(b['high']) >= arm_level:
            armed = True
            stop = max(stop, lock_stop)
        if float(b['low']) <= stop:
            exit_min = int((b['timestamp'] - open_ts).total_seconds() / 60)
            return stop, ('lock' if armed else 'stop_loss'), exit_min
    last = post.iloc[-1] if len(post) else None
    if last is not None:
        exit_min = int((last['timestamp'] - open_ts).total_seconds() / 60)
        return float(last['close']), 'eod', exit_min
    return None, 'no_exit', None


def main():
    print(f"\nORB BT — 'perfect execution' simulation for {DATE}\n")
    # Load yaml config
    with open('orb.yaml') as f:
        cfg = yaml.safe_load(f)
    z_params = load_feature_params(cfg.get('filter', {}))
    cutoffs = list(cfg.get('quintile_cutoffs', []))
    mults = load_adaptive_mults(cfg.get('adaptive_mults', {}))
    planner = OrbTradePlanner(cfg)
    ranking_order = cfg.get('ranking', {}).get('order', ['Q4', 'Q5', 'Q3', 'Q2', 'Q1'])
    filter_threshold = float(cfg.get('filter', {}).get('threshold', 0.0))

    c = Config()
    alpaca = AlpacaClient(c.alpaca_api_key, c.alpaca_api_secret, paper=True)
    db = Database()
    candidates = []
    for sym in UNIVERSE:
        bars = _load_bars(alpaca, sym, DATE)
        if bars.empty:
            continue
        rd = _build_range_data(bars)
        if rd is None:
            continue
        ctx = _get_prev_day_features(db, sym, DATE)
        prev = ctx.get('prev_day_bar', {})
        stats_20d = ctx.get('daily_stats_20d', {})
        comp = _compute_composite(rd, prev, stats_20d, z_params)
        if comp is None or comp < filter_threshold:
            candidates.append({'symbol': sym, 'range': rd, 'composite': comp,
                              'quintile': None, 'skipped': 'below_threshold', 'bars': bars})
            continue
        q = assign_quintile(comp, cutoffs)
        candidates.append({'symbol': sym, 'range': rd, 'composite': comp,
                          'quintile': q, 'skipped': None, 'bars': bars})
    db.close()

    # Rank + dedup top-K
    scored = [c for c in candidates if c['skipped'] is None]
    q_rank = {q: i for i, q in enumerate(ranking_order)}
    scored.sort(key=lambda c: (q_rank.get(c['quintile'], 99), -c['composite']))
    ranked_symbols = [c['symbol'] for c in scored]
    keep = set(dedup_candidates(ranked_symbols, max_keep=4, by_family=True, by_super_group=True))
    picks = [c for c in scored if c['symbol'] in keep][:4]

    print(f"Universe size: {len(UNIVERSE)}")
    print(f"Passed composite filter: {len(scored)}")
    print(f"Top-K after dedup: {len(picks)}\n")
    print(f"{'sym':6} {'Q':>3} {'comp':>6} {'rangeH':>7} {'rangeL':>7} {'range%':>6}  entry→exit  shares   pnl     reason")
    print('-' * 100)
    total_pnl = 0.0
    for c in picks:
        rd = c['range']
        plan = planner.build(
            symbol=c['symbol'],
            range_high=rd['range_high'],
            range_low=rd['range_low'],
            range_open=rd['range_open'],
            composite_score=c['composite'],
            quintile=c['quintile'],
            adaptive_mult=apply_adaptive_mult(c['quintile'], mults),
        )
        if not hasattr(plan, 'shares') or plan.shares < 1:
            print(f"{c['symbol']:6} — plan rejected: {plan}")
            continue
        exit_price, exit_reason, exit_min = _simulate_static_lock(
            c['bars'], plan.entry_price, rd['range_high'], rd['range_low']
        )
        if exit_price is None:
            print(f"{c['symbol']:6} {c['quintile']:>3} {c['composite']:>+6.2f}  NO_TRIGGER (never broke above range_high)")
            continue
        pnl = (exit_price - plan.entry_price) * plan.shares
        total_pnl += pnl
        print(f"{c['symbol']:6} {c['quintile']:>3} {c['composite']:>+6.2f} "
              f"{rd['range_high']:>7.2f} {rd['range_low']:>7.2f} {rd['range_size_pct']:>5.2f}%  "
              f"${plan.entry_price:.2f}→${exit_price:.2f}  "
              f"{plan.shares:>5}   ${pnl:>+8.2f}  {exit_reason}")

    print('-' * 100)
    print(f"{'BT DAY TOTAL':>72} ${total_pnl:>+10.2f}")

    # Also show what got filtered out
    rejected = [c for c in candidates if c['skipped']]
    if rejected:
        print(f"\nRejected by composite filter ({len(rejected)}):")
        for c in rejected:
            cs = c['composite']
            cs_s = f"{cs:+.2f}" if cs is not None else 'None'
            print(f"  {c['symbol']:6} composite={cs_s}  range%={c['range']['range_size_pct']:.2f}%")

    # Comparison with actual live P&L
    print("\n" + "=" * 100)
    print("COMPARISON: BT vs ACTUAL live on 2026-04-20")
    print("=" * 100)
    print(f"  BT (top-K at 9:35 ET, clean):         ${total_pnl:+,.2f}")
    print(f"  ACTUAL live (delayed entries, bugs):  $-2,293 realized + ANNA overnight leak (~$2.5K)")


if __name__ == '__main__':
    main()
