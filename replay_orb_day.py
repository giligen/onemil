"""Historical ORB replay — drive real ORBEngine through a past trading day's
bars, mocked Alpaca client, dry-run orders. Validates the entry-decision path
against known BT picks.

What this does (and doesn't):
  ✓ Exercises real _compute_features, composite_score, quintile, dedup, planner
  ✓ Uses real prev-day bar + 20-day stats from DB cache (feature providers)
  ✓ Loads 1-min bars from DB cache for each candidate
  ✓ Ingests bars into ORBEngine._ingest_bars(...) to synthesize RangeData
  ✓ Calls check_entries in dry_run=True mode → prints picked plans
  ✗ Does NOT run StopMonitor (uses no-op mock for add_watch etc.)
  ✗ Does NOT simulate fills/exits — this is decision validation only, not P&L

Output: table of ranked candidates, picks, plan shares/mults/quintiles.

Usage:
    # Replay a specific date with a specific candidate list:
    python3 replay_orb_day.py --date 2026-03-13 --symbols PLYX,TMCR,RGC
    # Auto-pick candidates from the bull flag cache for that date:
    python3 replay_orb_day.py --date 2026-03-13 --from-cache
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yaml


logger = logging.getLogger('orb_replay')


class MockStopMonitor:
    """Minimal StopMonitor stand-in — ORBEngine only needs these methods for replay."""

    def __init__(self):
        self.polling_mode = False
        self.watches: Dict[str, dict] = {}
        self.subscribed: set = set()
        self.bar_handlers: Dict[str, callable] = {}

    def subscribe_bars(self, symbol: str) -> None:
        self.subscribed.add(symbol)

    def register_bar_handler(self, handler_id: str, cb) -> None:
        self.bar_handlers[handler_id] = cb

    def add_watch(self, *args, **kwargs) -> None:
        sym = kwargs.get('symbol') or (args[0] if args else '?')
        self.watches[sym] = {'args': args, 'kwargs': kwargs}

    def remove_watch(self, symbol: str) -> bool:
        return self.watches.pop(symbol, None) is not None

    def drain_exit_events(self) -> List:
        return []

    def get_last_data_ts(self) -> float:
        return 0.0


class MockAlpaca:
    """Minimal AlpacaClient stand-in backed by the real DB for daily + 1-min bars."""

    def __init__(self, db, replay_date: str):
        self.db = db
        self.replay_date = replay_date
        self.is_paper = True
        self.submitted: List[dict] = []

    def test_connection(self) -> bool:
        return True

    def get_account_info(self) -> dict:
        return {'buying_power': 1_000_000, 'cash': 1_000_000}

    def get_positions(self) -> list:
        return []

    def get_orders(self, status='open', limit=500) -> list:
        return []

    def get_daily_bars(self, symbols: List[str], days: int = 25) -> Dict[str, list]:
        """Return most recent `days` daily bars ending at replay_date - 1 day."""
        start_dt = (pd.Timestamp(self.replay_date) - pd.Timedelta(days=days + 10)).date()
        end_dt = (pd.Timestamp(self.replay_date) - pd.Timedelta(days=1)).date()
        try:
            bulk = self.db.get_daily_bars_cached(list(symbols), str(start_dt), str(end_dt))
        except Exception as e:
            logger.debug(f"replay: get_daily_bars_cached bulk failed: {e}")
            return {}
        return {s: b[-days:] for s, b in bulk.items() if b}

    def get_snapshots(self, symbols, feed=None) -> dict:
        return {}

    def submit_stop_bracket_order(self, **kwargs) -> dict:
        """Record submission; return fake order id."""
        order_id = f"REPLAY-{len(self.submitted):03d}"
        self.submitted.append({'order_id': order_id, **kwargs})
        return {'id': order_id, 'status': 'new', 'symbol': kwargs.get('symbol')}

    def get_order(self, order_id: str) -> dict:
        return {'id': order_id, 'status': 'new'}


def _load_orb_cfg() -> dict:
    with open('orb.yaml') as f:
        return yaml.safe_load(f)


def _load_intraday_bars(db, symbol: str, date_str: str) -> pd.DataFrame:
    """Load cached 1-min bars for a symbol/date → DataFrame with UTC timestamps."""
    bars = db.get_intraday_bars_cached(symbol, date_str)
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.sort_values('timestamp').reset_index(drop=True)


def _cache_candidates_for_date(date_str: str) -> List[str]:
    """Auto-pick candidates from the bull flag cache for the target date."""
    try:
        df = pd.read_csv('data/bull_flag_cache_e50_x30.csv')
    except FileNotFoundError:
        logger.warning("bull_flag_cache_e50_x30.csv not found — no auto candidates")
        return []
    day_df = df[df['date'] == date_str] if 'date' in df.columns else df.iloc[0:0]
    if day_df.empty:
        return []
    syms = day_df['symbol'].drop_duplicates().tolist() if 'symbol' in day_df.columns else []
    return syms


def _build_feature_providers(db, symbols: List[str], date_str: str) -> Dict[str, dict]:
    """Build feature_providers dict: sym -> {prev_day_bar, daily_stats_20d}."""
    providers: Dict[str, dict] = {}
    if not symbols:
        return providers
    target = pd.Timestamp(date_str)
    start = (target - pd.Timedelta(days=40)).date()
    end = (target - pd.Timedelta(days=1)).date()
    try:
        bulk = db.get_daily_bars_cached(list(symbols), str(start), str(end))
    except Exception as e:
        logger.warning(f"replay: bulk daily bar fetch failed: {e}")
        return providers
    for sym, daily in bulk.items():
        if not daily:
            continue
        prev_bar = daily[-1]
        closes = [float(b.get('close', 0)) for b in daily[-20:] if b.get('close')]
        highs = [float(b.get('high', 0)) for b in daily[-20:] if b.get('high')]
        stats_20d = {
            'high_20d': max(highs) if highs else 0.0,
            'avg_close_20d': sum(closes) / len(closes) if closes else 0.0,
        }
        providers[sym] = {'prev_day_bar': prev_bar, 'daily_stats_20d': stats_20d}
    return providers


def run_replay(date_str: str, symbols: List[str], verbose: bool = False) -> int:
    """Run the replay. Returns 0 on success."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    from persistence.database import Database
    from trading.orb_engine import ORBEngine

    db = Database()
    try:
        symbols = sorted({s.strip().upper() for s in symbols if s and s.strip()})
        if not symbols:
            logger.error("no candidate symbols — aborting replay")
            return 1

        logger.info(f"Replay {date_str} — {len(symbols)} candidate symbols")

        cfg = _load_orb_cfg()
        # Force enabled=True for replay even if prod flag is off
        cfg.setdefault('strategy', {})['enabled'] = True

        mock_alpaca = MockAlpaca(db, date_str)
        mock_sm = MockStopMonitor()

        engine = ORBEngine(
            alpaca_client=mock_alpaca,
            db=db,
            stop_monitor=mock_sm,
            config=cfg,
            dry_run=True,  # NEVER submit real orders
        )
        engine.enabled = True
        engine.register_on_stop_monitor()

        # Seed universe
        engine.build_universe(source_loader=lambda: symbols)

        # Ingest historical bars for each candidate (simulates the bar stream)
        ingested = 0
        skipped = []
        for sym in symbols:
            bars = _load_intraday_bars(db, sym, date_str)
            if bars.empty:
                skipped.append(sym)
                continue
            engine._ingest_bars(sym, bars)
            ingested += 1
        logger.info(
            f"Bars ingested: {ingested} symbols, skipped (no cached bars): "
            f"{len(skipped)}{' — ' + ','.join(skipped[:10]) if skipped else ''}"
        )

        # Log which candidates have RangeData populated
        with_range = [s for s in symbols if engine.candidates.get(s) and engine.candidates[s].range_data]
        logger.info(f"RangeData populated for {len(with_range)}/{len(symbols)}")

        # Build feature_providers (prev-day bar + 20d stats from DB cache)
        providers = _build_feature_providers(db, with_range, date_str)
        logger.info(f"Feature providers built for {len(providers)}/{len(with_range)}")

        # Evaluate — dry_run ensures no real order submission
        submitted = engine.check_entries(symbols=with_range, feature_providers=providers)

        # Collect scored candidates (check_entries cleared them; re-read from state)
        print()
        print("=" * 100)
        print(f"ORB REPLAY — {date_str}")
        print("=" * 100)
        print()

        # Score all candidates for visibility
        scored = []
        for sym in with_range:
            cand = engine.candidates[sym]
            if cand.composite is not None:
                scored.append(cand)

        scored.sort(key=lambda c: -c.composite)
        print(f"{'SYM':<8} {'Q':<3} {'COMP':>7} {'RH':>8} {'RL':>8} {'R$':>7} "
              f"{'MULT':>5} {'STATUS':<18}")
        print("-" * 100)
        for c in scored[:40]:
            mult = engine.adaptive_mults.get(c.quintile, 1.0) if c.quintile else 0.0
            status = 'PLAN SUBMITTED' if c.plan_submitted else (c.rejected_reason or 'not-picked')
            r_dollar = c.range_data.range_high - c.range_data.range_low if c.range_data else 0
            print(f"{c.symbol:<8} {c.quintile or '-':<3} {c.composite:>+7.3f} "
                  f"{c.range_data.range_high:>8.2f} {c.range_data.range_low:>8.2f} "
                  f"{r_dollar:>7.2f} {mult:>5.2f} {status:<18}")

        print()
        print(f"SUBMITTED: {len(submitted)} orders (dry-run)")
        for rec in mock_alpaca.submitted:
            print(f"  {rec['symbol']:<8} qty={rec['qty']:>5} "
                  f"stop=${rec['stop_price']:.2f} limit=${rec['limit_price']:.2f} "
                  f"sl=${rec['sl_price']:.2f} tp=${rec['tp_price']:.2f}")

        # Rejected summary
        rejected = [c for c in (engine.candidates.get(s) for s in with_range)
                    if c and c.rejected_reason]
        if rejected:
            print()
            print(f"REJECTED ({len(rejected)}):")
            by_reason: Dict[str, int] = {}
            for c in rejected:
                by_reason[c.rejected_reason] = by_reason.get(c.rejected_reason, 0) + 1
            for reason, n in sorted(by_reason.items(), key=lambda x: -x[1]):
                print(f"  {reason:<30} {n}")

        # Sanity: what the family/group dedup kept
        from trading.orb_correlation import symbol_family, symbol_super_group
        picked = submitted
        if picked:
            print()
            print("PICKED symbols → family / super-group:")
            for s in picked:
                print(f"  {s:<8} family={symbol_family(s)!r:<20} super={symbol_super_group(s)!r}")

        print()
        print("=" * 100)
        return 0
    finally:
        try:
            db.close()
        except Exception:
            pass


def main() -> int:
    p = argparse.ArgumentParser(description="ORB historical replay")
    p.add_argument('--date', required=True, help='YYYY-MM-DD')
    p.add_argument('--symbols', default='',
                   help='Comma-separated ticker list')
    p.add_argument('--from-cache', action='store_true',
                   help='Auto-load candidates from bull_flag_cache_e50_x30.csv for this date')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    syms: List[str] = []
    if args.symbols:
        syms = args.symbols.split(',')
    elif args.from_cache:
        syms = _cache_candidates_for_date(args.date)
        if not syms:
            print(f"No candidates found in cache for {args.date}", file=sys.stderr)
            return 1
    else:
        print("Provide --symbols or --from-cache", file=sys.stderr)
        return 2

    return run_replay(args.date, syms, verbose=args.verbose)


if __name__ == '__main__':
    sys.exit(main())
