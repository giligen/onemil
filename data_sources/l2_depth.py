"""
L2 order book depth snapshots via Databento.

Logs 10 levels of bid/ask depth at trade entry time for position sizing research.
Non-blocking: failures are logged and silently ignored — never affects trading.
"""

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Callable

logger = logging.getLogger(__name__)

# Databento datasets — query all major exchanges for consolidated depth.
# Combined depth is ~4x any single exchange on small-cap stocks.
_DATASETS = ["XNAS.ITCH", "ARCX.PILLAR", "BATS.PITCH", "EDGX.PITCH"]


def snapshot_l2_at_fill(
    symbol: str,
    fill_time: datetime,
    window_seconds: int = 5,
) -> Optional[Dict]:
    """
    Fetch 10-level L2 depth snapshot around a trade fill time.

    Returns a dict with total bid/ask depth and per-level breakdown,
    or None if data unavailable. Never raises — all errors caught.

    Args:
        symbol: Stock ticker
        fill_time: UTC datetime of the fill
        window_seconds: Seconds before/after fill to query
    """
    api_key = os.environ.get("DATABENTO_API_KEY") or os.environ.get("databento_api_key")
    if not api_key:
        logger.debug("DATABENTO_API_KEY not set, skipping L2 snapshot")
        return None

    try:
        import databento as db
    except ImportError:
        logger.debug("databento package not installed, skipping L2 snapshot")
        return None

    # Ensure fill_time is timezone-aware UTC
    if fill_time.tzinfo is None:
        fill_time = fill_time.replace(tzinfo=timezone.utc)

    start = (fill_time - timedelta(seconds=window_seconds)).strftime('%Y-%m-%dT%H:%M:%S')
    end = (fill_time + timedelta(seconds=window_seconds)).strftime('%Y-%m-%dT%H:%M:%S')

    client = db.Historical(key=api_key)

    # Query all exchanges and aggregate depth
    exchange_depths = {}
    combined_ask_depth = 0
    combined_bid_depth = 0

    for dataset in _DATASETS:
        try:
            data = client.timeseries.get_range(
                dataset=dataset,
                symbols=[symbol],
                schema="mbp-10",
                start=start,
                end=end,
            )
            df = data.to_df()

            if len(df) == 0:
                continue

            # Take the row closest to fill time (middle of window)
            row = df.iloc[len(df) // 2]

            ask_depth = 0
            bid_depth = 0
            levels = []
            for lvl in range(10):
                bid_px = float(row.get(f'bid_px_{lvl:02d}', 0) or 0)
                ask_px = float(row.get(f'ask_px_{lvl:02d}', 0) or 0)
                bid_sz = int(row.get(f'bid_sz_{lvl:02d}', 0) or 0)
                ask_sz = int(row.get(f'ask_sz_{lvl:02d}', 0) or 0)
                bid_depth += bid_sz
                ask_depth += ask_sz
                levels.append({
                    'bid_px': bid_px, 'bid_sz': bid_sz,
                    'ask_px': ask_px, 'ask_sz': ask_sz,
                })

            exchange_depths[dataset] = {
                'ask_depth': ask_depth,
                'bid_depth': bid_depth,
                'levels': levels,
                'records': len(df),
            }
            combined_ask_depth += ask_depth
            combined_bid_depth += bid_depth

        except Exception as e:
            logger.debug(f"{symbol}: L2 from {dataset} failed: {e}")
            continue

    if not exchange_depths:
        logger.info(f"{symbol}: No L2 data available from any exchange")
        return None

    snapshot = {
        'symbol': symbol,
        'fill_time': fill_time.isoformat(),
        'combined_ask_depth': combined_ask_depth,
        'combined_bid_depth': combined_bid_depth,
        'exchanges': exchange_depths,
        'exchange_count': len(exchange_depths),
    }

    logger.info(
        f"{symbol}: L2 snapshot — "
        f"combined ask {combined_ask_depth:,} / bid {combined_bid_depth:,} "
        f"across {len(exchange_depths)} exchanges"
    )
    return snapshot


def log_l2_async(
    symbol: str,
    fill_time: datetime,
    trade_db_id: int,
    db_update_fn: Callable,
    column: str = 'entry_l2_depth',
) -> None:
    """
    Fire-and-forget L2 snapshot logging in a background thread.

    Queries Databento, stores result in DB. Never blocks the trading loop.
    All errors silently logged.

    Args:
        symbol: Stock ticker
        fill_time: UTC datetime of the fill/trigger
        trade_db_id: Trade ID in the DB for update
        db_update_fn: Callable(trade_id, dict) to update the trade record
        column: DB column name ('entry_l2_depth' or 'exit_l2_depth')
    """
    def _worker():
        try:
            l2 = snapshot_l2_at_fill(symbol, fill_time)
            if l2:
                db_update_fn(trade_db_id, {column: l2_to_json(l2)})
        except Exception as e:
            logger.debug(f"{symbol}: async L2 snapshot failed ({column}): {e}")

    t = threading.Thread(target=_worker, daemon=True, name=f"l2-{symbol}-{column}")
    t.start()


def l2_to_json(snapshot: Optional[Dict]) -> Optional[str]:
    """Serialize L2 snapshot to JSON string for DB storage."""
    if snapshot is None:
        return None
    return json.dumps(snapshot)
