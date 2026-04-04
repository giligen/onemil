"""
L2 order book depth snapshots via Databento.

Logs 10 levels of bid/ask depth at trade entry time for position sizing research.
Non-blocking: failures are logged and silently ignored — never affects trading.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Databento dataset mapping by exchange prefix
# NASDAQ-listed → XNAS.ITCH, BATS/CBOE-listed → try both
_DATASETS = ["XNAS.ITCH"]


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

            # Extract 10 levels of depth
            levels = []
            total_bid_depth = 0
            total_ask_depth = 0

            for lvl in range(10):
                bid_px = float(row.get(f'bid_px_{lvl:02d}', 0) or 0)
                ask_px = float(row.get(f'ask_px_{lvl:02d}', 0) or 0)
                bid_sz = int(row.get(f'bid_sz_{lvl:02d}', 0) or 0)
                ask_sz = int(row.get(f'ask_sz_{lvl:02d}', 0) or 0)

                total_bid_depth += bid_sz
                total_ask_depth += ask_sz

                levels.append({
                    'bid_px': bid_px, 'bid_sz': bid_sz,
                    'ask_px': ask_px, 'ask_sz': ask_sz,
                })

            snapshot = {
                'dataset': dataset,
                'symbol': symbol,
                'fill_time': fill_time.isoformat(),
                'snapshot_time': str(row.name) if hasattr(row, 'name') else None,
                'total_bid_depth': total_bid_depth,
                'total_ask_depth': total_ask_depth,
                'levels': levels,
                'records_in_window': len(df),
            }

            logger.info(
                f"{symbol}: L2 snapshot — "
                f"ask depth {total_ask_depth:,} shares (10 lvls), "
                f"bid depth {total_bid_depth:,} shares, "
                f"{len(df)} records in ±{window_seconds}s window"
            )
            return snapshot

        except Exception as e:
            logger.debug(f"{symbol}: L2 snapshot from {dataset} failed: {e}")
            continue

    logger.info(f"{symbol}: No L2 data available from any dataset")
    return None


def l2_to_json(snapshot: Optional[Dict]) -> Optional[str]:
    """Serialize L2 snapshot to JSON string for DB storage."""
    if snapshot is None:
        return None
    return json.dumps(snapshot)
