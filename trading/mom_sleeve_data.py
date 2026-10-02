"""Run-time data access for the weekly momentum sleeve: Alpaca asset roster and batched daily bars.

Bars: adjustment='all', SIP, batched 200 symbols per request with the per-symbol invalid-symbol strip/retry
from research/momentum_weekly/1700c_fetch.py (an invalid symbol is dropped only when the API NAMES it, one at
a time -- never the blind whole-batch removal that once lost NVDA/AAPL). Every loss is returned, not hidden.
"""
import logging
import re
import time
from datetime import date
from typing import Dict, List, Tuple

import pandas as pd

from trading.mom_sleeve_select import BAR_COLS, is_excluded

logger = logging.getLogger('mom_sleeve_data')
BATCH_SIZE = 200
INVALID_SYM_RE = re.compile(r'invalid symbol:\s*([^"\s]+)')


def fetch_asset_roster(alpaca_client) -> pd.DataFrame:
    """Active US-equity assets: symbol, name, exchange, tradable, fractionable (one trading-API call)."""
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus
    req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    assets = alpaca_client._call_with_timeout(lambda: alpaca_client.trading_client.get_all_assets(req),
                                              'mom_get_assets', timeout=120)
    rows = [dict(symbol=a.symbol, name=a.name or '', exchange=str(getattr(a.exchange, 'value', a.exchange)),
                 tradable=bool(a.tradable), fractionable=bool(getattr(a, 'fractionable', False)))
            for a in assets]
    df = pd.DataFrame(rows, columns=['symbol', 'name', 'exchange', 'tradable', 'fractionable'])
    logger.info("mom_sleeve_data: asset roster %d active us_equity (%d tradable, %d fractionable)",
                len(df), int(df.tradable.sum()), int(df.fractionable.sum()))
    return df


def candidate_symbols(roster: pd.DataFrame) -> List[str]:
    """Symbols worth fetching bars for: tradable, not OTC, not name/test/CUSIP-excluded (reference regex)."""
    r = roster[roster.tradable & (roster.exchange != 'OTC')]
    return sorted(s for s, n in zip(r.symbol, r.name) if not is_excluded(s, n))


def _bars_to_df(barset) -> pd.DataFrame:
    """BarSet -> flat frame with BAR_COLS (bar_date = tz-naive normalized timestamp)."""
    raw = barset.df
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=BAR_COLS)
    df = raw.reset_index().rename(columns={'timestamp': 'bar_date'})
    df['bar_date'] = pd.to_datetime(df['bar_date']).dt.tz_localize(None).dt.normalize()
    return df[BAR_COLS]


def _fetch_batch(alpaca_client, batch: List[str], start: date, end: date, idx: int, max_strip: int = 100):
    """One batch with strip-only-the-named-invalid-symbol retry. Returns (frame, removed_symbols)."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    remaining, removed = list(batch), []
    for _ in range(max_strip):
        if not remaining:
            break
        req = StockBarsRequest(symbol_or_symbols=remaining, timeframe=TimeFrame.Day, start=pd.Timestamp(start),
                               end=pd.Timestamp(end), adjustment=Adjustment.ALL, feed=DataFeed.SIP)
        try:
            barset = alpaca_client._call_with_timeout(lambda r=req: alpaca_client.data_client.get_stock_bars(r),
                                                      f'mom_fetch_batch_{idx}', timeout=300,
                                                      timeout_retries=2, rate_limit_retries=6)
            return _bars_to_df(barset), removed
        except Exception as e:
            m = INVALID_SYM_RE.search(str(e))
            bad = m.group(1) if m else None
            if bad and bad in remaining:
                remaining.remove(bad)
                removed.append(bad)
                logger.warning("mom_sleeve_data: batch %d: API rejected %s as invalid -- stripped, retrying the "
                               "other %d", idx, bad, len(remaining))
                continue
            raise
    logger.error("mom_sleeve_data: batch %d exceeded %d strips", idx, max_strip)
    return pd.DataFrame(columns=BAR_COLS), removed + remaining


def fetch_bars(alpaca_client, symbols: List[str], start: date, end: date) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Daily bars for `symbols` over [start, end]. Returns (frame, lost) where lost maps symbol -> reason
    (invalid symbol, failed batch). A failed batch logs ERROR and its symbols go to `lost`."""
    frames, lost = [], {}
    batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
    t0 = time.time()
    for i, batch in enumerate(batches, 1):
        try:
            df, removed = _fetch_batch(alpaca_client, batch, start, end, i)
            for s in removed:
                lost[s] = 'invalid_symbol_api_rejected'
            if len(df):
                frames.append(df)
        except Exception as e:
            logger.error("mom_sleeve_data: batch %d/%d FAILED (%s) -- %d symbols lost", i, len(batches), e, len(batch))
            for s in batch:
                lost[s] = f'batch_failed: {e}'
        if i % 5 == 0 or i == len(batches):
            logger.info("mom_sleeve_data: fetched %d/%d batches, %d lost, %.0fs", i, len(batches), len(lost),
                        time.time() - t0)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BAR_COLS)
    return out, lost
