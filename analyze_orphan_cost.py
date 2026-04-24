#!/usr/bin/env python3
"""Quantify the dollar cost of the _gc_stale_pending silent-purge bug.

For each orphan-suspect trade (exit_reason in bracket/sync/force_close_startup
AND entry telemetry missing), reconstruct 1-min bars from Alpaca and compute
when the MACD histogram first flipped negative after entry. Compare the
theoretical macd_flip exit P&L against the actual bracket-SL/sync P&L.

Caveats:
  - macd_flip exit fires at the NEXT poll after bar close, so theoretical
    exit price is the bar-close bid ≈ close × (1 - 0.001) for spread.
  - We use current-day MACD (fast=12, slow=26, signal=9) with warm-up from
    prior-day bars where available.
  - For orphaned WINNERS (LUNL, CDNA, SKYQ), the actual path may have been
    better or worse than macd_flip — we report the delta either way.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import yaml
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd

sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env')

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

_key = os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID')
_secret = os.getenv('ALPACA_API_SECRET') or os.getenv('APCA_API_SECRET_KEY')
if not _key:
    with open('config.yaml') as f: cfg = yaml.safe_load(f)
    _key, _secret = cfg['alpaca']['api_key'], cfg['alpaca']['api_secret']
CLIENT = StockHistoricalDataClient(_key, _secret)

# Actual fill times from Alpaca order history (critical — using bar index
# from price range alone is wrong since price may be visited repeatedly).
# Format: (date, symbol) → fill_time_utc_hhmm (str).
ACTUAL_FILLS = {
    # Apr 2 orphan-suspect trades — fill times from Alpaca order history
    # (to be filled in via order query below)
    ('2026-04-02', 'LUNL'): None,  # will look up
    ('2026-04-02', 'EEIQ'): None,
    ('2026-04-02', 'CORD'): None,
    ('2026-04-02', 'MGRT'): None,
    ('2026-04-02', 'EDSA'): None,
    ('2026-04-02', 'ANL'):  None,
    ('2026-04-08', 'SKYQ'): None,
    ('2026-04-13', 'BBGI'): None,
    ('2026-04-16', 'BBGI'): '14:23',  # known from earlier reconstruction
    ('2026-04-16', 'CDNA'): '14:26',  # known from earlier reconstruction
}

MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
SPREAD_BPS_APPROX = 0.001  # 10bps — conservative; live spreads were much worse
# Orphan signature: all entry telemetry missing. From prior audit query.
# (Format: (trade_date, symbol, entry_price, actual_exit, actual_pnl, notes))
ORPHANS = [
    ('2026-04-02', 'LUNL', 16.53, 18.91, +7189, 'winner — sync_reconcile'),
    ('2026-04-02', 'EEIQ', 8.72,  8.00,  -4123, 'loser — sync_reconcile'),
    ('2026-04-02', 'CORD', 14.19, 14.06, -458,  'loser — sync_reconcile'),
    ('2026-04-02', 'MGRT', 13.30, 11.99, -4920, 'loser — sync_reconcile'),
    ('2026-04-02', 'EDSA', 6.28,  6.20,  -636,  'loser — sync_reconcile'),
    ('2026-04-02', 'ANL',  8.90,  8.81,  -505,  'loser — sync_reconcile'),
    ('2026-04-08', 'SKYQ', 6.86,  6.94,  +583,  'winner — bracket_exit'),
    ('2026-04-13', 'BBGI', 10.58, 10.14, -2062, 'loser — bracket_sl_tp'),
    ('2026-04-16', 'BBGI', 15.51, 14.61, -4604, 'loser — bracket_sl_tp'),
    ('2026-04-16', 'CDNA', 21.43, 21.66, +805,  'winner — force_close_startup'),
]


def fetch_bars(symbol: str, date_str: str) -> Optional[pd.DataFrame]:
    dt = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    # Include prior session for MACD warm-up (+-1 day padding)
    start = dt - timedelta(days=3)
    end = dt + timedelta(days=1)
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol], timeframe=TimeFrame.Minute,
            start=start, end=end,
        )
        bars = CLIENT.get_stock_bars(req)
        if not bars.df.index.size:
            return None
        df = bars.df.reset_index()
        df = df[df['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"  WARN: fetch failed for {symbol} on {date_str}: {e}")
        return None


def lookup_fill_time_from_alpaca(symbol: str, date_str: str) -> Optional[str]:
    """Query Alpaca order history for the actual buy-fill timestamp."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    tc = TradingClient(_key, _secret, paper=True)
    try:
        d = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL, symbols=[symbol],
            after=d, until=d + timedelta(days=1), limit=50,
        )
        orders = tc.get_orders(filter=req)
        for o in orders:
            side = getattr(o.side, 'value', str(o.side))
            status = getattr(o.status, 'value', str(o.status))
            if side == 'buy' and status == 'filled' and o.filled_at:
                return o.filled_at.strftime('%H:%M')
    except Exception as e:
        print(f"  WARN: Alpaca order lookup failed for {symbol}/{date_str}: {e}")
    return None


def find_entry_bar_idx(df: pd.DataFrame, date_str: str, symbol: str,
                       entry_price: float) -> Optional[int]:
    """Locate the bar at/after the ACTUAL fill time (from Alpaca order history).

    Falls back to price-range search only if fill time cannot be recovered.
    """
    fill_hhmm = ACTUAL_FILLS.get((date_str, symbol))
    if fill_hhmm is None:
        # Look up from Alpaca and cache
        fill_hhmm = lookup_fill_time_from_alpaca(symbol, date_str)
        ACTUAL_FILLS[(date_str, symbol)] = fill_hhmm

    if fill_hhmm:
        target = pd.Timestamp(f"{date_str} {fill_hhmm}:00", tz='UTC')
        day = df[(df['timestamp'] >= target) &
                 (df['timestamp'].dt.strftime('%Y-%m-%d') == date_str)]
        if len(day):
            return day.index[0]

    # Fallback — first bar on the day matching price (less reliable)
    day = df[df['timestamp'].dt.strftime('%Y-%m-%d') == date_str]
    for idx in day.index:
        r = df.iloc[idx]
        if r['low'] <= entry_price <= r['high']:
            return idx
    return day.index[0] if len(day) else None


def macd_histogram(close: pd.Series) -> pd.Series:
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return macd - signal


def find_flip(hist: pd.Series, from_idx: int, max_idx: int) -> Optional[int]:
    """First bar AT OR AFTER from_idx (exclusive of entry bar itself + 1) where hist <= 0."""
    for i in range(from_idx + 1, min(max_idx + 1, len(hist))):
        if hist.iloc[i] <= 0:
            return i
    return None


def analyze_one(trade: Tuple) -> dict:
    date_str, symbol, entry_price, actual_exit, actual_pnl, notes = trade
    df = fetch_bars(symbol, date_str)
    if df is None:
        return {'symbol': symbol, 'date': date_str, 'status': 'NO_BARS'}

    hist = macd_histogram(df['close'])
    df['hist'] = hist

    entry_idx = find_entry_bar_idx(df, date_str, symbol, entry_price)
    if entry_idx is None:
        return {'symbol': symbol, 'date': date_str, 'status': 'NO_ENTRY_BAR'}

    # Cap search to same-day bars only (position would have been force-closed at EOD).
    day_mask = df['timestamp'].dt.strftime('%Y-%m-%d') == date_str
    day_end_idx = df[day_mask].index.max()

    flip_idx = find_flip(hist, entry_idx, day_end_idx)

    if flip_idx is None:
        # MACD never flipped on the day — position would have been force-closed at EOD
        eod_close = float(df.iloc[day_end_idx]['close'])
        theoretical_exit = eod_close * (1 - SPREAD_BPS_APPROX * 0.5)
        theoretical_pnl = (theoretical_exit - entry_price) * (actual_pnl / (actual_exit - entry_price)) if (actual_exit - entry_price) != 0 else 0
        # approximate shares from actual: shares = pnl / (exit - entry)
        if actual_exit == entry_price:
            shares = None
        else:
            shares = actual_pnl / (actual_exit - entry_price)
        theoretical_pnl = (theoretical_exit - entry_price) * shares if shares else 0
        return {
            'symbol': symbol, 'date': date_str, 'status': 'NO_FLIP_EOD',
            'entry_price': entry_price, 'actual_exit': actual_exit,
            'actual_pnl': actual_pnl, 'theoretical_pnl': theoretical_pnl,
            'flip_time': None, 'flip_close': None,
            'delta': theoretical_pnl - actual_pnl, 'notes': notes,
        }

    flip_bar = df.iloc[flip_idx]
    flip_close = float(flip_bar['close'])
    theoretical_exit = flip_close * (1 - SPREAD_BPS_APPROX * 0.5)  # mid-to-bid approx
    # Infer shares from actual trade (robust to partial fills)
    if actual_exit == entry_price:
        shares = None
        theoretical_pnl = 0
    else:
        shares = actual_pnl / (actual_exit - entry_price)
        theoretical_pnl = (theoretical_exit - entry_price) * shares
    return {
        'symbol': symbol, 'date': date_str, 'status': 'OK',
        'entry_price': entry_price, 'actual_exit': actual_exit,
        'actual_pnl': actual_pnl,
        'flip_time': flip_bar['timestamp'].strftime('%H:%M'),
        'flip_close': flip_close,
        'theoretical_exit': theoretical_exit,
        'theoretical_pnl': theoretical_pnl,
        'delta': theoretical_pnl - actual_pnl,
        'notes': notes,
    }


def main() -> None:
    print(f"{'Date':<11} {'Symbol':<7} {'Entry':>8} {'Actual$':>9} {'Flip':<6} {'FlipClose':>10} {'Hyp$':>10} {'Δ(bug$)':>10}  Notes")
    print('-' * 110)
    total_delta = 0
    ok_count = 0
    for trade in ORPHANS:
        r = analyze_one(trade)
        if r['status'] == 'OK':
            ok_count += 1
            delta = r['delta']
            total_delta += delta
            print(f"{r['date']:<11} {r['symbol']:<7} ${r['entry_price']:>6.2f}  "
                  f"${r['actual_pnl']:>+7,.0f}  {r['flip_time']:<6} ${r['flip_close']:>8.2f}  "
                  f"${r['theoretical_pnl']:>+8,.0f}  ${delta:>+8,.0f}  {r['notes']}")
        elif r['status'] == 'NO_FLIP_EOD':
            delta = r['delta']
            total_delta += delta
            print(f"{r['date']:<11} {r['symbol']:<7} ${r['entry_price']:>6.2f}  "
                  f"${r['actual_pnl']:>+7,.0f}  EOD    (no flip)   "
                  f"${r['theoretical_pnl']:>+8,.0f}  ${delta:>+8,.0f}  {r['notes']}")
        else:
            print(f"{r['date']:<11} {r['symbol']:<7} ${r['entry_price']:>6.2f}  "
                  f"${r['actual_pnl']:>+7,.0f}  [{r['status']}]")

    print('-' * 110)
    losers = [t for t in ORPHANS if t[4] < 0]
    winners = [t for t in ORPHANS if t[4] > 0]
    loser_actual = sum(t[4] for t in losers)
    winner_actual = sum(t[4] for t in winners)
    print(f"\nACTUAL: {len(losers)} losers (${loser_actual:+,.0f}) + {len(winners)} winners "
          f"(${winner_actual:+,.0f}) = ${loser_actual + winner_actual:+,.0f} net")
    print(f"HYPOTHETICAL (if macd_flip had worked): sum of theoretical P&Ls + total Δ")
    print(f"\n*** BUG COST ESTIMATE: ${total_delta:+,.0f} ***")
    print(f"    (positive = macd_flip would have beat the orphan outcome)")
    print(f"    analyzed {ok_count}/{len(ORPHANS)} trades")


if __name__ == '__main__':
    main()
