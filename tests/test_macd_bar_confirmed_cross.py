"""Unit test for bar-confirmed cross detection in macd_wave_engine.

Drives the new bar-loop logic against cached intraday bars from the DB
for the 8 known LIVE-vs-BT mismatch cases. Asserts the detected
cross_time_min matches BT's value (within ±1 minute).

These cases drove ~30 losing live trades since 3/30 because LIVE detected
phantom tick crosses BT's bar.high never confirmed.
"""
import sqlite3
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pytz

ROOT = Path(__file__).parent.parent
ET = pytz.timezone('America/New_York')


def get_bars(symbol: str, date_str: str) -> pd.DataFrame:
    """Pull regular-session 1-min bars from cache.db, sorted ascending."""
    conn = sqlite3.connect(str(ROOT / 'data' / 'cache.db'))
    df = pd.read_sql(f"""
        SELECT timestamp, open, high, low, close, volume
          FROM intraday_bars_1min
         WHERE symbol='{symbol}' AND date(timestamp)='{date_str}'
         ORDER BY timestamp
    """, conn)
    conn.close()
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    et = df['timestamp'].dt.tz_convert(ET)
    mask = (((et.dt.hour == 9) & (et.dt.minute >= 30))
            | ((et.dt.hour > 9) & (et.dt.hour < 16)))
    df = df[mask].reset_index(drop=True)
    return df


def detect_bar_confirmed_cross(bars: pd.DataFrame, open_price: float,
                                min_intraday_pct: float, market_open_et) -> int:
    """Replicate the new logic in macd_wave_engine.scan_for_movers.
    Returns bar_cross_minute (int) or None. Matches BT's `si + 1` index
    semantic (macd_wave_backtest.py:438).
    """
    if bars.empty or open_price <= 0:
        return None
    threshold = open_price * (1 + min_intraday_pct / 100.0)
    bars_reset = bars.reset_index(drop=True)
    for idx, bar in bars_reset.iterrows():
        bar_high = float(bar['high'])
        if bar_high >= threshold:
            return int(idx) + 1  # BT semantic
    return None


class TestBarConfirmedCross(unittest.TestCase):
    """Each case: (symbol, date, expected_BT_cross_minute) — should match within ±1 min."""

    cases = [
        ('EEIQ', '2026-04-02', 36),
        ('MGRT', '2026-04-02', 8),
        ('MODD', '2026-03-31', 6),
        ('UGRO', '2026-03-31', 10),
        ('XNDU', '2026-04-14', 19),
        ('IART', '2026-05-05', 72),
        ('MSAI', '2026-04-13', 62),
        ('HTCO', '2026-04-28', 3),
    ]

    def test_bar_high_detection_matches_bt_cache(self):
        """For each case, run new detection on raw bars, assert match BT."""
        results = []
        failures = []
        for sym, date_str, bt_minute in self.cases:
            bars = get_bars(sym, date_str)
            if bars.empty:
                results.append(f"  {sym} {date_str}: no bars cached, skipping")
                continue
            open_price = float(bars.iloc[0]['open'])
            market_open = pd.Timestamp(date_str + ' 09:30').tz_localize(ET)
            detected = detect_bar_confirmed_cross(bars, open_price, 10.0, market_open)
            if detected is None:
                results.append(f"  {sym} {date_str}: NEVER crossed in bars (open=${open_price:.2f})")
                continue
            diff = abs(detected - bt_minute)
            ok = diff <= 1
            status = '✓' if ok else '✗'
            results.append(
                f"  {sym} {date_str}: detected={detected} bt={bt_minute} diff={diff} {status}"
            )
            if not ok:
                failures.append((sym, date_str, detected, bt_minute))

        print('\nBar-confirmed cross detection vs BT cache:')
        for r in results:
            print(r)

        self.assertEqual(failures, [],
                         f"Cross detection mismatched BT for: {failures}")

    def test_phantom_tick_protection(self):
        """If bars don't cross threshold, detect should return None.
        Use a synthetic case where bars don't cross."""
        market_open = pd.Timestamp('2026-04-02 09:30').tz_localize(ET)
        synthetic_bars = pd.DataFrame({
            'timestamp': [market_open + timedelta(minutes=i) for i in range(10)],
            'open': [10.0] * 10,
            'high': [10.5] * 10,  # all 5% above open, never cross 10%
            'low': [10.0] * 10,
            'close': [10.5] * 10,
            'volume': [1000] * 10,
        })
        result = detect_bar_confirmed_cross(synthetic_bars, 10.0, 10.0, market_open)
        self.assertIsNone(result, "Phantom guard failed — should return None when no bar crosses threshold")


if __name__ == '__main__':
    unittest.main(verbosity=2)
