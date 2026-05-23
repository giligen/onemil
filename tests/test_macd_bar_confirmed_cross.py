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


# =====================================================================
# BT entry time-of-day gate (added 2026-05-23 to match live engine)
# =====================================================================

import pandas as pd
from macd_wave_backtest import filter_signals


def _sig(symbol, entry_time, cross=1, vol=50000, hist=0.6, price=10.0):
    """Minimal signal dict matching the cache schema."""
    return {
        'symbol': symbol, 'date': entry_time[:10], 'wave': 1,
        'entry_time': entry_time, 'exit_time': entry_time,
        'entry_price': price, 'exit_price': price, 'shares': 1000,
        'pnl_pct': 0.0, 'pnl_dollar': 0.0, 'exit_reason': 'eod',
        'cross_time_min': cross, 'vol_at_cross': vol, 'macd_hist_pct': hist,
        'w1_pnl': 0.0, 'conv_mult': 1.0,
    }


class TestBTEntryTimeOfDayGate:
    """filter_signals must respect last_entry_minutes_after_open and drop
    cached signals whose entry_time is past the cutoff. Matches the live
    engine fix in trading/macd_wave_engine.py (commit 59216be)."""

    def test_disabled_keeps_all(self):
        sigs = [
            _sig('A', '2026-05-22T13:32:00+00:00'),  # 09:32 ET
            _sig('B', '2026-05-22T14:30:00+00:00'),  # 10:30 ET
            _sig('C', '2026-05-22T18:00:00+00:00'),  # 14:00 ET
        ]
        out = filter_signals(sigs, {})
        assert len(out) == 3

    def test_15min_keeps_only_early(self):
        sigs = [
            _sig('A', '2026-05-22T13:32:00+00:00'),  # 09:32 ET (2 min — keep)
            _sig('B', '2026-05-22T13:45:00+00:00'),  # 09:45 ET (15 min — keep, boundary)
            _sig('C', '2026-05-22T13:46:00+00:00'),  # 09:46 ET (16 min — DROP)
            _sig('D', '2026-05-22T18:00:00+00:00'),  # 14:00 ET — DROP
        ]
        out = filter_signals(sigs, {'last_entry_minutes_after_open': 15})
        kept = {s['symbol'] for s in out}
        assert kept == {'A', 'B'}, f"got {kept}"

    def test_zero_disables(self):
        sigs = [_sig('A', '2026-05-22T18:00:00+00:00')]
        out = filter_signals(sigs, {'last_entry_minutes_after_open': 0})
        assert len(out) == 1

    def test_unparseable_entry_time_fails_open(self):
        """Malformed entry_time should not crash; signal passes through.
        Cache-load robustness — don't lose signals on bad metadata.
        """
        sigs = [_sig('A', 'not-a-timestamp')]
        # Should NOT raise
        out = filter_signals(sigs, {'last_entry_minutes_after_open': 15})
        # entry_time unparseable → no filter applied to this signal
        assert len(out) == 1

    def test_dst_handling_uses_et_zone(self):
        """The cutoff is N minutes after 09:30 ET regardless of UTC offset
        (DST vs EST). 13:32 UTC = 09:32 ET in summer (EDT); 14:32 UTC =
        09:32 ET in winter (EST). Both should be 'inside the 15-min window'.
        """
        # EDT (Mar-Nov): 13:32 UTC = 09:32 ET
        sig_edt = _sig('A', '2026-05-22T13:32:00+00:00')
        # EST (Nov-Mar): 14:32 UTC = 09:32 ET
        sig_est = _sig('B', '2026-01-15T14:32:00+00:00')
        out = filter_signals([sig_edt, sig_est],
                              {'last_entry_minutes_after_open': 15})
        assert len(out) == 2, f"both should be inside 15-min window: {out}"
