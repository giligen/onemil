#!/usr/bin/env python3
"""Unit tests for PREREG_1550: look-ahead (252d high, ADV), price-scale (raw close/open, +-30%
flag) and the top-N ranking. Run: python3 -m pytest research/overnight_high/test_cell_1550.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_panel import derive_fields, HIGH252_MIN_PERIODS  # noqa: E402
import cell_1550 as c1550  # noqa: E402


def make_symbol_days(symbol, n_days, start='2020-01-01', close=10.0, volume=1_000_000.0, open_=None):
    dates = pd.bdate_range(start, periods=n_days).strftime('%Y-%m-%d')
    return pd.DataFrame({
        'symbol': symbol, 'bar_date': dates,
        'open': open_ if open_ is not None else close,
        'high': close, 'low': close, 'close': close, 'volume': volume,
    })


class TestHigh252LookAhead:
    def test_high252_excludes_future_closes(self):
        """A close that is the highest in the WHOLE series but is at day t must not make day
        t-10's high252 reflect it -- high252 at t-10 only sees sessions strictly before t-10."""
        n = HIGH252_MIN_PERIODS + 20
        d = make_symbol_days('AAA', n, close=10.0)
        spike_idx = n - 5  # near the end
        d.loc[spike_idx, 'close'] = 999.0
        out = derive_fields(d)
        row_before_spike = out.iloc[spike_idx - 10]
        assert row_before_spike['high252'] < 999.0, (
            "high252 leaked a future close into an earlier row's rolling window")

    def test_high252_uses_strictly_prior_sessions(self):
        """high252 at row t must equal max(close[t-252:t]) i.e. exclude close[t] itself."""
        n = HIGH252_MIN_PERIODS + 5
        d = make_symbol_days('BBB', n, close=10.0)
        d.loc[n - 1, 'close'] = 500.0  # today's own close is a new high
        out = derive_fields(d)
        last = out.iloc[-1]
        assert last['high252'] < 500.0, "high252 must not include today's own close"
        assert last['close'] >= last['high252'], "today's close correctly exceeds the PRIOR high"


class TestVolRatioLookAhead:
    def test_adv20_uses_prior_volume_only(self):
        """A volume spike on day t must not appear in day t's own adv20 (denominator of vol_ratio)."""
        d = make_symbol_days('CCC', 40, volume=100_000.0)
        spike_idx = 30
        d.loc[spike_idx, 'volume'] = 10_000_000.0
        out = derive_fields(d)
        spike_row = out.iloc[spike_idx]
        assert spike_row['adv20'] < 1_000_000.0, (
            "adv20 on the spike day itself included the spike -- ADV must be through t-1 only")
        assert spike_row['vol_ratio'] > 5.0, "vol_ratio should show a large shock vs the prior ADV"

    def test_adv20_does_not_use_future_volume(self):
        d = make_symbol_days('DDD', 40, volume=100_000.0)
        d.loc[35, 'volume'] = 50_000_000.0  # future spike
        out = derive_fields(d)
        row10 = out.iloc[10]
        assert row10['adv20'] < 200_000.0, "adv20 at an early row must not see a later spike"


class TestPriceScale:
    def test_ret_on_next_uses_raw_close_and_raw_next_open(self):
        d = make_symbol_days('EEE', 5, close=10.0, open_=10.0)
        d.loc[2, 'close'] = 20.0
        d.loc[3, 'open'] = 22.0  # next day's open
        out = derive_fields(d)
        row = out.iloc[2]
        assert row['next_open'] == pytest.approx(22.0)
        assert row['ret_on_next'] == pytest.approx(22.0 / 20.0 - 1)

    def test_thirty_percent_night_is_flagged(self):
        book = pd.DataFrame({
            'bar_date': ['2024-01-02', '2024-01-03', '2024-01-04'],
            'symbol': ['AAA', 'BBB', 'CCC'],
            'ret_on_next': [0.02, 0.45, -0.02],  # BBB is a +45% night: must be flagged
        })
        n_flagged = int((book.ret_on_next.abs() > c1550.MAX_ABS_RET).sum())
        assert n_flagged == 1

    def test_no_false_flag_under_thirty_percent(self):
        rets = pd.Series([0.01, -0.02, 0.29, -0.29, 0.30])
        n_flagged = int((rets.abs() > c1550.MAX_ABS_RET).sum())
        assert n_flagged == 0  # exactly +-0.30 is NOT > 0.30


class TestTopNRanking:
    def test_top_n_picks_highest_vol_ratio_per_day(self):
        sig = pd.DataFrame({
            'bar_date': ['2024-01-02'] * 5,
            'symbol': ['A', 'B', 'C', 'D', 'E'],
            'vol_ratio': [1.5, 3.0, 2.0, 5.0, 1.7],
        })
        top3 = c1550.top_n_book(sig, 3)
        assert set(top3.symbol) == {'D', 'B', 'C'}, "top-3 by vol_ratio should be D(5.0) B(3.0) C(2.0)"

    def test_top_n_is_per_day_independent(self):
        sig = pd.DataFrame({
            'bar_date': ['2024-01-02', '2024-01-02', '2024-01-03', '2024-01-03'],
            'symbol': ['A', 'B', 'C', 'D'],
            'vol_ratio': [1.5, 9.0, 1.5, 9.0],
        })
        top1 = c1550.top_n_book(sig, 1)
        assert set(top1.symbol) == {'B', 'D'}
        assert top1.bar_date.nunique() == 2

    def test_top_n_fewer_than_n_available_keeps_all(self):
        sig = pd.DataFrame({'bar_date': ['2024-01-02'] * 2, 'symbol': ['A', 'B'],
                             'vol_ratio': [1.5, 2.0]})
        top10 = c1550.top_n_book(sig, 10)
        assert len(top10) == 2


class TestUniverseAndSignalGates:
    def test_base_universe_excludes_test_tickers(self):
        d = pd.DataFrame({
            'symbol': ['ZVZZT', 'ZZTEST', 'AAPL'], 'bar_date': ['2024-01-02'] * 3,
            'close': [10.0, 10.0, 10.0], 'dvol20': [2e7, 2e7, 2e7],
            'ret_on_next': [0.01, 0.01, 0.01],
        })
        u = c1550.base_universe(d)
        assert set(u.symbol) == {'AAPL'}

    def test_base_universe_price_and_dollar_volume_gate(self):
        d = pd.DataFrame({
            'symbol': ['CHEAP', 'THIN', 'OK'], 'bar_date': ['2024-01-02'] * 3,
            'close': [3.0, 10.0, 10.0], 'dvol20': [2e7, 5e6, 2e7],
            'ret_on_next': [0.01, 0.01, 0.01],
        })
        u = c1550.base_universe(d)
        assert set(u.symbol) == {'OK'}

    def test_signal_requires_new_high_and_volume_shock(self):
        u = pd.DataFrame({
            'symbol': ['NEWHIGH_NOSHOCK', 'NEWHIGH_SHOCK', 'NOTHIGH'], 'bar_date': ['2024-01-02'] * 3,
            'close': [10.0, 10.0, 5.0], 'high252': [9.0, 9.0, 9.0],
            'vol_ratio': [1.2, 1.5, 3.0],
        })
        sig = c1550.signal_rows(u)
        assert set(sig.symbol) == {'NEWHIGH_SHOCK'}


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
