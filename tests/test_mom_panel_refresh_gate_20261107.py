"""mom_panel_refresh completeness gate counts REQUESTED names only (2026-11-07).

The extension parquet carries the frozen panel's name-excluded ETFs/ADRs/REITs; `sleeve_universe` never requests
them. On 11/7 13:00 UTC a complete 5,740-symbol fetch failed the gate at 80.4 % because 157 liquid ETFs in the
extension had no 11/6 bar — they were never fetched. Unit: the helper; integration: the helper on a real
parquet-shaped frame (extension + fresh bars) with the real `sleeve.liquid_symbols` threshold.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import mom_panel_refresh as mpr  # noqa: E402
import momentum_sleeve as sleeve  # noqa: E402

ASOF = date(2026, 11, 6)
SESSIONS = pd.bdate_range('2026-10-01', '2026-11-06')          # 27 weekdays, > 20 before asof


def _liquid_bars(symbol: str, through: pd.Timestamp) -> pd.DataFrame:
    """Daily bars for a name well above LIQUID_ADV (close 100 x volume 50M = $5B/day) up to `through`."""
    days = SESSIONS[SESSIONS <= through]
    return pd.DataFrame({'symbol': symbol, 'bar_date': days, 'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 50_000_000.0})


def _probe(stale_etf_through='2026-10-30', stock_through='2026-11-06'):
    etf = _liquid_bars('AGG', pd.Timestamp(stale_etf_through))      # name-excluded, never requested
    stock = _liquid_bars('AAPL', pd.Timestamp(stock_through))      # requested
    spy = _liquid_bars('SPY', pd.Timestamp('2026-11-06'))
    return pd.concat([etf, stock, spy], ignore_index=True)


class TestCompletenessForRequested:
    def test_unrequested_stale_etf_does_not_fail_the_gate(self):
        info = mpr.completeness_for_requested(['AAPL', 'SPY'], _probe(), [], ASOF)
        assert info['n_liquid'] == 2 and info['liquid_with_asof'] == 2
        assert info['liquid_ratio'] == 1.0 >= sleeve.COMPLETENESS_MIN

    def test_requested_stale_name_still_fails(self):
        info = mpr.completeness_for_requested(['AAPL', 'SPY'], _probe(stock_through='2026-11-05'), [], ASOF)
        assert info['n_liquid'] == 2 and info['liquid_with_asof'] == 1
        assert info['liquid_ratio'] < sleeve.COMPLETENESS_MIN

    def test_unfiltered_gate_reproduces_the_11_07_failure(self):
        # The pre-fix call: the stale ETF sits in the denominator and the gate fails on a complete fetch.
        info = sleeve.completeness_info(2, _probe(), [], ASOF)
        assert info['n_liquid'] == 3 and info['liquid_with_asof'] == 2
        assert info['liquid_ratio'] < sleeve.COMPLETENESS_MIN


def test_integration_parquet_roundtrip_extension_plus_fresh(tmp_path):
    """Extension parquet (through 10/30, incl. the ETF) + fresh bars for the requested names only -> gate passes."""
    ext = _probe(stale_etf_through='2026-10-30', stock_through='2026-10-30')
    p = tmp_path / 'panel_ext.parquet'
    ext.to_parquet(p, index=False)
    old = mpr.load_extension(str(p))
    fresh = pd.concat([_liquid_bars('AAPL', pd.Timestamp('2026-11-06')), _liquid_bars('SPY', pd.Timestamp('2026-11-06'))])
    fresh = fresh[fresh['bar_date'] >= pd.Timestamp('2026-10-23')]
    probe = pd.concat([old, fresh], ignore_index=True).drop_duplicates(['symbol', 'bar_date'], keep='last')
    requested = mpr.sleeve_universe(pd.DataFrame({'symbol': ['AAPL', 'AGG'], 'name': ['Apple Inc.', 'iShares Core U.S. Aggregate Bond ETF']}),
                                    tracked=set(old['symbol']))
    assert 'AGG' not in requested and 'AAPL' in requested and 'SPY' in requested
    info = mpr.completeness_for_requested(requested, probe, [], ASOF)
    assert info['liquid_ratio'] == pytest.approx(1.0)
