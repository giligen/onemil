"""Equivalence of the vectorised ``risk_adjusted_momentum`` (``_grouped_pct_change``) with the pre-2026-11-02 code.

The reference below is the OLD implementation copied verbatim (``groupby('symbol')['close'].pct_change()`` over
63,666 groups, ~4 s per call). Both run on real research-panel windows the parity tests use and on synthetic edge
panels; rows, order, columns and dtypes must be identical (``assert_frame_equal(check_like=False)``).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trading import momentum_sleeve as ms  # noqa: E402
from trading.momentum_sleeve import ADV_WINDOW, HIST_MIN_ROWS_BACK, LAG_FORMATION, LAG_SKIP, VOL_WINDOW  # noqa: E402

PANEL = os.path.join(ROOT, 'research', 'momentum_weekly', 'panel_2016_2026.parquet')


def _risk_adjusted_momentum_reference(panel, asof):
    """OLD risk_adjusted_momentum (verbatim from the pre-vectorisation code)."""
    asof = pd.Timestamp(asof)
    sub = panel.loc[panel['bar_date'] <= asof, ['symbol', 'bar_date', 'close', 'volume']]
    sub = sub.sort_values(['symbol', 'bar_date'], kind='stable')
    sub = sub.drop_duplicates(['symbol', 'bar_date'], keep='last')
    grp = sub.groupby('symbol', sort=False, observed=True)
    sub = sub.assign(k=grp.cumcount(ascending=False))
    last = sub[sub['k'] == 0]
    live = set(last.loc[last['bar_date'] == asof, 'symbol'])
    if not live:
        return pd.DataFrame(columns=['close', 'adv20', 'sig_12_1', 'vol252', 'signal', 'history_ok'])
    sub = sub[sub['symbol'].isin(live)]
    sub = sub.assign(ret=sub.groupby('symbol', sort=False, observed=True)['close'].pct_change(),
                     dvol=sub['close'].astype('float64') * sub['volume'].astype('float64'))

    def at(lag):
        s = sub[sub['k'] == lag]
        return s.set_index('symbol')['close'].astype('float64')

    close0, c21, c252, c273 = at(0), at(LAG_SKIP), at(LAG_FORMATION), at(HIST_MIN_ROWS_BACK)
    win_a = sub[sub['k'] < ADV_WINDOW].groupby('symbol', observed=True)['dvol'].agg(['mean', 'count'])
    adv20 = win_a['mean'].where(win_a['count'] == ADV_WINDOW)
    win_v = sub[sub['k'] < VOL_WINDOW].groupby('symbol', observed=True)['ret'].agg(['std', 'count'])
    vol252 = win_v['std'].where(win_v['count'] == VOL_WINDOW)
    out = pd.DataFrame({'close': close0})
    out['adv20'] = adv20
    out['sig_12_1'] = c21 / c252 - 1.0
    out['vol252'] = vol252
    with np.errstate(divide='ignore', invalid='ignore'):
        out['signal'] = np.where(out['vol252'] > 0, out['sig_12_1'] / out['vol252'], np.nan)
    out['history_ok'] = c273.reindex(out.index).notna()
    return out


def _assert_equivalent(panel, asof):
    """Feature frame (index, order, columns, dtypes, values) identical between OLD and NEW."""
    new = ms.risk_adjusted_momentum(panel, asof)
    old = _risk_adjusted_momentum_reference(panel, asof)
    pd.testing.assert_frame_equal(new, old, check_like=False)
    return new


def _bars(symbol, dates, closes, dtype='float32'):
    return pd.DataFrame({'symbol': symbol, 'bar_date': pd.DatetimeIndex(dates),
                         'close': np.asarray(closes, dtype=dtype), 'volume': 1e6})


def _walk(n, seed, dtype='float32'):
    """A positive random-walk close series of n bars."""
    rng = np.random.default_rng(seed)
    return np.asarray(50 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), dtype=dtype)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_equivalent_on_long_histories_and_a_one_bar_symbol(dtype):
    dates = pd.bdate_range('2022-01-03', periods=330)
    panel = pd.concat([_bars('AAA', dates, _walk(330, 1, dtype), dtype), _bars('BBB', dates, _walk(330, 2, dtype), dtype),
                       _bars('ONE', dates[-1:], [20.0], dtype)], ignore_index=True)
    out = _assert_equivalent(panel, dates[-1])
    assert out['vol252'].notna().sum() == 2 and 'ONE' in out.index


def test_equivalent_with_duplicate_symbol_date_rows():
    dates = pd.bdate_range('2022-01-03', periods=300)
    a = _bars('DUP', dates, _walk(300, 3))
    dup = _bars('DUP', dates[100:110], _walk(10, 4))
    panel = pd.concat([a, dup, _bars('OK', dates, _walk(300, 5))], ignore_index=True)
    _assert_equivalent(panel, dates[-1])


def test_equivalent_with_nan_closes_including_a_leading_and_a_trailing_nan():
    dates = pd.bdate_range('2022-01-03', periods=300)
    c1, c2, c3 = _walk(300, 6), _walk(300, 7), _walk(300, 8)
    c1[[50, 51, 120]] = np.nan                    # a run of two NaNs: forward-filled inside the symbol
    c2[0] = np.nan                                # leading NaN
    c3[-1] = np.nan                               # NaN on the as-of bar itself
    panel = pd.concat([_bars('N1', dates, c1), _bars('N2', dates, c2), _bars('N3', dates, c3)], ignore_index=True)
    _assert_equivalent(panel, dates[-1])


def test_equivalent_on_a_split_day_and_zero_close():
    dates = pd.bdate_range('2022-01-03', periods=300)
    split = np.asarray([100.0] * 150 + [25.0] * 150, dtype='float32')
    zero = _walk(300, 9)
    zero[200] = 0.0
    panel = pd.concat([_bars('SPL', dates, split), _bars('ZER', dates, zero)], ignore_index=True)
    _assert_equivalent(panel, dates[-1])


def test_equivalent_when_a_symbol_has_gaps_in_its_history():
    dates = pd.bdate_range('2022-01-03', periods=330)
    keep = np.ones(330, dtype=bool)
    keep[40:80] = False                           # a 40-bar hole in GAP's history
    keep[[200, 201, 250]] = False
    panel = pd.concat([_bars('GAP', dates[keep], _walk(int(keep.sum()), 10)), _bars('FULL', dates, _walk(330, 11))],
                      ignore_index=True)
    _assert_equivalent(panel, dates[-1])


def test_equivalent_with_categorical_symbols_unsorted_rows_and_duplicate_index():
    dates = pd.bdate_range('2022-01-03', periods=300)
    panel = pd.concat([_bars('ZZZ', dates, _walk(300, 12)), _bars('AAA', dates, _walk(300, 13)),
                       _bars('LATE', dates[:100], _walk(100, 14))])      # index NOT reset: duplicated labels
    panel = panel.sample(frac=1.0, random_state=4)
    _assert_equivalent(panel, dates[-1])
    panel = panel.reset_index(drop=True)
    panel['symbol'] = panel['symbol'].astype('category')
    _assert_equivalent(panel, dates[-1])


def test_empty_after_asof_logs_error_and_returns_empty_frame(caplog):
    panel = _bars('LATE', pd.bdate_range('2024-06-03', periods=5), [10.0] * 5)
    new = ms.risk_adjusted_momentum(panel, '2024-01-02')
    old = _risk_adjusted_momentum_reference(panel, '2024-01-02')
    pd.testing.assert_frame_equal(new, old)
    assert any('no symbol has a bar' in r.getMessage() and r.levelname == 'ERROR' for r in caplog.records)


def _real_panel(target):
    """The research-panel window the parity tests use for rebalance ``target`` (float32 closes) and its as-of date."""
    import pyarrow.parquet as pq
    reb_ts = pd.Timestamp(target)
    lo = reb_ts - pd.Timedelta(days=460)
    tbl = pq.read_table(PANEL, columns=['symbol', 'bar_date', 'close', 'volume'],
                        filters=[('bar_date', '>=', lo.to_pydatetime()), ('bar_date', '<', reb_ts.to_pydatetime())])
    panel = tbl.to_pandas()
    panel['close'] = panel['close'].astype('float32')
    return panel, panel.loc[panel.symbol == 'SPY', 'bar_date'].max()


@pytest.mark.skipif(not os.path.exists(PANEL), reason='research panel not present')
def test_equivalent_on_the_full_real_panel_window_of_the_newest_parity_target():
    """Full 32k-symbol window of the newest parity target (the OLD groupby.pct_change takes ~5 s here)."""
    panel, asof = _real_panel('2026-09-21')
    out = _assert_equivalent(panel, asof)
    assert out['signal'].notna().sum() > 1000
    print(f'REAL PANEL 2026-09-21 (full): {len(out)} symbols identical')


@pytest.mark.skipif(not os.path.exists(PANEL), reason='research panel not present')
@pytest.mark.parametrize('target', ['2021-02-08', '2024-06-03', '2026-06-29'])
def test_equivalent_on_a_sampled_real_panel_window(target):
    """Other parity windows: a seeded 3,000-symbol sample (the per-symbol return is independent of other symbols)."""
    panel, asof = _real_panel(target)
    syms = np.array(sorted(set(panel['symbol'])))
    keep = set(np.random.default_rng(7).choice(syms, size=min(3000, len(syms)), replace=False)) | {'SPY'}
    out = _assert_equivalent(panel[panel['symbol'].isin(keep)], asof)
    print(f'REAL PANEL {target} (sampled {len(keep)} symbols): {len(out)} symbols identical')
