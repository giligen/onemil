"""Equivalence of the vectorised ``hygiene_events`` / ``_guard_frame`` with the pre-2026-10-31 implementation.

The reference below is the OLD implementation copied verbatim (per-symbol groupby.transform(pct_change)). Both are
run on the real research panel window the parity tests use and on synthetic edge panels; rows, order and dtypes must
be identical (``assert_frame_equal(check_like=False)``).
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
from trading.momentum_sleeve import (GUARD_LOOKBACK, GUARD_MAX_GAP_DAYS, GUARD_MOVE_DOWN,  # noqa: E402
                                     GUARD_MOVE_UP)

PANEL = os.path.join(ROOT, 'research', 'momentum_weekly', 'panel_2016_2026.parquet')


def _guard_frame_reference(panel, asof):
    """OLD per-bar guard features (verbatim from the pre-vectorisation hygiene_events)."""
    asof = pd.Timestamp(asof)
    sub = panel.loc[panel['bar_date'] <= asof, ['symbol', 'bar_date', 'close']]
    sub = sub[sub['close'] > 0]
    if sub.empty:
        return sub
    sub = sub.sort_values(['symbol', 'bar_date'], kind='stable').drop_duplicates(['symbol', 'bar_date'], keep='last')
    grp = sub.groupby('symbol', sort=False, observed=True)
    return sub.assign(k=grp.cumcount(ascending=False),
                      move=grp['close'].transform(lambda c: c.astype('float64').pct_change()),
                      gap_days=grp['bar_date'].diff().dt.days)


def _hygiene_events_reference(panel, asof):
    """OLD hygiene_events (verbatim, apart from reading the features from the reference frame)."""
    sub = _guard_frame_reference(panel, asof)
    if sub.empty:
        return {}
    sub = sub[sub['k'] < GUARD_LOOKBACK]
    moved = sub[(sub['move'] > GUARD_MOVE_UP) | (sub['move'] < GUARD_MOVE_DOWN)]
    gapped = sub[sub['gap_days'] > GUARD_MAX_GAP_DAYS]
    last_move = moved.groupby('symbol', observed=True)['bar_date'].max()
    last_gap = gapped.groupby('symbol', observed=True)['bar_date'].max()
    events = {}
    for sym in sorted(set(last_move.index) | set(last_gap.index)):
        in_move, in_gap = sym in last_move.index, sym in last_gap.index
        reason = 'move+gap' if in_move and in_gap else ('move' if in_move else 'gap')
        when = max(last_move[sym] if in_move else pd.Timestamp.min, last_gap[sym] if in_gap else pd.Timestamp.min)
        events[str(sym)] = (reason, when.date())
    return events


def _as_frame(events):
    """Events dict -> ordered frame so order and dtypes are compared by assert_frame_equal."""
    return pd.DataFrame([(s, r, d) for s, (r, d) in events.items()], columns=['symbol', 'reason', 'date'])


def _assert_equivalent(panel, asof):
    """Features frame and events must be identical (rows, order, dtypes) between OLD and NEW."""
    pd.testing.assert_frame_equal(ms._guard_frame(panel, asof), _guard_frame_reference(panel, asof),
                                  check_like=False)
    new, old = ms.hygiene_events(panel, asof), _hygiene_events_reference(panel, asof)
    assert list(new) == list(old)
    pd.testing.assert_frame_equal(_as_frame(new), _as_frame(old), check_like=False)
    return new


def _bars(symbol, dates, closes):
    return pd.DataFrame({'symbol': symbol, 'bar_date': pd.DatetimeIndex(dates),
                         'close': np.asarray(closes, dtype='float32'), 'volume': 1e6})


def test_equivalent_on_a_symbol_with_one_bar():
    panel = pd.concat([_bars('ONE', ['2024-03-04'], [50.0]),
                       _bars('TWO', pd.bdate_range('2024-01-01', periods=20), [10.0] * 19 + [60.0])],
                      ignore_index=True)
    assert "TWO" in _assert_equivalent(panel, "2024-03-04")


def test_equivalent_with_duplicate_symbol_date_rows():
    dates = list(pd.bdate_range('2024-01-01', periods=30))
    closes = [100.0] * 30
    closes[20] = 500.0
    a = _bars('DUP', dates, closes)
    dup = _bars('DUP', dates[18:24], [100.0, 100.0, 90.0, 400.0, 100.0, 100.0])   # same (symbol, date) again
    panel = pd.concat([a, dup, _bars('OK', dates, [20.0] * 30)], ignore_index=True)
    _assert_equivalent(panel, dates[-1])


def test_equivalent_with_nan_closes_and_non_positive():
    dates = list(pd.bdate_range('2024-01-01', periods=40))
    closes = [30.0] * 40
    closes[10] = np.nan
    closes[25] = 0.0
    closes[30] = 300.0
    other = [15.0] * 40
    other[5] = np.nan
    panel = pd.concat([_bars('NAN', dates, closes), _bars('NAN2', dates, other)], ignore_index=True)
    _assert_equivalent(panel, dates[-1])


def test_equivalent_on_a_split_day_and_a_calendar_gap():
    dates = list(pd.bdate_range('2024-01-01', periods=60))
    split = [100.0] * 30 + [25.0] * 30                       # 4:1 forward split not adjusted: -75 % exactly-ish
    reverse = [2.0] * 30 + [40.0] * 30                       # 1:20 reverse split: +1900 %
    gapped_dates = dates[:30] + [d + pd.Timedelta(days=15) for d in dates[30:]]
    panel = pd.concat([_bars('SPL', dates, split), _bars('REV', dates, reverse),
                       _bars('GAP', gapped_dates, [10.0] * 60)], ignore_index=True)
    ev = _assert_equivalent(panel, gapped_dates[-1])
    assert {'REV', 'GAP'} <= set(ev)


def test_equivalent_with_categorical_symbols_and_unsorted_rows():
    dates = list(pd.bdate_range('2024-01-01', periods=50))
    cl = [10.0] * 50
    cl[40] = 100.0
    panel = pd.concat([_bars('ZZZ', dates, cl), _bars('AAA', dates, [10.0] * 50)], ignore_index=True)
    panel = panel.sample(frac=1.0, random_state=4).reset_index(drop=True)
    _assert_equivalent(panel, dates[-1])
    panel['symbol'] = panel['symbol'].astype('category')
    _assert_equivalent(panel, dates[-1])


def test_equivalent_when_everything_is_after_asof_or_empty():
    panel = _bars('LATE', pd.bdate_range('2024-06-03', periods=5), [10.0] * 5)
    assert _assert_equivalent(panel, '2024-01-02') == {}


def _real_panel(target):
    """The research-panel window the parity tests use for rebalance ``target`` (float32 closes)."""
    import pyarrow.parquet as pq
    reb_ts = pd.Timestamp(target)
    lo = reb_ts - pd.Timedelta(days=460)
    tbl = pq.read_table(PANEL, columns=['symbol', 'bar_date', 'close', 'volume'],
                        filters=[('bar_date', '>=', lo.to_pydatetime()), ('bar_date', '<', reb_ts.to_pydatetime())])
    panel = tbl.to_pandas()
    panel['close'] = panel['close'].astype('float32')
    return panel, reb_ts - pd.Timedelta(days=3)


@pytest.mark.skipif(not os.path.exists(PANEL), reason='research panel not present')
def test_equivalent_on_the_full_real_panel_window_of_the_parity_tests():
    """Full 32k-symbol window of the newest parity target: the OLD per-symbol lambda takes ~20 s here."""
    panel, asof = _real_panel('2026-06-29')
    events = _assert_equivalent(panel, asof)
    print(f'REAL PANEL 2026-06-29 (full): {len(events)} guard events identical')


@pytest.mark.skipif(not os.path.exists(PANEL), reason='research panel not present')
@pytest.mark.parametrize('target', ['2021-02-08', '2025-12-29'])
def test_equivalent_on_a_sampled_real_panel_window(target):
    """Older parity windows: every symbol the NEW code flags, every symbol with a raw >2x/<-75 % or >10-day step
    (found by an independent numpy scan) plus a seeded 3,000-symbol sample -- keeps the reference run to ~2 s.
    The full-window equivalence of all three targets was run once when the vectorisation was written (identical)."""
    panel, asof = _real_panel(target)
    flagged = set(ms.hygiene_events(panel, asof))
    p = panel.sort_values(['symbol', 'bar_date'], kind='stable')
    c = p['close'].to_numpy().astype('float64')
    same = (p['symbol'].to_numpy()[1:] == p['symbol'].to_numpy()[:-1])
    with np.errstate(divide='ignore', invalid='ignore'):
        r = c[1:] / c[:-1] - 1.0
    d = np.diff(p['bar_date'].to_numpy()) // np.timedelta64(1, 'D')
    odd = same & ((r > 1.0) | (r < -0.5) | (d > 8))
    suspects = set(p['symbol'].to_numpy()[1:][odd])
    syms = np.array(sorted(set(panel['symbol'])))
    sample = set(np.random.default_rng(7).choice(syms, size=min(3000, len(syms)), replace=False))
    keep = flagged | suspects | sample
    events = _assert_equivalent(panel[panel['symbol'].isin(keep)], asof)
    assert set(events) == flagged, 'the subset must not change which symbols are flagged'
    print(f'REAL PANEL {target} (sampled {len(keep)} symbols): {len(events)} guard events identical')
