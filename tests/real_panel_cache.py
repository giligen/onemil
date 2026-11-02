"""Session cache of the research-panel parquet windows the momentum parity / equivalence tests read.

Every parity target reads the same ``[reb - 460 d, reb)`` slice of ``panel_2016_2026.parquet`` (3.7 M rows for the newest
one): the build-A parity, the guarded-BT parity and the two equivalence files used to decode the same slice up to three
times each (1.7 s per decode).  The decoded pyarrow ``Table`` is kept (compact: ~100 MB per window, vs ~320 MB as pandas
object strings) and every caller gets its OWN pandas frame with float32 closes, so a test that mutates its frame cannot
leak into another test.  Nothing about a window, a column or a filter changes -- only how often the file is decoded.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(ROOT, 'research', 'momentum_weekly', 'panel_2016_2026.parquet')
COLUMNS = ['symbol', 'bar_date', 'close', 'volume']
LOOKBACK_DAYS = 460

_TABLES = {}      # (lo, hi) -> pyarrow.Table, decoded once per process


def _window_table(lo: pd.Timestamp, hi: pd.Timestamp):
    """Decoded parquet slice ``bar_date in [lo, hi)``; read from disk once per process and window."""
    import pyarrow.parquet as pq
    key = (lo, hi)
    if key not in _TABLES:
        _TABLES[key] = pq.read_table(PANEL, columns=COLUMNS,
                                     filters=[('bar_date', '>=', lo.to_pydatetime()),
                                              ('bar_date', '<', hi.to_pydatetime())])
    return _TABLES[key]


def real_panel_window(reb) -> pd.DataFrame:
    """A fresh pandas frame (float32 ``close``) of the 460-day window ending (exclusive) at rebalance date ``reb``."""
    reb_ts = pd.Timestamp(reb)
    panel = _window_table(reb_ts - pd.Timedelta(days=LOOKBACK_DAYS), reb_ts).to_pandas()
    panel['close'] = panel['close'].astype('float32')
    return panel


_RAMOM = {}       # (reb, asof) -> risk_adjusted_momentum frame, computed once per process


def memoised_ramom(reb):
    """Context manager for the parity tests: ``trading.momentum_sleeve.risk_adjusted_momentum`` is computed once per
    (rebalance date, as-of) and re-served inside the block.  ``eligible_universe`` calls it again on the same panel and
    as-of that the test just passed to it directly (3 s each on the newest window); the function is deterministic and
    its result is only read, so both calls get the identical frame.  Its own correctness is pinned by
    ``test_risk_adjusted_momentum_equivalence_20261102.py`` (NEW vs the verbatim OLD function), which does not use this."""
    import contextlib
    from unittest import mock
    from trading import momentum_sleeve as ms
    real = ms.risk_adjusted_momentum

    def cached(panel, asof):
        key = (pd.Timestamp(reb), pd.Timestamp(asof))
        if key not in _RAMOM:
            _RAMOM[key] = real(panel, asof)
        return _RAMOM[key]

    @contextlib.contextmanager
    def ctx():
        with mock.patch.object(ms, 'risk_adjusted_momentum', cached):
            yield
    return ctx()
