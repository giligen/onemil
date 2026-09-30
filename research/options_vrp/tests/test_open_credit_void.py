"""Regression test for the v3 `_OPEN_CREDIT` key-miss defect in cell_1599.py `run_cycle`.

Defect (found while reviewing RESULT_1599.md / review/1599_refuter_1.md): the stop/profit checks
used to read `_OPEN_CREDIT.get((short_sym, long_sym, entry_d), mark + 1)` /
`_OPEN_CREDIT.get(..., mark - 1)`. On any key miss this compared `mark` against a value derived
from `mark` itself rather than against the real opening credit -- for ordinary marks that made
'stop' and 'profit_target' structurally unreachable, so the cycle always fell through to
'expiry_no_trigger', which `run_cell` prices through `intrinsic_settlement` IDENTICALLY to how
Management B prices 'expiry_intrinsic'. The observed effect in RESULT_1599.md's TRAIN table (v3,
Amendment 3): cell 1599 (mgmt A) and cell 1601 (mgmt B, holds to expiry unconditionally) reported
byte-for-byte identical stats across every column -- Management A had silently collapsed into
Management B.

This test reproduces the collapse mechanism on a synthetic cycle: a mark that is KNOWN to trip
'profit_target' when `_OPEN_CREDIT` is correctly seeded (same bid/ask/credit numbers as the
sibling test `test_run_cycle_management_a_profit_target_before_dte21` in ../test_cell_1599.py,
which proves those numbers trigger 'profit_target' on the hit path), but with the key deliberately
absent. It asserts the fix: VOID for management purposes, counted in warn_counter, never a silent
fall-through to an expiry-priced outcome indistinguishable from Management B.
"""
import datetime as dt
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd  # noqa: E402

import cell_1599 as m  # noqa: E402

ET = m.ET
UTC = dt.timezone.utc


def _bar(day, hh, mm, bid, ask):
    ts = dt.datetime(day.year, day.month, day.day, hh, mm, tzinfo=ET).astimezone(UTC)
    return {'ts_event': ts, 'bid_px_00': bid, 'ask_px_00': ask}


def _write_leg(tmp_path, symbol, bars):
    df = pd.DataFrame(bars)
    df.to_parquet(os.path.join(tmp_path, f"{symbol.strip()}.parquet"), index=False)


def test_open_credit_key_miss_voids_instead_of_collapsing_to_management_b(tmp_path):
    """Same mark/credit numbers as test_cell_1599.py's profit_target test (proven to trigger
    'profit_target' when the credit key IS present) but the key is absent here -- the fix must
    VOID the cycle for management purposes, never silently fall through to an expiry outcome."""
    entry, expiry = dt.date(2024, 3, 4), dt.date(2024, 4, 19)  # 46 DTE at entry
    trigger_day = expiry - dt.timedelta(days=20)  # inside 21 DTE
    exit_day = trigger_day + dt.timedelta(days=1)
    short_bars = [_bar(trigger_day, 15, 59, 1.00, 1.20), _bar(exit_day, 10, 0, 0.05, 0.10)]
    long_bars = [_bar(trigger_day, 15, 59, 0.30, 0.50), _bar(exit_day, 10, 0, 0.01, 0.05)]
    # Unique symbols: never used by any other test, so there is no leftover _OPEN_CREDIT entry in
    # this shared module-global dict that could accidentally "rescue" the lookup.
    short_sym, long_sym = 'ZVOID_S1599', 'ZVOID_L1599'
    _write_leg(tmp_path, short_sym, short_bars)
    _write_leg(tmp_path, long_sym, long_bars)
    legcache = m.LegCache(legs_dir=str(tmp_path))
    m._OPEN_CREDIT.pop((short_sym, long_sym, entry.isoformat()), None)  # defensively ensure absent

    warn = {'mark_missing': 0, 'exit_quote_missing': 0, 'open_credit_missing': 0}
    out = m.run_cycle(legcache, short_sym, long_sym, entry.isoformat(), expiry.isoformat(), 'A', warn)

    assert out['void_reason'] == 'open_credit_missing'
    assert out['exit_date'] is None
    assert out['exit_reason'] is None
    assert out['exit_cost'] is None
    assert warn['open_credit_missing'] == 1

    # Prove the collapse target directly: Management B on the exact same inputs really does
    # resolve to the expiry-priced shape the old default silently produced for Management A too.
    out_b = m.run_cycle(legcache, short_sym, long_sym, entry.isoformat(), expiry.isoformat(), 'B',
                         {'mark_missing': 0, 'exit_quote_missing': 0, 'open_credit_missing': 0})
    assert out_b == {'exit_date': expiry.isoformat(), 'exit_reason': 'expiry_intrinsic', 'exit_cost': None}
    assert out['void_reason'] != out_b.get('void_reason')  # A is VOID, B is a real settled cycle


def test_open_credit_present_still_triggers_profit_target(tmp_path):
    """Sanity check the fix does not disturb the normal (key-present) path: same numbers as
    test_cell_1599.py's sibling test, now driven by the real credit rather than the removed
    mark-derived defaults."""
    entry, expiry = dt.date(2024, 3, 4), dt.date(2024, 4, 19)
    trigger_day = expiry - dt.timedelta(days=20)
    exit_day = trigger_day + dt.timedelta(days=1)
    short_bars = [_bar(trigger_day, 15, 59, 1.00, 1.20), _bar(exit_day, 10, 0, 0.05, 0.10)]
    long_bars = [_bar(trigger_day, 15, 59, 0.30, 0.50), _bar(exit_day, 10, 0, 0.01, 0.05)]
    short_sym, long_sym = 'ZHIT_S1599', 'ZHIT_L1599'
    _write_leg(tmp_path, short_sym, short_bars)
    _write_leg(tmp_path, long_sym, long_bars)
    legcache = m.LegCache(legs_dir=str(tmp_path))
    m._OPEN_CREDIT[(short_sym, long_sym, entry.isoformat())] = 2.0  # mark 0.70 <= 0.5*2.0=1.0

    warn = {'mark_missing': 0, 'exit_quote_missing': 0, 'open_credit_missing': 0}
    out = m.run_cycle(legcache, short_sym, long_sym, entry.isoformat(), expiry.isoformat(), 'A', warn)
    assert out['exit_reason'] == 'profit_target'
    assert warn['open_credit_missing'] == 0


def test_run_cycle_source_no_longer_defaults_open_credit_from_mark():
    """Static guard: the removed defect compared `mark` to a value derived from `mark` itself
    (`mark + 1` / `mark - 1`) as a dict-get default. Assert that pattern is gone for good and the
    explicit VOID path is present."""
    src = inspect.getsource(m.run_cycle)
    assert 'mark + 1' not in src
    assert 'mark - 1' not in src
    assert 'open_credit_missing' in src
