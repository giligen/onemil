"""Unit tests for the --batched per-leg fetch path in research/options_vrp/fetch_dbn.py
(PULL_REPLAN_20260929). Synthetic frames only -- CLIENT.metadata.get_cost and
CLIENT.timeseries.get_range are mocked, so these tests make ZERO network calls and never touch
the real opt_cache/dbn/ (LEGS_DIR and SPEND_PATH are monkeypatched to tmp_path for every test).

Covers: one get_range call per entry-Monday group, resumability (cached legs dropped from the
group before the call; a fully-cached group makes no call at all), splitting the returned
multi-symbol frame into the same per-leg parquet files fetch_legs would write, a data gap (a
requested leg absent from the returned frame) logged as WARNING and not crashing, cost-only
making no purchase, and the spend cap / spend-ledger wiring (guarded_get_range) unchanged.

Also covers the PULL_REPLAN #5 blocker fix: rebuild_leg_plan_from_mondays (the --start/--end
range filter and the "regardless of prior state" rebuild-from-parquet-rows behaviour) and
print_dry_plan / ledger_mean_cost_per_leg (the pre-purchase dry-plan line, zero network calls).
"""
import datetime as dt
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research' / 'options_vrp'))
import fetch_dbn as m


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Every test gets its own legs dir + spend ledger; the real Databento client is replaced
    with a MagicMock (external SDK object -- CLAUDE.md allows omitting spec= for these)."""
    legs_dir = tmp_path / 'legs'
    legs_dir.mkdir()
    monkeypatch.setattr(m, 'LEGS_DIR', str(legs_dir))
    monkeypatch.setattr(m, 'SPEND_PATH', str(tmp_path / 'spend.json'))
    monkeypatch.setattr(m, 'SPEND', {'total_usd': 0.0, 'purchases': [], 'skipped': []})
    mock_client = MagicMock()
    monkeypatch.setattr(m, 'CLIENT', mock_client)
    return {'legs_dir': legs_dir, 'client': mock_client}


def _leg(osi, first_seen, expiry, strike):
    return osi, {'strike': strike, 'expiry': expiry, 'first_seen': first_seen}


def _synthetic_frame(symbols):
    """One row per symbol, a 'val' column identifying which leg it came from -- lets a test
    assert the split preserved the RIGHT rows, not just the right file count."""
    return pd.DataFrame({
        'symbol': symbols,
        'ts_event': ['2016-01-04T14:30:00Z'] * len(symbols),
        'val': [f"row-for-{s}" for s in symbols],
    })


def _cached_path(state, osi):
    return os.path.join(state['legs_dir'], f"{osi.strip()}.parquet")


# --------------------------------------------------------------------------- grouping / batching
def test_one_get_range_call_per_entry_monday_group(isolated_state):
    leg_plan = dict([
        _leg('SPY   160108P00190000', '2016-01-04', '2016-02-19', 190.0),
        _leg('SPY   160108P00180000', '2016-01-04', '2016-02-19', 180.0),
        _leg('SPY   160212P00195000', '2016-02-08', '2016-03-25', 195.0),
    ])
    client = isolated_state['client']
    client.metadata.get_cost.return_value = 0.01
    client.timeseries.get_range.side_effect = lambda **kw: MagicMock(to_df=lambda: _synthetic_frame(kw['symbols']))

    m.fetch_legs_batched(cost_only=False, leg_plan=leg_plan)

    assert client.timeseries.get_range.call_count == 2, "one call per distinct (first_seen, expiry) group, not per leg"
    calls_by_symbols = [set(c.kwargs['symbols']) for c in client.timeseries.get_range.call_args_list]
    assert {'SPY   160108P00190000'.strip(), 'SPY   160108P00180000'.strip()} in calls_by_symbols
    assert {'SPY   160212P00195000'.strip()} in calls_by_symbols
    for osi, _ in leg_plan.items():
        assert os.path.exists(_cached_path(isolated_state, osi))


def test_union_window_matches_the_non_batched_single_leg_formula(isolated_state):
    """The batched window for a group must be identical to what fetch_legs would compute for
    any one leg in that group (same first_seen/expiry -> same life window by construction)."""
    osi, info = _leg('SPY   160108P00190000', '2016-01-04', '2016-02-19', 190.0)
    leg_plan = {osi: info}
    expected_start, expected_end = m._leg_window(info)
    client = isolated_state['client']
    client.metadata.get_cost.return_value = 0.01
    client.timeseries.get_range.side_effect = lambda **kw: MagicMock(to_df=lambda: _synthetic_frame(kw['symbols']))

    m.fetch_legs_batched(cost_only=False, leg_plan=leg_plan)

    kw = m.CLIENT.timeseries.get_range.call_args.kwargs
    assert kw['start'] == expected_start
    assert kw['end'] == expected_end


# --------------------------------------------------------------------------- resumability
def test_cached_leg_dropped_from_its_group_before_the_call(isolated_state):
    leg_plan = dict([
        _leg('SPY   160108P00190000', '2016-01-04', '2016-02-19', 190.0),
        _leg('SPY   160108P00180000', '2016-01-04', '2016-02-19', 180.0),
    ])
    # Pre-seed one leg as already cached, with a marker we can prove was NOT overwritten.
    pd.DataFrame({'val': ['already-here']}).to_parquet(_cached_path(isolated_state, 'SPY   160108P00190000'), index=False)

    client = isolated_state['client']
    client.metadata.get_cost.return_value = 0.01
    client.timeseries.get_range.side_effect = lambda **kw: MagicMock(to_df=lambda: _synthetic_frame(kw['symbols']))

    m.fetch_legs_batched(cost_only=False, leg_plan=leg_plan)

    kw = client.timeseries.get_range.call_args.kwargs
    assert kw['symbols'] == ['SPY   160108P00180000'.strip()], "the cached leg must not be re-requested"
    untouched = pd.read_parquet(_cached_path(isolated_state, 'SPY   160108P00190000'))
    assert untouched['val'].tolist() == ['already-here']


def test_fully_cached_group_makes_no_call_at_all(isolated_state):
    leg_plan = dict([_leg('SPY   160108P00190000', '2016-01-04', '2016-02-19', 190.0)])
    pd.DataFrame({'val': ['already-here']}).to_parquet(_cached_path(isolated_state, 'SPY   160108P00190000'), index=False)
    client = isolated_state['client']

    m.fetch_legs_batched(cost_only=False, leg_plan=leg_plan)

    client.metadata.get_cost.assert_not_called()
    client.timeseries.get_range.assert_not_called()


# --------------------------------------------------------------------------- split correctness
def test_split_writes_correct_rows_per_leg(isolated_state):
    leg_plan = dict([
        _leg('SPY   160108P00190000', '2016-01-04', '2016-02-19', 190.0),
        _leg('SPY   160108P00180000', '2016-01-04', '2016-02-19', 180.0),
    ])
    client = isolated_state['client']
    client.metadata.get_cost.return_value = 0.01
    client.timeseries.get_range.side_effect = lambda **kw: MagicMock(to_df=lambda: _synthetic_frame(kw['symbols']))

    m.fetch_legs_batched(cost_only=False, leg_plan=leg_plan)

    df190 = pd.read_parquet(_cached_path(isolated_state, 'SPY   160108P00190000'))
    df180 = pd.read_parquet(_cached_path(isolated_state, 'SPY   160108P00180000'))
    assert df190['val'].tolist() == ['row-for-SPY   160108P00190000']
    assert df180['val'].tolist() == ['row-for-SPY   160108P00180000']


def test_missing_leg_in_returned_frame_is_logged_not_crashed(isolated_state, caplog):
    leg_plan = dict([
        _leg('SPY   160108P00190000', '2016-01-04', '2016-02-19', 190.0),
        _leg('SPY   160108P00180000', '2016-01-04', '2016-02-19', 180.0),
    ])
    client = isolated_state['client']
    client.metadata.get_cost.return_value = 0.01
    # Only one of the two requested legs comes back -- a real data gap.
    client.timeseries.get_range.side_effect = lambda **kw: MagicMock(
        to_df=lambda: _synthetic_frame(['SPY   160108P00190000']))

    with caplog.at_level('WARNING'):
        m.fetch_legs_batched(cost_only=False, leg_plan=leg_plan)

    assert os.path.exists(_cached_path(isolated_state, 'SPY   160108P00190000'))
    assert not os.path.exists(_cached_path(isolated_state, 'SPY   160108P00180000'))
    assert any('no rows returned for leg' in r.message for r in caplog.records)


def test_single_leg_group_without_symbol_column(isolated_state):
    """A batch of size 1 can come back with no 'symbol' column at all on some client versions;
    there is only one possible owner for every row, so the split must still succeed."""
    leg_plan = dict([_leg('SPY   160108P00190000', '2016-01-04', '2016-02-19', 190.0)])
    client = isolated_state['client']
    client.metadata.get_cost.return_value = 0.01
    no_symbol_df = pd.DataFrame({'ts_event': ['2016-01-04T14:30:00Z'], 'val': ['row']})
    client.timeseries.get_range.return_value = MagicMock(to_df=lambda: no_symbol_df)

    m.fetch_legs_batched(cost_only=False, leg_plan=leg_plan)

    assert os.path.exists(_cached_path(isolated_state, 'SPY   160108P00190000'))


# --------------------------------------------------------------------------- cost-only / spend
def test_cost_only_makes_no_purchase_and_writes_no_files(isolated_state):
    leg_plan = dict([_leg('SPY   160108P00190000', '2016-01-04', '2016-02-19', 190.0)])
    client = isolated_state['client']
    client.metadata.get_cost.return_value = 0.01

    m.fetch_legs_batched(cost_only=True, leg_plan=leg_plan)

    client.timeseries.get_range.assert_not_called()
    assert not os.path.exists(_cached_path(isolated_state, 'SPY   160108P00190000'))
    assert m.SPEND['purchases'][-1]['dry_run'] is True


def test_spend_ledger_gets_one_entry_per_batch_call(isolated_state):
    leg_plan = dict([
        _leg('SPY   160108P00190000', '2016-01-04', '2016-02-19', 190.0),
        _leg('SPY   160212P00195000', '2016-02-08', '2016-03-25', 195.0),
    ])
    client = isolated_state['client']
    client.metadata.get_cost.return_value = 0.01
    client.timeseries.get_range.side_effect = lambda **kw: MagicMock(to_df=lambda: _synthetic_frame(kw['symbols']))

    m.fetch_legs_batched(cost_only=False, leg_plan=leg_plan)

    assert len(m.SPEND['purchases']) == 2
    assert m.SPEND['total_usd'] == pytest.approx(0.02)
    assert os.path.exists(m.SPEND_PATH), "save_spend must persist the ledger exactly as the non-batched path does"


def test_spend_cap_hit_skips_the_group_and_logs(isolated_state):
    leg_plan = dict([_leg('SPY   160108P00190000', '2016-01-04', '2016-02-19', 190.0)])
    client = isolated_state['client']
    client.metadata.get_cost.return_value = m.SPEND_CAP_USD + 1.0  # one call blows the whole cap

    m.fetch_legs_batched(cost_only=False, leg_plan=leg_plan)

    client.timeseries.get_range.assert_not_called()
    assert not os.path.exists(_cached_path(isolated_state, 'SPY   160108P00190000'))
    assert len(m.SPEND['skipped']) == 1


# --------------------------------------------------------------------------- backlog resume fix
# PULL_REPLAN_20260929 #5: fetch_mondays() only ever returns leg_plan entries for Mondays it
# freshly processes this run; once a Monday is already in mondays.parquet, its legs were silently
# dropped from every later run's leg_plan. rebuild_leg_plan_from_mondays() must rebuild leg_plan
# from mondays.parquet's own cached rows for every Monday in [start, end], with no dependence on
# whether THIS run happened to fetch that Monday fresh.
def _mondays_row(entry_date, symbol, strike, expiry):
    return {'entry_date': entry_date, 'symbol': symbol, 'strike': strike, 'expiry': expiry,
            'bid': 1.0, 'ask': 1.1, 'bid_sz': 1, 'ask_sz': 1, 'spot_10': 100.0}


def _mondays_df():
    """Rows spanning three entry Mondays: one before, one inside, one after a [2018,2020] range,
    with the 2018 Monday holding two legs -- exercises both the date filter and per-Monday
    aggregation across duplicate entry_date rows."""
    return pd.DataFrame([
        _mondays_row('2016-06-06', 'SPY   160722P00190000', 190.0, '2016-07-22'),
        _mondays_row('2018-01-08', 'SPY   180216P00250000', 250.0, '2018-02-16'),
        _mondays_row('2018-01-08', 'SPY   180216P00245000', 245.0, '2018-02-16'),
        _mondays_row('2022-03-14', 'SPY   220422P00420000', 420.0, '2022-04-22'),
    ])


def test_rebuild_filters_to_start_end_range():
    """The 2016 and 2022 rows are OUTSIDE [2018-01-01, 2018-12-31] and must be excluded; only
    the 2018-01-08 Monday's two legs come back, matching its own row count exactly."""
    leg_plan, n_mondays = m.rebuild_leg_plan_from_mondays(
        _mondays_df(), dt.date(2018, 1, 1), dt.date(2018, 12, 31))

    assert n_mondays == 1
    assert set(leg_plan) == {'SPY   180216P00250000', 'SPY   180216P00245000'}
    assert leg_plan['SPY   180216P00250000']['strike'] == 250.0
    assert leg_plan['SPY   180216P00250000']['expiry'] == '2018-02-16'
    assert leg_plan['SPY   180216P00250000']['first_seen'] == '2018-01-08'


def test_rebuild_covers_already_cached_mondays_regardless_of_prior_state():
    """No notion of 'todo' vs 'already in mondays.parquet' exists in this function at all -- it
    rebuilds from whatever rows are in mondays_df, exactly the fix for the blocker (leg_plan used
    to come back empty for any Monday not freshly processed THIS run)."""
    leg_plan, n_mondays = m.rebuild_leg_plan_from_mondays(
        _mondays_df(), dt.date(2016, 1, 1), dt.date(2023, 12, 31))

    assert n_mondays == 3
    assert len(leg_plan) == 4
    assert 'SPY   160722P00190000' in leg_plan and 'SPY   220422P00420000' in leg_plan


def test_rebuild_empty_or_missing_mondays_df_warns_and_returns_empty(caplog):
    with caplog.at_level('WARNING'):
        leg_plan, n_mondays = m.rebuild_leg_plan_from_mondays(None, dt.date(2018, 1, 1), dt.date(2018, 12, 31))
    assert leg_plan == {} and n_mondays == 0
    assert any('mondays_df empty/missing' in r.message for r in caplog.records)

    with caplog.at_level('WARNING'):
        leg_plan2, n2 = m.rebuild_leg_plan_from_mondays(
            pd.DataFrame(columns=['entry_date', 'symbol', 'strike', 'expiry']), dt.date(2018, 1, 1), dt.date(2018, 12, 31))
    assert leg_plan2 == {} and n2 == 0


# --------------------------------------------------------------------------- dry-plan line
def test_print_dry_plan_resume_by_file_skip(isolated_state):
    """legs_cached counts only OSIs whose per-leg parquet already exists under LEGS_DIR (the same
    resume-by-file rule fetch_legs_batched uses); legs_to_pull is the remainder."""
    leg_plan = dict([
        _leg('SPY   180216P00250000', '2018-01-08', '2018-02-16', 250.0),
        _leg('SPY   180216P00245000', '2018-01-08', '2018-02-16', 245.0),
    ])
    pd.DataFrame({'val': ['cached']}).to_parquet(_cached_path(isolated_state, 'SPY   180216P00250000'), index=False)
    m.SPEND['purchases'] = [
        {'cost_usd': 0.002, 'label': "OPRA.PILLAR/cbbo-1m ['SPY   160722P00190000'] a..b"},
        {'cost_usd': 0.004, 'label': "OPRA.PILLAR/cbbo-1m ['SPY   180216P00999000'] a..b"},
    ]

    stats = m.print_dry_plan(leg_plan, dt.date(2018, 1, 1), dt.date(2018, 12, 31), n_mondays=1)

    assert stats['legs_needed'] == 2
    assert stats['legs_cached'] == 1
    assert stats['legs_to_pull'] == 1
    assert stats['mean_cost_per_leg'] == pytest.approx(0.003)
    assert stats['projected_usd'] == pytest.approx(0.003)  # 1 leg to pull * mean $/leg


def test_ledger_mean_cost_per_leg_excludes_parent_and_batched_labels(isolated_state):
    """Only exactly-one-symbol cbbo-1m labels count: excludes the 'SPY.OPT' parent-symbol
    Monday-ladder overhead call and a multi-leg batched purchase (not attributable to one leg)."""
    m.SPEND['purchases'] = [
        {'cost_usd': 0.02, 'label': "OPRA.PILLAR/cbbo-1m ['SPY.OPT'] 2018-01-08..2018-01-08"},
        {'cost_usd': 0.10, 'label': "OPRA.PILLAR/cbbo-1m ['A', 'B'] 2018-01-08..2018-02-16"},
        {'cost_usd': 0.001, 'label': "OPRA.PILLAR/cbbo-1m ['SPY   180216P00250000'] 2018-01-08..2018-02-16"},
        {'cost_usd': 0.003, 'label': "OPRA.PILLAR/cbbo-1m ['SPY   180216P00245000'] 2018-01-08..2018-02-16"},
    ]

    assert m.ledger_mean_cost_per_leg() == pytest.approx(0.002)  # mean of the two single-leg rows only


def test_ledger_mean_cost_per_leg_empty_ledger_warns_and_returns_zero(isolated_state, caplog):
    m.SPEND['purchases'] = []
    with caplog.at_level('WARNING'):
        result = m.ledger_mean_cost_per_leg()
    assert result == 0.0
    assert any('no single-leg cbbo-1m purchase' in r.message for r in caplog.records)
