"""Regression for study_orb_features.py incremental regen path.

The expensive part of `study_orb_features.py` is `get_intraday_bars_bulk`
— loading ~1-min bars for every (symbol, date) pair in history. The
incremental-regen refactor filters the pair list to dates >= start_date
(default: the last date in the most-recent features CSV), and merges the
new rows with the existing CSV so past trades aren't recomputed.

These tests cover the PURE merge + plan helpers without spinning up the
full feature-extraction pipeline. The end-to-end path is exercised by
the live BT run.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from study_orb_features import (
    _load_existing_features,
    _merge_new_with_existing,
    _resolve_incremental_plan,
    _atomic_write_csv,
    IncrementalSchemaDrift,
)


def _mk_existing_df():
    return pd.DataFrame([
        {'symbol': 'AAA', 'date': date(2026, 4, 20), 'entry_price': 5.1,
         'pnl': 100.0, 'pnl_pct': 2.0, 'exit_reason': 'target', 'win': 1,
         'gap_pct': 7.0, 'range_size_pct': 4.0},
        {'symbol': 'BBB', 'date': date(2026, 4, 21), 'entry_price': 10.0,
         'pnl': -50.0, 'pnl_pct': -1.0, 'exit_reason': 'stop', 'win': 0,
         'gap_pct': 5.0, 'range_size_pct': 3.0},
        {'symbol': 'CCC', 'date': date(2026, 4, 22), 'entry_price': 15.0,
         'pnl': 200.0, 'pnl_pct': 1.3, 'exit_reason': 'eod', 'win': 1,
         'gap_pct': 6.0, 'range_size_pct': 5.0},
    ])


# ---------------------------------------------------------------------------
# _merge_new_with_existing — the core correctness guarantee
# ---------------------------------------------------------------------------


class TestMergeSemantics:
    """The invariant: for every date in `new_rows`, drop existing rows on
    that date and replace with the new values. Dates NOT in new_rows pass
    through untouched (no silent data loss)."""

    def test_append_new_date_preserves_existing(self):
        existing = _mk_existing_df()
        new_rows = [{
            'symbol': 'DDD', 'date': date(2026, 4, 23), 'entry_price': 20.0,
            'pnl': 300.0, 'pnl_pct': 1.5, 'exit_reason': 'target', 'win': 1,
            'gap_pct': 8.0, 'range_size_pct': 4.5,
        }]
        merged = _merge_new_with_existing(existing, new_rows)
        assert len(merged) == 4
        dates = sorted(merged['date'].unique())
        assert dates == [date(2026, 4, 20), date(2026, 4, 21),
                         date(2026, 4, 22), date(2026, 4, 23)]

    def test_overlap_date_replaces_existing(self):
        """Post-close refresh: today's provisional row is replaced by the
        final one. Dates outside the overlap are untouched."""
        existing = _mk_existing_df()
        # Replace CCC's 4/22 row with updated values
        new_rows = [{
            'symbol': 'CCC', 'date': date(2026, 4, 22), 'entry_price': 15.0,
            'pnl': 500.0, 'pnl_pct': 3.3,  # different pnl — final vs provisional
            'exit_reason': 'eod', 'win': 1,
            'gap_pct': 6.5, 'range_size_pct': 5.1,
        }]
        merged = _merge_new_with_existing(existing, new_rows)
        assert len(merged) == 3  # same count — 4/22 replaced not appended
        row_422 = merged[merged['date'] == date(2026, 4, 22)].iloc[0]
        assert row_422['pnl'] == pytest.approx(500.0)  # new value
        # Dates before 4/22 untouched
        row_420 = merged[merged['date'] == date(2026, 4, 20)].iloc[0]
        assert row_420['pnl'] == pytest.approx(100.0)  # existing value

    def test_multi_date_new_rows_drop_all_matching_dates(self):
        """If new_rows spans multiple dates, each matching date in existing
        is dropped in one pass."""
        existing = _mk_existing_df()
        new_rows = [
            {'symbol': 'BBB', 'date': date(2026, 4, 21), 'entry_price': 10.0,
             'pnl': 999.0, 'pnl_pct': 10.0, 'exit_reason': 'target', 'win': 1,
             'gap_pct': 5.5, 'range_size_pct': 3.2},
            {'symbol': 'CCC', 'date': date(2026, 4, 22), 'entry_price': 15.0,
             'pnl': 888.0, 'pnl_pct': 5.9, 'exit_reason': 'target', 'win': 1,
             'gap_pct': 6.2, 'range_size_pct': 5.3},
        ]
        merged = _merge_new_with_existing(existing, new_rows)
        assert len(merged) == 3
        assert merged[merged['date'] == date(2026, 4, 21)].iloc[0]['pnl'] == 999.0
        assert merged[merged['date'] == date(2026, 4, 22)].iloc[0]['pnl'] == 888.0
        assert merged[merged['date'] == date(2026, 4, 20)].iloc[0]['pnl'] == 100.0

    def test_no_new_rows_returns_existing_unchanged(self):
        """The footgun case: mid-day run where no pairs qualify today
        (e.g., no daily-bar row yet, no provisional overlay). Existing CSV
        must pass through untouched — NO silent deletion of today's row
        that a prior run had added."""
        existing = _mk_existing_df()
        merged = _merge_new_with_existing(existing, [])
        assert len(merged) == 3
        assert sorted(merged['date'].unique()) == sorted(existing['date'].unique())

    def test_empty_existing_full_regen(self):
        new_rows = [
            {'symbol': 'X', 'date': date(2026, 4, 23), 'entry_price': 1.0,
             'pnl': 10.0, 'pnl_pct': 1.0, 'exit_reason': 'target', 'win': 1,
             'gap_pct': 5.0, 'range_size_pct': 2.0},
        ]
        merged = _merge_new_with_existing(pd.DataFrame(), new_rows)
        assert len(merged) == 1

    def test_column_drift_new_has_extra_feature_raises(self):
        """Post-refactor safety (I3): schema mismatch must fail loudly.
        A silent pd.concat would NaN-fill the missing column on old rows,
        which downstream dropna would silently delete."""
        existing = _mk_existing_df()
        new_rows = [{
            'symbol': 'X', 'date': date(2026, 4, 23), 'entry_price': 1.0,
            'pnl': 10.0, 'pnl_pct': 1.0, 'exit_reason': 'target', 'win': 1,
            'gap_pct': 5.0, 'range_size_pct': 2.0,
            'new_feature_added_today': 0.123,  # <- extra column
        }]
        with pytest.raises(IncrementalSchemaDrift) as excinfo:
            _merge_new_with_existing(existing, new_rows)
        msg = str(excinfo.value)
        assert 'new_feature_added_today' in msg
        assert '--force-full-regen' in msg

    def test_column_drift_new_missing_feature_raises(self):
        """Mirror case: new_rows dropped a column → old rows would keep
        their value but new rows would be NaN on concat."""
        existing = _mk_existing_df()
        new_rows = [{
            # Missing 'range_size_pct' that exists in existing_df
            'symbol': 'X', 'date': date(2026, 4, 23), 'entry_price': 1.0,
            'pnl': 10.0, 'pnl_pct': 1.0, 'exit_reason': 'target', 'win': 1,
            'gap_pct': 5.0,
        }]
        with pytest.raises(IncrementalSchemaDrift) as excinfo:
            _merge_new_with_existing(existing, new_rows)
        assert 'range_size_pct' in str(excinfo.value)

    def test_merged_is_sorted_by_date_then_symbol(self):
        existing = _mk_existing_df()
        new_rows = [
            {'symbol': 'ZZZ', 'date': date(2026, 4, 19), 'entry_price': 3.0,
             'pnl': 50.0, 'pnl_pct': 2.0, 'exit_reason': 'target', 'win': 1,
             'gap_pct': 7.0, 'range_size_pct': 2.5},
            {'symbol': 'AAA', 'date': date(2026, 4, 19), 'entry_price': 4.0,
             'pnl': 30.0, 'pnl_pct': 1.0, 'exit_reason': 'eod', 'win': 1,
             'gap_pct': 6.0, 'range_size_pct': 2.0},
        ]
        merged = _merge_new_with_existing(existing, new_rows)
        # Sorted by (date, symbol); 4/19 rows come first, AAA before ZZZ
        first_two = merged.head(2)
        assert list(first_two['date']) == [date(2026, 4, 19), date(2026, 4, 19)]
        assert list(first_two['symbol']) == ['AAA', 'ZZZ']


# ---------------------------------------------------------------------------
# _resolve_incremental_plan — start_date + existing_df selection
# ---------------------------------------------------------------------------


class TestResolveIncrementalPlan:

    def _args(self, **kwargs):
        ns = argparse.Namespace(
            force_full_regen=kwargs.get('force_full_regen', False),
            start_date=kwargs.get('start_date', None),
        )
        return ns

    def test_force_full_ignores_existing_csv(self, tmp_path, monkeypatch):
        # Drop an existing CSV in the analysis dir
        monkeypatch.setattr(
            'study_orb_features._latest_features_csv',
            lambda: None,  # we don't even read it when forcing full
        )
        existing, start = _resolve_incremental_plan(
            self._args(force_full_regen=True)
        )
        assert existing.empty
        assert start is None

    def test_explicit_start_date_overrides_auto_detect(self, monkeypatch):
        monkeypatch.setattr(
            'study_orb_features._latest_features_csv', lambda: None
        )
        existing, start = _resolve_incremental_plan(
            self._args(start_date='2026-04-15')
        )
        assert start == date(2026, 4, 15)

    def test_no_existing_csv_falls_through_to_full(self, monkeypatch):
        monkeypatch.setattr(
            'study_orb_features._latest_features_csv', lambda: None
        )
        existing, start = _resolve_incremental_plan(self._args())
        assert existing.empty
        assert start is None

    def test_existing_csv_sets_start_to_last_date(self, tmp_path, monkeypatch):
        fake_csv = tmp_path / 'orb_features_20260423_1647.csv'
        _mk_existing_df().to_csv(fake_csv, index=False)
        monkeypatch.setattr(
            'study_orb_features._latest_features_csv', lambda: str(fake_csv)
        )
        existing, start = _resolve_incremental_plan(self._args())
        assert start == date(2026, 4, 22)  # max date in _mk_existing_df
        assert len(existing) == 3


# ---------------------------------------------------------------------------
# _atomic_write_csv — crash-safety via .tmp + os.replace
# ---------------------------------------------------------------------------


class TestAtomicWrite:

    def test_writes_csv_and_cleans_tmp(self, tmp_path):
        csv_path = str(tmp_path / 'out.csv')
        df = _mk_existing_df()
        _atomic_write_csv(df, csv_path)
        assert Path(csv_path).exists()
        assert not Path(csv_path + '.tmp').exists()  # no leftover tmp
        # Round-trip
        re = pd.read_csv(csv_path)
        assert len(re) == 3

    def test_replace_overwrites_existing(self, tmp_path):
        csv_path = str(tmp_path / 'out.csv')
        pd.DataFrame([{'x': 1}]).to_csv(csv_path, index=False)
        # Now overwrite with a different DataFrame
        df = _mk_existing_df()
        _atomic_write_csv(df, csv_path)
        re = pd.read_csv(csv_path)
        assert 'symbol' in re.columns
        assert len(re) == 3


# ---------------------------------------------------------------------------
# _load_existing_features — graceful degradation
# ---------------------------------------------------------------------------


class TestLoadExisting:

    def test_none_path_returns_empty(self):
        df = _load_existing_features(None)
        assert df.empty

    def test_missing_file_returns_empty(self, tmp_path):
        df = _load_existing_features(str(tmp_path / 'does_not_exist.csv'))
        assert df.empty

    def test_well_formed_csv_loaded(self, tmp_path):
        p = tmp_path / 'ok.csv'
        _mk_existing_df().to_csv(p, index=False)
        df = _load_existing_features(str(p))
        assert len(df) == 3
        # date column normalized to datetime.date
        assert df['date'].iloc[0] == date(2026, 4, 20)

    def test_csv_missing_date_column_returns_empty(self, tmp_path):
        p = tmp_path / 'bad.csv'
        pd.DataFrame([{'symbol': 'X', 'close': 1.0}]).to_csv(p, index=False)
        df = _load_existing_features(str(p))
        assert df.empty
