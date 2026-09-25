"""Bars-source / ATR-source override + missing-bars gate (2026-09-25,
research/orb_verify/SPEC_RESIM_FIX.md). Defect: study_orb_pipeline_static_lock
sourced minute bars and daily ATR14 ONLY from data/cache.db (no coverage
before 2025-01-02); a symbol-day with no bars SILENTLY kept the legacy
features-CSV pnl (2R-target/range-low-stop/time-stop) instead of being
excluded, which is a different unvalidated exit rule. Fix: ORB_BT_BARS_DB /
ORB_BT_DAILY_SOURCE overrides, ERROR-logged exclusion, and a >2% missing-bars
gate that exits non-zero unless ORB_BT_ALLOW_MISSING_BARS=1."""
from __future__ import annotations

import importlib
import sqlite3

import pandas as pd
import pytest

import study_orb_pipeline_static_lock as pipe


class TestMissingBarsGate:
    def test_under_threshold_does_not_exceed(self):
        assert pipe.missing_bars_gate_exceeded(1, 100) is False  # 1% < 2%

    def test_over_threshold_exceeds(self):
        assert pipe.missing_bars_gate_exceeded(3, 100) is True  # 3% > 2%

    def test_exactly_at_threshold_does_not_exceed(self):
        assert pipe.missing_bars_gate_exceeded(2, 100) is False  # == 2%, not >

    def test_override_suppresses(self):
        assert pipe.missing_bars_gate_exceeded(50, 100, allow_override=True) is False

    def test_zero_entered_never_exceeds(self):
        assert pipe.missing_bars_gate_exceeded(0, 0) is False


class TestLoadBarsBulkResearchSchema:
    """bars(symbol, day, t, o, h, l, c, v) — research/orb_2023 & orb_2024 schema
    (research/orb_2023/fetch_minutes.py), distinct from cache.db's
    intraday_bars_1min."""

    def _make_db(self, tmp_path):
        db_path = tmp_path / 'bars.db'
        con = sqlite3.connect(db_path)
        con.execute('CREATE TABLE bars (symbol TEXT, day TEXT, t TEXT, o REAL, '
                    'h REAL, l REAL, c REAL, v REAL)')
        con.execute("INSERT INTO bars VALUES ('AAA','2023-05-01','2023-05-01T13:30:00Z',"
                    "1.0,2.0,0.5,1.5,1000)")
        con.commit()
        con.close()
        return str(db_path)

    def test_present_symbol_day_loaded_and_mapped(self, tmp_path):
        db_path = self._make_db(tmp_path)
        out = pipe._load_bars_bulk([('AAA', '2023-05-01'), ('BBB', '2023-05-01')], db_path)
        assert ('AAA', '2023-05-01') in out
        assert ('BBB', '2023-05-01') not in out  # missing bars -> absent from the dict
        row = out[('AAA', '2023-05-01')][0]
        assert row['open'] == 1.0 and row['high'] == 2.0 and row['low'] == 0.5
        assert row['close'] == 1.5 and row['timestamp'] == '2023-05-01T13:30:00Z'

    def test_neither_table_present_raises(self, tmp_path):
        db_path = tmp_path / 'empty.db'
        sqlite3.connect(db_path).close()
        with pytest.raises(SystemExit):
            pipe._load_bars_bulk([('AAA', '2023-05-01')], str(db_path))

    def test_empty_pairs_short_circuits(self, tmp_path):
        db_path = self._make_db(tmp_path)
        assert pipe._load_bars_bulk([], db_path) == {}


class TestBuildAtr14LookupDailySource:
    def _parquet(self, tmp_path):
        rows = []
        for i in range(20):
            d = f'2023-01-{i+1:02d}'
            rows.append({'bar_date': d, 'symbol': 'AAA', 'open': 10, 'high': 10.5,
                         'low': 9.5, 'close': 10.0, 'volume': 1000})
        p = tmp_path / 'daily.parquet'
        pd.DataFrame(rows).to_parquet(p)
        return str(p)

    def test_parquet_source_computes_atr(self, tmp_path):
        pq = self._parquet(tmp_path)
        out = pipe.build_atr14_lookup([('AAA', '2023-01-20')], daily_source=pq)
        assert out[('AAA', '2023-01-20')] is not None
        assert out[('AAA', '2023-01-20')] == pytest.approx(1.0, abs=0.01)

    def test_parquet_source_missing_symbol_is_none(self, tmp_path):
        pq = self._parquet(tmp_path)
        out = pipe.build_atr14_lookup([('ZZZ', '2023-01-20')], daily_source=pq)
        assert out[('ZZZ', '2023-01-20')] is None

    def test_sqlite_default_unchanged(self, tmp_path):
        db_path = tmp_path / 'cache.db'
        con = sqlite3.connect(db_path)
        con.execute('CREATE TABLE daily_bars (symbol TEXT, bar_date TEXT, high REAL, '
                    'low REAL, close REAL)')
        for i in range(20):
            con.execute("INSERT INTO daily_bars VALUES ('AAA', ?, 10.5, 9.5, 10.0)",
                        (f'2023-01-{i+1:02d}',))
        con.commit(); con.close()
        out = pipe.build_atr14_lookup([('AAA', '2023-01-20')], db_path=str(db_path))
        assert out[('AAA', '2023-01-20')] == pytest.approx(1.0, abs=0.01)


class TestEnvOverrideHonoured:
    def test_orb_bt_bars_db_env_read_at_import(self, tmp_path, monkeypatch):
        monkeypatch.setenv('ORB_BT_BARS_DB', str(tmp_path / 'custom.db'))
        importlib.reload(pipe)
        try:
            assert pipe.BARS_DB_PATH == str(tmp_path / 'custom.db')
        finally:
            monkeypatch.delenv('ORB_BT_BARS_DB', raising=False)
            importlib.reload(pipe)  # restore default for other tests in the session
