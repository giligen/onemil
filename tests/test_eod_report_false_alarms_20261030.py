"""EOD / green-check false alarms (spec docs/eod_report_false_alarms_spec_20261030.md).

1. fill-parity: a (symbol, day) is FILLED if ANY of its ORB rows filled (preplace writes a provisional row first).
2. selection parity: PREPLACE / ENTRY SUBMITTED journal lines count as engine-ranked / engine-ordered.
3. hygiene: trader NOT ACTIVE only inside the live window; macd-wave only when its unit file exists.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
import eod_report  # noqa: E402
import eod_sections as es  # noqa: E402
import report_common as rc  # noqa: E402

DAY = '2026-10-29'


def _row(sym, status, fill=None, qty=100, strategy='orb'):
    return {'symbol': sym, 'strategy': strategy, 'order_status': status, 'fill_price': fill, 'quantity_planned': qty,
            'exit_price': None, 'exit_reason': None}


# ------------------------------------------------------------------ defect 1: fill parity
class TestFillParity:
    def test_provisional_rows_then_filled_row_is_ok(self):
        rows = [_row('CORD', 'pending_new'), _row('CORD', 'stale_closed'), _row('CORD', 'filled', 4.78)]
        assert rc.unfilled_bt_filled_symbols(rows, {'CORD'}) == []

    def test_single_time_stop_canceled_row_is_flagged(self):
        rows = [_row('IREX', 'time_stop_canceled')]
        assert rc.unfilled_bt_filled_symbols(rows, {'IREX'}) == ['IREX']

    def test_closed_status_without_fill_price_counts_filled(self):
        assert rc.unfilled_bt_filled_symbols([_row('X', 'closed')], {'X'}) == []

    def test_non_bt_filled_and_other_strategy_ignored(self):
        rows = [_row('A', 'canceled'), _row('B', 'canceled', strategy='hod_break')]
        assert rc.unfilled_bt_filled_symbols(rows, {'B'}) == []

    def test_one_filled_symbol_does_not_hide_another_unfilled(self):
        rows = [_row('CORD', 'stale_closed'), _row('CORD', 'filled', 4.78), _row('IREX', 'time_stop_canceled')]
        assert rc.unfilled_bt_filled_symbols(rows, {'CORD', 'IREX'}) == ['IREX']

    def test_verdict_integration_real_sqlite_1029_shape(self, tmp_path, monkeypatch):
        """Temp sqlite with the 10/29 row shape read by the REAL load_live_rows -> green_verdict."""
        (tmp_path / 'data').mkdir()
        con = sqlite3.connect(tmp_path / 'data' / 'trades.db')
        con.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, trade_date TEXT, symbol TEXT, strategy TEXT, "
                    "order_status TEXT, fill_price REAL, exit_price REAL, exit_reason TEXT, quantity_planned INTEGER)")
        con.executemany("INSERT INTO trades (trade_date,symbol,strategy,order_status,fill_price,quantity_planned) "
                        "VALUES (?,?,?,?,?,?)",
                        [(DAY, 'CORD', 'orb', 'stale_closed', None, 1053), (DAY, 'CORD', 'orb', 'filled', 4.78, 697)])
        con.commit(); con.close()
        monkeypatch.setattr(rc, 'ROOT', tmp_path)
        monkeypatch.setattr(rc, 'load_bt_selected', lambda day: [{'symbol': 'CORD', 'entered': 1}])
        monkeypatch.setattr(rc, 'bt_data_max_date', lambda: '2099-01-01')
        monkeypatch.setattr(rc, 'journal_grep', lambda pat, day: [])
        monkeypatch.setattr(rc, 'read_selection_audit', lambda day: [])
        v = rc.green_verdict(DAY)
        assert v['checks']['fill_parity'] == 'OK'
        assert not any('fill-parity' in r for r in v['reasons'])


# ------------------------------------------------------------------ defect 2: preplace-aware selection parity
PRE_LOG = "\n".join([
    "Oct 29 13:34:57 h onemil-trader[1]: 2026-10-29 13:34:57 | INFO     | trading.orb_engine:4304 | "
    "[ORB PREPLACE] provisional top-2 @ 09:34:57.716 ET: IREZ(rh=$18.63,Q5), CORD(rh=$4.73,Q3)",
    "Oct 29 13:35:05 h onemil-trader[1]: 2026-10-29 13:35:05 | INFO     | trading.orb_engine:5939 | "
    "ORB ENTRY SUBMITTED: CORD order=f28e qty=1053 entry=$4.74 stop=$4.57 trade_id=647",
    "Oct 29 13:35:05 h onemil-trader[1]: 2026-10-29 13:35:05 | INFO     | trading.orb_engine:5939 | "
    "ORB ENTRY SUBMITTED: IREZ order=9eae qty=267 entry=$18.69 stop=$17.98 trade_id=648",
    "Oct 29 13:35:09 h onemil-trader[1]: 2026-10-29 13:35:09 | INFO     | trading.orb_engine:4495 | "
    "[ORB PREPLACE] reconcile final top-8 of 8 ranked: ['DAMD', 'PWP', 'IREZ', 'CORD'] | preplaced: ['IREZ', 'CORD']",
    "Oct 29 13:35:12 h onemil-trader[1]: 2026-10-29 13:35:12 | INFO     | trading.orb_engine:5939 | "
    "ORB ENTRY SUBMITTED: AXTQ order=26ed qty=116 entry=$28.50 stop=$26.38 trade_id=651",
    "Oct 29 13:35:08 h onemil-trader[1]: 2026-10-29 13:35:08 | INFO     | trading.orb_engine:3736 | "
    "ORB SCORED: DAMD comp=0.3 Q4 | gap=5 pool=production",
    "Oct 29 13:35:08 h onemil-trader[1]: 2026-10-29 13:35:08 | INFO     | trading.orb_engine:3736 | "
    "ORB SCORED: PWP comp=0.2 Q3 | gap=5 pool=production",
])


class TestSelectionParityPreplace:
    def test_parse_collects_preplace_and_submitted(self):
        p = es.parse_orb_log(PRE_LOG)
        assert {'IREZ', 'CORD', 'DAMD', 'PWP'} <= set(p['preplace_ranked'])
        assert set(p['submitted']) == {'CORD', 'AXTQ', 'IREZ'}

    def test_engine_lines_filter_keeps_entry_submitted(self):
        assert es._is_engine_line("x ORB ENTRY SUBMITTED: CORD order=1")

    def test_preplaced_names_are_engine_ranked_not_bt_only_and_never_why_not_ordered(self):
        parsed = es.parse_orb_log(PRE_LOG)
        bt_rank = {"n": 4, "ranked": ["DAMD", "PWP", "IREZ", "CORD"], "picks": ["IREZ", "CORD", "AXTQ"],
                   "comp": {}, "rows": 20, "pdr": 0, "g1": 0, "range": 0, "dedup": 0}
        rows = [{"symbol": "CORD", "order_status": "filled", "fill_price": 4.78},
                {"symbol": "IREZ", "order_status": "filled", "fill_price": 18.9},
                {"symbol": "AXTQ", "order_status": "filled", "fill_price": 28.5}]
        lines, _m = es.orb_parity_lines(DAY, rows, parsed, None, "", bt_rank)
        ranked = next(ln for ln in lines if ln.startswith("ORB ranked:"))
        assert "BT-only: -" in ranked, ranked
        assert not any(ln.startswith("ORB why not ordered:") for ln in lines), lines   # all 3 have a SUBMITTED line

    def test_why_not_ordered_empty_for_submitted_symbol(self):
        parsed = es.parse_orb_log(PRE_LOG)
        assert es.why_not_ordered('CORD', parsed, None) == ""


# ------------------------------------------------------------------ defect 3: hygiene
def _utc(y, mo, d, h, mi):
    return dt.datetime(y, mo, d, h, mi, tzinfo=dt.timezone.utc)


FRIDAY = (2026, 10, 30)
class TestHygieneServices:
    def test_trader_inactive_after_close_is_expected(self):
        flags, bad = eod_report.service_status_flags(_utc(*FRIDAY, 21, 57), lambda s: "inactive", lambda s: False)
        assert bad == []
        assert "onemil-trader=exited-at-close (expected)" in flags
        assert "onemil-macd-wave=not-installed" in flags

    def test_trader_inactive_in_session_is_not_active(self):
        flags, bad = eod_report.service_status_flags(_utc(*FRIDAY, 15, 0), lambda s: "inactive", lambda s: False)
        assert bad == ["onemil-trader"]
        assert "onemil-trader=inactive" in flags

    def test_weekend_inactive_is_expected(self):
        _f, bad = eod_report.service_status_flags(_utc(2026, 10, 31, 15, 0), lambda s: "inactive", lambda s: False)
        assert bad == []

    def test_window_edges(self):
        for hm, expect_bad in (((12, 34), False), ((12, 35), True), ((19, 55), True), ((19, 56), False)):
            _f, bad = eod_report.service_status_flags(_utc(*FRIDAY, *hm), lambda s: "inactive", lambda s: False)
            assert bool(bad) is expect_bad, hm

    def test_trader_active_is_listed_active(self):
        flags, bad = eod_report.service_status_flags(_utc(*FRIDAY, 15, 0), lambda s: "active", lambda s: False)
        assert bad == [] and "onemil-trader=active" in flags

    def test_macd_installed_and_inactive_is_flagged(self):
        _f, bad = eod_report.service_status_flags(_utc(*FRIDAY, 21, 57),
                                                  lambda s: "active" if s == "onemil-trader" else "inactive",
                                                  lambda s: True)
        assert bad == ["onemil-macd-wave"]

    def test_unit_installed_false_when_systemctl_cat_fails(self, monkeypatch):
        class R:
            returncode = 1; stdout = ""; stderr = "No files found"
        monkeypatch.setattr(eod_report.subprocess, "run", lambda *a, **k: R())
        assert eod_report.unit_installed("onemil-macd-wave") is False
        R.returncode = 0
        assert eod_report.unit_installed("onemil-trader") is True

    def test_hygiene_section_uses_helper(self, monkeypatch):
        monkeypatch.setattr(eod_report, "unit_installed", lambda s: False)
        monkeypatch.setattr(eod_report, "run", lambda cmd, timeout=180, **k: "inactive")
        monkeypatch.setattr(eod_report, "journal_error_count", lambda *a, **k: "0")
        monkeypatch.setattr(eod_report, "_utc_now", lambda: _utc(*FRIDAY, 21, 57))
        txt = eod_report.hygiene_section()
        assert "NOT ACTIVE" not in txt and "exited-at-close (expected)" in txt
