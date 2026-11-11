"""Unit + integration tests for scripts/orb_pick_parity.py (live final top-8 == point-in-time BT top-8).

Spec: docs/orb_selection_parity_test_spec_20261111.md. The integration fixture is the REAL 2026-11-03 engine log
(`tests/fixtures/orb_parity_2026-11-03.log`, lines grepped from logs/session_archive): on that day cell 1,733 found the
live final top-8 EQUAL to the PIT BT top-8 (8/8), with no difference to name.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest

from scripts import orb_pick_parity as opp

FIXTURE = Path(__file__).parent / "fixtures" / "orb_parity_2026-11-03.log"
DAY = "2026-11-03"
# bt_comp_pit / bt_q_pit of the 1,733 rows (research/book_review_20261111/selection_1733.csv), PIT top-8 of 11/3
BT_1103 = {"SRTA": (0.394281, "Q4"), "LABX": (0.391471, "Q4"), "AAOX": (0.298656, "Q4"), "SNXX": (0.295901, "Q4"),
           "SK": (0.767281, "Q5"), "NVDX": (0.734984, "Q5"), "SKHU": (0.719928, "Q5"), "KORU": (0.708671, "Q5")}
FILLED_1103 = {"LABX": "14:36:10", "SK": "14:36:40", "SKHU": "14:37:00", "AAOX": "14:37:30", "SNXX": "14:38:00"}


def bt_dict(ranked=None, picks=None, comp=None, vetoes=None, day_symbols=None):
    """A synthetic `bt_ranked` result."""
    ranked = list(ranked if ranked is not None else BT_1103)
    return {"ranked": ranked, "picks": list(picks if picks is not None else ranked), "n": 8,
            "comp": comp if comp is not None else {s: BT_1103.get(s, (0.5, "Q5")) for s in ranked},
            "vetoes": vetoes or {}, "day_symbols": day_symbols if day_symbols is not None else sorted(set(ranked) | {"ZZZ"}),
            "range_size": {s: 3.0 for s in ranked}, "params_refit_ts": "20261101_200000", "params_source": "pit"}


def live_dict(final=None, submitted=None, scored=None, range_done=None, vetoes=None, provisional=(), seen="", notes=None):
    """A synthetic live-side fact dict (shape of `parse_live`)."""
    final = list(final if final is not None else BT_1103)
    submitted = final if submitted is None else submitted     # default: live submitted every final name
    return {"final": sorted(final), "provisional": sorted(provisional), "submitted": sorted(submitted),
            "scored": scored if scored is not None else {s: BT_1103.get(s, (0.5, "Q5")) for s in final},
            "range_done": sorted(range_done if range_done is not None else final), "final_ranges": {},
            "vetoes": vetoes or {}, "notes": notes or {}, "seen": seen or " ".join(final), "has_reconcile": True}


def fill(pool="production", et="10:00:00"):
    """A live fill record."""
    return {"pool": pool, "composite": None, "filled_et": et, "status": "filled"}


def reasons(res):
    """{symbol: reason} of the differences."""
    return {d["symbol"]: d["reason"] for d in res["differences"]}


class TestReasons:
    def test_matched_has_no_difference(self):
        res = opp.compare(DAY, bt_dict(), live_dict(), {})
        assert res["matched"] == 8 and res["differences"] == [] and res["n_unexplained"] == 0
        assert "matched 8/8" in opp.summary_line(res) and "unexplained 0" in opp.summary_line(res)

    def test_universe_live_only(self):
        bt = bt_dict(ranked=["A", "B"], day_symbols=["A", "B"])
        res = opp.compare(DAY, bt, live_dict(final=["A", "X"]), {})
        assert reasons(res)["X"] == "universe_live_only"

    def test_universe_bt_only_when_live_never_saw_it(self):
        bt = bt_dict(ranked=["A", "B"])
        res = opp.compare(DAY, bt, live_dict(final=["A"], seen="A only"), {})
        assert reasons(res)["B"] == "universe_bt_only"

    def test_universe_bt_only_when_live_gap_below_floor(self):
        bt = bt_dict(ranked=["A", "B"])
        res = opp.compare(DAY, bt, live_dict(final=["A"], seen="A B", notes={"gap:B": "gap 3.070%"}), {})
        assert reasons(res)["B"] == "universe_bt_only"

    def test_gap_note_above_floor_does_not_explain(self):
        """GAP_GATE is logged for every candidate: a 11.7 % gap is not a drop reason."""
        bt = bt_dict(ranked=["A", "B"])
        res = opp.compare(DAY, bt, live_dict(final=["A"], seen="A B", range_done=["A"], notes={"gap:B": "gap 11.659%"}), {})
        assert reasons(res)["B"] == "range_missing"

    def test_range_missing_when_seen_but_never_ranged(self):
        bt = bt_dict(ranked=["A", "B"])
        res = opp.compare(DAY, bt, live_dict(final=["A"], seen="A B subscribed", range_done=["A"]), {})
        assert reasons(res)["B"] == "range_missing"

    def test_veto_input_range_bt_vetoes_live_submitted(self):
        bt = bt_dict(ranked=["A", "B"], picks=["A"], vetoes={"B": "range"})
        res = opp.compare(DAY, bt, live_dict(final=["A", "B"], submitted=["A", "B"]), {})
        assert reasons(res) == {"B": "veto_input_range"}

    def test_veto_input_pdr_live_vetoes_bt_passes(self):
        bt = bt_dict(ranked=["A", "B"])
        res = opp.compare(DAY, bt, live_dict(final=["A", "B"], submitted=["A"], vetoes={"B": "pdr"}), {})
        assert reasons(res) == {"B": "veto_input_pdr"}

    def test_param_ts_mismatch_from_stale_book(self):
        bt = bt_dict(ranked=["A"], comp={"A": (0.50, "Q5")})
        live = live_dict(final=["A"], scored={"A": (0.52, "Q5")})
        book = {"A": {"entered": 1, "params_refit_ts": "20261025_200000"}}
        res = opp.compare(DAY, bt, live, {}, book, pit_ts="20261101_200000")
        assert reasons(res) == {"A": "param_ts_mismatch"} and res["n_unexplained"] == 0

    def test_composite_difference_with_matching_ts_is_unexplained(self):
        bt = bt_dict(ranked=["A"], comp={"A": (0.50, "Q5")})
        res = opp.compare(DAY, bt, live_dict(final=["A"], scored={"A": (0.52, "Q5")}), {}, {}, pit_ts="20261101_200000")
        assert reasons(res) == {"A": "unexplained"} and res["n_unexplained"] == 1

    def test_composite_within_tolerance_passes(self):
        bt = bt_dict(ranked=["A"], comp={"A": (0.50000, "Q5")})
        res = opp.compare(DAY, bt, live_dict(final=["A"], scored={"A": (0.50005, "Q5")}), {})
        assert res["differences"] == []

    def test_addon_pool_fill_is_named_and_not_an_error(self):
        res = opp.compare(DAY, bt_dict(), live_dict(), {"APLX": fill(pool="addon_gap35")})
        assert reasons(res)["APLX"] == "addon_pool" and not opp.is_error(res)

    def test_provisional_fill_outside_final(self):
        res = opp.compare(DAY, bt_dict(), live_dict(provisional=["XMAX"]), {"XMAX": fill()})
        assert reasons(res)["XMAX"] == "provisional_fill" and opp.is_error(res)

    def test_time_stop_leak(self):
        res = opp.compare(DAY, bt_dict(), live_dict(), {"AXTL": fill(et="14:09:00")})
        assert reasons(res)["AXTL"] == "time_stop_leak" and opp.is_error(res)

    def test_no_fill_when_submitted_bt_entered(self):
        bt = bt_dict(ranked=["A"])
        res = opp.compare(DAY, bt, live_dict(final=["A"], submitted=["A"]), {}, {"A": {"entered": 1}})
        assert reasons(res) == {"A": "no_fill"} and res["n_unexplained"] == 0

    def test_bt_unavailable_is_pending_not_counted(self):
        res = opp.compare(DAY, None, live_dict(), {})
        assert res["bt_available"] is False and res["matched"] is None
        assert "unavailable" in opp.summary_line(res)


class TestAlarm:
    def test_error_only_for_unexplained(self, caplog):
        bt = bt_dict(ranked=["A"], comp={"A": (0.5, "Q5")})
        bad = opp.compare(DAY, bt, live_dict(final=["A"], scored={"A": (0.6, "Q5")}), {})
        with caplog.at_level(logging.INFO, logger="orb_pick_parity"):
            opp.log_result(bad)
        assert [r.levelname for r in caplog.records] == ["ERROR"] and "'A'" in caplog.records[0].getMessage()

    def test_explained_day_is_info(self, caplog):
        bt = bt_dict(ranked=["A", "B"])
        res = opp.compare(DAY, bt, live_dict(final=["A"], seen="A B"), {})
        with caplog.at_level(logging.INFO, logger="orb_pick_parity"):
            opp.log_result(res)
        assert [r.levelname for r in caplog.records] == ["INFO"]

    def test_bt_unavailable_is_warning(self, caplog):
        with caplog.at_level(logging.INFO, logger="orb_pick_parity"):
            opp.log_result(opp.compare(DAY, None, live_dict(), {}))
        assert [r.levelname for r in caplog.records] == ["WARNING"]


class TestParsers:
    def test_final_ranges(self):
        t = "x [ORB] final ranges: AAA 10.5000/9.9000, BB.B 3.2/3.1"
        assert opp.parse_final_ranges(t) == {"AAA": (10.5, 9.9), "BB.B": (3.2, 3.1)}

    def test_live_vetoes_skip_provisional(self):
        t = ("[ORB PREPLACE] provisional PDR VETO: AAA prev-day range 6%\n"
             "x | [ORB] PDR VETO: BBB prev-day range 6% <= 11% — quiet\n"
             "x | [ORB] RANGE-SIZE VETO: CCC range_size_pct 1.2\n")
        assert opp.parse_live_vetoes(t) == {"BBB": "pdr", "CCC": "range"}

    def test_previous_session_day_skips_weekend(self):
        assert opp.previous_session_day("2026-11-09") == "2026-11-06"
        assert opp.previous_session_day("2026-11-11") == "2026-11-10"


def make_db(path, rows):
    """A minimal trades table (the columns `live_fills` reads)."""
    con = sqlite3.connect(path)
    con.execute("create table trades (symbol, strategy, trade_date, fill_price, filled_at, order_status, pattern_data)")
    con.executemany("insert into trades values (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


class TestSources:
    def test_live_fills_reads_only_filled_orb_rows(self, tmp_path):
        db = tmp_path / "t.db"
        make_db(db, [("LABX", "orb", DAY, 6.49, "2026-11-03T14:36:10+00:00", "filled", json.dumps({"pool_id": "production"})),
                     ("XMAX", "orb", DAY, None, None, "preplace_canceled", "{}"),
                     ("APLX", "orb", DAY, 3.0, "2026-11-03T14:40:00+00:00", "filled", json.dumps({"pool": "addon_gap35"})),
                     ("OTHER", "bull_flag", DAY, 1.0, "2026-11-03T14:40:00+00:00", "filled", "{}")])
        f = opp.live_fills(DAY, db)
        assert set(f) == {"LABX", "APLX"} and f["LABX"]["filled_et"] == "09:36:10" and f["APLX"]["pool"] == "addon_gap35"

    def test_live_fills_missing_db_is_empty_with_warning(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="orb_pick_parity"):
            assert opp.live_fills(DAY, tmp_path / "nope.db") == {}
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_book_rows_filters_the_day(self, tmp_path):
        b = tmp_path / "book.csv"
        b.write_text("symbol,date,entered,params_refit_ts\nA,2026-11-03,1,20261101_200000\nB,2026-11-04,1,x\n")
        assert set(opp.book_rows(DAY, b)) == {"A"}


class TestIntegrationReal1103:
    """Real 11/3 log lines -> parse -> compare with the 1,733 PIT top-8: every boundary checked."""

    @pytest.fixture
    def parts(self, tmp_path):
        db = tmp_path / "t.db"
        make_db(db, [(s, "orb", DAY, 10.0, f"2026-11-03T{t}+00:00", "filled", json.dumps({"pool_id": "production"}))
                     for s, t in FILLED_1103.items()])
        book = tmp_path / "book.csv"
        book.write_text("symbol,date,entered,params_refit_ts\n" + "".join(
            f"{s},{DAY},1,20261101_200000\n" for s in FILLED_1103))
        return db, book

    def run(self, db, book, bt=None):
        return opp.run_parity(DAY, bt_fn=lambda d: (bt or bt_dict(vetoes={"SRTA": "pdr", "NVDX": "pdr", "KORU": "range"}), ""),
                              log_fn=lambda d: FIXTURE.read_text(), fills_fn=lambda d: opp.live_fills(d, db), book_path=book)

    def test_fixture_parses_the_real_final_top8_and_submissions(self):
        live = opp.parse_live(FIXTURE.read_text())
        assert set(live["final"]) == set(BT_1103)
        assert {"LABX", "SK", "SKHU", "AAOX", "SNXX", "XMAX", "SKUU"} <= set(live["submitted"])
        assert {"XMAX", "SKUU"} <= set(live["provisional"]) and "XMAX" not in live["final"]
        assert abs(live["scored"]["LABX"][0] - 0.3915) < 1e-3 and live["scored"]["SK"][1] == "Q5"

    def test_real_day_matches_8_of_8_with_zero_unexplained(self, parts, caplog):
        with caplog.at_level(logging.INFO, logger="orb_pick_parity"):
            res = self.run(*parts)
        assert res["matched"] == 8 and res["n_unexplained"] == 0 and res["fills_outside_final"] == []
        assert res["differences"] == [] and res["live_filled"] == sorted(FILLED_1103)
        assert res["summary"].startswith("ORB pick parity 2026-11-03: matched 8/8")
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)

    def test_fill_outside_final_is_an_error(self, parts, tmp_path, caplog):
        db, book = parts
        con = sqlite3.connect(db)   # XMAX was only a provisional order; a fill on it must alarm
        con.execute("insert into trades values ('XMAX','orb',?,4.3,'2026-11-03T14:35:30+00:00','filled',?)",
                    (DAY, json.dumps({"pool_id": "production"})))
        con.commit()
        con.close()
        with caplog.at_level(logging.INFO, logger="orb_pick_parity"):
            res = self.run(db, book)
        errs = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errs) == 1 and "XMAX" in errs[0].getMessage()
        assert reasons(res) == {"XMAX": "provisional_fill"} and opp.is_error(res)

    def test_one_name_off_is_named_not_swallowed(self, parts):
        bt = bt_dict(ranked=[s for s in BT_1103 if s != "KORU"] + ["GHOST"], vetoes={"SRTA": "pdr", "NVDX": "pdr"})
        res = self.run(*parts, bt=bt)
        assert res["matched"] == 7
        assert reasons(res) == {"GHOST": "universe_bt_only", "KORU": "universe_live_only"}
        assert res["n_unexplained"] == 0


class TestWiring:
    """Observer -> audit json -> EOD hygiene line."""

    def test_observer_records_both_days_and_flags_error_text(self):
        from scripts import orb_selection_observer as obs
        ok = {"summary": "s-ok", "differences": [], "bt_available": True, "matched": 8, "k": 8, "n_unexplained": 0,
              "fills_outside_final": []}
        bad = {"summary": "s-bad", "differences": [{"symbol": "ZZ", "reason": "unexplained"}], "bt_available": True,
               "matched": 7, "k": 8, "n_unexplained": 1, "fills_outside_final": []}
        out = obs.run_pick_parity("2026-11-11", parity_fn=lambda d: bad if d == "2026-11-10" else ok)
        assert out["error"] is True and "s-bad" in out["text"] and "ZZ:unexplained" in out["text"]
        assert set(out) >= {"2026-11-11", "2026-11-10"}
        json.dumps(out)   # the audit json must serialise

    def test_observer_failure_never_blocks_the_verdict(self):
        from scripts import orb_selection_observer as obs

        def boom(d):
            raise RuntimeError("journal gone")
        out = obs.run_pick_parity("2026-11-11", parity_fn=boom)
        assert out["error"] is False and "journal gone" in out["failed"]

    def test_eod_line_carries_picks_and_unexplained(self, tmp_path):
        from scripts import report_common as rc
        (tmp_path / "orb_selection_audit_20261111.json").write_text(json.dumps({
            "live_range": list("abcdefgh"), "expected_candidates": {s: {} for s in "abcdefghij"}, "drop_classification": {},
            "pick_parity": {"2026-11-11": {"bt_available": False},
                            "2026-11-10": {"bt_available": True, "matched": 7, "k": 8, "n_unexplained": 0}}}))
        line = rc.selection_parity_line("2026-11-11", logs_dir=tmp_path)
        assert "REAL drops 0" in line and "picks pending" in line and "11-10 picks 7/8, unexplained 0" in line
