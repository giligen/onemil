"""Unit + integration tests for scripts/eod_sections.py and its wiring in scripts/eod_report.py (fixtures only)."""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eod_report as er  # noqa: E402
import eod_sections as es  # noqa: E402

DAY = "2026-10-02"


def _weekly(tmp_path, rows):
    p = tmp_path / "weekly.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "equity"])
        w.writeheader()
        w.writerows({"date": d, "equity": e} for d, e in rows)
    return p


def test_pnl_line_day_week_since_and_dd():
    rows = [{"date": "2026-09-29", "equity": "20000"}, {"date": "2026-10-01", "equity": "20400"},
            {"date": DAY, "equity": "20134.49"}]
    line = es.sleeve_pnl_line(rows, DAY)
    assert "day $-266" in line and "since 2026-09-29 $+134" in line and "DD from peak 1.3 %" in line


def test_pnl_line_no_mark_is_no_data():
    assert "NO-DATA" in es.sleeve_pnl_line([], DAY)


def test_rotation_stats_and_non_rotation():
    led = [{"date": DAY, "symbol": "A", "slip_bps_vs_open": "10", "size_pct": "100"},
           {"date": DAY, "symbol": "B", "slip_bps_vs_open": "-30", "size_pct": "100"}]
    s = es.rotation_stats(led, DAY)
    assert s["slip_mean"] == -10 and s["slip_max"] == 30 and s["size"] == "100%"
    assert es.rotation_stats(led, "2026-10-05") is None


def test_rotation_line_diff_and_match():
    st = {"names": ["A", "B"], "slip_mean": 5.0, "slip_max": 9.0, "size": "100%"}
    line, same = es.rotation_line(DAY, st, ["A", "B"], ["A", "C"], "", "gate p21")
    assert not same and "+B" in line and "-C" in line and "gate p21" in line
    assert es.rotation_line(DAY, st, ["A"], ["A"], "", "gate n/a")[1]
    line, same = es.rotation_line(DAY, st, ["A"], None, "no cache", "gate n/a")
    assert not same and "NO-DATA (no cache)" in line


def test_reconcile_ok_mismatch_nodata():
    assert es.reconcile_line({"A": (10.0, 1000.0)}, {"A": 10.0}, {"A": 100.0})[1] is True
    line, ok = es.reconcile_line({"A": (10.0, 1000.0), "B": (1, 5)}, {"A": 10.0}, {"A": 100.0})
    assert ok is False and "MISMATCH" in line and "name B" in line
    assert es.reconcile_line(None, {}, {})[1] is None


def test_completeness_parse_and_verdict():
    t = f"{DAY} 16:56:25 INFO momentum_sleeve: COMPLETENESS: requested 1000 symbols, 900 with bars, LOST 10"
    info = es.completeness_from_logs([t], DAY)
    assert info["lost"] == 10
    assert es.completeness_line(info)[0].endswith("OK")
    refused = es.completeness_from_logs([t, f"{DAY} 17:00 ERROR MOM REFUSED: x"], DAY)
    assert es.completeness_line(refused)[0].endswith("REFUSED")
    assert "NO-DATA" in es.completeness_line({})[0]


def test_gate_label_fallbacks():
    assert es.gate_label_for(DAY, [{"run_date": DAY, "percentile": "0.21"}], []) == "gate p21"
    assert es.gate_label_for(DAY, [], ["x gate OFF (p55) y"]) == "gate p55"
    assert es.gate_label_for(DAY, [], []) == "gate n/a"


def _scored_q(sym, comp, q, hhmm="13:35:05"):
    """SCORED line with the logged composite and quintile, as the engine writes it."""
    return (f"Oct 02 {hhmm} host onemil-trader[1]: 2026-10-02 {hhmm} | INFO | trading.orb_engine:1 | "
            f"ORB SCORED: {sym} comp={comp:.4f} {q} | gap=1")


def _scored(sym, hhmm="13:35:05"):
    """Archive line as journald writes it; engine stamps are UTC (13:35 UTC = 09:35 ET in October)."""
    return f"Oct 02 {hhmm} host onemil-trader[1]: 2026-10-02 {hhmm} | INFO | trading.orb_engine:1 | ORB SCORED: {sym} comp=1"


ENG = [{"symbol": "AAA", "strategy": "orb", "account": "paper", "fill_price": 10.02, "exit_price": 10.5, "pnl": 40.0,
        "pattern_data": "{}"}]
BT = [{"symbol": "AAA", "entry_price": 10.0, "pnl": 55.0}]


def _rank(ranked, picks, **kw):
    d = {"ranked": ranked, "picks": picks, "rows": 12, "pdr": 0, "g1": 0, "range": 0, "dedup": 0, "n": 8}
    d.update(kw)
    return d


def test_orb_no_decision_is_neutral():
    lines, m = es.orb_parity_lines(DAY, [], es.parse_orb_log("nothing"), BT)
    assert lines[0].startswith("ORB picks: NO DECISION") and m["clean"] is None and not m["decision"]


def test_orb_clean_session_with_picks_and_defect_counts():
    parsed = es.parse_orb_log(_scored("AAA"))
    lines, m = es.orb_parity_lines(DAY, ENG, parsed, BT, "", _rank(["AAA"], ["AAA"]))
    assert m["clean"] is True and any("mean +20.0 bp" in ln for ln in lines)
    assert lines[0].startswith("ORB ranked: engine 1 vs BT 1 | match 1")
    assert any(ln.startswith("BT: rows 12 -> top-8 (1) -> vetoed 0") for ln in lines)
    assert any("ORB P&L: day $+40 on 1 exits | BT book $+55" in ln for ln in lines)
    p = es.parse_orb_log(_scored("AAA") + "\n2026 | ERROR | orb ORB boom\nEngine tick TIMEOUT (>50s)\n")
    assert p["errors"] == 1 and p["timeouts"] == 1
    _, m2 = es.orb_parity_lines(DAY, ENG, p, BT, "", _rank(["AAA"], ["AAA"]))
    assert m2["clean"] is False and "TIMEOUT" in m2["why"]


def test_orb_zero_pick_day_ranked_match_is_clean():
    parsed = es.parse_orb_log(_scored("AAA") + "\n" + _scored("BBB"))
    lines, m = es.orb_parity_lines(DAY, [], parsed, [], "", _rank(["AAA", "BBB"], [], pdr=2))
    assert m["clean"] is True and any("vetoed 2 (PDR 2" in ln for ln in lines)
    assert any("0 = 0" in ln for ln in lines)


def test_orb_ranked_mismatch_is_not_clean_even_at_zero_picks():
    parsed = es.parse_orb_log(_scored("AAA") + "\n" + _scored("ZZZ"))
    lines, m = es.orb_parity_lines(DAY, [], parsed, [], "", _rank(["AAA", "BBB"], [], pdr=2))
    assert m["clean"] is False and m["why"] == "ranked set != BT"
    assert "engine-only: ZZZ | BT-only: BBB" in lines[0]


def test_orb_bt_nodata_is_neutral():
    lines, m = es.orb_parity_lines(DAY, ENG, es.parse_orb_log(_scored("AAA")), None, "x", None, "features do not cover")
    assert "NO-DATA (features do not cover)" in lines[0] and m["clean"] is None


def test_orb_entry_diff_over_tolerance_and_pick_mismatch():
    eng = [dict(ENG[0], fill_price=10.1)]
    _, m = es.orb_parity_lines(DAY, eng, es.parse_orb_log(_scored("AAA")), BT, "", _rank(["AAA"], ["AAA"]))
    assert m["clean"] is False and "30 bp" in m["why"]
    _, m = es.orb_parity_lines(DAY, ENG, es.parse_orb_log(_scored("ZZZ")), BT, "", _rank(["AAA"], ["AAA"]))
    assert m["why"] == "ranked set != BT"
    _, m = es.orb_parity_lines(DAY, ENG, es.parse_orb_log(_scored("AAA")), BT, "", _rank(["AAA"], []))
    assert m["why"] == "picks != BT"


def test_neutral_orb_day_does_not_touch_the_counter(tmp_path):
    sp = tmp_path / "s.json"
    ok = {"clean": True, "why": ""}
    es.promotion_section("2026-10-05", None, ok, sp, closed=True)
    es.promotion_section("2026-10-06", None, {"clean": None, "why": "no 09:35 decision (neutral)"}, sp, closed=True)
    es.promotion_section("2026-10-07", None, ok, sp, closed=True)
    assert es.consecutive_clean(json.loads(sp.read_text())["orb"]) == 2
    es.promotion_section("2026-10-08", None, {"clean": False, "why": "ranked set != BT"}, sp, closed=True)
    assert es.consecutive_clean(json.loads(sp.read_text())["orb"]) == 0


def test_consecutive_clean():
    assert es.consecutive_clean({"a": True, "b": False, "c": True, "d": True}) == 2
    assert es.consecutive_clean({}) == 0


def test_sleeve_go_after_two_clean_and_hold_on_slip(tmp_path):
    sp = tmp_path / "s.json"
    clean = {"rotation": True, "clean": True, "why": ""}
    es.promotion_section("2026-10-02", clean, None, sp, closed=True)
    out = es.promotion_section("2026-10-09", clean, None, sp, closed=True)
    assert "Sleeve: GO LIVE $20K on 2026-10-12" in out[1]
    slow = {"rotation": True, "clean": False, "why": "slip mean 25 bp > 20"}
    out = es.promotion_section("2026-10-16", slow, None, sp, closed=True)
    assert "HOLD 0/2" in out[1] and "slip mean 25 bp > 20" in out[1]


def test_sleeve_hold_when_review_open(tmp_path):
    sp = tmp_path / "s.json"
    clean = {"rotation": True, "clean": True, "why": ""}
    es.promotion_section("2026-10-02", clean, None, sp, closed=False)
    out = es.promotion_section("2026-10-09", clean, None, sp, closed=False)
    assert "HOLD 2/2" in out[1] and "review fixes" in out[1]


def test_promotion_idempotent_and_non_rotation_day_not_counted(tmp_path):
    sp = tmp_path / "s.json"
    clean = {"rotation": True, "clean": True, "why": ""}
    es.promotion_section("2026-10-02", clean, None, sp, closed=True)
    es.promotion_section("2026-10-02", clean, None, sp, closed=True)
    es.promotion_section("2026-10-05", {"rotation": False, "clean": None, "why": ""}, None, sp, closed=True)
    assert es.consecutive_clean(json.loads(sp.read_text())["sleeve"]) == 1


def test_orb_promotion_five_clean_then_mismatch_resets(tmp_path):
    sp = tmp_path / "s.json"
    ok = {"clean": True, "why": ""}
    days = ["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"]
    for d in days[:4]:
        out = es.promotion_section(d, None, ok, sp, closed=True)
    assert "ORB: HOLD 4/5" in out[2]
    out = es.promotion_section(days[4], None, ok, sp, closed=True)
    assert "ORB: GO $10K stage ($375 R) on 2026-10-12" in out[2]
    out = es.promotion_section("2026-10-12", None, {"clean": False, "why": "ranked set != BT"}, sp, closed=True)
    assert "HOLD 0/5 (ranked set != BT)" in out[2]


def test_ramp_verdicts():
    assert "HOLD (no live stage" in es.ramp_verdict("ORB", None, None, None)
    r = {"stage_usd": 10000}
    assert "+$10,000 to $20,000" in es.ramp_verdict("MOM", r, 50.0, 20)
    assert "HOLD at $10,000" in es.ramp_verdict("MOM", r, -5.0, 30)
    assert "HOLD at $10,000" in es.ramp_verdict("MOM", r, 50.0, 10)
    assert "NO-DATA" in es.ramp_verdict("MOM", r, None, None)


def test_failed_sections_and_no_data_in_promotion(tmp_path):
    out = es.promotion_section(DAY, None, None, tmp_path / "s.json", closed=True)
    assert any("Sleeve: NO-DATA" in ln for ln in out) and out[-1].startswith("  HOD: dry-run only")


def test_safe_section_catches():
    lines, m = er.safe_section("MOM", lambda: 1 / 0)
    assert lines == ["MOM: FAILED (ZeroDivisionError: division by zero)"] and m is None


def test_split_promotion_passes_block_verbatim():
    summary = "BOOKS\nPROMOTION:\n  Sleeve: HOLD 0/2 [x]\n  HOD: y\nGUARDRAIL:\n  z"
    body, promo = er.split_promotion(summary)
    assert promo == "PROMOTION:\n  Sleeve: HOLD 0/2 [x]\n  HOD: y"
    assert "PROMOTION" not in body and "GUARDRAIL:" in body
    assert er.split_promotion("no block") == ("no block", "")


def test_load_bt_rows_reads_through_orb_csv_and_na_ticker(tmp_path):
    p = tmp_path / "bt.csv"
    p.write_text("symbol,date,entry_price,pnl\nNA,2026-10-02,5.0,1.0\nBBB,2026-10-01,6.0,2.0\n")
    rows, why = es.load_bt_rows(DAY, str(p))
    assert rows[0]["symbol"] == "NA" and why == ""
    rows, why = es.load_bt_rows("2026-10-05", str(p))
    assert rows is None and "no rows" in why


MON = "2026-10-05"


def test_sleeve_section_integration_with_fixtures(tmp_path, monkeypatch):
    led = tmp_path / "led.csv"
    led.write_text("date,symbol,side,qty,avg_price,notional,official_open,slip_bps_vs_open,client_order_id,size_pct\n"
                   f"{MON},AAA,buy,10,100,1000,100,8,c1,100\n")
    wk = _weekly(tmp_path, [("2026-09-29", 20000), (MON, 20050)])
    (tmp_path / "momentum_sleeve_t.log").write_text(f"{MON} 13:45:01,000 INFO momentum_sleeve: COMPLETENESS: requested 100 symbols, x LOST 0\n")
    monkeypatch.setattr(es, "SLEEVE_LOG_GLOB", str(tmp_path / "momentum_sleeve*.log"))
    st = tmp_path / "state.json"
    st.write_text(json.dumps({"positions": {"AAA": 10.0}, "last_rebalance": MON}))

    class P:
        symbol, qty, market_value = "AAA", "10", "1000"

    class TC:
        def get_all_positions(self):
            return [P()]

    class C:
        trading_client = TC()

    lines, m = es.sleeve_section(MON, client=C(), ledger_path=led, weekly_path=wk, state_path=st,
                                 bt_fn=lambda d: (["AAA"], ""))
    assert m["rotation"] is True and "picks 1/20 = BT top-20" in lines[1] and "slip mean 8.0 bp" in lines[1]
    assert any("reconcile: broker 1 names $1,000 vs state 1 names $1,000 | OK" in ln for ln in lines)
    assert lines[0].startswith("MOM P&L: day $+50")


def test_orb_late_scoring_after_restart_is_not_a_decision_if_after_ten():
    boot = "Oct 02 14:04:04 h onemil-trader[2]: 2026-10-02 14:04:04 | INFO | trading.orb_engine:827 | [ORB] WINNER STACK: x"
    p = es.parse_orb_log(_scored("ORCU", "14:05:10") + "\n" + boot)
    assert p["scored"] == [] and p["late"] == ["ORCU"] and p["boots"] == ["14:04"]
    lines, m = es.orb_parity_lines(DAY, [], p, [])
    assert lines[0] == ("ORB picks: NO DECISION (no SCORED line in the 09:34\u201310:00 ET window; "
                        "restart 14:04 UTC)")
    assert lines[1] == "ORB late scoring, not a decision: ORCU" and not m["decision"] and m["clean"] is None


def test_orb_window_edges():
    assert es.parse_orb_log(_scored("A", "13:34:00"))["scored"] == ["A"]
    assert es.parse_orb_log(_scored("A", "13:41:00"))["scored"] == ["A"]   # window runs to 10:00 ET
    assert es.parse_orb_log(_scored("A", "14:01:00"))["scored"] == []


def test_orb_zero_bt_picks_when_features_cover_the_day(tmp_path):
    p = tmp_path / "bt.csv"
    p.write_text("symbol,date,entry_price,pnl\nBBB,2026-10-01,6.0,2.0\n")
    rows, why = es.load_bt_rows("2026-10-05", str(p), covered=lambda d: True)
    assert rows == [] and why == ""
    rows, why = es.load_bt_rows("2026-10-05", str(p), covered=lambda d: False)
    assert rows is None and "do not cover" in why


def test_scheduled_rotation_rules():
    mon, thu = "2026-10-05", "2026-10-02"
    ok_log = [f"{mon} 13:45:01,000 INFO momentum_sleeve: x"]
    led = [{"date": mon, "client_order_id": "mom-20261005-A-b"}]
    assert es.is_scheduled_rotation(mon, led, ok_log) == (True, "")
    assert es.is_scheduled_rotation(thu, [], [f"{thu} 13:45:01 x"])[1] == "not a Monday"
    forced = [{"date": mon, "client_order_id": "mom-20261005-A-b-f134500"}]
    assert es.is_scheduled_rotation(mon, forced, ok_log)[1] == "force-tagged order ids"
    assert "13:40-15:00" in es.is_scheduled_rotation(mon, led, [f"{mon} 16:55:42 x"])[1]
    assert es.is_scheduled_rotation(mon, led, [])[0] is False


def test_forced_run_not_counted_in_sleeve_section(tmp_path, monkeypatch):
    led = tmp_path / "led.csv"
    led.write_text("date,symbol,side,qty,avg_price,notional,official_open,slip_bps_vs_open,client_order_id,size_pct\n"
                   f"{DAY},AAA,buy,10,100,1000,100,457,mom-20261002-AAA-b-f165537,\n")
    wk = _weekly(tmp_path, [("2026-09-29", 20000), (DAY, 20050)])
    st = tmp_path / "state.json"
    st.write_text(json.dumps({"positions": {"AAA": 10.0}, "last_rebalance": DAY}))
    lines, m = es.sleeve_section(DAY, client=None, ledger_path=led, weekly_path=wk, state_path=st,
                                 bt_fn=lambda d: (["AAA"], ""))
    assert any(ln.startswith("MOM forced run (not counted)") for ln in lines)
    assert m["rotation"] is False and m["clean"] is None


def test_sleeve_verdict_with_no_scheduled_rotation(tmp_path):
    out = es.promotion_section(DAY, {"rotation": False, "clean": None, "why": ""}, None, tmp_path / "s.json", closed=True)
    assert out[1] == ("  Sleeve: HOLD 0/2 (no scheduled rotation yet; first 2026-10-05) [2 clean scheduled Monday "
                      "rotations, slip <= 20 bp, picks = BT, reconcile OK]")


def test_engine_top_n_uses_logged_scores_and_quintile_order():
    txt = "\n".join([_scored_q("LOW", 0.9, "Q2"), _scored_q("AAA", 0.30, "Q4"), _scored_q("BBB", 0.50, "Q4"),
                     _scored_q("CCC", 0.20, "Q5"), _scored_q("ONE", 0.99, "Q1")])
    p = es.parse_orb_log(txt)
    assert es.engine_top_n(p, 3) == ["BBB", "AAA", "CCC"]
    assert es.engine_top_n(es.parse_orb_log(_scored("AAA")), 3) is None   # no quintile logged


def test_ranked_clean_when_engine_top_n_equals_bt_even_with_extra_scored():
    txt = "\n".join([_scored_q("AAA", 0.5, "Q4"), _scored_q("BBB", 0.4, "Q4"), _scored_q("EXTRA", 0.1, "Q2")])
    lines, m = es.orb_parity_lines(DAY, [], es.parse_orb_log(txt), [], "", _rank(["AAA", "BBB"], [], n=2))
    assert m["clean"] is True and "subset test" not in lines[0]
    _, m = es.orb_parity_lines(DAY, [], es.parse_orb_log(txt), [], "", _rank(["AAA", "EXTRA"], [], n=2))
    assert m["clean"] is False


def test_subset_fallback_and_late_decision_flag():
    lines, m = es.orb_parity_lines(DAY, [], es.parse_orb_log(_scored("AAA", "13:46:00") + "\n" + _scored("BBB", "13:47:00")),
                                   [], "", _rank(["AAA"], []))
    assert "(subset test - engine scores not logged)" in lines[0] and "late decision (09:46 ET)" in lines[0]
    assert m["clean"] is True
