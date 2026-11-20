"""Tests for the `bar_revision` reason of scripts/orb_pick_parity.py (2026-11-20, MUZ 2026-11-16).

Spec: docs/orb_pick_parity_bar_revision_20261120.md. A shared-name composite drift <= HARD_COMP_DRIFT whose composite
inputs differ between the live `ORB SCORED` line and the rebuilt features row is a vendor bar revision after the
09:35 snapshot: named, INFO. Identical inputs or a drift > HARD_COMP_DRIFT stay `unexplained`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from scripts import orb_pick_parity as opp
from tests.test_orb_pick_parity_20261111 import bt_dict, live_dict, reasons

DAY = "2026-11-16"
MUZ_LINE = ("Nov 16 14:35:11 host onemil-trader[1]: 2026-11-16 14:35:11 | INFO     | trading.orb_engine:4002 | "
            "ORB SCORED: MUZ comp=0.4531 Q5 | gap=11.482 rtv=125552 rabr=1.213 rs=3.933 p20h=-26.849 pdcp=0.373 "
            "rcp=0.939 | prev_close=9.5800 range_open=10.6800 pool=production")
LIVE_F = {"gap_pct": 11.482, "range_total_volume": 125552.0, "range_avg_bar_range_pct": 1.213,
          "range_size_pct": 3.933, "price_vs_20d_high_pct": -26.849, "prev_day_close_position": 0.373,
          "range_close_position": 0.939}
BT_F = {"gap_pct": 11.482, "range_total_volume": 125652.0, "range_avg_bar_range_pct": 1.2132,
        "range_size_pct": 3.9326, "price_vs_20d_high_pct": -26.849, "prev_day_close_position": 0.3725,
        "range_close_position": 0.9405}


def muz(live_comp=0.4531, bt_comp=0.4538, live_f=LIVE_F, bt_f=BT_F):
    """compare() of a day whose only shared name is MUZ with the given composites and feature dicts."""
    bt = bt_dict(ranked=["MUZ"], comp={"MUZ": (bt_comp, "Q5")})
    bt["features"] = {"MUZ": dict(bt_f)}
    live = live_dict(final=["MUZ"], scored={"MUZ": (live_comp, "Q5")})
    live["features"] = {"MUZ": dict(live_f)}
    return opp.compare(DAY, bt, live, {})


def test_muz_pair_is_bar_revision(caplog):
    res = muz()
    assert reasons(res) == {"MUZ": "bar_revision"} and res["n_unexplained"] == 0
    d = res["differences"][0]
    assert "range_total_volume 125552 vs 125652" in d["detail"] and "range_close_position" in d["detail"]
    with caplog.at_level(logging.INFO, logger="orb_pick_parity"):
        opp.log_result(res)
    assert [r.levelname for r in caplog.records] == ["INFO"] and not opp.is_error(res)
    assert "bar_revision 1" in opp.summary_line(res)


def test_identical_inputs_stay_unexplained():
    res = muz(live_f=BT_F)
    assert reasons(res) == {"MUZ": "unexplained"} and res["n_unexplained"] == 1
    assert "live comp 0.4531 vs BT 0.4538" in res["differences"][0]["detail"]


def test_drift_above_hard_limit_stays_unexplained():
    res = muz(bt_comp=0.4531 + 0.06)
    assert reasons(res) == {"MUZ": "unexplained"}


def test_missing_features_stay_unexplained():
    bt = bt_dict(ranked=["MUZ"], comp={"MUZ": (0.4538, "Q5")})
    res = opp.compare(DAY, bt, live_dict(final=["MUZ"], scored={"MUZ": (0.4531, "Q5")}), {})
    assert reasons(res) == {"MUZ": "unexplained"}


def test_feature_revisions_rounding():
    f = opp.feature_revisions
    assert f({"range_total_volume": 125652.0}, {"range_total_volume": 125652.4}) == []
    assert f({"range_close_position": 0.940}, {"range_close_position": 0.9404}) == []
    assert f({"range_total_volume": 125652.0}, {"range_total_volume": 125653.0}) == ["range_total_volume 125652 vs 125653"]
    assert len(f({"range_close_position": 0.939}, {"range_close_position": 0.9405})) == 1
    assert f({}, BT_F) == []


def test_parser_real_muz_line():
    live = opp.parse_live(MUZ_LINE)
    assert live["features"]["MUZ"] == pytest.approx(LIVE_F)


def test_parser_old_format_and_addon_ignored():
    old = MUZ_LINE.split(" | gap=")[0] + " | prev_close=9.58 pool=production"
    addon = MUZ_LINE.replace("pool=production", "pool=p4").replace("MUZ", "ADD")
    live = opp.parse_live(old + "\n" + addon)
    assert live["features"] == {}


REAL_LOG = Path(__file__).resolve().parents[1] / "logs" / "session_archive" / "2026-11-16.log"


@pytest.mark.skipif(not REAL_LOG.exists(), reason="real session archive absent")
def test_real_1116_muz_is_bar_revision():
    """Read-only run against the real archive + newest features CSV (BT side skipped when it does not cover the day)."""
    from scripts import eod_sections as es
    bt, why = es.bt_ranked(DAY)
    if bt is None:
        pytest.skip(f"BT unavailable: {why}")
    res = opp.run_parity(DAY, log_fn=lambda d: REAL_LOG.read_text(), fills_fn=lambda d: {}, emit=False)
    muz_d = [d for d in res["differences"] if d["symbol"] == "MUZ"]
    assert res["n_unexplained"] == 0 or all(d["symbol"] != "MUZ" for d in res["differences"] if d["reason"] == "unexplained")
    assert muz_d and muz_d[0]["reason"] == "bar_revision"
