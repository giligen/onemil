"""slot_displaced reason of scripts/orb_pick_parity.py (spec docs/orb_pick_parity_displaced_20261114.md).

2026-11-13: a GRACE stand-in (STRO, no BT feature row) took a live top-8 slot, so the BT's 8th name (BTCZ) was live rank 9.
The classifier named it `unexplained` (ERROR to the owner); it is a documented slot displacement, not a defect.
"""
from __future__ import annotations

import logging

from scripts import orb_pick_parity as opp

DAY = "2026-11-13"
BT_TOP8 = ["TNDM", "SKDD", "EQPT", "CABO", "DAMD", "KOLD", "WSE", "BTCZ"]
# the 11 live ranked names of 11/13 (quintile, then composite DESC): STRO is 3rd, BTCZ 9th
LIVE_SCORED = {"TNDM": (0.61, "Q4"), "SKDD": (0.55, "Q4"), "STRO": (0.50, "Q4"), "EQPT": (0.45, "Q4"),
               "CABO": (0.85, "Q5"), "DAMD": (0.80, "Q5"), "KOLD": (0.70, "Q5"), "WSE": (0.60, "Q5"),
               "BTCZ": (0.4325, "Q5"), "XAAA": (0.40, "Q5"), "XBBB": (0.55, "Q3")}
LIVE_FINAL = ["TNDM", "SKDD", "STRO", "EQPT", "CABO", "DAMD", "KOLD", "WSE"]


def bt_dict(ranked=None, day_symbols=None):
    """A synthetic `bt_ranked` result (BT top-8 of 11/13, STRO has no feature row)."""
    ranked = list(ranked if ranked is not None else BT_TOP8)
    comp = {s: LIVE_SCORED.get(s, (0.5, "Q5")) for s in ranked}
    return {"ranked": ranked, "picks": list(ranked), "n": 8, "comp": comp, "vetoes": {},
            "day_symbols": day_symbols if day_symbols is not None else sorted(set(ranked) | {"XAAA", "XBBB"}),
            "range_size": {s: 3.0 for s in ranked}, "params_refit_ts": "20261101_200000", "params_source": "pit"}


def live_dict(final=None, scored=None, extra_seen=""):
    """A synthetic live-side fact dict (shape of `parse_live`); every name ranged, every final name submitted."""
    final = list(final if final is not None else LIVE_FINAL)
    scored = dict(scored if scored is not None else LIVE_SCORED)
    return {"final": sorted(final), "provisional": [], "submitted": sorted(final), "scored": scored,
            "range_done": sorted(scored), "final_ranges": {}, "vetoes": {}, "notes": {},
            "seen": " ".join(scored) + " " + extra_seen, "has_reconcile": True}


def reasons(res):
    """{symbol: reason} of the differences."""
    return {d["symbol"]: d["reason"] for d in res["differences"]}


def test_live_ranks_engine_key():
    """Quintile order Q4, Q5, Q3 then composite DESC; BTCZ is 9th with STRO in the field."""
    r = opp.live_ranks(live_dict())
    assert r["TNDM"] == 1 and r["STRO"] == 3 and r["BTCZ"] == 9 and r["XAAA"] == 10 and r["XBBB"] == 11


def test_2026_11_13_shape_is_slot_displaced():
    """(a) one live-only stand-in displaces BTCZ to rank 9 -> slot_displaced, not an error."""
    res = opp.compare(DAY, bt_dict(), live_dict(), {})
    assert reasons(res) == {"BTCZ": "slot_displaced", "STRO": "universe_live_only"}
    assert res["n_unexplained"] == 0 and opp.is_error(res) is False
    d = next(d for d in res["differences"] if d["symbol"] == "BTCZ")
    assert d["detail"] == "live rank 9 > K=8: displaced by live-only ['STRO'] (no BT feature row)"
    assert "slot_displaced 1" in opp.summary_line(res)


def test_rank_beyond_displacement_stays_unexplained():
    """(b) a bt_only name ranked 10th live with ONE live-only name is not explained by it."""
    scored = dict(LIVE_SCORED, BTCZ=(0.30, "Q5"))     # now below XAAA (0.40): live rank 10
    res = opp.compare(DAY, bt_dict(), live_dict(scored=scored), {})
    assert reasons(res)["BTCZ"] == "unexplained" and res["n_unexplained"] == 1 and opp.is_error(res)


def test_two_live_only_displace_ranks_9_and_10():
    """(c) two live-only names in the live final -> bt_only names at live ranks 9 and 10 are both slot_displaced."""
    bt = bt_dict(ranked=["TNDM", "SKDD", "EQPT", "CABO", "DAMD", "KOLD", "WSE", "BTCZ"])
    scored = dict(LIVE_SCORED, STR2=(0.48, "Q4"))
    final = ["TNDM", "SKDD", "STRO", "STR2", "EQPT", "CABO", "DAMD", "KOLD"]
    res = opp.compare(DAY, bt, live_dict(final=final, scored=scored), {})
    r = reasons(res)
    ranks = opp.live_ranks(live_dict(final=final, scored=scored))
    assert ranks["WSE"] == 9 and ranks["BTCZ"] == 10
    assert r["STRO"] == r["STR2"] == "universe_live_only"
    assert r["WSE"] == r["BTCZ"] == "slot_displaced" and res["n_unexplained"] == 0


def test_no_live_only_name_never_displaces():
    """A bt_only name outside the live top-K with no live-only name in it stays unexplained (no relaxation)."""
    final = ["TNDM", "SKDD", "EQPT", "CABO", "DAMD", "KOLD", "WSE", "XAAA"]
    res = opp.compare(DAY, bt_dict(), live_dict(final=final), {})
    assert reasons(res)["BTCZ"] == "unexplained"


def test_log_result_emits_no_error(caplog):
    """(d) the 11/13 shape is logged at INFO, never ERROR."""
    res = opp.compare(DAY, bt_dict(), live_dict(), {})
    with caplog.at_level(logging.INFO, logger="orb_pick_parity"):
        line = opp.log_result(res)
    assert "slot_displaced 1" in line and "unexplained 0" in line
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert [r for r in caplog.records if r.levelno == logging.INFO]
