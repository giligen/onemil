"""Ranked-set parity on the real 2026-10-01 inputs (DIVE_ranked_set): the engine scores the production pool AND add-on
pool P1 (gap 3-5 %) in one session; the BT features universe is production only (gap >= 5 %). The monitor must compare
production-to-BT. Before the fix the engine-only APPS/CRMG/KD/TDAY (P1, gap 4.45-4.94) made the sets differ."""
import scripts.eod_sections as es

# (symbol, comp, quintile, gap) as logged by the engine at 2026-10-01 13:44-13:48 UTC.
ENGINE_1001 = [("WVE", .4010, "Q4", 6.757), ("MEDS", .0345, "Q1", 23.028), ("CBRX", .5883, "Q5", 5.407),
               ("LPA", .5181, "Q5", 10.140), ("IBX", .4326, "Q5", 10.201), ("RKLX", .1752, "Q2", 7.009),
               ("EFXT", .0843, "Q1", 12.988), ("CNXC", .1855, "Q2", 3.487), ("ADBG", .2784, "Q3", 4.011),
               ("CRMG", .4788, "Q5", 4.943), ("APPS", .5262, "Q5", 4.865), ("KD", .3553, "Q4", 4.618),
               ("TDAY", .4191, "Q4", 4.452), ("BRZE", .0455, "Q1", 4.223), ("DXC", .2355, "Q3", 4.524)]
BT_RANKED_1001 = {"WVE", "CBRX", "LPA", "IBX", "RKLX"}   # study_orb_pipeline_static_lock replica on the 10/1 rows


def _log(pool_tag):
    tag = " pool=production" if pool_tag == "all" else ""
    return "\n".join(
        f"Oct 01 13:44:33 h onemil-trader[1]: 2026-10-01 13:44:33 | INFO | trading.orb_engine:1 | ORB SCORED: {s} "
        f"comp={c:.4f} {q} | gap={g:.3f} rtv=1 | prev_close=1.0 range_open=1.0{tag if g >= 5 else ' pool=addon_gap35_range5'}"
        for s, c, q, g in ENGINE_1001)


def _check(text):
    p = es.parse_orb_log(text)
    assert set(p["scored"]) == {"WVE", "MEDS", "CBRX", "LPA", "IBX", "RKLX", "EFXT"}
    assert {"APPS", "CRMG", "KD", "TDAY"} <= set(p["addon"])
    assert set(es.engine_top_n(p, 8)) == BT_RANKED_1001


def test_ranked_set_matches_bt_with_pool_tag():
    _check(_log("all"))


def test_ranked_set_matches_bt_on_untagged_archive_via_gap_fallback():
    """Archives written before the pool tag: production = gap >= 5 %."""
    _check(_log("all").replace(" pool=production", "").replace(" pool=addon_gap35_range5", ""))
