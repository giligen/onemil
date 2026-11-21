"""MOM BT parity seam + window fix (spec docs/mom_bt_parity_seam_20261121.md): the adjustment=ALL extension is rescaled
onto the raw frozen panel at the seam, and the BT week return runs last close before Monday -> end close."""
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "research" / "momentum_weekly"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mom_bt_parity as par  # noqa: E402

COLS = ["symbol", "bar_date", "open", "high", "low", "close", "volume"]


def _series(start, end, base, vol=1000.0):
    """(dates, o, h, l, c, v) float arrays with a gentle drift, as the panel dict stores them."""
    d = pd.bdate_range(start, end)
    c = base * (1 + 0.001 * np.arange(len(d)))
    return (d.values.astype("datetime64[ns]"), c.copy(), c * 1.01, c * 0.99, c.copy(), np.full(len(d), vol))


def _ext(sym, start, end, base, vol=8000.0):
    """Extension frame for one symbol (same drift as _series so a unit scale gives a continuous seam)."""
    d = pd.bdate_range(start, end)
    c = base * (1 + 0.001 * np.arange(len(d)))
    return pd.DataFrame({"symbol": sym, "bar_date": d, "open": c, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "volume": vol})[COLS]


def test_sndk_split_rescaled(caplog):
    """Panel at 1700 scale, extension at 1/8 scale from 9/01: panel part x0.125, volume x8, continuous seam."""
    data = {"SNDK": _series("2026-09-01", "2026-09-29", 1700.0)}
    e = _ext("SNDK", "2026-09-01", "2026-11-20", 212.5)
    n_after = int((e["bar_date"] > "2026-09-29").sum())
    with caplog.at_level(logging.INFO, logger="mom_bt_parity"):
        n = par.extend_data(data, e)
    assert n == n_after
    d, o, h, l, c, v = data["SNDK"]
    i = int(np.searchsorted(d, np.datetime64("2026-09-29")))
    assert c[i] == pytest.approx(float(e.loc[e["bar_date"] == "2026-09-29", "close"].iloc[0]), rel=1e-9)
    assert v[0] == pytest.approx(8000.0)
    assert np.max(np.abs(c[1:] / c[:-1] - 1)) < 0.2
    msgs = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
    assert "SNDK" in msgs and "0.1250" in msgs


def test_small_seam_diff_not_rescaled():
    """0.2 % off at the seam is below SEAM_TOL: untouched."""
    data = {"AAA": _series("2026-09-01", "2026-09-29", 100.0)}
    c0 = data["AAA"][4].copy()
    e = _ext("AAA", "2026-09-01", "2026-10-05", 100.0 * 1.002)
    par.extend_data(data, e)
    assert np.array_equal(data["AAA"][4][:len(c0)], c0)
    d = data["AAA"][0]
    assert len(d) == len(c0) + int((e["bar_date"] > "2026-09-29").sum())
    assert par.rescale_seam({"AAA": _series("2026-09-01", "2026-09-29", 100.0)}, e) == {}


def test_no_seam_bar_appended_unscaled_with_warning(caplog):
    """Extension with no bar on the symbol's last panel date: appended as is, ONE WARNING naming it."""
    data = {"BBB": _series("2026-09-01", "2026-09-29", 100.0)}
    c0 = data["BBB"][4].copy()
    e = _ext("BBB", "2026-09-30", "2026-10-06", 12.5)
    with caplog.at_level(logging.WARNING, logger="mom_bt_parity"):
        n = par.extend_data(data, e)
    assert n == len(e)
    assert np.array_equal(data["BBB"][4][:len(c0)], c0)
    warns = [r for r in caplog.records if r.levelno == logging.WARNING and "BBB" in r.getMessage()]
    assert len(warns) == 1


def test_symbol_absent_from_panel_skipped_silently(caplog):
    """Pinned existing behaviour."""
    data = {"AAA": _series("2026-09-01", "2026-09-29", 100.0)}
    e = _ext("ZZZ", "2026-09-01", "2026-10-05", 50.0)
    with caplog.at_level(logging.WARNING, logger="mom_bt_parity"):
        assert par.extend_data(data, e) == 0
    assert "ZZZ" not in data and not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_real_sndk_seam():
    """Real files, SNDK only (parquet filter, never the full panel): seam within 0.5 %, 11/20 close 79.24."""
    if not (Path(par.PANEL_PATH).exists() and Path(par.EXT_PATH).exists()):
        pytest.skip("panel or extension parquet missing")
    flt = [("symbol", "==", "SNDK")]
    base = pd.read_parquet(par.PANEL_PATH, filters=flt).sort_values("bar_date")
    ext = pd.read_parquet(par.EXT_PATH, filters=flt).sort_values("bar_date")
    if base.empty or ext.empty:
        pytest.skip("SNDK not in both parquets")
    data = {"SNDK": (base["bar_date"].values.astype("datetime64[ns]"),
                     *[base[k].values.astype("float64") for k in ("open", "high", "low", "close", "volume")])}
    par.extend_data(data, ext)
    d, c = data["SNDK"][0], data["SNDK"][4]
    last_base = base["bar_date"].max()
    i = int(np.searchsorted(d, np.datetime64(last_base)))
    ext_c = float(ext.loc[ext["bar_date"] == last_base, "close"].iloc[0])
    assert abs(c[i] / ext_c - 1) < 0.005
    j = int(np.searchsorted(d, np.datetime64("2026-11-20")))
    if j < len(d) and d[j] == np.datetime64("2026-11-20"):
        assert c[j] == pytest.approx(79.24, abs=0.01)


class _Rule:
    """Stub rule object: TOPN names, same eligible rows for each date, px with REBUILD_1700tu's semantics."""
    TOPN = 3

    @staticmethod
    def features(data, cal, ts, _x):
        names = [s for s in data if s != "SPY"]
        return [[(s, float(k), False, 0.0) for k, s in enumerate(names)] for _ in ts]

    @staticmethod
    def px(data, sym, date, use_open):
        d, o, h, l, c, v = data[sym]
        p = np.searchsorted(d, np.datetime64(date), side="right") - 1
        if p < 0:
            return np.nan
        if d[p] == np.datetime64(date) and use_open:
            return o[p]
        return c[p]


def _gappy(dates, base):
    """Bars where every open is 5 % above the previous close (a gap each session)."""
    c = base * (1 + 0.01 * np.arange(len(dates)))
    o = np.concatenate([[c[0]], c[:-1] * 1.05])
    return (dates.values.astype("datetime64[ns]"), o, c * 1.1, o * 0.9, c, np.full(len(dates), 1e6))


def test_bt_week_window_is_close_to_close():
    """ret / spy_ret = last close before Monday -> end close; the Monday-open value must NOT match."""
    cal = pd.bdate_range("2026-09-01", "2026-11-20")
    data = {s: _gappy(cal, b) for s, b in (("SPY", 400.0), ("A", 10.0), ("B", 20.0), ("C", 30.0), ("D", 40.0))}
    w = par.bt_weeks(_Rule, data, n=4)
    assert len(w) >= 3
    for r in w:
        reb, end = r["monday"], r["end"]
        prev = cal[cal.searchsorted(reb) - 1]
        cl = lambda s, t: float(data[s][4][list(cal).index(t)])  # noqa: E731
        want = np.mean([cl(s, end) / cl(s, prev) - 1 for s in r["names"]])
        assert r["ret"] == pytest.approx(want, rel=1e-9)
        assert r["spy_ret"] == pytest.approx(cl("SPY", end) / cl("SPY", prev) - 1, rel=1e-9)
        opn = np.mean([cl(s, end) / float(data[s][1][list(cal).index(reb)]) - 1 for s in r["names"]])
        assert abs(r["ret"] - opn) > 1e-3


def test_format_line_and_is_mismatch_pinned():
    """19 names and -8.2 pp is a MISMATCH; 20 names and -0.7 pp is not."""
    bad = {"date": "2026-11-16", "bt_names_match": 19, "bt_ret": -0.114, "paper_ret": -0.196, "live_ret": -0.151,
           "spy_ret": -0.08, "residual_pp": -8.2}
    ok = dict(bad, bt_names_match=20, residual_pp=-0.7)
    assert par.is_mismatch(bad) and not par.is_mismatch(ok)
    assert par.format_line(bad).startswith("MISMATCH BT parity wk 11/16: names 19/20")
    assert not par.format_line(ok).startswith("MISMATCH")
    assert not math.isnan(bad["residual_pp"])
