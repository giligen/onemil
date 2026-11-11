"""ORB pick-level parity: the live engine's FINAL top-8 must equal the point-in-time BT top-8, every difference named.

Spec: docs/orb_selection_parity_test_spec_20261111.md (cell 1,733). This is precondition 1 of PREREG 1,709 Amendment 2:
10 consecutive sessions with a matched final top-8, every difference carrying a reason from REASONS, zero `unexplained`
and no live fill outside the final set (an add-on-pool fill is the one by-design exception).

BT side  = `eod_sections.bt_ranked(day)`: the newest features CSV re-scored with the parameters in force THAT day
           (`trading.orb_pit_params.params_for_day`, seed + live refit history), pre-veto, dedup, first 8, then the
           PDR / G1 / range-size vetoes. Needs the nightly features build to cover `day`; before that build the BT
           side is reported UNAVAILABLE (a WARNING, never an ERROR) and the day is not counted.
Live side = journal window (UTC 13:30-16:30) UNIONED with logs/session_archive/<day>.log (`eod_sections.engine_log_text`)
           for `[ORB PREPLACE] reconcile final top-K ...`, `ORB SCORED`, `range complete`, `ORB ENTRY SUBMITTED`,
           `[ORB] final ranges:` + data/trades.db (read-only) fills.

Read-only: never writes a file, a DB, a config or the journal. CLI: `--date YYYY-MM-DD [--json out.json]`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eod_sections as es  # noqa: E402  (ONE parser for the engine log; ONE BT ranking replica)

logger = logging.getLogger("orb_pick_parity")

REASONS = ("universe_live_only", "universe_bt_only", "veto_input_range", "veto_input_pdr", "param_ts_mismatch",
           "addon_pool", "provisional_fill", "no_fill", "range_missing", "time_stop_leak", "unexplained")
COMP_TOL = 1e-4                       # |live comp - BT comp| per shared name
JOURNAL_WINDOW_UTC = ("13:30", "16:30")   # covers 09:30-12:30 ET in both EDT and EST
TRADES_DB = ROOT / "data" / "trades.db"
BOOK = ROOT / "analysis_results" / "orb_bplus_book.csv"
MIN_GAP_PCT = 5.0                     # orb.yaml universe.min_gap_pct (the shared trading.orb_universe_gate floor)
TIME_STOP_MINUTES = 60                # orb.yaml entry.time_stop_minutes: an unfilled buy-stop dies at 09:35 + 60 min
_ET = ZoneInfo("America/New_York")
LIVE_PATTERNS = es.ENGINE_LINE_PATTERNS + ("final ranges",)


# ------------------------------------------------------------------ live side
def journal_window_lines(day: str, runner: Callable = subprocess.run) -> List[str]:
    """Engine lines of `onemil-trader` for `day` between 13:30 and 16:30 UTC (bounded `--since/--until`, `-o short-iso`
    so `eod_sections._line_key` dedups it against the archive). [] with a WARNING on any failure."""
    cmd = ["journalctl", "-u", "onemil-trader", "--since", f"{day} {JOURNAL_WINDOW_UTC[0]}",
           "--until", f"{day} {JOURNAL_WINDOW_UTC[1]}", "--no-pager", "-o", "short-iso",
           "-g", "|".join(re.escape(p) for p in LIVE_PATTERNS)]
    try:
        res = runner(cmd, capture_output=True, text=True, timeout=es.JOURNAL_TIMEOUT_S)
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("journalctl failed for %s (%s: %s) - session archive only", day, type(e).__name__, e)
        return []
    if getattr(res, "returncode", 0) not in (0, 1):
        logger.warning("journalctl exit %s for %s: %s", res.returncode, day, (res.stderr or "")[:200])
    return [ln for ln in res.stdout.splitlines()
            if not ln.startswith("-- ") and any(p in ln for p in LIVE_PATTERNS)]


def parse_final_ranges(text: str) -> Dict[str, Tuple[float, float]]:
    """{symbol: (range_high, range_low)} from the `[ORB] final ranges: SYM h/l, ...` line ({} when absent)."""
    out: Dict[str, Tuple[float, float]] = {}
    for ln in text.splitlines():
        m = re.search(r"\[ORB\] final ranges: (.*)$", ln)
        if m:
            for s, h, lo in re.findall(r"([A-Z][A-Z0-9.]*) ([\d.]+)/([\d.]+)", m.group(1)):
                out[s] = (float(h), float(lo))
    return out


def parse_live_vetoes(text: str) -> Dict[str, str]:
    """{symbol: 'pdr'|'g1'|'range'} for the engine's real (non-provisional) post-ranking vetoes."""
    out: Dict[str, str] = {}
    for ln in text.splitlines():
        if "provisional" in ln:
            continue
        m = re.search(r"(PDR|G1|RANGE-SIZE) VETO:? ([A-Z][A-Z0-9.]*)", ln)
        if m:
            out.setdefault(m.group(2), {"PDR": "pdr", "G1": "g1", "RANGE-SIZE": "range"}[m.group(1)])
    return out


def parse_live(text: str) -> Dict:
    """Live-side facts of one session from the engine log text (via `eod_sections.parse_orb_log`)."""
    p = es.parse_orb_log(text)
    prov = set(p["preplace"]["syms"] if p["preplace"] else ())
    for m in re.finditer(r"reconcile final top-\d+ of \d+ ranked: \[.*?\] \| preplaced: \[(.*?)\]", text):
        prov.update(re.findall(r"'([A-Z][A-Z0-9.]*)'", m.group(1)))
    return {"final": sorted(p["final_ranked"]), "provisional": sorted(prov),
            "submitted": sorted(p["submitted"]), "scored": {s: (c, q) for s, (c, q, _) in p["scored_any"].items()},
            "range_done": sorted(p["range_done"]), "final_ranges": parse_final_ranges(text),
            "vetoes": parse_live_vetoes(text), "notes": p["notes"], "seen": text,
            "has_reconcile": "reconcile final top-" in text}


def live_fills(day: str, db_path=TRADES_DB) -> Dict[str, Dict]:
    """{symbol: {pool, composite, filled_at_et, status}} for the ORB rows of `day` with a real fill (read-only DB)."""
    out: Dict[str, Dict] = {}
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error as e:
        logger.warning("trades.db unreadable (%s): live fills unknown for %s", e, day)
        return out
    try:
        rows = con.execute("select symbol, fill_price, filled_at, order_status, pattern_data from trades "
                           "where strategy='orb' and trade_date=? and fill_price is not null", (day,)).fetchall()
    except sqlite3.Error as e:
        logger.warning("trades.db query failed (%s): live fills unknown for %s", e, day)
        return out
    finally:
        con.close()
    for sym, fp, filled_at, status, pdata in rows:
        try:
            pd_ = json.loads(pdata or "{}")
        except ValueError:
            pd_ = {}
        fm = None
        if filled_at:
            try:
                t = dt.datetime.fromisoformat(filled_at)
                t = t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
                fm = t.astimezone(_ET).strftime("%H:%M:%S")
            except ValueError:
                fm = None
        out[sym] = {"pool": pd_.get("pool_id") or pd_.get("pool") or "production",
                    "composite": pd_.get("composite_score"), "filled_et": fm, "status": status}
    return out


# ------------------------------------------------------------------ BT side
def book_rows(day: str, book_path=BOOK) -> Dict[str, Dict]:
    """{symbol: row} of the nightly book for `day` (entered flag, params_refit_ts when the book carries it). {} if absent."""
    try:
        import pandas as pd
        df = pd.read_csv(book_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("BT book %s unreadable (%s): book columns unavailable for %s", book_path, e, day)
        return {}
    df = df[df["date"].astype(str).str[:10] == day]
    return {r["symbol"]: r for r in df.to_dict("records")}


def pit_refit_ts(day: str) -> Tuple[Optional[str], str]:
    """(refit ts, source) in force for `day` per `params_for_day`; (None, why) when it cannot be answered."""
    try:
        from trading.orb_pit_params import params_for_day
        pp = params_for_day(dt.date.fromisoformat(day), current=None, warn=False)
        return pp.refit_ts, pp.source
    except Exception as e:  # noqa: BLE001
        return None, f"unavailable: {type(e).__name__}"


# ------------------------------------------------------------------ comparison
def _diff(symbol: str, side: str, reason: str, detail: str) -> Dict:
    """One named difference; `reason` must be one of REASONS."""
    assert reason in REASONS, reason
    return {"symbol": symbol, "side": side, "reason": reason, "detail": detail}


def _bt_only_reason(s: str, live: Dict, bt: Dict, day_ts_ok: bool, ts_detail: str) -> Dict:
    """Why BT ranked `s` into its top-8 and the live final set does not hold it."""
    seen = live["seen"]
    gap = live["notes"].get("gap:" + s)   # "gap x.xxx%": logged for EVERY candidate; explains only when below the floor
    gm = re.search(r"gap (-?[\d.]+)%", gap or "")
    if gm and float(gm.group(1)) < MIN_GAP_PCT:
        return _diff(s, "bt_only", "universe_bt_only", f"live gap {gm.group(1)}% < {MIN_GAP_PCT}% universe floor; BT admits its feature gap")
    if s not in live["scored"] and s not in live["range_done"] and re.search(rf"\b{re.escape(s)}\b", seen) is None:
        return _diff(s, "bt_only", "universe_bt_only", "no live trace at all (never in the engine's universe)")
    if s not in live["range_done"] and s not in live["final_ranges"]:
        return _diff(s, "bt_only", "range_missing", "live logged no `range complete` / final range for it")
    if s in live["scored"]:
        dc = abs(live["scored"][s][0] - bt["comp"][s][0]) if s in bt["comp"] else None
        if dc is not None and dc > COMP_TOL:
            if not day_ts_ok:
                return _diff(s, "bt_only", "param_ts_mismatch", f"comp live {live['scored'][s][0]:.4f} vs BT {bt['comp'][s][0]:.4f}; {ts_detail}")
            return _diff(s, "bt_only", "unexplained", f"composite differs by {dc:.4f} with matching parameter timestamps")
    return _diff(s, "bt_only", "unexplained", "ranged live, not in the live final top-K, composite not comparable")


def compare(day: str, bt: Optional[Dict], live: Dict, fills: Dict[str, Dict], book: Optional[Dict[str, Dict]] = None,
            pit_ts: Optional[str] = None, pit_src: str = "") -> Dict:
    """The parity verdict of one day (module docstring). `bt` None = BT side unavailable (day not countable)."""
    live_final, subs = list(live["final"]), set(live["submitted"])
    res = {"date": day, "bt_available": bt is not None, "live_final": live_final,
           "live_submitted": sorted(subs), "live_filled": sorted(fills), "differences": [], "n_unexplained": 0,
           "bt_top8": list(bt["ranked"]) if bt else [], "bt_picks": list(bt["picks"]) if bt else [],
           "matched": None, "k": (bt or {}).get("n", 8), "params_refit_ts": (bt or {}).get("params_refit_ts"),
           "params_source": (bt or {}).get("params_source")}
    diffs: List[Dict] = res["differences"]
    # fills outside the live final set (a provisional / post-time-stop leak is a defect; an add-on pool fill is by design)
    out_of_final = []
    for s, f in sorted(fills.items()):
        if s in live_final:
            continue
        if f["pool"] != "production":
            diffs.append(_diff(s, "live_fill", "addon_pool", f"pool {f['pool']} (no BT book for it)"))
            continue
        out_of_final.append(s)
        if f["filled_et"] and f["filled_et"] > "10:35:00":
            diffs.append(_diff(s, "live_fill", "time_stop_leak", f"filled {f['filled_et']} ET, after the {TIME_STOP_MINUTES}-min time stop"))
        elif s in live["provisional"]:
            diffs.append(_diff(s, "live_fill", "provisional_fill", "filled on its provisional order, not in the final top-K"))
        else:
            diffs.append(_diff(s, "live_fill", "unexplained", "fill outside the final top-K, neither provisional nor late"))
    res["fills_outside_final"] = out_of_final
    if bt is None:
        res["n_unexplained"] = sum(d["reason"] == "unexplained" for d in diffs)
        return res
    B, L = set(bt["ranked"]), set(live_final)
    res["matched"] = len(B & L)
    ts_book = {r.get("params_refit_ts") for r in (book or {}).values() if r.get("params_refit_ts") not in (None, "")}
    ts_ok = not (ts_book and pit_ts and ts_book != {pit_ts})
    ts_detail = f"book refit {sorted(ts_book)} vs in force {pit_ts}" if not ts_ok else f"refit {pit_ts or bt.get('params_refit_ts')}"
    # per-name composite check on shared names
    for s in sorted(B & L):
        lc = live["scored"].get(s)
        if lc and abs(lc[0] - bt["comp"][s][0]) > COMP_TOL:
            diffs.append(_diff(s, "both", "param_ts_mismatch" if not ts_ok else "unexplained",
                               f"live comp {lc[0]:.4f} vs BT {bt['comp'][s][0]:.4f}; {ts_detail}"))
    bt_only = {s: _bt_only_reason(s, live, bt, ts_ok, ts_detail) for s in sorted(B - L)}
    diffs.extend(bt_only.values())
    for s in sorted(L - B):
        if s not in bt["day_symbols"]:
            diffs.append(_diff(s, "live_only", "universe_live_only", "no BT feature row for the day (builder/gate dropped it)"))
            continue
        lc, bc = live["scored"].get(s), bt["comp"].get(s)
        if lc and bc is not None and abs(lc[0] - bc[0]) > COMP_TOL and not ts_ok:
            diffs.append(_diff(s, "live_only", "param_ts_mismatch", f"live comp {lc[0]:.4f} vs BT {bc[0]:.4f}; {ts_detail}"))
        elif bt_only:
            by = sorted(bt_only)[0]
            diffs.append(_diff(s, "live_only", bt_only[by]["reason"], f"displaced a slot to BT-only {by}"))
        else:
            diffs.append(_diff(s, "live_only", "unexplained", "in live final top-K, a BT feature row exists, BT ranked it out"))
    # post-veto picks vs submissions
    named = {d["symbol"] for d in diffs}
    for s in sorted(B & L):
        bt_veto, lv = bt["vetoes"].get(s), live["vetoes"].get(s)
        if bt_veto and s in subs and s not in named:
            diffs.append(_diff(s, "live_only", "veto_input_pdr" if bt_veto == "pdr" else "veto_input_range",
                               f"BT {bt_veto} veto, live submitted; live range {live['final_ranges'].get(s)}, BT range_size {bt['range_size'].get(s)}"))
        elif not bt_veto and lv and s not in named:
            diffs.append(_diff(s, "bt_only", "veto_input_pdr" if lv == "pdr" else "veto_input_range",
                               f"live {lv} veto, BT passed it; live range {live['final_ranges'].get(s)}, BT range_size {bt['range_size'].get(s)}"))
        elif not bt_veto and not lv and s in bt["picks"] and s not in subs and s not in named:
            diffs.append(_diff(s, "bt_only", "unexplained", "BT pick, in live final top-K, no live veto and never submitted"))
    # the BT book entered it (its breakout triggered), live submitted it and never got a fill: entry obtainability, not selection
    entered = {s for s, r in (book or {}).items() if int(float(r.get("entered", 0) or 0)) == 1}
    for s in sorted(subs & entered):
        if s not in fills and s in L and s in bt["picks"] and s not in {d["symbol"] for d in diffs}:
            diffs.append(_diff(s, "bt_only", "no_fill", "BT entered; live submitted, no fill"))
    res["n_unexplained"] = sum(d["reason"] == "unexplained" for d in diffs)
    return res


def summary_line(res: Dict) -> str:
    """`ORB pick parity <date>: matched k/8 | diffs n (reasons...) | unexplained m`."""
    if not res["bt_available"]:
        return (f"ORB pick parity {res['date']}: BT side unavailable ({res.get('bt_why', 'features CSV does not cover the day')})"
                f" | live final {len(res['live_final'])} | not counted | unexplained {res['n_unexplained']}")
    reasons: Dict[str, int] = {}
    for d in res["differences"]:
        reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
    rs = ", ".join(f"{r} {n}" for r, n in sorted(reasons.items())) or "none"
    return (f"ORB pick parity {res['date']}: matched {res['matched']}/{res['k']} | diffs {len(res['differences'])} ({rs}) "
            f"| unexplained {res['n_unexplained']}")


def log_result(res: Dict) -> str:
    """Emit the summary: `logger.error` (reaches the owner) ONLY for an unexplained difference or a non-add-on live fill
    outside the final set; a BT-unavailable day is a WARNING; everything explained is INFO. Returns the line."""
    line = summary_line(res)
    bad = [d for d in res["differences"] if d["reason"] == "unexplained"]
    leaks = [s for s in res.get("fills_outside_final", [])]
    if bad or leaks:
        names = sorted({d["symbol"] for d in bad} | set(leaks))
        logger.error("%s | NAMES %s | %s", line, names, "; ".join(f"{d['symbol']}:{d['reason']}" for d in res["differences"]
                                                               if d["reason"] in ("unexplained", "time_stop_leak", "provisional_fill")))
    elif not res["bt_available"]:
        logger.warning(line)
    else:
        logger.info(line)
    return line


def is_error(res: Dict) -> bool:
    """True when `log_result` logged an ERROR for this result."""
    return res["n_unexplained"] > 0 or bool(res.get("fills_outside_final"))


def run_parity(day: str, bt_fn: Callable = es.bt_ranked, log_fn: Callable = es.engine_log_text,
               fills_fn: Callable = live_fills, book_path=BOOK, emit: bool = True) -> Dict:
    """Parity of `day`: assemble both sides, compare, optionally log (emit). Never raises on a missing side."""
    text = log_fn(day, journal_fn=journal_window_lines) if log_fn is es.engine_log_text else log_fn(day)
    live = parse_live(text)
    bt, why = bt_fn(day)
    ts, src = pit_refit_ts(day)
    res = compare(day, bt, live, fills_fn(day), book_rows(day, book_path), ts, src)
    if bt is None:
        res["bt_why"] = why
    res["pit_refit_ts"], res["pit_source"] = ts, src
    if not live["has_reconcile"]:
        res["live_note"] = "no `reconcile final top-K` line in the engine log"
    res["summary"] = log_result(res) if emit else summary_line(res)
    return res


def previous_session_day(day: str) -> str:
    """The prior weekday (holidays resolve to BT-unavailable, which is reported, not alarmed)."""
    d = dt.date.fromisoformat(day) - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d.isoformat()


def main(argv=None) -> int:
    """CLI: print the summary line and the named differences; `--json` also writes the result dict (caller's path)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    res = run_parity(a.date)
    print(res["summary"])
    for d in res["differences"]:
        print(f"  {d['symbol']:<6} {d['side']:<9} {d['reason']:<19} {d['detail']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))
    return 1 if is_error(res) else 0


if __name__ == "__main__":
    sys.exit(main())
