"""EOD report sections for the paper books and the paper->live promotion gates (spec 2026-10-03).

Three sections, each a list of text lines, each a pure function of its inputs (IO lives in the thin
`*_section` wrappers and in the injectable loaders), so the verdicts are deterministic code:

* `sleeve_section`          MOM  - momentum sleeve paper account: P&L, rotation vs the independent BT top-20,
                            broker/state reconcile, completeness.
* `orb_paper_parity_section` ORB paper account vs the nightly BT book (picks, fills, P&L, defects).
* `promotion_section`       pre-committed GO / HOLD gates, with the rule in brackets; counters persist in
                            logs/promotion_state.json (idempotent per date).

Every missing input is an explicit `NO-DATA (<why>)` line plus a WARNING, never a blank. Nothing here places
orders, sends Telegram or touches config.
"""
from __future__ import annotations

import contextlib
import csv
import io
import datetime as dt
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

log = logging.getLogger("eod_sections")

SLEEVE_START_DATE = "2026-09-29"
SLEEVE_START_EQUITY = 20000.0
SLEEVE_N = 20
SLEEVE_SLIP_MAX_BP = 20.0
SLEEVE_CLEAN_NEEDED = 2
ORB_ENTRY_TOL_BP = 30.0
ORB_CLEAN_NEEDED = 5
RAMP_MIN_DAYS = 20
RAMP_STEP_USD = 10000
REVIEW_FIX_FILES = ("docs/review_20261003/FIX_A_result.md", "docs/review_20261003/FIX_C_result.md")
HOD_LINE = "HOD: dry-run only - no live gate (closed as a money book 9/26)"
PROMOTION_PREFIX = "PROMOTION"

LEDGER = ROOT / "logs" / "momentum_sleeve_ledger.csv"
WEEKLY = ROOT / "logs" / "momentum_sleeve_weekly.csv"
SHADOW_GATE = ROOT / "logs" / "momentum_sleeve_shadow_gate.csv"
SLEEVE_LOG_GLOB = str(ROOT / "logs" / "momentum_sleeve*.log")
SESSION_ARCHIVE = ROOT / "logs" / "session_archive"
PROMOTION_STATE = Path(os.environ.get("ONEMIL_PROMOTION_STATE") or ROOT / "logs" / "promotion_state.json")


def no_data(label: str, why: str) -> str:
    """`<label>: NO-DATA (<why>)` and a WARNING - the single place a missing input is rendered."""
    log.warning("%s: NO-DATA (%s)", label, why)
    return f"{label}: NO-DATA ({why})"


def read_csv_rows(path) -> List[Dict]:
    """Rows of a csv as dicts; [] with a WARNING when the file is missing or unreadable."""
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except OSError as e:
        log.warning("csv unavailable %s: %s", path, e)
        return []


def next_monday(day: str) -> str:
    """The first Monday strictly after `day` (ISO string)."""
    d = dt.date.fromisoformat(day) + dt.timedelta(days=1)
    while d.weekday() != 0:
        d += dt.timedelta(days=1)
    return d.isoformat()


# ---------------------------------------------------------------- sleeve (MOM)
def sleeve_pnl_line(weekly: List[Dict], day: str, peak_hint: float = SLEEVE_START_EQUITY) -> str:
    """`MOM P&L: day | week | since start (equity, start) | DD from peak` from the weekly equity marks.

    The mark of a date is its LAST weekly row; day P&L = that mark minus the last mark of the previous date
    (the start equity if none); week P&L = minus the last mark before this ISO week's Monday."""
    marks: Dict[str, float] = {}
    for r in weekly:
        try:
            marks[r["date"]] = float(r["equity"])
        except (KeyError, ValueError, TypeError):
            log.warning("weekly row unreadable: %r", r)
    if day not in marks:
        return no_data("MOM P&L", f"no equity mark for {day} in the weekly ledger")
    d = dt.date.fromisoformat(day)
    monday = (d - dt.timedelta(days=d.weekday())).isoformat()
    prior = [k for k in marks if k < day]
    before_week = [k for k in marks if k < monday]
    eq = marks[day]
    day_base = marks[max(prior)] if prior else SLEEVE_START_EQUITY
    week_base = marks[max(before_week)] if before_week else SLEEVE_START_EQUITY
    peak = max([peak_hint] + [v for k, v in marks.items() if k <= day])
    dd = (peak - eq) / peak * 100 if peak else 0.0
    return (f"MOM P&L: day ${eq - day_base:+,.0f} | week ${eq - week_base:+,.0f} | since {SLEEVE_START_DATE} "
            f"${eq - SLEEVE_START_EQUITY:+,.0f} (equity ${eq:,.0f}, start ${SLEEVE_START_EQUITY:,.0f}) | "
            f"DD from peak {dd:.1f} %")


def rotation_stats(ledger: List[Dict], day: str) -> Optional[Dict]:
    """Fills of `day` from the sleeve ledger: names, count, slip mean/max (bp, absolute max) and size label.
    None when the ledger has no row for the day (non-rotation day)."""
    rows = [r for r in ledger if r.get("date") == day]
    if not rows:
        return None
    slips = []
    for r in rows:
        try:
            slips.append(float(r["slip_bps_vs_open"]))
        except (KeyError, ValueError, TypeError):
            log.warning("ledger row without slip: %r", r.get("symbol"))
    size = next((r.get("size_pct") for r in rows if r.get("size_pct")), "")
    return {"names": sorted({r["symbol"] for r in rows}), "n": len(rows),
            "slip_mean": sum(slips) / len(slips) if slips else None,
            "slip_max": max((abs(s) for s in slips), default=None),
            "size": f"{float(size):.0f}%" if size else "n/a"}


FORCE_TAG_RE = re.compile(r"-f\d{6}$")
SCHEDULE_WINDOW_UTC = ("13:40", "15:00")


def is_scheduled_rotation(day: str, ledger: List[Dict], log_texts: Sequence[str]) -> Tuple[bool, str]:
    """A rotation counts only when scheduled: `day` is a Monday, no ledger order id of that day carries the
    force tag (`-f<HHMMSS>`), and a sleeve log line of the day is stamped 13:40-15:00 UTC (the run started in
    the window). Returns (scheduled, why-not)."""
    if dt.date.fromisoformat(day).weekday() != 0:
        return False, "not a Monday"
    if any(FORCE_TAG_RE.search(r.get("client_order_id", "")) for r in ledger if r.get("date") == day):
        return False, "force-tagged order ids"
    stamps = [m.group(1) for t in log_texts for ln in t.splitlines() if ln.startswith(day)
              for m in [re.match(r"\S+ (\d\d:\d\d)", ln)] if m]
    if not any(SCHEDULE_WINDOW_UTC[0] <= h <= SCHEDULE_WINDOW_UTC[1] for h in stamps):
        return False, "no log line in the 13:40-15:00 UTC window"
    return True, ""


def bt_top20(day: str, panel_loader: Optional[Callable] = None, assets_loader: Optional[Callable] = None
             ) -> Tuple[Optional[List[str]], str]:
    """INDEPENDENT recomputation of the sleeve's top-20 for the rotation `day`: the latest cached panel
    strictly before `day` (asof), `eligible_universe(guard on)` + `risk_adjusted_momentum` + `select_top` -
    the functions the recon fixtures test. Returns (names, asof) or (None, why)."""
    import pandas as pd
    from trading import momentum_sleeve as ms
    import momentum_sleeve as sc
    asofs = sorted(p for p in glob.glob(os.path.join(sc.DATA_DIR, "daily_*.parquet")))
    cands = [re.search(r"daily_(\d{8})", p).group(1) for p in asofs]
    cands = [f"{c[:4]}-{c[4:6]}-{c[6:]}" for c in cands if f"{c[:4]}-{c[4:6]}-{c[6:]}" < day]
    if not cands:
        return None, f"no panel cache dated before {day}"
    asof = dt.date.fromisoformat(cands[-1])
    try:
        panel = (panel_loader or (lambda a: pd.read_parquet(sc.cache_file(a))))(asof)
        assets = (assets_loader or (lambda: None))()
        if assets is None:
            return None, "assets listing unavailable (broker read failed)"
        feat = ms.risk_adjusted_momentum(panel, asof)
        elig = [s for s in ms.eligible_universe(panel, asof, assets) if s != "SPY"]
        return ms.select_top(feat.loc[elig, "signal"], SLEEVE_N), str(asof)
    except Exception as e:  # noqa: BLE001
        log.warning("bt_top20 failed: %s", e)
        return None, f"recompute failed: {e}"


def rotation_line(day: str, stats: Dict, picks: Sequence[str], bt: Optional[Sequence[str]], bt_why: str,
                  gate: str, missing: int = 0) -> Tuple[str, bool]:
    """`MOM rotation <day>: picks n/20 = BT top-20 (diff) | fills | slip | size | gate` and whether picks == BT."""
    if bt is None:
        pk = f"picks {len(picks)}/{SLEEVE_N} BT top-20 NO-DATA ({bt_why})"
        same = False
    else:
        plus, minus = sorted(set(picks) - set(bt)), sorted(set(bt) - set(picks))
        same = not plus and not minus
        inter = len(set(picks) & set(bt))
        pk = (f"picks {inter}/{SLEEVE_N} = BT top-20"
              + ("" if same else f" (diff: {' '.join('+' + s for s in plus)} {' '.join('-' + s for s in minus)})"))
    n = len(stats["names"])
    slip = (f"slip mean {stats['slip_mean']:.1f} bp (max {stats['slip_max']:.1f})"
            if stats["slip_mean"] is not None else "slip NO-DATA (no slip in ledger)")
    return (f"MOM rotation {day}: {pk} | fills {n}/{n + missing} | {slip} | size {stats['size']} | {gate}", same)


def reconcile_line(broker: Optional[Dict[str, Tuple[float, float]]], state_pos: Dict[str, float],
                   state_prices: Dict[str, float]) -> Tuple[str, Optional[bool]]:
    """`MOM reconcile: broker n names $x vs state n names $x | OK|MISMATCH ...`.

    `broker` = {symbol: (qty, market_value)}; MISMATCH when the name sets differ, a quantity differs by more than
    0.1 %, or the dollar totals differ by more than 1 %. None broker = NO-DATA (verdict None)."""
    if broker is None:
        return no_data("MOM reconcile", "broker positions unavailable"), None
    st = {s: q for s, q in state_pos.items() if q and q > 1e-6}
    st_usd = sum(q * state_prices.get(s, 0.0) for s, q in st.items())
    b_usd = sum(mv for _, mv in broker.values())
    problems = [f"name {s}" for s in sorted(set(st) ^ set(broker))]
    for s in sorted(set(st) & set(broker)):
        bq = broker[s][0]
        if abs(bq - st[s]) > 1e-3 * max(abs(bq), 1e-9):
            problems.append(f"qty {s} broker {bq:.4f} state {st[s]:.4f}")
    if st_usd and abs(b_usd - st_usd) / st_usd > 0.01:
        problems.append(f"$ differs {abs(b_usd - st_usd):,.0f}")
    verdict = "OK" if not problems else "MISMATCH " + "; ".join(problems[:6])
    return (f"MOM reconcile: broker {len(broker)} names ${b_usd:,.0f} vs state {len(st)} names ${st_usd:,.0f} | "
            f"{verdict}", not problems)


def completeness_from_logs(log_texts: Sequence[str], day: str) -> Dict:
    """Parse the sleeve logs of `day`: last `COMPLETENESS: requested R symbols ... LOST n`, liquid ratio when the
    line carries it, and whether a `MOM REFUSED` / `REFUSED` line exists. Returns {} when no COMPLETENESS line."""
    out: Dict = {"refused": False}
    pat = re.compile(r"COMPLETENESS: requested (\d+) symbols.*?LOST (\d+)(?:.*?liquid[^0-9]*(\d+(?:\.\d+)?)\s*%)?")
    for text in log_texts:
        for ln in text.splitlines():
            if not ln.startswith(day) and not ln.startswith("COMPLETENESS"):
                continue
            if "REFUSED" in ln:
                out["refused"] = True
            m = pat.search(ln)
            if m:
                out.update(requested=int(m.group(1)), lost=int(m.group(2)),
                           liquid=float(m.group(3)) if m.group(3) else None)
    return out if "requested" in out or out["refused"] else {}


def completeness_line(info: Dict) -> Tuple[str, Optional[bool]]:
    """`MOM completeness: LOST x %, liquid x % | OK|REFUSED`; thresholds LOST <= 5 %, liquid >= 98 % (the
    sleeve's own --submit gate). A missing liquid share is stated, not guessed."""
    if not info:
        return no_data("MOM completeness", "no COMPLETENESS line in the sleeve logs"), None
    lost = 100.0 * info.get("lost", 0) / max(info.get("requested", 1), 1) if "requested" in info else None
    liquid = info.get("liquid")
    bad = info.get("refused") or (lost is not None and lost > 5.0) or (liquid is not None and liquid < 98.0)
    lost_t = f"{lost:.1f} %" if lost is not None else "n/a"
    liq_t = f"{liquid:.1f} %" if liquid is not None else "not in log"
    return f"MOM completeness: LOST {lost_t}, liquid {liq_t} | {'REFUSED' if bad else 'OK'}", not bad


def gate_label_for(day: str, rows: List[Dict], log_texts: Sequence[str]) -> str:
    """`gate pNN` from the shadow-gate csv row of `day`; falls back to the log's `gate ON/OFF (pNN)`; else `gate n/a`."""
    for r in reversed(rows):
        if r.get("run_date") == day and r.get("percentile"):
            try:
                return f"gate p{float(r['percentile']) * 100:.0f}"
            except ValueError:
                break
    for text in log_texts:
        m = re.findall(r"gate (?:STALE )?(?:ON|OFF) \(p(\d+)\)", text)
        if m:
            return f"gate p{m[-1]}"
    log.warning("shadow gate: no row for %s in %s and none in the logs", day, SHADOW_GATE)
    return "gate n/a"


def broker_positions(client) -> Optional[Dict[str, Tuple[float, float]]]:
    """Read-only broker positions {symbol: (qty, market_value)}; None with a WARNING on any failure."""
    try:
        return {p.symbol: (float(p.qty), float(p.market_value)) for p in client.trading_client.get_all_positions()
                if abs(float(p.qty)) > 1e-6}
    except Exception as e:  # noqa: BLE001
        log.warning("MOM broker positions failed: %s", e)
        return None


def _mom_client():
    """The sleeve's own paper client (scripts/momentum_sleeve.py main() factory), or None with a WARNING."""
    try:
        from config import Config
        from data_sources.alpaca_client import AlpacaClient
        Config()  # loads .env
        key, secret = os.environ.get("ALPACA_MOM_API_KEY", ""), os.environ.get("ALPACA_MOM_API_SECRET", "")
        if not key or not secret:
            log.error("ALPACA_MOM_API_KEY/SECRET not set - MOM broker reads unavailable")
            return None
        return AlpacaClient(key, secret, paper=True)
    except Exception as e:  # noqa: BLE001
        log.warning("MOM client unavailable: %s", e)
        return None


def sleeve_section(day: str, client=None, ledger_path=LEDGER, weekly_path=WEEKLY, state_path=None,
                   bt_fn: Optional[Callable] = None) -> Tuple[List[str], Dict]:
    """The MOM section lines plus the metrics the promotion gate reads.

    metrics: {'rotation': bool, 'clean': bool|None, 'why': first failing condition or ''}."""
    import momentum_sleeve as sc
    state_path = state_path or sc.STATE_PATH
    client = client if client is not None else _mom_client()
    lines = [sleeve_pnl_line(read_csv_rows(weekly_path), day)] if os.path.exists(weekly_path) else \
        [no_data("MOM P&L", f"{Path(weekly_path).name} missing")]
    ledger = read_csv_rows(ledger_path)
    stats = rotation_stats(ledger, day)
    try:
        state = json.load(open(state_path))
    except (OSError, ValueError) as e:
        log.warning("sleeve state unreadable %s: %s", state_path, e)
        state = None
    broker = broker_positions(client) if client is not None else None
    # state value at broker marks when available, else at the ledger's last fill price
    prices = {s: (mv / q if q else 0.0) for s, (q, mv) in (broker or {}).items()}
    for r in ledger:
        try:
            prices.setdefault(r["symbol"], float(r["avg_price"]))
        except (KeyError, ValueError):
            pass
    texts = [open(p, errors="replace").read() for p in sorted(glob.glob(SLEEVE_LOG_GLOB))]
    scheduled, not_sched = (is_scheduled_rotation(day, ledger, texts) if stats is not None else (False, ""))
    metrics: Dict = {"rotation": scheduled, "clean": None, "why": ""}
    reasons: List[str] = []
    if stats is not None:
        if not scheduled:
            lines.append(f"MOM forced run (not counted): {day} {len(stats['names'])} names filled ({not_sched})")
        picks = [s for s, q in (state or {}).get("positions", {}).items() if q and q > 1e-6] \
            if state and state.get("last_rebalance") == day else stats["names"]
        assets_loader = (lambda: sc.fetch_assets(client)) if client is not None else (lambda: None)
        bt, bt_why = (bt_fn or (lambda d: bt_top20(d, assets_loader=assets_loader)))(day)
        missing = len([s for s in (bt or []) if s not in (state or {}).get("positions", {})]) if state else 0
        rl, same = rotation_line(day, stats, picks, bt, bt_why if bt is None else "",
                                 gate_label_for(day, read_csv_rows(SHADOW_GATE), texts), missing)
        lines.append(rl)
        cl, comp_ok = completeness_line(completeness_from_logs(texts, day))
        if not same:
            reasons.append("picks != BT top-20" if bt is not None else "BT top-20 unavailable")
        if missing:
            reasons.append(f"{missing} picks not filled")
        if stats["slip_mean"] is None:
            reasons.append("slip n/a")
        elif stats["slip_mean"] > SLEEVE_SLIP_MAX_BP:
            reasons.append(f"slip mean {stats['slip_mean']:.0f} bp > {SLEEVE_SLIP_MAX_BP:.0f}")
    rec, rec_ok = reconcile_line(broker, (state or {}).get("positions", {}), prices) if state else \
        (no_data("MOM reconcile", "state file unreadable"), None)
    lines.append(rec)
    if stats is not None:
        lines.append(cl)
    if scheduled:
        if rec_ok is not True:
            reasons.append("reconcile not OK")
        if comp_ok is not True:
            reasons.append("completeness not OK")
        metrics.update(clean=not reasons, why=reasons[0] if reasons else "")
    return lines, metrics


# ---------------------------------------------------------------- ORB paper parity
DECISION_WINDOW_ET = ("09:34", "10:00")   # last_entry_submit_time
LATE_AFTER_ET = "09:40"                    # a first scoring after this is a decision, flagged late
_STAMP_RE = re.compile(r"(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d) \|")


def _et_hhmm(line: str) -> Optional[str]:
    """HH:MM:SS in US/Eastern of the engine's own `YYYY-MM-DD HH:MM:SS |` stamp (UTC on this node); None if absent."""
    from zoneinfo import ZoneInfo
    m = _STAMP_RE.search(line)
    if not m:
        return None
    t = dt.datetime.fromisoformat(f"{m.group(1)} {m.group(2)}").replace(tzinfo=dt.timezone.utc)
    return t.astimezone(ZoneInfo("US/Eastern")).strftime("%H:%M:%S")


P1_ID = "P1"
PRODUCTION_MIN_GAP_PCT = 5.0   # orb.yaml universe.min_gap_pct; the BT features universe (orb_backtest.MIN_GAP_PCT)


def _is_production_scored(line: str) -> bool:
    """True when an `ORB SCORED` line belongs to the PRODUCTION pool, the only pool the BT book models.
    The engine ranks each add-on pool separately (trading/orb_engine.py `_run_pool_selection`, pool_label) and
    logs `pool=<label>`; archives written before that tag fall back to the gap field (production gap >= 5 %,
    add-on P1 gap 3-5 %). A line with neither tag nor gap is treated as production (old format)."""
    m = re.search(r"\bpool=(\S+)", line)
    if m:
        return m.group(1) == "production"
    g = re.search(r"\| gap=(-?[\d.]+)", line)
    return g is None or float(g.group(1)) >= PRODUCTION_MIN_GAP_PCT


def _p1_pool_names() -> Tuple[str, ...]:
    """Names/ids of the add-on pool P1 from orb.yaml, through the engine's own pool reader."""
    from trading.orb_pool_defs import load_addon_pools
    names: List[str] = []
    for p in load_addon_pools(str(ROOT / "orb.yaml"))["pools"]:
        if p.get("pool_id") == P1_ID:
            names += [p.get("name"), p.get("pool_id")]
    return tuple(n for n in names if n)


def _is_p1_scored(line: str) -> bool:
    """True when an add-on `ORB SCORED` line carries pool=<P1 name or id> (the tag the engine logs)."""
    m = re.search(r"\bpool=(\S+)", line)
    return bool(m) and m.group(1) in _p1_pool_names()


def parse_orb_log(text: str) -> Dict:
    """Counts from an ORB session-archive text. `scored` = symbols of `ORB SCORED` lines stamped inside the
    09:34-09:40 ET window (the 09:35 decision); `late` = SCORED symbols outside it (a restart, not a decision);
    `boots` = UTC HH:MM of each engine boot (`WINNER STACK` line). Also Engine tick TIMEOUT, GAP_GATE WARNING and
    ORB ERROR counts. The archive is grep-filtered by cron, so the TIMEOUT count is a floor."""
    lines = text.splitlines()
    scored, late, detail, first_et, addon = set(), set(), {}, None, set()
    p1_scored, p1_detail = set(), {}
    for ln in lines:
        m = re.search(r"ORB SCORED: (\S+) comp=([-\d.]+) (Q\d)?", ln)
        m0 = m or re.search(r"ORB SCORED: (\S+)", ln)
        if not m0:
            continue
        if not _is_production_scored(ln):
            addon.add(m0.group(1))
            et1 = _et_hhmm(ln)
            if (m and m.group(3) and et1 and DECISION_WINDOW_ET[0] <= et1[:5] <= DECISION_WINDOW_ET[1]
                    and _is_p1_scored(ln)):
                p1_scored.add(m0.group(1))
                p1_detail[m0.group(1)] = (float(m.group(2)), m.group(3))
            continue
        et = _et_hhmm(ln)
        if et and DECISION_WINDOW_ET[0] <= et[:5] <= DECISION_WINDOW_ET[1]:
            scored.add(m0.group(1))
            first_et = min(first_et, et) if first_et else et
            if m and m.group(3):
                detail[m0.group(1)] = (float(m.group(2)), m.group(3))
        else:
            late.add(m0.group(1))
    boots = []
    for ln in lines:
        if "WINNER STACK" in ln:
            m = _STAMP_RE.search(ln)
            if m:
                boots.append(m.group(2)[:5])
    return {"scored": sorted(scored), "late": sorted(late - scored), "boots": boots, "addon": sorted(addon - scored),
            "detail": detail, "first_et": first_et[:5] if first_et else None,
            "p1_scored": sorted(p1_scored), "p1_detail": p1_detail,
            "timeouts": sum("Engine tick TIMEOUT" in ln for ln in lines),
            "gap_warn": sum("GAP_GATE" in ln and "| WARNING" in ln for ln in lines),
            "errors": sum("| ERROR" in ln and "ORB" in ln for ln in lines),
            "order_fail": order_failures(lines)}


ORDER_FAIL_RE = re.compile(r"ORB: (\S+) (submit_entry failed|alpaca submit returned empty)[:]? ?(.*)$")


def order_failures(lines: List[str]) -> List[Tuple[str, str]]:
    """(symbol, reason) for every ORB entry whose submit raised or returned empty (2026-10-05: a client signature
    mismatch failed every entry for three sessions and the report only counted ERRORs). Reason truncated to 120 chars."""
    out = []
    for ln in lines:
        m = ORDER_FAIL_RE.search(ln)
        if m:
            out.append((m.group(1), (m.group(3) or m.group(2)).strip()[:120]))
    return out


def bps(a: float, b: float) -> float:
    """Signed distance of a from b in bp of b."""
    return (a - b) / b * 1e4


def bt_ranked(day: str, features_csv: Optional[str] = None, n: Optional[int] = None,
              min_move_to_range_high: Optional[float] = None) -> Tuple[Optional[Dict], str]:
    """The BT's RANKED candidates for `day`, recomputed from the newest features CSV with the static-lock
    pipeline's own steps (replica of study_orb_pipeline_static_lock.main(), live orb.yaml params): composite >=
    threshold, not Q1, order Q4,Q5,Q3,Q2 then composite, family/super-group dedup, first N. Then the post-ranking
    vetoes (PDR, G1, range-size), no refill. Returns {'ranked', 'picks', 'rows', 'pdr', 'g1', 'range', 'dedup'}
    or (None, why) when the features CSV does not cover the day or the replica fails.
    `n` overrides the slot count (add-on pool: the whole scored set); `min_move_to_range_high` applies an add-on
    pool's 09:35 gate (NaN fails closed) before scoring, as the engine does."""
    try:
        import pandas as pd
        import yaml
        from trading.orb_csv import read_orb_csv
        from study_orb_filter import FILTER_FEATURES, composite_score
        from study_orb_sizing import assign_quintile
        from study_orb_pipeline_static_lock import load_bt_config
        from trading.orb_pdr_veto import pdr_veto_applies
        from trading.orb_g1_veto import g1_reject
        from trading.orb_range_size_veto import range_size_veto_applies
        from study_orb_correlation_filter import symbol_family, symbol_super_group
        files = sorted(glob.glob(str(ROOT / "analysis_results" / "orb_features_2*.csv")))
        path = features_csv or (files[-1] if files else None)
        if not path:
            return None, "no features CSV"
        with contextlib.redirect_stdout(io.StringIO()):   # load_bt_config prints its config banner
            cfg = load_bt_config()
        y = yaml.safe_load(open(ROOT / "orb.yaml"))
        raw = read_orb_csv(path)
        raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
        n_rows = int((raw["date"] == day).sum())
        if not n_rows:
            return None, f"features CSV has no rows for {day}"
        need = [f for f, _ in FILTER_FEATURES]
        df = raw.dropna(subset=need + ["pnl", "date", "pnl_pct", "range_size_pct", "entry_price"]).copy()
        if min_move_to_range_high is not None:
            df = df[df["move_to_range_high_pct"] >= float(min_move_to_range_high)].copy()
        fe = y["filter"]["features"]
        params = {f: {"mean": float(fe[f]["mean"]), "std": float(fe[f]["std"]), "sign": int(fe[f]["sign"])}
                  for f in need}
        df["c"] = composite_score(df, params)
        g = df[df["date"] == day]
        k = g[g["c"] >= cfg["threshold"]].copy()
        k["q"] = assign_quintile(k["c"], [float(x) for x in y["quintile_cutoffs"]])
        k = k[k["q"] != "Q1"]
        k = k.assign(qr=k["q"].map({"Q4": 0, "Q5": 1, "Q3": 2, "Q2": 3})).sort_values(["qr", "c"],
                                                                                    ascending=[True, False])
        sel, fams, grps, dedup = [], set(), set(), 0
        for _, r in k.iterrows():
            f, sg = symbol_family(r.symbol), symbol_super_group(r.symbol)
            if (f and f in fams) or (sg and sg in grps):
                dedup += 1
                continue
            fams.add(f) if f else None
            grps.add(sg) if sg else None
            sel.append(r)
            if len(sel) >= (n or cfg["n"]):
                break
        ranked = [r.symbol for r in sel]
        pdr = [r for r in sel if pdr_veto_applies(None if pd.isna(r.prev_day_range_pct)
                                                   else float(r.prev_day_range_pct), cfg["pdr_min"])]
        gone = {r.symbol for r in pdr}
        g1 = [r for r in sel if r.symbol not in gone and g1_reject(
            r.return_volatility_20d, r.prev_day_range_pct, cfg["g1_rv20_min"], cfg["g1_pdr_min"],
            short_history_veto=cfg["g1_short_history_veto"])]
        gone |= {r.symbol for r in g1}
        rs = [r for r in sel if r.symbol not in gone and range_size_veto_applies(r.range_size_pct, cfg["rs_min"])]
        gone |= {r.symbol for r in rs}
        picks = [r.symbol for r in sel if r.symbol not in gone]
        return {"ranked": ranked, "picks": picks, "rows": n_rows, "pdr": len(pdr), "g1": len(g1),
                "range": len(rs), "dedup": dedup, "n": (n or cfg["n"])}, ""
    except Exception as e:  # noqa: BLE001
        log.warning("bt_ranked failed: %s", e, exc_info=True)
        return None, f"BT ranking replica failed: {type(e).__name__}: {e}"


def funnel_line(b: Dict) -> str:
    """`BT: rows r -> top-8 -> vetoed k (PDR a, G1 b, range c, dedup d) -> picks p`."""
    k = b["pdr"] + b["g1"] + b["range"]
    return (f"BT: rows {b['rows']} -> top-{b['n']} ({len(b['ranked'])}) -> vetoed {k} (PDR {b['pdr']}, G1 {b['g1']}, "
            f"range {b['range']}, dedup {b['dedup']}) -> picks {len(b['picks'])}")


def engine_top_n(parsed: Dict, n: int, dedup: Callable = None) -> Optional[List[str]]:
    """The engine's top-`n` by its LOGGED score, in the pipeline's order (Q4, Q5, Q3, Q2, then composite desc;
    Q1 excluded; family/super-group dedup). None when any scored symbol lacks a logged comp/quintile."""
    det = parsed.get("detail", {})
    if not parsed["scored"] or any(sym not in det for sym in parsed["scored"]):
        return None
    from study_orb_correlation_filter import symbol_family, symbol_super_group
    order = {"Q4": 0, "Q5": 1, "Q3": 2, "Q2": 3}
    cands = sorted(((order[q], -c, sym) for sym, (c, q) in det.items() if q in order))
    out, fams, grps = [], set(), set()
    for _, _, sym in cands:
        f, g = symbol_family(sym), symbol_super_group(sym)
        if (f and f in fams) or (g and g in grps):
            continue
        fams.add(f) if f else None
        grps.add(g) if g else None
        out.append(sym)
        if len(out) >= n:
            break
    return out


def orb_parity_lines(day: str, engine_rows: List[Dict], parsed: Dict, bt_rows: Optional[List[Dict]],
                     bt_why: str = "", bt_rank: Optional[Dict] = None, rank_why: str = "") -> Tuple[List[str], Dict]:
    """The ORB PAPER PARITY lines and metrics.

    metrics['clean'] is tri-state: True (counts), False (resets), None (neutral: NO DECISION or BT NO-DATA).
    A session is clean only if the engine's RANKED set (ORB SCORED symbols, 09:34-09:40 ET) equals the BT's ranked
    top-N; with BT picks present the picks, fills (<= 30 bp), TIMEOUT and ERROR rules apply too. A 0-pick day with
    a ranked match is clean (the engine really did rank the same names)."""
    m: Dict = {"decision": bool(parsed["scored"]), "match": False, "timeouts": parsed["timeouts"],
               "errors": parsed["errors"], "clean": None, "why": "", "tol_ok": None}
    defects = (f"ORB defects: Engine tick TIMEOUT {parsed['timeouts']} | GAP_GATE WARN {parsed['gap_warn']} | "
               f"ERROR {parsed['errors']}")
    lines: List[str] = []
    fails = parsed.get("order_fail", [])
    if fails:
        # ACTION line first: every failed submit is a lost pick, never a count to skim past.
        lines.append(f"ORB ACTION: {len(fails)} entry submit(s) FAILED -- "
                     + "; ".join(f"{sym}: {why}" for sym, why in fails[:3])
                     + (" ..." if len(fails) > 3 else "") + " -- fix before the next session")
    if not parsed["scored"]:
        restart = [b for b in parsed.get("boots", []) if b >= "13:31"]
        why = (f"no SCORED line in the {DECISION_WINDOW_ET[0]}\u2013{DECISION_WINDOW_ET[1]} ET window"
               + (f"; restart {restart[0]} UTC" if restart else "")
               + (f"; {len(engine_rows)} paper rows in trades" if engine_rows else ""))
        lines.append(f"ORB picks: NO DECISION ({why})")
        if parsed.get("late"):
            lines.append(f"ORB late scoring, not a decision: {' '.join(parsed['late'])}")
        lines.append(defects)
        m["why"] = "no 09:35 decision (neutral)"
        return lines, m
    eng_scored = set(parsed["scored"])
    first = parsed.get("first_et")
    late_flag = f" | late decision ({first} ET)" if first and first > LATE_AFTER_ET else ""
    top = engine_top_n(parsed, bt_rank["n"]) if bt_rank else None
    eng = set(top) if top is not None else eng_scored
    if bt_rank is None:
        lines.append(f"ORB ranked: engine {len(eng)} vs {no_data('BT ranked', rank_why)}")
        m["why"] = "BT ranked set NO-DATA (neutral)"
        lines.append(defects)
        return lines, m
    bt = set(bt_rank["ranked"])
    subset = top is None
    m["match"] = bt <= eng_scored if subset else eng == bt
    note = " (subset test - engine scores not logged)" if subset else ""
    lines.append(f"ORB ranked: engine {len(eng)} vs BT {len(bt)} | match {len(eng & bt)} | "
                 f"engine-only: {' '.join(sorted(eng - bt)) or '-'} | BT-only: {' '.join(sorted(bt - eng)) or '-'}"
                 f"{note}{late_flag}")
    lines.append(funnel_line(bt_rank))
    traded = {r["symbol"] for r in engine_rows if r.get("order_status") not in ("cancelled", "canceled", "rejected")}
    bt_picks = set(bt_rank["picks"])
    reasons: List[str] = []
    if not m["match"]:
        reasons.append("ranked set != BT")
    if bt_picks or traded:
        lines.append(f"ORB picks: engine {len(traded)} vs BT {len(bt_picks)} | match {len(traded & bt_picks)} | "
                     f"engine-only: {' '.join(sorted(traded - bt_picks)) or '-'} | "
                     f"BT-only: {' '.join(sorted(bt_picks - traded)) or '-'}")
        if traded != bt_picks:
            reasons.append("picks != BT")
        filled = [r for r in engine_rows if r.get("fill_price")]
        diffs = []
        if bt_rows:
            btp = {r["symbol"]: float(r["entry_price"]) for r in bt_rows if r.get("entry_price")}
            diffs = [bps(float(r["fill_price"]), btp[r["symbol"]]) for r in filled if r["symbol"] in btp]
        addon = sum(1 for r in engine_rows if "production" not in _pool(r))
        if diffs:
            mean = sum(diffs) / len(diffs)
            m["tol_ok"] = abs(mean) <= ORB_ENTRY_TOL_BP
            diff_t = f"entry diff vs BT entry: mean {mean:+.1f} bp (max {max(abs(d) for d in diffs):.1f})"
            if not m["tol_ok"]:
                reasons.append(f"entry diff > {ORB_ENTRY_TOL_BP:.0f} bp")
        else:
            diff_t = "entry diff vs BT entry: " + no_data("fills", "no engine fill matched a BT entry")
            if bt_picks:
                reasons.append("no fills to compare")
        lines.append(f"ORB fills: {len(filled)}/{len(traded)} picks filled | {diff_t} | tilt mults engine vs BT: "
                     f"{no_data('mults', 'BT book carries no mult column')} | add-on events {addon} (BT n/a)")
    else:
        lines.append("ORB picks: engine 0 vs BT 0 (0 = 0, ranked set decides)")
    closed = [r for r in engine_rows if r.get("exit_price") is not None]
    pnl = sum(float(r.get("pnl") or 0) for r in closed)
    bt_pnl = (f"${sum(float(r.get('pnl') or 0) for r in bt_rows):+,.0f}" if bt_rows else "$+0" if bt_rows == [] else "NO-DATA")
    lines.append(f"ORB P&L: day ${pnl:+,.0f} on {len(closed)} exits | BT book {bt_pnl}")
    lines.append(defects)
    if parsed.get("order_fail"):
        reasons.insert(0, f"entry submit FAILED x{len(parsed['order_fail'])}")
    if parsed["timeouts"]:
        reasons.append(f"Engine tick TIMEOUT {parsed['timeouts']}")
    if parsed["errors"]:
        reasons.append(f"ERROR {parsed['errors']}")
    m["why"] = reasons[0] if reasons else ""
    m["clean"] = not reasons
    return lines, m


def _pool(row: Dict) -> str:
    """pool_id carried by a trades row's pattern_data ('production' when absent)."""
    try:
        return json.loads(row.get("pattern_data") or "{}").get("pool_id") or "production"
    except (TypeError, ValueError):
        return "production"


def features_cover(day: str) -> bool:
    """True when the newest nightly features CSV (analysis_results/orb_features_<date>_<time>.csv, the BT's input)
    has rows for `day`. The BT book lists SELECTED picks only, so a day without book rows is a day with zero BT
    picks - but only if the nightly run actually covered that day; this tells the two apart."""
    from trading.orb_csv import read_orb_csv
    files = sorted(glob.glob(str(ROOT / "analysis_results" / "orb_features_2*.csv")))
    if not files:
        log.warning("no orb_features_2*.csv found - cannot tell zero BT picks from a stale book")
        return False
    try:
        return day in set(read_orb_csv(files[-1], usecols=["date"])["date"].astype(str))
    except Exception as e:  # noqa: BLE001
        log.warning("features csv unreadable %s: %s", files[-1], e)
        return False


def load_bt_rows(day: str, csv_path: Optional[str] = None,
                 covered: Callable[[str], bool] = features_cover) -> Tuple[Optional[List[Dict]], str]:
    """Rows of the nightly ORB BT book (orb.yaml backtest.nightly_book_csv via scripts/report_common, produced
    by onemil-orb-backtest.service) for `day`, read through trading.orb_csv.read_orb_csv. The book holds selected
    picks only: no rows for a day the nightly features CSV covers = zero BT picks ([] , not NO-DATA); no rows and
    no coverage = (None, why)."""
    from trading.orb_csv import read_orb_csv
    try:
        path = csv_path or __import__("report_common").bt_book_csv_path()
        df = read_orb_csv(path)
    except Exception as e:  # noqa: BLE001
        log.warning("BT book unreadable: %s", e)
        return None, f"book unreadable: {e}"
    rows = df[df["date"].astype(str) == day]
    if rows.empty:
        if covered(day):
            return [], ""
        return None, f"BT book ends {df['date'].astype(str).max()}, no rows for {day} and features do not cover it"
    return rows.to_dict("records"), ""


P1_STATE = Path(os.environ.get("ONEMIL_P1_PARITY_STATE") or ROOT / "logs" / "orb_p1_parity_state.json")


def p1_bt_book(day: str, book_path=None, markers_path=None) -> Tuple[Optional[List[Dict]], str]:
    """Rows of the nightly P1 book for `day`. Zero rows only counts as 0 picks when the markers file has a
    computed-day row (picks=0) for (day, P1); otherwise NO-DATA (not computed), never silently 0."""
    from trading.orb_csv import read_orb_csv
    book_path = book_path or ROOT / "analysis_results" / f"orb_bplus_book_{P1_ID}.csv"
    markers_path = markers_path or ROOT / "analysis_results" / "orb_bplus_book_markers.csv"
    try:
        rows = []
        if Path(book_path).exists() and Path(book_path).stat().st_size > 2:
            df = read_orb_csv(book_path)
            rows = df[df["date"].astype(str) == day].to_dict("records")
        if rows:
            return rows, ""
        mk = read_csv_rows(markers_path)
        if any(r.get("date") == day and r.get("pool_id") == P1_ID for r in mk):
            return [], ""
        return None, f"P1 book has no row and no computed-day marker for {day} (not computed)"
    except Exception as e:  # noqa: BLE001
        log.warning("P1 book unreadable: %s", e)
        return None, f"P1 book unreadable: {e}"


def p1_ranked_loader(day: str) -> Tuple[Optional[Dict], str]:
    """BT ranked P1 set for `day` from the P1 features CSV (analysis_results/pool_P1), P1 gate applied, whole scored set."""
    from trading.orb_pool_defs import load_addon_pools
    files = sorted(f for f in glob.glob(str(ROOT / "analysis_results" / f"pool_{P1_ID}" / "orb_features_2*.csv"))
                   if "corrmatrix" not in f)
    if not files:
        return None, "no P1 features CSV"
    pool = next((p for p in load_addon_pools(str(ROOT / "orb.yaml"))["pools"] if p.get("pool_id") == P1_ID), {})
    return bt_ranked(day, files[-1], n=999, min_move_to_range_high=pool.get("min_move_to_range_high_pct"))


def p1_parity_lines(day: str, trades: List[Dict], parsed: Dict, bt_rows: Optional[List[Dict]], bt_why: str,
                    bt_rank: Optional[Dict], rank_why: str, state_path=None) -> List[str]:
    """`P1 ranked / P1 picks/fills / P1 P&L / P1 clean sessions` lines. P1 never feeds the production promotion
    counter; its own consecutive-clean counter lives in `P1_STATE` (100-fill forward read)."""
    state_path = state_path or P1_STATE
    rows = [t for t in trades if t.get("strategy") == "orb" and (t.get("account") or "") == "paper"
            and _pool(t) == P1_ID]
    eng = set(parsed.get("p1_scored", []))
    out: List[str] = []
    clean: Optional[bool] = None
    if bt_rank is None:
        out.append(f"P1 ranked: engine {len(eng)} vs {no_data('BT ranked', rank_why)}")
    else:
        bt = set(bt_rank["ranked"])
        eng_ranked = set(engine_top_n({"scored": sorted(eng), "detail": parsed.get("p1_detail", {})}, 999) or eng)
        ok = eng_ranked == bt
        out.append(f"P1 ranked: engine {len(eng_ranked)} vs BT {len(bt)} | match {len(eng_ranked & bt)} | "
                   f"engine-only: {' '.join(sorted(eng_ranked - bt)) or '-'} | BT-only: {' '.join(sorted(bt - eng_ranked)) or '-'}")
        clean = ok if (eng or bt) else None
    traded = {r["symbol"] for r in rows if r.get("order_status") not in ("cancelled", "canceled", "rejected")}
    if bt_rows is None:
        out.append(f"P1 picks/fills: engine {len(traded)} vs {no_data('BT book', bt_why)}")
    else:
        bp = {r["symbol"] for r in bt_rows}
        filled = [r for r in rows if r.get("fill_price")]
        out.append(f"P1 picks/fills: engine {len(traded)} vs BT {len(bp)} | match {len(traded & bp)} | "
                   f"engine-only: {' '.join(sorted(traded - bp)) or '-'} | BT-only: {' '.join(sorted(bp - traded)) or '-'} | "
                   f"{len(filled)}/{len(traded)} filled")
    closed = [r for r in rows if r.get("exit_price") is not None]
    pnl = sum(float(r.get("pnl") or 0) for r in closed)
    bt_pnl = ("NO-DATA" if bt_rows is None else f"${sum(float(r.get('pnl') or 0) for r in bt_rows):+,.0f}")
    out.append(f"P1 P&L: day ${pnl:+,.0f} on {len(closed)} exits | BT book {bt_pnl}")
    try:
        st = json.loads(Path(state_path).read_text()) if Path(state_path).exists() else {}
    except (OSError, ValueError):
        st = {}
    hist = st.setdefault("p1", {})
    if clean is not None:
        hist[day] = bool(clean)
        try:
            Path(state_path).write_text(json.dumps(st, indent=1, sort_keys=True))
        except OSError as e:
            log.error("P1 parity state not saved: %s", e)
    out.append(f"P1 clean sessions {consecutive_clean(hist)} (of {len(hist)} decided; own counter, not the promotion gate)")
    return out


def orb_paper_parity_section(day: str, trades: List[Dict], log_path: Optional[Path] = None,
                             bt_loader: Callable = load_bt_rows,
                             rank_loader: Callable = bt_ranked) -> Tuple[List[str], Dict]:
    """ORB paper account (trades.db strategy orb, account paper) vs the nightly BT book for `day`."""
    log_path = log_path or SESSION_ARCHIVE / f"{day}.log"
    rows = [t for t in trades if t.get("strategy") == "orb" and (t.get("account") or "") == "paper"]
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError as e:
        log.warning("ORB log archive missing %s: %s", log_path, e)
        text = ""
    bt_rows, bt_why = bt_loader(day)
    parsed = parse_orb_log(text)
    bt_rank, rank_why = rank_loader(day) if parsed["scored"] else (None, "no decision")
    lines, metrics = orb_parity_lines(day, rows, parsed, bt_rows, bt_why, bt_rank, rank_why)
    try:   # P1 add-on pool: appended lines only, the production metrics (promotion counter) are untouched
        p1_rows, p1_why = p1_bt_book(day)
        p1_rank, p1_rank_why = p1_ranked_loader(day) if parsed.get("p1_scored") else (None, "no P1 scoring")
        lines += p1_parity_lines(day, trades, parsed, p1_rows, p1_why, p1_rank, p1_rank_why)
    except Exception as e:  # noqa: BLE001
        log.warning("P1 parity section failed: %s", e, exc_info=True)
        lines.append(no_data("P1 parity", f"{type(e).__name__}: {e}"))
    return lines, metrics


# ---------------------------------------------------------------- promotion
def review_closed() -> bool:
    """REVIEW_20261003_CLOSED: true once the FIX_A and FIX_C result docs exist."""
    return all((ROOT / p).exists() for p in REVIEW_FIX_FILES)


def load_state(path=PROMOTION_STATE) -> Dict:
    """Promotion state {'sleeve': {date: bool}, 'orb': {date: bool}, 'live_ramp': {...}}; fresh when missing."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        log.warning("promotion state unreadable %s (%s) - starting fresh", path, e)
        d = {}
    d.setdefault("sleeve", {})
    d.setdefault("orb", {})
    return d


def save_state(state: Dict, path=PROMOTION_STATE) -> None:
    """Atomic write of the promotion state."""
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(state, indent=1, sort_keys=True))
    os.replace(tmp, path)


def consecutive_clean(history: Dict[str, bool]) -> int:
    """Trailing run of True values in date order (a False resets it)."""
    n = 0
    for k in sorted(history, reverse=True):
        if not history[k]:
            break
        n += 1
    return n


def sleeve_verdict(history: Dict[str, bool], next_mon: str, closed: bool, why: str) -> str:
    """GO LIVE $20K on <Monday> at >= 2 consecutive clean rotations AND the review fixes closed, else HOLD n/2."""
    n = consecutive_clean(history)
    if not history:
        return (f"Sleeve: HOLD 0/{SLEEVE_CLEAN_NEEDED} (no scheduled rotation yet; first {next_mon}) "
                f"[{SLEEVE_CLEAN_NEEDED} clean scheduled Monday rotations, slip <= {SLEEVE_SLIP_MAX_BP:.0f} bp, "
                f"picks = BT, reconcile OK]")
    rule = f"[{SLEEVE_CLEAN_NEEDED} clean rotations, slip <= {SLEEVE_SLIP_MAX_BP:.0f} bp, picks = BT, reconcile OK]"
    if n >= SLEEVE_CLEAN_NEEDED and closed:
        return f"Sleeve: GO LIVE $20K on {next_mon} {rule}"
    reason = why or ("review fixes A/C not closed" if n >= SLEEVE_CLEAN_NEEDED else f"{SLEEVE_CLEAN_NEEDED - n} more clean rotation(s) needed")
    if n >= SLEEVE_CLEAN_NEEDED:
        reason = "review fixes A/C not closed (REVIEW_20261003_CLOSED false)"
    return f"Sleeve: HOLD {n}/{SLEEVE_CLEAN_NEEDED} clean ({reason}) {rule}"


def orb_verdict(history: Dict[str, bool], day: str, why: str) -> str:
    """GO $10K stage on <next trading day> at >= 5 consecutive clean sessions, else HOLD n/5."""
    n = consecutive_clean(history)
    rule = (f"[{ORB_CLEAN_NEEDED} sessions: picks = BT, fills within {ORB_ENTRY_TOL_BP:.0f} bp, 0 tick TIMEOUT, "
            f"0 ERROR]")
    if n >= ORB_CLEAN_NEEDED:
        d = dt.date.fromisoformat(day) + dt.timedelta(days=1)
        while d.weekday() >= 5:
            d += dt.timedelta(days=1)
        return f"ORB: GO $10K stage ($375 R) on {d} {rule}"
    return f"ORB: HOLD {n}/{ORB_CLEAN_NEEDED} ({why or 'no clean session yet'}) {rule}"


def ramp_verdict(book: str, ramp: Optional[Dict], realized: Optional[float], trading_days: Optional[int]) -> str:
    """`+$10K` only when realized P&L since the last step >= 0 and >= 20 trading days passed, else HOLD at $x.
    `ramp` = {'stage_usd', 'since'} from the state file; None means the book is not live yet."""
    rule = f"[realized since last step >= 0 and >= {RAMP_MIN_DAYS} trading days]"
    if not ramp:
        return f"Ramp {book}: HOLD (no live stage started - paper) {rule}"
    stage = ramp.get("stage_usd", 0)
    if realized is None or trading_days is None:
        return f"Ramp {book}: HOLD at ${stage:,.0f} ({no_data('ramp', 'realized P&L unavailable')}) {rule}"
    if realized >= 0 and trading_days >= RAMP_MIN_DAYS:
        return f"Ramp {book}: +${RAMP_STEP_USD:,} to ${stage + RAMP_STEP_USD:,.0f} {rule}"
    return (f"Ramp {book}: HOLD at ${stage:,.0f} (realized ${realized:+,.0f}, {trading_days} trading days) {rule}")


def _ramp_inputs(book: str, ramp: Dict, day: str) -> Tuple[Optional[float], Optional[int]]:
    """Realized P&L and trading days since the ramp step: ORB via scripts/orb_ramp_check.load_fills, the sleeve
    via the weekly equity marks. (None, None) with a WARNING on failure."""
    since = ramp.get("since")
    try:
        days = len({d for d in range((dt.date.fromisoformat(day) - dt.date.fromisoformat(since)).days + 1)
                    if (dt.date.fromisoformat(since) + dt.timedelta(days=d)).weekday() < 5})
        if book == "ORB":
            import orb_ramp_check
            return sum(float(f["pnl"] or 0) for f in orb_ramp_check.load_fills(since)), days
        marks = {r["date"]: float(r["equity"]) for r in read_csv_rows(WEEKLY)}
        eq0 = marks[max(k for k in marks if k <= since)]
        return marks[max(marks)] - eq0, days
    except Exception as e:  # noqa: BLE001
        log.warning("ramp inputs failed for %s: %s", book, e)
        return None, None


def promotion_section(day: str, sleeve_m: Optional[Dict], orb_m: Optional[Dict],
                      state_path=PROMOTION_STATE, closed: Optional[bool] = None) -> List[str]:
    """The PROMOTION lines: sleeve, ORB, ramps, HOD. Updates the state once per date (idempotent: the day's
    result overwrites its own key). A book whose section failed (metrics None) is not counted either way and the
    line says so."""
    state = load_state(state_path)
    closed = review_closed() if closed is None else closed
    if sleeve_m is not None and sleeve_m.get("rotation") and sleeve_m.get("clean") is not None:
        state["sleeve"][day] = bool(sleeve_m["clean"])
    if orb_m is not None and orb_m.get("clean") is not None:   # None = neutral (NO DECISION / NO-DATA): not counted
        state["orb"][day] = bool(orb_m["clean"])
    try:
        save_state(state, state_path)
    except OSError as e:
        log.error("promotion state not saved: %s", e)
    out = [PROMOTION_PREFIX + ":"]
    out.append("  " + (sleeve_verdict(state["sleeve"], next_monday(day), closed, (sleeve_m or {}).get("why", ""))
                       if sleeve_m is not None else no_data("Sleeve", "MOM section failed")))
    out.append("  " + (orb_verdict(state["orb"], day, (orb_m or {}).get("why", ""))
                       if orb_m is not None else no_data("ORB", "ORB section failed")))
    for book in ("ORB", "MOM"):
        ramp = (state.get("live_ramp") or {}).get(book)
        realized, tdays = _ramp_inputs(book, ramp, day) if ramp else (None, None)
        out.append("  " + ramp_verdict(book, ramp, realized, tdays))
    out.append("  " + HOD_LINE)
    return out
