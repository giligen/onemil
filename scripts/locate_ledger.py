#!/usr/bin/env python3
"""Same-day locate ledger for the gap-fade short (PREREG 1,773; spec docs/locate_ledger_spec_20261114.md).

Modes: default (09:25 ET: find >= +10 % pre-market gappers, log borrow status + advisory locate quote),
`--score` (16:15 ET: add open/close/high/low and fade bp), `--weekly` (Fri 17:05 ET: Telegram summary).
Read-only on the broker: GET /v1/locates/quotes and GET /v2/assets only; NEVER POST /v1/locates (reserves shares).
Live keys only (paper returns 404 on locates).
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("locate_ledger")

BASE_URL = "https://api.alpaca.markets"
LOCATE_BATCH = 25
ASSET_GET_MIN_INTERVAL_S = 0.35  # per-symbol asset GET pacing (<= ~170/min; Alpaca 429'd 607 sequential GETs on 11/23)
ASSET_GET_RETRIES = 3
MIN_GAP_PCT = 10.0
MIN_PRICE = 1.0
MIN_DOLLAR_VOL = 1_000_000.0
BORROW_BP_PER_DAY = 100.0 / 252 * 100.0  # 100 % annualised borrow, one day, in bp
CACHE_DB = _REPO / "data" / "cache.db"
LEDGER = _REPO / "logs" / "locate_ledger.csv"
SCORED = _REPO / "logs" / "locate_ledger_scored.csv"
LEDGER_COLUMNS = ["date", "time_utc", "symbol", "prev_close", "pm_price", "gap_pct", "pm_vol", "borrow_status",
                  "locate_status", "fee_per_share", "fee_bp_of_price", "available_qty"]
SCORED_COLUMNS = LEDGER_COLUMNS + ["open", "close", "high", "low", "fade_bp"]
_last_asset_get = 0.0  # monotonic time of the last per-symbol asset GET
_ERR_CODES = {"easy_to_borrow": "etb", "threshold_security": "threshold", "symbol_not_found": "not_found"}


def gap_pct(prev_close: float, price: float):
    """Percent gap of price over prev_close; None when prev_close is not positive."""
    if not prev_close or prev_close <= 0:
        return None
    return (price / prev_close - 1.0) * 100.0


def select_gappers(prev_close: dict, snaps: dict, day: str) -> list:
    """Symbols with gap >= MIN_GAP_PCT, latest pre-market trade >= MIN_PRICE AND stamped TODAY (`day`) >= 04:00 ET, biggest gap first."""
    from trading.gap_short_rules import is_todays_premarket_trade
    out = []
    for sym, prev in prev_close.items():
        if not is_todays_premarket_trade((snaps.get(sym) or {}).get("latest_trade_ts"), day):
            continue
        px = (snaps.get(sym) or {}).get("latest_price") or 0
        g = gap_pct(prev, px)
        if g is None or px < MIN_PRICE or g < MIN_GAP_PCT:
            continue
        out.append({"symbol": sym, "prev_close": prev, "pm_price": px, "gap_pct": g,
                    "pm_vol": (snaps.get(sym) or {}).get("volume", 0)})
    return sorted(out, key=lambda r: -r["gap_pct"])


def load_universe(db_path=CACHE_DB, day: str = "") -> dict:
    """symbol -> prev close for the last bar date in daily_bars with close >= $1 and $vol >= $1M (read-only)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        last = con.execute("SELECT MAX(bar_date) FROM daily_bars").fetchone()[0]
        from trading.gap_short_rules import previous_session
        day = day or datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        if last != previous_session(day):
            logger.error("daily_bars stale (%s) - refusing to pick (expected %s)", last, previous_session(day))
            return {}
        rows = con.execute("SELECT symbol, close FROM daily_bars WHERE bar_date=? AND close>=? AND close*volume>=?",
                           (last, MIN_PRICE, MIN_DOLLAR_VOL)).fetchall()
    finally:
        con.close()
    logger.info("universe: %d symbols on last bar date %s", len(rows), last)
    return {s: float(c) for s, c in rows}


def make_session() -> requests.Session:
    """requests.Session with the LIVE Alpaca keys from .env (never printed)."""
    from dotenv import load_dotenv
    load_dotenv(str(_REPO / ".env"), override=True)
    key, sec = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
    if not key or not sec:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET missing in .env")
    s = requests.Session()
    s.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
    return s


def parse_locate_response(payload: dict) -> dict:
    """Parse {quotes:[...], errors:[...]} into symbol -> {locate_status, fee_per_share, available_qty}."""
    out = {}
    for q in payload.get("quotes") or []:
        out[q["symbol"]] = {"locate_status": "quoted", "fee_per_share": float(q["price"]) if q.get("price") not in (None, "") else "",
                            "available_qty": int(float(q.get("available_qty") or 0))}
    for e in payload.get("errors") or []:
        out[e["symbol"]] = {"locate_status": _ERR_CODES.get(e.get("code"), "error"),
                            "fee_per_share": "", "available_qty": ""}
        if e.get("code") not in _ERR_CODES:
            logger.warning("locate %s: unrecognised error code %s (%s)", e["symbol"], e.get("code"), e.get("message"))
    return out


def fetch_locates(session, symbols: list) -> dict:
    """Advisory locate quotes in batches of 25 (GET only). A failed batch marks its symbols 'error' and logs ERROR."""
    out = {}
    for i in range(0, len(symbols), LOCATE_BATCH):
        batch = symbols[i:i + LOCATE_BATCH]
        try:
            r = session.get(f"{BASE_URL}/v1/locates/quotes", params={"symbols": ",".join(batch)}, timeout=20)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            parsed = parse_locate_response(r.json())
        except Exception as e:  # noqa: BLE001 - recorded per symbol, ERROR logged
            logger.error("locate quotes failed for %s: %s", batch, e)
            parsed = {}
        for s in batch:
            out[s] = parsed.get(s, {"locate_status": "error", "fee_per_share": "", "available_qty": ""})
    return out


def _pace_asset_get() -> None:
    """Sleep so consecutive per-symbol asset GETs are at least ASSET_GET_MIN_INTERVAL_S apart."""
    global _last_asset_get
    wait = _last_asset_get + ASSET_GET_MIN_INTERVAL_S - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_asset_get = time.monotonic()


def _retry_delay(resp, attempt: int) -> float:
    """Seconds to wait after HTTP 429: the Retry-After header when parseable, else 2^attempt."""
    ra = (getattr(resp, "headers", None) or {}).get("Retry-After")
    try:
        return max(0.0, float(ra)) if ra is not None else float(2 ** attempt)
    except (TypeError, ValueError):
        logger.warning("unparseable Retry-After %r - using 2^%d s", ra, attempt)
        return float(2 ** attempt)


def fetch_asset_table(session) -> dict:
    """ONE GET /v2/assets?status=active&asset_class=us_equity -> {symbol: {borrow_status, shortable, easy_to_borrow}}.

    On any failure logs WARNING and returns {} (callers then fall back to the paced per-symbol GET).
    """
    try:
        r = session.get(f"{BASE_URL}/v2/assets", params={"status": "active", "asset_class": "us_equity"}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        return {a["symbol"]: {"borrow_status": a.get("borrow_status") or "unknown", "shortable": a.get("shortable"),
                              "easy_to_borrow": a.get("easy_to_borrow")} for a in r.json() if a.get("symbol")}
    except Exception as e:  # noqa: BLE001
        logger.warning("asset table fetch failed (%s) - falling back to per-symbol GETs", e)
        return {}


def fetch_asset_flags(session, symbol: str) -> dict:
    """GET /v2/assets/{symbol}, paced, retrying HTTP 429 -> {borrow_status, shortable, easy_to_borrow, reason}.

    Failure returns borrow_status 'error' with reason ('HTTP 429' or other text) and logs WARNING only;
    run_ledger aggregates failures into one ERROR.
    """
    reason = "unknown"
    for attempt in range(ASSET_GET_RETRIES + 1):
        resp = None
        try:
            _pace_asset_get()
            resp = session.get(f"{BASE_URL}/v2/assets/{symbol}", timeout=15)
            if resp.status_code == 429:
                reason = "HTTP 429"
                if attempt < ASSET_GET_RETRIES:
                    logger.warning("asset %s: HTTP 429, retry %d/%d", symbol, attempt + 1, ASSET_GET_RETRIES)
                    time.sleep(_retry_delay(resp, attempt))
                    continue
                break
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            body = resp.json()
            return {"borrow_status": body.get("borrow_status") or "unknown", "shortable": body.get("shortable"),
                    "easy_to_borrow": body.get("easy_to_borrow"), "reason": ""}
        except Exception as e:  # noqa: BLE001
            reason = str(e)
            break
    logger.warning("asset borrow_status failed for %s: %s", symbol, reason)
    return {"borrow_status": "error", "shortable": None, "easy_to_borrow": None, "reason": reason}


def resolve_borrow(session, symbol: str, table=None) -> tuple:
    """(borrow_status, failure_reason): table hit -> no GET; miss -> the paced per-symbol GET. reason '' on success."""
    if table and symbol in table:
        return table[symbol]["borrow_status"], ""
    f = fetch_asset_flags(session, symbol)
    return f["borrow_status"], f["reason"]


def fetch_borrow_status(session, symbol: str, table=None) -> str:
    """borrow_status from the asset table when present, else GET /v2/assets/{symbol}; 'error' (+WARNING) on failure."""
    return resolve_borrow(session, symbol, table)[0]


def log_borrow_failures(failures: dict, total: int) -> None:
    """ONE aggregated ERROR for all symbols whose borrow_status failed (symbol -> reason); nothing when empty."""
    if not failures:
        return
    n429 = sum(1 for v in failures.values() if v == "HTTP 429")
    logger.error("locate ledger: borrow_status failed for %d of %d gappers (HTTP 429 x %d, other x %d): first 10 symbols %s",
                 len(failures), total, n429, len(failures) - n429, list(failures)[:10])


def build_row(date: str, time_utc: str, g: dict, borrow_status: str, loc: dict) -> dict:
    """One ledger row (columns = LEDGER_COLUMNS)."""
    fee = loc.get("fee_per_share", "")
    fee_bp = fee / g["pm_price"] * 1e4 if fee != "" and g["pm_price"] else ("0" if loc["locate_status"] == "etb" else "")
    row = {"date": date, "time_utc": time_utc, "symbol": g["symbol"], "prev_close": g["prev_close"],
           "pm_price": g["pm_price"], "gap_pct": round(g["gap_pct"], 3), "pm_vol": g["pm_vol"],
           "borrow_status": borrow_status, "locate_status": loc["locate_status"], "fee_per_share": fee,
           "fee_bp_of_price": fee_bp, "available_qty": loc.get("available_qty", "")}
    return {k: row[k] for k in LEDGER_COLUMNS}


def read_csv(path) -> list:
    """Rows of a CSV as dicts ([] if the file is missing)."""
    p = Path(path)
    return list(csv.DictReader(p.open())) if p.exists() else []


def append_rows(path, rows: list, columns=None) -> int:
    """Append rows whose (date, symbol) is not already present; returns the number written."""
    columns = columns or LEDGER_COLUMNS
    path = Path(path)
    seen = {(r["date"], r["symbol"]) for r in read_csv(path)}
    new = []
    for r in rows:
        k = (r["date"], r["symbol"])
        if k not in seen:
            seen.add(k)
            new.append(r)
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if fresh:
                w.writeheader()
            w.writerows(new)
    return len(new)


def summary_line(rows: list) -> str:
    """One INFO summary: n gappers, etb/quoted/threshold counts, median fee bp of quoted."""
    cnt = lambda k: sum(1 for r in rows if r["locate_status"] == k)  # noqa: E731
    fees = [float(r["fee_bp_of_price"]) for r in rows if r["locate_status"] == "quoted" and r["fee_bp_of_price"] != ""]
    med = f"{statistics.median(fees):.1f}" if fees else "n/a"
    nerr = sum(1 for r in rows if r.get("borrow_status") == "error")
    err = f", error {nerr}" if nerr else ""
    return (f"LOCATE LEDGER: {len(rows)} gappers, {cnt('etb')} etb / {cnt('quoted')} quoted / "
            f"{cnt('threshold')} threshold, median fee {med} bp{err}")


def run_ledger(dry_run: bool, at: str | None, session=None, client=None) -> int:
    """Morning mode: universe -> pre-market gappers -> borrow status + locate quotes -> append."""
    session = session or make_session()
    prev = load_universe()
    if not prev:
        return 0
    if client is None:
        from data_sources.alpaca_client import AlpacaClient
        client = AlpacaClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=False)
    snaps = client.get_snapshots(list(prev))
    if not snaps:
        logger.error("locate ledger: snapshots empty for %d symbols", len(prev))
    gappers = select_gappers(prev, snaps, datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d"))
    now = datetime.now(timezone.utc)
    date = now.strftime("%Y-%m-%d")
    t = f"{at[:2]}:{at[2:]}:00" if at else now.strftime("%H:%M:%S")
    locs = fetch_locates(session, [g["symbol"] for g in gappers])
    table = fetch_asset_table(session) if gappers else {}
    rows, failures = [], {}
    for g in gappers:
        bs, why = resolve_borrow(session, g["symbol"], table)
        if why:
            failures[g["symbol"]] = why
        rows.append(build_row(date, t, g, bs, locs[g["symbol"]]))
    log_borrow_failures(failures, len(gappers))
    logger.info(summary_line(rows))
    if dry_run:
        for r in rows[:10]:
            print(r)
        return 0
    logger.info("appended %d new rows to %s", append_rows(LEDGER, rows), LEDGER)
    return 0


def repair_borrow(day: str, path=None, session=None, table=None) -> dict:
    """Re-fetch borrow_status for `day`'s 'error' rows and rewrite ONLY that field in place (atomic temp + os.replace).

    Returns {repaired, total, still_error}; rows that still fail stay 'error'. No other field or row changes.
    """
    path = Path(path or LEDGER)
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        cols, rows = rd.fieldnames, list(rd)
    targets = [r for r in rows if r["date"] == day and r["borrow_status"] == "error"]
    if not targets:
        logger.info("repaired 0 of 0 error rows for %s; still error: 0", day)
        return {"repaired": 0, "total": 0, "still_error": 0}
    session = session or make_session()
    table = fetch_asset_table(session) if table is None else table
    for r in targets:
        r["borrow_status"] = fetch_borrow_status(session, r["symbol"], table)
    still = sum(1 for r in targets if r["borrow_status"] == "error")
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    logger.info("repaired %d of %d error rows for %s; still error: %d", len(targets) - still, len(targets), day, still)
    return {"repaired": len(targets) - still, "total": len(targets), "still_error": still}


def score_rows(rows: list, snaps: dict) -> list:
    """Add open/close/high/low and short-side open->close fade bp (positive = the stock faded)."""
    out = []
    for r in rows:
        s = snaps.get(r["symbol"]) or {}
        o, c = s.get("open"), s.get("close")
        fade = (o - c) / o * 1e4 if o and c else ""
        out.append(dict(r, open=o or "", close=c or "", high=s.get("high", ""), low=s.get("low", ""), fade_bp=fade))
    return out


def weekly_summary(scored: list) -> dict:
    """n, ETB share, median fee bp, mean gross fade bp and net (fee + one-day 100 % borrow) over borrowable rows."""
    n = len(scored)
    fees = [float(r["fee_bp_of_price"]) for r in scored if r["locate_status"] == "quoted" and r["fee_bp_of_price"] != ""]
    bor = [r for r in scored if r["locate_status"] in ("etb", "quoted") and r.get("fade_bp") not in ("", None)]
    gross = statistics.mean(float(r["fade_bp"]) for r in bor) if bor else float("nan")
    net = statistics.mean(float(r["fade_bp"]) - float(r["fee_bp_of_price"] or 0) - BORROW_BP_PER_DAY for r in bor) if bor else float("nan")
    return {"n": n, "etb_share": sum(1 for r in scored if r["locate_status"] == "etb") / n if n else float("nan"),
            "median_fee_bp": statistics.median(fees) if fees else float("nan"),
            "mean_fade_bp_gross": gross, "mean_fade_bp_net": net, "n_borrowable": len(bor)}


def run_score(client=None) -> int:
    """EOD mode: score today's ledger rows and append to the scored CSV."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = [r for r in read_csv(LEDGER) if r["date"] == today]
    if not rows:
        logger.warning("score: no ledger rows for %s", today)
        return 0
    if client is None:
        make_session()  # loads .env
        from data_sources.alpaca_client import AlpacaClient
        client = AlpacaClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=False)
    snaps = client.get_snapshots([r["symbol"] for r in rows])
    missing = [r["symbol"] for r in rows if r["symbol"] not in snaps]
    if missing:
        logger.error("score: no snapshot for %s", missing)
    logger.info("scored rows appended: %d", append_rows(SCORED, score_rows(rows, snaps), SCORED_COLUMNS))
    return 0


def run_weekly() -> int:
    """Friday mode: summarise the last 5 trading dates of the scored CSV and send Telegram '[LOCATE]'."""
    scored = read_csv(SCORED)
    dates = sorted({r["date"] for r in scored})[-5:]
    s = weekly_summary([r for r in scored if r["date"] in dates])
    msg = (f"[LOCATE] week {dates[0] if dates else '-'}..{dates[-1] if dates else '-'}: n {s['n']}, "
           f"ETB {s['etb_share']:.0%}, median fee {s['median_fee_bp']:.1f} bp, fade gross {s['mean_fade_bp_gross']:.0f} bp, "
           f"net {s['mean_fade_bp_net']:.0f} bp (fee + 100% borrow, n {s['n_borrowable']})")
    logger.info(msg)
    rc = subprocess.run([sys.executable, str(_REPO / "scripts" / "send_telegram_alert.py"), msg]).returncode
    if rc != 0:
        logger.error("weekly Telegram send failed rc=%s", rc)
    return 0


def main() -> int:
    """CLI entry."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--at", help="HHMM label override (tests only)")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--repair-borrow", nargs="?", const="", default=None, metavar="YYYY-MM-DD",
                    help="re-fetch borrow_status of that day's 'error' ledger rows (default today, UTC)")
    ap.add_argument("--probe", help="comma symbols: print parsed locate quote (read-only demo)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from monitoring.cron_guard import attach_telegram_error_handler
    if not a.dry_run:
        attach_telegram_error_handler()
    if a.repair_borrow is not None:
        repair_borrow(a.repair_borrow or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        return 0
    if a.probe:
        syms = a.probe.split(",")
        s = make_session()
        for k, v in fetch_locates(s, syms).items():
            print(k, fetch_borrow_status(s, k), v)
        return 0
    if a.score:
        return run_score()
    if a.weekly:
        return run_weekly()
    return run_ledger(a.dry_run, a.at)


if __name__ == "__main__":
    from monitoring.cron_guard import run_guarded
    sys.exit(run_guarded(main, "locate_ledger"))
