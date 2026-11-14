#!/usr/bin/env python3
"""Gap-fade short PAPER instrument (docs/gap_short_paper_instrument_20261114.md), standalone cron script on the MOM paper account.

Modes (each idempotent, each with --dry-run = no order, no ledger write, no Telegram):
  --pick     09:20 ET  rules 1-3 -> advisory locate (live GET only) -> MOO short `gapshort-<day>-<sym>`
  --cover    10:30 ET  limit buy-to-cover at ask + 10 bp, poll 60 s, unfilled -> cancel + market
  --flatten  10:40 ET  safety: any remaining gapshort- short is market-covered and logged ERROR
  --score    16:15 ET  fill ledger rows (MOO fill vs open, cover fill vs ask, net bp), count-matched placebo pool
  --weekly   Fri 17:10 Telegram `[GAPSHORT]` vs the BT line
Rule maths lives in trading/gap_short_rules.py. Locate/universe helpers are reused from scripts/locate_ledger.py. Locates are
only ever GET (never POST). Keys: ALPACA_MOM_API_KEY/SECRET (paper); the locate GET uses the live keys via locate_ledger.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import locate_ledger as ll  # noqa: E402
from trading import gap_short_rules as R  # noqa: E402

logger = logging.getLogger("gap_short_paper")

LEDGER = _REPO / "logs" / "gap_short_ledger.csv"
GAPPERS = _REPO / "logs" / "gap_short_gappers.csv"
PAUSE_FLAG = _REPO / "logs" / "gap_short_pause.flag"
COLUMNS = ["date", "symbol", "pm_last", "prev_close", "gap_pct", "l1", "qty", "borrow_status", "locate_status",
           "locate_fee_bp", "order_id", "entry_status", "daily_open", "entry_fill", "ask_1030", "cover_limit", "cover_fill",
           "slip_bp_vs_ask", "net_bp", "pnl_usd", "note"]
GAPPER_COLUMNS = ["date", "symbol", "pm_last", "gap_pct", "l1", "picked", "open", "px_1030", "placebo_net_bp"]
ENV_KEY, ENV_SEC = "ALPACA_MOM_API_KEY", "ALPACA_MOM_API_SECRET"
OPG_CUTOFF = (9, 27, 30)   # hard clock guard: Alpaca rejects opg after 09:28:00 ET


def et_now() -> datetime:
    """Current time in New York."""
    return datetime.now(timezone.utc).astimezone(R.ET)


def make_client():
    """AlpacaClient on the MOM PAPER keys (.env loaded with override=True; keys never logged)."""
    from dotenv import load_dotenv
    load_dotenv(str(_REPO / ".env"), override=True)
    key, sec = os.getenv(ENV_KEY), os.getenv(ENV_SEC)
    if not key or not sec:
        raise RuntimeError(f"{ENV_KEY} / {ENV_SEC} missing in .env")
    from data_sources.alpaca_client import AlpacaClient
    return AlpacaClient(key, sec, paper=True)


# ---- ledger -----------------------------------------------------------------------------------------------------------
def read_ledger(path=None) -> list:
    """Ledger rows as dicts ([] when missing)."""
    return ll.read_csv(path or LEDGER)


def write_ledger(rows: list, path=None, columns=COLUMNS) -> None:
    """Atomic rewrite (tmp + rename) so a crash never truncates the ledger."""
    path = Path(path or LEDGER)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def update_ledger(day: str, symbol: str, fields: dict, path=None) -> bool:
    """Merge `fields` into the (day, symbol) row; False when the row does not exist."""
    rows = read_ledger(path)
    hit = False
    for r in rows:
        if r["date"] == day and r["symbol"] == symbol:
            r.update({k: v for k, v in fields.items()})
            hit = True
    if hit:
        write_ledger(rows, path)
    return hit


# ---- broker layer ------------------------------------------------------------------------------------------------------
class PaperBroker:
    """Thin wrapper over alpaca-py's TradingClient for the five calls this instrument needs (mockable with spec=)."""

    def __init__(self, client):
        """`client` is a data_sources.alpaca_client.AlpacaClient on the paper account."""
        self.c = client
        self.tc = client.trading_client

    def positions(self) -> dict:
        """symbol -> (signed qty, avg entry price)."""
        return {p.symbol: (float(p.qty), float(p.avg_entry_price)) for p in self.tc.get_all_positions()}

    def orders_today(self) -> list:
        """All orders (any status) as dicts with id/client_order_id/symbol/side/status/filled_avg_price/qty/filled_qty."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        out = self.tc.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500))
        return [{"id": str(o.id), "client_order_id": o.client_order_id or "", "symbol": o.symbol,
                 "side": o.side.value, "status": o.status.value, "qty": float(o.qty or 0),
                 "filled_qty": float(o.filled_qty or 0),
                 "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None} for o in out]

    def short_opg(self, symbol: str, qty: int, coid: str) -> dict:
        """Market-on-open sell short (time_in_force=opg)."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        o = self.tc.submit_order(MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL,
                                                    time_in_force=TimeInForce.OPG, client_order_id=coid))
        return {"id": str(o.id), "status": o.status.value}

    def cover_limit(self, symbol: str, qty: int, price: float, coid: str) -> dict:
        """Day limit buy-to-cover."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest
        o = self.tc.submit_order(LimitOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, limit_price=price,
                                                   time_in_force=TimeInForce.DAY, extended_hours=False, client_order_id=coid))
        return {"id": str(o.id), "status": o.status.value}

    def cover_market(self, symbol: str, qty: int, coid: str) -> dict:
        """Day market buy-to-cover."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        o = self.tc.submit_order(MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY,
                                                    time_in_force=TimeInForce.DAY, client_order_id=coid))
        return {"id": str(o.id), "status": o.status.value}

    def cancel(self, order_id: str) -> None:
        """Cancel an order by id."""
        self.tc.cancel_order_by_id(order_id)

    def order(self, order_id: str) -> dict:
        """One order's status / fill."""
        o = self.tc.get_order_by_id(order_id)
        return {"status": o.status.value, "filled_qty": float(o.filled_qty or 0),
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None}


# ---- pick ---------------------------------------------------------------------------------------------------------------
def load_universe(db_path=ll.CACHE_DB) -> dict:
    """symbol -> prior close for the last daily_bars date with PRIOR-DAY volume >= 500 K shares (cell 1,428 population; the
    locate_ledger $1 M dollar-volume rule is deliberately NOT applied). Read-only."""
    import sqlite3
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        last = con.execute("SELECT MAX(bar_date) FROM daily_bars").fetchone()[0]
        rows = con.execute("SELECT symbol, close FROM daily_bars WHERE bar_date=? AND volume>=? AND close>0",
                           (last, R.PRIOR_DAY_VOL_MIN)).fetchall()
    finally:
        con.close()
    logger.info("pick universe: %d symbols with prior-day volume >= %d on %s", len(rows), R.PRIOR_DAY_VOL_MIN, last)
    return {s: float(c) for s, c in rows if not R.is_excluded_symbol(s)}


def build_candidates(client, universe: dict, day: str) -> list:
    """Two stages: (1) snapshot gap screen (latest trade must be from TODAY >= 04:00 ET), (2) 1-min bars ONLY for the
    gap-qualified names -> L1 over 04:00-09:19 ET (deviation from the BT's 09:25: opg deadline)."""
    snaps = client.get_snapshots(list(universe))
    if not snaps:
        logger.error("pick: snapshots empty for %d symbols", len(universe))
        return []
    gappers = []
    for sym, prev in universe.items():
        snap = snaps.get(sym) or {}
        px = snap.get("latest_price")
        if not R.is_todays_premarket_trade(snap.get("latest_trade_ts"), day):
            continue
        g = R.gap_fraction(px, prev)
        if g is not None and R.price_ok(px) and R.MIN_GAP <= g <= R.MAX_GAP:
            gappers.append({"symbol": sym, "prev_close": prev, "pm_last": px, "gap": g})
    bars = client.get_premarket_1min_bars_multi([g["symbol"] for g in gappers]) if gappers else {}
    for g in gappers:
        df = bars.get(g["symbol"])
        if df is None or len(df) == 0:
            g["l1"] = None
            logger.warning("pick: no pre-market bars for %s (L1 unknown, not picked)", g["symbol"])
            continue
        g["l1"] = R.premarket_dollar_volume(
            {"t": r.timestamp.to_pydatetime(), "c": r.close, "v": r.volume} for r in df.itertuples())
    return gappers


def _past_cutoff(now: datetime) -> bool:
    """True at or after 09:27:30 ET."""
    return (now.hour, now.minute, now.second) >= OPG_CUTOFF


def run_pick(dry_run: bool, client=None, broker=None, session=None, universe=None, now=None, clock=None) -> int:
    """09:26 ET: build the pick list, log borrow + advisory locate fee, submit the MOO shorts."""
    clock = clock or (et_now if now is None else (lambda: now))   # injected `now` freezes the clock (tests)
    now = now or et_now()
    day = now.strftime("%Y-%m-%d")
    if PAUSE_FLAG.exists():
        logger.info("pick: pause flag %s present - no new picks (cleared only by --weekly after a green week)", PAUSE_FLAG)
        return 0
    if not dry_run and _past_cutoff(now):
        logger.warning("OPG deadline passed, all names not submitted (pick started %s ET)", now.strftime("%H:%M:%S"))
        return 1
    client = client or make_client()
    universe = universe if universe is not None else load_universe()
    broker = broker or PaperBroker(client)
    try:
        cands = with_retries(lambda: build_candidates(client, universe, day), "pick: data scan")
        held = set(with_retries(broker.positions, "pick: positions"))
        order_ids = with_retries(broker.orders_today, "pick: orders")
    except Exception as e:  # noqa: BLE001 - skip the day, WARNING not ERROR
        logger.warning("pick: API failure after 3 tries (%s) - day %s skipped", e, day)
        if not dry_run:
            ll.append_rows(LEDGER, [{"date": day, "symbol": "-", "entry_status": "skipped_api", "note": str(e)[:100]}], COLUMNS)
        return 0
    picks = R.select_picks(cands)
    for p in picks:
        p["qty"] = R.position_qty(p["pm_last"])
        if p["qty"] < 1:
            logger.warning("pick: %s skipped (qty floor(300/%.2f) < 1)", p["symbol"], p["pm_last"])
    picks = R.apply_gross_cap(picks)
    existing = {o["symbol"] for o in order_ids if o["client_order_id"].startswith(f"gapshort-{day}-")}   # ANY status, today
    session = session or ll.make_session()
    locs = ll.fetch_locates(session, [p["symbol"] for p in picks]) if picks else {}
    rows, log_gap = [], []
    picked = {p["symbol"] for p in picks}
    for c in cands:
        if c.get("l1") is not None:
            log_gap.append({"date": day, "symbol": c["symbol"], "pm_last": c["pm_last"], "gap_pct": round(c["gap"] * 100, 3),
                            "l1": round(c["l1"]), "picked": int(c["symbol"] in picked)})
    for p in picks:
        sym = p["symbol"]
        bs = ll.fetch_borrow_status(session, sym)
        loc = locs.get(sym, {"locate_status": "error", "fee_per_share": ""})
        fee = loc.get("fee_per_share", "")
        fee_bp = fee / p["pm_last"] * 1e4 if fee != "" else ""
        row = {"date": day, "symbol": sym, "pm_last": p["pm_last"], "prev_close": p["prev_close"],
               "gap_pct": round(p["gap"] * 100, 3), "l1": round(p["l1"]), "qty": p["qty"], "borrow_status": bs,
               "locate_status": loc["locate_status"], "locate_fee_bp": fee_bp, "entry_status": "dry_run" if dry_run else ""}
        skip = None
        if loc["locate_status"] == "threshold" or bs == "threshold_security":
            skip = "threshold-list"
        elif sym in held or sym in existing:
            skip = "existing position/order"
        if skip:
            row.update(entry_status="skipped", note=skip)
            logger.warning("pick: %s skipped (%s)", sym, skip)
        elif not dry_run and _past_cutoff(clock()):
            n_left = sum(1 for r in picks if r["symbol"] not in {x["symbol"] for x in rows}) 
            logger.warning("OPG deadline passed, %d names not submitted", n_left)
            row.update(entry_status="not_submitted", note="opg deadline")
            rows.append(row)
            break
        elif not dry_run:
            try:
                res = broker.short_opg(sym, p["qty"], R.client_order_id(day, sym))
                row.update(order_id=res["id"], entry_status=res["status"])
            except Exception as e:  # noqa: BLE001 - paper may reject HTB shorts; log, no retry
                row.update(entry_status="rejected", note=str(e)[:120])
                logger.warning("pick: order rejected for %s: %s", sym, e)
        rows.append(row)
    n_rej = sum(1 for r in rows if r.get("entry_status") == "rejected")
    n_ord = sum(1 for r in rows if r.get("entry_status") not in ("skipped", "not_submitted", "dry_run"))
    if n_ord and n_rej == n_ord:
        logger.error("pick: ALL %d submitted picks were rejected by the broker", n_rej)
    elif n_rej:
        logger.warning("pick: %d of %d submitted picks rejected by the broker (ledger entry_status=rejected)", n_rej, n_ord)
    logger.info("GAPSHORT PICK %s: %d gappers, %d L1-qualified top picks, %d ordered/skipped rows", day, len(cands), len(picks), len(rows))
    for r in rows:
        print({k: r.get(k, "") for k in ("date", "symbol", "gap_pct", "l1", "qty", "borrow_status", "locate_status", "entry_status", "note")})
    if dry_run:
        return 0
    logger.info("ledger rows appended: %d", ll.append_rows(LEDGER, rows, COLUMNS))
    ll.append_rows(GAPPERS, log_gap, GAPPER_COLUMNS)
    return 0


# ---- cover / flatten ------------------------------------------------------------------------------------------------------
ENTRY_ID_RE = re.compile(r"^gapshort-\d{4}-\d{2}-\d{2}-")
FLATTEN_AFTER = (10, 40)
COVER_STEP_S = 60


def with_retries(fn, label: str, tries: int = 3, backoff_s: float = 2.0):
    """Call fn() up to `tries` times with exponential backoff; re-raise the last error (caller decides, WARNING level)."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            logger.warning("%s failed (%s) - retry %d/%d", label, e, i + 1, tries - 1)
            time.sleep(backoff_s * (2 ** i))


def gap_positions(broker, day=None) -> list:
    """Open gapshort shorts [(symbol, qty, avg_entry)]: positions whose symbol has a gapshort-<date>- ENTRY order
    (today's only when `day` is given, any day otherwise)."""
    pos = broker.positions()
    ids = {o["symbol"] for o in broker.orders_today()
           if (o["client_order_id"].startswith(f"gapshort-{day}-") if day else ENTRY_ID_RE.match(o["client_order_id"]))}
    return [(s, abs(q), px) for s, (q, px) in pos.items() if q < 0 and s in ids]


def reconcile(broker, now: datetime, dry_run: bool, mode: str, force: bool = False) -> list:
    """Self-healing step run first by cover/flatten/score/parity: any gapshort short (any day) still open after 10:40 ET
    is market-covered at once and logged as ONE ERROR naming the symbols and why. `force` skips the clock test (flatten)."""
    if not force and (now.hour, now.minute) < FLATTEN_AFTER:
        return []
    left = gap_positions(broker)
    if not left:
        return []
    logger.error("FLATTEN (%s): gapshort short(s) %s still open at %s ET - left by an unfilled or failed cover; market covering%s", mode,
                 ", ".join(f"{s} x{q}" for s, q, _ in left), now.strftime("%H:%M"), " (dry-run)" if dry_run else "")
    if dry_run:
        return left
    day = now.strftime("%Y-%m-%d")
    orders = broker.orders_today()
    for sym, qty, _ in left:
        try:
            for o in orders:  # free the shares of any resting cover first
                if o["symbol"] == sym and o["side"] == "buy" and o["status"] in ("new", "accepted", "partially_filled"):
                    broker.cancel(o["id"])
            broker.cover_market(sym, qty, R.client_order_id(day, sym, "flat"))
        except Exception as e:  # noqa: BLE001
            logger.error("reconcile: cover failed for %s: %s", sym, e)
    return left


def _ask(client, sym):
    """Fresh ask from the paper client's latest quote (None if unusable)."""
    q = client.get_latest_quote(sym)
    ask = (q or {}).get("ask") or (q or {}).get("ask_price")
    return ask if ask and ask > 0 else None


def run_cover(dry_run: bool, client=None, broker=None, now=None, poll_s: int = COVER_STEP_S) -> int:
    """10:30 ET: limit buy-to-cover at ask + 10 bp; after poll_s cancel/replace once at the fresh ask + 10 bp; after 2*poll_s
    cancel and market. Fills are logged."""
    now = now or et_now()
    day = now.strftime("%Y-%m-%d")
    client = client or make_client()
    broker = broker or PaperBroker(client)
    reconcile(broker, now, dry_run, "cover")
    todo = gap_positions(broker, day)
    logger.info("GAPSHORT COVER %s: %d open gapshort shorts", day, len(todo))
    placed = []
    for sym, qty, entry in todo:
        ask = _ask(client, sym)
        if ask is None:
            logger.error("cover: no ask for %s - market cover", sym)
        lim = R.cover_limit_price(ask) if ask else None
        print({"symbol": sym, "qty": qty, "entry": entry, "ask": ask, "cover_limit": lim, "dry_run": dry_run})
        if dry_run:
            continue
        try:
            if lim is None or R.backstop_hit(entry, ask):
                res = broker.cover_market(sym, qty, R.client_order_id(day, sym, "covm"))
            else:
                res = broker.cover_limit(sym, qty, lim, R.client_order_id(day, sym, "cov"))
                placed.append([sym, qty, res["id"]])
            update_ledger(day, sym, {"ask_1030": ask or "", "cover_limit": lim or ""})
        except Exception as e:  # noqa: BLE001
            logger.error("cover: order failed for %s: %s", sym, e)
    for step in (1, 2):                      # step 1: cancel/replace at the fresh ask; step 2: cancel + market
        if not placed:
            break
        time.sleep(poll_s)
        nxt = []
        for sym, qty, oid in placed:
            st = broker.order(oid)
            if st["status"] == "filled":
                logger.info("cover: %s filled %s @ %s", sym, st["filled_qty"], st["filled_avg_price"])
                continue
            left = qty - int(st["filled_qty"])
            logger.warning("cover: %s unfilled (%s) after %ds - cancel and %s", sym, st["status"], step * poll_s,
                           "replace at fresh ask" if step == 1 else "market")
            try:
                broker.cancel(oid)
                fresh = _ask(client, sym) if step == 1 else None
                if step == 1 and fresh:
                    res = broker.cover_limit(sym, left, R.cover_limit_price(fresh), R.client_order_id(day, sym, "cov2"))
                    nxt.append([sym, left, res["id"]])
                else:
                    broker.cover_market(sym, left, R.client_order_id(day, sym, "covm"))
            except Exception as e:  # noqa: BLE001
                logger.error("cover: step %d failed for %s: %s", step, sym, e)
        placed = nxt
    return 0


def run_flatten(dry_run: bool, client=None, broker=None, now=None) -> int:
    """10:40 ET safety: market-cover every remaining gapshort short (reconcile, forced), ERROR logged once."""
    now = now or et_now()
    client = client or make_client()
    broker = broker or PaperBroker(client)
    left = reconcile(broker, now, dry_run, "flatten", force=True)
    logger.info("GAPSHORT FLATTEN %s: %d remaining", now.strftime("%Y-%m-%d"), len(left))
    return 0


# ---- score / weekly ---------------------------------------------------------------------------------------------------------
def score_row(row: dict, orders: list, snap: dict) -> dict:
    """Fill a ledger row's fills / slippage / net from broker orders and the day's snapshot."""
    day, sym = row["date"], row["symbol"]
    qty = float(row.get("qty") or 0)
    ent = next((o for o in orders if o["client_order_id"] == R.client_order_id(day, sym)), None)
    covs = [o for o in orders if o["client_order_id"] in (R.client_order_id(day, sym, "cov"), R.client_order_id(day, sym, "covm"), R.client_order_id(day, sym, "cov2"),
                                                         R.client_order_id(day, sym, "flat")) and o["filled_avg_price"]]
    out = {"daily_open": (snap or {}).get("open") or ""}
    if not ent or not ent["filled_avg_price"]:
        out["note"] = (row.get("note") or "") or "no entry fill"
        return out
    e = ent["filled_avg_price"]
    out["entry_fill"] = e
    if covs:
        fq = sum(o["filled_qty"] for o in covs)
        cv = sum(o["filled_avg_price"] * o["filled_qty"] for o in covs) / fq if fq else None
        out["cover_fill"] = cv
        ask = float(row["ask_1030"]) if row.get("ask_1030") not in ("", None) else None
        if ask and cv:
            out["slip_bp_vs_ask"] = round((cv / ask - 1) * 1e4, 1)
        nb = R.net_bp(e, cv, locate_bp=float(row["locate_fee_bp"]) if row.get("locate_fee_bp") not in ("", None) else R.LOCATE_BP)
        out["net_bp"] = round(nb, 1) if nb is not None else ""
        out["pnl_usd"] = round((e - cv) * qty, 2)
    else:
        out["note"] = "entry filled, no cover fill"
    return out


def run_score(dry_run: bool, client=None, broker=None, now=None) -> int:
    """16:15 ET: score today's ledger rows; set the pause flag when the realised week <= -$300."""
    now = now or et_now()
    day = now.strftime("%Y-%m-%d")
    rows = read_ledger()
    todays = [r for r in rows if r["date"] == day and r["symbol"] != "-"]
    if not todays:
        logger.warning("score: no ledger rows for %s", day)
        return 0
    client = client or make_client()
    broker = broker or PaperBroker(client)
    reconcile(broker, now, dry_run, "score")
    orders = broker.orders_today()
    snaps = client.get_snapshots([r["symbol"] for r in todays])
    for r in todays:
        upd = score_row(r, orders, snaps.get(r["symbol"]))
        print({"symbol": r["symbol"], **{k: upd.get(k, "") for k in ("entry_fill", "cover_fill", "net_bp", "pnl_usd", "note")}})
        r.update(upd)
    week = [x for x in rows if x["date"] >= week_start(day) and x.get("pnl_usd") not in ("", None)]
    wk_usd = sum(float(x["pnl_usd"]) for x in week)
    logger.info("GAPSHORT SCORE %s: %d rows, week realised $%.2f", day, len(todays), wk_usd)
    if dry_run:
        return 0
    write_ledger(rows)
    try:
        done = placebo_rows(client, day)
        if done:
            write_ledger([g for g in ll.read_csv(GAPPERS) if g["date"] != day] + done, GAPPERS, GAPPER_COLUMNS)
    except Exception as e:  # noqa: BLE001
        logger.warning("score: placebo pool failed (%s) - ledger rows are unaffected", e)
    if R.week_paused(wk_usd):
        PAUSE_FLAG.write_text(f"paused {day}: week realised ${wk_usd:.2f}\n")
        logger.error("GAPSHORT: week realised $%.2f <= -$300 - pause flag written (%s)", wk_usd, PAUSE_FLAG)
    return 0


def placebo_rows(client, day: str, limit: int = 30) -> list:
    """Non-picked L1-qualified gappers of `day`: net bp of a short at the daily open covered at the 10:29 bar close (proxy for the ask)."""
    rows = [g for g in ll.read_csv(GAPPERS) if g["date"] == day]
    snaps = client.get_snapshots([g["symbol"] for g in rows if g["picked"] == "0"][:limit])
    start = datetime.strptime(day, "%Y-%m-%d").replace(hour=10, minute=0, tzinfo=R.ET)
    end = start.replace(minute=30)
    for g in rows:
        if g["picked"] != "0" or g["symbol"] not in snaps:
            continue
        op = snaps[g["symbol"]].get("open")
        df = client.get_historical_1min_bars(g["symbol"], start, end)
        if df is None or len(df) == 0 or not op:
            continue
        px = float(df.iloc[-1]["close"])
        nb = R.net_bp(op, px)
        g.update(open=op, px_1030=px, placebo_net_bp=round(nb, 1) if nb is not None else "")
    return rows


def week_start(day: str) -> str:
    """ISO date of the Monday of `day`'s week."""
    d = datetime.strptime(day, "%Y-%m-%d")
    return (d - timedelta(days=d.weekday())).strftime("%Y-%m-%d")


def weekly_summary(rows: list) -> dict:
    """n, names/day, mean net bp, hit rate, worst net bp, total $ for scored rows (net_bp present)."""
    sc = [r for r in rows if r.get("net_bp") not in ("", None)]
    nets = [float(r["net_bp"]) for r in sc]
    days = {r["date"] for r in sc}
    return {"n": len(sc), "days": len(days), "names_per_day": len(sc) / len(days) if days else float("nan"),
            "mean_net_bp": statistics.mean(nets) if nets else float("nan"),
            "hit": sum(1 for x in nets if x > 0) / len(nets) if nets else float("nan"),
            "worst_bp": min(nets) if nets else float("nan"),
            "pnl_usd": sum(float(r["pnl_usd"]) for r in sc if r.get("pnl_usd") not in ("", None))}


def weekly_r_series(rows: list) -> list:
    """Realised $ per ISO week (oldest first) divided by the $150 unit = weekly R."""
    by = {}
    for r in rows:
        if r.get("pnl_usd") not in ("", None):
            by[week_start(r["date"])] = by.get(week_start(r["date"]), 0.0) + float(r["pnl_usd"])
    return [by[k] / R.UNIT_RISK_USD for k in sorted(by)]


def update_pause_flag(rows: list, week_usd: float, last_date: str) -> None:
    """--weekly: a GREEN week (> $0 realised) clears the pause flag; two consecutive weeks below the BT weekly P10 sets it."""
    if PAUSE_FLAG.exists() and week_usd > 0:
        PAUSE_FLAG.unlink()
        logger.info("GAPSHORT: green week ($%.2f) - pause flag cleared", week_usd)
    elif R.two_red_weeks_below_p10(weekly_r_series(rows)):
        PAUSE_FLAG.write_text(f"paused {last_date}: two consecutive weeks below the BT weekly P10 ({R.BT_WEEKLY_P10_R} R)\n")
        logger.warning("GAPSHORT: two red weeks below the BT P10 - pause flag written (%s)", PAUSE_FLAG)


def run_weekly(dry_run: bool, now=None) -> int:
    """Friday 17:10 ET: Telegram `[GAPSHORT]` week vs the BT line (+145 bp, 5 names/day) and the placebo pool."""
    rows = read_ledger()
    dates = sorted({r["date"] for r in rows})[-5:]
    s = weekly_summary([r for r in rows if r["date"] in dates])
    pool = [float(g["placebo_net_bp"]) for g in ll.read_csv(GAPPERS) if g["date"] in dates and g.get("placebo_net_bp") not in ("", None)]
    plc = f"{statistics.mean(pool):+.0f} bp (n {len(pool)})" if pool else "n/a"
    msg = (f"[GAPSHORT] week {dates[0] if dates else '-'}..{dates[-1] if dates else '-'}: n {s['n']} ({s['names_per_day']:.1f}/day vs BT 5), "
           f"mean net {s['mean_net_bp']:+.0f} bp vs BT +{R.BT_NET_BP:.0f}, hit {s['hit']:.0%}, worst {s['worst_bp']:+.0f} bp, "
           f"${s['pnl_usd']:+.0f}; non-picked-gapper placebo {plc}")
    logger.info(msg)
    if dry_run:
        print(msg)
        return 0
    update_pause_flag(rows, s["pnl_usd"], dates[-1] if dates else "")
    rc = subprocess.run([sys.executable, str(_REPO / "scripts" / "send_telegram_alert.py"), msg]).returncode
    if rc != 0:
        logger.error("weekly Telegram send failed rc=%s", rc)
    return 0


def main(argv=None) -> int:
    """CLI entry."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("pick", "cover", "flatten", "score", "weekly"):
        g.add_argument(f"--{m}", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from monitoring.cron_guard import attach_telegram_error_handler
    if not a.dry_run:
        attach_telegram_error_handler()
    if a.pick:
        return run_pick(a.dry_run)
    if a.cover:
        return run_cover(a.dry_run)
    if a.flatten:
        return run_flatten(a.dry_run)
    if a.score:
        return run_score(a.dry_run)
    return run_weekly(a.dry_run)


if __name__ == "__main__":
    from monitoring.cron_guard import run_guarded
    sys.exit(run_guarded(main, "gap_short_paper"))
