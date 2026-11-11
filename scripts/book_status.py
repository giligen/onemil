#!/usr/bin/env python3
"""One-command status of every strategy book (owner 2026-11-02: "status of the strategies?").

Read-only. Prints a <= 25-line plain-text brief, sections in a fixed order:
service, HOD paper, ORB paper, MOM, paused books, crons.  Every number is sourced from a named
file / table / journal line (see each section function).  Exit code is always 0; a section that
fails prints ``<section>: unavailable (<reason>)`` instead of raising.

    python3 scripts/book_status.py                 # today (ET)
    python3 scripts/book_status.py --date 2026-11-02 --json
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trading.ramp_stage import ORB_DAY_ONE, ORB_DAY_ONE_SUFFIX  # noqa: E402  (ONE constant for ORB day one)
TRADES_DB = ROOT / "data" / "trades.db"
RAMP_FREEZE = ROOT / "logs" / "ramp_freeze.json"
MOM_PAPER_WEEKLY = ROOT / "logs" / "momentum_sleeve_weekly.csv"
MOM_LIVE_WEEKLY = ROOT / "logs" / "momentum_sleeve_live_weekly.csv"
MOM_LIVE_STATE = ROOT / "data" / "momentum_sleeve" / "state_live.json"
MOM_PARITY = ROOT / "logs" / "mom_bt_parity.csv"
MOM_LOG_GLOB = str(ROOT / "logs" / "momentum_sleeve*.log")
CONFIG_YAML = ROOT / "config.yaml"
LOG_DIR = ROOT / "logs"

MOM_PAPER_START = 20000.0
MOM_PAUSE_FRAC = 0.75   # pause rail = 0.75 x start (-25 %)
MOM_HALVE_FRAC = 0.60   # halve rail = 0.60 x start (-40 %)
HOD_SINCE = "2026-09-29"

# Only the lines the brief counts; the full day is ~2.3M lines, so the -g pre-filter halves journalctl's runtime
# (still ~15-20 s on a full session day -- the spec's 5 s bound cannot be met from the journal).
JOURNAL_GREP = (r"Trading mode ACTIVE|\[HOD\] engine gates|\| ERROR|\| CRITICAL|ARM SKIP|LIVE cap reached"
                r"|SQUEEZE|PDR VETO|PREPLACE PARITY|\| WARNING|Q1 filter")
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# cron group -> (substrings that select its crontab lines, log files that carry its last-run status)
CRON_GROUPS = {
    "mom": (("momentum_sleeve.py",), ("momentum_sleeve.log", "momentum_sleeve_live.log")),
    "eod": (("eod_report", "eod_check"), ("eod_report.log",)),
    "refresh": (("mom_panel_refresh",), ("mom_panel_refresh.log",)),
    "parity": (("mom_bt_parity",), ("mom_bt_parity.log",)),
}


# --------------------------------------------------------------------------- external calls
def run_cmd(cmd: List[str], timeout: int = 20) -> str:
    """Run a read-only command and return stdout (raises on timeout / missing binary)."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout


def service_state() -> str:
    """``systemctl is-active onemil-trader`` (stubbed in tests)."""
    return run_cmd(["systemctl", "is-active", "onemil-trader"]).strip() or "unknown"


def journal_lines() -> List[str]:
    """Today's onemil-trader journal lines, ANSI stripped (``journalctl --since today -o cat``; stubbed in tests)."""
    out = run_cmd(["journalctl", "-u", "onemil-trader", "--since", "today", "-o", "cat", "--no-pager",
                   "-g", JOURNAL_GREP], timeout=120)
    return [ANSI.sub("", ln) for ln in out.splitlines()]


def crontab_lines() -> List[str]:
    """``crontab -l`` lines (read-only; stubbed in tests)."""
    return run_cmd(["crontab", "-l"]).splitlines()


# --------------------------------------------------------------------------- helpers
def today_et() -> str:
    """Today's date in America/New_York as YYYY-MM-DD."""
    return dt.datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def week_start(day: str) -> str:
    """Monday (YYYY-MM-DD) of the ISO week containing ``day``."""
    d = dt.date.fromisoformat(day)
    return (d - dt.timedelta(days=d.weekday())).isoformat()


def connect_ro(path: Path) -> sqlite3.Connection:
    """Open a sqlite file strictly read-only (``file:...?mode=ro``); raises if it does not exist."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)


def last_csv_row(path: Path) -> Optional[Dict[str, str]]:
    """Last data row of a CSV as a dict, ``None`` if the file has no data rows; raises if missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{Path(path).name} missing")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1] if rows else None


def money(x: Optional[float]) -> str:
    """Signed dollar string, e.g. -$46 / +$120."""
    x = float(x or 0.0)
    return f"{'-' if x < 0 else '+'}${abs(x):,.0f}"


# --------------------------------------------------------------------------- sections
def section_service(journal: List[str], state_fn: Callable[[], str] = service_state) -> dict:
    """Service state, boot time (`Trading mode ACTIVE`), today's ERROR/CRITICAL count, `[HOD] engine gates` fields."""
    boot = None
    gates = None
    n_err = 0
    for ln in journal:
        if "Trading mode ACTIVE" in ln:
            boot = ln[:19]
        if "[HOD] engine gates" in ln:
            gates = ln
        if "| ERROR" in ln or "| CRITICAL" in ln:
            n_err += 1
    fields = {}
    if gates:
        for key in ("dry_run", "risk", "per_day", "concurrent"):
            m = re.search(rf"\b{key}=(\S+)", gates)
            if m:
                fields[key] = m.group(1)
    return {"active": state_fn(), "boot": boot, "errors_today": n_err, "hod_gates": fields}


def section_hod(db: Path, day: str, journal: List[str]) -> dict:
    """HOD paper book from trades.db (strategy='hod_break') plus ARM SKIP / cap / SQUEEZE journal counts."""
    con = connect_ro(db)
    try:
        def agg(where: str, params: tuple) -> dict:
            """Fills / closed pnl / open count for a date predicate."""
            q = (
                "select count(*), coalesce(sum(case when pnl is not null then pnl end),0), "
                "sum(case when pnl is null and exit_reason is null then 1 else 0 end) "
                f"from trades where strategy='hod_break' and fill_price is not null and {where}"
            )
            n, pnl, op = con.execute(q, params).fetchone()
            return {"fills": n or 0, "pnl": float(pnl or 0.0), "open": int(op or 0)}

        today = agg("trade_date=?", (day,))
        wtd = agg("trade_date>=? and trade_date<=?", (week_start(day), day))
        since = agg("trade_date>=? and trade_date<=?", (HOD_SINCE, day))
        reasons = [r[0] or "" for r in con.execute(
            "select exit_reason from trades where strategy='hod_break' and trade_date=? and fill_price is not null "
            "and exit_reason is not null", (day,))]
        open_syms = [r[0] for r in con.execute(
            "select symbol from trades where strategy='hod_break' and trade_date=? and fill_price is not null "
            "and pnl is null and exit_reason is null", (day,))]
    finally:
        con.close()
    exits = {
        "stops": sum(1 for r in reasons if r.startswith("stop") or r.startswith("lock_stop")),
        "targets": sum(1 for r in reasons if r.startswith("target")),
        "squeeze": sum(1 for r in reasons if "squeeze" in r),
        "eod": sum(1 for r in reasons if r.startswith("eod")),
    }
    skips: Dict[str, int] = {"r_pct": 0, "min_price": 0, "other": 0}
    skip_syms = set()
    for ln in journal:
        m = re.search(r"ARM SKIP (\S+?):\s*(.*)", ln)
        if not m:
            continue
        skip_syms.add(m.group(1))
        rest = m.group(2)
        key = "r_pct" if "r_pct" in rest else "min_price" if "min_price" in rest else "other"
        skips[key] += 1
    return {
        "today": today, "wtd": wtd, "since_0929": since, "exits": exits, "open_symbols": open_syms,
        "arm_skips": skips, "skip_symbols": len(skip_syms),
        "cap_reached": sum(1 for ln in journal if "LIVE cap reached" in ln),
        "squeeze_lines": sum(1 for ln in journal if "SQUEEZE" in ln),
    }


def section_orb(db: Path, day: str, journal: List[str], ramp_path: Path = RAMP_FREEZE) -> dict:
    """ORB paper book from trades.db (strategy='orb'), PDR VETO / PREPLACE PARITY journal counts, ramp freeze state."""
    con = connect_ro(db)
    try:
        def agg(lo: str, hi: str) -> dict:
            """Fills and closed P&L between two trade_dates inclusive, never before ORB_DAY_ONE (A2: all earlier ORB rows are pre-parity)."""
            lo = max(lo, ORB_DAY_ONE.isoformat())
            n, pnl = con.execute(
                "select count(*), coalesce(sum(pnl),0) from trades where strategy='orb' and fill_price is not null "
                "and trade_date>=? and trade_date<=?", (lo, hi)).fetchone()
            return {"fills": n or 0, "pnl": float(pnl or 0.0)}
        today, wtd = agg(day, day), agg(week_start(day), day)
    finally:
        con.close()
    ramp = None
    if Path(ramp_path).exists():
        data = json.loads(Path(ramp_path).read_text())
        orb = data.get("orb", {}) if isinstance(data, dict) else {}
        ramp = {"frozen": bool(orb.get("frozen")), "reason": (orb.get("reason") or "")[:80]}
    return {
        "today": today, "wtd": wtd, "ramp": ramp,
        "pdr_veto": sum(1 for ln in journal if "PDR VETO" in ln),
        "preplace_parity": sum(1 for ln in journal if "PREPLACE PARITY" in ln),
    }


def _mom_book(weekly: Path, start: Optional[float]) -> dict:
    """Last weekly-CSV row of one MOM book plus equity vs start and the pause / halve rail distances."""
    row = last_csv_row(weekly)
    if row is None:
        raise ValueError(f"{Path(weekly).name} has no rows")
    eq = float(row["equity"])
    out = {"date": row["date"], "equity": eq,
           "names_in": row.get("names_in", "").split(), "names_out": row.get("names_out", "").split(),
           "week_r": row.get("week_r"), "weeks_since_strong": row.get("weeks_since_strong"), "start": start}
    if start:
        out["vs_start_pct"] = (eq / start - 1) * 100
        out["to_pause"] = eq - MOM_PAUSE_FRAC * start
        out["to_halve"] = eq - MOM_HALVE_FRAC * start
    return out


def section_mom(day: str, paper_csv: Path = MOM_PAPER_WEEKLY, live_csv: Path = MOM_LIVE_WEEKLY,
                live_state: Path = MOM_LIVE_STATE, parity_csv: Path = MOM_PARITY,
                log_glob: str = MOM_LOG_GLOB) -> dict:
    """MOM sleeve: paper + live last weekly rows, rails, last parity row, last notable log line today."""
    out: dict = {}
    try:
        out["paper"] = _mom_book(paper_csv, MOM_PAPER_START)
    except Exception as exc:  # noqa: BLE001 - reported in the brief, not raised
        out["paper"] = {"unavailable": str(exc)}
    try:
        start = float(json.loads(Path(live_state).read_text())["equity_start"])
        out["live"] = _mom_book(live_csv, start)
    except Exception as exc:  # noqa: BLE001
        out["live"] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    try:
        out["parity"] = last_csv_row(parity_csv)
    except Exception as exc:  # noqa: BLE001
        out["parity"] = {"unavailable": str(exc)}
    notable = []
    pat = re.compile(r"rotation due| ERROR | WARNING ")
    for path in sorted(glob.glob(log_glob)):
        try:
            with open(path, errors="replace") as fh:
                notable += [(ln[:19], Path(path).name, ln.strip()) for ln in fh if ln.startswith(day) and pat.search(ln)]
        except OSError:
            continue
    out["last_log"] = max(notable)[1:] if notable else None
    return out


def section_paused(config_path: Path = CONFIG_YAML) -> dict:
    """BF ``trading.enabled`` and R2G ``red_to_green.enabled`` parsed read-only from config.yaml."""
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    return {"bf_enabled": (cfg.get("trading") or {}).get("enabled"),
            "r2g_enabled": (cfg.get("red_to_green") or {}).get("enabled")}


def section_crons(day: str, cron_fn: Callable[[], List[str]] = crontab_lines, log_dir: Path = LOG_DIR) -> dict:
    """Crontab entries per group (mom / eod / refresh / parity) with the last-run status of their log, when one exists."""
    lines = [ln for ln in cron_fn() if ln.strip() and not ln.lstrip().startswith("#")]
    out = {}
    for group, (needles, logs) in CRON_GROUPS.items():
        sched = [" ".join(ln.split()[:5]) for ln in lines if any(n in ln for n in needles)]
        status = []
        for name in logs:
            p = Path(log_dir) / name
            if not p.exists():
                continue
            mtime = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")
            with open(p, errors="replace") as fh:
                errs = sum(1 for ln in fh if ln.startswith(day) and (" ERROR " in ln or "Traceback" in ln))
            status.append({"log": name, "mtime": mtime, "errors_today": errs})
        out[group] = {"schedules": sched, "logs": status}
    return out


# --------------------------------------------------------------------------- HTML-report data contract (spec 20261102)
SESSION_OPEN_MIN, SESSION_CLOSE_MIN = 9 * 60 + 30, 16 * 60
STALE_GRACE_MIN = 10          # a cron run younger than this is not yet "overdue"
WARN_REPEAT_MIN = 3           # a WARNING text seen this many times today becomes a flag
MOM_RAIL_WARN_FRAC = 0.10     # warn when equity is within 10 % of the pause rail
ARM_SKIP_INFO_MIN = 100       # info flag above this many distinct ARM SKIP symbols
ET = ZoneInfo("America/New_York")


def now_et() -> dt.datetime:
    """Current wall-clock time in America/New_York (stubbed in tests)."""
    return dt.datetime.now(ET)


def utc_iso_to_et_hms(ts: Optional[str]) -> Optional[str]:
    """``2026-11-02T14:46:37+00:00`` -> ``09:46:37`` (ET); ``None`` for empty / unparsable input."""
    if not ts:
        return None
    try:
        t = dt.datetime.fromisoformat(str(ts))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(ET).strftime("%H:%M:%S")


def trades_columns(con: sqlite3.Connection) -> set:
    """Column names of ``trades`` from ``pragma table_info`` (the contract never guesses a column)."""
    return {r[1] for r in con.execute("pragma table_info(trades)")}


def book_fills(db: Path, day: str, strategy: str) -> List[dict]:
    """Every filled ``trades`` row of ``strategy`` on ``day`` as the HTML fill dicts (columns absent from the schema -> None)."""
    con = connect_ro(db)
    try:
        have = trades_columns(con)
        want = ["symbol", "filled_at", "fill_price", "stop_loss_price", "take_profit_price", "shares", "total_risk", "pnl",
                "exit_reason", "exited_at", "entry_quote_spread", "exit_pricing_method", "account"]
        sel = ", ".join(c if c in have else f"NULL as {c}" for c in want)
        rows = con.execute(f"select {sel} from trades where strategy=? and trade_date=? and fill_price is not null "
                           f"order by filled_at, symbol", (strategy, day)).fetchall()
    finally:
        con.close()
    fills = []
    for sym, fat, px, stop, tgt, sh, risk, pnl, why, xat, spr, meth, acct in rows:
        fills.append({
            "symbol": sym, "entry_ts_et": utc_iso_to_et_hms(fat), "fill_price": px, "stop": stop, "target": tgt,
            "shares": sh, "risk_usd": risk, "pnl": pnl, "exit_reason": why, "exit_ts_et": utc_iso_to_et_hms(xat),
            "entry_quote_spread_bps": (spr / px * 1e4) if (spr is not None and px) else None,
            "pricing_method": meth, "account": acct,
        })
    return fills


def book_daily(db: Path, day: str, strategy: str, since: Optional[str] = None, floor: Optional[str] = None) -> dict:
    """Last 12 sessions ``{date, pnl, fills}`` of closed rows (+ ``cum_since_start`` when ``since`` is given) and the latest account.

    ``floor`` drops every session before that date (ORB passes ORB_DAY_ONE: earlier rows are pre-parity, never shown)."""
    con = connect_ro(db)
    try:
        have = trades_columns(con)
        rows = con.execute("select trade_date, coalesce(sum(pnl),0), count(*) from trades where strategy=? and "
                           "fill_price is not null and pnl is not null and trade_date<=? and trade_date>=? "
                           "group by trade_date order by trade_date", (strategy, day, floor or "")).fetchall()
        acct = None
        if "account" in have:
            r = con.execute("select account from trades where strategy=? order by trade_date desc, id desc limit 1",
                            (strategy,)).fetchone()
            acct = r[0] if r else None
    finally:
        con.close()
    out = {"daily": [{"date": d, "pnl": float(p), "fills": int(n)} for d, p, n in rows[-12:]], "account": acct}
    if since:
        cum, run = [], 0.0
        for d, p, _ in rows:
            if d >= since:
                run += float(p)
                cum.append({"date": d, "cum_pnl": run})
        out["cum_since_start"] = cum
    return out


def hod_gate(journal: List[str], hod: dict, config_path: Path = CONFIG_YAML) -> dict:
    """HOD gate card: ARM SKIP / cap / squeeze counts from ``hod`` plus caps, risk, dry_run from the gates line; squeeze flag from config."""
    fields = {}
    for ln in journal:
        if "[HOD] engine gates" in ln:
            for key in ("dry_run", "risk", "per_day", "concurrent", "kills", "flat"):
                m = re.search(rf"\b{key}=(\S+)", ln)
                if m:
                    fields[key] = m.group(1)
    kill = None
    if fields.get("kills"):
        try:
            kill = float(fields["kills"].split("/")[0])
        except ValueError:
            kill = None
    squeeze_enabled = None
    try:
        import yaml
        cfg = yaml.safe_load(Path(config_path).read_text()) or {}
        v = (cfg.get("hod_break") or {}).get("squeeze_enabled")
        squeeze_enabled = None if v is None else bool(v)
    except Exception:  # noqa: BLE001 - the flag rule then simply stays off; the gate card shows "unknown"
        squeeze_enabled = None
    risk = fields.get("risk", "").lstrip("$")
    return {
        "arm_skip_r_pct": hod.get("arm_skips", {}).get("r_pct", 0), "arm_skip_min_price": hod.get("arm_skips", {}).get("min_price", 0),
        "arm_skip_symbols": hod.get("skip_symbols", 0), "cap_hits": hod.get("cap_reached", 0),
        "squeeze_lines": hod.get("squeeze_lines", 0), "squeeze_enabled": squeeze_enabled,
        "concurrency_cap": fields.get("concurrent"), "per_day_cap": fields.get("per_day"),
        "risk_usd": float(risk) if risk.replace(".", "", 1).isdigit() else None,
        "dry_run": fields.get("dry_run"), "day_kill": kill,
    }


def orb_vetoes(journal: List[str]) -> List[dict]:
    """``{symbol, reason}`` from ``PDR VETO: SYM reason`` lines and ``Q1 filter dropped n candidate(s): A, B`` lines (deduped)."""
    out, seen = [], set()
    for ln in journal:
        m = re.search(r"PDR VETO: (\S+) (.*)", ln)
        if m:
            key, item = (m.group(1), "PDR"), {"symbol": m.group(1), "reason": "PDR VETO " + m.group(2)[:90]}
        else:
            m = re.search(r"Q1 filter dropped \d+ candidate\(s\): (.*)", ln)
            if not m:
                continue
            for sym in re.findall(r"[A-Z][A-Z0-9.]{0,5}", m.group(1)):
                if (sym, "Q1") not in seen:
                    seen.add((sym, "Q1"))
                    out.append({"symbol": sym, "reason": "Q1 filter (bottom quintile)"})
            continue
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def weekly_rows(path: Path, n: int = 8) -> List[dict]:
    """Last ``n`` weekly-CSV rows as ``{date, equity, week_pct, spy_pct}`` (pct from consecutive equity / spy_close)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for i, r in enumerate(rows):
        prev = rows[i - 1] if i else None
        eq = float(r["equity"])
        wp = (eq / float(prev["equity"]) - 1) * 100 if prev else None
        sp = None
        if prev and r.get("spy_close") and prev.get("spy_close"):
            sp = (float(r["spy_close"]) / float(prev["spy_close"]) - 1) * 100
        out.append({"date": r["date"], "equity": eq, "week_pct": wp, "spy_pct": sp})
    return out[-n:]


def parity_rows(path: Path, n: int = 4) -> List[dict]:
    """Last ``n`` rows of the MOM BT parity CSV as ``{date, names_n, resid_pp}``."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [{"date": r["date"], "names_n": int(float(r["bt_names_match"])), "resid_pp": float(r["residual_pp"])}
            for r in rows[-n:]]


def enrich_mom(mom: dict, paper_csv: Path = MOM_PAPER_WEEKLY, live_csv: Path = MOM_LIVE_WEEKLY,
               parity_csv: Path = MOM_PARITY) -> dict:
    """Add the HTML contract keys (rails in dollars, weekly[], parity[]) to each available MOM book dict in place."""
    try:
        par = parity_rows(parity_csv)
    except Exception as exc:  # noqa: BLE001
        par = []
        mom["parity_error"] = f"{type(exc).__name__}: {exc}"
    for key, csv_path in (("paper", paper_csv), ("live", live_csv)):
        book = mom.get(key)
        if not isinstance(book, dict) or "unavailable" in book:
            continue
        start = book.get("start")
        book.update({"start_equity": start, "pct_vs_start": book.get("vs_start_pct"),
                     "pause_rail": MOM_PAUSE_FRAC * start if start else None,
                     "halve_rail": MOM_HALVE_FRAC * start if start else None,
                     "last_rebalance": book.get("date"), "holdings_n": len(book.get("names_in", [])) or None,
                     "parity": par})
        try:
            book["weekly"] = weekly_rows(csv_path)
        except Exception as exc:  # noqa: BLE001
            book["weekly"] = []
            book["weekly_error"] = f"{type(exc).__name__}: {exc}"
    return mom


def _cron_field(spec: str, lo: int, hi: int) -> set:
    """Expand one cron field (``*``, ``*/n``, ``a-b``, ``a-b/n``, ``a,b``) into the set of matching ints in [lo, hi]."""
    vals = set()
    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/")
            step = int(s)
        if part in ("*", ""):
            a, b = lo, hi
        elif "-" in part:
            a, b = (int(x) for x in part.split("-"))
        else:
            a = b = int(part)
        vals.update(range(a, b + 1, step))
    return vals


def last_scheduled_run(schedule: str, now: dt.datetime, horizon_days: int = 8) -> Optional[dt.datetime]:
    """Most recent minute <= ``now`` matching the 5-field cron ``schedule`` (naive local time); ``None`` if none in the horizon."""
    try:
        mi, ho, dom, mon, dow = schedule.split()[:5]
        mins, hrs, doms, mons = _cron_field(mi, 0, 59), _cron_field(ho, 0, 23), _cron_field(dom, 1, 31), _cron_field(mon, 1, 12)
        dows = {d % 7 for d in _cron_field(dow, 0, 7)}
    except (ValueError, IndexError):
        return None
    t = now.replace(second=0, microsecond=0)
    for _ in range(horizon_days * 1440):
        if (t.minute in mins and t.hour in hrs and t.day in doms and t.month in mons
                and (t.isoweekday() % 7) in dows):
            return t
        t -= dt.timedelta(minutes=1)
    return None


def _cron_name(cmd: str) -> str:
    """Short cron name: the script stem plus ``live`` / ``submit`` markers (``momentum_sleeve live submit``)."""
    m = re.search(r"([\w.-]+)\.(?:py|sh)\b", cmd)
    name = m.group(1) if m else (cmd.split()[0] if cmd.split() else "?")
    if name == "main":  # the repo entry point: the first flag says which job it is
        flag = re.search(r"main\.py\s+(--[\w-]+)", cmd)
        name = f"main {flag.group(1)}" if flag else name
    return " ".join([name] + [w for w in ("live", "submit") if f"--{w}" in cmd])


def ops_crons(day: str, cron_fn: Callable[[], List[str]] = crontab_lines, log_dir: Path = LOG_DIR,
              now: Optional[dt.datetime] = None) -> List[dict]:
    """Every active crontab entry as ``{name, schedule, log, last_write, errors_today, last_run, overdue}``."""
    now = now or dt.datetime.now()
    out = []
    for ln in cron_fn():
        s = ln.strip()
        if not s or s.startswith("#") or re.match(r"^\w+=", s) or len(s.split()) < 6:
            continue
        parts = s.split(None, 5)
        sched, cmd = " ".join(parts[:5]), parts[5]
        m = re.search(r">>?\s*(\S+\.log)", cmd)
        logp = None
        if m:
            logp = Path(m.group(1))
            if not logp.is_absolute():
                logp = Path(log_dir).parent / logp
        last_write, errs, overdue = None, 0, False
        run = last_scheduled_run(sched, now)
        if logp is not None and logp.exists():
            mt = dt.datetime.fromtimestamp(logp.stat().st_mtime)
            last_write = mt.strftime("%m-%d %H:%M")
            if mt.strftime("%Y-%m-%d") == day:
                with open(logp, errors="replace") as fh:
                    errs = sum(1 for x in fh if x.startswith(day) and (" ERROR " in x or "Traceback" in x))
            if run is not None and now - run > dt.timedelta(minutes=STALE_GRACE_MIN) and mt < run:
                overdue = True
        elif logp is not None and run is not None and now - run > dt.timedelta(minutes=STALE_GRACE_MIN):
            overdue = True
        out.append({"name": _cron_name(cmd), "schedule": sched, "log": logp.name if logp else None,
                    "last_write": last_write, "errors_today": errs,
                    "last_run": run.strftime("%m-%d %H:%M") if run else None, "overdue": overdue})
    return out


def repeated_warnings(journal: List[str], min_count: int = WARN_REPEAT_MIN) -> List[dict]:
    """WARNING message texts (digits collapsed to ``#``) seen at least ``min_count`` times, trimmed to 80 chars."""
    counts: Dict[str, int] = {}
    for ln in journal:
        m = re.search(r"\| WARNING\s*\| [^|]*\| (.*)", ln)
        if m:
            key = re.sub(r"\d+(\.\d+)?", "#", m.group(1)).strip()[:80]
            counts[key] = counts.get(key, 0) + 1
    return [{"text": k, "count": c} for k, c in sorted(counts.items(), key=lambda kv: -kv[1]) if c >= min_count]


def section_ops(service: dict, journal: List[str], day: str, crons: List[dict]) -> dict:
    """Operations block: service state, boot, ERROR / WARNING counts, journal lines of the one pass, crons."""
    boot = service.get("boot")
    return {"service_active": service.get("active") == "active", "service_state": service.get("active"),
            "boot_utc": boot[11:16] if boot else None, "errors_today": service.get("errors_today", 0),
            "warnings_today": sum(1 for ln in journal if "| WARNING" in ln), "journal_lines_today": len(journal),
            "repeated_warnings": repeated_warnings(journal), "crons": crons}


def build_flags(data: dict, now: dt.datetime) -> List[dict]:
    """Pre-committed attention rules (spec section 1) -> ``[{severity, book, text, why}]`` sorted crit, warn, info."""
    flags: List[dict] = []

    def add(sev: str, book: str, text: str, why: str) -> None:
        """Append one flag."""
        flags.append({"severity": sev, "book": book, "text": text, "why": why})

    ops, hod, orb, mom = (data.get(k) or {} for k in ("ops", "hod", "orb", "mom"))
    minute = now.hour * 60 + now.minute
    weekday = now.weekday() < 5
    in_session = weekday and SESSION_OPEN_MIN <= minute < SESSION_CLOSE_MIN
    if ops.get("errors_today", 0) > 0:
        add("crit", "ops", f"{ops['errors_today']} ERROR line(s) in today's trader journal", "rule: errors_today > 0")
    for key in ("paper", "live"):
        b = mom.get(key) or {}
        if "equity" not in b or not b.get("pause_rail"):
            continue
        book = f"mom_{key}"
        if b["equity"] < b["pause_rail"]:
            add("crit", book, f"MOM {key} equity ${b['equity']:,.0f} is below the pause rail ${b['pause_rail']:,.0f}",
                "rule: equity < pause rail (-25 % of start)")
        elif b["equity"] < b["pause_rail"] * (1 + MOM_RAIL_WARN_FRAC):
            add("warn", book, f"MOM {key} equity ${b['equity']:,.0f} is within 10 % of the pause rail ${b['pause_rail']:,.0f}",
                "rule: equity < 1.10 x pause rail")
    kill = (hod.get("gate") or {}).get("day_kill")
    pnl = (hod.get("today") or {}).get("pnl")
    if kill is not None and pnl is not None and pnl <= kill:
        add("crit", "hod", f"HOD closed P&L today {money(pnl)} is at or below the day kill {money(kill)}", "rule: closed P&L <= day-kill")
    gate = hod.get("gate") or {}
    if gate.get("squeeze_enabled") and gate.get("squeeze_lines", 0) == 0 and weekday and minute >= 14 * 60 \
            and (hod.get("today") or {}).get("open", 0) > 0:
        add("warn", "hod", "Squeeze enabled but no SQUEEZE line today with HOD positions still open after 14:00 ET",
            "rule: squeeze_enabled and 0 squeeze lines and open position after 14:00")
    seen_logs = set()
    for c in ops.get("crons", []):
        if c.get("overdue") and (c.get("log"), c.get("schedule")) not in seen_logs:
            seen_logs.add((c.get("log"), c.get("schedule")))
            add("warn", "ops", f"Cron '{c['name']}' log {c.get('log') or '(none)'} not written since its last run {c.get('last_run')}",
                "rule: log's last write older than the most recent scheduled run")
    for w in ops.get("repeated_warnings", []):
        add("warn", "ops", f"WARNING x{w['count']}: {w['text']}", "rule: same WARNING text >= 3 times today")
    if weekday and not ops.get("service_active", True) and in_session:
        add("warn", "ops", f"onemil-trader is {ops.get('service_state')} during the session", "rule: service inactive 09:30-16:00 ET")
    if weekday and minute >= SESSION_OPEN_MIN + 30:
        for key, label in (("hod", "HOD"), ("orb", "ORB")):
            b = data.get(key) or {}
            if "today" in b and b["today"].get("fills", 1) == 0:
                add("info", key, f"{label} had 0 fills so far on a session day", "rule: 0 fills on a session day")
    if gate.get("arm_skip_symbols", 0) > ARM_SKIP_INFO_MIN:
        add("info", "hod", f"{gate['arm_skip_symbols']} distinct symbols ARM-SKIPped today", "rule: ARM SKIP symbols > 100")
    order = {"crit": 0, "warn": 1, "info": 2}
    return sorted(flags, key=lambda f: order[f["severity"]])


# --------------------------------------------------------------------------- rendering
def render(sections: Dict[str, dict], day: str) -> str:
    """Plain-text brief (<= 25 lines) from the section dicts; failed sections render as 'unavailable'."""
    L: List[str] = [f"STATUS {day} (ET)"]

    def bad(name: str) -> bool:
        """Append the unavailable line for a failed section and report whether it failed."""
        s = sections[name]
        if "unavailable" in s:
            L.append(f"{name}: unavailable ({s['unavailable']})")
            return True
        return False

    if not bad("service"):
        s = sections["service"]
        g = s["hod_gates"]
        L.append(f"service: {s['active']} | boot {s['boot'] or 'n/a'} | ERROR/CRITICAL today {s['errors_today']}")
        L.append("  HOD gates: " + (" ".join(f"{k}={v}" for k, v in g.items()) if g else "line not found in journal"))
    if not bad("hod"):
        h = sections["hod"]
        t, e, k = h["today"], h["exits"], h["arm_skips"]
        L.append(f"HOD paper: {t['fills']} fills, closed {money(t['pnl'])}, open {t['open']} "
                 f"({','.join(h['open_symbols'][:6]) or '-'}) | stop {e['stops']} tgt {e['targets']} "
                 f"sqz {e['squeeze']} eod {e['eod']}")
        L.append(f"  WTD {money(h['wtd']['pnl'])} ({h['wtd']['fills']}) | since {HOD_SINCE} {money(h['since_0929']['pnl'])} "
                 f"({h['since_0929']['fills']}) | ARM SKIP r_pct {k['r_pct']} min_price {k['min_price']} "
                 f"other {k['other']} ({h['skip_symbols']} syms) | cap {h['cap_reached']} | SQUEEZE lines {h['squeeze_lines']}")
    if not bad("orb"):
        o = sections["orb"]
        r = o["ramp"]
        ramp = "n/a" if r is None else ("FROZEN " + r["reason"] if r["frozen"] else "not frozen")
        L.append(f"ORB paper {ORB_DAY_ONE_SUFFIX}: {o['today']['fills']} fills {money(o['today']['pnl'])} today | WTD "
                 f"{money(o['wtd']['pnl'])} ({o['wtd']['fills']}) | PDR VETO {o['pdr_veto']} PARITY {o['preplace_parity']} | ramp {ramp}")
    if not bad("mom"):
        m = sections["mom"]
        for name in ("paper", "live"):
            b = m[name]
            if "unavailable" in b:
                L.append(f"MOM {name}: unavailable ({b['unavailable']})")
                continue
            extra = f" wk_r {b['week_r']} since_strong {b['weeks_since_strong']}" if b.get("week_r") else ""
            rails = (f" | {b['vs_start_pct']:+.1f}% vs ${b['start']:,.0f} | pause rail {money(b['to_pause'])} "
                     f"halve rail {money(b['to_halve'])}") if b.get("start") else ""
            L.append(f"MOM {name}: {b['date']} eq ${b['equity']:,.0f} in {len(b['names_in'])} out {len(b['names_out'])}{extra}{rails}")
        p = m["parity"]
        L.append("  parity: " + (f"unavailable ({p['unavailable']})" if p and "unavailable" in p
                                  else "n/a" if not p else f"{p['date']} names {p['bt_names_match']}/20 resid {float(p['residual_pp']):+.2f} pp"))
        if m["last_log"]:
            L.append(f"  last log: {m['last_log'][0]}: {m['last_log'][1][11:140]}")
    if not bad("paused"):
        p = sections["paused"]
        L.append(f"paused books: BF trading.enabled={p['bf_enabled']} | R2G red_to_green.enabled={p['r2g_enabled']}")
    if not bad("crons"):
        for group, c in sections["crons"].items():
            logs = "; ".join(f"{x['log']} {x['mtime']} err {x['errors_today']}" for x in c["logs"]) or "no log"
            sched = ", ".join(c["schedules"]) or "none in crontab"
            L.append(f"cron {group}: {sched} | {logs}")
    return "\n".join(L)


def build(day: str, journal: Optional[List[str]] = None, db: Path = TRADES_DB, **kw) -> dict:
    """Run every section, isolating failures; ``kw`` forwards injected paths/stubs to the section functions."""
    try:
        jl = journal if journal is not None else journal_lines()
        jerr = None
    except Exception as exc:  # noqa: BLE001
        jl, jerr = [], f"journal: {type(exc).__name__}: {exc}"

    def safe(fn: Callable[[], dict]) -> dict:
        """Run one section; a failure becomes {'unavailable': reason}."""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            return {"unavailable": f"{type(exc).__name__}: {exc}"}

    out = {
        "service": safe(lambda: section_service(jl, kw.get("state_fn", service_state))),
        "hod": safe(lambda: section_hod(db, day, jl)),
        "orb": safe(lambda: section_orb(db, day, jl, kw.get("ramp_path", RAMP_FREEZE))),
        "mom": safe(lambda: section_mom(day, **kw.get("mom", {}))),
        "paused": safe(lambda: section_paused(kw.get("config_path", CONFIG_YAML))),
        "crons": safe(lambda: section_crons(day, kw.get("cron_fn", crontab_lines), kw.get("log_dir", LOG_DIR))),
    }
    if jerr:
        out["service"].setdefault("journal_error", jerr)
    now = kw.get("now") or now_et()
    cfg = kw.get("config_path", CONFIG_YAML)

    def extend(book: str, strategy: str, since: Optional[str], floor: Optional[str] = None) -> None:
        """Attach fills / daily / cumulative series (and HOD gate, ORB vetoes) to an available book section."""
        if "unavailable" in out[book]:
            return
        try:
            out[book]["fills"] = book_fills(db, day, strategy)
            out[book].update(book_daily(db, day, strategy, since, floor))
            if book == "hod":
                out[book]["cum_since_start"] = out[book].get("cum_since_start", [])
                out[book]["gate"] = hod_gate(jl, out[book], cfg)
            else:
                out[book]["vetoes"] = orb_vetoes(jl)
        except Exception as exc:  # noqa: BLE001 - reported in the page, not raised
            out[book]["extras_error"] = f"{type(exc).__name__}: {exc}"

    extend("hod", "hod_break", HOD_SINCE)
    extend("orb", "orb", None, ORB_DAY_ONE.isoformat())
    if "unavailable" not in out["mom"]:
        enrich_mom(out["mom"], **{k: v for k, v in kw.get("mom", {}).items() if k in ("paper_csv", "live_csv")},
                   **({"parity_csv": kw["mom"]["parity_csv"]} if "parity_csv" in kw.get("mom", {}) else {}))
    out["generated_at_et"] = now.isoformat(timespec="seconds")
    out["session_date"] = day
    try:
        crons = ops_crons(day, kw.get("cron_fn", crontab_lines), kw.get("log_dir", LOG_DIR), now.replace(tzinfo=None) if kw.get("now") else None)
    except Exception as exc:  # noqa: BLE001
        crons = []
        out["service"].setdefault("crons_error", f"{type(exc).__name__}: {exc}")
    out["ops"] = section_ops(out["service"], jl, day, crons)
    out["flags"] = build_flags(out, now)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry: print the brief (or JSON). Always returns 0."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today ET)")
    ap.add_argument("--json", action="store_true", help="emit the section dicts as JSON")
    args = ap.parse_args(argv)
    day = args.date or today_et()
    sections = build(day)
    print(json.dumps({"date": day, **sections}, indent=1, default=str) if args.json else render(sections, day))
    return 0


if __name__ == "__main__":
    sys.exit(main())
