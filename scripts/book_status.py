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
                r"|SQUEEZE|PDR VETO|PREPLACE PARITY")
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
            """Fills and closed P&L between two trade_dates inclusive."""
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
        L.append(f"ORB paper: {o['today']['fills']} fills {money(o['today']['pnl'])} today | WTD "
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
