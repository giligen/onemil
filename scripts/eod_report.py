#!/usr/bin/env python3
"""Daily EOD report — deterministic numbers, Haiku only for phrasing (owner 2026-09-21).

Replaces the in-session "DAILY EOD REPORT v7" cron so the daily telegram no longer costs a full
interactive-session turn. Every number is computed here from the trades DB, the checker logs and
the ramp checkers; the LLM step is a rewrite of the assembled text into a telegram and is skipped
(with a WARNING and the raw summary sent instead) when the headless call fails.

Sections: books today · Gate-1 parity · ramp stage · hygiene · tomorrow's boot · research commits.
Output: logs/eod/<date>.md (the deterministic summary) + one telegram with prefix [EOD].

Usage: python3 scripts/eod_report.py [--date YYYY-MM-DD] [--no-send] [--no-llm]
"""
import argparse
import datetime as dt
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from persistence.database import Database  # noqa: E402
import guardrail as guardrail_cli  # noqa: E402  (scripts/guardrail.py: stage_risk_usd, band_p5)
from trading import live_guardrail as gr  # noqa: E402
import eod_sections  # noqa: E402  (MOM / ORB PAPER PARITY / PROMOTION sections)

log = logging.getLogger("eod_report")
LLM_MODEL = "haiku"
LLM_TIMEOUT_S = 120
TELEGRAM_MAX = 1800
BOOKS = ("orb", "bull_flag", "macd_wave", "hod_break")


def run(cmd: List[str], timeout: int = 180, cwd: Path = ROOT, env: dict = None) -> str:
    """Run a command, return stdout+stderr text; a failure becomes a WARNING line, never a crash."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        log.warning("command failed %s: %s", cmd[:2], e)
        return f"(failed: {e})"


def _pool_id_of(row: Dict) -> str:
    """pool_id a trades row's pattern_data carries; 'production' if the
    column is missing, unparseable, or the key is absent (pre-pool rows and
    every non-ORB book)."""
    try:
        pattern = json.loads(row.get("pattern_data") or "{}")
    except (TypeError, ValueError):
        return "production"
    return pattern.get("pool_id") or "production"


def orb_pool_detail(rows: List[Dict]) -> str:
    """'production n/$ | <pool_id> n/$ [symbols] | ...' for one ORB
    (strategy, account) group (owner 2026-10-01, "be clear on its trades"):
    production stays a count+$ summary (too many fills to list by symbol),
    every add-on pool's fills are named by symbol so a pool trade is
    unmistakable from the EOD line alone."""
    by_pool = defaultdict(list)
    for r in rows:
        by_pool[_pool_id_of(r)].append(r)
    parts = []
    for pool_id in sorted(by_pool, key=lambda p: (p != "production", p)):
        prows = by_pool[pool_id]
        closed = [r for r in prows if r.get("exit_price") is not None]
        pnl = sum(float(r.get("pnl") or 0) for r in closed)
        seg = f"{pool_id} {len(prows)}/${pnl:+,.0f}"
        if pool_id != "production":
            seg += f" [{', '.join(r['symbol'] for r in prows)}]"
        parts.append(seg)
    return " | ".join(parts)


def books_section(trades: List[Dict], day: str) -> str:
    """Per-strategy realized P&L and fills for one trade_date, from trades rows.

    Split by account (owner 9/28: HOD-break and ORB each now route real orders to
    their OWN Alpaca paper account) so a paper-account book's P&L is never summed
    onto the same line as a live-account book — each strategy gets one line per
    account seen in `trades` (paper/live, or 'unknown' for rows saved before the
    `account` column existed). ORB additionally splits its line by pool (owner
    2026-10-01, "be clear on its trades") via `orb_pool_detail` — every add-on
    pool trade is named by symbol."""
    by = defaultdict(list)
    for t in trades:
        by[t.get("strategy") or "unknown"].append(t)
    if not by:
        return f"BOOKS {day}: no fills."
    lines = [f"BOOKS {day}:"]
    for strat, rows in sorted(by.items()):
        by_account = defaultdict(list)
        for r in rows:
            by_account[r.get("account") or "unknown"].append(r)
        for account, acct_rows in sorted(by_account.items()):
            filled = [r for r in acct_rows if r.get("order_status") not in ("cancelled", "canceled", "expired", "rejected")]
            if strat == "orb":
                lines.append(f"  orb ({account}): " + orb_pool_detail(filled))
                continue
            closed = [r for r in filled if r.get("exit_price") is not None]
            pnl = sum(float(r.get("pnl") or 0) for r in closed)
            open_n = len(filled) - len(closed)
            detail = ", ".join(f"{r['symbol']} {float(r.get('pnl') or 0):+.0f} {r.get('exit_reason') or '?'}"
                               for r in closed[:8])
            lines.append(f"  {strat} ({account}): {len(filled)} fills, {len(closed)} closed, ${pnl:+,.0f}"
                         + (f", {open_n} still open" if open_n else "") + (f" [{detail}]" if detail else ""))
    return "\n".join(lines)


def last_matching_line(path: Path, needles: tuple) -> str:
    """Last line of a log containing any needle, or a WARNING marker if the log is absent."""
    if not path.exists():
        log.warning("log missing: %s", path)
        return f"(missing {path.name})"
    hits = [ln for ln in path.read_text(errors="replace").splitlines() if any(n in ln for n in needles)]
    return hits[-1].strip() if hits else "(no verdict line yet)"


def parity_section(day: str) -> str:
    """Gate-1: ORB green check verdict and BF decision-parity JSON status for the day."""
    green = last_matching_line(ROOT / "logs/daily_green_check.log", ("GREEN", "RED", "YELLOW"))
    bf_json = ROOT / f"logs/bf_parity/bf_parity_{day}.json"
    if bf_json.exists():
        d = json.loads(bf_json.read_text())
        bf = f"BF parity {d.get('status')} (bt {d.get('n_bt_trades')}, live {d.get('n_live_rows')}, stale={d.get('bt_stale')})"
    else:
        bf = "BF parity: not run yet for today (22:50 UTC)"
    freeze = ROOT / "logs/ramp_freeze.json"
    fz = f"freeze: {freeze.read_text().strip()[:200]}" if freeze.exists() else "freeze: none"
    return f"GATE-1:\n  ORB green check: {green}\n  {bf}\n  {fz}"


def ramp_section() -> str:
    """VERDICT + holding-on lines from both ramp checkers (they do not send telegrams)."""
    out = ["RAMP:"]
    for book, script in (("ORB", "scripts/orb_ramp_check.py"), ("BF", "scripts/bf_ramp_check.py")):
        txt = run([sys.executable, str(ROOT / script)])
        keep = [ln.strip() for ln in txt.splitlines() if "VERDICT" in ln or "holding on" in ln or "sessions" in ln]
        out.append(f"  {book}: " + (" | ".join(keep[-2:]) if keep else txt[-200:]))
    return "\n".join(out)


JOURNAL_ERROR_PATTERN = "ERROR|Traceback"


def journal_error_count(service: str, since: str, timeout: int = 90) -> str:
    """Count ERROR/Traceback lines in a service's journal since `since` (UTC, journalctl syntax).

    The match runs inside journalctl (`-g`, server-side regex) so the verbose scanner day (hundreds
    of thousands of RelVol lines) is never piped through Python; measured 45 s for a full day on the
    2-CPU node under research load, hence the 90 s bound. A timeout or any other failure becomes a
    WARNING line, never a crash or a hang that eats the whole report (2026-09-25: the 20 s bound with
    a Python-side filter reported "journal check timed out" on the real EOD run).
    """
    try:
        r = subprocess.run(["journalctl", "-u", service, "--since", since, "-o", "cat",
                            "--no-pager", "-q", "-g", JOURNAL_ERROR_PATTERN],
                           capture_output=True, text=True, timeout=timeout)
        return str(sum(1 for ln in r.stdout.splitlines() if ln.strip()))
    except subprocess.TimeoutExpired:
        log.warning("journalctl -u %s timed out after %ss", service, timeout)
        return "(journal check timed out)"
    except Exception as e:  # noqa: BLE001
        log.warning("journalctl -u %s failed: %s", service, e)
        return "(journal check failed)"


def hygiene_section() -> str:
    """Services, journal errors today, pre-boot test result, disk."""
    svc = {s: run(["systemctl", "is-active", s], timeout=20) for s in ("onemil-trader", "onemil-macd-wave")}
    since = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d") + " 12:00:00"
    err = journal_error_count("onemil-trader", since)
    preboot = last_matching_line(ROOT / "logs/preboot_tests.log", ("passed", "failed", "error"))
    disk = shutil.disk_usage("/")
    flags = [f"{k}={v}" for k, v in svc.items()]
    bad = [k for k, v in svc.items() if v != "active"]
    return ("HYGIENE:\n  services " + ", ".join(flags) + (f"  <-- NOT ACTIVE: {bad}" if bad else "")
            + f"\n  trader journal ERROR/Traceback lines today: {err}\n  pre-boot tests: {preboot[-120:]}"
            + f"\n  disk free {disk.free / 1e9:.1f} GB")


def yaml_flag(path: Path, key_path: List[str]) -> str:
    """Read a nested boolean from a yaml file without failing the report."""
    try:
        import yaml
        d = yaml.safe_load(path.read_text())
        for k in key_path:
            d = d[k]
        return str(d)
    except Exception as e:  # noqa: BLE001
        log.warning("yaml read failed %s %s: %s", path.name, key_path, e)
        return "?"


def boot_section(day: str) -> str:
    """What boots at 12:30 UTC on the next weekday, from the live config files."""
    d = dt.date.fromisoformat(day) + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    bf = yaml_flag(ROOT / "config.yaml", ["trading", "enabled"])
    orb = yaml_flag(ROOT / "orb.yaml", ["strategy", "enabled"])
    hod = yaml_flag(ROOT / "config.yaml", ["hod_break", "enabled"])
    hod_dry = yaml_flag(ROOT / "config.yaml", ["hod_break", "dry_run"])

    def on_off(v: str) -> str:
        return "ON" if v.lower() == "true" else ("OFF" if v.lower() == "false" else f"UNKNOWN({v})")

    def acct_mode(env_prefix: str) -> str:
        """PAPER or LIVE for a strategy's dedicated Alpaca account.

        Reads ALPACA_{prefix}_PAPER exactly as config.py's alpaca_{orb,hod}_paper
        properties do (default 'true') — the same value main.py passes into
        AlpacaClient(paper=...) and that becomes that client's is_paper. Never
        infer account identity from the strategy's own enabled/dry_run flag: that
        gates ORDER SUBMISSION, not which account the orders land in (2026-09-30:
        the report said "LIVE ORDERS" for HOD while its keys point at the paper
        account PA39QSZR60WC).
        """
        is_paper = os.getenv(f"ALPACA_{env_prefix}_PAPER", "true").lower() in ("true", "1", "yes")
        return "PAPER ORDERS" if is_paper else "LIVE ORDERS"

    hod_txt = on_off(hod)
    if hod.lower() == "true":
        hod_txt += " (dry-run, zero orders)" if hod_dry.lower() == "true" else f" ({acct_mode('HOD')})"
    orb_txt = on_off(orb)
    if orb.lower() == "true":
        orb_txt += f" ({acct_mode('ORB')})"
    return f"NEXT BOOT {d} 12:30 UTC: BF {on_off(bf)}, ORB {orb_txt}, HOD {hod_txt}"


def research_section() -> str:
    """Today's commits touching research/ (the ledger the owner reads)."""
    txt = run(["git", "log", "--since=midnight", "--format=%s", "--", "research", "docs"], timeout=30)
    lines = [ln for ln in txt.splitlines() if ln.strip()][:5]
    return "RESEARCH today: " + ("; ".join(lines) if lines else "no research commits")


def guardrail_section() -> str:
    """G1 cumulative live ledger + pause flag per book (docs/live_guardrails_spec_20260925.md).

    Read-only: reuses trading/live_guardrail.py (via scripts/guardrail.py's own
    stage_risk_usd/band_p5 config wiring, so this never re-derives the rules) —
    the cron's `scripts/guardrail.py --check` is what actually pauses a book;
    this section only surfaces the number that was missing for five months.
    """
    # db_path/path passed explicitly (gr.TRADES_DB / gr.STATE_PATH read at CALL
    # time) rather than left to live_record()/is_paused()'s default args, which
    # bind at trading.live_guardrail import time and would ignore a monkeypatch.
    lines = ["GUARDRAIL:"]
    for book in gr.BOOKS:
        risk = guardrail_cli.stage_risk_usd(book)
        ack = gr.acknowledged_through_utc(book, path=gr.STATE_PATH)
        stats = gr.live_record(book, stage_risk_usd=risk, db_path=gr.TRADES_DB, after_exited_at=ack)
        band_txt = "n/a"
        if book in gr.PAUSABLE_BOOKS:
            p5 = guardrail_cli.band_p5(book, stats.trailing_40_n)
            band_txt = f"{p5:+.3f}" if p5 is not None else "NO-DATA"
        flag = " [PAUSED_BY_GUARDRAIL]" if gr.is_paused(book, path=gr.STATE_PATH) else ""
        lines.append(f"  {stats.line()} | band p5 {band_txt}{flag}")
    return "\n".join(lines)


PROMOTION_MARK = "PROMOTION:"


def safe_section(name: str, fn) -> tuple:
    """(lines, metrics) from `fn()`; any exception becomes `<NAME>: FAILED (<exc>)` + a WARNING, never a blank."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        log.warning("section %s failed: %s", name, e, exc_info=True)
        return [f"{name}: FAILED ({type(e).__name__}: {e})"], None


def new_sections(day: str, trades: List[Dict]) -> List[str]:
    """MOM, ORB PAPER PARITY and PROMOTION (scripts/eod_sections.py), each isolated by `safe_section`."""
    mom, mom_m = safe_section("MOM", lambda: eod_sections.sleeve_section(day))
    orb, orb_m = safe_section("ORB PAPER PARITY", lambda: eod_sections.orb_paper_parity_section(day, trades))
    promo, _ = safe_section("PROMOTION", lambda: (eod_sections.promotion_section(day, mom_m, orb_m), None))
    return (["MOM:"] + ["  " + ln for ln in mom] + ["ORB PAPER PARITY:"] + ["  " + ln for ln in orb] + promo)


def split_promotion(summary: str) -> tuple:
    """(summary without the PROMOTION block, the PROMOTION block) - the block is appended verbatim after the
    LLM rewrite so the verdict text is never rephrased. The block runs from `PROMOTION:` to the next
    unindented line or the end."""
    lines = summary.splitlines()
    try:
        i = next(k for k, ln in enumerate(lines) if ln.startswith(PROMOTION_MARK))
    except StopIteration:
        return summary, ""
    j = i + 1
    while j < len(lines) and lines[j].startswith("  "):
        j += 1
    return "\n".join(lines[:i] + lines[j:]), "\n".join(lines[i:j])


def fit_telegram(text: str, promo: str) -> str:
    """`text` cut so that the PROMOTION block (verbatim, never rephrased or cut) still fits in TELEGRAM_MAX."""
    room = max(TELEGRAM_MAX - len(promo) - 1, 0)
    return (text[:room] + ("\n" + promo if promo else "")).strip()


def assemble(day: str, trades: List[Dict]) -> str:
    """The deterministic summary: every section, plain text."""
    return "\n".join([books_section(trades, day), parity_section(day), ramp_section(),
                      *new_sections(day, trades),
                      guardrail_section(), hygiene_section(), boot_section(day), research_section()])


def phrase_with_llm(summary: str) -> str:
    """Rewrite the summary as the telegram with Haiku; return '' on any failure (caller falls back)."""
    prompt = (f"Rewrite the following end-of-day trading summary as a Telegram message of at most "
              f"{TELEGRAM_MAX} characters. Start with '[EOD]'. Keep EVERY number and symbol exactly. "
              "Put anything RED, NOT ACTIVE, failed or frozen in the first line. No advice, no praise, "
              "no questions, no markdown headers. Plain short lines. Copy the NEXT BOOT line and every "
              "ON/OFF/NOT ACTIVE/HOLD/ADVANCE word verbatim — never drop a value.\n\n" + summary)
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env.setdefault("HOME", "/home/ec2-user")
    env["PATH"] = "/home/ec2-user/.local/bin:" + env.get("PATH", "/usr/bin:/bin")
    try:
        r = subprocess.run(["claude", "-p", "--model", LLM_MODEL, prompt], capture_output=True, text=True,
                           timeout=LLM_TIMEOUT_S, cwd=ROOT, env=env, stdin=subprocess.DEVNULL)
        text = r.stdout.strip()
        if r.returncode != 0 or not text.startswith("[EOD]"):
            log.warning("LLM phrasing rejected rc=%s head=%r", r.returncode, text[:80])
            return ""
        return text[:TELEGRAM_MAX]
    except Exception as e:  # noqa: BLE001
        log.warning("LLM phrasing failed: %s", e)
        return ""


def send_telegram(text: str) -> None:
    """Send through the repo's own alert script (same notifier as the service)."""
    out = run([sys.executable, str(ROOT / "scripts/send_telegram_alert.py"), text], timeout=60)
    log.info("telegram: %s", out[-200:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--no-send", "--no-telegram", dest="no_send", action="store_true",
                    help="print, do not telegram")
    ap.add_argument("--no-llm", action="store_true", help="send the deterministic summary as-is")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    trades = Database().get_trades_by_date(a.date)
    summary = assemble(a.date, trades)
    out_dir = ROOT / "logs/eod"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{a.date}.md").write_text(summary + "\n")
    body, promo = split_promotion(summary)
    text = "" if a.no_llm else phrase_with_llm(body)
    if text:
        text = fit_telegram(text, promo)
    else:
        if not a.no_llm:
            log.warning("falling back to the raw summary (LLM step unavailable)")
        text = fit_telegram("[EOD] " + body, promo)
    print("[EOD] " + summary if a.no_llm else text)   # --no-llm is the inspection mode: print the whole deterministic summary
    if not a.no_send:
        send_telegram(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
