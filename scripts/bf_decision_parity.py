#!/usr/bin/env python3
"""Bull Flag live-vs-backtest decision-parity harness (Stage 1).

Owner-mandated 2026-08-14 ("this is not a casino, we follow the BT"): an
independent audit found BF live and its backtest are not verifiably aligned
(only 3-of-7 symbol overlap in one August week; shared trade BEZ 8/6 flipped
sign — BT -$469 stop vs live +$221 trail_stop). This tool makes that
divergence VISIBLE nightly, mirroring ORB's decision-parity pattern in
scripts/report_common.py.

For a trading day D it:
  1. Runs the Stage-2 filtered backtest for D READ-ONLY off the production
     raw cache (data/bull_flag_cache_e50_x30.csv) via
     `batch_backtest.py --skip-missing` — `--skip-missing` is the safety
     switch: no auto-build, no cache append, no API bar fetches. Sizing args
     mirror scripts/nightly_bt_update.sh (--capital 5000 --risk 60
     --max-shares 15000) so BT pnl is directly comparable to live.
  2. Loads ALL live bull_flag rows (any order_status — placements and
     cancels included) for D from data/trades.db.
  3. Classifies every symbol: BOTH / BT_ONLY / LIVE_ONLY, with field-level
     comparison (entry bps, exit_reason, pnl sign, pnl delta) for BOTH and
     Stage-1-universe attribution for LIVE_ONLY.
  4. Emits a human summary to stdout, persists machine-readable JSON to
     logs/bf_parity/bf_parity_<D>.json (evidence retention — journald only
     holds ~4 days), and sends a compact '[BF PARITY]' Telegram.

STALENESS IS A FIRST-CLASS STATUS: if D is beyond the last cached date the
BT side is UNKNOWABLE and the day reports 'BT_STALE (cache ends X)' — it is
NEVER counted as clean parity (an 8/14 audit found stale-BT days silently
scoring green elsewhere; that failure mode is forbidden here). Known
limitation: a genuinely zero-mover day D right after the last cached date is
indistinguishable from "nightly cron didn't run" — we deliberately err on
the side of BT_STALE (conservative, never silently green).

Exit code is ALWAYS 0 (report tool, never blocks crons). Internal failures
log ERROR and still send a '[BF PARITY] FAILED: reason' Telegram — a dead
instrument must be visible.

Usage:
    python3 scripts/bf_decision_parity.py [--date YYYY-MM-DD]
                                          [--no-telegram] [--json-out PATH]
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import report_common as rc  # noqa: E402  (prev_trading_day_utc, send_telegram)

ROOT = Path(__file__).resolve().parent.parent
PARITY_DIR = ROOT / 'logs' / 'bf_parity'

# Sizing args MUST mirror scripts/nightly_bt_update.sh (which mirrors live
# config.yaml) so cached/BT pnl is directly comparable to live trades.
BT_SIZING_ARGS = ['--capital', '5000', '--risk', '60', '--max-shares', '15000']
BT_SUBPROCESS_TIMEOUT_S = 600

# Classification labels (LIVE_ONLY)
LABEL_OFF_UNIVERSE = 'off_bt_universe'
LABEL_OFF_UNIVERSE_LEV = 'off_bt_universe (leveraged/non-common by name)'
LABEL_IN_UNIVERSE = 'in_bt_universe (investigate — Stage-1 saw it, live traded it, Stage-2 did not)'
LABEL_UNIVERSE_UNKNOWN = 'universe_unknown (BT stale — day not in Stage-1 cache)'
LABEL_BT_ONLY = 'live_missed (investigate)'

logger = logging.getLogger('bf_parity')


# ---------------------------------------------------------------------------
# Production cache access (READ-ONLY — never write this file)
# ---------------------------------------------------------------------------

def production_cache_path() -> Path:
    """Path of the production Stage-1 raw-trade cache.

    Mirrors batch_backtest._get_bull_flag_cache_path's slippage-derived name
    WITHOUT importing batch_backtest (heavy) and WITHOUT honoring
    BT_CACHE_PATH_OVERRIDE — a parity harness must always judge against the
    PRODUCTION cache; redirecting it to an experiment cache would silently
    invalidate the whole instrument, so the override is ignored with a
    WARNING.
    """
    if os.environ.get('BT_CACHE_PATH_OVERRIDE'):
        logger.warning(
            "BT_CACHE_PATH_OVERRIDE is set but IGNORED — bf_decision_parity "
            "always reads the production cache (parity vs production only)")
    try:
        from config import Config
        cfg = Config._load_yaml_only().get('trading', {})
        entry_slip = float(cfg.get('entry_slippage_pct', 0.005))
        exit_slip = float(cfg.get('exit_slippage_pct', 0.003))
    except Exception as e:
        logger.warning(
            f"config.yaml unreadable for slippage-derived cache name ({e}) — "
            f"falling back to default slippage 0.5%/0.3% (e50_x30)")
        entry_slip, exit_slip = 0.005, 0.003
    return ROOT / 'data' / (
        f"bull_flag_cache_e{int(entry_slip * 10000)}"
        f"_x{int(exit_slip * 10000)}.csv")


def scan_cache(cache_path: Path, day: str) -> Tuple[Optional[str], Set[str]]:
    """Single read-only pass over the Stage-1 cache.

    Returns (max_cached_date, symbols_with_a_stage1_row_on_day). The max
    date is the freshness key: day > max_cached_date means the BT side of
    parity is unknowable (BT_STALE). The day-symbol set powers the
    LIVE_ONLY 'off_bt_universe' attribution (e.g. leveraged wrapper LUNL
    8/13 never appears in the raw cache at all).
    """
    max_date: Optional[str] = None
    day_symbols: Set[str] = set()
    with open(cache_path, newline='') as f:
        for row in csv.DictReader(f):
            d = row.get('date') or ''
            if max_date is None or d > max_date:
                max_date = d
            if d == day:
                day_symbols.add(row['symbol'])
    logger.info(
        f"Stage-1 cache scan: {cache_path.name} max_date={max_date}, "
        f"{len(day_symbols)} symbol(s) with raw rows on {day}: "
        f"{sorted(day_symbols)}")
    return max_date, day_symbols


# ---------------------------------------------------------------------------
# BT side — Stage-2 filtered backtest for one day (read-only subprocess)
# ---------------------------------------------------------------------------

def run_stage2_backtest(day: str, output_csv: Path) -> List[Dict]:
    """Run the Stage-2 filtered backtest for one day; return its trade rows.

    Invokes `batch_backtest.py --start D --end D --skip-missing` as a
    subprocess (NOT an import: batch_backtest mutates global Config state
    for sizing overrides — a subprocess keeps that contained).

    --skip-missing is load-bearing: it disables the auto-build path that
    would otherwise fetch bars from Alpaca AND APPEND to the production
    cache for uncovered dates. Callers must gate on staleness FIRST
    (scan_cache) so this only ever runs for days the cache covers.
    """
    cmd = [sys.executable, str(ROOT / 'batch_backtest.py'),
           '--start', day, '--end', day,
           '--skip-missing',
           '--output', str(output_csv)] + BT_SIZING_ARGS
    logger.info(f"Running Stage-2 BT: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                         timeout=BT_SUBPROCESS_TIMEOUT_S)
    elapsed = time.time() - t0
    logger.debug(f"Stage-2 BT stdout:\n{res.stdout}")
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or '')[-2000:]
        raise RuntimeError(
            f"Stage-2 backtest exited {res.returncode} for {day} "
            f"(elapsed {elapsed:.1f}s). Output tail:\n{tail}")
    if not output_csv.exists():
        raise RuntimeError(
            f"Stage-2 backtest produced no output CSV at {output_csv}")
    trades: List[Dict] = []
    with open(output_csv, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('date') != day:
                logger.warning(
                    f"Stage-2 output row for unexpected date "
                    f"{row.get('date')} (wanted {day}) — skipping")
                continue
            trades.append({
                'symbol': row['symbol'],
                'entry_time_et': row.get('entry_time_et'),
                'entry_price': _to_float(row.get('entry_price')),
                'exit_time_et': row.get('exit_time_et'),
                'exit_price': _to_float(row.get('exit_price')),
                'exit_reason': row.get('exit_reason') or None,
                'pnl': _to_float(row.get('pnl')),
                'shares': int(float(row.get('shares') or 0)),
            })
    logger.info(
        f"Stage-2 BT for {day}: {len(trades)} filtered trade(s) in "
        f"{elapsed:.1f}s: {sorted(t['symbol'] for t in trades)}")
    return trades


def _to_float(v) -> Optional[float]:
    """Lenient float parse for CSV fields ('' -> None)."""
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Live side — trades.db (all order_status values, placements included)
# ---------------------------------------------------------------------------

def load_live_bf_rows(day: str, db_path: Optional[Path] = None) -> List[Dict]:
    """All bull_flag trade rows for a date — every order_status.

    Same source as report_common.load_live_rows but with an injectable
    db_path so tests run against a tmp fixture DB.
    """
    path = db_path or (ROOT / 'data' / 'trades.db')
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=15)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT symbol, order_status, entry_price, fill_price, exit_price, "
        "       exit_reason, pnl, shares, filled_qty "
        "FROM trades WHERE trade_date = ? AND strategy = 'bull_flag'", (day,))]
    conn.close()
    logger.info(
        f"Live side for {day}: {len(rows)} bull_flag row(s): "
        f"{sorted(r['symbol'] for r in rows)}")
    return rows


def default_is_common_lookup(symbol: str) -> Optional[bool]:
    """Is `symbol` a common stock per the production universe classifier?

    Reads company_name from cache.db::universe (read-only) and applies
    AlpacaClient._is_common_stock — the exact rule that keeps leveraged
    wrappers (e.g. LUNL, 'Defiance Daily Target 2X Long LUNR ETF') out of
    the BF asset universe. Returns None when the answer is unknowable
    (symbol missing from universe table, DB error) — callers must treat
    None as 'unknown', never as 'common'.
    """
    try:
        from data_sources.alpaca_client import AlpacaClient
        conn = sqlite3.connect(
            f"file:{ROOT / 'data' / 'cache.db'}?mode=ro", uri=True, timeout=15)
        row = conn.execute(
            "SELECT company_name FROM universe WHERE symbol = ?",
            (symbol,)).fetchone()
        conn.close()
        if row is None:
            logger.warning(
                f"{symbol}: not in universe table — common-stock status "
                f"unknown")
            return None
        return AlpacaClient._is_common_stock(symbol, row[0] or '')
    except Exception as e:
        logger.warning(
            f"{symbol}: common-stock lookup failed ({e}) — status unknown")
        return None


# ---------------------------------------------------------------------------
# Pure classification / comparison (unit-tested, no I/O)
# ---------------------------------------------------------------------------

def pnl_sign(pnl: Optional[float]) -> Optional[int]:
    """-1 / 0 / +1 sign of a pnl value; None when pnl is unknown."""
    if pnl is None:
        return None
    return (pnl > 0) - (pnl < 0)


def entry_delta_bps(bt_entry: Optional[float],
                    live_entry: Optional[float]) -> Optional[float]:
    """(live - bt) / bt in basis points; None when either side is missing."""
    if not bt_entry or live_entry is None:
        return None
    return (live_entry - bt_entry) / bt_entry * 1e4


def _pick_live_row(rows: List[Dict]) -> Dict:
    """One representative live row per symbol: prefer the filled one."""
    filled = [r for r in rows if r.get('fill_price') is not None]
    return (filled or rows)[0]


def compare_both(bt: Dict, live: Dict) -> Dict:
    """Field-level comparison for a symbol that appears on BOTH sides.

    Divergence rules (Stage 1):
      - live order never filled while BT simulated a full trade
      - exit_reason mismatch (both sides closed)
      - pnl SIGN mismatch (both sides have pnl) — the BEZ 8/6 class
    Entry-price delta and pnl-delta dollars are reported but informational.
    """
    live_filled = live.get('fill_price') is not None
    live_entry = live.get('fill_price') or live.get('entry_price')
    live_pnl = _to_float(live.get('pnl'))
    bt_pnl = _to_float(bt.get('pnl'))
    reasons: List[str] = []

    if not live_filled:
        bt_side = (f" while BT simulated ${bt_pnl:+.2f} "
                   f"({bt.get('exit_reason')})" if bt_pnl is not None else '')
        reasons.append(
            f"live order never filled "
            f"(status={live.get('order_status')}){bt_side}")

    exit_reason_match: Optional[bool] = None
    if bt.get('exit_reason') and live.get('exit_reason'):
        exit_reason_match = bt['exit_reason'] == live['exit_reason']
        if not exit_reason_match:
            reasons.append(
                f"exit_reason BT={bt['exit_reason']} vs "
                f"live={live['exit_reason']}")

    sign_match: Optional[bool] = None
    if bt_pnl is not None and live_pnl is not None:
        sign_match = pnl_sign(bt_pnl) == pnl_sign(live_pnl)
        if not sign_match:
            reasons.append(
                f"pnl SIGN FLIP: BT ${bt_pnl:+.2f} vs live ${live_pnl:+.2f}")

    delta = (live_pnl - bt_pnl
             if live_pnl is not None and bt_pnl is not None else None)
    return {
        'symbol': bt['symbol'],
        'classification': 'BOTH',
        'bt_entry': bt.get('entry_price'),
        'live_entry': live_entry,
        'entry_delta_bps': entry_delta_bps(bt.get('entry_price'), live_entry),
        'bt_exit_reason': bt.get('exit_reason'),
        'live_exit_reason': live.get('exit_reason'),
        'exit_reason_match': exit_reason_match,
        'bt_pnl': bt_pnl,
        'live_pnl': live_pnl,
        'pnl_sign_match': sign_match,
        'pnl_delta_usd': delta,
        'live_status': live.get('order_status'),
        'live_filled': live_filled,
        'divergent': bool(reasons),
        'divergence_reasons': reasons,
    }


def classify_day(bt_trades: List[Dict],
                 live_rows: List[Dict],
                 stage1_day_symbols: Optional[Set[str]],
                 bt_stale: bool,
                 is_common_lookup: Optional[Callable[[str], Optional[bool]]]
                 = None) -> Dict:
    """Classify every symbol of day D into BOTH / BT_ONLY / LIVE_ONLY.

    Pure function — all I/O is done by callers and injected here.

    stage1_day_symbols: raw Stage-1 cache symbols for D (universe
        attribution for LIVE_ONLY). None when the cache is stale for D —
        attribution then falls back to `is_common_lookup` (the production
        common-stock classifier), which still nails the known leveraged-
        wrapper class (LUNL 8/13) on stale days.
    bt_stale: when True, bt_trades is ignored (unknowable) — every live
        symbol is reported LIVE_ONLY with a stale-aware label and NO
        divergence is charged (the day's status is BT_STALE, never AGREE).
    """
    both: List[Dict] = []
    bt_only: List[Dict] = []
    live_only: List[Dict] = []
    divergences: List[str] = []

    live_by_sym: Dict[str, List[Dict]] = {}
    for r in live_rows:
        live_by_sym.setdefault(r['symbol'], []).append(r)

    bt_by_sym: Dict[str, Dict] = {}
    if not bt_stale:
        for t in bt_trades:
            if t['symbol'] in bt_by_sym:
                logger.warning(
                    f"{t['symbol']}: multiple BT trades on one day — "
                    f"comparing the first only (Stage-1 harness limitation)")
                continue
            bt_by_sym[t['symbol']] = t

    for sym, bt in sorted(bt_by_sym.items()):
        if sym in live_by_sym:
            rows = live_by_sym[sym]
            if len(rows) > 1:
                logger.warning(
                    f"{sym}: {len(rows)} live rows on one day — comparing "
                    f"the filled one")
            cmp = compare_both(bt, _pick_live_row(rows))
            both.append(cmp)
            for reason in cmp['divergence_reasons']:
                divergences.append(f"{sym}: {reason}")
        else:
            bt_only.append({
                'symbol': sym,
                'classification': 'BT_ONLY',
                'label': LABEL_BT_ONLY,
                'bt_pnl': bt.get('pnl'),
                'bt_exit_reason': bt.get('exit_reason'),
            })
            bt_pnl_txt = (f" (BT ${bt['pnl']:+.2f} {bt.get('exit_reason')})"
                          if bt.get('pnl') is not None else '')
            divergences.append(
                f"{sym}: BT_ONLY — {LABEL_BT_ONLY}{bt_pnl_txt}")

    for sym in sorted(set(live_by_sym) - set(bt_by_sym)):
        row = _pick_live_row(live_by_sym[sym])
        if stage1_day_symbols is not None:
            if sym in stage1_day_symbols:
                label = LABEL_IN_UNIVERSE
            else:
                label = LABEL_OFF_UNIVERSE
        else:
            is_common = (is_common_lookup(sym) if is_common_lookup else None)
            if is_common is False:
                label = LABEL_OFF_UNIVERSE_LEV
            else:
                label = LABEL_UNIVERSE_UNKNOWN
        rec = {
            'symbol': sym,
            'classification': 'LIVE_ONLY',
            'label': label,
            'live_status': row.get('order_status'),
            'live_pnl': _to_float(row.get('pnl')),
            'live_exit_reason': row.get('exit_reason'),
        }
        live_only.append(rec)
        if not bt_stale:
            pnl_txt = (f" ${rec['live_pnl']:+.2f}"
                       if rec['live_pnl'] is not None else '')
            divergences.append(f"{sym}: LIVE_ONLY — {label}{pnl_txt}")

    return {
        'both': both,
        'bt_only': bt_only,
        'live_only': live_only,
        'divergences': divergences,
        'n_divergent': len(divergences),
    }


def overall_status(bt_stale: bool, cache_max_date: Optional[str],
                   n_divergent: int) -> str:
    """AGREE / DIVERGE(n) / BT_STALE(cache ends X) for the day.

    BT_STALE always wins — a stale-BT day must NEVER read as clean parity.
    """
    if bt_stale:
        return f"BT_STALE (cache ends {cache_max_date or 'NEVER'})"
    if n_divergent:
        return f"DIVERGE({n_divergent})"
    return "AGREE"


# ---------------------------------------------------------------------------
# Report assembly / rendering
# ---------------------------------------------------------------------------

def build_report(day: str,
                 db_path: Optional[Path] = None,
                 bt_output_csv: Optional[Path] = None) -> Dict:
    """Full parity report for day D (does all the I/O)."""
    cache_path = production_cache_path()
    cache_max_date, day_symbols = scan_cache(cache_path, day)
    bt_stale = cache_max_date is None or day > cache_max_date
    if bt_stale:
        logger.warning(
            f"BT side UNKNOWABLE for {day}: production cache ends "
            f"{cache_max_date} — reporting BT_STALE, live side only "
            f"(no clean-parity claim is possible)")
        bt_trades: List[Dict] = []
    else:
        out_csv = bt_output_csv or (PARITY_DIR / f'bf_parity_bt_{day}.csv')
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        bt_trades = run_stage2_backtest(day, out_csv)

    live_rows = load_live_bf_rows(day, db_path=db_path)
    cls = classify_day(
        bt_trades, live_rows,
        stage1_day_symbols=None if bt_stale else day_symbols,
        bt_stale=bt_stale,
        is_common_lookup=default_is_common_lookup)
    return {
        'day': day,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': overall_status(bt_stale, cache_max_date,
                                 cls['n_divergent']),
        'bt_stale': bt_stale,
        'cache_path': str(cache_path),
        'cache_max_date': cache_max_date,
        'stage1_day_symbols': sorted(day_symbols),
        'n_bt_trades': len(bt_trades),
        'n_live_rows': len(live_rows),
        'bt_trades': bt_trades,
        **cls,
    }


def format_summary(rep: Dict) -> str:
    """Human-readable stdout report."""
    lines = [
        "=" * 70,
        f"  BF DECISION PARITY — {rep['day']}",
        "=" * 70,
        f"  Status:        {rep['status']}",
        f"  Cache:         {rep['cache_path']} (ends {rep['cache_max_date']})",
        f"  BT trades:     {rep['n_bt_trades']}"
        + ("  [UNKNOWABLE — stale]" if rep['bt_stale'] else ""),
        f"  Live rows:     {rep['n_live_rows']} (all order_status)",
        "-" * 70,
    ]
    for c in rep['both']:
        mark = 'DIVERGE' if c['divergent'] else 'ok'
        eb = (f"{c['entry_delta_bps']:+.1f}bps"
              if c['entry_delta_bps'] is not None else 'n/a')
        dl = (f"${c['pnl_delta_usd']:+.2f}"
              if c['pnl_delta_usd'] is not None else 'n/a')
        bt_pnl = f"${c['bt_pnl']:+.2f}" if c['bt_pnl'] is not None else 'n/a'
        lv_pnl = (f"${c['live_pnl']:+.2f}"
                  if c['live_pnl'] is not None else 'n/a')
        lines.append(
            f"  BOTH  {c['symbol']:<6} [{mark}] entryΔ={eb} "
            f"exit BT={c['bt_exit_reason']}/live={c['live_exit_reason']} "
            f"pnl BT={bt_pnl}/live={lv_pnl} Δ={dl}")
        for r in c['divergence_reasons']:
            lines.append(f"        └─ {r}")
    for c in rep['bt_only']:
        pnl = f"${c['bt_pnl']:+.2f}" if c['bt_pnl'] is not None else 'n/a'
        lines.append(
            f"  BT_ONLY   {c['symbol']:<6} {c['label']} "
            f"(BT {pnl} {c['bt_exit_reason']})")
    for c in rep['live_only']:
        pnl = (f"${c['live_pnl']:+.2f}"
               if c['live_pnl'] is not None else 'n/a')
        lines.append(
            f"  LIVE_ONLY {c['symbol']:<6} {c['label']} "
            f"(status={c['live_status']}, pnl={pnl})")
    if not (rep['both'] or rep['bt_only'] or rep['live_only']):
        lines.append("  (no BF activity on either side)")
    lines.append("=" * 70)
    return '\n'.join(lines)


def format_telegram(rep: Dict) -> str:
    """Compact HTML-safe Telegram message: status + one line per divergence."""
    lines = [f"[BF PARITY] {rep['day']}: {rep['status']} — "
             f"BT {rep['n_bt_trades']} trade(s), live {rep['n_live_rows']} "
             f"row(s)"]
    for d in rep['divergences']:
        lines.append(f"• {d}")
    if rep['bt_stale']:
        for c in rep['live_only']:
            lines.append(
                f"• {c['symbol']}: LIVE_ONLY — {c['label']} "
                f"(status={c['live_status']})")
        lines.append("BT side unknowable — check nightly_bt_update cron.")
    return html.escape('\n'.join(lines), quote=False)


def write_json(rep: Dict, path: Path) -> None:
    """Persist the machine-readable report (evidence retention)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=1, default=str))
    logger.info(f"Parity JSON written: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bull Flag live-vs-BT decision-parity report (Stage 1)")
    parser.add_argument(
        '--date', default=None,
        help="Trading day YYYY-MM-DD (default: previous trading day UTC)")
    parser.add_argument(
        '--no-telegram', action='store_true',
        help="Skip the Telegram send (validation/dev runs)")
    parser.add_argument(
        '--json-out', default=None,
        help="Override JSON output path "
             "(default: logs/bf_parity/bf_parity_<D>.json)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])

    day = args.date or rc.prev_trading_day_utc()
    try:
        datetime.strptime(day, '%Y-%m-%d')
    except ValueError:
        # Report tool contract: never non-zero exit, but be loudly broken.
        logger.error(f"Invalid --date {day!r} — expected YYYY-MM-DD")
        if not args.no_telegram:
            rc.send_telegram(html.escape(
                f"[BF PARITY] FAILED: invalid --date {day!r}", quote=False))
        return 0

    try:
        rep = build_report(day)
        print(format_summary(rep))
        json_path = (Path(args.json_out) if args.json_out
                     else PARITY_DIR / f'bf_parity_{day}.json')
        write_json(rep, json_path)
        if args.no_telegram:
            logger.info("--no-telegram: skipping Telegram send")
        else:
            if not rc.send_telegram(format_telegram(rep)):
                logger.warning(
                    "Telegram send failed/skipped — report persisted to "
                    "JSON regardless")
    except Exception as e:
        logger.error(f"bf_decision_parity FAILED for {day}: {e}",
                     exc_info=True)
        if not args.no_telegram:
            rc.send_telegram(html.escape(
                f"[BF PARITY] FAILED for {day}: {e}", quote=False))
    return 0  # report tool — never blocks crons


if __name__ == '__main__':
    sys.exit(main())
