#!/usr/bin/env python3
"""MOM sleeve weekly rule-vs-book reconciliation: the guarded BT rule replayed on panel + extension against what
the PAPER and LIVE sleeves actually held / earned.  Spec: docs/mom_bt_current_spec_20261031.md (Deliverable 2).

For each of the last 8 Mondays: the 20 BT names (research/momentum_weekly/REBUILD_1700tu.py functions, imported,
never copied) vs the names the paper sleeve held (logs/momentum_sleeve_weekly.csv replayed from names_in/out),
BT week return (equal weight, Monday open -> last close of the week, before cost) vs the paper week return
(MOM paper account portfolio history, slice-normalised) vs the live slice week return (live weekly marks +
the sleeve STATE marked to its own names only -- never the shared live account equity) vs SPY.

Writes/updates logs/mom_bt_parity.csv (one row per Monday) and sends ONE Telegram line, prefix [MOM].
Names < 18/20 or |residual| > 2 pp -> the line says MISMATCH and a WARNING lists the differing names.
A missing/stale extension -> ERROR (that IS a failure).

    python3 scripts/mom_bt_parity.py [--dry-run] [--no-send]      (cron: Saturday 13:20 UTC, after the refresh)
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'research', 'momentum_weekly')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from monitoring.cron_guard import attach_telegram_error_handler, run_guarded  # noqa: E402

logger = logging.getLogger('mom_bt_parity')

D = os.path.join(ROOT, 'research', 'momentum_weekly')
PANEL_PATH = os.path.join(D, 'panel_2016_2026.parquet')
EXT_PATH = os.path.join(D, 'panel_ext.parquet')
ASSETS_PATH = os.path.join(D, '1700c_assets.csv')
PARITY_PATH = os.path.join(ROOT, 'logs', 'mom_bt_parity.csv')
WEEKLY_PATH = os.path.join(ROOT, 'logs', 'momentum_sleeve_weekly.csv')
LEDGER_PATH = os.path.join(ROOT, 'logs', 'momentum_sleeve_ledger.csv')
FIELDS = ['date', 'bt_names_match', 'bt_ret', 'paper_ret', 'live_ret', 'spy_ret', 'residual_pp']
N_WEEKS = 8
NAMES_MIN = 18                 # of 20
RESID_MAX_PP = 2.0
STALE_DAYS = 6                 # extension last bar older than this (calendar days) = the Saturday refresh did not run
TG_PREFIX = '[MOM]'
SLEEVE_START_EQUITY = 20000.0  # the paper slice (scripts/eod_sections.SLEEVE_START_EQUITY)


# ----------------------------------------------------------------------------- data
def load_rule_module():
    """REBUILD_1700tu with ABSOLUTE paths (it ships relative ones): the guarded sleeve rule, imported as is."""
    import REBUILD_1700tu as R
    R.D = D + os.sep
    R.PANEL, R.ASSETS = PANEL_PATH, ASSETS_PATH
    return R


def extend_data(data: Dict, ext: pd.DataFrame) -> int:
    """Append extension bars (bar_date after each symbol's last panel date) to the loaded panel dict, in place.
    Returns the number of appended rows."""
    n_add = 0
    for sym, g in ext.groupby('symbol', sort=False):
        if sym not in data:
            continue
        d, o, h, l, c, v = data[sym]
        g = g[g['bar_date'] > pd.Timestamp(d[-1])].sort_values('bar_date')
        if g.empty:
            continue
        data[sym] = (np.concatenate([d, g['bar_date'].values.astype('datetime64[ns]')]),
                     *[np.concatenate([a, g[k].values.astype('float64')])
                       for a, k in zip((o, h, l, c, v), ('open', 'high', 'low', 'close', 'volume'))])
        n_add += len(g)
    return n_add


def monday_rebalances(cal: pd.DatetimeIndex, n: int) -> List[pd.Timestamp]:
    """The first session on/after each of the last ``n`` Mondays that has a bar in ``cal``."""
    mondays = pd.date_range(end=cal[-1], periods=n, freq='W-MON')
    out = []
    for m in mondays:
        p = cal.searchsorted(m, side='left')
        if p < len(cal) and cal[p] - m < pd.Timedelta(days=5) and cal[p] <= cal[-1]:
            out.append(cal[p])
    return sorted(set(out))[-n:]


def bt_weeks(R, data: Dict, n: int = N_WEEKS) -> List[Dict]:
    """Guarded top-20 (REBUILD_1700tu.features, same selection as its main()) and the BT week return for the last
    ``n`` Mondays: {monday, signal, names, ret, spy_ret, end, complete}.  Return = equal weight, Monday open ->
    last close before the next Monday (or the last bar for the running week), before cost."""
    cal = pd.DatetimeIndex(data['SPY'][0])
    rebs = monday_rebalances(cal, n)
    ts = np.array([cal[cal.searchsorted(r) - 1] for r in rebs], dtype='datetime64[ns]')
    rows = R.features(data, cal, ts, None)
    out = []
    for i, reb in enumerate(rebs):
        elig = sorted([x for x in rows[i] if not x[2]], key=lambda x: -x[1])
        names = [x[0] for x in elig[:R.TOPN]] if len(rows[i]) >= R.TOPN else []
        complete = i + 1 < len(rebs)
        end = cal[cal.searchsorted(rebs[i + 1]) - 1] if complete else cal[-1]
        rets = [R.px(data, s, end, False) / R.px(data, s, reb, True) - 1 for s in names]
        spy = R.px(data, 'SPY', end, False) / R.px(data, 'SPY', reb, True) - 1
        out.append({'monday': reb, 'signal': pd.Timestamp(ts[i]), 'names': names,
                    'ret': float(np.nanmean(rets)) if rets else float('nan'), 'spy_ret': float(spy),
                    'end': end, 'complete': complete})
    return out


# ----------------------------------------------------------------------------- books
def read_rows(path: str) -> List[Dict]:
    """CSV rows as dicts ([] when the file is absent)."""
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def paper_held(weekly: List[Dict], monday: pd.Timestamp) -> List[str]:
    """Paper holdings after the last weekly row dated <= ``monday`` (names_in/names_out replayed from the start;
    the last row of a date wins).  A week without a rotation therefore shows the previous book."""
    held, book = set(), None
    for r in weekly:
        if r['date'] > str(monday.date()):
            break
        held |= set((r.get('names_in') or '').split())
        held -= set((r.get('names_out') or '').split())
        book = sorted(held)
    return book or []


def _mark(rows: List[Dict], day: str, key: str = 'equity') -> Optional[float]:
    """Last mark of ``day`` in a weekly csv, None when absent."""
    vals = [float(r[key]) for r in rows if r.get('date') == day and r.get(key)]
    return vals[-1] if vals else None


def paper_week_ret(history: Optional[List[Dict]], weekly: List[Dict], monday: pd.Timestamp, end: pd.Timestamp,
                   session_date: Callable) -> float:
    """Paper slice week return: account equity (daily history, session-keyed) at ``end`` minus the last session
    before ``monday``, over the slice equity marked at the rotation (the slice start when absent).  NaN when the
    history does not cover the week."""
    if not history:
        return float('nan')
    sess = {session_date(b['stamp']): float(b['equity']) for b in history}
    prior = [k for k in sess if k < monday.date()]
    inwk = [k for k in sess if monday.date() <= k <= end.date()]
    if not prior or not inwk:
        return float('nan')
    base = _mark(weekly, str(monday.date())) or SLEEVE_START_EQUITY
    return (sess[max(inwk)] - sess[max(prior)]) / base


def live_week_ret(live_weekly: List[Dict], monday: pd.Timestamp, next_monday: Optional[pd.Timestamp],
                  equity_now: Optional[float]) -> float:
    """Live slice week return from the live weekly marks: equity(monday) -> equity(next rotation mark) or, for the
    last live rotation, the sleeve equity now (state + own-name marks).  NaN when a mark is missing."""
    e0 = _mark(live_weekly, str(monday.date()))
    if e0 is None:
        return float('nan')
    if next_monday is not None:
        e1 = _mark(live_weekly, str(next_monday.date()))      # a skipped rotation leaves no mark: not attributable
    else:
        later = [r for r in live_weekly if r['date'] > str(monday.date())]
        e1 = equity_now if not later else None
    if e1 is None:
        return float('nan')
    return e1 / e0 - 1


def build_rows(weeks: List[Dict], weekly: List[Dict], live_weekly: List[Dict], history: Optional[List[Dict]],
               equity_now: Optional[float], session_date: Callable) -> List[Dict]:
    """One parity row per Monday (see FIELDS) plus the names that differ (key `diff`)."""
    rows = []
    first_paper = weekly[0]['date'] if weekly else None
    for i, w in enumerate(weeks):
        if first_paper is None or str(w['monday'].date()) < first_paper:
            logger.info("mom_bt_parity: wk %s predates the paper sleeve (first rotation %s) -- no row", w['monday'].date(),
                        first_paper)
            continue
        held = paper_held(weekly, w['monday'])
        match = len(set(w['names']) & set(held))
        paper = paper_week_ret(history, weekly, w['monday'], w['end'], session_date)
        nxt = weeks[i + 1]['monday'] if i + 1 < len(weeks) else None
        live = live_week_ret(live_weekly, w['monday'], nxt, equity_now)
        resid = (paper - w['ret']) * 100 if not (math.isnan(paper) or math.isnan(w['ret'])) else float('nan')
        rows.append({'date': str(w['monday'].date()), 'bt_names_match': match, 'bt_ret': w['ret'],
                     'paper_ret': paper, 'live_ret': live, 'spy_ret': w['spy_ret'], 'residual_pp': resid,
                     'diff': {'bt_only': sorted(set(w['names']) - set(held)),
                              'paper_only': sorted(set(held) - set(w['names']))}})
    return rows


def is_mismatch(row: Dict) -> bool:
    """Names under NAMES_MIN/20 or |residual| over RESID_MAX_PP."""
    r = row['residual_pp']
    return row['bt_names_match'] < NAMES_MIN or (not math.isnan(r) and abs(r) > RESID_MAX_PP)


def _pct(x: float) -> str:
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.1f} %"


def format_line(row: Dict) -> str:
    """`BT parity wk 10/26: names 19/20 | BT -2.4 % | paper -2.1 % | live -5.4 % | SPY -0.5 % | resid +0.3 pp`
    (prefixed MISMATCH when the row fails the bar; the [MOM] prefix is added by the sender)."""
    d = date.fromisoformat(row['date'])
    r = row['residual_pp']
    resid = 'n/a' if math.isnan(r) else f"{r:+.1f} pp"
    tag = 'MISMATCH ' if is_mismatch(row) else ''
    return (f"{tag}BT parity wk {d.month}/{d.day}: names {row['bt_names_match']}/20 | BT {_pct(row['bt_ret'])} | "
            f"paper {_pct(row['paper_ret'])} | live {_pct(row['live_ret'])} | SPY {_pct(row['spy_ret'])} | "
            f"resid {resid}")


def write_parity_csv(rows: List[Dict], path: str) -> None:
    """Upsert ``rows`` into the parity csv by date (atomic temp + rename); history outside the window is kept."""
    keep = {r['date']: r for r in read_rows(path)}
    for r in rows:
        keep[r['date']] = {k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in FIELDS}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(keep):
            w.writerow({f_: keep[k][f_] for f_ in FIELDS})
    os.replace(tmp, path)


def latest_parity_line(path: str = PARITY_PATH) -> str:
    """The `MOM BT parity:` EOD line from the newest row of the csv; `MOM BT parity: NO-DATA (...)` if absent."""
    rows = read_rows(path)
    if not rows:
        return f"MOM BT parity: NO-DATA ({os.path.basename(path)} missing or empty)"
    r = rows[-1]
    row = {'date': r['date'], 'bt_names_match': int(float(r['bt_names_match'])),
           'residual_pp': float(r['residual_pp']) if r['residual_pp'] not in ('', 'nan') else float('nan')}
    for k in ('bt_ret', 'paper_ret', 'live_ret', 'spy_ret'):
        row[k] = float(r[k]) if r[k] not in ('', 'nan') else float('nan')
    return "MOM BT parity: " + format_line(row).replace('BT parity ', '', 1)


# ----------------------------------------------------------------------------- run
def run(R, data: Dict, weekly_path: str, live_weekly_path: str, parity_path: str, history: Optional[List[Dict]],
        equity_now: Optional[float], session_date: Callable, dry_run: bool = False,
        send: Optional[Callable[[str], None]] = None) -> List[Dict]:
    """Compute the parity rows, log WARNING for each MISMATCH, write the csv and send the latest line."""
    weeks = bt_weeks(R, data)
    rows = build_rows(weeks, read_rows(weekly_path), read_rows(live_weekly_path), history, equity_now, session_date)
    for r in rows:
        logger.info("mom_bt_parity: %s", format_line(r))
        if is_mismatch(r):
            logger.warning("mom_bt_parity: MISMATCH wk %s: BT-only %s | paper-only %s | resid %s pp", r['date'],
                           r['diff']['bt_only'], r['diff']['paper_only'], r['residual_pp'])
    if dry_run:
        print("DRY-RUN (nothing written, nothing sent):")
        for r in rows:
            print("  " + format_line(r))
        return rows
    write_parity_csv(rows, parity_path)
    if send is not None and rows:
        send(format_line(rows[-1]))
    return rows


def live_equity_now(client) -> Optional[float]:
    """Live sleeve equity from the STATE and the marks of its own names only (never the account equity);
    None with a WARNING when the state or the marks are unavailable."""
    from trading import momentum_live as ml
    import momentum_sleeve as sleeve
    state = ml.read_json(ml.LIVE_STATE_PATH)
    if state is None:
        return None
    try:
        marks, _ = ml.sleeve_marks(client.trading_client.get_all_positions(), state)
    except Exception as e:  # noqa: BLE001
        logger.warning("mom_bt_parity: live marks unavailable (%s) -- live return of the running week n/a", e)
        return None
    return sleeve.sleeve_equity(state, marks)


def main() -> int:
    """CLI entry (cron: Saturday 13:20 UTC)."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dry-run', action='store_true', help='print the rows, write and send nothing')
    ap.add_argument('--no-send', action='store_true')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    attach_telegram_error_handler()
    if not os.path.exists(EXT_PATH):
        logger.error("mom_bt_parity: extension %s missing -- the weekly panel refresh did not run", EXT_PATH)
        return 1
    ext = pd.read_parquet(EXT_PATH)
    ext['bar_date'] = pd.to_datetime(ext['bar_date'])
    age = (datetime.now(timezone.utc).date() - ext['bar_date'].max().date()).days
    if age > STALE_DAYS:
        logger.error("mom_bt_parity: extension last bar %s is %d days old (> %d) -- the weekly refresh failed",
                     ext['bar_date'].max().date(), age, STALE_DAYS)
        return 1
    from config import Config
    cfg = Config()  # loads .env
    from data_sources.alpaca_client import AlpacaClient
    from hourly_summary import fetch_history, session_date
    key, secret = os.environ.get('ALPACA_MOM_API_KEY', ''), os.environ.get('ALPACA_MOM_API_SECRET', '')
    history = None
    if key and secret:
        try:
            history = fetch_history(AlpacaClient(key, secret, paper=True), period='3M')
        except Exception as e:  # noqa: BLE001
            logger.warning("mom_bt_parity: paper portfolio history failed (%s) -- paper return n/a", e)
    else:
        logger.warning("mom_bt_parity: ALPACA_MOM_API_KEY/SECRET not set -- paper return n/a")
    equity_now = None
    lk, ls = os.environ.get('ALPACA_API_KEY', ''), os.environ.get('ALPACA_API_SECRET', '')
    if lk and ls:
        equity_now = live_equity_now(AlpacaClient(lk, ls, paper=False))
    else:
        logger.warning("mom_bt_parity: live ALPACA_API_KEY/SECRET not set -- live return of the running week n/a")
    R = load_rule_module()
    logger.info("mom_bt_parity: loading panel (this takes ~1-2 min, ~2 GB)")
    data = R.load()
    logger.info("mom_bt_parity: extension rows appended: %d", extend_data(data, ext))
    send = None
    if not args.no_send and not args.dry_run and cfg.telegram_bot_token and cfg.telegram_chat_id:
        from notifications.telegram_notifier import TelegramNotifier
        notifier = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id, enabled=True)
        send = lambda msg: notifier.send_message_sync(f"{TG_PREFIX} {msg}")  # noqa: E731
    from trading import momentum_live as ml
    run(R, data, WEEKLY_PATH, ml.LIVE_WEEKLY_PATH, PARITY_PATH, history, equity_now, session_date,
        dry_run=args.dry_run, send=send)
    return 0


if __name__ == '__main__':
    sys.exit(run_guarded(main, 'mom_bt_parity'))
