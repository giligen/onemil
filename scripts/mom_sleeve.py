#!/usr/bin/env python3
"""Weekly momentum PAPER sleeve (risk-adjusted 12-1 top 20, equal weight, rebalanced every Monday).

Frozen rule: research/momentum_weekly/RECON_1700_sleeve.md, PREREG_1700j.md Amendment 1; selection lives in
trading/mom_sleeve_select.py (shared with the research reference, parity-tested). Paper account only
(ALPACA_ORB_API_KEY / ALPACA_ORB_API_SECRET, refuses anything else). The account also holds the ORB book and
the TOM QQQ position: the sleeve value is the market value of the positions THIS script owns (logs/
mom_sleeve_state.json) and ONLY those symbols are ever sold -- never the account equity, never QQQ.

Two phases so nothing heavy runs at the 09:30-09:36 ORB decision:
  --plan     any time before the open on the first session of the week: fetch + selection, writes
             logs/mom_sleeve_plan_<YYYYMMDD>.json (ranked picks, signals, prior closes, universe size, fetch
             completeness). No orders.
  --execute  window 09:40-09:55 ET: REQUIRES today's plan file (missing / another day -> ERROR, no orders), no bar
             fetch (positions + quotes only). All 20 names reset to sleeve value / 20: sells first (leavers in
             full, kept names trimmed), wait for fills, then buys (notional). DAY market orders,
             client_order_id 'mom-<YYYYMMDD>-<SYM>-<buy|sell>'.
  --liquidate  sell every position in the state file (paper only; rollback), client_order_id 'momliq-...'.
Idempotent: the order list is persisted in the state file ('pending') and every order is looked up by
client_order_id before submission, so a re-run submits nothing twice and a partial run resumes. The ledger
(logs/mom_sleeve_ledger.csv) records signal, prior close, 09:30 open and slippage vs that open in bps.

Kill rule (logged, not auto-acted): sleeve drawdown from its high >= 40 % -> ERROR + Telegram 'halve the sleeve'.

Usage:
    python3 scripts/mom_sleeve.py --plan [--dry-run]      # --dry-run writes <plan>.dryrun.json (ignored by --execute)
    python3 scripts/mom_sleeve.py --execute [--dry-run]   # --dry-run prints the orders, writes nothing
    python3 scripts/mom_sleeve.py --liquidate [--dry-run]
    python3 scripts/mom_sleeve.py --dry-run               # one-shot preview: plan + orders, text file under logs/
"""
import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd                                                  # noqa: E402
from config import Config                                            # noqa: E402
from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError  # noqa: E402
from notifications.telegram_notifier import TelegramNotifier         # noqa: E402
from trading import mom_sleeve_data as msd                           # noqa: E402
from trading import mom_sleeve_select as mss                         # noqa: E402

logger = logging.getLogger('mom_sleeve')

ET = ZoneInfo('US/Eastern')
WINDOW_START, WINDOW_END = dtime(9, 40), dtime(9, 55)   # --execute window (ORB decides 09:30-09:36)
PLAN_CUTOFF = dtime(9, 25)                                # --plan must finish before the open
DEFAULT_NOTIONAL = 20_000.0
HISTORY_CALENDAR_DAYS = 480           # ~330 trading days >= the 300 required
COMPLETENESS_FRACTION = 0.90
MIN_TRADE_USD = 5.0
KILL_DRAWDOWN = 0.40
FILL_TIMEOUT_SEC = 240
POLL_SEC = 3
TG_PREFIX = '[MOM]'
STATE_PATH = os.path.join(ROOT, 'logs', 'mom_sleeve_state.json')
LEDGER_PATH = os.path.join(ROOT, 'logs', 'mom_sleeve_ledger.csv')
PLAN_DIR = os.path.join(ROOT, 'logs')
LEDGER_FIELDS = ['date', 'symbol', 'side', 'qty', 'notional', 'fill_price', 'prior_close', 'open_0930',
                 'slippage_bps', 'rank', 'signal', 'client_order_id', 'note']
TERMINAL_BAD = ('canceled', 'expired', 'rejected', 'done_for_day', 'stopped', 'suspended')


class MomAbort(RuntimeError):
    """Raised when the run must stop without submitting anything (completeness gate, bad data)."""


# --------------------------------------------------------------------------- safety / calendar
def assert_paper_account(alpaca_client) -> None:
    """Refuse anything that is not unambiguously the paper account (wrapper flag AND broker base URL)."""
    if not getattr(alpaca_client, 'is_paper', False):
        raise RuntimeError("AlpacaClient.is_paper is False -- refusing to run against a live account")
    base = getattr(getattr(alpaca_client, 'trading_client', None), '_base_url', None)
    base_str = str(getattr(base, 'value', base) or '')
    if 'paper' not in base_str.lower():
        raise RuntimeError(f"broker base URL does not look like paper ({base_str!r}) -- refusing to run")


def _d(v) -> date:
    """Normalize a calendar 'date' (date or datetime) to a date."""
    return v.date() if isinstance(v, datetime) else v


def week_sessions(today: date, alpaca_client) -> List[date]:
    """Trading sessions of the Monday-Sunday week containing `today`, ascending ([] + ERROR on API failure)."""
    monday = today - timedelta(days=today.weekday())
    try:
        cal = alpaca_client.get_market_calendar(monday, monday + timedelta(days=6))
    except AlpacaAPIError as e:
        logger.error("mom_sleeve: calendar lookup failed: %s", e)
        return []
    return sorted(_d(c['date']) for c in cal)


def is_rebalance_day(today: date, alpaca_client) -> bool:
    """True iff `today` is the first trading session of its week (Monday, or the next session after a
    holiday Monday)."""
    s = week_sessions(today, alpaca_client)
    return bool(s) and today == s[0]


def prior_session(today: date, alpaca_client) -> date:
    """Last trading session strictly before `today` (the signal date)."""
    cal = alpaca_client.get_market_calendar(today - timedelta(days=10), today - timedelta(days=1))
    sessions = sorted(_d(c['date']) for c in cal)
    if not sessions:
        raise MomAbort("no trading session found in the 10 days before today -- calendar API problem")
    return sessions[-1]


def in_window(now_et: datetime) -> bool:
    """True iff the ET time is inside the 09:40-09:55 execute window."""
    return WINDOW_START <= now_et.time() <= WINDOW_END


# --------------------------------------------------------------------------- state / ledger
def load_state(path: str = STATE_PATH) -> Dict:
    """Read the sleeve state ({} skeleton when the file does not exist yet)."""
    if not os.path.exists(path):
        return {'positions': {}}
    with open(path) as f:
        st = json.load(f)
    st.setdefault('positions', {})
    return st


def save_state(state: Dict, path: str = STATE_PATH) -> None:
    """Atomic state write (tmp + rename)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def read_ledger(path: str = LEDGER_PATH) -> List[Dict]:
    """All ledger rows, oldest first ([] when absent)."""
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def append_ledger(row: Dict, path: str = LEDGER_PATH) -> None:
    """Append one row (header written for a new file)."""
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in LEDGER_FIELDS})


def notify(notifier: Optional[TelegramNotifier], msg: str) -> None:
    """One [MOM]-prefixed Telegram message; a send failure is a WARNING, never fatal."""
    if notifier is None:
        return
    try:
        notifier.send_message_sync(f"{TG_PREFIX} {msg}")
    except Exception as e:
        logger.warning("mom_sleeve: Telegram send failed (non-fatal): %s", e)


# --------------------------------------------------------------------------- broker helpers
def client_order_id(run_date: date, symbol: str, side: str) -> str:
    """'mom-<YYYYMMDD>-<SYM>-<buy|sell>'."""
    if side not in ('buy', 'sell'):
        raise ValueError(f"bad side {side!r}")
    return f"mom-{run_date:%Y%m%d}-{symbol}-{side}"


def _order_dict(o) -> Dict:
    """Broker order object -> plain dict with float fill fields (fractional-safe)."""
    def f(v):
        return float(v) if v not in (None, '') else 0.0
    return dict(id=str(o.id), coid=str(o.client_order_id), status=str(getattr(o.status, 'value', o.status)).lower(),
                filled_qty=f(o.filled_qty), fill_price=f(o.filled_avg_price) or None)


def get_order_by_coid(alpaca_client, coid: str) -> Optional[Dict]:
    """The order carrying `coid`, or None when the broker has none (404). Other errors propagate."""
    try:
        return _order_dict(alpaca_client.trading_client.get_order_by_client_id(coid))
    except Exception as e:
        if '404' in str(e) or 'not found' in str(e).lower():
            return None
        raise


def submit_order_once(alpaca_client, order: Dict) -> Dict:
    """Submit a DAY market order for `order` unless one with its client_order_id already exists (idempotent).
    Buys use notional, sells use qty (fractional). Returns the broker order dict."""
    existing = get_order_by_coid(alpaca_client, order['coid'])
    if existing is not None:
        logger.info("mom_sleeve: %s already at the broker (status %s) -- not resubmitted", order['coid'],
                    existing['status'])
        return existing
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    kw = dict(symbol=order['symbol'], time_in_force=TimeInForce.DAY, client_order_id=order['coid'],
              side=OrderSide.BUY if order['side'] == 'buy' else OrderSide.SELL)
    if order['side'] == 'buy':
        kw['notional'] = round(order['notional'], 2)
    else:
        kw['qty'] = order['qty']
    res = alpaca_client.trading_client.submit_order(MarketOrderRequest(**kw))
    logger.info("mom_sleeve: submitted %s %s %s", order['side'].upper(), order['symbol'], order['coid'])
    return _order_dict(res)


def wait_for_fills(alpaca_client, coids: List[str], timeout: float = FILL_TIMEOUT_SEC,
                   poll: float = POLL_SEC) -> Dict[str, Dict]:
    """Poll until every coid is filled or terminal-bad, or `timeout`. Returns coid -> last order dict."""
    deadline = time.time() + timeout
    last: Dict[str, Dict] = {}
    pending = set(coids)
    while pending:
        for c in list(pending):
            o = get_order_by_coid(alpaca_client, c)
            last[c] = o
            if o is not None and (o['status'] == 'filled' or o['status'] in TERMINAL_BAD):
                pending.discard(c)
        if not pending or time.time() >= deadline:
            break
        time.sleep(poll)
    for c in pending:
        logger.warning("mom_sleeve: %s not final after %ss (%s)", c, timeout, last.get(c))
    return last


def latest_prices(alpaca_client, symbols: List[str], fallback: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Latest SIP trade price per symbol; a missing price falls back to `fallback` (e.g. prior close) with a
    WARNING, or is left out with an ERROR when there is no fallback."""
    got = {}
    try:
        got = {s: v['price'] for s, v in alpaca_client.get_latest_trades(sorted(set(symbols))).items() if v['price']}
    except AlpacaAPIError as e:
        logger.error("mom_sleeve: get_latest_trades failed: %s", e)
    for s in set(symbols) - set(got):
        if fallback and fallback.get(s):
            logger.warning("mom_sleeve: no latest trade for %s -- using fallback price %.4f", s, fallback[s])
            got[s] = fallback[s]
        else:
            logger.error("mom_sleeve: no price at all for %s", s)
    return got


def official_opens(alpaca_client, symbols: List[str], today: date) -> Dict[str, float]:
    """Today's daily-bar open per symbol (the 09:30 official open proxy); {} entries missing -> WARNING."""
    out: Dict[str, float] = {}
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        for i in range(0, len(symbols), 200):
            req = StockBarsRequest(symbol_or_symbols=symbols[i:i + 200], timeframe=TimeFrame.Day,
                                   start=pd.Timestamp(today), feed=DataFeed.SIP)
            df = alpaca_client.data_client.get_stock_bars(req).df
            if len(df):
                for (sym, _), r in df.iterrows():
                    out[sym] = float(r['open'])
    except Exception as e:
        logger.warning("mom_sleeve: official-open lookup failed (%s) -- slippage will be blank", e)
    for s in set(symbols) - set(out):
        logger.warning("mom_sleeve: no 09:30 open for %s -- slippage blank", s)
    return out


# --------------------------------------------------------------------------- planning
def build_orders(state: Dict, picks: List[Dict], prices: Dict[str, float], run_date: date,
                 notional: float = DEFAULT_NOTIONAL) -> (List[Dict], float):
    """Equal-weight reset. Sleeve value = owned positions' market value (inception: `notional`). Returns
    (orders, sleeve_value); sells first in the list. Only symbols in state['positions'] are ever sold."""
    pos = state.get('positions', {})
    value = sum(p['qty'] * prices[s] for s, p in pos.items() if s in prices) if pos else notional
    missing = [s for s in pos if s not in prices]
    if missing:
        raise MomAbort(f"no price for owned positions {missing} -- cannot value the sleeve")
    target = value / mss.TOP_N
    pick_by = {p['symbol']: p for p in picks}
    sells, buys = [], []
    for s, p in pos.items():
        mv = p['qty'] * prices[s]
        meta = pick_by.get(s, {})
        base = dict(symbol=s, rank=meta.get('rank', ''), signal=meta.get('signal', ''),
                    prior_close=meta.get('prior_close', ''))
        if s not in pick_by:
            sells.append(dict(base, side='sell', qty=p['qty'], notional=mv, note='left_top20'))
        elif mv - target > MIN_TRADE_USD:
            q = round(min(p['qty'], (mv - target) / prices[s]), 6)
            sells.append(dict(base, side='sell', qty=q, notional=q * prices[s], note='trim'))
    for pk in picks:
        mv = pos.get(pk['symbol'], {}).get('qty', 0.0) * prices.get(pk['symbol'], 0.0)
        if target - mv > MIN_TRADE_USD:
            buys.append(dict(symbol=pk['symbol'], rank=pk['rank'], signal=pk['signal'], prior_close=pk['prior_close'],
                             side='buy', qty='', notional=round(target - mv, 2),
                             note='entrant' if pk['symbol'] not in pos else 'top_up'))
    orders = sells + buys
    for o in orders:
        o['coid'] = client_order_id(run_date, o['symbol'], o['side'])
    return orders, value


def format_plan(run_date: date, picks: List[Dict], skipped: List[Dict], orders: List[Dict], value: float,
                universe_size: int, signal_date: date) -> str:
    """Human-readable plan text (also the dated plan file)."""
    target = value / mss.TOP_N
    lines = [f"mom_sleeve plan for {run_date} (signal date {signal_date}, universe {universe_size} names, "
             f"sleeve value ${value:,.2f}, target ${target:,.2f}/name)", "",
             "| rank | symbol | signal | prior close | target notional |", "|---|---|---|---|---|"]
    for p in picks:
        lines.append(f"| {p['rank']} | {p['symbol']} | {p['signal']:.3f} | {p['prior_close']:.2f} | ${target:,.2f} |")
    if skipped:
        lines += ["", "Skipped (ineligible at Alpaca), next rank taken:"]
        lines += [f"- rank {s['rank']} {s['symbol']}: {s['reason']}" for s in skipped]
    lines += ["", "Orders (sells first):"]
    for o in orders:
        amt = f"qty {o['qty']}" if o['side'] == 'sell' else f"notional ${o['notional']:,.2f}"
        lines.append(f"- {o['coid']}: {o['side'].upper()} {o['symbol']} {amt} ({o['note']})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- execution
def apply_fill(state: Dict, order: Dict, fill_qty: float, fill_price: float) -> None:
    """Update owned positions (qty, cost basis) for one filled order."""
    pos = state['positions']
    s = order['symbol']
    if order['side'] == 'buy':
        p = pos.setdefault(s, {'qty': 0.0, 'cost': 0.0})
        p['qty'] = round(p['qty'] + fill_qty, 6)
        p['cost'] += fill_qty * fill_price
    else:
        p = pos.get(s)
        if p is None:
            logger.error("mom_sleeve: sell fill for %s which the state does not own -- ignoring", s)
            return
        frac = min(1.0, fill_qty / p['qty']) if p['qty'] else 1.0
        p['cost'] *= (1 - frac)
        p['qty'] = round(p['qty'] - fill_qty, 6)
        if p['qty'] <= 1e-6:
            del pos[s]


def record_fills(alpaca_client, state: Dict, orders: List[Dict], results: Dict[str, Dict], run_date: date,
                 opens: Dict[str, float], ledger_path: str, state_path: str) -> List[float]:
    """Ledger + state update for every filled order not already in the ledger. Returns slippage bps list."""
    done = {r['client_order_id'] for r in read_ledger(ledger_path)}
    slips: List[float] = []
    for o in orders:
        res = results.get(o['coid'])
        if res is None or res['status'] != 'filled' or o['coid'] in done:
            if res is None or res['status'] != 'filled':
                logger.error("mom_sleeve: %s not filled (%s)", o['coid'], res)
            continue
        px, q = res['fill_price'], res['filled_qty']
        op = opens.get(o['symbol'])
        slip = ''
        if op and px:
            slip = round(((px / op - 1) if o['side'] == 'buy' else (op / px - 1)) * 1e4, 2)
            slips.append(slip)
        apply_fill(state, o, q, px)
        append_ledger(dict(date=run_date.isoformat(), symbol=o['symbol'], side=o['side'], qty=q,
                           notional=round(q * px, 2), fill_price=px, prior_close=o['prior_close'],
                           open_0930=op or '', slippage_bps=slip, rank=o['rank'], signal=o['signal'],
                           client_order_id=o['coid'], note=o['note']), ledger_path)
        save_state(state, state_path)
    return slips


def execute_orders(alpaca_client, state: Dict, orders: List[Dict], run_date: date, ledger_path: str,
                   state_path: str, fill_timeout: float = FILL_TIMEOUT_SEC, poll: float = POLL_SEC) -> List[float]:
    """Sells first, wait for their fills, then buys. Aborts the buys (MomAbort) if any sell is not filled.
    Every submit is idempotent by client_order_id. Returns the slippage list (bps)."""
    sells = [o for o in orders if o['side'] == 'sell']
    buys = [o for o in orders if o['side'] == 'buy']
    opens = official_opens(alpaca_client, sorted({o['symbol'] for o in orders}), run_date)
    slips: List[float] = []
    for batch in (sells, buys):
        if not batch:
            continue
        for o in batch:
            submit_order_once(alpaca_client, o)
        res = wait_for_fills(alpaca_client, [o['coid'] for o in batch], fill_timeout, poll)
        slips += record_fills(alpaca_client, state, batch, res, run_date, opens, ledger_path, state_path)
        if batch is sells and any((res.get(o['coid']) or {}).get('status') != 'filled' for o in sells):
            raise MomAbort("a sell is not filled -- buys withheld; re-run to resume (pending plan kept)")
    return slips


# --------------------------------------------------------------------------- selection (live data)
def select_picks(alpaca_client, state: Dict, today: date):
    """Fetch roster + bars, rank, apply the completeness gate and the tradable/fractionable skip rule.
    Returns (picks, skipped, universe_size, signal_date, coverage)."""
    t0 = time.time()
    signal_date = prior_session(today, alpaca_client)
    roster = msd.fetch_asset_roster(alpaca_client)
    cand = msd.candidate_symbols(roster)
    logger.info("mom_sleeve: %d candidate symbols; fetching bars through %s", len(cand), signal_date)
    bars, lost = msd.fetch_bars(alpaca_client, cand, today - timedelta(days=HISTORY_CALENDAR_DAYS),
                                  signal_date + timedelta(days=1))   # end is a midnight timestamp: +1 day keeps the signal-date bar
    n_bars = bars['symbol'].nunique() if len(bars) else 0
    coverage = n_bars / max(1, len(cand) - sum(1 for r in lost.values() if r == 'invalid_symbol_api_rejected'))
    logger.info("mom_sleeve: bars for %d/%d candidates (coverage %.1f%%, %d lost), fetch %.0fs", n_bars, len(cand),
                100 * coverage, len(lost), time.time() - t0)
    table = mss.compute_signal_table(bars, pd.Timestamp(signal_date))
    if table.empty:
        raise MomAbort(f"no symbol has a bar dated {signal_date} with enough history -- stale/incomplete bars")
    ranked = mss.rank_universe(table)
    usize = ranked.attrs['universe_size']
    prev = state.get('universe_size')
    if prev:
        if usize < COMPLETENESS_FRACTION * prev:
            raise MomAbort(f"completeness gate: universe {usize} < {COMPLETENESS_FRACTION:.0%} of last week's {prev}")
    else:
        logger.warning("mom_sleeve: no prior universe size in state -- gating on fetch coverage %.1f%% instead",
                       100 * coverage)
        if coverage < COMPLETENESS_FRACTION:
            raise MomAbort(f"completeness gate: fetch coverage {coverage:.1%} < {COMPLETENESS_FRACTION:.0%}")
    if usize == 0 or ranked.empty:
        raise MomAbort("empty universe after gates")
    info = roster.set_index('symbol')

    def eligible(sym):
        r = info.loc[sym] if sym in info.index else None
        if r is None:
            return False, 'not in Alpaca asset roster'
        if not r.tradable:
            return False, 'not tradable'
        if not r.fractionable:
            return False, 'not fractionable'
        return True, ''
    picks, skipped = mss.pick_top(ranked, eligible)
    if len(picks) < mss.TOP_N:
        raise MomAbort(f"only {len(picks)} eligible picks")
    return picks, skipped, usize, signal_date, coverage


# --------------------------------------------------------------------------- run
def plan_path(plan_dir: str, today: date, dryrun: bool = False) -> str:
    """logs/mom_sleeve_plan_<YYYYMMDD>.json ('.dryrun.json' for --plan --dry-run, never read by --execute)."""
    return os.path.join(plan_dir, f"mom_sleeve_plan_{today:%Y%m%d}{'.dryrun' if dryrun else ''}.json")


def run_plan(alpaca_client, now_utc: datetime, dry_run: bool = False, force_window: bool = False,
             state_path: str = STATE_PATH, plan_dir: str = PLAN_DIR) -> str:
    """Phase 1: fetch + select, write the plan JSON. No orders. Real runs only before 09:25 ET on a rebalance day."""
    assert_paper_account(alpaca_client)
    now_et = now_utc.astimezone(ET)
    today = now_et.date()
    rebalance = is_rebalance_day(today, alpaca_client)
    if not rebalance:
        if not dry_run:
            return f"{today} is not the first session of its week -- no-op"
        logger.warning("mom_sleeve: %s is not a rebalance day -- dry-run plan only", today)
    if not dry_run and not force_window and now_et.time() >= PLAN_CUTOFF:
        logger.error("mom_sleeve: --plan at %s ET is after the 09:25 cutoff (heavy fetch must not overlap the ORB "
                     "open) -- refusing", f"{now_et:%H:%M}")
        return f"plan refused: {now_et:%H:%M} ET is after the 09:25 cutoff"
    state = load_state(state_path)
    picks, skipped, usize, signal_date, coverage = select_picks(alpaca_client, state, today)
    doc = dict(date=today.isoformat(), signal_date=signal_date.isoformat(), universe_size=usize,
               fetch_coverage=round(coverage, 4), picks=picks, skipped=skipped,
               created_utc=now_utc.isoformat())
    path = plan_path(plan_dir, today, dryrun=dry_run)
    os.makedirs(plan_dir, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)
    logger.info("mom_sleeve: plan written to %s (%d picks, universe %d, coverage %.1f%%)", path, len(picks), usize,
                100 * coverage)
    return format_plan(today, picks, skipped, [], float(state.get('post_value') or DEFAULT_NOTIONAL), usize, signal_date)


def load_plan(plan_dir: str, today: date) -> Dict:
    """Today's plan JSON; MomAbort (ERROR) when it is missing or dated another day."""
    path = plan_path(plan_dir, today)
    if not os.path.exists(path):
        raise MomAbort(f"no plan file for {today} ({path}) -- run --plan first; no orders")
    with open(path) as f:
        doc = json.load(f)
    if doc.get('date') != today.isoformat():
        raise MomAbort(f"plan file {path} is dated {doc.get('date')}, not {today} -- no orders")
    return doc


def run_execute(alpaca_client, notifier, now_utc: datetime, dry_run: bool = False, force_window: bool = False,
                notional: float = DEFAULT_NOTIONAL, state_path: str = STATE_PATH, ledger_path: str = LEDGER_PATH,
                plan_dir: str = PLAN_DIR, fill_timeout: float = FILL_TIMEOUT_SEC) -> str:
    """Phase 2: today's plan file + current positions/quotes -> orders (no bar fetch). Window 09:40-09:55 ET."""
    assert_paper_account(alpaca_client)
    now_et = now_utc.astimezone(ET)
    today = now_et.date()
    if not dry_run:
        if not is_rebalance_day(today, alpaca_client):
            return f"{today} is not the first session of its week -- no-op"
        if not (force_window or in_window(now_et)):
            return f"{now_et:%H:%M} ET outside the 09:40-09:55 window -- no-op"
    state = load_state(state_path)
    pending = state.get('pending')
    prices = None
    if pending and pending.get('date') == today.isoformat() and not dry_run:
        logger.info("mom_sleeve: resuming the pending order list of %s", today)
        orders, value, usize = pending['orders'], pending['value'], pending['universe_size']
        picks, skipped, signal_date = pending['picks'], [], pending['signal_date']
    else:
        try:
            doc = load_plan(plan_dir, today)
        except MomAbort as e:
            logger.error("mom_sleeve: %s", e)
            notify(notifier, f"ERROR {e}")
            raise
        picks, skipped, usize, signal_date = doc['picks'], doc.get('skipped', []), doc['universe_size'], doc['signal_date']
        need = sorted({p['symbol'] for p in picks} | set(state['positions']) | {'SPY'})
        prices = latest_prices(alpaca_client, need, {p['symbol']: p['prior_close'] for p in picks})
        orders, value = build_orders(state, picks, prices, today, notional)
    plan = format_plan(today, picks, skipped, orders, value, usize, date.fromisoformat(str(signal_date)))
    if dry_run:
        return plan
    if prices is not None:
        state.setdefault('inception', {'date': today.isoformat(), 'notional': value, 'spy': prices.get('SPY')})
        state['pending'] = dict(date=today.isoformat(), orders=orders, value=value, universe_size=usize,
                                picks=picks, signal_date=signal_date)
        save_state(state, state_path)
        _kill_rule(state, value, notifier)
    try:
        slips = execute_orders(alpaca_client, state, orders, today, ledger_path, state_path, fill_timeout)
    except MomAbort as e:
        logger.error("mom_sleeve: %s", e)
        notify(notifier, f"ERROR {e}")
        raise
    summary = _finish(alpaca_client, state, state_path, ledger_path, today, orders, value, usize, slips, prices)
    notify(notifier, summary)
    return summary


def run_liquidate(alpaca_client, notifier, now_utc: datetime, dry_run: bool = False,
                  state_path: str = STATE_PATH, ledger_path: str = LEDGER_PATH,
                  fill_timeout: float = FILL_TIMEOUT_SEC) -> str:
    """Rollback: sell every position in the state file (and ONLY those), paper account only."""
    assert_paper_account(alpaca_client)
    today = now_utc.astimezone(ET).date()
    state = load_state(state_path)
    orders = [dict(symbol=s, side='sell', qty=p['qty'], notional=0.0, rank='', signal='', prior_close='',
                   note='liquidate', coid=f"momliq-{today:%Y%m%d}-{s}-sell") for s, p in sorted(state['positions'].items())]
    if not orders:
        logger.warning("mom_sleeve: --liquidate with an empty state file -- nothing to sell")
        return "nothing to liquidate"
    text = "liquidate (sells only): " + ", ".join(f"{o['symbol']} {o['qty']}" for o in orders)
    if dry_run:
        return text
    execute_orders(alpaca_client, state, orders, today, ledger_path, state_path, fill_timeout)
    left = sorted(state['positions'])
    if left:
        logger.error("mom_sleeve: liquidate left positions %s", left)
    else:
        state.pop('pending', None)
        save_state(state, state_path)
    notify(notifier, f"liquidated sleeve: {len(orders) - len(left)}/{len(orders)} sells filled")
    return text


def run(alpaca_client, notifier, now_utc: datetime, dry_run: bool = True, notional: float = DEFAULT_NOTIONAL,
        state_path: str = STATE_PATH, plan_dir: str = PLAN_DIR) -> str:
    """One-shot dry-run preview (plan + order list from live data); never submits. Writes a dated text file."""
    assert_paper_account(alpaca_client)
    if not dry_run:
        raise ValueError("combined plan+execute is not allowed live: use --plan then --execute")
    today = now_utc.astimezone(ET).date()
    state = load_state(state_path)
    picks, skipped, usize, sd, _ = select_picks(alpaca_client, state, today)
    need = sorted({p['symbol'] for p in picks} | set(state['positions']) | {'SPY'})
    prices = latest_prices(alpaca_client, need, {p['symbol']: p['prior_close'] for p in picks})
    orders, value = build_orders(state, picks, prices, today, notional)
    plan = format_plan(today, picks, skipped, orders, value, usize, sd)
    os.makedirs(plan_dir, exist_ok=True)
    with open(os.path.join(plan_dir, f"mom_sleeve_plan_{today:%Y%m%d}.txt"), 'w') as f:
        f.write(plan + "\n")
    return plan


def _kill_rule(state: Dict, value: float, notifier) -> None:
    """Track the sleeve high; at drawdown >= 40 % log ERROR and tell the owner to halve (never auto-acts)."""
    peak = max(state.get('peak_value', 0.0), value)
    state['peak_value'] = peak
    dd = value / peak - 1
    if dd <= -KILL_DRAWDOWN:
        msg = f"KILL RULE: sleeve ${value:,.0f} is {dd:.1%} from its high ${peak:,.0f} -- halve the sleeve (owner)"
        logger.error("mom_sleeve: %s", msg)
        notify(notifier, msg)


def _finish(alpaca_client, state, state_path, ledger_path, today, orders, value, usize, slips, prices) -> str:
    """Close the run: clear pending, store post-trade value/SPY/universe size, build the weekly summary line."""
    pos = state['positions']
    px = prices or latest_prices(alpaca_client, sorted(set(pos) | {'SPY'}))
    post = sum(p['qty'] * px.get(s, 0.0) for s, p in pos.items())
    prev_post, prev_spy = state.get('post_value'), state.get('spy_last')
    spy = px.get('SPY')
    inc = state.get('inception', {})
    wk = f"{value / prev_post - 1:+.2%}" if prev_post else 'n/a'
    wk_spy = f"{spy / prev_spy - 1:+.2%}" if prev_spy and spy else 'n/a'
    cum = f"{value / inc['notional'] - 1:+.2%}" if inc.get('notional') else 'n/a'
    cum_spy = f"{spy / inc['spy'] - 1:+.2%}" if inc.get('spy') and spy else 'n/a'
    all_slips = [float(r['slippage_bps']) for r in read_ledger(ledger_path) if r.get('slippage_bps') not in ('', None)]
    avg = f"{sum(slips) / len(slips):+.1f}" if slips else 'n/a'
    avg_all = f"{sum(all_slips) / len(all_slips):+.1f}" if all_slips else 'n/a'
    ins = sorted(o['symbol'] for o in orders if o['note'] == 'entrant')
    outs = sorted(o['symbol'] for o in orders if o['note'] == 'left_top20')
    state.update(post_value=post, spy_last=spy, universe_size=usize, last_run_date=today.isoformat())
    state.pop('pending', None)
    save_state(state, state_path)
    return (f"{today} sleeve ${value:,.0f} -> ${post:,.0f} | in {ins or '-'} out {outs or '-'} | week {wk} vs SPY "
            f"{wk_spy} | since inception {cum} vs SPY {cum_spy} | slippage vs open avg {avg} bps "
            f"(all-time {avg_all}, n {len(all_slips)})")


# --------------------------------------------------------------------------- CLI
def main() -> int:
    """CLI entry: build the paper client, run once."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true', help="submit nothing (see Usage for per-mode behaviour)")
    ap.add_argument('--plan', action='store_true', help="phase 1: fetch + select, write the plan JSON")
    ap.add_argument('--execute', action='store_true', help="phase 2: orders from today's plan (09:40-09:55 ET)")
    ap.add_argument('--liquidate', action='store_true', help="sell every position in the state file (rollback)")
    ap.add_argument('--force-window', action='store_true', help="TESTS ONLY: skip the time-window checks")
    ap.add_argument('--notional', type=float, default=DEFAULT_NOTIONAL, help="inception sleeve notional")
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    cfg = Config()
    if not cfg.alpaca_orb_api_key or not cfg.alpaca_orb_api_secret:
        logger.error("mom_sleeve: ALPACA_ORB_API_KEY/SECRET not set -- cannot run")
        return 1
    client = AlpacaClient(cfg.alpaca_orb_api_key, cfg.alpaca_orb_api_secret, paper=cfg.alpaca_orb_paper)
    notifier = None
    if cfg.telegram_bot_token and cfg.telegram_chat_id:
        notifier = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id, enabled=True)
    else:
        logger.warning("mom_sleeve: Telegram not configured -- proceeding without notifications")
    now = datetime.now(timezone.utc)
    try:
        assert_paper_account(client)
        if args.plan:
            out = run_plan(client, now, args.dry_run, args.force_window)
        elif args.execute:
            out = run_execute(client, None if args.dry_run else notifier, now, args.dry_run, args.force_window,
                              args.notional)
        elif args.liquidate:
            out = run_liquidate(client, None if args.dry_run else notifier, now, args.dry_run)
        elif args.dry_run:
            out = run(client, None, now, True, args.notional)
        else:
            logger.error("mom_sleeve: choose one of --plan / --execute / --liquidate / --dry-run")
            return 1
        print(out)
    except (RuntimeError, MomAbort) as e:
        logger.error("mom_sleeve: aborted -- %s", e)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
