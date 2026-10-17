"""Pure helpers for the momentum sleeve's LIVE path on the SHARED live account.

The sleeve's own state file is the single source of truth for what it owns; the owner's manual positions are
invisible here (the BMNR rule: never touched, never listed). Everything in this module is pure (no broker, no
files except the optional state/weekly readers) so scripts/momentum_sleeve.py, scripts/hourly_summary.py and
scripts/eod_sections.py share ONE implementation of the sleeve view, the rails and the MOM-LIVE line.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger('momentum_live')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_STATE_PATH = os.path.join(ROOT, 'data', 'momentum_sleeve', 'state_live.json')
LIVE_LEDGER_PATH = os.path.join(ROOT, 'logs', 'momentum_sleeve_live_ledger.csv')
LIVE_WEEKLY_PATH = os.path.join(ROOT, 'logs', 'momentum_sleeve_live_weekly.csv')
LIVE_SHADOW_GATE_PATH = os.path.join(ROOT, 'logs', 'momentum_sleeve_live_shadow_gate.csv')
LIVE_WEEKLY_FIELDS = ['date', 'equity', 'cash', 'n_names', 'turnover_usd', 'names_in', 'names_out', 'spy_close',
                      'week_r', 'weeks_since_strong']
LIVE_TG_PREFIX = '[MOM LIVE]'
LIVE_COID_PREFIX = 'moml-'
PAPER_COID_PREFIX = 'mom-'
DD_PAUSE_FRAC = 0.75          # sleeve equity <= 0.75 x equity_start at a rotation -> liquidate + pause
STRONG_WEEK_R = 5.0           # a strong week is >= +5R, R = 1 % of the slice
R_FRACTION = 0.01
DEFAULT_CADENCE_WARN_WEEKS = 8
DUST_QTY = 1e-6
PAUSE_CLEAR_CMD = ("python3 -c \"import json,sys; p='data/momentum_sleeve/state_live.json'; s=json.load(open(p)); "
                   "s['paused']=False; s.pop('paused_reason',None); json.dump(s,open(p,'w'),indent=2,sort_keys=True)\"")


def sleeve_view(all_qty: Dict[str, float], state_symbols: Iterable[str]) -> Tuple[Dict[str, float], int]:
    """(qty of the sleeve's own symbols only, count of OTHER broker positions). Other symbols are never named."""
    own = set(state_symbols)
    view = {s: q for s, q in all_qty.items() if s in own and q > DUST_QTY}
    ignored = sum(1 for s, q in all_qty.items() if s not in own and q > DUST_QTY)
    return view, ignored


def find_collisions(selected: Iterable[str], all_qty: Dict[str, float], state_symbols: Iterable[str]) -> List[str]:
    """Selected names the broker holds that the sleeve does not own (held manually) -- their buys are skipped."""
    own = set(state_symbols)
    return [s for s in selected if s not in own and all_qty.get(s, 0.0) > DUST_QTY]


def drop_collision_buys(orders: List[Dict], collisions: Iterable[str]) -> List[Dict]:
    """Orders without the buys of collided names (the slot's cash stays in cash -- no substitute, no refill)."""
    bad = set(collisions)
    return [o for o in orders if not (o['side'] == 'buy' and o['symbol'] in bad)]


def sell_quantity(state_qty: float, broker_qty_: float, want: Optional[float]) -> Tuple[float, Optional[str]]:
    """(qty to sell, warning|None). Never more than the STATE qty; never more than the broker holds.

    ``want`` None = full exit of the state qty. Broker MORE than state: sell only the state qty (WARN, the extra is
    the owner's). Broker LESS: sell what is there (WARN; the caller sets the state qty to the broker's)."""
    target = state_qty if want is None else min(state_qty, want)
    warn = None
    if broker_qty_ > state_qty + 1e-6:
        warn = f"broker holds more than the sleeve state ({broker_qty_:.6f} > {state_qty:.6f}) -- selling the state qty only"
    elif broker_qty_ < state_qty - 1e-6:
        warn = f"broker holds less than the sleeve state ({broker_qty_:.6f} < {state_qty:.6f}) -- selling what is there"
    return max(0.0, min(target, broker_qty_)), warn


def apply_fills(state: Dict, fills: List[Dict]) -> None:
    """Book fills into the sleeve state: cash and positions are state-tracked (never read from the account)."""
    pos = state.setdefault('positions', {})
    for f in fills:
        sym, qty, notional = f['symbol'], float(f['qty']), float(f['qty']) * float(f['avg_price'])
        if f['side'] == 'buy':
            pos[sym] = pos.get(sym, 0.0) + qty
            state['cash'] = float(state['cash']) - notional
        else:
            pos[sym] = pos.get(sym, 0.0) - qty
            state['cash'] = float(state['cash']) + notional
    for s in [s for s, q in pos.items() if q <= DUST_QTY]:
        del pos[s]


def buying_power_ok(buying_power: float, buy_notional: float) -> bool:
    """True when the account's buying power covers the whole buy side of this rotation."""
    return float(buying_power) >= float(buy_notional)


def drawdown_breached(equity: float, equity_start: float) -> bool:
    """The pre-committed pause rail: sleeve equity <= 0.75 x equity_start."""
    return float(equity) <= DD_PAUSE_FRAC * float(equity_start)


def week_r_series(rows: List[Dict], equity_start: float) -> List[Tuple[str, float]]:
    """[(date, week R)] from the weekly rows: R = 1 % of the slice; the first week is measured from the start."""
    out, prev = [], float(equity_start)
    for r in rows:
        try:
            eq = float(r['equity'])
        except (KeyError, ValueError, TypeError):
            logger.warning("momentum_live: weekly row unreadable %r", r)
            continue
        out.append((r['date'], (eq - prev) / (R_FRACTION * equity_start)))
        prev = eq
    return out


def cadence(rows: List[Dict], equity_start: float, today: date, new_equity: Optional[float] = None,
            ) -> Tuple[Optional[float], int]:
    """(this rotation's week R or None, weeks since the last strong week). ``new_equity`` is today's mark.

    No strong week yet: weeks are counted from the first weekly row (or today when there is none)."""
    series = week_r_series(rows, equity_start)
    cur_r = None
    if new_equity is not None:
        prev = float(rows[-1]['equity']) if rows else float(equity_start)
        cur_r = (float(new_equity) - prev) / (R_FRACTION * equity_start)
        series.append((str(today), cur_r))
    strong = [d for d, r in series if r >= STRONG_WEEK_R]
    anchor = strong[-1] if strong else (series[0][0] if series else str(today))
    return cur_r, max(0, (today - date.fromisoformat(anchor)).days // 7)


def read_json(path: str) -> Optional[Dict]:
    """The state JSON or None (missing / unreadable; unreadable is a WARNING)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("momentum_live: state %s unreadable: %s", path, e)
        return None


def sleeve_marks(positions: Iterable, state: Dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(price, previous-close price) for the sleeve's OWN symbols from broker position objects; all others dropped."""
    own = set(state.get('positions', {}))
    px, prev = {}, {}
    for p in positions:
        sym = getattr(p, 'symbol', None)
        if sym not in own:
            continue
        cur, last = getattr(p, 'current_price', None), getattr(p, 'lastday_price', None)
        if cur is not None:
            px[sym] = float(cur)
        if last is not None:
            prev[sym] = float(last)
    return px, prev


def mom_live_line(state: Dict, marks: Dict[str, float], prev_close: Dict[str, float],
                  weekly_rows: List[Dict], day: date, equity_start: Optional[float] = None) -> str:
    """`MOM-LIVE: day | week | since start | DD` on the SLICE, from the state + marks of its own names only.

    day = sum qty x (mark - previous close); week = equity minus the last weekly mark before this Monday (the
    start equity if none); DD against the state's peak (or the equity itself when higher)."""
    start = float(equity_start or state.get('equity_start') or 0.0)
    pos = state.get('positions', {})
    eq = float(state.get('cash', 0.0))
    day_pl, missing = 0.0, []
    for sym, qty in pos.items():
        if sym not in marks:
            missing.append(sym)
            continue
        eq += qty * marks[sym]
        if sym in prev_close:
            day_pl += qty * (marks[sym] - prev_close[sym])
    if missing:
        logger.warning("momentum_live: %d sleeve names without a mark -- valued at 0", len(missing))
    monday = day.toordinal() - day.weekday()
    base = start
    for r in weekly_rows:
        try:
            if date.fromisoformat(r['date']).toordinal() < monday:
                base = float(r['equity'])
        except (KeyError, ValueError, TypeError):
            continue
    peak = max(float(state.get('peak_equity', eq) or eq), eq)
    dd = (peak - eq) / peak * 100 if peak else 0.0
    flag = " PAUSED" if state.get('paused') else ""
    return (f"MOM-LIVE{flag}: day ${day_pl:+,.0f} | week ${eq - base:+,.0f} | since start ${eq - start:+,.0f} "
            f"(slice ${eq:,.0f} of ${start:,.0f}) | DD from peak {dd:.1f} %")


def read_weekly(path: str) -> List[Dict]:
    """The live weekly CSV rows ([] when absent)."""
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))
