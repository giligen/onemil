"""D3_exec — minute-by-minute reconstruction of every pathological live exit.

READ-ONLY. Reads `data/trades.db` and `data/cache.db` through `mode=ro` URIs,
writes only under `research/fuckup_audit/D3_exec/`.

For each trade whose `exit_reason` is one of the execution-pathology values
(market fallback, timeout, bracket SL race, never reconciled, thin-liquidity
reject, unknown exit) this script prints:

  * the plan (fill, stop, shares, R per share)
  * the exit telemetry (trigger, quote at trigger, limit placed, pricing
    method, submit -> fill latency, realized fill)
  * the 1-min tape from two minutes before the exit submission to five
    minutes after, so the fill can be compared with what the market offered
  * the dollars and R lost BEYOND a clean stop fill, defined as
    `stop_price * 0.999` (the task's definition), and beyond the limit the
    engine itself computed
  * an obtainability check: was the engine's own limit reachable inside the
    order's life (bar high >= limit)?

Counterfactual conventions (stated so they can be checked):
  C1 "clean stop"    : fill = stop_price * 0.999
  C2 "own limit"     : fill = the limit price the engine computed, IF a bar
                       high during the order's life reached it (a resting sell
                       limit fills when the tape trades at or above it)
  C3 "5s escalation" : the limit's life is cut from 10s to 5s; the escalation
                       target is a marketable limit at bid - 2x the observed
                       spread instead of an unpriced market order. Evaluated
                       on the same bars.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

TRADES = 'file:data/trades.db?mode=ro'
CACHE = 'file:data/cache.db?mode=ro'
OUT = 'research/fuckup_audit/D3_exec'

REASONS = (
    'stop_loss_market_fallback', 'stop_loss_timeout',
    'stop_loss_bracket_sl_race', 'never_reconciled',
    'thin_liquidity_reject', 'unknown_exit',
)


def parse_ts(s):
    """Parse an ISO-8601 timestamp from the DB into an aware UTC datetime."""
    if not s:
        return None
    s = str(s).replace('Z', '+00:00')
    if ' ' in s and 'T' not in s:
        s = s.replace(' ', 'T')
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_events():
    """Return every closed trade row carrying a pathological exit_reason."""
    con = sqlite3.connect(TRADES, uri=True)
    con.row_factory = sqlite3.Row
    q = ("select * from trades where exit_reason in (%s) order by trade_date, id"
         % ','.join('?' * len(REASONS)))
    rows = [dict(r) for r in con.execute(q, REASONS)]
    con.close()
    return rows


def load_bars(symbol, date, t0, t1):
    """1-min bars for (symbol, date) between t0 and t1 inclusive (UTC)."""
    con = sqlite3.connect(CACHE, uri=True)
    q = ("select timestamp, open, high, low, close, volume from "
         "intraday_bars_1min where symbol=? and bar_date=? order by timestamp")
    out = []
    for ts, o, h, lo, c, v in con.execute(q, (symbol, date)):
        t = parse_ts(ts)
        if t is None:
            continue
        if t0 <= t <= t1:
            out.append((t, o, h, lo, c, v))
    con.close()
    return out


def fmt(x, nd=4):
    return '—' if x is None else f'{x:.{nd}f}'


def main():
    events = load_events()
    rows_out = []
    lines = []
    for e in events:
        sym, day = e['symbol'], e['trade_date']
        fill = e['fill_price']
        stop = e['real_stop_loss_price'] or e['stop_loss_price']
        qty = e['filled_qty'] or e['shares'] or 0
        exit_px = e['exit_price']
        pnl = e['pnl']
        t_exit = parse_ts(e['exited_at'])
        t_sub = parse_ts(e['exit_submitted_at'])
        t_fill = parse_ts(e['filled_at'])
        limit = e['exit_limit_price']
        bid = e['exit_quote_bid']
        ask = e['exit_quote_ask']
        trig = e['exit_trigger_price']

        anchor = t_sub or t_exit
        bars = []
        if anchor:
            bars = load_bars(sym, day, anchor - timedelta(minutes=3),
                             anchor + timedelta(minutes=6))

        # --- counterfactuals -------------------------------------------
        clean = stop * 0.999 if stop else None
        excess_clean = None      # $ lost beyond a clean stop fill
        r_excess = None
        r_per_share = (fill - stop) if (fill and stop) else None
        if exit_px is not None and clean is not None and qty and fill:
            excess_clean = (exit_px - clean) * qty
            if r_per_share and r_per_share > 0:
                r_excess = (exit_px - clean) / r_per_share

        excess_limit = None      # $ lost beyond the engine's own limit
        limit_reachable = None
        if exit_px is not None and limit and qty:
            excess_limit = (exit_px - limit) * qty
            if t_sub and bars:
                life = [b for b in bars
                        if t_sub <= b[0] <= (t_exit or t_sub)]
                if not life:
                    life = [b for b in bars if b[0] <= (t_exit or t_sub)][-1:]
                limit_reachable = any(b[2] >= limit for b in life) if life else None

        rows_out.append(dict(
            id=e['id'], symbol=sym, date=day, strategy=e['strategy'],
            exit_reason=e['exit_reason'], fill_price=fill, stop=stop, qty=qty,
            exit_price=exit_px, pnl=pnl, r_per_share=r_per_share,
            trigger=trig, bid=bid, ask=ask, limit=limit,
            method=e['exit_pricing_method'],
            latency_ms=e['exit_fill_latency_ms'],
            clean_stop_px=clean, excess_vs_clean=excess_clean,
            r_excess=r_excess, excess_vs_own_limit=excess_limit,
            limit_reachable_in_life=limit_reachable,
            submitted=str(t_sub) if t_sub else None,
            exited=str(t_exit) if t_exit else None,
            entered=str(t_fill) if t_fill else None,
        ))

        lines.append('=' * 96)
        lines.append(
            f"#{e['id']} {sym} {day} [{e['strategy']}] {e['exit_reason']}")
        lines.append(
            f"  plan: entry_fill={fmt(fill)} stop={fmt(stop)} qty={qty} "
            f"R/sh={fmt(r_per_share)} planned_risk=${(r_per_share or 0)*qty:,.0f}")
        lines.append(
            f"  exit: trigger={fmt(trig)} bid={fmt(bid,2)} ask={fmt(ask,2)} "
            f"spread={fmt((ask-bid) if (ask and bid) else None,3)} "
            f"limit={fmt(limit,2)} method={e['exit_pricing_method']} "
            f"latency={fmt(e['exit_fill_latency_ms'],0)}ms")
        lines.append(
            f"  result: exit_px={fmt(exit_px)} pnl=${(pnl if pnl is not None else 0):,.2f} "
            f"clean_stop={fmt(clean)} excess_vs_clean=${(excess_clean or 0):,.2f} "
            f"({fmt(r_excess,2)}R) excess_vs_own_limit=${(excess_limit or 0):,.2f} "
            f"limit_reachable={limit_reachable}")
        lines.append(f"  entered={t_fill} submitted={t_sub} exited={t_exit}")
        if bars:
            lines.append('  tape (UTC)      open    high     low   close      vol   mark')
            for t, o, h, lo, c, v in bars:
                mark = []
                if t_sub and t.replace(second=0) == t_sub.replace(second=0, microsecond=0):
                    mark.append('SUBMIT')
                if t_exit and t.replace(second=0) == t_exit.replace(second=0, microsecond=0):
                    mark.append('FILL')
                if t_fill and t.replace(second=0) == t_fill.replace(second=0, microsecond=0):
                    mark.append('ENTRY')
                lines.append(
                    f"   {t.strftime('%H:%M')}  {o:9.4f}{h:9.4f}{lo:9.4f}{c:9.4f}"
                    f"{v:9d}   {','.join(mark)}")
        else:
            lines.append('  tape: NO BARS in window')

    with open(f'{OUT}/events_walk.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    import csv
    with open(f'{OUT}/events.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print('\n'.join(lines))
    print(f'\nwrote {OUT}/events_walk.txt and {OUT}/events.csv  (n={len(rows_out)})')


if __name__ == '__main__':
    sys.exit(main())
