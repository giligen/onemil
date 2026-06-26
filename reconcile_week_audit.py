"""READ-ONLY reconciliation audit: DB trades vs Alpaca actual fills.

For every filled trade in a date range, list ALL of that symbol's orders on
that day from the correct Alpaca account, sum filled BUY executions (true
entry) and filled SELL executions (true exit), and compute the true realized
P&L. Diff against what the DB recorded.

NO WRITES. Prints a diff table. Use the output to decide which rows to fix.

Account routing: strategy 'orb' -> ALPACA_ORB_* ; everything else -> ALPACA_*.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

START = sys.argv[1] if len(sys.argv) > 1 else '2026-06-08'
END = sys.argv[2] if len(sys.argv) > 2 else '2026-06-12'

_clients = {}


def client_for(strategy: str) -> TradingClient:
    if strategy == 'orb':
        key, sec = os.getenv('ALPACA_ORB_API_KEY'), os.getenv('ALPACA_ORB_API_SECRET')
        paper = os.getenv('ALPACA_ORB_PAPER', 'true').lower() == 'true'
        tag = 'orb'
    else:
        key, sec = os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET')
        paper = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'
        tag = 'main'
    if tag not in _clients:
        if not key or not sec:
            raise RuntimeError(f"missing Alpaca creds for {tag}")
        _clients[tag] = TradingClient(key, sec, paper=paper)
        print(f"  [client {tag}: paper={paper}]", file=sys.stderr)
    return _clients[tag]


def true_fills(strategy: str, symbol: str, day: str):
    """Return (buy_qty, buy_notional, sell_qty, sell_notional, n_orders) of
    FILLED executions for `symbol` on `day` from the strategy's account."""
    tc = client_for(strategy)
    d0 = datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=timezone.utc) - timedelta(days=1)
    d1 = d0 + timedelta(days=3)
    req = GetOrdersRequest(
        status=QueryOrderStatus.ALL, symbols=[symbol],
        after=d0, until=d1, limit=500, nested=True,
    )
    orders = tc.get_orders(filter=req)
    bq = bn = sq = sn = 0.0
    n = 0
    # Flatten nested legs too (bracket SL/TP fills are separate executions)
    def walk(o):
        yield o
        for leg in (getattr(o, 'legs', None) or []):
            yield leg
    for top in orders:
        for o in walk(top):
            # restrict to the target trade day (submitted or filled that day)
            fa = getattr(o, 'filled_at', None)
            ref = fa or getattr(o, 'submitted_at', None)
            if ref is not None and ref.strftime('%Y-%m-%d') != day:
                continue
            fq = float(o.filled_qty) if o.filled_qty else 0.0
            if fq <= 0:
                continue
            fp = float(o.filled_avg_price) if o.filled_avg_price else 0.0
            side = str(o.side.value) if hasattr(o.side, 'value') else str(o.side)
            n += 1
            if side == 'buy':
                bq += fq; bn += fq * fp
            elif side == 'sell':
                sq += fq; sn += fq * fp
    return bq, bn, sq, sn, n


def main():
    conn = sqlite3.connect('data/trades.db')
    cur = conn.cursor()
    cur.execute("""
        SELECT id, trade_date, strategy, symbol, shares, filled_qty,
               fill_price, exit_price, pnl, exit_reason, order_status
          FROM trades
         WHERE trade_date BETWEEN ? AND ?
           AND filled_qty IS NOT NULL
         ORDER BY trade_date, id
    """, (START, END))
    rows = cur.fetchall()
    conn.close()

    print(f"\n{'='*118}")
    print(f"RECONCILIATION AUDIT (READ-ONLY)  {START}..{END}   {len(rows)} filled trades")
    print(f"{'='*118}")
    hdr = (f"{'db_id':>5} {'date':<10} {'strat':<9} {'sym':<6} "
           f"{'DB sh':>6} {'BRK buy':>8} {'BRK sell':>8} {'net':>5} "
           f"{'DB pnl':>10} {'TRUE pnl':>10} {'Δ pnl':>10}  flag")
    print(hdr); print('-'*118)

    fixes = []
    for r in rows:
        (tid, dt, strat, sym, sh, fq, fp, xp, pnl, xr, os_) = r
        try:
            bq, bn, sq, sn, n = true_fills(strat, sym, dt)
        except Exception as e:
            print(f"{tid:>5} {dt:<10} {strat:<9} {sym:<6}  ERROR: {e}")
            continue
        net_qty = bq - sq
        true_pnl = sn - bn if (bq > 0 and sq > 0) else None  # only if round-tripped
        db_pnl = pnl if pnl is not None else 0.0
        flag = ''
        dpnl = None
        if true_pnl is not None:
            dpnl = true_pnl - db_pnl
            if abs(dpnl) >= 1.0:
                flag = '*** PNL MISMATCH'
        if abs(bq - (fq or 0)) >= 1:
            flag += ' [QTY MISMATCH]'
        if abs(net_qty) >= 1:
            flag += f' [NOT FLAT net={net_qty:.0f}]'
        tp_s = f"${true_pnl:+,.2f}" if true_pnl is not None else '   (open?)'
        dp_s = f"${dpnl:+,.2f}" if dpnl is not None else '       —'
        print(f"{tid:>5} {dt:<10} {strat:<9} {sym:<6} "
              f"{sh or 0:>6} {bq:>8.0f} {sq:>8.0f} {net_qty:>5.0f} "
              f"${db_pnl:>+9.2f} {tp_s:>10} {dp_s:>10}  {flag}")
        if flag and 'NOT FLAT' not in flag:
            buy_avg = bn / bq if bq > 0 else None
            sell_avg = sn / sq if sq > 0 else None
            fixes.append((tid, sym, dt, sh, bq, db_pnl, true_pnl, buy_avg, sell_avg))

    print('-'*118)
    print(f"\nRows needing correction: {len(fixes)}")
    for tid, sym, dt, db_sh, brk_sh, db_pnl, true_pnl, buy_avg, sell_avg in fixes:
        tp = f"${true_pnl:+,.2f}" if true_pnl is not None else '?'
        ba = f"${buy_avg:.4f}" if buy_avg is not None else '?'
        sa = f"${sell_avg:.4f}" if sell_avg is not None else '?'
        print(f"  db_id={tid} {sym} {dt}: shares {db_sh}->{brk_sh:.0f}, "
              f"fill_avg={ba}, exit_avg={sa}, pnl ${db_pnl:+,.2f}->{tp}")
    print("\n(NO WRITES PERFORMED — this is an audit only.)")


if __name__ == '__main__':
    main()
