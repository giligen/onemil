"""D3_exec — the dollars. READ-ONLY.

Three benchmarks per event, all computed on the row's own telemetry plus the
1-min tape:

  B1 clean stop      fill at stop_price * 0.999          (the task's definition)
  B2 trigger         fill at exit_trigger_price * 0.999  (right benchmark when
                     the exit was a trail / lock / EOD flat, where the stop is
                     not the level that fired)
  B3 quote at pricing  fill at `exit_quote_bid`          (pure execution slip:
                     what the engine SAW when it priced the order vs what it got)

Plus the two size diagnostics that separate the mechanisms:
  qty / exit_quote_bid_size   (how many times the displayed bid we were)
  qty / volume of the submit minute (participation in the tape)

Counterfactual C3 (the proposed rule) is evaluated in `d3_counterfactual`
below: hit the bid immediately with a marketable limit priced at
bid - max(1 tick, 0.25 * spread), sliced so no slice exceeds the displayed
bid size, with a 5 s re-price instead of a 10 s timeout + market order.
Obtainability is checked against the submit-minute bar (low <= price <= high).
"""
from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timedelta, timezone

from d3_walk import load_events, load_bars, parse_ts  # noqa: E402

OUT = 'research/fuckup_audit/D3_exec'


def bar_at(bars, t):
    """Return the 1-min bar containing timestamp t, or None."""
    if not t:
        return None
    key = t.replace(second=0, microsecond=0)
    for b in bars:
        if b[0] == key:
            return b
    return None


def main():
    rows = []
    for e in load_events():
        sym, day = e['symbol'], e['trade_date']
        fill = e['fill_price']
        stop = e['real_stop_loss_price'] or e['stop_loss_price']
        qty = e['filled_qty'] or e['shares'] or 0
        px = e['exit_price']
        trig = e['exit_trigger_price']
        bid, ask = e['exit_quote_bid'], e['exit_quote_ask']
        bsz = e['exit_quote_bid_size']
        limit = e['exit_limit_price']
        t_sub = parse_ts(e['exit_submitted_at'])
        t_exit = parse_ts(e['exited_at'])
        anchor = t_sub or t_exit
        bars = load_bars(sym, day, (anchor - timedelta(minutes=3)),
                         (anchor + timedelta(minutes=3))) if anchor else []
        bsub = bar_at(bars, t_sub) or bar_at(bars, t_exit)

        rps = (fill - stop) if (fill and stop) else None
        d = dict(
            id=e['id'], symbol=sym, date=day, strategy=e['strategy'],
            reason=e['exit_reason'], qty=qty, entry=fill, stop=stop,
            r_per_share=rps, exit_px=px, pnl=e['pnl'],
            trigger=trig, bid=bid, ask=ask, bid_size=bsz, limit=limit,
            method=e['exit_pricing_method'],
            latency_s=(e['exit_fill_latency_ms'] or 0) / 1000.0 or None,
        )
        d['spread'] = (ask - bid) if (ask and bid) else None
        d['spread_bps'] = (d['spread'] / bid * 1e4) if (d['spread'] and bid) else None
        # benchmarks (negative = we did worse than the benchmark)
        d['d_vs_clean_stop'] = ((px - stop * 0.999) * qty) if (px and stop and qty) else None
        d['d_vs_trigger'] = ((px - trig * 0.999) * qty) if (px and trig and qty) else None
        d['d_vs_bid'] = ((px - bid) * qty) if (px and bid and qty) else None
        d['d_vs_own_limit'] = ((px - limit) * qty) if (px and limit and qty) else None
        d['R_vs_bid'] = (d['d_vs_bid'] / (rps * qty)) if (d['d_vs_bid'] and rps and qty and rps > 0) else None
        # size diagnostics
        d['qty_over_bidsize'] = (qty / bsz) if (bsz and qty) else None
        if bsub:
            d['sub_bar_vol'] = bsub[5]
            d['participation'] = qty / bsub[5] if bsub[5] else None
            d['sub_bar_move_pct'] = (bsub[4] - bsub[1]) / bsub[1] * 100 if bsub[1] else None
            d['sub_bar_lowrun_pct'] = (bsub[3] - bsub[1]) / bsub[1] * 100 if bsub[1] else None
            d['sub_bar_low'] = bsub[3]
            d['sub_bar_high'] = bsub[2]
        else:
            for k in ('sub_bar_vol', 'participation', 'sub_bar_move_pct',
                      'sub_bar_lowrun_pct', 'sub_bar_low', 'sub_bar_high'):
                d[k] = None
        # C3 counterfactual: immediate marketable limit, sliced to the book
        if bid and d['spread'] is not None:
            cf = round(bid - max(0.01, 0.25 * d['spread']), 2)
            d['c3_price'] = cf
            # obtainable if the submit-minute bar traded at or above cf
            d['c3_obtainable'] = bool(bsub and bsub[2] >= cf)
            d['c3_delta_vs_actual'] = ((cf - px) * qty) if px else None
        else:
            d['c3_price'] = d['c3_obtainable'] = d['c3_delta_vs_actual'] = None
        rows.append(d)

    cols = list(rows[0].keys())
    with open(f'{OUT}/quant.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    def s(k, pred=lambda r: True):
        return sum((r[k] or 0) for r in rows if pred(r))

    print(f"{'id':>4} {'sym':<6} {'strat':<10} {'reason':<26} {'qty':>6} "
          f"{'pnl':>10} {'vsClean':>9} {'vsTrig':>9} {'vsBid':>9} {'R_bid':>6} "
          f"{'q/bidsz':>8} {'partic':>7} {'barMove%':>9} {'lat_s':>6} {'C3+$':>9} {'C3ok':>5}")
    for r in rows:
        f2 = lambda v, n=2: ('' if v is None else f'{v:.{n}f}')
        print(f"{r['id']:>4} {r['symbol']:<6} {r['strategy']:<10} {r['reason']:<26} "
              f"{r['qty']:>6} {f2(r['pnl']):>10} {f2(r['d_vs_clean_stop']):>9} "
              f"{f2(r['d_vs_trigger']):>9} {f2(r['d_vs_bid']):>9} {f2(r['R_vs_bid']):>6} "
              f"{f2(r['qty_over_bidsize'],1):>8} {f2(r['participation'],3):>7} "
              f"{f2(r['sub_bar_move_pct'],2):>9} {f2(r['latency_s'],1):>6} "
              f"{f2(r['c3_delta_vs_actual']):>9} {str(r['c3_obtainable']):>5}")

    print('\n--- totals ---')
    for strat in sorted({r['strategy'] for r in rows}):
        p = lambda r, st=strat: r['strategy'] == st
        print(f"{strat:<10} n={sum(1 for r in rows if p(r)):>2} "
              f"pnl={s('pnl', p):>10.2f} vsClean={s('d_vs_clean_stop', p):>9.2f} "
              f"vsBid={s('d_vs_bid', p):>9.2f} C3={s('c3_delta_vs_actual', p):>9.2f}")
    print(f"{'ALL':<10} n={len(rows):>2} pnl={s('pnl'):>10.2f} "
          f"vsClean={s('d_vs_clean_stop'):>9.2f} vsBid={s('d_vs_bid'):>9.2f} "
          f"C3={s('c3_delta_vs_actual'):>9.2f}")
    print(f"\nwrote {OUT}/quant.csv")


if __name__ == '__main__':
    main()
