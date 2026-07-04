"""Spread-gate fine-tune study (owner ask 2026-07-04).

The live-only spread gate (skip entry when NBBO spread > 150bps at 9:35)
has never been BT-validated — BT has no quote data. This study fetches
HISTORICAL NBBO at submission time for every post-veto defended trade
and answers:
  1. What P&L did trades in each spread bucket produce? (Is spread a
     loser-signal, a monster-signal, or noise?)
  2. Threshold sweep 100/150/200/300/none with an HONEST exit-cost
     penalty on wide names (stop exits sell into the bid: extra
     spread/2 beyond the 10bps model; sensitivity at full spread).
  3. Do the top-10 giants have wide 9:35 spreads? (Gate vs lottery.)
  4. Delay variant: for >150bps names, does the spread compress by
     9:40/9:45 enough to enter late instead of skipping?

What bars CANNOT model (flagged, not silently assumed): fill probability
when the ask sits above our limit — live fills on wide names may skew
adverse. Conclusions phrased accordingly.

Usage: PYTHONPATH=/home/ec2-user/onemil python3 research/scripts/orb_spread_gate_study.py
Artifacts: /tmp/orb_spread_gate_quotes.csv, /tmp/orb_spread_gate_report.txt
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '.env'))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest

ET = ZoneInfo('America/New_York')
GATE_BPS = 150.0
REPORT = []


def log(m=''):
    print(m, flush=True)
    REPORT.append(str(m))


def quote_at(client, sym, day, hh, mm):
    """Last NBBO in the 15s window ending at hh:mm:05 ET. None if no quotes."""
    t = datetime(int(day[:4]), int(day[5:7]), int(day[8:10]), hh, mm, 5,
                 tzinfo=ET)
    try:
        req = StockQuotesRequest(
            symbol_or_symbols=sym, start=t - timedelta(seconds=15), end=t,
            limit=50)
        quotes = client.get_stock_quotes(req).data.get(sym, [])
        for q in reversed(quotes):
            if q.bid_price and q.ask_price and q.ask_price >= q.bid_price > 0:
                return (float(q.bid_price), float(q.ask_price),
                        (q.ask_price - q.bid_price) / q.bid_price * 10000)
    except Exception:
        pass
    return None


def main():
    sel = pd.read_csv('/tmp/orb_base_sel.csv')
    sel = sel[sel['prev_day_range_pct'] > 8.0].copy()   # shipped book
    sel['day'] = sel['date'].astype(str).str[:10]
    client = StockHistoricalDataClient(os.getenv('ALPACA_API_KEY'),
                                       os.getenv('ALPACA_API_SECRET'))
    rows = []
    for i, (_, r) in enumerate(sel.iterrows()):
        q935 = quote_at(client, r['symbol'], r['day'], 9, 35)
        rec = dict(symbol=r['symbol'], day=r['day'],
                   sized_pnl=r['_sized_pnl'], exit_reason=r['exit_reason'],
                   entry_price=r['entry_price'],
                   spread_bps=q935[2] if q935 else None,
                   bid=q935[0] if q935 else None,
                   ask=q935[1] if q935 else None)
        # delay probes only for wide names (API frugality)
        if q935 and q935[2] > GATE_BPS:
            for tag, mm in (('s940', 40), ('s945', 45)):
                qq = quote_at(client, r['symbol'], r['day'], 9, mm)
                rec[tag] = qq[2] if qq else None
        rows.append(rec)
        if (i + 1) % 100 == 0:
            log(f"  ... {i + 1}/{len(sel)} quoted")
    df = pd.DataFrame(rows)
    df.to_csv('/tmp/orb_spread_gate_quotes.csv', index=False)

    got = df.dropna(subset=['spread_bps'])
    log(f"\nquote coverage: {len(got)}/{len(df)} trades "
        f"({len(got) / len(df) * 100:.0f}%)")
    log(f"9:35 spread: median {got['spread_bps'].median():.0f}bps  "
        f"p75 {got['spread_bps'].quantile(.75):.0f}  "
        f"p90 {got['spread_bps'].quantile(.9):.0f}  "
        f"p97 {got['spread_bps'].quantile(.97):.0f}")

    # 1. P&L by spread bucket (as-simulated, i.e. 10bps exit model)
    got['bucket'] = pd.cut(got['spread_bps'], [0, 50, 100, 150, 250, 99999],
                           labels=['0-50', '50-100', '100-150', '150-250', '>250'])
    log("\nP&L by 9:35 spread bucket (defended book, model exits):")
    log(got.groupby('bucket', observed=True)['sized_pnl']
        .agg(['count', 'sum', 'mean']).round(0).to_string())

    # 3. giants
    log("\ntop-10 winners' 9:35 spreads:")
    for _, r in got.nlargest(10, 'sized_pnl').iterrows():
        log(f"  {r['symbol']:>6} {r['day']}  ${r['sized_pnl']:>+8,.0f}  "
            f"{r['spread_bps']:.0f}bps")

    # 2. threshold sweep with exit-cost honesty:
    # exit sells into the bid -> extra cost ~ spread/2 beyond the 10bps
    # model (sensitivity: full spread). Position value ~ _sized_pnl's
    # notional; approximate extra cost = notional * extra_bps. We don't
    # have notional here, so express via pnl adjustment per trade using
    # position $ implied by risk-parity: use sized position column if
    # present else skip exact and use bps of a $10K stage position note.
    # Simpler + honest: report the sweep on RAW sized_pnl (upper bound
    # for keeping wide names) AND with a penalty of spread_bps/2 and
    # spread_bps applied to a $16.7K avg model position.
    AVG_POS = 16700.0  # median risk-parity position on the defended book
    log("\nthreshold sweep (defended book, post-veto):")
    log(f"{'gate':>8} {'kept':>5} {'raw P&L':>12} {'-sprd/2 pen':>12} {'-full sprd':>12}")
    for thr in (100, 150, 200, 300, None):
        keep = got if thr is None else got[got['spread_bps'] <= thr]
        wide = keep[keep['spread_bps'] > GATE_BPS]
        pen_half = (wide['spread_bps'] / 2 / 10000 * AVG_POS).sum()
        pen_full = (wide['spread_bps'] / 10000 * AVG_POS).sum()
        raw = keep['sized_pnl'].sum()
        log(f"{str(thr) if thr else 'none':>8} {len(keep):>5} "
            f"${raw:>+11,.0f} ${raw - pen_half:>+11,.0f} ${raw - pen_full:>+11,.0f}")

    # 4. delay/compression on wide names
    wide = df[(df['spread_bps'].notna()) & (df['spread_bps'] > GATE_BPS)]
    if len(wide):
        log(f"\nwide (> {GATE_BPS:.0f}bps) at 9:35: {len(wide)} trades, "
            f"${wide['sized_pnl'].sum():+,.0f} model P&L")
        for tag, label in (('s940', '9:40'), ('s945', '9:45')):
            if tag in wide.columns:
                w = wide.dropna(subset=[tag])
                if len(w):
                    ok = (w[tag] <= GATE_BPS).sum()
                    log(f"  by {label}: {ok}/{len(w)} compressed under the gate "
                        f"(median {w[tag].median():.0f}bps)")
    with open('/tmp/orb_spread_gate_report.txt', 'w') as f:
        f.write('\n'.join(REPORT))
    log("\ndone")


if __name__ == '__main__':
    main()
