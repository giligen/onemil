"""Entry stop-limit buffer study — is 30bps right? (2026-07-04)

The limit sits at range_high x (1 + 30bps), a value inherited from bull
flag live data and never validated for ORB. Two questions, one dataset:
  1. BT FILL ASSUMPTION: BT counts a trade filled when bar-high crosses
     range_high. Was the ASK actually <= our +30bps limit during the
     breakout minute? If often not, BT overstates fills (and the monsters
     most of all — fast breakouts gap the ask through the limit).
  2. BUFFER SWEEP: what limit X ∈ {10,30,50,75,100,150}bps captures what
     fraction of breakouts, and at what extra entry cost?

Method: historical NBBO for [breakout_ts, breakout_ts + 90s] per defended
trade; fill(X) = any quote with ask <= rh*(1+X/1e4); entry cost = first
such ask. Honest limits: queue position / size at the ask unmodelable —
this bounds fill FEASIBILITY, not certainty.

Artifacts: /tmp/orb_entry_buffer.csv, /tmp/orb_entry_buffer_report.txt
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta, time as dtime

import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '.env'))

from persistence.database import Database
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest

BUFFERS = [10, 30, 50, 75, 100, 150]
REPORT = []


def log(m=''):
    print(m, flush=True)
    REPORT.append(str(m))


def main():
    sel = pd.read_csv('/tmp/orb_base_sel.csv')
    sel = sel[sel['prev_day_range_pct'] > 8.0].copy()
    db = Database(db_path=os.path.join(ROOT, 'data', 'cache.db'))
    raw = db.get_intraday_bars_bulk(
        [(r['symbol'], str(r['date'])[:10]) for _, r in sel.iterrows()])
    qc = StockHistoricalDataClient(os.getenv('ALPACA_API_KEY'),
                                   os.getenv('ALPACA_API_SECRET'))
    rows = []
    for i, (_, r) in enumerate(sel.iterrows()):
        sym, ds = r['symbol'], str(r['date'])[:10]
        bars = raw.get((sym, ds))
        if not bars:
            continue
        df = pd.DataFrame(bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        et = df['timestamp'].dt.tz_convert('America/New_York').dt.time
        m = df[et == dtime(9, 30)]
        if m.empty:
            continue
        ots = m.iloc[0]['timestamp']
        rb = df[(df['timestamp'] >= ots) & (df['timestamp'] < ots + timedelta(minutes=5))]
        if len(rb) < 5:
            continue
        rh = float(rb['high'].max())
        win = df[(df['timestamp'] >= ots + timedelta(minutes=5)) &
                 (df['timestamp'] < ots + timedelta(minutes=65))]
        hit = win[win['high'] > rh]
        if hit.empty:
            continue
        bts = hit.iloc[0]['timestamp']
        try:
            req = StockQuotesRequest(symbol_or_symbols=sym, start=bts.to_pydatetime(),
                                     end=(bts + timedelta(seconds=90)).to_pydatetime(),
                                     limit=500)
            quotes = qc.get_stock_quotes(req).data.get(sym, [])
        except Exception:
            quotes = []
        rec = dict(symbol=sym, day=ds, sized_pnl=r['_sized_pnl'],
                   range_high=rh, n_quotes=len(quotes))
        asks = [float(q.ask_price) for q in quotes if q.ask_price and q.ask_price > 0]
        if asks:
            rec['min_ask_bps'] = (min(asks) - rh) / rh * 10000
            for X in BUFFERS:
                lim = rh * (1 + X / 10000)
                fillable = [a for a in asks if a <= lim]
                rec[f'fill_{X}'] = bool(fillable)
                rec[f'cost_{X}'] = ((fillable[0] - rh) / rh * 10000) if fillable else None
        rows.append(rec)
        if (i + 1) % 100 == 0:
            log(f"  ... {i + 1}/{len(sel)}")
    out = pd.DataFrame(rows)
    out.to_csv('/tmp/orb_entry_buffer.csv', index=False)

    got = out[out['n_quotes'] > 0]
    log(f"\nquote coverage in breakout minute: {len(got)}/{len(out)}")
    log(f"min-ask over breakout+90s vs range_high: "
        f"median {got['min_ask_bps'].median():+.0f}bps  p75 {got['min_ask_bps'].quantile(.75):+.0f}  "
        f"p90 {got['min_ask_bps'].quantile(.9):+.0f}")
    log("\nbuffer sweep — fill feasibility + P&L at risk in unfillable trades:")
    log(f"{'buffer':>7} {'fillable':>9} {'unfillable P&L':>15} {'median cost':>12}")
    for X in BUFFERS:
        f = got[got[f'fill_{X}'] == True]   # noqa: E712
        nf = got[got[f'fill_{X}'] != True]  # noqa: E712
        med = f[f'cost_{X}'].median() if len(f) else float('nan')
        log(f"{X:>6}b {len(f)/len(got)*100:>8.0f}% ${nf['sized_pnl'].sum():>+14,.0f} {med:>+11.0f}b")
    # monsters check
    log("\ntop-10 winners: fillable at 30bps?")
    for _, r in got.nlargest(10, 'sized_pnl').iterrows():
        log(f"  {r['symbol']:>6} {r['day']} ${r['sized_pnl']:>+8,.0f}  "
            f"min_ask {r['min_ask_bps']:+.0f}bps  fill30={r['fill_30']}")
    with open('/tmp/orb_entry_buffer_report.txt', 'w') as fh:
        fh.write('\n'.join(REPORT))
    log('\ndone')


if __name__ == '__main__':
    main()
