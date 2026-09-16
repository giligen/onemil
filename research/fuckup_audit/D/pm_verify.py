#!/usr/bin/env python3
"""Independent hand-recompute of a few pm_bars.db rows straight from Alpaca.

Written WITHOUT reusing pm_backfill.py's aggregation code (CLAUDE.md
"no research claim ships without an independent check" — item 1). Pulls the
04:00-09:29 ET window fresh and recomputes every field of the pm row.
Read-only on pm_bars.db; market data only.

  python3 research/fuckup_audit/D/pm_verify.py [N]
"""
import sqlite3
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, '/home/ec2-user/onemil')
import logging
logging.basicConfig(level=logging.ERROR)
from config import Config
from data_sources.alpaca_client import AlpacaClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

ET = ZoneInfo('America/New_York')
DB = '/home/ec2-user/onemil/research/fuckup_audit/D/pm_bars.db'
COLS = 'symbol,day,n_pm_bars,pm_volume,pm_dollar_vol,pm_high,pm_low,pm_last,pm_vwap,src'


def main(n=3):
    cfg = Config()
    cl = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    rows = con.execute(f"select {COLS} from pm where src='alpaca' order by random() limit ?", (n,)).fetchall()
    rows += con.execute(f"select {COLS} from pm where src='sip_store' order by random() limit ?", (n,)).fetchall()
    rows += con.execute(f"select {COLS} from pm where src='none' order by random() limit ?", (n,)).fetchall()
    bad = 0
    for sym, day, dn, dv, dd, dh, dl, dla, dvw, src in rows:
        d = datetime.strptime(day, '%Y-%m-%d')
        s = datetime(d.year, d.month, d.day, 4, 0, tzinfo=ET).astimezone(timezone.utc)
        e = datetime(d.year, d.month, d.day, 9, 29, 59, tzinfo=ET).astimezone(timezone.utc)
        req = StockBarsRequest(symbol_or_symbols=[sym], timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                               start=s, end=e, feed=DataFeed.SIP, adjustment=Adjustment.RAW)
        bars = cl._to_dict(cl.data_client.get_stock_bars(req)).get(sym) or []
        bars = [b for b in bars
                if 4 <= b.timestamp.astimezone(ET).hour
                and (b.timestamp.astimezone(ET).hour, b.timestamp.astimezone(ET).minute) < (9, 30)]
        N = len(bars)
        V = sum(b.volume for b in bars)
        D = sum(b.close * b.volume for b in bars)
        H = max((b.high for b in bars), default=None)
        L = min((b.low for b in bars), default=None)
        LA = bars[-1].close if bars else None
        VW = (D / V) if V else None
        if N == 0:
            ok = (dn == 0 and dv is None and dd is None)
        else:
            ok = (N == dn and abs(V - dv) < 1e-6 and abs(D - dd) < 1e-3
                  and abs(H - dh) < 1e-9 and abs(L - dl) < 1e-9
                  and abs(LA - dla) < 1e-9 and abs(VW - dvw) < 1e-9)
        bad += (not ok)
        verdict = 'MATCH' if ok else 'MISMATCH'
        print(f'{sym:6s} {day} {src:9s} db=({dn}, {dv}, {dd}, {dh}, {dl}, {dla}) '
              f'fresh=({N}, {V}, {D}, {H}, {L}, {LA})  {verdict}', flush=True)
    print(f'{len(rows) - bad}/{len(rows)} rows reproduce exactly')
    return bad == 0


if __name__ == '__main__':
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 3) else 1)
