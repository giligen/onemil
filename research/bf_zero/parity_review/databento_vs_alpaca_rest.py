#!/usr/bin/env python3
"""(A) Cross-source: Databento EQUS.MINI ohlcv-1m (the spec's side-DB source, same call as
research/ignition_capcheck/fetch_missing_databento.py) vs Alpaca REST 1-Min SIP (cache.db's source AND the live
backfill) for ONE full prior session on the probe's 300 symbols. Per (symbol, minute) in 09:30-15:59 ET: exact /
o,h,l,c,v mismatches / one-sided bars. (B) Census of the spec's executable book (spec_book.csv): which source served
each traded symbol-day, whether its RTH bars start at 09:30, and Databento-only symbol names. Read-only; the
Databento call costs ~$0.0004 per symbol-day (cost printed and capped at $1)."""
import os, sys, sqlite3, csv
from datetime import datetime, timedelta, timezone
import pandas as pd
ROOT = '/home/ec2-user/onemil'; sys.path.insert(0, ROOT); os.chdir(ROOT)
OUT = f'{ROOT}/research/bf_zero/parity_review'
DAY = os.environ.get('PROBE_DAY', '2026-09-14')
from dotenv import load_dotenv; load_dotenv(f'{ROOT}/.env')
from config import Config
cfg = Config()
syms = [l.strip() for l in open(f'{ROOT}/logs/hod_stream_universe_2026-09-15.txt') if l.strip()][:300]
import pytz
ET = pytz.timezone('America/New_York')
d = datetime.strptime(DAY, '%Y-%m-%d')
o_utc = ET.localize(d.replace(hour=9, minute=30)).astimezone(timezone.utc); c_utc = ET.localize(d.replace(hour=15, minute=59)).astimezone(timezone.utc)

PART = os.environ.get('PART', 'AB')
if 'A' in PART:
  pass
# ---- Alpaca REST
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
hist = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
data = hist.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=o_utc, end=c_utc, feed=DataFeed.SIP)).data
A = {}
for s, lst in data.items():
    for b in lst:
        if o_utc <= b.timestamp <= c_utc: A[(s, b.timestamp)] = (float(b.open), float(b.high), float(b.low), float(b.close), float(b.volume), b.trade_count)
print(f'Alpaca REST: {len(A)} RTH bars for {len(data)} symbols on {DAY}', flush=True)

# ---- Databento
import databento as dbn
client = dbn.Historical(os.environ['DATABENTO_API_KEY'])
nxt = (d + timedelta(days=1)).strftime('%Y-%m-%d')
cost = client.metadata.get_cost(dataset='EQUS.MINI', schema='ohlcv-1m', symbols=syms, stype_in='raw_symbol', start=DAY, end=nxt)
print(f'Databento cost estimate ${cost:.4f}', flush=True)
if cost > 1.0: print('cost cap exceeded — abort'); sys.exit(2)
st = client.timeseries.get_range(dataset='EQUS.MINI', schema='ohlcv-1m', symbols=syms, stype_in='raw_symbol', start=DAY, end=nxt)
df = st.to_df().reset_index()
ts = pd.to_datetime(df['ts_event'], utc=True)
D = {}
for sym, t, o, h, l, c, v in zip(df.symbol, ts, df.open, df.high, df.low, df.close, df.volume):
    t = t.to_pydatetime()
    if o_utc <= t <= c_utc: D[(sym, t)] = (float(o), float(h), float(l), float(c), float(v), None)
print(f'Databento EQUS.MINI: {len(D)} RTH bars for {df.symbol.nunique()} symbols (all-session rows {len(df)})', flush=True)

both = sorted(set(A) & set(D)); onlyA = sorted(set(A) - set(D)); onlyD = sorted(set(D) - set(A))
names = ('open', 'high', 'low', 'close', 'volume'); mism = {k: 0 for k in names}; exact = 0; ex = []
hi = {'D>A': 0, 'A>D': 0}; lo = {'D<A': 0, 'A<D': 0}; vol = {'D>A': 0, 'A>D': 0}; hi_cents = []
for k in both:
    a, b = A[k], D[k]; dd = [n for n, i in zip(names, range(5)) if a[i] != b[i]]
    if not dd: exact += 1; continue
    for n in dd: mism[n] += 1
    if a[1] != b[1]: hi['D>A' if b[1] > a[1] else 'A>D'] += 1; hi_cents.append(round((b[1] - a[1]) * 100, 2))
    if a[2] != b[2]: lo['D<A' if b[2] < a[2] else 'A<D'] += 1
    if a[4] != b[4]: vol['D>A' if b[4] > a[4] else 'A>D'] += 1
    if len(ex) < 30: ex.append(f'{k[0]} {k[1].astimezone(ET):%H:%M} alpaca={a[:5]} databento={b[:5]} diff={dd}')
sym_any_hilo = len({k[0] for k in both if A[k][1] != D[k][1] or A[k][2] != D[k][2]})
sym_both = len({k[0] for k in both})
# per-symbol-day: would the running HOD differ at any minute?
hod_diff_days = 0
for s in {k[0] for k in both}:
    ks = sorted(k for k in both if k[0] == s); ha = hd = 0.0; diff = False
    for k in ks:
        ha = max(ha, A[k][1]); hd = max(hd, D[k][1])
        if ha != hd: diff = True; break
    hod_diff_days += diff
lines = [f'=== {DAY} Alpaca REST SIP vs Databento EQUS.MINI ohlcv-1m, {len(syms)} symbols, RTH 09:30-15:59',
         f'both {len(both)} | exact {exact} ({exact / max(1, len(both)):.1%}) | field mismatches {mism}',
         f'  high: {hi} (cents, first 40: {hi_cents[:40]}) | low: {lo} | volume: {vol}',
         f'  symbols with any high/low mismatch: {sym_any_hilo} of {sym_both}; symbol-days whose RUNNING HOD differs at some minute: {hod_diff_days}',
         f'only Alpaca {len(onlyA)} (e.g. {[(k[0], k[1].astimezone(ET).strftime("%H:%M"), A[k][4]) for k in onlyA[:12]]})',
         f'only Databento {len(onlyD)} (e.g. {[(k[0], k[1].astimezone(ET).strftime("%H:%M"), D[k][4]) for k in onlyD[:12]]})',
         'examples:', *ex]
print('\n'.join(lines), flush=True)

# ---- (B) spec book census
bk = pd.read_csv(f'{ROOT}/research/bf_zero/spec_book.csv', dtype={'symbol': str})
cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=60)
sides = [(p.split('/')[-2] + '/' + p.split('/')[-1], sqlite3.connect(f'file:{p}?mode=ro', uri=True, timeout=60)) for p in
         (f'{ROOT}/research/ignition_capcheck/topup.db', f'{ROOT}/data/research/databento/pit_bars_1min.db', f'{ROOT}/research/bf_zero/bars.db') if os.path.exists(p)]
src_n = {}; no930 = {}; rows = []
for r in bk.itertuples():
    src = None; first = None; n = 0
    q = cache.execute("select min(timestamp), count(*) from intraday_bars_1min where symbol=? and bar_date=? and timestamp >= ?", (r.symbol, r.day, r.day + 'T00:00:00')).fetchone()
    if q and q[1]:
        src = 'cache.db(Alpaca)'; n = q[1]
        first = cache.execute("select min(timestamp) from intraday_bars_1min where symbol=? and bar_date=? and time(timestamp) >= time(?)", (r.symbol, r.day, o_utc.strftime('%H:%M:%S'))).fetchone()[0]
    else:
        for name, con in sides:
            q = con.execute("select min(t), count(*) from bars where symbol=? and day=?", (r.symbol, r.day)).fetchone()
            if q and q[1]: src = name; n = q[1]; first = q[0]; break
    fm = None
    if first:
        t = pd.Timestamp(first)
        t = t.tz_localize('UTC') if t.tzinfo is None else t
        et = t.tz_convert('America/New_York'); fm = et.hour * 60 + et.minute
    src_n[src] = src_n.get(src, 0) + 1
    if fm is not None and fm != 570: no930[src] = no930.get(src, 0) + 1
    rows.append(dict(day=r.day, symbol=r.symbol, split=r.split, source=src, n_bars=n, first_minute=fm, entry_m=r.entry_m, rr=r.rr))
pd.DataFrame(rows).to_csv(f'{OUT}/spec_book_sources.csv', index=False)
hy = bk[bk.symbol.str.contains('-', regex=False)]
lines2 = [f'=== spec book census ({len(bk)} trades): source counts {src_n}; traded symbol-days whose first RTH bar is NOT 09:30: {no930}',
          f'  first-minute distribution (non-09:30): {pd.Series([x["first_minute"] for x in rows if x["first_minute"] not in (None, 570)]).value_counts().head(8).to_dict()}',
          f'  Databento-naming symbols (with "-") in the book: {len(hy)} trades, e.g. {hy.symbol.unique()[:10].tolist()}; split counts {bk.split.value_counts().to_dict()}']
print('\n'.join(lines2), flush=True)
open(f'{OUT}/databento_vs_alpaca_summary.txt', 'w').write('\n'.join(lines + lines2) + '\n')
