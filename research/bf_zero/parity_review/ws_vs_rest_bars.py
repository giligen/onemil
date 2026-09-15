#!/usr/bin/env python3
"""Parity probe: Alpaca SIP websocket 1-min bars vs SIP REST historical 1-min bars, same symbols, same minutes.

Subscribes StockDataStream(feed=SIP) to the first 300 names of today's HOD stream universe, records every
bar (symbol, ts, o,h,l,c,v, trade_count, vwap) plus wall-clock arrival for 6 full minutes, then 60 s after
the last minute closed fetches the same window via REST (StockBarsRequest 1-Min SIP) and diffs per
(symbol, minute). Read-only; writes only under research/bf_zero/parity_review/. Hard cap 9 min wall.
"""
import os, sys, csv, time, signal, threading, statistics
from datetime import datetime, timedelta, timezone

ROOT = '/home/ec2-user/onemil'; sys.path.insert(0, ROOT); os.chdir(ROOT)
OUT = f'{ROOT}/research/bf_zero/parity_review'
N_MIN = int(os.environ.get('PROBE_MINUTES', '6'))
T0 = time.time()

def log(msg):
    print(f'[{time.time() - T0:6.1f}s] {msg}', flush=True)

def hard_cap(signum, frame):
    log('HARD CAP 9 min hit — exiting'); os._exit(3)
signal.signal(signal.SIGALRM, hard_cap); signal.alarm(9 * 60 - 5)

from config import Config
cfg = Config()
KEY, SECRET = cfg.alpaca_api_key, cfg.alpaca_api_secret
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

syms = [l.strip() for l in open(f'{ROOT}/logs/hod_stream_universe_2026-09-15.txt') if l.strip()][:300]
# liquidity tag from daily_bars ADV (read-only), median split
import sqlite3
adv = {}
try:
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=30)
    q = ("with d as (select symbol, volume, row_number() over (partition by symbol order by bar_date desc) rn from daily_bars "
         f"where symbol in ({','.join('?' * len(syms))})) select symbol, avg(volume) from d where rn <= 20 group by symbol")
    adv = {s: float(a or 0) for s, a in con.execute(q, syms)}; con.close()
except Exception as e:
    log(f'ADV lookup failed: {e}')
med = statistics.median([adv.get(s, 0) for s in syms]) if adv else 0
liquid = {s for s in syms if adv.get(s, 0) >= med}
log(f'{len(syms)} symbols, ADV median {med:,.0f}: {len(liquid)} liquid / {len(syms) - len(liquid)} thin')

ws_rows = []            # dicts
ws_lock = threading.Lock()
n_total = [0]

async def on_bar(bar):
    arr = time.time()
    with ws_lock:
        n_total[0] += 1
        ws_rows.append(dict(symbol=bar.symbol, ts=bar.timestamp, open=float(bar.open), high=float(bar.high), low=float(bar.low),
                            close=float(bar.close), volume=float(bar.volume), trade_count=bar.trade_count, vwap=bar.vwap, arrival=arr))

stream = StockDataStream(KEY, SECRET, feed=DataFeed.SIP)
stream.subscribe_bars(on_bar, *syms)
th = threading.Thread(target=stream.run, daemon=True); th.start()
log('stream thread started; waiting for the next minute boundary')

now = datetime.now(timezone.utc)
win_start = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
if (win_start - now).total_seconds() < 8:                # too close: bars for this minute may be partial if the ws is not yet up
    win_start += timedelta(minutes=1)
win_end = win_start + timedelta(minutes=N_MIN)          # exclusive; minutes win_start .. win_end-1min
log(f'window {win_start:%H:%M:%S}Z .. {win_end:%H:%M:%S}Z ({N_MIN} minutes)')
# record until the last bar (minute win_end-1) has had 25 s to arrive
while time.time() < win_end.timestamp() + 25:
    time.sleep(1)
    if int(time.time()) % 60 == 0:
        with ws_lock: log(f'ws bars so far {n_total[0]}')
with ws_lock:
    ws = [r for r in ws_rows if win_start <= r['ts'] < win_end]
    ws_all = list(ws_rows)
log(f'ws collection done: {len(ws_all)} bars total, {len(ws)} inside the window; arrival tail for the window bars still open')
# wait until last minute closed + 60s
while time.time() < win_end.timestamp() + 60:
    time.sleep(1)
with ws_lock:
    late_extra = [r for r in ws_rows if win_start <= r['ts'] < win_end and r not in ws]
    ws = [r for r in ws_rows if win_start <= r['ts'] < win_end]
log(f'late arrivals between +25s and +60s: {len(late_extra)}')

# REST
hist = StockHistoricalDataClient(KEY, SECRET)
req = StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=win_start,
                       end=win_end - timedelta(seconds=1), feed=DataFeed.SIP)
t_rest = time.time()
bars = hist.get_stock_bars(req)
data = bars.data if hasattr(bars, 'data') else bars
rest = []
for s, lst in data.items():
    for b in lst:
        rest.append(dict(symbol=s, ts=b.timestamp, open=float(b.open), high=float(b.high), low=float(b.low), close=float(b.close),
                         volume=float(b.volume), trade_count=b.trade_count, vwap=b.vwap))
log(f'REST fetched {len(rest)} bars for {len(data)} symbols in {time.time() - t_rest:.1f}s')
rest = [r for r in rest if win_start <= r['ts'] < win_end]

try: stream.stop()
except Exception as e: log(f'stream.stop: {e}')

# write raw
for name, rows in (('ws_bars.csv', ws_all), ('rest_bars.csv', rest)):
    with open(f'{OUT}/{name}', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['symbol']); w.writeheader(); w.writerows(rows)

# diff
W = {(r['symbol'], r['ts']): r for r in ws}
R = {(r['symbol'], r['ts']): r for r in rest}
dup_ws = len(ws) - len(W)
both = sorted(set(W) & set(R)); only_ws = sorted(set(W) - set(R)); only_rest = sorted(set(R) - set(W))
exact = 0; mism = dict(open=0, high=0, low=0, close=0, volume=0, trade_count=0); examples = []
vol_ws_gt = vol_rest_gt = 0; hi_rest_gt = hi_ws_gt = 0; lo_rest_lt = lo_ws_lt = 0
for k in both:
    a, b = W[k], R[k]; diff = [f for f in mism if a[f] != b[f]]
    if not diff: exact += 1; continue
    for f in diff: mism[f] += 1
    if a['volume'] != b['volume']: vol_ws_gt += a['volume'] > b['volume']; vol_rest_gt += b['volume'] > a['volume']
    if a['high'] != b['high']: hi_rest_gt += b['high'] > a['high']; hi_ws_gt += a['high'] > b['high']
    if a['low'] != b['low']: lo_rest_lt += b['low'] < a['low']; lo_ws_lt += a['low'] < b['low']
    if len(examples) < 40:
        examples.append(f"{k[0]} {k[1]:%H:%M}Z ws o/h/l/c/v/n={a['open']}/{a['high']}/{a['low']}/{a['close']}/{a['volume']:.0f}/{a['trade_count']} "
                        f"rest={b['open']}/{b['high']}/{b['low']}/{b['close']}/{b['volume']:.0f}/{b['trade_count']} diff={diff}")
lat = sorted((r['arrival'] - (r['ts'] + timedelta(minutes=1)).timestamp()) for r in ws)
def pct(p): return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else float('nan')
per_min = {}
for r in ws: per_min[r['ts']] = per_min.get(r['ts'], 0) + 1
per_min_r = {}
for r in rest: per_min_r[r['ts']] = per_min_r.get(r['ts'], 0) + 1
liq_ws = sum(1 for r in ws if r['symbol'] in liquid); liq_rest = sum(1 for r in rest if r['symbol'] in liquid)
only_ws_liq = sum(1 for k in only_ws if k[0] in liquid); only_rest_liq = sum(1 for k in only_rest if k[0] in liquid)
mism_liq = sum(1 for k in both if k[0] in liquid and any(W[k][f] != R[k][f] for f in mism))
lines = [
    f'window {win_start:%Y-%m-%d %H:%M}Z .. {win_end:%H:%M}Z  symbols {len(syms)} (liquid {len(liquid)}, thin {len(syms) - len(liquid)})',
    f'ws bars in window {len(ws)} (dup keys {dup_ws}; liquid {liq_ws}) | rest bars {len(rest)} (liquid {liq_rest})',
    f'per-minute ws {[per_min.get(win_start + timedelta(minutes=i), 0) for i in range(N_MIN)]} rest {[per_min_r.get(win_start + timedelta(minutes=i), 0) for i in range(N_MIN)]}',
    f'both {len(both)} | exact {exact} ({exact / max(1, len(both)):.1%}) | mismatched {len(both) - exact} (liquid {mism_liq})',
    f'field mismatches: {mism}',
    f'  volume: ws>rest {vol_ws_gt}, rest>ws {vol_rest_gt} | high: rest>ws {hi_rest_gt}, ws>rest {hi_ws_gt} | low: rest<ws {lo_rest_lt}, ws<rest {lo_ws_lt}',
    f'only ws {len(only_ws)} (liquid {only_ws_liq}) | only rest {len(only_rest)} (liquid {only_rest_liq})',
    f'arrival latency (s after bar END): n={len(lat)} min {lat[0] if lat else 0:.2f} p50 {pct(.5):.2f} p90 {pct(.9):.2f} p99 {pct(.99):.2f} max {lat[-1] if lat else 0:.2f}',
    f'late arrivals (+25s..+60s after last minute end): {len(late_extra)}',
    'only-ws examples: ' + ', '.join(f'{k[0]}@{k[1]:%H:%M}' for k in only_ws[:25]),
    'only-rest examples: ' + ', '.join(f'{k[0]}@{k[1]:%H:%M}(v={R[k]["volume"]:.0f},n={R[k]["trade_count"]})' for k in only_rest[:25]),
    'mismatch examples:', *examples,
]
open(f'{OUT}/summary.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines), flush=True)
log('DONE'); os._exit(0)
