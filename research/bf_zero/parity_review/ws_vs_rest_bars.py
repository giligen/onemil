#!/usr/bin/env python3
"""Parity probe: Alpaca SIP websocket 1-min bars (+ updated bars) vs SIP REST historical 1-min bars, same symbols,
same minutes. Read-only; writes only into WS_PROBE_OUT.

Env:
  WS_PROBE_START=HH:MM     ET start of the window; waits until then if in the future (default: next minute boundary)
  WS_PROBE_MINUTES=6       window length in whole minutes
  WS_PROBE_SYMBOLS=...     comma list, or a file path (one symbol per line); default: first WS_PROBE_N (300) lines of the
                           newest logs/hod_stream_universe_*.txt
  WS_PROBE_OUT=dir         output dir for ws_bars.csv / updated_bars.csv / rest_bars.csv / summary.txt
                           (default research/bf_zero/parity_review/ws_probe_<ET date>)
Records every `bars` message and every `updatedBars` message with wall-clock arrival; 60 s after the last minute closed
fetches the same window via REST (StockBarsRequest 1-Min SIP) and diffs per (symbol, minute). Caps: 1 GB address
space (RLIMIT_AS) and a 9-minute wall limit counted from the window start (the pre-start wait is not counted).
Scheduled use: `WS_PROBE_START=09:30 python3 research/bf_zero/parity_review/ws_vs_rest_bars.py` at 09:29 ET.
"""
import os, sys, csv, glob, time, signal, resource, threading, statistics
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = '/home/ec2-user/onemil'; sys.path.insert(0, ROOT); os.chdir(ROOT)
ET = ZoneInfo('America/New_York')
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))          # 1 GB cap
T0 = time.time()

def log(msg):
    print(f'[{time.time() - T0:6.1f}s] {msg}', flush=True)

def hard_cap(signum, frame):
    log('HARD CAP 9 min hit — exiting'); os._exit(3)
signal.signal(signal.SIGALRM, hard_cap)

N_MIN = int(os.environ.get('WS_PROBE_MINUTES', '6'))
N_SYM = int(os.environ.get('WS_PROBE_N', '300'))
spec = os.environ.get('WS_PROBE_SYMBOLS', '').strip()
if spec and os.path.sep in spec:
    syms = [l.strip() for l in open(spec) if l.strip()][:N_SYM]
elif spec:
    syms = [s.strip().upper() for s in spec.split(',') if s.strip()]
else:
    import re
    files = sorted(f for f in glob.glob(f'{ROOT}/logs/hod_stream_universe_????-??-??.txt')
                   if re.search(r'_\d{4}-\d{2}-\d{2}\.txt$', f) and os.path.getsize(f) > 100)   # dated + non-trivial (a stray *_None.txt exists)
    if not files: log('no dated logs/hod_stream_universe_YYYY-MM-DD.txt — set WS_PROBE_SYMBOLS'); sys.exit(2)
    syms = [l.strip() for l in open(files[-1]) if l.strip()][:N_SYM]; log(f'symbols from {os.path.basename(files[-1])}')
    if len(syms) < 10: log(f'only {len(syms)} symbols in {files[-1]} — set WS_PROBE_SYMBOLS'); sys.exit(2)
now_et = datetime.now(ET)
OUT = os.environ.get('WS_PROBE_OUT') or f'{ROOT}/research/bf_zero/parity_review/ws_probe_{now_et:%Y-%m-%d}'
os.makedirs(OUT, exist_ok=True)

from config import Config
cfg = Config(); KEY, SECRET = cfg.alpaca_api_key, cfg.alpaca_api_secret
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# liquidity tag from daily_bars ADV (read-only), median split
import sqlite3
adv = {}
try:
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=10)
    q = ("with d as (select symbol, volume, row_number() over (partition by symbol order by bar_date desc) rn from daily_bars "
         f"where symbol in ({','.join('?' * len(syms))})) select symbol, avg(volume) from d where rn <= 20 group by symbol")
    adv = {s: float(a or 0) for s, a in con.execute(q, syms)}; con.close()
except Exception as e:
    log(f'ADV lookup failed: {e}')
med = statistics.median([adv.get(s, 0) for s in syms]) if adv else 0
liquid = {s for s in syms if adv.get(s, 0) >= med}
log(f'{len(syms)} symbols, ADV median {med:,.0f}: {len(liquid)} liquid / {len(syms) - len(liquid)} thin; out={OUT}')

# window
start_env = os.environ.get('WS_PROBE_START', '').strip()
if start_env:
    hh, mm = (int(x) for x in start_env.split(':'))
    win_start = now_et.replace(hour=hh, minute=mm, second=0, microsecond=0).astimezone(timezone.utc)
    if win_start < datetime.now(timezone.utc) - timedelta(seconds=5):
        log(f'WS_PROBE_START {start_env} ET is in the past — using the next minute boundary instead'); start_env = ''
if not start_env:
    now = datetime.now(timezone.utc); win_start = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    if (win_start - now).total_seconds() < 8: win_start += timedelta(minutes=1)
win_end = win_start + timedelta(minutes=N_MIN)
log(f'window {win_start.astimezone(ET):%H:%M:%S} .. {win_end.astimezone(ET):%H:%M:%S} ET ({N_MIN} min)')

rows, upd = [], []
lock = threading.Lock()

async def on_bar(bar):
    arr = time.time()
    with lock:
        rows.append(dict(symbol=bar.symbol, ts=bar.timestamp, open=float(bar.open), high=float(bar.high), low=float(bar.low), close=float(bar.close),
                         volume=float(bar.volume), trade_count=bar.trade_count, vwap=bar.vwap, arrival=arr))

async def on_updated(bar):
    arr = time.time()
    with lock:
        upd.append(dict(symbol=bar.symbol, ts=bar.timestamp, open=float(bar.open), high=float(bar.high), low=float(bar.low), close=float(bar.close),
                        volume=float(bar.volume), trade_count=bar.trade_count, vwap=bar.vwap, arrival=arr))

stream = StockDataStream(KEY, SECRET, feed=DataFeed.SIP)
stream.subscribe_bars(on_bar, *syms)
stream.subscribe_updated_bars(on_updated, *syms)
# connect ~45 s before the window (the SDK needs a few seconds; subscribing early only adds pre-window rows we drop)
while time.time() < win_start.timestamp() - 45:
    time.sleep(1)
th = threading.Thread(target=stream.run, daemon=True); th.start()
log('stream thread started (bars + updatedBars)')
while time.time() < win_start.timestamp():
    time.sleep(0.2)
signal.alarm(9 * 60 - 5)                                    # 9-minute wall limit from the window start
log('window open')
last_print = None
while time.time() < win_end.timestamp() + 25:              # last bar (minute win_end-1) gets 25 s to arrive
    time.sleep(1)
    cur = int(time.time() // 60)
    if cur != last_print:
        last_print = cur
        with lock: log(f'bars so far {len(rows)} (updated {len(upd)})')
with lock:
    n25 = sum(1 for r in rows if win_start <= r['ts'] < win_end)
while time.time() < win_end.timestamp() + 60:              # REST 60 s after the last minute closed
    time.sleep(1)
with lock:
    ws_all, upd_all = list(rows), list(upd)
ws = [r for r in ws_all if win_start <= r['ts'] < win_end]
uw = [r for r in upd_all if win_start <= r['ts'] < win_end]
log(f'ws collection done: {len(ws_all)} bars total, {len(ws)} in window (late +25..+60 s: {len(ws) - n25}); updated bars {len(upd_all)} total, {len(uw)} in window')

hist = StockHistoricalDataClient(KEY, SECRET)
t_rest = time.time()
data = hist.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=win_start,
                                            end=win_end - timedelta(seconds=1), feed=DataFeed.SIP)).data
rest = [dict(symbol=s, ts=b.timestamp, open=float(b.open), high=float(b.high), low=float(b.low), close=float(b.close), volume=float(b.volume),
             trade_count=b.trade_count, vwap=b.vwap) for s, lst in data.items() for b in lst if win_start <= b.timestamp < win_end]
log(f'REST fetched {len(rest)} bars for {len(data)} symbols in {time.time() - t_rest:.1f}s')
try: stream.stop()
except Exception as e: log(f'stream.stop: {e}')

for name, rs in (('ws_bars.csv', ws_all), ('updated_bars.csv', upd_all), ('rest_bars.csv', rest)):
    with open(f'{OUT}/{name}', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0].keys()) if rs else ['symbol']); w.writeheader(); w.writerows(rs)

FIELDS = ('open', 'high', 'low', 'close', 'volume', 'trade_count')
def same(a, b): return all(a[f] == b[f] for f in FIELDS)
W = {(r['symbol'], r['ts']): r for r in ws}; R = {(r['symbol'], r['ts']): r for r in rest}
U = {}
for r in uw: U[(r['symbol'], r['ts'])] = r                  # last update wins
both = sorted(set(W) & set(R)); only_ws = sorted(set(W) - set(R)); only_rest = sorted(set(R) - set(W))
exact = 0; mism = {f: 0 for f in FIELDS}; examples = []
for k in both:
    d = [f for f in FIELDS if W[k][f] != R[k][f]]
    if not d: exact += 1; continue
    for f in d: mism[f] += 1
    if len(examples) < 40:
        a, b = W[k], R[k]
        examples.append(f"{k[0]} {k[1].astimezone(ET):%H:%M} ws={a['open']}/{a['high']}/{a['low']}/{a['close']}/{a['volume']:.0f}/{a['trade_count']} "
                        f"rest={b['open']}/{b['high']}/{b['low']}/{b['close']}/{b['volume']:.0f}/{b['trade_count']} diff={d}"
                        + (f" updated={U[k]['open']}/{U[k]['high']}/{U[k]['low']}/{U[k]['close']}/{U[k]['volume']:.0f}" if k in U else ''))
# updated bars: does the final REST bar equal the (last) updated bar? did the update change anything vs the original?
u_eq_rest = sum(1 for k in U if k in R and same(U[k], R[k])); u_ne_rest = sum(1 for k in U if k in R and not same(U[k], R[k]))
u_changed = sum(1 for k in U if k in W and not same(U[k], W[k])); u_no_orig = sum(1 for k in U if k not in W)
mism_keys = {k for k in both if not same(W[k], R[k])}
mism_with_update = sum(1 for k in mism_keys if k in U)
lat = sorted((r['arrival'] - (r['ts'] + timedelta(minutes=1)).timestamp()) for r in ws)
def pct(p): return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else float('nan')
ulat = sorted((r['arrival'] - (r['ts'] + timedelta(minutes=1)).timestamp()) for r in uw)
per_min = [sum(1 for r in ws if r['ts'] == win_start + timedelta(minutes=i)) for i in range(N_MIN)]
per_min_r = [sum(1 for r in rest if r['ts'] == win_start + timedelta(minutes=i)) for i in range(N_MIN)]
liq = lambda ks: sum(1 for k in ks if k[0] in liquid)
lines = [
    f'window {win_start.astimezone(ET):%Y-%m-%d %H:%M} .. {win_end.astimezone(ET):%H:%M} ET | symbols {len(syms)} (liquid {len(liquid)}, thin {len(syms) - len(liquid)})',
    f'ws bars in window {len(ws)} (dup keys {len(ws) - len(W)}) | rest bars {len(rest)} | per-minute ws {per_min} rest {per_min_r}',
    f'both {len(both)} | exact {exact} ({exact / max(1, len(both)):.1%}) | mismatched {len(both) - exact} (liquid {liq(mism_keys)}) | field mismatches {mism}',
    f'only ws {len(only_ws)} (liquid {liq(only_ws)}) | only rest {len(only_rest)} (liquid {liq(only_rest)})',
    f'arrival latency of bars (s after bar END): n={len(lat)} min {lat[0] if lat else 0:.2f} p50 {pct(.5):.2f} p90 {pct(.9):.2f} p99 {pct(.99):.2f} max {lat[-1] if lat else 0:.2f}; late +25..+60 s: {len(ws) - n25}',
    f'UPDATED bars in window: {len(uw)} messages on {len(U)} (symbol,minute) keys | final REST == last updated bar: {u_eq_rest}, != {u_ne_rest} | '
    f'update differs from the original bar: {u_changed}, update without an original bar: {u_no_orig} | ws-vs-rest mismatches that had an update: {mism_with_update} of {len(mism_keys)} | '
    f'update arrival after bar END: {"n/a" if not ulat else f"min {ulat[0]:.1f} p50 {ulat[len(ulat) // 2]:.1f} max {ulat[-1]:.1f} s"}',
    'only-ws examples: ' + ', '.join(f'{k[0]}@{k[1].astimezone(ET):%H:%M}' for k in only_ws[:25]),
    'only-rest examples: ' + ', '.join(f'{k[0]}@{k[1].astimezone(ET):%H:%M}(v={R[k]["volume"]:.0f},n={R[k]["trade_count"]})' for k in only_rest[:25]),
    'mismatch examples:', *examples,
]
open(f'{OUT}/summary.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines), flush=True)
log('DONE'); os._exit(0)
