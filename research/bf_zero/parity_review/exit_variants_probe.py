#!/usr/bin/env python3
"""Parity review probe (read-only): re-walk a random sample of the spec's own trades (research/bf_zero/spec_trades.csv)
on the study bars with the exit mechanics the BROKER actually provides, to put a sign and a size on the structural
deviations between trading/hod_break.py::walk_exit and the live bracket legs:

  V0  spec           : from entry_idx+1; stop on low<=stop -> min(stop,open)*0.999; target on CLOSE>=target at target
  V1  wick target    : same, but target fills on HIGH>=target (a resting limit leg fills on any trade through it)
  V2  wick + bar0    : V1 and the legs are live inside the ENTRY bar too (walk from entry_idx; the spec skips it)
  V3  bar0 only      : spec target (close) but stop/target live inside the entry bar

Usage: ulimit -v 2000000 && python3 research/bf_zero/parity_review/exit_variants_probe.py [n_days] [seed]
"""
import os, sys, sqlite3, random
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import STOP_FILL_SLIP

N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
FLAT = 955

def walk(o, h, l, c, m, k0, stop, target, wick, flat=FLAT):
    for k in range(k0, len(o)):
        if int(m[k]) >= flat: return k, float(o[k]), 'eod'
        if l[k] <= stop: return k, float(min(stop, o[k]) * (1 - STOP_FILL_SLIP)), 'stop'
        if (h[k] if wick else c[k]) >= target: return k, float(target), 'target'
    return len(o) - 1, float(c[-1]), 'eod'

def minute_of(t):
    s = str(t)
    if len(s) <= 5 and ':' in s:
        hh, mm = s.split(':')[:2]; return int(hh) * 60 + int(mm)
    ts = pd.Timestamp(s)
    if ts.tzinfo is None: ts = ts.tz_localize('UTC')
    ts = ts.tz_convert('America/New_York'); return ts.hour * 60 + ts.minute

tr = pd.read_csv('research/bf_zero/spec_trades.csv', dtype={'symbol': str}, keep_default_na=False)
tr = tr[tr.entry >= 20.0]                                   # the live book's price floor (config min_price 20)
days = sorted(tr.day.unique()); random.seed(SEED); sample = sorted(random.sample(days, min(N_DAYS, len(days))))
def ro(path): return sqlite3.connect(f'file:{path}?mode=ro', uri=True)
CACHE = next((p for p in ('data/cache.db', 'cache.db') if os.path.exists(p)), None)
SIDE = [ro(p) for p in ('research/ignition_capcheck/topup.db', 'data/research/databento/pit_bars_1min.db', 'research/bf_zero/bars.db') if os.path.exists(p)]
cache = ro(CACHE) if CACHE else None

def load_day(day, syms):
    """same source chain as research/bf_zero/build_candidates.load_bars: cache.db intraday_bars_1min, then the side DBs"""
    parts = []
    if cache is not None:
        q = f"select symbol, timestamp as t, open as o, high as h, low as l, close as c from intraday_bars_1min where bar_date=? and symbol in ({','.join('?' * len(syms))})"
        parts.append(pd.DataFrame(cache.execute(q, (day, *syms)).fetchall(), columns=['symbol', 't', 'o', 'h', 'l', 'c']))
    for con in SIDE:
        have = set(pd.concat(parts).symbol) if parts else set(); left = [s for s in syms if s not in have]
        if not left: break
        q = f"select symbol, t, o, h, l, c from bars where day=? and symbol in ({','.join('?' * len(left))})"
        parts.append(pd.DataFrame(con.execute(q, (day, *left)).fetchall(), columns=['symbol', 't', 'o', 'h', 'l', 'c']))
    return pd.concat(parts) if parts else pd.DataFrame(columns=['symbol', 't', 'o', 'h', 'l', 'c'])

rows = []
for day in sample:
    sub = tr[tr.day == day]
    syms = sub.symbol.tolist()
    df = load_day(day, syms)
    if df.empty: continue
    df['m'] = df.t.map(minute_of)
    for r in sub.itertuples():
        g = df[(df.symbol == r.symbol) & (df.m >= 570) & (df.m < 960)].sort_values('m')
        if g.empty: continue
        o, h, l, c = (g[k].values.astype(float) for k in ('o', 'h', 'l', 'c')); m = g.m.values.astype(int)
        idx = np.flatnonzero(m == r.entry_m)
        if not len(idx): continue
        e = int(idx[0]); R = r.entry - r.stop
        if R <= 0: continue
        out = {'day': day, 'symbol': r.symbol, 'spec_rr_csv': r.rr, 'why_csv': r.why}
        for name, k0, wick in (('V0', e + 1, False), ('V1', e + 1, True), ('V2', e, True), ('V3', e, False)):
            k, px, why = walk(o, h, l, c, m, k0, r.stop, r.target, wick)
            out[f'{name}_rr'] = (px - r.entry) / R; out[f'{name}_why'] = why; out[f'{name}_exit_m'] = int(m[k])
        rows.append(out)
res = pd.DataFrame(rows)
print(f'sample: {len(sample)} days, {len(res)} spec trades (entry >= $20)')
print(f"V0 reproduces the CSV: max |V0_rr - csv rr| = {(res.V0_rr - res.spec_rr_csv).abs().max():.4f}")
for v, label in (('V0', 'spec (close target, from entry+1)'), ('V1', 'wick target'), ('V2', 'wick target + legs live in entry bar'), ('V3', 'close target + legs live in entry bar')):
    why = res[f'{v}_why'].value_counts().to_dict()
    print(f"{v} {label:42s} mean R {res[f'{v}_rr'].mean():+.4f}  target {why.get('target',0):4d}  stop {why.get('stop',0):4d}  eod {why.get('eod',0):4d}")
d = res.V1_rr - res.V0_rr
print(f"\nV1-V0 (wick vs close target): mean {d.mean():+.4f} R/trade; trades changed {int((d != 0).sum())} of {len(res)}; "
      f"better live {int((d > 0).sum())}, worse live {int((d < 0).sum())}")
chg = res[(res.V0_why != 'target') & (res.V1_why == 'target')]
print(f"  wick-touch trades the spec did NOT exit at target: {len(chg)} — of which spec later stopped {int((chg.V0_why == 'stop').sum())}, eod {int((chg.V0_why == 'eod').sum())}")
d2 = res.V3_rr - res.V0_rr
print(f"V3-V0 (legs live inside the entry bar): mean {d2.mean():+.4f} R/trade; trades changed {int((d2 != 0).sum())}; "
      f"stopped IN the entry bar {int(((res.V3_why == 'stop') & (res.V3_exit_m == tr.set_index(['day','symbol']).loc[list(zip(res.day, res.symbol))].entry_m.values)).sum())}; "
      f"target IN the entry bar {int(((res.V3_why == 'target') & (res.V3_exit_m == tr.set_index(['day','symbol']).loc[list(zip(res.day, res.symbol))].entry_m.values)).sum())}")
d3 = res.V2_rr - res.V0_rr
print(f"V2-V0 (both, = the live bracket): mean {d3.mean():+.4f} R/trade")
res.to_csv('research/bf_zero/parity_review/exit_variants_probe_rows.csv', index=False)
