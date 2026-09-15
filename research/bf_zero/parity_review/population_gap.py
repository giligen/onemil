#!/usr/bin/env python3
"""Study-population gap on real days: the spec (`detect`, h >= HOD exactly) vs the candidate filter that decided which
symbol-days spec_sim.py simulated (fam_hod: h >= HOD x 1.003, plus len(rth) >= 30 in build_candidates and >= 10 in
spec_sim). Read-only over the study's bar stores. Usage: population_gap.py DAY [DAY ...]"""
import os, sqlite3, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import HodBreakParams, detect, simulate
D = 'research/bf_zero'; P = HodBreakParams(); OPEN_M = 570; SLIP = 0.003
uni = pd.read_csv(f'{D}/universe.csv', dtype={'symbol': str}, keep_default_na=False)
for c_ in ('open', 'high', 'low', 'close', 'adv20'): uni[c_] = pd.to_numeric(uni[c_], errors='coerce')
cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
SIDE = [sqlite3.connect(f'file:{p}?mode=ro', uri=True, timeout=120) for p in
        (f'{ROOT}/research/ignition_capcheck/topup.db', f'{ROOT}/data/research/databento/pit_bars_1min.db', f'{D}/bars.db') if os.path.exists(p)]


def load_bars(day, syms):
    out = {}
    q = f"select symbol, timestamp as t, open as o, high as h, low as l, close as c, volume as v from intraday_bars_1min where bar_date=? and symbol in ({','.join('?' * len(syms))})"
    for s, gg in pd.read_sql(q, cache, params=[day] + list(syms)).groupby('symbol'): out[s] = gg
    for con in SIDE:
        left = [s for s in syms if s not in out]
        if not left: break
        t = pd.read_sql("select symbol, t, o, h, l, c, v from bars where day=?", con, params=[day])
        for s, gg in t[t.symbol.isin(left)].groupby('symbol'): out[s] = gg
    res = {}
    for s, gg in out.items():
        ts = pd.to_datetime(gg.t, utc=True).dt.tz_convert('America/New_York')
        gg = gg.assign(m=(ts.dt.hour * 60 + ts.dt.minute).values).sort_values('m').drop_duplicates('m')
        res[s] = gg[['m', 'o', 'h', 'l', 'c', 'v']].reset_index(drop=True)
    return res


def fam_hod_slip(h, l, K=5, X=0.04):
    n = len(h); hod = np.maximum.accumulate(h); lo = pd.Series(l).rolling(K, min_periods=K).min().values
    t = np.arange(n); j = np.clip(t - 1, 0, n - 1)
    ok = (t > K) & (lo[j] >= hod[j] * (1 - X)) & (h >= hod[j] * (1 + SLIP)) & (lo[j] < hod[j])
    idx = np.flatnonzero(ok); return int(idx[0]) if len(idx) else None


tot = dict(days=0, symdays=0, spec_trades=0, not_in_study_pop=0, short_day=0, range_def=0)
for day in sys.argv[1:]:
    sub = uni[uni.bar_date == day]; sub = sub[sub.high >= sub.open * (1 + P.min_dist_open_pct / 100)]
    bars = load_bars(day, sub.symbol.tolist()); tot['days'] += 1
    for r in sub.itertuples():
        gg = bars.get(r.symbol)
        if gg is None: continue
        rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
        if len(rth) < 7: continue
        tot['symdays'] += 1
        o, h, l, cl, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v')); m = rth.m.values.astype(int)
        t = simulate(o, h, l, cl, v, m, r.adv20, P)
        if t is None: continue
        tot['spec_trades'] += 1
        fh = fam_hod_slip(h, l)
        if fh is None:
            tot['not_in_study_pop'] += 1; print(f'  {day} {r.symbol}: spec trade (entry {t.entry:.2f}, rr {t.rr:+.2f}) but NO fam_hod break at +0.3% -> never in candidates_full/spec_sim')
        elif len(rth) < 30:
            tot['short_day'] += 1; print(f'  {day} {r.symbol}: spec trade on a {len(rth)}-bar day (< 30 bars: build_candidates skipped it)')
print(tot)
