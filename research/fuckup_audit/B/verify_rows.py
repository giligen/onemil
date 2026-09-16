#!/usr/bin/env python3
"""Hand-verification of candidates4 rows: take N random rows, reload the bars for those symbol-days straight from the
stores, and recompute the signal bar, the two fills and the exit walks with a PLAIN PYTHON LOOP (never the builder's
vectorised helpers) from the row's own (sig_m, level, stop). Catches vectorisation errors that parity against
candidates3 cannot (candidates3 shares only 5 of the 14 families and only the next-open fill).
Usage: ulimit -v 1500000; nice -n 15 python3 research/fuckup_audit/B/verify_rows.py [N] [--src FILE]"""
import os, sqlite3, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
ARGS = sys.argv[1:]
SRC = ARGS[ARGS.index('--src') + 1] if '--src' in ARGS else 'research/fuckup_audit/B/candidates4.csv'
N = int(ARGS[0]) if ARGS and not ARGS[0].startswith('--') else 5
EOD_M, OPEN_M, CAP = 955, 570, 0.006
cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
sip = sqlite3.connect(f'file:{ROOT}/research/bf_zero/bars_sip.db?mode=ro', uri=True, timeout=120)


def bars(day, sym):
    q = ("select timestamp as t, open as o, high as h, low as l, close as c, volume as v from intraday_bars_1min "
         "where bar_date=? and symbol=?")
    g = pd.read_sql(q, cache, params=[day, sym])
    if not len(g):
        g = pd.read_sql("select t, o, h, l, c, v from bars where day=? and symbol=?", sip, params=[day, sym])
    ts = pd.to_datetime(g.t, utc=True, format='mixed').dt.tz_convert('America/New_York')
    g = g.assign(m=(ts.dt.hour * 60 + ts.dt.minute).values).sort_values('m').drop_duplicates('m')
    pm = g[g.m < OPEN_M]
    g = g[(g.m >= OPEN_M) & (g.m < 960)].reset_index(drop=True)
    return g, pm


def walk(o, h, l, c, m, k0, stop, target):
    """the builder's exit convention, written as the plain loop it is specified as"""
    for k in range(k0, len(o)):
        if m[k] >= EOD_M: return float(o[k]), 'eod', int(m[k]), k
        if l[k] <= stop: return float(min(stop, o[k]) * 0.999), 'stop', int(m[k]), k
        if target is not None and c[k] >= target: return float(target), 'target', int(m[k]), k
    return float(c[-1]), 'eod', int(m[-1]), len(o) - 1


_RNG = np.random.default_rng(7)          # ONE generator — a fresh default_rng(7) per row would be constant
d = pd.read_csv(SRC, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str}, keep_default_na=False, na_values=[''],
                low_memory=False, skiprows=lambda i: i > 0 and _RNG.random() > 0.004)
print(f'sampled pool {len(d):,} rows from {SRC}', flush=True)
d = d.sample(N, random_state=7)
bad = 0
for r in d.itertuples():
    g, pm = bars(r.day, r.symbol)
    o, h, l, c, v = (g[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v'))
    m = g.m.values.astype(int)
    i = int(np.flatnonzero(m == r.sig_m)[0])
    chk = {'sig_o': (o[i], r.sig_o), 'sig_h': (h[i], r.sig_h), 'sig_l': (l[i], r.sig_l), 'sig_c': (c[i], r.sig_c),
           'sig_v': (v[i], r.sig_v),
           'range_so_far_pct': ((max(h[:i]) - min(l[:i])) / o[0] * 100, r.range_so_far_pct),
           'cum_dollar_vol': (float((c[:i + 1] * v[:i + 1]).sum()), r.cum_dollar_vol),
           'n_touches': (int(sum(1 for k in range(i) if r.level * 0.998 <= h[k] < r.level)), r.n_touches),
           'close_confirm': (int(c[i] >= r.level), r.close_confirm)}
    if i + 1 < len(o) and o[i + 1] <= r.level * (1 + CAP) and r.stop < o[i + 1]:
        e = float(o[i + 1]); R = e - r.stop
        px, why, xm, _ = walk(o, h, l, c, m, i + 2, r.stop, e + 2 * R)
        chk |= {'next_entry': (e, r.next_entry), 'next_r_pct': (R / e * 100, r.next_r_pct),
                'next_rr_2r': ((px - e) / R, r.next_rr_2r), 'next_exit_m_2r': (xm, r.next_exit_m_2r)}
        chk['next_why_2r'] = (why, r.next_why_2r)
        px, why, xm, _ = walk(o, h, l, c, m, i + 2, r.stop, None)
        chk['next_rr_hold'] = ((px - e) / R, r.next_rr_hold)
        s1 = r.stop - 0.01 * r.price; R1 = e - s1
        px, why, xm, _ = walk(o, h, l, c, m, i + 2, s1, e + 2 * R1)
        chk['next_rr_2r_stopm1'] = ((px - e) / R1, r.next_rr_2r_stopm1)
    else:
        chk['next_entry_isna'] = (True, bool(pd.isna(r.next_entry)))
    if r.fam not in ('F11', 'F12', 'F13'):
        e = max(float(o[i]), float(r.level))
        if e <= r.level * (1 + CAP) and r.stop < e:
            R = e - r.stop
            px, why, xm, _ = walk(o, h, l, c, m, i + 1, r.stop, e + 2 * R)
            chk |= {'rest_entry': (e, r.rest_entry), 'rest_rr_2r': ((px - e) / R, r.rest_rr_2r),
                    'rest_exit_m_2r': (xm, r.rest_exit_m_2r),
                    'rest_queue_ok': (int(v[i] >= 5.0 * (100.0 / R)), r.rest_queue_ok)}
    print(f'--- {r.day} {r.symbol} {r.fam} {r.cfg} sig_m={r.sig_m} level={r.level} stop={r.stop}')
    for k, (mine, theirs) in chk.items():
        ok = (mine == theirs) if isinstance(mine, (str, bool)) else (
            abs(float(mine) - float(theirs)) <= 1e-6 * max(1.0, abs(float(mine))))
        if not ok: bad += 1
        print(f'    {"OK " if ok else "BAD"} {k}: recomputed {mine!r} vs file {theirs!r}')
print(f'\n{"ALL MATCH" if bad == 0 else str(bad) + " MISMATCHES"}')
