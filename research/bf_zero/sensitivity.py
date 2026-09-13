#!/usr/bin/env python3
"""bf_zero — pre-registered sensitivities on the surviving configs (DESIGN.md: never selected on).

For the given family-configs (rows already in candidates_full.csv, causal floor applied), reload the
bars and re-simulate the fixed +2R/-1R exit (E1) under:
  slip 0.3% / target trade-through x1.002   (the base, must reproduce rr_e1)
  slip 0.6% / target trade-through x1.002   (double entry slippage; the fill needs bar high >= level x 1.006)
  slip 0.3% / target on bar CLOSE >= target (no wick fills at all)
  slip 0.6% / target on bar CLOSE           (both)
Usage: BFZ_KEYS='F5|{"K": 5, "X": 0.04};F5|{"K": 10, "X": 0.04}' python3 sensitivity.py
"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, f'{ROOT}/research/bf_zero')
import build_candidates as B          # module-level loads (universe, daily, spy) — main() not run
D = 'research/bf_zero'
KEYS = [k.split('|') for k in os.environ.get('BFZ_KEYS', 'F5|{"K": 5, "X": 0.04};F5|{"K": 10, "X": 0.04}').split(';')]

c = pd.read_csv(f'{D}/candidates_full.csv', usecols=['day', 'symbol', 'fam', 'cfg', 'entry_m', 'entry', 'stop', 'dist_open_pct', 'rr_e1'],
                dtype={'symbol': str, 'fam': 'category', 'cfg': 'category'}, keep_default_na=False, na_values=[''])
for k in ('entry_m', 'entry', 'stop', 'dist_open_pct', 'rr_e1'): c[k] = pd.to_numeric(c[k], errors='coerce')
sel = pd.concat([c[(c.fam == f) & (c.cfg == g)] for f, g in KEYS])
sel = sel[sel.dist_open_pct >= 5].copy()          # the causal floor used in scoring
print('rows', len(sel), 'symbol-days', sel[['day', 'symbol']].drop_duplicates().shape[0], flush=True)


def e1(o, h, l, m, i, entry, stop, close_fill):
    Rd = entry - stop; tgt = entry + 2 * Rd
    oo, hh, ll, cc, mm = o[i + 1:], h[i + 1:], l[i + 1:], B_close[i + 1:], m[i + 1:]
    if len(oo) == 0: return 0.0
    eod = B.first_true(mm >= B.EOD_M); eod = eod if eod is not None else len(oo) - 1
    s_idx = B.first_true(ll <= stop)
    t_idx = B.first_true(cc >= tgt) if close_fill else B.first_true(hh >= tgt * 1.002)
    cand = [(k, w) for k, w in ((s_idx, 'stop'), (t_idx, 'target'), (eod, 'eod')) if k is not None]
    k, w = min(cand, key=lambda x: (x[0], 0 if x[1] == 'stop' else 1))
    px = min(stop, oo[k]) * 0.999 if w == 'stop' else (tgt if w == 'target' else oo[k])
    return (px - entry) / Rd


rows = []
for n, (day, sub) in enumerate(sel.groupby('day')):
    bars = B.load_bars(day, sub.symbol.tolist())
    for r in sub.itertuples():
        gg = bars.get(r.symbol)
        if gg is None: continue
        rth = gg[(gg.m >= B.OPEN_M) & (gg.m < 960)].reset_index(drop=True)
        o, h, l = rth.o.values.astype(float), rth.h.values.astype(float), rth.l.values.astype(float); m = rth.m.values.astype(int)
        B_close = rth.c.values.astype(float)
        idx = np.flatnonzero(m == int(r.entry_m))
        if not len(idx): continue
        i = int(idx[0]); level = r.entry / (1 + B.SLIP); out = dict(day=day, symbol=r.symbol, fam=r.fam, cfg=r.cfg, rr_e1_stored=r.rr_e1)
        for slip, cf, name in ((0.003, False, 'base'), (0.006, False, 'slip6'), (0.003, True, 'close'), (0.006, True, 'slip6_close')):
            entry = level * (1 + slip)
            if h[i] < entry: out[name] = np.nan; continue          # no fill at the higher slipped level
            stop = r.stop
            if stop >= entry: out[name] = np.nan; continue
            out[name] = e1(o, h, l, m, i, entry, stop, cf)
        rows.append(out)
    if n % 50 == 0: print(f'{n} days, {len(rows)} rows', flush=True)
S = pd.DataFrame(rows); S.to_csv(f'{D}/sensitivity_rows.csv', index=False)
S['split'] = np.where(S.day < '2026-01-01', 'TRAIN', np.where(S.day < '2026-06-01', 'VAL', 'TEST'))
print('\nbase reproduces stored rr_e1: max abs diff', float((S.base - S.rr_e1_stored).abs().max()))
for (f, g), d in S.groupby(['fam', 'cfg'], observed=True):
    print(f'\n## {f} {g}')
    t = d.groupby('split').agg(n=('base', 'size'), base=('base', 'mean'), slip6=('slip6', 'mean'), close=('close', 'mean'), slip6_close=('slip6_close', 'mean'),
                               fill_rate_slip6=('slip6', lambda s: s.notna().mean())).round(3).reindex(['TRAIN', 'VAL', 'TEST'])
    print(t.to_string())
print('DONE', flush=True)
