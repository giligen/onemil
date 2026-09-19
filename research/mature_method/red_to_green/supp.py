#!/usr/bin/env python3
"""Supplementary: (a) a SECOND reproduction gate for scan rule S1 against the reconciliation's
independent L2 implementation; (b) the cost-model sensitivity, including an entry leg set from the
MEASURED ask-minus-fill rather than the contract's 0.25 x half; (c) TRAIN/VAL weekly and monthly
dollar paths for the S1 book; (d) the one authorised TEST read for the cells that passed BOTH claim
gates (FREEZE.md).  Run after score.py."""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book
OUT = 'research/mature_method/red_to_green'; RISK = 100.0
RD = lambda p, **k: pd.read_csv(p, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''], **k)
log = lambda *a: (print(*a), sys.stdout.flush())
L = []
say = lambda *a: (L.append(' '.join(str(x) for x in a)), log(*a))

PB = [(0, 5, '$1-5'), (5, 10, '$5-10'), (10, 20, '$10-20'), (20, 50, '$20-50'), (50, 200, '$50-200'), (200, 1e9, '$200+')]
HB = [(0, 575, '09:30-09:35'), (576, 600, '09:35-10:00'), (601, 660, '10:00-11:00'),
      (661, 780, '11:00-13:00'), (781, 9999, '13:00+')]
pband = lambda p: next(n for lo, hi, n in PB if lo < p <= hi)
hband = lambda m: next(n for lo, hi, n in HB if lo <= m <= hi)

d = RD(f'{OUT}/cands.csv')
d['split'] = np.where(d.day < '2026-01-01', 'TRAIN', np.where(d.day < '2026-06-01', 'VAL', 'TEST'))
d['adv20'] = d.adv20.fillna(0.0)
d['wk'] = pd.to_datetime(d.day).dt.to_period('W-FRI').astype(str); d['mo'] = d.day.str[:7]
d['pb'] = [pband(p) for p in d.entry]; d['hb'] = [hband(m) for m in d.entry_m]
_VS = d.variants.str.split('|')
S1 = d.index[['lvl1003_f5_S1' in s for s in _VS]]; S2 = d.index[['lvl1003_f5_S2' in s for s in _VS]]
del _VS

pop = RD(f'{OUT}/pop.csv', usecols=['day'])
dw = pd.DataFrame({'day': sorted(pop.day.unique())})
dw['split'] = np.where(dw.day < '2026-01-01', 'TRAIN', np.where(dw.day < '2026-06-01', 'VAL', 'TEST'))
dw['wk'] = pd.to_datetime(dw.day).dt.to_period('W-FRI').astype(str); dw['mo'] = dw.day.str[:7]
WEEKS = {s: sorted(dw[dw.split == s].wk.unique()) for s in ('TRAIN', 'VAL', 'TEST')}
MONTHS = {s: sorted(dw[dw.split == s].mo.unique()) for s in ('TRAIN', 'VAL', 'TEST')}

# ---------------------------------------------------------------- (a) reproduction gate for S1
say('## (a) REPRODUCTION GATE 2 - scan rule S1 vs the reconciliation ladder step L2 (an independent code path)')
l2 = RD('research/fuckup_audit/H/F6_reconcile/ladder_L2_engine_cut.csv')
mine = d.loc[S1]
mine = mine[(mine.pdr >= 8) & (mine.floor_val >= 5) & (mine.entry >= 5) & (mine.r_pct >= 1)
            & (mine.sig_m <= 840) & (mine.over_cap_bps <= 0)]
say('L2 candidates %d   this pass, L2 gates, honest population %d' % (len(l2), len(mine)))
kl, km = set(zip(l2.day, l2.symbol)), set(zip(mine.day, mine.symbol))
say('shared %d   L2-only %d   mine-only %d' % (len(kl & km), len(kl - km), len(km - kl)))
j = l2.merge(mine, on=['day', 'symbol'], suffixes=('_l', '_m'))
for f in ('sig_m', 'entry_m', 'entry', 'stop'):
    dd = (j[f + '_l'] - j[f + '_m']).abs()
    say('  |d %-8s| max %.6g  n differing %d of %d' % (f, dd.max(), int((dd > 1e-6).sum()), len(j)))
dd = (j['r2_grossR_l'] - j['r2_grossR_m']).abs()
say('  |d 2R gross R| max %.6g  n differing %d' % (dd.max(), int((dd > 1e-5).sum())))
say('  L2 applies 14:01 to the FILL bar and a $5 floor on the ENTRY; this pass applies 14:00 to the SIGNAL bar.')

# ---------------------------------------------------------------- (b) cost models
say('\n## (b) cost models - the leg weights are the thing the verdict turns on')
q = RD(f'{OUT}/nbbo_ok.csv')
dd2 = d.loc[S1].drop_duplicates(['day', 'symbol', 'entry_m'])[['day', 'symbol', 'entry_m', 'pdr', 'level', 'adv20', 'r_pct', 'sig_m']]
jq = q.merge(dd2, on=['day', 'symbol', 'entry_m'], how='left', suffixes=('', '_c'))
b0q = jq[(jq.pdr >= 8) & (jq.level >= 5) & (jq.adv20.fillna(0) >= 1e5) & (jq.r_pct_c >= 1) & (jq.sig_m <= 840)]
ratio = float((b0q.direct_bps / (b0q.full_bps / 2.0)).median())
say('measured, B0-eligible quotes n=%d: full spread median %.1f bps, direct (ask - modelled open fill) median %.1f bps'
    % (len(b0q), b0q.full_bps.median(), b0q.direct_bps.median()))
say('  -> the ENTRY leg actually costs %.2f x half-spread; contract (c) charges 0.25 x half (%.1fx too little)'
    % (ratio, ratio / 0.25))
say('  share of B0-eligible sampled signals whose quoted spread exceeds the live max_spread_bps 300 gate: %.1f%%'
    % (100.0 * (b0q.full_bps > 300).mean()))

mc = pd.read_csv(f'{OUT}/cost_curve_measured.csv')
MEAS = {(r.pb, r.hb): float(r.bps_median) for r in mc.itertuples()}
MFALL = float(np.median(list(MEAS.values())))
b = RD('research/lit_review_2026/cost_curve.csv')
b = b[(b.n_q > 0) & b.spread.notna() & (b.price > 0)].copy(); b['bps'] = b.spread / b.price * 1e4
BAND = {k: float(v) for k, v in b.groupby(['pb', 'hb']).bps.median().items()}; BFALL = float(np.median(list(BAND.values())))
del b
d['half_meas'] = 0.5 * (np.array([MEAS.get((p, h), MFALL) for p, h in zip(d.pb, d.hb)]) / 100.0) / np.clip(d.r_pct, 0.05, None)
d['half_band'] = 0.5 * (np.array([BAND.get((p, h), BFALL) for p, h in zip(d.pb, d.hb)]) / 100.0) / np.clip(d.r_pct, 0.05, None)
legs = [dict((t, 0.0) for t in ('stop', 'eod', 'target')) for _ in range(1)]


def legcost(mode, K):
    return np.array([sum(float(w) * K[t] for w, t in (s.split(':') for s in str(lg).split(';')))
                     for lg in d[f'{mode}_legs']])


MODELS = {
    'M1band': ('half_band', 0.25, {'stop': .875, 'eod': .412, 'target': .875}),
    'M2meas': ('half_meas', 0.25, {'stop': .875, 'eod': .412, 'target': .875}),
    'M3entry': ('half_meas', ratio, {'stop': .875, 'eod': .412, 'target': .875}),
    'M4live': ('half_meas', ratio, {'stop': 1.0, 'eod': 0.5, 'target': 0.0}),
    'M5cons': ('half_meas', 1.0, {'stop': 1.0, 'eod': 1.0, 'target': 1.0}),
}
say('  M1band = cost_curve.csv band, contract (c) legs (entry 0.25xhalf, stop/target 0.875, eod 0.412)')
say('  M2meas = measured NBBO curve, contract (c) legs           M3entry = M2 with the entry leg at the MEASURED %.2f x half' % ratio)
say('  M4live = M3 with the venue legs: TP is a resting limit (0), stop crosses (1.0), 15:55 flat (0.5)')
say('  M5cons = conservative bound: a full half-spread on BOTH legs')
for name, (hc, ew, K) in MODELS.items():
    for mode in ('r2', 'hold'):
        d[f'net_{mode}_{name}'] = d[f'{mode}_grossR'].values - ew * d[hc].values - legcost(mode, K) * d[hc].values


def book(ix, cell, split, costname, mode='r2'):
    x = d.loc[ix]
    x = x[(x.split == split) & (x.floor_val >= 5) & (x.prev_close >= 5) & (x.level >= 5) & (x.r_pct >= 1)
          & (x.sig_m <= 840) & (x.adv20 >= 1e5) & (x.over_cap_bps <= 0)]
    if cell == 'P4': x = x[x.pdr >= 12]
    elif cell == 'G1c': x = x[(x.pdr >= 8) & (x.next_clock == 1)]
    else: x = x[x.pdr >= 8]
    if len(x) == 0:
        return pd.DataFrame(columns=['day', 'entry_m', 'exit_m', 'symbol', 'net', 'gross', 'wk', 'mo'])
    col = f'net_{mode}_{costname}'
    rr = [(r.day, int(r.entry_m), int(getattr(r, f'{mode}_exit_m')), r.symbol, getattr(r, col),
           getattr(r, f'{mode}_grossR'), r.wk, r.mo) for r in x.itertuples()]
    return pd.DataFrame(run_book(rr, 12, 4), columns=['day', 'entry_m', 'exit_m', 'symbol', 'net', 'gross', 'wk', 'mo'])


def line(t, sp):
    if len(t) == 0: return dict(n=0)
    v = t.net.values; se = v.std(ddof=1) / np.sqrt(len(v))
    w = t.groupby('wk').net.sum().reindex(WEEKS[sp]).fillna(0.0) * RISK
    mo = t.groupby('mo').net.sum().reindex(MONTHS[sp]).fillna(0.0) * RISK
    return dict(n=len(v), grossR=round(t.gross.mean(), 4), netR=round(v.mean(), 4), t=round(v.mean() / se, 2),
                green=round((w > 0).mean() * 100, 1), tot=round(w.sum(), 0), worst=round(w.min(), 0),
                mdd=round(float((w.cumsum() - w.cumsum().cummax()).min()), 0), greenmo=round((mo > 0).mean() * 100, 1))


rows = []
for name in MODELS:
    for scan, ix in (('S1', S1), ('S2', S2)):
        for sp in ('TRAIN', 'VAL'):
            rows.append(dict(cost=name, scan=scan, split=sp, **line(book(ix, 'B0', sp, name), sp)))
cs = pd.DataFrame(rows); cs.to_csv(f'{OUT}/cost_models.csv', index=False)
say(cs.to_string(index=False))

# ---------------------------------------------------------------- (c) weekly + monthly dollar path, S1 B0
say('\n## (c) the S1 B0 dollar path at the live $100 risk (measured NBBO, contract c)')
for sp in ('TRAIN', 'VAL'):
    t = book(S1, 'B0', sp, 'M2meas')
    w = t.groupby('wk').net.sum().reindex(WEEKS[sp]).fillna(0.0) * RISK
    c = t.groupby('wk').size().reindex(WEEKS[sp]).fillna(0).astype(int)
    say(f'{sp} weekly $ (n): ' + ' '.join('%+d(%d)' % (a, b) for a, b in zip(w.values, c.values)))
    mo = t.groupby('mo').net.sum().reindex(MONTHS[sp]).fillna(0.0) * RISK
    say(f'{sp} monthly $: ' + ' | '.join('%s %+d' % (i, v) for i, v in mo.items()))
    say('  green %d/%d weeks, %d/%d months, total %+d, mean month %+d'
        % ((w > 0).sum(), len(w), (mo > 0).sum(), len(mo), w.sum(), mo.mean()))

# ---------------------------------------------------------------- (d) the authorised TEST read
say('\n## (d) TEST - opened ONLY for the two cells that passed BOTH claim gates (FREEZE.md)')
rows = []
for cell in ('P4', 'G1c'):
    for sp in ('TRAIN', 'VAL', 'TEST'):
        t = book(S1, cell, sp, 'M2meas')
        r = dict(cell=cell, scan='S1', split=sp, **line(t, sp))
        if len(t):
            w = t.groupby('wk').net.sum().reindex(WEEKS[sp]).fillna(0.0) * RISK
            r['tpw'] = round(len(t) / len(WEEKS[sp]), 1)
        rows.append(r)
te = pd.DataFrame(rows); te.to_csv(f'{OUT}/test_read.csv', index=False)
say(te.to_string(index=False))

open(f'{OUT}/supp.log', 'w').write('\n'.join(L) + '\n')
