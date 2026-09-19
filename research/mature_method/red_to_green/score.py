#!/usr/bin/env python3
"""Steps 2 / 4 / 5 / 6 / 7 / 8 / 9 - every declared cell of PREREG section 7, for BOTH scan rules.

Reads `cands.csv` (the one tape walk) and `cost_curve_measured.csv` (the measured NBBO), writes
`cells.csv`, `gatemap.csv`, `nulls.csv`, `weeks_*.csv` and `score.log`.

TEST is SEALED: nothing here reads a day >= 2026-06-01 unless `--open-test CELL` is passed, which also
appends to FREEZE.md.
"""
import os, sys, time
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book

OUT = 'research/mature_method/red_to_green'
RISK = 100.0
SPLITS = ('TRAIN', 'VAL')
log = lambda *a: (print(*a), sys.stdout.flush())

PB = [(0, 5, '$1-5'), (5, 10, '$5-10'), (10, 20, '$10-20'), (20, 50, '$20-50'), (50, 200, '$50-200'), (200, 1e9, '$200+')]
HB = [(0, 575, '09:30-09:35'), (576, 600, '09:35-10:00'), (601, 660, '10:00-11:00'),
      (661, 780, '11:00-13:00'), (781, 9999, '13:00+')]
pband = lambda p: next(n for lo, hi, n in PB if lo < p <= hi)
hband = lambda m: next(n for lo, hi, n in HB if lo <= m <= hi)
LEGK = {'stop': 0.875, 'eod': 0.412, 'target': 0.875}

# ---------------------------------------------------------------- load
d = pd.read_csv(f'{OUT}/cands.csv', dtype={'symbol': str, 'day': str, 'variants': str},
                keep_default_na=False, na_values=[''])
d['split'] = np.where(d.day < '2026-01-01', 'TRAIN', np.where(d.day < '2026-06-01', 'VAL', 'TEST'))
d = d[d.split != 'TEST'].reset_index(drop=True)                    # FREEZE
d['adv20'] = d.adv20.fillna(0.0)
d['wk'] = pd.to_datetime(d.day).dt.to_period('W-FRI').astype(str)
d['mo'] = d.day.str[:7]
d['pb'] = [pband(p) for p in d.entry]; d['hb'] = [hband(m) for m in d.entry_m]
log('candidate rows (TRAIN+VAL):', len(d))

pop = pd.read_csv(f'{OUT}/pop.csv', usecols=['day'], dtype={'day': str})
days = pd.Series(sorted(pop.day.unique()))
days = days[days < '2026-06-01']
dw = pd.DataFrame({'day': days})
dw['split'] = np.where(dw.day < '2026-01-01', 'TRAIN', 'VAL')
dw['wk'] = pd.to_datetime(dw.day).dt.to_period('W-FRI').astype(str)
dw['mo'] = dw.day.str[:7]
WEEKS = {s: sorted(dw[dw.split == s].wk.unique()) for s in SPLITS}
MONTHS = {s: sorted(dw[dw.split == s].mo.unique()) for s in SPLITS}
log('market weeks:', {s: len(WEEKS[s]) for s in SPLITS})

# ---------------------------------------------------------------- measured cost
mc = pd.read_csv(f'{OUT}/cost_curve_measured.csv')
MEAS = {(r.pb, r.hb): float(r.bps_median) for r in mc.itertuples()}
MEAS_ALL = float(np.median([v for v in MEAS.values()]))
b = pd.read_csv('research/lit_review_2026/cost_curve.csv', dtype={'symbol': str, 'day': str},
                keep_default_na=False, na_values=[''])
b = b[(b.n_q > 0) & b.spread.notna() & (b.price > 0)].copy(); b['bps'] = b.spread / b.price * 1e4
BAND = {k: float(v) for k, v in b.groupby(['pb', 'hb']).bps.median().items()}
del b


def attach_cost(x, curve, fallback):
    bps = np.array([curve.get((p, h), fallback) for p, h in zip(x.pb, x.hb)], dtype=float)
    half = 0.5 * (bps / 100.0) / np.clip(x.r_pct.values, 0.05, None)
    return bps, half


d['bps_meas'], d['half_meas'] = attach_cost(d, MEAS, MEAS_ALL)
d['bps_band'], d['half_band'] = attach_cost(d, BAND, float(np.median(list(BAND.values()))))
for mode in ('hold', 'r2', 'partial'):
    for tag, half in (('meas', d.half_meas.values), ('band', d.half_band.values)):
        lc = np.array([sum(float(w) * LEGK[t] for w, t in (s.split(':') for s in str(lg).split(';')))
                       for lg in d[f'{mode}_legs']])
        d[f'net_{mode}_{tag}'] = d[f'{mode}_grossR'].values - 0.25 * half - lc * half
    # conservative: full half-spread both legs
    d[f'net_{mode}_cons'] = d[f'{mode}_grossR'].values - 2.0 * d.half_meas.values
d['filled'] = d.over_cap_bps <= 0.0


# ---------------------------------------------------------------- cells
VARLABS = ['lvl1003_f5_S1', 'lvl1003_f5_S2', 'lvl1003_f8_S2', 'lvl1003_f10_S2',
           'lvl1000_f5_S1', 'lvl1000_f5_S2', 'lvl1006_f5_S1', 'lvl1006_f5_S2']
_VS = d.variants.str.split('|')
VARIX = {v: d.index[[v in s for s in _VS]] for v in VARLABS}
del _VS
log('rows per variant:', {v: len(VARIX[v]) for v in VARLABS})


def sel(var):
    return d.loc[VARIX[var]]


BASE = dict(pdr=8.0, floor=5.0, price=5.0, r=1.0, late=840, adv=1e5, prevclose=5.0,
            perday=12, conc=4, lvl='lvl1003', fl='f5', rv=False, clock=False)

CELLS = {
    'B0': {},
    'P1': dict(pdr=6.0), 'P3': dict(pdr=10.0), 'P4': dict(pdr=12.0),
    'F2': dict(floor=8.0, fl='f8'), 'F3': dict(floor=10.0, fl='f10'),
    'X2': dict(price=10.0), 'X3': dict(price=20.0),
    'R2': dict(r=2.0), 'R3': dict(r=3.0),
    'T1': dict(late=660), 'T2': dict(late=780), 'T4': dict(late=930),
    'A2': dict(adv=5e5), 'A3': dict(adv=0.0),
    'L2': dict(lvl='lvl1000'), 'L3': dict(lvl='lvl1006'),
    'K2': dict(perday=20, conc=8),
    'V1': dict(rv=True), 'G1c': dict(clock=True),
    'C1': dict(pdr=10.0, price=10.0), 'C2': dict(pdr=12.0, r=2.0),
    'C3': dict(floor=8.0, fl='f8', late=780),
    'C6': dict(rv=True, clock=True),
    'C4': dict(pdr=0.0, adv=0.0, late=930, perday=20, conc=8),
    'C5': dict(pdr=0.0, adv=0.0, late=930, perday=20, conc=8, price=1.0, prevclose=1.0),
}


def trades(cell, scan, mode='r2', cost='meas', split=None, extra=None):
    """The booked trade set for one cell x scan rule."""
    c = dict(BASE); c.update(CELLS[cell]); c.update(extra or {})
    # S1's signal bar is the first level break and does NOT move with the floor, so a raised floor is a
    # downstream test on floor_val; S2's scan skips floor-failing bars, so it needs its own walk.
    var = f"{c['lvl']}_{c['fl'] if scan == 'S2' else 'f5'}_{scan}"
    x = sel(var)
    if split:
        x = x[x.split == split]
    x = x[(x.pdr >= c['pdr']) & (x.floor_val >= c['floor']) & (x.level >= c['price'])
          & (x.prev_close >= c['prevclose']) & (x.r_pct >= c['r']) & (x.sig_m <= c['late'])
          & (x.adv20 >= c['adv']) & x.filled]
    if c['rv']:
        x = x[(x.rv >= 1.0) & (x.rv < 5.0)]
    if c['clock']:
        x = x[x.next_clock == 1]
    if len(x) == 0:
        return x.assign(net=[], gross=[])
    x = x.assign(net=x[f'net_{mode}_{cost}'].values, gross=x[f'{mode}_grossR'].values,
                 exit_m=x[f'{mode}_exit_m'].values, why=x[f'{mode}_exit_type'].values)
    rr = [(r.day, int(r.entry_m), int(r.exit_m), r.symbol, r.net, r.gross, r.wk, r.mo, r.why,
           r.entry, r.stop, r.R, r.r_pct, int(r.sig_m), r.pdr, r.rv, r.level, r.adv20)
          for r in x.itertuples()]
    t = pd.DataFrame(run_book(rr, c['perday'], c['conc']),
                     columns=['day', 'entry_m', 'exit_m', 'symbol', 'net', 'gross', 'wk', 'mo', 'why',
                              'entry', 'stop', 'R', 'r_pct', 'sig_m', 'pdr', 'rv', 'level', 'adv20'])
    return t


def streak(w):
    best = cur = 0
    for v in w:
        cur = cur + 1 if v <= 0 else 0
        best = max(best, cur)
    return best


def stats(t, sp):
    n = len(t)
    if n == 0:
        return dict(n=0, tpw=0.0, grossR=np.nan, netR=np.nan, t=np.nan, WR=np.nan, green=np.nan,
                    redstreak=np.nan, worst=np.nan, tot=0.0, mdd=0.0, greenmo=np.nan, ex1=np.nan, ex5=np.nan)
    v = t.net.values
    se = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    w = t.groupby('wk').net.sum().reindex(WEEKS[sp]).fillna(0.0) * RISK
    mo = t.groupby('mo').net.sum().reindex(MONTHS[sp]).fillna(0.0) * RISK
    s = np.sort(v)
    return dict(n=n, tpw=round(n / len(WEEKS[sp]), 1), grossR=round(t.gross.mean(), 4),
                netR=round(v.mean(), 4), t=round(v.mean() / se, 2) if se and se == se else np.nan,
                WR=round((v > 0).mean() * 100, 1), green=round((w > 0).mean() * 100, 1),
                redstreak=streak(w.values), worst=round(w.min(), 0), tot=round(w.sum(), 0),
                mdd=round(float((w.cumsum() - w.cumsum().cummax()).min()), 0),
                greenmo=round((mo > 0).mean() * 100, 1),
                ex1=round(s[:max(int(round(n * 0.99)), 1)].mean(), 4),
                ex5=round(s[:max(int(round(n * 0.95)), 1)].mean(), 4))


def mde(v, power=0.80):
    """Smallest per-trade mean detectable at alpha .05 two-sided, 80% power."""
    n = len(v)
    return 2.80 * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan


# ---------------------------------------------------------------- step 2: gross, no cost
L = []
L.append('## STEP 2 - GROSS, no cost at all (exit = the engine 2R bracket)')
rows = []
for scan in ('S1', 'S2'):
    for sp in SPLITS:
        # population = every live-config signal (all gates except the book slots)
        x = sel(f'lvl1003_f5_{scan}')
        x = x[(x.split == sp) & (x.pdr >= 8) & (x.floor_val >= 5) & (x.level >= 5) & (x.prev_close >= 5)
              & (x.r_pct >= 1) & (x.sig_m <= 840) & (x.adv20 >= 1e5) & x.filled]
        for lab, tt in (('every live-config signal', x.assign(net=x.r2_grossR, gross=x.r2_grossR)),
                        ('the shipped 12/4 book', trades('B0', scan, split=sp))):
            v = tt.gross.values if lab.startswith('every') else tt.gross.values
            rows.append(dict(scan=scan, population=lab, split=sp, n=len(v), tpw=round(len(v) / len(WEEKS[sp]), 1),
                             grossR=round(v.mean(), 4), se=round(v.std(ddof=1) / np.sqrt(len(v)), 4),
                             t=round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
                             MDE80=round(mde(v), 3), WR=round((v > 0).mean() * 100, 1)))
g2 = pd.DataFrame(rows); g2.to_csv(f'{OUT}/step2_gross.csv', index=False)
L.append(g2.to_string(index=False))

# ---------------------------------------------------------------- step 4: the fill model
L.append('\n## STEP 4 - the engine cap (level x 1.006) and the unfilled counterfactual')
rows = []
for scan in ('S1', 'S2'):
    x = sel(f'lvl1003_f5_{scan}')
    x = x[(x.pdr >= 8) & (x.floor_val >= 5) & (x.level >= 5) & (x.prev_close >= 5) & (x.r_pct >= 1)
          & (x.sig_m <= 840) & (x.adv20 >= 1e5)]
    rows.append(dict(scan=scan, signals=len(x), filled=int(x.filled.sum()),
                     fill_rate=round(100.0 * x.filled.mean(), 1),
                     over_cap_bps_med=round(x.loc[~x.filled, 'over_cap_bps'].median(), 0),
                     over_cap_bps_p90=round(x.loc[~x.filled, 'over_cap_bps'].quantile(0.9), 0)))
    for sp in SPLITS:
        for lab, sub in (('FILLED', x[x.filled & (x.split == sp)]), ('UNFILLED cf', x[~x.filled & (x.split == sp)])):
            rows.append(dict(scan=scan, split=sp, pop=lab, n=len(sub), grossR=round(sub.r2_grossR.mean(), 4),
                             holdR=round(sub.hold_grossR.mean(), 4), WR=round((sub.r2_grossR > 0).mean() * 100, 1)))
    xc = x[x.filled]
    rows.append(dict(scan=scan, pop='next-clock-minute share of fills', n=len(xc),
                     grossR=round(100.0 * xc.next_clock.mean(), 1)))
f4 = pd.DataFrame(rows); f4.to_csv(f'{OUT}/step4_fill.csv', index=False)
L.append(f4.to_string(index=False))

# ---------------------------------------------------------------- step 5: gate map
L.append('\n## STEP 5 - gate-separation map (leave-one-out on the whole live stack)')
GATES = [
    ('Gs first break clears the floor (S1 vs S2)', None),
    ('Gp pdr >= 8', 'pdr'), ('Gp pdr >= 10', 'pdr10'), ('Gp pdr >= 12', 'pdr12'),
    ('Gf floor >= 5 (S1 only: the floor test)', 'floor'),
    ('Gx level >= $5', 'price'), ('Gx level >= $10', 'price10'), ('Gx level >= $20', 'price20'),
    ('Gr r >= 1%', 'r'), ('Gr r >= 2%', 'r2'),
    ('Gt signal <= 14:00', 'late'), ('Gt signal <= 13:00', 'late780'),
    ('Ga adv20 >= 100K', 'adv'), ('Ga adv20 >= 500K', 'adv5'),
    ('Gu prev close >= $5', 'prevclose'),
    ('Gc fill <= level x 1.006 (the cap)', 'cap'),
    ('Gv rv in [1,5)', 'rv'), ('Gk next printed bar is the next clock minute', 'clock'),
]
rows = []
for scan in ('S1', 'S2'):
    for label, key in GATES:
        if key is None:
            continue
        thr = dict(pdr=8.0, pdr10=10.0, pdr12=12.0, price=5.0, price10=10.0, price20=20.0,
                   r=1.0, r2=2.0, late=840, late780=780, adv=1e5, adv5=5e5, prevclose=5.0).get(key)
        base = sel(f'lvl1003_f5_{scan}')
        cond = dict(pdr=8.0, floor=5.0, price=5.0, r=1.0, late=840, adv=1e5, prevclose=5.0)
        # the whole stack MINUS this gate
        m = np.ones(len(base), bool)
        if key not in ('pdr', 'pdr10', 'pdr12'): m &= (base.pdr >= cond['pdr']).values
        if key != 'floor': m &= (base.floor_val >= cond['floor']).values
        if key not in ('price', 'price10', 'price20'): m &= (base.level >= cond['price']).values
        if key not in ('r', 'r2'): m &= (base.r_pct >= cond['r']).values
        if key not in ('late', 'late780'): m &= (base.sig_m <= cond['late']).values
        if key not in ('adv', 'adv5'): m &= (base.adv20 >= cond['adv']).values
        if key != 'prevclose': m &= (base.prev_close >= cond['prevclose']).values
        if key != 'cap': m &= base.filled.values
        x = base[m]
        if key in ('pdr', 'pdr10', 'pdr12'): keep = x.pdr >= thr
        elif key == 'floor': keep = x.floor_val >= 5.0
        elif key in ('price', 'price10', 'price20'): keep = x.level >= thr
        elif key in ('r', 'r2'): keep = x.r_pct >= thr
        elif key in ('late', 'late780'): keep = x.sig_m <= thr
        elif key in ('adv', 'adv5'): keep = x.adv20 >= thr
        elif key == 'prevclose': keep = x.prev_close >= thr
        elif key == 'cap': keep = x.filled
        elif key == 'rv': keep = (x.rv >= 1.0) & (x.rv < 5.0)
        elif key == 'clock': keep = x.next_clock == 1
        a, bb = x[keep], x[~keep]
        if len(a) < 2 or len(bb) < 2:
            rows.append(dict(scan=scan, gate=label, nk=len(a), nr=len(bb), note='degenerate')); continue
        dg = a.r2_grossR.mean() - bb.r2_grossR.mean()
        dn = a.net_r2_meas.mean() - bb.net_r2_meas.mean()
        se = np.sqrt(a.net_r2_meas.var(ddof=1) / len(a) + bb.net_r2_meas.var(ddof=1) / len(bb))
        per = {}
        for sp in SPLITS:
            aa, bbb = a[a.split == sp], bb[bb.split == sp]
            per[sp] = round(aa.net_r2_meas.mean() - bbb.net_r2_meas.mean(), 4) if len(aa) > 1 and len(bbb) > 1 else np.nan
        rows.append(dict(scan=scan, gate=label, grossD=round(dg, 4), netD=round(dn, 4), t=round(dn / se, 2),
                         nk=len(a), nr=len(bb), TRAIN=per['TRAIN'], VAL=per['VAL'],
                         notl_k=round(float((RISK / a.R * a.entry).median()), 0),
                         notl_r=round(float((RISK / bb.R * bb.entry).median()), 0)))
# Gs: S1 vs S2 as a gate on S2's own population
for sp in ('POOLED',) + SPLITS:
    x2 = sel('lvl1003_f5_S2'); x1 = sel('lvl1003_f5_S1')
    f = lambda z: z[(z.pdr >= 8) & (z.floor_val >= 5) & (z.level >= 5) & (z.prev_close >= 5) & (z.r_pct >= 1)
                    & (z.sig_m <= 840) & (z.adv20 >= 1e5) & z.filled]
    x2, x1 = f(x2), f(x1)
    if sp != 'POOLED':
        x2, x1 = x2[x2.split == sp], x1[x1.split == sp]
    k1 = set(zip(x1.day, x1.symbol, x1.sig_m))
    keep = np.array([(dd, ss, mm) in k1 for dd, ss, mm in zip(x2.day, x2.symbol, x2.sig_m)])
    a, bb = x2[keep], x2[~keep]
    se = np.sqrt(a.net_r2_meas.var(ddof=1) / len(a) + bb.net_r2_meas.var(ddof=1) / len(bb))
    rows.append(dict(scan=sp, gate='Gs first break clears the floor (S1 kept vs the extra S2 signals)',
                     grossD=round(a.r2_grossR.mean() - bb.r2_grossR.mean(), 4),
                     netD=round(a.net_r2_meas.mean() - bb.net_r2_meas.mean(), 4),
                     t=round((a.net_r2_meas.mean() - bb.net_r2_meas.mean()) / se, 2), nk=len(a), nr=len(bb),
                     notl_k=round(float((RISK / a.R * a.entry).median()), 0),
                     notl_r=round(float((RISK / bb.R * bb.entry).median()), 0)))
gm = pd.DataFrame(rows); gm.to_csv(f'{OUT}/gatemap.csv', index=False)
L.append(gm.to_string(index=False))

# ---------------------------------------------------------------- steps 6/7: the frontier
L.append('\n## STEPS 6/7 - the declared cells, ranked on % GREEN WEEKS (dollars at $100 risk)')
rows = []
for scan in ('S1', 'S2'):
    for cell in CELLS:
        for sp in SPLITS:
            t = trades(cell, scan, split=sp)
            rows.append(dict(cell=cell, scan=scan, split=sp, **stats(t, sp)))
cells = pd.DataFrame(rows)
cells.to_csv(f'{OUT}/cells.csv', index=False)
for scan in ('S1', 'S2'):
    for sp in SPLITS:
        z = cells[(cells.scan == scan) & (cells.split == sp)].sort_values('green', ascending=False)
        L.append(f'### scan {scan}  split {sp}')
        L.append(z.drop(columns=['scan', 'split']).to_string(index=False))

# B0 secondary exits + the cost variants
L.append('\n## B0 secondary exits and cost variants')
rows = []
for scan in ('S1', 'S2'):
    for mode in ('r2', 'hold', 'partial'):
        for cost in ('meas', 'band', 'cons'):
            for sp in SPLITS:
                t = trades('B0', scan, mode=mode, cost=cost, split=sp)
                s = stats(t, sp); s.update(scan=scan, exit=mode, cost=cost, split=sp)
                rows.append(s)
sec = pd.DataFrame(rows); sec.to_csv(f'{OUT}/b0_variants.csv', index=False)
L.append(sec[['scan', 'exit', 'cost', 'split', 'n', 'grossR', 'netR', 't', 'green', 'tot', 'worst', 'mdd']].to_string(index=False))

# ---------------------------------------------------------------- step 8: weekly dollars
L.append('\n## STEP 8 - week by week at the live $100 risk (VAL, 2026-01 -> 2026-05)')
wk_rows = []
for scan in ('S1', 'S2'):
    for cell in ('B0', 'C1', 'C4'):
        t = trades(cell, scan, split='VAL')
        w = (t.groupby('wk').net.sum().reindex(WEEKS['VAL']).fillna(0.0) * RISK) if len(t) else \
            pd.Series(0.0, index=WEEKS['VAL'])
        cnt = t.groupby('wk').size().reindex(WEEKS['VAL']).fillna(0).astype(int) if len(t) else \
            pd.Series(0, index=WEEKS['VAL'])
        L.append(f'{scan} {cell}: ' + ' '.join('%+d(%d)' % (a, b) for a, b in zip(w.values, cnt.values)))
        L.append('   green %d/%d  total %+d  worst %+d' % ((w > 0).sum(), len(w), w.sum(), w.min()))
        for k, a, bq in zip(WEEKS['VAL'], w.values, cnt.values):
            wk_rows.append(dict(scan=scan, cell=cell, wk=k, dollars=a, n=bq))
pd.DataFrame(wk_rows).to_csv(f'{OUT}/weeks_val.csv', index=False)

# ---------------------------------------------------------------- step 9: count-matched null
L.append('\n## STEP 9 - count-matched permutation null (2,000 draws, per-week pick count fixed)')
rng = np.random.default_rng(20260919)
rows = []
for scan in ('S1', 'S2'):
    for cell in ('B0', 'P3', 'P4', 'X2', 'R2', 'C1', 'C2', 'C4'):
        for sp in SPLITS:
            t = trades(cell, scan, split=sp)
            if len(t) < 10:
                continue
            v = t.net.values
            cnt = t.groupby('wk').size().reindex(WEEKS[sp]).fillna(0).astype(int).values
            obs = float(((t.groupby('wk').net.sum().reindex(WEEKS[sp]).fillna(0.0)) > 0).mean() * 100)
            edges = np.cumsum(cnt)
            draws = np.empty(2000)
            for j in range(2000):
                p = rng.permutation(v)
                sums = np.add.reduceat(np.concatenate([p, [0.0]]), np.concatenate([[0], edges[:-1]]))
                sums = np.where(cnt > 0, sums, 0.0)
                draws[j] = (sums > 0).mean() * 100
            rows.append(dict(scan=scan, cell=cell, split=sp, obs=round(obs, 1), null_mean=round(draws.mean(), 1),
                             p5=round(np.percentile(draws, 5), 1), p95=round(np.percentile(draws, 95), 1),
                             outside='ABOVE' if obs > np.percentile(draws, 95) else
                                     ('below' if obs < np.percentile(draws, 5) else 'inside')))
nu = pd.DataFrame(rows); nu.to_csv(f'{OUT}/nulls.csv', index=False)
L.append(nu.to_string(index=False))

# ---------------------------------------------------------------- step 10: the bars
L.append('\n## STEP 10 - claim bar G1 (TRAIN net R > 0, t >= 2, >= 5 trades/wk) and G2 (VAL same sign, >= 55% green)')
rows = []
for scan in ('S1', 'S2'):
    for cell in CELLS:
        a = cells[(cells.scan == scan) & (cells.cell == cell) & (cells.split == 'TRAIN')].iloc[0]
        v = cells[(cells.scan == scan) & (cells.cell == cell) & (cells.split == 'VAL')].iloc[0]
        g1 = bool(a.netR > 0 and a.t >= 2 and a.tpw >= 5)
        g2 = bool(v.netR > 0 and v.green >= 55)
        rows.append(dict(scan=scan, cell=cell, TRAIN_netR=a.netR, TRAIN_t=a.t, TRAIN_tpw=a.tpw,
                         VAL_netR=v.netR, VAL_green=v.green, G1=g1, G2=g2, PASS=g1 and g2,
                         TRAIN_tot=a.tot, VAL_tot=v.tot, TRAIN_green=a.green))
bars = pd.DataFrame(rows); bars.to_csv(f'{OUT}/bars.csv', index=False)
L.append(bars.to_string(index=False))
L.append('\nG1 passes: %d of %d   BOTH: %d' % (int(bars.G1.sum()), len(bars), int(bars.PASS.sum())))
L.append('best TRAIN t: %.2f (%s)' % (bars.TRAIN_t.max(), bars.loc[bars.TRAIN_t.idxmax(), 'cell']))

# MDE on the primary metric
L.append('\nMDE, per trade (80%% power) and on the green-week share:')
for scan in ('S1', 'S2'):
    for sp in SPLITS:
        t = trades('B0', scan, split=sp)
        L.append('  %s %s: n=%d  MDE/trade=%.3f R   green-week MDE = +-%.1f pp over %d weeks'
                 % (scan, sp, len(t), mde(t.net.values), 100 * 2.80 * 0.5 / np.sqrt(len(WEEKS[sp])), len(WEEKS[sp])))

open(f'{OUT}/score.log', 'w').write('\n'.join(str(x) for x in L) + '\n')
log('\n'.join(str(x) for x in L))
