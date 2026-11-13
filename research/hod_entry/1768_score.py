"""Cell 1,768 (J7): short the >= 10 % gapper EARLY (E1-E4) with the opening-range high as the stop.
PREREG: research/hod_entry/PREREG_1768.md (frozen 2026-11-13 21:08 UTC).

Reuses by import from 1767_score.py: prev_closes, price_at, build-style per-day loop pieces, cost_table (symbol-median
half-spread, price-decile band fallback -- NOT a per-trade NBBO), S.day_t / S.ex_top5.  Timing: a price 'at HH:MM' = close of
the bar that ENDS at HH:MM; stop fills at max(stop, bar open) + 50 bp slip; minute bars cannot time intrabar (a stop and
a cover in one bar resolve to the stop).  E4 entry = close of the first bar (stamps 09:35..09:59, i.e. close at 09:36..10:00)
whose close < the 09:30-09:35 low.  Cost = 2*2 bp + 2*symbol half-spread + borrow 50 bp/day (100 bp sensitivity).
TRAIN grid is written before VAL is read.  Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1768_score.py
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_IO_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, importlib.util
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
HE = f'{ROOT}/research/hod_entry'
sys.path.insert(0, ROOT)
sp = importlib.util.spec_from_file_location('s1767', f'{HE}/1767_score.py')
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)
S, S3, log = P.S, P.S3, P.log
OPEN_M, SLIP, FEE, BORROW = 570, 50.0, 2.0, 50.0
COVERS = {'1030': 630, '1555': 955}
ENTRIES = ['E1', 'E2', 'E3', 'E4']
PRIMARY = ('E3', '1030')


def trade(m, o, h, l, c, op, kind, hs_bp):
    """Return list of (cover, net_bp, gross_bp, stop_hit, stop_dist_bp) for one entry kind on one symbol-day; [] if no signal."""
    if kind == 'E1':
        T, H = 575, float(h[m < 575].max()); ent_i = np.searchsorted(m, 574, side='right') - 1
    elif kind in ('E2', 'E3'):
        T, H = 585, float(h[m < 585].max()); ent_i = np.searchsorted(m, 584, side='right') - 1
        if kind == 'E3' and not (ent_i >= 0 and c[ent_i] < op):
            return []
    else:  # E4: first close below the 09:30-09:35 low, bars stamped 575..599
        H, lo = float(h[m < 575].max()), float(l[m < 575].min())
        idx = np.flatnonzero((m >= 575) & (m <= 599) & (c < lo))
        if not len(idx):
            return []
        ent_i = idx[0]; T = int(m[ent_i]) + 1
    if ent_i < 0 or not np.isfinite(c[ent_i]):
        return []
    entry = float(c[ent_i]); stop_px = H * (1 + SLIP / 1e4)
    sd = (stop_px / entry - 1) * 1e4
    out = []
    for cn, cm in COVERS.items():
        win = np.flatnonzero((m >= T) & (m <= cm - 1))
        hit = win[h[win] >= H] if len(win) else win
        if len(hit):
            j = hit[0]; exit_p = max(H, float(o[j])) * (1 + SLIP / 1e4); st = 1
        else:
            k = np.searchsorted(m, cm - 1, side='right') - 1
            if k < 0 or m[k] < T:
                continue
            exit_p = float(c[k]); st = 0
        gross = (entry - exit_p) / entry * 1e4
        net = gross - 2 * FEE - 2 * hs_bp - BORROW
        out.append((cn, net, gross, st, sd))
    return out


def symbol_day(g, row, hs_sym, hs_band):
    """All E1-E4 trades for one (day, symbol) as dict rows (long format)."""
    m = g['m'].to_numpy(); o, h, l, c = (g[k].to_numpy() for k in 'ohlc')
    if len(m) == 0 or m[0] != OPEN_M:
        return None, []
    op = float(o[0]); gap = op / row.prev_close - 1
    hs = hs_sym.get(row.symbol)
    hs = hs_band(op) if hs is None or not np.isfinite(hs) else hs
    rec = dict(day=row.day, symbol=row.symbol, split=row.split, wk=row.wk, open=op, gap=gap)
    rows = []
    if op < 5.0:
        return rec, rows
    for kind in ENTRIES:
        for cn, net, gross, st, sd in trade(m, o, h, l, c, op, kind, hs * 1e4):
            rows.append(dict(rec, entry=kind, cover=cn, net=net, gross=gross, stop=st, sd=sd, hs_bp=hs * 1e4))
    return rec, rows


def build(pop, days, hs_sym, hs_band, label):
    """Per-day pass; returns the long trade frame (gap |scale| filter applied)."""
    out, nsd = [], 0
    for k, day in enumerate(days):
        f = f'{HE}/1762_cache/bars/{day}.parquet'
        if not os.path.exists(f):
            continue
        b = pd.read_parquet(f); pr = pop[pop.day == day].set_index('symbol')
        b = b[b.symbol.isin(pr.index)]
        et = pd.to_datetime(b.t, utc=True).dt.tz_convert('America/New_York')
        b = b.assign(m=et.dt.hour * 60 + et.dt.minute)
        b = b[(b.m >= OPEN_M) & (b.m < 960)].sort_values(['symbol', 'm'])
        for sym, g in b.groupby('symbol', sort=False):
            row = P._R(pr.loc[sym], sym, day)
            if not np.isfinite(row.prev_close):
                continue
            rec, rows = symbol_day(g, row, hs_sym, hs_band)
            if rec is not None and abs(rec['gap']) <= P.GAP_SCALE_MAX:
                nsd += 1; out.extend(rows)
        if k % 25 == 0:
            log(f'{label}: day {k + 1}/{len(days)} {day}, trades {len(out)}')
    log(f'{label}: done, {len(out)} trades over {nsd} symbol-days')
    return pd.DataFrame(out)


def stats(s, weeks, wk_all):
    """One grid row of the PREREG numbers for a trade frame s (single entry x cover x pop)."""
    x, d = s.net.to_numpy(), s.day.to_numpy()
    mu, tt, se = S.day_t(x, d)
    dm = s.groupby('day').net.mean().sort_values().to_numpy()
    k5, k1 = max(1, int(np.floor(len(dm) * .05))), max(1, int(np.floor(len(dm) * .01)))
    wm = s.groupby('wk').net.mean()
    wk = wm.reindex(sorted(wk_all), fill_value=0.0)             # weeks with no trade count as 0
    return dict(n=len(x), per_wk=len(x) / weeks, mean_net=mu, day_t=tt, hit=float((x > 0).mean()), ex_top5=S.ex_top5(x),
                worst1_daymean=float(dm[:k1].mean()), worst5_daymean=float(dm[:k5].mean()), max_loss=float(x.min()),
                stop_hit=float(s.stop.mean()), mean_R=float((s.net / s.sd).mean()), wk_p10=float(np.percentile(wk, 10)),
                y2025=float(s[s.day < '2026'].net.mean()) if (s.day < '2026').any() else np.nan,
                y2026=float(s[s.day >= '2026'].net.mean()) if (s.day >= '2026').any() else np.nan,
                net_borrow100=mu - BORROW, MDE80=2.8 * se, gross=float(s.gross.mean()), hs_bp=float(s.hs_bp.mean()))


def grid(t, sample, weeks, wk_all):
    """Grid rows for every entry x cover x population (gap>=10 primary, gap 5-10 secondary)."""
    rows = []
    for pn, sel in (('gap>=10', t.gap >= .10), ('gap5-10', (t.gap >= .05) & (t.gap < .10))):
        for e in ENTRIES:
            for cn in COVERS:
                s = t[sel & (t.entry == e) & (t.cover == cn)]
                if len(s) >= 5:
                    rows.append(dict(sample=sample, pop=pn, entry=e, cover=cn, **stats(s, weeks, wk_all)))
    return rows


def main():
    """TRAIN halves + TRAIN grid, then VAL once, then the kill verdict on the primary (E3 x 10:30)."""
    pop = pd.read_csv(f'{HE}/cell_1428_population.csv', keep_default_na=False)
    pop = P.prev_closes(pop)
    hs_sym, hs_band = S3.cost_table()
    tr = sorted(pop[pop.split == 'TRAIN'].day.unique()); va = sorted(pop[pop.split == 'VAL'].day.unique())
    half = {d: ('H1' if k < len(tr) // 2 else 'H2') for k, d in enumerate(tr)}
    wkt = lambda dd: set(pop[pop.day.isin(dd)].wk)
    t = build(pop, tr, hs_sym, hs_band, 'TRAIN'); t['half'] = t.day.map(half)
    h1d = [d for d in tr if half[d] == 'H1']; h2d = [d for d in tr if half[d] == 'H2']
    rows = (grid(t[t.half == 'H1'], 'H1', len(wkt(h1d)), wkt(h1d)) + grid(t[t.half == 'H2'], 'H2', len(wkt(h2d)), wkt(h2d))
            + grid(t, 'TRAIN', len(wkt(tr)), wkt(tr)))
    pd.DataFrame(rows).to_csv(f'{HE}/1768_grid.csv', index=False)
    log('TRAIN grid written; reading VAL once')
    v = build(pop, va, hs_sym, hs_band, 'VAL')
    rows += grid(v, 'VAL', len(wkt(va)), wkt(va))
    g = pd.DataFrame(rows); g.to_csv(f'{HE}/1768_grid.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    cols = ['sample', 'pop', 'entry', 'cover', 'n', 'per_wk', 'mean_net', 'day_t', 'hit', 'ex_top5', 'worst1_daymean',
            'worst5_daymean', 'max_loss', 'stop_hit', 'mean_R', 'wk_p10', 'net_borrow100', 'MDE80', 'y2025', 'y2026']
    print(g[(g['pop'] == 'gap>=10')][cols].round(2).to_string())
    print(g[(g['pop'] == 'gap5-10') & (g['sample'].isin(['TRAIN', 'VAL']))][cols].round(2).to_string())
    e, c = PRIMARY
    q = lambda smp: g[(g['sample'] == smp) & (g['pop'] == 'gap>=10') & (g.entry == e) & (g.cover == c)].iloc[0]
    k, h1, h2 = q('VAL'), q('H1'), q('H2')
    kills = {'VAL net<=50bp': k.mean_net <= 50, 'day-t<2': k.day_t < 2, 'TRAIN halves opposite': np.sign(h1.mean_net) != np.sign(h2.mean_net),
             'ex-top5<=0': k.ex_top5 <= 0, 'worst5 day mean<-800bp': k.worst5_daymean < -800, '<3 names/wk': k.per_wk < 3,
             'net@borrow100<=0': k.net_borrow100 <= 0}
    log(f'PRIMARY H1 {h1.mean_net:.1f} H2 {h2.mean_net:.1f}; KILL CHECKS: {kills} => {"KILL" if any(kills.values()) else "SURVIVE"}')


if __name__ == '__main__':
    main()
