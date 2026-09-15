#!/usr/bin/env python3
"""bf_zero2 pass 1 — every entry family × config on the WHOLE point-in-time universe, SIP tape only (DESIGN.md).

Reuses bf_zero's family/exit functions (research/bf_zero/build_candidates.py) with: the SIP store as the only side store,
no hindsight bar-count gate (>= 10 bars, as the live spec), the pre-registered F5 grid, two new families (F9 gap-and-go,
F10 VWAP reclaim), and an extra exit rr_e1c = +2R target filled on a bar CLOSE (the live spec's fill), stop first.
Resumable per day (build_state.json) → candidates.csv + volume_profile.csv under research/bf_zero2/.
"""
import json, os, sys, time
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
os.environ.setdefault('BFZ_SIP_STORE', f'{ROOT}/research/bf_zero/bars_sip.db')
sys.path.insert(0, f'{ROOT}/research/bf_zero')
import build_candidates as B                                   # family functions, exits(), load_bars(), uni, spy, daily merges
D = 'research/bf_zero2'
STATE, OUT, MISS, VPROF = f'{D}/build_state.json', f'{D}/candidates.csv', f'{D}/coverage_missing.csv', f'{D}/volume_profile.csv'
COLS = B.COLS + ['rr_e1c', 'why_e1c', 'exit_m_e1c', 'n_bars_at_entry', 'pm_vol', 'pm_high_pct', 'range_so_far_pct', 'bars_per_min']
SLIP = B.SLIP; OPEN_M = B.OPEN_M; VP_MIN = B.VP_MIN

FAMS = ([('F1', dict(P=P)) for P in (0.05, 0.08, 0.12)] + [('F2', dict(P=P)) for P in (0.05, 0.08, 0.12)] +
        [('F3', dict(P=P, M=M)) for P in (0.05, 0.10) for M in (15, 30)] + [('F4', {})] +
        [('F5', dict(K=K, X=X)) for K in (3, 5, 8) for X in (0.02, 0.04, 0.06)] + [('F6', {})] + [('F7', {})] +
        [('F8', dict(N=N)) for N in (5, 15, 30)] + [('F9', dict(G=0.05))] + [('F10', {})])


def fam_vwap_reclaim(h, l, c, vwap, min_below=10):
    """F10: >= min_below consecutive bars closing below VWAP, then a bar closing above it; entry = break of that bar's high,
    stop = the lowest low of the below-VWAP stretch."""
    n = len(c); below = c < vwap; run = np.zeros(n, int)
    for i in range(1, n): run[i] = run[i - 1] + 1 if below[i] else 0
    t = np.arange(n); j = np.clip(t - 1, 0, n - 1); k = np.clip(t - 2, 0, n - 1)
    ok = (t >= 3) & (~below[j]) & (run[k] >= min_below) & (h >= h[j] * (1 + SLIP))
    i = B.first_true(ok)
    if i is None: return None
    start = i - 1 - run[i - 2]; stop = float(l[max(start, 0):i].min()); level = float(h[i - 1])
    if stop >= level: return None
    return (i, level, stop, dict())


def exit_close_fill(o, h, l, c, m, i, entry, stop, mult=2.0):
    """+mult R target filled when a bar CLOSES at/above it (the live bracket spec), stop first, flat at EOD_M.
    Returns (rr, why, exit_minute)."""
    Rd = entry - stop; oo, hh, ll, cc, mm = o[i + 1:], h[i + 1:], l[i + 1:], c[i + 1:], m[i + 1:]
    if len(oo) == 0: return 0.0, 'none', int(m[i])
    tgt = entry + mult * Rd
    for k in range(len(oo)):
        if mm[k] >= B.EOD_M: return (oo[k] - entry) / Rd, 'eod', int(mm[k])
        if ll[k] <= stop: return (min(stop, oo[k]) * 0.999 - entry) / Rd, 'stop', int(mm[k])
        if cc[k] >= tgt: return mult, 'target', int(mm[k])
    return (cc[-1] - entry) / Rd, 'eod', int(mm[-1])


def build_day(day, sub):
    bars = B.load_bars(day, sub.symbol.tolist()); rows = []; missing = []; spym = B.spy_by_day.get(day, {}); vprof = []
    for r in sub.itertuples():
        gg = bars.get(r.symbol)
        if gg is None: missing.append(r.symbol); continue
        rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
        if len(rth) < 10: missing.append(r.symbol); continue         # the live spec's floor, not a hindsight 30-bar gate
        o, h, l, c, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v')); m = rth.m.values.astype(int)
        o0 = o[0]; cumv = np.cumsum(v); vwap = np.cumsum(c * v) / np.maximum(cumv, 1)
        pm = gg[gg.m < OPEN_M]; pmh = float(pm.h.max()) if len(pm) else None; pmv = float(pm.v.sum()) if len(pm) else 0.0
        vp = {'day': day, 'symbol': r.symbol, 'day_vol': float(cumv[-1]), 'pm_vol': pmv}
        for cm in VP_MIN:
            k = np.searchsorted(m, cm, side='right') - 1
            vp[f'cv_{cm}'] = float(cumv[k]) if k >= 0 else 0.0
        vprof.append(vp)
        gap = (o0 / r.prev_close - 1) if r.prev_close == r.prev_close and r.prev_close > 0 else np.nan
        for fam, cfg in FAMS:
            if fam == 'F1': res = B.fam_flag(h, l, cfg['P'], (2, 3, 4, 5, 6), micro=False)
            elif fam == 'F2': res = B.fam_flag(h, l, cfg['P'], (1, 2, 3), micro=True)
            elif fam == 'F3': res = B.fam_drive(h, l, o0, m, cfg['P'], cfg['M'])
            elif fam == 'F4': res = B.fam_vwap(h, l, c, vwap)
            elif fam == 'F5': res = B.fam_hod(h, l, cfg['K'], cfg['X'])
            elif fam == 'F6': res = B.fam_r2g(h, l, o0, r.prev_close if r.prev_close == r.prev_close else None)
            elif fam == 'F7': res = B.fam_level(h, l, pmh, 1, float(l[:1].min())) if pmh else None
            elif fam == 'F8':
                N = cfg['N']; res = B.fam_level(h, l, float(h[:N].max()), N, float(l[:N].min())) if len(h) > N else None
            elif fam == 'F9':
                res = B.fam_level(h, l, float(h[:5].max()), 5, float(l[:5].min())) if (len(h) > 5 and gap == gap and gap >= cfg['G']) else None
            else: res = fam_vwap_reclaim(h, l, c, vwap)
            if res is None: continue
            i, level, stop, extra = res
            entry = level * (1 + SLIP)
            if stop >= entry or i + 1 >= len(o): continue
            ex = B.exits(o, h, l, c, m, i, entry, stop); rr1c, why1c, xm1c = exit_close_fill(o, h, l, c, m, i, entry, stop)
            row = dict(day=day, symbol=r.symbol, fam=fam, cfg=json.dumps(cfg, sort_keys=True), entry_m=int(m[i]), entry=entry, stop=stop,
                       r_pct=(entry - stop) / entry * 100, price=entry, dist_open_pct=(entry / o0 - 1) * 100,
                       rv_adv=cumv[i] / r.adv20 if r.adv20 == r.adv20 and r.adv20 > 0 else np.nan,
                       bar_vol_x=v[i] / v[:i].mean() if i > 0 and v[:i].mean() > 0 else np.nan,
                       above_vwap=int(entry > vwap[i - 1]) if i > 0 else 1, gap_pct=gap * 100 if gap == gap else np.nan,
                       prev_range_pct=(r.prev_high - r.prev_low) / r.prev_close * 100 if r.prev_close == r.prev_close and r.prev_close > 0 else np.nan,
                       adv20=r.adv20, dist_20d_high_pct=(entry / r.high20 - 1) * 100 if r.high20 == r.high20 and r.high20 > 0 else np.nan,
                       spy_5m_ret=((spym[int(m[i])] / spym[int(m[i]) - 5]) - 1) * 100 if spym.get(int(m[i])) and spym.get(int(m[i]) - 5) else np.nan,
                       spy_range3=float(B.spyd.range3.get(day, np.nan)), pm_covered=int(pmh is not None),
                       rr_e1c=rr1c, why_e1c=why1c, exit_m_e1c=xm1c, n_bars_at_entry=int(i + 1), pm_vol=pmv, pm_high_pct=(pmh / o0 - 1) * 100 if pmh else np.nan,
                       range_so_far_pct=(h[:i + 1].max() - l[:i + 1].min()) / o0 * 100, bars_per_min=(i + 1) / max(int(m[i]) - OPEN_M + 1, 1),
                       **extra, **ex)
            rows.append(row)
    if vprof:
        pd.DataFrame(vprof).to_csv(VPROF, mode='a', header=not os.path.exists(VPROF), index=False)
    return rows, missing


def main():
    uni = B.uni
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': [], 'missing': 0}
    done = set(state['done']); days = sorted(uni.bar_date.unique())
    only = os.environ.get('BFZ_DAYS')
    if only: days = days[:int(only)]
    todo = [d for d in days if d not in done]
    print(f'bf_zero2 pass 1 | universe days {len(days)} | done {len(done)} | todo {len(todo)} | fams {len(FAMS)} | slip {SLIP} | store {os.environ["BFZ_SIP_STORE"]}', flush=True)
    t0 = time.time(); nrows = 0
    for k, day in enumerate(todo):
        sub = uni[uni.bar_date == day]
        rows, miss = build_day(day, sub)
        if rows: pd.DataFrame(rows).reindex(columns=COLS).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        if miss: pd.DataFrame({'symbol': miss, 'bar_date': day}).to_csv(MISS, mode='a', header=not os.path.exists(MISS), index=False)
        nrows += len(rows); state['done'].append(day); state['missing'] += len(miss)
        json.dump(state, open(STATE, 'w'))
        if k % 5 == 0 or k == len(todo) - 1:
            el = time.time() - t0
            print(f'{k + 1}/{len(todo)} {day} syms={len(sub)} rows+={len(rows)} miss={len(miss)} | total rows {nrows} | {el / 60:.1f} min, {el / (k + 1):.1f} s/day', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
