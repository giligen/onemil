#!/usr/bin/env python3
"""Bull-flag from zero — pass 1: entry families x exits on the clean-sheet universe (DESIGN.md).

One row per (day, symbol, family, config) = the FIRST qualifying event of that family-config on
that symbol-day. Every quantity is computed on bars at or before the entry bar; exits walk from
the bar AFTER entry (entry bar excluded, as in the unified BF spec). Resumable per day
(`build_state.json`), appends to `candidates.csv`.

Bars: cache.db intraday_bars_1min, then the point-in-time side DBs (ignition top-up, ORB/BF
PIT, this study's bars.db). Universe: universe.csv (range >= 5%, price >= $1, ADV20 >= 100K).
"""
import json, os, sqlite3, sys, time
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
D = 'research/bf_zero'
STATE, OUT, MISS = f'{D}/build_state.json', f'{D}/candidates.csv', f'{D}/coverage_missing.csv'
VPROF = f'{D}/volume_profile.csv'                      # owner 9/13: same-clock cumulative volume checkpoints per symbol-day
VP_MIN = (575, 585, 600, 630, 660, 720, 780, 840, 900) # 9:35 9:45 10:00 10:30 11:00 12:00 13:00 14:00 15:00 ET
RETRY = os.environ.get('BFZ_RETRY') == '1'
COLS = ['day', 'symbol', 'fam', 'cfg', 'entry_m', 'entry', 'stop', 'r_pct', 'price', 'dist_open_pct', 'rv_adv', 'bar_vol_x', 'above_vwap',
        'gap_pct', 'prev_range_pct', 'adv20', 'dist_20d_high_pct', 'spy_5m_ret', 'spy_range3', 'pm_covered',
        'pole_gain', 'retrace', 'flag_len', 'drive_min', 'pull_len',
        'rr_e1', 'why_e1', 'rr_e2', 'why_e2', 'rr_e3', 'why_e3', 'rr_e4', 'why_e4']
SLIP = float(os.environ.get('BFZ_SLIP', '0.003'))
EOD_M = 955          # 15:55 ET force flat
OPEN_M = 570

cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
SIDE = [sqlite3.connect(f'file:{p}?mode=ro', uri=True, timeout=120) for p in
        (f'{ROOT}/research/ignition_capcheck/topup.db', f'{ROOT}/data/research/databento/pit_bars_1min.db', f'{D}/bars.db')
        if os.path.exists(p)]

uni = pd.read_csv(f'{D}/universe.csv', dtype={'symbol': str}, keep_default_na=False)
for _c in ('open', 'high', 'low', 'close', 'volume', 'adv20', 'prev_vol'): uni[_c] = pd.to_numeric(uni[_c], errors='coerce')
daily = pd.read_parquet(f'{ROOT}/data/research/databento/equs_daily_2025_2026.parquet')
daily['bar_date'] = daily.bar_date.astype(str).str[:10]
daily = daily[daily.symbol.notna() & (daily.symbol.astype(str).str.strip() != '')].sort_values(['symbol', 'bar_date'])
g = daily.groupby('symbol')
daily['prev_close'] = g.close.shift(1); daily['prev_high'] = g.high.shift(1); daily['prev_low'] = g.low.shift(1)
daily['high20'] = g.high.transform(lambda s: s.shift(1).rolling(20, min_periods=5).max())
dkey = daily.set_index(['symbol', 'bar_date'])[['prev_close', 'prev_high', 'prev_low', 'high20']]
uni = uni.merge(dkey, left_on=['symbol', 'bar_date'], right_index=True, how='left')
spyd = daily[daily.symbol == 'SPY'].set_index('bar_date')
spyd['range3'] = ((spyd.high - spyd.low) / spyd.close * 100).rolling(3).mean().shift(1)
spy = pd.read_sql("select bar_date, timestamp, close from intraday_bars_1min where symbol='SPY' and bar_date>='2025-01-02'", cache)
spy['m'] = pd.to_datetime(spy.timestamp, utc=True).dt.tz_convert('America/New_York').pipe(lambda t: t.dt.hour * 60 + t.dt.minute)
spy_by_day = {d: dict(zip(gg.m, gg.close)) for d, gg in spy.groupby('bar_date')}


def load_bars(day, syms):
    """{symbol: DataFrame(m,o,h,l,c,v) sorted by minute, RTH+PM} from the first source that has it."""
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


def roll_min(a, w):
    """rolling min over the last w values (inclusive), NaN-padded at the front."""
    return pd.Series(a).rolling(w, min_periods=w).min().values


def roll_max(a, w):
    return pd.Series(a).rolling(w, min_periods=w).max().values


def first_true(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


# ---------------- entry families: each returns (entry_idx, level, stop, extra) or None ----------------
def fam_flag(h, l, P, flag_lens, micro):
    """F1 bull flag (flag 2-6 bars, retrace <= 50%) or F2 micro pullback (1-3 bar pause, no new high).
    Pole: high[pe] vs min low over the 9 bars ending pe, gain >= P."""
    n = len(h); pole_low = roll_min(l, 9); pole_gain = h / pole_low - 1
    best = None
    for f in flag_lens:
        # candidate entry bar t = pe + f + 1 ; flag/pause bars pe+1..pe+f
        fh = roll_max(h, f); fl = roll_min(l, f)           # over bars ending at index j
        t = np.arange(n); pe = t - f - 1; j = t - 1
        ok = (pe >= 8)
        pe_c = np.clip(pe, 0, n - 1); j_c = np.clip(j, 0, n - 1)
        hp = h[pe_c]; pl = pole_low[pe_c]; gain = pole_gain[pe_c]
        flag_high = fh[j_c]; flag_low = fl[j_c]
        ok &= gain >= P
        ok &= flag_high <= hp * 1.002                        # no new high inside the flag
        ok &= flag_low >= hp - 0.5 * (hp - pl)               # retrace <= 50% of the pole
        if micro:
            level = h[j_c]                                    # break of the prior bar's high
        else:
            level = flag_high                                 # break of the flag high
        ok &= h >= level * (1 + SLIP)                         # fill possible at the slipped level
        ok &= flag_low < level                                # stop below entry
        i = first_true(ok)
        if i is not None and (best is None or i < best[0]):
            retr = (hp[i] - flag_low[i]) / max(hp[i] - pl[i], 1e-9)
            best = (i, float(level[i]), float(flag_low[i]), dict(pole_gain=float(gain[i]), retrace=float(retr), flag_len=f))
    return best


def fam_drive(h, l, o0, m, P, M):
    """F3 opening drive: >= P% above the open by minute M, then the first >=2-bar lower-high pullback; entry = break of prior bar high."""
    n = len(h); reached = first_true((h >= o0 * (1 + P)) & (m <= OPEN_M + M))
    if reached is None: return None
    lower = np.zeros(n, bool); lower[1:] = h[1:] <= h[:-1]
    run = np.zeros(n, int)
    for i in range(1, n): run[i] = run[i - 1] + 1 if lower[i] else 0
    t = np.arange(n); ok = (t > reached + 1) & (run[np.clip(t - 1, 0, n - 1)] >= 2) & (h >= h[np.clip(t - 1, 0, n - 1)] * (1 + SLIP))
    i = first_true(ok)
    if i is None: return None
    k = run[i - 1]; stop = float(l[i - k: i].min()); level = float(h[i - 1])
    if stop >= level: return None
    return (i, level, stop, dict(drive_min=int(m[reached] - OPEN_M), pull_len=int(k)))


def fam_vwap(h, l, c, vwap):
    """F4 VWAP bounce: after >= 5% above VWAP, a bar touching within 1% of VWAP (close holding >= VWAP*0.99), entry = break of that bar's high."""
    n = len(h); above = np.maximum.accumulate((c / vwap - 1) >= 0.05)
    touch = (l <= vwap * 1.01) & (c >= vwap * 0.99)
    t = np.arange(n); j = np.clip(t - 1, 0, n - 1)
    ok = (t >= 2) & above[j] & touch[j] & (h >= h[j] * (1 + SLIP)) & (l[j] < h[j])
    i = first_true(ok)
    if i is None: return None
    return (i, float(h[i - 1]), float(l[i - 1]), dict())


def fam_hod(h, l, K, X):
    """F5 HOD break after K bars all within X% of the running high."""
    n = len(h); hod = np.maximum.accumulate(h); lo = roll_min(l, K)
    t = np.arange(n); j = np.clip(t - 1, 0, n - 1)
    ok = (t > K) & (lo[j] >= hod[j] * (1 - X)) & (h >= hod[j] * (1 + SLIP)) & (lo[j] < hod[j])
    i = first_true(ok)
    if i is None: return None
    return (i, float(hod[i - 1]), float(lo[i - 1]), dict())


def fam_r2g(h, l, o0, prev_close):
    """F6 red-to-green: opened below the prior close, entry = first cross above it; stop = the day's low so far."""
    if not (prev_close and o0 < prev_close): return None
    n = len(h); lo = np.minimum.accumulate(l); t = np.arange(n)
    ok = (t >= 1) & (h >= prev_close * (1 + SLIP))
    i = first_true(ok)
    if i is None or lo[i - 1] >= prev_close: return None
    return (i, float(prev_close), float(lo[i - 1]), dict())


def fam_level(h, l, level, start_idx, stop_lo):
    """F7/F8: first break of a fixed level after start_idx; stop given."""
    n = len(h); t = np.arange(n); ok = (t >= start_idx) & (h >= level * (1 + SLIP))
    i = first_true(ok)
    if i is None or stop_lo >= level: return None
    return (i, float(level), float(stop_lo), dict())


# ---------------- exits ----------------
def exits(o, h, l, c, m, i, entry, stop):
    """Walk from bar i+1. Returns dict of rr_e1..rr_e4 (+ why). Stop before target inside one bar; target needs a trade-through."""
    Rd = entry - stop; out = {}
    oo, hh, ll, cc, mm = o[i + 1:], h[i + 1:], l[i + 1:], c[i + 1:], m[i + 1:]
    if len(oo) == 0:
        return {k: 0.0 for k in ('rr_e1', 'rr_e2', 'rr_e3', 'rr_e4')} | {'why_e1': 'none'}
    eod = first_true(mm >= EOD_M); eod = eod if eod is not None else len(oo) - 1
    s_idx = first_true(ll <= stop)
    def stop_fill(k): return min(stop, oo[k]) * 0.999
    for name, mult in (('e1', 2.0), ('e3', 3.0)):
        tgt = entry + mult * Rd; t_idx = first_true(hh >= tgt * 1.002)
        cand = [(s_idx, 'stop'), (t_idx, 'target'), (eod, 'eod')]
        cand = [(k, w) for k, w in cand if k is not None]; k, w = min(cand, key=lambda x: (x[0], 0 if x[1] == 'stop' else 1))
        px = stop_fill(k) if w == 'stop' else (tgt if w == 'target' else oo[k])
        out[f'rr_{name}'] = (px - entry) / Rd; out[f'why_{name}'] = w
    # e4 hold with -1R stop
    k, w = min([(x, y) for x, y in ((s_idx, 'stop'), (eod, 'eod')) if x is not None], key=lambda x: x[0])
    out['rr_e4'] = ((stop_fill(k) if w == 'stop' else oo[k]) - entry) / Rd; out['why_e4'] = w
    # e2 partial 50% @2R (trade-through), stop -> entry, remainder trails higher lows (bar-close ratchet), eod flat
    tgt = entry + 2 * Rd; cur = stop; taken = False; rr = 0.0; frac = 1.0; last_low = prev_low = None; why = 'eod'
    for k in range(len(oo)):
        if mm[k] >= EOD_M: rr += frac * (oo[k] - entry) / Rd; break
        if ll[k] <= cur: rr += frac * (min(cur, oo[k]) * 0.999 - entry) / Rd; why = 'stop' if not taken else 'trail'; break
        if not taken and hh[k] >= tgt * 1.002: taken = True; rr += 0.5 * 2.0; frac = 0.5; cur = max(cur, entry)
        if taken and prev_low is not None and last_low is not None and ll[k] > last_low > prev_low: cur = max(cur, last_low)
        prev_low, last_low = last_low, ll[k]
    else:
        rr += frac * (cc[-1] - entry) / Rd
    out['rr_e2'] = rr; out['why_e2'] = why
    return out


FAMS = ([('F1', dict(P=P)) for P in (0.05, 0.08, 0.12)] + [('F2', dict(P=P)) for P in (0.05, 0.08, 0.12)] +
        [('F3', dict(P=P, M=M)) for P in (0.05, 0.10) for M in (15, 30)] + [('F4', {})] +
        [('F5', dict(K=K, X=X)) for K in (5, 10, 15) for X in (0.02, 0.04)] + [('F6', {})] + [('F7', {})] +
        [('F8', dict(N=N)) for N in (15, 30)])


def build_day(day, sub):
    B = load_bars(day, sub.symbol.tolist()); rows = []; missing = []; spym = spy_by_day.get(day, {}); vprof = []
    for r in sub.itertuples():
        gg = B.get(r.symbol)
        if gg is None: missing.append(r.symbol); continue
        rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
        if len(rth) < 30: missing.append(r.symbol); continue
        o, h, l, c, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v')); m = rth.m.values.astype(int)
        o0 = o[0]; cumv = np.cumsum(v); vwap = np.cumsum(c * v) / np.maximum(cumv, 1)
        vp = {'day': day, 'symbol': r.symbol, 'day_vol': float(cumv[-1]), 'pm_vol': float(gg[gg.m < OPEN_M].v.sum())}
        for cm in VP_MIN:
            k = np.searchsorted(m, cm, side='right') - 1           # last bar at or before the checkpoint
            vp[f'cv_{cm}'] = float(cumv[k]) if k >= 0 else 0.0
        vprof.append(vp)
        pm = gg[gg.m < OPEN_M]; pmh = float(pm.h.max()) if len(pm) else None
        for fam, cfg in FAMS:
            if fam == 'F1': res = fam_flag(h, l, cfg['P'], (2, 3, 4, 5, 6), micro=False)
            elif fam == 'F2': res = fam_flag(h, l, cfg['P'], (1, 2, 3), micro=True)
            elif fam == 'F3': res = fam_drive(h, l, o0, m, cfg['P'], cfg['M'])
            elif fam == 'F4': res = fam_vwap(h, l, c, vwap)
            elif fam == 'F5': res = fam_hod(h, l, cfg['K'], cfg['X'])
            elif fam == 'F6': res = fam_r2g(h, l, o0, r.prev_close if r.prev_close == r.prev_close else None)
            elif fam == 'F7': res = fam_level(h, l, pmh, 1, float(l[:1].min())) if pmh else None
            else:
                N = cfg['N']; res = fam_level(h, l, float(h[:N].max()), N, float(l[:N].min())) if len(h) > N else None
            if res is None: continue
            i, level, stop, extra = res
            entry = level * (1 + SLIP)
            if stop >= entry or i + 1 >= len(o): continue
            ex = exits(o, h, l, c, m, i, entry, stop)
            row = dict(day=day, symbol=r.symbol, fam=fam, cfg=json.dumps(cfg, sort_keys=True), entry_m=int(m[i]), entry=entry, stop=stop,
                       r_pct=(entry - stop) / entry * 100, price=entry, dist_open_pct=(entry / o0 - 1) * 100,
                       rv_adv=cumv[i] / r.adv20 if r.adv20 == r.adv20 and r.adv20 > 0 else np.nan,
                       bar_vol_x=v[i] / v[:i].mean() if i > 0 and v[:i].mean() > 0 else np.nan,
                       above_vwap=int(entry > vwap[i - 1]) if i > 0 else 1, gap_pct=(o0 / r.prev_close - 1) * 100 if r.prev_close == r.prev_close and r.prev_close > 0 else np.nan,
                       prev_range_pct=(r.prev_high - r.prev_low) / r.prev_close * 100 if r.prev_close == r.prev_close and r.prev_close > 0 else np.nan,
                       adv20=r.adv20, dist_20d_high_pct=(entry / r.high20 - 1) * 100 if r.high20 == r.high20 and r.high20 > 0 else np.nan,
                       spy_5m_ret=((spym[int(m[i])] / spym[int(m[i]) - 5]) - 1) * 100 if spym.get(int(m[i])) and spym.get(int(m[i]) - 5) else np.nan,
                       spy_range3=float(spyd.range3.get(day, np.nan)), pm_covered=int(pmh is not None), **extra, **ex)
            rows.append(row)
    if vprof:
        pd.DataFrame(vprof).to_csv(VPROF, mode='a', header=not os.path.exists(VPROF), index=False)
    return rows, missing


def main():
    global uni
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': [], 'missing': 0}
    done = set(state['done']); days = sorted(uni.bar_date.unique())
    only = os.environ.get('BFZ_DAYS')                      # benchmark: limit to N days
    if only: days = days[:int(only)]
    if RETRY:                                              # re-scan only the pairs logged as missing (bars fetched since)
        mp = pd.read_csv(MISS, dtype=str, keep_default_na=False).drop_duplicates()
        uni = uni.merge(mp, on=['symbol', 'bar_date']); days = sorted(uni.bar_date.unique()); done = set()
        os.replace(MISS, MISS + '.prev')
        print(f'RETRY mode: {len(uni)} missing pairs over {len(days)} days', flush=True)
    todo = [d for d in days if d not in done]
    print(f'universe days {len(days)} | done {len(done)} | todo {len(todo)} | fams {len(FAMS)} | slip {SLIP}', flush=True)
    t0 = time.time(); nrows = 0
    for k, day in enumerate(todo):
        sub = uni[uni.bar_date == day]
        rows, miss = build_day(day, sub)
        if rows:
            pd.DataFrame(rows).reindex(columns=COLS).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        if miss:
            pd.DataFrame({'symbol': miss, 'bar_date': day}).to_csv(MISS, mode='a', header=not os.path.exists(MISS), index=False)
        nrows += len(rows); state['done'].append(day); state['missing'] += len(miss); miss = len(miss)
        json.dump(state, open(STATE, 'w'))
        if k % 5 == 0 or k == len(todo) - 1:
            el = time.time() - t0
            print(f'{k + 1}/{len(todo)} {day} syms={len(sub)} rows+={len(rows)} miss={miss} | total rows {nrows} | {el / 60:.1f} min, {el / (k + 1):.1f} s/day', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
