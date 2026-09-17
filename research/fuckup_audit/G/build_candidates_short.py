#!/usr/bin/env python3
"""Stage G pass 1 — candidates_short.csv: one row per SHORT SIGNAL of the six mirrored families.

Everything here is the pre-registration in `G/PREREG.md` turned into code. The long-side machinery is
IMPORTED, never copied: `research/bf_zero/build_candidates.py` (bars loader contract, `first_true`,
`roll_min`/`roll_max`, `OPEN_M`/`EOD_M`) and `research/fuckup_audit/B/build_candidates4.py`
(`cc_bps` cost curve, `spread_pct`, `asset_class`, `_first`). The SHORT detectors and the SHORT exit
walk are written as mirrors of the long bodies — a short's stop is ABOVE the entry, it fires on
`high >= stop`, and the target fires on a bar CLOSE at/below `entry - 2R`.

THE SCORER CONTRACT (G/score_short.py must use exactly this):

  POPULATION      entry >= $10 (the FILL), entry_m <= 841 (14:01), r_pct >= 1.0; universe UA = the
                  borrowable U1uU2 of `G/members_g.csv` (in_u12 == 1), UB = UA restricted to
                  range_so_far_pct >= 5. S5 runs on the attention list (attn_grp == 'attention'),
                  the control group (attn_grp == 'control') is the reference book.
                  SSR days (ssr == 1) are EXCLUDED under the primary contract.
  COST            half = 0.5 * (spread_cc_bps / 100) / max(r_pct, 0.05)        [R units]
                  entry 0.25 * half (next-open fill) · stop 0.875 · eod 0.412 · cover 0.412 ·
                  target 0.875.  Borrow: PRIMARY 0 (ETB assumption); SENSITIVITY 5 bps locate +
                  1%/365 for one day = 5.274 bps of notional = (5.274/100)/r_pct R.
  BOOK            trading.hod_break.run_book(rows, 12, 4).
  GATE            PLAN.md §1: G1 TRAIN mean net R > 0, t >= 2.0, >= 5 trades/week; G2 VAL mean > 0,
                  t >= 1.0, >= 55% weeks green, bar + 1 SE of weekly R per 10 G1 cells; G3 TEST
                  read ONCE for G2 survivors only.

FAMILIES (6 configs)
  S1 {}            gap >= +5%: break of the 09:30-09:34 range LOW; stop = that range's HIGH
  S2 {"N":15}      ORB breakdown, level = min(low[0:15]), stop = max(high[0:15])
  S2 {"N":30}      same, N = 30
  S3 {}            green-to-red: open > prev_close, level = prev_close*0.997, stop = max high before i
  S4 {"K":5,"X":0.04}  HOD-rejection: F5's K/X consolidation, first bar CLOSING below the
                   consolidation low; level = that low, stop = the HOD
  S5 {}            attention fade: short the 09:35 bar's OPEN, stop = max(high[0:5])

FILL: `entry = open[i+1]` accepted iff `open[i+1] >= level * (1 - 0.006)` (no chase below the cap).
S5 fills at the 09:35 bar's own open (no level, no cap). Exits start the bar AFTER the fill bar.

BARS: ONE tape (Alpaca SIP, adjustment=raw, 1-minute) from three stores in this order —
`E/bars_causal/day=*/bars.parquet`, `research/bf_zero/bars_sip.db`,
`research/lit_review_2026/attention.db`. Every key with no usable tape is written to
`G/coverage_short_missing.csv`; nothing is silently dropped.

WRITES ONLY research/fuckup_audit/G/{candidates_short*.csv, build_short_state*.json,
coverage_short_missing*.csv}. Every other path is read-only.

RUN:
  setsid nohup bash -c "cd /home/ec2-user/onemil; ulimit -v 2600000; nice -n 10 \\
    python3 research/fuckup_audit/G/build_candidates_short.py \\
    > research/fuckup_audit/G/build_short.log 2>&1; \\
    echo EXIT=\\$? >> research/fuckup_audit/G/build_short.log" >/dev/null 2>&1 </dev/null &
  G_DAYS=3 G_TAG=_smoke python3 research/fuckup_audit/G/build_candidates_short.py     # smoke
"""
import gc
import json
import os
import sqlite3
import sys
import time

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
os.environ.setdefault('BFZ_SIP_STORE', f'{ROOT}/research/bf_zero/bars_sip.db')
os.environ['BFZ_SLIP'] = '0.0'
sys.path.insert(0, f'{ROOT}/research/bf_zero')
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/B')
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

G = f'{ROOT}/research/fuckup_audit/G'
STORE = f'{ROOT}/research/fuckup_audit/E/bars_causal'
SIP = f'{ROOT}/research/bf_zero/bars_sip.db'
ATTN = f'{ROOT}/research/lit_review_2026/attention.db'
MEMBERS = f'{G}/members_g.csv'
TAG = os.environ.get('G_TAG', '')
OUT = f'{G}/candidates_short{TAG}.csv'
STATE = f'{G}/build_short_state{TAG}.json'
MISS = f'{G}/coverage_short_missing{TAG}.csv'

CAP = 0.006            # no chase below level*(1-CAP)
GAP_MIN_S1 = 5.0       # S1 requires gap_pct >= +5%
COVER_M = 630          # 10:30 ET
STOP_SLIP = 1.001      # a short's stop fill is WORSE = higher
RISK_USD = 100.0


def log(m):
    print(f'{time.strftime("%H:%M:%S")} {m}', flush=True)


# ---------------------------------------------------------------- the thin import (Stage E's trick)
def _stub_import():
    """`build_candidates` loads a 3 GB Databento panel and the >=5%-range universe at module scope;
    neither is used here (members come from G/members_g.csv). Stub the two reads, then import
    build_candidates4 on top — its own `import build_candidates` is a cache hit."""
    _rp, _rc = pd.read_parquet, pd.read_csv

    def stub_parquet(path, *a, **k):
        if 'equs_daily' in str(path):
            return pd.DataFrame({'symbol': ['SPY'], 'bar_date': ['2025-01-02'], 'open': [1.0],
                                 'high': [1.0], 'low': [1.0], 'close': [1.0], 'volume': [1.0]})
        return _rp(path, *a, **k)

    def stub_csv(path, *a, **k):
        if str(path).endswith('bf_zero/universe.csv'):
            d = pd.DataFrame({'symbol': ['SPY'], 'bar_date': ['2025-01-02'], 'open': [1.0],
                              'high': [1.0], 'low': [1.0], 'close': [1.0], 'volume': [1.0],
                              'adv20': [1.0], 'prev_vol': [1.0]})
            uc = k.get('usecols')
            return d[list(uc)] if uc else d
        return _rc(path, *a, **k)

    pd.read_parquet, pd.read_csv = stub_parquet, stub_csv
    try:
        import build_candidates                      # noqa: F401
    finally:
        pd.read_parquet, pd.read_csv = _rp, _rc
    import build_candidates4 as B4
    gc.collect()
    return B4


B4 = _stub_import()
B = B4.B
OPEN_M, EOD_M = B.OPEN_M, B.EOD_M
_first = B4._first
cc_bps = B4.cc_bps
spread_pct = B4.spread_pct

FAMS = [('S1', {}), ('S2', dict(N=15)), ('S2', dict(N=30)), ('S3', {}),
        ('S4', dict(K=5, X=0.04)), ('S5', {})]

COLS = ['day', 'symbol', 'fam', 'cfg', 'split', 'sig_m', 'minutes_since_open',
        'level', 'stop', 'price', 'dist_open_pct', 'range_so_far_pct', 'rv_adv', 'gap_pct',
        'adv20', 'dvol20_med', 'spread_pct', 'spread_cc_bps',
        'sig_o', 'sig_h', 'sig_l', 'sig_c', 'sig_v',
        'n_touches', 'consol_bars', 'consol_vol_ratio', 'cum_dollar_vol', 'vwap_dist_pct',
        'close_confirm', 'prev_day_range_pct', 'prev_close', 'asset_class',
        'in_u12', 'u1', 'u2', 'attn_grp', 'attn_rank', 'ssr', 'prev_ret_cc',
        'entry', 'entry_m', 'r_pct', 'fside_entry', 'queue_ok',
        'rr_hold', 'why_hold', 'exit_m_hold', 'fside_hold',
        'rr_2r', 'why_2r', 'exit_m_2r', 'fside_2r',
        'rr_1030', 'why_1030', 'exit_m_1030', 'fside_1030',
        'ret_1030_bps', 'ret_eod_bps', 'mae_pct', 'mfe_r']


# ---------------------------------------------------------------- bars: one tape, three stores
_sip = sqlite3.connect(f'file:{SIP}?mode=ro', uri=True, timeout=180)
_attn = sqlite3.connect(f'file:{ATTN}?mode=ro', uri=True, timeout=180) if os.path.exists(ATTN) else None


def _minute_from_iso(s):
    ts = pd.to_datetime(s, utc=True, format='ISO8601').dt.tz_convert('America/New_York')
    return (ts.dt.hour * 60 + ts.dt.minute).values


def load_bars_short(day, syms):
    """{symbol: DataFrame(m,o,h,l,c,v)} from E/bars_causal, then bars_sip.db, then attention.db.
    Same contract as build_candidates.load_bars (ET minute of day, sorted, one row per minute)."""
    want = set(syms)
    out = {}
    p = f'{STORE}/day={day}/bars.parquet'
    if os.path.exists(p):
        import pyarrow.parquet as pq
        t = pq.read_table(p, columns=['symbol', 't', 'o', 'h', 'l', 'c', 'v'])
        d = t.to_pandas()
        del t
        d = d[d.symbol.isin(want)]
        for s, gg in d.groupby('symbol', sort=False):
            out[s] = gg[['t', 'o', 'h', 'l', 'c', 'v']].rename(columns={'t': 'm'})
        del d
    left = [s for s in syms if s not in out]
    if left:
        for i in range(0, len(left), 400):
            ch = left[i:i + 400]
            q = ('select symbol, t, o, h, l, c, v from bars where day=? and symbol in '
                 f'({",".join("?" * len(ch))})')
            d = pd.read_sql(q, _sip, params=[day] + ch)
            if len(d):
                d['m'] = _minute_from_iso(d.t)
                for s, gg in d.groupby('symbol', sort=False):
                    out[s] = gg[['m', 'o', 'h', 'l', 'c', 'v']]
            del d
    left = [s for s in syms if s not in out]
    if left and _attn is not None:
        for i in range(0, len(left), 400):
            ch = left[i:i + 400]
            q = ('select symbol, m, o, h, l, c, v from bars where day=? and symbol in '
                 f'({",".join("?" * len(ch))})')
            d = pd.read_sql(q, _attn, params=[day] + ch)
            if len(d):
                for s, gg in d.groupby('symbol', sort=False):
                    out[s] = gg[['m', 'o', 'h', 'l', 'c', 'v']]
            del d
    res = {}
    for s, gg in out.items():
        gg = gg.astype({'m': int}).sort_values('m').drop_duplicates('m')
        res[s] = gg[['m', 'o', 'h', 'l', 'c', 'v']].reset_index(drop=True)
    return res


# ---------------------------------------------------------------- SHORT exit walks
def walk_short(o, h, l, c, m, k0, stop, target, eod_idx):
    """Mirror of build_candidates4.walk_2r with every inequality flipped.
    Within one bar: eod beats stop beats target. Stop fires on `high >= stop` and fills at
    `max(stop, open) * 1.001`; the target fires on a bar CLOSE <= target and fills AT it.
    Returns (px, why, exit_m, exit_idx)."""
    if k0 >= len(o):
        return float(c[-1]), 'eod', int(m[-1]), len(o) - 1
    cand = []
    if eod_idx is not None:
        cand.append((max(eod_idx, k0), 0, 'eod'))
    s = _first(h >= stop, k0)
    if s is not None:
        cand.append((s, 1, 'stop'))
    if target is not None:
        t = _first(c <= target, k0)
        if t is not None:
            cand.append((t, 2, 'target'))
    if not cand:
        return float(c[-1]), 'eod', int(m[-1]), len(o) - 1
    k, _, why = min(cand)
    px = (float(o[k]) if why == 'eod'
          else (float(max(stop, o[k]) * STOP_SLIP) if why == 'stop' else float(target)))
    return px, why, int(m[k]), k


def walk_timed(o, h, l, c, m, k0, stop, cover_idx, eod_idx):
    """The 10:30 cover: identical machinery with the horizon moved forward; the stop still applies."""
    ci = cover_idx if cover_idx is not None else eod_idx
    if ci is not None and eod_idx is not None:
        ci = min(ci, eod_idx)
    px, why, xm, k = walk_short(o, h, l, c, m, k0, stop, None, ci)
    return px, ('cover' if why == 'eod' else why), xm, k


def fill_side(px, l, h, k):
    """PLAN §1 obtainability, with the SIDE recorded: 0 = inside the filling bar (obtainable),
    +1 = above that bar's high, -1 = below its low. For a SHORT, a cover above the high is a price
    the market never printed but is WORSE than anything it did print (conservative); a cover below
    the low is FAVOURABLE and impossible — that is the only defect, and its share is reported."""
    if px > h[k] + 1e-9:
        return 1
    if px < l[k] - 1e-9:
        return -1
    return 0


# ---------------------------------------------------------------- SHORT detectors (mirrors)
def sfam_level(l, level, start_idx, stop_hi):
    """Mirror of build_candidates.fam_level: first bar at/after start_idx whose LOW breaks `level`."""
    n = len(l)
    t = np.arange(n)
    i = B.first_true((t >= start_idx) & (l <= level))
    if i is None or stop_hi <= level:
        return None
    return int(i), float(level), float(stop_hi)


def sfam_g2r(h, l, o0, prev_close):
    """S3, mirror of fam_r2g: opened ABOVE the prior close, first break BELOW prev_close*0.997;
    stop = the highest high strictly before the signal bar."""
    if not (prev_close and prev_close == prev_close and o0 > prev_close):
        return None
    level = float(prev_close) * 0.997
    n = len(l)
    t = np.arange(n)
    i = B.first_true((t >= 1) & (l <= level))
    if i is None:
        return None
    stop = float(np.maximum.accumulate(h)[i - 1])
    if stop <= level:
        return None
    return int(i), level, stop


def sfam_hod_reject(h, l, c, K, X):
    """S4, F5's consolidation entered the other way: K bars all within X of the running HOD, then the
    first bar whose CLOSE is below the consolidation low. level = that low, stop = the HOD."""
    n = len(h)
    hod = np.maximum.accumulate(h)
    lo = B.roll_min(l, K)
    t = np.arange(n)
    j = np.clip(t - 1, 0, n - 1)
    ok = (t > K) & (lo[j] >= hod[j] * (1 - X)) & (lo[j] < hod[j]) & (c < lo[j])
    i = B.first_true(ok)
    if i is None:
        return None
    level, stop = float(lo[i - 1]), float(hod[i - 1])
    if stop <= level:
        return None
    return int(i), level, stop


def detect_short(fam, cfg, arrays):
    """(i, level, stop, lvl_idx) or None. lvl_idx = the bar the level was set on."""
    o, h, l, c, v, m, o0, vwap, prev_close, gap_pct = arrays
    n = len(h)
    if fam == 'S1':
        if not (gap_pct == gap_pct and gap_pct >= GAP_MIN_S1 and n > 5):
            return None
        r = sfam_level(l, float(l[:5].min()), 5, float(h[:5].max()))
        return None if r is None else (r[0], r[1], r[2], 4)
    if fam == 'S2':
        N = cfg['N']
        if n <= N:
            return None
        r = sfam_level(l, float(l[:N].min()), N, float(h[:N].max()))
        return None if r is None else (r[0], r[1], r[2], N - 1)
    if fam == 'S3':
        r = sfam_g2r(h, l, o0, prev_close)
        return None if r is None else (r[0], r[1], r[2], 0)
    if fam == 'S4':
        r = sfam_hod_reject(h, l, c, cfg['K'], cfg['X'])
        return None if r is None else (r[0], r[1], r[2], max(r[0] - cfg['K'], 0))
    if fam == 'S5':
        w = np.flatnonzero(m == OPEN_M + 5)
        if not len(w) or n <= 5:
            return None
        i = int(w[0])
        if i < 1:
            return None
        stop = float(h[:i].max())
        entry_lvl = float(o[i])
        if stop <= entry_lvl:
            return None
        return i, entry_lvl, stop, max(i - 5, 0)
    return None


# ---------------------------------------------------------------- the day loop
def fill_block(o, h, l, c, m, entry, stop, price, fill_idx, cover_idx, eod_idx):
    """Every exit column for the one fill model. Exits start the bar AFTER the fill bar."""
    k0 = fill_idx + 1
    R = stop - entry
    d = {'entry': entry, 'entry_m': int(m[fill_idx]), 'r_pct': R / entry * 100.0,
         'fside_entry': fill_side(entry, l, h, fill_idx)}
    px, why, xm, kx = walk_short(o, h, l, c, m, k0, stop, None, eod_idx)
    d['rr_hold'], d['why_hold'], d['exit_m_hold'] = (entry - px) / R, why, xm
    d['fside_hold'] = fill_side(px, l, h, kx)
    px2, why2, xm2, kx2 = walk_short(o, h, l, c, m, k0, stop, entry - 2 * R, eod_idx)
    d['rr_2r'], d['why_2r'], d['exit_m_2r'] = (entry - px2) / R, why2, xm2
    d['fside_2r'] = fill_side(px2, l, h, kx2)
    px3, why3, xm3, kx3 = walk_timed(o, h, l, c, m, k0, stop, cover_idx, eod_idx)
    d['rr_1030'], d['why_1030'], d['exit_m_1030'] = (entry - px3) / R, why3, xm3
    d['fside_1030'] = fill_side(px3, l, h, kx3)
    # the pure M18 spec: no stop at all, in bps of the entry
    ci = cover_idx if cover_idx is not None else eod_idx
    if ci is not None and eod_idx is not None:
        ci = min(ci, eod_idx)
    n = len(o)

    def px_at(k):
        """the cover print at horizon k: that bar's open, or the last close if the tape ends first"""
        return float(o[k]) if k < n else float(c[-1])

    ci = max(ci, k0) if ci is not None else None
    d['ret_1030_bps'] = ((entry - px_at(ci)) / entry * 1e4) if ci is not None else ''
    ei = max(eod_idx, k0) if eod_idx is not None else None
    d['ret_eod_bps'] = ((entry - px_at(ei)) / entry * 1e4) if ei is not None else ''
    hi_seg = float(h[k0:kx2 + 1].max()) if kx2 >= k0 else entry
    lo_seg = float(l[k0:kx2 + 1].min()) if kx2 >= k0 else entry
    d['mae_pct'] = max(0.0, (hi_seg - entry) / price * 100.0)
    d['mfe_r'] = max(0.0, (entry - lo_seg) / R)
    return d


def build_day(day, sub):
    bars = load_bars_short(day, sorted(sub.symbol.unique().tolist()))
    rows, missing = [], []
    for r in sub.itertuples():
        gg = bars.get(r.symbol)
        if gg is None:
            missing.append(r.symbol)
            continue
        rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
        if len(rth) < 10:
            missing.append(r.symbol)
            continue
        o, h, l, c, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v'))
        m = rth.m.values.astype(int)
        o0 = o[0]
        cumv = np.cumsum(v)
        cumdv = np.cumsum(c * v)
        vwap = cumdv / np.maximum(cumv, 1)
        run_hi = np.maximum.accumulate(h)
        run_lo = np.minimum.accumulate(l)
        prev_close = r.prev_close if r.prev_close == r.prev_close else np.nan
        gap = ((o0 / prev_close - 1) * 100.0
               if (prev_close == prev_close and prev_close > 0) else np.nan)
        w = np.flatnonzero(m >= EOD_M)
        eod_idx = int(w[0]) if len(w) else None
        w = np.flatnonzero(m >= COVER_M)
        cover_idx = int(w[0]) if len(w) else eod_idx
        arrays = (o, h, l, c, v, m, o0, vwap, prev_close, gap)
        for fam, cfg in FAMS:
            if fam == 'S5':
                if not r.attn_grp:
                    continue
            elif int(r.in_u12) != 1:
                continue
            res = detect_short(fam, cfg, arrays)
            if res is None:
                continue
            i, level, stop, lvl_idx = res
            if i < 1 or stop <= level or not (level > 0):
                continue
            price = float(level)
            nt = int(((l[:i] <= level * 1.002) & (l[:i] > level)).sum())
            cb = max(int(i - lvl_idx), 0)
            seg = v[lvl_idx:i] if i > lvl_idx else v[:i]
            mv = float(seg.mean()) if len(seg) and seg.mean() > 0 else np.nan
            d = dict(day=day, symbol=r.symbol, fam=fam, cfg=json.dumps(cfg, sort_keys=True),
                     split=r.split, sig_m=int(m[i]), minutes_since_open=int(m[i]) - OPEN_M,
                     level=price, stop=float(stop), price=price,
                     dist_open_pct=(price / o0 - 1) * 100.0,
                     range_so_far_pct=(run_hi[i - 1] - run_lo[i - 1]) / o0 * 100.0,
                     rv_adv=cumv[i] / r.adv20 if (r.adv20 == r.adv20 and r.adv20 > 0) else np.nan,
                     gap_pct=gap, adv20=r.adv20, dvol20_med=r.dvol20_med,
                     spread_pct=spread_pct(price), spread_cc_bps=cc_bps(price, int(m[i])),
                     sig_o=float(o[i]), sig_h=float(h[i]), sig_l=float(l[i]), sig_c=float(c[i]),
                     sig_v=float(v[i]), n_touches=nt, consol_bars=cb,
                     consol_vol_ratio=float(v[i]) / mv if mv == mv else np.nan,
                     cum_dollar_vol=float(cumdv[i]),
                     vwap_dist_pct=(price / vwap[i - 1] - 1) * 100.0,
                     close_confirm=int(c[i] <= level),
                     prev_day_range_pct=r.prev_day_range_pct, prev_close=prev_close,
                     asset_class=r.asset_class, in_u12=int(r.in_u12), u1=int(bool(r.u1)),
                     u2=int(bool(r.u2)), attn_grp=r.attn_grp,
                     attn_rank=r.attn_rank, ssr=int(r.ssr), prev_ret_cc=r.prev_ret_cc)
            if fam == 'S5':
                fill_idx, e = i, float(o[i])            # short the 09:35 bar's own open
            else:
                if i + 1 >= len(o):
                    continue
                fill_idx, e = i + 1, float(o[i + 1])
                if e < level * (1 - CAP):               # gapped through: NO chase below the cap
                    rows.append(d)
                    continue
            if stop <= e:
                rows.append(d)
                continue
            fb = fill_block(o, h, l, c, m, e, float(stop), price, fill_idx, cover_idx, eod_idx)
            fb['queue_ok'] = int(v[fill_idx] >= 5.0 * (RISK_USD / (stop - e)))
            d.update(fb)
            rows.append(d)
    return rows, missing


def main():
    mem = pd.read_csv(MEMBERS, dtype={'symbol': str, 'bar_date': str, 'split': str,
                                      'attn_grp': str, 'asset_class': str},
                      keep_default_na=False, na_values=[''])
    for c in ('open', 'prev_close', 'prev_high', 'prev_low', 'gap_pct', 'prev_day_range_pct',
              'adv20', 'dvol20_med', 'attn_rank', 'prev_ret_cc'):
        mem[c] = pd.to_numeric(mem[c], errors='coerce')
    for c in ('u1', 'u2'):
        mem[c] = mem[c].astype(str).str.lower().isin(('true', '1'))
    # `na_values=['']` turns an empty attn_grp into NaN, and `not NaN` is False — without this the
    # S5 branch's `if not r.attn_grp: continue` never fires and S5 is built for every member.
    mem['attn_grp'] = mem.attn_grp.fillna('').astype(str)
    mem['in_u12'] = pd.to_numeric(mem.in_u12, errors='coerce').fillna(0).astype(int)
    mem['ssr'] = pd.to_numeric(mem.ssr, errors='coerce').fillna(0).astype(int)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    done = set(state['done'])
    days = sorted(mem.bar_date.unique())
    if os.environ.get('G_DAYS'):
        want = os.environ.get('G_DAY_LIST')
        days = want.split(',') if want else days[:int(os.environ['G_DAYS'])]
    todo = [d for d in days if d not in done]
    log(f'candidates_short | days {len(days)} | done {len(done)} | todo {len(todo)} | '
        f'fam-configs {len(FAMS)} | members {len(mem):,} | out {OUT}')
    t0 = time.time()
    n = 0
    for k, day in enumerate(todo):
        sub = mem[mem.bar_date == day]
        rows, miss = build_day(day, sub)
        if rows:
            pd.DataFrame(rows).reindex(columns=COLS).to_csv(
                OUT, mode='a', header=not os.path.exists(OUT), index=False)
        if miss:
            pd.DataFrame({'symbol': miss, 'bar_date': day}).to_csv(
                MISS, mode='a', header=not os.path.exists(MISS), index=False)
        n += len(rows)
        state['done'].append(day)
        json.dump(state, open(STATE, 'w'))
        el = time.time() - t0
        log(f'{k + 1}/{len(todo)} {day} keys {len(sub)} rows+={len(rows)} miss={len(miss)} '
            f'total {n:,} | {el / 60:.1f} min, {el / (k + 1):.1f} s/day, '
            f'ETA {(len(todo) - k - 1) * el / (k + 1) / 60:.0f} min')
    log(f'DONE rows {n:,}')


if __name__ == '__main__':
    main()
