#!/usr/bin/env python3
"""frames20 F57 — walk the POWERED matched-non-signal control P1x5 of the passive mirror short.

PREREG frames20/PREREG.md (frozen; FREEZE.md carries the git hash).  S is frames18/grid18.csv at
k=0.010 and is NOT re-walked here.  P3 is not walked (F56 answered it).  Entry/exit/cost stack
identical to S and to frames19 (PREREG §2).  TEST sealed.

Only change vs frames19/walk19.py: NMATCH 3 -> 5, SEED 19 -> 57, P3 removed, bucket-size /
shortfall bookkeeping added.
"""
import glob
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames16')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_frames5')
from common5 import attach_instrument                                   # noqa: E402
from short_walk import swalk                                            # noqa: E402

D = f'{ROOT}/research/mature_method/frames20'
D16 = f'{ROOT}/research/mature_method/frames16'
SIP = f'{ROOT}/research/bf_zero/bars_sip.db'
CACHE = f'{ROOT}/data/cache.db'
BORROW = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv'
TEST_FROM = '2026-06-01'
K = 0.010
WMAX = 10
NMATCH = 5          # PREREG §1 — the ONLY design change vs frames19
SEED = 57


def etb_map():
    b = pd.read_csv(BORROW, dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    b['etb'] = b.shortable.astype(str).str.lower().eq('true') & \
        b.easy_to_borrow.astype(str).str.lower().eq('true')
    return dict(zip(b.symbol, b.etb))


def sw_rows():
    fs = sorted(glob.glob(f'{D16}/sw_*.csv'))
    d = pd.concat([pd.read_csv(f, dtype={'symbol': str, 'day': str},
                               keep_default_na=False, na_values=['']) for f in fs],
                  ignore_index=True)
    return d[d.day < TEST_FROM].copy()


def population(sw, etb):
    d = sw[sw.gate5.astype(bool) & sw.f_mir2.astype(bool) & (sw.price >= 5)].copy()
    d = attach_instrument(d)
    d = d[d.asset_class != 'wrapper']
    d['etb'] = d.symbol.map(etb).fillna(False).astype(bool)
    d = d[d.etb].copy()
    d['split'] = np.where(d.day < '2026-01-01', 'TRAIN', 'VAL')
    print(f'[S] {len(d):,} signal rows, {d.day.nunique()} sessions', flush=True)
    return d


def daily_panel(symbols):
    """prev_close and adv20 (trailing-20 mean share volume, both CAUSAL: shifted by one session)."""
    syms = sorted(set(symbols))
    con = sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True, timeout=300)
    frames = []
    CH = 800
    for i in range(0, len(syms), CH):
        ch = syms[i:i + CH]
        qs = ','.join('?' * len(ch))
        frames.append(pd.read_sql(
            f"select symbol,bar_date,close,volume from daily_bars where symbol in ({qs}) "
            f"and bar_date >= '2024-10-01' and bar_date < '{TEST_FROM}'", con, params=ch))
    con.close()
    db = pd.concat(frames, ignore_index=True)
    db['bar_date'] = db.bar_date.astype(str).str.slice(0, 10)
    db = db.drop_duplicates(['symbol', 'bar_date']).sort_values(['symbol', 'bar_date'],
                                                               kind='mergesort')
    g = db.groupby('symbol', sort=False)
    db['prev_close'] = g['close'].shift(1)
    db['adv20'] = g['volume'].transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    db = db.dropna(subset=['prev_close', 'adv20'])
    db = db[db.adv20 > 0]
    print(f'[daily] {len(db):,} causal (day,symbol) price/ADV rows for {db.symbol.nunique():,} '
          f'symbols', flush=True)
    return {(d_, s): (float(p), float(a))
            for s, d_, p, a in zip(db.symbol, db.bar_date, db.prev_close, db.adv20)}


def hour_rows(mv, o, hi, lo, c, hour):
    """ref / entry bar for one hour boundary, exactly the frames16 sw convention."""
    cut = (hour + 1) * 60
    pre = mv < cut
    if not pre.any():
        return None
    nxt = np.flatnonzero(mv >= cut)
    if not len(nxt):
        return None
    e = int(nxt[0])
    if mv[e] > cut + 5:
        return None
    ref = float(c[pre][-1])
    entry_o = float(o[e])
    if ref <= 0 or entry_o <= 0:
        return None
    gate5 = bool(hi[pre].max() >= float(o[0]) * 1.05)
    return dict(e=e, ref=ref, price=entry_o, gate5=gate5)


def passive(mv, o, hi, lo, c, e, ref, pc):
    limit = ref * (1.0 + K)
    fj = None
    for off in range(WMAX):
        j = e + off
        if j >= len(mv):
            break
        if hi[j] >= limit:
            fj = j
            break
    base = dict(ref=ref, limit=limit)
    if fj is None:
        base.update(touch_off=-1, fill_m=-1, entry=np.nan, rpct=np.nan, rr=np.nan,
                    why='unfilled', exit_m=-1, ssr_active=False, ssr_undet=(pc is None))
        return base
    entry = limit
    stop = entry * 1.02
    rr, why, xm = swalk(o, hi, lo, c, mv, fj, entry, stop, 'bare')
    runmin = np.minimum.accumulate(lo)
    base.update(touch_off=fj - e, fill_m=int(mv[fj]), entry=entry, rpct=(stop - entry) / entry,
                rr=rr, why=why, exit_m=int(xm),
                ssr_active=bool(pc is not None and runmin[fj] <= pc * 0.90),
                ssr_undet=(pc is None))
    return base


def run():
    etb = etb_map()
    sw = sw_rows()
    S = population(sw, etb)
    assert S.day.max() < TEST_FROM

    sw_syms_by_day = sw.groupby('day').symbol.apply(lambda s: set(s)).to_dict()

    con = sqlite3.connect(f'file:{SIP}?mode=ro', uri=True, timeout=300)
    pool_by_day = {}
    for day in sorted(S.day.unique()):
        p = pd.read_sql('select symbol from fetch_log where day=? and n_bars>0', con,
                        params=(day,)).symbol.astype(str).tolist()
        pool_by_day[day] = p
    allsyms = set(S.symbol) | {s for v in pool_by_day.values() for s in v}
    print(f'[pool] {len(allsyms):,} distinct symbols across {len(pool_by_day)} sessions', flush=True)
    panel = daily_panel(allsyms)

    cls = attach_instrument(pd.DataFrame({'symbol': sorted(allsyms), 'day': '2025-01-02'}))
    wrapper = set(cls.symbol[cls.asset_class == 'wrapper'])

    rng = np.random.default_rng(SEED)
    p1_rows = []
    stat = dict(s_rows=0, matched=0, no_match=0, p1_drawn=0, p1_kept=0, shortfall=0,
                short_rows=0)
    bucket_sizes = []
    days = sorted(S.day.unique())
    for di, day in enumerate(days):
        g = S[S.day == day]
        banned = sw_syms_by_day.get(day, set())
        pool = [s for s in pool_by_day[day]
                if s not in banned and s not in wrapper and etb.get(s, False)
                and not pd.Series([s]).str.match(r'^Z[A-Z]ZZT$').iloc[0]
                and (day, s) in panel]
        pcs = np.array([panel[(day, s)][0] for s in pool]) if pool else np.zeros(0)
        advs = np.array([panel[(day, s)][1] for s in pool]) if pool else np.zeros(0)
        want = {}                                      # symbol -> list of (hour, sig_index)
        for r in g.itertuples():
            stat['s_rows'] += 1
            ps = panel.get((day, r.symbol))
            if ps is None or not len(pool):
                stat['no_match'] += 1
                continue
            ok = (np.abs(np.log(pcs / ps[0])) <= np.log(1.25)) & \
                 (np.abs(np.log2(advs / ps[1])) <= 1.0)
            cand = [pool[i] for i in np.flatnonzero(ok)]
            if not cand:
                stat['no_match'] += 1
                continue
            stat['matched'] += 1
            bucket_sizes.append(len(cand))
            if len(cand) < NMATCH:
                stat['shortfall'] += NMATCH - len(cand)
                stat['short_rows'] += 1
            pick = rng.choice(cand, size=min(NMATCH, len(cand)), replace=False)
            for s in pick:
                want.setdefault(str(s), []).append((int(r.hour), r.Index))
                stat['p1_drawn'] += 1

        need = sorted(set(want))
        if not need:
            continue
        qs = ','.join('?' * len(need))
        b = pd.read_sql(f'select symbol,t,o,h,l,c from bars where day=? and symbol in ({qs})',
                        con, params=[day] + need)
        if b.empty:
            continue
        off = 4 if ('2025-03-09' <= day < '2025-11-02') or ('2026-03-08' <= day < '2026-11-01') \
            else 5
        b['m'] = (b.t.str.slice(11, 13).astype(int) - off) * 60 + b.t.str.slice(14, 16).astype(int)
        b = b[(b.m >= 570) & (b.m < 960)].sort_values(['symbol', 'm'], kind='mergesort')
        byx = {s: x for s, x in b.groupby('symbol', sort=False)}

        for sym, jobs in want.items():
            x = byx.get(sym)
            if x is None or len(x) < 20:
                continue
            mv = x.m.values
            o, hi, lo, c = (x[k].values.astype(float) for k in ('o', 'h', 'l', 'c'))
            pc = panel.get((day, sym), (None, None))[0]
            for hour, sig_ix in jobs:
                hr = hour_rows(mv, o, hi, lo, c, hour)
                if hr is None or not hr['gate5'] or hr['price'] < 5:
                    continue
                rec = dict(pop='P1', day=day, symbol=sym, hour=hour, price=hr['price'],
                           entry_m=int(mv[hr['e']]), sig_ix=int(sig_ix))
                rec.update(passive(mv, o, hi, lo, c, hr['e'], hr['ref'], pc))
                p1_rows.append(rec)
                stat['p1_kept'] += 1
        if di % 40 == 0:
            print(f'  {di+1}/{len(days)} {day} -> P1 {len(p1_rows):,}', flush=True)
    con.close()

    z = pd.DataFrame(p1_rows)
    z['split'] = np.where(z.day < '2026-01-01', 'TRAIN', 'VAL')
    assert z.day.max() < TEST_FROM
    z.to_csv(f'{D}/p120.csv', index=False)
    bs = np.array(bucket_sizes)
    print(f'[p1] {len(z):,} rows, {z.day.nunique()} sessions, touch rate (w=10) '
          f'{float((z.touch_off >= 0).mean())*100:.1f}%', flush=True)
    print(f'[match] S rows {stat["s_rows"]:,}; matched {stat["matched"]:,} '
          f'({stat["matched"]/max(stat["s_rows"],1)*100:.1f}%); no match {stat["no_match"]:,}; '
          f'P1 drawn {stat["p1_drawn"]:,}; kept after gate5/price rails {stat["p1_kept"]:,}',
          flush=True)
    print(f'[shortfall] S rows with bucket < {NMATCH}: {stat["short_rows"]:,} '
          f'({stat["short_rows"]/max(stat["matched"],1)*100:.1f}% of matched); total missing draws '
          f'{stat["shortfall"]:,}; bucket size p10/p50/p90 '
          f'{np.percentile(bs,10):.0f}/{np.percentile(bs,50):.0f}/{np.percentile(bs,90):.0f}, '
          f'mean {bs.mean():.1f}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(run())
