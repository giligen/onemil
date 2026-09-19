#!/usr/bin/env python3
"""hod_frames — THE bar pass for Frame 1 (the short book) and Frame 2 (the noon range).

Emits
  short.csv  one row per (B2 pre-book signal x trigger x short-stop design), carrying the short's
             entry, its R, and the exit under each of the three declared targets.
  range.csv  one row per (symbol, day) in the union of the Frame-1 and Frame-2 populations:
             the 09:30->T session range for T in {11:00, 12:00, 13:00} and the RTH day range.

TRAIN + VAL days only -- TEST is sealed (`FREEZE.md`).  Read-only on every DB.  Resumable per day.
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S            # noqa: E402
import score2 as S2          # noqa: E402
from pass2 import load_bars  # noqa: E402

D = f'{ROOT}/research/mature_method/hod_frames'
OPEN_M, EOD_M, LAST_M = 570, 955, 840
SLIP, CAP = 0.001, 0.006
TS = (660, 720, 780)                      # 11:00, 12:00, 13:00 ET
DAY_LO, DAY_HI = '2025-01-02', '2026-05-31'      # TRAIN+VAL -- TEST sealed
ARM_R, WIN_M = 0.4, 15
TRIGS = (('a01', 0.1), ('a00', 0.0), ('am1', -0.1))
TGTS = ('t1', 't2', 'tls')

SCOLS = (['day', 'symbol', 'entry_m', 'level', 'fill', 'r_long', 'long_stop',
          'trig', 'stop_var', 'trig_m', 'trig_c', 'short_m', 'short_px', 'obtainable',
          'mfe_high', 'short_stop', 'r_short', 'r_pct_s']
         + [f'{p}_{t}' for t in TGTS for p in ('rr', 'why', 'exit_m')])
RCOLS = ['day', 'symbol', 'open_px'] + [f'rng_{t}' for t in TS] + ['rng_day']


def swalk(o, h, l, c, m, s0, stop, target):
    """The short exit walk from bar index s0 (the first bar AFTER the short's fill bar).
    Sign-flipped twin of hod_fresh/pass3.walk: priority eod -> stop -> target."""
    n = len(o)
    if s0 >= n:
        return n - 1, float(c[-1]), 'eod'
    eod = m[s0:] >= EOD_M
    hs = h[s0:] >= stop
    ht = c[s0:] <= target
    any_ = eod | hs | ht
    if not any_.any():
        return n - 1, float(c[-1]), 'eod'
    j = int(np.argmax(any_)); k = s0 + j
    if eod[j]:
        return k, float(o[k]), 'eod'
    if hs[j]:
        return k, float(max(stop, o[k]) * (1.0 + SLIP)), 'stop'
    return k, float(target), 'target'


def main():
    pop = S2.load_pop()
    S.build_impute(pop)
    sig = S2.sig_set(pop, **S2.BASES['B2'])
    sig = sig[(sig.day >= DAY_LO) & (sig.day <= DAY_HI)]
    print(f'Frame-1 population: B2 pre-book {len(sig)} signals, {sig.day.nunique()} days', flush=True)

    # Frame-2 candidate symbol-days: any break row after the earliest T passing the B2 DETECTION
    # gates (the cost gates are applied in the scorer, not here).
    f2 = pop[(pop.stop_n.notna()) & (pop.dist_open_pct >= 5.0) &
             (pop.entry_m > min(TS)) & (pop.entry_m <= LAST_M + 1) &
             (pop.day >= DAY_LO) & (pop.day <= DAY_HI)]
    print(f'Frame-2 candidate break rows after {min(TS)}: {len(f2)}', flush=True)

    need = pd.concat([sig[['day', 'symbol']], f2[['day', 'symbol']]]).drop_duplicates()
    print(f'symbol-days needing bars: {len(need)} over {need.day.nunique()} days', flush=True)

    st = f'{D}/walk_state.json'
    state = json.load(open(st)) if os.path.exists(st) else {'done': []}
    done = set(state['done'])
    days = [d for d in sorted(need.day.unique()) if d not in done]
    SG = {d: g for d, g in sig.groupby('day')}
    NE = {d: sorted(g.symbol.unique()) for d, g in need.groupby('day')}

    for nd, day in enumerate(days):
        syms = NE[day]
        bars = load_bars(day, syms)
        rrows, srows = [], []
        cache = {}
        for sym in syms:
            gg = bars.get(sym)
            if gg is None:
                continue
            rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)]
            if len(rth) < 10:
                continue
            m = rth.m.values; o = rth.o.values.astype(float); h = rth.h.values.astype(float)
            l = rth.l.values.astype(float); c = rth.c.values.astype(float)
            cache[sym] = (m, o, h, l, c)
            op = float(o[0])
            r = dict(day=day, symbol=sym, open_px=op)
            for T in TS:
                w = m < T
                r[f'rng_{T}'] = float((h[w].max() - l[w].min()) / op * 100) if w.any() and op > 0 else np.nan
            r['rng_day'] = float((h.max() - l.min()) / op * 100) if op > 0 else np.nan
            rrows.append(r)

        for rw in SG.get(day, pd.DataFrame()).itertuples():
            cc = cache.get(rw.symbol)
            if cc is None:
                continue
            m, o, h, l, c = cc
            idx = np.nonzero(m == rw.entry_m)[0]
            if not len(idx):
                continue
            i0 = int(idx[0])
            fill = float(rw.next_open); lst = float(rw.stop); rl = fill - lst
            if rl <= 0:
                continue
            level = float(rw.level)
            hi = np.maximum.accumulate(h[i0:])            # MFE high through each bar, fill bar incl.
            armed = hi >= fill + ARM_R * rl
            cs = c[i0:]; ms = m[i0:]
            base = dict(day=day, symbol=rw.symbol, entry_m=int(rw.entry_m), level=level,
                        fill=fill, r_long=rl, long_stop=lst)
            for tname, k in TRIGS + (('fb', None),):
                if k is None:
                    hit = np.nonzero(cs < level)[0]
                else:
                    hit = np.nonzero(armed & (cs <= fill + k * rl) & (ms - rw.entry_m <= WIN_M))[0]
                if not len(hit):
                    continue
                j = int(hit[0]) + i0
                if j + 1 >= len(m) or m[j + 1] >= EOD_M:
                    continue
                epx = float(o[j + 1]); tc = float(c[j])
                obt = bool(epx >= tc * (1.0 - CAP))
                mh = float(hi[j - i0])
                svars = (('mfe', mh),) if tname != 'fb' else \
                        (('mfe', mh), ('h05', level + 0.5 * rl), ('h10', level + 1.0 * rl))
                for sv, sp in svars:
                    rs = sp - epx
                    row = dict(base, trig=tname, stop_var=sv, trig_m=int(m[j]), trig_c=tc,
                               short_m=int(m[j + 1]), short_px=epx, obtainable=int(obt),
                               mfe_high=mh, short_stop=float(sp), r_short=float(rs),
                               r_pct_s=float(rs / epx * 100) if epx > 0 else np.nan)
                    for tg in TGTS:
                        tp = (epx - rs) if tg == 't1' else (epx - 2 * rs) if tg == 't2' else lst
                        if rs <= 0 or tp >= epx:
                            row[f'rr_{tg}'] = np.nan; row[f'why_{tg}'] = ''; row[f'exit_m_{tg}'] = np.nan
                            continue
                        k2, px, why = swalk(o, h, l, c, m, j + 2, sp, tp)
                        row[f'rr_{tg}'] = float((epx - px) / rs)
                        row[f'why_{tg}'] = why
                        row[f'exit_m_{tg}'] = int(m[k2])
                    srows.append(row)

        for path, rows, cols in ((f'{D}/range.csv', rrows, RCOLS), (f'{D}/short.csv', srows, SCOLS)):
            if rows:
                pd.DataFrame(rows)[cols].to_csv(path, mode='a', header=not os.path.exists(path),
                                                index=False)
        done.add(day)
        json.dump({'done': sorted(done)}, open(st, 'w'))
        if nd % 20 == 0 or nd == len(days) - 1:
            print(f'  [{nd + 1}/{len(days)}] {day}  range+{len(rrows)}  short+{len(srows)}', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
