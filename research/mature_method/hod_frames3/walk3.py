#!/usr/bin/env python3
"""hod_frames3 / F11 — THE BAR PASS.

Emits, for EVERY candidate break bar of every symbol-day that carries at least one qualifying
break in TRAIN+VAL, the four F11 decomposition fields, all computed from bars strictly BEFORE the
break bar `i` (so: causal at the decision instant `m[i+1]`):

  * `hl_n20`      count of higher lows among the 20 bars ending at `i-1`   -- the rising-lows
                  sequence, mechanism (a)
  * `lo_slope20`  OLS slope of those 20 lows vs bar index, in % of `level` per bar -- (a)
  * `atr_now`     mean 1-min true range over bars [i-14, i-1]              -- (b)
  * `atr_prev`    mean 1-min true range over bars [i-28, i-15]             -- (b)
  * `atr_ratio`   atr_now / atr_prev  (< 1 = the coil compressing into the break)

It does NOT re-simulate exits: the exits, the stop and the qualification live in
`hod_frames2/breaks2.csv` (one row per QUALIFYING break, already validated by that pass's exact
independent rebuild).  This pass only ATTACHES fields, keyed on (day, symbol, break_m).

TRAIN + VAL days only -- TEST is sealed (FREEZE.md).  Read-only on every DB.  Resumable per day.
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
from pass2 import load_bars                                          # noqa: E402  (byte-parity loader)

D = f'{ROOT}/research/mature_method/hod_frames3'
STATE, OUT = f'{D}/walk_state.json', f'{D}/feat3.csv'
OPEN_M, LAST_M = 570, 930
MIN_LEVEL, K_B = 1.0, 5
DAY_LO, DAY_HI = '2025-01-02', '2026-05-31'          # TRAIN+VAL -- TEST is sealed
LIMIT_DAYS = int(os.environ.get('MM_LIMIT_DAYS', '0'))
COLS = ['day', 'symbol', 'break_m', 'hl_n20', 'lo_slope20', 'atr_now', 'atr_prev', 'atr_ratio']


def day_rows(day, syms, bars):
    """The F11 fields at every candidate break bar of every symbol on `day`."""
    rows = []
    for s in syms:
        gg = bars.get(s)
        if gg is None:
            continue
        rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
        if len(rth) < 10:
            continue
        h, l, c = (rth[k].values.astype(float) for k in ('h', 'l', 'c'))
        m = rth.m.values.astype(int)
        n = len(h)
        o0 = float(rth.o.values[0])
        if not (o0 > 0):
            continue
        hod = np.maximum.accumulate(h)
        # --- precomputed cumulative sums (all strictly-prior windows) -----------------
        pc = np.empty(n); pc[0] = c[0]; pc[1:] = c[:-1]
        tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
        tr[0] = h[0] - l[0]
        ctr = np.concatenate(([0.0], np.cumsum(tr)))                 # ctr[b]-ctr[a] = sum tr[a:b]
        hl = np.zeros(n); hl[1:] = (l[1:] > l[:-1]).astype(float)
        chl = np.concatenate(([0.0], np.cumsum(hl)))
        cl = np.concatenate(([0.0], np.cumsum(l)))
        idx = np.arange(n, dtype=float)
        cil = np.concatenate(([0.0], np.cumsum(idx * l)))
        W = 20.0
        xbar = (W - 1) / 2.0
        vx = ((np.arange(W) - xbar) ** 2).sum()                      # fixed window variance*W
        for i in range(K_B + 1, n):
            if int(m[i]) > LAST_M:
                break
            level = float(hod[i - 1])
            if h[i] < level or level < o0 * 1.05 or level < MIN_LEVEL or i + 1 >= n:
                continue
            r = dict(day=day, symbol=s, break_m=int(m[i]), hl_n20=np.nan, lo_slope20=np.nan,
                     atr_now=np.nan, atr_prev=np.nan, atr_ratio=np.nan)
            a = i - 20
            if a >= 1:                                               # hl needs l[a-1]
                r['hl_n20'] = float(chl[i] - chl[a])
            if a >= 0:
                sl = cl[i] - cl[a]
                sil = cil[i] - cil[a]
                # slope of l over j in [a, i) vs x = j - a
                sxl = sil - a * sl
                cov = sxl - xbar * sl
                r['lo_slope20'] = float(cov / vx / level * 100.0)
            if i >= 15:
                an = (ctr[i] - ctr[i - 14]) / 14.0
                r['atr_now'] = float(an)
                if i >= 29:
                    ap = (ctr[i - 14] - ctr[i - 28]) / 14.0
                    r['atr_prev'] = float(ap)
                    r['atr_ratio'] = float(an / ap) if ap > 0 else np.nan
            rows.append(r)
    return rows


def main():
    bl = pd.read_csv(f'{ROOT}/research/mature_method/hod_frames2/breaks2.csv',
                     usecols=['day', 'symbol'], dtype={'day': str, 'symbol': str},
                     keep_default_na=False, na_values=[''])
    bl = bl.drop_duplicates(['day', 'symbol'])
    bl = bl[(bl.day >= DAY_LO) & (bl.day <= DAY_HI)]
    print(f'symbol-days to walk {len(bl)} over {bl.day.nunique()} days '
          f'[{bl.day.min()} .. {bl.day.max()}]  (TEST sealed)', flush=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    done = set(state['done'])
    days = [d for d in sorted(bl.day.unique()) if d not in done]
    if LIMIT_DAYS:
        days = days[:LIMIT_DAYS]
    for nd, day in enumerate(days):
        syms = bl[bl.day == day].symbol.tolist()
        rows = day_rows(day, syms, load_bars(day, syms))
        if rows:
            pd.DataFrame(rows)[COLS].to_csv(OUT, mode='a', header=not os.path.exists(OUT),
                                            index=False)
        state['done'].append(day); json.dump(state, open(STATE, 'w'))
        if nd % 10 == 0:
            print(f'{nd + 1}/{len(days)} {day} feat+{len(rows)}', flush=True)
    print('WALK DONE', flush=True)


if __name__ == '__main__':
    main()
