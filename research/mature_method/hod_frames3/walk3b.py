#!/usr/bin/env python3
"""hod_frames3 / F11 — the second attach pass: `consol_bars` at EVERY candidate break bar.

`hod_fresh/sig3.csv` carries `consol_bars` only for the first-qualifying break of each declared
rung; the F11 difference test needs it on every candidate so that the clock rung (`break_m >= 590`)
can be split by it.  Definition is `hod_losers/walk.py`'s, verbatim, as `pass3.py` used it:
consecutive bars back from `i-1` whose LOW >= `level x 0.96`.  Strictly prior bars only.

TRAIN + VAL days only -- TEST is sealed.  Read-only on every DB.  Resumable per day.
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
from pass2 import load_bars                                          # noqa: E402

D = f'{ROOT}/research/mature_method/hod_frames3'
STATE, OUT = f'{D}/walk3b_state.json', f'{D}/feat3b.csv'
OPEN_M, LAST_M, MIN_LEVEL, K_B = 570, 930, 1.0, 5
DAY_LO, DAY_HI = '2025-01-02', '2026-05-31'
LIMIT_DAYS = int(os.environ.get('MM_LIMIT_DAYS', '0'))
COLS = ['day', 'symbol', 'break_m', 'consol_bars', 'touch_n']


def main():
    bl = pd.read_csv(f'{ROOT}/research/mature_method/hod_frames2/breaks2.csv',
                     usecols=['day', 'symbol'], dtype={'day': str, 'symbol': str},
                     keep_default_na=False, na_values=['']).drop_duplicates(['day', 'symbol'])
    bl = bl[(bl.day >= DAY_LO) & (bl.day <= DAY_HI)]
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    done = set(state['done'])
    days = [d for d in sorted(bl.day.unique()) if d not in done]
    if LIMIT_DAYS:
        days = days[:LIMIT_DAYS]
    print(f'{len(days)} days to walk', flush=True)
    for nd, day in enumerate(days):
        syms = bl[bl.day == day].symbol.tolist()
        bars = load_bars(day, syms)
        rows = []
        for s in syms:
            gg = bars.get(s)
            if gg is None:
                continue
            rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
            if len(rth) < 10:
                continue
            h, l = rth.h.values.astype(float), rth.l.values.astype(float)
            m = rth.m.values.astype(int)
            n = len(h)
            o0 = float(rth.o.values[0])
            if not (o0 > 0):
                continue
            hod = np.maximum.accumulate(h)
            for i in range(K_B + 1, n):
                if int(m[i]) > LAST_M:
                    break
                level = float(hod[i - 1])
                if h[i] < level or level < o0 * 1.05 or level < MIN_LEVEL or i + 1 >= n:
                    continue
                jj, cb = i - 1, 0
                thr = level * 0.96
                while jj >= 0 and l[jj] >= thr:
                    cb += 1; jj -= 1
                rows.append(dict(day=day, symbol=s, break_m=int(m[i]), consol_bars=cb,
                                 touch_n=int((h[:i] >= level * 0.995).sum())))
        if rows:
            pd.DataFrame(rows)[COLS].to_csv(OUT, mode='a', header=not os.path.exists(OUT),
                                            index=False)
        state['done'].append(day); json.dump(state, open(STATE, 'w'))
        if nd % 20 == 0:
            print(f'{nd + 1}/{len(days)} {day} +{len(rows)}', flush=True)
    print('WALK3B DONE', flush=True)


if __name__ == '__main__':
    main()
