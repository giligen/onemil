#!/usr/bin/env python3
"""F39 stage 1 — THE MULTI-DAY HIGH WALK.

The object (PREREG §2): on the PIT daily panel (prev close >= $17, ADV20 >= 100K, test tickers and
non-`daily_bars` names out), the FIRST 1-minute bar whose CLOSE exceeds the prior-N-session high
(N in {5, 20, 252}, levels built from daily bars STRICTLY BEFORE the signal day), entry at the next
bar's open under the 0.6 % cap, window 09:37-14:01.

Prices five (stop, exit) combinations per signal:
    S20 x {bracket +2R, bare stop, static lock, hold-to-next-open}   and   SPD x {bracket}.

Resumable per session.  Every store READ-ONLY.  TEST never loaded (`day < 2026-06-01`).

  python3 w39.py            # appends to w39.csv, state in w39_state.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c12 import (D12, OPEN_M, CAP, FIRST_ENTRY_M, LAST_ENTRY_M, bars_arrays, load_bars,  # noqa: E402
                 load_panel, price_exit, profile_fraction)

OUT = f'{D12}/w39.csv'
ST = f'{D12}/w39_state.json'
LEVELS = (('H5', 'h5'), ('H20', 'h20'), ('H252', 'h252'))
COMBOS = (('S20', 'bracket', 'X2R'), ('S20', 'bare', 'XBR'), ('S20', 'lock', 'XLK'),
          ('S20', 'next', 'XNO'), ('SPD', 'bracket', 'X2R'))
HDR = (['day', 'symbol', 'lvl', 'level', 'break_m', 'entry_m', 'next_open', 'fill_capped',
        'rv_profile', 'adv20', 'open_px', 'stop_S20', 'stop_SPD', 'last_close', 'daily_close',
        'next_open_d']
       + [f'{s}_{x}_{k}' for (s, _w, x) in COMBOS for k in ('rr', 'why', 'exit_m')])


def main() -> int:
    u = load_panel()
    c = u[(u.prev_close >= 17.0) & (u.adv20 >= 100000) & u.h5.notna() & (u.high > u.h5)]
    print(f'  F39 candidate symbol-days {len(c):,} over {c.day.nunique()} sessions '
          f'({c.symbol.nunique():,} symbols) — TEST cut off', flush=True)
    done = set(json.load(open(ST))['days']) if os.path.exists(ST) else set()
    if not os.path.exists(OUT):
        with open(OUT, 'w') as f:
            f.write(','.join(HDR) + '\n')
    days = sorted(set(c.day) - done)
    print(f'  {len(days)} sessions to walk ({len(done)} done)', flush=True)
    nrow = 0
    for i, day in enumerate(days):
        sub = c[c.day == day]
        bars = load_bars(day, sorted(sub.symbol.unique()))
        rows = []
        for r in sub.itertuples():
            gg = bars.get(r.symbol)
            if gg is None:
                continue
            arr = bars_arrays(gg)
            if arr is None:
                continue
            o, h, l, cl, v, m = arr
            o0 = float(o[0])
            if not (o0 > 0):
                continue
            cumv = np.cumsum(v)
            adv = float(r.adv20)
            last_close = float(cl[-1])
            for tag, col in LEVELS:
                lvl = getattr(r, col)
                if not (lvl == lvl and lvl > 0):
                    continue
                # the FIRST bar whose CLOSE exceeds the level, with a next bar inside the window
                idx = np.flatnonzero(cl > lvl)
                e = -1
                for j in idx:
                    if j + 1 >= len(o):
                        break
                    if int(m[j + 1]) < FIRST_ENTRY_M:
                        continue
                    if int(m[j + 1]) > LAST_ENTRY_M:
                        break
                    e = int(j)
                    break
                if e < 0:
                    continue
                nxt = float(o[e + 1])
                rv = float(cumv[e]) / (adv * profile_fraction(int(m[e]))) if adv > 0 else np.nan
                s20 = float(np.min(l[max(0, e - 19): e + 1]))
                spd = float(r.prev_low) if r.prev_low == r.prev_low else np.nan
                row = [day, r.symbol, tag, float(lvl), int(m[e]), int(m[e + 1]), nxt,
                       int(nxt <= lvl * (1.0 + CAP)), rv, adv, o0, s20, spd, last_close,
                       float(r.close), float(r.next_open_d) if r.next_open_d == r.next_open_d
                       else np.nan]
                for (sk, wk, _x) in COMBOS:
                    st = s20 if sk == 'S20' else spd
                    if not (st == st) or st >= nxt:
                        row += [np.nan, '', -1]
                        continue
                    em, rr, why = price_exit(wk, o, h, l, cl, m, e + 1, st,
                                             close_px=float(r.close),
                                             next_open_px=row[15])
                    row += [rr, why, em]
                rows.append(row)
        if rows:
            pd.DataFrame(rows, columns=HDR).to_csv(OUT, mode='a', header=False, index=False)
            nrow += len(rows)
        done.add(day)
        json.dump({'days': sorted(done)}, open(ST, 'w'))
        if i % 25 == 0 or i == len(days) - 1:
            print(f'  {i+1}/{len(days)} {day} rows {nrow:,}', flush=True)
    print(f'  DONE — {nrow:,} rows -> w39.csv', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
