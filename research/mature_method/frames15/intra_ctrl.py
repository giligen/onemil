#!/usr/bin/env python3
"""frames15 ARM B (intraday) — the PLACEBO CONTROLS for B9..B14 (diagnostics D1/D2/D3).

For every session in the signal set, walks the SAME bracket (next-bar open under the +0.6 % cap,
2 % stop, exit at the stop or 15:55) at EVERY hour boundary, for:
  * the signal symbols themselves  -> D3, the same name-day at a LATER hour (causal: after the
    signal hour), and
  * a random 25-symbol sample of the session's other tape names -> D1 the universe bound and, with
    the price/ADV match applied at scoring time, D2 the matched non-signal name.
Every row carries `gate5` (session high up to that hour's close >= open x 1.05), the same causal
membership rail the cells use. Checkpointed per month to `ctrl_YYYY-MM.csv`.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/frames15')
from common15 import D, ROOT                                        # noqa: E402
from armB_intra import CAP, EOD_M, OPEN_M, SLIP, STOP, walk         # noqa: E402

NSAMP = 25
SEED = 15


def main():
    sig = pd.read_parquet(f'{D}/hourly15.parquet', columns=['symbol', 'day', 'hour', 'hrv',
                                                            'hour_ret', 'sus2_3'])
    sig['symbol'] = sig.symbol.astype(str)
    sig['day'] = sig.day.astype(str)
    fire = sig[((sig.hrv >= 3.0) | sig.sus2_3.fillna(False).astype(bool)) &
               sig.hour.between(9, 14)]
    sigset = set(zip(fire.symbol, fire.day, fire.hour.astype(int)))
    sig_days = sorted(fire.day.unique())
    by_day = fire.groupby('day').symbol.apply(lambda s: sorted(set(s)))
    con = sqlite3.connect(f'file:{ROOT}/research/bf_zero/bars_sip.db?mode=ro', uri=True,
                          timeout=120)
    rng = np.random.default_rng(SEED)
    months = sorted({d[:7] for d in sig_days if d < '2026-06-01'})   # FREEZE: TEST never walked
    for mo in months:
        out = f'{D}/ctrl_{mo}.csv'
        if os.path.exists(out):
            continue
        rows = []
        for day in [d for d in sig_days if d[:7] == mo]:
            pool = pd.read_sql('select symbol from fetch_log where day=? and n_bars>0', con,
                               params=(day,)).symbol.astype(str).tolist()
            sy = list(by_day[day])
            others = [s for s in pool if s not in set(sy)]
            if others:
                sy = sy + list(rng.choice(others, size=min(NSAMP, len(others)), replace=False))
            qs = ','.join('?' * len(sy))
            b = pd.read_sql(f'select symbol,t,o,h,l,c from bars where day=? and symbol in ({qs})',
                            con, params=[day] + sy)
            if b.empty:
                continue
            off = 4 if ('2025-03-09' <= day < '2025-11-02') or \
                       ('2026-03-08' <= day < '2026-11-01') else 5
            b['m'] = (b.t.str.slice(11, 13).astype(int) - off) * 60 + \
                     b.t.str.slice(14, 16).astype(int)
            b = b[(b.m >= OPEN_M) & (b.m < 960)].sort_values(['symbol', 'm'], kind='mergesort')
            for sym, x in b.groupby('symbol', sort=False):
                if len(x) < 20:
                    continue
                mv = x.m.values
                o, hi, lo, c = (x[k].values.astype(float) for k in ('o', 'h', 'l', 'c'))
                sess_open = o[0]
                for hour in range(9, 15):
                    cut = (hour + 1) * 60
                    pre = mv < cut
                    if not pre.any():
                        continue
                    nxt = np.flatnonzero(mv >= cut)
                    if not len(nxt):
                        continue
                    e = int(nxt[0])
                    if mv[e] > cut + 5:
                        continue
                    ref = float(c[pre][-1])
                    entry = float(o[e])
                    if entry > ref * (1.0 + CAP) or entry <= 0:
                        continue
                    rr, why, xm = walk(o, hi, lo, c, mv, e, entry, entry * (1.0 - STOP), 'bare')
                    rows.append(dict(day=day, symbol=sym, hour=hour, entry_m=int(mv[e]),
                                     entry=entry,
                                     gate5=bool(hi[pre].max() >= sess_open * 1.05),
                                     is_sig=(sym, day, hour) in sigset,
                                     sig_day=sym in set(by_day[day]),
                                     rr_bare=rr, why_bare=why, exitm_bare=xm))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'  ctrl {mo}: {len(rows):,} walks', flush=True)
    con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
