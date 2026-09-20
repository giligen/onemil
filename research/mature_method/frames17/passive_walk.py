#!/usr/bin/env python3
"""frames17 F52 — the mirror short entered on a RESTING limit (never a reacting order).

PREREG frames17/PREREG.md. Population: frames16 arm3's MIR2 signal (`sw_*.csv`, f_mir2 & gate5 &
price>=$5 & ex-wrapper & day<2026-06-01), restricted to Alpaca-ETB names (`borrow_flags.csv`) — the
same short-feasibility rail frames16 arm3 applied to S1. SSR's uptick-fill mechanic is NOT applied
here: it blocks a REACTING marketable sell that needs to already be at/above the bid at the instant
of a forced fill; a resting limit priced k>0 ABOVE the last close is, by construction, already
displayed above the market and is not the thing Reg SHO 201 restricts. Declared in PREREG before
scoring.

Entry: a resting SELL limit at `ref * (1+k)`, k in {0.3%, 0.6%}, `ref` = the same pre-cut close
frames16 used, placed at the signal-hour cut bar (bar e, frames16's own reacting-entry bar), live
for 5 minutes (bars e..e+4). Fills AT THE LIMIT the first bar whose HIGH >= limit; otherwise
UNFILLED (0 P&L). frames16's own `rr_a_bare` on the SAME (day,symbol,hour) row is carried through as
the unfilled counterfactual — it is exactly what the reacting fill already scored there.

Exit unchanged from arm3 spec A: stop = entry*1.02, bare exit (EOD -> stop -> nothing else),
walked from the fill bar + 1 with `frames16/short_walk.py::swalk` verbatim (imported, not copied).
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
from common5 import attach_instrument                                  # noqa: E402
from short_walk import swalk                                           # noqa: E402

D = f'{ROOT}/research/mature_method/frames17'
D16 = f'{ROOT}/research/mature_method/frames16'
SIP = f'{ROOT}/research/bf_zero/bars_sip.db'
BORROW = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv'
TEST_FROM = '2026-06-01'
KS = (0.003, 0.006)
WINDOW = 5


def population():
    fs = sorted(glob.glob(f'{D16}/sw_*.csv'))
    d = pd.concat([pd.read_csv(f, dtype={'symbol': str, 'day': str},
                               keep_default_na=False, na_values=['']) for f in fs],
                  ignore_index=True)
    d = d[(d.day < TEST_FROM) & d.gate5.astype(bool) & d.f_mir2.astype(bool)
          & (d.price >= 5)].copy()
    d = attach_instrument(d)
    d = d[d.asset_class != 'wrapper']
    b = pd.read_csv(BORROW, dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    b['etb'] = b.shortable.astype(str).str.lower().eq('true') & \
        b.easy_to_borrow.astype(str).str.lower().eq('true')
    m = dict(zip(b.symbol, b.etb))
    d['etb'] = d.symbol.map(m).fillna(False).astype(bool)
    d['split'] = np.where(d.day < '2026-01-01', 'TRAIN', 'VAL')
    print(f'[pop] {len(d):,} MIR2 gate5 price>=$5 ex-wrapper rows, ETB '
          f'{float(d.etb.mean())*100:.1f}%', flush=True)
    return d[d.etb].copy()


def run():
    d = population()
    con = sqlite3.connect(f'file:{SIP}?mode=ro', uri=True, timeout=180)
    rows = []
    ndays = d.day.nunique()
    for i, (day, g) in enumerate(d.groupby('day', sort=True)):
        syms = sorted(g.symbol.unique())
        qs = ','.join('?' * len(syms))
        b = pd.read_sql(f'select symbol,t,o,h,l,c,v from bars where day=? and symbol in ({qs})',
                        con, params=[day] + syms)
        if b.empty:
            continue
        off = 4 if ('2025-03-09' <= day < '2025-11-02') or ('2026-03-08' <= day < '2026-11-01') \
            else 5
        b['m'] = ((b.t.str.slice(11, 13).astype(int) - off) * 60
                  + b.t.str.slice(14, 16).astype(int))
        b = b[(b.m >= 570) & (b.m < 960)].sort_values(['symbol', 'm'], kind='mergesort')
        byx = {s: x for s, x in b.groupby('symbol', sort=False)}
        for r in g.itertuples():
            x = byx.get(r.symbol)
            if x is None:
                continue
            mv = x.m.values
            o, hi, lo, c = (x[k].values.astype(float) for k in ('o', 'h', 'l', 'c'))
            idx = np.flatnonzero(mv == int(r.entry_m))
            if not len(idx):
                continue
            e = int(idx[0])
            for k in KS:
                limit = r.ref * (1.0 + k)
                fj = None
                for off2 in range(WINDOW):
                    j = e + off2
                    if j >= len(mv):
                        break
                    if hi[j] >= limit:
                        fj = j
                        break
                base = dict(day=day, symbol=r.symbol, hour=int(r.hour), k=k, ref=r.ref,
                           split=r.split, rr_reacting=r.rr_a_bare,
                           react_entry_m=int(r.entry_m), react_entry=float(r.entry),
                           react_exit_m=int(r.exitm_a_bare), react_rpct=float(r.rpct_a))
                if fj is None:
                    base.update(filled=False, fill_m=np.nan, entry=np.nan, rpct=np.nan,
                               rr=np.nan, why='unfilled', exit_m=np.nan)
                else:
                    entry = limit
                    stop = entry * 1.02
                    rr, why, xm = swalk(o, hi, lo, c, mv, fj, entry, stop, 'bare')
                    base.update(filled=True, fill_m=int(mv[fj]), entry=entry,
                               rpct=(stop - entry) / entry, rr=rr, why=why, exit_m=int(xm))
                rows.append(base)
        if i % 40 == 0:
            print(f'  {i+1}/{ndays} days -> {len(rows):,} rows so far', flush=True)
    con.close()
    out = pd.DataFrame(rows)
    out.to_csv(f'{D}/passive17.csv', index=False)
    print(f'[walk] {len(out):,} (signal x k) rows -> passive17.csv; filled '
          f'{float(out.filled.mean())*100:.1f}%', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(run())
