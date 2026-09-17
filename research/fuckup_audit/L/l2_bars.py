#!/usr/bin/env python3
"""Stage L step 2 — the tape walk that makes T2 / T6 / T7 computable.

For every row of every run_book population (B1-B4, B6) it reads the (symbol, day) 1-minute tape ONCE
(bars_sip.db first, data/cache.db fallback, both read-only) and writes:

  v_fill      volume of the FILL minute                                   -> T7
  v_prev5     mean volume of the 5 bars before the fill minute            -> T7
  o_next      open of the first bar AFTER the fill minute  (+ its l/h)    -> T7's exit fill
  m_next      that bar's minute
  o_p10       open of the first bar at or after fill + 10 min (+ l/h)     -> T6's check and exit fill
  m_p10       that bar's minute
  rsf_tape    range-so-far % at the SIGNAL bar, bars strictly before it   -> T2 (B1 only; the others carry it)

Writes L/bars_<B>.csv and L/l2_coverage.md.  Missing = the filter fails open on that trade (PREREG §1).
"""
import os
import sqlite3
import sys
import time

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcore as C                                                             # noqa: E402

L = C.L


class Tape:
    """(symbol, day) -> {minute: (o, h, l, c, v)} over 09:30-15:59 ET, plain python (no pandas per key)."""

    def __init__(self):
        self.sip = sqlite3.connect(f'file:{C.BARS_SIP}?mode=ro', uri=True)
        self.cache = sqlite3.connect(f'file:{C.CACHE_DB}?mode=ro', uri=True)
        self.n = dict(sip=0, cache=0, none=0)
        self._offs = {}

    def _off(self, day):
        """ET UTC-offset in minutes for a trading day (-240 EDT / -300 EST), computed once per day."""
        o = self._offs.get(day)
        if o is None:
            o = int(datetime(int(day[:4]), int(day[5:7]), int(day[8:10]), 12,
                             tzinfo=ZoneInfo('America/New_York')).utcoffset().total_seconds() // 60)
            self._offs[day] = o
        return o

    def get(self, sym, day):
        r = self.sip.execute('select t, o, h, l, c, v from bars where symbol=? and day=? order by t',
                             (sym, day)).fetchall()
        src = 'sip'
        if not r:
            r = self.cache.execute('select timestamp, open, high, low, close, volume from '
                                   'intraday_bars_1min where symbol=? and bar_date=? order by timestamp',
                                   (sym, day)).fetchall()
            src = 'cache'
        if not r:
            self.n['none'] += 1
            return None
        self.n[src] += 1
        # both stores hold UTC ISO timestamps ('YYYY-MM-DDTHH:MM:SS+00:00'); plain arithmetic, one
        # zoneinfo lookup per DAY (the pandas conversion per key was 0.13 s/key = 2.4 h for this stage)
        off = self._off(day)
        out = {}
        for x in r:
            t = x[0]
            m = int(t[11:13]) * 60 + int(t[14:16]) + off
            if 570 <= m < 960:
                out[m] = (x[1], x[2], x[3], x[4], x[5])
        return out or None

    def close(self):
        self.sip.close()
        self.cache.close()


def walk(bid, tape, need_rsf):
    p = pd.read_csv(f'{L}/pop_{bid}.csv', keep_default_na=False, na_values=[''],
                    dtype={'symbol': str, 'day': str})
    p = p.sort_values(['symbol', 'day']).reset_index()
    out = []
    key = None
    bars = None
    t0 = time.time()
    for i, r in enumerate(p.itertuples()):
        if i % 2000 == 0:
            C.log(f'  {bid} {i}/{len(p)}  {(time.time()-t0)/60:.1f} min')
        k = (r.symbol, r.day)
        if k != key:
            key, bars = k, tape.get(*k)
        o = dict(idx=r.index, has_tape=int(bars is not None))
        if bars:
            em = int(r.entry_m)
            v_fill = bars.get(em, (np.nan,) * 5)[4]
            # the 5 bars BEFORE the fill minute, exactly as `live_followthrough.py` (the evidence T7
            # transfers from) reads them: the last five bars the tape PRINTS, not the five clock
            # minutes (thin names skip minutes, and requiring five consecutive clock minutes fails
            # open on 32-40% of this population)
            pre = [bars[m][4] for m in sorted(m for m in bars if m < em)][-5:]
            o['v_fill'] = v_fill
            o['v_prev5'] = float(np.mean(pre)) if len(pre) == 5 else np.nan
            nx = [m for m in bars if m > em]
            if nx:
                m1 = min(nx)
                b = bars[m1]
                o['m_next'], o['o_next'], o['h_next'], o['l_next'] = m1, b[0], b[1], b[2]
            t10 = [m for m in bars if m >= em + 10]
            if t10:
                m10 = min(t10)
                b = bars[m10]
                o['m_p10'], o['o_p10'], o['h_p10'], o['l_p10'] = m10, b[0], b[1], b[2]
            if need_rsf and r.sig_m == r.sig_m:
                sm = int(r.sig_m)
                pre_m = [m for m in bars if m < sm]
                if pre_m and 570 in bars:
                    hi = max(bars[m][1] for m in pre_m)
                    lo = min(bars[m][2] for m in pre_m)
                    o['rsf_tape'] = (hi - lo) / bars[570][0] * 100.0
        out.append(o)
    d = pd.DataFrame(out).set_index('idx').sort_index()
    d.to_csv(f'{L}/bars_{bid}.csv')
    return p, d


def main():
    tape = Tape()
    lines = ['# Stage L step 2 — tape coverage for the T2 / T6 / T7 inputs', '',
             '| book | rows | tape | v_fill | v_prev5 | next bar | fill+10 bar | rsf |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for bid in ('B1', 'B2', 'B3', 'B4', 'B6'):
        p, d = walk(bid, tape, need_rsf=(bid == 'B1'))
        n = len(d)
        def cov(c):
            return f'{(d[c].notna().mean() * 100 if c in d.columns else 0.0):.1f}%'
        lines.append(f"| {bid} | {n:,} | {d.has_tape.mean()*100:.1f}% | {cov('v_fill')} | "
                     f"{cov('v_prev5')} | {cov('o_next')} | {cov('o_p10')} | {cov('rsf_tape')} |")
        C.log(lines[-1])
    lines += ['', f"bar sources: sip {tape.n['sip']:,} keys, cache.db {tape.n['cache']:,}, "
                  f"no tape {tape.n['none']:,}"]
    tape.close()
    open(f'{L}/l2_coverage.md', 'w').write('\n'.join(lines))
    C.log('\n'.join(lines))


if __name__ == '__main__':
    main()
