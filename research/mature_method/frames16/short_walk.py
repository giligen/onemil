#!/usr/bin/env python3
"""frames16 ARM 3 (F49) — walk the discovered-mirror SHORT on the 1-minute SIP tape.

PREREG §5. The signal is `frames15` B12 re-derived: at a session-hour close, `hrv >= 3` and the
hour's return past the mirror cut, on a name whose session high has ALREADY reached open x 1.05 by
that hour's close (`gate5`, causal membership).

Entry is the SHORT-side mirror of the engine's no-chase cap: a resting sell limit at
`ref x (1 - cap)` that fills AT THE NEXT BAR'S OPEN iff that open >= the limit. A bar that opens
below the limit is a SKIP with 0 P&L, never a loss — and the skipped rows are walked anyway and
kept, because the unfilled counterfactual is the adverse-selection check this house owes every
passive entry (halt-resume, 2026-09-18).

Two stop specs (2 % of price, and the signal hour's high with a 1 % floor) x two exits (bare and a
+2R bracket filled on a bar CLOSE, never a touch). SSR and borrow are applied in the SCORER, not
here: this file emits the raw walk plus every flag the scorer needs, so a rail can be switched on
and off without re-walking.

Checkpointed per month. One process, nice, memory-capped; every store read-only.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

D = f'{ROOT}/research/mature_method/frames16'
D15 = f'{ROOT}/research/mature_method/frames15'
SIP = f'{ROOT}/research/bf_zero/bars_sip.db'
OPEN_M, EOD_M = 570, 955
CAP = 0.006
SLIP = 0.001
TEST_FROM = '2026-06-01'
RSEED = 16


def swalk(o, hi, lo, c, m, e, entry, stop, mode):
    """Price ONE short from bar e (entry at o[e]); exits from bar e+1.

    Priority EOD -> stop -> target, the exact mirror of `frames15/armB_intra.walk`.
    Stop covers at max(stop, that bar's open) x (1 + slip) — one slip AGAINST us.
    """
    n = len(o)
    R = stop - entry
    tgt = entry - 2.0 * R
    for k in range(e + 1, n):
        if m[k] >= EOD_M:
            return (entry - o[k]) / R, 'eod', m[k]
        if hi[k] >= stop:
            px = max(stop, o[k]) * (1.0 + SLIP)
            return (entry - px) / R, 'stop', m[k]
        if mode == 'tgt' and c[k] <= tgt:
            return 2.0, 'target', m[k]
    return (entry - c[-1]) / R, 'eod', m[-1]


def signals():
    h = pd.read_parquet(f'{D15}/hourly15.parquet',
                        columns=['symbol', 'day', 'hour', 'hrv', 'hour_ret'])
    h['symbol'] = h.symbol.astype(str)
    h['day'] = h.day.astype(str)
    h = h[(h.day < TEST_FROM) & h.hour.between(9, 14)]
    h = h[~h.symbol.str.match(r'^Z[A-Z]ZZT$', na=False)]          # the ZVZZT standing rule
    hv = h.hrv >= 3.0
    h['f_mir2'] = hv & (h.hour_ret.abs() > 0.02)
    h['f_up2'] = hv & (h.hour_ret > 0.02)
    h['f_abs'] = hv & (h.hour_ret.abs() <= 0.01)
    h = h[h.f_mir2 | h.f_up2 | h.f_abs]
    print(f'[sig] {len(h):,} candidate signal hours over {h.day.nunique()} sessions '
          f'(mir2 {int(h.f_mir2.sum()):,}, up2 {int(h.f_up2.sum()):,}, '
          f'abs {int(h.f_abs.sum()):,})', flush=True)
    return h


def run():
    sig = signals()
    rng = np.random.default_rng(RSEED)
    months = sorted(sig.day.str[:7].unique())
    con = sqlite3.connect(f'file:{SIP}?mode=ro', uri=True, timeout=180)
    for mo in months:
        out = f'{D}/sw_{mo}.csv'
        if os.path.exists(out):
            continue
        rows = []
        for day, g in sig[sig.day.str[:7] == mo].groupby('day', sort=True):
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
            b = b[(b.m >= OPEN_M) & (b.m < 960)].sort_values(['symbol', 'm'], kind='mergesort')
            byx = {s: x for s, x in b.groupby('symbol', sort=False)}
            # -------- the real signals
            todo = [(r.symbol, int(r.hour), bool(r.f_mir2), bool(r.f_up2), bool(r.f_abs), False)
                    for r in g.itertuples()]
            # -------- P2: one RANDOM other hour per gate5 MIR2 name-day (built after the walk)
            mir_syms = sorted(set(r.symbol for r in g.itertuples() if r.f_mir2))
            used = {(r.symbol, int(r.hour)) for r in g.itertuples()}
            for s in mir_syms:
                cand = [hh for hh in range(9, 15) if (s, hh) not in used]
                if cand:
                    todo.append((s, int(rng.choice(cand)), False, False, False, True))
            for sym, hour, f_mir2, f_up2, f_abs, f_rand in todo:
                x = byx.get(sym)
                if x is None or len(x) < 20:
                    continue
                mv = x.m.values
                o, hi, lo, c = (x[k].values.astype(float) for k in ('o', 'h', 'l', 'c'))
                cut = (hour + 1) * 60
                pre = mv < cut
                if not pre.any():
                    continue
                sess_open = float(o[0])
                gate5 = bool(hi[pre].max() >= sess_open * 1.05)
                inhr = (mv >= hour * 60) & (mv < cut)
                if not inhr.any():
                    continue
                hour_high = float(hi[inhr].max())
                nxt = np.flatnonzero(mv >= cut)
                if not len(nxt):
                    continue
                e = int(nxt[0])
                if mv[e] > cut + 5:
                    continue
                ref = float(c[pre][-1])
                entry = float(o[e])
                if entry <= 0 or ref <= 0:
                    continue
                low_to_entry = float(lo[:e + 1].min())
                rec = dict(day=day, symbol=sym, hour=hour, entry_m=int(mv[e]), entry=entry,
                           ref=ref, sess_open=sess_open, hour_high=hour_high, price=entry,
                           gate5=gate5, f_mir2=f_mir2, f_up2=f_up2, f_abs=f_abs, f_rand=f_rand,
                           filled_cap=bool(entry >= ref * (1.0 - CAP)),
                           uptick=bool(entry > ref), low_to_entry=low_to_entry,
                           sess_low=float(lo.min()))
                for tag, stop in (('a', entry * 1.02),
                                  ('b', max(hour_high, entry * 1.01))):
                    rec[f'stop_{tag}'] = stop
                    rec[f'rpct_{tag}'] = (stop - entry) / entry
                    for mode, mt in (('bare', 'bare'), ('tgt', 'tgt')):
                        rr, why, xm = swalk(o, hi, lo, c, mv, e, entry, stop, mode)
                        rec[f'rr_{tag}_{mt}'] = rr
                        rec[f'why_{tag}_{mt}'] = why
                        rec[f'exitm_{tag}_{mt}'] = xm
                rows.append(rec)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'  {mo}: {len(rows):,} walked -> sw_{mo}.csv', flush=True)
    con.close()
    return 0


if __name__ == '__main__':
    sys.exit(run())
