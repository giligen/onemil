#!/usr/bin/env python3
"""F37 stage 1 — the 5-MINUTE and 15-MINUTE HOD-break walk, plus its own two placebo arms.

The shipped spec re-expressed on clock-aligned coarse bars (PREREG §4): running high of day over
COMPLETE coarse bars; a consolidation of `k` closed coarse bars all holding within 4 % of that
high; the signal is the coarse bar whose HIGH reaches the high; entry at the NEXT coarse bar's open
under the 0.6 % cap; stop = the consolidation low; `rv_profile` in [1, 5) at the signal bar; entry
minute in 09:37-14:01.  Exits are walked on the 1-MINUTE tape (only the DECISION is coarse).

Row kinds: `S` the signal (tagged `bar`/`k`), `CB` matched non-signal at the signal's own minute,
`CA` the same name-day at a strictly later minute — both on the signal's own R geometry.

Resumable per session.  Every store READ-ONLY.  TEST never loaded.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c12 import (D12, ROOT, CAP, FIRST_ENTRY_M, LAST_ENTRY_M, OPEN_M, bars_arrays,  # noqa: E402
                 coarse, load_bars, load_panel, price_exit, profile_fraction)

OUT = f'{D12}/w37.csv'
ST = f'{D12}/w37_state.json'
COMBOS = ((5, 3), (5, 5), (15, 3), (15, 5))
X = 0.04
KINDS = ('X2R', 'XBR', 'XLK')
WALK = {'X2R': 'bracket', 'XBR': 'bare', 'XLK': 'lock'}
NPOOL, NLATER, SEED = 10, 10, 37
HDR = (['day', 'kind', 'symbol', 'bar', 'k', 'key_sym', 'key_m', 'entry_m', 'level', 'next_open',
        'stop', 'r_pct', 'rv_profile']
       + [f'{kk}_{f}' for kk in KINDS for f in ('rr', 'why', 'em')])


def detect_coarse(cb, o1, m1, cumv1, adv, size, k):
    """First qualifying coarse signal.  Returns (level, stop, entry_minute, rv) or None."""
    co, ch, cl_, cc, cv, cm, ce = cb
    n = len(co)
    o0 = float(o1[0])
    if not (o0 > 0):
        return None
    for i in range(k, n - 1):
        level = float(np.max(ch[:i]))
        if ch[i] < level or level < o0 * 1.05:
            continue
        lo = float(np.min(cl_[i - k:i]))
        if not (lo >= level * (1.0 - X) and lo < level):
            continue
        em = int(cm[i + 1])                       # the entry bar's FIRST minute
        if em < FIRST_ENTRY_M:
            continue
        if em > LAST_ENTRY_M:
            return None
        j = np.flatnonzero(m1 == int(ce[i]))      # rv as of the signal bar's CLOSE
        if not len(j):
            continue
        rv = float(cumv1[int(j[0])]) / (adv * profile_fraction(int(ce[i]))) if adv > 0 else np.nan
        if not (rv == rv and 1.0 <= rv < 5.0):
            continue
        nxt = float(co[i + 1])
        if nxt > level * (1.0 + CAP) or nxt <= lo:
            continue
        return level, lo, em, rv
    return None


def main() -> int:
    pop = pd.read_csv(f'{D12}/orbpop38.csv', dtype={'day': str, 'symbol': str})
    u = load_panel()
    elig = u[(u.prev_close >= 17.0) & (u.adv20 >= 100000)]
    sigset = set(zip(pop.day, pop.symbol))
    elig = elig[~pd.Series(list(zip(elig.day, elig.symbol)), index=elig.index).isin(sigset)]
    elig = elig.assign(lp=np.log(elig.prev_close), la=np.log(elig.adv20))
    ED = {d: g for d, g in elig.groupby('day')}
    ADV = {(d, s): a for d, s, a in u[['day', 'symbol', 'adv20']].itertuples(index=False)}
    days = sorted(pop.day.unique())
    done = set(json.load(open(ST))['days']) if os.path.exists(ST) else set()
    if not os.path.exists(OUT):
        with open(OUT, 'w') as f:
            f.write(','.join(HDR) + '\n')
    todo = [d for d in days if d not in done]
    print(f'  {len(todo)} sessions to walk ({len(done)} done)', flush=True)
    PD = {d: g for d, g in pop.groupby('day')}
    rng = np.random.default_rng(SEED)
    nrow = 0
    for i, day in enumerate(todo):
        sub = PD[day]
        bars = load_bars(day, sorted(sub.symbol.unique()))
        A = {}
        for s_, b_ in bars.items():
            x = bars_arrays(b_)
            if x is not None:
                A[s_] = x                      # o,h,l,c,v,m
        rows, sigs = [], []
        for sym, a in A.items():
            o, h, l, c, v, m = a
            adv = float(ADV.get((day, sym), np.nan))
            if not (adv == adv and adv > 0):
                continue
            cumv = np.cumsum(v)
            for size, k in COMBOS:
                cb = coarse(o, h, l, c, v, m, size)
                if cb is None:
                    continue
                r = detect_coarse(cb, o, m, cumv, adv, size, k)
                if r is None:
                    continue
                level, stop, em, rv = r
                j = np.flatnonzero(m == em)
                if not len(j):
                    continue
                e = int(j[0])
                nxt = float(o[e])
                rp = (nxt - stop) / nxt * 100.0
                out = [day, 'S', sym, size, k, sym, em, em, level, nxt, stop, rp, rv]
                for kk in KINDS:
                    exm, rr, why = price_exit(WALK[kk], o, h, l, c, m, e, stop)
                    out += [rr, why, exm]
                rows.append(out)
                sigs.append((sym, em, rp, nxt))
        # ---- the two placebo arms, on the pooled signal set ------------------------------
        seen, keys = set(), []
        for sym, em, rp, nxt in sigs:
            if (sym, em) in seen:
                continue
            seen.add((sym, em))
            g = ED.get(day)
            if g is not None and len(g):
                adv = float(ADV.get((day, sym), np.nan))
                if adv == adv and adv > 0:
                    d_ = np.abs(g.lp.values - np.log(nxt)) + np.abs(g.la.values - np.log(adv))
                    for z in np.argsort(d_, kind='mergesort')[:NPOOL]:
                        keys.append(('CB', g.symbol.values[z], em, sym, em, rp))
            hi = LAST_ENTRY_M
            if em + 1 <= hi:
                cand = np.arange(em + 1, hi + 1)
                pick = cand if len(cand) <= NLATER else rng.choice(cand, NLATER, replace=False)
                for mm in sorted(int(z) for z in pick):
                    keys.append(('CA', sym, mm, sym, em, rp))
        need = sorted({kk[1] for kk in keys} - set(A))
        if need:
            for s_, b_ in load_bars(day, need).items():
                x = bars_arrays(b_)
                if x is not None:
                    A[s_] = x
        for arm, csym, cm_, ksym, km, rp in keys:
            a = A.get(csym)
            if a is None:
                continue
            o, h, l, c, v, m = a
            j = np.flatnonzero(m == int(cm_))
            if not len(j) or int(j[0]) + 1 >= len(o):
                continue
            e = int(j[0]); E = float(o[e])
            if not (E > 0) or not (rp == rp):
                continue
            stop = E * (1.0 - rp / 100.0)
            out = [day, arm, csym, -1, -1, ksym, km, int(cm_), np.nan, E, stop, rp, np.nan]
            for kk in KINDS:
                if kk == 'XLK':
                    out += [np.nan, '', -1]; continue
                exm, rr, why = price_exit(WALK[kk], o, h, l, c, m, e, stop)
                out += [rr, why, exm]
            rows.append(out)
        if rows:
            pd.DataFrame(rows, columns=HDR).to_csv(OUT, mode='a', header=False, index=False)
            nrow += len(rows)
        done.add(day)
        json.dump({'days': sorted(done)}, open(ST, 'w'))
        if i % 25 == 0 or i == len(todo) - 1:
            print(f'  {i+1}/{len(todo)} {day} rows {nrow:,}', flush=True)
    print(f'  DONE — {nrow:,} rows -> w37.csv', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
