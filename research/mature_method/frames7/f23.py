#!/usr/bin/env python3
"""F23 — THE MIRROR: short the matched non-mover.

Pass 6's arm b — a matched NON-SIGNAL name entered at a mover's clock — reads **-0.161 / -0.160 R**
as a long on 51,051 era-stable trades.  This frame simulates the SHORT of that same population
properly (never a sign flip): sell-limit under a floor, stop ABOVE at the same % distance, cover at
-2R, flat 15:55, borrow screened, Reg SHO 201 excluded, short cost booked at 1.8x the long cost.

  python3 f23.py walk    # resumable bar pass -> p23.csv
  python3 f23.py score   # the 10 declared cells
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c7 import (D7, ROOT, SPLITS, arrays, asset_class, clustered_t, daily_fallback,      # noqa
                half_of, idx_of_minute, load_bars, mde, split_of, universe, walk_short)

D6 = f'{ROOT}/research/mature_method/hod_frames6'
STATE = f'{D7}/f23_state.json'
OUT = f'{D7}/p23.csv'
BORROW = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv'
CLOCKS = (('sig', 0), ('1000', 600), ('1100', 660), ('1200', 720))
LONG_COST_R = 0.063          # the programme's measured booked cost (hod_frames5 §2.5)
SHORT_MULT = 1.8             # F2 §3 — the short side pays 1.8x per R
COST = LONG_COST_R * SHORT_MULT
RISK = 100.0                 # $ per R, the HOD dry-run size
NDRAW = 2000
SEED = 20260920
HDR = ['day', 'symbol', 'entry_m', 'ctrl', 'clock', 'e_m', 'rr', 'why', 'rng_pct', 'down_pct']


def keys():
    """pb6's own (day, symbol, entry_m, ctrl) pairs with the booked trade's r_pct and split."""
    bk = pd.read_csv(f'{D6}/book6.csv', dtype={'day': str, 'symbol': str},
                     usecols=['day', 'symbol', 'entry_m', 'split', 'r_pct'])
    p = pd.read_csv(f'{D6}/pb6.csv', dtype={'day': str, 'symbol': str, 'ctrl': str},
                    usecols=['day', 'symbol', 'entry_m', 'ctrl'])
    p = p.merge(bk, on=['day', 'symbol', 'entry_m'], how='left')
    return p[p.r_pct.notna()]


def walk():
    p = keys()
    print(f'{len(p):,} arm-b pairs over {p.day.nunique()} sessions', flush=True)
    done = set(json.load(open(STATE))['done']) if os.path.exists(STATE) else set()
    days = [d for d in sorted(p.day.unique()) if d not in done]
    print(f'{len(days)} sessions to walk ({len(done)} done)', flush=True)
    u = universe()[['day', 'symbol', 'prev_close']]
    PC = {(r.day, r.symbol): r.prev_close for r in u.itertuples()}
    for i, day in enumerate(days):
        g = p[p.day == day]
        syms = sorted(set(g.ctrl.astype(str)))
        bars = load_bars(day, syms)
        arr = {}
        for s, gg in bars.items():
            a = arrays(gg)
            if a is not None:
                arr[s] = a
        rows = []
        for r in g.itertuples():
            A = arr.get(str(r.ctrl))
            if A is None:
                continue
            o, h, l, c, v, m = A
            pc = PC.get((day, str(r.ctrl)), np.nan)
            for tag, fixed in CLOCKS:
                em = int(r.entry_m) if tag == 'sig' else fixed
                e = idx_of_minute(m, em)
                if e < 1:
                    continue
                rng_pct = (h[:e].max() - l[:e].min()) / o[0] * 100.0 if o[0] > 0 else np.nan
                dn = (o[e] / pc - 1.0) * 100.0 if pc == pc and pc > 0 else np.nan
                rr, why, xm = walk_short(o, h, l, c, m, e, float(r.r_pct))
                rows.append((day, r.symbol, int(r.entry_m), str(r.ctrl), tag, em,
                             rr, why, rng_pct, dn))
        if rows:
            pd.DataFrame(rows, columns=HDR).to_csv(
                OUT, mode='a', header=not os.path.exists(OUT), index=False)
        done.add(day)
        json.dump({'done': sorted(done)}, open(STATE, 'w'))
        if i % 20 == 0 or i == len(days) - 1:
            print(f'  [{i+1}/{len(days)}] {day} ctrl {len(arr)}/{len(syms)} rows {len(rows)}',
                  flush=True)
    print('F23 WALK DONE', flush=True)


def wk_stats(a, weeks_all):
    """Weekly dollars at $100/R, green-week share over EVERY market week of the split."""
    if not len(a):
        return 0.0, 0.0, 0.0
    w = a.groupby('wk').net.sum() * RISK
    tot = float(w.sum())
    green = float((w.reindex(weeks_all).fillna(0.0) > 0).mean() * 100)
    return tot, green, float(w.reindex(weeks_all).fillna(0.0).min())


def score():
    d = pd.read_csv(OUT, dtype={'day': str, 'symbol': str, 'ctrl': str})
    n_raw = len(d)
    d['split'] = d.day.map(split_of)
    d = d[d.split.isin(SPLITS)]
    d['h'] = np.where(d.split == 'VAL', 'VAL', d.day.map(half_of))
    d['wk'] = pd.to_datetime(d.day).dt.strftime('%G-W%V')
    nofill = float((d.why == 'nofill').mean()) * 100
    # ---- Reg SHO 201 -------------------------------------------------------------------
    d['reg201'] = d.down_pct <= -10.0
    sho = float(d.reg201.mean()) * 100
    # ---- borrow ------------------------------------------------------------------------
    b = pd.read_csv(BORROW, dtype={'symbol': str})
    ok = set(b[(b.shortable.astype(str) == 'True') &
               (b.easy_to_borrow.astype(str) == 'True')].symbol)
    seen = set(b.symbol)
    d['borrow'] = d.ctrl.isin(ok)
    d['known'] = d.ctrl.isin(seen)
    print(f'== F23 — the mirror ==\n  rows walked {n_raw:,} | TRAIN+VAL {len(d):,} | '
          f'no-fill under the floor {nofill:.1f} % | Reg SHO 201 (tape <= -10 %) {sho:.2f} % | '
          f'borrow flags known for {d.known.mean()*100:.1f} % of control names, '
          f'BORROWABLE {d.borrow.mean()*100:.1f} %', flush=True)
    cls = asset_class(d.ctrl.unique())
    d['cls'] = d.ctrl.map(cls)
    d = d[(d.why != 'nofill') & d.rr.notna()].copy()
    d['net'] = d.rr - COST

    weeks = {sp: sorted(pd.to_datetime(
        pd.date_range('2025-01-01' if sp == 'TRAIN' else '2026-01-01',
                      '2025-12-31' if sp == 'TRAIN' else '2026-05-31',
                      freq='B')).strftime('%G-W%V').unique()) for sp in SPLITS}

    base = d[d.borrow & ~d.reg201]
    cells = [
        ('S-base   arm-b clock, borrowable, ex-201', base[base.clock == 'sig']),
        ('S-nm2    range-so-far <= 2 %', base[(base.clock == 'sig') & (base.rng_pct <= 2)]),
        ('S-nm3    range-so-far <= 3 %', base[(base.clock == 'sig') & (base.rng_pct <= 3)]),
        ('S-nm4    range-so-far <= 4 %', base[(base.clock == 'sig') & (base.rng_pct <= 4)]),
        ('S-c1000  entry 10:00', base[base.clock == '1000']),
        ('S-c1100  entry 11:00', base[base.clock == '1100']),
        ('S-c1200  entry 12:00', base[base.clock == '1200']),
        ('S-wrap   wrappers only', base[(base.clock == 'sig') & (base.cls == 'wrapper')]),
        ('S-stock  commons only', base[(base.clock == 'sig') & (base.cls == 'stock')]),
        ('S-nb     DIAGNOSTIC: no borrow screen', d[(d.clock == 'sig') & ~d.reg201]),
    ]
    rng = np.random.default_rng(SEED)
    out = []
    print('\n| cell | split | n | /wk | gross R | net R | $ | green % | worst wk $ | clust t | '
          'null p95 green | MDE |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for name, a in cells:
        for sp in SPLITS:
            s = a[a.split == sp]
            W = weeks[sp]
            tot, green, worst = wk_stats(s, W)
            per = len(s) / len(W)
            # count-matched permutation null on green weeks: the cell's own P&L reshuffled
            gn = np.nan
            if len(s) > 5:
                v = s.net.values * RISK
                gg = np.empty(NDRAW)
                for k in range(NDRAW):
                    tt = np.bincount(rng.integers(0, len(W), size=len(v)),
                                     weights=v, minlength=len(W))
                    gg[k] = (tt > 0).mean() * 100
                gn = float(np.percentile(gg, 95))
            t = clustered_t(s.net.values, s.day.values)
            out.append(dict(cell=name, split=sp, n=len(s), per_wk=per,
                            gross=float(s.rr.mean()) if len(s) else np.nan,
                            net=float(s.net.mean()) if len(s) else np.nan, total=tot,
                            green=green, worst=worst, clust_t=t, null_p95_green=gn,
                            mde=mde(s.net.values)))
            print(f'| {name} | {sp} | {len(s)} | {per:.1f} | '
                  f'{s.rr.mean() if len(s) else np.nan:+.4f} | '
                  f'{s.net.mean() if len(s) else np.nan:+.4f} | {tot:+,.0f} | {green:.1f} | '
                  f'{worst:+,.0f} | {t:+.2f} | {gn:.1f} | {mde(s.net.values):.3f} |')
        # halves
        g = a.groupby('h').net.mean()
        print(f'|   halves | | H1 {g.get("H1", np.nan):+.4f} | H2 {g.get("H2", np.nan):+.4f} | '
              f'VAL {g.get("VAL", np.nan):+.4f} | same-signed positive: '
              f'{all(g.get(k, -1) > 0 for k in ("H1","H2","VAL"))} | | | | | | |')
    c = pd.DataFrame(out)
    c.to_csv(f'{D7}/cells23.csv', index=False)
    # the pre-committed live-exploration bar
    piv = c.pivot(index='cell', columns='split')
    win = []
    for name, _ in cells:
        r = c[c.cell == name].set_index('split')
        if all(r.loc[sp, 'total'] > 0 and r.loc[sp, 'green'] >= 50 and r.loc[sp, 'per_wk'] >= 10
               and r.loc[sp, 'clust_t'] >= 2 for sp in SPLITS):
            win.append(name)
    print(f'\n  cells clearing the pre-committed live-exploration bar: '
          f'{win if win else "NONE (0 of 10)"}', flush=True)
    # the exit mix of the best cell
    bb = base[base.clock == 'sig']
    print(f'\n  exit mix, S-base: ' +
          ' '.join(f'{k} {100*v:.1f} %' for k, v in bb.why.value_counts(normalize=True).items()))


if __name__ == '__main__':
    {'walk': walk, 'score': score}[sys.argv[1]]()
