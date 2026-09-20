#!/usr/bin/env python3
"""frames10 / F32 (S) — THE WRAPPER SHORT: the bar pass.

Shorts the NON-MOVING wrapper — the cohort pass 9 isolated at **-0.25 R** as the matched control of
HOD's wrapper picks, and which F23's mirror never isolated.  Population, clocks, fill, stop, cost and
the two mechanism tests are declared in `PREREG.md` §2.2-§2.3 and are NOT re-decided here.

The short walker is `frames7/c7.walk_short` REUSED VERBATIM (floored sell-limit, stop ABOVE at the
same % distance, cover at -2R, flat 15:55) — parity with F23 by construction, not by re-writing.

Every row the walk can price is emitted; the range / underlying-flat / borrow / Reg-SHO conditions
are applied at SCORE time so the mechanism decompositions have their complement buckets.

Read-only on every store.  Resumable per session (`w32_state.json`).  TEST is never touched.

  python3 w32.py build   # the wrapper panel + the anchors
  python3 w32.py walk    # the bar pass -> p32.csv
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for _p in ('frames10', 'frames7'):
    sys.path.insert(0, f'{ROOT}/research/mature_method/{_p}')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')

from c7 import arrays, idx_of_minute, load_bars, walk_short           # noqa: E402

D10 = f'{ROOT}/research/mature_method/frames10'
UNIV = f'{ROOT}/research/bf_zero/universe.csv'
PANEL = f'{D10}/panel32.csv'
STATE = f'{D10}/w32_state.json'
OUT = f'{D10}/p32.csv'
TEST_FROM = '2026-06-01'
CLOCKS = (630, 660, 720)            # 10:30, 11:00, 12:00 ET
MIN_PRICE, MIN_ADV = 5.0, 100_000
OPEN_M, FLAT_M = 570, 955
HDR = ['day', 'symbol', 'anchor', 'lev', 'kdrag', 'clock', 'e_m', 'entry', 'r_pct', 'rng_pct',
       'u_move', 'own_move', 'day_ret', 'rr', 'why', 'xm', 'prev_close', 'down_pct', 'adv20']


# ------------------------------------------------------------------------------- leverage parsing
_LEVX = re.compile(r'\b([123])(?:\.\d+)?\s*X\b', re.I)
_INV = re.compile(r'\bInverse\b|\bShort\b|\bBear\b|UltraShort', re.I)


def leverage(name):
    """(signed leverage L, variance-drag coefficient k = L(L-1)/2) from the fund name, or (nan, nan).

    PREREG §2.3: `\\b([123])X\\b`, UltraPro = 3, Ultra / UltraShort = 2, a plain Inverse/Short/Bear
    with no multiplier = 1.  Unparseable names are excluded from mechanism test (i) and counted.
    """
    if not isinstance(name, str) or not name.strip():
        return np.nan, np.nan
    inv = bool(_INV.search(name))
    m = _LEVX.search(name)
    if m:
        lev = float(m.group(1))
    elif re.search(r'UltraPro', name, re.I):
        lev = 3.0
    elif re.search(r'UltraShort|\bUltra\b', name, re.I):
        lev = 2.0
    elif inv:
        lev = 1.0
    else:
        return np.nan, np.nan
    L = -lev if inv else lev
    return L, L * (L - 1.0) / 2.0


def build():
    """The wrapper panel: PIT day rows for wrappers with a resolvable single-stock underlying."""
    from trading.orb_asset_class import classify_asset, load_class_map, underlying_anchor
    u = pd.read_csv(UNIV, dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    u = u.rename(columns={'bar_date': 'day'})
    for k in ('open', 'high', 'low', 'close', 'volume', 'adv20', 'prev_vol'):
        u[k] = pd.to_numeric(u[k], errors='coerce')
    u = u.sort_values(['symbol', 'day'], kind='mergesort')
    u['prev_close'] = u.groupby('symbol', sort=False).close.shift(1)
    u = u[u.day < TEST_FROM]                                     # FREEZE.md
    u = u[(u.prev_close > 0) & (u.close > 0) & (u.adv20 > 0)]
    ass = pd.read_csv(f'{ROOT}/data/research/alpaca_assets_all_20260905.csv', dtype=str,
                      keep_default_na=False, na_values=[''])
    nm = dict(zip(ass.symbol, ass.name))
    cmap = load_class_map()
    syms = list(u.symbol.unique())
    cls = {s: (cmap.get(s) or classify_asset(s, nm.get(s))) for s in syms}
    wr = [s for s in syms if cls[s] == 'wrapper']
    anc = {s: underlying_anchor(s, nm.get(s), cmap) for s in wr}
    lv = {s: leverage(nm.get(s)) for s in wr}
    keep = {s for s in wr if anc.get(s)}
    p = u[u.symbol.isin(keep) & (u.close >= MIN_PRICE) & (u.adv20 >= MIN_ADV)].copy()
    p['anchor'] = p.symbol.map(anc)
    p['lev'] = p.symbol.map(lambda s: lv[s][0])
    p['kdrag'] = p.symbol.map(lambda s: lv[s][1])
    p[['day', 'symbol', 'anchor', 'lev', 'kdrag', 'open', 'prev_close', 'adv20']].to_csv(
        PANEL, index=False)
    print(f'  wrappers in the panel {len(wr)} | with a single-stock anchor {len(keep)} | '
          f'unparseable leverage {sum(1 for s in keep if not np.isfinite(lv[s][0]))}', flush=True)
    print(f'  panel rows {len(p)} | sessions {p.day.nunique()} | names {p.symbol.nunique()} | '
          f'anchors {p.anchor.nunique()} | median per session {p.groupby("day").size().median():.0f}',
          flush=True)


def walk():
    p = pd.read_csv(PANEL, dtype={'day': str, 'symbol': str, 'anchor': str},
                    keep_default_na=False, na_values=[''])
    done = set(json.load(open(STATE))['done']) if os.path.exists(STATE) else set()
    days = [d for d in sorted(p.day.unique()) if d not in done]
    print(f'{len(p):,} wrapper-days | {len(days)} sessions to walk ({len(done)} done)', flush=True)
    for nd, day in enumerate(days):
        g = p[p.day == day]
        syms = sorted(set(g.symbol) | set(g.anchor.dropna()))
        bars = load_bars(day, syms)
        arr = {}
        for s, gg in bars.items():
            a = arrays(gg)
            if a is not None:
                arr[s] = a
        rows = []
        for r in g.itertuples():
            A = arr.get(str(r.symbol))
            if A is None:
                continue
            o, h, l, c, v, m = A
            if o[0] <= 0:
                continue
            kflat = np.flatnonzero(m >= FLAT_M)
            day_ret = ((float(o[kflat[0]]) if len(kflat) else float(c[-1])) / o[0] - 1.0) * 100.0
            U = arr.get(str(r.anchor)) if isinstance(r.anchor, str) else None
            for T in CLOCKS:
                e = idx_of_minute(m, T + 1)
                if e < 1 or e + 1 >= len(o):
                    continue
                rng = (float(h[:e].max()) - float(l[:e].min())) / o[0] * 100.0
                hi = float(h[:e].max())
                E = float(o[e])
                if E <= 0:
                    continue
                r_pct = (hi / E - 1.0) * 100.0
                own = (E / o[0] - 1.0) * 100.0
                u_mv = np.nan
                if U is not None:
                    uo, uh, ul, uc, uv, um = U
                    ku = np.flatnonzero(um <= T)
                    if len(ku) and uo[0] > 0:
                        u_mv = (float(uc[ku[-1]]) / uo[0] - 1.0) * 100.0
                rr, why, xm = walk_short(o, h, l, c, m, e, r_pct) if r_pct > 0 else (
                    np.nan, 'badstop', -1)
                dn = (E / r.prev_close - 1.0) * 100.0 if r.prev_close > 0 else np.nan
                rows.append((day, r.symbol, r.anchor, r.lev, r.kdrag, T, int(m[e]), E, r_pct,
                             rng, u_mv, own, day_ret, rr, why, xm, r.prev_close, dn, r.adv20))
        if rows:
            pd.DataFrame(rows, columns=HDR).to_csv(
                OUT, mode='a', header=not os.path.exists(OUT), index=False)
        done.add(day)
        json.dump({'done': sorted(done)}, open(STATE, 'w'))
        if nd % 20 == 0 or nd == len(days) - 1:
            print(f'  [{nd + 1}/{len(days)}] {day} names {len(arr)}/{len(syms)} rows {len(rows)}',
                  flush=True)
    print('W32 WALK DONE', flush=True)


if __name__ == '__main__':
    {'build': build, 'walk': walk}[sys.argv[1]]()
