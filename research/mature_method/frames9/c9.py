#!/usr/bin/env python3
"""frames9 — the ponds, the reproduction gates and the pond-restricted control pools.

ONE definition used by F28 (`w9.py` the walk, `s9.py` the score) and F29 (`f29.py`).

A *pond* is a POPULATION OF NAME-DAYS, not a book: the set of (session, symbol) a live book's own
universe screen would have had in play that morning.  Pass 8 proved the transplantable object is
HOD-break's DETECTOR (name-day selection, +0.183 / +0.245 R against a matched non-mover) and that the
thing killing the book is the POND it fishes in (−0.16 R).  This module builds the two ponds whose
own bracket baseline pass 7 measured as POSITIVE, so the detector can be run on them with the clock,
the stop width and the bracket held at HOD's.

Every store is opened READ-ONLY.  Nothing outside `frames9/` is written.  TEST is never returned.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for _p in ('frames9', 'frames8', 'frames7', 'hod_frames6', 'hod_frames5', 'hod_frames4',
           'hod_frames3'):
    sys.path.insert(0, f'{ROOT}/research/mature_method/{_p}')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')

D9 = f'{ROOT}/research/mature_method/frames9'
D8 = f'{ROOT}/research/mature_method/frames8'
SPLITS = ('TRAIN', 'VAL')
TEST_FROM = '2026-06-01'
RUNGS = (20.0, 10.0, 5.0)
PONDS = ('ORB', 'BF')
NPOOL = 12                      # matched controls per signal (arm b) and random ones (arm u)
SEED = 20260920
GE = ('X0', 'G3')               # the shipped +2 R cap and F25's best (the bare stop)

UNIV = f'{ROOT}/research/bf_zero/universe.csv'
ORB_FEAT = f'{ROOT}/analysis_results/orb_features_20260918_2052.csv'
BF_CACHE = f'{ROOT}/data/bull_flag_cache_causal_full_20260905.csv'
ORB_BOOK = f'{ROOT}/research/orb_gates2/book_G3_meas.csv'
BF_BOOK = f'{ROOT}/research/bf_frequency/runs/P1.csv'

# ORB's live universe screen (study_orb_broad.py) — the SAME rule live's
# `orb_engine.build_orb_universe_from_snapshots` applies.
ORB_MIN_GAP, ORB_MIN_PREV_VOL, ORB_PX = 5.0, 500_000, (3.0, 30.0)
BF_PX = (2.0, 30.0)             # the BF universe band


# --------------------------------------------------------------------------------- the day panel
_P = {}


def panel():
    """The point-in-time day panel with prev_close / prev_vol / gap / adv20, TEST cut off."""
    if 'p' in _P:
        return _P['p']
    u = pd.read_csv(UNIV, dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    u = u.rename(columns={'bar_date': 'day'})
    for k in ('open', 'high', 'low', 'close', 'volume', 'adv20', 'prev_vol'):
        u[k] = pd.to_numeric(u[k], errors='coerce')
    u = u.sort_values(['symbol', 'day'], kind='mergesort')
    u['prev_close'] = u.groupby('symbol', sort=False).close.shift(1)
    u['age'] = u.groupby('symbol', sort=False).cumcount()          # prior sessions in the panel
    u = u[u.day < TEST_FROM]                                        # FREEZE.md — TEST never loaded
    u = u[(u.prev_close > 0) & (u.close > 0) & (u.adv20 > 0)]
    u['gap_pct'] = (u.open - u.prev_close) / u.prev_close * 100.0
    u['advd'] = u.adv20 * u.close
    _P['p'] = u.reset_index(drop=True)
    return _P['p']


def ponds(verbose=True):
    """{'ORB': DataFrame, 'BF': DataFrame} — the pond members, PREREG §3."""
    u = panel()
    of = pd.read_csv(ORB_FEAT, usecols=['symbol', 'date'], dtype=str)
    orb_names = set(of.symbol)
    scr = ((u.gap_pct >= ORB_MIN_GAP) & (u.prev_vol >= ORB_MIN_PREV_VOL) &
           (u.open >= ORB_PX[0]) & (u.open <= ORB_PX[1]))
    orb = u[scr & u.symbol.isin(orb_names)].copy()

    from trading.bf_universe_filter import is_bf_eligible, load_names
    nm = load_names()
    bf_names = set(pd.read_csv(BF_CACHE, usecols=['symbol'], dtype=str).symbol)
    bf_elig = {s for s in bf_names if is_bf_eligible(s, nm)}
    bf = u[u.symbol.isin(bf_elig) & (u.open >= BF_PX[0]) & (u.open <= BF_PX[1])].copy()

    if verbose:
        print(f'  POND ORB: {len(orb)} symbol-days / {orb.symbol.nunique()} names / '
              f'{orb.day.nunique()} sessions  (candidate names {len(orb_names)})', flush=True)
        print(f'  POND BF : {len(bf)} symbol-days / {bf.symbol.nunique()} names / '
              f'{bf.day.nunique()} sessions  (detected {len(bf_names)} -> eligible {len(bf_elig)})',
              flush=True)
    return {'ORB': orb, 'BF': bf}


# --------------------------------------------------------------------------- reproduction gates
def gates(verbose=True):
    """The three asserted reproduction gates of PREREG §0.  Raises on any mismatch."""
    from common6 import base_book
    from common5 import S
    b, s = base_book(verbose=False)
    ref = {'TRAIN': (1622, -17346.0), 'VAL': (706, 893.0)}
    for sp in SPLITS:
        w = S.week_stats(b, sp)
        n, tot = ref[sp]
        assert w['n'] == n, f'B2 repro FAIL {sp}: n={w["n"]} != {n}'
        assert abs(w['total'] - tot) < 1.0, f'B2 repro FAIL {sp}: ${w["total"]:,.0f} != ${tot:,.0f}'
        if verbose:
            print(f'  G-B2 {sp}: n={w["n"]} /wk={w["per_wk"]:.1f} gross={w["gross"]:+.3f} '
                  f'net={w["net"]:+.3f} total=${w["total"]:+,.0f} — MATCH', flush=True)

    o = pd.read_csv(ORB_BOOK, dtype={'symbol': str, 'date': str},
                    keep_default_na=False, na_values=[''])
    o['split'] = np.where(o.date < '2026-01-01', 'TRAIN',
                          np.where(o.date < TEST_FROM, 'VAL', 'TEST'))
    for sp, n in (('TRAIN', 282), ('VAL', 177)):
        got = int((o.split == sp).sum())
        assert got == n, f'ORB repro FAIL {sp}: {got} != {n}'
    if verbose:
        print('  G-ORB: book_G3_meas 282 TRAIN / 177 VAL picks — MATCH', flush=True)

    bf = pd.read_csv(BF_BOOK, dtype={'symbol': str, 'date': str},
                     keep_default_na=False, na_values=[''])
    tot = float(bf.pnl.sum())
    assert len(bf) == 56 and abs(tot - 139113.67) < 0.01, f'BF repro FAIL {len(bf)} / {tot}'
    if verbose:
        print(f'  G-BF : P1 56 trades / ${tot:,.2f} — MATCH to the cent', flush=True)
    return b, s, o, bf


# ------------------------------------------------------------------------------ the signal sets
def signals(rung=5.0, verbose=True):
    """HOD's EXACT B2 cascade with only the price floor moved, TRAIN+VAL, pond flags attached."""
    from common4 import load_breaks4, admit
    from common5 import sigset5, S, S2
    br = load_breaks4(verbose=False)
    S.build_impute(S2.load_pop())
    a = admit(br, pd.Series(True, index=br.index))
    x = sigset5(a, min_price=rung)
    x = x[x.split.isin(SPLITS)].copy()
    # the matching fields come from the SAME panel the controls are drawn from
    u = panel()[['day', 'symbol', 'prev_close', 'adv20', 'advd', 'gap_pct', 'age', 'open']]
    u = u.rename(columns={'prev_close': 'prev_close_p', 'adv20': 'adv20_p', 'advd': 'advd_p',
                          'gap_pct': 'gap_pct_p', 'age': 'age_p', 'open': 'open_p'})
    x = x.merge(u, on=['day', 'symbol'], how='left')
    P = ponds(verbose=False)
    k = list(zip(x.day, x.symbol))
    for p in PONDS:
        s = set(zip(P[p].day, P[p].symbol))
        x[f'in_{p}'] = [kk in s for kk in k]
    x['in_UNION'] = x.in_ORB | x.in_BF
    x['half'] = np.where(x.day < '2025-07-01', 'H1', 'H2')
    if verbose:
        print(f'  rung ${rung:.0f}: admitted TRAIN+VAL {len(x)} | ORB {int(x.in_ORB.sum())} '
              f'| BF {int(x.in_BF.sum())} | UNION {int(x.in_UNION.sum())} '
              f'| imputed {x.imputed.mean()*100:.0f} %', flush=True)
    return x


# ------------------------------------------------------------------------- the control pools
_CLS = {}


def asset_class(symbols):
    """{symbol: 'stock'|'wrapper'|'unknown'} — the shipped offline map, then the Alpaca name."""
    from trading.orb_asset_class import classify_asset, load_class_map
    if 'nm' not in _CLS:
        ass = pd.read_csv(f'{ROOT}/data/research/alpaca_assets_all_20260905.csv',
                          dtype=str, keep_default_na=False, na_values=[''])
        _CLS['nm'] = dict(zip(ass.symbol, ass.name))
        _CLS['cmap'] = load_class_map()
        _CLS['c'] = {}
    for s in set(map(str, symbols)):
        if s not in _CLS['c']:
            _CLS['c'][s] = _CLS['cmap'].get(s) or classify_asset(s, _CLS['nm'].get(s))
    return _CLS['c']


def pools(sig, pond_df, pond, npool=NPOOL, seed=SEED):
    """arm b (matched) and arm u (random) drawn FROM THE POND, per pond signal.

    PREREG §4: distance `|dlog(prev_close)| + |dlog(adv20)|`, exact asset class, same session; a
    control is never the signal's own name and never a name that itself produced an admitted signal
    that session (the pass-6 control rule).
    """
    rng = np.random.default_rng(seed)
    s = sig[sig[f'in_{pond}']].copy()
    sigkeys = set(zip(sig.day, sig.symbol))                 # ALL admitted signals, any pond
    u = pond_df[pond_df.day.isin(set(s.day))].copy()
    cls = asset_class(list(u.symbol.unique()) + list(s.symbol.unique()))
    u['cls'] = u.symbol.map(cls)
    u['lp'] = np.log(u.prev_close)
    u['la'] = np.log(u.adv20)
    u = u[~pd.Series(list(zip(u.day, u.symbol)), index=u.index).isin(sigkeys)]
    s['cls'] = s.symbol.map(cls)
    UD = {d: g.reset_index(drop=True) for d, g in u.groupby('day')}
    rb, ru, miss = [], [], 0
    for r in s.itertuples():
        g = UD.get(r.day)
        if g is None or len(g) < 3:
            miss += 1
            continue
        gg = g[g.cls == r.cls]
        if len(gg) < npool:
            gg = g
        d = np.abs(gg.lp.values - np.log(r.prev_close_p)) + np.abs(gg.la.values - np.log(r.adv20_p))
        for k in np.argsort(d, kind='mergesort')[:npool]:
            rb.append((r.day, r.symbol, int(r.entry_m), str(gg.symbol.values[k]), float(d[k])))
        for k in rng.choice(len(g), size=min(npool, len(g)), replace=False):
            ru.append((r.day, r.symbol, int(r.entry_m), str(g.symbol.values[k])))
    B = pd.DataFrame(rb, columns=['day', 'symbol', 'entry_m', 'ctrl', 'dist'])
    U = pd.DataFrame(ru, columns=['day', 'symbol', 'entry_m', 'ctrl'])
    return B, U, miss
