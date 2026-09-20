#!/usr/bin/env python3
"""F24 — THE PLACEBO, PORTED TO THE TWO LIVE BOOKS (ORB B+ / G3 and BF P1).

Stage 1 reproduces each book to the cent, builds the booked frame and the two control pools.
Stage 2 walks the bars (resumable per session).  Stage 3 scores.  TEST is never opened.

  python3 f24.py build   # reproduction gate + book*.csv + pool*.csv
  python3 f24.py walk    # the bar pass  (resumable, `f24_state.json`)
  python3 f24.py score   # the decomposition
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c7 import (D7, ROOT, SPLITS, arrays, asset_class, clustered_t, daily_fallback,      # noqa
                draw_band, half_of,
                idx_of_minute, load_bars, match_pool, mde, pctile, split_of, universe,
                walk_bf, walk_orb, ORB_FLAT_M, BF_FLAT_M)

NPOOL = 25
TOL = 5                           # a missing minute bar may be filled by the next print within 5 min
N_LATER = 10                      # random later non-signal minutes per booked trade (arm a')
SEED = 20260920
ORB_BOOK = f'{ROOT}/research/orb_gates2/book_G3_meas.csv'
ORB_DUMP = f'{ROOT}/research/fuckup_audit/Q_fill/dump_measured.csv'
BF_BOOK = f'{ROOT}/research/bf_frequency/runs/P1.csv'
STATE = f'{D7}/f24_state.json'
ARMS = ('sig', 'b', 'a2', 'u')


# ------------------------------------------------------------------------------------ stage 1
def build():
    print('== F24 stage 1 — reproduction gates ==', flush=True)

    # ---------------- ORB: the honest 8-slot G3 (catalyst-off) book -------------------------
    o = pd.read_csv(ORB_BOOK, dtype={'symbol': str, 'date': str},
                    keep_default_na=False, na_values=[''])
    o['split'] = o.date.map(split_of)
    ref = {'TRAIN': 282, 'VAL': 177}
    for sp in SPLITS:
        n = int((o.split == sp).sum())
        assert n == ref[sp], f'ORB repro FAIL {sp}: {n} != {ref[sp]}'
    print(f'  ORB G3-meas picks TRAIN {ref["TRAIN"]} / VAL {ref["VAL"]} — MATCH '
          f'(orb_gates2 REPORT §2); sized $ TRAIN {o[o.split=="TRAIN"]._sized_pnl.sum():+,.2f} '
          f'VAL {o[o.split=="VAL"]._sized_pnl.sum():+,.2f}', flush=True)

    # ---------------- BF: P1 exactly as it boots Monday -------------------------------------
    b = pd.read_csv(BF_BOOK, dtype={'symbol': str, 'date': str},
                    keep_default_na=False, na_values=[''])
    tot = float(b.pnl.sum())
    assert len(b) == 56 and abs(tot - 139113.67) < 0.01, f'BF repro FAIL: {len(b)} / {tot}'
    print(f'  BF P1 56 trades / ${tot:,.2f} — MATCH to the cent (bf_frequency REPORT §1)',
          flush=True)
    b['split'] = b.date.map(split_of)

    # ---------------- the booked frames the placebo needs -----------------------------------
    ob = o[(o.split.isin(SPLITS)) & (o.entered == 1)].copy()
    ob['r_pct'] = ob.range_size_pct.clip(lower=1.0)
    ob['book_r'] = ob.pnl_pct / ob.r_pct
    ob['entry_m'] = 0                       # resolved from the tape in stage 2
    ob = ob[['date', 'symbol', 'split', 'entry_price', 'r_pct', 'book_r', 'entry_m',
             'gap_pct']].rename(columns={'date': 'day'})

    bb = b[b.split.isin(SPLITS)].copy()
    bb['r_pct'] = (bb.entry_price - bb.stop_loss) / bb.entry_price * 100.0
    bb['book_r'] = (bb.exit_price - bb.entry_price) / (bb.entry_price - bb.stop_loss)
    bb['entry_m'] = bb.entry_time_et.str.slice(0, 2).astype(int) * 60 + \
        bb.entry_time_et.str.slice(3, 5).astype(int)
    bb['gap_pct'] = bb.qf_gap_pct
    bb = bb[['date', 'symbol', 'split', 'entry_price', 'r_pct', 'book_r', 'entry_m',
             'gap_pct']].rename(columns={'date': 'day'})

    # prev_close / adv20 on the SAME definition the controls are matched on
    for name, fr, gapband in (('orb', ob, True), ('bf', bb, False)):
        u = universe(set(fr.day))[['day', 'symbol', 'prev_close', 'adv20', 'advd']]
        fr = fr.merge(u, on=['day', 'symbol'], how='left')
        cov = float(fr.prev_close.notna().mean()) * 100
        gap = fr[fr.prev_close.isna()]
        if len(gap):
            fb = daily_fallback(set(zip(gap.day, gap.symbol))).set_index(['day', 'symbol'])
            fr = fr.set_index(['day', 'symbol'])
            for k in ('prev_close', 'adv20', 'advd'):
                fr[k] = fr[k].fillna(fb[k])
            fr = fr.reset_index()
        cov2 = float(fr.prev_close.notna().mean()) * 100
        print(f'  {name}: {len(fr)} booked trades, PIT-panel join {cov:.1f} % '
              f'-> {cov2:.1f} % after the daily_bars fallback', flush=True)
        fr.to_csv(f'{D7}/book_{name}.csv', index=False)

        if name == 'orb':
            sig = pd.read_csv(ORB_DUMP, usecols=['symbol', 'date'], dtype=str)
            sigset = set(zip(sig.date, sig.symbol))
        else:
            cache = pd.read_csv(f'{ROOT}/data/bull_flag_cache_causal_full_20260905.csv',
                                usecols=['symbol', 'date'], dtype=str)
            sigset = set(zip(cache.date, cache.symbol))
        fr2 = fr[fr.prev_close.notna() & fr.adv20.notna()]
        P, U, miss = match_pool(fr2, sigset, npool=NPOOL, gap_band=gapband, seed=SEED)
        P.to_csv(f'{D7}/poolb_{name}.csv', index=False)
        U.to_csv(f'{D7}/poolu_{name}.csv', index=False)
        nt = P.drop_duplicates(['day', 'symbol']).shape[0]
        print(f'  {name}: signal set {len(sigset)} symbol-days | matched pool {len(P)} rows for '
              f'{nt}/{len(fr)} trades (unmatchable {miss}) | median |dlog| {P.dist.median():.3f} | '
              f'universe-bound pool {len(U)} rows', flush=True)


# ------------------------------------------------------------------------------------ stage 2
def _orb_break_minute(h, m, level):
    """First bar at/after 09:36 whose high reaches the pick's own fill level."""
    k = np.where((m >= 576) & (h >= level))[0]
    return int(m[k[0]]) if len(k) else -1


def walk():
    rng = np.random.default_rng(SEED)
    done = set(json.load(open(STATE))['done']) if os.path.exists(STATE) else set()
    todo = []
    books = {}
    for name in ('orb', 'bf'):
        fr = pd.read_csv(f'{D7}/book_{name}.csv', dtype={'day': str, 'symbol': str})
        pb = pd.read_csv(f'{D7}/poolb_{name}.csv', dtype={'day': str, 'symbol': str, 'ctrl': str})
        pu = pd.read_csv(f'{D7}/poolu_{name}.csv', dtype={'day': str, 'symbol': str, 'ctrl': str})
        books[name] = (fr, {d: g for d, g in pb.groupby('day')},
                       {d: g for d, g in pu.groupby('day')})
        todo += [(name, d) for d in sorted(fr.day.unique())]
    todo = [t for t in todo if f'{t[0]}|{t[1]}' not in done]
    print(f'{len(todo)} (book, session) pairs to walk ({len(done)} already done)', flush=True)

    for i, (name, day) in enumerate(todo):
        fr, PB, PU = books[name]
        tr = fr[fr.day == day]
        pb = PB.get(day, None)
        pu = PU.get(day, None)
        syms = set(tr.symbol)
        for p in (pb, pu):
            if p is not None:
                syms |= set(p.ctrl.astype(str))
        bars = load_bars(day, sorted(syms))
        arr = {}
        for s, gg in bars.items():
            a = arrays(gg)
            if a is not None:
                arr[s] = a
        W = walk_orb if name == 'orb' else walk_bf
        FLAT = ORB_FLAT_M if name == 'orb' else BF_FLAT_M
        rows = []
        for r in tr.itertuples():
            A = arr.get(r.symbol)
            if A is None:
                continue
            o, h, l, c, v, m = A
            if name == 'orb':
                em = _orb_break_minute(h, m, float(r.entry_price))
                if em < 0:
                    continue
                em += 1                                   # the next bar's open — the convention
            else:
                # OBTAINABILITY (rail 1b): the booked fill minute is the minute in which price
                # ran UP THROUGH the breakout level, so that bar's OPEN is a price the engine
                # could not have had.  Entry is the NEXT bar's open — the programme's convention
                # and the same clock every control arm uses.  The bar-open version is kept as
                # `p24_bfvoid.csv` and reported as the size of that bias.
                em = int(r.entry_m) + 1
            e0 = idx_of_minute(m, em)
            if e0 < 0 or m[e0] > em + TOL:
                continue
            key = (name, day, r.symbol, em, r.split)
            rr, why, xm = W(o, h, l, c, m, e0, float(r.r_pct))
            if rr == rr:
                rows.append(key + ('sig', r.symbol, em, rr, why))
            # ---- arm a' : the SAME name-day at a random LATER minute with no signal ---------
            cand = m[(m > em) & (m <= FLAT - 30)]
            if len(cand):
                for mm in rng.choice(cand, size=min(N_LATER, len(cand)), replace=False):
                    rr, why, xm = W(o, h, l, c, m, idx_of_minute(m, mm), float(r.r_pct))
                    if rr == rr:
                        rows.append(key + ('a2', r.symbol, int(mm), rr, why))
            # ---- arms b and u : other names at the SAME clock -------------------------------
            for tag, pool in (('b', pb), ('u', pu)):
                if pool is None:
                    continue
                for cs in pool[pool.symbol == r.symbol].ctrl.astype(str).unique():
                    B = arr.get(cs)
                    if B is None:
                        continue
                    bo, bh, bl, bc, bv, bm = B
                    k = idx_of_minute(bm, em)
                    if k < 0 or bm[k] > em + TOL:
                        continue
                    rr, why, xm = W(bo, bh, bl, bc, bm, k, float(r.r_pct))
                    if rr == rr:
                        rows.append(key + (tag, cs, em, rr, why))
        if rows:
            pd.DataFrame(rows, columns=['book', 'day', 'symbol', 'entry_m', 'split', 'arm',
                                        'ctrl', 'ctrl_m', 'rr', 'why']).to_csv(
                f'{D7}/p24.csv', mode='a', header=not os.path.exists(f'{D7}/p24.csv'), index=False)
        done.add(f'{name}|{day}')
        json.dump({'done': sorted(done)}, open(STATE, 'w'))
        if i % 20 == 0 or i == len(todo) - 1:
            print(f'  [{i + 1}/{len(todo)}] {name} {day} syms {len(arr)} rows {len(rows)}',
                  flush=True)
    print('F24 WALK DONE', flush=True)


# ------------------------------------------------------------------------------------ stage 3
def score():
    p = pd.read_csv(f'{D7}/p24.csv', dtype={'day': str, 'symbol': str, 'ctrl': str})
    out = []
    for name in ('orb', 'bf'):
        fr = pd.read_csv(f'{D7}/book_{name}.csv', dtype={'day': str, 'symbol': str})
        d = p[p.book == name]
        sig = d[d.arm == 'sig']
        print(f'\n================ F24 · {name.upper()} ================', flush=True)
        # ---- availability -----------------------------------------------------------------
        print('| arm | split | booked | matched | coverage | ctrl/trade |')
        print('|---|---|---|---|---|---|')
        cover = {}
        for arm in ARMS:
            for sp in SPLITS:
                nb = int((fr.split == sp).sum())
                a = d[(d.arm == arm) & (d.split == sp)]
                nm = a.drop_duplicates(['day', 'symbol']).shape[0]
                per = a.groupby(['day', 'symbol']).size().median() if len(a) else 0
                cover[(arm, sp)] = nm / nb if nb else 0
                print(f'| {arm} | {sp} | {nb} | {nm} | {100*nm/max(nb,1):.1f} % | {per:.0f} |')
        # ---- parity diagnostic: the walker vs the book's own R ----------------------------
        s = sig.merge(fr[['day', 'symbol', 'book_r']], on=['day', 'symbol'], how='left')
        ok = s.book_r.notna() & np.isfinite(s.rr)
        print(f'\n  PARITY (walker vs the book\'s own R, entered picks): n={int(ok.sum())} '
              f'corr={np.corrcoef(s.rr[ok], s.book_r[ok])[0,1]:+.3f} '
              f'walker mean {s.rr[ok].mean():+.3f} vs book {s.book_r[ok].mean():+.3f}', flush=True)
        # ---- the decomposition -------------------------------------------------------------
        print('\n| object | split | n | gross R | ctrl mean | p5 | p95 | pctile | paired D | '
              'clust t | MDE |')
        print('|---|---|---|---|---|---|---|---|---|---|---|')
        for sp in SPLITS:
            sg = sig[sig.split == sp]
            base = float(sg.rr.mean())
            print(f'| **the BOOKED signal minute** | {sp} | {len(sg)} | {base:+.4f} | | | | | | '
                  f'{clustered_t(sg.rr.values, sg.day.values):+.2f} | {mde(sg.rr.values):.3f} |')
            for arm in ('b', 'a2', 'u'):
                a = d[(d.arm == arm) & (d.split == sp)]
                if not len(a):
                    continue
                mu, lo, hi, arr_ = draw_band(a, ['day', 'symbol'], 200, SEED)
                pc = pctile(base, arr_)
                # paired, day-clustered: the signal minus its own controls' mean, per trade
                cm = a.groupby(['day', 'symbol']).rr.mean().rename('cm')
                j = sg.set_index(['day', 'symbol']).join(cm, how='inner')
                dd = (j.rr - j.cm).values
                t = clustered_t(dd, j.index.get_level_values(0).values)
                out.append(dict(book=name, arm=arm, split=sp, n=len(a), n_pair=len(j),
                                sig=base, ctrl=float(a.rr.mean()), band_mu=mu, p5=lo, p95=hi,
                                pctile=pc, paired=float(np.nanmean(dd)), clust_t=t,
                                mde=mde(dd), cover=cover[(arm, sp)]))
                print(f'| {arm} | {sp} | {len(a)} | {base:+.4f} | {a.rr.mean():+.4f} | {lo:+.4f} | '
                      f'{hi:+.4f} | {pc:.1f} | {np.nanmean(dd):+.4f} | {t:+.2f} | {mde(dd):.3f} |')
        # ---- halves ------------------------------------------------------------------------
        print('\n| object | H1-2025 | H2-2025 | VAL |')
        print('|---|---|---|---|')
        for arm in ARMS:
            a = d[d.arm == arm].copy()
            a['h'] = np.where(a.split == 'VAL', 'VAL', a.day.map(half_of))
            g = a.groupby('h').rr.mean()
            print(f'| {arm} | {g.get("H1", np.nan):+.4f} | {g.get("H2", np.nan):+.4f} | '
                  f'{g.get("VAL", np.nan):+.4f} |')
        # ---- the exit mix ------------------------------------------------------------------
        print('\n| arm | split | n | mean R | WR | ' +
              ('stop % | lock % | flat % | eod % |' if name == 'orb'
               else 'stop % | trail % | flat % | eod % |'))
        print('|---|---|---|---|---|---|---|---|---|')
        for arm in ARMS:
            for sp in SPLITS:
                a = d[(d.arm == arm) & (d.split == sp)]
                if not len(a):
                    continue
                w = a.why.astype(str)
                if name == 'orb':
                    cols = [(w == 'stop').mean(), (w == 'lock').mean(),
                            (w == 'flat').mean(), (w == 'eod').mean()]
                else:
                    cols = [float((w.str.endswith('stop') & ~w.str.contains('trail')).mean()),
                            float(w.str.contains('trail').mean()),
                            float(w.str.contains('flat').mean()),
                            float(w.str.contains('eod').mean())]
                print(f'| {arm} | {sp} | {len(a)} | {a.rr.mean():+.4f} | {(a.rr>0).mean()*100:.1f} % | '
                      + ' | '.join(f'{100*x:.1f} %' for x in cols) + ' |')
    pd.DataFrame(out).to_csv(f'{D7}/cells24.csv', index=False)
    print(f'\ncells24.csv written ({len(out)} scored arm-cells)', flush=True)


if __name__ == '__main__':
    {'build': build, 'walk': walk, 'score': score}[sys.argv[1]]()
