#!/usr/bin/env python3
"""F42 — THE CLOSING AUCTION AS AN EXIT LEG, on the three books that force-close.

  python3 f42.py walk    # re-walk each book twice: force-close vs MOC (resumable per session)
  python3 f42.py score   # the 6 cells + the (official close - force-close print) distribution

The MOC variant keeps the STOP LIVE THROUGH 16:00 — a position that the shipped rule flattens at
15:45 / 15:55 can still be stopped at 15:52 under MOC.  That is walked, never assumed.

Declared in `frames13/PREREG.md` §3 BEFORE any cell was read.  Read-only on every store; nothing
is written outside `frames13/`.
"""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames7')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
from c7 import (D7, SLIP, SPLITS, arrays, clustered_t, half_of, idx_of_minute,      # noqa: E402
                load_bars, split_of, walk_bf, walk_orb, BF_FLAT_M, ORB_FLAT_M, HOD_FLAT_M)

D13 = f'{ROOT}/research/mature_method/frames13'
OUT = f'{D13}/p42.csv'
STATE = f'{D13}/f42_state.json'
LAST_M = 959                      # the last RTH 1-minute bar (15:59)
ORB_BOOK = f'{ROOT}/research/orb_gates2/book_G3_meas.csv'
BF_BOOK = f'{ROOT}/research/bf_frequency/runs/P1.csv'
HOD_BOOK = f'{ROOT}/research/mature_method/hod_frames6/book6.csv'
EOD_RATIO = 0.412                 # the programme's cost ratio on a marketable force-close leg
FLAT = {'orb': ORB_FLAT_M, 'bf': BF_FLAT_M, 'hod': HOD_FLAT_M}
RISK = {'orb': 375.0, 'bf': 150.0, 'hod': 100.0}    # the books' own live / dry risk per trade


# ------------------------------------------------------------------------------ the MOC walkers
def walk_orb_moc(o, h, l, c, m, e, r_pct, close_px, arm=1.75, lk=0.5):
    """ORB's static lock with NO 15:45 flat: the stop (and the lock) stay live to 16:00 and
    anything still open is sold in the CLOSING AUCTION at the official close."""
    n = len(o)
    if e < 0 or e + 1 >= n:
        return np.nan, '', -1
    E = float(o[e])
    R = E * (r_pct / 100.0)
    if not (R > 0) or not (E > 0) or not (close_px == close_px and close_px > 0):
        return np.nan, '', -1
    stop = E - R
    armv = E + arm * R
    locked = False
    for k in range(e + 1, n):
        if l[k] <= stop:
            px = min(stop, float(o[k])) * (1.0 - SLIP)
            return (px - E) / R, ('lock' if locked else 'stop'), int(m[k])
        if (not locked) and h[k] >= armv:
            locked = True
            stop = E + lk * R
    return (float(close_px) - E) / R, 'moc', LAST_M


def walk_bf_moc(o, h, l, c, m, e, r_pct, close_px, activate_at_r=2.0, trail_r=1.0,
                partial=True, partial_r=2.0, partial_frac=0.5):
    """BF P1's R-trail + 50 % @ +2 R partial with NO 15:45 flat; the remainder is sold in the
    closing auction at the official close."""
    n = len(o)
    if e < 0 or e + 1 >= n:
        return np.nan, '', -1
    E = float(o[e])
    R = E * (r_pct / 100.0)
    if not (R > 0) or not (E > 0) or not (close_px == close_px and close_px > 0):
        return np.nan, '', -1
    stop = E - R
    hi = E
    active = False
    pdone = False
    prr = 0.0
    frac = 1.0
    ptag = ''
    for k in range(e + 1, n):
        if l[k] <= stop:
            px = min(stop, float(o[k])) * (1.0 - SLIP)
            return prr + frac * (px - E) / R, ptag + ('trail_stop' if active else 'stop'), int(m[k])
        if partial and not pdone and h[k] >= E + partial_r * R:
            pdone = True
            prr = partial_frac * (float(c[k]) - E) / R
            frac = 1.0 - partial_frac
            stop = max(stop, E)
            ptag = 'pp+'
        hi = max(hi, float(h[k]))
        if not active and (hi - E) / R >= activate_at_r:
            active = True
        if active:
            stop = max(stop, hi - trail_r * R)
    return prr + frac * (float(close_px) - E) / R, ptag + 'moc', LAST_M


def walk_hod(o, h, l, c, m, e, stop, target_r=2.0, flat_m=HOD_FLAT_M, close_px=np.nan, moc=False):
    """The HOD-break B2 bracket: stop below, a resting +2 R limit that fills on a bar CLOSE, flat
    at 15:55 — or, under `moc`, no flat and the remainder into the closing auction."""
    n = len(o)
    E = float(o[e])
    R = E - stop
    if not (R > 0):
        return np.nan, '', -1
    tgt = E + target_r * R
    for k in range(e + 1, n):
        if (not moc) and int(m[k]) >= flat_m:
            return (float(o[k]) - E) / R, 'eod', int(m[k])
        if l[k] <= stop:
            px = float(min(stop, o[k]) * (1.0 - SLIP))
            return (px - E) / R, 'stop', int(m[k])
        if c[k] >= tgt:
            return (float(c[k]) - E) / R, 'target', int(m[k])
    if moc and close_px == close_px and close_px > 0:
        return (float(close_px) - E) / R, 'moc', LAST_M
    return (float(c[-1]) - E) / R, 'eod', int(m[-1])


# ------------------------------------------------------------------------------ the books
def daily_closes(keys):
    """The OFFICIAL daily close per (day, symbol) from `daily_bars` — opened READ-ONLY."""
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    days = sorted({d for d, _ in keys})
    q = ("select bar_date as day, symbol, close from daily_bars "
         f"where bar_date between '{days[0]}' and '{days[-1]}'")
    d = pd.read_sql(q, con)
    con.close()
    d['symbol'] = d.symbol.astype(str)
    return {(a, b): float(c) for a, b, c in zip(d.day, d.symbol, d.close)}


def books():
    """The three books' trade lists, each with the fields its own walker needs."""
    o = pd.read_csv(ORB_BOOK, dtype={'symbol': str, 'date': str},
                    keep_default_na=False, na_values=[''])
    o['split'] = o.date.map(split_of)
    for sp, n in (('TRAIN', 282), ('VAL', 177)):
        assert int((o.split == sp).sum()) == n, f'ORB repro FAIL {sp}'
    ob = o[o.split.isin(SPLITS) & (o.entered == 1)].copy()
    ob = ob.rename(columns={'date': 'day'})
    ob['r_pct'] = ob.range_size_pct.clip(lower=1.0)
    ob['level'] = ob.entry_price

    b = pd.read_csv(BF_BOOK, dtype={'symbol': str, 'date': str},
                    keep_default_na=False, na_values=[''])
    assert len(b) == 56 and abs(float(b.pnl.sum()) - 139113.67) < 0.01, 'BF repro FAIL'
    bb = pd.read_csv(f'{D7}/book_bf.csv', dtype={'symbol': str, 'day': str},
                     keep_default_na=False, na_values=[''])
    bb = bb[bb.split.isin(SPLITS)].copy()

    h = pd.read_csv(HOD_BOOK, dtype={'symbol': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    h = h[h.split.isin(SPLITS)].copy()
    for sp, n in (('TRAIN', 1622), ('VAL', 706)):
        got = int((h.split == sp).sum())
        assert got == n, f'HOD B2 repro FAIL {sp}: {got} != {n}'
    print(f'  G-ORB 282/177 · G-BF 56/$139,113.67 · G-HOD-B2 1,622/706 — MATCH', flush=True)
    return ob, bb, h


def _orb_break_minute(hi, m, level):
    k = np.where((m >= 576) & (hi >= level))[0]
    return int(m[k[0]]) if len(k) else -1


def walk():
    ob, bb, hb = books()
    keys = (set(zip(ob.day, ob.symbol)) | set(zip(bb.day, bb.symbol)) |
            set(zip(hb.day, hb.symbol)))
    cl = daily_closes(keys)
    print(f'  official daily closes for {len(keys):,} book keys: '
          f'{sum(1 for k in keys if k in cl) / len(keys) * 100:.1f} % joined', flush=True)
    done = set(json.load(open(STATE))['done']) if os.path.exists(STATE) else set()
    days = sorted(set(ob.day) | set(bb.day) | set(hb.day))
    for day in days:
        if day in done:
            continue
        sel = {'orb': ob[ob.day == day], 'bf': bb[bb.day == day], 'hod': hb[hb.day == day]}
        syms = sorted({s for f in sel.values() for s in f.symbol.astype(str)})
        arr = {}
        for s, gg in load_bars(day, syms).items():
            a = arrays(gg)
            if a is not None:
                arr[s] = a
        rows = []
        for book, fr in sel.items():
            for r in fr.itertuples():
                A = arr.get(r.symbol)
                if A is None:
                    continue
                o, h, l, c, v, m = A
                cpx = cl.get((day, r.symbol), np.nan)
                if book == 'orb':
                    em = _orb_break_minute(h, m, float(r.level))
                    if em < 0:
                        continue
                    em += 1
                    e0 = idx_of_minute(m, em)
                    if e0 < 0 or m[e0] > em + 5:
                        continue
                    rr0, w0, x0 = walk_orb(o, h, l, c, m, e0, float(r.r_pct))
                    rr1, w1, x1 = walk_orb_moc(o, h, l, c, m, e0, float(r.r_pct), cpx)
                elif book == 'bf':
                    em = int(r.entry_m) + 1
                    e0 = idx_of_minute(m, em)
                    if e0 < 0 or m[e0] > em + 5:
                        continue
                    rr0, w0, x0 = walk_bf(o, h, l, c, m, e0, float(r.r_pct), partial=True)
                    rr1, w1, x1 = walk_bf_moc(o, h, l, c, m, e0, float(r.r_pct), cpx)
                else:
                    em = int(r.entry_m)
                    e0 = idx_of_minute(m, em)
                    if e0 < 0 or m[e0] > em + 5:
                        continue
                    rr0, w0, x0 = walk_hod(o, h, l, c, m, e0, float(r.stop))
                    rr1, w1, x1 = walk_hod(o, h, l, c, m, e0, float(r.stop),
                                           close_px=cpx, moc=True)
                if rr0 != rr0 or rr1 != rr1:
                    continue
                fm = FLAT[book]
                kf = idx_of_minute(m, fm)
                fc_print = float(o[kf]) if kf >= 0 and abs(int(m[kf]) - fm) <= 5 else np.nan
                rows.append((book, day, r.symbol, r.split, em, float(o[e0]),
                             float(r.r_pct) if book != 'hod' else
                             (float(o[e0]) - float(r.stop)) / float(o[e0]) * 100.0,
                             rr0, w0, x0, rr1, w1, x1, cpx, fc_print,
                             float(getattr(r, 'sp_pct', np.nan))))
        if rows:
            pd.DataFrame(rows, columns=[
                'book', 'day', 'symbol', 'split', 'entry_m', 'entry', 'r_pct',
                'rr_flat', 'why_flat', 'xm_flat', 'rr_moc', 'why_moc', 'xm_moc',
                'close_px', 'fc_print', 'sp_pct']).to_csv(
                OUT, mode='a', header=not os.path.exists(OUT), index=False)
        done.add(day)
        json.dump({'done': sorted(done)}, open(STATE, 'w'))
        if len(done) % 50 == 0:
            print(f'  {len(done)}/{len(days)} sessions', flush=True)
    print('walk done', flush=True)


def score():
    p = pd.read_csv(OUT, dtype={'symbol': str, 'day': str, 'book': str, 'split': str},
                    keep_default_na=False, na_values=[''])
    p = p[p.rr_flat.notna() & p.rr_moc.notna()]
    p['d'] = p.rr_moc - p.rr_flat
    p['fc_flat'] = p.why_flat.isin(['flat', 'eod'])
    p['half'] = [half_of(d) if split_of(d) == 'TRAIN' else 'VAL' for d in p.day]
    # the (official close - force-close print) distribution, in % of price
    p['drift_pct'] = (p.close_px - p.fc_print) / p.entry * 100.0
    # the MECHANICAL part: the marketable exit's spread the auction does not pay
    p['spread_saving_R'] = EOD_RATIO * 0.5 * p.sp_pct / p.r_pct.clip(lower=0.05)

    rows = []
    for book in ('orb', 'bf', 'hod'):
        for sp in SPLITS:
            f = p[(p.book == book) & (p.split == sp)]
            if not len(f):
                continue
            fc = f[f.fc_flat]
            # "reason changed" = the position RESOLVED before 16:00 under MOC (a stop, a lock,
            # a target) instead of being sold in the auction — not the mere relabel flat -> moc.
            chg = fc[~fc.why_moc.str.endswith('moc')]
            rows.append(dict(
                book=book, split=sp, n=len(f), n_fc=len(fc),
                fc_share=len(fc) / len(f) * 100,
                d_per_fc=float(fc.d.mean()) if len(fc) else np.nan,
                d_per_trade=float(f.d.mean()),
                d_pct_per_trade=float((f.d * f.r_pct).mean()),
                tc=clustered_t(f.d.values, f.day.values),
                reason_changed=len(chg) / max(len(fc), 1) * 100,
                drift_med=float(fc.drift_pct.median()) if len(fc) else np.nan,
                drift_p5=float(fc.drift_pct.quantile(0.05)) if len(fc) else np.nan,
                drift_p95=float(fc.drift_pct.quantile(0.95)) if len(fc) else np.nan,
                drift_mean=float(fc.drift_pct.mean()) if len(fc) else np.nan,
                spread_saving_R=float(fc.spread_saving_R.mean()) if len(fc) else np.nan,
                spread_saving_pct=float((fc.spread_saving_R * fc.r_pct).mean())
                if len(fc) else np.nan,
                d_usd_per_wk=float(f.d.sum()) * RISK[book] /
                max(pd.to_datetime(f.day).dt.to_period('W').nunique(), 1)))
    t = pd.DataFrame(rows)
    t.to_csv(f'{D13}/cells42.csv', index=False)
    pd.set_option('display.width', 260)
    pd.set_option('display.max_columns', 30)
    print('\n== THE 6 SCORED CELLS — force close -> market-on-close ==')
    print(t[['book', 'split', 'n', 'n_fc', 'fc_share', 'd_per_fc', 'd_per_trade',
             'd_pct_per_trade', 'tc', 'reason_changed', 'd_usd_per_wk']]
          .to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print('\n== (OFFICIAL CLOSE - FORCE-CLOSE PRINT) on the book\'s OWN force-closed names, '
          '% of entry price — the MEDIAN is the falsifier, never the mean ==')
    print(t[['book', 'split', 'n_fc', 'drift_med', 'drift_p5', 'drift_p95', 'drift_mean',
             'spread_saving_R', 'spread_saving_pct']]
          .to_string(index=False, float_format=lambda v: f'{v:+.4f}'))

    print('\n== THE PRE-REGISTERED FALSIFIER (PREREG §3.3): positive on H1, H2 AND VAL, '
          'and the drift MEDIAN positive on both splits ==')
    for book in ('orb', 'bf', 'hod'):
        f = p[p.book == book]
        h = f.groupby('half').d.mean()
        med = f[f.fc_flat].groupby(f[f.fc_flat].split).drift_pct.median()
        ok_h = all(h.get(k, np.nan) > 0 for k in ('H1', 'H2', 'VAL'))
        ok_m = all(med.get(k, np.nan) > 0 for k in SPLITS)
        print(f'  {book.upper():4s} halves H1 {h.get("H1", np.nan):+.4f} / '
              f'H2 {h.get("H2", np.nan):+.4f} / VAL {h.get("VAL", np.nan):+.4f} R  | '
              f'drift median TRAIN {med.get("TRAIN", np.nan):+.4f} % / '
              f'VAL {med.get("VAL", np.nan):+.4f} %  => '
              f'{"IMPROVED" if (ok_h and ok_m) else "NOT IMPROVED"}')

    print('\n== EXIT-REASON MIX under the two specs (force-closed trades only) ==')
    for book in ('orb', 'bf', 'hod'):
        f = p[(p.book == book) & p.fc_flat]
        if len(f):
            print(f'  {book.upper():4s} n_fc {len(f):5d} | under MOC: '
                  + ', '.join(f'{k} {v / len(f) * 100:.1f} %'
                              for k, v in f.why_moc.value_counts().items()))


if __name__ == '__main__':
    {'walk': walk, 'score': score}[sys.argv[1]]()
