#!/usr/bin/env python3
"""Stage K diagnostics — is the NULL real, or is it my scorer?

MEMORY (`feedback_independent_check_before_claims`): of the four wrong conclusions of 2026-09-13..16,
one was a "nothing works" verdict that turned out to be three bugs in the scorer.  A negative result
is exactly as suspect as a positive one, so before "0 of 20 cells pass" is written down, the SAME
simulate/book/cost machinery is pointed at inputs whose answer is known in advance:

  D1 market control  — random symbol-days from the same universe, same fills, same costs, per hold.
                       A liquid long-only book held H days must earn roughly the market's drift
                       gross.  If a coin flip comes back at -80 bps gross, the exit walk is broken.
  D2 random book     — the 10/20-slot book filled by a random ranking.  Isolates the book layer.
  D3 K2 decomposition— K2 is the family whose earlier-test analogue (daily_addons M41, new 252-day
                       high on volume) was TRAIN/VAL positive.  Four variants isolate WHY this stage
                       disagrees: the 7% stop, the next-open entry vs M41's signal-close entry, and
                       the slot book vs M41's unconstrained top-4-per-day.
  D4 SPY             — buy and hold, as the scale reference for every bps number in the report.

These are DIAGNOSTICS, not pre-registered cells.  They are counted in the report's cell count and
none of them can promote anything: the gate is decided by the 20 declared cells alone.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_k as B                                                    # noqa: E402

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
K = f'{ROOT}/research/fuckup_audit/K'
SEED = 71


def walk(f, idx, hold, n_days, stop_kind=None, entry_at_close=False):
    """The same exit walk as build_k.simulate, with the entry convention switchable."""
    sym, op, lo, cl = f['sym'], f['open'], f['low'], f['close']
    ends_of = {}
    for a, b in zip(f['starts'], f['ends']):
        ends_of[sym[a]] = b
    rows = []
    for i in idx:
        if f['day'][i] + hold > n_days - 1:
            continue
        b = ends_of[sym[i]]
        if not entry_at_close and i + 1 >= b:
            continue
        entry = cl[i] if entry_at_close else op[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        stop = (lo[i] if stop_kind == 'gap_low' else
                entry * 0.93 if stop_kind == 'pct7' else
                entry * 0.95 if stop_kind == 'pct5' else np.nan)
        last = min(i + hold, b - 1)
        exit_i, why = last, ('hold' if last == i + hold else 'truncated')
        if np.isfinite(stop):
            for j in range(i + 1, last + 1):
                if cl[j] <= stop:
                    exit_i, why = j, 'stop'
                    break
        px = cl[exit_i]
        if not np.isfinite(px) or px <= 0:
            continue
        dv = f['dvol20_med'][i]
        cp, cd, ca = B.cost_rt(dv)
        g = px / entry - 1.0
        rows.append((int(sym[i]), int(f['day'][i]), int(f['day'][exit_i]), why,
                     float(g), float(g - cp), float(g - cd), float(g - ca), float(dv)))
    return pd.DataFrame(rows, columns=['sym', 'sig_day', 'exit_day', 'why', 'gross', 'net',
                                       'net_daily', 'net_auction', 'dvol20_med'])


def line(name, t, days, split):
    t = t[[B.split_of(days[i]) == split for i in t.sig_day]]
    if len(t) < 30:
        return f'| {name} | {split} | {len(t)} | — | — | — | — |'
    return (f'| {name} | {split} | {len(t):,} | {t.gross.mean() * 1e4:+.1f} | '
            f'{t.net.mean() * 1e4:+.1f} | {t.net_auction.mean() * 1e4:+.1f} | '
            f'{B.tstat(t.net):+.2f} |')


def main():
    f = B.build_features()
    days = f['days']
    n_days = len(days)
    u_prim, _u_sec, _c = B.build_universe(f)
    early = np.array([d in B.EARLY_CLOSES for d in days])
    u_prim = u_prim & (~early[f['day']])
    uni_idx = np.flatnonzero(u_prim)
    B.log(f'universe rows for the controls: {len(uni_idx):,}')

    L = ['# Stage K diagnostics — controls on the scorer', '',
         'Not pre-registered cells; they cannot promote anything. They exist because a NULL needs '
         'the same scrutiny as a finding (`feedback_independent_check_before_claims`).', '']

    # ---------------------------------------------------------------- D1/D2
    rng = np.random.default_rng(SEED)
    L.append('## D1 — market control: RANDOM symbol-days from the same universe, same machinery')
    L.append('')
    L.append('| control | split | n | gross bps | net bps (PREREG cost) | net bps (auction cost) | t |')
    L.append('|---|---|---:|---:|---:|---:|---:|')
    rand_tables = {}
    for hold in (1, 2, 3, 5, 10):
        samp = rng.choice(uni_idx, size=min(40000, len(uni_idx)), replace=False)
        t = walk(f, np.sort(samp), hold, n_days)
        rand_tables[hold] = t
        for sp in ('TRAIN', 'VAL'):
            L.append(line(f'random, hold {hold}', t, days, sp))
    L.append('')
    L.append('Read: this is what a coin flip earns on the K universe under the exact fills, exits '
             'and costs of the 20 cells. It is the zero line every declared family must beat.')
    L.append('')

    # ---------------------------------------------------------------- D3
    L.append('## D3 — K2 (new 250-day high on >=2x volume) decomposed against daily_addons M41')
    L.append('')
    L.append('`research/lit_review_2026/daily_addons.md` reports M41 (new 252d high on >=1.5x '
             'volume, top-4 per day, NO stop, entry at the SIGNAL CLOSE, hold 10) at TRAIN net '
             '+103.6 bps / VAL +389.3 / TEST -183.2. Stage K\'s K2 is TRAIN net -78.8. The '
             'difference is entirely in the three conventions below, each switched one at a time '
             'on the SAME signal set.')
    L.append('')
    m2 = B.family_mask('K2', f, u_prim)
    idx2 = np.flatnonzero(m2)
    L.append('| K2 variant | split | n | gross bps | net bps (PREREG cost) | net bps (auction cost) | t |')
    L.append('|---|---|---:|---:|---:|---:|---:|')
    variants = [
        ('as declared: next-open entry + 7% stop, hold 10', dict(stop_kind='pct7')),
        ('next-open entry, NO stop, hold 10', dict(stop_kind=None)),
        ('signal-CLOSE entry (M41 convention), NO stop, hold 10', dict(stop_kind=None,
                                                                      entry_at_close=True)),
        ('signal-CLOSE entry, 7% stop, hold 10', dict(stop_kind='pct7', entry_at_close=True)),
    ]
    for nm, kw in variants:
        t = walk(f, idx2, 10, n_days, **kw)
        for sp in ('TRAIN', 'VAL'):
            L.append(line(nm, t, days, sp))
    L.append('')
    L.append('All four are the WHOLE signal set (no book, no slot limit) — the book layer is '
             'isolated in D2 below.')
    L.append('')

    # ---------------------------------------------------------------- D2 book layer
    L.append('## D2 — the book layer on random signals (does first-come slotting destroy value?)')
    L.append('')
    L.append('| book | split | n | gross bps | net bps | net bps (auction) | t |')
    L.append('|---|---|---:|---:|---:|---:|---:|')
    for hold in (3, 5):
        t = rand_tables[hold].copy()
        t['entry_day'] = t.sig_day + 1
        t['fam'] = 'K4'          # strength key 'on20' -> constant here; ranking is the random draw
        t['on20'] = rng.random(len(t))
        t['hold'] = hold
        t = t.sort_values(['entry_day', 'on20'], ascending=[True, False], kind='mergesort')
        for slots in (10, 20):
            tb = B.run_book(t, f, slots)
            x = tb[tb.booked]
            for sp in ('TRAIN', 'VAL'):
                L.append(line(f'random hold {hold}, {slots} slots', x, days, sp))
    L.append('')

    # ---------------------------------------------------------------- D4 SPY
    L.append('## D4 — SPY buy and hold, the scale reference')
    L.append('')
    sidx = np.flatnonzero(np.array([str(s) == 'SPY' for s in f['syms']]))
    if len(sidx):
        rowsel = np.flatnonzero(f['sym'] == sidx[0])
        cl = f['close'][rowsel]
        dd = [days[i] for i in f['day'][rowsel]]
        for sp, a, b in B.SPLITS[:2]:
            m = [i for i, d in enumerate(dd) if a <= d <= b]
            if len(m) > 2:
                r = cl[m[-1]] / cl[m[0]] - 1
                L.append(f'- SPY {sp} ({dd[m[0]]}..{dd[m[-1]]}, {len(m)} days): '
                         f'{r * 100:+.2f}%  = {r / len(m) * 1e4:+.1f} bps/day')
        L.append('')
    with open(f'{K}/diagnostics.md', 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    B.log(f'wrote {K}/diagnostics.md')


if __name__ == '__main__':
    main()
