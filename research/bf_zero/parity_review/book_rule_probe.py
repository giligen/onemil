#!/usr/bin/env python3
"""Parity review probe (read-only): how often the book-rule ambiguities occur in the spec's own book.

Runs `spec_sim.run_book` semantics over research/bf_zero/spec_trades.csv under several slot-freeing rules
and same-minute tie-break orders, and counts the events where the spec and the live engine can diverge.
Writes nothing but stdout.
"""
import sys
import numpy as np, pandas as pd

T = pd.read_csv('/home/ec2-user/onemil/research/bf_zero/spec_trades.csv', dtype={'symbol': str}, keep_default_na=False, na_values=[''])
T['split'] = np.where(T.day < '2026-01-01', 'TRAIN', np.where(T.day < '2026-06-01', 'VAL', 'TEST'))
T['row'] = np.arange(len(T))
print('rows', len(T), 'days', T.day.nunique())

# within-day order in the file = the spec's tie-break: is it symbol order?
def day_is_sorted(g):
    s = g.symbol.tolist(); return s == sorted(s)
frac_sorted = T.groupby('day').apply(day_is_sorted).mean()
print(f'days whose file order is alphabetical by symbol: {frac_sorted:.1%}')


def run_book(F, n_day, n_conc, free='spec', order='file'):
    """free: 'spec' = open while exit_m > entry_m (spec_sim.py:53); 'causal' = open while exit_m >= entry_m;
    'lag1' = open while exit_m >= entry_m - 1 (live: exit polled once a minute AFTER the bar-close drain).
    order: 'file' (spec_sim: stable sort, file order), 'rev' (reverse within a minute), 'rand'."""
    F = F.copy()
    if order == 'rev': F['k'] = -F.row
    elif order == 'rand': F['k'] = np.random.default_rng(0).permutation(len(F))
    else: F['k'] = F.row
    out = []; skipped_conc = 0; skipped_day = 0; n_ge4 = 0
    for day, g in F.sort_values(['day', 'entry_m', 'k']).groupby('day'):
        open_exits = []; taken = 0
        for r in g.itertuples():
            if free == 'spec': open_exits = [e for e in open_exits if e > r.entry_m]
            elif free == 'causal': open_exits = [e for e in open_exits if e >= r.entry_m]
            elif free == 'lag1': open_exits = [e for e in open_exits if e >= r.entry_m - 1]
            if taken >= n_day: skipped_day += 1; continue
            if len(open_exits) >= n_conc: skipped_conc += 1; continue
            out.append(r); open_exits.append(r.exit_m); taken += 1
    return pd.DataFrame(out), skipped_day, skipped_conc


def summarize(bk, label):
    parts = []
    for s in ('TRAIN', 'VAL', 'TEST'):
        d = bk[bk.split == s]; parts.append(f'{s} n={len(bk[bk.split == s])} R={d.rr.sum():+.0f}')
    print(f'  {label:52s} ' + ' | '.join(parts))


def keyset(bk):
    return set(zip(bk.day, bk.symbol))


for name, F, nd, nc in (('STUDY 8/4, all prices, last_entry 930', T, 8, 4),
                        ('CONFIG 12/4, level>=20, break bar<=840', T[(T.level >= 20) & (T.entry_m <= 841)], 12, 4)):
    print(f'\n=== {name}: {len(F)} spec signals ===')
    base, sd, sc = run_book(F, nd, nc, 'spec', 'file')
    print(f'  spec rule: taken {len(base)} | day-cap skips {sd} ({sd / len(F):.1%}) | concurrency skips {sc} ({sc / len(F):.1%})')
    if nd == 8:
        SB = pd.read_csv('/home/ec2-user/onemil/research/bf_zero/spec_book.csv', dtype={'symbol': str}, keep_default_na=False, na_values=[''])
        print(f'  replica == spec_book.csv: {keyset(base) == set(zip(SB.day, SB.symbol))} (spec_book rows {len(SB)}); pandas multi-key sort stable: {list(SB.Index) == sorted(SB.Index, key=lambda i: (T.day[i], T.entry_m[i], i))}')
    summarize(base, 'spec (exit_m > entry_m keeps open) / file order')
    for free in ('causal', 'lag1'):
        bk, sd, sc = run_book(F, nd, nc, free, 'file')
        summarize(bk, f'{free} / file order')
        a, b = keyset(base), keyset(bk)
        print(f'      vs spec: only-spec {len(a - b)}  only-{free} {len(b - a)}  symmetric diff {len(a ^ b)} ({len(a ^ b) / len(a):.1%} of the spec book)')
    for order in ('rev', 'rand'):
        bk, _, _ = run_book(F, nd, nc, 'spec', order)
        summarize(bk, f'spec / {order} tie-break')
        a, b = keyset(base), keyset(bk)
        print(f'      vs spec: symmetric diff {len(a ^ b)} ({len(a ^ b) / len(a):.1%})')

    # same-minute ties among SIGNALS, and ties where the cap is decisive (some taken, some skipped in that minute)
    g = F.groupby(['day', 'entry_m']).size()
    tie_minutes = int((g > 1).sum()); tie_signals = int(g[g > 1].sum())
    print(f'  same-minute ties: {tie_minutes} minutes / {tie_signals} signals ({tie_signals / len(F):.1%} of signals)')
    taken = keyset(base)
    F2 = F.assign(taken=[(d, s) in taken for d, s in zip(F.day, F.symbol)])
    gg = F2.groupby(['day', 'entry_m']).taken.agg(['sum', 'count'])
    dec = gg[(gg['count'] > 1) & (gg['sum'] > 0) & (gg['sum'] < gg['count'])]
    print(f'  tie minutes where the tie-break DECIDES (some taken, some not): {len(dec)} minutes, {int(dec["count"].sum())} signals, {int(dec["count"].sum() - dec["sum"].sum())} losers')

    # entries taken by the spec within 0 / 1 minute of a book exit (the slot the spec frees in hindsight)
    ex = base.groupby('day').exit_m.apply(set).to_dict()
    at0 = sum(1 for r in base.itertuples() if r.entry_m in ex.get(r.day, set()))
    at1 = sum(1 for r in base.itertuples() if (r.entry_m - 1) in ex.get(r.day, set()))
    print(f'  spec entries in the SAME minute as a book exit: {at0} ({at0 / len(base):.1%}); one minute after: {at1} ({at1 / len(base):.1%})')
    # entries whose admission REQUIRED the same-minute freed slot (4 open counting exits at entry_m)
    need = 0
    for day, gd in base.groupby('day'):
        rows = gd.sort_values(['entry_m', 'row']).itertuples(); opened = []
        for r in rows:
            n_open_causal = sum(1 for e in opened if e >= r.entry_m)
            if n_open_causal >= nc: need += 1
            opened.append(r.exit_m)
    print(f'  spec entries admitted ONLY via a same-minute (hindsight) exit: {need} ({need / len(base):.1%} of the spec book)')
    print(f'  entry_m == exit_m (0-bar trades): {(base.entry_m == base.exit_m).sum()}')

# no-fill co-occurrence from the study set (book_sim: entry_m = break bar; the spec's fill minute = break bar + 1)
H = pd.read_csv('/home/ec2-user/onemil/research/bf_zero/hodbreak_trades_cap60.csv', dtype={'symbol': str}, keep_default_na=False, na_values=[''])
nf = H[H.filled == 0]; fl = H[H.filled == 1]
print(f'\nstudy set (book_sim, cap 60 bps at the break bar): signals {len(H)} no-fill {len(nf)} ({len(nf) / len(H):.1%})')
nf_min = set(zip(nf.day, nf.entry_m + 1)); nf_next = set(zip(nf.day, nf.entry_m + 2))
co0 = sum(1 for d, m in zip(fl.day, fl.entry_m + 1) if (d, m) in nf_min)
co1 = sum(1 for d, m in zip(fl.day, fl.entry_m + 1) if (d, m) in nf_next)
print(f'  filled signals with a NO-FILL order pending in the same fill minute: {co0} ({co0 / len(fl):.1%}); in the previous minute (live cancels at the next tick): {co1} ({co1 / len(fl):.1%})')
print('DONE')
