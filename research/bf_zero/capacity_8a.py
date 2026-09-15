#!/usr/bin/env python3
"""bf_zero REPORT §8a — the reproducible producer of the capacity/weekly table for the gated $20 book
(written 2026-09-15 by the parity review; the 9/14 table had no script and its headline row was one random draw).

Inputs: spec_trades.csv (every spec signal under the exact live fill rule, spec_sim.py), price >= 20, break bar
<= 14:00 ET (entry_m <= 841: the entry bar is the one after the break bar, last_entry_minute 840).

THE MODEL, exactly:
  - the spread gate is modeled as a cost filter INDEPENDENT of the outcome: each signal passes with p = 0.42
    (the study's pass rate: 42.1% of all >= $5 signals, 42.6% of the $20+ band, spread_study_clean.csv);
  - a passing signal is charged a flat 0.08R (the study's median spread/R of passing signals is 0.082; the §8 cost
    model — half a spread on entry, half on non-target exits — averages 0.065R on the same signals, so 0.08 is on
    the pessimistic side); failing signals are not traded;
  - the executable book = trading.hod_break.run_book (ONE rule with the EOD check and the engine's semantics):
    first-come by entry minute, ties by symbol, 12/day, 4 concurrent, CAUSAL slot freeing (exit_m < entry_m);
  - 20 seeds of the random gate; the table reports mean [min..max] over seeds per split.
Usage: python3 research/bf_zero/capacity_8a.py  -> prints the markdown table (paste into REPORT §8a).
"""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book  # noqa: E402

D = 'research/bf_zero'
PASS_P, COST_R, N_DAY, N_CONC, N_SEEDS, MIN_PRICE, MAX_ENTRY_M = 0.42, 0.08, 12, 4, 20, 20.0, 841
SPLITS = {'TRAIN': ('2025-01-02', '2025-12-31'), 'VAL': ('2026-01-01', '2026-05-31'), 'TEST': ('2026-06-01', '2026-09-11')}
NW = {s: len(pd.period_range(a, b, freq='W-FRI')) for s, (a, b) in SPLITS.items()}

T = pd.read_csv(f'{D}/spec_trades.csv', usecols=['day', 'symbol', 'entry_m', 'exit_m', 'rr', 'why', 'price'], dtype={'symbol': str}, keep_default_na=False)
for k in ('entry_m', 'exit_m', 'rr', 'price'): T[k] = pd.to_numeric(T[k])
T['split'] = np.where(T.day < '2026-01-01', 'TRAIN', np.where(T.day < '2026-06-01', 'VAL', 'TEST'))
T['wk'] = pd.to_datetime(T.day).dt.to_period('W-FRI').astype(str)
P20 = T[(T.price >= MIN_PRICE) & (T.entry_m <= MAX_ENTRY_M)].copy()
print(f'spec signals {len(T):,} | price >= ${MIN_PRICE:.0f} & entry_m <= {MAX_ENTRY_M}: {len(P20):,} {P20.split.value_counts().to_dict()}', flush=True)


def run_book_hindsight(rows, max_per_day, max_concurrent):
    """The study's OLD rule (spec_sim/book_sim before 2026-09-15): a slot freed when exit_m > entry_m (an exit during the
    entry bar freed it in hindsight). Kept here ONLY for the one-line old-vs-causal diff."""
    taken = []; by_day = {}
    for r in rows: by_day.setdefault(r[0], []).append(r)
    for day in sorted(by_day):
        open_exits = []; n_day = 0
        for r in sorted(by_day[day], key=lambda x: (int(x[1]), str(x[3]))):
            open_exits = [e for e in open_exits if e > int(r[1])]
            if n_day >= max_per_day or len(open_exits) >= max_concurrent: continue
            taken.append(r); open_exits.append(int(r[2])); n_day += 1
    return taken


def book_df(F, book_fn=run_book):
    rows = list(zip(F.day, F.entry_m, F.exit_m, F.symbol, F.net, F.split, F.wk))
    return pd.DataFrame(book_fn(rows, N_DAY, N_CONC), columns=['day', 'entry_m', 'exit_m', 'symbol', 'net', 'split', 'wk'])


def stats(bk):
    out = {}
    for s in SPLITS:
        d = bk[bk.split == s]; nw = NW[s]
        w = d.groupby('wk').net.sum().reindex(sorted(T[T.split == s].wk.unique())).fillna(0)
        out[s] = np.array([len(d) / nw, d.net.mean(), w.sum() / nw, (w > 0).sum(), w.min()])
    return out


def fmt(a, spec):
    return ' | '.join(spec.format(a[:, i].mean(), a[:, i].min(), a[:, i].max()) for i in range(a.shape[1]))


# reference: ungated $20 book (raw rr, no cost)
U = P20.assign(net=P20.rr); ref = stats(book_df(U))
# the gated model over seeds
res = {s: [] for s in SPLITS}; res_old = {s: [] for s in SPLITS}
for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    G = P20[rng.random(len(P20)) < PASS_P].copy(); G['net'] = G.rr - COST_R
    st = stats(book_df(G)); st_old = stats(book_df(G, run_book_hindsight))
    for s in SPLITS: res[s].append(st[s]); res_old[s].append(st_old[s])

lines = [f'| book | split | trades/wk | net R/trade | weekly net R | weeks green | worst week |', '|---|---|---|---|---|---|---|']
for s in SPLITS:
    r = ref[s]
    lines.append(f'| $20, no gate (raw, reference) | {s} | {r[0]:.1f} | {r[1]:+.3f} | {r[2]:+.1f} | {int(r[3])}/{NW[s]} | {r[4]:+.1f} |')
for s in SPLITS:
    a = np.array(res[s])
    lines.append(f'| **$20 + gate (42% pass, −0.08R)** | {s} | {a[:, 0].mean():.1f} [{a[:, 0].min():.1f}..{a[:, 0].max():.1f}] | '
                 f'{a[:, 1].mean():+.3f} [{a[:, 1].min():+.3f}..{a[:, 1].max():+.3f}] | {a[:, 2].mean():+.1f} [{a[:, 2].min():+.1f}..{a[:, 2].max():+.1f}] | '
                 f'{a[:, 3].mean():.1f}/{NW[s]} [{a[:, 3].min():.0f}..{a[:, 3].max():.0f}] | {a[:, 4].mean():+.1f} [{a[:, 4].min():+.1f}..{a[:, 4].max():+.1f}] |')
print('\n'.join(lines))
diff = []
for s in SPLITS:
    a, b = np.array(res[s]), np.array(res_old[s])
    diff.append(f'{s} causal {a[:, 0].mean():.1f}/wk {a[:, 1].mean():+.3f}R {a[:, 2].mean():+.1f}R/wk vs hindsight {b[:, 0].mean():.1f}/wk {b[:, 1].mean():+.3f}R {b[:, 2].mean():+.1f}R/wk')
print('\nrun_book old (exit_m > entry_m, hindsight) vs causal (exit_m < entry_m), gated model, seed means: ' + '; '.join(diff))
print(f'\nmodel: pass p={PASS_P}, cost {COST_R}R flat on passing signals, {N_DAY}/day, {N_CONC} concurrent, {N_SEEDS} seeds; weeks per split {NW}')
print('DONE', flush=True)
