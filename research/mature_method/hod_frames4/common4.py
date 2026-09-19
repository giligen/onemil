#!/usr/bin/env python3
"""hod_frames4 — shared loaders + the SLOT MACHINE.  ONE definition for F13, F14, F15.

`load_breaks4` is `hod_frames3.common3.load_breaks` with ONE declared difference: the F10 dedicated
NBBO fetch (`hod_frames3/nbbo3.csv`) is NOT merged by default, because it did not exist when pass 3
reproduced the reference and merging it moves the B2 rebuild off the reference by 17 trades.  The
primary population of this pass therefore reproduces `B2` EXACTLY; `use_nbbo3=True` is the declared
sensitivity arm.
"""
import os, sqlite3, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_frames3')
import score as S            # noqa: E402
import score2 as S2          # noqa: E402
from common3 import BR_COLS, RD, daily_ctx, sigset, admit, clustered_t, halves   # noqa: E402,F401
from research.scripts.pit_listings import is_test_ticker   # noqa: E402

D4 = f'{ROOT}/research/mature_method/hod_frames4'
SPLITS = S.SPLITS
RISK = S.RISK


def load_breaks4(use_nbbo3=False, verbose=True):
    br = pd.read_csv(f'{ROOT}/research/mature_method/hod_frames2/breaks2.csv', usecols=BR_COLS, **RD)
    n0 = len(br)
    br = br[~br.day.isin(S.EARLY_CLOSE)]
    br = br[~br.symbol.map(lambda s: is_test_ticker(str(s)))]
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    dbs = set(pd.read_sql('select distinct symbol from daily_bars', con).symbol.astype(str))
    con.close()
    br = br[br.symbol.isin(dbs)]
    g = pd.read_csv(f'{ROOT}/research/mature_method/hod_frames3/feat3b.csv',
                    dtype={'day': str, 'symbol': str}, keep_default_na=False,
                    na_values=['']).drop_duplicates(['day', 'symbol', 'break_m'])
    br = br.merge(g, on=['day', 'symbol', 'break_m'], how='left')
    br['split'] = S.split_of(br.day.values)
    br['wk'] = pd.to_datetime(br.day).dt.to_period('W-FRI').astype(str)
    nb = pd.read_csv(f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv',
                     dtype={'symbol': str, 'day': str}, keep_default_na=False,
                     na_values=['']).drop_duplicates(['day', 'symbol', 'entry_m'])
    if use_nbbo3:
        e = pd.read_csv(f'{ROOT}/research/mature_method/hod_frames3/nbbo3.csv',
                        dtype={'symbol': str, 'day': str}, keep_default_na=False,
                        na_values=['']).drop_duplicates(['day', 'symbol', 'entry_m'])
        nb = pd.concat([nb, e[~e.set_index(['day', 'symbol', 'entry_m']).index.isin(
            nb.set_index(['day', 'symbol', 'entry_m']).index)]], ignore_index=True)
    br = br.merge(nb[['day', 'symbol', 'entry_m', 'spread_mean', 'ask_dec', 'bid_dec']],
                  on=['day', 'symbol', 'entry_m'], how='left')
    br = br.merge(daily_ctx(), on=['day', 'symbol'], how='left')
    df = pd.read_csv(f'{ROOT}/research/mature_method/hod_preopen_regime/day_fields.csv',
                     dtype={'day': str}, keep_default_na=False, na_values=[''])
    br = br.merge(df[['day', 'spy_r5_pct']], on='day', how='left')
    br['rng_after'] = br.rng_day - br.rng_sig
    br['dollar_frac'] = br.cum_dollar / br.adv_dollar.replace(0, np.nan) * 100
    br = br.sort_values(['day', 'symbol', 'break_m'], kind='mergesort').reset_index(drop=True)
    if verbose:
        print(f'break rows {n0} -> {len(br)} | symbol-days '
              f'{br.drop_duplicates(["day","symbol"]).shape[0]} | days {br.day.nunique()}',
              flush=True)
    return br


# ---------------------------------------------------------------- THE SLOT MACHINE
def book_ranked(s, nday=12, nconc=4, score=None, descending=True, rng=None, reserve=0,
                reserve_until=0):
    """The slot rule, generalised.  IDENTICAL to `trading.hod_break.run_book` when `score is None`
    (first-come by entry minute, ties by SYMBOL alphabetical).

    `score`: a Series aligned to `s`.  Among the signals arriving in the SAME minute, the book fills
    slots in score order instead of alphabetical order — the AVAILABILITY RAIL: a signal's only fill
    is the open of its own break bar's next minute, so a signal from an earlier minute is no longer
    buyable and can never be re-ranked into a later slot.
    `rng`: a numpy Generator -> random tie-break within the minute (the tie-break control / null).
    `reserve`: keep `reserve` of the `nconc` slots empty until minute `reserve_until` (a slot rule,
    not an admission rule: it never looks at the candidate).
    Causal freeing is `run_book`'s: a slot is free at bar k only if exit_m < k.
    """
    if not len(s):
        return s.assign(pnl=pd.Series(dtype=float))
    sc = score.reindex(s.index).astype(float).values if score is not None else None
    sym = s.symbol.astype(str).values
    ent = s.entry_m.astype(int).values
    exi = s.exit_m.astype(int).values
    dayv = s.day.values
    n = len(s)
    jit = rng.random(n) if rng is not None else np.zeros(n)
    keys = []
    for i in range(n):
        if sc is None:
            keys.append((jit[i], sym[i]))
        else:
            v = sc[i]
            v = 1e18 if v != v else (-v if descending else v)     # NaN always last
            keys.append((v, jit[i], sym[i]))
    rows = list(zip(dayv, ent, exi, keys, s.index))
    by_day = {}
    for r in rows:
        by_day.setdefault(r[0], []).append(r)
    taken = []
    for day in sorted(by_day):
        open_exits = []; n_day = 0
        for r in sorted(by_day[day], key=lambda x: (int(x[1]),) + tuple(x[3])):
            entry_m, exit_m = int(r[1]), int(r[2])
            open_exits = [e for e in open_exits if e >= entry_m]
            cap = nconc - (reserve if entry_m < reserve_until else 0)
            if n_day >= nday or len(open_exits) >= cap:
                continue
            taken.append(r[4]); open_exits.append(exit_m); n_day += 1
    b = s.loc[taken].copy()
    b['pnl'] = b.net * RISK
    return b


def _max_conc(ivs):
    """Max simultaneous open positions over a set of [entry, exit] minute intervals."""
    ev = []
    for a, b in ivs:
        ev.append((a, 1)); ev.append((b + 1, -1))
    ev.sort()
    c = m = 0
    for _, d in ev:
        c += d; m = max(m, c)
    return m


def book_oracle(s, nday=12, nconc=None, col='net'):
    """The ORACLE ceiling — NOT a strategy, a bound.  Ranks each day's signals by REALISED `col`,
    takes the top `nday`; with `nconc` set, greedily, skipping a pick that would push the day's
    simultaneous-open count above `nconc`."""
    if not len(s):
        return s.assign(pnl=pd.Series(dtype=float))
    out = []
    for _, d in s.groupby('day', sort=True):
        d = d.sort_values(col, ascending=False, kind='mergesort')
        if nconc is None:
            out.extend(list(d.index[:nday]))
            continue
        ivs = []; k = 0
        for r in d.itertuples():
            if k >= nday:
                break
            cand = ivs + [(int(r.entry_m), int(r.exit_m))]
            if _max_conc(cand) > nconc:
                continue
            ivs = cand; out.append(r.Index); k += 1
    b = s.loc[out].copy()
    b['pnl'] = b.net * RISK
    return b
