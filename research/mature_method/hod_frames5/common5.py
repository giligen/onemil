#!/usr/bin/env python3
"""hod_frames5 — shared loaders for F16 (portfolio), F18 (population), F17 (horizon).

ONE definition, reused by every score script in this pass.  Everything here is a loader or a
book-builder; no cell is scored in this file.

The population object is `hod_frames4.common4.load_breaks4()` — the all-breaks stream
(`hod_frames2/breaks2.csv`) with the pass-4 membership rules, which reproduces `B2` EXACTLY.
`sigset5` generalises `hod_frames3.common3.sigset` so the price floor and the two spread gates
become PARAMETERS (that is F18's whole point); with its defaults it IS `sigset`.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_frames4')
from common4 import (S, S2, SPLITS, RISK, D4, load_breaks4, book_ranked, book_oracle,  # noqa: E402,F401
                     clustered_t, halves, admit, daily_ctx, BR_COLS, RD)

D5 = f'{ROOT}/research/mature_method/hod_frames5'


def load_breaks5(extra_nbbo=(), verbose=True):
    """`load_breaks4` with extra measured-NBBO files merged on (day, symbol, entry_m).

    The base population is byte-identical to pass 4's (the reproduction gate).  Each path in
    `extra_nbbo` adds quote-minutes the original fetch never covered; rows already measured are
    NEVER overwritten, so the base book is unchanged where it was already measured and only gains
    measurement where it had imputation.  Declared in PREREG §4.3: this CHANGES MEMBERSHIP in both
    directions (a measured spread can fail a gate an imputed one passed; a measured `ask_dec` can
    make a fill unobtainable where a missing quote defaulted to obtainable) and both effects are
    counted in the report.
    """
    br = load_breaks4(verbose=verbose)
    if not extra_nbbo:
        return br
    k = ['day', 'symbol', 'entry_m']
    have = br[br.spread_mean.notna()].set_index(k).index
    add = []
    for p in extra_nbbo:
        if not os.path.exists(p):
            print(f'  WARNING extra NBBO file missing, skipped: {p}', flush=True)
            continue
        e = pd.read_csv(p, dtype={'symbol': str, 'day': str}, keep_default_na=False,
                        na_values=['']).drop_duplicates(k)
        add.append(e[['day', 'symbol', 'entry_m', 'spread_mean', 'ask_dec', 'bid_dec']])
    if not add:
        return br
    e = pd.concat(add, ignore_index=True).drop_duplicates(k)
    e['entry_m'] = e.entry_m.astype(int)
    e = e[~e.set_index(k).index.isin(have)].rename(
        columns={'spread_mean': '_sm', 'ask_dec': '_ad', 'bid_dec': '_bd'})
    n0 = int(br.spread_mean.notna().sum())
    br = br.merge(e, on=k, how='left')
    for a, b in (('spread_mean', '_sm'), ('ask_dec', '_ad'), ('bid_dec', '_bd')):
        br[a] = br[a].where(br[a].notna(), br[b])
    br = br.drop(columns=['_sm', '_ad', '_bd'])
    if verbose:
        print(f'  extra NBBO merged: measured break rows {n0} -> {int(br.spread_mean.notna().sum())}',
              flush=True)
    return br


# ------------------------------------------------------------------ the pre-book cascade
def sigset5(d, min_price=20.0, r_min=1.0, max_bps=100.0, max_frac_r=0.15, last_m=840,
            obtain=True):
    """The shipped pre-book cascade with the THREE bolted-on gates as parameters.

    Defaults reproduce `common3.sigset` exactly:  next_open >= $20, spread <= 100 bps,
    spread <= 15 % of R.  `max_bps=None` / `max_frac_r=None` switch a gate OFF.
    """
    d = d[(d.entry_m <= last_m + 1) & d.r_pct_n.notna() & (d.fill_capped == 1) &
          (d.r_pct_n >= r_min) & (d.next_open >= min_price)]
    x = S.attach_cost(d, 'n')
    if max_bps is not None:
        x = x[(x.sp_pct * 100) <= max_bps]
    if max_frac_r is not None:
        x = x[(x.sp_pct / x.r_pct.clip(lower=0.05)) <= max_frac_r]
    return x[x.obtainable.astype(bool)] if obtain else x


# ------------------------------------------------------------------ F16: the portfolio slot machine
def book_portfolio(s, nday=12, nconc=4, key=None, per_key=1, score=None, descending=True,
                   rng=None, consume_on_reject=False):
    """`book_ranked` plus a PORTFOLIO constraint: at most `per_key` OPEN positions sharing `key`.

    `key`: column name in `s` (e.g. 'anchor', 'venue', 'advb').  None -> identical to `book_ranked`.
    `consume_on_reject`: a candidate rejected by the portfolio rule still spends one of the day's
    `nday` (ORB's NO-REFILL invariant).  Default False = the slot may be refilled.
    The constraint is on CONCURRENTLY OPEN positions, exactly like the concurrency cap: a slot is
    free at bar k only if exit_m < k (`run_book`'s causal freeing).  Rows whose key is NaN are
    treated as their own unique key (never de-duplicated against anything) — the fail-open rule.
    Admission is untouched: the same signals, the same prices, the same exits.
    """
    if not len(s):
        return s.assign(pnl=pd.Series(dtype=float))
    sc = score.reindex(s.index).astype(float).values if score is not None else None
    sym = s.symbol.astype(str).values
    ent = s.entry_m.astype(int).values
    exi = s.exit_m.astype(int).values
    dayv = s.day.values
    if key is None:
        kv = [None] * len(s)
    else:
        kv = [(k if isinstance(k, str) and k == k else f'__{i}') for i, k in
              enumerate(s[key].astype(object).values)]
    n = len(s)
    jit = rng.random(n) if rng is not None else np.zeros(n)
    keys = []
    for i in range(n):
        if sc is None:
            keys.append((jit[i], sym[i]))
        else:
            v = sc[i]
            v = 1e18 if v != v else (-v if descending else v)
            keys.append((v, jit[i], sym[i]))
    rows = list(zip(dayv, ent, exi, keys, s.index, kv))
    by_day = {}
    for r in rows:
        by_day.setdefault(r[0], []).append(r)
    taken = []
    for day in sorted(by_day):
        open_ex = []          # list of (exit_m, key)
        n_day = 0
        for r in sorted(by_day[day], key=lambda x: (int(x[1]),) + tuple(x[3])):
            entry_m, exit_m, k = int(r[1]), int(r[2]), r[5]
            open_ex = [e for e in open_ex if e[0] >= entry_m]
            if n_day >= nday or len(open_ex) >= nconc:
                continue
            if key is not None and per_key is not None:
                if sum(1 for e in open_ex if e[1] == k) >= per_key:
                    if consume_on_reject:      # ORB's NO-REFILL invariant: the day-slot is spent
                        n_day += 1
                    continue
            taken.append(r[4]); open_ex.append((exit_m, k)); n_day += 1
    b = s.loc[taken].copy()
    b['pnl'] = b.net * RISK
    return b


# ------------------------------------------------------------------ instrument attributes
_INST = {}


def attach_instrument(s):
    """asset_class / anchor / anchor_cohort — the two fields F16 and F18 need, cheaply.

    Same construction as `hod_frames4/score15.attach` (offline 33K class map -> `classify_asset`
    on the Alpaca asset name -> `underlying_anchor`).  Static at the open; nothing point-in-time
    to violate.
    """
    from trading.orb_asset_class import classify_asset, load_class_map, underlying_anchor
    if not _INST:
        ass = pd.read_csv(f'{ROOT}/data/research/alpaca_assets_all_20260905.csv',
                          dtype=str, keep_default_na=False, na_values=[''])
        _INST['nm'] = dict(zip(ass.symbol, ass.name))
        _INST['cmap'] = load_class_map()
        _INST['cls'] = {}; _INST['anc'] = {}
    nm, cmap = _INST['nm'], _INST['cmap']
    cls, anc = _INST['cls'], _INST['anc']
    for u in s.symbol.astype(str).unique():
        if u not in cls:
            cls[u] = cmap.get(u) or classify_asset(u, nm.get(u))
            anc[u] = underlying_anchor(u, nm.get(u), cmap)
    s = s.copy()
    s['asset_class'] = s.symbol.map(cls)
    s['anchor'] = s.symbol.map(anc)
    s['anchor_cohort'] = s.groupby(['day', 'anchor']).symbol.transform('size')
    s.loc[s.anchor.isna(), 'anchor_cohort'] = np.nan
    return s


# ------------------------------------------------------------------ reporting
HDR = ('| cell                                    | split | n     | /wk   | grossR | cost   | netR   |'
       '  t    | tc    | green | worst $  | total $   | MDD $    |')
SEP = '|' + '|'.join(['-' * 6] * 13) + '|'


class Sheet:
    """Accumulates cells, prints the standard row, keeps the books for the nulls."""

    def __init__(self):
        self.cells, self.books, self._first = [], {}, True

    def show(self, name, b, note=''):
        self.books[name] = b
        if self._first:
            print(HDR); print(SEP); self._first = False
        for sp in SPLITS:
            w = S.week_stats(b, sp); d = b[b.split == sp]
            cost = float((d.rr - d.net).mean()) if len(d) else np.nan
            print(f'| {name:<39s} | {sp:5s} | {w["n"]:5d} | {w["per_wk"]:5.1f} | {w["gross"]:+.3f} | '
                  f'{cost:+.3f} | {w["net"]:+.3f} | {w["t"]:+5.2f} | {clustered_t(d):+5.2f} | '
                  f'{w["green"]:5.1f} | {w["worst"]:8.0f} | {w["total"]:9.0f} | {w["mdd"]:8.0f} |',
                  flush=True)
        g, n, ok = halves(b)
        for sp in SPLITS:
            w = S.week_stats(b, sp); d = b[b.split == sp]
            w.update(cell=name, split=sp, tc=clustered_t(d), note=note,
                     cost=float((d.rr - d.net).mean()) if len(d) else np.nan,
                     mde=2.80 * float(d.net.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan,
                     h1=g[0], h2=g[1], vgross=g[2], half_ok=ok)
            self.cells.append(w)
        print(f'    halves H1 {g[0]:+.3f} (n {n[0]}) | H2 {g[1]:+.3f} (n {n[1]}) | VAL {g[2]:+.3f} '
              f'(n {n[2]}) -> same-signed POSITIVE in all three: {ok}', flush=True)
        return b

    def nulls(self, path):
        rows = []
        print('\n== count-matched permutation nulls (2,000 draws, pick count held fixed) ==')
        print('| cell | split | green % | null mean | null p5 | null p95 | verdict |')
        print('|---|---|---|---|---|---|---|')
        for name, b in self.books.items():
            for sp in SPLITS:
                obs, mu, lo, hi = S.null_band(b, sp)
                if obs != obs:
                    continue
                v = 'ABOVE' if obs > hi else ('below' if obs < lo else 'inside')
                rows.append(dict(cell=name, split=sp, green=obs, null_mean=mu, p5=lo, p95=hi,
                                 verdict=v))
                print(f'| {name} | {sp} | {obs:.1f} | {mu:.1f} | {lo:.1f} | {hi:.1f} | {v} |',
                      flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        c = pd.DataFrame(rows).verdict.value_counts().to_dict()
        print(f'\nnull summary: {c}')
        return rows

    def dump(self, path):
        pd.DataFrame(self.cells).to_csv(path, index=False)
        print(f'\ncells -> {path}  ({len(self.cells)} rows)')
