#!/usr/bin/env python3
"""Cells 1,717-1,719: one-day cross-sectional reversal (PREREG_1717.md, frozen 2026-10-29).

Signal at the close of session t, ENTRY at the official open of t+1 (MOO), primary EXIT the open of t+2 (MOO),
secondary EXIT the close of t+1 (MOC).  Universe each t: eligible (screen, close >= $10, >= 15 of the last 20 bars)
-> top 1,000 by 20-session median dollar volume.  Long the lowest decile of the market-adjusted signal, short the
highest.  Costs: `--cost-bps` per side at every entry/exit (a 1x long / 1x short book pays 4 sides = 4c).

Run (separate commands, TRAIN first because the VAL pass bar reads the TRAIN daily csv):
    python3 research/daily_reversal/cell_1717.py --split train
    python3 research/daily_reversal/cell_1717.py --split val
TEST (2023+) is never loaded: `fiftytwo_wh.load_panel` refuses end > SEAL_END and the parquet filter is pushed down.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research' / 'sleeves'))
import fiftytwo_wh as fw  # noqa: E402  (load_panel, symbol_screen, SEAL_END -- reused verbatim)

log = logging.getLogger('cell_1717')
CACHE = HERE / 'cache'
CELLS = {'1717': ('1717', 'close-to-close'), '1718a': ('1718a', 'overnight component'), '1718b': ('1718b', 'intraday component')}
# split -> (signal window start, end, panel load start incl. warm-up for the 20-session windows)
SPLITS = {'train': ('2016-01-01', '2019-12-31', '2016-01-01'), 'val': ('2020-01-01', '2022-12-31', '2019-10-01')}
N_UNIV, MIN_PRICE, MIN_BARS, DV_WIN, JUMP = 1000, 10.0, 15, 20, 0.5
N_NULL, N_BOOT = 1000, 10000


# ----------------------------------------------------------------------------- arrays
def build_arrays(P: dict) -> dict:
    """Derive signals, next-day returns and the close-to-close move from the raw [D, S] panel (all float32)."""
    O, C, V = P['open'].copy(), P['close'].copy(), P['volume']
    O[~(O > 0)] = np.nan
    C[~(C > 0)] = np.nan
    D, S = C.shape
    nan = lambda: np.full((D, S), np.nan, dtype=np.float32)  # noqa: E731
    cc, onight, intra, R1, R2 = nan(), nan(), nan(), nan(), nan()
    with np.errstate(all='ignore'):
        cc[1:] = C[1:] / C[:-1] - 1
        onight[1:] = O[1:] / C[:-1] - 1
        intra[:] = C / O - 1
        R1[:-2] = O[2:] / O[1:-1] - 1          # entry open t+1 -> exit open t+2 (MOO / MOO)
        R2[:-1] = C[1:] / O[1:] - 1            # entry open t+1 -> exit close t+1 (MOO / MOC)
    return {'O': O, 'C': C, 'V': V, 'cc': cc, 'X': {'1717': cc, '1718a': onight, '1718b': intra}, 'R1': R1, 'R2': R2}


def build_universe(A: dict, allowed: np.ndarray) -> tuple[list, np.ndarray]:
    """Per session t: indices of the top-1000 by 20-session median dollar volume (sorted desc), and the EW close move."""
    C, V, cc = A['C'], A['V'], A['cc']
    D = C.shape[0]
    uni: list = [np.empty(0, dtype=np.int64)] * D
    ew = np.full(D, np.nan)
    t0 = time.time()
    for t in range(1, D):
        lo = max(0, t - DV_WIN + 1)
        dv = C[lo:t + 1].astype(np.float64) * V[lo:t + 1]
        npres = np.isfinite(dv).sum(axis=0)
        cand = np.flatnonzero(allowed & np.isfinite(C[t]) & (C[t] >= MIN_PRICE) & (npres >= MIN_BARS))
        if len(cand) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(dv[:, cand], axis=0)
        top = cand[np.argsort(-med, kind='stable')[:N_UNIV]]
        uni[t] = top
        m = cc[t, top]
        m = m[np.isfinite(m) & (np.abs(m) <= JUMP)]
        if len(m):
            ew[t] = m.mean()
        if t % 250 == 0:
            log.info('  universe t=%d/%d (%.0fs)', t, D, time.time() - t0)
    return uni, ew


# ----------------------------------------------------------------------------- the book
def run_variant(A, uni, days, key, nu=N_UNIV, ngroups=10, drop=None, collect=False, symbols=None, dates=None):
    """Walk `days` (signal sessions); long lowest 1/ngroups, short highest 1/ngroups of the market-adjusted signal.

    drop: dict t -> index array of names removed from both legs (overlap variant).  Returns (daily df, decile
    matrix [days, 10] of next-day primary returns, per-day sets, picks df).
    """
    X, cc, R1, R2, O, C = A['X'][key], A['cc'], A['R1'], A['R2'], A['O'], A['C']
    rows, decs, sets, picks = [], [], {}, []
    for t in days:
        u = uni[t][:nu]
        if len(u) == 0:
            continue
        cm, x = cc[t, u], X[t, u]
        fin = np.isfinite(cm) & np.isfinite(x)
        jump = fin & (np.abs(cm) > JUMP)
        ok = fin & ~jump
        idx = u[ok]
        n = len(idx)
        if n < 5 * ngroups:
            continue
        xr = x[ok].astype(np.float64)
        order = np.argsort(xr - xr.mean(), kind='stable')
        k = n // ngroups
        lp, sp = order[:k], order[n - k:]
        if drop is not None:
            lp = lp[~np.isin(idx[lp], drop[t])]
            sp = sp[~np.isin(idx[sp], drop[t])]
        r1, r2 = np.nan_to_num(R1[t, idx]), np.nan_to_num(R2[t, idx])
        ent_miss = ~np.isfinite(O[t + 1, idx])
        ex1_miss = ~ent_miss & ~np.isfinite(O[t + 2, idx])
        ex2_miss = ~ent_miss & ~np.isfinite(C[t + 1, idx])
        pk = np.concatenate([lp, sp])
        mean = lambda a, p: float(a[p].mean()) if len(p) else 0.0  # noqa: E731
        rows.append(dict(t=t, n_ok=n, n_jump=int(jump.sum()), n_nosig=int((~fin).sum()), n_long=len(lp), n_short=len(sp),
                         long_ret=mean(r1, lp), short_ret=mean(r1, sp), long2_ret=mean(r2, lp), short2_ret=mean(r2, sp),
                         univ1=float(r1.mean()), univ2=float(r2.mean()), n_picks=len(pk),
                         miss_entry=int(ent_miss[pk].sum()), miss_exit1=int(ex1_miss[pk].sum()), miss_exit2=int(ex2_miss[pk].sum())))
        cs = np.concatenate([[0.0], np.cumsum(r1[order])])
        b = (np.arange(ngroups + 1) * n) // ngroups
        decs.append((cs[b[1:]] - cs[b[:-1]]) / np.diff(b))
        sets[t] = dict(idx=idx, long=idx[order[:k]], short=idx[order[n - k:]], k=k)
        if collect:
            for side, p in (('long', lp), ('short', sp)):
                picks.append(pd.DataFrame({'t': t, 'side': side, 'sym_i': idx[p], 'x': x[ok][p],
                                           'ret1': R1[t, idx[p]], 'ret2': R2[t, idx[p]]}))
    df = pd.DataFrame(rows)
    pk_df = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    if collect and len(pk_df):
        pk_df['symbol'] = np.asarray(symbols, dtype=object)[pk_df['sym_i'].to_numpy()]
        pk_df['date'] = dates[pk_df['t'].to_numpy()]
        pk_df = pk_df.drop(columns=['sym_i', 't'])[['date', 'symbol', 'side', 'x', 'ret1', 'ret2']]
    return df, np.array(decs), sets, pk_df


def finish(df: pd.DataFrame, dates: pd.DatetimeIndex, c_bps: float) -> pd.DataFrame:
    """Attach dates and the gross / net long-short columns (4 sides of c per day on a 1x/1x book)."""
    d = df.copy()
    d.insert(0, 'date', dates[d.pop('t').to_numpy()])
    cost = 4 * c_bps / 1e4
    d['ls_gross'] = d.long_ret - d.short_ret
    d['ls_net'] = d.ls_gross - cost
    d['ls2_gross'] = d.long2_ret - d.short2_ret
    d['ls2_net'] = d.ls2_gross - cost
    return d


def random_null(A, sets, days, rng) -> np.ndarray:
    """[days, N_NULL] gross daily LS of random k/k books drawn from the SAME eligible names each day (count matched)."""
    R1 = A['R1']
    out = np.full((len(days), N_NULL), np.nan, dtype=np.float32)
    t0 = time.time()
    for i, t in enumerate(days):
        if t not in sets:
            continue
        idx, k = sets[t]['idx'], sets[t]['k']
        rr = np.nan_to_num(R1[t, idx]).astype(np.float32)
        perm = np.argsort(rng.random((N_NULL, len(idx)), dtype=np.float32), axis=1)[:, :2 * k]
        g = rr[perm]
        out[i] = g[:, :k].mean(axis=1) - g[:, k:].mean(axis=1)
        if i % 200 == 0:
            log.info('  null day %d/%d (%.0fs)', i, len(days), time.time() - t0)
    return out


# ----------------------------------------------------------------------------- statistics
def tstat(x: np.ndarray) -> float:
    """iid t of the mean."""
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else float('nan')


def ex_top(x: np.ndarray, frac=0.05) -> float:
    """Mean after dropping the top `frac` of days by value."""
    s = np.sort(np.asarray(x))
    return float(s[:len(s) - int(np.ceil(frac * len(s)))].mean())


def block_ci(x: np.ndarray, keys: np.ndarray, rng) -> tuple[float, float]:
    """95 % percentile CI of the mean, resampling whole blocks (years / months) with replacement."""
    u, inv = np.unique(keys, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=x), np.bincount(inv).astype(float)
    pick = rng.integers(0, len(u), (N_BOOT, len(u)))
    m = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def week_sums(x: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Sum of daily values per ISO week of the SIGNAL date (weeks are contiguous in time)."""
    iso = pd.DatetimeIndex(dates).isocalendar()
    key = (iso.year.to_numpy().astype(int) * 100 + iso.week.to_numpy().astype(int))
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    return np.add.reduceat(np.asarray(x, dtype=np.float64), starts)


def cadence(x: np.ndarray, dates: pd.DatetimeIndex, R: float) -> dict:
    """Weekly table in R units (R = sd of daily net LS): P10, green share, strong (>= +5R) weeks, gaps, bleed."""
    w = week_sums(x, dates)
    wr = w / R
    strong = np.flatnonzero(wr >= 5)
    gaps = np.diff(strong)
    bleed = [min(0.0, float(np.cumsum(wr[a + 1:b]).min())) if b - a > 1 else 0.0 for a, b in zip(strong[:-1], strong[1:])]
    return dict(weeks=len(w), p10_R=float(np.percentile(wr, 10)), green=float((w > 0).mean()), n_strong=len(strong),
                gap_med=float(np.median(gaps)) if len(gaps) else float('nan'),
                gap_p90=float(np.percentile(gaps, 90)) if len(gaps) else float('nan'),
                bleed_p10_R=float(np.percentile(bleed, 10)) if bleed else float('nan'), weekly=w)


def ls_stats(d: pd.DataFrame, rng, c_bps: float, col='ls_net', gcol='ls_gross', lcol='long_ret', scol='short_ret', ucol='univ1') -> dict:
    """Every pass-bar statistic for one daily LS series (days are the units)."""
    x = d[col].to_numpy()
    yrs = d.date.dt.year.to_numpy()
    mo = (d.date.dt.year * 100 + d.date.dt.month).to_numpy()
    c = c_bps / 1e4
    long_net = d[lcol].to_numpy() - 2 * c
    ylo, yhi = block_ci(x, yrs, rng)
    mlo, mhi = block_ci(x, mo, rng)
    R = float(x.std(ddof=1))
    return dict(n=len(x), gross=d[gcol].mean() * 1e4, net=x.mean() * 1e4, sd=R * 1e4, t=tstat(x), ci_year=(ylo * 1e4, yhi * 1e4),
                ci_month=(mlo * 1e4, mhi * 1e4), ex5=ex_top(x) * 1e4, capped=np.minimum(x, 0.02).mean() * 1e4,
                long_gross=d[lcol].mean() * 1e4, long_net=long_net.mean() * 1e4, long_t=tstat(long_net),
                long_excess=(d[lcol] - d[ucol] - 2 * c).mean() * 1e4, short_net=(-d[scol] - 2 * c).mean() * 1e4,
                R=R, green_wk=float((week_sums(x, d.date) > 0).mean()), green_wk_gross=float((week_sums(d[gcol].to_numpy(), d.date) > 0).mean()),
                year_means={int(y): x[yrs == y].mean() * 1e4 for y in np.unique(yrs)})


# ----------------------------------------------------------------------------- markdown helpers
def md(rows: list[list], header: list[str]) -> str:
    """Minimal markdown table."""
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in rows]
    return '\n'.join(out) + '\n'


def f1(v, d=1):
    """Format a float, NaN-safe."""
    return 'n/a' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{d}f}'


def pf(b: bool) -> str:
    """PASS / FAIL string."""
    return 'PASS' if b else 'FAIL'


# ----------------------------------------------------------------------------- main
def main() -> None:
    """Run one split end to end and write the CSVs, picks, and the markdown section."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=list(SPLITS), required=True)
    ap.add_argument('--cost-bps', type=float, default=5.0)
    a = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
        logging.FileHandler(CACHE / 'run_1717.log', mode='a'), logging.StreamHandler(sys.stdout)])
    t0 = time.time()
    c = a.cost_bps
    s0, s1, load0 = SPLITS[a.split]
    log.info('=== cells 1717-1719 split=%s %s..%s cost=%.1f bps/side ===', a.split, s0, s1, c)
    P = fw.load_panel(load0, s1)                         # raises for end > SEAL_END
    allowed, screen_tbl = fw.symbol_screen(P['symbols'])
    log.info('symbol screen: %d of %d symbols allowed', allowed.sum(), len(allowed))
    A = build_arrays(P)
    dates, symbols = P['dates'], P['symbols']
    uni, ew = build_universe(A, allowed)
    D = len(dates)
    days = [t for t in range(1, D - 2) if dates[t] >= pd.Timestamp(s0)]     # t+2 must exist
    log.info('signal sessions %d (%s..%s); dropped last 2 (no t+2 exit)', len(days), dates[days[0]].date(), dates[days[-1]].date())
    sess_univ = np.array([len(uni[t]) for t in days])
    rng = np.random.default_rng(1717)

    # volatility state: 20-session realised vol of the universe EW move, known at t
    rv = np.full(D, np.nan)
    for t in range(D):
        w = ew[max(0, t - 19):t + 1]
        if np.isfinite(w).sum() >= 15:
            rv[t] = np.nanstd(w, ddof=1)

    res, daily, decs, sets_all = {}, {}, {}, {}
    for key in CELLS:
        log.info('cell %s: base book', key)
        df, dec, sets, picks = run_variant(A, uni, days, key, collect=True, symbols=symbols, dates=dates)
        daily[key] = finish(df, dates, c)
        decs[key], sets_all[key] = dec, sets
        daily[key].to_csv(HERE / f'daily_1717_{key}_{a.split}.csv', index=False)
        picks.to_parquet(HERE / f'picks_1717_{key}_{a.split}.parquet', index=False)
        res[key] = dict(primary=ls_stats(daily[key], rng, c), secondary=ls_stats(daily[key], rng, c, 'ls2_net', 'ls2_gross', 'long2_ret', 'short2_ret', 'univ2'),
                        picks=picks)
        log.info('  %s: net %.2f bps/day t %.2f gross %.2f', key, res[key]['primary']['net'], res[key]['primary']['t'], res[key]['primary']['gross'])

    # ---- 1,719 report-only variants (primary exit)
    var = {}
    for key in CELLS:
        for tag, kw in (('top500', dict(nu=500)), ('quintile', dict(ngroups=5))):
            df, _, _, _ = run_variant(A, uni, days, key, **kw)
            var[(key, tag)] = ls_stats(finish(df, dates, c), rng, c)
    drop = {t: np.union1d(sets_all['1718a'][t]['long'], sets_all['1718a'][t]['short']) for t in sets_all['1718a']}
    drop = {t: drop.get(t, np.empty(0, dtype=np.int64)) for t in days}
    dfo, _, _, _ = run_variant(A, uni, days, '1717', drop=drop)
    var[('1717', 'ex_1718a_overlap')] = ls_stats(finish(dfo, dates, c), rng, c)
    log.info('variants done (%.0fs); random null...', time.time() - t0)
    null = random_null(A, sets_all['1717'], days, rng)
    d17 = daily['1717']
    t_to_i = {t: i for i, t in enumerate(days)}
    keep = np.array([t_to_i[t] for t in sorted(sets_all['1717'])])
    null = null[keep]                                    # align with the base book's days
    null_net = null - 4 * c / 1e4
    nw = np.add.reduceat(null_net.astype(np.float64), np.r_[0, np.flatnonzero(np.diff(week_key(d17.date))) + 1], axis=0)
    null_green = (nw > 0).mean(axis=0)
    ngw = np.add.reduceat(null.astype(np.float64), np.r_[0, np.flatnonzero(np.diff(week_key(d17.date))) + 1], axis=0)
    null_green_gross = (ngw > 0).mean(axis=0)
    log.info('null done (%.0fs)', time.time() - t0)

    # ---- section text
    L = [f'## Split {a.split.upper()} ({s0}..{s1}), cost {c:g} bps/side\n']
    n_long_mean = d17.n_long.mean()
    L.append('### Data check\n')
    L.append(f'* Signal sessions: **{len(d17)}** ({d17.date.min().date()}..{d17.date.max().date()}); last 2 sessions of the split dropped (no t+2 exit).')
    L.append(f'* Mean universe size (top-1000 rule, eligible names): **{sess_univ.mean():.0f}**; mean names ranked after the signal screens (1717): **{d17.n_ok.mean():.0f}**; decile size **{n_long_mean:.0f}**.')
    L.append(f'* Symbol screen (`fiftytwo_wh.symbol_screen`): {int(allowed.sum())} of {len(allowed)} panel symbols allowed. Exclusions:\n')
    L.append(md([[i, int(r.n), f'{r.share * 100:.1f}%'] for i, r in screen_tbl.iterrows()], ['reason', 'symbols', 'share']))
    rows = []
    for key in CELLS:
        d = daily[key]
        pk = d.n_picks.sum()
        rows.append([key, int(d.n_jump.sum()), f'{d.n_jump.mean():.2f}', int(d.n_nosig.sum()), f'{d.n_nosig.mean():.1f}',
                     f'{d.miss_entry.sum() / pk * 100:.3f}%', f'{d.miss_exit1.sum() / pk * 100:.3f}%', f'{d.miss_exit2.sum() / pk * 100:.3f}%'])
    L.append('\nSignal exclusions (|1-day close move| > 50 % -> out of the ranking; no-signal = missing close_{t-1}/open_t) and picked name-days with no price (counted as 0):\n')
    L.append(md(rows, ['cell', 'jump excl (total)', '/day', 'no-signal (total)', '/day', 'entry open missing', 'exit missing MOO', 'exit missing MOC']))
    rows = []
    for key in CELLS:
        pk = res[key]['picks']
        for nm, col in (('ret1 (open t+1 -> open t+2)', 'ret1'), ('ret2 (open t+1 -> close t+1)', 'ret2')):
            r = pk[col].dropna()
            rows.append([key, nm, len(r), int((r > 5).sum()), int((r < -0.8).sum()), f'{r.max() * 100:+.0f}%', f'{r.min() * 100:+.0f}%'])
    L.append('\nPrice-scale check on picks (beyond +500 % / -80 % = unadjusted corporate action):\n')
    L.append(md(rows, ['cell', 'return', 'picks w/ price', '> +500%', '< -80%', 'max', 'min']))

    L.append('\n### Per-cell results (primary exit = open t+2; units bps/day, 1x long / 1x short, days are the observations)\n')
    hdr = ['cell', 'N', 'gross LS', 'net LS', 'sd', 'iid t', 'CI year-block', 'CI month-block', 'ex-top-5%', 'capped +2%', 'long gross', 'long net', 'long excess vs univ', 'short net', 'green wk net', 'green wk gross']
    rows = []
    for key in CELLS:
        s = res[key]['primary']
        rows.append([key, s['n'], f1(s['gross']), f1(s['net']), f1(s['sd'], 0), f1(s['t'], 2), f'[{s["ci_year"][0]:.1f}, {s["ci_year"][1]:.1f}]',
                     f'[{s["ci_month"][0]:.1f}, {s["ci_month"][1]:.1f}]', f1(s['ex5']), f1(s['capped']), f1(s['long_gross']), f1(s['long_net']),
                     f1(s['long_excess']), f1(s['short_net']), f'{s["green_wk"] * 100:.1f}%', f'{s["green_wk_gross"] * 100:.1f}%'])
    L.append(md(rows, hdr))
    L.append('\nSecondary exit (close t+1, MOC):\n')
    rows = []
    for key in CELLS:
        s = res[key]['secondary']
        rows.append([key, f1(s['gross']), f1(s['net']), f1(s['t'], 2), f1(s['ex5']), f1(s['long_gross']), f1(s['long_net']), f1(s['long_excess']), f1(s['short_net']), f'{s["green_wk"] * 100:.1f}%'])
    L.append(md(rows, ['cell', 'gross LS', 'net LS', 'iid t', 'ex-top-5%', 'long gross', 'long net', 'long excess', 'short net', 'green wk']))
    L.append(f'\nRandom count-matched null (1717 days, {N_NULL} draws of random k/k books, same costs): green-week share net mean {null_green.mean() * 100:.1f}% '
             f'(p95 {np.percentile(null_green, 95) * 100:.1f}%), gross mean {null_green_gross.mean() * 100:.1f}% (p95 {np.percentile(null_green_gross, 95) * 100:.1f}%). '
             f'Mean universe EW next-day open-to-open return (long beta reference) {d17.univ1.mean() * 1e4:.1f} bps/day.\n')

    L.append('\n### 1,719 decile tables (mean next-day open-to-open return by signal decile, bps; decile 1 = lowest signal; minus = excess over universe mean)\n')
    rows = []
    for key in CELLS:
        m = decs[key].mean(axis=0) * 1e4
        ex = m - d17.univ1.mean() * 1e4
        rho = float(np.corrcoef(np.argsort(np.argsort(m)), np.arange(10))[0, 1])
        rows.append([key, *[f1(v, 1) for v in ex], f1(rho, 2)])
    L.append(md(rows, ['cell', *[f'D{i}' for i in range(1, 11)], 'rank corr (-1 = monotone reversal)']))

    L.append('\n### Per-year net LS (bps/day) and calendar-year t\n')
    rows = []
    for key in CELLS:
        d = daily[key]
        for y, g in d.groupby(d.date.dt.year):
            rows.append([key, y, len(g), f1(g.ls_gross.mean() * 1e4), f1(g.ls_net.mean() * 1e4), f1(tstat(g.ls_net.to_numpy()), 2), f1(g.long_ret.mean() * 1e4 - 2 * c)])
    L.append(md(rows, ['cell', 'year', 'days', 'gross', 'net', 't', 'long net']))

    L.append('\n### Realised-vol state (20-session vol of the universe EW move, terciles within the split; Nagel: reversal stronger in high vol)\n')
    d17_t = np.array([days[i] for i in keep])
    rv_d = rv[d17_t]
    lo, hi = np.nanpercentile(rv_d, [100 / 3, 200 / 3])
    rows = []
    for nm, mask in (('low', rv_d <= lo), ('mid', (rv_d > lo) & (rv_d <= hi)), ('high', rv_d > hi)):
        for key in CELLS:
            g = daily[key][mask]
            rows.append([key, nm, len(g), f1(g.ls_gross.mean() * 1e4), f1(g.ls_net.mean() * 1e4), f1(tstat(g.ls_net.to_numpy()), 2)])
    L.append(md(rows, ['cell', 'vol tercile', 'days', 'gross', 'net', 't']))

    L.append('\n### Sensitivities (primary exit)\n')
    rows = []
    for key in CELLS:
        g, ls, lg = daily[key].ls_gross.to_numpy(), None, daily[key].long_ret.to_numpy()
        for cc_ in (2.5, 3, 5, 10):
            net = g - 4 * cc_ / 1e4
            rows.append([key, cc_, f1(net.mean() * 1e4), f1(tstat(net), 2), f1(ex_top(net) * 1e4), f1((lg - 2 * cc_ / 1e4).mean() * 1e4)])
    L.append(md(rows, ['cell', 'bps/side', 'net LS', 't', 'ex-top-5%', 'long net']))
    L.append('\n(2.5 bps/side equals the PREREG\'s "hurdle ~10 bps/day" reading, i.e. cost on half-weights; the literal 5 bps/side on a 1x/1x book is 20 bps/day.)\n')
    rows = []
    for (key, tag), s in var.items():
        rows.append([key, tag, s['n'], f1(s['gross']), f1(s['net']), f1(s['t'], 2), f1(s['ex5']), f1(s['long_net']), f1(s['long_excess'])])
    L.append(md(rows, ['cell', 'variant', 'days', 'gross LS', 'net LS', 't', 'ex-top-5%', 'long net', 'long excess']))

    L.append('\n### Weekly table (cadence; R = sd of daily net LS of the cell)\n')
    rows, cad = [], {}
    for key in CELLS:
        d = daily[key]
        cad[key] = cadence(d.ls_net.to_numpy(), d.date, res[key]['primary']['R'])
        k_ = cad[key]
        rows.append([key, k_['weeks'], f1(k_['p10_R'], 2), f'{k_["green"] * 100:.1f}%', k_['n_strong'], f1(k_['gap_med'], 1), f1(k_['gap_p90'], 1), f1(k_['bleed_p10_R'], 1)])
    L.append(md(rows, ['cell', 'weeks', 'weekly P10 (R)', 'green wk', 'strong wk (>= +5R)', 'gap median (wk)', 'gap P90 (wk)', 'bleed P90 (R, worst decile of inter-strong cum)']))

    # ---- VAL pass bar
    if a.split == 'val':
        L.append('\n### VAL pass bar (frozen) -- judged per cell\n')
        for key in CELLS:
            s = res[key]['primary']
            tr_path = HERE / f'daily_1717_{key}_train.csv'
            if tr_path.exists():
                tr = pd.read_csv(tr_path, parse_dates=['date'])
                trx = tr.ls_gross.to_numpy() - 4 * c / 1e4
                tr_net, tr_t = trx.mean() * 1e4, tstat(trx)
                ymeans = [trx[tr.date.dt.year.to_numpy() == y].mean() for y in np.unique(tr.date.dt.year)]
                same = np.sign(tr_net) == np.sign(s['net']) and ((tr_t >= 1) if s['net'] > 0 else (tr_t <= -1))
                n_pos_years, n_years = int(sum(v > 0 for v in ymeans)), len(ymeans)
            else:
                tr_net = tr_t = float('nan'); same = False; n_pos_years = n_years = 0
            k_ = cad[key]
            items = [
                ('mean net LS >= +5 bps/day', f'{s["net"]:.2f}', s['net'] >= 5),
                ('iid t >= 2.0', f'{s["t"]:.2f}', s['t'] >= 2.0),
                ('ex-top-5% of days > 0', f'{s["ex5"]:.2f}', s['ex5'] > 0),
                ('winner-capped (+2%/day) > 0', f'{s["capped"]:.2f}', s['capped'] > 0),
                ('LONG leg alone net > 0 (raw)', f'{s["long_net"]:.2f} (excess vs univ {s["long_excess"]:.2f})', s['long_net'] > 0),
                ('TRAIN same sign, t >= 1', f'TRAIN net {tr_net:.2f}, t {tr_t:.2f}', bool(same)),
                ('>= 3 of 4 TRAIN years positive', f'{n_pos_years} of {n_years}', n_pos_years >= 3),
                ('green-week share >= 55 %', f'{s["green_wk"] * 100:.1f}%', s['green_wk'] >= 0.55),
                ('green-week share > random-decile null mean', f'{s["green_wk"] * 100:.1f}% vs {null_green.mean() * 100:.1f}% (p95 {np.percentile(null_green, 95) * 100:.1f}%)', s['green_wk'] > null_green.mean()),
            ]
            allp = all(i[2] for i in items)
            verdict = 'PASS' if allp else ('REJECT' if (s['net'] <= 0 or s['ex5'] <= 0 or s['long_net'] <= 0) else ('UNDETERMINED (net > 0, t < 2)' if (s['net'] > 0 and s['t'] < 2) else 'FAIL'))
            L.append(f'\n**Cell {key} ({CELLS[key][1]}) -- verdict {verdict}**; year-block 95 % CI of net LS [{s["ci_year"][0]:.1f}, {s["ci_year"][1]:.1f}] bps (3 blocks, crude), month-block [{s["ci_month"][0]:.1f}, {s["ci_month"][1]:.1f}]\n')
            L.append(md([[n, v, pf(ok)] for n, v, ok in items], ['pass-bar item', 'value', 'result']))
            cad_items = [('weekly P10 >= -2R', f1(k_['p10_R'], 2), k_['p10_R'] >= -2), ('strong-week gap median <= 3 wk', f1(k_['gap_med'], 1), k_['gap_med'] <= 3),
                         ('strong-week gap P90 <= 6 wk', f1(k_['gap_p90'], 1), k_['gap_p90'] <= 6), ('bleed between strong weeks >= -4R at P90', f1(k_['bleed_p10_R'], 1), k_['bleed_p10_R'] >= -4),
                         ('>= 3 fills/week', 'by construction (200 names/day)', True)]
            L.append('Cadence bar lines (docs/cadence_bar.md; NaN gap = fewer than 2 strong weeks = FAIL):\n')
            L.append(md([[n, v, pf(bool(ok))] for n, v, ok in cad_items], ['cadence item', 'value', 'result']))

    sec = '\n'.join(L)
    (CACHE / f'section_{a.split}.md').write_text(sec)
    log.info('section written; total %.0fs', time.time() - t0)
    if a.split == 'val':
        assemble()


def week_key(dates: pd.DatetimeIndex) -> np.ndarray:
    """ISO year*100+week per signal date."""
    iso = pd.DatetimeIndex(dates).isocalendar()
    return iso.year.to_numpy().astype(int) * 100 + iso.week.to_numpy().astype(int)


def assemble() -> None:
    """Write RESULT_1717.md = header + TRAIN section + VAL section + auto caveats (hand notes appended afterwards)."""
    head = ('# RESULT -- cells 1,717 / 1,718a / 1,718b / 1,719 (one-day cross-sectional reversal), builder output\n\n'
            'Spec: `PREREG_1717.md` (frozen 2026-10-29). Script: `cell_1717.py` (`--split train|val --cost-bps 5`). TEST (2023+) was never loaded. '
            'Net = gross - 4 x cost per side (1x long / 1x short book, 5 bps/side placeholder = 20 bps/day literal reading). '
            'Not independently rebuilt yet: nothing here goes to the owner before the rebuild (ladder).\n\n')
    parts = [head]
    for sp in ('train', 'val'):
        p = CACHE / f'section_{sp}.md'
        if p.exists():
            parts.append(p.read_text() + '\n')
    (HERE / 'RESULT_1717.md').write_text('\n'.join(parts))
    log.info('RESULT_1717.md assembled')


if __name__ == '__main__':
    main()
