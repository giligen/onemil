"""Cell 1,712 -- sleeve #3 low-volatility anomaly (PREREG_LOWVOL_20261029), Stage 1, rejection-only.

Run:  nice -n 19 python3 research/sleeves/lowvol.py        (writes monthly_1712.csv, prints the evaluation)

Rule: at the last session of each calendar quarter, universe = top-1,000 by 60-session median dollar volume among
names with >= 252 sessions of history, price >= $5, 60d median $vol >= $5M, name-pattern exclusions; long the 30
LOWEST realised-vol names (std of daily log returns, prior 252 sessions), equal weight, entered at the next session's
open, held 3 months (buy-and-hold drift inside the quarter), full quarterly rebalance, 15 bps round trip charged on
traded turnover.  Sensitivities: 60-session vol, monthly rebalance, top 50, 30 bps.  TEST 2023+ is sealed: the parquet
scan stops at 2022-12-31 and the cohort whose exit open falls in 2023 is dropped.
"""
from __future__ import annotations

import logging
import re
import sys
import time

import numpy as np
import pandas as pd

from fiftytwo_wh import (ASSET_MAP, CACHE_DIR, DV_WINDOW, MIN_DV, OUT_DIR, PRICE_FLOOR, ROOT, SEAL_END, load_panel,
                         symbol_screen)

log = logging.getLogger('lowvol')
# The shared asset map labels many bond / gold / T-bill / tax-exempt funds 'stock' (MINT, GLD, IAU, VTEB).  A low-vol
# sort picks exactly those, so this sleeve adds a fund-name screen (deviation from the 52WH loader, reported both ways).
FUND_RE = re.compile(r'\b(ETF|ETFS|ETN|ETP|FUND|FUNDS|INDEX|ISHARES|SPDR|VANGUARD|PROSHARES|INVESCO|WISDOMTREE|'
                     r'DIREXION|GRAYSCALE|PIMCO|EXCHANGE[- ]TRADED|BULLION|PHYSICAL|ETRACS|'
                     r'(?:GOLD|SILVER|PLATINUM|PALLADIUM|BITCOIN|ETHER|CURRENCY|COMMODITY|OIL) (?:TRUST|SHARES)|'
                     r'(?:TREASURY|BOND|MUNICIPAL|TAX[- ]EXEMPT|T-BILL)S?)\b', re.I)
N_UNIV, K, VOL_WIN, COST_RT, N_NULL, MIN_HIST = 1000, 30, 252, 15.0, 1000, 245


# ----------------------------------------------------------------------------- the rule (shared)
def month_starts(dates: pd.DatetimeIndex):
    """First-session index of every calendar month in the panel and its 'YYYY-MM' label."""
    s = pd.Series(np.arange(len(dates)), index=dates)
    first = s.groupby(dates.to_period('M')).first()
    return first.to_numpy(), [str(p) for p in first.index]


def split_of(month: str) -> str:
    """TRAIN / VAL by entry month; anything later is sealed and must never appear."""
    if month <= '2019-12':
        return 'TRAIN'
    if month <= '2022-12':
        return 'VAL'
    raise ValueError(f'month {month} is in the sealed TEST window')


def month_returns(O: np.ndarray, C: np.ndarray, ent: np.ndarray) -> np.ndarray:
    """Rm[j] = open(first session of month j+1) / open(first session of month j) - 1 per name.

    A name with an entry open but no exit open (delisted / halted) falls back to the last close before the exit
    (WARNING with the count); names with no entry open stay NaN.
    """
    nm = len(ent)
    R = np.full((nm - 1, O.shape[1]), np.nan)
    n_fb = 0
    for j in range(nm - 1):
        o0, o1 = O[ent[j]].astype(np.float64), O[ent[j + 1]].astype(np.float64)
        r = o1 / o0 - 1
        miss = np.isfinite(o0) & ~np.isfinite(o1)
        if miss.any():
            win = pd.DataFrame(C[ent[j]:ent[j + 1]]).ffill().to_numpy()[-1]
            r = np.where(miss, win / o0 - 1, r)
            n_fb += int(miss.sum())
        R[j] = r
    if n_fb:
        log.warning('exit open missing for %d name-months; fell back to last close before the exit', n_fb)
    return R


def realised_vol(C: np.ndarray, t: int, win: int) -> tuple[np.ndarray, np.ndarray]:
    """std (ddof=1) of daily log returns over the `win` sessions ending at t, and a validity mask (>= 95 % present)."""
    lo = max(0, t - win)
    lr = np.diff(np.log(np.where(C[lo:t + 1] > 0, C[lo:t + 1], np.nan).astype(np.float64)), axis=0)
    n = np.isfinite(lr).sum(axis=0)
    with np.errstate(all='ignore'):
        vol = np.nanstd(lr, axis=0, ddof=1)
    return vol, (n >= 0.95 * win) | (n >= 0.95 * (t - lo))  # t=251 has only 251 returns


def universe_at(C, V, t, allowed, entry_open, n_univ=N_UNIV) -> np.ndarray:
    """Indices of the top-`n_univ` names by 60d median dollar volume that pass the PREREG filters at close t."""
    hist = np.isfinite(C[t - 251:t + 1]).sum(axis=0) >= MIN_HIST
    dv = C[t - DV_WINDOW + 1:t + 1].astype(np.float64) * V[t - DV_WINDOW + 1:t + 1]
    ok_n = np.isfinite(dv).sum(axis=0) >= 50
    with np.errstate(all='ignore'):
        med = np.nanmedian(dv, axis=0)
    ok = allowed & hist & ok_n & (med >= MIN_DV) & (C[t] >= PRICE_FLOOR) & np.isfinite(entry_open)
    idx = np.flatnonzero(ok)
    return idx[np.argsort(-med[idx], kind='stable')][:n_univ]


def hold_cohort(R: np.ndarray, j: int, step: int, names: np.ndarray, n_sym: int):
    """Equal-weight buy-and-hold of `names` over months j..j+step-1: (monthly gross returns, drifted end weights)."""
    g = np.cumprod(1 + np.nan_to_num(R[j:j + step][:, names], nan=0.0), axis=0)     # [step, n]
    v = np.r_[1.0, g.mean(axis=1)]
    w = np.zeros(n_sym)
    w[names] = g[-1] / g[-1].sum()
    return v[1:] / v[:-1] - 1, w


def simulate(P: dict, allowed: np.ndarray, step=3, vol_win=VOL_WIN, k=K, cost_rt=COST_RT, collect_null=False,
             seed=1712, jump_screen=False) -> dict:
    """Run the rule; returns monthly rows (strategy + universe EW + SPY), pick log and optional null matrix."""
    O, C, V = P['open'], P['close'], P['volume']
    dates, symbols = P['dates'], P['symbols']
    S = len(symbols)
    ent, months = month_starts(dates)
    R = month_returns(O, C, ent)
    if jump_screen:       # sensitivity: drop name-cohorts with a >50 % one-day close move in the hold (unadjusted split artefacts)
        with np.errstate(all='ignore'):
            jm = np.nan_to_num(np.abs(C[1:].astype(np.float64) / C[:-1] - 1) > 0.5).astype(int)
        cum_j = np.vstack([np.zeros((1, C.shape[1]), int), np.cumsum(jm, axis=0)])
    spy = symbols.index('SPY')
    rng = np.random.default_rng(seed)
    rows, picks, null_rows = [], [], []
    w_old = {'s': np.zeros(S), 'u': np.zeros(S)}
    for j in range(len(ent) - 1):
        m = months[j]
        if step == 3 and int(m[5:]) not in (1, 4, 7, 10):
            continue
        t = ent[j] - 1                                           # signal close: last session before the entry month
        if t < 251 or m < '2017-01':
            continue
        if j + step > len(ent) - 1:                              # exit open would fall in the sealed window: drop
            log.info('dropping cohort %s (exit open in sealed window)', m)
            continue
        uni = universe_at(C, V, t, allowed, O[ent[j]])
        vol, ok = realised_vol(C, t, vol_win)
        cand = uni[ok[uni] & np.isfinite(vol[uni])]
        sel = cand[np.argsort(vol[cand], kind='stable')][:k]
        if len(sel) < k:
            log.warning('%s: only %d of %d low-vol names', m, len(sel), k)
        if jump_screen:
            jumped = (cum_j[ent[j + step]] - cum_j[ent[j]]) > 0
            if jumped[sel].any():
                log.warning('%s: jump screen drops %s from the strategy cohort', m, [symbols[n] for n in sel[jumped[sel]]])
            sel, uni = sel[~jumped[sel]], uni[~jumped[uni]]
        gs, ws = hold_cohort(R, j, step, sel, S)
        gu, wu = hold_cohort(R, j, step, uni, S)
        to_s, to_u = 0.5 * np.abs(ws_new_of(sel, S) - w_old['s']).sum(), 0.5 * np.abs(ws_new_of(uni, S) - w_old['u']).sum()
        w_old['s'], w_old['u'] = ws, wu
        for q in range(step):
            rows.append({'month': months[j + q], 'cohort': m, 'n_univ': len(uni), 'n_longs': len(sel),
                         'gross': gs[q], 'univ_gross': gu[q], 'turnover': to_s if q == 0 else 0.0,
                         'univ_turnover': to_u if q == 0 else 0.0, 'spy': R[j + q, spy]})
        for n in sel:
            picks.append((m, symbols[n], float(vol[n]), float(C[t, n]), float(np.nansum(R[j:j + step, n]))))
        if collect_null:
            pk = np.argpartition(rng.random((N_NULL, len(uni))), k, axis=1)[:, :k]
            g = np.cumprod(1 + np.nan_to_num(R[j:j + step][:, uni[pk]], nan=0.0), axis=0)    # [step, D, k]
            v = np.concatenate([np.ones((1, N_NULL)), g.mean(axis=2)], axis=0)
            null_rows.append(v[1:] / v[:-1] - 1)
        log.info('%s n_univ=%d picks=%d gross/m=%+.2f%% univ/m=%+.2f%% turnover=%.2f', m, len(uni), len(sel),
                 100 * gs.mean(), 100 * gu.mean(), to_s)
    df = pd.DataFrame(rows)
    df['net'] = df['gross'] - df['turnover'] * cost_rt / 1e4          # one-way fraction replaced x round-trip bps
    df['net30'] = df['gross'] - df['turnover'] * 30.0 / 1e4
    df['univ_net'] = df['univ_gross'] - df['univ_turnover'] * cost_rt / 1e4
    df['split'] = df['month'].map(split_of)
    df['year'] = df['month'].str[:4].astype(int)
    out = {'df': df, 'picks': pd.DataFrame(picks, columns=['cohort', 'symbol', 'vol', 'price', 'ret'])}
    if collect_null:
        nm = np.concatenate(null_rows, axis=0)                       # [months, D] gross
        first = df['turnover'].to_numpy() > 0
        nm = nm - first[:, None] * cost_rt / 1e4                     # random books turn over fully
        out['null'] = nm
    return out


def fund_name_mask(symbols: list[str]) -> np.ndarray:
    """True for symbols whose asset-map name looks like a fund / ETP (missing name -> not flagged, WARNING counted)."""
    amap = pd.read_csv(ASSET_MAP).drop_duplicates('symbol').set_index('symbol')['name']
    names = pd.Series(symbols).map(amap)
    if names.isna().any():
        log.warning('%d symbols have no asset-map name; fund screen cannot apply to them', int(names.isna().sum()))
    return names.fillna('').str.contains(FUND_RE).to_numpy()


def ws_new_of(names: np.ndarray, n_sym: int) -> np.ndarray:
    """Target equal weights over `names` as a full-length vector."""
    w = np.zeros(n_sym)
    w[names] = 1.0 / len(names)
    return w


# ----------------------------------------------------------------------------- metrics
def max_dd(r: np.ndarray) -> float:
    """Max peak-to-trough drawdown of the compounded monthly series."""
    eq = np.r_[1.0, np.cumprod(1 + r)]
    return float(np.max(1 - eq / np.maximum.accumulate(eq)))


def capm(y: np.ndarray, x: np.ndarray, years: np.ndarray, n_boot=5000, seed=7) -> dict:
    """OLS y = a + b x (raw monthly returns, no rf available); year-block bootstrap CI for alpha and beta."""
    def fit(yy, xx):
        b = np.cov(xx, yy, ddof=1)[0, 1] / xx.var(ddof=1)
        return yy.mean() - b * xx.mean(), b
    a, b = fit(y, x)
    rng = np.random.default_rng(seed)
    yrs = np.unique(years)
    blk = [(y[years == v], x[years == v]) for v in yrs]
    bs = []
    for _ in range(n_boot):
        pick = [blk[i] for i in rng.integers(0, len(yrs), len(yrs))]
        bs.append(fit(np.concatenate([p[0] for p in pick]), np.concatenate([p[1] for p in pick])))
    bs = np.array(bs)
    res = y - a - b * x
    se_a = np.sqrt(res.var(ddof=2) * (1 / len(y) + x.mean() ** 2 / ((x - x.mean()) ** 2).sum()))
    return {'alpha': a, 'beta': b, 'a_lo': np.percentile(bs[:, 0], 2.5), 'a_hi': np.percentile(bs[:, 0], 97.5),
            'b_lo': np.percentile(bs[:, 1], 2.5), 'b_hi': np.percentile(bs[:, 1], 97.5), 'alpha_iid_t': a / se_a}


def block_ci(x: np.ndarray, years: np.ndarray, n_boot=5000, seed=7) -> tuple[float, float]:
    """Year-block bootstrap 95 % CI of the mean of x."""
    rng = np.random.default_rng(seed)
    yrs = np.unique(years)
    blk = [x[years == v] for v in yrs]
    bs = [np.concatenate([blk[i] for i in rng.integers(0, len(yrs), len(yrs))]).mean() for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def split_metrics(d: pd.DataFrame, col='net', bench='univ_net') -> dict:
    """PREREG Evaluation block for one split slice: mean, excess vs universe/SPY, Sharpe, DD, ex-top-5 %, green, CAPM."""
    net, bu, sp, yrs = d[col].to_numpy(), d[bench].to_numpy(), d['spy'].to_numpy(), d['year'].to_numpy()
    keep = np.sort(np.argsort(-net)[int(np.ceil(0.05 * len(net))):])
    ex_u = net - bu
    lo, hi = block_ci(ex_u, yrs)
    cm = capm(net, sp, yrs)
    return {'n': len(net), 'mean': net.mean(), 'univ_mean': bu.mean(), 'spy_mean': sp.mean(), 'ex_univ': ex_u.mean(),
            'ex_lo': lo, 'ex_hi': hi, 'ex_spy': (net - sp).mean(), 'sharpe': net.mean() / net.std(ddof=1) * np.sqrt(12),
            'dd': max_dd(net), 'univ_dd': max_dd(bu), 'spy_dd': max_dd(sp), 'ex_top5': net[keep].mean(),
            'green': float((net > 0).mean()), 'univ_green': float((bu > 0).mean()), 'spy_green': float((sp > 0).mean()),
            **cm}


def pass_bar(m: dict) -> dict:
    """Frozen VAL pass bar: alpha>0 with CI>0; DD <= 0.75 x SPY DD; green >= 50 %; mean >= universe mean - 0.2 %/mo."""
    r = {'alpha_ci_gt0': m['alpha'] > 0 and m['a_lo'] > 0, 'dd_le_0.75_spy': m['dd'] <= 0.75 * m['spy_dd'],
         'green_ge_50': m['green'] >= 0.5, 'mean_ge_univ-0.2': m['mean'] >= m['univ_mean'] - 0.002}
    r['PASS'] = all(r.values())
    return r


def fmt(m: dict) -> str:
    """One-line print of a split_metrics dict (percent units)."""
    p = lambda k: f'{m[k] * 100:+.2f}'
    return (f"n={m['n']} mean {p('mean')} univ {p('univ_mean')} spy {p('spy_mean')} | ex_univ {p('ex_univ')} CI[{p('ex_lo')},{p('ex_hi')}] "
            f"ex_spy {p('ex_spy')} | alpha {p('alpha')}/mo ({m['alpha'] * 1200:+.1f}%/yr) CI[{p('a_lo')},{p('a_hi')}] iid_t {m['alpha_iid_t']:.2f} "
            f"beta {m['beta']:.2f} [{m['b_lo']:.2f},{m['b_hi']:.2f}] | sharpe {m['sharpe']:.2f} DD {m['dd'] * 100:.1f}% (univ {m['univ_dd'] * 100:.1f}, "
            f"spy {m['spy_dd'] * 100:.1f}) | ex-top5% {p('ex_top5')} | green {m['green'] * 100:.0f}% (univ {m['univ_green'] * 100:.0f}, spy {m['spy_green'] * 100:.0f})")


def overlap(df: pd.DataFrame) -> dict:
    """Monthly correlation of net with SPY, TOM, MOM and TSM (monthly_1711.csv); sealed months cut at load."""
    tom = pd.read_csv(ROOT / 'research/index_overnight/tom_1651.csv')
    tom = tom[tom['month'] <= '2022-12'].groupby('month')['net_bps'].mean() / 1e4
    mom = pd.read_csv(ROOT / 'research/momentum_weekly/1700c_monthly.csv', parse_dates=['entry_date'])
    mom = mom[mom['entry_date'] <= SEAL_END].copy()
    mom['month'] = mom['entry_date'].dt.strftime('%Y-%m')
    mom = mom.groupby('month')['net_ret'].sum(min_count=1)
    tsm = pd.read_csv(OUT_DIR / 'monthly_1711.csv').set_index('month')['net']
    out = {}
    for name, s in (('SPY', df.set_index('month')['spy']), ('TOM', tom), ('MOM', mom), ('TSM', tsm)):
        j = df.set_index('month')[['net', 'split']].join(s.rename('x'), how='inner').dropna()
        out[name] = {sp: (round(float(np.corrcoef(g['net'], g['x'])[0, 1]), 2), len(g))
                     for sp, g in list(j.groupby('split')) + [('ALL', j)]}
    return out


# ----------------------------------------------------------------------------- driver
def main() -> None:
    """Frozen rule + sensitivities on TRAIN/VAL; write monthly_1712.csv; print the evaluation."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    CACHE_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    P = load_panel('2016-01-01', '2022-12-31')
    assert P['dates'].max() <= SEAL_END
    allowed_aw, _ = symbol_screen(P['symbols'])
    fund = fund_name_mask(P['symbols'])
    allowed = allowed_aw & ~fund
    log.info('allowed as-written %d, after fund-name screen %d (%d fund-like removed) / %d symbols', allowed_aw.sum(),
             allowed.sum(), (allowed_aw & fund).sum(), len(allowed))
    base = simulate(P, allowed, collect_null=True)
    df, picks, null = base['df'], base['picks'], base['null']
    sens = {'vol60': simulate(P, allowed, vol_win=60)['df'], 'monthly': simulate(P, allowed, step=1)['df'],
            'top50': simulate(P, allowed, k=50)['df'],
            'asWritten': simulate(P, allowed_aw)['df'], 'noJump': simulate(P, allowed, jump_screen=True)['df']}
    for name, d in sens.items():
        df[f'net_{name}'] = d.set_index('month')['net'].reindex(df['month']).to_numpy()
    df['excess'] = df['net'] - df['univ_net']
    df.to_csv(OUT_DIR / 'monthly_1712.csv', index=False)
    picks.to_parquet(CACHE_DIR / 'picks_1712.parquet')
    np.save(CACHE_DIR / 'null_1712.npy', null)
    log.info('wrote monthly_1712.csv (%d months) in %.0fs', len(df), time.time() - t0)
    for sp in ('TRAIN', 'VAL'):
        sel = (df['split'] == sp).to_numpy()
        d = df[sel]
        print(f"\n==== {sp} {d.month.iloc[0]}..{d.month.iloc[-1]} n_months={sel.sum()} cohorts={d.cohort.nunique()} "
              f"universe size median {int(d.n_univ.median())} min {d.n_univ.min()} longs min {d.n_longs.min()}")
        m = split_metrics(d)
        print('FROZEN RT15 :', fmt(m))
        print('  turnover one-way per rebalance: mean %.2f (strat) | %.2f (universe EW)' % (
            d.loc[d.turnover > 0, 'turnover'].mean(), d.loc[d.univ_turnover > 0, 'univ_turnover'].mean()))
        d30 = d.assign(net=d['net30'])
        print('RT30        :', fmt(split_metrics(d30)))
        print('  pass bar RT15:', pass_bar(m))
        print('  pass bar RT30:', pass_bar(split_metrics(d30)))
        for name in sens:
            ds = d.dropna(subset=[f'net_{name}'])
            print(f'SENS {name:8s}:', fmt(split_metrics(ds.assign(net=ds[f'net_{name}']))))
        nl = null[sel]
        ng = (nl > 0).mean(axis=0)
        print(f'  null random-{K}: green mean {ng.mean() * 100:.1f}% p95 {np.percentile(ng, 95) * 100:.1f}% | actual {m["green"] * 100:.1f}% '
              f'p(null>=actual) {(ng >= m["green"]).mean():.3f} | null mean {nl.mean() * 100:+.2f}%/mo vs strat {m["mean"] * 100:+.2f} '
              f'| null DD median {np.median([max_dd(nl[:, i]) for i in range(0, N_NULL, 10)]) * 100:.1f}%')
        yr = d.groupby('year').agg(strat=('net', 'mean'), univ=('univ_net', 'mean'), spy=('spy', 'mean'), n=('net', 'size'))
        print('  per-year mean monthly % (strat/univ/spy):', {int(y): tuple(np.round(r[:3] * 100, 2)) for y, r in yr.iterrows()})
        pk = picks[picks.cohort.isin(d.cohort)]
        bad = pk[(pk.ret > 5) | (pk.ret < -0.8)]
        print(f'  price-scale: {len(pk)} name-quarters, {len(bad)} beyond +500%/-80%; ret max {pk.ret.max() * 100:+.0f}% min {pk.ret.min() * 100:+.0f}%; '
              f'median pick vol {pk.vol.median() * 100:.2f}%/day, median price {pk.price.median():.1f}, '
              f'distinct names {pk.symbol.nunique()}; top-weight symbols {pk.symbol.value_counts().head(8).to_dict()}')
    print('\nOverlap (monthly net correlation, n):', overlap(df))
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
