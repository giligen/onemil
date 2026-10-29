"""Sleeve #2 (cell 1,711): ETF time-series momentum on a fixed 12-ETF basket (PREREG_TSM_20261029.md).

One rule module shared by the research driver (`main`) and any future live script:
  * signal  : on the last session of each month at the close, mom12 = close / close[252 sessions ago] - 1;
              long when mom12 > hurdle (0 in the frozen variant), else cash at 0 %.
  * weights : 1/12 per long ETF (cash fills the rest) -- frozen; sensitivity 1/N_longs (fully invested).
  * fills   : next session's (first of the month) official open; each position is held to the next month's
              first-session open.  Cost = `cost_rt_bps` per round trip on traded notional (turnover is measured
              against the weights after intra-month drift).
TRAIN entry months 2017-02..2019-12 (the 252-session lookback eats 2016), VAL 2020-01..2022-11.  The TEST window
(2023-01..) is SEALED: `load_panel` filters at the parquet scan, so no sealed bar is ever materialised.  The last
VAL entry month (2022-12) needs the 2023-01 first-session open to close its hold, so it is DROPPED (reported).

Run:  nice -n 19 python3 research/sleeves/tsm_etf.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / 'research' / 'momentum_weekly' / 'panel_2016_2026.parquet'
OUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = OUT_DIR / 'cache'
SEAL_END = pd.Timestamp('2022-12-31')       # nothing after this date may be read
BASKET = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'GLD', 'DBC', 'VNQ']
LOOKBACK = 252
COST_RT_BPS = 5.0
N_NULL = 1000
log = logging.getLogger('tsm_etf')


# ----------------------------------------------------------------------------- data
def load_panel(start: str = '2016-01-01', end: str = '2022-12-31') -> dict:
    """Read the basket's daily bars between start and end (end <= SEAL_END) into [D, S] float64 arrays.

    ETFs absent from the panel are dropped with a WARNING (never substituted).
    """
    end_ts = pd.Timestamp(end)
    if end_ts > SEAL_END:
        raise ValueError(f'end {end} is inside the sealed TEST window (> {SEAL_END.date()})')
    tab = pq.read_table(PANEL, filters=[('symbol', 'in', BASKET), ('bar_date', '>=', pd.Timestamp(start)),
                                        ('bar_date', '<=', end_ts)]).to_pandas()
    present = sorted(set(tab['symbol']))
    missing = [s for s in BASKET if s not in present]
    if missing:
        log.warning('ETFs ABSENT from the local panel, dropped (not substituted): %s', missing)
    symbols = [s for s in BASKET if s in present]
    out = {'symbols': symbols, 'missing': missing}
    for col in ('open', 'close'):
        p = tab.pivot(index='bar_date', columns='symbol', values=col)[symbols].sort_index()
        out[col] = p.to_numpy(dtype=np.float64)
        out['dates'] = p.index
    n_nan = int(np.isnan(out['close']).sum() + np.isnan(out['open']).sum())
    if n_nan:
        log.warning('%d NaN bars in the basket panel -- months touching them are unreliable', n_nan)
    log.info('basket %d/%d ETFs, %d sessions %s..%s, NaN bars %d', len(symbols), len(BASKET), len(out['dates']),
             out['dates'][0].date(), out['dates'][-1].date(), n_nan)
    return out


# ----------------------------------------------------------------------------- the rule (shared)
def month_structure(dates: pd.DatetimeIndex):
    """Signal index (last session of each month), entry index (first session of next month), entry-month labels."""
    s = pd.Series(np.arange(len(dates)), index=dates)
    ym = dates.to_period('M')
    last, first = s.groupby(ym).last(), s.groupby(ym).first()
    return last.iloc[:-1].to_numpy(), first.iloc[1:].to_numpy(), [str(p) for p in first.index[1:]]


def split_of(month: str) -> str:
    """TRAIN / VAL by entry month; anything later is sealed and must never appear."""
    if month <= '2019-12':
        return 'TRAIN'
    if month <= '2022-12':
        return 'VAL'
    raise ValueError(f'month {month} is in the sealed TEST window')


def signal_longs(close: np.ndarray, t: int, hurdle: float = 0.0) -> np.ndarray:
    """Boolean [S] long mask at signal session t: mom12 = close[t]/close[t-252]-1 > hurdle (NaN -> cash)."""
    mom = close[t] / close[t - LOOKBACK] - 1.0
    return np.nan_to_num(mom, nan=-1.0) > hurdle


def target_weights(longs: np.ndarray, mode: str = 'frozen') -> np.ndarray:
    """Frozen: 1/S per long (cash fills the rest). 'invested': 1/N_longs per long (all-cash if none)."""
    n = len(longs)
    if mode == 'frozen':
        return longs / n
    if mode == 'invested':
        k = longs.sum()
        return longs / k if k else np.zeros(n)
    raise ValueError(mode)


def run_strategy(P: dict, hurdle=0.0, mode='frozen', cost_rt_bps=COST_RT_BPS, random_n=None, rng=None) -> pd.DataFrame:
    """Monthly TSM ledger. random_n[i] (optional) replaces the signal with a random subset of that size (null)."""
    dates, op, cl = P['dates'], P['open'], P['close']
    sig, ent, months = month_structure(dates)
    S = len(P['symbols'])
    rows, w_prev = [], np.zeros(S)
    for j in range(len(ent) - 1):                      # last entry has no exit open inside the data -> dropped
        t, e, e2, m = sig[j], ent[j], ent[j + 1], months[j]
        if t < LOOKBACK or m > '2022-11':
            continue
        if random_n is None:
            longs = signal_longs(cl, t, hurdle)
        else:
            longs = np.zeros(S, bool)
            longs[rng.choice(S, random_n[len(rows)], replace=False)] = True
        w = target_weights(longs, mode)
        r = op[e2] / op[e] - 1.0                       # open-to-open hold return per ETF
        gross = float(w @ r)
        turnover = float(np.abs(w - w_prev).sum())     # one-way traded notional vs drifted holdings
        cost = turnover * cost_rt_bps / 2e4
        drift = w * (1 + r)
        w_prev = drift / (1 + gross) if (1 + gross) > 0 else w
        rows.append({'month': m, 'split': split_of(m), 'n_longs': int(longs.sum()), 'longs': ' '.join(
            s for s, l in zip(P['symbols'], longs) if l), 'gross': gross, 'turnover': turnover, 'cost': cost,
            'net': gross - cost, 'ew_gross': float(r.mean())})
    return pd.DataFrame(rows)


def benchmark_net(P: dict, cost_rt_bps=COST_RT_BPS) -> np.ndarray:
    """1/S-weighted basket, rebalanced monthly to equal weight, same cost model on drift turnover."""
    df = run_strategy(P, hurdle=-9.0, mode='frozen', cost_rt_bps=cost_rt_bps)   # hurdle -9 -> always long all
    return df['net'].to_numpy()


# ----------------------------------------------------------------------------- statistics
def metrics(net: np.ndarray, bench: np.ndarray, years: np.ndarray, n_boot=5000, seed=7) -> dict:
    """Mean/excess/CI(year-block bootstrap)/Sharpe/maxDD/ex-top-5% months/green share."""
    rng = np.random.default_rng(seed)
    exc = net - bench
    yrs = np.unique(years)
    blocks = [exc[years == y] for y in yrs]
    boots = np.array([np.concatenate([blocks[j] for j in rng.integers(0, len(yrs), len(yrs))]).mean()
                      for _ in range(n_boot)])
    keep = np.sort(np.argsort(-net)[int(np.ceil(0.05 * len(net))):])
    se = exc.std(ddof=1) / np.sqrt(len(exc))
    return {'n': len(net), 'mean_net': net.mean(), 'bench_mean': bench.mean(), 'mean_excess': exc.mean(),
            'ci_lo': float(np.percentile(boots, 2.5)), 'ci_hi': float(np.percentile(boots, 97.5)),
            'iid_t': float(exc.mean() / se) if se > 0 else np.nan,
            'sharpe': float(net.mean() / net.std(ddof=1) * np.sqrt(12)) if net.std() > 0 else np.nan,
            'bench_sharpe': float(bench.mean() / bench.std(ddof=1) * np.sqrt(12)),
            'max_dd': max_dd(net), 'bench_dd': max_dd(bench), 'mean_ex_top5': float(net[keep].mean()),
            'green': float((net > 0).mean()), 'bench_green': float((bench > 0).mean()),
            'cash_months': 0}


def max_dd(r: np.ndarray) -> float:
    """Max peak-to-trough drawdown of the compounded monthly series (start equity 1)."""
    eq = np.r_[1.0, np.cumprod(1 + r)]
    return float(np.max(1 - eq / np.maximum.accumulate(eq)))


def null_green(P: dict, n_longs: np.ndarray, sel: np.ndarray, n_draw=N_NULL, seed=1711) -> dict:
    """Count-matched null: each month hold a random subset of the same size (1/12 each), n_draw draws."""
    rng = np.random.default_rng(seed)
    g, dd, mean = [], [], []
    for _ in range(n_draw):
        d = run_strategy(P, random_n=n_longs, rng=rng)
        net = d['net'].to_numpy()[sel]
        g.append((net > 0).mean()); dd.append(max_dd(net)); mean.append(net.mean())
    return {'green_mean': float(np.mean(g)), 'green_p95': float(np.percentile(g, 95)), 'g': np.array(g),
            'dd_med': float(np.median(dd)), 'mean_mean': float(np.mean(mean))}


def passes(m: dict) -> dict:
    """Frozen VAL pass bar (PREREG): A = excess>0 with CI>0 OR (DD<=half basket DD AND mean>=basket-0.1%/mo);
    B = green>=50%; C = max DD <= 15%."""
    a1 = m['mean_excess'] > 0 and m['ci_lo'] > 0
    a2 = m['max_dd'] <= 0.5 * m['bench_dd'] and m['mean_net'] >= m['bench_mean'] - 0.001
    return {'A_excess_ci': a1, 'A_dd_clause': a2, 'A': a1 or a2, 'B_green>=50%': m['green'] >= 0.5,
            'C_dd<=15%': m['max_dd'] <= 0.15}


# ----------------------------------------------------------------------------- overlap
def overlap(df: pd.DataFrame) -> dict:
    """Monthly-return correlation with TOM (research/index_overnight/tom_1651.csv, mean of SPY/QQQ/IWM) and MOM
    (research/momentum_weekly/1700c_monthly.csv net_ret); sealed months are cut at load, before any computation."""
    out = {}
    tom = pd.read_csv(ROOT / 'research/index_overnight/tom_1651.csv')
    tom = tom[tom['month'] <= '2022-12'].groupby('month')['net_bps'].mean() / 1e4
    mom = pd.read_csv(ROOT / 'research/momentum_weekly/1700c_monthly.csv', parse_dates=['entry_date'])
    mom = mom[mom['entry_date'] <= SEAL_END].copy()
    mom['month'] = mom['entry_date'].dt.strftime('%Y-%m')
    mom = mom.groupby('month')['net_ret'].sum(min_count=1)
    for name, s in (('TOM', tom), ('MOM', mom)):
        j = df.set_index('month')[['net', 'split']].join(s.rename('x'), how='inner').dropna()
        out[name] = {sp: (float(np.corrcoef(g['net'], g['x'])[0, 1]) if len(g) > 3 else np.nan, len(g))
                     for sp, g in list(j.groupby('split')) + [('ALL', j)]}
    return out


def price_scale_check(P: dict) -> pd.DataFrame:
    """Per ETF: max |daily close return| and sessions with |ret| > 25 % (split-like jumps)."""
    c = pd.DataFrame(P['close'], index=P['dates'], columns=P['symbols'])
    r = c.pct_change().abs()
    return pd.DataFrame({'max_abs_ret': r.max(), 'n_jumps_25pct': (r > 0.25).sum(), 'first_close': c.iloc[0]})


# ----------------------------------------------------------------------------- driver
def main() -> None:
    """Run the frozen rule + sensitivities on TRAIN/VAL, write monthly_1711.csv, print the headline tables."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    CACHE_DIR.mkdir(exist_ok=True)
    P = load_panel()
    print(price_scale_check(P).round(4).to_string())
    base = run_strategy(P)
    bench = benchmark_net(P)
    base['bench_net'], base['excess'] = bench, base['net'] - bench
    variants = {'hurdle4': run_strategy(P, hurdle=0.04), 'invested': run_strategy(P, mode='invested'),
                'cost15': run_strategy(P, cost_rt_bps=15.0)}
    for k, d in variants.items():
        base[f'net_{k}'] = d['net'].to_numpy()
    base['year'] = base['month'].str[:4].astype(int)
    base.to_csv(OUT_DIR / 'monthly_1711.csv', index=False)
    log.info('wrote monthly_1711.csv (%d months)', len(base))
    res = {}
    for sp in ('TRAIN', 'VAL'):
        sel = (base['split'] == sp).to_numpy()
        yrs = base['year'].to_numpy()[sel]
        b = bench[sel]
        for name, net in (('FROZEN', base['net'].to_numpy()), *((k, base[f'net_{k}'].to_numpy()) for k in variants)):
            m = metrics(net[sel], b, yrs)
            m['cash_months'] = int((base['n_longs'].to_numpy()[sel] == 0).sum()) if name == 'FROZEN' else np.nan
            res[(sp, name)] = m
            print(f'\n== {sp} {name}: ' + ' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                                                     for k, v in m.items()))
        if sp == 'VAL' or True:
            nl = base['n_longs'].to_numpy()
            nl_sp = nl[sel]
            # null needs the full-length n_longs sequence aligned to run_strategy's month order
            nz = null_green(P, nl, sel)
            m = res[(sp, 'FROZEN')]
            print(f'   null({sp}) green mean {nz["green_mean"]:.3f} p95 {nz["green_p95"]:.3f} | actual {m["green"]:.3f} '
                  f'| p(null>=actual) {(nz["g"] >= m["green"]).mean():.3f} | null DD median {nz["dd_med"]:.3f} '
                  f'mean {nz["mean_mean"]:.4f}')
            print('   pass bar:', passes(m))
            print('   n_longs mean', nl_sp.mean().round(2), 'cash months', int((nl_sp == 0).sum()))
    print('\nOverlap:', overlap(base))


if __name__ == '__main__':
    main()
