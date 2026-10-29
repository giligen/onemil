#!/usr/bin/env python3
"""52-week-high momentum sleeve (George & Hwang 2004) -- ONE shared rule module (cell 1,710).

Rule (PREREG_52WH_20261029.md, Amendment 1 / Stage 1):
  * signal on the LAST session of each month at the close:
        ratio = close_t / max(high over the 252 sessions BEFORE t)
  * universe at t: adjusted close >= price floor, 60-session median dollar volume >= $5M,
    full 252-session high history, symbol passes the name/asset-class screen;
  * buy the top-K by ratio (equal weight, K=20, frozen) at the OFFICIAL OPEN of the next
    session (= first session of the next month); hold H months (H=1 frozen; K=30,H=6 is
    the published sensitivity); exit at the open of the first session of the month H later.
  * cost: round-trip bps per cohort, charged once per cohort lifetime (RT/H per month).

Stage 1 is rejection-only: it runs on the survivorship-EXPOSED Alpaca panel, so a pass is not a pass.
TEST (2023+) is SEALED: `load_panel` refuses an end date after SEAL_END and the SQL-style filter
is applied at read time, so a 2023 bar never enters memory.

Pure functions (`signal_ratio`, `select_top_k`, `cohort_period_returns`) take arrays and are what a
live script would import; `run_stage1` is the research driver.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

log = logging.getLogger('fiftytwo_wh')

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / 'research' / 'momentum_weekly' / 'panel_2016_2026.parquet'
ASSET_MAP = ROOT / 'data' / 'research' / 'orb_asset_class_map_20260711.csv'
CACHE_DIR = Path(__file__).resolve().parent / 'cache'
OUT_DIR = Path(__file__).resolve().parent

SEAL_END = pd.Timestamp('2022-12-31')        # nothing after this date may be read
LOOKBACK = 252
DV_WINDOW = 60
MIN_DV = 5e6
PRICE_FLOOR = 5.0
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_BAD_RE = re.compile(r'\b(PREFERRED|PFD|WARRANT|WARRANTS|UNIT|UNITS|RIGHT|RIGHTS|CONTRA|NOTES?|DEPOSITARY)\b', re.I)


# ----------------------------------------------------------------------------- universe screen
def symbol_screen(symbols: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Boolean allowed-mask over `symbols` plus a per-reason exclusion table.

    Reasons (first match wins): test ticker, punctuation (preferred/warrant/unit/class suffix),
    asset-class map 'wrapper' (ETF/ETN/2x-3x), map 'unknown', not in map, name pattern
    (preferred/warrant/unit/right/note/contra).  The map is a PRESENT-DAY static classification
    used only to EXCLUDE non-stocks, never to select.
    """
    amap = pd.read_csv(ASSET_MAP).drop_duplicates('symbol').set_index('symbol')
    reason = []
    for s in symbols:
        if TEST_RE.match(s):
            reason.append('test_ticker')
        elif not s.isalpha():
            reason.append('punctuation')
        elif s not in amap.index:
            reason.append('not_in_asset_map')
        elif amap.at[s, 'asset_class'] == 'wrapper':
            reason.append('wrapper_etf')
        elif amap.at[s, 'asset_class'] != 'stock':
            reason.append('asset_unknown')
        elif NAME_BAD_RE.search(str(amap.at[s, 'name'])):
            reason.append('name_pattern')
        else:
            reason.append('')
    r = pd.Series(reason)
    table = r[r != ''].value_counts().rename('n').to_frame()
    table['share'] = table['n'] / len(symbols)
    return (r == '').to_numpy(), table


# ----------------------------------------------------------------------------- data
def load_panel(start: str, end: str) -> dict:
    """Read the panel between start and end (end <= SEAL_END) into [D, S] float32 arrays.

    The date predicate is pushed into the parquet scan, so sealed bars are never materialised.
    """
    end_ts = pd.Timestamp(end)
    if end_ts > SEAL_END:
        raise ValueError(f'end {end} is inside the sealed TEST window (> {SEAL_END.date()})')
    t0 = time.time()
    log.info('reading %s %s..%s', PANEL.name, start, end)
    tab = pq.read_table(PANEL, filters=[('bar_date', '>=', pd.Timestamp(start)), ('bar_date', '<=', end_ts)])
    log.info('  %d rows in %.0fs', tab.num_rows, time.time() - t0)
    dates = pd.DatetimeIndex(np.sort(pc.unique(tab['bar_date']).to_numpy()))
    sym = tab['symbol'].dictionary_encode().combine_chunks()
    symbols = [str(x) for x in sym.dictionary.to_pylist()]
    si = sym.indices.to_numpy(zero_copy_only=False)
    di = np.searchsorted(dates.to_numpy(), tab['bar_date'].to_numpy())
    out = {'dates': dates, 'symbols': symbols}
    for col in ('open', 'high', 'close', 'volume'):
        a = np.full((len(dates), len(symbols)), np.nan, dtype=np.float32)
        a[di, si] = tab[col].to_numpy().astype(np.float32)
        out[col] = a
    del tab
    log.info('panel %d sessions x %d symbols, %.0fs', len(dates), len(symbols), time.time() - t0)
    return out


# ----------------------------------------------------------------------------- the rule (shared)
def signal_ratio(close: np.ndarray, high: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
    """ratio_t = close[t] / max(high[t-252 : t]) and a validity mask (>= 240 of the 252 highs present)."""
    win = high[t - LOOKBACK:t]
    valid = np.isfinite(win).sum(axis=0) >= 240
    with np.errstate(all='ignore'):
        hi = np.nanmax(np.where(np.isfinite(win), win, -np.inf), axis=0)
        ratio = close[t] / hi
    return ratio, valid & np.isfinite(ratio) & (hi > 0)


def eligible_mask(close, volume, high, t, allowed, price_floor=PRICE_FLOOR) -> np.ndarray:
    """Universe membership at the signal close t (knowable at t: only data through t is used)."""
    dv = (close[t - DV_WINDOW + 1:t + 1].astype(np.float64) * volume[t - DV_WINDOW + 1:t + 1])
    ok_n = np.isfinite(dv).sum(axis=0) >= 50
    with np.errstate(all='ignore'):
        med = np.nanmedian(dv, axis=0)
    ratio, ok_ratio = signal_ratio(close, high, t)
    return allowed & ok_n & (med >= MIN_DV) & ok_ratio & (close[t] >= price_floor)


def select_top_k(ratio: np.ndarray, elig: np.ndarray, k: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """Indices of the top-k eligible names by ratio (descending), skipping `exclude` (already held)."""
    ok = elig.copy()
    if exclude is not None:
        ok &= ~exclude
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(-ratio[idx], kind='stable')]
    return order[:k]


def cohort_period_returns(cohorts: list[np.ndarray], R: np.ndarray, i: int, hold: int) -> float:
    """Equal-weight average over the cohorts active in period i of each cohort's equal-weight return."""
    rets = []
    for j in range(max(0, i - hold + 1), i + 1):
        names = cohorts[j]
        if len(names):
            rets.append(np.nanmean(R[i, names]))
    return float(np.mean(rets)) if rets else np.nan


# ----------------------------------------------------------------------------- driver
def month_structure(dates: pd.DatetimeIndex):
    """Signal day index (last session of each month) and entry index (first session of next month)."""
    s = pd.Series(np.arange(len(dates)), index=dates)
    ym = dates.to_period('M')
    last = s.groupby(ym).last()
    first = s.groupby(ym).first()
    sig = last.iloc[:-1].to_numpy()                  # all months except the (partial) final one
    ent = first.iloc[1:].to_numpy()                  # first session of the following month
    months = [str(p) for p in first.index[1:]]       # period label = entry month
    return sig, ent, months


def split_of(month: str) -> str:
    """TRAIN / VAL by ENTRY month; anything later is sealed and must never appear."""
    if month <= '2019-12':
        return 'TRAIN'
    if month <= '2022-12':
        return 'VAL'
    raise ValueError(f'month {month} is in the sealed TEST window')


def run_stage1(k=20, hold=1, k_sens=30, hold_sens=6, n_null=1000, seed=1710, price_floor=PRICE_FLOOR):
    """Compute per-period returns for the frozen rule, the published sensitivity and the null."""
    P = load_panel('2016-01-01', '2022-12-31')
    dates, symbols = P['dates'], P['symbols']
    allowed, excl = symbol_screen(symbols)
    log.info('symbol screen: %d / %d allowed (%.1f%% excluded)\n%s', allowed.sum(), len(symbols),
             100 * (1 - allowed.mean()), excl.to_string())
    spy_i = symbols.index('SPY')
    sig, ent, months = month_structure(dates)
    # periods: entry E_i -> next entry E_{i+1}; the last entry has no exit before the seal -> dropped
    nper = len(ent) - 1
    O, C, H, V = P['open'], P['close'], P['high'], P['volume']
    R = np.full((nper, len(symbols)), np.nan, dtype=np.float32)
    n_fallback = 0
    for i in range(nper):
        o0, o1 = O[ent[i]], O[ent[i + 1]]
        r = o1 / o0 - 1
        miss = np.isfinite(o0) & ~np.isfinite(o1)        # delisted / halted at exit: last close before exit
        if miss.any():
            last_c = np.where(np.isfinite(C[ent[i]:ent[i + 1]]), C[ent[i]:ent[i + 1]], np.nan)
            lc = pd.DataFrame(last_c).ffill().to_numpy()[-1]
            r = np.where(miss, lc / o0 - 1, r)
            n_fallback += int(miss.sum())
        R[i] = r
    if n_fallback:
        log.warning('exit open missing for %d name-periods; fell back to last close before the exit date', n_fallback)
    rng = np.random.default_rng(seed)
    rows, picks_log = [], []
    coh_main, coh_sens = [], []
    null_ret = np.full((nper, n_null), np.nan)
    for i in range(nper):
        t = sig[i]
        if t < LOOKBACK or months[i] < '2016-01':
            coh_main.append(np.array([], int)); coh_sens.append(np.array([], int)); continue
        ratio, _ = signal_ratio(C, H, t)
        el = eligible_mask(C, V, H, t, allowed, price_floor)
        el &= np.isfinite(O[ent[i]]) & np.isfinite(R[i])           # tradable at the entry open
        main = select_top_k(ratio, el, k)
        held = np.zeros(len(symbols), bool)
        for j in range(max(0, i - hold_sens + 1), i):
            held[coh_sens[j]] = True
        sens = select_top_k(ratio, el, k_sens, exclude=held)
        coh_main.append(main); coh_sens.append(sens)
        if len(main) < k:
            log.warning('%s: only %d eligible names for top-%d', months[i], len(main), k)
        # null: count-matched random-k from the same eligible set
        idx = np.flatnonzero(el)
        if len(idx) >= k:
            pick = np.argpartition(rng.random((n_null, len(idx))), k, axis=1)[:, :k]
            null_ret[i] = np.nanmean(R[i][idx[pick]], axis=1)
        gm = float(np.nanmean(R[i, main])) if len(main) else np.nan
        rows.append({'month': months[i], 'split': split_of(months[i]), 'i': i, 'n_universe': int(el.sum()),
                     'n_picks': len(main), 'strat_gross': gm,
                     'univ_ew': float(np.nanmean(R[i, idx])), 'spy': float(R[i, spy_i]),
                     'median_ratio_picks': float(np.median(ratio[main])) if len(main) else np.nan})
        for n in main:
            picks_log.append((months[i], symbols[n], float(R[i, n])))
        log.info('%s n_univ=%d picks=%d gross=%+.2f%% univ=%+.2f%%', months[i], int(el.sum()), len(main),
                 100 * gm, 100 * rows[-1]['univ_ew'])
    df = pd.DataFrame(rows)
    df['sens_gross'] = [cohort_period_returns(coh_sens, R, int(i), hold_sens) for i in df['i']]
    # sensitivity warm-up: published variant is only fully invested once hold_sens cohorts exist
    df['sens_active_cohorts'] = [min(int(i) - int(df['i'].iloc[0]) + 1, hold_sens) for i in df['i']]
    picks = pd.DataFrame(picks_log, columns=['month', 'symbol', 'ret'])
    return df, picks, null_ret[df['i'].to_numpy()], excl, allowed


def metrics(excess: np.ndarray, net: np.ndarray, years: np.ndarray, n_boot=5000, seed=7) -> dict:
    """Mean excess, year-block bootstrap CI, Sharpe, max DD, ex-top-5% excess, per-year cell t."""
    rng = np.random.default_rng(seed)
    yrs = np.unique(years)
    blocks = [excess[years == y] for y in yrs]
    boots = np.array([np.concatenate([blocks[j] for j in rng.integers(0, len(yrs), len(yrs))]).mean()
                      for _ in range(n_boot)])
    order = np.argsort(-net)
    keep = np.sort(order[int(np.ceil(0.05 * len(net))):])
    eq = np.cumprod(1 + net)
    dd = float(np.max(1 - eq / np.maximum.accumulate(np.r_[1.0, eq])[1:]))
    cell_t = []
    for y in yrs:
        e = excess[years == y]
        cell_t.append(e.mean() / (e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 2 and e.std(ddof=1) > 0 else np.nan)
    se = excess.std(ddof=1) / np.sqrt(len(excess))
    return {'n': len(net), 'mean_net': net.mean(), 'mean_excess': excess.mean(),
            'ci_lo': np.percentile(boots, 2.5), 'ci_hi': np.percentile(boots, 97.5),
            'iid_t': excess.mean() / se, 'sharpe': net.mean() / net.std(ddof=1) * np.sqrt(12),
            'max_dd': dd, 'excess_ex_top5': excess[keep].mean(), 'green': float((net > 0).mean()),
            'max_abs_year_t': float(np.nanmax(np.abs(cell_t))), 'year_t': dict(zip(yrs.tolist(), np.round(cell_t, 2).tolist()))}


def evaluate(tag: str = '') -> None:
    """Print the PREREG Evaluation table per split from the cached monthly CSV, null and picks."""
    df = pd.read_csv(OUT_DIR / f'monthly_1710{tag}.csv')
    null = np.load(CACHE_DIR / f'null_ret_stage1{tag}.npy')
    picks = pd.read_parquet(CACHE_DIR / f'picks_stage1{tag}.parquet')
    df['year'] = df['month'].str[:4].astype(int)
    for sp in ('TRAIN', 'VAL'):
        m = (df['split'] == sp).to_numpy()
        d, nl = df[m], null[m]
        yrs = d['year'].to_numpy()
        print(f'==== {sp} n={m.sum()} months {d.month.iloc[0]}..{d.month.iloc[-1]}  universe median n={int(d.n_universe.median())}')
        print(f'  SPY mean {d.spy.mean()*100:+.2f}%/mo  universe-EW mean {d.univ_ew.mean()*100:+.2f}%/mo  strat gross {d.strat_gross.mean()*100:+.2f}%/mo')
        for rt in (15, 30):
            net = d[f'strat_net{rt}'].to_numpy()
            for bench in ('univ', 'spy'):
                ex = (net - d['univ_ew' if bench == 'univ' else 'spy']).to_numpy()
                r = metrics(ex, net, yrs)
                print(f'  RT{rt} vs {bench}: mean_ex {r["mean_excess"]*100:+.2f}%/mo CI[{r["ci_lo"]*100:+.2f},{r["ci_hi"]*100:+.2f}] iid_t {r["iid_t"]:.2f} '
                      f'ex-top5% {r["excess_ex_top5"]*100:+.2f}% sharpe {r["sharpe"]:.2f} maxDD {r["max_dd"]*100:.1f}% green {r["green"]*100:.0f}% maxYearT {r["max_abs_year_t"]:.2f}')
            nullgreen = ((nl - rt / 1e4) > 0).mean(axis=0)
            g = (net > 0).mean()
            print(f'  RT{rt} green {g*100:.0f}% vs null mean {nullgreen.mean()*100:.0f}% (p95 {np.percentile(nullgreen,95)*100:.0f}%, pctile {(nullgreen<g).mean()*100:.0f}%); '
                  f'null mean-net {np.nanmean(nl - rt/1e4)*100:+.2f}%/mo; strat mean-net {net.mean()*100:+.2f}%/mo')
        # published sensitivity (warm-up months with < 6 cohorts dropped)
        w = d[d.sens_active_cohorts >= 6]
        if len(w) > 3:
            sn = w['sens_net15'].to_numpy()
            r = metrics(sn - w['univ_ew'].to_numpy(), sn, w['year'].to_numpy())
            print(f'  30x6m (RT15/6 per month, n={len(w)} fully-invested months): mean_ex_univ {r["mean_excess"]*100:+.2f}% CI[{r["ci_lo"]*100:+.2f},{r["ci_hi"]*100:+.2f}] sharpe {r["sharpe"]:.2f} maxDD {r["max_dd"]*100:.1f}% green {r["green"]*100:.0f}%')
        pk = picks[picks.month.isin(d.month)]
        bad = pk[(pk.ret > 5) | (pk.ret < -0.8)]
        print(f'  price-scale: {len(pk)} name-months, {len(bad)} beyond +500%/-80%: {bad.values.tolist()[:8]}; max ret {pk.ret.max()*100:+.0f}% min {pk.ret.min()*100:+.0f}%')
    e = df.copy()
    print('ratio of picks median:', e.median_ratio_picks.describe()[['min','50%','max']].round(3).to_dict())


def main():
    """CLI: run Stage 1, write monthly_1710.csv and print the per-split evaluation."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--price-floor', type=float, default=PRICE_FLOOR)
    ap.add_argument('--evaluate', action='store_true')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stdout)
    CACHE_DIR.mkdir(exist_ok=True)
    tag = '' if a.price_floor == PRICE_FLOOR else f'_floor{a.price_floor:g}'
    if a.evaluate:
        return evaluate(tag)
    df, picks, null_ret, excl, allowed = run_stage1(price_floor=a.price_floor)
    for rt in (15, 30):
        df[f'strat_net{rt}'] = df['strat_gross'] - rt / 1e4
        df[f'sens_net{rt}'] = df['sens_gross'] - (rt / 1e4) / 6
        df[f'ex_univ{rt}'] = df[f'strat_net{rt}'] - df['univ_ew']
        df[f'ex_spy{rt}'] = df[f'strat_net{rt}'] - df['spy']
    tag = '' if a.price_floor == PRICE_FLOOR else f'_floor{a.price_floor:g}'
    df.to_csv(OUT_DIR / f'monthly_1710{tag}.csv', index=False)
    picks.to_parquet(CACHE_DIR / f'picks_stage1{tag}.parquet')
    np.save(CACHE_DIR / f'null_ret_stage1{tag}.npy', null_ret)
    excl.to_csv(CACHE_DIR / 'exclusions_stage1.csv')
    log.info('wrote monthly_1710.csv (%d periods)', len(df))


if __name__ == '__main__':
    main()
