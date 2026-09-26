#!/usr/bin/env python3
"""BUILDER B - cell I4 (initial Schedule 13D filings): signal construction, trades,
book simulation and PREREG pass-bar scoring. Shares the multiday programme's Panel,
cost model and book_sim EXACTLY (imported, not reimplemented) - parity by construction.

Signal: first panel session strictly after a SC 13D FILING_DATE, on a PIT-eligible name
(`elig`), taint excluded. Hold 20 sessions (primary), 60 (report-only). One position per
(symbol, signal session): a later overlapping signal on the same name within the hold
extends nothing (PREREG "Signal definitions").

TEST (2024-01-01 onward) is built (signals + trades written with split='TEST') but NEVER
scored: excluded from every table in RESULT_13D.md.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, '..', 'multiday')
sys.path.insert(0, MD)
from run_final import (Panel, cost_bps, concurrent_order_usd, book_sim, daily_series,  # noqa: E402
                        monthly, SPLITS, BOOK_USD, N_SLOTS, MIN_ADV, MIN_RAW_CLOSE)

DATA = os.path.join(HERE, 'data')
SC13D = os.path.join(DATA, 'sc13d.parquet')
NULL_SEED = 1477
N_NULL = 1000
SURVIVE_WINDOW = ('2018-01-01', '2023-12-31')


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


# ------------------------------------------------------------------ signal construction


def signal_session(sess_np: np.ndarray, filed: np.ndarray) -> np.ndarray:
    """First index in `sess_np` strictly after each date in `filed`.

    searchsorted(..., side='right') on a sorted, DISTINCT sessions array returns the
    insertion point after any exact match, i.e. exactly "first session strictly after" -
    including a filing dated on a non-trading day, which lands on the same index as one
    dated on the trading day right before it. Filings after the panel's last session get
    index == len(sess_np) (caller must drop those).
    """
    return np.searchsorted(sess_np, filed, side='right')


def dedupe_overlapping(df: pd.DataFrame, hold: int) -> pd.DataFrame:
    """PREREG: one position per (symbol, signal session); a later signal on the same name
    that falls inside an already-kept signal's hold window extends nothing. Keeps the
    EARLIEST signal per overlapping run, per symbol."""
    if not len(df):
        return df
    keep_idx = []
    df = df.sort_values(['symbol', 'S'])
    last_S: dict = {}
    for idx, sym, S in zip(df.index, df['symbol'], df['S']):
        prev = last_S.get(sym)
        if prev is None or S >= prev + hold:
            keep_idx.append(idx)
            last_S[sym] = S
        # else: inside the prior signal's hold -> drop, slot already owned
    return df.loc[keep_idx].sort_values('S').reset_index(drop=True)


def build_signals(p: Panel, sc13d: pd.DataFrame, hold: int) -> pd.DataFrame:
    """SC 13D rows -> one row per eligible, taint-free, in-panel signal."""
    df = sc13d.dropna(subset=['symbol']).copy()
    df = df[df['symbol'].isin(p.sidx)]
    df['si'] = df['symbol'].map(p.sidx).astype(int)
    filed = df['filed_date'].to_numpy(dtype='datetime64[ns]')
    df['S'] = signal_session(p.sess_np, filed)
    n0 = len(df)
    df = df[df['S'] < p.n_d].copy()
    log(f'{n0 - len(df)} filings drop (signal session beyond the panel end)')
    si = df['si'].to_numpy(); S = df['S'].to_numpy()
    elig_ok = p.elig[si, S]
    taint_ok = ~p.taint[si, S]
    df['elig_ok'] = elig_ok
    df['taint_ok'] = taint_ok
    n_elig_drop = int((~elig_ok).sum())
    n_taint_drop = int((elig_ok & ~taint_ok).sum())
    df = df[df['elig_ok'] & df['taint_ok']].copy()
    log(f'signals: {n0} filings -> {len(df)} eligible+untainted '
        f'(-{n_elig_drop} not elig, -{n_taint_drop} tainted)')
    df = dedupe_overlapping(df, hold)
    log(f'after one-slot-per-symbol-per-hold de-dup: {len(df)} signals')
    df['entry_dt'] = p.sessions[df['S'].to_numpy()]
    return df.reset_index(drop=True)


def build_trades(p: Panel, sig: pd.DataFrame, hold: int) -> pd.DataFrame:
    """Entry = close(S) (the signal session's own close - already the first PIT-knowable
    close after FILING_DATE). Exit = close(S + hold). Long only (leg = +1)."""
    t = sig.copy()
    t['a'] = t['S'].to_numpy()
    t['b'] = t['a'] + hold
    t = t[t['b'] < p.n_d].copy()
    si, a, b = t['si'].to_numpy(), t['a'].to_numpy(), t['b'].to_numpy()
    ca, cb = p.close[si, a], p.close[si, b]
    good = np.isfinite(ca) & np.isfinite(cb) & (ca > 0)
    t = t[good].copy()
    si, a, b = si[good], a[good], b[good]
    with np.errstate(invalid='ignore', divide='ignore'):
        t['gross'] = p.close[si, b] / p.close[si, a] - 1.0
    order_usd, npos = concurrent_order_usd(p, a, b)
    t['n_concurrent'] = npos
    hold_days = (b - a).astype(float)
    t['leg'] = 1
    t['hold'] = hold_days
    t['ends_mid_hold'] = p.last_fin[si] < b
    t['cost_bps'] = cost_bps(p.adv[si, a], order_usd, short=False, hold_days=hold_days)
    t['net'] = t['gross'] - t['cost_bps'] / 1e4
    t['contrib'] = t['gross'].to_numpy()
    t['exit_dt'] = p.sessions[b]
    t['rank'] = 0
    return t.reset_index(drop=True)


def bench_trades(p: Panel, t: pd.DataFrame) -> pd.DataFrame:
    """SPY over the identical entry/exit window, gross (never charged - standing rule)."""
    b = t.copy()
    b['si'] = p.spy
    b['leg'] = 1
    b['cost_bps'] = 0.0
    b['ends_mid_hold'] = False
    return b


def tag_split(dates: pd.Series) -> pd.Series:
    out = pd.Series('PRE', index=dates.index)
    for name, (a, bnd) in SPLITS.items():
        out[(dates >= a) & (dates <= bnd)] = name
    return out


# ------------------------------------------------------------------ stats


def day_clustered_t(net: np.ndarray, day: np.ndarray) -> float:
    """Cluster-robust t for the mean of `net`, clusters = calendar day of entry.
    Standard CRVE: Var(mean) = (1/N^2) sum_g u_g^2, u_g = sum_{i in g}(net_i - mean)."""
    n = len(net)
    if n < 2:
        return float('nan')
    mu = net.mean()
    resid = net - mu
    order = pd.Series(resid).groupby(day).sum()
    var = (order.to_numpy() ** 2).sum() / (n ** 2)
    if var <= 0:
        return float('nan')
    return float(mu / np.sqrt(var))


def ex_top_pct(net: np.ndarray, pct: float) -> float:
    if len(net) < 3:
        return float('nan')
    thr = np.quantile(net, 1 - pct)
    kept = net[net < thr]
    return float(kept.mean()) if len(kept) else float('nan')


def months_positive_share(monthly_ret: pd.Series) -> float:
    m = monthly_ret[monthly_ret != 0]
    if not len(m):
        return float('nan')
    return float(100.0 * (m > 0).mean())


def signals_per_week(entry_dt: pd.Series, a: str, b: str) -> float:
    m = entry_dt[(entry_dt >= a) & (entry_dt <= b)]
    weeks = max(1.0, (pd.Timestamp(b) - pd.Timestamp(a)).days / 7.0)
    return len(m) / weeks


def count_matched_null(p: Panel, t: pd.DataFrame, split: str, stat_fn, seed: int = NULL_SEED,
                        n_draws: int = N_NULL) -> tuple:
    """1,000 draws of random ELIGIBLE symbol-sessions on the SAME dates (same n) as the
    real signals in `split`; each draw scored with `stat_fn(net_array)`. Returns
    (observed_stat, percentile_of_observed_in_null)."""
    a0, b0 = SPLITS[split]
    sub = t[(t['entry_dt'] >= a0) & (t['entry_dt'] <= b0)]
    if len(sub) < 5:
        return float('nan'), float('nan')
    a_arr = sub['a'].to_numpy()
    hold = int(sub['hold'].iloc[0])
    obs = stat_fn(sub['net'].to_numpy())
    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_draws)
    uniq_a = np.unique(a_arr)
    elig_syms = {aa: np.nonzero(p.elig[:, aa])[0] for aa in uniq_a}
    for d in range(n_draws):
        si_draw = np.array([rng.choice(elig_syms[aa]) for aa in a_arr])
        b_draw = a_arr + hold
        ok = b_draw < p.n_d
        si_draw, a_draw, b_draw = si_draw[ok], a_arr[ok], b_draw[ok]
        ca, cb = p.close[si_draw, a_draw], p.close[si_draw, b_draw]
        fin = np.isfinite(ca) & np.isfinite(cb) & (ca > 0)
        if fin.sum() < 5:
            null_stats[d] = np.nan
            continue
        gross = cb[fin] / ca[fin] - 1.0
        order_usd, _ = concurrent_order_usd(p, a_draw[fin], b_draw[fin])
        c = cost_bps(p.adv[si_draw[fin], a_draw[fin]], order_usd, short=False,
                      hold_days=(b_draw[fin] - a_draw[fin]).astype(float))
        net_draw = gross - c / 1e4
        null_stats[d] = stat_fn(net_draw)
    valid = null_stats[np.isfinite(null_stats)]
    pctile = 100.0 * (valid <= obs).mean() if len(valid) else float('nan')
    return float(obs), float(pctile)


def book_stats(p: Panel, t: pd.DataFrame, split: str) -> dict:
    a0, b0 = SPLITS[split]
    bk = book_sim(p, t, 'rank', ascending=True)
    ds = daily_series(p, bk, charge=True)
    ds = ds[(ds.index >= a0) & (ds.index <= b0)]
    eq = (1.0 + ds).cumprod()
    mret = monthly(ds)
    mret = mret[(mret.index >= a0) & (mret.index <= b0)]
    return dict(book_monthly_pct=float(mret.mean() * 100) if len(mret) else float('nan'),
                book_max_dd_pct=float((eq / eq.cummax() - 1).min() * 100) if len(eq) else float('nan'),
                book_n=len(bk))


def score_split(p: Panel, t: pd.DataFrame, split: str, arm: str) -> dict:
    a0, b0 = SPLITS[split]
    sub = t[(t['entry_dt'] >= a0) & (t['entry_dt'] <= b0)]
    n = len(sub)
    row = dict(cell='I4_13D', split=split, arm=arm, n_trades=n)
    if n < 5:
        row.update(mean_net_pct=float('nan'), t_day_clustered=float('nan'),
                    ex_top5_mean_pct=float('nan'), months_positive_share=float('nan'),
                    signals_per_week=float('nan'), null_pctile=float('nan'),
                    book_monthly_pct=float('nan'), book_max_dd_pct=float('nan'))
        return row
    net = sub['net'].to_numpy()
    day = sub['entry_dt'].dt.date.to_numpy()
    day_key = pd.factorize(day)[0]
    row['mean_net_pct'] = float(net.mean() * 100)
    row['t_day_clustered'] = day_clustered_t(net, day_key)
    row['ex_top5_mean_pct'] = ex_top_pct(net, 0.05) * 100
    ds_full = daily_series(p, sub, charge=True)
    row['months_positive_share'] = months_positive_share(monthly(ds_full))
    row['signals_per_week'] = signals_per_week(sub['entry_dt'], a0, b0)
    obs, pct = count_matched_null(p, t, split, lambda x: x.mean())
    row['null_pctile'] = pct
    bstats = book_stats(p, t, split)
    row.update(bstats)
    return row


def passes_bar(row: dict) -> bool:
    if row['split'] == 'TRAIN':
        return (row['mean_net_pct'] >= 0.8 and row['t_day_clustered'] >= 2.5
                and row['months_positive_share'] >= 55)
    if row['split'] == 'VAL':
        return (row['mean_net_pct'] >= 0.5 and row['t_day_clustered'] >= 2.0
                and row['ex_top5_mean_pct'] > 0 and row['signals_per_week'] >= 3
                and row['book_monthly_pct'] >= 1.0 and row['book_max_dd_pct'] >= -15
                and row['null_pctile'] >= 99)
    return False


# ------------------------------------------------------------------ main


def main():
    p = Panel()
    sc13d = pd.read_parquet(SC13D)
    log(f'sc13d.parquet: {len(sc13d)} initial-13D rows, '
        f'{(sc13d["symbol"].isna()).sum()} unmapped')

    rows = []
    all_sig, all_trd = [], []
    for hold, label in ((20, 'primary'), (60, 'report_only')):
        sig = build_signals(p, sc13d, hold)
        trd = build_trades(p, sig, hold)
        trd['split'] = tag_split(trd['entry_dt'])
        trd['hold_label'] = label
        sig2 = sig.copy(); sig2['split'] = tag_split(sig2['entry_dt']); sig2['hold_label'] = label
        all_sig.append(sig2); all_trd.append(trd)

        if label != 'primary':
            continue  # pass-bar scoring is on the primary (20-session) hold only
        for split in ('TRAIN', 'VAL'):
            rows.append(score_split(p, trd, split, 'all'))

        # survivorship arm: 2018-01..2023-12, with vs without names that stop being priced
        # before the panel's last session (proxy for "delisted"; see RESULT caveats)
        surv = trd[(trd['entry_dt'] >= SURVIVE_WINDOW[0]) & (trd['entry_dt'] <= SURVIVE_WINDOW[1])]
        is_delisted = p.last_fin[surv['si'].to_numpy()] < (p.n_d - 1)
        for arm_name, keep in (('surv_with_delisted', np.ones(len(surv), dtype=bool)),
                               ('surv_ex_delisted', ~is_delisted)):
            sub = surv[keep]
            net = sub['net'].to_numpy()
            rows.append(dict(cell='I4_13D', split='2018-2023', arm=arm_name,
                              n_trades=len(sub),
                              mean_net_pct=float(net.mean() * 100) if len(sub) else float('nan'),
                              t_day_clustered=(day_clustered_t(net, pd.factorize(sub['entry_dt'].dt.date)[0])
                                               if len(sub) > 5 else float('nan')),
                              note=f'delisted-name trades excluded: {int(is_delisted.sum())}/{len(surv)}'))

    out_sig = pd.concat(all_sig, ignore_index=True)
    out_trd = pd.concat(all_trd, ignore_index=True)
    out_sig.drop(columns=['si'], errors='ignore').to_csv(os.path.join(HERE, 'signals_13d.csv'), index=False)
    out_trd.drop(columns=['si'], errors='ignore').to_csv(os.path.join(HERE, 'trades_13d.csv'), index=False)
    log(f'wrote signals_13d.csv ({len(out_sig)}) and trades_13d.csv ({len(out_trd)}), '
        f'TEST rows included but flagged split=TEST (not scored)')

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, 'scoring_13d.csv'), index=False)
    log('\n' + res.to_string(index=False))
    log('DONE')


if __name__ == '__main__':
    main()
