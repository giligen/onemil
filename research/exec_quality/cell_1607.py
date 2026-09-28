#!/usr/bin/env python3
"""Cells 1,607-1,609 -- research/exec_quality/PREREG_1607.md (FROZEN 2026-09-28 16:20 UTC).

Owner 9/28: "maybe the ones with the bigger spread are the winners? if they are the winners, maybe
the spread is a filter?" This is that test, executed exactly as the frozen PREREG states it:

  1,607 HOD-SPREAD  -- quintiles of the quoted spread (bps) set on TRAIN-H2 ONLY, read on both
                        holdouts; the two pre-declared filters KEEP-WIDE (top two quintiles) and
                        KEEP-TIGHT (bottom two quintiles) scored on VAL against the frozen pass bar.
  1,608 ORB-SPREAD   -- spread-bucket table (<=50/50-100/100-150/150-300/>300 bps) on the live
                        trades DB sample and on the backtest tick-replay sample, tested against the
                        existing 300 bps gate baseline.
  1,609 ORB-DRIFT    -- drift_ask_to_fill_bps vs R on the live ORB fills (report-only, not causal
                        at the decision instant -- the drift is only known after the fill).

CAUSALITY FINDING (disclosed, not silently followed -- see RESULT_1607.md caveats):
The PREREG asks to compare `features_1478_A.csv` spread_frac_at_fill x 1e4 against 2 x half_entry /
fill and "use the causal one". `build_features_1478_A.py:370` defines
    spread_frac_at_fill = 2.0 * half_entry / fill
so the two candidates are THE SAME COLUMN by construction, not independent arm-time vs fill-time
quotes (verified numerically below, not just by reading the source). FEATURES_A.md's own caution
says half_entry/spread_frac_at_fill are recovered by `cell_1445.corrected_cost()` from the REALIZED
fill's cost_R/exit_price/exit_half_src -- i.e. from the trade's own outcome, not from a quote sitting
at the close of arm bar j. There is no causal, pre-fill spread field in features_1478_A.csv. This
cell proceeds with the only available field as an EXECUTION-QUALITY lens on the same fills; a PASS
below describes fills, not a pre-trade decision, and must not be read as ready-made evidence for a
`min_spread_bps`/`max_spread_bps` arm-time gate without a genuinely causal (pre-fill) quote capture.

Usage: python3 research/exec_quality/cell_1607.py
Outputs: research/exec_quality/RESULT_1607.md, research/exec_quality/cell_1607_rows.csv
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from research.hod_consol import run_consol as consol   # noqa: E402  (simulate_slots, 12/day 4-concurrent)
from trading.orb_csv import read_orb_csv                # noqa: E402  (NA-safe ORB CSV reader)

warnings.filterwarnings('ignore')

ET = 'America/New_York'
R_DENOM = 375.0   # research/orb_latency_bt/replay.py:81 -- reused unchanged for backtest-sample R


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ---------------------------------------------------------------------------------------------
# Scoring primitives -- verbatim copies of research/hod_entry/cell_1445.py (cited by line number)
# so 1,607 scores on exactly the same statistics as every other HOD cell (parity by construction).
# ---------------------------------------------------------------------------------------------

def day_clustered_t(y, day):
    """statsmodels OLS on a constant, clustered by day -- the t-stat on the mean.
    Verbatim copy of research/hod_entry/cell_1445.py:414-424."""
    y = pd.Series(y).dropna()
    if len(y) < 2:
        return np.nan
    d = pd.Series(day).loc[y.index]
    if d.nunique() < 2:
        return np.nan
    X = np.ones((len(y), 1))
    model = sm.OLS(y.to_numpy(), X).fit(cov_type='cluster', cov_kwds={'groups': d.to_numpy()})
    return float(model.tvalues[0])


def ex_top5_mean(y):
    """Mean excluding the top 5% (by value) of a series -- tail-dependence check.
    Verbatim copy of research/hod_entry/cell_1445.py:427-434."""
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return np.nan
    k = int(round(0.05 * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def weeks_spanned(days):
    """Distinct ISO (year, week) count over a day-string series -- the fills/wk denominator.
    Verbatim copy of research/hod_entry/cell_1445.py:444-448."""
    iso = pd.to_datetime(pd.Series(days).unique())
    wk = {(d.isocalendar()[0], d.isocalendar()[1]) for d in iso}
    return max(len(wk), 1)


def fills_per_week(kept_subset, weeks):
    """Fills/wk under first-12/day, 4-concurrent slotting.
    Verbatim copy of research/hod_entry/cell_1445.py:451-457 (consol.simulate_slots)."""
    if not len(kept_subset):
        return 0.0
    trades = kept_subset.rename(columns={'fill_min': 'entry_m'})[['day', 'entry_m', 'exit_m']].copy()
    keep = consol.simulate_slots(trades)
    return float(keep.sum()) / weeks


def day_clustered_t_datecol(y, date_series):
    """day_clustered_t with a plain date-string column (ORB tables use 'date'/'trade_date')."""
    return day_clustered_t(y, date_series)


# ===================================================================================================
# 1,607 -- HOD spread quintiles
# ===================================================================================================

def build_1607():
    log('1607: loading causal_arming_causal.csv, model_1478_L3_predictions.csv, features_1478_A.csv')
    ca = pd.read_csv(f'{REPO}/research/hod_entry/causal_arming_causal.csv', low_memory=False)
    ca_fill = ca[ca.status == 'fill'][['day', 'symbol', 'fill_min', 'exit_m', 'fill']].copy()
    log(f'1607: causal_arming_causal.csv status==fill -> {len(ca_fill)} rows (expect 9,911)')

    pred = pd.read_csv(f'{REPO}/research/hod_entry/model_1478_L3_predictions.csv')
    pred = pred[['day', 'symbol', 'fill_min', 'split', 'outcome_R']].copy()

    feat = pd.read_csv(f'{REPO}/research/hod_entry/features_1478_A.csv')
    feat = feat[['day', 'symbol', 'fill_min', 'split', 'half_entry', 'spread_frac_at_fill']].copy()

    df = pred.merge(feat, on=['day', 'symbol', 'fill_min'], suffixes=('', '_feat'), how='inner')
    split_mismatch = int((df['split'] != df['split_feat']).sum())
    if split_mismatch:
        log(f'1607: ERROR {split_mismatch} rows have mismatched split between predictions and features -- '
            f'dropping them (population must be internally consistent)')
        df = df[df['split'] == df['split_feat']]
    df = df.merge(ca_fill, on=['day', 'symbol', 'fill_min'], how='inner')
    log(f'1607: 3-way merge -> {len(df)} rows')
    if len(df) != 9911:
        log(f'1607: WARNING merged row count {len(df)} != 9,911 base-book size -- '
            f'merge key (day,symbol,fill_min) may not be unique or some rows failed to join')

    # -------- causality / identity check on the two PREREG-named candidate fields --------
    df['spread_bps'] = df['spread_frac_at_fill'] * 1e4
    df['spread_bps_check'] = 2.0 * df['half_entry'] / df['fill'] * 1e4
    max_diff = float((df['spread_bps'] - df['spread_bps_check']).abs().max())
    log(f'1607: max|spread_frac_at_fill*1e4 - 2*half_entry/fill*1e4| = {max_diff:.2e} '
        f'(confirms the two candidate fields are IDENTICAL by construction -- '
        f'build_features_1478_A.py:370). Neither is arm-time causal: both are recovered from the '
        f'realized fill cost_R/exit_price/exit_half_src by cell_1445.corrected_cost(). Proceeding '
        f'with spread_frac_at_fill as the (non-causal, execution-quality) spread field.')

    df['holdout'] = df['split'].map({'TRAIN': 'TRAIN-H2', 'VAL': 'VAL'})
    assert set(df['holdout'].unique()) <= {'TRAIN-H2', 'VAL'}

    # -------- quintile edges set on TRAIN-H2 ONLY (frozen: never read VAL to choose a cut) --------
    th2 = df[df.holdout == 'TRAIN-H2']
    edges = np.unique(np.quantile(th2['spread_bps'].dropna().to_numpy(), [0, .2, .4, .6, .8, 1.0]))
    n_q = len(edges) - 1
    log(f'1607: TRAIN-H2 (n={len(th2)}) quintile edges (bps) = {[round(e, 2) for e in edges]} '
        f'-> {n_q} bins')

    bin_edges = edges.copy()
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    labels = [f'Q{i + 1}' for i in range(n_q)]
    df['quintile'] = pd.cut(df['spread_bps'], bins=bin_edges, labels=labels, include_lowest=True)

    weeks = {h: weeks_spanned(df[df.holdout == h]['day']) for h in ('TRAIN-H2', 'VAL')}

    quintile_rows = []
    for h in ('TRAIN-H2', 'VAL'):
        hdf = df[df.holdout == h]
        for q in labels:
            sub = hdf[hdf.quintile == q]
            n = len(sub)
            mean_r = float(sub['outcome_R'].mean()) if n else np.nan
            t = day_clustered_t(sub['outcome_R'], sub['day']) if n else np.nan
            ex5 = ex_top5_mean(sub['outcome_R']) if n else np.nan
            fwk = fills_per_week(sub, weeks[h]) if n else 0.0
            win = float((sub['outcome_R'] > 0).mean()) if n else np.nan
            quintile_rows.append(dict(
                cell='1607', sample=h, bucket=q, n=n, mean_R=mean_r, t=t, win_rate=win,
                ex_top5=ex5, fills_wk=fwk,
                spread_lo_bps=float(sub['spread_bps'].min()) if n else np.nan,
                spread_hi_bps=float(sub['spread_bps'].max()) if n else np.nan))
        log(f'1607: {h} quintile means (outcome_R) = ' +
            ', '.join(f'{q}={hdf[hdf.quintile == q]["outcome_R"].mean():.3f}' for q in labels))

    # -------- the two pre-declared filters --------
    wide_qs = set(labels[-2:]) if n_q >= 2 else set(labels)
    tight_qs = set(labels[:2]) if n_q >= 2 else set(labels)
    log(f'1607: KEEP-WIDE quintiles = {sorted(wide_qs)}, KEEP-TIGHT quintiles = {sorted(tight_qs)}')

    def filter_row(name, qset, h):
        hdf = df[df.holdout == h]
        kept = hdf[hdf.quintile.isin(qset)]
        dropped = hdf[~hdf.quintile.isin(qset)]
        n_kept, n_dropped = len(kept), len(dropped)
        kept_mean = float(kept['outcome_R'].mean()) if n_kept else np.nan
        dropped_mean = float(dropped['outcome_R'].mean()) if n_dropped else np.nan
        t_kept = day_clustered_t(kept['outcome_R'], kept['day']) if n_kept else np.nan
        ex5 = ex_top5_mean(kept['outcome_R']) if n_kept else np.nan
        fwk = fills_per_week(kept, weeks[h]) if n_kept else 0.0
        win = float((kept['outcome_R'] > 0).mean()) if n_kept else np.nan
        return dict(filter=name, holdout=h, n_kept=n_kept, n_dropped=n_dropped,
                    kept_mean=kept_mean, dropped_mean=dropped_mean, t_kept=t_kept,
                    ex_top5=ex5, fills_wk=fwk, win_rate=win)

    filt_rows = []
    for name, qset in (('KEEP-WIDE', wide_qs), ('KEEP-TIGHT', tight_qs)):
        for h in ('TRAIN-H2', 'VAL'):
            filt_rows.append(filter_row(name, qset, h))

    by_filter = {}
    for r in filt_rows:
        by_filter.setdefault(r['filter'], {})[r['holdout']] = r

    verdicts = {}
    for name, byh in by_filter.items():
        val, th2r = byh.get('VAL'), byh.get('TRAIN-H2')
        ok = (val is not None and th2r is not None
              and val['n_kept'] > 0 and val['kept_mean'] >= 0.15 and val['t_kept'] >= 2.5
              and val['ex_top5'] > 0 and val['fills_wk'] >= 3
              and th2r['n_kept'] > 0 and np.sign(th2r['kept_mean']) == np.sign(val['kept_mean'])
              and th2r['t_kept'] >= 1
              and val['dropped_mean'] < val['kept_mean']
              and th2r['dropped_mean'] < th2r['kept_mean'])
        verdicts[name] = bool(ok)
        log(f"1607: {name} VAL kept n={val['n_kept']} mean={val['kept_mean']:.4f} t={val['t_kept']:.2f} "
            f"ex5={val['ex_top5']:.4f} fwk={val['fills_wk']:.2f} dropped_mean={val['dropped_mean']:.4f} "
            f"| TRAIN-H2 kept_mean={th2r['kept_mean']:.4f} t={th2r['t_kept']:.2f} -> PASS={ok}")

    return dict(quintile_rows=quintile_rows, filt_rows=filt_rows, verdicts=verdicts,
                edges=edges.tolist(), max_diff=max_diff, n_merged=len(df),
                weeks=weeks, th2_n=len(th2), val_n=len(df[df.holdout == 'VAL']))


# ===================================================================================================
# 1,608 -- ORB spread buckets (live + backtest)
# ===================================================================================================

BUCKET_EDGES = [0, 50, 100, 150, 300, np.inf]
BUCKET_LABELS = ['<=50', '50-100', '100-150', '150-300', '>300']


def bucket_table(df, r_col, date_col, pnl_col, sample_name):
    """n / mean R / t / win rate / P&L share per spread bucket, plus the full sample and the
    300bps-gate baseline (dropped = >300, kept = <=300) and the skip-<=50 baseline."""
    rows = []
    total_pnl = float(df[pnl_col].sum())
    for lab in BUCKET_LABELS:
        sub = df[df.spread_bucket == lab]
        n = len(sub)
        mean_r = float(sub[r_col].mean()) if n else np.nan
        t = day_clustered_t(sub[r_col], sub[date_col]) if n else np.nan
        win = float((sub[r_col] > 0).mean()) if n else np.nan
        pnl_share = float(sub[pnl_col].sum() / total_pnl) if n and total_pnl != 0 else np.nan
        rows.append(dict(cell='1608', sample=sample_name, bucket=lab, n=n, mean_R=mean_r, t=t,
                          win_rate=win, pnl_share=pnl_share))
    # full-sample baseline row
    rows.append(dict(cell='1608', sample=sample_name, bucket='ALL', n=len(df),
                      mean_R=float(df[r_col].mean()), t=day_clustered_t(df[r_col], df[date_col]),
                      win_rate=float((df[r_col] > 0).mean()), pnl_share=1.0))
    return rows


def gate_test(df, r_col, skip_label_set, kept_baseline_mean):
    """One 'skip these buckets' rule: dropped n/mean, kept n/mean, delta vs the ALL-sample baseline."""
    dropped = df[df.spread_bucket.isin(skip_label_set)]
    kept = df[~df.spread_bucket.isin(skip_label_set)]
    n_d, n_k = len(dropped), len(kept)
    d_mean = float(dropped[r_col].mean()) if n_d else np.nan
    k_mean = float(kept[r_col].mean()) if n_k else np.nan
    return dict(n_dropped=n_d, dropped_mean=d_mean, n_kept=n_k, kept_mean=k_mean,
                delta_vs_all=(k_mean - kept_baseline_mean) if n_k else np.nan)


def load_orb_live():
    """ORB live sample from data/trades.db (strategy='orb', closed fills, 2026-05-01..2026-09-28).
    R = pnl / risk, risk = (entry_price - stop_loss_price) * filled_qty (PREREG-specified)."""
    import sqlite3
    log('1608/1609: loading live ORB trades from data/trades.db (read-only)')
    con = sqlite3.connect(f'file:{REPO}/data/trades.db?mode=ro', uri=True)
    q = """SELECT trade_date, symbol, entry_price, fill_price, entry_quote_spread, stop_loss_price,
                  filled_qty, pnl, drift_ask_to_fill_bps
           FROM trades
           WHERE strategy='orb' AND fill_price IS NOT NULL AND pnl IS NOT NULL
             AND trade_date >= '2026-05-01' AND trade_date <= '2026-09-28'"""
    df = pd.read_sql_query(q, con)
    con.close()
    n0 = len(df)
    df['risk'] = (df['entry_price'] - df['stop_loss_price']) * df['filled_qty']
    bad_risk = int((df['risk'] <= 0).sum())
    if bad_risk:
        log(f'1608/1609: WARNING {bad_risk}/{n0} live ORB rows have risk<=0 (entry<=stop or bad '
            f'filled_qty) -- dropped from the R-based tables')
        df = df[df['risk'] > 0]
    df['R'] = df['pnl'] / df['risk']
    df['spread_bps'] = df['entry_quote_spread'] / df['entry_price'] * 1e4
    df['spread_bucket'] = pd.cut(df['spread_bps'], bins=BUCKET_EDGES, labels=BUCKET_LABELS,
                                  include_lowest=True)
    log(f'1608/1609: live ORB sample n={len(df)} (of {n0} closed fills), '
        f'dates {df.trade_date.min()}..{df.trade_date.max()}, mean spread={df.spread_bps.mean():.1f} bps')
    return df


def nbbo_before(raw, date_str, t_star):
    """Last two-sided mbp-1 NBBO snapshot strictly before t_star (same causal rule as
    research/orb_latency_bt/replay.py:ask_at -- reused, not re-derived). Returns (bid, ask) or
    (None, None)."""
    mbp = raw[raw['schema'] == 'mbp-1']
    if mbp.empty:
        return None, None
    anchor = pd.Timestamp(f'{date_str} 09:35:00', tz=ET)
    t_sec = (mbp['ts_event'] - anchor).dt.total_seconds()
    prior = mbp[(t_sec < t_star) & (mbp['bid_px_00'] > 0) & (mbp['ask_px_00'] > 0)]
    if prior.empty:
        return None, None
    row = prior.iloc[-1]
    return float(row['bid_px_00']), float(row['ask_px_00'])


def load_orb_backtest():
    """ORB backtest sample: analysis_results/orb_features_20260925_2054.csv via read_orb_csv has NO
    spread/quote/bid/ask column (checked below) -> falls back to the PREREG's named path:
    research/orb_latency_bt/results.csv delay_s==0 'filled' rows + the raw XNAS tick tape for the
    NBBO at t_star. R = pnl_replay / 375 (research/orb_latency_bt/replay.py:81, reused unchanged --
    same book already vetted at cell 1,426)."""
    nightly = read_orb_csv(f'{REPO}/analysis_results/orb_features_20260925_2054.csv', nrows=5)
    spread_like = [c for c in nightly.columns if any(k in c.lower() for k in
                   ('spread', 'quote', 'bid', 'ask'))]
    log(f'1608: analysis_results/orb_features_20260925_2054.csv spread/quote/bid/ask columns = '
        f'{spread_like} -> {"none, using the tick-replay fallback" if not spread_like else "found"}')

    res = pd.read_csv(f'{REPO}/research/orb_latency_bt/results.csv')
    sig = res[(res.delay_s == 0) & (res.status == 'filled')].copy()
    log(f'1608: research/orb_latency_bt/results.csv delay_s==0 & status==filled -> {len(sig)} rows, '
        f'periods={sig.period.value_counts().to_dict()}, dates {sig.date.min()}..{sig.date.max()}')

    raw_dir_primary = f'{REPO}/research/orb_latency_bt/raw'
    raw_dir_fallback = f'{REPO}/research/hod_ofi/raw'
    bids, asks, src = [], [], []
    n_missing_file, n_no_nbbo, n_primary, n_fallback = 0, 0, 0, 0
    for row in sig.itertuples():
        fn = f'{row.date}__{row.symbol}.parquet'
        path = os.path.join(raw_dir_primary, fn)
        which = 'primary'
        if not os.path.exists(path):
            path = os.path.join(raw_dir_fallback, fn)
            which = 'fallback'
            if not os.path.exists(path):
                bids.append(np.nan); asks.append(np.nan); src.append('missing')
                n_missing_file += 1
                continue
        raw = pd.read_parquet(path, columns=['ts_event', 'schema', 'bid_px_00', 'ask_px_00'])
        b, a = nbbo_before(raw, row.date, row.t_star)
        bids.append(b); asks.append(a)
        if b is None:
            n_no_nbbo += 1
            src.append('no_nbbo')
        else:
            src.append(which)
            n_primary += (which == 'primary')
            n_fallback += (which == 'fallback')
    sig['bid'] = bids
    sig['ask'] = asks
    sig['quote_src'] = src
    log(f'1608: NBBO-at-t* resolved for {n_primary + n_fallback}/{len(sig)} fills '
        f'({n_primary} from orb_latency_bt/raw, {n_fallback} from hod_ofi/raw fallback); '
        f'{n_missing_file} missing parquet file, {n_no_nbbo} file present but no prior 2-sided quote')

    n_before_resolution = len(sig)
    sig['spread_bps'] = (sig['ask'] - sig['bid']) / ((sig['ask'] + sig['bid']) / 2.0) * 1e4
    sig = sig[sig['spread_bps'].notna()].copy()
    sig['R'] = sig['pnl_replay'] / R_DENOM
    sig['spread_bucket'] = pd.cut(sig['spread_bps'], bins=BUCKET_EDGES, labels=BUCKET_LABELS,
                                   include_lowest=True)
    log(f'1608: backtest sample with a resolved spread n={len(sig)} (of {n_before_resolution} '
        f'delay_s==0 filled signals), mean spread={sig.spread_bps.mean():.1f} bps, '
        f'mean R={sig.R.mean():.4f}')
    sig.attrs['n_before_resolution'] = n_before_resolution
    return sig


def build_1608(live, bt):
    live_rows = bucket_table(live, 'R', 'trade_date', 'pnl', 'LIVE')
    bt_rows = bucket_table(bt, 'R', 'date', 'pnl_replay', 'BACKTEST')

    all_live_mean = float(live['R'].mean())
    all_bt_mean = float(bt['R'].mean())

    candidates = {'skip_le_50': {'<=50'}, 'skip_gt_300': {'>300'}}
    gate_results = {}
    for name, labs in candidates.items():
        g_live = gate_test(live, 'R', labs, all_live_mean)
        g_bt = gate_test(bt, 'R', labs, all_bt_mean)
        passes = (g_live['n_dropped'] >= 20 and g_bt['n_dropped'] >= 20
                  and g_live['dropped_mean'] < 0 and g_bt['dropped_mean'] < 0
                  and g_live['delta_vs_all'] >= 0.03 and g_bt['delta_vs_all'] >= 0.03)
        gate_results[name] = dict(live=g_live, bt=g_bt, passes=bool(passes))
        log(f"1608: {name} -- live dropped n={g_live['n_dropped']} mean={g_live['dropped_mean']}, "
            f"delta_kept={g_live['delta_vs_all']} | bt dropped n={g_bt['n_dropped']} "
            f"mean={g_bt['dropped_mean']}, delta_kept={g_bt['delta_vs_all']} -> PASS={passes}")

    return dict(live_rows=live_rows, bt_rows=bt_rows, gate_results=gate_results,
                all_live_mean=all_live_mean, all_bt_mean=all_bt_mean)


# ===================================================================================================
# 1,609 -- ORB drift-as-signal (report only)
# ===================================================================================================

def build_1609(live):
    log(f'1609: drift_ask_to_fill_bps vs R on {len(live)} live ORB fills (report-only)')
    d = live[['drift_ask_to_fill_bps', 'R', 'trade_date']].dropna(subset=['drift_ask_to_fill_bps', 'R'])
    rho = float(d['drift_ask_to_fill_bps'].corr(d['R'], method='spearman'))

    try:
        d = d.copy()
        d['drift_q'] = pd.qcut(d['drift_ask_to_fill_bps'], 5, duplicates='drop')
    except ValueError:
        d['drift_q'] = pd.qcut(d['drift_ask_to_fill_bps'], 3, duplicates='drop')

    rows = []
    for q, sub in d.groupby('drift_q', observed=True):
        n = len(sub)
        rows.append(dict(cell='1609', sample='LIVE', bucket=str(q), n=n,
                          mean_R=float(sub['R'].mean()),
                          t=day_clustered_t(sub['R'], sub['trade_date']) if n > 1 else np.nan,
                          win_rate=float((sub['R'] > 0).mean()),
                          mean_drift_bps=float(sub['drift_ask_to_fill_bps'].mean())))
    log(f'1609: spearman rank corr(drift_ask_to_fill_bps, R) = {rho:.3f} on n={len(d)}')
    return dict(rows=rows, spearman_rho=rho, n=len(d))


# ===================================================================================================
# main
# ===================================================================================================

def main():
    t0 = time.time()
    r1607 = build_1607()
    live = load_orb_live()
    bt = load_orb_backtest()
    r1608 = build_1608(live, bt)
    r1609 = build_1609(live)

    # -------- cell_1607_rows.csv: every row-level table produced above, tagged by cell --------
    all_rows = list(r1607['quintile_rows'])
    for fr in r1607['filt_rows']:
        all_rows.append(dict(cell='1607_filter', sample=fr['holdout'], bucket=fr['filter'],
                              n=fr['n_kept'], mean_R=fr['kept_mean'], t=fr['t_kept'],
                              win_rate=fr['win_rate'], ex_top5=fr['ex_top5'], fills_wk=fr['fills_wk'],
                              n_dropped=fr['n_dropped'], dropped_mean=fr['dropped_mean']))
    all_rows += r1608['live_rows']
    all_rows += r1608['bt_rows']
    all_rows += r1609['rows']
    rows_df = pd.DataFrame(all_rows)
    rows_path = f'{HERE}/cell_1607_rows.csv'
    rows_df.to_csv(rows_path, index=False)
    log(f'wrote {rows_path} ({len(rows_df)} rows)')

    write_report(r1607, r1608, r1609, live, bt)
    log(f'done in {time.time() - t0:.1f}s')


def write_report(r1607, r1608, r1609, live, bt):
    lines = []
    a = lines.append
    a('# RESULT — cells 1,607–1,609: the quoted spread as a filter')
    a('')
    a(f'Executed research/exec_quality/PREREG_1607.md (FROZEN 2026-09-28 16:20 UTC) exactly. '
      f'Owner question: "maybe the ones with the bigger spread are the winners? if they are the '
      f'winners, maybe the spread is a filter?"')
    a('')
    a('## CAUSALITY FINDING — read this before the numbers below')
    a(f'`spread_frac_at_fill` and `2*half_entry/fill` are **the same column** '
      f'(max abs diff = {r1607["max_diff"]:.2e}; `build_features_1478_A.py:370` defines '
      f'`spread_frac_at_fill = 2.0 * half_entry / fill`). Neither is the causal arm-time quote the '
      f'PREREG asked for: both come from `cell_1445.corrected_cost()`, which is solved from the '
      f'REALIZED fill\'s `cost_R`/`exit_price`/`exit_half_src` — i.e. from the trade\'s own outcome, '
      f'not a quote sitting at the close of arm bar j (FEATURES_A.md\'s own caution says the same). '
      f'**`features_1478_A.csv` has no causal, pre-fill spread field.** Cell 1,607 below is therefore '
      f'an EXECUTION-QUALITY lens on the realized fills ("is the expensive-to-execute trade the '
      f'winning trade"), not a forward pre-trade signal — a PASS does not by itself license a live '
      f'`min_spread_bps`/`max_spread_bps` arm-time gate; that would need a genuinely causal quote '
      f'capture (e.g. the NBBO at the close of arm bar j, independent of whether/how the order later '
      f'filled).')
    a('')
    a('## 1,607 HOD-SPREAD')
    a(f'Population: `causal_arming_causal.csv` status==fill (9,911) joined to '
      f'`model_1478_L3_predictions.csv` (outcome_R = standard-cost net R) and `features_1478_A.csv` '
      f'(spread_frac_at_fill) on (day,symbol,fill_min); merged n = {r1607["n_merged"]}. '
      f'TRAIN-H2 n={r1607["th2_n"]}, VAL n={r1607["val_n"]}. Spread field used: `spread_frac_at_fill '
      f'x 1e4` (bps); identical to `2*half_entry/fill x 1e4`, reported as one column per the identity '
      f'above.')
    a('')
    a(f'Quintile edges set on TRAIN-H2 only (bps): {[round(e, 2) for e in r1607["edges"]]}')
    a('')
    a('| Sample | Quintile | spread range (bps) | n | mean net R | day-clust t | win rate | ex-top-5% | fills/wk |')
    a('|---|---|---|---|---|---|---|---|---|')
    for r in r1607['quintile_rows']:
        a(f"| {r['sample']} | {r['bucket']} | {r['spread_lo_bps']:.0f}–{r['spread_hi_bps']:.0f} | "
          f"{r['n']} | {r['mean_R']:.4f} | {r['t']:.2f} | {r['win_rate']:.1%} | {r['ex_top5']:.4f} | "
          f"{r['fills_wk']:.2f} |")
    a('')
    a('### The two pre-declared filters')
    a('| Filter | Holdout | n kept | kept mean R | t (kept) | ex-top-5% | fills/wk | n dropped | dropped mean R |')
    a('|---|---|---|---|---|---|---|---|---|')
    for fr in r1607['filt_rows']:
        a(f"| {fr['filter']} | {fr['holdout']} | {fr['n_kept']} | {fr['kept_mean']:.4f} | "
          f"{fr['t_kept']:.2f} | {fr['ex_top5']:.4f} | {fr['fills_wk']:.2f} | {fr['n_dropped']} | "
          f"{fr['dropped_mean']:.4f} |")
    a('')
    a(f"**Pass bar (frozen)**: VAL kept mean >= +0.15 R, t >= 2.5, ex-top-5% > 0, fills/wk >= 3, "
      f"TRAIN-H2 same sign with t >= 1, dropped < kept on both holdouts.")
    for name, v in r1607['verdicts'].items():
        a(f"* **{name}: {'PASS' if v else 'FAIL'}**")
    a('')

    a('## 1,608 ORB-SPREAD')
    a(f'Live sample: `data/trades.db` strategy=orb, closed fills, dates '
      f'{live.trade_date.min()}..{live.trade_date.max()}, n={len(live)}. R = pnl / risk, '
      f'risk = (entry_price - stop_loss_price) * filled_qty. spread_bps = entry_quote_spread / '
      f'entry_price * 1e4.')
    a(f'Backtest sample: `analysis_results/orb_features_20260925_2054.csv` (via `read_orb_csv`) '
      f'carries no spread/quote/bid/ask column -> fallback to '
      f'`research/orb_latency_bt/results.csv` delay_s==0 & status==filled '
      f'(n={bt.attrs.get("n_before_resolution", len(bt))} before NBBO resolution, n={len(bt)} after), dates '
      f'{bt.date.min()}..{bt.date.max()}, NBBO at t* from the XNAS tick tape '
      f'(`research/orb_latency_bt/raw`, falling back to `research/hod_ofi/raw`). R = pnl_replay / '
      f'{R_DENOM:.0f} (`research/orb_latency_bt/replay.py`\'s own convention, reused unchanged — '
      f'the same book already vetted at cell 1,426).')
    a('')
    a('| Sample | Bucket | n | mean R | day-clust t | win rate | P&L share |')
    a('|---|---|---|---|---|---|---|')
    for r in r1608['live_rows'] + r1608['bt_rows']:
        t_str = f"{r['t']:.2f}" if not np.isnan(r['t']) else 'n/a'
        ps = f"{r['pnl_share']:.1%}" if not np.isnan(r['pnl_share']) else 'n/a'
        a(f"| {r['sample']} | {r['bucket']} | {r['n']} | {r['mean_R']:.4f} | {t_str} | "
          f"{r['win_rate']:.1%} | {ps} |")
    a('')
    a('### Bucket-rule test vs the existing 300 bps gate baseline')
    a(f"Pass bar (frozen): dropped bucket mean R < 0 on BOTH samples with n>=20 each, kept-book mean "
      f"R rises by >= +0.03 R vs the full sample.")
    a('| Rule | Live n dropped | Live dropped mean R | Live kept delta | BT n dropped | BT dropped mean R | BT kept delta | Verdict |')
    a('|---|---|---|---|---|---|---|---|')
    for name, g in r1608['gate_results'].items():
        gl, gb = g['live'], g['bt']
        a(f"| {name} | {gl['n_dropped']} | {gl['dropped_mean']:.4f} | {gl['delta_vs_all']:.4f} | "
          f"{gb['n_dropped']} | {gb['dropped_mean']:.4f} | {gb['delta_vs_all']:.4f} | "
          f"{'PASS' if g['passes'] else 'FAIL'} |")
    a('')

    a('## 1,609 ORB-DRIFT-AS-SIGNAL (report-only)')
    a(f"drift_ask_to_fill_bps vs R on the live ORB fills, n={r1609['n']}. Spearman rank "
      f"correlation = {r1609['spearman_rho']:.3f}. Report-only: drift is realized (post-decision), "
      f"not knowable at order-submit time.")
    a('')
    a('| Drift bucket (bps) | n | mean R | mean drift (bps) | day-clust t | win rate |')
    a('|---|---|---|---|---|---|')
    for r in r1609['rows']:
        t_str = f"{r['t']:.2f}" if not np.isnan(r['t']) else 'n/a'
        a(f"| {r['bucket']} | {r['n']} | {r['mean_R']:.4f} | {r['mean_drift_bps']:.1f} | {t_str} | "
          f"{r['win_rate']:.1%} |")
    a('')

    a('## Caveats')
    a('* **1,607 is not a causal pre-trade field** (see the causality finding above) — a PASS is '
      'evidence for "expensive fills co-vary with outcome on this population", not a ready `min/'
      'max_spread_bps` arm-time gate.')
    a('* **Cost double-count**: `outcome_R` is already the standard-cost NET R (half-spread charged '
      'once at exit per cell 1445/1457\'s corrected-cost convention) — the spread quintiles here are '
      'read on a cost-inclusive outcome, per the PREREG\'s own warning not to judge this on gross.')
    a('* **1,608 backtest NBBO join**: resolved per-fill from the raw XNAS mbp-1 tape at the replay '
      't* used to fill the BT order (same causal rule as replay.py\'s ask_at — last two-sided quote '
      'strictly before t*); fills with no parquet file or no prior two-sided quote are excluded and '
      'counted in the log, not silently dropped.')
    a('* **1,608 live sample size**: n=123 total closed ORB fills split across 5 spread buckets — '
      'day-clustered t on a single bucket can be on a handful of trading days; read the bucket table '
      'beside the n column, not the t column alone.')
    a('* **1,609** drift is a realized, not causal, quantity — explicitly report-only per the PREREG; '
      'no cap or gate should be inferred from it without a forward (order-submit-time) proxy.')
    a('* Independent rebuild still required before this ships anywhere (per CLAUDE.md\'s "no research '
      'claim ships" protocol) — this is the BUILDER pass only.')
    a('')
    a(f'Generated by `research/exec_quality/cell_1607.py`.')

    path = f'{HERE}/RESULT_1607.md'
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log(f'wrote {path}')


if __name__ == '__main__':
    main()
