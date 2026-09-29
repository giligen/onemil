#!/usr/bin/env python3
"""Cells 1,655-1,657: the first five seconds after 09:35:00 ET.

Implements research/orb_deadzone/PREREG_1655.md (frozen 2026-09-29 08:20 UTC).
Reuses cell 1,426's frozen outputs and fill logic verbatim:
  * research/orb_latency_bt/results.csv (per-fill, per-delay replay of the exact
    same chase rule at delay_s in {0, 5} -- NOT recomputed here) for t* (AT-or-
    through: research/orb_latency_bt/replay.py::find_t_star, price >= trigger),
    fill status and pnl at delay 0 and delay 5.
  * research/orb_latency_bt/replay.py's day_clustered_t / ex_top5_mean_r / df_to_md
    / to_sec_since_0935 / R_DENOM / WIN_END_S are imported and called, not rewritten.
The only new computation here is the THROUGH-only t* variant (price > trigger,
strict), which cell 1,426 never stored -- that requires one pass over the raw
trade prints per fill.

Usage: python3 cell_1655.py
Writes: research/orb_deadzone/RESULT_1655.md, research/orb_deadzone/buckets_1655.csv
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/orb_latency_bt')
os.chdir(ROOT)

import replay as R  # research/orb_latency_bt/replay.py -- reuse, do not rewrite

OUT_DIR = f'{ROOT}/research/orb_deadzone'
POP_CSV = f'{ROOT}/research/orb_latency_bt/population.csv'
RESULTS_CSV = f'{ROOT}/research/orb_latency_bt/results.csv'
RAW_DIR = f'{ROOT}/research/orb_latency_bt/raw'
RESULT_MD = f'{OUT_DIR}/RESULT_1655.md'
BUCKETS_CSV = f'{OUT_DIR}/buckets_1655.csv'

BUCKET_EDGES = [('0-5', 0.0, 5.0), ('5-15', 5.0, 15.0), ('15-30', 15.0, 30.0),
                ('30-60', 30.0, 60.0), ('60-300', 60.0, 300.0)]
BUCKET_ORDER = [b[0] for b in BUCKET_EDGES] + ['no_t*']

# 1,426's published delay-0 totals (research/orb_latency_bt/REPORT.md), for the
# reconciliation check -- transcribed, not recomputed.
PUBLISHED_D0 = {
    'out_of_sample': dict(n=537, total_usd=5578.015),
    '2025H1': dict(n=80, total_usd=3674.511),
    'May-Sep_2026': dict(n=153, total_usd=729.269),
}


def log(msg: str) -> None:
    """Verbose, timestamped progress line (flushed immediately)."""
    print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {msg}', flush=True)


def bucket_of(t) -> str:
    """Assign a t* (seconds since 09:35:00.000 ET) to one of the five PREREG
    buckets, or 'no_t*' if no qualifying tick was found in the window."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return 'no_t*'
    for name, lo, hi in BUCKET_EDGES:
        if lo <= t < hi:
            return name
    return 'no_t*'  # outside [0,300) -- cannot happen given replay.py's signal window


def load_merged() -> pd.DataFrame:
    """One row per population fill: t_star (AT-or-through, from 1,426's results.csv),
    delay-0 and delay-5 status/pnl (from the same file -- NOT recomputed), period,
    trigger, and the original BT win/pnl for the coverage line."""
    pop = pd.read_csv(POP_CSV)
    res = pd.read_csv(RESULTS_CSV)
    d0 = res[res.delay_s == 0][['symbol', 'date', 'period', 'status', 'fill_price', 't_star', 'pnl_replay']] \
        .rename(columns={'status': 'status_d0', 'fill_price': 'fill_price_d0', 'pnl_replay': 'pnl_d0'})
    d5 = res[res.delay_s == 5][['symbol', 'date', 'status', 'fill_price', 'pnl_replay']] \
        .rename(columns={'status': 'status_d5', 'fill_price': 'fill_price_d5', 'pnl_replay': 'pnl_d5'})
    m = d0.merge(d5, on=['symbol', 'date'], how='left', validate='one_to_one')
    m = m.merge(pop[['symbol', 'date', 'win', 'pnl', 'trigger']], on=['symbol', 'date'],
                how='left', validate='one_to_one')
    m['has_ticks'] = m['status_d0'] != 'missing_tick_data'
    n_dupe_check = len(res[res.delay_s == 0])
    assert len(m) == n_dupe_check == len(pop), \
        f'row count mismatch: population={len(pop)} results_d0={n_dupe_check} merged={len(m)}'
    m.loc[~m.has_ticks, ['pnl_d0', 'pnl_d5']] = np.nan
    m['t_star_at'] = m['t_star']
    m['bucket_at'] = m['t_star_at'].apply(bucket_of)
    m['date_dt'] = pd.to_datetime(m['date'])
    return m


def compute_t_star_through(m: pd.DataFrame) -> dict:
    """First trade print STRICTLY greater than trigger in [0,300)s since 09:35:00,
    for every has_ticks fill -- read directly from raw/*.parquet (the AT-or-through
    variant is already in results.csv; this THROUGH-only variant is not, so it is
    the one genuinely new computation in this cell). Reuses R.to_sec_since_0935 and
    R.WIN_END_S; does not touch replay_one or the chase/fill logic."""
    out = {}
    todo = m[m.has_ticks]
    n = 0
    for r in todo.itertuples():
        path = f'{RAW_DIR}/{r.date}__{r.symbol}.parquet'
        if not os.path.exists(path):
            out[(r.symbol, r.date)] = None
            continue
        raw = pd.read_parquet(path, columns=['ts_event', 'price', 'schema'])
        trades = raw[raw['schema'] == 'trades']
        if trades.empty:
            out[(r.symbol, r.date)] = None
        else:
            t_sec = R.to_sec_since_0935(trades['ts_event'].values, r.date)
            sig = pd.DataFrame({'t_sec': t_sec, 'price': trades['price'].values})
            sig = sig[(sig.t_sec >= 0.0) & (sig.t_sec < R.WIN_END_S)].sort_values('t_sec')
            hit = sig[sig.price > r.trigger]
            out[(r.symbol, r.date)] = float(hit['t_sec'].iloc[0]) if len(hit) else None
        n += 1
        if n % 100 == 0 or n == len(todo):
            log(f'THROUGH-only t*: {n}/{len(todo)} fills read from raw parquet')
    return out


def day_t_of(df: pd.DataFrame, pnl_col: str) -> float:
    """Day-clustered t for pnl_col, via R.day_clustered_t (value_col-parametrized)."""
    if df.empty:
        return np.nan
    return R.day_clustered_t(df, value_col=pnl_col)


def ex_top5_of(df: pd.DataFrame, pnl_col: str) -> float:
    """Ex-top-5% mean R for pnl_col, via R.ex_top5_mean_r (hardcodes 'pnl_replay')."""
    if df.empty:
        return np.nan
    return R.ex_top5_mean_r(df.rename(columns={pnl_col: 'pnl_replay'}))


def ex_top5_trim_series(s: pd.Series) -> float:
    """Ex-top-5% trimmed mean of an already-in-R series (used for paired delta R,
    which is not a pnl_replay column so R.ex_top5_mean_r does not apply directly)."""
    if s.empty:
        return np.nan
    s = s.sort_values(ascending=False)
    k = int(np.ceil(0.05 * len(s)))
    return float(s.iloc[k:].mean()) if k > 0 else float(s.mean())


def bucket_table(df: pd.DataFrame, pnl_col: str, bucket_col: str = 'bucket_at') -> pd.DataFrame:
    """PREREG 1,655 bucket table: n, mean_R, day_t, win_pct, ex_top5pct_R, total_usd
    per t* bucket (including a 'no_t*' row so the table sums to the full book)."""
    rows = []
    for b in BUCKET_ORDER:
        sub = df[df[bucket_col] == b]
        n = len(sub)
        if n == 0:
            rows.append(dict(bucket=b, n=0, mean_R=np.nan, day_t=np.nan, win_pct=np.nan,
                              ex_top5pct_R=np.nan, total_usd=0.0))
            continue
        total_usd = float(sub[pnl_col].sum())
        mean_r = float((sub[pnl_col] / R.R_DENOM).mean())
        t = day_t_of(sub[['date', pnl_col]], pnl_col)
        ex5 = ex_top5_of(sub[['date', pnl_col]], pnl_col)
        win_pct = float((sub[pnl_col] > 0).mean())
        rows.append(dict(bucket=b, n=n, mean_R=mean_r, day_t=t, win_pct=win_pct,
                          ex_top5pct_R=ex5, total_usd=total_usd))
    return pd.DataFrame(rows)


def reconcile(name: str, df: pd.DataFrame) -> dict:
    """Compare this cell's delay-0 (n, total $) summed across ALL buckets against
    1,426's published REPORT.md numbers for the same period -- PREREG's pass bar
    for the independent check (same n, total $ within $1)."""
    n = len(df)
    total = float(df['pnl_d0'].sum())
    pub = PUBLISHED_D0[name]
    delta = total - pub['total_usd']
    return dict(period=name, n_mine=n, n_1426=pub['n'], total_mine=round(total, 3),
                total_1426=pub['total_usd'], delta_usd=round(delta, 3),
                match=(n == pub['n']) and (abs(delta) <= 1.0))


def delay5_stats(df: pd.DataFrame) -> dict:
    """1,656: mean R/fill, total $, n filled, n guard-skipped at delay 5, paired
    ΔR per fill (delay5 - delay0) and its ex-top-5% (tail check)."""
    n = len(df)
    n_filled = int((df.status_d5 == 'filled').sum())
    n_skipped = int((df.status_d5 == 'skipped_guard').sum())
    n_skipped_d0 = int((df.status_d0 == 'skipped_guard').sum())
    total_usd = float(df['pnl_d5'].sum())
    mean_r = float((df['pnl_d5'] / R.R_DENOM).mean())
    t = day_t_of(df[['date', 'pnl_d5']], 'pnl_d5')
    ex5 = ex_top5_of(df[['date', 'pnl_d5']], 'pnl_d5')
    baseline_mean_r = float((df['pnl_d0'] / R.R_DENOM).mean())
    delta_r = (df['pnl_d5'] - df['pnl_d0']) / R.R_DENOM
    return dict(n=n, n_filled=n_filled, n_skipped_guard=n_skipped, n_skipped_guard_d0=n_skipped_d0,
                total_usd=total_usd, mean_R=mean_r, day_t=t, ex_top5pct_R=ex5,
                baseline_mean_R=baseline_mean_r, mean_delta_R=float(delta_r.mean()),
                ex_top5pct_delta_R=ex_top5_trim_series(delta_r))


def skip_instant_stats(df: pd.DataFrame) -> dict:
    """1,657: drop fills with t*_at < 5s entirely (no refill), report the surviving
    book at delay 0 vs the unfiltered delay-0 baseline."""
    excluded = df[df.bucket_at == '0-5']
    remaining = df[df.bucket_at != '0-5']
    mean_r_remaining = float((remaining['pnl_d0'] / R.R_DENOM).mean()) if len(remaining) else np.nan
    t_remaining = day_t_of(remaining[['date', 'pnl_d0']], 'pnl_d0') if len(remaining) else np.nan
    return dict(n_total=len(df), n_excluded=len(excluded), n_remaining=len(remaining),
                mean_R_remaining=mean_r_remaining, day_t_remaining=t_remaining,
                total_usd_remaining=float(remaining['pnl_d0'].sum()),
                total_usd_baseline=float(df['pnl_d0'].sum()),
                usd_given_up=float(excluded['pnl_d0'].sum()))


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    log('loading population.csv + results.csv (delay 0 and delay 5, 1,426 frozen outputs)')
    m = load_merged()
    n_total = len(m)
    n_ticks = int(m['has_ticks'].sum())
    log(f'coverage: {n_ticks}/{n_total} fills have ticks ({n_ticks/n_total:.1%})')

    log('computing THROUGH-only t* (new pass over raw parquet trades)')
    through_map = compute_t_star_through(m)
    m['t_star_through'] = m.apply(
        lambda r: through_map.get((r.symbol, r.date), np.nan) if r.has_ticks else np.nan, axis=1)
    m['bucket_through'] = m['t_star_through'].apply(bucket_of)

    ticks = m[m.has_ticks].copy()
    no_ticks = m[~m.has_ticks].copy()

    oos = ticks[ticks.period.isin(['2023-24', '2025H2-2026-09'])]
    h1 = ticks[ticks.period == '2025H1']
    may = ticks[(ticks.date_dt >= '2026-05-01') & (ticks.date_dt <= '2026-09-30')]
    periods = [('out_of_sample', oos), ('2025H1', h1), ('May-Sep_2026', may)]

    tables_at = {name: bucket_table(df, 'pnl_d0', 'bucket_at') for name, df in periods}
    tables_through = {name: bucket_table(df, 'pnl_d0', 'bucket_through') for name, df in periods}
    recon = [reconcile(name, df) for name, df in periods]
    d5 = {name: delay5_stats(df) for name, df in periods}
    skip = {name: skip_instant_stats(df) for name, df in periods}

    # coverage of uncovered fills' BT outcomes (population.csv's own columns)
    cov_rows = []
    for name, df in [('all', no_ticks)] + [(p, no_ticks[no_ticks.period == p]) if p != 'out_of_sample'
                                            else (p, no_ticks[no_ticks.period.isin(['2023-24', '2025H2-2026-09'])])
                                            for p in ['out_of_sample', '2025H1']]:
        if len(df) == 0:
            continue
        cov_rows.append(dict(group=name, n=len(df), win_pct=float((df['win'] == 1).mean()),
                              mean_bt_pnl=float(df['pnl'].mean())))
    cov_df = pd.DataFrame(cov_rows)

    # ---- frozen pass bar (verbatim from PREREG_1655.md) ----
    b0_5_oos = tables_at['out_of_sample'].set_index('bucket').loc['0-5']
    b0_5_h1 = tables_at['2025H1'].set_index('bucket').loc['0-5']
    cond_a = bool(b0_5_oos['mean_R'] < 0 and b0_5_oos['day_t'] <= -1.5 and b0_5_h1['mean_R'] < 0)
    d5_oos = d5['out_of_sample']
    cond_b = bool(d5_oos['mean_R'] >= d5_oos['baseline_mean_R'] - 0.005
                  and d5_oos['ex_top5pct_delta_R'] >= -0.01)
    ships = cond_a and cond_b

    log(f'0-5s bucket oos: mean_R={b0_5_oos.mean_R:.4f} t={b0_5_oos.day_t:.2f} n={b0_5_oos.n:.0f}; '
        f'2025H1: mean_R={b0_5_h1.mean_R:.4f} n={b0_5_h1.n:.0f}; cond_a={cond_a}')
    log(f'1656 oos: mean_R={d5_oos["mean_R"]:.4f} baseline={d5_oos["baseline_mean_R"]:.4f} '
        f'ex5_deltaR={d5_oos["ex_top5pct_delta_R"]:.4f}; cond_b={cond_b}')
    log(f'VERDICT ships={ships}')

    # ---- write buckets_1655.csv ----
    out_cols = m[['symbol', 'date', 'period', 't_star_at', 't_star_through', 'bucket_at', 'bucket_through',
                  'has_ticks', 'status_d0', 'pnl_d0', 'status_d5', 'pnl_d5']].copy()
    out_cols['R_delay0'] = out_cols['pnl_d0'] / R.R_DENOM
    out_cols['R_delay5'] = out_cols['pnl_d5'] / R.R_DENOM
    out_cols['skip_d0'] = out_cols['status_d0'] == 'skipped_guard'
    out_cols['skip_d5'] = out_cols['status_d5'] == 'skipped_guard'
    out_cols = out_cols[['symbol', 'date', 'period', 't_star_at', 't_star_through', 'R_delay0', 'R_delay5',
                          'skip_d0', 'skip_d5', 'bucket_at', 'bucket_through', 'has_ticks',
                          'status_d0', 'status_d5']]
    out_cols.to_csv(BUCKETS_CSV, index=False)
    log(f'WROTE {BUCKETS_CSV} rows={len(out_cols)}')

    # ---- write RESULT_1655.md ----
    with open(RESULT_MD, 'w') as f:
        f.write('# RESULT — cells 1,655-1,657: the first five seconds after 09:35:00 ET\n\n')
        f.write('Report-only + one pre-committed decision (1,656 ships to paper only if the frozen pass bar '
                'passes). PREREG_1655.md frozen 2026-09-29 08:20 UTC. t* = AT-or-through (price >= trigger, '
                'research/orb_latency_bt/replay.py::find_t_star) unless labeled THROUGH-only. delay-0/delay-5 '
                'status and pnl are 1,426\'s own results.csv rows, not recomputed.\n\n')
        f.write('## Coverage\n')
        f.write(f'- fills with ticks / total: {n_ticks}/{n_total} ({n_ticks/n_total:.1%})\n')
        if len(cov_df):
            f.write('- uncovered fills\' BT outcome (population.csv, not replayed):\n\n')
            f.write(R.df_to_md(cov_df, float_cols=['win_pct', 'mean_bt_pnl']) + '\n\n')
        else:
            f.write('- no uncovered fills\n\n')

        f.write('## 1,655 — bucket table at delay 0 (t* AT-or-through)\n\n')
        for name, _ in periods:
            f.write(f'### {name}\n\n')
            f.write(R.df_to_md(tables_at[name], float_cols=['mean_R', 'day_t', 'win_pct', 'ex_top5pct_R', 'total_usd']) + '\n\n')

        f.write('## 1,655 — THROUGH-only t* variant beside it (0-5 / 5-15 buckets, same pnl)\n\n')
        cmp_rows = []
        for name, _ in periods:
            a5 = tables_at[name].set_index('bucket').loc['0-5']
            th5 = tables_through[name].set_index('bucket').loc['0-5']
            a15 = tables_at[name].set_index('bucket').loc['5-15']
            th15 = tables_through[name].set_index('bucket').loc['5-15']
            cmp_rows.append(dict(period=name, n_0_5_AT=int(a5.n), meanR_0_5_AT=a5.mean_R,
                                  n_0_5_THROUGH=int(th5.n), meanR_0_5_THROUGH=th5.mean_R,
                                  n_5_15_AT=int(a15.n), n_5_15_THROUGH=int(th15.n)))
        f.write(R.df_to_md(pd.DataFrame(cmp_rows), float_cols=['meanR_0_5_AT', 'meanR_0_5_THROUGH']) + '\n\n')

        f.write('## Reconciliation with 1,426\'s published delay-0 totals (pass bar: same n, total $ within $1)\n\n')
        f.write(R.df_to_md(pd.DataFrame(recon), float_cols=['total_mine', 'total_1426', 'delta_usd']) + '\n\n')
        all_match = all(r['match'] for r in recon)
        f.write(f'**Reconciliation: {"PASS -- all three periods match" if all_match else "MISMATCH -- see deltas above"}.**\n\n')

        f.write('## 1,656 — DELAY-5 (buy-stops armed at 09:35:05.000)\n\n')
        d5_rows = []
        for name, _ in periods:
            s = d5[name]
            d5_rows.append(dict(period=name, n=s['n'], n_filled=s['n_filled'],
                                 n_skipped_guard=s['n_skipped_guard'], n_skipped_guard_d0=s['n_skipped_guard_d0'],
                                 total_usd=s['total_usd'], mean_R=s['mean_R'], baseline_mean_R=s['baseline_mean_R'],
                                 day_t=s['day_t'], mean_delta_R=s['mean_delta_R'],
                                 ex_top5pct_delta_R=s['ex_top5pct_delta_R']))
        f.write(R.df_to_md(pd.DataFrame(d5_rows),
                            float_cols=['total_usd', 'mean_R', 'baseline_mean_R', 'day_t', 'mean_delta_R',
                                        'ex_top5pct_delta_R']) + '\n\n')

        f.write('## 1,657 — SKIP-INSTANT (t* < 5s excluded, no refill; report-only)\n\n')
        sk_rows = []
        for name, _ in periods:
            s = skip[name]
            sk_rows.append(dict(period=name, n_total=s['n_total'], n_excluded=s['n_excluded'],
                                 n_remaining=s['n_remaining'], mean_R_remaining=s['mean_R_remaining'],
                                 day_t_remaining=s['day_t_remaining'], total_usd_remaining=s['total_usd_remaining'],
                                 total_usd_baseline=s['total_usd_baseline'], usd_given_up=s['usd_given_up']))
        f.write(R.df_to_md(pd.DataFrame(sk_rows),
                            float_cols=['mean_R_remaining', 'day_t_remaining', 'total_usd_remaining',
                                        'total_usd_baseline', 'usd_given_up']) + '\n\n')

        f.write('## Verdict — frozen pass bar for `preplace_submit_delay_s: 5`\n\n')
        f.write(f'- Condition A (0-5s bucket negative, out-of-sample mean_R={b0_5_oos.mean_R:.4f} t={b0_5_oos.day_t:.2f} '
                f'n={b0_5_oos.n:.0f}; 2025H1 mean_R={b0_5_h1.mean_R:.4f} n={b0_5_h1.n:.0f}): '
                f'**{"PASS" if cond_a else "FAIL"}**\n')
        f.write(f'- Condition B (1,656 mean_R={d5_oos["mean_R"]:.4f} vs baseline-0.005={d5_oos["baseline_mean_R"]-0.005:.4f}, '
                f'ex-top5% ΔR={d5_oos["ex_top5pct_delta_R"]:.4f} vs -0.01 floor): **{"PASS" if cond_b else "FAIL"}**\n\n')
        if ships:
            f.write('**VERDICT: `preplace_submit_delay_s: 5` SHIPS to the paper session** (both conditions of the '
                    'frozen pass bar pass on the out-of-sample book).\n')
        else:
            f.write('**VERDICT: `preplace_submit_delay_s: 5` DOES NOT SHIP.** ')
            if not cond_a:
                f.write('The 0-5s bucket is not negative (or not significant / not same-signed on 2025H1) in the '
                        'backtest population -- per PREREG, the live -$24 on 42 fills is noise and nothing changes. ')
            if cond_a and not cond_b:
                f.write('The 0-5s bucket IS negative, but 1,656\'s delay-5 book does not clear the no-cost-to-tail-'
                        'winners bar. ')
            f.write('1,657 (SKIP-INSTANT) is report-only per PREREG and does not change this verdict.\n')
    log(f'WROTE {RESULT_MD}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
