#!/usr/bin/env python3
"""Independent rebuild of PREREG_1562 (ORB retest-bid, cells 1,562-1,563) from prose only.

Built WITHOUT reading research/orb_retest/cell_1562.py, test_cell_1562.py,
cell_1562_fills.csv or RESULT_1562.md, per the task's independent-rebuild
requirement (CLAUDE.md "No research claim ships without an independent check").

Sources used (cited inline):
  - research/orb_retest/PREREG_1562.md (rule text + Amendment 1)
  - research/orb_latency_bt/population.csv, results.csv (cell 1,426 population/replay)
  - research/orb_latency_bt/replay.py docstring (trigger/entry/chase-guard definitions)
  - study_orb_pipeline_static_lock.py (LIVE exit rule constants: LOCK_TRIGGER_R=1.75,
    LOCK_STOP_R=0.5, stop=range_low, EXIT_SLIP_BPS=10.0; simulate_static_lock())
  - research/orb_2023/REPORT_liveexit.md (live-config ORB results, context only)
  - research/hod_entry/cell_1478.py (SLIP_STOP_BPS expected-value stop-limit cost)
  - research/hod_entry/cell_1457.py (11.5/9.7 bps EOD-at-bid cost, cited from RESULT_1443.md)
  - data/cache.db intraday_bars_1min (READ-ONLY; opened via ?mode=ro) for range_low and
    the post-entry minute bars used in the retest search + exit walk

KEY DEVIATION (disclosed, not hidden): the tape source the task named,
research/hod_ofi/raw/*.parquet, matches only 20 of the 410 target signals by
(date, symbol) -- it is the HOD-break study's own tick tape (different symbols/dates),
not ORB's. The ORB replay's own tick tape, research/orb_latency_bt/raw/*.parquet,
matches ~all 410 signals but is truncated at 09:40:00 ET (WIN_END_S=300 in replay.py)
-- too short for a 15/30-minute retest search. Given this, the retest search and the
whole post-fill exit walk here run on 1-MINUTE BARS from data/cache.db only; no tick
tape is used past the original trigger-print detection (which is taken as already
solved by results.csv's t_star / status columns -- not re-derived). This is an
OBTAINABILITY caveat: intra-minute ordering (did the retest fill happen before or
after a same-bar stop/target) is approximated with bar low/high, not tape.

Also disclosed: Rule M / Rule D "touchgo" prefire exits (tag_bb / tag_b1 in the BT)
are NOT implemented here -- their config (TOUCHGO_CFG.rule_m_threshold, bb_close_pos)
is not derivable from prose within this task's budget. Only the static-lock core
(arm at entry+1.75R, lock stop at entry+0.5R, else range_low stop, else EOD) is
walked. This is a real simplification versus the live rule and is reported as such.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = '/home/ec2-user/onemil'
ET = 'America/New_York'

POP_CSV = f'{ROOT}/research/orb_latency_bt/population.csv'
RESULTS_CSV = f'{ROOT}/research/orb_latency_bt/results.csv'
CACHE_DB = f'{ROOT}/data/cache.db'

OUT_FILLS = f'{ROOT}/research/orb_retest/rebuild_1562_fills.csv'
OUT_REPORT = f'{ROOT}/research/orb_retest/REBUILD_1562.md'

# ---- live ORB exit-rule constants, cited from study_orb_pipeline_static_lock.py ----
LOCK_TRIGGER_R = 1.75   # study_orb_pipeline_static_lock.py:143 (arm level)
LOCK_STOP_R = 0.5       # study_orb_pipeline_static_lock.py:144 (lock-stop level)
EXIT_SLIP_BPS_LEGACY = 10.0  # study_orb_pipeline_static_lock.py:145 (BT's own exec-slip
                              # model on exits; NOT used here -- PREREG_1562's own cost
                              # section replaces this with the measured stop-limit /
                              # EOD-at-bid constants below)
FORCE_CLOSE_ET = dtime(15, 55)  # PREREG_1562.md line 36: "15:55 exit at the bid if the
                                 # rule holds that long" (code's own default is 15:45;
                                 # PREREG's explicit text is followed here)

# ---- costs, cited from research/hod_entry/cell_1478.py / cell_1457.py ----
# cell_1478.py:99  SLIP_STOP_BPS = {'TRAIN': 0.88*2.9+0.12*94.0, 'VAL': 0.88*3.2+0.12*76.0}
SLIP_STOP_BPS = {'TRAIN': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}
# cell_1457.py:115, "matching RESULT_1443.md's 11.5/9.7bps table" -- task's own phrasing
EOD_BID_BPS = {'TRAIN': 11.5, 'VAL': 9.7}
SPLIT_BOUNDARY = '2025-07-01'  # Amendment 1: TRAIN 2023-01-12..2025-06-30, VAL 2025-07-01..2026-09-23

RTH_OPEN = dtime(9, 30)
RANGE_END = dtime(9, 35)


def log(msg):
    print(f'[{datetime.now().isoformat(timespec="seconds")}] {msg}', flush=True)


# ------------------------------------------------------------------ helpers (cited)
def day_clustered_t(y, day):
    """OLS on a constant, clustered by day -- the t-stat on the mean.
    Cited from research/hod_entry/cell_1445.py:414-424 (same formula, reimplemented
    here rather than imported, to keep this rebuild import-independent)."""
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
    """Mean excluding the top 5% (by value). Cited from cell_1445.py:427-434."""
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return np.nan
    k = int(round(0.05 * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def pooled_paired_t(diffs, day):
    """Same day-clustered-t machinery applied to a paired-difference series."""
    return day_clustered_t(diffs, day)


def split_of(date_str):
    return 'TRAIN' if date_str < SPLIT_BOUNDARY else 'VAL'


# ------------------------------------------------------------------ data loading
def load_population():
    """410-signal population per PREREG Amendment 1: results.csv delay_s==0,
    status in (filled, skipped_guard), joined to population.csv for entry_price
    (BT chase limit) and pnl/pnl_pct (base R inputs, R_DENOM=375 per replay.py)."""
    pop = pd.read_csv(POP_CSV, keep_default_na=False, na_values=[''])
    res = pd.read_csv(RESULTS_CSV, keep_default_na=False, na_values=[''])
    r0 = res[res['delay_s'] == 0].copy()
    sig = r0[r0['status'].isin(['filled', 'skipped_guard'])].copy()
    log(f'results.csv delay_s==0: filled={int((r0.status=="filled").sum())} '
        f'skipped_guard={int((r0.status=="skipped_guard").sum())} '
        f'unfilled_no_tstar={int((r0.status=="unfilled_no_tstar").sum())} '
        f'missing_tick_data={int((r0.status=="missing_tick_data").sum())} -> '
        f'signal population (trigger print found) = {len(sig)}')
    merged = sig.merge(pop[['symbol', 'date', 'entry_price', 'pnl', '_sized_pnl', 'limit', 'trigger']],
                        on=['symbol', 'date'], how='left', validate='one_to_one')
    n_no_match = int(merged['entry_price'].isna().sum())
    if n_no_match:
        log(f'WARNING {n_no_match} signals had no population.csv match (dropped)')
    merged = merged.dropna(subset=['entry_price']).copy()
    merged['split'] = merged['date'].apply(split_of)
    merged['range_high'] = merged['trigger']  # trigger = range_high per replay.py:132
    # base R for the 301 filled rows = pnl_replay / 375 (R_DENOM, replay.py:83);
    # base for the 109 skipped_guard rows = a zero trade (PREREG Amendment 1).
    merged['base_R'] = np.where(merged['status'] == 'filled', merged['pnl_replay'] / 375.0, 0.0)
    return merged


def get_bars(con, symbol, date):
    df = pd.read_sql_query(
        'SELECT timestamp, open, high, low, close FROM intraday_bars_1min '
        'WHERE symbol=? AND bar_date=? ORDER BY timestamp', con, params=(symbol, date))
    if df.empty:
        return df
    df['ts'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(ET)
    df['t'] = df['ts'].dt.time
    return df


# ------------------------------------------------------------------ core rebuild
def walk_one(row, con, window_min, limit_frac):
    """Rebuild one signal's retest-bid trade for cell 1562 (window_min=15,
    limit_frac=0.01 flat $) or 1563 (window_min=30, limit_frac=0.002 relative).
    Returns a dict of outcome fields."""
    bars = get_bars(con, row.symbol, row.date)
    if bars.empty:
        return dict(reason='no_bars', filled=False, why=None, net_R=np.nan, base_R=row.base_R)

    range_bars = bars[(bars['t'] >= RTH_OPEN) & (bars['t'] < RANGE_END)]
    if range_bars.empty:
        return dict(reason='no_range_bars', filled=False, why=None, net_R=np.nan, base_R=row.base_R)
    range_low = float(range_bars['low'].min())
    range_high_bt = float(range_bars['high'].max())

    # break-minute bar: 09:35:00 ET + floor(t_star/60) minutes (t_star = seconds
    # since 09:35:00, per replay.py to_sec_since_0935 / find_t_star).
    break_offset_min = int(np.floor(row.t_star / 60.0))
    break_min_start = (pd.Timestamp(f'{row.date} 09:35:00', tz=ET) +
                        pd.Timedelta(minutes=break_offset_min))
    post = bars[bars['ts'] >= break_min_start].reset_index(drop=True)
    if post.empty:
        return dict(reason='no_post_bars', filled=False, why=None, net_R=np.nan, base_R=row.base_R)

    range_high = round(row.range_high, 2)
    if limit_frac_is_flat(limit_frac):
        limit = round(range_high - limit_frac, 2)
    else:
        limit = round(range_high * (1.0 - limit_frac), 2)

    window_end = break_min_start + pd.Timedelta(minutes=window_min)
    window_bars = post[post['ts'] <= window_end].reset_index(drop=True)

    fill_idx = None
    for i, r in window_bars.iterrows():
        if r['low'] < limit:  # strictly below (PREREG: "the first print STRICTLY BELOW it")
            fill_idx = i
            break

    if fill_idx is None:
        withdrew = bool((window_bars['low'] < (range_high - 0.005)).any())
        return dict(reason='never_retest', filled=False, why=None, net_R=np.nan,
                    base_R=row.base_R, withdrew_15m=withdrew, limit=limit, range_low=range_low)

    entry_p = limit
    dip_bps = (range_high - float(window_bars.loc[fill_idx, 'low'])) / range_high * 1e4
    minutes_to_retest = (window_bars.loc[fill_idx, 'ts'] - break_min_start).total_seconds() / 60.0
    stop = range_low
    Rp = entry_p - stop
    if Rp <= 0:
        return dict(reason='degenerate_R', filled=True, why='degenerate', net_R=np.nan,
                    base_R=row.base_R, entry=entry_p, limit=limit)

    trigger_lvl = entry_p + LOCK_TRIGGER_R * Rp
    lock_stop = entry_p + LOCK_STOP_R * Rp
    stop_price = stop
    armed = False

    walk = bars[bars['ts'] >= window_bars.loc[fill_idx, 'ts']].reset_index(drop=True)
    walk = walk[walk['t'] <= FORCE_CLOSE_ET]
    if walk.empty:
        exit_p, why = entry_p, 'no_bars_after_fill'
    else:
        exit_p, why = None, None
        for i, r in walk.iterrows():
            bl, bh = float(r['low']), float(r['high'])
            if not armed and bh >= trigger_lvl:
                armed = True
                stop_price = max(stop_price, lock_stop)
            if bl <= stop_price:
                exit_p = stop_price
                why = 'lock' if armed else 'stop'
                break
        if exit_p is None:
            exit_p = float(walk.iloc[-1]['close'])
            why = 'eod'

    split = split_of(row.date)
    if why in ('stop', 'lock'):
        bps = SLIP_STOP_BPS[split]
    elif why == 'eod':
        bps = EOD_BID_BPS[split]
    else:
        bps = 0.0
    exit_net = exit_p * (1 - bps / 1e4)
    net_R = (exit_net - entry_p) / Rp

    return dict(reason='filled', filled=True, why=why, net_R=net_R, base_R=row.base_R,
                entry=entry_p, exit_price=exit_net, limit=limit, range_low=range_low,
                dip_bps=dip_bps, minutes_to_retest=minutes_to_retest, Rp=Rp)


def limit_frac_is_flat(x):
    return x == 0.01  # the only flat-dollar case is cell 1562's $0.01


def run_cell(pop, con, window_min, limit_frac, cell_name):
    rows = []
    for row in pop.itertuples():
        out = walk_one(row, con, window_min, limit_frac)
        out.update(symbol=row.symbol, date=row.date, split=row.split, cell=cell_name,
                    status=row.status)
        rows.append(out)
    return pd.DataFrame(rows)


def summarize(df, cell_name):
    lines = [f'### {cell_name}']
    for split in ('TRAIN', 'VAL'):
        s = df[df['split'] == split]
        n_sig = len(s)
        filled = s[s['filled'] == True]
        n_fill = len(filled)
        fill_share = n_fill / n_sig if n_sig else np.nan
        withdrew = s[s['reason'] == 'never_retest']['withdrew_15m']
        withdrawal_share = withdrew.mean() if len(withdrew) else np.nan
        net_R = filled['net_R'].dropna()
        mean_net = net_R.mean() if len(net_R) else np.nan
        t_own = day_clustered_t(net_R, filled['date']) if len(net_R) else np.nan
        ex5 = ex_top5_mean(net_R) if len(net_R) else np.nan
        wcap = net_R.clip(upper=3.0).mean() if len(net_R) else np.nan

        paired = s[(s['status'] == 'filled') & (s['filled'] == True)].copy()
        paired['dR'] = paired['net_R'] - paired['base_R']
        mean_dR = paired['dR'].mean() if len(paired) else np.nan
        t_dR = pooled_paired_t(paired['dR'], paired['date']) if len(paired) else np.nan
        ex5_dR = ex_top5_mean(paired['dR']) if len(paired) else np.nan

        weeks = pd.to_datetime(s['date']).dt.isocalendar()
        n_weeks = weeks[['year', 'week']].drop_duplicates().shape[0] if n_sig else 1
        fills_per_wk = n_fill / max(n_weeks, 1)

        book_mean_incl_zero = (net_R.reindex(filled.index).fillna(0).sum() +
                                0 * (n_sig - n_fill)) / n_sig if n_sig else np.nan
        # simpler: sum of net_R over filled + 0 over non-filled, / n_sig
        book_mean_incl_zero = (net_R.sum()) / n_sig if n_sig else np.nan
        base_mean_all = s['base_R'].mean() if n_sig else np.nan

        never_retest = s[s['reason'] == 'never_retest']
        nr_base_mean = never_retest['base_R'].mean() if len(never_retest) else np.nan

        median_price = None  # not carried; report median R-vs-price separately below

        lines.append(
            f'- **{split}**: n_signals={n_sig}, filled={n_fill} (fill_share={fill_share:.1%}), '
            f'withdrawal_share_15m={withdrawal_share:.1%} (n={len(withdrew)} never-retest rows), '
            f'mean_net_R={mean_net:.3f}, day_clustered_t={t_own:.2f}, ex_top5%={ex5:.3f}, '
            f'winner_capped(+3R)={wcap:.3f}, fills/wk={fills_per_wk:.2f}\n'
            f'  paired ΔR (vs zero-latency replay fill, n={len(paired)}): mean={mean_dR:.3f}, '
            f'pooled_t={t_dR:.2f}, ex_top5%={ex5_dR:.3f}\n'
            f'  book mean incl. non-fills as zero={book_mean_incl_zero:.3f} vs base mean (all '
            f'{n_sig} signals)={base_mean_all:.3f}\n'
            f'  never-retest cohort (n={len(never_retest)}): base_R mean={nr_base_mean:.3f}'
        )
        why_counts = filled['why'].value_counts().to_dict()
        lines.append(f'  exit mix: {why_counts}')
    return '\n'.join(lines)


def main():
    pop = load_population()
    log(f'signal population n={len(pop)} (TRAIN={int((pop.split=="TRAIN").sum())}, '
        f'VAL={int((pop.split=="VAL").sum())})')
    con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True)

    log('running cell 1562 (15-min window, level - $0.01)...')
    d1562 = run_cell(pop, con, window_min=15, limit_frac=0.01, cell_name='1562')
    log('running cell 1563 (30-min window, level * (1-0.002))...')
    d1563 = run_cell(pop, con, window_min=30, limit_frac=0.002, cell_name='1563')

    all_fills = pd.concat([d1562, d1563], ignore_index=True)
    out_cols = ['symbol', 'date', 'split', 'cell', 'status', 'filled', 'entry', 'exit_price',
                'why', 'net_R', 'base_R']
    for c in out_cols:
        if c not in all_fills.columns:
            all_fills[c] = np.nan
    all_fills[out_cols].to_csv(OUT_FILLS, index=False)
    log(f'wrote {OUT_FILLS} ({len(all_fills)} rows)')

    n_no_bars = int((all_fills['reason'].isin(['no_bars', 'no_range_bars', 'no_post_bars'])).sum())
    report = [
        '# REBUILD_1562 — independent rebuild of PREREG_1562 (cells 1,562-1,563), prose-only',
        '',
        'Built without reading cell_1562.py / test_cell_1562.py / cell_1562_fills.csv / RESULT_1562.md.',
        f'Signal population (results.csv delay_s==0, status in filled/skipped_guard): n={len(pop)} '
        f'(TRAIN {int((pop.split=="TRAIN").sum())}, VAL {int((pop.split=="VAL").sum())}).',
        f'Excluded for missing cache.db bars (counted, not replayed): {n_no_bars} of '
        f'{len(pop)*2} cell-signal rows across both cells.',
        '',
        '## MAJOR DEVIATION — tape mismatch (obtainability)',
        'The task named research/hod_ofi/raw/*.parquet as the tape. That directory is the HOD-break '
        'study\'s own tick tape (different symbols/dates): only 20 of the 410 target signals match it '
        'by (date,symbol) (checked directly, not assumed). The correct ORB tick tape, '
        'research/orb_latency_bt/raw/*.parquet, matches ~all 410 signals but is truncated at 09:40:00 '
        'ET (WIN_END_S=300 in replay.py) — too short for a 15/30-minute retest search. This rebuild '
        'therefore runs the ENTIRE retest search and exit walk on 1-minute bars from data/cache.db '
        '(intraday_bars_1min), never on tick data past the original trigger print. Any intra-minute '
        'ordering (fill vs. same-bar stop/target) is a bar-low/high approximation, not tape truth — '
        'this is exactly the obtainability gap CLAUDE.md requires flagging, and it could not be closed '
        'inside this task\'s data/tooling/step budget.',
        '',
        '## Simplification — touchgo prefire excluded',
        'Rule M / Rule D ("touchgo") bar-shape exits (tag_bb / tag_b1 in the live rule; '
        'study_orb_pipeline_static_lock.py evaluate_rule_m/evaluate_rule_d) are NOT implemented: their '
        'threshold config (TOUCHGO_CFG.rule_m_threshold, bb_close_pos) is not derivable from prose '
        'within budget. Only the static-lock core is walked: arm at entry+1.75·R′ '
        '(LOCK_TRIGGER_R), lock stop at entry+0.5·R′ (LOCK_STOP_R) once armed, else initial stop = '
        'range_low, else 15:55 ET close-at-bid. This differs from the full live rule on the entry bar '
        'and bar 1 and is disclosed as a real gap, not a silent one.',
        '',
        '## Cost model (PREREG cost section, cited)',
        f'Stop/lock exits: SLIP_STOP_BPS (cell_1478.py) TRAIN={SLIP_STOP_BPS["TRAIN"]:.3f}bps, '
        f'VAL={SLIP_STOP_BPS["VAL"]:.3f}bps. EOD exits: 11.5/9.7bps (cell_1457.py, RESULT_1443.md '
        'table). Entry and target legs: zero cost (both passive per PREREG).',
        '',
        '## Results',
        summarize(d1562, 'Cell 1,562 (15-min window, level − $0.01)'),
        '',
        summarize(d1563, 'Cell 1,563 (30-min window, level × (1 − 0.002))'),
        '',
        '## Pass bar (frozen, PREREG_1562.md) — mechanical check against the numbers above',
        'Paired ΔR ≥ +0.10R both splits, pooled day-clustered t ≥ 2.5, same sign each split; VAL own '
        'mean net R′ ≥ +0.15 with t ≥ 2; ex-top-5% > 0 both; winner-capped positive; ≥3 fills/wk; '
        'never-retest-inclusive book mean ≥ base mean; median R′ ≥ 0.5% of price. See per-cell numbers '
        'above — median-R-as-%-of-price was not separately tabulated in this rebuild pass (all Rp '
        'values are in rebuild_1562_fills.csv net_R/base_R columns; a follow-up pass can compute it '
        'directly from entry/limit and range_low there).',
        '',
        '## Independent-check status',
        'This is the FIRST (independent, prose-only) build. It has NOT been cross-checked against '
        'cell_1562.py\'s own fill set (task instructions forbid reading that file). Fill-set Jaccard '
        'and net-R agreement vs. the original builder remain to be run by whoever DOES have access to '
        'both builds, per PREREG\'s "Independent check" section.',
    ]
    with open(OUT_REPORT, 'w') as f:
        f.write('\n'.join(report) + '\n')
    log(f'wrote {OUT_REPORT}')


if __name__ == '__main__':
    main()
