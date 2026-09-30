#!/usr/bin/env python3
"""Cell 1,673: short the predicted failure.

PREREG: research/hod_entry/PREREG_1673.md (FROZEN 2026-09-30 05:20 UTC).

Owner ask (2026-09-30 05:10 UTC): "find the errors and oversights in your
research, there's money there." Cells 1,668-1,670 established a stop-out is
predictable after entry (fast failure within 5 min at minute 1: AUC
0.76-0.81 oos; P(stop after k): AUC 0.63-0.67) and only ever asked whether to
CUT the long. This cell asks the direct question: is the same out-of-sample
P(stop) tradable as a SHORT on the same population?

Reuses (imported, not copied) from 1668_failure.py: BarStore, find_fill_index,
walk_k, iid_t, day_clustered_t, ex_top5_mean, mde, check_disk, EOD_M,
BARS_DB. Reuses (read, not re-fit) the out-of-sample P(stop) already scored
in 1669_per_fill.csv (pstop_FF5_1, pstop_FF10_1) and 1670_per_fill_k.csv
(p_stop_ALL at k=2,5,10). Builds only what those cells did not: the short's
own entry/exit mechanics and cost model.

Rule S(label, k, tau): at the close of bar fill+k, if the out-of-sample
P(stop) >= tau, sell short one unit at the open of bar fill+k+1. Base variant
target = the long's stop level; stop = short entry + 1R. Variant T2: target
2R below the long's entry; same stop. Variant H: stop = day's-high-so-far +
$0.01 instead of +1R. Exits are walked bar by bar on bars_sip.db starting at
the short's entry bar (EOD-time check first, then stop, then target -- same
precedence f1668.walk_k/walk_to_exit use for the long). SSR rail: a fill
whose short-entry price is >=10% below the prior session's close (daily
panel) is not shortable that day.

Usage:
    python3 1673_short.py [--limit N]

Outputs (research/hod_entry/):
    1673_per_short.csv -- one row per actual short taken (label,k,tau,variant,scoring)
    1673_reads.csv      -- one row per (label,k,tau,variant,scoring): short-only
                            stats + the paired portfolio-overlay stats
    1673_short.log
    RESULT_1673.md
"""
import argparse
import importlib.util
import logging
import math
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# --- reuse 1668_failure.py's tested machinery (module name starts with a
# digit, so importlib.util rather than a normal `import` statement). ---
_spec = importlib.util.spec_from_file_location('f1668', os.path.join(HERE, '1668_failure.py'))
f1668 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f1668)

PER_FILL_1669_CSV = os.path.join(HERE, '1669_per_fill.csv')
PER_FILL_K_1670_CSV = os.path.join(HERE, '1670_per_fill_k.csv')
PANEL_PARQUET = os.path.join(os.path.dirname(HERE), 'overnight_high', 'panel_2024_2026.parquet')
BARS_DB = f1668.BARS_DB

PER_SHORT_CSV = os.path.join(HERE, '1673_per_short.csv')
READS_CSV = os.path.join(HERE, '1673_reads.csv')
LOG_FILE = os.path.join(HERE, '1673_short.log')
RESULT_MD = os.path.join(HERE, 'RESULT_1673.md')

RULES = [('FF5', 1), ('FF10', 1), ('PSTOP', 2), ('PSTOP', 5), ('PSTOP', 10)]
TAUS = [0.6, 0.7, 0.8]
VARIANTS = ['base', 'T2', 'H']
SPLIT_TO_SCORING = {'VAL': 'TRAIN->VAL', 'TRAIN-H2': 'VAL->TRAIN-H2 (swap)'}

# Short costs (PREREG): entry 7bps (marketable sell short at next bar's open),
# cover 6bps (stop or target touch), EOD cover 11bps. Locate/borrow fee
# ignored on paper -- STATED here and in RESULT_1673.md.
ENTRY_BPS_SHORT = 0.0007
COVER_BPS = 0.0006
EOD_COVER_BPS = 0.0011
SSR_DROP = -0.10  # skip the short if entry is >=10% below the prior close

logger = logging.getLogger('1673')


def setup_logging():
    """Log to 1673_short.log and stdout, verbose (INFO), per project rules."""
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)


def load_inputs():
    """1669_per_fill.csv (population + FF5/FF10 P(stop) at k=1) joined to
    1670_per_fill_k.csv's P(stop after k) at k=2,5,10 (pstop_dict keyed by
    (fill_id,k)), plus the prior-session close from the overnight_high panel
    for the SSR rail. Nothing here is re-fit -- every probability is read."""
    pf = pd.read_csv(PER_FILL_1669_CSV, dtype={'date': str, 'symbol': str, 'split': str},
                      usecols=['fill_id', 'date', 'symbol', 'split', 'entry_price', 'stop',
                               'target_price', 'r_pct', 'fill_min', 'base_exit_type',
                               'base_net_R', 'pstop_FF5_1', 'pstop_FF10_1'])
    logger.info('1669_per_fill.csv: %d rows, split counts %s', len(pf), pf['split'].value_counts().to_dict())

    pk = pd.read_csv(PER_FILL_K_1670_CSV, usecols=['fill_id', 'k', 'p_stop_ALL'])
    pk = pk[pk['k'].isin([2, 5, 10])]
    pstop_dict = {(fid, k): p for fid, k, p in zip(pk['fill_id'], pk['k'], pk['p_stop_ALL'])}
    logger.info('1670_per_fill_k.csv: %d rows kept at k in {2,5,10}', len(pk))

    panel = pd.read_parquet(PANEL_PARQUET, columns=['symbol', 'bar_date', 'close'])
    panel = panel.sort_values(['symbol', 'bar_date'])
    panel['prior_close'] = panel.groupby('symbol', observed=True)['close'].shift(1)
    panel = panel.rename(columns={'bar_date': 'date'})[['symbol', 'date', 'prior_close']]
    before = len(pf)
    pf = pf.merge(panel, on=['symbol', 'date'], how='left', validate='many_to_one')
    n_missing_prior = pf['prior_close'].isna().sum()
    logger.info('prior-close join: %d/%d rows, %d missing prior close (treated as NOT SSR-restricted, logged)',
                before, len(pf), n_missing_prior)
    return pf, pstop_dict


def short_walk(bars, entry_idx, short_stop, short_target):
    """Bar-by-bar walk from the short's entry bar (inclusive) to its exit.
    Same precedence as f1668.walk_to_exit for the long: EOD-time check first,
    then stop, then target (stop wins a same-bar tie). Fill is assumed at the
    exact stop/target price (matches the codebase's existing convention for
    the long book -- no gap-through price capping; see RESULT's caveats for
    the measured share of exits where the bar's own open already gapped past
    the level, which this convention prices optimistically)."""
    n = len(bars['o'])
    for j in range(entry_idx, n):
        if bars['minarr'][j] >= f1668.EOD_M:
            return j, 'eod'
        if bars['h'][j] >= short_stop:
            return j, 'stop'
        if bars['l'][j] <= short_target:
            return j, 'target'
    return None, None


def variant_levels(variant, entry_price, stop, R_unit, short_entry_price, day_high_so_far):
    """Target/stop price levels for one of the three PREREG variants. Target
    and stop are always absolute prices; R is always reported in the long's
    units (dividing by the fixed R_unit), regardless of variant."""
    if variant == 'base':
        return stop, short_entry_price + R_unit
    if variant == 'T2':
        return entry_price - 2 * R_unit, short_entry_price + R_unit
    if variant == 'H':
        return stop, day_high_so_far + 0.01
    raise ValueError(variant)


def process(pf, pstop_dict, store, limit=None):
    """One pass over the population: for each fill, fetch its bars ONCE
    (per-(symbol,day) PK query, per project rule) and evaluate all 5 (label,k)
    rules x 3 tau x 3 variants from the same bars. Returns per_short rows and
    the eligible-fill records (fill_id,label,k,scoring,date) portfolio reads
    need to pad with zero for fills that were eligible but not shorted."""
    if limit:
        pf = pf.iloc[:limit].copy()
    per_short_rows = []
    eligible_rows = []
    n_no_bars = n_no_fill_idx = n_bad_R = 0
    n_no_prior_close = 0
    # eligible_n/fired_n/ssr_skip_n/no_entry_n are all variant-invariant (entry
    # mechanics don't depend on the exit-level variant); keyed precisely so
    # build_reads can report the PREREG's "SSR-skipped count" per rule, not
    # just a single run-wide total.
    eligible_n = defaultdict(int)      # key (label,k,scoring)
    fired_n = defaultdict(int)         # key (label,k,tau,scoring)
    ssr_skip_n = defaultdict(int)      # key (label,k,tau,scoring)
    no_entry_n = defaultdict(int)      # key (label,k,tau,scoring)
    n_day_ended_early = 0
    t0 = time.time()
    for i, r in enumerate(pf.itertuples(index=False)):
        bars = store.day_bars(r.symbol, r.date)
        if bars is None:
            n_no_bars += 1
            continue
        i0 = f1668.find_fill_index(bars, r.fill_min)
        if i0 is None:
            n_no_fill_idx += 1
            continue
        R_unit = r.entry_price - r.stop
        if not (R_unit > 0):
            n_bad_R += 1
            logger.error('fill_id=%s %s %s: R_unit<=0 (entry=%.4f stop=%.4f) -- skipped',
                          r.fill_id, r.symbol, r.date, r.entry_price, r.stop)
            continue
        prior_close = r.prior_close
        if pd.isna(prior_close):
            n_no_prior_close += 1
        scoring = SPLIT_TO_SCORING[r.split]

        for label, k in RULES:
            if label in ('FF5', 'FF10'):
                P = r.pstop_FF5_1 if label == 'FF5' else r.pstop_FF10_1
            else:
                P = pstop_dict.get((r.fill_id, k), np.nan)
            w = f1668.walk_k(bars, i0, k, r.stop, r.target_price)
            eligible = (w is not None) and (w['preempt'] == '') and pd.notna(P)
            if not eligible:
                continue
            eligible_n[(label, k, scoring)] += 1
            eligible_rows.append((r.fill_id, label, k, scoring, r.date))

            for tau in TAUS:
                fired = P >= tau
                if not fired:
                    continue
                key = (label, k, tau, scoring)
                fired_n[key] += 1
                if w['next_open'] is None:
                    no_entry_n[key] += 1
                    continue
                short_entry_idx = w['last_idx'] + 1
                short_entry_price = w['next_open']
                if pd.notna(prior_close) and prior_close > 0:
                    if (short_entry_price - prior_close) / prior_close <= SSR_DROP:
                        ssr_skip_n[key] += 1
                        continue
                day_high_so_far = bars['h'][:w['last_idx'] + 1].max()

                for variant in VARIANTS:
                    short_target, short_stop = variant_levels(
                        variant, r.entry_price, r.stop, R_unit, short_entry_price, day_high_so_far)
                    exit_idx, exit_kind = short_walk(bars, short_entry_idx, short_stop, short_target)
                    if exit_idx is None:
                        n_day_ended_early += 1
                        logger.warning('fill_id=%s %s %s label=%s k=%d variant=%s: day ended before EOD_M, '
                                       'forcing exit at the last bar', r.fill_id, r.symbol, r.date, label, k, variant)
                        exit_idx = len(bars['o']) - 1
                        exit_kind = 'eod'
                    cover_price = (bars['c'][exit_idx] if exit_kind == 'eod'
                                   else (short_stop if exit_kind == 'stop' else short_target))
                    exit_bps = EOD_COVER_BPS if exit_kind == 'eod' else COVER_BPS
                    gross_R = (short_entry_price - cover_price) / R_unit
                    cost_R = (ENTRY_BPS_SHORT * short_entry_price + exit_bps * cover_price) / R_unit
                    net_R = gross_R - cost_R
                    holding_min = float(bars['minarr'][exit_idx] - bars['minarr'][short_entry_idx])
                    gapped_through = bool(
                        (exit_kind == 'stop' and bars['o'][exit_idx] >= short_stop) or
                        (exit_kind == 'target' and bars['o'][exit_idx] <= short_target))
                    per_short_rows.append(dict(
                        fill_id=r.fill_id, date=r.date, symbol=r.symbol, split=r.split, scoring=scoring,
                        label=label, k=k, tau=tau, variant=variant, p_value=P, R_unit=R_unit,
                        short_entry_price=short_entry_price, short_stop=short_stop, short_target=short_target,
                        cover_price=cover_price, exit_kind=exit_kind, net_R=net_R,
                        holding_min=holding_min, gapped_through=gapped_through))

        if (i + 1) % 1000 == 0 or (i + 1) == len(pf):
            elapsed = time.time() - t0
            logger.info('processed %d/%d fills (%.1fs): eligible=%d fired=%d ssr_skip=%d no_entry=%d '
                        'day_ended_early=%d no_bars=%d no_fill_idx=%d',
                        i + 1, len(pf), elapsed, sum(eligible_n.values()), sum(fired_n.values()),
                        sum(ssr_skip_n.values()), sum(no_entry_n.values()), n_day_ended_early,
                        n_no_bars, n_no_fill_idx)
            per_short_df = pd.DataFrame(per_short_rows)
            tmp = PER_SHORT_CSV + '.tmp'
            per_short_df.to_csv(tmp, index=False)
            os.replace(tmp, PER_SHORT_CSV)

    logger.info('done: %d fills, no_bars=%d no_fill_idx=%d bad_R=%d no_prior_close=%d', len(pf),
                n_no_bars, n_no_fill_idx, n_bad_R, n_no_prior_close)
    logger.info('final totals: eligible=%d fired=%d ssr_skip=%d no_entry=%d day_ended_early=%d',
                sum(eligible_n.values()), sum(fired_n.values()), sum(ssr_skip_n.values()),
                sum(no_entry_n.values()), n_day_ended_early)
    counters = dict(eligible_n=dict(eligible_n), fired_n=dict(fired_n), ssr_skip_n=dict(ssr_skip_n),
                     no_entry_n=dict(no_entry_n), n_day_ended_early=n_day_ended_early)
    return (pd.DataFrame(per_short_rows), pd.DataFrame(eligible_rows,
            columns=['fill_id', 'label', 'k', 'scoring', 'date']), n_no_prior_close, counters)


def stats_block(vals, dates, worst_day_agg):
    """n, mean, iid t, day-clustered t, ex-top-5% mean, MDE, hit rate, worst
    day (aggregated per day by `worst_day_agg`, 'sum' for a short-only book of
    actual trades, 'mean' for a padded-with-zero paired-dR overlay read,
    matching 1670_timing_map.py's own R4 convention for that style of read)."""
    vals = pd.Series(vals, dtype=float)
    n = int(vals.notna().sum())
    if n == 0:
        return dict(n=0, mean_net_R=np.nan, iid_t=np.nan, day_t=np.nan, ex_top5_mean=np.nan,
                    mde=np.nan, hit_rate=np.nan, worst_day_R=np.nan)
    sd = vals.std(ddof=1) if n > 1 else np.nan
    grp = pd.DataFrame({'date': dates, 'v': vals}).groupby('date')['v']
    worst = (grp.sum().min() if worst_day_agg == 'sum' else grp.mean().min())
    return dict(n=n, mean_net_R=float(vals.mean()), iid_t=f1668.iid_t(vals),
                day_t=f1668.day_clustered_t(dates, vals), ex_top5_mean=f1668.ex_top5_mean(vals),
                mde=f1668.mde(sd, n), hit_rate=float((vals > 0).mean()), worst_day_R=float(worst))


def build_reads(per_short_df, eligible_df, pf, counters):
    """One row per (label,k,tau,variant,scoring): the short's own stats (over
    actual shorts only) plus the paired portfolio-overlay stats (over every
    eligible-at-k fill, padded with dR=0 where no short fired) -- 90 rows,
    each carrying both read types' full stat set. fired/ssr_skip/no_entry
    counts come from `process`'s own tallies (variant-invariant: entry
    mechanics don't depend on the exit-level variant), never re-derived."""
    fired_n, ssr_skip_n, no_entry_n = counters['fired_n'], counters['ssr_skip_n'], counters['no_entry_n']
    span_weeks = {}
    for half in ('VAL', 'TRAIN-H2'):
        d = pd.to_datetime(pf.loc[pf['split'] == half, 'date'])
        span_weeks[half] = max((d.max() - d.min()).days / 7.0, 1.0) if len(d) else np.nan
    scoring_to_split = {v: k for k, v in SPLIT_TO_SCORING.items()}

    rows = []
    for label, k in RULES:
        for scoring in SPLIT_TO_SCORING.values():
            elig = eligible_df[(eligible_df['label'] == label) & (eligible_df['k'] == k) &
                                (eligible_df['scoring'] == scoring)]
            n_eligible = len(elig)
            for tau in TAUS:
                for variant in VARIANTS:
                    sub = per_short_df[(per_short_df['label'] == label) & (per_short_df['k'] == k) &
                                        (per_short_df['tau'] == tau) & (per_short_df['variant'] == variant) &
                                        (per_short_df['scoring'] == scoring)]
                    short_stats = stats_block(sub['net_R'], sub['date'], 'sum')
                    n_shortable = short_stats['n']
                    weeks = span_weeks[scoring_to_split[scoring]]
                    shorts_per_week = n_shortable / weeks if weeks else np.nan

                    merged = elig.merge(sub[['fill_id', 'net_R']], on='fill_id', how='left')
                    merged['dR'] = merged['net_R'].fillna(0.0)
                    port_stats = stats_block(merged['dR'], merged['date'], 'mean')

                    key = (label, k, tau, scoring)
                    n_fired = fired_n.get(key, 0)
                    n_ssr_skip = ssr_skip_n.get(key, 0)
                    n_no_entry = no_entry_n.get(key, 0)
                    rows.append(dict(
                        label=label, k=k, tau=tau, variant=variant, scoring=scoring,
                        n_eligible=n_eligible, n_fired=n_fired, n_ssr_skip=n_ssr_skip, n_no_entry=n_no_entry,
                        n_shortable=n_shortable,
                        share_fired=(n_fired / n_eligible if n_eligible else np.nan),
                        share_shortable=(n_shortable / n_eligible if n_eligible else np.nan),
                        shorts_per_week=shorts_per_week, gapped_through_share=float(sub['gapped_through'].mean())
                        if n_shortable else np.nan,
                        short_mean_net_R=short_stats['mean_net_R'], short_iid_t=short_stats['iid_t'],
                        short_day_t=short_stats['day_t'], short_ex_top5_mean=short_stats['ex_top5_mean'],
                        short_mde=short_stats['mde'], short_hit_rate=short_stats['hit_rate'],
                        short_avg_holding_min=float(sub['holding_min'].mean()) if n_shortable else np.nan,
                        short_worst_day_R=short_stats['worst_day_R'],
                        portfolio_mean_dR=port_stats['mean_net_R'], portfolio_iid_t=port_stats['iid_t'],
                        portfolio_day_t=port_stats['day_t'], portfolio_ex_top5_mean=port_stats['ex_top5_mean'],
                        portfolio_mde=port_stats['mde'], portfolio_hit_rate=port_stats['hit_rate'],
                        portfolio_worst_day_R=port_stats['worst_day_R']))
    return pd.DataFrame(rows)


def evaluate_pass_bar(reads):
    """Per PREREG: short mean net R >= +0.10 R with day-clustered t >= 2.5 in
    BOTH scorings, ex-top-5% > 0 in both, >= 3 shorts/week in both scorings'
    own window. Placebo check is separately NOT AVAILABLE (see RESULT.md) --
    any 'pass' here is provisional pending that check and an independent
    rebuild, per the PREREG's own words."""
    out = []
    for (label, k, tau, variant), g in reads.groupby(['label', 'k', 'tau', 'variant']):
        if len(g) != 2:
            continue
        ok = bool((g['short_mean_net_R'] >= 0.10).all() and (g['short_day_t'] >= 2.5).all() and
                   (g['short_ex_top5_mean'] > 0).all() and (g['shorts_per_week'] >= 3).all())
        out.append(dict(label=label, k=k, tau=tau, variant=variant, passes_bar=ok,
                         mean_net_R_val=g.loc[g['scoring'] == 'TRAIN->VAL', 'short_mean_net_R'].iloc[0],
                         mean_net_R_trainh2=g.loc[g['scoring'] == 'VAL->TRAIN-H2 (swap)', 'short_mean_net_R'].iloc[0]))
    return pd.DataFrame(out)


def write_result_md(reads, pass_df, counters, n_no_prior_close, n_pop):
    """RESULT_1673.md exactly per the PREREG's Output section, <=120 lines."""
    L = []
    L.append('# RESULT 1,673 -- short the predicted failure')
    L.append('')
    n_ssr_total = sum(counters['ssr_skip_n'].values())
    n_no_entry_total = sum(counters['no_entry_n'].values())
    n_fired_total = sum(counters['fired_n'].values())
    L.append(f'Population: {n_pop} fills (1669/1670 join, primary r_pct>=1.5%, halves=split). '
             f'Fired (P>=tau) across all 5 rules x 3 tau x 2 scorings: {n_fired_total}. '
             f'SSR-skipped shorts (entry >=10% below prior close): {n_ssr_total}. '
             f'No-entry-bar (fired but no next bar to short into): {n_no_entry_total}. '
             f'Day ended before 15:55 ET (forced last-bar EOD): {counters["n_day_ended_early"]}. '
             f'Missing prior-close (SSR check skipped, treated as shortable): {n_no_prior_close}.')
    L.append('')
    L.append('Costs: short entry 7bps, cover 6bps (stop/target), EOD cover 11bps; '
             'locate/borrow fee ignored on paper -- STATED, not modeled. '
             'Fill convention: exact stop/target touch price (matches the existing long-book convention); '
             'gapped_through_share in 1673_per_short.csv/reads reports how often the exit bar\'s own open '
             'had already passed the level (that share is priced optimistically by this convention).')
    L.append('')
    L.append('Placebo: 1669/1670 did not save label-shuffled per-fill probabilities (only placebo AUC), '
             'so the placebo check is NOT AVAILABLE for this cell.')
    L.append('')
    L.append('## Pass bar (short mean net R >= +0.10R, day t >= 2.5, ex-top-5% > 0, >=3 shorts/wk, BOTH scorings)')
    n_pass = int(pass_df['passes_bar'].sum()) if len(pass_df) else 0
    L.append(f'{n_pass}/{len(pass_df)} (label,k,tau,variant) cells pass on both scorings (placebo not checked -- '
             'any pass below is provisional).')
    L.append('')
    if n_pass:
        L.append('| label | k | tau | variant | mean net R (VAL scoring) | mean net R (TRAIN-H2 scoring) |')
        L.append('|---|---|---|---|---|---|')
        for _, r in pass_df[pass_df['passes_bar']].iterrows():
            L.append(f"| {r['label']} | {r['k']} | {r['tau']} | {r['variant']} | "
                     f"{r['mean_net_R_val']:.3f} | {r['mean_net_R_trainh2']:.3f} |")
        L.append('')

    L.append('## Best cell per label (by min(day_t) across both scorings), short-only stats')
    reads['label_k'] = reads['label'] + '@' + reads['k'].astype(str)
    for lk, g in reads.groupby('label_k'):
        piv = g.pivot_table(index=['tau', 'variant'], columns='scoring',
                             values=['short_mean_net_R', 'short_day_t', 'short_hit_rate', 'n_shortable'])
        # rank by the worse (min) day_t across the two scorings
        day_t_cols = [c for c in piv.columns if c[0] == 'short_day_t']
        min_day_t = piv[day_t_cols].min(axis=1)
        best = min_day_t.idxmax()
        tau_b, var_b = best
        rows_b = g[(g['tau'] == tau_b) & (g['variant'] == var_b)]
        L.append(f'**{lk} best: tau={tau_b} variant={var_b}**')
        for _, r in rows_b.iterrows():
            L.append(f"  * {r['scoring']}: n={r['n_shortable']} mean_net_R={r['short_mean_net_R']:.3f} "
                     f"iid_t={r['short_iid_t']:.2f} day_t={r['short_day_t']:.2f} "
                     f"ex_top5={r['short_ex_top5_mean']:.3f} mde={r['short_mde']:.3f} "
                     f"hit_rate={r['short_hit_rate']:.3f} avg_hold_min={r['short_avg_holding_min']:.1f} "
                     f"worst_day_R={r['short_worst_day_R']:.3f} shorts/wk={r['shorts_per_week']:.2f} "
                     f"gapped_through={r['gapped_through_share']:.3f}")
    L.append('')

    L.append('## Portfolio read (paired dR of adding the short overlay to the base long book)')
    for lk, g in reads.groupby('label_k'):
        piv = g.pivot_table(index=['tau', 'variant'], columns='scoring', values='portfolio_day_t')
        day_t_cols = list(piv.columns)
        min_day_t = piv[day_t_cols].min(axis=1)
        best = min_day_t.idxmax()
        tau_b, var_b = best
        rows_b = g[(g['tau'] == tau_b) & (g['variant'] == var_b)]
        L.append(f'**{lk} best overlay: tau={tau_b} variant={var_b}**')
        for _, r in rows_b.iterrows():
            L.append(f"  * {r['scoring']}: portfolio_mean_dR={r['portfolio_mean_dR']:.4f} "
                     f"day_t={r['portfolio_day_t']:.2f} ex_top5={r['portfolio_ex_top5_mean']:.4f} "
                     f"worst_day_R={r['portfolio_worst_day_R']:.3f} n_eligible={r['n_eligible']}")
    L.append('')
    L.append('Full grid (all 90 (label,k,tau,variant,scoring) cells): `1673_reads.csv`. '
             'Row-level actual shorts: `1673_per_short.csv`.')
    L.append('')
    L.append('## Not allowed items honored')
    L.append('No re-fitting or re-scoring of any model (P(stop) values are read, not recomputed); '
             'tau/k/variant fixed by the PREREG before any number was seen; no bar after the decision bar '
             'used for the decision (entry/exit walk starts strictly after fill+k); SSR rail applied to every '
             'fired signal; both scorings reported throughout, never pooled-only.')
    text = '\n'.join(L)
    tmp = RESULT_MD + '.tmp'
    with open(tmp, 'w') as fh:
        fh.write(text + '\n')
    os.replace(tmp, RESULT_MD)
    n_lines = text.count('\n') + 1
    logger.info('RESULT_1673.md written: %d lines', n_lines)
    if n_lines > 120:
        logger.warning('RESULT_1673.md is %d lines, over the PREREG''s 120-line budget', n_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    setup_logging()
    f1668.check_disk(min_gb=5.0)
    logger.info('cell 1,673: short the predicted failure -- starting')

    pf, pstop_dict = load_inputs()
    store = f1668.BarStore(BARS_DB)
    per_short_df, eligible_df, n_no_prior_close, counters = process(pf, pstop_dict, store, limit=args.limit)
    store.close()

    per_short_df.to_csv(PER_SHORT_CSV + '.tmp', index=False)
    os.replace(PER_SHORT_CSV + '.tmp', PER_SHORT_CSV)
    logger.info('1673_per_short.csv: %d rows (actual shorts taken)', len(per_short_df))

    reads = build_reads(per_short_df, eligible_df, pf, counters)
    reads.to_csv(READS_CSV + '.tmp', index=False)
    os.replace(READS_CSV + '.tmp', READS_CSV)
    logger.info('1673_reads.csv: %d rows', len(reads))

    pass_df = evaluate_pass_bar(reads)
    write_result_md(reads, pass_df, counters, n_no_prior_close, len(pf))
    logger.info('cell 1,673 complete')


if __name__ == '__main__':
    main()
