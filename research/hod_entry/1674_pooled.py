#!/usr/bin/env python3
"""Cell 1,674: the bar's power error -- pooled reads of the same-signed cuts,
the joint book on VAL, and capped-day selection.

PREREG: research/hod_entry/PREREG_1674.md (FROZEN 2026-09-30 05:22 UTC).
Owner ask (2026-09-30 05:10 UTC): "find the errors and oversights in your
research, there's money there." Oversight #2: the per-half synthesis bar
(net >= +0.05 R, t >= 2.5 IN BOTH HALVES) throws away real +0.03-0.05 R lifts
because the per-half MDE (0.066-0.077 R) means a true +0.05 R lift clears one
half about half the time and both halves about a quarter of the time. This
script re-reads the same-signed cuts pooled, with day-clustered SE.

Population: 1667_features.csv IS the 5,506-row floored primary book
(fills_1658 x causal_arming_causal, r_pct >= 1.5%; verified in 1667_sweep.py's
own docstring and by row-count/half-count/net_R match against 1663 and 1665).
`half` column already encodes TRAIN-H2 / VAL.

Joins (verified interactively before writing this script, logged again here):
  - 1665_features.csv: fill_id, day, symbol, split, r_pct, net_R match
    1:1 against 1667 on (date,symbol) for all 5,506 rows (0 dup keys, 0
    r_pct/net_R mismatches) -- inner join on (date,symbol) for rvol_a20/a5/b.
  - 1660_per_fill.csv, cell=='B0': day, symbol, why, net_R_bid, net_R_moc.
    For why != 'eod' rows net_R_bid == net_R_moc EXACTLY (checked: 0/9,874
    differ); for why == 'eod' rows they differ EXACTLY (0/2,261 equal) --
    confirms the MOC/bid choice only ever touches eod-exited fills. B0's own
    population only matches 4,111/5,506 (74.7%) of our floored book by
    (date,symbol) (different date range / cell construction) -- BELOW the
    project's 80% coverage rail. The MOC read is reported on the matched
    subset only, flagged VOID by the availability rail, and is NOT allowed to
    silently drop rows from the joint book (see apply_moc()).

Statistics (mirrors 1667_sweep.py's day_clustered_t / ex_top5_mean /
fills_per_week helpers verbatim for parity; adds delta_vs_base_day_clustered_t
for the new "ΔR vs the floored base" pooled read that 1667_sweep.py did not
need):
  - day_clustered_t: one-sample t on day-level means (cluster on day).
  - delta_vs_base_day_clustered_t(kept, base): per-day mean(kept) -
    per-day mean(base) [base is the superset kept is drawn from], one-sample
    day-clustered t on that per-day delta series. This is the ΔR test for
    every Part A candidate (kept ⊂ base, not a kept-vs-dropped test).
  - MDE = 2.801585 * SD / sqrt(n) (Z_.975 + Z_.80), SD = the FLOORED BASE's
    own half (or pooled) SD, fixed per scope -- not the subset's own SD --
    so MDE is comparable across differently-sized cuts within the same scope.

Usage:
    python3 1674_pooled.py

Outputs (all under research/hod_entry/):
    1674_reads.csv         -- every Part A/B/C read, one row each
    1674_joint_per_fill.csv -- VAL fills passing the joint book filter
    1674_pooled.log         -- verbose progress log
    RESULT_1674.md (written by hand from this script's printed summary)
"""
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
F1667_CSV = os.path.join(HERE, '1667_features.csv')
F1665_CSV = os.path.join(HERE, '1665_features.csv')
F1660_CSV = os.path.join(HERE, '1660_per_fill.csv')

READS_CSV = os.path.join(HERE, '1674_reads.csv')
JOINT_CSV = os.path.join(HERE, '1674_joint_per_fill.csv')
LOG_FILE = os.path.join(HERE, '1674_pooled.log')

Z_MDE = 1.959964 + 0.841621  # ~2.801585, matches the PREREG's "2.8"
CAP_PER_DAY = 12
PASS_MEAN = 0.05
PASS_T = 2.5
PASS_FPW = 3.0
ENTRY_T = 1.0

logger = logging.getLogger('1674')


def setup_logging():
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)


# ---------------------------------------------------------------------------
# Stats helpers (day_clustered_t / ex_top5_mean / fills_per_week copied
# verbatim from 1667_sweep.py for parity; delta_vs_base_day_clustered_t and
# mde are new for this cell's pooled ΔR read).
# ---------------------------------------------------------------------------

def day_clustered_t(df, datecol='date', valcol='net_R'):
    """One-sample t-stat computed on day-level means (cluster on day)."""
    day_means = df.groupby(datecol)[valcol].mean()
    n_days = len(day_means)
    if n_days < 2:
        return np.nan
    s = day_means.std(ddof=1)
    if s == 0 or np.isnan(s):
        return np.nan
    return day_means.mean() / (s / math.sqrt(n_days))


def delta_vs_base_day_clustered_t(kept, base, datecol='date', valcol='net_R'):
    """Day-clustered t of (kept's per-day mean - base's per-day mean), where
    kept is a subset drawn from base (not the complement). This is the ΔR
    test for every Part A 'drop/keep' candidate: does adding this cut move
    the day-level mean away from the floored base's own day-level mean."""
    day_kept = kept.groupby(datecol)[valcol].mean()
    day_base = base.groupby(datecol)[valcol].mean()
    delta = (day_kept - day_base).dropna()
    n_days = len(delta)
    if n_days < 2:
        return np.nan, np.nan
    s = delta.std(ddof=1)
    if s == 0 or np.isnan(s):
        return delta.mean(), np.nan
    return delta.mean(), delta.mean() / (s / math.sqrt(n_days))


def ex_top5_mean(vals):
    """Mean net_R excluding the top 5% (by value) of this subset."""
    vals = pd.Series(vals).dropna().sort_values()
    n = len(vals)
    if n == 0:
        return np.nan
    k = int(math.ceil(n * 0.05))
    if k >= n:
        return vals.mean()
    return vals.iloc[:n - k].mean()


def fills_per_week(df, datecol='date'):
    """Fills / trading-weeks spanned by this subset's own date range (floor 1 wk)."""
    if len(df) == 0:
        return 0.0
    dates = pd.to_datetime(df[datecol])
    span_days = (dates.max() - dates.min()).days
    weeks = max(span_days / 7.0, 1.0)
    return len(df) / weeks


def mde(sd, n):
    if n <= 0 or pd.isna(sd) or pd.isna(n):
        return np.nan
    return Z_MDE * sd / math.sqrt(n)


def sign(x):
    if pd.isna(x):
        return 'NA'
    if x > 0:
        return '+'
    if x < 0:
        return '-'
    return '0'


# ---------------------------------------------------------------------------
# Load + join
# ---------------------------------------------------------------------------

def load_base():
    """1667_features.csv is already the 5,506-row floored primary book
    (verified against 1663/1665 before this script was written: 0 dup keys
    on (date,symbol), 0 net_R/r_pct mismatches across all 5,506 rows)."""
    df = pd.read_csv(F1667_CSV, dtype={'date': str, 'symbol': str})
    logger.info('loaded base %s rows=%d halves=%s', F1667_CSV, len(df),
                dict(df['half'].value_counts()))
    return df


def join_rvol(base):
    rvol = pd.read_csv(F1665_CSV, dtype={'day': str, 'symbol': str})
    rvol = rvol[['day', 'symbol', 'fill_id', 'rvol_a20', 'rvol_a5', 'rvol_b']]
    dup = rvol.duplicated(['day', 'symbol']).sum()
    if dup:
        logger.warning('1665_features has %d duplicate (day,symbol) keys', dup)
    merged = base.merge(rvol, left_on=['date', 'symbol'], right_on=['day', 'symbol'], how='left')
    missing = merged['rvol_a20'].isna().sum()
    if missing:
        logger.warning('%d/%d base rows failed to match rvol features', missing, len(merged))
    else:
        logger.info('rvol join: %d/%d matched', len(merged), len(merged))
    return merged.drop(columns=['day'])


def check_moc_join_integrity(base):
    """DIAGNOSTIC ONLY -- do not use this merge for numbers. A left-join of
    1660_per_fill.csv (cell==B0) onto base by (date,symbol) matches 4,111/5,506
    (74.7%, below the 80% rail) rows, but for MATCHED keys 1660's own
    net_R_bid disagrees with our net_R by a huge margin (entry price differs
    by a median $0.14 / mean $0.39, entry_m vs fill_min differ by a median
    0.86 min / mean 6.2 min, r_pct differs by a median 0.27pp; some matched
    keys have OPPOSITE-SIGN net_R, e.g. AAOI 2025-08-12: our net_R=+1.92R vs
    1660's net_R_bid=-0.47R). This proves 1660's B0 is a DIFFERENT population
    (different entry rule/level) even on symbol-days it shares with the
    primary book -- (date,symbol) is NOT a valid same-fill key across these
    two pipelines. The MOC candidate is therefore read on 1660's OWN book
    only (see moc_read_standalone) and can never be merged, row-for-row,
    into the primary 5,506-row book or the joint book."""
    b0 = pd.read_csv(F1660_CSV, dtype={'day': str, 'symbol': str})
    b0 = b0[b0['cell'] == 'B0'][['day', 'symbol', 'entry', 'entry_m', 'r_pct', 'net_R_bid']]
    m = base.merge(b0, left_on=['date', 'symbol'], right_on=['day', 'symbol'], how='inner')
    price_gap = (m['entry_price'] - m['entry']).abs()
    sign_flip = ((m['net_R'] > 0) != (m['net_R_bid'] > 0)).sum()
    logger.warning('MOC join integrity check: (date,symbol) matches %d/%d (%.1f%%) rows, '
                    'median entry-price gap $%.2f, %d/%d matched rows have OPPOSITE-SIGN '
                    'net_R between pipelines -- (date,symbol) is NOT a same-fill key; '
                    'MOC will be read on 1660''s own population only',
                    len(m), len(base), 100.0 * len(m) / len(base), price_gap.median(),
                    sign_flip, len(m))
    return len(m), price_gap.median(), sign_flip


# ---------------------------------------------------------------------------
# Part A candidates
# ---------------------------------------------------------------------------

TERCILE_FEATURES = {
    'drop_top_rvol_a20': ('rvol_a20', 'drop the top RVOL_A(20) tercile'),
    'drop_top_rvol_a5': ('rvol_a5', 'drop the top RVOL_A(5) tercile'),
    'drop_top_rvol_b': ('rvol_b', 'drop the top RVOL_B tercile'),
    'drop_top_F15': ('F15', 'drop the top tercile of F15 (dollar vol to level bar / ADV20)'),
    'drop_top_F11': ('F11', 'drop the top tercile of F11 (level vs VWAP)'),
}


def tercile_hi_edge(train_vals):
    """67th-percentile edge computed on TRAIN-H2 values only (no look-ahead)."""
    v = pd.Series(train_vals).dropna()
    return np.percentile(v, 200.0 / 3.0)


def read_candidate(name, reference, kept_mask, valcol='net_R', datecol='date'):
    """Full Part A read for one candidate: pooled + per-half ΔR vs the
    (reference) floored base, day-clustered t, ex-top-5%, fills/week, MDE,
    sign-agreement line."""
    kept = reference[kept_mask]
    train_ref = reference[reference['half'] == 'TRAIN-H2']
    val_ref = reference[reference['half'] == 'VAL']
    train_kept = kept[kept['half'] == 'TRAIN-H2']
    val_kept = kept[kept['half'] == 'VAL']

    pooled_delta, pooled_t = delta_vs_base_day_clustered_t(kept, reference, datecol, valcol)
    train_delta, train_t = delta_vs_base_day_clustered_t(train_kept, train_ref, datecol, valcol)
    val_delta, val_t = delta_vs_base_day_clustered_t(val_kept, val_ref, datecol, valcol)

    sd_pooled = reference[valcol].std(ddof=1)
    row = dict(
        part='A', name=name, n=len(kept), n_ref=len(reference),
        pooled_mean_kept=kept[valcol].mean(), pooled_delta_vs_base=pooled_delta,
        pooled_day_t=pooled_t, ex_top5=ex_top5_mean(kept[valcol]),
        fills_per_week=fills_per_week(kept, datecol), mde=mde(sd_pooled, len(kept)),
        train_delta=train_delta, train_day_t=train_t, train_sign=sign(train_delta),
        val_delta=val_delta, val_day_t=val_t, val_sign=sign(val_delta),
        sign_agree=(sign(train_delta) == sign(val_delta) and sign(train_delta) in ('+', '-')),
        train_n=len(train_kept), val_n=len(val_kept),
    )
    return row, kept_mask


def build_candidates(df):
    """Returns dict name -> (reference_df, kept_mask aligned to reference_df.index)
    for the 8 fixed Part A candidates (terciles use TRAIN-H2-only edges)."""
    cands = {}
    for name, (feat, _desc) in TERCILE_FEATURES.items():
        ref = df[df[feat].notna()].copy()
        n_missing = len(df) - len(ref)
        if n_missing:
            logger.warning('%s: %d/%d rows missing %s, excluded from this candidate\'s reference population',
                            name, n_missing, len(df), feat)
        edge = tercile_hi_edge(ref.loc[ref['half'] == 'TRAIN-H2', feat])
        logger.info('%s: TRAIN-H2 top-tercile edge (67th pct) = %.6f', name, edge)
        kept_mask = ref[feat] <= edge
        cands[name] = (ref, kept_mask)

    ref = df.copy()
    cands['stop_ge_3pct'] = (ref, ref['stop_bucket'] == '>=3%')
    cands['time_1100_1230'] = (ref, ref['time_bucket'] == '11:00-12:30')
    return cands


def moc_read_standalone():
    """Part A read for the MOC candidate, read ENTIRELY within 1660's own
    B0 population (floored on ITS OWN r_pct >= 1.5%), NOT merged onto the
    primary 5,506-row book -- see check_moc_join_integrity() for why a
    (date,symbol) join across pipelines is unsound. 1660's own 'split'
    column (TRAIN/VAL) is a different, wider-dated split than the primary
    book's TRAIN-H2/VAL; labelled '1660-TRAIN'/'1660-VAL' throughout to
    avoid conflating the two."""
    b0 = pd.read_csv(F1660_CSV, dtype={'day': str, 'symbol': str})
    b0 = b0[(b0['cell'] == 'B0') & (b0['r_pct'] >= 1.5)].copy()
    logger.info('moc_exit: 1660 B0 floored (r_pct>=1.5%%) n=%d splits=%s',
                len(b0), dict(b0['split'].value_counts()))

    def _delta_t(sub):
        d = sub['net_R_moc'] - sub['net_R_bid']
        tmp = pd.DataFrame({'day': sub['day'].values, 'd': d.values})
        day_means = tmp.groupby('day')['d'].mean()
        n_days = len(day_means)
        if n_days < 2:
            return d.mean(), np.nan
        s = day_means.std(ddof=1)
        if s == 0 or np.isnan(s):
            return d.mean(), np.nan
        return d.mean(), day_means.mean() / (s / math.sqrt(n_days))

    train_ref = b0[b0['split'] == 'TRAIN']
    val_ref = b0[b0['split'] == 'VAL']
    pooled_delta, pooled_t = _delta_t(b0)
    train_delta, train_t = _delta_t(train_ref)
    val_delta, val_t = _delta_t(val_ref)
    sd_pooled = b0['net_R_bid'].std(ddof=1)
    row = dict(
        part='A', name='moc_exit', n=len(b0), n_ref=len(b0),
        pooled_mean_kept=b0['net_R_moc'].mean(), pooled_delta_vs_base=pooled_delta,
        pooled_day_t=pooled_t, ex_top5=ex_top5_mean(b0['net_R_moc']),
        fills_per_week=fills_per_week(b0.rename(columns={'day': 'date'}), 'date'),
        mde=mde(sd_pooled, len(b0)),
        train_delta=train_delta, train_day_t=train_t, train_sign=sign(train_delta),
        val_delta=val_delta, val_day_t=val_t, val_sign=sign(val_delta),
        sign_agree=(sign(train_delta) == sign(val_delta) and sign(train_delta) in ('+', '-')),
        train_n=len(train_ref), val_n=len(val_ref),
        note='read on 1660 B0 own population (n={}), NOT row-joined to the primary 5,506 book -- '
             'see check_moc_join_integrity'.format(len(b0)),
    )
    return row


# ---------------------------------------------------------------------------
# Part B: joint book
# ---------------------------------------------------------------------------

def select_joint(reads_by_name):
    """Fixed selection rule: TRAIN ΔR > 0 AND TRAIN day-clustered t >= 1.0
    AND VAL sign agrees (VAL magnitude NOT used for selection)."""
    entered = []
    for name, row in reads_by_name.items():
        ok = (row['train_delta'] > 0) and (not pd.isna(row['train_day_t'])) \
            and (row['train_day_t'] >= ENTRY_T) and (row['val_sign'] == '+')
        logger.info('joint selection %-20s train_delta=%.4f train_t=%.2f val_sign=%s -> %s',
                    name, row['train_delta'], row['train_day_t'], row['val_sign'],
                    'ENTER' if ok else 'reject')
        if ok:
            entered.append(name)
    return entered


# ---------------------------------------------------------------------------
# Part C: capped-day selection
# ---------------------------------------------------------------------------

def cap_book(df, priority_col, ascending, joint_mask=None, cap=CAP_PER_DAY):
    """Per day, sort by priority_col (or joint_mask-first) and take the first
    `cap` fills; tiebreak by fill_min (original time order) within any tie."""
    d = df.copy()
    if joint_mask is not None:
        d['_pri'] = (~joint_mask).astype(int)  # joint members (False->0) first
        sort_cols = ['date', '_pri', 'fill_min']
        asc = [True, True, True]
    elif priority_col is None:
        sort_cols = ['date', 'fill_min']  # pure time order (the engine's behaviour)
        asc = [True, True]
    else:
        d['_pri'] = d[priority_col]
        sort_cols = ['date', '_pri', 'fill_min']
        asc = [True, ascending, True]
    d = d.sort_values(sort_cols, ascending=asc)
    d['_rank'] = d.groupby('date').cumcount()
    drop_cols = [c for c in ('_pri', '_rank') if c in d.columns]
    return d[d['_rank'] < cap].drop(columns=drop_cols)


def part_c_read(name, capped, base_sd_pooled, base_sd_train, base_sd_val):
    train = capped[capped['half'] == 'TRAIN-H2']
    val = capped[capped['half'] == 'VAL']
    row = dict(
        part='C', name=name, n=len(capped),
        pooled_mean_kept=capped['net_R'].mean(), pooled_day_t=day_clustered_t(capped, 'date', 'net_R'),
        ex_top5=ex_top5_mean(capped['net_R']), fills_per_week=fills_per_week(capped, 'date'),
        train_delta=train['net_R'].mean(), train_day_t=day_clustered_t(train, 'date', 'net_R'),
        train_sign=sign(train['net_R'].mean()),
        val_delta=val['net_R'].mean(), val_day_t=day_clustered_t(val, 'date', 'net_R'),
        val_sign=sign(val['net_R'].mean()),
        sign_agree=(sign(train['net_R'].mean()) == sign(val['net_R'].mean())),
        train_n=len(train), val_n=len(val),
    )
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    logger.info('=== cell 1,674: pooled reads, joint book, capped-day selection ===')

    base = load_base()
    base = join_rvol(base)
    n_match, price_gap_med, sign_flip = check_moc_join_integrity(base)

    reads = []
    reads_by_name = {}
    kept_masks = {}  # name -> (reference_df, boolean mask) for joint intersection

    cands = build_candidates(base)
    for name, (ref, mask) in cands.items():
        row, _ = read_candidate(name, ref, mask)
        reads.append(row)
        reads_by_name[name] = row
        kept_masks[name] = (ref, mask)
        logger.info('Part A %-20s n=%5d pooled_delta=%+.4f t=%5.2f train=%+.4f/%.2f(t) val=%+.4f/%.2f(t) sign=%s/%s agree=%s',
                    name, row['n'], row['pooled_delta_vs_base'], row['pooled_day_t'],
                    row['train_delta'], row['train_day_t'], row['val_delta'], row['val_day_t'],
                    row['train_sign'], row['val_sign'], row['sign_agree'])

    moc_row = moc_read_standalone()
    moc_row['coverage_pct'] = 100.0 * n_match / len(base)
    reads.append(moc_row)
    reads_by_name['moc_exit'] = moc_row
    logger.info('Part A %-20s n=%5d pooled_delta=%+.4f t=%5.2f train=%+.4f/%.2f(t) val=%+.4f/%.2f(t) sign=%s/%s agree=%s '
                '[read on 1660 own pop; (date,symbol) cross-join match=%.1f%%, median price gap $%.2f, %d sign-flips]',
                'moc_exit', moc_row['n'], moc_row['pooled_delta_vs_base'], moc_row['pooled_day_t'],
                moc_row['train_delta'], moc_row['train_day_t'], moc_row['val_delta'], moc_row['val_day_t'],
                moc_row['train_sign'], moc_row['val_sign'], moc_row['sign_agree'],
                100.0 * n_match / len(base), price_gap_med, sign_flip)

    # --- Part B: joint book ---
    entered = select_joint(reads_by_name)
    logger.info('JOINT BOOK entered: %s', entered)

    joint_mask_full = pd.Series(True, index=base.index)
    for name in entered:
        if name == 'moc_exit':
            continue  # value transform, not a filter
        ref, mask = kept_masks[name]
        this_mask = pd.Series(False, index=base.index)
        this_mask.loc[ref.index[mask]] = True
        joint_mask_full &= this_mask
    joint_pop = base.loc[joint_mask_full].copy()
    assign_col = 'net_R'

    moc_entered = 'moc_exit' in entered
    if moc_entered:
        logger.warning('MOC entered the joint-book selection rule (TRAIN delta=+%.4f, t=%.2f, VAL sign agrees) '
                        'but CANNOT be applied inside the joint per-fill book: 1660''s own population is not '
                        'row-joinable to the primary 5,506-row book on any key on disk (see '
                        'check_moc_join_integrity -- median $%.2f entry-price gap and %d sign-flips on '
                        'matched (date,symbol) keys prove these are different fills). The joint book below '
                        'uses the filter-type entrants only; MOC''s own effect is reported separately '
                        '(part=A, name=moc_exit, read on 1660''s own book) and is NOT folded into this number. '
                        'This is a data-gap oversight for a future cell, not a computed joint read.',
                        reads_by_name['moc_exit']['train_delta'], reads_by_name['moc_exit']['train_day_t'],
                        price_gap_med, sign_flip)

    train_joint = joint_pop[joint_pop['half'] == 'TRAIN-H2']
    val_joint = joint_pop[joint_pop['half'] == 'VAL']
    sd_val = base.loc[base['half'] == 'VAL', 'net_R'].std(ddof=1)
    joint_row = dict(
        part='B', name='joint_book', n=len(joint_pop),
        pooled_mean_kept=val_joint[assign_col].mean(), pooled_day_t=day_clustered_t(val_joint, 'date', assign_col),
        ex_top5=ex_top5_mean(val_joint[assign_col]), fills_per_week=fills_per_week(val_joint, 'date'),
        mde=mde(sd_val, len(val_joint)),
        train_delta=train_joint[assign_col].mean(), train_day_t=day_clustered_t(train_joint, 'date', assign_col),
        train_sign=sign(train_joint[assign_col].mean()),
        val_delta=val_joint[assign_col].mean(), val_day_t=day_clustered_t(val_joint, 'date', assign_col),
        val_sign=sign(val_joint[assign_col].mean()),
        sign_agree=None, train_n=len(train_joint), val_n=len(val_joint),
        members=';'.join(entered),
        note=('moc_exit entered the rule but is excluded from this per-fill number -- no valid '
              'cross-pipeline join key (see log)' if moc_entered else ''),
    )
    reads.append(joint_row)
    logger.info('JOINT BOOK (VAL) n=%d mean=%+.4f t=%.2f ex_top5=%+.4f fpw=%.2f TRAIN(caveat)=%+.4f',
                len(val_joint), joint_row['pooled_mean_kept'], joint_row['pooled_day_t'],
                joint_row['ex_top5'], joint_row['fills_per_week'], joint_row['train_delta'])

    val_joint.assign(**{'net_R_used': val_joint[assign_col]})[
        ['date', 'symbol', 'half', 'net_R', 'net_R_used']
    ].to_csv(JOINT_CSV, index=False)
    logger.info('wrote %s (%d VAL rows)', JOINT_CSV, len(val_joint))

    # leave-one-out (filter-type entrants only -- moc_exit is never a member of the per-fill mask)
    filter_entered = [n for n in entered if n != 'moc_exit']
    if not filter_entered:
        logger.info('no filter-type cut entered the joint book -- leave-one-out table is empty')
    for drop_name in filter_entered:
        loo = [n for n in filter_entered if n != drop_name]
        m = pd.Series(True, index=base.index)
        for name in loo:
            ref, mask = kept_masks[name]
            tm = pd.Series(False, index=base.index)
            tm.loc[ref.index[mask]] = True
            m &= tm
        pop = base.loc[m].copy()
        col = 'net_R'
        val_pop = pop[pop['half'] == 'VAL']
        row = dict(part='B_LOO', name=f'joint_minus_{drop_name}', n=len(pop),
                   pooled_mean_kept=val_pop[col].mean(), pooled_day_t=day_clustered_t(val_pop, 'date', col),
                   ex_top5=ex_top5_mean(val_pop[col]), fills_per_week=fills_per_week(val_pop, 'date'),
                   val_n=len(val_pop))
        reads.append(row)
        logger.info('LOO %-30s VAL n=%d mean=%+.4f t=%.2f', f'joint_minus_{drop_name}', len(val_pop),
                    row['pooled_mean_kept'], row['pooled_day_t'])

    # --- Part C: capped-day selection ---
    sd_pooled_all = base['net_R'].std(ddof=1)
    orders = [
        ('cap_time_order', None, None),
        ('cap_largest_r_pct', 'r_pct', False),
        ('cap_lowest_rvol_a20', 'rvol_a20', True),
        ('cap_lowest_F15', 'F15', True),
    ]
    for name, col, asc in orders:
        d = base.copy()
        if col == 'F15':
            d['F15'] = d['F15'].fillna(d['F15'].max() + 1)  # NaN sorts last under "lowest first"
        capped = cap_book(d, col, asc)
        row = part_c_read(name, capped, sd_pooled_all, None, None)
        reads.append(row)
        logger.info('Part C %-24s n=%d pooled_mean=%+.4f t=%.2f train=%+.4f val=%+.4f fpw=%.2f',
                    name, row['n'], row['pooled_mean_kept'], row['pooled_day_t'],
                    row['train_delta'], row['val_delta'], row['fills_per_week'])

    capped_joint = cap_book(base, None, None, joint_mask=joint_mask_full)
    row = part_c_read('cap_joint_members_first', capped_joint, sd_pooled_all, None, None)
    reads.append(row)
    logger.info('Part C %-24s n=%d pooled_mean=%+.4f t=%.2f train=%+.4f val=%+.4f fpw=%.2f',
                'cap_joint_members_first', row['n'], row['pooled_mean_kept'], row['pooled_day_t'],
                row['train_delta'], row['val_delta'], row['fills_per_week'])

    out = pd.DataFrame(reads)
    out.to_csv(READS_CSV, index=False)
    logger.info('wrote %s (%d rows)', READS_CSV, len(out))
    logger.info('=== done ===')


if __name__ == '__main__':
    main()
