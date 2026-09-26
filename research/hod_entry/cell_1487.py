#!/usr/bin/env python3
"""Cells 1,487-1,488 -- research/hod_entry/PREREG_1487.md (FROZEN 2026-09-26).

CONFIRMATION ENTRY: on the 9,911 cell-1,438 fills (base book, status == 'fill', TRAIN-H2 / VAL;
TEST does not exist in these files), buy the break only after it has held 15 RTH minutes without
a $0.01 dip back below the level -- pricing the 13% no-withdrawal cohort honestly instead of at
the break, where 87% of fills withdraw and lose -0.35R (PREREG's disclosed numbers).

Rule 1,487 (confirmation entry): eligible fills are base fills whose OWN exit minute is > fill_min
+ 15 AND no bar in (fill_min, fill_min+15] has low <= level - 0.01. Enter LONG at the ask of the
open of minute fill_min+16 (open + the base fill's recovered half_entry, the PREREG's disclosed
proxy for the confirmation entry's own spread). stop = level - 0.01 (the level held 15 minutes).
R'' = entry - stop. target = entry + 2*R''. Walk on minute bars with sip_rebuild.walk_path
semantics from the entry bar (inclusive -- conservative: a dip on the entry bar itself stops it).
15:55 EOD. Costs: entry half-spread paid via the ask; exit cost by why -- target = limit (zero
extra cost), stop/stop_bar = the cell-1,478 amendment's expected-value bps (SLIP_STOP_BPS, per
split), eod/eod_fallback = cell 1,443's measured EOD holdout means (EOD_BPS). R'' < 0.5% of price
is reported but excluded from the PRIMARY book.

Rule 1,488 (pyramid): on the SAME entered (post-floor) 1,487 cohort, the original 1/3-risk fill
is already open past minute 15 (guaranteed by 1,487's own eligibility, since its base exit_m >
fill_min+15); add 2/3 at the SAME confirmation ask, move the whole position's stop to level -
0.01, target = the ORIGINAL fill + 2 * the ORIGINAL R. Re-walked from the entry bar with this new
stop/target (same bars, since the target differs from 1,487's the exit can differ too). Booked in
ORIGINAL-R units (both legs divided by R_orig = base fill - base stop), paired against the base
book's own outcome_R on the same fills.

Independent-check note (disclosed, not silent): PREREG_1487 names `cell_1445_features.csv` as the
half_entry source ("inspect the header"); that CSV's write_features_csv() (cell_1445.py) drops
half_entry -- it carries only day/symbol/fill_min/split/flags/net_R_corr[_flat30]. half_entry is
instead recovered via the IDENTICAL formula (cell_1445.corrected_cost, called inside
cell_1478.build_outcome -> cell_1457.build_base_cost), the same recovery the PREREG's own text
describes. Flagged in RESULT_1487.md caveats.

Usage:
    nice -n 19 python3 research/hod_entry/cell_1487.py [--dry-run]

Outputs: cell_1487_fills.csv, cell_1488_fills.csv, RESULT_1487.md.
"""
import argparse
import os
import sqlite3
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from research.hod_entry import cell_1445 as c1445            # noqa: E402 -- base book, day_clustered_t, ex_top5_mean
from research.hod_entry import cell_1478 as c1478            # noqa: E402 -- load_base_fills, build_outcome, SLIP_STOP_BPS
from research.hod_entry import causal_arming as ca            # noqa: E402 -- _rth (UTC->ET RTH minute)
from research.hod_entry import sip_rebuild as sr              # noqa: E402 -- walk_path, EOD_M, ET
from research.hod_consol import run_consol as consol          # noqa: E402 -- simulate_slots

BARS_DB = os.path.join(HERE, 'bars_fills_1478.db')
CAUSAL_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
FEATURES_A = os.path.join(HERE, 'features_1478_A.csv')

WAIT_M = 15                # minutes the level must hold, measured from the fill bar
ENTRY_OFFSET_M = 16        # confirmation entry = fill bar's minute (floor) + 16
DIP_TICK = 0.01
TARGET_R_MULT = 2.0
RFLOOR_PCT = 0.005         # R'' (or R_orig) must be >= 0.5% of price -- desk's own obtainability rule

# Cell 1,478 amendment's expected-value stop-limit slip (bps), per split -- reused unchanged.
SLIP_STOP_BPS = c1478.SLIP_STOP_BPS
# Cell 1,443 (RESULT_1443.md) EOD holdout MEANS (bps), keyed like SLIP_STOP_BPS by the raw `split`
# value ('TRAIN' == the TRAIN-H2 holdout, 'VAL').
EOD_BPS = {'TRAIN': 11.5, 'VAL': 9.7}

NULL_SEED = 1487
NULL_DRAWS = 1000

# Pass bar (frozen, PREREG_1487.md) -- 1,487 (VAL primary book) and 1,488 (paired).
PASS_1487_MEAN, PASS_1487_T = 0.15, 2.5
PASS_1487_FILLS_WK = 3.0
PASS_1487_NULL_PCTILE = 99.0
PASS_1487_TRAINH2_T = 1.0
CACHEONLY_BASE, CACHEONLY_TOL = 0.195, 0.05
PASS_1488_DELTA = 0.05
PASS_1488_VAL_T = 2.5
PASS_1488_VAL_MEAN = 0.10


def log(msg):
    """Verbose progress, flushed immediately (nohup-safe)."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ================================================================================================
# Data loading
# ================================================================================================

def load_base():
    """Base book with the cell-1,478 outcome standard (outcome_R) and half_entry recovered via
    the SAME formula the PREREG names (cell_1445.corrected_cost, via cell_1457.build_base_cost).
    Adds `holdout` (TRAIN-H2 / VAL) and `wk` (trading-week label, for fills/week denominators)."""
    base = c1478.load_base_fills()
    base = c1478.build_outcome(base)
    base['holdout'] = np.where(base['split'] == 'VAL', 'VAL', 'TRAIN-H2')
    wk_map = pd.read_csv(CAUSAL_CSV, usecols=['day', 'wk'], low_memory=False).drop_duplicates('day')
    base = base.merge(wk_map, on='day', how='left')
    n_no_wk = int(base['wk'].isna().sum())
    if n_no_wk:
        log(f'WARNING: {n_no_wk} base rows have no wk match -- excluded from week denominators')
    log(f'load_base: {len(base)} base fills, holdout counts {base.holdout.value_counts().to_dict()}, '
        f'outcome_R mean={base.outcome_R.mean():.4f}')
    return base


def load_bars(symbol_days):
    """{(symbol, day): RTH minute bar frame (m,o,h,l,c,v)} from bars_fills_1478.db, batched one
    SQL query per day (mirrors cell_1445.load_sip_bars, pointed at the 1,478 SIP store)."""
    con = sqlite3.connect(f'file:{BARS_DB}?mode=ro', uri=True)
    by_day = {}
    for sym, day in symbol_days:
        by_day.setdefault(day, set()).add(sym)
    out = {}
    days_sorted = sorted(by_day)
    for di, day in enumerate(days_sorted):
        syms = sorted(by_day[day])
        q = (f"select symbol, t, o, h, l, c, v from bars where day=? and symbol in "
             f"({','.join('?' * len(syms))})")
        g = pd.read_sql(q, con, params=[day] + syms)
        for s, gg in g.groupby('symbol'):
            out[(s, day)] = ca._rth(gg, 't')
        if di % 50 == 0 or di == len(days_sorted) - 1:
            log(f'load_bars: day {di + 1}/{len(days_sorted)} ({day}), {len(out)} symbol-days loaded')
    con.close()
    return out


def load_cacheonly_flags():
    """{(day, symbol): store_served_1438} from features_1478_A.csv (the decoy column used for the
    'kept cache-only share' check)."""
    f = pd.read_csv(FEATURES_A, usecols=['day', 'symbol', 'store_served_1438'], low_memory=False)
    dup = f.duplicated(subset=['day', 'symbol']).sum()
    if dup:
        log(f'WARNING: {dup} duplicate (day,symbol) rows in features_1478_A.csv -- keeping first')
        f = f.drop_duplicates(subset=['day', 'symbol'], keep='first')
    return f.set_index(['day', 'symbol'])['store_served_1438'].to_dict()


# ================================================================================================
# Cell 1,487 -- confirmation entry
# ================================================================================================

def eligibility_and_entry(base, bars_by_sd):
    """Per base fill: eligibility (no dip in (fill_min, fill_min+15], base exit_m > fill_min+15),
    then -- for eligible rows with a resolvable entry bar -- the confirmation trade's walk and R''.
    Returns one row per base fill (eligible or not), so the caller can build every reported share
    off ONE frame."""
    rows = []
    n_no_bars, n_no_window_bars, n_no_entry_bar, n_bad_r2 = 0, 0, 0, 0
    for r in base.itertuples():
        bars = bars_by_sd.get((r.symbol, r.day))
        rec = dict(day=r.day, symbol=r.symbol, split=r.split, holdout=r.holdout, wk=r.wk,
                   fill_min=r.fill_min, base_exit_m=r.exit_m, base_why=r.why,
                   base_outcome_R=r.outcome_R, level=r.level)
        if bars is None or not len(bars):
            n_no_bars += 1
            rec.update(eligible=False, reason='no_bars')
            rows.append(rec)
            continue
        win = bars[(bars.m > r.fill_min) & (bars.m <= r.fill_min + WAIT_M)]
        if not len(win):
            n_no_window_bars += 1
        dip = bool((win.l <= r.level - DIP_TICK).any()) if len(win) else False
        eligible = bool((r.exit_m > r.fill_min + WAIT_M) and not dip)
        rec['eligible'] = eligible
        if not eligible:
            rec['reason'] = 'dip_in_window' if dip else 'base_exited_by_15'
            rows.append(rec)
            continue

        entry_bar_m = int(np.floor(r.fill_min)) + ENTRY_OFFSET_M
        eb = bars[bars.m == entry_bar_m]
        if not len(eb):
            n_no_entry_bar += 1
            rec.update(reason='no_entry_bar')
            rows.append(rec)
            continue
        entry = float(eb.o.iloc[0]) + float(r.half_entry)
        stop = r.level - DIP_TICK
        R2 = entry - stop
        if not (R2 > 0):
            n_bad_r2 += 1
            rec.update(reason='nonpositive_R2')
            rows.append(rec)
            continue
        target = entry + TARGET_R_MULT * R2
        path = bars[bars.m >= entry_bar_m].sort_values('m')
        exit_m2, exit_price2, why2 = sr.walk_path(entry, stop, target, path)
        raw_R2 = (exit_price2 - entry) / R2
        if why2 == 'target':
            cost_R2 = 0.0
        elif why2 in ('stop', 'stop_bar'):
            cost_R2 = exit_price2 * SLIP_STOP_BPS[r.split] / 1e4 / R2
        else:                                                     # eod, eod_fallback
            cost_R2 = exit_price2 * EOD_BPS[r.split] / 1e4 / R2
        net_R2 = raw_R2 - cost_R2
        rec.update(reason='entered', entry_bar_m=entry_bar_m, entry=entry, stop2=stop, R2=R2,
                   target=target, exit_m2=exit_m2, exit_price2=exit_price2, why2=why2,
                   raw_R2=raw_R2, cost_R2=cost_R2, net_R2=net_R2,
                   r2_pct_price=100.0 * R2 / entry, below_floor=(R2 / entry) < RFLOOR_PCT,
                   entry_m=r.fill_min + ENTRY_OFFSET_M)
        rows.append(rec)
    log(f'eligibility_and_entry: {len(rows)} base fills scored; data-loss counts: '
        f'no_bars={n_no_bars} no_window_bars={n_no_window_bars} no_entry_bar={n_no_entry_bar} '
        f'nonpositive_R2={n_bad_r2}')
    return pd.DataFrame(rows)


def count_matched_null(base, cohort_h, holdout, seed=NULL_SEED, n_draws=NULL_DRAWS):
    """1,000 day-stratified draws (same n per day as the observed ENTERED cohort) from the base
    book's own outcome_R on those SAME days, in that SAME holdout (cell 1,481's design, reused
    verbatim)."""
    if not len(cohort_h):
        return dict(null_pctile=np.nan, n_null=0)
    n_by_day = cohort_h.groupby('day').size()
    base_h = base[base.holdout == holdout]
    by_day = {d: g.outcome_R.to_numpy() for d, g in base_h.groupby('day')}
    rng = np.random.default_rng(seed)
    obs_mean = float(cohort_h.net_R2.mean())
    draws = []
    for _ in range(n_draws):
        vals = []
        for day, n in n_by_day.items():
            pool = by_day.get(day)
            if pool is None or not len(pool):
                continue
            vals.append(rng.choice(pool, size=int(n), replace=len(pool) < n))
        if vals:
            draws.append(float(np.concatenate(vals).mean()))
    draws = np.array(draws)
    pctile = float((draws < obs_mean).mean() * 100.0) if len(draws) else np.nan
    return dict(null_pctile=pctile, n_null=len(draws))


def slot_fills_wk(cohort_h, weeks):
    """Slot-capped fills/week (research/hod_consol/run_consol.simulate_slots, 4 concurrent / 12
    daily default caps) -- guards against reporting raw counts that a real book could never take
    concurrently."""
    if not len(cohort_h) or weeks == 0:
        return float('nan')
    trades = cohort_h.rename(columns={'entry_m': 'entry_m', 'exit_m2': 'exit_m'})[['day', 'entry_m', 'exit_m']]
    keep = consol.simulate_slots(trades)
    return float(keep.sum()) / weeks


def score_1487(scored, base, holdout, cacheonly_map):
    """One holdout's PRIMARY (R'' >= 0.5% floor) and ALL-eligible (floor included) books."""
    d = scored[scored.holdout == holdout]
    total = len(d)
    weeks = base[base.holdout == holdout].wk.nunique()
    eligible = d[d.eligible]
    eligible_share = len(eligible) / total if total else np.nan
    entered = d[d.reason == 'entered']
    runners_lost = int(((base.holdout == holdout) & (base.why == 'target') &
                         (base.exit_m <= base.fill_min + WAIT_M)).sum())
    runners_lost_share = runners_lost / total if total else np.nan
    calib_mean = float(eligible.base_outcome_R.mean()) if len(eligible) else np.nan
    cacheonly_base = base[base.holdout == holdout].apply(
        lambda r: cacheonly_map.get((r.day, r.symbol)), axis=1)

    out = {}
    # below_floor is object-dtype (NaN on non-entered rows elsewhere in the parent frame before
    # this filter) -- cast explicitly so `~` is a boolean NOT, never Python's integer bit-invert.
    floor_mask = entered.below_floor.astype(bool) if len(entered) else pd.Series(dtype=bool)
    for label, book in (('primary', entered[~floor_mask]), ('all_eligible', entered)):
        n = len(book)
        mean_net = float(book.net_R2.mean()) if n else np.nan
        t = c1445.day_clustered_t(book.net_R2, book.day) if n > 1 else np.nan
        extop5 = float(c1445.ex_top5_mean(book.net_R2)) if n else np.nan
        fwk = slot_fills_wk(book, weeks)
        nul = count_matched_null(base, book, holdout)
        paired_base_mean = float(book.base_outcome_R.mean()) if n else np.nan
        keep_flags = book.apply(lambda r: cacheonly_map.get((r.day, r.symbol)), axis=1) if n else pd.Series(dtype=float)
        cacheonly_share = float(pd.Series(keep_flags).mean()) if n else np.nan
        r_pct_median = float(book.r2_pct_price.median()) if n else np.nan
        out[label] = dict(n=n, mean_net_R=mean_net, t=t, ex_top5=extop5, fills_wk=fwk,
                           null_pctile=nul['null_pctile'], eligible_share=eligible_share,
                           runners_lost_share=runners_lost_share, paired_base_mean=paired_base_mean,
                           calibration_base_mean_on_cohort=calib_mean, r_pct_median=r_pct_median,
                           kept_cacheonly_share=cacheonly_share)
    return out


# ================================================================================================
# Cell 1,488 -- pyramid keyed on no-withdrawal
# ================================================================================================

def build_1488(scored, base):
    """On the 1,487 PRIMARY (post-floor) entered cohort: add 2/3 at the SAME confirmation ask,
    move the whole position's stop to level-0.01, target = ORIGINAL fill + 2*ORIGINAL R, re-walked
    from the entry bar (same bars, new target). Booked in ORIGINAL-R units."""
    is_entered = scored.reason == 'entered'
    floor_mask = scored.below_floor.astype(bool)          # NaN elsewhere; only compared where is_entered
    cohort = scored[is_entered & ~floor_mask].copy()
    base_idx = base.set_index(['day', 'symbol'])
    bars_needed = list(zip(cohort.symbol, cohort.day))
    bars_by_sd = load_bars(bars_needed)

    rows = []
    for r in cohort.itertuples():
        b = base_idx.loc[(r.day, r.symbol)]
        fill_orig, stop_orig, R_orig = float(b.fill), float(b.stop), float(b.R)
        target_1488 = fill_orig + TARGET_R_MULT * R_orig
        stop_1488 = r.stop2                                        # level - 0.01, same as 1,487
        bars = bars_by_sd.get((r.symbol, r.day))
        path = bars[bars.m >= r.entry_bar_m].sort_values('m')
        exit_m3, exit_price3, why3 = sr.walk_path(r.entry, stop_1488, target_1488, path)
        leg1_raw = (exit_price3 - fill_orig) / R_orig
        leg2_raw = (exit_price3 - r.entry) / R_orig
        raw_1488 = (1.0 / 3.0) * leg1_raw + (2.0 / 3.0) * leg2_raw
        if why3 == 'target':
            cost_1488 = 0.0
        elif why3 in ('stop', 'stop_bar'):
            cost_1488 = exit_price3 * SLIP_STOP_BPS[r.split] / 1e4 / R_orig
        else:
            cost_1488 = exit_price3 * EOD_BPS[r.split] / 1e4 / R_orig
        net_1488 = raw_1488 - cost_1488
        rows.append(dict(day=r.day, symbol=r.symbol, split=r.split, holdout=r.holdout,
                          fill_min=r.fill_min, exit_m3=exit_m3, exit_price3=exit_price3,
                          why3=why3, net_1488=net_1488, base_outcome_R=r.base_outcome_R,
                          delta=net_1488 - r.base_outcome_R))
    log(f'build_1488: {len(rows)} pyramid trades booked on the 1,487 primary cohort')
    return pd.DataFrame(rows)


def score_1488(df1488, base, holdout):
    d = df1488[df1488.holdout == holdout]
    n = len(d)
    total = len(base[base.holdout == holdout])
    mean_net = float(d.net_1488.mean()) if n else np.nan
    t = c1445.day_clustered_t(d.net_1488, d.day) if n > 1 else np.nan
    paired_t = c1445.day_clustered_t(d.delta, d.day) if n > 1 else np.nan
    paired_delta = float(d.delta.mean()) if n else np.nan
    return dict(n=n, share_added=n / total if total else np.nan, mean_net_R=mean_net, t=t,
                paired_delta=paired_delta, paired_delta_t=paired_t,
                paired_base_mean=float(d.base_outcome_R.mean()) if n else np.nan)


# ================================================================================================
# Report
# ================================================================================================

def evaluate_pass_1487(sc):
    val, tr = sc['VAL']['primary'], sc['TRAIN-H2']['primary']
    cacheonly_ok = (not np.isnan(val['kept_cacheonly_share']) and
                    abs(val['kept_cacheonly_share'] - CACHEONLY_BASE) <= CACHEONLY_TOL)
    val_pass = (val['mean_net_R'] >= PASS_1487_MEAN and val['t'] >= PASS_1487_T and
                val['ex_top5'] > 0 and val['fills_wk'] >= PASS_1487_FILLS_WK and
                val['null_pctile'] >= PASS_1487_NULL_PCTILE and
                np.sign(tr['mean_net_R']) == np.sign(val['mean_net_R']) and
                tr['t'] >= PASS_1487_TRAINH2_T and cacheonly_ok)
    tr_pass = tr['t'] >= PASS_1487_TRAINH2_T and np.sign(tr['mean_net_R']) == np.sign(val['mean_net_R'])
    return dict(val=bool(val_pass), trainh2=bool(tr_pass))


def evaluate_pass_1488(sc):
    val, tr = sc['VAL'], sc['TRAIN-H2']
    val_pass = (val['paired_delta'] >= PASS_1488_DELTA and tr['paired_delta'] >= PASS_1488_DELTA and
                val['paired_delta_t'] >= PASS_1488_VAL_T and val['mean_net_R'] >= PASS_1488_VAL_MEAN)
    tr_pass = tr['paired_delta'] >= PASS_1488_DELTA
    return dict(val=bool(val_pass), trainh2=bool(tr_pass))


def fmt_table(rows, cols):
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c)
            vals.append(f'{v:.4f}' if isinstance(v, float) else str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def write_result_md(sc1487, verdict1487, sc1488, verdict1488, caveats, path):
    lines = ['# RESULT -- cells 1,487-1,488: confirmation entry + no-withdrawal pyramid', '']
    lines.append('## Cell 1,487 -- confirmation entry (R\'\' = entry - stop)')
    vkey = {'TRAIN-H2': 'trainh2', 'VAL': 'val'}
    rows = []
    for h in ('TRAIN-H2', 'VAL'):
        for label in ('primary', 'all_eligible'):
            r = dict(sc1487[h][label])
            r['cell'] = '1487'
            r['holdout'] = h
            r['book'] = label
            r['passes_bar'] = verdict1487[vkey[h]]
            rows.append(r)
    cols = ['cell', 'holdout', 'book', 'n', 'eligible_share', 'r_pct_median', 'runners_lost_share',
            'mean_net_R', 't', 'ex_top5', 'fills_wk', 'null_pctile', 'paired_base_mean',
            'calibration_base_mean_on_cohort', 'kept_cacheonly_share', 'passes_bar']
    lines.append(fmt_table(rows, cols))
    lines.append('')
    lines.append('## Cell 1,488 -- pyramid keyed on no-withdrawal (original-R units, paired vs base)')
    rows2 = []
    for h in ('TRAIN-H2', 'VAL'):
        r = dict(sc1488[h])
        r['cell'] = '1488'
        r['holdout'] = h
        r['passes_bar'] = verdict1488[vkey[h]]
        rows2.append(r)
    cols2 = ['cell', 'holdout', 'n', 'share_added', 'mean_net_R', 't', 'paired_delta',
             'paired_delta_t', 'paired_base_mean', 'passes_bar']
    lines.append(fmt_table(rows2, cols2))
    lines.append('')
    lines.append('## Caveats')
    for c in caveats:
        lines.append(f'* {c}')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log(f'wrote {path}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true', help='300-row sample, seed 1487')
    args = ap.parse_args(argv)

    t0 = time.time()
    base = load_base()
    if args.dry_run:
        base = base.sample(n=min(300, len(base)), random_state=1487).reset_index(drop=True)
        log(f'--dry-run: subsampled to {len(base)} rows')

    log('loading minute bars (bars_fills_1478.db)')
    bars_by_sd = load_bars(list(zip(base.symbol, base.day)))

    log('scoring eligibility + confirmation entry walk')
    scored = eligibility_and_entry(base, bars_by_sd)
    cacheonly_map = load_cacheonly_flags()

    sc1487 = {h: score_1487(scored, base, h, cacheonly_map) for h in ('TRAIN-H2', 'VAL')}
    verdict1487 = evaluate_pass_1487(sc1487)
    log(f'1,487 VAL primary: n={sc1487["VAL"]["primary"]["n"]} '
        f'mean_net_R={sc1487["VAL"]["primary"]["mean_net_R"]:.4f} '
        f't={sc1487["VAL"]["primary"]["t"]:.2f} passes={verdict1487["val"]}')

    log('building cell 1,488 (pyramid)')
    df1488 = build_1488(scored, base)
    sc1488 = {h: score_1488(df1488, base, h) for h in ('TRAIN-H2', 'VAL')}
    verdict1488 = evaluate_pass_1488(sc1488)
    log(f'1,488 VAL: n={sc1488["VAL"]["n"]} paired_delta={sc1488["VAL"]["paired_delta"]:.4f} '
        f'passes={verdict1488["val"]}')

    scored.to_csv(os.path.join(HERE, 'cell_1487_fills.csv'), index=False)
    df1488.to_csv(os.path.join(HERE, 'cell_1488_fills.csv'), index=False)

    n_no_bars = int((scored.reason == 'no_bars').sum())
    n_no_entry_bar = int((scored.reason == 'no_entry_bar').sum())
    n_bad_r2 = int((scored.reason == 'nonpositive_R2').sum())
    caveats = [
        "half_entry source: PREREG names cell_1445_features.csv, whose write_features_csv() "
        "(cell_1445.py) drops half_entry -- it is recovered here via the IDENTICAL formula "
        "(cell_1445.corrected_cost via cell_1478.build_outcome), not read from that CSV. Flag for "
        "the independent rebuild.",
        f"data loss: no_bars={n_no_bars}, no_entry_bar={n_no_entry_bar} (halt/gap at minute "
        f"fill_min+16), nonpositive_R2={n_bad_r2} -- excluded from eligible/entered counts, not "
        "silently dropped (see cell_1487_fills.csv `reason`).",
        "1,488's 'same fills' = the 1,487 PRIMARY (post R''-floor) cohort, not the pre-floor "
        "eligible set -- a reading choice, since the PREREG does not separately floor-gate 1,488.",
        "EOD_BPS (11.5/9.7 bps) and SLIP_STOP_BPS are holdout-level EXPECTED VALUES (1,443/1,478), "
        "not this new leg's own measured tape -- same disclosed-proxy status as half_entry.",
        "fills_wk is slot-capped (research/hod_consol.simulate_slots, 4 concurrent/12 daily "
        "default caps), not a raw count/week ratio.",
        f"runtime {time.time() - t0:.0f}s.",
    ]
    write_result_md(sc1487, verdict1487, sc1488, verdict1488, caveats,
                     os.path.join(HERE, 'RESULT_1487.md'))
    log('done')
    return sc1487, verdict1487, sc1488, verdict1488


if __name__ == '__main__':
    main()
