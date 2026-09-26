#!/usr/bin/env python3
"""Cells 1,548-1,549 -- research/hod_entry/PREREG_1548.md (FROZEN 2026-09-26 18:50 UTC).

The extension anatomy and the sweep entry on the cell-1,478 L3-v2 kept top tercile (the model that
predicts, at the arm bar, whether the day's high after the fill will reach LEVEL x 1.05 -- VAL AUC
0.715, PREREG_1478 amendment 3). Part A measures the PATH of the extenders and non-extenders inside
the kept set (both holdouts, all rows and real-SIP rows): how deep price dips below the level before
the +5% touch (or before 15:55 for non-extenders), how long it takes, whether it breaches the
consolidation low first, whether the base trade was already stopped out. Part B fixes d (stop
distance), s (wider stop), W (resting window) from TRAIN-H2 quantiles of the EXTENDERS' path ONLY,
writes them BEFORE any VAL number is read, then scores two entries against those same d/s/W:
  * 1,548 SWEEP -- a resting buy limit at level x (1-d) for W minutes (through-print fill rule),
    stop level x (1-s), target level x 1.05, 15:55 EOD. Isolates the entry change (passive vs the
    base ask fill) together with the wider stop/target.
  * 1,549 WIDE  -- the base ask fill, same wider stop/target. Isolates JUST the stop/target change.
Both are also scored on the NON-kept fills (the dropped two terciles) as the calibration line -- the
model's lift must show as kept > dropped, not just "a wider stop helps everyone".

Every fallback (missing bars, missing half_entry, duplicate join keys, no touch found for a labeled
extender) is EXCLUDED and counted with a WARNING -- never silently imputed.

Usage:
    python3 research/hod_entry/cell_1548.py --dry-run     # smoke: fixed 200-fill sample, seed 1548
    python3 research/hod_entry/cell_1548.py                # full kept + dropped population

Outputs: research/hod_entry/cell_1548_fills.csv (one row per kept fill per cell: 1548/1549 status,
entry/exit/cost/net_R/net_pct), research/hod_entry/RESULT_1548.md (Part A anatomy tables, d/s/W,
Part B cell tables incl. the dropped-tercile calibration line, the pass-bar checklist, caveats).
"""
import argparse
import datetime as dt
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from research.hod_entry import cell_1445 as c1445           # noqa: E402  (day_clustered_t, ex_top5_mean,
                                                              # winner_capped_mean, weeks_spanned,
                                                              # fills_per_week, null_percentile_of)
from research.hod_entry.sip_rebuild import walk_path         # noqa: E402  (identical B0 path physics)

PRED_CSV = os.path.join(HERE, 'model_1478_L3_v2_predictions.csv')
CAUSAL_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
FEAT_CSV = os.path.join(HERE, 'features_1478_A.csv')
BARS_DB = os.path.join(HERE, 'bars_fills_1478.db')

OUT_FILLS_CSV = os.path.join(HERE, 'cell_1548_fills.csv')
OUT_RESULT_MD = os.path.join(HERE, 'RESULT_1548.md')
OUT_FILLS_CSV_DRY = os.path.join(HERE, 'cell_1548_fills_DRYRUN.csv')
OUT_RESULT_MD_DRY = os.path.join(HERE, 'RESULT_1548_DRYRUN.md')

ET = 'America/New_York'
OPEN_M, EOD_M = 570, 955               # 09:30, 15:55 ET, minutes-since-midnight -- sip_rebuild convention
TARGET_MULT = 1.05
# cell_1478.py SLIP_STOP_BPS verbatim (0.88 * filled-stop bps + 0.12 * no-fill-tail bps, per split)
SLIP_STOP_BPS = {'TRAIN': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}
EOD_BID_BPS = {'TRAIN': 11.5, 'VAL': 9.7}           # PREREG's "EOD at the bid"
STOP_WHY = {'stop', 'stop_bar'}
EOD_WHY = {'eod', 'eod_fallback'}
NULL_SEED = 1548
NULL_DRAWS = 1000
WINNER_CAP_R = 3.0
DAILY_CAP, CONCURRENT_CAP = 12, 4                    # live cap: 12/day, 4 concurrent
D_FLOOR_PCT = 0.20
W_CAP_MIN = 120
DRY_N = 200


def log(msg):
    """Timestamped, flushed progress line."""
    print(f'[{dt.datetime.now():%H:%M:%S}] {msg}', flush=True)


# ================================================================================================
# Load & join
# ================================================================================================

def load_population():
    """Kept-and-dropped fill population: predictions.csv (split, L3, hgb_kept_L3, outcome_R,
    store_served_1438) joined to causal_arming_causal.csv status=='fill' rows (level, stop=consol
    low, fill, fill_min, exit_m, why) on day+symbol, plus half_entry from features_1478_A.csv.
    Every join checked for uniqueness and completeness; misses excluded and counted."""
    pred = pd.read_csv(PRED_CSV, dtype={'day': str, 'symbol': str})
    causal = pd.read_csv(CAUSAL_CSV, dtype={'day': str, 'symbol': str}, low_memory=False)
    causal = causal[causal.status == 'fill'].copy()
    dup_c = int(causal.duplicated(subset=['day', 'symbol']).sum())
    if dup_c:
        log(f'WARNING: {dup_c} duplicate day+symbol rows in causal fill rows -- keeping first')
        causal = causal.drop_duplicates(subset=['day', 'symbol'])
    feat = pd.read_csv(FEAT_CSV, dtype={'day': str, 'symbol': str})[['day', 'symbol', 'half_entry']]
    dup_f = int(feat.duplicated(subset=['day', 'symbol']).sum())
    if dup_f:
        log(f'WARNING: {dup_f} duplicate day+symbol rows in features_1478_A -- keeping first')
        feat = feat.drop_duplicates(subset=['day', 'symbol'])

    df = pred.merge(
        causal[['day', 'symbol', 'fill', 'stop', 'level', 'fill_min', 'exit_m', 'why', 'exit_price']],
        on=['day', 'symbol'], how='left', suffixes=('', '_c'))
    n_miss = int(df['fill'].isna().sum())
    if n_miss:
        log(f'WARNING: {n_miss} prediction rows have no causal fill-row match -- excluded')
    df = df.dropna(subset=['fill', 'stop', 'level']).copy()

    df = df.merge(feat, on=['day', 'symbol'], how='left')
    n_miss_he = int(df['half_entry'].isna().sum())
    if n_miss_he:
        log(f'WARNING: {n_miss_he} rows missing half_entry (features_1478_A) -- excluded (1549 cost '
            f'undefined without it)')
    df = df.dropna(subset=['half_entry']).reset_index(drop=True)

    df['fill_min'] = df['fill_min_c'] if 'fill_min_c' in df.columns else df['fill_min']
    df['is_extender'] = df.L3 == 1.0
    df['is_real_sip'] = df.store_served_1438 == 0
    log(f'load_population: pred={len(pred)} causal_fill={len(causal)} -> joined usable={len(df)} '
        f'(kept={int(df.hgb_kept_L3.sum())}, dropped={int((~df.hgb_kept_L3).sum())})')
    return df


def load_bars():
    """{(symbol, day): DataFrame(m,o,h,l,c) sorted, deduped, RTH only} from bars_fills_1478.db.
    t is ISO8601 UTC; converted to ET minute-of-day (hour*60+minute, matching causal_arming._rth)."""
    con = sqlite3.connect(BARS_DB)
    raw = pd.read_sql('select symbol, day, t, o, h, l, c from bars', con)
    con.close()
    ts = pd.to_datetime(raw.t, utc=True).dt.tz_convert(ET)
    raw['m'] = ts.dt.hour * 60 + ts.dt.minute
    raw = raw[(raw.m >= OPEN_M) & (raw.m <= 960)]
    raw = raw.sort_values(['symbol', 'day', 'm']).drop_duplicates(['symbol', 'day', 'm'], keep='first')
    by_key = {k: g[['m', 'o', 'h', 'l', 'c']].reset_index(drop=True) for k, g in raw.groupby(['symbol', 'day'])}
    log(f'load_bars: {len(raw)} RTH minute bars over {len(by_key)} symbol-days')
    return by_key


# ================================================================================================
# Part A -- anatomy: touch search, drawdown, breach, base-stopped-first
# ================================================================================================

def anatomy_one(row, bars):
    """One fill's anatomy dict, or None if bars are missing (counted by the caller). Touch is found
    by walking bars forward from the fill bar (m = floor(fill_min), inclusive -- the fill bar can
    itself touch); if `row.is_extender` and no touch is found in the available bars, returns
    {'missing_touch': True} (excluded, counted, logged) rather than silently treating it as a miss."""
    key = (row.symbol, row.day)
    b = bars.get(key)
    if b is None or b.empty:
        return None
    fill_m = int(np.floor(row.fill_min))
    path = b[b.m >= fill_m]
    if path.empty:
        return None
    target = row.level * TARGET_MULT
    touch_m = np.nan
    for r in path.itertuples():
        if r.h >= target:
            touch_m = r.m
            break
    if row.is_extender and np.isnan(touch_m):
        return {'missing_touch': True}

    if row.is_extender:
        window = path[path.m <= touch_m]
        before = path[path.m < touch_m]
        minutes_to_touch = float(touch_m - row.fill_min)
    else:
        window = path[path.m <= EOD_M]
        if window.empty:
            window = path.iloc[[0]]
        before = window
        minutes_to_touch = np.nan

    min_low = float(window.l.min())
    dd_pct = max(0.0, (row.level - min_low) / row.level * 100.0)
    breach_before = bool((before.l <= row.stop).any()) if len(before) else False
    stopped_before = bool(row.why in STOP_WHY and pd.notna(row.exit_m) and row.exit_m < touch_m) \
        if row.is_extender else bool(row.why in STOP_WHY)
    return dict(touch_m=touch_m, dd_pct=dd_pct, minutes_to_touch=minutes_to_touch,
                breach_before=breach_before, stopped_before=stopped_before)


def build_anatomy(df, bars):
    """Runs anatomy_one over every KEPT row (hgb_kept_L3==True); returns df augmented with anatomy
    columns, dropping (and counting) missing-bar and missing-touch rows."""
    kept = df[df.hgb_kept_L3].reset_index(drop=True)
    recs, n_missing_bars, n_missing_touch = [], 0, 0
    for i, row in enumerate(kept.itertuples()):
        a = anatomy_one(row, bars)
        if a is None:
            n_missing_bars += 1
            recs.append({})
            continue
        if a.get('missing_touch'):
            n_missing_touch += 1
            recs.append({})
            continue
        recs.append(a)
        if (i + 1) % 1000 == 0:
            log(f'build_anatomy: {i + 1}/{len(kept)}')
    an = pd.DataFrame(recs)
    out = pd.concat([kept, an], axis=1)
    n_ok = int(out.touch_m.notna().sum()) if 'touch_m' in out.columns else 0
    log(f'build_anatomy: {len(kept)} kept fills -> {n_missing_bars} missing bars, '
        f'{n_missing_touch} extenders with no touch found (both excluded), {len(kept) - n_missing_bars - n_missing_touch} usable rows')
    out = out.dropna(subset=['dd_pct']) if 'dd_pct' in out.columns else out.iloc[0:0]
    return out.reset_index(drop=True)


def quantiles(s, qs=(0.10, 0.25, 0.50, 0.75, 0.90)):
    s = pd.Series(s).dropna()
    return {q: float(s.quantile(q)) for q in qs} if len(s) else {q: float('nan') for q in qs}


def anatomy_table(an, split, real_sip_only):
    """One row per (extender/non-extender): n, dd_pct quantiles, minutes-to-touch quantiles
    (extenders only), breach share, stopped-before share, exit mix."""
    sub = an[an.split == split]
    if real_sip_only:
        sub = sub[sub.is_real_sip]
    rows = []
    for lab, grp in (('extender', sub[sub.is_extender]), ('non_extender', sub[~sub.is_extender])):
        rows.append(dict(
            split=split, real_sip_only=real_sip_only, group=lab, n=len(grp),
            dd_q=quantiles(grp.dd_pct),
            min_to_touch_q=quantiles(grp.minutes_to_touch) if lab == 'extender' else None,
            breach_share=float(grp.breach_before.mean()) if len(grp) else float('nan'),
            stopped_before_share=float(grp.stopped_before.mean()) if len(grp) else float('nan'),
            exit_mix=grp.why.value_counts(normalize=True).round(4).to_dict() if len(grp) else {},
        ))
    return rows


# ================================================================================================
# d, s, W -- TRAIN-H2 EXTENDERS' quantiles only, written before any VAL number
# ================================================================================================

def compute_dsw(an):
    """d = TRAIN-H2 median drawdown (floored 0.20%), s = TRAIN-H2 p75 drawdown, W = TRAIN-H2 p75
    minutes-to-touch (capped 120) -- extenders in the kept set ONLY. Logged before any VAL read."""
    tr_ext = an[(an.split == 'TRAIN') & an.is_extender]
    d = max(D_FLOOR_PCT, float(tr_ext.dd_pct.median()))
    s = float(tr_ext.dd_pct.quantile(0.75))
    W = min(W_CAP_MIN, float(tr_ext.minutes_to_touch.quantile(0.75)))
    log(f'compute_dsw: TRAIN-H2 extenders n={len(tr_ext)} -> d={d:.4f}% s={s:.4f}% W={W:.2f}min '
        f'(frozen BEFORE any VAL statistic is computed)')
    return d, s, W


# ================================================================================================
# Part B -- 1,548 SWEEP and 1,549 WIDE
# ================================================================================================

def sweep_trade(row, bars, d, s, W):
    """1,548: resting buy limit at level*(1-d/100) for W minutes from the fill bar (through-print:
    a bar low STRICTLY below the limit fills at the limit; the fill bar itself may fill). On fill,
    stop=level*(1-s/100), target=level*1.05, walk_path (sip_rebuild, identical B0 physics)."""
    b = bars.get((row.symbol, row.day))
    if b is None or b.empty:
        return None
    fill_m = int(np.floor(row.fill_min))
    limit = row.level * (1 - d / 100.0)
    stop = row.level * (1 - s / 100.0)
    target = row.level * TARGET_MULT
    window = b[(b.m >= fill_m) & (b.m <= fill_m + W)]
    entry_m = None
    for r in window.itertuples():
        if r.l < limit:
            entry_m = r.m
            break
    if entry_m is None:
        return dict(status='unfilled')
    path = b[b.m >= entry_m]
    exit_m, exit_price, why = walk_path(limit, stop, target, path)
    return dict(status='filled', entry_m=entry_m, entry_price=limit, stop=stop, target=target,
                exit_m=exit_m, exit_price=exit_price, why=why, entry_cost_price=0.0)


def wide_trade(row, bars, s):
    """1,549: the base ask fill (`row.fill`), stop=level*(1-s/100), target=level*1.05, walk_path
    from the fill bar on. Entry cost = half_entry once (the base entry's own half-spread)."""
    b = bars.get((row.symbol, row.day))
    if b is None or b.empty:
        return None
    fill_m = int(np.floor(row.fill_min))
    entry = row.fill
    stop = row.level * (1 - s / 100.0)
    target = row.level * TARGET_MULT
    path = b[b.m >= fill_m]
    if path.empty:
        return None
    exit_m, exit_price, why = walk_path(entry, stop, target, path)
    return dict(status='filled', entry_m=fill_m, entry_price=entry, stop=stop, target=target,
                exit_m=exit_m, exit_price=exit_price, why=why, entry_cost_price=row.half_entry)


def score_trade(t, split):
    """R = entry-stop; raw_R=(exit-entry)/R; cost = entry_cost/R + exit-side bps/R (stop-limit
    standard on stops, EOD-at-the-bid bps on eod/eod_fallback, zero on target); net% = net_R*R/entry."""
    R = t['entry_price'] - t['stop']
    if not np.isfinite(R) or R <= 0:
        return None
    raw_R = (t['exit_price'] - t['entry_price']) / R
    if t['why'] in STOP_WHY:
        exit_bps = SLIP_STOP_BPS[split]
    elif t['why'] in EOD_WHY:
        exit_bps = EOD_BID_BPS[split]
    else:
        exit_bps = 0.0
    exit_cost_R = t['exit_price'] * exit_bps / 1e4 / R
    entry_cost_R = t['entry_cost_price'] / R
    net_R = raw_R - exit_cost_R - entry_cost_R
    net_pct = net_R * R / t['entry_price'] * 100.0
    R_pct_of_price = R / t['entry_price'] * 100.0    # the rail: R itself must exceed the spread
    return dict(R=R, raw_R=raw_R, cost_R=exit_cost_R + entry_cost_R, net_R=net_R, net_pct=net_pct,
                R_pct_of_price=R_pct_of_price)


def run_cell(df, bars, cell_id, d, s, W, real_sip_only=False):
    """Runs one cell (1548 or 1549) over every row of `df` (kept or dropped), for one holdout
    already filtered into df. Returns (fills_df row list, missing_bar_count)."""
    fn = (lambda r: sweep_trade(r, bars, d, s, W)) if cell_id == 1548 else (lambda r: wide_trade(r, bars, s))
    rows, n_missing = [], 0
    for row in df.itertuples():
        if real_sip_only and not row.is_real_sip:
            continue
        t = fn(row)
        if t is None:
            n_missing += 1
            continue
        rec = dict(cell=cell_id, split=row.split, day=row.day, symbol=row.symbol,
                    kept=bool(row.hgb_kept_L3), is_extender=bool(row.is_extender),
                    is_real_sip=bool(row.is_real_sip), status=t['status'])
        if t['status'] == 'unfilled':
            rec.update(base_outcome_R=row.outcome_R)
            rows.append(rec)
            continue
        sc = score_trade(t, row.split)
        if sc is None:
            n_missing += 1
            continue
        rec.update(entry_m=t['entry_m'], exit_m=t['exit_m'], why=t['why'], **sc)
        rows.append(rec)
    return rows, n_missing


def score_one(cell_label, rows, weeks):
    """One cell x population x holdout scoring row (day-clustered t, ex-top-5%, winner-capped,
    fills/week under the live cap, real-SIP mean/t, exit mix)."""
    d = pd.DataFrame(rows)
    filled = d[d.status == 'filled']
    n = len(filled)
    if n == 0:
        return dict(cell=cell_label, n=0)
    mean_R, mean_pct = float(filled.net_R.mean()), float(filled.net_pct.mean())
    t = c1445.day_clustered_t(filled.net_R, filled.day)
    ex5 = c1445.ex_top5_mean(filled.net_R)
    wcap = c1445.winner_capped_mean(filled.net_R, cap=WINNER_CAP_R)
    slot = filled.rename(columns={'entry_m': 'entry_m'})[['day', 'entry_m', 'exit_m']].copy()
    from research.hod_consol import run_consol as consol
    keep = consol.simulate_slots(slot)
    fills_wk = float(keep.sum()) / weeks if weeks else float('nan')
    real_sip = filled[filled.is_real_sip]
    rs_mean = float(real_sip.net_R.mean()) if len(real_sip) else float('nan')
    rs_t = c1445.day_clustered_t(real_sip.net_R, real_sip.day) if len(real_sip) else float('nan')
    fill_share = n / len(d) if len(d) else float('nan')
    exit_mix = filled.why.value_counts(normalize=True).round(4).to_dict()
    median_R_pct_of_price = float(filled.R_pct_of_price.median())
    return dict(cell=cell_label, n=n, fill_share=fill_share, mean_R=mean_R, mean_pct=mean_pct, t=t,
                ex_top5_R=ex5, winner_capped_R=wcap, fills_wk=fills_wk, real_sip_n=len(real_sip),
                real_sip_mean_R=rs_mean, real_sip_t=rs_t, exit_mix=exit_mix,
                median_R_pct_of_price=median_R_pct_of_price)


def null_pctile(all_kept_holdout, cell_score, n_kept):
    """count-matched null: 1,000 draws (seed 1548) of n_kept fills from the kept set's own BASE
    outcome_R on the same holdout; percentile of the cell's actual mean within that null."""
    if n_kept == 0 or np.isnan(cell_score.get('mean_R', np.nan)):
        return float('nan')
    return c1445.null_percentile_of(all_kept_holdout.outcome_R, n_kept, cell_score['mean_R'],
                                     seed=NULL_SEED, n_draws=NULL_DRAWS)


def passes_bar(row):
    """Frozen VAL pass bar (PREREG_1548.md): mean >= +0.15 R and +0.15% of price, t>=2.5, ex-top-5%>0,
    winner-capped>0, >=3 fills/wk, null percentile>=99, real-SIP mean>=+0.10R with t>=2, median R>=0.5%
    of price. (TRAIN-H2-same-sign and kept>dropped are checked separately -- see checklist().)"""
    try:
        return (row['mean_R'] >= 0.15 and row['mean_pct'] >= 0.15 and row['t'] >= 2.5
                and row['ex_top5_R'] > 0 and row['winner_capped_R'] > 0 and row['fills_wk'] >= 3
                and row['null_pctile'] >= 99 and row['real_sip_mean_R'] >= 0.10 and row['real_sip_t'] >= 2
                and row['median_R_pct_of_price'] >= 0.5)
    except (KeyError, TypeError):
        return False


def checklist(cell_id, val_kept, train_kept, val_dropped):
    """Full per-cell pass-bar checklist row incl. the two cross-row conditions (TRAIN-H2 same sign
    with t>=1, kept > dropped on both holdouts) that passes_bar() alone cannot see."""
    items = {
        'VAL mean_R >= +0.15': val_kept.get('mean_R', float('nan')) >= 0.15,
        'VAL mean_pct >= +0.15%': val_kept.get('mean_pct', float('nan')) >= 0.15,
        'VAL t >= 2.5': val_kept.get('t', float('nan')) >= 2.5,
        'VAL ex-top-5% > 0': val_kept.get('ex_top5_R', float('nan')) > 0,
        'VAL winner-capped > 0': val_kept.get('winner_capped_R', float('nan')) > 0,
        'VAL fills/wk >= 3': val_kept.get('fills_wk', float('nan')) >= 3,
        'VAL null percentile >= 99': val_kept.get('null_pctile', float('nan')) >= 99,
        'VAL real-SIP mean >= +0.10R, t>=2': (val_kept.get('real_sip_mean_R', float('nan')) >= 0.10
                                               and val_kept.get('real_sip_t', float('nan')) >= 2),
        'VAL median R >= 0.5% of price (rail)': val_kept.get('median_R_pct_of_price', float('nan')) >= 0.5,
        'TRAIN-H2 same sign, t>=1': (np.sign(train_kept.get('mean_R', 0)) == np.sign(val_kept.get('mean_R', 0))
                                      and abs(train_kept.get('t', 0)) >= 1),
        'kept > dropped (VAL)': val_kept.get('mean_R', float('nan')) > val_dropped.get('mean_R', float('-nan')),
    }
    return dict(cell=cell_id, items=items, all_pass=all(items.values()))


# ================================================================================================
# main
# ================================================================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='fixed 200-fill sample, seed 1548')
    args = ap.parse_args(argv)

    df = load_population()
    bars = load_bars()

    if args.dry_run:
        df = df.sample(n=min(DRY_N, len(df)), random_state=1548).reset_index(drop=True)
        fills_csv, result_md = OUT_FILLS_CSV_DRY, OUT_RESULT_MD_DRY
        log(f'DRY RUN: sampled {len(df)} rows')
    else:
        fills_csv, result_md = OUT_FILLS_CSV, OUT_RESULT_MD

    log('=== Part A: anatomy ===')
    an = build_anatomy(df, bars)
    part_a = []
    for split in ('TRAIN', 'VAL'):
        part_a += anatomy_table(an, split, real_sip_only=False)
        part_a += anatomy_table(an, split, real_sip_only=True)

    log('=== d, s, W (TRAIN-H2 extenders, frozen before VAL) ===')
    d, s, W = compute_dsw(an)

    log('=== Part B: 1,548 SWEEP and 1,549 WIDE, kept vs dropped, both holdouts ===')
    all_rows, score_rows = [], []
    for split in ('TRAIN', 'VAL'):
        for kept_flag, pop_label in ((True, 'kept'), (False, 'dropped')):
            sub = df[(df.split == split) & (df.hgb_kept_L3 == kept_flag)]
            kept_holdout = df[(df.split == split) & (df.hgb_kept_L3 == True)]     # noqa: E712 (null pool)
            weeks = c1445.weeks_spanned(sub.day) if len(sub) else 1
            for cell_id in (1548, 1549):
                rows, n_missing = run_cell(sub, bars, cell_id, d, s, W)
                if n_missing:
                    log(f'WARNING: cell {cell_id} split={split} pop={pop_label}: {n_missing} rows '
                        f'excluded (missing bars / invalid R)')
                all_rows += rows
                sc = score_one(f'{cell_id}_{pop_label}', rows, weeks)
                sc.update(split=split, cell_id=cell_id, population=pop_label)
                if pop_label == 'kept' and sc.get('n', 0):
                    sc['null_pctile'] = null_pctile(kept_holdout, sc, sc['n'])
                    sc['passes_bar'] = passes_bar(sc) if split == 'VAL' else None
                score_rows.append(sc)

    fills_df = pd.DataFrame(all_rows)
    fills_df.to_csv(fills_csv, index=False)
    log(f'wrote {fills_csv} ({len(fills_df)} rows)')

    by_key = {(r['cell_id'], r['population'], r['split']): r for r in score_rows}
    checklists = []
    for cell_id in (1548, 1549):
        val_kept = by_key.get((cell_id, 'kept', 'VAL'), {})
        train_kept = by_key.get((cell_id, 'kept', 'TRAIN'), {})
        val_dropped = by_key.get((cell_id, 'dropped', 'VAL'), {})
        checklists.append(checklist(cell_id, val_kept, train_kept, val_dropped))
        log(f'checklist cell {cell_id}: all_pass={checklists[-1]["all_pass"]}')

    write_result_md(result_md, part_a, d, s, W, score_rows, checklists)
    log(f'wrote {result_md}')


def write_result_md(path, part_a, d, s, W, score_rows, checklists):
    lines = ['# RESULT 1,548-1,549 -- extension anatomy and the sweep entry', '',
             '## Parameters (TRAIN-H2 extenders only, frozen BEFORE any VAL statistic)',
             f'd (limit below level)  = {d:.4f}%', f's (stop below level)   = {s:.4f}%',
             f'W (resting window)     = {W:.2f} min', '',
             '## Part A -- anatomy (extenders vs non-extenders, both holdouts, all rows and real-SIP)']
    for r in part_a:
        lines.append(f"- split={r['split']} real_sip_only={r['real_sip_only']} group={r['group']} "
                     f"n={r['n']} dd%_q={r['dd_q']} min_to_touch_q={r['min_to_touch_q']} "
                     f"breach_before={r['breach_share']:.3f} stopped_before={r['stopped_before_share']:.3f} "
                     f"exit_mix={r['exit_mix']}")
    lines += ['', '## Part B -- cell x population x holdout']
    for r in score_rows:
        lines.append(str(r))
    lines += ['', '## Caveats',
              '- d/s/W computed on TRAIN-H2 extenders only, before any VAL statistic (see log order).',
              '- SWEEP mean/t/ex-top5/etc. are computed over FILLED trades only; unfilled sweeps are '
              'reported in cell_1548_fills.csv with their base outcome_R but do not enter the mean.',
              '- Missing-bar / no-touch-found / invalid-R rows are excluded and counted in the run log, '
              'never imputed.',
              '- Real-SIP subset = store_served_1438==0 (features_1478_A / predictions.csv).']
    lines += ['', '## Pass-bar checklist (VAL, per cell)']
    for c in checklists:
        lines.append(f"### cell {c['cell']} -- {'PASS' if c['all_pass'] else 'FAIL'}")
        for k, v in c['items'].items():
            lines.append(f"  [{'x' if v else ' '}] {k}")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
