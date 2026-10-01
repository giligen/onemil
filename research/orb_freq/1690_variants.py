#!/usr/bin/env python3
"""Cell 1,690 -- can ORB add-on pool P1 (idea1, gap 3-5%) be improved?

PREREG: research/orb_freq/PREREG_1690.md (FROZEN 2026-10-01). Owner's ask: "the new pool that is
negative this quarter, can we improve it?" (P1 = cell 1,684 idea1, -0.008 R / 64 fills in Q3 2026).

Reused, unmodified (project convention: digit-prefixed filenames imported via importlib):
  research/orb_exit/1679_orb_exit.py   -- find_range_and_breakout, _lock_walk, _trail_walk,
                                           LOCK_TRIGGER_R_LIVE, LOCK_STOP_R_LIVE, EXIT_SLIP_BPS,
                                           ORB_EOD_M, R_FLOOR_PCT.
  research/hod_entry/1668_failure.py   -- minute_of_day, et_offset_minutes (ET conversion for the
                                           same ISO-UTC+offset timestamp format cache.db uses).
  research/orb_freq/1684_score.py-style stats() (reimplemented here verbatim, not imported --
  the source filename also starts with a digit and the function is 15 lines; copying it keeps this
  script single-file and auditable).

Bar source for variants (b)/(c): data/cache.db::intraday_bars_1min (read-only, ?mode=ro, indexed on
(symbol,bar_date) -- NOT research/bf_zero/bars_sip.db, whose coverage of P1's own population measured
32.5% (13/40 sample) during PREREG, below the 80% availability rail; cache.db is also the production
bar source (study_orb_pipeline_static_lock.BARS_DB_PATH default).

Usage: nice -n 10 python3 research/orb_freq/1690_variants.py
"""
import importlib.util
import logging
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_freq'
SUBPOOLS = OUT / 'subpools_1685'
CACHE_DB = ROOT / 'data/cache.db'
POOL_BOOK = OUT / '1684_pool_books.csv'

LOG_FILE = OUT / '1690_variants.log'
READS_CSV = OUT / '1690_reads.csv'

R_USD = 375.0
TRAIN = ('2025-01-01', '2025-12-31')
VAL = ('2026-01-01', '2026-06-30')
HELDOUT = ('2026-07-01', '2026-09-18')  # disclosed gap to 09-26 in PREREG_1690.md -- not built, not fetched
WINDOWS = [('TRAIN', TRAIN), ('VAL', VAL), ('HELDOUT', HELDOUT)]

COST_FRAC = 0.0013      # 13 bps round trip
RANGE_PCT_GATE = 0.75   # variant (c) eligibility
COST_CAP_R = 0.10       # variant (c) target cap

logger = logging.getLogger('1690')


def setup_logging():
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(sh)


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relpath))
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = old_argv
    return mod


def _ro(path):
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True)


# --------------------------------------------------------------------------- scoring (1684_score.stats)
def stats(df, label, r_col='R'):
    if df is None or len(df) == 0:
        return dict(label=label, n=0, fills_wk=np.nan, mean_r=np.nan, dc_t=np.nan,
                    ex_top5=np.nan, weekly_p10=np.nan, worst_week=np.nan, n_weeks=0,
                    green_weeks=0, total_usd=0.0)
    r = df[r_col].to_numpy(float)
    n = len(r)
    weeks = df['date'].dt.to_period('W')
    n_weeks = weeks.nunique()
    mean_r = r.mean()
    daily = df.groupby(df['date'].dt.date)[r_col].mean()
    n_days = len(daily)
    dsd = daily.std(ddof=1) if n_days > 1 else float('nan')
    dc_t = (daily.mean() / (dsd / np.sqrt(n_days))) if n_days > 1 and dsd > 0 else float('nan')
    k = max(1, int(np.ceil(n * 0.05)))
    thresh = pd.Series(r).nlargest(k).min()
    ex_top5 = r[r < thresh].mean() if (r < thresh).any() else float('nan')
    weekly_sum = df.groupby(weeks)[r_col].sum()
    p10 = weekly_sum.quantile(0.10)
    worst = weekly_sum.min()
    green = int((weekly_sum > 0).sum())
    return dict(label=label, n=n, fills_wk=n / n_weeks if n_weeks else float('nan'), mean_r=mean_r,
                dc_t=dc_t, ex_top5=ex_top5, weekly_p10=p10, worst_week=worst, n_weeks=n_weeks,
                green_weeks=green, total_usd=float(r.sum() * R_USD))


def window_slice(df, lo, hi):
    m = (df['date'] >= pd.Timestamp(lo)) & (df['date'] <= pd.Timestamp(hi))
    return df[m].copy()


# --------------------------------------------------------------------------- P1 base book
def load_p1_base():
    df = pd.read_csv(POOL_BOOK, keep_default_na=False, na_values=[''])
    d = df[(df['pool'] == 'idea1') & (df['window'] == 'in_regime')].copy()
    d['date'] = pd.to_datetime(d['date'])
    d['R'] = d['_sized_pnl'].astype(float) / R_USD
    d['datestr'] = d['date'].dt.strftime('%Y-%m-%d')
    return d.sort_values('date').reset_index(drop=True)


# --------------------------------------------------------------------------- variant (a): feature gates
def load_feature_members(feat):
    """Union of (date,symbol) from B{feat}/C{feat}_in_regime_true.csv, entered==1."""
    members = set()
    found_any = False
    for band in ('B', 'C'):
        fp = SUBPOOLS / f'{band}{feat.upper()}_in_regime_true.csv'
        if not fp.exists():
            logger.warning('variant a/%s: %s missing -- band %s skipped', feat, fp, band)
            continue
        found_any = True
        df = pd.read_csv(fp, keep_default_na=False, na_values=[''])
        if 'entered' in df.columns:
            df = df[df['entered'].astype(str).isin(['1', 'True', 'true'])]
        df['date'] = df['date'].astype(str)
        members.update(zip(df['date'], df['symbol']))
    if not found_any:
        logger.error('variant a/%s: NEITHER band file found -- variant VOID', feat)
        return None
    return members


def variant_a(p1_base, feat):
    members = load_feature_members(feat)
    if members is None:
        return None
    mask = [(d, s) in members for d, s in zip(p1_base['datestr'], p1_base['symbol'])]
    sub = p1_base[pd.Series(mask, index=p1_base.index)].copy()
    logger.info('variant a/%s: %d/%d P1 rows match (date,symbol) in BF%s|CF%s true-files',
                feat, len(sub), len(p1_base), feat[1:], feat[1:])
    return sub


# --------------------------------------------------------------------------- variants (b)/(c): bar walk
def load_bars_for(con, f1668, symbol, date_str):
    cur = con.execute(
        "SELECT timestamp,open,high,low,close FROM intraday_bars_1min "
        "WHERE symbol=? AND bar_date=? ORDER BY timestamp", (symbol, date_str))
    rows = cur.fetchall()
    if not rows:
        return None
    minarr = np.array([f1668.minute_of_day(r[0], date_str) for r in rows], dtype=float)
    return dict(minarr=minarr,
                o=np.array([r[1] for r in rows], dtype=float),
                h=np.array([r[2] for r in rows], dtype=float),
                l=np.array([r[3] for r in rows], dtype=float),
                c=np.array([r[4] for r in rows], dtype=float))


def _scale50_1R(bars, i0, entry, stop, R_unit, eod_m, slip_bps, e1679):
    """50% at touch+1R (slip), the other 50% independently rides the live-lock
    walk over the full path -- identical construction to 1679's scale50_1R_plus_live."""
    n = len(bars['o'])
    slip = slip_bps / 10000.0
    touch1r_j = None
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            break
        if float(bars['h'][j]) >= entry + 1.0 * R_unit:
            touch1r_j = j
            break
    px_e = e1679._lock_walk(bars, i0, entry, stop, R_unit, e1679.LOCK_TRIGGER_R_LIVE,
                             e1679.LOCK_STOP_R_LIVE, eod_m, slip_bps)
    if px_e is None:
        return np.nan
    if touch1r_j is None:
        return (px_e - entry) / R_unit
    leg1_px = (entry + 1.0 * R_unit) * (1 - slip)
    return 0.5 * (leg1_px - entry) / R_unit + 0.5 * (px_e - entry) / R_unit


def _half3r_trail1r_walk(bars, i0, entry, stop_orig, R_unit, target_r, trail_r, eod_m, slip_bps):
    """PREREG_1684 item 41 / PREREG_1690 (b): stop stays at the ORIGINAL stop (no tightening --
    the "no exit at 2R" is literal, 2R is a waypoint price must cross, not a trigger) until the
    first bar whose high >= entry+target_r*R; that bar, 50% exits at entry+target_r*R (touch+slip);
    the other 50% then trails at running-high - trail_r*R (floored at the original stop) to the
    stop/EOD. If the target never fires, 100% rides the ORIGINAL stop to the stop/EOD -- nothing in
    the spec triggers a tighter stop or a trail before the target. Same same-bar update-then-check
    precedence and touch+slip convention as 1679's _lock_walk/_trail_walk."""
    n = len(bars['o'])
    if i0 + 1 >= n:
        return np.nan
    slip = slip_bps / 10000.0
    target_lvl = entry + target_r * R_unit
    run_high = float(bars['h'][i0])
    armed = False
    stop_price = stop_orig
    leg1_px = None
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            exit_px = float(bars['c'][j]) * (1 - slip)
            if leg1_px is None:
                return (exit_px - entry) / R_unit
            return 0.5 * (leg1_px - entry) / R_unit + 0.5 * (exit_px - entry) / R_unit
        bh, bl = float(bars['h'][j]), float(bars['l'][j])
        if not armed:
            if bh >= target_lvl:
                armed = True
                leg1_px = target_lvl * (1 - slip)
                run_high = max(run_high, bh)
                stop_price = max(stop_price, run_high - trail_r * R_unit)
                if bl <= stop_price:
                    return 0.5 * (leg1_px - entry) / R_unit + 0.5 * (stop_price - entry) / R_unit
            elif bl <= stop_price:
                return (stop_price - entry) / R_unit
        else:
            run_high = max(run_high, bh)
            stop_price = max(stop_price, run_high - trail_r * R_unit)
            if bl <= stop_price:
                return 0.5 * (leg1_px - entry) / R_unit + 0.5 * (stop_price - entry) / R_unit
    exit_px = float(bars['c'][n - 1]) * (1 - slip)
    if leg1_px is None:
        return (exit_px - entry) / R_unit
    return 0.5 * (leg1_px - entry) / R_unit + 0.5 * (exit_px - entry) / R_unit


def walk_all(p1_base, con, f1668, e1679):
    rows = []
    n_no_bars = n_no_range = n_bad_R = 0
    for r in p1_base.itertuples():
        bars = load_bars_for(con, f1668, r.symbol, r.datestr)
        if bars is None or len(bars['o']) < 5:
            n_no_bars += 1
            logger.warning('walk %s %s: no/short bars in cache.db -- EXCLUDED', r.symbol, r.datestr)
            continue
        rec = e1679.find_range_and_breakout(bars)
        if rec is None:
            n_no_range += 1
            logger.warning('walk %s %s: no 5-min range/breakout bar -- EXCLUDED', r.symbol, r.datestr)
            continue
        i0 = rec['i0']
        entry, stop = float(r.entry_price), rec['range_low']
        R_unit = entry - stop
        if not (R_unit > 0):
            n_bad_R += 1
            logger.warning('walk %s %s: R_unit<=0 -- EXCLUDED', r.symbol, r.datestr)
            continue
        R_pct = R_unit / entry
        range_pct = (rec['range_high'] - rec['range_low']) / entry * 100.0
        px_live = e1679._lock_walk(bars, i0, entry, stop, R_unit, e1679.LOCK_TRIGGER_R_LIVE,
                                    e1679.LOCK_STOP_R_LIVE, e1679.ORB_EOD_M, e1679.EXIT_SLIP_BPS)
        live_rule_R = (px_live - entry) / R_unit if px_live is not None else np.nan
        scale50_R = _scale50_1R(bars, i0, entry, stop, R_unit, e1679.ORB_EOD_M, e1679.EXIT_SLIP_BPS, e1679)
        noexit_R = _half3r_trail1r_walk(bars, i0, entry, stop, R_unit, 3.0, 1.0,
                                         e1679.ORB_EOD_M, e1679.EXIT_SLIP_BPS)
        rows.append(dict(date=r.date, datestr=r.datestr, symbol=r.symbol, R_base=r.R,
                          R_unit=R_unit, R_pct=R_pct, range_pct=range_pct,
                          floor_ok=R_pct >= e1679.R_FLOOR_PCT,
                          live_rule_R=live_rule_R, scale50_1R_R=scale50_R, noexit2R_half3R_trail1R_R=noexit_R))
    n_total = len(p1_base)
    n_ok = len(rows)
    logger.info('bar walk coverage: %d/%d usable (%.1f%%) | no_bars=%d no_range=%d bad_R=%d',
                n_ok, n_total, 100.0 * n_ok / max(n_total, 1), n_no_bars, n_no_range, n_bad_R)
    if n_ok / max(n_total, 1) < 0.80:
        logger.warning('bar walk coverage %.1f%% is BELOW the 80%% availability rail -- '
                        'variants (b)/(c) numbers below are on a reduced, non-random population; '
                        'disclosed in RESULT, not hidden', 100.0 * n_ok / max(n_total, 1))
    return pd.DataFrame(rows), dict(n_total=n_total, n_ok=n_ok, n_no_bars=n_no_bars,
                                     n_no_range=n_no_range, n_bad_R=n_bad_R)


def apply_cost_sizing(walked):
    rp = walked['range_pct'] / 100.0
    eligible = rp < (RANGE_PCT_GATE / 100.0)
    s = np.where(eligible, np.minimum(1.0, COST_CAP_R * rp / COST_FRAC), 1.0)
    walked = walked.copy()
    walked['sizing_factor'] = s
    walked['cost_sized_R'] = s * walked['R_base']
    n_eligible = int(eligible.sum())
    logger.info('variant c: %d/%d fills have range_pct<%.2f%% and are sized down (mean factor %.3f)',
                n_eligible, len(walked), RANGE_PCT_GATE, s[eligible].mean() if n_eligible else float('nan'))
    return walked


# --------------------------------------------------------------------------- selection rule
def is_candidate(train_s, val_s):
    return (train_s['mean_r'] >= 0.05 and train_s['dc_t'] >= 1.5 and
            val_s['mean_r'] >= 0.05 and val_s['dc_t'] >= 1.5)


def main():
    setup_logging()
    logger.info('=== cell 1,690: P1 improvement variants -- starting ===')
    st = os.statvfs('/')
    free_gb = st.f_bavail * st.f_frsize / (1024 ** 3)
    logger.info('disk free on /: %.1f GB', free_gb)
    if free_gb < 5.0:
        logger.error('disk free %.1f GB < 5.0 GB floor -- aborting', free_gb)
        sys.exit(1)

    p1_base = load_p1_base()
    logger.info('P1 base book: n=%d, %s..%s', len(p1_base), p1_base['datestr'].min(), p1_base['datestr'].max())

    all_reads = []

    # plain P1 reference, all three windows
    for wname, (lo, hi) in WINDOWS:
        s = stats(window_slice(p1_base, lo, hi), f'P1_plain__{wname}')
        s['variant'] = 'P1_plain'
        s['window'] = wname
        all_reads.append(s)

    # variant (a): feature gates F1,F3,F4,F5,F6 (F7 VOID -- PREREG_1690.md)
    for feat in ('f1', 'f3', 'f4', 'f5', 'f6'):
        sub = variant_a(p1_base, feat)
        vname = f'a_{feat.upper()}'
        if sub is None:
            all_reads.append(dict(label=f'{vname}__VOID', n=0, variant=vname, window='VOID'))
            continue
        for wname, (lo, hi) in WINDOWS:
            s = stats(window_slice(sub, lo, hi), f'{vname}__{wname}')
            s['variant'] = vname
            s['window'] = wname
            all_reads.append(s)

    # variants (b)/(c): shared bar walk
    f1668 = _load_module('f1668_1690', 'research/hod_entry/1668_failure.py')
    e1679 = _load_module('e1679_1690', 'research/orb_exit/1679_orb_exit.py')
    con = _ro(CACHE_DB)
    walked, cov = walk_all(p1_base, con, f1668, e1679)
    con.close()
    walked['date'] = pd.to_datetime(walked['datestr'])

    # (b) live_rule sanity check vs base book R (not a candidate -- logged, not scored for selection)
    sane = walked.dropna(subset=['live_rule_R'])
    if len(sane):
        diff = (sane['live_rule_R'] - sane['R_base']).abs()
        logger.info('live_rule sanity check: n=%d median|diff|=%.4f R mean|diff|=%.4f R '
                    '(expect small -- independent i0/stop reconstruction vs the base book)',
                    len(sane), diff.median(), diff.mean())

    floored = walked[walked['floor_ok']]
    n_floor_excl = len(walked) - len(floored)
    logger.info('R floor (<0.5%% of entry): %d/%d fail -- excluded from (b) R-unit reads', n_floor_excl, len(walked))

    for rule, vname in (('live_rule_R', 'b_live_rule'), ('scale50_1R_R', 'b_scale50_1R'),
                         ('noexit2R_half3R_trail1R_R', 'b_noexit2R_half3R_trail1R')):
        sub = floored.dropna(subset=[rule]).copy()
        for wname, (lo, hi) in WINDOWS:
            s = stats(window_slice(sub, lo, hi), f'{vname}__{wname}', r_col=rule)
            s['variant'] = vname
            s['window'] = wname
            all_reads.append(s)

    # (c) cost-aware sizing -- same population as plain P1 (join R_base back), no R floor (dollar-based R)
    costed = apply_cost_sizing(walked)
    for wname, (lo, hi) in WINDOWS:
        s = stats(window_slice(costed, lo, hi), f'c_cost_sizing__{wname}', r_col='cost_sized_R')
        s['variant'] = 'c_cost_sizing'
        s['window'] = wname
        all_reads.append(s)

    reads_df = pd.DataFrame(all_reads)
    reads_df.to_csv(READS_CSV, index=False)
    logger.info('wrote %s (%d rows)', READS_CSV, len(reads_df))

    # selection rule: candidate iff TRAIN and VAL both clear mean_r>=0.05 and dc_t>=1.5
    pivot = {}
    for v in reads_df['variant'].unique():
        rows = {r['window']: r for r in reads_df[reads_df['variant'] == v].to_dict('records')}
        pivot[v] = rows

    candidates = []
    for v, rows in pivot.items():
        if v in ('P1_plain', 'b_live_rule'):
            continue  # reference / sanity check, never eligible
        if 'TRAIN' not in rows or 'VAL' not in rows or rows['TRAIN'].get('n', 0) == 0 or rows['VAL'].get('n', 0) == 0:
            continue
        if is_candidate(rows['TRAIN'], rows['VAL']):
            candidates.append(v)
    logger.info('candidates clearing TRAIN+VAL (mean_r>=0.05, dc_t>=1.5 both): %s', candidates or 'NONE')

    winner = None
    best_held_mean = None
    for v in candidates:
        held = pivot[v].get('HELDOUT')
        if held is None or held.get('n', 0) == 0:
            continue
        if held['mean_r'] >= 0 and (best_held_mean is None or held['mean_r'] > best_held_mean):
            best_held_mean = held['mean_r']
            winner = v
    verdict = winner if winner else 'no improvement'
    logger.info('WINNER: %s', verdict)

    with open(OUT / '1690_verdict.txt', 'w') as fh:
        fh.write(f'candidates={candidates}\nwinner={verdict}\ncoverage={cov}\n')

    logger.info('=== DONE ===')
    return dict(candidates=candidates, winner=verdict, coverage=cov)


if __name__ == '__main__':
    main()
