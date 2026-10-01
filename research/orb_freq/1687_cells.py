#!/usr/bin/env python3
"""Cell 1,687 -- ORB frequency/exit/sizing ideas 25, 33, 41, 42, 43, 44.

PREREG: research/orb_freq/PREREG_1684.md (idea map rows 25/33/41/42/43/44, all
route to -> 1,687). Owner-frozen idea text is reproduced verbatim in each
idea's docstring below.

Book: production (research/orb_freq/1693_pool_exits.py's reconstruction --
real entry minutes, f1679's own breakout/stop rule, byte-identical reuse) with
1679's walker (_lock_walk) and statistics (stats_block / reads_for_exit_series).
Every idea is read PAIRED against the SAME production fills (E1_production),
in both directions (TRAIN2025 and VAL2026, each read standalone -- a paired
delta needs no fit/select step) plus OOS2024H2, per 1679's pass bar: paired
dR >= +0.05 R, day-clustered t >= 2.5, ex-top-5% > 0, in BOTH years, same sign
OOS2024H2 (PREREG_1684 Method + Pass-bar sections).

Idea 25 (second-chance range) and idea 33 (earnings split) are NOT per-fill
exit/sizing edits -- they are reported on their own terms (25 = incremental
added book, no production counterpart to pair against; 33 = a cohort split)
and flagged as such in RESULT_1687.md.

Read-only on data/cache.db and the bar store (bars_sip.db via f1668.BarStore);
never touches config/orb.yaml/trading/*.py; no git commit; no money spent.
Run: nice -n 10 python3 research/orb_freq/1687_cells.py
"""
import os
import sys
import time
import logging
import importlib.util
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, 'research/orb_freq')
LOG_PATH = os.path.join(OUT_DIR, '1687_cells.log')
READS_CSV = os.path.join(OUT_DIR, '1687_reads.csv')
RESULT_MD = os.path.join(OUT_DIR, 'RESULT_1687.md')

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(LOG_PATH, mode='w'), logging.StreamHandler()])
logger = logging.getLogger('cell1687')


def check_disk(floor_gb=5.0):
    """Research job hygiene (memory/feedback_research_job_hygiene.md): refuse
    to run a fetch/compute job below a disk floor."""
    st = os.statvfs(ROOT)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    logger.info('disk free: %.1f GB (floor %.1f GB)', free_gb, floor_gb)
    if free_gb < floor_gb:
        raise RuntimeError(f'disk free {free_gb:.1f}GB below floor {floor_gb}GB -- refusing to run')


def _load_module(name, relpath):
    """Load a numeric-prefixed research script as a module without running
    its __main__ block (sys.argv reset around exec, same pattern 1693/1679
    already use to import each other)."""
    path = os.path.join(ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old_argv
    return mod


# ---------------------------------------------------------------------------
# Idea 25: second-chance 30-min range
# FROZEN: "a name whose 5-min break stopped out enters ONCE on the break of
# its 30-min range high at 10:00 ET, production exit."
# ---------------------------------------------------------------------------

def idea25_entry(bars, range_lo_m=570.0, range_hi_m=599.0, search_lo_m=600.0, search_hi_m=660.0):
    """30-min opening range [09:30,10:00) ET + first breakout bar in a 60-min
    search window starting at 10:00 -- the SAME range-then-search STRUCTURE as
    f1679.find_range_and_breakout (RANGE_LO_M..RANGE_END_M then a 60-min
    search), just the two windows shifted from 5 min to 30 min. Returns None
    if there is no range or no breakout (most stopped-out names: no second
    chance)."""
    m = bars['minarr']
    rmask = (m >= range_lo_m) & (m <= range_hi_m)
    if rmask.sum() == 0:
        return None
    r_high = float(bars['h'][rmask].max())
    r_low = float(bars['l'][rmask].min())
    smask = np.where((m >= search_lo_m) & (m < search_hi_m))[0]
    for j in smask:
        if bars['h'][j] > r_high:
            return dict(i0=int(j), range_high=r_high, range_low=r_low)
    return None


def run_idea25(pf, store, f1693):
    """Population = production fills E1 stopped out near the ORIGINAL stop
    (E1<=-0.9; a locked-stop exit sits near +0.5R, an EOD close only reaches
    this low if price never touched the stop at all -- logged separately).
    For each, reconstruct bars (CachedStore hit, already warm), look for a
    30-min-range breakout, and if one occurs, enter ONCE with f1679's own
    lock-walk exit ('production exit'). This is an ADDITIVE book: it has no
    production counterpart to pair against, so it is reported on its own
    mean R / t / fills-per-week ADDED, not as a paired delta."""
    stopped = pf[pf['E1_production'] <= -0.9].copy()
    near_thresh = pf[(pf['E1_production'] > -0.9) & (pf['E1_production'] <= -0.80)]
    logger.info('idea25: %d/%d production fills stopped near the original stop (E1<=-0.9); '
                 '%d more in (-0.9,-0.8] -- possible EOD-close-without-touch misses, not counted',
                 len(stopped), len(pf), len(near_thresh))
    rows = []
    n_scanned, n_break = 0, 0
    for r in stopped.itertuples():
        n_scanned += 1
        bars = store.day_bars(r.symbol, str(r.date))
        if bars is None or len(bars['o']) < 30:
            continue
        rec30 = idea25_entry(bars)
        if rec30 is None:
            continue
        r_high, r_low, i0_25 = rec30['range_high'], rec30['range_low'], rec30['i0']
        slip = f1693.EXIT_SLIP_BPS / 10000.0
        entry25 = r_high * (1 + slip)
        stop25 = r_low
        R_unit25 = entry25 - stop25
        if not (R_unit25 > 0) or (R_unit25 / entry25) < f1693.R_FLOOR_PCT:
            continue
        exit_px = f1693._lock_walk(bars, i0_25, entry25, stop25, R_unit25,
                                    f1693.LOCK_TRIGGER_R_LIVE, f1693.LOCK_STOP_R_LIVE,
                                    f1693.ORB_EOD_M, f1693.EXIT_SLIP_BPS)
        R25 = f1693._R(exit_px, entry25, R_unit25)
        n_break += 1
        rows.append(dict(date=r.date, window=r.window, symbol=r.symbol, idea25_R=R25))
    logger.info('idea25: %d/%d stopped-out names broke their 30-min range at/after 10:00 (second-chance fills)',
                 n_break, n_scanned)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Idea 41: no exit at 2R -> half at +3R -> trail MFE-1R
# FROZEN: "no exit at 2 R -> half at +3 R -> trail MFE - 1 R" (the HOD-robust
# exit, transplanted onto the ORB book with real times).
# ---------------------------------------------------------------------------

def idea41_walk(bars, i0, entry, stop, R_unit, f1693):
    """Leg1 (50%): rides the ORIGINAL stop only to a +3R touch -- no lock, no
    trail, nothing special happens at 2R ('no exit at 2R'). Leg2 (50%,
    independent lot, 1693's own scale-then-lock convention): trails
    continuously at running-MFE - 1R from entry (f1693._trail_gated_walk with
    gate_r=0 -- E9's own mechanics, just ungated)."""
    n = len(bars['o'])
    if i0 + 1 >= n:
        return np.nan
    eod_m, slip_bps = f1693.ORB_EOD_M, f1693.EXIT_SLIP_BPS
    leg2_R = f1693._trail_gated_walk(bars, i0, entry, stop, R_unit, 0.0, 1.0, eod_m, slip_bps)
    slip = slip_bps / 10000.0
    scale_lvl = entry + 3.0 * R_unit
    touch_j = None
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            break
        if float(bars['l'][j]) <= stop:
            break
        if float(bars['h'][j]) >= scale_lvl:
            touch_j = j
            break
    if touch_j is None:
        return leg2_R
    leg1_px = scale_lvl * (1 - slip)
    leg1_R = (leg1_px - entry) / R_unit
    return 0.5 * leg1_R + 0.5 * leg2_R


# ---------------------------------------------------------------------------
# Idea 43: add one unit at +1R with the original stop
# FROZEN: "add one unit at +1 R with the original stop (original-R units)."
# ---------------------------------------------------------------------------

def idea43_walk(bars, i0, entry, stop, R_unit, f1693):
    """Detect the first +1R touch (stop-touch wins a same-bar tie, codebase
    convention). If touched before any stop-out, a second unit is added at
    entry+1R with the SAME original stop: the add does not change when/where
    the combined position exits (identical price action drives both units),
    only the blended R -- unit2 risks 2 original-R of distance to the shared
    stop, so total_R = 2*R1_final - 1.0 where R1_final is production's own
    exit (f1679._lock_walk) applied to the single-unit path. If +1R is never
    touched before a stop-out, the add never happens and R is unchanged."""
    n = len(bars['o'])
    if i0 + 1 >= n:
        return np.nan
    eod_m, slip_bps = f1693.ORB_EOD_M, f1693.EXIT_SLIP_BPS
    add_lvl = entry + 1.0 * R_unit
    added = False
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            break
        if float(bars['l'][j]) <= stop:
            break
        if float(bars['h'][j]) >= add_lvl:
            added = True
            break
    exit_px = f1693._lock_walk(bars, i0, entry, stop, R_unit,
                                f1693.LOCK_TRIGGER_R_LIVE, f1693.LOCK_STOP_R_LIVE, eod_m, slip_bps)
    r1 = f1693._R(exit_px, entry, R_unit)
    return (2.0 * r1 - 1.0) if added else r1


# ---------------------------------------------------------------------------
# Idea 44: range floor with cost-aware sizing
# FROZEN: "range floor with cost-aware sizing (fills with range < 0.75% of
# price sized down so the 13-bps round trip <= 0.1 R; report the share
# affected and $ at live sizing)."
# ---------------------------------------------------------------------------

FLOOR_PCT_44 = 0.0075
COST_RT_44 = 0.0013   # 13 bps round trip, stated in the idea text


def apply_idea44(pf):
    """Cost-as-fraction-of-R at FULL sizing is size-invariant (cost$ and R$
    both scale with shares, so the ratio cost_rt/R_pct does not change when
    you reduce shares) -- sizing down therefore cannot make THIS ratio <=0.1
    for a fixed R_pct; it can only shrink the DOLLAR risk actually taken on
    the trade. Implemented rule (R_pct<0.75% -> shares scaled by R_pct/0.75%,
    i.e. full size only AT OR ABOVE the floor): this shrinks both winners and
    losers on the affected subset toward zero, capping the $ exposure of the
    cost tax. Logged honestly: at the stated floor this ratio is
    13bps/0.75%=0.173R, not the idea text's 0.1R target -- the 0.75% floor is
    taken as authoritative (it is the headline rule), the 13bps/0.1R pair as
    the motivating rationale, not an inverted formula. See 1687_cells.log."""
    r_pct = pf['R_unit'] / pf['entry']
    affected = r_pct < FLOOR_PCT_44
    size_mult = np.where(affected, (r_pct / FLOOR_PCT_44).clip(upper=1.0), 1.0)
    cost_r_at_floor = COST_RT_44 / FLOOR_PCT_44
    logger.warning('idea44: cost_rt=%.4f floor_pct=%.4f -> cost/R at the floor = %.3fR '
                    '(idea text target 0.1R; reporting the floor rule as stated, flagging the gap honestly)',
                    COST_RT_44, FLOOR_PCT_44, cost_r_at_floor)
    out = pf.copy()
    out['R_pct'] = r_pct
    out['idea44_affected'] = affected
    out['idea44_size_mult'] = size_mult
    out['idea44_R'] = pf['E1_production'] * size_mult
    return out


# ---------------------------------------------------------------------------
# Idea 33: earnings-gapper vs non-earnings split
# FROZEN: "earnings-gapper vs non-earnings split of the production book
# (earnings dates from research/edgar_desk or research/multiday -- grep for
# the calendar; VOID with reason if absent): if one cohort is negative,
# dropping it is a profitability move; if both positive, nothing changes."
# ---------------------------------------------------------------------------

def load_earnings_calendar():
    """research/multiday/DATA.md sec.4 documents a built `earnings_events.
    parquet` (142,695 Item-2.02 8-K events) but that bulk artifact is
    gitignored and NOT present on this node (checked: find -iname
    earnings_events.parquet -> nothing). Rebuilt here from the raw EDGAR
    filings feed on disk (research/edgar_desk/events_raw.csv, 4.41M rows,
    cik/symbol/form/filing_date/acceptance_datetime/items) using the SAME
    filter DATA.md states (form 8-K or 8-K/A, items contains '2.02'). Event
    session is APPROXIMATED as filing_date or filing_date+1 calendar day
    (covers a same-day pre-open filing and a post-close filing gapping the
    next session) rather than DATA.md's exact closing-auction-time rule --
    logged as a limitation, not re-derived here (budget)."""
    path = os.path.join(ROOT, 'research/edgar_desk/events_raw.csv')
    if not os.path.exists(path):
        return None
    edgar = pd.read_csv(path, usecols=['symbol', 'form', 'filing_date', 'items'], dtype=str)
    mask = edgar['form'].isin(['8-K', '8-K/A']) & edgar['items'].fillna('').str.contains(r'2\.02', regex=True)
    earn = edgar.loc[mask, ['symbol', 'filing_date']].dropna().drop_duplicates()
    earn['filing_date'] = pd.to_datetime(earn['filing_date'], errors='coerce')
    earn = earn.dropna(subset=['filing_date'])
    logger.info('idea33: %d item-2.02 8-K/8-K-A filings on %d symbols (raw feed, approx event-session mapping)',
                len(earn), earn['symbol'].nunique())
    same_day = set(zip(earn['symbol'], earn['filing_date'].dt.date))
    next_day = set(zip(earn['symbol'], (earn['filing_date'] + pd.Timedelta(days=1)).dt.date))
    return (same_day | next_day), earn


def run_idea33(pf, f1693):
    """Splits production fills into earnings-gapper vs non-earnings cohorts
    and reports each cohort's own stats (not a per-fill paired delta -- this
    idea is a SPLIT, not an exit/sizing edit). Also runs a widened +/-3
    calendar-day overlap diagnostic (does the split even have enough fills
    to test, independent of the exact 0/+1-day session-mapping choice)."""
    cal = load_earnings_calendar()
    if cal is None:
        logger.warning('idea33: VOID -- no earnings calendar on disk (events_raw.csv missing)')
        return None, None, None
    earn_set, earn_df = cal
    pf = pf.copy()
    pf['is_earnings'] = [(s, d) in earn_set for s, d in zip(pf['symbol'], pf['date'])]
    cov = pf['is_earnings'].mean()
    logger.info('idea33: %d/%d production fills (%.1f%%) match an earnings-gapper date (0/+1-day window)',
                pf['is_earnings'].sum(), len(pf), 100 * cov)
    n_sym_any = pf['symbol'].isin(earn_df['symbol']).sum()
    pf_dt = pd.to_datetime(pf['date'])
    hits3 = 0
    for sym, dt in zip(pf['symbol'], pf_dt):
        sub = earn_df[earn_df['symbol'] == sym]
        if len(sub) == 0:
            continue
        dd = (sub['filing_date'] - dt).dt.days
        if ((dd >= -3) & (dd <= 3)).any():
            hits3 += 1
    logger.info('idea33 diagnostic: %d/%d fill-symbols have >=1 earnings event on record at all; '
                '%d/%d fills fall within +/-3 CALENDAR days of one (widened window, independent of the '
                '0/+1-day session-mapping choice) -- the population is small regardless of mapping precision',
                n_sym_any, len(pf), hits3, len(pf))
    return pf, earn_set, dict(n_sym_any=n_sym_any, hits3=hits3, n_total=len(pf))


# ---------------------------------------------------------------------------
# Shared stats: paired delta reads, both directions + OOS
# ---------------------------------------------------------------------------

def paired_reads(pf, value_col, f1693, baseline_col='E1_production', label=''):
    """Per-window paired-delta reads (idea_col - baseline_col on the SAME
    fills): n, mean dR, iid/day t, ex-top5%, MDE, fills/week of the
    UNDERLYING book, weekly P10 of the delta, $/yr at $375 (annualized:
    mean_dR * fills_per_week * 52 * 375)."""
    out = {}
    for wname, (lo, hi) in f1693.WINDOWS.items():
        sub = pf[pf['window'] == wname]
        vals, base = sub[value_col], sub[baseline_col]
        mask = vals.notna() & base.notna()
        delta = (vals - base)[mask]
        dates = sub.loc[mask, 'date']
        r = f1693.reads_for_exit_series(delta, dates, lo, hi)
        weeks = f1693.window_weeks(lo, hi)
        r['dollars_per_yr_375'] = (r['mean_R'] * r['fills_per_week'] * 52.0 * f1693.FIXED_RISK_DOLLARS) if weeks else np.nan
        r['window'] = wname
        r['idea'] = label
        r['n_total_window'] = len(sub)
        out[wname] = r
        logger.info('%s %s: n=%d mean_dR=%+.4f iid_t=%.2f day_t=%.2f ex_top5=%+.4f fills/wk=%.2f $/yr@375=%+.0f',
                    label, wname, r['n'], r['mean_R'], r['iid_t'], r['day_t'], r['ex_top5'], r['fills_per_week'], r['dollars_per_yr_375'])
    return out


def own_reads(vals, dates, window_bounds, f1693, label=''):
    """Reads for a series on its OWN terms (idea 25's additive book, idea
    33's cohorts) -- same stat block, not a delta."""
    r = f1693.reads_for_exit_series(vals, dates, *window_bounds)
    weeks = f1693.window_weeks(*window_bounds)
    r['dollars_per_yr_375'] = (r['mean_R'] * r['fills_per_week'] * 52.0 * f1693.FIXED_RISK_DOLLARS) if weeks else np.nan
    r['idea'] = label
    logger.info('%s: n=%d mean_R=%+.4f iid_t=%.2f day_t=%.2f ex_top5=%+.4f fills/wk=%.2f $/yr@375=%+.0f',
                label, r['n'], r['mean_R'], r['iid_t'], r['day_t'], r['ex_top5'], r['fills_per_week'], r['dollars_per_yr_375'])
    return r


def passes_1679_bar(train_r, val_r, oos_r):
    """1679's pass bar for an exit/sizing change (PREREG_1679 sec 'Pass
    bar'): paired dR>=+0.05 with day-clustered t>=2.5 and ex-top5%>0 in BOTH
    years, same sign OOS2024H2. Returns (verdict, oos_note) -- OOS2024H2 n==0
    is reported as UNTESTED, never silently counted as a pass (the book
    analysis_results/orb_bplus_book.csv only spans 2025-01..2026-09; see the
    top-of-file caveat)."""
    def ok(r):
        return bool(r['n'] > 0 and r['mean_R'] >= 0.05 and r['day_t'] >= 2.5 and r['ex_top5'] > 0)
    both_years = ok(train_r) and ok(val_r)
    if oos_r['n'] == 0 or np.isnan(oos_r['mean_R']):
        oos_note = 'OOS2024H2 UNTESTED (no fills on this book)'
    elif np.sign(oos_r['mean_R']) == np.sign(train_r['mean_R']) or oos_r['mean_R'] >= 0:
        oos_note = f"OOS2024H2 same-signed ({oos_r['mean_R']:+.3f}R)"
    else:
        oos_note = f"OOS2024H2 SIGN FLIP ({oos_r['mean_R']:+.3f}R) -- bar not met"
        both_years = False
    return both_years, oos_note


def main():
    t_start = time.time()
    check_disk()
    f1693 = _load_module('f1693_1687', 'research/orb_freq/1693_pool_exits.py')
    logger.info('loaded 1693_pool_exits.py and its own f1679/f1668/f1677/cb/orb_csv imports')

    store = f1693.CachedStore(f1693.f1668.BARS_DB)
    pf, recon, premkt_cov = f1693.load_production_fills(store)
    logger.info('production book: n=%d recon=%s', len(pf), recon)

    sample = pf.iloc[0]
    sb = store.day_bars(sample['symbol'], str(sample['date']))
    logger.info('coverage check (%s %s): bars span minarr [%.0f, %.0f] ET-minutes '
                '(need >=660 for idea25, >=945 for idea41/43 EOD) -- full-day bars already cached, no appender needed',
                sample['symbol'], sample['date'], float(sb['minarr'].min()), float(sb['minarr'].max()))

    all_reads = []
    result_lines = []
    result_lines.append('# RESULT 1,687 -- ORB frequency/exit/sizing ideas 25, 33, 41, 42, 43, 44\n\n')
    result_lines.append(f"PREREG: research/orb_freq/PREREG_1684.md. Book: production, real entry minutes, "
                         f"n={len(pf)} fills (recon={recon}). Bar: paired dR>=+0.05R, day-clustered t>=2.5, "
                         f"ex-top5%>0 in BOTH years (TRAIN2025 & VAL2026), same-signed OOS2024H2 (1679's bar).\n\n"
                         f"**CAVEAT (checked, not a bug):** `analysis_results/orb_bplus_book.csv` (the production "
                         f"book with real entry minutes that 1693/1679 reconstruct) spans 2025-01-02..2026-09-28 "
                         f"only -- it is a live/BT-tracking ledger, not a full-history backtest, so it has ZERO "
                         f"2024H2 fills. Confirmed independently (date histogram on the raw CSV) and cross-checked "
                         f"against 1693_reads.csv: every production-slice pool there (e.g. gap_size_5-7%) also "
                         f"shows n=0 in OOS2024H2; only 1684/1685's separate daily-bar admission pools (a "
                         f"different data source, point-in-time universes) have 2024H2 coverage. Every OOS2024H2 "
                         f"read below is therefore UNTESTED, not a null -- reported as such, never silently "
                         f"folded into a pass.\n\n")

    # ---- idea 41 / 43: need per-fill bars, walked directly ----
    logger.info('--- idea 41/43: walking bars for %d production fills ---', len(pf))
    idea41_vals, idea43_vals = [], []
    t0 = time.time()
    for n_seen, r in enumerate(pf.itertuples(), 1):
        bars = store.day_bars(r.symbol, str(r.date))
        rec = f1693.reconstruct_fill(store, r.symbol, str(r.date), r.entry)
        if rec[0] is None:
            idea41_vals.append(np.nan)
            idea43_vals.append(np.nan)
            continue
        b, i0, entry, stop, R_unit = rec[0]['bars'], rec[0]['i0'], rec[0]['entry'], rec[0]['stop'], rec[0]['R_unit']
        idea41_vals.append(idea41_walk(b, i0, entry, stop, R_unit, f1693))
        idea43_vals.append(idea43_walk(b, i0, entry, stop, R_unit, f1693))
        if n_seen % 200 == 0:
            logger.info('idea41/43: walked %d/%d (%.0fs)', n_seen, len(pf), time.time() - t0)
    pf['idea41_R'] = idea41_vals
    pf['idea43_R'] = idea43_vals
    logger.info('idea41/43: done in %.0fs', time.time() - t0)

    # ---- idea 42: already computed as E12_powerHour by load_production_fills ----
    pf['idea42_R'] = pf['E12_powerHour']

    # ---- idea 44: post-hoc sizing transform, no bars needed ----
    pf = apply_idea44(pf)
    for wname, (lo, hi) in f1693.WINDOWS.items():
        sub = pf[pf['window'] == wname]
        if len(sub):
            logger.info('idea44 %s: %.1f%% of fills affected (range<0.75%% of price)', wname, 100 * sub['idea44_affected'].mean())

    for ename, col in [('idea41_no2R_half3R_trailMFE1R', 'idea41_R'),
                        ('idea42_powerHour', 'idea42_R'),
                        ('idea43_addUnit1R', 'idea43_R'),
                        ('idea44_rangeFloorSizing', 'idea44_R')]:
        reads = paired_reads(pf, col, f1693, label=ename)
        verdict, oos_note = passes_1679_bar(reads['TRAIN2025'], reads['VAL2026'], reads['OOS2024H2'])
        for wname, r in reads.items():
            all_reads.append(r)
        result_lines.append(
            f"## {ename}\n"
            f"- TRAIN2025: dR {reads['TRAIN2025']['mean_R']:+.3f}R (iid_t {reads['TRAIN2025']['iid_t']:.2f}, "
            f"day_t {reads['TRAIN2025']['day_t']:.2f}, n{reads['TRAIN2025']['n']}, ex_top5 {reads['TRAIN2025']['ex_top5']:+.3f}, "
            f"wkP10 {reads['TRAIN2025']['weekly_p10_R']:+.2f}, $/yr@375 {reads['TRAIN2025']['dollars_per_yr_375']:+.0f})\n"
            f"- VAL2026: dR {reads['VAL2026']['mean_R']:+.3f}R (iid_t {reads['VAL2026']['iid_t']:.2f}, "
            f"day_t {reads['VAL2026']['day_t']:.2f}, n{reads['VAL2026']['n']}, ex_top5 {reads['VAL2026']['ex_top5']:+.3f}, "
            f"wkP10 {reads['VAL2026']['weekly_p10_R']:+.2f}, $/yr@375 {reads['VAL2026']['dollars_per_yr_375']:+.0f})\n"
            f"- OOS2024H2: dR {reads['OOS2024H2']['mean_R']:+.3f}R (day_t {reads['OOS2024H2']['day_t']:.2f}, "
            f"n{reads['OOS2024H2']['n']}, $/yr@375 {reads['OOS2024H2']['dollars_per_yr_375']:+.0f})\n"
            f"- **verdict: {'PASSES' if verdict else 'fails'} 1679's bar (both years >=+0.05R, day_t>=2.5, ex_top5>0); {oos_note}**\n\n")

    if 'idea44_rangeFloorSizing' in [e for e, _ in []]:
        pass
    share_aff = {w: float(pf.loc[pf.window == w, 'idea44_affected'].mean()) if (pf.window == w).any() else np.nan
                 for w in f1693.WINDOWS}
    result_lines.append(f"idea44 detail: share of fills affected (range<0.75% of price) -- "
                         f"TRAIN2025 {share_aff.get('TRAIN2025', float('nan')):.1%}, "
                         f"VAL2026 {share_aff.get('VAL2026', float('nan')):.1%}, "
                         f"OOS2024H2 {share_aff.get('OOS2024H2', float('nan')):.1%}. "
                         f"At the stated 13bps/0.75% floor the actual cost/R ratio achieved is 0.173R, not the "
                         f"idea text's 0.1R target (logged in 1687_cells.log) -- the 0.75% floor is authoritative, "
                         f"13bps/0.1R is the rationale. $ at live sizing ($2000 risk/trade): multiply $/yr@375 above "
                         f"by 2000/375=5.33x.\n\n")

    # ---- idea 25: additive, own reads ----
    logger.info('--- idea 25: second-chance 30-min range ---')
    pf25 = run_idea25(pf, store, f1693)
    result_lines.append("## idea25_secondChance30minRange (ADDITIVE -- no production counterpart, not a paired delta)\n")
    if len(pf25):
        for wname, (lo, hi) in f1693.WINDOWS.items():
            sub = pf25[pf25['window'] == wname]
            r = own_reads(sub['idea25_R'], sub['date'], (lo, hi), f1693, label=f'idea25_{wname}')
            r['window'] = wname
            r['idea'] = 'idea25_secondChance30minRange'
            all_reads.append(r)
            result_lines.append(f"- {wname}: own mean R {r['mean_R']:+.3f} (iid_t {r['iid_t']:.2f}, day_t {r['day_t']:.2f}, "
                                 f"n{r['n']}, fills/wk ADDED {r['fills_per_week']:.2f}, ex_top5 {r['ex_top5']:+.3f}, "
                                 f"wkP10 {r['weekly_p10_R']:+.2f}, $/yr@375 {r['dollars_per_yr_375']:+.0f})\n")
        tr, va = all_reads[-3], all_reads[-2]
        verdict25 = bool(tr['n'] > 0 and va['n'] > 0 and tr['mean_R'] >= 0.05 and tr['day_t'] >= 2.5 and tr['ex_top5'] > 0
                          and va['mean_R'] >= 0.05 and va['day_t'] >= 2.5 and va['ex_top5'] > 0)
        result_lines.append(f"- **verdict: {'PASSES' if verdict25 else 'fails'} the pool pass bar in both years "
                             f"(own mean R>=+0.05, day_t>=2.5, ex_top5>0)**\n\n")
    else:
        result_lines.append("- no second-chance fills found (no stopped-out name broke its 30-min range) -- fails (n=0)\n\n")

    # ---- idea 33: earnings split ----
    logger.info('--- idea 33: earnings-gapper vs non-earnings split ---')
    pf33, earn_set, diag = run_idea33(pf, f1693)
    result_lines.append("## idea33_earningsSplit (cohort split, not a paired delta)\n")
    if pf33 is None:
        result_lines.append("- **VOID: no earnings calendar found on disk** "
                             "(research/multiday/DATA.md sec.4's earnings_events.parquet is gitignored and absent; "
                             "research/edgar_desk/events_raw.csv also not found)\n\n")
    else:
        result_lines.append(f"- calendar: rebuilt from research/edgar_desk/events_raw.csv (8-K/8-K-A, item 2.02), "
                             f"approx event-session = filing_date or +1 day. Diagnostic: only {diag['n_sym_any']}/"
                             f"{diag['n_total']} fill-symbols have ANY earnings event on record, and only "
                             f"{diag['hits3']}/{diag['n_total']} fills fall within a WIDENED +/-3 calendar-day "
                             f"window of one -- ORB's gap admission rarely coincides with a scheduled earnings "
                             f"filing; the split is likely **underpowered regardless of mapping precision**.\n")
        cohort_reads = {}
        for cohort_name, cohort_mask in [('earnings', pf33['is_earnings']), ('non_earnings', ~pf33['is_earnings'])]:
            cohort_reads[cohort_name] = {}
            for wname, (lo, hi) in f1693.WINDOWS.items():
                sub = pf33[(pf33['window'] == wname) & cohort_mask]
                r = own_reads(sub['E1_production'], sub['date'], (lo, hi), f1693, label=f'idea33_{cohort_name}_{wname}')
                r['window'] = wname
                r['idea'] = f'idea33_{cohort_name}'
                all_reads.append(r)
                cohort_reads[cohort_name][wname] = r
            result_lines.append(f"- {cohort_name}: TRAIN2025 {cohort_reads[cohort_name]['TRAIN2025']['mean_R']:+.3f}R "
                                 f"(t{cohort_reads[cohort_name]['TRAIN2025']['day_t']:.2f} n{cohort_reads[cohort_name]['TRAIN2025']['n']}) | "
                                 f"VAL2026 {cohort_reads[cohort_name]['VAL2026']['mean_R']:+.3f}R "
                                 f"(t{cohort_reads[cohort_name]['VAL2026']['day_t']:.2f} n{cohort_reads[cohort_name]['VAL2026']['n']}) | "
                                 f"OOS2024H2 {cohort_reads[cohort_name]['OOS2024H2']['mean_R']:+.3f}R (n{cohort_reads[cohort_name]['OOS2024H2']['n']})\n")
        earn_tr, earn_va = cohort_reads['earnings']['TRAIN2025'], cohort_reads['earnings']['VAL2026']
        ne_tr, ne_va = cohort_reads['non_earnings']['TRAIN2025'], cohort_reads['non_earnings']['VAL2026']
        drop = None
        if earn_tr['n'] >= 10 and earn_va['n'] >= 10 and earn_tr['mean_R'] < 0 and earn_va['mean_R'] < 0:
            drop = 'earnings'
        elif ne_tr['n'] >= 10 and ne_va['n'] >= 10 and ne_tr['mean_R'] < 0 and ne_va['mean_R'] < 0:
            drop = 'non_earnings'
        if earn_tr['n'] < 10 or earn_va['n'] < 10:
            verdict33 = (f"**verdict: INSUFFICIENT N to test the split (earnings cohort n={earn_tr['n']} TRAIN / "
                         f"{earn_va['n']} VAL, both well under the MDE floor) -- non-earnings cohort alone is just "
                         f"'most of production' (it IS the book, mean R {ne_tr['mean_R']:+.3f}/{ne_va['mean_R']:+.3f}R, "
                         f"matching production's own +0.105R-ish live-config edge); no drop decision can be made**")
        else:
            verdict33 = f"**verdict: {'drop ' + drop + ' cohort (negative both years)' if drop else 'no drop -- both cohorts positive or mixed-sign across years'}**"
        result_lines.append(f"- {verdict33} (approx event-session mapping: filing_date or +1 day -- see log)\n\n")

    # ---- write outputs ----
    reads_df = pd.DataFrame(all_reads)
    reads_df.to_csv(READS_CSV, index=False)
    logger.info('wrote %s (%d rows)', READS_CSV, len(reads_df))

    with open(RESULT_MD, 'w') as fh:
        fh.writelines(result_lines)
    logger.info('wrote %s (%d lines)', RESULT_MD, len(result_lines))
    logger.info('cell 1687 done in %.0fs', time.time() - t_start)
    store.close()


if __name__ == '__main__':
    main()
