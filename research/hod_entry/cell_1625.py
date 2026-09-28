"""Cell 1,625 -- the index (SPY/IWM) as the instrument on a break-count burst. PREREG_1623.md
lines 27-34 (FROZEN 2026-09-28). Delegated build; owner ask was "improve 0.2R... different
thinking" -- this is the index-leg frame: trade SPY/IWM off a market-wide breadth burst, not the
individual HOD-break fill.

Signal: B30(all names) -- the count of ARM events across the whole candidate universe in the
trailing 30 minutes -- crosses into its TRAIN-H2 top decile for the first time that day, at
minute m*. Trade: buy SPY (and, second leg, IWM) at the open of minute m*+1, exit at the open of
minute m*+61 (60-min hold) or 15:55 ET, whichever first; cost 2 bps round trip; one signal/day.

*** DATA-AVAILABILITY DEVIATION (read before trusting a number) ***
PREREG_1623.md line 22: "B30(f) = the number of ARM events (ANY STATUS, all names)". Verified at
runtime: causal_arming_causal.csv carries a timestamp (fill_min) ONLY for status=='fill' rows;
'nofill' and 'not_armed' rows have no time column anywhere on disk (checked causal_arming_causal.csv,
causal_arming_tick_rv.csv, features_1478_A.csv, bars_fills_1478.db -- all four either lack a time
field for non-fills or are scoped to the 9,911 fills only). Re-deriving nofill arm-crossing times
would mean re-walking causal_arming.py's armed_crossing_bars() against raw bars for the FULL daily
universe (every candidate symbol, every day, with its own ADV20/floor) -- out of scope for this cell.
So B30 here is built from FILL-event arm times only (features_1478_A.csv's arm_m, the 9,911 fills'
own arming minute, NOT fill_min): B30_FILL_PROXY(day, t) = count of that day's FILLED arms with
arm_m in (t-30, t]. This is a genuine causality weakening, not just an undercount: at the real arm
moment you cannot yet know whether an order will end up 'fill' or 'nofill', so conditioning on
eventual fill status is information a live system would not have. Flagged, not smoothed over --
see RESULT_1625.md caveats. A PASS on this proxy is evidence for the DRY instrument only, not a
clean test of the literal 'any status' spec.
"""
import argparse
import datetime as dt
import logging
import os
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import cell_1445 as h1445  # day_clustered_t, ex_top5_mean, weeks_spanned

CAUSAL_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
FEATURES_A_CSV = os.path.join(HERE, 'features_1478_A.csv')
INDEX_PARQUET = os.path.join(HERE, 'index_bars_1625.parquet')
SIGNALS_CSV = os.path.join(HERE, 'cell_1625_signals.csv')
RESULT_MD = os.path.join(HERE, 'RESULT_1625.md')

ET = ZoneInfo('America/New_York')
SESSION_START_M = 570   # 09:30
SESSION_LAST_M = 959    # last valid bar-open minute (15:59)
CUTOFF_M = 955           # 15:55 forced exit
WINDOW = 30              # B30 window (minutes)
DECILE_PCT = 90.0
COST_BPS = 2.0            # round trip
WINNER_CAP_BPS = 100.0    # +1%
FILL_TOLERANCE_M = 5      # "next bar's open under a cap" -- obtainability rule
PLACEBO_SEED = 1625

BAR_MEAN_BPS = 8.0
BAR_T = 2.5
BAR_PLACEBO_MARGIN_BPS = 5.0
BAR_PLACEBO_T = 2.0
BAR_SIGNALS_PER_WEEK = 2.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('cell_1625')


# --------------------------------------------------------------------------------------------
# Step 1: base fills + arm minute (the B30_FILL_PROXY input)
# --------------------------------------------------------------------------------------------

def load_fills_with_arm():
    """The 9,911 fills of causal_arming_causal.csv, each carrying its own arm_m (features_1478_A.csv)
    -- the actual crossing/arm minute, distinct from (and <=) fill_min. holdout = VAL if split==VAL
    else TRAIN-H2 (identical convention to cell_1623.py / cell_1445.py)."""
    c = pd.read_csv(CAUSAL_CSV, low_memory=False)
    c = c[c.status == 'fill'][['day', 'symbol', 'split', 'half', 'fill_min']].copy()
    log.info('causal_arming_causal.csv fills: %d rows', len(c))
    a = pd.read_csv(FEATURES_A_CSV, usecols=['day', 'symbol', 'fill_min', 'arm_m'])
    log.info('features_1478_A.csv (arm_m source): %d rows', len(a))
    m = c.merge(a, on=['day', 'symbol', 'fill_min'], how='inner')
    lost = len(c) - len(m)
    if lost:
        log.warning('%d/%d fills DROPPED on the (day,symbol,fill_min) join to features_1478_A.csv '
                    '(no arm_m match) -- excluded, not imputed', lost, len(c))
    else:
        log.info('OK: all %d fills matched an arm_m (0 dropped)', len(m))
    m['holdout'] = np.where(m['split'] == 'VAL', 'VAL', 'TRAIN-H2')
    bad_half = m[(m['split'] == 'TRAIN') & (m['half'] != 'H2')]
    if len(bad_half):
        log.warning('%d TRAIN rows have half != H2 -- excluded (spec is TRAIN-H2 only)', len(bad_half))
        m = m.drop(bad_half.index)
    return m[['day', 'symbol', 'holdout', 'arm_m']].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Step 2: B30(day, minute) grid + TRAIN-H2 top-decile threshold + first-crossing minute m*
# --------------------------------------------------------------------------------------------

def build_b30_grid(fills):
    """Returns {day: np.array of B30 values over MINUTE_GRID} using ONLY that day's own arm_m
    (all symbols). Window: arm_m in (t-30, t] -- 'the 30 minutes before' t, inclusive of t."""
    grid = np.arange(SESSION_START_M, SESSION_LAST_M + 1)  # 570..959
    out = {}
    for day, sub in fills.groupby('day'):
        arm = np.sort(sub['arm_m'].to_numpy())
        hi = np.searchsorted(arm, grid, side='right')
        lo = np.searchsorted(arm, grid - WINDOW, side='right')
        out[day] = (hi - lo).astype(int)
    return grid, out


def train_h2_decile_threshold(grid, b30_by_day, day_holdout):
    pooled = np.concatenate([b30_by_day[d] for d in b30_by_day if day_holdout.get(d) == 'TRAIN-H2'])
    tau = float(np.percentile(pooled, DECILE_PCT))
    frac_at_or_above = float((pooled >= tau).mean())
    log.info('TRAIN-H2 pooled (day,minute) B30 values: n=%d, tau(p%.0f)=%.3f, empirical share >= tau = %.1f%% '
             '(B30 max=%.0f, mean=%.2f)', len(pooled), DECILE_PCT, tau, frac_at_or_above * 100,
             pooled.max(), pooled.mean())
    return tau


def first_crossing(grid, b30_arr, tau):
    idx = np.argmax(b30_arr >= tau)
    if b30_arr[idx] < tau:
        return None, None
    return int(grid[idx]), float(b30_arr[idx])


# --------------------------------------------------------------------------------------------
# Step 3: SPY/IWM minute-open lookup with a forward-search fill tolerance (obtainability rule:
# "reachable by an order the engine would have had resting: next bar's open under a cap")
# --------------------------------------------------------------------------------------------

def load_open_lookup():
    df = pd.read_parquet(INDEX_PARQUET)
    ts = pd.to_datetime(df['t'], utc=True).dt.tz_convert(ET)
    df['minute'] = ts.dt.hour * 60 + ts.dt.minute
    df['day'] = ts.dt.strftime('%Y-%m-%d')
    lut = {}
    for (symbol, day), sub in df.groupby(['symbol', 'day']):
        sub = sub.sort_values('minute')
        lut[(symbol, day)] = (sub['minute'].to_numpy(), sub['o'].to_numpy())
    log.info('index_bars_1625.parquet: %d (symbol,day) keys loaded for open lookup', len(lut))
    return lut


class Excl:
    """Counts exclusions by reason -- 'exclusions counted, never imputed'."""
    def __init__(self):
        self.c = {}

    def bump(self, reason):
        self.c[reason] = self.c.get(reason, 0) + 1

    def report(self):
        for k, v in sorted(self.c.items()):
            log.warning('exclusion: %s = %d', k, v)


def open_at_or_after(lut, symbol, day, target_m, excl, tag, tol=FILL_TOLERANCE_M):
    key = (symbol, day)
    if key not in lut:
        excl.bump(f'{tag}:no_bars_for_symbol_day')
        return None
    minutes, opens = lut[key]
    pos = np.searchsorted(minutes, target_m, side='left')
    if pos >= len(minutes) or minutes[pos] > target_m + tol:
        excl.bump(f'{tag}:no_bar_within_{tol}m')
        return None
    return float(opens[pos])


def ret_bps(entry_open, exit_open):
    if entry_open is None or exit_open is None or entry_open <= 0:
        return None
    return (exit_open / entry_open - 1.0) * 1e4 - COST_BPS


def build_trade(lut, symbol, day, m, hold, excl, tag):
    """m -> entry at open(m+1), exit at open(min(m+hold, CUTOFF_M)), 'next bar' tolerance."""
    entry_target = m + 1
    exit_target = min(m + hold, CUTOFF_M)
    if entry_target > SESSION_LAST_M:
        excl.bump(f'{tag}:entry_past_session_end')
        return None
    eo = open_at_or_after(lut, symbol, day, entry_target, excl, tag + ':entry')
    xo = open_at_or_after(lut, symbol, day, exit_target, excl, tag + ':exit')
    return ret_bps(eo, xo)


# --------------------------------------------------------------------------------------------
# Step 4: stats helpers
# --------------------------------------------------------------------------------------------

def capped_mean_bps(y, cap=WINNER_CAP_BPS):
    y = pd.Series(y).dropna()
    return float(np.minimum(y, cap).mean()) if len(y) else float('nan')


def t_iid(y):
    y = pd.Series(y).dropna()
    n = len(y)
    if n < 2 or y.std(ddof=1) == 0:
        return float('nan')
    return float(y.mean() / (y.std(ddof=1) / np.sqrt(n)))


def holdout_stats(sub, col, days_col='day'):
    y = sub[col].dropna()
    n = len(y)
    if n == 0:
        return dict(n=0, mean=float('nan'), t_dc=float('nan'), t_iid=float('nan'),
                    ex_top5=float('nan'), capped=float('nan'))
    return dict(
        n=n,
        mean=float(y.mean()),
        t_dc=h1445.day_clustered_t(y, sub.loc[y.index, days_col]),
        t_iid=t_iid(y),
        ex_top5=h1445.ex_top5_mean(y),
        capped=capped_mean_bps(y),
    )


def placebo_margin_stats(sub, real_col, placebo_col, days_col='day'):
    d = sub[[days_col, real_col, placebo_col]].dropna()
    if not len(d):
        return dict(n=0, real_mean=float('nan'), placebo_mean=float('nan'), margin=float('nan'), t=float('nan'))
    diff = d[real_col] - d[placebo_col]
    return dict(
        n=len(d),
        real_mean=float(d[real_col].mean()),
        placebo_mean=float(d[placebo_col].mean()),
        margin=float(diff.mean()),
        t=h1445.day_clustered_t(diff, d[days_col]),
    )


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.parse_args()

    fills = load_fills_with_arm()
    day_holdout = fills.drop_duplicates('day').set_index('day')['holdout'].to_dict()
    log.info('days with >=1 fill: %d (TRAIN-H2 %d, VAL %d)', len(day_holdout),
             sum(v == 'TRAIN-H2' for v in day_holdout.values()), sum(v == 'VAL' for v in day_holdout.values()))

    grid, b30_by_day = build_b30_grid(fills)
    tau = train_h2_decile_threshold(grid, b30_by_day, day_holdout)

    lut = load_open_lookup()
    excl = Excl()
    rng = np.random.default_rng(PLACEBO_SEED)

    rows = []
    n_no_signal = 0
    for day in sorted(b30_by_day):
        holdout = day_holdout.get(day)
        if holdout is None:
            continue
        m_star, b30_val = first_crossing(grid, b30_by_day[day], tau)
        if m_star is None:
            n_no_signal += 1
            continue
        placebo_m = int(rng.integers(SESSION_START_M, SESSION_LAST_M))

        row = dict(day=day, split=holdout, m_star=m_star, B30=b30_val)
        row['spy_ret_bps_30'] = build_trade(lut, 'SPY', day, m_star, 31, excl, 'spy30')
        row['spy_ret_bps_60'] = build_trade(lut, 'SPY', day, m_star, 61, excl, 'spy60')
        row['spy_ret_bps_120'] = build_trade(lut, 'SPY', day, m_star, 121, excl, 'spy120')
        row['iwm_ret_bps_30'] = build_trade(lut, 'IWM', day, m_star, 31, excl, 'iwm30')
        row['iwm_ret_bps_60'] = build_trade(lut, 'IWM', day, m_star, 61, excl, 'iwm60')
        row['iwm_ret_bps_120'] = build_trade(lut, 'IWM', day, m_star, 121, excl, 'iwm120')
        row['placebo_m'] = placebo_m
        row['placebo_ret_bps'] = build_trade(lut, 'SPY', day, placebo_m, 61, excl, 'placebo_spy60')
        row['placebo_ret_bps_iwm'] = build_trade(lut, 'IWM', day, placebo_m, 61, excl, 'placebo_iwm60')
        rows.append(row)

    log.info('signal days: %d, no-signal days (B30 never reached tau): %d', len(rows), n_no_signal)
    excl.report()

    sig = pd.DataFrame(rows)
    cols = ['day', 'split', 'm_star', 'B30', 'spy_ret_bps_60', 'iwm_ret_bps_60', 'placebo_ret_bps',
            'spy_ret_bps_30', 'spy_ret_bps_120', 'iwm_ret_bps_30', 'iwm_ret_bps_120',
            'placebo_ret_bps_iwm', 'placebo_m']
    sig = sig[cols]
    sig.to_csv(SIGNALS_CSV, index=False)
    log.info('wrote %s: %d rows', SIGNALS_CSV, len(sig))

    # -------------------------------------------------------------------------------------
    # stats per holdout x leg x hold-length
    # -------------------------------------------------------------------------------------
    holdouts = ['TRAIN-H2', 'VAL']
    legs = {'spy_60': 'spy_ret_bps_60', 'iwm_60': 'iwm_ret_bps_60',
            'spy_30': 'spy_ret_bps_30', 'spy_120': 'spy_ret_bps_120',
            'iwm_30': 'iwm_ret_bps_30', 'iwm_120': 'iwm_ret_bps_120'}
    stats = {}
    for h in holdouts:
        sub = sig[sig.split == h]
        weeks = h1445.weeks_spanned([d for d, ho in day_holdout.items() if ho == h])
        for leg, col in legs.items():
            st = holdout_stats(sub, col)
            st['weeks'] = weeks
            st['per_week'] = st['n'] / weeks if weeks else float('nan')
            stats[(h, leg)] = st
        stats[(h, 'spy_60_placebo')] = placebo_margin_stats(sub, 'spy_ret_bps_60', 'placebo_ret_bps')
        stats[(h, 'iwm_60_placebo')] = placebo_margin_stats(sub, 'iwm_ret_bps_60', 'placebo_ret_bps_iwm')

    for (h, leg), st in stats.items():
        log.info('%-9s %-16s %s', h, leg, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()})

    # -------------------------------------------------------------------------------------
    # pass bar (graded on VAL, SPY 60-min leg, per PREREG_1623.md lines 31-34)
    # -------------------------------------------------------------------------------------
    val = stats[('VAL', 'spy_60')]
    val_pl = stats[('VAL', 'spy_60_placebo')]
    th2 = stats[('TRAIN-H2', 'spy_60')]

    checks = [
        ('mean net (VAL, SPY 60-min) >= +%.0f bps' % BAR_MEAN_BPS, val['mean'], val['mean'] >= BAR_MEAN_BPS if not np.isnan(val['mean']) else False),
        ('day-clustered t (VAL) >= %.1f' % BAR_T, val['t_dc'], (not np.isnan(val['t_dc'])) and val['t_dc'] >= BAR_T),
        ('ex-top-5% mean (VAL) > 0', val['ex_top5'], (not np.isnan(val['ex_top5'])) and val['ex_top5'] > 0),
        ('placebo margin (VAL) >= +%.0f bps' % BAR_PLACEBO_MARGIN_BPS, val_pl['margin'], (not np.isnan(val_pl['margin'])) and val_pl['margin'] >= BAR_PLACEBO_MARGIN_BPS),
        ('placebo margin t (VAL) >= %.1f' % BAR_PLACEBO_T, val_pl['t'], (not np.isnan(val_pl['t'])) and val_pl['t'] >= BAR_PLACEBO_T),
        ('signals/week (VAL) >= %.0f' % BAR_SIGNALS_PER_WEEK, val['per_week'], (not np.isnan(val['per_week'])) and val['per_week'] >= BAR_SIGNALS_PER_WEEK),
        ('TRAIN-H2 same sign as VAL', (th2['mean'], val['mean']),
         (not np.isnan(th2['mean'])) and (not np.isnan(val['mean'])) and np.sign(th2['mean']) == np.sign(val['mean']) and val['mean'] != 0),
    ]
    overall = all(c[2] for c in checks)
    verdict = 'PASS' if overall else 'FAIL'
    log.info('=== PASS BAR VERDICT: %s ===', verdict)
    for name, val_, ok in checks:
        log.info('  [%s] %s (%s)', 'x' if ok else ' ', name, val_)

    # time-of-day diagnostic (report-only -- the confound named for 1624 applies structurally to
    # 1625 too since B30 rises through the morning; NOT used to gate anything)
    tod = sig.assign(hhmm=sig.m_star.apply(lambda m: f'{m // 60:02d}:{m % 60:02d}'))
    tod_counts = tod.groupby(sig.split)['m_star'].agg(['count', 'min', 'median', 'max'])

    write_result_md(sig, stats, checks, overall, verdict, tau, day_holdout, n_no_signal, excl, tod_counts)
    log.info('wrote %s', RESULT_MD)
    return 0 if overall else 1


def write_result_md(sig, stats, checks, overall, verdict, tau, day_holdout, n_no_signal, excl, tod_counts):
    def fmt(x, nd=2):
        return 'nan' if (x is None or (isinstance(x, float) and np.isnan(x))) else f'{x:.{nd}f}'

    lines = []
    lines.append('# RESULT — Cell 1,625: the index (SPY/IWM) as the instrument\n')
    lines.append(f'PREREG_1623.md lines 27-34 (FROZEN). Verdict: **{verdict}**.\n')

    lines.append('## CAVEAT — read first (data-availability deviation from the literal spec)\n')
    lines.append(
        "PREREG says B30 = ARM events of **any status**. causal_arming_causal.csv, "
        "causal_arming_tick_rv.csv, features_1478_A.csv and bars_fills_1478.db were all inspected; "
        "none carries a timestamp for 'nofill'/'not_armed' rows (verified: bars_fills_1478.db and "
        "features_1478_A.csv are each scoped to exactly the 9,911 fills, 0 nofill symbol-days). "
        "B30 here is **B30_FILL_PROXY**: built from the 9,911 fills' own arm_m (features_1478_A.csv), "
        "not fill_min. This is a real causality weakening (at the true arm moment you do not yet "
        "know whether the order will fill), not just an undercount — a live system's true 'any "
        "status' B30 would run higher and possibly cross the decile earlier. A PASS below is "
        "evidence for the dry-instrument path only, not a clean test of the frozen spec; a true "
        "rebuild needs a full-universe crossing re-derivation from causal_arming.py's arm logic, "
        "out of scope for this cell.\n")
    lines.append(
        f"Separately: the task's assumption that data/cache.db covers SPY *and* IWM through "
        f"2026-03-20 was checked and found wrong for IWM — cache.db had only 1,094 IWM rows "
        f"(2025-04-07..09). IWM was pulled from Alpaca SIP for effectively the full 230-day range "
        f"(226 days), SPY for the 48 days cache.db was missing (mostly 2026-03-21..05-29, plus "
        f"scattered earlier gaps); both now show 230/230 required days with 0 lost "
        f"(research/hod_entry/fetch_index_bars_1625.py log). Source is 'cache' vs 'alpaca' per row "
        f"in index_bars_1625.parquet.\n")

    lines.append('## Data\n')
    lines.append(f'- Base fills + arm minute: causal_arming_causal.csv (status==fill, n=9,911) INNER '
                 f'JOIN features_1478_A.csv on (day,symbol,fill_min) for arm_m. holdout = VAL if '
                 f'split==VAL else TRAIN-H2 (n_days TRAIN-H2={sum(v=="TRAIN-H2" for v in day_holdout.values())}, '
                 f'VAL={sum(v=="VAL" for v in day_holdout.values())}).\n')
    lines.append(f'- SPY/IWM minute bars: research/hod_entry/index_bars_1625.parquet (built by '
                 f'fetch_index_bars_1625.py from data/cache.db READ-ONLY + Alpaca SIP).\n')
    lines.append(f'- TRAIN-H2 top-decile threshold tau = **{tau:.3f}** (B30 >= tau counts as "top decile"; '
                 f'pooled over every (day, minute) pair, minute grid 09:30-15:59 ET, TRAIN-H2 days only).\n')
    lines.append(f'- Signal days found: {len(sig)}; days with a fill but B30 never reached tau: {n_no_signal}.\n')
    if excl.c:
        lines.append('- Trade-construction exclusions (bar not found within the {}-minute tolerance, '
                     'never imputed):\n'.format(FILL_TOLERANCE_M))
        for k, v in sorted(excl.c.items()):
            lines.append(f'  - {k}: {v}\n')

    lines.append('\n## Time-of-day of m* (report-only — the confound PREREG names for 1,624 applies '
                 'structurally to 1,625 too, since B30 rises through the morning; not hour-adjusted '
                 'here because the 1,625 prose does not call for it, and thresholds cannot be tuned '
                 'post-hoc)\n\n')
    lines.append('| split | n | first m* (min) | median m* | last m* |\n|---|---|---|---|---|\n')
    for h, r in tod_counts.iterrows():
        def hhmm(m):
            m = int(m)
            return f'{m // 60:02d}:{m % 60:02d}'
        lines.append(f"| {h} | {int(r['count'])} | {hhmm(r['min'])} | {hhmm(r['median'])} | {hhmm(r['max'])} |\n")

    lines.append('\n## Per-holdout stats — primary (60-minute hold)\n\n')
    lines.append('| holdout | leg | n | mean net bps | t (day-clust) | t (iid) | ex-top-5% | '
                 'winner-capped +1% | signals/wk |\n|---|---|---|---|---|---|---|---|---|\n')
    for h in ['TRAIN-H2', 'VAL']:
        for leg, label in [('spy_60', 'SPY'), ('iwm_60', 'IWM')]:
            st = stats[(h, leg)]
            lines.append(f"| {h} | {label} | {st['n']} | {fmt(st['mean'])} | {fmt(st['t_dc'])} | "
                         f"{fmt(st['t_iid'])} | {fmt(st['ex_top5'])} | {fmt(st['capped'])} | "
                         f"{fmt(st['per_week'])} |\n")

    lines.append('\n## Placebo (random minute, same day, seed 1625) — 60-minute hold\n\n')
    lines.append('| holdout | leg | n | real mean | placebo mean | margin | t (margin) |\n|---|---|---|---|---|---|---|\n')
    for h in ['TRAIN-H2', 'VAL']:
        for leg, label in [('spy_60_placebo', 'SPY'), ('iwm_60_placebo', 'IWM')]:
            st = stats[(h, leg)]
            lines.append(f"| {h} | {label} | {st['n']} | {fmt(st['real_mean'])} | {fmt(st['placebo_mean'])} | "
                         f"{fmt(st['margin'])} | {fmt(st['t'])} |\n")

    lines.append('\n## 30- / 120-minute holds (report-only, SPY + IWM, not pass-bar-graded)\n\n')
    lines.append('| holdout | leg | n | mean net bps | t (day-clust) | ex-top-5% |\n|---|---|---|---|---|---|\n')
    for h in ['TRAIN-H2', 'VAL']:
        for leg, label in [('spy_30', 'SPY 30m'), ('spy_120', 'SPY 120m'), ('iwm_30', 'IWM 30m'), ('iwm_120', 'IWM 120m')]:
            st = stats[(h, leg)]
            lines.append(f"| {h} | {label} | {st['n']} | {fmt(st['mean'])} | {fmt(st['t_dc'])} | {fmt(st['ex_top5'])} |\n")

    lines.append('\n## Pass-bar checklist (graded on VAL, SPY 60-min leg — PREREG_1623.md line 33-34)\n\n')
    lines.append('| check | value | pass |\n|---|---|---|\n')
    for name, v, ok in checks:
        vs = v if not isinstance(v, tuple) else f'TRAIN-H2={fmt(v[0])}, VAL={fmt(v[1])}'
        if isinstance(v, float):
            vs = fmt(v)
        lines.append(f"| {name} | {vs} | {'PASS' if ok else 'FAIL'} |\n")
    lines.append(f'\n**Overall: {verdict}** ({sum(c[2] for c in checks)}/{len(checks)} checks passed).\n')

    lines.append('\n## Consequence (per PREREG line 46-48)\n')
    if overall:
        lines.append('PASS on the fill-proxy B30 — per the frozen spec this would route to an index leg '
                     'as a DRY instrument, but given the causality caveat above (fill-only arm times), '
                     'this should be treated as provisional: re-run against a true any-status B30 before '
                     'committing to a dry run.\n')
    else:
        lines.append('FAIL — this frame closes on this population as tested. Note the proxy caveat above: '
                     'a FAIL here does not by itself refute the literal any-status spec, since the '
                     'signal tested is a weaker (fill-only) proxy for it; nothing on this population '
                     'says the any-status version would behave the same, better, or worse.\n')

    lines.append('\n## Not allowed (honored)\n')
    lines.append('Decile threshold (90th pct), hold length (60 min), cost (2 bps) and tolerance (5 min) were '
                 'set from the frozen PREREG/task text before any number was computed and were not adjusted '
                 'after seeing results.\n')

    with open(RESULT_MD, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    sys.exit(main())
