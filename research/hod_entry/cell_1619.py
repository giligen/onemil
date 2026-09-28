#!/usr/bin/env python3
"""Cell 1,619/1,620 -- research/hod_entry/PREREG_1617.md Frame B: burst fade (FROZEN 2026-09-28).

Owner 9/28: "I won't let you go till you find a profitable HOD angle." Rule of this loop: a new
MECHANISM per pass, PREREG + rebuild + refuters. This is the BUILDER for Frame B.

Mechanism (short the chasers who lift the offer above the HOD break level, cover on the retest):
at the base fill instant (cell 1,438's 9,911 causal fills) a SELL LIMIT rests at level * 1.0015
(the same 15 bps as the base long's buy cap -- we ARE the offer the chasers lift); filled AT THE
LIMIT at the first tape print STRICTLY ABOVE it within the fill minute and the next 2 minutes
(through-print rule); no such print -> no trade (counted, not an error). Once short: cover at a
resting BID filled at the first print strictly below it (1,619: level - $0.01; 1,620 report-only:
level * (1 - 0.20%)); stop = level * 1.0075 (a print/bar >= stop -> stopped, mirroring the base's
stop-limit standard cost -- SLIP_STOP_BPS from cell_1478.py, expected-value tail included); 15:55
cover at the ask (EOD_ASK_BPS fallback when no live ask is observable that late); shortable
excluded via borrow_flags.csv; borrow 3%/yr pro rata on the holding period.

Data path: tape (sip_cache_1481/, sip_cache_1480/ -- per-minute SYMBOL_DAY_M.pkl files, the
population's own retest-cell cache, covering the fill minute and up to 15 minutes after) resolves
entry/cover/stop minute-by-minute while a minute is cached; bars_fills_1478.db (full-day 1-minute
OHLC) fills every gap inside that window AND carries the walk from +15 min to 15:55, mirroring
sip_rebuild.walk_path's tie-break (stop wins on a bar/print that could satisfy both) and EOD-at-
minute-955 convention. A run of >=3 consecutive missing RTH minutes in the bar store is counted as
a halt candidate; the position is simply marked at the next print/bar (the reopen), per the PREREG.

Units: R_f = stop - entry (a SHORT's risk distance, positive since stop > entry). net_R_f and
net_pct (of entry price) are reported; ex_top5_mean and day_clustered_t come from cell_1445.py's
exact formulas (reimplemented verbatim below, attributed, to avoid pulling in cell_1445's own
heavier import graph -- research.hod_consol.run_consol etc. -- for two one-screen functions).

Usage:
    python3 research/hod_entry/cell_1619.py --smoke 200      # smoke: first 200 base fills
    python3 research/hod_entry/cell_1619.py                  # full 9,911-fill book

Outputs: research/hod_entry/cell_1619_fills.csv (cell, split, day, symbol, filled, entry, exit,
why, net_Rf, net_pct) and research/hod_entry/RESULT_1619.md.

Not allowed (PREREG "Not allowed"): moving the offer/cover levels, the stop, or the window after a
number is seen; selecting among variants on VAL.
"""
import argparse
import datetime as dt
import os
import pickle
import sqlite3
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

# Reused verbatim, not reimplemented: the PREREG names this file as the stop-limit standard cost
# source. Import only the constant (no heavy pipeline runs at import time -- cell_1478.py's module
# body is function/constant definitions only, guarded by `if __name__ == '__main__'`).
from research.hod_entry.cell_1478 import SLIP_STOP_BPS          # noqa: E402

ET = ZoneInfo('America/New_York')

# --------------------------------------------------------------------------------------------- paths
CAUSAL_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
MODEL_CSV = os.path.join(HERE, 'model_1478_L3_predictions.csv')
FEATURES_A_CSV = os.path.join(HERE, 'features_1478_A.csv')
BARS_DB = os.path.join(HERE, 'bars_fills_1478.db')
BORROW_CSV = os.path.join(REPO, 'research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv')
SIP_DIRS = [os.path.join(HERE, 'sip_cache_1481'), os.path.join(HERE, 'sip_cache_1480')]
FILLS_CSV = os.path.join(HERE, 'cell_1619_fills.csv')
RESULT_MD = os.path.join(HERE, 'RESULT_1619.md')

# --------------------------------------------------------------------------------------- mechanism
OFFER_BPS = 0.0015              # SELL LIMIT at level * (1 + OFFER_BPS) -- the base's 15bps buy cap
COVER_ABS_1619 = 0.01           # 1,619: cover BID at level - $0.01
COVER_PCT_1620 = 0.0020         # 1,620 report-only: cover BID at level * (1 - 0.20%)
STOP_BPS = 0.0075               # stop at level * (1 + STOP_BPS)
ENTRY_WINDOW_MIN = 2            # through-print rule: fill minute + next 2 minutes
TAPE_WINDOW_MIN = 15            # population's cached tape horizon (the retest cells)
EOD_M = 955                     # 15:55 ET -- sip_rebuild.py's OPEN_M/EOD_M convention (570, 955)
HALT_GAP_MIN = 3                # >=3 consecutive missing RTH bar-minutes = halt candidate
BORROW_APY = 0.03               # 3%/yr pro rata on the holding period
# CLAUDE.md standing cost convention: "EOD at the bid/ask {TRAIN: 11.5, VAL: 9.7} bps" -- the cost
# of crossing to the ask when no live quote is observable this late past the fill.
EOD_ASK_BPS = {'TRAIN': 11.5, 'VAL': 9.7}

PASS_BAR = dict(mean_net_Rf=0.15, mean_net_pct=0.10, t=2.5, min_fills_wk=3.0, median_Rf_pct=0.5)


def log(msg):
    """Verbose progress line, flushed immediately (print() is buffered otherwise)."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ================================================================================================ time
def et_ns(day, seconds):
    """UTC epoch ns of `seconds` after ET midnight on `day` -- sip_rebuild.py's et_ns, verbatim."""
    d = dt.date.fromisoformat(day)
    base = dt.datetime(d.year, d.month, d.day, tzinfo=ET)
    return int((base + dt.timedelta(seconds=float(seconds))).timestamp() * 1e9)


def minute_start_ns(day, minute):
    return et_ns(day, minute * 60)


def ns_to_et_minutes(ns, day):
    """sip_rebuild.py's ns_to_et_minutes, verbatim."""
    return (ns - et_ns(day, 0)) / 60e9


# ============================================================================== cell_1445.py helpers
# Reimplemented verbatim from research/hod_entry/cell_1445.py (attributed) rather than imported, to
# avoid pulling in that module's own import graph (research.hod_consol.run_consol, causal_arming)
# for two one-screen statistics functions.
def day_clustered_t(y, day):
    """statsmodels OLS on a constant, clustered by day -- the t-stat on the mean. cell_1445.py."""
    import statsmodels.api as sm
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
    """Mean excluding the top 5% (by value) of a series -- tail-dependence check. cell_1445.py."""
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return np.nan
    k = int(round(0.05 * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def weeks_spanned(days):
    """Distinct ISO (year, week) count over a day-string series. cell_1445.py."""
    iso = pd.to_datetime(pd.Series(days).unique())
    wk = {(d.isocalendar()[0], d.isocalendar()[1]) for d in iso}
    return max(len(wk), 1)


# ================================================================================================ IO
def load_shortable():
    """symbol -> shortable bool. Checked 2026-09-28: borrow_flags.csv columns are symbol, tradable,
    shortable, easy_to_borrow, exchange -- ONE static row per symbol (14,356/14,356 unique), no
    `day` column and no SSR field. The PREREG asks for shortable AND SSR exclusion; only the
    static shortable flag is applicable from this source."""
    df = pd.read_csv(BORROW_CSV)
    n_shortable = int((df.shortable == True).sum())            # noqa: E712
    log(f'load_shortable: {len(df)} symbols in borrow_flags.csv, {n_shortable} shortable')
    log('WARNING: borrow_flags.csv has no per-day SSR field (static snapshot, no `day` column) -- '
        'the PREREG\'s SSR exclusion is NOT APPLIED here; only the static `shortable` flag is used '
        '(reported as a caveat, not imputed as pass or fail)')
    return dict(zip(df.symbol, df.shortable))


_PKL_CACHE = {}


def load_minute_trades(symbol, day, minute):
    """Trades DataFrame[ts,price,size] for one cached minute (sip_cache_1481 preferred, then
    sip_cache_1480), or None if neither has it. Memoized -- files are re-requested across the
    entry search and the walk for the same minute."""
    key = (symbol, day, minute)
    if key in _PKL_CACHE:
        return _PKL_CACHE[key]
    trades = None
    for d in SIP_DIRS:
        p = os.path.join(d, f'{symbol}_{day}_{minute}.pkl')
        if os.path.exists(p):
            with open(p, 'rb') as fh:
                trades, _quotes = pickle.load(fh)
            break
    _PKL_CACHE[key] = trades
    return trades


_BARS_CACHE = {}


def load_bars(symbol, day, conn):
    """Full-day 1-minute OHLC for (symbol, day) from bars_fills_1478.db, with an ET minute-of-day
    column `m`. Memoized per (symbol, day)."""
    key = (symbol, day)
    if key in _BARS_CACHE:
        return _BARS_CACHE[key]
    df = pd.read_sql_query(
        'SELECT t, o, h, l, c FROM bars WHERE symbol = ? AND day = ? ORDER BY t',
        conn, params=(symbol, day))
    if len(df):
        ts_ns = pd.to_datetime(df.t, utc=True).astype('int64')
        df['m'] = ((ts_ns - et_ns(day, 0)) / 60e9).round().astype(int)
    else:
        df['m'] = pd.Series(dtype=int)
    _BARS_CACHE[key] = df
    return df


def load_population(smoke=None):
    """9,911 base fills (causal_arming_causal.csv, status==fill), joined to the base outcome
    (model_1478_L3_predictions.csv outcome_R) and the spread/R-scale rail (features_1478_A.csv
    half_entry, spread_frac_at_fill, R_pct), all on the exact-float key (day, symbol, fill_min,
    split) -- the same join key cell_1478.py itself uses on these three files."""
    causal = pd.read_csv(CAUSAL_CSV, low_memory=False)
    causal = causal[causal.status == 'fill'].copy()
    causal = causal.sort_values(['day', 'symbol']).reset_index(drop=True)
    log(f'load_population: {len(causal)} base fills (status==fill) in causal_arming_causal.csv')
    if smoke:
        causal = causal.head(smoke).copy()
        log(f'--smoke: truncated to first {len(causal)} fills')

    keys = ['day', 'symbol', 'fill_min', 'split']
    model = pd.read_csv(MODEL_CSV)[keys + ['outcome_R']]
    feat = pd.read_csv(FEATURES_A_CSV)[keys + ['half_entry', 'spread_frac_at_fill', 'R_pct']]

    n0 = len(causal)
    pop = causal.merge(model, on=keys, how='left')
    if len(pop) != n0:
        log(f'ERROR: model merge changed row count {n0} -> {len(pop)} (duplicate keys) -- aborting')
        sys.exit(1)
    n_no_outcome = int(pop.outcome_R.isna().sum())
    if n_no_outcome:
        log(f'WARNING: {n_no_outcome}/{n0} fills have no matching row in '
            f'model_1478_L3_predictions.csv on {keys} -- base outcome_R NaN for these (counted, '
            f'not imputed; excluded from the paired mirror-check only, not from the short book)')

    pop = pop.merge(feat, on=keys, how='left')
    if len(pop) != n0:
        log(f'ERROR: features merge changed row count {n0} -> {len(pop)} (duplicate keys) -- aborting')
        sys.exit(1)
    n_no_feat = int(pop.half_entry.isna().sum())
    if n_no_feat:
        log(f'WARNING: {n_no_feat}/{n0} fills have no matching row in features_1478_A.csv -- '
            f'R-vs-spread rail NaN for these (counted, not imputed)')

    shortable_map = load_shortable()
    pop['shortable'] = pop.symbol.map(shortable_map)
    n_missing_borrow = int(pop.shortable.isna().sum())
    n_not_shortable = int((pop.shortable == False).sum())       # noqa: E712
    log(f'shortable filter: {n_not_shortable}/{n0} excluded (shortable==False), '
        f'{n_missing_borrow}/{n0} excluded (symbol absent from borrow_flags.csv)')
    eligible = pop[pop.shortable == True].copy()                # noqa: E712
    log(f'population after shortable filter: {len(eligible)}/{n0} eligible for the short')
    return pop, eligible


# ========================================================================================= entry
def find_short_entry(symbol, day, level, fill_min):
    """Through-print rule: a SELL LIMIT rests at level*(1+OFFER_BPS); filled AT THE LIMIT at the
    first tape print STRICTLY ABOVE it within the fill minute and the next 2 minutes. Returns
    status in {'fill','no_fill','no_fill_gap','no_tape'} -- 'no_fill' is a genuine, fully-observed
    absence of a qualifying print (a real result, counted); 'no_fill_gap' means some minute in the
    window was not cached so absence could not be confirmed (counted separately, WARNING-worthy in
    aggregate, not upgraded to a fill)."""
    limit = level * (1.0 + OFFER_BPS)
    m0 = int(np.floor(fill_min))
    start_ts = et_ns(day, fill_min * 60.0)
    end_ts = minute_start_ns(day, m0 + ENTRY_WINDOW_MIN + 1)

    frames, gap, have_m0 = [], False, False
    for m in range(m0, m0 + ENTRY_WINDOW_MIN + 1):
        tr = load_minute_trades(symbol, day, m)
        if tr is None:
            if m != m0:
                gap = True
            continue
        if m == m0:
            have_m0 = True
        frames.append(tr)

    if not have_m0:
        return dict(status='no_tape', entry_price=np.nan, entry_ts=np.nan, entry_m=np.nan,
                    print_price=np.nan, gap=True)

    if frames:
        tape = pd.concat(frames, ignore_index=True)
        tape = tape[(tape.ts >= start_ts) & (tape.ts < end_ts)].sort_values('ts', kind='stable')
    else:
        tape = pd.DataFrame(columns=['ts', 'price', 'size'])

    hit = tape[tape.price > limit + 1e-9]
    if len(hit):
        row = hit.iloc[0]
        entry_m = int(np.floor(ns_to_et_minutes(int(row.ts), day)))
        return dict(status='fill', entry_price=limit, entry_ts=int(row.ts), entry_m=entry_m,
                    print_price=float(row.price), gap=gap)
    return dict(status=('no_fill_gap' if gap else 'no_fill'), entry_price=np.nan, entry_ts=np.nan,
                entry_m=np.nan, print_price=np.nan, gap=gap)


# ========================================================================================== walk
def walk_short(symbol, day, entry_ts, entry_m, entry_print_price, stop_price, cover_target,
               bars_conn, split):
    """Mirrors sip_rebuild.walk_path's tie-break (stop wins when a single bar could satisfy both)
    and EOD-at-minute-955 convention, for a SHORT. Cover and stop are both resting orders (we ARE
    the bid on the cover, we ARE the stop-limit on the stop) so on the TAPE they fill exactly at
    their own resting price once touched; on a BAR (gap inside a minute we don't have on tape) a
    stop that gapped through at the open fills at the worse (higher) open price, mirroring
    walk_path's `row.o if row.o <= stop else stop` for the long, sign-flipped for the short. EOD
    (minute >= 955) covers at the bar's open crossed to the ask via EOD_ASK_BPS. Runs of >=3
    missing RTH bar-minutes are counted as halt candidates; the walk simply resumes at the next
    print/bar (the reopen), per the PREREG."""
    halt_gap = 0
    if entry_print_price >= stop_price - 1e-9:
        # The triggering print itself already gapped through the stop -- a fast market, PREREG's
        # own refuter #2. Use the real observed print price, not the nominal stop trigger.
        return dict(exit_ts=entry_ts, exit_m=entry_m, exit_price=entry_print_price, why='stop',
                    ambiguous_bar=False, halt_gap_minutes=0, resolved_within_tape=True,
                    eod_fallback=False)

    m = entry_m
    cur_ts = entry_ts
    consecutive_missing = 0
    while m < EOD_M:
        tr = load_minute_trades(symbol, day, m) if m <= entry_m + TAPE_WINDOW_MIN else None
        if tr is not None:
            sub = tr[tr.ts > cur_ts].sort_values('ts', kind='stable')
            stop_hit = sub[sub.price >= stop_price - 1e-9]
            cover_hit = sub[sub.price <= cover_target + 1e-9]
            t_stop = int(stop_hit.ts.iloc[0]) if len(stop_hit) else None
            t_cover = int(cover_hit.ts.iloc[0]) if len(cover_hit) else None
            if t_stop is not None and (t_cover is None or t_stop <= t_cover):
                return dict(exit_ts=t_stop, exit_m=m, exit_price=stop_price, why='stop',
                            ambiguous_bar=False, halt_gap_minutes=halt_gap,
                            resolved_within_tape=True, eod_fallback=False)
            if t_cover is not None:
                return dict(exit_ts=t_cover, exit_m=m, exit_price=cover_target, why='cover',
                            ambiguous_bar=False, halt_gap_minutes=halt_gap,
                            resolved_within_tape=True, eod_fallback=False)
            consecutive_missing = 0
        else:
            brow = load_bars(symbol, day, bars_conn)
            brow = brow[brow.m == m]
            if not len(brow):
                consecutive_missing += 1
                if consecutive_missing == HALT_GAP_MIN:
                    halt_gap += 1
                m += 1
                continue
            consecutive_missing = 0
            b = brow.iloc[0]
            stop_touch = b.h >= stop_price - 1e-9
            cover_touch = b.l <= cover_target + 1e-9
            if stop_touch:
                px = b.o if b.o >= stop_price else stop_price
                return dict(exit_ts=minute_start_ns(day, m), exit_m=m, exit_price=px, why='stop',
                            ambiguous_bar=bool(cover_touch), halt_gap_minutes=halt_gap,
                            resolved_within_tape=False, eod_fallback=False)
            if cover_touch:
                return dict(exit_ts=minute_start_ns(day, m), exit_m=m, exit_price=cover_target,
                            why='cover', ambiguous_bar=False, halt_gap_minutes=halt_gap,
                            resolved_within_tape=False, eod_fallback=False)
        cur_ts = minute_start_ns(day, m + 1) - 1
        m += 1

    bars = load_bars(symbol, day, bars_conn)
    brow = bars[bars.m == EOD_M]
    if len(brow):
        base_px, fb = float(brow.iloc[0].o), False
    else:
        earlier = bars[bars.m < EOD_M].sort_values('m')
        if len(earlier):
            base_px, fb = float(earlier.iloc[-1].c), True
        else:
            base_px, fb = float(cover_target), True
        log(f'WARNING: {symbol} {day} -- no bar at/after minute {EOD_M} (15:55 ET), EOD cover '
            f'fell back to the last available bar close (or the cover target if none exists)')
    ask_px = base_px * (1.0 + EOD_ASK_BPS[split] / 1e4)
    return dict(exit_ts=minute_start_ns(day, EOD_M), exit_m=EOD_M, exit_price=ask_px, why='eod',
                ambiguous_bar=False, halt_gap_minutes=halt_gap, resolved_within_tape=False,
                eod_fallback=fb)


def cost_and_r(entry_price, exit_price, why, entry_ts, exit_ts, stop_price, split):
    """gross = entry - exit (a short profits when exit < entry). Stop rows carry the PREREG's
    stop-limit standard expected-value tail (SLIP_STOP_BPS, mirroring cell_1478's
    substitute_stop_slip -- exit_price * bps/1e4, in $, deducted). Borrow (3%/yr pro rata on the
    holding period) is deducted on every row regardless of why."""
    R_f = stop_price - entry_price
    gross = entry_price - exit_price
    hold_s = max(0.0, (exit_ts - entry_ts) / 1e9)
    borrow = entry_price * BORROW_APY * (hold_s / (365.0 * 86400.0))
    slip = exit_price * SLIP_STOP_BPS[split] / 1e4 if why == 'stop' else 0.0
    net = gross - slip - borrow
    net_Rf = net / R_f if R_f else np.nan
    net_pct = net / entry_price * 100.0
    return R_f, net_Rf, net_pct, borrow, slip


# ========================================================================================== main
def simulate(pop, bars_conn):
    """One pass over the eligible population: entry once, walk twice (1,619 / 1,620 report-only
    cover targets share the same entry and stop). Returns a long DataFrame, two rows per fill
    (cell 1619, cell 1620), plus unfilled rows (one row per cell, NaN exit fields, counted)."""
    rows = []
    n = len(pop)
    t0 = time.time()
    counts = dict(fill=0, no_fill=0, no_fill_gap=0, no_tape=0)
    for i, r in enumerate(pop.itertuples()):
        if i and i % 500 == 0:
            log(f'simulate: {i}/{n} ({time.time() - t0:.0f}s) -- {counts}')
        level = float(r.level)
        stop_price = level * (1.0 + STOP_BPS)
        e = find_short_entry(r.symbol, r.day, level, float(r.fill_min))
        counts[e['status']] = counts.get(e['status'], 0) + 1

        base_row = dict(split=r.split, day=r.day, symbol=r.symbol, level=level,
                        fill_min=r.fill_min, entry_status=e['status'],
                        outcome_R=getattr(r, 'outcome_R', np.nan),
                        R_pct=getattr(r, 'R_pct', np.nan),
                        half_entry=getattr(r, 'half_entry', np.nan))
        for cell, cover_target in ((1619, level - COVER_ABS_1619),
                                    (1620, level * (1.0 - COVER_PCT_1620))):
            row = dict(base_row, cell=cell)
            if e['status'] != 'fill':
                row.update(filled=False, entry=np.nan, exit=np.nan, why=np.nan, net_Rf=np.nan,
                          net_pct=np.nan, exit_m=np.nan, ambiguous_bar=np.nan,
                          halt_gap_minutes=np.nan, resolved_within_tape=np.nan,
                          eod_fallback=np.nan, entry_m=np.nan)
                rows.append(row)
                continue
            w = walk_short(r.symbol, r.day, e['entry_ts'], e['entry_m'], e['print_price'],
                           stop_price, cover_target, bars_conn, r.split)
            R_f, net_Rf, net_pct, borrow, slip = cost_and_r(
                e['entry_price'], w['exit_price'], w['why'], e['entry_ts'], w['exit_ts'],
                stop_price, r.split)
            row.update(filled=True, entry=e['entry_price'], exit=w['exit_price'], why=w['why'],
                      net_Rf=net_Rf, net_pct=net_pct, entry_m=e['entry_m'], exit_m=w['exit_m'],
                      R_f=R_f, borrow=borrow, slip=slip,
                      ambiguous_bar=w['ambiguous_bar'], halt_gap_minutes=w['halt_gap_minutes'],
                      resolved_within_tape=w['resolved_within_tape'],
                      eod_fallback=w['eod_fallback'])
            rows.append(row)
    log(f'simulate: DONE {n}/{n} ({time.time() - t0:.0f}s) -- final entry counts {counts}')
    return pd.DataFrame(rows), counts


def report_cell(df_cell, cell_id, report_only):
    """One cell's report block across TRAIN-H2 / VAL: fill share, cover share within 15 min, stop
    share, runner-cohort cost, mean net, day-clustered t, ex-top-5%, R_f rail, fills/week, pass
    bar."""
    lines = [f'## Cell {cell_id}{" (report-only)" if report_only else ""}']
    split_mean_Rf = {}
    for split in ('TRAIN', 'VAL'):
        sub = df_cell[df_cell.split == split]
        n_pop = len(sub)
        filled = sub[sub.filled == True]                        # noqa: E712
        n_fill = len(filled)
        fill_share = n_fill / n_pop if n_pop else np.nan
        cover15 = filled[(filled.why == 'cover') & (filled.exit_m - filled.entry_m <= TAPE_WINDOW_MIN)]
        cover_share_15 = len(cover15) / n_fill if n_fill else np.nan
        stop = filled[filled.why == 'stop']
        stop_share = len(stop) / n_fill if n_fill else np.nan
        runner = filled[~((filled.why == 'cover') & (filled.exit_m - filled.entry_m <= TAPE_WINDOW_MIN))]
        runner_mean_pct = runner.net_pct.mean() if len(runner) else np.nan
        mean_Rf = filled.net_Rf.mean()
        split_mean_Rf[split] = mean_Rf if n_fill else np.nan
        mean_pct = filled.net_pct.mean()
        t_Rf = day_clustered_t(filled.net_Rf, filled.day)
        ex5_Rf = ex_top5_mean(filled.net_Rf)
        median_Rf_pct = float(filled.R_f.div(filled.entry).mul(100).median()) if n_fill else np.nan
        wk = weeks_spanned(filled.day) if n_fill else np.nan
        fills_wk = n_fill / wk if n_fill else np.nan
        ambiguous = int(filled.ambiguous_bar.eq(True).sum())
        halts = int((filled.halt_gap_minutes.fillna(0) > 0).sum())
        eod_fb = int(filled.eod_fallback.eq(True).sum())

        passed = (not report_only and n_fill > 0 and mean_Rf >= PASS_BAR['mean_net_Rf'] and
                 mean_pct >= PASS_BAR['mean_net_pct'] and (t_Rf or 0) >= PASS_BAR['t'] and
                 ex5_Rf > 0 and fills_wk >= PASS_BAR['min_fills_wk'] and
                 median_Rf_pct >= PASS_BAR['median_Rf_pct'])

        lines += [
            f'',
            f'### {split} (holdout)',
            f'* population (eligible, shortable): {n_pop}; short entries filled: {n_fill} '
            f'(fill share {fill_share:.3f})',
            f'* cover share within {TAPE_WINDOW_MIN} min: {cover_share_15:.3f} '
            f'({len(cover15)}/{n_fill})' if n_fill else '* cover share within 15 min: n/a (0 fills)',
            f'* stop share: {stop_share:.3f} ({len(stop)}/{n_fill})' if n_fill else '* stop share: n/a',
            f'* runner cohort (no cover within 15 min, n={len(runner)}): mean net % of price = '
            f'{runner_mean_pct:.4f}' if len(runner) else '* runner cohort: n/a',
            f'* mean net R_f = {mean_Rf:.4f}; mean net % of price = {mean_pct:.4f}; '
            f'day-clustered t (R_f) = {t_Rf:.2f}' if n_fill else '* mean net: n/a',
            f'* ex-top-5% mean net R_f = {ex5_Rf:.4f}' if n_fill else '* ex-top-5%: n/a',
            f'* median R_f as % of price = {median_Rf_pct:.4f} (rail: >= 0.5 or NOT SHIPPABLE)'
            if n_fill else '* median R_f %: n/a',
            f'* fills/week (raw, unslotted -- see caveats) = {fills_wk:.2f} over {wk} weeks'
            if n_fill else '* fills/week: n/a',
            f'* ambiguous bars (both stop and cover touched in one un-taped bar, stop-first '
            f'tie-break applied): {ambiguous}; halt candidates (>=3 consecutive missing RTH '
            f'bar-minutes): {halts}; EOD-fallback exits (no bar at/after 15:55): {eod_fb}',
        ]
        if not report_only and split == 'VAL':
            lines.append(f'* **PASS BAR (frozen, VAL only binds): {"PASS" if passed else "FAIL"}**')

    if not report_only:
        tr_Rf, val_Rf = split_mean_Rf.get('TRAIN'), split_mean_Rf.get('VAL')
        if pd.notna(tr_Rf) and pd.notna(val_Rf):
            same_sign = (tr_Rf > 0) == (val_Rf > 0)
            direction = 'positive' if val_Rf > 0 else 'negative'
            lines.append(f'* TRAIN-H2 same-sign check: TRAIN-H2 mean net R_f = {tr_Rf:.4f}, VAL '
                        f'mean net R_f = {val_Rf:.4f} -> {"SAME sign" if same_sign else "DIFFERENT sign"} '
                        f'({"both" if same_sign else "VAL"} {direction}{" -- the losing direction" if val_Rf < 0 else ""})')
        else:
            lines.append('* TRAIN-H2 same-sign check: n/a (no fills in one holdout)')
    return '\n'.join(lines), df_cell


def spread_rail(filled_1619):
    """R-vs-spread rail: net % of price by half_entry/level quartile (does the edge depend on
    trading a wide-spread name)."""
    d = filled_1619.dropna(subset=['half_entry', 'level']).copy()
    if len(d) < 8:
        return 'R-vs-spread rail: n/a (< 8 fills with half_entry)', np.nan
    d['spread_pct'] = d.half_entry / d.level * 100.0
    d['q'] = pd.qcut(d.spread_pct, 4, labels=['Q1 (tight)', 'Q2', 'Q3', 'Q4 (wide)'], duplicates='drop')
    corr = d.net_pct.corr(d.spread_pct)
    lines = [f'R-vs-spread rail (corr(net_pct, half_entry/level) = {corr:.3f}):']
    for q, g in d.groupby('q', observed=True):
        lines.append(f'  * {q}: n={len(g)}, mean net % = {g.net_pct.mean():.4f}')
    return '\n'.join(lines), corr


def mirror_check(filled_1619):
    """Paired against the base long on the same fills: base_net_pct = outcome_R * R_pct (features_
    1478_A's R as % of price) vs the short's net_pct, same (day, symbol) row. Expect a negative
    correlation (a mirror bet) if the mechanism is real."""
    d = filled_1619.dropna(subset=['outcome_R', 'R_pct']).copy()
    if len(d) < 8:
        return 'Mirror check: n/a (< 8 fills with a matched base outcome)'
    d['base_net_pct'] = d.outcome_R * d.R_pct
    corr = d.net_pct.corr(d.base_net_pct)
    return (f'Mirror check (n={len(d)}): short mean net % = {d.net_pct.mean():.4f}, '
           f'base long mean net % (outcome_R * R_pct) = {d.base_net_pct.mean():.4f}, '
           f'corr(short, base long) = {corr:.3f} (expect negative if the fade is a real mirror)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=None, help='limit to the first N base fills')
    args = ap.parse_args()

    t0 = time.time()
    log('cell_1619: START (PREREG_1617.md Frame B -- burst fade)')
    _pop_all, eligible = load_population(smoke=args.smoke)

    bars_conn = sqlite3.connect(f'file:{BARS_DB}?mode=ro', uri=True)
    df, entry_counts = simulate(eligible, bars_conn)
    bars_conn.close()

    out_cols = ['cell', 'split', 'day', 'symbol', 'filled', 'entry', 'exit', 'why', 'net_Rf', 'net_pct']
    df[out_cols].to_csv(FILLS_CSV, index=False)
    log(f'wrote {FILLS_CSV} ({len(df)} rows)')

    d1619 = df[df.cell == 1619].copy()
    d1620 = df[df.cell == 1620].copy()
    filled_1619 = d1619[d1619.filled == True]                    # noqa: E712

    block_1619, _ = report_cell(d1619, 1619, report_only=False)
    block_1620, _ = report_cell(d1620, 1620, report_only=True)
    rail_txt, _ = spread_rail(filled_1619)
    mirror_txt = mirror_check(filled_1619)

    lines = [
        '# RESULT -- cells 1,619 / 1,620 (Frame B: burst fade)',
        '',
        f'PREREG: `research/hod_entry/PREREG_1617.md` (FROZEN 2026-09-28 17:00 UTC). Builder run '
        f'{"(SMOKE, n=" + str(args.smoke) + ")" if args.smoke else "(FULL, n=9,911 base fills)"}, '
        f'elapsed {time.time() - t0:.0f}s.',
        '',
        '## Population funnel (both cells share entry/exclusion; cover target differs)',
        f'* base fills (status==fill, causal_arming_causal.csv): {len(_pop_all)}',
        f'* excluded, not shortable or missing from borrow_flags.csv: '
        f'{int((_pop_all.shortable != True).sum())}',
        f'* eligible (shortable) population: {len(eligible)}',
        f'* entry search outcome on the eligible population: {entry_counts}',
        f'  (`no_fill` = confirmed absent burst print, a real result; `no_fill_gap` = some minute '
        f'in the 3-minute entry window was not cached, absence unconfirmed; `no_tape` = the fill '
        f'minute itself has no cached tape, excluded from the population)',
        '',
        block_1619,
        '',
        block_1620,
        '',
        '## Rails',
        rail_txt,
        '',
        mirror_txt,
        '',
        '## Caveats (read as an adversary before relaying)',
        '* **SSR not applied**: borrow_flags.csv (research/fuckup_audit/O_halt/PASSIVE/) is a '
        'static one-row-per-symbol snapshot (symbol, tradable, shortable, easy_to_borrow, '
        'exchange) with no `day` column and no SSR field. Only the static `shortable` flag is '
        'excluded here; a per-day SSR trigger (10% intraday decline) is NOT modeled. If the true '
        'HOD-break population has meaningful SSR incidence this book is optimistic by that share.',
        '* **fills/week is unslotted**: cell_1445.fills_per_week applies research/hod_consol/'
        'run_consol.simulate_slots (first-12/day, 4-concurrent). That module was not imported '
        'here (to avoid its wider dependency graph inside the 40-call builder budget); the '
        'fills/week reported above is the raw count / distinct ISO weeks, an UPPER BOUND on the '
        'slotted number.',
        '* **R_pct inferred**: the mirror check assumes features_1478_A.csv\'s `R_pct` column is '
        'the base long\'s R expressed as a percent of price (base_net_pct = outcome_R * R_pct) by '
        'name and units alone -- not independently confirmed against its builder script.',
        '* **Halt handling is approximate**: a halt is inferred as >=3 consecutive missing RTH '
        'minutes in bars_fills_1478.db; the walk resumes at the next available print/bar (the '
        'reopen), which is what "the position is marked at the reopen print" was read to mean, '
        'but no explicit halt flag was cross-checked against an independent halt calendar.',
        '* **Ambiguous bars use stop-first**, mirroring sip_rebuild.walk_path\'s documented tie-'
        'break for the long, sign-flipped; this is conservative (never overstates the fade\'s '
        'edge) but is a modeled assumption on bars where tape was unavailable, not an observed '
        'fact.',
        '* **No independent reimplementation yet**: this is the BUILDER only, per CLAUDE.md\'s '
        'independent-check protocol a rebuild from this prose by an agent that has not read '
        'cell_1619.py is required before any number here is relayed to the owner.',
        '* Price-scale (split/adjustment) check was not run: this is a same-day, intraday-only '
        'book (entry and exit both inside one session), so a raw-vs-adjusted daily-bar mismatch '
        'cannot fabricate the R -- the usual multi-day risk in CLAUDE.md item 3 does not apply the '
        'same way here.',
    ]
    with open(RESULT_MD, 'w') as fh:
        fh.write('\n'.join(lines))
    log(f'wrote {RESULT_MD}')
    log('cell_1619: DONE')


if __name__ == '__main__':
    main()
