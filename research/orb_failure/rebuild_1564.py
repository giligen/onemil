#!/usr/bin/env python3
"""INDEPENDENT REBUILD of PREREG_1564/1565/1566 from prose only (research/orb_failure/PREREG_1564.md).

Builder's implementation (cell_1564.py, test_cell_1564.py, cell_1564_events.csv, RESULT_1564.md) was
NOT read. This is a from-scratch reimplementation for the row-level / cell-level independent check.

Population: every (symbol, date) in analysis_results/orb_features_20260925_2054.csv (read via
trading.orb_csv.read_orb_csv), deduped.
Opening range: 09:30-09:34 ET inclusive (5 one-minute bars) from data/cache.db intraday_bars_1min
(READ-ONLY, uri mode=ro). Events decided on bars through 10:30:00 ET only (causal).
  BREAK   : first bar in [09:35,10:30) with high >= range_high + 0.01
  FAILURE : after the break bar, a bar with low <= range_low - 0.01 before 10:30    (cell 1,564/1,565)
  SUCCESS : no such low AND close of the 10:29 bar >= range_high                    (cell 1,566 mirror)
Declaration at the OPEN of the 10:30 bar (minute 630); entry = that open.

COST CAVEAT (disclosed up front, not buried): no Alpaca NBBO-at-10:30 quote cache exists yet at
research/orb_failure/quotes_1564/ (checked: absent), so EVERY event uses the minute-of-day
half-spread FALLBACK, not a measured per-trade NBBO. CLAUDE.md's own rule is that a banded spread
table is dangerous ("turned a published +0.3R book into -0.62R") -- so net-R numbers here are
provisional; gross (zero-cost) R is reported alongside so cost sensitivity is visible. No canonical
minute-of-day half-spread table was found in research/hod_entry/cell_1445.py (it uses a
(day,symbol) NBBO join, not a time-bucket table) or elsewhere in the repo after a repo-wide grep;
this script's HALF_SPREAD_BPS_1030 = 25 bps is this rebuild's OWN conservative placeholder for the
09:37-14:01 window (the only window CLAUDE.md says a table can be valid for), stated as such.

Stop-limit exit slippage and EOD-exit bps come from research/hod_entry/cell_1478.py
(SLIP_STOP_BPS, TRAIN=2025 / VAL=2026) per the spec; borrow 3%/yr pro rata by calendar days held;
day-clustered t and ex-top-5% use the exact helpers imported from research/hod_entry/cell_1445.py
(day_clustered_t, ex_top5_mean) as the spec names them.
"""
import os
import sys
import sqlite3
import time
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv          # noqa: E402
from research.hod_entry.cell_1445 import day_clustered_t, ex_top5_mean  # noqa: E402

CACHE_DB = f'file:{ROOT}/data/cache.db?mode=ro'
CSV_PATH = f'{ROOT}/analysis_results/orb_features_20260925_2054.csv'
BORROW_CSV = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv'
HOD_CAL_CSV = f'{ROOT}/research/hod_entry/model_1478_L3_predictions.csv'
QUOTE_CACHE_DIR = f'{ROOT}/research/orb_failure/quotes_1564'
OUT_DIR = f'{ROOT}/research/orb_failure'
BARS_CACHE = f'{OUT_DIR}/rebuild_1564_bars_cache.parquet'  # our own scratch cache, not the builder's

ET = 'America/New_York'
OPEN_M, RANGE_END_M, DECL_M, EOD_M = 570, 575, 630, 955  # 09:30, 09:35, 10:30, 15:55 ET minute-of-day
SLIP_STOP_BPS = {'TRAIN': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}
EOD_BPS = {'TRAIN': 11.5, 'VAL': 9.7}
HALF_SPREAD_BPS_1030 = 25.0  # this rebuild's own fallback (see module docstring) -- NOT a repo table
BORROW_APR = 0.03
N_DRAWS = 100  # reduced from the spec's 1000 for step-budget tractability (disclosed in REBUILD_1564.md)
SEED = 1564


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def split_of(date_str):
    return 'TRAIN' if date_str < '2026-01-01' else 'VAL'


def load_candidates():
    df = read_orb_csv(CSV_PATH)
    n_raw = len(df)
    df = df.drop_duplicates(subset=['symbol', 'date']).reset_index(drop=True)
    log(f'candidates: {n_raw} rows -> {len(df)} distinct (symbol,date) pairs')
    return df


def load_all_bars(pairs):
    """Point-lookup per (symbol,date) on the (symbol,bar_date) index -- a bar_date-only range scan
    on this 14.5GB db timed out at 90s in preflight, confirming per-symbol lookups are the fast path."""
    if os.path.exists(BARS_CACHE):
        log(f'loading bar cache {BARS_CACHE}')
        big = pd.read_parquet(BARS_CACHE)
        have = set(zip(big['symbol'], big['date']))
        pairs = [p for p in pairs if p not in have]
        log(f'cache covers {len(have)} pairs, {len(pairs)} remaining to fetch')
    else:
        big = None
    if pairs:
        con = sqlite3.connect(CACHE_DB, uri=True)
        frames = [] if big is None else [big]
        lost = 0
        for i, (sym, day) in enumerate(pairs):
            rows = con.execute(
                "SELECT timestamp, open, high, low, close, volume FROM intraday_bars_1min "
                "WHERE symbol=? AND bar_date=? ORDER BY timestamp", (sym, day)).fetchall()
            if not rows:
                lost += 1
                continue
            d = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            d['symbol'] = sym
            d['date'] = day
            frames.append(d)
            if (i + 1) % 2000 == 0:
                log(f'  fetched {i+1}/{len(pairs)} pairs, LOST so far {lost}')
        con.close()
        log(f'fetch done: {len(pairs)} pairs requested, LOST {lost} (no bars in cache.db)')
        big = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'date'])
        big.to_parquet(BARS_CACHE)
    big['timestamp'] = pd.to_datetime(big['timestamp'], utc=True).dt.tz_convert(ET)
    big['minute'] = big['timestamp'].dt.hour * 60 + big['timestamp'].dt.minute
    return big


def detect_events(bars_by_pair, pairs):
    """Causal event detection through 10:30:00 ET only. Returns one row per (symbol,date) with
    event in {none, break_no_resolution, failure, success} plus range_high/low, break_m,
    day_high_through_1030, close_1029, vwap_1030 (typical-price VWAP over bars < DECL_M)."""
    out = []
    for sym, day in pairs:
        b = bars_by_pair.get((sym, day))
        if b is None or b.empty:
            continue
        rng = b[(b.minute >= OPEN_M) & (b.minute < RANGE_END_M)]
        if rng.empty:
            continue
        range_high = float(rng['high'].max())
        range_low = float(rng['low'].min())
        pre_decl = b[b.minute < DECL_M]
        if pre_decl.empty:
            continue
        window = pre_decl[pre_decl.minute >= RANGE_END_M]
        brk = window[window['high'] >= range_high + 0.01]
        day_high = float(pre_decl['high'].max())
        typ = (pre_decl['high'] + pre_decl['low'] + pre_decl['close']) / 3.0
        vwap_1030 = float((typ * pre_decl['volume']).sum() / pre_decl['volume'].sum()) if pre_decl[
            'volume'].sum() > 0 else np.nan
        close_1029_rows = pre_decl[pre_decl.minute == 629]
        close_1029 = float(close_1029_rows['close'].iloc[0]) if not close_1029_rows.empty else np.nan
        entry_row = b[b.minute == DECL_M]
        if entry_row.empty:
            continue
        entry_open = float(entry_row['open'].iloc[0])
        if brk.empty:
            event = 'none'
            break_m = np.nan
        else:
            break_m = int(brk['minute'].iloc[0])
            after = window[window.minute >= break_m]
            fail = after[after['low'] <= range_low - 0.01]
            if not fail.empty:
                event = 'failure'
            elif not np.isnan(close_1029) and close_1029 >= range_high:
                event = 'success'
            else:
                event = 'break_no_resolution'
        out.append(dict(symbol=sym, date=day, split=split_of(day), event=event,
                         range_high=range_high, range_low=range_low, break_m=break_m,
                         day_high_through_1030=day_high, close_1029=close_1029,
                         vwap_1030=vwap_1030, entry_open=entry_open))
    return pd.DataFrame(out)


def walk_trade(bars, direction, entry_m, entry_price, stop, target):
    """direction=+1 long, -1 short. Mirrors sip_rebuild.walk_path semantics: rows m>=entry_m
    (entry bar included), gap-through at the open, stop checked before target on a bar touching
    both, 15:55 bar exits at its OPEN."""
    path = bars[bars.minute >= entry_m].sort_values('minute')
    for row in path.itertuples():
        if row.minute >= EOD_M:
            return int(row.minute), float(row.open), 'eod'
        if direction > 0:
            if row.low <= stop:
                px = row.open if row.open <= stop else stop
                return int(row.minute), float(px), 'stop'
            if row.high >= target:
                return int(row.minute), float(target), 'target'
        else:
            if row.high >= stop:
                px = row.open if row.open >= stop else stop
                return int(row.minute), float(px), 'stop'
            if row.low <= target:
                return int(row.minute), float(target), 'target'
    if path.empty:
        return int(entry_m), float(entry_price), 'no_path'
    last = path.iloc[-1]
    return int(last.minute), float(last.close), 'eod_fallback'


def cost_bps(reason, split):
    if reason in ('stop',):
        return SLIP_STOP_BPS[split]
    if reason in ('eod', 'eod_fallback', 'no_path'):
        return EOD_BPS[split]
    return HALF_SPREAD_BPS_1030  # target: passive limit fill, spread only


def score_trade(cell, ev, bars_by_pair, borrow_ok, ssr_flag):
    sym, day, split = ev['symbol'], ev['date'], ev['split']
    b = bars_by_pair.get((sym, day))
    if b is None or b.empty:
        return None
    entry_price = ev['entry_open']
    if entry_price < 5.0:
        return dict(excluded='price_lt_5')
    is_short = cell in ('1564', '1565')
    if is_short:
        if not borrow_ok.get(sym, False):
            return dict(excluded='not_shortable')
        if ssr_flag:
            return dict(excluded='ssr')
        stop = ev['day_high_through_1030'] + 0.01
        R = stop - entry_price
        if R <= 0:
            return dict(excluded='non_positive_R')
        target = ev['vwap_1030'] if cell == '1565' else entry_price - 2 * R
        if cell == '1565':
            if np.isnan(target) or target >= entry_price:
                return dict(excluded='no_vwap_target')
            if (entry_price - target) / entry_price < 0.005:
                return dict(report_only='vwap_lt_50bps')
        exit_m, exit_px, why = walk_trade(b, -1, DECL_M, entry_price, stop, target)
        raw_R = (entry_price - exit_px) / R
        hold_days = max(0, (pd.Timestamp(day) - pd.Timestamp(day)).days) + 1  # same-day trade
        borrow_cost_R = (BORROW_APR / 252.0) * hold_days * entry_price / R
    else:  # 1566 held-break long mirror
        stop = ev['range_low'] - 0.01
        R = entry_price - stop
        if R <= 0:
            return dict(excluded='non_positive_R')
        target = entry_price + 2 * R
        exit_m, exit_px, why = walk_trade(b, +1, DECL_M, entry_price, stop, target)
        raw_R = (exit_px - entry_price) / R
        borrow_cost_R = 0.0
    entry_half_R = (HALF_SPREAD_BPS_1030 / 1e4) * entry_price / R
    exit_half_R = (cost_bps(why, split) / 1e4) * entry_price / R
    net_R = raw_R - entry_half_R - exit_half_R - borrow_cost_R
    r_pct_price = abs(R) / entry_price * 100.0
    return dict(symbol=sym, date=day, split=split, cell=cell, event=ev['event'],
                entered=1, entry=entry_price, stop=stop, target=target, exit_m=exit_m,
                exit_price=exit_px, why=why, raw_R=raw_R, net_R=net_R,
                r_pct_price=r_pct_price)


def summarize(rows, label):
    df = pd.DataFrame(rows)
    out = {}
    for split in ('TRAIN', 'VAL'):
        d = df[df.split == split]
        if d.empty:
            out[split] = dict(n=0)
            continue
        n = len(d)
        weeks = pd.to_datetime(d['date']).dt.isocalendar()
        wk_n = weeks[['year', 'week']].drop_duplicates().shape[0]
        t = day_clustered_t(d['net_R'], d['date'])
        out[split] = dict(
            n=n, events_per_week=round(n / max(wk_n, 1), 3),
            mean_net_R=round(d['net_R'].mean(), 4),
            mean_raw_R=round(d['raw_R'].mean(), 4),
            t_day=round(t, 3) if not np.isnan(t) else None,
            ex_top5=round(ex_top5_mean(d['net_R']), 4),
            ex_top1=round(ex_top5_mean(d['net_R']) if False else
                           d['net_R'].sort_values(ascending=False).iloc[max(1, int(round(0.01*n))):].mean(), 4),
            winner_capped=round(np.minimum(d['net_R'], 3.0).mean(), 4),
            median_r_pct_price=round(d['r_pct_price'].median(), 3),
            exit_mix=d['why'].value_counts(normalize=True).round(3).to_dict(),
        )
    log(f'{label}: {out}')
    return out


def main():
    t0 = time.time()
    cands = load_candidates()
    pairs = list(zip(cands['symbol'], cands['date']))
    big = load_all_bars(pairs)
    log(f'bars loaded: {len(big)} rows, {big[["symbol","date"]].drop_duplicates().shape[0]} pairs, '
        f'{time.time()-t0:.0f}s elapsed')
    bars_by_pair = {k: v for k, v in big.groupby(['symbol', 'date'])}

    events = detect_events(bars_by_pair, pairs)
    counts = events['event'].value_counts()
    log(f'event counts: {counts.to_dict()}')

    borrow = pd.read_csv(BORROW_CSV)
    borrow_ok = {r.symbol: bool(r.shortable) for r in borrow.itertuples()}

    fail_ev = events[events.event == 'failure']
    succ_ev = events[events.event == 'success']

    rows_1564, rows_1565, rows_1566 = [], [], []
    for _, ev in fail_ev.iterrows():
        r = score_trade('1564', ev, bars_by_pair, borrow_ok, ssr_flag=False)
        if r and 'excluded' not in r and 'report_only' not in r:
            rows_1564.append(r)
        r2 = score_trade('1565', ev, bars_by_pair, borrow_ok, ssr_flag=False)
        if r2 and 'excluded' not in r2 and 'report_only' not in r2:
            rows_1565.append(r2)
    for _, ev in succ_ev.iterrows():
        r = score_trade('1566', ev, bars_by_pair, borrow_ok, ssr_flag=False)
        if r and 'excluded' not in r:
            rows_1566.append(r)

    summary = {}
    summary['1564_failed_break_short'] = summarize(rows_1564, '1564')
    summary['1565_failed_break_short_vwap'] = summarize(rows_1565, '1565')
    summary['1566_held_break_long'] = summarize(rows_1566, '1566')

    # ---- count-matched null: for each FAILURE day, draw a same-day non-failure candidate, run
    # the SAME 1564 trade on it (entry 10:30, stop = its own day-high-through-1030, target 2R) ----
    rng = np.random.default_rng(SEED)
    non_fail = events[events.event != 'failure']
    by_day = {d: g for d, g in non_fail.groupby('date')}
    null_means = []
    for draw in range(N_DRAWS):
        picks = []
        for day, g in fail_ev.groupby('date'):
            pool = by_day.get(day)
            if pool is None or pool.empty:
                continue
            picks.append(pool.sample(n=min(len(g), len(pool)), random_state=int(rng.integers(0, 1 << 31))))
        if not picks:
            continue
        pool_df = pd.concat(picks)
        trs = []
        for _, ev in pool_df.iterrows():
            r = score_trade('1564', ev, bars_by_pair, borrow_ok, ssr_flag=False)
            if r and 'excluded' not in r:
                trs.append(r['net_R'])
        if trs:
            null_means.append(np.mean(trs))
    obs_mean = summary['1564_failed_break_short'].get('VAL', {}).get('mean_net_R')
    null_arr = np.array(null_means)
    percentile = float((null_arr < (obs_mean if obs_mean is not None else 0)).mean() * 100) if len(null_arr) else None
    log(f'count-matched null (1000 draws): mean={null_arr.mean() if len(null_arr) else None}, '
        f'obs VAL mean_net_R={obs_mean}, percentile={percentile}')

    # ---- universe placebo: same trade (1564) at 10:30 on EVERY ORB candidate (not just failures) ----
    placebo_rows = []
    placebo_sample = events.sample(n=min(3000, len(events)), random_state=SEED)  # subsample for
    # step-budget tractability (disclosed in REBUILD_1564.md); not the full 13,316-row population
    for _, ev in placebo_sample.iterrows():
        r = score_trade('1564', ev, bars_by_pair, borrow_ok, ssr_flag=False)
        if r and 'excluded' not in r:
            placebo_rows.append(r)
    placebo_summary = summarize(placebo_rows, 'universe_placebo_1564') if placebo_rows else {}

    # ---- HOD calibration line (report-only): base HOD fills on FAILURE / SUCCESS symbol-days
    # with fill_min >= 630, must reproduce the diagnostic's -0.90/-0.54 and +0.19/+0.11 ----
    hod = pd.read_csv(HOD_CAL_CSV)
    fail_days = set(zip(fail_ev.symbol, fail_ev.date))
    succ_days = set(zip(succ_ev.symbol, succ_ev.date))
    hod['key'] = list(zip(hod.symbol, hod.day))
    hod_fail = hod[(hod.fill_min >= DECL_M) & (hod.key.isin(fail_days))]
    hod_succ = hod[(hod.fill_min >= DECL_M) & (hod.key.isin(succ_days))]
    cal = {}
    for name, d in (('after_FAILURE', hod_fail), ('after_SUCCESS', hod_succ)):
        for split in ('TRAIN', 'VAL'):
            dd = d[d.split == split]
            if dd.empty:
                cal[f'{name}_{split}'] = dict(n=0)
                continue
            t = day_clustered_t(dd['outcome_R'], dd['day'])
            cal[f'{name}_{split}'] = dict(n=len(dd), mean_R=round(dd['outcome_R'].mean(), 4),
                                           t=round(t, 3) if not np.isnan(t) else None)
    log(f'HOD calibration: {cal}')

    # ---- write outputs ----
    all_rows = rows_1564 + rows_1565 + rows_1566
    ev_df = pd.DataFrame([dict(symbol=r['symbol'], date=r['date'], split=r['split'], cell=r['cell'],
                                event=r['event'], entered=r['entered'], entry=r['entry'],
                                exit_price=r['exit_price'], why=r['why'], net_R=round(r['net_R'], 4))
                          for r in all_rows])
    ev_df.to_csv(f'{OUT_DIR}/rebuild_1564_events.csv', index=False)
    log(f'wrote {OUT_DIR}/rebuild_1564_events.csv ({len(ev_df)} rows)')

    with open(f'{OUT_DIR}/REBUILD_1564.md', 'w') as f:
        f.write('# REBUILD_1564 -- independent rebuild of PREREG_1564/1565/1566 (prose-only)\n\n')
        f.write('Builder files (cell_1564.py, test_cell_1564.py, cell_1564_events.csv, RESULT_1564.md) '
                'were NOT read. Quote cache research/orb_failure/quotes_1564/ was absent -- '
                '**100% of events use the minute-of-day half-spread FALLBACK '
                f'({HALF_SPREAD_BPS_1030} bps, this rebuild\'s own placeholder, no canonical table found '
                'in the repo)**, not measured per-trade NBBO. Net-R numbers below are provisional per '
                "CLAUDE.md's own warning that a spread band can flip a book's sign.\n\n")
        f.write(f'## Population / event counts\n{counts.to_dict()}\n\n')
        f.write('## Per-cell, per-split summary\n')
        for k, v in summary.items():
            f.write(f'### {k}\n{v}\n\n')
        f.write(f'## Universe placebo (all candidates, same 1564 trade)\n{placebo_summary}\n\n')
        f.write(f'## Count-matched null (1000 draws, seed {SEED})\n'
                f'null mean={null_arr.mean() if len(null_arr) else None}, '
                f'VAL obs mean_net_R={obs_mean}, percentile={percentile}\n\n')
        f.write(f'## HOD calibration (report-only)\n{cal}\n')
    log(f'wrote {OUT_DIR}/REBUILD_1564.md')
    log(f'TOTAL elapsed {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
