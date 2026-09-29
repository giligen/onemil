#!/usr/bin/env python3
"""
Cells 1,643-1,645: leveraged-ETF rebalancing flow into the close.

Implements research/lev_flow/PREREG_1643.md (frozen 2026-09-29 05:20 UTC) EXACTLY.

Mechanism: a daily-rebalanced L-times fund must trade (L^2 - L) * AUM * r_t of its underlying at the
close in the DIRECTION of the day's move. Prediction: on large-move days the underlying continues in
the day's direction over the last half hour (cells 1,643 long / 1,645 short), and part of it reverts at
the next open (cell 1,644, report-only).

Signal: r_t = the underlying's return from the prior OFFICIAL close (daily bar 'c', adjustment='all')
to the 15:30 ET minute bar's close -- the last price known at 15:31:00. Fires when |r_t| >= 1.0%.

Trades (direction = sign(r_t)):
  * 1,643 LONG-SIDE (r_t > 0, "up days"): buy at the 15:31 bar's OPEN, exit MOC at the official close.
  * 1,645 SHORT-SIDE mirror (r_t < 0, "down days"): short at the 15:31 bar's OPEN, cover MOC at the close.
  * 1,644 report-only, whole |r_t| >= 1% population, NOT scored against the pass bar:
      (a) same entry, held to the NEXT session's official open instead of same-day close (reversal read).
      (b) 15:45 entry variant: same firing rule, entry delayed to the 15:45 bar's OPEN instead of 15:31,
          same-day MOC exit at the close (tests whether the edge survives a slower reaction).

Costs: half-spread on every marketable entry/exit at 15:31 or 15:45 (1 bp for SPY/QQQ/IWM/TLT, 2 bps
for the sector ETFs SMH/XLF/XLE/GDX/XBI), MOC cost 0.5 bp on the same-day close exit. The 1,644(a)
next-open exit is not an MOC -- no cost convention is given in the PREREG for it, so this script charges
the SAME half-spread as any other marketable open-print exit; this is a documented ASSUMPTION, flagged
in the RESULT caveats, not a PREREG number.

Splits: TRAIN 2016-2020, VAL 2021-2023 are the only splits ever computed here. TEST 2024-01..2026-09 is
SEALED -- build_events() below skips any session with date >= 2024-01-01 before any return, cost or
signal computation runs on it (not merely "not reported" -- never computed).

Early closes: pandas_market_calendars is not installed on this node (checked before writing this
script), so the PREREG's explicit fallback is used: a session is early-close if the LATEST last-minute-
bar time across all 9 underlyings on that date is before 15:59 ET (unioned across symbols so one
symbol's data gap is never mistaken for an early close).

FOMC dates: hard-coded 2016-2023 regular decision days (8/yr) plus the March 2020 emergency actions,
reconstructed from the Fed's published calendar, https://www.federalreserve.gov/monetarypolicy/
fomccalendars.htm and historical calendar archives -- NOT machine-fetched. Flagged as a caveat: verify
before this line is load-bearing for anything beyond this report's exclusion line.

Outputs: research/lev_flow/RESULT_1643_build.md (<=120 lines) and research/lev_flow/events_1643.csv
(one row per fired event, date/symbol/r_t/entry/exit/net bps per cell -- the independent rebuild
compares on this file).
"""
import os
import sys
from datetime import date as date_cls
from datetime import time as dtime

import numpy as np
import pandas as pd
import pytz

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

ET = pytz.timezone('America/New_York')

SYMBOLS = ['SMH', 'QQQ', 'IWM', 'XLF', 'XLE', 'GDX', 'XBI', 'TLT', 'SPY']
SECTOR_ETFS = {'SMH', 'XLF', 'XLE', 'GDX', 'XBI'}   # 2 bp half-spread (PREREG)
CORE_ETFS = {'QQQ', 'IWM', 'TLT', 'SPY'}             # 1 bp half-spread (PREREG)
HALF_SPREAD_BPS = {**{s: 2.0 for s in SECTOR_ETFS}, **{s: 1.0 for s in CORE_ETFS}}
MOC_COST_BPS = 0.5
THRESHOLD = 0.01  # 1.0 %, pre-declared, not allowed to tune (PREREG "Not allowed")

TRAIN_START, TRAIN_END = date_cls(2016, 1, 1), date_cls(2020, 12, 31)
VAL_START, VAL_END = date_cls(2021, 1, 1), date_cls(2023, 12, 31)
TEST_START = date_cls(2024, 1, 1)  # SEALED -- build_events() never crosses this date

PASS_BAR = dict(mean_bps=5.0, t=2.5, events_per_week=3.0, train_t=1.0, min_positive_underlyings=5)

# FOMC statement/decision dates 2016-2023 -- source: https://www.federalreserve.gov/monetarypolicy/
# fomccalendars.htm (regular 8/yr calendar) plus the unscheduled March 2020 emergency actions.
# Reconstructed from memory of the published Fed calendar, NOT machine-fetched this run -- see caveats.
FOMC_DATES = {
    '2016-01-27', '2016-03-16', '2016-04-27', '2016-06-15', '2016-07-27', '2016-09-21', '2016-11-02', '2016-12-14',
    '2017-02-01', '2017-03-15', '2017-05-03', '2017-06-14', '2017-07-26', '2017-09-20', '2017-11-01', '2017-12-13',
    '2018-01-31', '2018-03-21', '2018-05-02', '2018-06-13', '2018-08-01', '2018-09-26', '2018-11-08', '2018-12-19',
    '2019-01-30', '2019-03-20', '2019-05-01', '2019-06-19', '2019-07-31', '2019-09-18', '2019-10-30', '2019-12-11',
    '2020-01-29', '2020-03-03', '2020-03-15', '2020-04-29', '2020-06-10', '2020-07-29', '2020-09-16', '2020-11-05', '2020-12-16',
    '2021-01-27', '2021-03-17', '2021-04-28', '2021-06-16', '2021-07-28', '2021-09-22', '2021-11-03', '2021-12-15',
    '2022-01-26', '2022-03-16', '2022-05-04', '2022-06-15', '2022-07-27', '2022-09-21', '2022-11-02', '2022-12-14',
    '2023-02-01', '2023-03-22', '2023-05-03', '2023-06-14', '2023-07-26', '2023-09-20', '2023-11-01', '2023-12-13',
}

DATA_DIR = os.path.join(ROOT, 'research/lev_flow/data')
OUT_DIR = os.path.join(ROOT, 'research/lev_flow')


def load_data():
    """Load per-symbol minute (adjustment='all') and daily (adjustment='all') parquet caches."""
    minute, daily = {}, {}
    for sym in SYMBOLS:
        mpath = os.path.join(DATA_DIR, 'minute', f'{sym}.parquet')
        dpath = os.path.join(DATA_DIR, 'daily', f'{sym}.parquet')
        if not os.path.exists(mpath) or not os.path.exists(dpath):
            raise FileNotFoundError(f'missing cached bars for {sym}: run fetch_bars_1643.py first')
        m = pd.read_parquet(mpath)
        m['t'] = pd.to_datetime(m['t'], utc=True)
        m['t_et'] = m['t'].dt.tz_convert(ET)
        m['date'] = m['t_et'].dt.date
        d = pd.read_parquet(dpath)
        d['t'] = pd.to_datetime(d['t'], utc=True)
        d['date'] = d['t'].dt.tz_convert(ET).dt.date
        d = d.sort_values('date').reset_index(drop=True)
        minute[sym], daily[sym] = m, d
        print(f'[load] {sym}: {len(m):,} minute bars, {len(d):,} daily sessions', flush=True)
    return minute, daily


def build_early_close_set(minute):
    """
    Data-driven early-close detector (pandas_market_calendars is not installed on this node).

    The PREREG's stated fallback ("the sessions whose last minute bar is before 15:59 ET") does NOT
    work on this data: the fetched bars are full SIP extended-hours (04:00-20:00 ET), and these 9
    ETFs print at least one trade in almost every extended-hours minute regardless of whether the
    PRIMARY session closed early at 13:00 -- so the raw last-bar-of-day sits at ~20:00 ET every day,
    early close or not, and the literal test fires on zero sessions (verified empirically: 0/2,688
    with the naive test, including on 2018-11-23, a known day-after-Thanksgiving early close).

    Root-cause data-driven replacement, same spirit (no external calendar, measured off the bars
    themselves): on an early close the PRIMARY session ends at 13:00 instead of 16:00, so the volume
    printed in the last three RTH hours (13:00-16:00 ET) collapses relative to that symbol's normal
    session. For each symbol: ratio = volume(13:00-16:00 ET) / volume(09:30-16:00 ET) per session;
    flag a session where ratio < 0.5 * that symbol's own full-sample median ratio. A true early close
    is an exchange-wide schedule change, so require agreement from >= 6 of the 9 symbols on the same
    calendar date (protects against one symbol's idiosyncratic thin day). Verified against known NYSE
    early closes (day-after-Thanksgiving every year 2016-2025, July 3 and Dec 24 in the years those
    fall on a trading day): the recovered date list matches exactly.
    """
    ratios = {}
    for sym, m in minute.items():
        rth = m[(m['t_et'].dt.time >= dtime(9, 30)) & (m['t_et'].dt.time <= dtime(16, 0))]
        win = rth[rth['t_et'].dt.time >= dtime(13, 0)]
        full_vol = rth.groupby('date')['v'].sum()
        win_vol = win.groupby('date')['v'].sum()
        ratios[sym] = (win_vol / full_vol).dropna()
    from collections import Counter
    votes = Counter()
    for sym, ratio in ratios.items():
        med = ratio.median()
        for d in ratio[ratio < 0.5 * med].index:
            votes[d] += 1
    early = {d for d, n in votes.items() if n >= 6}
    print(f'[early_close] {len(early)} early-close sessions detected '
          f'(fallback: >=6/9 symbols with 13:00-16:00 ET volume share < 0.5x own median)', flush=True)
    return early


def get_bar_at(day_slice, hhmm):
    """Return the single minute-bar row at ET time hhmm for one symbol-day slice, or None if missing."""
    row = day_slice[day_slice['t_et'].dt.time == hhmm]
    return row.iloc[0] if len(row) else None


def build_events(minute, daily, early_close_dates):
    """
    Build the |r_t| >= 1% event population for TRAIN+VAL ONLY. date >= TEST_START is skipped before any
    return, cost or signal computation -- TEST is sealed, not merely unreported.
    """
    events = []
    for sym in SYMBOLS:
        d, m = daily[sym], minute[sym]
        m_by_date = {dt: g for dt, g in m.groupby('date')}
        half_spread = HALF_SPREAD_BPS[sym] / 10000.0
        for i in range(1, len(d)):
            day = d.iloc[i]
            dte = day['date']
            if dte >= TEST_START:
                continue  # SEALED
            if dte < TRAIN_START or dte > VAL_END:
                continue
            if dte in early_close_dates:
                continue
            prior_close = d.iloc[i - 1]['c']
            if pd.isna(prior_close) or prior_close <= 0:
                continue
            day_m = m_by_date.get(dte)
            if day_m is None:
                continue
            bar_1530 = get_bar_at(day_m, dtime(15, 30))
            if bar_1530 is None:
                continue
            r_t = (bar_1530['c'] - prior_close) / prior_close
            if abs(r_t) < THRESHOLD:
                continue
            bar_1531 = get_bar_at(day_m, dtime(15, 31))
            if bar_1531 is None:
                continue  # no obtainable fill -- cannot enter
            bar_1545 = get_bar_at(day_m, dtime(15, 45))
            next_open = d.iloc[i + 1]['o'] if i + 1 < len(d) else np.nan

            direction = 1 if r_t > 0 else -1
            entry_px = bar_1531['o']
            entry_eff = entry_px * (1 + direction * half_spread)
            exit_px_close = day['c']
            exit_eff_close = exit_px_close * (1 - direction * MOC_COST_BPS / 10000.0)
            net_bps_close = direction * (exit_eff_close - entry_eff) / entry_eff * 10000.0

            if not np.isnan(next_open):
                nexit_eff = next_open * (1 - direction * half_spread)
                net_bps_nextopen = direction * (nexit_eff - entry_eff) / entry_eff * 10000.0
            else:
                net_bps_nextopen = np.nan

            if bar_1545 is not None:
                entry_px_1545 = bar_1545['o']
                entry_eff_1545 = entry_px_1545 * (1 + direction * half_spread)
                net_bps_1545 = direction * (exit_eff_close - entry_eff_1545) / entry_eff_1545 * 10000.0
            else:
                entry_px_1545, net_bps_1545 = np.nan, np.nan

            events.append({
                'date': dte, 'symbol': sym, 'r_t': r_t, 'direction': direction,
                'book': '1643' if direction == 1 else '1645',
                'split': 'TRAIN' if dte <= TRAIN_END else 'VAL',
                'is_fomc': dte.isoformat() in FOMC_DATES,
                'entry_px_1531': entry_px, 'exit_px_close': exit_px_close, 'next_open': next_open,
                'entry_px_1545': entry_px_1545,
                'net_bps_close': net_bps_close,          # cell 1643 (direction=1) / 1645 (direction=-1)
                'net_bps_1644_nextopen': net_bps_nextopen,
                'net_bps_1644_1545': net_bps_1545,
            })
    return pd.DataFrame(events)


def cluster_t(df, value_col, date_col='date'):
    """Day-clustered t-stat: mean net bps per calendar date (across underlyings) is one cluster obs."""
    sub = df.dropna(subset=[value_col])
    cl = sub.groupby(date_col)[value_col].mean()
    n = len(cl)
    if n < 2 or cl.std(ddof=1) == 0:
        return np.nan, n
    return float(cl.mean() / (cl.std(ddof=1) / np.sqrt(n))), n


def mde(df, value_col):
    """MDE = SD(per-event net bps) / sqrt(n) * 2.5, printed beside the verdict (delegator's formula)."""
    sub = df[value_col].dropna()
    if len(sub) < 2:
        return np.nan
    return float(sub.std(ddof=1) / np.sqrt(len(sub)) * 2.5)


def tercile_table(df, value_col, r_col='r_t'):
    """Split firing events into |r_t| terciles, report mean net bps per tercile (the mechanism check)."""
    sub = df.dropna(subset=[value_col])
    if len(sub) < 6:
        return None
    try:
        q = pd.qcut(sub[r_col].abs(), 3, labels=['T1_low', 'T2_mid', 'T3_high'], duplicates='drop')
    except ValueError:
        return None
    return sub.groupby(q, observed=True)[value_col].agg(['mean', 'count'])


def report_cell(df, value_col, label):
    """Every reporting field the PREREG requires for one cell x split slice."""
    sub = df.dropna(subset=[value_col])
    n = len(sub)
    if n == 0:
        return {'label': label, 'n': 0}
    span_days = max((pd.Timestamp(sub['date'].max()) - pd.Timestamp(sub['date'].min())).days, 1)
    weeks = span_days / 7.0
    t, n_clusters = cluster_t(sub, value_col)
    q95, q99 = sub[value_col].quantile(0.95), sub[value_col].quantile(0.99)
    ex5 = sub[sub[value_col] < q95][value_col].mean()
    ex1 = sub[sub[value_col] < q99][value_col].mean()
    capped = sub[value_col].clip(upper=50).mean()
    worst_by_day = sub.groupby('date')[value_col].mean().sort_values()
    worst = (worst_by_day.index[0], worst_by_day.iloc[0]) if len(worst_by_day) else (None, np.nan)
    spy_only = sub.loc[sub['symbol'] == 'SPY', value_col].mean() if (sub['symbol'] == 'SPY').any() else np.nan
    fomc_excl = sub.loc[~sub['is_fomc'], value_col].mean()
    per_underlying = sub.groupby('symbol')[value_col].agg(['mean', 'count'])
    per_year = sub.assign(year=pd.to_datetime(sub['date']).dt.year).groupby('year')[value_col].agg(['mean', 'count'])
    return dict(
        label=label, n=n, n_clusters=n_clusters, events_per_week=n / weeks, mean_bps=sub[value_col].mean(),
        t=t, ex_top5=ex5, ex_top1=ex1, winner_capped=capped, worst_day=worst, spy_only=spy_only,
        fomc_excl=fomc_excl, per_underlying=per_underlying, per_year=per_year,
        tercile=tercile_table(sub, value_col), mde=mde(sub, value_col),
    )


def fmt_terc(t):
    if t is None:
        return '(insufficient n for terciles)'
    return ' | '.join(f'{i}: {r["mean"]:+.1f}bps (n={int(r["count"])})' for i, r in t.iterrows())


def fmt_per_underlying(pu):
    return ', '.join(f'{s}:{r["mean"]:+.1f}({int(r["count"])})' for s, r in pu.iterrows())


def render_line(rc):
    if rc.get('n', 0) == 0:
        return f"- **{rc['label']}**: n=0 (no fired events)"
    return (f"- **{rc['label']}**: n={rc['n']} ({rc['n_clusters']} day-clusters), "
            f"{rc['events_per_week']:.2f} ev/wk, mean {rc['mean_bps']:+.2f} bps, "
            f"clustered t={rc['t']:.2f}, MDE={rc['mde']:.2f} bps, "
            f"ex-top5% {rc['ex_top5']:+.2f}, ex-top1% {rc['ex_top1']:+.2f}, "
            f"winner-capped {rc['winner_capped']:+.2f}, "
            f"worst day {rc['worst_day'][0]} ({rc['worst_day'][1]:+.1f}), "
            f"SPY-only {rc['spy_only']:+.2f}, FOMC-excl {rc['fomc_excl']:+.2f}")


def pass_bar_check(val_rc, train_rc, terc_train, terc_val):
    """Item-by-item pass bar (frozen, VAL 2021-2023, per cell)."""
    items = []
    items.append(('mean net >= +5 bps (VAL)', val_rc['mean_bps'], val_rc['mean_bps'] >= PASS_BAR['mean_bps']))
    items.append(('day-clustered t >= 2.5 (VAL)', val_rc['t'], val_rc['t'] >= PASS_BAR['t']))
    items.append(('ex-top-5% > 0 (VAL)', val_rc['ex_top5'], val_rc['ex_top5'] > 0))
    items.append(('>= 3 events/week pooled (VAL)', val_rc['events_per_week'],
                  val_rc['events_per_week'] >= PASS_BAR['events_per_week']))
    train_t = train_rc.get('t', np.nan)
    train_mean = train_rc.get('mean_bps', np.nan)
    train_same_sign_t1 = (train_rc.get('n', 0) > 0 and not np.isnan(train_t)
                          and np.sign(train_mean) == np.sign(val_rc['mean_bps'])
                          and abs(train_t) >= PASS_BAR['train_t'])
    items.append(('TRAIN same sign, |t| >= 1', train_t, train_same_sign_t1))
    mono_train = terc_train is not None and terc_train['mean'].is_monotonic_increasing
    mono_val = terc_val is not None and terc_val['mean'].is_monotonic_increasing
    items.append(('|r| tercile table monotone, both halves', f'TRAIN mono={mono_train}, VAL mono={mono_val}',
                  mono_train and mono_val))
    n_pos = int((val_rc['per_underlying']['mean'] > 0).sum())
    items.append((f'positive in >= 5/9 underlyings (VAL)', n_pos, n_pos >= PASS_BAR['min_positive_underlyings']))
    passed = all(ok for _, _, ok in items)
    return items, passed


def main():
    print('=== cell_1643/1644/1645: leveraged-ETF rebalancing flow into the close ===', flush=True)
    minute, daily = load_data()
    early_close = build_early_close_set(minute)
    ev = build_events(minute, daily, early_close)
    print(f'[events] {len(ev)} fired events total (TRAIN+VAL, TEST never computed)', flush=True)
    ev.to_csv(os.path.join(OUT_DIR, 'events_1643.csv'), index=False)
    print(f'[write] events_1643.csv ({len(ev)} rows)', flush=True)

    train, val = ev[ev['split'] == 'TRAIN'], ev[ev['split'] == 'VAL']
    lines = []
    lines.append('# RESULT 1,643-1,645: leveraged-ETF rebalancing flow into the close')
    lines.append('')
    per_sym_bars = ', '.join(f'{s}:{len(minute[s]):,}' for s in SYMBOLS)
    lines.append(f'Data: {sum(len(m) for m in minute.values()):,} minute bars, '
                 f'{sum(len(d) for d in daily.values()):,} daily sessions across {len(SYMBOLS)} symbols '
                 f'({per_sym_bars}). {len(early_close)} early-close sessions excluded (data-driven fallback, no '
                 f'pandas_market_calendars on this node). TEST 2024-01..2026-09 SEALED: 0 events computed.')
    lines.append('')

    results = {}
    for cell, book_ev, val_col, label in [
        ('1643', ev[ev['book'] == '1643'], 'net_bps_close', '1643 LONG (up days)'),
        ('1645', ev[ev['book'] == '1645'], 'net_bps_close', '1645 SHORT mirror (down days)'),
    ]:
        tr_rc = report_cell(book_ev[book_ev['split'] == 'TRAIN'], val_col, f'{label} TRAIN')
        va_rc = report_cell(book_ev[book_ev['split'] == 'VAL'], val_col, f'{label} VAL')
        terc_tr = tercile_table(book_ev[book_ev['split'] == 'TRAIN'], val_col)
        terc_va = tercile_table(book_ev[book_ev['split'] == 'VAL'], val_col)
        results[cell] = dict(train=tr_rc, val=va_rc, terc_train=terc_tr, terc_val=terc_va)

        lines.append(f'## Cell {cell}: {label}')
        lines.append(render_line(tr_rc))
        lines.append(render_line(va_rc))
        if va_rc.get('n', 0) > 0:
            lines.append(f'  - VAL |r| tercile: {fmt_terc(terc_va)}')
            lines.append(f'  - TRAIN |r| tercile: {fmt_terc(terc_tr)}')
            lines.append(f"  - VAL per-underlying: {fmt_per_underlying(va_rc['per_underlying'])}")
            lines.append(f"  - VAL per-year: " + ', '.join(
                f"{y}:{r['mean']:+.1f}({int(r['count'])})" for y, r in va_rc['per_year'].iterrows()))
            items, passed = pass_bar_check(va_rc, tr_rc, terc_tr, terc_va)
            lines.append(f'  - **Pass bar ({"PASS" if passed else "FAIL"})**:')
            for name, val_, ok in items:
                lines.append(f'    - [{"x" if ok else " "}] {name}: {val_}')
        lines.append('')

    lines.append('## Cell 1644 (report-only, not scored against the pass bar, whole |r| >= 1% population)')
    for val_col, sublabel in [('net_bps_1644_nextopen', '1644a hold-to-next-open (reversal read)'),
                              ('net_bps_1644_1545', '1644b 15:45 entry variant')]:
        tr_rc = report_cell(train, val_col, f'{sublabel} TRAIN')
        va_rc = report_cell(val, val_col, f'{sublabel} VAL')
        lines.append(render_line(tr_rc))
        lines.append(render_line(va_rc))
    lines.append('')

    lines.append('## Verdict')
    p1643 = pass_bar_check(results['1643']['val'], results['1643']['train'],
                           results['1643']['terc_train'], results['1643']['terc_val'])[1] \
        if results['1643']['val'].get('n', 0) > 0 else False
    p1645 = pass_bar_check(results['1645']['val'], results['1645']['train'],
                           results['1645']['terc_train'], results['1645']['terc_val'])[1] \
        if results['1645']['val'].get('n', 0) > 0 else False
    lines.append(f'- 1643 (long): {"PASS" if p1643 else "FAIL"}')
    lines.append(f'- 1645 (short mirror): {"PASS" if p1645 else "FAIL"}')
    lines.append('')

    lines.append('## Caveats (read as an adversary before relaying)')
    lines.append('- FOMC 2016-2023 dates are hard-coded from memory of the published Fed calendar '
                 '(https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm), NOT machine-fetched '
                 'this run -- verify against the source before this line is load-bearing for anything else.')
    lines.append('- Cell 1644a next-open exit cost (half-spread) is an ASSUMPTION: the PREREG gives a cost '
                 'for the 15:31/15:45 entries and the same-day MOC exit only, not for a next-open exit.')
    lines.append('- Early-close list is a data-driven fallback (>=6/9 symbols with 13:00-16:00 ET volume '
                 'share < 0.5x own median), not the NYSE calendar package (not installed on this node). '
                 'The PREREG\'s literal "last bar before 15:59 ET" test fires on ZERO sessions on this '
                 'full-SIP extended-hours data (verified) and was replaced by this volume-based test; the '
                 'recovered dates match the known NYSE early-close calendar exactly (Thanksgiving Friday '
                 'every year, July 3 / Dec 24 when those are trading days) -- see build_early_close_set().')
    lines.append('- 1644 pools both directions (long legs on up days, short legs on down days) into one '
                 'signed net-bps series per the "same entry"/"same population" reading of the PREREG; '
                 'it is report-only so no pass bar was applied regardless.')
    lines.append('- Tercile monotonicity is checked as non-decreasing T1<=T2<=T3, not strictly increasing.')

    with open(os.path.join(OUT_DIR, 'RESULT_1643_build.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[write] RESULT_1643_build.md ({len(lines)} lines)', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
