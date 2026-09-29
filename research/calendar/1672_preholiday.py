"""
Cell 1,672 -- pre-holiday index sleeve (PREREG_1672.md, FROZEN 2026-09-29).

Mechanism: buy SPY at the close of the session BEFORE a US market holiday (MOC, 1bp),
sell at the close of the next session (MOC, 1bp). Variant B: exit at the next OPEN
(+open-auction 2bps) instead of the next close. Variant C: QQQ instead of SPY.
Mirror (post-holiday, decay check): identical event shape but anchored on the session
AFTER the holiday instead of the session before it.

Data: data/cache.db daily_bars (read-only) for SPY/QQQ. On this node that table covers
only 2024-06-03..2026-09-28 (SPY) / 2024-12-30..2026-09-28 (QQQ) -- NOT the 2016-2026
range the PREREG's Data section names. This is reported as a mismatch, not silently
patched; every read below runs on the population actually on disk.

Holiday detection: a weekday absent from a symbol's own daily_bars date set = holiday.
Verified by hand against the known NYSE calendar (fixed-date holidays observed the
nearest weekday when they fall on a weekend) for 2024-06..2026-09.
"""
import sqlite3
import numpy as np
import pandas as pd
from datetime import date, timedelta
from dateutil.easter import easter

RNG_SEED = 1672
N_DRAWS = 1000
NOTIONAL = 60000.0          # $ per event, TOM sleeve size (PREREG Reads line)
BASELINE_EVENTS_PER_YEAR = 9.5   # PREREG "~9/year" -- midpoint of 9-10
CSV_PATH = 'research/calendar/spy_qqq_daily_2016_2026.csv'

# ---- known one-off NYSE closures (not on the fixed annual schedule) ----
ONEOFF_CLOSURES = {
    date(2018, 12, 5): "National Day of Mourning (Pres. G.H.W. Bush)",
    date(2025, 1, 9): "National Day of Mourning (Pres. Carter)",
}


def _nearest_weekday(d):
    """NYSE observed-date rule for fixed-date holidays: Sat -> prev Fri, Sun -> next Mon."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year, month, weekday, n):
    """n-th (1-indexed) given weekday (Mon=0) of a month."""
    d = date(year, month, 1)
    count = 0
    while True:
        if d.weekday() == weekday:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)


def _last_weekday(year, month, weekday):
    """Last given weekday (Mon=0) of a month."""
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def nyse_holidays(year):
    """Explicit, rule-based NYSE holiday calendar for one year (observed dates).
    Rules: New Year's/Juneteenth/Independence Day/Christmas = fixed date, nearest weekday
    if it falls on a weekend; MLK = 3rd Mon Jan; Washington's Birthday = 3rd Mon Feb;
    Good Friday = Easter Sunday - 2 days; Memorial Day = last Mon May; Labor Day = 1st Mon
    Sep; Thanksgiving = 4th Thu Nov. Juneteenth is an NYSE holiday only from 2022 onward.
    New Year's Day is the one fixed-date holiday NYSE does NOT roll back to the preceding
    Friday when it falls on a Saturday (Dec 31 stays a trading day -- confirmed against the
    fetched data: 2021-12-31, before the 2022-01-01 Saturday, traded on both SPY and QQQ);
    it still rolls forward to Monday when Jan 1 falls on a Sunday."""
    out = {}
    new_years = date(year, 1, 1)
    if new_years.weekday() == 6:          # Sunday -> observed Monday
        out[new_years + timedelta(days=1)] = "New Year's"
    elif new_years.weekday() != 5:        # Saturday -> not observed at all
        out[new_years] = "New Year's"
    out.update({
        _nth_weekday(year, 1, 0, 3): "MLK Day",
        _nth_weekday(year, 2, 0, 3): "Washington's Birthday",
        easter(year) - timedelta(days=2): "Good Friday",
        _last_weekday(year, 5, 0): "Memorial Day",
        _nearest_weekday(date(year, 7, 4)): "Independence Day",
        _nth_weekday(year, 9, 0, 1): "Labor Day",
        _nth_weekday(year, 11, 3, 4): "Thanksgiving",
        _nearest_weekday(date(year, 12, 25)): "Christmas",
    })
    if year >= 2022:
        out[_nearest_weekday(date(year, 6, 19))] = "Juneteenth"
    return out


def known_holidays_in_range(lo, hi):
    """Programmatic recurring calendar (nyse_holidays) + hand-known one-off closures,
    restricted to [lo, hi]. This is the 'hand-verified NYSE list' the PREREG asks for --
    built from explicit, auditable rules rather than a black-box calendar library."""
    known = {}
    for y in range(lo.year, hi.year + 1):
        known.update(nyse_holidays(y))
    known.update(ONEOFF_CLOSURES)
    return {d: n for d, n in known.items() if lo <= d <= hi}


def load_daily(symbol):
    """Read-only load of one symbol's daily bars from cache.db. Never opens for write."""
    conn = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)
    df = pd.read_sql_query(
        "SELECT bar_date, open, close FROM daily_bars WHERE symbol = ? ORDER BY bar_date",
        conn, params=(symbol,), parse_dates=['bar_date'])
    conn.close()
    df['bar_date'] = df['bar_date'].dt.date
    return df.set_index('bar_date')


def load_daily_csv(symbol, csv_path=CSV_PATH):
    """Load one symbol's daily bars from the Alpaca-fetched 2016-2026 CSV (full span)."""
    df = pd.read_csv(csv_path, parse_dates=['bar_date'])
    df = df[df.symbol == symbol].copy()
    df['bar_date'] = df['bar_date'].dt.date
    return df.set_index('bar_date')[['open', 'close']].sort_index()


def verify_vs_cache_db(csv_df, symbol, tol=0.01):
    """Cross-check the fetched CSV's close prices against data/cache.db (read-only) for
    every overlapping date; report the count of >1ct mismatches. Never writes cache.db."""
    db_df = load_daily(symbol)
    joined = csv_df[['close']].join(db_df[['close']], how='inner', lsuffix='_csv', rsuffix='_db')
    diff = (joined['close_csv'] - joined['close_db']).abs()
    n_checked = len(joined)
    n_bad = int((diff > tol).sum())
    max_diff = float(diff.max()) if n_checked else float('nan')
    return n_checked, n_bad, max_diff


def detect_holidays(calendar_dates, lo, hi):
    """Weekday absent from the trading calendar between lo/hi (inclusive) = holiday."""
    out = []
    d = lo
    cal = set(calendar_dates)
    while d <= hi:
        if d.weekday() < 5 and d not in cal:
            out.append(d)
        d += timedelta(days=1)
    return out


def verify_holiday_count(detected, lo, hi):
    """Compare detected weekday-gaps vs the rule-based NYSE list; report mismatches."""
    known_in_range = known_holidays_in_range(lo, hi)
    detected_set = set(detected)
    known_set = set(known_in_range)
    missing = sorted(known_set - detected_set)   # known holiday, no gap in data (data hole?)
    extra = sorted(detected_set - known_set)      # gap in data, not a known holiday
    per_year = {}
    for d in detected:
        per_year[d.year] = per_year.get(d.year, 0) + 1
    return known_in_range, missing, extra, per_year


def build_pairs(prior_dates):
    """Consecutive (day_i, day_i+1) pairs from a sorted list of trading dates."""
    return list(zip(prior_dates[:-1], prior_dates[1:]))


def cost_bps(variant):
    """Round-trip cost per PREREG mechanism: MOC 1bp both legs, or 2bp open-auction exit (B)."""
    return 3.0 if variant == 'B' else 2.0


def event_return(bars, entry_d, exit_d, variant):
    """gross/net bps for one event given entry/exit dates and the price columns used."""
    if entry_d not in bars.index or exit_d not in bars.index:
        return None
    entry_px = bars.loc[entry_d, 'close']
    exit_px = bars.loc[exit_d, 'open'] if variant == 'B' else bars.loc[exit_d, 'close']
    if pd.isna(entry_px) or pd.isna(exit_px) or entry_px <= 0:
        return None
    gross = (exit_px - entry_px) / entry_px * 10000.0
    return entry_px, exit_px, gross, gross - cost_bps(variant)


def half_of(year):
    return 'odd' if year % 2 == 1 else 'even'


def cluster_t(x):
    """iid t (== day-clustered here: at most one event per calendar day in this population)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return np.nan, np.nan, n
    sd = x.std(ddof=1)
    se = sd / np.sqrt(n)
    t = x.mean() / se if se > 0 else np.nan
    return x.mean(), t, n


def mde(x):
    """PREREG cell-1672 formula (no reusable count-matched-null/MDE helper found in
    research/index_overnight/*.py -- cell_1649.py's mde_clustered() uses a different,
    study-specific 2.5x multiplier for a different bar; not reused here)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    return 2.8 * x.std(ddof=1) / np.sqrt(len(x))


def ex_top5(x):
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    k = max(1, int(np.ceil(0.05 * len(x))))
    return x[k:].mean() if len(x) > k else np.nan


def null_percentile(actual_mean, pool_returns, n_events, variant, rng):
    """1,000 draws of n_events ordinary (non-holiday-adjacent) session pairs, same cost
    basis as the variant; percentile rank of the actual sleeve mean in that distribution."""
    pool = np.asarray(pool_returns, dtype=float)
    if len(pool) == 0 or n_events == 0:
        return np.nan, 0
    draws = np.empty(N_DRAWS)
    replace = len(pool) < n_events
    for i in range(N_DRAWS):
        draws[i] = rng.choice(pool, size=n_events, replace=replace).mean()
    pct = float((draws < actual_mean).mean() * 100.0)
    return pct, len(pool)


def main():
    print('[1672] loading SPY/QQQ daily bars from the Alpaca 2016-2026 CSV (full span) ...')
    spy = load_daily_csv('SPY')
    qqq = load_daily_csv('QQQ')
    print(f'[1672] SPY {spy.index.min()}..{spy.index.max()} n={len(spy)}; '
          f'QQQ {qqq.index.min()}..{qqq.index.max()} n={len(qqq)}')

    # cross-check fetched closes vs the on-disk cache.db (read-only) on the overlap window
    n_chk_spy, n_bad_spy, maxd_spy = verify_vs_cache_db(spy, 'SPY')
    n_chk_qqq, n_bad_qqq, maxd_qqq = verify_vs_cache_db(qqq, 'QQQ')
    print(f'[1672] cross-check vs cache.db: SPY {n_chk_spy} dates checked, {n_bad_spy} >1ct '
          f'mismatches, max diff ${maxd_spy:.4f}')
    print(f'[1672] cross-check vs cache.db: QQQ {n_chk_qqq} dates checked, {n_bad_qqq} >1ct '
          f'mismatches, max diff ${maxd_qqq:.4f}')

    spy_cal = sorted(spy.index)
    qqq_cal = sorted(qqq.index)

    spy_holidays = detect_holidays(spy_cal, spy_cal[0], spy_cal[-1])
    qqq_holidays = detect_holidays(qqq_cal, qqq_cal[0], qqq_cal[-1])

    known_spy, miss_spy, extra_spy, py_spy = verify_holiday_count(spy_holidays, spy_cal[0], spy_cal[-1])
    known_qqq, miss_qqq, extra_qqq, py_qqq = verify_holiday_count(qqq_holidays, qqq_cal[0], qqq_cal[-1])
    print(f'[1672] SPY holidays detected={len(spy_holidays)} known-in-range={len(known_spy)} '
          f'missing={miss_spy} extra={extra_spy} per_year={py_spy}')
    print(f'[1672] QQQ holidays detected={len(qqq_holidays)} known-in-range={len(known_qqq)} '
          f'missing={miss_qqq} extra={extra_qqq} per_year={py_qqq}')

    # ordinary (non-holiday-adjacent) session pairs, per symbol, for the count-matched null
    def ordinary_pairs(cal, holidays):
        holiday_adjacent = set(holidays)
        pairs = build_pairs(cal)
        clean = []
        for a, b in pairs:
            # drop any pair touching a holiday date's neighbourhood (prior/next/next-next)
            if any(abs((a - h).days) <= 4 or abs((b - h).days) <= 4 for h in holiday_adjacent):
                continue
            clean.append((a, b))
        return clean

    spy_ordinary = ordinary_pairs(spy_cal, spy_holidays)
    qqq_ordinary = ordinary_pairs(qqq_cal, qqq_holidays)

    events = []
    variants = [('A', 'SPY', spy, spy_cal, spy_holidays), ('B', 'SPY', spy, spy_cal, spy_holidays),
                ('C', 'QQQ', qqq, qqq_cal, qqq_holidays)]
    for vcode, sym, bars, cal, hols in variants:
        for h in hols:
            before = [d for d in cal if d < h]
            after = [d for d in cal if d > h]
            if not before or not after:
                continue
            prior_s, next_s = before[-1], after[0]
            after2 = [d for d in cal if d > next_s]
            next2_s = after2[0] if after2 else None

            r = event_return(bars, prior_s, next_s, vcode)
            if r:
                entry_px, exit_px, gross, net = r
                events.append(dict(date=h.isoformat(), etf=sym, variant=vcode, leg='pre',
                                    entry_date=prior_s.isoformat(), exit_date=next_s.isoformat(),
                                    entry_close=entry_px, exit_price=exit_px,
                                    gross_bps=gross, net_bps=net, year=h.year,
                                    half=half_of(h.year)))
            if next2_s:
                r2 = event_return(bars, next_s, next2_s, vcode)
                if r2:
                    entry_px, exit_px, gross, net = r2
                    events.append(dict(date=h.isoformat(), etf=sym, variant=vcode, leg='post',
                                        entry_date=next_s.isoformat(), exit_date=next2_s.isoformat(),
                                        entry_close=entry_px, exit_price=exit_px,
                                        gross_bps=gross, net_bps=net, year=h.year,
                                        half=half_of(h.year)))

    ev = pd.DataFrame(events)
    ev.to_csv('research/calendar/1672_events.csv', index=False)
    print(f'[1672] wrote 1672_events.csv ({len(ev)} rows)')

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for vcode, sym in [('A', 'SPY'), ('B', 'SPY'), ('C', 'QQQ')]:
        ordinary = spy_ordinary if sym == 'SPY' else qqq_ordinary
        bars = spy if sym == 'SPY' else qqq
        pool_net = []
        for a, b in ordinary:
            r = event_return(bars, a, b, vcode)
            if r:
                pool_net.append(r[3])
        for leg in ['pre', 'post']:
            for half in ['odd', 'even']:
                sub = ev[(ev.variant == vcode) & (ev.leg == leg) & (ev.half == half)]
                x = sub['net_bps'].values
                mean_bps, t, n = cluster_t(x)
                worst = x.min() if n else np.nan
                hit = float((x > 0).mean()) if n else np.nan
                m = mde(x)
                extop5 = ex_top5(x) if n else np.nan
                pct, poolsize = null_percentile(mean_bps, pool_net, n, vcode, rng) if n else (np.nan, 0)
                rows.append(dict(variant=vcode, etf=sym, leg=leg, half=half, n=n,
                                  mean_net_bps=mean_bps, t=t, mde_bps=m, hit_rate=hit,
                                  worst_bps=worst, ex_top5_bps=extop5, null_pctile=pct,
                                  pool_n=poolsize))
    stats = pd.DataFrame(rows)
    stats.to_csv('research/calendar/1672_stats_fullspan.csv', index=False)
    print(stats.to_string(index=False))

    # $/month, primary variant A pre-holiday, pooled across halves
    a_pre = ev[(ev.variant == 'A') & (ev.leg == 'pre')]
    mean_a = a_pre['net_bps'].mean()
    months_spanned = (spy_cal[-1] - spy_cal[0]).days / 30.44
    events_per_month_actual = len(a_pre) / months_spanned if months_spanned > 0 else np.nan
    events_per_month_baseline = BASELINE_EVENTS_PER_YEAR / 12.0
    dollars_actual = mean_a / 10000.0 * NOTIONAL * events_per_month_actual
    dollars_baseline = mean_a / 10000.0 * NOTIONAL * events_per_month_baseline
    print(f'[1672] variant A pre-holiday pooled: n={len(a_pre)} mean={mean_a:.2f}bps '
          f'events/mo actual={events_per_month_actual:.3f} (${dollars_actual:,.0f}/mo) '
          f'baseline={events_per_month_baseline:.3f} (${dollars_baseline:,.0f}/mo)')

    # pass bar (primary = variant A, pre-holiday leg, both halves)
    a_odd = stats[(stats.variant == 'A') & (stats.leg == 'pre') & (stats.half == 'odd')].iloc[0]
    a_even = stats[(stats.variant == 'A') & (stats.leg == 'pre') & (stats.half == 'even')].iloc[0]
    passes = all([
        a_odd.mean_net_bps >= 8, a_even.mean_net_bps >= 8,
        a_odd.t >= 2.0, a_even.t >= 2.0,
        a_odd.null_pctile >= 95, a_even.null_pctile >= 95,
        a_odd.worst_bps > -300, a_even.worst_bps > -300,
    ])
    print(f'[1672] PASS BAR (variant A, pre-holiday): odd={dict(a_odd)} even={dict(a_even)} -> '
          f'PASSES={passes}')

    with open('research/calendar/1672_summary_fullspan.txt', 'w') as f:
        f.write(stats.to_string(index=False) + '\n')
        f.write(f'\nvariant A pre-holiday pooled n={len(a_pre)} mean={mean_a:.2f}bps '
                f'$/mo actual={dollars_actual:,.0f} baseline={dollars_baseline:,.0f}\n')
        f.write(f'PASSES={passes}\n')
        f.write(f'SPY holiday count-check: detected={len(spy_holidays)} known={len(known_spy)} '
                f'missing={miss_spy} extra={extra_spy} per_year={py_spy}\n')
        f.write(f'QQQ holiday count-check: detected={len(qqq_holidays)} known={len(known_qqq)} '
                f'missing={miss_qqq} extra={extra_qqq} per_year={py_qqq}\n')
        f.write(f'cross-check vs cache.db: SPY {n_chk_spy} checked {n_bad_spy} bad maxdiff '
                f'{maxd_spy:.4f}; QQQ {n_chk_qqq} checked {n_bad_qqq} bad maxdiff {maxd_qqq:.4f}\n')

    print('[1672] done')


if __name__ == '__main__':
    main()
