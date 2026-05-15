"""BF pattern-detection vs breakout-time study (research, 2026-05-15).

For every BULL FLAG SETUP logged this week (May 11-14, the 4 days the
service ran), measure:
  - When the pattern detector first fired (`BULL FLAG SETUP` log line).
  - When intraday `bar.high` first crossed the buy-stop trigger
    (`breakout_level` from the same log line).
  - The lag between the two.

Three categories:
  A) Breakout ALREADY FIRED before detection — fundamentally late;
     the buy-stop is placed in price territory the market has left.
  B) Breakout fires AFTER detection within the 60-min time-stop window —
     the buy-stop should have triggered (subject to fill mechanics +
     pattern invalidation).
  C) Breakout fires AFTER the 60-min time-stop window — AMPG-class:
     the time_stop_minutes config kills the order before the trigger.
  D) Breakout NEVER fires all day — the setup correctly didn't trigger.

For each category, also report what the DAY HIGH did relative to entry
(would-have-won potential).

Reads:
  /tmp/jc_thisweek.log (journalctl dump from earlier this session)
  data/cache.db (intraday_bars_1min)
"""
from __future__ import annotations

import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

LOG_PATH = '/tmp/jc_thisweek.log'
DB_PATH = 'data/cache.db'
TIME_STOP_MIN = 60  # matches config

# Parse: "2026-05-14 13:51:00 | INFO ... | SYM: BULL FLAG SETUP — ... buy-stop @ 9.71"
SETUP_RE = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    r'\s*\|\s*[A-Z]+\s*\|.*?'
    r'\b([A-Z]{2,6})\s*:\s*BULL FLAG SETUP.*?'
    r'buy-stop\s*@\s*([\d.]+)'
)


def load_first_detections() -> dict:
    """{(symbol, date_iso): (detect_ts_utc, breakout_level)} — earliest per pair."""
    first = {}
    with open(LOG_PATH, 'r', errors='replace') as f:
        for line in f:
            m = SETUP_RE.search(line)
            if not m:
                continue
            ts_str, sym, blvl = m.group(1), m.group(2), m.group(3)
            try:
                ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S').replace(
                    tzinfo=timezone.utc
                )
                blvl_f = float(blvl)
            except ValueError:
                continue
            day = ts.date().isoformat()
            key = (sym, day)
            if key not in first or ts < first[key][0]:
                first[key] = (ts, blvl_f)
    return first


def fetch_intraday(symbol: str, day: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        '''SELECT timestamp, high, low, close FROM intraday_bars_1min
           WHERE symbol=? AND date(timestamp)=?
           ORDER BY timestamp''',
        (symbol, day),
    )
    bars = cur.fetchall()
    con.close()
    return bars  # list of (ts_str, high, low, close)


def analyze(detection_ts, breakout_level, bars):
    """Return dict with breakout timing analysis."""
    if not bars:
        return {'no_bars': True}
    # First-ever cross of breakout_level
    first_cross_ts = None
    for ts_str, h, l, c in bars:
        if float(h) >= breakout_level:
            first_cross_ts = datetime.strptime(
                ts_str[:19], '%Y-%m-%dT%H:%M:%S'
            ).replace(tzinfo=timezone.utc) if 'T' in ts_str else \
                datetime.strptime(ts_str[:19], '%Y-%m-%d %H:%M:%S').replace(
                    tzinfo=timezone.utc
                )
            break
    # Post-detection cross (and within time-stop window)
    post_detect_cross_ts = None
    for ts_str, h, l, c in bars:
        bar_ts = datetime.strptime(
            ts_str[:19], '%Y-%m-%dT%H:%M:%S'
        ).replace(tzinfo=timezone.utc) if 'T' in ts_str else \
            datetime.strptime(ts_str[:19], '%Y-%m-%d %H:%M:%S').replace(
                tzinfo=timezone.utc
            )
        if bar_ts >= detection_ts and float(h) >= breakout_level:
            post_detect_cross_ts = bar_ts
            break
    day_high = max(float(b[1]) for b in bars)
    day_close = float(bars[-1][3])
    return {
        'no_bars': False,
        'first_cross_ts': first_cross_ts,
        'post_detect_cross_ts': post_detect_cross_ts,
        'day_high': day_high,
        'day_close': day_close,
    }


def main():
    first = load_first_detections()
    print(f'Parsed {len(first)} unique (symbol, date) BULL FLAG SETUPs.\n')

    rows = []
    no_bars = 0
    for (sym, day), (det_ts, blvl) in sorted(first.items()):
        bars = fetch_intraday(sym, day)
        info = analyze(det_ts, blvl, bars)
        if info.get('no_bars'):
            no_bars += 1
            continue
        first_cross = info['first_cross_ts']
        post = info['post_detect_cross_ts']
        day_high = info['day_high']
        day_close = info['day_close']

        # Categorize — the question that matters is "would the buy-stop have
        # had a chance to trigger?" That depends on whether bar.high crosses
        # breakout_level AT OR AFTER detection (within the 60-min window).
        # Whether it crossed earlier (gap-and-pullback) is informational.
        had_prior_cross = first_cross is not None and first_cross < det_ts
        if post is not None:
            mins_after = (post - det_ts).total_seconds() / 60.0
            if mins_after <= TIME_STOP_MIN:
                cat = 'B_within_window'
            else:
                cat = 'C_past_window'
            lag_min = mins_after
        elif first_cross is not None and first_cross >= det_ts:
            # Edge: identical timestamp = treat as within window
            cat = 'B_within_window'
            lag_min = 0.0
        elif had_prior_cross:
            # Earlier print but never re-crossed post-detect → buy-stop dead
            cat = 'A_prior_only_no_retrigger'
            lag_min = (det_ts - first_cross).total_seconds() / 60.0
        else:
            cat = 'D_never'
            lag_min = None

        gain_pct = (day_high - blvl) / blvl * 100 if blvl > 0 else 0
        close_pct = (day_close - blvl) / blvl * 100 if blvl > 0 else 0
        rows.append({
            'sym': sym, 'day': day, 'det_ts': det_ts, 'blvl': blvl,
            'first_cross': first_cross, 'post_detect_cross': post,
            'cat': cat, 'lag_min': lag_min, 'day_high': day_high,
            'day_close': day_close, 'gain_pct': gain_pct, 'close_pct': close_pct,
        })

    # ---- AGGREGATE ----
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r['cat']].append(r)
    total = len(rows)
    print(f'Analyzed: {total} setups ({no_bars} skipped: no intraday bars).\n')
    print('=' * 92)
    print('TIMING CATEGORY DISTRIBUTION')
    print('=' * 92)
    cat_labels = [
        ('A_prior_only_no_retrigger',
         'A) Earlier print only — never re-crossed post-detect (dead buy-stop)'),
        ('B_within_window',
         'B) Re-crosses within 60-min time-stop window (SHOULD trigger)'),
        ('C_past_window',
         'C) Re-crosses AFTER 60-min time-stop window (time-stop too tight)'),
        ('D_never',
         'D) Never broke out at any point (setup correctly skipped)'),
    ]
    for cat_key, label in cat_labels:
        cat_rows = by_cat[cat_key]
        pct = len(cat_rows) / total * 100 if total else 0
        print(f'\n{label}')
        print(f'  count: {len(cat_rows)}/{total} = {pct:.0f}%')
        if cat_rows and cat_key != 'D_never':
            lags = sorted([r['lag_min'] for r in cat_rows if r['lag_min'] is not None])
            if lags:
                med = lags[len(lags) // 2]
                print(f'  lag median: {med:.0f} min  | min: {lags[0]:.0f}  max: {lags[-1]:.0f}')
            # missed-winner potential — day_high vs entry
            gains = [r['gain_pct'] for r in cat_rows]
            big_winners = sum(1 for g in gains if g >= 5)
            print(f'  day_high >= entry+5%: {big_winners}/{len(cat_rows)} ({big_winners/len(cat_rows)*100:.0f}%)')
            print(f'  median day_high gain over entry: {sorted(gains)[len(gains)//2]:+.1f}%')

    # ---- DETAIL: category A (the structurally dead ones) ----
    a_rows = sorted(by_cat['A_prior_only_no_retrigger'], key=lambda r: -r['lag_min'])
    print('\n' + '=' * 92)
    print('CATEGORY A DETAIL — prior print, never re-crossed post-detect (top 15 by lag)')
    print('=' * 92)
    print(f"{'symbol':<6} {'date':<11} {'detect_utc':<10} {'bo_level':>9} {'prior_lag':>9} "
          f"{'day_high':>9} {'gain%':>7} {'close%':>7}")
    print('-' * 92)
    for r in a_rows[:15]:
        print(f"{r['sym']:<6} {r['day']:<11} {r['det_ts'].strftime('%H:%M:%S'):<10} "
              f"${r['blvl']:>7.2f}  {r['lag_min']:>8.0f}  "
              f"${r['day_high']:>7.2f}  {r['gain_pct']:>+6.1f}%  {r['close_pct']:>+6.1f}%")

    # ---- DETAIL: category B (would-have-triggered) ----
    b_rows = sorted(by_cat['B_within_window'], key=lambda r: -r['gain_pct'])
    if b_rows:
        print('\n' + '=' * 92)
        print(f'CATEGORY B DETAIL — buy-stop should have triggered ({len(b_rows)} cases)')
        print('=' * 92)
        print(f"{'symbol':<6} {'date':<11} {'detect':<10} {'lag_min':>8} {'gain%':>7} {'close%':>7}")
        print('-' * 92)
        for r in b_rows[:15]:
            print(f"{r['sym']:<6} {r['day']:<11} {r['det_ts'].strftime('%H:%M:%S'):<10} "
                  f"{r['lag_min']:>7.0f}  {r['gain_pct']:>+6.1f}%  {r['close_pct']:>+6.1f}%")

    # ---- DETAIL: category C (time-stop too tight) ----
    c_rows = sorted(by_cat['C_past_window'], key=lambda r: -r['gain_pct'])
    if c_rows:
        print('\n' + '=' * 92)
        print(f'CATEGORY C DETAIL — re-cross AFTER 60-min time-stop ({len(c_rows)} cases)')
        print('=' * 92)
        print(f"{'symbol':<6} {'date':<11} {'detect':<10} {'lag_min':>8} {'gain%':>7} {'close%':>7}")
        print('-' * 92)
        for r in c_rows[:10]:
            print(f"{r['sym']:<6} {r['day']:<11} {r['det_ts'].strftime('%H:%M:%S'):<10} "
                  f"{r['lag_min']:>7.0f}  {r['gain_pct']:>+6.1f}%  {r['close_pct']:>+6.1f}%")

    # ---- HEADLINE ----
    a_count = len(by_cat['A_prior_only_no_retrigger'])
    b_count = len(by_cat['B_within_window'])
    c_count = len(by_cat['C_past_window'])
    d_count = len(by_cat['D_never'])
    a_winners = sum(1 for r in by_cat['A_prior_only_no_retrigger'] if r['gain_pct'] >= 5)
    a_big_winners = sum(1 for r in by_cat['A_prior_only_no_retrigger'] if r['gain_pct'] >= 10)
    print('\n' + '=' * 92)
    print('HEADLINE')
    print('=' * 92)
    print(f'  {a_count}/{total} ({a_count/total*100:.0f}%) had a prior print at the buy-stop level but NEVER re-crossed it post-detect.')
    print(f'    → buy-stop placed in dead price territory. {a_winners} ({a_winners/max(a_count,1)*100:.0f}%) had day_high >= entry+5%, {a_big_winners} >= +10%.')
    print(f'  {b_count}/{total} ({b_count/total*100:.0f}%) had a post-detect re-cross within the 60-min time-stop (SHOULD have triggered).')
    print(f'  {c_count}/{total} ({c_count/total*100:.0f}%) had a post-detect re-cross AFTER the 60-min time-stop (window too tight).')
    print(f'  {d_count}/{total} ({d_count/total*100:.0f}%) never broke out at any point — correctly skipped.')


if __name__ == '__main__':
    main()
