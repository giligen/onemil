#!/usr/bin/env python3
"""Independent reimplementation of the N3 zero-cost level-break simulation.

Written from a prose specification only, to cross-check another implementation.
Rule:
  level = h5 (long) / l5 (short); R = 0.10 * atr14
  bars with m >= 575 ascending; trigger = first bar touching level;
  fill = open of the NEXT bar; stop = fill - side*R;
  walk from the fill bar inclusive, stop-out exit = min(stop, o) long /
  max(stop, o) short; else exit at the close of the last bar (eod).
  rr = side * (exit - fill) / R
"""
import math
import sqlite3
from collections import defaultdict

import pandas as pd

BASE = '/home/ec2-user/onemil/research/fuckup_audit/N_databento/N3'
TAPE = f'{BASE}/tape.db'
PICKS = f'{BASE}/top20.csv'
OUT = f'{BASE}/indep_trades.csv'
CHECK = f'{BASE}/indep_check.md'
FIRST_M = 575


def load_picks():
    """Read the daily picks, keeping the ticker 'NA' as a string."""
    df = pd.read_csv(PICKS, keep_default_na=False, na_values=[''])
    return df


def load_bars(conn, days):
    """Load all bars with m >= 575 for the requested days, keyed (day, symbol)."""
    bars = defaultdict(list)
    cur = conn.cursor()
    day_list = sorted(days)
    chunk = 200
    for i in range(0, len(day_list), chunk):
        part = day_list[i:i + chunk]
        q = ('SELECT day, symbol, m, o, h, l, c FROM bars '
             'WHERE m >= ? AND day IN (%s)' % ','.join('?' * len(part)))
        for day, sym, m, o, h, l, c in cur.execute(q, [FIRST_M] + part):
            bars[(day, sym)].append((m, o, h, l, c))
    for k in bars:
        bars[k].sort(key=lambda r: r[0])
    return bars


def simulate_one(side, level, R, rows):
    """Simulate a single pick. Returns (status, entry, exit_px, exit_m, why, rr)."""
    if len(rows) < 2:
        return ('no_trigger', None, None, None, None, None)

    trig = None
    for i, (m, o, h, l, c) in enumerate(rows):
        if (side == 1 and h >= level) or (side == -1 and l <= level):
            trig = i
            break
    if trig is None:
        return ('no_trigger', None, None, None, None, None)
    if trig + 1 >= len(rows):
        return ('no_fill', None, None, None, None, None)

    first = trig + 1
    fill = rows[first][1]  # open of the bar after the trigger bar
    stop_price = fill - side * R

    for j in range(first, len(rows)):
        m, o, h, l, c = rows[j]
        if side == 1 and l <= stop_price:
            exit_px = min(stop_price, o)
            return ('filled', fill, exit_px, m, 'stop', side * (exit_px - fill) / R)
        if side == -1 and h >= stop_price:
            exit_px = max(stop_price, o)
            return ('filled', fill, exit_px, m, 'stop', side * (exit_px - fill) / R)

    m, o, h, l, c = rows[-1]
    return ('filled', fill, c, m, 'eod', side * (c - fill) / R)


def main():
    picks = load_picks()
    total = len(picks)

    conn = sqlite3.connect(f'file:{TAPE}?mode=ro', uri=True)
    bars = load_bars(conn, set(picks['day'].astype(str)))
    conn.close()

    counts = {'no_order': 0, 'no_trigger': 0, 'no_fill': 0, 'filled': 0}
    out_rows = []

    for r in picks.itertuples(index=False):
        side = int(r.side) if str(r.side) != '' and not pd.isna(r.side) else 0
        atr = r.atr14
        try:
            atr = float(atr)
        except (TypeError, ValueError):
            atr = float('nan')
        if side == 0 or not math.isfinite(atr) or atr <= 0:
            counts['no_order'] += 1
            continue

        level = float(r.h5) if side == 1 else float(r.l5)
        R = 0.10 * atr
        rows = bars.get((str(r.day), str(r.symbol)), [])
        status, entry, exit_px, exit_m, why, rr = simulate_one(side, level, R, rows)
        counts[status] += 1
        if status == 'filled':
            out_rows.append({'day': str(r.day), 'symbol': str(r.symbol),
                             'entry': entry, 'exit_px': exit_px, 'exit_m': exit_m,
                             'why': why, 'rr': rr})

    trades = pd.DataFrame(out_rows,
                          columns=['day', 'symbol', 'entry', 'exit_px', 'exit_m', 'why', 'rr'])
    trades.to_csv(OUT, index=False)

    print('=== INDEPENDENT SIM ===')
    print(f'total picks : {total}')
    print(f'no_order    : {counts["no_order"]}')
    print(f'no_trigger  : {counts["no_trigger"]}')
    print(f'no_fill     : {counts["no_fill"]}')
    print(f'filled      : {counts["filled"]}')

    trades['year'] = trades['day'].str[:4]
    print('\nyear   trades   mean_rr   win%     sum_rr')
    for y, g in trades.groupby('year'):
        print(f'{y}   {len(g):6d}   {g.rr.mean():7.4f}   {100.0*(g.rr>0).mean():5.2f}   {g.rr.sum():10.2f}')
    print(f'ALL    {len(trades):6d}   {trades.rr.mean():7.4f}   '
          f'{100.0*(trades.rr>0).mean():5.2f}   {trades.rr.sum():10.2f}')

    compare(trades)


def compare(mine):
    """Compare against the other implementation's trades.csv (3 columns only)."""
    theirs = pd.read_csv(f'{BASE}/trades.csv', keep_default_na=False, na_values=[''])
    theirs = theirs[['day', 'symbol', 'ours_entry', 'ours_exit', 'ours_rr']].copy()
    for c in ('ours_entry', 'ours_exit', 'ours_rr'):
        theirs[c] = pd.to_numeric(theirs[c], errors='coerce')
    theirs = theirs.dropna(subset=['ours_rr'])

    mk = set(zip(mine.day, mine.symbol))
    tk = set(zip(theirs.day.astype(str), theirs.symbol.astype(str)))
    only_mine = sorted(mk - tk)
    only_theirs = sorted(tk - mk)

    j = mine.merge(theirs, on=['day', 'symbol'], how='inner', validate='one_to_one')
    de = (j.entry - j.ours_entry).abs()
    dx = (j.exit_px - j.ours_exit).abs()
    dr = (j.rr - j.ours_rr).abs()
    bad = j[dr > 1e-9]

    lines = []
    lines.append('# Independent check — N3 zero-cost level-break sim\n')
    lines.append(f'Mine: {len(mine)} filled trades, pooled mean rr {mine.rr.mean():.6f}.')
    lines.append(f'Theirs: {len(theirs)} filled trades, pooled mean rr {theirs.ours_rr.mean():.6f}.')
    lines.append(f'Common keys: {len(j)}.\n')
    lines.append('| metric | value |')
    lines.append('|---|---|')
    lines.append(f'| filled only in mine | {len(only_mine)} |')
    lines.append(f'| filled only in theirs | {len(only_theirs)} |')
    lines.append(f'| max abs entry diff | {de.max() if len(j) else float("nan"):.10g} |')
    lines.append(f'| max abs exit diff | {dx.max() if len(j) else float("nan"):.10g} |')
    lines.append(f'| max abs rr diff | {dr.max() if len(j) else float("nan"):.10g} |')
    lines.append(f'| keys with abs rr diff > 1e-9 | {len(bad)} |')
    lines.append(f'| pooled mean rr (mine) | {mine.rr.mean():.6f} |')
    lines.append(f'| pooled mean rr (theirs) | {theirs.ours_rr.mean():.6f} |')
    lines.append('')
    if only_mine:
        lines.append('Examples filled only in mine: ' + ', '.join(f'{d}/{s}' for d, s in only_mine[:5]))
    if only_theirs:
        lines.append('Examples filled only in theirs: ' + ', '.join(f'{d}/{s}' for d, s in only_theirs[:5]))
    if len(bad):
        lines.append('\nrr disagreements (up to 5):\n')
        lines.append('| day | symbol | my entry | their entry | my exit | their exit | my rr | their rr |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for r in bad.head(5).itertuples(index=False):
            lines.append(f'| {r.day} | {r.symbol} | {r.entry:.6f} | {r.ours_entry:.6f} | '
                         f'{r.exit_px:.6f} | {r.ours_exit:.6f} | {r.rr:.6f} | {r.ours_rr:.6f} |')
    text = '\n'.join(lines) + '\n'
    with open(CHECK, 'w') as f:
        f.write(text)
    print('\n' + text)


if __name__ == '__main__':
    main()
