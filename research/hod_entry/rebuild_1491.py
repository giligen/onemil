#!/usr/bin/env python3
"""Independent rebuild of cells 1,491-1,492 (research/hod_entry/PREREG_1491.md, FROZEN 2026-09-26)
-- the SHALLOW STOP re-walk: stop = level * (1 - s) for s in {0.25%, 0.50%, 0.75%}, two targets each
(SCALP: fill + 2*R_s; ASYMMETRIC: fill + 2*base_R, base_R = the original consolidation-low R).

Written from the PREREG prose ONLY -- this agent has NOT opened cell_1491.py, test_cell_1491.py or
RESULT_1491.md. Ingredients:
  * Base book: causal_arming_causal.csv, status == 'fill' (9,911 rows).
  * Half-spread at the fill: features_1478_A.csv column half_entry, joined on
    (day, symbol, fill_min) -- exact float merge, verified 9,911/9,911 with zero NaN before this
    script was written (the task's pointer to 'cell_1445_features.csv (half_entry)' does not match
    any file on disk with that column; features_1478_A.csv is the file that actually carries it,
    joined on the same key the task specifies).
  * Minute bars: bars_fills_1478.db (sqlite table `bars`: symbol, day, t [UTC ISO], o,h,l,c,v).
    Converted to ET minutes-since-midnight (America/New_York) to match fill_min/exit_m's units
    (verified: fill_min in the base book is a fractional ET minute, e.g. 576.0001 = 9:36am + secs).
  * Walk semantics (sip_rebuild.walk_path, read for its physics only -- a DIFFERENT tape-based
    engine, reused here only for its bar-order-of-operations): iterate bars with m >= floor(fill_min)
    (the fill bar itself is IN the walk -- the PREREG's own "the fill bar's low <= stop_s => stopped
    (conservative)" sentence), in increasing m; on each bar, check EOD (m >= 955 i.e. 15:55) first,
    then stop (bar low <= stop_s), then target (bar high >= target); gap-through at the open (exit
    at the open if the open is already through the stop, else at the stop price itself); EOD exit
    is the 15:55 bar's OPEN; if bars run out before 955, exit at the last bar's close ('eod_fallback',
    logged as a WARNING per bug-protocol / fallback-logging rules).
  * why-category inference: cell_1491_fills.csv (the reference OUTPUT, not the code) has both 'stop'
    and 'stop_bar' as distinct why values (18,610 'stop_bar' vs 27,686 'stop' rows). cell_1478.py's
    STOP_WHY = {'stop','stop_bar'} charges BOTH identically for slip purposes. The only bar-walk
    distinction available from the PREREG prose is "the FILL BAR's low <= stop_s" (singled out as the
    conservative case) vs a stop on a LATER bar -- so this rebuild labels a stop hit on the fill bar
    itself 'stop_bar', and a stop hit on any subsequent bar 'stop' (both slipped identically).
  * Costs: entry cost = half_entry / R_s, charged on every exit type (entry mechanics unchanged from
    the base fill). Exit-side cost by why:
      - target: 0 (a resting limit fill -- PREREG "target = limit").
      - stop / stop_bar: PREREG's own expected-value slip -- 0.88 * (2.9/3.2 bps, TRAIN/VAL filled-
        stop holdout mean) + 0.12 * (94/76 bps, the no-fill tail mean), in the BOOK'S OWN R (R_s):
        slip_R = exit_price * bps / 1e4 / R_s. (Identical constants to cell_1478's SLIP_STOP_BPS --
        expected, since the PREREG cites this exact frozen amendment standard by name.)
      - eod / eod_fallback: 1,443's measured EOD holdout means (TRAIN-H2 11.5bps, VAL 9.7bps, from
        RESULT_1443.md's slip table), same R_s-denominated conversion.
    net_R_s = raw_R_s - entry_cost_R - exit_slip_R; net_pct = net_R_s * R_s / fill * 100.
  * R-must-exceed-spread guard: rows where R_s <= 0 (stop_s >= fill, mostly at the tightest s on
    fills where level sits very close to or above fill) are DROPPED per book, counted and logged --
    never silently.

Usage: nice -n 19 python3 research/hod_entry/rebuild_1491.py
Output: research/hod_entry/rebuild_1491_fills.csv, research/hod_entry/rebuild_1491_report.md
"""
import os
import sys
import sqlite3
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

os.environ.setdefault('OMP_NUM_THREADS', '2')

HERE = os.path.dirname(os.path.abspath(__file__))
CAUSAL_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
FEATURES_A = os.path.join(HERE, 'features_1478_A.csv')
BARS_DB = os.path.join(HERE, 'bars_fills_1478.db')
OUT_CSV = os.path.join(HERE, 'rebuild_1491_fills.csv')
OUT_MD = os.path.join(HERE, 'rebuild_1491_report.md')
REF_CSV = os.path.join(HERE, 'cell_1491_fills.csv')          # comparison target, read AFTER building

ET = ZoneInfo('America/New_York')
EOD_M = 955                     # 15:55 ET, sip_rebuild.py convention
STOP_TIERS = [0.0025, 0.0050, 0.0075]
STOP_LABELS = ['025', '050', '075']
SLIP_STOP_BPS = {'TRAIN': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}
SLIP_EOD_BPS = {'TRAIN': 11.5, 'VAL': 9.7}                    # RESULT_1443.md eod row, mean column
R_PCT_SHIP_FLOOR = 0.5                                        # R-must-exceed-spread rail, % of price


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def et_minute(iso_ts):
    """UTC ISO string -> ET minutes-since-midnight (float, but bar timestamps are :00-aligned)."""
    dt = datetime.fromisoformat(iso_ts).astimezone(ET)
    return dt.hour * 60 + dt.minute + dt.second / 60.0


def load_base():
    causal = pd.read_csv(CAUSAL_CSV, low_memory=False)
    fills = causal[causal.status == 'fill'].reset_index(drop=True)
    keep = ['day', 'symbol', 'split', 'wk', 'fill', 'stop', 'level', 'fill_min']
    fills = fills[keep].rename(columns={'stop': 'base_stop'})
    fills['base_R'] = fills['fill'] - fills['base_stop']
    log(f'load_base: {len(fills)} fills, splits {fills.split.value_counts().to_dict()}')

    feat = pd.read_csv(FEATURES_A, usecols=['day', 'symbol', 'fill_min', 'half_entry'])
    merged = fills.merge(feat, on=['day', 'symbol', 'fill_min'], how='left')
    n_na = int(merged.half_entry.isna().sum())
    if n_na:
        log(f'load_base: WARNING {n_na} fills had no half_entry match on '
            f'(day,symbol,fill_min) -- these rows dropped from every book (cost unknown)')
        merged = merged.dropna(subset=['half_entry']).reset_index(drop=True)
    return merged


def load_bars(symdays):
    """Load only the bars needed (symbol,day pairs in symdays), grouped, ET-minute-tagged, sorted."""
    con = sqlite3.connect(BARS_DB)
    con.execute('CREATE TEMP TABLE need (symbol TEXT, day TEXT)')
    con.executemany('INSERT INTO need VALUES (?,?)', list(symdays))
    q = ('SELECT b.symbol, b.day, b.t, b.o, b.h, b.l, b.c FROM bars b '
         'JOIN need n ON b.symbol = n.symbol AND b.day = n.day')
    df = pd.read_sql_query(q, con)
    con.close()
    log(f'load_bars: {len(df)} bar rows for {len(symdays)} (symbol,day) pairs')
    df['m'] = df['t'].map(et_minute)
    df = df.sort_values(['symbol', 'day', 'm'], kind='stable')
    groups = {k: g[['m', 'o', 'h', 'l', 'c']].to_numpy(float)
              for k, g in df.groupby(['symbol', 'day'], sort=False)}
    return groups


def walk(bars_arr, fill_bar_m, fill, stop_s, target):
    """Bar-OHLC walk starting at the first row with m >= fill_bar_m. Returns (exit_m, exit_price,
    why). why in {stop_bar, stop, target, eod, eod_fallback, no_bar}."""
    if bars_arr is None or len(bars_arr) == 0:
        return np.nan, np.nan, 'no_bar'
    sub = bars_arr[bars_arr[:, 0] >= fill_bar_m]
    if len(sub) == 0:
        return np.nan, np.nan, 'no_bar'
    for i in range(len(sub)):
        m, o, h, l, c = sub[i]
        if m >= EOD_M:
            return float(m), float(o), 'eod'
        if l <= stop_s:
            px = o if o <= stop_s else stop_s
            why = 'stop_bar' if i == 0 else 'stop'
            return float(m), float(px), why
        if h >= target:
            return float(m), float(target), 'target'
    last = sub[-1]
    return float(last[0]), float(last[4]), 'eod_fallback'


def build_book(base, bars, s, label, mode):
    """mode in {'SCALP','ASYM'}. Returns a per-fill DataFrame for this one book, R_s<=0 rows dropped."""
    stop_s = base['level'].to_numpy(float) * (1.0 - s)
    R_s = base['fill'].to_numpy(float) - stop_s
    valid = R_s > 0
    n_drop = int((~valid).sum())
    if n_drop:
        log(f'  book {mode}_{label}: dropping {n_drop} rows with R_s<=0 (stop_s>=fill, guard)')

    fill = base['fill'].to_numpy(float)
    base_R = base['base_R'].to_numpy(float)
    half_entry = base['half_entry'].to_numpy(float)
    fill_min = base['fill_min'].to_numpy(float)
    split = base['split'].to_numpy()

    target = fill + 2.0 * (R_s if mode == 'SCALP' else base_R)

    rows = []
    for i in range(len(base)):
        if not valid[i]:
            continue
        key = (base['symbol'].iat[i], base['day'].iat[i])
        exit_m, exit_price, why = walk(bars.get(key), np.floor(fill_min[i]), fill[i], stop_s[i],
                                        target[i])
        if why == 'no_bar':
            log(f'  ERROR: no bars for {key} at/after fill_min={fill_min[i]:.2f} -- row dropped')
            continue
        R_si = R_s[i]
        raw_R = (exit_price - fill[i]) / R_si
        entry_cost_R = half_entry[i] / R_si
        if why in ('stop', 'stop_bar'):
            slip_bps = SLIP_STOP_BPS[split[i]]
            slip_R = exit_price * slip_bps / 1e4 / R_si
        elif why in ('eod', 'eod_fallback'):
            slip_bps = SLIP_EOD_BPS[split[i]]
            slip_R = exit_price * slip_bps / 1e4 / R_si
        else:
            slip_R = 0.0
        net_R = raw_R - entry_cost_R - slip_R
        net_pct = net_R * R_si / fill[i] * 100.0
        rows.append(dict(book=f'{mode}_{label}', day=base['day'].iat[i], symbol=base['symbol'].iat[i],
                          split=split[i], wk=base['wk'].iat[i], fill=fill[i], level=base['level'].iat[i],
                          stop_s=stop_s[i], R_s=R_si, base_R=base_R[i], fill_min=fill_min[i],
                          exit_m=exit_m, exit_price=exit_price, why=why, raw_R=raw_R,
                          net_R_own=net_R, net_pct=net_pct))
    return pd.DataFrame(rows), n_drop


def day_clustered_t(y, day):
    d = pd.DataFrame({'y': y, 'day': day}).dropna()
    if d.day.nunique() < 2:
        return np.nan
    g = d.groupby('day').y.mean()
    n = len(g)
    se = g.std(ddof=1) / np.sqrt(n)
    return float(g.mean() / se) if se > 0 else np.nan


def main():
    log('rebuild_1491: loading base book + half_entry')
    base = load_base()
    symdays = set(zip(base.symbol, base.day))
    log('rebuild_1491: loading bars')
    bars = load_bars(symdays)

    all_books = []
    drop_counts = {}
    for s, label in zip(STOP_TIERS, STOP_LABELS):
        for mode in ('SCALP', 'ASYM'):
            log(f'rebuild_1491: walking {mode}_{label} (s={s:.4%})')
            book_df, n_drop = build_book(base, bars, s, label, mode)
            all_books.append(book_df)
            drop_counts[f'{mode}_{label}'] = n_drop

    out = pd.concat(all_books, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    log(f'rebuild_1491: wrote {len(out)} rows -> {OUT_CSV}')

    # ---- summary report (own book, VAL, pass-bar-adjacent stats; report-only here) ----
    lines = ['# rebuild_1491 -- independent rebuild summary (report-only; scoring is the caller\'s job)',
              '', f'Base fills: {len(base)}; guard drops per book: {drop_counts}', '',
              '| book | split | n | mean net% | mean net_R_own | t (day-clustered) | why mix |',
              '|---|---|---|---|---|---|---|']
    for bk in out.book.unique():
        for sp in ('TRAIN', 'VAL'):
            sub = out[(out.book == bk) & (out.split == sp)]
            if not len(sub):
                continue
            whymix = sub.why.value_counts(normalize=True).round(3).to_dict()
            lines.append(f'| {bk} | {sp} | {len(sub)} | {sub.net_pct.mean():.4f} | '
                          f'{sub.net_R_own.mean():.4f} | {day_clustered_t(sub.net_R_own, sub.day):.2f} '
                          f'| {whymix} |')
    with open(OUT_MD, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log(f'rebuild_1491: wrote summary -> {OUT_MD}')

    # ---- comparison against cell_1491_fills.csv (read now, AFTER the independent build) ----
    if not os.path.exists(REF_CSV):
        log(f'ERROR: reference {REF_CSV} not found -- cannot compare')
        return
    ref = pd.read_csv(REF_CSV, low_memory=False)
    log(f'compare: reference has {len(ref)} rows, mine has {len(out)}')

    key = ['day', 'symbol', 'fill_min', 'book']
    m = out.merge(ref[key + ['net_R_own', 'net_pct', 'why', 'exit_price', 'exit_m']],
                  on=key, how='outer', suffixes=('_mine', '_ref'), indicator=True)
    log('compare: merge indicator counts:\n' + str(m._merge.value_counts()))
    both = m[m._merge == 'both'].copy()
    both['abs_diff_R'] = (both.net_R_own_mine - both.net_R_own_ref).abs()
    within = (both.abs_diff_R <= 0.01).mean()
    log(f'compare: {len(both)} rows matched on key; share within 0.01 R = {within:.4%}')

    val_mine = out[out.split == 'VAL'].groupby('book').net_pct.mean()
    val_ref = ref[ref.holdout.str.contains('VAL', na=False)].groupby('book').net_pct.mean() \
        if 'holdout' in ref.columns else ref.groupby('book').net_pct.mean()
    best_mine_book = val_mine.idxmax()
    best_ref_book = val_ref.idxmax()
    log(f'compare: VAL best book mine={best_mine_book} ({val_mine[best_mine_book]:.4f}%), '
        f'ref={best_ref_book} ({val_ref[best_ref_book]:.4f}%)')
    log('compare: VAL mean net_pct by book, mine:\n' + str(val_mine))
    log('compare: VAL mean net_pct by book, ref:\n' + str(val_ref))

    n_only_mine = int((m._merge == 'left_only').sum())
    n_only_ref = int((m._merge == 'right_only').sum())
    log(f'compare: rows only in mine={n_only_mine}, rows only in ref={n_only_ref}')

    bad = both[both.abs_diff_R > 0.01].sort_values('abs_diff_R', ascending=False)
    if len(bad):
        log('compare: top 10 discrepancies:\n' +
            bad[['day', 'symbol', 'fill_min', 'book', 'why_mine', 'why_ref', 'exit_price_mine',
                 'exit_price_ref', 'net_R_own_mine', 'net_R_own_ref', 'abs_diff_R']]
            .head(10).to_string())
        log('compare: discrepancy why-pair cross-tab:\n' +
            str(bad.groupby(['why_mine', 'why_ref']).size().sort_values(ascending=False).head(15)))

    log('rebuild_1491: DONE')


if __name__ == '__main__':
    main()
