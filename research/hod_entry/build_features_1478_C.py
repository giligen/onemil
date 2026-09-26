"""Cell 1,478 — Feature Set C (the tape), PREREG item 4.

For every 'fill' row of `causal_arming_causal.csv` (9,911 rows), build tape-derived features from
the 1,438 tick-window cache (`sip_cache/c1438_{day}.pkl.gz`, keyed 'symbol|m_lo|m_hi'):

  trigger_print_odd_lot, trigger_print_size  — the print that crossed the trigger inside bar j+1.
  pre_break_odd_lot_share, pre_break_mean_trade_size, pre_break_print_count, pre_break_buy_share,
  spread_bps_at_arm                          — using ONLY prints/quotes strictly before the start
                                                of bar j+1 (arm bar j's own window; per fetch_tape's
                                                span_s math this is a ~5s sliver before the window_ns
                                                boundary, so NaN is expected on most rows — reported).

ALL features use only data at or before the close of arm bar j / the trigger print inside bar j+1 that
produced the recorded fill (the fill itself, and its trigger print, are already realised facts of the
base book — no field here looks past the fill_ts that generated that row). No refit, no label use.

Resumable: writes to OUT_CSV in append mode, skipping (day, symbol, fill_min) keys already written, so
a killed run can be restarted with the same command.
"""
import glob
import gzip
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import causal_arming as ca          # noqa: E402
import sip_rebuild as sr            # noqa: E402

FILLS_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
CACHE_DIR = os.path.join(HERE, 'sip_cache')
OUT_CSV = os.path.join(HERE, 'features_1478_C.csv')
LOG_PATH = os.path.join(HERE, 'build_features_1478_C.log')
COLUMNS = ['day', 'symbol', 'fill_min', 'window_found', 'has_prebreak',
           'trigger_print_odd_lot', 'trigger_print_size',
           'pre_break_odd_lot_share', 'pre_break_mean_trade_size', 'pre_break_print_count',
           'pre_break_buy_share', 'spread_bps_at_arm']

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                     handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()])
log = logging.getLogger(__name__).info


def load_fills():
    """The 9,911 status=='fill' rows, needed columns only."""
    df = pd.read_csv(FILLS_CSV, dtype={'symbol': str, 'day': str}, low_memory=False)
    f = df[df.status == 'fill'][['day', 'symbol', 'fill_min', 'level']].reset_index(drop=True)
    return f


def already_done():
    """(day, symbol, fill_min) keys already in OUT_CSV, for resume."""
    if not os.path.exists(OUT_CSV):
        return set()
    d = pd.read_csv(OUT_CSV, dtype={'symbol': str, 'day': str}, usecols=['day', 'symbol', 'fill_min'])
    return set(zip(d.day, d.symbol, d.fill_min))


def arm_bar(bars, fill_min):
    """(m_lo, m_hi) = (arm bar j, fill/breakout bar j+1). The fill bar is the RTH bar whose [m, m+1)
    contains the fill (largest m with m <= fill_min); the arm bar is the immediately preceding row in
    the bars array (adjacent-index j, j+1, exactly as armed_crossing_bars builds its candidates — gaps
    in m are carried through unchanged). Verified against sip_cache keys: AAP 2025-07-01 fill_min
    605.311 -> fill bar m=605, arm bar m=604 -> key 'AAP|604|605', present in c1438_2025-07-01.pkl.gz.
    None if the fill bar is the first row (no preceding bar) or fill_min precedes all bars."""
    m = bars.m.values
    fill_idx = np.searchsorted(m, fill_min, side='right') - 1
    if fill_idx <= 0:
        return None, None
    return int(m[fill_idx - 1]), int(m[fill_idx])


def classify_prints(pre, quotes):
    """Lee-Ready-lite on prints strictly before the window's start: at/above the prevailing ask = buy,
    at/below the prevailing bid = sell; buy_share = n_ask / (n_ask + n_bid), NaN if neither classifies."""
    if not len(pre):
        return np.nan
    qv = sr._valid(quotes).sort_values('ts')
    if not len(qv):
        return np.nan
    p = pre.sort_values('ts')
    m = pd.merge_asof(p, qv, on='ts', direction='backward')
    n_ask = int((m.price >= m.ask - 1e-9).sum())
    n_bid = int((m.price <= m.bid + 1e-9).sum())
    denom = n_ask + n_bid
    return (n_ask / denom) if denom else np.nan


def features_for_fill(day, symbol, fill_min, level, bars, cache):
    """One row of Feature Set C, or an all-NaN row with window_found/has_prebreak flags set."""
    out = dict(day=day, symbol=symbol, fill_min=fill_min, window_found=0, has_prebreak=0,
               trigger_print_odd_lot=np.nan, trigger_print_size=np.nan,
               pre_break_odd_lot_share=np.nan, pre_break_mean_trade_size=np.nan,
               pre_break_print_count=np.nan, pre_break_buy_share=np.nan, spread_bps_at_arm=np.nan)
    if bars is None or len(bars) < 2:
        return out
    m_lo, m_hi = arm_bar(bars, fill_min)
    if m_lo is None:
        return out
    key = f'{symbol}|{m_lo}|{m_hi}'
    tq = cache.get(key)
    if tq is None:
        return out
    t, q = tq
    out['window_found'] = 1
    start, end = ca.window_ns(day, m_lo, m_hi)
    w = t[(t.ts >= start) & (t.ts < end)].sort_values('ts', kind='stable')
    hits = w[w.price >= level - 1e-9]
    if len(hits):
        sz = float(hits['size'].iloc[0])
        out['trigger_print_size'] = sz
        out['trigger_print_odd_lot'] = int(sz < 100)
    pre = t[t.ts < start]
    if len(pre):
        out['has_prebreak'] = 1
        out['pre_break_print_count'] = int(len(pre))
        out['pre_break_mean_trade_size'] = float(pre['size'].mean())
        out['pre_break_odd_lot_share'] = float((pre['size'] < 100).mean())
        out['pre_break_buy_share'] = classify_prints(pre, q)
    first_print_ts = w['ts'].min() if len(w) else np.nan
    if np.isfinite(first_print_ts):
        pq = sr.prevailing_quote(q, int(first_print_ts))
        if pq is not None:
            bid, ask = pq
            mid = 0.5 * (bid + ask)
            if mid > 0:
                out['spread_bps_at_arm'] = (ask - bid) / mid * 1e4
    return out


def run():
    fills = load_fills()
    done = already_done()
    log(f'[start] {len(fills)} fills total, {len(done)} already written, resuming')
    con = sr_con = None
    con = __import__('sqlite3').connect(sr.CACHE_DB_URI, uri=True, timeout=120)
    sipcon = __import__('sqlite3').connect(ca.BARS_SIP_URI, uri=True, timeout=120)
    days = sorted(fills.day.unique())
    t0 = time.time()
    n_written = n_skipped_done = n_no_bars = n_no_cache = 0
    write_header = not os.path.exists(OUT_CSV)
    fh = open(OUT_CSV, 'a')
    if write_header:
        fh.write(','.join(COLUMNS) + '\n')
    for di, day in enumerate(days):
        sub = fills[fills.day == day]
        pending = [r for r in sub.itertuples() if (r.day, r.symbol, r.fill_min) not in done]
        if not pending:
            n_skipped_done += len(sub)
            continue
        syms = sorted(set(r.symbol for r in pending))
        counts = {}
        bars_map = ca.load_day_bars(con, day, syms, sipcon, counts)
        cache_path = os.path.join(CACHE_DIR, f'c1438_{day}.pkl.gz')
        if not os.path.exists(cache_path):
            log(f'[day] WARNING {day}: no c1438 cache file — {len(pending)} fills get window_found=0')
            cache = {}
        else:
            with gzip.open(cache_path, 'rb') as f:
                cache = pickle.load(f)
        rows = []
        for r in pending:
            b = bars_map.get(r.symbol)
            if b is None or len(b) < 2:
                n_no_bars += 1
            row = features_for_fill(day, r.symbol, r.fill_min, r.level, b, cache)
            if not row['window_found']:
                n_no_cache += 1
            rows.append(row)
        out_df = pd.DataFrame(rows)[COLUMNS]
        out_df.to_csv(fh, header=False, index=False, mode='a')
        fh.flush()
        n_written += len(rows)
        if (di + 1) % 20 == 0 or di == len(days) - 1:
            el = time.time() - t0
            log(f'[progress] day {di + 1}/{len(days)} ({day}) | written {n_written} | '
                f'no_cache_key {n_no_cache} | no_bars {n_no_bars} | {el:.0f}s elapsed')
    fh.close()
    log(f'[done] written {n_written} new rows, {n_skipped_done} already-done rows skipped, '
        f'no_cache_key {n_no_cache}, no_bars {n_no_bars}, total time {time.time() - t0:.0f}s')


if __name__ == '__main__':
    run()
