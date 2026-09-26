"""Independent rebuild of cell 1,481 (PREREG_1481.md) from prose alone -- never read cell_1481.py,
test_cell_1481.py or RESULT_1481.md while writing this.

Rule (PREREG_1481.md): at the instant of the base fill (cell 1,438's 9,911 `status=='fill'` rows),
rest a BUY LIMIT at level-$0.01 for 15 RTH minutes after the fill bar (window starts at the exact
base-fill timestamp, not the arm). Obtainability: filled AT THE LIMIT PRICE at the first tape print
STRICTLY BELOW the limit inside the window (traded-through convention); a print exactly at the limit
does not fill; no such print -> no trade. Stop = the base stop (consolidation low). R' = entry-stop.
Target = entry + 2R'. Path: the retest minute is resolved off the TAPE (a print <= stop after entry
= stopped; a print >= target = target); every later minute is resolved off `bars_fills_1478.db` with
sip_rebuild.walk_path semantics. Costs: entry free (passive limit, no spread); target exit free (a
limit, no slip); stop exit and EOD/eod_fallback exit both take the SAME kind of expected-value slip
cell 1,478 already uses for stop exits (0.88*filled-stop-bps + 0.12*no-fill-tail-bps, per split) --
EOD's own bps come from cell 1,443's measured EOD-holdout means (RESULT_1443.md: TRAIN-H2 11.5bps,
VAL 9.7bps), converted to R via exit_price*bps/1e4/R' (PREREG_1478's own phrasing).

Usage:
    python3 rebuild_1481.py [--days D1,D2,...] [--limit N] [--resume]
Writes/updates research/hod_entry/rebuild_1481_fills.csv incrementally (one day at a time, flushed),
so a killed run resumes from the last fully-written day with --resume.
"""
import argparse
import os
import pickle
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import causal_arming as ca          # noqa: E402  (shared: fetch_window -- infra, not cell 1,481's logic)
import sip_rebuild as sr            # noqa: E402  (shared: et_ns, walk_path, EOD_M, TICK -- infra)

CAUSAL_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
BARS_DB = os.path.join(HERE, 'bars_fills_1478.db')
CACHE_DIRS = [os.path.join(HERE, 'sip_cache_1481'), os.path.join(HERE, 'sip_cache_1480')]
OUT_CSV = os.path.join(HERE, 'rebuild_1481_fills.csv')

WINDOW_MIN = 15                     # PREREG: 15-minute resting window
LIMIT_OFFSET = 0.01                 # PREREG: level - $0.01
TARGET_R = 2.0                      # PREREG: target = entry + 2 R'
EOD_M = sr.EOD_M                    # 955 = 15:55 ET
EPS = 1e-9

# PREREG_1478 amendment's expected-value stop-limit slip, per split (bps).
SLIP_STOP_BPS = {'TRAIN': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}
# RESULT_1443.md measured EOD-holdout mean slip (bps), the number PREREG_1481 points to
# ("EOD exit at the bid (cell 1443's EOD measure)").
SLIP_EOD_BPS = {'TRAIN': 11.5, 'VAL': 9.7}

OUT_COLS = ['day', 'symbol', 'split', 'wk', 'fill', 'stop', 'level', 'fill_min', 'base_why',
            'base_net_R', 'limit', 'status', 'filled', 'retest_minute', 'retest_ts', 'dip_low',
            'entry', 'stop_used', 'target', 'Rp', 'r_pct_price', 'retest_delay_min', 'exit_m',
            'exit_price', 'why', 'raw_R', 'cost_R', 'net_R_prime']


def log(msg):
    """Verbose, immediately-flushed progress line (nohup buffers print() otherwise)."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_base_fills(limit=None, days=None):
    """cell 1,438's causal_arming_causal.csv, status=='fill' rows only (9,911).
    split 'TRAIN' in this file is already TRAIN-H2-only (verified: half.unique()==['H2'])."""
    df = pd.read_csv(CAUSAL_CSV, low_memory=False)
    f = df[df.status == 'fill'].copy()
    f = f[['day', 'symbol', 'split', 'wk', 'fill', 'stop', 'level', 'fill_min', 'why', 'net_R']]
    f = f.rename(columns={'why': 'base_why', 'net_R': 'base_net_R'})
    if days:
        f = f[f.day.isin(days)]
    f = f.sort_values(['day', 'symbol']).reset_index(drop=True)
    if limit:
        f = f.head(limit)
    log(f'load_base_fills: {len(f)} rows, {f.day.nunique()} days, {f.symbol.nunique()} symbols')
    return f


def bars_for_day(con, day, symbols):
    """{symbol: DataFrame(m,o,h,l,c) sorted, RTH minute bars} for one day, from bars_fills_1478.db."""
    syms = sorted(set(symbols))
    q = (f"select symbol, t, o, h, l, c from bars where day=? and symbol in "
         f"({','.join('?' * len(syms))})")
    df = pd.read_sql(q, con, params=[day] + syms)
    if not len(df):
        return {}
    et = pd.to_datetime(df['t'], utc=True).dt.tz_convert(sr.ET)
    df['m'] = et.dt.hour * 60 + et.dt.minute
    df = df.sort_values(['symbol', 'm'], kind='stable').drop_duplicates(['symbol', 'm'], keep='first')
    return {s: g[['m', 'o', 'h', 'l', 'c']].reset_index(drop=True) for s, g in df.groupby('symbol')}


def _cache_path(cache_dir, symbol, day, m):
    return os.path.join(cache_dir, f'{symbol}_{day}_{m}.pkl')


def fetch_minute(symbol, day, m, n_fetched):
    """Trades+quotes for one minute (fetch_window(symbol,day,m,m+1) semantics), reusing sip_cache_1481
    or sip_cache_1480 if already cached there; else a fresh fetch, cached under sip_cache_1481 (resumable).
    Returns (trades_df, quotes_df) or (None, None) on a hard fetch error (logged)."""
    for d in CACHE_DIRS:
        p = _cache_path(d, symbol, day, m)
        if os.path.exists(p):
            with open(p, 'rb') as fh:
                return pickle.load(fh)
    try:
        t, q = ca.fetch_window(symbol, day, m, m + 1)
    except Exception as e:                                              # noqa: BLE001 -- network
        log(f'  [fetch_error] {symbol} {day} m={m}: {e}')
        return None, None
    p = _cache_path(CACHE_DIRS[0], symbol, day, m)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as fh:
        pickle.dump((t, q), fh)
    n_fetched[0] += 1
    return t, q


def find_retest(symbol, day, fill_min, limit, m_lo, m_hi, bars, n_fetched):
    """Scan bars[m_lo..m_hi] in order for the first minute whose bar low is below `limit`, fetch its
    tape, and look for the first print STRICTLY BELOW `limit` (after the base-fill timestamp, for the
    fill bar itself). Returns dict(status, retest_minute, retest_ts, dip_low) -- status in
    {'fill','no_retest','bar_tick_disagree','no_bars','no_tape'}; 'bar_tick_disagree' is only final
    if EVERY candidate minute in the window disagreed with the tape (a later minute can still fill)."""
    fill_ts = sr.et_ns(day, fill_min * 60)
    disagreed = False
    dip_low = np.inf
    if symbol not in bars:
        return dict(status='no_bars', retest_minute=None, retest_ts=None, dip_low=None)
    b = bars[symbol]
    b = b[(b.m >= m_lo) & (b.m <= m_hi)]
    for row in b.itertuples():
        if row.l < limit - EPS:
            t, q = fetch_minute(symbol, day, int(row.m), n_fetched)
            if t is None:
                disagreed = True
                continue
            if len(t):
                dip_low = min(dip_low, float(t.price.min()))
            # the fill bar itself (m == m_lo) only counts trades AFTER the base-fill instant --
            # the retest window starts at the fill, not at the start of its bar.
            tt = t[t.ts > fill_ts] if int(row.m) == m_lo else t
            w = tt[tt.price < limit - EPS].sort_values('ts', kind='stable')
            if len(w):
                return dict(status='fill', retest_minute=int(row.m), retest_ts=int(w.ts.iloc[0]),
                            dip_low=(dip_low if np.isfinite(dip_low) else None))
            disagreed = True
    if not len(b):
        return dict(status='no_bars', retest_minute=None, retest_ts=None, dip_low=None)
    if disagreed:
        return dict(status='bar_tick_disagree', retest_minute=None, retest_ts=None,
                     dip_low=(dip_low if np.isfinite(dip_low) else None))
    return dict(status='no_retest', retest_minute=None, retest_ts=None, dip_low=None)


def resolve_same_minute(symbol, day, m, entry_ts, stop, target, n_fetched):
    """Inside the retest minute, the tape decides: after `entry_ts`, the first print <= stop or
    >= target (whichever comes first in time) closes the trade there. Returns (exit_m, exit_price,
    why) or None if neither is hit in that minute's remaining tape."""
    t, _ = fetch_minute(symbol, day, m, n_fetched)
    if t is None or not len(t):
        return None
    after = t[t.ts > entry_ts].sort_values('ts', kind='stable')
    if not len(after):
        return None
    hit_stop = after[after.price <= stop + EPS]
    hit_target = after[after.price >= target - EPS]
    ts_stop = int(hit_stop.ts.iloc[0]) if len(hit_stop) else None
    ts_target = int(hit_target.ts.iloc[0]) if len(hit_target) else None
    if ts_stop is None and ts_target is None:
        return None
    if ts_target is None or (ts_stop is not None and ts_stop <= ts_target):
        return m, float(stop), 'stop'
    return m, float(target), 'target'


def simulate_fill(row, bars, n_fetched):
    """One base fill -> one rebuild_1481 row (dict), per PREREG_1481.md prose."""
    day, symbol, split = row.day, row.symbol, row.split
    limit = round(row.level - LIMIT_OFFSET, 8)
    m_fill = int(row.fill_min // 1)
    m_hi = min(m_fill + WINDOW_MIN, EOD_M)

    out = dict(day=day, symbol=symbol, split=split, wk=row.wk, fill=row.fill, stop=row.stop,
               level=row.level, fill_min=row.fill_min, base_why=row.base_why,
               base_net_R=row.base_net_R, limit=limit, filled=False, retest_minute=None,
               retest_ts=None, dip_low=None, entry=None, stop_used=None, target=None, Rp=None,
               r_pct_price=None, retest_delay_min=None, exit_m=None, exit_price=None, why=None,
               raw_R=None, cost_R=None, net_R_prime=None)

    ret = find_retest(symbol, day, row.fill_min, limit, m_fill, m_hi, bars, n_fetched)
    out['status'] = ret['status']
    out['retest_minute'] = ret['retest_minute']
    out['retest_ts'] = ret['retest_ts']
    out['dip_low'] = ret['dip_low']
    if ret['status'] != 'fill':
        return out

    entry = limit
    stop = row.stop
    Rp = entry - stop
    if not np.isfinite(Rp) or Rp <= 0:
        out['status'] = 'bad_R'
        return out
    target = entry + TARGET_R * Rp
    m_retest = ret['retest_minute']
    entry_ts = ret['retest_ts']

    out.update(filled=True, entry=entry, stop_used=stop, target=target, Rp=Rp,
               r_pct_price=Rp / entry * 100.0,
               retest_delay_min=sr.ns_to_et_minutes(entry_ts, day) - row.fill_min)

    same_min = resolve_same_minute(symbol, day, m_retest, entry_ts, stop, target, n_fetched)
    if same_min is not None:
        exit_m, exit_price, why = same_min
    else:
        b = bars.get(symbol)
        path = b[(b.m > m_retest) & (b.m <= EOD_M)] if b is not None else None
        if path is None or not len(path):
            log(f'  [WARNING] {symbol} {day}: no bars after retest minute {m_retest} -- eod_fallback at entry')
            exit_m, exit_price, why = m_retest, entry, 'eod_fallback'
        else:
            exit_m, exit_price, why = sr.walk_path(entry, stop, target, path)

    raw_R = (exit_price - entry) / Rp
    if why in ('stop', 'stop_bar'):
        bps = SLIP_STOP_BPS[split]
        slip_R = exit_price * bps / 1e4 / Rp
    elif why in ('eod', 'eod_fallback'):
        bps = SLIP_EOD_BPS[split]
        slip_R = exit_price * bps / 1e4 / Rp
    else:                                                                # 'target': a limit, no slip
        slip_R = 0.0
    out.update(exit_m=exit_m, exit_price=exit_price, why=why, raw_R=raw_R, cost_R=slip_R,
               net_R_prime=raw_R - slip_R)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', default=None, help='comma-separated YYYY-MM-DD subset (debug)')
    ap.add_argument('--limit', type=int, default=None, help='cap on base fills (debug)')
    ap.add_argument('--resume', action='store_true', help='skip days already fully written to OUT_CSV')
    args = ap.parse_args(argv)

    days = args.days.split(',') if args.days else None
    fills = load_base_fills(limit=args.limit, days=days)

    done_days = set()
    if args.resume and os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV, low_memory=False)
        done_days = set(prev.day.unique())
        log(f'--resume: {len(done_days)} days already written, {len(prev)} rows kept')
    else:
        prev = pd.DataFrame(columns=OUT_COLS)
        if os.path.exists(OUT_CSV):
            os.remove(OUT_CSV)

    con = sqlite3.connect(f'file:{BARS_DB}?mode=ro', uri=True, timeout=120)
    n_fetched = [0]
    t0 = time.time()
    all_days = [d for d in fills.day.unique() if d not in done_days]
    wrote_header = not args.resume or not os.path.exists(OUT_CSV) or not len(prev)
    for i, day in enumerate(all_days):
        day_fills = fills[fills.day == day]
        bars = bars_for_day(con, day, day_fills.symbol.unique())
        rows = [simulate_fill(r, bars, n_fetched) for r in day_fills.itertuples()]
        out_df = pd.DataFrame(rows)[OUT_COLS]
        out_df.to_csv(OUT_CSV, mode='a', header=wrote_header, index=False)
        wrote_header = False
        n_fill = int((out_df.status == 'fill').sum())
        elapsed = time.time() - t0
        log(f'[{i + 1}/{len(all_days)}] day={day}: {len(day_fills)} base fills, {n_fill} retest-fills, '
            f'{n_fetched[0]} fresh minute-fetches so far, {elapsed:.0f}s elapsed')
    con.close()
    log(f'DONE: {len(all_days)} days processed, {n_fetched[0]} fresh tape fetches total, '
        f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
