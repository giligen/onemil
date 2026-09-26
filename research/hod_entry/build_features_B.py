#!/usr/bin/env python3
"""Feature Set B ("the crowd") for PREREG_1478 cell 1,478 -- market-breadth, sector-peer and SPY
features at the arm bar, causal by construction (every value uses only bars/sessions strictly
before the arm minute). See FEATURES_B.md (written at the end of a run) for the one-line
timestamp proof of each column.

Base book: research/hod_entry/causal_arming_causal.csv rows status=='fill' (9,911 rows).
Per fill day this script loads minute bars for the WHOLE HOD universe that day (adv20 >= 100k,
test tickers excluded) once, computes a per-minute breadth series (count/share of names >= 5%
above their session open) and a per-symbol/per-minute boolean matrix (for sector-peer counts),
and caches both under research/hod_entry/breadth_cache/<day>_*.{csv,npz} so a re-run resumes for
free. This is the heavy step (~12 s/day at ~2,000 symbols measured on 2025-07-01); ~250 distinct
days (fill days + their 5 prior sessions for the regime features) -> ~50 min single-process.

Run:
    nice -n 19 python3 research/hod_entry/build_features_B.py \
        >> research/hod_entry/build_features_B.log 2>&1 &

Writes: research/hod_entry/features_1478_B.csv (final, one row per base fill)
        research/hod_entry/breadth_cache/<day>_breadth.csv   (m, count, denom, share)
        research/hod_entry/breadth_cache/<day>_matrix.npz    (symbols, is5, have_data, opens)
        research/hod_entry/breadth_cache/<day>_spy.csv       (m, o, c_ffill)
        research/hod_entry/breadth_cache/_progress.json      (resumable checkpoint)
"""
import os
import sys
import json
import time
import sqlite3

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, os.path.join(ROOT, 'research/hod_entry'))
import causal_arming as ca  # noqa: E402
import sip_rebuild as sr  # noqa: E402

HOD_DIR = os.path.join(ROOT, 'research/hod_entry')
CACHE_DIR = os.path.join(HOD_DIR, 'breadth_cache')
OUT_CSV = os.path.join(HOD_DIR, 'features_1478_B.csv')
LOG_PATH = os.path.join(HOD_DIR, 'build_features_B.log')
PROGRESS_PATH = os.path.join(CACHE_DIR, '_progress.json')

BASE_CSV = os.path.join(HOD_DIR, 'causal_arming_causal.csv')
UNIVERSE_CSV = os.path.join(ROOT, 'research/bf_zero/universe.csv')
PANEL_NPZ = os.path.join(ROOT, 'research/multiday/data/panel_f3f4.npz')
DAILY_PARQUET = os.path.join(ROOT, 'data/research/databento/equs_daily_2025_2026.parquet')

MIN_ADV20 = 100_000
OPEN_M, CLOSE_M = 570, 960  # RTH minute grid [570, 960) matches causal_arming._rth
ELEVEN_ET_M = 660  # 11:00 ET
MINUTES = np.arange(OPEN_M, CLOSE_M)
N_MIN = len(MINUTES)


def log(msg):
    line = f'{pd.Timestamp.now()} {msg}'
    print(line, flush=True)


def arm_minute(fill_min):
    """Integer minute used for every crowd feature: fill_min - 1 (PREREG_1478 wording), one whole
    minute earlier than cell_1445's own arm bar (m = floor(fill_min)) -- the breadth/sector/SPY
    snapshot is always taken from a bar that closed strictly before the arm bar itself starts."""
    return int(fill_min) - 1


# --------------------------------------------------------------------------------------- universe
def load_universe_all():
    u = pd.read_csv(UNIVERSE_CSV, dtype={'symbol': str}, keep_default_na=False)
    for c in ('open', 'high', 'low', 'close', 'volume', 'adv20', 'prev_vol'):
        u[c] = pd.to_numeric(u[c], errors='coerce')
    u = u.rename(columns={'bar_date': 'day'})
    u = u[~u.symbol.str.match(ca.TEST_TICKER)]
    return u


def day_universe(u_all, day):
    """HOD universe for `day`: adv20 >= 100k, test tickers already excluded. symbol -> open."""
    sub = u_all[(u_all.day == day) & (u_all.adv20 >= MIN_ADV20)]
    sub = sub.drop_duplicates('symbol')
    return sub[['symbol', 'open']].reset_index(drop=True)


# --------------------------------------------------------------------------------- breadth (crowd)
def cache_paths(day):
    return (os.path.join(CACHE_DIR, f'{day}_breadth.csv'),
            os.path.join(CACHE_DIR, f'{day}_matrix.npz'),
            os.path.join(CACHE_DIR, f'{day}_spy.csv'))


def build_day(con, sipcon, day, u_all, counts):
    """Build (and cache) this day's breadth series, per-symbol >=5%-above-open matrix, and SPY
    minute series. Skips the DB reads entirely if the cache already exists (resumable)."""
    bcsv, bnpz, spycsv = cache_paths(day)
    if os.path.exists(bcsv) and os.path.exists(bnpz) and os.path.exists(spycsv):
        return

    dsub = day_universe(u_all, day)
    syms = dsub.symbol.tolist()
    opens = dict(zip(dsub.symbol, dsub.open))
    bars = ca.load_day_bars(con, day, syms, sipcon, counts)

    is5 = np.zeros((len(syms), N_MIN), dtype=bool)
    have = np.zeros((len(syms), N_MIN), dtype=bool)
    for i, s in enumerate(syms):
        b = bars.get(s)
        o = opens.get(s)
        if b is None or len(b) == 0 or not o or o <= 0:
            continue
        ser = pd.Series(b.c.values, index=b.m.values).reindex(MINUTES).ffill()
        valid = ser.notna().values
        have[i] = valid
        is5[i, valid] = (ser.values[valid] >= 1.05 * o)

    count = is5.sum(axis=0)
    denom = have.sum(axis=0)
    share = np.divide(count, denom, out=np.full(N_MIN, np.nan), where=denom > 0)
    pd.DataFrame({'m': MINUTES, 'count': count, 'denom': denom, 'share': share}).to_csv(bcsv, index=False)
    np.savez_compressed(bnpz, symbols=np.array(syms, dtype=object), is5=is5, have=have,
                         opens=np.array([opens.get(s, np.nan) for s in syms], dtype=float))

    # SPY minute series, cached alongside (own query, cache.db only per spec)
    q = ("select timestamp as t, open as o, high as h, low as l, close as c, volume as v "
         "from intraday_bars_1min where bar_date=? and symbol='SPY'")
    g = pd.read_sql(q, con, params=[day])
    spy_open = float(opens['SPY']) if 'SPY' in opens else np.nan
    if len(g):
        b = ca._rth(g, 't')
        ser = pd.Series(b.c.values, index=b.m.values).reindex(MINUTES).ffill()
        if np.isnan(spy_open) and len(b):
            spy_open = float(b.o.iloc[0])
        pd.DataFrame({'m': MINUTES, 'o': spy_open, 'c_ffill': ser.values}).to_csv(spycsv, index=False)
    else:
        pd.DataFrame({'m': MINUTES, 'o': spy_open, 'c_ffill': np.nan}).to_csv(spycsv, index=False)


def load_cached_day(day, sic2_map):
    bcsv, bnpz, spycsv = cache_paths(day)
    bdf = pd.read_csv(bcsv).set_index('m')
    z = np.load(bnpz, allow_pickle=True)
    symbols = z['symbols']
    sic2_arr = np.array([sic2_map.get(s) for s in symbols], dtype=object)
    spy = pd.read_csv(spycsv).set_index('m')
    return bdf, symbols, sic2_arr, z['is5'], z['have'], spy


# ------------------------------------------------------------------------------------------ sic2
def load_sic2_map():
    d = np.load(PANEL_NPZ, allow_pickle=True)
    return dict(zip(d['symbols'].tolist(), d['sic2'].tolist()))


# ------------------------------------------------------------------------------------- SPY daily
def load_spy_daily():
    df = pd.read_parquet(DAILY_PARQUET, columns=['bar_date', 'symbol', 'close'])
    df = df[df.symbol == 'SPY'].drop_duplicates('bar_date').sort_values('bar_date')
    return df.set_index('bar_date')['close']


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    t_start = time.time()
    log('[start] Feature Set B build')

    base = pd.read_csv(BASE_CSV, low_memory=False)
    fills = base[base.status == 'fill'][['day', 'symbol', 'fill_min', 'split']].reset_index(drop=True)
    log(f'[base] {len(fills)} fill rows, {fills.day.nunique()} distinct fill days')

    u_all = load_universe_all()
    trading_days = np.array(sorted(u_all.day.unique()))
    log(f'[universe] {len(trading_days)} distinct trading days in universe.csv '
        f'({trading_days[0]}..{trading_days[-1]})')

    fill_days = sorted(fills.day.unique())
    needed = set(fill_days)
    day_idx = {d: i for i, d in enumerate(trading_days)}
    for d in fill_days:
        i = day_idx.get(d)
        if i is None:
            log(f'[WARN] fill day {d} not found in universe.csv trading-day calendar')
            continue
        for k in range(1, 6):
            if i - k >= 0:
                needed.add(trading_days[i - k])
    needed = sorted(needed)
    log(f'[plan] {len(needed)} distinct days need a breadth build (fill days + prior-5 regime days)')

    con = sqlite3.connect(sr.CACHE_DB_URI, uri=True, timeout=120)
    sipcon = sqlite3.connect(ca.BARS_SIP_URI, uri=True, timeout=120)
    counts = {}
    done = 0
    for day in needed:
        bcsv, bnpz, spycsv = cache_paths(day)
        already = os.path.exists(bcsv) and os.path.exists(bnpz) and os.path.exists(spycsv)
        t0 = time.time()
        build_day(con, sipcon, day, u_all, counts)
        done += 1
        if not already:
            log(f'[day {done}/{len(needed)}] {day} built in {time.time()-t0:.1f}s')
        if done % 10 == 0 or done == len(needed):
            with open(PROGRESS_PATH, 'w') as f:
                json.dump({'done': done, 'total': len(needed), 'last_day': day,
                            'elapsed_s': time.time() - t_start, 'counts': counts}, f)
            log(f'[progress] {done}/{len(needed)} days ({time.time()-t_start:.0f}s elapsed)')

    log('[phase] all day caches built -- computing per-fill features')

    sic2_map = load_sic2_map()
    spy_daily = load_spy_daily()

    day_cache = {}

    def get_day(day):
        if day not in day_cache:
            day_cache[day] = load_cached_day(day, sic2_map)
        return day_cache[day]

    rows = []
    n_oob = 0
    for r in fills.itertuples(index=False):
        day, symbol, fill_min, split = r.day, r.symbol, r.fill_min, r.split
        am = arm_minute(fill_min)
        out = dict(day=day, symbol=symbol, fill_min=fill_min, split=split, arm_minute=am)

        if am < OPEN_M or am >= CLOSE_M:
            n_oob += 1
            out.update(breadth_count_j=np.nan, breadth_share_j=np.nan, n_universe_j=np.nan,
                       sector_peers_j=np.nan, sector_peers_pool_j=np.nan, sic2=np.nan,
                       spy_ret_open_to_j=np.nan, breadth_5d=np.nan, spy_ret_5d=np.nan)
            rows.append(out)
            continue

        bdf, symbols, sic2_arr, is5, have, spy = get_day(day)
        out['breadth_count_j'] = float(bdf.loc[am, 'count'])
        out['breadth_share_j'] = float(bdf.loc[am, 'share'])
        out['n_universe_j'] = float(bdf.loc[am, 'denom'])

        col = am - OPEN_M
        sic2 = sic2_map.get(symbol)
        out['sic2'] = sic2 if sic2 is not None else np.nan
        if sic2 is None:
            out['sector_peers_j'] = np.nan
            out['sector_peers_pool_j'] = np.nan
        else:
            peer_mask = (sic2_arr == sic2) & (symbols != symbol)
            pool = have[peer_mask, col]
            out['sector_peers_pool_j'] = float(pool.sum())
            out['sector_peers_j'] = float((is5[peer_mask, col] & pool).sum())

        spy_o = spy['o'].iloc[0]
        spy_c = spy.loc[am, 'c_ffill'] if am in spy.index else np.nan
        out['spy_ret_open_to_j'] = (float(spy_c) / float(spy_o) - 1.0) if (spy_o and not pd.isna(spy_c)) else np.nan

        i = day_idx.get(day)
        if i is not None:
            prior = [trading_days[i - k] for k in range(1, 6) if i - k >= 0]
            shares = []
            for pday in prior:
                pb, *_ = get_day(pday)
                if ELEVEN_ET_M in pb.index:
                    v = pb.loc[ELEVEN_ET_M, 'share']
                    if not pd.isna(v):
                        shares.append(v)
            out['breadth_5d'] = float(np.mean(shares)) if shares else np.nan
        else:
            out['breadth_5d'] = np.nan

        before = spy_daily[spy_daily.index < day]
        if len(before) >= 6:
            out['spy_ret_5d'] = float(before.iloc[-1] / before.iloc[-6] - 1.0)
        else:
            out['spy_ret_5d'] = np.nan

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    log(f'[write] {OUT_CSV} ({len(out_df)} rows, {n_oob} arm-minute-out-of-range rows)')

    for c in ['breadth_count_j', 'breadth_share_j', 'sector_peers_j', 'spy_ret_open_to_j',
              'breadth_5d', 'spy_ret_5d']:
        cov = 1.0 - out_df[c].isna().mean()
        log(f'[coverage] {c}: {cov:.3%} non-NaN')

    log(f'[done] total elapsed {time.time()-t_start:.0f}s')


if __name__ == '__main__':
    main()
