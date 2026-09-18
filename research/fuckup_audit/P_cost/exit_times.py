#!/usr/bin/env python3
"""Stage P step 1 — the ENTRY and EXIT instants of every ORB fill in the D1 dump.

The D1 candidate dump (`candidates_dump.csv`) carries `exit_reason` but no
timestamp, so a measured cost needs the exit bar recovered.  This replays the
SHIPPED exit physics (`study_orb_pipeline_static_lock.simulate_winner_stack`,
winner stack ON per orb.yaml) with an instrumented twin that also records the
bar timestamps, and ASSERTS the twin reproduces the shipped function's
(exit_price, reason) for every row.  Parity by construction.

Output: P_cost/exit_times.csv — one row per entered candidate:
    symbol, date, entry_price, range_high, range_low, entry_ts, exit_ts,
    exit_price, exit_reason, scale_ts, scale_px, shares, n_bars

Usage: python3 research/fuckup_audit/P_cost/exit_times.py
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import study_orb_pipeline_static_lock as P            # noqa: E402
from study_orb import _bars_to_df                     # noqa: E402
from persistence.database import Database             # noqa: E402
from trading.orb_csv import read_orb_csv              # noqa: E402
from trading.orb_touchgo_filter import (              # noqa: E402
    evaluate_rule_m, evaluate_rule_d, find_breakout_bar_ts,
)
from trading.orb_winner_stack import (                # noqa: E402
    floored_stop, scale_params, atr14_t1,
)

D1 = f'{ROOT}/research/fuckup_audit/D1_orb'
OUT = f'{ROOT}/research/fuckup_audit/P_cost/exit_times.csv'
CHUNK = 500


def instrumented_walk(bars, entry_price, range_high, range_low, entry_time,
                      shares, atr14, cfg):
    """Twin of simulate_winner_stack that also returns the bar timestamps.

    Returns (exit_price, reason, exit_ts, scale_ts, scale_px).
    For a scale_* outcome the blended price is returned (as the shipped
    function does) and BOTH leg timestamps are reported.
    """
    from datetime import time as _dtime
    range_size = range_high - range_low
    slip = 1 - P.EXIT_SLIP_BPS / 10000
    _et = bars['timestamp'].dt.tz_convert('America/New_York').dt.time
    _fc_h, _fc_m = (int(x) for x in P.FORCE_CLOSE_ET.split(':'))
    bars = bars[_et <= _dtime(_fc_h, _fc_m)]
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    if len(post) == 0:
        return entry_price, 'no_bars', None, None, None
    ts = post['timestamp'].to_numpy()

    eb = post.iloc[0]
    fire_m, exit_m = evaluate_rule_m(float(eb['open']), float(eb['high']),
                                     float(eb['low']), float(eb['close']),
                                     P.TOUCHGO_CFG)
    if fire_m and exit_m is not None:
        return exit_m * slip, 'tag_bb', ts[0], None, None
    if len(post) >= 2:
        b1 = post.iloc[1]
        fire_d, exit_d = evaluate_rule_d(entry_price, float(b1['low']),
                                         range_size, P.TOUCHGO_CFG)
        if fire_d and exit_d is not None:
            return exit_d * slip, 'tag_b1', ts[1], None, None

    stop0 = range_low
    if cfg['atr_floor_enabled']:
        stop0, _st = floored_stop(range_low, entry_price, atr14,
                                  cfg['atr_floor_k'])
    scale_px = None
    frac_eff = 0.0
    if cfg['scale_enabled'] and shares >= 1:
        px, qty = scale_params(entry_price, range_size, cfg['scale_frac'],
                               cfg['scale_level_r'], shares)
        if qty >= 1:
            scale_px = px
            frac_eff = qty / float(shares)

    trig = entry_price + P.LOCK_TRIGGER_R * range_size
    lock = entry_price + P.LOCK_STOP_R * range_size
    highs = post['high'].to_numpy(dtype=float)
    lows = post['low'].to_numpy(dtype=float)
    closes = post['close'].to_numpy(dtype=float)
    n = len(post)

    stop = stop0
    armed = False
    scale_i = None
    for i in range(1, n):
        if not armed and highs[i] >= trig:
            armed = True
            stop = max(stop, lock)
        if lows[i] <= stop and (scale_px is None or highs[i] < scale_px):
            return stop * slip, ('lock' if armed else 'stop'), ts[i], None, None
        if scale_px is not None and highs[i] >= scale_px:
            scale_i = i
            break
    if scale_i is None:
        return float(closes[-1]) * slip, 'eod', ts[-1], None, None

    stop2 = stop0
    armed2 = bool(highs[1:scale_i + 1].max() >= trig) if scale_i >= 1 else False
    if armed2:
        stop2 = max(stop2, lock)
    run_px = None
    run_rsn = 'eod'
    run_ts = ts[-1]
    for i in range(scale_i, n):
        if not armed2 and highs[i] >= trig:
            armed2 = True
            stop2 = max(stop2, lock)
        if lows[i] <= stop2:
            run_px = stop2 * slip
            run_rsn = 'lock' if armed2 else 'stop'
            run_ts = ts[i]
            break
    if run_px is None:
        run_px = float(closes[-1]) * slip
    ret = (frac_eff * (scale_px * slip / entry_price - 1)
           + (1 - frac_eff) * (run_px / entry_price - 1))
    return (entry_price * (1 + ret), f'scale_{run_rsn}', run_ts,
            ts[scale_i], scale_px)


def main() -> int:
    cfg = P.load_bt_config()
    P.MIN_STOP_PCT = cfg['min_stop_pct']
    P.OLD_POS = cfg['old_pos_ref']
    P.LOCK_TRIGGER_R = cfg['lock_arm_r']
    P.LOCK_STOP_R = cfg['lock_stop_r']
    P.EXIT_SLIP_BPS = cfg['exit_slip_bps']
    P.FORCE_CLOSE_ET = cfg['force_close_et']
    assert cfg['atr_floor_enabled'] and cfg['scale_enabled'], 'winner stack expected ON'

    d = read_orb_csv(f'{D1}/candidates_dump.csv')
    d['date'] = pd.to_datetime(d['date'])
    d = d[d['entered'] == 1].reset_index(drop=True)
    print(f'entered candidates: {len(d)}', flush=True)

    pairs = [(r.symbol, r.date.strftime('%Y-%m-%d')) for r in d.itertuples()]
    atr = P.build_atr14_lookup(pairs)
    print(f'ATR14 built for {sum(1 for v in atr.values() if v is not None)}'
          f'/{len(atr)} pairs', flush=True)

    rows = []
    mismatch = 0
    nofit = 0
    # Indexed per-pair reads (idx_intraday_bars_symbol_date). The Database
    # bulk helper does a full table scan per call, which is ~15x slower here.
    import sqlite3
    con = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    SQL = ("SELECT timestamp, open, high, low, close, volume FROM intraday_bars_1min "
           "WHERE symbol=? AND bar_date=? ORDER BY timestamp")
    for c0 in range(0, len(d), CHUNK):
        sub = d.iloc[c0:c0 + CHUNK]
        cache = {}
        for r in sub.itertuples():
            k = (r.symbol, r.date.strftime('%Y-%m-%d'))
            if k in cache:
                continue
            cache[k] = _bars_to_df([dict(x) for x in con.execute(SQL, k)])
        for r in sub.itertuples():
            key = (r.symbol, r.date.strftime('%Y-%m-%d'))
            bars = cache.get(key)
            if bars is None or bars.empty:
                nofit += 1
                continue
            open_ts = P._session_open_timestamp(bars)
            if open_ts is None:
                nofit += 1
                continue
            range_end = open_ts + timedelta(minutes=5)
            rb = bars[(bars['timestamp'] >= open_ts) & (bars['timestamp'] < range_end)]
            if len(rb) < 5:
                nofit += 1
                continue
            rh = float(rb['high'].max())
            rl = float(rb['low'].min())
            search = bars[(bars['timestamp'] >= range_end) &
                          (bars['timestamp'] < range_end + timedelta(minutes=60))]
            entry_ts = find_breakout_bar_ts(search, rh)
            if entry_ts is None:
                nofit += 1
                continue
            entry_p = float(r.entry_price)
            shares = max(1, int(P.OLD_POS / entry_p))
            a14 = atr.get(key)
            ref_px, ref_rsn = P.simulate_winner_stack(
                bars, entry_p, rh, rl, entry_ts, shares, atr14=a14,
                atr_floor_enabled=cfg['atr_floor_enabled'],
                atr_floor_k=cfg['atr_floor_k'],
                scale_enabled=cfg['scale_enabled'],
                scale_frac=cfg['scale_frac'],
                scale_level_r=cfg['scale_level_r'])
            px, rsn, xts, sts, spx = instrumented_walk(
                bars, entry_p, rh, rl, entry_ts, shares, a14, cfg)
            if rsn != ref_rsn or abs(px - ref_px) > 1e-9:
                mismatch += 1
                continue
            rows.append(dict(
                symbol=r.symbol, date=key[1], entry_price=entry_p,
                range_high=rh, range_low=rl, atr14=a14, shares=shares,
                entry_ts=pd.Timestamp(entry_ts).tz_convert('UTC').isoformat(),
                exit_ts=pd.Timestamp(xts).tz_convert('UTC').isoformat() if xts is not None else '',
                exit_price=px, exit_reason=rsn,
                scale_ts=pd.Timestamp(sts).tz_convert('UTC').isoformat() if sts is not None else '',
                scale_px=spx if spx is not None else np.nan,
                dump_reason=r.exit_reason, dump_pnl_pct=r.pnl_pct))
        del cache
        print(f'  {min(c0 + CHUNK, len(d))}/{len(d)}  rows={len(rows)} '
              f'mismatch={mismatch} nofit={nofit}', flush=True)
    con.close()

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    # parity against the dump itself
    same = (out.dump_reason == out.exit_reason).mean()
    pnl_rec = (out.exit_price - out.entry_price) / out.entry_price * 100
    print(f'\nrows {len(out)} | twin==shipped 100% (mismatch {mismatch}) | '
          f'reason==dump {same * 100:.2f}% | '
          f'max |pnl_pct diff| {np.nanmax(np.abs(pnl_rec - out.dump_pnl_pct)):.6f}',
          flush=True)
    print(f'unresolved (no bars / no breakout): {nofit}', flush=True)
    print(f'wrote {OUT}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
