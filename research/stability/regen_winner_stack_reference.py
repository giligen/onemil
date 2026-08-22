#!/usr/bin/env python3
"""Regenerate the winner-stack reference targets on CURRENT cache data
(design §1b P0-6: gate-2's target is the REGENERATED book, recomputed at
validation time — never the stale artifact).

Computes, per trade of the current B+ book (analysis_results/orb_bplus_book.csv):
  BASE — static-lock repro (sz1 with floor = range_low) — sanity vs _sized_pnl
  B    — SZ1 ATR stop-floor k=0.25 only (phaseB_regime_atr.sz1_exit)
  C    — SCALE 40%@3.0R + SZ1 k=0.25 (phaseB_frontier.variant_scale_sz1
         semantics, replicated verbatim below)

ATR source: the SHARED trading/orb_winner_stack.atr14_t1 over cache
daily_bars strictly before the trade date (the frozen ≥15-bar/fail-open
rule — P0-6.1; NOT the frontier's 14-bar variant). Run AFTER
scripts/backfill_daily_bars_gaps.py.

Physics: per-pair indexed cache queries (identical reconstruction to
orb_clean_harness.build_physics, validated during the adversarial review —
the C-point reproduced $9,092 exactly on pre-backfill data).

Outputs research/stability/winner_stack_regen_reference.json with monthly
tables for BASE/B/C + a per-trade CSV. The Monday pre-boot flip is GATED on
this regen succeeding and the flags-on pipeline matching C to < $5/month.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, 'research', 'scripts'))
sys.path.insert(0, os.path.join(REPO, 'research', 'stability'))

from resim_exit import EXIT_SLIP, TG, _clip_to_fc, touchgo_prefire  # noqa: E402
from phaseB_regime_atr import sz1_exit                              # noqa: E402
from trading.orb_touchgo_filter import find_breakout_bar_ts         # noqa: E402
from trading.orb_winner_stack import atr14_t1, floored_stop         # noqa: E402

ENTRY_SLIP_BPS = 30.0
K, FRAC, LVL = 0.25, 0.40, 3.0
BOOK = 'analysis_results/orb_bplus_book.csv'
OUT_JSON = 'research/stability/winner_stack_regen_reference.json'
OUT_CSV = 'research/stability/winner_stack_regen_pertrade.csv'


def _et_minutes(ts):
    et = ts.dt.tz_convert('America/New_York')
    return (et.dt.hour * 60 + et.dt.minute).to_numpy()


def build_pair(con, sym, date_str, csv_entry):
    """Per-pair physics record (validated reconstruction, review session)."""
    q = ("SELECT timestamp, open, high, low, close, volume "
         "FROM intraday_bars_1min WHERE symbol=? AND bar_date=? "
         "ORDER BY timestamp")
    df = pd.read_sql_query(q, con, params=(sym, date_str))
    if df.empty:
        return None
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    etm = _et_minutes(df['timestamp'])
    open_idx = np.where(etm == 570)[0]
    if len(open_idx) == 0:
        return None
    open_ts = df['timestamp'].iloc[open_idx[0]]
    range_end = open_ts + timedelta(minutes=5)
    rbars = df[(df['timestamp'] >= open_ts) & (df['timestamp'] < range_end)]
    if len(rbars) < 5:
        return None
    rh = float(rbars['high'].max())
    rl = float(rbars['low'].min())
    search = df[(df['timestamp'] >= range_end)
                & (df['timestamp'] < range_end + timedelta(minutes=60))]
    entry_ts = find_breakout_bar_ts(search, rh)
    if entry_ts is None:
        return None
    entry = rh * (1 + ENTRY_SLIP_BPS / 10000)
    if csv_entry is not None and abs(csv_entry - entry) / entry > 0.001:
        return None
    post = df[df['timestamp'] >= entry_ts].reset_index(drop=True)
    petm = _et_minutes(post['timestamp'])
    keep = petm <= 959
    return {'entry': entry, 'rh': rh, 'rl': rl, 'rsize': rh - rl,
            'etm': petm[keep],
            'o': post['open'].to_numpy(float)[keep],
            'h': post['high'].to_numpy(float)[keep],
            'l': post['low'].to_numpy(float)[keep],
            'c': post['close'].to_numpy(float)[keep]}


def variant_scale_sz1(p, floor_price, frac, level_R, arm_R=1.75, stop_R=0.5):
    """VERBATIM phaseB_frontier.variant_scale_sz1 (the frozen C-point walk),
    except the floored stop is precomputed by the caller through the SHARED
    trading/orb_winner_stack.floored_stop (frozen ATR rule + P1-3 degenerate
    clamp) — `floor_price` here equals the harness' internal
    max(range_low, entry − k×atr) on every non-degenerate path."""
    tg = touchgo_prefire(p, TG)
    if tg is not None:
        px, rsn = tg
        return px / p['entry'] - 1, rsn
    ep, R = p['entry'], p['rsize']
    n = _clip_to_fc(p)
    h, l = p['h'][:n], p['l'][:n]
    slip = 1 - EXIT_SLIP
    rl = floor_price
    scale_px = ep + level_R * R
    trig = ep + arm_R * R
    lock = ep + stop_R * R
    stop = rl
    armed = False
    scale_i = None
    for i in range(1, n):
        if not armed and h[i] >= trig:
            armed = True
            stop = max(stop, lock)
        if l[i] <= stop and h[i] < scale_px:
            return stop * slip / ep - 1, ('lock' if armed else 'stop')
        if h[i] >= scale_px:
            scale_i = i
            break
    if scale_i is None:
        stop = rl
        armed = False
        for i in range(1, n):
            if not armed and h[i] >= trig:
                armed = True
                stop = max(stop, lock)
            if l[i] <= stop:
                return stop * slip / ep - 1, ('lock' if armed else 'stop')
        return float(p['c'][:n][n - 1]) * slip / ep - 1, 'eod'
    trigR = ep + arm_R * R
    lockR = ep + stop_R * R
    stop2 = rl
    armed2 = bool(np.max(h[1:scale_i + 1]) >= trigR)
    if armed2:
        stop2 = max(stop2, lockR)
    run_px = None
    for i in range(scale_i, n):
        if not armed2 and h[i] >= trigR:
            armed2 = True
            stop2 = max(stop2, lockR)
        if l[i] <= stop2:
            run_px = stop2 * slip
            break
    if run_px is None:
        run_px = float(p['c'][:n][n - 1]) * slip
    ret = (frac * (scale_px * slip / ep - 1)
           + (1 - frac) * (run_px / ep - 1))
    return ret, 'scale_sz1'


def atr_frozen(con, sym, date_str):
    """ATR14 for day T via the SHARED frozen rule (bars strictly before T)."""
    d = pd.read_sql_query(
        "SELECT bar_date, high, low, close FROM daily_bars "
        "WHERE symbol=? AND bar_date<? ORDER BY bar_date", con,
        params=(sym, date_str))
    return atr14_t1(d.tail(40))


def monthly(dfd, col):
    return {m: round(v, 2)
            for m, v in dfd.groupby('month')[col].sum().items()}


def main():
    con = sqlite3.connect('data/cache.db')
    b = pd.read_csv(BOOK)
    b['date_dt'] = pd.to_datetime(b['date'])
    rows = []
    n_bound = 0
    for _, r in b.iterrows():
        ds = r['date_dt'].strftime('%Y-%m-%d')
        p = build_pair(con, r['symbol'], ds, float(r['entry_price']))
        atr = atr_frozen(con, r['symbol'], ds)
        rec = dict(symbol=r['symbol'], date=ds, month=r['month'],
                   base_book=float(r['_sized_pnl']), atr14=atr)
        if p is None:
            print(f"MISSING PATH {r['symbol']} {ds} — falling back to "
                  f"book pnl for all variants")
            rec.update(base=float(r['_sized_pnl']), b=float(r['_sized_pnl']),
                       c=float(r['_sized_pnl']), reason_c='MISSING',
                       floored=False)
            rows.append(rec)
            continue
        rp = float(r['_rp_position'])
        floor_price, status = floored_stop(p['rl'], p['entry'], atr, K)
        if status == 'bound':
            n_bound += 1
        base_ret = sz1_exit(p, p['rl'])
        b_ret = sz1_exit(p, floor_price)
        c_ret = variant_scale_sz1(p, floor_price, FRAC, LVL)
        rec.update(base=rp * base_ret[0], b=rp * b_ret[0], c=rp * c_ret[0],
                   reason_base=base_ret[1], reason_b=b_ret[1],
                   reason_c=c_ret[1], floored=(status == 'bound'),
                   floor_status=status, floor_price=floor_price)
        rows.append(rec)
    con.close()
    d = pd.DataFrame(rows)
    out = {
        'generated_from': BOOK,
        'n_trades': len(d),
        'n_floor_bound': n_bound,
        'totals': {k: round(float(d[k].sum()), 2)
                   for k in ('base_book', 'base', 'b', 'c')},
        'greens': {k: int((d.groupby('month')[k].sum() > 0).sum())
                   for k in ('base', 'b', 'c')},
        'monthly': {k: monthly(d, k) for k in ('base_book', 'base', 'b', 'c')},
        'reason_mix_c': d['reason_c'].value_counts().to_dict(),
    }
    d.to_csv(OUT_CSV, index=False)
    json.dump(out, open(OUT_JSON, 'w'), indent=1)
    print(f"\nBASE(book)  total = {out['totals']['base_book']:+.2f}")
    print(f"BASE(repro) total = {out['totals']['base']:+.2f}  "
          f"(max per-trade err "
          f"${(d['base'] - d['base_book']).abs().max():.2f})")
    print(f"B (SZ1 k={K})        total = {out['totals']['b']:+.2f}  "
          f"greens {out['greens']['b']}/20")
    print(f"C (scale40@3R + SZ1) total = {out['totals']['c']:+.2f}  "
          f"greens {out['greens']['c']}/20  floor bound on {n_bound} trades")
    print(f"reason mix (C): {out['reason_mix_c']}")
    print(f"wrote {OUT_JSON} + {OUT_CSV}")


if __name__ == '__main__':
    main()
