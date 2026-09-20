#!/usr/bin/env python3
"""frames8 / F25 rail 1, second half — the two LIVE-CODE gates.

  P4  G1 (`g8._lock`) vs the shipped ORB exit walk
      `study_orb_pipeline_static_lock.simulate_static_lock`, on 50 REAL ORB trades, touchgo OFF
      (Rule M/D are entry-quality rules, not the exit geometry).  Compared: the exit REASON and the
      exit PRICE on every stop/lock leg.  The force-close leg is a DECLARED deviation — ORB exits at
      the last <= 15:45 bar's CLOSE with 10 bps of slip, this programme exits at the force-close
      bar's OPEN with none; it is common-mode across every frames8 cell.

  P5  G2's stop path (`g8._trail_path`) vs `trading/bf_trail.arm_and_ratchet` fed the same closed
      bars, bar for bar, on real tape.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
os.environ['ORB_TOUCHGO_ENABLED'] = '0'
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames8')
sys.path.insert(0, f'{ROOT}/research/mature_method/frames7')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import g8                                                              # noqa: E402
import c7                                                              # noqa: E402
import study_orb_pipeline_static_lock as P                             # noqa: E402
from study_orb import _bars_to_df                                      # noqa: E402
from trading.bf_trail import arm_and_ratchet                           # noqa: E402
from pass2 import load_bars                                            # noqa: E402

PC = f'{ROOT}/research/fuckup_audit/P_cost'
SQL = ("SELECT timestamp, open, high, low, close, volume FROM intraday_bars_1min "
       "WHERE symbol=? AND bar_date=? ORDER BY timestamp")
N = 50


def p4():
    cfg = P.load_bt_config()
    P.MIN_STOP_PCT = cfg['min_stop_pct']; P.OLD_POS = cfg['old_pos_ref']
    P.LOCK_TRIGGER_R = cfg['lock_arm_r']; P.LOCK_STOP_R = cfg['lock_stop_r']
    P.EXIT_SLIP_BPS = cfg['exit_slip_bps']; P.FORCE_CLOSE_ET = cfg['force_close_et']
    xt = pd.read_csv(f'{PC}/exit_times.csv', keep_default_na=False, na_values=[''],
                     dtype={'symbol': str, 'date': str})
    xt = xt[xt.range_high.notna() & xt.range_low.notna() & (xt.range_high > xt.range_low)]
    xt = xt.sort_values(['date', 'symbol']).head(400)
    con = sqlite3.connect('file:data/cache.db?mode=ro', uri=True, timeout=120)
    con.row_factory = sqlite3.Row
    ok = dev = eodcase = 0
    seen = 0
    for r in xt.itertuples():
        if seen >= N:
            break
        bars = _bars_to_df([dict(x) for x in con.execute(SQL, (r.symbol, r.date))])
        if not len(bars):
            continue
        brk = pd.Timestamp(r.entry_ts).tz_convert('UTC')
        entry = float(r.entry_price) if hasattr(r, 'entry_price') else float(r.range_high)
        px_s, why_s = P.simulate_static_lock(bars, entry, float(r.range_high),
                                             float(r.range_low), brk)
        et = bars['timestamp'].dt.tz_convert('America/New_York')
        m = (et.dt.hour * 60 + et.dt.minute).values.astype(int)
        o = bars['open'].values.astype(float); h = bars['high'].values.astype(float)
        l = bars['low'].values.astype(float); c = bars['close'].values.astype(float)
        k = np.where(bars['timestamp'].values >= np.datetime64(brk))[0]
        if not len(k):
            continue
        e = int(k[0])
        seen += 1
        oo, hh, ll, cc, mm = o[e + 1:], h[e + 1:], l[e + 1:], c[e + 1:], m[e + 1:]
        if not len(oo):
            continue
        flat = mm >= 945
        R = float(r.range_high) - float(r.range_low)
        rr, why, _ = g8._lock(oo, hh, ll, cc, mm, flat, entry, R, float(r.range_low))
        px = entry + rr * R
        if why == 'eod' or why_s == 'eod':
            eodcase += 1
            continue
        if why == why_s and abs(px - px_s) < 1e-6:
            ok += 1
        else:
            dev += 1
            if dev <= 5:
                print(f'   DEV {r.symbol} {r.date}: g8 {why} {px:.4f} vs ORB {why_s} {px_s:.4f}',
                      flush=True)
    con.close()
    print(f'P4 G1 vs simulate_static_lock: matched {ok}, deviations {dev}, '
          f'force-close (declared convention deviation) {eodcase}, of {seen} real ORB trades',
          flush=True)
    assert dev == 0, 'P4 FAILED'


def p5():
    """The stop path of `g8._trail_path` must equal `trading/bf_trail.arm_and_ratchet`'s."""
    bk = pd.read_csv(f'{ROOT}/research/mature_method/hod_frames6/book6.csv',
                     dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
    bk = bk.head(400)
    worst = 0.0
    n = 0
    for day, tr in bk.groupby('day'):
        bars = load_bars(day, sorted(set(tr.symbol)))
        for r in tr.itertuples():
            gg = bars.get(r.symbol)
            if gg is None:
                continue
            a = c7.arrays(gg)
            if a is None:
                continue
            o, h, l, c, v, m = a
            k = np.where(m == int(r.entry_m))[0]
            if not len(k):
                continue
            e = int(k[0])
            E = float(o[e]); stop = float(r.stop); R = E - stop
            if R <= 0 or e + 1 >= len(o):
                continue
            hh = h[e + 1:]
            sp, armed = g8._trail_path(hh, E, stop, E, R, 2.0, 1.0)
            # the live contract, bar by bar: the stop a bar produces is live from the NEXT bar
            hi, st, act = E, stop, False
            for i in range(len(hh)):
                worst = max(worst, abs(sp[i] - st))
                t = arm_and_ratchet(float(hh[i]), hi, st, act, E, R, 2.0, 1.0)
                hi, st, act = t.highest, t.stop, t.trailing_active
            n += 1
    print(f'P5 G2 stop path vs bf_trail.arm_and_ratchet: n={n}  max|d| = {worst:.3e}', flush=True)
    assert worst < 1e-9, 'P5 FAILED'


if __name__ == '__main__':
    p4()
    p5()
    print('PARITY 4-5 OK', flush=True)
