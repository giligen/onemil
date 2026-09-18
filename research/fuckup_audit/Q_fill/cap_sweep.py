#!/usr/bin/env python3
"""Stage Q step 6 — what a WIDER stop-limit cap would have been worth, measured.

`orb.yaml::entry.stop_limit_buffer_bps` (30) sets the only knob in the fill
model: limit = range_high x (1 + bps/10000).  Widening it converts a
non-marketable order (ask above the cap at the trigger instant -> rest, fill
late or never) into an immediate fill AT THE ASK.  The trade is then re-simulated
from the breakout bar with that higher entry, so the lock trigger, the scale
level and the ATR-floored stop all move with it — the cost of the wider cap is
paid, not assumed.

Rows the wider cap still cannot reach keep the MEASURED treatment (the delayed
fill / no fill measured at the 30-bps cap) — conservative, since a higher cap
would in truth be reached sooner.

Output: Q_fill/dump_cap{bps}.csv  (feed to rescore_q.run-style pipeline)
Usage: python3 cap_sweep.py 50 100
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/P_cost')

import study_orb_pipeline_static_lock as P            # noqa: E402
from study_orb import _bars_to_df                     # noqa: E402
from exit_times import instrumented_walk              # noqa: E402
from trading.orb_csv import read_orb_csv              # noqa: E402

Q = f'{ROOT}/research/fuckup_audit/Q_fill'
PC = f'{ROOT}/research/fuckup_audit/P_cost'
D1 = f'{ROOT}/research/fuckup_audit/D1_orb'
SQL = ("SELECT timestamp, open, high, low, close, volume FROM intraday_bars_1min "
       "WHERE symbol=? AND bar_date=? ORDER BY timestamp")
OLD_POS = 50_000.0


def main() -> int:
    bpss = [int(x) for x in sys.argv[1:]] or [50]
    cfg = P.load_bt_config()
    P.MIN_STOP_PCT = cfg['min_stop_pct']; P.OLD_POS = cfg['old_pos_ref']
    P.LOCK_TRIGGER_R = cfg['lock_arm_r']; P.LOCK_STOP_R = cfg['lock_stop_r']
    P.EXIT_SLIP_BPS = cfg['exit_slip_bps']; P.FORCE_CLOSE_ET = cfg['force_close_et']

    dump = read_orb_csv(f'{D1}/candidates_dump.csv')
    dump['date'] = pd.to_datetime(dump['date']).dt.strftime('%Y-%m-%d')
    dump['key'] = dump.symbol + '|' + dump.date
    shares = np.maximum(1, (OLD_POS / dump.entry_price).astype(int))
    base = pd.read_csv(f'{Q}/per_trade_arms.csv', keep_default_na=False,
                       na_values=[''], dtype={'symbol': str, 'date': str})
    meas = dict(zip(base.key, base.pnl_measured))
    xt = pd.read_csv(f'{PC}/exit_times.csv', keep_default_na=False, na_values=[''],
                     dtype={'symbol': str, 'date': str},
                     usecols=['symbol', 'date', 'range_high', 'range_low',
                              'atr14', 'entry_ts'])
    xt['key'] = xt.symbol + '|' + xt.date
    X = xt.set_index('key')
    sp = pd.read_parquet(f'{PC}/spreads.parquet')
    sp['key'] = sp.symbol + '|' + sp.date
    ask = dict(zip(sp[sp.cov_entry].key, sp[sp.cov_entry].entry_ask))

    con = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    for bps in bpss:
        pnl = dump.pnl.astype(float).values.copy()
        n_conv = n_keep = 0
        for i, r in enumerate(dump.itertuples()):
            if r.entered != 1 or r.key not in meas:
                continue
            pnl[i] = float(meas[r.key])                 # the measured baseline
            a = ask.get(r.key)
            cap30 = float(r.entry_price)
            if a is None or not np.isfinite(a) or a <= cap30 * (1 + 1e-12):
                continue                                 # already marketable
            row = X.loc[r.key]
            rh, rl = float(row.range_high), float(row.range_low)
            newcap = round(rh * (1 + bps / 10000.0), 2)
            if a > newcap + 1e-9:
                n_keep += 1
                continue                                 # still not marketable
            n_conv += 1
            bars = _bars_to_df([dict(x) for x in con.execute(SQL, (r.symbol, r.date))])
            brk = pd.Timestamp(row.entry_ts).tz_convert('UTC')
            sh = int(shares.iat[i])
            a14 = float(row.atr14) if np.isfinite(row.atr14) else None
            px, rsn, _, _, _ = instrumented_walk(bars, float(a), rh, rl, brk,
                                                 sh, a14, cfg)
            pnl[i] = (px - float(a)) * sh if rsn != 'no_bars' else 0.0
        out = read_orb_csv(f'{D1}/candidates_dump.csv')
        out['pnl'] = pnl
        out['pnl_pct'] = pnl / (out.entry_price * shares.values) * 100
        out.loc[out.entered != 1, ['pnl', 'pnl_pct']] = 0.0
        out.to_csv(f'{Q}/dump_cap{bps}.csv', index=False)
        print(f'cap {bps} bps: converted to an immediate fill {n_conv}, '
              f'still resting {n_keep}', flush=True)
        for n in (8, 3):
            e = dict(os.environ)
            e.update(ORB_BT_FEATURES_CSV='analysis_results/orb_features_20260916_2053.csv',
                     ORB_BT_RISK='375', ORB_BT_RESIM_CACHE=f'{Q}/dump_cap{bps}.csv',
                     ORB_BT_N=str(n), ORB_BT_ACCOUNT=repr(3333.333333333333 * n),
                     ORB_SKIP_Q1='1', ORB_BT_BOOK_OUT=f'{Q}/book_cap{bps}_n{n}.csv',
                     ORB_BT_MONTHLY_OUT=f'{Q}/monthly_cap{bps}_n{n}.csv')
            with open(f'{Q}/log_cap{bps}_n{n}.txt', 'w') as fh:
                rc = subprocess.call(['nice', '-n', '10', 'python3', '-u',
                                      'study_orb_pipeline_static_lock.py'],
                                     stdout=fh, stderr=subprocess.STDOUT, env=e)
            print(f'  cap{bps} n={n} rc={rc}', flush=True)
    con.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
