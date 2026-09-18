#!/usr/bin/env python3
"""Stage Q step 2 — re-simulate the trades whose stop-limit was NOT marketable
at the trigger instant and only filled later (Q_fill/walk_rows.csv).

Convention (stated in REPORT.md §1):
  * the elected stop-limit rests as a BID at the cap, so the later fill prices
    at the CAP -- identical to the as-is book's entry price.  What changes is
    the CLOCK: the trade starts at the bar containing the later fill, not at the
    breakout bar.
  * the exit physics are the SHIPPED ones (`simulate_winner_stack` via Stage P's
    instrumented twin, winner stack per orb.yaml), so arms (a)/(b)/(c) differ
    ONLY in the fill model.
  * touchgo: the live engine re-keys Rule M/D to the MARKET breakout bar and
    SKIPS touchgo when the fill lags it by more than
    `filter.touchgo.max_breakout_age_min` (15).  Here a fill inside the guard is
    evaluated on the FILL bar (the breakout bar has already closed with no
    position on) and a fill outside it gets no touchgo at all.

Output: Q_fill/delayed_resim.csv
Usage: python3 resim_delayed.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
from dataclasses import replace as dc_replace
from datetime import timedelta

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/P_cost')

import study_orb_pipeline_static_lock as P            # noqa: E402
from study_orb import _bars_to_df                     # noqa: E402
from exit_times import instrumented_walk              # noqa: E402

Q = f'{ROOT}/research/fuckup_audit/Q_fill'
PC = f'{ROOT}/research/fuckup_audit/P_cost'
OUT = f'{Q}/delayed_resim.csv'
SQL = ("SELECT timestamp, open, high, low, close, volume FROM intraday_bars_1min "
       "WHERE symbol=? AND bar_date=? ORDER BY timestamp")


def main() -> int:
    cfg = P.load_bt_config()
    P.MIN_STOP_PCT = cfg['min_stop_pct']
    P.OLD_POS = cfg['old_pos_ref']
    P.LOCK_TRIGGER_R = cfg['lock_arm_r']
    P.LOCK_STOP_R = cfg['lock_stop_r']
    P.EXIT_SLIP_BPS = cfg['exit_slip_bps']
    P.FORCE_CLOSE_ET = cfg['force_close_et']
    assert cfg['atr_floor_enabled'] and cfg['scale_enabled'], 'winner stack expected ON'
    tg_on = P.TOUCHGO_CFG
    tg_off = dc_replace(tg_on, master_enabled=False)
    max_age = int(getattr(tg_on, 'max_breakout_age_min', 15) or 15)

    w = pd.read_csv(f'{Q}/walk_rows.csv', keep_default_na=False, na_values=[''],
                    dtype={'symbol': str, 'date': str})
    w = w[w.filled_later == 1].copy()
    xt = pd.read_csv(f'{PC}/exit_times.csv', keep_default_na=False, na_values=[''],
                     dtype={'symbol': str, 'date': str},
                     usecols=['symbol', 'date', 'entry_price', 'range_high',
                              'range_low', 'atr14', 'shares', 'entry_ts',
                              'exit_price', 'exit_reason'])
    m = w.merge(xt, on=['symbol', 'date'], how='left', suffixes=('', '_x'))
    assert m.shares.notna().all(), 'a walked row is missing from exit_times.csv'
    print(f'delayed fills to re-simulate: {len(m)}', flush=True)

    con = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    rows = []
    for i, r in enumerate(m.itertuples(), 1):
        bars = _bars_to_df([dict(x) for x in con.execute(SQL, (r.symbol, r.date))])
        fill_ts = pd.Timestamp(r.fill_ts).tz_convert('UTC')
        brk_ts = pd.Timestamp(r.entry_ts).tz_convert('UTC')
        fill_bar = fill_ts.floor('min')
        lag_min = (fill_bar - brk_ts).total_seconds() / 60.0
        ep = float(r.entry_price)            # = the cap; a resting bid fills AT its limit
        sh = int(r.shares)
        a14 = r.atr14 if np.isfinite(r.atr14) else None
        # PRIMARY (c): touchgo applies exactly when the fill lands in the
        # BREAKOUT BAR itself (then the trade is the as-is trade -- same bar,
        # same rules, and the arm must reproduce as-is to the cent).  When the
        # fill lands in a LATER bar the Rule M/D decision bars closed before we
        # held the position and a retroactive tag exit at that stale limit is
        # not obtainable, so no touchgo.
        P.TOUCHGO_CFG = tg_on if lag_min < 0.5 else tg_off
        px0, rsn0, _, _, _ = instrumented_walk(
            bars, ep, float(r.range_high), float(r.range_low), fill_bar, sh, a14, cfg)
        # SENSITIVITY (c-tg): touchgo re-keyed to the FILL bar, honouring the
        # live 15-minute late-fill guard.
        P.TOUCHGO_CFG = tg_on if lag_min <= max_age else tg_off
        px1, rsn1, _, _, _ = instrumented_walk(
            bars, ep, float(r.range_high), float(r.range_low), fill_bar, sh, a14, cfg)
        rows.append(dict(symbol=r.symbol, date=r.date, entry_price=ep, shares=sh,
                         brk_ts=brk_ts.isoformat(), fill_ts=fill_ts.isoformat(),
                         lag_min=round(lag_min, 2), touchgo_ok=int(lag_min <= max_age),
                         new_exit_price=px0, new_exit_reason=rsn0,
                         new_pnl=(px0 - ep) * sh if rsn0 != 'no_bars' else 0.0,
                         new_pnl_pct=(px0 / ep - 1) * 100 if rsn0 != 'no_bars' else 0.0,
                         tg_exit_price=px1, tg_exit_reason=rsn1,
                         tg_pnl=(px1 - ep) * sh if rsn1 != 'no_bars' else 0.0,
                         asis_exit_price=r.exit_price, asis_exit_reason=r.exit_reason,
                         asis_pnl=(float(r.exit_price) - ep) * sh))
        if i % 200 == 0:
            print(f'  {i}/{len(m)}', flush=True)
    P.TOUCHGO_CFG = tg_on
    con.close()
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}  rows={len(out)}', flush=True)
    print(f'as-is P&L on these (unsized): {out.asis_pnl.sum():,.0f}  '
          f'delayed-fill P&L: {out.new_pnl.sum():,.0f}  '
          f'no-fill arm would book 0', flush=True)
    print('  touchgo sensitivity (c-tg) P&L: {:,.0f}'.format(out.tg_pnl.sum()), flush=True)
    print(out.new_exit_reason.value_counts().to_dict(), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
