#!/usr/bin/env python3
"""Independent rebuild of cell 1,464 (R floor at 2.5 % of price) from PREREG_1466.md's "Verification of
1,464" paragraph ALONE. The rebuilder has NOT read cell_1440.py, cell_1457.py, test_cell_1457.py or
RESULT_1457.md -- only sip_rebuild.walk_path (bar-walk physics, reused directly) and
causal_arming.load_day_bars (bar source: data/cache.db intraday_bars_1min UNION
research/bf_zero/bars_sip.db, the source with MORE RTH bars wins per symbol-day, SIP on a tie), both
opened read-only.

Per base fill (causal_arming_causal.csv, status == 'fill'): stop' = min(stop, fill * 0.975),
R' = fill - stop', target' = fill + 2 R'. The fill bar (m == floor(fill_min)) is stopped at stop' if its
LOW <= stop' (conservative); otherwise walk the bars after the fill bar with sip_rebuild.walk_path
(stop-first on a bar touching both, gap-through at the open, the 15:55 bar exits at its open). Fills whose
stop is unchanged keep the base outcome exactly.

COST METHOD NOTE (one documented interpretation choice, since cell_1457_features.csv's own slip_bps_1464
column turns out to be the BUILDER's rewalk output, not a base-row input -- see cell_1464_rebuild.csv's
sibling report): the round-2 corrected cost (net_R_corr_v2, base R) is decomposed into an "exit
half-spread" dollar amount that PREREG says stays UNCHANGED under the re-walk, plus a stop-slip dollar
amount charged only when the RELEVANT exit (base, for the decomposition; new, for the recharge) is a stop.
The stop-slip bps rate used on both sides of that decomposition is the PREREG-given holdout mean (35.9 bps
TRAIN-H2 / 34.8 bps VAL) -- "unchanged bps" re-applied to whichever exit price is relevant, per the PREREG's
appear/disappear rule. This keeps the whole recipe inside the PREREG's own numbers with no read of the
builder's script.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import sip_rebuild as sr          # noqa: E402  (walk_path, CACHE_DB_URI)
import causal_arming as ca        # noqa: E402  (load_day_bars, BARS_SIP_URI)

ROOT = '/home/ec2-user/onemil'
BASE_CSV = os.path.join(ROOT, 'research/hod_entry/causal_arming_causal.csv')
F1457_CSV = os.path.join(ROOT, 'research/hod_entry/cell_1457_features.csv')
OUT_CSV = os.path.join(ROOT, 'research/hod_entry/cell_1464_rebuild.csv')

HOLDOUT_BPS = {'TRAIN-H2': 35.9, 'VAL': 34.8}
STOP_WHY = {'stop', 'stop_bar'}


def log(msg):
    print(msg, flush=True)


def load_base():
    """Base book: causal_arming_causal.csv status == fill, joined to cell_1457_features.csv's round-2
    corrected cost (net_R_corr_v2) on (day, symbol, fill_min, split). The base book's split label 'TRAIN'
    is the other files' 'TRAIN-H2' (same rows) -- normalized before the join."""
    base = pd.read_csv(BASE_CSV, low_memory=False)
    bf = base[base.status == 'fill'].copy()
    bf['split'] = bf.split.replace({'TRAIN': 'TRAIN-H2'})
    f57 = pd.read_csv(F1457_CSV)
    m = bf.merge(f57[['day', 'symbol', 'fill_min', 'split', 'net_R_corr_v2']],
                 on=['day', 'symbol', 'fill_min', 'split'], how='left')
    n_missing = int(m.net_R_corr_v2.isna().sum())
    if n_missing:
        log(f'[WARNING] {n_missing} base fills have no cell_1457_features.csv match (net_R_corr_v2 NaN) '
            f'-- excluded from the rebuild')
    return m[m.net_R_corr_v2.notna()].reset_index(drop=True)


def decompose_base_cost(row):
    """(exit_half_dollars, is_stop_base): split the round-2 corrected total cost ($ at the base R) into
    the stop-slip $ (holdout-mean bps x base exit price, only if the base exit was a stop) and the
    remainder ("exit half-spread"), which PREREG says stays unchanged in $ under the re-walk."""
    cost_v2_dollars = (row.raw_R - row.net_R_corr_v2) * row.R
    is_stop_base = row.why in STOP_WHY
    bps = HOLDOUT_BPS[row.split]
    stop_slip_base = (bps / 1e4) * row.exit_price if is_stop_base else 0.0
    return cost_v2_dollars - stop_slip_base, is_stop_base


def rewalk_one(row, bars_by_symbol):
    """Re-walk one base fill under the 2.5 % R floor. Returns a result dict, or None if unsimulable
    (no bars for the symbol-day, or nothing left in the path past the fill bar)."""
    stop_new = min(row.stop, row.fill * 0.975)
    R_new = row.fill - stop_new
    target_new = row.fill + 2 * R_new
    unchanged = np.isclose(stop_new, row.stop)

    exit_half_dollars, _ = decompose_base_cost(row)

    if unchanged:
        exit_m_new, exit_price_new, why_new = row.exit_m, row.exit_price, row.why
        raw_new = row.raw_R
    else:
        bars = bars_by_symbol.get(row.symbol)
        if bars is None or bars.empty:
            return None
        fm = int(row.fill_min)  # floor to the integer ET minute the fill bar keys on
        fill_bar = bars[bars.m == fm]
        if len(fill_bar) and fill_bar.iloc[0].l <= stop_new:
            exit_m_new, exit_price_new, why_new = fm, stop_new, 'stop'
        else:
            path = bars[bars.m > fm]
            if path.empty:
                return None
            exit_m_new, exit_price_new, why_new = sr.walk_path(row.fill, stop_new, target_new, path)
        raw_new = (exit_price_new - row.fill) / R_new

    is_stop_new = why_new in STOP_WHY
    bps = HOLDOUT_BPS[row.split]
    stop_slip_new = (bps / 1e4) * exit_price_new if is_stop_new else 0.0
    cost_new_R = (exit_half_dollars + stop_slip_new) / R_new
    net_new = raw_new - cost_new_R

    return dict(day=row.day, symbol=row.symbol, fill_min=row.fill_min, split=row.split,
                stop_new=stop_new, R_new=R_new, exit_m_new=exit_m_new, exit_price_new=exit_price_new,
                why_new=why_new, raw_new=raw_new, cost_new=cost_new_R, net_new=net_new,
                delta_vs_base=net_new - row.net_R_corr_v2, changed=(not unchanged), why_base=row.why)


def main():
    m = load_base()
    log(f'[run] {len(m)} base fills loaded (after 1457 merge)')

    con = sqlite3.connect(sr.CACHE_DB_URI, uri=True)
    sipcon = sqlite3.connect(ca.BARS_SIP_URI, uri=True)

    rows, n_nobars, counts = [], 0, {}
    days = sorted(m.day.unique())
    for di, day in enumerate(days):
        g = m[m.day == day]
        syms = sorted(g.symbol.unique())
        bars_by_symbol = ca.load_day_bars(con, day, syms, sipcon=sipcon, counts=counts)
        for r in g.itertuples():
            res = rewalk_one(r, bars_by_symbol)
            if res is None:
                n_nobars += 1
                continue
            rows.append(res)
        if (di + 1) % 50 == 0 or di == len(days) - 1:
            log(f'[run] day {di + 1}/{len(days)} ({day}) | rows so far {len(rows)}')

    con.close()
    sipcon.close()

    if n_nobars:
        log(f'[WARNING] {n_nobars} fills had no re-walkable bars past the fill minute -- dropped')

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    log(f'[run] wrote {len(out)} rows to {OUT_CSV}')
    log(f'[run] bar source counts: {counts}')
    log(f"[run] changed (R floor bound) rows: {int(out.changed.sum())} / {len(out)}")


if __name__ == '__main__':
    main()
