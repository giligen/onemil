"""Cell 1,684 build pipeline. Stages run independently so progress survives a restart.

Stage candidates: daily-bar admission for all 4 ideas, both windows -> research/orb_freq/1684_reads.csv
  (one row per (window,symbol,date,idea) candidate) and a (day,symbol) list for the backfill wrapper.
Stage backfill:   check research/bf_zero/bars_sip.db coverage of the fresh-path candidates (idea10/11
  both windows, idea1/2 out-of-regime), write missing pairs, call the scratchpad backfill wrapper.
Stage features:   idea1/idea2 in-regime = filter+reuse the existing wide CSVs (no builder run).
                  idea10/idea11 in-regime + all-4 out-of-regime = run study_orb_features.main() with
                  patched seams (load_broad_universe/load_daily_bars_frame/load_spy_intraday/
                  Database.get_intraday_bars_bulk), one run per window.
Stage pipeline:   run study_orb_pipeline_static_lock.py per (pool,window) via ORB_BT_FEATURES_CSV/
                  ORB_BT_BOOK_OUT/ORB_BT_BARS_DB/ORB_BT_DAILY_SOURCE/ORB_CATALYST_VETO=0.
"""
import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/orb_freq')
sys.path.insert(0, '/home/ec2-user/onemil')
from pools_1684_lib import (ROOT, OUT, CACHE_DB, BARS_SIP, WIDE_CSVS, PRICE_MIN, PRICE_MAX,
                             PREV_VOL_MIN, PROD_GAP_MIN, IN_REGIME, OUT_REGIME, load_daily,
                             band_ok, is_production, log)

EQUS_2024H2 = ROOT / 'data/research/databento/equs_daily_2024H2.parquet'


def flag_ideas(df):
    """Add idea1_pre/idea2/idea10/idea11 boolean admission columns to a daily-bars frame that
    already has open,prev_close,prev_high,prev_volume,gap_pct,gap_vs_high_pct. idea1_pre is the
    gap-only prefilter; the range>=5%-by-09:35 condition is applied later once features exist."""
    df = df.copy()
    df['prev_day_gap_pct'] = df.groupby('symbol')['gap_pct'].shift(1)
    band = band_ok(df)
    prod = is_production(df)
    df['band'] = band
    df['is_prod'] = prod
    df['idea1_pre'] = band & (df['gap_pct'] >= 3.0) & (df['gap_pct'] < 5.0) & ~prod
    df['idea2'] = band & (df['gap_vs_high_pct'] >= 3.0) & ~prod
    df['idea10'] = band & (df['gap_pct'] <= -5.0) & ~prod
    df['idea11'] = band & (df['prev_day_gap_pct'] >= 10.0) & ~prod
    return df


def stage_candidates():
    """Daily-bar pass, both windows. Writes 1684_reads.csv (long: window,idea,symbol,date,gap_pct,
    gap_vs_high_pct,prev_day_gap_pct,overlap_prod_raw) and a fresh-path candidate (day,symbol) csv
    per window for the backfill/feature stages."""
    log.info('=== stage candidates: in-regime (cache.db) ===')
    sym_all = pd.read_sql_query("SELECT DISTINCT symbol FROM daily_bars", sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True))['symbol'].tolist()
    log.info('cache.db universe: %d symbols', len(sym_all))
    din = load_daily(symbols=None, start='2024-09-01', end=IN_REGIME[1])
    din = flag_ideas(din)
    din_win = din[(din.bar_date >= IN_REGIME[0]) & (din.bar_date <= IN_REGIME[1])].copy()
    din_win['window'] = 'in_regime'
    log.info('in-regime rows in window: %d; idea1_pre=%d idea2=%d idea10=%d idea11=%d prod=%d',
              len(din_win), din_win.idea1_pre.sum(), din_win.idea2.sum(), din_win.idea10.sum(),
              din_win.idea11.sum(), din_win.is_prod.sum())

    log.info('=== stage candidates: out-of-regime (EQUS 2024H2 + cache.db lookback stub) ===')
    equs = pd.read_parquet(EQUS_2024H2, columns=['bar_date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    equs['symbol'] = equs['symbol'].str.replace(r'\+$', '.WS', regex=True)
    con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True)
    stub = pd.read_sql_query("SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars WHERE bar_date BETWEEN '2024-06-03' AND '2024-06-30'", con)
    con.close()
    dout = pd.concat([stub, equs], ignore_index=True)
    dout['bar_date'] = dout['bar_date'].astype(str)
    dout = dout.drop_duplicates(['symbol', 'bar_date']).sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    g = dout.groupby('symbol')
    dout['prev_close'] = g['close'].shift(1)
    dout['prev_high'] = g['high'].shift(1)
    dout['prev_volume'] = g['volume'].shift(1)
    dout['gap_pct'] = (dout['open'] - dout['prev_close']) / dout['prev_close'] * 100
    dout['gap_vs_high_pct'] = (dout['open'] - dout['prev_high']) / dout['prev_high'] * 100
    dout = flag_ideas(dout)
    dout_win = dout[(dout.bar_date >= OUT_REGIME[0]) & (dout.bar_date <= OUT_REGIME[1])].copy()
    dout_win['window'] = 'out_regime'
    log.info('out-of-regime rows in window: %d; idea1_pre=%d idea2=%d idea10=%d idea11=%d prod=%d',
              len(dout_win), dout_win.idea1_pre.sum(), dout_win.idea2.sum(), dout_win.idea10.sum(),
              dout_win.idea11.sum(), dout_win.is_prod.sum())

    cols = ['window', 'symbol', 'bar_date', 'open', 'prev_close', 'prev_high', 'prev_volume',
            'gap_pct', 'gap_vs_high_pct', 'prev_day_gap_pct', 'idea1_pre', 'idea2', 'idea10', 'idea11', 'is_prod']
    any_idea = lambda d: d.idea1_pre | d.idea2 | d.idea10 | d.idea11  # noqa: E731
    both = pd.concat([din_win.loc[any_idea(din_win), cols], dout_win.loc[any_idea(dout_win), cols]],
                      ignore_index=True)
    both.to_csv(OUT / '1684_reads.csv', index=False)
    log.info('wrote %s rows=%d (candidate rows only, any idea flag true)', OUT / '1684_reads.csv', len(both))

    # Fresh-path candidate (day,symbol) lists for the backfill+feature stages.
    fresh_in = din_win[din_win.idea10 | din_win.idea11][['bar_date', 'symbol']].drop_duplicates()
    fresh_in.columns = ['day', 'symbol']
    fresh_in.to_csv(SCRATCH_fresh_in, index=False)
    fresh_out = dout_win[dout_win.idea1_pre | dout_win.idea2 | dout_win.idea10 | dout_win.idea11][['bar_date', 'symbol']].drop_duplicates()
    fresh_out.columns = ['day', 'symbol']
    fresh_out.to_csv(SCRATCH_fresh_out, index=False)
    log.info('fresh-path candidates: in-regime(idea10/11)=%d pairs, out-regime(all 4)=%d pairs',
              len(fresh_in), len(fresh_out))


SCRATCH_fresh_in = Path('/tmp/orb1684/fresh_in_candidates.csv')
SCRATCH_fresh_out = Path('/tmp/orb1684/fresh_out_candidates.csv')
Path('/tmp/orb1684').mkdir(parents=True, exist_ok=True)


def stage_backfill_check():
    """Report how many fresh-path (day,symbol) pairs are missing from bars_sip.db. Does not fetch."""
    con = sqlite3.connect(f'file:{BARS_SIP}?mode=ro', uri=True)
    have = pd.read_sql_query("SELECT DISTINCT symbol, day FROM bars", con)
    con.close()
    have_set = set(map(tuple, have[['symbol', 'day']].values))
    for name, path in [('in-regime idea10/11', SCRATCH_fresh_in), ('out-regime all-4', SCRATCH_fresh_out)]:
        c = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
        want = set(map(tuple, c[['symbol', 'day']].values))
        missing = want - have_set
        log.info('%s: want=%d have=%d missing=%d (%.1f%%)', name, len(want), len(want & have_set),
                  len(missing), 100 * len(missing) / max(1, len(want)))
        pd.DataFrame(list(missing), columns=['symbol', 'day']).sort_values(['day', 'symbol']).to_csv(
            f'/tmp/orb1684/missing_{name.split()[0]}.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['candidates', 'backfill_check'])
    a = ap.parse_args()
    if a.stage == 'candidates':
        stage_candidates()
    elif a.stage == 'backfill_check':
        stage_backfill_check()
