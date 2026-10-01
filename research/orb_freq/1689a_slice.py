"""Cell 1,689a -- recovery sub-pools 19'/19/20: gap>=5%, $3-30, prior-day volume 100K-500K (the
population production's absolute 500K floor drops; RESULT_1685.md found NO existing feature build
covers it in either window -- built fresh here, per the owner's direct instruction).

Pools (own selection chain each -- study_orb_pipeline_static_lock.py run SEPARATELY per pool so its
ranking/8-slot cap competes within that pool's own population only, cell 1,684/1,685 convention,
NOT one run on the slice with F1/F2 masks applied after):
  19' = the whole slice, no further gate.
  19  = slice x F1 relative volume at 09:35>=3x (frozen 1,685 TRAIN(2025) profile, not refit).
  20  = slice x F2 pre-market dollar volume>=$5M (appended 04:00-09:30 ET bars; coverage stated).

See pools_1689a_lib.py for constants/paths and the two daily-bar panel loaders (cache.db + databento
EQUS.SUMMARY cross-check, delisted included). See 1689a_features.py for the study_orb_features.py
loader-seam build (run once per window, after backfill).

Usage: python3 research/orb_freq/1689a_slice.py --stage {candidates,poolsplit,pipeline,score,tercile,all}
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pools_1689a_lib import (  # noqa: E402
    ROOT, OUT, POOLDIR, CACHE_DB, BARS_SIP, PRICE_MIN, PRICE_MAX, REC_VOL_LO, REC_VOL_HI,
    PROD_GAP_MIN, FETCH_GAP_MIN, F1_FROZEN_PROFILE, F1_MULT, F2_PREMARKET_USD, R_USD,
    IN_REGIME, OUT_REGIME, log, load_in_regime_panel, load_out_regime_panel,
)

FEAT_DIR = {'in_regime': OUT / 'out_1689a_in_regime', 'out_regime': OUT / 'out_1689a_out_regime'}
DAILY_SRC = {'in_regime': OUT / 'daily_source_1689a_in_regime.parquet',
             'out_regime': OUT / 'daily_source_1689a_out_regime.parquet'}
POOLS = ['19PRIME', '19', '20']


# ===================================================================== stage: candidates ===========
def stage_candidates():
    rows = []
    for win, (lo, hi), loader in (('in_regime', IN_REGIME, load_in_regime_panel),
                                   ('out_regime', OUT_REGIME, load_out_regime_panel)):
        panel, xcheck_syms = loader()
        d = panel[(panel.bar_date >= lo) & (panel.bar_date <= hi)].dropna(subset=['prev_close', 'prev_volume']).copy()
        d['price_ok'] = (d['open'] >= PRICE_MIN) & (d['open'] <= PRICE_MAX)
        d['vol_ok'] = (d['prev_volume'] >= REC_VOL_LO) & (d['prev_volume'] < REC_VOL_HI)
        sl = d[d['price_ok'] & d['vol_ok'] & (d['gap_pct_daily'] >= FETCH_GAP_MIN)].copy()
        sl['window'] = win
        sl['in_xcheck_only'] = sl['symbol'].isin(xcheck_syms)
        sl['buffer_zone'] = sl['gap_pct_daily'] < PROD_GAP_MIN
        n_days = d['bar_date'].nunique()
        log.info('%s: %d candidate rows (gap>=%.1f%% daily-open screen, $3-30, prevVol[100K,500K)) '
                  'over %d calendar days (%.2f/day); %d (%.1f%%) in the [4,5)%% buffer zone; '
                  '%d distinct symbols reachable ONLY via databento', win, len(sl), FETCH_GAP_MIN, n_days,
                  len(sl) / n_days if n_days else float('nan'), int(sl['buffer_zone'].sum()),
                  100 * sl['buffer_zone'].mean() if len(sl) else 0.0, sl.loc[sl['in_xcheck_only'], 'symbol'].nunique())
        per_day = sl.groupby('bar_date').size()
        log.info('%s per-day slice size: mean=%.2f median=%.1f p90=%.1f max=%d (%d days with >=1 row of %d total)',
                  win, per_day.mean(), per_day.median(), per_day.quantile(0.9), per_day.max(), len(per_day), n_days)
        rows.append(sl[['symbol', 'bar_date', 'window', 'open', 'prev_close', 'prev_volume', 'gap_pct_daily',
                         'in_xcheck_only', 'buffer_zone']])
    cand = pd.concat(rows, ignore_index=True).rename(columns={'bar_date': 'day'})
    cand.to_csv(OUT / '1689a_candidates.csv', index=False)
    cand[['day', 'symbol']].drop_duplicates().to_csv(OUT / '1689a_fetch_list.csv', index=False)
    log.info('wrote %s rows=%d; %s distinct symbol-days=%d', OUT / '1689a_candidates.csv', len(cand),
             OUT / '1689a_fetch_list.csv', cand[['day', 'symbol']].drop_duplicates().shape[0])


# ===================================================================== stage: poolsplit ============
def _premarket_dollar_volume(pairs):
    """sum(volume*typical_price) over each (symbol,day)'s 04:00-09:30 ET bars in bars_sip.db,
    typical_price=(h+l+c)/3 (study_orb_features.py's own 5-min VWAP convention). Temp-table join
    (bars' own (symbol,day) primary-key prefix) -- same trick as 1684_features.py seam 4, never a
    day-range scan of the 133M-row bars table."""
    import sqlite3
    if not pairs:
        return pd.DataFrame(columns=['symbol', 'day', 'premarket_usd', 'premarket_bars'])
    con = sqlite3.connect(f'file:{BARS_SIP}?mode=ro', uri=True)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS temp.want")
    cur.execute("CREATE TEMP TABLE want (symbol TEXT, day TEXT)")
    cur.executemany("INSERT INTO want VALUES (?, ?)", pairs)
    cur.execute("CREATE INDEX temp.idx_want ON want(symbol, day)")
    cur.execute("SELECT b.symbol, b.day, b.t, b.o, b.h, b.l, b.c, b.v FROM bars b "
                "JOIN want w ON b.symbol = w.symbol AND b.day = w.day")
    df = pd.DataFrame(cur.fetchall(), columns=['symbol', 'day', 't', 'o', 'h', 'l', 'c', 'v'])
    cur.execute("DROP TABLE want")
    con.close()
    if df.empty:
        return pd.DataFrame(columns=['symbol', 'day', 'premarket_usd', 'premarket_bars'])
    ts = pd.to_datetime(df['t'], utc=True).dt.tz_convert('America/New_York')
    premkt = df[(ts.dt.hour * 60 + ts.dt.minute >= 4 * 60) & (ts.dt.hour * 60 + ts.dt.minute < 9 * 60 + 30)].copy()
    typ = (premkt['h'] + premkt['l'] + premkt['c']) / 3.0
    premkt['usd'] = typ * premkt['v']
    g = premkt.groupby(['symbol', 'day']).agg(premarket_usd=('usd', 'sum'), premarket_bars=('usd', 'size'))
    return g.reset_index()


def stage_poolsplit():
    cand = pd.read_csv(OUT / '1689a_candidates.csv', keep_default_na=False, na_values=[''])
    for win in ('in_regime', 'out_regime'):
        fdir = FEAT_DIR[win]
        cands_files = sorted(p for p in fdir.glob('orb_features_*.csv') if 'corrmatrix' not in p.name) if fdir.exists() else []
        if not cands_files:
            log.warning('%s: no orb_features_*.csv under %s -- run 1689a_features.py first, skipping', win, fdir)
            continue
        feat = pd.read_csv(cands_files[-1], keep_default_na=False, na_values=[''])
        n_candidates = (cand['window'] == win).sum()
        log.info('%s: features built for %d/%d candidate rows (%.1f%% coverage -- minute-bar fetch '
                  'success rate)', win, len(feat), n_candidates, 100 * len(feat) / n_candidates if n_candidates else 0.0)
        final = feat[feat['gap_pct'] >= PROD_GAP_MIN].copy()
        log.info('%s: %d/%d feature rows clear the TRUE (minute-bar) gap_pct>=5.0%% cut -- this is pool 19\'',
                  win, len(final), len(feat))
        # F1: relative volume at 09:35, frozen 1,685 TRAIN(2025) profile
        final['f1_raw'] = final['range_total_volume'] / final['avg_daily_volume_20d'].replace(0, np.nan)
        final['f1'] = (final['f1_raw'] / F1_FROZEN_PROFILE) >= F1_MULT
        # F2: pre-market dollar volume from the appended 04:00-09:30 ET bars
        pairs = list(zip(final['symbol'], final['date']))
        pm = _premarket_dollar_volume(pairs)
        final = final.merge(pm, how='left', left_on=['symbol', 'date'], right_on=['symbol', 'day'])
        final['premarket_usd'] = final['premarket_usd'].fillna(0.0)
        final['premarket_bars'] = final['premarket_bars'].fillna(0)
        n_cov = (final['premarket_bars'].fillna(0) > 0).sum()
        log.info('%s: premarket (04:00-09:30 ET) bar coverage %d/%d rows (%.1f%%)', win, n_cov, len(final),
                  100 * n_cov / len(final) if len(final) else 0.0)
        final['f2'] = final['premarket_usd'] >= F2_PREMARKET_USD
        log.info('%s pool sizes: 19\'=%d  19(F1)=%d  20(F2)=%d', win, len(final), int(final['f1'].sum()), int(final['f2'].sum()))

        feat_cols = [c for c in feat.columns if c not in ('day',)]
        for pool, mask in (('19PRIME', pd.Series(True, index=final.index)), ('19', final['f1'].fillna(False)),
                            ('20', final['f2'].fillna(False))):
            sub = final.loc[mask, feat_cols]
            fp = POOLDIR / f'{pool}_{win}_features.csv'
            sub.to_csv(fp, index=False)
            log.info('%s/%s: %d rows -> %s', pool, win, len(sub), fp)


# ===================================================================== stage: pipeline ==============
def stage_pipeline():
    for win in ('in_regime', 'out_regime'):
        for pool in POOLS:
            fp = POOLDIR / f'{pool}_{win}_features.csv'
            if not fp.exists():
                log.warning('%s/%s: %s missing -- run poolsplit first, skipping', pool, win, fp)
                continue
            n = sum(1 for _ in open(fp)) - 1
            outp = POOLDIR / f'{pool}_{win}_true.csv'
            if n <= 0:
                log.warning('%s/%s: 0 candidates -- skipping pipeline run', pool, win)
                continue
            env = dict(os.environ, ORB_BT_FEATURES_CSV=str(fp), ORB_BT_BOOK_OUT=str(outp),
                       ORB_BT_BARS_DB=str(BARS_SIP), ORB_BT_DAILY_SOURCE=str(DAILY_SRC[win]),
                       ORB_CATALYST_VETO='0', PYTHONPATH=str(ROOT))
            log_path = POOLDIR / f'{pool}_{win}_true.log'
            log.info('running pipeline for %s/%s (n=%d) -> %s', pool, win, n, outp)
            with open(log_path, 'w') as lf:
                rc = subprocess.run(['nice', '-n', '10', 'python3', 'study_orb_pipeline_static_lock.py'],
                                     cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            log.info('%s/%s pipeline rc=%d', pool, win, rc)


# ===================================================================== stage: score =================
def _score1684():
    import importlib.util
    spec = importlib.util.spec_from_file_location('score1684', OUT / '1684_score.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_score():
    from datetime import date
    s = _score1684()
    IN_LO, IN_HI = date(2025, 1, 1), date(2026, 9, 26)
    OUT_LO, OUT_HI = date(2024, 7, 1), date(2024, 12, 31)

    prod_in = s.load_book(OUT / 'fastpath/prod_true.csv')
    prod_out = s.load_book(ROOT / 'research/orb_2024/book_1415_liveexit.csv')
    print(f"production in_regime  n={0 if prod_in is None else len(prod_in)}")
    print(f"production out_regime n={0 if prod_out is None else len(prod_out)}")

    all_rows = []
    for pool in POOLS:
        for win, prod, (lo, hi) in (('in_regime', prod_in, (IN_LO, IN_HI)), ('out_regime', prod_out, (OUT_LO, OUT_HI))):
            path = POOLDIR / f'{pool}_{win}_true.csv'
            df = s.load_book(path)
            n = 0 if df is None else len(df)
            print(f"\n--- POOL {pool} / {win} (n={n}) ---")
            if df is not None and len(df):
                tag = df.copy()
                tag['window'] = win
                tag['pool'] = pool
                all_rows.append(tag[['window', 'pool', 'date', 'symbol', 'entry_price', '_sized_pnl', 'R', '_composite']])
                if win == 'in_regime':
                    for yr in (2025, 2026):
                        print(s.fmt(s.stats(df[df.date.dt.year == yr], f'{pool}/{win}/{yr}')))
                st = s.stats(df, f'{pool}/{win}/FULL')
                print(s.fmt(st))
                union, addon, raw_overlap = s.union_book(prod, df)
                print(f"  raw_overlap_with_prod={raw_overlap:.1%} added_after_excl={len(addon)} "
                      f"(frequency gain = {len(addon)} fills)")
                print(s.fmt(s.stats(union, f'{pool}/{win}/UNION')))
                print(s.cadence_report(union, f'{pool}/{win}/UNION', lo, hi))
                print(s.cadence_report(prod, f'{win}/prod-alone', lo, hi))
                # shared worst day
                if len(df):
                    worst_pool_day = df.groupby(df.date.dt.date)['R'].sum().idxmin()
                    prod_day_r = prod[prod.date.dt.date == worst_pool_day]['R'].sum() if prod is not None else 0.0
                    print(f"  pool's worst day {worst_pool_day}: pool R sum={df[df.date.dt.date==worst_pool_day]['R'].sum():+.2f} "
                          f"production R sum that SAME day={prod_day_r:+.2f}")
            else:
                print(f"{pool}/{win}: n=0 (no fills)")

    if all_rows:
        out_df = pd.concat(all_rows, ignore_index=True)
        out_df.to_csv(OUT / '1689a_pool_books.csv', index=False)
        print(f"\nwrote {OUT / '1689a_pool_books.csv'} rows={len(out_df)}")

    print("\n=== PASS BAR (own mean R>=+0.05 & dc_t>=2.0 in-regime, >=0 out-of-regime (not p30-neg), exTop5>0) ===")
    for pool in POOLS:
        din = s.load_book(POOLDIR / f'{pool}_in_regime_true.csv')
        dout = s.load_book(POOLDIR / f'{pool}_out_regime_true.csv')
        sin = s.stats(din, f'{pool}/in')
        sout = s.stats(dout, f'{pool}/out')
        ok_in = sin.get('n', 0) > 0 and sin.get('mean_r', -9) >= 0.05 and (sin.get('dc_t') or -9) >= 2.0 \
            and (sin.get('ex_top5') or -9) > 0
        ok_out = sout.get('n', 0) == 0 or sout.get('mean_r', -9) >= 0.0
        verdict = 'PASS' if (ok_in and ok_out) else 'FAIL'
        print(f"{pool}: in n={sin.get('n',0)} meanR={sin.get('mean_r', float('nan')):+.3f} "
              f"dc_t={sin.get('dc_t', float('nan')):.2f} exTop5={sin.get('ex_top5', float('nan')):+.3f} | "
              f"out n={sout.get('n',0)} meanR={sout.get('mean_r', float('nan')):+.3f} -> {verdict}")


# ===================================================================== stage: tercile ===============
def stage_tercile():
    """Repeat RESULT_1685.md's production-book-by-prior-volume-tercile read, same method, to confirm
    the exact numbers already reported there (in: low n=161 meanR=+0.100, mid n=160 meanR=+0.119,
    high n=161 meanR=+0.098; out: low n=20 meanR=+0.272, mid n=19 meanR=-0.031, high n=20 meanR=-0.152)."""
    s = _score1684()
    panel_in, _ = load_in_regime_panel()
    panel_out, _ = load_out_regime_panel()
    prod_books = {'in_regime': (OUT / 'fastpath/prod_true.csv', panel_in),
                  'out_regime': (ROOT / 'research/orb_2024/book_1415_liveexit.csv', panel_out)}
    for win, (path, panel) in prod_books.items():
        book = s.load_book(path)
        if book is None:
            log.warning('tercile %s: production book missing at %s', win, path)
            continue
        d = panel[['symbol', 'bar_date', 'prev_volume']].rename(columns={'bar_date': 'date'})
        d['date'] = pd.to_datetime(d['date'])
        m = book.merge(d, on=['symbol', 'date'], how='left')
        n_nomatch = m['prev_volume'].isna().sum()
        m = m.dropna(subset=['prev_volume'])
        try:
            m['tercile'] = pd.qcut(m['prev_volume'], 3, labels=['low', 'mid', 'high'], duplicates='drop')
            tg = m.groupby('tercile', observed=True)['R'].agg(['count', 'mean', 'std'])
            print(f"\n{win} tercile split (n_nomatch={n_nomatch}/{len(book)}):\n{tg.to_string()}")
        except ValueError as e:
            log.warning('%s tercile split failed (%s)', win, e)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                     choices=['candidates', 'poolsplit', 'pipeline', 'score', 'tercile', 'all'])
    a = ap.parse_args()
    if a.stage in ('candidates', 'all'):
        stage_candidates()
    if a.stage in ('poolsplit', 'all'):
        stage_poolsplit()
    if a.stage in ('pipeline', 'all'):
        stage_pipeline()
    if a.stage in ('score', 'all'):
        stage_score()
    if a.stage in ('tercile', 'all'):
        stage_tercile()
