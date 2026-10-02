#!/usr/bin/env python3
"""Cell 1,693b: the missing 2-3% gap band (band A) from PREREG_1693's own escape clause ("Both
original missing seeds ... require NEW minute bars via the bars_sip.db appender for a candidate
universe that does not exist in any CSV on disk ... infeasible inside this cell's call budget").
Built fresh here: Alpaca SIP minute bars are a FREE pull (not the Databento spend the owner's
"convince first" rule covers), so the appender ran via scratchpad/backfill_fill_days.py.

Six sub-pools AF1-AF6 = band A (gap [2,3)%, $3-30, prior-day volume >=500K, excluding production
>=5% and the 3-5% seed by construction -- disjoint gap bands, re-checked in build_candidates.py)
x {F1 relative volume at 09:35, F2 pre-market dollar volume, F3 above-VWAP+top-half range, F4
within 5% of the 52-week high, F5 prior-day range>=1.5xATR14, F6 day-2 of a >=10% gapper} -- SAME
feature letters and filter mechanisms cell 1685_subpools.py used for bands B/C (F1/F3/F4/F5/F6)
and cell 1693's own F2 (premkt_dollar_vol), just applied to this freshly-fetched population.

Upstream (scratchpad, experiment caches per CLAUDE.md -- never under research/orb_freq):
  scratchpad/band23/build_candidates.py   - daily-bar fetch screen + true-band count
  scratchpad/backfill_fill_days.py        - the appender wrapper (bars_sip.db, APPEND-only)
  scratchpad/band23/features_build.py     - study_orb_features.py loader-seam build (per window)

This script (the deliverable):
  --stage pools    : F1-F6 admission (TRAIN2025-frozen profiles for F1/F2, same x3 rule as F1;
                     direct structural conditions for F3/F4/F5/F6, same as 1685_subpools.py),
                     true-up on the minute-bar gap_pct, per-pool features CSVs.
  --stage pipeline : study_orb_pipeline_static_lock.py per (pool,window) at the LIVE config
                     (ORB_CATALYST_VETO=0) -- each pool's OWN selection chain (ranking/8-slot cap).
  --stage score    : reuse research/orb_freq/1693_pool_exits.py's CachedStore/reconstruct_fill/
                     EXITS/build_per_fill_table/score_table/classify_pool/cb VERBATIM (imported
                     read-only via importlib, never edited) for the 12-exit grid x both directions,
                     classification, and the union with production.

Usage: python3 research/orb_freq/1693b_band23.py --stage {pools,pipeline,score,all}
"""
import argparse
import importlib.util
import logging
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_freq'
SCRATCH = Path('/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/band23')
BARS_SIP = ROOT / 'research/bf_zero/bars_sip.db'
CACHE_DB = ROOT / 'data/cache.db'
XNAS_2023_2024H1 = ROOT / 'data/research/databento/xnas_daily_2023_2024H1.parquet'

LOG_FILE = OUT / '1693b_band23.log'
READS_CSV = OUT / '1693b_reads.csv'
POOL_BOOKS_CSV = OUT / '1693b_pool_books.csv'
RESULT_MD = OUT / 'RESULT_1693b.md'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                     handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode='a')],
                     force=True)
log = logging.getLogger('band23')

WINDOWS_DAILY = {'in_regime': ('2025-01-01', '2026-09-26'), 'out_regime': ('2024-07-01', '2024-12-31')}
DAILY_SRC = {w: SCRATCH / f'daily_source_band23_{w}.parquet' for w in WINDOWS_DAILY}
GAP_LO, GAP_HI = 2.0, 3.0
TRAIN_YEAR = 2025
POOLS = ['AF1', 'AF2', 'AF3', 'AF4', 'AF5', 'AF6']


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relpath))
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = old_argv
    return mod


# =================================================================== daily lookback (F4/F5/F6) =====
def _ro(p, retries=10, wait_s=30):
    """Read-only connect with a 30s backoff up to 10 retries on 'database is locked' -- cache.db is
    written by the live trading service through 20:00 UTC; never kill it, back off instead. bars_sip.db
    reads use the same helper and the same courtesy even though this cell is the only writer today."""
    last_err = None
    for attempt in range(retries):
        try:
            con = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
            con.execute('SELECT 1')
            return con
        except sqlite3.OperationalError as e:
            last_err = e
            if 'locked' not in str(e).lower():
                raise
            log.warning('%s locked, retry %d/%d in %ds: %s', p, attempt + 1, retries, wait_s, e)
            time.sleep(wait_s)
    raise last_err


def _clean_symbol(s):
    return s.astype(str).str.replace(r'\+$', '.WS', regex=True)


def load_long_daily():
    """Full daily-bar panel for F4/F5/F6 causal lookback, ported read-only from 1685_subpools.py's
    own load_long_daily() (XNAS 2023-01..2024-06-02 prefix UNION cache.db 2024-06-03+, source of
    record on overlap) -- not imported live to avoid that module's import-time logging.basicConfig
    (force=True) side effect, which would hijack this script's OWN log file."""
    xnas = pd.read_parquet(XNAS_2023_2024H1, columns=['bar_date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    xnas['symbol'] = _clean_symbol(xnas['symbol'])
    con = _ro(CACHE_DB)
    cache = pd.read_sql_query("SELECT symbol,bar_date,open,high,low,close,volume FROM daily_bars", con)
    con.close()
    df = pd.concat([xnas[xnas.bar_date < '2024-06-03'], cache], ignore_index=True)
    df = df.drop_duplicates(['symbol', 'bar_date']).sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    log.info('long daily panel: %d rows, %d symbols, %s..%s', len(df), df.symbol.nunique(),
              df.bar_date.min(), df.bar_date.max())
    g = df.groupby('symbol')
    df['prev_close'] = g['close'].shift(1)
    df['prev_high'] = g['high'].shift(1)
    df['prev_low'] = g['low'].shift(1)
    df['gap_pct_daily'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
    g2 = df.groupby('symbol')
    df['prev_day_gap_pct'] = g2['gap_pct_daily'].shift(1)
    df['roll252_high'] = g2['high'].transform(lambda s: s.rolling(252, min_periods=200).max().shift(1))
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['prev_close']).abs(),
                    (df['low'] - df['prev_close']).abs()], axis=1).max(axis=1)
    df['atr14'] = tr.groupby(df['symbol']).transform(lambda s: s.rolling(14, min_periods=10).mean().shift(1))
    return df[['symbol', 'bar_date', 'prev_day_gap_pct', 'roll252_high', 'prev_high', 'prev_low', 'atr14', 'prev_close']]


# =================================================================== premarket $ volume (F2) =======
def bulk_premarket_usd(pairs):
    """Sum(typical_price*volume) over bars_sip.db minutes before 09:30 ET, bulk temp-table join --
    same trick as this cell's features_build.py seam 4, never a day-range scan of the 133M+-row
    bars table. `pairs`: iterable of (symbol, day) strings."""
    pairs = sorted(set(pairs))
    if not pairs:
        return pd.DataFrame(columns=['symbol', 'day', 'premkt_usd'])
    con = _ro(BARS_SIP)
    cur = con.cursor()
    cur.execute("CREATE TEMP TABLE want (symbol TEXT, day TEXT)")
    cur.executemany("INSERT INTO want VALUES (?, ?)", pairs)
    cur.execute("CREATE INDEX idx_want ON want(symbol, day)")
    q = ("SELECT b.symbol, b.day, b.t, b.h, b.l, b.c, b.v FROM bars b "
         "JOIN want w ON b.symbol = w.symbol AND b.day = w.day")
    df = pd.read_sql_query(q, con)
    con.close()
    log.info('premarket bulk query: %d (symbol,day) requested, %d bar rows returned', len(pairs), len(df))
    if df.empty:
        return pd.DataFrame(columns=['symbol', 'day', 'premkt_usd'])
    hh = df['t'].str.slice(11, 13).astype(int)
    mm = df['t'].str.slice(14, 16).astype(int)
    minute_et = ((hh - 4) % 24) * 60 + mm
    pre = df.loc[minute_et < 570].copy()
    pre['usd'] = (pre['h'] + pre['l'] + pre['c']) / 3.0 * pre['v']
    agg = pre.groupby(['symbol', 'day'], as_index=False)['usd'].sum().rename(columns={'usd': 'premkt_usd'})
    return agg


# =================================================================== stage: pools ==================
def _latest_features_csv(window):
    fdir = SCRATCH / f'out_{window}'
    cands = sorted(p for p in fdir.glob('orb_features_*.csv') if 'corrmatrix' not in p.name)
    if not cands:
        raise FileNotFoundError(f'no orb_features_*.csv under {fdir} -- run features_build.py first')
    return cands[-1]


PREMKT_COVERAGE_RAIL = 0.80  # CLAUDE.md availability rail: >=80% coverage else VOID


def stage_pools():
    long_daily = load_long_daily()
    manifest = []
    frames = []
    premkt_cov_by_window = {}
    for window in WINDOWS_DAILY:
        fp = _latest_features_csv(window)
        feat = pd.read_csv(fp, keep_default_na=False, na_values=[''])
        orig_cols = feat.columns.tolist()
        n_built = len(feat)
        feat = feat.merge(long_daily, left_on=['symbol', 'date'], right_on=['symbol', 'bar_date'], how='left')
        true_band = (feat['gap_pct'] >= GAP_LO) & (feat['gap_pct'] < GAP_HI)
        n_true = int(true_band.sum())
        log.info('%s: %d candidates with built features, %d true-up to band A [2,3)%% on the '
                  'minute-bar gap_pct (coverage vs fetch-list in build_candidates.log)', window, n_built, n_true)
        feat = feat.loc[true_band].copy()
        feat['window'] = window

        pm = bulk_premarket_usd(list(zip(feat['symbol'], feat['date'])))
        feat = feat.merge(pm, left_on=['symbol', 'date'], right_on=['symbol', 'day'], how='left')
        premkt_cov = float((feat['premkt_usd'].notna()).mean()) if len(feat) else float('nan')
        premkt_cov_by_window[window] = premkt_cov
        log.info('%s: pre-market (04:00-09:30 ET) bar coverage %d/%d (%.1f%%)', window,
                  int(feat['premkt_usd'].notna().sum()), len(feat), 100 * premkt_cov if len(feat) else 0.0)

        feat['f3'] = (feat['range_vwap_distance_pct'] > 0) & (feat['range_close_position'] >= 0.5)
        frames.append((window, feat, orig_cols))

    # F4 needs TODAY's price vs the 52w high; the features CSV carries no raw 'open' (admission
    # price was screened upstream in build_candidates.py's daily panel) -- use entry_price (the
    # book's own reconstructed first-bar/entry value, confirmed present in study_orb_features.py's
    # own output, same column every exit walker in this programme treats as ground truth) as the
    # price input, same convention CLAUDE.md requires ("the book's own value").
    all_feat = []
    for window, feat, orig_cols in frames:
        feat['f4'] = feat['roll252_high'].notna() & (feat['entry_price'] >= 0.95 * feat['roll252_high'])
        feat['f5'] = feat['atr14'].notna() & ((feat['prev_high'] - feat['prev_low']) >= 1.5 * feat['atr14'])
        feat['f6'] = feat['prev_day_gap_pct'] >= 10.0
        all_feat.append((window, feat, orig_cols))

    # TRAIN2025-frozen profiles for F1 (reuse 1685_subpools.py's ratio) and F2 (this cell's analog,
    # same x3 mechanism) -- fit ONCE on band A's own TRAIN2025 rows (both windows' true-band pool
    # concatenated, but only 2025 dates contribute), frozen, applied unchanged to VAL/OOS.
    base_all = pd.concat([f for _, f, _ in all_feat], ignore_index=True)
    yr = pd.to_datetime(base_all['date']).dt.year
    train_mask = yr == TRAIN_YEAR
    f1_raw_train = (base_all.loc[train_mask, 'range_total_volume'] / base_all.loc[train_mask, 'avg_daily_volume_20d']).replace([np.inf, -np.inf], np.nan)
    F1_PROFILE = f1_raw_train.median()
    f2_denom_train = base_all.loc[train_mask, 'avg_daily_volume_20d'] * base_all.loc[train_mask, 'prev_close']
    f2_raw_train = (base_all.loc[train_mask, 'premkt_usd'] / f2_denom_train).replace([np.inf, -np.inf], np.nan)
    F2_PROFILE = f2_raw_train.median()
    log.info('TRAIN2025 band-A frozen profiles: F1 median(range_total_volume/avg_daily_volume_20d)=%.6f (n=%d), '
              'F2 median(premkt_usd/(avg_daily_volume_20d*prev_close))=%.6f (n=%d)', F1_PROFILE,
              int(train_mask.sum()), F2_PROFILE, int(base_all.loc[train_mask, 'premkt_usd'].notna().sum()))

    combined_premkt_cov = np.nanmean(list(premkt_cov_by_window.values())) if premkt_cov_by_window else float('nan')
    af2_void = not (combined_premkt_cov >= PREMKT_COVERAGE_RAIL)
    if af2_void:
        log.error('AF2 VOID: combined pre-market coverage %.1f%% < %.0f%% rail (CLAUDE.md availability '
                   'rail) -- AF2 built with 0 candidates below, reported VOID in RESULT, never scored '
                   'as a normal pass/fail pool', 100 * combined_premkt_cov if combined_premkt_cov == combined_premkt_cov else 0.0,
                   100 * PREMKT_COVERAGE_RAIL)
    else:
        log.info('AF2 coverage rail OK: combined pre-market coverage %.1f%% >= %.0f%%',
                  100 * combined_premkt_cov, 100 * PREMKT_COVERAGE_RAIL)

    manifest = []
    for window, feat, orig_cols in all_feat:
        feat['f1_raw'] = feat['range_total_volume'] / feat['avg_daily_volume_20d']
        feat['f1'] = (feat['f1_raw'] / F1_PROFILE) >= 3.0
        feat['f2_raw'] = feat['premkt_usd'] / (feat['avg_daily_volume_20d'] * feat['prev_close'])
        feat['f2'] = (not af2_void) & feat['premkt_usd'].notna() & ((feat['f2_raw'] / F2_PROFILE) >= 3.0)
        for i, pool in enumerate(POOLS, 1):
            mask = feat[f'f{i}'].fillna(False)
            sub = feat.loc[mask, [c for c in orig_cols if c in feat.columns]]
            fp = SCRATCH / f'{pool}_{window}_features.csv'
            sub.to_csv(fp, index=False)
            manifest.append((pool, window, len(sub)))
            log.info('%s/%s: %d candidate rows -> %s', pool, window, len(sub), fp)
    man_df = pd.DataFrame(manifest, columns=['pool', 'window', 'n_candidates'])
    man_df.to_csv(SCRATCH / 'manifest.csv', index=False)
    (SCRATCH / 'premkt_coverage.txt').write_text(
        f"combined={combined_premkt_cov}\nvoid={af2_void}\n" +
        "\n".join(f"{w}={c}" for w, c in premkt_cov_by_window.items()))
    log.info('stage_pools DONE (AF2 void=%s)', af2_void)


# =================================================================== stage: pipeline ================
def stage_pipeline():
    manifest = pd.read_csv(SCRATCH / 'manifest.csv')
    for _, row in manifest.iterrows():
        pool, window, n = row['pool'], row['window'], row['n_candidates']
        fp = SCRATCH / f'{pool}_{window}_features.csv'
        outp = SCRATCH / f'{pool}_{window}_true.csv'
        if n == 0:
            log.warning('%s/%s: 0 candidates -- skipping pipeline run', pool, window)
            continue
        env = dict(os.environ, ORB_BT_FEATURES_CSV=str(fp), ORB_BT_BOOK_OUT=str(outp),
                   ORB_CATALYST_VETO='0', PYTHONPATH=str(ROOT),
                   ORB_BT_BARS_DB=str(BARS_SIP), ORB_BT_DAILY_SOURCE=str(DAILY_SRC[window]))
        log_path = SCRATCH / f'{pool}_{window}_true.log'
        log.info('running pipeline for %s/%s (n=%d) -> %s', pool, window, n, outp)
        with open(log_path, 'w') as lf:
            rc = subprocess.run(['nice', '-n', '10', 'python3', 'study_orb_pipeline_static_lock.py'],
                                 cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            log.error('%s/%s: pipeline FAILED rc=%d -- see %s for the traceback; this '
                      '(pool, window) book is MISSING, not legitimately empty, until '
                      'fixed', pool, window, rc, log_path)
        else:
            log.info('%s/%s pipeline rc=%d', pool, window, rc)


# =================================================================== stage: score ===================
def stage_score():
    c93 = _load_module('cell1693_band23', 'research/orb_freq/1693_pool_exits.py')
    # CRITICAL: c93.append_result_md() writes to c93's OWN module-level RESULT_MD global, which
    # defaults to the ORIGINAL cell 1,693's RESULT_1693.md -- redirect it to THIS cell's own file
    # before any append, or every call below would corrupt a sibling deliverable that is not ours
    # to touch (observed concurrently written during this session).
    c93.RESULT_MD = str(RESULT_MD)
    RESULT_MD.write_text(
        "# RESULT 1,693b -- band A (gap [2,3)%) sub-pools AF1-AF6, the missing seed from "
        "PREREG_1693's own escape clause\n\n"
        "PREREG: research/orb_freq/PREREG_1693.md (original, FROZEN). This cell builds the ONE "
        "seed that cell 1,693 explicitly declared NOT built (2-3% gap band x F1/F3/F4/F5/F6), plus "
        "F2 (pre-market dollar volume) on the same fresh population, since new minute bars had to "
        "be fetched for it anyway. Harness reused verbatim from research/orb_freq/1693_pool_exits.py "
        "(CachedStore/reconstruct_fill/EXITS/build_per_fill_table/score_table/classify_pool/cb), "
        "imported read-only via importlib -- never edited.\n\n"
        "## Band-A sub-pools (6) -- best exit per direction, classification\n")
    cov_txt = (SCRATCH / 'premkt_coverage.txt').read_text() if (SCRATCH / 'premkt_coverage.txt').exists() else ''
    af2_void = 'void=True' in cov_txt
    if af2_void:
        c93.append_result_md(f"- **AF2** [VOID] pre-market (04:00-09:30 ET) bar coverage below the "
                              f"{PREMKT_COVERAGE_RAIL:.0%} CLAUDE.md availability rail -- not scored as "
                              f"pass/fail, a claim about coverage, not about the feature ({cov_txt.strip()})\n")
    store = c93.CachedStore(str(BARS_SIP))

    def book_rows(fp):
        if not Path(fp).exists():
            return []
        try:
            b = pd.read_csv(fp, keep_default_na=False, na_values=[''])
        except pd.errors.EmptyDataError:
            # A pipeline run can legitimately veto 100% of a (pool, window)'s
            # picks with no refill (root-caused 2026-10-02, AF4/out_regime:
            # study_orb_pipeline_static_lock.py now writes a 0-row-but-headered
            # CSV for this case). A truly header-less/0-byte file reaching here
            # means the pipeline stage crashed before writing anything -- VOID
            # this book as n=0 rather than crash the whole score stage, but say
            # so loudly: stage_pipeline's own log/rc is the place to diagnose it.
            log.warning('%s: EMPTY/unreadable CSV (no columns) -- recording '
                        'n=0 for this book; check stage_pipeline rc for this '
                        '(pool, window) before trusting that as a real zero', fp)
            return []
        if 'entered' not in b.columns:
            log.warning('%s: 0 rows and no `entered` column -- recording n=0 '
                        '(legitimately empty book, not a read failure)', fp)
            return []
        b = b[b['entered'].astype(str).isin(['1', 'True', 'true'])]
        return list(zip(b['date'], b['symbol'], b['entry_price']))

    all_reads, all_cls, pool_books = [], [], []
    pf_by_pool = {}
    for pool in POOLS:
        if pool == 'AF2' and af2_void:
            log.warning('AF2 VOID (pre-market coverage rail) -- skipped in the scoring loop, see the '
                        'VOID line already written to RESULT_1693b.md')
            continue
        rows = []
        for window in WINDOWS_DAILY:
            rows += book_rows(SCRATCH / f'{pool}_{window}_true.csv')
        pf, recon = c93.build_per_fill_table(rows, store, pool)
        pf_by_pool[pool] = pf
        reads = c93.score_table(pf, pool)
        cls = c93.classify_pool(reads, pool)
        all_reads.append(reads)
        all_cls.append(cls)
        if len(pf):
            pb = pf.copy()
            pb['pool'] = pool
            pool_books.append(pb)
        c93.append_result_md(c93.pool_summary_line(cls, pool) + f" (n_fills reconstructed={recon['ok']}, dropped={sum(v for k, v in recon.items() if k != 'ok')})\n")
        log.info('%s scored: recon=%s', pool, recon)

    # production reference, reconstructed the SAME way (never trust a different code path for the baseline)
    prod_rows = book_rows(OUT / 'fastpath/prod_true.csv') + book_rows(ROOT / 'research/orb_2024/book_1415_liveexit.csv')
    pf_prod, recon_prod = c93.build_per_fill_table(prod_rows, store, 'production')
    log.info('production reconstructed for the union baseline: %s', recon_prod)

    reads_df = pd.concat(all_reads, ignore_index=True)
    cls_df = pd.concat(all_cls, ignore_index=True)
    reads_df.to_csv(READS_CSV, index=False)
    if pool_books:
        pd.concat(pool_books, ignore_index=True).to_csv(POOL_BOOKS_CSV, index=False)
    log.info('wrote %s (%d rows), %s', READS_CSV, len(reads_df), POOL_BOOKS_CSV)

    robust_pairs = cls_df[cls_df.classification == 'robust']
    regime_pairs = cls_df[cls_df.classification == 'regime_specific']
    c93.append_result_md("\n## Robust pairs (both directions confirm) -- band A\n")
    for r in robust_pairs.itertuples():
        c93.append_result_md(f"- {r.pool} x {r.exit}: TRAIN {r.train_mean_R:+.3f}R (t{r.train_t:.1f}, n{r.train_n}) / "
                              f"VAL {r.val_mean_R:+.3f}R (t{r.val_t:.1f}, n{r.val_n}) / OOS2024H2 {r.oos_mean_R:+.3f}R (n{r.oos_n})\n")
    if not len(robust_pairs):
        c93.append_result_md("- none\n")
    c93.append_result_md("\n## Regime-specific pairs (one direction only; reported, not shipped) -- band A\n")
    for r in regime_pairs.itertuples():
        c93.append_result_md(f"- {r.pool} x {r.exit}: TRAIN {r.train_mean_R:+.3f}R (t{r.train_t:.1f}, n{r.train_n}) / "
                              f"VAL {r.val_mean_R:+.3f}R (t{r.val_t:.1f}, n{r.val_n}) / OOS2024H2 {r.oos_mean_R:+.3f}R (n{r.oos_n})\n")
    if not len(regime_pairs):
        c93.append_result_md("- none\n")

    # --- union: production(E1) + band-A robust pairs -----------------------------------------
    prod_union = pf_prod[pf_prod['window'].notna()].copy()
    prod_union['union_R'] = prod_union['E1_production']
    prod_union['src'] = 'production'

    def pairs_to_fills(pairs_df):
        out = []
        for r in pairs_df.itertuples():
            pf = pf_by_pool.get(r.pool)
            if pf is None or not len(pf):
                continue
            sub = pf[pf['window'].notna()].copy()
            sub['union_R'] = sub[r.exit]
            sub['src'] = f'{r.pool}/{r.exit}'
            out.append(sub[['date', 'window', 'symbol', 'union_R', 'src']])
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['date', 'window', 'symbol', 'union_R', 'src'])

    robust_fills = pairs_to_fills(robust_pairs)
    base_cols = ['date', 'window', 'symbol', 'union_R', 'src']

    def dedup_union(frames):
        u = pd.concat([f[base_cols] for f in frames], ignore_index=True)
        before = len(u)
        u = u.drop_duplicates(subset=['date', 'symbol'], keep='first')
        log.info('union dedup: %d -> %d rows (%d overlap collapsed)', before, len(u), before - len(u))
        return u.dropna(subset=['union_R'])

    union_robust = dedup_union([prod_union, robust_fills])
    prod_alone = prod_union[base_cols].dropna(subset=['union_R'])

    def union_metrics(u, lo, hi, label):
        sub = u[(u['date'] >= lo) & (u['date'] <= hi)]
        vals, dts = sub['union_R'], sub['date']
        base = c93.reads_for_exit_series(vals, dts, lo, hi)
        trades = [{'date': d, 'r': v} for d, v in zip(dts, vals)]
        weekly = c93.cb.build_weekly_series(trades, lo, hi)
        weekly_r = [w[1] for w in weekly]
        cycles, strong_idx = c93.cb.compute_cycles(weekly, c93.STRONG_WEEK_R)
        c1 = c93.cb.score_c1(cycles, gap_median_thresh=3, gap_p90_thresh=6)
        c4 = c93.cb.score_c4(weekly, trades, green_thresh=0.55, green_margin=0.0, n_null=500)
        fpw = base['fills_per_week']
        return dict(label=label, **base, weekly_p10_dollars=base['weekly_p10_R'] * c93.FIXED_RISK_DOLLARS,
                    weekly_p10_R_per_fill=(base['weekly_p10_R'] / fpw if fpw else np.nan),
                    strong_week_gap_median=c1['median'], strong_week_gap_p90=c1['p90'],
                    green_share=c4['green'], green_null_mean=c4['null'])

    TR_LO, VAL_HI = c93.TRAIN_LO, c93.VAL_HI
    m_prod = union_metrics(prod_alone, TR_LO, VAL_HI, 'production_alone/FULL_2025_2026')
    m_union = union_metrics(union_robust, TR_LO, VAL_HI, 'union_robust/FULL_2025_2026')
    pd.DataFrame([m_prod, m_union]).to_csv(SCRATCH / 'union_band23.csv', index=False)

    c93.append_result_md(
        "\n## Union (2025-01-01..2026-09-18, production + band-A robust pairs; fixed $375/fill)\n"
        f"- production ALONE: n={m_prod['n']:.0f}, {m_prod['fills_per_week']:.2f} fills/wk, mean {m_prod['mean_R']:+.3f} R/fill, "
        f"weekly P10 {m_prod['weekly_p10_R']:+.2f} R (${m_prod['weekly_p10_dollars']:+.0f}), "
        f"strong-week gap median/p90={m_prod['strong_week_gap_median']}/{m_prod['strong_week_gap_p90']} wk, "
        f"green {m_prod['green_share']} vs null {m_prod['green_null_mean']}\n"
        f"- UNION (production+band-A robust): n={m_union['n']:.0f}, {m_union['fills_per_week']:.2f} fills/wk, "
        f"mean {m_union['mean_R']:+.3f} R/fill, weekly P10 {m_union['weekly_p10_R']:+.2f} R (${m_union['weekly_p10_dollars']:+.0f}), "
        f"strong-week gap median/p90={m_union['strong_week_gap_median']}/{m_union['strong_week_gap_p90']} wk, "
        f"green {m_union['green_share']} vs null {m_union['green_null_mean']}\n"
        f"- added frequency: {m_union['fills_per_week'] - m_prod['fills_per_week']:+.2f} fills/wk from band A's robust pairs\n")
    store.close()
    log.info('=== stage_score DONE ===')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['pools', 'pipeline', 'score', 'all'], default='all')
    a = ap.parse_args()
    if a.stage in ('pools', 'all'):
        stage_pools()
    if a.stage in ('pipeline', 'all'):
        stage_pipeline()
    if a.stage in ('score', 'all'):
        stage_score()


if __name__ == '__main__':
    main()
