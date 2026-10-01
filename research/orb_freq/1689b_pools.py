"""Cell 1,689b -- ORB frequency sub-pools 21-30 (PREREG_1684.md amendment 2, owner 10/1 ~10:00 UTC
"get me more sub-pools"). Reuses the 1,684/1,685/1,689a harness: same production pipeline
(study_orb_pipeline_static_lock.py) at the LIVE config, same scoring/union/cadence machinery
(1684_score.py, imported verbatim), same window convention (in-regime 2025-01-01..2026-09-26
halves by calendar year, out-regime 2024H2).

Pools 23/24/25/26/30 (cheap -- built from the two seeds already on disk, pools_1689b_lib's "2-5%"
(effectively >=3%, see that module's docstring) wide-seed feature CSVs and cell 1,689a's own >=5%
100K-500K slice): no new minute-bar fetch, only a join to the daily-bar panel (open/prev_volume/
yesterday's own return+range) and a direct bars_sip.db query for pre-market/range aggregates
production's own feature build does not expose as a column (own-range high, pre-market high/
volume, first-bar low -- the SAME RANGE_MINUTES=5 window study_orb_features.py uses).

Pools 21/27/28 (need a fresh build -- price bands outside both seeds' $1-50 coverage, or a volume
floor neither seed carries): fresh daily-bar candidate list (gap screen buffered -1.0pp, same
1,689a convention, trued up on the minute-bar gap_pct after the feature build) -> backfill_bars_
sip.py via a monkeypatch wrapper (FEATURES/STATE redirected to this cell's own files, NEVER the
HOD causal-filter features.csv or its state) -> 1689b_features.py loader-seam build -> same
pipeline.

Pools 22 (earnings-day) and 29 (sector sympathy): VOID -- see VOID_POOLS below for the reason
(checked before writing any code here: no earnings-date calendar, no ticker-to-sector/GICS map
exists anywhere on disk).

Every pool runs ONLY the live exit (no exit chosen from the fixed menu on TRAIN) -- stated here
per this cell's own budget allowance, not hidden: running 3 exits x 8 pools x 2 windows was outside
the <=120-tool-call budget; "own selection chain" (the per-pool ranking/8-slot competition) is
unaffected, since study_orb_pipeline_static_lock.py's own ranking runs on every pool's candidates
in complete isolation regardless of which exit is active.

Usage: python3 research/orb_freq/1689b_pools.py --stage {prep,backfill,build_feats,pipeline,score,all}
"""
import argparse
import importlib.util
import os
import subprocess
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pools_1689b_lib import (  # noqa: E402
    ROOT, OUT, POOLDIR, BARS_SIP, IN_REGIME, OUT_REGIME, log,
    load_panel_with_lags, load_wide_seed, join_price_fields, bar_aggregates,
)

WINDOWS = ('in_regime', 'out_regime')
DAILY_SRC = {w: OUT / f'daily_source_1689b_{w}.parquet' for w in WINDOWS}

VOID_POOLS = {
    22: "no EARNINGS-date calendar exists: research/edgar_desk's own events (cell_1552_events.csv) "
        "carry 9 filing classes -- OFFERING/OFFICER_EXIT/CONTRACT/REVERSE_SPLIT/SHELF/ACTIVIST/"
        "AUDITOR/LATE_FILING/NON_RELIANCE -- NO EARNINGS class (grepped the cls column directly); "
        "research/multiday/data (which held edgar_earnings.py/edgar_eps.py) is DELETED on disk, "
        "uncommitted, mid-edit elsewhere on this branch right now -- not ours to touch or rebuild "
        "under this budget. No other earnings-date source was found under research/ or data/.",
    29: "no ticker-to-sector/GICS map exists anywhere on disk: grepped data/research and research/ "
        "for sector/gics/industry file names and code -- data/research/orb_asset_class_map_"
        "20260711.csv is wrapper/asset-class identification (stock vs 2x-leveraged wrapper), not a "
        "sector map; research/sector_mom/holdings_1652.csv is a sector-ETF ROTATION strategy's own "
        "monthly XLB/XLE/XLI/... ETF holdings, not a per-ticker constituent map. There is no way to "
        "look up 'the sector of today's >=10% gapper' for an arbitrary symbol -- VOID per the "
        "PREREG's own instruction, not skipped silently.",
}

# gap bounds on the TRUE (minute-bar) gap_pct column already in the WIDE feature rows. 2-3% is a
# <150-row sliver in BOTH windows (101/22,606 in-regime, 88/5,539 out-regime -- checked empirically
# before writing this file) -- the same artifact 1685_subpools.py found and scoped band A out for.
# These pools use gap>=3%, stated, not hidden (pool 23 has no upper or lower gap bound of its own;
# see its docstring note in _build_cheap_pools).
CHEAP_SPECS = {
    23: dict(gap_lo=None, gap_hi=None, price_lo=3.0, price_hi=30.0, needs_premkt=False, needs_range=False),
    24: dict(gap_lo=3.0, gap_hi=5.0, price_lo=3.0, price_hi=30.0, needs_premkt=True, needs_range=True),
    25: dict(gap_lo=3.0, gap_hi=5.0, price_lo=3.0, price_hi=30.0, needs_premkt=True, needs_range=False),
    26: dict(gap_lo=3.0, gap_hi=5.0, price_lo=3.0, price_hi=30.0, needs_premkt=True, needs_range=True),
    30: dict(gap_lo=3.0, gap_hi=5.0, price_lo=3.0, price_hi=30.0, needs_premkt=False, needs_range=True),
}
BUILD_SPECS = {
    21: dict(gap_lo=4.0, price_lo=50.0, price_hi=200.0, vol_lo=1_000_000),
    27: dict(gap_lo=5.0, price_lo=1.0, price_hi=3.0, vol_lo=2_000_000),
    28: dict(gap_lo=5.0, price_lo=30.0, price_hi=100.0, vol_lo=500_000),
}
LIVE_POOLS = sorted(list(CHEAP_SPECS) + list(BUILD_SPECS))   # [21,23,24,25,26,27,28,30]


# ===================================================================== cheap pools (23-26,30) =====
def _cheap_extra_mask(pool_id, band):
    if pool_id == 23:
        # "yesterday's strong close": up>=5% YESTERDAY (own return, not today's gap) AND closed in
        # the top 10% of yesterday's own range -- today's gap is whatever the seed contains (see
        # module docstring: effectively >=3%, with a sanity cap already applied by the caller).
        return (band['yday_return_pct'] >= 5.0) & (band['yday_close_position'] >= 0.9)
    if pool_id == 24:
        return band['range_high'] > band['premarket_high']
    if pool_id == 25:
        return band['premarket_volume'] >= 0.20 * band['avg_daily_volume_20d']
    if pool_id == 26:
        return (band['range_size_pct'] <= 1.5) & (band['range_high'] > band['premarket_high'])
    if pool_id == 30:
        return (band['first_bar_low'] >= band['open']) & (band['range_close_position'] >= 0.75)
    raise ValueError(pool_id)


def _build_cheap_pools(window, panel):
    w = load_wide_seed(window)
    base_cols = w.columns.tolist()
    extra_1689a = None
    p19 = OUT / f'subpools_1689a/19PRIME_{window}_features.csv'
    if p19.exists():
        e = pd.read_csv(p19, keep_default_na=False, na_values=[''])
        extra_1689a = e[[c for c in base_cols if c in e.columns]]
    lo, hi = IN_REGIME if window == 'in_regime' else OUT_REGIME

    for pool_id, spec in CHEAP_SPECS.items():
        src = w.assign(_bars_source='wide')
        if pool_id == 23 and extra_1689a is not None:
            src = pd.concat([src, extra_1689a.assign(_bars_source='1689a')], ignore_index=True)
            src = src.drop_duplicates(['symbol', 'date'])
            log.info('pool 23/%s: unioned WIDE seed (%d) with 1689a 19PRIME slice (%d) -> %d rows',
                      window, len(w), len(extra_1689a), len(src))
        m = join_price_fields(src, panel)
        m = m[(m['date'] >= lo) & (m['date'] <= hi)].copy()
        before = len(m)
        price_ok = (m['open'] >= spec['price_lo']) & (m['open'] <= spec['price_hi'])
        gap_ok = pd.Series(True, index=m.index)
        if spec['gap_lo'] is not None:
            gap_ok &= (m['gap_pct'] >= spec['gap_lo'])
        if spec['gap_hi'] is not None:
            gap_ok &= (m['gap_pct'] < spec['gap_hi'])
        if pool_id == 23:
            n_outlier = int((m['gap_pct'].abs() > 500.0).sum())
            if n_outlier:
                log.warning('pool 23/%s: %d/%d rows have |gap_pct|>500%% (near-zero prev_close '
                            'artifacts) -- excluded as a sanity cap', window, n_outlier, before)
            gap_ok &= (m['gap_pct'].abs() <= 500.0)
        band = m[price_ok & gap_ok].copy()
        log.info('pool %d/%s: %d/%d rows pass price[%.0f,%.0f] x gap band', pool_id, window,
                  len(band), before, spec['price_lo'], spec['price_hi'])

        if spec['needs_premkt'] or spec['needs_range']:
            pairs = list(zip(band['symbol'], band['date']))
            agg = bar_aggregates(pairs, window)
            band = band.merge(agg, how='left', left_on=['symbol', 'date'], right_on=['symbol', 'day'])
            if spec['needs_premkt']:
                n_cov = int((band['premarket_bars'].fillna(0) > 0).sum())
                log.info('pool %d/%s: pre-market (04:00-09:30) bar coverage %d/%d (%.1f%%)',
                          pool_id, window, n_cov, len(band), 100 * n_cov / len(band) if len(band) else 0.0)
            if spec['needs_range']:
                n_covr = int((band['range_bars'].fillna(0) > 0).sum())
                log.info('pool %d/%s: own-range (09:30-09:35) bar coverage %d/%d (%.1f%%)',
                          pool_id, window, n_covr, len(band), 100 * n_covr / len(band) if len(band) else 0.0)

        extra_mask = _cheap_extra_mask(pool_id, band).fillna(False)
        final = band[extra_mask].copy()
        log.info('pool %d/%s: FINAL %d rows after the pool-specific condition', pool_id, window, len(final))
        if pool_id == 23 and window == 'in_regime':
            # WIDE-sourced rows' minute bars live in cache.db (production's own scan store);
            # 1689a-sourced rows' minute bars were backfilled into bars_sip.db -- the pipeline needs
            # a DIFFERENT ORB_BT_BARS_DB for each, so they are written as two files and run as two
            # pipeline invocations (stage_pipeline), then concatenated back before scoring.
            for src_tag in ('wide', '1689a'):
                part = final[final['_bars_source'] == src_tag]
                fp = POOLDIR / f'23_{window}_{src_tag}_features.csv'
                part[base_cols].to_csv(fp, index=False)
                log.info('pool 23/%s/%s: %d rows -> %s', window, src_tag, len(part), fp)
        else:
            fp = POOLDIR / f'{pool_id}_{window}_features.csv'
            final[base_cols].to_csv(fp, index=False)
            log.info('wrote %s rows=%d', fp, len(final))
        log.info('wrote %s rows=%d', fp, len(final))


# ===================================================================== build pools (21,27,28) =====
def _build_candidates_2128(window, panel_w):
    rows = panel_w.dropna(subset=['prev_close', 'prev_volume']).copy()
    out = rows[['symbol', 'bar_date', 'open', 'prev_close', 'prev_volume', 'gap_pct_daily']].rename(
        columns={'bar_date': 'date'})
    for pool_id, spec in BUILD_SPECS.items():
        fetch_gap_lo = spec['gap_lo'] - 1.0   # buffer, same convention as cell 1,689a's FETCH_GAP_MIN
        mask = ((rows['gap_pct_daily'] >= fetch_gap_lo) & (rows['open'] >= spec['price_lo']) &
                (rows['open'] <= spec['price_hi']) & (rows['prev_volume'] >= spec['vol_lo']))
        out[f'idea{pool_id}'] = mask.values
        log.info('build-candidate pool %d/%s: %d daily-screen rows (buffered gap>=%.1f%%, '
                  'price[%.0f,%.0f], prevVol>=%.0f)', pool_id, window, int(mask.sum()), fetch_gap_lo,
                  spec['price_lo'], spec['price_hi'], spec['vol_lo'])
    any_flag = out[[f'idea{p}' for p in BUILD_SPECS]].any(axis=1)
    out = out[any_flag].copy()
    fp = POOLDIR / f'build_candidates_{window}.csv'
    out.to_csv(fp, index=False)
    log.info('wrote %s rows=%d (distinct symbol-days needing a fresh fetch, pre-backfill-check)',
              fp, len(out))


# ===================================================================== stage: prep =================
def stage_prep():
    for window in WINDOWS:
        panel, _xcheck = load_panel_with_lags(window)
        lo, hi = IN_REGIME if window == 'in_regime' else OUT_REGIME
        dump = panel[['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']].copy()
        dump.to_parquet(DAILY_SRC[window], index=False)
        log.info('%s: wrote %s (%d rows, full market panel, for ATR14/lookback)', window,
                  DAILY_SRC[window], len(dump))

        panel_w = panel[(panel['bar_date'] >= lo) & (panel['bar_date'] <= hi)].copy()
        log.info('%s: %d rows / %d symbols / %d days in [%s,%s]', window, len(panel_w),
                  panel_w['symbol'].nunique(), panel_w['bar_date'].nunique(), lo, hi)

        _build_cheap_pools(window, panel)
        _build_candidates_2128(window, panel_w)


# ===================================================================== stage: backfill =============
def stage_backfill():
    disk = shutil.disk_usage('/')
    avail_gb = disk.free / 1e9
    log.info('df -h / : %.1f GB available (floor 5 GB, owner instruction)', avail_gb)
    if avail_gb < 5.0:
        log.error('ABORT backfill: only %.1f GB available, below the 5 GB floor -- not fetching', avail_gb)
        return
    frames = []
    for window in WINDOWS:
        fp = POOLDIR / f'build_candidates_{window}.csv'
        if not fp.exists():
            log.warning('%s missing -- run --stage prep first', fp)
            continue
        frames.append(pd.read_csv(fp, keep_default_na=False, na_values=['']))
    if not frames:
        log.error('no build_candidates files found -- aborting backfill')
        return
    cand = pd.concat(frames, ignore_index=True)[['date', 'symbol']].drop_duplicates()
    cand = cand.rename(columns={'date': 'day'})
    fetch_fp = POOLDIR / '1689b_fetch_list.csv'
    cand.to_csv(fetch_fp, index=False)
    log.info('fetch list: %d distinct (symbol,day) pairs -> %s', len(cand), fetch_fp)

    sys.path.insert(0, str(ROOT / 'research/bf_zero'))
    import backfill_bars_sip as bf  # noqa: E402
    bf.FEATURES = fetch_fp
    bf.STATE = POOLDIR / '1689b_backfill_state.json'
    log.info('backfill wrapper: FEATURES->%s STATE->%s (never touches the HOD causal_filter or '
              '1689a state files)', bf.FEATURES, bf.STATE)
    sys.argv = ['backfill_bars_sip.py']
    rc = bf.main()
    log.info('backfill rc=%d', rc)
    disk2 = shutil.disk_usage('/')
    log.info('df -h / after backfill: %.1f GB available', disk2.free / 1e9)


# ===================================================================== stage: build_feats ==========
def stage_build_feats():
    for window in WINDOWS:
        env = dict(os.environ, ORB1689B_WINDOW=window, PYTHONPATH=str(ROOT))
        log_path = POOLDIR / f'1689b_features_{window}.log'
        log.info('building features for %s -> %s', window, log_path)
        with open(log_path, 'w') as lf:
            rc = subprocess.run(['nice', '-n', '10', 'python3', str(OUT / '1689b_features.py')],
                                 cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        log.info('%s features build rc=%d', window, rc)

        fdir = OUT / f'out_1689b_{window}'
        cands_files = sorted(p for p in fdir.glob('orb_features_*.csv') if 'corrmatrix' not in p.name) if fdir.exists() else []
        if not cands_files:
            log.error('%s: no orb_features_*.csv produced under %s -- see %s', window, fdir, log_path)
            continue
        feat = pd.read_csv(cands_files[-1], keep_default_na=False, na_values=[''])
        cands = pd.read_csv(POOLDIR / f'build_candidates_{window}.csv', keep_default_na=False, na_values=[''])
        idea_cols = [f'idea{p}' for p in BUILD_SPECS]
        m = feat.merge(cands[['date', 'symbol'] + idea_cols], on=['symbol', 'date'], how='left')
        n_screen = int(cands[idea_cols].any(axis=1).sum())
        log.info('%s: minute-bar features built for %d/%d daily-screened candidates (%.1f%% fetch '
                  'success)', window, len(feat), n_screen, 100 * len(feat) / n_screen if n_screen else 0.0)
        for pool_id, spec in BUILD_SPECS.items():
            col = f'idea{pool_id}'
            flagged = m[col].fillna(False).astype(bool)
            true_gap = m['gap_pct'] >= spec['gap_lo']   # true-up on the minute-bar gap, 1689a convention
            sub = m.loc[flagged & true_gap, feat.columns]
            fp = POOLDIR / f'{pool_id}_{window}_features.csv'
            sub.to_csv(fp, index=False)
            log.info('pool %d/%s: %d rows (%d daily-flagged, true-gap>=%.1f%% kept) -> %s', pool_id,
                      window, len(sub), int(flagged.sum()), spec['gap_lo'], fp)


# ===================================================================== stage: pipeline ==============
def _pipeline_env_for(pool_id, window):
    """Same rule 1685_subpools.py's own _pipeline_env_for() documented and this cell verified
    empirically (see pools_1689b_lib.bar_aggregates docstring): in-regime WIDE-seed-sourced cheap
    pools (24,25,26,30) reuse cache.db DEFAULTS (their candidates' bars are in the normal
    production intraday_bars_1min store, never bars_sip.db) -- everything else (out-regime always;
    the fresh-build pools 21/27/28 both windows; the 1689a-sourced half of pool 23 in-regime) uses
    bars_sip.db + this cell's own daily_source_1689b_<window>.parquet."""
    if window == 'in_regime' and pool_id in (24, 25, 26, 30):
        # ORB_BT_BARS_DB left UNSET (defaults to cache.db, correct per the coverage check) but
        # ORB_BT_DAILY_SOURCE IS set to this cell's own already-built parquet -- leaving it unset
        # too makes the pipeline re-query cache.db's full daily_bars table for ATR14/lookback from
        # scratch, which hung for 10+ min with 0% CPU (shared-box contention, same as stage_prep's
        # slow full-panel load) on the first attempt; reusing the parquet already on disk is the
        # SAME data, just not re-fetched.
        return {'ORB_BT_DAILY_SOURCE': str(DAILY_SRC[window])}
    return {'ORB_BT_BARS_DB': str(BARS_SIP), 'ORB_BT_DAILY_SOURCE': str(DAILY_SRC[window])}


def _run_pipeline_once(fp, outp, extra_env, tag):
    if not fp.exists():
        log.warning('%s: %s missing -- skipping pipeline run', tag, fp)
        return False
    n = sum(1 for _ in open(fp)) - 1
    if n <= 0:
        log.warning('%s: 0 candidates -- skipping pipeline run', tag)
        return False
    env = dict(os.environ, ORB_BT_FEATURES_CSV=str(fp), ORB_BT_BOOK_OUT=str(outp),
               ORB_CATALYST_VETO='0', PYTHONPATH=str(ROOT), **extra_env)
    log_path = Path(str(outp).replace('_true.csv', '_true.log'))
    log.info('running pipeline for %s (n=%d, env=%s) -> %s', tag, n,
              'cache.db defaults' if not extra_env else extra_env.get('ORB_BT_BARS_DB'), outp)
    with open(log_path, 'w') as lf:
        rc = subprocess.run(['nice', '-n', '10', 'python3', 'study_orb_pipeline_static_lock.py'],
                             cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    log.info('%s pipeline rc=%d', tag, rc)
    return rc == 0


CACHE_DB_CONTENDED = {'24', '25', '26', '30'}   # in-regime, cache.db-default leg -- see note below


def stage_pipeline():
    """NOTE (added after a live retry): ORB_BT_BARS_DB left at its cache.db default hung at
    'Database initialized: data/cache.db' for 10+ min at ~0%% CPU on TWO separate attempts (one
    with ORB_BT_DAILY_SOURCE also defaulted, one with it pointed at this cell's own parquet -- the
    daily-source re-scan was ruled out as the cause). cache.db is the LIVE trading service's own
    database (confirmed active, PID checked) -- this looks like a write-lock wait on its Migration/
    dry_trades check, not a slow read. Per CLAUDE.md (never touch the live service) this cell does
    NOT retry cache.db-default mode again: pools 24/25/26/30's in-regime leg and pool 23's
    in-regime WIDE-sourced half are VOIDED here (not attempted), leaving every out-regime read
    (bars_sip.db only, measured 95-100% bar coverage, no cache.db contact) and pool 23's in-regime
    1689a-sourced half (bars_sip.db) as the only in-regime-adjacent reads this cell reports for
    those pools. 1685_subpools.py's own successful cache.db-default runs happened earlier, at a
    different, unknown lock-contention moment -- not reproducible safely under this budget."""
    for window in WINDOWS:
        for pool_id in LIVE_POOLS:
            if pool_id == 23 and window == 'in_regime':
                fp = POOLDIR / f'23_{window}_1689a_features.csv'
                outp = POOLDIR / f'23_{window}_1689a_true.csv'
                extra_env = {'ORB_BT_BARS_DB': str(BARS_SIP), 'ORB_BT_DAILY_SOURCE': str(DAILY_SRC[window])}
                ok = _run_pipeline_once(fp, outp, extra_env, f'pool 23/{window}/1689a')
                combined = POOLDIR / f'23_{window}_true.csv'
                if ok and outp.exists():
                    pd.read_csv(outp, keep_default_na=False, na_values=['']).to_csv(combined, index=False)
                    log.info('pool 23/%s: WIDE-sourced half VOIDED (cache.db contended, not run) -- '
                              '%s is the 1689a-sourced half ONLY', window, combined)
                continue
            if window == 'in_regime' and str(pool_id) in CACHE_DB_CONTENDED:
                log.warning('pool %d/%s: VOID -- cache.db-default leg not attempted (see '
                            'stage_pipeline docstring: hung 10+min at 0%% CPU, live-service '
                            'contention risk)', pool_id, window)
                continue
            fp = POOLDIR / f'{pool_id}_{window}_features.csv'
            outp = POOLDIR / f'{pool_id}_{window}_true.csv'
            _run_pipeline_once(fp, outp, _pipeline_env_for(pool_id, window), f'pool {pool_id}/{window}')


# ===================================================================== stage: score =================
def _score1684():
    spec = importlib.util.spec_from_file_location('score1684b', OUT / '1684_score.py')
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
    reads = []
    for pool_id in LIVE_POOLS:
        for window, prod, (lo, hi) in (('in_regime', prod_in, (IN_LO, IN_HI)),
                                        ('out_regime', prod_out, (OUT_LO, OUT_HI))):
            path = POOLDIR / f'{pool_id}_{window}_true.csv'
            df = s.load_book(path)
            n = 0 if df is None else len(df)
            print(f"\n--- POOL {pool_id} / {window} (n={n}) ---")
            if df is not None and len(df):
                tag = df.copy()
                tag['window'] = window
                tag['pool'] = pool_id
                all_rows.append(tag[['window', 'pool', 'date', 'symbol', 'entry_price', '_sized_pnl', 'R', '_composite']])
                if window == 'in_regime':
                    for yr in (2025, 2026):
                        print(s.fmt(s.stats(df[df.date.dt.year == yr], f'{pool_id}/{window}/{yr}')))
                st = s.stats(df, f'{pool_id}/{window}/FULL')
                print(s.fmt(st))
                st2 = dict(st); st2['pool'] = pool_id; st2['window'] = window
                reads.append(st2)
                union, addon, raw_overlap = s.union_book(prod, df)
                print(f"  raw_overlap_with_prod={raw_overlap:.1%} added_after_excl={len(addon)} "
                      f"(frequency gain = {len(addon)} fills)")
                print(s.fmt(s.stats(union, f'{pool_id}/{window}/UNION')))
                print(s.cadence_report(union, f'{pool_id}/{window}/UNION', lo, hi))
                print(s.cadence_report(prod, f'{window}/prod-alone', lo, hi))
                worst_pool_day = df.groupby(df.date.dt.date)['R'].sum().idxmin()
                prod_day_r = prod[prod.date.dt.date == worst_pool_day]['R'].sum() if prod is not None else 0.0
                print(f"  pool's worst day {worst_pool_day}: pool R sum="
                      f"{df[df.date.dt.date==worst_pool_day]['R'].sum():+.2f} production R sum that "
                      f"SAME day={prod_day_r:+.2f}")
            else:
                print(f"{pool_id}/{window}: n=0 (no fills)")
                reads.append(dict(label=f'{pool_id}/{window}', n=0, pool=pool_id, window=window))

    if all_rows:
        out_df = pd.concat(all_rows, ignore_index=True)
        out_df.to_csv(OUT / '1689b_pool_books.csv', index=False)
        print(f"\nwrote {OUT / '1689b_pool_books.csv'} rows={len(out_df)}")
    if reads:
        pd.DataFrame(reads).to_csv(OUT / '1689b_reads.csv', index=False)
        print(f"wrote {OUT / '1689b_reads.csv'} rows={len(reads)}")

    print("\n=== PASS BAR (own meanR>=+0.05 & dc_t>=2.0 in-regime, >=0 out-of-regime (not "
          "p30-negative), exTop5>0) ===")
    for pool_id in LIVE_POOLS:
        din = s.load_book(POOLDIR / f'{pool_id}_in_regime_true.csv')
        dout = s.load_book(POOLDIR / f'{pool_id}_out_regime_true.csv')
        sin = s.stats(din, f'{pool_id}/in')
        sout = s.stats(dout, f'{pool_id}/out')
        ok_in = sin.get('n', 0) > 0 and sin.get('mean_r', -9) >= 0.05 and (sin.get('dc_t') or -9) >= 2.0 \
            and (sin.get('ex_top5') or -9) > 0
        ok_out = sout.get('n', 0) == 0 or sout.get('mean_r', -9) >= 0.0
        verdict = 'PASS' if (ok_in and ok_out) else 'FAIL'
        print(f"{pool_id}: in n={sin.get('n',0)} meanR={sin.get('mean_r', float('nan')):+.3f} "
              f"dc_t={sin.get('dc_t', float('nan')):.2f} exTop5={sin.get('ex_top5', float('nan')):+.3f} | "
              f"out n={sout.get('n',0)} meanR={sout.get('mean_r', float('nan')):+.3f} -> {verdict}")
    for pool_id, reason in VOID_POOLS.items():
        print(f"{pool_id}: VOID -- {reason}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                     choices=['prep', 'backfill', 'build_feats', 'pipeline', 'score', 'all'])
    a = ap.parse_args()
    if a.stage in ('prep', 'all'):
        stage_prep()
    if a.stage in ('backfill', 'all'):
        stage_backfill()
    if a.stage in ('build_feats', 'all'):
        stage_build_feats()
    if a.stage in ('pipeline', 'all'):
        stage_pipeline()
    if a.stage in ('score', 'all'):
        stage_score()
