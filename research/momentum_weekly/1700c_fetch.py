#!/usr/bin/env python3
"""Cell 1,700c step 2 -- fetch Alpaca daily bars 2015-07-01..2026-09-30 for every us_equity asset
(active + inactive, from 1700c_assets.csv) plus SPY, batched by symbol, resumable, rate-limit aware.

PREREG: research/momentum_weekly/PREREG_1700c.md (FROZEN). adjustment=Adjustment.ALL (splits AND
dividends back-adjusted by Alpaca -- the only adjustment mode requested by the owner-level task
instructions; RESULT_1700c.md states this plainly since the PREREG's "dividends ignored" line assumes a
price-return series, and all-adjusted closes fold dividends into one-time back-adjustments rather than
cash distributions -- stated, not hidden).

Resumability: one parquet shard per 200-symbol batch under shards/, named by batch index. A batch whose
shard file already exists is skipped (safe to re-run after a crash, OOM kill, or manual stop). A batch
that errors out after the client's own retry/backoff is NOT retried forever: it is written as an empty
(zero-row) shard so the job still terminates, and its symbols go to 1700c_lost.csv with the error reason
-- this is the LOST list the completeness gate reads.

Disk safety: this node is at ~95% disk and shares the box with the live trading service (never touch its
cache.db). Every batch checks free disk >= 1 GB before fetching and aborts loudly if not.

One process, single-threaded, run via nice -n 10, intended to be launched detached (setsid nohup ... &).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research' / 'momentum_weekly'
SHARD_DIR = OUT / 'shards'
SHARD_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / '.env')

logging.basicConfig(
    filename=str(OUT / '1700c_fetch.log'), filemode='a', level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700c_fetch')
log.addHandler(logging.StreamHandler(sys.stdout))

sys.path.insert(0, str(ROOT))
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402
from alpaca.data.enums import Adjustment, DataFeed  # noqa: E402
from data_sources.alpaca_client import AlpacaClient  # noqa: E402

START = pd.Timestamp('2015-07-01')
END = pd.Timestamp('2026-09-30')
BATCH_SIZE = 200
MIN_FREE_BYTES = 1 * 1024 ** 3  # 1 GB floor, shared disk with the live trading service
COLS = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']
t0 = time.time()


def elapsed() -> str:
    return f'{time.time() - t0:6.0f}s'


def free_gb() -> float:
    return shutil.disk_usage('/').free / 1024 ** 3


api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_API_SECRET')
if not api_key or not api_secret:
    log.error('ALPACA_API_KEY / ALPACA_API_SECRET missing from .env -- aborting, no mock fallback')
    raise SystemExit(1)
paper = os.getenv('ALPACA_PAPER', 'true').strip().lower() == 'true'
client = AlpacaClient(api_key, api_secret, paper=paper)
log.info('%s AlpacaClient initialized (paper=%s) -- read-only get_stock_bars calls only', elapsed(), paper)

assets_path = OUT / '1700c_assets.csv'
if not assets_path.exists():
    log.error('%s %s missing -- run 1700c_assets.py (step 1) first', elapsed(), assets_path)
    raise SystemExit(1)
assets_df = pd.read_csv(assets_path)
symbols = sorted(set(assets_df['symbol'].dropna().astype(str)) | {'SPY'})
log.info('%s universe: %d distinct symbols (incl. SPY) from %s', elapsed(), len(symbols), assets_path.name)

lost_path = OUT / '1700c_lost.csv'
if not lost_path.exists():
    pd.DataFrame(columns=['batch', 'symbol', 'reason']).to_csv(lost_path, index=False)

# Alpaca's inactive-asset list includes CUSIP-shaped placeholder identifiers for corporate-action claims
# (contingent value rights / escrow / contingent-payment entitlements -- 'CVR','ESC','CNT','RGT' suffixes),
# never real tradable tickers. Empirically confirmed: a live run rejected 100% of a ~190-symbol sample of
# these with "invalid symbol". No legitimate Alpaca equity ticker starts with a digit, so this is a safe,
# narrow, evidence-based pre-filter (NOT the blind retry/removal bug from feedback_fetch_completeness_gate
# -- it only ever drops digit-leading symbols, never a letter-leading one like NVDA/AAPL) that saves
# hundreds of guaranteed-to-fail API calls. Still logged to LOST, not silently dropped.
cusip_like = [s for s in symbols if re.match(r'^[0-9]', s)]
if cusip_like:
    log.warning('%s %d/%d symbols start with a digit (CUSIP-shaped placeholders, e.g. %s) -- pre-filtered '
                'out of the fetch before any API call, logged to LOST reason=cusip_like_pre_filtered',
                elapsed(), len(cusip_like), len(symbols), cusip_like[:3])
    pd.DataFrame({'batch': -1, 'symbol': cusip_like, 'reason': 'cusip_like_pre_filtered'}).to_csv(
        lost_path, mode='a', header=False, index=False)
    symbols = [s for s in symbols if s not in set(cusip_like)]

batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
n_batches = len(batches)
log.info('%s %d batches of <=%d symbols, window %s..%s, adjustment=ALL, feed=SIP',
         elapsed(), n_batches, BATCH_SIZE, START.date(), END.date())


def bars_to_df(barset, requested: list[str]) -> pd.DataFrame:
    """BarSet -> flat DataFrame with COLS only. Tries the SDK's .df property first (documented on
    BaseDataSet/TimeSeriesMixin); falls back to walking .data if that ever changes shape."""
    try:
        raw = barset.df
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=COLS)
        df = raw.reset_index()
        df = df.rename(columns={'timestamp': 'bar_date'})
        df['bar_date'] = pd.to_datetime(df['bar_date']).dt.tz_localize(None).dt.normalize()
        return df[COLS]
    except Exception as e:
        log.warning('%s barset.df path failed (%s) -- falling back to manual .data walk', elapsed(), e)
        rows = []
        data = getattr(barset, 'data', {}) or {}
        for sym in requested:
            for b in data.get(sym, []):
                rows.append(dict(symbol=sym, bar_date=pd.Timestamp(b.timestamp).tz_localize(None).normalize(),
                                  open=float(b.open), high=float(b.high), low=float(b.low),
                                  close=float(b.close), volume=int(b.volume)))
        return pd.DataFrame(rows, columns=COLS) if rows else pd.DataFrame(columns=COLS)


INVALID_SYM_RE = re.compile(r'invalid symbol:\s*([^"\s]+)')


def fetch_batch_resilient(batch: list[str], idx: int, max_strip: int = 250):
    """Fetch one batch; if Alpaca rejects a specific symbol as invalid (CUSIP-like placeholder
    symbols exist in the inactive-asset list, e.g. '0029900E0'), strip ONLY that symbol and retry the
    rest -- never the whole batch. This is the fix for the known failure mode (memory:
    feedback_fetch_completeness_gate.md) where a shared invalid-symbol/retry loop silently dropped
    ~5,000 good tickers (NVDA, AAPL) in a past run: here, a symbol is only ever dropped because the
    API itself named it invalid, one at a time, never by a local heuristic.
    Returns (barset_or_None, remaining_symbols, bad_symbols_removed).
    """
    remaining = list(batch)
    removed = []
    for _ in range(max_strip):
        if not remaining:
            return None, remaining, removed
        req = StockBarsRequest(symbol_or_symbols=remaining, timeframe=TimeFrame.Day, start=START, end=END,
                                adjustment=Adjustment.ALL, feed=DataFeed.SIP)
        try:
            barset = client._call_with_timeout(
                lambda r=req: client.data_client.get_stock_bars(r), f'fetch_batch_{idx}',
                timeout=300, timeout_retries=2, rate_limit_retries=6)
            return barset, remaining, removed
        except Exception as e:
            m = INVALID_SYM_RE.search(str(e))
            bad = m.group(1) if m else None
            if bad and bad in remaining:
                remaining.remove(bad)
                removed.append(bad)
                log.warning('%s batch %d: API rejected symbol %s as invalid -- stripped, retrying '
                            'the other %d symbols (not the whole batch)', elapsed(), idx + 1, bad, len(remaining))
                continue
            raise  # not a per-symbol invalid-symbol error -- let the outer handler treat the whole batch
    log.error('%s batch %d: exceeded %d invalid-symbol strips -- giving up on remaining %d symbols',
              elapsed(), idx + 1, max_strip, len(remaining))
    return None, remaining, removed


n_done_now = n_skipped = n_failed = 0
for idx, batch in enumerate(batches):
    shard_path = SHARD_DIR / f'shard_{idx:04d}.parquet'
    if shard_path.exists():
        n_skipped += 1
        continue

    if free_gb() < MIN_FREE_BYTES / 1024 ** 3:
        log.error('%s disk free %.2fGB < 1GB floor -- ABORTING fetch to protect the live trading '
                   'service''s disk (%d/%d batches done, resume later with the same command)',
                   elapsed(), free_gb(), idx, n_batches)
        sys.exit(1)

    try:
        barset, remaining, bad_syms = fetch_batch_resilient(batch, idx)
        if bad_syms:
            pd.DataFrame({'batch': idx, 'symbol': bad_syms, 'reason': 'invalid_symbol_api_rejected'}).to_csv(
                lost_path, mode='a', header=False, index=False)
        df = bars_to_df(barset, remaining) if barset is not None else pd.DataFrame(columns=COLS)
        if barset is None and remaining:
            # exceeded max_strip -- treat whatever is left as lost-this-run (not individually rejected,
            # so logged distinctly; a future re-run starts fresh since no shard exists for this batch... )
            # NOTE: a shard IS written below (empty-for-these-symbols) so the run terminates; see log.
            pd.DataFrame({'batch': idx, 'symbol': remaining, 'reason': 'gave_up_after_max_strip'}).to_csv(
                lost_path, mode='a', header=False, index=False)
        tmp = shard_path.with_suffix('.tmp')
        df.to_parquet(tmp, index=False)
        os.replace(tmp, shard_path)
        n_with_bars = df['symbol'].nunique() if len(df) else 0
        n_done_now += 1
        log.info('%s batch %d/%d: %d requested, %d invalid-stripped, %d symbols with >=1 bar, %d rows, '
                  'free_disk=%.1fGB', elapsed(), idx + 1, n_batches, len(batch), len(bad_syms), n_with_bars,
                  len(df), free_gb())
    except Exception as e:
        log.error('%s batch %d/%d FAILED after retries (%s, not a per-symbol invalid-symbol case) -- '
                   'all %d symbols -> LOST (written as empty shard so the run terminates; re-running will '
                   'NOT retry these automatically)', elapsed(), idx + 1, n_batches, e, len(batch))
        pd.DataFrame({'batch': idx, 'symbol': batch, 'reason': str(e)}).to_csv(
            lost_path, mode='a', header=False, index=False)
        tmp = shard_path.with_suffix('.tmp')
        pd.DataFrame(columns=COLS).to_parquet(tmp, index=False)
        os.replace(tmp, shard_path)
        n_failed += 1

log.info('%s fetch loop done: %d fetched now, %d skipped (already shardded), %d failed-this-run',
          elapsed(), n_done_now, n_skipped, n_failed)

shard_files = sorted(SHARD_DIR.glob('shard_*.parquet'))
if len(shard_files) < n_batches:
    log.warning('%s INCOMPLETE: %d/%d shards exist -- NOT merging yet. Re-run this script '
                '(nice -n 10 python3 research/momentum_weekly/1700c_fetch.py) to resume the remaining '
                '%d batches.', elapsed(), len(shard_files), n_batches, n_batches - len(shard_files))
    sys.exit(0)

log.info('%s all %d shards present -- STEP: streaming merge to panel_2016_2026.parquet (row-group-wise, '
          'RAM-safe on this 1.8GB-free node)', elapsed(), n_batches)
final_path = OUT / 'panel_2016_2026.parquet'
tmp_final = OUT / 'panel_2016_2026.parquet.tmp'
writer = None
total_rows = 0
symbols_with_bars: set[str] = set()
try:
    for sf in shard_files:
        table = pq.read_table(sf)
        if table.num_rows == 0:
            continue
        if writer is None:
            writer = pq.ParquetWriter(str(tmp_final), table.schema)
        writer.write_table(table)
        total_rows += table.num_rows
        symbols_with_bars.update(table.column('symbol').unique().to_pylist())
finally:
    if writer is not None:
        writer.close()
if writer is None:
    log.error('%s merge produced ZERO rows across all shards -- aborting, nothing to write', elapsed())
    sys.exit(1)
os.replace(tmp_final, final_path)
log.info('%s wrote %s: %d rows, %d distinct symbols with >=1 bar', elapsed(), final_path.name,
          total_rows, len(symbols_with_bars))

# ========================================================== completeness line ==
lost_df = pd.read_csv(lost_path)
lost_symbols = set(lost_df['symbol'].astype(str)) if len(lost_df) else set()
active_syms = set(assets_df.loc[assets_df.status == 'active', 'symbol'].astype(str))
active_with_bars = active_syms & symbols_with_bars
active_cov_pct = 100.0 * len(active_with_bars) / max(1, len(active_syms))
comp = pd.DataFrame([dict(
    assets_requested=len(symbols), symbols_with_bars=len(symbols_with_bars),
    symbols_lost=len(lost_symbols), active_assets_requested=len(active_syms),
    active_with_bars=len(active_with_bars), active_coverage_pct=active_cov_pct,
    gate_95pct_pass=active_cov_pct >= 95.0)])
comp.to_csv(OUT / '1700c_completeness.csv', index=False)
gate = 'PASS' if active_cov_pct >= 95.0 else 'FAIL -> VOID per PREREG'
log.info('%s COMPLETENESS: requested=%d with_bars=%d LOST=%d | active_requested=%d active_with_bars=%d '
          'active_coverage=%.1f%% gate(>=95%%)=%s', elapsed(), len(symbols), len(symbols_with_bars),
          len(lost_symbols), len(active_syms), len(active_with_bars), active_cov_pct, gate)
log.info('%s DONE step 2', elapsed())
