#!/usr/bin/env python3
"""MOM sleeve backtest panel refresh: append free Alpaca daily bars to ONE canonical extension file.

Spec: docs/mom_bt_current_spec_20261031.md (Deliverable 1).  The frozen panel
``research/momentum_weekly/panel_2016_2026.parquet`` is NEVER rewritten; every bar after it lives in
``research/momentum_weekly/panel_ext.parquet`` (same schema: symbol, bar_date, open, high, low, close, volume),
fetched with the sleeve's OWN machinery (``momentum_sleeve.fetch_batch_resilient`` / ``bars_to_df``,
adjustment=ALL, SIP) -- convention decision of cell 1,700z: Alpaca ALL as fetched, no rescale.

Run (Saturday, after the week closed):
    python3 scripts/mom_panel_refresh.py [--dry-run]

Idempotent (a second run the same day appends nothing).  If adjustment=ALL restates history (a split or
dividend in the week) the affected symbols' extension rows are REPLACED from a fresh fetch and listed (INFO).
Any failure of the completeness gate leaves the extension file UNTOUCHED and logs ERROR (-> Telegram).
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import momentum_sleeve as sleeve                                     # noqa: E402  (the sleeve's fetch machinery)
from monitoring.cron_guard import attach_telegram_error_handler, run_guarded  # noqa: E402

logger = logging.getLogger('mom_panel_refresh')

D = os.path.join(ROOT, 'research', 'momentum_weekly')
EXT_PATH = os.path.join(D, 'panel_ext.parquet')
MIN_FREE_BYTES = 5 * 1024 ** 3
OVERLAP_DAYS = 7                 # re-fetched calendar days before the last extension date: the restatement probe
RESTATE_TOL = 1e-4               # relative close difference that counts as a restated (re-adjusted) bar
COLS = sleeve.COLS
# Same common-stock name filter and test-ticker rule as research/momentum_weekly/REBUILD_1700tu.py
EXCL_RE = re.compile(r"\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|\bPREFERRED\b|\bRIGHTS?\b",
                     re.IGNORECASE)
ZZZT = re.compile(r"^Z[A-Z]ZZT$")


def free_bytes(path: str) -> int:
    """Free bytes on the filesystem holding ``path``."""
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def sleeve_universe(assets: pd.DataFrame, tracked: Optional[set] = None) -> List[str]:
    """Common-stock symbols of the sleeve's asset list (symbol, name): name-excluded and test tickers dropped,
    SPY always included.  ``tracked`` (the symbols already in the extension = the frozen panel's) restricts the
    list to names the backtest can ever use -- a listing without 273 bars of panel history cannot rank, and a
    symbol that simply did not trade in the 9-day probe window must not count as LOST (it did, 5.1 % on the
    first real dry run).  Sorted, unique."""
    names = assets['name'].fillna('')
    keep = assets.loc[~names.str.contains(EXCL_RE), 'symbol'].dropna()
    syms = {s for s in keep if not ZZZT.match(s) and (tracked is None or s in tracked)}
    return sorted(syms | {'SPY'})


def load_extension(path: str) -> Optional[pd.DataFrame]:
    """The extension parquet, or None (WARNING) when it does not exist."""
    if not os.path.exists(path):
        logger.warning("mom_panel_refresh: extension %s does not exist -- nothing to extend from", path)
        return None
    df = pd.read_parquet(path)
    df['bar_date'] = pd.to_datetime(df['bar_date'])
    return df


def fetch_bars(client, symbols: List[str], start: pd.Timestamp, end: pd.Timestamp,
               fetch_fn: Optional[Callable] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Daily bars for ``symbols`` start..end in batches of the sleeve's BATCH_SIZE.  Returns (bars, LOST symbols).
    CUSIP-shaped (digit-leading) symbols are pre-filtered and counted LOST; a symbol is dropped only when the API
    names it invalid; a failed batch counts all its symbols LOST (ERROR)."""
    fetch_fn = fetch_fn or sleeve.fetch_batch_resilient
    cusip = [s for s in symbols if re.match(r'^[0-9]', s)]
    syms = [s for s in symbols if s not in set(cusip)]
    batches = [syms[i:i + sleeve.BATCH_SIZE] for i in range(0, len(syms), sleeve.BATCH_SIZE)]
    frames, lost = [], list(cusip)
    for i, batch in enumerate(batches):
        try:
            barset, remaining, removed = fetch_fn(client, batch, start, end)
        except Exception as e:  # noqa: BLE001 - every symbol of the batch is counted LOST and reported
            logger.error("mom_panel_refresh: batch %d/%d failed (%s) -- %d symbols LOST", i + 1, len(batches), e,
                         len(batch))
            lost += batch
            continue
        lost += removed
        if barset is None:
            lost += remaining
            continue
        df = sleeve.bars_to_df(barset)
        lost += [s for s in remaining if s not in set(df['symbol'])]
        frames.append(df)
        if (i + 1) % 10 == 0 or i + 1 == len(batches):
            logger.info("mom_panel_refresh: %d/%d batches fetched, %d LOST so far", i + 1, len(batches), len(lost))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    out['bar_date'] = pd.to_datetime(out['bar_date'])
    return out, sorted(set(lost))


def find_restated(old: pd.DataFrame, new: pd.DataFrame) -> List[str]:
    """Symbols whose stored close differs (> RESTATE_TOL, relative) from the fresh fetch on a shared bar_date."""
    m = old[['symbol', 'bar_date', 'close']].merge(new[['symbol', 'bar_date', 'close']], on=['symbol', 'bar_date'],
                                                  suffixes=('_old', '_new'))
    if m.empty:
        return []
    rel = (m['close_new'].astype('float64') / m['close_old'].astype('float64') - 1).abs()
    return sorted(m.loc[rel > RESTATE_TOL, 'symbol'].unique())


def merge_extension(old: pd.DataFrame, new: pd.DataFrame, restated_full: Optional[pd.DataFrame] = None
                    ) -> Tuple[pd.DataFrame, int, List[str]]:
    """Merge fresh bars into the extension.  Bars already present (symbol, bar_date) are left as stored, new bars
    are appended; ``restated_full`` (fresh bars of the restated symbols over their whole extension span) REPLACES
    those symbols' rows.  Returns (merged, n_appended, restated_symbols)."""
    restated = sorted(restated_full['symbol'].unique()) if restated_full is not None and len(restated_full) else []
    base = old[~old['symbol'].isin(restated)] if restated else old
    pool = pd.concat([base, restated_full[COLS]] if restated else [base], ignore_index=True)
    have = set(zip(pool['symbol'], pool['bar_date']))
    fresh = new[[(s, d) not in have for s, d in zip(new['symbol'], new['bar_date'])]]
    merged = pd.concat([pool, fresh[COLS]], ignore_index=True)
    merged = merged.sort_values(['symbol', 'bar_date'], kind='stable').reset_index(drop=True)
    return merged, len(fresh), restated


def write_atomic(df: pd.DataFrame, path: str) -> None:
    """Write the parquet to a temp file in the same directory, then rename over ``path``."""
    tmp = path + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def refresh(client, assets: pd.DataFrame, ext_path: str = EXT_PATH, dry_run: bool = False,
            today: Optional[date] = None, free_fn: Callable[[str], int] = free_bytes,
            fetch_fn: Optional[Callable] = None) -> int:
    """Run one refresh.  Returns 0 on success (incl. nothing to append), 1 on any failure (file untouched)."""
    today = today or datetime.now(timezone.utc).date()
    free = free_fn(os.path.dirname(ext_path) or '.')
    if free < MIN_FREE_BYTES:
        logger.error("mom_panel_refresh: disk free %.1f GB < %.0f GB -- refusing to fetch",
                     free / 1e9, MIN_FREE_BYTES / 1e9)
        return 1
    old = load_extension(ext_path)
    if old is None or old.empty:
        logger.error("mom_panel_refresh: extension %s missing or empty -- adopt panel_ext.parquet first", ext_path)
        return 1
    last = old['bar_date'].max()
    start = last - pd.Timedelta(days=OVERLAP_DAYS)
    end = pd.Timestamp(today) + pd.Timedelta(days=1)
    symbols = sleeve_universe(assets, set(old['symbol'].unique()))
    logger.info("mom_panel_refresh: extension %s rows %s..%s; fetching %d symbols %s..%s",
                f"{len(old):,}", old['bar_date'].min().date(), last.date(), len(symbols), start.date(), end.date())
    new, lost = fetch_bars(client, symbols, start, end, fetch_fn)
    if new.empty or 'SPY' not in set(new['symbol']):
        logger.error("mom_panel_refresh: fetch returned %s -- extension untouched",
                     "no bars" if new.empty else "no SPY bars")
        return 1
    asof = new.loc[new['symbol'] == 'SPY', 'bar_date'].max()
    # completeness gate (the sleeve's own): LOST share and liquid-name asof coverage, on extension + fresh bars
    probe = pd.concat([old, new[COLS]], ignore_index=True).drop_duplicates(['symbol', 'bar_date'], keep='last')
    info = sleeve.completeness_info(len(symbols), probe, lost, asof.date())
    line = (f"COMPLETENESS: requested {info['requested']}, LOST {len(lost)} ({info['lost_share']:.1%}, max "
            f"{sleeve.LOST_MAX:.0%}), liquid {info['liquid_with_asof']}/{info['n_liquid']} with an {asof.date()} bar "
            f"({info['liquid_ratio']:.1%}, min {sleeve.COMPLETENESS_MIN:.0%})")
    if info['lost_share'] > sleeve.LOST_MAX or info['liquid_ratio'] < sleeve.COMPLETENESS_MIN:
        logger.error("mom_panel_refresh: GATE FAILED, extension untouched -- %s", line)
        return 1
    logger.info("mom_panel_refresh: %s", line)
    restated_syms = find_restated(old, new)
    restated_full = None
    if restated_syms:
        span_start = old.loc[old['symbol'].isin(restated_syms), 'bar_date'].min()
        full, lost2 = fetch_bars(client, restated_syms, span_start, end, fetch_fn)
        if lost2:
            logger.error("mom_panel_refresh: restated symbols %s could not be re-fetched -- extension untouched",
                         lost2)
            return 1
        restated_full = full
    merged, n_new, restated = merge_extension(old, new, restated_full)
    if restated:
        logger.info("mom_panel_refresh: adjustment=ALL restated %d symbols, their extension rows REPLACED: %s",
                    len(restated), restated)
    new_dates = sorted(new.loc[~new['bar_date'].isin(old['bar_date'].unique()), 'bar_date'].unique())
    span = f"{pd.Timestamp(new_dates[0]).date()}..{pd.Timestamp(new_dates[-1]).date()}" if new_dates else "none"
    if n_new == 0 and not restated:
        logger.info("mom_panel_refresh: nothing to append (extension already current through %s)", last.date())
        return 0
    if dry_run:
        print(f"DRY-RUN: would append {n_new:,} rows (new dates {span}), replace {len(restated)} restated symbols; "
              f"extension {len(old):,} -> {len(merged):,} rows, last bar {merged['bar_date'].max().date()}")
        return 0
    write_atomic(merged, ext_path)
    print(f"mom_panel_refresh: appended {n_new:,} rows (new dates {span}), replaced {len(restated)} restated "
          f"symbols; extension {len(old):,} -> {len(merged):,} rows, last bar {merged['bar_date'].max().date()}")
    return 0


def main() -> int:
    """CLI entry (cron: Saturday 13:00 UTC)."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dry-run', action='store_true', help='fetch and report, write nothing')
    ap.add_argument('--ext-path', default=EXT_PATH)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    attach_telegram_error_handler()
    from config import Config
    Config()  # loads .env
    key, secret = os.environ.get('ALPACA_MOM_API_KEY', ''), os.environ.get('ALPACA_MOM_API_SECRET', '')
    if not key or not secret:
        logger.error("mom_panel_refresh: ALPACA_MOM_API_KEY/SECRET not set -- cannot fetch")
        return 1
    from data_sources.alpaca_client import AlpacaClient
    client = AlpacaClient(key, secret, paper=True)       # data API only; no order call is ever made
    assets = sleeve.fetch_assets(client)
    return refresh(client, assets, ext_path=args.ext_path, dry_run=args.dry_run)


if __name__ == '__main__':
    sys.exit(run_guarded(main, 'mom_panel_refresh'))
