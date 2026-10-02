"""Weekly momentum sleeve: universe filter + 12-1 / vol signal + top-N ranking (PURE functions).

ONE spec shared by the research reference (research/momentum_weekly/1700j_frontier.py, reconciled book,
RECON_1700_sleeve.md, PREREG_1700j.md Amendment 1) and the paper/live script (scripts/mom_sleeve.py).
The panel math below is copied verbatim from the reference (float32 prices, shift(21)/shift(252)/
shift(273), 252-day std of pct_change, adv20 = 20-day mean of close*volume); tests/test_mom_sleeve.py
holds the parity test that re-runs the reference code on a fixture panel and compares.

Causality: signals are built from bars dated <= `signal_date` (the prior trading day) only.
"""
import logging
import re
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger('mom_sleeve_select')

PRICE_MIN = 10.0
ADV_CUT = 200_000_000.0
TOP_N = 20
MAXN = 40                      # ranked depth kept so a skipped (non-tradable) pick can be replaced
MIN_BARS = 274                 # reference `ok = close.shift(273).notna()` <=> >= 274 bars (273 prior bars)
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|\bPREFERRED\b|\bRIGHTS?\b',
                     re.I)
BAR_COLS = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']


def is_excluded(symbol: str, name: Optional[str]) -> bool:
    """True for a test ticker, a digit-leading CUSIP placeholder, or a name matching the reference
    word-boundary exclusion list (ETF/ETN/fund/trust/warrant/unit/preferred/right)."""
    if TEST_RE.match(symbol) or re.match(r'^[0-9]', symbol):
        return True
    return bool(NAME_RE.search(name or ''))


def clean_panel(bars: pd.DataFrame) -> pd.DataFrame:
    """Reference panel hygiene: float32 prices, dedupe (symbol, bar_date) keeping the last, drop rows with a
    non-positive open/high/low/close. Returns a new frame sorted by (symbol, bar_date)."""
    df = bars[BAR_COLS].copy()
    for c in ('open', 'high', 'low', 'close', 'volume'):
        df[c] = df[c].astype('float32')
    df['bar_date'] = pd.to_datetime(df['bar_date'])
    df = df.drop_duplicates(subset=['symbol', 'bar_date'], keep='last')
    df = df[~((df.open <= 0) | (df.high <= 0) | (df.low <= 0) | (df.close <= 0))]
    return df.sort_values(['symbol', 'bar_date']).reset_index(drop=True)


def compute_signal_table(bars: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
    """One row per symbol that has a bar ON `signal_date`: close, adv20, sigV2, ok (history) and the bar count.

    sigV2 = (close[t-21]/close[t-252] - 1) / std(daily returns, 252d ending t) -- the reference definition.
    Prefilters (last close >= PRICE_MIN, >= MIN_BARS bars) are necessary conditions of the universe, so
    they drop nothing the reference would keep.
    """
    df = clean_panel(bars)
    df = df[df.bar_date <= signal_date]
    last = df.groupby('symbol', sort=False).agg(last_date=('bar_date', 'max'), n=('bar_date', 'size'),
                                                 last_close=('close', 'last'))
    keep = last[(last.last_date == signal_date) & (last.n >= MIN_BARS) & (last.last_close >= PRICE_MIN)].index
    stale = int((last.last_date < signal_date).sum())
    logger.info("mom_sleeve_select: %d symbols with bars, %d stale (last bar before %s), %d pass price/history",
                len(last), stale, signal_date.date(), len(keep))
    df = df[df.symbol.isin(keep)].sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=['symbol', 'close', 'adv20', 'sigV2', 'ok', 'n'])
    g = df.groupby('symbol', sort=False)
    df['adv20'] = (df.close * df.volume).groupby(df.symbol).rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    c21, c252, c273 = g['close'].shift(21), g['close'].shift(252), g['close'].shift(273)
    df['sig12_1'] = c21 / c252 - 1
    df['ret1d'] = g['close'].pct_change()
    df['vol252'] = df.groupby('symbol', sort=False)['ret1d'].rolling(252, min_periods=252).std().reset_index(level=0, drop=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['sigV2'] = np.where(df.vol252 > 0, df.sig12_1 / df.vol252, np.nan)
    df['ok'] = c273.notna()
    out = df.groupby('symbol', sort=False).tail(1)
    out = out[out.bar_date == signal_date]
    out = out.merge(last[['n']], left_on='symbol', right_index=True)
    return out[['symbol', 'close', 'adv20', 'sigV2', 'ok', 'n']].reset_index(drop=True)


def rank_universe(table: pd.DataFrame, excluded_symbols: Iterable[str] = (), maxn: int = MAXN) -> pd.DataFrame:
    """Apply the reference universe gates (close >= $10, adv20 >= $200M, history ok, finite signal, not name-
    excluded) and return the top `maxn` by sigV2 with a 1-based `rank` column. `attrs['universe_size']` holds
    the number of symbols that passed the gates (the completeness-gate quantity)."""
    ex = set(excluded_symbols)
    u = table[(table.close >= PRICE_MIN) & table.ok & (table.adv20 >= ADV_CUT) & table.sigV2.notna()
              & ~table.symbol.isin(ex)]
    top = u.nlargest(maxn, 'sigV2').reset_index(drop=True)
    top['rank'] = np.arange(1, len(top) + 1)
    top.attrs['universe_size'] = int(len(u))
    return top


def pick_top(ranked: pd.DataFrame, is_eligible, n: int = TOP_N) -> (List[dict], List[dict]):
    """Walk the ranked list; take names for which `is_eligible(symbol) -> (bool, reason)` is True until `n`
    are taken. Returns (picks, skipped) as dict rows; every skip is logged WARNING with its reason."""
    picks, skipped = [], []
    for row in ranked.itertuples(index=False):
        if len(picks) >= n:
            break
        ok, reason = is_eligible(row.symbol)
        rec = dict(symbol=row.symbol, rank=int(row.rank), signal=float(row.sigV2), prior_close=float(row.close))
        if ok:
            picks.append(rec)
        else:
            rec['reason'] = reason
            skipped.append(rec)
            logger.warning("mom_sleeve_select: rank %d %s skipped (%s) -- taking the next rank",
                           rec['rank'], row.symbol, reason)
    if len(picks) < n:
        logger.error("mom_sleeve_select: only %d/%d picks available after eligibility skips", len(picks), n)
    return picks, skipped
