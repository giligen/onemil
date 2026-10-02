"""Weekly risk-adjusted 12-1 momentum sleeve -- pure selection / sizing functions (no I/O).

Reproduces build A of the reconciled reference (research/momentum_weekly/recon/A_dump.py, cell V2_N20):
  * universe at signal date t (the prior trading day): close >= $10, 20-row average dollar volume
    >= $200M, history gate (a close 273 rows back exists), name exclusions with WORD-BOUNDARY matching,
    test tickers ``^Z[A-Z]ZZT$`` removed;
  * signal = (close[t-21] / close[t-252] - 1) / std(daily returns over the 252 rows ending t);
  * top 20; every Monday ALL names are reset to 1/20 of the sleeve equity.

The lags are counted in ROWS of each symbol's own bar history (as build A's groupby shift/rolling do).
``panel`` is a long DataFrame with columns symbol, bar_date, close, volume (open/high/low optional).
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRICE_MIN = 10.0
ADV_CUTOFF = 200_000_000.0
HIST_MIN_ROWS_BACK = 273      # build A: close_lag[273] must exist
LAG_SKIP = 21                 # the "-1" month
LAG_FORMATION = 252           # the "12" months
ADV_WINDOW = 20
VOL_WINDOW = 252
DEFAULT_N = 20

TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_EXCLUDE_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|'
                             r'\bPREFERRED\b|\bRIGHTS?\b', re.IGNORECASE)


def excluded_symbols(assets: pd.DataFrame, symbols: Optional[Iterable[str]] = None) -> set:
    """Symbols removed from the universe: name matches NAME_EXCLUDE_RE (word-boundary), or test ticker.

    ``assets`` has columns symbol, name. Missing names are treated as '' (never excluded by name).
    """
    names = assets['name'].fillna('').astype(str)
    by_name = set(assets.loc[names.str.contains(NAME_EXCLUDE_RE), 'symbol'])
    pool = symbols if symbols is not None else assets['symbol']
    by_test = {s for s in pool if TEST_RE.match(str(s))}
    return by_name | by_test


def risk_adjusted_momentum(panel: pd.DataFrame, asof) -> pd.DataFrame:
    """Per-symbol features at ``asof`` using only rows with bar_date <= asof.

    Returns a DataFrame indexed by symbol with columns close, adv20, sig_12_1, vol252, signal,
    history_ok. Only symbols that have a bar ON ``asof`` are returned (a stale symbol is not tradable
    this week). A symbol lacking the rows needed has NaN in the affected columns.
    """
    asof = pd.Timestamp(asof)
    sub = panel.loc[panel['bar_date'] <= asof, ['symbol', 'bar_date', 'close', 'volume']]
    sub = sub.sort_values(['symbol', 'bar_date'], kind='stable')
    sub = sub.drop_duplicates(['symbol', 'bar_date'], keep='last')
    grp = sub.groupby('symbol', sort=False, observed=True)
    sub = sub.assign(k=grp.cumcount(ascending=False))          # 0 = latest row of the symbol
    last = sub[sub['k'] == 0]
    live = set(last.loc[last['bar_date'] == asof, 'symbol'])
    if not live:
        logger.error("momentum_sleeve: no symbol has a bar on %s -- empty signal", asof.date())
        return pd.DataFrame(columns=['close', 'adv20', 'sig_12_1', 'vol252', 'signal', 'history_ok'])
    sub = sub[sub['symbol'].isin(live)]
    # float32 closes -> float32 pct_change, as build A (keeps tie-breaks identical)
    sub = sub.assign(ret=sub.groupby('symbol', sort=False, observed=True)['close'].pct_change(),
                     dvol=sub['close'].astype('float64') * sub['volume'].astype('float64'))

    def at(lag: int) -> pd.Series:
        s = sub[sub['k'] == lag]
        return s.set_index('symbol')['close'].astype('float64')

    close0, c21, c252, c273 = at(0), at(LAG_SKIP), at(LAG_FORMATION), at(HIST_MIN_ROWS_BACK)
    win_a = sub[sub['k'] < ADV_WINDOW].groupby('symbol', observed=True)['dvol'].agg(['mean', 'count'])
    adv20 = win_a['mean'].where(win_a['count'] == ADV_WINDOW)
    win_v = sub[sub['k'] < VOL_WINDOW].groupby('symbol', observed=True)['ret'].agg(['std', 'count'])
    vol252 = win_v['std'].where(win_v['count'] == VOL_WINDOW)
    out = pd.DataFrame({'close': close0})
    out['adv20'] = adv20
    out['sig_12_1'] = c21 / c252 - 1.0
    out['vol252'] = vol252
    with np.errstate(divide='ignore', invalid='ignore'):
        out['signal'] = np.where(out['vol252'] > 0, out['sig_12_1'] / out['vol252'], np.nan)
    out['history_ok'] = c273.reindex(out.index).notna()
    return out


def eligible_universe(panel: pd.DataFrame, asof, assets: pd.DataFrame) -> List[str]:
    """Symbols passing every universe gate at ``asof`` (sorted).

    Gates: bar on asof, close >= $10, adv20 >= $200M, history_ok, a valid signal, not excluded by
    name / test-ticker. ``assets`` has columns symbol, name.
    """
    feat = risk_adjusted_momentum(panel, asof)
    if feat.empty:
        return []
    excl = excluded_symbols(assets, feat.index)
    m = ((feat['close'] >= PRICE_MIN) & (feat['adv20'] >= ADV_CUTOFF) & feat['history_ok']
         & feat['signal'].notna() & ~feat.index.isin(excl))
    return sorted(feat.index[m])


def select_top(signal: pd.Series, n: int = DEFAULT_N) -> List[str]:
    """Top ``n`` symbols by signal, descending (NaN dropped; ties broken by symbol for determinism).

    Fewer than ``n`` eligible names returns what exists and logs a WARNING (the sleeve then holds
    fewer, equal-weighted at 1/n of equity so the unfilled slots stay in cash).
    """
    s = signal.dropna()
    if len(s) < n:
        logger.warning("momentum_sleeve: only %d eligible names for n=%d -- holding fewer", len(s), n)
    ranked = s.reset_index()
    ranked.columns = ['symbol', 'signal']
    ranked = ranked.sort_values(['signal', 'symbol'], ascending=[False, True], kind='stable')
    return ranked['symbol'].head(n).tolist()


def target_weights(selected: List[str], n: int = DEFAULT_N) -> Dict[str, float]:
    """Equal weights 1/n per selected name (unfilled slots are cash)."""
    return {s: 1.0 / n for s in selected}


def target_dollars(selected: List[str], equity: float, n: int = DEFAULT_N) -> Dict[str, float]:
    """Dollar target per name = equity / n."""
    return {s: w * equity for s, w in target_weights(selected, n).items()}


def rebalance_orders(current_positions_usd: Dict[str, float], targets_usd: Dict[str, float],
                     min_trade_usd: float = 5.0) -> List[dict]:
    """Notional deltas that reset every name to its target; SELLS FIRST, then buys.

    A name absent from targets is sold in full (``full_exit`` True). Deltas smaller than
    ``min_trade_usd`` are skipped (except full exits of any positive size). Each row:
    {symbol, side ('sell'|'buy'), notional (positive USD), full_exit}.
    """
    sells, buys = [], []
    for sym in sorted(set(current_positions_usd) | set(targets_usd)):
        cur = float(current_positions_usd.get(sym, 0.0))
        tgt = float(targets_usd.get(sym, 0.0))
        delta = tgt - cur
        leaving = sym not in targets_usd or tgt <= 0
        if leaving and cur > 0:
            sells.append({'symbol': sym, 'side': 'sell', 'notional': cur, 'full_exit': True})
        elif delta <= -min_trade_usd:
            sells.append({'symbol': sym, 'side': 'sell', 'notional': -delta, 'full_exit': False})
        elif delta >= min_trade_usd:
            buys.append({'symbol': sym, 'side': 'buy', 'notional': delta, 'full_exit': False})
    return sells + buys
