#!/usr/bin/env python3
"""Cell 1,700g -- owner 10/2: "I do want to beat SPY on most of the years." Nine cells testing whether
risk-adjusted ranking and/or volatility-scaled exposure turn the year-by-year HIT RATE (not CAGR) against
SPY, on top of the A1 book (U2 large caps, 12-1 momentum, weekly, equal weight) that only beats SPY in
6/10 years 2017-2026 (cell 1,700d/e).

PREREG: research/momentum_weekly/PREREG_1700g.md (FROZEN 2026-10-02 12:05 UTC).

Reuses research/momentum_weekly/1700d_grid.py's panel loader, U2 universe/name-exclusion filter, cost
model, calendar builder, ann_stats/ols_alpha_beta/window_read, and same-N null-draw machinery, plus
1700e_regime.py's pattern of a shared exogenous weekly multiplier on top of a selection (there: an on/off
cash switch; here: a continuous vol-targeted exposure) -- COPIED, not imported: both scripts execute their
full grid at module scope, so `import` would re-run them as a side effect.

Nine cells, all U2 (price>=$10, ADV20>=$200M, point-in-time incl. delisted), weekly Monday rebalance,
equal weight, cost = 5bps/side + half the (high-low)/close spread proxy (capped 20bps), no leverage:
  V1_N20  top 20 by 12-1 return (the A1 reference).
  V1_N50  top 50 by 12-1 return.
  V2_N20  top 20 by (12-1 return) / (252-day daily-return std) -- MSCI risk-adjusted construction.
          Ranking is invariant to annualising the vol (the same sqrt(252) divides every name), so the raw
          daily std is used for the rank.
  V5_N50  V2 ranking, top 50 (PREREG's "V2 with 50 names").
  V3_N20  V1_N20's OWN book, exposure = min(1, 20% / trailing-126-trading-day realised vol of V1_N20's
          own daily equal-weight close-to-close return, annualised). Vol is computed on the UNSCALED book
          (Barroso-Santa-Clara: scaling the already-scaled series would create a feedback loop). Weeks
          before 126 days of the book's own history exist default to exposure=1.0 (fail-safe, logged
          WARNING) -- unavoidable in the first ~5 months of 2017 given HIST_MIN_DAYS=273 ties the book's
          earliest start to the panel's 2016-01-04 start.
  V3_N50  same mechanism on V1_N50's own book.
  V4_N20  V2_N20's OWN book with the SAME exposure mechanism (V2 selection + V3 exposure).
  V6_N20  top 20 by the t-252..t-126 "12-7" intermediate-momentum return.
  V6_N50  top 50, same signal.
Cash (the un-exposed remainder for V3/V3_N50/V4) earns 0%. V1/V2/V5/V6 are always 100% exposed.

Reads: whole 2017-01-01..2026-09-30, H1 2017-01-01..2021-12-31, H2 2022-01-01..2026-09-30, by calendar
year 2017..2026 (2026 partial, through Sep). Null = 300 same-N draws/period (PREREG allows 1,000 only if
the whole run clears 10 minutes; 9 cells x ~500 weekly periods extrapolates from 1700d's measured
~0.0005s/period-draw to well over 10 minutes at 1,000 draws -- so 300 throughout, stated here and in
RESULT, matching 1700d/1700e's own precedent). Hit-rate null (PREREG's "Reads"): for the SAME 300 draws,
compound by calendar year and count years beating SPY; the real book's hit-rate percentile = share of
draws with hit-count <= the real count.

Pass bar (PREREG, exact): years beating SPY >= 7/10 AND excess > 0 in BOTH halves AND max DD <= 1.25x
SPY's AND hit-rate null percentile >= 95%. No parameter (vol target 20%, window 126d, N) is tuned after
seeing numbers.
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research' / 'momentum_weekly'

logging.basicConfig(filename=str(OUT / 'recon' / 'A_run.log'), filemode='w', level=logging.INFO,
                     format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700g')
log.addHandler(logging.StreamHandler(sys.stdout))

SEED = 17001
N_DRAWS = 300  # see module docstring -- 1,000 would exceed the PREREG's 10min budget across 9 weekly cells
PRICE_MIN = 10.0
HIST_MIN_DAYS = 273
ADV_CUTOFF_U2 = 200_000_000.0
VOL_TARGET = 0.20
VOL_WINDOW = 126
ANNUALIZER = np.sqrt(252)
WIN_START = pd.Timestamp('2017-01-01')
WIN_END = pd.Timestamp('2026-09-30')
HALF_CUT = pd.Timestamp('2022-01-01')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_EXCLUDE_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|'
                              r'\bPREFERRED\b|\bRIGHTS?\b', re.IGNORECASE)
WINDOWS = {'halfA': (WIN_START, HALF_CUT - pd.Timedelta(days=1)), 'halfB': (HALF_CUT, WIN_END),
           'whole': (WIN_START, WIN_END)}

t0 = time.time()


def elapsed() -> str:
    """Seconds since script start, for log lines."""
    return f'{time.time() - t0:6.0f}s'


# =================================================================== load panel (copied from 1700d) ==
panel_path = OUT / 'panel_2016_2026.parquet'
log.info('%s STEP 1: loading Alpaca panel %s', elapsed(), panel_path.name)
raw = pd.read_parquet(panel_path)
raw['symbol'] = raw['symbol'].astype('category')
for c in ('open', 'high', 'low', 'close'):
    raw[c] = raw[c].astype('float32')
raw['bar_date'] = pd.to_datetime(raw['bar_date'])
n_before = len(raw)
raw = raw.drop_duplicates(subset=['symbol', 'bar_date'], keep='last')
if len(raw) != n_before:
    log.warning('%s %d duplicate (symbol,bar_date) rows dropped', elapsed(), n_before - len(raw))
bad_px = (raw.open <= 0) | (raw.high <= 0) | (raw.low <= 0) | (raw.close <= 0)
if bad_px.any():
    log.warning('%s %d/%d rows non-positive OHLC -- dropped', elapsed(), int(bad_px.sum()), len(raw))
    raw = raw[~bad_px].reset_index(drop=True)
log.info('%s loaded: %d rows, %d symbols, %s..%s', elapsed(), len(raw), raw.symbol.nunique(),
          raw.bar_date.min().date(), raw.bar_date.max().date())

spy_df = raw.loc[raw.symbol == 'SPY', ['bar_date', 'open', 'close']].sort_values('bar_date').reset_index(drop=True)
if spy_df.empty:
    log.error('%s SPY missing -- abort', elapsed())
    raise SystemExit(1)
trading_days = sorted(spy_df['bar_date'].unique())

assets_df = pd.read_csv(OUT / '1700c_assets.csv', dtype={'symbol': str, 'name': str})
assets_df['name'] = assets_df['name'].fillna('')
excluded_by_name = set(assets_df.loc[assets_df['name'].str.contains(NAME_EXCLUDE_RE), 'symbol'])
excluded_by_test = {s for s in raw['symbol'].unique() if TEST_RE.match(s)}
excluded = excluded_by_name | excluded_by_test
log.info('%s universe exclusion: %d symbols (name-pattern + test tickers)', elapsed(), len(excluded))

panel = raw[(raw.symbol != 'SPY') & ~raw.symbol.isin(excluded)].copy()
panel['symbol'] = panel['symbol'].cat.remove_unused_categories()
del raw
panel = panel.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
g = panel.groupby('symbol', sort=False)
log.info('%s STEP 2: adv20, spread_proxy, 12-1 / 12-7 / risk-adjusted signals, history gate...', elapsed())
panel['dvol'] = panel['close'] * panel['volume']
panel['adv20'] = g['dvol'].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
panel['spread_proxy'] = (((panel['high'] - panel['low']) / panel['close']).clip(lower=0) * 0.1).clip(upper=0.002)
lag_needed = sorted({0, 21, 126, 252, 273})
close_lag = {lag: (panel['close'] if lag == 0 else g['close'].shift(lag)) for lag in lag_needed}
panel['sig_12_1'] = close_lag[21] / close_lag[252] - 1
panel['sig_12_7'] = close_lag[126] / close_lag[252] - 1
panel['ret1d'] = panel.groupby('symbol', sort=False)['close'].pct_change()
panel['vol252'] = panel.groupby('symbol', sort=False)['ret1d'].rolling(252, min_periods=252).std() \
    .reset_index(level=0, drop=True)
with np.errstate(divide='ignore', invalid='ignore'):
    panel['sig_V2'] = np.where(panel['vol252'] > 0, panel['sig_12_1'] / panel['vol252'], np.nan)
panel['history_ok'] = close_lag[273].notna()
panel = panel.drop(columns=['dvol'])
n_v2_nan = int((panel['history_ok'] & panel['sig_12_1'].notna() & panel['sig_V2'].isna()).sum())
if n_v2_nan:
    log.warning('%s %d history_ok rows have sig_12_1 but no sig_V2 yet (vol252 needs 252 ret1d obs)',
                elapsed(), n_v2_nan)
log.info('%s signals built; history_ok True for %d/%d rows', elapsed(), int(panel['history_ok'].sum()), len(panel))

# Slim long-format close table for the V3/V4 own-book daily-vol pivot (STEP 5) -- kept AFTER `del panel`
# below; restricted to bar_date>=WIN_START since no book can hold a position before the sim starts.
close_long = panel.loc[panel.bar_date >= WIN_START, ['symbol', 'bar_date', 'close']].copy()

# ============================================================================== calendar (weekly) ==
cal = pd.Series(trading_days)
day_idx = {d: i for i, d in enumerate(trading_days)}


def build_calendar(freq: str) -> pd.DataFrame:
    """First trading day of each week (Mon, or next trading day if Monday is a holiday), paired with
    the PRIOR trading day (signal date, no look-ahead) and the NEXT rebalance date."""
    period = cal.dt.to_period('W' if freq == 'weekly' else 'M')
    first = cal.groupby(period).min().sort_index()
    dates = first[(first >= WIN_START) & (first <= WIN_END)].reset_index(drop=True)
    prior_signal = [trading_days[day_idx[d] - 1] if day_idx[d] > 0 else pd.NaT for d in dates]
    df = pd.DataFrame({'entry_date': dates.values, 'prior_signal_date': prior_signal})
    df['next_entry_date'] = df['entry_date'].shift(-1)
    df = df.dropna(subset=['next_entry_date']).reset_index(drop=True)
    return df


weekly_cal = build_calendar('weekly')
log.info('%s weekly calendar: %d complete periods %s..%s', elapsed(), len(weekly_cal),
          weekly_cal.entry_date.min().date(), weekly_cal.entry_date.max().date())

all_dates = sorted(set(weekly_cal.entry_date) | set(weekly_cal.prior_signal_date) | set(weekly_cal.next_entry_date))
signal_dates = sorted(set(weekly_cal.prior_signal_date))

rows_at_dates = panel[panel.bar_date.isin(all_dates)]
open_piv = rows_at_dates.pivot_table(index='bar_date', columns='symbol', values='open', aggfunc='last').reindex(all_dates)
spread_piv = rows_at_dates.pivot_table(index='bar_date', columns='symbol', values='spread_proxy', aggfunc='last').reindex(all_dates)
cost_rate_piv = 0.0005 + 0.5 * spread_piv.fillna(0.002)
spy_idx = spy_df.set_index('bar_date').reindex(all_dates)
log.info('%s pivots built: open_piv shape %s', elapsed(), open_piv.shape)

sig_cols = ['sig_12_1', 'sig_12_7', 'sig_V2']
sig_rows = panel.loc[panel.bar_date.isin(signal_dates) & (panel['close'] >= PRICE_MIN) & panel['history_ok'],
                      ['symbol', 'bar_date', 'close', 'adv20'] + sig_cols]
sig_by_date = {d: sub.drop(columns='bar_date') for d, sub in sig_rows.groupby('bar_date')}
del panel, rows_at_dates, sig_rows
log.info('%s sig_by_date built: %d signal dates', elapsed(), len(sig_by_date))


def fwd_ret_row(entry_date, next_entry_date):
    return open_piv.loc[next_entry_date] / open_piv.loc[entry_date] - 1


def spy_fwd(entry_date, next_entry_date):
    a, b = spy_idx['open'].get(entry_date), spy_idx['open'].get(next_entry_date)
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return np.nan
    return b / a - 1


# ========================================================================= stats helpers (copied) ==
def ann_stats(returns, periods_per_year: int) -> dict:
    """Annualised return/vol/Sharpe/maxDD/green-share of a period-return series (weekly here)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return dict(n_periods=0, ann_return=np.nan, sharpe=np.nan, max_dd=np.nan, green_share=np.nan)
    comp = np.prod(1 + r)
    ann_return = comp ** (periods_per_year / n) - 1
    sd = np.std(r, ddof=1) if n > 1 else np.nan
    sharpe = (np.mean(r) / sd * np.sqrt(periods_per_year)) if (n > 1 and sd and sd > 0) else np.nan
    curve = np.cumprod(1 + r)
    max_dd = ((curve - np.maximum.accumulate(curve)) / np.maximum.accumulate(curve)).min()
    return dict(n_periods=n, ann_return=ann_return, sharpe=sharpe, max_dd=max_dd, green_share=float((r > 0).mean()))


def ols_alpha_beta(y, x):
    """OLS alpha/beta of book period-return y on SPY period-return x, with alpha's t-stat."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    valid = np.isfinite(y) & np.isfinite(x)
    y, x = y[valid], x[valid]
    n = len(y)
    if n < 3 or np.var(x) == 0:
        return np.nan, np.nan, np.nan, n
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coef
    resid = y - X @ coef
    dof = n - 2
    sigma2 = (resid @ resid) / dof if dof > 0 else np.nan
    xtx_inv = np.linalg.inv(X.T @ X)
    se_alpha = np.sqrt(sigma2 * xtx_inv[0, 0]) if pd.notna(sigma2) else np.nan
    t_alpha = alpha / se_alpha if se_alpha and se_alpha > 0 else np.nan
    return alpha, beta, t_alpha, n


def window_read(real_df: pd.DataFrame, null_mat: np.ndarray, mask: np.ndarray, ppy: int) -> dict:
    """whole/H1/H2 read: ann return, excess, alpha/t, Sharpe, maxDD, turnover, cost drag, null pct."""
    sub = real_df.loc[mask]
    spy_sub = sub['spy_fwd'].values
    net_stats = ann_stats(sub['net_ret'].values, ppy)
    spy_stats = ann_stats(spy_sub, ppy)
    valid = np.isfinite(sub['net_ret'].values) & np.isfinite(spy_sub)
    excess_geo = (1 + sub['net_ret'].values[valid]) / (1 + spy_sub[valid]) - 1 if valid.any() else np.array([])
    excess_stats = ann_stats(excess_geo, ppy)
    alpha, beta, t_alpha, _ = ols_alpha_beta(sub['net_ret'].values, spy_sub)
    ann_alpha = alpha * ppy if pd.notna(alpha) else np.nan
    null_sub = null_mat[:, mask]
    null_ann_ret = np.array([ann_stats(null_sub[i], ppy)['ann_return'] for i in range(null_sub.shape[0])])
    null_excess = (1 + null_sub) / (1 + np.where(np.isfinite(spy_sub), spy_sub, 0.0)) - 1
    null_excess_ann = np.array([ann_stats(null_excess[i], ppy)['ann_return'] for i in range(null_sub.shape[0])])
    pct_ret = 100 * np.nanmean(null_ann_ret <= net_stats['ann_return']) if pd.notna(net_stats['ann_return']) else np.nan
    pct_excess = 100 * np.nanmean(null_excess_ann <= excess_stats['ann_return']) if pd.notna(excess_stats['ann_return']) else np.nan
    return dict(n_periods=net_stats['n_periods'], ann_return_net=net_stats['ann_return'],
                ann_return_spy=spy_stats['ann_return'], excess_ann_return=excess_stats['ann_return'],
                beta_spy=beta, ann_alpha_spy=ann_alpha, t_alpha=t_alpha, sharpe_net=net_stats['sharpe'],
                max_dd=net_stats['max_dd'], max_dd_spy=spy_stats['max_dd'],
                turnover_avg=float(np.nanmean(sub['turnover'])) if len(sub) else np.nan,
                cost_drag_annual=float(np.nanmean(sub['cost'])) * ppy if len(sub) else np.nan,
                green_share=net_stats['green_share'], null_pct_return=pct_ret, null_pct_excess=pct_excess)


# ============================================================================== selection engine ==
def run_cell(cal_df_: pd.DataFrame, adv_cutoff: float, sig_col: str, n_list: list[int]):
    """One pass over the weekly calendar: eligible-pool ranking computed ONCE per period on sig_col,
    top-N sliced for every N in n_list from that ranking (1700d_grid.py's run_cell, extended to also
    capture the held-symbol set per period -- needed downstream for the V3/V4 own-book daily vol)."""
    maxn = max(n_list)
    prev_port = {n: set() for n in n_list}
    rows = {n: [] for n in n_list}
    port_syms = {n: {} for n in n_list}
    k_lt_n = {n: 0 for n in n_list}
    pools_for_null = []
    for _, wrow in cal_df_.iterrows():
        sdate, edate, nxt = wrow['prior_signal_date'], wrow['entry_date'], wrow['next_entry_date']
        day = sig_by_date.get(sdate)
        if day is not None:
            m = (day['adv20'] >= adv_cutoff) & day[sig_col].notna()
            pool = day.loc[m, ['symbol', sig_col]]
        else:
            pool = pd.DataFrame(columns=['symbol', sig_col])
        k = len(pool)
        ranked = pool.nlargest(min(k, maxn), sig_col)['symbol'].tolist() if k else []
        fwd_full = fwd_ret_row(edate, nxt) if k else None
        spy_r = spy_fwd(edate, nxt)
        cr = cost_rate_piv.loc[edate] if k else None
        pools_for_null.append((pool['symbol'].values if k else np.array([]), fwd_full))
        for n in n_list:
            if k < n:
                k_lt_n[n] += 1
            port = set(ranked[:n])
            bought, sold = port - prev_port[n], prev_port[n] - port
            if port:
                fwd = fwd_full.reindex(list(port))
                n_missing = int(fwd.isna().sum())
                gross = fwd.fillna(0).mean()
            else:
                n_missing, gross = 0, np.nan
            denom = len(port) if port else 1
            cost = 0.0
            if bought or sold:
                cost = (cr.reindex(list(bought)).fillna(0.002).sum() + cr.reindex(list(sold)).fillna(0.002).sum()) / denom
            turnover = len(bought) / denom
            rows[n].append(dict(entry_date=edate, n_eligible=k, port_n=len(port), gross_ret=gross, cost=cost,
                                  net_ret=(gross - cost if pd.notna(gross) else np.nan), turnover=turnover,
                                  n_fwd_missing=n_missing, spy_fwd=spy_r))
            port_syms[n][edate] = port
            prev_port[n] = port
    real_by_n = {n: pd.DataFrame(rows[n]).set_index('entry_date') for n in n_list}
    for n in n_list:
        if k_lt_n[n]:
            log.warning('%s sig=%s N=%d: %d/%d periods had fewer than %d eligible names', elapsed(),
                        sig_col, n, k_lt_n[n], len(cal_df_), n)
    return real_by_n, pools_for_null, port_syms


def run_null_for_n(pools_for_null, n, rng) -> np.ndarray:
    """N_DRAWS random same-N draws from the same eligible pool each period (shape: N_DRAWS x periods)."""
    cols = []
    for pool_syms, fwd_full in pools_for_null:
        k = len(pool_syms)
        if k == 0 or fwd_full is None:
            cols.append(np.full(N_DRAWS, np.nan))
            continue
        rets = fwd_full.reindex(pool_syms).fillna(0).values
        nn = min(n, k)
        rand = rng.random((N_DRAWS, k))
        idx = np.argpartition(rand, nn - 1, axis=1)[:, :nn]
        cols.append(rets[idx].mean(axis=1))
    return np.column_stack(cols)


# ======================================================================= own-book daily vol (V3/V4) ==
def daily_book_return_series(port_syms: dict) -> pd.Series:
    """Equal-weight close-to-close daily return of a weekly-rebalanced book, held flat between its own
    entry dates (no look-ahead: day d's return only uses closes through day d). port_syms: entry_date ->
    held symbol set. Builds a small pivot restricted to the union of symbols ever held, for speed."""
    entry_sorted = sorted(port_syms.keys())
    if not entry_sorted:
        return pd.Series(dtype=float)
    entry_set = set(entry_sorted)
    all_syms = sorted(set().union(*port_syms.values()))
    sub_piv = (close_long[close_long.symbol.isin(all_syms)]
               .pivot_table(index='bar_date', columns='symbol', values='close', aggfunc='last')
               .reindex(columns=all_syms))
    days = [d for d in trading_days if d >= entry_sorted[0]]
    current_port: list[str] = []
    prev_row = None
    out = []
    for d in days:
        if d in entry_set:
            current_port = list(port_syms[d])
        row = sub_piv.loc[d] if d in sub_piv.index else None
        if prev_row is not None and current_port and row is not None:
            c0 = prev_row.reindex(current_port)
            c1 = row.reindex(current_port)
            valid = c0.notna() & c1.notna() & (c0 > 0)
            r = float((c1[valid] / c0[valid] - 1).mean()) if valid.any() else np.nan
        else:
            r = np.nan
        out.append(r)
        prev_row = row
    return pd.Series(out, index=days)


def exposure_from_own_vol(port_syms: dict, cal_df_: pd.DataFrame, label: str) -> pd.Series:
    """Barroso-Santa-Clara vol-target: exposure = min(1, 20% / trailing-126-day realised vol of the
    book's OWN unscaled daily return), evaluated at each week's prior_signal_date (no look-ahead). Fails
    safe to 100% exposure (never silently understates) when the 126-day history isn't ready yet; logs a
    WARNING with the count."""
    daily_ret = daily_book_return_series(port_syms)
    vol = daily_ret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std(ddof=1)
    exp = {}
    fallback_n = 0
    for _, wrow in cal_df_.iterrows():
        sdate, edate = wrow['prior_signal_date'], wrow['entry_date']
        v = vol.get(sdate, np.nan)
        if pd.isna(v) or v <= 0:
            exp[edate] = 1.0
            fallback_n += 1
        else:
            exp[edate] = float(min(1.0, VOL_TARGET / (v * ANNUALIZER)))
    if fallback_n:
        log.warning('%s exposure[%s]: %d/%d weeks fell back to 100%% exposure (no 126-day own-vol history yet)',
                    elapsed(), label, fallback_n, len(cal_df_))
    return pd.Series(exp)


def apply_exposure(real_df: pd.DataFrame, exposure: pd.Series) -> pd.DataFrame:
    """Scale a book's gross/net weekly return by its exposure series; the un-exposed remainder is cash
    at 0% (net_ret_scaled = exposure * net_ret, so cost -- already inside net_ret -- scales with it too,
    matching only the invested fraction being traded)."""
    df = real_df.copy()
    exp = exposure.reindex(df.index).fillna(1.0)
    df['exposure'] = exp.values
    df['gross_ret'] = df['gross_ret'] * exp.values
    df['net_ret'] = df['net_ret'] * exp.values
    return df


def scale_null(null_mat: np.ndarray, exposure: pd.Series, entry_dates) -> np.ndarray:
    """Scale a null matrix (N_DRAWS x periods) by the SAME per-period exposure as the real book (the
    exposure decision is exogenous to which names were drawn, mirroring 1700e_regime.py's cash-switch
    null treatment)."""
    exp_arr = exposure.reindex(entry_dates).fillna(1.0).values
    return null_mat * exp_arr[np.newaxis, :]


def hit_rate_null(real_df: pd.DataFrame, null_mat: np.ndarray) -> tuple[int, int, float]:
    """Count-matched null for the YEAR hit-rate: for each of the N_DRAWS same-N draws, compound within
    each calendar year and count years beating SPY's (deterministic) year return; percentile = share of
    draws with hit-count <= the real book's hit-count."""
    years = real_df.index.year.values
    uniq_years = sorted(set(years))
    null_hits = np.zeros(null_mat.shape[0], dtype=int)
    real_hits = 0
    for yr in uniq_years:
        m = years == yr
        book_yr = float(np.nanprod(1 + real_df['net_ret'].values[m]) - 1) if m.any() else np.nan
        spy_yr = float(np.nanprod(1 + np.nan_to_num(real_df['spy_fwd'].values[m], nan=0.0)) - 1) if m.any() else np.nan
        if pd.notna(book_yr) and pd.notna(spy_yr) and book_yr > spy_yr:
            real_hits += 1
        draw_yr = np.nanprod(1 + np.nan_to_num(null_mat[:, m], nan=0.0), axis=1) - 1
        null_hits += (draw_yr > spy_yr).astype(int)
    pct = 100.0 * float(np.mean(null_hits <= real_hits))
    return real_hits, len(uniq_years), pct


def by_year(real_df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-year book/SPY compounded return + $ balance compounding continuously from $50K."""
    rows = []
    bal_book, bal_spy = 50_000.0, 50_000.0
    for yr, sub in real_df.groupby(real_df.index.year):
        book_r = float(np.nanprod(1 + sub['net_ret'].fillna(0)) - 1)
        spy_r = float(np.nanprod(1 + sub['spy_fwd'].fillna(0)) - 1)
        bal_book *= (1 + book_r)
        bal_spy *= (1 + spy_r)
        rows.append(dict(year=int(yr), n_periods=len(sub), book_return=book_r, spy_return=spy_r,
                          excess=book_r - spy_r, book_dollars=bal_book, spy_dollars=bal_spy))
    return pd.DataFrame(rows)


real_B, pools_B, ports_B = run_cell(weekly_cal, ADV_CUTOFF_U2, 'sig_V2', [20])
rows=[]
for e,syms in ports_B[20].items():
    nxt=weekly_cal.loc[weekly_cal.entry_date==e,'next_entry_date'].iloc[0]
    fw=fwd_ret_row(e,nxt)
    sd=weekly_cal.loc[weekly_cal.entry_date==e,'prior_signal_date'].iloc[0]
    if sd not in sig_by_date: continue
    day=sig_by_date[sd].set_index('symbol')
    for s in syms:
        rows.append(dict(rebalance_date=e,symbol=s,weight=1/len(syms),signal=day.loc[s,'sig_V2'],entry_open=open_piv.loc[e,s],next_open=open_piv.loc[nxt,s],wk_ret=fw.get(s)))
pd.DataFrame(rows).to_csv(str(OUT/'recon'/'A_holdings.csv'),index=False)
real_B[20].to_csv(str(OUT/'recon'/'A_weekly.csv'))
print('DONE',flush=True)
