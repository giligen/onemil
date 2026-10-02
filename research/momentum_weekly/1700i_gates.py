#!/usr/bin/env python3
"""Cell 1,700i -- owner 10/2: "how can we know if now is the right time to continue with the strat? is
there a way to improve the losing/weak years?" PREREG: research/momentum_weekly/PREREG_1700i.md (FROZEN
2026-10-02 12:50 UTC).

Two books (U2 universe: price>=$10, ADV20>=$200M, point-in-time incl. delisted; weekly Monday rebalance,
top 20, equal weight; cost = 5bps/side + half the (high-low)/close spread proxy capped 20bps; no leverage
-- IDENTICAL construction to 1700g_vol.py's V1_N20 / V2_N20):
  A1  top 20 by 12-1 return (sig_12_1) -- the plain control.
  V2  top 20 by (12-1 return) / (252-day daily-return std) (sig_V2) -- 1700g's best construction
      (CAGR 27.4%, DD -42%, 5/10 years).

Reuses 1700g_vol.py's panel loader, U2 universe/name-exclusion filter, cost model, calendar builder,
ann_stats / ols_alpha_beta / window_read / by_year / hit_rate_null, and 1700e_regime.py's weekly
cash-switch `simulate` pattern -- COPIED, not imported (both source scripts execute their grid at module
scope, so `import` would re-run them as a side effect).

Three GATES ("is it the time" -- cash-switch overlays on the baseline "none" book, decided weekly from
data through the prior Friday; OFF = cash, a deliberate 0% that week):
  G1 the book's own momentum: ON when the BASELINE (gate-free) book's own trailing-126-TRADING-day
     compounded close-to-close return > SPY's trailing-126-day compounded return, both read at the
     signal date. Uses the baseline (un-gated) book so the gate cannot create a feedback loop on itself
     -- the same reason 1700g_vol.py's V3/V4 exposure is computed on the unscaled book (Barroso-Santa-
     Clara).
  G2 cross-sectional dispersion (Stivers & Sun: momentum pays in dispersed markets): evaluated WEEKLY
     (not daily -- the U2 pool is only materialised at the weekly signal dates in this codebase's
     `sig_by_date`), as the cross-sectional std, across that week's U2-eligible pool, of names' trailing
     21-trading-day ("monthly") returns; smoothed by a trailing 13-week mean; ON when that smoothed value
     is above its own trailing 156-week (3y) rolling median. 63 trading days ~= 13 weeks and 756 trading
     days ~= 156 weeks -- stated here since the PREREG's day-counts are implemented in week-units for
     this one gate, a deliberate choice, not an accident. Shared by both books (U2 membership doesn't
     depend on the ranking signal).
  G3 Daniel-Moskowitz crash guard: a trigger DAY is any day SPY's 252-day return < 0 AND SPY's 63-day
     realised vol (std of daily returns) is above its own trailing 756-day (3y) rolling median; the gate
     is OFF on any day that itself is a trigger OR falls within the 62 TRADING days after a trigger (a
     rolling "any trigger in the trailing 63 days" window) -- a mechanical reading of "OFF for 3 months
     after a trigger" that retriggers/extends for as long as the crash condition persists. Shared by both
     books (SPY-only).
All three default ON when history is insufficient (fail-safe -- never silently understates exposure),
logged WARNING with the count, matching 1700e_regime.py's F1-F3 precedent.

Three REPAIRS:
  R1 per-name stop: the only repair needing INTRA-week (daily) resolution -- gates and R2 stay on the
     weekly open-to-open engine. Each week's target book is the SAME top-20 ranking as "none". A name
     newly entering the book is bought at that day's OPEN (entry_price); a continuing holder's day-0 basis
     is the prior trading day's close (the week's own signal date). Every day held, if that day's LOW <=
     entry_price*0.80, the position is stopped: that day's contribution is capped at the stop level
     (stop_price / prior_basis - 1) and the position earns a flat 0% (cash) for the rest of its holding
     stint. At the NEXT rebalance the ranking is recomputed from scratch; a stopped name that still ranks
     top-20 re-enters as a FRESH buy (new entry_price, stop reset) -- "re-entry only at a later rebalance
     when it re-qualifies", per PREREG. A name the ranking drops is marked at the CLOSE of the last
     trading day before the next rebalance (not that rebalance's open) -- a one-day-earlier exit than the
     weekly engine's open-to-open convention, stated here since R1 is testing a daily risk control, not
     reproducing the weekly engine's exact exit timing; the effect is a same-direction overnight-gap
     difference, not a systematic bias. Cost is charged only on actual weekly buy/sell turnover (the same
     cost_rate_piv convention as the weekly engine) -- a stop is a mark against an already-open position,
     not a new trade, so it charges no separate cost; stated explicitly since it is the one place this
     script's cost treatment differs from the weekly engine.
  R2 index blend: post-hoc 50% "none" weekly net_ret + 50% SPY weekly return (no new engine -- a linear
     combination of two already-simulated series). The SPY leg's own rebalance cost is treated as zero
     (stated simplification); R2's cost/turnover = 0.5x "none"'s, consistent with only the book leg
     trading.
  R3 R1 + G1: the R1 daily engine with the target set forced empty on weeks the SAME baseline G1 series
     (as used for the standalone G1 cell) is OFF.
  R1/R3's hit-rate null reuses the WEEKLY (un-gated name-draw) null mechanism, not a re-simulation of the
     daily stop on each of the 300 random draws (computationally intractable at this budget) -- a stated,
     conservative approximation: the stop can only IMPROVE a draw's left tail, so the null is, if
     anything, too easy to beat. Flagged here and in RESULT.md, not hidden.

14 cells = 2 books x {none, G1, G2, G3, R1, R2, R3} (incl. the two "none" references). Reads: whole
2017-01-01..2026-09-30, H1 2017-01-01..2021-12-31, H2 2022-01-01..2026-09-30, by calendar year. Null: 300
same-N draws/period (1700g/e precedent, 10min budget). Pass bar (PREREG, exact, identical to 1700g):
years_beat_spy>=7/10 AND excess>0 both halves AND max_dd<=1.25x SPY's AND hit-rate null percentile>=95%.
Secondary (reported, not a pass): share of rolling 5-year monthly-start windows beating SPY. Tripwire
read: trailing-12-month (52-week) book-vs-SPY shortfall EPISODES (rising-edge, no double count) where the
book trailed by >25 points, and the counterfactual $ outcome of pausing to cash for the next 26 weeks
after each trigger. Multiplicity: 14 cells, stated here. Nothing tuned after seeing numbers.
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

logging.basicConfig(filename=str(OUT / '1700i.log'), filemode='w', level=logging.INFO,
                     format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700i')
log.addHandler(logging.StreamHandler(sys.stdout))

SEED = 170091
N_DRAWS = 300  # 1700g/e precedent -- 1,000 would exceed the 10min budget across 14 cells
TOP_N = 20
PRICE_MIN = 10.0
HIST_MIN_DAYS = 273
ADV_CUTOFF_U2 = 200_000_000.0
STOP_DD = 0.20
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


# =================================================================== load panel (copied from 1700g) ==
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
day_idx = {d: i for i, d in enumerate(trading_days)}

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
log.info('%s STEP 2: adv20, spread_proxy, 12-1 / risk-adjusted / 21d signals, history gate...', elapsed())
panel['dvol'] = panel['close'] * panel['volume']
panel['adv20'] = g['dvol'].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
panel['spread_proxy'] = (((panel['high'] - panel['low']) / panel['close']).clip(lower=0) * 0.1).clip(upper=0.002)
close_lag = {lag: (panel['close'] if lag == 0 else g['close'].shift(lag)) for lag in (0, 21, 252, 273)}
panel['sig_12_1'] = close_lag[21] / close_lag[252] - 1
panel['ret1d'] = panel.groupby('symbol', sort=False)['close'].pct_change()
panel['ret21'] = panel.groupby('symbol', sort=False)['close'].pct_change(21)
panel['vol252'] = panel.groupby('symbol', sort=False)['ret1d'].rolling(252, min_periods=252).std() \
    .reset_index(level=0, drop=True)
with np.errstate(divide='ignore', invalid='ignore'):
    panel['sig_V2'] = np.where(panel['vol252'] > 0, panel['sig_12_1'] / panel['vol252'], np.nan)
panel['history_ok'] = close_lag[273].notna()
panel = panel.drop(columns=['dvol'])
log.info('%s signals built; history_ok True for %d/%d rows', elapsed(), int(panel['history_ok'].sum()), len(panel))

# Slim daily OHL(low)C frame for R1/R3's intra-week stop engine and G1's own-book daily return -- kept
# AFTER `del panel` below; restricted to bar_date>=WIN_START (no book can hold before the sim starts).
daily_raw = panel.loc[panel.bar_date >= WIN_START, ['symbol', 'bar_date', 'open', 'close', 'low']].copy()

# ============================================================================== calendar (weekly) ==
cal = pd.Series(trading_days)


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

sig_cols = ['sig_12_1', 'sig_V2', 'ret21']
sig_rows = panel.loc[panel.bar_date.isin(signal_dates) & (panel['close'] >= PRICE_MIN) & panel['history_ok'],
                      ['symbol', 'bar_date', 'close', 'adv20'] + sig_cols]
sig_by_date = {d: sub.drop(columns='bar_date') for d, sub in sig_rows.groupby('bar_date')}
del panel, rows_at_dates, sig_rows
log.info('%s sig_by_date built: %d signal dates', elapsed(), len(sig_by_date))

# SPY over EVERY trading day -- trailing-126/252 return, 63d vol + its 3y median (G1, G3)
spy_close_all = spy_df.set_index('bar_date')['close'].reindex(trading_days)
spy_ret126 = spy_close_all / spy_close_all.shift(126) - 1
spy_ret252 = spy_close_all / spy_close_all.shift(252) - 1
spy_ret1d = spy_close_all.pct_change()
spy_vol63 = spy_ret1d.rolling(63, min_periods=63).std()
spy_vol63_med3y = spy_vol63.rolling(756, min_periods=504).median()


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
    """whole/H1/H2 read: ann return, excess, alpha/t, Sharpe, maxDD, turnover, cost drag, null pct,
    weeks-in-cash share, switch count (copied from 1700e_regime.py, requires an 'is_on' column)."""
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
    is_on = sub['is_on'].values if len(sub) else np.array([], dtype=bool)
    n_switch = int(np.sum(np.diff(is_on.astype(int)) != 0)) if len(is_on) > 1 else 0
    return dict(n_periods=net_stats['n_periods'], ann_return_net=net_stats['ann_return'],
                ann_return_spy=spy_stats['ann_return'], excess_ann_return=excess_stats['ann_return'],
                beta_spy=beta, ann_alpha_spy=ann_alpha, t_alpha=t_alpha, sharpe_net=net_stats['sharpe'],
                max_dd=net_stats['max_dd'], max_dd_spy=spy_stats['max_dd'],
                turnover_avg=float(np.nanmean(sub['turnover'])) if len(sub) else np.nan,
                cost_drag_annual=float(np.nanmean(sub['cost'])) * ppy if len(sub) else np.nan,
                green_share=net_stats['green_share'], null_pct_return=pct_ret, null_pct_excess=pct_excess,
                weeks_cash_share=float((~is_on).mean()) if len(is_on) else np.nan, n_switches=n_switch)


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


def hit_rate_null(real_df: pd.DataFrame, null_mat: np.ndarray) -> tuple[int, int, float]:
    """Count-matched null for the YEAR hit-rate (copied from 1700g_vol.py)."""
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


def tripwire_read(real_df: pd.DataFrame) -> dict:
    """Trailing-12-month (52-week) book-vs-SPY shortfall tripwire: at each week, compound the trailing 52
    weekly net_ret/spy_fwd; an EPISODE = a rising-edge week where book trails SPY by >25 points over that
    window (counted once per episode, not once per week inside it). Counterfactual: pause to cash (0%)
    for the 26 weeks AFTER each trigger week, compare total compounded return with vs without."""
    net = real_df['net_ret'].fillna(0).values
    spy = real_df['spy_fwd'].fillna(0).values
    n = len(net)
    shortfall = np.zeros(n, dtype=bool)
    for i in range(51, n):
        tb = np.prod(1 + net[i - 51:i + 1]) - 1
        ts = np.prod(1 + spy[i - 51:i + 1]) - 1
        shortfall[i] = (tb - ts) <= -0.25
    triggers = [i for i in range(n) if shortfall[i] and not shortfall[i - 1]] if n else []
    paused = net.copy()
    for t in triggers:
        end = min(n, t + 1 + 26)
        paused[t + 1:end] = 0.0
    return dict(n_triggers=len(triggers),
                trigger_weeks=[real_df.index[t].date().isoformat() for t in triggers],
                total_return_actual=float(np.prod(1 + net) - 1),
                total_return_paused=float(np.prod(1 + paused) - 1))


def rolling5y_read(real_df: pd.DataFrame) -> dict:
    """Secondary read (not a pass bar): share of rolling 5-year monthly-start windows beating SPY."""
    starts = pd.date_range(real_df.index.min(), real_df.index.max(), freq='MS')
    wins = beats = 0
    for s in starts:
        e = s + pd.DateOffset(years=5)
        if e > real_df.index.max():
            continue
        mask = (real_df.index >= s) & (real_df.index < e)
        if mask.sum() < 200:
            continue
        book_r = float(np.prod(1 + real_df.loc[mask, 'net_ret'].fillna(0)) - 1)
        spy_r = float(np.prod(1 + real_df.loc[mask, 'spy_fwd'].fillna(0)) - 1)
        wins += 1
        beats += int(book_r > spy_r)
    return dict(n_windows=wins, share_beat=(beats / wins if wins else np.nan))


# ============================================================================== pool builder ==
def build_pools(cal_df: pd.DataFrame, adv_cutoff: float, sig_col: str, maxn: int) -> list[dict]:
    """One row per week: top-maxn ranking by sig_col among the U2-eligible pool, plus the full eligible
    pool (for the null draws) and the week's forward returns/cost rates."""
    pools = []
    for _, wrow in cal_df.iterrows():
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
        pools.append(dict(sdate=sdate, edate=edate, nxt=nxt, ranked=ranked,
                           pool_syms=pool['symbol'].values if k else np.array([]), fwd_full=fwd_full,
                           spy_r=spy_r, cr=cr, k=k))
    return pools


def run_null_for_n(pools_: list[dict], n: int, rng, on_mask=None) -> np.ndarray:
    """N_DRAWS same-N random draws from each week's eligible pool (1700g/e precedent); on_mask (optional,
    aligned to pools_) forces a draw to 0% (deliberate cash) on OFF weeks, matching the real gate."""
    cols = []
    for i, p in enumerate(pools_):
        is_on = True if on_mask is None else bool(on_mask[i])
        if not is_on:
            cols.append(np.zeros(N_DRAWS))
            continue
        pool_syms, fwd_full = p['pool_syms'], p['fwd_full']
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


# ============================================================================ weekly gated engine ==
def simulate_weekly(pools_: list[dict], n: int, on_series, label: str) -> pd.DataFrame:
    """Weekly open-to-open engine with an optional precomputed cash-switch gate (on_series indexed by
    entry_date; None = always on). Copied from 1700e_regime.py's `simulate`, F4's path-dependent branch
    dropped (G1/G2/G3 here are precomputed series, not path-dependent on this book's own equity)."""
    prev_port: set = set()
    rows = []
    n_cash_fallback = 0
    for p in pools_:
        edate, ranked, fwd_full, spy_r, cr, k = p['edate'], p['ranked'], p['fwd_full'], p['spy_r'], p['cr'], p['k']
        is_on = True if on_series is None else bool(on_series.get(edate, True))
        port = set(ranked[:n]) if (is_on and k) else set()
        empty_pool_while_on = is_on and k == 0
        if empty_pool_while_on:
            n_cash_fallback += 1
        bought, sold = port - prev_port, prev_port - port
        if port:
            fwd = fwd_full.reindex(list(port))
            gross = float(fwd.fillna(0).mean())
        elif empty_pool_while_on:
            gross = np.nan
        else:
            gross = 0.0
        cost = 0.0
        if bought:
            cost += cr.reindex(list(bought)).fillna(0.002).sum() / max(len(port), 1)
        if sold:
            cost += cr.reindex(list(sold)).fillna(0.002).sum() / max(len(prev_port), 1)
        net = gross - cost if pd.notna(gross) else np.nan
        turnover = len(bought) / max(len(port), len(prev_port), 1)
        rows.append(dict(entry_date=edate, net_ret=net, gross_ret=gross, cost=cost, turnover=turnover,
                          spy_fwd=spy_r, is_on=is_on))
        prev_port = port
    if n_cash_fallback:
        log.warning('%s %s: %d ON periods had an empty eligible pool (k=0) -- NaN, dropped from '
                    'compounding (data gap, not a filter decision)', elapsed(), label, n_cash_fallback)
    return pd.DataFrame(rows).set_index('entry_date')


# ================================================================= R1/R3 daily per-name stop engine ==
def build_daily_arrays(pools_: list[dict], n: int):
    """Open/close/low numpy arrays (rows=trading_days, cols=symbols ever ranked top-n) + a symbol->col
    index map, for the R1/R3 intra-week stop engine. Small universe even though the source panel is
    ~390MB: only names that ever ranked top-n across the whole grid are included."""
    all_syms = sorted(set().union(*(set(p['ranked'][:n]) for p in pools_)))
    sub = daily_raw[daily_raw.symbol.isin(all_syms)]
    opn = sub.pivot_table(index='bar_date', columns='symbol', values='open', aggfunc='last').reindex(
        index=trading_days, columns=all_syms)
    cls = sub.pivot_table(index='bar_date', columns='symbol', values='close', aggfunc='last').reindex(
        index=trading_days, columns=all_syms)
    low = sub.pivot_table(index='bar_date', columns='symbol', values='low', aggfunc='last').reindex(
        index=trading_days, columns=all_syms)
    sym_idx = {s: i for i, s in enumerate(all_syms)}
    return opn.values, cls.values, low.values, sym_idx


def simulate_r1_daily(pools_: list[dict], n: int, on_series, opn_a, cls_a, low_a, sym_idx: dict,
                       label: str, stop_dd: float = STOP_DD) -> pd.DataFrame:
    """R1 (on_series=None) / R3 (on_series=G1 gate): daily-resolution per-name -20%-from-entry stop. See
    module docstring for the exact mechanism (entry at open, continuing-holder day-0 basis = prior close,
    stop freezes the position at 0% for the rest of its stint, re-entry only at the next rebalance)."""
    held: dict[str, dict] = {}
    prev_port: set = set()
    rows = []
    n_cash_fallback = 0
    for p in pools_:
        edate, ranked, spy_r, cr, k, nxt = p['edate'], p['ranked'], p['spy_r'], p['cr'], p['k'], p['nxt']
        is_on = True if on_series is None else bool(on_series.get(edate, True))
        target = set(ranked[:n]) if (is_on and k) else set()
        empty_pool_while_on = is_on and k == 0
        if empty_pool_while_on:
            n_cash_fallback += 1
        bought, sold = target - prev_port, prev_port - target
        for s in sold:
            held.pop(s, None)
        i0, i1 = day_idx[edate], day_idx[nxt]
        daily_rets = []
        for di in range(i0, i1):
            d = trading_days[di]
            if di == i0:
                for s in bought:
                    ci = sym_idx.get(s)
                    o = opn_a[di, ci] if ci is not None else np.nan
                    if pd.notna(o) and o > 0:
                        held[s] = dict(entry_price=float(o), stopped=False)
            day_rets = []
            for s in target:
                st = held.get(s)
                if st is None or st['stopped']:
                    day_rets.append(0.0)
                    continue
                ci = sym_idx.get(s)
                c_today = cls_a[di, ci] if ci is not None else np.nan
                l_today = low_a[di, ci] if ci is not None else np.nan
                if di == i0:
                    basis = st['entry_price'] if s in bought else (
                        cls_a[i0 - 1, ci] if (ci is not None and i0 > 0) else np.nan)
                else:
                    basis = cls_a[di - 1, ci] if ci is not None else np.nan
                if pd.isna(c_today) or pd.isna(basis) or basis <= 0:
                    day_rets.append(0.0)
                    continue
                stop_price = st['entry_price'] * (1 - stop_dd)
                if pd.notna(l_today) and l_today <= stop_price:
                    day_rets.append(stop_price / basis - 1)
                    st['stopped'] = True
                else:
                    day_rets.append(c_today / basis - 1)
            daily_rets.append(float(np.mean(day_rets)) if day_rets else 0.0)
        if daily_rets:
            gross = float(np.prod(1 + np.array(daily_rets)) - 1)
        else:
            gross = np.nan if empty_pool_while_on else 0.0
        cost = 0.0
        if bought:
            cost += cr.reindex(list(bought)).fillna(0.002).sum() / max(len(target), 1)
        if sold:
            cost += cr.reindex(list(sold)).fillna(0.002).sum() / max(len(prev_port), 1)
        net = gross - cost if pd.notna(gross) else np.nan
        turnover = len(bought) / max(len(target), len(prev_port), 1)
        rows.append(dict(entry_date=edate, net_ret=net, gross_ret=gross, cost=cost, turnover=turnover,
                          spy_fwd=spy_r, is_on=is_on))
        prev_port = target
    if n_cash_fallback:
        log.warning('%s %s: %d ON periods had an empty eligible pool (k=0)', elapsed(), label, n_cash_fallback)
    return pd.DataFrame(rows).set_index('entry_date')


def daily_book_return_series(port_syms: dict) -> pd.Series:
    """Equal-weight close-to-close daily return of a weekly-rebalanced book held flat between its own
    entry dates (no look-ahead). Copied from 1700g_vol.py, sourced from daily_raw. Used ONLY to build G1's
    trailing-126-day own-momentum read on the UNGATED baseline book (never the gated/repaired book, to
    avoid a feedback loop)."""
    entry_sorted = sorted(port_syms.keys())
    if not entry_sorted:
        return pd.Series(dtype=float)
    entry_set = set(entry_sorted)
    all_syms = sorted(set().union(*port_syms.values()))
    sub_piv = (daily_raw[daily_raw.symbol.isin(all_syms)]
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


def build_gate_g1(pools_: list[dict], n: int, label: str) -> pd.Series:
    """G1: ON when the BASELINE book's own trailing-126-trading-day compounded return > SPY's trailing
    126-day compounded return, read at each week's signal date. Defaults ON when either history is
    missing (fail-safe), logged WARNING with the count."""
    port_syms = {p['edate']: set(p['ranked'][:n]) for p in pools_}
    daily_ret = daily_book_return_series(port_syms)
    trail_book = (1 + daily_ret).rolling(126, min_periods=126).apply(np.prod, raw=True) - 1
    vals, missing = [], 0
    for p in pools_:
        sdate, edate = p['sdate'], p['edate']
        b = trail_book.get(sdate, np.nan)
        sidx = day_idx.get(sdate)
        s = spy_ret126.iloc[sidx] if sidx is not None else np.nan
        if pd.isna(b) or pd.isna(s):
            vals.append(True)
            missing += 1
        else:
            vals.append(bool(b > s))
    if missing:
        log.warning('%s G1[%s]: %d/%d weeks lacked 126d history -- defaulted ON', elapsed(), label, missing, len(pools_))
    return pd.Series(vals, index=[p['edate'] for p in pools_])


def build_gate_g2(pools_: list[dict], adv_cutoff: float) -> pd.Series:
    """G2: cross-sectional std of U2-eligible names' trailing 21d return, smoothed by a trailing 13-week
    mean, ON when above its own trailing 156-week (3y) rolling median. Weekly-frequency implementation of
    the PREREG's 63-day/3y windows (see module docstring). Shared across books -- built once from the
    U2 pool only (independent of the ranking signal)."""
    disp = []
    for p in pools_:
        sdate = p['sdate']
        day = sig_by_date.get(sdate)
        if day is None:
            disp.append(np.nan)
            continue
        m = (day['adv20'] >= adv_cutoff) & day['ret21'].notna()
        vals = day.loc[m, 'ret21']
        disp.append(float(vals.std(ddof=1)) if len(vals) >= 10 else np.nan)
    disp_s = pd.Series(disp, index=[p['edate'] for p in pools_])
    smoothed = disp_s.rolling(13, min_periods=13).mean()
    med3y = smoothed.rolling(156, min_periods=104).median()
    on_vals, missing = [], 0
    for edate in disp_s.index:
        sv, mv = smoothed.get(edate, np.nan), med3y.get(edate, np.nan)
        if pd.isna(sv) or pd.isna(mv):
            on_vals.append(True)
            missing += 1
        else:
            on_vals.append(bool(sv > mv))
    if missing:
        log.warning('%s G2: %d/%d weeks lacked 3y dispersion history -- defaulted ON', elapsed(), missing, len(pools_))
    return pd.Series(on_vals, index=disp_s.index)


def build_gate_g3(pools_: list[dict]) -> pd.Series:
    """G3 Daniel-Moskowitz crash guard: OFF on any trigger day (SPY 252d return<0 AND 63d vol > its 3y
    median) and for the 62 trading days after, else ON. SPY-only, shared across books."""
    trigger = (spy_ret252 < 0) & (spy_vol63 > spy_vol63_med3y)
    trigger = trigger.fillna(False)
    off_window = trigger.rolling(63, min_periods=1).max().astype(bool)
    missing = int((spy_ret252.isna() | spy_vol63.isna() | spy_vol63_med3y.isna()).reindex(
        [p['sdate'] for p in pools_]).fillna(True).sum())
    if missing:
        log.warning('%s G3: %d/%d weeks lacked 3y vol history -- defaulted ON (no trigger possible)',
                    elapsed(), missing, len(pools_))
    vals = [not bool(off_window.get(p['sdate'], False)) for p in pools_]
    return pd.Series(vals, index=[p['edate'] for p in pools_])


def blend_r2(none_df: pd.DataFrame) -> pd.DataFrame:
    """R2: 50% book + 50% SPY, rebalanced weekly -- a linear blend of the already-simulated 'none' book
    and SPY's own weekly return. Cost/turnover scaled by 0.5 (only the book leg trades); the SPY leg's
    rebalance cost is treated as zero (stated simplification, see module docstring)."""
    df = none_df.copy()
    spy = df['spy_fwd'].fillna(0)
    book = df['net_ret'].fillna(0)
    df['net_ret'] = 0.5 * book + 0.5 * spy
    df['gross_ret'] = 0.5 * df['gross_ret'].fillna(0) + 0.5 * spy
    df['cost'] = 0.5 * df['cost']
    df['turnover'] = 0.5 * df['turnover']
    df['is_on'] = True
    return df


# ===================================================================================== main grid ==
rng = np.random.default_rng(SEED)
PPY = 52
BOOKS = {'A1': 'sig_12_1', 'V2': 'sig_V2'}

cells: dict[str, pd.DataFrame] = {}
nulls: dict[str, np.ndarray] = {}
pools_by_book: dict[str, list] = {}

log.info('%s STEP 3: building pools + baseline ("none") for both books...', elapsed())
for bname, sig_col in BOOKS.items():
    pools = build_pools(weekly_cal, ADV_CUTOFF_U2, sig_col, TOP_N)
    pools_by_book[bname] = pools
    none_df = simulate_weekly(pools, TOP_N, None, f'{bname}_none')
    cells[f'{bname}_none'] = none_df
    nulls[f'{bname}_none'] = run_null_for_n(pools, TOP_N, rng)
    log.info('%s %s: %d weekly periods, baseline ann_net built', elapsed(), bname, len(pools))

log.info('%s STEP 4: building gates G1 (per book), G2/G3 (shared)...', elapsed())
g2_on = build_gate_g2(pools_by_book['A1'], ADV_CUTOFF_U2)   # U2 pool identical regardless of sig_col
g3_on = build_gate_g3(pools_by_book['A1'])                  # SPY-only
g1_on_by_book = {}
for bname in BOOKS:
    g1_on_by_book[bname] = build_gate_g1(pools_by_book[bname], TOP_N, bname)
log.info('%s STEP 4 done', elapsed())

log.info('%s STEP 5: simulating G1/G2/G3/R2 (weekly engine) for both books...', elapsed())
for bname in BOOKS:
    pools = pools_by_book[bname]
    g1_on = g1_on_by_book[bname]
    g1_mask = g1_on.reindex([p['edate'] for p in pools]).values
    g2_mask = g2_on.reindex([p['edate'] for p in pools]).values
    g3_mask = g3_on.reindex([p['edate'] for p in pools]).values
    cells[f'{bname}_G1'] = simulate_weekly(pools, TOP_N, g1_on, f'{bname}_G1')
    nulls[f'{bname}_G1'] = run_null_for_n(pools, TOP_N, rng, on_mask=g1_mask)
    cells[f'{bname}_G2'] = simulate_weekly(pools, TOP_N, g2_on, f'{bname}_G2')
    nulls[f'{bname}_G2'] = run_null_for_n(pools, TOP_N, rng, on_mask=g2_mask)
    cells[f'{bname}_G3'] = simulate_weekly(pools, TOP_N, g3_on, f'{bname}_G3')
    nulls[f'{bname}_G3'] = run_null_for_n(pools, TOP_N, rng, on_mask=g3_mask)
    cells[f'{bname}_R2'] = blend_r2(cells[f'{bname}_none'])
    spy_half = cells[f'{bname}_none']['spy_fwd'].reindex(range(len(pools))).values if False else None
    nulls[f'{bname}_R2'] = 0.5 * nulls[f'{bname}_none'] + 0.5 * np.nan_to_num(
        np.array([p['spy_r'] for p in pools]), nan=0.0)[np.newaxis, :]
    log.info('%s %s: G1/G2/G3/R2 done', elapsed(), bname)
log.info('%s STEP 5 done', elapsed())

log.info('%s STEP 6: R1/R3 daily stop engine (both books)...', elapsed())
for bname in BOOKS:
    pools = pools_by_book[bname]
    opn_a, cls_a, low_a, sym_idx = build_daily_arrays(pools, TOP_N)
    log.info('%s %s: daily arrays built, %d symbols', elapsed(), bname, len(sym_idx))
    cells[f'{bname}_R1'] = simulate_r1_daily(pools, TOP_N, None, opn_a, cls_a, low_a, sym_idx, f'{bname}_R1')
    nulls[f'{bname}_R1'] = nulls[f'{bname}_none']  # stated approximation -- see module docstring
    cells[f'{bname}_R3'] = simulate_r1_daily(pools, TOP_N, g1_on_by_book[bname], opn_a, cls_a, low_a, sym_idx,
                                              f'{bname}_R3')
    nulls[f'{bname}_R3'] = nulls[f'{bname}_G1']  # stated approximation -- see module docstring
    log.info('%s %s: R1/R3 done', elapsed(), bname)
log.info('%s STEP 6 done', elapsed())

CELL_ORDER = [f'{b}_{v}' for b in BOOKS for v in ('none', 'G1', 'G2', 'G3', 'R1', 'R2', 'R3')]
assert len(CELL_ORDER) == 14, f'expected 14 cells, got {len(CELL_ORDER)}'

log.info('%s STEP 7: window reads, by-year, hit-rate null, tripwire, rolling5y, pass flags...', elapsed())
cells_rows, summary_rows, by_year_rows = [], [], []
for label in CELL_ORDER:
    real_df, null_mat = cells[label], nulls[label]
    entry_dates = real_df.index.values
    win_reads = {}
    for win_name, (ws, we) in WINDOWS.items():
        mask = (entry_dates >= ws) & (entry_dates <= we)
        win_reads[win_name] = window_read(real_df, null_mat, mask, PPY)
        cells_rows.append(dict(cell=label, window=win_name, **win_reads[win_name]))
    whole, h1, h2 = win_reads['whole'], win_reads['halfA'], win_reads['halfB']
    byr = by_year(real_df)
    for _, r in byr.iterrows():
        by_year_rows.append(dict(cell=label, **r.to_dict()))
    real_hits, n_years, null_pct_hit = hit_rate_null(real_df, null_mat)
    worst = byr.loc[byr.book_return.idxmin()]
    tw = tripwire_read(real_df)
    r5 = rolling5y_read(real_df)
    pass_flag = bool(real_hits >= 7 and pd.notna(h1['excess_ann_return']) and h1['excess_ann_return'] > 0
                       and pd.notna(h2['excess_ann_return']) and h2['excess_ann_return'] > 0
                       and pd.notna(whole['max_dd']) and pd.notna(whole['max_dd_spy'])
                       and abs(whole['max_dd']) <= 1.25 * abs(whole['max_dd_spy'])
                       and null_pct_hit >= 95.0)
    summary_rows.append(dict(cell=label, years_beat_spy=real_hits, n_years=n_years, ann_net=whole['ann_return_net'],
        spy=whole['ann_return_spy'], excess_whole=whole['excess_ann_return'], excess_H1=h1['excess_ann_return'],
        excess_H2=h2['excess_ann_return'], alpha_t=whole['t_alpha'], sharpe=whole['sharpe_net'],
        maxDD_book=whole['max_dd'], maxDD_spy=whole['max_dd_spy'], worst_year=int(worst['year']),
        worst_year_ret=worst['book_return'], weeks_cash_share=whole['weeks_cash_share'],
        n_switches=whole['n_switches'], null_pct_hit=null_pct_hit, PASS=pass_flag,
        tripwire_n=tw['n_triggers'], tripwire_weeks=';'.join(tw['trigger_weeks']),
        tripwire_ret_actual=tw['total_return_actual'], tripwire_ret_paused=tw['total_return_paused'],
        roll5y_n=r5['n_windows'], roll5y_share_beat=r5['share_beat']))
    log.info('%s %s: years_beat_spy=%d/%d null_pct_hit=%.1f ann_net=%s spy=%s PASS=%s tripwire_n=%d', elapsed(),
              label, real_hits, n_years, null_pct_hit,
              f"{whole['ann_return_net']:.3f}" if pd.notna(whole['ann_return_net']) else 'NA',
              f"{whole['ann_return_spy']:.3f}" if pd.notna(whole['ann_return_spy']) else 'NA',
              pass_flag, tw['n_triggers'])
    pd.DataFrame(cells_rows).to_csv(OUT / '1700i_cells.csv', index=False)
    pd.DataFrame(by_year_rows).to_csv(OUT / '1700i_by_year.csv', index=False)

summary_df = pd.DataFrame(summary_rows)
log.info('%s STEP 7 done: %d cells summarised', elapsed(), len(summary_df))


# ================================================================================ RESULT.md ==
def fmt(v, pct=True, dp=1):
    if pd.isna(v):
        return 'NA'
    return f'{100*v:+.{dp}f}%' if pct else f'{v:.2f}'


passers = summary_df.loc[summary_df.PASS, 'cell'].tolist()
lines = []
lines.append('# RESULT 1,700i -- gates and weak-year repairs for the momentum book')
lines.append('')
lines.append(f'PREREG: PREREG_1700i.md. 14 cells (2 books x 7 variants incl. 2 "none" refs). '
             f'Pass bar: >=7/10 years beating SPY, excess>0 both halves, maxDD<=1.25x SPY, hit-rate null>=95%.')
lines.append(f'**PASS: {", ".join(passers) if passers else "NONE"}**')
lines.append('')
lines.append('R1/R3 null reuses the un-gated weekly name-draw null (not a daily-stop re-simulation on '
             'each of 300 draws) -- stated approximation, conservative (a stop can only help a draw\'s left tail).')
lines.append('')
lines.append('| cell | yrs beat SPY | CAGR | SPY CAGR | maxDD | SPY maxDD | worst yr | worst yr ret | '
             'null% | PASS |')
lines.append('|---|---|---|---|---|---|---|---|---|---|')
for _, r in summary_df.iterrows():
    lines.append(f"| {r['cell']} | {int(r['years_beat_spy'])}/{int(r['n_years'])} | {fmt(r['ann_net'])} | "
                 f"{fmt(r['spy'])} | {fmt(r['maxDD_book'])} | {fmt(r['maxDD_spy'])} | {int(r['worst_year'])} | "
                 f"{fmt(r['worst_year_ret'])} | {r['null_pct_hit']:.0f}% | {'PASS' if r['PASS'] else 'fail'} |")
lines.append('')
lines.append('## Tripwire read (trailing-12mo book-vs-SPY shortfall >25pts, rising-edge episodes; '
             '6-month-pause counterfactual)')
lines.append('| cell | n episodes | trigger weeks | actual total ret | paused-after-each total ret |')
lines.append('|---|---|---|---|---|')
for _, r in summary_df.iterrows():
    wk = r['tripwire_weeks'] if r['tripwire_weeks'] else '(none)'
    lines.append(f"| {r['cell']} | {int(r['tripwire_n'])} | {wk} | {fmt(r['tripwire_ret_actual'])} | "
                 f"{fmt(r['tripwire_ret_paused'])} |")
lines.append('')
lines.append('## Rolling 5-year windows beating SPY (secondary read, not a pass bar)')
lines.append('| cell | n windows | share beating SPY |')
lines.append('|---|---|---|')
for _, r in summary_df.iterrows():
    lines.append(f"| {r['cell']} | {int(r['roll5y_n'])} | {fmt(r['roll5y_share_beat'])} |")
lines.append('')
best_cell = summary_df.loc[summary_df['ann_net'].idxmax(), 'cell'] if summary_df['ann_net'].notna().any() else None
lines.append('## By-year $ (from $50K) -- the two best cells (by whole-window CAGR) vs SPY')
if best_cell:
    top2 = summary_df.nlargest(2, 'ann_net')['cell'].tolist()
    byyr_df = pd.DataFrame(by_year_rows)
    for c in top2:
        lines.append(f'### {c}')
        lines.append('| year | book $ | spy $ | excess |')
        lines.append('|---|---|---|---|')
        sub = byyr_df[byyr_df.cell == c]
        for _, r in sub.iterrows():
            lines.append(f"| {int(r['year'])} | ${r['book_dollars']:,.0f} | ${r['spy_dollars']:,.0f} | "
                         f"{fmt(r['excess'])} |")
lines.append('')
lines.append(f'Gate mechanisms: G1 book-own-momentum vs SPY trailing 126d; G2 cross-sectional dispersion '
             f'(63d, weekly-unit implementation, vs 3y median); G3 Daniel-Moskowitz crash guard (3mo after '
             f'a 252d-return-negative + elevated-63d-vol trigger). Repairs: R1 -20% per-name stop, daily '
             f'resolution, re-entry next rebalance; R2 50/50 SPY blend; R3 R1+G1. See module docstring for '
             f'exact mechanisms and stated simplifications (R2 SPY-leg cost=0; R1/R3 null approximation).')
result_text = '\n'.join(lines)
(OUT / 'RESULT_1700i.md').write_text(result_text + '\n')
log.info('%s RESULT_1700i.md written (%d lines)', elapsed(), len(lines))
log.info('%s DONE', elapsed())
