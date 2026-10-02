#!/usr/bin/env python3
"""Cell 1,700e -- owner: "can we discover the bearish market and exit?" Six regime filters (F0..F5)
applied as a cash-switch on top of cell 1,700d's "A1" book (U2 large caps, 12-1 momentum, weekly
rebalance, equal weight), each also run on the top-10 and top-50 books (same universe) = 18 cells.

PREREG: research/momentum_weekly/PREREG_1700e.md (FROZEN 2026-10-02 11:35 UTC).

Reuses research/momentum_weekly/1700d_grid.py's panel loader, universe/name-exclusion filter, signal
construction, calendar builder, cost model, ann_stats/ols_alpha_beta, and null-draw machinery -- COPIED
(not imported: 1700d_grid.py executes its full 72-cell grid at module scope, so `import` would re-run
that grid as a side effect). Trimmed to the single (universe=U2, lookback=12-1, rebalance=weekly) book
this PREREG tests, since that is "A1".

Regime filters, all evaluated with NO look-ahead (prior Friday close, or prior month-end close for F3):
  F0 none (reference, always ON).
  F1 SPY close > SPY 200-day SMA.
  F2 absolute momentum: SPY 252-day return > 0.
  F3 Faber: SPY close > 10-month SMA at month-end; the state decided at month-end M is held for the
     WHOLE of calendar month M+1 (standard Faber/GTAA timing -- a month-end signal cannot apply to
     weeks that already happened within the same month).
  F4 book drawdown kill: OFF after a 20% drawdown from the FILTERED book's own net-of-cost equity peak;
     cleared only when SPY closes back above its 200-day SMA (never by the book's own recovery). This
     is path-dependent per N (each book has its own equity curve), so it is simulated in chronological
     order inside simulate(), not precomputed like F1/F2/F3/F5.
  F5 F1 with a 2% hysteresis band (ON above SMA*1.02, OFF below SMA*0.98, else hold prior state).
When OFF the book is 100% cash (0% return that week); re-entry always buys the FRESH top-N ranking at
the next ON week, never a stale list (ranking is recomputed every week regardless of on/off state).

Early periods lacking enough history for a filter's moving average/return default to ON (not OFF) and
are logged as a WARNING with a count -- a fail-safe that never silently understates the unfiltered book.

Cost-model note (deliberate, not a silent rewrite): 1700d_grid.py's turnover cost divided the whole
bought+sold notional by a single `denom = len(new_port) or 1`. That is exactly right whenever portfolio
size is constant (true for F0 always, and for every ON-steady week here), but blows up on a full
liquidation or full re-entry (denom=1 while n names trade). Here each leg is divided by its OWN
portfolio's size (new port for buys, prior port for sells), which is algebraically identical to the
1700d formula in the constant-size case and the correct generalisation when the book goes to/from cash.

Null: 300 draws/period (PREREG target 1,000, reduced for runtime -- same justification as 1700d, same
order of magnitude of periods x pool size). For an OFF period the null column is 0 for every draw
(cash, matched to the real book) so the percentile isolates stock-selection skill from the shared,
filter-driven regime timing; for an ON period it is the same same-N random draw from the eligible pool
as 1700d.
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

logging.basicConfig(filename=str(OUT / '1700e.log'), filemode='w', level=logging.INFO,
                     format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700e')
log.addHandler(logging.StreamHandler(sys.stdout))

SEED = 17001
N_DRAWS = 300
PRICE_MIN = 10.0
WIN_START = pd.Timestamp('2016-01-01')
WIN_END = pd.Timestamp('2026-09-30')
HALF_CUT = pd.Timestamp('2021-07-01')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_EXCLUDE_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|'
                              r'\bPREFERRED\b|\bRIGHTS?\b', re.IGNORECASE)

ADV_CUTOFF_U2 = 200_000_000.0
N_LIST = [10, 20, 50]
MAXN = max(N_LIST)
LAG_FAR, LAG_NEAR = 252, 21  # 12-1 momentum, same as 1700d's L2_12_1
WINDOWS = {'halfA': (WIN_START, HALF_CUT - pd.Timedelta(days=1)), 'halfB': (HALF_CUT, WIN_END),
           'whole': (WIN_START, WIN_END)}
SUBWINDOWS = {'2020-02..04': (pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-30')),
              '2021': (pd.Timestamp('2021-01-01'), pd.Timestamp('2021-12-31')),
              '2022': (pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'))}

t0 = time.time()


def elapsed() -> str:
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
log.info('%s STEP 2: adv20, spread_proxy, 12-1 signal, history gate...', elapsed())
panel['dvol'] = panel['close'] * panel['volume']
panel['adv20'] = g['dvol'].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
panel['spread_proxy'] = (((panel['high'] - panel['low']) / panel['close']).clip(lower=0) * 0.1).clip(upper=0.002)
lag_needed = sorted({0, 21, 252, 273})  # 273 kept to replicate 1700d's A1 universe history gate exactly
close_lag = {lag: (panel['close'] if lag == 0 else g['close'].shift(lag)) for lag in lag_needed}
panel['sig_12_1'] = close_lag[LAG_NEAR] / close_lag[LAG_FAR] - 1
panel['history_ok'] = close_lag[273].notna()
panel = panel.drop(columns=['dvol'])
log.info('%s signals built; history_ok True for %d/%d rows', elapsed(), int(panel['history_ok'].sum()), len(panel))

# ============================================================================== weekly calendar ==
cal = pd.Series(trading_days)


def build_calendar(freq: str) -> pd.DataFrame:
    period = cal.dt.to_period('W' if freq == 'weekly' else 'M')
    first = cal.groupby(period).min().sort_index()
    dates = first[(first >= WIN_START) & (first <= WIN_END)].reset_index(drop=True)
    prior_signal = [trading_days[day_idx[d] - 1] if day_idx[d] > 0 else pd.NaT for d in dates]
    df = pd.DataFrame({'entry_date': dates.values, 'prior_signal_date': prior_signal})
    df['next_entry_date'] = df['entry_date'].shift(-1)
    df = df.dropna(subset=['next_entry_date']).reset_index(drop=True)
    return df


weekly_cal = build_calendar('weekly')
log.info('%s weekly calendar: %d complete periods', elapsed(), len(weekly_cal))

all_dates = sorted(set(weekly_cal.entry_date) | set(weekly_cal.prior_signal_date) | set(weekly_cal.next_entry_date))
signal_dates = sorted(set(weekly_cal.prior_signal_date))

rows_at_dates = panel[panel.bar_date.isin(all_dates)]
open_piv = rows_at_dates.pivot_table(index='bar_date', columns='symbol', values='open', aggfunc='last').reindex(all_dates)
spread_piv = rows_at_dates.pivot_table(index='bar_date', columns='symbol', values='spread_proxy', aggfunc='last').reindex(all_dates)
cost_rate_piv = 0.0005 + 0.5 * spread_piv.fillna(0.002)
spy_entry_idx = spy_df.set_index('bar_date').reindex(all_dates)  # open/close at entry/signal/next dates only
log.info('%s pivots built: open_piv shape %s', elapsed(), open_piv.shape)

sig_rows = panel.loc[panel.bar_date.isin(signal_dates) & (panel['close'] >= PRICE_MIN) & panel['history_ok'],
                      ['symbol', 'bar_date', 'close', 'adv20', 'sig_12_1']]
sig_by_date = {d: sub.drop(columns='bar_date') for d, sub in sig_rows.groupby('bar_date')}
del panel, rows_at_dates, sig_rows
log.info('%s sig_by_date built: %d signal dates', elapsed(), len(sig_by_date))

# SPY over EVERY trading day (not just calendar dates) -- for the 200sma / 252d-return / Faber regime signals
spy_close_all = spy_df.set_index('bar_date')['close'].reindex(trading_days)
spy_sma200 = spy_close_all.rolling(200, min_periods=200).mean()
spy_ret252 = spy_close_all / spy_close_all.shift(252) - 1


def fwd_ret_row(entry_date, next_entry_date):
    return open_piv.loc[next_entry_date] / open_piv.loc[entry_date] - 1


def spy_fwd(entry_date, next_entry_date):
    a, b = spy_entry_idx['open'].get(entry_date), spy_entry_idx['open'].get(next_entry_date)
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return np.nan
    return b / a - 1


# ========================================================================= stats helpers (copied) ==
def ann_stats(returns, periods_per_year: int) -> dict:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return dict(n_periods=0, ann_return=np.nan, sharpe=np.nan, max_dd=np.nan, green_share=np.nan, worst=np.nan)
    comp = np.prod(1 + r)
    ann_return = comp ** (periods_per_year / n) - 1
    sd = np.std(r, ddof=1) if n > 1 else np.nan
    sharpe = (np.mean(r) / sd * np.sqrt(periods_per_year)) if (n > 1 and sd and sd > 0) else np.nan
    curve = np.cumprod(1 + r)
    max_dd = ((curve - np.maximum.accumulate(curve)) / np.maximum.accumulate(curve)).min()
    return dict(n_periods=n, ann_return=ann_return, sharpe=sharpe, max_dd=max_dd,
                green_share=float((r > 0).mean()), worst=float(r.min()))


def ols_alpha_beta(y, x):
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
                max_dd=net_stats['max_dd'], max_dd_spy=spy_stats['max_dd'], worst_period=net_stats['worst'],
                turnover_avg=float(np.nanmean(sub['turnover'])) if len(sub) else np.nan,
                cost_drag_annual=float(np.nanmean(sub['cost'])) * ppy if len(sub) else np.nan,
                green_share=net_stats['green_share'], null_pct_return=pct_ret, null_pct_excess=pct_excess,
                weeks_cash_share=float((~is_on).mean()) if len(is_on) else np.nan, n_switches=n_switch)


# =================================================================== regime filters (F1/F2/F3/F5) ==
missing_f1 = missing_f2 = missing_f3 = missing_f5 = 0
f1_vals, f2_vals, f5_vals = [], [], []
f5_state = None
for d in weekly_cal['prior_signal_date']:
    sc, sm = spy_close_all.get(d, np.nan), spy_sma200.get(d, np.nan)
    rr = spy_ret252.get(d, np.nan)
    if pd.isna(sc) or pd.isna(sm):
        f1_vals.append(True); missing_f1 += 1
        if f5_state is None:
            f5_state = True
        missing_f5 += 1
        f5_vals.append(f5_state)
    else:
        f1_vals.append(bool(sc > sm))
        if f5_state is None:
            f5_state = bool(sc > sm)
        elif f5_state and sc < sm * 0.98:
            f5_state = False
        elif (not f5_state) and sc > sm * 1.02:
            f5_state = True
        f5_vals.append(f5_state)
    if pd.isna(rr):
        f2_vals.append(True); missing_f2 += 1
    else:
        f2_vals.append(bool(rr > 0))
on_f0 = pd.Series(True, index=weekly_cal['entry_date'])
on_f1 = pd.Series(f1_vals, index=weekly_cal['entry_date'])
on_f2 = pd.Series(f2_vals, index=weekly_cal['entry_date'])
on_f5 = pd.Series(f5_vals, index=weekly_cal['entry_date'])
if missing_f1:
    log.warning('%s F1: %d/%d periods lacked 200sma history -- defaulted ON', elapsed(), missing_f1, len(weekly_cal))
if missing_f2:
    log.warning('%s F2: %d/%d periods lacked 252d-return history -- defaulted ON', elapsed(), missing_f2, len(weekly_cal))
if missing_f5:
    log.warning('%s F5: %d/%d periods lacked 200sma history -- defaulted ON', elapsed(), missing_f5, len(weekly_cal))

# F3 Faber: 10-month SMA of month-end closes; state decided at month-end M applies to ALL of month M+1
month_end_s = cal.groupby(cal.dt.to_period('M')).max()
month_end_s = month_end_s[(month_end_s >= WIN_START) & (month_end_s <= WIN_END)].sort_index()
spy_month_close = spy_close_all.reindex(month_end_s.values)
spy_month_close.index = month_end_s.index  # PeriodIndex, the month that just ended
sma10 = spy_month_close.rolling(10, min_periods=10).mean()
faber_by_period = {}
for per in spy_month_close.index:
    v, s = spy_month_close.loc[per], sma10.loc[per]
    faber_by_period[per + 1] = None if (pd.isna(v) or pd.isna(s)) else bool(v > s)
f3_vals = []
for d in weekly_cal['entry_date']:
    st = faber_by_period.get(d.to_period('M'), None)
    if st is None:
        f3_vals.append(True); missing_f3 += 1
    else:
        f3_vals.append(st)
on_f3 = pd.Series(f3_vals, index=weekly_cal['entry_date'])
if missing_f3:
    log.warning('%s F3 (Faber): %d/%d periods lacked a resolved month-end state -- defaulted ON',
                elapsed(), missing_f3, len(weekly_cal))

FILTERS = {'F0': on_f0, 'F1': on_f1, 'F2': on_f2, 'F3': on_f3, 'F4': None, 'F5': on_f5}
log.info('%s regime filters built: %s', elapsed(), {k: ('path-dependent' if v is None else f'{(~v).mean():.1%} cash')
                                                     for k, v in FILTERS.items()})


# ============================================================================== pool builder ==
def build_pools(cal_df: pd.DataFrame, adv_cutoff: float, sig_col: str) -> list[dict]:
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
        ranked = pool.nlargest(min(k, MAXN), sig_col)['symbol'].tolist() if k else []
        fwd_full = fwd_ret_row(edate, nxt) if k else None
        spy_r = spy_fwd(edate, nxt)
        cr = cost_rate_piv.loc[edate] if k else None
        pools.append(dict(sdate=sdate, edate=edate, ranked=ranked,
                           pool_syms=pool['symbol'].values if k else np.array([]), fwd_full=fwd_full,
                           spy_r=spy_r, cr=cr, k=k))
    return pools


pools = build_pools(weekly_cal, ADV_CUTOFF_U2, 'sig_12_1')
log.info('%s pools built: %d weekly periods', elapsed(), len(pools))


# ============================================================================== simulate (F0..F5) ==
def simulate(pools_: list[dict], n: int, on_series, filter_name: str):
    """Chronological pass: target portfolio each week is the top-n ranking if ON, empty (cash) if OFF.
    F4's ON/OFF is path-dependent on THIS n's own net-of-cost equity curve, so it is decided inline;
    all other filters pass a precomputed on_series keyed by entry_date."""
    prev_port: set = set()
    rows = []
    prev_state = None
    equity, peak, in_dd_kill = 1.0, 1.0, False
    n_cash_fallback = 0
    for p in pools_:
        edate, sdate, ranked, fwd_full, spy_r, cr, k = (p['edate'], p['sdate'], p['ranked'], p['fwd_full'],
                                                          p['spy_r'], p['cr'], p['k'])
        if filter_name == 'F4':
            sc, sm = spy_close_all.get(sdate, np.nan), spy_sma200.get(sdate, np.nan)
            if in_dd_kill and pd.notna(sc) and pd.notna(sm) and sc > sm:
                in_dd_kill = False
            is_on = not in_dd_kill
        else:
            is_on = bool(on_series.get(edate, True))
        prev_state = is_on if prev_state is None else prev_state

        port = set(ranked[:n]) if (is_on and k) else set()
        empty_pool_while_on = is_on and k == 0
        if empty_pool_while_on:
            n_cash_fallback += 1
        bought, sold = port - prev_port, prev_port - port
        if port:
            fwd = fwd_full.reindex(list(port))
            gross = float(fwd.fillna(0).mean())
        elif empty_pool_while_on:
            # Data gap (no eligible U2 names this week), NOT a filter decision -- NaN so it is
            # dropped from compounding/annualisation, exactly matching 1700d_grid.py's treatment
            # of a k==0 week (its `gross = np.nan` branch). Must NOT be confused with a deliberate
            # cash week (filter OFF), which really is a 0% realized return.
            gross = np.nan
        else:
            gross = 0.0  # filter OFF: deliberate cash
        cost = 0.0
        if bought:
            cost += cr.reindex(list(bought)).fillna(0.002).sum() / max(len(port), 1)
        if sold:
            cost += cr.reindex(list(sold)).fillna(0.002).sum() / max(len(prev_port), 1)
        net = gross - cost
        turnover = len(bought) / max(len(port), len(prev_port), 1)
        rows.append(dict(entry_date=edate, net_ret=net, gross_ret=gross, cost=cost, turnover=turnover,
                          spy_fwd=spy_r, is_on=is_on))
        prev_port = port
        if filter_name == 'F4':
            if pd.notna(net):
                equity *= (1 + net)
            if equity > peak:
                peak = equity
            dd = equity / peak - 1
            if (not in_dd_kill) and dd <= -0.20:
                in_dd_kill = True
    if n_cash_fallback:
        log.warning('%s %s/N%d: %d ON periods had an empty eligible pool (k=0) -- return set NaN and '
                    'dropped from compounding (data gap, not a filter decision), matching 1700d_grid.py',
                    elapsed(), filter_name, n, n_cash_fallback)
    real_df = pd.DataFrame(rows).set_index('entry_date')
    return real_df


def run_null_for_n_regime(pools_: list[dict], on_mask: np.ndarray, n: int, rng) -> np.ndarray:
    cols = []
    for p, is_on in zip(pools_, on_mask):
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


# ===================================================================================== main grid ==
rng = np.random.default_rng(SEED)
cells_rows, summary_rows = [], []
real_cache: dict = {}
cell_i, cell_total = 0, len(FILTERS) * len(N_LIST)
log.info('%s STEP 3: simulating %d (filter x N) cells, %d-draw null each...', elapsed(), cell_total, N_DRAWS)

for fname, on_series in FILTERS.items():
    for n in N_LIST:
        cell_i += 1
        real_df = simulate(pools, n, on_series, fname)
        real_cache[(fname, n)] = real_df
        null_mat = run_null_for_n_regime(pools, real_df['is_on'].values, n, rng)
        entry_dates = real_df.index.values
        win_reads = {}
        for win_name, (ws, we) in WINDOWS.items():
            mask = (entry_dates >= ws) & (entry_dates <= we)
            win_reads[win_name] = window_read(real_df, null_mat, mask, 52)
            cells_rows.append(dict(filter=fname, N=n, window=win_name, **win_reads[win_name]))
        whole, h1, h2 = win_reads['whole'], win_reads['halfA'], win_reads['halfB']
        pass_flag = bool(pd.notna(h1['excess_ann_return']) and h1['excess_ann_return'] > 0
                           and pd.notna(h2['excess_ann_return']) and h2['excess_ann_return'] > 0
                           and pd.notna(whole['t_alpha']) and whole['t_alpha'] >= 2.0
                           and pd.notna(whole['max_dd']) and pd.notna(whole['max_dd_spy'])
                           and abs(whole['max_dd']) <= 1.25 * abs(whole['max_dd_spy'])
                           and pd.notna(h1['null_pct_excess']) and h1['null_pct_excess'] >= 95.0
                           and pd.notna(h2['null_pct_excess']) and h2['null_pct_excess'] >= 95.0)
        summary_rows.append(dict(filter=fname, N=n, ann_net=whole['ann_return_net'], spy=whole['ann_return_spy'],
            excess_whole=whole['excess_ann_return'], excess_H1=h1['excess_ann_return'], excess_H2=h2['excess_ann_return'],
            alpha_t=whole['t_alpha'], maxDD_book=whole['max_dd'], maxDD_spy=whole['max_dd_spy'],
            null_pct_H1=h1['null_pct_excess'], null_pct_H2=h2['null_pct_excess'], weeks_cash=whole['weeks_cash_share'],
            switches=whole['n_switches'], PASS=pass_flag))
        log.info('%s cell %d/%d %s/N%d done: ann_net=%s spy=%s weeks_cash=%.1f%% switches=%d PASS=%s',
                  elapsed(), cell_i, cell_total, fname, n, f"{whole['ann_return_net']:.3f}" if pd.notna(whole['ann_return_net']) else 'NA',
                  f"{whole['ann_return_spy']:.3f}" if pd.notna(whole['ann_return_spy']) else 'NA',
                  100 * whole['weeks_cash_share'], whole['n_switches'], pass_flag)
        pd.DataFrame(cells_rows).to_csv(OUT / '1700e_cells.csv', index=False)

log.info('%s STEP 3 done: %d cells x 3 windows = %d rows written', elapsed(), cell_total, len(cells_rows))
summary_df = pd.DataFrame(summary_rows)

# =============================================================== A1 (N=20) by-year, all 6 filters ==
yr_rows = []
for fname in FILTERS:
    rdf = real_cache[(fname, 20)].reset_index()
    for yr, sub in rdf.groupby(rdf.entry_date.dt.year):
        book_ann = float(np.prod(1 + sub['net_ret'].fillna(0)) - 1)
        spy_ann = float(np.prod(1 + sub['spy_fwd'].fillna(0)) - 1)
        yr_rows.append(dict(filter=fname, year=int(yr), n_periods=len(sub), book_return=book_ann,
                             spy_return=spy_ann, excess=book_ann - spy_ann,
                             weeks_cash_share=float((~sub['is_on']).mean())))
by_year_df = pd.DataFrame(yr_rows)
by_year_df.to_csv(OUT / '1700e_by_year.csv', index=False)
log.info('%s wrote 1700e_by_year.csv (%d rows)', elapsed(), len(by_year_df))

# ============================================================= sub-windows, A1, F0 vs each filter ==
sub_rows = []
for sw_name, (ws, we) in SUBWINDOWS.items():
    for fname in FILTERS:
        rdf = real_cache[(fname, 20)]
        mask = (rdf.index >= ws) & (rdf.index <= we)
        sub = rdf.loc[mask]
        if len(sub) == 0:
            continue
        book_ret = float(np.prod(1 + sub['net_ret'].fillna(0)) - 1)
        spy_ret = float(np.prod(1 + sub['spy_fwd'].fillna(0)) - 1)
        sub_rows.append(dict(subwindow=sw_name, filter=fname, n_periods=len(sub), book_return=book_ret,
                              spy_return=spy_ret, excess=book_ret - spy_ret,
                              weeks_cash_share=float((~sub['is_on']).mean())))
sub_df = pd.DataFrame(sub_rows)

# ================================================================================ RESULT.md ==
def fmt(v, pct=True, dp=1):
    if pd.isna(v):
        return 'NA'
    return f'{100*v:+.{dp}f}%' if pct else f'{v:.2f}'


n_pass = int(summary_df['PASS'].sum())
bonf = 0.05 * 0.05 * len(summary_df)
lines = []
lines.append('# RESULT -- cell 1,700e: can we discover the bearish market and exit?')
lines.append('')
lines.append(f"PREREG_1700e.md (FROZEN). Book = 1,700d's A1 (U2 large caps, 12-1 momentum, weekly, equal "
             f"weight) plus top-10/top-50 robustness. 6 filters x 3 N = 18 cells x 3 windows = "
             f"{len(cells_rows)} rows in 1700e_cells.csv. Null = {N_DRAWS} draws/period (stated, matches "
             f"1700d); OFF periods draw 0 (cash) in both the real book and the null. Windows: whole "
             f"{WIN_START.date()}..{WIN_END.date()}, H1 ..{(HALF_CUT-pd.Timedelta(days=1)).date()}, "
             f"H2 {HALF_CUT.date()}..")
lines.append('')
lines.append('## 18-cell table (filter x N): whole-window read + both-halves pass checks')
lines.append('')
cols = ['filter', 'N', 'ann_net', 'spy', 'excess_whole', 'excess_H1', 'excess_H2', 'alpha_t', 'maxDD_book',
        'maxDD_spy', 'null_H1', 'null_H2', 'weeks_cash', 'switches', 'PASS']
lines.append('| ' + ' | '.join(cols) + ' |')
lines.append('|' + '---|' * len(cols))
for _, r in summary_df.sort_values(['filter', 'N']).iterrows():
    lines.append(f"| {r['filter']} | {int(r.N)} | {fmt(r.ann_net)} | {fmt(r.spy)} | {fmt(r.excess_whole)} | "
                 f"{fmt(r.excess_H1)} | {fmt(r.excess_H2)} | {fmt(r.alpha_t, pct=False)} | {fmt(r.maxDD_book)} | "
                 f"{fmt(r.maxDD_spy)} | {fmt(r.null_pct_H1, pct=False)} | {fmt(r.null_pct_H2, pct=False)} | "
                 f"{fmt(r.weeks_cash)} | {int(r.switches)} | {'PASS' if r.PASS else 'fail'} |")
lines.append('')
pass_list = ', '.join(f"{r['filter']}/N{int(r.N)}" for _, r in summary_df[summary_df.PASS].iterrows())
lines.append(f'## Pass list ({n_pass}/18): ' + (pass_list if n_pass else
             'NONE -- no cell clears both-halves excess>0 + alpha t>=2.0 + maxDD<=1.25xSPY + null pct>=95 both halves.'))
lines.append(f'**Bonferroni**: with 18 cells at a 5% per-half chance bar, ~{bonf:.2f} cells are expected to clear '
             f'BOTH halves by chance alone; {n_pass} observed passes {"is" if n_pass <= bonf*3 else "exceeds"} '
             f'{"within" if n_pass <= bonf*3 else "well above"} that noise floor.')
lines.append('')
lines.append('## A1 (U2, top 20, 12-1, weekly) by year: book return per filter vs SPY')
lines.append('')
lines.append('| Year | SPY | ' + ' | '.join(FILTERS.keys()) + ' |')
lines.append('|---|---|' + '---|' * len(FILTERS))
for yr in sorted(by_year_df.year.unique()):
    sub = by_year_df[by_year_df.year == yr].set_index('filter')
    spy_val = sub['spy_return'].iloc[0] if len(sub) else np.nan
    row = f"| {yr} | {fmt(spy_val)} |"
    for fname in FILTERS:
        row += f" {fmt(sub.loc[fname, 'book_return']) if fname in sub.index else 'NA'} |"
    lines.append(row)
lines.append('')
lines.append('## Sub-windows, A1: F0 vs each filter (book / SPY / excess / weeks-in-cash)')
lines.append('')
lines.append('| Sub-window | Filter | Book | SPY | Excess | Weeks cash |')
lines.append('|---|---|---|---|---|---|')
for sw in SUBWINDOWS:
    for fname in FILTERS:
        row = sub_df[(sub_df.subwindow == sw) & (sub_df['filter'] == fname)]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(f"| {sw} | {fname} | {fmt(r.book_return)} | {fmt(r.spy_return)} | {fmt(r.excess)} | {fmt(r.weeks_cash_share)} |")
lines.append('')
lines.append('## Methodology notes')
lines.append('- Universe/signal/cost/stats/null machinery copied from 1700d_grid.py (U2 ADV20>=$200M, price>=$10, '
             '>=273d history, 12-1 momentum t-252..t-21, 5bps/side + half spread-proxy cost, 300-draw null).')
lines.append('- F1/F2/F5 evaluated on prior_signal_date (prior close, no look-ahead); F3 evaluated at month-end '
             'close, held for the FOLLOWING calendar month (standard Faber/GTAA timing, not the same month).')
lines.append('- F4 (own-book 20% DD kill) is path-dependent per N: simulated in chronological order on that N\'s '
             'own net-of-cost equity curve; cleared only by SPY closing back above its 200-day SMA.')
lines.append('- OFF weeks = 0% return (cash); re-entry always buys the FRESH top-N ranking at the next ON week.')
lines.append('- Switch cost (full liquidation / full re-entry) charges each leg against its OWN portfolio size '
             '(new port for buys, prior port for sells) -- identical to 1700d\'s single-denom formula whenever '
             'portfolio size is constant (every F0 period), and the correct generalisation when it empties to cash.')
lines.append(f'- Null = {N_DRAWS} random same-N draws from the eligible pool on ON periods, 0 on OFF periods '
             '(cash, matched to the real book) -- isolates stock-selection skill from the shared regime timing.')
lines.append('- Early periods lacking SMA/return/month-end history default to ON (logged WARNING in 1700e.log), never OFF.')
lines.append('- No parameter outside this PREREG was tried after seeing numbers.')
(OUT / 'RESULT_1700e.md').write_text('\n'.join(lines) + '\n')
log.info('%s wrote RESULT_1700e.md (%d lines)', elapsed(), len(lines))
log.info('%s ALL DONE', elapsed())
