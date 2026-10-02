#!/usr/bin/env python3
"""Cell 1,700c -- monthly momentum decile on free Alpaca history 2016->2026, vs SPY.

PREREG: research/momentum_weekly/PREREG_1700c.md (FROZEN). Owner 10/2: "get the data, run it on
history, show me it beats SPY". The ONLY book carried forward from 1,700/1,700b is the monthly-rebalanced
top decile. Same cost model and pass-bar PHILOSOPHY as 1700b; same panel SHAPE (symbol, bar_date, open,
high, low, close, volume) so the adv20/spread_proxy/12-1-momentum/OLS-alpha machinery below is the same
arithmetic as 1700b_momentum.py, just re-run on an Alpaca-sourced panel with a name-pattern universe
filter instead of Databento security_type (1700b_momentum.py is a top-level script, not an importable
library -- its logic is reproduced here, not imported, to avoid re-running its own Databento pipeline).

Universe at each rebalance (first trading day of the month, entered AT THE OPEN; eligibility evaluated on
the PRIOR trading day's close -- no look-ahead): price >= $10 at that prior close, 20-day average dollar
volume >= $20M, >=252 days of price history (this is NOT a separate check: requiring the 12-1 momentum
signal sig_M2 to be non-NaN already requires >=252 prior rows via close.shift(252) -- the two conditions
are mathematically identical, documented here so it is not an accident), not ETF/ETN/fund/trust/warrant/
unit/preferred/right by Alpaca asset-name pattern, not ^Z[A-Z]ZZT$. Signal: 12-1 momentum. Portfolio: top
decile, equal weight, held one month. Costs: 5bps/side + half the spread proxy (capped 20bps) on every
traded dollar. Dividends ignored (price return) on both legs -- Alpaca adjustment=ALL back-adjusts splits
AND dividends into the price series (the only mode the task instructions specified); this makes the
"price return" here closer to a total-return proxy than a pure split-adjusted price return, stated
plainly rather than hidden (PREREG's own "else state it" clause for the adjustment mode).
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
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(OUT / '1700c.log'), filemode='w', level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700c')
log.addHandler(logging.StreamHandler(sys.stdout))

SEED = 17000
N_DRAWS = 1000
PRICE_MIN = 10.0
ADV_MIN = 20_000_000.0
DECILE_FRAC = 0.10
WIN_START = pd.Timestamp('2016-01-01')
WIN_END = pd.Timestamp('2026-09-30')
HALF_CUT = pd.Timestamp('2021-07-01')
COVID_START = pd.Timestamp('2020-02-01')
COVID_END = pd.Timestamp('2020-04-30')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_EXCLUDE_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|'
                              r'\bPREFERRED\b|\bRIGHTS?\b', re.IGNORECASE)

t0 = time.time()


def elapsed() -> str:
    return f'{time.time() - t0:6.0f}s'


# =================================================================== load panel ==
panel_path = OUT / 'panel_2016_2026.parquet'
if not panel_path.exists():
    log.error('%s %s missing -- run 1700c_fetch.py to completion first', elapsed(), panel_path)
    raise SystemExit(1)

log.info('%s STEP 1: loading Alpaca panel %s', elapsed(), panel_path.name)
raw = pd.read_parquet(panel_path)
# 21.9M rows: symbol as plain object strings would blow well past this node's 1.8GB free RAM. category
# dtype collapses the ~15,800 distinct symbols to small integer codes immediately after load.
raw['symbol'] = raw['symbol'].astype('category')
for c in ('open', 'high', 'low', 'close'):
    raw[c] = raw[c].astype('float32')
raw['bar_date'] = pd.to_datetime(raw['bar_date'])
log.info('%s memory after dtype-downcast: %.0f MB', elapsed(), raw.memory_usage(deep=True).sum() / 1e6)
n_before = len(raw)
raw = raw.drop_duplicates(subset=['symbol', 'bar_date'], keep='last')
if len(raw) != n_before:
    log.warning('%s %d duplicate (symbol,bar_date) rows dropped (should be ~0: batches partition '
                'symbols disjointly)', elapsed(), n_before - len(raw))
bad_px = (raw.open <= 0) | (raw.high <= 0) | (raw.low <= 0) | (raw.close <= 0)
if bad_px.any():
    log.warning('%s %d/%d rows have non-positive OHLC -- dropped', elapsed(), int(bad_px.sum()), len(raw))
    raw = raw[~bad_px].reset_index(drop=True)
log.info('%s loaded: %d rows, %d distinct symbols, %s..%s', elapsed(), len(raw), raw.symbol.nunique(),
          raw.bar_date.min().date(), raw.bar_date.max().date())

spy_df = raw.loc[raw.symbol == 'SPY', ['bar_date', 'open', 'close']].sort_values('bar_date').reset_index(drop=True)
if spy_df.empty:
    log.error('%s SPY missing from the panel -- cannot benchmark, aborting', elapsed())
    raise SystemExit(1)
log.info('%s SPY pulled BEFORE universe exclusions (benchmark only, never a candidate holding): rows=%d '
          '%s..%s', elapsed(), len(spy_df), spy_df.bar_date.min().date(), spy_df.bar_date.max().date())
trading_days = sorted(spy_df['bar_date'].unique())

# =========================================================== name-pattern universe ==
assets_df = pd.read_csv(OUT / '1700c_assets.csv', dtype={'symbol': str, 'name': str})
assets_df['name'] = assets_df['name'].fillna('')
excluded_by_name = set(assets_df.loc[assets_df['name'].str.contains(NAME_EXCLUDE_RE), 'symbol'])
# Conservative: if ANY asset record for a symbol string (active or inactive) matches the exclusion
# pattern, the whole symbol is excluded -- stated simplification for tickers Alpaca has reused across
# two different historical entities.
excluded_by_test = {s for s in raw['symbol'].unique() if TEST_RE.match(s)}
excluded = excluded_by_name | excluded_by_test
log.info('%s name-pattern universe filter: %d symbols excluded (ETF/ETN/fund/trust/warrant/unit/'
          'preferred/right by name, %d test tickers)', elapsed(), len(excluded), len(excluded_by_test))

panel = raw[(raw.symbol != 'SPY') & ~raw.symbol.isin(excluded)].copy()
panel['symbol'] = panel['symbol'].cat.remove_unused_categories()
log.info('%s candidate panel after exclusions: %d symbols, %d rows', elapsed(), panel.symbol.nunique(), len(panel))

panel = panel.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
g = panel.groupby('symbol', sort=False)
log.info('%s STEP 2: computing adv20, spread_proxy, 12-1 momentum...', elapsed())
panel['dvol'] = panel['close'] * panel['volume']
panel['adv20'] = g['dvol'].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
panel['spread_proxy'] = (((panel['high'] - panel['low']) / panel['close']).clip(lower=0) * 0.1).clip(upper=0.002)
panel['close_lag21'] = g['close'].shift(21)
panel['close_lag252'] = g['close'].shift(252)
panel['sig_M2'] = panel['close_lag21'] / panel['close_lag252'] - 1
# sig_M2.notna() below REQUIRES close_lag252 to exist, i.e. >=252 prior rows -- this IS the ">=252 days
# history" universe condition from the PREREG; no separate check is coded because none is needed.

# ================================================================== monthly calendar ==
cal = pd.Series(trading_days)
ym = cal.dt.to_period('M')
month_first = cal.groupby(ym).min().sort_index()
months_all = month_first[(month_first >= WIN_START) & (month_first <= WIN_END)].reset_index(drop=True)
day_idx = {d: i for i, d in enumerate(trading_days)}
prior_signal = [trading_days[day_idx[d] - 1] if day_idx[d] > 0 else pd.NaT for d in months_all]
cal_df = pd.DataFrame({'entry_date': months_all.values, 'prior_signal_date': prior_signal})
cal_df['next_entry_date'] = cal_df['entry_date'].shift(-1)
n_nominal = len(cal_df)
cal_df = cal_df.dropna(subset=['next_entry_date']).reset_index(drop=True)
log.info('%s monthly calendar: %d nominal rebalances %s..%s, %d complete holding periods (last '
          'rebalance %s is open-ended beyond the fetch boundary %s and excluded from performance stats)',
          elapsed(), n_nominal, months_all.min().date(), months_all.max().date(), len(cal_df),
          months_all.max().date(), WIN_END.date())

all_dates = sorted(set(cal_df.entry_date) | set(cal_df.prior_signal_date) | set(cal_df.next_entry_date))
mon_rows = panel[panel.bar_date.isin(all_dates)]
open_piv = mon_rows.pivot_table(index='bar_date', columns='symbol', values='open', aggfunc='last').reindex(all_dates)
spread_piv = mon_rows.pivot_table(index='bar_date', columns='symbol', values='spread_proxy', aggfunc='last').reindex(all_dates)
cost_rate_piv = 0.0005 + 0.5 * spread_piv.fillna(0.002)
spy_idx = spy_df.set_index('bar_date').reindex(all_dates)
log.info('%s pivots built: open_piv shape %s', elapsed(), open_piv.shape)


def fwd_ret_row(entry_date, next_entry_date):
    """Per-symbol open-to-open forward return (explicit label lookup, not a positional shift)."""
    return open_piv.loc[next_entry_date] / open_piv.loc[entry_date] - 1


def spy_fwd(entry_date, next_entry_date):
    a, b = spy_idx['open'].get(entry_date), spy_idx['open'].get(next_entry_date)
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return np.nan
    return b / a - 1


# ============================================================================ simulation ==
def run_real(cal_df_: pd.DataFrame) -> pd.DataFrame:
    """Decile-momentum simulation: top 10% of the eligible pool by 12-1 momentum, equal weight,
    monthly rebalance, costed."""
    prev_port: set[str] = set()
    rows = []
    for _, wrow in cal_df_.iterrows():
        sdate, edate, nxt = wrow['prior_signal_date'], wrow['entry_date'], wrow['next_entry_date']
        day = panel.loc[(panel.bar_date == sdate) & (panel['close'] >= PRICE_MIN) & (panel['adv20'] >= ADV_MIN)
                         & panel['sig_M2'].notna(), ['symbol', 'sig_M2', 'adv20']]
        k = len(day)
        n = max(1, round(DECILE_FRAC * k))
        if k < n:
            log.warning('period=%s: eligible pool %d < decile N=%d -- taking all available', edate.date(), k, n)
        port = set(day.nlargest(n, 'sig_M2')['symbol']) if k else set()
        bought, sold = port - prev_port, prev_port - port
        if port:
            fwd = fwd_ret_row(edate, nxt).reindex(list(port))
            n_missing = int(fwd.isna().sum())
            gross = fwd.fillna(0).mean()
        else:
            n_missing, gross = 0, np.nan
        denom = len(port) if port else 1
        cost = 0.0
        if bought or sold:
            cr = cost_rate_piv.loc[edate]
            cost = (cr.reindex(list(bought)).fillna(0.002).sum() + cr.reindex(list(sold)).fillna(0.002).sum()) / denom
        turnover = len(bought) / denom
        spy_r = spy_fwd(edate, nxt)
        rows.append(dict(entry_date=edate, n_eligible=k, port_n=len(port), gross_ret=gross, cost=cost,
                          net_ret=(gross - cost if pd.notna(gross) else np.nan), turnover=turnover,
                          n_fwd_missing=n_missing, spy_fwd=spy_r))
        prev_port = port
    return pd.DataFrame(rows).set_index('entry_date')


def run_null(cal_df_: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """1,000 random decile-sized draws from the SAME eligible pool each month -> (1000, n_periods)."""
    cols_out = []
    for _, wrow in cal_df_.iterrows():
        sdate, edate, nxt = wrow['prior_signal_date'], wrow['entry_date'], wrow['next_entry_date']
        pool = panel.loc[(panel.bar_date == sdate) & (panel['close'] >= PRICE_MIN) & (panel['adv20'] >= ADV_MIN)
                          & panel['sig_M2'].notna(), 'symbol'].values
        k = len(pool)
        if k == 0:
            cols_out.append(np.full(N_DRAWS, np.nan))
            continue
        n = max(1, round(DECILE_FRAC * k))
        rets = fwd_ret_row(edate, nxt).reindex(pool).fillna(0).values
        rand = rng.random((N_DRAWS, k))
        idx = np.argpartition(rand, n - 1, axis=1)[:, :n]
        cols_out.append(rets[idx].mean(axis=1))
    return np.column_stack(cols_out)


def ann_stats(returns, periods_per_year: int = 12) -> dict:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return dict(n_periods=0, ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan, max_dd=np.nan,
                    worst_period=np.nan, best_period=np.nan, green_share=np.nan)
    comp = np.prod(1 + r)
    ann_return = comp ** (periods_per_year / n) - 1
    sd = np.std(r, ddof=1) if n > 1 else np.nan
    ann_vol = sd * np.sqrt(periods_per_year) if n > 1 else np.nan
    sharpe = (np.mean(r) / sd * np.sqrt(periods_per_year)) if (n > 1 and sd and sd > 0) else np.nan
    curve = np.cumprod(1 + r)
    peak = np.maximum.accumulate(curve)
    max_dd = ((curve - peak) / peak).min()
    return dict(n_periods=n, ann_return=ann_return, ann_vol=ann_vol, sharpe=sharpe, max_dd=max_dd,
                worst_period=r.min(), best_period=r.max(), green_share=float((r > 0).mean()))


def ols_alpha_beta(y, x):
    """Monthly OLS y = alpha + beta*x + eps. Returns (alpha, beta, t_alpha, n)."""
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


log.info('%s STEP 3: simulating decile_monthly, real + %d-draw null...', elapsed(), N_DRAWS)
rng = np.random.default_rng(SEED)
real_df = run_real(cal_df)
null_mat = run_null(cal_df, rng)
log.info('%s done: n_periods=%d median_pool_n=%.0f median_port_n=%.0f', elapsed(), len(cal_df),
          np.nanmedian(real_df['n_eligible']), np.nanmedian(real_df['port_n']))
real_df.reset_index().to_csv(OUT / '1700c_monthly.csv', index=False)
log.info('%s wrote 1700c_monthly.csv (%d rows)', elapsed(), len(real_df))

# ====================================================================== windowed reads ==
WINDOWS = {
    'halfA': (WIN_START, HALF_CUT - pd.Timedelta(days=1)),
    'halfB': (HALF_CUT, WIN_END),
    'whole': (WIN_START, WIN_END),
    'covid_2020': (COVID_START, COVID_END),
}
entry_dates = cal_df.entry_date.values
reads = []
for win_name, (ws, we) in WINDOWS.items():
    mask = (entry_dates >= ws) & (entry_dates <= we)
    sub = real_df.loc[mask]
    spy_sub = sub['spy_fwd'].values
    net_stats = ann_stats(sub['net_ret'].values)
    gross_stats = ann_stats(sub['gross_ret'].values)
    spy_stats = ann_stats(spy_sub)
    valid = np.isfinite(sub['net_ret'].values) & np.isfinite(spy_sub)
    excess_geo = (1 + sub['net_ret'].values[valid]) / (1 + spy_sub[valid]) - 1 if valid.any() else np.array([])
    excess_stats = ann_stats(excess_geo)
    alpha, beta, t_alpha, n_ols = ols_alpha_beta(sub['net_ret'].values, spy_sub)
    ann_alpha = alpha * 12 if pd.notna(alpha) else np.nan
    null_sub = null_mat[:, mask]
    null_ann_ret = np.array([ann_stats(null_sub[i]) ['ann_return'] for i in range(N_DRAWS)])
    null_excess = (1 + null_sub) / (1 + np.where(np.isfinite(spy_sub), spy_sub, 0.0)) - 1
    null_excess_ann = np.array([ann_stats(null_excess[i])['ann_return'] for i in range(N_DRAWS)])
    pct_ret = 100 * np.nanmean(null_ann_ret <= net_stats['ann_return']) if pd.notna(net_stats['ann_return']) else np.nan
    pct_excess = 100 * np.nanmean(null_excess_ann <= excess_stats['ann_return']) if pd.notna(excess_stats['ann_return']) else np.nan
    reads.append(dict(
        window=win_name, n_periods=net_stats['n_periods'], median_pool_n=float(np.nanmedian(sub['n_eligible'])) if len(sub) else np.nan,
        median_port_n=float(np.nanmedian(sub['port_n'])) if len(sub) else np.nan,
        ann_return_net=net_stats['ann_return'], ann_return_gross=gross_stats['ann_return'],
        ann_return_spy=spy_stats['ann_return'], excess_ann_return=excess_stats['ann_return'],
        ann_vol=net_stats['ann_vol'], sharpe_net=net_stats['sharpe'], max_dd=net_stats['max_dd'],
        max_dd_spy=spy_stats['max_dd'], worst_month=net_stats['worst_period'], best_month=net_stats['best_period'],
        green_share=net_stats['green_share'], turnover_avg=float(np.nanmean(sub['turnover'])) if len(sub) else np.nan,
        cost_drag_annual=float(np.nanmean(sub['cost'])) * 12 if len(sub) else np.nan,
        beta_spy=beta, ann_alpha_spy=ann_alpha, t_alpha=t_alpha, null_pct_return=pct_ret,
        null_pct_excess=pct_excess, n_fwd_missing=int(sub['n_fwd_missing'].sum())))
reads_df = pd.DataFrame(reads)
reads_df.to_csv(OUT / '1700c_reads.csv', index=False)
log.info('%s wrote 1700c_reads.csv (%d rows)', elapsed(), len(reads_df))

# =========================================================================== by-year table ==
yr_rows = []
for yr, sub in real_df.reset_index().groupby(real_df.reset_index().entry_date.dt.year):
    book_ann = np.prod(1 + sub['net_ret'].fillna(0)) - 1
    spy_ann = np.prod(1 + sub['spy_fwd'].fillna(0)) - 1
    yr_rows.append(dict(year=int(yr), n_months=len(sub), book_return=book_ann, spy_return=spy_ann,
                         excess=book_ann - spy_ann))
by_year_df = pd.DataFrame(yr_rows)
by_year_df.to_csv(OUT / '1700c_by_year.csv', index=False)
log.info('%s wrote 1700c_by_year.csv (%d years)', elapsed(), len(by_year_df))

# =========================================================================== survivorship ==
log.info('%s STEP 4: survivorship statement vs Databento PIT panel (read-only)...', elapsed())
db_paths = [ROOT / 'data/research/databento/equs_daily_2024H2.parquet',
            ROOT / 'data/research/databento/equs_daily_2025_2026.parquet']
surv_rows = []
alpaca_syms_by_year = {yr: set(g_['symbol']) for yr, g_ in raw.groupby(raw.bar_date.dt.year)}
inactive_syms = set(assets_df.loc[assets_df.status == 'inactive', 'symbol'])
alpaca_all_syms = set(raw['symbol'].unique())
inactive_with_bars = len(inactive_syms & alpaca_all_syms)
for p in db_paths:
    if not p.exists():
        log.warning('%s %s not found -- skipped (survivorship coverage reduced, not fabricated)', elapsed(), p)
        continue
    db = pd.read_parquet(p, columns=['symbol', 'bar_date'])
    db['bar_date'] = pd.to_datetime(db['bar_date'])
    for yr, g_ in db.groupby(db.bar_date.dt.year):
        db_syms = set(g_['symbol'].dropna())
        alp_syms = alpaca_syms_by_year.get(int(yr), set())
        missing = db_syms - alp_syms
        surv_rows.append(dict(year=int(yr), db_symbols=len(db_syms), missing_from_alpaca=len(missing),
                               pct_missing=100 * len(missing) / max(1, len(db_syms))))
surv_df = pd.DataFrame(surv_rows).drop_duplicates(subset='year').sort_values('year')
log.info('%s survivorship by year:\n%s', elapsed(), surv_df.to_string(index=False))
log.info('%s inactive assets with >=1 Alpaca bar: %d / %d (%.1f%%)', elapsed(), inactive_with_bars,
          len(inactive_syms), 100 * inactive_with_bars / max(1, len(inactive_syms)))

# =========================================================================== pass bar ==
def cond_half(win):
    row = reads_df[reads_df.window == win].iloc[0]
    return pd.notna(row.excess_ann_return) and row.excess_ann_return > 0, row


a_ok, a_row = cond_half('halfA')
b_ok, b_row = cond_half('halfB')
whole_row = reads_df[reads_df.window == 'whole'].iloc[0]
spy_whole = reads_df[reads_df.window == 'whole'].iloc[0]
alpha_ok = pd.notna(whole_row.t_alpha) and whole_row.t_alpha >= 2.0
dd_ok = pd.notna(whole_row.max_dd) and pd.notna(whole_row.max_dd_spy) and abs(whole_row.max_dd) <= 1.25 * abs(whole_row.max_dd_spy)
null_a_ok = pd.notna(a_row.null_pct_excess) and a_row.null_pct_excess >= 95.0
null_b_ok = pd.notna(b_row.null_pct_excess) and b_row.null_pct_excess >= 95.0
PASSES = a_ok and b_ok and alpha_ok and dd_ok and null_a_ok and null_b_ok
log.info('%s PASS BAR: excessA>0=%s excessB>0=%s alpha_t_whole=%.2f(>=2.0 %s) maxDD book=%.1f%% vs '
          '1.25xSPY=%.1f%% (%s) null_pctA=%.1f(>=95 %s) null_pctB=%.1f(>=95 %s) -> %s', elapsed(),
          a_ok, b_ok, whole_row.t_alpha, alpha_ok, 100 * whole_row.max_dd, 125 * abs(whole_row.max_dd_spy),
          dd_ok, a_row.null_pct_excess, null_a_ok, b_row.null_pct_excess, null_b_ok,
          'PASS' if PASSES else 'FAIL')

# =========================================================================== RESULT.md ==
lines = []
lines.append('# RESULT -- cell 1,700c: monthly momentum decile, Alpaca free history, vs SPY')
lines.append('')
lines.append(f'PREREG_1700c.md (FROZEN). Window {WIN_START.date()}..{WIN_END.date()}, {n_nominal} nominal '
             f'monthly rebalances, {len(cal_df)} complete holding periods (last rebalance '
             f'{months_all.max().date()} is open beyond the fetch boundary, excluded from stats).')
lines.append('')
lines.append('## By-year: book vs SPY (net of cost; price return, dividends not separately modeled -- '
              'see Data note)')
lines.append('')
lines.append('| Year | Months | Book | SPY | Excess |')
lines.append('|---|---|---|---|---|')
for _, r in by_year_df.iterrows():
    lines.append(f"| {int(r.year)} | {int(r.n_months)} | {100*r.book_return:+.1f}% | {100*r.spy_return:+.1f}% "
                 f"| {100*r.excess:+.1f}% |")
lines.append('')
lines.append('## Reads (halfA/halfB/whole/covid_2020; annualised net-of-cost unless marked)')
lines.append('')
cols_show = ['window', 'n_periods', 'ann_return_net', 'ann_return_spy', 'excess_ann_return', 'ann_alpha_spy',
             't_alpha', 'sharpe_net', 'max_dd', 'max_dd_spy', 'null_pct_return', 'null_pct_excess',
             'turnover_avg', 'cost_drag_annual', 'green_share']
lines.append('| ' + ' | '.join(cols_show) + ' |')
lines.append('|' + '---|' * len(cols_show))
for _, r in reads_df.iterrows():
    vals = []
    for c in cols_show:
        v = r[c]
        if c == 'window':
            vals.append(str(v))
        elif c in ('n_periods',):
            vals.append(f'{int(v)}')
        elif isinstance(v, float) and pd.notna(v):
            vals.append(f'{v:.4f}')
        else:
            vals.append('NA')
    lines.append('| ' + ' | '.join(vals) + ' |')
lines.append('')
lines.append(f'**Pass bar (PREREG): excess>0 both halves AND alpha_t(whole)>=2.0 AND maxDD<=1.25xSPY AND '
             f'null_pct_excess>=95 both halves -> {"PASS" if PASSES else "FAIL"}**')
lines.append(f'- excess>0: halfA={a_ok} ({100*a_row.excess_ann_return:+.1f}%), halfB={b_ok} '
             f'({100*b_row.excess_ann_return:+.1f}%)')
lines.append(f'- alpha t-stat whole window = {whole_row.t_alpha:.2f} (need >=2.0): {alpha_ok}')
lines.append(f'- max DD book={100*whole_row.max_dd:.1f}% vs 1.25x SPY={125*abs(whole_row.max_dd_spy):.1f}% '
             f'(SPY maxDD={100*whole_row.max_dd_spy:.1f}%): {dd_ok}')
lines.append(f'- null percentile of excess: halfA={a_row.null_pct_excess:.1f} (>=95: {null_a_ok}), '
             f'halfB={b_row.null_pct_excess:.1f} (>=95: {null_b_ok})')
lines.append('')
lines.append('## Survivorship statement (Databento PIT panel, read-only, 2024-07-> coverage only)')
lines.append('')
for _, r in surv_df.iterrows():
    lines.append(f'- {int(r.year)}: {int(r.db_symbols)} Databento PIT symbols, {int(r.missing_from_alpaca)} '
                 f'missing from the Alpaca panel ({r.pct_missing:.1f}%)')
lines.append(f'- Inactive Alpaca assets with >=1 bar served: {inactive_with_bars} / {len(inactive_syms)} '
             f'({100*inactive_with_bars/max(1,len(inactive_syms)):.1f}%)')
lines.append('')
lines.append('## Delisted-bars answer (step 1 probe, 1700c_delisted_probe.csv)')
try:
    probe_df = pd.read_csv(OUT / '1700c_delisted_probe.csv')
    for _, r in probe_df.iterrows():
        lines.append(f"- {r.symbol} ({r.window_start}..{r.window_end}): served={r.served}, n_bars={r.n_bars}")
except FileNotFoundError:
    lines.append('- probe file missing')
lines.append('')
try:
    comp = pd.read_csv(OUT / '1700c_completeness.csv').iloc[0]
    lines.append(f'## Completeness: requested={int(comp.assets_requested)} with_bars={int(comp.symbols_with_bars)} '
                 f'LOST={int(comp.symbols_lost)} | active_requested={int(comp.active_assets_requested)} '
                 f'active_with_bars={int(comp.active_with_bars)} active_coverage={comp.active_coverage_pct:.1f}% '
                 f'gate(>=95%)={"PASS" if comp.gate_95pct_pass else "FAIL -> VOID"}')
except FileNotFoundError:
    lines.append('## Completeness: 1700c_completeness.csv missing')
lines.append('')
lines.append('## Methodology notes / caveats')
lines.append('- Universe common-stock filter is Alpaca asset-NAME pattern only (ETF/ETN/fund/trust/'
             'warrant/unit/preferred/right), applied for the full window and both active+inactive '
             'assets; Databento security_type cross-check (PREREG\'s "where known") was NOT run this '
             'pass for time -- a scoping cut, not a silent gap, logged here.')
lines.append('- adjustment=ALL folds dividends into back-adjusted price (total-return-like), not a pure '
              'split-only price return -- see module docstring.')
lines.append('- A ticker string reused by two different Alpaca asset records (one active, one inactive) '
              'is excluded from the universe if EITHER record\'s name matches the exclusion pattern.')
lines.append(f'- No variant (decile width, skip, universe floor) was tried after seeing these numbers, '
              f'per PREREG.')
(OUT / 'RESULT_1700c.md').write_text('\n'.join(lines) + '\n')
log.info('%s wrote RESULT_1700c.md (%d lines)', elapsed(), len(lines))
log.info('%s DONE', elapsed())
