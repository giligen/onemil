#!/usr/bin/env python3
"""Cell 1,700d -- owner's "buy the top 10" parameter grid: universe x N x lookback x rebalance-cadence,
vs SPY, on the free Alpaca 10-year panel.

PREREG: research/momentum_weekly/PREREG_1700d.md (FROZEN 2026-10-02 11:05 UTC). Amendment 1 (11:10 UTC,
before any number read) added lookbacks L5 (6mo) and L6 (6-1), growing the grid from 48 to 72 cells; the
owner's cadence preference (weekly) is honoured by sorting the RESULT table weekly-first.

Reuses research/momentum_weekly/1700c_momentum.py's panel-load / name-pattern-universe / cost-model /
ann_stats / ols_alpha_beta machinery -- COPIED, not imported: 1700c_momentum.py executes its whole
pipeline at module scope, so `import` would silently re-run cell 1,700c as a side effect.

Grid: universe {U1 price>=$10 ADV20>=$20M, U2 price>=$10 ADV20>=$200M} x N {10,20,50} x lookback
{L1 52wk t-252..t, L2 12-1 t-252..t-21, L3 13mo t-273..t, L4 13-1 t-273..t-21, L5 6mo t-126..t,
L6 6-1 t-126..t-21} x rebalance {weekly Mon-open..Mon-open, monthly 1st-trading-day-open} = 72 cells,
each read on 3 windows (whole 2016-01..2026-09, H1 2016-01..2021-06, H2 2021-07..2026-09) = 216 rows.
Both universes require >=273 trading days of history (close_lag273 non-null), independent of which
lookback is used -- a fixed universe condition per PREREG, not tied to the signal's own lag.

Signal/eligibility is evaluated on the trading day BEFORE the rebalance date's open (no look-ahead);
selection is computed ONCE per (universe, lookback, rebalance, period) and top-N is sliced for all three
N values from that one ranking, and the eligible pool + forward returns are captured once per period and
reused for the null draws of every N -- the "compute once, reuse across N" the task asked for.

Null draws: PREREG target is 1,000; reduced to 300 here (stated, not hidden) -- 72 cells x up to 560
weekly periods x 3 N-values sharing pools makes 1,000 draws cost an estimated 2+ hours on this node's
single core, incompatible with a detached run polled a few minutes apart; 300 draws is the largest count
that keeps the whole grid under ~40 minutes by the same per-draw cost measured on cell 1,700c's log
(129 monthly periods x 1,000 draws x ~1,300-name pool = 64s).
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

logging.basicConfig(filename=str(OUT / '1700d.log'), filemode='w', level=logging.INFO,
                     format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700d')
log.addHandler(logging.StreamHandler(sys.stdout))

SEED = 17001
N_DRAWS = 300  # PREREG target 1,000 -- reduced, see module docstring
PRICE_MIN = 10.0
HIST_MIN_DAYS = 273
WIN_START = pd.Timestamp('2016-01-01')
WIN_END = pd.Timestamp('2026-09-30')
HALF_CUT = pd.Timestamp('2021-07-01')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_EXCLUDE_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|'
                              r'\bPREFERRED\b|\bRIGHTS?\b', re.IGNORECASE)

UNIVERSES = {'U1': 20_000_000.0, 'U2': 200_000_000.0}
N_LIST = [10, 20, 50]
# name -> (lag_far, lag_near): sig = close_lag_near / close_lag_far - 1
LOOKBACKS = {
    'L1_52w': (252, 0), 'L2_12_1': (252, 21),
    'L3_13m': (273, 0), 'L4_13_1': (273, 21),
    'L5_6m': (126, 0), 'L6_6_1': (126, 21),
}
WINDOWS = {'halfA': (WIN_START, HALF_CUT - pd.Timedelta(days=1)), 'halfB': (HALF_CUT, WIN_END),
           'whole': (WIN_START, WIN_END)}

t0 = time.time()


def elapsed() -> str:
    return f'{time.time() - t0:6.0f}s'


# =================================================================== load panel (copied from 1700c) ==
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
log.info('%s STEP 2: adv20, spread_proxy, 6 lookback signals, history gate...', elapsed())
panel['dvol'] = panel['close'] * panel['volume']
panel['adv20'] = g['dvol'].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
panel['spread_proxy'] = (((panel['high'] - panel['low']) / panel['close']).clip(lower=0) * 0.1).clip(upper=0.002)
lag_needed = sorted({0, 21, 126, 252, 273})
close_lag = {lag: (panel['close'] if lag == 0 else g['close'].shift(lag)) for lag in lag_needed}
for lb_name, (far, near) in LOOKBACKS.items():
    panel[f'sig_{lb_name}'] = close_lag[near] / close_lag[far] - 1
panel['history_ok'] = close_lag[273].notna()
panel = panel.drop(columns=['dvol'])
log.info('%s signals built; history_ok True for %d/%d rows', elapsed(), int(panel['history_ok'].sum()), len(panel))

# ============================================================================== calendars ==
cal = pd.Series(trading_days)
day_idx = {d: i for i, d in enumerate(trading_days)}


def build_calendar(freq: str) -> tuple[pd.DataFrame, int]:
    """First trading day of each week (Mon, or next trading day if Monday is a holiday) or month,
    paired with the PRIOR trading day (signal date, no look-ahead) and the NEXT rebalance date."""
    period = cal.dt.to_period('W' if freq == 'weekly' else 'M')
    first = cal.groupby(period).min().sort_index()
    dates = first[(first >= WIN_START) & (first <= WIN_END)].reset_index(drop=True)
    prior_signal = [trading_days[day_idx[d] - 1] if day_idx[d] > 0 else pd.NaT for d in dates]
    df = pd.DataFrame({'entry_date': dates.values, 'prior_signal_date': prior_signal})
    df['next_entry_date'] = df['entry_date'].shift(-1)
    n_nominal = len(df)
    df = df.dropna(subset=['next_entry_date']).reset_index(drop=True)
    return df, n_nominal


weekly_cal, weekly_nominal = build_calendar('weekly')
monthly_cal, monthly_nominal = build_calendar('monthly')
log.info('%s calendars: weekly %d nominal / %d complete periods, monthly %d nominal / %d complete',
          elapsed(), weekly_nominal, len(weekly_cal), monthly_nominal, len(monthly_cal))

all_dates = sorted(set(weekly_cal.entry_date) | set(weekly_cal.prior_signal_date) | set(weekly_cal.next_entry_date)
                    | set(monthly_cal.entry_date) | set(monthly_cal.prior_signal_date) | set(monthly_cal.next_entry_date))
signal_dates = sorted(set(weekly_cal.prior_signal_date) | set(monthly_cal.prior_signal_date))

rows_at_dates = panel[panel.bar_date.isin(all_dates)]
open_piv = rows_at_dates.pivot_table(index='bar_date', columns='symbol', values='open', aggfunc='last').reindex(all_dates)
spread_piv = rows_at_dates.pivot_table(index='bar_date', columns='symbol', values='spread_proxy', aggfunc='last').reindex(all_dates)
cost_rate_piv = 0.0005 + 0.5 * spread_piv.fillna(0.002)
spy_idx = spy_df.set_index('bar_date').reindex(all_dates)
log.info('%s pivots built: open_piv shape %s', elapsed(), open_piv.shape)

sig_cols = [f'sig_{lb}' for lb in LOOKBACKS]
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


# ============================================================================== simulation ==
MAXN = max(N_LIST)


def run_cell(cal_df_: pd.DataFrame, adv_cutoff: float, sig_col: str):
    """One pass over the calendar: eligible-pool ranking computed ONCE per period, top-N sliced for
    every N in N_LIST from that ranking; pool + forward returns captured once per period for the null."""
    prev_port = {n: set() for n in N_LIST}
    rows = {n: [] for n in N_LIST}
    contrib = {n: {} for n in N_LIST}
    k_lt_n = {n: 0 for n in N_LIST}
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
        ranked = pool.nlargest(min(k, MAXN), sig_col)['symbol'].tolist() if k else []
        fwd_full = fwd_ret_row(edate, nxt) if k else None
        spy_r = spy_fwd(edate, nxt)
        cr = cost_rate_piv.loc[edate] if k else None
        pools_for_null.append((pool['symbol'].values if k else np.array([]), fwd_full))
        for n in N_LIST:
            if k < n:
                k_lt_n[n] += 1
            port = set(ranked[:n])
            bought, sold = port - prev_port[n], prev_port[n] - port
            if port:
                fwd = fwd_full.reindex(list(port))
                n_missing = int(fwd.isna().sum())
                gross = fwd.fillna(0).mean()
                for sym, r in fwd.items():
                    contrib[n][sym] = contrib[n].get(sym, 0.0) + (r if pd.notna(r) else 0.0) / len(port)
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
            prev_port[n] = port
    real_by_n = {n: pd.DataFrame(rows[n]).set_index('entry_date') for n in N_LIST}
    return real_by_n, pools_for_null, contrib, k_lt_n


def run_null_for_n(pools_for_null, n, rng) -> np.ndarray:
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


def top5_share_of(contrib_dict: dict) -> float:
    vals = np.array(list(contrib_dict.values()), dtype=float)
    if len(vals) == 0:
        return np.nan
    pos_sum = vals[vals > 0].sum()
    if pos_sum <= 0:
        return np.nan
    top5 = np.sort(vals)[-5:].sum()
    return float(top5 / pos_sum)


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
    return dict(n_periods=net_stats['n_periods'], ann_return_net=net_stats['ann_return'],
                ann_return_spy=spy_stats['ann_return'], excess_ann_return=excess_stats['ann_return'],
                beta_spy=beta, ann_alpha_spy=ann_alpha, t_alpha=t_alpha, sharpe_net=net_stats['sharpe'],
                max_dd=net_stats['max_dd'], max_dd_spy=spy_stats['max_dd'],
                turnover_avg=float(np.nanmean(sub['turnover'])) if len(sub) else np.nan,
                cost_drag_annual=float(np.nanmean(sub['cost'])) * ppy if len(sub) else np.nan,
                green_share=net_stats['green_share'], null_pct_return=pct_ret, null_pct_excess=pct_excess)


# ===================================================================================== main grid ==
rng = np.random.default_rng(SEED)
calendars = {'weekly': (weekly_cal, 52), 'monthly': (monthly_cal, 12)}
cells_rows, summary_rows = [], []
real_cache: dict = {}
cell_i, cell_total = 0, len(calendars) * len(UNIVERSES) * len(LOOKBACKS)
log.info('%s STEP 3: simulating %d (rebalance x universe x lookback) cells x %d N-values x %d-draw null...',
          elapsed(), cell_total, len(N_LIST), N_DRAWS)

for rebalance, (cal_df, ppy) in calendars.items():
    for uni_name, adv_cutoff in UNIVERSES.items():
        for lb_name in LOOKBACKS:
            cell_i += 1
            sig_col = f'sig_{lb_name}'
            real_by_n, pools_for_null, contrib, k_lt_n = run_cell(cal_df, adv_cutoff, sig_col)
            for n in N_LIST:
                real_df = real_by_n[n]
                real_cache[(uni_name, lb_name, rebalance, n)] = real_df[['net_ret', 'spy_fwd']].copy()
                null_mat = run_null_for_n(pools_for_null, n, rng)
                entry_dates = real_df.index.values
                win_reads = {}
                for win_name, (ws, we) in WINDOWS.items():
                    mask = (entry_dates >= ws) & (entry_dates <= we)
                    win_reads[win_name] = window_read(real_df, null_mat, mask, ppy)
                    cells_rows.append(dict(universe=uni_name, N=n, lookback=lb_name, rebalance=rebalance,
                                             window=win_name, **win_reads[win_name]))
                top5 = top5_share_of(contrib[n])
                whole, h1, h2 = win_reads['whole'], win_reads['halfA'], win_reads['halfB']
                pass_flag = bool(pd.notna(h1['excess_ann_return']) and h1['excess_ann_return'] > 0
                                   and pd.notna(h2['excess_ann_return']) and h2['excess_ann_return'] > 0
                                   and pd.notna(whole['t_alpha']) and whole['t_alpha'] >= 2.0
                                   and pd.notna(whole['max_dd']) and pd.notna(whole['max_dd_spy'])
                                   and abs(whole['max_dd']) <= 1.25 * abs(whole['max_dd_spy'])
                                   and pd.notna(h1['null_pct_excess']) and h1['null_pct_excess'] >= 95.0
                                   and pd.notna(h2['null_pct_excess']) and h2['null_pct_excess'] >= 95.0)
                summary_rows.append(dict(universe=uni_name, N=n, lookback=lb_name, rebalance=rebalance,
                    ann_net=whole['ann_return_net'], spy=whole['ann_return_spy'], excess_whole=whole['excess_ann_return'],
                    excess_H1=h1['excess_ann_return'], excess_H2=h2['excess_ann_return'], alpha_t=whole['t_alpha'],
                    maxDD_book=whole['max_dd'], maxDD_spy=whole['max_dd_spy'], null_pct_H1=h1['null_pct_excess'],
                    null_pct_H2=h2['null_pct_excess'], turnover=whole['turnover_avg'], top5_share=top5, PASS=pass_flag))
            log.info('%s cell %d/%d %s/%s/%s done (periods k<N count by N: %s)', elapsed(), cell_i, cell_total,
                      rebalance, uni_name, lb_name, k_lt_n)
            pd.DataFrame(cells_rows).to_csv(OUT / '1700d_cells.csv', index=False)
            pd.DataFrame(summary_rows).to_csv(OUT / '1700d_summary_tmp.csv', index=False)

log.info('%s STEP 3 done: %d cells x 3 windows = %d rows written', elapsed(), cell_total * len(N_LIST), len(cells_rows))
summary_df = pd.DataFrame(summary_rows)

# =========================================================================== best-cell by-year ==
weekly_summary = summary_df[summary_df.rebalance == 'weekly'].sort_values('excess_H2', ascending=False)
best = weekly_summary.iloc[0]
best_key = (best.universe, best.lookback, best.rebalance, int(best.N))
best_real = real_cache[best_key].reset_index()
yr_rows = []
for yr, sub in best_real.groupby(best_real.entry_date.dt.year):
    book_ann = np.prod(1 + sub['net_ret'].fillna(0)) - 1
    spy_ann = np.prod(1 + sub['spy_fwd'].fillna(0)) - 1
    yr_rows.append(dict(year=int(yr), n_periods=len(sub), book_return=book_ann, spy_return=spy_ann, excess=book_ann - spy_ann))
by_year_df = pd.DataFrame(yr_rows)
by_year_df.to_csv(OUT / '1700d_best_by_year.csv', index=False)
log.info('%s best cell (weekly, by H2 excess) = %s -- wrote 1700d_best_by_year.csv (%d years)', elapsed(), best_key, len(by_year_df))

# ================================================================================ RESULT.md ==
def fmt(v, pct=True, dp=1):
    if pd.isna(v):
        return 'NA'
    return f'{100*v:+.{dp}f}%' if pct else f'{v:.2f}'


n_pass = int(summary_df['PASS'].sum())
bonf = 0.05 * 0.05 * len(summary_df)
lines = []
lines.append('# RESULT -- cell 1,700d: owner\'s "top 10 works" parameter grid vs SPY')
lines.append('')
lines.append(f'PREREG_1700d.md + Amendment 1 (FROZEN). 72 cells (2 universe x 3 N x 6 lookback x 2 rebalance) '
             f'x 3 windows = {len(cells_rows)} rows in 1700d_cells.csv. Null = {N_DRAWS} draws/cell '
             f'(PREREG target 1,000; reduced for runtime -- see script docstring). Windows: whole '
             f'{WIN_START.date()}..{WIN_END.date()}, H1 ..{(HALF_CUT-pd.Timedelta(days=1)).date()}, '
             f'H2 {HALF_CUT.date()}..')
lines.append('')
lines.append('## 72-cell table, WEEKLY cells first then MONTHLY, each sorted by H2 excess (owner prefers weekly)')
lines.append('')
cols = ['universe', 'N', 'lookback', 'rebalance', 'ann_net', 'spy', 'excess_whole', 'excess_H1', 'excess_H2',
        'alpha_t', 'maxDD_book', 'maxDD_spy', 'null_pct_H1', 'null_pct_H2', 'turnover', 'top5_share', 'PASS']
lines.append('| ' + ' | '.join(cols) + ' |')
lines.append('|' + '---|' * len(cols))
ordered = pd.concat([summary_df[summary_df.rebalance == 'weekly'].sort_values('excess_H2', ascending=False),
                      summary_df[summary_df.rebalance == 'monthly'].sort_values('excess_H2', ascending=False)])
for _, r in ordered.iterrows():
    lines.append(f"| {r.universe} | {int(r.N)} | {r.lookback} | {r.rebalance} | {fmt(r.ann_net)} | {fmt(r.spy)} | "
                 f"{fmt(r.excess_whole)} | {fmt(r.excess_H1)} | {fmt(r.excess_H2)} | {fmt(r.alpha_t, pct=False)} | "
                 f"{fmt(r.maxDD_book)} | {fmt(r.maxDD_spy)} | {fmt(r.null_pct_H1, pct=False)} | "
                 f"{fmt(r.null_pct_H2, pct=False)} | {fmt(r.turnover)} | {fmt(r.top5_share)} | "
                 f"{'PASS' if r.PASS else 'fail'} |")
lines.append('')
lines.append(f'## Pass list ({n_pass}/72): ' + (', '.join(f"{r.universe}/N{int(r.N)}/{r.lookback}/{r.rebalance}"
             for _, r in summary_df[summary_df.PASS].iterrows()) if n_pass else 'NONE -- no cell clears both-halves '
             'excess>0 + alpha t>=2.0 + maxDD<=1.25xSPY + null pct>=95 both halves.'))
lines.append(f'**Bonferroni**: with 72 cells at a 5% per-half chance bar, ~{bonf:.1f} cells are expected to clear '
             f'BOTH halves by chance alone; {n_pass} observed passes {"is" if n_pass <= bonf*3 else "exceeds"} '
             f'{"within" if n_pass <= bonf*3 else "well above"} that noise floor.')
lines.append('')
lines.append(f'## Best cell by H2 excess among WEEKLY cells: {best.universe} N={int(best.N)} {best.lookback} '
             f'weekly ({"PASS" if best.PASS else "fails the pass bar"}) -- by-year vs SPY')
lines.append('')
lines.append('| Year | Periods | Book | SPY | Excess |')
lines.append('|---|---|---|---|---|')
for _, r in by_year_df.iterrows():
    lines.append(f"| {int(r.year)} | {int(r.n_periods)} | {fmt(r.book_return)} | {fmt(r.spy_return)} | {fmt(r.excess)} |")
lines.append('')
lines.append('## Owner\'s two named cells (N=10, weekly; both universes; full reads even though shown above)')
lines.append('')
lines.append('| universe | lookback | ann_net | spy | excess_whole | excess_H1 | excess_H2 | alpha_t | null_H1 | null_H2 | PASS |')
lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
for uni in ('U1', 'U2'):
    for lb in ('L1_52w', 'L3_13m'):
        r = summary_df[(summary_df.universe == uni) & (summary_df.N == 10) & (summary_df.lookback == lb)
                        & (summary_df.rebalance == 'weekly')].iloc[0]
        lines.append(f"| {uni} | {lb} | {fmt(r.ann_net)} | {fmt(r.spy)} | {fmt(r.excess_whole)} | {fmt(r.excess_H1)} | "
                     f"{fmt(r.excess_H2)} | {fmt(r.alpha_t, pct=False)} | {fmt(r.null_pct_H1, pct=False)} | "
                     f"{fmt(r.null_pct_H2, pct=False)} | {'PASS' if r.PASS else 'fail'} |")
lines.append('')
lines.append('## Methodology notes')
lines.append('- Universe fixed for both U1/U2: price>=$10 at the signal-day close, >=273 trading days of history, '
             'ETF/ETN/fund/trust/warrant/unit/preferred/right-by-name and `^Z[A-Z]ZZT$` excluded; only the ADV20 '
             'cutoff ($20M vs $200M) differs.')
lines.append('- Cost: 5bps/side + half the spread proxy ((high-low)/close x0.1, capped 20bps) on traded dollars, '
             'charged on buys and sells vs the prior portfolio; price return, dividends not separately modeled.')
lines.append('- top5_share = whole-window top-5 NAMES\' summed equal-weight contribution to period returns, '
             'divided by the sum of all POSITIVE per-name contributions (lottery check; NaN if no positive names).')
lines.append(f'- Null = {N_DRAWS} random same-N draws from the same eligible pool each period (not the PREREG\'s '
             '1,000 -- runtime, stated above); percentile is of ann. excess-over-SPY among the draws.')
lines.append('- No parameter outside this grid was tried after seeing numbers, per PREREG.')
(OUT / 'RESULT_1700d.md').write_text('\n'.join(lines) + '\n')
log.info('%s wrote RESULT_1700d.md (%d lines)', elapsed(), len(lines))
log.info('%s ALL DONE', elapsed())
