#!/usr/bin/env python3
"""Cell 1,700b -- weekly momentum sleeve, literature-definition universe fix, PHASE 1 ONLY.

PREREG: research/momentum_weekly/PREREG_1700b.md (FROZEN). Fixes cell 1,700's universe-definition gap
(RESULT_1700.md: the naive top-N-by-trailing-return universe selected micro-cap hype names and an outright
warrant, QBTS+) with the literature's definition: domestic common stock only, SPAC names excluded, a
size/liquidity floor as a market-cap proxy, and conventional portfolio widths (decile / top-50 / top-20,
weekly, plus a monthly twin of the decile book). Derived from research/momentum_weekly/1700_momentum.py
(kept unchanged); same data, same cost model, same windows, same pass bar.

Price source (primary): Databento EQUS.SUMMARY point-in-time daily OHLCV, same two parquet files as 1700.
Common-stock source (THE FIX): Databento's point-in-time definitions feed,
data/research/databento/pit_definition/def_YYYYMM.parquet, field `security_type`, matched by (symbol,
calendar month) so a reclass is knowable only from that month forward. Decoded empirically against known
tickers before freezing anything (see PREREG_1700b.md): C=common (AAPL, QBTS), Q=ETF (SPY, QQQ, IWM, TQQQ,
leveraged single-name ETFs), W=warrant (QBTS+), P=preferred (ABR-D), U=unit (AACBU), R=rights (AACBR),
L=limited partnership (AB, ARLP, BBU), V=royalty trust (BPT, CRT, PBT), O/A=foreign-ordinary/ADR (ACN,
ABEV, AEG), S=small mixed REIT/closed-end-fund bucket. Kept class: C only.
SPAC-name source: data/research/orb_asset_class_map_20260711.csv 'name' column, pattern 'acquisition'
(case-insensitive) -- catches a SPAC's own common unit, which is still security_type=='C'.

Cross-checks (not the primary source): data/cache.db daily_bars (read-only) and
research/overnight_high/panel_2024_2026.parquet dvol20, same mechanism as 1700.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research' / 'momentum_weekly'
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(OUT / '1700b_momentum.log'), filemode='w', level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700b')
log.addHandler(logging.StreamHandler(sys.stdout))

SEED = 17000
N_DRAWS = 1000
VARIANT = 'M2'  # 12-1 only, per PREREG_1700b -- the variant/N sweep is retired by this cell
PRICE_MIN = 10.0
ADV_MIN = 20_000_000.0
DECILE_FRAC = 0.10
CAPITAL = 65_000.0
WIN_START = pd.Timestamp('2025-07-01')
WIN_END = pd.Timestamp('2026-09-30')
HALF_CUT = pd.Timestamp('2026-01-01')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')

t0 = time.time()


def elapsed() -> str:
    """Seconds since script start, formatted for log lines."""
    return f'{time.time() - t0:6.1f}s'


# =================================================================== load ==
log.info('STEP 1: loading EQUS.SUMMARY point-in-time daily panels (phase 1, on-disk only)')
cols = ['bar_date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
p1 = pd.read_parquet(ROOT / 'data/research/databento/equs_daily_2024H2.parquet', columns=cols)
p2 = pd.read_parquet(ROOT / 'data/research/databento/equs_daily_2025_2026.parquet', columns=cols)
panel_raw = pd.concat([p1, p2], ignore_index=True)
panel_raw['bar_date'] = pd.to_datetime(panel_raw['bar_date'])
panel_raw = (panel_raw.drop_duplicates(subset=['symbol', 'bar_date'], keep='last')
             .sort_values(['symbol', 'bar_date']).reset_index(drop=True))
log.info('%s loaded panel_raw rows=%d symbols=%d dates %s..%s', elapsed(),
         len(panel_raw), panel_raw.symbol.nunique(), panel_raw.bar_date.min(), panel_raw.bar_date.max())

bad_px = (panel_raw.open <= 0) | (panel_raw.high <= 0) | (panel_raw.low <= 0) | (panel_raw.close <= 0)
if bad_px.any():
    log.warning('%d / %d rows have a non-positive OHLC price -- dropped', int(bad_px.sum()), len(panel_raw))
    panel_raw = panel_raw[~bad_px].reset_index(drop=True)

spy = panel_raw.loc[panel_raw.symbol == 'SPY', ['bar_date', 'open', 'close']].sort_values('bar_date').reset_index(drop=True)
if spy.empty:
    log.error('SPY not found in panel_raw -- benchmark series unavailable, aborting')
    raise SystemExit('SPY missing from EQUS panel')
log.info('%s SPY pulled BEFORE exclusions (benchmark only, not a candidate holding): rows=%d %s..%s',
         elapsed(), len(spy), spy.bar_date.min(), spy.bar_date.max())

all_syms_raw = set(s for s in panel_raw['symbol'].unique() if s is not None)
n_none_sym = panel_raw['symbol'].isna().sum()
if n_none_sym:
    log.warning('%d rows have a null symbol -- dropped before any filtering', n_none_sym)
    panel_raw = panel_raw[panel_raw['symbol'].notna()].reset_index(drop=True)
n_test_sym = sum(1 for s in all_syms_raw if TEST_RE.match(s))

# ================================================= common-stock filter (THE FIX) ==
log.info('%s STEP 1b: loading point-in-time definitions feed (security_type), one file per calendar month', elapsed())
def_dir = ROOT / 'data/research/databento/pit_definition'
def_files = sorted(def_dir.glob('def_*.parquet'))
type_frames = []
for f in def_files:
    ym = f.stem.replace('def_', '')
    ym_fmt = f'{ym[:4]}-{ym[4:]}'
    d = pd.read_parquet(f, columns=['raw_symbol', 'security_type']).rename(columns={'raw_symbol': 'symbol'})
    d = d.drop_duplicates(subset=['symbol'], keep='last')
    d['year_month'] = ym_fmt
    type_frames.append(d[['year_month', 'symbol', 'security_type']])
type_by_month = pd.concat(type_frames, ignore_index=True)
log.info('%s loaded %d monthly definition files, %d (month,symbol) rows, months %s..%s',
         elapsed(), len(def_files), len(type_by_month), type_by_month.year_month.min(), type_by_month.year_month.max())

type_last = type_by_month.sort_values('year_month').drop_duplicates('symbol', keep='last')
type_counts = type_last.security_type.value_counts().to_dict()
n_common_syms = int((type_last.security_type == 'C').sum())
log.info('%s security_type counts (unique symbols, last-seen in window): %s -> %d kept as common (C)',
         elapsed(), type_counts, n_common_syms)

ac = pd.read_csv(ROOT / 'data/research/orb_asset_class_map_20260711.csv')
spac_syms = set(ac.loc[ac.name.str.contains('acquisition', case=False, na=False), 'symbol'])
ac_syms = set(ac['symbol'])
n_spac_sym = len(spac_syms)
n_ac_covered = len(all_syms_raw & ac_syms)
log.info('%s SPAC-name filter: %d symbols with "acquisition" in name (orb_asset_class_map); that map covers '
         '%d / %d raw-panel symbols (uncovered symbols cannot be name-checked, undercount only)',
         elapsed(), n_spac_sym, n_ac_covered, len(all_syms_raw))

panel_raw['year_month'] = panel_raw['bar_date'].dt.strftime('%Y-%m')
panel_raw = panel_raw.merge(type_by_month, on=['year_month', 'symbol'], how='left')
n_unmatched = int(panel_raw['security_type'].isna().sum())
log.warning('%s %d / %d panel rows have no (symbol,month) match in the definitions feed -- treated as NOT '
            'common stock (excluded), undercount only', elapsed(), n_unmatched, len(panel_raw))
panel_raw['is_common'] = panel_raw['security_type'] == 'C'
panel_raw['is_spac_name'] = panel_raw['symbol'].isin(spac_syms)

# ---------------------------------------------------- loose liquidity prefilter (speed only, cannot bias results) --
panel_raw['dvol'] = panel_raw['close'] * panel_raw['volume']
sym_stats = panel_raw.groupby('symbol').agg(max_close=('close', 'max'), max_dvol=('dvol', 'max'))
keep_syms = set(sym_stats[(sym_stats.max_close >= PRICE_MIN) & (sym_stats.max_dvol >= 5_000_000)].index)
log.info('%s loose liquidity prefilter: %d / %d symbols can ever clear $%d price & $5M single-day $vol '
         '(necessary, not sufficient, vs the real $%.0fM ADV20 gate below)',
         elapsed(), len(keep_syms), len(all_syms_raw), int(PRICE_MIN), ADV_MIN / 1e6)

panel = panel_raw[panel_raw.symbol.isin(keep_syms) & panel_raw.is_common & ~panel_raw.is_spac_name
                   & ~panel_raw.symbol.map(lambda s: bool(TEST_RE.match(s)))].copy()
log.info('%s tradable universe after liquidity prefilter + common-stock + SPAC-name + test-ticker exclusions: '
         '%d symbols, %d rows', elapsed(), panel.symbol.nunique(), len(panel))

# =============================================================== features ==
panel = panel.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
g = panel.groupby('symbol', sort=False)
log.info('%s computing adv20 and the 12-1 momentum signal...', elapsed())
panel['adv20'] = g['dvol'].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
panel['spread_proxy'] = (((panel['high'] - panel['low']) / panel['close']).clip(lower=0) * 0.1).clip(upper=0.002)
panel['close_lag21'] = g['close'].shift(21)
panel['close_lag252'] = g['close'].shift(252)
panel['sig_M2'] = panel['close_lag21'] / panel['close_lag252'] - 1
log.info('%s features done', elapsed())

# ============================================================ week calendar ==
cal = pd.Series(sorted(panel_raw['bar_date'].unique()))
iso = pd.DatetimeIndex(cal).isocalendar()
wk = pd.DataFrame({'date': cal, 'yr': iso.year.values, 'wk': iso.week.values})
first = wk.groupby(['yr', 'wk'])['date'].min()
last = wk.groupby(['yr', 'wk'])['date'].max()
weeks = (pd.DataFrame({'entry_date': first, 'signal_date': last}).reset_index()
         .sort_values('entry_date').reset_index(drop=True))
weeks['prior_signal_date'] = weeks['signal_date'].shift(1)
weeks['next_entry_date'] = weeks['entry_date'].shift(-1)
weeks = weeks.dropna(subset=['prior_signal_date', 'next_entry_date']).reset_index(drop=True)
weeks = weeks[(weeks.entry_date >= WIN_START) & (weeks.entry_date <= WIN_END)].reset_index(drop=True)
log.info('%s phase-1 weekly rebalances: %d, entry %s..%s', elapsed(), len(weeks),
         weeks.entry_date.min().date(), weeks.entry_date.max().date())

# ============================================================ month calendar ==
cal_df = pd.DataFrame({'date': cal})
cal_df['ym'] = cal_df['date'].dt.strftime('%Y-%m')
m_first = cal_df.groupby('ym')['date'].min()
m_last = cal_df.groupby('ym')['date'].max()
months = (pd.DataFrame({'entry_date': m_first, 'signal_date': m_last}).reset_index()
          .sort_values('entry_date').reset_index(drop=True))
months['prior_signal_date'] = months['signal_date'].shift(1)
months['next_entry_date'] = months['entry_date'].shift(-1)
months = months.dropna(subset=['prior_signal_date', 'next_entry_date']).reset_index(drop=True)
months = months[(months.entry_date >= WIN_START) & (months.entry_date <= WIN_END)].reset_index(drop=True)
log.info('%s phase-1 monthly rebalances: %d, entry %s..%s', elapsed(), len(months),
         months.entry_date.min().date(), months.entry_date.max().date())

all_dates = pd.Index(sorted(set(weeks.entry_date) | set(weeks.next_entry_date)
                             | set(months.entry_date) | set(months.next_entry_date)))

# ======================================================= open/cost pivots ==
log.info('%s pivoting open price and cost-rate on %d entry-equivalent dates...', elapsed(), len(all_dates))
mon_rows = panel[panel.bar_date.isin(all_dates)]
open_piv = mon_rows.pivot_table(index='bar_date', columns='symbol', values='open', aggfunc='last').reindex(all_dates)
spread_piv = mon_rows.pivot_table(index='bar_date', columns='symbol', values='spread_proxy', aggfunc='last').reindex(all_dates)
cost_rate_piv = 0.0005 + 0.5 * spread_piv.fillna(0.002)
log.info('%s pivots built: shape %s', elapsed(), open_piv.shape)

spy_idx = spy.set_index('bar_date').reindex(all_dates)


def fwd_ret_row(entry_date, next_entry_date):
    """Per-symbol open-to-open forward return between two specific dates (explicit index lookup, NOT a
    positional shift -- required because the weekly and monthly calendars interleave in the same pivot)."""
    return open_piv.loc[next_entry_date] / open_piv.loc[entry_date] - 1


def spy_fwd(entry_date, next_entry_date):
    """SPY open-to-open forward return between two specific dates."""
    a, b = spy_idx['open'].get(entry_date), spy_idx['open'].get(next_entry_date)
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return np.nan
    return b / a - 1


# ==================================================== universe benchmark ==
def build_bench(cal_df_):
    """Equal-weight forward return of the full eligible pool each period (for alpha-vs-universe)."""
    out = {}
    for _, row in cal_df_.iterrows():
        d, ed, nxt = row['prior_signal_date'], row['entry_date'], row['next_entry_date']
        pool = panel.loc[(panel.bar_date == d) & (panel['close'] >= PRICE_MIN) & (panel['adv20'] >= ADV_MIN), 'symbol']
        if len(pool) == 0:
            out[ed] = np.nan
            continue
        out[ed] = fwd_ret_row(ed, nxt).reindex(pool.values).mean(skipna=True)
    return pd.Series(out).reindex(cal_df_.entry_date.values)


def build_spy_fwd(cal_df_):
    """SPY forward return series aligned to one calendar's entry dates."""
    out = {row['entry_date']: spy_fwd(row['entry_date'], row['next_entry_date']) for _, row in cal_df_.iterrows()}
    return pd.Series(out).reindex(cal_df_.entry_date.values)


# ============================================================= simulation ==
def run_real(cal_df_: pd.DataFrame, n_or_frac, is_frac: bool) -> pd.DataFrame:
    """Momentum-ranked simulation for one book over the given calendar (weekly or monthly rows).
    n_or_frac is a fixed N (is_frac=False) or a decile fraction of that period's pool (is_frac=True)."""
    sig_col = f'sig_{VARIANT}'
    prev_port: set[str] = set()
    rows = []
    for _, wrow in cal_df_.iterrows():
        sdate, edate, nxt = wrow['prior_signal_date'], wrow['entry_date'], wrow['next_entry_date']
        day = panel.loc[(panel.bar_date == sdate) & (panel['close'] >= PRICE_MIN) & (panel['adv20'] >= ADV_MIN)
                         & panel[sig_col].notna(), ['symbol', sig_col, 'adv20']]
        k = len(day)
        n = max(1, round(n_or_frac * k)) if is_frac else int(n_or_frac)
        if k < n:
            log.warning('book is_frac=%s n_or_frac=%s period=%s: eligible pool %d < N=%d -- taking all available',
                        is_frac, n_or_frac, edate.date(), k, n)
        port = set(day.nlargest(n, sig_col)['symbol']) if k else set()
        bought, sold = port - prev_port, prev_port - port
        if port:
            fwd = fwd_ret_row(edate, nxt).reindex(list(port))
            n_missing = int(fwd.isna().sum())
            gross = fwd.fillna(0).mean()
        else:
            n_missing, gross = 0, np.nan
        denom = len(port) if port else max(n, 1)
        cost = 0.0
        if bought or sold:
            cr = cost_rate_piv.loc[edate]
            cost = (cr.reindex(list(bought)).fillna(0.002).sum() + cr.reindex(list(sold)).fillna(0.002).sum()) / denom
        turnover = len(bought) / denom
        adv_ratio = np.nan
        if port:
            advs = day.set_index('symbol').reindex(list(port))['adv20']
            adv_ratio = ((CAPITAL / denom) / advs).mean()
        rows.append(dict(entry_date=edate, n_eligible=k, port_n=len(port), gross_ret=gross,
                          cost=cost, net_ret=(gross - cost if pd.notna(gross) else np.nan),
                          turnover=turnover, n_fwd_missing=n_missing, adv_impact=adv_ratio))
        prev_port = port
    return pd.DataFrame(rows).set_index('entry_date')


def run_null(cal_df_: pd.DataFrame, n_or_frac, is_frac: bool, rng: np.random.Generator) -> np.ndarray:
    """1,000 random count-matched draws from the SAME eligible pool each period -> (1000, n_periods) gross returns."""
    sig_col = f'sig_{VARIANT}'
    cols_out = []
    for _, wrow in cal_df_.iterrows():
        sdate, edate, nxt = wrow['prior_signal_date'], wrow['entry_date'], wrow['next_entry_date']
        pool = panel.loc[(panel.bar_date == sdate) & (panel['close'] >= PRICE_MIN) & (panel['adv20'] >= ADV_MIN)
                          & panel[sig_col].notna(), 'symbol'].values
        k = len(pool)
        if k == 0:
            cols_out.append(np.full(N_DRAWS, np.nan))
            continue
        n = max(1, round(n_or_frac * k)) if is_frac else min(int(n_or_frac), k)
        rets = fwd_ret_row(edate, nxt).reindex(pool).fillna(0).values
        rand = rng.random((N_DRAWS, k))
        idx = np.argpartition(rand, n - 1, axis=1)[:, :n]
        cols_out.append(rets[idx].mean(axis=1))
    return np.column_stack(cols_out)


def ann_stats(returns, periods_per_year: int = 52) -> dict:
    """Annualised return/vol/Sharpe/drawdown stats for a period-return series at the given frequency."""
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


BOOKS = [
    ('decile_weekly', weeks, DECILE_FRAC, True, 52),
    ('top50_weekly', weeks, 50, False, 52),
    ('top20_weekly', weeks, 20, False, 52),
    ('decile_monthly', months, DECILE_FRAC, True, 12),
]

log.info('%s STEP 2: simulating %d books, real + %d-draw null each', elapsed(), len(BOOKS), N_DRAWS)
rng = np.random.default_rng(SEED)
all_real, all_null, all_bench, all_spyfwd = {}, {}, {}, {}
for name, cal_df_, nf, is_frac, ppy in BOOKS:
    all_real[name] = run_real(cal_df_, nf, is_frac)
    all_null[name] = run_null(cal_df_, nf, is_frac, rng)
    all_bench[name] = build_bench(cal_df_)
    all_spyfwd[name] = build_spy_fwd(cal_df_)
    log.info('%s  done book=%s n_periods=%d median_pool_n=%.0f median_port_n=%.0f', elapsed(), name, len(cal_df_),
             np.nanmedian(all_real[name]['n_eligible']), np.nanmedian(all_real[name]['port_n']))

WINDOWS = {
    'halfA': (WIN_START, HALF_CUT - pd.Timedelta(days=1)),
    'halfB': (HALF_CUT, WIN_END),
    'whole': (WIN_START, WIN_END),
}

reads, period_rows = [], []
for name, cal_df_, nf, is_frac, ppy in BOOKS:
    real_df = all_real[name]
    null_mat = all_null[name]
    bench_ser = all_bench[name]
    spy_fwd_ser = all_spyfwd[name]
    entry_dates = cal_df_.entry_date.values
    freq_label = 'monthly' if ppy == 12 else 'weekly'
    for edate, r in real_df.iterrows():
        period_rows.append(dict(book=name, freq=freq_label, entry_date=edate, **r.to_dict()))
    for win_name, (ws, we) in WINDOWS.items():
        mask = (entry_dates >= ws) & (entry_dates <= we)
        sub = real_df.loc[mask]
        net_stats = ann_stats(sub['net_ret'].values, ppy)
        gross_stats = ann_stats(sub['gross_ret'].values, ppy)
        spy_sub = spy_fwd_ser.reindex(sub.index).values
        valid = ~np.isnan(sub['net_ret'].values) & ~np.isnan(spy_sub)
        if valid.sum() > 2 and np.var(spy_sub[valid]) > 0:
            beta = np.cov(sub['net_ret'].values[valid], spy_sub[valid])[0, 1] / np.var(spy_sub[valid], ddof=1)
        else:
            beta = np.nan
        if valid.sum() and pd.notna(beta):
            ann_alpha_spy = np.mean(sub['net_ret'].values[valid] - beta * spy_sub[valid]) * ppy
        else:
            ann_alpha_spy = np.nan
        bench_sub = bench_ser.reindex(sub.index).values
        diff = sub['net_ret'].values - bench_sub
        ann_alpha_universe = np.nanmean(diff) * ppy if np.isfinite(diff).any() else np.nan
        cost_drag_annual = np.nanmean(sub['cost'].values) * ppy
        turnover_avg = np.nanmean(sub['turnover'].values)
        null_sub = null_mat[:, mask]
        null_ann_ret = np.array([ann_stats(null_sub[i], ppy)['ann_return'] for i in range(N_DRAWS)])
        null_sharpe = np.array([ann_stats(null_sub[i], ppy)['sharpe'] for i in range(N_DRAWS)])
        pct_ret = 100 * np.nanmean(null_ann_ret <= net_stats['ann_return']) if pd.notna(net_stats['ann_return']) else np.nan
        pct_sharpe = 100 * np.nanmean(null_sharpe <= net_stats['sharpe']) if pd.notna(net_stats['sharpe']) else np.nan
        reads.append(dict(
            book=name, freq=freq_label, window=win_name, n_periods=net_stats['n_periods'],
            median_pool_n=float(np.nanmedian(sub['n_eligible'].values)) if len(sub) else np.nan,
            median_port_n=float(np.nanmedian(sub['port_n'].values)) if len(sub) else np.nan,
            ann_return_net=net_stats['ann_return'], ann_return_gross=gross_stats['ann_return'],
            ann_vol=net_stats['ann_vol'], sharpe_net=net_stats['sharpe'], max_dd=net_stats['max_dd'],
            worst_period=net_stats['worst_period'], best_period=net_stats['best_period'],
            green_share=net_stats['green_share'], turnover_avg=turnover_avg, cost_drag_annual=cost_drag_annual,
            beta_spy=beta, ann_alpha_spy=ann_alpha_spy, ann_alpha_universe=ann_alpha_universe,
            null_pct_return=pct_ret, null_pct_sharpe=pct_sharpe,
            null_ann_ret_mean=np.nanmean(null_ann_ret), null_ann_ret_std=np.nanstd(null_ann_ret),
            null_sharpe_mean=np.nanmean(null_sharpe), null_sharpe_std=np.nanstd(null_sharpe),
            n_fwd_missing=int(sub['n_fwd_missing'].sum())))

reads_df = pd.DataFrame(reads)
period_df = pd.DataFrame(period_rows)
reads_df.to_csv(OUT / '1700b_reads.csv', index=False)
period_df.to_csv(OUT / '1700b_weekly.csv', index=False)
log.info('%s wrote 1700b_reads.csv (%d rows, expect 12) and 1700b_weekly.csv (%d rows)',
         elapsed(), len(reads_df), len(period_df))


def passes_bar(name: str) -> bool:
    """Identical pass bar to PREREG_1700: alpha>=8%/yr, Sharpe>=1.0, null pct(return)>=95, both halves;
    cost drag<4%/yr; max DD<=20%."""
    a = reads_df[(reads_df.book == name) & (reads_df.window == 'halfA')].iloc[0]
    b = reads_df[(reads_df.book == name) & (reads_df.window == 'halfB')].iloc[0]

    def cond(row):
        return (pd.notna(row.ann_alpha_spy) and row.ann_alpha_spy >= 0.08
                and pd.notna(row.sharpe_net) and row.sharpe_net >= 1.0
                and pd.notna(row.null_pct_return) and row.null_pct_return >= 95
                and row.cost_drag_annual < 0.04 and abs(row.max_dd) <= 0.20)
    return cond(a) and cond(b)


# ================================================================ cross-checks ==
log.info('%s STEP 3: cross-check 1 -- cache.db daily_bars (read-only) vs EQUS close, 300-row sample', elapsed())
sample = panel[['symbol', 'bar_date', 'close']].sample(min(300, len(panel)), random_state=SEED)
match = checked = 0
con = None
for attempt in range(3):
    try:
        con = sqlite3.connect('file:' + str(ROOT / 'data/cache.db') + '?mode=ro', uri=True, timeout=30)
        break
    except sqlite3.OperationalError as e:
        log.warning('cache.db locked, retry %d/3 in 30s: %s', attempt + 1, e)
        time.sleep(30)
if con is None:
    log.error('cache.db unreachable after 3 attempts -- cross-check 1 SKIPPED (not fatal, EQUS is the primary source)')
else:
    cur = con.cursor()
    for _, r in sample.iterrows():
        cur.execute('SELECT close FROM daily_bars WHERE symbol=? AND bar_date=?', (r.symbol, r.bar_date.strftime('%Y-%m-%d')))
        row = cur.fetchone()
        checked += 1
        if row and row[0] and r.close and abs(row[0] - r.close) / r.close < 0.005:
            match += 1
    con.close()
    log.info('%s cache.db cross-check: %d/%d sampled closes within 0.5%% (rest = not cached on that date or diverge)',
             elapsed(), match, checked)

log.info('%s cross-check 2 -- overnight_high panel dvol20 ($ volume) vs our adv20 ($ volume), full join', elapsed())
ov = pd.read_parquet(ROOT / 'research/overnight_high/panel_2024_2026.parquet', columns=['symbol', 'bar_date', 'dvol20'])
ov['bar_date'] = pd.to_datetime(ov['bar_date'])
merged = panel[['symbol', 'bar_date', 'adv20']].merge(ov, on=['symbol', 'bar_date'], how='inner')
merged = merged.dropna(subset=['adv20', 'dvol20'])
merged = merged[merged.dvol20 > 0]
if len(merged):
    reldiff = (merged.adv20 - merged.dvol20).abs() / merged.dvol20
    pct_close = float((reldiff < 0.10).mean() * 100)
    log.info('%s overnight-panel cross-check: %d matched rows, %.1f%% within 10%% $-volume agreement, median reldiff %.3f',
             elapsed(), len(merged), pct_close, reldiff.median())
else:
    pct_close = float('nan')
    log.warning('overnight-panel cross-check: no matched rows (different symbol universes)')

# =================================================================== RESULT.md ==
log.info('%s STEP 4: writing RESULT_1700b.md', elapsed())
lines = []
lines.append('# RESULT -- cell 1,700b: weekly momentum sleeve, literature-definition universe fix')
lines.append('')
lines.append('PREREG_1700b.md (FROZEN). Fixes cell 1,700 (RESULT_1700.md): the naive top-N universe selected '
              'micro-cap hype names and a warrant (QBTS+). This cell restricts to domestic common stock + a '
              'size/liquidity floor + conventional portfolio widths, pre-declared before any number was read.')
lines.append(f'Window {WIN_START.date()}..{WIN_END.date()}: {len(weeks)} weekly, {len(months)} monthly rebalances '
              f'(halfA {WINDOWS["halfA"][0].date()}..{WINDOWS["halfA"][1].date()}, '
              f'halfB {WINDOWS["halfB"][0].date()}..{WINDOWS["halfB"][1].date()}).')
lines.append(f'Common-stock filter (Databento point-in-time security_type, {len(def_files)} monthly files, unique '
              f'symbols last-seen in window): {type_counts} -> **{n_common_syms} kept as C (common)**; excluded '
              f'by class: Q=ETF, P=preferred, W=warrant, U=unit, R=rights, L=LP, V=royalty-trust, '
              f'O/A=foreign-ordinary/ADR, S=mixed REIT/closed-end-fund. {n_unmatched} panel rows unmatched to a '
              f'(symbol,month) definition -> excluded (undercount only). SPAC-name filter: {n_spac_sym} symbols '
              f'with "acquisition" in name (orb_asset_class_map, covering {n_ac_covered}/{len(all_syms_raw)} raw '
              f'symbols). {n_test_sym} test-ticker symbols (`^Z[A-Z]ZZT$`). Price >= ${PRICE_MIN:.0f}, adv20 >= '
              f'${ADV_MIN/1e6:.0f}M, both at the signal date.')
lines.append(f'Cross-checks: cache.db daily_bars (read-only) {match}/{checked} sampled closes within 0.5%; '
              f'overnight_high panel $-volume (dvol20) agreement {pct_close:.1f}% of {len(merged)} matched rows within 10%.')
lines.append('')
lines.append('## Reads (12 = 4 books x 3 windows; annualised net-of-cost unless marked)')
lines.append('')
rcols = ['book', 'freq', 'window', 'n_periods', 'median_pool_n', 'median_port_n', 'ann_return_net',
         'ann_alpha_spy', 'sharpe_net', 'max_dd', 'turnover_avg', 'cost_drag_annual', 'null_pct_return', 'null_pct_sharpe']
lines.append('| ' + ' | '.join(rcols) + ' |')
lines.append('|' + '---|' * len(rcols))
for _, r in reads_df.sort_values(['book', 'window']).iterrows():
    vals = [f'{r[c]:.3f}' if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in rcols]
    lines.append('| ' + ' | '.join(vals) + ' |')
lines.append('')
lines.append('## Pass bar (identical to 1700): alpha>=+8%/yr AND Sharpe>=1.0 AND null pct(return)>=95, both halves; '
              'cost drag<4%/yr; max DD<=20%')
lines.append('')
any_pass = False
for name, cal_df_, nf, is_frac, ppy in BOOKS:
    p = passes_bar(name)
    any_pass = any_pass or p
    whole = reads_df[(reads_df.book == name) & (reads_df.window == 'whole')].iloc[0]
    if p:
        rec = 'PASS -> independent rebuild from prose, then paper sleeve at $20K notional'
    elif pd.notna(whole.ann_alpha_spy) and whole.ann_alpha_spy > 0:
        rec = 'fail -> positive whole-window point estimate, phase 2 (free Alpaca history) worth running'
    else:
        rec = 'fail -> non-positive whole-window point estimate, no phase-2 case on this read'
    lines.append(f'- {name}: {"PASS" if p else "fail"}; whole-window alpha {whole.ann_alpha_spy:.3f}, Sharpe '
                  f'{whole.sharpe_net:.2f}, null-pct(ret) {whole.null_pct_return:.0f}, median pool '
                  f'{whole.median_pool_n:.0f}, median N {whole.median_port_n:.0f} -> {rec}')
lines.append('')
lines.append('## MDE and capital note')
lines.append('')
lines.append('MDE for a 65-week series is ~2 Sharpe units of noise (PREREG figure); the monthly book has only '
              f'~{len(months)} periods and is far thinner still -- read it as directional, not conclusive. See '
              'null_*_std columns in 1700b_reads.csv for the empirical per-cell null spread.')
lines.append(f'Capital note: the sleeve holds ${CAPITAL:,.0f} overnight all week/month. Same equity base as '
              'ORB/HOD overnight exposure on this account -- shared tail on a gap-down morning.')
lines.append('')
lines.append(f'Overall phase-1 verdict: {"at least one book PASSES the pre-registered bar" if any_pass else "no book passes the pre-registered bar on both halves"}.')
(OUT / 'RESULT_1700b.md').write_text('\n'.join(lines) + '\n')
log.info('%s wrote RESULT_1700b.md (%d lines)', elapsed(), len(lines))
log.info('%s TOTAL RUNTIME %.1fs -- any_pass=%s', elapsed(), time.time() - t0, any_pass)
