#!/usr/bin/env python3
"""Cell 1,700 — weekly-rebalanced 12-month momentum sleeve, PHASE 1 ONLY.

PREREG: research/momentum_weekly/PREREG_1700.md (FROZEN 2026-10-01).
Owner ask: "every Monday buy the stocks that did the best P&L over the past
year; every Monday rebalance." This script IS that request, phase 1 (data on
disk, no fetches of any kind).

Price source (primary): Databento EQUS.SUMMARY point-in-time daily OHLCV,
data/research/databento/equs_daily_2024H2.parquet (2024-07..2024-12) +
equs_daily_2025_2026.parquet (2025-01..2026-09) — delisted names included,
this panel IS the survivorship fix (a delisted symbol stops appearing, it is
never silently dropped from a "current" list).

Cross-checks (not the primary source): research/overnight_high/panel_2024_2026
.parquet (adv20 already computed there) and data/cache.db daily_bars
(opened read-only, ?mode=ro, 30s backoff on lock).

Exclusions: NASDAQ test tickers (^Z[A-Z]ZZT$) and wrapper-tagged symbols from
data/research/orb_asset_class_map_20260711.csv (asset_class == 'wrapper').
NOTE logged at runtime: that CSV's 'wrapper' tag is coarse — it flags plain
index ETFs (SPY, IWM) as 'wrapper' too, not only levered/inverse names
(TQQQ, SQQQ); QQQ is tagged 'stock' in the same file. Per the task's explicit
instruction this CSV is the designated exclusion mechanism, applied exactly
as given and BEFORE any numbers were looked at. SPY itself is pulled from the
panel before this exclusion so it remains available as the benchmark series.

Momentum signal, measured at the Friday close preceding each Monday
rebalance, lookbacks counted in TRADING-DAY rows within each symbol's own
series (not calendar days):
    M1 = 12-0   close[t]      / close[t-252] - 1
    M2 = 12-1   close[t-21]   / close[t-252] - 1   (skip the last month)
    M3 = 6-1    close[t-21]   / close[t-126] - 1

Portfolio: top N in {10,20,50} by signal, equal weight, bought at Monday's
open, held one week, rebalanced the next Monday's open (names that stay in
the portfolio pay no transaction cost). Cost = 5bps per side + half the
spread proxy (daily (high-low)/close * 0.1, capped at 20bps) on every TRADED
dollar.

Null: 1,000 random top-N draws from the SAME eligible, history-sufficient
pool each Monday (count-matched on N and on the variant's history
requirement), gross of cost — isolates "ranking skill" from "any similarly
sized basket of this universe".
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
    filename=str(OUT / '1700_momentum.log'), filemode='w', level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700')
log.addHandler(logging.StreamHandler(sys.stdout))

SEED = 1700
N_DRAWS = 1000
VARIANTS = ['M1', 'M2', 'M3']
NS = [10, 20, 50]
CAPITAL = 65_000.0
WIN_START = pd.Timestamp('2025-07-01')
WIN_END = pd.Timestamp('2026-09-30')
HALF_CUT = pd.Timestamp('2026-01-01')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')

t0 = time.time()


def elapsed() -> str:
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
    log.warning('%d / %d rows have a non-positive OHLC price (no-trade rows on thin warrants/units/rights) — '
                'dropped, they would otherwise divide-by-zero or fabricate a -100%% return', int(bad_px.sum()), len(panel_raw))
    panel_raw = panel_raw[~bad_px].reset_index(drop=True)

spy = panel_raw.loc[panel_raw.symbol == 'SPY', ['bar_date', 'open', 'close']].sort_values('bar_date').reset_index(drop=True)
if spy.empty:
    log.error('SPY not found in panel_raw — benchmark series unavailable, aborting')
    raise SystemExit('SPY missing from EQUS panel')
log.info('%s SPY pulled BEFORE exclusions (benchmark only, not a candidate holding): rows=%d %s..%s',
         elapsed(), len(spy), spy.bar_date.min(), spy.bar_date.max())

# ============================================================ exclusions ==
all_syms_raw = set(s for s in panel_raw['symbol'].unique() if s is not None)
n_none_sym = panel_raw['symbol'].isna().sum()
if n_none_sym:
    log.warning('%d rows have a null symbol in the raw panel — dropped before any filtering', n_none_sym)
    panel_raw = panel_raw[panel_raw['symbol'].notna()].reset_index(drop=True)
n_test_sym = sum(1 for s in all_syms_raw if TEST_RE.match(s))
ac = pd.read_csv(ROOT / 'data/research/orb_asset_class_map_20260711.csv')
wrapper_syms = set(ac.loc[ac.asset_class == 'wrapper', 'symbol'])
n_wrapper_sym = len(all_syms_raw & wrapper_syms)
log.info('%s exclusion counts (against the full %d-symbol raw panel): %d test-ticker symbols (regex %s), '
         '%d wrapper-tagged symbols (orb_asset_class_map_20260711.csv) — NOTE: that map also tags plain index '
         'ETFs (SPY, IWM) as "wrapper", not only levered/inverse names; QQQ is tagged "stock" in the same file. '
         'Applied exactly as instructed, pre-committed before any result was read.',
         elapsed(), len(all_syms_raw), n_test_sym, TEST_RE.pattern, n_wrapper_sym)

# ---------------------------------------------------- loose liquidity prefilter (speed only, cannot bias results) --
panel_raw['dvol'] = panel_raw['close'] * panel_raw['volume']
sym_stats = panel_raw.groupby('symbol').agg(max_close=('close', 'max'), max_dvol=('dvol', 'max'))
keep_syms = set(sym_stats[(sym_stats.max_close >= 5) & (sym_stats.max_dvol >= 1_000_000)].index)
log.info('%s loose liquidity prefilter: %d / %d symbols can ever clear $5 price & $1M single-day $vol '
         '(a necessary, not sufficient, condition for the real $5M ADV20 gate below) — dropping the rest before '
         'the expensive per-symbol feature computation', elapsed(), len(keep_syms), len(all_syms_raw))

panel = panel_raw[panel_raw.symbol.isin(keep_syms) & ~panel_raw.symbol.isin(wrapper_syms)
                   & ~panel_raw.symbol.map(lambda s: bool(TEST_RE.match(s)))].copy()
log.info('%s tradable universe after liquidity prefilter + exclusions: %d symbols, %d rows',
         elapsed(), panel.symbol.nunique(), len(panel))

# =============================================================== features ==
panel = panel.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
g = panel.groupby('symbol', sort=False)
log.info('%s computing adv20 (20-trading-day rolling $ volume, per symbol, min_periods=20)...', elapsed())
panel['adv20'] = g['dvol'].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
panel['spread_proxy'] = (((panel['high'] - panel['low']) / panel['close']).clip(lower=0) * 0.1).clip(upper=0.002)
log.info('%s computing momentum signals M1/M2/M3 (row-shift = trading-day lookback, per symbol)...', elapsed())
panel['close_lag21'] = g['close'].shift(21)
panel['close_lag126'] = g['close'].shift(126)
panel['close_lag252'] = g['close'].shift(252)
panel['sig_M1'] = panel['close'] / panel['close_lag252'] - 1
panel['sig_M2'] = panel['close_lag21'] / panel['close_lag252'] - 1
panel['sig_M3'] = panel['close_lag21'] / panel['close_lag126'] - 1
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
log.info('%s phase-1 weekly rebalances: %d, entry %s..%s (PREREG states ~65 Mondays)',
         elapsed(), len(weeks), weeks.entry_date.min().date(), weeks.entry_date.max().date())

monday_dates = pd.Index(sorted(set(weeks.entry_date) | set(weeks.next_entry_date)))

# ======================================================= open/cost pivots ==
log.info('%s pivoting open price and cost-rate on %d Monday-equivalent dates...', elapsed(), len(monday_dates))
mon_rows = panel[panel.bar_date.isin(monday_dates)]
open_piv = mon_rows.pivot_table(index='bar_date', columns='symbol', values='open', aggfunc='last').reindex(monday_dates)
spread_piv = mon_rows.pivot_table(index='bar_date', columns='symbol', values='spread_proxy', aggfunc='last').reindex(monday_dates)
weekly_ret_wide = open_piv.shift(-1) / open_piv - 1
cost_rate_piv = 0.0005 + 0.5 * spread_piv.fillna(0.002)
log.info('%s pivots built: shape %s', elapsed(), open_piv.shape)

spy_idx = spy.set_index('bar_date').reindex(monday_dates)
spy_ret = spy_idx['open'].shift(-1) / spy_idx['open'] - 1

# ==================================================== universe benchmark ==
elig_mask_base = (panel['close'] >= 5) & (panel['adv20'] >= 5_000_000)
bench_rets, bench_poolsz = {}, {}
for _, row in weeks.iterrows():
    d, ed = row['prior_signal_date'], row['entry_date']
    pool = panel.loc[(panel.bar_date == d) & elig_mask_base, 'symbol']
    bench_poolsz[ed] = len(pool)
    if len(pool) == 0:
        bench_rets[ed] = np.nan
        continue
    bench_rets[ed] = weekly_ret_wide.loc[ed].reindex(pool.values).mean(skipna=True)
bench_ser = pd.Series(bench_rets).reindex(weeks.entry_date.values)
log.info('%s equal-weight universe benchmark built, %d weeks, median eligible pool size %d',
         elapsed(), len(bench_ser), int(np.median(list(bench_poolsz.values()))))


# ============================================================= simulation ==
def run_real(variant: str, N: int) -> pd.DataFrame:
    """Momentum-ranked weekly simulation for one (variant, N). Rows in weeks order."""
    sig_col = f'sig_{variant}'
    prev_port: set[str] = set()
    rows = []
    for _, wrow in weeks.iterrows():
        sdate, edate = wrow['prior_signal_date'], wrow['entry_date']
        day = panel.loc[(panel.bar_date == sdate) & (panel['close'] >= 5) & (panel['adv20'] >= 5_000_000)
                         & panel[sig_col].notna(), ['symbol', sig_col, 'adv20']]
        if len(day) < N:
            log.warning('variant=%s N=%d week=%s: eligible pool %d < N — taking all available',
                        variant, N, edate.date(), len(day))
        port = set(day.nlargest(N, sig_col)['symbol']) if len(day) else set()
        bought, sold = port - prev_port, prev_port - port
        if port:
            fwd = weekly_ret_wide.loc[edate].reindex(list(port))
            n_missing = int(fwd.isna().sum())
            gross = fwd.fillna(0).mean()
        else:
            n_missing, gross = 0, np.nan
        cost = 0.0
        if bought or sold:
            cr = cost_rate_piv.loc[edate]
            cost = (cr.reindex(list(bought)).fillna(0.002).sum() + cr.reindex(list(sold)).fillna(0.002).sum()) / N
        turnover = len(bought) / N
        adv_ratio = np.nan
        if port:
            advs = day.set_index('symbol').reindex(list(port))['adv20']
            adv_ratio = ((CAPITAL / N) / advs).mean()
        rows.append(dict(entry_date=edate, n_eligible=len(day), port_n=len(port), gross_ret=gross,
                          cost=cost, net_ret=(gross - cost if pd.notna(gross) else np.nan),
                          turnover=turnover, n_fwd_missing=n_missing, adv_impact=adv_ratio))
        prev_port = port
    return pd.DataFrame(rows).set_index('entry_date')


def run_null(variant: str, N: int, rng: np.random.Generator) -> np.ndarray:
    """1,000 random count-matched draws from the SAME eligible pool each week -> (1000, n_weeks) gross returns."""
    sig_col = f'sig_{variant}'
    cols_out = []
    for _, wrow in weeks.iterrows():
        sdate, edate = wrow['prior_signal_date'], wrow['entry_date']
        pool = panel.loc[(panel.bar_date == sdate) & (panel['close'] >= 5) & (panel['adv20'] >= 5_000_000)
                          & panel[sig_col].notna(), 'symbol'].values
        K = len(pool)
        if K == 0:
            cols_out.append(np.full(N_DRAWS, np.nan))
            continue
        n = min(N, K)
        rets = weekly_ret_wide.loc[edate].reindex(pool).fillna(0).values
        rand = rng.random((N_DRAWS, K))
        idx = np.argpartition(rand, n - 1, axis=1)[:, :n]
        cols_out.append(rets[idx].mean(axis=1))
    return np.column_stack(cols_out)


def ann_stats(weekly_returns) -> dict:
    r = np.asarray(weekly_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return dict(n_weeks=0, ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan, max_dd=np.nan,
                    worst_week=np.nan, best_week=np.nan, green_share=np.nan)
    comp = np.prod(1 + r)
    ann_return = comp ** (52.0 / n) - 1
    sd = np.std(r, ddof=1) if n > 1 else np.nan
    ann_vol = sd * np.sqrt(52) if n > 1 else np.nan
    sharpe = (np.mean(r) / sd * np.sqrt(52)) if (n > 1 and sd and sd > 0) else np.nan
    curve = np.cumprod(1 + r)
    peak = np.maximum.accumulate(curve)
    max_dd = ((curve - peak) / peak).min()
    return dict(n_weeks=n, ann_return=ann_return, ann_vol=ann_vol, sharpe=sharpe, max_dd=max_dd,
                worst_week=r.min(), best_week=r.max(), green_share=float((r > 0).mean()))


log.info('%s STEP 2: simulating %d variants x %d N = %d (variant,N) cells, real + %d-draw null each',
          elapsed(), len(VARIANTS), len(NS), len(VARIANTS) * len(NS), N_DRAWS)
rng = np.random.default_rng(SEED)
all_real, all_null = {}, {}
for variant in VARIANTS:
    for N in NS:
        all_real[(variant, N)] = run_real(variant, N)
        all_null[(variant, N)] = run_null(variant, N, rng)
        log.info('%s  done variant=%s N=%d', elapsed(), variant, N)

WINDOWS = {
    'halfA': (WIN_START, HALF_CUT - pd.Timedelta(days=1)),
    'halfB': (HALF_CUT, WIN_END),
    'whole': (WIN_START, WIN_END),
}
entry_dates = weeks.entry_date.values

reads, weekly_rows = [], []
for variant in VARIANTS:
    for N in NS:
        real_df = all_real[(variant, N)]
        null_mat = all_null[(variant, N)]
        for edate, r in real_df.iterrows():
            weekly_rows.append(dict(variant=variant, N=N, entry_date=edate, **r.to_dict()))
        for win_name, (ws, we) in WINDOWS.items():
            mask = (entry_dates >= ws) & (entry_dates <= we)
            sub = real_df.loc[mask]
            net_stats = ann_stats(sub['net_ret'].values)
            gross_stats = ann_stats(sub['gross_ret'].values)
            spy_sub = spy_ret.reindex(sub.index).values
            valid = ~np.isnan(sub['net_ret'].values) & ~np.isnan(spy_sub)
            if valid.sum() > 2 and np.var(spy_sub[valid]) > 0:
                beta = np.cov(sub['net_ret'].values[valid], spy_sub[valid])[0, 1] / np.var(spy_sub[valid], ddof=1)
            else:
                beta = np.nan
            spy_ann = ann_stats(spy_sub)['ann_return']
            if valid.sum() and pd.notna(beta):
                ann_alpha_spy = np.mean(sub['net_ret'].values[valid] - beta * spy_sub[valid]) * 52
            else:
                ann_alpha_spy = np.nan
            bench_sub = bench_ser.reindex(sub.index).values
            diff = sub['net_ret'].values - bench_sub
            ann_alpha_universe = np.nanmean(diff) * 52 if np.isfinite(diff).any() else np.nan
            cost_drag_annual = np.nanmean(sub['cost'].values) * 52
            turnover_avg = np.nanmean(sub['turnover'].values)
            null_sub = null_mat[:, mask]
            null_ann_ret = np.array([ann_stats(null_sub[i])['ann_return'] for i in range(N_DRAWS)])
            null_sharpe = np.array([ann_stats(null_sub[i])['sharpe'] for i in range(N_DRAWS)])
            pct_ret = 100 * np.nanmean(null_ann_ret <= net_stats['ann_return']) if pd.notna(net_stats['ann_return']) else np.nan
            pct_sharpe = 100 * np.nanmean(null_sharpe <= net_stats['sharpe']) if pd.notna(net_stats['sharpe']) else np.nan
            reads.append(dict(
                variant=variant, N=N, window=win_name, n_weeks=net_stats['n_weeks'],
                ann_return_net=net_stats['ann_return'], ann_return_gross=gross_stats['ann_return'],
                ann_vol=net_stats['ann_vol'], sharpe_net=net_stats['sharpe'], max_dd=net_stats['max_dd'],
                worst_week=net_stats['worst_week'], best_week=net_stats['best_week'],
                green_share=net_stats['green_share'], turnover_avg=turnover_avg, cost_drag_annual=cost_drag_annual,
                beta_spy=beta, ann_alpha_spy=ann_alpha_spy, ann_alpha_universe=ann_alpha_universe,
                spy_ann_return=spy_ann, null_pct_return=pct_ret, null_pct_sharpe=pct_sharpe,
                null_ann_ret_mean=np.nanmean(null_ann_ret), null_ann_ret_std=np.nanstd(null_ann_ret),
                null_sharpe_mean=np.nanmean(null_sharpe), null_sharpe_std=np.nanstd(null_sharpe),
                median_position_usd=CAPITAL / N, avg_adv_impact=np.nanmean(sub['adv_impact'].values),
                share_impact_gt1pct=float(np.nanmean((sub['adv_impact'].values > 0.01).astype(float))),
                n_fwd_missing=int(sub['n_fwd_missing'].sum())))

reads_df = pd.DataFrame(reads)
weekly_df = pd.DataFrame(weekly_rows)
reads_df.to_csv(OUT / '1700_reads.csv', index=False)
weekly_df.to_csv(OUT / '1700_weekly.csv', index=False)
log.info('%s wrote 1700_reads.csv (%d rows, expect 27) and 1700_weekly.csv (%d rows)',
          elapsed(), len(reads_df), len(weekly_df))


def passes_bar(variant: str, N: int) -> bool:
    a = reads_df[(reads_df.variant == variant) & (reads_df.N == N) & (reads_df.window == 'halfA')].iloc[0]
    b = reads_df[(reads_df.variant == variant) & (reads_df.N == N) & (reads_df.window == 'halfB')].iloc[0]
    def cond(row):
        return (pd.notna(row.ann_alpha_spy) and row.ann_alpha_spy >= 0.08
                and pd.notna(row.sharpe_net) and row.sharpe_net >= 1.0
                and pd.notna(row.null_pct_return) and row.null_pct_return >= 95
                and row.cost_drag_annual < 0.04 and abs(row.max_dd) <= 0.20)
    return cond(a) and cond(b)


# ================================================================ cross-checks ==
log.info('%s STEP 3: cross-check 1 — cache.db daily_bars (read-only) vs EQUS close, 300-row sample', elapsed())
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
    log.error('cache.db unreachable after 3 attempts — cross-check 1 SKIPPED (not fatal, EQUS is the primary source)')
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

log.info('%s cross-check 2 — overnight_high panel dvol20 ($ volume) vs our adv20 ($ volume), full join', elapsed())
# NOTE: the overnight_high panel's own 'adv20' column is 20-day AVERAGE SHARE volume (confirmed: adv20 ~=
# rolling mean of its 'volume' column); its DOLLAR-volume column is 'dvol20'. Our panel's 'adv20' column is
# dollar volume (close*volume rolled 20d) per the PREREG's "$ volume" gate, so the correct cross-check is
# our adv20 vs THEIR dvol20, not their same-named-but-different-unit 'adv20'.
ov = pd.read_parquet(ROOT / 'research/overnight_high/panel_2024_2026.parquet', columns=['symbol', 'bar_date', 'dvol20'])
ov['bar_date'] = pd.to_datetime(ov['bar_date'])
merged = panel[['symbol', 'bar_date', 'adv20']].merge(ov, on=['symbol', 'bar_date'], how='inner')
merged = merged.dropna(subset=['adv20', 'dvol20'])
merged = merged[merged.dvol20 > 0]
if len(merged):
    reldiff = (merged.adv20 - merged.dvol20).abs() / merged.dvol20
    pct_close = float((reldiff < 0.10).mean() * 100)
    log.info('%s overnight-panel cross-check: %d matched (symbol,date) rows, %.1f%% within 10%% $-volume agreement, median reldiff %.3f',
              elapsed(), len(merged), pct_close, reldiff.median())
else:
    pct_close = float('nan')
    log.warning('overnight-panel cross-check: no matched rows (different symbol universes)')

# =================================================================== RESULT.md ==
log.info('%s STEP 4: writing RESULT_1700.md', elapsed())
lines = []
lines.append('# RESULT — cell 1,700: weekly-rebalanced 12-month momentum sleeve, phase 1')
lines.append('')
lines.append(f'PREREG_1700.md (FROZEN). Owner ask: "every Monday buy the stocks that did the best P&L over the '
              f'past year; every Monday rebalance." Phase 1 only — data on disk, zero fetches.')
lines.append(f'Window {WIN_START.date()}..{WIN_END.date()}, {len(weeks)} weekly rebalances '
              f'(halfA {WINDOWS["halfA"][0].date()}..{WINDOWS["halfA"][1].date()}, '
              f'halfB {WINDOWS["halfB"][0].date()}..{WINDOWS["halfB"][1].date()}).')
lines.append(f'Exclusions (full raw panel, {len(all_syms_raw)} symbols): {n_test_sym} test-ticker symbols '
              f'(`^Z[A-Z]ZZT$`), {n_wrapper_sym} wrapper-tagged symbols (orb_asset_class_map_20260711.csv). '
              f'That map\'s "wrapper" tag also strips plain index ETFs (SPY, IWM) as holdings — SPY kept only as '
              f'the benchmark, pulled before exclusion; QQQ is tagged "stock" in the same file and stays tradable.')
lines.append(f'Cross-checks: cache.db daily_bars (read-only) {match}/{checked} sampled closes within 0.5%; '
              f'overnight_high panel $-volume (dvol20) agreement {pct_close:.1f}% of {len(merged)} matched rows within 10%.')
lines.append('')
lines.append('## Reads (27 = 3 variants x 3 N x 3 windows; annualised net-of-cost unless marked)')
lines.append('')
rcols = ['variant', 'N', 'window', 'n_weeks', 'ann_return_net', 'ann_alpha_spy', 'sharpe_net', 'max_dd',
         'turnover_avg', 'cost_drag_annual', 'null_pct_return', 'null_pct_sharpe']
lines.append('| ' + ' | '.join(rcols) + ' |')
lines.append('|' + '---|' * len(rcols))
for _, r in reads_df.sort_values(['variant', 'N', 'window']).iterrows():
    vals = [f'{r[c]:.3f}' if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in rcols]
    lines.append('| ' + ' | '.join(vals) + ' |')
lines.append('')
lines.append('## Pass bar (phase 1): alpha>=+8%/yr AND Sharpe>=1.0 AND null pct(return)>=95, both halves; '
              'cost drag<4%/yr; max DD<=20%')
lines.append('')
any_pass = False
for variant in VARIANTS:
    for N in NS:
        p = passes_bar(variant, N)
        any_pass = any_pass or p
        whole = reads_df[(reads_df.variant == variant) & (reads_df.N == N) & (reads_df.window == 'whole')].iloc[0]
        if p:
            rec = 'PASS -> independent rebuild from prose, then paper sleeve at $20K notional'
        elif pd.notna(whole.ann_alpha_spy) and whole.ann_alpha_spy > 0:
            rec = 'fail -> positive whole-window point estimate, phase 2 (free Alpaca history) worth running'
        else:
            rec = 'fail -> non-positive whole-window point estimate, no phase-2 case on this read'
        lines.append(f'- {variant} N={N}: {"PASS" if p else "fail"}; whole-window alpha {whole.ann_alpha_spy:.3f}, '
                      f'Sharpe {whole.sharpe_net:.2f}, null-pct(ret) {whole.null_pct_return:.0f} -> {rec}')
lines.append('')
lines.append('## MDE and capital note')
lines.append('')
lines.append('MDE for a 65-week series is ~2 Sharpe units of noise (PREREG figure) — thin by construction; see '
              'null_*_std columns in 1700_reads.csv for the empirical per-cell null spread.')
lines.append(f'Capital note: the sleeve holds ${CAPITAL:,.0f} overnight all week (median position '
              f'${CAPITAL/50:,.0f}-${CAPITAL/10:,.0f} across N=50..10). Same equity base as ORB/HOD overnight '
              f'exposure on this account — a market-wide gap-down morning hits all three simultaneously (shared '
              f'tail, not diversified away by running multiple books).')
lines.append('')
lines.append(f'Overall phase-1 verdict: {"at least one cell PASSES the pre-registered bar" if any_pass else "no cell passes the pre-registered bar on both halves"}.')
(OUT / 'RESULT_1700.md').write_text('\n'.join(lines) + '\n')
log.info('%s wrote RESULT_1700.md (%d lines)', elapsed(), len(lines))
log.info('%s TOTAL RUNTIME %.1fs — any_pass=%s', elapsed(), time.time() - t0, any_pass)
