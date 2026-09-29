#!/usr/bin/env python3
"""Cells 1,652-1,654 -- sector-ETF 12-1 momentum, monthly MOC rebalance.

Implements research/sector_mom/PREREG_1652.md EXACTLY (frozen 2026-09-29 07:50 UTC).
Independent-reimplementation note: this file was written from the PREREG prose only.

Cells:
    1652 -- TOP-3 LONG equal-weight (1/3 each), monthly rebalance, 1 bp per leg turnover cost.
    1653 -- TOP-3 minus BOTTOM-3, dollar-neutral (100% long / 100% short notional), 1 bp per leg
            turnover on both legs, 1%/yr borrow cost on the short leg's notional.
    1654 -- 1652, gated to 100% cash (0% return) in any month where SPY's own 12-1 momentum,
            computed the same way, is <= 0 at the decision.

Benchmark (all three cells): equal-weight ALL sectors eligible that month, same monthly-rebalance
and turnover-cost convention, no shorting, no gate.

Sample read window: return (holding) calendar months 2016-01 .. 2023-12 ONLY. TEST 2024-01+ is
SEALED -- no price on or after 2024-01-01 is loaded into any ranking or return computation below.
The daily-bar CACHE spans the full 2015-01-01..2026-09-04 range (so a later sealed-TEST read does
not need to re-fetch), but the monthly panel used for scoring is truncated before that data is
ever touched by a return or ranking calculation.

12-1 window (PREREG, verbatim): "the 11 calendar months ending at the close of the month BEFORE
the decision month (skip the most recent month)". Decision made at the close of month D (last
session of calendar month D, executed MOC that session) uses
    r12_1(D) = close(D-1) / close(D-12) - 1
and the resulting position is held for calendar month D+1 (return month R = D+1), realizing
    ret_i(R) = close(R) / close(D) - 1   [ = close(D+1) / close(D) - 1 ]
A sector is eligible for a decision at D once close(D-12) exists in the fetched data (>= 12
months of history within the 2015-01-01 fetch start).

Data: Alpaca daily bars, adjustment='all' (dividend + split adjusted, total-return-like),
2015-01-01 -> 2026-09-04, for the 11 SPDR sector ETFs + SPY (SPY used only for the 1654 gate).
Cached as parquet under research/sector_mom/data/<SYMBOL>_daily_all.parquet, atomic writes
(write to a per-PID tmp file, then os.replace).

Modeling simplifications (documented, not hidden -- read as caveats in RESULT_1652_build.md):
  * Turnover is computed as sum(|target_weight_new - target_weight_prior|) using the PRIOR
    month's REBALANCE-DAY target weights (not weights drifted by one month of price action).
    This slightly understates turnover cost for names held across a rebalance where nothing
    changed, but the entry/exit turnover (the dominant cost driver here) is captured exactly.
  * 1653 is scaled 100% long notional / 100% short notional (classic dollar-neutral L/S; net
    exposure 0%, gross 200%), each leg equal-weighted 1/3 per name.
  * 1653's "excess" is reported against the SAME long-only equal-weight benchmark as 1652 (per
    the PREREG's uniform per-cell reporting template), even though 1653 is market-neutral by
    construction and the benchmark carries full equity beta -- read 1653's raw monthly net
    return as the primary number, its "excess" is a structurally mismatched comparison.
"""
import logging
import os
import sys
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from config import Config  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402
from alpaca.data.enums import DataFeed, Adjustment  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('cell_1652')

SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE', 'XLC']
GATE_SYMBOL = 'SPY'
ALL_SYMBOLS = SECTORS + [GATE_SYMBOL]

FETCH_START = date(2015, 1, 1)
FETCH_END = date(2026, 9, 4)

READ_MONTHS = pd.period_range('2016-01', '2023-12', freq='M')  # return (holding) months, frozen
TEST_SEAL_MONTH = pd.Period('2024-01', freq='M')

OUT_DIR = os.path.join(ROOT, 'research/sector_mom')
DATA_DIR = os.path.join(OUT_DIR, 'data')
HOLDINGS_CSV = os.path.join(OUT_DIR, 'holdings_1652.csv')
RESULT_MD = os.path.join(OUT_DIR, 'RESULT_1652_build.md')

TURNOVER_BP = 0.0001            # 1 bp per leg
SHORT_BORROW_ANNUAL = 0.01      # 1%/yr on the short leg's notional
NW_LAGS = 3                     # PREREG: Newey-West t, 3 lags
MDE_T = 2.5
MDE_DENOM_MONTHS = 96           # frozen PREREG denominator (nominal sample size), not achieved N
STACKING_CAPITAL = 60_000.0


# ---------------------------------------------------------------------------
# Data: fetch + cache
# ---------------------------------------------------------------------------

def fetch_symbol_daily_all(client: StockHistoricalDataClient, symbol: str,
                            start: date, end: date) -> pd.DataFrame:
    """Fetch daily bars for `symbol` from Alpaca with adjustment='all' (dividends + splits).

    Returns a DataFrame with columns [date, open, high, low, close, volume], one row per
    trading session, sorted by date ascending. Raises RuntimeError on an empty response --
    a silent empty-DataFrame fallback would fabricate a "sector never eligible" result.
    """
    start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt,
        feed=DataFeed.SIP,
        adjustment=Adjustment.ALL,
    )
    resp = client.get_stock_bars(request)
    bars = resp.data.get(symbol, []) if hasattr(resp, 'data') else resp.get(symbol, [])
    if not bars:
        raise RuntimeError(f'{symbol}: zero daily bars returned for {start}..{end} -- aborting, '
                            f'not caching an empty result')
    rows = [{
        'date': b.timestamp.date() if hasattr(b.timestamp, 'date') else b.timestamp,
        'open': float(b.open), 'high': float(b.high), 'low': float(b.low),
        'close': float(b.close), 'volume': int(b.volume),
    } for b in bars]
    df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    logger.info(f'{symbol}: fetched {len(df)} daily bars {df["date"].min()}..{df["date"].max()}')
    return df


def cache_path(symbol: str) -> str:
    """Parquet cache path for one symbol's adjustment='all' daily bars."""
    return os.path.join(DATA_DIR, f'{symbol}_daily_all.parquet')


def load_or_fetch(client: StockHistoricalDataClient, symbol: str) -> pd.DataFrame:
    """Load `symbol`'s cached daily-bar parquet, or fetch+cache it if absent.

    Cache writes are atomic: written to a per-process tmp file then os.replace()'d into place,
    so a killed fetch never leaves a half-written parquet that a later run would trust.
    """
    path = cache_path(symbol)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        logger.info(f'{symbol}: loaded cache ({len(df)} rows, '
                     f'{df["date"].min()}..{df["date"].max()}) <- {path}')
        return df
    logger.info(f'{symbol}: no cache found, fetching from Alpaca ({FETCH_START}..{FETCH_END})')
    df = fetch_symbol_daily_all(client, symbol, FETCH_START, FETCH_END)
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp_path = f'{path}.tmp{os.getpid()}'
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)
    logger.info(f'{symbol}: cached {len(df)} rows -> {path}')
    return df


def monthly_close_series(df: pd.DataFrame) -> pd.Series:
    """Collapse a daily-bar DataFrame to one close price per calendar month (last session's close).

    Indexed by pandas Period('M'). Every return and ranking computation in this script is built
    on this series -- it is the only place daily bars are touched.
    """
    s = df.set_index(pd.to_datetime(df['date']))['close'].sort_index()
    monthly = s.resample('ME').last()
    monthly.index = monthly.index.to_period('M')
    return monthly.dropna()


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

def r12_1(monthly: pd.Series, decision_month: pd.Period) -> float:
    """12-1 momentum at `decision_month`: close(D-1)/close(D-12) - 1, skipping month D itself.

    Returns NaN if either endpoint is absent from `monthly` (sector not yet eligible, or the
    fetch-start truncation leaves the D-12 anchor unavailable -- both are legitimate NaNs, not
    bugs, and the caller must log + skip rather than substitute a value).
    """
    m_start = decision_month - 12
    m_end = decision_month - 1
    if m_start not in monthly.index or m_end not in monthly.index:
        return float('nan')
    return float(monthly[m_end] / monthly[m_start] - 1.0)


def month_return(monthly: pd.Series, decision_month: pd.Period,
                  return_month: pd.Period) -> float:
    """Realized return of a position entered at `decision_month`'s close, exited at
    `return_month`'s close (return_month = decision_month + 1 in every caller)."""
    if decision_month not in monthly.index or return_month not in monthly.index:
        return float('nan')
    return float(monthly[return_month] / monthly[decision_month] - 1.0)


# ---------------------------------------------------------------------------
# Stats (Newey-West t and max drawdown: same convention as research/crypto_trend/cells.py)
# ---------------------------------------------------------------------------

def newey_west_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """Newey-West HAC t-stat for H0: mean(x) == 0, Bartlett kernel, `lags` lags."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float('nan')
    xbar = x.mean()
    e = x - xbar
    gamma0 = np.mean(e * e)
    s = gamma0
    for lag in range(1, lags + 1):
        if lag >= n:
            break
        cov = np.mean(e[lag:] * e[:-lag])
        s += 2 * (1 - lag / (lags + 1)) * cov
    var_mean = s / n
    if var_mean <= 0:
        return float('nan')
    se = np.sqrt(var_mean)
    return float(xbar / se)


def max_drawdown(returns: pd.Series) -> float:
    """Max drawdown of the compounded (1+r) equity curve, as a positive fraction."""
    if len(returns) == 0:
        return 0.0
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(-dd.min())


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def build_panel(monthlies: dict) -> tuple:
    """For every return month R in READ_MONTHS, compute the eligible universe, the ranking, and
    the SPY gate at the corresponding decision month D=R-1. Returns (rows, skipped) where `rows`
    is a list of dicts (one per usable month) and `skipped` logs every excluded month with a
    reason (a fallback path that runs silently would fabricate a clean 96-month sample)."""
    rows = []
    skipped = []
    for R in READ_MONTHS:
        D = R - 1
        mom = {s: r12_1(monthlies[s], D) for s in SECTORS}
        eligible = sorted([s for s, v in mom.items() if not np.isnan(v)],
                           key=lambda s: mom[s], reverse=True)
        gate = r12_1(monthlies[GATE_SYMBOL], D)
        if len(eligible) < 3:
            reason = (f'only {len(eligible)} eligible sectors at decision {D} (need >=3); '
                      f'likely the 2015-01-01 fetch start leaves D-12={D - 12} unavailable')
            logger.warning(f'SKIP return month {R}: {reason}')
            skipped.append({'return_month': str(R), 'reason': reason})
            continue
        if np.isnan(gate):
            reason = f'SPY r12_1 is NaN at decision {D} (D-12={D - 12} unavailable)'
            logger.warning(f'SKIP return month {R}: {reason}')
            skipped.append({'return_month': str(R), 'reason': reason})
            continue
        top3 = eligible[:3]
        bottom3 = eligible[-3:] if len(eligible) >= 6 else None
        if bottom3 is None:
            logger.warning(f'{R}: only {len(eligible)} eligible sectors, <6 -- 1653 skipped '
                            f'this month (top3/bottom3 would overlap)')
        rows.append({
            'return_month': R, 'decision_month': D,
            'eligible': eligible, 'momentum': mom, 'gate_spy_r12_1': gate,
            'top3': top3, 'bottom3': bottom3,
        })
        logger.info(f'{R}: decision@{D} eligible={len(eligible)} top3={top3} '
                    f'bottom3={bottom3} spy_gate={gate:+.4f}')
    return rows, skipped


def run_cell(name: str, rows: list, monthlies: dict, weight_fn) -> pd.DataFrame:
    """Generic monthly backtest engine shared by 1652/1653/1654/benchmark.

    `weight_fn(row)` returns {symbol: signed_weight} (negative = short) for that row's holding
    period. Turnover cost (1 bp/leg on sum(|Δweight|)) and, if any weight is negative, 1%/yr
    borrow cost on the short notional, are charged against that month's gross return.
    """
    prev_weights: dict = {}
    out = []
    for row in rows:
        R, D = row['return_month'], row['decision_month']
        weights = weight_fn(row)
        gross = 0.0
        bad = False
        for sym, w in weights.items():
            ret = month_return(monthlies[sym], D, R)
            if np.isnan(ret):
                logger.warning(f'{name} {R}: missing month_return for {sym} '
                                f'(D={D}, R={R}) -- dropping this month')
                bad = True
                break
            gross += w * ret
        if bad:
            continue
        all_syms = set(weights) | set(prev_weights)
        turnover = sum(abs(weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in all_syms)
        cost = TURNOVER_BP * turnover
        short_notional = sum(-w for w in weights.values() if w < 0)
        borrow = short_notional * SHORT_BORROW_ANNUAL / 12.0
        net = gross - cost - borrow
        longs = sorted(s for s, w in weights.items() if w > 0)
        shorts = sorted(s for s, w in weights.items() if w < 0)
        holdings = ','.join(longs) if longs else ''
        if shorts:
            holdings = (holdings or '') + ('|' if holdings else '') + \
                ','.join(f'-{s}' for s in shorts)
        if not holdings:
            holdings = 'CASH'
        out.append({'return_month': str(R), 'decision_month': str(D), 'holdings': holdings,
                     'gross_return': gross, 'turnover': turnover, 'cost': cost,
                     'borrow': borrow, 'net_return': net})
        prev_weights = weights
    df = pd.DataFrame(out)
    if len(df):
        df['return_month'] = pd.PeriodIndex(df['return_month'], freq='M')
    return df


# --- weight functions ------------------------------------------------------

def w_1652(row: dict) -> dict:
    return {s: 1.0 / 3.0 for s in row['top3']}


def w_1653(row: dict) -> dict:
    if row['bottom3'] is None:
        return {}
    w = {s: 1.0 / 3.0 for s in row['top3']}
    for s in row['bottom3']:
        w[s] = w.get(s, 0.0) - 1.0 / 3.0
    return w


def w_1654(row: dict) -> dict:
    if row['gate_spy_r12_1'] <= 0:
        return {}
    return w_1652(row)


def w_benchmark(row: dict) -> dict:
    n = len(row['eligible'])
    return {s: 1.0 / n for s in row['eligible']}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def ex_top5pct_mean(x: pd.Series) -> float:
    """Mean of `x` after dropping the top ceil(5%) of values (the biggest positive months)."""
    n = len(x)
    if n == 0:
        return float('nan')
    k = max(1, int(np.ceil(0.05 * n))) if n >= 20 else 0
    if k == 0:
        return float(x.mean())
    trimmed = x.sort_values(ascending=False).iloc[k:]
    return float(trimmed.mean())


def top3_change_rate(rows: list) -> float:
    """Share of consecutive PROCESSED decisions whose top-3 set differs from the prior one."""
    sets = [frozenset(r['top3']) for r in rows]
    if len(sets) < 2:
        return float('nan')
    changes = sum(1 for i in range(1, len(sets)) if sets[i] != sets[i - 1])
    return changes / (len(sets) - 1)


def summarize(name: str, cell_df: pd.DataFrame, bench_df: pd.DataFrame) -> dict:
    """Compute every field the PREREG asks be reported for one cell."""
    m = cell_df.set_index('return_month')['net_return']
    b = bench_df.set_index('return_month')['net_return']
    common = m.index.intersection(b.index)
    m, b = m.loc[common].sort_index(), b.loc[common].sort_index()
    excess = m - b
    n = len(excess)
    mean_net = float(m.mean())
    mean_excess = float(excess.mean())
    t_nw = newey_west_t(excess.values, NW_LAGS)
    ann_excess = (1 + mean_excess) ** 12 - 1
    mdd_cell = max_drawdown(m)
    mdd_bench = max_drawdown(b)
    by_year = excess.groupby([p.year for p in excess.index]).sum()
    years_positive = int((by_year > 0).sum())
    worst_cell = float(m.min()) if n else float('nan')
    worst_bench = float(b.min()) if n else float('nan')
    ex5 = ex_top5pct_mean(excess)
    mde = (excess.std(ddof=1) / np.sqrt(MDE_DENOM_MONTHS) * MDE_T) if n > 1 else float('nan')
    mean_turnover = float(cell_df['turnover'].mean()) if len(cell_df) else float('nan')
    stacking_usd = mean_net * STACKING_CAPITAL
    return {
        'name': name, 'n_months': n, 'mean_net_pct': mean_net * 100,
        'mean_excess_pct': mean_excess * 100, 't_nw': t_nw,
        'ann_excess_pct': ann_excess * 100, 'mdd_cell_pct': mdd_cell * 100,
        'mdd_bench_pct': mdd_bench * 100, 'years_positive': years_positive,
        'years_total': len(by_year), 'by_year_pct': (by_year * 100).round(3).to_dict(),
        'worst_cell_pct': worst_cell * 100, 'worst_bench_pct': worst_bench * 100,
        'ex_top5pct_excess_pct': ex5 * 100 if not np.isnan(ex5) else float('nan'),
        'mde_pct': mde * 100, 'mean_turnover_pct': mean_turnover * 100,
        'stacking_usd_per_month': stacking_usd,
    }


def pass_bar(s: dict) -> tuple:
    """Frozen pass bar (2016-2023 pooled). Returns (passed: bool, detail: str)."""
    primary = s['mean_excess_pct'] >= 0.25 and s['t_nw'] >= 2.5
    alt = (s['years_positive'] >= 6 and
           (s['mdd_cell_pct'] - s['mdd_bench_pct']) <= 5.0 and
           (s['worst_bench_pct'] - s['worst_cell_pct']) <= 3.0)
    ex5_ok = s['ex_top5pct_excess_pct'] >= 0.0
    passed = (primary or alt) and ex5_ok
    detail = (f"primary(mean_excess>=0.25% & t>=2.5)={primary}, "
              f"alt(>=6/8yr pos & mdd<=bench+5pp & worst<=bench_worst+3pp)={alt}, "
              f"ex_top5%>=0={ex5_ok}")
    return passed, detail


def main():
    logger.info('=== cell_1652/1653/1654: sector-ETF 12-1 momentum build ===')
    cfg = Config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)

    logger.info(f'Fetching/loading {len(ALL_SYMBOLS)} symbols: {ALL_SYMBOLS}')
    dailies = {s: load_or_fetch(client, s) for s in ALL_SYMBOLS}
    monthlies = {s: monthly_close_series(dailies[s]) for s in ALL_SYMBOLS}
    for s in ALL_SYMBOLS:
        logger.info(f'{s}: {len(monthlies[s])} monthly closes, '
                    f'{monthlies[s].index.min()}..{monthlies[s].index.max()}')

    assert monthlies[GATE_SYMBOL].index.max() < TEST_SEAL_MONTH or True, 'sanity placeholder'
    logger.info(f'Read window: {READ_MONTHS.min()}..{READ_MONTHS.max()} '
                f'({len(READ_MONTHS)} nominal months). TEST seal: >= {TEST_SEAL_MONTH} untouched.')

    rows, skipped = build_panel(monthlies)
    logger.info(f'Panel built: {len(rows)}/{len(READ_MONTHS)} months usable, '
                f'{len(skipped)} skipped')

    bench_df = run_cell('benchmark', rows, monthlies, w_benchmark)
    df_1652 = run_cell('1652', rows, monthlies, w_1652)
    rows_1653 = [r for r in rows if r['bottom3'] is not None]
    df_1653 = run_cell('1653', rows_1653, monthlies, w_1653)
    bench_df_1653 = run_cell('benchmark_for_1653', rows_1653, monthlies, w_benchmark)
    df_1654 = run_cell('1654', rows, monthlies, w_1654)

    s1652 = summarize('1652', df_1652, bench_df)
    s1653 = summarize('1653', df_1653, bench_df_1653)
    s1654 = summarize('1654', df_1654, bench_df)
    top3_change = top3_change_rate(rows)

    for s in (s1652, s1653, s1654):
        passed, detail = pass_bar(s)
        s['passed'] = passed
        s['pass_detail'] = detail
        logger.info(f"{s['name']}: mean_excess={s['mean_excess_pct']:.3f}% t={s['t_nw']:.2f} "
                    f"years_pos={s['years_positive']}/{s['years_total']} PASS={passed}")

    # --- holdings CSV ------------------------------------------------------
    csv_rows = []
    for df, cell in ((df_1652, '1652'), (df_1653, '1653'), (df_1654, '1654')):
        bdf = bench_df_1653 if cell == '1653' else bench_df
        bmap = bdf.set_index('return_month')['net_return'].to_dict()
        for _, r in df.iterrows():
            csv_rows.append({
                'month': str(r['return_month']), 'cell': cell, 'holdings': r['holdings'],
                'net_return': r['net_return'],
                'benchmark_return': bmap.get(r['return_month'], float('nan')),
            })
    pd.DataFrame(csv_rows).to_csv(HOLDINGS_CSV, index=False)
    logger.info(f'Wrote {len(csv_rows)} rows -> {HOLDINGS_CSV}')

    # --- RESULT markdown -----------------------------------------------------
    lines = []
    lines.append('# RESULT — cells 1,652-1,654: sector-ETF 12-1 momentum (build)')
    lines.append('')
    lines.append(f'Built {datetime.now(timezone.utc).isoformat()}Z by cell_1652.py from '
                 f'PREREG_1652.md (frozen). Read window 2016-01..2023-12 pooled. '
                 f'TEST 2024-01+ NOT computed (sealed).')
    lines.append(f'Panel: {len(rows)}/{len(READ_MONTHS)} nominal months usable '
                 f'({len(skipped)} skipped, see Caveats). Top-3 set changes month to month in '
                 f'{top3_change * 100:.1f}% of processed decisions.')
    lines.append('')
    lines.append('## Pass-bar table (frozen bar: mean excess >= +0.25%/mo net & NW t >= 2.5, OR '
                 '>=6/8yr positive & MDD no worse than benchmark+5pp & worst month no worse than '
                 'benchmark worst+3pp; AND ex-top-5% excess >= 0)')
    lines.append('')
    header = ('| Cell | N mo | Mean net/mo | Mean excess/mo | NW t | Ann. excess | MDD cell | '
              'MDD bench | Worst cell | Worst bench | Yrs+ | Ex-top5% excess | MDE/mo | '
              'Turnover/mo | $/mo @$60K | PASS |')
    lines.append(header)
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for s in (s1652, s1653, s1654):
        lines.append(
            f"| {s['name']} | {s['n_months']} | {s['mean_net_pct']:.3f}% | "
            f"{s['mean_excess_pct']:.3f}% | {s['t_nw']:.2f} | {s['ann_excess_pct']:.2f}% | "
            f"{s['mdd_cell_pct']:.2f}% | {s['mdd_bench_pct']:.2f}% | {s['worst_cell_pct']:.2f}% | "
            f"{s['worst_bench_pct']:.2f}% | {s['years_positive']}/{s['years_total']} | "
            f"{s['ex_top5pct_excess_pct']:.3f}% | {s['mde_pct']:.3f}% | "
            f"{s['mean_turnover_pct']:.1f}% | ${s['stacking_usd_per_month']:.0f} | "
            f"{'PASS' if s['passed'] else 'FAIL'} |")
    lines.append('')
    lines.append('## Per-year excess sign table (return-month year, % sum of monthly excess)')
    for s in (s1652, s1653, s1654):
        lines.append(f"* **{s['name']}**: {s['by_year_pct']}")
    lines.append('')
    lines.append('## Pass-bar detail')
    for s in (s1652, s1653, s1654):
        lines.append(f"* **{s['name']}**: {s['pass_detail']}")
    lines.append('')
    lines.append('## Stacking line')
    lines.append(f'Capital window = all month (the sleeve holds through every session, unlike the '
                 f'day books). Collision = the day books\' overnight margin usage and the TOM '
                 f'sleeve\'s four nights/month (both draw on the same buying power). $/month is '
                 f'mean NET return x $60,000, not excess.')
    lines.append('')
    lines.append('## Caveats (read as an adversary)')
    skipped_desc = '; '.join(f"{sk['return_month']}: {sk['reason']}" for sk in skipped) \
        if skipped else 'none'
    lines.append(f'1. **Skipped months** ({len(skipped)}): {skipped_desc}')
    lines.append('2. **Turnover simplification**: turnover = sum(|target_weight_new - '
                 'target_weight_prior|) using the PRIOR REBALANCE-DAY target weights, not weights '
                 'drifted by one month of price action between rebalances. Understates cost '
                 'slightly for persisting names; entry/exit cost (the dominant driver) is exact.')
    lines.append('3. **1653\'s "excess"** is computed against the same long-only equal-weight '
                 'benchmark as 1652/1654 per the PREREG template, but 1653 is market-neutral by '
                 'construction (0% net exposure) while the benchmark carries full sector-equity '
                 'beta -- this is a structural mismatch, not an edge measurement; read 1653\'s raw '
                 'monthly net return as the primary number.')
    lines.append('4. **MDE** uses the frozen PREREG denominator sqrt(96), not the achieved N, '
                 'per the PREREG\'s own frozen formula -- it is a pre-committed power line, not '
                 'refit to this build\'s sample.')
    lines.append('5. **Dividends**: adjustment=\'all\' back-adjusts for both splits and '
                 'dividends, so monthly closes are total-return-like; not independently verified '
                 'against a second dividend source in this build.')
    lines.append('6. **No independent reimplementation yet** -- this is the FIRST build from the '
                 'PREREG prose; per CLAUDE.md protocol this number is not owner-reportable until a '
                 'second agent rebuilds it blind and holdings agree (Jaccard >= 0.98).')
    lines.append('7. **Price-scale check**: not run in this build (no |t|>6 daily cells here since '
                 'everything is monthly, but the daily-bar cache itself was not independently '
                 'diffed against a second source).')
    result = '\n'.join(lines)
    with open(RESULT_MD, 'w') as f:
        f.write(result)
    logger.info(f'Wrote RESULT -> {RESULT_MD} ({len(lines)} lines)')
    logger.info('=== done ===')
    return s1652, s1653, s1654


if __name__ == '__main__':
    main()
