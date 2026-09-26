#!/usr/bin/env python3
"""PREREG_1550 step 3 — cells 1,550 (N=10) and 1,551 (N=25): the overnight new-high leg.

Rule (frozen, `research/overnight_high/PREREG_1550.md`): universe close >= $5 and 20-day dollar
volume >= $10M, no test tickers; signal = close_t is a new 252-session high (on CLOSE, strictly
prior sessions) with volume_t >= 1.5x ADV20 (through t-1); rank the day's signal names by volume
ratio descending, take the top N; buy the day-t MOC, sell the day-t+1 MOO; report gross and net
(2/5/10 bps round trip) per sample and per calendar year.

Samples: EXTENSION (`panel_extension.parquet`, 2019-01-02..2024-06-28, single undivided read --
never previously seen) and PANEL (`panel_2024_2026.parquet`, TRAIN < 2026-01-01, VAL 2026-01..05,
TEST >= 2026-06-01, TEST already spent by the disclosed prior run).

Every statistic the PREREG lists is written to RESULT_1550.md's tables by
`research/overnight_high/RESULT_1550.md` (built from this script's stdout + the nights CSV);
this script also writes `cell_1550_nights.csv` (one row per name-night in a book).
"""
import logging
import os
import random
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from cadence_bar import (  # noqa: E402
    build_weekly_series, compute_cycles, score_c1, score_c3, percentile,
    classify_week, _green_share, parse_date,
)

log = logging.getLogger('cell_1550')

TEST_TICKER_RE = re.compile(r'^Z[A-Z]ZZT$|^ZZ')
MIN_PRICE = 5.0
MIN_DVOL20 = 1e7
VOL_SHOCK_MULT = 1.5
N_CELLS = {1550: 10, 1551: 25}
COST_BPS = (2, 5, 10)
PRIMARY_COST_BPS = 5
NULL_DRAWS = 1000
NULL_SEED = 1550
STRONG_R = 5.0
R_DOLLARS_PER_NAME = 3000
MAX_ABS_RET = 0.30  # +-30% night flag (price-scale refuter)


def base_universe(d: pd.DataFrame) -> pd.DataFrame:
    """Rows passing the universe gate: price, dollar volume, no test ticker, a valid ret_on_next."""
    keep = (
        (d.close >= MIN_PRICE) & (d.dvol20 >= MIN_DVOL20)
        & d.ret_on_next.notna() & (d.ret_on_next.abs() <= 0.5)
        & ~d.symbol.astype(str).str.match(TEST_TICKER_RE)
    )
    return d[keep].copy()


def signal_rows(u: pd.DataFrame) -> pd.DataFrame:
    """Rows passing the H signal on top of the base universe: new 252d close-high, vol shock."""
    sig = u.high252.notna() & (u.close >= u.high252) & u.vol_ratio.notna() & (u.vol_ratio >= VOL_SHOCK_MULT)
    return u[sig].copy()


def top_n_book(sig: pd.DataFrame, n: int) -> pd.DataFrame:
    return (sig.sort_values(['bar_date', 'vol_ratio'], ascending=[True, False])
               .groupby('bar_date', as_index=False, group_keys=False).head(n))


def day_clustered_t(day_means: pd.Series) -> float:
    """t-stat on the per-day mean series (the night is the cluster)."""
    n = len(day_means)
    if n < 2 or day_means.std(ddof=1) == 0 or np.isnan(day_means.std(ddof=1)):
        return float('nan')
    return day_means.mean() / (day_means.std(ddof=1) / np.sqrt(n))


def week_str(d: str) -> str:
    return parse_date(d).isocalendar()[:2].__repr__()  # placeholder, unused; kept for clarity


def cadence_stats(book: pd.DataFrame, cost_frac: float):
    """Weekly P10 and strong-week gap via scripts/cadence_bar.py, R := net_ret at $3K/name
    (position size == the notional at risk, so R == the net fractional return)."""
    trades = [{'date': parse_date(r.bar_date), 'r': (r.ret_on_next - cost_frac), 'symbol': r.symbol}
              for r in book.itertuples()]
    if not trades:
        return {'p10': None, 'green_share': None, 'gap_median': None, 'gap_p90': None, 'n_strong_weeks': 0}
    lo = min(t['date'] for t in trades)
    hi = max(t['date'] for t in trades)
    weekly = build_weekly_series(trades, lo, hi)
    c3 = score_c3(weekly, p10_thresh=-2.0, min_thresh=-4.0, mdd_thresh=8.0, underwater_thresh=6)
    green = _green_share(weekly)
    cycles, strong_idx = compute_cycles(weekly, STRONG_R)
    c1 = score_c1(cycles, gap_median_thresh=3.0, gap_p90_thresh=6.0)
    return {'p10': c3['p10'], 'green_share': green, 'gap_median': c1['median'], 'gap_p90': c1['p90'],
            'n_strong_weeks': len(strong_idx)}


def count_matched_null(u: pd.DataFrame, n: int, seed: int, n_draws: int = NULL_DRAWS) -> np.ndarray:
    """1,000 draws of N random ELIGIBLE (base-universe) names per night; net bps/night per draw."""
    rng = random.Random(seed)
    by_day = {day: grp['ret_on_next'].tolist() for day, grp in u.groupby('bar_date')}
    days = [d for d, rets in by_day.items() if len(rets) >= 1]
    draw_means = np.empty(n_draws)
    for k in range(n_draws):
        picked = []
        for d in days:
            rets = by_day[d]
            m = min(n, len(rets))
            picked.extend(rng.sample(rets, m))
        draw_means[k] = float(np.mean(picked)) if picked else float('nan')
    return draw_means


def stats_block(book: pd.DataFrame, universe_for_period: pd.DataFrame, n: int, cost_bps_primary: int,
                 null_seed: int) -> dict:
    """Every PREREG-listed statistic for one (sample, cell, period) slice."""
    out = {'n_nights': book.bar_date.nunique(), 'n_fills': len(book),
           'names_per_day': len(book) / max(1, book.bar_date.nunique())}
    gross = book.ret_on_next
    out['gross_bps'] = gross.mean() * 1e4 if len(gross) else float('nan')
    day_means_gross = book.groupby('bar_date').ret_on_next.mean()
    for cb in COST_BPS:
        net = gross - cb / 1e4
        out[f'net_bps_{cb}'] = net.mean() * 1e4 if len(net) else float('nan')
        day_means_net = book.groupby('bar_date').apply(lambda g: (g.ret_on_next - cb / 1e4).mean())
        out[f't_dayclust_{cb}'] = day_clustered_t(day_means_net)
    # ex-top-1%/5% and winner-capped, computed at the PRIMARY cost
    net_primary = (gross - cost_bps_primary / 1e4).sort_values()
    n_fills = len(net_primary)
    if n_fills >= 20:
        k5 = max(1, int(round(0.05 * n_fills)))
        k1 = max(1, int(round(0.01 * n_fills)))
        out['ex_top5_bps'] = net_primary.iloc[:-k5].mean() * 1e4
        out['ex_top1_bps'] = net_primary.iloc[:-k1].mean() * 1e4
    else:
        out['ex_top5_bps'] = float('nan')
        out['ex_top1_bps'] = float('nan')
    capped = (gross - cost_bps_primary / 1e4).clip(upper=0.05)
    out['capped5_bps'] = capped.mean() * 1e4 if len(capped) else float('nan')
    out['flagged_30pct_nights'] = int((gross.abs() > MAX_ABS_RET).sum())
    # cadence: weekly P10 / green share / strong-week gap at the primary cost, $3K/name
    cad = cadence_stats(book, cost_bps_primary / 1e4)
    out.update({f'weekly_{k}': v for k, v in cad.items()})
    # placebo: the whole eligible (base-universe) population's own overnight return on these nights
    nights = book.bar_date.unique()
    placebo_pop = universe_for_period[universe_for_period.bar_date.isin(nights)]
    placebo_gross = placebo_pop.ret_on_next.mean() * 1e4 if len(placebo_pop) else float('nan')
    out['universe_bps'] = placebo_gross
    net_book_primary = out[f'net_bps_{cost_bps_primary}']
    out['placebo_margin_bps'] = net_book_primary - (placebo_gross - cost_bps_primary)
    day_means_book = book.groupby('bar_date').apply(lambda g: (g.ret_on_next - cost_bps_primary / 1e4).mean())
    day_means_universe = (placebo_pop.groupby('bar_date').ret_on_next.mean() - cost_bps_primary / 1e4)
    joined = pd.concat([day_means_book.rename('book'), day_means_universe.rename('universe')], axis=1).dropna()
    diff = joined.book - joined.universe
    out['placebo_t'] = day_clustered_t(diff) if len(diff) >= 2 else float('nan')
    # count-matched null
    null_dist = count_matched_null(placebo_pop, n, null_seed)
    null_net_bps = (null_dist * 1e4) - cost_bps_primary
    valid = null_net_bps[~np.isnan(null_net_bps)]
    if len(valid):
        out['null_pctile'] = float((valid < net_book_primary).mean() * 100)
    else:
        out['null_pctile'] = float('nan')
    return out


def year_of(bar_date_series):
    return bar_date_series.str[:4]


def run_sample(panel_path: str, sample_name: str, periods: dict, nights_rows: list) -> list:
    """periods: {period_label: (lo_date_str or None, hi_date_str or None)} inclusive bounds."""
    d = pd.read_parquet(panel_path)
    u_all = base_universe(d)
    sig_all = signal_rows(u_all)
    rows = []
    for period, (lo, hi) in periods.items():
        u = u_all
        sig = sig_all
        if lo:
            u = u[u.bar_date >= lo]; sig = sig[sig.bar_date >= lo]
        if hi:
            u = u[u.bar_date <= hi]; sig = sig[sig.bar_date <= hi]
        for cell, n in N_CELLS.items():
            book = top_n_book(sig, n)
            st = stats_block(book, u, n, cost_bps_primary=PRIMARY_COST_BPS, null_seed=NULL_SEED)
            st.update({'sample': sample_name, 'cell': cell, 'n': n, 'period': period})
            rows.append(st)
            for r in book.itertuples():
                nights_rows.append({'sample': sample_name, 'cell': cell, 'date': r.bar_date,
                                     'symbol': r.symbol, 'ret_on_next': r.ret_on_next})
        # per-calendar-year breakdown within this period
        for yr, grp_sig in sig.groupby(year_of(sig.bar_date)):
            grp_u = u[year_of(u.bar_date) == yr]
            for cell, n in N_CELLS.items():
                book = top_n_book(grp_sig, n)
                if book.empty:
                    continue
                st = stats_block(book, grp_u, n, cost_bps_primary=PRIMARY_COST_BPS, null_seed=NULL_SEED)
                st.update({'sample': sample_name, 'cell': cell, 'n': n, 'period': f'{period}-{yr}'})
                rows.append(st)
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    nights_rows = []
    all_rows = []
    all_rows += run_sample(os.path.join(HERE, 'panel_extension.parquet'), 'EXTENSION',
                            {'FULL': (None, None)}, nights_rows)
    all_rows += run_sample(os.path.join(HERE, 'panel_2024_2026.parquet'), 'PANEL',
                            {'TRAIN': (None, '2025-12-31'), 'VAL': ('2026-01-01', '2026-05-31'),
                             'TEST': ('2026-06-01', None)}, nights_rows)
    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(HERE, 'cell_1550_stats.csv'), index=False)
    pd.DataFrame(nights_rows).to_csv(os.path.join(HERE, 'cell_1550_nights.csv'), index=False)
    with pd.option_context('display.width', 220, 'display.max_columns', 40):
        print(res[['sample', 'period', 'cell', 'n_nights', 'names_per_day', 'gross_bps',
                    f'net_bps_{PRIMARY_COST_BPS}', f't_dayclust_{PRIMARY_COST_BPS}',
                    'ex_top5_bps', 'ex_top1_bps', 'capped5_bps', 'universe_bps',
                    'placebo_margin_bps', 'placebo_t', 'null_pctile', 'flagged_30pct_nights']]
              .to_string(index=False))
    log.info('wrote cell_1550_stats.csv (%d rows) and cell_1550_nights.csv (%d rows)',
              len(res), len(nights_rows))


if __name__ == '__main__':
    main()
