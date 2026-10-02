#!/usr/bin/env python3
"""Cell 1,700h -- owner: "so this is due to NVIDIA and similar?" Per-name attribution of the A1 momentum
book (cell 1,700g's V1_N20: U2 large caps -- price>=$10, ADV20>=$200M point-in-time incl. delisted --
top 20 by 12-1 momentum, equal weight, weekly Monday rebalance, cost 5bps/side + half the (high-low)/close
spread proxy capped 20bps) year by year 2017-2026: which names drove the book's return, and whether the
AI/semiconductor complex dominates the strong years (2020, 2024-2026, where the book beat SPY by 40-50pts)
versus the weak years (2017, 2021, 2023, where it lost to SPY).

Reuses research/momentum_weekly/1700g_vol.py's panel loader, U2 universe/name-exclusion filter, 12-1
signal construction, weekly calendar builder and cost model VERBATIM (COPIED, not imported --
1700g_vol.py executes its full 9-cell grid at module scope, so importing it would re-run that grid as a
side effect; same reason 1700g copied from 1700d rather than importing it). V2/V6 signals and the V3/V4
own-book-vol machinery are dropped -- out of scope for the A1 (V1_N20) reference book.

The ONLY change versus 1700g/1700d's `run_cell`: instead of only aggregating gross_ret = mean(fwd) per
week, this keeps each held name's own weekly forward return and its contribution to that week's gross
book return (contribution_i = fwd_i.fillna(0) / port_n). By construction sum_i(contribution_i) ==
gross_ret for the week -- the same identity 1700d_grid.py's `contrib` dict and `top5_share_of` rely on --
checked explicitly below, not assumed.

Diagnostic, not a new edge claim: no parameter is chosen after seeing the attribution. N=20,
sig_col='sig_12_1', adv_cutoff=ADV_CUTOFF_U2 are 1,700g's V1_N20 cell, unchanged.

Output: 1700h_by_year_names.csv (year x symbol: weeks_held, contribution_pts, rank, plus the year's
book/SPY return, distinct-name count and top-5 gross-upside share repeated per row) and RESULT_1700h.md.
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

logging.basicConfig(filename=str(OUT / '1700h.log'), filemode='w', level=logging.INFO,
                     format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700h')
log.addHandler(logging.StreamHandler(sys.stdout))

PRICE_MIN = 10.0
HIST_MIN_DAYS = 273
ADV_CUTOFF_U2 = 200_000_000.0
WIN_START = pd.Timestamp('2017-01-01')
WIN_END = pd.Timestamp('2026-09-30')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_EXCLUDE_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|'
                              r'\bPREFERRED\b|\bRIGHTS?\b', re.IGNORECASE)
# Owner's candidate list verbatim (task prompt) -- membership is reported from the data, not assumed.
AI_SEMI_LIST = ['NVDA', 'AVGO', 'AMD', 'SMCI', 'ARM', 'TSM', 'MU', 'VST', 'PLTR', 'APP', 'MSTR', 'COIN']

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
lag_needed = sorted({0, 21, 252, 273})
close_lag = {lag: (panel['close'] if lag == 0 else g['close'].shift(lag)) for lag in lag_needed}
panel['sig_12_1'] = close_lag[21] / close_lag[252] - 1
panel['history_ok'] = close_lag[273].notna()
panel = panel.drop(columns=['dvol'])
log.info('%s signals built; history_ok True for %d/%d rows', elapsed(), int(panel['history_ok'].sum()), len(panel))

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

sig_rows = panel.loc[panel.bar_date.isin(signal_dates) & (panel['close'] >= PRICE_MIN) & panel['history_ok'],
                      ['symbol', 'bar_date', 'close', 'adv20', 'sig_12_1']]
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


# ============================================================================== selection + attribution ==
def run_cell_attrib(cal_df_: pd.DataFrame, adv_cutoff: float, sig_col: str, n: int):
    """1700g_vol.py's run_cell (N fixed to one value here), EXTENDED to also record each held name's own
    weekly forward return and its contribution (fwd.fillna(0)/port_n -- sums exactly to gross_ret, the
    identity 1700d_grid.py's `contrib` dict relies on) per week, for the per-year rollup below."""
    prev_port = set()
    rows = []
    name_weeks = []  # (entry_date, symbol, weekly_return_raw, contribution)
    k_lt_n = 0
    for _, wrow in cal_df_.iterrows():
        sdate, edate, nxt = wrow['prior_signal_date'], wrow['entry_date'], wrow['next_entry_date']
        day = sig_by_date.get(sdate)
        if day is not None:
            m = (day['adv20'] >= adv_cutoff) & day[sig_col].notna()
            pool = day.loc[m, ['symbol', sig_col]]
        else:
            pool = pd.DataFrame(columns=['symbol', sig_col])
        k = len(pool)
        ranked = pool.nlargest(min(k, n), sig_col)['symbol'].tolist() if k else []
        fwd_full = fwd_ret_row(edate, nxt) if k else None
        spy_r = spy_fwd(edate, nxt)
        cr = cost_rate_piv.loc[edate] if k else None
        if k < n:
            k_lt_n += 1
        port = set(ranked[:n])
        bought, sold = port - prev_port, prev_port - port
        if port:
            fwd = fwd_full.reindex(list(port))
            n_missing = int(fwd.isna().sum())
            fwd_filled = fwd.fillna(0)
            gross = fwd_filled.mean()
            for sym, r_raw, r_fill in zip(fwd.index, fwd.values, fwd_filled.values):
                name_weeks.append((edate, sym, float(r_raw), float(r_fill) / len(port)))
        else:
            n_missing, gross = 0, np.nan
        denom = len(port) if port else 1
        cost = 0.0
        if bought or sold:
            cost = (cr.reindex(list(bought)).fillna(0.002).sum() + cr.reindex(list(sold)).fillna(0.002).sum()) / denom
        turnover = len(bought) / denom
        rows.append(dict(entry_date=edate, n_eligible=k, port_n=len(port), gross_ret=gross, cost=cost,
                          net_ret=(gross - cost if pd.notna(gross) else np.nan), turnover=turnover,
                          n_fwd_missing=n_missing, spy_fwd=spy_r))
        prev_port = port
    real_df = pd.DataFrame(rows).set_index('entry_date')
    if k_lt_n:
        log.warning('%s sig=%s N=%d: %d/%d periods had fewer than %d eligible names', elapsed(),
                    sig_col, n, k_lt_n, len(cal_df_), n)
    name_df = pd.DataFrame(name_weeks, columns=['entry_date', 'symbol', 'weekly_return', 'contribution'])
    name_df['entry_date'] = pd.to_datetime(name_df['entry_date'])
    return real_df, name_df


log.info('%s STEP 3: running V1_N20 (A1 reference: top 20 by 12-1, U2, weekly) with attribution...', elapsed())
real_df, name_df = run_cell_attrib(weekly_cal, ADV_CUTOFF_U2, 'sig_12_1', 20)
real_df.index = pd.to_datetime(real_df.index)
real_df['year'] = real_df.index.year
name_df['year'] = name_df['entry_date'].dt.year
log.info('%s A1 book: %d weeks, %d (week,name) holdings, %d distinct names ever held', elapsed(),
          len(real_df), len(name_df), name_df['symbol'].nunique())

# Sanity check the identity this whole attribution depends on: sum of a week's per-name contributions
# must equal that week's gross_ret -- else the per-name split is wrong. Tolerance is 1e-6, not 0: `open`
# is cast to float32 (memory choice, copied from 1700g/1700d) so gross_ret = fwd.fillna(0).mean() is a
# float32 reduction while each contribution is upcast to float64 (via float(r_fill)) before summing --
# the two accumulate rounding differently. Observed gap on this panel is ~1.8e-8, consistent with
# float32 ULP at return-sized magnitudes; a real indexing/logic bug would show up orders of magnitude
# larger (any single misattributed name moves the gap into the 1e-2..1e-1 range), so 1e-6 still catches
# that while not flagging float32 noise as an error.
chk = name_df.groupby('entry_date')['contribution'].sum()
gross_chk = real_df['gross_ret'].reindex(chk.index)
max_gap = float((chk - gross_chk).abs().max()) if len(chk) else float('nan')
if not (max_gap <= 1e-6):
    log.error('%s contribution identity BROKEN: max |sum(contrib)-gross_ret| = %.3e -- attribution invalid',
              elapsed(), max_gap)
    raise SystemExit(1)
log.info('%s contribution identity holds (max gap %.2e over %d weeks, float32-dtype tolerance 1e-6)',
          elapsed(), max_gap, len(chk))


def top5_share_of(contrib_vals) -> float:
    """1700d_grid.py's top5_share_of: top-5 contribution as a share of the sum of POSITIVE contributions
    (the book's gross upside that period), not of the net book return."""
    vals = np.asarray(contrib_vals, dtype=float)
    if len(vals) == 0:
        return np.nan
    pos_sum = vals[vals > 0].sum()
    if pos_sum <= 0:
        return np.nan
    top5 = np.sort(vals)[-5:].sum()
    return float(top5 / pos_sum)


# ============================================================================== year-by-year rollup ==
out_records = []
md_lines = [
    '# RESULT 1700h -- per-name attribution of the A1 momentum book, 2017-2026',
    '',
    f'Generated {pd.Timestamp.now(tz="UTC").isoformat()}. Book = cell 1,700g\'s V1_N20 (U2 large caps: '
    'price>=$10, ADV20>=$200M point-in-time incl. delisted; top 20 by 12-1 momentum; equal weight; weekly '
    'Monday rebalance; cost 5bps/side + half the high-low/close spread proxy, capped 20bps).',
    '',
    'Convention: contribution_pts = 100 * sum of weekly `fwd_ret.fillna(0)/port_n` for that name in that '
    'year. By construction this sums EXACTLY to the year\'s arithmetic sum of weekly gross (pre-cost) '
    'returns -- checked explicitly in 1700h.log, not assumed. It is NOT percentage points of the '
    '`book_net_return` column (the compounded, cost-adjusted headline figure) -- cost and arithmetic-vs-'
    'geometric compounding both drive the gap, left visible rather than asserted away. top5_share is the '
    'top-5 names\' contribution as a share of that year\'s sum of POSITIVE contributions only (gross '
    'upside), matching 1700d_grid.py\'s existing top5_share_of.',
]

years = sorted(real_df['year'].unique())
for yr in years:
    yr_book = real_df[real_df['year'] == yr]
    yr_names = name_df[name_df['year'] == yr]
    book_net = float(np.nanprod(1 + yr_book['net_ret'].values) - 1)
    spy_ret = float(np.nanprod(1 + np.nan_to_num(yr_book['spy_fwd'].values, nan=0.0)) - 1)
    distinct_names = int(yr_names['symbol'].nunique())
    per_name = yr_names.groupby('symbol').agg(weeks_held=('contribution', 'size'),
                                               contribution=('contribution', 'sum')).reset_index()
    per_name['contribution_pts'] = per_name['contribution'] * 100
    per_name = per_name.sort_values('contribution_pts', ascending=False).reset_index(drop=True)
    per_name['rank'] = per_name.index + 1
    top5_share = top5_share_of(per_name['contribution'].values)
    for _, r in per_name.iterrows():
        out_records.append(dict(year=yr, symbol=r['symbol'], weeks_held=int(r['weeks_held']),
                                 contribution_pts=round(float(r['contribution_pts']), 4), rank=int(r['rank']),
                                 book_net_return_pct=round(book_net * 100, 2), spy_return_pct=round(spy_ret * 100, 2),
                                 distinct_names=distinct_names,
                                 top5_share_of_gross_upside=round(top5_share, 4) if pd.notna(top5_share) else np.nan))

    top10 = per_name.head(10)
    bottom5 = per_name.tail(5).sort_values('contribution_pts')
    ai_in_top10 = top10[top10['symbol'].isin(AI_SEMI_LIST)]
    top10_sum = top10['contribution_pts'].sum()
    ai_share_top10 = float(ai_in_top10['contribution_pts'].sum() / top10_sum) if top10_sum > 0 else np.nan
    ai_names_seen = sorted(set(top10['symbol']) & set(AI_SEMI_LIST))

    md_lines.append(f'\n## {yr}: book net {book_net*100:+.1f}%  SPY {spy_ret*100:+.1f}%  '
                     f'(excess {(book_net-spy_ret)*100:+.1f} pts)  --  {distinct_names} distinct names held')
    if pd.notna(top5_share):
        md_lines.append(f'Top-5 share of gross upside: {top5_share*100:.0f}%')
    else:
        md_lines.append('Top-5 share of gross upside: n/a (no positive gross)')
    md_lines.append('\nTop 10 contributors (symbol, weeks held, contribution pts):')
    for _, r in top10.iterrows():
        md_lines.append(f'  {r["symbol"]:<6} weeks={int(r["weeks_held"]):>3}  {r["contribution_pts"]:+6.2f} pts')
    md_lines.append('\nBottom 5 detractors:')
    for _, r in bottom5.iterrows():
        md_lines.append(f'  {r["symbol"]:<6} weeks={int(r["weeks_held"]):>3}  {r["contribution_pts"]:+6.2f} pts')
    if pd.notna(ai_share_top10):
        md_lines.append(f'\nAI/semiconductor complex in top 10: {ai_names_seen or "none"} -- '
                         f'{ai_share_top10*100:.0f}% of top-10 contribution')
    else:
        md_lines.append(f'\nAI/semiconductor complex in top 10: {ai_names_seen or "none"}')
    if pd.notna(ai_share_top10) and ai_share_top10 >= 0.5:
        verdict = 'DOMINATED'
    elif ai_names_seen:
        verdict = 'partially drove'
    else:
        verdict = 'did NOT drive'
    md_lines.append(f'Verdict: the AI/semiconductor complex {verdict} this year\'s top contributors.')
    log.info('%s %d: book %+.1f%% spy %+.1f%% names=%d top5share=%s ai_top10=%s', elapsed(), yr,
             book_net * 100, spy_ret * 100, distinct_names,
             f'{top5_share*100:.0f}%' if pd.notna(top5_share) else 'n/a', ai_names_seen)

out_df = pd.DataFrame(out_records)
out_path = OUT / '1700h_by_year_names.csv'
out_df.to_csv(out_path, index=False)
log.info('%s wrote %s (%d rows)', elapsed(), out_path.name, len(out_df))

md_path = OUT / 'RESULT_1700h.md'
md_text = '\n'.join(md_lines) + '\n'
md_path.write_text(md_text)
log.info('%s wrote %s (%d lines)', elapsed(), md_path.name, md_text.count('\n'))
log.info('%s DONE', elapsed())
