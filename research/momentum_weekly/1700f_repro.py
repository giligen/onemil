#!/usr/bin/env python3
"""Cell 1,700f -- reconciliation diagnostic (NOT a strategy claim): does another bot's claimed
A1-CORE/A2/A3 momentum table match our point-in-time (PIT) universe, or a "liquid-today held fixed"
look-ahead + survivorship universe?

Owner's request: our cell 1,700d (U2 = PIT price>=$10 & ADV20>=$200M incl. delisted, top20, 12-1,
weekly, 5bps/side + half-spread) broadly matches the other bot's claimed numbers in recent years but
diverges hugely 2017-2022 (e.g. 2020: ours +68.0% vs their +119.7%; 2021: ours -11.9% vs their +34.0%).
Hypothesis under test: their ~400-name universe is today's most-liquid names held fixed through all
history (look-ahead: you cannot know in 2017 which names will be liquid in 2026; survivorship: a name
that later delisted can never appear in a "liquid as of 2026-09-30" list).

Reuses research/momentum_weekly/1700d_grid.py's panel-load / name-pattern-exclusion / cost-model
machinery -- COPIED, not imported (1700d executes its whole pipeline at module scope as a side effect,
same reason 1700d itself copied from 1700c instead of importing it).

Three universes x N in {10,20,30}, single lookback (12-1, skip 21, t-252..t-21), weekly Mon-open
rebalance, equal weight, cost = 5bps/side ONLY (flat, no spread-proxy add-on -- matches the other
bot's stated "5bps" cost; this deliberately differs from 1700d's U2 cost model, which added a
half-spread proxy on top of the 5bps):
  (a) PIT       -- price>=$10 AND adv20>=$200M, evaluated AT EACH signal date (= 1700d's U2 rule)
  (b) TODAYFIX  -- 400 symbols with the highest adv20 on the last trading day <=2026-09-30
                   (price>=$10 that day), membership held FIXED for every 2017-2026 period
  (c) TODAYFIX+FULLHIST -- (b) AND a 273-day-history requirement at each date; NOT separately run --
                   identical to (b) by construction (see NOTE at the simulation call site below).

No null draws, no SPY column, no PASS bar -- this is a reconciliation table against a hypothesis, not
a research claim for the owner to act on.
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

logging.basicConfig(filename=str(OUT / '1700f.log'), filemode='w', level=logging.INFO,
                     format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700f')
log.addHandler(logging.StreamHandler(sys.stdout))

PRICE_MIN = 10.0
ADV_CUTOFF_PIT = 200_000_000.0
FIXED_N = 400
WIN_START = pd.Timestamp('2017-01-02')
WIN_END = pd.Timestamp('2026-09-30')
START_CAPITAL = 50_000.0
COST_PER_SIDE = 0.0005  # 5bps -- matches the other bot's claimed cost, NOT 1700d's +half-spread model
N_LIST = [10, 20, 30]
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_EXCLUDE_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|'
                              r'\bPREFERRED\b|\bRIGHTS?\b', re.IGNORECASE)

t0 = time.time()


def elapsed() -> str:
    return f'{time.time() - t0:6.0f}s'


# ===== claimed numbers from the other bot's table (owner-supplied; hardcoded for the comparison) =====
# A1-CORE (~400 liquid names, top20, skip21, weekly, 5bps, $50K from 2017-01-02) -> $967,704 end-2026
CLAIM_N20 = {2017: 27.9, 2018: -29.4, 2019: 43.6, 2020: 119.7, 2021: 34.0, 2022: -0.4,
             2023: 18.8, 2024: 62.0, 2025: 51.5, 2026: 63.0}
CLAIM_N20_END = 967_704.0
CLAIM_N10 = {2020: 164.0, 2021: 79.7}   # A2 top-10, only these two years given by the owner
CLAIM_N30 = {2020: 108.7, 2021: 37.7}   # A3 top-30, only these two years given by the owner
CLAIM = {10: CLAIM_N10, 20: CLAIM_N20, 30: CLAIM_N30}
CLAIM_END = {10: None, 20: CLAIM_N20_END, 30: None}

# ============================================================ load panel (copied from 1700d_grid.py) ==
panel_path = OUT / 'panel_2016_2026.parquet'
log.info('%s STEP 1: loading panel %s', elapsed(), panel_path.name)
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

assets_df = pd.read_csv(OUT / '1700c_assets.csv', dtype={'symbol': str, 'name': str})
assets_df['name'] = assets_df['name'].fillna('')
excl_name = set(assets_df.loc[assets_df['name'].str.contains(NAME_EXCLUDE_RE), 'symbol'])
excl_test = {s for s in raw['symbol'].unique() if TEST_RE.match(s)}
excluded = excl_name | excl_test
log.info('%s universe exclusion: %d symbols (%d name-pattern ETF/ETN/FUND/TRUST/WARRANT/UNIT/'
          'PREFERRED/RIGHT + %d test-ticker ^Z[A-Z]ZZT$)', elapsed(), len(excluded), len(excl_name), len(excl_test))

panel = raw[(raw.symbol != 'SPY') & ~raw.symbol.isin(excluded)].copy()
panel['symbol'] = panel['symbol'].cat.remove_unused_categories()
del raw
panel = panel.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
g = panel.groupby('symbol', sort=False)
panel['dvol'] = panel['close'] * panel['volume']
panel['adv20'] = g['dvol'].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
close_lag21 = g['close'].shift(21)
close_lag252 = g['close'].shift(252)
close_lag273 = g['close'].shift(273)
panel['sig'] = close_lag21 / close_lag252 - 1
panel['history_ok'] = close_lag273.notna()
panel = panel.drop(columns=['dvol'])
log.info('%s signal built (12-1, skip 21); history_ok True for %d/%d rows', elapsed(),
          int(panel['history_ok'].sum()), len(panel))

# ===== fixed-400 universe: highest adv20 on the last trading day <=2026-09-30, price>=$10 that day ====
asof_date = panel.loc[panel.bar_date <= WIN_END, 'bar_date'].max()
asof_rows = panel[(panel.bar_date == asof_date) & (panel['close'] >= PRICE_MIN) & panel['adv20'].notna()]
FIXED_400 = set(asof_rows.nlargest(FIXED_N, 'adv20')['symbol'])
min_adv_selected = asof_rows.nlargest(FIXED_N, 'adv20')['adv20'].min() if len(FIXED_400) else float('nan')
log.info('%s TODAY-FIXED-%d universe snapshot date %s: %d eligible (price>=$10, adv20 notna), %d selected, '
          'min selected adv20 $%.0fM', elapsed(), FIXED_N, asof_date.date(), len(asof_rows), len(FIXED_400),
          min_adv_selected / 1e6)
if len(FIXED_400) < FIXED_N:
    log.warning('%s only %d/%d names available for the fixed universe on %s', elapsed(), len(FIXED_400),
                FIXED_N, asof_date.date())

# ============================================================================== weekly calendar ==
trading_days = sorted(panel['bar_date'].unique())
day_idx = {d: i for i, d in enumerate(trading_days)}
cal = pd.Series(trading_days)
period = cal.dt.to_period('W')
first = cal.groupby(period).min().sort_index()
dates = first[(first >= WIN_START) & (first <= WIN_END)].reset_index(drop=True)
prior_signal = [trading_days[day_idx[d] - 1] if day_idx[d] > 0 else pd.NaT for d in dates]
weekly_cal = pd.DataFrame({'entry_date': dates.values, 'prior_signal_date': prior_signal})
weekly_cal['next_entry_date'] = weekly_cal['entry_date'].shift(-1)
weekly_cal = weekly_cal.dropna(subset=['next_entry_date']).reset_index(drop=True)
log.info('%s weekly calendar: %d complete periods, %s .. %s', elapsed(), len(weekly_cal),
          weekly_cal.entry_date.min().date(), weekly_cal.entry_date.max().date())

needed_dates = sorted(set(weekly_cal.entry_date) | set(weekly_cal.next_entry_date))
signal_dates = sorted(set(weekly_cal.prior_signal_date))
open_piv = (panel[panel.bar_date.isin(needed_dates)]
            .pivot_table(index='bar_date', columns='symbol', values='open', aggfunc='last')
            .reindex(needed_dates))

sig_rows = panel.loc[panel.bar_date.isin(signal_dates) & (panel['close'] >= PRICE_MIN) & panel['history_ok'],
                      ['symbol', 'bar_date', 'adv20', 'sig']]
sig_by_date = {d: sub.drop(columns='bar_date') for d, sub in sig_rows.groupby('bar_date')}
log.info('%s sig_by_date built: %d signal dates (price>=$10 & >=273d history enforced PIT, per date)',
          elapsed(), len(sig_by_date))
del panel, sig_rows


def fwd_ret_row(entry_date, next_entry_date):
    return open_piv.loc[next_entry_date] / open_piv.loc[entry_date] - 1


MAXN = max(N_LIST)


def run_universe(mask_fn, label: str):
    """One pass over the weekly calendar: rank the eligible pool ONCE per period (mask_fn decides
    eligibility), slice top-N for every N in N_LIST from that single ranking (1700d's "compute once,
    reuse across N" pattern)."""
    prev_port = {n: set() for n in N_LIST}
    rows = {n: [] for n in N_LIST}
    k_lt_n = {n: 0 for n in N_LIST}
    for _, wrow in weekly_cal.iterrows():
        sdate, edate, nxt = wrow['prior_signal_date'], wrow['entry_date'], wrow['next_entry_date']
        day = sig_by_date.get(sdate)
        if day is not None:
            m = mask_fn(day) & day['sig'].notna()
            pool = day.loc[m, ['symbol', 'sig']]
        else:
            pool = pd.DataFrame(columns=['symbol', 'sig'])
        k = len(pool)
        ranked = pool.nlargest(min(k, MAXN), 'sig')['symbol'].tolist() if k else []
        fwd_full = fwd_ret_row(edate, nxt) if k else None
        for n in N_LIST:
            if k < n:
                k_lt_n[n] += 1
            port = set(ranked[:n])
            bought, sold = port - prev_port[n], prev_port[n] - port
            if port:
                fwd = fwd_full.reindex(list(port))
                gross = fwd.fillna(0).mean()
            else:
                gross = np.nan
            denom = len(port) if port else 1
            cost = (len(bought) + len(sold)) * COST_PER_SIDE / denom
            net = gross - cost if pd.notna(gross) else np.nan
            rows[n].append(dict(entry_date=edate, n_eligible=k, port_n=len(port), net_ret=net))
            prev_port[n] = port
    log.info('%s %s done; periods with k<N by N: %s', elapsed(), label, k_lt_n)
    return {n: pd.DataFrame(rows[n]).set_index('entry_date') for n in N_LIST}, k_lt_n


def mask_pit(day):
    return day['adv20'] >= ADV_CUTOFF_PIT


def mask_todayfix(day):
    return day['symbol'].isin(FIXED_400)


log.info('%s STEP 2: simulating universe (a) PIT ...', elapsed())
pit_by_n, pit_klt = run_universe(mask_pit, 'PIT (U2: price>=$10 & adv20>=$200M, PIT per date)')
log.info('%s STEP 3: simulating universe (b) TODAY-FIXED-%d ...', elapsed(), FIXED_N)
fix_by_n, fix_klt = run_universe(mask_todayfix, f'TODAY-FIXED-{FIXED_N} (membership frozen at {asof_date.date()})')
# universe (c): TODAY-FIXED intersected with a 273-day-history requirement at each date -- sig_by_date
# (shared by (a) AND (b)) already requires history_ok for every row it contains, so (c) cannot differ
# from (b) on this population; not separately simulated (reported as identical-by-construction).
log.info('%s universe (c) TODAYFIX+FULLHIST == (b) by construction (sig_by_date already requires '
          'history_ok for every row) -- not separately simulated', elapsed())

# ======================================================================= by-year + end-2026 $ table ==
def by_year(df: pd.DataFrame) -> dict:
    out = {}
    for yr, sub in df.groupby(df.index.year):
        out[int(yr)] = float(np.prod(1 + sub['net_ret'].fillna(0)) - 1)
    return out


def end_value(df: pd.DataFrame) -> float:
    return float(START_CAPITAL * np.prod(1 + df['net_ret'].fillna(0)))


years = sorted(set(weekly_cal.entry_date.dt.year))
out_rows = []
end_vals = {}
for n in N_LIST:
    pit_yr, fix_yr = by_year(pit_by_n[n]), by_year(fix_by_n[n])
    end_vals[(n, 'PIT')] = end_value(pit_by_n[n])
    end_vals[(n, 'TODAYFIX')] = end_value(fix_by_n[n])
    claim_yr = CLAIM[n]
    for yr in years:
        out_rows.append(dict(N=n, year=yr, pit_return_pct=100 * pit_yr.get(yr, float('nan')),
                              todayfix_return_pct=100 * fix_yr.get(yr, float('nan')),
                              claim_return_pct=claim_yr.get(yr, float('nan')),
                              pit_n_periods=int(len(pit_by_n[n].loc[pit_by_n[n].index.year == yr])),
                              fix_n_periods=int(len(fix_by_n[n].loc[fix_by_n[n].index.year == yr]))))
by_year_df = pd.DataFrame(out_rows)
by_year_df.to_csv(OUT / '1700f_by_year.csv', index=False)
log.info('%s wrote 1700f_by_year.csv (%d rows)', elapsed(), len(by_year_df))

# ================================================================================ RESULT.md ==
def fmt(v):
    return 'NA' if v is None or pd.isna(v) else f'{v:+.1f}'


lines = []
lines.append('# RESULT -- cell 1,700f: reconciliation of 1,700d vs the other bot\'s A1/A2/A3 table')
lines.append('')
lines.append('NOT a strategy claim -- a diagnostic to see which universe construction reproduces the '
             'other bot\'s 2017-2022 numbers. 12-1 momentum (skip 21), weekly Mon-open rebalance, equal '
             'weight, 5bps/side ONLY (flat -- matches the other bot\'s stated cost, not 1700d\'s '
             f'+half-spread model). $50,000 compounding from {WIN_START.date()}. Universes: (a) PIT = '
             'price>=$10 & adv20>=$200M evaluated at each signal date (=1700d U2); (b) TODAY-FIXED-400 = '
             f'the 400 highest-adv20 names (price>=$10) on {asof_date.date()}, membership frozen for all '
             'periods (deliberate look-ahead + survivorship); (c) TODAY-FIXED-400 + full-273-day-history '
             '== (b) by construction here -- sig_by_date already requires history_ok (>=273 trading days) '
             'for every row shared by (a) and (b), so adding that requirement to (b) changes nothing; not '
             'separately run.')
lines.append('')
lines.append('Two ETP/name-pattern exclusions applied (shared with cell 1,700d -- same panel, same assets '
             'file): (1) company-name regex ETF/ETN/FUND/TRUST/WARRANT/UNIT/PREFERRED/RIGHT '
             '(case-insensitive); (2) test-ticker regex `^Z[A-Z]ZZT$`.')
lines.append('')
for n in N_LIST:
    lines.append(f'## N={n}')
    lines.append('')
    lines.append('| Year | PIT (a) | TODAY-FIXED-400 (b) | Their claim |')
    lines.append('|---|---|---|---|')
    sub = by_year_df[by_year_df.N == n]
    for _, r in sub.iterrows():
        lines.append(f"| {int(r.year)} | {fmt(r.pit_return_pct)}% | {fmt(r.todayfix_return_pct)}% | "
                     f"{fmt(r.claim_return_pct)}% |")
    lines.append('')
    pit_end, fix_end = end_vals[(n, 'PIT')], end_vals[(n, 'TODAYFIX')]
    claim_end = CLAIM_END[n]
    claim_end_str = f'${claim_end:,.0f}' if claim_end else 'NA (not given for this N)'
    lines.append(f'End-2026 $ from $50,000 ({WIN_START.date()}): PIT = ${pit_end:,.0f}; '
                 f'TODAY-FIXED-400 = ${fix_end:,.0f}; their claim = {claim_end_str}.')
    lines.append('')
    early_years = [int(y) for y in sub.year if pd.notna(sub[sub.year == y].claim_return_pct.iloc[0])]

    def within10(col):
        diffs = [abs(sub[sub.year == y][col].iloc[0] - sub[sub.year == y].claim_return_pct.iloc[0])
                 for y in early_years]
        return all(d <= 10 for d in diffs), max(diffs)

    if not early_years:
        concl = 'no claim values given for this N -- see table above.'
    else:
        pit_ok, pit_max = within10('pit_return_pct')
        fix_ok, fix_max = within10('todayfix_return_pct')
        concl = (f'PIT {"matches" if pit_ok else "does NOT match"} (max |diff| {pit_max:.1f}pt); '
                 f'TODAY-FIXED-400 {"matches" if fix_ok else "does NOT match"} (max |diff| {fix_max:.1f}pt) '
                 f'within +-10pt/yr over {early_years}.')
    lines.append(f'**Conclusion N={n}**: {concl}')
    lines.append('')
(OUT / 'RESULT_1700f.md').write_text('\n'.join(lines) + '\n')
log.info('%s wrote RESULT_1700f.md (%d lines)', elapsed(), len(lines))
log.info('%s ALL DONE', elapsed())
