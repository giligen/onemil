#!/usr/bin/env python3
"""Cells 1,617-1,618 -- research/hod_entry/PREREG_1617.md (FROZEN 2026-09-28 17:00 UTC).

Frame A -- held break overnight. For every HOD-break base fill (cell 1,438's book, status ==
'fill' in causal_arming_causal.csv, 9,911 rows) whose break DAY closed >= the break level (the
break held into the close, per the RAW Databento daily panel
research/overnight_high/panel_2024_2026.parquet): buy the 16:00 closing auction (MOC) of the
break day, sell the next session's opening auction (MOO). Return = next_open / close - 1 (the
panel's own `ret_on_next` field -- build_panel.py L26-27: "raw close, raw next open, no
adjustment", so this is apples-to-apples against the RAW intraday `level` from
causal_arming_causal.csv -- no daily-vs-intraday price-scale mismatch). Cost = 5 bps per auction
leg, 10 bps round trip, SUBTRACTED from the raw bps return (additive, matching this repo's own
net_R = raw_R - cost_R convention in causal_arming_causal.csv).

Cell 1,617 (scored against the frozen pass bar, VAL): the held-break population above.
Cell 1,618 (report-only, no pass bar):
  (a) the mirror population -- base fills whose break day closed BELOW the level (the failed
      breaks), same stats;
  (b) the whole in-play scanner-day universe -- EVERY row of causal_arming_causal.csv (any
      status: not_armed / nofill / fill alike, ~147 symbols/day) on the SAME NIGHTS as the 1,617
      population. This is the placebo: is the overnight drift specific to a break that held, or
      just to being a name the scanner looked at that night. The placebo margin (1,617's mean net
      bps minus the universe's, same nights, paired day by day) is part of 1,617's own pass bar.

Earnings dates: NO earnings-date calendar exists in this repo. Grepped data_sources/ and
research/ for `earnings_date` / `earnings_calendar` / any `*earnings*.csv|parquet` file: none
found (research/multiday/data/edgar_earnings.py existed but is deleted on this branch; cell_1483's
"earnings_guidance" is a news HEADLINE class, not a date list). AlpacaClient.get_market_calendar
(data_sources/alpaca_client.py) is a TRADING-DAY calendar -- date/open/close -- not an earnings
calendar. Per the PREREG's own fallback, this is taken literally: coverage is reported as 0 %,
NO night is excluded for earnings, and the +-30 % raw-price check (run unconditionally -- it is
also the PREREG's independent split/price-scale refuter) is reported as the disclosed substitute
control, not as an earnings proxy.

Usage:
    nice -n 19 python3 research/hod_entry/cell_1617.py [--smoke]

    --smoke scores a fixed 200-row sample of the base fills (seed 1617, numpy RandomState) instead
    of the full ~9,911-row book, for fast iteration; writes prefixed outputs
    (cell_1617_nights_SMOKE.csv / RESULT_1617_SMOKE.md) so a smoke run never collides with the
    full run.

Outputs:
    research/hod_entry/cell_1617_nights.csv -- one row per (night, population): cell, split, date,
        symbol, close, next_open, ret_bps [RAW, UNCOSTED -- net = ret_bps - TOTAL_COST_BPS], flags
        (pipe-joined exclusion/data-quality flags; every row is kept, exclusions are flagged, never
        dropped or imputed).
    research/hod_entry/RESULT_1617.md -- every statistic in the PREREG's "Report per holdout" list
        for cells 1,617 and 1,618, the placebo margin, and the frozen 1,617 pass-bar checklist
        scored on VAL (TRAIN-H2 sign/t as the secondary confirmation line).
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s cell_1617: %(message)s')
log = logging.getLogger('cell_1617')

BASE_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
PANEL_PARQUET = os.path.join(REPO, 'research/overnight_high/panel_2024_2026.parquet')

AUCTION_COST_BPS_PER_LEG = 5.0
TOTAL_COST_BPS = 2 * AUCTION_COST_BPS_PER_LEG            # MOC entry + MOO exit, additive on bps
PRICE_SCALE_BOUND_BPS = 3000.0                           # +-30% raw-return check (splits), in bps
WINNER_CAP_BPS = 1000.0                                  # "winner-capped +10%" in bps
EARNINGS_CALENDAR_AVAILABLE = False                      # see module docstring: none found
SMOKE_N = 200
SMOKE_SEED = 1617
HOLDOUTS = ['TRAIN', 'VAL']                               # split column values (TRAIN == TRAIN-H2)

PASS_BAR = dict(
    mean_net_bps=8.0, t=2.5, nights_per_week=3.0, placebo_margin_bps=5.0, placebo_t=2.0,
    months_positive_of=(4, 6), train_t=1.0,
)


# --------------------------------------------------------------------------------------------
# Shared helpers -- ported verbatim from research/hod_entry/cell_1445.py (the PREREG names this
# file as the source of day_clustered_t / ex_top5_mean; copied in rather than imported so this
# cell has no dependency on cell_1445's heavier module-level imports (causal_arming, hod_consol)).
# --------------------------------------------------------------------------------------------

def day_clustered_t(y, day):
    """statsmodels OLS on a constant, clustered by day -- the t-stat on the mean."""
    y = pd.Series(y).dropna()
    if len(y) < 2:
        return np.nan
    d = pd.Series(day).loc[y.index]
    if d.nunique() < 2:
        return np.nan
    X = np.ones((len(y), 1))
    model = sm.OLS(y.to_numpy(), X).fit(cov_type='cluster', cov_kwds={'groups': d.to_numpy()})
    return float(model.tvalues[0])


def ex_top_pct_mean(y, pct):
    """Mean excluding the top `pct` fraction (by value) -- tail-dependence check.
    Generalises cell_1445.ex_top5_mean(pct=0.05) to also give ex-top-1% (pct=0.01)."""
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return np.nan
    k = int(round(pct * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def winner_capped_mean(y, cap=WINNER_CAP_BPS):
    y = pd.Series(y).dropna()
    if not len(y):
        return np.nan
    return float(np.minimum(y, cap).mean())


def weeks_spanned(days):
    """Distinct ISO (year, week) count over a day-string series -- the denominator for nights/wk."""
    iso = pd.to_datetime(pd.Series(days).unique())
    wk = {(d.isocalendar()[0], d.isocalendar()[1]) for d in iso}
    return max(len(wk), 1)


# --------------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------------

def load_base():
    """The full scanner-day book -- every status (not_armed/nofill/fill) is the in-play universe."""
    log.info('loading base scanner-day book %s', BASE_CSV)
    df = pd.read_csv(BASE_CSV, low_memory=False)
    df['day'] = df['day'].astype(str)
    df['symbol'] = df['symbol'].astype(str)
    log.info('base book: %d rows, %d days (%s..%s), status counts: %s',
              len(df), df['day'].nunique(), df['day'].min(), df['day'].max(),
              df['status'].value_counts().to_dict())
    dupe = df.duplicated(subset=['day', 'symbol'], keep=False).sum()
    if dupe:
        log.warning('base book has %d rows sharing a (day,symbol) key -- kept as-is, not deduped '
                     '(each row is one arming/fill event; the universe groupby is symbol-agnostic '
                     'of this)', dupe)
    return df


def load_panel():
    """Raw Databento daily panel; drop zero-OHLCV rows per the shared-input instruction."""
    log.info('loading daily panel %s', PANEL_PARQUET)
    cols = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume', 'next_open', 'ret_on_next']
    df = pd.read_parquet(PANEL_PARQUET, columns=cols)
    n0 = len(df)
    zero_mask = (df[['open', 'high', 'low', 'close', 'volume']] <= 0).any(axis=1)
    n_zero = int(zero_mask.sum())
    if n_zero:
        log.warning('panel: dropping %d/%d zero-OHLCV rows', n_zero, n0)
    df = df.loc[~zero_mask].copy()
    df['symbol'] = df['symbol'].astype(str)
    df['bar_date'] = df['bar_date'].astype(str)
    log.info('panel after zero-OHLCV drop: %d rows, %d symbols, %s..%s',
              len(df), df['symbol'].nunique(), df['bar_date'].min(), df['bar_date'].max())
    return df


# --------------------------------------------------------------------------------------------
# Population construction
# --------------------------------------------------------------------------------------------

def attach_panel(rows, panel):
    """Left-merge (day,symbol) -> panel (bar_date,symbol); count and flag every miss, never impute.

    Adds: close, next_open, ret_bps (RAW, uncosted, = next_open/close - 1 in bps), and boolean
    flag_no_panel_match / flag_no_next_open / flag_price_scale_fail / usable columns.
    """
    n0 = len(rows)
    m = rows.merge(panel[['symbol', 'bar_date', 'close', 'next_open']],
                    left_on=['symbol', 'day'], right_on=['symbol', 'bar_date'], how='left')
    no_match = m['close'].isna()
    n_no_match = int(no_match.sum())
    if n_no_match:
        log.warning('attach_panel: %d/%d rows have no panel (symbol,day) match -- excluded from '
                     'headline stats, kept in the CSV with flag_no_panel_match', n_no_match, n0)
    no_next = (~no_match) & m['next_open'].isna()
    n_no_next = int(no_next.sum())
    if n_no_next:
        log.warning('attach_panel: %d/%d matched rows have no next_open (last panel session for '
                     'the symbol) -- excluded, flag_no_next_open', n_no_next, n0)
    m['flag_no_panel_match'] = no_match
    m['flag_no_next_open'] = no_next
    valid = (~no_match) & (~no_next) & (m['close'] > 0)
    m['ret_bps'] = np.where(valid, (m['next_open'] / m['close'] - 1.0) * 10000.0, np.nan)
    m['flag_price_scale_fail'] = valid & (m['ret_bps'].abs() > PRICE_SCALE_BOUND_BPS)
    n_scale = int(m['flag_price_scale_fail'].sum())
    if n_scale:
        log.warning('attach_panel: %d rows fail the +-30%% raw-price check (likely an unadjusted '
                     'split) -- excluded from headline stats, flag_price_scale_fail, reported '
                     'separately per the PREREG', n_scale)
    m['usable'] = valid & (~m['flag_price_scale_fail'])
    log.info('attach_panel: %d/%d rows usable', int(m['usable'].sum()), n0)
    return m


def build_frame_a(base, panel, sample_n=None):
    """Returns (held_break, failed_break, universe) DataFrames, each with attach_panel columns."""
    fills = base[base['status'] == 'fill'].copy()
    if sample_n is not None:
        fills = fills.sample(n=min(sample_n, len(fills)), random_state=SMOKE_SEED).copy()
        log.info('--smoke: sampled %d/%d fills (seed %d)', len(fills), (base['status'] == 'fill').sum(), SMOKE_SEED)
    fills = attach_panel(fills, panel)

    classified = fills[fills['close'].notna()].copy()
    n_unclassifiable = len(fills) - len(classified)
    if n_unclassifiable:
        log.warning('build_frame_a: %d fills have no panel close at all -- cannot classify '
                     'held vs failed break, excluded from BOTH 1,617 and 1,618(a)', n_unclassifiable)
    held_break = classified[classified['close'] >= classified['level']].copy()
    failed_break = classified[classified['close'] < classified['level']].copy()
    log.info('build_frame_a: %d held-break candidates, %d failed-break candidates (of %d classifiable fills)',
              len(held_break), len(failed_break), len(classified))

    held_nights = set(held_break['day'].unique())
    log.info('build_frame_a: %d distinct held-break nights -> universe placebo population', len(held_nights))
    universe = base[base['day'].isin(held_nights)].copy()
    universe = attach_panel(universe, panel)

    return held_break, failed_break, universe


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------

def net_series(df):
    return df.loc[df['usable'], 'ret_bps'] - TOTAL_COST_BPS


def month_table(df):
    """Per-(year,month) mean net bps and n, on the usable subset."""
    u = df[df['usable']].copy()
    if not len(u):
        return pd.DataFrame(columns=['month', 'n', 'mean_net_bps', 'green_share'])
    u['month'] = pd.to_datetime(u['day']).dt.to_period('M').astype(str)
    net = u['ret_bps'] - TOTAL_COST_BPS
    g = pd.DataFrame({'month': u['month'], 'net': net})
    out = g.groupby('month')['net'].agg(n='count', mean_net_bps='mean',
                                          green_share=lambda s: float((s > 0).mean())).reset_index()
    return out.sort_values('month')


def score_population(df, label, holdout=None):
    """One row of the PREREG's 'Report per holdout' table for one population."""
    sub = df if holdout is None else df[df['split'] == holdout]
    y = net_series(sub)
    n = len(y)
    days = sub.loc[y.index, 'day'] if n else sub['day']
    weeks = weeks_spanned(sub['day']) if len(sub) else 1
    mtab = month_table(sub)
    months_pos = int((mtab['mean_net_bps'] > 0).sum())
    months_n = len(mtab)
    return dict(
        cell=label, holdout=holdout or 'POOLED', n_nights=n,
        mean_net_bps=float(y.mean()) if n else np.nan,
        t=day_clustered_t(y, days) if n else np.nan,
        ex_top5=ex_top_pct_mean(y, 0.05) if n else np.nan,
        ex_top1=ex_top_pct_mean(y, 0.01) if n else np.nan,
        winner_capped=winner_capped_mean(y) if n else np.nan,
        green_share=float((y > 0).mean()) if n else np.nan,
        nights_per_week=n / weeks if n else 0.0,
        months_positive=months_pos, months_total=months_n,
        n_price_scale_excl=int(sub['flag_price_scale_fail'].sum()),
        n_no_match_excl=int(sub['flag_no_panel_match'].sum() + sub['flag_no_next_open'].sum()),
        month_table=mtab,
    )


def placebo_margin(held, universe, holdout=None):
    """Day-level (held mean - universe mean) net bps, paired on the SAME calendar day.

    One row per day already (both series are collapsed by groupby('day') first), so the
    day-clustered estimator and the plain iid one-sample t coincide by construction; reported as
    a single t, labelled as such in RESULT.md.
    """
    h = held if holdout is None else held[held['split'] == holdout]
    u = universe if holdout is None else universe[universe['split'] == holdout]
    h_day = (h[h['usable']].assign(net=lambda d: d['ret_bps'] - TOTAL_COST_BPS)
             .groupby('day')['net'].mean())
    u_day = (u[u['usable']].assign(net=lambda d: d['ret_bps'] - TOTAL_COST_BPS)
             .groupby('day')['net'].mean())
    common = h_day.index.intersection(u_day.index)
    diff = (h_day.loc[common] - u_day.loc[common]).dropna()
    n = len(diff)
    if n < 2:
        return dict(n_nights=n, mean_margin_bps=np.nan, t_margin=np.nan)
    mean_margin = float(diff.mean())
    sd = float(diff.std(ddof=1))
    t_margin = float(mean_margin / (sd / np.sqrt(n))) if sd > 0 else np.nan
    return dict(n_nights=n, mean_margin_bps=mean_margin, t_margin=t_margin)


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------

def make_flags(row):
    flags = []
    if row['flag_no_panel_match']:
        flags.append('no_panel_match')
    if row['flag_no_next_open']:
        flags.append('no_next_open')
    if row['flag_price_scale_fail']:
        flags.append('price_scale_fail_30pct')
    flags.append('earnings_calendar_unavailable')  # always true on this repo -- see docstring
    if not row['usable']:
        flags.append('excluded_from_headline_stats')
    return '|'.join(flags)


def write_nights_csv(held, failed, universe, path):
    rows = []
    for df, cell in [(held, '1617'), (failed, '1618_failed'), (universe, '1618_universe')]:
        out = pd.DataFrame({
            'cell': cell,
            'split': df['split'],
            'date': df['day'],
            'symbol': df['symbol'],
            'close': df['close'],
            'next_open': df['next_open'],
            'ret_bps': df['ret_bps'],
            'flags': df.apply(make_flags, axis=1),
        })
        rows.append(out)
    full = pd.concat(rows, ignore_index=True)
    full.to_csv(path, index=False)
    log.info('wrote %s (%d rows: %d 1617 / %d 1618_failed / %d 1618_universe)',
              path, len(full), len(held), len(failed), len(universe))
    return full


def fmt(x, nd=2):
    return 'nan' if (x is None or (isinstance(x, float) and np.isnan(x))) else f'{x:.{nd}f}'


def result_table_md(rows):
    hdr = ('| population | holdout | n | mean net bps | t (day-clust) | ex-top5% | ex-top1% | '
           'winner-cap | green % | nights/wk | months pos | scale-excl | match-excl |\n'
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|\n')
    body = ''
    for r in rows:
        body += (f"| {r['cell']} | {r['holdout']} | {r['n_nights']} | {fmt(r['mean_net_bps'])} | "
                 f"{fmt(r['t'])} | {fmt(r['ex_top5'])} | {fmt(r['ex_top1'])} | {fmt(r['winner_capped'])} | "
                 f"{fmt(100*r['green_share'])} | {fmt(r['nights_per_week'],1)} | "
                 f"{r['months_positive']}/{r['months_total']} | {r['n_price_scale_excl']} | "
                 f"{r['n_no_match_excl']} |\n")
    return hdr + body


def month_tables_md(rows):
    out = ''
    for r in rows:
        out += f"\n**{r['cell']} / {r['holdout']} by month**\n\n| month | n | mean net bps | green % |\n|---|---|---|---|\n"
        for _, mr in r['month_table'].iterrows():
            out += f"| {mr['month']} | {int(mr['n'])} | {fmt(mr['mean_net_bps'])} | {fmt(100*mr['green_share'])} |\n"
    return out


def evaluate_pass_bar(val_1617, train_1617, val_margin):
    checks = []
    def add(name, ok, detail):
        checks.append((name, 'PASS' if ok else 'FAIL', detail))

    mn = val_1617['mean_net_bps']
    add('mean net bps/night >= +8 (VAL)', not np.isnan(mn) and mn >= PASS_BAR['mean_net_bps'], fmt(mn))
    t = val_1617['t']
    add('day-clustered t >= 2.5 (VAL)', not np.isnan(t) and t >= PASS_BAR['t'], fmt(t))
    ex5 = val_1617['ex_top5']
    add('ex-top-5% > 0 (VAL)', not np.isnan(ex5) and ex5 > 0, fmt(ex5))
    wc = val_1617['winner_capped']
    add('winner-capped positive (VAL)', not np.isnan(wc) and wc > 0, fmt(wc))
    nw = val_1617['nights_per_week']
    add('>= 3 nights/week (VAL)', nw >= PASS_BAR['nights_per_week'], fmt(nw, 1))
    mm, mt = val_margin['mean_margin_bps'], val_margin['t_margin']
    add('placebo margin >= +5 bps, t >= 2 (VAL)',
        not np.isnan(mm) and mm >= PASS_BAR['placebo_margin_bps'] and not np.isnan(mt) and mt >= PASS_BAR['placebo_t'],
        f"margin={fmt(mm)} t={fmt(mt)}")
    need, of = PASS_BAR['months_positive_of']
    add(f'>= {need} of {of} months positive (VAL)', val_1617['months_positive'] >= need,
        f"{val_1617['months_positive']}/{val_1617['months_total']}")
    tt = train_1617['t']
    same_sign = (not np.isnan(tt)) and (not np.isnan(t)) and (np.sign(tt) == np.sign(t))
    train_ok = same_sign and (abs(tt) >= PASS_BAR['train_t'])
    add('TRAIN-H2 same sign, t >= 1', train_ok,
        f"t={fmt(tt)} mean={fmt(train_1617['mean_net_bps'])}")
    overall = all(c[1] == 'PASS' for c in checks)
    return checks, overall


def write_result_md(path, tables_all, val_1617, train_1617, val_margin, pooled_margin,
                     n_base_fills, n_smoke, held, failed, universe):
    checks, overall = evaluate_pass_bar(val_1617, train_1617, val_margin)
    lines = []
    lines.append('# RESULT -- cells 1,617-1,618 (Frame A: held break overnight)\n')
    lines.append(f'Generated by `research/hod_entry/cell_1617.py`'
                  f'{" (--smoke, 200-fill sample)" if n_smoke else ""}. PREREG: `PREREG_1617.md` '
                  '(FROZEN 2026-09-28 17:00 UTC).\n')
    lines.append(f'\nBase HOD-break fills (status==fill): {n_base_fills:,}. '
                  f'Held-break (1,617) classifiable population: {len(held):,}. '
                  f'Failed-break (1,618a) population: {len(failed):,}. '
                  f'In-play universe on the 1,617 nights (1,618b): {len(universe):,} rows across '
                  f"{universe['day'].nunique():,} nights.\n")
    lines.append('\nCost: 5 bps per auction leg (MOC entry + MOO exit) = 10 bps round trip, '
                  'subtracted additively from the panel\'s raw next_open/close-1 return (bps), '
                  'matching this repo\'s net_R = raw_R - cost_R convention. Both the break `level` '
                  '(intraday, causal_arming_causal.csv) and the panel close/next_open '
                  '(research/overnight_high/build_panel.py: "raw close, raw next open, no '
                  'adjustment") are RAW/unadjusted -- no daily-vs-intraday price-scale mismatch.\n')
    lines.append('\n## Earnings coverage\n')
    lines.append('**0 % -- no earnings-date calendar exists in this repo.** Grepped `data_sources/` '
                  'and `research/` for `earnings_date` / `earnings_calendar` / any '
                  '`*earnings*.csv|parquet`: none found (research/multiday/data/edgar_earnings.py '
                  'existed historically but is deleted on this branch; cell_1483\'s '
                  '"earnings_guidance" is a news-headline classification, not a date list). '
                  '`AlpacaClient.get_market_calendar` (data_sources/alpaca_client.py) is a '
                  'trading-day calendar (date/open/close), not an earnings calendar. No night is '
                  'excluded for earnings. Per the PREREG\'s fallback instruction, the +-30 % '
                  'raw-price check below (run unconditionally, also the required independent '
                  'split/price-scale refuter) is reported as the disclosed substitute control.\n')
    lines.append('\n## Report per holdout\n')
    lines.append(result_table_md(tables_all))
    lines.append('\n"scale-excl" = rows dropped from headline stats for failing the +-30% raw '
                  'price check (flagged, never imputed, kept in cell_1617_nights.csv). '
                  '"match-excl" = rows with no panel (symbol,day) match or no next_open.\n')
    lines.append('\n## Placebo margin (1,617 held-break minus 1,618b universe, same nights, paired by day)\n')
    lines.append('| holdout | n nights (paired) | mean margin bps | t |\n|---|---|---|---|\n')
    lines.append(f"| VAL | {val_margin['n_nights']} | {fmt(val_margin['mean_margin_bps'])} | {fmt(val_margin['t_margin'])} |\n")
    lines.append(f"| POOLED | {pooled_margin['n_nights']} | {fmt(pooled_margin['mean_margin_bps'])} | {fmt(pooled_margin['t_margin'])} |\n")
    lines.append('\nt is the plain one-sample t on the day-level (held-mean minus universe-mean) '
                  'series; both series are already collapsed to one observation per calendar day '
                  'before differencing, so day-clustering and iid coincide by construction here.\n')
    lines.append('\n## Frozen pass bar (cell 1,617, scored on VAL; TRAIN-H2 as the secondary line)\n')
    lines.append('| check | result | value |\n|---|---|---|\n')
    for name, res, detail in checks:
        lines.append(f'| {name} | {res} | {detail} |\n')
    lines.append(f'\n**Overall: {"PASS" if overall else "FAIL"}**\n')
    lines.append('\n## Per-month detail\n')
    lines.append(month_tables_md(tables_all))
    lines.append('\n## Caveats (read as an adversary)\n')
    lines.append('- Earnings exclusion is NOT applied (0% coverage, disclosed above); any night '
                  'that happens to be an earnings reaction is still IN the primary book.\n')
    lines.append('- The universe (1,618b) includes the 1,617 fills themselves (it is the whole '
                  'in-play scanner-day population, not a disjoint control) -- by design, this is a '
                  'conservative placebo (dilutes rather than inflates any margin).\n')
    lines.append('- "not_armed" rows in the universe never got a `level`; the universe placebo uses '
                  'only the panel-derived overnight return of the NAME, independent of any level.\n')
    lines.append('- Cost is a flat 10 bps additive assumption (no measured per-trade NBBO for an '
                  'auction fill -- MOC/MOO fills do not have a quoted spread at the print); this is '
                  'a modeled cost, not a measured one, and should be treated as a lower bound on '
                  'the true cost until a live/paper MOC-MOO fill is observed.\n')
    lines.append(f'- Smoke mode: {"YES, 200-fill sample, NOT the full book -- do not act on this file" if n_smoke else "no, full book"}.\n')
    val_months = val_1617['months_total']
    if val_months != PASS_BAR['months_positive_of'][1]:
        lines.append(f"- The VAL split only spans {val_months} calendar months (2026-01..2026-05), not the "
                      f"6 the pass bar's \"{PASS_BAR['months_positive_of'][0]} of {PASS_BAR['months_positive_of'][1]}\" "
                      f"literally assumes; the checklist above tests \"{PASS_BAR['months_positive_of'][0]} of "
                      f"{val_months}\" as a result, a slightly STRICTER bar than a true 6-month window would be. "
                      f"Flagged, not silently passed through.\n")
    with open(path, 'w') as f:
        f.writelines(lines)
    log.info('wrote %s (pass bar: %s)', path, 'PASS' if overall else 'FAIL')
    return overall


# --------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='200-fill sample, fast iteration, prefixed outputs')
    args = ap.parse_args(argv)

    base = load_base()
    panel = load_panel()
    n_base_fills = int((base['status'] == 'fill').sum())

    held, failed, universe = build_frame_a(base, panel, sample_n=SMOKE_N if args.smoke else None)

    tables_all = []
    per_holdout_1617 = {}
    for holdout in HOLDOUTS:
        r = score_population(held, '1617_held_break', holdout)
        tables_all.append(r)
        per_holdout_1617[holdout] = r
    tables_all.append(score_population(held, '1617_held_break', None))
    for holdout in HOLDOUTS:
        tables_all.append(score_population(failed, '1618_failed_break', holdout))
    tables_all.append(score_population(failed, '1618_failed_break', None))
    for holdout in HOLDOUTS:
        tables_all.append(score_population(universe, '1618_universe', holdout))
    tables_all.append(score_population(universe, '1618_universe', None))

    val_margin = placebo_margin(held, universe, 'VAL')
    pooled_margin = placebo_margin(held, universe, None)
    log.info('placebo margin VAL: n=%d mean=%.2fbps t=%.2f', val_margin['n_nights'],
              val_margin['mean_margin_bps'] if not np.isnan(val_margin['mean_margin_bps']) else float('nan'),
              val_margin['t_margin'] if not np.isnan(val_margin['t_margin']) else float('nan'))

    suffix = '_SMOKE' if args.smoke else ''
    csv_path = os.path.join(HERE, f'cell_1617_nights{suffix}.csv')
    md_path = os.path.join(HERE, f'RESULT_1617{suffix}.md')
    write_nights_csv(held, failed, universe, csv_path)
    overall = write_result_md(md_path, tables_all, per_holdout_1617['VAL'], per_holdout_1617['TRAIN'],
                                val_margin, pooled_margin, n_base_fills, args.smoke, held, failed, universe)

    log.info('DONE. cell 1,617 VAL pass bar: %s', 'PASS' if overall else 'FAIL')
    return 0


if __name__ == '__main__':
    sys.exit(main())
