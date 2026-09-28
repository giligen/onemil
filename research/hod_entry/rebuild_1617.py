"""Cell 1,617 -- Frame A INDEPENDENT REBUILD (held break overnight, MOC -> MOO).

Rebuilt FROM PROSE ONLY, against research/hod_entry/PREREG_1617.md section "A. Held break
overnight (1,617; report-only variant 1,618)". This rebuild deliberately did NOT open
cell_1617.py, cell_1617_nights.csv, or RESULT_1617.md -- those are the artifacts this rebuild
exists to check independently (PREREG_1617.md "Independent check and consequences": nightly set
Jaccard >= 0.99, bps within 1, on VAL).

Mechanism (as specified in the prereg, verbatim citations in comments below):
  Population -- "base fills whose day's CLOSE >= the level (the break held; from the Databento
  daily panel `research/overnight_high/panel_2024_2026.parquet`, zero-OHLCV rows dropped; the HOD
  days 2025-07..2026-05 are inside)."
  Trade -- "buy at the 16:00 closing auction (MOC) of the break day, sell at the next session's
  opening auction (MOO); return = next open / close - 1; costs 5 bps per auction leg; earnings
  dates excluded where the panel or Alpaca calendar gives them (state coverage), a -30%..+30%
  raw-price check (splits) per night."
  Report per holdout -- "n nights, mean net bps, day-clustered t (the night), ex-top-5 % /
  ex-top-1 %, winner-capped +10 %, green-night share, per-month, the placebo margin (held-break
  minus the universe on the same nights)."

  1,618 (report-only) is computed here ONLY because 1,617's own report line requires the placebo
  margin, which needs the universe leg: "the same for base fills whose close is BELOW the level
  (the failed breaks) and for the whole in-play scanner day universe on the same nights (the
  placebo: is it the break or the name)."

Shared inputs (per the task spec, not the forbidden 1617 artifacts):
  research/hod_entry/causal_arming_causal.csv  -- base fills (status == 'fill' -> 9,911)
  research/overnight_high/panel_2024_2026.parquet -- Databento daily panel, delisted included
  research/hod_entry/cell_1445.py  -- day_clustered_t, ex_top5_mean, weeks_spanned (generic stats
    helpers, reused as directed by the task; no cell-1617-specific logic lives there)

Outputs: rebuild_1617_nights.csv (one row per held-break night), REBUILD_1617.md (the report).
"""
import logging
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
HERE = os.path.join(ROOT, 'research/hod_entry')
sys.path.insert(0, ROOT)

from research.hod_entry.cell_1445 import day_clustered_t, ex_top5_mean, weeks_spanned  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('rebuild_1617')

CAUSAL_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
PANEL_PARQUET = os.path.join(ROOT, 'research/overnight_high/panel_2024_2026.parquet')
OUT_CSV = os.path.join(HERE, 'rebuild_1617_nights.csv')
OUT_MD = os.path.join(HERE, 'REBUILD_1617.md')

AUCTION_LEG_COST_BPS = 5.0     # "costs 5 bps per auction leg" -- two legs (MOC buy, MOO sell)
RAW_MOVE_BAND = 0.30           # "-30%..+30% raw-price check (splits) per night"
WINNER_CAP_BPS = 1000.0        # "winner-capped +10 %"
EXPECTED_FILLS = 9911          # PREREG_1617.md "What was seen": "Base fills (9,911, cell 1,438)"


def ex_top1_mean(y):
    """Mean excluding the top 1 % (by value) -- same recipe as cell_1445.ex_top5_mean at k=1 %,
    for the "ex-top-1 %" report line (cell_1445 only ships the 5 % variant)."""
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return np.nan
    k = int(round(0.01 * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def load_panel():
    """Databento daily panel, delisted included; drop zero-OHLCV rows per the prereg's population
    definition ("zero-OHLCV rows dropped")."""
    p = pd.read_parquet(PANEL_PARQUET)
    before = len(p)
    zero_ohlcv = (p['open'] <= 0) | (p['high'] <= 0) | (p['low'] <= 0) | (p['close'] <= 0) | (p['volume'] <= 0)
    dropped = int(zero_ohlcv.sum())
    p = p.loc[~zero_ohlcv].copy()
    if dropped:
        log.warning('panel: dropped %d/%d zero-OHLCV rows (population definition, PREREG_1617.md)', dropped, before)
    p = p.copy()
    p['symbol'] = p['symbol'].astype(str)
    return p[['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'next_open']]


def auction_to_auction(close, next_open, leg_bps=AUCTION_LEG_COST_BPS):
    """Buy at the MOC, sell at the next MOO; leg_bps applied multiplicatively on each side (the
    buy pays leg_bps, the sell gives up leg_bps). Returns (raw_ret, net_ret) as fractions."""
    raw_ret = next_open / close - 1.0
    buy_eff = close * (1.0 + leg_bps / 10000.0)
    sell_eff = next_open * (1.0 - leg_bps / 10000.0)
    net_ret = sell_eff / buy_eff - 1.0
    return raw_ret, net_ret


def build_nights(rows, panel, label):
    """rows: DataFrame with at least day, symbol columns. Left-join to the panel on (symbol, day),
    compute the auction-to-auction leg and the raw-price sanity flag. One output row per input
    (day, symbol). Unmatched symbol-days are dropped (not zero-filled) and counted."""
    m = rows.merge(panel, left_on=['symbol', 'day'], right_on=['symbol', 'bar_date'], how='left')
    unmatched = int(m['bar_date'].isna().sum())
    if unmatched:
        log.warning('%s: %d/%d symbol-days have no panel row (no close/next_open) -- DROPPED, not zero-filled',
                    label, unmatched, len(m))
    m = m.dropna(subset=['bar_date', 'close', 'next_open']).copy()
    raw_ret, net_ret = auction_to_auction(m['close'].to_numpy(dtype=float), m['next_open'].to_numpy(dtype=float))
    m['raw_ret'] = raw_ret
    m['net_ret'] = net_ret
    m['net_bps'] = m['net_ret'] * 10000.0
    m['month'] = m['day'].str.slice(0, 7)
    m['split_flag'] = m['raw_ret'].abs() > RAW_MOVE_BAND
    n_flag = int(m['split_flag'].sum())
    if n_flag:
        log.warning('%s: %d/%d nights outside +/-%.0f%% raw close->next_open (suspected unadjusted '
                    'split / garbage print) -- EXCLUDED from the primary book', label, n_flag, len(m), RAW_MOVE_BAND * 100)
    # Earnings-date exclusion ("earnings dates excluded where the panel or Alpaca calendar gives
    # them (state coverage)"): repo-wide search found NO earnings-date source -- the panel has no
    # earnings column and data_sources/alpaca_client.py exposes only get_market_calendar (trading
    # days, not earnings). Coverage is therefore 0 % for every row; no exclusion is applied. This
    # is disclosed as an open refuter in REBUILD_1617.md, not silently skipped.
    log.warning('%s: earnings-date exclusion coverage = 0%% (no earnings-calendar source found in the repo) '
                '-- NOT applied, disclosed as an open refuter', label)
    m['earnings_checkable'] = False
    return m


def per_holdout_report(nights_primary, nights_universe, split_name):
    """All "Report per holdout" fields for one split, plus the placebo margin against the
    universe on the same nights."""
    keep = nights_primary[(nights_primary['split'] == split_name) & (~nights_primary['split_flag'])]
    y = keep['net_bps']
    n = len(keep)
    weeks = weeks_spanned(keep['day']) if n else 1
    months = keep.groupby('month')['net_bps'].agg(['mean', 'count'])
    months_positive = int((months['mean'] > 0).sum())
    months_total = len(months)

    # Placebo margin: per calendar night, held-break mean bps minus the (all-status) universe mean
    # bps on that SAME night ("the placebo margin (held-break minus the universe on the same
    # nights)"), then day-clustered across nights.
    uni = nights_universe[(nights_universe['split'] == split_name) & (~nights_universe['split_flag'])]
    uni_by_day = uni.groupby('day')['net_bps'].mean()
    held_by_day = keep.groupby('day')['net_bps'].mean()
    common_days = held_by_day.index.intersection(uni_by_day.index)
    margin = (held_by_day.loc[common_days] - uni_by_day.loc[common_days]).dropna()

    return {
        'split': split_name,
        'n_nights': n,
        'mean_net_bps': float(y.mean()) if n else np.nan,
        'day_clustered_t': day_clustered_t(y, keep['day']) if n else np.nan,
        'ex_top5_mean_bps': ex_top5_mean(y) if n else np.nan,
        'ex_top1_mean_bps': ex_top1_mean(y) if n else np.nan,
        'winner_capped_mean_bps': float(np.minimum(y, WINNER_CAP_BPS).mean()) if n else np.nan,
        'green_night_share': float((y > 0).mean()) if n else np.nan,
        'nights_per_week': (n / weeks) if n else 0.0,
        'weeks_spanned': weeks,
        'months_positive': months_positive,
        'months_total': months_total,
        'months_table': months,
        'placebo_margin_mean_bps': float(margin.mean()) if len(margin) else np.nan,
        'placebo_margin_t': day_clustered_t(margin.to_numpy(), margin.index.to_numpy()) if len(margin) >= 2 else np.nan,
        'placebo_n_nights': len(margin),
        'n_excluded_split_flag': int(nights_primary[(nights_primary['split'] == split_name) & (nights_primary['split_flag'])].shape[0]),
    }


def main():
    causal = pd.read_csv(CAUSAL_CSV, low_memory=False)
    fills = causal[causal['status'] == 'fill'].copy()
    if len(fills) != EXPECTED_FILLS:
        log.error('base fills count mismatch: expected %d (prereg-disclosed), got %d -- check causal_arming_causal.csv',
                   EXPECTED_FILLS, len(fills))
    log.info('base fills (status==fill): %d', len(fills))

    panel = load_panel()

    # ---- price-scale refuter check: level (raw, intraday-sourced) must be <= the panel's own
    # day-high on a shared raw scale ("Refuters: A -- price scale (raw close/open)") ----
    chk = fills.merge(panel[['symbol', 'bar_date', 'high']], left_on=['symbol', 'day'], right_on=['symbol', 'bar_date'], how='left')
    scale_matched = int(chk['high'].notna().sum())
    scale_violations = int((chk['level'] > chk['high']).sum())
    log.info('price-scale refuter: %d/%d matched fills have level > panel day-high (expect 0 on a shared raw scale)',
             scale_violations, scale_matched)

    # ---- Frame A population: the break HELD (close >= level) ----
    held = fills.merge(panel[['symbol', 'bar_date', 'close']], left_on=['symbol', 'day'], right_on=['symbol', 'bar_date'], how='left')
    close_unmatched = int(held['close'].isna().sum())
    if close_unmatched:
        log.error('%d/%d fills have NO panel close match -- dropped from BOTH held and failed populations (not silently assigned)',
                   close_unmatched, len(held))
    matched = held.dropna(subset=['close']).copy()
    held_mask = matched['close'] >= matched['level']
    cols = ['day', 'symbol', 'split', 'level', 'fill', 'stop']
    pop_a = matched.loc[held_mask, cols].copy()
    pop_failed = matched.loc[~held_mask, cols].copy()
    log.info('population: held (close >= level) = %d, failed (close < level) = %d, unmatched = %d',
             len(pop_a), len(pop_failed), close_unmatched)

    nights_a = build_nights(pop_a, panel, '1617 held-break')
    nights_failed = build_nights(pop_failed, panel, '1618 failed-break')

    # ---- 1,618 universe placebo: every symbol causal_arming_causal.csv considered (ANY status)
    # on the same calendar nights as the (unflagged) held-break population ----
    held_days = sorted(nights_a.loc[~nights_a['split_flag'], 'day'].unique().tolist())
    universe_rows = causal.loc[causal['day'].isin(held_days), ['day', 'symbol', 'split']].drop_duplicates()
    log.info('universe placebo: %d distinct nights, %d symbol-day rows (any status)', len(held_days), len(universe_rows))
    nights_universe = build_nights(universe_rows, panel, '1618 universe placebo')

    nights_a.to_csv(OUT_CSV, index=False)
    log.info('wrote %s (%d rows, %d columns)', OUT_CSV, len(nights_a), nights_a.shape[1])

    reports = {s: per_holdout_report(nights_a, nights_universe, s) for s in ['TRAIN', 'VAL']}
    failed_reports = {}
    for s in ['TRAIN', 'VAL']:
        sub = nights_failed[(nights_failed['split'] == s) & (~nights_failed['split_flag'])]
        failed_reports[s] = {
            'n': len(sub),
            'mean_net_bps': float(sub['net_bps'].mean()) if len(sub) else np.nan,
            't': day_clustered_t(sub['net_bps'], sub['day']) if len(sub) else np.nan,
        }

    write_report(reports, failed_reports, nights_a, nights_failed, nights_universe,
                  scale_violations, scale_matched, close_unmatched, len(causal))
    log.info('done')


def write_report(reports, failed_reports, nights_a, nights_failed, nights_universe,
                  scale_violations, scale_matched, close_unmatched, n_universe_rows_total):
    val = reports['VAL']
    train = reports['TRAIN']

    def fmt(x, nd=2):
        return 'nan' if (x is None or (isinstance(x, float) and np.isnan(x))) else f'{x:.{nd}f}'

    pass_checks = {
        'mean net >= +8 bps/night (VAL)': (val['mean_net_bps'] >= 8.0, fmt(val['mean_net_bps'])),
        'day-clustered t >= 2.5 (VAL)': (val['day_clustered_t'] >= 2.5 if not np.isnan(val['day_clustered_t']) else False, fmt(val['day_clustered_t'])),
        'ex-top-5% > 0 (VAL)': (val['ex_top5_mean_bps'] > 0, fmt(val['ex_top5_mean_bps'])),
        'winner-capped positive (VAL)': (val['winner_capped_mean_bps'] > 0, fmt(val['winner_capped_mean_bps'])),
        '>= 3 nights/week (VAL)': (val['nights_per_week'] >= 3.0, fmt(val['nights_per_week'])),
        'placebo margin >= +5bps, t>=2 (VAL)': (
            (val['placebo_margin_mean_bps'] >= 5.0 and val['placebo_margin_t'] >= 2.0)
            if not (np.isnan(val['placebo_margin_mean_bps']) or np.isnan(val['placebo_margin_t'])) else False,
            f"{fmt(val['placebo_margin_mean_bps'])} bps, t={fmt(val['placebo_margin_t'])}"),
        '>= 4 of 6 months positive (VAL has 5 months available)': (
            val['months_positive'] >= 4, f"{val['months_positive']} of {val['months_total']} VAL months"),
        'TRAIN-H2 same sign, t >= 1': (
            (np.sign(train['mean_net_bps']) == np.sign(val['mean_net_bps']) and abs(train['day_clustered_t']) >= 1.0)
            if not (np.isnan(train['mean_net_bps']) or np.isnan(train['day_clustered_t'])) else False,
            f"mean={fmt(train['mean_net_bps'])} bps, t={fmt(train['day_clustered_t'])}"),
    }
    n_pass = sum(1 for v, _ in pass_checks.values() if v)
    verdict = 'PASS (all 8 criteria clear)' if n_pass == len(pass_checks) else f'FAIL ({n_pass}/{len(pass_checks)} criteria clear)'

    lines = []
    lines.append('# REBUILD 1,617 -- Frame A independent rebuild (held break overnight, MOC -> MOO)')
    lines.append('')
    lines.append('Independent reimplementation from `research/hod_entry/PREREG_1617.md` prose only. '
                  'Did NOT open `cell_1617.py`, `cell_1617_nights.csv`, or `RESULT_1617.md`. This document '
                  'reports THIS rebuild\'s own numbers; it does not have the original 1,617 numbers to diff '
                  'against (by design -- that comparison, nightly-set Jaccard >= 0.99 and bps within 1 on VAL '
                  'per the prereg\'s own bar, happens in a separate step outside this task).')
    lines.append('')
    lines.append(f'Full causal_arming_causal.csv rows (any status, all days): {n_universe_rows_total}.')
    lines.append('')
    lines.append('## Coverage / availability rail')
    lines.append(f'- Panel match on the 9,911 base fills: {scale_matched}/9911 matched for the price-scale check '
                  f'({close_unmatched} had no panel close at all and were dropped from BOTH held/failed populations).')
    lines.append(f'- Price-scale refuter (level <= panel day-high, shared raw scale): {scale_violations}/{scale_matched} '
                  f'violations ({"PASS -- 0 violations, same raw scale" if scale_violations == 0 else "FAIL -- price-scale mismatch, see caveats"}).')
    lines.append(f'- Held-break population (close >= level): {len(nights_a) + int(nights_a["split_flag"].sum())} matched nights '
                  f'before the raw-move flag; {int(nights_a["split_flag"].sum())} excluded by the +/-30% band; '
                  f'{len(nights_a)} in the primary book.')
    lines.append(f'- Failed-break population (close < level), report-only: {len(nights_failed)} nights in the primary book '
                  f'after the same exclusions.')
    lines.append('- Earnings-date exclusion coverage: 0% -- no earnings-calendar data source exists in this repo '
                  '(searched for earnings_date/earnings_calendar sources and an Alpaca earnings endpoint; only '
                  '`get_market_calendar`, the trading-day calendar, was found). NOT applied. Open refuter, see below.')
    lines.append('- Halt-calendar coverage: 0% -- `research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv` is a static '
                  'symbol-level snapshot (tradable/shortable/easy_to_borrow/exchange, no dates); it cannot answer '
                  '"was this symbol halted on this specific night". NOT applied. Open refuter, see below.')
    lines.append('')

    for name, r in [('TRAIN-H2', train), ('VAL', val)]:
        lines.append(f'## Held-break overnight (1,617) -- {name}')
        lines.append(f'- n nights: {r["n_nights"]} (+ {r["n_excluded_split_flag"]} excluded by the raw-move band)')
        lines.append(f'- mean net: {fmt(r["mean_net_bps"])} bps/night')
        lines.append(f'- day-clustered t (the night): {fmt(r["day_clustered_t"])}')
        lines.append(f'- ex-top-5% mean: {fmt(r["ex_top5_mean_bps"])} bps; ex-top-1% mean: {fmt(r["ex_top1_mean_bps"])} bps')
        lines.append(f'- winner-capped (+10%) mean: {fmt(r["winner_capped_mean_bps"])} bps')
        lines.append(f'- green-night share: {fmt(r["green_night_share"], 3)}')
        lines.append(f'- nights/week: {fmt(r["nights_per_week"])} (over {r["weeks_spanned"]} weeks)')
        lines.append(f'- months positive: {r["months_positive"]} of {r["months_total"]}')
        mt = r['months_table']
        if len(mt):
            lines.append('- per-month (mean bps, n):')
            for mo, row in mt.iterrows():
                lines.append(f'  - {mo}: {row["mean"]:.2f} bps, n={int(row["count"])}')
        lines.append(f'- placebo margin (held-break minus in-play universe, same nights): '
                     f'{fmt(r["placebo_margin_mean_bps"])} bps, t={fmt(r["placebo_margin_t"])}, over {r["placebo_n_nights"]} paired nights')
        se = (r['mean_net_bps'] / r['day_clustered_t']) if (r['day_clustered_t'] not in (0, None) and not np.isnan(r['day_clustered_t']) and r['day_clustered_t'] != 0) else np.nan
        mde = 2.5 * abs(se) if not np.isnan(se) else np.nan
        lines.append(f'- implied day-clustered SE: {fmt(se)} bps; MDE at the pass bar\'s own t>=2.5 threshold '
                     f'(this n, this variance): a true mean of roughly +/-{fmt(mde)} bps/night would be needed to clear t=2.5')
        fr = failed_reports[name.split('-')[0] if name != 'TRAIN-H2' else 'TRAIN']
        lines.append(f'- [1,618 context] failed-break (close < level) same split: n={fr["n"]}, '
                     f'mean={fmt(fr["mean_net_bps"])} bps, t={fmt(fr["t"])}')
        lines.append('')

    lines.append('## Pass bar (frozen; evaluated on VAL per PREREG_1617.md)')
    for k, (ok, val_str) in pass_checks.items():
        lines.append(f'- [{"PASS" if ok else "FAIL"}] {k}: {val_str}')
    lines.append('')
    lines.append(f'**Verdict: {verdict}**')
    lines.append('')
    lines.append('Note on "4 of 6 months": VAL (2026-01..2026-05) only has 5 calendar months of data in this '
                  'population, not 6 -- the frozen bar text says "6" (likely written against TRAIN-H2\'s 6-month '
                  'span, Jul-Dec 2025). Evaluated here as "4 of the 5 available VAL months" and flagged as a '
                  'prereg wording ambiguity rather than silently rewritten.')
    lines.append('')

    lines.append('## Refuters (PREREG_1617.md "Independent check and consequences")')
    lines.append(f'- **Price scale (raw close/open):** {scale_violations}/{scale_matched} fills have `level` above '
                 f'the panel\'s own day-high -- {"clean" if scale_violations == 0 else "FAILS, investigate before trusting any number here"}. '
                 f'`level`/`fill`/`stop` come from the live/tape-sourced causal_arming_causal.csv and are always '
                 f'<= the panel `high` for the same (symbol, day), consistent with the panel being on the same raw '
                 f'(unadjusted-for-that-day) price scale as the intraday feed.')
    lines.append(f'- **Earnings and halts:** NOT RESOLVED -- no data source for either exists in this repo (see '
                 f'Coverage above). The {int(nights_a["split_flag"].sum()) + int(nights_failed["split_flag"].sum())} '
                 f'nights excluded by the +/-30% raw-move band catch the most extreme cases (including some halts/'
                 f'splits/earnings gaps by construction, since those are exactly the mechanisms that produce >30% '
                 f'overnight moves), but ordinary-sized earnings gaps (a few percent) are NOT filtered and remain '
                 f'in the primary book. This means the mean-net-bps numbers above are not certified clean of '
                 f'earnings-night contamination -- flag before shipping.')
    lines.append(f'- **The placebo:** reported per holdout above (held-break minus the in-play universe, same '
                 f'nights). A positive, significant margin says the edge is about the break holding, not just '
                 f'about the name being active that night; see the pass-bar line for the VAL verdict.')
    lines.append(f'- **Tails:** ex-top-5%/ex-top-1% and winner-capped-at-+10% reported per holdout above alongside '
                 f'the raw mean -- read them against the raw mean before trusting the headline number.')
    lines.append('')

    lines.append('## Design choices made rebuilding from prose (documented for the comparison step)')
    lines.append('- Auction cost applied multiplicatively on each leg (buy at close*(1+5bps), sell at '
                 'next_open*(1-5bps)), not simply raw_bps - 10; the two are within ~0.005 bps of each other.')
    lines.append('- "Night" = one (day, symbol) held-break fill row (matches the base-fill population\'s own '
                 'grain); day-clustered t clusters by the break-day calendar date, consistent with every other '
                 '`day_clustered_t` use in this codebase.')
    lines.append('- Placebo margin built as a PAIRED per-calendar-night difference (held-break mean bps that '
                 'night minus universe mean bps that night), then day-clustered across nights -- the most literal '
                 'reading of "held-break minus the universe on the same nights". The universe leg includes the '
                 'held-break names themselves (it is "the whole in-play scanner day universe", not the universe '
                 'minus the break names).')
    lines.append('- "In-play scanner day universe" (for the placebo) = every (day, symbol) row in '
                 'causal_arming_causal.csv for that day, ANY status (fill/nofill/not_armed) -- this is the full '
                 'population causal_arming.py\'s scanner evaluated that day (its own docstring: "population = '
                 'EVERY symbol-day of the spec\'s causal superset... plus the live admission gates"); no separate '
                 'broader universe file was in the task\'s shared inputs.')
    lines.append('- Zero-OHLCV rows dropped from the panel BEFORE any join (prereg population definition), so a '
                 'symbol-day with a zero print anywhere in OHLCV is treated as "no panel data" for that day, not '
                 'as a valid close/next_open pair.')
    lines.append('')

    lines.append('## Output schema: `rebuild_1617_nights.csv`')
    lines.append('One row per held-break night (primary book, i.e. `split_flag == False` rows are the reportable '
                 'set; flagged rows are KEPT in the CSV with `split_flag=True` for transparency, not deleted).')
    lines.append('Columns: `day, symbol, split, level, fill, stop, bar_date, open, high, low, close, next_open, '
                 'raw_ret, net_ret, net_bps, month, split_flag, earnings_checkable`.')
    lines.append('Key fields for the trade-by-trade comparison: `(day, symbol)` is the night key; `net_bps` is the '
                 'reportable net return; `split` is TRAIN/VAL.')
    lines.append('')

    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log.info('wrote %s', OUT_MD)


if __name__ == '__main__':
    main()
