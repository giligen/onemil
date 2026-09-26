#!/usr/bin/env python3
"""Cells I1 (CLUSTER), I2 (OFFICER), I3 (OPPORTUNISTIC) -- Form 4 insider open-market
purchases. Spec: research/insider_events/PREREG.md. Reuses research/multiday machinery
for parity by construction: Panel (research/multiday/run_final.py) for prices/eligibility/
taint, and the cost + trade-return + book_sim helpers of research/multiday/run_f2_a1.py.

Design decisions not fully pinned down by the PREREG prose (documented here so the
independent-check agent can audit them without reading this file):

  * I1 cluster window "10 sessions ending on this filing date": computed on F = the last
    panel session <= the row's FILING_DATE (not TRANS_DATE), window = [F-9, F] inclusive,
    summed/counted over PRIOR + CURRENT filings of the SAME symbol only (never a future
    filing -- causal by construction since F is monotone with filing_date).
  * I2 "flagged Officer or Director with a title containing CEO/CFO/President/Chair/
    Director": RPTOWNER_RELATIONSHIP is a flag set (Officer/Director/TenPercentOwner/
    Other); RPTOWNER_TITLE is free text, blank for plain directors. Implemented as:
    (is_officer or is_director) AND (relationship+title text contains one of the five
    keywords, case-insensitive) -- a bare Director always matches on "Director"; a bare
    "Officer" with no CEO/CFO/President/Chair title does NOT qualify.
  * I3 "no purchase of that issuer in the prior 12 months": per (owner_cik, symbol),
    look back from TRANS_DATE (not filing_date -- this is a trading-history definition,
    Cohen-Malloy-Pomorski) at that owner+issuer's OWN prior purchase rows; opportunistic
    if none in the preceding 365 days. The owner's purchase HISTORY used for this lookback
    is the full row set (no separate causality issue: we are only ever comparing a row's
    trans_date against earlier trans_dates of the same owner+issuer).
  * Signal entry: a = S (the signal session itself). The signal is defined as knowable at
    the CLOSE of session S (PREREG timestamp rule), and PREREG's own rationale says fills
    are closing-auction prints, so entry is the closing print of session S itself (no extra
    T+1 skip -- unlike F2, this signal is not an intraday/same-day-of-print event).
  * No-double-slot dedup: per symbol, candidates sorted by S; a candidate is dropped if its
    S falls before the previous KEPT signal's S + hold (hold=20, the primary/pass-bar hold,
    used for ALL cells' spacing including I1's 5/60 report-only hold variants -- i.e. one
    dedup pass per cell, hold length changes only the EXIT, never which signals fire).
  * Test-ticker belt-and-suspenders: drop symbols matching ^Z[A-Z]ZZT$ (CLAUDE.md
    red-to-green exclusion pattern) even though the panel is already PIT-curated.
  * Book tie-break within a session: book_sim() ranks candidates competing for the same
    entry session by `value` descending (larger disclosed purchase / cluster value wins
    the slot first) -- an explicit, arbitrary-but-documented choice; PREREG does not specify one.
"""
from __future__ import annotations

import os
import re
import sys
import time
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, '/home/ec2-user/onemil/research/multiday')
from run_final import Panel  # noqa: E402
from run_f2_a1 import (round_trip_cost_bps, trade_returns, book_sim,  # noqa: E402
                        daily_series, monthly, BOOK_USD, N_SLOTS)

HERE = os.path.dirname(os.path.abspath(__file__))
D = f'{HERE}/data'

SPLITS = {'TRAIN': ('2016-01-01', '2021-12-31'),
          'VAL': ('2022-01-01', '2023-12-31'),
          'TEST': ('2024-01-01', '2026-09-18')}
SURV_WINDOW = ('2018-01-01', '2023-12-31')
HOLD_PRIMARY = 20
I1_HOLDS = (5, 20, 60)
NULL_DRAWS = 1000
NULL_SEED = 1474
TEST_TICKER_RE = re.compile(r'^Z[A-Z]ZZT$')


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


# ------------------------------------------------------------------ load / gate


def load_purchases(p: Panel, pur: pd.DataFrame = None) -> pd.DataFrame:
    """Load purchases.parquet (or use `pur` directly, for tests), map symbol->panel index,
    compute F (last session <= filing_date) and S (first session strictly after filing_date).
    Logs every drop."""
    if pur is None:
        pur = pd.read_parquet(f'{D}/purchases.parquet')
    else:
        pur = pur.copy()
    n0 = len(pur)
    pur['si'] = pur['symbol'].map(p.sidx)
    n_unmapped = int(pur['si'].isna().sum())
    log(f'purchases raw {n0:,}; unmapped to panel symbol {n_unmapped:,} '
        f'({n_unmapped / n0:.2%})')
    pur = pur[pur['si'].notna()].copy()
    pur['si'] = pur['si'].astype(int)

    test_mask = pur['symbol'].str.match(TEST_TICKER_RE)
    n_test = int(test_mask.sum())
    if n_test:
        log(f'dropping {n_test:,} rows matching test-ticker pattern ^Z[A-Z]ZZT$')
    pur = pur[~test_mask]

    sess = p.sess_np
    fd = pur['filing_date'].values
    S = np.searchsorted(sess, fd, side='right')
    F = S - 1
    n_oob = int(((S < 0) | (S >= p.n_d) | (F < 0)).sum())
    pur['S'] = S
    pur['F'] = F
    pur = pur[(pur['S'] >= 0) & (pur['S'] < p.n_d) & (pur['F'] >= 0)].copy()
    log(f'out-of-panel-range filing dates dropped {n_oob:,}; remaining {len(pur):,}')
    return pur.reset_index(drop=True)


def gate_and_dedup(cand: pd.DataFrame, p: Panel, hold: int) -> pd.DataFrame:
    """Apply elig/taint/bounds gating at the signal session, then the no-double-slot
    dedup (per symbol, spacing = hold sessions)."""
    n0 = len(cand)
    si = cand['si'].to_numpy()
    S = cand['S'].to_numpy()
    elig_ok = p.elig[si, S]
    taint_bad = p.taint[si, S]
    keep = elig_ok & ~taint_bad
    n_inelig = int((~elig_ok).sum())
    n_taint = int(taint_bad.sum())
    cand = cand[keep].copy()
    log(f'  gating: {n0:,} candidates -> ineligible at signal session {n_inelig:,}, '
        f'tainted {n_taint:,} -> {len(cand):,} survive')

    cand = cand.sort_values(['symbol', 'S', 'value'], ascending=[True, True, False])
    keep_idx = []
    last_end = {}
    for row in cand.itertuples():
        sym = row.symbol
        if sym not in last_end or row.S >= last_end[sym]:
            keep_idx.append(row.Index)
            last_end[sym] = row.S + hold
    n_before = len(cand)
    cand = cand.loc[keep_idx].sort_values('S').reset_index(drop=True)
    log(f'  no-double-slot dedup (hold={hold}): {n_before:,} -> {len(cand):,}')
    return cand


# ------------------------------------------------------------------ I1 / I2 / I3 candidates


def build_i1(pur: pd.DataFrame) -> pd.DataFrame:
    """Cluster: >=2 distinct owners of the same issuer filed purchases within the 10
    sessions ending on this filing's session (F), total value >= $100K."""
    rows = []
    for sym, g in pur.groupby('symbol', sort=False):
        g = g.sort_values('F')
        F_arr = g['F'].to_numpy()
        own_arr = g['owner_cik'].to_numpy()
        val_arr = g['value'].to_numpy()
        S_arr = g['S'].to_numpy()
        si_arr = g['si'].to_numpy()
        lo = np.searchsorted(F_arr, F_arr - 9, side='left')
        for i in range(len(g)):
            lo_i = lo[i]
            owners = set(own_arr[lo_i:i + 1])
            if len(owners) >= 2:
                tot_val = float(val_arr[lo_i:i + 1].sum())
                if tot_val >= 100_000.0:
                    rows.append((sym, int(si_arr[i]), int(S_arr[i]), tot_val, len(owners)))
    cand = pd.DataFrame(rows, columns=['symbol', 'si', 'S', 'value', 'n_owners'])
    cand = cand.drop_duplicates(['symbol', 'S'])
    log(f'I1 raw cluster-qualifying rows (pre-gate): {len(cand):,}')
    return cand


def build_i2(pur: pd.DataFrame) -> pd.DataFrame:
    """Officer/Director single purchase >= $50K, title containing a C-suite/board keyword."""
    text = (pur['owner_relationship'].fillna('') + ' ' + pur['owner_title'].fillna('')).str.upper()
    kw = text.str.contains('CEO|CFO|PRESIDENT|CHAIR|DIRECTOR', regex=True, na=False)
    rel_ok = pur['is_officer'] | pur['is_director']
    mask = rel_ok & kw & (pur['value'] >= 50_000.0)
    cand = pur.loc[mask, ['symbol', 'si', 'S', 'value']].copy()
    log(f'I2 raw officer-qualifying rows (pre-gate): {len(cand):,}')
    return cand


def build_i3(pur: pd.DataFrame) -> pd.DataFrame:
    """Opportunistic: no purchase of the SAME issuer by the SAME owner in the prior 365
    days (trans_date lookback, Cohen-Malloy-Pomorski non-routine definition)."""
    n0 = len(pur)
    pur = pur[pur['trans_date'].notna()]
    n_bad = n0 - len(pur)
    if n_bad:
        log(f'I3: dropping {n_bad:,} rows with unparseable TRANS_DATE before the lookback')
    g = pur.sort_values(['owner_cik', 'symbol', 'trans_date']).copy()
    prev = g.groupby(['owner_cik', 'symbol'])['trans_date'].shift(1)
    gap = (g['trans_date'] - prev).dt.days
    opportunistic = prev.isna() | (gap > 365)
    mask = opportunistic & (g['value'] >= 25_000.0)
    cand = g.loc[mask, ['symbol', 'si', 'S', 'value']].copy()
    log(f'I3 raw opportunistic-qualifying rows (pre-gate): {len(cand):,}')
    return cand


# ------------------------------------------------------------------ trades


def make_trades(p: Panel, sig: pd.DataFrame, hold: int) -> pd.DataFrame:
    """One row per (symbol, S) signal: entry a=S, exit b=S+hold, gross/net return, costs."""
    si = sig['si'].to_numpy()
    a = sig['S'].to_numpy()
    b = a + hold
    ok = b < p.n_d
    n_oob = int((~ok).sum())
    t = sig[ok].copy()
    si, a, b = si[ok], a[ok], b[ok]
    t['a'] = a
    t['b'] = b
    t['gross'] = trade_returns(p, si, a, b)
    adv_in = p.adv[si, a]
    adv_out = p.adv[si, np.minimum(b, p.n_d - 1)]
    t['cost_bps'] = round_trip_cost_bps(adv_in, np.where(np.isfinite(adv_out), adv_out, adv_in))
    t['net'] = t['gross'] - t['cost_bps'] / 1e4
    t['entry_date'] = p.sessions[a]
    t['exit_date'] = p.sessions[b]
    t['hold'] = hold
    if n_oob:
        log(f'  hold={hold}: dropped {n_oob:,} signals with no room to exit (panel end)')
    return t


def tag_split(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t['split'] = 'EXCLUDED'
    for name, (a, b) in SPLITS.items():
        m = (t['entry_date'] >= a) & (t['entry_date'] <= b)
        t.loc[m, 'split'] = name
    return t


# ------------------------------------------------------------------ null


def count_matched_null(p: Panel, sig: pd.DataFrame, hold: int, split: str,
                        n_draws=NULL_DRAWS, seed=NULL_SEED) -> np.ndarray:
    """1,000 draws: for every real signal, replace its symbol with a random ELIGIBLE,
    non-tainted symbol at the SAME session (same date, same n per session). Returns the
    per-draw mean net return array (length n_draws)."""
    rng = np.random.default_rng(seed)
    a0, b0 = SPLITS[split]
    sub = sig[(p.sessions[sig['S'].to_numpy()] >= a0) & (p.sessions[sig['S'].to_numpy()] <= b0)]
    if len(sub) == 0:
        return np.array([])
    S_vals = sub['S'].to_numpy()
    uniq_S, counts = np.unique(S_vals, return_counts=True)
    draw_cols = []  # each: (n_draws, count_at_this_S) array of net returns
    for s, c in zip(uniq_S, counts):
        elig_syms = np.where(p.elig[:, s] & ~p.taint[:, s])[0]
        if len(elig_syms) == 0:
            continue
        picks = rng.choice(elig_syms, size=(n_draws, c), replace=True)
        flat_si = picks.reshape(-1)
        a_arr = np.full(len(flat_si), s)
        b_arr = a_arr + hold
        oob = b_arr >= p.n_d
        if oob.all():
            continue
        gross = np.full(len(flat_si), np.nan)
        gross[~oob] = trade_returns(p, flat_si[~oob], a_arr[~oob], b_arr[~oob])
        adv_in = p.adv[flat_si, a_arr]
        adv_out = p.adv[flat_si, np.minimum(b_arr, p.n_d - 1)]
        cost = round_trip_cost_bps(adv_in, np.where(np.isfinite(adv_out), adv_out, adv_in))
        net = gross - cost / 1e4
        draw_cols.append(net.reshape(n_draws, c))
    if not draw_cols:
        return np.array([])
    allnet = np.concatenate(draw_cols, axis=1)
    return np.nanmean(allnet, axis=1)


# ------------------------------------------------------------------ scoring


def score_split(t_all_holds: pd.DataFrame, hold: int, split: str, p: Panel,
                 sig_for_null: pd.DataFrame, compute_null: bool) -> dict:
    """One (cell already selected upstream, hold, split) row of the pass-bar table."""
    t = t_all_holds[(t_all_holds['hold'] == hold) & (t_all_holds['split'] == split)].copy()
    n = len(t)
    out = dict(split=split, n_trades=n)
    if n < 5:
        out.update(mean_net_pct=np.nan, t_day_clustered=np.nan, months_positive_share=np.nan,
                   ex_top5_mean_pct=np.nan, signals_per_week=np.nan, null_pctile=np.nan,
                   book_monthly_pct=np.nan, book_max_dd_pct=np.nan)
        return out

    net_pct = t['net'].to_numpy() * 100.0
    out['mean_net_pct'] = float(net_pct.mean())

    X = sm.add_constant(np.ones(n))
    try:
        m = sm.OLS(net_pct, X).fit(cov_type='cluster', cov_kwds={'groups': t['a'].to_numpy()})
        out['t_day_clustered'] = float(m.tvalues[0])
    except Exception as e:
        log(f'  WARNING clustered OLS failed ({e}); falling back to iid t')
        out['t_day_clustered'] = float(net_pct.mean() / (net_pct.std(ddof=1) / np.sqrt(n)))

    cut = np.quantile(net_pct, 0.95)
    ex5 = net_pct[net_pct < cut]
    out['ex_top5_mean_pct'] = float(ex5.mean()) if len(ex5) else np.nan

    a0, b0 = SPLITS[split]
    n_weeks = (pd.Timestamp(b0) - pd.Timestamp(a0)).days / 7.0
    out['signals_per_week'] = n / n_weeks

    all_s, _ = daily_series(p, t.assign(si=t['si']))
    m_all = monthly(all_s)
    m_all = m_all[(m_all.index >= a0) & (m_all.index <= b0)]
    m_all = m_all[m_all != 0]
    out['months_positive_share'] = float((m_all > 0).mean()) if len(m_all) else np.nan

    book_t = book_sim(p, t, rank_col='value', ascending=False)
    bs, _ = daily_series(p, book_t.assign(si=book_t['si']))
    m_book = monthly(bs)
    m_book = m_book[(m_book.index >= a0) & (m_book.index <= b0)]
    if len(m_book):
        out['book_monthly_pct'] = float(m_book.mean() * 100.0)
        eq = (1 + m_book).cumprod()
        out['book_max_dd_pct'] = float((eq / eq.cummax() - 1).min() * 100.0)
    else:
        out['book_monthly_pct'] = np.nan
        out['book_max_dd_pct'] = np.nan

    if compute_null and split == 'VAL':
        null_means = count_matched_null(p, sig_for_null, hold, split)
        if len(null_means):
            out['null_pctile'] = float(100.0 * (null_means <= out['mean_net_pct'] / 100.0).mean())
        else:
            out['null_pctile'] = np.nan
    else:
        out['null_pctile'] = np.nan
    return out


def apply_pass_bar(row: dict, split: str, is_primary_hold: bool) -> bool:
    if not is_primary_hold or split not in ('TRAIN', 'VAL'):
        return False
    if any(np.isnan(row.get(k, np.nan)) for k in ('mean_net_pct', 't_day_clustered')):
        return False
    if split == 'TRAIN':
        return (row['mean_net_pct'] >= 0.8 and row['t_day_clustered'] >= 2.5
                and row.get('months_positive_share', 0) >= 0.55)
    return (row['mean_net_pct'] >= 0.5 and row['t_day_clustered'] >= 2.0
            and row.get('ex_top5_mean_pct', -1) > 0 and row.get('signals_per_week', 0) >= 3
            and row.get('book_monthly_pct', -99) >= 1.0 and row.get('book_max_dd_pct', -99) <= 15
            and row.get('null_pctile', 0) >= 99)


# ------------------------------------------------------------------ main


def process_cell(cell: str, sig_raw: pd.DataFrame, p: Panel, holds, results: list,
                  all_trades: list, all_signals: list):
    sig = gate_and_dedup(sig_raw, p, HOLD_PRIMARY)
    sig_out = sig.copy()
    sig_out['cell'] = cell
    sig_out['signal_date'] = p.sessions[sig['S'].to_numpy()]
    all_signals.append(sig_out)

    trades_holds = []
    for hold in holds:
        th = make_trades(p, sig, hold)
        th = tag_split(th)
        trades_holds.append(th)
    t_all = pd.concat(trades_holds, ignore_index=True) if trades_holds else pd.DataFrame()
    if len(t_all):
        t_all_out = t_all.copy()
        t_all_out['cell'] = cell
        t_all_out['arm'] = 'all'
        all_trades.append(t_all_out)

    for hold in holds:
        is_primary = hold == HOLD_PRIMARY
        for split in ('TRAIN', 'VAL'):
            row = score_split(t_all, hold, split, p, sig, compute_null=is_primary)
            row.update(cell=cell, arm='all', hold=hold)
            row['passes_bar'] = apply_pass_bar(row, split, is_primary)
            results.append(row)

        # survivorship arm: only meaningful for the primary hold, sub-window 2018-23.
        # TWO independent delisting flags are used, per the task's "delisted_names.parquet
        # AND the panel's price availability":
        #   (a) named: symbol appears in research/multiday/data/delisted_names.parquet
        #       (built from the XNAS tape for the multiday universe -- ZERO of our Form4
        #       signal symbols matched this list; see RESULT_FORM4.md, this arm is VOID).
        #   (b) price-availability proxy: the panel's OWN last finite close for that symbol
        #       (`Panel.last_fin`) is more than 60 sessions before the panel's last session
        #       -- i.e. the name stopped printing prices well before "today" in this data,
        #       a direct, always-populated delisting/data-disappearance proxy.
        if is_primary:
            delisted = pd.read_parquet(f'{D}/../../multiday/data/delisted_names.parquet',
                                        columns=['symbol'])
            named_delisted_set = set(delisted['symbol'])
            base_window = t_all[(t_all['hold'] == hold)
                                 & (t_all['entry_date'] >= SURV_WINDOW[0])
                                 & (t_all['entry_date'] <= SURV_WINDOW[1])].copy()
            n_named_overlap = int(base_window['symbol'].isin(named_delisted_set).sum())
            log(f'  {cell} survivorship: named delisted_names.parquet overlap with '
                f'2018-23 signal symbols = {n_named_overlap} of {len(base_window)} rows')
            price_gone = p.last_fin[base_window['si'].to_numpy()] < (p.n_d - 1 - 60)
            n_price_gone = int(price_gone.sum())
            log(f'  {cell} survivorship: price-availability-gone (last close >60 sessions '
                f'before panel end) = {n_price_gone} of {len(base_window)} rows')

            for arm_name, keep_mask in (
                ('2018-23 delisted-included (named)', np.ones(len(base_window), dtype=bool)),
                ('2018-23 survivors-only (named)',
                 ~base_window['symbol'].isin(named_delisted_set).to_numpy()),
                ('2018-23 delisted-included (price-avail)', np.ones(len(base_window), dtype=bool)),
                ('2018-23 survivors-only (price-avail)', ~price_gone)):
                tt = base_window[keep_mask].copy()
                tt = tag_split(tt)
                for split in ('TRAIN', 'VAL'):
                    ttx = tt[tt['split'] == split]
                    if len(ttx) < 5:
                        row = dict(cell=cell, arm=arm_name, hold=hold, split=split, n_trades=len(ttx),
                                   mean_net_pct=np.nan, t_day_clustered=np.nan, passes_bar=False,
                                   note='n<5')
                        results.append(row)
                        continue
                    net_pct = ttx['net'].to_numpy() * 100.0
                    X = sm.add_constant(np.ones(len(ttx)))
                    try:
                        m = sm.OLS(net_pct, X).fit(cov_type='cluster',
                                                    cov_kwds={'groups': ttx['a'].to_numpy()})
                        tstat = float(m.tvalues[0])
                    except Exception:
                        tstat = float(net_pct.mean() / (net_pct.std(ddof=1) / np.sqrt(len(ttx))))
                    row = dict(cell=cell, arm=arm_name, hold=hold, split=split, n_trades=len(ttx),
                               mean_net_pct=float(net_pct.mean()), t_day_clustered=tstat,
                               passes_bar=False, note='survivorship arm, report-only vs primary arm')
                    results.append(row)


def main():
    t0 = time.time()
    p = Panel()
    pur = load_purchases(p)

    log('=== building I1 CLUSTER candidates ===')
    i1_raw = build_i1(pur)
    log('=== building I2 OFFICER candidates ===')
    i2_raw = build_i2(pur)
    log('=== building I3 OPPORTUNISTIC candidates ===')
    i3_raw = build_i3(pur)

    results, all_trades, all_signals = [], [], []
    process_cell('I1', i1_raw, p, I1_HOLDS, results, all_trades, all_signals)
    process_cell('I2', i2_raw, p, (HOLD_PRIMARY,), results, all_trades, all_signals)
    process_cell('I3', i3_raw, p, (HOLD_PRIMARY,), results, all_trades, all_signals)

    signals_df = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    signals_df.to_csv(f'{HERE}/signals_form4.csv', index=False)
    trades_df.to_csv(f'{HERE}/trades_form4.csv', index=False)
    with open(f'{HERE}/data/results_table.json', 'w') as f:
        json.dump(results, f, default=str, indent=1)
    log(f'wrote signals_form4.csv ({len(signals_df):,} rows), trades_form4.csv '
        f'({len(trades_df):,} rows incl. TEST), results_table.json ({len(results)} rows)')
    log(f'DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
