#!/usr/bin/env python3
"""
PASS 2 scorer -- HOD-break trade-level filters, entry mechanics, overnight runner, slots.
Cells 1,380-1,388 per PREREG_PASS2.md, paired against B0 exactly as PREREG.md defines it.

Reuses score_cells.py's B0 loader, fill mechanics, paired-stat helpers (day_clustered_t,
weekly_mdd, ex_top5, score_exit_cell, score_cohort_cell) and the cadence-bar driver by import
-- no logic is re-derived, only the 9 new cell mechanisms and their report-only tables.

Data sources (per the harness's own constraint: no fresh DB pass over bars_sip.db):
  - paths.parquet / signals.parquet / b0_trades.csv (pass-1 cache), via score_cells.load_data()
  - features.csv, for the 'level' field only (E1's break level; not cached in signals.parquet)
  - data/cache.db table daily_bars (symbol, bar_date, o/h/l/c/volume) for O1's day-range decision
    and the next session's 09:30 open -- point queries, PK (symbol, bar_date)
  - bars_sip.db, ONLY for O2's next-day 10:00 bar open, ONLY for the small O1-held subset, as
    point queries on the (symbol, day, t) primary key (not a table scan) -- run after 20:05 UTC.

Cells:
  T1/T2  cohort: drop r_pct < 1.5% / < 2.5%              (cost-gate filters)
  M1/M2  cohort: symbol had >=1 / net-positive prior-5-session signal(s)   (cross-day, causal)
  E1     exit-style (paired on intersection): retest entry within 15 min, else no trade
  O1/O2  exit-style (paired, selective): overnight hold on close-at-high strength
  S1/S2  cohort (slot simulator): 8-concurrent/12-day cap; 4-concurrent/20-day cap
"""
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_cells as sc

ROOT = sc.ROOT
LAB = sc.LAB
FEATURES_CSV = sc.FEATURES_CSV
CADENCE = sc.CADENCE
EOD_M = sc.EOD_M
CACHE_DB = f'{ROOT}/data/cache.db'
BARS_SIP_DB = f'{ROOT}/research/bf_zero/bars_sip.db'

OUTDIR = LAB
TRADES2 = f'{OUTDIR}/trades2'


def log(msg):
    print(f'[score_pass2] {msg}', flush=True)


# ================================================================================================
# Data loading
# ================================================================================================

def load_all():
    b0, sig, feat_np, idx, spy_open930, spy_at_m, hmm = sc.load_data(smoke=False)
    f = pd.read_csv(FEATURES_CSV, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    feat_lvl = f[['day', 'symbol', 'entry_m', 'level']].drop_duplicates(['day', 'symbol', 'entry_m'])
    sig_full = pd.read_parquet(f'{LAB}/signals.parquet')  # FULL population, all splits -- cross-day calendar
    log(f'b0={len(b0)}  sig(matched)={len(sig)}  sig_full={len(sig_full)}  feat_lvl={len(feat_lvl)}')
    return b0, sig, idx, feat_lvl, sig_full


# ================================================================================================
# T1/T2 -- cost-gate cohort cells (r_pct threshold)
# ================================================================================================

def mask_t(b0, sig_full, thr):
    m = b0.merge(sig_full[['day', 'symbol', 'entry_m', 'r_pct']], on=['day', 'symbol', 'entry_m'], how='left')
    return (m.r_pct >= thr).values, m.r_pct.values


def t_dose_curve(b0, r_pct):
    d = b0.copy()
    d['r_pct'] = r_pct
    d = d.dropna(subset=['r_pct'])
    d['decile'] = pd.qcut(d.r_pct, 10, duplicates='drop')
    tab = d.groupby('decile', observed=True).agg(mean_net_R=('net_R', 'mean'), n=('net_R', 'size'),
                                                   mean_r_pct=('r_pct', 'mean'))
    return tab.reset_index()


# ================================================================================================
# M1/M2 -- cross-day symbol-attention cohort cells
# ================================================================================================

def build_calendar(sig_full):
    days = sorted(sig_full.day.unique())
    day_idx = {d: i for i, d in enumerate(days)}
    return days, day_idx


def prior_window(day, days, day_idx, n=5):
    i = day_idx.get(day)
    if i is None:
        return set()
    lo = max(0, i - n)
    return set(days[lo:i])


def m1_prior_count(b0, sig_full, days, day_idx):
    sym_days = sig_full.groupby('symbol')['day'].apply(lambda s: set(s.unique())).to_dict()
    counts = []
    for r in b0.itertuples():
        win = prior_window(r.day, days, day_idx, 5)
        sd = sym_days.get(r.symbol, set())
        counts.append(len(win & sd))
    return pd.Series(counts, index=b0.index)


def m2_prior_net(b0, days, day_idx):
    sym_day_net = b0.groupby(['symbol', 'day']).net_R.sum().to_dict()
    sym_days_with = {}
    for (sym, day) in sym_day_net:
        sym_days_with.setdefault(sym, set()).add(day)
    totals = []
    for r in b0.itertuples():
        win = prior_window(r.day, days, day_idx, 5)
        sd = sym_days_with.get(r.symbol, set())
        hit_days = win & sd
        totals.append(sum(sym_day_net[(r.symbol, d)] for d in hit_days) if hit_days else 0.0)
    return pd.Series(totals, index=b0.index)


def m1_by_count_table(b0, prior_count):
    d = b0.copy()
    d['prior_count'] = prior_count.clip(upper=5)
    return d.groupby('prior_count').agg(mean_net_R=('net_R', 'mean'), n=('net_R', 'size')).reset_index()


# ================================================================================================
# E1 -- retest entry (changes entry price/time; own-R -> baseline-R, same convention as X11/X12)
# ================================================================================================

def find_retest(g, entry_m, level):
    window = g[(g.m > entry_m) & (g.m <= min(entry_m + 15, EOD_M))]
    if window.empty or pd.isna(level):
        return None
    hit = window[window.l <= level]
    if hit.empty:
        return None
    m_retest = int(hit.iloc[0].m)
    nxt = g[g.m == m_retest + 1]
    if nxt.empty:
        return None
    return m_retest + 1, float(nxt.iloc[0].o)


def run_e1(b0, sig, idx, feat_lvl):
    lvl = feat_lvl.set_index(['day', 'symbol', 'entry_m']).level
    sm = sig.set_index(['day', 'symbol', 'entry_m']).spread_mean
    rows, unfilled = [], []
    for r in b0.itertuples():
        key = (r.day, r.symbol, r.entry_m)
        key2 = (r.day, r.symbol)
        level = lvl.get(key, np.nan)
        if key2 not in idx.index or pd.isna(level):
            unfilled.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, split=r.split,
                                  b0_net_R=r.net_R, reason='no_level_or_path'))
            continue
        g = idx.loc[[key2]]
        res = find_retest(g, r.entry_m, level)
        if res is None:
            unfilled.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, split=r.split,
                                  b0_net_R=r.net_R, reason='no_retest_in_15min'))
            continue
        m_fill, new_entry = res
        R_new = new_entry - r.stop
        if R_new <= 0:
            unfilled.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, split=r.split,
                                  b0_net_R=r.net_R, reason='degenerate_R_new'))
            continue
        bars = g[(g.m > m_fill) & (g.m <= EOD_M)]
        if bars.empty:
            unfilled.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, split=r.split,
                                  b0_net_R=r.net_R, reason='no_bars_after_retest'))
            continue
        fillres = sc.b0_style_fill(new_entry, r.stop, r.target, bars.itertuples())
        if fillres is None:
            continue
        exit_m, exit_px, why = fillres
        spread_mean = sm.get(key, np.nan)
        cost_own, net_own = sc.cost_net(new_entry, exit_px, R_new, spread_mean)
        net_base = net_own * (R_new / r.R) if not pd.isna(net_own) else np.nan
        rows.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, retest_m=m_fill, new_entry=new_entry,
                          exit_m=exit_m, exit_price=exit_px, why=why, split=r.split, half=r.half, wk=r.wk,
                          net_R=net_base, net_R_own=net_own, R_own=R_new, b0_net_R=r.net_R))
    return pd.DataFrame(rows), pd.DataFrame(unfilled)


# ================================================================================================
# O1/O2 -- overnight runner (selective exit-style; day-t decision, day-t+1 outcome)
# ================================================================================================

def run_o1(b0, sig, conn):
    sm = sig.set_index(['day', 'symbol', 'entry_m']).spread_mean
    rows = []
    n_eod, n_hold, n_missing_daily, n_gap_anomaly = 0, 0, 0, 0
    for r in b0.itertuples():
        base = dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, split=r.split, half=r.half, wk=r.wk,
                    b0_net_R=r.net_R, held=False)
        if r.why != 'eod':
            rows.append(dict(base, net_R=r.net_R))
            continue
        n_eod += 1
        bar = conn.execute('SELECT open,high,low,close FROM daily_bars WHERE symbol=? AND bar_date=?',
                            (r.symbol, r.day)).fetchone()
        if bar is None:
            n_missing_daily += 1
            rows.append(dict(base, net_R=r.net_R))
            continue
        o, h, l, c = bar
        rng = h - l
        top20 = (c - l) >= 0.8 * rng if rng > 0 else False
        cond = (c >= r.entry + r.R) and top20
        if not cond:
            rows.append(dict(base, net_R=r.net_R))
            continue
        nxt = conn.execute('SELECT bar_date, open FROM daily_bars WHERE symbol=? AND bar_date>? '
                            'ORDER BY bar_date ASC LIMIT 1', (r.symbol, r.day)).fetchone()
        if nxt is None:
            n_missing_daily += 1
            rows.append(dict(base, net_R=r.net_R))
            continue
        next_day, next_open = nxt
        gap_days = (pd.Timestamp(next_day) - pd.Timestamp(r.day)).days
        if gap_days > 5:
            n_gap_anomaly += 1
            rows.append(dict(base, net_R=r.net_R))
            continue
        spread_mean = sm.get((r.day, r.symbol, r.entry_m), np.nan)
        cost_R, net_R = sc.cost_net(r.entry, next_open, r.R, spread_mean)
        n_hold += 1
        rows.append(dict(base, net_R=net_R, held=True, next_day=next_day, next_open=next_open,
                          day_close=c, day_high=h, day_low=l))
    df = pd.DataFrame(rows)
    log(f'  O1: eod_pop={n_eod} held={n_hold} missing_daily_bar={n_missing_daily} gap_anomaly={n_gap_anomaly}')
    return df, dict(n_eod=n_eod, n_hold=n_hold, n_missing_daily=n_missing_daily, n_gap_anomaly=n_gap_anomaly)


def run_o2(b0, sig, o1_df):
    sm = sig.set_index(['day', 'symbol', 'entry_m']).spread_mean
    held = o1_df[o1_df.held].set_index(['day', 'symbol', 'entry_m'])
    conn_sip = sqlite3.connect(BARS_SIP_DB)
    rows = []
    n_hold, n_missing_bar = 0, 0
    for r in b0.itertuples():
        key = (r.day, r.symbol, r.entry_m)
        base = dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, split=r.split, half=r.half, wk=r.wk,
                    b0_net_R=r.net_R, held=False)
        if key not in held.index:
            rows.append(dict(base, net_R=r.net_R))
            continue
        next_day = held.loc[key].next_day
        row = conn_sip.execute('SELECT o FROM bars WHERE symbol=? AND day=? AND t=?',
                                (r.symbol, next_day, '10:00')).fetchone()
        if row is None:
            n_missing_bar += 1
            rows.append(dict(base, net_R=r.net_R))
            continue
        exit_px = row[0]
        spread_mean = sm.get(key, np.nan)
        cost_R, net_R = sc.cost_net(r.entry, exit_px, r.R, spread_mean)
        n_hold += 1
        rows.append(dict(base, net_R=net_R, held=True, next_day=next_day, exit_10am=exit_px))
    conn_sip.close()
    df = pd.DataFrame(rows)
    log(f'  O2: held={n_hold} missing_10am_bar={n_missing_bar}')
    return df, dict(n_hold=n_hold, n_missing_bar=n_missing_bar)


def overnight_gap_table(o_df):
    h = o_df[o_df.held].copy()
    h['gap_dR'] = h.net_R - h.b0_net_R
    if h.empty:
        return dict(n=0, p1=None, worst_day=None, worst_symbol=None, worst_dR=None, mean=None)
    worst = h.loc[h.gap_dR.idxmin()]
    return dict(n=len(h), p1=float(h.gap_dR.quantile(0.01)), mean=float(h.gap_dR.mean()),
                worst_day=str(worst.day), worst_symbol=str(worst.symbol), worst_dR=float(worst.gap_dR))


# ================================================================================================
# S1/S2 -- slot simulator (cohort cells; overflow is random-equivalent, not quality-selected)
# ================================================================================================

def simulate_slots(b0, concurrent_cap, daily_cap):
    keep = pd.Series(False, index=b0.index)
    for day, g in b0.groupby('day'):
        g = g.sort_values('entry_m')
        open_exits = []
        daily_count = 0
        for row in g.itertuples():
            open_exits = [x for x in open_exits if x > row.entry_m]
            if len(open_exits) < concurrent_cap and daily_count < daily_cap:
                keep.loc[row.Index] = True
                open_exits.append(row.exit_m)
                daily_count += 1
    return keep


def score_s_cell(cell_id, name, b0, keep_mask, note=''):
    """Reuses score_cohort_cell for all stats, then recomputes 'pass' WITHOUT rule5 (dropped<0),
    since PREREG_PASS2 scopes the D/W dropped-cohort clause to T/M cells only -- slot overflow is
    declared random-equivalent by the cell's own mechanism, so a quality drop is not expected."""
    r = sc.score_cohort_cell(cell_id, name, b0, keep_mask, note=note)
    rule1 = (r.get('train_dR', np.nan) >= 0.10) and (r.get('val_dR', np.nan) >= 0.10) and \
            (not pd.isna(r.get('val_t_dR'))) and (r.get('val_t_dR', -99) >= 2.0)
    rule2 = (r.get('h1_dR', np.nan) > 0) and (r.get('h2_dR', np.nan) > 0)
    rule3 = r['extop5_ok']
    rule4 = r['mdd_ok']
    r['pass'] = bool(rule1 and rule2 and rule3 and rule4)
    r['note'] = (r.get('note', '') + ' | rule5 (dropped-cohort<0) NOT applied: PREREG scopes the D/W '
                 'dropped clause to T/M cells; slot overflow is random-equivalent by mechanism.').strip(' |')
    return r


def weekly_p10(kept_val):
    wk_sum = kept_val.groupby('wk').net_R.sum()
    if len(wk_sum) == 0:
        return None
    return float(wk_sum.quantile(0.10))


# ================================================================================================
# Main
# ================================================================================================

def main():
    os.makedirs(TRADES2, exist_ok=True)
    b0, sig, idx, feat_lvl, sig_full = load_all()
    results = []
    report_tables = {}

    # ---------------- T1/T2 ----------------
    log('T1/T2: r_pct cost-gate cohort cells...')
    t1_mask, r_pct_vals = mask_t(b0, sig_full, 1.5)
    r = sc.score_cohort_cell('T1', "drop r_pct < 1.5% (cost-gate)", b0, t1_mask)
    results.append(r)
    r['_kept'].assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{TRADES2}/T1.csv', index=False)

    t2_mask, _ = mask_t(b0, sig_full, 2.5)
    r = sc.score_cohort_cell('T2', "drop r_pct < 2.5% (cost-gate, dose check)", b0, t2_mask)
    results.append(r)
    r['_kept'].assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{TRADES2}/T2.csv', index=False)

    report_tables['T_dose_curve_by_r_pct_decile'] = t_dose_curve(b0, r_pct_vals).to_dict('records')

    # ---------------- M1/M2 ----------------
    log('M1/M2: cross-day symbol-attention cohort cells...')
    days, day_idx = build_calendar(sig_full)
    prior_count = m1_prior_count(b0, sig_full, days, day_idx)
    m1_mask = (prior_count >= 1).values
    r = sc.score_cohort_cell('M1', 'symbol had >=1 HOD-break signal in prior 5 sessions (any outcome)', b0, m1_mask)
    results.append(r)
    r['_kept'].assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{TRADES2}/M1.csv', index=False)

    prior_net = m2_prior_net(b0, days, day_idx)
    m2_mask = (prior_net > 0).values
    r = sc.score_cohort_cell('M2', "symbol's prior-5-session signals net POSITIVE under B0", b0, m2_mask)
    results.append(r)
    r['_kept'].assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{TRADES2}/M2.csv', index=False)

    report_tables['M1_by_prior_signal_count'] = m1_by_count_table(b0, prior_count).to_dict('records')

    # ---------------- E1 ----------------
    log('E1: retest entry (path re-walk, own-R -> baseline-R)...')
    e1_work, e1_unfilled = run_e1(b0, sig, idx, feat_lvl)
    if e1_work.empty:
        results.append(dict(id='E1', name='retest entry within 15 min, else no trade', kind='exit',
                             **{'pass': False}, note='EMPTY (no retests found)'))
    else:
        r = sc.score_exit_cell('E1', 'retest entry within 15 min, else no trade', e1_work)
        results.append(r)
        e1_work.assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{TRADES2}/E1.csv', index=False)
    e1_unfilled.to_csv(f'{TRADES2}/E1_unfilled.csv', index=False)

    unf_tab = {}
    for split in ('TRAIN', 'VAL'):
        u = e1_unfilled[e1_unfilled.split == split]
        unf_tab[split] = dict(n=len(u), mean_b0_net_R=float(u.b0_net_R.mean()) if len(u) else None)
    fb = {}
    for split in ('TRAIN', 'VAL'):
        eb = e1_work[e1_work.split == split] if not e1_work.empty else e1_work
        bb = b0[b0.split == split]
        fb[split] = dict(e1_n=len(eb), e1_mean_net_R=float(eb.net_R.mean()) if len(eb) else None,
                          b0_n=len(bb), b0_mean_net_R=float(bb.net_R.mean()) if len(bb) else None)
    report_tables['E1_unfilled_counterfactual'] = unf_tab
    report_tables['E1_full_book_unpaired'] = fb

    # ---------------- O1/O2 ----------------
    log('O1/O2: overnight runner...')
    conn_daily = sqlite3.connect(CACHE_DB)
    o1_work, o1_stats = run_o1(b0, sig, conn_daily)
    conn_daily.close()
    r = sc.score_exit_cell('O1', 'overnight hold on close-at-high strength (else B0)', o1_work,
                            note=f"eod_pop={o1_stats['n_eod']} held={o1_stats['n_hold']} "
                                 f"missing_daily_bar={o1_stats['n_missing_daily']} "
                                 f"gap_anomaly={o1_stats['n_gap_anomaly']} (fell back to B0)")
    results.append(r)
    o1_work.assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{TRADES2}/O1.csv', index=False)
    report_tables['O1_overnight_gap'] = overnight_gap_table(o1_work)

    log('O2: next-day 10:00 open (bars_sip.db point queries, O1-held subset only)...')
    o2_work, o2_stats = run_o2(b0, sig, o1_work)
    r = sc.score_exit_cell('O2', 'O1 with exit at next day 10:00 open instead of 09:30', o2_work,
                            note=f"held={o2_stats['n_hold']} missing_10am_bar={o2_stats['n_missing_bar']} "
                                 f"(fell back to B0)")
    results.append(r)
    o2_work.assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{TRADES2}/O2.csv', index=False)
    report_tables['O2_overnight_gap'] = overnight_gap_table(o2_work)

    # ---------------- S1/S2 ----------------
    log('S1/S2: slot simulator (concurrent cap / daily cap)...')
    s1_mask = simulate_slots(b0, concurrent_cap=8, daily_cap=12)
    r = score_s_cell('S1', '8 concurrent slots instead of 4 (same first-12/day cap)', b0, s1_mask)
    results.append(r)
    r['_kept'].assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{TRADES2}/S1.csv', index=False)

    s2_mask = simulate_slots(b0, concurrent_cap=4, daily_cap=20)
    r = score_s_cell('S2', 'daily cap 20 instead of 12, 4 concurrent (same as baseline)', b0, s2_mask)
    results.append(r)
    r['_kept'].assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{TRADES2}/S2.csv', index=False)

    s_tab = {}
    for cid, mask in (('S1', s1_mask), ('S2', s2_mask)):
        kept_val = b0[(b0.split == 'VAL') & mask.values]
        s_tab[cid] = dict(val_fills_wk=float(len(kept_val) / kept_val.wk.nunique()) if kept_val.wk.nunique() else 0.0,
                           weekly_p10=weekly_p10(kept_val))
    report_tables['S_fills_and_weekly_P10'] = s_tab

    # ---------------- B0 reference for cadence ----------------
    b0.assign(date=b0.day, pnl_R=b0.net_R).to_csv(f'{TRADES2}/B0.csv', index=False)

    # ---------------- cadence bar (B0 + every passing cell) ----------------
    log('running cadence_bar.py on B0 and every passing cell (VAL split)...')
    cadence = {}
    to_run = ['B0'] + [r['id'] for r in results if r.get('pass')]
    for cid in to_run:
        csvp = f'{TRADES2}/{cid}.csv'
        if not os.path.exists(csvp):
            continue
        try:
            p = subprocess.run(['python3', CADENCE, '--trades', csvp, '--split', 'VAL'],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
            cadence[cid] = dict(returncode=p.returncode, stdout=p.stdout[-4000:], stderr=p.stderr[-2000:])
            log(f'  cadence[{cid}]: rc={p.returncode}')
        except Exception as e:
            cadence[cid] = dict(error=str(e))
            log(f'  cadence[{cid}]: ERROR {e}')

    # ---------------- write cells_pass2.json ----------------
    clean = []
    for r in results:
        rr = {k: v for k, v in r.items() if not k.startswith('_')}
        for k, v in list(rr.items()):
            if isinstance(v, (np.floating, np.integer)):
                rr[k] = float(v) if not pd.isna(v) else None
            elif isinstance(v, bool):
                rr[k] = bool(v)
            elif isinstance(v, float) and pd.isna(v):
                rr[k] = None
        clean.append(rr)
    with open(f'{OUTDIR}/cells_pass2.json', 'w') as fh:
        json.dump(dict(cells=clean, cadence=cadence, report_tables=report_tables), fh, indent=2, default=str)
    log(f'wrote {OUTDIR}/cells_pass2.json')

    write_md(results, cadence, report_tables, b0)
    log(f'wrote {OUTDIR}/CELLS_PASS2.md')
    log('DONE')


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 'n/a'
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f'{v:.{nd}f}'


def write_md(results, cadence, report_tables, b0):
    lines = []
    lines.append('# CELLS PASS 2 -- HOD-break trade-level filters, entry mechanics, overnight runner, slots')
    lines.append('')
    lines.append('Cells 1,380-1,388 per `PREREG_PASS2.md`. Scored via `score_pass2.py`, importing B0/paired-stat/'
                  'cadence helpers from `score_cells.py` (pass 1) unchanged. Programme count after this pass: 1,388.')
    lines.append('')
    lines.append(f'B0 population: {len(b0)} trades (TRAIN={(b0.split=="TRAIN").sum()}, VAL={(b0.split=="VAL").sum()}).'
                 f' B0 TRAIN mean net R = {b0[b0.split=="TRAIN"].net_R.mean():.4f}, '
                 f'B0 VAL mean net R = {b0[b0.split=="VAL"].net_R.mean():.4f}.')
    lines.append('')
    lines.append('## Cells')
    lines.append('| id | name | kind | train_dR | val_dR | val_t_dR | h1_dR | h2_dR | extop5_ok | mdd_ok | '
                  'val_fills_wk | PASS |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in results:
        lines.append(f"| {r.get('id')} | {r.get('name','')[:60]} | {r.get('kind','')} | "
                      f"{fmt(r.get('train_dR'))} | {fmt(r.get('val_dR'))} | {fmt(r.get('val_t_dR'),2)} | "
                      f"{fmt(r.get('h1_dR'))} | {fmt(r.get('h2_dR'))} | {r.get('extop5_ok')} | {r.get('mdd_ok')} | "
                      f"{fmt(r.get('val_fills_wk'),2)} | {'**PASS**' if r.get('pass') else 'fail'} |")
    lines.append('')
    for r in results:
        if r.get('note'):
            lines.append(f"- **{r.get('id')}** note: {r['note']}")
    lines.append('')

    lines.append('## Report-only: T1/T2 dose curve by r_pct decile (B0, all trades)')
    lines.append('| decile (r_pct range) | mean r_pct | mean net_R | n |')
    lines.append('|---|---|---|---|')
    for row in report_tables.get('T_dose_curve_by_r_pct_decile', []):
        lines.append(f"| {row['decile']} | {fmt(row['mean_r_pct'],2)} | {fmt(row['mean_net_R'])} | {row['n']} |")
    lines.append('')

    lines.append('## Report-only: M1 by count of prior-5-session signals (B0 net_R)')
    lines.append('| prior_count (capped 5) | mean net_R | n |')
    lines.append('|---|---|---|')
    for row in report_tables.get('M1_by_prior_signal_count', []):
        lines.append(f"| {row['prior_count']} | {fmt(row['mean_net_R'])} | {row['n']} |")
    lines.append('')

    lines.append('## Report-only: E1 unfilled-counterfactual cohort (B0 outcome of signals that never retested)')
    for split, d in report_tables.get('E1_unfilled_counterfactual', {}).items():
        lines.append(f"- {split}: n={d['n']}, mean B0 net_R={fmt(d['mean_b0_net_R'])}")
    lines.append('')
    lines.append('## Report-only: E1 full-book comparison (retest book vs B0 book, unpaired)')
    for split, d in report_tables.get('E1_full_book_unpaired', {}).items():
        lines.append(f"- {split}: E1 n={d['e1_n']} mean={fmt(d['e1_mean_net_R'])}  |  "
                      f"B0 n={d['b0_n']} mean={fmt(d['b0_mean_net_R'])}")
    lines.append('')

    for cid in ('O1', 'O2'):
        g = report_tables.get(f'{cid}_overnight_gap', {})
        lines.append(f'## Report-only: {cid} overnight gap distribution (held subset, gap dR = variant - B0)')
        lines.append(f"- n held = {g.get('n')}, mean dR = {fmt(g.get('mean'))}, P1 = {fmt(g.get('p1'))}, "
                      f"worst night = {g.get('worst_symbol')} {g.get('worst_day')} dR={fmt(g.get('worst_dR'))}")
        lines.append('')

    lines.append('## Report-only: S1/S2 fills/week and weekly P10 (VAL, kept cohort)')
    for cid, d in report_tables.get('S_fills_and_weekly_P10', {}).items():
        lines.append(f"- {cid}: fills/week={fmt(d['val_fills_wk'],2)}, weekly P10={fmt(d['weekly_p10'])}")
    lines.append('')

    lines.append('## Cadence bar (scripts/cadence_bar.py --split VAL), B0 + passing cells')
    for cid, c in cadence.items():
        lines.append(f'### {cid}')
        lines.append('```')
        lines.append(c.get('stdout', c.get('error', ''))[-1500:])
        lines.append('```')

    with open(f'{OUTDIR}/CELLS_PASS2.md', 'w') as fh:
        fh.write('\n'.join(lines))


if __name__ == '__main__':
    main()
