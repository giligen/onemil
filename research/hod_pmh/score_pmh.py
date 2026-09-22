#!/usr/bin/env python3
"""PMH-break -- score_pmh.py (PREREG_PMH.md, cells 1,389-1,392).

Reuses fill/cost/stat primitives from research/hod_exit_lab/score_cells.py by import (cost_net,
b0_style_fill, x1_no_target, day_clustered_t, ex_top5, weekly_mdd, EOD_M, SLIP_BP) -- NO DB access,
reads only the cached research/hod_pmh/{signals,paths}.parquet built by build_pmh.py.

Cells (PREREG_PMH.md):
  1389 P1  target +2R, stop, flat 15:55
  1390 P2  no target, stop, flat 15:55
  1391 P3  breakeven lock at +1R (no target), flat 15:55
  1392 P4  P1 restricted to breaks in 09:31-10:00 ET (CONDITION cell; dropped cohort reported)

Pass bar (PREREG_PMH.md "Splits, statistics, pass bar"): net mean R >= +0.10 both splits (TRAIN,
VAL), VAL day-clustered t >= 2, TRAIN halves both > 0, ex-top-5% > 0 (both splits), >= 3 fills/week
on VAL at a first-12/day 4-concurrent slot rule, cadence C3/C4 on VAL, D1/D3 placebos beaten by
>= +0.10 R (VAL).

D1/D3 placebo note (data-availability caveat, stated per CLAUDE.md "state coverage"): paths.parquet
caches ONLY each signal's own entry_m -> EOD_M bars (confirmed: cross-symbol min m ~572, i.e. no
pre-break bars are cached for any signal). A literal "uniform random intraday minute" or "any random
unrelated symbol" placebo would need a fresh bars_sip.db query; deferred. Implemented instead as:
  D1 (same name-day random minute): re-walk the SAME cell rule from a RANDOM bar already in THIS
     signal's own cached post-break path (i.e. a random later minute on the same name, same day),
     same dollar R. Biased toward the post-break window (a WEAKER placebo than a truly free minute);
     flagged in CELLS.md.
  D3 (random other name): re-walk the SAME cell rule, same dollar R, on a RANDOMLY DRAWN OTHER
     signal's cached path, entered at the first available bar with m >= this trade's own entry_m
     (so the clock-time is held fixed and only the NAME varies); skipped if no candidate other name
     has cached coverage at that clock-time. Coverage share reported.
Both use N_DRAWS seeded random draws per real trade, averaged.
"""
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
PMH = f'{ROOT}/research/hod_pmh'
LAB = f'{ROOT}/research/hod_exit_lab'
CADENCE = f'{ROOT}/scripts/cadence_bar.py'
TRADES_DIR = f'{PMH}/trades'

sys.path.insert(0, LAB)
from score_cells import cost_net, b0_style_fill, x1_no_target, day_clustered_t, ex_top5, \
    weekly_mdd, EOD_M, SLIP_BP  # noqa: E402

TRAIN_HALF_SPLIT = '2025-07-02'   # calendar midpoint of TRAIN (2025); stated convention, no prior code found
CONCURRENT_CAP = 4
DAILY_CAP = 12
N_DRAWS = 8
SEED = 1389
PASS_BAR_VAL_T = 2.0
PASS_BAR_R = 0.10
PLACEBO_BEAT_R = 0.10

# price x hour-bucket FULL spread (bps), median, from research/fuckup_audit/probe_costs.py
# COST_CURVE_BPS -- "the minute-of-day half-spread table" (CLAUDE.md: valid 09:37-14:01 only).
# Used ONLY as a sensitivity line, never as the base cost (PREREG_PMH.md "Data").
PRICE_BANDS = [(10, '$5-10'), (20, '$10-20'), (50, '$20-50'), (1e18, '$50+')]
COST_CURVE_BPS = {
    ('$5-10', '09:30-09:35'): 35.8, ('$5-10', '09:35-10:00'): 34.9, ('$5-10', '10:00-11:00'): 29.8,
    ('$5-10', '11:00-13:00'): 26.6, ('$5-10', '13:00+'): 22.5,
    ('$10-20', '09:30-09:35'): 43.8, ('$10-20', '09:35-10:00'): 42.2, ('$10-20', '10:00-11:00'): 28.8,
    ('$10-20', '11:00-13:00'): 21.7, ('$10-20', '13:00+'): 28.6,
    ('$20-50', '09:30-09:35'): 51.6, ('$20-50', '09:35-10:00'): 46.0, ('$20-50', '10:00-11:00'): 45.3,
    ('$20-50', '11:00-13:00'): 32.2, ('$20-50', '13:00+'): 34.9,
    ('$50+', '09:30-09:35'): 66.8, ('$50+', '09:35-10:00'): 63.4, ('$50+', '10:00-11:00'): 39.2,
    ('$50+', '11:00-13:00'): 30.3, ('$50+', '13:00+'): 36.4,
}


def log(msg):
    print(f'[score_pmh] {msg}', flush=True)


def week_monday(d):
    dt = datetime.strptime(d, '%Y-%m-%d').date()
    return (dt - timedelta(days=dt.weekday())).isoformat()


def half_of(day, split):
    if split != 'TRAIN':
        return None
    return 'H1' if day < TRAIN_HALF_SPLIT else 'H2'


def hour_bucket(m):
    hh, mm = divmod(int(m), 60)
    t = hh * 100 + mm
    if t < 935:
        return '09:30-09:35'
    if t < 1000:
        return '09:35-10:00'
    if t < 1100:
        return '10:00-11:00'
    if t < 1300:
        return '11:00-13:00'
    return '13:00+'


def price_band(p):
    for hi, name in PRICE_BANDS:
        if p < hi:
            return name
    return '$50+'


def band_cost_R(price, entry_m, R):
    """Sensitivity-line cost: swap measured NBBO spread for the price x hour-bucket band's FULL
    spread (bps of price), same cost_net formula (2 x half-spread + 2bp/side slip)."""
    bps = COST_CURVE_BPS[(price_band(price), hour_bucket(entry_m))]
    spread_mean_band = price * bps / 10000.0
    return spread_mean_band


def load():
    sig = pd.read_parquet(f'{PMH}/signals.parquet')
    paths = pd.read_parquet(f'{PMH}/paths.parquet')
    sig = sig[sig.split.isin(['TRAIN', 'VAL'])].copy()   # TEST sealed, never touched
    sig['wk'] = sig.day.map(week_monday)
    sig['half'] = [half_of(d, s) for d, s in zip(sig.day, sig.split)]
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    log(f'signals: TRAIN={ (sig.split=="TRAIN").sum() } VAL={ (sig.split=="VAL").sum() }')
    return sig, idx


# ------------------------------------------------------------------------------------------------
# Cell exit rules -- each fill_fn(entry, stop, R, bars) -> (exit_m, exit_price, why) or None
# ------------------------------------------------------------------------------------------------

def fill_p1(entry, stop, R, bars):
    return b0_style_fill(entry, stop, entry + 2.0 * R, bars)


def fill_p2(entry, stop, R, bars):
    return x1_no_target(entry, stop, R, bars)


def fill_p3(entry, stop, R, bars, trigger_mult=1.0, lock_mult=0.0):
    """Breakeven lock at +1R, NO target, flat 15:55."""
    cur_stop = stop
    for row in bars:
        if row.m >= EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.l <= cur_stop:
            px = row.o if row.o <= cur_stop else cur_stop
            return int(row.m), float(px), 'stop'
        if row.h >= entry + trigger_mult * R:
            cur_stop = max(cur_stop, entry + lock_mult * R)
    return None


CELL_FILL = {'1389': fill_p1, '1390': fill_p2, '1391': fill_p3, '1392': fill_p1}
CELL_NAME = {
    '1389': 'P1 target +2R, stop, 15:55',
    '1390': 'P2 no target, stop, 15:55',
    '1391': 'P3 breakeven lock at +1R, no target, 15:55',
    '1392': 'P4 = P1 restricted to breaks in 09:31-10:00 ET',
}


def walk(sig_frame, idx, fill_fn):
    """Walk every row's own cached path (entry_m -> EOD_M) with fill_fn; returns a per-trade frame."""
    rows = []
    for r in sig_frame.itertuples():
        key = (r.day, r.symbol)
        if key not in idx.index:
            continue
        g = idx.loc[[key]]
        bars = g[(g.m > r.entry_m) & (g.m <= EOD_M)]
        if bars.empty:
            continue
        res = fill_fn(r.entry, r.stop, r.R, bars.itertuples())
        if res is None:
            continue
        exit_m, exit_px, why = res
        cost_R, net_R = cost_net(r.entry, exit_px, r.R, r.spread_mean)
        band_spread = band_cost_R(r.price, r.entry_m, r.R)
        band_cost_r, band_net_R = cost_net(r.entry, exit_px, r.R, band_spread)
        rows.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, break_m=r.break_m,
                          exit_m=exit_m, exit_price=exit_px, why=why, split=r.split, half=r.half,
                          wk=r.wk, entry=r.entry, stop=r.stop, R=r.R, price=r.price,
                          spread_mean_raw=r.spread_mean,
                          net_R=net_R, cost_R=cost_R, band_net_R=band_net_R))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# D1 / D3 placebos
# ------------------------------------------------------------------------------------------------

def d1_placebo(work, idx, fill_fn, rng):
    """Same name-day, random later minute within the signal's own cached path."""
    vals = []
    for r in work.itertuples():
        key = (r.day, r.symbol)
        g = idx.loc[[key]]
        cand_m = sorted(m for m in g.m.unique() if m > r.entry_m and m < EOD_M - 5)
        if not cand_m:
            continue
        draws = []
        for _ in range(N_DRAWS):
            m_r = rng.choice(cand_m)
            row0 = g[g.m == m_r].iloc[0]
            entry_r = float(row0.o)
            stop_r = entry_r - r.R
            after = g[g.m > m_r]
            res = fill_fn(entry_r, stop_r, r.R, after.itertuples())
            if res is None:
                continue
            _, exit_px, _ = res
            _, net_R = cost_net(entry_r, exit_px, r.R, r.spread_mean_raw)
            if not pd.isna(net_R):
                draws.append(net_R)
        if draws:
            vals.append(float(np.mean(draws)))
    return vals


def d3_placebo(work, sig_all, idx, fill_fn, rng):
    """Same clock-time (entry_m), a randomly drawn OTHER signal's own name/price path."""
    others = sig_all[['day', 'symbol', 'entry_m']].drop_duplicates().values.tolist()
    vals, covered, total = [], 0, 0
    for r in work.itertuples():
        total += 1
        draws = []
        tries = 0
        while len(draws) < N_DRAWS and tries < N_DRAWS * 5:
            tries += 1
            oday, osym, oentry_m = others[rng.randrange(len(others))]
            if oday == r.day and osym == r.symbol:
                continue
            okey = (oday, osym)
            if okey not in idx.index:
                continue
            g = idx.loc[[okey]]
            avail = g[g.m >= r.entry_m]
            if avail.empty:
                continue
            row0 = avail.iloc[0]
            entry_r = float(row0.o)
            stop_r = entry_r - r.R
            after = g[g.m > row0.m]
            res = fill_fn(entry_r, stop_r, r.R, after.itertuples())
            if res is None:
                continue
            _, exit_px, _ = res
            _, net_R = cost_net(entry_r, exit_px, r.R, r.spread_mean_raw)
            if not pd.isna(net_R):
                draws.append(net_R)
        if draws:
            vals.append(float(np.mean(draws)))
            covered += 1
    return vals, covered, total


# ------------------------------------------------------------------------------------------------
# Slot simulator (first-12/day, 4-concurrent) -- research/hod_exit_lab/score_pass2.py::simulate_slots
# ------------------------------------------------------------------------------------------------

def simulate_slots(trades, concurrent_cap=CONCURRENT_CAP, daily_cap=DAILY_CAP):
    keep = pd.Series(False, index=trades.index)
    for day, g in trades.groupby('day'):
        g = g.sort_values('entry_m')
        open_exits, daily_count = [], 0
        for row in g.itertuples():
            open_exits = [x for x in open_exits if x > row.entry_m]
            if len(open_exits) < concurrent_cap and daily_count < daily_cap:
                keep.loc[row.Index] = True
                open_exits.append(row.exit_m)
                daily_count += 1
    return keep


# ------------------------------------------------------------------------------------------------
# Scorer
# ------------------------------------------------------------------------------------------------

def score_cell(cell_id, work, d1_val_mean, d3_val_mean, slot_fills_wk):
    out = dict(id=cell_id, name=CELL_NAME[cell_id])
    tr = work[work.split == 'TRAIN']
    va = work[work.split == 'VAL']
    h1 = tr[tr.half == 'H1']
    h2 = tr[tr.half == 'H2']
    out['train_n'], out['val_n'] = int(len(tr)), int(len(va))
    out['train_R'] = float(tr.net_R.mean()) if len(tr) else float('nan')
    out['val_R'] = float(va.net_R.mean()) if len(va) else float('nan')
    out['h1_R'] = float(h1.net_R.mean()) if len(h1) else float('nan')
    out['h2_R'] = float(h2.net_R.mean()) if len(h2) else float('nan')
    t, ndays = day_clustered_t(va.net_R, va.day) if len(va) else (float('nan'), 0)
    out['val_t'], out['val_t_ndays'] = t, ndays
    out['extop5_train'] = ex_top5(tr.net_R) if len(tr) else float('nan')
    out['extop5_val'] = ex_top5(va.net_R) if len(va) else float('nan')
    out['val_fills_wk'] = slot_fills_wk
    out['d1_val'] = d1_val_mean
    out['d3_val'] = d3_val_mean
    out['band_val_R'] = float(va.band_net_R.mean()) if len(va) else float('nan')

    rule_R = (out['train_R'] >= PASS_BAR_R) and (out['val_R'] >= PASS_BAR_R)
    rule_t = (not pd.isna(t)) and (t >= PASS_BAR_VAL_T)
    rule_halves = (out['h1_R'] > 0) and (out['h2_R'] > 0)
    rule_et5 = (out['extop5_train'] > 0) and (out['extop5_val'] > 0)
    rule_fills = out['val_fills_wk'] >= 3.0
    rule_d1 = (not pd.isna(d1_val_mean)) and (out['val_R'] - d1_val_mean >= PLACEBO_BEAT_R)
    rule_d3 = (not pd.isna(d3_val_mean)) and (out['val_R'] - d3_val_mean >= PLACEBO_BEAT_R)
    out['pass'] = bool(rule_R and rule_t and rule_halves and rule_et5 and rule_fills
                        and rule_d1 and rule_d3)
    out['_rules'] = dict(rule_R=bool(rule_R), rule_t=bool(rule_t), rule_halves=bool(rule_halves),
                          rule_et5=bool(rule_et5), rule_fills=bool(rule_fills),
                          rule_d1=bool(rule_d1), rule_d3=bool(rule_d3))
    out['_trades'] = work
    return out


def run_cadence(cell_id, work):
    va = work[work.split == 'VAL'].copy()
    if va.empty:
        return dict(error='no VAL trades')
    csvp = f'{TRADES_DIR}/{cell_id}.csv'
    va.assign(date=va.day, pnl_R=va.net_R).to_csv(csvp, index=False)
    try:
        p = subprocess.run(['python3', CADENCE, '--trades', csvp, '--split', 'VAL'],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
        return dict(returncode=p.returncode, stdout=p.stdout[-4000:], stderr=p.stderr[-1500:])
    except Exception as e:
        return dict(error=str(e))


def fmt(v, nd=3):
    if v is None:
        return 'n/a'
    try:
        if pd.isna(v):
            return 'n/a'
    except TypeError:
        pass
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f'{v:.{nd}f}'


def main():
    os.makedirs(TRADES_DIR, exist_ok=True)
    log('loading signals.parquet / paths.parquet (TRAIN+VAL only, TEST sealed)...')
    sig, idx = load()

    p1_work = walk(sig, idx, fill_p1)
    p2_work = walk(sig, idx, fill_p2)
    p3_work = walk(sig, idx, fill_p3)
    log(f'P1 n={len(p1_work)} P2 n={len(p2_work)} P3 n={len(p3_work)}')

    # P4 (1392): P1's own trades restricted to break_m in 09:31-10:00 ET (m 571-600)
    p4_work = p1_work[(p1_work.break_m >= 571) & (p1_work.break_m <= 600)].copy()
    p4_dropped = p1_work[~((p1_work.break_m >= 571) & (p1_work.break_m <= 600))].copy()
    log(f'P4 (1392) kept n={len(p4_work)}, dropped n={len(p4_dropped)}')

    works = {'1389': p1_work, '1390': p2_work, '1391': p3_work, '1392': p4_work}
    cell_order = {'1389': 1, '1390': 2, '1391': 3, '1392': 4}   # deterministic per-cell seed offset
    d1_cov, d3_cov = {}, {}
    results = []
    for cid, work in works.items():
        fill_fn = CELL_FILL[cid]
        va = work[work.split == 'VAL']
        rng1 = random.Random(SEED + cell_order[cid])
        rng3 = random.Random(SEED + 500 + cell_order[cid])
        log(f'{cid}: D1 placebo ({len(va)} VAL trades x {N_DRAWS} draws)...')
        d1v = d1_placebo(va, idx, fill_fn, rng1)
        log(f'{cid}: D3 placebo ({len(va)} VAL trades x up to {N_DRAWS} draws)...')
        d3v, cov, tot = d3_placebo(va, sig, idx, fill_fn, rng3)
        d1_cov[cid] = len(d1v) / len(va) if len(va) else 0.0
        d3_cov[cid] = cov / tot if tot else 0.0
        d1_mean = float(np.mean(d1v)) if d1v else float('nan')
        d3_mean = float(np.mean(d3v)) if d3v else float('nan')

        slot_keep = simulate_slots(va) if len(va) else pd.Series(dtype=bool)
        nwk = va.loc[slot_keep.index[slot_keep]].wk.nunique() if len(va) else 0
        fills_wk = float(slot_keep.sum() / nwk) if nwk else 0.0

        r = score_cell(cid, work, d1_mean, d3_mean, fills_wk)
        r['d1_coverage'] = d1_cov[cid]
        r['d3_coverage'] = d3_cov[cid]
        r['cadence'] = run_cadence(cid, work)
        results.append(r)
        work.assign(date=work.day, pnl_R=work.net_R).to_csv(f'{TRADES_DIR}/{cid}_all.csv', index=False)
        log(f'{cid}: train_R={fmt(r["train_R"])} val_R={fmt(r["val_R"])} val_t={fmt(r["val_t"])} '
            f'd1={fmt(d1_mean)} d3={fmt(d3_mean)} fills/wk={fmt(fills_wk,2)} pass={r["pass"]}')

    p4_dropped.assign(date=p4_dropped.day, pnl_R=p4_dropped.net_R).to_csv(
        f'{TRADES_DIR}/1392_dropped_cohort.csv', index=False)
    p4d_train = p4_dropped[p4_dropped.split == 'TRAIN'].net_R.mean() if len(p4_dropped[p4_dropped.split=='TRAIN']) else float('nan')
    p4d_val = p4_dropped[p4_dropped.split == 'VAL'].net_R.mean() if len(p4_dropped[p4_dropped.split=='VAL']) else float('nan')

    # ---------------- cells.json ----------------
    clean = []
    for r in results:
        c = {k: v for k, v in r.items() if not k.startswith('_')}
        c['cadence_stdout'] = c.get('cadence', {}).get('stdout', '')
        clean.append(c)
    with open(f'{PMH}/cells.json', 'w') as fh:
        json.dump(dict(cells=clean, p4_dropped_cohort=dict(train_R=float(p4d_train), val_R=float(p4d_val),
                        n_train=int(len(p4_dropped[p4_dropped.split=='TRAIN'])),
                        n_val=int(len(p4_dropped[p4_dropped.split=='VAL'])))),
                  fh, indent=2, default=str)
    log('wrote cells.json')

    # ---------------- CELLS.md ----------------
    lines = ['# CELLS.md -- PMH-break (PREREG_PMH.md, cells 1,389-1,392)', '',
             'Population: 4,810 PMH-break candidates over the HOD-universe symbol-days; TRAIN=2025 '
             '(halves split at 2025-07-02, no prior convention found in this repo for a fresh '
             'population -- stated explicitly, a deviation to flag), VAL=2026-01..05, TEST sealed '
             'and untouched. Base cost = measured half-spread at the HOD signal minute of the same '
             'name-day (100% NBBO coverage per build.log); the price x hour-bucket band cost '
             '(research/fuckup_audit/probe_costs.py COST_CURVE_BPS) is reported ONLY as a '
             'sensitivity line below, never as the pass-bar cost, per PREREG_PMH.md.', '',
             '## D1/D3 placebo caveat', '',
             'paths.parquet caches ONLY each signal\'s own entry_m -> 15:55 bars (no pre-break bars '
             'are cached for any signal -- confirmed on the raw parquet). D1 is therefore a '
             '"random LATER minute on the same name-day" (drawn from the signal\'s own cached '
             'post-break window), not a free uniform-intraday-minute placebo -- WEAKER than the '
             'PREREG\'s literal spec because the post-break window still carries some of the same '
             'move; flagged, not silently narrowed. D3 holds entry_m (clock time) fixed and swaps in '
             'a random OTHER signal\'s own name/price path from the first cached bar at or after that '
             'clock time; skipped when no candidate has coverage there (coverage % reported per '
             'cell). Both average N_DRAWS=8 seeded draws per trade.', '',
             '## Cells', '',
             '| cell | name | train_n | val_n | train_R | val_R | val_t | h1_R | h2_R | '
             'extop5_tr | extop5_val | fills/wk(VAL,4c/12d) | D1(VAL) | D3(VAL) | band_R(VAL) | pass |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        lines.append('| ' + ' | '.join([
            r['id'], r['name'], fmt(r['train_n'], 0), fmt(r['val_n'], 0), fmt(r['train_R']),
            fmt(r['val_R']), fmt(r['val_t']), fmt(r['h1_R']), fmt(r['h2_R']),
            fmt(r['extop5_train']), fmt(r['extop5_val']), fmt(r['val_fills_wk'], 2),
            fmt(r['d1_val']), fmt(r['d3_val']), fmt(r['band_val_R']), str(r['pass'])]) + ' |')
    lines += ['', '## Pass-bar rule detail per cell', '']
    for r in results:
        lines.append(f"- **{r['id']}** ({r['name']}): {r['_rules']} | D1 coverage "
                      f"{fmt(r['d1_coverage']*100 if not pd.isna(r['d1_coverage']) else float('nan'),1)}% "
                      f"| D3 coverage {fmt(r['d3_coverage']*100 if not pd.isna(r['d3_coverage']) else float('nan'),1)}%")
    lines += ['', '## 1392 (P4) dropped cohort -- breaks OUTSIDE 09:31-10:00 ET', '',
              f"train_R={fmt(p4d_train)} (n={len(p4_dropped[p4_dropped.split=='TRAIN'])}), "
              f"val_R={fmt(p4d_val)} (n={len(p4_dropped[p4_dropped.split=='VAL'])})", '',
              '## Cadence bar (scripts/cadence_bar.py --split VAL), per cell', '']
    for r in results:
        cad = r.get('cadence', {})
        lines.append(f"### {r['id']}")
        lines.append('```')
        lines.append(cad.get('stdout', cad.get('error', 'n/a')).strip() or '(no stdout)')
        lines.append('```')
    with open(f'{PMH}/CELLS.md', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log('wrote CELLS.md')
    log('ALL DONE')


if __name__ == '__main__':
    main()
