#!/usr/bin/env python3
"""CAUSAL_FILTER — PREREG_FAILED_BREAK_SHORT.md, cells 1,357-1,358.

Short the failed HOD break. Population: every HOD-break LONG signal row of features.csv
(day, symbol, entry_m = the long signal/entry minute, L = the `entry` column = the long entry
price, `stop`/`level` unused here). Failure = first 1-min bar with close < L within K=10 minutes
strictly after entry_m. Entry = SHORT at the open of the bar after the failing bar. Stop H =
max(high) over bars from entry_m through the failing bar (touched -> fill at H; gap-through bar
opens above H -> fill at that open). R = H - entry, must be >= 0.3% of price else no trade.

Cell 1357: cover at the open of the 15:55 bar (flat_minute=955).
Cell 1358: cover at entry - 2R when a bar's low <= that target (fill at target; gap below -> that
open), else the 15:55 open.

Cost: this study's own per-signal NBBO (nbbo.csv, `spread_mean` = the MEAN full bid-ask spread at
the signal minute entry_m-1, used as the proxy for the short's cost on both legs, matching the
PREREG's own instruction) + 2bp slippage on a stop or target fill (not on an eod fill, which is a
clean market print). net_R = (entry - exit)/R - spread_mean/R - slip_R.

Shortable proxy: price >= $5 and adv20 >= 1,000,000 shares. SSR proxy: exclude rows whose `price`
(the features.csv signal-time price) is <= 90% of `prev_close` (uptick rule in force). Both shares
reported against the TRAIN+VAL population (TEST is never touched -- PREREG "Not allowed").

Placebo decomposition (diagnostics, not cells), both using the SAME generic, signal-free recipe
(stop = the prior 10-minute high, no failure-detection) so both isolate one axis of "is this just
shorting gappers":
  D1 (time-shuffle, same symbol):  entry at a random minute in [10:00, 14:00) on the SAME symbol-day,
      seeded by (day, symbol). Stop = max(high) over the 10 minutes strictly before that minute.
  D3 (symbol-shuffle, same day):   entry at the REAL trade's own entry_m, on a random OTHER symbol
      that also fired a signal that day, seeded by (day, symbol). Stop = max(high) over the 10
      minutes strictly before entry_m on that other symbol's bars.
Both walk the SAME cell's exit rule and are charged the SAME NBBO proxy as the real trade they are
attached to (no separate NBBO fetch for the placebo instants -- out of scope, noted as a limitation).

Resumable is not needed (one sqlite pass, ~15-20 min); writes trades CSVs + FAILED_BREAK_SHORT_REPORT.md.
Reads bars_sip.db ONLY by (symbol, day) -- the indexed primary-key prefix -- never a full scan.
"""
import json
import os
import random
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

D = f'{ROOT}/research/bf_zero/causal_filter'
DB = f'{ROOT}/research/bf_zero/bars_sip.db'
ET = ZoneInfo('America/New_York')

K = 10                    # minutes to look for a failure after entry_m
FLAT_MINUTE = 955         # 15:55 ET
TARGET_R = 2.0            # cell 1358
MIN_R_PCT = 0.003         # 0.3% of price
SPREAD_SLIP_BP = 0.0002   # 2bp, charged on a stop or target fill only
SHORTABLE_PRICE = 5.0
SHORTABLE_ADV20 = 1_000_000
SSR_RATIO = 0.90
NW = {'TRAIN': 53, 'VAL': 23}   # weeks per split (causal_filter programme convention, cells.py)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ---------------------------------------------------------------- bars access (indexed symbol+day)
_BARS_CACHE = {}


def get_bars(conn, symbol, day):
    """All RTH 1-min bars for (symbol, day), minute = ET minutes-since-midnight, sorted. Cached."""
    key = (symbol, day)
    if key in _BARS_CACHE:
        return _BARS_CACHE[key]
    cur = conn.execute('SELECT t, o, h, l, c FROM bars WHERE symbol=? AND day=? ORDER BY t', (symbol, day))
    rows = cur.fetchall()
    if not rows:
        _BARS_CACHE[key] = None
        return None
    mins, o, h, l, c = [], [], [], [], []
    for t, oo, hh, ll, cc in rows:
        et = datetime.fromisoformat(t).astimezone(ET)
        m = et.hour * 60 + et.minute
        if m < 570 or m >= 960:      # RTH only, 09:30-16:00
            continue
        mins.append(m); o.append(oo); h.append(hh); l.append(ll); c.append(cc)
    if not mins:
        _BARS_CACHE[key] = None
        return None
    arr = dict(m=np.array(mins), o=np.array(o), h=np.array(h), l=np.array(l), c=np.array(c))
    _BARS_CACHE[key] = arr
    return arr


def first_after(bars, minute):
    """Index of the first bar with m > minute, or None."""
    idx = np.searchsorted(bars['m'], minute, side='right')
    return idx if idx < len(bars['m']) else None


def walk_exit(bars, start_idx, entry_price, H, use_target):
    """From bars[start_idx] forward: eod at 15:55 open; stop on high>=H (gap-through at open);
    (cell 1358 only) target = entry-2R on low<=target (gap-through at open). Stop wins a same-bar tie
    over target (mirrors trading/hod_break.py's walk_exit convention)."""
    R = H - entry_price
    target = entry_price - TARGET_R * R
    m, o, h, l = bars['m'], bars['o'], bars['h'], bars['l']
    for k in range(start_idx, len(m)):
        if m[k] >= FLAT_MINUTE:
            return o[k], 'eod'
        if h[k] >= H:
            return (o[k] if o[k] > H else H), 'stop'
        if use_target and l[k] <= target:
            return (o[k] if o[k] < target else target), 'target'
    return bars['c'][-1], 'eod'   # day ran out without a flat bar (rare, last close)


def find_failure(bars, entry_m, L):
    """First bar with m in (entry_m, entry_m+K] and close < L. Returns its index or None."""
    m, c = bars['m'], bars['c']
    lo = np.searchsorted(m, entry_m, side='right')
    hi = np.searchsorted(m, entry_m + K, side='right')
    for k in range(lo, hi):
        if c[k] < L:
            return k
    return None


def real_trade(bars, entry_m, L, use_target):
    """The real failed-break-short recipe. Returns dict or None (with a `why_none` reason)."""
    fk = find_failure(bars, entry_m, L)
    if fk is None:
        return None
    fm = bars['m'][fk]
    ek = first_after(bars, fm)
    if ek is None:
        return {'why_none': 'no_entry_bar'}
    entry_price = bars['o'][ek]
    lo = np.searchsorted(bars['m'], entry_m, side='left')
    H = float(bars['h'][lo:fk + 1].max())
    R = H - entry_price
    return dict(entry_price=entry_price, H=H, R=R, entry_idx=ek, fail_idx=fk)


def generic_trade(bars, anchor_m, use_target):
    """The signal-free placebo recipe: stop = prior-10-min high, entry at open of the first bar
    at/after anchor_m. Used by both D1 (random minute, same symbol) and D3 (real entry_m, other
    symbol)."""
    m = bars['m']
    lo = np.searchsorted(m, anchor_m - K, side='left')
    hi = np.searchsorted(m, anchor_m, side='left')   # strictly before anchor_m
    if hi <= lo:
        return {'why_none': 'no_prior_window'}
    H = float(bars['h'][lo:hi].max())
    ek = np.searchsorted(m, anchor_m, side='left')
    if ek >= len(m):
        return {'why_none': 'no_entry_bar'}
    entry_price = bars['o'][ek]
    R = H - entry_price
    return dict(entry_price=entry_price, H=H, R=R, entry_idx=ek)


def cost(spread, why, R, price):
    """net_R cost: the signal-minute NBBO full spread (both legs) + 2bp slip on stop/target only."""
    if pd.isna(spread) or R <= 0:
        return np.nan, np.nan
    spread_R = spread / R
    slip_R = (SPREAD_SLIP_BP * price / R) if why in ('stop', 'target') else 0.0
    return spread_R, slip_R


def run_cell(conn, pop, cell_name, use_target, day_symbols, rng_seed=0):
    """Run the real trade + D1 + D3 placebo for every row of `pop`. Returns a DataFrame of trades
    (one row per symbol-day that produced a real trade) with net_R for real/D1/D3."""
    out = []
    n = len(pop)
    for i, row in enumerate(pop.itertuples()):
        if i % 1000 == 0:
            log(f'  cell {cell_name}: {i}/{n}')
        bars = get_bars(conn, row.symbol, row.day)
        rec = dict(day=row.day, symbol=row.symbol, wk=row.wk, split=row.split, half=row.half,
                    price=row.price, spread_mean=row.spread_mean, why=None, net=np.nan,
                    gross=np.nan, d1_net=np.nan, d3_net=np.nan, exit_reason=None)
        if bars is None:
            rec['why'] = 'no_bars'
            out.append(rec); continue

        # -------- real trade
        tr = real_trade(bars, int(row.entry_m), float(row.entry), use_target)
        if tr is None:
            rec['why'] = 'no_failure'
        elif 'why_none' in tr:
            rec['why'] = tr['why_none']
        elif tr['R'] < MIN_R_PCT * row.price:
            rec['why'] = 'r_too_small'
        else:
            exit_px, reason = walk_exit(bars, tr['entry_idx'], tr['entry_price'], tr['H'], use_target)
            gross = (tr['entry_price'] - exit_px) / tr['R']
            sp_R, slip_R = cost(row.spread_mean, reason, tr['R'], row.price)
            rec.update(why='trade', exit_reason=reason, gross=gross,
                        net=(gross - sp_R - slip_R) if pd.notna(sp_R) else np.nan)

            # -------- D1: random minute in [10:00,14:00), same symbol
            rng = random.Random(f'D1_{row.day}_{row.symbol}')
            rm = rng.randint(600, 839)
            d1 = generic_trade(bars, rm, use_target)
            if d1 and 'why_none' not in d1 and d1['R'] >= MIN_R_PCT * row.price:
                ex1, why1 = walk_exit(bars, d1['entry_idx'], d1['entry_price'], d1['H'], use_target)
                g1 = (d1['entry_price'] - ex1) / d1['R']
                sp1, sl1 = cost(row.spread_mean, why1, d1['R'], row.price)
                rec['d1_net'] = (g1 - sp1 - sl1) if pd.notna(sp1) else np.nan

            # -------- D3: real entry_m, random OTHER symbol on the same day
            others = [s for s in day_symbols.get(row.day, []) if s != row.symbol]
            if others:
                rng3 = random.Random(f'D3_{row.day}_{row.symbol}')
                sym_j = rng3.choice(others)
                bars_j = get_bars(conn, sym_j, row.day)
                if bars_j is not None:
                    d3 = generic_trade(bars_j, int(row.entry_m), use_target)
                    if d3 and 'why_none' not in d3 and d3['R'] >= MIN_R_PCT * row.price:
                        ex3, why3 = walk_exit(bars_j, d3['entry_idx'], d3['entry_price'], d3['H'], use_target)
                        g3 = (d3['entry_price'] - ex3) / d3['R']
                        sp3, sl3 = cost(row.spread_mean, why3, d3['R'], row.price)
                        rec['d3_net'] = (g3 - sp3 - sl3) if pd.notna(sp3) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- stats helpers (score_model.py convention)
def clustered_t(x, day):
    if len(x) < 3:
        return np.nan
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    g = pd.Series(x - mu).groupby(pd.Series(day).values).sum().values
    se = np.sqrt((g ** 2).sum()) / len(x)
    return float(mu / se) if se > 0 else np.nan


def iid_t(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else np.nan


def split_stats(t, split):
    """t: DataFrame of trades (why=='trade', net notna) for one split."""
    if t is None or len(t) == 0:
        return None
    x = t.net.values
    nw = NW.get(split, max(t.wk.nunique(), 1))
    w = t.groupby('wk').net.sum()
    q95 = t.net.quantile(0.95)
    reasons = t.exit_reason.value_counts(normalize=True)
    return dict(
        n=len(t), tpw=round(len(t) / nw, 2),
        meanR_gross=round(float(t.gross.mean()), 3), meanR=round(float(t.net.mean()), 3),
        sd=round(float(t.net.std(ddof=1)), 3) if len(t) > 1 else np.nan,
        t_iid=round(iid_t(x), 2), t_clust=round(clustered_t(x, t.day.values), 2),
        WR=round(float((t.net > 0).mean() * 100), 1),
        ex5=round(float(t.net[t.net <= q95].mean()), 3),
        cap5=round(float(t.net.clip(upper=5.0).mean()), 3),
        share_stop=round(float(reasons.get('stop', 0.0)), 3),
        share_target=round(float(reasons.get('target', 0.0)), 3),
        share_eod=round(float(reasons.get('eod', 0.0)), 3),
        green=round(float((w > 0).sum() / nw), 2),
    )


def main():
    t0 = time.time()
    log('loading features.csv + nbbo.csv')
    feat = pd.read_csv(f'{D}/features.csv', dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    nbbo = pd.read_csv(f'{D}/nbbo.csv', dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    nbbo = nbbo.drop_duplicates(['day', 'symbol'])[['day', 'symbol', 'spread_mean']]
    feat = feat.merge(nbbo, on=['day', 'symbol'], how='left')

    pop_all = feat[feat.split.isin(['TRAIN', 'VAL'])].copy()
    n_pop = len(pop_all)
    log(f'population (TRAIN+VAL, TEST excluded): {n_pop}')

    shortable = (pop_all.price >= SHORTABLE_PRICE) & (pop_all.adv20 >= SHORTABLE_ADV20)
    ssr = pop_all.price <= SSR_RATIO * pop_all.prev_close
    share_not_shortable = round(float((~shortable).mean()) * 100, 2)
    share_ssr = round(float(ssr.mean()) * 100, 2)
    pop = pop_all[shortable & ~ssr].copy()
    log(f'shortable excl {share_not_shortable}% | SSR excl {share_ssr}% | kept {len(pop)}/{n_pop}')
    if '--limit' in sys.argv:
        lim = int(sys.argv[sys.argv.index('--limit') + 1])
        pop = pop.head(lim).copy()
        log(f'SMOKE TEST: --limit {lim} rows')

    # day -> all population symbols (any filter status), for D3's "random other symbol" pool
    day_symbols = pop_all.groupby('day').symbol.apply(list).to_dict()

    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    results = {}
    for cell_name, use_target in (('1357', False), ('1358', True)):
        log(f'=== cell {cell_name} (use_target={use_target}) ===')
        t = run_cell(conn, pop, cell_name, use_target, day_symbols)
        t.to_csv(f'{D}/failed_break_short_{cell_name}_trades.csv', index=False)
        results[cell_name] = t
        log(f'cell {cell_name}: {len(t)} rows, {(t.why == "trade").sum()} real trades')
    conn.close()

    # -------- report tables
    lines = []
    lines.append('# FAILED_BREAK_SHORT_REPORT — PREREG_FAILED_BREAK_SHORT.md (cells 1357-1358)\n')
    lines.append(f'Population (TRAIN+VAL, TEST untouched): {n_pop}. '
                 f'Shortable-proxy excluded: {share_not_shortable}%. SSR-proxy excluded: {share_ssr}%. '
                 f'Kept for simulation: {len(pop)}.\n')

    verdicts = {}
    for cell_name, t in results.items():
        lines.append(f'\n## Cell {cell_name}\n')
        trades = t[t.why == 'trade'].copy()
        no_trade_reasons = t[t.why != 'trade'].why.value_counts()
        lines.append('No-trade reasons (share of kept population): ' +
                     ', '.join(f'{k} {v/len(t)*100:.1f}%' for k, v in no_trade_reasons.items()) + '\n')
        cov = round(float(trades.spread_mean.notna().mean()) * 100, 1) if len(trades) else 0.0
        lines.append(f'NBBO coverage on real trades: {cov}%\n')

        rows = []
        for split in ('TRAIN', 'VAL'):
            st = split_stats(trades[(trades.split == split) & trades.net.notna()], split)
            if st:
                rows.append(dict(split=split, **st))
        for half in ('H1', 'H2'):
            sub = trades[(trades.split == 'TRAIN') & (trades.half == half) & trades.net.notna()]
            st = split_stats(sub, 'TRAIN')
            if st:
                rows.append(dict(split=f'TRAIN-{half}', **st))
        rep = pd.DataFrame(rows)
        lines.append(rep.to_markdown(index=False) if len(rep) else '(no valid trades)')
        lines.append('')

        # placebo (VAL + TRAIN)
        pb_rows = []
        for split in ('TRAIN', 'VAL'):
            sub = trades[(trades.split == split)]
            for tag, col in (('D1 (time-shuffle, same sym)', 'd1_net'), ('D3 (symbol-shuffle, same day)', 'd3_net')):
                v = sub[col].dropna()
                pb_rows.append(dict(split=split, placebo=tag, n=len(v),
                                    meanR=round(float(v.mean()), 3) if len(v) else np.nan))
        pb = pd.DataFrame(pb_rows)
        lines.append('\nPlacebo means:\n' + pb.to_markdown(index=False))

        # week-by-week VAL P&L at R=$100
        val = trades[(trades.split == 'VAL') & trades.net.notna()]
        wk_pnl = (val.groupby('wk').net.sum() * 100).round(0)
        lines.append('\nVAL week-by-week P&L at R=$100:\n' + wk_pnl.to_string())

        # cadence bar on VAL
        cad_csv = f'{D}/failed_break_short_{cell_name}_val_cadence.csv'
        val_out = val[['day', 'symbol', 'net']].rename(columns={'day': 'date', 'net': 'pnl_R'})
        val_out.to_csv(cad_csv, index=False)
        try:
            cad = subprocess.run([sys.executable, f'{ROOT}/scripts/cadence_bar.py', '--trades', cad_csv,
                                   '--split', 'VAL', '--book', f'HOD-S-{cell_name}'],
                                  capture_output=True, text=True, timeout=120)
            lines.append('\nCadence bar (VAL):\n```\n' + (cad.stdout or cad.stderr) + '\n```')
        except Exception as e:
            lines.append(f'\nCadence bar FAILED: {e}')

        # pass-bar verdict
        val_st = split_stats(trades[(trades.split == 'VAL') & trades.net.notna()], 'VAL')
        tr_st = split_stats(trades[(trades.split == 'TRAIN') & trades.net.notna()], 'TRAIN')
        h1_st = split_stats(trades[(trades.split == 'TRAIN') & (trades.half == 'H1') & trades.net.notna()], 'TRAIN')
        h2_st = split_stats(trades[(trades.split == 'TRAIN') & (trades.half == 'H2') & trades.net.notna()], 'TRAIN')
        val_d1 = trades[trades.split == 'VAL'].d1_net.dropna().mean()
        val_d3 = trades[trades.split == 'VAL'].d3_net.dropna().mean()
        criteria = {
            '1. net>=+0.10R TRAIN & VAL, VAL t>=2': bool(
                tr_st and val_st and tr_st['meanR'] >= 0.10 and val_st['meanR'] >= 0.10 and val_st['t_clust'] >= 2),
            '2. both TRAIN halves > 0': bool(h1_st and h2_st and h1_st['meanR'] > 0 and h2_st['meanR'] > 0),
            '3. ex-top-5% > 0 both splits': bool(tr_st and val_st and tr_st['ex5'] > 0 and val_st['ex5'] > 0),
            '4. beats D1 & D3 by >=0.10R on VAL': bool(
                val_st and pd.notna(val_d1) and pd.notna(val_d3)
                and (val_st['meanR'] - val_d1) >= 0.10 and (val_st['meanR'] - val_d3) >= 0.10),
            '5. >=3 trades/week VAL': bool(val_st and val_st['tpw'] >= 3),
        }
        verdict = 'PASS' if all(criteria.values()) else 'FAIL'
        verdicts[cell_name] = verdict
        lines.append(f'\n### Pass-bar criteria — cell {cell_name}: **{verdict}**\n')
        for k, v in criteria.items():
            lines.append(f'- {k}: {"PASS" if v else "FAIL"}')
        lines.append(f'- VAL D1 mean {round(float(val_d1),3) if pd.notna(val_d1) else "NA"} R, '
                     f'VAL D3 mean {round(float(val_d3),3) if pd.notna(val_d3) else "NA"} R')

    lines.append(f'\n## Overall: cell 1357 {verdicts.get("1357")}, cell 1358 {verdicts.get("1358")}')
    lines.append(f'\nRuntime: {(time.time()-t0)/60:.1f} min.')

    with open(f'{D}/FAILED_BREAK_SHORT_REPORT.md', 'w') as f:
        f.write('\n'.join(str(x) for x in lines))
    log('report written: FAILED_BREAK_SHORT_REPORT.md')
    log('DONE')


if __name__ == '__main__':
    main()
