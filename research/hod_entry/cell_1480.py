#!/usr/bin/env python3
"""Cell 1,480 -- research/hod_entry/PREREG_1478.md (FROZEN 2026-09-26) + amendment item 2.

FAILED-BREAK FLIP SHORT overlay on the 9,911-fill base book. Detection is BAR-LEVEL (no network) on
research/hod_entry/bars_fills_1478.db, the SAME single fresh store cell 1,479 uses: a failed break is
the first bar with low <= level - $0.01, strictly after the fill (break) bar, within 15 minutes of it,
and before the base trade's own target bar (if the base trade's `why` is 'target', the window closes
at that bar; the 15-minute cap otherwise applies against the min of fill_min+15 and EOD_M). Only a
bar-level candidate triggers a network call: research/hod_entry/causal_arming.fetch_window(symbol,
day, m, m+1) prices the exact print, cached under research/hod_entry/sip_cache_1480/ so a rerun never
re-fetches. Candidates are capped at 1,500 fetches PER HOLDOUT (random.Random(1480).sample when a
holdout has more); the cap and how many candidates it dropped are reported.

Long leg: exits at the failed-break print (this REPLACES the base trade's own exit for 1,480's own
book; the base book itself is untouched). Short leg (amendment item 2's stop floor):
  short_entry  = NBBO bid at the print
  short_stop   = max(break-bar high + $0.01, short_entry * 1.01)      (amendment: a doji break bar
                 made the pre-amendment R ~= 0; the 1% floor prevents that)
  short_target = short_entry - 2 * (short_stop - short_entry)
  cover        = 15:55 at that bar's open, priced at the ask (see cost model below); stop covers at
                 the stop price (or the bar's open if it gaps through), also at the ask.
Costs: entry at the bid (no extra charge, per spec -- the print's own bid IS the fill). Exit (cover)
at the ask: modeled as short_entry's/short_exit's own day-symbol exit_half (EH, the SAME real
half-spread cell_1479 recovers for the long leg, reused here as the day's measured spread proxy --
no fresh NBBO exists for an arbitrary future cover bar either) plus the flat SLIP_BP impact term.
Stop-type covers additionally pay the PREREG's literal fallback: "35 bps fallback where unmeasured"
(SHORT_STOP_FALLBACK_BPS below) -- unlike cell 1,463's long-stop slip, no cell has ever measured a
short-stop fill rate on this population, so the fallback IS the number, per spec.
Shortable / SSR: shortable is read via cell_1439.load_borrow_flags() (today's asset-dump snapshot,
same caveat as cells 1,431/1,439 -- reported as a SHARE, not a filter). SSR (prior close * 0.9
breached, PIT daily, at or before the short's own entry bar) is computed from the Databento daily
panel (cell_1445.build_daily_panel, causal prev_close) and EXCLUDES the row from the primary book
per spec (rows with no resolvable prev_close are treated the same way -- conservative, logged).
"""
import os
import sys
import time
import random
import pickle
import sqlite3
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sip_rebuild as sr           # noqa: E402  (TICK, SLIP_BP, prevailing_quote)
import causal_arming as ca         # noqa: E402  (fetch_window)
import cell_1445 as c1445          # noqa: E402  (day_clustered_t, ex_top5_mean, symbol map / daily panel)
import cell_1439 as c1439          # noqa: E402  (load_borrow_flags)
import cell_1479 as c1479          # noqa: E402  (load_base, load_bars_1478, BARS_DB, EOD_M, OPEN_M)

OUT_CSV = os.path.join(HERE, 'cell_1480_fills.csv')
LOG_PATH = os.path.join(HERE, 'cell_1480.log')
CACHE_DIR = os.path.join(HERE, 'sip_cache_1480')
os.makedirs(CACHE_DIR, exist_ok=True)

FETCH_CAP_PER_HOLDOUT = 1500
SEED = 1480
SHORT_STOP_FALLBACK_BPS = 35.0        # PREREG amendment: "35 bps fallback where unmeasured"
FAILED_BREAK_WINDOW_MIN = 15


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


# ------------------------------------------------------------------------------------------- stage 1
def find_candidates(fills, con):
    """Bar-level (no network) failed-break detection. Returns a DataFrame: one row per fill that has
    a candidate bar, with day/symbol/fill_min/m_candidate/brk_high/holdout carried through. A fill
    with no candidate bar contributes zero rows (it is not a 1,480 case, not a fetch, not counted
    anywhere but the scan)."""
    rows = []
    n_scanned = 0
    for day, g in fills.groupby('day'):
        syms = sorted(g.symbol.unique())
        bars = c1479.load_bars_1478(con, day, syms)
        for r in g.itertuples():
            n_scanned += 1
            if r.why == 'stop_bar':
                continue                                  # never left the break bar; no post-fill window
            b = bars.get(r.symbol)
            if b is None or not len(b):
                continue
            m_break = int(np.floor(r.fill_min))
            cutoff = m_break + FAILED_BREAK_WINDOW_MIN
            if r.why == 'target':
                cutoff = min(cutoff, int(r.exit_m))        # "before the target": stop at the base's own target bar
            cutoff = min(cutoff, c1479.EOD_M)
            path = b[(b.m > m_break) & (b.m <= cutoff)]
            if not len(path):
                continue
            hit = path[path.l <= r.level - 0.01 + 1e-9]
            if not len(hit):
                continue
            m_cand = int(hit.m.iloc[0])
            brk_row = b[b.m == m_break]
            brk_high = float(brk_row.h.iloc[0]) if len(brk_row) else float(r.level)
            rows.append(dict(day=r.day, symbol=r.symbol, split=r.split, holdout=r.holdout, wk=r.wk,
                              fill_min=r.fill_min, fill=r.fill, stop=r.stop, R=r.R, level=r.level,
                              base_why=r.why, base_net_R=r.net_R, half_entry=r.half_entry,
                              exit_half=r.exit_half, m_break=m_break, m_candidate=m_cand, brk_high=brk_high))
    log(f'find_candidates: scanned {n_scanned} fills, {len(rows)} bar-level failed-break candidates')
    return pd.DataFrame(rows)


def cap_candidates(cands):
    """random.Random(SEED) subsample to FETCH_CAP_PER_HOLDOUT per holdout; reports the cap when hit."""
    keep_idx = []
    for holdout, g in cands.groupby('holdout'):
        if len(g) > FETCH_CAP_PER_HOLDOUT:
            rng = random.Random(SEED)
            idx = rng.sample(list(g.index), FETCH_CAP_PER_HOLDOUT)
            log(f'cap_candidates: {holdout} has {len(g)} candidates > cap {FETCH_CAP_PER_HOLDOUT} -- '
                f'random.Random({SEED}) sampled {len(idx)}, dropped {len(g) - len(idx)}')
            keep_idx.extend(idx)
        else:
            keep_idx.extend(list(g.index))
    return cands.loc[sorted(keep_idx)].reset_index(drop=True)


# ------------------------------------------------------------------------------------------- stage 2
def _cache_path(symbol, day, m):
    return os.path.join(CACHE_DIR, f'{symbol}_{day}_{m}.pkl')


def fetch_print(symbol, day, m, level):
    """One minute of trades+quotes via causal_arming.fetch_window(symbol, day, m, m+1), cached to
    disk. Returns dict(print_ts, print_px, bid, ask) for the first trade <= level-0.01, or None if
    the fresh tape disagrees with the bar (no such print -- logged), or 'no_quote'/'fetch_error'."""
    cp = _cache_path(symbol, day, m)
    if os.path.exists(cp):
        with open(cp, 'rb') as f:
            t, q = pickle.load(f)
    else:
        try:
            t, q = ca.fetch_window(symbol, day, m, m + 1)
        except Exception as e:                                          # noqa: BLE001 -- network
            return 'fetch_error', str(e)
        with open(cp, 'wb') as f:
            pickle.dump((t, q), f)
    if not len(t):
        return 'no_tape', None
    w = t[t.price <= level - 0.01 + 1e-9].sort_values('ts', kind='stable')
    if not len(w):
        return 'bar_tick_disagree', None                  # bar low <= level-0.01 but no such print in the tape
    t_hit = int(w.ts.iloc[0])
    pq = sr.prevailing_quote(q, t_hit)
    if pq is None:
        return 'no_quote', None
    bid, ask = pq
    return dict(print_ts=t_hit, print_px=float(w.price.iloc[0]), bid=bid, ask=ask), None


def walk_short(entry, stop, target, path):
    """Mirror of sip_rebuild.walk_path for a SHORT: gap-through at the open, high>=stop -> stop
    (stop-priority when a bar touches both), low<=target -> target, 15:55 -> its open."""
    for row in path.itertuples():
        m, o, h, l, c = row.m, row.o, row.h, row.l, row.c
        if m >= c1479.EOD_M:
            return int(m), float(o), 'eod'
        if h >= stop - 1e-9:
            return int(m), float(o if o >= stop else stop), 'stop'
        if l <= target + 1e-9:
            return int(m), float(target), 'target'
    last = path.iloc[-1]
    return int(last.m), float(last.c), 'eod_fallback'


def price_leg(row, bars_cache):
    """Long-exit + short-trade pricing for one (already tick-priced) candidate row. Returns a dict
    of output columns, or None if the short never gets a valid (positive-risk) stop."""
    R0 = row.R
    long_exit = row.print_px
    long_raw_R = (long_exit - row.fill) / R0
    long_cost_R = (row.half_entry + row.exit_half + sr.SLIP_BP * long_exit) / R0
    long_net_R = long_raw_R - long_cost_R

    short_entry = row.bid
    short_stop = max(row.brk_high + sr.TICK, short_entry * 1.01)          # amendment item 2 floor
    short_R = short_stop - short_entry
    if short_R <= 0:
        return None
    short_target = short_entry - 2 * short_R

    b = bars_cache.get((row.day, row.symbol))
    spath = b[(b.m > row.m_candidate) & (b.m <= c1479.EOD_M)] if b is not None else pd.DataFrame()
    if b is None or not len(spath):
        s_exit_m, s_exit_px, s_why = row.m_candidate, short_entry, 'no_path'
    else:
        s_exit_m, s_exit_px, s_why = walk_short(short_entry, short_stop, short_target, spath)
    short_raw_R = (short_entry - s_exit_px) / short_R
    stop_type = s_why == 'stop'
    extra_slip = SHORT_STOP_FALLBACK_BPS * 1e-4 * s_exit_px / short_R if stop_type else 0.0
    short_cost_R = (row.exit_half + sr.SLIP_BP * s_exit_px) / short_R + extra_slip
    short_net_R = short_raw_R - short_cost_R
    return dict(long_exit=long_exit, long_raw_R=long_raw_R, long_net_R=long_net_R,
                long_delta_R=long_net_R - row.base_net_R,
                short_entry=short_entry, short_stop=short_stop, short_target=short_target,
                short_exit_m=s_exit_m, short_exit_px=s_exit_px, short_why=s_why,
                short_raw_R=short_raw_R, short_net_R=short_net_R)


# ------------------------------------------------------------------------------------------- SSR / shortable
def attach_ssr_shortable(df):
    """Adds `shortable` (bool, from the asset-dump snapshot, cell_1439.load_borrow_flags) and `ssr`
    (bool/NaN: prior close * 0.9 breached on the signal day at or before m_candidate, PIT daily,
    causal prev_close from cell_1445.build_daily_panel). NaN ssr (no resolvable prev_close) is
    logged and treated as excluded from the primary book, same as ssr==True, per spec."""
    borrow = c1439.load_borrow_flags()
    df['shortable'] = df.symbol.map(lambda s: bool(borrow.loc[s].shortable) if s in borrow.index else False)
    n_missing_borrow = int((~df.symbol.isin(borrow.index)).sum())
    if n_missing_borrow:
        log(f'attach_ssr_shortable: {n_missing_borrow}/{len(df)} symbols missing from the borrow-flags '
            f'asset dump -- shortable defaulted False for those (counted, not silently dropped)')

    map_df = c1445.load_symbol_map()
    pairs = list(zip(df.symbol, df.day))
    instr = c1445.resolve_instrument_ids(pairs, map_df)
    daily = c1445.build_daily_panel(set(instr.values())) if instr else pd.DataFrame()
    daily_idx = daily.set_index(['instrument_id', 'bar_date']) if len(daily) else None

    ssr_vals, n_no_prevclose = [], 0
    for r in df.itertuples():
        iid = instr.get((r.symbol, r.day))
        prev_close = np.nan
        if iid is not None and daily_idx is not None:
            key = (iid, pd.Timestamp(r.day))
            if key in daily_idx.index:
                prev_close = daily_idx.loc[key, 'prev_close']
                if isinstance(prev_close, pd.Series):
                    prev_close = prev_close.iloc[0]
        if not np.isfinite(prev_close):
            ssr_vals.append(np.nan)
            n_no_prevclose += 1
            continue
        floor = prev_close * 0.9
        b = r.bars_obj
        breached = bool((b[b.m <= r.m_candidate].l <= floor + 1e-9).any()) if b is not None and len(b) else False
        ssr_vals.append(breached)
    if n_no_prevclose:
        log(f'attach_ssr_shortable: {n_no_prevclose}/{len(df)} rows had no resolvable prior close '
            f'(PIT daily panel) -- ssr set NaN, excluded from the primary book per spec')
    df = df.drop(columns=['bars_obj'])
    df['ssr'] = ssr_vals
    return df


# ------------------------------------------------------------------------------------------- scoring
def score_holdout(df, holdout, base_book):
    """short-leg mean net R, day-clustered t, ex-top-5 %, shorts/week; long-leg ΔR mean; on the
    PRIMARY book = ssr != True (NaN ssr excluded too, conservative). shortable share reported on the
    FULL (pre-SSR-exclusion) candidate set, per spec ("share reported", not a filter)."""
    d_all = df[df.holdout == holdout]
    if not len(d_all):
        return dict(n=0)
    prim = d_all[d_all.ssr == False]                                    # noqa: E712 -- NaN excluded too
    weeks = base_book[base_book.holdout == holdout].wk.nunique()
    t = c1445.day_clustered_t(prim.short_net_R, prim.day) if len(prim) > 1 else np.nan
    return dict(n=len(prim), n_all=len(d_all), mean_short_net_R=float(prim.short_net_R.mean()) if len(prim) else np.nan,
                t=t, extop5=float(c1445.ex_top5_mean(prim.short_net_R)) if len(prim) else np.nan,
                shorts_wk=len(prim) / max(weeks, 1), long_delta_mean=float(d_all.long_delta_R.mean()),
                share_shortable=float(d_all.shortable.mean()), share_ssr_true=float((d_all.ssr == True).mean()),  # noqa: E712
                share_ssr_nan=float(d_all.ssr.isna().mean()))


def main(argv=None):
    global FETCH_CAP_PER_HOLDOUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--fetch-cap', type=int, default=FETCH_CAP_PER_HOLDOUT)
    args = ap.parse_args(argv)
    FETCH_CAP_PER_HOLDOUT = args.fetch_cap

    log('=== cell_1480 start ===' + (' (SMOKE)' if args.smoke else ''))
    fills = c1479.load_base()
    if args.smoke:
        fills = fills.head(300).copy()
    con = sqlite3.connect(f'file:{c1479.BARS_DB}?mode=ro', uri=True, timeout=120)

    cands = find_candidates(fills, con)
    if not len(cands):
        log('no bar-level failed-break candidates found -- writing empty output')
        pd.DataFrame().to_csv(OUT_CSV, index=False)
        return pd.DataFrame()
    cands = cap_candidates(cands)

    log(f'fetching {len(cands)} candidate minute windows (cache: {CACHE_DIR})...')
    n_fetch = n_disagree = n_error = n_noquote = n_priced = 0
    priced_rows = []
    bars_cache = {}
    for i, r in enumerate(cands.itertuples()):
        if (i + 1) % 200 == 0:
            log(f'  ... {i + 1}/{len(cands)} fetched ({n_priced} priced, {n_disagree} bar/tick '
                f'disagreements, {n_error} fetch errors)')
        res, err = fetch_print(r.symbol, r.day, r.m_candidate, r.level)
        n_fetch += 1
        if res == 'fetch_error':
            n_error += 1
            log(f'  WARNING fetch_error {r.symbol} {r.day} m={r.m_candidate}: {err}')
            continue
        if res in ('no_tape', 'bar_tick_disagree'):
            n_disagree += 1
            continue
        if res == 'no_quote':
            n_noquote += 1
            continue
        n_priced += 1
        d = r._asdict()
        d.update(print_ts=res['print_ts'], print_px=res['print_px'], bid=res['bid'], ask=res['ask'])
        priced_rows.append(d)
        if (r.day, r.symbol) not in bars_cache:
            bars_cache[(r.day, r.symbol)] = c1479.load_bars_1478(con, r.day, [r.symbol]).get(r.symbol)
    log(f'fetch stage done: {n_fetch} fetched, {n_priced} priced, {n_disagree} bar/tick disagreements, '
        f'{n_noquote} no-quote, {n_error} fetch errors')

    priced = pd.DataFrame(priced_rows)
    legs = [price_leg(r, bars_cache) for r in priced.itertuples()]
    keep = [i for i, l in enumerate(legs) if l is not None]
    n_zero_risk = len(legs) - len(keep)
    if n_zero_risk:
        log(f'WARNING {n_zero_risk} candidates had short_R <= 0 even after the amendment floor -- dropped')
    priced = priced.iloc[keep].reset_index(drop=True)
    legs_df = pd.DataFrame([legs[i] for i in keep])
    out = pd.concat([priced.drop(columns=['Index'], errors='ignore').reset_index(drop=True), legs_df], axis=1)
    out['bars_obj'] = [bars_cache.get((r.day, r.symbol)) for r in out.itertuples()]
    out = attach_ssr_shortable(out)

    cols = [c for c in out.columns if c not in ('bars_obj',)]
    out[cols].to_csv(OUT_CSV, index=False)
    log(f'wrote {OUT_CSV} ({len(out)} rows)')

    for sp in ('TRAIN-H2', 'VAL'):
        s = score_holdout(out, sp, fills)
        if s.get('n_all'):
            log(f'{sp}: n_primary={s["n"]}/{s["n_all"]} mean_short_net_R={s["mean_short_net_R"]:+.4f} '
                f't={s["t"]:.2f} extop5={s["extop5"]:+.4f} shorts_wk={s["shorts_wk"]:.2f} '
                f'long_delta_mean={s["long_delta_mean"]:+.4f} share_shortable={s["share_shortable"]:.2f} '
                f'share_ssr_true={s["share_ssr_true"]:.3f} share_ssr_nan={s["share_ssr_nan"]:.3f}')
    log('=== cell_1480 END ===')
    return out


if __name__ == '__main__':
    main()
