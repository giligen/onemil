"""Cell 1,439 — low-of-day mirror (research/hod_entry/PREREG_WEEKEND.md, frozen 2026-09-25 18:35
UTC). The HOD-break resting-entry spec (`trading/hod_break.py`, `research/hod_entry/causal_arming.py`
cell 1,438) MIRRORED for the low side:

  * level = running LOD through bar j (min low, not max high).
  * armed at the close of bar j iff, through bar j only: a MIRRORED consolidation holds — bars
    j-K+1..j all sit within consol_pct ABOVE the running LOD (their max high, not min low) —
    price <= open x (1 - min_dist_open_pct%), rv_profile in band, m[j+1] <= last_entry_minute,
    trigger >= floor (the min_price admission gate is about the underlying's price, not direction)
    and (stop - trigger)/trigger >= min_r_pct%.
  * armed: resting SELL stop-limit, trigger = level - $0.01, limit = level x (1 - LIMIT_BPS) i.e.
    LOD x 0.9985; fills at the prevailing NBBO BID at the first SIP print <= trigger inside bar
    j+1's window if bid >= limit (mirror of "ask <= limit"); a print >= stop later in that window
    = stopped.
  * stop = the mirrored consolidation HIGH (not low); target = fill - 2 R; flat 15:55 ET (cover).
  * data: `research/bf_zero/bars_sip.db` minute bars for levels/arming (cache.db has the same
    sparse-bar problem cell 1,438 documented — SIP wins on tie, same as causal_arming.load_day_bars);
    SIP trades+quotes fetched new into `research/hod_entry/sip_cache_lod/` (resumable per-day
    pickle cache, mirrors causal_arming's c1438_{day}.pkl.gz).
  * universe: the causal LOW-side superset (mirror of spec_sim.py's high>=open*(1+min_dist)):
    symbol-days with low <= open x (1 - min_dist_open_pct%), adv20 >= min_adv20, low <= floor - 1c
    (a trigger <= the floor from below needs it — mirrored inequality), test tickers excluded.
  * shortable + borrow-fee-aware: `research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv` (same file
    cell 1,431 used — the only Alpaca asset dump in the repo carrying shortable/easy_to_borrow;
    today's-snapshot survivorship caveat applies identically). Scored ALL and shortable-only.
  * costs: measured half-spread at fill (bid-side), B0's exit cost rule (half-spread + 2 bp of the
    exit price), a 30 bps stop-slip variant on stop exits, reported both with and without — the
    PREREG_WEEKEND.md header convention, identical to causal_arming/cell_1431/1442.

Pass bar (frozen): fill mean net R >= +0.10 both holdouts (TRAIN-H2, VAL), VAL t >= 2, ex-top-5% > 0,
coverage >= 80%, winner/loser missingness gap <= 5pp, >= 3 fills/week (simulate_slots, first-12/day
4-concurrent). Coverage below 80% of the population VOIDS the cell (no PASS/FAIL claim) per the
project's phrasing rule — never report a closure without an adequacy review.
"""
import argparse
import gzip
import os
import pickle
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sip_rebuild as sr                                    # noqa: E402
import causal_arming as ca                                  # noqa: E402 — window_ns, fetch_window, load_day_bars, live_params

ROOT = sr.ROOT
sys.path.insert(0, os.path.join(ROOT, 'research/hod_consol'))
sys.path.insert(0, os.path.join(ROOT, 'research/hod_entry'))
import cell_1430 as c1430                                   # noqa: E402 — day_clustered_t, STOP_SLIP_BP
import run_consol                                            # noqa: E402 — simulate_slots
from trading.hod_break import rv_profile                     # noqa: E402 — side-agnostic (volume only)

UNIVERSE_CSV = os.path.join(ROOT, 'research/bf_zero/universe.csv')
BORROW_CSV = os.path.join(ROOT, 'research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv')
CACHE_DIR = os.path.join(HERE, 'sip_cache_lod')
OUT_CSV = os.path.join(HERE, 'cell_1439_fills.csv')
REPORT_MD = os.path.join(HERE, 'RESULT_1439.md')
SPLITS = {'TRAIN': ('2025-07-01', '2025-12-31'), 'VAL': ('2026-01-01', '2026-05-31')}
TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')
STOP_SLIP_BP = 0.0030
log = sr.log
os.makedirs(CACHE_DIR, exist_ok=True)


# --------------------------------------------------------------------------------------------- mirrored arming
def consolidation_high_mirror(l, h, j, p):
    """Mirror of trading.hod_break.consolidation_low: bars j-K+1..j hold within consol_pct ABOVE
    the running LOD at j -> return their max high; else None."""
    if j + 1 < p.consol_bars + 1:
        return None
    lod = float(np.min(l[: j + 1]))
    highs = np.asarray(h[j + 1 - p.consol_bars: j + 1], dtype=float)
    hi = float(highs.max())
    if hi <= lod * (1.0 + p.consol_pct) and hi > lod:
        return hi
    return None


def arm_state_low(o, h, l, v, m, j, adv20, p, floor, use_rv=True):
    """Mirror of causal_arming.arm_state for a SELL stop-limit. Returns dict(level, trigger, limit,
    stop) or None. level/trigger are the running LOD; stop is the mirrored consolidation HIGH."""
    if int(m[j]) + 1 > p.last_entry_minute:
        return None
    stop = consolidation_high_mirror(l, h, j, p)
    if stop is None:
        return None
    level = float(np.min(l[: j + 1]))
    if level > float(o[0]) * (1.0 - p.min_dist_open_pct / 100.0):
        return None
    if use_rv:
        rv = rv_profile(float(np.sum(v[: j + 1])), adv20, int(m[j]))
        if not (p.rv_lo <= rv < p.rv_hi):
            return None
    trigger = round(level - sr.TICK, 6)
    if trigger < floor or (stop - trigger) / trigger * 100.0 < p.min_r_pct:
        return None
    return dict(level=level, trigger=trigger, limit=level * (1.0 - sr.LIMIT_BPS), stop=stop)


def armed_crossing_bars_low(bars, adv20, p, floor, use_rv=True):
    """Armed bars j whose next bar's minute LOW reaches the trigger (mirror of armed_crossing_bars)."""
    o, h, l, v = (bars[k].values.astype(float) for k in ('o', 'h', 'l', 'v'))
    m = bars.m.values.astype(int)
    lod = np.minimum.accumulate(l)
    out = []
    for j in np.nonzero(l[1:] <= lod[:-1] - sr.TICK + 1e-9)[0]:
        a = arm_state_low(o, h, l, v, m, int(j), adv20, p, floor, use_rv)
        if a is not None and l[j + 1] <= a['trigger'] + 1e-9:
            out.append(dict(a, j=int(j), m_lo=int(m[j]), m_hi=int(m[j + 1]), cumv_j=float(np.sum(v[: j + 1]))))
    return out


# --------------------------------------------------------------------------------------------- fill resolution
def resolve_window_low(trades, quotes, start, end, arm):
    """Mirror of causal_arming.resolve_window for a SELL stop-limit: fill at the BID at the first
    print <= trigger, valid iff bid >= limit; stopped iff a later print >= stop."""
    out = dict(status='no_tape', fill=np.nan, fill_ts=np.nan, half_spread=np.nan, stopped=False)
    w = trades[(trades.ts >= start) & (trades.ts < end)].sort_values('ts', kind='stable')
    if not len(w) or not len(sr._valid(quotes[quotes.ts < end])):
        return out
    hits = w[w.price <= arm['trigger'] + 1e-9]
    if not len(hits):
        out['status'] = 'nofill'
        return out
    t_hit = int(hits.ts.iloc[0])
    pq = sr.prevailing_quote(quotes, t_hit)
    if pq is None:
        return out
    bid, ask = pq
    if bid < arm['limit'] - 1e-9:
        out['status'] = 'nofill'
        return out
    after = w[w.ts > t_hit]
    out.update(status='fill', fill=bid, fill_ts=t_hit, half_spread=0.5 * (ask - bid),
               stopped=bool((after.price >= arm['stop'] - 1e-9).any()))
    return out


def resolve_day_low(cands, tape_for, day):
    for a in cands:
        start, end = ca.window_ns(day, a['m_lo'], a['m_hi'])
        t, q = tape_for(a)
        e = resolve_window_low(t, q, start, end, a)
        if e['status'] == 'no_tape':
            return 'no_tape', a, e
        if e['status'] == 'fill':
            return 'fill', a, e
    return 'nofill', None, None


def proxy_outcome_low(cands, bars):
    """Winner/loser proxy for the missingness gap (mirror): fill at the trigger on the first crossing
    armed bar, B0-mirrored path rules."""
    if not cands:
        return np.nan
    a = cands[0]
    fill, stop = a['trigger'], a['stop']
    path = bars[(bars.m > a['m_hi']) & (bars.m <= sr.EOD_M)]
    if not len(path):
        return np.nan
    _, px, _ = walk_path_short(fill, stop, fill - sr.TARGET_R * (stop - fill), path)
    return (fill - px) / (stop - fill)


def walk_path_short(fill, stop, target, path):
    """Mirror of sip_rebuild.walk_path for a SHORT: high>=stop -> stop (stop-first on a bar touching
    both), low<=target -> target, gap-through at the open, 15:55 bar -> its open."""
    for row in path.itertuples():
        if row.m >= sr.EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.h >= stop:
            return int(row.m), float(row.o if row.o >= stop else stop), 'stop'
        if row.l <= target:
            return int(row.m), float(target), 'target'
    last = path.iloc[-1]
    log(f'[walk_short] WARNING: path ended before 15:55 (last m={int(last.m)}) — exit at its close')
    return int(last.m), float(last.c), 'eod_fallback'


def trade_result_short(fill, stop, half_entry, exit_px, exit_half):
    """Mirror of sip_rebuild.trade_result: R = stop - fill (short); raw = (fill - exit)/R."""
    R = stop - fill
    raw = (fill - exit_px) / R
    cost = (half_entry + exit_half + sr.SLIP_BP * exit_px) / R
    return raw, cost, raw - cost, R


# --------------------------------------------------------------------------------------------- population
def load_population_low(p, floor, min_adv):
    """Mirror of causal_arming.load_population: low <= open x (1 - min_dist%), adv20 >= min_adv,
    day HIGH >= floor + 1c (a trigger >= the floor is impossible otherwise), test tickers out.

    9/25 judge's review VOIDED the first full run: the filter read `low <= floor - 1c`, i.e. it KEPT only symbol-days
    whose price later fell below the $20 floor while every trigger sits above it — the running low at arm time is
    above the floor, so the sub-floor print always comes AFTER the entry: a look-ahead on the short's outcome
    (+0.6 R). A daily-bar filter may only encode a condition that is NECESSARY for an armed cross to exist."""
    u = pd.read_csv(UNIVERSE_CSV, dtype={'symbol': str}, keep_default_na=False)
    for c in ('open', 'high', 'low', 'adv20'):
        u[c] = pd.to_numeric(u[c], errors='coerce')
    u = u.rename(columns={'bar_date': 'day'})
    u = u[(u.day >= SPLITS['TRAIN'][0]) & (u.day <= SPLITS['VAL'][1])]
    u = u[(u.low <= u.open * (1 - p.min_dist_open_pct / 100)) & (u.adv20 >= min_adv) & (u.high >= floor + sr.TICK)]
    u = u[~u.symbol.str.match(TEST_TICKER)].drop_duplicates(['day', 'symbol'])
    u['split'] = np.where(u.day <= SPLITS['TRAIN'][1], 'TRAIN', 'VAL')
    u['wk'] = pd.to_datetime(u.day).dt.to_period('W-FRI').astype(str)
    log(f'[pop] {len(u)} symbol-days in the low-side causal superset (TRAIN-H2 {int((u.split == "TRAIN").sum())}, '
        f'VAL {int((u.split == "VAL").sum())})')
    return u[['day', 'symbol', 'adv20', 'split', 'wk']].reset_index(drop=True)


def load_borrow_flags():
    d = pd.read_csv(BORROW_CSV)
    d = d.drop_duplicates('symbol').set_index('symbol')
    return d[['shortable', 'easy_to_borrow']]


# --------------------------------------------------------------------------------------------- run
def process_symbol_day(r, bars, p, floor, cache, new):
    """One symbol-day. Returns a list of result dicts (0 or 1 rows)."""
    def tape_for(a):
        k = f"{r.symbol}|{a['m_lo']}|{a['m_hi']}"
        if k in cache:
            return cache[k]
        if k in new:
            return new[k]
        new[k] = ca.fetch_window(r.symbol, r.day, a['m_lo'], a['m_hi'])
        return new[k]

    cands = armed_crossing_bars_low(bars, r.adv20, p, floor, use_rv=True)
    base = dict(day=r.day, symbol=r.symbol, split=r.split, wk=r.wk, n_cross=len(cands))
    if not cands:
        return [{**base, 'status': 'not_armed'}]
    base['b0_net_R'] = proxy_outcome_low(cands, bars)
    status, a, e = resolve_day_low(cands, tape_for, r.day)
    if status != 'fill':
        return [{**base, 'status': status}]
    fill, stop = e['fill'], a['stop']
    R = stop - fill
    fill_min = sr.ns_to_et_minutes(e['fill_ts'], r.day)
    if e['stopped']:
        exit_m, px, why = fill_min, stop, 'stop_bar'
    else:
        path = bars[(bars.m > a['m_hi']) & (bars.m <= sr.EOD_M)]
        if not len(path):
            log(f'[sim] WARNING {r.day} {r.symbol}: no minute bars after the fill bar — exit at the fill (0 R raw)')
            exit_m, px, why = fill_min, fill, 'no_path'
        else:
            exit_m, px, why = walk_path_short(fill, stop, fill - sr.TARGET_R * R, path)
    raw, cost, net, _ = trade_result_short(fill, stop, e['half_spread'], px, e['half_spread'])
    stop_slip_net = net - (STOP_SLIP_BP if why == 'stop_bar' else 0.0)
    return [{**base, 'status': 'fill', 'fill': fill, 'stop': stop, 'level': a['level'], 'fill_min': fill_min,
             'exit_m': exit_m, 'exit_price': px, 'why': why, 'R': R, 'raw_R': raw, 'cost_R': cost, 'net_R': net,
             'net_R_stopslip': stop_slip_net}]


def run(workers=4, day_limit=None, time_budget_s=None):
    """Scan, fetch (resumable per-day cache sip_cache_lod/c1439_{day}.pkl.gz), resolve, write CSV."""
    import time
    t0 = time.time()
    p, floor, min_adv = ca.live_params()
    log(f'[params] {p} | min_price {floor} | min_adv20 {min_adv}')
    pop = load_population_low(p, floor, min_adv)
    con = sqlite3.connect(sr.CACHE_DB_URI, uri=True, timeout=120)
    sipcon = sqlite3.connect(ca.BARS_SIP_URI, uri=True, timeout=120)
    rows, n_nobars, src = [], 0, {}
    days = sorted(pop.day.unique())
    if day_limit:
        days = days[:day_limit]
    for day in days:
        if time_budget_s and time.time() - t0 > time_budget_s:
            log(f'[run] time budget hit after {len(rows)} symbol-days — stopping (resumable)')
            break
        sub = pop[pop.day == day]
        bars = ca.load_day_bars(con, day, sub.symbol.tolist(), sipcon, src)
        path = os.path.join(CACHE_DIR, f'c1439_{day}.pkl.gz')
        cache = {}
        if os.path.exists(path):
            with gzip.open(path, 'rb') as f:
                cache = pickle.load(f)
        jobs = []
        for r in sub.itertuples():
            b = bars.get(r.symbol)
            if b is None or len(b) < p.consol_bars + 2:
                n_nobars += 1
                continue
            jobs.append((r, b))
        new_all = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(process_symbol_day, r, b, p, floor, cache, new_all) for r, b in jobs]
            for fu in futs:
                try:
                    rows.extend(fu.result())
                except Exception as exc:
                    log(f'[run] ERROR symbol-day failed: {exc}')
        if new_all:
            cache.update(new_all)
            with gzip.open(path, 'wb') as f:
                pickle.dump(cache, f)
        log(f'[run] {day}: {len(sub)} symbol-days, cache +{len(new_all)}, total rows {len(rows)}')
    con.close(); sipcon.close()
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    log(f'[run] wrote {OUT_CSV} ({len(df)} rows, {n_nobars} no-bars, src={src})')
    return df, len(pop), n_nobars


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--day-limit', type=int, default=None)
    ap.add_argument('--time-budget-s', type=float, default=None)
    a = ap.parse_args(argv)
    run(a.workers, a.day_limit, a.time_budget_s)


if __name__ == '__main__':
    main()
