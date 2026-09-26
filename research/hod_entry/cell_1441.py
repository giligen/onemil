"""Cell 1,441 — prior-day-high (PDH) break, SAME order as cell 1,438 (research/hod_entry/PREREG_WEEKEND.md
"## 1,441", frozen). Only the LEVEL changes: instead of the running HOD through bar j, the level is fixed for
the whole session at the PRIOR trading session's RTH high (09:30-15:59 ET), read from
`research/bf_zero/bars_sip.db` only (never cache.db / daily_bars — a missing prior session is a distinct
'no_pdh' outcome, counted and reported, never backfilled).

Because this is still the LONG side with the same fill mechanics as cell 1,438, everything except arming is
reused directly from `causal_arming` (`ca`) and `sip_rebuild` (`sr`): ca.resolve_window / ca.resolve_day /
ca.proxy_outcome (fill-at-trigger proxy + B0 path physics), sr.walk_path, sr.trade_result, sr.score,
ca.stop_slip_net, ca._legs, ca.load_day_bars, ca.fetch_window, ca.window_ns, ca.live_params, ca.BARS_SIP_URI.

New in this file:
  * `consolidation_low_pdh` — mirror of trading.hod_break.consolidation_low with the level SUBSTITUTED: bars
    j-K+1..j must sit within consol_pct BELOW the (fixed) PDH (their highs <= PDH, their min low >= PDH x
    (1 - consol_pct)) -> return their min low (the stop); else None. The original computes the level (running
    HOD) INSIDE the function — here the level is an argument.
  * `arm_state_pdh` — cell 1,438's arm_state with the level fixed at PDH: armed at the close of bar j iff the
    running high through bar j is BELOW PDH (the break has not printed yet -- this alone gives "one entry per
    symbol-day, first cross only", since a bar that touches/exceeds PDH permanently disarms every later bar),
    PDH >= open x (1 + min_dist_open_pct%), PDH >= floor + 1c, the mirrored consolidation holds, rv_profile in
    band, m[j]+1 <= last_entry_minute, and (trigger - stop) / trigger >= min_r_pct%. trigger = PDH + $0.01,
    limit = PDH x (1 + LIMIT_BPS), exactly as cell 1,438.
  * `armed_crossing_bars_pdh` — mirror of ca.armed_crossing_bars with a session-constant trigger (no running
    HOD to recompute per bar).
  * `build_trading_calendar` / `prior_trading_day` — the trading-day sequence is universe.csv's own sorted
    distinct `day` values (real NYSE sessions the population already spans) — NOT a bars_sip.db distinct-day
    scan (that table has no index on `day` and a full scan is impractically slow; the calendar only decides
    WHICH day to look up in bars_sip.db, the presence check itself is always against bars_sip.db).
  * `load_pdh_table` — one batched bars_sip.db query per (prior) day for exactly the symbols needed that day;
    PDH = max high of that symbol's RTH bars on the prior day. A symbol absent from that day's result is
    'no_pdh' for every population row keyed to it.
  * `load_population_pdh` — cell 1,438's load_population's base filters (date range, adv20, test-ticker
    exclusion) THEN the PDH-implied daily-bar filter: day high >= PDH + 1c (the trigger must print), PDH >=
    open x (1 + min_dist%), PDH >= floor + 1c. No condition is added that is not implied by the arming rule
    (9/25 cell 1,439 was voided for exactly that).

Pass bar (frozen, identical to 1,439's / 1,438's VAL bar): fill mean net R >= +0.10 both holdouts (TRAIN-H2,
VAL), VAL day-clustered t >= 2, ex-top-5% > 0, coverage >= 80%, winner/loser missingness gap <= 5pp, >= 3
fills/week (sr.score's slot-simulated fills_wk). Coverage < 80% on either holdout VOIDS the cell.
"""
import argparse
import bisect
import gzip
import os
import pickle
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sip_rebuild as sr                                    # noqa: E402
import causal_arming as ca                                  # noqa: E402 — window_ns, fetch_window, load_day_bars,
                                                              #   live_params, resolve_window, resolve_day,
                                                              #   proxy_outcome, stop_slip_net, _legs, BARS_SIP_URI

ROOT = sr.ROOT
sys.path.insert(0, ROOT)
from trading.hod_break import rv_profile                    # noqa: E402 — side/level-agnostic (volume only)

UNIVERSE_CSV = os.path.join(ROOT, 'research/bf_zero/universe.csv')
CACHE_DIR = os.path.join(HERE, 'sip_cache_pdh')
OUT_CSV = os.path.join(HERE, 'cell_1441_fills.csv')
REPORT_MD = os.path.join(HERE, 'RESULT_1441.md')
SPLITS = {'TRAIN': ('2025-07-01', '2025-12-31'), 'VAL': ('2026-01-01', '2026-05-31')}
TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')
log = sr.log
os.makedirs(CACHE_DIR, exist_ok=True)


# --------------------------------------------------------------------------------------------- PDH arming
def consolidation_low_pdh(l, h, j, p, level):
    """Mirror of trading.hod_break.consolidation_low with the level SUBSTITUTED for the running HOD: bars
    j-K+1..j must hold within consol_pct BELOW `level` (their highs <= level, their min low >= level x
    (1 - consol_pct)) -> return their min low (the stop); else None. `j` is the last CLOSED bar before the
    break bar."""
    if j + 1 < p.consol_bars + 1:
        return None
    lows = np.asarray(l[j + 1 - p.consol_bars: j + 1], dtype=float)
    highs = np.asarray(h[j + 1 - p.consol_bars: j + 1], dtype=float)
    lo = float(lows.min())
    if float(highs.max()) <= level and lo >= level * (1.0 - p.consol_pct) and lo < level:
        return lo
    return None


def arm_state_pdh(o, h, l, v, m, j, adv20, p, floor, pdh, use_rv=True):
    """Mirror of causal_arming.arm_state with a FIXED level = the prior session's RTH high (PDH). Returns
    dict(level, trigger, limit, stop) or None. The `running_hi < pdh` guard is what gives "armed iff the break
    has not happened yet, one entry per symbol-day at the first cross": once a bar's high reaches PDH every
    later bar's running high is >= PDH too, so re-arming is impossible."""
    if int(m[j]) + 1 > p.last_entry_minute:
        return None
    running_hi = float(np.max(h[: j + 1]))
    if running_hi >= pdh - 1e-9:
        return None
    if pdh < floor + sr.TICK or pdh < float(o[0]) * (1.0 + p.min_dist_open_pct / 100.0):
        return None
    stop = consolidation_low_pdh(l, h, j, p, pdh)
    if stop is None:
        return None
    if use_rv:
        rv = rv_profile(float(np.sum(v[: j + 1])), adv20, int(m[j]))
        if not (p.rv_lo <= rv < p.rv_hi):
            return None
    trigger = round(pdh + sr.TICK, 6)
    if trigger < floor or (trigger - stop) / trigger * 100.0 < p.min_r_pct:
        return None
    return dict(level=pdh, trigger=trigger, limit=pdh * (1.0 + sr.LIMIT_BPS), stop=stop)


def armed_crossing_bars_pdh(bars, adv20, pdh, p, floor, use_rv=True):
    """Armed bars j whose next bar's minute high reaches the (session-constant) trigger — mirror of
    ca.armed_crossing_bars with a fixed level."""
    o, h, l, v = (bars[k].values.astype(float) for k in ('o', 'h', 'l', 'v'))
    m = bars.m.values.astype(int)
    trigger_guess = round(pdh + sr.TICK, 6)
    out = []
    for j in np.nonzero(h[1:] >= trigger_guess - 1e-9)[0]:
        a = arm_state_pdh(o, h, l, v, m, int(j), adv20, p, floor, pdh, use_rv)
        if a is not None and h[j + 1] >= a['trigger'] - 1e-9:
            out.append(dict(a, j=int(j), m_lo=int(m[j]), m_hi=int(m[j + 1]), cumv_j=float(np.sum(v[: j + 1]))))
    return out


# --------------------------------------------------------------------------------------------- PDH table
def build_trading_calendar():
    """Sorted distinct trading days from universe.csv (real NYSE sessions the population spans) — used only
    to find WHICH prior day to look up in bars_sip.db; presence is always checked against bars_sip.db."""
    u = pd.read_csv(UNIVERSE_CSV, usecols=['bar_date'])
    return sorted(u.bar_date.unique().tolist())


def prior_trading_day(calendar, day):
    """The calendar day immediately before `day`, or None if `day` is the first (or absent)."""
    i = bisect.bisect_left(calendar, day)
    if i == 0:
        return None
    return calendar[i - 1]


def load_pdh_table(sipcon, days, symbols_by_day, calendar):
    """{(day, symbol): PDH} from bars_sip.db — one batched query per PRIOR day for exactly the symbols needed
    that day. A symbol with no rows on the prior day (holiday gap in bars_sip.db, new listing, etc.) is simply
    absent from the returned dict -> 'no_pdh' for the caller."""
    out = {}
    by_prior = {}
    for day in days:
        prev = prior_trading_day(calendar, day)
        if prev is not None:
            by_prior.setdefault(prev, []).append(day)
    for i, (prev, days_using_it) in enumerate(sorted(by_prior.items())):
        syms = sorted({s for day in days_using_it for s in symbols_by_day[day]})
        if not syms:
            continue
        q = (f"select symbol, t, o, h, l, c, v from bars where day=? and symbol in "
             f"({','.join('?' * len(syms))})")
        t = pd.read_sql(q, sipcon, params=[prev] + syms)
        pdh_by_sym = {}
        for s, g in t.groupby('symbol'):
            b = ca._rth(g, 't')
            if len(b):
                pdh_by_sym[s] = float(b.h.max())
        for day in days_using_it:
            for s in symbols_by_day[day]:
                if s in pdh_by_sym:
                    out[(day, s)] = pdh_by_sym[s]
        if (i + 1) % 20 == 0:
            log(f'[pdh] prior-day lookups {i + 1}/{len(by_prior)}')
    return out


# --------------------------------------------------------------------------------------------- population
def load_population_pdh(p, floor, min_adv, sipcon):
    """cell 1,438's base filters (date range, adv20, test-ticker exclusion) THEN the PDH-implied filter: day
    high >= PDH + 1c (the trigger must print), PDH >= open x (1 + min_dist%), PDH >= floor + 1c. No condition
    beyond what the arming rule implies (9/25 cell 1,439's look-ahead lesson)."""
    u = pd.read_csv(UNIVERSE_CSV, dtype={'symbol': str}, keep_default_na=False)
    for c in ('open', 'high', 'low', 'adv20'):
        u[c] = pd.to_numeric(u[c], errors='coerce')
    u = u.rename(columns={'bar_date': 'day'})
    u = u[(u.day >= SPLITS['TRAIN'][0]) & (u.day <= SPLITS['VAL'][1])]
    u = u[u.adv20 >= min_adv]
    u = u[~u.symbol.str.match(TEST_TICKER)].drop_duplicates(['day', 'symbol'])
    n_base = len(u)
    calendar = build_trading_calendar()
    symbols_by_day = {d: g.symbol.tolist() for d, g in u.groupby('day')}
    pdh_map = load_pdh_table(sipcon, sorted(u.day.unique()), symbols_by_day, calendar)
    u['pdh'] = [pdh_map.get((d, s), np.nan) for d, s in zip(u.day, u.symbol)]
    n_no_pdh = int(u.pdh.isna().sum())
    log(f'[pop] {n_no_pdh}/{n_base} base symbol-days have no prior-session bars_sip.db data -> no_pdh (excluded)')
    u = u[u.pdh.notna()].copy()
    u = u[(u.high >= u.pdh + sr.TICK) & (u.pdh >= u.open * (1 + p.min_dist_open_pct / 100)) &
          (u.pdh >= floor + sr.TICK)]
    u['split'] = np.where(u.day <= SPLITS['TRAIN'][1], 'TRAIN', 'VAL')
    u['wk'] = pd.to_datetime(u.day).dt.to_period('W-FRI').astype(str)
    log(f'[pop] {len(u)} symbol-days in the PDH causal superset (TRAIN-H2 {int((u.split == "TRAIN").sum())}, '
        f'VAL {int((u.split == "VAL").sum())})')
    return u[['day', 'symbol', 'adv20', 'split', 'wk', 'pdh']].reset_index(drop=True), n_no_pdh, n_base


# --------------------------------------------------------------------------------------------- run
def process_symbol_day(r, bars, p, floor, cache, new):
    """One symbol-day. Returns a list of result dicts (0 or 1 rows). Fill mechanics (resolve_day / walk_path /
    trade_result) are cell 1,438's, unchanged — only the arming (armed_crossing_bars_pdh) differs."""
    def tape_for(a):
        k = f"{r.symbol}|{a['m_lo']}|{a['m_hi']}"
        if k in cache:
            return cache[k]
        if k in new:
            return new[k]
        new[k] = ca.fetch_window(r.symbol, r.day, a['m_lo'], a['m_hi'])
        return new[k]

    cands = armed_crossing_bars_pdh(bars, r.adv20, r.pdh, p, floor, use_rv=True)
    base = dict(day=r.day, symbol=r.symbol, split=r.split, wk=r.wk, pdh=r.pdh, n_cross=len(cands))
    if not cands:
        return [{**base, 'status': 'not_armed'}]
    base['b0_net_R'] = ca.proxy_outcome(cands, bars)
    status, a, e = ca.resolve_day(cands, tape_for, r.day, r.adv20, p, tick_rv=False)
    if status != 'fill':
        return [{**base, 'status': status}]
    fill, stop = e['fill'], a['stop']
    R = fill - stop
    fill_min = sr.ns_to_et_minutes(e['fill_ts'], r.day)
    if e['stopped']:
        exit_m, px, why = fill_min, stop, 'stop_bar'
    else:
        path = bars[(bars.m > a['m_hi']) & (bars.m <= sr.EOD_M)]
        if not len(path):
            log(f'[sim] WARNING {r.day} {r.symbol}: no minute bars after the fill bar — exit at the fill (0 R raw)')
            exit_m, px, why = fill_min, fill, 'no_path'
        else:
            exit_m, px, why = sr.walk_path(fill, stop, fill + sr.TARGET_R * R, path)
    raw, cost, net, _ = sr.trade_result(fill, stop, e['half_spread'], px, e['half_spread'])
    return [{**base, 'status': 'fill', 'fill': fill, 'stop': stop, 'level': a['level'], 'fill_min': fill_min,
             'exit_m': exit_m, 'exit_price': px, 'why': why, 'R': R, 'raw_R': raw, 'cost_R': cost, 'net_R': net}]


def run(workers=2, day_limit=None, time_budget_s=None):
    """Scan, fetch (resumable per-day cache sip_cache_pdh/c1441_{day}.pkl.gz), resolve, write CSV."""
    t0 = time.time()
    p, floor, min_adv = ca.live_params()
    log(f'[params] {p} | min_price {floor} | min_adv20 {min_adv}')
    con = sqlite3.connect(sr.CACHE_DB_URI, uri=True, timeout=120)
    sipcon = sqlite3.connect(ca.BARS_SIP_URI, uri=True, timeout=120)
    pop, n_no_pdh, n_base = load_population_pdh(p, floor, min_adv, sipcon)
    rows, n_nobars, src = [], 0, {}
    days = sorted(pop.day.unique())
    if day_limit:
        days = days[:day_limit]
    for di, day in enumerate(days):
        if time_budget_s and time.time() - t0 > time_budget_s:
            log(f'[run] time budget hit after {len(rows)} symbol-days — stopping (resumable)')
            break
        sub = pop[pop.day == day]
        bars = ca.load_day_bars(con, day, sub.symbol.tolist(), sipcon, src)
        path = os.path.join(CACHE_DIR, f'c1441_{day}.pkl.gz')
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
                except Exception as exc:                          # noqa: BLE001
                    log(f'[run] ERROR symbol-day failed: {exc} — LOST (re-run to resume)')
        if new_all:
            cache.update(new_all)
            tmp = path + '.tmp'
            with gzip.open(tmp, 'wb') as f:
                pickle.dump(cache, f)
            os.replace(tmp, path)
        if (di + 1) % 10 == 0 or di == len(days) - 1:
            log(f'[run] day {di + 1}/{len(days)} ({day}): {len(sub)} symbol-days, cache +{len(new_all)}, '
                f'total rows {len(rows)}')
    con.close(); sipcon.close()
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    log(f'[run] wrote {OUT_CSV} ({len(df)} rows, {n_nobars} <K+2 bars, {n_no_pdh}/{n_base} no_pdh, src={src})')
    return df, n_nobars, n_no_pdh, n_base


def _legs(v):
    return ca._legs(v)


def report(res, n_nobars, n_no_pdh, n_base):
    """RESULT_1441.md, SHORT format (owner): one row per split + <= 5 lines."""
    sc = {}
    for sp in ('TRAIN', 'VAL'):
        allv = res[res.split == sp]
        sc[sp] = sr.score(allv[allv.status != 'not_armed'].copy(), allv)
    lines = ['# RESULT — cell 1,441: prior-day-high (PDH) break, same order (causal arming) — TEST NOT read', '',
             '| split | fills (rate) | mean net R | day-clust t | ex-top-5 % | fills/wk | coverage / gap | verdict |',
             '|---|---|---|---|---|---|---|---|']
    for sp, cn in (('TRAIN', 'TRAIN-H2'), ('VAL', 'VAL')):
        v = sc[sp]
        legs = _legs(v)
        verdict = ('PASS' if all(x[1] for x in legs) else 'FAIL: ' + ', '.join(n for n, g, _ in legs if not g)) \
            if sp == 'VAL' else 'report-only'
        lines.append(f'| {cn} | {v["fills"]} ({v["fill_rate"] * 100:.1f} %) | {v["mean_R"]:+.3f} | {v["t"]:.2f} | '
                      f'{v["extop5"]:+.3f} | {v["fills_wk"]:.1f} | {v["coverage"] * 100:.1f} % / '
                      f'{v["miss_gap"] * 100:.1f} pp | {verdict} |')
    sl = []
    for sp, cn in (('TRAIN', 'TRAIN-H2'), ('VAL', 'VAL')):
        f = res[(res.split == sp) & (res.status == 'fill')]
        sl.append(f'{cn} {ca.stop_slip_net(f).mean():+.3f}' if len(f) else f'{cn} n/a')
    lines += ['', f'* 30 bps stop-slip (every stop exit 0.3 % worse): mean net R {", ".join(sl)}.',
              f'* Scope: superset symbol-days (day high >= PDH+1c, PDH >= open x (1+min_dist), PDH >= floor+1c); '
              f'{n_no_pdh}/{n_base} base symbol-days had no prior-session bars_sip.db data (no_pdh, excluded); '
              f'{n_nobars} symbol-days had < K+2 minute bars. Coverage / gap use the proxy winner (fill at the '
              f'trigger on the first crossing armed bar; B0 path physics).', '']
    with open(REPORT_MD, 'w') as fh:
        fh.write('\n'.join(lines))
    log('[report] VAL ' + '; '.join(f'{n}={x}' for n, _, x in _legs(sc['VAL'])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--day-limit', type=int, default=None)
    ap.add_argument('--time-budget-s', type=float, default=None)
    a = ap.parse_args(argv)
    df, n_nobars, n_no_pdh, n_base = run(a.workers, a.day_limit, a.time_budget_s)
    report(df, n_nobars, n_no_pdh, n_base)


if __name__ == '__main__':
    main()
