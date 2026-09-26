"""Independent rebuild of cells 1,479 (PYRAMID) and 1,480 (FAILED-BREAK FLIP SHORT).

Built from research/hod_entry/PREREG_1478.md prose ONLY. This agent did NOT open cell_1479.py,
cell_1480.py or RESULT_1479_1480.md — per the CLAUDE.md independent-reimplementation rule.

Reused (read, not guessed): sip_rebuild.walk_path / trade_result / prevailing_quote / fetch_tape /
_client / sig_key / EOD_M / TARGET_R / SLIP_BP / TICK ; causal_arming.load_day_bars / window_ns ;
score_1439.ssr_today_hit / ssr_carried_hit / prior_close_map ; cell_1439.load_borrow_flags
(BORROW_CSV = research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv, same file cell 1,431/1,439 use).

Base book: causal_arming_causal.csv, variant=='causal' & status=='fill' (9,911 rows).

--------------------------------------------------------------------------------------------------
CELL 1,479 — PYRAMID (bars-only, no network; local cache.db / bars_sip.db read-only)
1/3 risk at the break (already-filled `fill`, `stop`, R0 = fill-stop). Walk minute bars AFTER the
break bar (m > floor(fill_min)) with sip_rebuild.walk_path's own priority convention (EOD-bar exit
at its open; stop before target when both touch the same bar — kept here as "stop-before-add" too,
the same pessimism). If a bar's high >= fill+1R *before* the (pre-add) stop: add 2/3 at fill+1R,
move stop to `fill` (breakeven), and — since the add is not itself an exit — re-check the SAME bar
against the new stop/target before moving on. Target stays fill+2R for the combined position
(TARGET_R=2 is already sip_rebuild's constant, so cell 1,479 changes SIZE/STOP only, not target).
`stop_bar` fills (stopped inside the break bar, 76 of 9,911) never get a chance to add: pyramid_R =
-1/3 exactly (only the first third was ever on).

R-in-original-units algebra (from the PREREG's own worked breakeven case): with N shares as the
non-pyramid size, leg1 = 1/3 N @ fill, leg2 = 2/3 N @ fill+R0. At an eventual exit price X (post-add):
  combined $ = (X-fill)*(1/3 N) + (X-fill-R0)*(2/3 N)  =>  combined_R = (X-fill)/R0 - 2/3
Check: X=fill (new stop) -> -2/3 R ("breakeven for the first third, -1R on the added two-thirds",
i.e. -1 * 2/3 weight = -0.667R -- matches). X=fill+2R0 (target) -> +1.333 R (less than the
un-pyramided +2R, because only 1/3 size was on for the first R of the move -- the expected
trade-off of this rule). Pre-add exit: combined_R = (X-fill)/R0 * (1/3).

COST (disclosed approximation -- no half_entry/exit_half columns survive into the base CSV, only
the bundled cost_R = (half_entry+exit_half+SLIP_BP*exit_price)/R0; SLIP_BP is a known constant so
the spread-only remainder is recoverable and split 50/50 as a stand-in for entry vs exit):
  slip_R      = SLIP_BP * exit_price_mine / R0                      (SLIP_BP=0.0002, sip_rebuild)
  spread_R    = max(cost_R_base - SLIP_BP*exit_price_base/R0, 0) / 2      (per-share half-spread proxy)
  pre-add:  cost_R = spread_R*2*(1/3) + slip_R                       (entry+exit halves, 1/3 size)
  post-add: cost_R = spread_R*2 + spread_R*(2/3) + slip_R            (leg1 entry+full exit, PLUS a
                     second entry half-spread on the 2/3 add -- "charge the measured spread again")
Verified stop-limit slip (cell 1,463 unbiased, RESULT_1463_unbiased.md): the 20 bps-limit order's
probability-weighted mean slip, applied on why in {stop, stop_be}:
  TRAIN-H2: 0.861*2.9bps + 0.139*93.9bps = 15.55 bps ;  VAL: 0.894*3.2bps + 0.106*75.8bps = 10.92 bps
(these replace the old flat 30 bps STOP_SLIP; applied as extra_slip_R = bps*1e-4*exit_price/R0).

Pass bar (report-only here, not scored against anything): PREREG's own bar is ΔR vs base >= +0.05
both holdouts, VAL t>=2.5, kept mean >= +0.10 VAL.

--------------------------------------------------------------------------------------------------
CELL 1,480 — FAILED-BREAK FLIP SHORT (needs SIP ticks; no research/hod_entry/sip_cache_1480/ exists,
so this fetches fresh via sip_rebuild.fetch_tape, capped at FETCH_CAP fills -- a SAMPLE, not the
full 9,911; reported as such.)
For each sampled fill: fetch trades+quotes for [fill_ts, fill_ts+15min). If a trade print <=
level-0.01 occurs before the original 2R target is reached (checked against the SAME bars used for
1,479's plain-book walk, i.e. re-walked with sip_rebuild.walk_path on the base fill/stop/target,
no pyramid): that print is the new long exit (long ΔR reported vs the base book's own net_R for
that row) and a short opens at the prevailing NBBO bid at that print. short_stop = break-bar high +
0.01 ; short_target = short_entry - 2*(short_stop-short_entry) ; cover at 15:55 via a mirrored
walk on minute bars (stop = price >= short_stop, target = price <= short_target, EOD = bar open,
stop-priority first). Cost: entry at the bid (no extra charge, per spec); exit (cover) at the ask,
modeled as the SAME spread_R proxy defined above (half-spread, unscaled); stop-type short exits get
a flat 35 bps fallback slip (PREREG: "35 bps fallback where unmeasured" -- no cell has measured the
short-stop fill rate the way 1,463 measured the long stop, so the fallback IS the number here).
shortable / SSR: shortable from cell_1439.load_borrow_flags's borrow_flags.csv (today's-snapshot,
same caveat as 1,431/1,439); SSR via score_1439.ssr_today_hit/ssr_carried_hit against daily_bars in
data/cache.db (same `con`) -- SSR=True rows are computed and flagged but EXCLUDED from the primary
book per spec, not dropped from the file.
"""
import os
import sys
import pickle
import sqlite3
import time
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import sip_rebuild as sr  # noqa: E402
import causal_arming as ca  # noqa: E402
import score_1439 as s1439  # noqa: E402

BASE_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
OUT_CSV = os.path.join(HERE, 'rebuild_1479_1480.csv')
LOG_PATH = os.path.join(HERE, 'rebuild_1479_1480.log')
DONE_MARK = os.path.join(HERE, 'rebuild_1479_1480.DONE')
BORROW_CSV = os.path.join(ROOT, 'research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv')

FETCH_CAP = 500

# cell 1,463 unbiased (RESULT_1463_unbiased.md) -- probability-weighted mean stop-limit-20bps slip
STOP_LIMIT_BLENDED_BPS = {'TRAIN': 0.861 * 2.9 + 0.139 * 93.9, 'VAL': 0.894 * 3.2 + 0.106 * 75.8}
SHORT_STOP_FALLBACK_BPS = 35.0   # PREREG 1,480: "35 bps fallback where unmeasured"


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def load_base():
    df = pd.read_csv(BASE_CSV, low_memory=False)
    f = df[(df.variant == 'causal') & (df.status == 'fill')].copy()
    if len(f) != 9911:
        log(f'[WARN] expected 9911 fill rows, got {len(f)}')
    return f.reset_index(drop=True)


# --------------------------------------------------------------------------------------- cell 1,479
def pyramid_walk(fill, stop, path):
    """path: bars with m > break bar's minute, m<=EOD_M, sorted. Returns dict(exit_m, exit_price,
    why, added, raw_R) with raw_R in ORIGINAL-R units (see module docstring algebra)."""
    R0 = fill - stop
    add_th = fill + R0
    target = fill + sr.TARGET_R * R0
    added = False
    cur_stop = stop
    eps = 1e-9
    for row in path.itertuples():
        m, o, h, l, c = row.m, row.o, row.h, row.l, row.c
        if m >= sr.EOD_M:
            X = float(o)
            raw = (X - fill) / R0 * (1 / 3) if not added else (X - fill) / R0 - 2 / 3
            return dict(exit_m=int(m), exit_price=X, why='eod', added=added, raw_R=raw)
        if l <= cur_stop + eps:
            X = float(o) if o <= cur_stop else float(cur_stop)
            why = 'stop' if not added else 'stop_be'
            raw = (X - fill) / R0 * (1 / 3) if not added else (X - fill) / R0 - 2 / 3
            return dict(exit_m=int(m), exit_price=X, why=why, added=added, raw_R=raw)
        if not added and h >= add_th - eps:
            added = True
            cur_stop = fill
            if l <= cur_stop + eps:
                X = float(o) if o <= cur_stop else float(cur_stop)
                raw = (X - fill) / R0 - 2 / 3
                return dict(exit_m=int(m), exit_price=X, why='stop_be', added=True, raw_R=raw)
            if h >= target - eps:
                raw = (target - fill) / R0 - 2 / 3
                return dict(exit_m=int(m), exit_price=float(target), why='target', added=True, raw_R=raw)
            continue
        if added and h >= target - eps:
            raw = (target - fill) / R0 - 2 / 3
            return dict(exit_m=int(m), exit_price=float(target), why='target', added=True, raw_R=raw)
    last = path.iloc[-1]
    X = float(last.c)
    raw = (X - fill) / R0 * (1 / 3) if not added else (X - fill) / R0 - 2 / 3
    return dict(exit_m=int(last.m), exit_price=X, why='eod_fallback', added=added, raw_R=raw)


def cost_1479(raw_row, mine, split):
    """Disclosed-approximation cost, see module docstring. Returns (cost_R, net_R)."""
    R0 = raw_row.fill - raw_row.stop
    slip_base_R = sr.SLIP_BP * raw_row.exit_price / R0 if np.isfinite(raw_row.exit_price) else 0.0
    spread_R = max(raw_row.cost_R - slip_base_R, 0.0) / 2.0
    slip_mine_R = sr.SLIP_BP * mine['exit_price'] / R0
    stop_type = mine['why'] in ('stop', 'stop_be')
    extra_slip_R = (STOP_LIMIT_BLENDED_BPS.get(split, STOP_LIMIT_BLENDED_BPS['VAL']) * 1e-4
                     * mine['exit_price'] / R0) if stop_type else 0.0
    if not mine['added']:
        cost_R = spread_R * 2 * (1 / 3) + slip_mine_R + extra_slip_R
    else:
        cost_R = spread_R * 2 + spread_R * (2 / 3) + slip_mine_R + extra_slip_R
    return cost_R, mine['raw_R'] - cost_R


def run_1479(base, con, sipcon, validate_n=200):
    """Returns a DataFrame indexed like `base` with 1,479 columns, plus a validation report printed
    to the log (re-walk of the PLAIN base-book trade on the same bars, compared to the base CSV's
    own exit_m/exit_price/why/raw_R -- a correctness gate on this script's bar-loading, independent
    of any builder file)."""
    rows = []
    n_val_ok, n_val_bad, n_val_checked = 0, 0, 0
    for day, g in base.groupby('day'):
        syms = sorted(g.symbol.unique())
        bars = ca.load_day_bars(con, day, syms, sipcon, {})
        for r in g.itertuples():
            b = bars.get(r.symbol)
            m_break = int(np.floor(r.fill_min))
            if r.why == 'stop_bar' or b is None or not len(b):
                mine = dict(exit_m=m_break, exit_price=r.stop, why='stop_bar', added=False,
                            raw_R=-1 / 3)
            else:
                path = b[(b.m > m_break) & (b.m <= sr.EOD_M)]
                if not len(path):
                    mine = dict(exit_m=m_break, exit_price=r.fill, why='no_path', added=False, raw_R=0.0)
                else:
                    mine = pyramid_walk(r.fill, r.stop, path)
                    if n_val_checked < validate_n:
                        n_val_checked += 1
                        target = r.fill + sr.TARGET_R * (r.fill - r.stop)
                        base_walk = sr.walk_path(r.fill, r.stop, target, path)
                        ok = (base_walk[2] == r.why and abs(base_walk[1] - r.exit_price) < 0.02)
                        n_val_ok += int(ok)
                        n_val_bad += int(not ok)
            cost_R, net_R = cost_1479(r, mine, r.split)
            rows.append(dict(day=r.day, symbol=r.symbol, split=r.split, fill_min=r.fill_min,
                              base_why=r.why, base_net_R=r.net_R,
                              c1479_exit_m=mine['exit_m'], c1479_exit_price=mine['exit_price'],
                              c1479_why=mine['why'], c1479_added=mine['added'],
                              c1479_raw_R=mine['raw_R'], c1479_cost_R=cost_R, c1479_net_R=net_R))
    log(f'[1479] bar-walk self-check (plain re-walk of the BASE trade, vs the base CSV, '
        f'n={n_val_checked}): {n_val_ok} match / {n_val_bad} mismatch '
        f'({100 * n_val_ok / max(n_val_checked, 1):.1f}% -- validates load_day_bars+walk_path use, '
        f'not the builder)')
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- cell 1,480
def walk_short(entry, stop, target, path):
    """Mirror of sip_rebuild.walk_path for a SHORT: gap-through at the open, high>=stop -> stop
    (stop-priority when a bar touches both), low<=target -> target, 15:55 -> its open."""
    for row in path.itertuples():
        m, o, h, l, c = row.m, row.o, row.h, row.l, row.c
        if m >= sr.EOD_M:
            return int(m), float(o), 'eod'
        if h >= stop - 1e-9:
            return int(m), float(o if o >= stop else stop), 'stop'
        if l <= target + 1e-9:
            return int(m), float(target), 'target'
    last = path.iloc[-1]
    return int(last.m), float(last.c), 'eod_fallback'


SIP_CACHE_1480 = os.path.join(HERE, 'sip_cache_1480')


def _cache_path(symbol, day, m_break):
    return os.path.join(SIP_CACHE_1480, f'{symbol}_{day}_{int(m_break)}.pkl')


def cached_fetch_tape(symbol, day, m_break, S_ns, new_fetch_budget, fetch_cap=FETCH_CAP, retries=3, span_s=900):
    """Reuse research/hod_entry/sip_cache_1480/<symbol>_<day>_<m_break>.pkl when present (per-task
    instruction: the tape window is raw market data, already fetched once for this same population --
    reusing it is not reading the target's code or logic). Only a cache MISS counts against the
    500-new-fetch cap (new_fetch_budget, a 1-element list so the caller can inspect it); a cache hit
    is free and uncapped, so the sample is (all cache hits) + (up to 500 fresh fetches)."""
    p = _cache_path(symbol, day, m_break)
    if os.path.exists(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    if new_fetch_budget[0] >= fetch_cap:
        raise RuntimeError('new-fetch cap reached (cache miss, budget exhausted)')
    new_fetch_budget[0] += 1
    return sr.fetch_tape(symbol, S_ns, retries=retries, span_s=span_s)


def find_failed_break(symbol, day, fill_ts, level, target, R0, fill, stop, bars, m_break, new_fetch_budget,
                       fetch_cap=FETCH_CAP):
    """Fetch [fill_ts, fill_ts+15min) ticks; return None or dict(print_ts, print_px, bid, ask) for
    the first print <= level-0.01 that occurs strictly before the base-book plain walk would have
    hit the +2R target (walked on minute bars, sip_rebuild.walk_path semantics, no pyramid)."""
    path = bars[(bars.m > m_break) & (bars.m <= sr.EOD_M)]
    if len(path):
        _, _, why0 = sr.walk_path(fill, stop, target, path)
    else:
        why0 = 'no_path'
    S_ns = fill_ts + 900 * 10 ** 9
    try:
        t, q = cached_fetch_tape(symbol, day, m_break, S_ns, new_fetch_budget, fetch_cap=fetch_cap,
                                  retries=3, span_s=900)
    except Exception as e:  # noqa: BLE001 -- network/cap; caller counts LOST
        return 'fetch_error', str(e)
    w = t[(t.ts >= fill_ts) & (t.ts < S_ns)].sort_values('ts', kind='stable')
    hit = w[w.price <= level - 0.01 + 1e-9]
    if not len(hit):
        return None, why0
    t_hit = int(hit.ts.iloc[0])
    pq = sr.prevailing_quote(q, t_hit)
    if pq is None:
        return 'no_quote', why0
    bid, ask = pq
    return dict(print_ts=t_hit, print_px=float(hit.price.iloc[0]), bid=bid, ask=ask), why0


def run_1480(base, con, sipcon, fetch_cap=FETCH_CAP):
    """Cache-first scan: EVERY non-stop_bar candidate is attempted (cache hits under sip_cache_1480/
    are free and uncapped -- reusing an already-fetched raw tape window is not reading the target's
    code); only a cache MISS spends the 500-new-fetch budget, after which further misses are skipped
    (logged, not counted as fetch_error) rather than truncating the whole scan."""
    borrow = None
    if os.path.exists(BORROW_CSV):
        borrow = pd.read_csv(BORROW_CSV).drop_duplicates('symbol').set_index('symbol')
    rows = []
    n_cache_hit, n_new_fetch, n_skipped_cap, n_error, n_flip = 0, 0, 0, 0, 0
    new_fetch_budget = [0]
    days = sorted(base.day.unique())
    for day in days:
        g = base[base.day == day]
        syms = sorted(g.symbol.unique())
        bars = ca.load_day_bars(con, day, syms, sipcon, {})
        prior_close = s1439.prior_close_map(con, syms, days)
        for r in g.itertuples():
            b = bars.get(r.symbol)
            if b is None or not len(b) or r.why == 'stop_bar':
                continue
            m_break = int(np.floor(r.fill_min))
            fill_ts = ca.window_ns(r.day, m_break, m_break)[1]  # end of the break bar (fill instant proxy)
            R0 = r.fill - r.stop
            target = r.fill + sr.TARGET_R * R0
            was_cached = os.path.exists(_cache_path(r.symbol, r.day, m_break))
            budget_before = new_fetch_budget[0]
            res, why0 = find_failed_break(r.symbol, r.day, fill_ts, r.level, target, R0, r.fill, r.stop,
                                           b, m_break, new_fetch_budget, fetch_cap=fetch_cap)
            if was_cached:
                n_cache_hit += 1
            elif new_fetch_budget[0] > budget_before:
                n_new_fetch += 1
            elif res == 'fetch_error' and 'cap reached' in str(why0):
                n_skipped_cap += 1
                continue
            if res in (None, 'fetch_error', 'no_quote'):
                if res == 'fetch_error' and 'cap reached' not in str(why0):
                    n_error += 1
                    log(f'[1480] fetch_error {r.day} {r.symbol}: {why0}')
                continue
            n_flip += 1
            long_exit = res['print_px']
            long_raw_R = (long_exit - r.fill) / R0
            slip_base_R = sr.SLIP_BP * r.exit_price / R0 if np.isfinite(r.exit_price) else 0.0
            spread_R = max(r.cost_R - slip_base_R, 0.0) / 2.0
            long_cost_R = spread_R + spread_R + sr.SLIP_BP * long_exit / R0
            long_net_R = long_raw_R - long_cost_R
            short_entry = res['bid']
            brk_high = float(b[b.m == m_break].h.iloc[0]) if (b.m == m_break).any() else r.level
            # Amendment 2026-09-26 item 2 (R floor): short stop = max(break-bar high + $0.01,
            # entry x 1.01) -- a doji break bar made brk_high approx short_entry, so short_R approx 0
            # and short_net_R blew up (undefined mean in the pre-amendment rebuild, disclosed in the
            # PREREG amendment). The floor guarantees short_R >= 1% of entry.
            short_stop = max(brk_high + sr.TICK, short_entry * 1.01)
            short_R = short_stop - short_entry
            short_target = short_entry - 2 * short_R
            spath = b[(b.m > m_break) & (b.m <= sr.EOD_M)]
            if short_R <= 0 or not len(spath):
                continue
            s_exit_m, s_exit_px, s_why = walk_short(short_entry, short_stop, short_target, spath)
            short_raw_R = (short_entry - s_exit_px) / short_R
            s_stop_type = s_why == 'stop'
            s_extra_slip_R = (SHORT_STOP_FALLBACK_BPS * 1e-4 * s_exit_px / short_R) if s_stop_type else 0.0
            short_cost_R = spread_R + s_extra_slip_R
            short_net_R = short_raw_R - short_cost_R
            sym_borrow = borrow.loc[r.symbol] if (borrow is not None and r.symbol in borrow.index) else None
            shortable = bool(sym_borrow.shortable) if sym_borrow is not None else False
            pc = None
            cand = sorted([dt for (s, dt) in prior_close if s == r.symbol and dt < r.day], reverse=True)
            if cand:
                pc = prior_close.get((r.symbol, cand[0]))
            ssr = True if pc is None else s1439.ssr_today_hit(b, r.fill_min, pc)
            rows.append(dict(day=r.day, symbol=r.symbol, split=r.split, fill_min=r.fill_min,
                              base_why=r.why, base_net_R=r.net_R, why0_plain=why0,
                              print_ts=res['print_ts'], print_px=res['print_px'],
                              long_exit=long_exit, long_raw_R=long_raw_R, long_net_R=long_net_R,
                              long_delta_R=long_net_R - r.net_R,
                              short_entry=short_entry, short_stop=short_stop, short_target=short_target,
                              short_exit_m=s_exit_m, short_exit_px=s_exit_px, short_why=s_why,
                              short_net_R=short_net_R, shortable=shortable, ssr=ssr)
                         )
    log(f'[1480] {n_cache_hit} cache hits + {n_new_fetch} new fetches (cap {fetch_cap}), '
        f'{n_skipped_cap} skipped (cap exhausted, cache miss), {n_flip} failed-break flips found, '
        f'{n_error} fetch errors')
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='tiny subset, foreground sanity check')
    ap.add_argument('--fetch-cap', type=int, default=FETCH_CAP)
    args = ap.parse_args()

    log('=== rebuild_1479_1480 start ===' + (' (SMOKE)' if args.smoke else ''))
    base = load_base()
    if args.smoke:
        base = base.head(60).copy()

    con = sqlite3.connect(sr.CACHE_DB_URI, uri=True, timeout=120)
    sipcon = sqlite3.connect(ca.BARS_SIP_URI, uri=True, timeout=120)

    log(f'[1479] walking {len(base)} fills...')
    r1479 = run_1479(base, con, sipcon, validate_n=300)
    log(f'[1479] done: {len(r1479)} rows')

    fcap = 8 if args.smoke else args.fetch_cap
    log(f'[1480] fetching ticks, cap={fcap}...')
    r1480 = run_1480(base, con, sipcon, fetch_cap=fcap)
    log(f'[1480] done: {len(r1480)} flip rows')

    merged = r1479.merge(
        r1480[['day', 'symbol', 'fill_min', 'long_delta_R', 'short_net_R', 'shortable', 'ssr',
               'short_why']],
        on=['day', 'symbol', 'fill_min'], how='left')
    merged.to_csv(OUT_CSV, index=False)
    log(f'[done] wrote {OUT_CSV} ({len(merged)} rows)')

    for sp in ('TRAIN', 'VAL'):
        m1479 = r1479[r1479.split == sp].c1479_net_R
        log(f'[1479] {sp}: n={len(m1479)} mean_net_R={m1479.mean():+.4f} base_mean={r1479[r1479.split==sp].base_net_R.mean():+.4f}')
        if sp in r1480.split.values:
            ms = r1480[r1480.split == sp].short_net_R
            ml = r1480[r1480.split == sp].long_delta_R
            log(f'[1480] {sp}: n={len(ms)} short_mean_net_R={ms.mean():+.4f} long_delta_mean={ml.mean():+.4f} '
                f'share_shortable={r1480[r1480.split==sp].shortable.mean():.2f} '
                f'share_ssr={r1480[r1480.split==sp].ssr.mean():.2f}')

    with open(DONE_MARK, 'w') as f:
        f.write('done\n')
    log('=== rebuild_1479_1480 END ===')


if __name__ == '__main__':
    main()
