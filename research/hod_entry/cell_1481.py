#!/usr/bin/env python3
"""Cells 1,481-1,482 -- research/hod_entry/PREREG_1481.md (FROZEN 2026-09-26).

BUY THE RETEST, NOT THE BREAK. For every one of the 9,911 base fills (cell 1,438), instead of
buying the ask at the break, a passive BUY LIMIT rests at level - $0.01 (1,481) or
level * (1 - 0.2%) (1,482, the deeper retest) from the base fill instant through the end of the
15th (1,481) / 30th (1,482) RTH minute after the fill bar. Fill rule: the first tape print
STRICTLY BELOW the limit after the base fill instant fills at the limit price (traded-through, the
passive-limit standard); a print exactly at the limit does not fill (report-only variant tracked
alongside: at-or-below). No print below anywhere in the window -> no trade (the never-retest
cohort is reported separately at its base net R, since the selection is real and must be visible).

Path: stop = the base's own consolidation low (unchanged); R' = entry - stop; target =
entry + 2 R'. Inside the retest minute the TAPE decides first (a print <= stop after the retest
fill = stopped; a print >= target = target -- whichever prints first); if neither happens in that
minute, `sip_rebuild.walk_path` walks the minute bars strictly after the retest bar through 15:55
(gap-through-at-the-open / stop-priority-on-a-touch semantics, unchanged).

Costs: entry is passive (no spread charged, per spec). Target is a limit (no slip, per spec).
Stop pays cell_1478's verified stop-limit standard (`cell_1478.SLIP_STOP_BPS`, the SAME
expected-value blend the base book itself uses: 0.88 * filled-stop bps + 0.12 * no-fill-tail bps,
per split). EOD (15:55) pays cell 1,443's measured EOD-exit slip (`RESULT_1443.md`'s "eod" row
means, 11.5 / 9.7 bps TRAIN-H2 / VAL) -- "exit at the bid" priced as that expected-value discount
off the bar's open, mirroring the stop leg's own EV-slip convention (no separate NBBO exists for an
arbitrary future EOD bar either, same reasoning `cell_1480.py` uses for its own EOD/cover leg).

Detection is bar-level first (no network) on `bars_fills_1478.db` (the SAME single fresh store
cells 1,478-1,480 use): a candidate minute is any bar with low <= limit, from `m_break` (the fill
bar itself -- the fill instant is INSIDE this bar, so its remainder can retest too, unlike
cell 1,480's failed-break window which starts strictly after) through the window cutoff. Only a
bar-level candidate triggers a network call: `causal_arming.fetch_window(symbol, day, m, m+1)`,
cached to disk under `sip_cache_1481/` so a rerun never re-fetches; `sip_cache_1480/`'s windows are
reused by key (symbol_day_m) wherever a candidate minute (m > m_break) matches one 1,480 already
fetched (same level-1c threshold, same 15-minute cap, m strictly after m_break -- an exact key
match, not an approximation). No fetch cap (PREREG: "the population is the point"); candidate
minutes are walked in time order and fetched one at a time until the first qualifying print is
found or the window is exhausted (bar/tick disagreement -> next candidate minute, if any).
"""
import os
import sys
import time
import pickle
import sqlite3
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'hod_consol'))
import sip_rebuild as sr           # noqa: E402  (TICK, et_ns, walk_path, ns/day helpers)
import causal_arming as ca         # noqa: E402  (fetch_window)
import cell_1445 as c1445          # noqa: E402  (load_base_book, day_clustered_t, ex_top5_mean)
import cell_1478 as c1478          # noqa: E402  (build_outcome -> outcome_R, SLIP_STOP_BPS)
import cell_1479 as c1479          # noqa: E402  (load_bars_1478, BARS_DB, EOD_M, OPEN_M)
import run_consol as rc            # noqa: E402  (simulate_slots, CONCURRENT_CAP, DAILY_CAP)

CACHE_DIR_1480 = os.path.join(HERE, 'sip_cache_1480')
CACHE_DIR_1481 = os.path.join(HERE, 'sip_cache_1481')
os.makedirs(CACHE_DIR_1481, exist_ok=True)
LOG_PATH = os.path.join(HERE, 'cell_1481.log')

SEED = 1481
CELLS = {
    '1481': dict(window_min=15, limit_fn=lambda level: level - 0.01),
    '1482': dict(window_min=30, limit_fn=lambda level: level * (1.0 - 0.002)),
}
# cell 1,443's EOD holdout means (RESULT_1443.md "eod" rows), keyed like cell_1478.SLIP_STOP_BPS
# (on `split` -- 'TRAIN' means TRAIN-H2 here, the only TRAIN half ever kept in this book).
EOD_SLIP_BPS = {'TRAIN': 11.5, 'VAL': 9.7}
PASS = dict(mean=0.15, t=2.5, fills_wk=3.0, null_pctile=99.0, paired_delta=0.10)


def log(msg):
    """Verbose progress line, flushed immediately (print() is buffered under nohup otherwise)."""
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


# ================================================================================================
# Stage 0: base book (the 9,911 fills, standard-cost outcome, exact fill instant)
# ================================================================================================

def load_base():
    """The 9,911-fill base book with the standard-cost outcome (cell_1478.build_outcome's
    `outcome_R` = net_R_corr_v2 with the amendment's stop-limit slip substituted), the base fill
    instant in ns (`fill_min` is itself sr.ns_to_et_minutes(fill_ts) at build time -- exact to the
    print, no re-fetch needed to recover it), and the break-bar index `m_break`."""
    base = c1445.load_base_book()
    enriched = c1478.build_outcome(base)
    enriched = enriched.reset_index(drop=True)
    enriched['t_hit_ns'] = [sr.et_ns(r.day, r.fill_min * 60.0) for r in enriched.itertuples()]
    enriched['m_break'] = np.floor(enriched['fill_min'].to_numpy()).astype(int)
    return enriched


# ================================================================================================
# Stage 1: bar-level candidate scan (no network)
# ================================================================================================

def find_candidate_minutes(m_break, bars, window_min, limit, eod_m):
    """All bar-level candidate minutes (low <= limit) from m_break (INCLUSIVE -- the fill instant
    is inside this bar) through min(m_break + window_min, eod_m), in time order. Pure function of
    (m_break, bars, window_min, limit, eod_m) so it is directly unit-testable."""
    cutoff = min(m_break + window_min, eod_m)
    path = bars[(bars.m >= m_break) & (bars.m <= cutoff)]
    hit = path[path.l <= limit + 1e-9]
    return [int(m) for m in hit.m.tolist()]


# ================================================================================================
# Stage 2: tape fetch (resumable disk cache, 1,480 cache reused by exact key)
# ================================================================================================

def _cache_path(symbol, day, m):
    return os.path.join(CACHE_DIR_1481, f'{symbol}_{day}_{m}.pkl')


def _atomic_pickle_dump(obj, path):
    """Write a pickle atomically: to `<path>.tmp.<pid>` then os.replace(). Never leaves a
    partial/zero-size file at `path` for a concurrent reader (rebuild_1481.py shares this cache
    dir) to trip over mid-write."""
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _load_pickle_or_none(path):
    """Load a cache pickle; returns None (after a WARNING) on a missing, zero-size, or corrupt
    file instead of raising. A concurrent writer to the SAME cache dir (rebuild_1481.py) can leave
    a partial file mid-write -- `pickle.load` on it raises EOFError/UnpicklingError. Treating that
    as 'not cached' and re-fetching is the only correct recovery; silently returning is not (every
    fallback path logs)."""
    if not os.path.exists(path):
        return None
    try:
        if os.path.getsize(path) == 0:
            raise EOFError('zero-size cache file (concurrent writer mid-write)')
        with open(path, 'rb') as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, OSError) as e:
        log(f'  WARNING cache read failed {path}: {type(e).__name__}: {e} -- '
            f'treating as not cached, re-fetching')
        return None


def fetch_minute_tape(symbol, day, m):
    """One minute of trades+quotes via causal_arming.fetch_window(symbol, day, m, m+1), cached
    under sip_cache_1481/. Reuses sip_cache_1480/'s pickle for the SAME (symbol, day, m) key
    verbatim when it exists (1,480 only ever cached m > m_break candidates at this same
    level-1c threshold, so a key match is an exact reuse, not an approximation). Cache reads are
    crash-tolerant: a corrupt/partial pickle (rebuild_1481.py writes the same dir concurrently) is
    treated as a cache miss, logged, and re-fetched; writes are atomic (tmp file + os.replace)."""
    cp = _cache_path(symbol, day, m)
    cached = _load_pickle_or_none(cp)
    if cached is not None:
        return cached
    cp_1480 = os.path.join(CACHE_DIR_1480, f'{symbol}_{day}_{m}.pkl')
    cached_1480 = _load_pickle_or_none(cp_1480)
    if cached_1480 is not None:
        _atomic_pickle_dump(cached_1480, cp)
        return cached_1480
    t, q = ca.fetch_window(symbol, day, m, m + 1)
    _atomic_pickle_dump((t, q), cp)
    return t, q


def resolve_retest(symbol, day, m_break, t_hit_ns, cands, limit, fetch_fn=fetch_minute_tape):
    """Walks candidate minutes in time order; the fill rule is EXACT: the first print STRICTLY
    BELOW `limit` with ts > t_hit_ns (m_break's own remainder only -- every later candidate minute
    is entirely after t_hit_ns by construction) fills at that price. Returns a dict with
    status in {'never_retest' (no candidates), 'fill', 'bar_tick_disagree' (candidates existed,
    tape never confirmed one), 'fetch_error'}. `fetch_fn` is injectable for tests."""
    if not cands:
        return dict(status='never_retest', n_fetched=0)
    n_fetched = 0
    at_or_below_ts = at_or_below_px = None
    for m in cands:
        try:
            t, q = fetch_fn(symbol, day, m)
        except Exception as e:                                          # noqa: BLE001 -- network
            log(f'  WARNING fetch_error {symbol} {day} m={m}: {type(e).__name__}: {e}')
            return dict(status='fetch_error', n_fetched=n_fetched, err=str(e))
        n_fetched += 1
        if not len(t):
            continue
        w = t[t.ts > t_hit_ns].sort_values('ts', kind='stable') if m == m_break \
            else t.sort_values('ts', kind='stable')
        if not len(w):
            continue
        eq = w[w.price <= limit + 1e-9]
        if len(eq) and at_or_below_ts is None:
            at_or_below_ts = int(eq.ts.iloc[0])
            at_or_below_px = float(eq.price.iloc[0])
        strict = w[w.price < limit - 1e-9]
        if len(strict):
            return dict(status='fill', fill_ts=int(strict.ts.iloc[0]), fill_px=float(strict.price.iloc[0]),
                        m_retest=m, n_fetched=n_fetched, dip_low=float(w.price.min()),
                        at_or_below_ts=at_or_below_ts, at_or_below_px=at_or_below_px)
    return dict(status='bar_tick_disagree', n_fetched=n_fetched,
                at_or_below_ts=at_or_below_ts, at_or_below_px=at_or_below_px)


# ================================================================================================
# Stage 3: path pricing from the retest fill (tape inside the retest minute, then walk_path)
# ================================================================================================

def walk_retest_path(stop, retest, bars, split, eod_m, limit, fetch_fn=fetch_minute_tape,
                      symbol=None, day=None):
    """Path pricing from the retest fill onward. PREREG obtainability: the resting buy limit is
    filled AT THE LIMIT PRICE on a strict-below print (traded-through), never at the print itself
    -- so entry = `limit` (the SAME value process_cell computed via limit_fn(row.level) and passed
    to find_candidate_minutes/resolve_retest), not `retest['fill_px']` (kept only as the
    report-only `print_px` of the triggering tape print). Returns None if R' <= 0 (limit at/below
    the consolidation low -- a degenerate retest, dropped and counted, never silently kept).
    Otherwise a dict of entry/print_px/target/exit/why/raw_R/cost_R/net_R. `fetch_fn`/`symbol`/
    `day` are injectable for tests that already hold the retest-minute tape; the live path
    re-fetches (cache hit, no extra network call -- Stage 2 already populated this exact key)."""
    entry = limit
    print_px = retest['fill_px']
    Rp = entry - stop
    if Rp <= 0:
        return None
    target = entry + 2.0 * Rp

    m = retest['m_retest']
    t, _q = fetch_fn(symbol, day, m)
    after = t[t.ts > retest['fill_ts']].sort_values('ts', kind='stable')
    stop_hits = after[after.price <= stop + 1e-9]
    target_hits = after[after.price >= target - 1e-9]
    exit_m = exit_px = why = None
    s_ts = int(stop_hits.ts.iloc[0]) if len(stop_hits) else None
    g_ts = int(target_hits.ts.iloc[0]) if len(target_hits) else None
    if s_ts is not None and (g_ts is None or s_ts <= g_ts):
        exit_m, exit_px, why = m, stop, 'stop'
    elif g_ts is not None:
        exit_m, exit_px, why = m, target, 'target'

    if why is None:
        path = bars[(bars.m > m) & (bars.m <= eod_m)] if bars is not None else pd.DataFrame()
        if not len(path):
            exit_m, exit_px, why = m, entry, 'no_path'
        else:
            exit_m, exit_px, why = sr.walk_path(entry, stop, target, path)

    raw_R = (exit_px - entry) / Rp
    if why in ('stop', 'stop_bar'):
        cost_R = exit_px * c1478.SLIP_STOP_BPS[split] / 1e4 / Rp
    elif why in ('eod', 'eod_fallback'):
        cost_R = exit_px * EOD_SLIP_BPS[split] / 1e4 / Rp
    else:                                                   # target / no_path: a limit, no slip
        cost_R = 0.0
    net_R = raw_R - cost_R
    return dict(entry=entry, print_px=print_px, target=target, Rp=Rp, exit_m=exit_m,
                exit_price=exit_px, why=why, raw_R=raw_R, cost_R=cost_R, net_R=net_R)


# ================================================================================================
# Driver: one cell (1481 or 1482) over the full base book
# ================================================================================================

def process_cell(base, con, cell_name, cfg):
    window_min, limit_fn = cfg['window_min'], cfg['limit_fn']
    bars_by_sd, rows = {}, []
    n_never = n_disagree = n_error = n_filled = n_zero_risk = 0
    for i, row in enumerate(base.itertuples()):
        if (i + 1) % 500 == 0:
            log(f'{cell_name}: {i + 1}/{len(base)} processed ({n_filled} filled, {n_never} never-retest, '
                f'{n_disagree} disagree, {n_zero_risk} zero-risk, {n_error} fetch errors)')
        key = (row.day, row.symbol)
        if key not in bars_by_sd:
            bars_by_sd[key] = c1479.load_bars_1478(con, row.day, [row.symbol]).get(row.symbol)
        bars = bars_by_sd[key]
        out = dict(day=row.day, symbol=row.symbol, split=row.split, holdout=row.holdout, wk=row.wk,
                   fill=row.fill, stop=row.stop, level=row.level, fill_min=row.fill_min,
                   base_why=row.why, base_net_R=row.outcome_R)
        if bars is None or not len(bars):
            out['status'] = 'never_retest'
            rows.append(out); n_never += 1
            continue
        limit = limit_fn(row.level)
        out['limit'] = limit
        cands = find_candidate_minutes(row.m_break, bars, window_min, limit, c1479.EOD_M)
        retest = resolve_retest(row.symbol, row.day, row.m_break, row.t_hit_ns, cands, limit)
        if retest['status'] == 'never_retest':
            out['status'] = 'never_retest'; n_never += 1
        elif retest['status'] == 'fetch_error':
            out['status'] = 'fetch_error'; n_error += 1
        elif retest['status'] == 'bar_tick_disagree':
            out['status'] = 'bar_tick_disagree'; n_disagree += 1
            out['at_or_below_ts'] = retest.get('at_or_below_ts')
        else:
            path_res = walk_retest_path(row.stop, retest, bars, row.split, c1479.EOD_M, limit,
                                         symbol=row.symbol, day=row.day)
            if path_res is None:
                out['status'] = 'zero_risk'; n_zero_risk += 1
            else:
                out['status'] = 'fill'
                out.update(retest_ts=retest['fill_ts'], retest_minute=retest['m_retest'],
                           dip_low=retest['dip_low'], at_or_below_ts=retest.get('at_or_below_ts'),
                           entry=path_res['entry'], print_px=path_res['print_px'],
                           target=path_res['target'], Rp=path_res['Rp'],
                           exit_m=path_res['exit_m'], exit_price=path_res['exit_price'], why=path_res['why'],
                           raw_R=path_res['raw_R'], cost_R=path_res['cost_R'], net_R_prime=path_res['net_R'])
                out['r_pct_price'] = 100.0 * path_res['Rp'] / path_res['entry']
                out['retest_delay_min'] = (retest['fill_ts'] - row.t_hit_ns) / 60e9
                n_filled += 1
        rows.append(out)
    log(f'{cell_name} DONE: n={len(base)} filled={n_filled} never_retest={n_never} '
        f'disagree={n_disagree} zero_risk={n_zero_risk} fetch_error={n_error}')
    return pd.DataFrame(rows)


# ================================================================================================
# Scoring
# ================================================================================================

def count_matched_null(base, retest_df, holdout, n_draws=1000, seed=SEED):
    """1,000 stratified draws (same n per day as the observed retest-fill cohort) from the base
    book's own `outcome_R` on those SAME days -- the percentile rank of the observed mean net R'
    inside that null distribution."""
    obs = retest_df[(retest_df.holdout == holdout) & (retest_df.status == 'fill')]
    if not len(obs):
        return dict(null_pctile=np.nan, n_null=0)
    n_by_day = obs.groupby('day').size()
    base_h = base[base.holdout == holdout]
    by_day = {d: g.outcome_R.to_numpy() for d, g in base_h.groupby('day')}
    rng = np.random.default_rng(seed)
    obs_mean = float(obs.net_R_prime.mean())
    draws = []
    for _ in range(n_draws):
        vals = []
        for day, n in n_by_day.items():
            pool = by_day.get(day)
            if pool is None or not len(pool):
                continue
            vals.append(rng.choice(pool, size=int(n), replace=len(pool) < n))
        if vals:
            draws.append(float(np.concatenate(vals).mean()))
    draws = np.array(draws)
    pctile = float((draws < obs_mean).mean() * 100.0) if len(draws) else np.nan
    return dict(null_pctile=pctile, n_null=len(draws))


def score_holdout(fills_df, base, holdout):
    d = fills_df[fills_df.holdout == holdout]
    filled = d[d.status == 'fill']
    n, n_all = len(filled), len(d)
    weeks = base[base.holdout == holdout].wk.nunique()
    mean_net = float(filled.net_R_prime.mean()) if n else np.nan
    t = c1445.day_clustered_t(filled.net_R_prime, filled.day) if n > 1 else np.nan
    extop5 = float(c1445.ex_top5_mean(filled.net_R_prime)) if n else np.nan
    delta = (filled.net_R_prime - filled.base_net_R) if n else pd.Series(dtype=float)
    paired_t = c1445.day_clustered_t(delta, filled.day) if n > 1 else np.nan
    paired_delta = float(delta.mean()) if n else np.nan
    paired_base_mean = float(filled.base_net_R.mean()) if n else np.nan
    never = d[d.status == 'never_retest']
    never_mean = float(never.base_net_R.mean()) if len(never) else np.nan
    if n:
        trades = pd.DataFrame({'day': filled.day.to_numpy(),
                                'entry_m': filled.retest_minute.to_numpy(dtype=float),
                                'exit_m': filled.exit_m.to_numpy(dtype=float)})
        fills_wk = float(rc.simulate_slots(trades).sum()) / max(weeks, 1)
    else:
        fills_wk = 0.0
    null = count_matched_null(base, fills_df, holdout)
    return dict(n=n, n_all=n_all, fill_share=(n / n_all) if n_all else np.nan, mean_net_R=mean_net, t=t,
                extop5=extop5, fills_wk=fills_wk, paired_delta=paired_delta, paired_t=paired_t,
                paired_base_mean=paired_base_mean, never_retest_base_mean=never_mean,
                r_pct_median=float(filled.r_pct_price.median()) if n else np.nan,
                dip_median=float((filled.level - filled.dip_low).median()) if n else np.nan,
                delay_median=float(filled.retest_delay_min.median()) if n else np.nan,
                null_pctile=null['null_pctile'], weeks=int(weeks))


def cell_passes(scores):
    val, tr = scores.get('VAL'), scores.get('TRAIN-H2')
    if not val or not val['n']:
        return False
    ok = (val['mean_net_R'] >= PASS['mean'] and val['t'] >= PASS['t'] and val['extop5'] > 0 and
          val['fills_wk'] >= PASS['fills_wk'] and val['null_pctile'] >= PASS['null_pctile'] and
          val['paired_delta'] >= PASS['paired_delta'])
    if tr and tr['n']:
        ok = ok and (np.sign(tr['mean_net_R']) == np.sign(val['mean_net_R'])) and tr['t'] >= 1 \
            and tr['paired_delta'] >= PASS['paired_delta']
    return bool(ok)


def write_result_md(all_scores, path):
    lines = ['# RESULT -- cells 1,481-1,482: buy the retest, not the break',
             '',
             '`PREREG_1481.md`. Base = the 9,911 fills of cell 1,438; base net R = cell_1478\'s '
             'standard-cost `outcome_R`. n/n_all = primary-fill / all candidates (never-retest + '
             'disagree + zero-risk make up the gap). Paired ΔR = net R\' - base net R on the SAME '
             'fills. Null = count-matched (1,000 draws, seed 1481) percentile of the observed mean.',
             '',
             '| cell | holdout | n | n_all | fill % | mean net R\' | t | ex-top5 | fills/wk | '
             'paired ΔR | paired t | null %ile | never-retest base R | R\' % price | dip $ | delay min |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for cname, scores in all_scores.items():
        for h in ('TRAIN-H2', 'VAL'):
            s = scores.get(h, {})
            if not s or not s.get('n_all'):
                continue
            lines.append(
                f"| {cname} | {h} | {s['n']} | {s['n_all']} | {100*s['fill_share']:.1f}% | "
                f"{s['mean_net_R']:+.3f} | {s['t']:.2f} | {s['extop5']:+.3f} | {s['fills_wk']:.2f} | "
                f"{s['paired_delta']:+.3f} | {s['paired_t']:.2f} | {s['null_pctile']:.1f} | "
                f"{s['never_retest_base_mean']:+.3f} | {s['r_pct_median']:.2f}% | "
                f"{s['dip_median']:.3f} | {s['delay_median']:.1f} |")
        lines.append('')
        lines.append(f"**{cname} pass bar: {'PASS' if scores.get('_pass') else 'FAIL'}**")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log(f'wrote {path}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0, help='process only the first N base fills')
    ap.add_argument('--cells', default='1481,1482')
    args = ap.parse_args(argv)
    log('=== cell_1481 start ===' + (f' (SMOKE n={args.smoke})' if args.smoke else ''))
    base = load_base()
    log(f'base book: {len(base)} fills loaded (cell_1478 standard-cost outcome_R attached)')
    if args.smoke:
        base = base.head(args.smoke).copy()
    con = sqlite3.connect(f'file:{c1479.BARS_DB}?mode=ro', uri=True, timeout=120)

    all_scores = {}
    for cname in args.cells.split(','):
        cfg = CELLS[cname]
        log(f'--- {cname}: window={cfg["window_min"]}min ---')
        out = process_cell(base, con, cname, cfg)
        out_csv = os.path.join(HERE, f'cell_{cname}_fills.csv')
        out.to_csv(out_csv, index=False)
        log(f'wrote {out_csv} ({len(out)} rows)')
        scores = {h: score_holdout(out, base, h) for h in ('TRAIN-H2', 'VAL')}
        scores['_pass'] = cell_passes(scores)
        all_scores[cname] = scores
        for h in ('TRAIN-H2', 'VAL'):
            s = scores[h]
            if s['n_all']:
                log(f"{cname} {h}: n={s['n']}/{s['n_all']} fill_share={s['fill_share']:.3f} "
                    f"mean_net_R={s['mean_net_R']:+.4f} t={s['t']:.2f} extop5={s['extop5']:+.4f} "
                    f"fills_wk={s['fills_wk']:.2f} paired_delta={s['paired_delta']:+.4f} "
                    f"paired_t={s['paired_t']:.2f} null_pctile={s['null_pctile']:.1f} "
                    f"never_retest_mean={s['never_retest_base_mean']:+.4f}")
        log(f"{cname} PASS bar: {scores['_pass']}")

    write_result_md(all_scores, os.path.join(HERE, 'RESULT_1481.md'))
    log('=== cell_1481 END ===')
    return all_scores


if __name__ == '__main__':
    main()
