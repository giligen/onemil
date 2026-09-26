"""Cell 1,463 (unbiased) -- stop-limit exit slip, measured on a FRESH random sample.
PREREG_1466.md, "Unbiased 1,463" paragraph.

Fixes the bias in the earlier 1,463 columns (cell_1457_features.csv: slip_bps_1463_20bps /
_50bps), which reused cell_1443's cache/measurement machinery on the WHOLE measurable
population rather than an independent random draw with its own fresh tape. This cell:

  * samples, seed 1463, 400 stop exits (why in {stop, stop_bar}) per holdout (TRAIN-H2, VAL)
    from the base book (causal_arming_causal.csv, status == fill), on a canonical
    (day, symbol, fill_min)-sorted frame so the draw is reproducible independent of file order;
  * fetches FRESH SIP tape per row via causal_arming.fetch_window(symbol, day, exit_m, exit_m+1)
    -- its own cache dir (sip_cache_1463/, per-day pickle), never cell_1443's sip_cache_stopslip/;
  * why == 'stop': window = the exit-minute bar (m = int(exit_m)); why == 'stop_bar': window =
    the fill-minute bar (m = floor(fill_min)) AND candidate prints are restricted to strictly
    after the fill instant (et_ns(day, fill_min*60)) -- the position is not open before that
    instant, so an unfiltered first-print-<=-stop is look-ahead inside the fill bar. A stop_bar
    row with no finite fill_min, or no print after the fill instant, is FLAGGED and excluded from
    n (reason recorded, never silently dropped).
  * t0 = first print <= stop inside the (filtered) window.
  * stop-market: bid at t0+250ms (sip_rebuild.prevailing_quote), slip_bps = (stop-bid)/stop*1e4.
    Its per-holdout mean must land within 5 bps of cell_1443's 35.9 (TRAIN-H2) / 34.8 (VAL) or
    the sample is VOID (sample_valid=False) -- this cell measures 1,443's own population under a
    fresh, independent fetch, so a large drift means the fetch/quote logic disagrees, not that
    the population moved.
  * stop-limit L in {20, 50} bps: limit = stop*(1 - L/1e4). Filled at bid_250 if bid_250 >= limit;
    else at the first print >= limit strictly after t0 within the window; else a no-fill tail at
    the window's last (filtered) print.

Resumable: sip_cache_1463/{day}.pkl.gz maps row_key (symbol|exit_m|why|fill_min) -> the fetched
(trades, quotes) tuple. --fetch-only populates this cache and can be killed/rerun for free;
--report recomputes every statistic from the cache with zero network calls.
"""
import argparse
import gzip
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sip_rebuild as sr                                    # noqa: E402
import causal_arming as ca                                  # noqa: E402 -- window_ns, fetch_window

CACHE_DIR = os.path.join(HERE, 'sip_cache_1463')
os.makedirs(CACHE_DIR, exist_ok=True)
FILLS_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
RESULT_MD = os.path.join(HERE, 'RESULT_1463_unbiased.md')
LOG_PATH = os.path.join(HERE, 'cell_1463_unbiased_fetch.log')

SEED = 1463
N_SAMPLE = 400
LIMITS_BPS = (20, 50)
HOLDOUT_MAP = {'TRAIN': 'TRAIN-H2', 'VAL': 'VAL'}
REF_STOP_MARKET_BPS = {'TRAIN-H2': 35.9, 'VAL': 34.8}         # cell_1443 (whole population)
REF_TOL_BPS = 5.0
SHIP_MARGIN_BPS = 10.0
SHIP_NOFILL_MAX_BPS = 100.0


def log(msg):
    line = f'[cell_1463] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fh:
        fh.write(line + '\n')


def row_key(symbol, exit_m, why, fill_min):
    """Cache/resume key for one measured row (mirrors cell_1443's convention)."""
    return f'{symbol}|{exit_m}|{why}|{fill_min}'


def build_sample():
    """400 stop/stop_bar fills per holdout, seed 1463, from a canonically-sorted frame (order-
    independent of the CSV's own row order, so the draw is reproducible)."""
    df = pd.read_csv(FILLS_CSV, low_memory=False)
    f = df[(df.status == 'fill') & (df.why.isin(['stop', 'stop_bar']))].copy()
    f['holdout'] = f['split'].map(HOLDOUT_MAP)
    out = []
    for h, sub in f.groupby('holdout'):
        sub = sub.sort_values(['day', 'symbol', 'fill_min']).reset_index(drop=True)
        n = min(N_SAMPLE, len(sub))
        if n < N_SAMPLE:
            log(f'WARNING holdout={h} has only {len(sub)} stop/stop_bar fills, sampling {n} < {N_SAMPLE}')
        out.append(sub.sample(n=n, random_state=SEED))
    return pd.concat(out).reset_index(drop=True)


def load_cache(day):
    path = os.path.join(CACHE_DIR, f'{day}.pkl.gz')
    if os.path.exists(path):
        with gzip.open(path, 'rb') as fh:
            return pickle.load(fh)
    return {}


def save_cache(day, cache):
    path = os.path.join(CACHE_DIR, f'{day}.pkl.gz')
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wb') as fh:
        pickle.dump(cache, fh)
    os.replace(tmp, path)


def fetch_one(symbol, day, exit_m, why, fill_min):
    """Fresh (trades, quotes) for one row's window. Never raises -- caller stores the exception
    string so a fetch failure is counted, never silently dropped."""
    if why == 'stop_bar':
        m = int(np.floor(fill_min)) if np.isfinite(fill_min) else int(exit_m) - 1
    else:
        m = int(exit_m)
    try:
        t, q = ca.fetch_window(symbol, day, m, m + 1)
        return dict(ok=True, m=m, t=t, q=q)
    except Exception as e:                                   # noqa: BLE001 -- network/rate limit
        return dict(ok=False, m=m, reason=f'{type(e).__name__}: {e}')


def fetch_day(day, rows, workers):
    """Fetch every not-yet-cached row for one day, `workers` threads, save once done."""
    cache = load_cache(day)
    todo = [r for r in rows if row_key(r.symbol, r.exit_m, r.why, r.fill_min) not in cache]
    if not todo:
        return cache, 0, 0
    ok, lost = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, r.symbol, r.day, r.exit_m, r.why, r.fill_min): r for r in todo}
        for fu in futs:
            r = futs[fu]
            res = fu.result()
            k = row_key(r.symbol, r.exit_m, r.why, r.fill_min)
            cache[k] = res
            if res['ok']:
                ok += 1
            else:
                lost += 1
                log(f'  ERROR fetch {day} {r.symbol} exit_m={r.exit_m} why={r.why}: {res["reason"]} -- LOST')
    save_cache(day, cache)
    return cache, ok, lost


def fetch_all(sample, workers=2):
    """Resumable fetch loop over every sampled row, grouped by day. Logs progress per day."""
    days = sorted(sample.day.unique())
    n_ok, n_lost, t0 = 0, 0, time.time()
    for di, day in enumerate(days):
        rows = list(sample[sample.day == day].itertuples())
        _, ok, lost = fetch_day(day, rows, workers)
        n_ok += ok
        n_lost += lost
        if (di + 1) % 5 == 0 or di == len(days) - 1:
            el = time.time() - t0
            log(f'day {di + 1}/{len(days)} ({day}) | fetched this run: ok={n_ok} lost={n_lost} '
                f'| {el:.0f}s elapsed')
    log(f'fetch complete: {n_ok} ok, {n_lost} lost, {len(days)} days')


def measure_row(r):
    """One sampled row -> dict with market/limit-20/limit-50 slip outcomes, or reason=<why not
    measured>. Reads only from the sip_cache_1463 cache (must be pre-fetched)."""
    cache = load_cache(r.day)
    k = row_key(r.symbol, r.exit_m, r.why, r.fill_min)
    entry = cache.get(k)
    if entry is None:
        return dict(measured=False, reason='not_fetched')
    if not entry['ok']:
        return dict(measured=False, reason=f'fetch_error:{entry["reason"]}')
    t, q, m = entry['t'], entry['q'], entry['m']
    if not len(t) or not len(q):
        return dict(measured=False, reason='no_tape')
    start, end = ca.window_ns(r.day, m, m + 1)
    t_win = t[(t.ts >= start) & (t.ts < end)].sort_values('ts', kind='stable')
    if r.why == 'stop_bar':
        if not np.isfinite(r.fill_min):
            return dict(measured=False, reason='no_fill_instant_flagged')
        fill_ns = sr.et_ns(r.day, r.fill_min * 60)
        t_win = t_win[t_win.ts > fill_ns]
        if not len(t_win):
            return dict(measured=False, reason='no_print_after_fill_flagged')
    hit = t_win[t_win.price <= r.stop + 1e-9]
    if not len(hit):
        return dict(measured=False, reason='no_print_le_stop')
    t0 = int(hit.ts.iloc[0])
    pq = sr.prevailing_quote(q, t0 + 250_000_000)
    if pq is None:
        return dict(measured=False, reason='no_valid_quote')
    bid_250 = pq[0]
    out = dict(measured=True, t0=t0, bid_250=float(bid_250),
                market_slip_bps=float((r.stop - bid_250) / r.stop * 1e4))
    after_t0 = t_win[t_win.ts > t0]
    for L in LIMITS_BPS:
        limit = r.stop * (1 - L / 1e4)
        if bid_250 >= limit:
            price, nofill = bid_250, False
        else:
            cand = after_t0[after_t0.price >= limit - 1e-9]
            if len(cand):
                price, nofill = float(cand.price.iloc[0]), False
            else:
                price, nofill = float(t_win.price.iloc[-1]), True
        out[f'slip_{L}'] = float((r.stop - price) / r.stop * 1e4)
        out[f'nofill_{L}'] = nofill
    return out


def _stats(s):
    s = pd.Series(s, dtype=float).dropna()
    if not len(s):
        return dict(n=0, mean=float('nan'), median=float('nan'), p90=float('nan'))
    return dict(n=len(s), mean=float(s.mean()), median=float(s.median()), p90=float(s.quantile(0.9)))


def report(sample):
    """Compute the per-variant x holdout table from the (already fetched) cache; write
    RESULT_1463_unbiased.md; return (rows, stop_market_means, sample_valid, ship_20, ship_50)."""
    measured = [measure_row(r) for r in sample.itertuples()]
    m = sample.copy().reset_index(drop=True)
    m['measured'] = [d['measured'] for d in measured]
    m['reason'] = [d.get('reason', '') for d in measured]
    m['market_slip_bps'] = [d.get('market_slip_bps', np.nan) for d in measured]
    for L in LIMITS_BPS:
        m[f'slip_{L}'] = [d.get(f'slip_{L}', np.nan) for d in measured]
        m[f'nofill_{L}'] = [d.get(f'nofill_{L}', False) for d in measured]

    reason_counts = m[~m.measured].reason.value_counts().to_dict()
    log(f'unmeasured reasons: {reason_counts}')

    rows = []
    stop_market_means = {}
    for h in ('TRAIN-H2', 'VAL'):
        sub = m[m.holdout == h]
        meas = sub[sub.measured]
        mk = _stats(meas.market_slip_bps)
        stop_market_means[h] = mk['mean']
        rows.append(dict(variant='stop-market', holdout=h, n=mk['n'], mean_bps=mk['mean'],
                          median_bps=mk['median'], p90_bps=mk['p90'], n_no_fill=0, no_fill_mean_bps=float('nan')))
        for L in LIMITS_BPS:
            filled = meas[~meas[f'nofill_{L}']]
            nofilled = meas[meas[f'nofill_{L}']]
            st = _stats(filled[f'slip_{L}'])
            nf = _stats(nofilled[f'slip_{L}'])
            rows.append(dict(variant=f'stop-limit-{L}bps', holdout=h, n=st['n'], mean_bps=st['mean'],
                              median_bps=st['median'], p90_bps=st['p90'], n_no_fill=nf['n'],
                              no_fill_mean_bps=nf['mean']))

    sample_valid = all(
        np.isfinite(stop_market_means[h]) and abs(stop_market_means[h] - REF_STOP_MARKET_BPS[h]) <= REF_TOL_BPS
        for h in ('TRAIN-H2', 'VAL'))

    def ship(L):
        ok = True
        for h in ('TRAIN-H2', 'VAL'):
            row = next(r for r in rows if r['variant'] == f'stop-limit-{L}bps' and r['holdout'] == h)
            margin_ok = np.isfinite(row['mean_bps']) and (stop_market_means[h] - row['mean_bps']) >= SHIP_MARGIN_BPS
            nofill_ok = (row['n_no_fill'] == 0) or (np.isfinite(row['no_fill_mean_bps'])
                                                      and row['no_fill_mean_bps'] <= SHIP_NOFILL_MAX_BPS)
            ok = ok and margin_ok and nofill_ok
        return ok

    ship_20, ship_50 = ship(20), ship(50)
    write_result_md(rows, stop_market_means, sample_valid, ship_20, ship_50, reason_counts, m)
    return rows, stop_market_means, sample_valid, ship_20, ship_50


def write_result_md(rows, stop_market_means, sample_valid, ship_20, ship_50, reason_counts, m):
    lines = ['# RESULT 1,463 (unbiased) -- stop-limit exit slip, fresh random sample', '',
             '| variant | holdout | n | mean bps | median bps | p90 bps | no-fill n | no-fill mean bps |',
             '|---|---|---|---|---|---|---|---|']
    for r in rows:
        nf = f"{r['no_fill_mean_bps']:.1f}" if np.isfinite(r['no_fill_mean_bps']) else 'n/a'
        lines.append(f"| {r['variant']} | {r['holdout']} | {r['n']} | {r['mean_bps']:.1f} | "
                      f"{r['median_bps']:.1f} | {r['p90_bps']:.1f} | {r['n_no_fill']} | {nf} |")
    n_flagged = sum(v for k, v in reason_counts.items() if 'flagged' in k)
    lines.append('')
    lines.append(f"Stop-market mean vs cell_1443 ref (35.9/34.8, tol {REF_TOL_BPS}): "
                 f"TRAIN-H2 {stop_market_means['TRAIN-H2']:.1f}, VAL {stop_market_means['VAL']:.1f} "
                 f"-> sample_valid={sample_valid}.")
    lines.append(f"Ship bar (mean slip <= stop-market by >= {SHIP_MARGIN_BPS} bps AND no-fill mean "
                 f"<= {SHIP_NOFILL_MAX_BPS} bps, both holdouts): 20bps={ship_20}, 50bps={ship_50}.")
    lines.append(f"Unmeasured/flagged: {reason_counts} ({n_flagged} stop_bar fill-instant flags).")
    with open(RESULT_MD, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log(f'wrote {RESULT_MD}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch-only', action='store_true', help='populate sip_cache_1463/ only, no report')
    ap.add_argument('--report', action='store_true', help='compute stats from the cache only, no fetching')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    sample = build_sample()
    log(f'sample built: {len(sample)} rows ({sample.holdout.value_counts().to_dict()})')

    if args.report:
        report(sample)
        return
    fetch_all(sample, workers=args.workers)
    if not args.fetch_only:
        report(sample)


if __name__ == '__main__':
    main()
