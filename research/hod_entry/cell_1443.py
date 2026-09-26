"""Cell 1,443 — stop slippage measured on the tape (research/hod_entry/WEEKEND_QUEUE.md row 10,
frozen). Report-only: no pass bar; its number restates every cell's net R and gates size (mean slip
> 40 bps => no size increase until live stops confirm).

Base fills = research/hod_entry/causal_arming_causal.csv (cell 1,438, correct levels, `causal_arming.py`)
rows with status == 'fill'. The 1,427 "E1 fills" are VOID (sparse levels) so 1,438's fills replace them.
TEST was never run for 1,438 (only TRAIN-H2 / VAL exist in this CSV) so TEST is not read here.

Measurement, per stop exit (why in {stop, stop_bar}):
  * window = the exit minute (why == 'stop') or the fill minute floor(fill_min) (why == 'stop_bar' — the
    fill instant is not stored in this CSV, so stop_bar rows are always flagged 'fill_bar_approx').
  * t0 = first SIP print <= stop inside the window (causal_arming.fetch_window, same fetch/cache
    machinery as cell_1428/1439/1442).
  * bid_250 = the NBBO bid prevailing at t0 + 250ms (sip_rebuild.prevailing_quote, last valid quote
    ts <= t0+250ms).
  * slip_bps = (stop - bid_250) / stop * 1e4 (positive = worse than the stop).

Per EOD exit (why == 'eod'): window = 15:55-15:56 ET; t0 = first print of that minute; bid = the NBBO
bid prevailing AT t0 (no +250ms offset — spec asks for "the bid at the first print"); slip_bps =
(exit_price - bid) / exit_price * 1e4.

slip in R units = slip_$ / R (R column = the $ risk per share, fill - stop; same convention as
cell_1430's STOP_SLIP_BP / stop_slip_net, so it composes directly with net_R = raw_R - cost_R).

Resumable: per-day pickle cache under sip_cache_stopslip/{day}.pkl.gz maps row key
(symbol|exit_m|why|fill_min) -> measurement dict. Re-running `run` only fetches missing keys.
"""
import argparse
import gzip
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sip_rebuild as sr                                    # noqa: E402
import causal_arming as ca                                  # noqa: E402 — window_ns, fetch_window

ROOT = sr.ROOT
CACHE_DIR = os.path.join(HERE, 'sip_cache_stopslip')
os.makedirs(CACHE_DIR, exist_ok=True)
FILLS_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
STATE_JSON = os.path.join(HERE, 'cell_1443_state.json')
EOD_MIN = 955
SIZE_GATE_BPS = 40.0

# holdout label used throughout the report (this CSV's 'half' column is always H2)
HOLDOUT_MAP = {'TRAIN': 'TRAIN-H2', 'VAL': 'VAL'}


def log(msg):
    """Verbose progress line, flushed immediately (print() is buffered under nohup otherwise)."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def row_key(symbol, exit_m, why, fill_min):
    """Cache/resume key for one measured row."""
    return f'{symbol}|{exit_m}|{why}|{fill_min}'


def load_base_fills():
    """1,438's FULL fill population (status == 'fill', every why), holdout-labelled. The net-R
    restatement needs every exit (target fills included) so mean_net_R_before/after is the real
    book average, not an average over losers only — only the stop/stop_bar subset is tape-fetched
    and only that subset gets a slip charge in the 'after' figure."""
    df = pd.read_csv(FILLS_CSV, low_memory=False)
    f = df[df.status == 'fill'].copy()
    f['holdout'] = f['split'].map(HOLDOUT_MAP)
    return f.reset_index(drop=True)


def to_measure(fills):
    """Subset that needs a tape fetch: stop/stop_bar (slip) and eod (bid-at-first-print) exits."""
    return fills[fills.why.isin(['stop', 'stop_bar', 'eod'])]


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


def measure_stop(symbol, day, stop, exit_m, why, fill_min):
    """One stop/stop_bar exit -> dict(measured, slip_bps, bid_250, t0, flag, reason). Never raises —
    fetch failures are counted as unmeasured (LOST), never silently dropped."""
    if why == 'stop_bar':
        m = int(np.floor(fill_min)) if np.isfinite(fill_min) else int(exit_m) - 1
        flag = 'fill_bar_approx'          # fill instant not in the CSV — see module docstring
    else:
        m = int(exit_m)
        flag = ''
    try:
        t, q = ca.fetch_window(symbol, day, m, m + 1)
    except Exception as e:                                   # noqa: BLE001 — network/rate limit
        log(f'  WARNING fetch failed {symbol} {day} m={m}: {type(e).__name__}: {e}')
        return dict(measured=False, reason=f'fetch_error:{type(e).__name__}', flag=flag)
    if not len(t) or not len(q):
        return dict(measured=False, reason='no_tape', flag=flag)
    hit = t[t.price <= stop + 1e-9].sort_values('ts', kind='stable')
    if not len(hit):
        return dict(measured=False, reason='no_print_le_stop', flag=flag)
    t0 = int(hit.ts.iloc[0])
    pq = sr.prevailing_quote(q, t0 + 250_000_000)
    if pq is None:
        return dict(measured=False, reason='no_valid_quote', flag=flag)
    bid_250 = pq[0]
    slip_bps = (stop - bid_250) / stop * 1e4
    return dict(measured=True, t0=t0, bid_250=bid_250, slip_bps=float(slip_bps), flag=flag)


def measure_eod(symbol, day, exit_price):
    """The 15:55 exit vs the NBBO bid at the first print of that minute."""
    try:
        t, q = ca.fetch_window(symbol, day, EOD_MIN, EOD_MIN + 1)
    except Exception as e:                                   # noqa: BLE001
        log(f'  WARNING fetch failed {symbol} {day} EOD: {type(e).__name__}: {e}')
        return dict(measured=False, reason=f'fetch_error:{type(e).__name__}', flag='')
    if not len(t) or not len(q):
        return dict(measured=False, reason='no_tape', flag='')
    t0 = int(t.sort_values('ts', kind='stable').ts.iloc[0])
    pq = sr.prevailing_quote(q, t0)
    if pq is None:
        return dict(measured=False, reason='no_valid_quote', flag='')
    bid = pq[0]
    slip_bps = (exit_price - bid) / exit_price * 1e4
    return dict(measured=True, t0=t0, bid_250=bid, slip_bps=float(slip_bps), flag='')


def run(time_budget_s=None, day_limit=None):
    """Fetch/measure every missing row, day by day, checkpointing the per-day cache after each day.
    ≤ 2 workers, nice -19 per the token-discipline budget — this loop itself stays single-threaded
    and relies on the caller's `nice` wrapper; a light ThreadPoolExecutor is used only inside
    causal_arming.fetch_window's own retry logic, not here, to keep the SIP request rate predictable."""
    fills = to_measure(load_base_fills())
    days = sorted(fills.day.unique())
    if day_limit:
        days = days[:day_limit]
    log(f'{len(fills)} rows to measure across {len(days)} days')
    t_start = time.time()
    n_done, n_new = 0, 0
    for di, day in enumerate(days):
        if time_budget_s and time.time() - t_start > time_budget_s:
            log(f'time budget {time_budget_s}s hit at day {di}/{len(days)} — stopping, resumable')
            break
        day_rows = fills[fills.day == day]
        cache = load_cache(day)
        changed = False
        for r in day_rows.itertuples():
            key = row_key(r.symbol, r.exit_m, r.why, r.fill_min)
            if key in cache:
                n_done += 1
                continue
            if r.why == 'eod':
                res = measure_eod(r.symbol, r.day, r.exit_price)
            else:
                res = measure_stop(r.symbol, r.day, r.stop, r.exit_m, r.why, r.fill_min)
            cache[key] = res
            changed = True
            n_done += 1
            n_new += 1
        if changed:
            save_cache(day, cache)
        if di % 10 == 0 or di == len(days) - 1:
            elapsed = time.time() - t_start
            log(f'day {di + 1}/{len(days)} ({day}): {n_done} rows done ({n_new} new this run), '
                f'{elapsed:.0f}s elapsed')
    log(f'run() finished: {n_done} rows done, {n_new} newly fetched this run')


def collect(fills):
    """Attach cached measurements to the FULL fills frame. Rows outside {stop, stop_bar, eod} (e.g.
    target exits) were never tape-fetched — they get slip_bps/slip_R = NaN, flag='not_applicable',
    and are never charged a slip in the restatement. Returns fills with slip_bps, measured, flag,
    slip_R columns added."""
    out = fills.copy()
    slip_bps, measured, flag = [], [], []
    cache_by_day = {}
    for r in out.itertuples():
        if r.why not in ('stop', 'stop_bar', 'eod'):
            slip_bps.append(np.nan)
            measured.append(False)
            flag.append('not_applicable')
            continue
        cache = cache_by_day.get(r.day)
        if cache is None:
            cache = load_cache(r.day)
            cache_by_day[r.day] = cache
        key = row_key(r.symbol, r.exit_m, r.why, r.fill_min)
        res = cache.get(key)
        if res is None or not res.get('measured'):
            slip_bps.append(np.nan)
            measured.append(False)
            flag.append(res.get('reason', 'not_run') if res else 'not_run')
        else:
            slip_bps.append(res['slip_bps'])
            measured.append(True)
            flag.append(res.get('flag', ''))
    out['slip_bps'] = slip_bps
    out['measured'] = measured
    out['flag'] = flag
    out['slip_dollar'] = out['stop'].where(out.why != 'eod', out['exit_price']) * out['slip_bps'] / 1e4
    out['slip_R'] = out['slip_dollar'] / out['R']
    return out


def _stats(s):
    """mean/median/p75/p90 of a numeric series, NaN-safe."""
    s = s.dropna()
    if not len(s):
        return dict(n=0, mean=np.nan, median=np.nan, p75=np.nan, p90=np.nan)
    return dict(n=len(s), mean=s.mean(), median=s.median(), p75=s.quantile(.75), p90=s.quantile(.90))


def slip_table(measured_fills):
    """Per-holdout slip stats for stop exits and for EOD exits. Returns a list of dict rows."""
    rows = []
    for holdout in ['TRAIN-H2', 'VAL']:
        h = measured_fills[measured_fills.holdout == holdout]
        stops = h[h.why.isin(['stop', 'stop_bar'])]
        eod = h[h.why == 'eod']
        st = _stats(stops.slip_bps)
        sr_ = stops.slip_bps.dropna()
        share30 = float((sr_ > 30).mean()) if len(sr_) else np.nan
        share100 = float((sr_ > 100).mean()) if len(sr_) else np.nan
        r_units = _stats(stops.slip_R)
        eodst = _stats(eod.slip_bps)
        rows.append(dict(holdout=holdout, kind='stop', n_requested=len(stops), **st,
                          share_gt30bps=share30, share_gt100bps=share100, mean_slip_R=r_units['mean']))
        rows.append(dict(holdout=holdout, kind='eod', n_requested=len(eod), **eodst,
                          share_gt30bps=np.nan, share_gt100bps=np.nan, mean_slip_R=np.nan))
    return rows


def restate_1438(measured_fills):
    """1,438's mean net R per holdout over the FULL fill population (every why, target exits
    included), before (as-recorded net_R) and after (measured per-trade slip charged ONLY on
    stop/stop_bar exits — target/eod rows are untouched; unmeasured stop rows keep net_R unchanged
    and are counted in n_stops vs n_measured, never dropped, never imputed)."""
    rows = []
    for holdout in ['TRAIN-H2', 'VAL']:
        h = measured_fills[measured_fills.holdout == holdout]
        stops = h[h.why.isin(['stop', 'stop_bar'])]
        apply_slip = np.where(h.why.isin(['stop', 'stop_bar']), h['slip_R'].fillna(0.0), 0.0)
        after = h['net_R'] - apply_slip
        rows.append(dict(cell='1438', holdout=holdout, n_stops=len(stops),
                          n_measured=int(stops.measured.sum()),
                          mean_slip_bps=stops.slip_bps.mean(), mean_net_R_before=h['net_R'].mean(),
                          mean_net_R_after=after.mean()))
    return rows


def restate_other(cell_name, csv_path, holdout_mean_bps, holdout_col='split'):
    """Apply the 1,438 HOLDOUT-MEAN measured bps to another cell's own stop/stop_bar fills (per the
    spec: no separate tape fetch for 1439/1428/1441 — the mean bps from 1,438 is reused)."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, low_memory=False)
    f = df[df.status == 'fill'].copy()
    f['holdout'] = f[holdout_col].map(HOLDOUT_MAP)
    rows = []
    for holdout in ['TRAIN-H2', 'VAL']:
        h = f[f.holdout == holdout].copy()          # FULL population (target exits included)
        stops = h[h.why.isin(['stop', 'stop_bar'])]
        mean_bps = holdout_mean_bps.get(holdout, np.nan)
        slip_dollar = stops['stop'] * mean_bps / 1e4
        slip_R = slip_dollar / stops['R']
        after_all = h['net_R'].copy()
        after_all.loc[stops.index] = stops['net_R'] - slip_R
        rows.append(dict(cell=cell_name, holdout=holdout, n_stops=len(stops),
                          n_measured=len(stops), mean_slip_bps=mean_bps,
                          mean_net_R_before=h['net_R'].mean(), mean_net_R_after=after_all.mean()))
    return rows


def build_report():
    """Full report: slip table + restatement table + verdict text. Returns (slip_rows, restate_rows,
    verdict_lines)."""
    fills = load_base_fills()
    measured = collect(fills)
    tape_pop = measured[measured.why.isin(['stop', 'stop_bar', 'eod'])]
    n_req = len(tape_pop)                                    # denominator = the tape-fetched subset only
    n_meas = int(tape_pop.measured.sum())
    log(f'coverage: {n_meas}/{n_req} rows measured ({n_meas / n_req * 100:.1f}%)')

    slip_rows = slip_table(measured)
    holdout_mean_bps = {}
    for row in slip_rows:
        if row['kind'] == 'stop':
            holdout_mean_bps[row['holdout']] = row['mean']

    restate_rows = restate_1438(measured)
    for name, path in [('1439', os.path.join(HERE, 'cell_1439_fills.csv')),
                        ('1428', os.path.join(HERE, 'cell_1428_causal.csv')),
                        ('1441', os.path.join(HERE, 'cell_1441_fills.csv'))]:
        rr = restate_other(name, path, holdout_mean_bps)
        if rr is None:
            restate_rows.append(dict(cell=name, holdout='(no fills CSV found — skipped)', n_stops=0,
                                      n_measured=0, mean_slip_bps=np.nan, mean_net_R_before=np.nan,
                                      mean_net_R_after=np.nan))
        else:
            restate_rows.extend(rr)

    val_mean = holdout_mean_bps.get('VAL', np.nan)
    gate = 'FAIL — size increase blocked' if (np.isfinite(val_mean) and val_mean > SIZE_GATE_BPS) \
        else 'PASS — no block from this gate' if np.isfinite(val_mean) else 'UNRESOLVED (no VAL coverage)'
    verdict = [f'VAL mean measured stop slip = {val_mean:.1f} bps vs the {SIZE_GATE_BPS:.0f} bps size gate: {gate}.',
               f'Coverage: {n_meas}/{n_req} rows measured ({n_meas / max(n_req, 1) * 100:.1f}%).',
               'Report-only cell: no pass bar; this restates every cell\'s net R and gates size only.']
    return slip_rows, restate_rows, verdict


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cmd', choices=['run', 'report'])
    p.add_argument('--time-budget-s', type=float, default=None)
    p.add_argument('--day-limit', type=int, default=None)
    a = p.parse_args(argv)
    if a.cmd == 'run':
        run(time_budget_s=a.time_budget_s, day_limit=a.day_limit)
    else:
        slip_rows, restate_rows, verdict = build_report()
        for r in slip_rows:
            log(str(r))
        for r in restate_rows:
            log(str(r))
        for line in verdict:
            log(line)


if __name__ == '__main__':
    main()
