"""Cell 1,438 — the E1 fill book as a LIVE resting order would produce it (causal arming). PREREG_1438.md.

Change vs cell 1,427 (research/hod_entry/sip_rebuild.py, whose fetch / cache / fill helpers are reused):
  * population = EVERY symbol-day of the spec's causal superset (research/bf_zero/spec_sim.py lines 17-25:
    universe symbol-days with day high >= open x (1 + min_dist)), plus the live admission gates the b0 book
    carries (adv20 >= min_adv20, trigger >= min_price), test tickers ^Z[A-Z]ZZT$ excluded;
  * at the close of each RTH bar j the order for the next bar is ARMED iff, through bar j only:
    consolidation_low(l, h, j) is valid, level = HOD through j, level >= open x (1 + min_dist %),
    rv_profile(cumv[j], adv20, m[j]) in [rv_lo, rv_hi), m[j+1] <= last_entry_minute, trigger >= min_price and
    (trigger - stop) / trigger >= min_r_pct %;
  * armed: buy-stop-limit at level + 0.01, limit level x 1.0015, fills at the prevailing NBBO ask at the first SIP
    print >= trigger during bar j+1's window [end of minute m[j], end of minute m[j+1]) if ask <= limit; a print
    <= stop later in that window = stopped; then B0 path physics on the cache.db minute bars after bar j+1;
    target = fill + 2 R; 15:55 exit; cost = fill-instant half-spread + B0's exit leg (half the measured
    per-signal spread from causal_filter/nbbo.csv + 2 bp; fill-instant half-spread where nbbo.csv lacks the day);
  * no cross, or a cross with ask > limit, = no position; the order re-arms at the next close if the conditions
    hold. First fill per symbol-day only.
  * report-only variant 'tick_rv': rv is evaluated at each print >= trigger with cumv[j] + the SIP volume of the
    window up to and including that print, at the clock m[j+1]; the first such print with rv in band fills.
Tape rail: a crossing armed bar (minute high >= trigger) whose window has no print or no prevailing NBBO makes the
symbol-day UNUSABLE (its fill cannot be known). Winner/loser for the missingness gap = a proxy outcome (fill at
the trigger on the first crossing armed bar, same path rules, raw R > 0), known for every symbol-day in scope.
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
import sip_rebuild as sr  # noqa: E402

ROOT = sr.ROOT
sys.path.insert(0, ROOT)
from trading.hod_break import HodBreakParams, consolidation_low, rv_profile  # noqa: E402

UNIVERSE_CSV = os.path.join(ROOT, 'research/bf_zero/universe.csv')
NBBO_CSV = os.path.join(ROOT, 'research/bf_zero/causal_filter/nbbo.csv')
REPORT_MD = os.path.join(HERE, 'REPORT_1438.md')
OUT_CSV = os.path.join(HERE, 'causal_arming_{variant}.csv')
C1427_CSV = os.path.join(HERE, 'sip_rebuild_val.csv')
SPLITS = {'TRAIN': ('2025-07-01', '2025-12-31'), 'VAL': ('2026-01-01', '2026-05-31')}
TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')
log = sr.log


def live_params():
    """HodBreakParams from config.Config().hod_break_cfg['params'] (as spec_sim.py) + the live admission gates."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, '.env'))
    from config import Config
    cfg = Config().hod_break_cfg
    p = HodBreakParams(**(cfg.get('params') or {}))
    return p, float(cfg.get('min_price') or 0), float(cfg.get('min_adv20') or 0)


# --------------------------------------------------------------------------------------------- arming
def arm_state(o, h, l, v, m, j, adv20, p, floor, use_rv=True):
    """Arming decision at the close of bar j from bars 0..j only. Returns dict(level, trigger, limit, stop) or
    None. `use_rv=False` skips the rv band (the tick_rv variant checks it at the print instead)."""
    if int(m[j]) + 1 > p.last_entry_minute:   # RTH minutes are consecutive — parity with trading.hod_break.arm_state's live fix
        return None
    stop = consolidation_low(l, h, j, p)
    if stop is None:
        return None
    level = float(np.max(h[: j + 1]))
    if level < float(o[0]) * (1.0 + p.min_dist_open_pct / 100.0):
        return None
    if use_rv:
        rv = rv_profile(float(np.sum(v[: j + 1])), adv20, int(m[j]))
        if not (p.rv_lo <= rv < p.rv_hi):
            return None
    trigger = round(level + sr.TICK, 6)
    if trigger < floor or (trigger - stop) / trigger * 100.0 < p.min_r_pct:
        return None
    return dict(level=level, trigger=trigger, limit=level * (1.0 + sr.LIMIT_BPS), stop=stop)


def armed_crossing_bars(bars, adv20, p, floor, use_rv=True):
    """Armed bars j whose next bar's minute high reaches the trigger (the only bars that can fill), in order.
    Each item: dict(j, level, trigger, limit, stop, m_lo, m_hi) with the fill window [end of m_lo, end of m_hi)."""
    o, h, l, v = (bars[k].values.astype(float) for k in ('o', 'h', 'l', 'v'))
    m = bars.m.values.astype(int)
    hod = np.maximum.accumulate(h)
    out = []
    for j in np.nonzero(h[1:] >= hod[:-1] + sr.TICK - 1e-9)[0]:     # cheap pre-filter: bar j+1 crosses HOD_j + 1c
        a = arm_state(o, h, l, v, m, int(j), adv20, p, floor, use_rv)
        if a is not None and h[j + 1] >= a['trigger'] - 1e-9:
            out.append(dict(a, j=int(j), m_lo=int(m[j]), m_hi=int(m[j + 1]), cumv_j=float(np.sum(v[: j + 1]))))
    return out


def window_ns(day, m_lo, m_hi):
    """UTC-ns [start, end) of the fill window: end of minute m_lo to end of minute m_hi (ET)."""
    return sr.et_ns(day, (m_lo + 1) * 60), sr.et_ns(day, (m_hi + 1) * 60)


def resolve_window(trades, quotes, start, end, arm, rv_ctx=None):
    """Fill decision inside one armed window. Returns dict(status in {'no_tape','nofill','fill'}, fill, fill_ts,
    half_spread, stopped). rv_ctx=(cumv_j, adv20, minute, p) turns on the tick_rv variant."""
    out = dict(status='no_tape', fill=np.nan, fill_ts=np.nan, half_spread=np.nan, stopped=False)
    w = trades[(trades.ts >= start) & (trades.ts < end)].sort_values('ts', kind='stable')
    if not len(w) or not len(sr._valid(quotes[quotes.ts < end])):
        return out
    hits = w[w.price >= arm['trigger'] - 1e-9]
    if rv_ctx is not None and len(hits):
        cumv_j, adv20, minute, p = rv_ctx
        cum = cumv_j + w['size'].cumsum()
        rv = cum.loc[hits.index].map(lambda x: rv_profile(float(x), adv20, minute))
        hits = hits[(rv >= p.rv_lo) & (rv < p.rv_hi)]
    if not len(hits):
        out['status'] = 'nofill'
        return out
    t_hit = int(hits.ts.iloc[0])
    pq = sr.prevailing_quote(quotes, t_hit)
    if pq is None:
        return out
    bid, ask = pq
    if ask > arm['limit'] + 1e-9:
        out['status'] = 'nofill'
        return out
    after = w[w.ts > t_hit]
    out.update(status='fill', fill=ask, fill_ts=t_hit, half_spread=0.5 * (ask - bid),
               stopped=bool((after.price <= arm['stop'] + 1e-9).any()))
    return out


def resolve_day(cands, tape_for, day, adv20, p, tick_rv=False):
    """First fill of a symbol-day: walk the crossing armed bars in order; an unusable tape stops the day
    (status no_tape). Returns (status, arm, entry-dict)."""
    for a in cands:
        start, end = window_ns(day, a['m_lo'], a['m_hi'])
        t, q = tape_for(a)
        ctx = (a['cumv_j'], adv20, a['m_hi'], p) if tick_rv else None
        e = resolve_window(t, q, start, end, a, ctx)
        if e['status'] == 'no_tape':
            return 'no_tape', a, e
        if e['status'] == 'fill':
            return 'fill', a, e
    return 'nofill', None, None


def proxy_outcome(cands, bars):
    """Winner/loser proxy for the missingness gap: fill at the trigger on the first crossing armed bar, B0 path
    rules from the next bar. Returns raw R or NaN."""
    if not cands:
        return np.nan
    a = cands[0]
    fill, stop = a['trigger'], a['stop']
    path = bars[(bars.m > a['m_hi']) & (bars.m <= sr.EOD_M)]
    if not len(path):
        return np.nan
    _, px, _ = sr.walk_path(fill, stop, fill + sr.TARGET_R * (fill - stop), path.rename(columns={}))
    return (px - fill) / (fill - stop)


# --------------------------------------------------------------------------------------------- data
def load_population(p, floor, min_adv):
    """The causal superset (spec_sim.py): universe symbol-days with high >= open x (1 + min_dist %), in TRAIN-H2/VAL,
    adv20 >= min_adv20, day high >= min_price + 1c (a trigger >= the floor needs it), test tickers excluded."""
    u = pd.read_csv(UNIVERSE_CSV, dtype={'symbol': str}, keep_default_na=False)
    for c in ('open', 'high', 'adv20'):
        u[c] = pd.to_numeric(u[c], errors='coerce')
    u = u.rename(columns={'bar_date': 'day'})
    u = u[(u.day >= SPLITS['TRAIN'][0]) & (u.day <= SPLITS['VAL'][1])]
    u = u[(u.high >= u.open * (1 + p.min_dist_open_pct / 100)) & (u.adv20 >= min_adv) & (u.high >= floor + sr.TICK)]
    u = u[~u.symbol.str.match(TEST_TICKER)].drop_duplicates(['day', 'symbol'])
    u['split'] = np.where(u.day <= SPLITS['TRAIN'][1], 'TRAIN', 'VAL')
    u['wk'] = pd.to_datetime(u.day).dt.to_period('W-FRI').astype(str)
    log(f'[pop] {len(u)} symbol-days in the causal superset (TRAIN-H2 {int((u.split == "TRAIN").sum())}, '
        f'VAL {int((u.split == "VAL").sum())})')
    return u[['day', 'symbol', 'adv20', 'split', 'wk']].reset_index(drop=True)


def load_day_bars(con, day, syms):
    """{symbol: RTH minute bars (m,o,h,l,c,v)} from data/cache.db intraday_bars_1min (read-only)."""
    q = (f"select symbol, timestamp as t, open as o, high as h, low as l, close as c, volume as v from "
         f"intraday_bars_1min where bar_date=? and symbol in ({','.join('?' * len(syms))})")
    res = {}
    for s, g in pd.read_sql(q, con, params=[day] + list(syms)).groupby('symbol'):
        ts = pd.to_datetime(g.t, utc=True).dt.tz_convert(sr.ET)
        g = g.assign(m=(ts.dt.hour * 60 + ts.dt.minute).values).sort_values('m').drop_duplicates('m')
        res[s] = g[(g.m >= sr.OPEN_M) & (g.m < 960)][['m', 'o', 'h', 'l', 'c', 'v']].reset_index(drop=True)
    return res


def fetch_window(symbol, day, m_lo, m_hi):
    """SIP trades+quotes for one window, with the prevailing-quote history (as 1,427's fix): if no valid quote at or
    before the window start, the last valid quote in [start-900 s, start-5 s), else back to 09:30."""
    start, end = window_ns(day, m_lo, m_hi)
    t, q = sr.fetch_tape(symbol, end, span_s=(m_hi - m_lo) * 60)
    if not len(sr._valid(q)[lambda d: d.ts <= start]):
        w_end = start - sr.QUOTE_LOOKBACK_S * 10**9
        opn = sr.et_ns(day, sr.OPEN_M * 60)
        hq = sr._fetch_quotes(symbol, max(opn, start - sr.QUOTE_HISTORY_S * 10**9), w_end) if w_end > opn else q[:0]
        if not len(hq) and start - sr.QUOTE_HISTORY_S * 10**9 > opn:
            hq = sr._fetch_quotes(symbol, opn, start - sr.QUOTE_HISTORY_S * 10**9)
        if len(hq):
            q = pd.concat([hq.sort_values('ts').tail(1), q], ignore_index=True).sort_values('ts', kind='stable')
    return t, q


def load_1427_cache(day):
    """1,427's tapes for this day (break-bar windows, quote history merged) keyed symbol|break_m, or {}."""
    out = {}
    for name in (f'{day}.pkl.gz', f'qhist_{day}.pkl.gz'):
        path = os.path.join(sr.CACHE_DIR, name)
        if os.path.exists(path):
            with gzip.open(path, 'rb') as f:
                d = pickle.load(f)
            if name.startswith('qhist'):
                for k, hq in d.items():
                    if k in out and len(hq):
                        t, q = out[k]
                        out[k] = (t, pd.concat([hq, q], ignore_index=True).sort_values('ts', kind='stable'))
            else:
                out = d
    return out


# --------------------------------------------------------------------------------------------- run
def process_symbol_day(r, bars, p, floor, nbbo_half, cache, c1427):
    """Both variants for one symbol-day. Returns (rows, new_cache_entries)."""
    new, rows = {}, []

    def tape_for(a):
        """Cached tape for an armed window (own cache, then 1,427's for a one-minute window), else fetch."""
        k = f"{r.symbol}|{a['m_lo']}|{a['m_hi']}"
        if k in cache:
            return cache[k]
        if k in new:
            return new[k]
        k27 = sr.sig_key(r.symbol, a['m_hi'])
        if a['m_hi'] - a['m_lo'] == 1 and k27 in c1427:
            new[k] = c1427[k27]
        else:
            new[k] = fetch_window(r.symbol, r.day, a['m_lo'], a['m_hi'])
        return new[k]

    for variant, use_rv in (('causal', True), ('tick_rv', False)):
        cands = armed_crossing_bars(bars, r.adv20, p, floor, use_rv)
        base = dict(day=r.day, symbol=r.symbol, split=r.split, wk=r.wk, half='H2' if r.split == 'TRAIN' else '',
                    variant=variant, n_cross=len(cands))
        if not cands:
            rows.append({**base, 'status': 'not_armed'})
            continue
        base['b0_net_R'] = proxy_outcome(cands, bars)           # proxy outcome (see module docstring)
        status, a, e = resolve_day(cands, tape_for, r.day, r.adv20, p, tick_rv=not use_rv)
        if status != 'fill':
            rows.append({**base, 'status': status})
            continue
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
        xh = nbbo_half.get((r.day, r.symbol), np.nan)
        exit_half = xh if np.isfinite(xh) else e['half_spread']
        raw, cost, net, _ = sr.trade_result(fill, stop, e['half_spread'], px, exit_half)
        rows.append({**base, 'status': 'fill', 'fill': fill, 'stop': stop, 'level': a['level'], 'fill_min': fill_min,
                     'exit_m': exit_m, 'exit_price': px, 'why': why, 'R': R, 'raw_R': raw, 'cost_R': cost,
                     'net_R': net, 'exit_half_src': 'nbbo' if np.isfinite(xh) else 'fill_instant'})
    return rows, new


def run(workers):
    """Scan, fetch (resumable per-day cache sip_cache/c1438_{day}.pkl.gz), resolve, write per-variant CSVs."""
    p, floor, min_adv = live_params()
    log(f'[params] {p} | min_price {floor} | min_adv20 {min_adv}')
    pop = load_population(p, floor, min_adv)
    nb = pd.read_csv(NBBO_CSV, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    nbbo_half = (0.5 * nb.drop_duplicates(['day', 'symbol']).set_index(['day', 'symbol']).spread_mean).to_dict()
    con = sqlite3.connect(sr.CACHE_DB_URI, uri=True, timeout=120)
    rows, n_nobars = [], 0
    days = sorted(pop.day.unique())
    for di, day in enumerate(days):
        sub = pop[pop.day == day]
        bars = load_day_bars(con, day, sub.symbol.tolist())
        path = os.path.join(sr.CACHE_DIR, f'c1438_{day}.pkl.gz')
        cache = {}
        if os.path.exists(path):
            with gzip.open(path, 'rb') as f:
                cache = pickle.load(f)
        c1427 = load_1427_cache(day)
        jobs = []
        for r in sub.itertuples():
            b = bars.get(r.symbol)
            if b is None or len(b) < p.consol_bars + 2:
                n_nobars += 1
                continue
            jobs.append((r, b))
        new_all = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(process_symbol_day, r, b, p, floor, nbbo_half, cache, c1427) for r, b in jobs]
            for fu in futs:
                try:
                    rr, new = fu.result()
                    rows += rr
                    new_all.update(new)
                except Exception as e:                           # noqa: BLE001
                    log(f'[run] ERROR {day}: {type(e).__name__}: {e} — symbol-day LOST (re-run to resume)')
        if new_all:
            cache.update(new_all)
            tmp = path + '.tmp'
            with gzip.open(tmp, 'wb') as f:
                pickle.dump(cache, f)
            os.replace(tmp, path)
        if (di + 1) % 10 == 0 or di == len(days) - 1:
            d = pd.DataFrame(rows)
            nf = int(((d.status == 'fill') & (d.variant == 'causal')).sum()) if len(d) else 0
            log(f'[run] day {di + 1}/{len(days)} ({day}) | causal fills so far {nf} | new windows this day '
                f'{len(new_all)}')
    con.close()
    if n_nobars:
        log(f'[run] WARNING: {n_nobars} superset symbol-days have no (or < K+2) cache.db minute bars — not simulable')
    res = pd.DataFrame(rows)
    for v in ('causal', 'tick_rv'):
        res[res.variant == v].to_csv(OUT_CSV.format(variant=v), index=False)
    return res, n_nobars


def report(res, n_nobars):
    """REPORT_1438.md: per split and variant the stats, plus overlap with 1,427's fills and the 1,438-only outcome."""
    lines = ['# REPORT — cell 1,438: causal arming (PREREG_1438.md) — TRAIN-H2 / VAL only, TEST NOT read', '',
             'Generated by research/hod_entry/causal_arming.py. Scope = superset symbol-days with >= 1 armed bar whose '
             'next bar crosses the trigger (minute bars). Coverage = usable tape / scope. Winner/loser for the '
             'missingness gap and the "no-fill cohort" column = PROXY outcome (fill at the trigger on the first '
             f'crossing armed bar, raw R). Superset symbol-days without cache.db minute bars: {n_nobars}.', '',
             sr.TABLE_HEAD]
    for v in ('causal', 'tick_rv'):
        for cn, sp in (('TRAIN-H2', 'TRAIN'), ('VAL', 'VAL')):
            d = res[(res.variant == v) & (res.split == sp) & (res.status != 'not_armed')].copy()
            lines.append(sr.fmt_row(f'{v} — {cn}', sr.score(d, res[(res.variant == v) & (res.split == sp)])))
    c27 = pd.read_csv(C1427_CSV, dtype={'day': str, 'symbol': str})
    c27 = c27[c27.status == 'fill'][['day', 'symbol', 'split', 'net_R']].rename(columns={'net_R': 'R27'})
    lines += ['', '## Overlap with cell 1,427 fills (same day, symbol)', '',
              '| split | 1,438 fills | both (1,438 R / 1,427 R) | 1,438-only (R) | 1,427-only (R) |', '|---|---|---|---|---|']
    for cn, sp in (('TRAIN-H2', 'TRAIN'), ('VAL', 'VAL')):
        f = res[(res.variant == 'causal') & (res.split == sp) & (res.status == 'fill')][['day', 'symbol', 'net_R']]
        c = c27[c27.split == sp]
        m = f.merge(c, on=['day', 'symbol'], how='outer', indicator=True)
        b, lo, ro = m[m._merge == 'both'], m[m._merge == 'left_only'], m[m._merge == 'right_only']
        lines.append(f'| {cn} | {len(f)} | {len(b)} ({b.net_R.mean():+.3f} / {b.R27.mean():+.3f}) | '
                     f'{len(lo)} ({lo.net_R.mean():+.3f}) | {len(ro)} ({ro.R27.mean():+.3f}) |')
    v = sr.score(res[(res.variant == 'causal') & (res.split == 'VAL') & (res.status != 'not_armed')],
                 res[(res.variant == 'causal') & (res.split == 'VAL')])
    legs = [('mean net R ≥ +0.10', v['mean_R'] >= 0.10, f'{v["mean_R"]:+.3f}'),
            ('day-clustered t ≥ 2', v['t'] >= 2, f'{v["t"]:.2f}'),
            ('ex-top-5 % > 0', v['extop5'] > 0, f'{v["extop5"]:+.3f}'),
            ('coverage ≥ 80 %', v['coverage'] >= 0.8, f'{v["coverage"]*100:.1f} %'),
            ('gap ≤ 5 pp', v['miss_gap'] <= 0.05, f'{v["miss_gap"]*100:.1f} pp'),
            ('≥ 3 fills/week', v['fills_wk'] >= 3, f'{v["fills_wk"]:.1f}')]
    ok = all(x[1] for x in legs)
    lines += ['', '## VAL pass bar (causal variant)', '', '| leg | value | pass |', '|---|---|---|']
    lines += [f'| {n} | {val} | {"PASS" if g else "FAIL"} |' for n, g, val in legs]
    lines += ['', f'**VAL VERDICT: {"PASS" if ok else "FAIL"}** — TEST not read (coordinator decides).', '']
    with open(REPORT_MD, 'w') as fh:
        fh.write('\n'.join(lines))
    log('[report] VAL ' + ('PASS' if ok else 'FAIL') + ': ' + '; '.join(f'{n}={x}' for n, _, x in legs))


def main(argv=None):
    """CLI: --workers N. Runs TRAIN-H2 + VAL only; TEST is out of scope of this script."""
    ap = argparse.ArgumentParser(description='cell 1,438 causal arming (TRAIN-H2 + VAL)')
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args(argv)
    res, n_nobars = run(a.workers)
    report(res, n_nobars)
    return 0


if __name__ == '__main__':
    sys.exit(main())
