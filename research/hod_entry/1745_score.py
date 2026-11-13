"""PREREG 1,745 scorer (agent B): the live HOD-break spec on the monthly top-300 liquid universe, 2026-01-02..2026-11-12.

Stages (run in order, each resumable):
  signals : per symbol-day, trading.hod_break.arm_state + arm_gate_reject (THE live helpers, min_r_pct 1.5, defaults
            otherwise) -> first resting-stop-limit fill at the bar level -> 1745_fills.csv (raw, zero cost) and
            1745_arms.csv (every crossing arm: obtainability).
  quotes  : Alpaca /v2/stocks/quotes at each fill minute (all fills, >= the 400-fill calibration sample), cached in
            1745_quotes_cache/ (one json per symbol-day-minute); <= 170 requests/min; keys never printed.
  score   : costs (measured entry half-spread; per-name x hour-bucket median table for exit legs), books uncapped and
            capped (trading.hod_break.run_book 8/day, 4 concurrent), splits, ex-top-5 %, placebo, cadence CSV,
            1745_summary.csv.
Bar-level fill model (bars only, no tape): bar j+1's high >= trigger fills at max(open, trigger) iff <= limit
(= level x 1.0015); open above the limit = no fill (the obtainability share). A fill bar whose low <= stop is a
stop in the fill bar (conservative; `stop_bar` flag, optimistic variant also stored). Exits from the next bar:
low <= stop -> min(open, stop); high >= 2R target -> target; 15:55 bar open. Costs: entry = measured half-spread +
8 bps chase; stop leg = 13 bps slip; target/eod leg = table half-spread + 2 bp.
Usage: nice -n 19 python3 1745_score.py {signals|quotes|score}
"""
import glob
import json
import os
import re
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
from trading.hod_break import HodBreakParams, arm_gate_reject, arm_state, run_book  # noqa: E402

BARS = os.path.join(HERE, '1745_bars')
FILLS = os.path.join(HERE, '1745_fills.csv')
ARMS = os.path.join(HERE, '1745_arms.csv')
QCACHE = os.path.join(HERE, '1745_quotes_cache')
SUMMARY = os.path.join(HERE, '1745_summary.csv')
CAD_CSV = os.path.join(HERE, '1745_capped_trades.csv')
TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')
PARAMS = HodBreakParams(min_r_pct=1.5)          # live floor; every other knob = the class default (PREREG)
MIN_PRICE = 5.0                                  # PREREG universe floor (the live $20 is a reported sensitivity)
EOD_M, TARGET_R = 955, 2.0
CHASE, STOP_SLIP, SLIP_BP = 0.0008, 0.0013, 0.0002
SPLIT_EDGE = '2026-07-01'
MAX_DAY, MAX_CONC = 8, 4
CAL_N = 400


def log(msg):
    """Verbose progress line (flushed)."""
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def load_adv20(symbols, days):
    """ADV20 (mean SHARE volume of the 20 daily_bars rows strictly before `day`) per (symbol, day); read-only DB."""
    con = sqlite3.connect('file:' + os.path.join(ROOT, 'data/cache.db') + '?mode=ro', uri=True)
    q = ','.join('?' * len(symbols))
    d = pd.read_sql_query(f"select symbol, bar_date, volume from daily_bars where symbol in ({q}) and bar_date >= '2025-09-01' "
                          f"order by symbol, bar_date", con, params=list(symbols))
    con.close()
    d['adv20'] = d.groupby('symbol').volume.transform(lambda s: s.rolling(20, min_periods=15).mean().shift(1))
    return d.set_index(['symbol', 'bar_date']).adv20.to_dict()


def walk_exit(o, h, l, m, start, stop, target):
    """Exit from bar index `start` on: stop first (price min(open, stop)), then target touch, then 15:55 open."""
    for k in range(start, len(o)):
        if m[k] >= EOD_M:
            return int(m[k]), float(o[k]), 'eod'
        if l[k] <= stop:
            return int(m[k]), float(min(o[k], stop)), 'stop'
        if h[k] >= target:
            return int(m[k]), float(target), 'target'
    return int(m[-1]), float(o[-1]), 'eod_fallback'


def day_signals(day, g, adv, params, min_price):
    """All arms and first fills for one day. Returns (arm_rows, fill_rows)."""
    arms, fills = [], []
    for sym, b in g.groupby('symbol', sort=False):
        a20 = adv.get((sym, day))
        if TEST_TICKER.match(sym) or a20 is None or not np.isfinite(a20) or a20 <= 0 or len(b) < 30:
            continue
        b = b.sort_values('t')
        t = b.t.dt.tz_convert('America/New_York')
        m = (t.dt.hour * 60 + t.dt.minute).values.astype(int)
        o, h, l, c, v = (b[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v'))
        if m[0] != 570:
            continue                                   # the 09:30 open anchors min_dist_open
        hod = np.maximum.accumulate(h)
        for j in np.nonzero(h[1:] >= hod[:-1] + 0.01 - 1e-9)[0]:   # cheap pre-filter: next bar can cross HOD_j + 1c
            j = int(j)
            if m[j + 1] != m[j] + 1:
                continue
            ar = arm_state(o, h, l, v, m, j, a20, params)
            if ar is None or arm_gate_reject(ar['trigger'], ar['stop'], min_price, params.min_r_pct) is not None:
                continue
            if h[j + 1] < ar['trigger'] - 1e-9:
                continue
            px = max(float(o[j + 1]), ar['trigger'])
            ok = px <= ar['limit'] + 1e-9
            arms.append(dict(day=day, symbol=sym, j=j, m=int(m[j + 1]), obtainable=int(ok), open_gt_limit=int(o[j + 1] > ar['limit'])))
            if not ok:
                continue
            stop, R = ar['stop'], px - ar['stop']
            if R <= 0:
                continue
            target = px + TARGET_R * R
            stop_bar = bool(l[j + 1] <= stop)
            if stop_bar:
                xm, xp, why = int(m[j + 1]), (float(stop) if o[j + 1] > stop else float(o[j + 1])), 'stop'
            else:
                xm, xp, why = walk_exit(o, h, l, m, j + 2, stop, target)
            xm_a, xp_a, why_a = walk_exit(o, h, l, m, j + 2, stop, target)      # optimistic: ignore the fill bar's low
            fills.append(dict(day=day, symbol=sym, j=j, fill_m=int(m[j + 1]), level=ar['level'], trigger=ar['trigger'], limit=ar['limit'],
                              stop=stop, fill=px, R=R, r_pct=R / px * 100.0, stop_bar=int(stop_bar), exit_m=xm, exit_px=xp, why=why,
                              exit_m_alt=xm_a, exit_px_alt=xp_a, why_alt=why_a, raw_R=(xp - px) / R, raw_R_alt=(xp_a - px) / R))
            break                                      # first fill per symbol-day only
    return arms, fills


def stage_signals():
    """Walk every day parquet; write 1745_fills.csv / 1745_arms.csv."""
    files = sorted(glob.glob(os.path.join(BARS, '*.parquet')))
    uni = pd.read_csv(os.path.join(HERE, '1745_universe.csv'))
    adv = load_adv20(sorted(uni.symbol.unique()), None)
    arms, fills = [], []
    for i, f in enumerate(files):
        day = os.path.basename(f)[:-8]
        g = pd.read_parquet(f)
        mon = day[:7]
        g = g[g.symbol.isin(set(uni[uni.month == mon].symbol))]          # the month's universe only
        a, fl = day_signals(day, g, adv, PARAMS, MIN_PRICE)
        arms += a; fills += fl
        if i % 20 == 0 or i == len(files) - 1:
            log(f'[signals] {i + 1}/{len(files)} {day} arms={len(arms)} fills={len(fills)}')
    pd.DataFrame(arms).to_csv(ARMS, index=False)
    pd.DataFrame(fills).to_csv(FILLS, index=False)
    log(f'[signals] DONE arms={len(arms)} fills={len(fills)}')


# ------------------------------------------------------------------------------------------------ quotes
def stage_quotes():
    """Measured NBBO half-spread at each fill minute (median over the minute's valid quotes), cached."""
    import requests
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, '.env'))
    hdr = {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'], 'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']}
    os.makedirs(QCACHE, exist_ok=True)
    fl = pd.read_csv(FILLS)
    rng = np.random.default_rng(1745)
    order = rng.permutation(len(fl))          # the first CAL_N of this order = the calibration sample
    fl = fl.iloc[order].reset_index(drop=True)
    last, n_req, n_new = 0.0, 0, 0
    for i, r in fl.iterrows():
        path = os.path.join(QCACHE, f'{r.day}_{r.symbol}_{int(r.fill_m)}.json')
        if os.path.exists(path):
            continue
        hh, mm = divmod(int(r.fill_m), 60)
        off = '-05:00' if r.day >= '2026-11-01' or r.day < '2026-03-08' else '-04:00'
        start = f'{r.day}T{hh:02d}:{mm:02d}:00{off}'
        end = f'{r.day}T{hh:02d}:{mm:02d}:59{off}'
        wait = 0.36 - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        last = time.time()
        for attempt in range(6):
            try:
                resp = requests.get('https://data.alpaca.markets/v2/stocks/quotes', headers=hdr, timeout=60,
                                    params=dict(symbols=r.symbol, start=start, end=end, limit=1000, feed='sip'))
                if resp.status_code == 429:
                    log('[quotes] WARNING 429, sleeping 20 s')
                    time.sleep(20); continue
                resp.raise_for_status()
                qs = resp.json().get('quotes', {}).get(r.symbol, [])
                break
            except Exception as e:                       # network error: logged, retried
                log(f'[quotes] WARNING {r.symbol} {r.day}: {type(e).__name__} attempt {attempt}')
                time.sleep(3)
        else:
            log(f'[quotes] ERROR giving up {r.symbol} {r.day} {r.fill_m}')
            continue
        n_req += 1
        hs = [(q['ap'] - q['bp']) / 2 for q in qs if q.get('ap', 0) > 0 and q.get('bp', 0) > 0 and q['ap'] > q['bp']]
        mids = [(q['ap'] + q['bp']) / 2 for q in qs if q.get('ap', 0) > 0 and q.get('bp', 0) > 0 and q['ap'] > q['bp']]
        out = dict(n_quotes=len(qs), n_valid=len(hs), half=float(np.median(hs)) if hs else None,
                   half_bps=float(np.median([a / b * 1e4 for a, b in zip(hs, mids)])) if hs else None,
                   half_first=hs[0] if hs else None)
        with open(path + '.tmp', 'w') as fh:
            json.dump(out, fh)
        os.replace(path + '.tmp', path)
        n_new += 1
        if n_new % 100 == 0:
            log(f'[quotes] {n_new} new, position {i + 1}/{len(fl)} requests={n_req}')
    log(f'[quotes] DONE new={n_new} total_cached={len(os.listdir(QCACHE))}')


# ------------------------------------------------------------------------------------------------ scoring
def load_half(fl):
    """Attach the measured entry half-spread (bps of price) per fill; NaN where unmeasured/empty."""
    hb = []
    for r in fl.itertuples():
        p = os.path.join(QCACHE, f'{r.day}_{r.symbol}_{int(r.fill_m)}.json')
        if os.path.exists(p):
            d = json.load(open(p))
            hb.append(d['half_bps'] if d['half_bps'] is not None else np.nan)
        else:
            hb.append(np.nan)
    fl = fl.copy()
    fl['half_bps_meas'] = hb
    return fl


def half_table(fl):
    """Per-name x hour-bucket median half-spread (bps) from the measured fills, with name and global fallbacks."""
    d = fl.dropna(subset=['half_bps_meas']).copy()
    d['hb'] = d.fill_m // 60
    by_nh = d.groupby(['symbol', 'hb']).half_bps_meas.median().to_dict()
    by_n = d.groupby('symbol').half_bps_meas.median().to_dict()
    by_h = d.groupby('hb').half_bps_meas.median().to_dict()
    glob_med = float(d.half_bps_meas.median())
    return by_nh, by_n, by_h, glob_med


def lookup_half(tab, symbol, minute):
    """Median half-spread bps for (symbol, hour of `minute`) falling back to name, hour, global; logs the fallback tier."""
    by_nh, by_n, by_h, g = tab
    hb = int(minute) // 60
    for tier, v in (('name_hour', by_nh.get((symbol, hb))), ('name', by_n.get(symbol)), ('hour', by_h.get(hb))):
        if v is not None and np.isfinite(v):
            return float(v), tier
    return g, 'global'


def apply_costs(fl, tab, alt=False, zero=False, half_mult=1.0):
    """Net R per fill. Entry: measured half-spread (table where unmeasured) + CHASE; stop leg STOP_SLIP; target/eod leg
    table half-spread + SLIP_BP. `alt` uses the optimistic exit; `zero` the raw R."""
    sfx = '_alt' if alt else ''
    out = []
    for r in fl.itertuples():
        raw = getattr(r, 'raw_R_alt' if alt else 'raw_R')
        if zero:
            out.append(raw); continue
        hb_in = r.half_bps_meas if np.isfinite(r.half_bps_meas) else lookup_half(tab, r.symbol, r.fill_m)[0]
        xp, why, xm = getattr(r, 'exit_px' + sfx), getattr(r, 'why' + sfx), getattr(r, 'exit_m' + sfx)
        entry_cost = r.fill * (hb_in * half_mult * 1e-4 + CHASE)
        if why == 'stop':
            exit_cost = xp * STOP_SLIP
        else:
            exit_cost = xp * (lookup_half(tab, r.symbol, xm)[0] * half_mult * 1e-4 + SLIP_BP)
        out.append(raw - (entry_cost + exit_cost) / r.R)
    return np.array(out)


def day_t(x, days):
    """Mean, day-clustered t (cluster-robust SE of the mean), and iid t."""
    x = np.asarray(x, float); n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    mu = x.mean()
    s = pd.Series(x - mu).groupby(np.asarray(days)).sum()
    se_c = np.sqrt((s ** 2).sum()) / n
    return mu, mu / se_c if se_c > 0 else np.nan, mu / (x.std(ddof=1) / np.sqrt(n))


def split_stats(df, col, label, n_weeks):
    """The PREREG per-split statistic block for net-R column `col`."""
    x = df[col].values
    if len(x) < 3:
        return dict(label=label, n=len(x))
    mu, tc, ti = day_t(x, df.day.values)
    k = int(np.ceil(0.05 * len(x)))
    ex5 = np.sort(x)[:-k].mean()
    wk = df.groupby('wk')[col].sum()
    cum = np.cumsum(df.sort_values(['day', 'fill_m'])[col].values)
    return dict(label=label, n=len(x), fills_wk=len(x) / n_weeks, r_pct_med=float(df.r_pct.median()), mean_R=mu, t_day=tc, t_iid=ti,
                ex_top5=ex5, win_cap3=float(np.minimum(x, 3).mean()), stop_share=float((df.why == 'stop').mean()),
                green_wk=float((wk > 0).mean()), wk_p10=float(np.percentile(wk, 10)) if len(wk) else np.nan,
                max_dd=float((np.maximum.accumulate(cum) - cum).max()), weeks=n_weeks)


def placebo(fl, bars_by_day, tab, rng, draws=300):
    """Same geometry from a random later minute: for each fill, candidates = open of bars k > arm bar + 1 up to the last
    entry minute, stop = open x (1 - r_pct), 2R target, same exits and costs in bps. Returns the per-draw mean net R."""
    cand = []
    for r in fl.itertuples():
        b = bars_by_day[(r.day, r.symbol)]
        o, h, l, m = b['o'], b['h'], b['l'], b['m']
        rows = []
        for k in range(r.j + 2, len(o)):
            if m[k] > PARAMS.last_entry_minute:
                break
            f = o[k]; Rp = f * r.r_pct / 100.0
            stop, tgt = f - Rp, f + TARGET_R * Rp
            if l[k] <= stop:
                xm, xp, why = int(m[k]), float(stop if o[k] > stop else o[k]), 'stop'
            else:
                xm, xp, why = walk_exit(o, h, l, m, k + 1, stop, tgt)
            hb_in = r.half_bps_meas if np.isfinite(r.half_bps_meas) else lookup_half(tab, r.symbol, r.fill_m)[0]
            ec = f * (hb_in * 1e-4 + CHASE)
            xc = xp * STOP_SLIP if why == 'stop' else xp * (lookup_half(tab, r.symbol, xm)[0] * 1e-4 + SLIP_BP)
            rows.append((xp - f - ec - xc) / Rp)
        cand.append(np.array(rows) if rows else np.array([r.net_R]))
    means = np.empty(draws)
    for d in range(draws):
        means[d] = np.mean([c[rng.integers(len(c))] for c in cand])
    return means


def stage_score():
    """Costs, books, splits, placebo, cadence CSV, summary."""
    fl = load_half(pd.read_csv(FILLS))
    arms = pd.read_csv(ARMS)
    fl['split'] = np.where(fl.day < SPLIT_EDGE, 'TRAIN', 'VAL')
    fl['wk'] = pd.to_datetime(fl.day).dt.to_period('W-FRI').astype(str)
    n_meas = int(fl.half_bps_meas.notna().sum())
    log(f'[score] fills={len(fl)} measured half-spread={n_meas} median={fl.half_bps_meas.median():.2f} bps '
        f'p90={fl.half_bps_meas.quantile(0.9):.2f}')
    tab = half_table(fl)
    fl['net_R'] = apply_costs(fl, tab)
    fl['net_R_alt'] = apply_costs(fl, tab, alt=True)
    fl['zero_R'] = apply_costs(fl, tab, zero=True)
    fl['net_R_h2'] = apply_costs(fl, tab, half_mult=2.0)
    # capped book: ONE helper shared with the live engine
    rows = [(r.day, r.fill_m, r.exit_m, r.symbol, i) for i, r in enumerate(fl.itertuples())]
    taken = {t[4] for t in run_book(rows, MAX_DAY, MAX_CONC)}
    fl['capped'] = [i in taken for i in range(len(fl))]
    fl.to_csv(FILLS.replace('1745_fills.csv', '1745_fills_scored.csv'), index=False)
    weeks = {s: len({w for w in pd.to_datetime(fl.day.unique()).to_period('W-FRI')}) for s in ('TRAIN', 'VAL')}
    days_all = sorted(f[-18:-8] for f in glob.glob(os.path.join(BARS, '*.parquet')))
    nd = {'TRAIN': sum(d < SPLIT_EDGE for d in days_all), 'VAL': sum(d >= SPLIT_EDGE for d in days_all)}
    nweeks = {s: nd[s] / 5.0 for s in nd}
    # obtainability
    arms['split'] = np.where(arms.day < SPLIT_EDGE, 'TRAIN', 'VAL')
    obt = arms.groupby('split').obtainable.mean().to_dict()
    log(f'[score] crossing arms {len(arms)} obtainable share {obt}; open>limit share {arms.open_gt_limit.mean():.3f}; '
        f'days {nd}')
    res = []
    th = {'TRAIN_H1': (fl.day < '2026-04-01'), 'TRAIN_H2': (fl.day >= '2026-04-01') & (fl.day < SPLIT_EDGE)}
    for book, mask in (('uncapped', np.ones(len(fl), bool)), ('capped', fl.capped.values)):
        for col, tag in (('net_R', 'net'), ('zero_R', 'zero'), ('net_R_alt', 'net_optfillbar'), ('net_R_h2', 'net_2xhalf')):
            for s in ('TRAIN', 'VAL', 'TRAIN_H1', 'TRAIN_H2'):
                sub = fl[mask & ((fl.split == s).values if s in nd else th[s].values)]
                nw = nweeks[s] if s in nd else (nweeks['TRAIN'] / 2)
                res.append(dict(book=book, cost=tag, **split_stats(sub, col, s, nw)))
    # price >= 20 sensitivity (capped, net)
    hi = fl.capped & (fl.fill >= 20)
    for s in ('TRAIN', 'VAL'):
        sub = fl[hi & (fl.split == s)]
        res.append(dict(book='capped_px20_postfilter', cost='net', **split_stats(sub, 'net_R', s, nweeks[s])))
    summ = pd.DataFrame(res)
    summ.to_csv(SUMMARY, index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    log('\n' + summ.round(3).to_string())
    # placebo on the capped book per split
    bars_by_day = {}
    cap = fl[fl.capped]
    for day, g in cap.groupby('day'):
        bd = pd.read_parquet(os.path.join(BARS, f'{day}.parquet'))
        bd = bd[bd.symbol.isin(set(g.symbol))]
        for sym, b in bd.groupby('symbol'):
            b = b.sort_values('t'); tt = b.t.dt.tz_convert('America/New_York')
            bars_by_day[(day, sym)] = dict(o=b.o.values.astype(float), h=b.h.values.astype(float), l=b.l.values.astype(float),
                                           m=(tt.dt.hour * 60 + tt.dt.minute).values.astype(int))
    rng = np.random.default_rng(745)
    for s in ('TRAIN', 'VAL'):
        sub = cap[cap.split == s]
        means = placebo(sub, bars_by_day, tab, rng)
        real = sub.net_R.mean()
        pct = float((means < real).mean() * 100)
        log(f'[placebo] {s}: real mean {real:.4f} R; placebo mean {means.mean():.4f} sd {means.std():.4f}; real percentile {pct:.1f}')
        pd.DataFrame([dict(split=s, real=real, placebo_mean=means.mean(), placebo_sd=means.std(), percentile=pct)]).to_csv(
            os.path.join(HERE, f'1745_placebo_{s}.csv'), index=False)
    cap[['day', 'symbol', 'net_R', 'split']].rename(columns={'day': 'date', 'net_R': 'pnl_R'}).to_csv(CAD_CSV, index=False)
    log('[score] DONE')


if __name__ == '__main__':
    {'signals': stage_signals, 'quotes': stage_quotes, 'score': stage_score}[sys.argv[1]]()
