#!/usr/bin/env python3
"""Cell 1,709 -- ORB exits re-simulated at the LIVE / tape-obtainable fill price.

Implements research/orb_exit_resim/PREREG_1709.md (frozen 2026-10-29).

    python3 resim_1709.py --cell a     # mechanism: 123 live/paper fills with bars
    python3 resim_1709.py --cell b     # money: 617 replayed fills at delay 0 s and 30 s
    python3 resim_1709.py --cell b --report   # only (re)build tables from the CSV

Exit rule (ONE implementation, `simulate_exit`), taken from the live code:
  * stop = max(range_low, fill - k*ATR14(T-1)) (trading.orb_winner_stack.floored_stop,
    k = 0.25, anchored on the fill like live); fail-open to range_low with a share report.
  * Rule M (tag_bb): at the close of the fill bar, if the market breakout bar's close
    position < 0.5 -> exit at that close.  Rule D (tag_b1): at the close of the bar after the
    fill bar, if (fill - low)/range_size >= 0.75 -> exit at fill - 0.5*range_size.
  * lock: bar high reaches fill + 1.75*range_size -> stop to fill + 0.5*range_size.
  * force close: close of the last bar that closes by 15:45 ET (live ledger: 15:45:01;
    PREREG text says 15:55 -> run as a sensitivity).
  * within a bar the stop is checked BEFORE arming the lock (conservative); entry bar cannot
    stop unless its CLOSE is at/below the stop (sensitivity: its LOW can).
  * stops fill at min(stop, bar open) x (1 - 35 bps); Rule M/force close at the bar close.
Not simulated (stated in RESULT): add-on at 1R and the 40 % @ 3R scale-out (both ON in orb.yaml).
Read-only on every database.  No data is fetched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading.orb_winner_stack import atr14_t1, floored_stop  # noqa: E402

HERE = f'{ROOT}/research/orb_exit_resim'
EXEC = f'{ROOT}/research/exec_learnings_oct2026'
BARS_DBS = [  # (label, path, kind) -- first hit wins; all opened read-only
    ('sip', f'{ROOT}/research/bf_zero/bars_sip.db', 'sip'),
    ('y2024', f'{ROOT}/research/day_breadth/y2024/bars.db', 'sip'),
    ('gapper', f'{ROOT}/research/hod_entry/bars_sip_gapper.db', 'sip'),
    ('cache', f'{ROOT}/data/cache.db', 'cache'),
]
TRADES_DB = f'{ROOT}/data/trades.db'
CACHE_DB = f'{ROOT}/data/cache.db'
ET = 'America/New_York'

STOP_SLIP = 0.0035      # cell 1,443 measured stop slippage, mean
LOCK_ARM_R, LOCK_STOP_R = 1.75, 0.5
ATR_K = 0.25
RULE_M_THR, RULE_D_REVERT, RULE_D_EXIT = 0.5, 0.75, -0.5
R_DENOM = 375.0         # cell 1,708 convention for the replay population
BOOT = 2000

VARIANTS = {   # name -> (entry_bar_stop, force_close_hhmm, nonstop_cost_bps)
    'main': ('close', (15, 45), 0.0),
    'ebstop': ('low', (15, 45), 0.0),
    'fc1555': ('close', (15, 55), 0.0),
    'cost10': ('close', (15, 45), 10.0),
}


def log(msg: str) -> None:
    """Timestamped, flushed progress line."""
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


# ------------------------------------------------------------------ data access
class BarStore:
    """Read-only minute-bar lookup across the local stores (no fetching)."""

    def __init__(self):
        """Open every store with mode=ro."""
        self.conns = []
        for label, path, kind in BARS_DBS:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                self.conns.append((label, kind, sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=30)))
        self.cache = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True, timeout=30)

    def get(self, symbol: str, day: str):
        """Return (DataFrame[ts,o,h,l,c] RTH sorted, source label) or (None, 'uncovered')."""
        for label, kind, con in self.conns:
            if kind == 'sip':
                q = 'select t,o,h,l,c from bars where symbol=? and day=? order by t'
            else:
                q = ('select timestamp,open,high,low,close from intraday_bars_1min '
                     'where symbol=? and bar_date=? order by timestamp')
            rows = con.execute(q, (symbol, day)).fetchall()
            if len(rows) >= 30:
                df = pd.DataFrame(rows, columns=['ts', 'o', 'h', 'l', 'c'])
                df['ts'] = pd.to_datetime(df['ts'], utc=True)
                et = df['ts'].dt.tz_convert(ET)
                mins = et.dt.hour * 60 + et.dt.minute
                df = df[(mins >= 570) & (mins < 960)].reset_index(drop=True)
                if len(df) >= 30:
                    return df, label
        return None, 'uncovered'

    def atr14(self, symbol: str, day: str):
        """ATR14 ending T-1 from cache.db daily_bars (read-only), None if < 15 bars."""
        rows = self.cache.execute(
            'select high,low,close from daily_bars where symbol=? and bar_date<? '
            'order by bar_date desc limit 40', (symbol, day)).fetchall()
        if len(rows) < 15:
            return None
        df = pd.DataFrame(rows[::-1], columns=['high', 'low', 'close'])
        return atr14_t1(df)


# ------------------------------------------------------------------ exit simulation
def simulate_exit(bars: pd.DataFrame, day: str, fill_ts: pd.Timestamp, fill: float,
                  range_high: float, range_low: float, atr14, variant: str) -> dict:
    """Walk the exit rules from `fill` at `fill_ts` over `bars`; return exit dict.

    Returns dict(reason, exit_px, exit_ts, stop_px, floor_status, fill_idx) -- exit_px is the net
    sale price per share (slippage included); reason in stop/lock/tag_bb/tag_b1/force_close/no_bars.
    """
    eb_mode, (fch, fcm), nonstop_bps = VARIANTS[variant]
    ts = bars['ts'].values
    o, h, l, c = (bars[k].to_numpy(float) for k in ('o', 'h', 'l', 'c'))
    n = len(bars)
    ft = np.datetime64(fill_ts.tz_convert('UTC').tz_localize(None), 'ns')
    tsn = ts.astype('datetime64[ns]')
    idx = int(np.searchsorted(tsn, ft, side='right')) - 1
    if idx < 0 or (ft - tsn[idx]) >= np.timedelta64(60, 's'):
        idx = int(np.searchsorted(tsn, ft, side='left'))   # missing minute: next available bar
    if idx >= n:
        return dict(reason='no_bars', exit_px=fill, exit_ts=fill_ts, stop_px=np.nan, floor_status='na', fill_idx=-1)
    rsize = range_high - range_low
    stop, fstat = floored_stop(round(range_low, 2), fill, atr14, ATR_K)   # live stop = round(range_low, 2)
    stop_init = stop
    trig = fill + LOCK_ARM_R * rsize
    lockpx = fill + LOCK_STOP_R * rsize
    cost = 1.0 - nonstop_bps / 1e4
    sl = 1.0 - STOP_SLIP
    et = pd.DatetimeIndex(bars['ts']).tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()
    fc_min = fch * 60 + fcm

    def done(reason, px, i):
        return dict(reason=reason, exit_px=px, exit_ts=bars['ts'].iloc[i], stop_px=stop_init,
                    floor_status=fstat, fill_idx=idx)

    # breakout bar = first bar from 09:35 ET whose high reaches the trigger (market source)
    start = int(np.argmax(mins >= 575)) if (mins >= 575).any() else 0
    hit = np.where(h[start:] >= range_high)[0]
    bb = min(start + int(hit[0]), idx) if len(hit) else idx
    # Rule M at the close of the fill bar (live evaluates once the breakout bar has closed)
    brng = h[bb] - l[bb]
    if brng > 0 and (c[bb] - l[bb]) / brng < RULE_M_THR:
        return done('tag_bb', c[idx] * cost, idx)
    # entry-bar stop rule
    if eb_mode == 'close' and c[idx] <= stop:
        return done('stop', min(stop, c[idx]) * sl, idx)
    if eb_mode == 'low' and l[idx] <= stop:
        return done('stop', stop * sl, idx)
    armed = False
    for i in range(idx + 1, n):
        if mins[i] >= fc_min:                      # bar opens at/after force close -> prior close
            return done('force_close', c[i - 1] * cost, i - 1)
        if o[i] <= stop:
            return done('lock' if armed else 'stop', o[i] * sl, i)
        if l[i] <= stop:
            return done('lock' if armed else 'stop', stop * sl, i)
        if i == idx + 1 and (fill - l[i]) / rsize >= RULE_D_REVERT:   # Rule D at bar-1 close
            return done('tag_b1', (fill + RULE_D_EXIT * rsize) * sl, i)
        if not armed and h[i] >= trig:
            armed = True
            stop = max(stop, lockpx)
    return done('force_close', c[n - 1] * cost, n - 1)


# ------------------------------------------------------------------ statistics
def day_boot_ci(vals: np.ndarray, days: np.ndarray, seed: int = 7):
    """Day-clustered bootstrap (resample days) 95 % CI of the mean; returns (mean, lo, hi)."""
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    udays, inv = np.unique(days, return_inverse=True)
    sums = np.bincount(inv, weights=vals)
    cnts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    ks = rng.integers(0, len(udays), size=(BOOT, len(udays)))
    means = sums[ks].sum(1) / cnts[ks].sum(1)
    return (float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def ex_top(vals: np.ndarray, frac: float = 0.05) -> float:
    """Mean after dropping the top ceil(frac*n) values."""
    v = np.sort(np.asarray(vals, float))[::-1]
    k = int(math.ceil(frac * len(v)))
    return float(v[k:].mean()) if len(v) > k else float('nan')


# ------------------------------------------------------------------ cell a
def ledger_rows() -> pd.DataFrame:
    """fills_1702 joined with pattern_data (range_high/low, atr14) from trades.db (read-only)."""
    f = pd.read_csv(f'{EXEC}/fills_1702.csv')
    con = sqlite3.connect(f'file:{TRADES_DB}?mode=ro', uri=True)
    pdx = pd.read_sql_query(
        "select id, json_extract(pattern_data,'$.range_high') rh, json_extract(pattern_data,'$.range_low') rl, "
        "json_extract(pattern_data,'$.atr14') atr_pd from trades where strategy='orb'", con)
    return f.merge(pdx, on='id', how='left')


def map_actual(reason: str) -> str:
    """Collapse ledger exit reasons onto the simulator's classes."""
    return {'stop_loss': 'stop', 'stop_loss_market_fallback': 'stop', 'lock_stop': 'lock',
            'tag_bb': 'tag_bb', 'force_close': 'force_close'}.get(reason, reason)


def run_cell_a(store: BarStore, out_csv: str):
    """Re-simulate each ledger fill from (i) BT entry and (ii) live fill under all variants."""
    led = ledger_rows()
    led = led[led['date'] <= '2026-09-28']          # PREREG: the 14 later paper fills are excluded (no fallback)
    done = set(pd.read_csv(out_csv)['id']) if os.path.exists(out_csv) else set()
    rows = []
    for k, r in enumerate(led.to_dict('records'), 1):
        if r['id'] in done:
            continue
        bars, src = store.get(r['symbol'], r['date'])
        base = dict(id=r['id'], date=r['date'], symbol=r['symbol'], bars_src=src, actual_reason=r['exit_reason'],
                    actual_R=r['R'], qty=r['filled_qty'], fill=r['entry'], ledger_stop=r['stop'],
                    pnl_actual=r['pnl'], account=r['account'])
        if bars is None:
            rows.append(base)
            continue
        # PREREG: ATR as stored in pattern_data; absent -> range_low (share reported)
        atr_src = 'pattern' if pd.notna(r['atr_pd']) else 'range_low'
        atr = float(r['atr_pd']) if pd.notna(r['atr_pd']) else None
        bt_entry = r['rh'] * 1.003
        base.update(rh=r['rh'], rl=r['rl'], atr14=atr, atr_src=atr_src, bt_entry=bt_entry)
        fill_ts = pd.Timestamp(r['brk_filled'])
        for v in VARIANTS:
            for tag, px in (('i', bt_entry), ('ii', r['entry'])):
                ex = simulate_exit(bars, r['date'], fill_ts, px, r['rh'], r['rl'], atr, v)
                risk = px - ex['stop_px'] if np.isfinite(ex['stop_px']) and px > ex['stop_px'] else np.nan
                base[f'{tag}_{v}_reason'] = ex['reason']
                base[f'{tag}_{v}_exit'] = ex['exit_px']
                base[f'{tag}_{v}_R'] = (ex['exit_px'] - px) / risk
                if v == 'main':
                    base[f'{tag}_exit_ts'] = str(ex['exit_ts'])
                    base[f'{tag}_floor'] = ex['floor_status']
                    base[f'{tag}_stop_init'] = floored_stop(round(r['rl'], 2), px, atr, ATR_K)[0]
        rows.append(base)
        if k % 25 == 0:
            log(f'cell a: {k}/{len(led)} fills')
            _append(out_csv, rows)
            rows = []
    _append(out_csv, rows)


def _append(path: str, rows: list):
    """Append rows to the intermediate CSV (union of columns kept by rewriting)."""
    if not rows:
        return
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    tmp = path + '.tmp'
    new.to_csv(tmp, index=False)
    os.replace(tmp, path)


def report_a(csv: str) -> list:
    """Tables for cell a; returns markdown lines."""
    d = pd.read_csv(csv)
    cov = d['bars_src'].ne('uncovered')
    L = [f'Population: fills_1702 {len(d)} rows; bars found {int(cov.sum())} '
         f'(sources {d.loc[cov, "bars_src"].value_counts().to_dict()}); uncovered {int((~cov).sum())} '
         f'(dates {sorted(d.loc[~cov, "date"].unique())[:3]}..., all after the bars_sip.db end 2026-09-28; no fallback).']
    c = d[cov].copy()
    acct = c['account'].value_counts().to_dict()
    L.append(f'Covered by account: {acct}. ATR source: {c["atr_src"].value_counts().to_dict()}. '
             f'Ledger stop == sim initial stop (ii, 1c): {np.mean(np.isclose(c["ledger_stop"], c["ii_stop_init"], atol=0.011)):.0%}; '
             f'== range_low alone: {np.mean(np.isclose(c["ledger_stop"], c["rl"], atol=0.011)):.0%}.')
    c['act_cls'] = c['actual_reason'].map(map_actual)
    days = c['date'].values
    L.append('| book | n | stop share | mean R [day-boot 95 % CI] | lock | tag | force |')
    L.append('|---|---|---|---|---|---|---|')

    def row(name, reasons, R):
        m, lo, hi = day_boot_ci(R.values, days)
        rs = reasons.replace({'tag_b1': 'tag_bb'})
        sh = lambda k: f'{(rs == k).mean():.1%}'
        return f'| {name} | {len(c)} | {sh("stop")} | {m:+.3f} [{lo:+.3f},{hi:+.3f}] | {sh("lock")} | {sh("tag_bb")} | {sh("force_close")} |'

    L.append(row('actual ledger', c['act_cls'], c['actual_R']))
    for v in VARIANTS:
        L.append(row(f'(i) BT entry, {v}', c[f'i_{v}_reason'], c[f'i_{v}_R']))
        L.append(row(f'(ii) live fill, {v}', c[f'ii_{v}_reason'], c[f'ii_{v}_R']))
    s_act = (c['act_cls'] == 'stop').mean()
    s_i = (c['i_main_reason'] == 'stop').mean()
    s_ii = (c['ii_main_reason'] == 'stop').mean()
    ex_pub = s_act - 0.284
    L.append(f'H1 numbers (main): ledger stop share {s_act:.1%} (published BT 28.4 %) -> excess {ex_pub*100:+.1f} pp; '
             f'(i) {s_i:.1%}, (ii) {s_ii:.1%}; (ii)-(i) = {(s_ii-s_i)*100:+.1f} pp = '
             f'{(s_ii-s_i)/ex_pub:.0%} of the ledger-vs-published excess; (ii)-ledger = {(s_ii-s_act)*100:+.1f} pp.')
    agree = (c['ii_main_reason'].replace({'tag_b1': 'tag_bb'}) == c['act_cls']).mean()
    L.append(f'Exit-reason agreement (ii main vs ledger): {agree:.1%} of {len(c)}. Confusion (rows=ledger, cols=sim):')
    cm = pd.crosstab(c['act_cls'], c['ii_main_reason'].replace({'tag_b1': 'tag_bb'}))
    L.append('`' + cm.to_string().replace('\n', ' / ') + '`')
    # hand check: first fill of each ledger class
    L.append('Hand-check (5 fills; ledger vs sim (ii) exit time UTC / price / reason):')
    seen = set()
    for _, r in c.iterrows():
        if r['actual_reason'] in seen or len(seen) >= 5:
            continue
        seen.add(r['actual_reason'])
        ledger = led_exit(r)
        L.append(f'* id {r["id"]} {r["symbol"]} {r["date"]} fill {r["fill"]:.3f} stop {r["ledger_stop"]:.3f}: '
                 f'ledger {r["actual_reason"]} {ledger}; sim {r["ii_main_reason"]} @ {r["ii_main_exit"]:.3f} '
                 f'{str(r["ii_exit_ts"])[11:19]} (sim stop {r["ii_stop_init"]:.3f})')
    return L


def led_exit(r) -> str:
    """Ledger exit time/price string for a cell-a row (looked up from fills_1702)."""
    f = pd.read_csv(f'{EXEC}/fills_1702.csv')
    x = f[f['id'] == r['id']].iloc[0]
    return f'{str(x["exited_at"])[11:19]} @ {x["exit_price"]:.3f}'


# ------------------------------------------------------------------ cell b
def run_cell_b(store: BarStore, out_csv: str):
    """Re-simulate every replay-filled fill (delay 0 and 30 s) under all variants."""
    pop = pd.read_csv(f'{ROOT}/research/orb_latency_bt/population.csv')
    rep = pd.read_csv(f'{EXEC}/fills_1708.csv')
    rep = rep[(rep['delay_s'].isin([0, 30])) & (rep['status'] == 'filled')]
    done = set(map(tuple, pd.read_csv(out_csv)[['symbol', 'date', 'delay_s']].values)) if os.path.exists(out_csv) else set()
    popi = pop.set_index(['symbol', 'date'])
    rows, cache = [], {}
    for k, r in enumerate(rep.sort_values(['date', 'symbol', 'delay_s']).to_dict('records'), 1):
        key = (r['symbol'], r['date'], r['delay_s'])
        if key in done:
            continue
        p = popi.loc[(r['symbol'], r['date'])]
        sd = (r['symbol'], r['date'])
        if sd not in cache:
            cache.clear()
            cache[sd] = store.get(*sd)
        bars, src = cache[sd]
        shares = p['_rp_position'] / p['entry_price']
        base = dict(symbol=r['symbol'], date=r['date'], delay_s=r['delay_s'], period=r['period'], bars_src=src,
                    win_bt=p['win'], fill=r['fill_price'], t_star=r['t_star'], shares=shares,
                    pnl_1708=r['pnl_replay'])
        if bars is None:
            rows.append(base)
            continue
        et = pd.DatetimeIndex(bars['ts']).tz_convert(ET)
        mins = (et.hour * 60 + et.minute).to_numpy()
        rng = bars[(mins >= 570) & (mins < 575)]
        rl = float(rng['l'].min()) if len(rng) >= 3 else np.nan
        rh_bars = float(rng['h'].max()) if len(rng) >= 3 else np.nan
        rh = float(p['trigger'])
        if not np.isfinite(rl) or rl >= rh:
            base['note'] = 'bad_range'
            rows.append(base)
            continue
        atr = None   # no stored ATR for the replay population -> range_low stop (PREREG)
        t0 = pd.Timestamp(f'{r["date"]} 09:35:00', tz=ET)
        fill_ts = (t0 + pd.Timedelta(seconds=max(float(r['t_star']), float(r['delay_s'])))).tz_convert('UTC')
        base.update(rh=rh, rh_bars=rh_bars, rl=rl, atr14=atr, atr_src='range_low')
        for v in VARIANTS:
            ex = simulate_exit(bars, r['date'], fill_ts, float(r['fill_price']), rh, rl, atr, v)
            pnl = shares * (ex['exit_px'] - float(r['fill_price']))
            base[f'{v}_reason'] = ex['reason']
            base[f'{v}_pnl'] = pnl
            base[f'{v}_R375'] = pnl / R_DENOM
            risk = shares * (float(r['fill_price']) - ex['stop_px'])
            base[f'{v}_Rown'] = pnl / risk if risk > 0 else np.nan
        rows.append(base)
        if k % 100 == 0:
            log(f'cell b: {k}/{len(rep)}')
            _append(out_csv, rows)
            rows = []
    _append(out_csv, rows)


def span_weeks(dates) -> float:
    """Calendar weeks spanned by a list of ISO dates (at least 1)."""
    d = pd.to_datetime(pd.Series(dates))
    return max(1.0, (d.max() - d.min()).days / 7.0)


def report_b(csv: str) -> list:
    """Tables for cell b; returns markdown lines."""
    d = pd.read_csv(csv)
    d['reg'] = np.where(d['date'] < '2024-07-01', 'oor_2023-24H1', np.where(d['date'] < '2025-01-01', 'oor_2024H2',
                        np.where(d['date'] < '2025-07-01', 'in_2025H1', 'in_2025H2+2026')))
    d['cov'] = d['bars_src'].ne('uncovered') & d['main_R375'].notna()
    L = []
    pop = pd.read_csv(f'{ROOT}/research/orb_latency_bt/population.csv')
    L.append(f'Replay-filled rows: {len(d)} ({len(d)//2} per delay approx). Coverage by regime x delay (bars found, range ok):')
    cv = d.groupby(['reg', 'delay_s'])['cov'].agg(['sum', 'count'])
    cv['pct'] = (cv['sum'] / cv['count']).round(2)
    L.append('`' + cv.to_string().replace('\n', ' / ') + '`')
    L.append('Availability rail (delay 0): coverage and winner(BT win=1)/loser miss-rate gap, per regime:')
    for g, x in d[d.delay_s == 0].groupby('reg'):
        mw = 1 - x.loc[x.win_bt == 1, 'cov'].mean() if (x.win_bt == 1).any() else np.nan
        ml = 1 - x.loc[x.win_bt == 0, 'cov'].mean() if (x.win_bt == 0).any() else np.nan
        ok = (x['cov'].mean() >= 0.8) and (abs(mw - ml) <= 0.05)
        L.append(f'* {g}: cov {x["cov"].mean():.0%} (n {len(x)}), miss win {mw:.0%} / lose {ml:.0%} -> {"OK" if ok else "VOID"}')
    L.append(f'Bad-range/no-range rows: {int((d.get("note", pd.Series(dtype=str)) == "bad_range").sum())}; '
             f'range_high from bars vs trigger |diff|>0.5 %: '
             f'{int((np.abs(d["rh_bars"] / d["rh"] - 1) > 0.005).sum())}; ATR floor applied: '
             f'0% (stop = range_low for all replay rows, no stored ATR).')
    L.append('| regime | delay | variant | n cov | mean R375 [day-boot CI] | ex-top5% | +3R cap | stop share | /wk (cov) | /wk (all replay-filled) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    groups = [('oor_2023-24H1', lambda x: x.reg == 'oor_2023-24H1'), ('oor_2024H2', lambda x: x.reg == 'oor_2024H2'),
              ('oor_all', lambda x: x.reg.str.startswith('oor')), ('in_2025H1', lambda x: x.reg == 'in_2025H1'),
              ('in_2025H2+2026', lambda x: x.reg == 'in_2025H2+2026'), ('in_all', lambda x: x.reg.str.startswith('in'))]
    for gname, f in groups:
        for dl in (0, 30):
            for v in (['main', 'ebstop', 'fc1555', 'cost10'] if dl == 0 else ['main']):
                x = d[f(d) & (d.delay_s == dl)]
                xc = x[x['cov']]
                if len(xc) == 0:
                    L.append(f'| {gname} | {dl} | {v} | 0 | n/a (uncovered) | | | | | {len(x)/span_weeks(x.date) if len(x) else 0:.2f} |')
                    continue
                R = xc[f'{v}_R375'].values
                m, lo, hi = day_boot_ci(R, xc['date'].values)
                st = (xc[f'{v}_reason'] == 'stop').mean()
                L.append(f'| {gname} | {dl} | {v} | {len(xc)} | {m:+.3f} [{lo:+.3f},{hi:+.3f}] | {ex_top(R):+.3f} | '
                         f'{np.minimum(R, 3).mean():+.3f} | {st:.1%} | {len(xc)/span_weeks(xc.date):.2f} | {len(x)/span_weeks(x.date):.2f} |')
    # 1708 reference on the same covered rows
    xc = d[d['cov'] & (d.delay_s == 0)]
    L.append(f'Reference: 1,708 replay pnl (BT exits) on the same covered delay-0 rows: mean R375 '
             f'{(xc.pnl_1708 / R_DENOM).mean():+.3f} (n {len(xc)}) vs re-sim main {xc.main_R375.mean():+.3f}.')
    # H2 evaluation
    L.append('H2 pass bar (out-of-regime = 2023-01..2024-12, delay 0, main):')
    x = d[(d.reg.str.startswith('oor')) & (d.delay_s == 0)]
    xc = x[x['cov']]
    if len(xc) == 0:
        L.append('* no out-of-regime bars -> rail VOID -> H2 cannot pass.')
    return L


def main():
    """CLI entry."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', choices=['a', 'b'], required=True)
    ap.add_argument('--report', action='store_true', help='only print tables from the existing CSV')
    a = ap.parse_args()
    out = f'{HERE}/resim_1709{a.cell}.csv'
    if not a.report:
        free = os.statvfs('/').f_bavail * os.statvfs('/').f_frsize / 1e9
        if free < 5:
            raise SystemExit(f'only {free:.1f} GB free')
        t = time.time()
        store = BarStore()
        (run_cell_a if a.cell == 'a' else run_cell_b)(store, out)
        log(f'cell {a.cell} done in {time.time()-t:.0f}s -> {out}')
    lines = report_a(out) if a.cell == 'a' else report_b(out)
    path = f'{HERE}/tables_1709{a.cell}.md'
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
