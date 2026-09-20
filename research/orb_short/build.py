#!/usr/bin/env python3
"""ORB short mirror — Stage A signal/control builder. PREREG.md §1-§2, §5.

Reads data/cache.db READ-ONLY. Writes research/orb_short/sig.csv + ctl.csv.
TEST (>= 2026-06-01) is never queried.
"""
import os
import re
import sqlite3
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

D = f'{ROOT}/research/orb_short'
DB = f'file:{ROOT}/data/cache.db?mode=ro'

GAP_MAX = -5.0          # mirrored from the long study's +5.0
MIN_PREV_VOL = 500_000
MIN_OPEN, MAX_OPEN = 5.0, 30.0
MIN_935_VOL = 15_000
BUF_BPS = 30.0
LOCK_ARM_R, LOCK_STOP_R = 1.75, 0.5
START, END = '2025-01-01', '2026-05-31'   # TEST SEALED
TESTRE = re.compile(r'^Z[A-Z]ZZT$')


def _safe_div(a, b):
    return a / b if b else 0.0


def universe(conn):
    q = """
    WITH d AS (SELECT symbol,bar_date,open,
      LAG(close) OVER (PARTITION BY symbol ORDER BY bar_date) pc,
      LAG(volume) OVER (PARTITION BY symbol ORDER BY bar_date) pv
      FROM daily_bars)
    SELECT symbol,bar_date FROM d
    WHERE bar_date BETWEEN ? AND ?
      AND pc>0 AND (open-pc)/pc*100 <= ? AND pv>=?
      AND open BETWEEN ? AND ?
      AND EXISTS(SELECT 1 FROM intraday_bars_1min i
                 WHERE i.symbol=d.symbol AND i.bar_date=d.bar_date)
    ORDER BY bar_date,symbol"""
    rows = conn.execute(q, (START, END, GAP_MAX, MIN_PREV_VOL,
                            MIN_OPEN, MAX_OPEN)).fetchall()
    return [(s, str(b)) for s, b in rows if not TESTRE.match(s)]


def load_daily(conn, syms):
    out = {}
    syms = sorted(syms)
    for i in range(0, len(syms), 500):
        ch = syms[i:i + 500]
        ph = ','.join('?' * len(ch))
        df = pd.read_sql_query(
            f"SELECT symbol,bar_date,open,high,low,close,volume FROM daily_bars "
            f"WHERE symbol IN ({ph}) AND bar_date <= ? ORDER BY symbol,bar_date",
            conn, params=ch + [END])
        for s, g in df.groupby('symbol', sort=False):
            out[s] = g.reset_index(drop=True)
    return out


def feats(rb, symbol, ds, sd):
    f = {}
    rh = float(rb['high'].max()); rl = float(rb['low'].min())
    op = float(rb['open'].iloc[0]); cp = float(rb['close'].iloc[-1])
    rs = rh - rl
    f['range_high'] = rh; f['range_low'] = rl; f['open_p'] = op
    f['range_size_pct'] = _safe_div(rs, op) * 100
    f['range_total_volume'] = float(rb['volume'].sum())
    br = (rb['high'] - rb['low']) / rb['close'].replace(0, np.nan)
    f['range_avg_bar_range_pct'] = float(br.mean(skipna=True) * 100) if not br.isna().all() else 0.0
    f['range_close_position'] = _safe_div(cp - rl, rs) if rs > 0 else 0.5
    prev = sd[sd['bar_date'] < ds]
    if prev.empty:
        return None
    pr = prev.iloc[-1]
    pc, ph_, pl_ = float(pr['close']), float(pr['high']), float(pr['low'])
    f['prev_close'] = pc
    f['gap_pct'] = _safe_div(op - pc, pc) * 100
    f['prev_day_range_pct'] = _safe_div(ph_ - pl_, pc) * 100
    prng = ph_ - pl_
    f['prev_day_close_position'] = _safe_div(pc - pl_, prng) if prng > 0 else 0.5
    pri = prev.tail(20)
    if len(pri) < 5:
        f['price_vs_20d_high_pct'] = 0.0; f['return_volatility_20d'] = 0.0
        f['adv20'] = 0.0
    else:
        h20 = float(pri['high'].max())
        f['price_vs_20d_high_pct'] = _safe_div(op - h20, h20) * 100
        cl = pri['close'].to_numpy(float)
        f['return_volatility_20d'] = float((np.diff(cl) / cl[:-1]).std() * 100) if len(cl) > 1 else 0.0
        f['adv20'] = float(pri['volume'].mean())
    return f


def walk(post, entry, rh, i0):
    """Short walk from bar index i0 (entry bar, excluded). Returns (exit_px, reason, exit_i)."""
    R = rh - entry
    stop = rh
    arm = entry - LOCK_ARM_R * R
    lock = entry - LOCK_STOP_R * R
    armed = False
    n = len(post)
    for i in range(i0 + 1, n):
        lo = float(post['low'].iat[i]); hi = float(post['high'].iat[i])
        if not armed and lo <= arm:
            armed = True
            stop = min(stop, lock)
        if hi >= stop:
            return max(stop, float(post['open'].iat[i])), ('lock' if armed else 'stop'), i
    return float(post['close'].iat[n - 1]), 'eod', n - 1


def main():
    conn = sqlite3.connect(DB, uri=True)
    cand = universe(conn)
    print(f'[uni] {len(cand):,} candidate symbol-days', flush=True)
    bydate = {}
    for s, d in cand:
        bydate.setdefault(d, []).append(s)
    daily = load_daily(conn, {s for s, _ in cand})
    print(f'[daily] {len(daily):,} symbols', flush=True)

    sig, ctl = [], []
    for k, (ds, syms) in enumerate(sorted(bydate.items())):
        ph = ','.join('?' * len(syms))
        bars = pd.read_sql_query(
            f"SELECT symbol,timestamp,open,high,low,close,volume FROM intraday_bars_1min "
            f"WHERE bar_date=? AND symbol IN ({ph}) ORDER BY symbol,timestamp",
            conn, params=[ds] + syms)
        if bars.empty:
            continue
        ts = pd.to_datetime(bars['timestamp'], utc=True, format='mixed').dt.tz_convert('America/New_York')
        bars['m'] = ts.dt.hour * 60 + ts.dt.minute
        bars = bars[(bars.m >= 570) & (bars.m <= 945)]
        for sym, g in bars.groupby('symbol', sort=False):
            g = g.reset_index(drop=True)
            rb = g[(g.m >= 570) & (g.m < 575)]
            if len(rb) < 5:
                continue
            if float(g.loc[g.m < 575, 'volume'].sum()) < MIN_935_VOL:
                continue
            sd = daily.get(sym)
            if sd is None or sd.empty:
                continue
            f = feats(rb, sym, ds, sd)
            if f is None:
                continue
            post = g[g.m >= 575].reset_index(drop=True)
            if len(post) < 3:
                continue
            L = f['range_low'] * (1 - BUF_BPS / 10000)
            trig = None
            for i in range(len(post)):
                if post['m'].iat[i] > 635:
                    break
                if float(post['low'].iat[i]) <= L:
                    trig = i
                    break
            base = dict(day=ds, symbol=sym, **{k2: v for k2, v in f.items()})
            if trig is None:
                # control: no break in the 60-min window; short the 09:36 open
                j = post.index[post.m == 576]
                if len(j) == 0:
                    continue
                j = int(j[0])
                e = float(post['open'].iat[j])
                if f['range_high'] - e <= 0:
                    continue
                px, rsn, xi = walk(post, e, f['range_high'], j)
                ctl.append(dict(base, entry=e, exit=px, reason=rsn,
                                entry_m=int(post['m'].iat[j]), exit_m=int(post['m'].iat[xi]),
                                R=f['range_high'] - e))
                continue
            # signal
            ssr = (float(post['close'].iat[trig]) / f['prev_close'] - 1.0) <= -0.10
            row = dict(base, trig_m=int(post['m'].iat[trig]), ssr=int(ssr), limit=L)
            if trig + 1 >= len(post):
                sig.append(dict(row, filled=0, nofill='no_next_bar'))
                continue
            nb = post.iloc[trig + 1]
            no = float(nb['open'])
            if no < L:
                sig.append(dict(row, filled=0, nofill='gap_through'))
                continue
            if ssr and not (no > float(nb['low'])):
                sig.append(dict(row, filled=0, nofill='ssr_no_uptick'))
                continue
            e = no
            if f['range_high'] - e <= 0:
                sig.append(dict(row, filled=0, nofill='no_risk'))
                continue
            px, rsn, xi = walk(post, e, f['range_high'], trig + 1)
            sig.append(dict(row, filled=1, nofill='', entry=e, exit=px, reason=rsn,
                            entry_m=int(nb['m']), exit_m=int(post['m'].iat[xi]),
                            R=f['range_high'] - e))
        if k % 50 == 0:
            print(f'[{k}] {ds} sig={len(sig)} ctl={len(ctl)}', flush=True)
    pd.DataFrame(sig).to_csv(f'{D}/sig.csv', index=False)
    pd.DataFrame(ctl).to_csv(f'{D}/ctl.csv', index=False)
    print(f'[done] sig={len(sig)} ctl={len(ctl)}', flush=True)


if __name__ == '__main__':
    main()
