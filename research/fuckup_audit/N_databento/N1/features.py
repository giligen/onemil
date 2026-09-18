#!/usr/bin/env python3
"""Stage N1 step 2 — the declared order-flow features, per ORB candidate.

Reads the N1/tbbo/*.parquet pulls (EQUS.MINI `tbbo`) plus the Alpaca 1-min bars
in data/cache.db (READ-ONLY) and emits N1/features.parquet keyed (symbol, date):

  ofi_range            Cont-Kukanov-Stoikov order-flow imbalance over
                       09:30:00-09:34:59.999 ET, summed over successive BBO
                       observations, divided by mean displayed depth
                       (bid_sz+ask_sz)/2 over the window.
  tai_range            trade aggressor imbalance over the same window:
                       (buy$ - sell$)/total$, tick rule vs the prevailing BBO.
  ofi_break60          the same OFI over [break_ts, break_ts+60s].
  tai_break60          the same TAI over [break_ts, break_ts+60s].
  spread_at_break_bps  NBBO spread at the breakout trade, in bps of the mid.
  n_trades_range       trade count in the range window (coverage).
  n_trades_break60     trade count in the 60s after the break.

break_ts = the first trade at or after 09:35:00 ET whose price is strictly
above range_high; range_high = max high of the five 09:30..09:34 ET Alpaca
bars (the pipeline's own definition, study_orb_features.py:248).

CAVEAT recorded in the report: tbbo carries the BBO *at each trade*, not every
BBO update, so OFI here is the CKS sum over quote observations sampled at trade
times. It is a sampled OFI, not the full-message OFI.
"""
from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')

D1 = 'research/fuckup_audit/D1_orb'
N1 = 'research/fuckup_audit/N_databento/N1'
TBBO = f'{N1}/tbbo'
RANGE_START, RANGE_END = '09:30:00', '09:35:00'

RNG = np.random.RandomState(20260918)

# --- availability / QA counters (reported in REPORT.md) ---
QA = {'assert_range_ts': 0, 'assert_break60_ts': 0,
      'price_scale_checked': 0, 'price_scale_outside': 0}


def load_candidates() -> pd.DataFrame:
    d = pd.read_csv(f'{D1}/candidates_dump.csv', usecols=['symbol', 'date'],
                    keep_default_na=False, na_values=[''])
    d['date'] = pd.to_datetime(d['date']).dt.strftime('%Y-%m-%d')
    return d.drop_duplicates()


def bars_for_date(con: sqlite3.Connection, date: str, symbols: list[str]) -> pd.DataFrame:
    """09:30-09:45 ET 1-min bars for the day's candidates (read-only cache)."""
    out = []
    for i in range(0, len(symbols), 400):
        chunk = symbols[i:i + 400]
        q = ("SELECT symbol, timestamp, high, low FROM intraday_bars_1min "
             f"WHERE bar_date = ? AND symbol IN ({','.join('?' * len(chunk))})")
        out.append(pd.read_sql_query(q, con, params=[date] + chunk))
    b = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=['symbol', 'timestamp', 'high', 'low'])
    if not len(b):
        return b
    b['ts'] = pd.to_datetime(b['timestamp'], utc=True, format='ISO8601'
                             ).dt.tz_convert('America/New_York')
    b['hhmm'] = b['ts'].dt.strftime('%H:%M')
    return b[(b['hhmm'] >= '09:30') & (b['hhmm'] <= '09:45')]


def ofi(bid_px: np.ndarray, bid_sz: np.ndarray,
        ask_px: np.ndarray, ask_sz: np.ndarray) -> float:
    """CKS order-flow imbalance summed over successive BBO observations."""
    if len(bid_px) < 2:
        return np.nan
    db, da = np.diff(bid_px), np.diff(ask_px)
    e = (np.where(db >= 0, bid_sz[1:], 0.0) - np.where(db <= 0, bid_sz[:-1], 0.0)
         - np.where(da <= 0, ask_sz[1:], 0.0) + np.where(da >= 0, ask_sz[:-1], 0.0))
    return float(np.nansum(e))


def window_feats(w: pd.DataFrame) -> tuple[float, float, int]:
    """(depth-normalised OFI, trade aggressor imbalance, n_trades) for a window."""
    n = len(w)
    if n == 0:
        return np.nan, np.nan, 0
    bp = w['bid_px_00'].to_numpy(float); ap = w['ask_px_00'].to_numpy(float)
    bs = w['bid_sz_00'].to_numpy(float); asz = w['ask_sz_00'].to_numpy(float)
    px = w['price'].to_numpy(float); sz = w['size'].to_numpy(float)
    depth = np.nanmean((bs + asz) / 2.0)
    raw = ofi(bp, bs, ap, asz)
    o = raw / depth if (depth and depth > 0 and np.isfinite(raw)) else np.nan
    mid = (bp + ap) / 2.0
    sign = np.where(px >= ap, 1.0, np.where(px <= bp, -1.0,
                    np.where(px > mid, 1.0, np.where(px < mid, -1.0, 0.0))))
    dollars = px * sz
    tot = float(np.nansum(dollars))
    t = float(np.nansum(sign * dollars) / tot) if tot > 0 else np.nan
    return o, t, n


def main() -> int:
    cands = load_candidates()
    by_date = {k: sorted(set(v)) for k, v in cands.groupby('date')['symbol']}
    dates = sorted(by_date)
    con = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)
    rows = []
    scale_rows = []
    for i, date in enumerate(dates, 1):
        path = f'{TBBO}/{date}.parquet'
        if not os.path.exists(path):
            print(f"  MISSING tbbo for {date} — skipped", flush=True)
            continue
        syms = by_date[date]
        bars = bars_for_date(con, date, syms)
        rh, rl = {}, {}
        if len(bars):
            rb = bars[(bars['hhmm'] >= '09:30') & (bars['hhmm'] < '09:35')]
            g = rb.groupby('symbol').agg(hi=('high', 'max'), lo=('low', 'min'),
                                         n=('high', 'size'))
            rh = {s: float(r['hi']) for s, r in g.iterrows() if r['n'] >= 5}
            rl = {s: float(r['lo']) for s, r in g.iterrows() if r['n'] >= 5}
        bar_hl = {}
        if len(bars):
            for s, ts, h, l in zip(bars['symbol'], bars['hhmm'], bars['high'], bars['low']):
                bar_hl[(s, ts)] = (float(h), float(l))

        t = pd.read_parquet(path)
        if len(t):
            t['ts'] = pd.to_datetime(t['ts_event'], utc=True).dt.tz_convert('America/New_York')
            t['hhmmss'] = t['ts'].dt.strftime('%H:%M:%S')
            t = t.sort_values(['symbol', 'ts'])
        groups = dict(tuple(t.groupby('symbol'))) if len(t) else {}

        for s in syms:
            w = groups.get(s)
            rec = {'symbol': s, 'date': date,
                   'ofi_range': np.nan, 'tai_range': np.nan, 'n_trades_range': 0,
                   'ofi_break60': np.nan, 'tai_break60': np.nan,
                   'n_trades_break60': 0, 'spread_at_break_bps': np.nan,
                   'break_hhmmss': '',
                   'range_high': rh.get(s, np.nan), 'range_low': rl.get(s, np.nan)}
            if w is None or not len(w):
                rows.append(rec)
                continue
            rw = w[(w['hhmmss'] >= RANGE_START) & (w['hhmmss'] < RANGE_END)]
            if len(rw):
                # AVAILABILITY ASSERTION: nothing in the range features may be
                # timestamped at or after 09:35:00 ET.
                assert rw['hhmmss'].max() < RANGE_END, (s, date, rw['hhmmss'].max())
                QA['assert_range_ts'] += 1
            rec['ofi_range'], rec['tai_range'], rec['n_trades_range'] = window_feats(rw)

            high = rh.get(s)
            if high is not None and high > 0:
                post = w[w['hhmmss'] >= RANGE_END]
                brk = post[post['price'].astype(float) > high]
                if len(brk):
                    b0 = brk.iloc[0]
                    bts = b0['ts']
                    rec['break_hhmmss'] = b0['hhmmss']
                    bw = w[(w['ts'] >= bts) & (w['ts'] <= bts + pd.Timedelta(seconds=60))]
                    # AVAILABILITY ASSERTION: break60 uses nothing after +60s.
                    assert bw['ts'].max() <= bts + pd.Timedelta(seconds=60)
                    QA['assert_break60_ts'] += 1
                    (rec['ofi_break60'], rec['tai_break60'],
                     rec['n_trades_break60']) = window_feats(bw)
                    bid, ask = float(b0['bid_px_00']), float(b0['ask_px_00'])
                    mid = (bid + ask) / 2.0
                    if mid > 0 and ask >= bid:
                        rec['spread_at_break_bps'] = (ask - bid) / mid * 1e4
                    hl = bar_hl.get((s, b0['hhmmss'][:5]))
                    if hl:
                        scale_rows.append((s, date, float(b0['price']), hl[0], hl[1]))
            rows.append(rec)
        if i % 20 == 0 or i == len(dates):
            print(f"  features {i}/{len(dates)} dates, {len(rows)} candidates", flush=True)
    con.close()

    f = pd.DataFrame(rows)
    f.to_parquet(f'{N1}/features.parquet', index=False)
    # Sidecar CSV for ORB_BT_SIDECAR_CSV (joined on symbol+date by the pipeline).
    f[['symbol', 'date', 'ofi_range', 'tai_range', 'ofi_break60', 'tai_break60',
       'spread_at_break_bps', 'n_trades_range', 'n_trades_break60']].to_csv(
        f'{N1}/sidecar.csv', index=False)

    # ---- price-scale check on 200 random candidates ----
    sc = pd.DataFrame(scale_rows, columns=['symbol', 'date', 'px', 'bar_high', 'bar_low'])
    if len(sc):
        take = sc.sample(min(200, len(sc)), random_state=20260918)
        outside = ((take['px'] > take['bar_high'] * 1.0001)
                   | (take['px'] < take['bar_low'] * 0.9999))
        QA['price_scale_checked'] = int(len(take))
        QA['price_scale_outside'] = int(outside.sum())

    cov = {
        'candidates': int(len(f)),
        'no_tbbo_rows': int((f['n_trades_range'] == 0).sum()),
        'thin_lt20_range_trades': int((f['n_trades_range'] < 20).sum()),
        'has_break': int((f['break_hhmmss'] != '').sum()),
        'has_ofi_range': int(f['ofi_range'].notna().sum()),
        'has_ofi_break60': int(f['ofi_break60'].notna().sum()),
    }
    qa = {**QA, **cov}
    pd.Series(qa).to_csv(f'{N1}/features_qa.csv')
    print("QA / coverage:", qa, flush=True)
    print(f"Wrote {N1}/features.parquet ({len(f)} rows)", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
