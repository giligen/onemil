#!/usr/bin/env python3
"""Zarattini noise-area sleeve — independent re-implementation with LIVE fill conventions.

Written from the prose spec (see Q/REPORT.md §2) for the obtainability / fill-realism audit of
research/lit_review_2026/test_zarattini_spy.py rows 1/1b of RESULTS.md.

Conventions implemented here (all switchable):
  band       : sigma_t(k) = mean over the 14 PRIOR days of |C[d,k]/O_day[d] - 1|   (shift(1), min 10 obs)
               UB = max(O_day, prev_close) * (1 + VM*sigma) ; LB = min(O_day, prev_close) * (1 - VM*sigma)
  checks     : k in {30,60,...,360}  ->  10:00, 10:30, ... 15:30 ET  (k = ET minute - 570)
  signal     : the CLOSE of the check bar k.  flat: long if c>UB, short if c<LB.
               long: stop = max(UB[k], VWAP[k]); short: stop = min(LB[k], VWAP[k]); on stop, the
               opposite side may be opened at the same check (never the side just stopped).
  fill       : 'close'      = the check bar's own close      (the paper / the repo script: LOOK-AHEAD of <=60s)
               'next_open'  = the open of bar k+1            (the live convention: decide on a closed bar,
                                                              market order fills at the next bar's open)
  slippage   : per leg, applied to the fill price: buy * (1+s), sell * (1-s).
  eod        : 'moc'   = last bar's close (paper; proxy for the closing auction print)
               'open'  = the 15:59 bar's OPEN (decision at the 15:58 close, market order)
  sizing     : 1x   = AUM / O_day shares ; dyn = AUM * min(4, 0.02/sig14) / O_day
               sig14 = std of the 14 daily close-to-close returns ending at d-1.

Read-only on the DB. No file is written by this module.
"""
import math
import sqlite3
import time

import numpy as np
import pandas as pd

DB = '/home/ec2-user/onemil/research/lit_review_2026/etf_1min.db'
N_MIN = 390
CHECKS_SEMI = list(range(30, 390, 30))
COST_PAPER_PER_SHARE = 0.0035 + 0.001
IS_END = pd.Timestamp('2023-12-31')
OOS_START = pd.Timestamp('2024-01-01')

# ---------------------------------------------------------------------------------------------
# etf_1min.db holds RAW (unadjusted) bars.  A share split therefore shows up as a ~0.5x / 0.33x
# jump between one day's RTH close and the next day's RTH open, which corrupts BOTH band anchors
# (UB = max(open, prev_close)) and the 14-day close-to-close vol used by the dynamic sizer.
# The factors below were derived, not guessed: factor = (TQQQ open/prev_close) / (1 + 3 * QQQ gap)
# on the same session -- see Q/step5_extras.md a).  All five land on 0.4996..0.5000 / 0.3333.
# 2020-03-16 (TQQQ 0.709x) is NOT a split: the implied factor is 1.0225, i.e. the COVID crash gap.
SPLITS = {
    'TQQQ': {'2017-01-12': 0.5, '2018-05-24': 1.0 / 3.0, '2021-01-21': 0.5,
             '2022-01-13': 0.5, '2025-11-20': 0.5},
}
JUMP_LO, JUMP_HI = 0.75, 1.33


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_symbol(symbol, db=DB, split_adjust=True):
    """Regular-hours 1-min bars -> (days x 390) matrices. Memory-light: chunked read.

    split_adjust: back-adjust the pre-split history by the known factor so that prev_close and the
    14-day daily-vol series are on one price scale. Every intraday ratio (sigma, VWAP distance,
    the band width, the P&L in return terms) is scale-free, so the adjustment changes ONLY the
    split days themselves and the 14 days of daily vol after each.
    """
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    parts = []
    for ch in pd.read_sql('select t,o,h,l,c,v,vw from bars where symbol = ? order by t',
                          con, params=(symbol,), chunksize=250_000):
        ts = pd.to_datetime(ch['t'], utc=True, format='ISO8601').dt.tz_convert('America/New_York')
        idx = (ts.dt.hour * 60 + ts.dt.minute - 570).astype('int32')
        m = (idx >= 0) & (idx < N_MIN)
        if not m.any():
            continue
        parts.append(pd.DataFrame({
            'date': ts[m].dt.tz_localize(None).dt.normalize().values,
            'idx': idx[m].values,
            'o': ch['o'][m].astype('float64').values, 'h': ch['h'][m].astype('float64').values,
            'l': ch['l'][m].astype('float64').values, 'c': ch['c'][m].astype('float64').values,
            'v': ch['v'][m].astype('float64').values, 'vw': ch['vw'][m].astype('float64').values}))
    con.close()
    df = pd.concat(parts, ignore_index=True)
    del parts
    days = np.sort(df['date'].unique())
    di = {d: i for i, d in enumerate(days)}
    r = df['date'].map(di).values.astype('int32')
    k = df['idx'].values
    D = len(days)
    C = np.full((D, N_MIN), np.nan); O = C.copy(); H = C.copy(); L = C.copy()
    V = np.zeros((D, N_MIN)); PV = np.zeros((D, N_MIN))
    C[r, k] = df['c'].values; O[r, k] = df['o'].values
    H[r, k] = df['h'].values; L[r, k] = df['l'].values
    V[r, k] = df['v'].values
    vw = np.where(df['vw'].values > 0, df['vw'].values, (df['h'] + df['l'] + df['c']).values / 3.0)
    PV[r, k] = vw * df['v'].values
    n_all = len(df)
    del df
    has = ~np.isnan(C)
    first = np.argmax(has, axis=1)
    last = N_MIN - 1 - np.argmax(has[:, ::-1], axis=1)
    nbars = has.sum(axis=1)
    keep = nbars >= 150
    n_dropped = int((~keep).sum())
    days, C, O, H, L, V, PV, first, last = (days[keep], C[keep], O[keep], H[keep], L[keep],
                                            V[keep], PV[keep], first[keep], last[keep])
    D = len(days); ar = np.arange(D)
    n_split = 0
    if split_adjust and symbol in SPLITS:
        daystr = np.array([str(d)[:10] for d in days])
        adj = np.ones(D)
        for ds, f in SPLITS[symbol].items():
            w = np.where(daystr == ds)[0]
            if len(w):
                adj[:w[0]] *= f
                n_split += 1
        for M in (C, O, H, L):
            M *= adj[:, None]
        PV *= adj[:, None]
    dopen = O[ar, first]; dclose = C[ar, last]
    prevclose = np.r_[np.nan, dclose[:-1]]
    jump = dopen / prevclose
    bad = np.where((jump < JUMP_LO) | (jump > JUMP_HI))[0]
    for i in bad:
        log(f'  WARNING {symbol} unexplained prev_close->open jump {jump[i]:.3f} on {str(days[i])[:10]} '
            f'({prevclose[i]:.2f} -> {dopen[i]:.2f}) -- check for an unlisted split')
    cv = np.cumsum(V, axis=1); cpv = np.cumsum(PV, axis=1)
    VW = np.where(cv > 0, cpv / np.maximum(cv, 1e-12), np.nan)
    dret = pd.Series(dclose).pct_change()
    sig14 = dret.rolling(14).std(ddof=1).shift(1).values
    move = np.abs(C / dopen[:, None] - 1.0)
    sigma = pd.DataFrame(move).rolling(14, min_periods=10).mean().shift(1).values
    log(f'{symbol}: {n_all:,} RTH bars, {D:,} days ({str(days[0])[:10]} -> {str(days[-1])[:10]}), '
        f'{n_dropped} short days dropped, {n_split} splits back-adjusted')
    return dict(symbol=symbol, days=pd.to_datetime(days), C=C, O=O, H=H, L=L, VW=VW,
                dopen=dopen, dclose=dclose, prevclose=prevclose, sig14=sig14, sigma=sigma,
                first=first, last=last, n_dropped=n_dropped, n_split=n_split)


def bands(data, vm=1.0, anchor='open_prevclose'):
    if anchor == 'open_prevclose':
        up = np.fmax(data['dopen'], data['prevclose']); dn = np.fmin(data['dopen'], data['prevclose'])
    else:
        up = dn = data['dopen']
    return up[:, None] * (1.0 + vm * data['sigma']), dn[:, None] * (1.0 - vm * data['sigma'])


def simulate(data, UB, LB, checks=CHECKS_SEMI, fill='close', eod='moc', slip_bp=0.0):
    """One pass. Returns (trades, pos_minute) where
       trades = list of dicts, pos_minute = (D x N_MIN) int8 signed position per minute (for attribution)."""
    C, O, VW, last = data['C'], data['O'], data['VW'], data['last']
    D = C.shape[0]
    s = slip_bp * 1e-4
    trades = []
    posm = np.zeros((D, N_MIN), dtype=np.int8)

    def leg(px, side):
        """side=+1 buying, -1 selling -> executed price after slippage."""
        return px * (1.0 + s) if side > 0 else px * (1.0 - s)

    for d in range(D):
        ld = last[d]
        if np.isnan(UB[d, 30]) or ld < 60:
            continue
        pos = 0; entry = np.nan; ek = -1
        for k in checks:
            if k >= ld:
                break
            px = C[d, k]
            ub, lb, vw = UB[d, k], LB[d, k], VW[d, k]
            if np.isnan(px) or np.isnan(ub) or np.isnan(vw):
                continue
            if fill == 'next_open':
                fp = O[d, k + 1]
                fk = k + 1
                if np.isnan(fp):
                    fp = px; fk = k
            else:
                fp = px; fk = k
            stopped_side = 0
            if pos == 1 and px < max(ub, vw):
                trades.append(dict(d=d, side=1, e=entry, x=leg(fp, -1), ek=ek, xk=fk, why='stop'))
                posm[d, ek + 1:fk] = 1
                pos = 0; stopped_side = 1
            elif pos == -1 and px > min(lb, vw):
                trades.append(dict(d=d, side=-1, e=entry, x=leg(fp, 1), ek=ek, xk=fk, why='stop'))
                posm[d, ek + 1:fk] = -1
                pos = 0; stopped_side = -1
            if pos == 0:
                if px > ub and stopped_side != 1:
                    pos = 1; entry = leg(fp, 1); ek = fk
                elif px < lb and stopped_side != -1:
                    pos = -1; entry = leg(fp, -1); ek = fk
        if pos != 0:
            if eod == 'open' and ld == N_MIN - 1 and not np.isnan(O[d, ld]):
                xp, xk = O[d, ld], ld
            else:
                xp, xk = C[d, ld], ld
            trades.append(dict(d=d, side=pos, e=entry, x=leg(xp, -pos), ek=ek, xk=xk, why='eod'))
            posm[d, ek + 1:xk] = pos
    return trades, posm


def daily_returns(data, trades, cost='paper'):
    """Per-day 1x and dyn return series under one cost model (costs charged on top of slippage)."""
    D = len(data['days'])
    pnl = np.zeros(D); ntr = np.zeros(D, dtype=int)
    for tr in trades:
        e, x, side = tr['e'], tr['x'], tr['side']
        if cost == 'paper':
            c = 2 * COST_PAPER_PER_SHARE
        elif cost == 'bp':
            c = 1e-4 * (e + x)
        elif cost == 'halfbp':
            c = 0.5e-4 * (e + x)
        else:
            c = 0.0
        pnl[tr['d']] += side * (x - e) - c
        ntr[tr['d']] += 1
    r1 = pnl / data['dopen']
    lev = np.fmin(4.0, 0.02 / data['sig14'])
    valid = ~np.isnan(data['sigma'][:, 30]) & ~np.isnan(lev)
    df = pd.DataFrame({'date': data['days'], 'r1x': r1, 'rdyn': lev * r1, 'ntr': ntr, 'pnl_ps': pnl,
                       'valid': valid})
    return df[df['valid']].reset_index(drop=True)


def metrics(df, col):
    n = len(df)
    if n == 0:
        return dict(days=0, traded=0, bps=np.nan, t=np.nan, hit=np.nan, ann=np.nan,
                    sharpe=np.nan, mdd=np.nan, trades_day=np.nan)
    r = df[col].values
    eq = np.cumprod(1 + r)
    ann = (eq[-1] ** (252.0 / n) - 1) * 100 if n > 20 else np.nan
    sd = r.std(ddof=1)
    sharpe = r.mean() / sd * math.sqrt(252) if sd > 0 else np.nan
    mdd = (1 - eq / np.maximum.accumulate(eq)).max() * 100
    tr = df[df['ntr'] > 0][col].values * 1e4
    bps = tr.mean() if len(tr) else np.nan
    t = bps / tr.std(ddof=1) * math.sqrt(len(tr)) if len(tr) > 2 and tr.std(ddof=1) > 0 else np.nan
    hit = (tr > 0).mean() * 100 if len(tr) else np.nan
    return dict(days=n, traded=len(tr), bps=bps, t=t, hit=hit, ann=ann, sharpe=sharpe, mdd=mdd,
                trades_day=df['ntr'].sum() / n)


def split(df):
    return df[df['date'] <= IS_END], df[df['date'] >= OOS_START]


def fmt(v, nd=1):
    return '' if v is None or (isinstance(v, float) and (np.isnan(v))) else f'{v:.{nd}f}'


def pnl_by_minute(data, trades, posm):
    """Per-minute P&L in $/share, using the position path; entry/exit stubs included.

    Closes are forward-filled along the day so a missing bar never turns a stub into NaN.
    """
    C = pd.DataFrame(data['C']).ffill(axis=1).bfill(axis=1).values
    dC = np.zeros_like(C)
    dC[:, 1:] = C[:, 1:] - C[:, :-1]
    m = posm.astype(np.float64) * np.nan_to_num(dC)
    for tr in trades:
        d, ek, xk, side = tr['d'], tr['ek'], tr['xk'], tr['side']
        if ek >= 0:
            m[d, ek] += side * (C[d, ek] - tr['e'])
        if 0 < xk < N_MIN:
            m[d, xk] += side * (tr['x'] - C[d, xk - 1])
    return np.nan_to_num(m)


def pnl_by_entry_minute(trades):
    """Gross $/share P&L bucketed by the ENTRY bar's minute index."""
    rows = [dict(ek=t['ek'], pnl=t['side'] * (t['x'] - t['e']), why=t['why']) for t in trades]
    return pd.DataFrame(rows)
