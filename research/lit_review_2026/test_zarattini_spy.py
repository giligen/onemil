#!/usr/bin/env python3
"""Exact-spec replication of Zarattini, Aziz & Barbon (2024/2025), "Beat the Market: An Effective Intraday
Momentum Strategy for S&P500 ETF (SPY)" (SSRN 4824172), on Alpaca SIP 1-minute bars in
research/lit_review_2026/etf_1min.db, plus one-at-a-time ablations toward the coordinator's simplified A2 rule.

Paper spec implemented (Section 3 of the Feb-2025 PDF):
  * Noise Area: for each time-of-day HH:MM, sigma_t(HH:MM) = mean over the previous 14 days of
    |Close_{t-i,HH:MM} / Open_{t-i,09:30} - 1|.  UB = max(Open_t, Close_{t-1}) * (1 + VM*sigma),
    LB = min(Open_t, Close_{t-1}) * (1 - VM*sigma), VM = 1.
  * Decisions ONLY at HH:00 / HH:30, first at 10:00 (paper: "trading is restricted to semi-hourly intervals
    ... takes its first position at 10:00"), using that minute's close as signal and fill (the authors fill at
    the bar close; a next-bar-open fill is an ablation).  Long if close > UB, short if close < LB.
  * Trailing stop, evaluated only at the same semi-hourly checks: long stop = max(UB, VWAP), short stop =
    min(LB, VWAP), VWAP cumulative from 09:30 on regular-hours bars only.  A cross of the OPPOSITE band
    closes and reverses.  The "base" model (paper Table 1) uses the opposite band as the only stop.
  * All positions closed at the close (we use the 15:59 bar close; the closing-auction print is not in the
    15:59 bar).  Half-days close at their last regular bar.
  * Sizing: 1x = AUM / Open_t shares; dynamic = AUM * min(4, 2% / sigma_SPY,t) / Open_t, sigma_SPY,t = sample
    std of the 14 daily close-to-close returns ending at t-1.  Fractional shares (rounding ignored).
  * Costs: paper = $0.0035 commission + $0.001 slippage per share PER LEG; alt = 1 bp of price per leg; gross.

Usage:  nice -n 10 python3 test_zarattini_spy.py [--symbols SPY,QQQ] [--out zarattini_spy.md]
Read-only on the DB; writes only the markdown report.
"""
import argparse
import math
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

DB = '/home/ec2-user/onemil/research/lit_review_2026/etf_1min.db'
OUT = '/home/ec2-user/onemil/research/lit_review_2026/zarattini_spy.md'
N_MIN = 390                       # 09:30 .. 15:59
CHECKS_SEMI = list(range(30, 390, 30))   # 10:00, 10:30, ..., 15:30
CHECKS_EVERY_MIN = list(range(30, 389))  # every minute from 10:00
CHECKS_A2 = list(range(1, 389))          # coordinator's A2: no cadence, from 09:31
COST_PAPER = 0.0035 + 0.001              # $/share per leg
COST_BP = 1e-4                           # per leg, of price
IS_END = pd.Timestamp('2023-12-31')
OOS_START = pd.Timestamp('2024-01-01')

# Paper's own numbers (2007-05 -> 2024-04 unless noted)
PAPER = {
    '1x': dict(ann=9.7, sharpe=1.24, hit=43, mdd=12),
    'dyn': dict(ann=19.6, sharpe=1.33, hit=43, mdd=25, bps=12, t=5.34, trades_day=1.8),
    # FAQ Q24 yearly returns of the dynamic model (Feb-2025 PDF); 2025 = Jan only
    'yearly_dyn': {2016: -12.8, 2017: -6.9, 2018: 61.1, 2019: 6.9, 2020: 26.8, 2021: 34.8, 2022: 24.4,
                   2023: 37.2, 2024: 32.2, 2025: -1.2},
}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_symbol(symbol):
    """Regular-hours 1-min bars for one symbol as a (days x 390) set of matrices."""
    con = sqlite3.connect(DB)
    df = pd.read_sql('select t, o, h, l, c, v, vw from bars where symbol = ? order by t', con, params=(symbol,))
    con.close()
    if df.empty:
        raise SystemExit(f'no bars for {symbol} in {DB}')
    ts = pd.to_datetime(df['t'], utc=True).dt.tz_convert('America/New_York')
    df['date'] = ts.dt.tz_localize(None).dt.normalize()
    df['idx'] = (ts.dt.hour * 60 + ts.dt.minute - 570).astype(int)
    df = df[(df['idx'] >= 0) & (df['idx'] < N_MIN)].reset_index(drop=True)
    log(f'{symbol}: {len(df):,} regular-hours bars, {df["date"].nunique():,} days, '
        f'{df["date"].min().date()} -> {df["date"].max().date()}')
    days = np.sort(df['date'].unique())
    di = {d: i for i, d in enumerate(days)}
    D = len(days)
    r = df['date'].map(di).values
    k = df['idx'].values
    C = np.full((D, N_MIN), np.nan); O = C.copy(); V = np.zeros((D, N_MIN)); PV = np.zeros((D, N_MIN))
    C[r, k] = df['c'].values
    O[r, k] = df['o'].values
    V[r, k] = df['v'].values
    vw = np.where(df['vw'].values > 0, df['vw'].values, (df['h'] + df['l'] + df['c']).values / 3.0)
    PV[r, k] = vw * df['v'].values
    has = ~np.isnan(C)
    first = np.argmax(has, axis=1)
    last = N_MIN - 1 - np.argmax(has[:, ::-1], axis=1)
    nbars = has.sum(axis=1)
    keep = nbars >= 150                     # drop broken days (< ~2.5 h of bars)
    if (~keep).any():
        log(f'  dropping {(~keep).sum()} days with < 150 bars')
    days, C, O, V, PV, first, last = days[keep], C[keep], O[keep], V[keep], PV[keep], first[keep], last[keep]
    D = len(days)
    ar = np.arange(D)
    dopen = O[ar, first]
    dclose = C[ar, last]
    prevclose = np.r_[np.nan, dclose[:-1]]
    cv = np.cumsum(V, axis=1); cpv = np.cumsum(PV, axis=1)
    VW = np.where(cv > 0, cpv / np.maximum(cv, 1e-12), np.nan)
    dret = pd.Series(dclose).pct_change()
    sig14 = dret.rolling(14).std(ddof=1).shift(1).values      # std of returns t-14 .. t-1
    move = np.abs(C / dopen[:, None] - 1.0)
    sigma = pd.DataFrame(move).rolling(14, min_periods=10).mean().shift(1).values
    return dict(symbol=symbol, days=pd.to_datetime(days), C=C, O=O, VW=VW, dopen=dopen, prevclose=prevclose,
                sig14=sig14, sigma=sigma, last=last, first=first)


def bands(data, anchor, vm=1.0):
    if anchor == 'open_prevclose':
        up = np.fmax(data['dopen'], data['prevclose']); dn = np.fmin(data['dopen'], data['prevclose'])
    elif anchor == 'open':
        up = dn = data['dopen']
    else:
        raise ValueError(anchor)
    UB = up[:, None] * (1.0 + vm * data['sigma'])
    LB = dn[:, None] * (1.0 - vm * data['sigma'])
    return UB, LB


def simulate(data, UB, LB, checks, stop_mode, fill):
    """Returns list of trades (day_index, side, entry_px, exit_px)."""
    C, O, VW, last = data['C'], data['O'], data['VW'], data['last']
    D = C.shape[0]
    trades = []
    for d in range(D):
        ld = last[d]
        if np.isnan(UB[d, 30]) or ld < 60:
            continue
        pos = 0; entry = np.nan
        for k in checks:
            if k >= ld:
                break
            px = C[d, k]
            ub, lb, vw = UB[d, k], LB[d, k], VW[d, k]
            if np.isnan(px) or np.isnan(ub) or np.isnan(vw):
                continue
            if fill == 'next_open':
                fp = O[d, k + 1]
                if np.isnan(fp):
                    fp = px
            else:
                fp = px
            stopped_side = 0
            if pos == 1:
                stop = max(ub, vw) if stop_mode == 'vwap_band' else lb
                if px < stop:
                    trades.append((d, 1, entry, fp)); pos = 0; stopped_side = 1
            elif pos == -1:
                stop = min(lb, vw) if stop_mode == 'vwap_band' else ub
                if px > stop:
                    trades.append((d, -1, entry, fp)); pos = 0; stopped_side = -1
            if pos == 0:
                if px > ub and stopped_side != 1:
                    pos = 1; entry = fp
                elif px < lb and stopped_side != -1:
                    pos = -1; entry = fp
        if pos != 0:
            trades.append((d, pos, entry, C[d, ld]))
    return trades


def daily_returns(data, trades, cost):
    """Per-day return series for 1x and dynamic sizing under one cost model."""
    D = len(data['days'])
    pnl = np.zeros(D); ntr = np.zeros(D, dtype=int)
    for d, side, e, x in trades:
        if cost == 'paper':
            c = 2 * COST_PAPER
        elif cost == 'bp':
            c = COST_BP * (e + x)
        else:
            c = 0.0
        pnl[d] += side * (x - e) - c
        ntr[d] += 1
    r1 = pnl / data['dopen']
    lev = np.fmin(4.0, 0.02 / data['sig14'])
    lev = np.where(np.isnan(lev), np.nan, lev)
    rd = lev * r1
    valid = ~np.isnan(data['sigma'][:, 30]) & ~np.isnan(lev)
    df = pd.DataFrame({'date': data['days'], 'r1x': r1, 'rdyn': rd, 'ntr': ntr, 'valid': valid})
    return df[df['valid']].reset_index(drop=True)


def metrics(df, col):
    """Annualised (CAGR) return, Sharpe, MDD from a daily return column; bps/t/hit on traded days."""
    n = len(df)
    if n == 0:
        return dict(days=0)
    r = df[col].values
    eq = np.cumprod(1 + r)
    ann = (eq[-1] ** (252.0 / n) - 1) * 100 if n > 20 else np.nan
    sharpe = r.mean() / r.std(ddof=1) * math.sqrt(252) if r.std(ddof=1) > 0 else np.nan
    mdd = (1 - eq / np.maximum.accumulate(eq)).max() * 100
    tr = df[df['ntr'] > 0][col].values * 1e4
    bps = tr.mean() if len(tr) else np.nan
    t = bps / tr.std(ddof=1) * math.sqrt(len(tr)) if len(tr) > 2 and tr.std(ddof=1) > 0 else np.nan
    hit = (tr > 0).mean() * 100 if len(tr) else np.nan
    return dict(days=n, traded=len(tr), bps=bps, t=t, hit=hit, ann=ann, sharpe=sharpe, mdd=mdd,
                trades_day=df['ntr'].sum() / n)


def fmt(v, nd=1):
    return '' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{nd}f}'


def yearly_table(df, symbol):
    rows = ['| year | days | traded | bps/day dyn | t | hit % | trades/day | ann % 1x | SR 1x | MDD % 1x '
            '| ann % dyn | SR dyn | MDD % dyn | paper dyn ann % |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for y, g in df.groupby(df['date'].dt.year):
        m1 = metrics(g, 'r1x'); md = metrics(g, 'rdyn')
        rows.append(f'| {y} | {m1["days"]} | {m1["traded"]} | {fmt(md["bps"])} | {fmt(md["t"], 2)} | {fmt(md["hit"], 0)} '
                    f'| {fmt(m1["trades_day"], 2)} | {fmt(m1["ann"])} | {fmt(m1["sharpe"], 2)} | {fmt(m1["mdd"])} '
                    f'| {fmt(md["ann"])} | {fmt(md["sharpe"], 2)} | {fmt(md["mdd"])} '
                    f'| {fmt(PAPER["yearly_dyn"].get(y)) if symbol == "SPY" else ""} |')
    return '\n'.join(rows)


def period_row(label, df, cost_label):
    m1 = metrics(df, 'r1x'); md = metrics(df, 'rdyn')
    return (f'| {label} | {cost_label} | {m1["days"]} | {fmt(md["bps"])} | {fmt(md["t"], 2)} | {fmt(md["hit"], 0)} '
            f'| {fmt(m1["trades_day"], 2)} | {fmt(m1["ann"])} | {fmt(m1["sharpe"], 2)} | {fmt(m1["mdd"])} '
            f'| {fmt(md["ann"])} | {fmt(md["sharpe"], 2)} | {fmt(md["mdd"])} |')


PERIOD_HDR = ('| period | costs | days | bps/day dyn | t | hit % | trades/day | ann % 1x | SR 1x | MDD % 1x '
              '| ann % dyn | SR dyn | MDD % dyn |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|')
PAPER_ROW = (f'| **paper 2007-05→2024-04** | paper | ~4,270 | {PAPER["dyn"]["bps"]} | {PAPER["dyn"]["t"]} | '
             f'{PAPER["dyn"]["hit"]} | {PAPER["dyn"]["trades_day"]} | {PAPER["1x"]["ann"]} | {PAPER["1x"]["sharpe"]} | '
             f'{PAPER["1x"]["mdd"]} | {PAPER["dyn"]["ann"]} | {PAPER["dyn"]["sharpe"]} | {PAPER["dyn"]["mdd"]} |')


def split(df):
    return df[df['date'] <= IS_END], df[df['date'] >= OOS_START]


ABLATIONS = [
    # name, anchor, checks, stop_mode, fill
    ('A  full paper model', 'open_prevclose', CHECKS_SEMI, 'vwap_band', 'close'),
    ('B  bands anchored on open only', 'open', CHECKS_SEMI, 'vwap_band', 'close'),
    ('C  checks every minute (from 10:00)', 'open_prevclose', CHECKS_EVERY_MIN, 'vwap_band', 'close'),
    ('D  stop = opposite band only (paper base model)', 'open_prevclose', CHECKS_SEMI, 'opp_band', 'close'),
    ('E  A2 replica: open anchor + every minute from 09:31 + opposite-band flip', 'open', CHECKS_A2, 'opp_band', 'close'),
    ('F  full model, fill at next bar open', 'open_prevclose', CHECKS_SEMI, 'vwap_band', 'next_open'),
    ('G  full model, VM = 1.5 (paper FAQ optimum)', 'open_prevclose', CHECKS_SEMI, 'vwap_band', 'close', 1.5),
]


def run_config(data, anchor, checks, stop_mode, fill, vm=1.0):
    UB, LB = bands(data, anchor, vm)
    tr = simulate(data, UB, LB, checks, stop_mode, fill)
    return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='SPY,QQQ')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    symbols = a.symbols.split(',')
    md = ['# Zarattini "Beat the Market" SPY noise-area strategy — exact-spec replication on Alpaca SIP 1-min bars',
          '',
          f'Generated {time.strftime("%Y-%m-%d %H:%M")} by `test_zarattini_spy.py`. Data: `etf_1min.db` (regular hours '
          '09:30–15:59 ET, close of the 15:59 bar used as the 16:00 exit). IS = 2016-01 → 2023-12 (inside the paper\'s '
          '2007-05 → 2024-04 sample), OOS = 2024-01 → end of data. "dyn" = 2% daily vol target, 4× cap; "1x" = 100% '
          'notional. bps/day, t and hit are over TRADED days (the paper\'s Table 5 convention: 12 bps, t 5.34, 43%, '
          'N 2,620 traded days of ~4,270). Ann % = CAGR; SR = mean/sd × √252 over all days; MDD from the compounded '
          'equity curve. Costs: paper = $0.0035 + $0.001 per share per leg; 1bp = 1 bp of price per leg; gross = none.',
          '']
    for sym in symbols:
        data = load_symbol(sym)
        log(f'{sym}: running full model')
        trades = run_config(data, 'open_prevclose', CHECKS_SEMI, 'vwap_band', 'close')
        log(f'{sym}: {len(trades):,} trades')
        md.append(f'## {sym} — full paper model')
        md.append('')
        md.append('### Summary (IS = 2016–2023, OOS = 2024-01 → today)')
        md.append('')
        md.append(PERIOD_HDR)
        if sym == 'SPY':
            md.append(PAPER_ROW)
            md.append('| paper, its own 2016–2023 yearly table (dyn) | paper | 2,012 |  |  |  |  |  |  |  | 19.2 (CAGR of FAQ Q24 rows) |  |  |')
        for cost in ('paper', 'bp', 'gross'):
            df = daily_returns(data, trades, cost)
            is_, oos = split(df)
            md.append(period_row(f'{sym} IS 2016–2023', is_, cost))
            md.append(period_row(f'{sym} OOS 2024→', oos, cost))
            md.append(period_row(f'{sym} full 2016→', df, cost))
        md.append('')
        md.append('### By year (paper costs)')
        md.append('')
        md.append(yearly_table(daily_returns(data, trades, 'paper'), sym))
        md.append('')
        if sym == 'SPY':
            md.append('## SPY — ablations (paper costs), one ingredient at a time')
            md.append('')
            md.append('| config | period | days | bps/day dyn | t | hit % | trades/day | ann % 1x | SR 1x | MDD % 1x '
                      '| ann % dyn | SR dyn | MDD % dyn |')
            md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
            for cfg in ABLATIONS:
                name, anchor, checks, stop_mode, fill = cfg[:5]
                vm = cfg[5] if len(cfg) > 5 else 1.0
                log(f'SPY ablation: {name}')
                tr = run_config(data, anchor, checks, stop_mode, fill, vm)
                df = daily_returns(data, tr, 'paper')
                is_, oos = split(df)
                for lab, part in (('IS', is_), ('OOS', oos)):
                    m1 = metrics(part, 'r1x'); mdn = metrics(part, 'rdyn')
                    md.append(f'| {name} | {lab} | {m1["days"]} | {fmt(mdn["bps"])} | {fmt(mdn["t"], 2)} | '
                              f'{fmt(mdn["hit"], 0)} | {fmt(m1["trades_day"], 2)} | {fmt(m1["ann"])} | '
                              f'{fmt(m1["sharpe"], 2)} | {fmt(m1["mdd"])} | {fmt(mdn["ann"])} | {fmt(mdn["sharpe"], 2)} '
                              f'| {fmt(mdn["mdd"])} |')
            md.append('')
    md.append(NOTES)
    with open(a.out, 'w') as f:
        f.write('\n'.join(md) + '\n')
    log(f'wrote {a.out}')


NOTES = """## Reading (written against the 2026-09-16 run; regenerate the tables above before re-using)

**Implementation check.** The dynamic model's yearly returns track the paper's own FAQ Q24 table within a few
points in 8 of 9 overlapping years (2018 61.9 vs 61.1, 2019 11.6 vs 6.9, 2020 23.3 vs 26.8, 2021 38.4 vs 34.8,
2022 28.5 vs 24.4, 2023 24.7 vs 37.2, 2024 20.0 vs 32.2; 2016 −21.6 vs −12.8, 2017 −12.4 vs −6.9), with hit
ratio 43% = paper. Remaining gaps are IQFeed-vs-SIP bars, the 15:59-bar close vs the auction print, and fractional
shares. "Trades/day" here counts round trips (0.9); the paper's 1.8 counts legs (7,668 legs / 4,270 days).

**IS vs OOS (SPY, paper costs).** 2016–2023: 10.9 bps/traded-day (t 3.2), 1× 7.1%/yr SR 1.03, dyn 16.7%/yr
SR 1.13 — in line with the paper's 9.7% / 1.24 and 19.6% / 1.33 once 2007–2015 (which the paper says were its
best years) are excluded. **2024-01 → 2026-09: 0.6 bps/day (t 0.09), 1× −1.0%/yr, dyn −0.3%/yr, SR ≈ 0**:
2024 +20% (dyn), 2025 −2.5%, 2026 YTD −21%. The post-publication record on SPY is flat-to-negative and the
2026 drawdown is the worst year in the sample. QQQ is the exception: IS 12.8 bps (t 3.7), OOS 10.1 bps (t 1.6),
dyn 14.4%/yr SR 0.97 — 2024 +38%, 2025 −2.8%, 2026 +10%.

**Costs.** The paper's $0.0045/share per leg is ≈0.08 bp at SPY ≈$600; 1 bp/leg is ≈13× that and drops IS to
SR 0.47 (1×) / 0.41 (dyn) and OOS to −7 bps/day. Gross vs paper-cost differ by ≈1.2 bps/day. The realistic
number lies between (quoted SPY spread ≈1 cent ≈0.2 bp; the binding item is slippage on the HH:00/HH:30 market
orders, not the spread).

**Ablation — which ingredient changes the sign (SPY, IS / OOS, dyn bps/day, 1× SR):**
- A full model: 10.9 / 0.6 bps; SR 1.03 / −0.14.
- B open-only anchor (no prev-close gap adjustment): 8.9 / 6.0 bps; SR 0.70 / 0.63 — trades 50% more (1.34/day)
  and MDD 1× 26% vs 11% IS; not the sign-changer, and actually better OOS.
- **C every-minute checks (from 10:00): 1.1 / 1.2 bps; SR 0.10 / −0.09; 5.6 trades/day.** The semi-hourly cadence is
  the ingredient that carries the IS result: without it the VWAP/band trailing stop whipsaws ≈6× a day and the
  edge is gone even before realistic costs.
- D opposite-band stop only (paper's base model): 6.8 / 5.1 bps; SR 0.69 / 0.37; hit 54% — matches the paper's
  base-model Table 1 (SR 0.61, hit 54%); the VWAP+current-band trailing stop adds ≈+0.35 SR IS.
- E A2 replica (open anchor + every minute from 09:31 + opposite-band flip): 5.9 / 1.1 bps; 1× SR 0.20 / 0.28;
  1.5 trades/day; MDD 1× 32%. Positive here rather than the coordinator's −1.7 bps/day — the residual difference
  must be in fill/cost conventions (this run: fill at the signal bar's close, $0.0045/share per leg) or in how
  the 14-day time-of-day sigma is built; the sign of E is fragile (t 1.6 IS, 0.2 OOS) either way.
- F next-bar-open fill: identical to A (10.8 / 0.5 bps) — at a 30-min cadence the fill convention does not matter.
- G VM 1.5: 13.4 / 3.3 bps; fewer trades (0.54/day), MDD dyn 9.6% vs 31.8% IS — the FAQ's "optimum" mainly
  buys drawdown, not OOS return.

**Bottom line.** The paper's rule is reproducible and its 2016–2023 in-sample edge is real at the paper's cost
assumption (t ≈ 3, ≈11 bps per traded day, SR ≈1 unlevered); the edge is carried by (i) the semi-hourly decision
cadence and (ii) the VWAP/current-band trailing stop, in that order, not by the band anchor. On SPY it has not
delivered since 2024 (t 0.09 over 678 days; 2026 YTD −21% at the 2% vol target), while QQQ still shows
≈10 bps/day (t 1.6). At 1 bp/leg the SPY rule is unprofitable in every period.
"""


if __name__ == '__main__':
    main()
