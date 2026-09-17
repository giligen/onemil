#!/usr/bin/env python3
"""H/QQQ step 2/3 — candidate filters on TRAIN (2016-01..2022-12).

Day filters are a mask on the day book (the sleeve is flat at every close, so skipping a day is
exactly the removal of that day's return).  Shape filters (entry count, side gate, last-check)
are re-simulated with `simulate2`, a copy of `zsim.simulate` with three extra switches and no
other change.

Run with `--val` to apply the FROZEN stack to VAL, `--test` for TEST (once).
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/Q')
import zsim as Z                                                    # noqa: E402

H = '/home/ec2-user/onemil/research/fuckup_audit/H/QQQ/'
N_MIN = Z.N_MIN
CHECKS = Z.CHECKS_SEMI
SPLITS = {'TRAIN': ('2016-01-01', '2022-12-31'),
          'VAL': ('2023-01-01', '2024-06-30'),
          'TEST': ('2024-07-01', '2026-09-30')}
H1 = ('2016-01-01', '2019-06-30')
H2 = ('2019-07-01', '2022-12-31')


# ------------------------------------------------------------------------------------------
def simulate2(data, UB, LB, checks=CHECKS, fill='next_open', eod='moc', slip_bp=0.5,
              day_ok=None, max_entries=99, long_ok=None, short_ok=None, last_check=999):
    """zsim.simulate + (day_ok, max_entries, long_ok, short_ok, last_check). Identical otherwise.

    day_ok / long_ok / short_ok: bool arrays of length D (None = all True).
    max_entries: cap on the number of ENTRIES taken in one day.
    last_check: the largest check index k at which a NEW entry may be opened (stops always run).
    """
    C, O, VW, last = data['C'], data['O'], data['VW'], data['last']
    D = C.shape[0]
    s = slip_bp * 1e-4
    trades = []
    if day_ok is None:
        day_ok = np.ones(D, bool)
    if long_ok is None:
        long_ok = np.ones(D, bool)
    if short_ok is None:
        short_ok = np.ones(D, bool)

    def leg(px, side):
        return px * (1.0 + s) if side > 0 else px * (1.0 - s)

    for d in range(D):
        ld = last[d]
        if np.isnan(UB[d, 30]) or ld < 60 or not day_ok[d]:
            continue
        pos = 0; entry = np.nan; ek = -1; n_ent = 0
        for k in checks:
            if k >= ld:
                break
            px = C[d, k]
            ub, lb, vw = UB[d, k], LB[d, k], VW[d, k]
            if np.isnan(px) or np.isnan(ub) or np.isnan(vw):
                continue
            if fill == 'next_open':
                fp = O[d, k + 1]; fk = k + 1
                if np.isnan(fp):
                    fp = px; fk = k
            else:
                fp = px; fk = k
            stopped_side = 0
            if pos == 1 and px < max(ub, vw):
                trades.append(dict(d=d, side=1, e=entry, x=leg(fp, -1), ek=ek, xk=fk, why='stop'))
                pos = 0; stopped_side = 1
            elif pos == -1 and px > min(lb, vw):
                trades.append(dict(d=d, side=-1, e=entry, x=leg(fp, 1), ek=ek, xk=fk, why='stop'))
                pos = 0; stopped_side = -1
            if pos == 0 and n_ent < max_entries and k <= last_check:
                if px > ub and stopped_side != 1 and long_ok[d]:
                    pos = 1; entry = leg(fp, 1); ek = fk; n_ent += 1
                elif px < lb and stopped_side != -1 and short_ok[d]:
                    pos = -1; entry = leg(fp, -1); ek = fk; n_ent += 1
        if pos != 0:
            if eod == 'open' and ld == N_MIN - 1 and not np.isnan(O[d, ld]):
                xp, xk = O[d, ld], ld
            else:
                xp, xk = C[d, ld], ld
            trades.append(dict(d=d, side=pos, e=entry, x=leg(xp, -pos), ek=ek, xk=xk, why='eod'))
    return trades


# ------------------------------------------------------------------------------------------
def extra_features(data, UB, LB):
    """Causal day features beyond base.py: trailing MAs, prior-week movement."""
    dclose, dopen, prevclose = data['dclose'], data['dopen'], data['prevclose']
    c = pd.Series(dclose)
    ma20 = c.rolling(20).mean().shift(1).values      # MA of closes ending yesterday
    ma5 = c.rolling(5).mean().shift(1).values
    dret = c.pct_change()
    absmove5 = dret.abs().rolling(5).sum().shift(1).values * 100.0
    trend5 = dret.rolling(5).sum().shift(1).values * 100.0
    band_w = 100.0 * (UB[:, 30] - LB[:, 30]) / dopen
    gap = 100.0 * (dopen / prevclose - 1.0)
    rv20 = dret.rolling(20).std(ddof=1).shift(1).values * 100.0
    return pd.DataFrame(dict(date=data['days'], band_w=band_w, gap=gap,
                             gap_over_band=np.abs(gap) / band_w, rv20=rv20,
                             absmove5=absmove5, trend5=trend5,
                             above_ma20=prevclose > ma20, above_ma5=prevclose > ma5))


def day_df(data, trades, cost='gross'):
    df = Z.daily_returns(data, trades, cost)
    df['bps'] = df['r1x'] * 1e4
    return df


def stats(df, lo=None, hi=None):
    d = df
    if lo:
        d = d[(d['date'] >= lo) & (d['date'] <= hi)]
    tdd = d[d['ntr'] > 0]
    if len(tdd) < 3:
        return dict(days=len(d), traded=len(tdd), bps=np.nan, t=np.nan, sum=np.nan,
                    green=np.nan, cal=np.nan, sharpe=np.nan, mdd=np.nan)
    m = Z.metrics(d, 'r1x')
    return dict(days=len(d), traded=len(tdd), bps=tdd['bps'].mean(),
                t=tdd['bps'].mean() / tdd['bps'].std(ddof=1) * np.sqrt(len(tdd)),
                sum=d['bps'].sum(), green=100 * (tdd['bps'] > 0).mean(),
                cal=d['bps'].mean(), sharpe=m['sharpe'], mdd=m['mdd'])


def line(tag, df, lo, hi):
    a = stats(df, lo, hi)
    return (f'{tag:44s} n={a["traded"]:5d}/{a["days"]:5d} bps={a["bps"]:+7.2f} t={a["t"]:+5.2f} '
            f'sum={a["sum"]:+8.0f} green={a["green"]:4.1f}% cal={a["cal"]:+6.2f} '
            f'SR={a["sharpe"]:+5.2f} MDD={a["mdd"]:5.1f}')


def weekly_green(df, lo, hi):
    d = df[(df['date'] >= lo) & (df['date'] <= hi)].copy()
    d['wk'] = pd.to_datetime(d['date']).dt.to_period('W').astype(str)
    g = d.groupby('wk').agg(bps=('bps', 'sum'), ntr=('ntr', 'sum'))
    g = g[g['ntr'] > 0]
    return 100 * (g['bps'] > 0).mean(), len(g), g['bps'].min(), g['bps'].mean()
