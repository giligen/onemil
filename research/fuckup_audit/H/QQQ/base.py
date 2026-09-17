#!/usr/bin/env python3
"""H/QQQ step 0 — reproduce the Q baseline and build the DAY-level feature table.

The sleeve is a daily-frequency ETF book: the natural unit is the DAY (12 semi-hourly
decisions inside it).  This script

  1. reproduces Q/REPORT.md's scenario-C rows (live-convention fill, 0.5 bp/leg) on the
     Q IS/OOS split, so the simulator is anchored;
  2. writes the day-level book + every causal and descriptive day feature H needs
     (H/QQQ/day_book.csv) and the per-trade table (H/QQQ/trades.csv).

Every feature is flagged CAUSAL (known at 09:30 today or earlier) or POST (known only at
or after the close) in FEATURE_KIND below.  Only CAUSAL features may enter a filter.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/Q')
import zsim as Z                                                    # noqa: E402

H = '/home/ec2-user/onemil/research/fuckup_audit/H/QQQ/'
CHECKS = Z.CHECKS_SEMI
SLIP = 0.5

# H splits (declared in the stage brief, read once)
TRAIN = ('2016-01-01', '2022-12-31')
VAL = ('2023-01-01', '2024-06-30')
TEST = ('2024-07-01', '2026-09-30')

FEATURE_KIND = {
    # causal: computable at 09:31 ET today from bars strictly before the first decision (10:00)
    'gap_pct': 'CAUSAL', 'prev_ret_pct': 'CAUSAL', 'prev_range_pct': 'CAUSAL',
    'rv20_pct': 'CAUSAL', 'sig14_pct': 'CAUSAL', 'sigma30_pct': 'CAUSAL',
    'band_w_pct': 'CAUSAL', 'gap_over_band': 'CAUSAL', 'dow': 'CAUSAL',
    'prevrange_over_band': 'CAUSAL', 'rv20_over_band': 'CAUSAL',
    'open_ret30_pct': 'CAUSAL@10:00',   # 09:30->10:00 QQQ return, known at the first decision
    'first_cross_k': 'POST-hoc-within-day (known when it happens)',
    # post: end-of-day facts, used for anatomy only, never in a filter
    'oc_ret_pct': 'POST', 'day_range_pct': 'POST', 'noise_mult': 'POST',
    'n_cross': 'POST', 'trend_after_cross': 'POST', 'eff_ratio': 'POST',
}


def load():
    data = Z.load_symbol('QQQ')
    UB, LB = Z.bands(data, 1.0)
    return data, UB, LB


def run(data, UB, LB, fill='next_open', eod='moc', slip=SLIP, cost='gross', checks=CHECKS):
    tr, posm = Z.simulate(data, UB, LB, checks=checks, fill=fill, eod=eod, slip_bp=slip)
    df = Z.daily_returns(data, tr, cost)
    return tr, df


def day_features(data, UB, LB):
    C, O, HH, LL, VW = data['C'], data['O'], data['H'], data['L'], data['VW']
    days, dopen, dclose, prevclose = data['days'], data['dopen'], data['dclose'], data['prevclose']
    D = len(days)
    last = data['last']

    day_hi = np.nanmax(HH, axis=1)
    day_lo = np.nanmin(LL, axis=1)
    day_range_pct = 100.0 * (day_hi - day_lo) / dopen
    oc_ret_pct = 100.0 * (dclose / dopen - 1.0)
    gap_pct = 100.0 * (dopen / prevclose - 1.0)

    prev_ret_pct = np.r_[np.nan, 100.0 * (dclose[:-1] / prevclose[:-1] - 1.0)]
    prev_range_pct = np.r_[np.nan, day_range_pct[:-1]]

    dret = pd.Series(dclose).pct_change()
    rv20 = dret.rolling(20).std(ddof=1).shift(1).values * 100.0      # % daily, causal
    sig14 = data['sig14'] * 100.0

    sigma30 = data['sigma'][:, 30] * 100.0
    # band half-width at the first decision, as % of the open (the "noise band" size)
    band_w_pct = 100.0 * (UB[:, 30] - LB[:, 30]) / dopen

    # 09:30 -> 10:00 return (known AT the first decision minute)
    open_ret30_pct = 100.0 * (C[:, 30] / dopen - 1.0)

    # --- band crossings on the 12 decision bars ------------------------------------
    n_cross = np.zeros(D); first_cross_k = np.full(D, np.nan)
    first_cross_side = np.zeros(D)
    trend_after = np.full(D, np.nan); eff_ratio = np.full(D, np.nan)
    for d in range(D):
        ld = last[d]
        state = 0
        for k in CHECKS:
            if k >= ld:
                break
            px, ub, lb = C[d, k], UB[d, k], LB[d, k]
            if np.isnan(px) or np.isnan(ub):
                continue
            s = 1 if px > ub else (-1 if px < lb else 0)
            if s != 0 and s != state:
                n_cross[d] += 1
                if np.isnan(first_cross_k[d]):
                    first_cross_k[d] = k
                    first_cross_side[d] = s
                state = s
        fk = first_cross_k[d]
        if not np.isnan(fk):
            fk = int(fk)
            path = C[d, fk:ld + 1]
            path = path[~np.isnan(path)]
            if len(path) > 2:
                net = path[-1] - path[0]
                gross = np.abs(np.diff(path)).sum()
                trend_after[d] = first_cross_side[d] * 100.0 * net / dopen[d]
                eff_ratio[d] = abs(net) / gross if gross > 0 else np.nan

    noise_mult = day_range_pct / band_w_pct

    return pd.DataFrame(dict(
        date=days, dow=pd.to_datetime(days).dayofweek,
        dopen=dopen, dclose=dclose, prevclose=prevclose,
        oc_ret_pct=oc_ret_pct, day_range_pct=day_range_pct, gap_pct=gap_pct,
        prev_ret_pct=prev_ret_pct, prev_range_pct=prev_range_pct,
        rv20_pct=rv20, sig14_pct=sig14, sigma30_pct=sigma30, band_w_pct=band_w_pct,
        gap_over_band=np.abs(gap_pct) / band_w_pct,
        prevrange_over_band=prev_range_pct / band_w_pct,
        rv20_over_band=rv20 / band_w_pct,
        open_ret30_pct=open_ret30_pct,
        noise_mult=noise_mult, n_cross=n_cross, first_cross_k=first_cross_k,
        first_cross_side=first_cross_side, trend_after_cross=trend_after,
        eff_ratio=eff_ratio))


def trades_table(data, trades):
    days = data['days']
    rows = []
    for t in trades:
        d = t['d']
        rows.append(dict(date=days[d], side=t['side'], entry=t['e'], exit=t['x'],
                         ek=t['ek'], xk=t['xk'], why=t['why'],
                         ret_bps=1e4 * t['side'] * (t['x'] - t['e']) / data['dopen'][d]))
    return pd.DataFrame(rows)


def label_split(dates):
    s = pd.Series('none', index=range(len(dates)))
    dts = pd.to_datetime(dates)
    s[(dts >= TRAIN[0]) & (dts <= TRAIN[1])] = 'TRAIN'
    s[(dts >= VAL[0]) & (dts <= VAL[1])] = 'VAL'
    s[(dts >= TEST[0]) & (dts <= TEST[1])] = 'TEST'
    return s.values


def main():
    data, UB, LB = load()
    tr, df = run(data, UB, LB)

    # --- anchor: reproduce Q/REPORT.md scenario C -----------------------------------
    is_, oos = Z.split(df)
    for lab, part in (('IS 2016-2023', is_), ('OOS 2024-2026', oos)):
        m = Z.metrics(part, 'r1x')
        print(f'ANCHOR QQQ C {lab}: days={m["days"]} traded={m["traded"]} '
              f'bps={m["bps"]:.2f} t={m["t"]:.2f} ann={m["ann"]:.2f} SR={m["sharpe"]:.2f} '
              f'MDD={m["mdd"]:.1f} trades/day={m["trades_day"]:.2f}', flush=True)

    feat = day_features(data, UB, LB)
    book = df.merge(feat, on='date', how='left')
    book['split'] = label_split(book['date'].values)
    book['bps'] = book['r1x'] * 1e4
    book['week'] = pd.to_datetime(book['date']).dt.to_period('W').astype(str)
    book['month'] = pd.to_datetime(book['date']).dt.to_period('M').astype(str)
    book.to_csv(H + 'day_book.csv', index=False)

    tt = trades_table(data, tr)
    tt['split'] = label_split(tt['date'].values)
    tt.to_csv(H + 'trades.csv', index=False)

    print(f'\nday_book.csv {len(book)} rows; trades.csv {len(tt)} rows')
    for s in ('TRAIN', 'VAL', 'TEST'):
        b = book[book['split'] == s]
        t2 = b[b['ntr'] > 0]['bps']
        print(f'{s}: days={len(b)} traded={len(t2)} bps/traded={t2.mean():.2f} '
              f'bps/cal={b["bps"].mean():.2f} sum={b["bps"].sum():.0f}', flush=True)


if __name__ == '__main__':
    main()
