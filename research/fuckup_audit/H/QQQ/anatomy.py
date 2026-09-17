#!/usr/bin/env python3
"""H/QQQ step 1 — anatomy of the losing DAYS and WEEKS on TRAIN (2016-01..2022-12).

TRAIN only.  VAL and TEST are not read here.
Outputs: H/QQQ/step1_*.csv and a printed markdown-ready digest.
"""
import numpy as np
import pandas as pd

H = '/home/ec2-user/onemil/research/fuckup_audit/H/QQQ/'
pd.set_option('display.width', 200)


def q(s, p):
    return s.quantile(p)


def main():
    b = pd.read_csv(H + 'day_book.csv', parse_dates=['date'])
    tr = pd.read_csv(H + 'trades.csv', parse_dates=['date'])
    T = b[b['split'] == 'TRAIN'].reset_index(drop=True)
    TT = tr[tr['split'] == 'TRAIN'].reset_index(drop=True)
    td = T[T['ntr'] > 0].copy()          # traded days only

    print('== 0. TRAIN book ==')
    print(f'days={len(T)} traded={len(td)} bps/traded={td["bps"].mean():.2f} '
          f'sd={td["bps"].std():.1f} t={td["bps"].mean()/td["bps"].std()*np.sqrt(len(td)):.2f} '
          f'sum={T["bps"].sum():.0f} bps  green={100*(td["bps"]>0).mean():.1f}%')
    h1 = T[T['date'] < '2019-07-01']; h2 = T[T['date'] >= '2019-07-01']
    for lab, part in (('H1 2016-2019H1', h1), ('H2 2019H2-2022', h2)):
        p = part[part['ntr'] > 0]
        print(f'  {lab}: traded={len(p)} bps/traded={p["bps"].mean():.2f} '
              f't={p["bps"].mean()/p["bps"].std()*np.sqrt(len(p)):.2f} sum={part["bps"].sum():.0f}')

    # ---- 1. concentration -------------------------------------------------------
    print('\n== 1. concentration (traded days) ==')
    s = td['bps'].sort_values()
    tot = s.sum()
    losers = s[s < 0]
    print(f'total {tot:.0f} bps; losing days {len(losers)} ({100*len(losers)/len(s):.1f}%) '
          f'summing {losers.sum():.0f} bps; winning days sum {s[s>0].sum():.0f}')
    for frac in (0.05, 0.10, 0.20):
        n = int(round(frac * len(s)))
        print(f'  worst {frac:.0%} of traded days (n={n}): {s.iloc[:n].sum():.0f} bps '
              f'= {100*s.iloc[:n].sum()/losers.sum():.0f}% of all losses, '
              f'{-100*s.iloc[:n].sum()/s[s>0].sum():.0f}% of all gains')
        print(f'  best  {frac:.0%} of traded days (n={n}): {s.iloc[-n:].sum():.0f} bps '
              f'= {100*s.iloc[-n:].sum()/s[s>0].sum():.0f}% of all gains')
    for lab, keep in (('full', s), ('ex top 5% days', s.iloc[:int(0.95*len(s))]),
                      ('ex worst 5% days', s.iloc[int(0.05*len(s)):]),
                      ('capped at +/-100bps', s.clip(-100, 100))):
        print(f'  {lab}: mean {keep.mean():.2f} bps  sum {keep.sum():.0f}')

    # ---- 2. the worst 30 days, annotated ---------------------------------------
    cols = ['date', 'dow', 'bps', 'ntr', 'oc_ret_pct', 'gap_pct', 'day_range_pct',
            'band_w_pct', 'noise_mult', 'n_cross', 'first_cross_k', 'first_cross_side',
            'trend_after_cross', 'eff_ratio', 'prev_ret_pct', 'prev_range_pct',
            'rv20_pct', 'open_ret30_pct']
    w30 = td.nsmallest(30, 'bps')[cols].copy()
    # side of the day's first trade + whether the day's close agreed with it
    fs = TT.sort_values(['date', 'ek']).groupby('date').first()['side']
    w30['first_side'] = w30['date'].map(fs)
    w30['side_x_oc'] = w30['first_side'] * w30['oc_ret_pct']
    w30.to_csv(H + 'step1_worst30.csv', index=False)
    b30 = td.nlargest(30, 'bps')[cols].copy()
    b30['first_side'] = b30['date'].map(fs)
    b30['side_x_oc'] = b30['first_side'] * b30['oc_ret_pct']
    b30.to_csv(H + 'step1_best30.csv', index=False)
    print('\n== 2. worst 30 TRAIN days -> step1_worst30.csv ==')
    print(w30.head(30).to_string(index=False,
          float_format=lambda v: f'{v:,.2f}'))

    print('\n-- worst30 vs best30 vs all-traded, means --')
    feat = ['oc_ret_pct', 'gap_pct', 'day_range_pct', 'band_w_pct', 'noise_mult', 'n_cross',
            'first_cross_k', 'trend_after_cross', 'eff_ratio', 'prev_ret_pct',
            'prev_range_pct', 'rv20_pct', 'open_ret30_pct', 'ntr', 'side_x_oc']
    td2 = td.copy(); td2['first_side'] = td2['date'].map(fs)
    td2['side_x_oc'] = td2['first_side'] * td2['oc_ret_pct']
    comp = pd.DataFrame({'worst30': w30[feat].mean(), 'best30': b30[feat].mean(),
                         'all_traded': td2[feat].mean(),
                         'losing_days': td2[td2['bps'] < 0][feat].mean(),
                         'winning_days': td2[td2['bps'] > 0][feat].mean()})
    print(comp.to_string(float_format=lambda v: f'{v:,.3f}'))
    comp.to_csv(H + 'step1_loser_vs_winner.csv')

    # dow table
    print('\n-- by day of week (traded days) --')
    g = td2.groupby('dow')['bps'].agg(['count', 'mean', 'sum'])
    g['green%'] = td2.groupby('dow')['bps'].apply(lambda x: 100 * (x > 0).mean())
    print(g.to_string(float_format=lambda v: f'{v:,.2f}'))

    # ---- 3. terciles / quartiles of every causal feature ------------------------
    print('\n== 3. causal-feature buckets on TRAIN (traded days) ==')
    rows = []
    causal = ['band_w_pct', 'noise_mult_NA', 'gap_pct', 'abs_gap_pct', 'gap_over_band',
              'prev_ret_pct', 'prev_range_pct', 'prevrange_over_band', 'rv20_pct',
              'rv20_over_band', 'sig14_pct', 'sigma30_pct', 'open_ret30_pct',
              'abs_open_ret30_pct', 'first_cross_k']
    td2['abs_gap_pct'] = td2['gap_pct'].abs()
    td2['abs_open_ret30_pct'] = td2['open_ret30_pct'].abs()
    td2 = td2.drop(columns=[c for c in ['noise_mult_NA'] if c in td2])
    h1t = td2[td2['date'] < '2019-07-01']; h2t = td2[td2['date'] >= '2019-07-01']
    for f in causal:
        if f not in td2.columns:
            continue
        x = td2[f].dropna()
        if len(x) < 100:
            continue
        cuts = x.quantile([0, .25, .5, .75, 1.0]).values
        cuts[0] -= 1e-9
        lab = pd.cut(td2[f], bins=np.unique(cuts), labels=False, include_lowest=True)
        for i in range(4):
            m = lab == i
            if m.sum() < 20:
                continue
            sub = td2[m]
            s1 = h1t[(h1t.index.isin(td2[m].index))]['bps']
            s2 = h2t[(h2t.index.isin(td2[m].index))]['bps']
            rows.append(dict(feature=f, q=i + 1,
                             lo=np.unique(cuts)[i], hi=np.unique(cuts)[i + 1],
                             n=int(m.sum()), mean_bps=sub['bps'].mean(),
                             t=sub['bps'].mean() / sub['bps'].std() * np.sqrt(len(sub)),
                             sum_bps=sub['bps'].sum(),
                             green=100 * (sub['bps'] > 0).mean(),
                             h1_n=len(s1), h1_mean=s1.mean() if len(s1) else np.nan,
                             h2_n=len(s2), h2_mean=s2.mean() if len(s2) else np.nan))
    bk = pd.DataFrame(rows)
    bk.to_csv(H + 'step1_buckets.csv', index=False)
    print(bk.to_string(index=False, float_format=lambda v: f'{v:,.2f}'))

    # ---- 4. weeks ---------------------------------------------------------------
    print('\n== 4. weeks ==')
    wk = T.groupby('week').agg(days=('bps', 'size'), traded=('ntr', lambda x: (x > 0).sum()),
                               bps=('bps', 'sum'), oc=('oc_ret_pct', 'sum'),
                               rng=('day_range_pct', 'mean'), rv=('rv20_pct', 'mean'),
                               bw=('band_w_pct', 'mean'), nm=('noise_mult', 'mean'),
                               ncr=('n_cross', 'mean')).reset_index()
    wk = wk[wk['traded'] > 0]
    wk['abs_oc'] = wk['oc'].abs()
    print(f'weeks={len(wk)} green={100*(wk["bps"]>0).mean():.1f}% mean={wk["bps"].mean():.1f} bps '
          f'worst={wk["bps"].min():.0f} best={wk["bps"].max():.0f}')
    for f in ('abs_oc', 'oc', 'rv', 'bw', 'nm', 'ncr', 'rng'):
        cuts = wk[f].quantile([0, 1 / 3, 2 / 3, 1.0]).values
        cuts[0] -= 1e-9
        lab = pd.cut(wk[f], bins=np.unique(cuts), labels=False, include_lowest=True)
        line = []
        for i in range(3):
            m = lab == i
            line.append(f'T{i+1} n={m.sum()} {wk[m]["bps"].mean():+7.1f} '
                        f'green {100*(wk[m]["bps"]>0).mean():.0f}%')
        print(f'  week {f:8s}: ' + ' | '.join(line))
    wk.to_csv(H + 'step1_weeks.csv', index=False)

    # ---- 5. path shape: did the day trend or chop after the first crossing ------
    print('\n== 5. day shape after the first crossing (traded days) ==')
    x = td2.dropna(subset=['eff_ratio'])
    for lab, m in (('eff_ratio < 0.15 (chop)', x['eff_ratio'] < 0.15),
                   ('0.15-0.35', (x['eff_ratio'] >= 0.15) & (x['eff_ratio'] < 0.35)),
                   ('>= 0.35 (trend)', x['eff_ratio'] >= 0.35)):
        sub = x[m]
        print(f'  {lab:24s} n={len(sub):4d} mean={sub["bps"].mean():+7.2f} '
              f'green={100*(sub["bps"]>0).mean():.0f}%')
    print('  corr(bps, trend_after_cross) = '
          f'{x["bps"].corr(x["trend_after_cross"]):.3f}; '
          f'corr(bps, eff_ratio) = {x["bps"].corr(x["eff_ratio"]):.3f}; '
          f'corr(bps, n_cross) = {x["bps"].corr(x["n_cross"]):.3f}')

    # ---- 6. autocorrelation: is there anything in "after N losing days"? --------
    print('\n== 6. day-to-day dependence ==')
    tdx = td2.sort_values('date').reset_index(drop=True)
    tdx['prev_bps'] = tdx['bps'].shift(1)
    tdx['prev2_bps'] = tdx['bps'].shift(2)
    print(f'  lag-1 corr of traded-day bps: {tdx["bps"].corr(tdx["prev_bps"]):.3f}')
    for lab, m in (('after a losing traded day', tdx['prev_bps'] < 0),
                   ('after a winning traded day', tdx['prev_bps'] > 0),
                   ('after 2 losing traded days', (tdx['prev_bps'] < 0) & (tdx['prev2_bps'] < 0))):
        sub = tdx[m]
        print(f'  {lab:28s} n={len(sub):4d} mean={sub["bps"].mean():+7.2f}')

    # ---- 7. entry-minute / trade-level -----------------------------------------
    print('\n== 7. trade level (TRAIN) ==')
    TT2 = TT.copy()
    print(TT2.groupby('why')['ret_bps'].agg(['count', 'mean', 'sum']).to_string(
        float_format=lambda v: f'{v:,.2f}'))
    print(TT2.groupby('side')['ret_bps'].agg(['count', 'mean', 'sum']).to_string(
        float_format=lambda v: f'{v:,.2f}'))
    em = TT2.groupby('ek')['ret_bps'].agg(['count', 'mean', 'sum'])
    print('-- by entry minute index (30=10:00) --')
    print(em.to_string(float_format=lambda v: f'{v:,.2f}'))
    em.to_csv(H + 'step1_entry_minute.csv')


if __name__ == '__main__':
    main()
