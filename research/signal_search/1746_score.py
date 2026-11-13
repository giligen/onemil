"""PREREG 1,746 scorer: FINRA short-volume-ratio deciles, bottom (long) minus top (short), entry open t+1, exit close t+5.

TEST (2024+) is sealed: signal dates >= 2024-01-01 are never scored (hard filter below, no flag).
Outputs: 1746_summary.csv (+ 1746_cost_sample.csv, the 400 name-day sample for the quote-cost script) and a printed report.
Usage: nice -n 19 python3 1746_score.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '1746_cache')
SEAL = '2024-01-01'
SPLITS = {'TRAIN': ('2016-01-01', '2020-12-31'), 'T_H1_16_18': ('2016-01-01', '2018-12-31'),
          'T_H2_19_20': ('2019-01-01', '2020-12-31'), 'VAL': ('2021-01-01', '2023-12-31')}
HOLD = 5
HIGH, LOW = 100e6, 10e6
COST_FILE = os.path.join(HERE, '1746_cost.csv')
EXTRA_BP_PER_LEG = 2.0


def nw_t(x, lag=HOLD):
    """Newey-West (Bartlett) t of the mean of series x; returns (mean, t_nw, t_iid, se_nw)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n, m = len(x), x.mean()
    e = x - m
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = np.sqrt(v / n)
    return m, m / se, m / (e.std(ddof=1) / np.sqrt(n)), se


def build_panel():
    """Name-day panel: SVR on t, universe tier from t-1 (20d median raw dollar volume, raw price), returns from adjusted bars."""
    svr = pd.read_parquet(os.path.join(HERE, '1746_svr.parquet'))
    svr['svr'] = svr.short_vol / svr.total_vol
    raw = pd.read_parquet(os.path.join(CACHE, 'bars_raw.parquet'))
    adj = pd.read_parquet(os.path.join(CACHE, 'bars_all.parquet'))
    days = pd.read_csv(os.path.join(CACHE, 'trading_days.csv')).date.tolist()
    allb = sorted(set(raw.date) | set(adj.date))
    idx = {d: i for i, d in enumerate(allb)}
    raw['dv'] = raw.c * raw.v
    raw = raw.sort_values(['symbol', 'date'])
    raw['adv20'] = raw.groupby('symbol').dv.transform(lambda s: s.rolling(20, min_periods=15).median())
    # known at t-1: shift by one bar within symbol, then align to date t
    raw['adv_prev'] = raw.groupby('symbol').adv20.shift(1)
    raw['px_prev'] = raw.groupby('symbol').c.shift(1)
    adj = adj.sort_values(['symbol', 'date'])
    adj['di'] = adj.date.map(idx)
    g = adj.set_index(['symbol', 'di'])
    p = svr.merge(raw[['symbol', 'date', 'adv_prev', 'px_prev']], on=['symbol', 'date'], how='inner')
    p['di'] = p.date.map(idx)
    for name, off, col in (('o1', 1, 'o'), ('c1', 1, 'c'), ('c5', HOLD, 'c')):
        key = pd.MultiIndex.from_arrays([p.symbol, p.di + off])
        p[name] = g[col].reindex(key).to_numpy()
    p['r5'] = p.c5 / p.o1 - 1
    p['r1'] = p.c1 / p.o1 - 1
    p = p[(p.px_prev >= 5) & p.o1.notna()]
    p['year'] = p.date.str[:4].astype(int)
    p['month'] = p.date.str[:7]
    return p[p.date < SEAL]


def decile_book(p, ret='r5', sig='svr', lo=None, hi=None):
    """Per-day deciles of `sig` within the tier; returns daily DataFrame (L, S, U, nL, nS, n) and per-name contribution rows."""
    q = p[(p.adv_prev > lo) & (p.adv_prev <= hi) & p[ret].notna() & p[sig].notna()].copy()
    cnt = q.groupby('date')[sig].transform('size')
    q = q[cnt >= 50]
    q['dec'] = q.groupby('date')[sig].transform(lambda s: pd.qcut(s.rank(method='first'), 10, labels=False))
    d = q.groupby('date').apply(lambda x: pd.Series({
        'L': x.loc[x.dec == 0, ret].mean(), 'S': x.loc[x.dec == 9, ret].mean(), 'U': x[ret].mean(),
        'nL': (x.dec == 0).sum(), 'nS': (x.dec == 9).sum(), 'n': len(x)}), include_groups=False)
    d['LS'] = d.L - d.S
    return q, d


def ex_top5(q, ret):
    """Mean daily LS after dropping the 5 % of long/short name-days with the largest positive contribution."""
    nd = q.date.nunique()
    leg = q[q.dec.isin([0, 9])].copy()
    leg['w'] = 1.0 / leg.groupby(['date', 'dec'])[ret].transform('size')
    leg['c'] = np.where(leg.dec == 0, 1, -1) * leg.w * leg[ret]
    cut = int(np.ceil(0.05 * len(leg)))
    keep = leg.sort_values('c', ascending=False).iloc[cut:]
    return keep.c.sum() / nd


def report_split(q, d, name, lo, hi, ret):
    """Statistics of one split for one tier."""
    x = d[(d.index >= lo) & (d.index <= hi)]
    m, t, ti, se = nw_t(x.LS)
    qq = q[(q.date >= lo) & (q.date <= hi)]
    row = {'split': name, 'days': len(x), 'LS_bp': m * 1e4, 'NW_t': t, 'iid_t': ti, 'L_vs_U_bp': (x.L - x.U).mean() * 1e4,
           'U_minus_S_bp': (x.U - x.S).mean() * 1e4, 'long_only_vs_U_t': nw_t(x.L - x.U)[1],
           'short_share': (x.U - x.S).mean() / x.LS.mean() if x.LS.mean() != 0 else np.nan,
           'ex_top5pct_bp': ex_top5(qq, ret) * 1e4, 'green_months': (x.groupby(x.index.str[:7]).LS.mean() > 0).mean(),
           'MDE_bp_80pw': 2.8 * se * 1e4, 'mean_n': x.n.mean()}
    return row


def main():
    p = build_panel()
    print(f'panel rows {len(p)} symbols {p.symbol.nunique()} dates {p.date.min()}..{p.date.max()}', flush=True)
    rows, yearly = [], []
    for tier, lo, hi in (('liquid>100M', HIGH, 1e18), ('mid10-100M', LOW, HIGH)):
        for ret in ('r5', 'r1'):
            q, d = decile_book(p, ret, 'svr', lo, hi)
            for name, (a, b) in SPLITS.items():
                r = report_split(q, d, name, a, b, ret); r.update(tier=tier, ret=ret, sig='svr'); rows.append(r)
            if ret == 'r5':
                for y, x in d.groupby(d.index.str[:4]):
                    yearly.append({'tier': tier, 'year': y, 'LS_bp': x.LS.mean() * 1e4, 'NW_t': nw_t(x.LS)[1], 'days': len(x),
                                   'mean_n': x.n.mean()})
                if tier == 'liquid>100M':
                    q.to_parquet(os.path.join(CACHE, 'q_liquid.parquet'))
        # secondary: SVR averaged over 5 days
        pp = p.sort_values(['symbol', 'date']).copy()
        pp['svr5'] = pp.groupby('symbol').svr.transform(lambda s: s.rolling(5, min_periods=4).mean())
        q, d = decile_book(pp, 'r5', 'svr5', lo, hi)
        for name, (a, b) in SPLITS.items():
            r = report_split(q, d, name, a, b, 'r5'); r.update(tier=tier, ret='r5', sig='svr5'); rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, '1746_summary.csv'), index=False)
    pd.DataFrame(yearly).to_csv(os.path.join(HERE, '1746_yearly.csv'), index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    print(out.round(3).to_string(), flush=True)
    print(pd.DataFrame(yearly).round(2).to_string(), flush=True)
    # universe size by year (liquid tier, mean names per day)
    ql = pd.read_parquet(os.path.join(CACHE, 'q_liquid.parquet'))
    print('universe names/day by year (liquid):', ql.groupby(['year', 'date']).size().groupby('year').mean().round(0).to_dict())
    # cost sample: 400 extreme-decile VAL name-days, liquid tier
    v = ql[(ql.date >= SPLITS['VAL'][0]) & (ql.date <= SPLITS['VAL'][1]) & ql.dec.isin([0, 9])]
    v[['date', 'symbol', 'dec']].sample(400, random_state=1746).sort_values('date').to_csv(
        os.path.join(HERE, '1746_cost_sample.csv'), index=False)
    # cost application (needs 1746_cost.csv from the quote script)
    if os.path.exists(COST_FILE):
        c = pd.read_csv(COST_FILE)
        hs = c.half_spread_bp.dropna()
        print(f'measured half-spread bp: n {len(hs)} mean {hs.mean():.2f} median {hs.median():.2f} p90 {hs.quantile(.9):.2f}')
        for tier_row in out[(out.tier == 'liquid>100M') & (out.ret == 'r5') & (out.sig == 'svr')].itertuples():
            net_a = tier_row.LS_bp - 2 * (hs.mean() + EXTRA_BP_PER_LEG)
            net_b = tier_row.LS_bp - 2 * (2 * hs.mean() + EXTRA_BP_PER_LEG)
            print(f'{tier_row.split}: gross {tier_row.LS_bp:.2f} bp, net(entry hs + 2bp per leg) {net_a:.2f}, '
                  f'net(round-trip hs + 2bp per leg) {net_b:.2f}')


if __name__ == '__main__':
    main()
