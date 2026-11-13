"""Cell 1,748 score: no-news vs news extreme-loser reversal, ranks 1,001-3,000.  GROSS auction-open(t+1) -> close(t+1) (primary)
and close(t+3) (secondary).  TEST (>=2024) excluded unless --test.  Kill list of PREREG_1748 applied literally."""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '1748_cache')
SPLITS = {'TRAIN': ('2016-01-01', '2020-12-31'), 'T_H1_16_18': ('2016-01-01', '2018-12-31'), 'T_H2_19_20': ('2019-01-01', '2020-12-31'),
          'VAL': ('2021-01-01', '2023-12-31'), 'TEST': ('2024-01-01', '2026-11-30')}
EXTRA_BP = 2.0
rng = np.random.default_rng(1748)


def log(m):
    print(m, flush=True)


def cl_mean_t(df, col):
    """(mean, day-clustered t, iid t) of df[col] (percent)."""
    x = df[col].values; n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan
    mu = x.mean(); e = (df[col] - mu).groupby(df.date).sum()
    se_c = np.sqrt((e ** 2).sum()) / n
    return mu, mu / se_c if se_c > 0 else np.nan, mu / (x.std(ddof=1) / np.sqrt(n))


def diff_t(df, col):
    """NO-NEWS minus NEWS mean and day-clustered t (OLS on news dummy, cluster by date)."""
    a, b = df[df.news == 0], df[df.news == 1]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    d = a[col].mean() - b[col].mean()
    x = (df.news == 0).astype(float).values; y = df[col].values
    xc = x - x.mean(); beta = (xc * (y - y.mean())).sum() / (xc ** 2).sum()
    res = y - y.mean() - beta * xc
    g = pd.Series(xc * res).groupby(df.date.values).sum()
    se = np.sqrt((g ** 2).sum()) / (xc ** 2).sum()
    return d, beta / se


def ex_top5(x):
    """Mean after dropping the top 5 % of returns."""
    x = np.sort(np.asarray(x)); k = int(np.ceil(0.05 * len(x)))
    return x[:len(x) - k].mean() if len(x) > k else np.nan


def placebo_pct(sub, pl, obs, B=500):
    """Percentile of the observed mean among B draws of one random same-name post-window placebo day per event."""
    cands = [pl.get(f'{r.date}|{r.symbol}', []) for r in sub.itertuples()]
    ok = [c for c in cands if len(c)]
    if len(ok) < 20:
        return np.nan, np.nan
    m = np.array([[c[rng.integers(len(c))] for c in ok] for _ in range(B)]).mean(axis=1) * 100
    return (m < obs).mean() * 100, m.mean()


def load():
    """Events with price-scale check, returns, VIX proxy tercile."""
    ev = pd.read_parquet(os.path.join(HERE, '1748_events.parquet'))
    n0 = len(ev)
    ev = ev.dropna(subset=['o_1', 'c_1', 'c_3']); ev = ev[ev.o_1 > 0]
    n1 = len(ev)
    bad = (ev.ret - ev.ret_raw).abs() > 0.03
    log(f'events {n0}; dropped missing next-day bars {n0 - n1}; price-scale mismatch (|adj-raw| > 3pp) dropped {int(bad.sum())}')
    ev = ev[~bad].copy()
    ev['r1'] = (ev.c_1 / ev.o_1 - 1) * 100; ev['r3'] = (ev.c_3 / ev.o_1 - 1) * 100
    ev['gap_dn'] = (ev.o_1 < ev.c_t).astype(int); ev['gap'] = (ev.o_1 / ev.c_t - 1) * 100
    ev['atr_pct'] = ev.atr14 / ev.c_t * 100
    ev['bucket'] = np.where(ev['rank'] <= 2000, '1001-2000', '2001-3000')
    sv = pd.read_parquet(os.path.join(CACHE, 'spy_rv.parquet'))
    ev = ev.merge(sv.reset_index().rename(columns={'index': 'date'}), on='date', how='left')
    ev['resid'] = ev.ret - ev.spy_ret
    return ev


def main(test=False):
    ev = load()
    pl = pd.read_pickle(os.path.join(CACHE, 'placebo.pkl'))
    tr = ev[(ev.date >= SPLITS['TRAIN'][0]) & (ev.date <= SPLITS['TRAIN'][1])]
    q1, q2 = tr.rv20.quantile([1 / 3, 2 / 3])
    ev['vix_t'] = np.where(ev.rv20 <= q1, 'low', np.where(ev.rv20 <= q2, 'mid', 'high'))
    log(f'VIX proxy = SPY 20d realised vol (cache.db has no VIX before 2024-12); TRAIN tercile cuts {q1:.1f} / {q2:.1f}')
    cost = pd.read_csv(os.path.join(HERE, '1748_cost.csv')) if os.path.exists(os.path.join(HERE, '1748_cost.csv')) else None
    rows, out = [], {}
    names = ['TRAIN', 'T_H1_16_18', 'T_H2_19_20', 'VAL'] + (['TEST'] if test else [])
    for sp in names:
        lo, hi = SPLITS[sp]; s = ev[(ev.date >= lo) & (ev.date <= hi)]
        yrs = (pd.Timestamp(hi) - pd.Timestamp(lo)).days / 365.25
        for leg, sub in (('NONEWS', s[s.news == 0]), ('NEWS', s[s.news == 1]), ('ALL', s)):
            for hz in ('r1', 'r3'):
                mu, tc, ti = cl_mean_t(sub, hz)
                r = dict(split=sp, leg=leg, hz=hz, n=len(sub), per_yr=len(sub) / yrs, per_wk=len(sub) / yrs / 52, mean_bp=mu * 100, t_clust=tc, t_iid=ti,
                         ex_top5_bp=ex_top5(sub[hz]) * 100 if len(sub) else np.nan, capped10_bp=sub[hz].clip(upper=10).mean() * 100 if len(sub) else np.nan,
                         gap_down_share=sub.gap_dn.mean() if len(sub) else np.nan, atr_pct=sub.atr_pct.median() if len(sub) else np.nan)
                if hz == 'r1' and leg != 'ALL':
                    p = placebo_pct(sub, pl, mu); r['placebo_pct'], r['placebo_mean_bp'] = p[0], p[1] * 100 if pd.notna(p[1]) else np.nan
                rows.append(r)
        d, td = diff_t(s, 'r1'); rows.append(dict(split=sp, leg='NONEWS-NEWS', hz='r1', n=len(s), mean_bp=d * 100, t_clust=td))
        d, td = diff_t(s, 'r3'); rows.append(dict(split=sp, leg='NONEWS-NEWS', hz='r3', n=len(s), mean_bp=d * 100, t_clust=td))
        for v in ('low', 'mid', 'high'):
            sub = s[(s.news == 0) & (s.vix_t == v)]; mu, tc, _ = cl_mean_t(sub, 'r1')
            rows.append(dict(split=sp, leg=f'NONEWS_VIX_{v}', hz='r1', n=len(sub), mean_bp=mu * 100, t_clust=tc))
        for b in ('1001-2000', '2001-3000'):
            sub = s[(s.news == 0) & (s.bucket == b)]; mu, tc, _ = cl_mean_t(sub, 'r1')
            rows.append(dict(split=sp, leg=f'NONEWS_RANK_{b}', hz='r1', n=len(sub), mean_bp=mu * 100, t_clust=tc))
        sub = s[(s.news == 0) & (s.resid <= -0.08)]; mu, tc, _ = cl_mean_t(sub, 'r1')
        rows.append(dict(split=sp, leg='NONEWS_SPYRESID<=-8%', hz='r1', n=len(sub), mean_bp=mu * 100, t_clust=tc))
    R = pd.DataFrame(rows)
    # measured cost on VAL no-news
    hs_e = hs_x = np.nan
    if cost is not None:
        hs_e, hs_x = cost.entry_hs_bp.mean(), cost.exit_hs_bp.mean()
        log(f'cost sample n {len(cost)}: entry half-spread mean {hs_e:.1f} bp (cover {cost.entry_hs_bp.notna().mean():.2f}), exit {hs_x:.1f} bp (cover {cost.exit_hs_bp.notna().mean():.2f})')
    rt = (hs_e + hs_x + 4 * EXTRA_BP / 2) if cost is not None else np.nan
    R['cost_rt_bp'] = rt; R['net_bp'] = R.mean_bp - rt
    R.to_csv(os.path.join(HERE, '1748_summary.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
    log(R.round(2).to_string())
    # kill list
    g = lambda sp, leg, hz='r1', c='mean_bp': R[(R.split == sp) & (R.leg == leg) & (R.hz == hz)][c].iloc[0]
    v = g('VAL', 'NONEWS'); tv = g('VAL', 'NONEWS', c='t_clust')
    se_v = v / tv if tv and not np.isnan(tv) else np.nan
    kills = []
    if not (v > 40 and abs(tv) >= 2): kills.append(f'VAL no-news gross {v:.1f} bp, t {tv:.2f} (need > +40 and |t|>=2)')
    if not (v > g('VAL', 'NEWS')): kills.append('NO-NEWS not above NEWS on VAL')
    for sp in ('TRAIN', 'VAL'):
        t3 = {k: g(sp, f'NONEWS_VIX_{k}') for k in ('low', 'mid', 'high')}
        if not (t3['high'] == max(t3.values())): kills.append(f'{sp}: high-VIX tercile not largest {({k: round(x, 1) for k, x in t3.items()})}')
    if not (np.sign(g('T_H1_16_18', 'NONEWS')) == np.sign(g('T_H2_19_20', 'NONEWS')) == np.sign(g('TRAIN', 'NONEWS'))):
        kills.append(f"TRAIN halves sign {g('T_H1_16_18', 'NONEWS'):.1f}/{g('T_H2_19_20', 'NONEWS'):.1f} bp")
    if not (g('VAL', 'NONEWS', c='ex_top5_bp') > 0): kills.append(f"VAL ex-top-5 % {g('VAL', 'NONEWS', c='ex_top5_bp'):.1f} bp <= 0")
    if not (g('VAL', 'NONEWS', c='net_bp') > 0): kills.append(f"VAL net {g('VAL', 'NONEWS', c='net_bp'):.1f} bp <= 0 (cost rt {rt:.1f})")
    if not (g('VAL', 'NONEWS', c='per_wk') >= 3): kills.append(f"VAL no-news {g('VAL', 'NONEWS', c='per_wk'):.2f} events/week < 3")
    log(f'MDE 80 % power on VAL no-news r1 (2.8 x clustered SE): {2.8 * abs(se_v):.1f} bp; kill threshold +40 bp')
    log('VERDICT: ' + ('KILL -- ' + ' | '.join(kills) if kills else 'SURVIVE (TEST still sealed; run with --test once)'))


if __name__ == '__main__':
    main('--test' in sys.argv)
