"""Cell 1,747 score: dividend ex-day (B1), run-up (B2), post-ex short (B3).  Gross excess vs matched control (same name,
same calendar day +-1 month, no dividend within 20 d) and vs the equal-weight universe mean.  TEST (>=2024) is excluded
unless --test is passed.  Usage: python 1747_score.py [--test]"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '1747_cache')
SPLITS = {'TRAIN': ('2016-01-01', '2020-12-31'), 'T_H1_16_18': ('2016-01-01', '2018-12-31'),
          'T_H2_19_20': ('2019-01-01', '2020-12-31'), 'VAL': ('2021-01-01', '2023-12-31'),
          'VAL_2022': ('2022-01-01', '2022-12-31'), 'TEST': ('2024-01-01', '2026-11-30')}
TIERS = {'liq20M': (20e6, 1e18), 'mid5_20M': (5e6, 20e6)}
RUNUP, POST = 10, 40
COST_FILE = os.path.join(HERE, '1747_cost.csv')
EXTRA_BP = 2.0


def log(m):
    print(m, flush=True)


def clean_events():
    """Regular cash dividends: rate>0, not special, 3x-trailing-rate rule, dedup (symbol, ex_date); writes 1747_events.parquet."""
    ev = pd.read_parquet(os.path.join(HERE, '1747_events_raw.parquet'))
    n0 = len(ev)
    ev = ev[(ev.rate > 0) & (~ev.special.fillna(False)) & (~ev.foreign.fillna(False)) & ev.currency.isna() & ev.sub_type.isna()
            & (ev.ex_date <= '2026-12-31')].copy()   # foreign issuers: rate currency unverifiable -> excluded (stated in RESULT)
    ev = ev.sort_values(['symbol', 'ex_date', 'rate'], ascending=[True, True, False]).drop_duplicates(['symbol', 'ex_date'])
    ev['trail'] = ev.groupby('symbol').rate.transform(lambda s: s.shift(1).rolling(4, min_periods=1).median())
    big = (ev.trail.notna()) & (ev.rate > 3 * ev.trail)
    log(f'events raw {n0}; regular {len(ev)}; dropped by 3x-trailing rule {int(big.sum())}')
    ev = ev[~big].copy()
    ev.to_parquet(os.path.join(HERE, '1747_events.parquet'))
    return ev


def load_panels():
    """Wide close/volume matrices (split-adjusted) and raw close; calendar of trading days."""
    b = pd.read_parquet(os.path.join(CACHE, 'bars_split.parquet')).drop_duplicates(['symbol', 'date'])
    r = pd.read_parquet(os.path.join(CACHE, 'bars_raw.parquet')).drop_duplicates(['symbol', 'date'])
    C = b.pivot(index='date', columns='symbol', values='c').sort_index()
    V = b.pivot(index='date', columns='symbol', values='v').reindex(C.index)
    RC = r.pivot(index='date', columns='symbol', values='c').reindex(index=C.index, columns=C.columns)
    keep = C.notna().sum(axis=1) > 500
    return C[keep], V[keep], RC[keep]


def tstats(x, cl):
    """(mean, iid t, cluster t) for 1-D array x with cluster labels cl."""
    x = np.asarray(x, float); n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan
    m = x.mean(); sd = x.std(ddof=1)
    ti = m / (sd / np.sqrt(n))
    g = pd.Series(x - m).groupby(np.asarray(cl)).sum()
    se_c = np.sqrt((g ** 2).sum()) / n
    return m, ti, m / se_c if se_c > 0 else np.nan, se_c


def build(ev, C, V, RC, with_test):
    """One row per event with returns, controls, universe means for B1/B2/B3."""
    dates = C.index.values.astype(str); di = pd.Index(dates)
    dv = (C * V).rolling(20, min_periods=15).median()
    R1 = C.pct_change(fill_method=None)
    elig = (dv.shift(1) >= 5e6) & (C.shift(1) >= 5)
    U = R1.where(elig).mean(axis=1).fillna(0.0).values
    cumU = np.concatenate([[1.0], np.cumprod(1 + U[1:])])         # universe index
    symcol = {s: i for i, s in enumerate(C.columns)}
    Cv, DV, RCv = C.values, dv.values, RC.values
    allex = ev.groupby('symbol').ex_date.apply(lambda s: pd.to_datetime(s).values.astype('datetime64[D]'))
    dts = pd.to_datetime(dates).values.astype('datetime64[D]')
    out = []
    lo_dt = '2016-01-01'; hi_dt = '2026-12-31' if with_test else '2023-12-31'
    for e in ev.itertuples():
        if e.ex_date < lo_dt or e.ex_date > hi_dt or e.symbol not in symcol:
            continue
        j = symcol[e.symbol]; i = di.searchsorted(e.ex_date)
        if i >= len(di) or dates[i] != e.ex_date or i < RUNUP + 3 or i + POST >= len(di):
            continue
        a = Cv[:, j]
        px_prev, dvv = a[i - 2], DV[i - 2, j]
        if not (px_prev >= 5) or np.isnan(dvv) or dvv < 5e6:
            continue
        fac = a[i - 1] / RCv[i - 1, j] if RCv[i - 1, j] > 0 else np.nan   # split-adjust the dividend
        div = e.rate * fac
        row = {'symbol': e.symbol, 'ex_date': e.ex_date, 'dv20': dvv, 'px': px_prev, 'div_yield_bp': div / a[i - 1] * 1e4}
        # event returns
        row['B1'] = (a[i] + div - a[i - 1]) / a[i - 1]
        row['B2'] = (a[i - 1] - a[i - RUNUP]) / a[i - RUNUP]
        row['B3_long'] = (a[i + POST] - a[i]) / a[i]
        row['U1'] = cumU[i] / cumU[i - 1] - 1
        row['U2'] = cumU[i - 1] / cumU[i - RUNUP] - 1
        row['U3'] = cumU[i + POST] / cumU[i] - 1
        # controls: same name, calendar day +-30 d, no dividend within 20 d of the window
        exs = allex.get(e.symbol, np.array([], dtype='datetime64[D]'))
        for book, (s_off, L) in {'B1': (-1, 1), 'B2': (-RUNUP, RUNUP - 1), 'B3': (0, POST)}.items():
            cs = []
            for sh in (-30, 30):
                tgt = dts[i + s_off] + np.timedelta64(sh, 'D')
                k = int(np.searchsorted(dts, tgt))
                if k < 1 or k + L >= len(dts):
                    continue
                w0, w1 = dts[k] - np.timedelta64(20, 'D'), dts[k + L] + np.timedelta64(20, 'D')
                if ((exs >= w0) & (exs <= w1)).any():
                    continue
                p0, p1 = a[k], a[k + L]
                if np.isnan(p0) or np.isnan(p1) or p0 <= 0:
                    continue
                cs.append(p1 / p0 - 1)
            row[f'ctl_{book}'] = np.mean(cs) if cs else np.nan
        out.append(row)
    d = pd.DataFrame(out)
    d['year'] = d.ex_date.str[:4]; d['month'] = d.ex_date.str[:7]
    d['B3'] = -d.B3_long
    d['ctl_B3'] = -d.ctl_B3
    d['U3s'] = -d.U3
    log(f'events scored {len(d)} (ex_date <= {hi_dt})')
    return d


def read(d, book, ret, ctl, uni, cost_bp=None):
    """Summary dict for one (split, tier, book) cell."""
    out = {'n': len(d)}
    x_c = (d[ret] - d[ctl]).dropna(); idx = x_c.index
    x_u = (d[ret] - d[uni]).dropna()
    for nm, x in (('ctl', x_c), ('uni', x_u)):
        m, ti, tc, se = tstats(x.values, d.loc[x.index, 'ex_date'].values)
        out[f'{nm}_n'] = len(x); out[f'{nm}_mean_bp'] = m * 1e4; out[f't_iid_{nm}'] = ti; out[f't_day_{nm}'] = tc
        out[f'mde80_bp_{nm}'] = 2.8 * se * 1e4 if se == se else np.nan
    x = x_c
    k = max(1, int(round(0.05 * len(x))))
    out['ex_top5_ctl_bp'] = x.sort_values().iloc[:-k].mean() * 1e4 if len(x) > 20 else np.nan
    out['gross_raw_bp'] = d[ret].mean() * 1e4
    out['tail_gt5pct_share'] = float((d[ret].abs() > 0.05).mean())
    mm = x.groupby(d.loc[x.index, 'month']).mean()
    out['month_green'] = float((mm > 0).mean()) if len(mm) else np.nan
    out['n_months'] = len(mm)
    yr = x.groupby(d.loc[x.index, 'year']).mean() * 1e4
    out['years'] = ' '.join(f'{y}:{v:+.0f}' for y, v in yr.items())
    return out


def main():
    with_test = '--test' in sys.argv
    ev = clean_events()
    C, V, RC = load_panels()
    log(f'panels {C.shape}; dates {C.index[0]}..{C.index[-1]}')
    d = build(ev, C, V, RC, with_test)
    d.to_parquet(os.path.join(HERE, '1747_scored.parquet' if not with_test else '1747_scored_test.parquet'))
    books = {'B1': ('B1', 'ctl_B1', 'U1'), 'B2': ('B2', 'ctl_B2', 'U2'), 'B3_short': ('B3', 'ctl_B3', 'U3s'),
             'B3_long': ('B3_long', None, 'U3')}
    d['U3s'] = -d.U3; d['ctl_B3_long'] = -d.ctl_B3
    rows = []
    for tier, (lo, hi) in TIERS.items():
        dt = d[(d.dv20 >= lo) & (d.dv20 < hi)]
        for sp, (a, b) in SPLITS.items():
            if sp == 'TEST' and not with_test:
                continue
            ds = dt[(dt.ex_date >= a) & (dt.ex_date <= b)]
            if len(ds) < 20:
                continue
            weeks = (pd.Timestamp(b) - pd.Timestamp(a)).days / 7 * 5 / 7 * 7 / 5 if False else (pd.Timestamp(b) - pd.Timestamp(a)).days / 7
            for bk, (ret, ctl, uni) in books.items():
                if ctl is None:
                    ctl = 'ctl_B3_long'
                r = read(ds, bk, ret, ctl, uni); r.update({'tier': tier, 'split': sp, 'book': bk, 'events_per_week': len(ds) / weeks})
                rows.append(r)
    s = pd.DataFrame(rows)
    cols = ['tier', 'split', 'book', 'n', 'events_per_week', 'ctl_n', 'gross_raw_bp', 'ctl_mean_bp', 't_iid_ctl', 't_day_ctl',
            'mde80_bp_ctl', 'ex_top5_ctl_bp', 'uni_mean_bp', 't_day_uni', 'tail_gt5pct_share', 'month_green', 'n_months', 'years']
    s = s[cols]
    s.to_csv(os.path.join(HERE, '1747_summary.csv' if not with_test else '1747_summary_test.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.float_format', lambda v: f'{v:.2f}')
    print(s.drop(columns=['years']).to_string(index=False))
    for _, r in s[s.book.isin(['B1'])].iterrows():
        print(r.tier, r.split, 'years:', r.years)


if __name__ == '__main__':
    main()
