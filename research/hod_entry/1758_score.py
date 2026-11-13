"""Cell 1,758 (PREREG_1758.md, frozen): F3 leveraged-ETF rebalance into the close; I2 calendar cuts on the HOD book.

TEST rows are dropped at load and counted.  TRAIN rows of every read are written to the CSV before VAL is computed.
Run: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1758_score.py
"""
import json
import os
import time

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = '/home/ec2-user/onemil/research'
EL = f'{BASE}/hod_exit_lab'
OUT = f'{BASE}/hod_entry'
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
M_SIG, M_EXIT = 900, 955          # 15:00 and 15:55 ET minute-of-day (m = hh*60+mm, as in 1757_score.py)
LEG_BP = 1.0                      # per leg; round trip = 2 bp
KS = (0.005, 0.010, 0.015)
K_PRE = 0.010

# Hard-coded FOMC scheduled-meeting second days (Fed public calendar, from memory; emergency meetings excluded).
# 2016-2023 as the PREREG asks; 2024-2026 added because the data on disk are 2025-01..2026-05 (2016-2023 never match).
FOMC = """2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14
2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13
2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19
2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11
2020-01-29 2020-03-18 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28 2026-12-09""".split()


def log(*a):
    """Timestamped progress line, flushed."""
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def tstat(x):
    """(mean, t) of one-observation-per-day values."""
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return np.nan, np.nan
    sd = x.std(ddof=1)
    return x.mean(), (x.mean() / (sd / np.sqrt(len(x))) if sd > 0 else np.nan)


def excess_t(y, ind, days):
    """Difference mean(y | ind) - mean(y | ~ind) and its day-clustered t (OLS sandwich on [1, ind])."""
    y, ind = np.asarray(y, float), np.asarray(ind, float)
    n, n1 = len(y), int(ind.sum())
    if n1 < 5 or n - n1 < 5:
        return np.nan, np.nan
    X = np.column_stack([np.ones(n), ind])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ (X.T @ y)
    u = y - X @ b
    codes = pd.factorize(np.asarray(days))[0]
    S = np.zeros((codes.max() + 1, 2))
    np.add.at(S, codes, X * u[:, None])
    V = XtXi @ (S.T @ S) @ XtXi
    return b[1], (b[1] / np.sqrt(V[1, 1]) if V[1, 1] > 0 else np.nan)


def load_signals():
    """Non-TEST signals (day, symbol, split, half) with the TEST count."""
    sig = pd.read_parquet(f'{EL}/signals.parquet', columns=['day', 'symbol', 'split', 'half'])
    n_test = int((sig.split == 'TEST').sum())
    return sig[sig.split != 'TEST'].reset_index(drop=True), n_test, sig


def read_spy(keep_days, test_days):
    """SPY close per (day, m) from paths.parquet.spy_close on non-TEST days only; counts TEST-day rows dropped."""
    pf = pq.ParquetFile(f'{EL}/paths.parquet')
    parts, n_drop = [], 0
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=['day', 'm', 'spy_close']).to_pandas()
        t = t[t.spy_close.notna()]
        k = t.day.isin(keep_days)
        n_drop += int(t.day.isin(test_days).sum())
        parts.append(t[k].drop_duplicates(['day', 'm']))
        log(f'row group {i}: kept {int(k.sum()):,} spy rows')
    spy = pd.concat(parts).drop_duplicates(['day', 'm']).astype({'m': 'int32'})
    return spy, n_drop


def day_frame(spy, dsplit):
    """One row per day: open (first spy bar, minute m0), SPY 15:00, SPY 15:55 closes, r and the 15:00->15:55 move in bp."""
    rows = []
    for day, g in spy.groupby('day'):
        g = g.sort_values('m')
        a, b = g[g.m == M_SIG], g[g.m == M_EXIT]
        if not len(a) or not len(b):
            continue
        rows.append({'day': day, 'm0': int(g.m.iloc[0]), 's0': float(g.spy_close.iloc[0]),
                     's1500': float(a.spy_close.iloc[0]), 's1555': float(b.spy_close.iloc[0])})
    d = pd.DataFrame(rows).merge(dsplit, on='day')
    d['r'] = d.s1500 / d.s0 - 1.0
    d['move_bp'] = (d.s1555 / d.s1500 - 1.0) * 1e4
    return d


def parts_of(f):
    """Split masks: TRAIN, the two TRAIN halves, VAL."""
    return {'TRAIN': f.split == 'TRAIN', 'TRAIN_H1': (f.split == 'TRAIN') & (f.half == 'H1'),
            'TRAIN_H2': (f.split == 'TRAIN') & (f.half == 'H2'), 'VAL': f.split == 'VAL'}


def f3_rows(d, names):
    """F3 table rows for the given part names: k x (trade |r|>=k, comparator |r|<k) x (all, up, down)."""
    pm, rows = parts_of(d), []
    for k in KS:
        for kind, sel in (('trade', d.r.abs() >= k), ('comparator', d.r.abs() < k)):
            for side, ss in (('all', d.r == d.r), ('up', d.r > 0), ('down', d.r < 0)):
                for p in names:
                    f = d[pm[p] & sel & ss]
                    pnl = np.sign(f.r) * f.move_bp - 2 * LEG_BP
                    mu, t = tstat(pnl)
                    rows.append({'k': k, 'kind': kind, 'side': side, 'part': p, 'n': len(f),
                                 'n_per_yr': len(f) / max(pm[p].sum() and d[pm[p]].day.nunique() / 252.0, 1e-9),
                                 'mean_bp': mu, 't': t, 'hit': float((pnl > 0).mean()) if len(f) else np.nan})
    return rows


def run_f3(sig, spy, n_test_days):
    """F3: TRAIN rows to CSV first, then VAL once; coverage per split/year; verdict at k = 1 %."""
    dsplit = sig.groupby('day').agg(split=('split', 'first'), half=('half', 'first')).reset_index()
    d = day_frame(spy, dsplit)
    firsts = spy.groupby('day').m.min()
    bd = pd.bdate_range(sig.day.min(), sig.day.max())
    cov = {'signal_days': int(len(dsplit)), 'spy_days_with_1500_and_1555': int(len(d)),
           'business_days_in_range': int(len(bd)), 'median_first_minute': int(firsts.median()),
           'first_minute_min_max': [int(firsts.min()), int(firsts.max())],
           'cov_vs_signal_days': len(d) / len(dsplit), 'cov_vs_business_days': len(d) / len(bd)}
    for s in ('TRAIN', 'TRAIN_H1', 'TRAIN_H2', 'VAL'):
        cov[s] = [int(parts_of(dsplit)[s].sum()), int(parts_of(d)[s].sum())]
    cov['by_year'] = {y: [int((dsplit.day.str[:4] == y).sum()), int((d.day.str[:4] == y).sum())] for y in sorted(dsplit.day.str[:4].unique())}
    log('F3 coverage', json.dumps(cov))
    tr = pd.DataFrame(f3_rows(d, ['TRAIN', 'TRAIN_H1', 'TRAIN_H2']))
    tr.to_csv(f'{OUT}/1758_f3.csv', index=False)
    log('F3 TRAIN table written')
    val = pd.DataFrame(f3_rows(d, ['VAL']))
    full = pd.concat([tr, val], ignore_index=True)
    full.to_csv(f'{OUT}/1758_f3.csv', index=False)
    g = lambda kind, side, p, k=K_PRE: full[(full.k == k) & (full.kind == kind) & (full.side == side) & (full.part == p)].iloc[0]
    v, c = g('trade', 'all', 'VAL'), g('comparator', 'all', 'VAL')
    h1, h2 = g('trade', 'all', 'TRAIN_H1'), g('trade', 'all', 'TRAIN_H2')
    kill = {'val_mean_le_5bp_or_t_lt_2': bool(v.mean_bp <= 5 or v.t < 2 or np.isnan(v.t)),
            'halves_opposite': bool(np.sign(h1.mean_bp) != np.sign(h2.mean_bp)),
            'comparator_not_below_trade': bool(c.mean_bp >= v.mean_bp)}
    ver = {'coverage': cov, 'n_test_days_rows_dropped': n_test_days, 'kill_flags_k1pct': kill,
           'verdict': 'KILL' if any(kill.values()) else 'SURVIVE'}
    log('F3 verdict', ver['verdict'], json.dumps(kill))
    return full, ver


def is_month_end(day):
    """True for the last two business days of the calendar month (weekday calendar, holidays not removed)."""
    ts = pd.Timestamp(day)
    last = ts + pd.offsets.BMonthEnd(0)
    return ts >= last - pd.offsets.BDay(1)


def is_opex(day):
    """True for the 3rd Friday of the month."""
    ts = pd.Timestamp(day)
    return ts.dayofweek == 4 and 15 <= ts.day <= 21


def i2_cuts(df):
    """Indicator columns for every cut level (5 DOW + 4 binary cuts)."""
    ts = pd.to_datetime(df.day)
    c = {f'dow_{n}': (ts.dt.dayofweek == i).to_numpy() for i, n in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])}
    c['month_end_last2'] = df.day.map(is_month_end).to_numpy()
    c['opex_friday'] = df.day.map(is_opex).to_numpy()
    c['fomc_day'] = df.day.isin(set(FOMC)).to_numpy()
    c['first_half_month'] = (ts.dt.day <= 15).to_numpy()
    return c


def i2_rows(df, cuts, names):
    """Rows per cut level and part: n, level mean net R, excess vs rest and its day-clustered t."""
    pm, rows = parts_of(df), []
    for cut, ind in cuts.items():
        for p in names:
            m = pm[p].to_numpy()
            y, i, dd = df.b0_net_R.to_numpy()[m], ind[m], df.day.to_numpy()[m]
            ex, t = excess_t(y, i, dd)
            rows.append({'cut': cut, 'part': p, 'n': int(i.sum()), 'n_days': int(pd.Series(dd[i]).nunique()),
                         'mean_net_R': float(y[i].mean()) if i.any() else np.nan, 'rest_mean_net_R': float(y[~i].mean()),
                         'excess_R': ex, 'cl_t': t})
    return rows


def run_i2(sig):
    """I2: B0 net R per arm joined to the non-TEST signals; TRAIN rows first, VAL once; structure test."""
    ps = pd.read_csv(f'{OUT}/1735_per_signal.csv', usecols=['day', 'symbol', 'split', 'half', 'b0_net_R'])
    n_test = int((ps.split == 'TEST').sum())
    ps = ps[ps.split != 'TEST'].dropna(subset=['b0_net_R']).reset_index(drop=True)
    log(f'I2 arms: TRAIN {int((ps.split == "TRAIN").sum())}, VAL {int((ps.split == "VAL").sum())}; TEST dropped {n_test}')
    cuts = i2_cuts(ps)
    tr = pd.DataFrame(i2_rows(ps, cuts, ['TRAIN', 'TRAIN_H1', 'TRAIN_H2']))
    tr.to_csv(f'{OUT}/1758_i2.csv', index=False)
    log('I2 TRAIN table written')
    full = pd.concat([tr, pd.DataFrame(i2_rows(ps, cuts, ['VAL']))], ignore_index=True)
    full.to_csv(f'{OUT}/1758_i2.csv', index=False)
    passes = []
    for cut in cuts:
        q = full[full.cut == cut].set_index('part')
        s = [np.sign(q.loc[p, 'excess_R']) for p in ('TRAIN_H1', 'TRAIN_H2', 'VAL')]
        if s[0] == s[1] == s[2] != 0 and abs(q.loc['VAL', 'cl_t']) >= 2:
            passes.append(cut)
    log('I2 cuts passing the structure test:', passes or 'none (noise)')
    return full, passes, n_test


def main():
    """Run F3 then I2; write the CSVs and a verdict JSON."""
    sig, n_test_sig, sig_all = load_signals()
    test_days = set(sig_all[sig_all.split == 'TEST'].day) - set(sig.day)
    log(f'signals non-TEST {len(sig)}; TEST signals dropped {n_test_sig}; TEST-only days {len(test_days)}')
    spy, n_drop = read_spy(set(sig.day), test_days)
    f3, ver = run_f3(sig, spy, n_drop)
    i2, passes, n_test_i2 = run_i2(sig)
    ver.update({'n_test_signals_dropped': n_test_sig, 'n_test_arms_dropped_i2': n_test_i2, 'i2_passing_cuts': passes})
    json.dump(ver, open(f'{OUT}/1758_verdicts.json', 'w'), indent=1, default=str)
    log('done')


if __name__ == '__main__':
    main()
