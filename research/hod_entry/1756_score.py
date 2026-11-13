"""Cell 1,756 -- the GAP FADE as a FILTER on the HOD / ORB longs (PREREG_1756.md, frozen).

Reads the LIVE rule's outcome (HOD B0 net_R per signal; ORB _sized_pnl/375 per entered fill), buckets by the RAW
open-vs-prior-close gap, reports n / mean / day-clustered t / ex-top-5 % per bucket x break hour, then the paired
filter read (drop gap >= 10 %) and a prior-day-range placebo. HOD TEST rows are dropped at load; VAL is read once,
after the TRAIN table is written.
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv  # noqa: E402

OUT = os.path.join(ROOT, 'research/hod_entry')
C48 = os.path.join(ROOT, 'research/signal_search/1748_cache')
C46 = os.path.join(ROOT, 'research/signal_search/1746_cache')
ORB_BOOK = os.path.join(ROOT, 'research/thermo/book_2025_26.csv')
ORB_R_DOLLARS = 375.0
GAP_CUT = 10.0
GAP_EDGES = [-np.inf, 5, 10, 20, np.inf]
GAP_LABELS = ['<5', '5-10', '10-20', '>=20']
HOUR_EDGES = [0, 600, 660, 780, 10_000]          # entry_m = minutes since midnight ET
HOUR_LABELS = ['0930-1000', '1000-1100', '1100-1300', '1300-close']


def log(m):
    """Flush-print progress."""
    print(m, flush=True)


def cl_mean_t(y, d):
    """Mean of y with day-clustered SE; returns (mean, t)."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 3:
        return (float(y.mean()) if n else np.nan), np.nan
    m = y.mean()
    s = pd.Series(y - m).groupby(np.asarray(d)).sum()
    se = np.sqrt((s ** 2).sum()) / n
    return float(m), float(m / se) if se > 0 else np.nan


def ex_top5(y):
    """Mean after removing the top 5 % of values."""
    y = np.sort(np.asarray(y, float))
    if len(y) < 20:
        return np.nan
    return float(y[:len(y) - int(np.ceil(0.05 * len(y)))].mean())


def diff_t(y, d, flag):
    """Day-clustered t of mean(y | ~flag) - mean(y | flag): regression on dummy, CR0 clusters."""
    y = np.asarray(y, float)
    f = np.asarray(flag, float)
    if f.sum() < 5 or (1 - f).sum() < 5:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(y)), f])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    xtx = np.linalg.inv(X.T @ X)
    sc = pd.DataFrame(X * e[:, None]).groupby(np.asarray(d)).sum().to_numpy()
    V = xtx @ (sc.T @ sc) @ xtx
    se = np.sqrt(V[1, 1])
    return float(-b[1]), float(-b[1] / se) if se > 0 else np.nan   # rest minus gap


def load_daily(syms, start='2024-11-15'):
    """Raw o/h/l/c daily bars for syms from 1748 (wins) + 1746 (fills holes); returns gap and prior-day range."""
    parts = []
    for c in (C48, C46):
        p = pq.ParquetFile(os.path.join(c, 'bars_raw.parquet'))
        fr = []
        for g in range(p.metadata.num_row_groups):
            t = p.read_row_group(g, columns=['symbol', 'date', 'o', 'h', 'l', 'c'], use_threads=False).to_pandas()
            fr.append(t[t.symbol.isin(syms) & (t.date >= start)])
            del t
        parts.append(pd.concat(fr))
        log(f'loaded {c}: {len(parts[-1])} rows')
    b = pd.concat(parts).drop_duplicates(['symbol', 'date'], keep='first').sort_values(['symbol', 'date'])
    b['date'] = b.date.astype(str)
    g = b.groupby('symbol')
    b['pc'] = g.c.shift(1)
    b['pdate'] = g.date.shift(1)
    b['rng'] = (b.h - b.l) / b.c
    b['prng'] = b.groupby('symbol').rng.shift(1)
    gap = (pd.to_datetime(b.date) - pd.to_datetime(b.pdate)).dt.days
    b['gap_pct'] = np.where(gap <= 5, (b.o / b.pc - 1) * 100, np.nan)
    b['prng_pct'] = np.where(gap <= 5, b.prng * 100, np.nan)
    return b[['symbol', 'date', 'gap_pct', 'prng_pct']]


def load_hod():
    """HOD signals joined to per-arm B0 net_R and the raw gap; TEST dropped (counted); returns (df, coverage dict)."""
    s = pd.read_parquet(os.path.join(OUT, '../hod_exit_lab/signals.parquet'), columns=['day', 'symbol', 'entry_m', 'price', 'split', 'half', 'wk'])
    assert not s.duplicated(['day', 'symbol']).any()
    ps = pd.read_csv(os.path.join(OUT, '1735_per_signal.csv'), usecols=['day', 'symbol', 'split', 'b0_net_R'])
    n_test = int((s.split == 'TEST').sum())
    s = s[s.split != 'TEST'].drop(columns='split')
    h = s.merge(ps, on=['day', 'symbol'], how='inner')
    n_nan = int(h.b0_net_R.isna().sum())
    h = h[h.b0_net_R.notna()].copy()
    log(f'HOD: signals {len(s)} (TEST dropped {n_test}), per_signal {len(ps)}, joined {len(h)}')
    daily = load_daily(set(h.symbol))
    h = h.merge(daily, left_on=['day', 'symbol'], right_on=['date', 'symbol'], how='left')
    cov = {'test_dropped': n_test, 'nan_R_dropped': n_nan, 'joined': len(h), 'gap_cov': float(h.gap_pct.notna().mean()),
           'extreme_gap_gt200': int((h.gap_pct.abs() > 200).sum())}
    for sp in ('TRAIN', 'VAL'):
        cov[f'gap_cov_{sp}'] = float(h[h.split == sp].gap_pct.notna().mean())
    h['gap_b'] = pd.cut(h.gap_pct, GAP_EDGES, right=False, labels=GAP_LABELS).astype(object).fillna('nogap')
    h['hour_b'] = pd.cut(h.entry_m, HOUR_EDGES, right=False, labels=HOUR_LABELS).astype(str)
    h['big'] = (h.gap_pct >= GAP_CUT).astype(int)
    return h, cov


def table(df, ycol, bcol, scope, book, hours=True):
    """Per-bucket (and bucket x hour) n / mean / t / ex-top-5 % rows for one scope."""
    rows = []
    hs = [('ALL', slice(None))] + ([(x, None) for x in HOUR_LABELS] if hours else [])
    for hname, _ in hs:
        sub = df if hname == 'ALL' else df[df.hour_b == hname]
        for bk in list(GAP_LABELS) + ['nogap'] if bcol == 'gap_b' else sorted(sub[bcol].dropna().unique()):
            x = sub[sub[bcol] == bk]
            m, t = cl_mean_t(x[ycol], x.day) if len(x) else (np.nan, np.nan)
            rows.append(dict(book=book, scope=scope, bucket=bk, hour=hname, n=len(x), mean_R=m, t=t,
                             ex_top5=ex_top5(x[ycol]) if len(x) else np.nan))
    return pd.DataFrame(rows)


def filter_read(df, ycol, scope, book, weekcol, flagcol='big'):
    """Paired-on-weeks filter read: drop flagged arms; dR/wk, fills/wk kept/lost, green-week share, ex-top-5 % total."""
    wk = df[weekcol]
    nw = wk.nunique()
    keep = df[df[flagcol] == 0]
    thr = np.sort(df[ycol].to_numpy())[::-1][:int(np.ceil(0.05 * len(df)))].min()
    ex_all = df[df[ycol] < thr][ycol].sum() / nw
    ex_keep = keep[keep[ycol] < thr][ycol].sum() / nw
    wu = df.groupby(weekcol)[ycol].sum().reindex(sorted(wk.unique()), fill_value=0)
    wf = keep.groupby(weekcol)[ycol].sum().reindex(wu.index, fill_value=0)
    dw = (wf - wu)
    dm, dt = cl_mean_t(dw.to_numpy(), np.arange(len(dw)))
    return dict(book=book, scope=scope, weeks=nw, fills_wk_all=len(df) / nw, fills_wk_kept=len(keep) / nw,
                fills_wk_lost=(len(df) - len(keep)) / nw, R_wk_all=wu.sum() / nw, R_wk_kept=wf.sum() / nw,
                dR_wk=dm, dR_wk_t=dt, green_all=float((wu > 0).mean()), green_kept=float((wf > 0).mean()),
                extop5_R_wk_all=ex_all, extop5_R_wk_kept=ex_keep, n_flagged=int(df[flagcol].sum()))


def placebo_flag(df, ref):
    """Range-bucket flag with the same TRAIN-defined share as gap>=10 % (top-quantile of prior-day range)."""
    share = float((ref.gap_pct >= GAP_CUT).mean())
    cut = ref.prng_pct.quantile(1 - share)
    return cut, share


def main():
    """Run: TRAIN tables -> write -> VAL (single read) -> ORB -> write."""
    h, cov = load_hod()
    log(f'coverage {cov}')
    # sanity: computed raw gap vs ORB's own gap_pct (validates the join/scale)
    o = read_orb_csv(ORB_BOOK)
    o = o[o.entered == 1].copy()
    o['R'] = o._sized_pnl / ORB_R_DOLLARS
    od = load_daily(set(o.symbol), start='2024-11-15')
    om = o.merge(od, on=['symbol', 'date'], how='left', suffixes=('', '_c'))
    ok = om.gap_pct_c.notna()
    cov['orb_gap_corr_vs_column'] = float(np.corrcoef(om.gap_pct[ok], om.gap_pct_c[ok])[0, 1])
    cov['orb_gap_within1pt'] = float((abs(om.gap_pct[ok] - om.gap_pct_c[ok]) < 1).mean())
    cov['orb_computed_cov'] = float(ok.mean())
    log(f'ORB gap sanity {cov}')

    cut, share = placebo_flag(h[h.split == 'TRAIN'], h[h.split == 'TRAIN'])
    h['rb'] = np.where(h.prng_pct.isna(), 'norng', np.where(h.prng_pct >= cut, 'rng_top', 'rng_rest'))
    h['rbig'] = (h.prng_pct >= cut).astype(int)
    tr = h[h.split == 'TRAIN']
    scopes = {'TRAIN_H1': tr[tr.half == 'H1'], 'TRAIN_H2': tr[tr.half == 'H2'], 'TRAIN': tr}
    tabs, filt, diffs = [], [], []

    def run_scope(name, d):
        """All reads for one scope; appends to tabs / filt / diffs."""
        tabs.append(table(d, 'b0_net_R', 'gap_b', name, 'HOD'))
        tabs.append(table(d, 'b0_net_R', 'rb', name + '_PLACEBO_RANGE', 'HOD', hours=False))
        g = d[d.gap_pct.notna()]
        for tag, col in (('gap>=10', 'big'), ('RANGEplacebo', 'rbig')):
            dd, tt = diff_t(g.b0_net_R, g.day, g[col])
            diffs.append(dict(scope=name, test=tag, rest_minus_flag_R=dd, day_t=tt, n_flag=int(g[col].sum()), n=len(g)))
        filt.append(filter_read(d, 'b0_net_R', name, 'HOD', 'wk'))
        d2 = d.copy()
        d2['big'] = d2.rbig
        filt.append(filter_read(d2, 'b0_net_R', name + '_PLACEBO_RANGE', 'HOD', 'wk'))

    for k, d in scopes.items():
        run_scope(k, d)
    pd.concat(tabs).to_csv(os.path.join(OUT, '1756_buckets_TRAIN_only.csv'), index=False)   # TRAIN table written before VAL
    log('TRAIN table written; reading VAL once')
    run_scope('VAL', h[h.split == 'VAL'])

    # ORB by half-year (no TRAIN/VAL split), gap column as-is
    o['gap_b'] = pd.cut(o.gap_pct, GAP_EDGES, right=False, labels=GAP_LABELS).astype(str)
    o['hour_b'] = 'ALL'
    o['big'] = (o.gap_pct >= GAP_CUT).astype(int)
    o['day'] = o.date
    o['hy'] = o.date.str[:4] + np.where(o.date.str[5:7].astype(int) <= 6, 'H1', 'H2')
    o['wk'] = pd.to_datetime(o.date).dt.strftime('%G-%V')
    for hy, d in list(o.groupby('hy')) + [('ALL', o)]:
        tabs.append(table(d, 'R', 'gap_b', f'ORB_{hy}', 'ORB', hours=False))
        dd, tt = diff_t(d.R, d.day, d.big)
        diffs.append(dict(scope=f'ORB_{hy}', test='gap>=10', rest_minus_flag_R=dd, day_t=tt, n_flag=int(d.big.sum()), n=len(d)))
        filt.append(filter_read(d, 'R', f'ORB_{hy}', 'ORB', 'wk'))

    bt = pd.concat(tabs)
    bt.to_csv(os.path.join(OUT, '1756_buckets.csv'), index=False)
    os.remove(os.path.join(OUT, '1756_buckets_TRAIN_only.csv'))
    ft = pd.DataFrame(filt)
    ft.to_csv(os.path.join(OUT, '1756_filter.csv'), index=False)
    dt_ = pd.DataFrame(diffs)
    dt_.to_csv(os.path.join(OUT, '1756_diffs.csv'), index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    log('COVERAGE ' + str(cov) + f' placebo cut {cut:.2f}% share {share:.3f}')
    log('\n' + dt_.round(3).to_string())
    log('\n' + ft.round(3).to_string())
    v = bt[(bt.scope.isin(['VAL', 'ORB_ALL'])) & (bt.hour == 'ALL')]
    log('\n' + v.round(3).to_string())
    log('\nBREAK HOUR x gap>=10 (VAL, TRAIN)')
    log(bt[(bt.scope.isin(['VAL', 'TRAIN'])) & (bt.hour != 'ALL') & (bt.book == 'HOD') & (~bt.scope.str.contains('PLACEBO'))].round(3).to_string())
    log('\nORB halves\n' + bt[(bt.book == 'ORB')].round(3).to_string())


if __name__ == '__main__':
    main()
