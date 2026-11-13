"""Cell 1,771 (PREREG_1771.md): stop slip vs distance of the stop to round numbers. Read 1 only; read 2
is VOID when the cache holds no tape window (checked here, written to 1771_counterfactual.csv)."""
import glob, gzip, os, pickle, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cell_1443 as c            # noqa: E402  (load_base_fills, collect, CACHE_DIR)


def cache_inspect():
    """Union of record keys across every cache file (what the cache actually holds)."""
    keys, n = set(), 0
    for f in sorted(glob.glob(os.path.join(c.CACHE_DIR, '*.pkl.gz'))):
        with gzip.open(f, 'rb') as fh:
            d = pickle.load(fh)
        for v in d.values():
            keys |= set(v.keys()); n += 1
    return keys, n


def signed_dist_cents(stop, grid_c):
    """Signed distance (cents) of stop to nearest multiple of grid_c cents; + = stop above the round number."""
    c_ = np.round(stop * 100.0, 4)
    return c_ - np.round(c_ / grid_c) * grid_c


def bucket(d, grid_c):
    """PREREG buckets: above 0-2 / above 2-10 / >10 away / below 0-2 / below 2-10 (cents)."""
    out = np.full(len(d), '', dtype=object)
    out[(d >= 0) & (d <= 2)] = 'above_0-2'
    out[(d > 2) & (d <= 10)] = 'above_2-10'
    out[(d < 0) & (d >= -2)] = 'below_0-2'
    out[(d < -2) & (d >= -10)] = 'below_2-10'
    out[np.abs(d) > 10] = 'gt10'
    return out


def cluster_t(y, g, days):
    """Mean diff (g=1 minus g=0) and cluster-robust (by day) t from the sandwich OLS on [1, g]."""
    X = np.column_stack([np.ones(len(y)), g.astype(float)])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    for d in np.unique(days):
        m = days == d
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    V = bread @ meat @ bread
    return beta[1], beta[1] / np.sqrt(V[1, 1]) if V[1, 1] > 0 else np.nan


def main():
    keys, nrec = cache_inspect()
    print('cache record keys:', sorted(keys), 'records', nrec, flush=True)
    tape_stored = bool(keys & {'tape', 'trades', 'window', 'prints', 't', 'ticks'})
    f = c.collect(c.load_base_fills())
    f = f[f.why.isin(['stop', 'stop_bar']) & f.measured & np.isfinite(f.slip_bps)].copy()
    print('measured stops by holdout:', f.groupby('holdout').size().to_dict(), flush=True)
    rows, prim = [], {}
    for name, g in [('dollar', 100), ('half', 50), ('dime', 10)]:
        f['d'] = signed_dist_cents(f.stop.values, g)
        f['b'] = bucket(f.d.values, g)
        for ho, h in f.groupby('holdout'):
            for b in ['above_0-2', 'above_2-10', 'gt10', 'below_0-2', 'below_2-10']:
                s = h[h.b == b].slip_bps
                rows.append(dict(grid=name, holdout=ho, bucket=b, n=len(s), mean=s.mean(), median=s.median(),
                                 p90=s.quantile(.9) if len(s) else np.nan, n_days=h[h.b == b].day.nunique()))
            sub = h[h.b.isin(['above_0-2', 'gt10'])]
            if (sub.b == 'above_0-2').sum() > 1 and (sub.b == 'gt10').sum() > 1:
                diff, t = cluster_t(sub.slip_bps.values, (sub.b == 'above_0-2').values, sub.day.values)
            else:
                diff, t = np.nan, np.nan
            rows.append(dict(grid=name, holdout=ho, bucket='CONTRAST_above0-2_minus_gt10', n=len(sub),
                             mean=diff, median=t, p90=np.nan, n_days=sub.day.nunique()))
    # tier / stop-distance / minute context (descriptive, VAL+TRAIN pooled by holdout)
    f['tier'] = pd.cut(f.stop, [0, 5, 20, 1e9], labels=['1-5', '5-20', '>20'])
    f['stop_bp'] = (f.fill - f.stop) / f.fill * 1e4
    for ho, h in f.groupby('holdout'):
        for t_, s in h.groupby('tier', observed=True).slip_bps:
            rows.append(dict(grid='tier', holdout=ho, bucket=str(t_), n=len(s), mean=s.mean(), median=s.median(), p90=s.quantile(.9)))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, '1771_buckets.csv'), index=False)
    print(pd.DataFrame(rows).round(2).to_string(), flush=True)
    with open(os.path.join(HERE, '1771_counterfactual.csv'), 'w') as fh:
        fh.write('VOID: cache records hold only %s (no tape window); read 2 cannot run without a network re-fetch\n' % sorted(keys)
                 if not tape_stored else 'TAPE STORED - implement\n')
    print('tape_stored', tape_stored)


if __name__ == '__main__':
    main()
