"""Cell 1,741 scorer (PREREG_1741): option-flow before the break as a HOD-break filter. TRAIN+VAL only (TEST sealed unless --test and primary passed).
Joins 1741_flow.csv to signals.parquet (split/half) and b0_trades.csv net_R on (day, symbol). Features strictly pre-break (fetch reads bars up to entry_m-2).
Covered-with-print = n_prints > 0.  F1 call share (vol), F2 relative activity (pre-break contracts/min over prior_day_vol/390; minutes = entry_m-571),
F4 call share (notional).  Tercile cuts from TRAIN covered set; lo = f<=c1, hi = f>=c2 (c2>c1) else f>c2 (ties).  Day-clustered t of a spread =
mean_hi-mean_lo with cluster-by-day SE of per-observation residual contributions.  ex_top5 of spread = drop top 5 % net_R in the hi tercile.
Permutation: 300 within-day shuffles of the feature among covered rows, fixed TRAIN cuts, percentile of observed spread.
Usage: nice -n 19 python3 1741_score.py [--test]"""
import sys, math
import numpy as np, pandas as pd
R = '/home/ec2-user/onemil/research'
H = f'{R}/hod_entry'
TEST = '--test' in sys.argv

def log(m): print(m, flush=True)

def load(include_test):
    """Flow joined to signals and B0 net_R; returns frame and join-loss report."""
    fl = pd.read_csv(f'{H}/1741_flow.csv', dtype={'symbol': str, 'day': str})
    sg = pd.read_parquet(f'{R}/hod_exit_lab/signals.parquet')[['day', 'symbol', 'entry_m', 'split', 'half']]
    nd = sg.duplicated(['day', 'symbol']).sum()
    sg = sg.drop_duplicates(['day', 'symbol'])
    if include_test: sg = sg[sg.split == 'TEST']
    else: sg = sg[sg.split.isin(['TRAIN', 'VAL'])]
    b0 = pd.read_csv(f'{R}/hod_exit_lab/b0_trades.csv').drop_duplicates(['day', 'symbol'])[['day', 'symbol', 'net_R']]
    d = sg.merge(fl, on=['day', 'symbol'], how='left', indicator='fm').merge(b0, on=['day', 'symbol'], how='left')
    log(f'signals dup (day,symbol) dropped={nd}; signals={len(sg)} no_flow_row={(d.fm != "both").sum()} no_b0={d.net_R.isna().sum()}')
    d = d[d.fm == 'both'].drop(columns='fm')
    d['covered'] = d.covered.astype(str).str.lower().eq('true')
    d['has_print'] = d.covered & (d.n_prints.fillna(0) > 0)
    tv = d.call_vol + d.put_vol
    d['F1'] = np.where(tv > 0, d.call_vol / tv.replace(0, np.nan), np.nan)
    el = (d.entry_m - 571).clip(lower=1)
    d['F2'] = np.where(d.prior_day_vol > 0, (tv / el) / (d.prior_day_vol / 390.0), np.nan)
    tn = d.call_notional + d.put_notional
    d['F4'] = np.where(tn > 0, d.call_notional / tn.replace(0, np.nan), np.nan)
    return d

def dayt(x, col='net_R'):
    """Mean and day-clustered / iid t of a trade frame."""
    n = len(x); r = x[col].values; m = r.mean()
    se = math.sqrt((x.assign(_r=r - m).groupby('day')._r.sum() ** 2).sum()) / n
    iid = m / (r.std(ddof=1) / math.sqrt(n)) if n > 2 and r.std(ddof=1) > 0 else np.nan
    return m, (m / se if se > 0 else np.nan), iid

def spread(x, hi, lo):
    """hi-lo mean spread, day-clustered t, iid t, ex-top5 spread for boolean masks."""
    a, b = x[hi], x[lo]
    if len(a) < 3 or len(b) < 3: return dict(spread=np.nan, day_t=np.nan, iid_t=np.nan, ex5=np.nan)
    ma, mb = a.net_R.mean(), b.net_R.mean()
    ca = a.assign(_c=(a.net_R - ma) / len(a)).groupby('day')._c.sum(); cb = b.assign(_c=(b.net_R - mb) / len(b)).groupby('day')._c.sum()
    c = ca.sub(cb, fill_value=0)
    se = math.sqrt((c ** 2).sum()); sp = ma - mb
    iid = sp / math.sqrt(a.net_R.var(ddof=1) / len(a) + b.net_R.var(ddof=1) / len(b))
    k = int(math.ceil(0.05 * len(a))); ex = np.sort(a.net_R.values)[:-k].mean() - mb if len(a) > k else np.nan
    return dict(spread=sp, day_t=sp / se if se > 0 else np.nan, iid_t=iid, ex5=ex)

def masks(f, cuts):
    """(hi, lo) boolean masks from tercile cuts."""
    c1, c2 = cuts
    hi = f >= c2 if c2 > c1 else f > c2
    return hi, f <= c1

def nweeks(g): return pd.to_datetime(g.day).dt.strftime('%G-%V').nunique()

def perm_pct(cov, feat, cuts, nperm=300, seed=741):
    """Percentile of observed hi-lo spread vs within-day shuffles of the feature."""
    rng = np.random.default_rng(seed)
    f0 = cov[feat].values; r = cov.net_R.values
    def sp(f):
        hi, lo = masks(f, cuts)
        return r[hi].mean() - r[lo].mean() if hi.sum() > 2 and lo.sum() > 2 else np.nan
    obs = sp(f0)
    idx = [ix for ix in cov.groupby('day').indices.values()]
    null = np.empty(nperm)
    for p in range(nperm):
        f = f0.copy()
        for ix in idx:
            if len(ix) > 1: f[ix] = f0[rng.permutation(ix)]
        null[p] = sp(f)
    return obs, float((null < obs).mean() * 100), float(np.nanmean(null))

def groups(d):
    """Scoring groups."""
    g = {}
    if (d.split == 'TRAIN').any():
        g['TRAIN_H1'] = d[(d.split == 'TRAIN') & (d.half == 'H1')]; g['TRAIN_H2'] = d[(d.split == 'TRAIN') & (d.half == 'H2')]; g['TRAIN'] = d[d.split == 'TRAIN']
    for s in ('VAL', 'TEST'):
        if (d.split == s).any(): g[s] = d[d.split == s]
    return g

def main():
    d = load(TEST)
    d = d[d.net_R.notna()]
    log(f'rows scored={len(d)} (b0 net_R present)')
    gs = groups(d); out = []
    # (1) availability rail
    log('--- RAIL (coverage = >=1 pre-break option print)')
    for gn, g in gs.items():
        cov = g.has_print.mean(); w = g[g.net_R > 0]; l = g[g.net_R <= 0]
        cw, cl = w.has_print.mean(), l.has_print.mean()
        log(f'{gn:9s} n={len(g)} chain_listed={g.covered.mean():.3f} coverage={cov:.3f} win_cov={cw:.3f}(n{len(w)}) lose_cov={cl:.3f}(n{len(l)}) gap_pp={(cw-cl)*100:.1f}')
        out.append(dict(table='rail', group=gn, n=len(g), chain_listed=g.covered.mean(), coverage=cov, win_cov=cw, lose_cov=cl, gap_pp=(cw - cl) * 100))
    # (2) F3
    log('--- F3 covered-with-print vs not (B0 net R)')
    for gn, g in gs.items():
        a, b = g[g.has_print], g[~g.has_print]
        ma, ta, _ = dayt(a); mb, tb, _ = dayt(b)
        sp = spread(g, g.has_print.values, ~g.has_print.values)
        log(f'{gn:9s} cov n={len(a)} mean={ma:.3f} dayt={ta:.2f} | notcov n={len(b)} mean={mb:.3f} dayt={tb:.2f} | diff={sp["spread"]:.3f} dayt={sp["day_t"]:.2f}')
        out.append(dict(table='F3', group=gn, n_cov=len(a), mean_cov=ma, t_cov=ta, n_not=len(b), mean_not=mb, t_not=tb, diff=sp['spread'], diff_day_t=sp['day_t']))
    # (3) terciles; cuts from TRAIN covered set (when scoring TEST, cuts still from TRAIN: reload)
    tr = load(False); tr = tr[(tr.split == 'TRAIN') & tr.has_print]
    cuts = {f: tuple(tr[f].dropna().quantile([1 / 3, 2 / 3]).values) for f in ('F1', 'F2', 'F4')}
    log(f'--- cuts from TRAIN covered n={len(tr)}: {cuts}')
    res = {}
    for f in ('F1', 'F2', 'F4'):
        for gn, g in gs.items():
            c = g[g.has_print & g[f].notna()]
            hi, lo = masks(c[f], cuts[f]); mid = ~hi & ~lo
            s = spread(c, hi.values, lo.values); nw = nweeks(g)
            row = dict(table='tercile', feature=f, group=gn, n_cov=len(c), n_lo=int(lo.sum()), n_mid=int(mid.sum()), n_hi=int(hi.sum()),
                       mean_lo=c[lo].net_R.mean(), mean_mid=c[mid].net_R.mean(), mean_hi=c[hi].net_R.mean(), top_fills_wk=hi.sum() / nw, **s)
            res[(f, gn)] = row; out.append(row)
            log(f'{f} {gn:9s} n={len(c)} lo/mid/hi={int(lo.sum())}/{int(mid.sum())}/{int(hi.sum())} means={row["mean_lo"]:.3f}/{row["mean_mid"]:.3f}/{row["mean_hi"]:.3f} '
                f'spread={s["spread"]:.3f} dayt={s["day_t"]:.2f} iidt={s["iid_t"]:.2f} ex5={s["ex5"]:.3f} top_fills/wk={hi.sum()/nw:.2f}')
    # (4) permutation, primary F1 on TRAIN and VAL (and TEST if scoring it)
    log('--- permutation F1 (300 within-day shuffles)')
    for gn in [k for k in gs if k in ('TRAIN', 'VAL', 'TEST')]:
        c = gs[gn][gs[gn].has_print & gs[gn].F1.notna()].reset_index(drop=True)
        obs, pct, nm = perm_pct(c, 'F1', cuts['F1'])
        log(f'{gn} observed spread={obs:.3f} null_mean={nm:.3f} percentile={pct:.1f}')
        res[('perm', gn)] = pct; out.append(dict(table='perm', feature='F1', group=gn, observed=obs, null_mean=nm, percentile=pct))
    # (5) pass bar (primary F1)
    def g_(k): return res.get(('F1', k), {})
    hv = g_('VAL'); ht = g_('TRAIN')
    tests = {
        'spread>=0.15 & day_t>=2 TRAIN': ht.get('spread', -9) >= 0.15 and ht.get('day_t', -9) >= 2,
        'spread>=0.15 & day_t>=2 VAL': hv.get('spread', -9) >= 0.15 and hv.get('day_t', -9) >= 2,
        'same sign both TRAIN halves': g_('TRAIN_H1').get('spread', 0) > 0 and g_('TRAIN_H2').get('spread', 0) > 0,
        'top mean>=0 VAL': hv.get('mean_hi', -9) >= 0,
        'ex5 spread>=0 (TRAIN & VAL)': ht.get('ex5', -9) >= 0 and hv.get('ex5', -9) >= 0,
        'perm pct>=95 VAL': res.get(('perm', 'VAL'), 0) >= 95,
        'top fills/wk>=3 (VAL & TRAIN)': hv.get('top_fills_wk', 0) >= 3 and ht.get('top_fills_wk', 0) >= 3,
    }
    if not TEST:
        for k, v in tests.items(): log(f'BAR {k}: {v}')
        log('PRIMARY ' + ('PASS' if all(tests.values()) else 'FAIL'))
    pd.DataFrame(out).to_csv(f'{H}/1741_summary{"_test" if TEST else ""}.csv', index=False)
    log('DONE')
main()
