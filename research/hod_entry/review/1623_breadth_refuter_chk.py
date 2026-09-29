#!/usr/bin/env python3
"""Adversarial refuter checks for cell 1,624 (break breadth, PREREG_1623.md).

Re-derives the gate from the builder's own outputs (cell_1624_fills.csv, cell_1624_arm_events_cache.csv)
and attacks it on six lenses:
  1. window  - the builder counts m_hi in [fill_min-30, fill_min) with a FRACTIONAL fill_min, i.e. it
               includes the fill's own minute (same-minute events of other names can post-date the fill);
               the spec says 'before f's fill MINUTE' -> spec window m_hi in [floor-30, floor-1];
  2. statuses - which causal_arming_causal.csv statuses the event cache covers; TEST days absent;
  3. hour adjustment - kept share / outcome by 15-min bucket; 15-min-bucket cuts; within-holdout ranks;
               distinct-name breadth (a choppy name re-crossing is depth, not breadth);
  4. placebo - PREREG-literal OTHER-holdout day swap (seed 1623 + 199 more), a 500-draw within-hour
               permutation distribution, day-block bootstrap of kept-minus-dropped;
  5. tails and day concentration (thrust days);
  6. cache-only share.
Read-only on every input; prints to stdout only.
"""
import numpy as np
import pandas as pd

HD = '/home/ec2-user/onemil/research/hod_entry'


def p(*a):
    """Print with flush so progress is visible."""
    print(*a, flush=True)


def day_t(x, days):
    """Day-clustered (CR0) t-statistic of the mean of x."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float('nan')
    m = x.mean()
    s = pd.Series(x - m).groupby(np.asarray(days)).sum().to_numpy()
    se = np.sqrt((s ** 2).sum()) / n
    return m / se if se > 0 else float('nan')


def ex_top(x, q):
    """Mean after dropping the top q share of values."""
    x = np.sort(np.asarray(x, float))
    k = int(round(len(x) * q))
    return x[:len(x) - k].mean()


def ex_bottom(x, q):
    """Mean after dropping the bottom q share of values."""
    x = np.sort(np.asarray(x, float))
    k = int(round(len(x) * q))
    return x[k:].mean()


def freeze(train, col, bucket, min_n=1):
    """Per-bucket TRAIN-H2 cutpoints at the 1/3 and 2/3 percentiles (builder's method)."""
    cuts = {}
    for b, g in train.groupby(bucket):
        if len(g) >= min_n:
            cuts[b] = tuple(np.percentile(g[col].to_numpy(), [100 / 3, 200 / 3]))
    return cuts


def assign(df, col, bucket, cuts):
    """top if > c67, bottom if <= c33, else mid; no_cut when the bucket has no cutpoint."""
    c33 = df[bucket].map(lambda b: cuts[b][0] if b in cuts else np.nan).to_numpy()
    c67 = df[bucket].map(lambda b: cuts[b][1] if b in cuts else np.nan).to_numpy()
    v = df[col].to_numpy()
    out = np.where(np.isnan(c33), 'no_cut', np.where(v > c67, 'top', np.where(v <= c33, 'bottom', 'mid')))
    return pd.Series(out, index=df.index)


def summarize(df, lab, name):
    """Kept (top) vs dropped per holdout; prints one line per holdout, returns the dict."""
    rows = {}
    for h in ('TRAIN-H2', 'VAL'):
        d = df[df.split == h]
        lb = lab[d.index]
        k = d[lb == 'top']
        dr = d[(lb != 'top') & (lb != 'no_cut')]
        r = dict(nk=len(k), share=len(k) / max(1, int((lb != 'no_cut').sum())), mk=k.outcome_R.mean(),
                 tk=day_t(k.outcome_R, k.day), ex5=ex_top(k.outcome_R, .05), md=dr.outcome_R.mean(),
                 delta=k.outcome_R.mean() - dr.outcome_R.mean(), cache=100 * k.store_served_1438.mean(),
                 nocut=int((lb == 'no_cut').sum()))
        rows[h] = r
        p(f"  {name:34s} {h:8s} kept {r['nk']:5d} ({r['share'] * 100:4.1f}%) mean {r['mk']:+.4f} "
          f"t {r['tk']:+.2f} ex5 {r['ex5']:+.4f} | dropped {r['md']:+.4f} | kept-dropped {r['delta']:+.4f} "
          f"| cache {r['cache']:.1f}% | no_cut {r['nocut']}")
    return rows


def boot_delta(d, lab, reps=1000, seed=7):
    """Day-block bootstrap SE and t of (kept mean - dropped mean) inside one holdout."""
    k = (lab[d.index] == 'top').to_numpy()
    g = pd.DataFrame({'day': d.day.to_numpy(), 'x': d.outcome_R.to_numpy(), 'k': k})
    agg = g.groupby('day')[['x', 'k']].apply(lambda z: pd.Series({
        'sk': z.x[z.k].sum(), 'nk': z.k.sum(), 'sd': z.x[~z.k].sum(), 'nd': (~z.k).sum()}))
    a = agg.to_numpy()
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(reps):
        s = a[rng.randint(0, len(a), len(a))].sum(0)
        out.append(s[0] / s[1] - s[2] / s[3])
    point = a[:, 0].sum() / a[:, 1].sum() - a[:, 2].sum() / a[:, 3].sum()
    se = float(np.std(out))
    return point, se, point / se


def main():
    """Run every lens and print the evidence."""
    f = pd.read_csv(HD + '/cell_1624_fills.csv', dtype={'day': str})
    e = pd.read_csv(HD + '/cell_1624_arm_events_cache.csv', dtype={'day': str})
    f['fmin'] = np.floor(f.fill_min).astype(int)
    f['b15'] = (f.fmin // 15) * 15
    ev = {d: (g.m_hi.to_numpy(), g.symbol.to_numpy()) for d, g in e.groupby('day')}
    empty = (np.array([], int), np.array([], object))

    # ---- lens 1: windows -------------------------------------------------------------------------------
    cols = {c: np.zeros(len(f), int) for c in ('B30_b', 'B30_s', 'B60_s', 'same_other', 'own_b', 'own_s',
                                               'names30_s', 'own_samemin')}
    for d, g in f.groupby('day'):
        m, s = ev.get(d, empty)
        fm = g.fill_min.to_numpy()[:, None]
        fl = g.fmin.to_numpy()[:, None]
        sym = g.symbol.to_numpy()[:, None]
        diff = fm - m[None, :]
        inb = (diff > 0) & (diff <= 30)
        in30 = (m[None, :] >= fl - 30) & (m[None, :] <= fl - 1)
        in60 = (m[None, :] >= fl - 60) & (m[None, :] <= fl - 1)
        own = s[None, :] == sym
        same = m[None, :] == fl
        ix = g.index.to_numpy()
        cols['B30_b'][ix] = inb.sum(1)
        cols['B30_s'][ix] = in30.sum(1)
        cols['B60_s'][ix] = in60.sum(1)
        cols['same_other'][ix] = (same & ~own).sum(1)
        cols['own_b'][ix] = (inb & own).sum(1)
        cols['own_s'][ix] = (in30 & own).sum(1)
        cols['own_samemin'][ix] = (same & own).sum(1)
        cols['names30_s'][ix] = [len(set(s[row])) for row in in30]
    for c, v in cols.items():
        f[c] = v
    p('== LENS 1: arm-count window ==')
    p(f"builder B30 reproduced by m_hi in [fill_min-30, fill_min): {np.mean(f.B30_b == f.B30) * 100:.2f}% of {len(f)} fills")
    p(f"fill_min fractional in {np.mean(f.fill_min != f.fmin) * 100:.1f}% of fills -> builder window includes the fill's own minute")
    p(f"fills with >=1 OTHER-name event in the fill's own minute (post-fill info possible): "
      f"{np.mean(f.same_other > 0) * 100:.1f}%; mean {f.same_other.mean():.2f}, max {f.same_other.max()}")
    p(f"own event inside builder window: {np.mean(f.own_b > 0) * 100:.1f}% of fills; own event in the fill's own minute: "
      f"{np.mean(f.own_samemin > 0) * 100:.1f}%; own-name events in spec window: mean {f.own_s.mean():.2f}")
    p(f"B30 builder mean {f.B30.mean():.2f} vs spec {f.B30_s.mean():.2f}; |diff| mean {np.abs(f.B30 - f.B30_s).mean():.2f}; "
      f"Spearman {f[['B30', 'B30_s']].corr('spearman').iloc[0, 1]:.4f}")

    tr = f[f.split == 'TRAIN-H2']
    cutb = freeze(tr, 'B30', 'hour')
    labb = assign(f, 'B30', 'hour', cutb)
    p(f"builder tercile30 reproduced: {np.mean(labb == f.tercile30) * 100:.2f}%")
    p('-- gate variants (TRAIN-H2 frozen cuts unless stated) --')
    res = {'builder': summarize(f, labb, 'builder B30 (hour)')}
    cuts = freeze(tr, 'B30_s', 'hour')
    labs = assign(f, 'B30_s', 'hour', cuts)
    res['spec'] = summarize(f, labs, 'SPEC B30 strictly-before (hour)')
    p(f"  tercile assignment changes builder->spec: {np.mean(labb != labs) * 100:.1f}% of fills; kept-set Jaccard VAL "
      f"{((labb == 'top') & (labs == 'top') & (f.split == 'VAL')).sum() / ((labb == 'top') | (labs == 'top'))[f.split == 'VAL'].sum():.3f}")
    summarize(f, assign(f, 'B60_s', 'hour', freeze(tr, 'B60_s', 'hour')), 'SPEC B60 (hour, report-only)')
    summarize(f, assign(f, 'names30_s', 'hour', freeze(tr, 'names30_s', 'hour')), 'SPEC distinct names 30 (hour)')

    # ---- lens 2: statuses / TEST ----------------------------------------------------------------------
    p('== LENS 2: statuses covered, TEST sealing ==')
    base = pd.read_csv(HD + '/causal_arming_causal.csv', usecols=['day', 'symbol', 'split', 'status', 'n_cross'],
                       dtype={'day': str}, low_memory=False).drop_duplicates(['day', 'symbol'])
    cnt = e.groupby(['day', 'symbol']).size().rename('n_ev').reset_index()
    j = cnt.merge(base, on=['day', 'symbol'], how='left')
    p(f"events by status of their symbol-day: {j.groupby(j.status.fillna('NOT_IN_BASE')).n_ev.sum().to_dict()}")
    p(f"events by split of their symbol-day: {j.groupby(j.split.fillna('NOT_IN_BASE')).n_ev.sum().to_dict()}")
    p(f"base splits present: {base.split.value_counts().to_dict()}; event days {e.day.min()}..{e.day.max()}, "
      f"fill days {f.day.min()}..{f.day.max()}; event days not among fill days: {len(set(e.day) - set(f.day))}")

    # ---- lens 3: hour adjustment ----------------------------------------------------------------------
    p('== LENS 3: time-of-day ==')
    for h in ('TRAIN-H2', 'VAL'):
        d = f[f.split == h]
        tab = d.assign(k=(labb[d.index] == 'top')).groupby('b15').agg(n=('k', 'size'), kept=('k', 'mean'),
                                                                     R=('outcome_R', 'mean'))
        tab = tab[tab.n >= 30]
        p(f"  {h} by 15-min bucket (n>=30): " + '; '.join(
            f"{int(b) // 60}:{int(b) % 60:02d} n{int(r.n)} kept{r.kept * 100:.0f}% R{r.R:+.2f}" for b, r in tab.iterrows()))
        rho = [(len(g), g[['B30_s', 'outcome_R']].corr('spearman').iloc[0, 1]) for _, g in d.groupby('hour') if len(g) > 30]
        p(f"  {h} within-hour Spearman(B30_s, outcome_R), n-weighted: {sum(n * r for n, r in rho) / sum(n for n, _ in rho):+.4f}; "
          f"within-hour Spearman(B30_b, minute): "
          f"{np.average([g[['B30', 'fmin']].corr('spearman').iloc[0, 1] for _, g in d.groupby('hour') if len(g) > 30]):+.3f}")
    summarize(f, assign(f, 'B30_s', 'b15', freeze(tr, 'B30_s', 'b15', min_n=9)), 'SPEC B30, 15-min bucket cuts')
    lab_w = pd.Series('no_cut', index=f.index)
    for h in ('TRAIN-H2', 'VAL'):
        d = f[f.split == h]
        lab_w[d.index] = assign(d, 'B30_s', 'hour', freeze(d, 'B30_s', 'hour'))
    summarize(f, lab_w, 'SPEC B30, within-holdout hour ranks')
    f['month'] = f.day.str[:7]
    p('  builder kept share by month: ' + ', '.join(
        f"{mo}:{(labb[g.index] == 'top').mean() * 100:.0f}%" for mo, g in f.groupby('month')))

    # ---- lens 4: placebos -----------------------------------------------------------------------------
    p('== LENS 4: placebos ==')
    groups = {h: [(d, g.index.to_numpy(), g.fmin.to_numpy()) for d, g in f[f.split == h].groupby('day')]
              for h in ('TRAIN-H2', 'VAL')}
    days = {h: sorted(f[f.split == h].day.unique()) for h in groups}
    true_k = {h: res['spec'][h]['mk'] for h in groups}
    pl = {h: [] for h in groups}
    for seed in range(1623, 1823):
        rng = np.random.RandomState(seed)
        bp = np.zeros(len(f), int)
        for h, other in (('TRAIN-H2', 'VAL'), ('VAL', 'TRAIN-H2')):
            pick = rng.choice(days[other], size=len(days[h]), replace=len(days[h]) > len(days[other]))
            mp = dict(zip(days[h], pick))
            for d, ix, fl in groups[h]:
                m, _ = ev.get(mp[d], empty)
                bp[ix] = ((m[None, :] >= fl[:, None] - 30) & (m[None, :] <= fl[:, None] - 1)).sum(1)
        lp = assign(f.assign(Bp=bp), 'Bp', 'hour', cuts)
        for h in groups:
            d = f[f.split == h]
            pl[h].append(d.outcome_R[lp[d.index] == 'top'].mean())
    for h in groups:
        a = np.array(pl[h])
        p(f"  day-swap placebo (other holdout's days, seeds 1623..1822) {h}: seed-1623 placebo kept {a[0]:+.4f}, "
          f"mean {a.mean():+.4f} sd {a.std():.4f}; true SPEC kept {true_k[h]:+.4f}; margin {true_k[h] - a.mean():+.4f} "
          f"(z {(true_k[h] - a.mean()) / a.std():+.2f}); seed-1623 margin {true_k[h] - a[0]:+.4f}")
    rng = np.random.RandomState(99)
    for h in groups:
        d = f[f.split == h]
        dist = []
        for _ in range(500):
            sh = d.B30_s.to_numpy().copy()
            for _, g in d.groupby('hour'):
                pos = d.index.get_indexer(g.index)
                sh[pos] = sh[pos][rng.permutation(len(pos))]
            lp = assign(d.assign(Bp=sh), 'Bp', 'hour', cuts)
            dist.append(d.outcome_R[lp == 'top'].mean())
        dist = np.array(dist)
        p(f"  within-hour permutation (500) {h}: placebo kept mean {dist.mean():+.4f} sd {dist.std():.4f}; true "
          f"{true_k[h]:+.4f}; margin {true_k[h] - dist.mean():+.4f}; one-sided p {np.mean(dist >= true_k[h]):.3f}")
        pt, se, t = boot_delta(d, labs)
        pb, seb, tb = boot_delta(d, labb)
        p(f"  day-bootstrap kept-minus-dropped {h}: SPEC {pt:+.4f} (se {se:.4f}, t {t:+.2f}); builder {pb:+.4f} "
          f"(se {seb:.4f}, t {tb:+.2f})")

    # ---- lens 5: tails and day concentration -----------------------------------------------------------
    p('== LENS 5: tails (VAL) and day concentration (both holdouts) ==')
    for nm, lab in (('builder', labb), ('spec', labs)):
        d = f[f.split == 'VAL']
        k = d[lab[d.index] == 'top'].outcome_R
        dr = d[lab[d.index] != 'top'].outcome_R
        for tn, fn in (('mean', np.mean), ('ex-top-1%', lambda x: ex_top(x, .01)), ('ex-top-5%', lambda x: ex_top(x, .05)),
                       ('cap +3R', lambda x: np.minimum(np.asarray(x), 3).mean()),
                       ('ex-bottom-5%', lambda x: ex_bottom(x, .05))):
            p(f"  {nm:7s} {tn:12s} kept {fn(k):+.4f} dropped {fn(dr):+.4f} delta {fn(k) - fn(dr):+.4f}")
    for hh in ('TRAIN-H2', 'VAL'):
        d = f[f.split == hh].assign(k=(labb[f.split == hh] == 'top'))
        per = d.groupby('day').agg(nk=('k', 'sum'), n=('k', 'size'), B=('B30_s', 'mean'))
        per['sk'] = d[d.k].groupby('day').outcome_R.sum()
        per['sk'] = per.sk.fillna(0)
        kd = per[per.nk > 0]
        top10 = kd.nk.sort_values(ascending=False).head(10)
        thrust = per.B.sort_values(ascending=False).head(10).index
        p(f"  {hh} builder kept: {int(per.nk.sum())} fills on {len(kd)} days; top-10 days hold {top10.sum() / per.nk.sum() * 100:.1f}% "
          f"of kept fills; day-weighted kept mean {(kd.sk / kd.nk).mean():+.4f}; positive-sum days {np.mean(kd.sk > 0) * 100:.0f}%")
        ex = d[~d.day.isin(thrust)]
        p(f"  10 thrust days (highest mean B30): {len(d[d.day.isin(thrust) & d.k])} kept fills, mean "
          f"{d[d.day.isin(thrust) & d.k].outcome_R.mean():+.4f} (dropped on those days n {int((d.day.isin(thrust) & ~d.k).sum())} "
          f"{d[d.day.isin(thrust) & ~d.k].outcome_R.mean():+.4f}); ex-thrust kept {ex[ex.k].outcome_R.mean():+.4f} "
          f"(n {int(ex.k.sum())}, t {day_t(ex[ex.k].outcome_R, ex[ex.k].day):+.2f}) dropped {ex[~ex.k].outcome_R.mean():+.4f}")
        srt = kd.sk.sort_values()
        for lbl, drop in (('worst 5', srt.index[:5]), ('best 5', srt.index[-5:])):
            z = d[d.k & ~d.day.isin(drop)]
            p(f"  kept ex {lbl} days: {z.outcome_R.mean():+.4f} (n {len(z)}); those days' share of kept R-sum "
              f"{srt[drop].sum() / srt.sum() * 100:.1f}%")


    # ---- lens 6: cache-only share ----------------------------------------------------------------------
    p('== LENS 6: cache-only share ==')
    p(f"  population cache-only share {f.store_served_1438.mean() * 100:.2f}% (TRAIN-H2 "
      f"{tr.store_served_1438.mean() * 100:.2f}%, VAL {f[f.split == 'VAL'].store_served_1438.mean() * 100:.2f}%)")
    d = f[f.split == 'VAL'].assign(k=(labb[f.split == 'VAL'] == 'top'))
    p('  VAL mean R by (kept, cache-only): ' + '; '.join(
        f"kept={a} cache={b}: n {len(g)} R {g.outcome_R.mean():+.4f}" for (a, b), g in d.groupby(['k', 'store_served_1438'])))
    p(f"  VAL B30 mean by cache-only: {d.groupby('store_served_1438').B30.mean().round(1).to_dict()}")


if __name__ == '__main__':
    main()
