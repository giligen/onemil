#!/usr/bin/env python3
"""hod_filter_stack — stage 5: the PURGED, EMBARGOED WALK-FORWARD learner on B2.

Method as validated in research/meta_label/: for each test month M, train on the 182 days before
it, embargo the 5 sessions immediately before M, purge any training row whose outcome window
overlaps M (0 by construction — an HOD trade closes in its own session — the count is reported),
minimum 50 distinct training sessions or the month is left UNSCORED. n_jobs=1. No random K-fold.

Cells: rank the day's signals by P(net R > 0) and take the top k per day, k in {4, 8, 12} (as
declared), plus a LIVE-CAUSAL threshold twin (the threshold is fit on TRAIN only, so slot
allocation never sees the rest of the day). Controls: shuffled label, per-family ablation.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
import score as S                                    # noqa: E402

D = f'{ROOT}/research/mature_method/hod_filter_stack'
WINDOW_DAYS, EMBARGO, MIN_SESS = 182, 5, 50
DEPTH, TREES, LR, MIN_CHILD = 3, 200, 0.05, 20

FAMILIES = {
    'breadth': ['breadth_min', 'breadth_15m', 'breadth_day', 'cohort_rank_dist', 'rs_vs_cohort',
                'breadth_by_1000'],
    'index': ['spy_dist_hod_pct', 'spy_ret_30m', 'spy_ret_open_sig', 'spy_5m_ret', 'spy_range3',
              'spy_vol20', 'spy_ret_0930_1000'],
    'accel': ['ret_open_1000', 'ret_1000_sig', 'ret_open_sig', 'slope5', 'slope_prior10',
              'slope_accel'],
    'voltrend': ['vol3_over_10', 'bar_vol', 'rv_profile', 'rv_clock', 'cumv'],
    'shape': ['vwap_dist_pct', 'range_pos', 'atr14_pct', 'hod_age_bars', 'n_break',
              'dist_open_pct', 'r_pct', 'level', 'price'],
    'day': ['gap_pct', 'prev_range_pct', 'prev_close_pos', 'dist_20d_high_pct', 'dow', 'adv20',
            'is_wrapper', 'sym_prior_n', 'sym_prior_meanR'],
    'clock': ['entry_m', 'break_m'],
}


def month_bounds(d):
    return sorted(d.month.unique())


def run(d, cols, shuffle=False, seed=7):
    import xgboost as xgb
    d = d.sort_values(['day', 'symbol']).reset_index(drop=True)
    sess = np.array(sorted(d.day.unique()))
    out = np.full(len(d), np.nan)
    purged_total = 0
    for M in month_bounds(d):
        te = d.month == M
        m0 = d.day[te].min()
        i0 = int(np.searchsorted(sess, m0))
        emb_end = sess[max(i0 - EMBARGO, 0)]
        start = (pd.Timestamp(m0) - pd.Timedelta(days=WINDOW_DAYS)).strftime('%Y-%m-%d')
        tr = d[(d.day >= start) & (d.day < emb_end)]
        # purge: any training row whose outcome window overlaps the test month
        pur = tr.day >= m0
        purged_total += int(pur.sum())
        tr = tr[~pur]
        if tr.day.nunique() < MIN_SESS or te.sum() == 0:
            continue
        y = (tr.net.values > 0).astype(int)
        if shuffle:
            y = np.random.default_rng(seed + hash(M) % 1000).permutation(y)
        mdl = xgb.XGBClassifier(max_depth=DEPTH, n_estimators=TREES, learning_rate=LR,
                                min_child_weight=MIN_CHILD, subsample=0.8, colsample_bytree=0.8,
                                n_jobs=1, tree_method='hist', eval_metric='logloss',
                                random_state=seed, verbosity=0)
        mdl.fit(tr[cols].values, y)
        out[te.values] = mdl.predict_proba(d.loc[te, cols].values)[:, 1]
    return d, out, purged_total


def book_topk(d, prob, k):
    dd = d.assign(p=prob)
    dd = dd[dd.p.notna()]
    keep = dd.groupby('day', group_keys=False).apply(lambda g: g.nlargest(k, 'p'))
    return S.apply_book(keep, 12, 4)


def book_thresh(d, prob, thr):
    dd = d.assign(p=prob)
    return S.apply_book(dd[dd.p.notna() & (dd.p >= thr)], 12, 4)


def report(name, b, scored_weeks):
    for sp in S.SPLITS:
        w = S.week_stats(b, sp)
        nw = scored_weeks[sp]
        d = b[b.split == sp]
        wk = d.groupby('wk').pnl.sum()
        allw = [x for x in S.ALL_WEEKS[sp] if x in scored_weeks[f'{sp}_set']]
        ww = wk.reindex(allw).fillna(0.0)
        streak = mx = 0
        for v in (ww < 0).values:
            streak = streak + 1 if v else 0
            mx = max(mx, streak)
        print(f'| {name:<26s} | {sp:5s} | {len(d):5d} | {len(d)/max(nw,1):5.1f} | '
              f'{d.rr.mean():+.3f} | {d.net.mean():+.3f} | '
              f'{(d.net.mean()/(d.net.std(ddof=1)/np.sqrt(len(d)))) if len(d)>2 else float("nan"):+5.2f} | '
              f'{(ww>0).mean()*100:5.1f} | {mx:2d} | {ww.min():+8.0f} | {ww.sum():+9.0f} |')


def main():
    b2 = pd.read_pickle(f'{D}/b2.pkl')
    b2['month'] = b2.day.str[:7]
    b2 = b2[b2.split.isin(('TRAIN', 'VAL'))].copy()
    av = pd.read_csv(f'{D}/availability.csv')
    dropped = set(av[(av.get('drop') == True) | (av.get('note') == 'ABSENT')].feat)  # noqa: E712
    cols = []
    for fam, fs in FAMILIES.items():
        for f in fs:
            if f in b2.columns and f not in dropped and f not in cols:
                v = pd.to_numeric(b2[f], errors='coerce')
                if v.notna().mean() >= 0.80 and v.nunique() > 2:
                    b2[f] = v
                    cols.append(f)
    print(f'learner features ({len(cols)}): {cols}')
    print(f'rows {len(b2)} | months {b2.month.nunique()}')

    d, prob, purged = run(b2, cols)
    scored = d[~np.isnan(prob)]
    sw = {}
    for sp in S.SPLITS:
        ws = set(scored[scored.split == sp].wk.unique())
        sw[sp] = len(ws); sw[f'{sp}_set'] = ws
    print(f'purged training rows: {purged} (0 by construction — the outcome closes its own session)')
    print(f'scored rows {len(scored)} of {len(d)}; scored weeks TRAIN {sw["TRAIN"]} VAL {sw["VAL"]}')
    np.save(f'{D}/prob.npy', prob)
    d[['day', 'symbol', 'entry_m', 'split', 'wk', 'rr', 'net']].assign(p=prob).to_csv(
        f'{D}/learn_scores.csv', index=False)

    print('\n| cell | split | n | /wk | grossR | netR | t | green% | rs | worst $ | total $ |')
    base = S.apply_book(d[~np.isnan(prob)], 12, 4)
    report('B2 (scored months only)', base, sw)
    for k in (4, 8, 12):
        report(f'L-k{k} top-{k}/day', book_topk(d, prob, k), sw)
    # LIVE-CAUSAL twin: threshold fit on TRAIN only to retain ~k per day
    tr = d[(d.split == 'TRAIN') & ~np.isnan(prob)]
    ptr = prob[(d.split == 'TRAIN').values & ~np.isnan(prob)]
    nday_tr = tr.day.nunique()
    for k in (4, 8, 12):
        want = min(k * nday_tr, len(ptr))
        thr = float(np.sort(ptr)[::-1][want - 1]) if want > 0 else 1.0
        report(f'L-t{k} thr {thr:.3f} (causal)', book_thresh(d, prob, thr), sw)

    # controls
    print('\n== CONTROLS ==')
    _, ps, _ = run(b2, cols, shuffle=True)
    report('SHUFFLED top-8 (must be ~0)', book_topk(d, ps, 8), sw)
    print('\n-- per-family ablation (top-8, VAL net R) --')
    for fam in FAMILIES:
        cc = [c for c in cols if c not in FAMILIES[fam]]
        if len(cc) == len(cols) or not cc:
            continue
        _, pa, _ = run(b2, cc)
        bb = book_topk(d, pa, 8)
        v = bb[bb.split == 'VAL']; t = bb[bb.split == 'TRAIN']
        print(f'  drop {fam:9s} -> TRAIN net {t.net.mean():+.3f} (n {len(t)})  '
              f'VAL net {v.net.mean():+.3f} (n {len(v)})')


if __name__ == '__main__':
    main()
