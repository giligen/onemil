"""Cell 1,762 score - Stage A (TRAIN: floor audit, anatomy, approach-only grid, pick) and Stage B (one VAL read).

PREREG: research/hod_entry/PREREG_1762.md.  Statistics helpers (load, day_t, ex_top5, auc, clustered_slope_t, book)
are imported from 1752_score.py with its EV path pointed at 1762_events.parquet.  Frozen changes vs 1,752:
  1. R floor: only entries with r_pct >= 1.0 are eligible (every etype, placebo and comparators included);
  2. the grid is the APPR / APPR2 entries only; BREAK, HOLD10 and RETEST are one comparator row each (unfiltered,
     at the picked stop / K), TRAIN and VAL.
TEST is never built or read.  Stage B refuses to run unless 1762_rule.json is frozen.

Usage: ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1762_score.py --stage A|B
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_IO_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, json, argparse, time, importlib.util
import numpy as np
import pandas as pd

HE = '/home/ec2-user/onemil/research/hod_entry'
_sp = importlib.util.spec_from_file_location('s1752', f'{HE}/1752_score.py')
S = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(S)
S.EV = f'{HE}/1762_events.parquet'
FEATS, QS, REF_S, REF_K = S.FEATS, S.QS, S.REF_S, S.REF_K
FLOOR = 1.0
APPR = ('APPR_0.5', 'APPR_1.0', 'APPR2_0.5', 'APPR2_1.0')
COMPARATORS = ('BREAK', 'HOLD10', 'RETEST')
log = S.log


def stage_a():
    """TRAIN floor audit, anatomy, approach-only grid, pick; writes 1762_floor/anatomy/grid.csv and 1762_rule.json."""
    e = S.load('TRAIN')
    log(f'TRAIN rows {len(e):,}; etypes {e.etype.value_counts().to_dict()}')
    W = e.wk.nunique(); WH = {h: e[e.half == h].wk.nunique() for h in ('H1', 'H2')}
    fl = []
    for (et, s_), g in e.groupby(['etype', 'S']):
        g = g[g.K == REF_K]
        fl.append(dict(split='TRAIN', etype=et, S=s_, n_before=len(g), n_after=int((g.r_pct >= FLOOR).sum()),
                       removed=int((g.r_pct < FLOOR).sum()), removed_share=float((g.r_pct < FLOOR).mean()),
                       median_r_pct=float(g.r_pct.median())))
    pd.DataFrame(fl).to_csv(f'{HE}/1762_floor_train.csv', index=False)
    e = e[e.r_pct >= FLOOR]
    log(f'after R >= {FLOOR} % floor: {len(e):,} rows')
    an = []
    ref = e[(e.S == REF_S) & (e.K == REF_K)]
    for et, g in ref.groupby('etype'):
        y, yb = (g.net_R > 0).to_numpy().astype(int), g.brk.to_numpy().astype(int)
        an.append(dict(kind='summary', etype=et, n=len(g), mean_net=float(g.net_R.mean()), gross=float(g.gross_R.mean()),
                       p_brk=yb.mean(), win_rate=y.mean(), median_r_pct=float(g.r_pct.median())))
        for f in FEATS:
            x = g[f].to_numpy(float)
            an.append(dict(kind='auc', etype=et, feature=f, n=len(g), coverage=float(np.isfinite(x).mean()),
                           auc_win=S.auc(x, y), auc_brk=S.auc(x, yb), t_win=S.clustered_slope_t(x, y, g.day),
                           t_brk=S.clustered_slope_t(x, yb, g.day), win_rate=y.mean(), p_brk=yb.mean()))
            ok = np.isfinite(x)
            if ok.sum() >= 100:
                dec = pd.qcut(pd.Series(x[ok]).rank(method='first'), 10, labels=False).to_numpy()
                for k in range(10):
                    s = dec == k
                    an.append(dict(kind='decile', etype=et, feature=f, decile=k, lo=float(x[ok][s].min()),
                                   hi=float(x[ok][s].max()), n=int(s.sum()), p_brk=float(yb[ok][s].mean()),
                                   mean_net=float(g.net_R.to_numpy()[ok][s].mean()), win_rate=float(y[ok][s].mean())))
    pd.DataFrame(an).to_csv(f'{HE}/1762_anatomy.csv', index=False)
    log('anatomy written')
    grid = []
    for (et, s_, K), g in e[e.etype.isin(APPR)].groupby(['etype', 'S', 'K']):
        cells = [('none', '', np.nan, np.ones(len(g), bool))]
        for f in FEATS:
            x = g[f].to_numpy(float)
            if np.isfinite(x).mean() < 0.5:
                continue
            for q in QS:
                thr = float(np.nanquantile(x, q))
                cells += [(f, '>=', thr, x >= thr), (f, '<=', thr, x <= thr)]
        for f, op, thr, msk in cells:
            s = g[msk]
            if len(s) < 20:
                continue
            r = dict(etype=et, S=s_, K=K, feature=f, op=op, thr=thr, **S.book(s, W))
            for h in ('H1', 'H2'):
                sh = s[s.half == h]
                r[f'{h}_mean'] = float(sh.net_R.mean()) if len(sh) else np.nan
                r[f'{h}_n'] = len(sh); r[f'{h}_per_wk'] = len(sh) / WH[h]
            r['same_sign'] = bool(np.sign(r['H1_mean']) == np.sign(r['H2_mean']))
            grid.append(r)
    G = pd.DataFrame(grid)
    G.to_csv(f'{HE}/1762_grid.csv', index=False)
    log(f'grid: {len(G):,} cells; net>0 on TRAIN {int((G.mean_net > 0).sum())}; '
        f'net>0 both halves {int(((G.H1_mean > 0) & (G.H2_mean > 0)).sum())}; '
        f'both halves >0 and t>=2 {int(((G.H1_mean > 0) & (G.H2_mean > 0) & (G.day_t >= 2)).sum())}')
    elig = G[(G.per_wk >= 3) & (G.H2_per_wk >= 3)].sort_values('H2_mean', ascending=False)
    pick = elig.iloc[0].to_dict()
    pick['frozen_at'] = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())
    json.dump({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in pick.items()},
              open(f'{HE}/1762_rule.json', 'w'), indent=1, default=str)
    unf = G[G.feature == 'none'].set_index(['etype', 'S', 'K']).loc[(pick['etype'], pick['S'], pick['K'])]
    cmp_rows = []
    for et in COMPARATORS:
        g = e[(e.etype == et) & (e.S == pick['S']) & (e.K == int(pick['K']))]
        if len(g):
            cmp_rows.append(dict(split='TRAIN', etype=et, **S.book(g, W)))
    pd.DataFrame(cmp_rows).to_csv(f'{HE}/1762_comparators_train.csv', index=False)
    with open(f'{HE}/RESULT_1762.md', 'w') as fh:
        fh.write(f"# RESULT 1,762 (stage A frozen {pick['frozen_at']}, before VAL)\n\n"
                 f"Grid cells: {len(G)}. Picked: {pick['etype']} S={pick['S']} K={pick['K']} {pick['feature']} "
                 f"{pick['op']} {pick['thr']:.4g}; TRAIN n {pick['n']} ({pick['per_wk']:.1f}/wk) net "
                 f"{pick['mean_net']:+.3f} R (t {pick['day_t']:.2f}), H1 {pick['H1_mean']:+.3f} / H2 "
                 f"{pick['H2_mean']:+.3f}, ex-top-5 % {pick['extop5']:+.3f}, P(break) {pick['p_brk']:.3f}; "
                 f"unfiltered {unf['mean_net']:+.3f} R.\n")
    log(f'PICK {pick}')


def stage_b():
    """One VAL read of the frozen rule: book, calibration, placebo, null, cost x1.5, MDE, comparators, kill checks."""
    R = json.load(open(f'{HE}/1762_rule.json'))
    assert R.get('frozen_at'), 'rule not frozen'
    v = S.load('VAL')
    fl = [dict(split='VAL', etype=et, S=s_, n_before=len(g), removed=int((g.r_pct < FLOOR).sum()),
               removed_share=float((g.r_pct < FLOOR).mean())) for (et, s_), g in v[v.K == REF_K].groupby(['etype', 'S'])]
    pd.DataFrame(fl).to_csv(f'{HE}/1762_floor_val.csv', index=False)
    v = v[v.r_pct >= FLOOR]
    W = v.wk.nunique()

    def apply(df, et, filt=True):
        s = df[(df.etype == et) & (df.S == R['S']) & (df.K == int(R['K']))]
        if filt and R['feature'] != 'none':
            x = s[R['feature']].to_numpy(float)
            s = s[(x >= R['thr']) if R['op'] == '>=' else (x <= R['thr'])]
        return s
    rule, plc = apply(v, R['etype']), apply(v, 'PLACEBO')
    unf = apply(v, R['etype'], filt=False)
    out = dict(val_weeks=W, **{f'rule_{k}': x for k, x in S.book(rule, W).items()})
    out.update({f'placebo_{k}': x for k, x in S.book(plc, W).items()})
    out.update({f'unfiltered_{k}': x for k, x in S.book(unf, W).items()})
    for et in COMPARATORS:
        c = apply(v, et, filt=False)
        out.update({f'cmp_{et}_{k}': x for k, x in S.book(c, W).items()} if len(c) else {})
    out['rule_cost15_mean'] = float((rule.gross_R - 1.5 * rule.cost_R).mean())
    out['rule_cost_src_symbol_median_share'] = float((rule.cost_src == 'symbol_median').mean())
    out['train_p_brk'] = R['p_brk']
    wk = rule.groupby('wk').net_R.sum().reindex(sorted(v.wk.unique()), fill_value=0.0)
    out['weekly_P10'] = float(np.quantile(wk, 0.10)); out['green_week_share'] = float((wk > 0).mean())
    rng = np.random.default_rng(1762); cnt = rule.groupby('wk').size()
    pools = {w: unf[unf.wk == w].net_R.to_numpy() for w in cnt.index}
    null = []
    for _ in range(2000):
        s = pd.Series({w: rng.choice(pools[w], min(cnt[w], len(pools[w])), replace=False).sum() for w in cnt.index})
        null.append(float((s.reindex(wk.index, fill_value=0) > 0).mean()))
    out['null_green_share'] = float(np.mean(null)); out['null_green_p'] = float(np.mean(np.array(null) >= out['green_week_share']))
    out['MDE80_R'] = 2.8 * out['rule_se']
    kills = {'val_net_le0': out['rule_mean_net'] <= 0, 'day_t_lt2': not (out['rule_day_t'] >= 2),
             'train_halves_opposite': not R['same_sign'], 'val_extop5_le0': not (out['rule_extop5'] > 0),
             'fills_lt3': out['rule_per_wk'] < 3, 'placebo_ge_rule': out['placebo_mean_net'] >= out['rule_mean_net'],
             'pbrk_drop_gt10pp': out['rule_p_brk'] < R['p_brk'] - 0.10}
    out.update({f'kill_{k}': bool(x) for k, x in kills.items()})
    out['verdict'] = 'KILL' if any(kills.values()) else 'SURVIVE'
    pd.Series(out).to_csv(f'{HE}/1762_val.csv', header=['value'])
    for k, x in out.items():
        log(f'{k}: {x}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True, choices=['A', 'B'])
    a = ap.parse_args()
    stage_a() if a.stage == 'A' else stage_b()
