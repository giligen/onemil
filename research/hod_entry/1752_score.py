"""Cell 1,752 score — Stage A (TRAIN anatomy + one-rule grid + pick) and Stage B (frozen rule, one VAL read).

PREREG: research/hod_entry/PREREG_1752.md (+ Amendment 1). TEST is never built or read.
Stage A loads TRAIN rows only, writes 1752_anatomy.csv, 1752_grid.csv, 1752_rule.json and the pick section of
RESULT_1752.md. Stage B refuses to run unless the rule is frozen in RESULT_1752.md, then reads VAL once.

Usage: ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1752_score.py --stage A|B
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_IO_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, json, argparse, time
import numpy as np
import pandas as pd

HE = '/home/ec2-user/onemil/research/hod_entry'
EV = f'{HE}/1752_events.parquet'
FEATS = ['minute', 'min_since_hod', 'attempts', 'min_since_low', 'vol5_dayavg', 'volt_vol5', 'cumv_adv', 'dist_open',
         'dist_hod', 'hs_pct', 'spy_ret', 'pc_ma5', 'pc_ma10', 'pc_ma20', 'pc_ma50', 'ma20_slope5', 'hod_vs_20dh', 'gap',
         'prev_range']
QS = (0.2, 0.4, 0.6, 0.8)
REF_S, REF_K = 'min3', 20


def log(m):
    """Timestamped progress line."""
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def load(split):
    """Rows of one split only (the other split is never materialised)."""
    e = pd.read_parquet(EV, filters=[('split', '==', split)])
    e['wk'] = pd.to_datetime(e.day).dt.to_period('W-FRI').astype(str)
    for c in FEATS + ['net_R', 'gross_R', 'cost_R', 'r_pct']:
        if c in e:
            e[c] = e[c].astype('float32')
    return e


def day_t(x, days):
    """Mean and day-clustered t of x (clusters = trading days)."""
    x = np.asarray(x, float); n = len(x)
    if n < 3:
        return (float(np.mean(x)) if n else np.nan), np.nan, np.nan
    mu = x.mean(); s = pd.Series(x - mu).groupby(np.asarray(days)).sum()
    se = np.sqrt((s ** 2).sum()) / n
    return mu, (mu / se if se > 0 else np.nan), se


def ex_top5(x):
    """Mean after dropping the top 5 % of outcomes."""
    x = np.sort(np.asarray(x, float))
    return float(x[:int(np.floor(len(x) * 0.95))].mean()) if len(x) >= 20 else np.nan


def auc(score, y):
    """Rank AUC of score for binary y (0.5 = no separation)."""
    ok = np.isfinite(score); s, y = score[ok], y[ok]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def clustered_slope_t(f, y, days):
    """Day-clustered t of the winner-minus-loser difference in the z-scored feature."""
    ok = np.isfinite(f); f, y, d = f[ok], y[ok].astype(float), np.asarray(days)[ok]
    if len(f) < 30 or f.std() == 0:
        return np.nan
    z = (f - f.mean()) / f.std(); xc = y - y.mean()
    beta = (xc * z).sum() / (xc ** 2).sum(); e = z - z.mean() - beta * xc
    g = pd.Series(xc * e).groupby(d).sum()
    se = np.sqrt((g ** 2).sum()) / (xc ** 2).sum()
    return beta / se if se > 0 else np.nan


def book(sub, weeks):
    """Standard book statistics for a set of trades."""
    mu, t, se = day_t(sub.net_R, sub.day)
    return dict(n=len(sub), per_wk=len(sub) / weeks, mean_net=mu, day_t=t, se=se, gross=float(sub.gross_R.mean()),
                extop5=ex_top5(sub.net_R), wincap3=float(np.minimum(sub.net_R, 3).mean()) if len(sub) else np.nan,
                p_brk=float(sub.brk.mean()) if len(sub) else np.nan)


def stage_a():
    """TRAIN anatomy, the full one-rule grid, the pick, and the frozen rule in RESULT_1752.md."""
    e = load('TRAIN')
    log(f'TRAIN rows {len(e):,}; etypes {e.etype.value_counts().to_dict()}')
    W = e.wk.nunique(); WH = {h: e[e.half == h].wk.nunique() for h in ('H1', 'H2')}
    an = []
    ref = e[(e.S == REF_S) & (e.K == REF_K)]
    for et, g in ref.groupby('etype'):
        y, yb = (g.net_R > 0).to_numpy().astype(int), g.brk.to_numpy().astype(int)
        for f in FEATS:
            x = g[f].to_numpy(float)
            an.append(dict(kind='auc', etype=et, feature=f, n=len(g), coverage=float(np.isfinite(x).mean()),
                           auc_win=auc(x, y), auc_brk=auc(x, yb), t_win=clustered_slope_t(x, y, g.day),
                           t_brk=clustered_slope_t(x, yb, g.day), win_rate=y.mean(), p_brk=yb.mean()))
            ok = np.isfinite(x)
            if ok.sum() >= 100:
                dec = pd.qcut(pd.Series(x[ok]).rank(method='first'), 10, labels=False).to_numpy()
                for k in range(10):
                    s = dec == k
                    an.append(dict(kind='decile', etype=et, feature=f, decile=k, lo=float(x[ok][s].min()),
                                   hi=float(x[ok][s].max()), n=int(s.sum()), p_brk=float(yb[ok][s].mean()),
                                   mean_net=float(g.net_R.to_numpy()[ok][s].mean()), win_rate=float(y[ok][s].mean())))
    pd.DataFrame(an).to_csv(f'{HE}/1752_anatomy.csv', index=False)
    log('anatomy written')
    grid = []
    for (et, S, K), g in e[e.etype != 'PLACEBO'].groupby(['etype', 'S', 'K']):
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
            r = dict(etype=et, S=S, K=K, feature=f, op=op, thr=thr, **book(s, W))
            for h in ('H1', 'H2'):
                sh = s[s.half == h]
                r[f'{h}_mean'] = float(sh.net_R.mean()) if len(sh) else np.nan
                r[f'{h}_n'] = len(sh); r[f'{h}_per_wk'] = len(sh) / WH[h]
            r['same_sign'] = bool(np.sign(r['H1_mean']) == np.sign(r['H2_mean']))
            grid.append(r)
    G = pd.DataFrame(grid)
    G.to_csv(f'{HE}/1752_grid.csv', index=False)
    log(f'grid: {len(G):,} cells; net>0 on TRAIN {int((G.mean_net > 0).sum())}; '
        f'net>0 both halves {int(((G.H1_mean > 0) & (G.H2_mean > 0)).sum())}')
    elig = G[(G.per_wk >= 3) & (G.H2_per_wk >= 3)].sort_values('H2_mean', ascending=False)
    pick = elig.iloc[0].to_dict()
    json.dump({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in pick.items()},
              open(f'{HE}/1752_rule.json', 'w'), indent=1, default=str)
    unf = G[(G.feature == 'none')].set_index(['etype', 'S', 'K'])
    u = unf.loc[(pick['etype'], pick['S'], pick['K'])]
    with open(f'{HE}/RESULT_1752.md', 'w') as fh:
        fh.write(f"# RESULT 1,752 (stage A frozen {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}, before VAL)\n\n"
                 f"Grid cells: {len(G)}. Picked (best TRAIN-H2 net, >= 3 fills/wk): {pick['etype']} S={pick['S']} "
                 f"K={pick['K']} {pick['feature']} {pick['op']} {pick['thr']:.4g}; TRAIN n {pick['n']} "
                 f"({pick['per_wk']:.1f}/wk) net {pick['mean_net']:+.3f} R (t {pick['day_t']:.2f}), H1 "
                 f"{pick['H1_mean']:+.3f} / H2 {pick['H2_mean']:+.3f}, ex-top-5 % {pick['extop5']:+.3f}, "
                 f"P(break) {pick['p_brk']:.3f}; unfiltered {u['mean_net']:+.3f} R.\n")
    log(f'PICK {pick}')


def stage_b():
    """One VAL read of the frozen rule: book, calibration, placebo, null, cost x1.5, MDE, kill checks."""
    assert os.path.exists(f'{HE}/RESULT_1752.md') and 'stage A frozen' in open(f'{HE}/RESULT_1752.md').read()
    R = json.load(open(f'{HE}/1752_rule.json'))
    v = load('VAL')
    W = v.wk.nunique()

    def apply(df, et):
        s = df[(df.etype == et) & (df.S == R['S']) & (df.K == int(R['K']))]
        if R['feature'] != 'none':
            x = s[R['feature']].to_numpy(float)
            s = s[(x >= R['thr']) if R['op'] == '>=' else (x <= R['thr'])]
        return s
    rule, plc = apply(v, R['etype']), apply(v, 'PLACEBO')
    unf = v[(v.etype == R['etype']) & (v.S == R['S']) & (v.K == int(R['K']))]
    out = dict(val_weeks=W, **{f'rule_{k}': x for k, x in book(rule, W).items()})
    out.update({f'placebo_{k}': x for k, x in book(plc, W).items()})
    out.update({f'unfiltered_{k}': x for k, x in book(unf, W).items()})
    out['rule_measured_cost_only_mean'] = float(rule[rule.cost_src == 'measured'].net_R.mean())
    out['rule_n_measured_cost'] = int((rule.cost_src == 'measured').sum())
    out['rule_cost15_mean'] = float((rule.gross_R - 1.5 * rule.cost_R).mean())
    out['train_p_brk'] = R['p_brk']
    wk = rule.groupby('wk').net_R.sum().reindex(sorted(v.wk.unique()), fill_value=0.0)
    out['weekly_P10'] = float(np.quantile(wk, 0.10)); out['green_week_share'] = float((wk > 0).mean())
    rng = np.random.default_rng(1752); cnt = rule.groupby('wk').size()
    pools = {w: unf[unf.wk == w].net_R.to_numpy() for w in cnt.index}
    null = []
    for _ in range(2000):
        s = pd.Series({w: rng.choice(pools[w], cnt[w], replace=False).sum() for w in cnt.index})
        null.append(float((s.reindex(wk.index, fill_value=0) > 0).mean()))
    out['null_green_share'] = float(np.mean(null)); out['null_green_p'] = float(np.mean(np.array(null) >= out['green_week_share']))
    for mth, g in rule.groupby(rule.day.str[:7]):
        out[f'month_{mth}'] = float(g.net_R.mean())
    out['MDE80_R'] = 2.8 * out['rule_se']
    kills = {'val_net_le0': out['rule_mean_net'] <= 0, 'day_t_lt2': not (out['rule_day_t'] >= 2),
             'train_halves_opposite': not R['same_sign'], 'val_extop5_le0': not (out['rule_extop5'] > 0),
             'fills_lt3': out['rule_per_wk'] < 3, 'placebo_ge_rule': out['placebo_mean_net'] >= out['rule_mean_net'],
             'pbrk_drop_gt10pp': out['rule_p_brk'] < R['p_brk'] - 0.10}
    out.update({f'kill_{k}': bool(x) for k, x in kills.items()})
    out['verdict'] = 'KILL' if any(kills.values()) else 'SURVIVE'
    pd.Series(out).to_csv(f'{HE}/1752_val.csv', header=['value'])
    for k, x in out.items():
        log(f'{k}: {x}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True, choices=['A', 'B'])
    a = ap.parse_args()
    stage_a() if a.stage == 'A' else stage_b()
