#!/usr/bin/env python3
"""Adversarial refuter checks for cell 1,623 (PREREG_1623.md): the day's own feedback gate.

Lenses: (1) causality of F -- exit-minute semantics, same-minute resolution, outcome/exit pairing;
(2) the shuffle placebo -- both readings over many seeds, plus a within-holdout same-clock null;
(3) tails and day concentration of the kept set; (4) the cache-only share; (5) the time-of-day
confound (n_res >= 2 excludes the morning) vs a plain after-11:00 gate and a TOD-matched baseline.

Read-only inputs (TEST rows dropped at load, never scored): cell_1623_fills.csv (builder output),
causal_arming_causal.csv, model_1478_L3_predictions.csv.
Output: research/hod_entry/review/1623_feedback_refuter_chk.json (every number printed too).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HOD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(HOD))
sys.path.insert(0, REPO)
from research.hod_entry import cell_1445 as h1445  # noqa: E402

MIN_N, GP = 2, 0.5
N_PERM = 500
OUT_JSON = os.path.join(HERE, '1623_feedback_refuter_chk.json')
OUT = {}


def log(msg):
    """Verbose progress, flushed."""
    print(msg, flush=True)


def stats(y, day):
    """n, mean, day-clustered t (the builder's helper), ex-top-5 %, n days, median for one sample."""
    y = pd.Series(np.asarray(y, float)).reset_index(drop=True)
    day = pd.Series(np.asarray(day)).reset_index(drop=True)
    if len(y) == 0:
        return dict(n=0)
    return dict(n=int(len(y)), mean=round(float(y.mean()), 4),
                t=round(float(h1445.day_clustered_t(y, day)), 2),
                ex5=round(float(h1445.ex_top5_mean(y)), 4), days=int(day.nunique()),
                median=round(float(y.median()), 4))


def cl_se(y, day):
    """Day-clustered SE of the mean, statsmodels 'cluster' default (G/(G-1) correction, K=1)."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 2:
        return np.nan
    s = pd.Series(y - y.mean()).groupby(np.asarray(day)).sum().to_numpy()
    g = len(s)
    if g < 2:
        return np.nan
    return float(np.sqrt(g / (g - 1) * (s ** 2).sum() / n ** 2))


def margin_t(a, a_day, b, b_day):
    """Two-sample margin (mean a - mean b) and its t with day-clustered SEs (independent samples)."""
    m = float(np.mean(a) - np.mean(b))
    se = np.sqrt(cl_se(a, a_day) ** 2 + cl_se(b, b_day) ** 2)
    return round(m, 4), round(float(m / se), 2) if se > 0 else float('nan')


def load():
    """Builder rows + fill_min/exit_m/why/net_R (causal file) + why/store flag (1478 file); TEST dropped."""
    b = pd.read_csv(os.path.join(HOD, 'cell_1623_fills.csv')).rename(columns={'split': 'holdout'})
    ca = pd.read_csv(os.path.join(HOD, 'causal_arming_causal.csv'), low_memory=False)
    ca = ca[(ca.status == 'fill') & ((ca.split == 'VAL') | ((ca.split == 'TRAIN') & (ca.half == 'H2')))]
    ca = ca[['day', 'symbol', 'fill_min', 'exit_m', 'why', 'net_R', 'raw_R']]
    pr = pd.read_csv(os.path.join(HOD, 'model_1478_L3_predictions.csv'), low_memory=False)
    pr = pr[pr.split != 'TEST'][['day', 'symbol', 'fill_min', 'why', 'outcome_R', 'store_served_1438']]
    pr = pr.rename(columns={'why': 'why_1478', 'outcome_R': 'outcome_R_1478'})
    df = b.merge(ca, on=['day', 'symbol'], how='inner', validate='one_to_one')
    df = df.merge(pr, on=['day', 'symbol', 'fill_min'], how='left', validate='one_to_one')
    log(f'rows: builder {len(b)}, merged {len(df)}, 1478 unmatched {int(df.outcome_R_1478.isna().sum())}')
    return df.reset_index(drop=True)


def build_tabs(ctx_day, ctx_exit, ctx_out):
    """Per day label: sorted exit times and the cumulative outcome sums (prefix 0)."""
    ctx = pd.DataFrame({'d': np.asarray(ctx_day), 'e': np.asarray(ctx_exit, float),
                        'o': np.asarray(ctx_out, float)})
    tabs = {}
    for d, g in ctx.groupby('d', sort=False):
        order = np.argsort(g.e.to_numpy(), kind='mergesort')
        tabs[d] = (g.e.to_numpy()[order], np.concatenate([[0.0], np.cumsum(g.o.to_numpy()[order])]))
    return tabs


def query(tabs, q_idx, q_thr, day_map=None):
    """n_res, F for query rows grouped by day (q_idx: {day: row idx}); context rows of the day
    day_map[day] (identity if None) whose exit < q_thr (strict)."""
    n = len(q_thr)
    n_res = np.zeros(n, int)
    F = np.full(n, np.nan)
    for d, idx in q_idx.items():
        src = d if day_map is None else day_map[d]
        if src not in tabs:
            continue
        e, cs = tabs[src]
        k = np.searchsorted(e, q_thr[idx], side='left')
        n_res[idx] = k
        F[idx] = np.where(k > 0, cs[k] / np.maximum(k, 1), np.nan)
    return n_res, F


def groups(day):
    """{day: row indices} for a day array."""
    return pd.Series(np.asarray(day)).groupby(np.asarray(day), sort=False).indices


def gate(n_res, F):
    """G+ : n_res >= 2 and F >= +0.5 R."""
    return (n_res >= MIN_N) & (np.nan_to_num(F, nan=-9.0) >= GP)


def check_causality(df):
    """Reproduce F/n_res; exit/outcome pairing; same-minute resolution; strict-minute and +1-min variants."""
    tabs = build_tabs(df.day, df.exit_m, df.outcome_R)
    qi = groups(df.day)
    fm = df.fill_min.to_numpy(float)
    nr, F = query(tabs, qi, fm)
    g = gate(nr, F)
    r = dict(nres_mismatch=int((nr != df.n_res.to_numpy()).sum()),
             F_maxdiff=float(np.nanmax(np.abs(np.nan_to_num(F, nan=0) - np.nan_to_num(df.F.to_numpy(), nan=0)))),
             gate_mismatch=int((g.astype(int) != df.gate_plus.to_numpy()).sum()))
    r['outcome_vs_1478_maxdiff'] = float(np.abs(df.outcome_R - df.outcome_R_1478).max())
    r['why_crosstab'] = {f'{a}|{b}': int(v) for (a, b), v in
                         df.groupby(['why', 'why_1478']).size().items()}
    d = (df.outcome_R - df.net_R).abs()
    r['abs_outcome_minus_causal_netR_q'] = {q: round(float(d.quantile(q)), 4) for q in (0.5, 0.9, 0.99, 1.0)}
    r['n_abs_diff_gt_0.25'] = int((d > 0.25).sum())
    r['n_sign_differs'] = int((np.sign(df.outcome_R) != np.sign(df.net_R)).sum())
    r['exit_before_fill_violations'] = int((df.exit_m < df.fill_min).sum())
    r['max_fill_min'] = float(df.fill_min.max())
    r['n_eod_exits'] = int((df.why == 'eod').sum())
    r['frac_exit_integer'] = round(float((df.exit_m == np.floor(df.exit_m)).mean()), 4)
    # strict minute: the resolved fill's whole exit minute precedes the fill minute (exit < floor(fill))
    for tag, thr in (('minute', np.floor(fm)), ('lat1', np.floor(fm) - 1.0)):
        nr2, F2 = query(tabs, qi, thr)
        g2 = gate(nr2, F2)
        r[f'{tag}_fills_with_samemin_resolved'] = int((nr2 != nr).sum())
        r[f'{tag}_gate_flips_out'] = int((g & ~g2).sum())
        r[f'{tag}_gate_flips_in'] = int((~g & g2).sum())
        for h in ('TRAIN-H2', 'VAL'):
            m = (df.holdout == h).to_numpy()
            r[f'{tag}_{h}_kept'] = stats(df.outcome_R[m & g2], df.day[m & g2])
        df[f'g_{tag}'] = g2
    # kept fills that rely on at least one same-minute resolution (the builder's own rule)
    r['kept_relying_on_samemin'] = int((g & (df.g_minute.to_numpy() == 0) | (g & (query(tabs, qi, np.floor(fm))[0] != nr))).sum())
    return r


def placebo_block(df, h, seed, tabs_o, qi_o, fm_o, oth):
    """Builder reading: the other holdout's fills, F from a permuted other-holdout DAY (block)."""
    days = np.array(sorted(tabs_o.keys()))
    pm = dict(zip(days, np.random.RandomState(seed).permutation(days)))
    nr, F = query(tabs_o, qi_o, fm_o, pm)
    return gate(nr, F)


def placebo_row(oth, seed):
    """Rebuild reading: the other holdout's per-row day labels permuted, F recomputed on pseudo-days."""
    lab = np.random.RandomState(seed).permutation(oth.day.to_numpy())
    tabs = build_tabs(lab, oth.exit_m, oth.outcome_R)
    nr, F = query(tabs, groups(lab), oth.fill_min.to_numpy(float))
    return gate(nr, F), lab


def check_placebo(df):
    """Seed-1623 reproduction of both readings, 500-seed distributions, day-clustered margin t, and the
    within-holdout same-clock day-block null (1,000 perms) -- the TOD-controlled test of 'own day'."""
    r = {}
    for h in ('VAL', 'TRAIN-H2'):
        o = 'TRAIN-H2' if h == 'VAL' else 'VAL'
        real = df[(df.holdout == h) & (df.gate_plus == 1)]
        oth = df[df.holdout == o].reset_index(drop=True)
        tabs_o, qi_o, fm_o = build_tabs(oth.day, oth.exit_m, oth.outcome_R), groups(oth.day), oth.fill_min.to_numpy(float)
        rr = {'real_kept': stats(real.outcome_R, real.day)}
        gb = placebo_block(df, h, 1623, tabs_o, qi_o, fm_o, oth)
        pk = oth[gb]
        rr['block_1623'] = dict(stats(pk.outcome_R, pk.day), margin_t=margin_t(real.outcome_R, real.day, pk.outcome_R, pk.day))
        gr, lab = placebo_row(oth, 1623)
        pk = oth[gr]
        rr['row_1623'] = dict(stats(pk.outcome_R, lab[gr]), margin_t_pseudoday=margin_t(real.outcome_R, real.day, pk.outcome_R, lab[gr]),
                              margin_t_realday=margin_t(real.outcome_R, real.day, pk.outcome_R, pk.day))
        for name in ('block', 'row'):
            ms, ts, ns = [], [], []
            for s in range(N_PERM):
                if name == 'block':
                    gg = placebo_block(df, h, s, tabs_o, qi_o, fm_o, oth)
                    dd = oth.day[gg]
                else:
                    gg, lab = placebo_row(oth, s)
                    dd = oth.day[gg]
                y = oth.outcome_R[gg]
                m, t = margin_t(real.outcome_R, real.day, y, dd)
                ms.append(m); ts.append(t); ns.append(int(gg.sum()))
            ms, ts = np.array(ms), np.array(ts)
            rr[f'{name}_{N_PERM}seeds'] = dict(
                placebo_mean_p50=round(float(real.outcome_R.mean() - np.median(ms)), 4),
                margin_p05_p50_p95=[round(float(np.percentile(ms, q)), 4) for q in (5, 50, 95)],
                t_p05_p50_p95=[round(float(np.nanpercentile(ts, q)), 2) for q in (5, 50, 95)],
                share_margin_ge_0p10=round(float((ms >= 0.10).mean()), 3),
                share_pass_margin_and_t2=round(float(((ms >= 0.10) & (ts >= 2)).mean()), 3),
                kept_n_p50=int(np.median(ns)))
        # within-holdout same-clock null: the fill's own holdout, F from a random OTHER day's resolved book
        hh = df[df.holdout == h].reset_index(drop=True)
        tabs_h, qi_h, fm_h = build_tabs(hh.day, hh.exit_m, hh.outcome_R), groups(hh.day), hh.fill_min.to_numpy(float)
        days = np.array(sorted(tabs_h.keys()))
        null_m, null_lift, null_n = [], [], []
        for s in range(1000):
            pm = dict(zip(days, np.random.RandomState(10_000 + s).permutation(days)))
            nr, F = query(tabs_h, qi_h, fm_h, pm)
            gg = gate(nr, F)
            null_m.append(hh.outcome_R[gg].mean()); null_n.append(int(gg.sum()))
            null_lift.append(hh.outcome_R[gg].mean() - hh.outcome_R[~gg].mean())
        null_m, null_lift = np.array(null_m), np.array(null_lift)
        real_m = float(real.outcome_R.mean())
        real_lift = real_m - float(hh.outcome_R[hh.gate_plus == 0].mean())
        rr['sameclock_null'] = dict(null_mean_p05_p50_p95=[round(float(np.percentile(null_m, q)), 4) for q in (5, 50, 95)],
                                    real_mean=round(real_m, 4), pct_null_ge_real=round(float((null_m >= real_m).mean()), 3),
                                    null_lift_p50=round(float(np.median(null_lift)), 4), real_lift=round(real_lift, 4),
                                    pct_null_lift_ge_real=round(float((null_lift >= real_lift).mean()), 3),
                                    null_kept_n_p50=int(np.median(null_n)))
        r[h] = rr
        log(f'placebo {h}: {json.dumps(rr)}')
    return r


def check_tails(df):
    """Day concentration and tails of the kept set per holdout."""
    r = {}
    for h in ('VAL', 'TRAIN-H2'):
        k = df[(df.holdout == h) & (df.gate_plus == 1)]
        byd = k.groupby('day').outcome_R.agg(['sum', 'size']).sort_values('sum', ascending=False)
        tot = float(byd['sum'].sum())
        top = byd.index[:3].tolist()
        r[h] = dict(n=int(len(k)), days=int(len(byd)), sum_R=round(tot, 2),
                    top3_days=[(d, round(float(byd.loc[d, 'sum']), 2), int(byd.loc[d, 'size'])) for d in top],
                    top1_share_pct=round(float(byd['sum'].iloc[0] / tot * 100), 1) if tot else None,
                    mean_ex_top1_day=round(float(k[~k.day.isin(top[:1])].outcome_R.mean()), 4),
                    mean_ex_top3_days=round(float(k[~k.day.isin(top)].outcome_R.mean()), 4),
                    share_days_positive=round(float((byd['sum'] > 0).mean()), 3),
                    top5_days_share_of_fills=round(float(byd['size'].sort_values(ascending=False).iloc[:5].sum() / len(k)), 3),
                    max_fills_one_day=int(byd['size'].max()),
                    max_R=round(float(k.outcome_R.max()), 3), win_rate=round(float((k.outcome_R > 0).mean()), 3))
    log(f'tails: {json.dumps(r)}')
    return r


def check_cache(df):
    """Cache-only (store_served_1438) share: baseline vs kept, and kept mean split by the flag."""
    r = {'baseline_all_pct': round(float(df.store_served_1438.mean() * 100), 1)}
    for h in ('VAL', 'TRAIN-H2'):
        hh = df[df.holdout == h]
        k = hh[hh.gate_plus == 1]
        r[h] = dict(base_pct=round(float(hh.store_served_1438.mean() * 100), 1),
                    kept_pct=round(float(k.store_served_1438.mean() * 100), 1),
                    kept_cache=stats(k.outcome_R[k.store_served_1438 == 1], k.day[k.store_served_1438 == 1]),
                    kept_noncache=stats(k.outcome_R[k.store_served_1438 == 0], k.day[k.store_served_1438 == 0]))
    log(f'cache: {json.dumps(r)}')
    return r


def check_tod(df):
    """Is G+ a time-of-day gate in disguise? after-11:00 gate, n_res>=2 population, TOD-matched lift."""
    r = {}
    df['bucket'] = np.floor((df.fill_min - 570) / 30).astype(int)
    for h in ('VAL', 'TRAIN-H2'):
        hh = df[df.holdout == h].copy()
        k = hh[hh.gate_plus == 1]
        nk = hh[hh.gate_plus == 0]
        aft = hh[hh.fill_min >= 660]
        exp = nk.groupby('bucket').outcome_R.mean()
        adj = k.outcome_R - k.bucket.map(exp)
        g2 = hh[hh.n_res >= MIN_N]
        exp2 = g2[g2.gate_plus == 0].groupby('bucket').outcome_R.mean()
        adj2 = k.outcome_R - k.bucket.map(exp2)
        r[h] = dict(all=stats(hh.outcome_R, hh.day),
                    kept_fillmin_q25_50_75=[round(float(k.fill_min.quantile(q)), 1) for q in (.25, .5, .75)],
                    dropped_fillmin_q25_50_75=[round(float(nk.fill_min.quantile(q)), 1) for q in (.25, .5, .75)],
                    kept_share_after_11=round(float((k.fill_min >= 660).mean()), 3),
                    all_share_after_11=round(float((hh.fill_min >= 660).mean()), 3),
                    after_11_gate=stats(aft.outcome_R, aft.day),
                    before_11=stats(hh.outcome_R[hh.fill_min < 660], hh.day[hh.fill_min < 660]),
                    nres_ge2_pop=stats(g2.outcome_R, g2.day),
                    nres_ge2_share_after_11=round(float((g2.fill_min >= 660).mean()), 3),
                    after_11_Gplus=stats(aft.outcome_R[aft.gate_plus == 1], aft.day[aft.gate_plus == 1]),
                    after_11_notGplus=stats(aft.outcome_R[aft.gate_plus == 0], aft.day[aft.gate_plus == 0]),
                    tod_adj_vs_all_dropped=stats(adj.dropna(), k.day[adj.notna()]),
                    tod_adj_vs_gateable_dropped=stats(adj2.dropna(), k.day[adj2.notna()]))
        r[h]['bucket_means'] = {int(b): [int(len(g)), round(float(g.outcome_R.mean()), 3),
                                         int(g.gate_plus.sum())] for b, g in hh.groupby('bucket')}
    log(f'tod: {json.dumps(r)}')
    return r


def main():
    """Run every lens and write the JSON."""
    df = load()
    OUT['causality'] = check_causality(df)
    log(f'causality: {json.dumps(OUT["causality"])}')
    OUT['tails'] = check_tails(df)
    OUT['cache'] = check_cache(df)
    OUT['tod'] = check_tod(df)
    OUT['placebo'] = check_placebo(df)
    with open(OUT_JSON, 'w') as fh:
        json.dump(OUT, fh, indent=1, default=str)
    log(f'wrote {OUT_JSON}')


if __name__ == '__main__':
    main()
