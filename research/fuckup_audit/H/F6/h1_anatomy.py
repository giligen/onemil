#!/usr/bin/env python3
"""Stage H / F6 — METHOD step 1: the anatomy of the losers, TRAIN only.

Runs on BOTH populations (Q = the causal U1uU2 universe, the one the METHOD's reference numbers
come from and the PRIMARY here; P = the >=5%-range-day universe of Stage B/C/D1, the twin).

  1.0  availability audit of every feature with < 100% coverage (PLAN §1 standing rule)
  1.1  concentration: P&L by day / by week, worst 5%/10% of days, the worst 20 days annotated
  1.2  trade anatomy: every causal feature, losers vs winners, bucket mean net R,
       booked AND whole-population, with the H1/H2 era split for every bucket
  1.3  path anatomy: minutes to stop, MAE/MFE, +0.5R/+1R on the table for the losers
  1.4  era consistency inside TRAIN (H1 = 2025-01..06, H2 = 2025-07..12)

VAL and TEST are NOT touched by this script.

Usage: ulimit -v 1500000; nice -n 10 python3 research/fuckup_audit/H/F6/h1_anatomy.py
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/H/F6')
import h_core as C

H = C.H
V = 'hold'
NET = f'net_{V}'
pd.set_option('display.width', 250)

# feature -> cut spec.  'q4' = population quartiles (coarse, the METHOD's rule), or explicit edges.
NUM_FEATS = [
    ('next_entry_m', [570, 600, 660, 780, 842]),
    ('price', [5, 10, 20, 50, 1e9]),
    ('spread_cc_bps', 'q4'),
    ('spread_over_r', 'q4'),
    ('next_r_pct', [1, 2, 3, 5, 1e9]),
    ('gap_pct', [-1e9, 0, 3, 10, 1e9]),
    ('prev_day_range_pct', [0, 5, 8, 15, 1e9]),
    ('range_so_far_pct', [5, 8, 12, 20, 1e9]),
    ('dist_open_pct', 'q4'),
    ('rv_adv', 'q4'),
    ('log_adv20', 'q4'),
    ('consol_bars', 'q4'),
    ('n_touches', 'q4'),
    ('consol_vol_ratio', 'q4'),
    ('vwap_dist_pct', [-1e9, 0, 1, 3, 1e9]),
    ('log_cum_dv', 'q4'),
    ('sig_close_pos', [0, 0.25, 0.5, 0.75, 1.0001]),
    ('sig_body_pct', 'q4'),
    ('sig_range_pct', 'q4'),
    ('sig_seq', [0, 1, 2, 4, 1e9]),
    ('spy_ret', [-1e9, -0.3, 0, 0.3, 1e9]),
    ('iwm_ret', [-1e9, -0.3, 0, 0.3, 1e9]),
    ('spy_gap', [-1e9, -0.2, 0.2, 1e9]),
    ('iwm_gap', [-1e9, -0.2, 0.2, 1e9]),
    ('spy_prev_ret', [-1e9, -0.3, 0.3, 1e9]),
    ('spy_vol20', 'q4'),
    ('spy_vs_sma20', [-1e9, 0, 1e9]),
    ('log_pm', 'q4'),
]
CAT_FEATS = ['band', 'close_confirm', 'asset_class', 'dow', 'has_news', 'news_known', 'pm_known',
             'regime', 'gap_up', 'mon']


def era(day):
    return np.where(day < '2025-07-01', 'H1', 'H2')


def edges_for(ref, feat, spec):
    """ONE set of bucket edges per feature, fixed on the TRAIN POPULATION, used for BOTH scopes.
    (Computing quartiles separately per scope makes the two tables un-joinable — the bug that made
    the first run's candidate list empty.)"""
    if spec != 'q4':
        return spec
    s = ref[feat].dropna()
    if len(s) < 50:
        return None
    e = list(np.unique(np.quantile(s, [0, .25, .5, .75, 1.0])))
    if len(e) < 3:
        return None
    e[0], e[-1] = -np.inf, np.inf
    return e


def buckets(s, spec):
    lab = pd.cut(s, bins=spec, right=False, include_lowest=True)
    return lab.astype(str)


def bucket_table(d, feat, spec, label):
    """mean net R by bucket, overall and per TRAIN half, plus the population column."""
    if feat not in d.columns:
        return []
    s = d[feat]
    if s.notna().sum() < 50:
        return []
    b = buckets(s, spec)
    b = b.where(s.notna(), 'MISSING')
    out = []
    g = d.assign(_b=b).groupby('_b', observed=True)
    for k, x in g:
        if len(x) < 20:
            continue
        h1, h2 = x[x.era == 'H1'], x[x.era == 'H2']
        out.append(dict(scope=label, feat=feat, bucket=k, n=len(x),
                        meanR=round(float(x[NET].mean()), 4),
                        share=round(len(x) / len(d), 3),
                        WR=round(float((x[NET] > 0).mean() * 100), 1),
                        stopP=round(float((x[C.VARIANTS[V][1]] == 'stop').mean() * 100), 1),
                        n_H1=len(h1), H1=round(float(h1[NET].mean()), 4) if len(h1) >= 15 else np.nan,
                        n_H2=len(h2), H2=round(float(h2[NET].mean()), 4) if len(h2) >= 15 else np.nan,
                        both_neg=bool(len(h1) >= 15 and len(h2) >= 15
                                      and h1[NET].mean() < 0 and h2[NET].mean() < 0)))
    return out


def md_table(rows, cols, hdr):
    L = ['| ' + ' | '.join(hdr) + ' |', '|' + '|'.join(['---'] * len(hdr)) + '|']
    for r in rows:
        L.append('| ' + ' | '.join('' if (isinstance(r.get(c), float) and np.isnan(r.get(c)))
                                   else str(r.get(c, '')) for c in cols) + ' |')
    return L


def run(pop, L, cells):
    d = C.load(pop)
    wk = C.weeks_of(pop)
    d['log_pm'] = np.log10(d.pm_any.clip(lower=1)) if 'pm_any' in d.columns else np.nan
    x = C.scoreable(d, V, floor=True)
    st, t = C.book_stats(x, 'TRAIN', wk, V)
    pop_tr = x[x.split == 'TRAIN'].copy()
    t = t.copy()
    for f in (t, pop_tr):
        f['era'] = era(f.day.values)
    L += [f'# universe {pop}', '',
          f'TRAIN book: n {st["n"]}, {st["tpw"]}/wk, net R **{st["meanR"]:+.4f}** (t {st["t"]:.2f}), '
          f'gross {st["grossR"]:+.4f}, WR {st["WR"]:.1f}%, stop {st["stopP"]:.1f}%, '
          f'weekly {st["wkR"]:+.2f}R, green {st["green"]:.2f}, worst week {st["worstWk"]:+.1f}R, '
          f'MDD {st["mdd"]:+.1f}R.  TRAIN population (unbooked): {len(pop_tr)} rows, '
          f'net R {pop_tr[NET].mean():+.4f}.', '']

    # ---------------------------------------------------------------- 1.0 availability
    L += ['## 1.0 availability audit (coverage < 100%)', '',
          '| feature | scope | TRAIN cov | 0930_0935 | 0935_1000 | 1000_1100 | 1100_1300 | 1300_1401 | '
          'mean net R known | unknown |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for feat, knowncol in (('has_news', 'news_known'), ('pm_any', 'pm_known'),
                           ('spy_ret', None), ('prev_day_range_pct', None), ('adv20', None),
                           ('regime', None)):
        for scope, f in (('book', t), ('pop', pop_tr)):
            known = f[knowncol] == 1 if knowncol else f[feat].notna()
            cov = known.mean()
            by = [f'{known[f.band == b].mean():.2f}' if (f.band == b).any() else '-'
                  for _, _, b in C.BANDS]
            mk = f.loc[known, NET].mean() if known.any() else np.nan
            mu = f.loc[~known, NET].mean() if (~known).any() else np.nan
            L.append(f'| {feat} | {scope} | {cov:.3f} | ' + ' | '.join(by) +
                     f' | {mk:+.4f} | {"" if np.isnan(mu) else f"{mu:+.4f}"} |')
    L.append('')

    # ---------------------------------------------------------------- 1.1 concentration
    byday = t.groupby('day')[NET].agg(['size', 'sum']).rename(columns={'size': 'n', 'sum': 'R'})
    byday = byday.sort_values('R')
    tot = byday.R.sum()
    neg = byday[byday.R < 0]
    nd = len(byday)
    w5, w10 = max(int(nd * 0.05), 1), max(int(nd * 0.10), 1)
    byweek = t.groupby('wk')[NET].sum().reindex(wk['TRAIN']).fillna(0.0)
    L += ['## 1.1 concentration', '',
          f'- {nd} trading days with a booked trade; total **{tot:+.1f}R**; '
          f'{len(neg)} red days summing **{neg.R.sum():+.1f}R**, '
          f'{nd-len(neg)} green summing {byday[byday.R >= 0].R.sum():+.1f}R.',
          f'- worst 5% of days ({w5}) = **{byday.R.head(w5).sum():+.1f}R** '
          f'({byday.R.head(w5).sum()/neg.R.sum():.0%} of all loss); '
          f'worst 10% ({w10}) = **{byday.R.head(w10).sum():+.1f}R** '
          f'({byday.R.head(w10).sum()/neg.R.sum():.0%} of all loss).',
          f'- best 5% of days = {byday.R.tail(w5).sum():+.1f}R '
          f'({byday.R.tail(w5).sum()/tot:.0%} of the total) — the book is two-tailed, not one.',
          f'- weeks: {len(byweek)}, green {float((byweek > 0).mean()):.2f}, '
          f'worst {byweek.min():+.1f}R, best {byweek.max():+.1f}R, '
          f'worst 5 weeks {byweek.nsmallest(5).sum():+.1f}R.', '']
    df = C._dayfeat().set_index('day')
    L += ['### the worst 20 days (market context = SPY/IWM open->close and gap, from `day_features.csv`)', '',
          '| day | n | R | exit mix | SPY gap | SPY o->c | IWM o->c | SPY prev | regime | worst trade |',
          '|---|---:|---:|---|---:|---:|---:|---:|---|---|']
    for day, r in byday.head(20).iterrows():
        z = t[t.day == day]
        mix = ', '.join(f'{k}:{v}' for k, v in z[C.VARIANTS[V][1]].value_counts().items())
        f = df.loc[day] if day in df.index else {}
        wt = z.loc[z[NET].idxmin()]
        g = lambda k: ('' if k not in f or pd.isna(f[k]) else f'{f[k]:+.2f}')                # noqa: E731
        L.append(f'| {day} | {int(r.n)} | {r.R:+.1f} | {mix} | {g("spy_gap")} | {g("spy_co")} | '
                 f'{g("iwm_co")} | {g("spy_prev_ret")} | {f.get("regime","")} | '
                 f'{wt.symbol} {wt[NET]:+.2f} |')
    L.append('')

    # ---------------------------------------------------------------- 1.2 trade anatomy
    L += ['## 1.2 trade anatomy — losers vs winners on every causal feature', '', '**booked TRAIN trades**', '',
          '| feature | losers mean | winners mean | diff | t |', '|---|---:|---:|---:|---:|']
    lw = []
    for feat, _ in NUM_FEATS:
        if feat not in t.columns:
            continue
        a = t.loc[t[NET] <= 0, feat].dropna()
        b = t.loc[t[NET] > 0, feat].dropna()
        if len(a) < 30 or len(b) < 30:
            continue
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        tt = (b.mean() - a.mean()) / se if se > 0 else 0.0
        lw.append(dict(feat=feat, lose=round(float(a.mean()), 4), win=round(float(b.mean()), 4),
                       diff=round(float(b.mean() - a.mean()), 4), t=round(float(tt), 2)))
        L.append(f'| {feat} | {a.mean():+.4f} | {b.mean():+.4f} | {b.mean()-a.mean():+.4f} | {tt:+.2f} |')
    pd.DataFrame(lw).to_csv(f'{H}/h1_loserwinner_{pop}.csv', index=False)
    L.append('')

    rows = []
    for feat, spec in NUM_FEATS:
        e = edges_for(pop_tr, feat, spec)
        if e is None:
            continue
        rows += bucket_table(t, feat, e, 'book')
        rows += bucket_table(pop_tr, feat, e, 'pop')
    for feat in CAT_FEATS:
        if feat not in t.columns:
            continue
        for lbl, f in (('book', t), ('pop', pop_tr)):
            g = f.assign(_b=f[feat].astype(str))
            for k, xx in g.groupby('_b', observed=True):
                if len(xx) < 20:
                    continue
                h1, h2 = xx[xx.era == 'H1'], xx[xx.era == 'H2']
                rows.append(dict(scope=lbl, feat=feat, bucket=k, n=len(xx),
                                 meanR=round(float(xx[NET].mean()), 4), share=round(len(xx) / len(f), 3),
                                 WR=round(float((xx[NET] > 0).mean() * 100), 1),
                                 stopP=round(float((xx[C.VARIANTS[V][1]] == 'stop').mean() * 100), 1),
                                 n_H1=len(h1), H1=round(float(h1[NET].mean()), 4) if len(h1) >= 15 else np.nan,
                                 n_H2=len(h2), H2=round(float(h2[NET].mean()), 4) if len(h2) >= 15 else np.nan,
                                 both_neg=bool(len(h1) >= 15 and len(h2) >= 15
                                               and h1[NET].mean() < 0 and h2[NET].mean() < 0)))
    B = pd.DataFrame(rows)
    B.to_csv(f'{H}/h1_buckets_{pop}.csv', index=False)
    cols = ['feat', 'bucket', 'n', 'share', 'meanR', 'WR', 'stopP', 'n_H1', 'H1', 'n_H2', 'H2', 'both_neg']
    hdr = ['feature', 'bucket', 'n', 'share', 'mean net R', 'WR%', 'stop%', 'n H1', 'H1', 'n H2', 'H2', 'both neg']
    L += ['### bucket mean net R — BOOKED TRAIN trades (H1 = 2025-01..06, H2 = 2025-07..12)', '']
    L += md_table(B[B.scope == 'book'].to_dict('records'), cols, hdr) + ['']
    L += ['### bucket mean net R — the whole TRAIN POPULATION (a filter must not be a book artefact)', '']
    L += md_table(B[B.scope == 'pop'].to_dict('records'), cols, hdr) + ['']
    cells['buckets'] = cells.get('buckets', 0) + len(B)

    # ---------------------------------------------------------------- 1.3 path anatomy
    whyc = C.VARIANTS[V][1]
    t['mins_held'] = t[C.VARIANTS[V][2]] - t.next_entry_m
    lose, win = t[t[NET] <= 0], t[t[NET] > 0]
    stp = t[t[whyc] == 'stop']
    L += ['## 1.3 path anatomy', '',
          '| cohort | n | mean net R | median minutes held | mean MAE% | mean MFE (R) | '
          'MFE>=0.5R | MFE>=1R | MFE>=2R |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for nm, f in (('all booked', t), ('losers', lose), ('winners', win), ('stopped', stp),
                  ('eod exits', t[t[whyc] == 'eod'])):
        if not len(f):
            continue
        L.append(f'| {nm} | {len(f)} | {f[NET].mean():+.4f} | {f.mins_held.median():.0f} | '
                 f'{f.next_mae_pct.mean():.2f} | {f.next_mfe_r.mean():+.2f} | '
                 f'{(f.next_mfe_r >= 0.5).mean():.2f} | {(f.next_mfe_r >= 1).mean():.2f} | '
                 f'{(f.next_mfe_r >= 2).mean():.2f} |')
    L += ['',
          f'- **the stopped cohort had money on the table**: {(stp.next_mfe_r >= 0.5).mean():.0%} reached '
          f'+0.5R and {(stp.next_mfe_r >= 1).mean():.0%} reached +1R before the stop; median minutes to '
          f'the stop **{stp.mins_held.median():.0f}**, 25th pct {stp.mins_held.quantile(.25):.0f}, '
          f'75th {stp.mins_held.quantile(.75):.0f}.',
          f'- a naive breakeven-after-+1R rule would therefore convert at most '
          f'{(stp.next_mfe_r >= 1).mean():.0%} of stops into ~0R and would cap every winner that '
          f'dipped after +1R — Stage A measured that exact variant as NEGATIVE, so the SHAPE of the '
          f'filter must be entry-side, not exit-side.', '',
          '### minutes-to-stop distribution (stopped trades)', '',
          '| <=5 min | 6-15 | 16-30 | 31-60 | 61-120 | >120 |', '|---:|---:|---:|---:|---:|---:|']
    q = pd.cut(stp.mins_held, [-1, 5, 15, 30, 60, 120, 1e9])
    vc = q.value_counts(normalize=True).sort_index()
    L.append('| ' + ' | '.join(f'{v:.2f}' for v in vc.values) + ' |')
    L += ['', '### mean net R by MFE bucket (booked TRAIN) — what a trade that never runs is worth', '',
          '| MFE bucket | n | mean net R | stop% |', '|---|---:|---:|---:|']
    mb = pd.cut(t.next_mfe_r, [-1e9, 0, 0.5, 1, 2, 1e9])
    for k, xx in t.groupby(mb, observed=True):
        L.append(f'| {k} | {len(xx)} | {xx[NET].mean():+.4f} | {(xx[whyc]=="stop").mean()*100:.1f} |')
    L.append('')

    # ---------------------------------------------------------------- 1.4 era table
    L += ['## 1.4 era consistency inside TRAIN', '',
          '| half | n | mean net R | t | WR% | stop% | weekly R |', '|---|---:|---:|---:|---:|---:|---:|']
    for e in ('H1', 'H2'):
        f = t[t.era == e]
        sd = f[NET].std(ddof=1)
        L.append(f'| {e} | {len(f)} | {f[NET].mean():+.4f} | '
                 f'{f[NET].mean()/(sd/np.sqrt(len(f))):.2f} | {(f[NET]>0).mean()*100:.1f} | '
                 f'{(f[whyc]=="stop").mean()*100:.1f} | {f.groupby("wk")[NET].sum().mean():+.2f} |')
    L += ['', '### candidate veto buckets — negative in BOTH TRAIN halves, booked AND population', '',
          '| feature | bucket | book n | book R | book H1 | book H2 | pop n | pop R | pop H1 | pop H2 |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    bk = B[(B.scope == 'book') & B.both_neg]
    pp = B[B.scope == 'pop'].drop_duplicates(['feat', 'bucket']).set_index(['feat', 'bucket'])
    cand = []
    for r in bk.itertuples():
        key = (r.feat, r.bucket)
        p = pp.loc[key] if key in pp.index else None
        g = (lambda c: float(p[c]) if p is not None and not pd.isna(p[c]) else np.nan)      # noqa: E731
        cand.append(dict(feat=r.feat, bucket=r.bucket, n=r.n, share=r.share, meanR=r.meanR,
                         H1=r.H1, H2=r.H2, stopP=r.stopP,
                         pop_n=(int(p.n) if p is not None else 0), pop_R=g('meanR'),
                         pop_H1=g('H1'), pop_H2=g('H2'),
                         pop_both_neg=bool(p.both_neg) if p is not None else False))
        fm = lambda v: '' if (v is None or (isinstance(v, float) and np.isnan(v))) else f'{v:+.4f}'  # noqa: E731
        L.append(f'| {r.feat} | {r.bucket} | {r.n} | {r.meanR:+.4f} | {r.H1:+.4f} | {r.H2:+.4f} | '
                 f'{int(p.n) if p is not None else 0} | {fm(g("meanR"))} | {fm(g("H1"))} | {fm(g("H2"))} |')
    pd.DataFrame(cand).to_csv(f'{H}/h1_candidates_{pop}.csv', index=False)
    nb = sum(1 for c in cand if c['pop_both_neg'])
    L += ['', f'{len(cand)} buckets are negative in BOTH TRAIN halves on the booked trades (the ORB '
              f'veto rule); {nb} of them are also negative in both halves of the whole TRAIN population. '
              f'They are the only inputs to step 2.', '']
    t.to_csv(f'{H}/h1_booked_train_{pop}.csv', index=False)
    return L


def main():
    L = ['# Stage H / F6 — step 1: the anatomy of the losers (TRAIN only)', '',
         f'generated {time.strftime("%Y-%m-%d %H:%M:%S")}', '',
         'Book: F6 red-to-green, next-open fill, hold to 15:55 with the touch stop, contract (c), '
         'run_book(12,4), `range_so_far_pct >= 5`, entries 09:30-14:01, price >= $5, R >= 1% of price. '
         'VAL and TEST are not read here.', '']
    cells = {}
    for pop in ('Q', 'P'):
        L = run(pop, L, cells)
    open(f'{H}/h1_anatomy.md', 'w').write('\n'.join(L))
    print(f'wrote {H}/h1_anatomy.md  ({len(L)} lines)  cells {cells}')


if __name__ == '__main__':
    main()
