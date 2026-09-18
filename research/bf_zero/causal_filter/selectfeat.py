#!/usr/bin/env python3
"""CAUSAL_FILTER step 3 — TRAIN-only loser anatomy + the pre-registered selection rule.

PREREG rule (fixed 2026-09-18 15:10 UTC, before any run):
  keep a feature if (a) the TRAIN spread between its best and worst tercile is >= 0.20R with
  n >= 300 in each, and (b) the sign of that spread is the same in BOTH halves of TRAIN
  (H1 2025 vs H2 2025). At most FIVE survive; ties broken by n.

The `cohort` column is END-OF-DAY information. `assert_no_cohort` proves it is not among the
scored features; it appears only in the diagnostic "cohort mix" column of the anatomy tables.

Outputs: causal_filter/anatomy.md, causal_filter/selection.json
"""
import json, os, sys
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
D = f'{ROOT}/research/bf_zero/causal_filter'

# The PREREG's candidate list, verbatim. `pull_len` is constant by construction for this family
# (the spec fixes K = 5 consolidation bars) and is reported as degenerate, not scored.
FEATS = ['gap_pct', 'prev_range_pct', 'dist_20d_high_pct', 'bar_vol_x', 'above_vwap', 'spy_5m_ret',
         'spy_range3', 'dist_open_pct', 'rv_clock', 'rv_profile', 'drive_min', 'n_prior',
         'is_wrapper', 'coh_by_t', 'entry_m', 'price', 'has_news']
BANNED = {'cohort', 'rr', 'why', 'exit_m', 'target', 'day_vol'}
MIN_SPREAD, MIN_N = 0.20, 300


def assert_no_cohort(feats):
    """The cohort is end-of-day information: it may never be a feature, a label or a rule input."""
    bad = sorted(set(feats) & BANNED)
    assert not bad, f'END-OF-DAY column(s) in the feature set: {bad}'


def groups(s, k=3):
    """Terciles for a continuous feature; the distinct values for a low-cardinality one."""
    vals = s.dropna().unique()
    if len(vals) <= k:
        return s.astype('object').where(s.notna()), 'cat'
    try:
        return pd.qcut(s, k, labels=[f'T{i + 1}' for i in range(k)], duplicates='drop'), 'ter'
    except ValueError:
        # too many ties to cut into k groups — the feature is degenerate on this population
        return s.astype('object').where(s.notna()), 'tied'


def table(d, f, k):
    g, kind = groups(d[f], k)
    t = d.assign(_g=g).dropna(subset=['_g']).groupby('_g', observed=True).agg(
        n=('rr', 'size'), meanR=('rr', 'mean'), WR=('rr', lambda x: (x > 0).mean() * 100),
        cache_pct=('cohort', lambda x: (x == 'cache').mean() * 100))
    return t.round(3), kind


def main():
    assert_no_cohort(FEATS)
    c = pd.read_csv(f'{D}/features.csv', dtype={'symbol': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    tr = c[c.split == 'TRAIN']
    out = ['# Causal-filter — TRAIN loser anatomy and the pre-registered selection',
           '',
           f'TRAIN signals {len(tr)} · mean R {tr.rr.mean():+.3f} · WR {(tr.rr > 0).mean() * 100:.1f}% · '
           f'cohort mix cache {(tr.cohort == "cache").mean() * 100:.1f}%',
           '',
           '## Loser anatomy — deciles (or categories) on TRAIN; `cache %` is DIAGNOSTIC only',
           '']
    sel = []
    for f in FEATS:
        if f not in c.columns or tr[f].notna().sum() == 0:
            out += [f'### {f}', '', 'not available on this population — not scored.', '']
            continue
        d10, kind = table(tr, f, 10)
        cov = tr[f].notna().mean() * 100
        out += [f'### {f}  (coverage {cov:.1f}%, {kind})', '', d10.to_markdown(), '']
        d3, k3 = table(tr, f, 3)
        if len(d3) < 2 or d3.n.min() < MIN_N:
            top = tr[f].value_counts(normalize=True).head(2).round(3).to_dict()
            out += [f'*DEGENERATE on this population ({k3}): {len(d3)} group(s) with n >= 1, smallest '
                    f'{int(d3.n.min()) if len(d3) else 0} — fails the n >= {MIN_N} rule. Modal values '
                    f'{top}. Not scored.*', '']
            continue
        best, worst = d3.meanR.idxmax(), d3.meanR.idxmin()
        spread = float(d3.meanR.max() - d3.meanR.min())
        # sign agreement across the two halves of TRAIN, same group edges
        g3, _ = groups(tr[f], 3)
        sub = tr.assign(_g=g3).dropna(subset=['_g'])
        halves = {}
        for hlf in ('H1', 'H2'):
            hh = sub[sub.half == hlf]
            mb = hh[hh._g == best].rr
            mw = hh[hh._g == worst].rr
            halves[hlf] = float(mb.mean() - mw.mean()) if len(mb) and len(mw) else np.nan
        agree = (halves['H1'] > 0) and (halves['H2'] > 0)
        ok = spread >= MIN_SPREAD and agree
        out += [f'*terciles: best {best} {d3.meanR[best]:+.3f} (n {int(d3.n[best])}) · worst {worst} '
                f'{d3.meanR[worst]:+.3f} (n {int(d3.n[worst])}) · spread {spread:.3f}R · '
                f'H1 {halves["H1"]:+.3f} H2 {halves["H2"]:+.3f} · '
                f'{"KEEP" if ok else "reject"}*', '']
        sel.append(dict(feat=f, spread=round(spread, 4), best=str(best), worst=str(worst),
                        n_best=int(d3.n[best]), n_worst=int(d3.n[worst]),
                        h1=round(halves['H1'], 4), h2=round(halves['H2'], 4), keep=bool(ok),
                        n=int(d3.n.sum())))
    S = pd.DataFrame(sel)
    keep = S[S.keep].sort_values(['spread', 'n'], ascending=[False, False]).head(5)
    out += ['## Selection', '', S.sort_values('spread', ascending=False).to_markdown(index=False), '',
            f'**Survivors (max 5, ties by n): {list(keep.feat) if len(keep) else "NONE"}**', '']
    open(f'{D}/anatomy.md', 'w').write('\n'.join(out))
    json.dump(dict(survivors=keep.to_dict('records'), all=S.to_dict('records')),
              open(f'{D}/selection.json', 'w'), indent=1)
    print(S.sort_values('spread', ascending=False).to_string(index=False), flush=True)
    print('\nSURVIVORS:', list(keep.feat), flush=True)


if __name__ == '__main__':
    main()
