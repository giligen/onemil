#!/usr/bin/env python3
"""Owner question 2026-09-07: "why don't we train and optimize weekly on the
past 2 months and test on the rest?" — answered on the HONEST ORB dump.

Walk-forward: every Monday from 2025-03-03, the composite's z-params, the
quintile cutoffs and the adaptive mults are re-fit on the trailing window of
candidates (strictly before the week), then the week is traded with those
parameters through the same selection + veto stack as the pipeline. The
frozen variant (orb.yaml literals) must reproduce the honest book ($6,531 /
114 picks with the two 9/8 vetoes) before any refit row counts.

Variants: frozen | refit 8w | refit 13w | refit 26w | refit expanding (all
history before the week). Same exit physics (static-lock dump), same stage
sizing ($10K / 3 / $375), same vetoes.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import study_orb_pipeline_static_lock as P                       # noqa: E402
from study_orb_sizing import fit_quintile_cutoffs, assign_quintile, ADAPTIVE_MULT_MIN  # noqa: E402
from study_orb_correlation_filter import symbol_family, symbol_super_group   # noqa: E402
sys.path.insert(0, os.path.join(ROOT, 'research', 'orb_slot_recycle'))
import simulate as S                                              # noqa: E402  (vetoed(), load of news/cohorts)
import yaml                                                       # noqa: E402

DUMP = 'research/orb_veto_study/candidates_static_lock_dump.csv'
OUT = 'research/orb_refit_walkforward'
N = 3
FEATS = P.FILTER_FEATURES
Q_ORDER, Q_CAPS = P.Q_ORDER, P.Q_CAPS
cfg = yaml.safe_load(open('orb.yaml'))
THRESH = float(cfg['filter']['threshold'])
FROZEN = ({f: {'mean': float(cfg['filter']['features'][f]['mean']), 'std': float(cfg['filter']['features'][f]['std']),
               'sign': int(cfg['filter']['features'][f]['sign'])} for f, _ in FEATS},
          [float(x) for x in cfg['quintile_cutoffs']],
          {q: float(cfg['adaptive_mults'][q]) for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5')})


def fit(train: pd.DataFrame):
    """The pipeline's refit path, on an arbitrary window."""
    params = P.fit_z_params(train, FEATS)
    comp = P.composite_score(train, params)
    tk = train[comp >= THRESH].copy(); tk['_c'] = comp[comp >= THRESH]
    if len(tk) < 25:
        return None
    cutoffs = fit_quintile_cutoffs(tk['_c'])
    tk['_q'] = assign_quintile(tk['_c'], cutoffs)
    avg = float(tk['_rp_pnl'].mean())
    mults = {}
    for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5'):
        sub = tk[tk['_q'] == q]
        m = float(sub['_rp_pnl'].mean()) / avg if len(sub) and avg else 1.0
        mults[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], m))
    return params, cutoffs, mults


def trade_week(dw: pd.DataFrame, params, cutoffs, mults, cohorts, news):
    """Select + veto the week's days with given parameters; return per-pick rows."""
    d = dw.copy()
    d['_composite'] = P.composite_score(d, params)
    d = d[d['_composite'] >= THRESH].copy()
    d['_quintile'] = assign_quintile(d['_composite'], cutoffs)
    d = d[d['_quintile'] != 'Q1']                       # skip_q1 (live default)
    d['_q_rank'] = d['_quintile'].map(Q_ORDER)
    rows = []
    for day, g in d.groupby('date'):
        g = g.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        seen_f, seen_s, picks = set(), set(), []
        for _, r in g.iterrows():                       # iterrows: itertuples renames '_'-prefixed columns
            f, s_ = symbol_family(r['symbol']), symbol_super_group(r['symbol'])
            if (f and f in seen_f) or (s_ and s_ in seen_s):
                continue
            if f: seen_f.add(f)
            if s_: seen_s.add(s_)
            picks.append(r)
            if len(picks) >= N:
                break
        for r in picks:
            row = r.copy(); row['date'] = pd.Timestamp(day).strftime('%Y-%m-%d')
            v = S.vetoed(row, cohorts, news)
            pnl = 0.0 if v else float(r['_rp_pnl']) * mults[r['_quintile']]
            rows.append(dict(date=day, symbol=r['symbol'], pnl=pnl, filled=0 if v else int(r['entered']), veto=v or ''))
    return rows


def run(df, cohorts, news, mode, weeks):
    rows = []
    for w0 in weeks:
        w1 = w0 + pd.Timedelta(days=6)
        if mode == 'frozen':
            prm = FROZEN
        else:
            # mode grammar: '<window>' | '<window>_zc' (refit z+cutoffs, FROZEN mults)
            #               | '<window>_m' (FROZEN z+cutoffs, refit mults); window = '8w' | '26w' | 'expanding'
            base, _, part = mode.partition('_')
            lb0 = pd.Timestamp('2000-01-01') if base == 'expanding' else w0 - pd.Timedelta(weeks=int(base[:-1]))
            train = df[(df['date'] >= lb0) & (df['date'] < w0)]
            prm = fit(train)
            if prm is None:
                continue                                  # too little history: no trades this week
            if part == 'zc':
                prm = (prm[0], prm[1], FROZEN[2])
            elif part == 'm':
                prm = (FROZEN[0], FROZEN[1], prm[2])
        dw = df[(df['date'] >= w0) & (df['date'] <= w1)]
        if len(dw):
            rows.extend(trade_week(dw, *prm, cohorts, news))
    b = pd.DataFrame(rows); b['date'] = pd.to_datetime(b['date'])
    months = pd.period_range('2025-03', '2026-09', freq='M')
    m = b.groupby(b['date'].dt.to_period('M')).pnl.sum().reindex(months).fillna(0)
    c = m.cumsum()
    era = lambda a, z: round(b[(b.date >= a) & (b.date <= z)].pnl.sum())
    return dict(mode=mode, picks=len(b), fills=int(b.filled.sum()), total=round(m.sum()), mdd=round((c - c.cummax()).min()),
                red=int((m < 0).sum()), worst=round(m.min()), e25=era('2025-01-01', '2025-12-31'), e2026=era('2026-01-01', '2026-12-31')), b


def main():
    df = pd.read_csv(DUMP, low_memory=False); df['date'] = pd.to_datetime(df['date'])
    anchors, cohorts, news = S.load_catalyst_inputs()
    df['_anchor'] = df['symbol'].map(anchors)
    _freq = os.environ.get('REFIT_WEEK_FREQ', 'W-MON')   # week-phase robustness: W-WED etc.
    weeks = pd.date_range('2025-03-03', '2026-09-04', freq=_freq)
    out = []
    modes = sys.argv[1:] or ['frozen', '8w', '13w', '26w', 'expanding']
    for mode in modes:
        s, b = run(df, cohorts, news, mode, weeks)
        b.to_csv(f'{OUT}/{mode}_book.csv', index=False)
        print(s, flush=True); out.append(s)
    T = pd.DataFrame(out); T.to_csv(f'{OUT}/summary' + ('_' + '_'.join(modes) if sys.argv[1:] else '') + ('' if _freq == 'W-MON' else '_' + _freq) + '.csv', index=False)
    print('\n' + T.to_string(index=False))


if __name__ == '__main__':
    main()
