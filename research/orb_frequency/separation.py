#!/usr/bin/env python3
"""PART 1 — the ORB gate separation map (PREREG §4).

For every gate, at ITS OWN position in the live cascade, on the population that
actually reaches it:  sep = mean R(kept) - mean R(rejected), per year and
pooled, with n on both sides and a Welch t.  Purely descriptive: no rule is
drawn from this file (PREREG §4).

Also prints BF's diagnostic — median position size and median dollar risk on
the kept vs the rejected side — because the bull flag's broken gate hid behind
the sizer already shrinking the names it was cutting.

Every gate is evaluated through the SHIPPED helper with the value that is in
orb.yaml right now.  Nothing is refit.
"""
from __future__ import annotations

import csv
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv                       # noqa: E402
from trading.orb_pdr_veto import pdr_veto_applies              # noqa: E402
from trading.orb_g1_veto import g1_reject                      # noqa: E402
from trading.orb_range_size_veto import range_size_veto_applies  # noqa: E402
from trading.orb_catalyst_veto import (DEFAULT_MIN_COHORT,     # noqa: E402
                                       anchor_cohort_counts,
                                       catalyst_veto_applies)
from trading.orb_asset_class import (DEFAULT_CLASS_MAP,        # noqa: E402
                                     load_class_map, underlying_anchor)
from trading.orb_correlation import symbol_family, symbol_super_group  # noqa: E402

D = f'{ROOT}/research/orb_frequency'
FEATURES = f'{ROOT}/analysis_results/orb_features_20260916_2053.csv'
DUMP = f'{ROOT}/research/fuckup_audit/Q_fill/dump_measured.csv'
RANKED = f'{D}/ranked_all.csv'

CFG = dict(threshold=0.012081536791, pdr_min=11.0, g1_rv20=7.106,
           g1_pdr=9.226, g1_short_hist=False, rs_min=2.221,
           min_stop_pct=1.0, n_slots=8)
Q_ORDER = {'Q5': 0, 'Q4': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}


def build_ranked():
    """One pipeline run with every selection gate OPEN, dumping the ranked list."""
    if os.path.exists(RANKED):
        return
    e = dict(os.environ)
    e.update({'ORB_BT_FEATURES_CSV': FEATURES, 'ORB_BT_RESIM_CACHE': DUMP,
              'ORB_BT_RISK': '375', 'ORB_BT_N': '8',
              'ORB_BT_ACCOUNT': repr(3333.333333333333 * 8),
              'ORB_BT_THRESHOLD': '-99', 'ORB_SKIP_Q1': '0',
              'ORB_BT_DUMP_RANKED': RANKED,
              'ORB_BT_BOOK_OUT': f'{D}/book_rankdump.csv',
              'ORB_BT_MONTHLY_OUT': f'{D}/monthly_rankdump.csv'})
    with open(f'{D}/log_rankdump.txt', 'w') as fh:
        rc = subprocess.call(['nice', '-n', '10', 'python3', '-u',
                              'study_orb_pipeline_static_lock.py'],
                             env=e, stdout=fh, stderr=subprocess.STDOUT)
    if rc:
        raise SystemExit('ranked dump failed — see log_rankdump.txt')


def load() -> pd.DataFrame:
    d = read_orb_csv(RANKED)
    d['date'] = pd.to_datetime(d['date'])
    d['R'] = d['pnl_pct'].astype(float) / d['range_size_pct'].clip(lower=CFG['min_stop_pct'])
    d['yr'] = d['date'].dt.year
    d['risk$'] = d['_rp_position'] * d['range_size_pct'].clip(lower=CFG['min_stop_pct']) / 100.0
    return d


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float((a.mean() - b.mean()) / np.sqrt(va + vb)) if va + vb > 0 else float('nan')


ROWS = []


def record(name: str, pop: pd.DataFrame, keep: pd.Series, note: str = ''):
    """One separation row.  `keep` is a boolean mask over `pop`."""
    k, r = pop[keep], pop[~keep]
    out = {'gate': name, 'n_kept': len(k), 'n_rej': len(r)}
    for yr in (2025, 2026):
        ky, ry = k[k['yr'] == yr]['R'].values, r[r['yr'] == yr]['R'].values
        out[f'sep_{yr}'] = (ky.mean() - ry.mean()) if len(ky) and len(ry) else np.nan
        out[f't_{yr}'] = welch_t(ky, ry)
    ka, ra = k['R'].values, r['R'].values
    out['sep_pooled'] = (ka.mean() - ra.mean()) if len(ka) and len(ra) else np.nan
    out['R_kept'] = ka.mean() if len(ka) else np.nan
    out['R_rej'] = ra.mean() if len(ra) else np.nan
    out['t'] = welch_t(ka, ra)
    out['med_pos_kept'] = k['_rp_position'].median() if len(k) else np.nan
    out['med_pos_rej'] = r['_rp_position'].median() if len(r) else np.nan
    out['med_risk_kept'] = k['risk$'].median() if len(k) else np.nan
    out['med_risk_rej'] = r['risk$'].median() if len(r) else np.nan
    out['note'] = note
    ROWS.append(out)
    return k


def rank_within_day(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d['_qr'] = d['_quintile'].map(Q_ORDER)
    d = d.sort_values(['date', '_qr', '_composite'], ascending=[True, True, False],
                      kind='mergesort')
    d['_r'] = d.groupby('date').cumcount() + 1
    return d


def dedup_mask(d: pd.DataFrame) -> pd.Series:
    """True = survives family / super-group dedup (rank order within the day)."""
    keep = []
    for _day, g in d.groupby('date', sort=False):
        fams, sups = set(), set()
        for s in g['symbol']:
            f, sp = symbol_family(s), symbol_super_group(s)
            if (f and f in fams) or (sp and sp in sups):
                keep.append(False)
                continue
            if f:
                fams.add(f)
            if sp:
                sups.add(sp)
            keep.append(True)
    return pd.Series(keep, index=d.index)


def catalyst_mask(sel: pd.DataFrame, full: pd.DataFrame) -> pd.Series:
    """True = vetoed (newsless and alone).  Same construction as the pipeline."""
    names = {}
    try:
        with open(DEFAULT_CLASS_MAP, newline='') as fh:
            for row in csv.DictReader(fh):
                names[row['symbol']] = row.get('name', '')
    except Exception as exc:                                   # pragma: no cover
        print(f'catalyst: class-map names unavailable ({exc}) — fail-open')
    cmap = load_class_map()
    anchors = {s: underlying_anchor(s, names.get(s), cmap) for s in set(full['symbol'])}
    day_anchor = full.assign(_a=full['symbol'].map(anchors))
    cohorts = {day: anchor_cohort_counts(g['_a'])
               for day, g in day_anchor.groupby(day_anchor['date'].dt.strftime('%Y-%m-%d'))}
    raw_news = {}
    for p in sorted(glob.glob('data/research/orb_news_catalyst_*.csv')):
        for _, r in read_orb_csv(p).iterrows():
            raw_news[(r['symbol'], r['day'])] = (r['n_articles'] or 0) > 0
    sd = sel['date'].dt.strftime('%Y-%m-%d')
    hn = [raw_news.get((s, d)) for s, d in zip(sel['symbol'], sd)]
    a = sel['symbol'].map(anchors)
    return pd.Series([catalyst_veto_applies(h, an, cohorts.get(d, {}), DEFAULT_MIN_COHORT)
                      for h, an, d in zip(hn, a, sd)], index=sel.index)


def main():
    build_ranked()
    d = load()
    print(f'population: {len(d)} candidates, {d.date.nunique()} days, '
          f'{d.entered.sum()} fills')

    # ---- S0 composite threshold (on the whole candidate universe) ---------
    pop = d
    keep = pop['_composite'] >= CFG['threshold']
    pop = record('S0 composite >= 0.012082', pop, keep)

    # ---- S1 Q1 quintile filter -------------------------------------------
    pop = record('S1 Q1 filter (drop bottom quintile)', pop,
                 pop['_quintile'] != 'Q1')

    # ---- S2 the quintile ladder itself (descriptive, per-quintile R) ------
    qtab = (pop.groupby('_quintile')['R'].agg(['size', 'mean'])
            .reindex(['Q5', 'Q4', 'Q3', 'Q2']))
    print('\nS2 per-quintile R (post-threshold, post-Q1):')
    print(qtab.to_string())

    # ---- S3 family / super-group dedup -----------------------------------
    pop = rank_within_day(pop)
    pop = record('S3 family/super-group dedup', pop, dedup_mask(pop))

    # ---- S4 top-K slot cut (rank <= 8) -----------------------------------
    pop = rank_within_day(pop)          # re-rank after dedup
    ranked = pop.copy()
    pop = record('S4 top-8 slot cut (rank <= 8)', pop, pop['_r'] <= CFG['n_slots'])
    # marginal rank bands, descriptive
    print('\nS4 R by rank band (post-dedup):')
    bands = pd.cut(ranked['_r'], [0, 3, 4, 6, 8, 12, 16, 10_000],
                   labels=['1-3', '4', '5-6', '7-8', '9-12', '13-16', '17+'])
    print(ranked.groupby(bands, observed=False)['R'].agg(['size', 'mean']).to_string())

    # ---- S5 PDR veto ------------------------------------------------------
    pop = record('S5 PDR veto (prev_day_range > 11.0)', pop,
                 ~pop['prev_day_range_pct'].apply(
                     lambda v: pdr_veto_applies(None if pd.isna(v) else float(v),
                                                CFG['pdr_min'])))

    # ---- S6 G1 volatility fingerprint ------------------------------------
    pop = record('S6 G1 fingerprint (rv20>=7.106 & pdr>=9.226)', pop,
                 ~pd.Series([g1_reject(rv, p, CFG['g1_rv20'], CFG['g1_pdr'],
                                       short_history_veto=CFG['g1_short_hist']) is not None
                             for rv, p in zip(pop['return_volatility_20d'],
                                              pop['prev_day_range_pct'])],
                            index=pop.index))

    # ---- S7 range-size veto ----------------------------------------------
    pop = record('S7 range-size veto (range_size_pct > 2.221)', pop,
                 ~pop['range_size_pct'].apply(
                     lambda v: range_size_veto_applies(v, CFG['rs_min'])))

    # ---- S8 catalyst veto -------------------------------------------------
    pop = record('S8 catalyst veto (news or cohort>=2)', pop,
                 ~catalyst_mask(pop, d))

    # ---- S11 whole stack --------------------------------------------------
    picked = set(zip(pop['symbol'], pop['date']))
    allc = d.copy()
    record('S11 WHOLE STACK picked vs rejected', allc,
           pd.Series([(s, t) in picked for s, t in zip(allc['symbol'], allc['date'])],
                     index=allc.index))

    out = pd.DataFrame(ROWS)
    out.to_csv(f'{D}/separation.csv', index=False)
    pd.set_option('display.width', 250)
    print('\n=== SEPARATION MAP (R per pick; kept minus rejected) ===')
    print(out[['gate', 'n_kept', 'n_rej', 'R_kept', 'R_rej', 'sep_2025', 't_2025',
               'sep_2026', 't_2026', 'sep_pooled', 't', 'med_risk_kept',
               'med_risk_rej']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
