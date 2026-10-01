#!/usr/bin/env python3
"""Cell 1,693 addendum: the union with PER-SLICE exits actually applied.

Amendment 1's own words: "the union uses per-slice exits where robust,
production's elsewhere." The first pass's "union_robust" accidentally
equaled "production alone" because both traced back to the SAME
(already-swapped) DataFrame -- there was no distinct "production as-is"
baseline to diff against. This script builds both series explicitly and
reports the paired delta directly, plus the direction-B exit assignment as
a robustness check on the assignment itself.

Reuses research/orb_freq/1693_pool_exits.py (imported as a module, UNCHANGED
-- no bars are re-walked, no new reconstruction, pure re-aggregation of the
per-fill exit columns that script already computed).

Usage: nice -n 10 python3 research/orb_freq/1693b_per_slice_union.py
"""
import importlib.util
import logging
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('c1693_main', os.path.join(HERE, '1693_pool_exits.py'))
c = importlib.util.module_from_spec(spec)
_old_argv = sys.argv
sys.argv = [sys.argv[0]]
spec.loader.exec_module(c)
sys.argv = _old_argv

LOG_PATH = os.path.join(HERE, '1693b_per_slice_union.log')
logger = logging.getLogger('cell1693b')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                     handlers=[logging.FileHandler(LOG_PATH, mode='w'), logging.StreamHandler()])

RESULT_MD = os.path.join(HERE, 'RESULT_1693.md')
UNION_CSV = os.path.join(HERE, '1693_union.csv')


def pick_exit(slice_cls, name, by):
    """by='A': among this slice's ROBUST exits, the one with the highest
    VAL2026 mean_R (direction A's own selection criterion). by='B': the one
    with the highest TRAIN2025 mean_R (direction B's)."""
    ct = slice_cls[name]
    rob = ct[ct.classification == 'robust']
    col = 'val_mean_R' if by == 'A' else 'train_mean_R'
    best = rob.sort_values(col, ascending=False).iloc[0]
    return best['exit'], float(best['train_t']), float(best['train_mean_R']), float(best['val_mean_R'])


def assign_union_R(pf_prod, slices, picks):
    """Default = E1_production for every fill; for each robust slice, lowest
    TRAIN-t precedence first so the HIGHEST-t slice is applied LAST and wins
    any overlap (the owner's stated precedence rule)."""
    col = pf_prod['E1_production'].copy()
    order = sorted(picks.items(), key=lambda kv: kv[1][1])  # ascending train_t
    src = pd.Series('production', index=pf_prod.index, dtype=object)
    for name, (exit_name, train_t, _, _) in order:
        idx = slices[name].index.intersection(pf_prod.index)
        col.loc[idx] = slices[name].loc[idx, exit_name]
        src.loc[idx] = f'{name}(t{train_t:.1f})'
    return col, src


def main():
    logger.info('=== 1693b: union with per-slice exits actually applied -- starting ===')
    store = c.CachedStore(c.f1668.BARS_DB)
    pf_prod, prod_recon, premkt_cov = c.load_production_fills(store)
    store.close()
    slices = c.make_slices(pf_prod, premkt_cov)

    slice_cls = {}
    for name, sub in slices.items():
        rt = c.score_table(sub, name)
        slice_cls[name] = c.classify_pool(rt, name)
    robust_names = [n for n, ct in slice_cls.items() if (ct.classification == 'robust').any()]
    logger.info('robust slices: %s', robust_names)

    picks_A = {n: pick_exit(slice_cls, n, 'A') for n in robust_names}
    picks_B = {n: pick_exit(slice_cls, n, 'B') for n in robust_names}
    logger.info('direction-A picks: %s', {k: v[0] for k, v in picks_A.items()})
    logger.info('direction-B picks: %s', {k: v[0] for k, v in picks_B.items()})

    idx_sets = {n: set(slices[n].index) for n in robust_names}
    overlap_lines = []
    for i, n1 in enumerate(robust_names):
        for n2 in robust_names[i + 1:]:
            ov = len(idx_sets[n1] & idx_sets[n2])
            if ov:
                overlap_lines.append(f'{n1} & {n2}: {ov} fills')
    membership_count = pd.Series(0, index=pf_prod.index)
    for n in robust_names:
        membership_count.loc[list(idx_sets[n])] += 1
    n_multi = int((membership_count >= 2).sum())
    logger.info('overlap: %d fills in >=2 robust slices; %s', n_multi, overlap_lines)

    union_R_A, src_A = assign_union_R(pf_prod, slices, picks_A)
    union_R_B, src_B = assign_union_R(pf_prod, slices, picks_B)
    logger.info('direction-A assignment source counts: %s', src_A.value_counts().to_dict())

    pf = pf_prod.copy()
    pf['prod_asis_R'] = pf_prod['E1_production']
    pf['union_R_dirA'] = union_R_A
    pf['union_R_dirB'] = union_R_B

    windows_plus_whole = dict(c.WINDOWS)
    windows_plus_whole['WHOLE_2025_2026'] = (c.TRAIN_LO, c.VAL_HI)

    rows = []
    for label, col in [('production_as_is', 'prod_asis_R'), ('union_dirA_exits', 'union_R_dirA'), ('union_dirB_exits', 'union_R_dirB')]:
        for wname, (lo, hi) in windows_plus_whole.items():
            sub = pf[(pf['date'] >= lo) & (pf['date'] <= hi)]
            vals = sub[col].dropna()
            dts = sub.loc[vals.index, 'date']
            base = c.reads_for_exit_series(vals, dts, lo, hi)
            trades = [{'date': d, 'r': v} for d, v in zip(dts, vals)]
            weekly = c.cb.build_weekly_series(trades, lo, hi)
            cycles, _ = c.cb.compute_cycles(weekly, c.STRONG_WEEK_R)
            c1 = c.cb.score_c1(cycles, 3, 6)
            c3 = c.cb.score_c3(weekly, -2, -999, 999, 999)
            n_days = (hi - lo).days + 1
            dollars_per_year = (base['n'] * base['mean_R'] * 375.0 / n_days * 365.25) if base['n'] else np.nan
            rows.append(dict(label=f'{label}/{wname}', lo=lo, hi=hi, **base,
                              max_drawdown_R=c3['mdd'], strong_week_gap_median=c1['median'],
                              strong_week_gap_p90=c1['p90'], dollars_per_year_at_375=dollars_per_year))
    rdf = pd.DataFrame(rows)

    delta_rows = []
    for label, col in [('dirA_minus_prod', 'union_R_dirA'), ('dirB_minus_prod', 'union_R_dirB')]:
        for wname, (lo, hi) in windows_plus_whole.items():
            sub = pf[(pf['date'] >= lo) & (pf['date'] <= hi)]
            both = sub.dropna(subset=[col, 'prod_asis_R'])
            dR = both[col] - both['prod_asis_R']
            st = c.stats_block(dR.values, both['date'].values)
            delta_rows.append(dict(label=f'{label}/{wname}', n=st['n'], mean_dR=st['mean_dR'],
                                    iid_t=st['iid_t'], day_t=st['day_t'], ex_top5=st['ex_top5']))
    ddf = pd.DataFrame(delta_rows)

    logger.info('reads:\n%s', rdf.to_string())
    logger.info('paired deltas:\n%s', ddf.to_string())

    rdf.to_csv(UNION_CSV, mode='a', header=False, index=False)
    logger.info('appended %d rows to %s', len(rdf), UNION_CSV)

    with open(RESULT_MD, 'a') as fh:
        fh.write("\n## Union with per-slice exits (Amendment 1: a robust slice's own exit, production's E1 elsewhere)\n")
        fh.write(f"Precedence on overlap = higher TRAIN2025 day-clustered t wins (applied last); "
                 f"{n_multi} fills belong to >=2 robust slices. Overlaps: {'; '.join(overlap_lines) if overlap_lines else 'none'}.\n")
        fh.write(f"Direction-A picks (highest VAL2026 mean_R among robust exits): {{{', '.join(f'{k}: {v[0]}' for k, v in picks_A.items())}}}\n")
        fh.write(f"Direction-B picks (highest TRAIN2025 mean_R among robust exits): {{{', '.join(f'{k}: {v[0]}' for k, v in picks_B.items())}}}\n\n")
        for wname in windows_plus_whole:
            rp = rdf[rdf.label == f'production_as_is/{wname}'].iloc[0]
            rA = rdf[rdf.label == f'union_dirA_exits/{wname}'].iloc[0]
            rB = rdf[rdf.label == f'union_dirB_exits/{wname}'].iloc[0]
            dA = ddf[ddf.label == f'dirA_minus_prod/{wname}'].iloc[0]
            dB = ddf[ddf.label == f'dirB_minus_prod/{wname}'].iloc[0]
            fh.write(
                f"- **{wname}**: production {rp['mean_R']:+.3f}R (n{rp['n']:.0f}) | "
                f"union-dirA {rA['mean_R']:+.3f}R (n{rA['n']:.0f}, ΔR {dA['mean_dR']:+.3f} iid-t {dA['iid_t']:.2f} day-t {dA['day_t']:.2f} ex5 {dA['ex_top5']:+.3f}) | "
                f"union-dirB {rB['mean_R']:+.3f}R (n{rB['n']:.0f}, ΔR {dB['mean_dR']:+.3f} iid-t {dB['iid_t']:.2f} day-t {dB['day_t']:.2f} ex5 {dB['ex_top5']:+.3f}) | "
                f"weekly P10 R prod/dirA/dirB = {rp['weekly_p10_R']:+.2f}/{rA['weekly_p10_R']:+.2f}/{rB['weekly_p10_R']:+.2f} | "
                f"max DD {rp['max_drawdown_R']:.1f}/{rA['max_drawdown_R']:.1f}/{rB['max_drawdown_R']:.1f} R | "
                f"strong-wk gap median {rp['strong_week_gap_median']}/{rA['strong_week_gap_median']}/{rB['strong_week_gap_median']} wk | "
                f"$/yr@$375 {rp['dollars_per_year_at_375']:+.0f}/{rA['dollars_per_year_at_375']:+.0f}/{rB['dollars_per_year_at_375']:+.0f}\n")
    logger.info('=== DONE ===')
    return rdf, ddf


if __name__ == '__main__':
    main()
