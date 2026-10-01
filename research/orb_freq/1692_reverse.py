#!/usr/bin/env python3
"""Cell 1,692 -- reversed protocol: SELECT (tune) on 2026-01-01..2026-09-26, TEST (eval) on
2025-01-01..2025-12-31. Owner's ask: "can we tune on 2026 and test/eval on 2025?"

PREREG: research/orb_freq/PREREG_1692.md (FROZEN 2026-10-01).

Reused, unmodified (project convention: digit-prefixed filenames imported via importlib):
  research/orb_freq/1690_variants.py -- load_p1_base, variant_a, walk_all, apply_cost_sizing,
                                         window_slice, _load_module, _ro, CACHE_DB, TRAIN, VAL
                                         (P1 family + its own bar-walk exit variants).
  research/orb_freq/1684_score.py    -- load_book (production reference loader).
stats() below merges 1684_score.py's (iid_t, MDE) and 1690_variants.py's (r_col param, green_weeks
added) -- both exist verbatim in this codebase; copied per 1690_variants.py's own documented
convention (digit-prefixed filename, <20-line function, kept single-file/auditable), not imported.

Inputs (read-only, no R recomputed except re-walking 1690's own (b)/(c) bar exits with its own code):
  research/orb_freq/1684_pool_books.csv, 1685_pool_books.csv, research/orb_freq/1690_reads.csv,
  analysis_results/orb_bplus_book.csv, data/cache.db (?mode=ro, for (b)/(c) bar walk only).

Usage: nice -n 10 python3 research/orb_freq/1692_reverse.py
"""
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_freq'
LOG_FILE = OUT / '1692_reverse.log'
READS_CSV = OUT / '1692_reads.csv'
RESULT_MD = OUT / 'RESULT_1692.md'

R_USD = 375.0
Z80 = 2.802
SELECT = ('2026-01-01', '2026-09-26')   # tune; data truncates at 2026-09-18, disclosed in PREREG
TEST = ('2025-01-01', '2025-12-31')     # eval; identical range to v1690.TRAIN (parity check below)

logger = logging.getLogger('1692')


def setup_logging():
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(sh)


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relpath))
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = old_argv
    return mod


def window_slice(df, lo, hi):
    m = (df['date'] >= pd.Timestamp(lo)) & (df['date'] <= pd.Timestamp(hi))
    return df[m].copy()


def stats(df, label, r_col='R'):
    """Merge of 1684_score.py's stats() (iid_t, MDE) and 1690_variants.py's stats() (r_col,
    green_weeks) -- see module docstring. Not a new formula."""
    if df is None or len(df) == 0:
        return dict(label=label, n=0, fills_wk=float('nan'), mean_r=float('nan'), iid_t=float('nan'),
                    dc_t=float('nan'), ex_top5=float('nan'), mde=float('nan'), weekly_p10=float('nan'),
                    worst_week=float('nan'), n_weeks=0, green_weeks=0, total_usd=0.0)
    r = df[r_col].to_numpy(float)
    n = len(r)
    weeks = df['date'].dt.to_period('W')
    n_weeks = weeks.nunique()
    mean_r = r.mean()
    sd = r.std(ddof=1) if n > 1 else float('nan')
    iid_t = mean_r / (sd / np.sqrt(n)) if n > 1 and sd > 0 else float('nan')
    daily = df.groupby(df['date'].dt.date)[r_col].mean()
    n_days = len(daily)
    dsd = daily.std(ddof=1) if n_days > 1 else float('nan')
    dc_t = (daily.mean() / (dsd / np.sqrt(n_days))) if n_days > 1 and dsd > 0 else float('nan')
    k = max(1, int(np.ceil(n * 0.05)))
    thresh = pd.Series(r).nlargest(k).min()
    ex_top5 = r[r < thresh].mean() if (r < thresh).any() else float('nan')
    mde = Z80 * sd / np.sqrt(n) if n > 1 else float('nan')
    weekly_sum = df.groupby(weeks)[r_col].sum()
    p10 = weekly_sum.quantile(0.10)
    worst = weekly_sum.min()
    green = int((weekly_sum > 0).sum())
    return dict(label=label, n=n, fills_wk=n / n_weeks if n_weeks else float('nan'), mean_r=mean_r,
                iid_t=iid_t, dc_t=dc_t, ex_top5=ex_top5, mde=mde, weekly_p10=p10, worst_week=worst,
                n_weeks=n_weeks, green_weeks=green, total_usd=float(r.sum() * R_USD))


def passes(s):
    return (not pd.isna(s.get('mean_r')) and not pd.isna(s.get('dc_t'))
            and s['mean_r'] >= 0.05 and s['dc_t'] >= 1.5)


def classify(sel_s, test_s):
    if not passes(sel_s):
        return 'fails_both'
    return 'robust' if passes(test_s) else 'regime_specific'


def load_pool_csv(path, pool_col_val, window='in_regime'):
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    d = df[(df['pool'] == pool_col_val) & (df['window'] == window)].copy()
    d['date'] = pd.to_datetime(d['date'])
    return d.sort_values('date').reset_index(drop=True)


def main():
    setup_logging()
    logger.info('=== cell 1,692: reversed protocol (SELECT 2026 -> TEST 2025) -- starting ===')

    v1690 = _load_module('v1690_1692', 'research/orb_freq/1690_variants.py')
    s1684 = _load_module('s1684_1692', 'research/orb_freq/1684_score.py')

    existing_reads = pd.read_csv(OUT / '1690_reads.csv')

    rows = []          # long-format output rows for 1692_reads.csv
    classification = {}  # name -> (class, group)

    def emit(name, group, sel_df, test_df, r_col='R', orig_train=None, orig_val=None,
              orig_is_fresh=False, is_candidate=True):
        sel_s = stats(sel_df, f'{name}__SELECT2026', r_col=r_col)
        test_s = stats(test_df, f'{name}__TEST2025', r_col=r_col)
        cls = classify(sel_s, test_s) if is_candidate else 'context_only'
        classification[name] = (cls, group)
        for direction, window, s in (('reversed', 'SELECT2026', sel_s), ('reversed', 'TEST2025', test_s)):
            rows.append(dict(candidate=name, group=group, direction=direction, window=window,
                              fresh=True, classification=cls, **s))
        if orig_train is not None:
            ot = {k: v for k, v in orig_train.items() if k != 'window'}
            rows.append(dict(candidate=name, group=group, direction='original', window='TRAIN2025',
                              fresh=orig_is_fresh, classification=cls, **ot))
        if orig_val is not None:
            ov = {k: v for k, v in orig_val.items() if k != 'window'}
            rows.append(dict(candidate=name, group=group, direction='original', window='VAL2026',
                              fresh=orig_is_fresh, classification=cls, **ov))
        return sel_s, test_s

    # --------------------------------------------------------------- P1 family (8 candidates + 2 context)
    p1_base = v1690.load_p1_base()
    logger.info('P1 base: n=%d %s..%s', len(p1_base), p1_base['datestr'].min(), p1_base['datestr'].max())

    def orig_reads(vname):
        tr = existing_reads[existing_reads['label'] == f'{vname}__TRAIN']
        va = existing_reads[existing_reads['label'] == f'{vname}__VAL']
        tr = tr.iloc[0].to_dict() if len(tr) else None
        va = va.iloc[0].to_dict() if len(va) else None
        return tr, va

    tr, va = orig_reads('P1_plain')
    sel_s, test_s = emit('P1_plain', 'baseline', window_slice(p1_base, *SELECT), window_slice(p1_base, *TEST),
                          orig_train=tr, orig_val=va, is_candidate=False)
    if tr and not pd.isna(test_s.get('mean_r')) and not pd.isna(tr.get('mean_r')):
        diff = abs(test_s['mean_r'] - tr['mean_r'])
        (logger.warning if diff > 0.01 else logger.info)(
            'parity check P1_plain TEST2025 vs existing TRAIN row: mean_r %.4f vs %.4f (diff %.4f)',
            test_s['mean_r'], tr['mean_r'], diff)

    for feat in ('f1', 'f3', 'f4', 'f5', 'f6'):
        sub = v1690.variant_a(p1_base, feat)
        vname = f'a_{feat.upper()}'
        if sub is None:
            logger.error('%s: VOID (band files missing) -- excluded', vname)
            continue
        tr, va = orig_reads(vname)
        emit(vname, 'p1_variant', window_slice(sub, *SELECT), window_slice(sub, *TEST), orig_train=tr, orig_val=va)

    f1668 = v1690._load_module('f1668_1692', 'research/hod_entry/1668_failure.py')
    e1679 = v1690._load_module('e1679_1692', 'research/orb_exit/1679_orb_exit.py')
    con = v1690._ro(v1690.CACHE_DB)
    walked, cov = v1690.walk_all(p1_base, con, f1668, e1679)
    con.close()
    walked['date'] = pd.to_datetime(walked['datestr'])
    logger.info('bar-walk coverage: %d/%d usable (%.1f%%)', cov['n_ok'], cov['n_total'],
                100.0 * cov['n_ok'] / max(cov['n_total'], 1))

    tr, va = orig_reads('b_live_rule')
    sane = walked.dropna(subset=['live_rule_R'])
    emit('b_live_rule', 'baseline', window_slice(sane, *SELECT), window_slice(sane, *TEST),
         r_col='live_rule_R', orig_train=tr, orig_val=va, is_candidate=False)

    floored = walked[walked['floor_ok']]
    for rule, vname in (('scale50_1R_R', 'b_scale50_1R'), ('noexit2R_half3R_trail1R_R', 'b_noexit2R_half3R_trail1R')):
        sub = floored.dropna(subset=[rule]).copy()
        tr, va = orig_reads(vname)
        emit(vname, 'p1_variant', window_slice(sub, *SELECT), window_slice(sub, *TEST), r_col=rule,
             orig_train=tr, orig_val=va)

    costed = v1690.apply_cost_sizing(walked)
    tr, va = orig_reads('c_cost_sizing')
    emit('c_cost_sizing', 'p1_variant', window_slice(costed, *SELECT), window_slice(costed, *TEST),
         r_col='cost_sized_R', orig_train=tr, orig_val=va)

    # --------------------------------------------------------------- 15 pools (no existing TRAIN/VAL split)
    for pool_id in ('idea1', 'idea2', 'idea10', 'idea11'):
        d = load_pool_csv(OUT / '1684_pool_books.csv', pool_id)
        if len(d) == 0:
            logger.error('pool %s: 0 rows in 1684_pool_books.csv in_regime -- skipped', pool_id)
            continue
        orig_tr = stats(window_slice(d, *TEST), f'{pool_id}__TRAIN2025_fresh')
        orig_va = stats(window_slice(d, *v1690.VAL), f'{pool_id}__VAL2026_fresh')
        emit(pool_id, 'pool', window_slice(d, *SELECT), window_slice(d, *TEST),
             orig_train=orig_tr, orig_val=orig_va, orig_is_fresh=True)

    for pool_id in ('AF6', 'BF1', 'BF3', 'BF4', 'BF5', 'BF6', 'CF1', 'CF3', 'CF4', 'CF5', 'CF6'):
        d = load_pool_csv(OUT / '1685_pool_books.csv', pool_id)
        if len(d) == 0:
            logger.error('subpool %s: 0 rows in 1685_pool_books.csv in_regime -- skipped', pool_id)
            continue
        orig_tr = stats(window_slice(d, *TEST), f'{pool_id}__TRAIN2025_fresh')
        orig_va = stats(window_slice(d, *v1690.VAL), f'{pool_id}__VAL2026_fresh')
        emit(pool_id, 'pool', window_slice(d, *SELECT), window_slice(d, *TEST),
             orig_train=orig_tr, orig_val=orig_va, orig_is_fresh=True)

    # --------------------------------------------------------------- production reference
    prod = s1684.load_book(ROOT / 'analysis_results/orb_bplus_book.csv', entered_only=True)
    if prod is None or len(prod) == 0:
        logger.error('production reference book empty/missing -- excluded')
    else:
        orig_tr = stats(window_slice(prod, *TEST), 'prod__TRAIN2025_fresh')
        orig_va = stats(window_slice(prod, *v1690.VAL), 'prod__VAL2026_fresh')
        emit('production', 'production', window_slice(prod, *SELECT), window_slice(prod, *TEST),
             orig_train=orig_tr, orig_val=orig_va, orig_is_fresh=True)

    reads_df = pd.DataFrame(rows)
    reads_df.to_csv(READS_CSV, index=False)
    logger.info('wrote %s (%d rows)', READS_CSV, len(reads_df))

    counts = {}
    for name, (cls, group) in classification.items():
        if cls == 'context_only':
            continue
        counts[cls] = counts.get(cls, 0) + 1
    logger.info('classification counts (24 candidates, context rows excluded): %s', counts)

    write_result_md(reads_df, classification, cov)
    logger.info('=== DONE ===')


def fmt_cell(s, cols):
    return '/'.join('nan' if pd.isna(s.get(c)) else f'{s[c]:+.3f}' if c in ('mean_r', 'ex_top5') else
                     f'{s[c]:.2f}' if c in ('iid_t', 'dc_t', 'fills_wk', 'mde') else f'{int(s[c])}'
                     for c in cols)


def write_result_md(reads_df, classification, cov):
    lines = []
    lines.append('# RESULT_1692 -- reversed protocol: SELECT 2026-01-01..2026-09-26, TEST 2025 (vs ORIGINAL TRAIN25->VAL26)')
    lines.append('')
    lines.append(f'Bar-walk coverage for (b)/(c) P1 exit variants: {cov["n_ok"]}/{cov["n_total"]} '
                 f'({100.0*cov["n_ok"]/max(cov["n_total"],1):.1f}%). SELECT2026 truncates at the '
                 f'2026-09-18 data cutoff (8 trading days short of 09-26, disclosed in PREREG_1692.md). '
                 f'Pool "original" TRAIN25/VAL26 numbers marked `fresh` below were never previously '
                 f'published (RESULT_1684/1685 scored full in-regime only) -- computed here, same method.')
    lines.append('')
    lines.append('## Side-by-side (REVERSED protocol first, then ORIGINAL direction, then class)')
    lines.append('| Candidate | grp | n/wk/R/t_sel26 | pass26 | n/wk/R/iid/dc_test25 | exTop5/MDE/grn_test25 | '
                 'R/t_train25(orig) | R/t_val26(orig) | class |')
    lines.append('|---|---|---|---|---|---|---|---|---|')

    order = []
    for g in ('baseline', 'p1_variant', 'pool', 'production'):
        order += [n for n, (c, grp) in classification.items() if grp == g]

    for name in order:
        cls, grp = classification[name]
        sub = reads_df[reads_df['candidate'] == name]
        sel = sub[(sub['direction'] == 'reversed') & (sub['window'] == 'SELECT2026')].iloc[0]
        test = sub[(sub['direction'] == 'reversed') & (sub['window'] == 'TEST2025')].iloc[0]
        otr = sub[(sub['direction'] == 'original') & (sub['window'] == 'TRAIN2025')]
        ova = sub[(sub['direction'] == 'original') & (sub['window'] == 'VAL2026')]
        otr = otr.iloc[0] if len(otr) else None
        ova = ova.iloc[0] if len(ova) else None
        pass26 = 'Y' if cls in ('robust', 'regime_specific') else ('N' if cls == 'fails_both' else '-')
        sel_cell = fmt_cell(sel, ['n', 'fills_wk', 'mean_r', 'dc_t'])
        test_cell = fmt_cell(test, ['n', 'fills_wk', 'mean_r', 'iid_t', 'dc_t'])
        test_tail = fmt_cell(test, ['ex_top5', 'mde', 'green_weeks'])
        otr_cell = fmt_cell(otr, ['mean_r', 'dc_t']) if otr is not None else 'n/a'
        ova_cell = fmt_cell(ova, ['mean_r', 'dc_t']) if ova is not None else 'n/a'
        tag = name + (' (fresh-orig)' if otr is not None and bool(otr.get('fresh')) else '') + \
              (' [context]' if cls == 'context_only' else '')
        lines.append(f'| {tag} | {grp} | {sel_cell} | {pass26} | {test_cell} | {test_tail} | '
                     f'{otr_cell} | {ova_cell} | {cls} |')

    lines.append('')
    lines.append('Columns: sel26 = n/fills-wk/meanR/dc_t on SELECT2026. test25 = n/fills-wk/meanR/iid_t/dc_t '
                 'on TEST2025. tail25 = exTop5/MDE/greenWeeks on TEST2025. orig = existing-file (P1 family) '
                 'or freshly-computed-here (pools/production) meanR/dc_t on the ORIGINAL TRAIN2025 and VAL2026 windows.')
    lines.append('')

    counts = {}
    for name, (cls, grp) in classification.items():
        if cls == 'context_only':
            continue
        counts[cls] = counts.get(cls, 0) + 1
    lines.append(f'## Classification counts (24 candidates; P1_plain/b_live_rule shown as context, excluded): {counts}')
    lines.append('')

    regime_specific = [n for n, (c, g) in classification.items() if c == 'regime_specific']
    lines.append('## Regime-specific candidates -- exploration-tier line + forward read needed')
    if not regime_specific:
        lines.append('None. No candidate passed SELECT2026 and failed TEST2025.')
    else:
        reads_idx = reads_df.set_index(['candidate', 'direction', 'window'])
        for name in regime_specific:
            sel = reads_idx.loc[(name, 'reversed', 'SELECT2026')]
            test = reads_idx.loc[(name, 'reversed', 'TEST2025')]
            lines.append(f'- **{name}**: SELECT26 meanR={sel["mean_r"]:+.3f} dc_t={sel["dc_t"]:.2f} '
                         f'(n={int(sel["n"])}) but TEST25 meanR={test["mean_r"]:+.3f} dc_t={test["dc_t"]:.2f} '
                         f'(n={int(test["n"])}). Exploration-tier line: positive point estimate on the latest '
                         f'regime (2026) + mechanism stated in PREREG_1684/1685/1690 + bounded downside at '
                         f'minimum size = eligible to run live at minimum size per `feedback_live_exploration_tier`, '
                         f'NOT as a proven edge. Forward read needed: >=40 live fills forward from today '
                         f'(2026-10-01), out-of-both-samples, before any ramp -- same bar as the ORB ramp-advance rule.')

    passers = [n for n, (c, g) in classification.items() if c in ('robust',) and g != 'context_only']
    lines.append('')
    lines.append(f'## Robust candidates (pass SELECT26 AND TEST25): {passers or "NONE"}')
    lines.append('')
    lines.append('## Caveats (read before relaying): multiplicity continues the programme ledger (24 cands '
                 'x 2 windows re-cut on populations already scored twice in cells 1684/1685/1690 -- "robust" '
                 'here means "survives a second cut of the SAME book," not a fresh out-of-sample population). '
                 'No placebo/green-week-null decomposition run (budget). Causality/fill-realism/price-scale '
                 'inherited unchanged from PREREG_1684/1685/1690, not re-verified here.')
    lines.append('')
    lines.append('Files: research/orb_freq/PREREG_1692.md, 1692_reads.csv, 1692_reverse.py, 1692_reverse.log.')

    RESULT_MD.write_text('\n'.join(lines) + '\n')
    logger.info('wrote %s (%d lines)', RESULT_MD, len(lines))


if __name__ == '__main__':
    main()
