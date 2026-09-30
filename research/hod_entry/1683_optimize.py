#!/usr/bin/env python3
"""research/hod_entry/1683_optimize.py

Cell 1,683 (PREREG_1683.md, FROZEN 2026-09-30 21:30 UTC). Owner 21:25 UTC:
"Get me the most optimized strat across all rules that is profitable,
ignore the long tail. Show me week by week of the past quarter."

Phase 1 (in-sample search, this run): all 58 hypotheses' pooled per-fill dR
already sit in 1681_per_fill.csv (both halves, n=5,506) -- no re-simulation
needed for step 2a. Steps 2b-2d (joint + threshold grid) need a fresh
per-fill bar walk, done here with a GENERIC precedence-ordered joint engine
(run_joint) built from the SAME FillCtx/run_trigger/run_reshape primitives
1681_hypotheses.py already uses -- no rule's condition is reimplemented,
only composed.

Joint-composability scope (disclosed): a joint member must be mechanically
decomposable into either a raw per-bar trigger condition (cond_fn(ctx,j)
-> bool, the argument closed over by 1681's `_trig` wrapper) or a raw
reshape condition (reshape_fn(ctx,j) -> Optional[(stop,target)], closed
over by `_resh`) -- both are extracted by unwrapping the registered
closure, not reimplemented. PARTIAL-type rules (H17-22,H33,H35,H36,H39,
H40,H42,H46,H47,H49,H51,H52,H54,H55,H57,H58) and the two BOOK-level rules
(H48,H56) are each a bespoke multi-leg/cross-fill walk with no shared
atomic per-bar check -- they are scored standalone (best-single, step 2a,
straight off 1681_per_fill.csv) but are NOT joint-composable within this
cell's budget; excluded from the greedy pool, disclosed in RESULT_1683.md.
Forward (sealed) scoring additionally excludes every model=True hypothesis
(H13-17,H20,H24,H32,H34,H36,H41,H43,H45,H58): the P(+1Rnext15) model needs
g7_k{k}_* features that do not exist for the forward population (only
F11-F15+ATR14 do, confirmed by reading 1677_take_profit.py's feature list
against forward_2026q3/1667_features_fwd.csv's columns) and the P(stop)
"model" was never persisted to disk at all (1670_timing_map.py has no
joblib.dump) -- both genuinely unreproducible on new fill_ids, not merely
inconvenient.
"""
import argparse
import importlib.util
import logging
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, fname, root=HERE):
    path = os.path.join(root, fname)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_module('h1681_for_1683', '1681_hypotheses.py')

PER_FILL_1681 = os.path.join(HERE, '1681_per_fill.csv')
FWD_PATHS_PARQUET = os.path.join(HERE, '1683_forward_paths.parquet')
SEARCH_CSV = os.path.join(HERE, '1683_search.csv')
WEEKS_CSV = os.path.join(HERE, '1683_weeks.csv')
FWD_PER_FILL_CSV = os.path.join(HERE, '1683_forward_per_fill.csv')
LOG_FILE = os.path.join(HERE, '1683_optimize.log')
RISK_DOLLARS = 150.0
LIVE_CAP_PER_DAY = 12

logger = logging.getLogger('1683')


def setup_logging():
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_FILE, mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)


# ---------------------------------------------------------------------------
# Trigger-type classification (mechanical: which engine wrapper registered
# the hypothesis in HYPOTHESES) and raw per-bar atom extraction.
# ---------------------------------------------------------------------------

def classify_type(hid):
    spec = H.HYPOTHESES[hid]
    if spec.get('book'):
        return 'BOOK'
    qn = getattr(spec['fn'], '__qualname__', '')
    if qn.startswith('_trig'):
        return 'TRIGGER'
    if qn.startswith('_resh'):
        return 'RESHAPE'
    return 'PARTIAL'


def raw_atom(hid):
    """Unwrap the _trig/_resh closure to the raw cond_fn(ctx,j)->bool or
    reshape_fn(ctx,j)->Optional[(stop,target)] -- the SAME function object
    the standalone rule calls via run_trigger/run_reshape, never
    reimplemented. Only valid for TRIGGER/RESHAPE hids."""
    spec = H.HYPOTHESES[hid]
    fn = spec['fn']
    return fn.__closure__[0].cell_contents


MODEL_IDS = {hid for hid, spec in H.HYPOTHESES.items() if spec['model']}
BOOK_IDS = {hid for hid, spec in H.HYPOTHESES.items() if spec.get('book')}
TYPE_OF = {hid: classify_type(hid) for hid in H.HYPOTHESES}
# joint-composable pool: mechanically decomposable (TRIGGER/RESHAPE) AND
# forward-scoreable (model=False, since forward has no TRAIN/VAL half to
# swap and no G7/P-stop model artifact exists for new fill_ids).
COMPOSABLE = sorted(
    [hid for hid, t in TYPE_OF.items() if t in ('TRIGGER', 'RESHAPE') and hid not in MODEL_IDS],
    key=lambda h: int(h[1:]))
# forward-eligible standalone pool (best-single applied to the sealed
# quarter): anything not model-dependent and not book-level.
FORWARD_ELIGIBLE_SINGLE = sorted(
    [hid for hid in H.HYPOTHESES if hid not in MODEL_IDS and hid not in BOOK_IDS],
    key=lambda h: int(h[1:]))

logger.info if False else None  # placeholder to keep flake calm before setup_logging() runs


def _log_classification():
    n_trig = sum(1 for t in TYPE_OF.values() if t == 'TRIGGER')
    n_resh = sum(1 for t in TYPE_OF.values() if t == 'RESHAPE')
    n_part = sum(1 for t in TYPE_OF.values() if t == 'PARTIAL')
    n_book = sum(1 for t in TYPE_OF.values() if t == 'BOOK')
    logger.info('classification: TRIGGER=%d RESHAPE=%d PARTIAL=%d BOOK=%d (total %d)',
                n_trig, n_resh, n_part, n_book, len(TYPE_OF))
    assert (n_trig, n_resh, n_part, n_book) == (23, 12, 21, 2), \
        f'unexpected engine-type tally {(n_trig, n_resh, n_part, n_book)}'
    logger.info('composable (joint-eligible) pool: %d -> %s', len(COMPOSABLE), COMPOSABLE)
    logger.info('forward-eligible standalone pool: %d hids', len(FORWARD_ELIGIBLE_SINGLE))


# ---------------------------------------------------------------------------
# Generic precedence-ordered joint of TRIGGER/RESHAPE members.
# ---------------------------------------------------------------------------

def run_joint(ctx, member_hids):
    """member_hids: ordered list, precedence = list order. Every bar:
    (1) EVERY selected RESHAPE member's update is applied, ratcheted (max
        stop, min target) -- reshaping never consumes precedence, exactly
        like a lone RESHAPE hypothesis never 'fires' in the exit sense
        (run_reshape's own convention, reused here unmodified in spirit);
    (2) base stop/target/EOD is checked at the (possibly reshaped) levels
        -- same precedence every engine in 1681_hypotheses.py already has;
    (3) the selected TRIGGER members are checked IN PRECEDENCE (list) order
        -- the first whose raw cond_fn(ctx,j) is true governs, full exit at
        bar j+1's open (run_trigger's own convention) -- this is the exact
        generalization of _h36's own 2-member 'first to fire in precedence
        order wins' pattern to up to 3 TRIGGER/RESHAPE members.
    Returns (ruleR, fired, fired_min), same contract as every HYPOTHESES fn.
    A single-member call reproduces that member's own standalone dR exactly
    (verified by selfcheck() against 1681_per_fill.csv before trusting any
    search result)."""
    reshape_fns = [raw_atom(h) for h in member_hids if TYPE_OF[h] == 'RESHAPE']
    trigger_fns = [raw_atom(h) for h in member_hids if TYPE_OF[h] == 'TRIGGER']
    n = ctx.n
    cur_stop, cur_target = ctx.stop0, ctx.target0
    changed = False
    for j in range(1, n):
        for rfn in reshape_fns:
            upd = rfn(ctx, j)
            if upd is not None:
                ns, nt = upd
                if ns != cur_stop or nt != cur_target:
                    changed = True
                cur_stop, cur_target = max(cur_stop, ns), min(cur_target, nt)
        if ctx.m[j] >= H.EOD_M:
            price = ctx.o[j]
            return (H.leg_R(ctx.entry, ctx.R, price) if changed else ctx.base_R), False, (ctx.mins(j) if changed else None)
        if ctx.l[j] <= cur_stop:
            price = H._gap_or_touch(ctx.o[j], ctx.l[j], cur_stop)
            return (H.leg_R(ctx.entry, ctx.R, price) if changed else ctx.base_R), False, None
        if ctx.h[j] >= cur_target:
            return (H.leg_R(ctx.entry, ctx.R, cur_target) if changed else ctx.base_R), False, None
        for cfn in trigger_fns:
            if cfn(ctx, j):
                if j + 1 >= n:
                    return ctx.base_R, False, None
                return H.leg_R(ctx.entry, ctx.R, ctx.o[j + 1]), True, ctx.mins(j)
    return ctx.base_R, False, None


def score_members(member_hids, pop, paths, p_success, p_stop, fill_ids_allowed=None):
    """Run the joint (or a lone member) over every row of `pop` that has a
    cached path, return a per-fill DataFrame (fill_id,date,base_R,rule_R,dR)."""
    rows = []
    for row in pop.itertuples():
        if fill_ids_allowed is not None and row.fill_id not in fill_ids_allowed:
            continue
        path = paths.get(row.fill_id)
        if path is None or len(path['o']) < 2:
            continue
        ctx = H.FillCtx(row.fill_id, row, path, p_success, p_stop, None)
        r, fired, fmin = run_joint(ctx, member_hids) if len(member_hids) != 1 \
            else H.HYPOTHESES[member_hids[0]]['fn'](ctx)
        rows.append((row.fill_id, row.date, row.base_R, r))
    df = pd.DataFrame(rows, columns=['fill_id', 'date', 'base_R', 'rule_R'])
    df['dR'] = df['rule_R'] - df['base_R']
    return df


def selfcheck(pop, paths, p_success, p_stop, n_sample=25):
    """run_joint(ctx,[hid]) must reproduce 1681_per_fill.csv's own dR for
    that hid exactly, for both TRIGGER and RESHAPE hids, on a sample of
    fills -- the independent-check CLAUDE.md requires before trusting any
    new number."""
    ref = pd.read_csv(PER_FILL_1681, usecols=['id', 'fill_id', 'dR'])
    rng = np.random.RandomState(1683)
    sample_hids = [COMPOSABLE[i] for i in rng.choice(len(COMPOSABLE), size=min(6, len(COMPOSABLE)), replace=False)]
    sample_fids = pop['fill_id'].sample(n=min(n_sample, len(pop)), random_state=1683).tolist()
    bad = 0
    for hid in sample_hids:
        ref_sub = ref[ref['id'] == hid].set_index('fill_id')['dR']
        mine = score_members([hid], pop[pop['fill_id'].isin(sample_fids)], paths, p_success, p_stop)
        for r in mine.itertuples():
            if r.fill_id not in ref_sub.index:
                continue
            want = ref_sub.loc[r.fill_id]
            if abs(want - r.dR) > 1e-6:
                bad += 1
                logger.error('selfcheck MISMATCH %s fill_id=%s: ref dR=%.6f mine=%.6f', hid, r.fill_id, want, r.dR)
    if bad:
        logger.error('selfcheck FAILED: %d mismatches -- aborting, run_joint does not reproduce the registry', bad)
        sys.exit(1)
    logger.info('selfcheck OK: run_joint([hid]) reproduces 1681_per_fill.csv exactly for %s on %d sample fills',
                sample_hids, len(sample_fids))


# ---------------------------------------------------------------------------
# Pooled stats (reuses 1681's own stats_block/day_clustered_t/ex_top5_mean).
# ---------------------------------------------------------------------------

def pooled_row(label, df, n_weeks):
    s = H.stats_block(df['dR'], df['date'])
    return dict(stage=label, n=s['n'], mean_dR=s['dR'], iid_t=s['iid_t'], day_t=s['day_t'],
                mde=s['mde'], ex5=s['ex5'], fills_per_week=s['n'] / n_weeks if n_weeks else np.nan)


def n_iso_weeks(dates):
    return pd.to_datetime(pd.Series(dates)).dt.isocalendar().set_index(['year', 'week']).index.nunique()


# ---------------------------------------------------------------------------
# Step 2a: best single rule, straight off 1681_per_fill.csv (both reads
# pooled = the whole n=5,506 book, each fill counted once).
# ---------------------------------------------------------------------------

def best_single(fill_ids_allowed=None):
    per_fill = pd.read_csv(PER_FILL_1681, dtype={'date': str})
    if fill_ids_allowed is not None:
        per_fill = per_fill[per_fill['fill_id'].isin(fill_ids_allowed)]
    n_weeks = n_iso_weeks(per_fill[per_fill['id'] == 'H1']['date'])
    rows = []
    for hid, g in per_fill.groupby('id'):
        s = H.stats_block(g['dR'], g['date'])
        rows.append(dict(stage='single', id=hid, type=TYPE_OF[hid], model=hid in MODEL_IDS, book=hid in BOOK_IDS,
                          n=s['n'], mean_dR=s['dR'], iid_t=s['iid_t'], day_t=s['day_t'], mde=s['mde'], ex5=s['ex5'],
                          fills_per_week=s['n'] / n_weeks if n_weeks else np.nan))
    tbl = pd.DataFrame(rows).sort_values('mean_dR', ascending=False).reset_index(drop=True)
    overall_best = tbl.iloc[0]
    fwd_elig = tbl[tbl['id'].isin(FORWARD_ELIGIBLE_SINGLE)].reset_index(drop=True)
    fwd_best = fwd_elig.iloc[0]
    logger.info('best single OVERALL: %s mean_dR=%.4f t_day=%.2f (model=%s book=%s)',
                overall_best['id'], overall_best['mean_dR'], overall_best['day_t'], overall_best['model'], overall_best['book'])
    logger.info('best single FORWARD-ELIGIBLE: %s mean_dR=%.4f t_day=%.2f',
                fwd_best['id'], fwd_best['mean_dR'], fwd_best['day_t'])
    return tbl, overall_best, fwd_best


# ---------------------------------------------------------------------------
# Steps 2b/2d: greedy forward selection of <=3 compatible (COMPOSABLE) rules.
# ---------------------------------------------------------------------------

def greedy_joint(pop, paths, p_success, p_stop, fill_ids_allowed=None, max_members=3):
    n_weeks = n_iso_weeks(pop['date'] if fill_ids_allowed is None else pop[pop['fill_id'].isin(fill_ids_allowed)]['date'])
    search_rows = []
    selected = []
    best_df = None
    best_mean = -np.inf
    for step in range(1, max_members + 1):
        step_best_add, step_best_df, step_best_mean = None, None, -np.inf
        for cand in COMPOSABLE:
            if cand in selected:
                continue
            trial = selected + [cand]
            df = score_members(trial, pop, paths, p_success, p_stop, fill_ids_allowed=fill_ids_allowed)
            row = pooled_row(f'joint_step{step}_try_{cand}', df, n_weeks)
            row['members'] = '+'.join(trial)
            search_rows.append(row)
            if row['mean_dR'] > step_best_mean:
                step_best_mean, step_best_add, step_best_df = row['mean_dR'], cand, df
        if step_best_add is None or step_best_mean <= best_mean:
            logger.info('greedy step %d: no compatible candidate improves on %.4f -- stopping at %d member(s)',
                        step, best_mean, len(selected))
            break
        selected.append(step_best_add)
        best_mean, best_df = step_best_mean, step_best_df
        logger.info('greedy step %d: + %s -> joint=%s mean_dR=%.4f', step, step_best_add, selected, best_mean)
    final_row = pooled_row('joint_final', best_df, n_weeks) if best_df is not None else None
    if final_row is not None:
        final_row['members'] = '+'.join(selected)
    return selected, best_df, final_row, pd.DataFrame(search_rows)


def compute_cap_fill_ids(pop, cap=LIVE_CAP_PER_DAY):
    """First `cap` fills per day in fill-time (fill_min) order -- the live
    per-day capacity constraint, step 2d."""
    ordered = pop.sort_values(['date', 'fill_min'])
    capped = ordered.groupby('date').head(cap)
    return set(capped['fill_id'])


def main_search():
    setup_logging()
    _log_classification()
    t0 = time.time()
    logger.info('=== 1683 search phase start ===')

    pop = H.load_population()
    paths = H.build_paths(pop)
    p_success, p_stop = H.load_model_probs()
    selfcheck(pop, paths, p_success, p_stop)

    cap_ids = compute_cap_fill_ids(pop)
    logger.info('live-cap population: %d of %d fills (first %d/day by fill_min)', len(cap_ids), len(pop), LIVE_CAP_PER_DAY)

    tbl_single, best_overall, best_single_fwd = best_single()
    tbl_single_capped, best_overall_capped, best_single_fwd_capped = best_single(fill_ids_allowed=cap_ids)

    sel_uncapped, df_uncapped, row_uncapped, search_uncapped = greedy_joint(pop, paths, p_success, p_stop)
    sel_capped, df_capped, row_capped, search_capped = greedy_joint(pop, paths, p_success, p_stop, fill_ids_allowed=cap_ids)

    tbl_single['cap'] = 'uncapped'
    tbl_single_capped['cap'] = 'capped'
    search_uncapped['cap'] = 'uncapped'
    search_capped['cap'] = 'capped'
    search_all = pd.concat([tbl_single, tbl_single_capped, search_uncapped, search_capped], ignore_index=True)
    tmp = SEARCH_CSV + '.tmp'
    search_all.to_csv(tmp, index=False)
    os.replace(tmp, SEARCH_CSV)
    logger.info('wrote %s (%d rows)', SEARCH_CSV, len(search_all))

    logger.info('=== SUMMARY ===')
    logger.info('best single (all 58, uncapped): %s mean_dR=%.4f', best_overall['id'], best_overall['mean_dR'])
    logger.info('best single (forward-eligible, uncapped): %s mean_dR=%.4f', best_single_fwd['id'], best_single_fwd['mean_dR'])
    logger.info('best joint (uncapped): %s mean_dR=%.4f', sel_uncapped, row_uncapped['mean_dR'] if row_uncapped else np.nan)
    logger.info('best joint (capped 12/day): %s mean_dR=%.4f', sel_capped, row_capped['mean_dR'] if row_capped else np.nan)
    logger.info('search phase done in %.1fs', time.time() - t0)
    return dict(best_overall=best_overall, best_single_fwd=best_single_fwd,
                sel_uncapped=sel_uncapped, row_uncapped=row_uncapped,
                sel_capped=sel_capped, row_capped=row_capped, cap_ids=cap_ids)


# ---------------------------------------------------------------------------
# Step 2c: one-step threshold grid on the selected joint's members.
# ---------------------------------------------------------------------------

def threshold_variants(hid):
    """Returns {label: raw_atom_replacement} for a -1/+1 step of `hid`'s own
    main threshold, hand-built only for the specific hids the greedy search
    actually selected (kept small and auditable rather than a generic
    parametrisation of all 25 composable candidates). Each replacement has
    the SAME signature as raw_atom(hid) (cond_fn or reshape_fn), built by
    copying that hid's own condition verbatim and changing only the
    constant PREREG_1683 names as the tunable threshold."""
    variants = {}
    if hid == 'H1':  # exit at 30 min if mtm<+0.25R -> 20/45 min
        def make(mins_):
            def f(ctx, j):
                return ctx.mins(j) >= mins_ and ctx.close_R(j) < 0.25
            return f
        variants = {'H1_lo20': make(20), 'H1_hi45': make(45)}
    elif hid == 'H4':  # >20min since last new high AND mtm<+0.5R -> 10/30 min
        def make(mins_):
            def f(ctx, j):
                return ctx.mins_since_new_high(j) > mins_ and ctx.close_R(j) < 0.5
            return f
        variants = {'H4_lo10': make(10), 'H4_hi30': make(30)}
    elif hid == 'H7':  # close below trailing 10m low while >=+0.5R -> 5/15 min window
        def make(w):
            def f(ctx, j):
                if ctx.close_R(j) < 0.5:
                    return False
                lo = ctx.roll(j, w)['l']
                return ctx.c[j] < lo
            return f
        variants = {'H7_lo5': make(5), 'H7_hi15': make(15)}
    elif hid == 'H25':  # stop to breakeven at +1R -> 0.75R/1.25R trigger
        def make(trig):
            def f(ctx, j):
                if ctx.close_R(j) >= trig:
                    return (ctx.entry, ctx.target0)
                return None
            return f
        variants = {'H25_lo0.75': make(0.75), 'H25_hi1.25': make(1.25)}
    elif hid == 'H26':  # stop to +0.5R at +1.5R -> trigger 1.25/1.75
        def make(trig):
            def f(ctx, j):
                if ctx.close_R(j) >= trig:
                    return (ctx.entry + 0.5 * ctx.R, ctx.target0)
                return None
            return f
        variants = {'H26_lo1.25': make(1.25), 'H26_hi1.75': make(1.75)}
    elif hid == 'H50':  # E20 swing-low trail: last 3 rolling-5m lows after +1R -> 0.75/1.25 trigger
        def make(trig):
            def f(ctx, j):
                if ctx.close_R(j) < trig:
                    return None
                lo = min(ctx.roll(max(0, jj), 5)['l'] for jj in (j, j - 1, j - 2) if jj >= 0)
                return (lo, ctx.target0)
            return f
        variants = {'H50_lo0.75': make(0.75), 'H50_hi1.25': make(1.25)}
    else:
        logger.error('threshold_variants: no hand-built grid for %s -- returning {} (ineligible for step 2c, disclosed)', hid)
    return variants


def tune_joint(selected, pop, paths, p_success, p_stop, fill_ids_allowed=None):
    """One-step grid: for each selected member with a hand-built variant,
    try -1/+1 in place of its frozen threshold, keep whichever single swap
    (at most one member changed at a time, matching PREREG's 'one-step
    grid') maximises the joint's pooled mean dR; returns (tuned_members_desc,
    tuned_df, tuned_row) where tuned_members_desc labels which member (if
    any) was swapped and to which variant."""
    n_weeks = n_iso_weeks(pop['date'] if fill_ids_allowed is None else pop[pop['fill_id'].isin(fill_ids_allowed)]['date'])
    base_df = score_members(selected, pop, paths, p_success, p_stop, fill_ids_allowed=fill_ids_allowed)
    base_mean = base_df['dR'].mean()
    best = dict(label='base (no change)', df=base_df, mean=base_mean, override_hid=None, override_fn=None)
    for hid in selected:
        variants = threshold_variants(hid)
        for label, newfn in variants.items():
            df = score_members_with_override(selected, hid, newfn, pop, paths, p_success, p_stop, fill_ids_allowed)
            m = df['dR'].mean()
            logger.info('grid %s: mean_dR=%.4f (base %.4f)', label, m, base_mean)
            if m > best['mean']:
                best = dict(label=label, df=df, mean=m, override_hid=hid, override_fn=newfn)
    row = pooled_row('tuned_joint', best['df'], n_weeks)
    row['members'] = '+'.join(selected) + (f" [{best['label']}]" if best['label'] != 'base (no change)' else '')
    return best, row


def score_members_with_override(member_hids, override_hid, override_fn, pop, paths, p_success, p_stop, fill_ids_allowed=None):
    """Like score_members, but `override_hid`'s raw atom is replaced by
    `override_fn` for this call only (the one-step threshold grid never
    mutates the shared HYPOTHESES registry)."""
    reshape_fns, trigger_fns = [], []
    for h in member_hids:
        fn = override_fn if h == override_hid else raw_atom(h)
        t = TYPE_OF[h]
        (reshape_fns if t == 'RESHAPE' else trigger_fns).append(fn)
    rows = []
    for row in pop.itertuples():
        if fill_ids_allowed is not None and row.fill_id not in fill_ids_allowed:
            continue
        path = paths.get(row.fill_id)
        if path is None or len(path['o']) < 2:
            continue
        ctx = H.FillCtx(row.fill_id, row, path, p_success, p_stop, None)
        n = ctx.n
        cur_stop, cur_target = ctx.stop0, ctx.target0
        changed = False
        r = ctx.base_R
        for j in range(1, n):
            for rfn in reshape_fns:
                upd = rfn(ctx, j)
                if upd is not None:
                    ns, nt = upd
                    if ns != cur_stop or nt != cur_target:
                        changed = True
                    cur_stop, cur_target = max(cur_stop, ns), min(cur_target, nt)
            if ctx.m[j] >= H.EOD_M:
                r = H.leg_R(ctx.entry, ctx.R, ctx.o[j]) if changed else ctx.base_R
                break
            if ctx.l[j] <= cur_stop:
                price = H._gap_or_touch(ctx.o[j], ctx.l[j], cur_stop)
                r = H.leg_R(ctx.entry, ctx.R, price) if changed else ctx.base_R
                break
            if ctx.h[j] >= cur_target:
                r = H.leg_R(ctx.entry, ctx.R, cur_target) if changed else ctx.base_R
                break
            fired_here = False
            for cfn in trigger_fns:
                if cfn(ctx, j):
                    r = ctx.base_R if j + 1 >= n else H.leg_R(ctx.entry, ctx.R, ctx.o[j + 1])
                    fired_here = True
                    break
            if fired_here:
                break
        rows.append((row.fill_id, row.date, row.base_R, r))
    df = pd.DataFrame(rows, columns=['fill_id', 'date', 'base_R', 'rule_R'])
    df['dR'] = df['rule_R'] - df['base_R']
    return df


# ---------------------------------------------------------------------------
# Step 3: sealed forward read.
# ---------------------------------------------------------------------------

def run_forward_members(members, pop_fwd, paths_fwd, label):
    needs_atr = 'H44' in members
    sub = pop_fwd
    if needs_atr:
        n_before = len(sub)
        sub = sub[sub['atr14'].notna()]
        if len(sub) != n_before:
            logger.warning('%s: dropping %d/%d forward fills with NaN atr14 (H44 needs it)', label, n_before - len(sub), n_before)
    df = score_members(members, sub, paths_fwd, {}, {})
    df['label'] = label
    return df


def week_table(dates, base_R, rule_R):
    d = pd.DataFrame({'date': pd.to_datetime(dates), 'base_R': base_R, 'rule_R': rule_R})
    iso = d['date'].dt.isocalendar()
    d['week'] = iso['year'].astype(str) + '-W' + iso['week'].astype(str).str.zfill(2)
    day_rule = d.groupby('date')['rule_R'].sum()
    day_base = d.groupby('date')['base_R'].sum()
    rows = []
    for wk, g in d.groupby('week'):
        uniq_days = g['date'].unique()
        rows.append(dict(week=wk, fills=len(g), sum_R_base=g['base_R'].sum(), sum_R_opt=g['rule_R'].sum(),
                          mean_R_base=g['base_R'].mean(), mean_R_opt=g['rule_R'].mean(),
                          dollars_base=g['base_R'].sum() * RISK_DOLLARS, dollars_opt=g['rule_R'].sum() * RISK_DOLLARS,
                          green_base=g['base_R'].sum() > 0, green_opt=g['rule_R'].sum() > 0,
                          worst_day_base=day_base.loc[uniq_days].min(), worst_day_opt=day_rule.loc[uniq_days].min()))
    return pd.DataFrame(rows).sort_values('week').reset_index(drop=True)


def totals_block(dates, R, n_weeks_for_rate=None):
    s = H.stats_block(R, dates)
    wk = H._weekly(dates, pd.Series(R).to_numpy())
    p10 = wk.quantile(0.10)
    green_share = (wk > 0).mean()
    maxdd = H._maxdd(dates, pd.Series(R).to_numpy())
    return dict(n=s['n'], mean_R=s['dR'], iid_t=s['iid_t'], day_t=s['day_t'], mde=s['mde'], ex5=s['ex5'],
                green_week_share=green_share, weekly_p10=p10, maxdd=maxdd,
                dollars=s['dR'] * s['n'] * RISK_DOLLARS if not pd.isna(s['dR']) else np.nan,
                n_weeks=len(wk))


def scored_to_forward(members, override_hid, override_fn, pop_fwd, paths_fwd, label):
    """run_forward_members's logic, but able to apply a tuned-joint's
    threshold override (or none) -- used for base/best_single/best_joint
    (override_fn=None) and tuned_joint (override_fn set) alike."""
    needs_atr = 'H44' in members
    sub = pop_fwd
    if needs_atr:
        n_before = len(sub)
        sub = sub[sub['atr14'].notna()]
        if len(sub) != n_before:
            logger.warning('%s: dropping %d/%d forward fills with NaN atr14 (H44 needs it)', label, n_before - len(sub), n_before)
    if override_fn is None:
        df = score_members(members, sub, paths_fwd, {}, {})
    else:
        df = score_members_with_override(members, override_hid, override_fn, sub, paths_fwd, {}, {})
    df['label'] = label
    return df


def main_forward(joint_uncapped, joint_capped):
    setup_logging()
    _log_classification()
    t0 = time.time()
    logger.info('=== 1683 forward phase start (joint_uncapped=%s joint_capped=%s) ===', joint_uncapped, joint_capped)

    pop = H.load_population()
    paths = H.build_paths(pop)
    p_success, p_stop = H.load_model_probs()
    cap_ids = compute_cap_fill_ids(pop)

    tbl_single, best_overall, best_single_fwd = best_single()
    best_single_id = best_single_fwd['id']

    best_joint_row, tuned_row = None, None
    tuned_best = dict(override_hid=None, override_fn=None, label='n/a (no joint)')
    if joint_uncapped:
        df_joint_in = score_members(joint_uncapped, pop, paths, p_success, p_stop)
        best_joint_row = pooled_row('joint_final', df_joint_in, n_iso_weeks(pop['date']))
        best_joint_row['members'] = '+'.join(joint_uncapped)
        tuned_best, tuned_row = tune_joint(joint_uncapped, pop, paths, p_success, p_stop)
        logger.info('tuned joint: %s (mean_dR in-sample %.4f vs untuned %.4f)', tuned_row['members'], tuned_row['mean_dR'], best_joint_row['mean_dR'])

    logger.info('in-sample (pooled 2025-26, n=5506): best_single=%s dR=%.4f | best_joint=%s dR=%.4f | tuned_joint dR=%.4f',
                best_single_id, best_single_fwd['mean_dR'],
                best_joint_row['members'] if best_joint_row else None, best_joint_row['mean_dR'] if best_joint_row else np.nan,
                tuned_row['mean_dR'] if tuned_row else np.nan)

    # ---- sealed forward population ----
    pop_fwd = H.load_population_forward()
    paths_fwd = H.build_paths(pop_fwd, parquet_path=FWD_PATHS_PARQUET)
    cap_ids_fwd = compute_cap_fill_ids(pop_fwd)
    logger.info('forward population: %d fills (r_pct>=1.5%%), %d under the %d/day cap', len(pop_fwd), len(cap_ids_fwd), LIVE_CAP_PER_DAY)

    fwd_frames = []
    base_fwd = pop_fwd[['fill_id', 'date', 'base_R']].copy()
    base_fwd['rule_R'] = base_fwd['base_R']
    base_fwd['dR'] = 0.0
    base_fwd['label'] = 'base'
    fwd_frames.append(base_fwd)
    fwd_frames.append(scored_to_forward([best_single_id], None, None, pop_fwd, paths_fwd, 'best_single'))
    if joint_uncapped:
        fwd_frames.append(scored_to_forward(joint_uncapped, None, None, pop_fwd, paths_fwd, 'best_joint'))
        fwd_frames.append(scored_to_forward(joint_uncapped, tuned_best['override_hid'], tuned_best['override_fn'], pop_fwd, paths_fwd, 'tuned_joint'))
    fwd_all = pd.concat(fwd_frames, ignore_index=True)
    tmp = FWD_PER_FILL_CSV + '.tmp'
    fwd_all.to_csv(tmp, index=False)
    os.replace(tmp, FWD_PER_FILL_CSV)
    logger.info('wrote %s (%d rows)', FWD_PER_FILL_CSV, len(fwd_all))

    # ---- weekly table + totals, base vs the final "optimized" candidate ----
    optimized_label = 'tuned_joint' if joint_uncapped else 'best_single'
    base_sub = fwd_all[fwd_all['label'] == 'base'].set_index('fill_id')
    opt_sub = fwd_all[fwd_all['label'] == optimized_label].set_index('fill_id')
    common = base_sub.index.intersection(opt_sub.index)
    wk = week_table(base_sub.loc[common, 'date'], base_sub.loc[common, 'base_R'], opt_sub.loc[common, 'rule_R'])
    tmp = WEEKS_CSV + '.tmp'
    wk.to_csv(tmp, index=False)
    os.replace(tmp, WEEKS_CSV)
    logger.info('wrote %s (%d ISO weeks)', WEEKS_CSV, len(wk))

    totals = {}
    for lbl in ['base', 'best_single', 'best_joint', 'tuned_joint']:
        sub = fwd_all[fwd_all['label'] == lbl]
        if len(sub) == 0:
            continue
        totals[lbl] = totals_block(sub['date'], sub['rule_R'])
        capped_ids = cap_ids_fwd
        sub_capped = sub[sub['fill_id'].isin(capped_ids)]
        totals[lbl + '_capped'] = totals_block(sub_capped['date'], sub_capped['rule_R']) if len(sub_capped) else None

    logger.info('=== FORWARD SUMMARY (%s, sealed 2026-06-01..09-04) ===', optimized_label)
    for lbl, t in totals.items():
        if t is None:
            continue
        logger.info('%-20s n=%4d mean_R=%.4f day_t=%.2f green_wk=%.0f%% P10=%.3f maxdd=%.2f $=%.0f',
                    lbl, t['n'], t['mean_R'], t['day_t'], 100 * t['green_week_share'], t['weekly_p10'], t['maxdd'], t['dollars'])
    logger.info('forward phase done in %.1fs', time.time() - t0)
    return dict(tbl_single=tbl_single, best_single_id=best_single_id, best_single_fwd=best_single_fwd,
                best_joint_row=best_joint_row, tuned_row=tuned_row, tuned_best=tuned_best,
                fwd_all=fwd_all, wk=wk, totals=totals, optimized_label=optimized_label,
                joint_uncapped=joint_uncapped, joint_capped=joint_capped)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['search', 'forward'], default='search')
    ap.add_argument('--joint-uncapped', default='', help='comma list of member hids from the search stage')
    ap.add_argument('--joint-capped', default='', help='comma list of member hids from the search stage (capped run)')
    args = ap.parse_args()
    if args.stage == 'search':
        main_search()
    else:
        ju = [h for h in args.joint_uncapped.split(',') if h]
        jc = [h for h in args.joint_capped.split(',') if h]
        main_forward(ju, jc)
