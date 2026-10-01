#!/usr/bin/env python3
"""Cell 1,696: what 'rank_not_selected' really is, at full bar coverage.

PREREG: research/orb_freq/PREREG_1696.md (FROZEN 2026-10-01 16:45 UTC). From cell 1,694 Part C2:
13,163 admitted ORB candidates 2025-26 = 476 taken + 7,021 vetoed (PDR/G1/range-size) + 5,666
no-fill. The residual 'rank_not_selected' bucket (1,937 candidates, +0.284 R, 6% runner share,
~$19.7K/yr) was read on only 16% bar coverage and was NOT individually decomposed into score
threshold / Q1 / slot-cap / dedup (1694_money.py's own classify_missed docstring: "NOT
reconstructable... without re-running the per-day ranking pass"). This cell does that pass.

Reuses (imported read-only via importlib, project convention for digit-prefixed cell scripts):
  research/orb_freq/1694_money.py (f1694, which transitively carries f1693/f1679/CachedStore/
    stats_block/ORB_EOD_M/FIXED_RISK_DOLLARS/VAL_HI/etc.) -- CachedStore, reconstruct_fill (entry-
    minute reconstruction: book's own entry_price, breakout bar, stop = range low), A_EXITS[BASELINE]
    (the production exit, A3_noTarget_liveLock / the live lock-stop walker), stats_block (day-
    clustered t / ex-top-5%), latest_features_csv, ORB_BOOK_CSV, VAL_HI, FIXED_RISK_DOLLARS,
    YEARS_2025_2026, ORB_EOD_M. The production entry/exit model is NEVER changed here (PREREG "Not
    allowed").
  study_orb_filter.composite_score, study_orb_sizing.assign_quintile, study_orb_correlation_filter.
    symbol_family/symbol_super_group, study_orb_pipeline_static_lock.Q_ORDER -- the pipeline's OWN
    ranking functions, called read-only. orb.yaml itself is read (never written) for filter.features
    (z-params), quintile_cutoffs, filter.threshold, filter.skip_q1, dedup.by_family/by_super_group,
    sizing.max_concurrent -- the SAME keys study_orb_pipeline_static_lock.py's load_bt_config /
    main() read, at this node's current (B+ RESTART 2026-08-15) values. dedup.by_anchor is ABSENT
    from orb.yaml -> anchor dedup is OFF in both live and this replay (matches the pipeline's own
    default), so it is not replayed.
  research/orb_freq/1694_missed.csv -- the 12,687-row admitted-but-not-taken population with its
    missed_reason (PDR_veto/G1_veto/range_size_veto/rank_not_selected/no_fill) and its ORIGINAL
    (16%/13% coverage) counterfactual_R/status, reused for the coverage-bias check (Step 3).

The per-day ranking loop (sort by (quintile Q_ORDER asc, composite desc), then one pass applying
family/super-group dedup with an 8-slot cap, NO refill) is replayed here faithfully following
study_orb_pipeline_static_lock.py main()'s own control flow (lines ~1074-1113: Top-K + dedup per
day) -- that control flow is inlined in main(), not a separate function, so it is reproduced
line-for-line (same sort key, same dedup/slot order, same break-on-full semantics), while every
actual computation (composite, quintile, family, super-group) calls the pipeline's real functions.
Independent check (CLAUDE.md "no research claim ships without an independent check"): the replay's
'selected' set is cross-validated against the known-selected population (taken + PDR_veto + G1_veto
+ range_size_veto + no_fill, from 1694_missed.csv + the book) -- Part 0 below; any mismatch is
reported, not hidden.

Usage: nice -n 10 python3 research/orb_freq/1696_missed.py
"""
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

LOG_FILE = os.path.join(HERE, '1696_missed.log')
MISSED_FULL_CSV = os.path.join(HERE, '1696_missed_full.csv')
REASONS_CSV = os.path.join(HERE, '1696_reasons.csv')
RESULT_MD = os.path.join(HERE, 'RESULT_1696.md')
ORIG_MISSED_CSV = os.path.join(HERE, '1694_missed.csv')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                     handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode='w')])
logger = logging.getLogger('cell1696')


def _load_module(name, fname, root=ROOT):
    """Verbatim copy of 1694_money.py's own loader (project convention: a module whose filename
    starts with a digit cannot be `import`ed by name)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, fname))
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = old_argv
    return mod


def append_result_md(text):
    with open(RESULT_MD, 'a') as fh:
        fh.write(text)
        fh.flush()


logger.info('=== cell 1,696: rank_not_selected at full coverage -- starting ===')
f1694 = _load_module('f1694_1696', 'research/orb_freq/1694_money.py')
orb_csv = f1694.orb_csv
CachedStore = f1694.CachedStore
reconstruct_fill = f1694.reconstruct_fill
A_EXITS = f1694.A_EXITS
BASELINE = f1694.BASELINE
stats_block = f1694.stats_block
ORB_EOD_M = f1694.ORB_EOD_M
VAL_HI = f1694.VAL_HI
FIXED_RISK_DOLLARS = f1694.FIXED_RISK_DOLLARS
YEARS_2025_2026 = f1694.YEARS_2025_2026
ORB_BOOK_CSV = f1694.ORB_BOOK_CSV

from study_orb_filter import composite_score          # noqa: E402
from study_orb_sizing import assign_quintile           # noqa: E402
from study_orb_correlation_filter import symbol_family, symbol_super_group  # noqa: E402
from study_orb_pipeline_static_lock import Q_ORDER     # noqa: E402

# ---------------------------------------------------------------------------
# orb.yaml -- READ ONLY (never written; CLAUDE.md "never touch config/orb.yaml"),
# the SAME keys study_orb_pipeline_static_lock.load_bt_config()/main() read.
# ---------------------------------------------------------------------------
with open(os.path.join(ROOT, 'orb.yaml')) as fh:
    _cfg = yaml.safe_load(fh)
_feats_cfg = _cfg['filter']['features']
PARAMS = {f: {'mean': float(v['mean']), 'std': float(v['std']), 'sign': int(v['sign'])}
          for f, v in _feats_cfg.items()}
CUTOFFS = [float(x) for x in _cfg['quintile_cutoffs']]
THRESHOLD = float(_cfg['filter']['threshold'])
SKIP_Q1 = bool(_cfg['filter']['skip_q1'])
DEDUP_BY_FAMILY = bool(_cfg['dedup']['by_family'])
DEDUP_BY_SUPER_GROUP = bool(_cfg['dedup']['by_super_group'])
N_PER_DAY = int(_cfg['sizing']['max_concurrent'])
ANCHOR_DEDUP_ON = 'by_anchor' in (_cfg.get('dedup') or {})
logger.info('orb.yaml (read-only): threshold=%.9f skip_q1=%s dedup(family=%s,super_group=%s) '
            'n_per_day=%d anchor_dedup=%s cutoffs=%s',
            THRESHOLD, SKIP_Q1, DEDUP_BY_FAMILY, DEDUP_BY_SUPER_GROUP, N_PER_DAY,
            ANCHOR_DEDUP_ON, CUTOFFS)
if ANCHOR_DEDUP_ON:
    logger.error('dedup.by_anchor IS present in orb.yaml -- this replay does NOT implement anchor '
                 'dedup and its reason split would be WRONG; aborting rather than silently mis-rank')
    sys.exit(1)


def replay_ranking(full):
    """Replay the pipeline's ranking stage on the FULL admitted-candidate population (needed for
    correct per-day rank/dedup/slot simulation -- taken/vetoed/no-fill rows all occupy ranking slots
    before PDR/G1/range-size vetoes remove them POST-selection with NO refill). Returns `full` with
    two new columns: _ranking_reason ('selected'|'score_threshold'|'skip_q1'|'dedup'|'slot_cap') and
    _quintile (NaN for score_threshold rows, which never reach quintile assignment)."""
    missing_feat = [f for f in PARAMS if full[f].isna().any()]
    if missing_feat:
        logger.warning('FILTER_FEATURES with NaN in the full population: %s (NaN composite -> '
                       'correctly falls below threshold, not a bug, but flagged)', missing_feat)
    full = full.copy()
    full['_composite'] = composite_score(full, PARAMS)
    below_thr = ~(full['_composite'] >= THRESHOLD)   # NaN composite -> True (below), logged above
    reason = pd.Series('selected', index=full.index, dtype=object)
    reason[below_thr] = 'score_threshold'
    quintile = pd.Series(np.nan, index=full.index, dtype=object)
    kept_idx = full.index[~below_thr]
    quintile.loc[kept_idx] = assign_quintile(full.loc[kept_idx, '_composite'], CUTOFFS).values
    if SKIP_Q1:
        q1_idx = kept_idx[quintile.loc[kept_idx] == 'Q1']
        reason.loc[q1_idx] = 'skip_q1'
    rank_idx = kept_idx[reason.loc[kept_idx] == 'selected']  # survives threshold + Q1 -> ranking loop
    rk = full.loc[rank_idx, ['date', 'symbol']].copy()
    rk['_quintile'] = quintile.loc[rank_idx]
    rk['_q_rank'] = rk['_quintile'].map(Q_ORDER)
    rk['_composite'] = full.loc[rank_idx, '_composite']
    n_selected, n_dedup, n_slot = 0, 0, 0
    for day, dg in rk.groupby('date'):
        d = dg.sort_values(['_q_rank', '_composite'], ascending=[True, False], kind='mergesort')
        seen_fam, seen_sup = set(), set()
        day_n_selected = 0
        slots_full = False
        for idx, r in d.iterrows():
            if slots_full:
                reason.loc[idx] = 'slot_cap'
                n_slot += 1
                continue
            fam = symbol_family(r['symbol']) if DEDUP_BY_FAMILY else None
            sup = symbol_super_group(r['symbol']) if DEDUP_BY_SUPER_GROUP else None
            if (fam and fam in seen_fam) or (sup and sup in seen_sup):
                reason.loc[idx] = 'dedup'
                n_dedup += 1
                continue
            if fam:
                seen_fam.add(fam)
            if sup:
                seen_sup.add(sup)
            day_n_selected += 1
            n_selected += 1
            if day_n_selected >= N_PER_DAY:
                slots_full = True
    logger.info('ranking replay: %d score_threshold, %d skip_q1, %d selected, %d dedup, %d slot_cap '
                '(of %d admitted candidates)', int(below_thr.sum()),
                int((reason == 'skip_q1').sum()), n_selected, n_dedup, n_slot, len(full))
    full['_ranking_reason'] = reason
    full['_quintile'] = quintile
    return full


def independent_check(full, missed):
    """Cross-validate the replay against ground truth: book_csv (the pipeline's OWN output,
    `sel` after ranking AND the PDR/G1/range-size vetoes, in-range rows only) is the true
    ranking+veto survivor set. 1694_missed.csv's PDR_veto/G1_veto/range_size_veto buckets are NOT
    usable as a ranking ground truth -- classify_missed applies those veto functions to EVERY
    non-taken candidate regardless of whether it would have reached ranking at all (its own
    docstring says so), so e.g. 'PDR_veto' (4323) hugely overcounts true post-ranking PDR vetoes.
    The correct test: take the replay's ranking-stage survivors ('selected', pre-veto) and apply
    the SAME three veto functions in the pipeline's OWN order (PDR -> G1 -> range-size, each
    POST-selection, NO refill) -- the result must equal book_csv's in-range rows exactly. Any
    mismatch is reported, not hidden (CLAUDE.md: no research claim ships without an independent
    check)."""
    from trading.orb_pdr_veto import pdr_veto_applies
    from trading.orb_g1_veto import g1_reject
    from trading.orb_range_size_veto import range_size_veto_applies
    ORB_PDR_MIN = f1694.ORB_PDR_MIN
    ORB_G1_RV20_MIN = f1694.ORB_G1_RV20_MIN
    ORB_G1_PDR_MIN = f1694.ORB_G1_PDR_MIN
    ORB_RANGE_SIZE_MIN = f1694.ORB_RANGE_SIZE_MIN

    book = orb_csv.read_orb_csv(ORB_BOOK_CSV)
    book['date'] = book['date'].astype(str)
    book_in_range = book[(book['date'] >= '2025-01-01') & (book['date'] <= str(VAL_HI))]
    book_keys = set(zip(book_in_range['symbol'], book_in_range['date']))

    ranked = full[full['_ranking_reason'] == 'selected'].copy()
    pdr_v = ranked['prev_day_range_pct'].apply(lambda v: pdr_veto_applies(None if pd.isna(v) else float(v), ORB_PDR_MIN))
    after_pdr = ranked[~pdr_v]
    g1_v = after_pdr.apply(lambda r: g1_reject(
        None if pd.isna(r['return_volatility_20d']) else float(r['return_volatility_20d']),
        None if pd.isna(r['prev_day_range_pct']) else float(r['prev_day_range_pct']),
        ORB_G1_RV20_MIN, ORB_G1_PDR_MIN, short_history_veto=False) is not None, axis=1)
    after_g1 = after_pdr[~g1_v]
    rs_v = after_g1['range_size_pct'].apply(lambda v: range_size_veto_applies(v, ORB_RANGE_SIZE_MIN))
    after_rs = after_g1[~rs_v]
    replay_final_keys = set(zip(after_rs['symbol'], after_rs['date']))

    only_book = book_keys - replay_final_keys
    only_replay = replay_final_keys - book_keys
    n_match = len(book_keys & replay_final_keys)
    logger.info('INDEPENDENT CHECK: ranking-survivors=%d, -PDR=%d -G1=%d -rangesize=%d -> '
                'replay-final=%d vs book_csv in-range=%d; agree=%d only_book=%d only_replay=%d '
                '(match rate %.2f%%)', len(ranked), len(after_pdr), len(after_g1), len(after_rs),
                len(replay_final_keys), len(book_keys), n_match, len(only_book), len(only_replay),
                100.0 * n_match / max(len(book_keys), 1))
    if only_book or only_replay:
        logger.warning('independent check MISMATCHES exist -- sample only_book=%s '
                       'only_replay=%s', list(only_book)[:8], list(only_replay)[:8])
    return dict(ranking_survivors=len(ranked), replay_final=len(replay_final_keys),
                book_in_range=len(book_keys), agree=n_match, only_book=len(only_book),
                only_replay=len(only_replay))


def walk_counterfactual(store, rows):
    """Production entry (book's own entry_price, breakout bar, stop=range low) + production exit
    (A3_noTarget_liveLock, the baseline/live lock-stop) walked on the bars, VERBATIM via f1694's
    reconstruct_fill + A_EXITS -- the entry/exit model is never changed (PREREG 'Not allowed').
    Also computes MFE_R (full remaining day to the EOD truncation bar, same definition as 1694's
    build_fill_table runner3) for the >=3R runner-share read the PREREG asks for."""
    cf_R, cf_status, mfe_R, runner3 = [], [], [], []
    t0 = time.time()
    for n_seen, r in enumerate(rows.itertuples(), 1):
        rec, st = reconstruct_fill(store, r.symbol, r.date, r.entry_price)
        if rec is None:
            cf_R.append(np.nan); cf_status.append(st); mfe_R.append(np.nan); runner3.append(0)
            continue
        bars, i0, entry, stop, R_unit = rec['bars'], rec['i0'], rec['entry'], rec['stop'], rec['R_unit']
        rv, _ = A_EXITS[BASELINE](bars, i0, entry, stop, R_unit)
        cf_R.append(rv); cf_status.append('ok')
        eod_idx = np.where(bars['minarr'] >= ORB_EOD_M)[0]
        last_idx = int(eod_idx[0]) if len(eod_idx) else len(bars['o']) - 1
        seg_h = bars['h'][i0 + 1:last_idx + 1]
        if len(seg_h):
            m = (float(seg_h[int(np.argmax(seg_h))]) - entry) / R_unit
        else:
            m = np.nan
        mfe_R.append(m)
        runner3.append(int(m >= 3.0) if m == m else 0)
        if n_seen % 500 == 0:
            logger.info('counterfactual walk: %d/%d (%.0fs)', n_seen, len(rows), time.time() - t0)
    rows = rows.copy()
    rows['counterfactual_R'] = cf_R
    rows['counterfactual_status'] = cf_status
    rows['mfe_R'] = mfe_R
    rows['runner3'] = runner3
    return rows


def half_of(date_str):
    return date_str[:4]  # '2025' / '2026' -- PREREG's "per half (2025/2026)"


def bucket_stats(sub):
    ok = sub[sub['counterfactual_status'] == 'ok']
    n, n_ok = len(sub), len(ok)
    if n_ok == 0:
        return dict(n=n, n_ok=0, coverage_pct=0.0, mean_R=np.nan, day_t=np.nan, runner_share=np.nan,
                    ex_top5=np.nan, dollars_per_yr=np.nan)
    st = stats_block(ok['counterfactual_R'].values, ok['date'].values)
    runner_share = float(ok['runner3'].mean())
    dollars_per_yr = st['mean_dR'] * n_ok / YEARS_2025_2026 * FIXED_RISK_DOLLARS
    return dict(n=n, n_ok=n_ok, coverage_pct=100.0 * n_ok / n, mean_R=st['mean_dR'], day_t=st['day_t'],
                runner_share=runner_share, ex_top5=st['ex_top5'], dollars_per_yr=dollars_per_yr)


def main():
    free_gb = __import__('shutil').disk_usage('/').free / 1e9
    logger.info('df -h / : %.2f GB free', free_gb)
    with open(RESULT_MD, 'w') as fh:
        fh.write(f"# RESULT 1,696 -- rank_not_selected at full coverage (run started "
                  f"{datetime.now().isoformat()})\n\nPREREG: research/orb_freq/PREREG_1696.md "
                  f"(FROZEN). Written incrementally per step; coverage first.\n\n")

    # ---- Step 1 (coverage) was done by the scratchpad backfill wrapper BEFORE this script ran
    # (research/bf_zero/backfill_bars_sip.py via the scratchpad wrapper, 1,349,355 bars appended
    # across 407/407 days, 0 chunk failures, disk 8.75->8.51 GB -- see 1696_backfill.log). Report
    # the realized coverage on the actual candidate rows below (Part 1).

    missed = pd.read_csv(ORIG_MISSED_CSV, dtype={'date': str})
    feat_csv = f1694.latest_features_csv()
    full = orb_csv.read_orb_csv(feat_csv)
    full['date'] = full['date'].astype(str)
    full = full[(full['date'] >= '2025-01-01') & (full['date'] <= str(VAL_HI))].copy()
    logger.info('full admitted population 2025-01-01..%s: %d rows', VAL_HI, len(full))

    # ---- Step 2: replay the ranking stage, split rank_not_selected into real reasons -----------
    full = replay_ranking(full)
    check = independent_check(full, missed)
    append_result_md(f"## Independent check (replay + live PDR/G1/range-size vetoes vs book_csv)\n\n"
                      f"ranking-stage survivors (pre-veto) {check['ranking_survivors']}; after replaying "
                      f"PDR->G1->range-size veto (same functions, same order, no refill): "
                      f"{check['replay_final']} vs book_csv in-range {check['book_in_range']}; agree "
                      f"{check['agree']} ({100.0*check['agree']/max(check['book_in_range'],1):.2f}%), "
                      f"only-in-book {check['only_book']}, only-in-replay {check['only_replay']}.\n\n")

    rns = missed[missed.missed_reason == 'rank_not_selected'][['symbol', 'date']].copy()
    full_key = full.set_index(['symbol', 'date'])
    rns = rns.join(full_key[['_ranking_reason']], on=['symbol', 'date'])
    reason_counts = rns['_ranking_reason'].value_counts().to_dict()
    logger.info('rank_not_selected (n=%d) split by real reason: %s', len(rns), reason_counts)
    append_result_md(f"## Step 2 -- rank_not_selected ({len(rns)}) split by real reason\n\n"
                      + '\n'.join(f"- {k}: {v}" for k, v in sorted(reason_counts.items(),
                                                                     key=lambda kv: -kv[1])) + '\n\n')

    # ---- Step 1 coverage report on the actual candidate rows (rank_not_selected + G1_veto) ------
    target = missed[missed.missed_reason.isin(['rank_not_selected', 'G1_veto'])].copy()
    rns_reason_map = rns['_ranking_reason']
    target.loc[target.missed_reason == 'rank_not_selected', '_bucket'] = \
        rns_reason_map.reindex(target.index[target.missed_reason == 'rank_not_selected'])
    target.loc[target.missed_reason == 'G1_veto', '_bucket'] = 'G1_veto'

    store = CachedStore(f1694.f1693.f1668.BARS_DB)
    target = walk_counterfactual(store, target)
    cov = 100.0 * (target['counterfactual_status'] == 'ok').mean()
    logger.info('COVERAGE after backfill on rank_not_selected+G1_veto (n=%d): %.1f%% ok', len(target),
                cov)
    append_result_md(f"## Step 1 -- coverage after backfill\n\n"
                      f"rank_not_selected + G1_veto candidates: {len(target)}; bar coverage now "
                      f"{cov:.1f}% ok (was 16.3% / 12.7% respectively before the backfill).\n\n"
                      f"By old-vs-new reason:\n\n"
                      + target.groupby('_bucket')['counterfactual_status'].apply(
                          lambda s: f"{(s=='ok').mean()*100:.1f}% ok (n={len(s)})").to_string() + '\n\n')
    target.to_csv(MISSED_FULL_CSV, index=False)
    logger.info('wrote %s (%d rows)', MISSED_FULL_CSV, len(target))

    # ---- Step 3: per reason x per half, with the taken book beside it ----------------------------
    target['half'] = target['date'].map(half_of)
    pf, _recon, _premkt_cov = f1694.build_fill_table(store)  # (df, recon_counts, premkt_coverage)
    pf['date'] = pf['date'].astype(str)
    pf = pf[(pf['date'] >= '2025-01-01') & (pf['date'] <= str(VAL_HI))].copy()  # match target's window
    pf['half'] = pf['date'].map(half_of)
    logger.info('taken book (in-range, matches target window): %d rows', len(pf))

    rows = []
    for bucket in ['score_threshold', 'skip_q1', 'dedup', 'slot_cap', 'G1_veto']:
        sub = target[target['_bucket'] == bucket]
        if len(sub) == 0:
            continue
        pooled = bucket_stats(sub)
        pooled.update(veto=bucket, half='pooled')
        rows.append(pooled)
        for half in ['2025', '2026']:
            hs = bucket_stats(sub[sub['half'] == half])
            hs.update(veto=bucket, half=half)
            rows.append(hs)
    # taken book beside it
    taken_pooled = dict(n=len(pf), n_ok=len(pf), coverage_pct=100.0,
                        mean_R=float(pf[BASELINE].mean()),
                        day_t=stats_block(pf[BASELINE].values, pf['date'].astype(str).values)['day_t'],
                        runner_share=float(pf['runner3'].mean()),
                        ex_top5=stats_block(pf[BASELINE].values, pf['date'].astype(str).values)['ex_top5'],
                        dollars_per_yr=float(pf[BASELINE].mean()) * len(pf) / YEARS_2025_2026 * FIXED_RISK_DOLLARS,
                        veto='TAKEN', half='pooled')
    rows.append(taken_pooled)
    for half in ['2025', '2026']:
        hp = pf[pf['half'] == half]
        if len(hp) == 0:
            continue
        st = stats_block(hp[BASELINE].values, hp['date'].astype(str).values)
        rows.append(dict(n=len(hp), n_ok=len(hp), coverage_pct=100.0, mean_R=st['mean_dR'],
                         day_t=st['day_t'], runner_share=float(hp['runner3'].mean()), ex_top5=st['ex_top5'],
                         dollars_per_yr=st['mean_dR'] * len(hp) / YEARS_2025_2026 * FIXED_RISK_DOLLARS,
                         veto='TAKEN', half=half))
    out = pd.DataFrame(rows)
    out.to_csv(REASONS_CSV, index=False)
    logger.info('wrote %s (%d rows)', REASONS_CSV, len(out))
    append_result_md('## Step 3 -- per reason x per half (taken book beside it)\n\n'
                      '| reason | half | n | n_ok | cov% | mean R | day_t | runner>=3R | ex_top5 | $/yr@375 |\n'
                      '|---|---|---|---|---|---|---|---|---|---|\n')
    for _, r in out.iterrows():
        append_result_md(f"| {r['veto']} | {r['half']} | {r['n']:.0f} | {r['n_ok']:.0f} | "
                          f"{r['coverage_pct']:.1f} | {r['mean_R']:+.3f} | {r['day_t']:.2f} | "
                          f"{r['runner_share']:.3f} | {r['ex_top5']:+.3f} | ${r['dollars_per_yr']:+,.0f} |\n")

    # ---- Coverage-bias check: originally-covered (16%/13%) vs newly-covered -----------------------
    orig_status = missed.set_index(['symbol', 'date'])['counterfactual_status']
    target_idx = target.set_index(['symbol', 'date'], drop=False)
    was_ok = target_idx.index.map(lambda k: orig_status.get(k) == 'ok')
    cov_bias_rows = []
    for label, mask in [('originally_covered_16pct', was_ok), ('newly_covered_after_backfill', ~was_ok)]:
        sub = target_idx[mask]
        s = bucket_stats(sub)
        s.update(group=label)
        cov_bias_rows.append(s)
    append_result_md('\n## Coverage-bias check (rank_not_selected + G1_veto pooled)\n\n'
                      '| group | n | n_ok | cov% | mean R | day_t | runner>=3R | $/yr@375 |\n'
                      '|---|---|---|---|---|---|---|---|\n')
    for r in cov_bias_rows:
        append_result_md(f"| {r['group']} | {r['n']:.0f} | {r['n_ok']:.0f} | {r['coverage_pct']:.1f} | "
                          f"{r['mean_R']:+.3f} | {r['day_t']:.2f} | {r['runner_share']:.3f} | "
                          f"${r['dollars_per_yr']:+,.0f} |\n")
    logger.info('coverage-bias check: %s', cov_bias_rows)

    # ---- Step 4: pre-declared promotion read -------------------------------------------------------
    append_result_md('\n## Step 4 -- promotion read (pre-declared: mean R >= +0.15 BOTH halves, '
                      'runner share >= 15%, pooled day_t >= 2.0)\n\n')
    promoted = []
    for bucket in ['score_threshold', 'skip_q1', 'dedup', 'slot_cap', 'G1_veto']:
        sub = target[target['_bucket'] == bucket]
        if len(sub) == 0:
            continue
        pooled = bucket_stats(sub)
        h25 = bucket_stats(sub[sub['half'] == '2025'])
        h26 = bucket_stats(sub[sub['half'] == '2026'])
        passes = (pooled['coverage_pct'] >= 99.0 and h25['mean_R'] >= 0.15 and h26['mean_R'] >= 0.15
                 and pooled['runner_share'] >= 0.15 and pooled['day_t'] >= 2.0)
        append_result_md(f"- **{bucket}**: cov {pooled['coverage_pct']:.1f}%, 2025 mean R "
                          f"{h25['mean_R']:+.3f}, 2026 mean R {h26['mean_R']:+.3f}, runner "
                          f"{pooled['runner_share']:.3f}, pooled day_t {pooled['day_t']:.2f} -> "
                          f"{'PASSES' if passes else 'fails'} the promotion bar\n")
        if passes:
            promoted.append(bucket)
    append_result_md(f"\nBuckets meeting the promotion bar: {promoted or 'NONE'}. Per PREREG: "
                      f"nothing changes in orb.yaml without the promoted cell passing both "
                      f"directions AND a rebuild -- not done in this cell.\n")
    logger.info('promotion read: %s', promoted or 'NONE')
    logger.info('=== cell 1,696 DONE ===')


if __name__ == '__main__':
    sys.exit(main())
