"""Independent rebuild of cells 1,483-1,485 (PREREG_1483.md) from prose + raw data.

Written by the independent-rebuilder agent WITHOUT reading cell_1483.py, cell_1483_fills.csv or
RESULT_1483.md during construction. Those three are opened only in the final compare_to_builder()
step, after this script's own kept sets and VAL means are already computed and frozen in memory --
mirroring CLAUDE.md's independent-reimplementation protocol (trade-by-trade compare after the fact,
never before).

Data-quality findings surfaced while building this (reported, not silently absorbed):
  1. The news fetch (fetch_full.log) crashed at row ~6600/9911 with `OSError: No space left on
     device` -- the node's root filesystem is at 100% (54M free of 100G, confirmed live via `df -h`
     at rebuild time). Fills at or after that position (day >= 2026-03-03, ALL of them VAL) never
     had a news-fetch attempt: 3,311 of 5,513 VAL fills (60%) are UNATTEMPTED, not "no_news" -- the
     two are conflated by any code that left-joins fill_item_map and defaults missing to no_news.
     This script keeps them as a distinct NaN/"unattempted" class and excludes them from cell 1,483's
     kept-set denominator, reporting coverage on the attempted-only population.
  2. labels_B (the second, independently-prompted Haiku pass used for the PREREG's classification
     robustness gate) has only ONE batch on disk (200 item-level labels), not the ~2,247 the full
     labels_A pass covers and nowhere near the PREREG's "500-item sample" -- almost certainly the
     same disk-full event truncated it after batch 0. This script uses ALL 200 available overlapping
     items (not a padded or fabricated 500) and reports n_compared honestly against the requested 500.
  3. XBRL runway coverage is 57.5% (4,211/9,911 fills have no matched cash/OCF row) -- below the
     PREREG's 70% coverage bar for cell 1,484 taken alone.
Both (1) and (3) independently breach the PREREG's "feature coverage >= 70% of fills (else VOID)"
pass-bar clause; this script computes the numbers anyway (that is the task) but the VOID flag is
carried in the output and must not be dropped when this is relayed upward.
"""
import json
import glob
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAUSAL_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
PRED_CSV = os.path.join(HERE, 'model_1478_predictions.csv')
RUNWAY_CSV = os.path.join(HERE, 'runway_1484.csv')
NEWS_DIR = os.path.join(HERE, 'news_1483')
BUILDER_CSV = os.path.join(HERE, 'cell_1483_fills.csv')
OUT_CSV = os.path.join(HERE, 'rebuild_1483_fills.csv')

HARD_CLASSES = {'earnings_guidance', 'fda_clinical', 'ma_strategic', 'contract_product'}
# Position in the status=='fill' row order at/after which fetch_full.log shows NO attempt
# (log's last completed progress line is 6600/9911; see docstring). 0-indexed cutoff.
FETCH_ATTEMPTED_CUTOFF = 6600
SEED = 1483


def log(msg):
    print(f'[rebuild_1483] {msg}', flush=True)


# ------------------------------------------------------------------------------------------------
# Step 1: base fills + outcome (the 1,478 standard, taken directly from model_1478_predictions.csv)
# ------------------------------------------------------------------------------------------------

def load_base_and_outcome():
    causal = pd.read_csv(CAUSAL_CSV, low_memory=False)
    fills = causal[causal.status == 'fill'].reset_index(drop=True)
    fills['fetch_attempted'] = fills.index < FETCH_ATTEMPTED_CUTOFF
    log(f'base fills: {len(fills)}; fetch_attempted=True for {fills.fetch_attempted.sum()} '
        f'({fills.fetch_attempted.mean():.1%}), split of the unattempted tail: '
        f'{fills.loc[~fills.fetch_attempted, "split"].value_counts().to_dict()}')

    pred = pd.read_csv(PRED_CSV)
    pred = pred[['day', 'symbol', 'fill_min', 'split', 'outcome_R']]
    merged = fills.merge(pred, on=['day', 'symbol', 'fill_min', 'split'], how='left')
    n_missing_outcome = merged['outcome_R'].isna().sum()
    log(f'outcome join: {n_missing_outcome} fills with no match in model_1478_predictions.csv '
        f'(dropped from all downstream means)')
    return merged


# ------------------------------------------------------------------------------------------------
# Step 2: catalyst classification -- labels_A (full pass) is the ONLY complete classification on
# disk, so it is what the kept sets are built from; labels_B (200-item partial second pass) is used
# ONLY for the A-vs-B robustness/agreement number the PREREG requires, never for the kept sets
# themselves (using B for both would make the agreement check circular).
# ------------------------------------------------------------------------------------------------

def load_labels_a():
    """dict[str article_id] -> (cls, materiality), from the full labels_A/labelsA_batch_*.json run."""
    out = {}
    paths = sorted(glob.glob(os.path.join(NEWS_DIR, 'labels_A', 'labelsA_batch_*.json')))
    for p in paths:
        d = json.load(open(p))
        for k, v in d.items():
            out[str(k)] = (v[0], v[1])
    log(f'labels_A: {len(paths)} batch files, {len(out)} unique article_ids classified')
    return out


def load_labels_b_and_a_sample():
    """labels_B/batch_000.json is a list of {item_id:'SYM__DAY::artid', cls, materiality} -- the
    PREREG's independently-prompted second pass. labels_A/batch_000.json is the SAME 200-item
    sample under the first prompt (identical item_id keys), used only for the A-vs-B agreement
    check, never merged into load_labels_a()'s full-coverage dict."""
    b_path = os.path.join(NEWS_DIR, 'labels_B', 'batch_000.json')
    a_sample_path = os.path.join(NEWS_DIR, 'labels_A', 'batch_000.json')
    b = json.load(open(b_path))
    a_sample = json.load(open(a_sample_path))
    bmap = {x['item_id']: (x['cls'], x['materiality']) for x in b}
    amap = {x['item_id']: (x['cls'], x['materiality']) for x in a_sample}
    return amap, bmap


def build_fill_articles():
    """dict[(day, symbol)] -> list of (item_id_str, created_at) from fill_item_map.parquet, for
    fills where a news fetch was actually attempted (see load_base_and_outcome)."""
    fm = pd.read_parquet(os.path.join(NEWS_DIR, 'fill_item_map.parquet'))
    out = {}
    for day, symbol, item_id, created_at in fm.itertuples(index=False):
        out.setdefault((day, symbol), []).append((str(item_id), created_at))
    return out


def classify_fills(fills, labels_a, fill_articles, respect_attempted_flag=True):
    """Per PREREG: catalyst class = highest materiality own-name article; latest created_at breaks
    ties. No own-name article (and fetch WAS attempted) -> 'no_news'. Fetch not attempted -> NaN
    (a coverage gap, not a class) when respect_attempted_flag=True (this script's strict, causally
    correct convention: NEVER-checked != checked-and-empty). With respect_attempted_flag=False,
    every fill without a mapped article defaults to 'no_news' regardless of the fetch_attempted
    flag -- this is the convention the builder's cell_1483_fills.csv 'class' column implies (every
    one of the 9,911 rows has a non-null class; see the value_counts diagnostic in
    compare_to_builder()), used here ONLY to test whether matching that convention closes the
    Jaccard gap -- it does not change which convention is correct, only isolates the cause.
    Returns (catalyst_class, materiality, hard_catalyst) arrays."""
    classes, materialities, hards = [], [], []
    n_article_missing_from_a = 0
    for day, symbol, attempted in zip(fills.day, fills.symbol, fills.fetch_attempted):
        if respect_attempted_flag and not attempted:
            classes.append(np.nan); materialities.append(np.nan); hards.append(np.nan)
            continue
        arts = fill_articles.get((day, symbol), [])
        candidates = []
        for item_id, created_at in arts:
            if item_id in labels_a:
                cls, mat = labels_a[item_id]
                candidates.append((mat, created_at, cls))
            else:
                n_article_missing_from_a += 1
        if not candidates:
            classes.append('no_news'); materialities.append(0); hards.append(False)
            continue
        candidates.sort(key=lambda t: (t[0], t[1]))  # highest materiality, latest created_at last
        best_mat, _, best_cls = candidates[-1]
        classes.append(best_cls); materialities.append(best_mat)
        hards.append(bool(best_cls in HARD_CLASSES and best_mat >= 2))
    if n_article_missing_from_a:
        log(f'WARNING: {n_article_missing_from_a} fetched articles had no labels_A classification '
            f'(excluded from the winning-article contest for their fill)')
    fills = fills.copy()
    fills['catalyst_class'] = classes
    fills['materiality'] = materialities
    fills['hard_catalyst'] = pd.array(hards, dtype='boolean')
    return fills


def class_agreement_a_vs_b(seed=SEED, n_requested=500):
    """A-vs-B classification-robustness check, PREREG gate >= 85%. Only 200 (fill,article) pairs
    have a labels_B entry on disk (see module docstring finding #2) -- ALL of them are used; no
    padding to 500. Returns (n_pool, n_used, n_agree, agreement_frac)."""
    amap, bmap = load_labels_b_and_a_sample()
    shared = sorted(set(amap) & set(bmap))
    rng = np.random.RandomState(seed)
    n_use = min(n_requested, len(shared))
    idx = rng.choice(len(shared), size=n_use, replace=False) if len(shared) > n_use else np.arange(len(shared))
    keys = [shared[i] for i in idx]
    n_agree = sum(1 for k in keys if amap[k][0] == bmap[k][0])
    frac = n_agree / len(keys) if keys else float('nan')
    log(f'class_agreement_a_vs_b: pool={len(shared)} (labels_B only has {len(bmap)} items total, '
        f'{n_requested} were requested by the task) n_used={len(keys)} n_agree={n_agree} '
        f'agreement={frac:.4f} (PREREG gate 0.85, {"PASS" if frac >= 0.85 else "FAIL"})')
    return len(shared), len(keys), n_agree, frac


# ------------------------------------------------------------------------------------------------
# Step 3: runway -- re-derive runway_q from cash/quarterly_ocf/cf_positive per the PREREG formula,
# never trust runway_1484.csv's own runway_q column.
# ------------------------------------------------------------------------------------------------

def attach_runway(fills):
    runway = pd.read_csv(RUNWAY_CSV)
    burn = (-runway['quarterly_ocf']).clip(lower=0)
    runway_q_mine = np.where(runway['cf_positive'].astype(bool), np.inf,
                              np.where(burn > 0, runway['cash'] / burn, np.inf))
    runway_q_mine = pd.Series(runway_q_mine, index=runway.index)
    runway_ok = runway['cf_positive'].astype(bool) | (runway_q_mine >= 4)
    runway_ok = runway_ok.where(runway['cash'].notna() & runway['quarterly_ocf'].notna())
    r = runway[['day', 'symbol']].copy()
    r['runway_q_mine'] = runway_q_mine
    r['runway_ok'] = runway_ok.astype('boolean')
    dup = r.duplicated(subset=['day', 'symbol']).sum()
    if dup:
        log(f'WARNING: {dup} duplicate (day,symbol) rows in runway_1484.csv, keeping first')
        r = r.drop_duplicates(subset=['day', 'symbol'], keep='first')
    n_before = len(fills)
    out = fills.merge(r, on=['day', 'symbol'], how='left')
    assert len(out) == n_before, f'runway merge changed row count {n_before} -> {len(out)}'
    log(f'runway coverage: {out.runway_ok.notna().mean():.1%} of fills have a cash+OCF match '
        f'(PREREG bar 70%, {"PASS" if out.runway_ok.notna().mean() >= 0.70 else "FAIL"})')
    return out


# ------------------------------------------------------------------------------------------------
# Step 4: kept sets + VAL means
# ------------------------------------------------------------------------------------------------

def val_mean(df, mask):
    sub = df.loc[mask & (df.split == 'VAL'), 'outcome_R']
    return float(sub.mean()), int(sub.notna().sum())


def build_kept_sets(fills):
    kept_1483 = fills['hard_catalyst'].fillna(False).astype(bool) & fills['hard_catalyst'].notna()
    kept_1484 = fills['runway_ok'].fillna(False).astype(bool) & fills['runway_ok'].notna()
    kept_1485 = kept_1483 & kept_1484
    fills = fills.copy()
    fills['kept_1483'] = kept_1483
    fills['kept_1484'] = kept_1484
    fills['kept_1485'] = kept_1485
    for name, mask in (('1483', kept_1483), ('1484', kept_1484), ('1485', kept_1485)):
        vm, n = val_mean(fills, mask)
        dm, dn = val_mean(fills, ~mask)
        log(f'cell {name}: VAL kept n={n} mean={vm:.4f}R | VAL dropped n={dn} mean={dm:.4f}R')
    return fills


# ------------------------------------------------------------------------------------------------
# Step 5: compare with the builder's cell_1483_fills.csv (opened ONLY here, after the above is
# already computed and logged -- see module docstring).
# ------------------------------------------------------------------------------------------------

def jaccard(a_keys, b_keys):
    a, b = set(a_keys), set(b_keys)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def verdict(jac, mean_diff):
    if jac >= 0.90 and mean_diff <= 0.03:
        return 'REPRODUCED'
    if jac >= 0.80 and mean_diff <= 0.06:
        return 'PARTIAL'
    return 'NOT_REPRODUCED'


def compare_to_builder(mine):
    if not os.path.exists(BUILDER_CSV):
        log(f'WARNING: {BUILDER_CSV} not found, skipping comparison')
        return {}
    theirs = pd.read_csv(BUILDER_CSV, low_memory=False)
    log(f"builder cell_1483_fills.csv columns: {theirs.columns.tolist()}")
    # The builder's own column names for the three kept-flags, discovered from the header above
    # (hard_catalyst = 1483's flag, runway_ok = 1484's, 1485 = their AND).
    theirs_bool_by_cell = {
        '1483': theirs['hard_catalyst'].astype('boolean').fillna(False).astype(bool),
        '1484': theirs['runway_ok'].astype('boolean').fillna(False).astype(bool),
    }
    theirs_bool_by_cell['1485'] = theirs_bool_by_cell['1483'] & theirs_bool_by_cell['1484']

    outcome_col_candidates = [c for c in theirs.columns if 'outcome' in c.lower() or c == 'net_R']
    log(f'builder outcome column candidates: {outcome_col_candidates}')

    results = {}
    mine_key = list(zip(mine.day, mine.symbol, mine.fill_min))
    theirs_key_col = list(zip(theirs.day, theirs.symbol,
                              theirs.fill_min if 'fill_min' in theirs.columns
                              else [None] * len(theirs)))
    for cell in ('1483', '1484', '1485'):
        theirs_bool = theirs_bool_by_cell[cell]
        mine_keys = [k for k, keep in zip(mine_key, mine[f'kept_{cell}']) if keep]
        theirs_keys = [k for k, keep in zip(theirs_key_col, theirs_bool) if keep]
        jac = jaccard(mine_keys, theirs_keys)

        my_vm, my_n = val_mean(mine, mine[f'kept_{cell}'])
        theirs_outcome_col = outcome_col_candidates[0] if outcome_col_candidates else None
        their_vm, their_n = (float('nan'), 0)
        if theirs_outcome_col and 'split' in theirs.columns:
            sub = theirs.loc[theirs_bool & (theirs.split == 'VAL'), theirs_outcome_col]
            their_vm, their_n = float(sub.mean()), int(sub.notna().sum())
        mean_diff = abs(my_vm - their_vm) if their_n else float('nan')
        v = verdict(jac, mean_diff) if their_n else 'NOT_REPRODUCED (no builder outcome column found)'
        log(f'cell {cell}: jaccard={jac:.4f} my_VAL_mean={my_vm:.4f}(n={my_n}) '
            f'builder_VAL_mean={their_vm:.4f}(n={their_n}) mean_diff={mean_diff:.4f} verdict={v}')
        results[cell] = dict(jaccard=jac, my_val_mean=my_vm, my_val_n=my_n,
                              builder_val_mean=their_vm, builder_val_n=their_n,
                              mean_diff=mean_diff, verdict=v)

    # Diagnostic: how did the builder's 'class' column populate, vs this script's fetch_attempted
    # split -- explains any large Jaccard gap on cell 1483 (see module docstring finding #1).
    if 'class' in theirs.columns:
        log(f"builder 'class' value_counts (all splits): {theirs['class'].value_counts().to_dict()}")
        if 'split' in theirs.columns:
            for sp in ('TRAIN', 'VAL'):
                vc = theirs.loc[theirs.split == sp, 'class'].value_counts()
                log(f"builder 'class' value_counts split={sp}: {vc.to_dict()}")
    return results


def compare_alt_to_builder(alt):
    """Same Jaccard/mean-diff comparison as compare_to_builder(), but against the alt (no_news-
    default) convention's kept sets, to check whether matching the builder's implicit convention
    for missing-article fills closes cell 1483/1485's gap."""
    if not os.path.exists(BUILDER_CSV):
        return {}
    theirs = pd.read_csv(BUILDER_CSV, low_memory=False)
    theirs_hard = theirs['hard_catalyst'].astype('boolean').fillna(False).astype(bool)
    theirs_runway = theirs['runway_ok'].astype('boolean').fillna(False).astype(bool)
    theirs_bool_by_cell = {'1483': theirs_hard, '1485': theirs_hard & theirs_runway}
    alt_key = list(zip(alt.day, alt.symbol, alt.fill_min))
    theirs_key_col = list(zip(theirs.day, theirs.symbol, theirs.fill_min))
    for cell in ('1483', '1485'):
        theirs_bool = theirs_bool_by_cell[cell]
        alt_keys = [k for k, keep in zip(alt_key, alt[f'kept_{cell}']) if keep]
        theirs_keys = [k for k, keep in zip(theirs_key_col, theirs_bool) if keep]
        jac = jaccard(alt_keys, theirs_keys)
        my_vm, my_n = val_mean(alt, alt[f'kept_{cell}'])
        sub = theirs.loc[theirs_bool & (theirs.split == 'VAL'), 'outcome_R']
        their_vm, their_n = float(sub.mean()), int(sub.notna().sum())
        mean_diff = abs(my_vm - their_vm)
        log(f'cell {cell} (alt no_news-default convention): jaccard={jac:.4f} '
            f'my_VAL_mean={my_vm:.4f}(n={my_n}) builder_VAL_mean={their_vm:.4f}(n={their_n}) '
            f'mean_diff={mean_diff:.4f} verdict={verdict(jac, mean_diff)}')


def main():
    df_pool, n_used, n_agree, agree_frac = class_agreement_a_vs_b()

    fills = load_base_and_outcome()
    labels_a = load_labels_a()
    fill_articles = build_fill_articles()
    fills = classify_fills(fills, labels_a, fill_articles, respect_attempted_flag=True)
    fills = attach_runway(fills)
    fills = build_kept_sets(fills)

    # Alt convention (diagnostic only, see classify_fills docstring): match the builder's implicit
    # "no article mapped -> no_news regardless of fetch_attempted" default, to isolate whether the
    # coverage-gap convention (not a code bug) explains cell 1483/1485's Jaccard gap.
    alt = load_base_and_outcome()
    alt = classify_fills(alt, labels_a, fill_articles, respect_attempted_flag=False)
    alt = attach_runway(alt)
    alt = build_kept_sets(alt)
    for name, mask in (('1483_altnodefault', alt['kept_1483']), ('1485_altnodefault', alt['kept_1485'])):
        vm, n = val_mean(alt, mask)
        log(f'cell {name}: VAL kept n={n} mean={vm:.4f}R (using no_news-default convention)')
    compare_alt_to_builder(alt)

    out_cols = ['day', 'symbol', 'split', 'fill_min', 'fetch_attempted', 'catalyst_class',
                'materiality', 'hard_catalyst', 'runway_q_mine', 'runway_ok', 'outcome_R',
                'kept_1483', 'kept_1484', 'kept_1485']
    fills[out_cols].to_csv(OUT_CSV, index=False)
    log(f'wrote {OUT_CSV} ({len(fills)} rows)')

    cmp_results = compare_to_builder(fills)

    log('=== SUMMARY ===')
    log(f'class_agreement_A_vs_B: n_requested=500 pool={df_pool} n_used={n_used} '
        f'n_agree={n_agree} agreement={agree_frac:.4f} gate=0.85 '
        f'{"PASS" if agree_frac >= 0.85 else "FAIL"}')
    for cell, r in cmp_results.items():
        log(f'cell {cell}: {r}')
    log('VOID flags (PREREG coverage>=70% clause): '
        f'news fetch attempted on {fills.fetch_attempted.mean():.1%} of fills '
        f'({"OK" if fills.fetch_attempted.mean() >= 0.70 else "VOID-triggering"}); '
        f'runway matched on {fills.runway_ok.notna().mean():.1%} of fills '
        f'({"OK" if fills.runway_ok.notna().mean() >= 0.70 else "VOID-triggering"})')


if __name__ == '__main__':
    main()
