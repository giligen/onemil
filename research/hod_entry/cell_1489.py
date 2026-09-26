"""
Cell 1,489 (BUY) / 1,490 (SHORT) -- retest-instant classifier.

PREREG: research/hod_entry/PREREG_1489.md (protocol ref research/hod_entry/PREREG_1478.md).
Feature matrix: research/hod_entry/features_1489.csv (built per FEATURES_1489.md).

Fits HistGradientBoostingClassifier (seed 1489) + LogisticRegression on the non-decoy
features, 5-fold CV inside TRAIN over the PREREG_1478 fixed grid, fit once on TRAIN,
scored once on VAL. Thresholds (TRAIN-only): top-tercile p_hgb for 1,489 BUY, bottom-tercile
p_hgb for 1,490 SHORT. Also fits a label-shuffled placebo and a decoy-only metadata model.

Writes:
  research/hod_entry/model_1489_predictions.csv
  research/hod_entry/RESULT_1489.md
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

SEED = 1489
FEAT_CSV = "research/hod_entry/features_1489.csv"
PRED_CSV = "research/hod_entry/model_1489_predictions.csv"
RESULT_MD = "research/hod_entry/RESULT_1489.md"

GRID = [
    {"max_depth": md, "learning_rate": lr, "max_iter": mi, "min_samples_leaf": msl}
    for md in (3, 5)
    for lr in (0.03, 0.1)
    for mi in (200, 600)
    for msl in (50, 200)
]

ID_COLS = ["day", "symbol", "split", "retest_ts"]
LABEL_COLS = ["Y", "net_R_prime", "base_net_R"]
BOOKKEEPING_1490 = ["c1490_short_net_R", "c1490_shortable", "c1490_ssr"]
DECOY_COLS = [
    "decoy_store_served_1438", "decoy_rth_bar_count_1438",
    "decoy_tick_window_has_bar_j", "decoy_window_found_1478C",
    "decoy_has_prebreak_1478C",
]
# redundant / non-market metadata excluded from the real feature set (documented in caveats)
DROP_EXTRA = ["store_served_1438", "arm_arm_m", "arm_arm_minute", "tape_coverage", "tape_source"]

BOOLLIKE = ["dip_bid_stepped_down_thru_level_5s"]


def to_num_bool(s):
    return s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)


def day_clustered_t(day, x):
    """t-stat of per-day means against zero (day is the cluster unit)."""
    d = pd.DataFrame({"day": day, "x": x}).dropna()
    means = d.groupby("day")["x"].mean()
    n = len(means)
    if n < 2 or means.std(ddof=1) == 0 or np.isnan(means.std(ddof=1)):
        return float("nan")
    return float(means.mean() / (means.std(ddof=1) / np.sqrt(n)))


def ex_top5(x):
    x = pd.Series(x).dropna().sort_values(ascending=False)
    if len(x) == 0:
        return float("nan")
    k = int(np.ceil(0.05 * len(x)))
    rest = x.iloc[k:]
    return float(rest.mean()) if len(rest) else float("nan")


def fills_per_week(day_series):
    if len(day_series) == 0:
        return 0.0
    wk = pd.to_datetime(day_series).dt.to_period("W")
    n_weeks = wk.nunique()
    return len(day_series) / n_weeks if n_weeks else float("nan")


def best_cv_params(X, y, seed):
    best_auc, best_params = -1, GRID[0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for params in GRID:
        model = HistGradientBoostingClassifier(random_state=seed, **params)
        try:
            scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        except ValueError:
            continue
        m = np.nanmean(scores)
        print(f"  grid {params} cv_auc={m:.4f}", flush=True)
        if m > best_auc:
            best_auc, best_params = m, params
    return best_params, best_auc


def main():
    df = pd.read_csv(FEAT_CSV, low_memory=False)
    for c in BOOLLIKE + BOOKKEEPING_1490[1:]:
        df[c] = to_num_bool(df[c])
    df["c1490_short_net_R"] = pd.to_numeric(df["c1490_short_net_R"], errors="coerce")

    exclude = set(ID_COLS + LABEL_COLS + BOOKKEEPING_1490 + DECOY_COLS + DROP_EXTRA)
    feat_cols = [c for c in df.columns if c not in exclude]
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in feat_cols), "non-numeric feature leaked in"

    tr = df[df["split"] == "TRAIN"].copy()
    va = df[df["split"] == "VAL"].copy()
    Xtr, ytr = tr[feat_cols], tr["Y"].astype(int)
    Xva, yva = va[feat_cols], va["Y"].astype(int)

    # ---- real HGB model ----
    params, cv_auc = best_cv_params(Xtr, ytr, SEED)
    hgb = HistGradientBoostingClassifier(random_state=SEED, **params).fit(Xtr, ytr)
    p_hgb_tr = hgb.predict_proba(Xtr)[:, 1]
    p_hgb_va = hgb.predict_proba(Xva)[:, 1]
    val_auc = roc_auc_score(yva, p_hgb_va)
    train_auc = roc_auc_score(ytr, p_hgb_tr)

    # ---- LR beside (median impute + standardize) ----
    lr = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ]).fit(Xtr, ytr)
    p_lr_tr = lr.predict_proba(Xtr)[:, 1]
    p_lr_va = lr.predict_proba(Xva)[:, 1]

    # ---- placebo: label-shuffled TRAIN, same grid winner, refit ----
    rng = np.random.RandomState(SEED)
    ytr_shuf = ytr.sample(frac=1, random_state=SEED).reset_index(drop=True).values
    placebo = HistGradientBoostingClassifier(random_state=SEED, **params).fit(Xtr, ytr_shuf)
    placebo_auc = roc_auc_score(yva, placebo.predict_proba(Xva)[:, 1])

    # ---- decoy: metadata-only model ----
    Xtr_dec, Xva_dec = tr[DECOY_COLS], va[DECOY_COLS]
    dparams, _ = best_cv_params(Xtr_dec, ytr, SEED)
    decoy = HistGradientBoostingClassifier(random_state=SEED, **dparams).fit(Xtr_dec, ytr)
    decoy_auc = roc_auc_score(yva, decoy.predict_proba(Xva_dec)[:, 1])
    decoy_void = decoy_auc > 0.55

    # ---- thresholds from TRAIN only ----
    threshold_buy = float(np.quantile(p_hgb_tr, 2 / 3))    # top tercile
    threshold_short = float(np.quantile(p_hgb_tr, 1 / 3))  # bottom tercile

    tr = tr.assign(p_hgb=p_hgb_tr, p_lr=p_lr_tr)
    va = va.assign(p_hgb=p_hgb_va, p_lr=p_lr_va)

    pop_cache_share_tr = tr["decoy_store_served_1438"].mean()
    pop_cache_share_va = va["decoy_store_served_1438"].mean()

    def cell_1489_row(holdout_df, holdout_name):
        n_pop = len(holdout_df)
        kept = holdout_df[holdout_df["p_hgb"] >= threshold_buy]
        dropped = holdout_df[holdout_df["p_hgb"] < threshold_buy]
        n_kept = len(kept)
        kept_mean = float(kept["net_R_prime"].mean()) if n_kept else float("nan")
        t_kept = day_clustered_t(kept["day"], kept["net_R_prime"])
        ex5 = ex_top5(kept["net_R_prime"])
        dropped_mean = float(dropped["net_R_prime"].mean()) if len(dropped) else float("nan")
        auc = roc_auc_score(holdout_df["Y"], holdout_df["p_hgb"]) if holdout_df["Y"].nunique() > 1 else float("nan")
        fwk = fills_per_week(kept["day"])
        cache_kept = kept["decoy_store_served_1438"].mean() if n_kept else float("nan")
        pop_cache = pop_cache_share_tr if holdout_name == "TRAIN" else pop_cache_share_va
        paired = float((kept["net_R_prime"] - kept["base_net_R"]).mean()) if n_kept else float("nan")
        passes = (
            holdout_name != "VAL" or (
                kept_mean >= 0.15 and t_kept >= 2.5 and (ex5 > 0) and fwk >= 3
                and dropped_mean < kept_mean and val_auc >= 0.60 and placebo_auc <= 0.53
                and not decoy_void and abs(cache_kept - pop_cache) <= 0.05 and paired >= 0.10
            )
        )
        return dict(
            cell="1489", holdout=holdout_name, n_pop=n_pop, n_kept=n_kept,
            kept_mean=kept_mean, t_kept=t_kept, ex_top5=ex5, dropped_mean=dropped_mean,
            auc=auc, placebo_auc=placebo_auc, decoy_auc=decoy_auc,
            cacheonly_share_kept=cache_kept, cacheonly_share_pop=pop_cache,
            fills_wk=fwk, paired_dR_vs_base=paired, passes_bar=bool(passes),
        )

    rows = [cell_1489_row(tr, "TRAIN"), cell_1489_row(va, "VAL")]

    # ---- cell 1,490 SHORT: matched subset only (shortable & not SSR) ----
    def cell_1490_row(holdout_df, holdout_name):
        matched = holdout_df[(holdout_df["c1490_shortable"] == 1.0) & (holdout_df["c1490_ssr"] == 0.0)]
        n_pop = len(matched)
        note = f"match rate {n_pop}/{len(holdout_df)} ({100*n_pop/len(holdout_df):.1f}%)" if len(holdout_df) else ""
        if n_pop == 0:
            return dict(
                cell="1490", holdout=holdout_name, n_pop=0, n_kept=0,
                kept_mean=float("nan"), t_kept=float("nan"), ex_top5=float("nan"),
                dropped_mean=float("nan"), auc=float("nan"), placebo_auc=placebo_auc,
                decoy_auc=decoy_auc, cacheonly_share_kept=float("nan"),
                cacheonly_share_pop=float("nan"), fills_wk=0.0,
                passes_bar=False, note=note + " -- NOT SCORABLE, no matched short outcomes in this holdout",
            )
        kept = matched[matched["p_hgb"] <= threshold_short]
        dropped = matched[matched["p_hgb"] > threshold_short]
        n_kept = len(kept)
        kept_mean = float(kept["c1490_short_net_R"].mean()) if n_kept else float("nan")
        t_kept = day_clustered_t(kept["day"], kept["c1490_short_net_R"])
        ex5 = ex_top5(kept["c1490_short_net_R"])
        dropped_mean = float(dropped["c1490_short_net_R"].mean()) if len(dropped) else float("nan")
        auc = roc_auc_score(matched["Y"], matched["p_hgb"]) if matched["Y"].nunique() > 1 else float("nan")
        fwk = fills_per_week(kept["day"])
        cache_kept = kept["decoy_store_served_1438"].mean() if n_kept else float("nan")
        cache_pop = matched["decoy_store_served_1438"].mean()
        passes = (
            holdout_name == "VAL" and n_pop >= 1 and (
                kept_mean >= 0.15 and t_kept >= 2.5 and (ex5 > 0) and fwk >= 3
                and dropped_mean < kept_mean and val_auc >= 0.60 and placebo_auc <= 0.53
                and not decoy_void and abs(cache_kept - cache_pop) <= 0.05
            )
        )
        return dict(
            cell="1490", holdout=holdout_name, n_pop=n_pop, n_kept=n_kept,
            kept_mean=kept_mean, t_kept=t_kept, ex_top5=ex5, dropped_mean=dropped_mean,
            auc=auc, placebo_auc=placebo_auc, decoy_auc=decoy_auc,
            cacheonly_share_kept=cache_kept, cacheonly_share_pop=cache_pop,
            fills_wk=fwk, passes_bar=bool(passes), note=note,
        )

    rows += [cell_1490_row(tr, "TRAIN"), cell_1490_row(va, "VAL")]

    # ---- permutation importance on VAL (HGB) ----
    perm = permutation_importance(hgb, Xva, yva, n_repeats=5, random_state=SEED, scoring="roc_auc", n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=feat_cols).sort_values(ascending=False)
    top10 = imp.head(10)

    # ---- predictions file ----
    out = pd.concat([tr, va], ignore_index=True)[
        ["day", "symbol", "split", "p_hgb", "p_lr", "net_R_prime", "c1490_short_net_R"]
    ]
    out["kept_1489"] = out["p_hgb"] >= threshold_buy
    out["kept_1490"] = out["p_hgb"] <= threshold_short
    out.to_csv(PRED_CSV, index=False)

    # ---- results markdown ----
    lines = []
    lines.append("# RESULT — cells 1,489 (BUY) / 1,490 (SHORT): retest-instant classifier\n")
    lines.append(f"Best HGB params (5-fold CV AUC {cv_auc:.3f} inside TRAIN): {params}\n")
    lines.append(f"TRAIN AUC (in-sample) {train_auc:.3f}; VAL AUC {val_auc:.3f}; placebo VAL AUC {placebo_auc:.3f}; "
                  f"decoy VAL AUC {decoy_auc:.3f} ({'VOID -- leaks label' if decoy_void else 'ok, <=0.55'})\n")
    lines.append(f"threshold_buy (TRAIN top-tercile p_hgb) = {threshold_buy:.4f}; "
                  f"threshold_short (TRAIN bottom-tercile p_hgb) = {threshold_short:.4f}\n")
    lines.append(f"Base rate Y=1: TRAIN {ytr.mean():.3f}, VAL {yva.mean():.3f}\n")

    cols = ["cell", "holdout", "n_pop", "n_kept", "kept_mean", "t_kept", "ex_top5",
            "dropped_mean", "auc", "fills_wk", "cacheonly_share_kept", "cacheonly_share_pop",
            "paired_dR_vs_base", "passes_bar"]
    lines.append("\n## Per cell x holdout\n")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = f"{v:.4f}" if not np.isnan(v) else "NaN"
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("\n## Notes on 1,490 (SHORT)\n")
    for r in rows:
        if r["cell"] == "1490" and "note" in r:
            lines.append(f"- {r['holdout']}: {r['note']}")

    lines.append("\n## Top-10 VAL permutation importances (HGB, AUC drop)\n")
    lines.append("| feature | importance |")
    lines.append("|---|---|")
    for k, v in top10.items():
        lines.append(f"| {k} | {v:.4f} |")

    lines.append("\n## Caveats\n")
    lines.append("- 1,490 SHORT has ZERO matched short outcomes in VAL: rebuild_1479_1480.csv's short leg was "
                  "built only on the TRAIN-period sample (2025-07-01..2025-12-31); all 406 matched rows "
                  "(253 shortable & non-SSR) fall in TRAIN, VAL match count = 0. Cell 1,490 is therefore "
                  "NOT SCORABLE against the frozen pass bar (VAL kept mean requires VAL rows) and FAILS by "
                  "construction/data-coverage, not by an estimated negative edge. TRAIN-only diagnostics are "
                  "reported for information, never as a pass.")
    lines.append("- Match rate for 1,490's population overall: 406/8973 fills (4.5%), as flagged in FEATURES_1489.md; "
                  "of those, 253 are shortable & non-SSR (the primary book population).")
    lines.append("- `ctx_n_prior_retests_same_level` has zero variance in this population (always 0); included per "
                  "spec but contributes no signal.")
    lines.append("- `arm_arm_m` / `arm_arm_minute` (redundant identity index) and `tape_coverage` / `tape_source` "
                  "(constant / near-constant provenance flags) excluded from the real feature set as non-market "
                  "metadata; `store_served_1438` excluded as a duplicate of `decoy_store_served_1438`.")
    lines.append("- Breadth-at-retest-minute (PREREG item 4) is not implemented as specified (FEATURES_1489.md): "
                  "`arm_breadth_share_j`/`arm_breadth_count_j` (breadth at the ARM bar, median 0.85 min earlier) "
                  "stand in as the causal proxy, disclosed there.")
    lines.append("- `ctx_spy_ret_fill_to_tr` is NaN for 70% of rows (cache.db SPY series ends 2026-03-20, "
                  "bars_fills_1478.db carries no SPY rows) -- HGB handles this natively; LR median-imputes it, "
                  "which is a real information loss for the simple model on the majority of 2026 rows.")
    lines.append("- Day-clustered t computed on per-day mean returns (weighting every day equally), per the "
                  "programme's standing convention.")
    lines.append("- `dip_bid_stepped_down_thru_level_5s`, `c1490_shortable`, `c1490_ssr` were stored as "
                  "True/False/NaN objects in the CSV and were cast to 1.0/0.0/NaN before modeling.")

    with open(RESULT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("DONE")
    for r in rows:
        print(r)
    print("top10 importances:")
    print(top10)


if __name__ == "__main__":
    main()
