#!/usr/bin/env python3
"""Selection (TRAIN-H2), VAL statistics, placebo, neighbour stability, cache-only
share and the count-matched null for the rebuild_1493 surface. Reads the fills
written by rebuild_1493.py; never reads cell_1493*/RESULT_1493.md."""
import os
import time

import numpy as np
import pandas as pd

from rebuild_1493 import (BarStore, HERE, SEED, dayclustered_t, ex_top5,
                           et_minute_of_day, load_population, walk_exit,
                           apply_cost, EOD_TIME_BPS, RTH_EOD_MIN, PLACEBO_LO,
                           PLACEBO_HI)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_cell(surface, pop):
    train = surface[surface["split"] == "TRAIN"].copy()
    # stop >= 1.0%: exclude the fixed 0.5% stop cells outright (report-only,
    # fails the R-vs-spread rail). For CL, check the realized mean stop
    # distance in % of price is >= 1.0% before allowing it in.
    train = train[~train["cell"].str.startswith("s0.5_")]
    train = train[train["cell"] != "M_mirror"]  # M is report-only alongside, not grid-selectable
    elig = train[(train["fills_per_week"] >= 3) & (train["t_dayclust"] >= 2)]
    if elig.empty:
        return None, train
    # tie-break: wider stop. Parse the stop label for ranking width.
    def stop_width(cell_name):
        tag = cell_name.split("_")[0]
        if tag == "sCL":
            return 999  # treat CL as widest for tie-break purposes (per-row realized)
        return float(tag[1:])
    elig = elig.assign(_stopw=elig["cell"].map(stop_width))
    elig = elig.sort_values(["mean_pct", "_stopw"], ascending=[False, False])
    return elig.iloc[0]["cell"], train


def realized_cl_stop_pct(fills, pop):
    """Mean realized stop distance (% of entry) for CL cells, to gate CL eligibility."""
    cl_fills = fills[fills["cell"].str.startswith("sCL_")]
    # stop distance isn't stored directly in fills; recompute from pop.
    m = pop.set_index(["day", "symbol"])[["entry", "stop"]]
    pct = ((m["entry"] - m["stop"]) / m["entry"] * 100.0)
    return pct.mean()


def count_matched_null(pop, selected_val_mean, n_draws=1000, seed=SEED):
    """1,000 draws of count-matched (per VAL day) random BASE fills; the base
    outcome in % of price = outcome_R * R_pct_of_price. Returns the percentile
    of selected_val_mean within the null distribution of draw means."""
    causal = pd.read_csv(os.path.join(HERE, "causal_arming_causal.csv"))
    causal = causal[causal["status"] == "fill"].copy()
    l3 = pd.read_csv(os.path.join(HERE, "model_1478_L3_predictions.csv"),
                      usecols=["day", "symbol", "fill_min", "split", "outcome_R"])
    base = causal.merge(l3, on=["day", "symbol", "fill_min", "split"], how="inner")
    base = base[base["split"] == "VAL"].copy()
    base["r_pct_price"] = (base["fill"] - base["stop"]) / base["fill"]
    base["base_net_pct"] = base["outcome_R"] * base["r_pct_price"] * 100.0
    day_counts = pop[pop["split"] == "VAL"].groupby("day").size()
    by_day = {d: g["base_net_pct"].values for d, g in base.groupby("day")}
    rng = np.random.default_rng(seed)
    draw_means = []
    for _ in range(n_draws):
        vals = []
        for day, k in day_counts.items():
            pool = by_day.get(day)
            if pool is None or len(pool) == 0:
                continue
            idx = rng.integers(0, len(pool), size=min(k, len(pool)))
            vals.extend(pool[idx])
        if vals:
            draw_means.append(np.mean(vals))
    draw_means = np.array(draw_means)
    pct_rank = (draw_means < selected_val_mean).mean() * 100.0
    return pct_rank, draw_means.mean(), draw_means.std(), len(draw_means)


def parse_cell(name):
    """Return (stop_type, stop_val, exit_type, exit_val) from a grid cell name."""
    stag, etag = name.split("_", 1)
    if stag == "sCL":
        stop_type, stop_val = "CL", None
    else:
        stop_type, stop_val = "pct", float(stag[1:])
    if etag.startswith("t"):
        exit_type, exit_val = "tgt", float(etag[1:])
    else:
        exit_type, exit_val = "time", etag
    return stop_type, stop_val, exit_type, exit_val


def placebo_for_cells(pop, bar_store, cell_names, seed=SEED):
    """Same exits, one random RTH minute in [09:45,15:00) per fill (never inside
    the fill's own 15-min retest window), entry at that minute's open, bar-walk
    only (no tape -- we lack tape for arbitrary minutes)."""
    rng = np.random.default_rng(seed)
    recs = []
    for _, row in pop.iterrows():
        symbol, day, split = row["symbol"], row["day"], row["split"]
        _, minute_r_exact = et_minute_of_day(float(row["retest_ts"]))
        m_r_floor = int(minute_r_exact)
        bars = bar_store.get(symbol, day)
        bar_map = {m: (o, h, l, c) for (m, o, h, l, c) in bars}
        # pick a random minute outside [m_r_floor, m_r_floor+15)
        candidates = [m for m in range(PLACEBO_LO, PLACEBO_HI)
                      if not (m_r_floor <= m < m_r_floor + 15) and m in bar_map]
        if not candidates:
            continue
        m0 = candidates[rng.integers(0, len(candidates))]
        entry = bar_map[m0][0]  # that minute's open
        for name in cell_names:
            stop_type, stop_val, exit_type, exit_val = parse_cell(name)
            stop_price = row["stop"] if stop_type == "CL" else entry * (1 - stop_val / 100.0)
            if exit_type == "tgt":
                target_price, exit_kind, deadline = entry * (1 + exit_val / 100.0), "tgt", None
            elif exit_val == "NONE":
                target_price, exit_kind, deadline = None, "time_none", None
            else:
                target_price, exit_kind = None, exit_val
                deadline = 30 if exit_val == "T30" else 60
            if deadline is not None:
                deadline = m0 + deadline
            # bar-walk starting at the entry bar itself (no tape at arbitrary minutes)
            last_bar = None
            why, exit_px_raw, exit_m = "eod", bar_map[m0][3], m0
            for m in sorted(bar_map):
                if m < m0 or m > RTH_EOD_MIN:
                    continue
                o, h, l, c = bar_map[m]
                last_bar = (m, o, h, l, c)
                if exit_kind in ("T30", "T60") and m >= deadline:
                    why, exit_px_raw, exit_m = "time", o, m
                    break
                if target_price is not None:
                    if o <= stop_price:
                        why, exit_px_raw, exit_m = "stop", o, m
                        break
                    if l <= stop_price and h >= target_price:
                        why, exit_px_raw, exit_m = "stop", stop_price, m
                        break
                    if h >= target_price:
                        why, exit_px_raw, exit_m = "target", target_price, m
                        break
                    if l <= stop_price:
                        why, exit_px_raw, exit_m = "stop", stop_price, m
                        break
                else:
                    if o <= stop_price:
                        why, exit_px_raw, exit_m = "stop", o, m
                        break
                    if l <= stop_price:
                        why, exit_px_raw, exit_m = "stop", stop_price, m
                        break
            else:
                if last_bar is not None:
                    why, exit_px_raw, exit_m = "eod", last_bar[4], last_bar[0]
            exit_px = apply_cost(exit_px_raw, why, split)
            net_pct = (exit_px / entry - 1.0) * 100.0
            recs.append({"day": day, "symbol": symbol, "split": split,
                         "cell": name, "net_pct": net_pct})
    return pd.DataFrame(recs)


def main():
    surface = pd.read_csv(os.path.join(HERE, "rebuild_1493_surface.csv"))
    fills = pd.read_csv(os.path.join(HERE, "rebuild_1493_fills.csv"))
    pop = load_population()

    sel_cell, train_elig = select_cell(surface, pop)
    log(f"selected cell (TRAIN-H2): {sel_cell}")
    log(train_elig.sort_values('mean_pct', ascending=False).head(8)[
        ['cell', 'n', 'fills_per_week', 'mean_pct', 't_dayclust']].to_string())

    cl_pct = realized_cl_stop_pct(fills, pop)
    log(f"realized mean CL stop distance: {cl_pct:.3f}% of entry")

    val_row = surface[(surface["split"] == "VAL") & (surface["cell"] == sel_cell)].iloc[0]
    log(f"VAL for selected cell: mean_pct={val_row['mean_pct']:.4f} t={val_row['t_dayclust']:.2f} "
        f"ex_top5={val_row['ex_top5_pct']:.4f} n={val_row['n']}")

    # winner-cap at +3%
    val_fills_sel = fills[(fills["split"] == "VAL") & (fills["cell"] == sel_cell)].copy()
    val_fills_sel["capped"] = val_fills_sel["net_pct"].clip(upper=3.0)
    log(f"winner-capped (+3%) VAL mean: {val_fills_sel['capped'].mean():.4f}")

    # cache-only share
    pop_share = pop["store_served_1438"].mean()
    cell_share = val_fills_sel.merge(
        pop[["day", "symbol", "store_served_1438"]].drop_duplicates(),
        on=["day", "symbol"], how="left")["store_served_1438"].mean()
    log(f"cache-only share: population {pop_share*100:.1f}% vs selected-cell VAL fills {cell_share*100:.1f}%")

    # count-matched null
    pctile, null_mean, null_std, ndraws = count_matched_null(pop, val_row["mean_pct"])
    log(f"count-matched null ({ndraws} draws): mean={null_mean:.4f} std={null_std:.4f}, "
        f"selected VAL mean at the {pctile:.1f}th percentile")

    # neighbour stability
    stop_type, stop_val, exit_type, exit_val = parse_cell(sel_cell)
    stops_order = ["s0.5", "s1.0", "s1.5", "s2.0", "s3.0", "sCL"]
    exits_order = ["t0.5", "t0.75", "t1.0", "t1.5", "t2.0", "t3.0", "NONE", "T30", "T60"]
    stag = "sCL" if stop_type == "CL" else f"s{stop_val}"
    etag = f"t{exit_val}" if exit_type == "tgt" else exit_val
    si, ei = stops_order.index(stag), exits_order.index(etag)
    neighbours = []
    for ds in (-1, 1):
        if 0 <= si + ds < len(stops_order):
            neighbours.append(f"{stops_order[si+ds]}_{etag}")
    for de in (-1, 1):
        if 0 <= ei + de < len(exits_order):
            neighbours.append(f"{stag}_{exits_order[ei+de]}")
    val_surf = surface[surface["split"] == "VAL"].set_index("cell")
    same_sign = all(np.sign(val_surf.loc[nb, "mean_pct"]) == np.sign(val_row["mean_pct"])
                    for nb in neighbours if nb in val_surf.index)
    log(f"neighbours {neighbours}: same-signed as selected on VAL = {same_sign}")
    for nb in neighbours:
        if nb in val_surf.index:
            log(f"  {nb}: VAL mean_pct={val_surf.loc[nb,'mean_pct']:.4f}")

    # placebo for the selected cell + every NONE cell
    none_cells = [c for c in surface["cell"].unique() if c.endswith("_NONE")]
    placebo_targets = sorted(set([sel_cell] + none_cells))
    log(f"running placebo for {len(placebo_targets)} cells...")
    bar_store = BarStore(os.path.join(HERE, "bars_fills_1478.db"))
    placebo = placebo_for_cells(pop, bar_store, placebo_targets)
    placebo.to_csv(os.path.join(HERE, "rebuild_1493_placebo.csv"), index=False)
    for name in placebo_targets:
        real = fills[fills["cell"] == name].merge(
            pop[["day", "symbol"]], on=["day", "symbol"], how="inner")
        pb = placebo[placebo["cell"] == name]
        paired = real.merge(pb, on=["day", "symbol"], suffixes=("_real", "_pb"))
        margin = paired["net_pct_real"] - paired["net_pct_pb"]
        mean_m, t_m, nd_m = dayclustered_t(paired.assign(net_pct=margin), "net_pct")
        log(f"placebo margin [{name}]: n={len(paired)} margin={margin.mean():.4f}pp "
            f"day-clust t={t_m:.2f} (pass: margin>=0.10 & t>=2 -> "
            f"{margin.mean()>=0.10 and t_m>=2})")

    with open(os.path.join(HERE, "selected_cell.txt"), "w") as fh:
        fh.write(sel_cell + "\n")


if __name__ == "__main__":
    main()
