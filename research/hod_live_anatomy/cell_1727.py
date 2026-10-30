"""Cell 1,727: double down on the 14:00 runner at power (PREREG_1727.md, frozen).

Population: cell 1,681's live-rule fills = research/hod_entry/1663_features.csv joined to 1676_features.csv on
(date, symbol, half) for fill_id (the same join 1681_hypotheses.load_population does; 5,506 fills, TRAIN-H2 + VAL,
2025-07-01..2026-05-29, TEST never present).  Bars: research/hod_entry/1681_paths.parquet = the read-only
bars_sip.db bars of every fill from its fill bar to the end of the day, cached once by cell 1,681 (m = ET minute of
the bar START, 840 = 14:00).  The 27 GB DB is therefore not touched by this cell.

Live-rule replay (bar-by-bar, same precedence as 1668 walk_k: EOD at m >= 955, then stop on low <= stop, then target
on high >= target; stop wins a tie).  The stored exit_type is reproduced for >= 99.7 % of fills (count logged).
EOD exit price = open of the 15:55 bar (matches the stored exit_price on 861 / 902 EOD exits).

Timing: the 14:00 state uses bars with m < 840 only (last bar 13:59, closed at 14:00:00).  The add buys at the
14:00 bar OPEN + 12 bps.  Outputs: RESULT_1727.md, addleg_1727.csv in this directory.
"""
import math
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HOD = os.path.join(os.path.dirname(HERE), "hod_entry")
M_1330, M_1400, M_EOD = 810.0, 840.0, 955.0
COST = 0.0012          # 12 bps on the add buy and on the add's non-target sells
LIVE_FILLS_PER_WEEK = 50.0
Z_MDE = 1.959964 + 0.841621


def log(msg):
    """Verbose progress line."""
    print(f"[1727] {msg}", flush=True)


def load_population():
    """1663 x 1676 join (as 1681_hypotheses.load_population) -> one row per fill with fill_id, split, levels."""
    p63 = pd.read_csv(os.path.join(HOD, "1663_features.csv"))
    p76 = pd.read_csv(os.path.join(HOD, "1676_features.csv"), usecols=["fill_id", "date", "symbol", "half"])
    pop = p76.merge(p63, on=["date", "symbol", "half"], how="inner")
    assert len(pop) == len(p63), "join did not reproduce 1663 1:1"
    pop["R_unit"] = pop["entry_price"] - pop["stop"]
    return pop.sort_values(["date", "symbol"]).reset_index(drop=True)


def replay_exit(b, stop, target):
    """Live-rule replay on path arrays b (dict m,o,h,l,c). Returns (exit_type, exit_index)."""
    m, h, l = b["m"], b["h"], b["l"]
    for j in range(1, len(m)):
        if m[j] >= M_EOD:
            return "eod", j
        if l[j] <= stop:
            return "stop", j
        if h[j] >= target:
            return "target", j
    return "eod", len(m) - 1


def eod_price(b):
    """Price of the 15:55 EOD exit: open of the first bar with m >= 955, else the last close (flagged)."""
    k = np.where(b["m"] >= M_EOD)[0]
    if len(k):
        return float(b["o"][k[0]]), True
    return float(b["c"][-1]), False


def analyse_fill(r, b):
    """Build the per-fill record: eligibility, 13:59 state, R1/R2, uncensored 14:00 path, B1 add leg."""
    et, ix = replay_exit(b, r.stop, r.target_price)
    out = dict(fill_id=r.fill_id, date=r.date, symbol=r.symbol, split=r.half, stored_exit=r.exit_type,
               replay_exit=et, base_R=r.net_R, eligible=False, R1=False, R2=False)
    if r.fill_min > M_1330:
        return out
    out["fill_ok"] = True
    if b["m"][ix] < M_1400:        # live-rule exit already happened before 14:00
        return out
    pre = np.where((b["m"] < M_1400))[0]
    post = np.where((b["m"] >= M_1400) & (b["m"] < M_EOD))[0]
    if len(pre) < 2 or len(post) == 0 or b["m"][post[0]] > M_1400 + 5:
        out["skip"] = "no 13:59 or 14:00 bars"
        return out
    out["eligible"] = True
    R = r.R_unit
    c13 = float(b["c"][pre[-1]])
    hi = float(b["h"][pre[1:]].max())   # post-entry high: bars after the fill bar, through 13:59
    open_r, dist, mfe = (c13 - r.entry_price) / R, (hi - c13) / R, (hi - r.entry_price) / R
    out.update(open_r=open_r, dist_high_r=dist, mfe_r=mfe)
    out["R1"] = bool(open_r >= 0.5 and dist <= 0.5)
    out["R2"] = bool(mfe >= 1.0 and open_r >= 0.5 * mfe)
    # uncensored 14:00 open -> 15:55 path
    p0 = float(b["o"][post[0]])
    pend, has_eod = eod_price(b)
    seg = post
    out.update(p0=p0, pend=pend, has_eod_bar=has_eod,
               path_bps=(pend / p0 - 1) * 1e4, path_R=(pend - p0) / R,
               mfe_after_R=(float(b["h"][seg].max()) - p0) / R, mae_after_R=(float(b["l"][seg].min()) - p0) / R)
    if out["R1"]:
        out.update(add_leg(r, b, post, c13, p0, pend))
    return out


def add_leg(r, b, post, c13, p0, pend):
    """B1: buy original size at 14:00 open + 12 bps; stop = 13:59 close - 1 R at stop - 12 bps (gap-aware);
    target of the original leg (bar high >= target) sells at the target; else 15:55 open - 12 bps.  Stop wins ties."""
    R = r.R_unit
    ent = p0 * (1 + COST)
    stop_a = c13 - R
    exit_px, how = None, "eod"
    for j in post:
        if b["l"][j] <= stop_a:
            exit_px, how = min(stop_a, float(b["o"][j])) * (1 - COST), "stop"
            break
        if b["h"][j] >= r.target_price:
            exit_px, how = float(r.target_price), "target"
            break
    if exit_px is None:
        exit_px = pend * (1 - COST)
    return dict(add_entry=ent, add_stop=stop_a, add_exit=exit_px, add_how=how, add_R=(exit_px - ent) / R)


def day_clustered(df, col):
    """(mean, day-clustered t, SE_days) of col; days are the clusters."""
    dm = df.groupby("date")[col].mean()
    n = len(dm)
    if n < 2 or dm.std(ddof=1) == 0:
        return float(df[col].mean()), float("nan"), float("nan")
    se = dm.std(ddof=1) / math.sqrt(n)
    return float(df[col].mean()), float(dm.mean() / se), float(se)


def ex_top5(v):
    """Mean after dropping the top ceil(5 %) values."""
    v = np.sort(np.asarray(v, float))
    k = int(math.ceil(len(v) * 0.05))
    return float(v[: len(v) - k].mean()) if len(v) > k else float(v.mean())


def alt_week_halves(df):
    """Label each row A/B by alternating ISO weeks (sorted unique weeks, even index -> A)."""
    iso = pd.to_datetime(df["date"]).dt.isocalendar()
    key = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    order = {k: i for i, k in enumerate(sorted(key.unique()))}
    return key.map(order).map(lambda i: "A" if i % 2 == 0 else "B")


def addleg_stats(d):
    """Add-leg stat row for a frame of adds."""
    if len(d) < 2:
        return dict(n=len(d), mean=np.nan, t=np.nan, se=np.nan, ex5=np.nan, capped=np.nan, hit_stop=np.nan, hit_tgt=np.nan)
    mean, t, se = day_clustered(d, "add_R")
    return dict(n=len(d), mean=mean, t=t, se=se, ex5=ex_top5(d["add_R"]), capped=float(d["add_R"].clip(upper=2.0).mean()),
                hit_stop=float((d.add_how == "stop").mean()), hit_tgt=float((d.add_how == "target").mean()))


def fmt(x, k=3, sign=True):
    """Format a float (nan-safe)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{k}f}" if sign else f"{x:.{k}f}"


def main():
    """Run the cell and write RESULT_1727.md + addleg_1727.csv."""
    import shutil
    free_gb = shutil.disk_usage(HERE).free / 1e9
    log(f"disk free {free_gb:.1f} GB")
    assert free_gb >= 5, "less than 5 GB free"
    t0 = time.time()
    pop = load_population()
    log(f"population {len(pop)} fills; splits {pop.half.value_counts().to_dict()}; dates {pop.date.min()}..{pop.date.max()}")
    pa = pd.read_parquet(os.path.join(HOD, "1681_paths.parquet"))
    groups = {k: v for k, v in pa.groupby("fill_id")}
    log(f"paths loaded: {len(pa)} rows, {len(groups)} fills in {time.time() - t0:.1f}s")
    rows = []
    for i, r in enumerate(pop.itertuples()):
        g = groups.get(r.fill_id)
        if g is None:
            log(f"WARNING fill {r.fill_id} {r.symbol} {r.date}: no path, skipped")
            continue
        b = {k: g[k].values for k in ("m", "o", "h", "l", "c")}
        rows.append(analyse_fill(r, b))
        if (i + 1) % 500 == 0:
            log(f"{i + 1}/{len(pop)} fills processed ({time.time() - t0:.1f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "addleg_1727.csv"), index=False)
    write_result(df, pop)
    log(f"done in {time.time() - t0:.1f}s")


def write_result(df, pop):
    """Compose RESULT_1727.md from the per-fill frame."""
    L = []
    splits = ["TRAIN-H2", "VAL"]
    nmis = int((df.stored_exit != df.replay_exit).sum())
    n_noeod = int((df.eligible & (~df.has_eod_bar.fillna(True))).sum())
    span = {s: max((pd.to_datetime(pop[pop.half == s].date).max() - pd.to_datetime(pop[pop.half == s].date).min()).days / 7, 1) for s in splits}
    fills_wk = {s: (pop.half == s).sum() / span[s] for s in splits}
    bt_fpw = len(pop) / (span["TRAIN-H2"] + span["VAL"])
    ratio = LIVE_FILLS_PER_WEEK / bt_fpw
    L.append("# RESULT cell 1,727 - double down on the 14:00 runner (PREREG_1727.md)\n")
    L.append(f"Population: cell 1,681 live-rule fills, n {len(pop)} (1663_features x 1676_features; per-fill file used = "
             f"`research/hod_entry/1663_features.csv` + `1681_paths.parquet` bars cached from bars_sip.db; the DB itself was not "
             f"re-queried). Dates {pop.date.min()}..{pop.date.max()}; TEST is absent from this population (nothing excluded, nothing sealed "
             f"was read). Full population, no subsample. Replay exit_type differs from stored on {nmis} / {len(df)} fills; "
             f"{n_noeod} eligible fills lack a 15:55 bar (last close used).\n")
    L.append("## 1. Eligible counts (fill <= 13:30 ET, live-rule position still open at the 14:00 bar)\n")
    L.append("| split | fills | fill<=13:30 | open at 14:00 | R1 | R2 | R1 share |\n|---|---|---|---|---|---|---|")
    for s in splits:
        d = df[df.split == s]
        e = d[d.eligible]
        L.append(f"| {s} | {len(d)} | {int(d.fill_ok.fillna(False).sum())} | {len(e)} | {int(e.R1.sum())} | {int(e.R2.sum())} | {e.R1.mean():.1%} |")
    L.append("\n## 2. Uncensored path 14:00 open -> 15:55 open (live exits ignored), per eligible position\n")
    L.append("| split | group | n | mean bps | median bps | mean R | median R | MFE after (R) | MAE after (R) | share up | day-clustered t (bps) |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for s in splits:
        for lab, mask in (("R1", lambda e: e.R1), ("non-R1", lambda e: ~e.R1), ("R2", lambda e: e.R2)):
            e = df[(df.split == s) & df.eligible]
            e = e[mask(e)]
            m, t, _ = day_clustered(e, "path_bps")
            L.append(f"| {s} | {lab} | {len(e)} | {m:+.1f} | {e.path_bps.median():+.1f} | {e.path_R.mean():+.3f} | {e.path_R.median():+.3f} | "
                     f"{e.mfe_after_R.mean():+.2f} | {e.mae_after_R.mean():+.2f} | {(e.path_bps > 0).mean():.1%} | {fmt(t, 2)} |")
    adds = df[df.R1.fillna(False) & df.add_R.notna()].copy()
    adds["half"] = alt_week_halves(adds)
    L.append("\n## 3. B1 add leg (R1 names, original share count, risk unit = original R per share)\n")
    L.append("| set | n | adds/wk (BT) | mean R | day-clust t | ex-top-5 % | winner-capped +2R | add stopped | add at target |\n|---|---|---|---|---|---|---|---|---|")
    stats = {}
    for lab, d, wk in (("TRAIN-H2", adds[adds.split == "TRAIN-H2"], span["TRAIN-H2"]), ("VAL", adds[adds.split == "VAL"], span["VAL"]),
                       ("TRAIN-H2 wk-A", adds[(adds.split == "TRAIN-H2") & (adds.half == "A")], span["TRAIN-H2"] / 2),
                       ("TRAIN-H2 wk-B", adds[(adds.split == "TRAIN-H2") & (adds.half == "B")], span["TRAIN-H2"] / 2),
                       ("ALL", adds, span["TRAIN-H2"] + span["VAL"])):
        s = addleg_stats(d)
        s["wk"] = len(d) / wk
        stats[lab] = s
        L.append(f"| {lab} | {s['n']} | {s['wk']:.2f} | {fmt(s['mean'])} | {fmt(s['t'], 2)} | {fmt(s['ex5'])} | {fmt(s['capped'])} | {fmt(s['hit_stop'], 2, False)} | {fmt(s['hit_tgt'], 2, False)} |")
    L.append("\n### Combined position vs the live rule (R1 positions only; original leg unchanged, add = 1x size)\n")
    L.append("| split | n R1 | live mean R | live + add mean R | delta per R1 position | delta per ALL fill in split (sum add_R / fills) |\n|---|---|---|---|---|---|")
    for s in splits:
        d = adds[adds.split == s]
        nfill = (df.split == s).sum()
        L.append(f"| {s} | {len(d)} | {fmt(d.base_R.mean())} | {fmt((d.base_R + d.add_R).mean())} | {fmt(d.add_R.mean())} | {fmt(d.add_R.sum() / nfill, 4)} |")
    L.append(f"\n## 4. Dollar translation (VAL rate and VAL mean; live/BT fill ratio = {LIVE_FILLS_PER_WEEK:.0f} / {bt_fpw:.1f} BT fills per week = {ratio:.3f})\n")
    sv = stats["VAL"]
    live_wk = sv["wk"] * ratio
    L.append(f"BT adds/week VAL {sv['wk']:.2f} -> live-scaled {live_wk:.2f} adds/week.\n")
    L.append("| risk per original position | add-leg $ per month (VAL mean R x live adds/wk x 52/12) | with VAL mean - 1 SE |\n|---|---|---|")
    for R_usd in (50, 150):
        mo = sv["mean"] * live_wk * 52 / 12 * R_usd
        lo = (sv["mean"] - sv["se"]) * live_wk * 52 / 12 * R_usd
        L.append(f"| ${R_usd} | {mo:+,.0f} | {lo:+,.0f} |")
    L.append("\n## 5. Frozen pass bar\n")
    tr, va = stats["TRAIN-H2"], stats["VAL"]
    items = [
        ("mean >= +0.10 R in TRAIN-H2", tr["mean"] >= 0.10, fmt(tr["mean"])),
        ("mean >= +0.10 R in VAL", va["mean"] >= 0.10, fmt(va["mean"])),
        ("day-clustered t >= 2 on VAL", va["t"] >= 2, fmt(va["t"], 2)),
        ("ex-top-5 % > 0 on VAL", va["ex5"] > 0, fmt(va["ex5"])),
        ("winner-capped (+2R) > 0 (TRAIN-H2 and VAL)", tr["capped"] > 0 and va["capped"] > 0, f"{fmt(tr['capped'])} / {fmt(va['capped'])}"),
        ("both TRAIN halves same sign", np.sign(stats["TRAIN-H2 wk-A"]["mean"]) == np.sign(stats["TRAIN-H2 wk-B"]["mean"]),
         f"{fmt(stats['TRAIN-H2 wk-A']['mean'])} / {fmt(stats['TRAIN-H2 wk-B']['mean'])}"),
        (">= 3 adds/week at the live fill rate", live_wk >= 3, f"{live_wk:.2f}"),
    ]
    for name, ok, val in items:
        L.append(f"* {'PASS' if ok else 'FAIL'}: {name} ({val})")
    allok = all(bool(i[1]) for i in items)
    L.append(f"\n**VERDICT: {'PASS' if allok else 'FAIL'}** ({sum(bool(i[1]) for i in items)} / {len(items)} items).\n")
    L.append("## 6. MDE\n")
    for lab in ("TRAIN-H2", "VAL"):
        L.append(f"* {lab}: add-leg day-clustered SE {fmt(stats[lab]['se'], 3, False)} R, MDE (alpha .05 two-sided, 80 % power) = {Z_MDE * stats[lab]['se']:.3f} R "
                 f"on n {stats[lab]['n']}; the +0.10 R bar is {'above' if 0.10 > Z_MDE * stats[lab]['se'] else 'below'} the MDE.")
    L.append("\n## 7. Caveats (own implementation)\n")
    L.append("* 1-minute bars: stop and target touches inside a bar are ordered stop-first (conservative for the add, and for the live replay); "
             "the add's stop fill is `min(stop, bar open) - 12 bps`, a target sell carries NO cost (limit sell at the printed target, as in the live base).")
    L.append("* Add buy fills at the 14:00 bar open + 12 bps: obtainable (the 14:00 open is known only at 14:00:00, so a real order would hit the first prints; "
             "12 bps is the PREREG's flat charge, not measured NBBO). EOD sells use the 15:55 bar open - 12 bps; stored live EOD uses the same open with 6 bps.")
    L.append("* Original-leg stop: the add stop (13:59 close - 1R) is always above the original stop (open P&L >= +0.5R), so the add stops first; "
             "the original leg is never altered, hence the combined delta is exactly the add leg.")
    L.append("* R1 is evaluated on one state per position (the PREREG definition); the 14:00 cohort is selected on the live book's own survival, "
             "so TRAIN halves by alternating weeks are not independent of the VAL window in the calendar sense but are disjoint in weeks.")
    L.append("* Adds/week live scaling assumes R1 frequency per fill is the same on live as in the BT (the live fills use the same population rule but BT fill rate is "
             "overstated vs live per cells 1,438/1,443).")
    L.append("* The population is the 1,681 full-day-range-free causal population; 1,681's own fills at 82 % fill rate; no change here. "
             "Single cell with one definition set: multiplicity = 1 new cell (1,727), no parameter varied.")
    with open(os.path.join(HERE, "RESULT_1727.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
