"""Independent rebuild of cell 1,621 (PREREG_1617 Frame C, W=3 RTH minutes).

Built from the PROSE of PREREG_1617.md (Frame C) and PREREG_1487.md only. Does NOT read
cell_1487.py or cell_1621.py / cell_1621_fills.csv / RESULT_1621.md -- this is the
independent-rebuild leg of the 1,621 check (fill-set agreement, R agreement within 0.01).

Rule (from PREREG_1617 Frame C, applying PREREG_1487's mechanics with W=3):
  For every base fill (causal_arming_causal.csv, status == 'fill') still OPEN at the end
  of minute fill_min + W (base exit_m > fill_min + W) with NO bar low <= level - $0.01 in
  the RTH minutes (fill_min, fill_min+W]: enter LONG at the ask at the open of minute
  fill_min+W+1, where ask = that bar's open + the fill's half_entry (features_1478_A.csv,
  disclosed proxy: the fill-instant half-spread, not the later entry's own spread).
  stop = level - $0.01; R'' = entry - stop; target = entry + 2*R''; 15:55 (minute 955) exit.
  Path walked on minute bars, stop-first on a bar touching both, gap-through at the open.
  Costs: entry half-spread already paid via the ask price; stop exit = SLIP_STOP_BPS[split]
  (cell_1478.py, blended filled/tail bps); target = limit (zero extra cost); EOD exit =
  {TRAIN: 11.5, VAL: 9.7} bps (task's disclosed constants, "EOD at the bid").
  R'' < 0.5% of price: reported but excluded from the primary book.

Minute convention: fill_min is a fractional ET minute-of-day (verified against
bars_fills_1478.db: fill_min=605.31 for AAP/2025-07-01 lands on the bar at ET minute 605).
m0 = floor(fill_min) is the fill bar's minute label. A bar with integer minute label m is
"in (fill_min, fill_min+W]" iff m0+1 <= m <= m0+W (window has exactly W bars). The entry
bar is minute m0+W+1.
"""
import sqlite3
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cell_1445 import day_clustered_t, ex_top5_mean  # noqa: E402  (explicitly allowed shared helper)

HERE = Path(__file__).parent
W = 3  # cell 1,621
EOD_M = 955  # 15:55 ET
NY = ZoneInfo("America/New_York")

# cell_1478.py SLIP_STOP_BPS = 0.88*filled + 0.12*tail, per split (read, not imported --
# cell_1478.py is an explicitly-allowed shared input, not part of the frame-C rebuild).
SLIP_STOP_BPS = {"TRAIN": 0.88 * 2.9 + 0.12 * 94.0, "VAL": 0.88 * 3.2 + 0.12 * 76.0}
EOD_BPS = {"TRAIN": 11.5, "VAL": 9.7}


def load_base():
    """Base fills + base outcome_R + half_entry, joined on (day, symbol, fill_min)."""
    base = pd.read_csv(HERE / "causal_arming_causal.csv", low_memory=False)
    base = base[base["status"] == "fill"].copy()
    pred = pd.read_csv(HERE / "model_1478_L3_predictions.csv")[["day", "symbol", "fill_min", "outcome_R"]]
    feat = pd.read_csv(HERE / "features_1478_A.csv")[["day", "symbol", "fill_min", "half_entry"]]
    base = base.merge(pred, on=["day", "symbol", "fill_min"], how="left")
    base = base.merge(feat, on=["day", "symbol", "fill_min"], how="left")
    assert base["outcome_R"].notna().all(), "missing base outcome_R after join"
    assert base["half_entry"].notna().all(), "missing half_entry after join"
    return base.reset_index(drop=True)


def load_bars_index():
    """{(symbol, day): DataFrame[m, o, h, l, c] sorted by m} with m = ET minute-of-day."""
    con = sqlite3.connect(HERE / "bars_fills_1478.db")
    df = pd.read_sql("SELECT symbol, day, t, o, h, l, c FROM bars", con)
    con.close()
    ts = pd.to_datetime(df["t"], utc=True).dt.tz_convert(NY)
    df["m"] = ts.dt.hour * 60 + ts.dt.minute
    idx = {}
    for (sym, day), g in df.groupby(["symbol", "day"], sort=False):
        idx[(sym, day)] = g.sort_values("m")[["m", "o", "h", "l", "c"]].reset_index(drop=True)
    return idx


def walk(path_after_entry, stop, target):
    """Same physics as sip_rebuild.walk_path (read for semantics only, not imported):
    stop-first on a bar touching both target and stop, gap-through at the open, EOD bar
    exits at its open. `path_after_entry` = bars with m >= entry_m, sorted, entry row first."""
    for i, row in enumerate(path_after_entry.itertuples()):
        if row.m >= EOD_M:
            return int(row.m), float(row.o), "eod"
        if row.l <= stop:
            px = row.o if row.o <= stop else stop
            return int(row.m), float(px), "stop"
        if row.h >= target:
            return int(row.m), float(target), "target"
    last = path_after_entry.iloc[-1]
    print(f"[rebuild_1621] WARNING: path for a fill ended before 15:55 (last m={int(last.m)}) "
          f"-- eod_fallback at its close", file=sys.stderr)
    return int(last.m), float(last.c), "eod_fallback"


def process_one(row, bars_idx):
    """Return a dict of rebuild outputs for one base fill, or an ineligibility reason."""
    sym, day, split = row.symbol, row.day, row.split
    fill_min, level, exit_m, why0 = row.fill_min, row.level, row.exit_m, row.why
    m0 = int(np.floor(fill_min))
    win_lo, win_hi = m0 + 1, m0 + W
    entry_m = m0 + W + 1

    out = dict(day=day, symbol=sym, split=split, fill_min=fill_min, level=level,
               base_why=why0, base_exit_m=exit_m, base_outcome_R=row.outcome_R,
               window_lo=win_lo, window_hi=win_hi, entry_m=entry_m)

    bars = bars_idx.get((sym, day))
    if bars is None:
        out.update(eligible=False, reason="no_bars_for_day")
        return out

    still_open = exit_m > fill_min + W
    win = bars[(bars["m"] >= win_lo) & (bars["m"] <= win_hi)]
    withdrew = bool((win["l"] <= (level - 0.01)).any())

    out["still_open"] = still_open
    out["withdrew_in_window"] = withdrew
    out["window_bar_count"] = len(win)

    if not still_open or withdrew:
        out.update(eligible=False, reason=("withdrew" if withdrew else "closed_in_window"))
        return out

    entry_bar = bars[bars["m"] == entry_m]
    if entry_bar.empty:
        out.update(eligible=False, reason="no_entry_bar")
        return out
    entry_bar = entry_bar.iloc[0]

    entry_px = float(entry_bar["o"]) + float(row.half_entry)
    stop_px = level - 0.01
    R2 = entry_px - stop_px
    if not (R2 > 0):
        out.update(eligible=False, reason="nonpositive_R")
        return out
    target_px = entry_px + 2 * R2

    path = bars[bars["m"] >= entry_m].reset_index(drop=True)
    exit_m2, exit_px, why2 = walk(path, stop_px, target_px)

    raw_R2 = (exit_px - entry_px) / R2
    if why2 == "stop":
        cost_R2 = SLIP_STOP_BPS[split] / 10000.0 * exit_px / R2
    elif why2 in ("eod", "eod_fallback"):
        cost_R2 = EOD_BPS[split] / 10000.0 * exit_px / R2
    else:  # target
        cost_R2 = 0.0
    net_R2 = raw_R2 - cost_R2
    R2_pct = R2 / entry_px * 100.0

    out.update(eligible=True, reason="ok", entry=entry_px, stop=stop_px, target=target_px,
               R=R2, R_pct_of_price=R2_pct, exit_m2=exit_m2, exit_price=exit_px, why=why2,
               raw_R=raw_R2, cost_R=cost_R2, net_R=net_R2,
               primary_book=bool(R2_pct >= 0.5))
    return out


def weeks_spanned(days):
    iso = pd.to_datetime(pd.Series(days).unique())
    return len({(d.isocalendar()[0], d.isocalendar()[1]) for d in iso})


def report(fills, base_all):
    rows = []
    for split in ["TRAIN", "VAL"]:
        f = fills[fills["split"] == split]
        elig = f[f["eligible"]]
        prim = elig[elig["primary_book"]]
        base_n = (base_all["split"] == split).sum()
        base_target_n = ((base_all["split"] == split) & (base_all["why"] == "target")).sum()
        runners_lost = f[(f["base_why"] == "target") & (f["base_exit_m"] <= f["fill_min"] + W)]
        wk = weeks_spanned(prim["day"]) if len(prim) else np.nan
        paired_diff = prim["net_R"] - prim["base_outcome_R"] if len(prim) else pd.Series(dtype=float)
        rows.append(dict(
            split=split, base_fills=base_n, eligible=len(elig), eligible_share=len(elig) / base_n,
            base_target_hitters=base_target_n, runners_lost=len(runners_lost),
            runners_lost_share_of_target=len(runners_lost) / base_target_n if base_target_n else np.nan,
            primary_book_n=len(prim),
            mean_net_R=prim["net_R"].mean() if len(prim) else np.nan,
            day_clustered_t=day_clustered_t(prim["net_R"], prim["day"]) if len(prim) else np.nan,
            ex_top5_mean=ex_top5_mean(prim["net_R"]) if len(prim) else np.nan,
            fills_per_week=len(prim) / wk if wk else np.nan,
            calibration_base_outcome_R_eligible=elig["base_outcome_R"].mean() if len(elig) else np.nan,
            base_outcome_R_primary=prim["base_outcome_R"].mean() if len(prim) else np.nan,
            paired_mean_diff=paired_diff.mean() if len(paired_diff) else np.nan,
            paired_t=day_clustered_t(paired_diff, prim["day"]) if len(paired_diff) else np.nan,
        ))
    return pd.DataFrame(rows)


def main():
    base = load_base()
    bars_idx = load_bars_index()
    print(f"[rebuild_1621] base fills: {len(base)}; unique (symbol,day) bar groups: {len(bars_idx)}")

    out_rows = [process_one(r, bars_idx) for r in base.itertuples(index=False)]
    fills = pd.DataFrame(out_rows)
    for col in ["eligible", "primary_book"]:
        if col in fills.columns:
            fills[col] = fills[col].fillna(False)

    cols = ["day", "symbol", "split", "fill_min", "level", "base_why", "base_exit_m",
            "base_outcome_R", "window_lo", "window_hi", "entry_m", "still_open",
            "withdrew_in_window", "window_bar_count", "eligible", "reason", "entry", "stop",
            "target", "R", "R_pct_of_price", "exit_m2", "exit_price", "why", "raw_R",
            "cost_R", "net_R", "primary_book"]
    for c in cols:
        if c not in fills.columns:
            fills[c] = np.nan
    fills = fills[cols]
    fills.to_csv(HERE / "rebuild_1621_fills.csv", index=False)

    rep = report(fills, base)
    print(rep.to_string(index=False))
    rep.to_csv(HERE / "rebuild_1621_report.csv", index=False)
    print("[rebuild_1621] wrote rebuild_1621_fills.csv, rebuild_1621_report.csv")


if __name__ == "__main__":
    main()
