"""Compare the independent rebuild (events_1643_rebuild.csv) against the first build
(events_1643.csv). Run AFTER rebuild_1643.py has already written its own numbers.
"""
import numpy as np
import pandas as pd

mine = pd.read_csv("research/lev_flow/events_1643_rebuild.csv", parse_dates=["date"])
first = pd.read_csv("research/lev_flow/events_1643.csv", parse_dates=["date"])

print("mine rows:", len(mine), "first rows:", len(first))
print("first columns:", list(first.columns))
print("first split values:", first["split"].value_counts().to_dict())
print("first r_t abs min:", first["r_t"].abs().min())

for split in ["TRAIN", "VAL"]:
    m_keys = set(map(tuple, mine.loc[mine.split == split, ["date", "symbol"]].values))
    f_keys = set(map(tuple, first.loc[first.split == split, ["date", "symbol"]].values))
    inter = m_keys & f_keys
    union = m_keys | f_keys
    jac = len(inter) / len(union) if union else float("nan")
    print(f"{split}: mine={len(m_keys)} first={len(f_keys)} inter={len(inter)} union={len(union)} Jaccard={jac:.4f}")
    only_mine = m_keys - f_keys
    only_first = f_keys - m_keys
    print(f"  only-mine sample: {sorted(only_mine)[:5]}")
    print(f"  only-first sample: {sorted(only_first)[:5]}")

# mean bps difference per cell (on the intersection of events, matched by date+symbol)
first_idx = first.set_index(["date", "symbol"])
mine_idx = mine.set_index(["date", "symbol"])
for cell, book in [(1643, 1643), (1645, 1645)]:
    m = mine_idx[mine_idx.book == book]
    f = first_idx[first_idx.book == book]
    common = m.index.intersection(f.index)
    diff = m.loc[common, "net_bps"] - f.loc[common, "net_bps_close"]
    print(f"cell {cell}: n_common={len(common)} mean(mine)={m.loc[common,'net_bps'].mean():.3f} "
          f"mean(first)={f.loc[common,'net_bps_close'].mean():.3f} mean_diff={diff.mean():.4f} "
          f"max_abs_diff={diff.abs().max():.4f}")
    # split-level means (not just on common, on each build's own full split population)
    for split in ["TRAIN", "VAL"]:
        mv = mine[(mine.book == book) & (mine.split == split)]["net_bps"]
        fv = first[(first.book == book) & (first.split == split)]["net_bps_close"]
        print(f"  {split}: mine mean={mv.mean():.3f} (n={len(mv)}) | first mean={fv.mean():.3f} (n={len(fv)})")

# sign convention check on the FIRST build too
b43 = first[first.book == 1643]
sign_first = (np.sign(b43["exit_px_close"] - b43["entry_px_1531"]) == np.sign(b43["net_bps_close"] + 3)).mean()
raw_sign_first = (np.sign(b43["exit_px_close"] - b43["entry_px_1531"]) ==
                  np.sign(b43["exit_px_close"] / b43["entry_px_1531"] - 1.0)).mean()
print(f"first build 1643 raw sign check (exit>entry <=> raw_r>0): {raw_sign_first:.4f}")
updays_are_long = (first.loc[first.book == 1643, "direction"] > 0).mean()
print(f"first build: fraction of book==1643 rows with direction>0 (up day): {updays_are_long:.4f}")
