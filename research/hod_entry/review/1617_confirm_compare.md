# Frame C (cell 1,621) — builder vs. independent rebuild compare

Spec: `research/hod_entry/PREREG_1617.md`, frame C (3/5-minute confirmation entries), W=3 sub-cell (cell 1,621).

Builder: `cell_1621.py` -> `cell_1621_fills.csv` (also carries the W=5 sibling, cell 1,622, stacked in the
same file) -> `RESULT_1621.md`.
Rebuild: `rebuild_1621.py` -> `rebuild_1621_fills.csv` + `rebuild_1621_report.csv` -> `REBUILD_1621.md`, built
independently from the prose spec (not from `cell_1621.py`).

## Method
- Restricted the builder CSV to `cell == 1621` (the W=3 sub-cell; W=5 lives in the same file under `cell == 1622`
  and is out of scope for this compare).
- Joined builder and rebuild on `(day, symbol)` — no duplicate keys in either file, so this key is a safe join
  (confirmed: 0 duplicated (day,symbol) pairs in each).
- Base population (9,911 rows each): set Jaccard on (day,symbol) = **1.000** (9,911/9,911 both sides).
- Entered/eligible population (builder `entered==True`, rebuild `eligible==True`, 2,208 rows each): set Jaccard =
  **1.000**.
- Passing ("primary") population (builder `entered==True & below_floor==False`, rebuild `eligible==True &
  primary_book==True`, the R''-as-%-of-price >= 0.5% floor from cell 1,487): 1,654 rows on both sides, set Jaccard
  = **1.000** — the two independent implementations select the exact same trades into the reported book.
- Row-level R compare on the 2,208 matched entered/eligible rows: builder `net_R2` vs. rebuild `net_R`.

## Results
| metric | value |
|---|---|
| set Jaccard (base population, 9,911 rows) | 1.000 |
| set Jaccard (entered/eligible population, 2,208 rows) | 1.000 |
| set Jaccard (primary/passing population, 1,654 rows) | 1.000 |
| share of matched rows within 0.01 R | **1.000** (2,208 / 2,208) |
| max abs(net_R2 − net_R) | 3.2e-13 R |
| mean abs diff | 6.2e-16 R |
| passing cells match? | **Yes** — identical 1,654-row primary set on both sides (builder: below_floor filter; rebuild: primary_book flag) |

Bar (>= 99% of rows within 0.01 R): **PASS**, at 100.0%.

## Dominant cause of the largest differences
The 10 largest |diff| values are all on the order of 1e-13 to 1e-15 R — floating-point representation noise from
two independent codebases doing the same chain of subtractions/divisions in a different order (e.g. price-diff /
risk vs. (exit-entry)/(entry-stop) computed with slightly different intermediate roundings), not a real
methodological or data disagreement. There is no cluster of differences above ~1e-13 R, so there is no secondary
cause to report — builder and rebuild agree to machine precision on every one of the 2,208 matched fills, and
agree exactly (set Jaccard 1.0) on which 1,654 of those are in the reported "primary" book.

## Read
Frame C / cell 1,621 clears the independent-rebuild bar cleanly: same base population, same entered population,
same primary (passing) population, same R per fill to float precision. This does **not** change cell 1,621's own
verdict (RESULT_1621.md / REBUILD_1621.md both report the confirmation-entry mean net R as negative on TRAIN-H2
and VAL — see the reports for that number); this compare only certifies that the two independent builds computed
the same thing, so that negative number can be trusted as a coding-error-free result of the frame C mechanism.
