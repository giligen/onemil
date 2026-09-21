# Independent rebuild spec — cell 1,328 "union of pools" (do NOT read any other script in this folder)

You are rebuilding a research result from this prose only. Do not open `score_*.py`, `build_*.py`,
`REPORT_*.md` or `PREREG_*.md` in `research/orb_seed_wide/`. Budget: at most 15 tool calls. Write your result to
`research/orb_seed_wide/INDEPENDENT_1328.md` and reply with at most 100 words.

## Inputs (all under research/orb_seed_wide/out/, read with `pd.read_csv(path, keep_default_na=False, na_values=[""])`
because the ticker "NA" is a real symbol)
* `orb_features_20260920_2142.csv` — every candidate row of the wide universe, columns include `symbol`, `date`,
  `entry_price`, `gap_pct`.
* `runCOMB_features.csv` — the candidate rows that were fed to the combined walk.
* `runB_true.csv` — the walked PRODUCTION book (its rows are the picks; `entered` 1 = filled, `_sized_pnl` = dollar
  P&L at $375 risk, `date`, `symbol`, `_composite`).
* `runCOMB_true.csv` — the walked COMBINED book, same columns.

## Task A — verify the combined seed definition
Rebuild from the wide CSV: rows with `entry_price >= 3` and ((`entry_price <= 30` and `gap_pct >= 4`) or
(`30 < entry_price <= 50` and `3 <= gap_pct < 5`)). Compare the set of (symbol, date) with `runCOMB_features.csv`:
report counts of both sets and of rows only in one of them.

## Task B — build the UNION book and score it
1. Production picks = all rows of `runB_true.csv`.
2. Add-on picks = rows of `runCOMB_true.csv` whose (date, symbol) is NOT in the production picks, ordered within
   each date by `_composite` descending; per date admit at most `8 - (number of production picks that date)`.
3. Union book = production picks + admitted add-ons. Use each row's own `_sized_pnl`.
4. For TRAIN (dates in 2025) and VAL (dates 2026-01-01..2026-05-31), on `entered == 1` rows only, report for the
   production book and the union book: n, mean R (`_sized_pnl / 375`), total $, fills per week (n divided by the
   number of distinct ISO weeks that have at least one production fill in that split), and the weekly max drawdown
   in R (cumulative sum of weekly P&L in R, minus its running maximum, minimum). Also the added rows alone: n, mean R,
   total $, and mean R after removing the top 5 % of added rows by R (at least one row removed).
5. Assert no row has a date on or after 2026-06-01.

## Output
A markdown table with the numbers above per split, the Task-A counts, and one line saying whether the union book
raised total $ on both splits and by how much. No interpretation.
