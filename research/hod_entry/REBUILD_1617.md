# REBUILD 1,617 -- Frame A independent rebuild (held break overnight, MOC -> MOO)

Independent reimplementation from `research/hod_entry/PREREG_1617.md` prose only. Did NOT open `cell_1617.py`, `cell_1617_nights.csv`, or `RESULT_1617.md`. This document reports THIS rebuild's own numbers; it does not have the original 1,617 numbers to diff against (by design -- that comparison, nightly-set Jaccard >= 0.99 and bps within 1 on VAL per the prereg's own bar, happens in a separate step outside this task).

Full causal_arming_causal.csv rows (any status, all days): 33852.

## Coverage / availability rail
- Panel match on the 9,911 base fills: 9911/9911 matched for the price-scale check (0 had no panel close at all and were dropped from BOTH held/failed populations).
- Price-scale refuter (level <= panel day-high, shared raw scale): 0/9911 violations (PASS -- 0 violations, same raw scale).
- Held-break population (close >= level): 5083 matched nights before the raw-move flag; 10 excluded by the +/-30% band; 5073 in the primary book.
- Failed-break population (close < level), report-only: 4836 nights in the primary book after the same exclusions.
- Earnings-date exclusion coverage: 0% -- no earnings-calendar data source exists in this repo (searched for earnings_date/earnings_calendar sources and an Alpaca earnings endpoint; only `get_market_calendar`, the trading-day calendar, was found). NOT applied. Open refuter, see below.
- Halt-calendar coverage: 0% -- `research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv` is a static symbol-level snapshot (tradable/shortable/easy_to_borrow/exchange, no dates); it cannot answer "was this symbol halted on this specific night". NOT applied. Open refuter, see below.

## Held-break overnight (1,617) -- TRAIN-H2
- n nights: 2198 (+ 3 excluded by the raw-move band)
- mean net: 6.08 bps/night
- day-clustered t (the night): 0.25
- ex-top-5% mean: -50.06 bps; ex-top-1% mean: -12.18 bps
- winner-capped (+10%) mean: -3.48 bps
- green-night share: 0.450
- nights/week: 81.41 (over 27 weeks)
- months positive: 2 of 6
- per-month (mean bps, n):
  - 2025-07: 31.80 bps, n=263
  - 2025-08: -37.70 bps, n=336
  - 2025-09: -7.05 bps, n=314
  - 2025-10: 88.45 bps, n=505
  - 2025-11: -28.69 bps, n=466
  - 2025-12: -36.38 bps, n=314
- placebo margin (held-break minus in-play universe, same nights): -4.25 bps, t=-0.28, over 127 paired nights
- implied day-clustered SE: 24.77 bps; MDE at the pass bar's own t>=2.5 threshold (this n, this variance): a true mean of roughly +/-61.93 bps/night would be needed to clear t=2.5
- [1,618 context] failed-break (close < level) same split: n=2196, mean=40.22 bps, t=1.74

## Held-break overnight (1,617) -- VAL
- n nights: 2865 (+ 7 excluded by the raw-move band)
- mean net: -7.00 bps/night
- day-clustered t (the night): -0.25
- ex-top-5% mean: -63.28 bps; ex-top-1% mean: -25.25 bps
- winner-capped (+10%) mean: -16.17 bps
- green-night share: 0.472
- nights/week: 130.23 (over 22 weeks)
- months positive: 3 of 5
- per-month (mean bps, n):
  - 2026-01: 6.85 bps, n=439
  - 2026-02: -47.85 bps, n=552
  - 2026-03: -69.70 bps, n=518
  - 2026-04: 60.95 bps, n=551
  - 2026-05: 7.30 bps, n=805
- placebo margin (held-break minus in-play universe, same nights): -11.22 bps, t=-0.74, over 102 paired nights
- implied day-clustered SE: 27.86 bps; MDE at the pass bar's own t>=2.5 threshold (this n, this variance): a true mean of roughly +/-69.65 bps/night would be needed to clear t=2.5
- [1,618 context] failed-break (close < level) same split: n=2635, mean=-1.02 bps, t=-0.04

## Pass bar (frozen; evaluated on VAL per PREREG_1617.md)
- [FAIL] mean net >= +8 bps/night (VAL): -7.00
- [FAIL] day-clustered t >= 2.5 (VAL): -0.25
- [FAIL] ex-top-5% > 0 (VAL): -63.28
- [FAIL] winner-capped positive (VAL): -16.17
- [PASS] >= 3 nights/week (VAL): 130.23
- [FAIL] placebo margin >= +5bps, t>=2 (VAL): -11.22 bps, t=-0.74
- [FAIL] >= 4 of 6 months positive (VAL has 5 months available): 3 of 5 VAL months
- [FAIL] TRAIN-H2 same sign, t >= 1: mean=6.08 bps, t=0.25

**Verdict: FAIL (1/8 criteria clear)**

Note on "4 of 6 months": VAL (2026-01..2026-05) only has 5 calendar months of data in this population, not 6 -- the frozen bar text says "6" (likely written against TRAIN-H2's 6-month span, Jul-Dec 2025). Evaluated here as "4 of the 5 available VAL months" and flagged as a prereg wording ambiguity rather than silently rewritten.

## Refuters (PREREG_1617.md "Independent check and consequences")
- **Price scale (raw close/open):** 0/9911 fills have `level` above the panel's own day-high -- clean. `level`/`fill`/`stop` come from the live/tape-sourced causal_arming_causal.csv and are always <= the panel `high` for the same (symbol, day), consistent with the panel being on the same raw (unadjusted-for-that-day) price scale as the intraday feed.
- **Earnings and halts:** NOT RESOLVED -- no data source for either exists in this repo (see Coverage above). The 15 nights excluded by the +/-30% raw-move band catch the most extreme cases (including some halts/splits/earnings gaps by construction, since those are exactly the mechanisms that produce >30% overnight moves), but ordinary-sized earnings gaps (a few percent) are NOT filtered and remain in the primary book. This means the mean-net-bps numbers above are not certified clean of earnings-night contamination -- flag before shipping.
- **The placebo:** reported per holdout above (held-break minus the in-play universe, same nights). A positive, significant margin says the edge is about the break holding, not just about the name being active that night; see the pass-bar line for the VAL verdict.
- **Tails:** ex-top-5%/ex-top-1% and winner-capped-at-+10% reported per holdout above alongside the raw mean -- read them against the raw mean before trusting the headline number.

## Design choices made rebuilding from prose (documented for the comparison step)
- Auction cost applied multiplicatively on each leg (buy at close*(1+5bps), sell at next_open*(1-5bps)), not simply raw_bps - 10; the two are within ~0.005 bps of each other.
- "Night" = one (day, symbol) held-break fill row (matches the base-fill population's own grain); day-clustered t clusters by the break-day calendar date, consistent with every other `day_clustered_t` use in this codebase.
- Placebo margin built as a PAIRED per-calendar-night difference (held-break mean bps that night minus universe mean bps that night), then day-clustered across nights -- the most literal reading of "held-break minus the universe on the same nights". The universe leg includes the held-break names themselves (it is "the whole in-play scanner day universe", not the universe minus the break names).
- "In-play scanner day universe" (for the placebo) = every (day, symbol) row in causal_arming_causal.csv for that day, ANY status (fill/nofill/not_armed) -- this is the full population causal_arming.py's scanner evaluated that day (its own docstring: "population = EVERY symbol-day of the spec's causal superset... plus the live admission gates"); no separate broader universe file was in the task's shared inputs.
- Zero-OHLCV rows dropped from the panel BEFORE any join (prereg population definition), so a symbol-day with a zero print anywhere in OHLCV is treated as "no panel data" for that day, not as a valid close/next_open pair.

## Output schema: `rebuild_1617_nights.csv`
One row per held-break night (primary book, i.e. `split_flag == False` rows are the reportable set; flagged rows are KEPT in the CSV with `split_flag=True` for transparency, not deleted).
Columns: `day, symbol, split, level, fill, stop, bar_date, open, high, low, close, next_open, raw_ret, net_ret, net_bps, month, split_flag, earnings_checkable`.
Key fields for the trade-by-trade comparison: `(day, symbol)` is the night key; `net_bps` is the reportable net return; `split` is TRAIN/VAL.

