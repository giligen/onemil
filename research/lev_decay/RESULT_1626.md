# RESULT — cells 1,626-1,629: leveraged single-stock ETF decay pair

Generated 2026-09-28T18:11:46+00:00 by research/lev_decay/cell_1626.py. PREREG: research/lev_decay/PREREG_1626.md (FROZEN 2026-09-28 18:30 UTC). This is the BUILDER run: numbers below are **NOT yet independently reimplemented** (PREREG "Independent check" step 1 is a separate, not-yet-run task) -- do not relay the headline to the owner until that rebuild, the obtainability check and the tail check (steps 1-5 of the CLAUDE.md research-claim gate) have run.

## Pair matching
34 pairs matched from `/home/ec2-user/onemil/data/research/databento/alpaca_assets_all_20260905.csv`: 34 at 2x, 0 at 3x, 0 at 1.5x/1.75x. Parser: `parse_name()` in this file -- leverage factor from a `\d+(\.\d+)?X` regex on the asset name; direction from LONG/BULL/UP vs SHORT/BEAR/INVERSE/DOWN/REVERSE tokens; underlying resolved (1) via a company-name alias table [T-Rex spells out names like "NVIDIA"], (2) via direct match against the full Alpaca symbol universe, (3) weak fallback = first remaining candidate token (flagged `weak_unconfirmed` in `leveraged_names_parsed.csv` for manual review). Full parsed set: `leveraged_names_parsed.csv`; matched pairs: `pairs.csv`; alternate-issuer legs not used (e.g. TSLA has both TSLL/TSLR as 2x-long and TSDD/TSLQ as 2x-short -- one of each picked deterministically, tradable+active first, then alphabetical): `pairs_alt_legs_skipped.txt`.

**Cell 1,629 is EMPTY by finding, not by omission.** Every 3x/1.5x/1.75x name that matched the leverage-factor regex tracks a sector, country or broad-index basket (e.g. TNA/TZA=Russell small-cap, EDC/EDZ=MSCI EM, DRN/DRV=real estate, TMF/TMV=20yr Treasury, HIBL/HIBS=S&P500 high-beta, YINN/YANG=FTSE China, DPST/WDRW=regional banks, GASL/GASX=natural gas) -- NONE are single-stock. In this Alpaca asset snapshot, single-stock leveraged products are essentially all 2x (matches the PREREG examples: TSLR/TSDD, AMZU/AMZD, AVGG, BABX). The 17 excluded underlyings, and the one confirmed FALSE PAIR the hand-check caught (TBXU "Biotech Top 5 Bull 2X" wrongly paired with TSXD "Semiconductors Top 5 Bear 2X" on the shared word "TOP" -- two different baskets, not a real pair), are in `pairs_excluded_non_single_stock.csv` with the blocklist and rationale in `NON_SINGLE_STOCK_BLOCKLIST` at the top of this file.

Survivorship: 0/34 pairs have at least one leg NOT status=active in the (current, 2026-09-05) asset-list snapshot -- these are closed/delisted legs, kept in the population and dated by their available bars, per PREREG ("count the ones that closed").

### Hand-check of 20 pairs
| pair | underlying | factor | long | short | long_conf | short_conf |
|---|---|---|---|---|---|---|
| AAOI_2.0x | AAOI | 2.0 | AAOG | AAOZ | known_symbol | known_symbol |
| AI_2.0x | AI | 2.0 | AIBU | AIBD | known_symbol | known_symbol |
| APLD_2.0x | APLD | 2.0 | APLX | APLZ | known_symbol | known_symbol |
| BE_2.0x | BE | 2.0 | BEG | BEZ | known_symbol | known_symbol |
| BMNR_2.0x | BMNR | 2.0 | BMNG | BMNZ | known_symbol | known_symbol |
| CBRS_2.0x | CBRS | 2.0 | CBRG | CBRZ | known_symbol | known_symbol |
| CRWV_2.0x | CRWV | 2.0 | CRWG | CORD | known_symbol | known_symbol |
| IONQ_2.0x | IONQ | 2.0 | IONL | IONZ | known_symbol | known_symbol |
| IREN_2.0x | IREN | 2.0 | IRE | IREZ | known_symbol | known_symbol |
| META_2.0x | META | 2.0 | FBL | METQ | known_symbol | known_symbol |
| MSTR_2.0x | MSTR | 2.0 | MSTP | MSTZ | known_symbol | known_symbol |
| MU_2.0x | MU | 2.0 | MUG | MUZ | known_symbol | known_symbol |
| NVDA_2.0x | NVDA | 2.0 | NVDG | NVD | known_symbol | known_symbol |
| PLTR_2.0x | PLTR | 2.0 | PLTG | PLTZ | known_symbol | known_symbol |
| QBTS_2.0x | QBTS | 2.0 | QBTX | QBTZ | known_symbol | known_symbol |
| RKLB_2.0x | RKLB | 2.0 | RKLX | RKLZ | known_symbol | known_symbol |
| SMR_2.0x | SMR | 2.0 | SMU | SMZ | known_symbol | known_symbol |
| SNDK_2.0x | SNDK | 2.0 | SNDG | SNDQ | known_symbol | known_symbol |
| SPCX_2.0x | SPCX | 2.0 | SPAX | SPCQ | known_symbol | known_symbol |
| TSM_2.0x | TSM | 2.0 | TSMG | STSM | known_symbol | known_symbol |

Hand-checked against the source `name` strings in `leveraged_names_parsed.csv`; any row not visibly a correct (underlying, factor, direction) triple is a parser bug, not a data error.

## First bar date per pair (listed-date coverage)
Earliest: 2024-05-15 (AI_2.0x). Latest: 2026-08-19 (MU_2.0x). Median: 2026-01-22. 30/34 pairs first-traded after 2025-03-31 (TRAIN-absent or TRAIN-partial, mostly VAL).

## Cell 1,626 / 1,627 / 1,629 — per (cell, split, borrow rail)

| cell | split | rail | n pair-days | n pairs | mean bps/day | t (day-clustered) | share pairs+ | worst month % | worst day % | max DD % | pairs ETB-both | drag realised/theory (top-vol tercile) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1626 | TRAIN | 5pct | 371 | 4 | 1.20 | 0.28 | 0.75 | -5.86 | -1.74 | 4.86 | 0 | 0.20 |
| 1626 | TRAIN | 15pct | 371 | 4 | -2.72 | -0.63 | 0.50 | -6.69 | -1.78 | 10.73 | 0 | 0.20 |
| 1626 | TRAIN | 30pct | 371 | 4 | -8.60 | -1.98 | 0.25 | -7.94 | -1.84 | 23.35 | 0 | 0.20 |
| 1626 | VAL | 5pct | 6153 | 34 | 9.13 | 1.61 | 0.82 | -41.85 | -4.06 | 10.89 | 0 | 0.38 |
| 1626 | VAL | 15pct | 6153 | 34 | 5.19 | 0.91 | 0.76 | -42.53 | -4.10 | 15.66 | 0 | 0.38 |
| 1626 | VAL | 30pct | 6153 | 34 | -0.73 | -0.13 | 0.50 | -43.56 | -4.16 | 28.78 | 0 | 0.38 |
| 1627 | TRAIN | 5pct | 234 | 4 | 6.84 | 1.22 | 0.50 | -1.98 | -0.97 | 2.60 | 0 | 0.42 |
| 1627 | TRAIN | 15pct | 234 | 4 | 2.96 | 0.53 | 0.50 | -2.74 | -1.01 | 4.02 | 0 | 0.42 |
| 1627 | TRAIN | 30pct | 234 | 4 | -2.86 | -0.51 | 0.50 | -3.87 | -1.07 | 9.15 | 0 | 0.42 |
| 1627 | VAL | 5pct | 4885 | 33 | 8.83 | 1.33 | 0.70 | -41.85 | -3.85 | 14.00 | 0 | 0.31 |
| 1627 | VAL | 15pct | 4885 | 33 | 4.90 | 0.74 | 0.67 | -42.53 | -3.89 | 17.85 | 0 | 0.31 |
| 1627 | VAL | 30pct | 4885 | 33 | -0.99 | -0.15 | 0.45 | -43.56 | -3.94 | 33.31 | 0 | 0.31 |

## Realised-vol tercile table (cell 1,626, VAL, gross -- the mechanism check)
Low/mid/high terciles of the underlying 20-session realised vol across all VAL pair-days; drag must RISE with sigma^2 if the mechanism (not a fluke) is driving the P&L.

| tercile | n pair-days | mean vol (ann) | mean realised bps/day (gross) | mean theory bps/day (1/2*(L^2-L)*sigma^2) | realised / theory |
|---|---|---|---|---|---|
| low  | 2039 | 0.454 | 5.44  | 8.54  | 0.64 |
| mid  | 2039 | 0.772 | 3.80  | 23.95 | 0.16 |
| high | 2039 | 1.171 | 24.50 | 56.06 | 0.44 |

**Not monotonic**: low-to-mid realised P&L FALLS (5.44 -> 3.80 bps/day) while theory more than doubles (8.5 -> 23.9) -- the calibration check fails in the middle tercile, only partially recovers in the high tercile, and realised/theory stays well below the 0.7-1.3 pass-bar band (0.16-0.64) in every tercile. This is a real, negative finding about the mechanism at a 5-session rebalance on this population, not a coding artefact -- full row-level source: cell_1626_days.csv, rebuild query in vol_tercile_table_1626_VAL.csv.

## Pass-bar checklist (frozen, VAL, per cell)
Mean >= +4 bps/day net @ 15%/yr rail; day-clustered t >= 2.5; >= 60% of pairs positive; TRAIN same sign t >= 1; realised drag within 30% of theory in the top-vol tercile; worst month >= -3%; >= 10 pairs ETB-both at the snapshot.

**Cell 1626** (2/7 pass):
- [x] mean >= +4 bps/day (VAL, 15%)
- [ ] t >= 2.5 (VAL, 15%)
- [x] >=60% pairs positive (VAL)
- [ ] TRAIN same sign, t>=1
- [ ] drag within 30% of theory (top-vol)
- [ ] worst month >= -3%
- [ ] >=10 pairs ETB-both

**Cell 1627** (2/7 pass):
- [x] mean >= +4 bps/day (VAL, 15%)
- [ ] t >= 2.5 (VAL, 15%)
- [x] >=60% pairs positive (VAL)
- [ ] TRAIN same sign, t>=1
- [ ] drag within 30% of theory (top-vol)
- [ ] worst month >= -3%
- [ ] >=10 pairs ETB-both

## Cell 1,628 — single-leg hedge (report-only, no pass bar)
- TRAIN 5pct: n=1130 mean=2.43 bps/day (of $3 gross) t=0.74
- TRAIN 15pct: n=1130 mean=1.11 bps/day (of $3 gross) t=0.34
- TRAIN 30pct: n=1130 mean=-0.87 bps/day (of $3 gross) t=-0.26
- VAL 5pct: n=8060 mean=2.00 bps/day (of $3 gross) t=1.82
- VAL 15pct: n=8060 mean=0.67 bps/day (of $3 gross) t=0.61
- VAL 30pct: n=8060 mean=-1.31 bps/day (of $3 gross) t=-1.20

## Costs, borrow, conventions (exact, for the independent rebuild)
- Bars: Alpaca daily, `adjustment=split` (NOT raw -- these products carry frequent reverse splits; raw closes would fabricate +/-90% jump days).
- Gross notional basis: $2 fixed (1,626/1,627/1,629) or $3 fixed (1,628), i.e. bps are relative to the STATIC target notional at the last rebalance, not a daily mark-to-market renormalisation -- this is exactly how the position drifts between rebalances (reported as `notional_gap`).
- Rebalance: every 5 AVAILABLE trading sessions (not calendar days) since the last rebalance/open; cost = 5.0 bps * dollar size of the rebalancing trade, per leg (the opening trade is a full $1/leg; later rebalances trade only the drift back to $1).
- Borrow: notional-based, `(D_long_prev + D_short_prev)/2 * rail/252`, charged daily on both short legs at the 5%/15%/30% annual rails; NOT a real historical rate (Alpaca has none) -- a sensitivity rail, per PREREG.
- ETB flags are TODAY's snapshot (2026-09-28) applied to the whole history -- disclosed hindsight, not a historical borrowability series.
- 1,627 gate: uses the UNDERLYING'S 20-session realised vol (log-return std, annualised by sqrt(252)); ON at >=60%, OFF below 40%; a gate transition forces close (cost) then reopen (cost) rather than netting.
- Theory drag per pair-day: (L^2-L)/2 * sigma_daily^2 * 10000 bps, using that day's rolling 20-session realised vol as the sigma estimate (a proxy for the true path variance, not a perfect match).

## Caveats (read as an adversary, per CLAUDE.md)
- **No independent reimplementation yet.** PREREG requires an agent that has not read this code to rebuild from prose and match trade-by-trade before this goes in front of the owner. Not done in this BUILDER task.
- **Obtainability not separately verified.** P&L uses CLOSE-to-close leg returns with no explicit fill-price/spread model beyond the flat 5 bps/leg cost; a close print is not always a restable order for an illiquid single-stock leveraged ETF -- the 5 bps cost is a placeholder, not a measured NBBO cost (CLAUDE.md #4 wants measured per-trade NBBO, not a band; not done here).
- **"Weak_unconfirmed" underlying matches** should be treated as unverified until hand-checked (see `leveraged_names_parsed.csv` `match_confidence` column) -- the 20-pair sample above covers only part of the matched set.
- **Multi-issuer underlyings**: only ONE long/short pair kept per (underlying, factor); skipped alternates in `pairs_alt_legs_skipped.txt` are a second, correlated pair on the same name -- not counted toward `n_pairs`, by design (avoids pseudo-replication), but means true product coverage is wider than `n_pairs` suggests.
- **Theory-vs-realised drag ratio** uses the rolling 20-session vol as the sigma proxy for that single day's theoretical drag, not the realised path variance since the last rebalance -- a coarse calibration check, not exact.
- **Trend-path tail** (the mechanism's stated failure mode) is only visible via worst-month/worst-day/max-DD above; no separate decomposition of trend vs chop regimes was run.
- **Borrow cost is a rail, not a rate**: real borrow on illiquid single-stock 2x products can spike far above 30%/yr or be recalled outright; that scenario is not modelled beyond the static-rail sensitivity.


## Judge (main session, 2026-09-28 18:45 UTC) — FAIL; the drag is real in gross, the trade is not executable here and its tail is catastrophic
* Mechanism: gross of costs the pair short earns +9 bps/day of gross notional on VAL (34 pairs, 82 % positive; the top
  realised-vol tercile +24.5 bps/day, ≈ 0.4 of the theoretical drag — the 5-session rebalance and the funds' own daily
  rebalancing leave delta drift; the middle tercile breaks the σ² monotonicity). Net: +5 bps/day at 15 %/yr borrow
  (t 0.9), −1 at 30 %. Rebuild (93 pairs vs 34 — different matching; Jaccard 0.86 on the common set) +7 bps/day
  (t 1.3). Nowhere near t 2.5.
* Executability: 0 of 34 pairs (67 of 68 legs) are shortable at Alpaca — the whole single-stock leveraged category is
  not shortable on this account; the PREREG required ≥ 10 pairs easy-to-borrow. Closed on this alone.
* Tail: worst pair-month −42 % of gross notional, portfolio worst month −6 % (the trend path: an underlying that runs
  without pullbacks crushes the short 2x-long leg); TRAIN is 4 pairs (the category is a 2025 phenomenon).
* Refuter defects (all lowering the number): AIBU/AIBD is a basket not a single stock (59 % of TRAIN pair-days), CONI
  tracked −1x until 2025Q1, one "underlying" is itself a 2x ETF; leverage from today's fund names is hindsight.
Consequence: closed on this account. Programme count 1,629. The drag exists; capturing it needs a prime broker with
borrow on leveraged ETFs and daily rebalancing — not a retail Alpaca book.
