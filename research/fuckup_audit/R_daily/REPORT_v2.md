# Stage R_daily v2 — the same 20 pre-registered cells on a CORRECTED panel

`REPORT.md` reported 0/20 on a panel it had itself flagged as contaminated: 1,872 split candidates
detected and **never adjusted**, and t-stats of -7.9 to -25.2 on the family most exposed to overnight
corporate actions. The owner rejected the verdict. He was right to: the verdict was reported on a
contaminated panel, and one of its two headline sentences (the "one positive cell") was an artifact.
This is the same 20 cells, same PREREG, same splits, same gates, same tails, same permutation, on a
panel where the contamination is fixed rather than flagged. **Nothing was added, tuned or dropped.**

---

## 1. How much of the original result was contamination (measured before anything was changed)

`contam_step1.py` on the ORIGINAL per-trade files. Two overlap definitions, because `controls.py`
used only the first and it is the wrong one: **HOLD** = a flagged split date inside
`[entry_date, exit_date]` (43 trades — what the original control measured); **SIGNAL** = a flagged
split inside `[sig_date - the family's own feature window, exit_date]`, i.e. the window K4's 20-day
mean overnight, K2's 250-day high, K5's 50-day high/SMA and K3's 5-day return are built from. An
unadjusted split there fabricates the **selection**, not just the exit.

| family | booked trades (TRAIN+VAL) | net bps | contaminated | % of trades | share of total net | net bps ex-contaminated | t |
|---|---:|---:|---:|---:|---:|---:|---:|
| K1 | 830 | -195.4 | 24 | 2.9% | **32.2%** | -136.4 | -3.9 |
| K2 | 6,583 | -95.4 | 214 | 3.3% | **16.7%** | -82.2 | -6.9 |
| K3 | 50,253 | -28.9 | 629 | 1.3% | 2.7% | -28.5 | -1.4 |
| K4 | 71,946 | -79.2 | 1,946 | 2.7% | 2.9% | -79.0 | -35.1 |
| K5 | 18,693 | -60.5 | 519 | 2.8% | 4.3% | -59.6 | -9.3 |

**So the owner's hypothesis is half right.** K1 and K2 were materially contaminated (a third and a
sixth of their loss). K3/K4/K5 were not — and in particular **K4's t = -25 is not a split artifact**
(removing every split-overlap trade moves it to t = -35). The flag list was also incomplete: the
split scan in `seam_check.py` covered the **ITCH era only**, so no split after 2024-07 — half of VAL
and all of TEST — was ever flagged.

**The 20 most extreme single trades** (`contam_extremes.csv`) are a catalogue of corporate actions and
bad prints, not of trading: WMT 2024-02-21 -66% (3:1 forward split, ex-date 2024-02-26), ACMR
2022-03-23 -66% (3:1, ex-date 2022-03-24), MPWR 2022-10-14 **+7,178%** (entry $4.26 against a $334.95
prior close — an extended-session print, no split exists), BOLD entered 2020-01-07 and "exited"
2024-03-28 (-76%) because the hold was counted in row offsets and the ticker was reissued after a
delisting. The rest (KOSS, CODX, QUBT, MXL, VTYX, GERN, SBET, REPL, HTZ) are real events.

---

## 2. The fix

**Splits — a real source, not a deletion.** Source order from the task: (a) the bought Databento
`pit_definition/` files carry instrument definitions only, no split ratios, and no corporate-actions
schema is entitled on this account; so (b) **Alpaca's corporate-actions endpoint** (free, 2016 on).
`fetch_splits.py` -> `corporate_actions.csv`: **5,889 split-like events over 4,493 symbols,
2016-10-07 .. 2026-09-04** (5,092 reverse, 797 forward). `build_panel_r2.py` back-adjusts: for a row
dated *d*, `factor = product of ratio over events with ex_date > d`; prices / factor, volume x factor —
identically on both sides of the ITCH/EQUS seam. **680,338 of 3,566,236 panel rows (19.1%) were
rescaled.** Validation: of 3,335 raw overnight close ratios <= 0.55 or >= 1.80, **1,069 sat exactly on
a fetched ex-date; 25 survive the adjustment.** The task's fallback rule (c) was then run on the
residual as a check — 65 jumps are within 2% of an exact split ratio *and* have an inverse volume
jump — and **rejected**: they are real news (SRPT 2021-01-07 DMD failure, UPST 2022-05-09, APA
2020-03-09, LXRX, IAC's Match spin-off), not unadjusted splits, so no heuristic adjustment is
applied. `residual_split_candidates.csv` lists all 65.

**Bad-print opens.** After adjustment, an `open` more than 50% from the prior close whose own day's
close came back inside 20% of that prior close is an extended-session outlier, not an obtainable
market-on-open fill (CLAUDE.md 1b). **224 rows (0.006%)** blanked; the rest of the bar is kept, so no
rolling window loses a real day. `bad_print_opens.csv`.

**Session-faithful holds.** `K/build_k.py::simulate` walks row offsets inside a symbol block, so a
delisting-and-reissue turns a 10-session hold into a 1,543-day one (28 such trades, 115 entries more
than five calendar days after the signal). PREREG says "the next trading bar's OPEN" and "the close
of the hold's last bar"; `run_r2.py` implements exactly that in **session index**: 99 signals dropped
for a late entry bar, 2,100 holds truncated at the symbol's last available bar.

**The price gate on RAW prices (a look-ahead I introduced and then removed).** Back-adjustment
multiplies pre-event prices UP, and 5,092 of 5,889 events are *reverse* splits — so a $0.80 penny
stock that later reverse-split 1:20 would read as $16 in 2020 and sail through PREREG's `close >= $5`.
The panel therefore carries `close_raw` and the universe gate uses it: **8,970 primary-universe rows
dropped** that only passed on the inflated price. (The $10M dollar-volume gate needs no such care —
close x volume is invariant under the adjustment.)

**The ITCH open vs the true auction open**, measured where both tapes exist (2024-09, ITCH vs
EQUS.SUMMARY consolidated, 11,569 liquid Nasdaq-listed common symbol-days). Signed median **0.0 bps**,
mean -10.8, |median| 25.3 bps, but strongly **conditional on the K4 selection**:

| ITCH overnight-return decile | 1 (lowest) | ... | 9 | 10 (K4's bucket) |
|---|---:|---:|---:|---:|
| ITCH open - consolidated open, mean bps | **-64.8** | ~-10..0 | +11.0 | **+20.6** |

So on exactly the days K4 buys, the ITCH-era entry price is on average **21 bps above** the price the
consolidated auction offered — a fill-price bias that makes the ITCH era's K4 look ~20 bps worse than
it was. It is used as a correction to the interpretation, not to the data: the EQUS era already uses
the consolidated open, and no minute tape exists for 2018-2024 to repair the ITCH one.

---

## 3. Before / after — the 20 pre-registered cells

`net` = bps per trade, primary cost model, primary (point-in-time Nasdaq-listed) universe.
`before` = `REPORT.md`; `after` = `v2/cells_trainval.csv`. **No cell added, no cut moved.**

| cell | TRAIN before | t | MDE | **TRAIN after** | **t** | **MDE** | VAL before | **VAL after** | G1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `K1_h5_n10` | -428.0 | -1.75 | 490 | **-385.8** | -1.64 | 469 | -139.1 | **-139.1** | fail |
| `K1_h5_n20` | -428.0 | -1.75 | 490 | **-385.8** | -1.64 | 469 | -126.5 | **-126.5** | fail |
| `K1_h2_n10` | -394.8 | -2.13 | 370 | **-404.8** | -2.16 | 375 | -125.8 | **-125.8** | fail |
| `K1_h2_n20` | -394.8 | -2.13 | 370 | **-404.8** | -2.16 | 375 | -131.2 | **-131.2** | fail |
| `K2_h10_n10` | -101.8 | -2.49 | 82 | **-97.6** | -2.65 | 74 | -144.5 | **-104.8** | fail |
| `K2_h10_n20` | -91.0 | -2.81 | 65 | **-76.0** | -2.47 | 62 | -107.4 | **-73.9** | fail |
| `K2_h5_n10` | -94.9 | -3.20 | 59 | **-84.9** | -2.95 | 58 | -100.7 | **-73.2** | fail |
| `K2_h5_n20` | -80.4 | -2.96 | 54 | **-74.4** | -2.80 | 53 | -91.7 | **-74.7** | fail |
| `K3_h3_n10` | -40.7 | -3.01 | 27 | **-49.4** | -3.66 | 27 | -67.8 | **-65.0** | fail |
| `K3_h3_n20` | -38.4 | -4.12 | 19 | **-37.8** | -4.11 | 18 | -57.5 | **-60.8** | fail |
| `K3_h1_n10` | **+7.6** [1] | 0.11 | 139 | **-61.3** | **-13.9** | **8.8** | -67.9 | **-72.5** | fail |
| `K3_h1_n20` | -15.9 [1] | -0.36 | 89 | **-60.3** | **-17.8** | **6.8** | -72.5 | **-74.7** | fail |
| `K4_h1_n10` | -85.3 | -17.76 | 10 | **-84.1** | -17.5 | 9.6 | -82.4 | **-75.6** | fail |
| `K4_h1_n20` | -79.5 | -25.16 | 6.3 | **-79.5** | -25.2 | 6.3 | -69.3 | **-66.3** | fail |
| `K4_h2_n10` | -75.9 | -7.91 | 19 | **-71.9** | -7.7 | 19 | -104.7 | **-89.1** | fail |
| `K4_h2_n20` | -74.9 | -11.96 | 13 | **-75.0** | -12.0 | 13 | -79.5 | **-72.4** | fail |
| `K5_h5_n10` | -29.8 | -1.31 | 45 | **-28.4** | -1.25 | 45 | -76.3 | **-58.8** | fail |
| `K5_h5_n20` | -40.5 | -2.32 | 35 | **-36.7** | -2.10 | 35 | -91.6 | **-77.9** | fail |
| `K5_h2_n10` | -57.3 | -4.67 | 25 | **-54.8** | -4.46 | 25 | -103.6 | **-92.8** | fail |
| `K5_h2_n20` | -57.8 | -5.66 | 20 | **-56.0** | -5.47 | 20 | -96.6 | **-88.3** | fail |

[1] the MPWR bad print, now blanked at the panel. Full tables: `v2/before_after_cells.csv`,
`v2/cells_trainval.csv`, `v2/phaseA.md`, `v2/perm_p.csv`, `v2/controls_v2.csv`, `v2/trades/*.csv`.

**What the correction actually did.** (i) It **destroyed the one positive cell**: `K3_h1_n10` TRAIN
goes from +7.6 bps at t 0.11 with an MDE of 139 bps to **-61.3 bps at t -13.9 with an MDE of 8.8 bps**
— the original report's footnoted "one bad print" was the whole cell, and the corrected cell is now
one of the best-powered in the stage. (ii) It **helped K2 by up to +40 bps on VAL** (the WMT/ACMR-type
fake -66% exits are gone) — and K2 is still -73 to -105 bps. (iii) It left K4 essentially unchanged.
0 of 20 TRAIN cells and 0 of 20 VAL cells are positive. Permutation p (5,000 sign-flip draws across
the 20) = 1.00 on both splits. Residual corporate-action overlap in the corrected books: **19 trades
out of 148,703, moving no cell by more than 0.6 bps** (`v2/controls_v2.csv`).

**Tails and cost.** Removing the top 5% of trades makes every cell worse (-120 to -624 bps); winners
capped at the 95th percentile of winners, likewise. Four TRAIN cells have a positive **gross** mean
(max +24.7 bps) and at the most generous defensible fill (auction, 10 bps round trip) three turn
positive — `K3_h3_n20` +5.1, `K5_h5_n10` +14.7, `K5_h5_n20` +5.9 — and **all three are negative on
VAL** (-20.3, -19.0, -37.8). There is no gross edge for the cost model to be blamed for.

**Why K4 reads t = -25 and it is not contamination.** Split by tape: `K4_h1_n20` is **-80.1 bps
(t -26.4) on the ITCH era and -55.7 bps (t -6.1) on the EQUS consolidated era**. The ~24 bps
difference is the measured ITCH open bias of section 2 (+21 bps on exactly K4's decile). The sign
survives on the clean consolidated tape at t = -6 on 4,580 trades. K4 buys the open and sells the
close of the same day on names selected for high overnight returns; its gross mean is -25.6 bps
before any cost. That is the documented overnight/intraday decomposition, not a data defect, and
t = -25 is what 25,000 trades do to a -80 bps mean.

---

## 4. Verdict

**No edge was detectable for the five pre-registered multi-day long-only families — gap continuation,
52-week-high breakout on volume, short-term reversal, overnight continuation, uptrend pullback — in
the point-in-time Nasdaq-listed US common-stock universe with a 20-day median dollar volume >= $10M and
a RAW traded close >= $5, at 1-10 session holds, in a 10-20 position equal-$ book, over
2019-01-01 .. 2025-06-30, at half the liquidity-band spread + 5 bps per side on a market-on-open fill,
on a SPLIT-ADJUSTED panel; and the smallest per-trade effect the test could have seen was 6.3-30 bps
in 10 of the 20 cells and 35-469 bps in the other 10.** 0 of 20 cells clears G1, so G2 is unreachable,
**TEST (2025-07-01 .. 2026-09-04) was NOT read, no cell was frozen and `FREEZE.md` is not written.**
In the ten powered cells this is stronger than a null: the books are 37-84 bps per trade below
break-even at t = -4 to -25, on 3,400-25,000 booked trades, and negative again out of sample.

**The original verdict's direction survives; its evidence did not.** `REPORT.md` reached "0 of 20" with
a contaminated panel, a mis-specified split control, a family (K1) a third of whose loss was corporate
actions, and a footnoted positive cell that was a single bad print. That is luck, not method — the
conclusion happened to be robust to the defects that were flagged and ignored. Under the corrected
panel the same answer now rests on evidence that can be defended.

**This supersedes Stage K and Stage N2 on their overlapping years.** Both ran on unadjusted panels:
`research/lit_review_2026/daily_panel.parquet` (Stage K, 2025-01 .. 2026-09) shows the raw unadjusted
jump on **750 of 1,255** checkable split ex-dates, and N2's EQUS panel is the same vendor file used
here. Their windows sit inside this corrected panel's 2018-05 .. 2026-09, so for the K families on
Nasdaq-listed names the numbers above replace theirs. One check the coordinator asked for: **N2's
single VAL-positive cell (`K2_h10_n20`, +154 bps) is NOT manufactured by reverse splits** — exactly
1 of its 250 VAL trades touches any corporate action and removing it makes the cell *better* (+162);
its top ten are real 2026 moves (MXL, CAR, AEHR, SNDK). It dies on the pre-registered tail test
instead: **-164.7 bps with the top 5% removed.** A right-tail lottery ticket, not a contamination
artifact.

**Standing rules earned here.** (1) A flagged data defect is not a reported data defect — if a panel
is unadjusted, adjust it or do not draw the table. (2) A split control keyed to the hold window is
the wrong control; the feature window is where an unadjusted split fabricates the selection. (3)
Back-adjusting prices is itself a look-ahead for any PRICE gate; keep the raw close and gate on it.
(4) Counting a hold in row offsets is a bug on any panel with delistings; count sessions.

## 5. Files

`contam_step1.py` -> `contam_step1.csv`, `contam_step1_fam.csv`, `contam_extremes.csv` ·
`fetch_splits.py` -> `corporate_actions.csv` (5,889 events) · `build_panel_r2.py` ->
`daily_panel_2018_2026_adj.parquet` (gitignored), `bad_print_opens.csv`,
`residual_split_candidates.csv` · `run_r2.py` (drives the same unmodified `K/build_k.py` +
`K/report_k.py`, patching only the panel, the session-faithful hold and the raw-price gate) ->
`v2/phaseA.md`, `v2/cells_trainval.csv`, `v2/perm_p.csv`, `v2/signal_counts.csv`,
`v2/availability.md`, `v2/trades/*.csv`, `v2/before_after_cells.csv`, `v2/controls_v2.csv` ·
`v2_phaseA.log`. Data spent on this step: **$0.00** (Alpaca corporate actions are free; no Databento
byte was bought).
