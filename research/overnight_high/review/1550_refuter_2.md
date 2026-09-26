# Cell 1,550/1,551 — refuter 2 (statistics, execution, multiplicity)

Scripts (read-only on inputs): `refuter2_recompute.py`, `refuter2_corrected.py`, `refuter2_dups_1545.py`.
Outputs: `refuter2_stats.csv`, `refuter2_placebo.csv`, `refuter2_corrected_stats.csv`, `refuter2_corrected_placebo.csv`,
`refuter2_1545_survival.csv`.

## Verdict: the FAIL stands (refuted = false), but the builder's numbers are wrong in ways that matter

### Defects in the builder (none flips FAIL to PASS)
1. **Outcome gate.** `cell_1550.py:57` keeps only rows with |ret_on_next| <= 0.5, a filter on the trade's own
   outcome. Rebuilding the book from the builder's panel columns WITH the gate reproduces the builder CSV exactly
   (Jaccard 1.0000 on all four books). WITHOUT it, the EXTENSION Jaccard is 0.9951 / 0.9976, the same numbers as
   `1550_compare.md`. So **the builder/rebuild divergence comes from this gate, not the ADV min_periods convention
   the compare named.** EXTENSION removes real squeezes (GME 2021-01-26 +140 %, VIRX +215 %, CODX, NVAX): 1550 net
   4.75 -> 17.22 bps. PANEL removes deal-close delistings whose next_open is 0 (FARO, SPTN, HOLX, EXAS, CFLT, CRNX ...
   booked as -100 %) plus real movers.
2. **Renamed-ticker duplicates in EXTENSION.** The same security appears under its old and new symbols
   (EMBJ/ERJ, FB/META, NBIS/YNDX, CMBT/EURN, AGNT/EXPI, CHK/EXE, EYES/VANI): 396 of 9,362 fills (4.2 %) in 1550 and
   700 of 17,058 in 1551. That is double-counting and takes top-N slots. There are 70K duplicate universe rows.
3. **PANEL next_open == 0.** There are 227K such rows in the panel, and they give ret = -100 %. The builder's gate hides them
   by accident. They should be treated causally (deal cash-out, return about 0) or excluded by a causal rule.
4. **Reporting zeros.** The PANEL pooled row shows ex_top1 = capped5 = universe = 0. The recompute gives -9.9, -19.1
   and 5.6 bps. green_weeks = 0 everywhere, but the actual share is about 0.5.
5. **The count-matched null cannot fail.** Its sd is 1.5 to 8 bps because it resamples names within the same nights,
   so night co-movement is removed. It gives the 100th percentile even when the paired placebo t is 1.3. The ">= 99"
   criterion is uninformative (multiplicity/bar-design issue).

### Causal corrected book (no outcome gate, deal-close = 0, duplicates collapsed)
| | EXT 1550 | EXT 1551 | PANEL T+V 1550 | PANEL T+V 1551 | TEST 1550 / 1551 |
|---|---|---|---|---|---|
| net bps @5 | +18.8 | +14.6 | +33.0 | +21.7 | -21.0 / -18.8 |
| t (night) | **2.62** | 2.16 | 1.66 | 1.48 | -0.87 / -1.32 |
| ex-top-5 % fills | -48.8 | -37.9 | -53.6 | -40.2 | |
| ex-top-5 % nights | -19.5 | -11.7 | -13.3 | -4.3 | |
| drop best 2 nights | +14.4 | +12.2 | +14.9 | +13.0 | |
| capped +5 % | -22.3 | -13.2 | -25.8 | -13.7 | |
| placebo margin (t) | +21.0 (2.57) | +14.2 (2.11) | +29.6 (1.52) | +14.9 (1.30) | -25 / -26 |
| margin ex-top-5 % nights | -19.2 | -16.6 | -17.2 | -11.7 | |
| years net > 0 | 3/5 (2020 +72, 2021 +21, 2024 +5; 2022 -16, 2023 -6) | 3/5 | | | |

EXT 1550 ex-2020 is about +2.5 bps. **2020 carries the extension.**
Pass-bar status of the corrected 1550: t >= 2.5 PASS (2.62), placebo >= 5 bps with t >= 2 PASS, null PASS (uninformative),
**ex-top-5 % > 0 FAIL (-49), >= 4/5.5 years FAIL (3/5), PANEL T+V t >= 2 FAIL (1.66)** -> FAIL.
The builder's stated reasons ("t 1.29", "placebo t 1.29") are wrong. The correct reasons are tail dependence,
concentration in 2020, and the panel t.

### Cadence at $3K/name (R = $150, `docs/cadence_bar.md` thresholds)
Corrected EXT 1550: weekly P10 = -$974 (-6.5 R vs the >= -2 R bar), green weeks 50 % (bar 55 %), strong-week gap
median 3 / P90 7.7 wk (bar <= 3 / <= 6). 1551: P10 -8.6 R, green 51 %. **The cadence bar FAILS at every slice.**
The builder's "not computable" is wrong: with R = $150 a strong week is $750, which is computable.

### Execution
* MOC cutoff: Alpaca `cls` orders must be in by 15:50 ET (NYSE imbalance). The 15:59 signal is not executable as stated.
  15:45 survival on the PANEL 1551 members that have minute bars (1,837 of 6,294, 2025+, coverage biased to movers):
  price >= high252 at 15:45 for 95.6 %, cumulative volume >= 1.5 x ADV20 for 70 %, both for **67 %** -> about 1/3 of
  membership changes. The kept members net +8.4 bps (n 1,239) and the dropped ones -23.8 bps (n 598). The members
  a 15:45 rule would ADD are unobserved (the store has no full-universe minute data). **The loss cannot be signed.
  It cannot rescue the FAIL** because the EXTENSION has no minute coverage at all.
* Auction fill: $3K is below 0.03 % of a $10M-ADV name's daily value, so zero impact is plausible for MOC. BUT
  ret uses the daily-bar `open`/`close` (consolidated first/last print), not the primary auction prints. On thin
  names the gap between first print and auction price can be tens of bps. Not measured.

### Multiplicity
This is the second read of a rule chosen on TRAIN/VAL (the lit-review note) with TEST already spent. Cells 1,550 and 1,551
are two correlated cells (N = 10 / 25). A t of 2.6 on the extension after the 2020 concentration is not
multiplicity-robust.
