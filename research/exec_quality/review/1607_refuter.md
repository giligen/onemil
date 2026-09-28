# Refuter review: cells 1,607–1,609 (the quoted spread as a filter)

Reviewer role: adversarial refuter of the builder (`cell_1607.py` / `RESULT_1607.md`) and the rebuild (`rebuild_1607.py` /
`REBUILD_1607.md`) against the frozen `PREREG_1607.md`. Script: `review/refuter_1607.py` (read-only on `data/trades.db`).
Raw output: `review/refuter_1607_out.txt` (HOD section) and `review/refuter_1607_out_BC.txt` (live and backtest ORB).

## Verdict
**The FAIL verdicts stand (refuted = false).** Every pass/fail result holds under every lens below: both HOD filters,
on both spread fields, and on the gross, reported-net and costfix outcomes. The ORB bucket rules fail as well, and cell
1,609 has nothing to report. **The builder's headline needs a correction before it goes to the owner:** "refuted in the OPPOSITE direction —
wide-spread fills are the book's worst trades (t −7.7)" is mostly the spread's own cost. The gross R is flat across
spread quintiles.

## 1. Causality of the HOD spread field: DEFECT in the builder, verdict unchanged
* The builder's field `spread_frac_at_fill` (= 2·half_entry/fill, the same number) is not a quote. `cell_1445.corrected_cost()`
  back-solves it from the realized `cost_R` (`half_entry = cost_R·R − exit_half − fee`). On 6,523 of 9,911 rows,
  `exit_half` is half the **(day, symbol) whole-day mean NBBO spread**, so the field carries information from later in
  the day, and any error in the exit-side cost ends up in it. It has residual artifacts: 31 rows are negative and 60 are
  below 1 bp.
* The builder disclosed this but still scored the verdict on it. A causal field exists and was not used:
  `features_1478_C.csv::spread_bps_at_arm`, the prevailing quote strictly before the trigger print, with 98.2 %
  coverage. The rebuild substituted it. On it the TRAIN-H2 edges are [10.91, 21.10, 34.33, 58.44] bps, and on VAL:
  KEEP-WIDE net −0.259 R (t −6.42), KEEP-TIGHT net −0.060 R (t −1.26). **FAIL, the same as the builder.** The two
  fields correlate 0.83.

## 2. Cost double-count and the gross beside the net: headline DEFECT, verdict unchanged
* The builder's caveat says the cost is the half-spread "charged once at exit per the corrected-cost convention". **This is
  wrong.** `outcome_R` is neither `net_R` (max |diff| 1.46 R) nor the 1445 costfix `net_R + half_entry/R` (max |diff|
  1.11 R). The cost it charges scales about 1 : 1 with the **full** quoted spread: the median ratio of charged cost to
  spread is 1.01 and the correlation is 0.76. Q1 is charged 0.10 R and Q5 0.33 R.
* **GROSS R per quintile of the builder's field (the owner's actual question):**

| Quintile | VAL gross (t) | TRAIN-H2 gross (t) | VAL cost charged | VAL net |
|---|---|---|---|---|
| Q1 ≤ 8.6 bps | +0.059 (1.16) | +0.100 (1.67) | 0.097 | −0.037 |
| Q2 | +0.027 (0.52) | −0.011 (−0.22) | 0.129 | −0.102 |
| Q3 | +0.049 (0.80) | −0.045 (−0.83) | 0.149 | −0.099 |
| Q4 | +0.009 (0.19) | +0.081 (1.34) | 0.192 | −0.184 |
| Q5 > 48 bps | −0.053 (−1.12) | −0.025 (−0.47) | 0.327 | −0.381 |

  The VAL Q1→Q5 net gap is 0.34 R. About 0.23 R of it is the charge that net R makes for the spread by construction.
  Only about 0.11 R is gross, and that part is not significant (t ≈ −1.1 on VAL, −0.5 on TRAIN). The causal field gives
  the same picture: VAL gross runs from +0.084 in Q1 to −0.051 in Q5, all |t| ≤ 1.5.
* **The correct statement:** wide-spread fills are **not** the winners. Their gross R is no higher, it is flat to
  slightly lower. They lose more net because they pay about 0.3 R of spread that the flat gross does not cover. The
  spread is a **cost, not a signal**. "Opposite-direction signal, t −7.7" overstates the finding.
* The filters on every outcome definition (VAL, builder's field) all fail the +0.15 R / t 2.5 bar:
  * KEEP-TIGHT: gross +0.042 (t 0.97, ex-top-5 % −0.060), costfix −0.020, reported net −0.072.
  * KEEP-WIDE: gross −0.025, costfix −0.240, reported net −0.290.
  * On the causal field, KEEP-TIGHT gross is +0.050 (t 1.09).

## 3. Quintile edges set on TRAIN only: CLEAN
The edges come from TRAIN-H2 only (4,398 rows, all `half == H2`): [8.58, 17.85, 28.28, 48.00]. They match the builder and
the rebuild exactly. The outer bins are open, so the VAL maximum of 610 bps lands in Q5. VAL was never read to choose a
cut.

## 4. Tails and day concentration of any positive bucket: NONE SURVIVES
* **1,607:** there is no positive net bucket. The gross positives die under both checks:
  * VAL Q1 +0.059: ex-top-5 % −0.043, ex-top-5-days −0.017.
  * VAL Q3 +0.049: ex-top-5 % −0.052.
* **1,608 backtest:** every positive bucket is tail- or day-carried:
  * ALL +0.082 (t 1.98): ex-top-5 % −0.034, median −0.107. The top 5 % carry 140 % of the sum. 2023–24 alone (n 87): −0.005.
  * ≤ 50 bps +0.091: ex-top-5 % −0.017. 2023–24 alone (n 43): −0.098.
  * 100–150 bps +0.174 (n 21): ex-top-5-days −0.095.
  * 150–300 bps +0.120 (n 29): ex-top-5-days −0.148.
* **1,609:** the only positive drift bucket, (10, 31] at +0.172, is one +6.19 R trade (150 % of the bucket's sum).
  Without it the bucket is −0.090. Faster bursts are not the winners.

## 5. Live ORB sample and R definition: CLEAN, with one nit
* There are 123 fills over 53 days, 2026-05-19 to 09-23. By bucket: ≤50: 81, 50–100: 31, 100–150: 9, 150–300: 2, >300: 0.
  The entry quote is causal. `orb_engine` sets `entry_quote_*` once at submit, and the fill quote is a separate column.
* The builder's R = pnl / ((entry − stop)·filled_qty). It matches pnl/total_risk on 122 of 123 rows (overall −0.104 vs
  −0.111). There are no partial or scale legs, and pnl equals the single leg on every row. No row has its stop above the
  fill.
* One dust fill inflates the 100–150 bucket: ATPC 6/01, 6 shares, risk $1.08, gives +0.61 R against +0.003 on
  total_risk. That bucket's mean is −0.184 on total_risk, not −0.119. It changes no verdict.
* The "P&L share" column is in dollars and confounded by risk size. The 100–150 bucket's 48 % of the −$5,281 comes from
  two losers with about $1,000 of risk each (FABC, EHGO). R is the only comparable unit here.
* The builder tested only two bucket rules: skip ≤ 50 and skip > 300. The one other rule with n ≥ 20 on live, skip
  50–100, fails too: its backtest dropped mean is +0.004. **No bucket rule can pass. The FAIL is complete.**

## 6. Is the ORB backtest spread point-in-time? YES, with one labelling nit
* The spread is the last two-sided XNAS mbp-1 quote strictly before t*. Quote age: median 0 s, mean 0.84 s, maximum
  76 s. None is crossed or locked, and all 301 rows come from the primary tape.
* The `hod_ofi/raw` fallback is dead code. That directory is named `date__NNNN`, not `date__SYMBOL`, so no other quote
  schema entered the data.
* The builder calls this spread "NBBO", but XNAS.ITCH top-of-book is the **Nasdaq-only** best bid and offer. On 20
  (date, symbol) overlaps with the live SIP quote, the median ratio is 1.00 (30.5 vs 35.3 bps). The medians of the full
  distributions are also close (33.9 vs 31.3). The difference is not material to the buckets.

## Required corrections before the owner sees it
1. Lead with the causal field (`spread_bps_at_arm`), not `spread_frac_at_fill`.
2. Put the gross beside the net. The owner's answer is: "no, the wide-spread trades are not the winners. Gross R is flat
   across spread quintiles (±0.1 R, |t| < 1.7). The wide ones lose more only because they pay the spread."
3. Fix the cost caveat: `outcome_R` charges about the full quoted spread, not a half-spread at exit.
4. State the verdict unchanged: spread is closed as a filter on HOD, the ORB 300 bps gate stays, and 1,609 has no signal.
