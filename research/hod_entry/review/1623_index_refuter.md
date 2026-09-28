# Cell 1,625 (index as the instrument) — adversarial refuter

Spec: `PREREG_1623.md` lines 27-34 (FROZEN). Under review: `cell_1625.py`, `cell_1625_signals.csv`,
`index_bars_1625.parquet`, `RESULT_1625.md`. Checks: `review/1623_index_refuter_chk.py` (+ `_chk2.py`),
numbers in `review/1623_index_refuter_chk.json`. Report-only; nothing tuned.

**Verdict: the FAIL stands (not refuted).** No defect found here turns it into a PASS. But one of the two
"passing" lines the builder lists, the VAL placebo margin of +5.90 bps, comes from time of day. It is not
evidence and must not go to the owner as a partial positive.

## 1. Signal minute and causality
- The builder's per-day SPY and IWM returns were recomputed from the parquet with a separate lookup. They match
  on all 209 rows (max abs diff 0.0000 bps). The entry is the open of m*+1 and the exit is the open of m*+61.
  Every entry and exit bar sits at its exact minute, so the 5-minute tolerance was never used.
- B30 counts arms in (m*−30, m*], which includes minute m* itself. `arm_m` equals floor(fill_min) for 97.8 % of
  fills, so an arm counted at m* is a break inside minute m*. It is known by the m*+1 open and the entry is
  causal. On 3 of the 209 days the count includes a fill that lands after the m*+1 open. On only 1 of them
  would removing it leave the count below τ.
- A strictly causal re-run counts fills by floor(fill_min) instead of arm_m. It reproduces the book to 0.01 bps
  (VAL +1.34, t 0.39; TRAIN-H2 −4.16). The builder's caveat calls the fill-proxy a "causality weakening". For
  timing that is overstated. The real gap is that nofill arms are missing, which the rebuild also flags.
- Entering 1 minute later changes little: VAL +3.03 (t 0.88), TRAIN-H2 −5.13 (t −2.04). Still FAIL.
- τ comes from TRAIN-H2 minute values only. It reproduces at 7.0. Integer ties put 11.8 % of pooled minutes at or
  above τ, not 10 %. 43 % of pooled minutes are zero, because fills end at 14:01 and B30 is 0 from about 14:31.
  So the "top decile" fires on 209 of 230 days (91 %). It behaves like a morning clock, not a burst.
- The verdict does not depend on how τ is read:

| τ reading | VAL n | VAL mean bps | VAL t | TRAIN-H2 mean | share of VAL signals < 10:00 |
|---|---|---|---|---|---|
| full-session grid p90 = 7 (builder) | 101 | +1.34 | 0.39 | −4.17 | 77 % |
| grid p90 over 09:30–14:30 only = 9 | 98 | +1.63 | 0.49 | −5.43 | 69 % |
| event-pooled 23 (rebuild) | 42 | −0.42 | −0.09 | +2.47 (n 16) | 31 % |

Every reading fails, and in every reading TRAIN-H2 has the opposite sign to VAL.

## 2. Bar source and timestamps (UTC vs ET)
- cache.db stores UTC-aware ISO timestamps (`2025-01-02T14:30:00+00:00`), and the builder converts them to ET
  correctly. No bar's day label disagrees with its ET date.
- The 09:30-ET bar open was compared with the `daily_bars` open. SPY from the cache (179 days): median 0.15 bps,
  max 0.63. SPY from Alpaca SIP (51 days; the builder's text says 48): median 0.07, max 0.46. IWM from Alpaca
  (230 days): median 0.00, max 2.04. No day is off by more than 20 bps.
- Both DST transitions fall inside the window and there is no hour shift. The cache/Alpaca splice is clean.

## 3. Random-minute placebo
- The seed-1625 draws replicate exactly.
- **Defect (immaterial):** `build_trade` has no guard for exit ≤ entry. The 2 placebo draws at 15:57 and 15:58
  (both TRAIN-H2) produce a backward "hold": exit at the 15:55 open, entry at 15:58/15:59. Dropping them moves the
  TRAIN-H2 margin from −2.53 to −2.64. VAL is unaffected.
- **Design confound (it changes the "passing" line):** the placebo minute is uniform over the session, with a
  median of 12:51. Real signals have a median of 09:51 on VAL. Unconditional VAL SPY 60-minute returns by entry
  hour: 09h +0.52, 10h +1.51, 11h −2.36, 12h −2.59, 14h −2.40, 15h −1.48 bps. The fair comparison is the
  same-minute baseline: the mean over every other day of the same holdout. Against it the margins are:

| leg | holdout | real | same-minute baseline | margin | t |
|---|---|---|---|---|---|
| SPY | VAL | +1.34 | +0.64 | **+0.70** | 0.20 |
| SPY | TRAIN-H2 | −4.17 | −4.17 | 0.00 | 0.00 |
| IWM | VAL | +6.19 | +4.51 | +1.68 | 0.28 |
| IWM | TRAIN-H2 | −6.95 | −5.01 | −1.94 | −0.43 |

The builder's +5.90 (SPY) and +11.56 (IWM, t 2.10) VAL margins are almost all "morning vs afternoon". Only
signals per week genuinely clears its line.

## 4. Opening drift and signal-time distribution
When m* falls (bucket counts TRAIN-H2 / VAL):
- 09:30–09:44: 2 / 16
- 09:45–09:59: 51 / 62
- 10:00–10:29: 46 / 20
- 10:30–10:59: 6 / 1
- 11:00+: 3 / 2

The result is not an opening drift. The pre-10:00 signals lose money:

| subset | VAL SPY | TRAIN-H2 SPY | VAL IWM | TRAIN-H2 IWM |
|---|---|---|---|---|
| m* < 10:00 | −2.37 (n 78, t −0.58) | −5.45 (n 53) | −0.35 | −12.17 (t −2.11) |
| m* ≥ 10:00 | +13.90 (n 23, t 2.36) | −2.93 (n 55, t −0.83) | +28.34 (t 2.85) | −1.91 |

The VAL mean is carried by 23 signals after 10:00. They cannot be used:
- It is a post-hoc subset chosen on VAL.
- TRAIN-H2 has the opposite sign for the same subset.
- It gives 1.05 signals per week on VAL, below the ≥ 2/week bar.

## 5. Tails and day concentration (VAL)
- SPY: the sum over 101 days is +135 bps. The top 5 days add +381 bps, 2.8 times the total. Mean without the
  top day: +0.12. Without the top 5: −2.56. The biggest day is 2026-04-02 at +122.7 bps.
- By month: Jan −1.5, Feb −11.2, Mar +7.1, Apr +3.2, May +8.1 bps.
- IWM: the top 5 days are 98 % of the +625 bps sum. Without them the mean is +0.12.
- The builder's VAL placebo margin: the top 5 days are 64 % of the sum. Without them it is +2.22 bps (t 0.78).

## Bottom line
The cell 1,625 FAIL is correct and robust:
- 7 checks rule out a hidden PASS: causal timing, the timestamps, obtainability, a 1-minute delay, three τ
  readings, the time-of-day-matched placebo and the tails.
- The builder's partial positives do not hold up. The placebo margin is time of day. The "causality weakening"
  caveat is immaterial for timing.
- What the index frame does not carry: on this population, a morning break count in the top decile (as a
  fill-proxy) predicts nothing about the next 60 minutes of SPY or IWM beyond the time of day. The TRAIN-H2 and
  VAL signs disagree under every reading tried.
- Minimum detectable effect: roughly 2.8 × the SE of about 3.5 bps, so about ±10 bps per signal on VAL SPY.
