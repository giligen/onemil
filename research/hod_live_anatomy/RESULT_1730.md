# RESULT — cell 1,730: the entry-cap ladder (PREREG_1730.md) — Opus agent, 2026-10-31

**Verdict: FAIL. The cap stays at +15 bps.** No rung clears the bar on either split. Tighter caps lean slightly
positive (+0.005 … +0.023 R/trade, best t 1.9), looser caps lean negative (A TRAIN-H2 c = 50: −0.009 R, t −2.9;
B1 c = ∞: −0.034 / −0.015 R). On the bar-open fill model a tighter cap mostly UN-FILLS: 81 % of BT signals open
at/below the level (d = 0), so the cap only touches the 19 % that open above it, and their gross at the level is
not consistently worse.

Script `cell_1730.py`; ladder `ladder_1730.csv`; full log `cell_1730_out.txt`; cache `cache/opens_1730.parquet`
(36,184 armed crossings + the 5,417 population-A crossings, trigger-bar OHLCV, d, outcome, features).

## Cap parameters (from the code)
* `entry_limit_pct` (`trading/hod_break.py` l.44, default 0.0015, not set in config) = the **resting stop-limit**
  limit: `arm_state` l.127 `limit = level × (1 + entry_limit_pct)`, trigger = level + 0.01. The engine logs it at
  boot (`hod_break_engine.py` l.325, "limit 0.1500%"), and a triggered order never fills below its trigger (l.992).
  **This is the cap the ladder varies.**
* `cap` (`config.yaml hod_break.cap: 0.006` = 60 bps) only applies to the legacy `next_open` marketable entry
  (`_try_enter`, l.2299: `limit = level × (1 + cap)`, NO CHASE if ask > limit). It does nothing under
  `entry_mode: resting_stop_limit`.

## Fill-model counts
* Population A (the PREREG's, 5,506 BT fills, 1663 ⋈ causal_arming_causal 1:1, 0 join misses, stops agree 100 %):
  5,417 recovered, **89 dropped from every rung** (no SIP bar at the trigger minute). d = 0: 80.9 %; d > 15: 1.4 %.
  Walk check vs 1663: gross corr 0.984 (+0.063 level basis vs +0.037 fill basis); my entry is a median 4.4 bps
  below the 1,438 NBBO-ask fill, so absolute net levels are optimistic vs measured cost (live −0.026); paired Δs
  are unaffected (same entry on every rung).
* Population B (report-only; deviation 1): 11,921 symbol-days with an armed crossing in 1,438 → 8,462 with ≥ 1
  armed crossing under the CURRENT live gates (min_price 20, min_r_pct 1.5), 36,184 crossings; B1 = first armed
  crossing (d > 15: 15.0 %), B2 = a declined order re-arms at the next armed crossing. 119 symbol-days had no bars.
* Weeks: TRAIN-H2 27, VAL 22. Live: 231 fills / 5 weeks (227 with entry slip). Live/BT fill ratio at c = 15:
  46/84 = 0.55 (TRAIN-H2), 46/139 = 0.33 (VAL) → every rung ≥ 3 fills/wk at the live rate.

## L1 — the ladder (population A, net R/trade at the level basis, measured exit cost)
| cap | TRAIN-H2 n | fills/wk | net | R/wk | Δ vs 15 (boot t) | ex5 Δ | A/B Δ | VAL n | fills/wk | net | R/wk | Δ vs 15 (boot t) | ex5 Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,841 | 68 | +0.046 | 3.16 | +0.009 (0.68) | +0.012 | +0.037/−0.018 | 2,541 | 116 | +0.040 | 4.56 | +0.023 (1.94) | +0.027 |
| 5 | 2,057 | 76 | +0.052 | 3.96 | +0.014 (1.48) | +0.018 | +0.033/−0.003 | 2,777 | 126 | +0.021 | 2.64 | +0.005 (0.57) | +0.008 |
| 10 | 2,181 | 81 | +0.044 | 3.56 | +0.007 (0.86) | +0.007 | +0.011/+0.002 | 2,966 | 135 | +0.017 | 2.31 | +0.001 (0.21) | +0.002 |
| **15** | 2,273 | 84 | +0.038 | 3.16 | — | — | — | 3,068 | 139 | +0.016 | 2.25 | — | — |
| 25 | 2,295 | 85 | +0.033 | 2.82 | −0.004 (−1.62) | −0.005 | | 3,083 | 140 | +0.016 | 2.28 | +0.000 (0.09) | −0.000 |
| 50 | 2,309 | 86 | +0.028 | 2.43 | −0.009 (−2.88) | −0.010 | | 3,104 | 141 | +0.016 | 2.23 | −0.000 (−0.11) | −0.001 |
| ∞ | 2,312 | 86 | +0.032 | 2.72 | −0.006 (−1.48) | −0.008 | | 3,105 | 141 | +0.015 | 2.17 | −0.001 (−0.25) | −0.002 |

Day-clustered t of the books ≤ 1.0 everywhere; ex-top-5 % of every book −0.05 … −0.09; winner-capped ≈ net.
B1 (first armed crossing, current gates): Δ vs 15 TRAIN-H2 +0.002/+0.005/+0.007 (c 0/5/10), −0.012/−0.028/−0.034
(25/50/∞, t −2.1/−2.9/−2.9); VAL +0.012/−0.002/+0.000, +0.004/−0.007/−0.015. B2 (re-arm) same picture
(TRAIN-H2 c = 5 +0.053 vs 15 +0.041; VAL c = 5 +0.010 vs +0.008; ∞ +0.005 / +0.001).

**Pass bar.** c* < 15 needs Δ ≥ +0.03 with t ≥ 2 (Bonferroni over 6 rungs → |t| ≥ 2.64) on BOTH splits: best is
c = 0 VAL +0.023 (t 1.94) with TRAIN-H2 +0.009 (t 0.68) and TRAIN-H2 R/wk 3.157 < 3.163 → FAIL on every rung.
Looser caps: Δ ≤ 0 everywhere → FAIL (the live > 15 bucket, −0.31 R n 22, agrees in sign).
**MDE of the paired Δ** (80 % power, Bonferroni z 2.64 + 0.84): c = 5 ≈ 0.031 (TRAIN-H2, SE 0.0094) / 0.031 (VAL,
SE 0.0088) R; c = 0 ≈ 0.046 / 0.041 R. The test could see the pre-committed +0.03 at c = 5 and did not.

## L2 — BT twin of the live slip table (c = 15)
| d bucket | TRAIN-H2 n / net / gross | VAL n / net / gross | LIVE slip bucket n / net / gross |
|---|---|---|---|
| ≤ 5 | 2,057 / +0.052 / +0.091 | 2,777 / +0.021 / +0.061 | 55 / +0.131 / +0.119 |
| 5–10 | 124 / −0.086 / −0.041 | 189 / −0.038 / +0.004 | 72 / −0.051 / +0.035 |
| 10–15 | 92 / −0.117 / −0.074 | 102 / −0.013 / +0.029 | 78 / −0.002 / +0.145 |
The BT ≤ 5 bucket IS better than 10–15 in both splits (+0.17 / +0.03 R), but every bucket t is < 1 and the ≤ 5
bucket is 90 % d = 0. VAL 0 < d ≤ 5 is −0.180 (n 236) vs TRAIN-H2 +0.100 → no gradient within the cap.
Report-only read on the 1,438 NBBO-ask entry slip (the live table's true twin): net (1663) ≤ 5 / 5–10 / 10–15 =
+0.011 / −0.012 / −0.088 (TRAIN-H2), −0.015 / −0.044 / −0.036 (VAL), but **gross at the level** +0.083 / +0.088 /
+0.031 and +0.056 / +0.052 / +0.084 — the net gradient is the chase cost itself, not selection. The live table's
+0.13 vs −0.00 is the same: noise + cost (live 10–15 gross +0.145 is the best gross bucket).

## L3 — what a tighter cap gives up (population A)
| split | c | declined n | their net if filled at open_T | kept net | retrace share (trigger-bar low ≤ level + c) |
|---|---|---|---|---|---|
| TRAIN-H2 | 5 | 216 | −0.099 (t −0.9) | +0.052 | 59.7 % |
| TRAIN-H2 | 10 | 92 | −0.117 (t −0.6) | +0.044 | 71.7 % |
| VAL | 5 | 291 | −0.029 (t −0.3) | +0.021 | 59.1 % |
| VAL | 10 | 102 | −0.013 (t −0.1) | +0.017 | 59.8 % |
Declined signals are mildly worse but not significantly; and 53–72 % of them trade back to the tighter limit
inside the same minute, so a live tighter cap would RETRACE-FILL most of what this model declines (no position →
a limit fill on a pull-back), i.e. the ladder overstates the selection a live cap change would produce.

## Live OOS (cohort with slip ≤ c vs live mean −0.026)
≤ 0: n 18 −0.204 [−0.91, +0.61]; ≤ 5: n 55 +0.131 [−0.25, +0.54]; ≤ 10: n 127 +0.028 [−0.21, +0.27]; ≤ 15: n 205
+0.017; ≤ ∞: n 227 −0.015. Sign agrees for c = 5/10, CIs straddle zero; cannot pass alone (and the bar failed).

## Ranked further entry-quality hypotheses (report-only reads on B1, NOT tests; each needs its own PREREG)
1. **Break volume vs consolidation volume.** B1 quintiles of v_T / mean(v over the 10 bars before): Q1 (≤ 0.65×)
   −0.095 (TRAIN-H2) / −0.086 (VAL), both splits; Q5 +0.113 / +0.013. Mechanism: a break on thin prints is a drift
   into an empty book, no demand behind it. **Caveat: v_T includes post-cross volume of the trigger minute
   (fractional-minute look-ahead)** — the causal version is SIP trade volume up to the crossing print vs the
   consolidation, from the 1,438 tape cache (`research/hod_entry/` sip_rebuild / causal_arming `fetch_window`), or
   bar j volume (`cache/opens_1730.parquet` has v_cons10; add v_j).
2. **Retrace fills = the real cost of a tighter cap.** 53–72 % of the signals a tighter cap declines come back to
   the limit in the same minute; the passive-entry memo (`project_passive_entry_adverse_selection`) says limit
   fills on pull-backs select the broken breaks. Test on the 1,438 tick tape: fill those at the limit on the
   retrace print and walk them (`cache/opens_1730.parquet` l_T, level, d_bps + the tape windows).
3. **Gross at the level vs NBBO chase inside the cap is flat** (L2 report-only read): the lever is not WHICH breaks
   but WHAT is paid — e.g. trigger at level + 0.01 with the limit at the touch for wide-spread names vs a
   spread-scaled cap (`spread_bps` in fills_1721.csv; BT spread from `research/bf_zero/causal_filter/nbbo.csv`).
   d as a continuous feature (B1 d_raw quintiles: Q5 −0.145 / −0.074, inner quintiles split-disagree) and minutes
   since the level formed (Q1 VAL −0.131 vs TRAIN-H2 +0.004) do not agree across splits — not ranked.

## Deviations
1. Population B added: A is conditioned on the 1,438 BT's own 15 bps NBBO fill (only 1.4 % of A has d > 15), so
   A cannot price the looser rungs; B re-derives every armed crossing with `causal_arming.armed_crossing_bars` on
   the same SIP bars under the current live gates (`live_params()`: min_price 20, min_r_pct 1.5). Report-only.
2. open_T < level fills AT the level (PREREG); the engine fills ≥ trigger = level + 0.01 (1 cent).
3. Trigger bar: when it opened below the level, a stop in that bar counts only if it CLOSED ≤ stop (the low may
   precede the cross); stop before target when both are in one bar; target = entry + 2 × (entry − stop) as the
   engine (l.1076); R unit = level − stop on every rung.
4. Paired-Δ t = day-cluster bootstrap (2,000 multinomial day weights) of mean(c) − mean(15); the per-signal paired
   t (persig_t in the CSV) agrees in sign. Bonferroni on 6 rungs → |t| ≥ 2.64 (moot: no Δ reached +0.03).
5. Entry model ≈ 4 bps cheaper than the BT's NBBO ask (see counts): absolute net levels are optimistic.
6. Extra report-only read (BT net / gross by the 1,438 NBBO-ask slip) and the three feature quintile reads.

## Multiplicity
Cell 1,730 = one family of 7 rungs (A primary) + B1/B2 report-only + L2/L3 + 4 report-only reads. HOD live-book
cells since 10/30: 1,721–1,730 (10 cells); every one inside noise of the live rule or negative.
