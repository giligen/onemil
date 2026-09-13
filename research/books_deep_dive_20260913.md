# BF and ORB — deep dive (2026-09-13, after the ignition kill)

Owner: "ultrathink and go deep on BF and ORB". Method: the checklist that killed ignition,
applied to the two surviving books — (1) look-ahead in every decision-time feature,
(2) survivorship, (3) tail dependence with per-era ex-tail expectancy, (4) how the rules were
chosen (in-sample vs out-of-sample), (5) what the live tape says so far, (6) what the numbers
mean for $10K/month.

## 0. Bottom line

- **Neither book has the ignition disease.** Every feature that decides a BF entry or an ORB
  order is computed at or before the decision bar (full audit §2). The ORB catalyst cohort is
  the 9:35 candidate list, not an outcome; BF's tier feature is a strict pre-entry replay.
- **Both books are positive with the tail removed in every era** — the bar ignition failed:
  BF ex-tail +0.17 / +0.42 / +0.15 R, ORB +0.38 / +0.19 / +0.10 R (25H1 / 25H2 / 2026).
- **Both are small samples and both had their rules chosen on the whole sample.** BF 56 trades,
  ORB 61 fills, in 21 months. P1's three rules and ORB's veto thresholds were picked looking at
  2025 AND 2026. There is no clean out-of-sample backtest for either; the only out-of-sample
  data is the live tape, which is 0 BF trades (5 sessions) and 2 ORB fills (20 sessions).
- **Both books decayed in Jun–Aug 2026** (BF 7 trades −$1.6K at $2K risk; ORB 10 fills −$321
  at stage). The combined honest book is negative the last two months.
- **$10K/month is a full-scale number, not a current one.** At $2K BF risk and ORB stage S3
  ($100K) the combined honest book averages $9.6K/month, median $6.9K, 9 of 20 months ≥ $10K,
  3 red, worst −$6.4K. At today's sizes (BF $150, ORB $10K) the same book is ≈ $800/month.
  Getting from here to there is the ramp, gated on live positive P&L, and that takes months.

## 1. The honest books (regen-7 / entered-inclusive B+; relative tools, not forecasts)

| | BF P1 ($2K risk, $50K) | ORB B+ ($10K stage, $375 risk) |
|---|---|---|
| period | 2025-02 → 2026-08 | 2025-01 → 2026-08 |
| trades | 56 (2.7 / month) | 61 fills of 79 picks (3 / month) |
| total | $126,257 | $6,627 |
| mean R | +0.66 (t = 3.4) | +0.29 (t = 2.6) |
| mean R ex-tail (R < 2) | +0.22 (t = 1.3, P(≤0) = 0.10) | +0.21 (t = 2.7, P(≤0) = 0.002) |
| WR | 66% | 43% |
| trades ≥ 2R | 10 = 79% of dollars | 1 = 28% of dollars |
| top-5 trades share | 55% | 66% |
| weeks green (weeks with trades) | 28/40 | 22/42 |
| worst month | −$12.7K (Feb-25) | −$187 |
| months red | 5/19 | 6/19 |
| Jun–Aug 2026 | 7 trades, −$1.6K | 10 fills, −$321 |

Per era (ex-tail mean R / WR): BF 25H1 +0.17 / 63%, 25H2 +0.42 / 73%, 2026 +0.15 / 64%.
ORB 25H1 +0.38 / 53%, 25H2 +0.19 / 40%, 2026 +0.10 / 38%.

Reading. BF's edge is a 66% hit rate on the +2R partial (median trade +1.04R) with the trail
running on a third of them — the dollars are tail-heavy but the per-trade expectancy is not
tail-only. ORB's edge is smaller, steadier, almost tail-free, and shrinking era over era.
Bootstrap 5th percentile of mean R: BF +0.35, ORB +0.12 — both books are more likely real
than not on their own history, with the caveat in §3 that the history helped write the rules.

## 2. Look-ahead audit (code read, file:line in the audit transcript)

BF decision features — `intraday_change_at_entry` (tier), `intraday_range_pct` (V-reversal),
gap, pole gain, flag volume, every conviction input, `conviction_mult`, MACD zone, VWAP gate,
regime (T-1), `max_entry_price`, `min_pole_gain_pct`: all computed on bars up to the setup bar
or prior sessions. **Causal.**

ORB decision features — the 7 composite inputs, quintile, PM$ mult (04:00–09:30), news flag
(prev-day 15:00 → 09:35), catalyst cohort (9:35 candidate list), PDR / range-size / G1 vetoes,
ATR floor, universe screen (today's OPEN only, prev-day close/volume): all at or before 9:35.
**Causal.** The one same-day daily-bar field used anywhere is the open.

Found, ranked:
1. **BF cache screen applies the $1–$30 price band to the daily CLOSE** (`bf_selection.py`
   `price_ref` = close). Measured on daily_bars 2025-01 → 2026-09-11: 271,195 eligible
   symbol-days; the close band drops **1,779 that opened in band and closed > $30** (runners —
   conservative bias) and **3,534 that opened in band and closed < $1** (crashers — optimistic
   bias); it keeps 4,983 that opened < $1 and closed in band. Small (0.7% / 1.3%) and
   two-sided; not tested through the detector. Fix at the next cache regen: band on "any moment
   in band" (`high ≥ min and low ≤ max`), the monotone causal superset the module already
   documents for the range term.
2. **ORB universe requires cached 1-min bars, and one historical feeder chose pairs by full-day
   range** — already tested by the 9/5 Databento point-in-time top-up: +$395 at book level, not
   load-bearing. Closed.
3. **BF composite feature `qf_fill_vwap_dist_pct`** is the completed FILL bar in BT (entry is
   intrabar → sub-minute look-ahead) while live substitutes the setup-bar value. Tested: Stage-2
   with the live definition reproduces P1 exactly ($126,256.65 / 56). **Inert on P1**; still a
   one-spec violation to unify (BT should use the setup-bar value). Same class: `entry_minute`
   = fill minute in BT vs now() at setup live; `qf_vwap_dist_pct` boundary one bar fresher in BT.
4. **BF 200K volume gate fallback** uses today's universe snapshot when the point-in-time
   20-day average is missing. Mild.
5. **Latent, off**: `scanner.min_dollar_volume` (close × total volume) and the EOD
   `daily_range_pct` Stage-2 filter. Keep off; redefine before ever enabling.

## 3. How the rules were chosen (the real weakness)

- **BF P1**: the three raw rules (VWAP gate, pole ≥ 5%, entry ≤ $20) were the worst buckets in
  BOTH years — chosen with 2026 in view. §6e of the consistency study says a 2025-only picker
  finds the same rules "mostly". The composite z-params were fit on 2025 (OOS for 2026: +$38K
  on 22 trades). Walk-forward refit of those z-params (13w–39w, expanding) is WORSE than the
  frozen fit everywhere ($121–123K vs $139K from 2025-03) — no more fitting to do.
- **ORB B+**: z-params, quintile cutoffs and mults are TRAIN-frozen (weekly 26-week selection
  refit walk-forward validated 9/8). But the 7-feature set is a whole-sample correlation
  ranking, and the range-size (2.221), G1 and PDR thresholds are full-period searches. The
  reported $6.6K spans the fitting period.
- Consequence: the backtests overstate both books by an unknown amount. Ignition's ex-tail
  numbers were negative even in-sample; these are positive in-sample. That is the difference,
  and it is not proof.

## 4. Live so far (the only out-of-sample data)

- **BF P1** (from 9/8, L0 $150): 0 trades in 5 sessions. Expected at 2.7/month: ~0.6/week —
  zero is normal, not a defect. Ramp needs ≥ 8 trades and ≥ 15 sessions: earliest mid-October.
- **ORB B+** (from 8/17, S0 $10K): 2 fills, +$113, slip −26 bps vs model 30. Ramp checker
  HOLD on fills 2/8 and "parity defects 2": PFSA 8/31 (the entered-only look-ahead, fixed 9/5
  by the entered-inclusive rebuild) and ALMS 9/2 (live scored it Q2 and did not order; the
  pre-9/8 pipeline picked it; the rebuilt book does not). Both explained; neither is a live
  defect. The checker window starts 8/17 so it keeps counting them — note for the next review.
- Ignition: killed today (shadow, prestage, engine off; crons removed).

## 5. Decisions and actions

Done today: ignition off. This document. No config changes to BF or ORB.

Recommended (my call, will do unless told otherwise):
1. Unify the BF composite feature definitions BT ↔ live (`qf_fill_vwap_dist_pct` → setup-bar
   value, `entry_minute` → setup minute) and regression-pin with a parity test. Inert on P1
   now; cheap; removes a producer-of-two-numbers.
2. Cache regen-8 rule: price band on any-moment-in-band. Do it when regen-8 runs for other
   reasons; not worth a standalone regen (two-sided, ~1% of pairs).
3. Pre-registration for every future rule change on either book: the rule is chosen on data
   before a date written down first, scored once on the data after it, and the TEST
   week-by-week is reported with the tail removed. Same as `research/ignition_zero/DESIGN.md`.
4. No new features, no new vetoes, no re-tuning on either book until the live tape has
   ≥ 8 trades per book. The backtests have been mined; the marginal rule from here is more
   likely to be noise than edge.

What I would not do: scale either book on the backtest. The ramp gates on live P&L are the
right instrument and they are already in place (`docs/bf_p1_ramp.md`, `scripts/orb_ramp_check.py`).

## Files
`research/bf_consistency/stage2_P1_liveconfig_2k_dll5u.csv` (BF P1 honest book),
`analysis_results/orb_bplus_book.csv` (ORB B+ honest book),
`research/bf_refit_walkforward/summary.csv`, `research/orb_refit_walkforward/REPORT.md`,
`research/ignition_zero/REPORT.md`.
