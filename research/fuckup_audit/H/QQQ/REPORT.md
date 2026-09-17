# H/QQQ — bottom-up loser analysis of the QQQ noise-band sleeve

Stage H of `research/fuckup_audit/PLAN.md`, method `H/METHOD.md`, applied on the **DAY** unit (this is a
daily-frequency ETF book: the 12 semi-hourly decisions live inside one day, the book is flat at every close).
Run 2026-09-17. Everything here is RESEARCH: no config, no service, no order was touched. All work under
`H/QQQ/`.

Simulator: `Q/zsim.py`, scenario **C** (decide on the closed bar, market order at the **next bar's open**,
**0.5 bp/leg**, flat at the 15:59 close), VM = 1.0, checks 10:00-15:30.
**Anchor reproduced exactly**: Q IS 2016-2023 = 6.48 bps/traded day (t 3.32, SR 1.18, MDD 9.9%);
Q OOS 2024-> = **4.57 bps/traded day, t 1.27**, SR 0.77, MDD 10.6% — identical to `Q/REPORT.md` section 3.

Splits for THIS stage (declared in the brief, read in order): **TRAIN 2016-01-01..2022-12-31** (1,750 days,
1,044 traded) · **VAL 2023-01-01..2024-06-30** (374 / 222) · **TEST 2024-07-01..2026-09-30** (554 / 313).

---

## 0. One page

**Two causal day filters were found on the TRAIN losers, frozen in writing, and then FAILED on VAL. TEST was
not read.**

The TRAIN anatomy is clean and the mechanism is real: the sleeve loses on days whose **noise band is narrow
in absolute terms** and whose **overnight gap is small relative to that band** — days with no overnight
impulse, where the close crosses a tight band on noise and the VWAP/band stop sits immediately under the
entry. The two vetoes built on that mechanism (`band_w >= 0.8 %`, `|gap|/band >= 0.20`) took TRAIN from
**6.24 -> 11.04 bps per traded day** (t 2.91 -> 3.55), halved the drawdown (9.9 % -> 4.7 %), improved **both**
TRAIN halves, and — the one test the raw sleeve fails — survived the tail: ex-top-5%-days **+0.32** vs
**-3.79** for the unfiltered book, capped at +1%/day **+4.35** vs **+1.01**.

On VAL the stack raises the mean the same way (7.82 -> 12.54 bps/traded day) **but the days it throws away
are profitable**: the vetoed set is **+1.39 bps/day** (n = 94), so the pre-committed criterion "the vetoed
bucket is negative on VAL" fails and, by rule 5 of `H/QQQ/PREREG.md`, the stage stops.

The decomposition says why, and it is the important result:

| | TRAIN | VAL |
|---|---|---|
| unfiltered book | +6,515 bps over 1,044 traded days (+6.24/day) | +1,736 over 222 (+7.82/day) |
| filtered book | +6,456 over 585 (**+11.04**/day) | +1,605 over 128 (**+12.54**/day) |
| **the days removed** | **+59 bps over 459 days (+0.13/day)** | **+131 over 94 (+1.39/day)** |

The filter's whole apparent gain is a **denominator effect**. It removes low-variance days; the mean per
traded day rises, the money falls. At $60K the unfiltered sleeve is 3.72 bps/calendar day on TRAIN
(~ **$469/month**) and the filtered one is 3.69 (~ **$465/month**); on VAL 4.64 (~ $584) vs 4.29 (~ $541).
**The filter costs money in both splits while improving every ratio.** My own pre-registration required the
mean to improve and did not require the sum to — that is a defect in the pre-registration, and it is recorded
here rather than repaired after the fact.

Only the **intersection** cell of the 2x2 is genuinely bad on TRAIN (both conditions failing: n = 171,
**-5.18 bps/day**, -886 bps total) — and on VAL that same cell is **+3.93**. The sign flips out of sample on
47 days. The one-sided marginals that looked like filters (F1 alone -1.50/day, F2 alone -1.04/day on TRAIN)
are the same 171 days plus a positive remainder.

**Power — the smallest effect these tests could see.** TRAIN traded-day sd 69.3 bps over 1,044 days ->
MDE@80% = **6.01 bps/traded day**, against an observed unfiltered book of 6.24: the in-sample edge is sitting
exactly on its own detection floor. VAL: sd 50.1 over 222 days -> MDE **9.42 bps/traded day**. The VAL test of
the veto itself (n = 94, sd 49.1) -> MDE **14.18 bps/day**: VAL could only have confirmed a bucket losing more
than ~14 bps a day, and the bucket came in at +1.4.

**Supported statement.** *No day-level separation of this sleeve's losers was detectable from causal
09:31-ET day features, on QQQ, at a 30-minute decision cadence, at 1x notional, over 2016-01->2024-06, at
0.5 bp/leg; the smallest bucket effect the VAL test could have seen is 14.2 bps/day (~ $8.5/day, ~ $180/month
at $60K on the 94 vetoed days), and the observed effect was +1.4 bps/day of the wrong sign.* "No edge exists"
is not claimed and is not supported.

**Disclosure on TEST.** No filtered TEST book was computed. The bookkeeping script `base.py` printed the
**unfiltered** TEST-window aggregate once while writing the day table (554 days, 313 traded, 3.97 bps/traded
day, 2.24 bps/calendar day). That is a baseline read, not a filter evaluation; nothing in this stage was
chosen using it.

---

## 1. Anatomy of the losing days — TRAIN (2016-01..2022-12)

Scripts `base.py` (day table) -> `anatomy.py`. Tables: `step1_worst30.csv`, `step1_best30.csv`,
`step1_loser_vs_winner.csv`, `step1_buckets.csv`, `step1_weeks.csv`, `step1_entry_minute.csv`.

**The book.** 1,044 traded days of 1,750; **43.1 % green**; mean +6.24 bps (t 2.91); +6,515 bps total;
SR 1.10, MDD 9.92 %; 84 months, 61.9 % green, worst month -405 bps; 357 weeks, 55.5 % green, worst week
-442 bps. Halves: H1 (2016-01..2019-06) +7.15 (t 2.57), H2 (2019-07..2022-12) +5.40 (t 1.67).

**Where the loss is.** 594 losing traded days sum -20,803 bps against +27,318 from 450 winners.

| cut | bps | share of all losses | share of all gains |
|---|---|---|---|
| worst 5 % of traded days (n = 52) | -6,881 | 33 % | 25 % |
| worst 10 % (n = 104) | -10,236 | 49 % | 37 % |
| worst 20 % (n = 209) | -14,678 | 71 % | 54 % |
| best 5 % (n = 52) | +10,140 | — | 37 % |
| best 10 % (n = 104) | +15,769 | — | 58 % |

Both tails are fat and they nearly cancel: removing the top 5 % of days gives **-3.79 bps/day**, removing the
worst 5 % gives **+13.50**, capping at +/-100 bps gives **+2.87**. This is a symmetric breakout book, not a
lottery ticket *in sample* — but the whole TRAIN profit is the top decile minus the bottom decile.

**Exit mix (trade level, TRAIN):** 1,044 stop exits at **-19.12 bps** each (-19,960 total) against 517
end-of-day exits at **+51.21** (+26,475). The loss is entirely the stop leg — the VWAP/current-band trailing
stop whipsawing. Long 817 trades +4.13 bps, short 744 +4.22 — no side asymmetry.

### The worst 30 TRAIN days, annotated (`step1_worst30.csv` has all 30 rows and all columns)

`noise_mult` = the day's high-low range / the band span at 10:00. `eff_ratio` = |net move| / path length after
the first crossing (low = chop). `side x oc` = the first trade's side x the day's open->close return, i.e. did
the day finish in the direction we were put into.

| date | dow | bps | ntr | open->close % | gap % | range % | band % | noise x | crossings | 1st cross | trend after % | eff | prev ret % | prev range % | rv20 % | side x oc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2020-03-25 | Wed | -321 | 2 | -1.25 | +0.34 | 5.17 | 2.71 | 1.91 | 1 | 14:00 | -2.90 | 0.15 | +7.73 | 3.27 | 5.84 | -1.25 |
| 2020-03-19 | Thu | -319 | 3 | +1.14 | -0.39 | 6.75 | 2.85 | 2.37 | 1 | 10:30 | -0.84 | 0.01 | -3.07 | 7.24 | 5.48 | +1.14 |
| 2020-03-23 | Mon | -242 | 2 | -0.20 | +0.13 | 5.43 | 2.57 | 2.11 | 1 | 11:30 | -2.33 | 0.05 | -3.94 | 7.03 | 5.53 | +0.20 |
| 2021-02-26 | Fri | -242 | 3 | -0.47 | +1.00 | 2.61 | 1.49 | 1.75 | 2 | 10:30 | -0.92 | 0.04 | -3.52 | 3.98 | 1.38 | +0.47 |
| 2022-12-01 | Thu | -225 | 2 | +0.02 | +0.09 | 1.66 | 0.76 | 2.17 | 2 | 10:00 | -0.33 | 0.02 | +4.59 | 4.80 | 2.45 | +0.02 |
| 2020-03-18 | Wed | -217 | 1 | +2.79 | -5.70 | 7.24 | 8.16 | 0.89 | 1 | 14:30 | -4.63 | 0.16 | +7.33 | 8.33 | 5.49 | -2.79 |
| 2022-04-27 | Wed | -211 | 2 | -0.16 | +0.10 | 2.48 | 1.25 | 1.98 | 1 | 10:00 | -1.38 | 0.05 | -3.86 | 3.30 | 1.80 | -0.16 |
| 2021-02-23 | Tue | -199 | 2 | +1.43 | -1.69 | 3.85 | 2.09 | 1.84 | 1 | 10:00 | -2.34 | 0.10 | -2.60 | 1.71 | 1.29 | -1.43 |
| 2016-02-03 | Wed | -184 | 2 | -1.10 | +0.63 | 2.87 | 1.73 | 1.65 | 1 | 10:30 | -1.46 | 0.06 | -2.13 | 2.05 | 1.85 | +1.10 |
| 2020-09-04 | Fri | -172 | 4 | -0.80 | -0.50 | 5.99 | 1.22 | 4.93 | 1 | 10:00 | -0.58 | 0.01 | -5.13 | 4.77 | 1.65 | +0.80 |

Four of the ten worst days are 2020-03-18..25 — the COVID crash week, where the band was 2.6-8.2 % wide and
the book was still whipsawed. They are NOT caught by any band-width filter: they are wide-band days.

**What the worst days have in common** (means, worst 30 vs best 30 vs all traded):

| feature | worst 30 | best 30 | all traded | losing days | winning days |
|---|---|---|---|---|---|
| open->close % | +0.26 | -1.43 | 0.00 | +0.03 | -0.04 |
| day range % | 3.18 | 4.27 | 1.74 | 1.49 | 2.07 |
| band span % | 1.99 | 1.71 | 1.18 | 1.12 | 1.27 |
| **noise multiple** | 1.83 | **3.00** | 1.65 | 1.47 | 1.90 |
| **crossings** | **1.20** | **1.00** | 1.03 | 1.05 | 1.02 |
| first cross (min after 09:30) | 74 | 86 | 91 | 90 | 94 |
| **trend after 1st cross %** | **-1.05** | **+2.39** | +0.06 | -0.36 | +0.62 |
| **efficiency ratio** | **0.052** | **0.132** | 0.067 | 0.055 | 0.084 |
| prev-day return % | **-1.20** | -0.09 | -0.05 | -0.09 | +0.01 |
| prev-day range % | 3.71 | 2.73 | 1.58 | 1.56 | 1.61 |
| rv20 % (daily) | 2.31 | 1.86 | 1.24 | 1.21 | 1.28 |
| trades that day | **2.30** | **1.13** | 1.50 | 1.59 | 1.37 |
| side x open->close % | +0.27 | **+3.59** | +0.79 | +0.33 | +1.40 |

Read: **the losing day is a day that crossed the band and then did not go** — noise multiple 1.8 against 3.0
on winners, efficiency ratio 0.05 against 0.13, `trend_after_cross` -1.05 % against +2.39 %. `corr(day bps,
trend_after_cross) = +0.71`, `corr(day bps, eff_ratio) = +0.36`, `corr(day bps, n_cross) = -0.14`. All three
are **post-hoc** descriptions of the day; none is knowable at 09:31, which is the whole problem.

By day shape: days with efficiency ratio < 0.15 (chop, n = 972) average **+2.19 bps**; 0.15-0.35 (n = 71)
average **+61.46**. The book is one long bet that the day after a crossing trends.

**Day of week** (traded days): Mon +2.93, Tue +4.60, Wed +8.98, Thu +2.86, Fri +12.23. No mechanism offered;
not carried forward.

**Calendar events.** No FOMC/CPI calendar exists on this node for 2016-2022 (the only dated news file,
`research/ignition_news/articles_25H1.csv`, covers 2025 H1). **FOMC/CPI day analysis was skipped** and is
stated here as skipped rather than approximated. **VIX is not on disk either** (`etf_1min.db` holds
SPY/QQQ/TQQQ/DIA/IWM/SOXL/SQQQ/UVXY only); the volatility level is carried by the causal 20-day realised vol
of QQQ (`rv20_pct`) and by the paper's own `sig14`, both computed from closes ending yesterday.

### Losing weeks (357 TRAIN weeks, `step1_weeks.csv`)

Terciles of week features vs the week's summed bps (green % in brackets):

| week feature | T1 (low) | T2 | T3 (high) |
|---|---|---|---|
| **abs(sum of daily open->close)** | **-4.0 (41 %)** | +2.7 (52 %) | **+56.0 (73 %)** |
| sum of daily open->close (signed) | +34.9 (55 %) | -2.3 (45 %) | +22.2 (66 %) |
| mean rv20 | -0.3 (49 %) | +13.5 (56 %) | +41.1 (61 %) |
| mean band span | +2.9 (45 %) | +15.6 (60 %) | +36.3 (61 %) |
| mean noise multiple | -2.4 (51 %) | -0.5 (49 %) | +57.6 (66 %) |
| mean crossings/day | +18.1 (57 %) | +20.7 (53 %) | +10.5 (62 %) |
| mean day range | -7.7 (45 %) | +11.2 (57 %) | +51.2 (65 %) |

The losing week is the **quiet, directionless** week (bottom tercile of absolute weekly movement: -4.0 bps,
41 % green), not the down week — the signed-return terciles are +34.9 / -2.3 / +22.2, i.e. the book is
symmetric in direction and asymmetric in movement. This is the mechanism the filters below try to make causal.

**Day-to-day dependence is the wrong way round for a "stand down" rule.** lag-1 correlation of traded-day bps
is **-0.13**; after a losing traded day the next traded day is **+11.00 bps**, after a winning one **-0.04**;
after two consecutive losing traded days **+14.41**. A "skip after N losing days" filter removes the best
days. Refuted, not run.

---

## 2. Candidate filters (pre-registered in `PREREG.md` before the grid)

Every filter is computable at **09:31 ET** from bars strictly before the first decision (10:00), or is a
shape change in the decision loop. The cuts are coarse (quartile boundaries and round numbers) and were never
refined.

| id | rule | mechanism | grid |
|---|---|---|---|
| F1 | skip the day if the band span `(UB-LB)/open` at 10:00 < c | `sigma` is a 14-day trailing mean of intraday excursion; after a quiet stretch the band is narrow relative to today's real range, so the close crosses it on noise and the VWAP/band stop sits immediately under the entry | 0.5 / 0.6 / 0.67 / 0.7 / 0.8 % |
| F2 | skip the day if `abs(gap) / band span` < c | the band is anchored `max(open, prev_close)` / `min(...)`; with a small gap the anchors coincide — a symmetric band around a market with no overnight impulse, nothing to continue | 0.15 / 0.20 / 0.25 / 0.30 |
| F3 | cap the entries taken in a day (trade only the first crossing) | re-entries are entries into a market that has already proven it chops across the band (losing days 1.59 entries vs 1.37 on winners; worst 30 = 2.30) | max 1, max 2 |
| F4 | side gate on the trailing trend: long only above the MA, short only below | Cooper-Gutierrez-Hameed (2004 JF) — breakout pays in the direction of market state | MA20, MA5 |
| F5 | new entries only at checks <= 12:00 / <= 13:00 | the "trade the morning" prior, and a direct test of the brief's candidate | k <= 150, k <= 210 |
| F6 | skip if causal 20-day realised vol < c | vol-regime gate | 0.70 %, 1.00 % |
| F7 | skip if the prior 5 sessions' summed abs return < c | causal proxy for the weekly finding | bottom tercile (2.93 %), bottom quartile (2.47 %) |

Refuted before the grid by step 1 and therefore not run as filters (counted): "skip after an overnight gap
**larger** than the band" (the largest `gap/band` quartile is the **best**, +10.77 bps) and "stand down after
N consecutive losing days" (above).

### The TRAIN grid (`step2_grid.csv`, `step2b_stacks.csv`)

Decision rule, pre-committed: a filter is adopted only if the **vetoed** bucket is negative in **both** TRAIN
halves *and* the kept book improves in both halves.

**Vetoed-bucket era test** — this is what killed five of the seven families:

| cut | vetoed n | mean | H1 | H2 | passes? |
|---|---|---|---|---|---|
| F1 band >= 0.5 % | 108 | -1.21 | -2.87 | **+9.89** | no |
| F1 band >= 0.6 % | 192 | -1.23 | -3.22 | **+5.46** | no |
| F1 band >= 0.67 % | 259 | -1.54 | -2.15 | **+0.09** | no |
| F1 band >= 0.7 % | 291 | -1.67 | -2.53 | **+0.58** | no |
| **F1 band >= 0.8 %** | 371 | -1.50 | -1.34 | -1.85 | **yes** |
| F2 gap/band >= 0.15 | 193 | +0.86 | +1.81 | -0.12 | no |
| **F2 gap/band >= 0.20** | 259 | -1.04 | -0.20 | -1.86 | **yes** |
| F2 gap/band >= 0.25 | 344 | +0.58 | +1.18 | +0.04 | no |
| F2 gap/band >= 0.30 | 413 | +2.73 | +3.91 | +1.64 | no |
| F6 rv20 >= 0.70 % | 265 | +2.65 | +3.59 | -0.02 | no |
| F6 rv20 >= 1.00 % | 521 | +2.11 | +2.24 | +1.79 | no |
| F7 absmove5 >= 2.93 % | 352 | +0.65 | +0.61 | +0.76 | no |
| F7 absmove5 >= 2.47 % | 259 | +1.84 | +1.13 | +4.15 | no |

**Shape filters, all rejected on TRAIN** (book mean, TRAIN / H1 / H2 against base 6.24 / 7.15 / 5.40):

| rule | TRAIN | H1 | H2 | verdict |
|---|---|---|---|---|
| F3 max 1 entry/day | 5.15 | 6.86 | 3.57 | worse everywhere — **reverse causality**: the chop produces the re-entries, not the other way round |
| F3 max 2 entries/day | 5.94 | 6.64 | 5.30 | worse |
| F4 MA20 side gate | 8.26 | **6.21** | 10.09 | fails H1 |
| F4 MA5 side gate | 3.30 | 8.95 | **-1.82** | fails H2 |
| F5 entries <= 12:00 | 4.72 | 7.80 | 1.76 | worse everywhere; refutes the brief's "only crossings before 12:00" |
| F5 entries <= 13:00 | 4.40 | 6.52 | 2.38 | worse |

F3's rejection is the most useful single line in the stage: the step-1 association (losing days have 1.59
entries, winners 1.37, worst-30 days 2.30) is entirely explained by the day, not by the re-entry.

---

## 3. The stack on TRAIN, before and after (`step3_train_robust.csv`)

Frozen stack: **F1 (band span >= 0.8 %) AND F2 (abs(gap) / band >= 0.20)** — two filters, both day-level.

| book | split | traded | bps/traded day | t | SR | MDD % | day green % | months green % | worst month | **ex-top-5 % days** | **capped +1 %/day** | sum bps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE | TRAIN | 1,044 | +6.24 | 2.91 | 1.10 | 9.92 | 43.1 | 61.9 | -405 | **-3.79** | +1.01 | +6,515 |
| BASE | TRAIN-H1 | 501 | +7.15 | 2.57 | 1.38 | 8.22 | 40.7 | 57.1 | -340 | -2.85 | +2.38 | +3,581 |
| BASE | TRAIN-H2 | 543 | +5.40 | 1.67 | 0.89 | 9.92 | 45.3 | 66.7 | -405 | -4.72 | -0.25 | +2,935 |
| F1 only | TRAIN | 673 | +10.51 | 3.39 | 1.28 | 10.04 | 45.8 | 60.7 | -405 | -0.71 | +2.92 | +7,072 |
| F2 only | TRAIN | 785 | +8.64 | 3.58 | 1.35 | 5.54 | 45.0 | 61.9 | -371 | -1.17 | +3.50 | +6,785 |
| **STACK** | TRAIN | 585 | **+11.04** | **3.55** | **1.34** | **4.74** | 46.2 | 56.0 | -371 | **+0.32** | **+4.35** | +6,456 |
| **STACK** | TRAIN-H1 | 230 | +15.69 | 3.11 | 1.66 | 4.60 | 45.7 | 57.1 | -198 | +3.73 | +7.80 | +3,608 |
| **STACK** | TRAIN-H2 | 355 | +8.02 | 2.03 | 1.08 | 4.74 | 46.5 | 54.8 | -371 | -1.88 | +2.13 | +2,848 |

Weeks: base 357, 55.5 % green, worst -442; stack 293, 56.3 % green, worst -442 (the worst week survives both
filters — it is not the kind of week they address).

On TRAIN this looks like a good filter by every ratio: t up, Sharpe up, drawdown halved, worst month better,
both halves improved, and — uniquely in this program — the **tail tests turn positive**. The sum-bps column is
the warning that was under-weighted at freeze time: **+6,456 vs +6,515**. The stack makes no money; it
makes a better ratio out of less money.

---

## 4. VAL — read once, with the frozen stack (`step4_val.csv`)

| book | traded / days | bps/traded day | t | SR | MDD % | day green % | weeks | week green % | worst week | sum bps |
|---|---|---|---|---|---|---|---|---|---|---|
| BASE | 222 / 374 | +7.82 | 2.33 | 1.90 | 2.8 | 48.6 | 76 | 59.2 | -194 | +1,736 |
| F1 only | 140 / 374 | +10.80 | 2.47 | 2.01 | 2.9 | 48.6 | 64 | 53.1 | -95 | +1,512 |
| F2 only | 163 / 374 | +10.09 | 2.50 | 2.03 | 2.8 | 47.2 | 70 | 57.1 | -194 | +1,645 |
| **STACK** | 128 / 374 | **+12.54** | 2.81 | 2.27 | 2.9 | 49.2 | 60 | 55.0 | -95 | +1,605 |

**Vetoed buckets on VAL** — the pre-committed pass criterion:

| bucket | n | mean bps/day | sum bps |
|---|---|---|---|
| F1 dropped (band < 0.8 %) | 82 | **+2.74** | +224 |
| F2 dropped (gap/band < 0.20) | 59 | **+1.54** | +91 |
| **STACK dropped (either)** | 94 | **+1.39** | +131 |

**Verdict against `PREREG.md` rule 3:** mean improves YES · filtered book > 0 with >= 55 % weeks green YES
(55.0 %, exactly at the bar) · **vetoed days negative on VAL NO (+1.39)**. The conjunction fails. **The stack
is not adopted and TEST is not read.**

### The 2x2, TRAIN vs VAL — why it failed

mean bps/day (n) per cell of the two conditions:

| | TRAIN | VAL |
|---|---|---|
| band >= 0.8 **and** gap/band >= 0.20 | +11.04 (585) | +12.54 (128) |
| band >= 0.8 **only** | +7.00 (88) | **-7.81 (12)** |
| gap/band >= 0.20 **only** | +1.65 (200) | +1.13 (35) |
| **neither** | **-5.18 (171)** | **+3.93 (47)** |

The only genuinely negative TRAIN cell is "neither" (-5.18 over 171 days). On VAL it is **+3.93 over 47 days**
— the sign flips. The "band >= 0.8 only" cell that is negative on VAL has 12 observations. At a VAL bucket sd
of ~49 bps, a 47-day cell has an 80 %-power MDE of about **20 bps/day**; a 12-day cell about **40**. VAL
cannot resolve any of these cells; it can only refuse to confirm them, which it did.

### Permutation and multiplicity

Random subsets of the same size drawn from the TRAIN traded days (5,000 draws, seed 20260917) reproduce the
stack's mean lift (+4.80 bps over the all-day mean) with **p = 0.0118**. This stage looked at **30 distinct
rule variants** and **53 cell-instances** including the step-1 descriptive splits; Q had already looked at 36
variants of this same sleeve, so the sleeve's running total is **66 variants / 89 cell-instances**.
Sidak-adjusted alpha at 30 variants is **0.0017**, at 48 cell-instances **0.0011** — **p = 0.0118 clears
neither.** The TRAIN result was not significant after multiplicity even before VAL disagreed with it.

### One post-hoc diagnostic, reported and NOT adopted

Because only the "neither" cell is negative, the natural repair is an **AND-veto**: skip the day only when
`band < 0.8 % AND abs(gap)/band < 0.20`. It was not pre-registered. It is reported for the record and is
**not a rule**:

| AND-veto | traded | bps/day | t | SR | MDD % | sum bps |
|---|---|---|---|---|---|---|
| TRAIN | 873 | +8.48 | 3.44 | 1.30 | 10.04 | **+7,401** |
| TRAIN-H1 | 390 | +10.05 | 3.00 | 1.61 | 6.06 | +3,920 |
| TRAIN-H2 | 483 | +7.21 | 2.03 | 1.08 | 10.04 | +3,481 |
| VAL | 175 | +8.86 | 2.24 | 1.83 | 2.8 | +1,551 |

It is the only variant that raises the **sum** on TRAIN (+7,401 vs +6,515). It **also fails the VAL criterion**:
its vetoed bucket on VAL is **+3.93 bps/day** (n = 47). So the repair fails the same test as the thing it
repairs, and there is no remaining untouched split in this window to test it on. Acting on it would be
exactly the move that produced the four false conclusions of 9/13->9/16.

---

## 5. Money line

Not computed on TEST, because TEST was not read. What the two read splits say, at $60K equity and 1x
notional (1 bps of calendar-day return = $6.00/day ~ $126/month at 21 sessions):

| | TRAIN (84 months) | VAL (18 months) |
|---|---|---|
| unfiltered | 3.72 bps/cal day ~ **$469/month** | 4.64 ~ **$584/month** |
| frozen stack | 3.69 ~ **$465/month** | 4.29 ~ **$541/month** |
| stack, the days it skips | +0.13 bps/day x 459 days ~ **+$35 forgone** | +1.39 x 94 ~ **+$79 forgone** |

Levered forms scale linearly from `Q/REPORT.md` section 4 (QQQ 3x notional = 3x, intraday only so zero margin
interest; TQQQ 1x ~ 3.4x with its own band). None of that changes the conclusion: the filter does not add
dollars in either split, so no leverage line is offered for it.

---

## 6. Cell count

| where | rule variants | cell-instances |
|---|---|---|
| step 1 anatomy: 15 causal features x quartiles, 7 weekly terciles, day-of-week, shape/autocorrelation/entry-minute slices | 0 (descriptive) | 23 |
| step 2 grid (`step2_grid.csv`): base + F1x5 + F2x4 + F3x2 + F4x2 + F5x2 + F6x2 + F7x2 + 8 stacks | 28 | 28 |
| step 2b: the rule-valid stack F1(0.8)+F2(0.20) | 1 | 1 |
| step 5: the post-hoc AND-veto | 1 | 1 |
| **this stage** | **30** | **53** |
| Q, the same sleeve, prior | 36 | 36 |
| **sleeve running total** | **66** | **89** |

Reported but not counted as selection cells (slices of one rule, not variants of it): the 3 TRAIN robustness
splits x 4 books, the 4 VAL books, the tail treatments, the 2x2 decomposition, the monthly/weekly tables.

---

## 7. Smallest visible effect (the power statement, in full)

| test | n | sd (bps/day) | MDE @ 80 % power | observed |
|---|---|---|---|---|
| TRAIN book, traded days | 1,044 | 69.3 | **6.01** | +6.24 |
| TRAIN book, calendar days | 1,750 | 53.6 | 3.59 | +3.72 |
| TRAIN vetoed bucket | 459 | 60.6 | 7.92 | +0.13 |
| VAL book, traded days | 222 | 50.1 | **9.42** | +7.82 |
| VAL book, calendar days | 374 | 38.8 | 5.62 | +4.64 |
| **VAL vetoed bucket** | 94 | 49.1 | **14.18** | **+1.39** |

The unfiltered TRAIN edge (6.24) sits on its own detection floor (6.01). The VAL veto test could only have
confirmed a bucket losing more than ~14 bps/day. Neither split is large enough to resolve a day filter worth
a few bps — which is the honest reason this stage ends where it does, and the reason a larger-N version of
the same question (more index ETFs, or a longer history than `etf_1min.db`'s 2016 start) is the only way to
answer it rather than re-cutting these 2,124 days.

---

## 8. Files

| file | what |
|---|---|
| `PREREG.md` | the pre-registration and the frozen stack, written before VAL |
| `base.py` | anchor to `Q/REPORT.md` + the day/trade tables |
| `anatomy.py` | step 1 (TRAIN loser anatomy) |
| `filters.py` | `simulate2` (zsim + day mask / entry cap / side gate / last-check) and the reporting helpers |
| `step2_train.py`, `step2b_train.py` | the filter grid and the era test |
| `step3_robust.py` | TRAIN robustness of the stack |
| `step4_val.py` | VAL, read once |
| `step5_diag.py` | power, the numerator/denominator decomposition, the post-hoc cell, the permutation |
| **`final_book_days.csv`** | **the day-level book: date, split, ntr, bps, r1x and every causal feature, with `f1_band_w_ge_0.8`, `f2_gap_over_band_ge_0.20`, `frozen_stack_keep` flags — 2,678 rows, enough for an independent rebuild of every number above** |
| `day_book.csv`, `trades.csv` | the unfiltered day book (with the post-hoc anatomy columns) and the 2,362 trades |
| `step1_*.csv`, `step2_grid.csv`, `step2b_stacks.csv`, `step3_train_robust.csv`, `step4_val.csv` | the tables |

## 9. The filter rules in prose (for an independent rebuild)

At 09:31 ET, after the 09:30 bar has closed, compute for QQQ:

1. `sigma[k]` for k = 30 (the 10:00 clock minute) as the mean over the previous 14 sessions (>= 10 present) of
   `abs(close(k)/open_0930 - 1)`, using only sessions strictly before today.
2. `UB = max(open_0930, prev_session_1559_close) * (1 + sigma[30])` and
   `LB = min(open_0930, prev_session_1559_close) * (1 - sigma[30])`.
3. **`band_w` = 100 * (UB - LB) / open_0930** — the band span as a percent of the open.
4. **`gap` = 100 * (open_0930 / prev_session_1559_close - 1)**; **`gap_over_band` = abs(gap) / band_w**.

**Filter F1** — trade the day only if `band_w >= 0.8`. **Filter F2** — trade the day only if
`gap_over_band >= 0.20`. The frozen stack required both. Neither is adopted: both failed VAL, where the days
they skip earned +1.39 bps/day.

---

## 10. What this stage did NOT establish

- **No independent blind rebuild.** `zsim.py` is Q's re-implementation; `simulate2` is a copy of its loop with
  three switches. `final_book_days.csv` exists so a second agent can rebuild every number from section 9's prose.
- **No quote data.** 0.5 bp/leg is an assumption (QQQ's quoted half spread is ~0.09 bp, so it is conservative
  by ~6x); market-order slippage at the decision minutes is unmeasured, as `Q/REPORT.md` section 6 says.
- **No FOMC/CPI split** — the calendar is not on this node (section 1).
- **TEST is intact for this sleeve** apart from the one unfiltered aggregate disclosed in section 0. Any future
  H attempt on QQQ has a clean TEST window only if it does not re-cut TRAIN/VAL first.
- **Only QQQ.** SPY and TQQQ were not filtered here; SPY's OOS book is already negative (`Q/REPORT.md`
  section 3) and TQQQ is the same signal on a different vehicle.
