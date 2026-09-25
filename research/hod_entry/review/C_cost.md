# LENS C — Is the +0.330R TEST result cost-realistic? (adversarial review)

**Claim under review:** cell 1,427's resting buy-stop-limit (trigger = level + $0.01, limit = level x
1.0015), filled at the prevailing SIP NBBO ask at the first consolidated print >= trigger, B0's stop /
2R target / 15:55 exit, cost = half the NBBO spread at the fill instant on entry + B0's per-signal
exit cost rule, nets **+0.330R TEST** (n=972 fills, 2026-06-01..09-04). Source: `sip_rebuild.py`,
`sip_rebuild_test.csv`, `research/hod_exit_lab/PREREG.md` (B0 cost rule), `research/hod_exit_lab/walker.py`
(`b0_fill`, `simulate_b0`).

## 1. What the cost model actually charges (read from code, not prose)

`research/hod_exit_lab/walker.py::b0_fill` (lines 192-213): a bar whose low <= stop fills **exactly at
the stop price** (no incremental slippage) unless the bar's *open* already gapped through the stop, in
which case it fills at that open. `simulate_b0`/`cost_net` then charges, **identically for every exit
reason** (stop / target / eod):
```
cost_R = 2 x (0.5 x spread_mean)/R + 2bp x (entry+exit_price)/R      # walker.py:130-136, 237-241
```
Confirmed on the TEST fills themselves (`sip_rebuild_test.csv`, n=972): mean `cost_R` by `why` —
stop 0.174R (n=422), target 0.173R (n=424), eod 0.119R (n=119). **The model charges a stop-loss exit
the same 17.4c-on-the-dollar-of-R as a target exit that hits calmly with price moving in the trader's
favor.** That is the crux of this lens: a HOD-break stop fires exactly when the name is failing —
often the same minute other momentum longs are also bailing — which is a structurally worse liquidity
moment than a target hit or a scheduled 15:55 flat, and the flat model has no term for that.

Cross-reference: the OLDER cost convention this study explicitly rejected, `research/bf_zero/causal_filter/cells.py`
(`RATIO = {'stop': 0.875, 'eod': 0.412, 'target': 0.0}`), already encoded the opposite prior — stops pay
0.875x an extra half-spread, targets pay 0 extra — precisely because stop exits are the expensive side.
B0 discarded that asymmetry for a flat charge (walker.py:220, "intentionally NOT cells.py's why-weighted
ratio cost"). No evidence in this programme justifies that simplification; it was a convenience for the
exit-lab's paired-cell harness, not a cost-realism finding.

## 2. Live dry-run evidence: does not exist yet for this mechanism

The task asks to check "the live HOD dry run's logs... for the last 9 sessions." I did — with a
material finding: **there is no live evidence for the resting-stop-limit mechanism under review.**
`git log` shows it was only added today (commit 841089e, "HOD-break resting stop-limit entry... for the
10-session dry run"); `trading/hod_break_engine.py:218` points the ledger at
`logs/hod_dry_entry_ledger.csv`, which **does not exist** (`find` confirms). The prior dry-run mode
(`logs/session_archive/2026-09-2{1,2,3,4}.log`, `[HOD]` lines) produced **zero fills** over the last 4
sessions — every candidate hit `"stop within 1.0% of the ask — skip"` (a pre-existing R-too-small gate
unrelated to E1). 2026-09-22, the busiest of the recent days (90 `[HOD]` lines), admitted exactly one
candidate (JAGX) and generated no fill/exit line at all. So the claim's premise that live logs show
"would-be entries and exits" is not currently true — the dry run has not run the mechanism being
reviewed for even one session. **This is a hole in the evidence base, not a pass.** Separately,
`hod_break_engine.py:680-698` documents that even once live, the engine resolves a trigger cross using
"the bar's high and the current quote" polled per-bar, and logs this WARNING itself: `"no live print
stream — resolving the cross... (not tape-accurate)"`. The cost model assumes a tape print at the fill
instant; the live engine does not have one. That gap is a fill-mechanism issue (lens B's territory) but
it also undermines the cost claim, because the "half-spread at the fill instant" input the live engine
will actually use is a polled quote, not the print-level SIP quote `sip_rebuild.py` used to build the
0.167R TEST cost figure.

## 3. Realized stop-loss slippage on the live account (BF/ORB, same $20+ bracket)

Since the HOD dry run has no fills, I used `data/trades.db` (BF + ORB, real fills, same account) as the
closest available evidence of what stop-type exits actually cost on this system, restricted first to
entry_price >= $20 (the task's bracket, and it matches: 970/972 of cell 1,427's TEST fills are already
priced >= $20, median $33.84, mean R/price = 2.2%, matching the PREREG's "~2% of price" stop distance).

`SELECT ... WHERE exit_reason IN ('stop_loss','trail_stop','stop_loss_market_fallback') AND strategy IN
('bull_flag','orb') AND entry_price>=20` → **n=10** (small; strategy tags skew low-price — mean entry
price across ALL stop_loss exits, any price, is $12.1, so $20+ stops are a minority of this system's
trade book). `exit_price - stop_loss_price` as bps of the stop price:

| symbol | date | strategy | reason | slip vs stop (bps) |
|---|---|---|---|---|
| CRCA | 2026-09-22 | orb | stop_loss | **-79.4** |
| ARQQ | 2026-06-17 | bull_flag | stop_loss_market_fallback (quoted spread $0.25 = 108bps full) | **-78.9** |
| YANG | 2026-05-21 | orb | stop_loss | -2.5 |
| OSCR | 2026-06-10 | orb | stop_loss | -3.6 |
| KOLD | 2026-06-17 | orb | stop_loss | -2.1 |
| MUU/PLTZ/BTC/BKSY/HPQ | — | orb | stop_loss/trail | +6 to +291 (trailing stop moved up before the hit — not comparable to a fixed-stop slip) |

Widening to **all** BF/ORB `stop_loss`/`trail_stop`/`stop_loss_market_fallback` exits, any price (n=86,
larger sample, same live system, same order machinery): median slip = 0bps, but **48.8% filled worse
than the recorded stop price**, averaging **-47bps** among those, **p10 = -79bps**. (Mean is +84bps,
pulled positive by trailing-stop rows where the recorded "stop" had already moved in the trader's
favor before the hit — not a clean slippage measure; median and the worse-than-stop tail are the
honest reads here.) **Caveat on power:** n=10 in the exact $20+/HOD-relevant bracket is too small to
fit a replacement cost model; n=86 all-price is still thin and mixes trailing with fixed stops. I am
reporting this as directional evidence of tail risk, not a calibrated slippage table (per the "no
banded table" rule — a single point estimate here would repeat the mistake the band table made on BF).

**Comparison to the model's allowance.** The model's flat 2bp/side slip term, in R, is a small fraction
of the 0.167R mean total cost (most of that 0.167R is the half-spread term, not the 2bp add-on). Even
one leg of the observed -47 to -79bps real slippage, converted at the TEST fills' own R/price = 2.2%,
is **0.21-0.36R** — 1.2-2.2x the model's ENTIRE two-leg cost allowance, on ONE exit leg alone. That is
the central finding of this lens: the exit-cost model's blind spot (flat cost, no stop-specific term)
is not a rounding error against this system's own realized fills — it is the same order of magnitude
as the whole modeled cost.

## 4. Decomposition: gross vs. cost, and is the lift a cost artifact (per the OFI precedent)

`research/hod_ofi/REPORT.md`'s finding was that OFI's entire net lift was cost (gross flat, cost fully
explained kept-vs-dropped). Same test here, using `sip_rebuild_test.csv` (E1 TEST fills) against
`research/hod_exit_lab/b0_trades.csv` (whole B0 population, 2025-01-10..2026-09-04, n=15,656 — the
"enter every signal at next-open" baseline, includes the TEST window):

| cohort | n | mean raw_R (gross) | mean cost_R | mean net_R |
|---|---|---|---|---|
| E1 TEST fills | 972 | **+0.497** | 0.167 | +0.330 |
| Whole B0 population (all signals, all dates) | 15,656 | +0.026 | 0.295 | -0.270 |
| Difference (fills minus population) | — | **+0.471** | -0.128 | +0.600 |

Of the +0.600R gap between E1-fills and the raw population, **+0.471R (78%) is gross, +0.128R (21%) is
lower cost.** Unlike OFI (100% cost, 0% gross), this is NOT purely a cost artifact — most of the lift is
real directional edge in the fill-selected population. But the 21% cost component is real and worth
naming: E1's fill condition (ask <= level x 1.0015) mechanically requires a tight spread at the trigger
instant, so it selects lower-cost fills by construction, the same mechanism that drove the OFI result.
**Within the TEST fill cohort itself**, correlation(raw_R, cost_R) = **-0.018** (essentially zero, n=972)
— so among the trades E1 actually takes, the edge is not concentrated in the low-cost tail; the 21%
population-level cost-selection effect does not further compound once you're already inside the fill
cohort. This bounds the artifact: real, non-trivial (>1/5 of the lift), but not the whole story.

## 5. Stress test: realistic exit cost applied to the TEST fills

I cannot rebuild a calibrated per-trade slippage model from n=10-86 live rows (would repeat the banded-
cost mistake). As a bounded stress test instead: apply the live system's own worse-than-stop tail
(-47bps, the mean when a real stop misses, n=42/86) as an ADDITIONAL exit-side haircut to the 422 `why
== 'stop'` TEST fills only (43.4% of the 972), at the 43.4% x 48.8% share this system's own data says
actually slips:
```
haircut ≈ 0.434 (stop share of fills) x 0.488 (share that slip) x (0.0047/0.02197 R) ≈ 0.046R
```
Stress-tested TEST mean net_R: **0.330 - 0.046 ≈ +0.284R** — still clears the +0.10R pass bar and the
t >= 2 bar is unaffected by a mean shift this size relative to the reported SE (day-clustered t was 2.12
on the unadjusted number; a 14% mean cut with unchanged day-clustered variance does not flip the sign).
This is illustrative, not a replacement number — it is built on 42 live rows, not the population — but
it shows the claim's TEST pass is not knife-edge to a realistic stop-cost adjustment of this size. A
larger adjustment (e.g., the p10 tail of -79bps applied to all stop fills) would cut closer to
0.330 - 0.434*(0.0079/0.02197) ≈ +0.174R — still a pass, but a materially thinner one.

## Verdict

**Threat level: MODERATE, not fatal.** Two real defects: (1) the cost model charges stop, target and
eod exits identically, which this system's own live fills say is optimistic specifically for stops —
the model has zero mechanism for the fact that a HOD-break stop fires into the worst liquidity moment
of the trade; (2) the "live dry-run last 9 sessions" evidence the task assumed exists does not — the
resting-stop-limit mechanism has produced zero live rows (deployed today, ledger file absent), so cost
realism for THIS mechanism is still untested in production, only reconstructed from historical tape.
Neither defect reverses the TEST pass under a bounded stress test (+0.284R at a plausible haircut,
+0.174R at a severe one, both > the +0.10R bar), and the gross/cost decomposition shows the lift is
78% gross, not a cost artifact like OFI was. But the PASS should not be read as "cost-validated" — it is
"cost-plausible under this population's own spread data, unconfirmed by any live stop-fill of this
exact mechanism." Recommend: do not skip the 10-session dry run before sizing up: it is the only source
that will show REAL fill-instant slippage on REAL resting stop-limit orders, not the reconstructed-tape
number this review had to substitute for it.

**Every number's provenance:** cost formula — `research/hod_exit_lab/walker.py:130-136,192-213,216-247`.
Stop/target/eod cost_R by `why` — computed from `research/hod_entry/sip_rebuild_test.csv` (972 fill rows)
in this review, not previously reported. cells.py ratio cost — `research/bf_zero/causal_filter/cells.py:15,29,66`.
Dry-run log absence — `logs/session_archive/2026-09-{21,22,23,24}.log`, `git log -1 841089e`,
`trading/hod_break_engine.py:218,680-698`; ledger file checked with `find`, absent. BF/ORB stop slippage —
`data/trades.db` (read-only), `trades` table, `exit_reason`/`stop_loss_price`/`exit_price`/`entry_price`/
`strategy` columns, queried live in this review (n=10 at entry_price>=20, n=86 all prices). Population
B0 gross/cost — `research/hod_exit_lab/b0_trades.csv` (15,656 rows, computed in this review). TEST
fill gross/cost and price bracket (970/972 >= $20, R/price 2.2%) — `sip_rebuild_test.csv`, computed in
this review. OFI precedent — `research/hod_ofi/REPORT.md:150-163`.
