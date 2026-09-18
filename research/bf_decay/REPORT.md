# What changed in the bull-flag book between 2025 and 2026?

2026-09-18. Diagnosis only — this stage proposes nothing and ships nothing.
Source: the honest regen-7 cache `data/bull_flag_cache_causal_full_20260905.csv`
(896 rows, 886 in 2025-01-01 → 2026-08-31), **its own exits**, never a resim
(BT_STATUS_20260918 §2a: the resim path is −$10.3K / 7.4% unfaithful, so every
number here is immune to those seven defects by construction).

Scripts: `decay_attrib.py` (layer chain + gate separation), `decay_part2.py`
(stack reconciliation, outcome side, power), `decay_part3.py` (population drift,
R cross-check, regime). Outputs: `layer_chain.csv`, `gate_separation.csv`,
`stack_reconciliation.csv`, `population_drift.csv`, `p1_picks.csv`,
`asis_picks.csv`, `picked_book.csv`, `attrib.json`.

**Answer in one line: NOISE — with a named population drift underneath it that
the shipped stack already gates, and two measurement artifacts that manufacture
part of the apparent gap.**

---

## 0. The established facts, verified

| claim | source | verified? |
|---|---|---|
| raw detector has no edge in either year | CLAUDE.md / README §6 | **YES.** 2025 −0.010R (n=463, WR 42.3%); 2026 −0.077R (n=423, WR 40.4%). Welch t = −0.71. |
| stack picked +0.61R in 2025, +0.06R in 2026 | README §6 | **YES — for the AS-IS (pre-P1) stack.** Reproduced: +0.606R (n=57) → +0.028R (n=37). |
| …and that is the shipped stack | *implied by the pause* | **NO.** P1 (price ≤ $20 + pole ≥ 5% + VWAP gate) has been the config since 9/7. On the same cache P1 reads **+0.651R (39) → +0.469R (23)**. |
| today's run: 54 tr / $142,775 / 63% WR at `--risk 150` | owner-facing run | reproduced as the $2K book: 56 tr / $139,114 / 62.5% WR. (`--risk 150` on a cached book only moves the rail — BT_STATUS §1 caveat; the 54-trade variant is the −$750 rail, not an L0 re-size.) |
| Jul-26 −$1,856, Aug-26 −$4,272, Q3-26 −$6,128 on 3 trades | owner-facing run | **YES.** 3 picks, mean −0.588R, one-sample t vs 0 = **−1.06**. |

R is defined once: `R = pnl / (shares × (entry_price − stop_loss))`.
**Independent check leg 1:** this R reproduces the R column of
`research/bf_consistency/trade_features_regen7.csv` (built by a different script,
earlier) on all 631 matched rows, `max|diff| = 0.000000`.

---

## 1. Layer-by-layer attribution (R per pick, regen-7's own exits)

Filters applied in the live order, using the **same shared modules** live and BT
use (`trading/bf_universe_filter.py`, `trading/bf_vwap_gate.py`,
`trading/two_tier_filter.py`) and the live `config.yaml` values, read-only.

| # | layer | n 2025 | R 2025 | WR | n 2026 | R 2026 | WR | Δ R |
|---|---|---|---|---|---|---|---|---|
| L0 | raw detections | 463 | −0.010 | 42.3% | 423 | −0.077 | 40.4% | −0.067 |
| L1 | live universe rule (by name) | 386 | +0.047 | 43.5% | 244 | −0.102 | 40.6% | −0.148 |
| L2 | ADV20 ≥ 200K | 220 | +0.033 | 42.3% | 151 | −0.108 | 41.1% | −0.141 |
| L3 | entry ≤ $20 | 194 | +0.067 | 42.8% | 141 | −0.100 | 41.8% | −0.167 |
| L4 | pole ≥ 5% | 121 | **+0.287** | 48.8% | 85 | +0.006 | 44.7% | −0.281 |
| L5 | conviction ≥ 1.8 | 57 | +0.334 | 47.4% | 34 | +0.205 | 52.9% | −0.129 |
| L6 | pole_bars ≤ 3 | 57 | +0.334 | 47.4% | 34 | +0.205 | 52.9% | 0.000 |
| L7 | VWAP gate | 51 | +0.394 | 49.0% | 27 | +0.373 | 59.3% | **−0.021** |
| L8 | two-tier filter | 39 | **+0.651** | 56.4% | 23 | **+0.469** | 60.9% | −0.183 |

Read down the Δ column: the gap does **not** open at any fitted layer. It is
already −0.148R after L1 (a *name* rule, nothing fitted), peaks at −0.281 after
the pole rule, and then **narrows** through conviction, the VWAP gate and the
two-tier filter. The fitted stack in 2026 *adds* +0.57R over its own input
(−0.100 → +0.469); in 2025 it adds +0.58R (+0.067 → +0.651). **The selection
stack's lift is the same in both years.**

### 1b. Isolated gate separation — kept R minus rejected R, per year

This is the inversion test: a layer that still works keeps better trades than it
throws away. A negative `sep` means the layer is picking the *wrong* side.

| gate | sep 2025 | sep 2026 | Δ sep | verdict |
|---|---|---|---|---|
| entry ≤ $20 | +0.282 | +0.121 | −0.161 | still right side |
| pole ≥ 5% | +0.587 | +0.268 | −0.318 | still right side |
| conviction ≥ 1.8 | +0.089 | **+0.332** | **+0.243** | *better* in 2026 |
| pole_bars ≤ 3 | n/a (0 rejected) | n/a | — | inert |
| VWAP gate | +0.567 | **+0.815** | **+0.248** | *better* in 2026 |
| two-tier filter | +1.094 | +0.644 | −0.450 | still right side, large |
| MACD zone ≥ 1.5 (sizing) | **−0.340** | +0.072 | +0.412 | was wrong-signed in 2025 |
| conv ≥ 1.9 within the book | +0.512 | +0.124 | −0.388 | weak ranker both years |
| **WHOLE STACK picked vs rejected** | **+0.751** (t 2.56) | **+0.680** (t 2.19) | **−0.071** | **intact** |

**No layer inverted.** Not one gate has a negative `sep` in 2026. The whole-stack
separation — the only cell here with any power — is +0.751R in 2025 and +0.680R
in 2026, a difference of 0.07R against a standard error of ~0.43. The stack
separates winners from losers in 2026 essentially as well as it did in 2025.

The single genuinely wrong-signed cell in the table is **2025's** MACD zone
multiplier (−0.340): in 2025 the high-MACD trades the stack sizes *up* were
worse than the low-MACD ones it sizes down. That is a 2025 defect, not a 2026 one.

---

## 2. Did the POPULATION change? Yes — and in a named direction

| | 2025 | 2026 (8 mo) | note |
|---|---|---|---|
| detections | 463 over 12 mo = **38.6/mo** | 423 over 8 mo = **52.9/mo** | **more** setups, not fewer |
| distinct symbols / days | 348 / 204 | 297 / 148 | |
| entry price p50 | $7.71 | **$9.81** | drifted up |
| pole gain p50 | 6.22% | **5.46%** | drifted down |
| VWAP dist p50 | 4.28% | **3.71%** | drifted toward VWAP |
| ADV20 p50 | 295K | 376K | more liquid |
| conviction p50 | 1.50 | 1.50 | unchanged |
| entry minute p50 | 605 (10:05) | 607 | unchanged |
| SPY 20d vol (mean) | 15.8% | 13.4% | calmer |
| risk/share p50 | $0.270 | $0.300 | unchanged |

Share of raw detections falling in each of the three era-consistent bad buckets
(README §6 — the buckets P1 was built to gate):

| bucket | 2025 share | 2026 share | Δ pp | 2-prop z | R inside (25 / 26) |
|---|---|---|---|---|---|
| entry price > $20 | 11.9% | 16.1% | +4.2 | 1.80 | −0.31 / −0.17 |
| pole gain < 5% | 32.4% | **42.3%** | +9.9 | **3.05** | −0.18 / −0.20 |
| breakout at/below VWAP | 19.9% | 22.0% | +2.1 | 0.77 | −0.29 / −0.43 |
| **ANY of the three** | **49.7%** | **60.1%** | **+10.4** | **3.10** | −0.18 / −0.27 |

**This is the one significant, mechanically-interpretable change in the whole
diagnosis.** The 2026 detection pool contains ~10 percentage points more of the
material that loses in *both* years. It is not that the losing buckets got worse;
it is that there are more of them. That is why the **as-is** stack — which had no
defense against price, pole or VWAP — decayed, and why P1 — which gates exactly
these three — does not.

### 2b. Stack reconciliation: the quoted decay belongs to a retired stack

| stack | n25 | R25 | n26 | R26 | Δ R | t | MDE₈₀ |
|---|---|---|---|---|---|---|---|
| **AS-IS** (pre-P1: no price cap, pole ≥ 3, no gate) | 57 | +0.606 | 37 | **+0.028** | −0.579 | −1.86 | 0.915 |
| + price cap only | 52 | +0.572 | 33 | +0.163 | −0.409 | −1.25 | 0.957 |
| + pole ≥ 5% only | 49 | +0.617 | 34 | +0.098 | −0.520 | −1.53 | 0.991 |
| + VWAP gate only | 49 | +0.733 | 29 | +0.180 | −0.553 | −1.57 | 1.042 |
| **P1 SHIPPED** (all three) | 39 | +0.651 | 23 | **+0.469** | **−0.183** | **−0.46** | 1.182 |

Each knob alone recovers a slice of 2026; together they recover most of it.
CLAUDE.md's "+0.61R → +0.06R" is verified and is **the AS-IS row**. It is the
motivation for P1, not a property of P1. Quoting it as the current state of the
book is a stale-stack error.

---

## 3. Entries or exits?

Same simulator, same unified trail spec (`trading/bf_trail.py`), same cache, both
years — so any exit-machinery change would show as a mix or magnitude shift on
the **raw pool** (n = 886, MDE₈₀ = 0.263R, the only high-power cell available).

| | 2025 | 2026 | Δ | t |
|---|---|---|---|---|
| exit mix: `stop` | 55.7% | 56.5% | +0.8pp | — |
| exit mix: `trail_stop` | 35.4% | 32.6% | −2.8pp | — |
| exit mix: `exhaust+trail_stop` | 5.8% | 6.4% | +0.6pp | — |
| `stop` mean R | −1.0899 | −1.0933 | −0.003 | −0.96 |
| `trail_stop` mean R | +1.168 | +0.993 | −0.175 | −1.86 |
| `exhaust+trail_stop` mean R | +3.019 | +2.956 | −0.063 | −0.21 |
| all winners mean R | +1.420 | +1.344 | −0.076 | −0.67 |
| all losers mean R | −1.060 | −1.041 | +0.019 | +1.05 |
| ≥2R rate | 8.6% | 9.0% | +0.4pp | — |

**Exits did not break.** The loss side is identical to three decimal places, the
exit-reason mix is within 3pp, the exhaustion path is unchanged, the ≥2R rate is
unchanged. The only cell that even approaches significance is trail-stop winners
running 0.175R shorter in 2026 (t = −1.86) — a market-continuation property, not
a machinery property, and it would cost the book ~0.06R/pick, a third of the P1 gap.

Additive decomposition of the P1 pick delta (−0.183R):

| component | 2025 → 2026 | contribution |
|---|---|---|
| win rate | 56.4% → 60.9% | **+0.133R** |
| average win size | 1.947R → 1.406R | **−0.330R** |
| average loss size | −1.026R → −0.990R | +0.014R |

The 2026 book wins *more often* with *smaller* wins. The whole of the gap is win
size, and win size is a tail count: the P1 book has **two** ≥3R trades in 2025 and
**zero** in 2026 (5.1% vs 0.0%; max R 5.41 vs 2.97). On 23 picks, zero-vs-one 3R
trade moves the mean by ~0.11R. This is the same tail-dependence CLAUDE.md
already flags for the book (top-5 = 80% of P&L).

---

## 4. Is it a market variable we could gate on?

| | 2025 | 2026 | |
|---|---|---|---|
| regime day mix (raw detections) | A 334 / B 54 / C1 48 / C2 27 | A 308 / **B 0** / C1 66 / C2 49 | 2026 SPY vol never reached 22% |
| raw R by regime | A −0.036 / B −0.035 / C1 +0.150 / C2 +0.072 | A −0.105 / C1 +0.126 / **C2 −0.173** | |
| **P1 picks, A-regime days** | n=20, **+0.783R** | n=16, **+0.751R** | **flat** |
| P1 picks, C1 days | n=13, +0.394R | n=6, −0.026R | |
| P1 picks, C2 days | n=1, +1.333R | n=1, −1.075R | |

**Observation, explicitly NOT a proposal:** every bit of P1's 2026 softness sits
in the 7 non-A-regime picks. Restricted to A-regime days the book reads +0.783R
(2025) vs +0.751R (2026) — no decay at all — and pooled it is +0.769R over 36
picks. The C1 cell's year-over-year delta is −0.420R on **se 0.811, t = −0.52**;
the 2026 non-A sample is **7 trades**. This cannot support a gate. It is logged
here as a **pre-registerable hypothesis** (an A-regime-only or non-A-sized-down
P1) requiring its own DESIGN.md, split plan and independent rebuild before any
number from it is quoted. Nothing about it is proposed, and the multiplicity
below applies to it in full.

SPY realized vol cannot be the variable: 2026 is the *calmer* year (13.4% vs
15.8% mean 20d vol) and contains zero B-regime days, yet is the weaker one.

---

## 5. Two artifacts that manufacture part of the gap

**(a) A survivorship snapshot inside Stage-2.** `batch_backtest.py:3409` filters
the cache by `db.get_active_universe()` — **today's** 5,973-symbol universe
table. Delisted 2025 names are silently deleted from the 2025 book:

| | share of detections removed | mean R of removed |
|---|---|---|
| 2025 | 10.4% (48 rows) | **−0.357R** |
| 2026 | 15.8% (67 rows) | −0.036R |

At the picked level the effect is stark: of the 6 P1 picks the Stage-2 day-loop +
universe layers remove, **four are 2025 losers at ≈ −1.1R each** (AXGN, VSTD,
CAPS, ITOC — none present in the universe table) and **one is a 2026 winner at
+2.51R / $22,616** (PMN, removed by the concurrency/daily-loss rail). This alone
is why the owner-facing 56-trade book shows a bigger gap than the raw selection:

| book | n25 | R25 | n26 | R26 | Δ R | t |
|---|---|---|---|---|---|---|
| P1 selection, pre-slot (survivorship-free) | 39 | +0.651 | 23 | +0.469 | −0.183 | −0.46 |
| Stage-2 56-trade book (today's run) | 34 | **+0.912** | 22 | +0.376 | −0.536 | −1.31 |

**Two-thirds of the apparent decay in the headline book is a today's-universe
snapshot deleting 2025's losers, plus one 2026 winner lost to a slot rail.**

**(b) The resim drift.** Every P1 level in CLAUDE.md and README §5/§6 is a resim
number carrying BT_STATUS §2a's −$10.3K / 7.4% exit-reconstruction drift. This
report uses regen-7's own exits throughout, so it is not exposed — but it means
the documented P1 monthlies (including Jul/Aug-26) are not the faithful path.

---

## 6. Sample honesty — n, MDE, and whether "decay" is measurable at all

| population | n25 | n26 | pooled sd | se(Δ) | observed Δ | t | p | **MDE₈₀** |
|---|---|---|---|---|---|---|---|---|
| raw detections | 463 | 423 | 1.40 | 0.094 | −0.067R | −0.71 | 0.48 | **0.263R** |
| AS-IS picks | 57 | 37 | 1.55 | 0.312 | −0.579R | −1.86 | 0.063 | **0.915R** |
| **P1 picks** | **39** | **23** | **1.61** | **0.401** | **−0.183R** | **−0.46** | **0.65** | **1.182R** |
| Stage-2 56-book | 34 | 22 | — | 0.409 | −0.536R | −1.31 | 0.19 | 1.200R |

Bootstrap 95% CI on the P1 delta (20,000 resamples): **[−0.965R, +0.580R]**.

Sample sizes needed to detect a decay of a given size at 80% power / 5%, at the
book's own variance and 2.8 picks/month:

| effect to detect | picks needed **per year** | months of live trading |
|---|---|---|
| 0.18R (the observed P1 gap) | 1,249 | **446** |
| 0.30R | 449 | **161** |
| 0.50R | 162 | **58** |
| 0.75R | 72 | **26** |

**The decay is not distinguishable from noise — and it is not close.** The
instrument's smallest detectable effect on the shipped stack is 1.18R; the
observed gap is 0.18R, roughly one-sixth of it. To confirm even a large 0.5R
decay would take 58 months of live trading at this book's frequency.

Three more honesty items:
- **2026 alone cannot be shown positive either.** P1 2026 = +0.469R ± 0.292
  (t = 1.61, p = 0.11); the 56-trade book's 2026 = +0.376R ± 0.289 (t = 1.30).
  The book is undecidable in *both* directions. Nothing here says it works.
- **The "decay" is one quarter of 3 trades.** By half-year the P1 book reads
  2025H1 +0.365 (24) · 2025H2 +1.109 (15) · **2026H1 +0.627 (20)** · 2026H2
  −0.588 (**3**). 2026H1 is *better* than 2025H1. There is no monotone decay;
  there is one 3-trade quarter, whose own t against zero is −1.06.
- **Multiplicity.** ~100 cells were inspected across the three scripts (9 layer
  rows, 9 gate separations, 5 stack variants, 6 population buckets, 13
  distribution moments, 8 regime cells, 6 exit-reason cells, 8 half-year cells,
  4 vol-split cells, ×2 years). No rule is drawn from any of them; the §4
  observation would need pre-registration and is not proposed.

**Power statement, per the phrasing rule:** no decay of the bull-flag selection
stack was detectable in THIS cache (886 regen-7 detections, 62 P1 picks), over
THIS window (2025-01 → 2026-08), at THIS book size (2.8 picks/month, 4 slots),
under regen-7's own exits. The smallest decay this test could have seen is 1.18R
per pick.

---

## 7. Shrank, inverted, or noise?

**NOISE**, on the shipped stack — with one real, significant sub-finding.

- **Not inverted.** Every gate keeps the better side in 2026. Whole-stack
  separation +0.751R (t 2.56) in 2025 vs +0.680R (t 2.19) in 2026.
- **Not shrank, measurably.** The P1 gap is −0.18R against an MDE of 1.18R
  (p = 0.65, CI spans zero by a wide margin). Even the headline 56-trade book's
  −0.54R is p = 0.19, and two-thirds of it is the survivorship artifact of §5a.
- **The AS-IS gap (−0.58R, p = 0.063) is the strongest signal in the file — and
  it is about a stack that was retired on 2026-09-07.** Its mechanism is real and
  significant: the detection population drifted +10.4pp (z = 3.10) into the three
  era-consistent bad buckets, and the as-is stack had no gate for them. P1 does.
- **What actually moved, within the noise:** win *size*, not win rate (WR rose
  56% → 61%); the 2026 book simply has no ≥3R trade. The exit machinery is
  demonstrably unchanged.

---

## 8. Recommendation

### (a) RESUME — the stated precondition is answered, and the answer voids the decay grounds for the pause.

The book was paused with "the selection stack decayed" unresolved. On the shipped
P1 stack that decay is **unmeasurable** (−0.18R vs an MDE of 1.18R, p = 0.65),
the stack has **not inverted** (+0.68R picked-vs-rejected in 2026, t = 2.19), no
named layer is dead, and the one significant change — a +10.4pp population drift
into the price/pole/VWAP buckets — is **already gated by the config that is
live**. Two-thirds of the residual gap in the owner-facing run is a
today's-universe survivorship snapshot that deletes 2025's losers (§5a).

**Grounds for resuming are not "the backtest says it works" — it doesn't, and
can't.** 2026's own +0.47R is t = 1.61. The grounds are:

1. **The pause's stated reason does not survive measurement.** A book cannot stay
   paused for a decay that no available sample can detect.
2. **L0 is the only instrument that can ever decide this.** The BT is exhausted:
   446 months of this book would be needed to resolve the observed gap. Live
   trades at $150 risk are the only new information, and P1 has produced **zero**
   of them (BT_STATUS §5 — the 9/7–9/14 week generated no BF trades at all).
3. **The risk is already bounded by machinery that exists.** L0 = $150/trade,
   rails −750/−1050/−1200, `daily_loss_limit` −750, `scripts/bf_ramp_check.py`
   enforcing the above-water advancement rule. Worst month at L0 ≈ −$965.
   No new rule, no new flag, no config change other than `trading.enabled`.

**The gate is the existing ramp, not a new filter.** Resume = `trading.enabled:
true` + restart. Nothing else changes.

**What I am explicitly not claiming, and the owner's separate call:** CLAUDE.md
records the 9/14 pause reason as *account/StopMonitor/symbol isolation while the
HOD-break was being validated*, not the decay. That rationale is an owner
decision and is outside this diagnosis — though HOD-break's own edge was refuted
on 9/15 and its engine now runs dry as instrumentation only. If isolation is
still wanted, the correct outcome of this report is "the decay grounds are void;
the pause continues for an unrelated reason" — and that should be said in those
words rather than attributed to the book's performance.

### What would change this to (b) STAY PAUSED

A named layer would have to show negative separation — `sep < 0` in a year on
n ≥ 30 per side — or the whole-stack separation would have to fail to clear zero.
Neither is true today. Re-run `decay_attrib.py` after every 20 further picks.

### Pre-registration queue (nothing proposed, nothing measured beyond §4)

1. **A-regime-only P1** (§4): pooled +0.769R on 36 picks; 2026 non-A n = 7.
   Needs DESIGN.md, a declared split, an independent rebuild, and a tail test.
2. **Fix the Stage-2 survivorship snapshot** (§5a) — `get_active_universe()`
   should be point-in-time (the Databento PIT universe already exists for ORB).
   This is a measurement defect, not a strategy change, and it currently biases
   every 2025-vs-2026 comparison the BT produces.
3. **Quote the consistency bar at the live rail** (BT_STATUS §7.3, unchanged).
