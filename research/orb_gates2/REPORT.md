# ORB gate removal — ten declared cells, scored on GREEN WEEKS against each cell's own count-matched null

2026-09-19. Pre-registration `PREREG.md`, written and committed **before any cell was scored**
(commit `067d436`). TEST seal `FREEZE.md`. Stage 1 is `research/orb_frequency/` (`a12bc6e`),
which built the separation map and named this study as its next step.

**Nothing ships from this stage.** ORB is paused (`orb.yaml strategy.enabled: false` since
9/14); any change needs the owner's word.

---

## 0. The answer, before the tables

**No cell beats its own count-matched null. Not one, on either split, under either fill model —
0 of 20 measured cells and 0 of 20 as-is cells exceed the 95th percentile of their own
permutation band, and the average cell lands 1.8pp *below* its own null's mean.** The
count-matched null predicts each cell's green-week share with a correlation of **0.966**.

> **The honest conclusion, in the phrasing rule's terms: in THIS candidate population, over
> THIS window (75 TRAIN+VAL market weeks), at THIS book size (8 slots, $3,333 per position),
> at THIS cost model, ORB's week shape is a function of HOW MANY PICKS IT TAKES and nothing
> else. No gate in the declared grid — including the shipped stack — demonstrated week-level
> timing skill. The smallest green-week effect this instrument could have seen is ±25.8pp on
> TRAIN and ±39.3pp on VAL.**

That is not a null result about ORB's P&L; the stack's R/pick separation is intact (§3c). It is
a null result about the *owner's primary metric*: green weeks cannot be engineered by choosing
gates. They can only be **bought with picks** — and every pick added past the shipped stack is
an edgeless pick, so the flat weeks it removes convert to red about as often as to green.

Per PREREG §4 the recommendation is therefore the null finding, with **G3 (catalyst veto OFF)**
named as an **exploration candidate only, explicitly not a claim** (§7).

---

## 1. Instrument and the reproduction gate

- **Engine**: the shipped `study_orb_pipeline_static_lock.py` replayed off a candidate dump via
  `ORB_BT_RESIM_CACHE` (selector-only; exit physics are the dump's). Every constant comes from
  `orb.yaml` as it stands today. **Nothing was refit.** Knobs moved only through documented env
  overrides — except G6/G8, which are row-exact derivations validated against the pipeline
  (§4). `orb.yaml`, `config.yaml`, production caches, orders, the service and the crons were
  never written; every artifact is under `research/orb_gates2/`.
- **Reproduction gate — PASSED BEFORE `PREREG.md` WAS WRITTEN.** As-is dump, N=8,
  `ORB_BT_ACCOUNT=26666.666666666664`, `ORB_BT_RISK=375`, Q1 on reproduces
  `research/fuckup_audit/D1_orb/book_n8_q1on.csv` with `DataFrame.equals() -> True`:
  **215 picks / $14,428.616990972434** (`repro.sh`, `repro.log`, `repro_n8_q1on.csv`).
- **PRIMARY fill model**: Stage Q's **measured** capped-limit arm — the elected stop-limit rests
  at the cap and fills at `min(ask, cap)` the first time the walked **per-trade SIP NBBO** ask
  reaches it before the 10:35 time stop; **ask > cap => no fill**, $0 and a spent slot. Never the
  band constant. **Secondary bracket**: the as-is dump. Both are reported; they agree on every
  sign in this report.
- **Entered-inclusive**: a modelled non-fill is R = 0 and **still a pick that burned a slot**.
- **Population**: 13,033 candidates, 427 trading days, 90 market weeks, 2025-01-02 -> 2026-09-16.
- **R** = `pnl_pct / max(range_size_pct, 1.0)` from the candidate row — invariant to the
  quintile mult, the per-position cap and the account size.
- Splits: **TRAIN 2025 (53 wk)** / **VAL 2026-01..05 (22 wk)** / **TEST 2026-06+ (§8, sealed)**.
- Slots are **8** throughout — settled by stage 1 §5 and not re-opened.

---

## 2. The ten declared cells

| cell | definition | picks TR/VA |
|---|---|---|
| **G0** | shipped B+ — every gate on (**baseline**) | 105 / 53 |
| G1 | Q1 quintile filter OFF | 129 / 61 |
| G2 | range-size veto OFF | 108 / 54 |
| G3 | catalyst veto OFF | 282 / 177 |
| G4 | G1 + G2 (the two wrong-side gates) | 132 / 62 |
| G5 | G4 + catalyst OFF (everything the map flags) | 378 / 218 |
| G6 | catalyst PARTIAL, `min_cohort: 2 -> 1` | 268 / 172 |
| G7 | PDR veto OFF | 117 / 61 |
| G8 | G4 + catalyst PARTIAL | 343 / 208 |
| G9 | G4 + PDR OFF (the full documented-defect cleanup) | 148 / 70 |

No cell was added after scoring began.

---

## 3. THE CELL TABLE — ranked on green weeks, with each cell's own null band beside it

Measured fill model, N=8. **TR** = TRAIN (53 wk), **VA** = VAL (22 wk). The **null band** is
2,000 draws of that cell's own pick-level P&L shuffled across its own picks with each week's
pick COUNT held fixed — i.e. what a book of exactly that frequency and exactly that P&L
distribution produces with **zero** week-timing skill.

| cell | TR pk/wk | VA pk/wk | **TR green%** | TR null mean (p5–p95) | **VA green%** | VA null mean (p5–p95) | TR flat% | VA flat% | TR streak | VA streak | TR worst wk | VA worst wk | TR MDD | TR $ | VA $ | explore bar | null clause |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **G8** | 6.5 | 9.5 | **47.2** | 52.1 (45.3–58.5) | **63.6** | 59.4 (50.0–68.2) | **1.9** | **4.5** | **7** | 2 | −995 | −438 | −2,707 | 6,822 | **9,771** | **no** (streak) | **no** |
| **G5** | 7.1 | 9.9 | **47.2** | 53.0 (45.3–60.4) | **63.6** | 58.9 (50.0–68.2) | **1.9** | **4.5** | **7** | 2 | −995 | −443 | −2,916 | 7,059 | 9,531 | **no** (streak) | **no** |
| **G3** | 5.3 | 8.0 | **47.2** | 50.4 (43.4–58.5) | **54.5** | 52.4 (40.9–63.6) | 7.5 | 4.5 | 4 | 2 | −895 | −364 | −2,639 | 6,515 | 4,287 | **YES** | **no** |
| **G6** | 5.1 | 7.8 | 45.3 | 49.3 (41.5–56.6) | **54.5** | 53.0 (40.9–63.6) | 7.5 | 4.5 | 4 | 2 | −895 | −377 | −2,421 | 6,229 | 4,410 | **YES** | **no** |
| G9 | 2.8 | 3.2 | 35.8 | 37.7 (32.1–43.4) | 40.9 | 40.2 (31.8–50.0) | 22.6 | 22.7 | 4 | 2 | −496 | −368 | −813 | 4,271 | 7,883 | no (flat) | no |
| G1 | 2.4 | 2.8 | 35.8 | 38.4 (32.1–45.3) | 40.9 | 41.3 (31.8–50.0) | 28.3 | 22.7 | 3 | 2 | −418 | −368 | −792 | 5,312 | 8,245 | no (flat) | no |
| G4 | 2.5 | 2.8 | 35.8 | 38.1 (32.1–45.3) | 40.9 | 40.9 (31.8–50.0) | 28.3 | 22.7 | 3 | 2 | −418 | −368 | −792 | 5,207 | 8,180 | no (flat) | no |
| **G0** | 2.0 | 2.4 | **34.0** | 37.5 (32.1–43.4) | **31.8** | 36.7 (27.3–45.5) | 37.7 | 22.7 | 3 | 3 | −418 | −338 | −560 | 5,673 | 3,939 | baseline | **no** |
| G2 | 2.0 | 2.5 | 34.0 | 37.5 (32.1–43.4) | 31.8 | 36.6 (27.3–45.5) | 37.7 | 22.7 | 3 | 3 | −418 | −338 | −601 | 5,568 | 3,873 | no | no |
| G7 | 2.2 | 2.8 | 34.0 | 36.7 (30.2–43.4) | 31.8 | 35.9 (27.3–45.5) | 30.2 | 22.7 | 4 | 3 | −496 | −338 | −780 | 4,805 | 3,642 | no | no |

**MDE80 on the green-week share, unpaired: TRAIN ±25.8pp (53 wk), VAL ±39.3pp (22 wk).** The
largest green-week gap in the whole table — G0's 34.0 -> G5/G8's 47.2 on TRAIN — is **+13.2pp,
half the TRAIN MDE**. On VAL the +31.8pp gap exceeds the MDE, but VAL is 22 weeks and 53
baseline picks.

### 3a. The null clause, cell by cell — the decisive column

| | TRAIN obs − null mean | VAL obs − null mean | above own p95? |
|---|---|---|---|
| G0 | **−3.6** | **−4.8** | no |
| G1 | −2.5 | −0.3 | no |
| G2 | −3.5 | −4.8 | no |
| G3 | −3.2 | +2.1 | no |
| G4 | −2.2 | −0.0 | no |
| G5 | −5.8 | +4.7 | no |
| G6 | −4.0 | +1.5 | no |
| G7 | −2.7 | −4.1 | no |
| G8 | −4.9 | +4.2 | no |
| G9 | −1.8 | +0.7 | no |

**0 of 20 measured cells and 0 of 20 as-is cells exceed their own p95. None falls below its own
p5 either.** Mean residual −1.76pp (sd 3.08) measured, −4.42pp as-is.
`corr(green%, null mean) = 0.966`; `corr(green%, picks/wk) = 0.964`.

**Read the sign on the baseline.** G0 sits **below** its own count-matched null on both splits
(−3.6 / −4.8). The shipped gate stack does not merely fail to time weeks — the weeks it
concentrates its P&L into are marginally *worse* clustered than a random reallocation of its own
trades. At this n that is not a claim of anti-skill; it is the absence of skill, twice.

### 3b. Both worst-week rails — the §0b disclosure, honoured

PREREG §0b disclosed that this stage's rule was written knowing stage 1's numbers, and committed
to reporting **both** rails.

| cell | TR worst | scaled rail | flat 1.5x rail | VA worst | scaled rail | flat 1.5x rail | passes scaled | passes flat 1.5x |
|---|---|---|---|---|---|---|---|---|
| G0 | −418 | −628 | −628 | −338 | −506 | −506 | — | — |
| G1 / G4 | −418 | −696 / −704 | −628 | −368 | −543 / −548 | −506 | yes | **yes** |
| G2 | −418 | −636 | −628 | −338 | −511 | −506 | yes | **yes** |
| G7 / G9 | −496 | −662 / −745 | −628 | −338 / −368 | −543 / −582 | −506 | yes | **yes** |
| **G3** | **−895** | **−1,028** | −628 | −364 | −926 | −506 | **yes** | **NO** |
| **G6** | **−895** | −1,003 | −628 | −377 | −912 | −506 | **yes** | **NO** |
| **G5 / G8** | **−995** | −1,191 / −1,134 | −628 | −443 / −438 | −1,027 / −1,003 | −506 | **yes** | **NO** |

**G3, G5, G6 and G8 reach clause 5 ONLY under the pick-scaled rail.** Under stage 1's flat 1.5x
rail all four fail, exactly as catalyst-OFF failed in stage 1. The scaling is mechanistically
justified (a book at k× the picks has ≈√k× the weekly sigma) and was declared in advance — but a
reader who prefers the flat rail should know it changes which cells clear clauses 1–5, and it
does **not** change this report's recommendation, because **clause N fails for every cell under
either rail**.

### 3c. Claim bar (PLAN §1) per cell — reported, never merged into the rule

TRAIN t on mean R/pick, VAL sign agreement:

| cell | TR R/pick | TR t | VA R/pick | claim bar (t >= 2 TRAIN, VAL same sign) | ex-top5% R TR / VA |
|---|---|---|---|---|---|
| G0 | +0.410 | **2.78** | +0.469 | pass | +0.150 / +0.049 |
| G1 | +0.309 | 2.47 | +0.940 | pass | +0.069 / +0.048 |
| G2 | +0.384 | 2.66 | +0.437 | pass | +0.130 / +0.024 |
| G3 | +0.175 | 2.27 | +0.149 | pass | **−0.057 / −0.083** |
| G4 | +0.290 | 2.36 | +0.905 | pass | +0.054 / +0.026 |
| G5 | +0.131 | 2.08 | +0.311 | pass | **−0.082 / −0.045** |
| G6 | +0.187 | 2.35 | +0.160 | pass | **−0.045 / −0.078** |
| G7 | +0.309 | 2.27 | +0.373 | pass | +0.072 / **−0.061** |
| G8 | +0.152 | 2.25 | +0.339 | pass | **−0.074 / −0.033** |
| G9 | +0.210 | 1.88 | +0.771 | **fail** | −0.025 / −0.010 |

Two things must be said about this column and neither is comfortable:

1. **The baseline has the highest t in the table (2.78).** Every gate removal *lowers* R/pick and
   lowers t. Nine of ten cells nominally "pass" a t >= 2 bar — which, over this programme's ~62
   runs, is exactly what a pure null produces (expected largest |t| ≈ 2.9–3.1, §9). **The claim
   bar does not separate these cells and must not be read as nine discoveries.**
2. **Every catalyst-relaxing cell goes negative ex-top-5%** (G3 −0.057/−0.083, G5 −0.082/−0.045,
   G6, G8). The green weeks those cells buy are paid for with a body of losing picks under a
   thinner tail. Per the owner's rule this is a **diagnostic, not a penalty** — monsters are
   permitted — but it is the mechanism, and it is why the extra picks convert flat weeks to red
   as readily as to green (G5/G8: a **7-week TRAIN red streak** against the baseline's 3).

### 3d. Tail concentration — diagnostic only, never a penalty

Top-5 picks as a share of split P&L: G0 **54.8% TR / 118.2% VA**; G3 59.5% / 111.7%;
G5 55.5% / 89.9%; G8 57.4% / 87.7%; G9 74.1% / 107.0%. Every ORB book at this size is
tail-carried in both directions; removing gates does not reduce the concentration, it adds
losers underneath it. Reported because it was asked for; it entered no ranking.

### 3e. The as-is bracket (secondary fill model)

Same ordering, larger dollars, same null verdict (0 of 20 above own p95):

| cell | TR green% | VA green% | TR flat% | VA flat% | TR $ | VA $ |
|---|---|---|---|---|---|---|
| G0 | 37.7 | 31.8 | 35.8 | 22.7 | 6,662 | 6,386 |
| G3 | 45.3 | 63.6 | 7.5 | 4.5 | 8,307 | 7,293 |
| G5 | 41.5 | 63.6 | 1.9 | 4.5 | 8,603 | 12,537 |
| G6 | 45.3 | 63.6 | 7.5 | 4.5 | 8,257 | 7,416 |
| G8 | 43.4 | 63.6 | 1.9 | 4.5 | 8,603 | 12,777 |
| G1 / G4 | 39.6 | 40.9 | 26.4 | 22.7 | 6,301 / 6,196 | 10,693 / 10,628 |
| G9 | 39.6 | 40.9 | 20.8 | 22.7 | 5,260 | 10,331 |
| G2 | 37.7 | 31.8 | 35.8 | 22.7 | 6,556 | 6,321 |
| G7 | 37.7 | 31.8 | 28.3 | 22.7 | 5,794 | 6,089 |

---

## 4. G6 — the catalyst partial, and what it turned out to be

**The declared rule was `filter.catalyst_veto.min_cohort: 2 -> 1`** — the only partial the shipped
rule exposes, one line of `orb.yaml`, with no value to tune (PREREG §2a).

**Validation gate: PASSED, four of four, to the cent.** Because the catalyst veto is
post-selection with **no refill**, the catalyst-ON book is exactly the catalyst-OFF book minus
the vetoed rows. Re-deriving the veto with the **shipped shared helpers**
(`trading.orb_catalyst_veto.catalyst_veto_applies`, `orb_asset_class.underlying_anchor`, the
shipped class map, cohorts over the full day's candidate universe, the same tri-state raw-news
source) at `min_cohort=2` reproduces the pipeline exactly:

```
derive(G3, mc=2) == G0 : 215/215 picks, $10,297.090989 vs $10,297.090989   PASS   (measured)
derive(G5, mc=2) == G4 : 256/256 picks, $13,856.791914 vs $13,856.791914   PASS   (measured)
derive(G3, mc=2) == G0 : 215/215 picks, $14,428.616991 vs $14,428.616991   PASS   (as-is)
derive(G5, mc=2) == G4 : 256/256 picks, $18,129.309879 vs $18,129.309879   PASS   (as-is)
```

That is an independent re-derivation of a shipped gate agreeing with the pipeline to the cent,
and it is what licenses G6/G8.

**The finding about the knob itself.** At `min_cohort=1` a common stock anchors its own complex,
so every newsless pick with an **identifiable** underlying is confirmed; the veto survives only
for newsless picks with **no identifiable anchor at all**. Measured: the full veto cuts
**301 picks** (G3 459 -> G0 158); `min_cohort=1` re-vetoes only **19 of them — 6.3%**.

> **`min_cohort: 2 -> 1` is not a partial. It is 94% of the full removal.** The shipped rule has
> no usable middle setting. A genuine partial (a liquidity or range floor under the veto) would
> need a new threshold fitted on this data and new live code, and this stage pre-committed not
> to fit one. **If the owner ever wants the catalyst veto softened rather than removed, the
> softening has to be BUILT, not configured** — that is the actionable output of G6.

G6's scores duly sit between G3 and G0 but far closer to G3 (45.3/54.5 green vs G3's 47.2/54.5),
and it fails clause N identically.

---

## 5. The three dead knobs — confirmed, as the brief asked

**(a) `filter.threshold` is INERT.** Stage 1's books at threshold -> −0.5 and threshold -> OFF are
`DataFrame.equals()`-identical to the shipped book (215 picks, both). Q1's cut sits at composite
0.1059, far above the threshold 0.0121, so once Q1 has run the threshold can never bind on
anything that could win a slot. **Confirmed: a dead knob at 8 slots, and it should be documented
as such.**

**(b) PDR@11.0 — the LADDER is dead, the GATE is not, and the precise statement matters.**
PDR 11.0 -> 8.0 -> 6.0 -> OFF are all `equals()`-identical to each other (249 picks): relaxing the
*threshold* does nothing, because G1's `pdr >= 9.226` leg re-imposes a stricter cut underneath.
But removing the gate entirely is **not** free: **G7 takes 34 more picks than G0 over 90 weeks
(+0.38/wk) and LOSES $1,762** ($10,297 -> $8,535), with green weeks unchanged (34.0 / 31.8) and
flat weeks barely moved (37.7 -> 30.2 TR, 22.7 -> 22.7 VA). The band PDR@11 uniquely governs is
`9.226 < pdr <= 11.0` — **8.7% of the candidate population**. **Verdict: the PDR *threshold* knob
is dead below 9.226 and carrying it as a separate tunable is the accidental-rule pattern; the
PDR *veto* earns its keep and must NOT be removed** — G7 fails clauses 1 and 2 and costs money.
This corrects the "~97% redundant" shorthand: redundant as a knob, not as a gate.

**(c) Touchgo Rule D is inert AT THE SHIPPED GATE SET, not universally.** `tag_b1` fires on
**0 of 162 fills** in G0 — confirmed. But it fires 3 of 462 in G3 and 9 of 571 in G5: the rule is
dormant only because the shipped stack's 162 fills never present the pattern. **It still carries
config surface, tests and a live code path for zero firings in the shipped configuration**, and
that should be documented; it is not evidence the rule is wrong.
(Rule M fires 46 / 162 in G0, 134 / 462 in G3, 161 / 571 in G5.)

---

## 6. Availability audit on every gating field (PREREG §5)

Missingness over the candidate population, per split:

| field | TRAIN | VAL | TEST | what the cascade does with a missing value |
|---|---|---|---|---|
| `range_size_pct` | 0.00% | 0.00% | 0.00% | veto skipped, fail-open, WARNING |
| `prev_day_range_pct` | 0.00% | 0.00% | 0.00% | PDR fails open |
| `return_volatility_20d` | 0.00% | 0.00% | 0.00% | G1 fails open on NaN; the `0.0` short-history **marker** is vetoed (shipped 9/8) |
| `range_total_volume`, `range_close_position`, `range_return_pct`, `gap_pct`, `avg_daily_volume_20d` (composite inputs) | 0.00% | 0.00% | 0.00% | — |
| **news pair (tri-state)** | **0.00%** | **0.00%** | **0.00%** | absent pair -> `None` -> **fail-open, never veto** |
| **underlying anchor** | **10.30%** | **9.26%** | **12.28%** | `None` -> never complex-confirmed -> newsless **IS** vetoed |

Two things this audit establishes:

1. **The news channel has complete coverage in the BT**, so the fail-open path is never exercised
   here. The live engine's fail-open (fetch timeout) has **no BT counterpart** — a live news
   fetch failure makes the live book strictly *looser* than any cell in this report.
2. **The anchor is the only field with real missingness (9–12%)**, and its missingness is
   **decision-relevant in one direction only**: an unresolvable anchor can only ever *cause* a
   veto, never prevent one. Anchors are derived from the static class map + fund names, not from
   any later stage's key set, so there is no cohort leak of the D1 kind. The 9–12% band is stable
   across all three splits, including the sealed one, which is itself evidence against a
   look-ahead in the anchor construction.

Unmeasured, and said plainly (unchanged from stage 1): the **universe screens**
(`prev_volume >= 500K`, the $3–30 price band, the 15K RTH-9:35 range-computability floor) are
applied upstream in `study_orb_broad.py`, so the dump has **zero rejected rows** on them and
their separation cannot be measured without a features rebuild. The **live spread gate (300 bps)**
has **no BT counterpart at all**.

---

## 7. THE RECOMMENDATION

**Recommended action: NONE. No cell is recommended for a live change.**

The pre-committed outcome, in PREREG §4's own words:

> **No cell demonstrated week-timing skill; ORB's week shape is a function of its pick count, and
> the only lever on green weeks is how many picks the book takes.**

Two cells (**G3** catalyst-OFF and **G6** catalyst-partial) cleared the live-exploration bar's
clauses 1–5 — more green weeks on both splits, fewer flat weeks on both splits, red streak within
the allowance, P&L positive on both splits, worst week inside the pick-scaled rail. **Neither
cleared clause N**, and under stage 1's flat worst-week rail neither clears clause 5 either. Per
the pre-committed rule they are **exploration candidates, not claims**.

**The exploration candidate, if the owner wants one: G3 — catalyst veto OFF.**

- **Exact live config diff** (one key, zero state to unwind):
  ```yaml
  # orb.yaml
  filter:
    catalyst_veto:
      enabled: false        # was: true
  ```
  then `sudo systemctl restart onemil-trader`. Rollback is the same key back to `true` + restart.
  (ORB is currently paused at `strategy.enabled: false`; this would also require un-pausing,
  which is a separate owner decision.)
- **What it buys**: green weeks 34.0 -> 47.2% TRAIN and 31.8 -> 54.5% VAL; **flat weeks
  37.7 -> 7.5% and 22.7 -> 4.5%**; picks 2.0 -> 5.3/wk and 2.4 -> 8.0/wk; total P&L **up** on both
  splits and under both fill models ($5,673 -> $6,515 and $3,939 -> $4,287 measured;
  $6,662 -> $8,307 and $6,386 -> $7,293 as-is).
- **What it costs — stated first, not buried**: **worst TRAIN week −$418 -> −$895**;
  **TRAIN MDD −$560 -> −$2,639** (VAL MDD −$489 -> −$698); green months TRAIN 75% -> 58%;
  R/pick +0.410 -> +0.175 and **ex-top-5% R/pick +0.150 -> −0.057**. At $10K-stage sizing these
  dollars are a relative tool, never a forecast; the *ratios* are the content.
- **Pre-committed stop rule** (written here, before any live exposure): revert
  `catalyst_veto.enabled: true` immediately if **any** of —
  (a) a single live week realises worse than **−$1,030** at $10K-stage sizing (the pick-scaled
      rail this stage used to admit the cell at all);
  (b) after **40 live picks**, the realised green-week share is **below the shipped stack's 34%**;
  (c) **two consecutive red months**, or a red-week streak of **5**;
  (d) realised cumulative P&L below **−$1,500**.
  Review at 40 picks regardless of outcome. The counterfactual (catalyst-ON on the same days) is
  computable from the same pipeline every evening and must be logged beside the live book.
- **Why it is NOT a claim**: its green-week share is inside its own count-matched null on both
  splits; its +13.2pp TRAIN gain is half the TRAIN MDE80; it is the best of 20 scored cells in
  this stage and ~62 runs in the programme; and stage 1 reached the same cell by the same route,
  so this is a second look at one hypothesis, not two independent confirmations.

**What NOT to do, on this stage's evidence:**

1. **Do not remove the PDR veto** (G7): 34 extra picks, −$1,762, zero green-week gain.
2. **Do not ship G5/G8** ("everything off"): the highest green-week share in the table and a
   **7-week TRAIN red streak** with MDD −$2,707 to −$2,916. This is the frontier bending exactly
   where stage 1 predicted — flat weeks past a point convert to red, not green.
3. **Do not treat `min_cohort: 1` as a softening** (§4): it is 94% of the full removal.
4. **Do not re-open slots.** Settled at 8.

**Documentation changes that stand on their own** (no P&L claim attached, owner's word still
required before any file changes): `filter.threshold` is inert at 8 slots; the PDR *threshold*
knob is dead below 9.226 while the PDR *gate* earns its keep; touchgo Rule D fires zero times in
the shipped configuration. All three are the "accidental rule" pattern the machine-rules doctrine
forbids, and all three are config surface carrying no behaviour.

**The frame the owner should hear first**: stage 1 said green weeks are bought with pick count.
This stage tested that against ten gate configurations and **twenty count-matched nulls, and did
not find a single exception**. Choosing gates does not buy green weeks; taking more picks does —
and every pick ORB has left to take is an edgeless one, so the flat weeks it removes come back as
red weeks about as often as green. **If the owner wants materially more green weeks from ORB,
the answer is not in this gate set — it is a bigger candidate universe or a different book.**

---

## 8. TEST — the sealed split

Sealed per `FREEZE.md`; opened **once**, after §7 was committed, for exactly the two cells the
seal named: the baseline **G0** and — since the recommendation is the null finding — the
**highest-green-week cell**, which is a tie between G5 and G8 at 52.0% pooled green and 2.67%
pooled flat, broken by PREREG §4's pre-committed tie-break (higher pooled total R: G8 122.7 vs
G5 117.2) -> **G8**. **TEST is 16 weeks; its green-week MDE80 is >= 38pp. It cannot select
anything and it did not.**

*(§8a is filled in after the recommendation commit.)*

---

## 9. Multiplicity

**This stage: 17 pipeline runs (16 cells + the reproduction gate), 2 row-exact derivations,
40 scored cells (10 cells x 2 splits x 2 fill models) + 4 TEST cells**, plus 40 permutation nulls
of 2,000 draws each and the descriptive availability / dead-knob / tail tables.
**Programme total including stage 1's 45 runs: ~62 pipeline runs.** Under a pure null the
expected largest |t| over that many cells is **≈ 2.9–3.1** — above **every** t in §3c including
the baseline's 2.78. **Every "best" point in this report is a maximum over a grid, not a
discovery.** No per-cell p-value was treated as evidence on its own; only PREREG §4 selected, and
it selected nothing. Zero thresholds, z-params, quintile cutoffs or adaptive mults were fitted
anywhere in this stage.

---

## 10. Caveats

1. **Power.** 53 TRAIN and 22 VAL weeks at 2.0–9.9 picks/week. Unpaired green-week MDE80
   ±25.8pp / ±39.3pp. Only the VAL catalyst-family gaps exceed it, on 22 weeks.
2. **Relative tool, never a forecast.** The ORB pipeline at $10K-stage sizing is a relative
   instrument (standing rule). No dollar figure here is a projection.
3. **Both fill models are simulations.** The measured arm walked the real per-trade SIP NBBO for
   the flagged orders; it is still a model. As-is and measured bracket every headline and agree
   on every sign.
4. **The universe screens and the live spread gate are unmeasured** (§6). The gate map is complete
   only for the gates that live *inside* the candidate population.
5. **Exit rules are out of scope.** `research/green_weeks/REPORT.md` already showed ORB's
   flat-week share is identical across all 17 exit cells; exits cannot change a pick count.
6. **The permutation null is count-matched, not autocorrelation-matched.** It holds each week's
   pick count fixed and shuffles P&L across picks; it does not preserve within-day clustering of
   a symbol's siblings. That makes it, if anything, a *slightly easy* null to beat — and nothing
   beat it.
7. **2026-09 is a partial month** and ORB has been paused since 9/14, so its 2026-09 picks are
   BT-only.
8. **No independent rebuild of the pipeline was made** — the instrument IS the shipped pipeline,
   byte-reproducing an already independently checked book (D1 -> Stage Q -> stage 1). The one
   piece of new logic, the G6/G8 catalyst derivation, was validated against the pipeline to the
   cent on four checks (§4).
9. **This stage's rule was not blind** (PREREG §0b): stage 1's numbers were known when the rails
   were written. Both rails are reported (§3b) and the recommendation is invariant to which one a
   reader prefers.

---

## 11. Artifacts

```
research/orb_gates2/
  PREREG.md  FREEZE.md  REPORT.md
  repro.sh repro.log repro_n8_q1on.csv     the reproduction gate (byte-identical to D1)
  run_grid.py                              the 16 declared pipeline cells
  partial_catalyst.py                      G6/G8 + the 4-check validation gate
  analyse.py  cells.csv grid_meas.csv grid_asis.csv
  diag.py     availability.csv nulls_and_tails.csv rails.csv
  book_G*_{meas,asis}.csv  monthly_G*_{meas,asis}.csv  log_G*_{meas,asis}.txt
```
