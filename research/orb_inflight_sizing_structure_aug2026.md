# ORB in-flight family: SIZING STRUCTURE (2026-08-14)

Owner mandate: cut the clean book's ~-$101K loser drag (ranks 41+) toward
-$50K while retaining >=85-90% of the top-40 winners' P&L. This family tests
whether losers can be made **structurally cheaper** without predicting them:
H1 half-entry + confirmation add, H2 conviction-scaled risk, H3 progressive
day re-risk, H4 stop-distance skip.

**Family verdict: NO SHIP — best_rule NONE.** H2 and H3 are hard NO-GOs
(they halve ANNA/BNAI-class monsters). H1 and H4 fail on evidence quality:
H1 is an arithmetic near-wash (drag falls only ~6% while every winner pays a
~0.25R toll), H4's premise is refuted by the data and its only pass-shaped
configuration kills IREX 2026-07-30, a top-10 winner.

## Method

- Harness: `research/scripts/orb_clean_harness.py` reused untouched; family
  code in scratchpad `sz_lib.py` (extended static-lock simulator that
  additionally tracks the first bar/minute price touches entry+0.5R,
  pessimistic on same-bar touch+stop). **Parity check: extended sim
  reproduces the harness per-share exit price and reason on all 6,918
  physics keys exactly.**
- Books: (a) shipped-clean $100K/$3K yaml-fit — reproduces +$96,454 / 219
  trades, top-5 +$101,700, drag(ranks 41+) **-$100,640**; (b) B+ ($10K,
  refit TRAIN, q40, pdr11, N3, cvY, $375 risk, uniform, no PM) — +$9,931 /
  105 trades, drag -$5,574. Pipeline-integrated: H1-H3 are per-trade P&L
  transforms of the same selected book (selection/slots/dedup identical);
  H4 is a post-selection no-refill veto (house style).
- Windows: TRAIN=2025H1 (derive/tune), VAL=2025H2, OOS=2026. Pass bar: drag
  falls AND top-40 retention >=85% in ALL THREE windows, no named monster
  casualty, no sensitivity cliff.
- Top-40 / drag cohorts defined per book on its own baseline ranking.

## H1 — Half-entry + confirmation add at +0.5R : **FAIL**

Enter half risk at trigger; add the second half at entry+0.5R (+10bps add
slip) when MFE reaches +0.5R. Add fills modeled pessimistically (same-bar
touch+stop => add filled, then stopped; only 3-4 such trades per book).

| book | rule | total | drag(41+) | ret40 TRAIN/VAL/OOS | drag falls in |
|---|---|---|---|---|---|
| A base | — | +96,454 | -100,640 | — | — |
| A | H1@0.5R | +87,205 | -94,558 | **83.2** / 88.0 / 95.4 | TRAIN, OOS (VAL RISES +$89) |
| A | H1@0.4R | +91,338 | -93,371 | 86.4 / 90.3 / 96.2 | TRAIN, OOS (VAL rises +$138) |
| A | H1@0.6R | +85,274 | -93,543 | **80.0** / 85.7 / 94.5 | all |
| B base | — | +9,931 | -5,574 | — | — |
| B | H1@0.5R | +9,529 | -4,547 | **82.2** / 87.1 / 94.6 | all |
| B | H1@0.4R | +9,639 | -4,709 | 85.6 / 89.6 / 95.6 | all |

Mechanism (why it can't work, from the touch diagnostics):
- **100% of top-40 winners touch +0.5R** — the add always fills, so monster
  exposure is preserved. Good. But every touched trade pays an unavoidable
  toll of 0.5x(0.5R+slip) per share ~= $750+ at $3K risk sizing.
- **50.3% of the drag cohort ALSO touches +0.5R before dying** (book A: 90
  of 179 drag trades, -$31.4K). Those losers end up with FULL size at a
  WORSE blended basis — they lose MORE than baseline. Only the 89 untouched
  drag trades (-$69.3K) get halved.
- Net on A: drag improves just $6.1K (of the $50K target) while total P&L
  drops $9.2K, MDD worsens (-$20.8K -> -$22.4K), worst month worsens
  (-$13.2K -> -$13.7K).

Monster cost, named (book A, H1@0.5R): ANNA 2026-03-20 -$786 (of $41.0K),
BNAI 2026-01-23 -$548, ANTX 2026-03-09 -$255, ZURA 2025-09-30 -$439, IREX
2026-07-30 -$514 — individually tolerable (~1-2% each); the rule dies on
the loser side, not the winner side.

Sensitivity: no cliff, but monotone convergence to a no-op — the only
variant that clears every bar (0.4R on B+) buys $865 of drag reduction for
-$292 total P&L on a $10K book, plus two-legged order complexity and real
chase risk on thin gappers. Spec 0.5R fails TRAIN retention (<85%) on BOTH
books. OOS numbers (A@0.5R): drag reduction +$3,994, retention 95.4%.

## H2 — Conviction-scaled risk (composite quartile) : **NO-GO**

Full $risk if composite >= TRAIN-q75 (frozen: A 0.3997, B 0.3501), else
half. Cheap to implement — and catastrophic:

| book | OOS retention | OOS total vs base | named casualties (halved) |
|---|---|---|---|
| A | **54.6%** | +45,301 vs +91,409 | ANNA -$20.5K, BNAI -$13.6K, ANTX -$9.1K, BNAI2 -$3.9K |
| B | **54.0%** | +3,480 vs +7,089 | same class |

Drag does fall hard (A: -$100.6K -> -$63.6K; OOS reduction $11.9K) but the
composite score does not rank winners *within* the selected book — the 2026
monsters sit BELOW the TRAIN-q75 cut and get halved. Sensitivity (q50/q80):
OOS retention 52-64% everywhere — the failure is structural, not a
threshold artifact. Kills ANNA/BNAI class => dead rule by mandate.

## H3 — Progressive day re-risk (half until day's first +0.5R) : **NO-GO**

Start the day at half risk; amend resting orders to full risk once any
earlier trade touches +0.5R (live-implementable via cancel/replace).

- Top-40 retention is **exactly 50.0% in every window, both books, at every
  trigger tested (0.4R / 0.5R / 0.6R)** — i.e. NOT ONE top-40 winner ever
  trades at full risk. ORB entries cluster at 9:35-9:40; the monster IS the
  day's pathfinder — the "day quality" signal arrives only after the
  monster's entry has already filled at half size.
- The trades that DO unlock full risk are net LOSERS: A 12/219 unlocked,
  combined -$5,281 (0.4R: 13 unlocked, -$4,979; 0.6R: 9, -$7,058); B 2/105,
  -$268. The rule is an inverse-quality filter: it doubles up on late
  same-day follower trades, which are exactly the crowding losers.
- A totals: +$45,586 vs +$96,454 (ANNA -$20.5K, BNAI -$13.6K, ANTX -$9.1K).

Drag halves (-$53.0K) only because everything halves. Dead rule.

## H4 — Stop-distance-aware skip (range > tau% of price) : **FAIL**

Premise check first: **the drag cohort does NOT skew wide-R.** Book A
median range_size_pct: top-40 = 4.34%, drag = 4.20% (means 4.93 vs 4.59) —
if anything winners are wider. Wide-R is where the fireworks live: trades
with range > 6% are net **+$30,080** on book A (12 of them top-40 winners).

- tau=10: inert (1 trade, +$45). tau=8: touches only 10 trades in 20
  months; aggregates technically pass (drag falls all windows, ret40
  100/96.5/96.6, total +$98,862, MDD -$18.3K) — but the skips include
  **IREX 2026-07-30 +$4,361 (the #10 winner and the biggest 2026H2 trade)**
  and SDOT 2025-09-15 +$1,642, while the entire benefit rests on two trades
  (JLHL 2026-07-17 -$4,230, AXTI 2025-11-03 -$2,185). n=10 is luck-level
  evidence and it deletes a top-10 monster => dead by mandate rule 4.
- Sensitivity cliff: tau=6.4 looks spectacular overall (+$106.6K) but VAL
  retention collapses to **60.8%** — classic single-era overfit texture.
  tau=9.6 inert. No robust plateau exists.
- On B+ the rule is a literal no-op at tau=8/10 (0 trades — the B+ stack
  already excludes wide-R rows). OOS (A, tau=8): drag reduction +$5,029,
  retention 96.6% — reported for completeness, not shippable.

Note on the intuition behind H4: it's also mis-aimed mechanically — under
risk-parity sizing wide-R trades hold dollar risk constant ($3K) while
NARROW-R trades are the ones whose position hits the $25K cap and risk
less; there is no per-dollar-of-edge penalty to being wide that the sizer
hasn't already paid for.

## Family conclusions (for the parent program)

1. **Sizing structure cannot manufacture the -$50K drag cut.** The four
   structural levers fail for four distinct, data-backed reasons: the
   composite doesn't order winners within the book (H2); monsters are the
   day's first movers, so day-sequencing signals arrive too late (H3);
   the +0.5R confirmation gate taxes 100% of winners while only halving
   the 50% of losers that never touch it (H1); and losers are not
   structurally wider-R than winners (H4).
2. **Useful decomposition for the early-cut family**: of book A's -$100.6K
   drag, **-$69.3K comes from 89 trades that NEVER see +0.5R MFE**, and
   -$99.7K of gross drag is plain full stops (71 trades). That untouched-
   loser pool is addressable by trade-level early exits (price-path
   information), not by position sizing — sizing can only halve it (H1)
   at the cost of taxing everything else; an exit rule can zero-in on it.
3. If a future variant of H1 is ever revisited, the only bar-clearing form
   found is add@0.4R on the B+ $10K book — economically negligible
   (-$292 total for +$865 drag) and operationally the most complex rule in
   the family. Not recommended.

Artifacts: scratchpad `sz_lib.py`, `sz_step1-3.py`, `sz_state.pkl`,
`sz_results.pkl`. Baselines, cohort sums and parity checks logged in step-1
output. Nothing committed; orb.yaml / trading/ untouched.
