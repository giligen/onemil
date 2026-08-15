# The ORB Machine — Deliberate Rule Book (2026-07-11)

Owner mandate: "we don't trade on 'accidentally', we trade deliberately
based on rules." This document is the machine reassembled: every layer as
an explicit rule with its MECHANISM (why it should work), EVIDENCE (what
proved it, era-consistency), and MONITOR (how we see it working live).
A rule without a mechanism is a statistic; a mechanism without evidence
is a story. Every layer here must have both or be marked for review.

## The thesis (zoom-out)

We buy **day-2 momentum continuation**: a small-cap that already proved it
can move (yesterday's range), that the market is actively trading before
the open (premarket dollar volume), breaking above its first-5-minute
range — and we size UP only where we have era-consistent evidence of a
fatter right tail, cut failed breakouts within minutes, and lock profits
mechanically. Two asset classes flow through the machine and they are NOT
the same instrument:

- **Common stocks**: move on company events. Premarket news = a fresh
  catalyst being priced in real time.
- **Leveraged wrappers** (45% of the qualified universe: 2x/inverse
  single-stock ETFs — Tradr/T-Rex/Direxion/ProShares): move on their
  underlying's momentum and derivative-complex flows. They have no
  company events; "their" news belongs to the underlying, and when the
  underlying's story is big enough to make headlines, the wrapper crowd
  is late retail — crowding, not catalyst.

## Layer rules

### L1. Universe & candidacy
- **RULE**: gap-up small caps, price band $2–30, premarket movers feed.
- **MECHANISM**: retail-momentum land; large caps don't produce the
  30-80% continuation days the exit structure is built to catch.
- **EVIDENCE**: $30–60 band study (rejected — no edge above 30);
  $2 floor validated. Band widening rejected 2026-07 (keep-simple).

### L2. PDR veto — "day-2 of the fireworks, not day-1"
- **RULE**: skip picks whose PREVIOUS day range ≤ 8% (slot stays empty,
  never refilled).
- **MECHANISM**: ORB monetizes continuation. A gapper with a quiet
  yesterday is a fresh pop (day-1) — those mean-revert; a gapper that
  already ran yesterday has a proven crowd.
- **EVIDENCE**: +35% TOT, MDD −$29K→−$20K, all 3 eras, monotone in
  threshold 6–10. Refill form tested TOXIC.
- **MONITOR**: `grep "PDR VETO"` journald; green-check parity.

### L3. Composite quality score + quintiles
- **RULE**: 7-feature z-composite, threshold ≥0, quintiles from
  **frozen H1-2025 fit**. Q1 dropped.
- **MECHANISM**: cross-sectional quality ranking of the day's gappers.
- **EVIDENCE**: Q1 filter +$8.5K OOS. **Frozen-fit doctrine**: quarterly
  refits tested $34–47K WORSE OOS — the frozen fit is a regularizer.
- **STATUS**: deliberate; refit cadence CANCELLED (money-machine audit).

### L4. Concentration control
- **RULE**: max 4 slots, family + super-group dedup, Q4-first ordering.
- **MECHANISM**: the daily P&L is monster-driven; slots exist to hold
  independent bets, not 4 copies of the same trade (MSTR-complex days).
- **EVIDENCE**: orderings within ±$8K (re-audited 2026-07-10);
  Q2-first doubles MDD via crowded-day junk. Ordering is NOT the lever;
  sizing is.

### L5. Sizing stack (the deliberate-rules core)
Risk parity ($risk/stop%) capped per-pos, then multipliers stack:
1. **Quintile mults (corrected 2026-07-10)**: Q2 1.5 / Q3 1.4 / Q4 0.5 /
   Q5 0.5. MECHANISM: under the true 15:45 exit physics, monster
   frequency lives in Q2/Q3; Q4/Q5's TRAIN edge was an artifact of
   never-traded 15:59 exits. EVIDENCE: leave-out-stable inversion, all
   eras; old config on corrected physics = 2026 −$21K vs +$109K.
2. **News-gated PM mult**: see L6 — the class-aware rule.
3. **Caps**: mults clipped ≤1.5 each (Q5-cap anti-overfit doctrine);
   stacked max 3.0× per-pos cap; buying-power gate at submit.

### L6. The news × premarket-volume rule (class-aware, deliberate)

**Definitions** (answering the owner's window question precisely):
- **PM$** = TODAY's premarket dollar volume, 4:00–9:29 ET. It is NOT
  yesterday's run — yesterday's run is L2 (PDR). PM$ measures whether
  the crowd is HERE THIS MORNING, before our 9:35 entry.
- **News windows**: A = yesterday's session (the catalyst that caused
  the run PDR requires), B = overnight/after-hours, C = today's fresh
  premarket. Studied separately 2026-07-11 (see addendum + tables in
  research/orb_news_catalyst_jul2026.md).

**RULE (stocks)**: PM$ > $5.82M (frozen TRAIN tercile) AND premarket
news on the ticker → 2.0×. Either alone → 1.0×.
- MECHANISM: a fresh company catalyst (news) that the market is actually
  trading (PM$) is the highest-monster-rate state we can observe at
  9:31. News without volume = story nobody trades (negative all eras).
  Volume without news = flow without a reason — flat all eras.
- EVIDENCE: combo cell +$1,580/+$1,569/+$935 per trade per era; monster
  rate 28/15/13% vs 6-8% rest; +$51K/18mo pipeline, MDD improves.

**Window-A verdict (owner's prev-day question, tested 2026-07-11)**:
yesterday-session news as a boost extension is NO-SHIP — mean-positive
all eras but the payload is 2 trades from H1-2025 (ASST/PONY) and 2026
adds n=4 mean +$54. Same "positive with bad recency" pattern that killed
pole=2. Re-check if 2026H2 A-cell shows life. The market prices
yesterday's session news by today's open; what it hasn't priced is the
FRESH morning catalyst — that's the deliberate mechanism of this rule.

**RULE (wrappers)**: news component is **structurally out of scope** —
wrappers never receive the news boost. This is now an explicit class
rule, not an accident of Benzinga tagging:
- MECHANISM: a wrapper has no company events. Its underlying's headline
  days are crowded late-retail days in the derivative — the WORST
  wrapper cell, not the best.
- EVIDENCE (2026-07-11): underlying-news × PM$ for wrappers NEGATIVE all
  3 eras (−$324/−$125/−$27); mapping the gate to underlyings loses in
  every era; wrapper monsters are mostly newsless momentum (8/11 had no
  underlying news). Own-ticker wrapper news ≈ 1% (fund PR) — excluded
  by the same rule.
- The wrapper edge lives in L2+L3+L5.1 (momentum quality), not news.
- IMPLEMENTATION: `trading/orb_asset_class.py` (shared BT+live):
  lev-family sets → 33K offline class map (active+inactive dump,
  2026-07-11, incl. verified delisted-stock overrides) → asset-name API
  fetch (8s bounded) → 'unknown' = never boost blind. Book with the
  explicit rule: **$295,896** (vs $301,518 accidental — the $5.6K is the
  price of not being one Benzinga tagging change away from 2x-boosting
  the crowding cell; MDD identical −$18,174, 25H1/25H2 unchanged).

**MONITOR**: EoD sizing attribution per trade (quintile × pm × news ×
class); pool-level Benzinga lag audit; news-boost frequency by class.

### L7. Entry mechanics
- **RULE**: pre-placed stop-limit at range_high +30bps, 60min cancel,
  spread gate 300bps, buy-stop guard (never chase past limit).
- **EVIDENCE**: 30bps validated; 150→300 spread loosening kept monsters
  (BKKT/XNDU); guard-skips accepted as live-only divergence (~3% picks).

### L8. Exits
- **RULE**: stop at range_low; static lock (arm 1.75R → stop 0.5R);
  touchgo M (breakout-bar close-position < 0.5 → out) and D (bar-1
  revert ≥0.75R → out at −0.5R); 15:45 force close.
- **MECHANISM**: momentum either continues within minutes or it was a
  failed breakout; no profit targets because the P&L is right-tail.
- **EVIDENCE**: static-lock Pareto frontier (50+ variants); touchgo
  +$27K OOS, WR 47.8→52.1%; breakout-bar re-key fixed BT↔live parity.

### L9. Operations & self-validation
- **RULE**: every day is adjudicated (green streak = ramp gate); every
  sizing decision recomputed EoD from recorded inputs (hard gate);
  vendor drift (news lag) measured nightly pool-wide; BT ground truth
  refreshed nightly incl. PM$/news appends.
- **MECHANISM**: the machine must PROVE it did what the rules say, every
  day, or capital doesn't ramp.

## Standing doctrines
1. **Frozen-fit**: no refits without walk-forward proof (refits chase
   regimes; the frozen fit is an accidental regularizer — now a
   deliberate one).
2. **Upsize-only sizing**: no signal may size a trade below its
   risk-parity base (fail-open = 1.0×). Never boost blind.
3. **No-refill**: vetoed/skipped picks consume their slot.
4. **Ship rule**: profit up AND MDD not worse, era-consistent, or no-ship.
5. **Fail-open + loud**: any data-source failure degrades to base sizing
   with a WARNING; nothing in the sizing stack may delay entry >8s.
6. **Era-consistency over magnitude**: a smaller edge in all 3 eras
   beats a bigger edge in 2 of 3.

## OWNER ORDER 2026-08-14: ORB LIVE = ZERO
Owner: "Go to zero" (after the DST session-open bug voided all BT-derived
param evidence — see research/orb_clean_rederivation_aug2026.md).
Executed 2026-08-14 21:03 UTC: orb.yaml strategy.enabled=false, service
restarted, engine verified enabled=False (bar intake + entry submission
hard-gated at orb_engine.py:932/:1564). Positions flat, zero ORB rows 8/14.
RE-ENTRY BAR: owner-approved rebuilt param set at $10K budget, derived on
clean data with walk-forward evidence (candidates: B+/D+ per the clean
re-derivation; drag-reduction program running). Preconditions for any live
dollar: <=5s order-latency fix, 1 month forward validation, Nov-2026
first-EST-week live-vs-BT parity audit.
