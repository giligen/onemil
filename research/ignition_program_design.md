# The Ignition Program — design + first research (2026-07-18)

Owner mandate: "the market has monster movers every week, our ORB must
find them." The bottom-up census proved him right and located the pool.

## The supply (measured, 19 months)

- Market monster-days (open→high ≥30%, $2–50, ≥$5M day-$vol): **2,744
  = ~34/week, every week** (81/81 weeks).
- ORB's universe admits 16%; captured 34 of them = **97% of the whole
  book** ($279K of $288K). Every selection gate re-validated under
  live-parity — the gapper architecture is at its information ceiling.
- **The untouched pool: 1,380 flat-open monsters (~17/wk), median run
  +42%, median day-$vol $24M** — excluded by the gap≥5% screen alone.

## The anatomy (322-sample, bar-level, 2026-07-18)

- **Ignition timing: 50% ignite in the 9:xx hour, 30% in 10:xx — 73%
  inside the existing ORB session window.** These are OPEN-HOUR movers
  that didn't gap; not a midday regime.
- **The entry moment exists with room**: after first crossing +10% from
  open, median further continuation **+28%**; 97% continue ≥15% more;
  peak ≥30min after ignition in 86% of cases.
- Character at ignition: median 30-min pre-range 9.3% (moving, not
  coiled), ignition-minute volume ≈2.3× the prior half hour.
- ⚠ SURVIVORSHIP: these stats are monster-conditioned. The junk river
  (all +10%-crossers that die) is unmeasured — the harness's whole job.

## Design (deliberate; maximum reuse)

**Trigger**: first cross of +10% intraday change from open (the scanner
ALREADY computes this every minute for the full 5,628-symbol universe —
the INTRADAY QUALIFIED stream currently feeds only BF, which cannot use
it post-10:45). New consumer, zero new scanning infrastructure.

**Entry**: on trigger cross (marketable, ORB-style stop-limit semantics
+ spread gate + buy-stop guard), entry window 9:35–~13:00 (tunable in
harness; 80% of ignitions are before 11:00).

**Risk structure**: stop = pre-ignition consolidation low (trailing
30-min low), risk-parity sizing with the ORB caps; exits = the
battle-tested static-lock family + touchgo-style failure cut + 15:45
flat. The move shape (fast ignition, deep tail) is exactly what these
exits were built for.

**Filter candidates (from the validated taxonomy, to be tested, not
assumed)**: catalyst (news / complex-confirmation), RelVol at trigger,
float band, price band, PDR-style prior-day context, one-per-family.

**What it is NOT**: not a BF flag (shape refuted), not a gap play (ORB
owns that), not day-rank (refuted), not curated sectors (era-flip).

## The harness (the honest EV answer — next sessions)

1. Universe: EVERY symbol-day crossing +10% from open (est. 10–20K over
   19mo) — full junk river included.
2. Bars backfill for those symbol-days (background, ~news-backfill
   scale ×3; all persisted to cache).
3. Bar-accurate entry/stop/exit sim; era splits (25H1/25H2/2026),
   leave-out-top-K, negMo/MDD/worst-month; live-parity discipline from
   day one (no fitted constants without orb.yaml pinning; decision-
   parity hooks designed in, not bolted on).
4. Ship bar: owner rule + era-consistency + leave-out, same as ORB.

## Why this can change the machine's shape

ORB's 4 monster months = fair capture of 16% of monster supply. The
ignition pool is ~3× larger and weekly. Even ORB-like capture rates on
a fraction of it converts "4 monster months" toward "monsters monthly,"
which is the owner's stated goal. All downstream machinery (sizing,
exits, StopMonitor, attribution, green-check) is reusable.

Artifacts: /tmp/market_monsters.csv, /tmp/flat_open_monsters.csv,
/tmp/ignition_bars.csv, /tmp/ignition_anatomy.csv,
/tmp/fetch_ignition_bars.py.


## Checkpoint 1 — full junk-river v0 numbers (2026-07-19, cached 86% sample)

**15,466 trades** (every +10% cross, gap<5, 9:35–13:00, no filters):
- **Mean +$132/trade (+0.062R), WR 46% — POSITIVE IN ALL THREE ERAS**
  (25H1 +$35 / 25H2 +$159 / 2026 +$184). The naked trigger carries edge
  on the full river — ORB's raw river is negative-mean by comparison.
- Naive 4-slot FCFS book: $654K/19mo model — NOT bookable as-is
  (worstMo −$51K, MDD −$55K; FCFS is the dumbest possible selector).

**Honest caveats (the program's next gates):**
1. LOTTERY CONCENTRATION: median trade −$174; top ~300 of 15,466 carry
   everything (drop-top-300 → negative). Familiar texture, magnified.
2. **EXECUTION IS THE EDGE'S BUDGET**: at 0.6% entry slip 25H1 goes
   negative; at 1.0% the river flips negative overall. Fill realism
   (participation-scaled slippage, minute-volume capacity) is gate #2 —
   nothing ships on 30bps assumptions for fast movers.
3. Era-consistent structure found (filters for v1): trigger ≤10:30
   (+$199 vs +$60 later), consolidation R ≥5% (+$174 vs +$27, 2026
   negative below) — both all-era-positive. vol-surge mild.
4. COMPLETE RIVER (incl. the 3,994 least-traded names, 2026-07-19):
   **17,925 trades, mean +$115/trade, WR 45% — STILL positive all three
   eras (+$39/+$150/+$135)**. The edge survives the full honest sample;
   the softening (132→115) came, as expected, from the thin tail.

**Program gates to ship**: (G2) participation-based fill model;
(G3) v1 selector (early-window + min-R + quality rank, slot-capped) with
era + leave-out + monthly texture; (G4) catalyst/complex/news filters;
(G5) ORB-overlap and account-level interaction; (G6) walk-forward halves
+ live-parity constants pinned in yaml from day one; (G7) paper-shadow
period before real size.


## Checkpoint 2 — GATES 2-5 COMPLETE (2026-07-19): the proven candidate book

**G2 (execution realism)**: participation model (position ≤15% of
trigger-minute $vol; slippage 30→150bps scaled by participation;
sub-$2K capacity dropped). VERDICT: the naked river's edge DIES
(+$115→−$1/trade) — execution is the gate everything must pass.

**G3 (price/volume selectors)**: NO 2025-consistent rule exists in
price/volume space under realistic fills. The 2026-only cells are
regime bait — refused.

**G4 (catalyst taxonomy — transferred from ORB as an out-of-domain
prior)**: THE RESCUE, same as ORB's own history:
- complex-confirmed (uc≥2): +$321/trade, WR 56%, ALL ERAS (+573/+160/+294)
- catalyst (news|uc≥2): +$154/trade all eras; 'neither' cell −$45 (junk river confirmed)

**THE CANDIDATE BOOK** (catalyst-required, trig≤10:30, R≥5%, 4-slot
FCFS, REALISTIC fills): **$288,546/19mo — a second ORB-sized engine from
a DISJOINT universe (gap<5)**:
- **11 of 19 months ≥ $10K — monster months MOST months** (owner's goal)
- **leave-out ROBUST: drop top-10 trades → +$166,680 (58% remains)** —
  NOT a lottery book; WR 50%, ~70 trades/mo
- eras +$107K/+$19K/+$163K (all positive); 2x-impact stress $274K
- July-2026: +$23,420 (the drought month, positive)
- disclosed: MDD −$44K model (2.7× ORB's), worst month −$29.6K,
  early/R thresholds chosen on full sample (catalyst prior is clean)

**G5**: zero symbol-overlap with ORB (disjoint by construction); capital
design needed (8 combined slots vs budget — shared pool decision).

**REMAINING before live**: implementation (new consumer of the
INTRADAY QUALIFIED stream + planner/exits reuse), yaml-pinned constants,
decision-parity hooks, paper-shadow period. The research says the engine
is real; the shadow says when to trust it with money.


## Checkpoint 3 — latency-honest + shadow architecture (2026-07-19)

**LATENCY TEST (entry at NEXT bar open — models the 60s scan cadence —
+ 5% chase guard): $245,824/19mo, eras +$94.9K/+$24.1K/+$126.9K (all
positive; 25H1 per-trade mean IMPROVES to +$277), 12/19 monster months,
negMo 6/19.** 85% of the instant-entry book survives a full minute of
latency — the moves run 30+ min, so detection cadence is NOT a blocker;
the chase guard doubles as a natural worst-slippage filter (drops 129
runaway entries).

**The book is now proven under**: full junk river ✓, participation-
scaled fills ✓, 2x slippage stress ✓, 1-minute latency ✓, out-of-domain
catalyst prior ✓, leave-out (58% ex-top-10) ✓, all-era consistency ✓.

**Shadow architecture (validation-fidelity principle: each environment
validates only what it exercises):**
- S1 signal shadow, LIVE node, no orders (1-2wk): true detection
  latency, real NBBO spread/depth at trigger, quote-implied fill cost,
  hypothetical book vs harness (decision parity pre-trading).
- S2 paper node in parallel: ORDER-LIFECYCLE ONLY (paper fills are
  synthetic — explicitly not fill evidence; the ORB Pre-Stage-0 lesson).
- S3 micro-live (Pre-Stage-0 pattern): $250-500 risk, ≤2/day, −$750
  daily stop, 2-4wk → 20-40 REAL fills → recalibrate the participation
  model → staged ramp under green-check machinery.

## S1 EXTENSION — week 2 (owner-approved 2026-07-24)

Week 1 verdict: instrument validated (parity by construction via
trading/ignition_rules.py + nightly BT-replay agreement line); strategy
unjudgeable at n=15 BT-legal trades — honest week ≈ −$8.8K at BT sizing,
and the BT replay loses the same money on the same days (tape, not
implementation). S1 runs a second week, 7/27–7/31, first week with a
clean instrument from day 1.

**Decision gates (owner-approved, written before the data arrives):**
- GO to S3 micro-live ($100–150 risk/trade, ≤2/day, −$300 daily /
  −$750 weekly kills) when EITHER:
  (a) a cumulative-positive 5-trading-day window exists, OR
  (b) the first ≥+3R trigger prints (the monster the book depends on),
  AND the nightly BT-replay line is clean (no unexplained drift).
- ABANDON (back to research, no live $) if the cumulative honest
  shadow book since 7/20 breaches −$23,000 (worse than the BT's
  worst-ever month, −$22.7K).
- NO parameter changes while measuring. ARM/LOCK question is closed
  (plateau; scratchpad arm_lock_sweep.csv): revisit 1.65 only with 50+
  affected live triggers.
- Known residual parity gap (documented, not a bug): BT sizes on the
  next-bar-after-cross, shadow on its own later entry bar — affects
  marginal-liquidity triggers only (RADX 7/24 class); the nightly
  parity line quantifies it.

Cumulative tracker starts at: −$8.8K (BT-parity, 7/20–7/24).
Decision review: Friday 2026-07-31, ~21:55 UTC, on the week-2 digest.

### Cumulative tracker update (2026-07-30 EOD)
BT-parity daily: 7/27 +$1,171 | 7/28 +$596 | 7/29 -$4,298 | 7/30 -$104.
Cumulative since 7/20: **-$11.4K** (abandon gate -$23K: 49% consumed).
GO gates: no positive 5-day window yet; best trigger +0.49R (RADX) — no
>=+3R monster. Known-gap note: ADVB 7/30 (-$786 shadow-only) is the $2M
EOD-dollar universe lookahead — calibrate a live cum-dollar-at-trigger
proxy from the book (task: 7/31). Eval cap 60->150 after 71 capped on
the hot 7/30 tape.

### GATE EVENT — 2026-08-03 (week 3, day 1)
BT-parity day: **+$25,510 (13 trades)** — the complex-cluster cohort
delivering as researched (CRWV/IONQ/ONDS/NBIS/IREN complexes).
- GO gate (a) MET: 5-day BT-parity window 7/28-8/3 = +$16,609.
- GO gate (b) MET on the BT-parity ledger: CRWU +3.23R.
- Cumulative since 7/20: **+$8,977** (was -$16,533).
- Caveat: shadow captured 9/13 (+$16,843) — 4 misses were the ONE-SHOT
  EVALUATION bug (sighted before level cross, never re-checked:
  NBIG/NBIL/IREX/IREG ≈ $9.0K). Fixed same day (_await_level re-eval,
  +2 tests); deploys 8/4 12:30.
- Recommendation: conditional GO — S3 micro-live from 8/5 if 8/4's
  replay shows clean capture under the fix. Owner decides.

### 8/4 — conditional-GO test PASSED
Level-park fix live: parked symbols re-evaluated every sighting; BT
replay shows ZERO BT-only misses (only diffs = the 2 known benign
classes LNAI pos_lt_2k / SEAT u_dollar_2M). Day: BT -$2,216 (DFNS
stop, shared). Cumulative: **+$6,761**. Gates remain met (5-day window
+$18,095). Efficiency patch same day: price-precondition before
re-fetch (deploys 8/5). **S3 micro-live READY — awaiting owner GO.**

### 8/5 — first PERFECT parity day
BT +$5,288, 4/4 trades shared (FUBO/CRCT/ZTG/OSS) after fixing the
replay's first-record-wins info-staleness (ZTG's parked record carried
has_news=None; merge fix in ignition_bt_replay.py). Level re-eval
price-gate verified: 2 gated re-evals (vs 90 blind 8/4), one converted
(ZTG trigger). Cumulative: **+$12,049**. Gates met. S3 awaiting GO.

### 8/6 — red day, parity holds
BT -$3,168 (5 kept, ALL shared). Shadow extras: 3x pos_lt_2k (known
sizing-moment class — S3 will size at its own entry like the shadow,
so shadow semantics are the live-honest ones) + IREX catalyst-drop =
CASCADE of same class (cohort partner IREG was the pos_lt_2k). No new
bug classes. Cumulative: **+$8,881**. Gates met (5-day +$20,319).
S3 awaiting GO, day 3.

### 8/14 — WEEK-3 RECONCILE + LATE-SIGHTING PARITY FIX
Gap: Claude EOD deep-dives went dark 8/10-8/13 (session idle); mechanical
reports ran throughout. Reconcile found a NEW parity bug on 8/13: shadow
missed 4 BT monsters (CRWU/CWVX chase-skip, SMCL/SMCX r-too-small, ~$8K)
because it computed chase/stop/R from the SCANNER SIGHTING price/minute,
while the BT keys to the ACTUAL trigger bar. Fixed: trigger mechanics
extracted to ignition_rules.trigger_entry_stop() (shared by shadow +
replay logic); shadow _finalize now reconstructs the trigger bar from
fetched bars. Verified: all 4 now trigger. Deploys 8/14 12:30.
BT-parity cumulative since 7/20: ~+$1,300 (abandon gate -$23K NOT
breached; the 8/13 -$8K was mostly the now-fixed miss + a genuine red).
Gates still met. S3 STILL awaiting owner GO (never started — no live $).
