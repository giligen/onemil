# ORB drag-reduction program — SYNTHESIS (2026-08-15)

Owner order (8/14): "cut the losers from 100K to 50K and we have a strategy"
— cut the clean book's ~-$101.5K drag (ranks 41-238) toward -$50K while
keeping >=85-90% of top-40 winner P&L. Six research tracks ran in parallel
(~31 hypotheses); this report re-verifies the survivors independently,
tests their combination (chosen on TRAIN+VAL only, 2026-OOS unveiled once),
runs the adversarial pass, and gives the honest verdict.

**HEADLINE: The combined 3-rule program cuts the drag 44% on the shipped
$100K clean book (-$101.5K → -$56.6K) at 91.9% total top-40 winner
retention, and 53% on the B+ $10K restart book at 100% retention with ZERO
winner casualties. The owner's 50% bar is MET at the B+ scale and NEAR-met
(44%) at $100K. Ex-top-5 floor flips from -$6.9K to +$22.1K ($100K) /
+$1.2K to +$2.9K (B+). What remains (-$56.6K/20mo at $100K) is structural
ticket price: hyper-volatile stop-outs that are feature-identical to the
monsters at 9:35 and path-identical in their first hour — every rule that
cut deeper killed ANNA/BNAI-class trades in some window. 0 of 100 random
equal-rate rule-sets match the combo on either total P&L or retention.**

Survivors combined: **G1** 9:35 selection veto (return_volatility_20d >=
7.106 AND prev_day_range_pct >= 9.226, post-selection, NO refill,
fail-open incl. RV20==0) + **STAG** stagnation time-stop (MFE < +0.25R
within 20 min of fill → exit at market) + **MID** SPY context stop-tighten
(SPY < 9:35-open × (1-0.75%) intraday → all open stops to breakeven).

## 0. Data integrity work done first

- **12-pair winter-bar repair** (owner-granted): 12 symbol-days truncated
  at 20:00 UTC in cache.db repaired from Alpaca SIP (full log in
  `research/orb_inflight_midflight_aug2026.md` §0). 4 of the 12 trades'
  exits changed (eod closes); shipped baseline moves +$96,454 → **+$94,797**
  (-$1,657), B+ -$6. Everything below is on repaired data; the five family
  reports (pre-repair) are ~$1.7K optimistic on the shipped baseline —
  structure unaffected.
- **SPY 1-min gap closed**: the 136/404 missing SPY days (day-meta §6.2
  artifact) fetched fresh to a scratchpad cache; the MID rule is evaluated
  on 404/404 real SPY sessions, not the zero-filled artifact.
- Physics rebuilt from repaired DB: 6,918/6,918 pairs, 0 trigger
  mismatches, 0 drops. Baselines (repaired): shipped $100K +$94,797 / 219
  trades / drag -$101,547 / top-40 +$196,344; B+ +$9,925 / 105 / drag
  -$3,165 / top-15 +$13,090.

## 1. Independent re-verification of the survivors' OOS claims

Each rule re-implemented from its spec (not the family agent's code) and
run through `orb_clean_harness.py` books; exit-sim parity vs the harness =
$0.00 on both configs.

| rule | family claim (OOS-2026) | my reproduction | verdict |
|---|---|---|---|
| STAG (0.25R, 20m) | dragΔ +$1,522, ret 100%, 20 fires | dragΔ **+$1,522**, ret 100.0%, 20 fires, book +$6,809 | **CONFIRMED exactly** |
| G1 (RV20/PDR) | dragrem ~$6.3K, ret 0.974, casualties PACS/KZR/QURE/OMER/SDOT/BKSY | dragΔ **+$4,953-5,444**, ret **97.4%**, same 6 casualties by name | **CONFIRMED** (dollar delta from 12-pair repair + fail-open incl. RV20==0 + 219-trade harness book vs 238-trade pipeline frame) |
| MID (SPY 0.75% → BE) | (new, Part A) dragΔ +$8,729, ret 97.5% | same (own derivation) | walk-forward clean, threshold spec-pinned a priori |

MID is the single biggest honest OOS-2026 drag lever of the whole program
(+$8.7K = 24% of OOS drag), G1 second (+$5.0-5.4K), STAG third (+$1.5K).

## 2. The combined program (combo chosen on TRAIN+VAL ONLY, then one OOS unveil)

Candidates: all subsets of {G1, STAG, MID}. Pre-declared criterion: max
TRAIN+VAL drag reduction s.t. every component passed its own family bar and
no new casualties beyond the components' disclosed ones. Chosen:
**G1+STAG+MID** (TRAIN dragΔ +20,840 / VAL +9,923 — the max in both
windows). Conservative alternative kept for reference: **STAG+MID**
(exit-only, 100% retention in every window, no veto layer).

### OOS unveil + full-timeline results

**Config a — shipped-clean $100K ($3K risk, yaml fit):**

| book | total | n | tWR | MDD | months+ | worst m | ex-top5 | top5 | eras H1/H2/26 |
|---|---|---|---|---|---|---|---|---|---|
| BASE | +$94,797 | 219 | 38.8% | -$21,739 | 9/20 | -$13,362 | **-$6,903** | 107.3% | -8.9K/+12.3K/+91.4K |
| **G1+STAG+MID** | **+$123,780** | 143 | 39.2% | **-$17,043** | 12/20 | **-$8,694** | **+$22,080** | 82.2% | **+11.9K/+12.8K/+99.1K** |
| STAG+MID (alt) | +$113,451 | 219 | 37.0% | -$21,053 | 10/20 | -$12,859 | +$11,750 | 89.6% | +3.3K/+13.0K/+97.2K |

Drag by window (combo): TRAIN -$30.3K → **-$9.5K** (-69%), VAL -$34.9K →
**-$24.9K** (-28%), OOS -$36.4K → **-$22.2K** (-39%). Total -$101.5K →
**-$56.6K (-44.3%)**. Winner retention 100 / 80.0 / 94.9 by window, 91.9%
total. 2025H1 flips six-red-months (-$8.9K) to +$11.9K; all 3 eras
positive.

Monthly (base → combo): 2025: +465→+4,586, +353→+3,726, -6,448→-1,050,
-84→+4,139, -2,269→+205, -955→+296, -816→+1,706, -980→-245, +8,511→+4,228,
+23,715→+22,214, **-4,741→-6,400**, -13,362→-8,694; 2026: +20,444→+23,376,
+16,903→+15,906, +60,885→+62,374, +574→-1,167, -560→-243, +1,501→+3,557,
-5,296→-1,692, -3,042→-3,042. Disclosed costs: Sep-2025 -$4.3K (PACS/SDOT
vetoes), Nov-2025 worsens to -$6.4K (G1 keeps volatile losers by design),
Apr-2026 flips slightly red (QURE veto + CRML wound).

**Config b — B+ $10K (TRAIN-refit q40, pdr11, N3, cvY, $375 risk):**

| book | total | n | tWR | MDD | months+ | worst m | ex-top5 | top5 | eras |
|---|---|---|---|---|---|---|---|---|---|
| BASE | +$9,925 | 105 | 39.0% | -$1,210 | 12/20 | -$413 | +$1,179 | 88.1% | +2.1K/+0.7K/+7.1K |
| **G1+STAG+MID** | **+$11,610** | 81 | 43.2% | **-$657** | 13/20 | **-$308** | **+$2,864** | 75.3% | +2.4K/+1.2K/+8.1K |

Drag -$3,165 → -$1,481 (**-53.2%**). Retention **100/100/100 — zero
casualties of any size**. 4.0 trades/mo (helps PDT headroom). All negative
months shallow (worst -$308). Monthly deltas nonnegative in 14 of 20
months, worst single-month cost -$133.

Month-by-month, B+ + G1+STAG+MID at $10K (owner-requested full breakdown;
same run as the +$11,610 / 81-trade book above; moMDD = intra-month max
drawdown of daily cumulative P&L from the month's 0-floored peak; cumP&L =
cumulative book P&L at month end; 2026-08 is partial, through 08-13):

```
month    combo P&L   n   WR%    avgW    avgL     maxW     maxL    moMDD    cumP&L   base B+   delta
---------------------------------------------------------------------------------------------------
2025-01       +406   4  50.0    +316    -113     +534     -146     -146      +406      +261    +145
2025-02       +526   3  66.7    +268     -10     +450      -10      -10      +932      +312    +214
2025-03        +18   2  50.0     +81     -63      +81      -63      -63      +950      -133    +151
2025-04       +565   2 100.0    +283      +0     +386       +0       +0    +1,515      +565      +0
2025-05       +919   2 100.0    +460      +0     +644       +0       +0    +2,434    +1,032    -113
2025-06        -45   5  40.0     +88     -74     +122     -145     -221    +2,389       +88    -133
2025-07        +94   1 100.0     +94      +0      +94       +0       +0    +2,483      +121     -27
2025-08        -32   2  50.0     +50     -83      +50      -83      -83    +2,451       -32      +0
2025-09       -258   9  22.2     +48     -51      +64     -112     -310    +2,193      -329     +71
2025-10     +1,328   5  60.0    +449     -10     +735      -15      -15    +3,521    +1,314     +13
2025-11        +50   4  25.0    +342     -98     +342     -138     -170    +3,571       +50      +0
2025-12        -31   7  28.6    +275    -116     +283     -205     -318    +3,540      -413    +382
2026-01     +1,662   5  40.0    +927     -64   +1,815      -99     -159    +5,202    +1,410    +252
2026-02     +1,060   4  50.0    +589     -59   +1,026     -100     -118    +6,262      +938    +123
2026-03     +5,449   3 100.0  +1,816      +0   +2,734       +0       +0   +11,711    +5,345    +104
2026-04       +445   3  33.3    +572     -63     +572     -111     -111   +12,156      +481     -36
2026-05       -258   2   0.0      +0    -129       +0     -193     -258   +11,898      -369    +110
2026-06       +107   6  50.0     +55     -19      +70      -45      -57   +12,005      -131    +238
2026-07        -87   9  33.3    +124     -76     +174     -214     -276   +11,918      -276    +189
2026-08       -308   3   0.0      +0    -103       +0     -139     -308   +11,610      -308      +0
---------------------------------------------------------------------------------------------------
TOTAL      +11,610  81  43.2    +428     -73   +2,734     -214     -657   +11,610    +9,925  +1,684
```
(TOTAL row: WR/avg/max over all 81 trades; the -$657 in the moMDD column
is the standard full-timeline MDD of the book. 13/20 months green; every
red month is shallower than or equal to base; the combo never turns a
green base month red except 2025-06 at -$45; biggest single loser across
20 months is -$214, biggest winner +$2,734 = ANNA.)

### Interaction decomposition (redundancy / destruction)

Sum-of-parts dragΔ vs combo (config a): TRAIN +31,753 vs +20,840
(**-10,913 redundancy** — G1 vetoes many of the quiet losers STAG/MID
would have rescued), VAL +11,377 vs +9,923 (-1,454), OOS +15,204 vs
+14,177 (**-1,028, 93% additive**). No destruction anywhere: the combo
beats every component alone in every window; STAG fires drop 20→9 and MID
fires 15→6 on the G1-kept book (the exit rules specialize in exactly the
volatile-loser cohort G1 deliberately keeps). No scratch-exit prevented a
later monster (zero new casualties vs the components' own lists).

## 3. Adversarial pass on the final combo

**±20% one-at-a-time on all five thresholds (config a):**

| knob | total | OOS dragΔ / ret | weak window |
|---|---|---|---|
| FROZEN | +$123,780 | +14,177 / 94.9% | VAL ret 80.0 (G1 known) |
| RV20 -20% (5.68) | +$119,118 | +10,762 / 95.9% | VAL 83.5 |
| RV20 +20% (8.53) | +$111,333 | +13,772 / 91.6% | **TRAIN 72.7 / VAL 64.2** |
| PDR -20% (7.38) | +$120,157 | +15,132 / 94.9% | VAL 80.0 |
| PDR +20% (11.07) | +$128,815 | +16,811 / 93.2% | TRAIN 88.5 |
| stag mfe 0.20R | +$120,633 | +13,206 / 94.9% | VAL 80.0 |
| stag mfe 0.30R | +$121,780 | +15,018 / 92.7% | VAL 80.0 |
| stag win 16m | +$119,185 | +15,599 / 94.9% | **TRAIN 84.3 / VAL 77.8** |
| stag win 24m | +$121,424 | +12,433 / 94.9% | VAL 80.0 |
| spy 0.60% | +$121,297 | +14,177 / 94.9% | **TRAIN 84.1** (TUYA) |
| spy 0.90% | +$123,780 | identical | VAL 80.0 |

Read: **OOS-2026 is robust under every perturbation** (ret 91.6-95.9%,
dragΔ +10.8K to +16.8K, total never below +$111K). Every retention failure
under perturbation is a 2025-window, known single-name cliff, and every
one sits in the TIGHTENING direction (RV20 up, stag window down, SPY
threshold down). Deployment rule: **these thresholds may be loosened,
never tightened**, without a new walk-forward.

**Era stability**: combo eras +$11.9K / +$12.8K / +$99.1K (a) and +$2.4K /
+$1.2K / +$8.1K (b) — all positive, both scales, including the 2025H1
regime that was -$8.9K at baseline.

**Full winner-casualty list (config a, all 7, total -$16.0K of $196.3K):**
PACS 2025-09-11 -$3,176 (veto), CRML 2026-02-03 -$3,162 (MID wound: below
BE at SPY trigger, later recovered), KZR 2025-10-17 -$2,535 (veto), QURE
2026-04-30 -$2,107 (veto), OMER 2025-11-14 -$2,087 (veto), SDOT 2025-09-15
-$1,642 (veto), BKSY 2026-03-31 -$1,247 (veto). **No top-10 trade touched;
ANNA, BNAI×2, ANTX, ZURA, CRCL-trio, IREX all 100% intact. Config b: zero
casualties.**

**Null — 100 random rule-sets at identical intervention rates** (76 random
vetoes + 9 random forced-20-min exits + 6 random BE-conversions per draw):
null totals mean +$55,485 (p95 +$95,487, max +$116,629); null retention
mean 58.9% (p95 78.6%, max 85.3%). **Draws matching the combo: 0/100 on
total, 0/100 on retention, 0/100 jointly.** The program's information
content is decisively non-random. (Component-level honesty stands: the
loserdisc label-shuffle put G1's in-sample magnitude at p≈0.05 — haircut
G1's blended contribution; its OOS-2026 rate is the anchor.)

## 4. Full program table (6 tracks, 31 hypotheses)

Track reports: `orb_inflight_entry_mechanics_aug2026.md`,
`orb_inflight_early_cut_aug2026.md`, `orb_inflight_day_meta_aug2026.md`,
`orb_inflight_sizing_structure_aug2026.md`,
`orb_loser_discrimination_aug2026.md`, `orb_inflight_midflight_aug2026.md`.

| # | track | hypothesis | verdict | one-line reason |
|---|---|---|---|---|
| 1 | entry | close-confirmation (cc) | FAIL | drag WORSENS all 3 windows (worse fills; touchgo already covers) |
| 2 | entry | cc-first-bar | FAIL | TRAIN retention 61% |
| 3 | entry | retest entry | NOGO | ret 31/76/57% — monsters never pull back |
| 4 | entry | breakout-bar volume qualifier | FAIL | ret 21-53%; structurally anti-monster |
| 5 | entry | trigger-time cap 9:40/45/10:00 | FAIL | 9:40 kills ZURA; 9:45 misses TRAIN bar by 1 trade; **9:47 = forward-test only** |
| 6 | entry | spread proxy | NOGO | inert on bar data |
| 7 | early-cut | touchgo re-tune (m=0.7) | FAIL | VAL ret 83.2%; m=0.6 full-visibility only |
| 8 | early-cut | **stagnation 0.25R@20min** | **PASS** | 100% retention everywhere; OOS +$1.5K |
| 9 | early-cut | range re-entry | NOGO | ret 56-89%; everything trades back inside |
| 10 | early-cut | below-entry dwell | NOGO | no TRAIN-eligible parameter |
| 11 | early-cut | VWAP loss | NOGO | ret 41-77%; kills BNAI-class |
| 12 | day-meta | consecutive-loser day stop | NOGO | fires once in 20 months |
| 13 | day-meta | early-day quality read | NOGO | never fires (71% of days are 1-trade) |
| 14 | day-meta | family-crowd cut | NOGO | VAL 65%; kills CRCL-trio crowd-monster mode |
| 15 | day-meta | SPY gap day-skip | NOGO | kills ANNA + ANTX (monsters live on red opens) |
| 16 | day-meta | SPY quiet-range skip | NOGO | keyed off missing data (artifact) |
| 17 | sizing | half-entry + add@0.5R | FAIL | taxes 100% of winners, halves only 50% of losers |
| 18 | sizing | conviction-scaled risk | NOGO | OOS ret 54.6%; composite doesn't rank within book |
| 19 | sizing | progressive day re-risk | NOGO | ret exactly 50%; monster IS the day's first trade |
| 20 | sizing | wide-R skip | FAIL | kills IREX; n=10 evidence; premise refuted |
| 21 | selection | **G1 RV20>=7.1 & PDR>=9.2 veto** | **PASS (partial)** | era-consistent, null-surviving; VAL ret 80% is its cost |
| 22 | selection | G2/G3/G4 volatility siblings | pass-dominated | same story as G1, slightly less dragrem |
| 23 | selection | prev_day_volume atom | DEAD OOS | 2026 ret 0.52 — kills 3 top-10 |
| 24 | selection | range_vs_gap / close-position atoms | DEAD OOS | 2026 ret 0.64-0.67 |
| 25 | selection | premarket-structure atoms | DEAD OOS | 2026 ret 0.45 |
| 26 | selection | all >=50%-drag-cut combos | DEAD OOS | uniformly fail OOS retention — why 50% is unreachable at 9:35 |
| 27 | midflight | earlier BE ratchet 0.4-0.9R | NOGO | **REFUTED: converts ANNA (-$41K)/BNAI (-$27K) to scratches** |
| 28 | midflight | asym 11:30 cut <0.5R | technical PASS | wash economics (LOBO -$1.3K, 2026 net -$7.0K) — excluded |
| 29 | midflight | time-tighten tt60 | FAIL | ±20% shatters it (VAL 45-82%); wounds ZURA -47% |
| 30 | midflight | lower-low scale-out | NOGO | VAL 71.5%; monsters print midday lower-lows |
| 31 | midflight | **SPY-drop stop-tighten 0.75%→BE** | **PASS** | best midflight rule; OOS dragΔ +$8.7K at 97.5% ret |

Survivors: **3 of 31** (G1, STAG, MID-SPY). Flagged for forward-shadow
only: 9:47 trigger-time cap (log-only), touchgo m 0.5→0.6, asym-11:30 cut.

## 5. Recommended configuration + projections

**Rules (all frozen, no refits):**
1. G1 veto at ranking time: require RV20 >= 7.106 AND prev-day-range >=
   9.226; post-selection, slot consumed, NO refill; fail-open when either
   feature is missing/uncomputable (incl. RV20==0 from short history).
2. STAG: if running MFE < entry + 0.25×(range) at the first bar-close >=
   20 min after fill → exit at market.
3. MID: SPY 1-min low <= 9:35-open × 0.9925 → amend every open ORB stop to
   max(current stop, entry). One-way, day-latched.

**Projections (20-month resim, repaired data; NOT a forecast — anchor
forward expectations on the OOS-2026 column):**

| scale | total | drag | drag cut | ex-top5 | MDD | months+ | worst m | top5 share | retention |
|---|---|---|---|---|---|---|---|---|---|
| $100K shipped | +$123,780 | -$56,607 | **-44.3%** (OOS -39%) | +$22,080 | -$17,043 | 12/20 | -$8,694 | 82.2% | 91.9% (VAL 80.0) |
| B+ $10K | +$11,610 | -$1,481 | **-53.2%** | +$2,864 | -$657 | 13/20 | -$308 | 75.3% | 100% |

**Preconditions before ANY restart (unchanged from the rederivation plan,
plus program-specific):**
1. Fix the order-placement latency regression to <=5s (monster-carried
   book; a missed monster fill flips a quarter).
2. >=1 month paper/micro-size forward validation of ALL frozen thresholds
   (B+ z-params/cutoffs + G1 RV20/PDR + stag 0.25R/20m + SPY 0.75%) —
   the combo's thresholds may be LOOSENED but never tightened without a
   new walk-forward (§3 sensitivity asymmetry).
3. Nov-2026 first-EST-week live-vs-BT parity audit (DST class of bug).
4. Live wiring + parity tests for the two exit rules: stag timer keyed to
   actual fill (reuse the touchgo late-fill guard pattern), SPY 1-min
   subscription + BE amendment path through StopMonitor; G1 computes from
   the same daily-bars fetch the PDR veto already uses (zero new data
   dependencies).
5. Optional log-only shadows: 9:47 auto-cancel, touchgo m=0.6, asym-11:30
   cut — free forward evidence, no capital risk.

## 6. The honest NO-GO section

- **The 50% bar as stated (on the $100K shipped-clean book): NOT met.**
  44.3% blended, 39% in OOS-2026. The residual -$56.6K/20mo is
  concentrated in hyper-volatile stop-outs (JLHL/EHGO/CAPR/CYCU class,
  RV20 40-500+) that are feature-identical to ANNA/BNAI at 9:35 (loserdisc
  §5) and path-identical through their first hour (midflight H1/H3
  refutations). Every deeper cut tested in 31 hypotheses killed monsters
  in some window. **This is the ticket price of the lottery, not
  inefficiency.** If the owner requires a strict 50% at $100K, the answer
  is NO-GO and the structural reason is above.
- **At the B+ restart scale the bar IS met** (-53%, zero casualties, MDD
  -$657 = 6.6% of the $10K book) — and B+ is the config the rederivation
  report already recommends for any restart. The actionable statement:
  restart, if it happens, is B+ + this 3-rule stack.
- What the program does NOT change: the book remains monster-carried
  (top-5 = 75-82% of P&L even after the floor-raise). A monster-less year
  is still roughly breakeven-to-slightly-positive (ex-top5 +$22.1K/20mo at
  $100K ≈ +$13K/yr). The combo raises the floor and cuts the trough; it
  does not create a high-WR machine — 12-13/20 positive months, not 70%.
- Evidence-quality flags to carry forward: G1 shuffle p≈0.05 + era-decay
  (blended flatters ~2x vs OOS rate); STAG+MID are 15 exit events on the
  combo book (sign-consistent 19/21 on the full book, but thin); VAL-2025H2
  winner retention is 80.0% — the low-RV20 biotech-pop casualty texture
  (PACS/KZR/OMER class) WILL recur; combination assembled from survivors
  of 31 attempts — the null says it isn't luck (0/100), but only the
  forward month is real out-of-sample for the stack as a whole.
- 2026-08 (partial month, -$3,042) is untouched by every rule — the
  current bleed is not in the drag cohorts these rules address.

## 7. Artifacts

Scratchpad (`/tmp/.../scratchpad/`): `repair_12_pairs.py` (DB repair log),
`build_spy_trig.py` + `spy_trig_cache.pkl` (404/404 SPY triggers),
`midflight_harness.py` (overlay sim), `run_midflight.py` / `run_mf_sens2.py`
(Part A grids), `syn_lib.py` / `syn_step1_verify.py` (independent
re-verification), `syn2_combo.py` (stage A/B combo protocol),
`syn3_adversarial.py` (sensitivity + null), `midflight_sels.pkl`,
`mf_analysis.pkl`. Pre-repair physics preserved as `*.pre_repair`.
Nothing in trading/ or orb.yaml touched; nothing committed; ORB live
remains at zero.
