# ORB in-flight family: MIDFLIGHT MANAGEMENT (minute 20 → close) — 2026-08-15

Re-run of the drag-reduction program's lost MIDFLIGHT family (original agent
died without writing structured output; this is a fresh derivation, same
pre-registered methodology as the sibling reports). Owner goal: cut the clean
book's ~-$101.5K drag (ranks 41+) toward -$50K keeping >=85-90% of top-40
winner P&L, via rules that act DURING the trade after the touchgo/stagnation
window.

**Family verdict: one clean survivor — H5 SPY context stop-tighten (SPY drops
>0.75% from its 9:35 level intraday → move all open ORB stops to breakeven).
It passes the full pre-registered bar on both configs (drag falls + retention
>=85% in all three windows), kills zero monsters, and is the single biggest
honest OOS-2026 drag lever found by the entire program (+$8.7K, 24% of OOS
drag) at a cost of one wounded mid-winner. H1 — the program's highest-prior
hypothesis (earlier breakeven ratchet) — is decisively REFUTED: monsters
routinely revisit breakeven after +0.5-0.9R MFE before running; the ratchet
converts ANNA (-$41.1K) and BNAI (-$27.3K) into scratches. H3 passes point
estimates but sits on a sensitivity knife-edge (adjacent params collapse VAL
retention to 45-82%) and wounds ZURA — dead by the program's own overfit
test. H2 asymmetric 11:30 cut is a technical pass with wash economics.
H4 scale-out fails validation.**

## 0. Data repair performed first (owner-granted)

12 winter symbol-days in cache.db `intraday_bars_1min` were truncated at
20:00 UTC (old-vintage EDT-window fetches missing 15:00-16:00 ET). Repaired
before any resims: fresh Alpaca SIP 1-min bars fetched per
`ignition_shadow_report.py::day_bars` pattern, old rows DELETED for exactly
those (symbol, date) pairs, full 9:30-16:00 ET sessions INSERTED
(`scratchpad/repair_12_pairs.py` logs before/after counts):

```
AMPX 2025-01-10 355->391   ARQQ 2025-01-16 363->386   NTRB 2025-01-23 346->385
TNXP 2025-02-07 384->386   QMCO 2025-11-14 329->345   AIRJ 2025-12-11 295->289
FJET 2025-12-24 339->310   DFLI 2026-01-06 278->304   MIGI 2026-01-13 193->205
RGTU 2026-01-22 227->229   KRRO 2026-01-28 203->244   TOPS 2026-03-03 243->246
```
(count drops = removed premarket rows; every pair now covers the full session)

**Exit changes from the repair: 4 of the 12 trades changed** — AMPX
2025-01-10, ARQQ 2025-01-16, QMCO 2025-11-14, AIRJ 2025-12-11, all `eod`
exits whose true 15:45 close is lower than the truncated ~15:00 close. The
other 8 (locks/stops/touchgo, all resolved before 15:00) are identical.
Book impact: **shipped-clean $100K baseline +$96,454 → +$94,797 (-$1,657);
B+ -$6.** All numbers below are on repaired data; sibling family reports
(written pre-repair) are ~$1.7K optimistic on the shipped baseline and
unaffected in structure. Physics rebuilt from the repaired DB: 6,918/6,918
pairs, 0 trigger mismatches, 0 drops.

## 1. Method

- Harness: `research/scripts/orb_clean_harness.py` imported unmodified;
  overlay exit simulator in scratchpad `midflight_harness.py` (baseline
  XCfg with no overlay is byte-identical to the parent `simulate_exit` —
  verified $0.00 parity via the synthesis-stage independent sim).
- SPY data for H5: cache.db lacks SPY 1-min bars on **136/404 book days**
  (the day-meta report's §6.2 artifact). All 136 fetched fresh from Alpaca
  SIP into a scratchpad-only cache (`build_spy_trig.py`; no DB writes).
  H5 triggers computed on 404/404 days — the artifact does NOT affect this
  family. SPY dropped >0.75% intraday from its 9:35 open on 102/404 days.
- Configs: (a) shipped-clean $100K yaml-fit: 219 trades, +$94,797, top-40
  winners +$196,344, drag (ranks 41+) **-$101,547**; (b) B+ $10K
  (TRAIN-refit q40, pdr11, N3, cvY, $375 risk, uniform, no PM): 105 trades,
  +$9,925, top-15 +$13,090, drag -$3,165.
- Exit overlays don't touch selection → identical trade set, paired
  per-trade comparison, pipeline-integrated by construction (no-refill
  invariants unchanged).
- Walk-forward: parameter grids fixed a priori from the task spec;
  TRAIN=2025H1 choice (max TRAIN drag reduction s.t. TRAIN retention >=85%),
  frozen, then VAL=2025H2 / OOS=2026. Pass bar: drag falls AND retention
  >=85% in ALL THREE windows on both configs, no ANNA/BNAI-class casualty,
  no ±20% sensitivity cliff.

Baseline (repaired): shipped a: MDD -$21,739, months+ 9/20, worst
-$13,362, top5 107.3%, ex-top5 -$6,903, eras -8,938/+12,327/+91,409.
B+: MDD -$1,210, months+ 12/20, worst -$413, top5 88.1%, ex-top5 +$1,179.

## 2. H1 — Earlier breakeven ratchet (+0.5R/+0.75R MFE → stop to BE, keeping the 1.75R→+0.5R lock) — **NOGO (refuted, monsters killed)**

The program's highest-prior hypothesis. It is wrong about the drag AND wrong
about the winners:

| variant (config a) | total | ret TRAIN/VAL/OOS | dragΔ total | monster casualties |
|---|---|---|---|---|
| BE@0.50R | +$18,831 | 77.8/47.6/**29.1** | +44,090 | **ANNA -41,086, BNAI -27,271, BNAI2 -7,728** |
| BE@0.75R | +$76,231 | 77.8/64.8/77.0 | +32,203 | **BNAI -27,271** |
| BE@0.40R (sens) | +$5,801 | 66.1/34.1/25.7 | +44,235 | ANNA+BNAI+BNAI2 |
| BE@0.60R (sens) | +$72,108 | 77.8/56.9/69.9 | +40,826 | BNAI |
| BE@0.90R (sens) | +$65,355 | 77.8/64.8/77.0 | +21,327 | BNAI |
| BE@0.50R pess-arm | +$33,766 | 77.8/47.6/35.2 | +51,297 | ANNA+BNAI |
| BE@0.75R pess-arm | +$83,437 | 77.8/64.8/77.0 | +39,410 | BNAI |

- **No variant is TRAIN-eligible** (retention 66-78% vs 85% bar) — the rule
  dies before validation even starts, at every MFE level 0.4-0.9R.
- The drag conversion works exactly as hypothesized (stops → ~0R scratches,
  drag -$101.5K → -$57.5K at 0.5R — the family's largest raw drag cut) but
  the monsters do the SAME thing early: ANNA 2026-03-20 touched +0.5R and
  then traded back through breakeven before its +$41K run; BNAI 2026-01-23
  did it from as high as +0.9R. Only the shipped 1.75R arm clears their
  early chop. B+ mirrors it (ANNA/BNAI casualties, ret 14-83%).
- Pessimistic same-bar arming (defer arm to next bar) changes nothing —
  the BE revisits are real multi-bar moves, not intra-bar artifacts.
- Verdict for the parent program: **the drag cohort cannot be scratched out
  at the minute scale without scratching the monsters**. This closes the
  "convert full stops to 0R" axis for good; the only BE-move that survives
  anywhere in the program is H5's, which conditions on the MARKET (SPY
  collapse) rather than on the trade's own path.

## 3. H2 — Asymmetric 11:30 loser-cut (at 11:30 ET exit only positions below +0.5R; winners run) — **technical PASS, wash economics — do not ship alone**

| config | ret TRAIN/VAL/OOS | dragΔ TRAIN/VAL/OOS | total | MDD |
|---|---|---|---|---|
| a | 92.9/95.0/92.1 | +7,383/+5,352/+3,103 | +$96,637 (+1,840) | -$17,000 (better) |
| b | 100/100/93.6 | +183/+213/+274 | +$10,025 (+100) | -$965 (better) |

- Passes the pre-registered bar on both configs (unlike the symmetric 11:30
  timebox, which killed monsters — the asymmetry does its job: all top-5
  kept 100%).
- **But the economics are a wash**: 42 fires on config a, drag cut +$15.8K
  paid back -$13.9K in winner damage (ROLR -4,277, CRML -3,875, BKSY
  -1,950, KZR -1,616, URGN -1,525, SDOT -755). Leave-one-best-fire-out
  (CYCU 11-11 +3,159) flips the net to **-$1,319**; on B+ LOBO is -$39.
  2026 era net is **-$6,999** (dragΔ +3.1K but winner cost -10.1K) — in the
  monster year the rule costs money.
- What it does buy: risk shape — MDD -21.7K→-17.0K, worst month
  -13.4K→-9.7K, and the biggest single drag saves (CYCU, BILI, AMPY, YSXT)
  are late-day bleeders nothing else catches.
- Sensitivity: plateau [11:30, 12:00] × [0.4R, 0.6R] all pass; **11:00
  fails VAL at 59.9%** (the CRCL trio 2025-10-02 sat below +0.5R at 11:00
  and above it at 11:30 — cutting 30min earlier amputates +$16.5K of
  crowd-monsters). 11:45 interior passes (92.4/92.2/92.6). Never deploy
  earlier than 11:30.
- Verdict: keep as a documented option for a risk-shape-first owner;
  NOT selected for the combined program (LOBO-negative, OOS-negative net).

## 4. H3 — Time-tightening (after 60/90/120 min, stop → max(entry-0.5R, low of last 30 min)) — **FAIL (sensitivity knife-edge + wounds ZURA)**

TRAIN-only selection picks tt60 (TRAIN dragΔ +9,755 at ret 93.2%; tt90
+5,597/103.8%, tt120 +8,032/86.8%). Frozen tt60 then reads
**93.2/88.5/97.8 retention, dragΔ +9,755/+8,109/+4,096, total +$107,014,
MDD -$17,421** — a nominal pass with the family's biggest point-estimate
value (+$12.2K over baseline; B+ +$261, ret 92.6/91.4/99.0). But:

- **±20% sweep shatters it**: tt48 TRAIN ret 81.1 (fail); tt72 VAL ret
  **45.2%** (CRCL trio + ZURA all stop out); floor -0.4R VAL ret **63.8%**;
  lookback 24min VAL ret **81.8%**; only floor -0.6R and lookback 36
  survive. 4 of 6 perturbations fail the bar — the same sign-flipping-cliff
  texture the day-meta family used to kill its depth-qualified crowd cut.
  The rule works only at exactly (60min, -0.5R, 30min lookback).
- **Wounds a top-5 monster**: ZURA 2025-09-30 +$7,460 → +$3,977 (-47%) in
  every passing variant (ZURA consolidated below its 60-min lows before its
  afternoon run). Also CRCA/CCUP 12-16 (-$1.3K/-$1.1K), URGN, ARQQ, IREX,
  IRE, EDBL, BKSY — 9 named winners damaged.
- 61 trades change; the big saves (CYCU +3,159, IDN +2,518, BAIG +1,308)
  overlap heavily with what H5 and the stagnation stop already catch (CYCU,
  CAPR, AMPY, RGTX, BILI, YSXT, GALT, SGML appear in both lists).
- Verdict: NOGO. The point estimate is a local optimum on 20 months of
  data, not a rule.

## 5. H4 — Lower-low scale-out (first 30-min lower-low after 60 min → sell half, rest keeps lock) — **NOGO**

- Config a ll60/30: retention 101.5/**71.5**/90.6 — fails VAL hard, and the
  casualty list is exactly the monster roll: ANNA -4,781, BNAI -3,645,
  CRCL trio -8,720 combined, ZURA -1,853, BNAI2 -1,145 (all `half_eod` —
  monsters print 30-min lower-lows mid-day and then keep going).
- All four sens variants (48/72 min, 24/36 lookback) fail VAL at 64-78%;
  ll60/24 also fails OOS-adjacent (82.2%). B+ fails TRAIN/VAL (78-88/64-78).
- Total P&L -$8.4K vs baseline on a. Structurally the same lesson as H1:
  the winners' paths are indistinguishable from losers' at these scales.

## 6. H5 — SPY context stop-tighten (SPY < 9:35 open × (1-0.75%) intraday → all open stops to breakeven) — **PASS (family best rule)**

| config | ret TRAIN/VAL/OOS | dragΔ TRAIN/VAL/OOS | total | MDD | months+ |
|---|---|---|---|---|---|
| a | 100/100/97.5 | **+9,353/+1,133/+8,729** | **+$110,851** (+16,053) | -$20,606 | 11/20 |
| b | 100/100/100 | +308/+127/+467 | **+$10,828** (+902) | -$755 | 14/20 |

- Drag falls in all three windows on both configs; retention never below
  97.5%. Total drag: -$101.5K → -$82.3K (**-19%**). OOS-2026 dragΔ +$8,729
  is ~24% of OOS drag — the biggest honest OOS drag reduction of any rule
  in the six-family program (G1 gate: +$5.4K; stagnation: +$1.5K).
- **Fire anatomy (config a): 15 trades change, 13 helped (+$19.3K), 2 hurt
  (-$3.2K)**. Saves are hyper-vol stop-outs converted to ~0R scratches:
  EHGO 06-25 +3,576, CAPR 03-04 +2,220, AMPY 04-28 +2,191, RGTX 07-27
  +1,871, BIYA 07-31 +1,718, IDN +1,563, YSXT +1,450, BILI +1,441, plus 5
  losing-eods cut (SGML, GALT, CENX, NSP, BW). Casualties: **CRML
  2026-02-03 +$3,112 → -$50** (was below BE when SPY triggered, recovered
  later — the one real wound) and MRAL -61. Zero top-5 damage: monsters are
  far above breakeven by the time SPY breaks (and per the day-meta finding
  they run on SPY-red mornings — but red-open ≠ intraday collapse; the BE
  floor only binds trades that are already failing).
- Not single-trade-carried: leave-one-best-fire-out +$12,477 (a) / +$664
  (b). Positive in 10 of 11 months where it fires; both hurts and 13 saves
  spread over 2025-02 → 2026-07. B+: 6 fires, all positive, zero
  casualties, months+ 12→14.
- Sensitivity: 0.90 is **identical** to 0.75 (no book trade sits between
  the thresholds — wide plateau on the deep side); 0.60 fails TRAIN ret
  84.1% (TUYA 02-20 killed) — the shallow edge is real. ±20% verdict:
  passes at +20%, fails at -20% by one trade; treat 0.75 as the floor,
  never tighten below it.
- Event count honesty: 15 fires/20mo at $100K scale (6 at B+) is thin —
  this is the strongest surviving rule of a weak family, not a statistical
  slam-dunk. Its defense is sign-consistency (13/15), era-consistency
  (2025 +$10.5K, 2026 +$5.5K), zero-monster structure, and a mechanism
  (idiosyncratic gappers below breakeven don't survive a broad-market
  -0.75% air pocket) that matches the book's known SPY texture.
- TRAIN-only discipline: threshold 0.75 was spec-pinned a priori, TRAIN
  eligible (dragΔ +9,353, ret 100%), frozen → VAL/OOS pass. Clean.
- Live wiring sketch (NOT done here): orb_engine already streams bars;
  subscribe SPY 1-min, anchor = 9:35 bar open, on first bar with
  low <= anchor×0.9925 → `stop_monitor` amend all ORB stops to
  max(current_stop, entry). One config value + parity test. Note the BT
  convention arms stops on the trigger bar's close-time; live should apply
  on bar close to match.

## 7. Family verdict + program handoff

- **best_rule: H5 spy_tighten(0.75%, BE)** — the only midflight rule that
  passes everything with no texture caveats beyond event count.
- H2 asym-cut 11:30<0.5R: technical pass, wash economics, LOBO-negative —
  available if the owner wants pure risk-shape, excluded from the combo.
- H1/H3/H4: dead as specified above. H1's refutation is the family's most
  valuable negative result: **stops cannot be converted to scratches off
  the trade's own path** — every own-path early-BE variant in this program
  (H1 here, sizing-family H1, early-cut H3/H4/H5) kills monsters, while
  the two surviving BE-class rules condition on external/absence signals
  (SPY collapse; 20-min no-progress).
- Interaction caveat: H5 was validated on top of shipped touchgo only. Its
  overlap with G1 (both love hyper-vol losers: CAPR/EHGO-class) and with
  the stagnation stop is measured in the synthesis report
  (`research/orb_drag_program_aug2026.md`).
- Artifacts (scratchpad): `midflight_harness.py` (overlay sim + multi-mode
  combos + stag parity), `run_midflight.py` (grid), `run_mf_sens2.py`
  (±20% + anatomy), `repair_12_pairs.py`, `build_spy_trig.py`,
  `spy_trig_cache.pkl`, `midflight_sels.pkl`. Nothing in trading/ or
  orb.yaml touched; nothing committed.
