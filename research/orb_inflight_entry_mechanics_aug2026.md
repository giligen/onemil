# ORB entry-mechanics family — drag-reduction study (2026-08-14, parallel program)

Owner order: cut the clean book's −$101.7K loser drag toward −$50K while keeping
>=85-90% of the top-40 winners' P&L, by changing HOW we enter (a sibling agent
owns 9:35 selection). Data: DST-clean `analysis_results/orb_features_20260814_1741.csv`
+ clean-book baseline. **Headline: every hypothesis in this family FAILS the
pass bar (drag falls AND winner retention >=85% in TRAIN/2025H2/2026-OOS).
Best rule = NONE.** The one near-miss (H4 trigger-time cap ~9:45) misses the
TRAIN retention bar by a single trade and adds nothing on the B+ book; a
9:47 variant that would pass was found only with full-sample visibility and is
flagged for forward validation, not passed.

## 0. Tooling and validation

- `scratchpad/entry_harness.py` — extension of the validated
  `research/scripts/orb_clean_harness.py` (original untouched): volume-aware
  physics rebuilt from cache.db (6,918/6,918 pairs, **0 trigger mismatches, 0
  drops**), selection stack verbatim, entry variants applied POST-selection
  (declined entry = slot consumed, NO refill — matches the shipped no-refill
  invariant; entry mechanics cannot change 9:35 selection by construction).
- **Parity gate passed**: touch-mode reproduction of both eval configs is
  byte-identical to the parent harness (max per-trade |diff| = $0.0000):
  - Config A (shipped-clean $100K): n=219, +$96,454, eras −$8,188 / +$13,233 /
    +$91,409, MDD −$20,832. Top-40 winners = +$197,094; drag cohort (all
    non-top-40 trades) = **−$100,640** (the owner's −$101.7K).
  - Config B+ ($10K, refit q40 + pdr11 + N3 + catalystY): n=95, +$11,355,
    MDD −$715, worst month −$346 (matches the re-derivation report).
- Walk-forward discipline: all thresholds derived on **TRAIN = 2025H1 config-A
  trades only** (n=53, 9 top-40 winners, drag −$30,323), frozen, then scored on
  2025H2 and 2026 (OOS). Scripts: `scratchpad/{build_and_validate,train_diag,
  eval_grid,casualties_sens}.py`.
- Fill model for open-entry variants (H1/H2/H3-confirm): next-bar-open +10bps
  slip (sensitivity at 30bps shown; verdicts unchanged). Touch-mode keeps the
  shipped 30bps stop-limit band. Rule M is not applied to confirmed entries
  (its information predates the entry decision); Rule D applies to the first
  full bar of trade life.

## 1. TRAIN-2025H1 derivation (config A; the only window used for tuning)

Baseline TRAIN: n=53, total −$8,188, winners +$22,134, drag −$30,323.

| rule | n | total | winner ret% | dragΔ |
|---|---|---|---|---|
| H1 cc (any bar closes>rh → next open) | 51 | −22,063 | 66.6 | **−6,487 (worse)** |
| H1 cc_first (first breakout bar must close>rh else skip) | 38 | −14,386 | 60.7 | +2,512 |
| H2 retest (±0.2% of rh, green, holds) | 37 | −21,780 | 30.9 | +1,699 |
| H3 vgate X=1.5 / 2 / 3 | 24/15/5 | −1,194/+5,336/+263 | 52.8/46.7/5.8 | +17,451/+25,324/+29,292 |
| H4 by 9:40 / 9:45 / 10:00 | 38/40/50 | +2,457/−354/−4,956 | 84.8/84.8/100.0 | +14,020/+11,208/+3,232 |
| H4 exclude-bar-1 | 34 | −13,334 | 48.5 | +6,247 |
| H5 spread-proxy k=1.5/2/3 | 53 | −8,188 | 100.0 | **+0 (inert)** |

TRAIN texture that drove the freeze: 9:35 instant breakouts = +$5,145 (19
trades) vs ALL later breakouts −$13,334 (34 trades); breakouts after 9:40 =
−$10,645. Frozen for validation: H4 {580, 585, 600}; H3 {1.5, 2} (already
retention-failing but validated for the record); H1 both forms; H2; H5 k=1.5.

## 2. Full-window validation (frozen rules; 2026 = OOS)

### Config A ($100K shipped-clean; baseline +$96,454, drag −$100,640)

| rule | n | total | MDD | winner ret% H1/H2/26 | dragΔ H1/H2/26 | verdict |
|---|---|---|---|---|---|---|
| **H4 by945** | 177 | **+102,414** | −16,499 | **84.8**/94.2/94.0 | +11,208/+7,578/+905 | FAIL (TRAIN ret bar, by 1 trade) |
| H4 by940 | 155 | +97,239 | −17,892 | 84.8/**71.7**/92.1 | +14,020/+11,012/+2,566 | FAIL (kills ZURA — top-5 monster) |
| H4 by1000 | 205 | +95,603 | −17,316 | 100/97.7/94.9 | +3,232/+4,150/**−662** | FAIL (2026 drag rises) |
| H4 nobar1 | 131 | +59,405 | −16,699 | 48.5/43.6/63.5 | +6,247/+20,165/+21,154 | NOGO (de-monsters) |
| H1 cc | 207 | +55,056 | −32,315 | 66.6/92.3/94.8 | **−6,487/−10,705/−6,537** | FAIL (drag WORSENS everywhere) |
| H1 cc_first | 147 | +92,134 | −22,874 | 60.7/92.3/92.5 | +2,512/+6,696/+8,430 | FAIL (TRAIN ret 61%) |
| H2 retest | 169 | +22,350 | −29,987 | 30.9/76.4/57.5 | +1,699/−3,165/+8,148 | NOGO |
| H3 vgate1.5 | 78 | +38,786 | −7,260 | 52.8/59.1/**23.0** | +17,451/+22,102/+30,961 | FAIL (winner kill) |
| H3 vgate2 | 49 | +46,515 | −3,777 | 46.7/36.5/**21.3** | +25,324/+27,801/+39,245 | FAIL (winner kill) |
| H3 vconf2 | 49 | +34,151 | −5,347 | 20.9/33.2/20.9 | +25,295/+25,770/+36,752 | FAIL |
| H5 sk1.5 | 211 | +91,138 | −20,193 | 100/96.5/97.4 | +0/+261/−615 | NOGO (inert) |

### Config B+ ($10K; baseline +$11,355, MDD −$715)

| rule | n | total | MDD | ret% H1/H2/26 | dragΔ H1/H2/26 |
|---|---|---|---|---|---|
| H4 by945 | 79 | +10,503 | −681 | 66.7/100/92.3 | +187/+110/+27 |
| H4 by940 | 71 | +10,651 | −893 | 66.7/100/90.6 | +286/+315/+32 |
| H4 by1000 | 88 | +11,104 | −767 | 100/100/93.9 | +268/+220/−167 |
| H1 cc | 89 | +8,814 | −1,187 | 59.0/90.3/96.2 | −396/−402/−528 |
| H2 retest | 73 | +4,558 | −1,221 | 18.3/77.7/51.4 | −822/−28/+444 |
| H3 vgate2 | 20 | +3,314 | −217 | 66.7/23.7/19.2 | +75/+915/+1,425 |
| H5 sk1.5 | 91 | +11,306 | −756 | 100/100/98.3 | +0/−8/+121 |

**On B+ every rule is neutral-to-negative on total** — B+'s q40 threshold +
pdr11 + catalyst veto already remove most of what entry lateness proxies.
Entry mechanics have no incremental room there.

## 3. Winner casualties (named, config A)

Top-40 winner trigger-minute distribution: 27/40 trigger by **9:37**
(+$163.1K = 83% of winner P&L); all five monsters trigger early —
ANNA 2026-03-20 **9:37**, BNAI 2026-01-23 **9:35**, ANTX 2026-03-09 **9:36**,
BNAI 2026-02-24 **9:35**, ZURA 2025-09-30 **9:41**.

- H4 by945 kills (6): ROLR 2026-04-16 $4,295 (10:13), TUYA 2025-02-20 $3,374
  (9:46), MSTZ 2026-06-09 $2,178 (10:16), SDOT 2025-09-15 $1,642 (9:56),
  MST 2026-02-25 $1,145 (9:50), LIMN 2025-07-24 $1,099 (10:23). Total $13.7K
  (93.0% full retention). **All top-5 monsters kept.**
- H4 by940 additionally kills: **ZURA 2025-09-30 $7,460 (9:41 — top-5
  monster → dead rule per owner rule #4)**, PACS 2025-09-11 $3,176 (9:41),
  CRWU 2026-07-21 $1,231, CWVX 2026-07-21 $1,214.
- H4 nobar1 kills the two biggest trades outright (BNAI ×2 at 9:35) — the
  monsters ARE the instant breakouts; the inverse hypothesis is refuted hard.
- H3 vgate2 keeps only 21% of 2026 winner P&L: winners' trigger-bar volume is
  LOW relative to their own opening range (denominator effect — monster
  mornings already carry huge 9:30-9:34 volume), so the qualifier is
  systematically anti-winner. It is a de-monstering filter wearing a
  drag-reduction costume: drag −$100.6K → −$8.3K but the book drops to
  +$46.5K and the 2026 monster year to +$30.1K.
- H1 cc worst case: converts drag −$100.6K → −$124.4K. Confirmed entries fill
  next-bar-open (systematically higher), widening every loser's
  entry-to-stop distance, while the shipped touchgo already exits weak-close
  breakouts at nearly the same price — close-confirmation is dominated by
  touchgo-post-entry and pays for itself twice.

## 4. Sensitivity (frozen rule ±20%, config A)

H4 cutoff sweep — smooth surface, no cliff:

| cutoff | n | total | ret% H1/H2/26 | dragΔ H1/H2/26 |
|---|---|---|---|---|
| 9:38 | 141 | +92,025 | 85/61/92 | +13,888/+11,772/+1,836 |
| 9:40 | 155 | +97,239 | 85/72/92 | +14,020/+11,012/+2,566 |
| 9:42 | 168 | +101,086 | 85/94/92 | +11,208/+7,794/+1,806 |
| 9:45 | 177 | +102,414 | 85/94/94 | +11,208/+7,578/+905 |
| **9:47** | 184 | +102,969 | **100/94/94** | +8,209/+6,679/+1,985 |
| 9:50 | 190 | +98,298 | 100/94/95 | +4,578/+4,493/+1,985 |
| 10:00 | 205 | +95,603 | 100/98/95 | +3,232/+4,150/−662 |
| 10:10 | 210 | +92,833 | 100/98/95 | −161/+2,866/+1,245 |
| none | 219 | +96,454 | 100/100/100 | 0/0/0 |

The **9:47** row passes every bar (drag falls all 3 windows, ret 94-100%) —
but it was located in this full-sample sweep, not on TRAIN (TRAIN grid held
{9:40, 9:45, 10:00}; 9:47's TRAIN advantage is exactly one trade, TUYA at
9:46). Selecting it now would be the same full-visibility trap the
re-derivation report flags on pdr11/N3. **Status: forward-validation
candidate, not a pass.**

H3 X sweep (1.2/1.6/2.0/2.4/3.0): retention 63/34/33/29/9% (full) — no
region approaches 85%; failure is structural, not a threshold cliff.
H1 slip 10→30bps: totals −$6.5K, retention structure unchanged — verdicts are
slip-robust. H2 tol variants (hold / no-hold) both fail (ret 59/54% full).

## 5. Monthly deltas for the near-miss (H4 by945, config A)

Baseline → by945: 2025-01 +5,337, 2025-02 −3,374 (TUYA), 2025-03 +1,950,
2025-05 +1,839, 2025-06 +1,965, 2025-07 −1,199, 2025-09 −1,493, 2025-11
+4,569, 2025-12 +3,591 (worst month −$13,187 → −$9,596), 2026-01 −1,080,
2026-03 +1,320, 2026-04 −4,294 (ROLR), 2026-06 −4,477 (MSTZ), 2026-07 +1,743.
Ex-top-5 flips −$5,246 → **+$713**; MDD −$20,832 → −$16,499; top5 share
105.4% → 99.3%; months+ 9/20 (unchanged-ish). Real but modest — and the 2026
OOS drag improvement is only +$905, i.e. ~90% of the rule's value sits in
2025. On B+ it is a net −$852 total. This is a 2025-chop rule, not a durable
drag machine.

## 6. Verdict table

| hypothesis | verdict | one-line reason |
|---|---|---|
| H1 close-confirmation | FAIL | cc: drag WORSENS in all 3 windows (next-open fills are worse; touchgo already covers weak closes). cc_first: TRAIN ret 61%. |
| H2 retest entry | NOGO | Winner retention 31/76/57% — monsters never pull back to the range high; forfeits the entire edge. |
| H3 breakout-bar volume qualifier | FAIL | Drag −$100.6K→−$8.3K but winner retention 21-53%; qualifier is structurally anti-monster (denominator effect). No X region passes. |
| H4 trigger-time window | FAIL | Frozen 9:40 kills ZURA (top-5 monster); frozen 9:45 misses TRAIN ret bar by one trade (84.8%); frozen 10:00's drag rises in 2026-OOS. 9:47 passes but is full-sample-picked → forward-validate only. Inverse (exclude bar-1) refuted: 83% of winner P&L triggers by 9:37. |
| H5 trigger-minute spread proxy | NOGO | Inert — post-open bar ranges compress; the proxy exceeds 1.5× the opening-range mean on ~0-4% of trades. Bar data cannot measure spread degradation; would need live NBBO capture. |

**best_rule: NONE.** Family-level read for the owner: the drag does not live
in HOW we enter. Entry timing/confirmation/volume filters either (a) barely
overlap the drag (H4/H5), or (b) cut drag only by also cutting the monster
tail that IS the book (H2/H3), or (c) actively add drag via worse fills (H1).
The one honest crumb: a trigger-time cap in the 9:45-9:47 region (i.e.
shortening the order auto-cancel from 60 min to ~10-12 min) is worth a
forward-validation slot — it is live-trivial (one config value), keeps all
five monsters, and its failure mode is bounded (foregone late trades, no new
risk). Recommend: log-only shadow of "would-have-been-canceled at 9:47" for
4-6 weeks alongside whatever book restarts, then re-decide.

## 7. Caveats

- TRAIN window is 53 trades / 9 winners — a single trade moves retention 5-11pp;
  that is why the 84.8% miss is reported as FAIL rather than rounded to a pass.
- Open-entry fill model (next-bar-open +10bps) is optimistic on thin names;
  since H1/H2 already fail with the optimistic model, verdicts are safe.
- H5 tested only the bar-range proxy named in the brief; a true quote-spread
  gate at trigger time is untestable on cached bars (live NBBO logging would
  be the prerequisite, not more backtesting).
- Slot/concurrency modeled at selection level (as in the parent harness);
  killed entries do not free slots intraday for later candidates — consistent
  with no-refill, slightly pessimistic for gating rules.
- Artifacts: `scratchpad/entry_harness.py` (extended physics + variants),
  `scratchpad/orb_physx_ext_full.pkl` (physics cache),
  `scratchpad/{train_diag,eval_grid,casualties_sens}.py` (derivation /
  validation / sensitivity runs). Parent harness untouched.
