# ORB loser-side discrimination study (2026-08-14, post-DST-clean rebuild)

Owner order (8/14): "cut the losers from 100K to 50K and we have a strategy."
Target: a selection-side discriminator, computable at 9:35 ET with
live-available information, that cuts the drag cohort's −$101.7K roughly in
half while retaining ≥85-90% of the top-40 winners' P&L, validated
out-of-sample.

**One-line verdict: a real, era-consistent, null-test-surviving discriminator
exists — multi-day volatility (RV20 ≥ 7.1% & prev-day range ≥ 9.2%) — but it
cuts the drag ~33-36% blended and only ~16-17% in OOS-2026, not 50%. The
50% bar is unreachable with 9:35 information: every candidate that cut more
killed monster winners out-of-sample. Partial-GO at best; the remaining
~$65K of drag is irreducible high-volatility stop-outs that are
feature-identical to the winners at 9:35.**

Data: `analysis_results/orb_static_lock_trades.csv` (DST-clean shipped book,
238 trades, +$96,642, regenerated 8/14 from
`orb_features_20260814_1741.csv`). Resim:
`research/scripts/orb_clean_harness.py` (validated to $3 vs pipeline).
Premarket bars: fresh Alpaca fetch, 238/238 coverage (scratchpad
`loserdisc/pm_bars_cache.json`). All analysis scripts in scratchpad
`loserdisc/` (build_frame / univariate / sweep / derive / nulls / stack /
refill_test).

---

## 1. Drag-cohort characterization

Rank structure (by `_sized_pnl`, $100K/$3K-risk shipped book):

| cohort | n | P&L | 2025H1 | 2025H2 | 2026 |
|---|---|---|---|---|---|
| top-5 | 5 | +$101,700 | 0 | 1 (+$7.5K) | 4 (+$94.2K) |
| ranks 6-40 | 35 | +$96,602 | 10 (+$24.0K) | 13 (+$39.9K) | 12 (+$32.7K) |
| drag (41-238) | 198 | **−$101,660** | 51 (−$31.5K) | 60 (−$33.4K) | 87 (−$36.8K) |

Drag is era-uniform (≈−$32-37K per era) — this is not one bad regime.

**Exit anatomy** (the drag *is* the stop bucket): top-40 winners exit 95%
`eod`, 5% `lock`, 0% `stop`. Drag: 76 stops (−$104.5K), 50 tag_bb
(−$16.9K), 1 tag_b1, partially offset by 33 locks (+$16.5K) and 38 small
eods (+$3.5K).

**Univariate separation at 9:35** (AUC of top-40 vs drag; full table in
scratchpad `loserdisc/univariate.csv`). Nothing is strong — best AUCs
0.60-0.64 — but a coherent, era-consistent family emerges:

| feature | AUC | per-era AUC (H1/H2/26) | direction |
|---|---|---|---|
| spy_3d_range_pct | 0.637 | 0.53/0.61/0.77 | higher better (H1 weak) |
| avg_daily_range_pct_20d | 0.624 | 0.74/0.50/0.63 | higher better |
| prev_day_range_pct | 0.623 | 0.72/0.57/0.59 | higher better (consistent) |
| bars_green_in_range | 0.618 | 0.60/0.65/0.60 | higher better (consistent) |
| range_return_pct | 0.617 | 0.59/0.76/0.53 | higher better |
| gap_over_prev_range | 0.378 | 0.31/0.41/0.40 | LOWER better (consistent) |
| prev_day_close_position | 0.407 | 0.43/0.41/0.40 | LOWER better (consistent) |
| return_volatility_20d | 0.604 | 0.79/0.47/0.57 | higher better |

Premarket structure (fresh bars, 11 engineered features: pm range, VWAP
distance, volume acceleration, pm-high vs range-high, pm-range vs prev-day
range, dollar-vol per gap-point…): **weaker than the daily-bar family** —
best deviations 0.08-0.11 (pmx_last_vs_rh 0.393, pmx_range_vs_prev_range
0.423), era-shaky. The already-known pm_dollar_vol level: AUC 0.601 with a
dead 2025H2 (0.50).

The theme: **winners come from stocks that were already multi-day volatile;
the harvestable drag is "quiet stocks having their first loud morning."**
Range internals, gap size, SPY 5-min, news flags, day-of-week, price level:
all ≤0.54 or era-inconsistent.

## 2. Gate derivation (walk-forward, TRAIN = 2025H1 only)

Frozen procedure (also reused verbatim for the label-shuffle null): atoms =
(feature, direction, TRAIN-quantile threshold q10-q90) with TRAIN retention
≥0.90/0.95, TRAIN drag-removal ≥$8K, pass-rate 0.30-0.95; rank atoms by
TRAIN drag-removal; AND-combine top atoms into pairs/triples under the same
TRAIN constraints; freeze top candidates; only then look at VAL=2025H2 and
OOS=2026. Pass bar: drag-removal > 0 AND retention ≥0.80 in all three
windows, total retention ≥0.85. TRAIN has only 61 trades / 10 winners —
max 2-3 hard-threshold conditions, no fitted models.

**TRAIN-favorites that died out-of-sample** (the discipline working —
full tables in `loserdisc/derived_candidates{,_B,_C}.csv`):

- `prev_day_volume_vs_20d ≤ 2.3` — TRAIN's #1 atom ($16.3K TRAIN
  drag-removal). OOS-2026 retention **0.52** — kills 3 top-10 trades. Dead.
- `range_vs_gap ≤ 0.91`, `prev_day_close_position ≤ 0.44` — TRAIN-clean
  (ret 1.000), 2026 retention 0.67/0.64, each kills 2 top-10. Dead.
- `rvol_20d ≤ 0.23` — 2026 drag-removal *negative*. Dead.
- PM atoms (`pmx_range_vs_prev_range ≤ 1.17`, `pmx_high_vs_rh ≤ 1.05`)
  qualified on TRAIN, 2026 retention 0.45 (kill 2 top-10). Dead.
- Every ≥3-condition combo and every combo containing the above: dead the
  same way. Deeper cuts (50%+ of drag in-sample) uniformly fail OOS
  retention — **that is why the 50% target is unreachable.**

**Frozen finalists that pass all three windows** (thresholds are TRAIN
quantiles, verbatim; per-window = drag-removed / winner-retention):

| gate | 2025H1 | 2025H2 | 2026 (OOS) | total dragrem | tot ret | top-40 kills (top-10) |
|---|---|---|---|---|---|---|
| **G1: RV20≥7.106 & PDR≥9.226** | $17.7K / 1.000 | $12.5K / 0.801 | $6.3K / 0.974 | **$36.5K (36%)** | 0.935 | 6 (0) |
| G2: RV20≥7.106 | $11.8K / 1.000 | $11.5K / 0.801 | $7.2K / 0.974 | $30.5K (30%) | 0.935 | 6 (0) |
| G3: ADR20≥8.581 & PDR≥9.226 | $17.1K / 1.000 | $11.7K / 0.801 | $5.8K / 0.983 | $34.5K (34%) | 0.942 | 5 (0) |
| G4: ADR20≥8.581 | $10.4K / 1.000 | $10.6K / 0.801 | $6.9K / 0.983 | $28.0K (28%) | 0.942 | 5 (0) |

All four are the same story (RV20/ADR20 correlate ~strongly): **require a
multi-day volatility fingerprint.** RV20 = stddev of prior-20d
close-to-close returns ×100 (`study_orb_features.py:312`), computable live
from the same daily-bars fetch the PDR veto already uses. Zero lookahead.

With live fail-open semantics (missing 20d history → keep; 4 trades,
−$2.6K): G1 total dragrem $33.1K (33%), retention unchanged 0.935.
Alternative deliberate rule "missing history → veto" (can't certify RV20)
is worth $3.4K more but must be an explicit owner decision.

**The honest era-decay**: G1's drag cut is 56% → 38% → **17%** across
H1/H2/2026. Reason (measured): the drag cohort itself drifted volatile —
drag median RV20 8.1 → 10.5 → 9.5, PDR 11.8 → 14.4 → 13.6. The shipped
pipeline's own vetoes plus regime change already squeezed out quiet names
by 2026. Forward expectation should anchor on the OOS ~16-17%, not the
blended 33-36%.

## 3. Null tests

**(a) 100 random gates at G1's pass-rate (0.634)**: raw drag-removal mean
$37.1K — i.e. on raw dollars-removed a random gate ties G1 (removing 37%
of trades removes ~37% of drag). The entire information content is in the
joint: G1 retention 0.935 vs random mean 0.619, p95 0.818, **max 0.903 in
100 draws**. Draws matching G1's drag-removal at retention ≥0.85: 2/100
(p≈0.02). Net book improvement: G1 +$23.7K vs random p95 +$3.4K.

**(b) 100 era-stratified label shuffles, full derivation procedure rerun**:
51/100 shuffled worlds let the procedure find *some* gate passing the full
bar — the procedure is flexible; a "passing gate" per se means little.
Passing-gate size: median $9.9K, p95 $34.3K; shuffles ≥ G1's $36.5K:
**5/100 (empirical p ≈ 0.05, borderline)**. Read: the effect is real but
its in-sample magnitude sits at the edge of what selection flexibility can
manufacture — haircut the blended number hard (the era-decay independently
says the same).

## 4. Winner casualties (by name — every finalist)

G1 kills 6 of the top-40, none of the top-10, none of the monsters
(ANNA, BNAI×2, ANTX, ZURA, CWVX, CRCA, CCUP, CRCG, IREX all retained):

| rank | symbol | date | P&L | window | RV20 | PDR |
|---|---|---|---|---|---|---|
| 15 | PACS | 2025-09-11 | +$3,176 | VAL | 6.7 | 8.1 |
| 20 | KZR | 2025-10-17 | +$2,535 | VAL | 3.4 | 8.9 |
| 25 | QURE | 2026-04-30 | +$2,108 | OOS | 5.2 | 8.9 |
| 28 | OMER | 2025-11-14 | +$2,087 | VAL | 3.5 | 10.2 |
| 33 | SDOT | 2025-09-15 | +$1,642 | VAL | 7.0 | 11.0 |
| 39 | BKSY | 2026-03-31 | +$1,247 | OOS | 6.2 | 11.0 |

(G3/G4 kill the same list minus BKSY, plus FLNC +$1,084 under fail-open on
the stacked book.) Casualties cluster in 2025H2 — that's the 0.801 VAL
retention, sitting exactly at the pass bar. These are all
low-RV20 biotech/smallcap pops — the gate's false negatives are real and
concentrated in one texture.

## 5. Pipeline-integrated stacked results (harness resim, fail-open semantics)

Gate applied **post-selection, no refill** (same invariant as PDR/catalyst
vetoes). Refill tested and rejected again on clean data: pre-ranking G1
with slot refill → total +$122.7K but MDD −$22.5K (worse than baseline),
months+ 11/20, worst month −$8.4K, and −$8.3K on OOS-2026 — gross-positive,
risk-negative, in-sample-concentrated. The no-refill invariant stands.

| book | total | ex-top5 | n | tWR | dWR | MDD | months+ | worst m | top5 share | eras H1/H2/26 |
|---|---|---|---|---|---|---|---|---|---|---|
| SHIPPED $100K baseline | +$96,454 | −$5,246 | 219 | 39.7% | 41.1% | −$20,832 | 9/20 | −$13,187 | 105.4% | −8.2/+13.2/+91.4K |
| **SHIPPED + G1** | **+$115,064** | **+$13,364** | 143 | 40.6% | 40.7% | −$17,225 | 12/20 | −$8,694 | 88.4% | +9.4/+12.6/+93.0K |
| B+ $10K baseline | +$9,931 | +$1,185 | 105 | 39.0% | 40.4% | −$1,210 | 12/20 | −$413 | 88.1% | +2.1/+0.7/+7.1K |
| **B+ + G1** | **+$10,712** | **+$1,966** | 81 | 43.2% | 45.1% | −$1,183 | 12/20 | −$407 | 81.6% | +2.1/+1.2/+7.5K |

(Harness baseline 219 trades vs pipeline 238: physics-reconstruction drops;
totals agree to 0.2%. My B+ reproduction: +$9,931 vs the rederivation doc's
+$11,303 — same config, small frame/month-count differences; treat as the
same book. On B+, G1 degenerates to RV20-only since pdr11 subsumes PDR≥9.2
— a consistency check that passed. B+ + G1 kills **zero** winners >$500 and
cuts trades to 4.0/mo, which further helps PDT headroom. Strict
missing→veto semantics on B+: +$10,983 / 13/20 months+ / MDD −$1,117.)

SHIPPED + G1 monthly: +3158 +2681 −1050 +4139 +205 +296 | +1706 −245 +4228
+22214 −6582 −5611 | +21883 +19068 +62085 −1557 +247 +776 −5962 −3042.
2025H1 flips six-red-months (−$8.2K) to +$9.4K; Nov-2025 worsens (−$6.6K,
gate keeps the volatile losers by design).

Ex-top-5 is the crux: the shipped book goes from "monster lottery with a
negative floor" (−$5.2K ex-top5) to a **positive floor** (+$13.4K ex-top5,
top-5 share 105%→88%). At B+ scale, ex-top5 +$1.2K→+$2.0K (~+66%).

**Residual drag is irreducible with 9:35 information**: post-G1 drag is 117
trades / −$65.2K, dominated by hyper-volatile stop-outs (JLHL RV20=77,
EHGO RV20=40/PDR=80, CAPR RV20=83, CYCU RV20=533…). These are
feature-identical to the monsters at entry — they ARE the cost of holding
the lottery tickets. Cutting them cuts ANNA/BNAI. Every attempt died in
Section 2.

## 6. Ranked recommendation

1. **G1 (RV20 ≥ 7.106 & prev_day_range_pct ≥ 9.226), post-selection
   no-refill veto, fail-open on missing data** — the only OOS-robust
   discriminator found. Honest projection if shipped on the shipped-clean
   config: book +$96.5K → +$115.1K/20mo (+19%), ex-top5 −$5.2K → +$13.4K,
   MDD −$20.8K → −$17.2K, months+ 9→12, zero top-10 casualties. **Anchor
   forward expectations on the OOS-2026 rate (~$4-6K/yr at $100K scale,
   ~16-17% of drag), not the blended +$11K/yr** — era-decay and the p≈0.05
   shuffle test both say the in-sample number flatters. Preconditions:
   ≥1 month paper/forward validation of the frozen thresholds; the same
   latency/parity preconditions as the B+ restart plan
   (orb_clean_rederivation_aug2026.md §7).
2. G3 (ADR20 ≥ 8.581 & PDR ≥ 9.226) — statistically indistinguishable
   sibling (slightly better WR/retention, slightly less dragrem). If the
   owner prefers a range-based over a return-based volatility measure,
   equivalent.
3. **Do NOT ship**: any gate from the failed list (§2) regardless of
   in-sample dollars — prev_day_volume_vs_20d, prev_day_close_position,
   range_vs_gap, rvol_20d, PM-structure gates, and all ≥50%-drag-cut
   combos. All kill top-10 winners OOS. Do NOT apply G1 pre-ranking with
   refill.
4. **The owner's "100K → 50K" bar: NO-GO as stated.** Half the drag is not
   separable from the winners at 9:35 with any information tested here
   (in-CSV features, premarket bar structure, news, SPY context, calendar,
   interactions). The strategy after G1 remains a monster-carried lottery
   with a modestly positive floor — G1 raises the floor; it does not
   change the texture.

## 7. Caveats

- 238 trades, 61 in TRAIN, 10 TRAIN winners. The thresholds are coarse
  quantiles and should be treated as "volatile vs quiet," not precise.
- Label-shuffle p≈0.05 on the magnitude: this is a one-in-twenty-worlds
  result, defended mainly by mechanism coherence (same axis as the
  independently-validated PDR veto) and zero-top-10-casualty structure.
- VAL-window retention 0.801 is exactly at the bar; the casualty texture
  (low-vol biotech pops) will recur.
- Feature-selection saw the full sample before TRAIN-only thresholding
  (per the prescribed methodology); the shuffle null covers the procedure,
  not the analyst's eyes.
- Harness note for future users: `_NEWS_CACHE`/anchors are module-level
  and keyed to the first `run_book` call's universe — running a filtered
  universe first poisons later runs (cost me a 3-trade discrepancy before
  diagnosing). Reset the module or run full-universe first.
- Nothing ships from this study: no orb.yaml/trading/ changes, no commits.
