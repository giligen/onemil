# ORB in-flight family: DAY-LEVEL META (hostile days, not hostile trades) — 2026-08-14

Parallel drag-reduction program (owner order: cut clean-book loser drag −$101.7K → −$50K
while keeping ≥85-90% of top-40 winner P&L). This family: recognize hostile DAYS.

**Family verdict: NO-GO across the board. best_rule = NONE.** All four hypotheses fail —
two are structurally inert on this book (fire ~never), two fire but destroy named
top-40 winners or rest on a data artifact. One important structural anti-correlation
and one data-quality bug were found (see §6) — both matter to the OTHER families.

Tooling: `scratchpad/day_meta_lib.py` + `day_meta_recon.py` / `day_meta_h123.py` /
`day_meta_h4.py` / `day_meta_h4b.py` — extends `research/scripts/orb_clean_harness.py`
(unmodified) with per-trade entry/exit ET minutes (timed exit sim asserted
price-identical to the harness sim on every trade). All rules applied
pipeline-integrated, post-selection, NO REFILL. Data: DST-clean
`analysis_results/orb_features_20260814_1741.csv`, physics rebuilt from cache.db 1-min bars.

## 1. Baselines (harness, Jan'25 → Aug 14 '26)

| config | total | n | tWR | dWR | MDD | months+ | top5 | eras H1/H2/26 |
|---|---|---|---|---|---|---|---|---|
| shipped clean $100K | +$96,454 | 219 | 39.7% | 41.1% | −$20,832 | 9/20 | 105.4% | −8,188 / +13,233 / +91,409 |
| B+ ($10K, SL q40 pdr11 N3 cvY, TRAIN-refit) | +$9,931 | 105 | 39.0% | 40.4% | −$1,210 | 12/20 | 88.1% | +2,131 / +711 / +7,089 |

Windows: TRAIN=2025H1 (derive), VAL=2025H2 (freeze-validate), OOS=2026.
Winner cohort = baseline top-40 by sized P&L; drag cohort = ranks 41+.

**Day structure — the family's central obstacle**: shipped book = 163 trade-days:
115×1 trade, 40×2, only 8×3. Median entry 9:36 ET, 75th pct 9:42, latest 10:33
(60-min order auto-cancel). B+ = 94 trade-days: 84×1, 9×2, 1×3. Day-level rules need
*multiple sequential same-day events*; this book barely produces any. 90 of 132 shipped
losers do exit before 10:35 — but almost never before a *later same-day entry* exists.

## 2. H1 consecutive-loser day stop — NO-GO (structurally inert)

Rule: after k same-day realized-loss exits (stop/tag_bb/tag_b1/any loss — definition
makes no difference), cancel later entries.

- **k=2 (hypothesized): fires ONCE in 20 months** on shipped — and cancels a +$386
  winner (VAL drag *worsens* $386). k=3: zero firings. B+: zero firings at k=1,2,3.
- OOS drag reduction: **$0**. Winner retention 100% everywhere (nothing happens).
- Mechanism of failure: entries cluster 9:35–9:42; two losses must *realize* before a
  third trade *triggers*. Only touchgo exits are fast enough, and 3-trade days are 8/20mo.
- Exploratory k=1 (NOT the hypothesis, most aggressive possible): 9 cancels/20mo,
  VAL +$5,043 / OOS +$2,164 drag reduction, one top-40 casualty (SDOT 2025-09-15
  +$1,642→0). **Zero TRAIN firings → underivable walk-forward**; 9 events is not a
  rule, it's an anecdote. Also NO-GO.

## 3. H2 early-day quality read (first 2 fills underwater at 10:15 → cancel pending) — NO-GO (never fires)

Implemented with marks from post-entry 1-min closes (exit price if already exited).
Checked at 10:00, 10:15, 10:30 on both configs: **zero trades cancelled at every
setting** — a day with ≥2 fills by the check time AND a still-pending later entry AND
both fills underwater does not occur in 20 months. Same structural cause as H1.
OOS drag reduction $0. Nothing to tune, nothing to validate.

## 4. H3 family-crowd failure (2+ same-anchor positions both below entry → cut both) — NO-GO (kills the crowd-winner mode)

Prevalence: 14 same-anchor same-day groups in 20 months (shipped); **only 1 in TRAIN**
(AVGO 2025-01-30) → the rule is underivable under the walk-forward protocol from the
start. Simulated anyway (minute-level joint marks, cut both at bar close, 10bps slip):

- As hypothesized (both below entry): net **−$16,539** on shipped. Winner retention
  TRAIN 100 / **VAL 65%** / OOS 94% — fails the ≥85%-everywhere bar.
- **Named casualties**: CRCA+CCUP+CRCG 2025-10-02 (CRCL trio, ranks 6-8, +$16.5K → −$148
  combined), IREX+IRE 2026-03-11 (+$4.2K → −$208), CRWU+CWVX 2026-07-21 (+$2.4K → −$417).
- The loser side works as imagined (NBIS 2026-01-26 +$2.8K saved, AVGO +$1.7K, IREN
  2026-08-11 +$1.6K, CRCL 09-24 +$1.5K…) but anchor crowds are **bimodal — they win
  together or fail together**, and the winners dip below entry jointly before running.
  Cutting on the joint dip amputates exactly the crowd-monster mode (the CRCL-trio class).
- Depth-qualified variant (cut only when both < entry − d·R): d=0.5R spares all winners,
  +$848/+$910/+$121 by window — but d=0.25R flips to **−$9.7K** (VAL retention 76%).
  A sign-flipping cliff between adjacent thresholds = overfit by the program's own
  test; OOS effect at d=0.5R is $121 ≈ zero; and TRAIN still has 1 event. NO-GO.
- B+: 5 groups/20mo, net +$45 as hypothesized, 2 top-40 casualties. Inert-to-negative.

## 5. H4 SPY day-type prefilter at 9:35 (skip the whole day) — NO-GO (gap gate kills ANNA/ANTX; range gate is a data artifact)

TRAIN-only derivation (45 trade-days, book −$8,188): all SPY-feature/day-P&L
correlations are noise (|pearson| ≤ 0.12, n=45). Two quartile patterns were frozen and
tested forward anyway:

**G1 — skip days with spy_gap_pct ≤ −0.30** (TRAIN worst-quartile edge; TRAIN +$10.5K
drag reduction): forward it is fatal. **Kills ANNA 2026-03-20 (SPY gap −0.50, +$41.0K)
and ANTX 2026-03-09 (gap −0.89, +$18.3K)** → OOS winner retention **54%**; shipped book
total drops $96.5K → $46.4K, MDD *worsens* to −$27.3K. ±20% threshold shifts: identical
casualties. Dead by the program's named-monster rule. Same on B+ (OOS retention 47%).

**G2 — skip days with spy_range_pct_5min < 0.07** ("quiet SPY open"; looked monotone on
TRAIN, OOS drag −$36K→−$11K, retention 87/100/81): **exposed as a data artifact.**
`spy_range_pct_5min == 0.000` exactly on **136 of 404 days** — missing SPY 1-min bars in
cache.db (clustered 2025-03/04 and 2026-03→08, i.e. most of the OOS year), zero-filled by
the feature builder. Among days with real SPY data, only **2** fall below the 0.07
threshold. The gate is really "skip days where the research cache lacks SPY data" — not
live-implementable (live SPY always prints), an accidental rule of exactly the class the
owner forbids, and it deletes the IREX 2026-07-30 (+$4.4K) winner day. TRAIN retention
81% (<85%) fails the bar regardless. NO-GO.

## 6. Cross-cutting findings for the other families (the useful output of this NO-GO)

1. **Monsters live on SPY-red mornings.** All five top-5 winner days had negative SPY
   gaps (ANNA −0.50, ANTX −0.89, BNAI −0.12, BNAI2 −0.07, ZURA −0.11). The clean-book
   edge is idiosyncratic story stocks that gap up *against* a soft tape. Any day-level
   "risk-off skip" or SPY-correlated de-risking gate is structurally anti-correlated
   with the edge and will fail the winner-retention bar. Do not re-derive this class.
2. **DATA-QUALITY BUG: `spy_range_pct_5min` / `spy_return_5min_pct` are zero-filled on
   136/404 days** in `orb_features_20260814_1741.csv` (missing SPY 1-min coverage in
   cache.db, spanning 5 of 8 OOS-2026 months). Any sibling agent using these features
   (composite refits, regime gates) must exclude or backfill those days first, or their
   TRAIN/OOS splits inherit an artifact that happens to align with the 2026 bleed months.
3. **The book has no day-level texture to exploit**: 71% of trade-days have exactly one
   trade; sequential-information rules (loser counts, early reads, crowd cuts) cannot
   accumulate evidence before the entry window closes at ~10:35. Day-meta rules would
   only become testable if the book widened to N≥4 with refills — which the no-refill
   invariants forbid for validated reasons.

## 7. Scorecard

| rule | fires/20mo | OOS drag red. | winner ret. (TRAIN/VAL/OOS) | named casualties | verdict |
|---|---|---|---|---|---|
| H1 k=2 (k=3) | 1 (0) | $0 | 100/100/100 | none (rule inert) | **NO-GO** |
| H2 @10:15 (±) | 0 | $0 | 100/100/100 | none (rule inert) | **NO-GO** |
| H3 crowd-cut | 26 exits | +$2,925 | 100/**65**/94 | CRCL trio 10-02, IREN 03-11, CRWV 07-21 | **NO-GO** |
| H4 G1 gap≤−0.30 | 36 days | +$6,395 | 87/90/**54** | **ANNA 03-20, ANTX 03-09** | **NO-GO** |
| H4 G2 r5<0.07 | 136 days | (+$25,338)* | 81/100/87 | IREX 07-30 + artifact | **NO-GO** (artifact) |

\* not real — gate keys off missing data, see §5/§6.2.

Scripts: `scratchpad/day_meta_lib.py`, `day_meta_recon.py`, `day_meta_h123.py`,
`day_meta_h4.py`, `day_meta_h4b.py`. Baselines pickled: `scratchpad/day_meta_baselines.pkl`.
Nothing committed; orb.yaml/trading/ untouched.
