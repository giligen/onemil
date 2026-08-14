# ORB clean re-derivation (2026-08-14, post-DST-bug emergency rebuild)

Owner-ordered rebuild after the session-open DST bug (3fab1f9) voided every
prior ORB research verdict. ORB live is at zero; it re-enables only on an
owner-approved rebuilt config. This document contains: the repo-wide bug-class
audit, the regen validation, the honest clean-book baseline, a walk-forward
re-derivation on clean data, candidate configs vs the owner's design bars, and
a ranked recommendation. **Headline: the clean 20-month book is +$96.6K, not
the void $251.6K; no candidate meets the full owner bar set; two candidates
(B+/D+) meet everything except the high-WR and top5<40% bars, which appear
STRUCTURALLY unattainable for ORB.**

Tooling: `research/scripts/orb_clean_harness.py` — rebuilds per-trade physics
(exact 9:30 ET anchor) from cache.db bars, cross-checks every row's trigger
against the features CSV (>10bps disagreement = dropped as contaminated), and
re-simulates exits/selection/sizing/vetoes with every knob parameterized.
Validated twice: (a) on the pre-fix CSV's EDT rows it reproduces the shipped
pipeline's monthly table; (b) on the regenerated CSV it reproduces the
pipeline book to $3 total ($96,639 vs $96,642) with **0 trigger mismatches
across all 6,918 rows**.

---

## 1. Bug-class audit (repo-wide, all strategies) — VERDICT: ORB-research-specific

Pattern hunted: session/range windows via naive-UTC hour comparisons
(`hour == 13/14`, `hour.isin([13,14])`, `minute == 30`, hardcoded UTC times).

**Fixed today (verified in git):**
- `study_orb.py::_session_open_timestamp` — THE bug (3fab1f9).
- `study_orb_features.py` SPY 5-min features (`spy_range_pct_5min`,
  `spy_return_5min_pct`) had the same hour-in-{13,14} mask — fixed in
  35e9935 (17:33 UTC today). **Empirically confirmed the regenerated CSV used
  the fixed code**: EST-season row 2025-01-03 spy features match the true
  9:30 ET SPY window to 4 decimals. (These features were never in the shipped
  composite, so the book was not affected through them.)

**Residual bug-class hits (research/diagnostic only, NOT fixed):**
- `research/study_macd_day_from_high.py:52-53` — RTH window hardcoded
  13:30→20:00 UTC "liberal"; winter rows include 8:30-9:30 premarket and
  exclude 15:00-16:00 ET. Any winter conclusions from this MACD study are
  suspect.
- `study_orb_exit_variants.py:79-82` — "11:00 ET" close matches hour in
  (15,16) UTC; in EST season it closes at 10:00 ET. (Superseded research
  script; already carries a stale-warning header class.)
- `diagnose_morning2.py` (and sibling) — EDT-hardcoded UTC bucket edges
  (13.5/14.0/14.5). Diagnostic only.

**Verified tz-correct (no action):** live ORB engine
(`trading/orb_engine.py::_first_session_open_ts_utc` — per-date UTC offset,
fixed 2026-07-03), `study_orb_pipeline_static_lock.py`'s own session-open,
bull-flag backtest/live stack, MACD-wave engine/backtest, ignition stack (ET
conversions throughout). **The bull-flag and MACD-wave BT verdicts do NOT
inherit this bug class.**

## 2. Regen validation (Phase B gate) — PASSED

- CRNC 2025-01-03: regenerated entry **11.7953 = 11.76 × 1.003** (true 9:30
  range), vs the void book's premarket-derived 10.42.
- 6/6 random EST-season spot checks: CSV entry == reconstructed 9:30-ET
  trigger to <10bps.
- Harness full-book pass: 6,918/6,918 rows reconstruct with 0 mismatches.
- Note: the first regen attempt (via `orb_backtest.py --force-full-regen`)
  died in `regen_features`'s **900s subprocess timeout** — a full regen takes
  ~23 min and cannot complete through that path (needs a fix or a direct
  `study_orb_features.py --force-full-regen` invocation, which is what
  produced `orb_features_20260814_1741.csv`).

## 3. The honest clean-book baseline (shipped params, $100K book, Jan'25→Aug 13 '26)

`study_orb_pipeline_static_lock.py` on the regenerated CSV (this also
regenerated the official `analysis_results/orb_static_lock_trades.csv` /
`orb_monthly_static_lock.csv` artifacts):

| metric | clean book | void book |
|---|---|---|
| Total P&L (20 months) | **+$96,642** | $251,647 |
| Negative months | **13/20 (65%)** | 6-9 dep. on vintage |
| Median month | **−$408** | positive |
| Best month | +$60,710 (2026-03) | — |
| Worst month | −$12,443 (2025-12) | −$10.3K |
| Trade WR / day WR | 39.9% / 41.7% | ~47% claimed |
| MDD (daily cum) | −$21,835 | −$14-18K claimed |
| Top-5 trades' share | **105%** (ANNA +$41.0K, BNAI +$27.2K, ANTX +$18.3K, BNAI2 +$7.7K, ZURA +$7.5K) | — |
| Eras | 2025H1 **−$7.5K** · 2025H2 +$14.0K · 2026 +$90.1K | 2025H1 +$60K claimed |

Monthly: −156, +558, −4578, −84, −2269, −955, −816, −980, +8511, +23715,
−4010, −12441 | +18698, +18392, +60709, −256, −560, +1501, −5296, −3042.

**Read: on clean data the shipped ORB is a breakeven grinder that got saved by
five monster trades, four of them in a single 3-month window (Jan-Mar 2026).**
Ex-top-5 the book is ≈ −$5K. Every prior headline ($342K / Calmar 18.9 /
$251.6K / "1 red month") is void.

## 4. Phase A (clean-EDT slice, pre-regen) — what survived and what didn't

Run on the uncontaminated EDT-season rows of the pre-fix CSV
(2025-03-10..2025-10-31, 2026-03-09..2026-08-13) at shipped $100K sizing.
Kept findings (later re-confirmed on the full clean sample):

- **Touchgo re-validates**: removing it costs −$21.8K and doubles MDD.
- **PDR veto re-validates, optimum ABOVE the shipped 8.0**: monotone
  improvement through 10-12 on every bar tested (P&L, months+, WR, tail).
- **Quick profit targets (≤1R) destroy the edge** (+$0.75R target keeps only
  ~2% of static-lock P&L at $100K scale). The edge lives in the ≥1.5R tail.
  ORB cannot be turned into a high-WR scalper.
- **Composite threshold ~q40-q55 of TRAIN > keep-almost-all**.
- **PM news-gated 2.0× mult: evidence is monster-concentrated** (mostly the
  single ANNA trade in the EDT slice) — excluded from small-book candidates.

Phase A traps documented for posterity:
- The EDT slice made target exits look era-consistent and low-tail (top5
  ~34-40%) — an artifact of the window excluding the 2026 winter monsters.
  On the full clean timeline the same configs are 2025H2- and 2026-negative.
- **Fit-provenance trap**: with the frozen orb.yaml z-params the EDT
  candidates made +$7K; refit honestly they made ≈$0. The yaml fit was
  produced (April 2026) by a process that observed the whole contaminated
  period — its apparent superiority is hindsight leakage. All Phase B numbers
  use TRAIN-only refits.

## 5. Phase B walk-forward re-derivation (clean data)

Protocol: z-params + quintile cutoffs + threshold (as TRAIN-quantile) fit on
**TRAIN = 2025H1 only**; 2025H2 = validation; **2026 = OOS, never used for
fitting**. One frozen fit tested forward (per the 2026-07-03 refit-cadence
lesson). Pipeline-integrated simulation (selection/dedup/slots/vetoes), not
per-trade sums. $10K book: N=2-3 picks/day, $375 risk/trade, $10K/N per-pos
cap, uniform mults, no PM mult, 30bps entry + 10bps exit slip, integer shares.

32-config grid (exit × threshold × PDR × catalyst-veto), all-era columns in
`scratchpad/phaseB_grid.csv`. Signal summary:

- **Static-lock and partial-1.5R exits are the only era-consistent shapes.**
  Full target exits (1.5R/2R) are 2025H2-negative AND 2026-OOS-negative.
- **q40 > q60** (breadth wins), **PDR 10 > 8** (again), catalyst-veto is a
  2025-cost / 2026-OOS-benefit trade (see below).
- **Null test (random scores, same pipeline): −$5,827.** The TRAIN-refit
  composite layer beats random by ~+$14-22K depending on shape — the
  selection layer carries real, walk-forward-honest signal on clean data.
  (This was NOT distinguishable from luck on the EDT-only slice.)

### Candidate table ($10K book, 19-20 months Jan'25→Aug 13 '26; 2026 column is OOS)

| cfg | shape | total | tWR | dWR | MDD | months+ | worst m | top5 | ex-top5 | eras H1/H2/26 | tr/mo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | SL q40 pdr10 cvN N2 | +$16,648 | 32.0% | 33.0% | −$2,444 | 13/20 | −$1,408 | 92% | +$1.3K | +4.8/+1.9/+10.0K | 12.5 |
| B | SL q40 pdr10 cvY N2 | +$14,320 | 41.6% | 40.6% | −$1,184 | 10/19 | −$438 | 92% | +$1.2K | +0.2/+1.7/+12.4K | 3.9 |
| C | part1.5R q40 pdr10 cvN N2 | +$8,149 | 34.0% | 37.1% | −$2,316 | 12/20 | −$1,376 | 96% | +$0.3K | +4.1/+0.8/+3.3K | 12.5 |
| D | part1.5R q40 pdr10 cvY N2 | +$7,890 | 42.9% | 43.5% | −$1,168 | 10/19 | −$820 | 83% | +$1.3K | +0.9/+1.4/+5.6K | 3.9 |
| **D+** | part1.5R q40 **pdr11 N3** cvY | +$7,085 | 42.6% | 44.2% | **−$524** | **13/19 (68%)** | **−$292** | 63.5% | **+$2.6K** | +1.6/+1.5/+4.0K | 5.2 |
| **B+** | SL q40 **pdr11 N3** cvY | **+$11,303** | 41.6% | 41.9% | −$715 | 13/19 (68%) | −$346 | 77% | +$2.6K | +1.4/+2.0/+8.0K | 5.2 |

D robustness (every perturbation positive, all eras positive in all of them):
q30/q50 threshold, pdr 9/11, risk 250/500, q4-first vs composite ranking,
N=2/3. The pdr11+N3 combos (B+/D+) are robustness-region neighbors but their
exact values were chosen with full-sample visibility — treat the increment
over B/D as needing forward validation.

Catalyst veto on clean data (nuanced, differs from the contaminated evidence):
costs ~$4-5K of 2025 P&L (mostly H1 winter), but on OOS-2026 it ADDS P&L
(B vs A: +$2.4K; D vs C: +$2.3K), lifts trade WR ~10pp, cuts MDD in half, and
cuts trade count 70% (PDT-relevant). KEEP for a small book.

### $10K-scaled projection with frictions

B+: ≈ **+$7.0K/yr nominal (≈70%/yr)**; D+: ≈ +$4.4K/yr (≈44%/yr). Haircuts:
- Fill reality (research/orb_bar1_fill_study.md): ~96.7% P&L capture at ≤5s
  placement latency, ~85.5% at the current degraded ~74s. **Deployment
  assumes the post-2026-07-07 latency regression is fixed**; because these
  books are monster-carried, ONE missed IREX/ANNA-class fill erases months.
- PDT: ~5 trades/mo average is compatible with a <$25K margin account's
  3-day-trade/5-day limit on average, but trades cluster; a cash account or
  >$25K umbrella account avoids forced skips.
- Winter is now the untested season for LIVE (live has only run May-Aug 2026);
  BT winter rows are clean but live-vs-BT winter parity has never been
  observed. First EST week (Nov 2026) needs a parity audit.

## 6. Owner design-bar scorecard

| bar | A | B | D | D+ | B+ |
|---|---|---|---|---|---|
| Deployable at $10K | ✓ | ✓ | ✓ | ✓ | ✓ |
| HIGH win-rate | ✗ 32% | ✗ 42% | ✗ 43% | ✗ 43% | ✗ 42% |
| Top-5 < 40% of P&L | ✗ 92% | ✗ 92% | ✗ 83% | ✗ 64% | ✗ 77% |
| ≥70% positive months | ✗ 65% | ✗ 53% | ✗ 53% | ✗ 68% | ✗ 68% |
| Worst month bounded | ~ −14% | ✓ −4.4% | ✓ −8.2% | ✓ −2.9% | ✓ −3.5% |
| All 3 eras positive | ✓ | ✓ (H1 +$228) | ✓ | ✓ | ✓ |

**No config meets the full bar set.** The failures are structural, not
tuning gaps:
- WR ceiling: every P&L-positive exit policy sits at 30-46% trade WR. The
  only ways to push WR above ~50% (sub-1R profit targets) destroy 90%+ of
  P&L (frontier measured in Phase A and re-confirmed on clean data).
- Tail concentration: ORB's clean-data edge IS the monster tail. De-tailing
  exits (1.5R/2R targets) are OOS-negative. Best achievable while keeping
  the edge: ~63% top-5 share (D+), vs the 40% bar.

## 7. Ranked recommendation

1. **Honest default: ORB stays at zero** against the stated bars. The
   clean-data strategy is a monster-lottery with a ~breakeven floor; it
   cannot honestly be represented as high-WR / low-lottery / 70%-green at
   any parameterization found in a 32-config walk-forward plus targeted
   robustness sweeps. NO-GO is the recommendation as the bars stand.
2. **If the owner explicitly re-accepts lottery texture with hard risk
   caps** (the bars re-stated as: tiny MDD, bounded worst month, era
   consistency, ex-top5 ≥ 0): **B+** (static-lock, q40 TRAIN-quantile
   threshold, PDR 11, N=3, catalyst veto ON, $375 risk, no PM mult, uniform
   mults) is the best clean-data book: +$11.3K/19mo on $10K, MDD −$715
   (7% of book), worst month −$346, 13/19 months green, all eras positive,
   OOS-2026 +$8.0K. **D+** (same but partial-1.5R exit) is the
   lower-variance sibling (MDD −$524, worst month −$292) at ~2/3 the P&L.
   Preconditions for ANY restart: (a) fix the order-placement latency
   regression to ≤5s; (b) run ≥1 month paper/micro-size forward validation
   of the frozen TRAIN fit (the exact z-params/cutoffs/threshold used here);
   (c) schedule the Nov-2026 first-EST-week live-vs-BT parity audit.
3. **Do NOT re-ship**: PM news-gated 2.0× mult (monster-concentrated
   evidence, amplifies exactly the lottery the owner wants less of),
   adaptive quintile mults (already neutralized), threshold 0.0-absolute
   (use TRAIN quantile), PDR 8.0 (11 dominates on clean data).
4. **New ways worth researching** (not ready to ship): monster-precursor
   modeling — the clean book's entire edge is ~6 trades with identifiable
   pre-open texture (news + PM$ + wrapper-cohort anchors + now-clean SPY
   features); a classifier that trades ONLY monster-candidate mornings at
   2-3× size, sitting out otherwise, is the only route that could raise both
   WR and tail-share simultaneously. Also: intraday re-entry on the same
   symbol after touchgo exits (the current book forfeits the re-break), and
   a regime gate on the 2025H1-style chop that produced 6 straight red
   months.

## 8. Caveats (owner-demanded brutal honesty)

- 20 months of data, ONE clean pass; the candidate increments beyond B/D
  (pdr11, N=3) saw the full sample. Expect haircut on everything.
- The books are monster-carried even at their best (top5 63-92%). A single
  missed monster fill (latency!) or a monster-less year changes the sign of
  a quarter, possibly the year.
- BT fill model = stop-limit at trigger×1.003 fills whenever touched;
  bar-1 study says that's ~3% optimistic at healthy latency, ~15% at
  current latency; slot/concurrency modeled at selection level only.
- 2025H1 (clean) was six red months for the shipped config. Nothing found
  prevents a repeat; B+/D+ merely bound its cost (~−$300/mo at $10K).
- Regen artifacts: `analysis_results/orb_features_20260814_1741.csv` (clean,
  6,918 rows, incl. clean SPY features), official book regenerated via the
  shipped pipeline. Grid: scratchpad `phaseB_grid.csv`. Harness:
  `research/scripts/orb_clean_harness.py` (suites: baseline / exits /
  entries / small / wr / pdr / final / small_risk).
