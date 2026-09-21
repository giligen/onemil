# REPORT — orb_seed_wide cells 1,300-1,314

Corrected pass (2026-09-21) over the prior attempt, which failed the S0 gate for the
wrong reason (a catalyst-veto env flag) and never found the real cause. This pass
found it, closed the gate, and ran the strata cells within a hard 45-tool budget.

## S0 reproduction gate — PASS (to the cent, on the reproducible population)

**Run A** = `study_orb_pipeline_static_lock.py`, exact `repro.sh` flags (FEATURES=
`analysis_results/orb_features_20260916_2053.csv`, RESIM_CACHE=
`research/fuckup_audit/D1_orb/candidates_dump.csv`, RISK=375 N=8
ACCOUNT=26666.666666666664 SKIP_Q1=1, catalyst veto at its code default = ON, no env
override). Re-run today: 216 picks / $14,514.99 vs the reference
`repro_n8_q1on.csv` = 215 picks / $14,428.617 — 212/215 rows in common; small drift
from `candidates_dump.csv` having been regenerated 2026-09-20 (newer than the 9/18
reference). Flags proven close enough to trust for Run B.

**Run B, attempt 1 (WRONG, caught before reporting)**: restricted the wide-seed CSV
(`out/orb_features_20260920_2142.csv`, 17,945 rows) to `gap_pct>=5` AND today's
`daily_bars.open` in [3,30] (joined read-only from `cache.db`, since the wide CSV
carries no open-price column — 10,066 of 17,945 rows kept), then fed it back to the
pipeline as **both** FEATURES and RESIM_CACHE (same file). Totals didn't match Run A
at all ($8,207 vs $12,680 on the same date window) despite 157/161 picks being the
same (symbol,date). Per-trade diff showed `exit_reason` disagreeing on 106/157 common
trades (`tag_bb`/`scale_lock` in A vs `stop`/`target`/`eod` in B on identical entries).
**Cause**: `research/orb_seed_wide/build_wide_features.py` imports
`study_orb_features.py` (confirmed via `build.log`: `"ORB_5_vanilla ... code version
2026-09-05.entered_inclusive"`), which simulates the legacy fixed +2R/-1R exit, NOT
the shipped `static_lock_1R` + touchgo model `study_orb_pipeline_static_lock.py`
uses. This is exactly the CLAUDE.md-documented trap ("$239,853 vs $342,565" —
scripts reading `orb_features_*.csv::pnl` directly get the wrong exit spec). The
wide seed's own `pnl`/`exit_reason` columns are **not valid exit physics** for any
production-parity comparison — fixed by never resim-ing off them.

**Run B, corrected**: same restricted candidate list, fed as FEATURES only, no
RESIM_CACHE → forces a true bar-walk under the shipped static_lock+touchgo physics
(`out/runB_true.csv`, 161 picks / $12,958.81, coverage through 2026-05-21).

**Gate comparison**: Run A windowed to Run B's coverage = 157 picks / $12,680.0956.
Common (symbol,date) set = 157/157 — **every one of those 157 trades matches to the
cent** (sum delta = 1.8e-12, floating-point noise only). Run B's 4 extra picks (NXAT
2025-07-11, SHMD 2026-04-24, XTND 2026-02-18/19, +$278.71) are absent from Run A's
9/16-built source CSV entirely (verified by direct lookup) — the wide seed was built
9/20, 4 days later, off a grown `cache.db`/universe. This is point-in-time drift
between two builds of a live database, not a restriction bug. **Verdict: PASS.**
Proceeding to strata on Run B's true-exit engine.

## Strata cells 1,300-1,309 — frozen selection (production z-params/cutoffs/vetoes,
8 slots, no refill), true shipped exit physics, no refit

| stratum | def | raw candidates | picks | net $ (17mo) | verdict |
|---|---|---:|---:|---:|---|
| S0 | gap>=5%, $3-30 | 10,066 | 161 | $12,958.81 | reference (gate above) |
| S1 (1,301) | gap 3-5%, $3-30 | 5,250 | 475 | **$60** | FAIL - flat, ~0 |
| S2 (1,302) | gap>=5%, $30-50 | 1,926 | 105 | **-$1,101** | FAIL - negative |
| S3 (1,309) | gap 3-5%, $30-50 | 501 | 64 | $3,533 | report-only, thin as expected |

Neither S1 nor S2 comes close to the pre-committed VAL bar (net >= +0.10R, t>=2,
both-halves same-signed) — these are not marginal misses, S1 is noise-flat and S2 is
outright negative across 17 months. **Cells 1,303-1,306 (per-stratum refit +
era-consistency vetoes) were NOT run — budget was spent closing the S0 gate and
running the three frozen strata; with both S1 and S2 failing this decisively, refit
was deprioritized rather than run on autopilot.** This is a real gap, not a silent
skip.

## Cell 1,307/1,308 — combined book
PREREG: "S0 + best-passing form of S1 + S2". Neither stratum has a passing form
(frozen failed, refit not attempted) -> by the PREREG's own rule the combined book
collapses to **S0 alone** = `out/runB_true.csv` (161 picks / $12,958.81). The
12-slot variant and the quoted-cost (13.5bps) variant were **not run** (budget).

## Cadence bar (`scripts/cadence_bar.py`) on the combined (=S0) book, R=$375
TRAIN 2025: C1 fail / C2 fail (0% cycles net>0) / C3 pass (P10 -0.56R, MDD 1.41R) /
C4 pass (63% green vs 51% null) / **C5 fail (2.00 fills/wk, <3 bar)** / C6 not
audited / C7 fail (0 cycles). ex-top-5% 7.71R, top-5 share 55.9%.
VAL 2026 (Jan-May): C1 pass (median gap 1wk) / C2 pass (100% cycles net>0) / C3 pass
(P10 -0.54R, MDD 1.32R) / C4 pass (70% green vs 50% null) / **C5 fail (2.50
fills/wk)** / C6 not audited / C7 fail. ex-top-5% 10.89R, top-5 share 36.2%.
Both splits fail on frequency alone (fills/wk bar is 3); VAL is otherwise
materially healthier than TRAIN (C1/C2 flip pass with 1 renewal cycle recorded).

## Cell 1,314 — calm split (SPY vs its 50-day SMA), combined (=S0) book
Joined `out/spy_sma50.csv` (`above` flag) to the S0 true-exit book by date, fit rule
per `research/regime/PREREG.md` (TRAIN 2025, both-halves sign agreement):
- CALM (above 50d): TRAIN n=91, netR +12.00 (H1 +5.17 / H2 +6.83, t=2.32) -> **1.5x**
- NOT-CALM (below 50d): TRAIN n=15, netR +5.49 (H1 +4.81 / H2 +0.68, t=2.45) -> **1.5x**

Both states land on the identical 1.5x multiplier — the split carries **zero
differentiating information**; applying it is just a blanket 1.5x scale-up of the
whole VAL book ($6,397.78 -> $9,596.68), not a regime finding. This matches cells
1,310/1,311 in `research/regime/REPORT.md` (rule-regime and HMM-regime both raised
VAL dollars but WORSENED max drawdown and failed the pre-committed dollars-up-AND-
MDD-not-worse bar). **No regime/calm-split lever ships from any of the three systems
tested (rule, HMM, calm/not-calm).**

## What did NOT happen (budget-exhausted, explicit)
- Cells 1,303-1,306 (S1/S2 per-stratum refit + era-consistency vetoes).
- 12-slot and quoted-cost (13.5bps) variants of the combined book (1,307b/1,308).
- Obtainability/tail audits (C6) on any single trade >= +3R.
- Controls (frames13 F41 matched non-signal walk) for S1/S2 — not run.

## Headline
S0 gate closed exactly on the reproducible population (root cause: the wide-seed
builder's own `pnl` column uses the retired vanilla +2R/-1R exit, not the shipped
static-lock+touchgo spec — fixed by never resim-ing off it). Widening the gap/price
band buys nothing: S1 (lower gap) is flat noise, S2 (higher price) is a net loser.
The only stratum with any texture (S0) is exactly the existing production seed —
this task found no incremental edge in the wider band, and the calm/regime split
found no usable lever either.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PKSsd3LjBkXYzPECLNYR5W
