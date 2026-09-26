# Diagnostic — cell 1,445 corrected-cost/slip arithmetic reconciliation, 2026-09-26

Read-only rebuild of `research/hod_entry/cell_1445.py` Step 1 (its own `load_base_book`,
`corrected_cost`, `apply_stop_slip` functions, called directly, no files written by the module).
Base book = `causal_arming_causal.csv`, status=='fill', TRAIN-H2+VAL, n=9,911 (TRAIN-H2 4,398 /
VAL 5,513). Cross-checked against the already-written `cell_1445_features.csv`: my `net_R_corr`
and `net_R_corr_flat30` match it to 6+ decimals (mean net_R_corr -0.251319226762738 both ways;
per-split -0.246762/-0.254955 TRAIN-H2/VAL, matching RESULT_1445.md's reported book). This is a
diagnostic, not a research cell — every number below is a direct query result, not a claim.

## Reconciliation table (mean R, whole book n=9,911, then split)

| step | formula | whole book | TRAIN-H2 (n=4,398) | VAL (n=5,513) |
|---|---|---|---|---|
| base | `net_R` | -0.21734 | -0.20838 | -0.22449 |
| + cost fix | `+ half_entry/R` | +0.10268 | +0.10059 | +0.10435 |
| = costfix | `net_R_costfix` | -0.11466 | -0.10779 | -0.12014 |
| − slip (measured) | `- slip_R` | -0.13666 | -0.13897 | -0.13481 |
| = corr | `net_R_corr` | **-0.25132** | **-0.24676** | **-0.25496** |
| − slip (flat 30bps) | `- slip_R_flat30` | -0.13238 | n/a | n/a |
| = corr_flat30 | `net_R_corr_flat30` | **-0.24704** | | |

Arithmetic chains exactly (base + cost − slip = corr, to the last digit, both variants) — **there
is no bookkeeping error in the addition/subtraction itself.** The gap to the PREREG's ≈ -0.20
expectation is entirely inside the slip term: expected subtraction ≈0.08R, actual 0.137R (+0.057R
more negative than budgeted), while the cost-fix add (+0.103) matches the ≈+0.10 expectation
closely.

## Where the extra -0.057R comes from

`mean slip_R` by exit type (0 for `target`/`eod_fallback`, confirmed exactly 0.0 both):

| why | n (whole book) | mean slip_R | share of book mean slip (0.1367) |
|---|---|---|---|
| stop | 5,699 | 0.2077 (TRAIN-H2 0.2120 / VAL 0.2042) | 0.1194 (87%) |
| stop_bar | 76 | 0.2274 (0.2104 / 0.2405) | 0.0017 (1%) |
| eod | 1,321 | 0.1163 (0.1192 / 0.1141) | 0.0155 (11%) |
| target | 2,757 | 0.0 | 0 |
| eod_fallback | 58 | 0.0 | 0 |

Two things drive the -0.057R gap, in order of size:

1. **The "35 bps ≈ 0.08R" back-of-envelope understates the true bps→R conversion for this book.**
   Cell 1,443's own measured mean stop slip is 35.9bps (TRAIN-H2) / 34.8bps (VAL) — matching the
   PREREG's "35 bps" — but converted through each row's own `stop`-price and `R` (dollar risk/share,
   which is small relative to price for HOD-break stops — consistent with the `R must exceed the
   spread` project note), the SAME 35bps measurement is **0.208R** on average (RESULT_1443.md's own
   "mean slip (R)" column reports 0.209/0.199, which matches this exactly). Stop exits alone
   (87% of the total slip subtraction) are already ~2.5x the -0.08R the PREREG expected — this is
   the dominant term, not a code defect: 1,443 published this same 0.199–0.209 R/stop-exit number
   itself.
2. **A real, smaller defect: 1,321 `eod` exits (13% of the book) get slipped too, and unmeasured
   ones use the STOP fallback bps, not an EOD fallback.** `SLIP_APPLICABLE_WHY = {stop, stop_bar,
   eod}` intentionally slips eod exits (cell 1,443 also measured EOD bid-at-first-print, per its own
   `to_measure()`), and the PREREG's back-of-envelope ("cell 1,443: mean 35 bps on stop exits ...
   subtracts about 0.08R") only budgeted for stop exits, so eod's +0.0155R (11% of the slip
   subtraction) was never in the expectation at all. Within that: of 1,321 eod rows, 757 (57.3%)
   get 1,443's own measured eod bps (mean 10.48bps, matching RESULT_1443.md's 11.5/9.7bps table)
   but the other 564 (42.7%, uncached) fall back to `FLAT_SLIP_FALLBACK_BPS = 35.0` — the STOP
   holdout mean — instead of an eod-specific fallback (~10-11bps). That overcharges each of those
   564 rows by ~0.139R (0.199R charged vs ~0.060R at the true eod-fallback rate), adding **+0.0079R**
   of avoidable extra negative to the whole-book mean (i.e. without this defect net_R_corr would be
   ≈ -0.2434 instead of -0.2513) — real, but a small share (≈14%) of the -0.057R gap.

Net: -0.217 (base) + 0.103 (cost, ≈ as expected) − 0.137 (slip, of which stop-exit's own known
0.20R/exit rate is 87%, eod-inclusion 11%, and the eod-fallback-bps defect ≈6% of the slip term /
≈14% of the total gap) = -0.251, reconciling the reported -0.247/-0.255 (TRAIN-H2/VAL) exactly.

## Answers to (a)/(b)/(c)

**(a) Does the nbbo.csv join recover half_entry for 'nbbo' rows, or fall back?** Recovers it fully:
`nbbo_fallback.sum() == 0` of 6,523 `exit_half_src=='nbbo'` rows (0.0% fell back to the
fill_instant formula); the other 3,388 rows (`exit_half_src=='fill_instant'`) use that formula by
design, not fallback. `corrected_cost()`'s own log line would read "6523 nbbo-src rows, 0 fell
back (0.0%)".

**(b) Is slip applied to EOD/target exits?** Target: no, confirmed 0.0 mean slip_R for all 2,757
target rows and all 58 `eod_fallback` rows — code path is `if row.why not in SLIP_APPLICABLE_WHY:
continue`. EOD: **yes, by design** (`SLIP_APPLICABLE_WHY = {'stop','stop_bar','eod'}`, matching
cell 1,443's own `to_measure()` set which also measures eod's bid-at-first-print) — but the
**fallback bps for unmeasured eod rows is the stop-derived flat 35bps, not an eod-specific rate**,
which is the one real defect above (+0.0079R of the -0.057R gap, ≈14%).

**(c) Is the flat-30 variant on stop exits only?** **No.** `apply_stop_slip(fills, flat_bps=30.0)`
applies the flat 30bps to every row in `SLIP_APPLICABLE_WHY` — stop, stop_bar, AND eod — the same
population as the measured variant, just at a uniform 30bps instead of measured-or-35-fallback.
`net_R_corr_flat30` (-0.247) also carries the eod-at-stop-bps issue, at 30bps instead of 35bps
(smaller effect: mean slip_R flat30 = 0.1324 vs measured 0.1367).

## Verdict

The reported -0.247 (TRAIN-H2) / -0.255 (VAL) is **arithmetically correct given the code as
written** — no addition/subtraction bug, and the cost-fix half of the formula (recovering
half_entry via nbbo.csv) is exact with zero fallback. The gap to the PREREG's ≈-0.20 expectation is
a **calibration miss in the back-of-envelope, not a pipeline bug**: (1) 35bps of stop-price slip is
~0.21R per stop exit once divided by this book's small per-share R, ~2.5x the "~0.08R" assumed, and
(2) the expectation silently excluded the 13% of the book that exits at EOD, which also gets
slipped. Layered on top is one real, small, fixable defect: **eod's unmeasured fallback should use
an eod-specific pooled mean (~10-11bps per RESULT_1443.md), not the stop pooled mean (35bps)** —
worth ~0.008R (≈14% of the total gap) if fixed, applying to 564/9,911 rows (5.7% of the book).

## Queries run (reproducible)

```python
import sys; sys.path.insert(0, '/home/ec2-user/onemil')
from research.hod_entry import cell_1445 as c
fills = c.load_base_book()                                    # n=9,911
nbbo_lookup = c.load_nbbo_lookup()
half_entry, net_R_costfix, nbbo_fallback = c.corrected_cost(fills, nbbo_lookup)
# nbbo_fallback.sum() == 0 of (fills.exit_half_src=='nbbo').sum() == 6,523
slip_R = c.apply_stop_slip(fills)                              # measured-or-35bps-fallback
slip_R_flat30 = c.apply_stop_slip(fills, flat_bps=30.0)        # flat 30bps, same population
# means and groupby('why') as tabulated above
```
Script: `/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/reconcile.py`
(scratchpad only, not committed). No database written; `bars_sip.db`/`cache.db` not touched (Step
1 needs neither). `research/hod_entry/cell_1445_features.csv` and `causal_arming_causal.csv` were
only read.
