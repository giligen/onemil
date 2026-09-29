# RESULT_1665 -- relative volume to the arm minute vs. prior N sessions (1,438 population, 1.5% floor)

PREREG: research/hod_entry/PREREG_1665.md (FROZEN). Population: research/hod_entry/1663_features.csv (5,506 fills,
stop>=1.5% floor) + `level`/`fill_id`/`bucket_r` joined back in from causal_arming_causal.csv (status==fill) and
fills_1658.csv on (date,symbol), 1:1, exactly as cell 1,663's own join (validated: 0 join-gap ERROR/WARNING logged).

## Coverage line (read first)

| | RVOL_A(20) | RVOL_A(5) | rail |
|---|---|---|---|
| feature present | 4,162 / 5,506 = **75.6%** | 4,392 / 5,506 = **79.8%** | need >=80% |
| winner missingness | 28.3% | 24.5% | |
| loser missingness | 22.0% | 17.6% | |
| winner-vs-loser gap | **6.2pp** | **7.0pp** | need <=5pp |

**Both N fail both legs of the availability rail (CLAUDE.md item 7 / PREREG's own rail) -> Definition A is VOID.**
Definition B (ADV20 x P(m) fallback) was **not computed** -- out of the 45-tool-call budget once A's void was
established this late in the run. This cell is INCONCLUSIVE, not closed; B is the pre-declared next step (see Adequacy
review).

**Root cause of the shortfall, traced (not just measured):** 1,022 / 5,506 fills (18.6%) have **zero** bars_sip.db
rows for the symbol on the fill's own day at/after 09:30 ET -- a store coverage gap, not a level/price mismatch: of
every fill where the day *is* present in the store, the level bar was found 100% of the time (0 genuine "high does not
reproduce `level` within 1ct" cases). All 1,022 day-absent fills carry `missing_level=True`, which cascades into
`missing_a20`/`missing_a5`. The remaining shortfall (day present, but <10 / <3 *valid* prior sessions after excluding
prior days with no bars before m_arm) accounts for the rest. A cheap pre-check on raw `SELECT DISTINCT day` (no
09:30 filter) had suggested 91.3%/97.5% session-count eligibility -- that check does not require the level bar or the
CV window to resolve, so it materially overstated what this causal, fractional-minute-consistent feature can actually
deliver. **The differential winner/loser missingness (6-7pp, losers overrepresented in the available rows) means the
diagnostic table below is very likely biased negative relative to the true population -- read it as a sanity check
only, never as the cell's finding.**

## Reads (informational only -- Definition A is VOID; shown because the pass bar also fails outright, belt-and-braces)
Terciles/quintiles cut on pooled edges (available rows only), applied to both halves. iid t and day-clustered t both
shown; MDE = 2.802 x this read's own iid SD / sqrt(n), per PREREG_1665's literal wording ("the book's SD").

### RVOL_A(20) terciles
| bucket | half | n | mean net_R | iid t | day-clust t | MDE | ex-top5% | fills/wk |
|---|---|---|---|---|---|---|---|---|
| T1(low) | TRAIN-H2 | 555 | -0.098 | -1.77 | -1.09 | 0.155 | -0.208 | 20.6 |
| T2(mid) | TRAIN-H2 | 579 | -0.013 | -0.24 | -0.19 | 0.154 | -0.117 | 21.4 |
| T3(high) | TRAIN-H2 | 606 | -0.208 | -4.08 | -3.16 | 0.143 | -0.321 | 22.4 |
| T1(low) | VAL | 833 | -0.043 | -0.95 | -0.60 | 0.128 | -0.150 | 37.9 |
| T2(mid) | VAL | 807 | -0.094 | -2.08 | -1.46 | 0.127 | -0.201 | 36.7 |
| T3(high) | VAL | 782 | -0.085 | -1.81 | -1.35 | 0.132 | -0.192 | 35.5 |

### RVOL_A(5) terciles
| bucket | half | n | mean net_R | iid t | day-clust t | MDE | ex-top5% | fills/wk |
|---|---|---|---|---|---|---|---|---|
| T1(low) | TRAIN-H2 | 566 | -0.110 | -2.00 | -1.19 | 0.154 | -0.218 | 21.0 |
| T2(mid) | TRAIN-H2 | 644 | -0.006 | -0.12 | -0.09 | 0.146 | -0.109 | 23.9 |
| T3(high) | TRAIN-H2 | 637 | -0.204 | -4.09 | -3.42 | 0.140 | -0.319 | 23.6 |
| T1(low) | VAL | 898 | -0.095 | -2.19 | -1.32 | 0.122 | -0.204 | 40.8 |
| T2(mid) | VAL | 820 | -0.062 | -1.34 | -0.95 | 0.128 | -0.168 | 37.3 |
| T3(high) | VAL | 827 | -0.075 | -1.65 | -1.29 | 0.128 | -0.181 | 37.6 |

Every bucket in both halves, both N, is net-negative -- no tercile is even close to the +0.05R floor, let alone
clearing it. T3(high) is the *worst* TRAIN-H2 bucket for both N (day-clustered t -3.16 / -3.42), i.e. the point
estimate runs opposite the owner's hypothesized direction; given the VOID rail and the missingness skew above, this
is not read as "low RVOL is good," only as "nothing here clears the bar."

Quintiles, the top-tercile-vs-rest paired cut, the stop-bucket x RVOL interaction, and the Spearman rho (all 4 reads
x 2 N x 2 halves, 64 rows total) are in `1665_reads.csv`; none changes the verdict.

## Pass-bar verdict
Rebuilt mechanically (net>=+0.05R AND day-clustered t>=2.5 AND ex-top5%>0 AND fills/wk>=3, all four on BOTH halves):
**0 / 32 (cut,bucket) combinations pass**, across terciles, quintiles, top-vs-rest and the interaction cells, for both
RVOL_A(20) and RVOL_A(5). Combined with the VOID availability rail, cell 1,665 ships nothing to paper.

## Adequacy review (MDE vs. the +0.05R lift the pass bar requires)
MDE at the full floored n (5,506, iid SD 1.335): **0.050R** -- already sitting exactly on the pass-bar floor with zero
margin. At the RVOL_A(20)-available n (4,162): **0.056R** -- *above* the +0.05R lift this cell exists to detect. So
even setting the VOID rail aside, this cut's own sample size cannot reliably distinguish "no lift" from "a lift of
exactly the minimum size that would ship" -- a null read here is a claim about this test's power, not evidence the
volume feature carries nothing. This is why B (or a bars_sip.db backfill for the missing 1,022 fill-days) is the next
step, not a closure: the population may still carry a lift the current coverage cannot resolve.

## Not allowed -- compliance
N and the definition were pre-declared (both N run, A is primary); no conditioning on exit_type; fill bar volume never
entered CV (search window and CV cutoff both stop at least 1 minute before the fill-containing bar); no same-day
ratio computed; stop floor reused verbatim from 1663, not refit.

## Files
`research/hod_entry/1665_features.csv` (5,506 rows: fill_id, day, symbol, split, r_pct, bucket_r, net_R, rvol_a20,
rvol_a5, rvol_b [all-NaN, B not run], m_arm, missing_level, missing_a20, missing_a5, n_prior_store_days);
`research/hod_entry/1665_reads.csv` (64 rows, one per (cut,bucket,half)); `research/hod_entry/1665_rvol.py`
(feature build + `--stats-only` reads/coverage regeneration); `research/hod_entry/1665_rvol.log`. 0 ERROR/WARNING
lines in the run log (clean join, no fallback paths triggered).
