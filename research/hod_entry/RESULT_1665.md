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
Definition B (ADV20 x P(m) fallback) was computed in a follow-up pass (see "Definition B" section below) and is
**also VOID** on the missingness-gap leg. With both A and B void, cell 1,665 is genuinely CLOSED on this population at
this coverage -- not "no edge," a claim about what this store can resolve (see Adequacy
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

## Definition B (coordinator follow-up, 2026-09-29 -- fresh 25-call budget)
RVOL_B = CV(d, m_arm) / (ADV20(d-1) x P(m_arm)). CV(d,m_arm) reuses A's own `m_arm` (no re-search for the level bar,
so this carries no new causality risk). ADV20(d-1): asof from `research/overnight_high/panel_2024_2026.parquet`,
last panel row **strictly before** the fill date (same causal convention cell 1,663 used for ATR14). P(m): for every
TRAIN-H2 (symbol,day) pair with that day present in the store, the cumulative-volume fraction (bars closed before m,
same fractional-minute rule as CV, over 09:30-16:00 ET) at every integer minute 570..960; P(m) is the cross-sectional
median of that fraction across 1,898 TRAIN-H2 (symbol,day) curves, looked up at each fill's own integer `m_arm`
(re-estimating this from the *same* bars-store days already loaded for A, not a fresh broad pull, per the
coordinator's instruction). A second bars_sip.db pass was required since raw bars from A's run were not persisted to
disk (only the aggregated CV outputs were) -- 5/4,484 fills (0.11%) picked up a day now absent on re-fetch
("unexpected re-fetch misses" in the log); negligible, noted rather than hidden.

### B coverage vs. the rail
| | RVOL_B |
|---|---|
| feature present | 4,436 / 5,506 = **80.6%** (clears the >=80% leg) |
| winner missingness | 23.8% |
| loser missingness | 16.7% |
| winner-vs-loser gap | **7.1pp** (fails the <=5pp leg) |

B clears coverage but fails the missingness-gap leg -- **B is also VOID.** The same winner/loser skew seen under A
persists (winners more likely to lack a resolvable `m_arm`/day than losers), so B's diagnostic numbers below carry the
same negative bias caveat as A's.

### B reads (informational only -- VOID; read 1 + read 3 per the coordinator's scoped request)
Terciles, pooled edges on available rows, both halves:

| bucket | half | n | mean net_R | iid t | day-clust t | MDE | ex-top5% | fills/wk |
|---|---|---|---|---|---|---|---|---|
| T1(low) | TRAIN-H2 | 619 | -0.093 | -1.79 | -1.23 | 0.145 | -0.201 | 22.9 |
| T2(mid) | TRAIN-H2 | 633 | -0.076 | -1.44 | -1.06 | 0.148 | -0.184 | 23.4 |
| T3(high) | TRAIN-H2 | 622 | -0.153 | -2.95 | -2.15 | 0.145 | -0.264 | 23.0 |
| T1(low) | VAL | 860 | -0.126 | -2.93 | -1.97 | 0.121 | -0.236 | 39.1 |
| T2(mid) | VAL | 845 | -0.019 | -0.42 | -0.33 | 0.128 | -0.123 | 38.4 |
| T3(high) | VAL | 857 | -0.087 | -1.92 | -1.32 | 0.127 | -0.195 | 39.0 |

Quintiles (5 x 2 halves) and the stop-bucket x tercile interaction (2 x 3 x 2 halves) are in `1665_reads.csv`
(cuts `read1_quintile_B`, `read3_interaction_B`). Same pattern as A: every tercile/quintile is net-negative both
halves except two thin, non-significant `>=3%` interaction cells in VAL (T1/T2, net +0.12/+0.12 but day-clustered
t 0.91/1.03, n 126/148 -- nowhere near t>=2.5). **0 / 14 (cut,bucket) combinations pass the pass bar** (same
net>=+0.05R AND day-clustered t>=2.5 AND ex-top5%>0 AND fills/wk>=3 on both halves).

### Combined verdict
Both Definition A (N=20 and N=5) and Definition B fail the availability rail; 0/32 + 0/14 = **0/46 reads pass** across
every pre-declared cut this cell ran. Per PREREG_1665's own rail language ("if both are VOID the cell is VOID and
says why"): cell 1,665 is VOID -- relative volume to the arm minute cannot be resolved on this population at this
bars_sip.db coverage (81% same-day, ~76-81% after the prior-session/P(m) requirements stack on top), and the
resolvable subset is not a random sample (6-7pp winner/loser missingness gap under every variant tried). This is a
statement about what bars_sip.db can currently answer, not a statement that relative volume carries no signal for
HOD-break fills -- the adequacy review above (MDE ~0.05-0.06R against a +0.05R bar) already flags that even a valid
read would be marginally powered at this n. Next step, if the owner wants this resolved rather than closed: backfill
the missing symbol-days in bars_sip.db for this population (the 1,022+ day-level gaps are the root cause of every
leg's failure), then re-run this exact script unmodified.

## Files
`research/hod_entry/1665_features.csv` (5,506 rows: fill_id, day, symbol, split, r_pct, bucket_r, net_R, rvol_a20,
rvol_a5, rvol_b [4,436/5,506 populated], m_arm, missing_level, missing_a20, missing_a5, n_prior_store_days);
`research/hod_entry/1665_reads.csv` (92 rows: 64 Definition-A + 28 Definition-B, one per (cut,bucket,half));
`research/hod_entry/1665_rvol.py` (feature build + `--stats-only` reads/coverage regeneration + `--defB` Definition-B
pass); `research/hod_entry/1665_rvol.log` (both runs appended). 0 ERROR lines in the run log; the only WARNING-level
signal is the 5-row re-fetch-miss count logged at INFO (see Definition B).
