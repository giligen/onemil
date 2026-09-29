# PULL_REPLAN_20260929 -- SPY put-spread ladder legs: remaining plan, scoreability, batched mode

## 1. Remaining-plan table (candidates = unique OSI in mondays.parquet, by entry-year range)
| Range | Candidates | Cached | Remaining | Projected $ (remaining) |
|---|---|---|---|---|
| 2016-2023 | 16,102 | 0 | 16,102 | $32.38 |
| 2018-2023 | 12,853 | 0 | 12,853 | $25.84 |
| 2013-2023 | 23,499 | 3,603 | 19,896 | $39.85 |

Method: per-leg cost from spend.json purchases (regex-matched on the per-leg `cbbo-1m [raw_symbol]` label only,
excluding the Monday-ladder definition/quotes overhead calls), grouped by expiry year (parsed from the OSI) x
full-life-window flag (>=40 days vs truncated-at-today). Observed FULL-window mean/leg: 2013 $0.00201, 2014
$0.00201, 2015 $0.00197; pooled FULL mean $0.00201 used for 2016-2023 (no observed sample yet) -- all remaining
legs there are historical, so none would be truncated. Caveat: 0 of the 5,030 already-cached legs are 2016-2023
mondays.parquet candidates -- 3,603 cached legs are the 2013-2015 candidates; the other 1,427 (the run's "2016-2026:
~120/yr") are superset partner strikes OUTSIDE the band, absent from mondays.parquet (see #3). 2016-2023 progress is
genuinely 0 legs, not ~13%.

## 2. Is 2013-2015 scoreable without the SPY underlying price?
YES, mechanically, per the spec's own words. REBUILD_1599.md (independent build under PREREG_1567 Amendment 3):
"SPY spot 2013-04-08..2015-12-31 (EXTENSION only) has no Alpaca/Databento equity source; used put-call parity from
the same 10:00 options chain -- flagged `used_parity_spot`." But that EXTENSION read carries void_rail = 0.65,
far past Amendment 3's own VOID cap ("rail 10%... > 10% -> the cell is VOID"). So: scoreable in principle via
parity; the one attempt on record is itself VOID by the spec's own rule (the leg cache was too incomplete at read
time), not yet a reliable score.

## 3. Is the candidate set wider than the ladder's rule requires?
Reconstructed the delta rule (0.20/0.30-delta +/-3 strikes, +$10-below partner, deduped) offline from
mondays.parquet's own cached bid/ask via fetch_dbn's own bs_put_delta/implied_vol_put -- no network, no purchase.
Over the 547 Mondays with known spot: band-mask rows (what fetch_legs actually queues) average 53.9/Monday; the
rule needs 24.0/Monday deduped -- **2.24x wider than necessary**. For the 116 unbanded 2013-2015 Mondays (spot=NaN)
`fetch_mondays` applies no mask at all (`else: strikes_by_leg = first_bar`) -- average 80.1 rows/Monday, unbounded
vs. the rule, since delta can't even be computed there. This band-mask, not the ladder's own math, is why 2013-2014
already hold 71% of all spend to date.

## 4. Batched mode (`--batched`, not run)
`fetch_legs_batched` groups leg_plan by (first_seen, expiry) [identical per Monday by construction], drops
already-cached legs per group, issues ONE guarded_get_range per group (zero calls if the group is fully cached),
splits the returned frame by `symbol` into the existing per-OSI parquet files; missing legs in the response are
logged WARNING and skipped, never silently dropped. Tests: `tests/test_fetch_dbn_batched.py`, 10 tests, mocked
client, no network -- all PASS (grouping, resume-skip, fully-cached-skip, split correctness, missing-leg
WARNING+no-crash, single-leg no-symbol-column edge case, cost-only no-purchase, one spend-ledger entry per batch,
spend-cap-hit skip).

Real `metadata.get_cost` check, entry_date=2016-09-06 (44 uncached legs, same window both ways): BATCHED (1 call)
= $0.09028; SUM OF SINGLES (44 calls) = $0.09028 -- **identical, ratio 1.0000**. Databento bills by data volume, not
request count: batching saves wall-clock and API round trips (~17s/call overhead x 44 -> 1 call), **not dollars**.
The $ totals in #1 will not shrink under `--batched`; the same spend reaches the cap in far less wall-clock time.

## 5. Blocker found (pre-existing, not part of this change)
`fetch_mondays()` only returns non-empty `leg_plan` for Mondays it freshly processes this run (`todo`); once a
Monday is in mondays.parquet, `main()` never rebuilds its leg_plan (`if args.stage in ('legs','all') and leg_plan:`
is falsy). 693 total entry Mondays are in scope; 663 are already in mondays.parquet, 30 remain `todo`. A plain
rerun -- batched or not -- pulls only those 30 new Mondays' legs, NOT the 16,102-leg 2016-2023 backlog. Draining it
needs a small follow-up: reconstruct leg_plan from mondays_df's own cached rows (same formula as #3) before
`--stage legs` runs. Flagged here, not fixed -- out of this task's scope.

## Resume command (once the #5 blocker is fixed)
`python3 research/options_vrp/fetch_dbn.py --stage legs --batched`
