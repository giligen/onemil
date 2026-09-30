# Refuter 1: cell 1,599 (v3, Amendment 3). Lens: obtainability, look-ahead and data

**Verdict: REFUTED as "built and verified".** No v3 number exists yet: cell_1599_cycles.csv has only a header and
RESULT_1599.md says BLOCKED. That BLOCKED status is correct. What is wrong is the claim that the pipeline is ready to
run once the fetch finishes. It has three defects that would change the verdict and one that makes the extension
test impossible to evaluate. Fix all of them before anyone runs `cell_1599.py` on the finished cache.

Probe scripts and data are in `review/r1599_1/` (`probe.py`, `void_est.py`, `cbbo_*.parquet`, `void_est.csv`,
`probe_out.json`). My own Databento spend was **$0.1213** (9 purchases of the SPY.OPT cbbo-1m window at 10:00–10:02 ET),
logged in `review/r1599_1/spend_own.json`.

## Defects that change the verdict
1. **The leg files lose the sample timestamp.** In `fetch_dbn.py` (around line 534), `data.to_df()` is followed by
   `to_parquet(index=False)`. That drops `ts_recv`, which is the index `to_df()` sets. Only `ts_event` survives, and in
   cbbo-1m `ts_event` is the time of the *last quote update*, not the time of the sample. Measured on 9 sessions: the
   10:00:00 samples have `ts_event` older than 1 minute for 29–44 % of puts, and a median age of 100–870 s.
   `LegCache.quote_at` finds the 15:59 mark and the 10:00 exit by `ts_event` minute. As a result:
   - a leg whose quote last changed before 15:59 has no "15:59 mark";
   - with both legs required, roughly half of the sessions are skipped (`mark_missing`);
   - management-A's stop, its profit target and the 21-DTE exit are checked only on a random subset of days;
   - exits slip to later sessions or to `*_fallback_expiry`, which is intrinsic value with no spread cost.

   This is not look-ahead, but management A as coded is not the rule in the PREREG.
   **Fix:** keep `ts_recv` (`reset_index()` before saving) and key the lookups on it.
2. **The VOID rail breaks for the 20-delta cells.** I replayed the builder's selection rules on the 125 panel entry
   sessions in the rebuild's 10:00 snapshots (`void_est.csv`). The rules: the expiry nearest 45 DTE within [38, 52],
   strikes within [0.8·spot, spot], the nearest-delta short strike, and a partner exactly K−10 (`np.isclose`).
   - At 20 delta, **18 of 124 cycles (14.5 %) are VOID** with `no_partner`. The short lands on a half-dollar strike or
     a $1 strike on a sparse weekly grid, and K−10 is not listed. Cells 1,599–1,602 are over the 10 % rail at entry
     alone.
   - At 30 delta, 11 of 124 (8.9 %) are VOID, just under the rail.

   The sparse weeklies come from the nearest-45 rule (for example, 2025-07-07 picks 2025-08-22, which had 29 strikes,
   instead of 2025-08-15, which had 241). This rule is frozen, so the VOIDs are real. The builder has not stated this
   outcome in advance.
3. **Sizing skips never reach the cycle table.** When `contracts <= 0` the code just does `continue`. On 2024-08-05, the
   spike Monday, the 20-delta credit at bid/ask was **−$1.15** (mid +$0.74) with 1×1 sizes, so the week vanishes
   without being counted against the rail. This is the same kind of defect as v1/v2: rows missing from the table are
   correlated with the spike. The effect is small (1 of 124), but it is the defect the amendment was written to remove.
4. **The extension cannot be evaluated for 2013-04 → 2015-12.** `spy_prices.parquet` marks 691 days as `GAP`, so
   `spot_10` and `spot_16` are NaN.
   - In the gated cells, `iv_gate_pass` returns False on a NaN spot. About 140 weeks are silently skipped, not VOIDed,
     and that includes the 2015-08-24 spike.
   - In the ungated cells those weeks are VOID, which puts the extension far over the 10 % rail.
   - Separately, 21 of the first ~47 Mondays of 2013–14 have no expiry between 38 and 52 DTE (`fetch_dbn.log`).
   - The "≥ 8 of 11 years positive" test and the 2015-08 spike read cannot be run as frozen.

   A spot source, for example put-call parity on the same cbbo bar, has to be registered **before** the extension is
   read.
5. **The $150 cap is not enforced.** `fetch_dbn.py` loads `spend.json` once, keeps the running total in memory, and
   rewrites the whole file on every save. `rebuild_1599.py` writes to the same file. The ledger total went from
   $0.4609 down to $0.3153 between two of my reads. My 9 purchases ($0.121) have since been erased, and the 125
   rebuild snapshots and 30 rebuild legs have 0 entries in the ledger. The real spend is higher than `spend.json` says.

## Checks that pass or only need a caveat
- **Order size against the quote.** Every panel cycle is sized to 1 contract (credit $0.81–2.32, worst case
  $770–920 against B/6 = $1,083). Quoted sizes are usually 40–470 contracts; on spike days they fall to 1×1
  (2020-03-17 short leg, 2024-08-05). A 1-lot order at the NBBO is still obtainable.
- **Spread cost on spike days.** The round-trip spread cost at 30 delta was:

  | Session | Round-trip cost |
  |---|---|
  | 2018-02-06 | $0.42 |
  | 2020-03-16 | $0.55 |
  | 2020-03-17 | $1.28 |
  | 2025-04-04 | $0.39 |
  | 2025-04-07 | $0.52 |
  | 2022-06-13 (20 delta) | $0.09 |
  | 2024-08-05 (20 delta) | **$8.09** |

  Every one of these sessions had a two-sided 10:00 quote on at least 92 % of puts.
- **Exit cost is not capped at the spread width.** A 10:00 exit on a market like 2024-08-05 can book a loss above
  W−credit. That is conservative, but it breaks the worst-case budget.
- **Look-ahead at entry.** Selection and fill use the same 10:00:00 sample (`first_bar` sorted by `ts_event`, which
  is the oldest sample and so causal). Management A triggers on the 15:59 mark and acts at the next session's 10:00,
  so there is no look-ahead. The IV gate uses the same 10:00 chain.
- **SPY prices from 2016 on** are raw and match (2016-01-04 close 200.99, 2020-03-16 close 239.41). Settlement uses
  the 15:59 close, which is fine for PM-settled SPY options.
- **Fixed equity of $65K.** `assert_budget` always passes (6 × ≤ $920 < $6,500). A fixed $10 width, though, is about 6 %
  of spot in 2013 and about 1.5 % in 2026. The extension trades a structurally different spread, so it should be
  reported per year and not pooled.
- **Minor.** The $0.03 per contract regulatory fee in the PREREG is not charged in `cell_1599.py`. Early assignment and
  dividends are not modelled; the loss stays bounded by the width.
- **The rebuild does not handle holidays.** It sends Monday windows on holidays (422 errors on 2024-02-19, 2024-09-02,
  2025-01-20 and others), while the builder rolls to the next session. The builder-vs-rebuild comparison will
  therefore mismatch on about 6 weeks a year.
