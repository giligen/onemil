# SPEC_FIX — correct the OFI pipeline before the verdict counts (main session, 2026-09-24, before any corrected number)

The first run (`REPORT.md`, 08:29 UTC) is VOID and its verdicts are not a result: the main-session review found coding
defects. Fix them in `research/hod_ofi/pipeline.py`, add unit tests, and prepare a guarded rerun. Sonnet implementer,
≤ 40 tool calls, write only under `research/hod_ofi/`, do not commit. Grep/offset reads only (pipeline.py is ~500
lines: read the functions you change, not the whole file).

**Market-hours rule:** between 13:25 and 20:05 UTC do NOT run the full `features` step (it reads all 1,131 raw files,
~70 min). Test on ≤ 2 days of raw data only, output to a separate file. The full rerun is a script you write and do
NOT launch.

## Defects → fixes (each with a unit test in `research/hod_ofi/test_pipeline.py`, synthetic data, pytest)
1. **CKS ask side has the wrong sign** (`ofi_updates`). Cont-Kukanov-Stoikov:
   e = [b>b' ? q_b : b==b' ? q_b − q_b' : −q_b'] + [a>a' ? +q_a' : a==a' ? q_a' − q_a : −q_a]
   (primes = previous record). Now: ask moved up → −q_a' (should be +q_a'); ask unchanged → q_a − q_a' (sign flipped);
   ask moved down → +q_a' (should be −q_a). Test: a hand-computed 5-record book (ask lifted, ask size added, ask
   improved, bid improved, bid size cut) with the exact expected e per record.
2. **Coverage counts seconds with an UPDATE, not seconds with a quote** (PREREG: "≥ 60 % of its window seconds have a
   quote"). New `coverage_frac` = share of the 300 window seconds s ∈ [end−300, end) at whose end (instant s+1, capped
   at `end`) the prevailing mbp-1 record — the last record at or before that instant, INCLUDING records fetched before
   the window start — is two-sided (bid > 0, ask > 0, ask ≥ bid). No prior record → unquoted. The main session measured
   this on 187 windows: 99 % usable vs 22 % under the old rule. Test: a book with one quote at t=−10 s and no updates
   → coverage 1.0; a book whose first record arrives at window second 150 → 0.5.
3. **Lee-Ready must use the quote strictly before the trade**: `merge_asof(..., allow_exact_matches=False)`, and all
   sorts by `sec` must be stable (`kind='mergesort'`) so same-timestamp records keep Databento's order (`sec` drops
   nanoseconds). Test: a same-timestamp quote change must not classify its own trade.
4. **Fallback end is not causal**: when no XNAS print ≥ the level is found in minute `entry_m`, end = S (the minute's
   START), not S + 60 (the old value used up to 60 s after the break). Keep the `locate_fallback` flag; report its share.
5. **Placebo ↔ signal join is ambiguous** when a name-day has several signals. Add `sig_entry_m` (the signal's
   `entry_m`) to BOTH rows in `build_windows` and to `window_features.csv`; join each placebo to its own signal on
   (day, symbol, sig_entry_m). Assert the seeded placebo minutes are unchanged: every placebo window of the 2 test days
   must find its raw rows (count and report any that do not).
6. **Score (`cmd_score`)**:
   * drop NaN feature rows before any cut / rank / Spearman (TSI and placebo Spearman were NaN);
   * `t_lift` = t of the keep dummy in OLS `net_R ~ 1 + keep` with day-clustered SE (statsmodels,
     `cov_type='cluster'`, groups = day); print the iid t beside it;
   * the pass-bar "deciles monotone (Spearman ≥ 0.6)" = Spearman between decile index 1–10 (qcut of the feature
     within that holdout, `duplicates='drop'`) and the decile's mean net R — NOT the trade-level rho (report that as
     info). Write both decile tables (TRAIN-H2, VAL: n, mean net R per decile) into REPORT.md;
   * placebo: cut the placebo feature at ITS OWN TRAIN-H1 median; placebo lift = kept − dropped mean net R of the
     matched signal's trade on TRAIN-H2 and VAL. PASS additionally requires placebo VAL lift < +0.10;
   * REPORT.md header: coverage, fallback share, winner/loser missingness gap, and a "Corrections vs run 1" section
     listing defects 1–6 verbatim from this file.
   Unchanged: the three cells, the TRAIN-H1 median cut, lift ≥ +0.10 on TRAIN-H2 and VAL, VAL t ≥ 2, ≥ 3 fills/wk,
   ex-top-5 % lift ≥ 0, the 60 % / 80 % / 5 pp availability rail.
7. **(Main-session amendment, 2026-09-24 ~19:55 UTC, before any corrected number.) The window end was a spec error.**
   `SPEC.md` clarification 1 assumed an intra-bar stop at the level. The HOD-break book enters at the OPEN of bar
   `entry_m` after the break bar (`entry_m − 1`) CLOSES (`research/bf_zero/spread_study.py:48`), so the decision
   instant is S = the start of minute `entry_m` — exactly the PREREG's "before the break bar closes". Every signal
   window now ends at S; `locate_fallback` is a diagnostic only (no XNAS print ≥ entry inside the entry minute). Found
   by the window-causality lens: 58 % of signal windows have ROUND-LOT XNAS prints ≥ entry in the minute before
   `entry_m` (natural for a next-open entry, impossible for a stop at the prior high), and the old end sat a median
   ~6 s (up to 60 s) after the decision.

## Deliverables
* Keep run 1: rename `REPORT.md`, `score_summary.json`, `window_features.csv` → `*_run1.*` (rename, never delete).
* `pytest research/hod_ofi/test_pipeline.py` green.
* A 2-day smoke (`features --days D1,D2` or equivalent → `window_features_smoke.csv`): print per-kind coverage,
  fallback share, and OFI_5 / TSI_5 quantiles. Sanity: median signal coverage ≈ 1.0.
* `research/hod_ofi/rerun.sh`: `nohup setsid`-safe; waits while UTC is in [13:25, 20:05); `nice -n 19 ionice -c3`;
  runs `features` then `score`; logs to `rerun.log`; ends with the same Telegram ping style as `after_fetch.sh`
  (no numbers). Do NOT launch it.

Return ≤ 150 words: tests passed, smoke coverage / fallback share, anything that surprised you.
