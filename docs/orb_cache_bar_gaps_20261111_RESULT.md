# ORB BT cache: opening-range bar gaps 10/1-11/10 (2026-11-11)

**Finding: the premise "cache gaps" is mostly wrong.** 1,159 candidate symbol-days (production gates, no bars-exist
clause): 1,053 complete (5/5), 99 partial, 7 empty (9 %; 11/6 worst, 26/161). The gaps span all of October, not just
11/2-11/10. SIP spot-check of 20 gapped symbol-days, minute by minute (`research/book_review_20261111/cache_gaps_1111.csv`,
`measure_cache_gaps.py`): cache == SIP on 20/20 (18 partial, 2 empty where SIP also has no bar). The missing minutes
are minutes with NO TRADE (thin names), not vendor gaps. 0 of 28 live ORB fills sit in a partial/empty symbol-day.
The 67 `range_not_computable` drops are therefore thin names; whether live may range them on < 5 bars is a LIVE-vs-BT
rule question (not a data defect) for the selection-parity work.

**Root cause search:** writer = `orb_backtest.py` `fill_intraday_for_pairs` (L316+; window 13:00-20:30 UTC, per symbol,
skips any pair that already has ANY bar via `_intraday_bars_cached_pairs` L231). No failure logged 11/2-11/10 (journal:
28 pairs / 9,823 bars on 11/2). Latent defects (not the cause here): (1) the any-bar skip never repairs a partial pair;
(2) the fixed 20:30 UTC end truncates the last 30 min after the 11/1 DST end (15:30 ET) - affects exits, not the range.

**Fix:** `batch/orb_bar_gap_gate.py` (gate: WARNING with count + first 20 names; ERROR when > 10 % of the latest day
was REALLY missing = SIP had minutes the cache lacked; `--refill-gaps --start --end` re-fetches only gapped windows,
INSERT OR IGNORE), wired after the nightly fill in `orb_backtest.py` (`_run_bar_gap_gate`, never aborts). ERROR goes to
the log; Telegram relay depends on the timer's wrapper (not added).

**Tests:** `tests/test_orb_cache_gaps_20261111.py` 7 passed (6 unit/temp-sqlite, 1 real-SIP integration on a copy of 2
symbol-days: restores exactly the deleted minute, re-run adds 0); `test_orb_backtest_cli_validation.py` + vol935: 12 passed.

**Session command (not run):**
`python3 batch/orb_bar_gap_gate.py --refill-gaps --start 2026-09-29 --end 2026-11-10` (writes data/cache.db).
Expected to add ~0 bars. A features/book rebuild is needed only if it adds any bar; the earlier planned rebuild is
for the 15K-proxy -> bar-count universe change, not for this.
