# Cell 1,733 — why does the live ORB paper book pick different names than the BT? — PREREG 2026-11-11

Follow-on to cell 1,732 (`RESULT_1732.md`, `stops_1732.csv` — read both first): on 9/29–11/9 the live paper book
and the nightly production BT share only 21 of 32 live fills; 11 live fills are names the BT never entered (7 of
them stopped, −$690 in total) and the BT entered 10 names live never filled. Stop MECHANICS are at parity. The
owner's ask is "fix ORB": the fix is selection parity — ONE spec for BT and live (CLAUDE.md). This task IS the
owner's request; do not pivot. READ-ONLY diagnosis, writes only under `research/book_review_20261111/`
(`RESULT_1733.md` or `.txt`, `selection_1733.csv`, scratch .py). No code changes, no git, no orders, no service
action, no writes to data/*.db / config / orb.yaml / .env / crontab / logs/*. Never print API keys. Never Read a
file > 300 lines in full. Journal greps ALWAYS bounded: `journalctl -u onemil-trader --since "<date> HH:MM"
--until "<date> HH:MM" --no-pager -o cat | grep -E "..."` (EDT dates ≤ 10/30: open = 13:30 UTC; EST from 11/2:
14:30 UTC). Budget ≤ 40 calls. Return ≤ 150 words.

## Inputs
- BT book: `analysis_results/orb_bplus_book.csv` (built nightly, covers through 11/10); candidate sim:
  `analysis_results/orb_features_20261110_2203.csv` (per symbol-day features, `entered`, exit, rank/score columns —
  find them). The BT's ranking/slot rule: `study_orb_pipeline_static_lock.py` (top-K per day, K = ?, tie-break,
  vetoes — quote file:line) and the shared helpers in `trading/orb_planner.py`, `orb_filter.py`, `orb_gap_gate.py`,
  `orb_range_size_veto.py`, `orb_pdr_veto.py`, `orb_g1_veto.py`, `orb_anchor_dedup.py`, `orb_correlation.py`.
- Live: journal `[ORB PREPLACE]` lines per day (provisional top-K @ 09:34:58 with rh/Q, provisional vetoes, final
  reconcile/PARITY line, "ORB ENTRY SUBMITTED", late-window admissions), e.g. 11/3 14:34:50–14:40 UTC. Live config
  `orb.yaml` (slots, `preplace_at_close`, `preplace_rank_lead_s`, late-submission window, vetoes, `rvol_tilt`,
  `addon_pools` — read only the relevant keys with grep). Ledger rows: `stops_1732.csv` (bt_taken, feat_present,
  feat_entered per live fill) and trades.db (ro) `pattern_data` JSON (rank, score, features at decision time, if
  stored). Observer audits `logs/orb_selection_audit_<YYYYMMDD>.json` (`expected_candidates` is a candidate pool,
  not a top-K — note its limits).

## Questions
1. Per day (14 live days + the days the BT entered names live did not): a table BT top-K (symbol, rank, score,
   gap %, range %, price, Q bucket, veto flags) vs live provisional top-K vs live final submissions vs live fills,
   with the same feature values from the live log. Write it to `selection_1733.csv`.
2. Classify every divergence (both directions, 11 live-only fills + 10 BT-only fills + any submitted-not-filled):
   U universe (name absent from the BT candidate sim — ASTX 9/30, ASTY 10/12, SK 10/23: why? listing date, daily_bars
   coverage, wrapper/test-ticker filter, PIT universe vs nightly), V veto applied on one side only (which veto, which
   input differed — provisional range vs final range, prev-day range source), R ranking (same candidates, different
   order: which feature differs and by how much — snapshot daily bar vs BT daily bar, rvol, score formula), K slot
   count (live K ≠ BT K; add-on pool slots), T timing (BT picks once at 09:35; live admitted the name later / the
   BT's name never triggered its buy-stop live / the live order was cancelled — preplace reconcile, STRADDLED,
   lock-wait), P pool (`addon_gap35_range5` fills have no BT book), O other. Sum $ and stop counts per class.
3. For class R specifically: recompute the live score from the BT's features for those days and show whether the
   BT's own top-K would have been produced had the live engine used the final 09:35 range and the BT's daily bar.
4. The fix per class as file:line + ≤ 10 lines of pseudo-code, and the PARITY TEST that would hold it: a nightly
   check (extend `scripts/orb_selection_observer.py` or the EOD report) asserting live-submitted == BT top-K per day
   with a named reason for every difference, Telegram ERROR when an unexplained difference appears. Say which
   classes are legitimate by design (e.g. the preplace provisional ranking exists for latency — then the final
   reconcile must converge to the BT set) and which are defects.
5. Re-sim: the paper book's realized $ and stop share had live traded exactly the BT's names (from the BT rows), and
   the BT's own realized on those 14 days — both already partly in RESULT_1732 §3; make them exact and day-clustered.

## Output
`RESULT_1733.md` ≤ 600 words + tables; `selection_1733.csv`. Return ≤ 150 words: divergence classes with $ and
counts, the root cause(s), the fix list in priority order, the parity test.
