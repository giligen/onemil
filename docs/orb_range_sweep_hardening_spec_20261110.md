# ORB post-open range sweep: retry until filled + loud fallbacks; selection observer quiet on green days — spec 2026-11-10

Owner incident 11/10 (ORB PAPER). Alpaca data was slow at the open (scanner snapshot fetch 61.7 s / 79.6 s vs ~27 s
normal, two "Cycle overrun" WARNINGs 14:32/14:34 UTC). `ORBEngine._post_open_sweep` (`trading/orb_engine.py`
~1726–1918) ran at 09:35:00 ET, `_fetch_sweep_bars` returned non-empty bars for all 59 rangeless candidates,
`_ingest_batch` applied **0** (fewer than 5 bars in 09:30–09:34 — the 09:34 bar had not consolidated), retried ONCE
after `sweep_retry_delay_s` (4 s), 0 again, logged "filled 0/59 ranges" at INFO and set `_post_open_range_sweep_done`.
No "range complete" line was logged all day; ORB ranked 4 names + 2 preplaced instead of ~8–26. Reference days:
11/9 filled 148/172 (134 + 14 on retry), 11/6 209/245, 11/5 111/126. The same REST bars fetched at 15:00 UTC today
give NVD/BEZ 5/5 bars and `_first_session_open_ts_utc` = 14:30Z → the logic is fine, the vendor was late.
The observer cron `scripts/orb_selection_observer.py` (13:36 / 14:36 UTC) correctly flagged 21 REAL drops today; on
ordinary days it sends "No REAL drops — Observer on duty", which the owner calls noise.

Rules for the agent: no git (no commit/stash), no service action, no orders, never write orb.yaml / config.yaml /
.env / data/*.db / crontab / logs state files. pytest only on the listed files, `--ignore=tests/integration`, one
run at a time, never piped through head/tail. This task IS the owner's request; do not pivot. Budget ≤ 40 calls.

## 1. Sweep: retry until filled, bounded by the clock, loud when it gives up
In `_post_open_sweep`:
- Replace the single retry with a loop: after the first pass, while `still_missing` and ET now < `sweep_deadline_et`
  (new `entry_cfg['sweep_deadline_et']`, default `"09:40:00"`), sleep `sweep_retry_delay_s` (keep 4 s default; read
  from config as today) and re-fetch ONLY `still_missing`. Each pass: `_log_fetch(..., tag=f' retry {k}')`. Keep the
  `_day_rolled()` guards and the lock discipline exactly as they are (fetch outside, ingest inside).
- Keep `_sweep_retry_used_today` semantics compatible (set True after the first retry) so existing tests still hold;
  add `_sweep_retry_count_today`.
- Classify the leftovers per pass (cheap, from the fetched frame): `no_bars` (None/empty), `short_window`
  (bars present, < 5 in [session_open, +5 min)), `no_open_bar` (`_first_session_open_ts_utc` None). Log the final
  outcome as:
  - all filled → INFO "filled n/n".
  - leftovers → WARNING "post-open range sweep gave up at HH:MM:SS ET — m/n rangeless: short_window [..], no_bars
    [..], no_open_bar [..]" (first 8 names per class).
  - filled < 50 % of `missing` AND `len(missing) >= 10` at the deadline → `logger.error("[ORB] post-open range sweep:
    vendor bars late — only f/n candidates have a range by HH:MM ET (short_window m) — ORB is trading a thin field
    today")` once per day. This is the degradation alarm; it must NOT fire on a normal day (11/9: 148/172 = 86 %).
- The first ranking must not wait for the loop: the sweep already returns after the first pass's `filled_syms` to
  `check_entries` via the first-rank grace; confirm from the code how `_ensure_ranges_post_open` is called (drain
  thread vs tick) and make the retry loop run where the single retry ran, so the engine's tick thread is blocked no
  longer than today's `retry_delay + fetch` per pass. If the sweep runs on the tick thread, cap each pass's fetch at
  the existing budget and return between passes (i.e. make the retry re-entrant on the next tick: `not done` until
  deadline) rather than sleeping in a loop — pick the one the current call structure supports and document it in a
  comment. Tests must cover the chosen structure.
- `_ingest_batch`: when bars are non-empty but the range does not complete, count `short_window`/`no_open_bar`
  instead of silently continuing.

## 2. Selection observer: quiet on green days, correct window
`scripts/orb_selection_observer.py`:
- `_classify_drop`: the bars query `end` must be exclusive of 09:35 (use 09:34:59 or filter `t < 09:35`) so the
  count is out of 5, never "6/5".
- Verdict `'clean'` (no REAL drops): log the line, append it to `logs/orb_observer_cron.log` as today, **do not send
  Telegram**. Verdict `'caught'` and `'ops_warning'`: unchanged (Telegram).
- `scripts/report_common.py` / the EOD report: if there is an existing daily hygiene/selection line, make sure it shows
  "selection parity: live ranged x / BT-eligible y, REAL drops z" from `logs/orb_selection_audit_<date>.json`; if
  there is none, add exactly one such line to the EOD text where the other ORB hygiene lines are built. ≤ 15 lines.

## 3. Tests (unit + integration per the bug protocol)
`tests/test_orb_range_sweep_hardening_20261110.py` (+ edits to existing sweep tests only to match the new
log/structure, never deleting a test):
- unit: a fake `alpaca.get_1min_bars_multi` that returns 3 bars in the window on passes 1–2 and 5 bars on pass 3 →
  range completes on pass 3, counts/logs as specified, `_post_open_range_sweep_done` True afterwards.
- unit: deadline reached with leftovers → WARNING with the three classes, no ERROR when ≥ 50 % filled.
- unit: 0/59 at the deadline → ERROR fired once; a second call the same day does not fire again.
- unit: 11/9 shape (134 then 14 of 172) → INFO only, no ERROR, exactly one retry.
- integration: real `ORBEngine` with the standard test fixture (see `tests/test_orb_engine*.py` for how the engine is
  built with a mocked client, `spec=`), frozen ET clock at 09:35:00 → sweep → `candidates[sym].range_data` set with
  the right H/L for the fed bars; `_day_rolled` during a retry → nothing ingested.
- observer: `_classify_drop` window excludes 09:35 (mock requests); clean verdict sends nothing (mock notifier),
  caught verdict sends.
- Run: `python3 -m pytest tests/test_orb_range_sweep_hardening_20261110.py tests/test_orb_engine*.py
  tests/test_orb_preplace*.py tests/test_orb_selection_observer*.py -q --ignore=tests/integration` → 0 failures
  (if an observer test file does not exist, create it with the observer tests above).

## Output
`docs/orb_range_sweep_hardening_20261110_RESULT.md` ≤ 200 words: structure chosen in §1 (loop vs re-entrant),
file:line of the changes, test counts, anything you could not do. Return ≤ 120 words.
