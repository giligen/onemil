# ORB: measure the decision→broker gap per order, tripwire on the broker clock (spec, Sat 2026-10-17)

Facts (`docs/week_review_20261012.md` §2): the engine's "first order submit 8.5 s" is its own stamp; the broker's
`submitted_at` for the same orders is 4.8–10.1 s later (CORD 10/16: engine 8.1 s, broker 15.3 s; ASTY 10/12: 26.4 vs
31.2). Node clock is NTP-synced (offset < 1 µs, checked 10/17) — the gap is real client-side time between the decision
and the HTTP call reaching Alpaca. The latency BT (`research/orb_latency_bt/REPORT.md`) says the per-fill edge is
intact to 20 s and ≈ 0 at ≥ 30 s; we have been measuring the wrong clock. This task INSTRUMENTS; it changes no decision.

## Required (`trading/orb_engine.py`; grep: `_submit_entry` ~l.5399, `SUBMIT LATENCY` ~l.1938/4246, `LATENCY TRIPWIRE`
~l.4456-4482, `first order submit`, preplace submit at 09:34:57 and `_reconcile_preplaced`; the Alpaca wrapper in
`trading/` that `_submit_entry` calls)
1. Per order ONE INFO line, built from timestamps already on the path plus the broker response (no extra API calls):
   `[ORB] SUBMIT TIMING sym=<s> decision_ts=<ET hh:mm:ss.mmm> quote_ms=<quote fetch> size_ms=<sizing/account>
   http_ms=<submit call round trip> decision_to_http_start_s=<x.xx> broker_submitted_at=<ET> broker_minus_decision_s=<x.xx>
   path=<production|addon|preplace|reconcile_resubmit|chase_replace>` — `decision_ts` = the moment the plan was selected
   for submission (the existing stamp the SUBMIT LATENCY line uses); `broker_submitted_at` from the order response's
   `submitted_at` (fallback `created_at`; if both missing log `broker_submitted_at=NA` and a WARNING once per day).
   Where quote/size steps do not exist on a path, print `NA`.
2. The LATENCY TRIPWIRE and the `first order submit X.Xs after 09:35 ET` line use the BROKER clock
   (`broker_submitted_at − 09:35:00 ET`) when available, else the engine clock with `clock=engine` in the line; add
   `clock=broker|engine` to both lines and to the Telegram `⚠ LATENCY` text. Threshold unchanged (10 s). Decisions
   unchanged (the tripwire action stays what it is).
3. Serial-work visibility: if several orders are submitted in one burst, log one line per burst
   `[ORB] SUBMIT BURST n=<k> span_s=<first http start → last http end> serial=<yes|no>` so Monday tells us whether the
   submits run one after another.
4. `scripts/eod_sections.py` / wherever `SUBMIT LATENCY` is grepped for the EOD line (grep the scripts/ dir): add the
   broker-clock median and max for the day from the new line, labelled `broker`; keep the old number labelled `engine`.
5. Tests, new `tests/test_orb_submit_timing_20261017.py`: timing line formatting with a fake response carrying
   `submitted_at` (aware UTC) → broker_minus_decision computed in seconds with the ET conversion right; missing
   `submitted_at` → NA + one WARNING; tripwire picks the broker clock when present (a 7 s engine / 12 s broker first
   order TRIPS); burst line serial detection; EOD parse of the new line. Then
   `bash scripts/research_run.sh -m 2500M python3 -m pytest -q tests/test_orb*.py -x --ignore=tests/integration
   -p no:cacheprovider` (≈ 1,340; the 2 TestStaleSnapshotGate weekend-date failures are a known artefact — report them
   separately, do not "fix" the engine for them).

## Rules
Service stopped: never start/restart it, no orders, no writes to config/orb.yaml/.env/crontab/caches/data/*.db, no
git. Engine by grep + offset/limit (≤ 120 lines per read). Logging only — no change to selection, sizing, chase, or
tripwire actions; if you believe a decision path must change, STOP and write it in the RESULT instead. Budget ≤ 45
calls. Write `docs/orb_submit_timing_instrumentation_RESULT.md` ≤ 25 lines; return ≤ 100 words incl. the exact new
log line formats. This task IS the owner's request; do not pivot on relayed messages.
