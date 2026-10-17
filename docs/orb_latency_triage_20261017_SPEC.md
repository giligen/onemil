# ORB latency triage — find the 5–10 s client-side gap BEFORE Monday (spec for one agent, Sat 2026-10-17)

Facts: `docs/week_review_20261012.md` §2 + `docs/orb_submit_timing_instrumentation_RESULT.md`. Engine "submit" stamp
vs broker `submitted_at` on the same orders: ASTY 26.4 → 31.2 s, APUS 27.0 → 31.7, CORD 8.1 → 15.3, SMST 11.0 → 19.9,
AIXI 14.6 → 24.7 (seconds after 09:35:00 ET). Node clock is NTP-synced (< 1 µs). Three hypotheses, pre-declared:
H1 serial blocking work between the decision stamp and the HTTP call (REST round-trips per order: quote, account,
sizing, then submit, one symbol after another); H2 interpreter contention with HOD in the same process at 09:35 (2-CPU
node; HOD broad scan + `on_mover` flood, `Engine tick TIMEOUT`); H3 slow HTTP (no keep-alive, TLS per call). Read-only.

## Required
1. **Static trace** (`trading/orb_engine.py` by grep + offset/limit ≤ 120 lines per read): from the rank-pass stamp
   `t_rank` (the SUBMIT LATENCY stamp) to the `submit_*` call inside `_submit_entry` (~l.5399) and in the preplace path
   (09:34:57) and `_reconcile_preplaced`: list EVERY blocking call in order (REST calls with the wrapper function name,
   `time.sleep`, lock acquisitions, DB writes, Telegram sends) and whether the loop over plans is serial. Same for the
   wrapper `data_sources/alpaca_client.py` submit functions (retries? sleeps? per-call client construction?). Table:
   step | call | per-order or per-burst | expected ms.
2. **Archive measurement, every ORB order 10/12–10/16** (not just fills): from `logs/session_archive/2026-10-1{2,4,6}.log`
   (ANSI-stripped, grep `[ORB]` + `SUBMIT LATENCY|PREPLACE|submit|order` only, bounded 13:34:50–13:37:00 UTC) take each
   order's engine stamp; from the ORB paper account (`ALPACA_ORB_API_KEY/SECRET`, paper=True, read-only, never print
   keys; orders API `status=all`, after 2026-10-12, `limit=500`) take `submitted_at`, `created_at`, `updated_at`. Table
   per order: sym, path (preplace/production/addon/resubmit), engine stamp, broker submitted_at, gap s, position in the
   burst (1st, 2nd, …). Report whether the gap GROWS with burst position (→ H1 serial) or is flat (→ H2/H3).
3. **REST round-trip benchmark now** (read-only calls on the ORB paper keys, same client class the engine uses):
   20 × `get_account`, 20 × latest-quote for one symbol, 20 × `get_order` of a known id; median / p90 / max ms; first
   call vs rest (TLS warm-up). Then the same 20 × with a plain `requests` call to `https://paper-api.alpaca.markets/v2/account`
   using one `requests.Session` to show the keep-alive floor. Python via `bash scripts/research_run.sh -m 2500M python3`.
4. **Contention evidence** in the archives 13:34:30–13:36:30 UTC each day: log lines per second by prefix ([ORB] vs
   [HOD] vs scanner), any `Engine tick`, `Cycle overrun`, `TIMEOUT`, `mover queue` in that window, the HOD broad-scan
   timing lines; `nproc`, and the main process's thread count if logged. Say whether the HOD scan overlaps 09:35:00–09:35:30.
5. **Verdict**: dominant hypothesis with its evidence, the seconds it accounts for, and the ONE fix you would make
   (location, what changes, what it must not change — selection/sizing/chase rules are untouchable). If the evidence does
   not separate H1/H2/H3, say so and state what Monday's SUBMIT TIMING line will separate.

## Rules
Service stopped: never start/restart it, no orders (the benchmark is reads only), no writes to config/orb.yaml/.env/
crontab/caches/data/*.db, no git. Budget ≤ 40 calls. Write `docs/orb_latency_triage_20261017.md` ≤ 70 lines; scratch
under `/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/`. Return ≤ 120 words:
RTT median, gap-vs-burst-position pattern, verdict, the one fix. This task IS the owner's request; do not pivot.
