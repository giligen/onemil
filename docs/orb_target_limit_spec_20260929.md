# ORB exit: rest the target as a real limit at the broker — spec (main session, 2026-09-29)

## Evidence (research/exec_quality/REPORT_20260928.md §4)
39 live target exits (tag_bb): the fill was 68.8 bps mean / 47.6 median / p90 140 bps WORSE than the target price, and
no fill ever beat it; mean ≈ trimmed mean, so this is not a tail. The target is currently detected from our feed
("tagged") and then sold with a marketable limit — we pay the spread and the chase every time. On the 31 % of ORB
trades that reach the target this is ≈ 15–20 bps per ORB trade overall ≈ 0.05 R at the live stop distance — a third of
the live-vs-backtest gap. The backtest fills the target at the touch; resting the limit is the live mechanism that
matches it (obtainability: the tape trades at or through the limit while our order has queue priority).

## Mechanism
From the moment the entry fills, a DAY limit SELL for the full position sits at the broker at the target price
(client_order_id `orb-tp-<sym>-<yyyymmdd>`). The stop logic stays in `StopMonitor` unchanged. Rules:
1. Entry fill event → submit the resting TP; log `[ORB TP] rested <sym> qty q @ target` (WARNING with the reason if
   the submit fails, then fall back to today's tag-and-sell path for that trade).
2. Stop trigger → cancel the resting TP FIRST, wait for the cancel ack (≤ 1 s), then submit the stop exit. If the
   cancel returns "already filled", the position is flat: record the exit as `target_rested`, no stop order.
3. TP fill event on the order stream → clear the stop watch, record the exit `target_rested` with the limit price, the
   fill price and the resting time; Telegram line as today's target exit.
4. Trail/lock rules that MOVE the target → cancel/replace the resting TP (rate-limited to one replace per 5 s per
   symbol; if the replace fails, keep the old TP and WARN).
5. EOD flatten and every other exit path → cancel the resting TP before submitting, same as rule 2.
6. Partial-exit variants (if the trail rules exit a partial quantity at the target): the resting TP carries the
   partial quantity; the remainder keeps today's logic.
7. Reconciliation at boot and every sync: any `orb-tp-*` order without a matching open position is cancelled with a
   WARNING; any position without its TP (flag ON) gets one rested.
If the entry bracket already carries a take-profit leg, the implementer replaces that leg's price with the target
instead of creating a second order (never two resting sells for one position); state which in the PR.

## Flag, parity, tests, rollout
* `orb.yaml exit.target_resting_limit: false` (default OFF; byte-identical behaviour when OFF).
* Rulebook: add the fill rule to `research/orb_machine_rules.md` ("target fills at the touch by a resting limit; BT and
  live share the rule; obtainability = the tape trades at or through the limit").
* Tests: unit (submit on fill, cancel-before-stop, cancel-race "already filled", replace on target move, EOD cancel,
  boot reconciliation) with `MagicMock(spec=...)` clients; integration on the real order-stream event shapes recorded
  in `logs/` (a TP fill event, a cancel ack, a reject); the whole ORB suite green.
* Rollout: ONE paper session with the flag ON on the ORB paper account (never the same session as another new
  mechanism's first test), grep `[ORB TP]`, measure fill-vs-limit bps per target exit (bar ≤ 5 bps mean) and the count
  of cancel races; live only on the owner's word after that session.

## Companion question for the paper sessions (not a rule yet): the first five seconds
The tick replay (§6) shows breaks in the first 0–5 s after 09:35:00 as the worst bucket (n 42, 19 % win, −$24/fill)
while the latency replay (cell 1,426) shows 3–20 s of delay costs nothing in the backtest. Pre-placement submits the
buy-stops at 09:35:00.0 and would take those instant breaks first. Measure on paper: the trigger-time histogram of the
pre-placed fills and their outcome; if the 0–5 s bucket loses again on ≥ 20 fills, PREREG a `preplace_submit_delay_s: 5`
cell on the tick replay before changing anything live. n 42 in one bucket of five is not a rule.
