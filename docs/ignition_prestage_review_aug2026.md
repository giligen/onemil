# Ignition Pre-Staged Entry — ADVERSARIAL DESIGN REVIEW (2026-08-22)

**Reviewer role: hostile production engineer + hostile market. Read-only session.**
Inputs: `docs/ignition_prestage_design_aug2026.md`,
`research/ignition_capcheck/{RESTING_MODEL,REPORT}.md`,
`trading/{ignition_engine,ignition_rules,ignition_shadow,stop_monitor}.py`,
`research/ignition_fill_optimization_evidence.md`,
`research/ignition_week_aug17_review.md`, `persistence/database.py`,
`scripts/{report_common,ignition_shadow_report,daily_green_check,weekly_report}.py`,
`main.py` wiring.

## VERDICT: **CONDITIONAL-GO — architecture sound, Monday LIVE staging is NOT.**
**6 P0s (must fix before ANY live staged order), 8 P1s (before scale).**
The resting-order economics are real and the studies are honest. But the design
as written contains one internal contradiction (take-all staging fills the
−$567K ungated river; catalyst-subset staging can't know the complex leg at
9:35 — and the complex leg is the cohort the economics were validated on) and
four unbuilt mechanisms whose absence is precisely our incident history
(BMNR/DFNS/SMU-QBTZ class). Recommended path: shadow-stage Mon–Tue on real
tape, live at $50 from Wed ONLY if P0s are closed with tests + drills green.

---

# P0 — MUST FIX BEFORE ANY LIVE STAGED ORDER

## P0-1. No disposition rule for staged fills that are NOT book trades
The K-study's own conclusion is "stage ~the full prefiltered universe"
(median 109 / p90 310). A staged stop-limit fills on EVERY universe crosser:
the 19-month river is 9,499 crossers (~24/trading-day) at **−$567K**, while
the book keeps only catalyst-confirmed structure-passers (~2.7/day CC leg).
Structure gates are unknowable at 9:35 by construction — `skip_r_too_small`
(pre-30min low of a trigger bar that hasn't happened), `skip_illiquid`
(participation = 15% × next-bar dollar volume), `skip_chase_guard`,
`skip_pre_bars` — so a staged order CANNOT pre-encode them; and catalyst may
never confirm. §B7 models fill-without-trigger as a rare odd-print; under
take-all it is the DOMINANT staged-fill class (~10× the kept-trade rate).
Even under the catalyst-subset refinement, news-having names that cross and
then structure-fail still fill (week of 8/17: illiquid 15, r-class skips vs
9 triggers over 334 sightings).
**Required**: an explicit at-fill validation pass — on adopt, run
`trigger_entry_stop` + catalyst state; if the name is not a kept trigger,
exit immediately by a defined rule (and model that exit's cost: entry
~level+30bps, exit next-minute ≈ small but nonzero); every such fill gets an
explicit parity reason so the nightly set-equality HARD gate doesn't red
daily. Without this rule the gate as specced fails on day one, or worse —
nobody notices we are trading the river.

## P0-2. Complex-confirmation catalyst is UNKNOWABLE at 9:35 — the refinement inverts its own evidence
`_day_anchor_counts` (ignition_shadow.py:469-477) counts **structure-PASSED
triggers**, which only exist after 9:35–10:30 level crosses. At 9:35 staging
time every anchor cohort is 0. "Stage the catalyst-eligible subset" therefore
= stage the NEWS leg only. The complex leg — historically 27/83 ≈ 33% of
triggers, including the 8/3 cluster (CRWU +3.23R, CWVX +$5.8K, AAOX/CRWG/NB*)
that carries the ENTIRE positive live ledger — is late-eligible by
construction and falls to the chase path at ~180bps median.
The killer: the RESTING_MODEL's validated surface (CC book, +$77.1K delta,
0 monster misses) **is the complex-confirmed cohort** — exactly the cohort
this refinement cannot stage. The news-leg staging delta was never separately
measured (news leg not reconstructable historically); its only support is the
slip-fragile river surface.
**Required**: pick one, with evidence: (a) reactive sibling staging — the
moment the FIRST anchor member passes structure (cohort=1), stage every
anchor-sharing universe name (its own cross completes cohort≥2); needs a
sibling-lag distribution mini-study from the shadow journals before ship;
or (b) accept chase for the complex leg and restate expected capture
(staged ≈ 2/3 of triggers) in the doc — no silent optimism.
Note also breaking-news names (the `_resolve_news` same-morning window):
news status itself is not frozen at 9:35 either.

## P0-3. Chase-fallback double-entry race — the take-all invariant is not safe as specced
Fallback rule: chase "any shadow trigger with no staged fill". A staged fill
can exist at the broker while being locally unconfirmed (fill-event lag,
cancel in flight, sweep in progress). Local state is not proof. All four
sweep paths widen the window: boot reconciliation (§A1), kill sweep (§C10),
feed-watchdog cancel-all (§D12), demote/13:00 cancels — each is 1–3 minutes
of cancels at rate-limit on a p90 day, during which legitimate crosses fill.
The shadow-trigger callback (worker thread) and fill confirmation (stream/
poll thread) race; `_entered_today` dedup only helps if adoption wins.
**Required**: the chase submit for any symbol with a staged order this day
must first prove at the BROKER that the stage is terminal-and-unfilled
(cancel confirmed AND filled_qty=0); on cancel-reject, poll-and-adopt (§A3)
— and this composed rule needs its own integration test, because the doc's
items 1, 3, 10, 12 each assume the others didn't just fire.

## P0-4. Fill-event transport unstated — poll-per-order dies at K=310, and the fill→watch gap is naked
`process_tick` polls `get_order` per pending order. At 110–310 staged orders
that is 100+ trading-API requests per tick — the 200/min budget (SHARED with
bull flag on the main account) dies instantly, taking bull-flag order ops
with it. The design doc never says how staged fills are detected.
**Required**: fills via the existing main-account `OrderStreamWatcher`
(trade_updates), keyed by the `ign-stage-` client_order_id prefix; REST poll
only as a degraded slow-path (batchable: `get_orders` list call, not
per-order). AND: between broker fill and watch registration the position is
naked — today's chase path closes that gap with bracket legs (broker
dead-man SL). Staged orders must either be bracket stop-limits (Alpaca
supports stop_limit entry + bracket) or get a dead-man SL placed at adopt
time. §A1 covers restarts; this window is the ALIVE process's gap.

## P0-5. Stop/R basis at fill: current engine code contradicts the validated model
RESTING_MODEL re-derives everything at the fill: `stop_f = min(pre-30min
low, fill×0.99)`, `rp_f`, exits from fill-based R. A pre-staged order's qty
is frozen at 9:35 against a stop ESTIMATE for a trigger bar that doesn't
exist yet. Meanwhile `_confirm_fill` passes `lock_r_unit = p.entry − p.stop`
(staging-time estimates) and keys `skip_exits_until_ts` to CONFIRM wall-clock
(`time.time()`), not broker fill time — a late confirmation skips extra
minutes of exit checks on exactly the fast-reversal names, and lock/arm math
runs on stale plan values. Exit physics are where the modeled money lives
(the −$31K…−$366K fader-avoidance leg and the 1.75R/0.5R lock).
**Required**: at adopt, fetch bars, recompute stop_f/R per `ignition_rules`
(shared helper — parity by construction, same as every prior fix in this
codebase), register the watch with fill-based stop/R, key the skip window to
`od['filled_at']`. Accept that qty (staged) ≠ ideal qty (fill-derived);
telemetry the realized-risk-vs-$50 ratio — it is a live measurement the
model assumed away.

## P0-6. Already-crossed names cannot be staged as stop-limits — the cohort that decided cap300 gets forfeited
A buy stop-limit with stop ≤ current price REJECTS at the broker. This hits:
(a) the preopen cohort — above level at 9:35, 451 trades/19mo, **+$39.5K at
cap300 and the stated reason cap300 beats cap100**; (b) every name that
crosses during the multi-minute placement rollout. The design has no routing
for either. Day-one failure: a burst of broker rejects at 9:35:01 and the
richest cohort silently falling to the chase path.
**Required**: crossed-at-staging (or crossed-before-placement) names route
to a plain LIMIT at cap_px (the model's gap-into-band fill is the opening
print — a marketable limit reproduces it) or explicitly to chase, chosen and
documented BEFORE build; plus the sub-$1 rounding helper (§F17) on this path
too.

---

# P1 — FIX BEFORE SCALE (live staging may start at $50 with these open, tracked)

## P1-1. BP watermark vs take-all: pruning lands exactly on monster days, and the ramp story is unsolved
At $50 risk, staged notional ≈ $600/name (R~8%). p90 day: 310 × $600 ≈
$190K reserved on a $73K account (DTBP ≈ $292K) shared with bull flag
(3 × $25K positions). The watermark will prune on p90 cluster days — and
cluster days ARE the monster days (8/3). Pruning by proximity rank on those
days reintroduces the small-K monster-miss the K-study refuted. No priority
rule exists between bull-flag BP needs and staging. And the wall: at book
risk ($3K) take-all staging ≈ $7.7M reserved — **the prestage architecture
cannot carry book sizing on this account, ever**. The doc's ramp section
("sizing decisions stay with the existing ramp gates") hides this; state it:
scale requires either K-selection that beats the K-study or a different
capital structure. Also decide watermark ordering: reserve nearest-to-level
first (K-study rank logic), and report BP-pruned names in the parity ledger.

## P1-2. StopMonitor watch-key collision (pre-existing, amplified 100×)
`self._watches[symbol] = entry` — keyed by symbol ONLY. A bull-flag position
and an ignition position on the same symbol silently overwrite; the earlier
strategy's position becomes unmanaged and its exit event never drains
(`drain_exit_events` filters by the surviving entry's strategy). Staging
puts 100–310 momentum symbols/day on the shared account next to bull flag's
mover universe; weeks-scale collision probability is material. Minimum: an
ERROR + telegram on strategy-mismatch overwrite at `add_watch`; correct fix:
per-strategy keying or reject + alert.

## P1-3. force_exit whitelist blocks every staged-disposition exit
`_FORCE_EXIT_REASON_WHITELIST = {tag_bb, tag_b1}` — the P0-1 disposition
exits (e.g. `stage_no_trigger`, `stage_no_catalyst`) and any watchdog-driven
force-flat will be **silently ignored** (logged, no exit). Extend the
whitelist + tests before wiring any disposition path.

## P1-4. DB row lifecycle for stages poisons restart dedup
If stages create `pending_new` rows at placement (current `_submit`
pattern), an intraday restart's `sync_positions` resumes them AND adds every
staged symbol to `_entered_today`; after the boot sweep cancels the stages,
those symbols are dedup-blocked from the chase path for the rest of the day
— a coverage hole that breaks the take-all invariant precisely after the
failure the invariant exists for. Create trade rows at FILL only (stage
lifecycle lives in the §G event log), or teach sync to distinguish
stage-pending from entry-pending.

## P1-5. Gap-through cancel policy (§B6) smuggles in an unmodeled rule — the retrace-cap mistake again
The study measured late-limit ("adverse") fills as net POSITIVE at cap≤300
(+$14.0K CC / +$15.4K river) and found the 60-min auto-cancel inert (<$3K).
§B6's "cancel after N minutes, N from data / do not accept stale limit
fills" invents a selection rule the harness never ran — structurally the
same error class as the refuted retrace cap (a rule whose miss condition
correlates with trade quality). Freeze N=60 (ORB convention, validated
inert). Any tighter N needs its own harness pass first.

## P1-6. Parallel-fill concurrency and kill semantics
Staged entries fire in parallel — no detection serialization. A cluster
morning can fill 10–20 names in minutes; realized-P&L kills (−$300/day) lag
open losses, and `max_concurrent=15` is enforced only in `_handle_trigger`
(chase path) — adoption has no gate. Add: max staged-fill positions → on
breach, sweep remaining stages; and write down the kill-race semantics: a
fill that raced the kill sweep is adopted, managed, and counted (never
dropped, never retro-vetoed).

## P1-7. Fill-vs-level parity redefinition — consumer audit (target 3)
Checked every reader of ignition trades rows:
- `IgnitionEngine._realized_pnl` (kills): reads `pnl` only — basis-agnostic. OK.
- `weekly_report.py` per-strategy P&L: `pnl` only. OK.
- Orphan reconciler / `get_open_trades`: status fields only. OK.
- `ignition_shadow_report.py`: resims from JOURNAL hypo entries, not trades
  rows — unaffected. Its BT-agree line is also unaffected (shadow unchanged).
- `daily_green_check.py`: has NO ignition section today — the §G PRESTAGE
  section is new build, not a retrofit. OK but must be built.
- **BREAKS**: the `FILL QUALITY` log line computes `chase = fill vs p.entry`
  (plan = BT next-bar-open entry). For staged fills this goes near-zero/
  negative and means something different; any EOD/weekly grep that pools it
  (the line exists precisely to be grepped) corrupts the S3 fill ledger and
  the ~9/15 ramp evidence. Required: `path=staged|chase` tag in the log line
  AND in pattern_data; all fill-quality aggregation splits by path; the S3
  verdict metric for staged fills = fill-vs-level (per §B8), chase path keeps
  the old metric. The nightly set-equality checker is a NEW consumer and must
  understand P0-1's explicit reasons or it reds daily.

## P1-8. Slip and stop-offset sensitivity (target 4) — the ship bar dies at ~30bps real slip
Interpolating the CC cap300 grid (+$77.1K s0, +$58.2K s10, +$27.0K s25 ≈
−$2.0K/bps): **at 40bps the CC delta ≈ $0 and eras go ~1/3 — ship-bar leg 3
fails**; at 30bps ≈ +$17K with 2026-era ≈ −$10K, marginal. The bar holds
only for slip ≤ ~25bps. Context that keeps this survivable: the baseline is
the BT entry, not live chase (−$13.2K/25d realistic) — vs LIVE the staged
book wins even at 40bps. But the ship claim was made against the bar, so:
week-1 live must measure the slip-at-stop distribution per fill (clean fills
only, not gap-into-band) with an explicit pre-declared gate — median >30bps
sustained = staging OFF pending re-study. Separately, the 30bps stop OFFSET
is unswept and load-bearing (offset→0 refills the −$31K CC fader cohort;
offset too high creates its own miss channel) — no constant changes without
the 0/15/50 sweep the study itself demands.

---

# OBSERVATIONS (no number, still real)

- **Heap contradiction**: the K-study refuted small-K, so under take-all
  there IS no promote/demote churn — §E15's hysteresis machinery is mostly
  vestigial. The doc should commit: take-all + placement SCHEDULER (ordering
  = proximity, budget = OPS_PER_MIN) and delete the heap-depth framing, or
  defend a K. Carrying both invites building the refuted version.
- **Rollout timing (target 1b, worked)**: at an honest self-cap of ~100
  order-ops/min (leaving headroom for bull flag + cancels + retries), p90
  310 orders ≈ 3.1 min to full coverage; median 109 ≈ 1.1 min. Proximity
  ordering genuinely front-loads: monster rank p90 = 62 (T-5) → first ~62
  placements (~40–60s) cover ~90% of monsters whose trigger is ≥5min out;
  for sub-9:40 triggers (21% of all trades) T-5 rank does not exist — that
  cohort is carried by P0-6 routing + chase fallback, not by the scheduler.
  Shadow-stage must report coverage-at-9:36/9:38/9:40 before live.
- **Clock skew (§F20 gap)**: an order live at the exchange at 9:34:58 can be
  elected by a pre-window print the book excludes. Boot-time NTP offset
  assert (<1s) + first placement at 9:35:02.
- **13:00 sweep**: 310 cancels ≈ 2–3 min at budget. Cancel nearest-to-level
  FIRST (shrinks the fill-during-sweep window); adopt-on-cancel-reject
  applies here too.
- **Dead process ≠ dead orders**: DAY TIF does not stop a 14:30 fill while
  the trader is down. `trader_watchdog.py` (or a cron) should cancel all
  `ign-stage-*` at the broker whenever the service is down >X min.
- **Two managers after watchdog restart**: overlapping old/new processes
  both holding watches on the same position → double exit → net short.
  Client_order_id idempotency covers placements, nothing covers double exit.
  Singleton guard (pidfile/flock) or systemd KillMode audit.
- **9:30 open availability**: thin names with a late first print have no
  day_open at 9:35 → cannot compute level → not stageable; fine (chase
  covers), but log it as an explicit stage-skip reason so parity accounting
  closes.

# REHEARSAL DRILLS TO ADD (target 6 — beyond the doc's four)
1. Order-stream fill drill: staged (paper) order fills → event arrives keyed
   by `ign-stage-` id → adopt path, watch registered, dead-man present.
2. Rate-limit storm: place 310 paper orders at 9:35 sim; measure
   time-to-coverage, 429 handling, zero wedges (E16 under real load).
3. Cancel-reject-means-filled: forced race, assert poll-and-adopt (A3) from
   ALL four sweep contexts (boot, kill, watchdog, 13:00).
4. Chase-vs-staged duplicate: shadow trigger while stage fill in flight →
   assert single position (P0-3 broker-proof check).
5. Preopen/crossed-name routing: stop≤market submit → broker reject →
   limit-at-cap fallback fires (P0-6).
6. BP watermark with bull flag holding 3 positions: staging stops at budget,
   alert at 80%, bull-flag entry NOT starved.
7. Kill fires with 5 staged + 1 fill racing: sweep + adopt + counted.
8. Double-manager: two engine instances against paper — assert no double
   exit (or document the singleton guard that prevents instance two).
9. Dead-process afternoon: kill −9 at 12:00, watchdog cancels stages at
   broker within N min.
10. Sub-$1 live paper probe (one $0.xx name) + rounding property test.
11. Synthetic parity day: staged + chased + missed + staged-fill-no-trigger
    names → nightly checker green with explicit reasons (P0-1).
12. NTP/clock-skew assert in boot sequence.

# MONDAY GO — BOTH SIDES ARGUED (target 7)

**Against Monday live (steelman)**: (1) Every new order-lifecycle mechanism
this program has shipped bit in week one (BMNR restart, DFNS shutdown race,
orphaned exit row) — this build adds four new mechanisms at once (scheduler,
budgeter, adoption, sweeps) with zero live hours. (2) The three numbers that
decide the design — coverage-at-time, late-eligible/complex rate, BP
watermark hits on a real tape — are all measurable in shadow-stage mode at
zero order risk; Sunday rehearsal cannot produce a real 9:35 tape. (3) At
$50 risk, live adds no economics evidence (median week ≈ −$125 by design);
the only unique live output is slip-at-stop, which loses nothing by waiting
two days. (4) Four of the six P0s are exactly week-one incident shapes with
live-order blast radius (river fills, double entries, naked fills, forfeited
preopen cohort).
**For Monday live**: (1) The whole thesis is fill price — only live fills
measure real slip-at-stop, and P1-8 shows the ship bar hinges on it. (2) $50
risk bounds the worst day at kill = −$300; DFNS cost $58 and bought the
decisive datapoint. (3) The chase fallback means a staging failure degrades
toward today's behavior — IF P0-3 holds. (4) The retirement-evidence window
(~11/15) makes weeks scarce.
**Where the evidence lands**: the fallback-degrades-safely argument is
circular until P0-3/P0-4 exist — a staging bug's failure mode today is a
naked or duplicated position, which is strictly worse than the chase path.
And per the owner's own retirement-validation rule, we optimize for evidence
QUALITY: shadow-stage on Mon–Tue real tape produces better evidence than
live does, at zero incident risk. **Recommendation: shadow-stage Mon–Tue;
live at $50 Wednesday IF P0-1..6 are closed with unit+integration tests and
drills 1–12 green; any P0 open Wednesday = shadow-stage the full week. This
is the CONDITIONAL in the GO.**

---
*Filed by the adversarial reviewer, 2026-08-22. Read-only session; nothing
in the design or code was modified.*
