# Ignition Pre-Staged Entry ("heap") — Design + Pre-Mortem
**Status: DESIGN (2026-08-22). Nothing here is implemented. Gate: Sunday rehearsal → Monday GO decision (owner).**
**Economics gate: research/ignition_capcheck/RESTING_MODEL.md (running) must pass the 4-leg ship bar.**

## 0. Why (one paragraph)
Live ignition chases the ask 30-60s after the level cross and pays ~180bps median
(realistic-quote 25-day book: −$13.2K vs BT +$5.0K at book sizing). The late-entry
retrace CAP is REFUTED (19-month harness: capped misses are the book's best trades
— gap-and-go runners, WR 67%). The remaining fix changes WHEN the order exists, not
its price ceiling: a stop-limit RESTING at the level before the cross fills on the
way up. Detection speed alone is worthless (book entry vs level: median +3bps —
measured); placement is everything.

## 1. Mechanism
- Candidate feed: existing scanner sightings (flat-open movers) → proximity list.
- **Heap**: rank candidates by distance-to-level ((level−last)/level), maintained
  from the existing websocket quote/trade stream. Top-K get staged orders.
- **Staged order**: stop-limit BUY, stop = level×1.003, limit = level×(1+CAP_BPS),
  qty = floor($RISK / (entry_est − stop_est)) at $50 risk, TIF=DAY,
  client_order_id = `ign-stage-{YYYYMMDD}-{sym}` (idempotency + attribution).
- **Windows**: stage no earlier than 9:35 ET (book trigger window opens 9:35; a
  pre-9:35 cross must NOT fill → never stage before 9:35). Cancel-all at 13:00 ET
  (book TRIGGER_MIN_END). EOD safety: DAY orders die at close regardless.
- **Fallback (take-all invariant)**: the current sight→chase path stays ON for any
  shadow trigger with no staged fill (heap miss, gap-through, cancel race). Live
  trade set must equal shadow trigger set — staging changes PRICE, never COVERAGE.
- Fill → same path as today: StopMonitor watch (1.75R/0.5R), DB row, telemetry.

Parameters (from studies, frozen before ship): K (heap depth), CAP_BPS,
promote/demote thresholds, churn budget. All in config under `ignition_live.prestage`
with `enabled: false` default and env kill `IGNITION_PRESTAGE=0`.

## 2. PRE-MORTEM — where this breaks in production (each item: failure → design answer → test)
Ordered by (probability × damage). Items marked ★ have already happened to us in
another guise — they are not hypothetical.

### A. Order-state lifecycle (our #1 real incident class)
1. ★ **Restart with staged orders live** (BMNR/DFNS class). Process dies; resting
   stop-limits stay at the broker, unmanaged; one fills while we're down →
   position with no stop watch. → Boot reconciliation: list open orders by
   `ign-stage-` prefix + positions; adopt fills (register watch from DB/state),
   cancel every stale stage. Runs in sync_positions before anything trades.
   TEST: kill −9 mid-stage in rehearsal; boot; assert adopted/cancelled, zero
   unmanaged.
2. ★ **Shutdown leaves stages** (DFNS shutdown-race class). → finalize step:
   cancel-sweep all `ign-stage-*` in the scanner shutdown sequence (before
   finalize_eod); DAY TIF as belt-and-braces. TEST: shutdown with 3 staged; assert
   zero open ign-stage at broker after exit.
3. **Cancel/fill race**: heap demotes a name; cancel arrives after the fill.
   Cancel-reject ≠ cancelled — it usually means FILLED. → On cancel error, poll
   the order; if filled, adopt (watch + DB), never drop tracking on an assumed
   cancel. TEST: mocked cancel→"order filled" path adopts position.
4. **Partial fills** (thin tape): stop-limit fills 60/168 shares, rest resting. →
   watch/DB use filled_qty; cancel remainder immediately; risk recomputed on
   actual qty. TEST: partial-fill mock asserts remainder cancel + correct watch.
5. **Duplicate entry**: state desync re-stages a filled symbol → double position.
   → Idempotency: one stage per (symbol, day) EVER (client_order_id uniqueness is
   broker-enforced: same id rejects); `_entered_today` check before every place.
   TEST: forced double-stage attempt rejected locally AND by id collision.

### B. Fill semantics vs the model
6. **Gap-through-band**: crossing bar opens above limit → no marketable fill;
   order rests at limit → may fill LATER on a fade = adverse fill at the worst
   price. → Policy from the resting-model study: if gap-through, CANCEL the
   resting order after N minutes (do not accept stale limit fills) OR route to
   chase-path evaluation; N from data. TEST: replay gap-through days; assert no
   stale fills beyond policy.
7. **Trigger-on-odd-print**: SIP oddities (late/odd-lot prints at/above stop) can
   trigger the stop without a real cross. → Accept (rare, micro size), but flag:
   any staged fill with no shadow trigger within M minutes → telemetry counter +
   EOD-dive line item; reconcile daily. TEST: synthetic fill-without-trigger
   raises the flag.
8. **Parity redefinition**: staged fills at ~level ≠ book entries at
   next-bar-open. → The BOOK stays untouched; live records fill vs level AND vs
   book-entry (two fill-quality lines); the S3 verdict metric becomes
   fill-vs-level for staged fills. The take-all set-equality check (live trades
   ≡ shadow triggers, modulo explicit reasons) becomes a nightly HARD gate.
   TEST: parity checker on synthetic day with staged + chased + missed names.

### C. Buying power & capital (owner-flagged)
9. **BP reservation**: every open stop-limit BUY reserves buying power. K staged
   + fills accumulating on a cluster day can exhaust the envelope → later
   stage/entry REJECTED by broker (coverage hole → falls to chase path, cost not
   loss — but must be visible). → **Watermarks**: staging budget =
   min(PRESTAGE_BP_FRAC × DTBP, PRESTAGE_BP_ABS). Stage only while
   reserved+filled < budget; telemetry records the daily high-watermark; alert at
   80%; hard stop at 100% (chase path continues). TEST: cluster-day replay (8/3)
   through the budgeter; assert never exceeds envelope, fallback engaged, alert
   fired.
10. ★ **Kill rails vs pre-committed entries** (kills-only-block-new-entries
    class): daily −$300 fires while K orders are STAGED — those are entries
    already armed. → Kill firing does an immediate stage-sweep (cancel all) in
    the same code path that raises the kill; weekly kill likewise. TEST: kill
    trigger with 5 staged asserts zero staged after, before any new tick.
11. **Equity dip below $25K PDT line** (main account $73K today, not close — but
    check is cheap): PDT guard already exists for ORB; extend its account read to
    gate staging count if equity < threshold. TEST: mocked sub-25K equity halves
    staging.

### D. Data/feed failures
12. ★ **Quote stream stalls** (feed-gap class): heap goes stale; broker-side
    orders still live (good — entries survive feed death), but demote-cancels
    stop → stale stages can fill on faded names. → Feed watchdog: no heap update
    for STALE_S (60s) → cancel-all stages + alert + chase-only mode until feed
    recovers. TEST: feed-stall sim asserts sweep + mode switch + recovery
    restage.
13. **Sighting feed lag** (candidates never enter heap): chase path covers, at
    the old cost. → Telemetry: fills-by-path (staged vs chase) daily ratio; a
    falling staged-ratio is the smoke alarm. TEST: synthetic late-sighted name
    still trades via chase.
14. **Halts (LULD)**: staged order through a halt → reopen gap chaos. → Halt
    detect (quote condition/no prints) → cancel stage on halted names; re-stage
    on clean reopen only if still in window and below cap. TEST: historical halt
    day replay.

### E. Rate limits & churn (owner-flagged: "placing and cancelling all the time")
15. **200 req/min API budget**: volatile mornings + naive re-ranking = place/
    cancel storms. → Dual-threshold hysteresis (promote at rank ≤ K−2 AND
    distance < D_in for 2 consecutive updates; demote only at rank > K+3 OR
    distance > D_out), plus a hard churn limiter (max OPS_PER_MIN order ops;
    excess deferred — staging deferred is safe, the chase path covers). Alpaca
    charges $0 per order/cancel; the budget is rate + broker optics, not fees.
    TEST: 8/3 hot-tape replay through the heap sim; assert ops/min under budget
    with zero coverage loss (deferred names that triggered were caught by
    fallback).
16. ★ **API errors in the hot path** (news-timeout class): place/cancel timeouts
    must not wedge the worker. → Bounded retries (1 retry, 2s), then defer and
    flag; never block the tick loop. TEST: injected 500s/timeouts keep the loop
    alive.

### F. Mechanics details that bite
17. **Sub-$1 tick rules**: limit prices need penny (≥$1) / 4-decimal (<$1)
    rounding or the order REJECTS. → Central rounding helper used for stop and
    limit; property test across price grid. (Universe is low-priced — this WILL
    fire day one without the helper.)
18. **client_order_id limits**: length/charset caps. → Fixed short scheme,
    validated in tests.
19. **Level recomputation drift**: level = day_open×1.10 — day_open must be the
    OFFICIAL 9:30 open (first regular-session bar), same source as the shadow,
    or staged stop ≠ book level. → Single helper shared with ignition_rules;
    parity test on 20 random historical days.
20. **Clock edges**: stage-at-9:35:00 vs bar timestamps; 13:00 sweep exactness;
    DST (the house scar tissue): all times via the ET helpers, never UTC math.
    TEST: window-edge cases + a synthetic EST-season day.

### G. Observability (how we catch what we didn't predict)
- Per-stage lifecycle event log (staged/promoted/demoted/cancelled/filled/
  adopted/swept + reason) → `logs/session_archive` daily file.
- Telemetry counters surfaced in the nightly EOD dive: stage ops, fills by path,
  rank-at-trigger for every trigger (measured vs K-study prediction), BP
  high-watermark, feed-stale events, gap-through count, adverse-fill count,
  churn-limiter activations, fallback activations.
- Green check gains a PRESTAGE section: set-equality parity (hard), zero
  unmanaged fills (hard), watermark < 100% (hard), staged-ratio (soft trend).

## 3. Test plan (suite = every pre-mortem item above + the ladder)
1. Unit: heap ordering/hysteresis, rounding, id scheme, budgeter, window edges.
2. Integration (mocked alpaca): full lifecycle incl. every race in §A, kill
   sweep, watchdog, partial fills.
3. Replay: historical-day driver (bars → heap sim → staged/filled set) asserting
   staged-set ⊇ triggers per K-study, ops budget, and P&L within the resting
   model's tolerance on 5 sample days (incl. 8/3 hot, one halt day, one quiet).
4. Rehearsal (Sunday): boot with prestage enabled on PAPER keys or enabled with
   0-qty dry-stage mode; kill −9 mid-stage; shutdown sweep; feed-stall drill.
5. Shadow-stage mode (`prestage.shadow: true`): live decisions, zero orders,
   full telemetry — the default state until the Monday GO and the fallback state
   any time a hard gate reds.

## 3b. P0 RESOLUTIONS (v2, post-review — docs/ignition_prestage_review_aug2026.md)
- **P0-1 disposition rule**: at FILL time, run the SAME structure gates the book
  runs (R>=5% from actual pre-bars, participation, pre-bars count, chase guard)
  via a shared ignition_rules helper on bars up to the fill. PASS -> book-valid
  trade (watch + DB, parity reason 'staged_book'). FAIL -> immediate marketable
  exit + DB row with parity reason 'stage_reject_structure' (scratch cost is the
  price of pre-commitment; telemetry counts it; shadow week measures it before
  any live order exists).
- **P0-2 complex leg**: staged set = news-eligible names only at 9:35; the
  complex leg is handled REACTIVELY — when trigger #1 of an anchor fires
  (staged fill or chase), immediately stage all same-anchor siblings still below
  level (sibling-lag mini-study quantifies the capture window). Capture
  restated honestly: news leg staged, complex leg sibling-staged-late + chase
  fallback; set-equality gate unchanged.
- **P0-3 fallback race**: chase entry for a symbol allowed ONLY after broker-
  confirmed stage disposition (cancel acked AND filled_qty=0, or no stage ever
  placed). One state machine per (symbol, day): STAGED -> (FILLED | CANCEL_
  CONFIRMED | REJECTED) -> CHASE_ELIGIBLE. All four sweep paths route through it.
- **P0-4 fill transport**: stage fills consumed from the existing main-account
  OrderStreamWatcher (prefix-matched ign-stage-*), NOT per-order polling.
  Naked-fill window covered by a broker-side safety SL leg attached at adopt
  (ORB dead-man pattern) or submitted immediately on the stream fill event.
- **P0-5 stop/R at fill**: on adopt, recompute stop/R/lock_r_unit from actual
  fill + actual pre-bars via the shared helper (never staging-time estimates);
  skip_exits_until_ts keyed to the FILL timestamp minute.
- **P0-6 already-crossed routing**: names above level at staging time (or
  crossing during rollout before their stage is placed) NEVER get a buy-stop;
  they route to the standard chase path evaluation (which already handles
  above-level entries with its own guards). Stage placement checks last < level.
- **Frozen constants (post-studies, research/ignition_capcheck/SIBLING_OFFSET.md)**:
  stop offset **15bps** (30 was mid-downslope; 0 fails leg-2), cap **300bps**,
  gap-through cancel N=60min. P0-2 SEVERITY DOWNGRADED: trigger #1 carries 98%
  of CC P&L; reactive sibling-staging ships as a ~5% bonus; fast-twins to chase
  cost ~nothing (net-negative cohort). Consequence: trigger-#1 arming quality is
  the economics — which is the news-keyed Tier-1 staging track (NEWS_LEAD study
  pending; Tier-1 builds as a SEPARATE phase after prestage core, shadow-tested
  like everything else).
- P1 adoptions: DB rows created at FILL only (P1-4); gap-through cancel frozen
  at N=60min (P1-5); adoption-side max_concurrent + kill-race semantics written
  (P1-6); FILL QUALITY line gains path=staged|chase (P1-7); stop-offset sweep
  {0,15,30,50}bps pre-ship (P1-8); StopMonitor same-symbol cross-strategy
  collision fix (P1-2) and force_exit whitelist extension (P1-3) included in
  the build; BP-watermark monster-day pruning documented as the ramp wall
  (P1-1) — take-all staging never carries book sizing, stated plainly.

## 4. Rollout
- Mon GO gate (owner): RESTING_MODEL ship bar passed + suite green + rehearsal
  clean → live at $50 risk, K from study, watermarks armed, shadow-stage
  telemetry running in parallel. Any wobble → shadow-stage-only week.
- Abort: `IGNITION_PRESTAGE=0` or config flip + restart → chase path only
  (today's behavior). Kill rails unchanged and now stage-aware (§C10).
- Scale: prestage adds NO new sizing — $50 risk unchanged; sizing decisions stay
  with the existing ramp gates.
