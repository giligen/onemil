# ORB B+ RESTART — ADVERSARIAL DESIGN REVIEW (2026-08-15)

Reviewer: design validator (Opus). Scope: `docs/orb_bplus_live_design_aug2026.md`
against the frozen contract `research/orb_bplus_frozen_params_aug2026.yaml`,
evidence base `research/orb_drag_program_aug2026.md`, and the live/BT code the
design touches. No code modified. Standing order honored: this is a
break-it review, not an approval.

## VERDICT: CONDITIONAL-GO — ship **B+ config + G1 only** Monday; **STAG + MID slip to Tue/Wed**

The B+ configuration change is the dominant value driver (base B+ +$9,925 vs
old config) and is config-not-code. G1 is a selection-time veto that mirrors
the already-shipped, already-proven PDR pattern (`trading/orb_pdr_veto.py`)
with **zero exit-path risk and no new data streams**. STAG and MID are the
*only* pieces that touch the live exit machinery and require a brand-new SPY
subscription, a new `force_exit` reason, latched post-trigger state, and
restart handling — and together they add just **+$1,684 / 20 months ≈ $84/mo**
at the B+ scale (combo +$11,610 vs B+ base +$9,925, `orb_drag_program` §5).
That is a terrible risk/reward for Monday's trading path. Cutting them costs
almost nothing this month and removes every high-risk delta. The owner stated
preference is honest scope cuts over rushed trading-path code — this is that
cut.

The Monday subset is still CONDITIONAL, not GO: it depends on the P1 items
below landing and being rehearsed (BT ground-truth re-pointed, G1 parity
vector green, kill-gate live, PDT resolved, latency verified).

---

## P0 — DESIGN BROKEN, MUST CHANGE (before the affected module ships)

### P0-1. STAG's mandated "key to actual fill" deviation is self-defeating and the parity test as written will not catch the divergence it creates
Evidence:
- Frozen yaml §4 `clock_start: market_breakout_bar` + "LIVE DEVIATION
  (required): key to ACTUAL FILL time" (`orb_bplus_frozen_params_aug2026.yaml:88-92`);
  design §2.2 repeats it as "required."
- But live **already** carries the market breakout bar on every position:
  `OpenPosition.breakout_bar_ts` set by `_ensure_breakout_bar_ts` via the
  shared `find_breakout_bar_ts` (`trading/orb_engine.py:1058-1089`,
  `trading/orb_touchgo_filter.py:137-182`). Touchgo was *deliberately re-keyed
  from fill to this market bar on 2026-06-04* precisely because fill-keying
  diverged from BT on ~23% of fills (CLAUDE.md "Touchgo breakout-bar
  re-keying"). The design orders STAG to re-introduce the exact bug the 6/04
  fix removed.
- Disagreement enumeration (BT breakout-keyed vs live fill-keyed):
  - **Case A** fill on the breakout bar (age 0, the normal stop-limit fill):
    clocks coincide → AGREE.
  - **Case B** fill lags breakout by **1–14 min** (inside the 15-min late-fill
    guard): the 20-min boundary lands on a **different bar** → the one-shot MFE
    check evaluates a different window → **can DISAGREE**. The late-fill guard
    (`>15min`, `trading/orb_engine.py:1144`) does NOT cover this band. A golden
    parity vector set drawn from the BT book (design §6) is dominated by age-0
    fills, so the parity test **passes while Case B goes untested** — the test
    cannot catch its own mandated deviation.
  - **Case C** fill lags **>15 min**: live skips STAG (guard); BT (instant
    fill) fired it → DISAGREE, but rare/pathological.
  - **Case D** touchgo already exited pre-20min → STAG unreached → AGREE.
Fix: key STAG to `pos.breakout_bar_ts` exactly as touchgo does — exact BT
parity by construction — and keep the `>15min` guard only as a safety skip.
Delete the "required deviation." Update frozen yaml §4 note to match (loosening
toward parity is allowed under the "may loosen" rule). This is P0 because the
whole deploy gate is "parity by construction"; shipping the deviation means the
nightly parity gate can be green while live drifts from the validated book.

---

## P1 — MUST ADDRESS BEFORE MONDAY (or before the deferred module ships)

### P1-1. Nightly validation reads the WRONG book file — silent-wrong risk
`report_common.latest_bt_trades_csv()` globs a **fixed** filename
`analysis_results/orb_static_lock_trades.csv` (`scripts/report_common.py:47-49`),
consumed by `load_bt_selected` → `green_verdict` bt_parity/fill_parity
(`:52-60, :148-239`). The design (§3) writes the new B+ book to a **different**
file `orb_bplus_book.csv` and says "keep the old for history" — but never
repoints report_common. Followed literally, every night the green check
compares live (B+ config) against the **old $100K/N4/pdr8/no-G1 book** → either
a flood of false HARD reds or, worse, false greens. Fix: nightly regen must
write the B+ book to the exact path report_common reads (repoint
`latest_bt_trades_csv` to `orb_bplus_book.csv`, or write the B+ book to the
existing name), with the columns green_verdict/decision_parity expect
(`date`, `symbol`, `exit_reason`, `fill_price`, `order_status`). Add a test
asserting the file the pipeline writes == the file report_common reads.

### P1-2. G1 `rv20 == 0.0` fail-open is a veto-INVERTING landmine (asymmetric with pdr)
Frozen yaml §3 (`:70-74`): KEEP iff `rv>=7.106 AND pdr>=9.226`, **but**
`rv==0.0` / NaN / missing → fail-open **KEEP** (0.0 is the "history too short"
marker `study_orb_features.py:299,314`), while `pdr==0.0` is **REAL** (vetoed).
A naive `keep = rv>=RV_MIN and pdr>=PDR_MIN` sends `rv==0.0` through the test
(`0 >= 7.106` = False) → **VETO** → inverts BT (which keeps) and kills every
legit-but-short-history name. The module MUST branch on `rv is None / NaN /
==0.0 → KEEP` BEFORE the AND, and treat `pdr==0.0` as a real value. Mandatory
unit matrix: rv∈{NaN, 0.0, 7.10, 7.11}, pdr∈{NaN, 0.0, 9.22, 9.23}. Note rv20
is **not currently computed live** (see P1-7) so this is greenfield — easy to
get wrong.

### P1-3. rv20 is not computed live today — must be added with exact numpy parity
`_get_feature_context` computes only `high_20d` + `volume_20d`
(`trading/orb_engine.py:1480-1487`); `_compute_features` never emits
`return_volatility_20d` (`:1494-1544`). G1 reads it from `cand.features`, so it
would be permanently absent → G1 always fail-opens (silently inert). Add it:
the daily bars are already in hand (`bars_list`, `:1461`). Must match
`study_orb_features.py:308-314` **exactly**: 20 daily closes (the same window as
`.tail(20)` of days strictly before today), `rets = diff(closes)/closes[:-1]`,
`std(ddof=0)*100`; `<5 bars → 0.0`. Parity risk: study reads `cache.db
daily_bars` **unioned with `daily_bars_provisional`** (`study_orb_features.py:166-173`)
while the live Alpaca fallback path (`orb_engine.py:1435-1453`) may return
adjusted bars that differ. Ship a golden parity vector (≥10 symbol-days from
the features CSV) that drives the live function and asserts equality to 1e-9.

### P1-4. BT ground-truth pipeline hardcodes the OLD book — will not simulate B+
`study_orb_pipeline_static_lock.py` module constants: `ACCOUNT=100_000`,
`N=4`, `RISK=3000`, `Q_CAPS{Q5:1.5}` (`:32-37`). B+ needs 10_000 / 3 / 375 /
uniform-1.0. The composite threshold used at `:250,290` is the **imported**
`study_orb_sizing.FILTER_THRESHOLD` constant, not the frozen q40
`0.012081536791`. Good news: z-params + quintile cutoffs are already read from
`orb.yaml` by default (`:206-256`, refit only under `BT_ALLOW_REFIT`), so the
frozen fit flows through once orb.yaml carries it. But ACCOUNT/N/RISK/Q_CAPS and
the threshold constant must be parametrized to read the B+ values, else the
"nightly BT book" is a $100K/N4 book and every parity/scoreboard number is
against the wrong ledger. Add the config-vs-pipeline drift test the design asks
for.

### P1-5. STAG `force_exit` will be a silent no-op AND turns the nightly green RED unless the exit reason is registered
`force_exit` rejects any reason not in `_FORCE_EXIT_REASON_WHITELIST` =
`{TAG_BB, TAG_B1}` — logs ERROR and returns False, position does NOT exit
(`trading/stop_monitor.py:1159-1199`). `ExitReason` has **no STAG / SPY_BE
member** (`trading/exit_reasons.py`, grep clean). Separately, `green_verdict`
flags any exit whose reason fails `is_known` (`:1160-166`,
`exit_reasons.py is_known`) → a raw `'stag'` string is unknown → **HARD red
every STAG night**. Required (with STAG): add `ExitReason.STAG` (and decide
MID: BE-stop hits can remain `STOP_LOSS` class, or add `SPY_BE`), add STAG to
the whitelist, and confirm `is_known` accepts it. MID uses `update_stop` (no
whitelist) so no reason needed for the raise itself.

### P1-6. MID "positions opened AFTER the trigger get a BE floor" is NOT implementable as the proposed one-shot
Frozen yaml §5 `applies_to: all_open_ORB_positions_from_trigger_minute_onward
— including positions opened AFTER the trigger` (`:111`). ORB entries are
stop-limit orders that can fill anytime 9:35–10:35, so a fill at 10:20 after a
9:50 MID trigger is routine and BT simulates it. The design §2.3 wiring ("once
per day … for each open watch") only touches positions open **at** the trigger
minute. Required (with MID): a **latched day flag** (`self._mid_triggered`,
DB-persisted for restart) that is re-applied at *every subsequent fill* (raise
that position's stop to `max(stop, entry)` when it fills while latched), not
just once. Without it live under-tightens vs BT → exit-parity divergence.

### P1-7. MID needs a NEW SPY 1-min stream — the design's "existing per-minute tick path" does not exist
There is no SPY subscription or SPY tick/bar path in the ORB engine (grep
clean). `_ingest_bars` only handles candidate/position symbols. `_on_bar` in
StopMonitor early-returns for any symbol not in `_bar_symbols`
(`trading/stop_monitor.py:728`). To feed MID: call
`stop_monitor.subscribe_bars('SPY')` (SPY then fans out to all bar handlers,
watch-less, fine) **and** add a `symbol == 'SPY'` branch in the engine's bar
handler. Two operational traps: (a) Alpaca WS delivers bars **from
subscribe-time forward** (documented repeatedly in this codebase, e.g.
`orb_engine.py:763,1069`) so SPY must be subscribed **before 9:35** or the
09:35 anchor bar is lost and MID is silently dormant; add anchor capture +
backfill like `_backfill_range_if_needed`. (b) fail-open dormant + WARNING when
SPY data is absent (design says so — good). This is new wiring, not a config
tweak; it belongs with the MID slip.

### P1-8. `orb_backtest.py` 900s features-regen timeout blinds the validation
`regen_features` subprocess timeout is 900s (`orb_backtest.py:376-378`) and the
design §3 flags it can never complete a full regen. If it times out, the
nightly features CSV is stale → `bt_data_max_date` goes stale
(`report_common.py:63-84`) → bt_parity/decision_parity **SKIP** (not red, but
blinded — the owner's "validate every night" silently no-ops). Fix: raise /
parametrize the timeout and keep the regen incremental so the nightly features
CSV is always current before the 21:30 check.

### P1-9. Kill rails are pre-committed but unbuilt — daily kill must gate the LIVE entry path
Design §5 commits: day ≤ −$500 → no new entries; week ≤ −$750 → flat + ping;
month ≤ −$1,500 → abandon. None exist in code today. The daily rail is an
**intraday** gate: `check_entries` must consult DB-derived realized ORB day
P&L (`trades.db`, `strategy='orb'`, `trade_date` = ET) and block new entries at
≤ −$500. At $10K this CAN fire (3 concurrent × ~−$214 max loss ≈ −$642 worst
day, `orb_drag_program` monthly table), so it is a real rail, not decorative.
Week/month rails + parity-kill (any HARD red → no new entries) go in the
nightly digest. All must be DB-derived (restart-safe) and ET-dated; verify the
15:45 eod-flat writes realized P&L same ET day so the rail can't be blinded (the
8/14 ignition lesson).

### P1-10. PDT at $10K equity — a burst WILL breach; needs concrete handling before real money
Every ORB trade is a same-day round trip = a day trade. `max_concurrent=3`
means up to 3 day-trades in ONE day; the book shows 9-trade months (2025-09,
2026-07) → multiple day-trades inside a rolling 5-business-day window. On a
**margin** account with equity <$25K, Alpaca enforces PDT: the 4th day-trade in
5 business days flags the account and restricts it (90-day). This is a live
GO-condition, not a footnote. Concrete handling required:
- Confirm the ORB account type. If **cash** account: PDT does not apply, but
  T+1 settlement does — $10K fully deployed today can't be reused tomorrow;
  add a settled-funds-aware entry throttle (else Good-Faith Violations).
- If **margin** <$25K: add a DB-derived rolling **5-business-day day-trade
  counter** in `check_entries`; block the entry that would be the 4th and
  Telegram-warn. Cap effectively ≤3 day-trades/5 days.
Do not go live Monday without one of these wired and tested.

---

## P2 — ADDRESS DURING THE MONTH

- **STAG restart mid-window (MFE loss).** STAG running-MFE + one-shot flag are
  in-memory; `_bar_windows` only holds bars from subscribe-time
  (`stop_monitor.py:713-722`). A restart inside the 20-min window loses/underruns
  MFE. Persist STAG state (or re-derive MFE from rehydrated bars) like touchgo's
  breakout-bar re-capture. Low frequency; ship with STAG.
- **STAG market-exit slippage in a spread blowout.** `force_exit(reason='stag',
  limit_price=None)` uses quote-based marketable pricing; a wide-spread exit can
  realize more than the BT `close*(1-10bps)` model. The §4 exit-parity gate
  (>$75/trade) catches it after the fact; consider a spread guard on STAG.
- **`sizing_attribution` spurious news_drift when PM is OFF.** With B+ PM/news
  disabled, the EoD news-drift soft-warning (`report_common.py:407-414`) still
  fires but is meaningless (news doesn't size anything). Gate it off when
  `pm_mult_enabled=False` to avoid nightly noise. (Not green-blocking — soft.)
- **Selection-observer spread line stale (150 vs 300).** `orb_selection_observer.py:144-148`
  flags >150bps as "live would skip," but the live spread gate is 300bps
  (CLAUDE.md, catalyst-veto ship). Pre-existing over-flag, not a B+ regression;
  the observer's `bt_parity_filter` (`:74-95`) only does the coarse
  gap/price/volume universe screen and does NOT recompute composite/pdr/G1, so
  it will **not** false-alarm on the new z-params/pdr11/G1 — config-sensitivity
  lives entirely in `report_common` (P1-1). Fix the 150→300 line opportunistically.
- **Stale docstrings.** `ExitReason.LOCK_STOP` says "+1.5R…+1R"
  (`exit_reasons.py`); CLAUDE.md/README ORB say 1.5R/1R. B+ uses 1.75R/0.5R
  (matches `study_orb_pipeline_static_lock.py:39-40`). Design §1 already calls
  for the doc fix — do it in this build.
- **STAG-vs-stop same-bar under SIP tick gaps.** Live stop fires intra-bar
  (tick path) and STAG at bar-close, so the stop naturally wins (matches yaml
  precedence `stop_on_low` > `stag_check`). But the OPTX-class SIP tick-gap
  (`stop_monitor.py:747-755`) can let a bar close with STAG firing where BT
  stopped out — same exit direction, small P&L delta, different reason. Note in
  the exit-parity classifier so it isn't read as a bug.

---

## PRECEDENCE ENUMERATION (design §2.2 / yaml §4 completeness check)

Within one 1-min bar, ordering the four events for a live position:
1. `lock_arm_on_high` — armed off `bar.high` (`_maybe_ratchet_from_bar_high`,
   `stop_monitor.py:784-853`) and off ticks.
2. `stop_on_low` — fires intra-bar on the tick crossing `stop_price` (tick
   path), before the bar closes.
3. `mfe_update` — STAG running-max of bar highs (new).
4. `stag_check` — one-shot at bar close when age ≥ 20min.
Plus touchgo Rule M/D on bars 0/1 (evaluated first, `_evaluate_touchgo`).
Live honors yaml precedence `[lock_arm_on_high, stop_on_low, mfe_update,
stag_check]` **by mechanism**: lock/stop are tick/high driven (intra-bar) and
STAG is bar-close driven, so a bar whose low hit the stop is already
exit-in-progress when STAG would evaluate (`force_exit` is idempotent via
`_exit_in_progress`, `stop_monitor.py:1208-1214`) → stop wins, STAG skipped.
The design's precedence list is COMPLETE and matches the yaml, with two caveats:
(a) touchgo-first must be stated as the outermost gate (it is, yaml §4
`interaction`); (b) the SIP tick-gap edge (P2) is the only path that can
reorder stop vs STAG, and it's benign. "STAG never fires when lock armed"
holds: lock arms at `entry+1.75·R_lock` where `R_lock = entry−range_low >
R_stag = range_high−range_low`, so armed ⇒ MFE ≥ 1.75·R_lock ≫ entry+0.25·R_stag
⇒ STAG condition false. NOTE the impl must use `R = range_high−range_low` for
STAG (yaml §4 `R_definition`), NOT the watch's `planned_risk_per_share` used by
the lock (`_r_baseline_and_unit`, `stop_monitor.py:444-456`).

## RISK-HOLE CHECK (design §2.4 "risk-reducing only")

- **G1**: removes trades only → holds. (Landmine is correctness P1-2, not risk.)
- **STAG**: exits early only; never adds size/exposure/widens a stop → holds.
  Residual: market-exit slippage > model in a blowout (P2), and STAG converts a
  possibly-recovering flat trade to a certain scratch — that's the accepted
  design cost, not an exposure add.
- **MID**: `update_stop` is raise-only (`stop_monitor.py:1396`
  `new_stop_price > watch.stop_price`); on a lock-armed position `entry+0.5R >
  entry` so `update_stop(entry)` is a no-op → matches yaml `no_op_if_lock_armed`.
  Cannot lower a stop. Holds. (Note: `update_stop` mutates only the in-memory
  monitor stop, not any resting broker stop leg — fine, StopMonitor fires the
  exit; the broker leg at range_low remains a backstop.)
No counterexample found that adds exposure. The safety property survives, but
only after P1-2 (G1 zero-handling), P1-6 (MID post-trigger fills), and P0-1
(STAG keying) are fixed.

## LATENCY (precondition #1) — largely bought by the config change
B+ disables the PM/news mult (`sizing.pm_dollar_vol_mult: OFF`). `_get_pm_mult`
short-circuits `if not self.pm_mult_enabled: return 1.0` BEFORE any news fetch
(`orb_engine.py:1882` vs the fetch at `:1887`). The 32–73s regression was the
blocking Benzinga/Alpaca news prefetch — with PM off, that call is never made,
so the config change itself removes the primary latency source. Still: keep the
>10s WARNING tripwire and assert (mock-timed test, design §6) that no blocking
news/PM call remains between the 9:35 signal and submit.

---

## ORDERED WORK LIST FOR IMPLEMENTATION AGENTS

### Monday-safe subset (Sat build / Sun rehearse)
1. **orb.yaml + template** — write frozen B+ values from
   `orb_bplus_frozen_params_aug2026.yaml`: z-params (`filter.features`),
   `filter.threshold: 0.012081536791`, `quintile_cutoffs`, `skip_q1: true`,
   `ranking.order Q4-first`, `adaptive_mults` all 1.0, `sizing.pm_dollar_vol_mult.enabled:
   false`, `account_budget_usd:10000 / risk_per_trade_usd:375 / max_concurrent:3`,
   per-pos cap 3333.33, `prev_day_range_veto.min_prev_day_range_pct: 11.0`,
   catalyst_veto on, static lock 1.75R/0.5R, `strategy.enabled: true`.
   Add `filter.g1_veto.{enabled, return_volatility_20d_min:7.106,
   prev_day_range_pct_min:9.226}`. Config-load test.
2. **rv20 live** — add `return_volatility_20d` to `_compute_features` /
   `_get_feature_context` (`orb_engine.py`), numpy `std(ddof=0)*100` over the
   same 20 closes as `study_orb_features.py:308-314`. Golden parity vector test
   (≥10 symbol-days) vs the features CSV. [P1-3]
3. **trading/orb_g1_veto.py** — `g1_reject(rv20, pdr) -> Optional[str]`;
   fail-open KEEP on rv∈{None,NaN,0.0} BEFORE the AND; pdr==0.0 is real. Env
   `ORB_G1_VETO=0`. Unit matrix. Wire `_g1_veto_reject(cand)` into the submit
   loop beside PDR (`orb_engine.py:1836`), `_pdr_vetoed_today.add + plan_submitted`
   for slot-consumed/no-refill parity. Import the same module in the BT
   pipeline. [P1-2]
4. **BT pipeline B+ mode** — parametrize `ACCOUNT/N/RISK/Q_CAPS` and the
   composite threshold in `study_orb_pipeline_static_lock.py:32-41,250,290` to
   read B+ values; integrate `orb_g1_veto`. Regen the B+ ground-truth book to
   the exact path `report_common` reads. Config-vs-pipeline drift test. [P1-4]
5. **report_common repoint + G1 gate** — `latest_bt_trades_csv` → B+ book
   (`report_common.py:47`); extend `decision_parity`/gate-parity to include the
   G1 veto decision live-vs-BT; month scoreboard (running live vs BT-same-window
   band, DD vs −$657, days elapsed); gate `news_drift` off when PM disabled.
   File-wiring test (pipeline-writes == report-reads). [P1-1, §4/§6]
6. **Kill rails** — DB-derived, ET-dated: intraday daily-kill gate in
   `check_entries` (≤ −$500 → no new entries); week/month rails + parity-kill in
   the nightly digest; flat trigger. Tests with synthetic DB rows. [P1-9]
7. **PDT handling** — confirm account type; wire cash-settlement throttle OR a
   rolling 5-business-day day-trade counter that blocks the 4th entry.
   Tests. [P1-10]
8. **orb_backtest timeout** — raise/parametrize the 900s regen timeout; keep
   incremental. [P1-8]
9. **Docs** — fix 1.5R/1R → 1.75R/0.5R in CLAUDE.md/README/`ExitReason.LOCK_STOP`.
10. Full suite green → weekend boot rehearsal (real ExecStart, grep init lines
    incl. G1 gate line, zero orders) → nightly-validation dry run on a historical
    day.

### Tuesday+ (STAG, then MID — each behind its own env kill, parity-gated)
11. **STAG** — fix the design first: **key to `pos.breakout_bar_ts`, not fill**
    [P0-1]. `trading/orb_stag_exit.py` (R = range_high−range_low; MFE from bar
    highs incl. breakout+eval bar; one-shot at first bar-close age≥20min).
    Add `ExitReason.STAG` + whitelist + `is_known` [P1-5]. OpenPosition MFE/flag
    state + restart re-derivation [P2]. Wire from `_ingest_bars`. Import same
    module in BT. Parity vectors incl. a STAG fire, a near-miss, and a
    **lagged-fill** case.
12. **MID** — `trading/orb_spy_tighten.py`. New `subscribe_bars('SPY')` before
    9:35 + anchor capture/backfill + SPY branch in the bar handler [P1-7].
    Latched day-flag re-applied at every post-trigger fill [P1-6]. `update_stop`
    (already exists, `stop_monitor.py:1383`) for the BE raise. Parity vectors
    incl. a MID day and a position that fills after the trigger.

## NOTES CONFIRMED FEASIBLE (no change needed)
- `StopMonitor.update_stop` already exists and is raise-only — MID's amend path
  is NOT missing (design §2.3 "implement if absent" is stale).
  (`stop_monitor.py:1383-1404`)
- z-params/cutoffs already flow from orb.yaml into both live and the BT pipeline
  (`orb_engine.py:417-429`, `study_orb_pipeline_static_lock.py:206-256`) — the
  frozen fit propagates once orb.yaml carries it.
- G1 slot-accounting reuses the proven `_pdr_vetoed_today` mechanism verbatim.
