# ORB B+ RESTART — LIVE DESIGN (2026-08-15)

> **SCOPE DECISION (post-review, accepted)**: per
> `docs/orb_bplus_design_review_aug2026.md` (CONDITIONAL-GO) — **Monday
> ships B+ config + G1 veto only**; STAG + MID slip to Tue/Wed behind
> parity tests (they add ~$84/mo at B+ scale but carry all exit-path/
> new-stream risk). P0 correction adopted: STAG keys to the live
> `pos.breakout_bar_ts` (market breakout bar — exact BT parity), NOT to
> fill time as §2.2 originally proposed. All review P1s are in the
> implementation work list.

**Owner decision (8/15):** restart ORB live **Monday 2026-08-17** on the
rebuilt clean-data configuration — **B+ + the 3-gate combo (G1+STAG+MID)**
at **$10K budget** — run **one month** as the live forward-validation
window with **nightly automated BT-vs-Live validation**, then owner
decision to scale to $20-30K. This document is the implementation spec;
implementation and testing run on Opus agents; nothing ships until the
test plan passes and the weekend rehearsal is green.

Evidence base: `research/orb_drag_program_aug2026.md` (combined book
+$11,610/20mo at $10K, MDD −$657, worst month −$308, 13/20 months+,
zero winner casualties at B+ scale; combo beats 0/100 random rule-sets;
chosen on TRAIN+VAL, single OOS unveil). Clean-book context:
`research/orb_clean_rederivation_aug2026.md`. **The live month IS the
forward validation** the program demanded — at $10K the worst clean
month is −$308 and the risk of running validation live (real fills)
exceeds paper only marginally while producing strictly better evidence.

## 0. Non-negotiable preconditions (before Monday 9:35 ET)
1. **Latency fix**: first-order placement ≤5s after 9:35:00 ET (currently
   32-73s since the 7/07 news prefetch — cost the IREX monster). Root
   cause fix + a >10s WARNING tripwire. (Was already scheduled for Monday
   pre-boot; now part of this program, must land in the weekend build.)
2. All tests green (unit + parity + integration), full suite no
   regressions, weekend boot rehearsal green.
3. Config frozen from `research/orb_bplus_frozen_params_aug2026.yaml`
   (the synthesis agent's authoritative dump — **the only permitted
   source for every threshold; no value may be typed from memory**).

## 1. Configuration (orb.yaml — new baseline)
Source of truth: `research/orb_bplus_frozen_params_aug2026.yaml`.
Summary of deltas vs the retired Stage-0 config:
- `strategy.enabled: true` (re-enable — this REPLACES the 8/14
  go-to-zero state; the owner order's re-entry bar is met: rebuilt
  owner-approved param set at $10K)
- `sizing.account_budget_usd: 10000`, `risk_per_trade_usd: 375`,
  `max_concurrent: 3`, per-position cap per frozen yaml
- **Composite selection: TRAIN-refit z-params (clean 2025H1 fit)**
  replace the void contaminated fit; threshold = frozen q40 VALUE
  (0.012081536791). **Quintile logic REMAINS on top of the threshold**:
  frozen TRAIN quintile cutoffs, skip_q1: true, Q4-first ranking —
  only the adaptive MULTS are neutralized to 1.0. Per-pos cap
  $3,333.33 (budget/N), min_stop_pct sizing floor 1.0.
- `filter.prev_day_range_veto.min_prev_day_range_pct: 11.0` (was 8.0)
- Catalyst veto: ON (unchanged semantics)
- PM news-gated sizing mult: **OFF** (monster-concentrated evidence;
  amplifies tail risk on a small book)
- Touchgo: unchanged (re-validated on clean data at shipped thresholds)
- Static lock: **arm +1.75R → stop +0.5R** (the values the B+ book
  actually used — orb.yaml 2026-05-08; the "1.5R/1R" in CLAUDE.md/README
  is stale docs-drift, fix the docs in this build) + 15:45 flat
- `orb.yaml.template` updated to match (new-node parity)

## 2. New shared modules (parity by construction — each imported by BOTH
the live engine and the BT pipeline; the ONLY place its logic lives)

### 2.1 `trading/orb_g1_veto.py` — G1 selection veto
- `g1_reject(return_volatility_20d, prev_day_range_pct) -> Optional[str]`
  KEEP iff (rv >= RV_MIN and pdr >= PDR_MIN); missing/NaN/zero rv →
  KEEP (fail-open, matches BT). Thresholds imported from config with
  frozen defaults.
- Live call site: `orb_engine` submit loop, POST-ranking, alongside the
  PDR veto — **vetoed pick's slot stays EMPTY (no refill)**, same
  invariant as PDR (refill re-tested toxic).
- Log line: `G1 VETO {sym}: rv={..} pdr={..}` (grep-able); env kill
  `ORB_G1_VETO=0`.
- Note: `prev_day_range_pct` and `return_volatility_20d` are existing
  features; live must compute `return_volatility_20d` identically to
  `study_orb_features.py` (20-day daily-return stddev — verify the
  exact formula and daily-bars source in the impl, add a parity test
  vector from the features CSV).

### 2.2 `trading/orb_stag_exit.py` — stagnation exit (STAG)
- Rule: if MFE < entry + 0.25×R (R = range_high − range_low, the
  OPENING RANGE, not entry−stop) within 20 minutes → exit at market.
  One-shot check at the first bar close ≥20min after clock start; MFE
  from bar highs incl. breakout + evaluation bars. **Known live
  deviation (required)**: BT keys the clock to the market breakout
  bar; live keys to ACTUAL FILL + the touchgo late-fill-guard pattern
  (max_breakout_age_min). Parity test mandatory before enable. All
  other semantics per frozen yaml §4 verbatim (precedence: stop-on-low
  beats stag within a bar; touchgo evaluated first).
- Live wiring: evaluated from `orb_engine._ingest_bars` on each 1-min
  bar close for open positions younger than the window (same pattern as
  touchgo `_evaluate_touchgo`); fires
  `stop_monitor.force_exit(sym, reason='stag', ...)`. Telegram
  `[ORB] STAG EXIT {sym}` with MFE and age.
- Interaction: touchgo (min 1-2) fires first if both would; STAG never
  evaluates after the window closes; never fires on a position whose
  lock is already armed (by construction MFE ≥ arm ⇒ MFE ≥ 0.25R).
- Env kill `ORB_STAG_EXIT=0`; config block `filter.stag_exit.{enabled,
  mfe_r, window_min}`.

### 2.3 `trading/orb_spy_tighten.py` — SPY context stop-tighten (MID)
- Rule: if SPY declines ≥0.75% from its 9:35 anchor (anchor + trigger
  arithmetic + window per frozen yaml §MID) → move ALL open ORB stops
  to breakeven (entry fill price), once per day (one-shot; positions
  already lock-armed keep their higher lock stop — stop only ever moves
  UP).
- Live wiring: `orb_engine` already has a per-minute tick path; add a
  SPY minute-close check (SPY bars via the existing market-data client;
  if SPY data unavailable → rule dormant + WARNING, fail-open). On
  trigger: for each open watch, `stop_monitor.update_stop(sym,
  max(current_stop, entry_fill))` — implement `update_stop` if absent
  (raise-only guard: never lowers a stop).
- Telegram `[ORB] SPY TIGHTEN fired — {n} stops → breakeven`; env kill
  `ORB_SPY_TIGHTEN=0`.

### 2.4 Safety property (review gate for every module)
All three gates are **risk-reducing only**: G1 removes trades, STAG
exits early, MID raises stops. No code path may add exposure, widen a
stop, or increase size. The implementation review must verify this
property explicitly per module.

## 3. BT ground truth (nightly parity target)
`study_orb_pipeline_static_lock.py` + `orb_backtest.py` must simulate
the NEW config (B+ sizing, TRAIN-refit composite + threshold, pdr11,
G1/STAG/MID via the SAME shared modules) so the nightly BT book is the
thing live is compared against. Deliverables:
- Pipeline reads the new orb.yaml values (kill the hardcoded copies the
  8/14 audit flagged where feasible; at minimum sync + add a
  config-vs-pipeline drift test).
- Regenerate the BT ground-truth book under the new config for the full
  clean window → this becomes `analysis_results/orb_bplus_book.csv`
  (keep the old static-lock book file for history).
- `orb_backtest.py`: raise/parametrize the 900s features-regen
  subprocess timeout (it can never complete a full regen — found 8/14).

## 4. Nightly BT-vs-Live validation (the owner's "validate every night")
Extend the existing 21:30 green check (report_common) — new/updated
gates, all Telegram-reported nightly, each ET-dated:
1. **Selection parity**: live SCORED/ranked set vs BT candidate set for
   the day; every BT pick ordered live or explained (existing gate) —
   now against the B+ book.
2. **Gate parity**: per-day G1/PDR/catalyst veto decisions live vs BT
   replay — any decision-flip = HARD red.
3. **Fill parity**: BT-filled + live-unfilled = HARD red (IREX gate,
   already shipped 8/14) + entry slippage vs BT fill model (30bps).
4. **Exit parity**: per-trade exit reason class (stag/touchgo/lock/
   stop/spy_be/eod) live vs BT resim of the same day; P&L delta per
   trade > $75 (20% of risk) = investigate.
5. **Latency tripwire**: first-order placement >10s = WARNING on the
   digest.
6. **Month scoreboard**: running live P&L vs BT-same-window ± band;
   drawdown vs the −$657 clean MDD; days elapsed of the validation
   month. Printed nightly, feeds the month-end scale decision.

## 5. Ramp & kill rules (pre-committed)
- **Daily kill**: realized ORB day P&L ≤ −$500 → no new entries that day
  (mechanical; ~1.3x worst clean month in a day = way outside model).
- **Week kill**: realized week ≤ −$750 → flat + owner ping (2.5x worst
  clean month).
- **Abandon gate (month)**: cumulative ≤ −$1,500 (≈2.3x clean MDD) →
  ORB back to zero, no owner approval needed to STOP (stopping is
  always allowed; only restarting needs approval).
- **Parity kill**: any HARD red (decision-flip class) → no new entries
  until root-caused (same doctrine as ignition).
- **Scale rule**: after ~21 trading days (≈9/15): if (a) live P&L within
  the BT band, (b) zero unexplained HARD reds, (c) above water → owner
  decision to $20K or $30K (2-3x). Not automatic — owner says the word.
- All kills DB-derived (restart-safe), ET-dated, Telegram on trigger.

## 6. Test plan (Opus implementation agents; suite must stay green)
- Unit per module: G1 keep/veto/fail-open matrix; STAG fires/holds at
  boundary (19:59 vs 20:01), R arithmetic, never-after-window,
  never-when-armed; MID trigger arithmetic, one-shot, raise-only,
  SPY-data-missing dormant path.
- **Parity tests** (the load-bearing ones): for each module, golden
  vectors extracted from the B+ book resim — same inputs through the
  live-code path must reproduce the BT decision/exit for ≥10 real
  trades incl. edge cases (a STAG fire, a STAG near-miss, a MID day,
  each G1 side). Mirrors `tests/test_orb_touchgo_parity.py`.
- Config tests: orb.yaml values load; drift test pipeline-vs-config.
- Integration: synthetic day driving scanner→engine→StopMonitor with
  all three gates active (mocked alpaca, real Database — the 8/14
  save_trade lesson).
- Latency: entry-path test asserting no blocking news/PM call remains
  between 9:35 signal and submit (mock-timed).
- Full suite green. Then weekend boot rehearsal (real ExecStart, grep
  init lines incl. new gate lines, zero orders).

## 7. Timeline
- **Sat**: params yaml verified → Opus design validation → Opus
  implementation (modules + config + pipeline + tests) → review vs §2.4
  safety property → full suite.
- **Sun**: BT ground-truth regen under new config; boot rehearsal;
  nightly-validation dry run against a historical day; fix cycle.
- **Mon pre-boot (11:20 cron exists)**: latency fix verification, final
  config check, `strategy.enabled: true`, boot 12:30, I verify init
  lines + first session live watch (9:35-10:35 ET).
- **Mon night → +1 month**: nightly validation digest; weekly rollup in
  the Saturday review.

## 8. Rollback
- Any gate individually: env kill / config flag + restart (zero state).
- Whole restart: `strategy.enabled: false` + restart (the 8/14
  go-to-zero path, proven).
- Config rollback: git revert of the config commit; the frozen-params
  yaml + this doc pin every prior value.

## 9. Open items for the design validator (Opus) to check hard
- Exact live formula availability for `return_volatility_20d` at 9:35.
- StopMonitor `update_stop` semantics vs existing lock machinery.
- STAG/touchgo/lock precedence — enumerate all orderings.
- Whether the BT pipeline can consume the shared modules without
  perf regressions (nightly must still finish before 21:30 check).
- PDT rule interaction at $10K equity (margin account, ~4 trades/mo at
  B+ — but bursts may exceed 3 day-trades/5 days; specify handling).
- The 2026-08 partial month: live starts mid-August — the month
  scoreboard's BT band must use same-window BT, not calendar-month.
