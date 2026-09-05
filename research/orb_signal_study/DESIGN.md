# ORB entry-signal study — DESIGN (pre-registered 2026-09-05, before any number)

Owner (9/5): "AFTER we are done with the current study reports, design the
research and fire it up. I won't be here so no questions — run it from start
to finish. End only once you have full results and recommendations/proposals.
Telegram status every hour." Deliverable = `REPORT.md` in this directory with
proposals; **nothing ships to live from this study** (decisions are joint;
proposals go to the owner first).

Source sweep: `research/orb_entry_signals_web_sweep_sep2026.md`.

## 0. Ground rules

- **Base features**: the entered-inclusive, wrappers-IN production regen of
  9/5 (`analysis_results/orb_features_<ts>.csv`), frozen as
  `features_base.csv` here. Baseline book = the pipeline on it, must equal the
  production book (`compare_books.py` diff = 0).
- **Full-stack lever isolation**: every candidate is ONE knob added to the
  production stack (universe → composite → Q1 → PDR → catalyst → slots →
  sizing → touchgo → static lock → 15:45 flat). Per-rule sign-flip analysis is
  diagnostic only (feedback memory: it misled ship decisions in August).
- **Eras**: TRAIN = 2025-01-01..2025-06-30; OOS-1 = 2025-07-01..2025-12-31;
  OOS-2 = 2026-01-01..2026-09-04. Thresholds are fit on TRAIN only.
  Composite weights, quintile edges and adaptive mults stay FROZEN
  (2026-07-03 audit: refits lost $34–47K OOS).
- **Two levels of evidence** (the book is thin — ~70 fills / 21 months):
  - **L1 candidate level**: all ~13.5K candidate rows, `pnl_pct` of the
    raw simulated ORB trade (entered rows) — "does the signal exist":
    mean pnl_pct above vs below threshold, bootstrap 95% CI of the
    difference (2,000 resamples), sign consistent in TRAIN, OOS-1, OOS-2.
  - **L2 book level**: the full-stack pipeline at $10K stage sizing —
    "does it help THE book": sized P&L, MDD (daily curve), negative months,
    per-era P&L, fill rate, picks/day, top-10 giants retained.
- **One knob at a time, one pass at a time**, `ulimit -v 6500000`, no passes
  13:00–20:15 UTC Mon–Fri (live session; node-freeze rule). All outputs in
  `research/orb_signal_study/`; production `analysis_results/`, `orb.yaml`,
  `cache.db` untouched (side DB only for the new feature's data).

## 1. Pre-committed decision rules

### 1a. Single candidate → PROPOSE-TO-SHIP if ALL hold (L2, OOS-1 + OOS-2 combined vs baseline)
1. Sized P&L ≥ baseline × 1.05 (≥ +5%).
2. MDD not worse than baseline − $100 (baseline ≈ −$509 → floor −$609).
3. Negative months ≤ baseline.
4. Neither OOS era worse than baseline by more than $250.
5. Top-10 giants (by sized P&L in the baseline) retained: ≥ 8 of 10 still picked.
6. Picks/day not cut by more than 50% (a filter that guts the book is a different strategy — flagged, not shipped).
7. L1 effect exists: bootstrap CI of the pnl_pct difference excludes 0 in the pooled sample AND the sign holds in ≥ 2 of 3 eras.

Outcomes: **PROPOSE** (all 7), **PARK** (L1 yes, L2 no — signal exists but the stack already captures it or the book is too thin; revisit with more live data), **REJECT** (L1 no).

### 1b. Threshold fitting
Grid on TRAIN only; pick the threshold with the best TRAIN L2 P&L subject to MDD ≤ baseline TRAIN MDD − $100; then evaluate that single threshold on OOS. If the TRAIN-best is at a grid edge, report the edge and do not extend the grid.

### 1c. Pair → PROPOSE only if
- both singles PROPOSE (or one PROPOSE + one PARK with L1 yes),
- the pair beats the better single on P&L AND MDD AND negative months,
- interaction = (A+B) − A − B reported; a pair whose lift is the sum of the singles is "additive" (ship both as singles); a pair with negative interaction is "redundant" (ship the better single); a positive interaction must hold in both OOS eras.
- Max 3 new knobs stacked. Order of operations is part of the rule and stated.

## 2. Candidates (from the sweep, ranked)

| # | Knob | Layer | Form(s) tested | Hook |
|---|---|---|---|---|
| C1 | `rvol_open5` = vol(9:30–9:35) / mean of the same window over the prior 14 trading days (Zarattini stocks-in-play) | pre-ranking | (a) veto `< t`, grid t ∈ {0.5, 1.0, 1.5, 2.0}; (b) rank key: order by rvol desc, composite as tie-break (the paper's top-N form); (c) L1 quintile read | `ORB_EXP_RVOL_VETO`, `ORB_EXP_RVOL_RANK` |
| C2 | range-candle direction gate (5-min range closed green: `range_return_pct > 0`; alt `range_close_position ≥ 0.5`) | pre-ranking (a) and post-ranking no-refill (b) | both forms × both placements | `ORB_EXP_RCP_GATE=pre|post`, `ORB_EXP_RCP_FORM=green|upper` |
| C3 | midpoint-reversal kill: after fill, first CLOSED 1-min bar with close < range midpoint before +0.5R was touched → exit at next open (NQ tick study 71%→23%) | post-fill exit (shared-module shape, like touchgo) | on/off | `ORB_EXP_MID_KILL=1` |
| C4 | one re-arm: after a touchgo/mid-kill exit inside the 60-min window, re-place the stop-limit at range_high + 30 bps once, same slot, same day (second-episode) | entry mechanics | on/off (requires C3 or touchgo exit first) | `ORB_EXP_REARM=1` |
| C5 | range/ATR14 tier: `range_size / atr14_t1` buckets {<0.3, 0.3–0.6, >0.6}: READ first; a veto test only if a tier is negative in both OOS eras | read → optional pre-ranking veto | read; then `ORB_EXP_RATR_MIN/MAX` | env |

Diagnostics run for every candidate: (c) candidate-alone on the raw candidate set (L1), (d) leave-one-out: full stack + candidate minus the overlapping existing layer (C1 vs `range_total_volume`/`prev_day_volume_vs_20d` weights; C2 vs `range_close_position`/`last_bar_green`; C3 vs touchgo Rule D; C4 vs the no-refill invariant).

Pairs (Phase 3, survivors only): C1+C2 (the Zarattini pair), C3+C4 (the NQ pair), C1+C5, plus any other survivor pair; triple only if two pairs PROPOSE.

## 3. Engineering plan

**P0 — hooks (behind env flags, default OFF → byte-identical baseline, guarded by a test)**
- `trading/orb_experimental_rules.py`: pure functions `rvol_veto`, `rvol_rank_key`, `range_direction_gate`, `midpoint_kill`, `rearm_allowed` + docstrings citing the sweep.
- `study_orb_pipeline_static_lock.py`: read the sidecar (`ORB_BT_SIDECAR_CSV`, joined on symbol+date), apply the pre-ranking / post-ranking hooks, exit-sim hooks for C3/C4.
- **Exit-sim cache** (`ORB_BT_EXITSIM_CACHE`): variants that do not touch entry/exit mechanics (C1, C2, C5, their pairs) reuse the per-row simulated exits from the baseline pass instead of reloading 13.5K pairs of bars (~35 min → seconds). Cache keyed by a hash of the exit config. C3/C4 passes always reload bars.
- `tests/test_orb_experimental_rules.py`: unit + "flags off = byte-identical book" guard.

**P1 — data for C1** (parallel with P0): `scripts/build_orb_open5_volume.py` → side DB `data/research/orb_open5_volume.db` (symbol, date, vol_open5) for every candidate symbol-day and its 14 prior trading days: cache.db bars first, Alpaca 1-min (9:30–9:35 window) for the rest, resumable. Sidecar `sidecar_rvol.csv` with `rvol_open5` (NaN when < 10 of 14 prior days exist → fail-open, counted).

**P2 — singles**: baseline pass (must equal production) → C5 read → C1 (grid on TRAIN, then OOS) → C2 → C3 → C4. Each: L1 table, L2 table, diagnostics (c)/(d), verdict per §1a, appended to `REPORT.md` as it lands.

**P3 — pairs** among survivors per §1c.

**P4 — REPORT.md** final: summary table, verdict per candidate/pair, proposals (what to shadow live first, what to park), caveats, exact commands to reproduce; Telegram final.

**Status**: `STATUS.md` here is rewritten at every step; `scripts/study_status_pinger.sh` sends its last `STATUS:` line to Telegram hourly until `DONE` appears.

## 4. Schedule (UTC)
T0 = BF chain complete (regen-7 → exact Stage-2 → BF Stage-2 name filter → PIT top-up → union → BF verdict), expected ~00:00 Sun 9/6.
P0+P1 ~2–3h → baseline + C5 + C1 grid ~1h (cached passes) → C2 ~0.5h → C3, C4 ~1.5h each (full passes + code) → pairs ~1–2h → report. **ETA Sunday 9/6 afternoon UTC.** Sunday has no live session, so no pause applies; if it spills into Monday the 13:00–20:15 pause holds.

## 5. Questions logged for the owner (answer when back; none block the study)
1. **Promote regen-7** (`data/bull_flag_cache_causal_full_20260905.csv`) to the production BF cache path (with a .bak)? Never-overwrite rule → needs your word. Until then production Stage-2 still reads the retired cache.
2. **Study outcomes are proposals only** — confirm nothing ships to live without the joint call (assumed).
3. **Live-session pause** for heavy BT jobs 13:00–20:15 UTC weekdays (assumed yes; costs time, buys node safety).
4. **The WhatsApp ORB rule, verbatim** — to map against the candidate table.
5. ~~Trader start on Monday~~ — RESOLVED: auto-start at 12:30 UTC weekdays (journal 9/4: "Started onemil-trader.service" 12:30:02; 12:40 watchdog). Monday boots on the new BF trail code without anyone present.
