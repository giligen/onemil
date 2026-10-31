# PREREG — cell 1,731: thin breaks — a CAUSAL break-volume filter (Fable, 2026-10-31; FROZEN before any number)

Read `CONTEXT_BRIEF.md` and `RESULT_1730.md` §"Ranked further hypotheses" first.

## Provenance (honest)
1,730's report-only read: quintiles of v_T / mean(v over the 10 bars before the trigger) on the BT population — Q1
(≤ 0.65×) −0.095 R (TRAIN-H2) / −0.086 R (VAL), Q5 +0.113 / +0.013. **v_T is not causal**: the trigger minute's volume
includes prints after the cross (fractional-minute look-ahead, `feedback_fractional_minute_bar_lookahead`). This cell
measures the same idea with information available AT the crossing print. The read was seen on both splits, so
TRAIN-H2/VAL are a replication; the pass bar below is the only thing that counts.

## Mechanism
A break on thin prints is a drift into an empty book — no demand behind it — and fails back into the base; a break
with heavy participation before/at the cross is a real auction. Alternative: volume before the cross is mostly the
consolidation's own churn and carries nothing (then the causal measure is flat and the 1,730 read was look-ahead).

## Data
BT population and paths as in 1,730 (`cache/opens_1730.parquet` has the trigger bar T, bar j = T−1, v_cons10, level,
d_bps; add what is missing). Tick tape: the 1,438 SIP caches under `research/hod_entry/sip_cache*` (trades with
timestamps; `causal_arming.py fetch_window` l.213 / `resolve_window` l.105 show the format) — report coverage of the
population (availability rail: ≥ 80 % and winner/loser missingness gap ≤ 5 pp, else the tape cell is VOID and only
the bar-level measure is read). Live OOS: `fills_1721.csv` joined to the dry entry ledgers (`logs/hod_dry_entry_ledger*.csv`
columns `arm_bar_vol_x`, `cumvol_to_arm`, `cumvol_over_adv20`) — read-only, as the sign check.

## Measures (all causal; computed per signal before anything after the cross is seen)
M1 **pre-cross rate** = (SIP volume inside minute T strictly before the crossing print) / (seconds elapsed in T) ÷
   (mean per-second volume over the 10 consolidation bars). Tape only. Signals crossing in the first 5 s of T have an
   undefined M1 → excluded from M1 and counted.
M2 **last-bar ratio** = v_j / mean(v over the 10 bars before T) — bars only, always available.
M3 **cumulative participation** = cumulative volume from 09:30 to the end of bar j ÷ ADV20 (the ledger's
   `cumvol_over_adv20` twin).

## Cells (frozen)
V1 for each measure: quintiles on TRAIN-H2 (cuts fixed there, applied to VAL): net R/trade, n, day-clustered t per
   quintile, both splits; ex-top-5 % per quintile. Sign agreement Q1 vs Q5 across splits is the read.
V2 the filter: DROP signals in the thinnest TRAIN-H2 quintile of M1 (tape) and separately of M2 (bars): paired Δ vs
   the unfiltered book (net R/trade, R/week, fills/week, ex-top-5 % Δ, winner-capped), day-clustered t, both TRAIN-H2
   halves (alternating weeks) and VAL.
V3 live OOS: the live fills split at the same cut on the ledger twin of the chosen measure (M2 ↔ `arm_bar_vol_x` if
   that is what it is — the agent documents the ledger column's definition from the engine code before using it):
   net R of kept vs dropped with CI.

## Pass bar (pre-committed)
Adopt the filter on ONE measure iff on BOTH TRAIN-H2 and VAL: paired Δ net ≥ +0.03 R/trade, day-clustered t ≥ 2,
R/week ≥ the unfiltered R/week, ex-top-5 % Δ ≥ 0, both TRAIN halves same sign, fills/week ≥ 3 at the live rate, Q1 the
worst quintile in both splits, AND the live kept-vs-dropped sign agrees (CI reported; it cannot pass alone). Tape
measure preferred if both pass (it is the mechanism); bar measure ships only if the tape cell is VOID or fails while
M2 passes on its own. PASS → spec the engine filter (the engine has bar streams and trade prints via the StopMonitor;
paper book first, rehearsed). FAIL → stated MDE per measure and whether the 1,730 read was look-ahead (M1/M2 flat while
v_T is not = yes). Multiplicity: HOD live-book cells 1,721–1,731; 3 measures × 1 filter = Bonferroni on 3.

## Agent (Opus — anatomy; ≤ 40 calls; one python process; bars_sip.db and the sip caches READ ONLY; no --build-cache,
## no data/*.db writes, no commits/stash, no orders, no config edits; never Read > 300 lines at once; this task IS the
## owner's request)
Script `cell_1731.py` (resumable, verbose, background with a log file), outputs `vol_1731.csv` (quintiles × measure ×
split), `filter_1731.csv` (V2 per split/half), `RESULT_1731.md` (coverage/availability rail, every cell, verdict with
the pre-committed consequence, MDE on FAIL, deviations, ≤ 3 ranked next hypotheses), CONTEXT_BRIEF bullet.
Return ≤ 200 words.
