# PREREG — cell 1,730: the entry-cap ladder (does a tighter resting stop-limit cap buy gross, or only un-fill?) — Fable, 2026-10-31 (FROZEN before any number)

Read `CONTEXT_BRIEF.md` first. Owner's standing ask: improve HOD; the gross lever is entry quality.

## Where it comes from
Live fills (RESULT_1721.md, `fills_1721.csv`, n 231): net R by entry slip (fill − level, bps): ≤ 5 **+0.13**, 5–10 −0.05,
10–15 −0.00 (gross +0.145), > 15 −0.31 (n 22). The live resting stop-limit caps the fill at level + 15 bps (engine log:
`limit 0.1500%`; `config.yaml hod_break.cap: 0.006` is a DIFFERENT parameter — the agent states what each one does from
the code, `trading/hod_break_engine.py` ~l.324 and the order build ~l.2338). Everyone has ASSUMED a tighter cap un-fills
the good 10–15 bps breaks rather than filling them cheaper; nobody has measured the unfilled counterfactual. Cell 1,728
showed BT gross at the level = live gross, so the BT population can answer this at power.

## Mechanism (stated before the number)
A break whose first print after the cross is far above the level is a fast, crowded break: the chase is the spread of
a thin name and the follow-through is already spent (live > 15 bps bucket −0.31 R). A cap at c bps declines those and
keeps the orderly breaks. Alternative: fast breaks are the real ones and declining them removes the winners (then the
ladder shows net R FALLING as the cap tightens, and the cap stays).

## Data
* BT population: `research/hod_entry/1663_features.csv` + `1676_features.csv` (5,506 live-rule fills, TRAIN-H2 / VAL;
  columns for level, stop, trigger minute, fill, exit; the agent documents which), minute paths `1681_paths.parquet`,
  raw bars `research/bf_zero/bars_sip.db` `bars(symbol, day, t, o, h, l, c, v)` (27 GB, READ ONLY, query per (symbol,
  day) only; the trigger-minute bar and the bars after it are needed).
* Live: `fills_1721.csv` (entry_slip_bps per fill, net R) as the out-of-sample check — the bucket table above is the
  pre-existing read; NOT re-fit.
* Cost (measured, never a band): exit 12 bps marketable on stop/EOD exits, target prints pay nothing; entry = the fill
  price defined below (no extra slip charged — the open IS the slip).

## Fill model (causal, obtainable; the same for every rung)
For each BT signal with trigger bar T (the first 1-min bar whose high crosses the level), `d = (open_T − level)/level`
in bps. Rung cap c ∈ {0, 5, 10, 15, 25, 50, ∞}: **fills iff d ≤ c, at price open_T** (never the touch of the level;
`open_T < level` fills at the level — a resting stop is triggered by the cross, not before). `d > c` = declined (the
engine's limit would not be marketable at the first print). Report separately the share of declined signals where the
bar's low later came back to ≤ level + c within the same minute ("retrace fills" the live engine might get) — report
only, not counted. Exits as in the live rule (stop = consol low, 2 R target, 15:55 flat) on `1681_paths.parquet` / bars.
No-trade minutes = flat. If `open_T` is not recoverable for a signal, the signal is dropped from EVERY rung (same
population across rungs) and the count is reported.

## Cells (frozen)
L1 ladder c ∈ {0, 5, 10, 15, 25, 50, ∞}: n fills, fills/week, net R/trade, R/week, day-clustered t of the PAIRED Δ vs
   c = 15 (per signal: kept-in-both vs changed), ex-top-5 %, winner-capped (+2 R), both TRAIN-H2 halves (alternating
   weeks) and VAL; per split.
L2 the decomposition at c = 15: net R of fills by d bucket (≤ 5, 5–10, 10–15) — the BT twin of the live table. If the
   BT's ≤ 5 bucket is NOT better than 10–15, the live table was noise and the ladder is read as such.
L3 declined-signal counterfactual at c = 5 and c = 10: what the declined signals would have earned if filled at open_T
   (the value we give up) and the retrace-fill share.

## Pass bar (pre-committed)
Adopt cap c* < 15 iff on BOTH TRAIN-H2 and VAL: paired Δ net ≥ +0.03 R/trade with day-clustered t ≥ 2, R/week ≥ the
c = 15 R/week (fewer fills must not cost weekly R), ex-top-5 % Δ ≥ 0, both TRAIN halves same sign, fills/week at the
live rate (scale by the live/BT fill ratio) ≥ 3, AND the live fills' cohort with slip ≤ c* has net ≥ the live mean
(OOS sign agreement; its CI is reported, it cannot pass alone). PASS → spec the engine cap change (paper book, rehearsed)
the same day. FAIL → cap stays 15; the RESULT states the MDE of the paired Δ and which direction the ladder leaned.
A LOOSER cap (25/50) is adopted under the same bar mirrored; the live > 15 bucket (−0.31 R, n 22) is the OOS check.
Multiplicity: HOD live-book cells tonight 1,721–1,730; the ladder is one family (7 rungs), Bonferroni on the rung.

## Agent (Opus — anatomy; ≤ 40 calls; one python process; no --build-cache, no writes to data/*.db or bars_sip.db, no
## commits/stash, no orders, no config edits; never Read a file > 300 lines in full; this task IS the owner's request)
Script `cell_1730.py` (verbose, resumable per (symbol, day) chunk; cache the trigger-bar opens to
`research/hod_live_anatomy/cache/opens_1730.parquet`), outputs `ladder_1730.csv` (per rung × split), `RESULT_1730.md`
(every cell, the fill-model counts, the cap-parameter clarification, deviations, MDE on a FAIL), and — this is the
digging part — a ranked list of ≤ 3 further entry-quality hypotheses with mechanisms from what the ladder shows (e.g.
d as a continuous feature, the trigger bar's volume vs the consolidation's, time since the level formed), each with the
file/column that would test it. Return ≤ 200 words. Update `CONTEXT_BRIEF.md` with a 1,730 bullet.
