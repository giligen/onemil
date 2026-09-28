# REBUILD 1,621 — independent rebuild of PREREG_1617 Frame C (W = 3 RTH minutes)

Built from the prose of `PREREG_1617.md` (Frame C) and `PREREG_1487.md` (the mechanics C reuses)
ONLY. Did not open `cell_1487.py`, `cell_1621.py`, `cell_1621_fills.csv` or `RESULT_1621.md`.
Code: `rebuild_1621.py`. Fill-level output: `rebuild_1621_fills.csv` (9,911 rows, one per base
fill, eligible or not). Summary: `rebuild_1621_report.csv`.

## Rule implemented
Base fills = `causal_arming_causal.csv` status == `fill` (9,911; TRAIN 4,398 / VAL 5,513), joined
to `model_1478_L3_predictions.csv.outcome_R` (base net R, 0 missing) and
`features_1478_A.csv.half_entry` (0 missing).

`fill_min` is a fractional ET minute-of-day (verified: AAP/2025-07-01 fill_min=605.31 lands on
the ET-minute-605 bar in `bars_fills_1478.db`, converted via `America/New_York`). m0 =
floor(fill_min). A base fill is ELIGIBLE iff both hold:
1. still open: base `exit_m > fill_min + 3`;
2. no withdrawal: no bar with minute label in [m0+1, m0+3] has low <= level - $0.01.

Eligible fills enter LONG at the open of the minute-(m0+4) bar, ask = that bar's open +
`half_entry` (disclosed proxy: the fill-instant half-spread). stop = level - $0.01, R'' = entry -
stop, target = entry + 2R''. Path walked forward from the entry bar: EOD bar (minute >= 955,
15:55 ET) exits at its open; else low <= stop exits at stop (or the bar's open if it gapped
through); else high >= target exits at target; first match wins (stop-priority on a bar
touching both). Costs: stop exit charges `SLIP_STOP_BPS` from `cell_1478.py`
(TRAIN 13.83 bps, VAL 11.94 bps, = 0.88*filled + 0.12*tail); target exit charges 0 (limit fill);
EOD exit charges the task-given constants (TRAIN 11.5 bps, VAL 9.7 bps). `R'' < 0.5% of price`
kept in the CSV but flagged `primary_book=False` and excluded from the reported means.

## Result (VAL / TRAIN-H2)
| split | eligible / base | runners lost (target hit within W) | primary n | mean net R'' | day-clustered t | ex-top-5% | fills/wk | base outcome, eligible (calib.) | base outcome, primary (paired) | paired mean Δ | paired t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN | 1,011/4,398 (23.0%) | 18/1,240 (1.5%) | 752 | **-0.160** | -2.53 | -0.275 | 27.9 | +0.324 | +0.457 | -0.617 | -11.74 |
| VAL | 1,197/5,513 (21.7%) | 24/1,517 (1.6%) | 902 | **-0.253** | -4.28 | -0.372 | 41.0 | +0.321 | +0.428 | -0.682 | -12.49 |

`why` on eligible fills: stop 1,542, target 624, eod 42 (of 2,208 eligible; 554 fall under the
0.5%-of-price floor and are excluded from primary). Ineligibility reasons on the 9,911 base
fills: withdrew-in-window 7,321, no entry bar (sparse tape) 292, closed-in-window (before
entry, non-withdrawal) 48, non-positive R (stop >= would-be entry) 42.

## Read against the frozen pass bar (C, PREREG_1617 §Pass bar)
Bar: mean net R'' >= +0.15, t >= 2.5, ex-top-5% > 0, >= 3 fills/wk, TRAIN-H2 same sign.
VAL mean is **-0.253** (t -4.28, ex-top-5% -0.372) — wrong sign, large magnitude, and TRAIN-H2
agrees in sign (-0.160) rather than diverging, so this is not a TRAIN-only artifact. **FAILS
on every leg of the bar.** The paired comparison shows why: on the same eligible fills the base
rule (enter at the break, standard exits) nets +0.43R (VAL), while waiting 3 minutes for
no-withdrawal confirmation and re-entering at a tighter, closer stop nets -0.25R — a ~0.68R
paired loss (t ~ -12.5), consistent in direction with the disclosed 1,487 result at W=15
(-0.05R): shortening the confirmation window from 15 to 3 minutes does not fix the mechanism,
it makes it worse (tighter stop -> more stop-outs before the 2R target: 1,542 stops vs 624
targets among eligible fills, a ~2.5:1 ratio).

## Refuters checked
- **Look-ahead of the window**: the withdrawal check only reads bars strictly after `fill_min`
  and up to `fill_min+3`; the entry bar is the first bar after that window; the base
  eligibility check (`exit_m > fill_min+3`) uses the base engine's own recorded exit minute,
  not anything derived from the new trade. No field after minute m0+3 is read before the
  eligibility decision is made.
- **Run-up cost**: entry is priced at the post-window bar's open + the fill-instant half-spread
  (a disclosed proxy, stated as such in both PREREGs), not the level or the pre-window price —
  the 3-minute run while waiting is priced into the entry via the later bar's open.
- **Obtainability**: entry is a buy at the ask at the open of a specific future minute bar —
  obtainable by a resting/marketable order at that bar's open, same obtainability class as
  1,487's own entry.
- **Tails**: ex-top-5% mean stays negative and larger in magnitude than the raw mean on both
  splits (VAL -0.372 vs -0.254 raw) — the negative result is not a top-tail artifact; it's
  broad-based (the tail, if anything, cushions the loss rather than carrying it).
- **Price scale**: entries (min $20.1, max unremarkable) and net R bounded in [-2.60, +2.00]
  (target cap) with no extreme outliers — no raw/adjusted split issue detected on this
  minute-bar population (bars are unadjusted intraday prints already used by the base engine).

## Caveats / what this rebuild cannot certify
- This is the INDEPENDENT rebuild leg only; the >=99%-agreement cross-check against
  `cell_1621_fills.csv` (which this task deliberately never opened) must be run by whoever holds
  both files.
- `window_bar_count` is 3 for 1,914/2,208 eligible fills; 294 have 1-2 bars (sparse tape,
  thin names) — the withdrawal check still holds (any bar present with low <= stop trips it),
  but a name with a fully missing window would silently pass the no-withdrawal test on no
  evidence; this rebuild does not separately flag "window has 0 of 3 possible bars" as
  ineligible, unlike the "no entry bar" case which IS excluded. Report this share if the
  original scores it differently.
- Count-matched null (>=99th pct, part of the "as 1,487" pass bar) was not computed here — out
  of scope for the rebuild leg per this task's step budget; moot given the sign and magnitude
  of the raw failure.

## Verdict
**FAILS the Frame C (1,621) pass bar on this independent rebuild** — wrong sign on VAL (-0.253R,
t -4.28), same wrong sign on TRAIN-H2 (-0.160R), tails don't rescue it. Per PREREG_1617 §
Independent check and consequences: "FAIL -> each frame closes with its numbers; the loop
continues with the next mechanism." Mechanism: the 3-minute no-withdrawal filter selects fills
that already re-entered near the level with a tighter stop; that stop gets hit ~2.5x more often
than the 2R target is reached, which is not offset by the calibration cohort's own (positive)
base-rule edge.
