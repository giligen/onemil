# PREREG — weekend cells 1,429 / 1,430 / 1,431 / 1,439 / 1,440 / 1,441 / 1,442, frozen 2026-09-25 18:35 UTC before any number

Common to every cell: base book = the E1 fills of `PREREG_1427.md` on TRAIN-H2 and VAL (`research/hod_entry/
sip_rebuild_val.csv` + the TRAIN-H2 output; the TEST file is read ONCE per cell only if the VAL bar passes); once cell
1,438 reports a PASS on VAL, its causal-arming fills replace the E1 fills as the base for any cell not yet run.
Statistics: per-fill paired ΔR against the base where the cell modifies the same fills; day-clustered t via OLS on the
day cluster; ex-top-5 %; fills/week at first-12/day 4-concurrent (`research/hod_consol/run_consol.simulate_slots`).
Costs: the measured half-spread on entry, B0's exit cost, and — new for every cell — a stop-slip charge of 30 bps on
stop exits (the review's realistic middle case), reported both with and without. No constant of E1 changes. Multiplicity:
the programme count is 1,442; every cell is reported whether it passes or fails.

## 1,430 — exits on the winning fills
Variants, each paired on the same fills against B0's exit: (a) time stop at 90 min after entry (exit at that bar's open);
(b) breakeven lock: stop → entry once a bar's high ≥ entry + 1 R; (c) ORB-style lock: high ≥ entry + 1.5 R → stop →
entry + 0.5 R; (d) 50 % scale-out at +2 R, remainder to the B0 rules; (e) exit when a bar closes below session VWAP after
+0.5 R was reached; (f) close at 14:30 instead of 15:55. Pass: ΔR ≥ +0.05 on TRAIN-H2 AND VAL, VAL t ≥ 2, worst week not
worse than B0's. Consequence: PASS → the winning exit ships to the live engine behind a flag after a rehearsal.

## 1,429 — fill-quality sizing
From the quote the instant BEFORE the trigger print: ask distance d = ask/level − 1. Risk multiplier 1.5× if d ≤ 5 bps,
1× if 5 < d ≤ 15 bps (fills only exist for d ≤ 15). Report-only: the trigger print's size class (round lot / odd lot)
and its mean R. Pass: book R per unit of risk ≥ the flat book's + 0.05 on both holdouts, worst week not worse.

## 1,431 — no-fill cohort short
Population: signals where the break bar closed ≥ 15 bps above the level and E1 did NOT fill (ask > limit at the cross).
Short at the next bar's open (the B0 fill rule mirrored, cap 60 bps), stop = break-bar high + $0.01, target 2 R, cover
at 15:55; borrow: names flagged shortable in the asset dump, borrow fee ignored (report the share not shortable). Pass:
mean net R ≥ +0.10 on both holdouts, VAL t ≥ 2, ≥ 60 % shortable, ≥ 3 fills/week.

## 1,439 — low-of-day mirror
The HOD-break spec mirrored: level = running LOD through bar j; armed when price is ≥ 5 % BELOW the open, 5 bars within
4 % above the LOD, rv in band; resting SELL stop-limit at LOD − $0.01, limit = LOD × 0.9985, fill at the NBBO bid at the
first print ≤ trigger inside the break bar, stop = consolidation high, target 2 R, cover 15:55. Universe = the HOD
universe ∩ shortable. Data: `research/bf_zero/bars_sip.db` minute bars for the levels, SIP ticks fetched for the break
bars (new). Pass: fill mean net R ≥ +0.10 both holdouts, VAL t ≥ 2, ex-top-5 % > 0, coverage ≥ 80 %, gap ≤ 5 pp, ≥ 3
fills/week.

## 1,440 — stop distance
Stop = consolidation low, but floored at 0.8 % of price and capped at 3 % (two variants: floor only; floor + cap). R and
qty recomputed. Pass: ΔR ≥ +0.05 both holdouts, VAL t ≥ 2.

## 1,441 — prior-day-high break, same order
Level = the prior session's high; same arming conditions with the level substituted; same order. Pass: as 1,439's.

## 1,442 — tape-triggered override
Every E1 fill re-priced two ways, paired: (a) broker rule — fill at the ask at the first ROUND-LOT (≥ 100 sh) print ≥
trigger; (b) override — fill at the ask 300 ms after the first print of ANY size ≥ trigger; both capped at the limit,
no fill if the ask is above it. Report the fill rate and mean R of each and the split odd-lot-led vs round-lot-led
crosses. Pass to ship the 1-second override: (b) − (a) ≥ +0.03 R per fill on both holdouts with no fill-rate loss.

## Not allowed
Reordering cells to chase a number; changing E1 or these definitions after a number exists; reading TEST before the
VAL bar; more than one heavy job while the trader runs.
