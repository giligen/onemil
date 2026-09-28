# PREREG — cells 1,617–1,622: three NEW frames on the HOD-break population (overnight hold, burst fade, short confirmation)

FROZEN 2026-09-28 17:00 UTC before any number. Programme count: 1,616 → 1,622. Owner 9/28: "I won't let you go till you
find a profitable HOD angle." Rule of this loop: a new MECHANISM per pass, never a variant; PREREG + rebuild + refuters.

## What was seen (disclosed)
Base fills (9,911, cell 1,438): intraday path after the break flat in gross at every entry/exit measured (1,547 cells);
87–91 % dip through the level within 15 min; 55 % trade ≥ 0.25 % under the level inside the fill minute; the 9–13 %
never-retest fills earn +0.6–0.9 R; the 15-minute confirmation (1,487) −0.05 R; the failed-break short with a 1 % stop
and 2 R target −0.53 / −0.68 R; the close→next-open return of these symbol-days by cash-runway bucket +0.01 to +0.28 %
(cell 1,484 report-only, never negative); the overnight new-high leg (1,550) was tail-carried (ex-top-5 % negative).

## Cells
### A. Held break overnight (1,617; report-only variant 1,618)
Population: base fills whose day's CLOSE ≥ the level (the break held; from the Databento daily panel
`research/overnight_high/panel_2024_2026.parquet`, zero-OHLCV rows dropped; the HOD days 2025-07..2026-05 are inside).
1,617: buy at the 16:00 closing auction (MOC) of the break day, sell at the next session's opening auction (MOO);
return = next open / close − 1; costs 5 bps per auction leg; earnings dates excluded where the panel or Alpaca
calendar gives them (state coverage), a −30 %..+30 % raw-price check (splits) per night. 1,618 (report-only): the same
for base fills whose close is BELOW the level (the failed breaks) and for the whole in-play scanner day universe on the
same nights (the placebo: is it the break or the name).
Report per holdout: n nights, mean net bps, day-clustered t (the night), ex-top-5 % / ex-top-1 %, winner-capped +10 %,
green-night share, per-month, the placebo margin (held-break minus the universe on the same nights).

### B. Burst fade (1,619; report-only variant 1,620)
Population: base fills with the tape window of the fill minute (`sip_cache_1481/`, `sip_cache_1480/`; the fill bar
and the 15 minutes after, as the retest cells). 1,619: at the base fill instant a SELL LIMIT rests at level × 1.0015
(the same 15 bps as the base's buy cap — we are the offer the chasers lift); filled at the limit at the first print
STRICTLY ABOVE it within the fill minute and the next 2 minutes (through-print rule); no print → no trade (counted).
Once short: cover at a resting BID at level − $0.01 filled at the first print strictly below it (the retest);
stop = level × 1.0075 (a print ≥ stop → stopped, stop-limit standard cost on a short = buy at the ask + the 12 % tail
at 94 / 76 bps); 15:55 cover at the ask; shortable flag and SSR excluded (`research/fuckup_audit/O_halt/PASSIVE/
borrow_flags.csv`), borrow 3 %/yr pro rata; halts → the position is marked at the reopen print (counted).
Units: R_f = stop − entry (≈ 0.6 % of price — the rail: median R_f ≥ 0.5 % or NOT SHIPPABLE), net R_f and net % of
price, paired against the base long on the same fills as the mirror check. 1,620 (report-only): the same with the
cover at level − 0.20 %.
Report: fill share, cover share within 15 min, stop share, runner losses (the never-retest cohort's cost to the short),
mean net, day-clustered t, ex-top-5 %, fills/week at 12/4.

### C. Short-window confirmation (1,621: 3 minutes; 1,622: 5 minutes)
As cell 1,487 with the window W ∈ {3, 5} RTH minutes instead of 15: for base fills still open at the end of minute
fill_min + W with NO bar low ≤ level − $0.01 in (fill_min, fill_min + W], enter at the ask at the open of minute
fill_min + W + 1 (ask = bar open + the fill's half_entry), stop = level − $0.01, target entry + 2 R″, 15:55 at the bid;
costs as 1,487; R″ < 0.5 % of price reported but excluded from the primary book. Report: eligible share, runners lost
(base target-hitters exited before W), mean net R″, t, ex-top-5 %, fills/week, the base outcome of the eligible
cohort (calibration), paired vs the base on the same fills.

## Pass bar (frozen; VAL, per cell)
A: mean net ≥ +8 bps/night, day-clustered t ≥ 2.5, ex-top-5 % > 0, winner-capped positive, ≥ 3 nights/week, placebo
margin ≥ +5 bps t ≥ 2, ≥ 4 of 6 months positive, TRAIN-H2 same sign t ≥ 1.
B: mean net R_f ≥ +0.15 and ≥ +0.10 % of price, t ≥ 2.5, ex-top-5 % > 0, ≥ 3 fills/week, TRAIN-H2 same sign, median
R_f ≥ 0.5 % of price.
C: as 1,487 (mean net R″ ≥ +0.15, t ≥ 2.5, ex-top-5 % > 0, ≥ 3 fills/week, count-matched null ≥ 99, TRAIN-H2 same sign).
TEST: none exists for A–C beyond the sealed HOD TEST, which stays sealed.

## Independent check and consequences
Rebuild from the prose per cell (A: nightly set Jaccard ≥ 0.99, bps within 1; B: fill-set Jaccard ≥ 0.98, ≥ 99 % within
0.01 R_f; C: ≥ 99 % within 0.01 R). Refuters: A — price scale (raw close/open), earnings and halts, the placebo, tails;
B — obtainability of a resting offer inside the burst (queue: the share of prints above the limit and their size),
the short's stop in a fast market, borrow/SSR, the runner tail; C — look-ahead of the window, the run-up cost, tails.
PASS on A → a MOC/MOO leg in the HOD engine (the EOD-mode work provides the order types) on the dry ledger first;
PASS on B → a short book proposal for the owner (his manual shorts share the account); PASS on C → `confirm_minutes`
in the HOD engine for the $50 run. FAIL → each frame closes with its numbers; the loop continues with the next mechanism.

## Not allowed
Moving W, the offer/cover levels, the stop, or the auction legs after a number; selecting among A/B/C variants on VAL.
