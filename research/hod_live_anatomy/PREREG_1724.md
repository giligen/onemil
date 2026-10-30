# PREREG — cell 1,724: time-aware exits on the live HOD book (owner 2026-10-30: "improve the exit when we are 2h before
closing and we likely won't hit 2R")

Frozen before any bar is read. Supersedes item 2 of PREREG_1723 (the data turns out to be free: Alpaca 1-min bars).

## Mechanism
A breakout that has not reached 2 R by early afternoon has lost its momentum; the live rule holds it to the stop or the
15:55 flat. The 53 EOD exits closed at +0.36 R (p90 +1.23 R): open profit is being carried into the close, where it is
exposed to the afternoon fade and sold at the close's worst liquidity. Taking or protecting it earlier may keep more of it.

## Data
The 231 R-valid fills in `fills_1721.csv` (fill time, fill price, stop, target, shares, realized exit). Alpaca 1-min
regular-session bars (SIP feed, free) for each (symbol, trade_date) from fill minute to 16:00 ET, cached atomically
under `research/hod_live_anatomy/cache/bars_1724/`. Completeness gate: report symbol-days with < 90 % of the expected
minutes; a variant is VOID if coverage < 80 % or the missing share differs by > 5 pp between winners and losers.

## Live rule replay (sanity)
Replay the live rule on the bars (stop at the bid proxy = bar low ≤ stop → fill at stop − measured 12 bps; target: bar
high ≥ target → fill at target; else 15:55 close). Report the per-fill agreement with the realized ledger exit (reason
and R within ±0.15 R). If agreement < 80 % the replay is not trusted and the cell stops there.

## Variants (ONLY these four; parameters fixed here)
V1 **time stop 14:00**: at the 14:00 ET bar close, exit every open position at the next bar's open − 12 bps.
V2 **take-the-profit 14:00**: at 14:00, exit only positions with open P&L ≥ 0 R; losers keep stop/EOD as live.
V3 **protect 14:00**: at 14:00, positions with open P&L ≥ +0.5 R move the stop to break-even (fill − 12 bps on touch) and
keep the 2 R target; others unchanged.
V4 **1.5 R target** from the start (same stop), the 1,723 item.

## Amendment 1 (owner 2026-10-30, frozen before any variant number was read): "squeeze the exit — the closer to the
bell, the less you should need"
V5 **decaying target**: the target in R is a function of ET clock time: 2.0 R until 12:00, then LINEAR to 0.5 R at
15:55 (floor 0.5 R); exit when a bar's high ≥ fill + target(t) × R (fill at that target); stop unchanged; 15:55 flat.
V6 **decaying target, earlier and deeper**: 2.0 R until 11:00, then linear to 0.25 R at 15:55. Two schedules only, no
sweep; the same pass bar. If a decaying target and a 14:00 variant both pass, the higher ex-top-5 % Δ ships.

## Scoring (paired, on the same fills)
Δ = variant R − live-replay R per fill. Report pooled n, Δ mean, trade-clustered t, Δ ex-top-5 % (the paired-lift tail
check), Δ winner-capped (+2 R), and the A/B halves (alternating trade dates). Also: number of fills the variant touches
(only positions open at 14:00 for V1–V3).
**Pass:** Δ ≥ +0.10 R/trade in BOTH halves, pooled t ≥ 2, Δ ex-top-5 % > 0, coverage gate met. One variant may pass; if
more than one passes, the one with the higher ex-top-5 % Δ ships. Fail all → report Δ with its CI and the MDE at this n;
no parameter sweeps.

## Ship path (if a variant passes)
Spec for `trading/hod_break_engine.py` as a flagged exit rule (default OFF, dry/paper first), weekend rehearsal, Monday
boot. Nothing changes in the engine inside this cell.
