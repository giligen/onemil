# The real losers, trade by trade (2026-09-17 night, done by hand on the live books)

Owner: "You didn't deep dive on the losing trades/days." Right. This file is that dive, on the only losers that are real:
the 116 closed live ORB trades and the 53 closed live bull-flag trades in `data/trades.db` (real fills, real stops, real
quotes), with the 1-minute bars around each one (`live_loser_paths.py` → `live_loser_paths.csv`) and counterfactuals
re-walked on those bars (`live_counterfactuals.py`, `live_followthrough.py`, `orb_timestop_validation.py`).

## Bull flag (53 real trades, −10.4R, 34% winners)
- The book loses on stops: 21 plain stops −$13.5K at **−1.30R each** (stop slippage makes a 1R stop a 1.3R loss),
  **median 6 minutes after the fill**. Winners: 12 trail stops +1.30R (median 4 min), 9 holds to the close +0.59R.
- **75% of the 40 stop-outs were wicks** (the stop bar closed back above the stop); 72% were back above the fill within
  the hour; 60% had +0.5R on the table first, 42% had +1R. And yet: re-walking those trades with a CLOSE-based stop or
  a 0.5% buffered stop makes the book WORSE (−10.4R → −20R / −23R). The wick trades that bounce go on to lose anyway;
  the stop shape is not the money (the owner's stop-at-support signature is real in the counts, not in the P&L).
- **What separates real winners from real losers at the trade level**: follow-through volume in the fill minute
  (winners 1.8× the prior 5 bars, losers 0.9×) and the gap (winners +4.6% median, losers −0.4%). On the 53:
  vol ≥ 1.5× +7.2R (20 trades, 55% WR) vs −17.6R (33, 21% WR); gap ≥ 2% +3.2R vs −13.7R; both +0.77R/trade (12).
- **Validation on the honest raw cache (885 detections, 2025-01..2026-08)**: vol ≥ 1.5× beats < 1.5× on TRAIN
  (+0.075 vs −0.066) and TEST (+0.140 vs −0.189) and REVERSES on VAL (−0.072 vs −0.049); gap ≥ 2% likewise
  (+0.060 / −0.114 / −0.089). The exit version (sell a quiet fill at the next open) is worse in every period
  (−0.014 → −0.102 TRAIN). **Verdict: a two-thirds truth, not a rule.** The 53 live trades were the 2025+2026-summer
  regime talking. Not adopted.
- Chase is not the problem: winners were filled further above plan (64 bps) than losers (32 bps).

## ORB (116 real trades, −10.7R, 38% winners)
- The book loses on 55 stops (**−0.9R each, 38 minutes after a 09:35 fill**) and wins on the 24 trades that hold to
  the 15:45 close (+0.87R) and 12 lock-stops (+0.34R); the 38 touchgo exits net +0.12R.
- **Nothing at entry separates the losers**: gap 6.5 vs 6.7%, prior-day range 9.4 vs 11.1%, 5-min range 3.5 vs 3.4%,
  breakout-bar volume 1.1× vs 1.2×, SPY flat both. The losers' whole story is the PATH: max favourable excursion
  **0.35R, reached 3 minutes after the fill**, then a slow bleed to the stop; winners' MFE 1.32R at 52 minutes.
  A trade that has not made +0.25R ten minutes in is a loser 3 times out of 4.
- Close-based / buffered stops: −10.7 → −8.3 / −7.5R (small; the wicks that recover are the minority here — 63% wick,
  41% recover within the hour).
- **The one live-computable lever with size: a 10-minute time stop.** Exit at fill+10 min if below +0.25R: on the
  real trades −13.4R → −2.4R (50 of 113 cut early). ONE post-hoc rule on one sample; its validation on the 7,402
  honest fills (Jan-25..Sep-26, entered-inclusive rebuild), 9 declared cells (5/10/15 min × 0/0.25/0.5R), per split
  and on the B+ book, is `orb_timestop_validation.md` (running at the time of writing; primary cell T10/0.25).

## What this dive changes
- Stop placement: measured on real trades, in both books, it is not the lever. Closed.
- Bull flag: the follow-through/gap separator is regime-dependent; it goes on the list as a hypothesis for the next
  live quarter, not as a filter.
- ORB: if the time stop holds on the honest fills in every split, it is a change to a live, parity-audited book —
  the first deployable finding of this program. If it does not, the ORB loss is what it looks like: half the breakouts
  fail slowly and nothing at 09:35 tells them apart.
