# RESULT 1,701 — spread -> outcome (built 10/23 per PREREG.md; same script/table as 1,700; log run_1700_1701.log)
Data: same 123 LIVE fills (5/19-9/23) + 14 paper. spread_bps = entry_quote_spread / mid x 1e4 from the entry_quote_bid/ask stored
on the trades row (engine quote at order decision; NOT research/exec_cost NBBO, whose per_trade.csv covers 2025-26 BT trades).
exit_slip_bps = (exit_trigger_price - exit_price)/trigger (stop price if stop_loss and no trigger recorded). CI day-clustered bootstrap.
## Buckets, LIVE (bps = mean)
| spread | n | mean R (sem) | entry drift | exit slip | RT cost | Σ$ |
| <50 | 81 | -0.05 (.15) CI[-0.31,0.29] | 54 | -30 | 25 | -1,456 |
| 50-150 | 40 | -0.20 (.13) CI[-0.46,0.03] | 101 | +2 | 102 | -3,270 |
| 150-300 | 2 | -0.75 (.25) | 225 | -98 | 127 | -556 |
Pooled with paper (137): <50 n88 R -0.07 | 50-150 n46 R -0.30 CI[-0.55,-0.07] | 150-300 n3 R -0.89. Median spread 32 bps.
Spearman(spread,R)=-0.035 CI[-0.23,0.15]; Spearman(spread,drift)=+0.264 CI[0.11,0.42] (wider spread -> more entry drift).
Share of loss from names >=150 bps: 2 fills (2 %), -$556 = 3 % of the book's total losing dollars (-$16,298) and 11 % of the net -$5,281.
Half-spread at entry = 0.065 R mean (0.26 R for the 2 wide fills).
## Read
The loss is NOT in the wide-spread tail: 150+ bps is 2 fills. 50-150 bps (33 % of fills) holds -$3,270 (62 % of the net -$5,281) at
-0.20 R and 100 bps drift, vs <50 bps at -0.05 R; the R difference (0.15) is far inside the MDE (0.36 R), so the bucket gap is
suggestive, unresolved. This cell never tightens the gate (orb_spread_gate_verdict.md): 50-150 is inside the 150 bps gate.
## Caveats (adversarial)
1. Spread and drift are mechanically linked: a wide quote inflates (fill - ask) only if the fill crosses it; the drift column mixes
   spread, the resting-order wait and the breakout run (see RESULT_1700 caveat 1) — "entry drift" is not pure slippage.
2. Exit slip sign/meaning mixes exit types (tag_bb, lock_stop use different trigger prices; the -30 bps in <50 is exits ABOVE the
   recorded trigger, i.e. trigger definition artifacts, not a gain). RT_bps column is therefore indicative only.
3. entry_quote_spread is the stored row quote; 150-300 bucket only passes the 150 gate if the gate saw a different (earlier) quote —
   gate-time quote not joined (spread-gate log lines not parsed). n=2 live in that bucket: nothing is estimable.
4. Day-clustered CIs; symbol repetition, tail dependence (ex-top-5 %), and placebo not run. ~10 tests, uncorrected.
5. Not independently rebuilt yet; paper-endpoint ~7 s queue affects paper drift.
