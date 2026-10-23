# RESULT 1,702 — ORB exit decomposition (built 10/23 per PREREG.md; script build_1702_1706.py, log run_1702_1706.log, per-fill fills_1702.csv)
Data: trades.db strategy='orb', pnl not null: 137 fills (LIVE 123 5/19-9/23 + PAPER 14 9/29-10/22), Σ$ -6,208 (live -5,281 = ledger). R = pnl/(qty x (fill-stop)).
Entry time = broker filled_at (broker_ts.csv); DB filled_at differs for 1 row only (reconcile, up to 4.5 h late); exit time = DB exited_at.
MFE/MAE in R of (fill - stop) from 1-min bars_sip.db: STRICT = bars after the entry minute through the exit minute (main); "incl" adds the entry bar.
Bars coverage: 123/137. The 14 missing are ALL paper fills after 9/28 (bars_sip.db ends 9/28; 9 days) -> MFE/MAE on live n=123 only; no fallback
to Alpaca bars (14 symbol-days ≤ 30 allowed, but not fetched: budget; MFE/MAE for paper not reported). CI = day-clustered bootstrap, 2000.
## By exit_reason (n, Σ$, meanR [CI], MFE/MAE strict, minutes fill->exit median [p25,p75,p90])
force_close   29  +6,304  +0.84 [0.24,1.53]  MFE 1.85 MAE 0.47   366 [359,369,370]
lock_stop     13  +1,189  +0.45 [0.37,0.52]  MFE 2.85 MAE 0.13    45 [33,151,263]
tag_bb        38  +1,325  +0.14 [-0.06,0.41] MFE 0.42 MAE 0.30   1.9 [1.3,10.8,60]
stop_loss     53 -12,616  -1.00 [-1.04,-0.96] MFE 0.53 MAE 1.09   32 [13,50,178]
stop_mkt_fb    4  -2,410  -1.14 [-1.32,-1.04] MFE 0.17 MAE 1.19   18 [11,78,176]
ALL          137  -6,208  -0.16 [-0.34,0.07]  MFE 0.98 MAE 0.63   38 [6,245,366]
Stop-type exits = 57 (41.6 % of fills; BT book 5/19-9/23 n 134: 28.4 % 'stop'; BT 2025-26 31.6 %). The 4 market-fallback stops lose -0.14 R beyond the stop.
## Pre-committed question: stop exits within 5 min of entry
4 of 57 stops (7 %), 2.9 % of all fills, within 5 min; 2 within 2 min, 0 within 1 min (Σ -$285, meanR -0.90). Median stop takes 32 min. All 4 are resting-cohort
(2 live, 2 paper). So the entry-bar / immediate-stop (spread or latency) mechanism does NOT explain the stop rate: stops are slow. NOT RESOLVED beyond that:
the BT has no exit timestamps, so "more than the BT implies" cannot be tested; only the stop SHARE can be compared (41.6 % live vs 28.4 % BT, n 137 vs 134;
a 13pp excess, iid SE ~5.9pp => ~2.2 sigma, day clustering not applied).
Other: 29 % of fills reach MFE >= 1R (strict); 14 % of stop exits had MFE >= 1R first (round trips, which lock_arm 1.75R/lock 0.5R do not catch).
## ATR-floor counterfactual: SKIPPED. pattern_data holds only range_high/low/size, composite, quintile, adaptive_mult, lock_arm_at_r, lock_stop_r; no ATR or
stop-derivation field (stop = range_low). Rebuilding ATR14 point-in-time from daily_bars is a separate cell; no knob change follows from this one anyway.
## Caveats (adversarial)
1. Exit decomposition is by OUTCOME class; exit_reason is endogenous (stops have R≈-1 and winners long holds by construction) — per-class means are not tests of any rule.
2. MFE/MAE use 1-min H/L: intrabar order unknown; strict variant drops the entry minute (understates MAE for fast stops; incl variant differs <=0.04 R except lock_stop MAE 0.13->0.41).
3. force_close +0.84 R [0.24,1.53] is the tail-carried class (29 fills, 4-5 big winners; no ex-top test run here). tag_bb is a bar-close exit, median 1.9 min: its
   fills include marketable-at-submit chasers (see 1,706).
4. Time-to-exit uses engine-recorded exited_at (could lag the broker fill by seconds-minutes). 5-min bucket n=4 is tiny: the share CI is wide (~2-17 %).
5. Paper (14) has Alpaca's ~7 s OMS queue; excluded from MFE/MAE but included in counts/R. Not independently rebuilt yet.
