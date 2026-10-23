# RESULT 1,700 — latency -> outcome (built 10/23 per PREREG.md; script build_1700_1701.py, log run_1700_1701.log, table fills_1700.csv)
Data: trades.db strategy='orb', filled rows with pnl: LIVE 123 fills (53 days, 5/19-9/23, Σpnl -$5,281 = ledger, R mean -0.109, sd 1.19)
+ PAPER 14 (9/29-10/22). L_brk = broker filled_at - created_at (all 137 order ids fetched read-only, 0 lost). L_eng = engine
SUBMIT LATENCY log where present (NONE of the live fills; logs start 9/28) else broker created_at - 09:35:00 (offset). drift = (fill -
entry_quote_ask)/ask. R = pnl/(filled_qty x (fill - stop_loss_price)). CI = day-clustered bootstrap, 2000 resamples.
## Pre-committed tests, LIVE n=123
(a) Spearman(L_brk,R)=+0.282 CI[0.113,0.429]; Spearman(L_brk,drift)=+0.472 CI[0.312,0.611]. Spearman(L_eng,R)=-0.136 CI[-0.296,0.041];
    (L_eng,drift)=+0.002 CI[-0.19,0.17]. Pooled 137: L_brk,R +0.170 CI[-0.022,0.348].
(b) R by L_brk tercile (edges 0-34 s | 34-226 s | 235-3374 s): T1 -0.329 (sem .145), T2 -0.264 (.245), T3 +0.267 (.138), n 41 each.
    T1-T3 = -0.596 R CI[-1.00,-0.16]. Pooled 137: -0.316 CI[-0.74,0.12].
(c) drift <=30 bps: n 43, R -0.118 | >30 bps: n 80, R -0.103 | diff -0.015 CI[-0.42,0.48]. Only 35 % of fills are inside the BT's 30 bps
    entry model; median drift 49.7 bps.
MDE (PREREG) 0.36 R at 80 % power iid; the day-clustered (c) CI half-width 0.45 R is wider, so (c) is NOT RESOLVED, not "no effect".
## Read
The sign is the OPPOSITE of "slower fill hurts": longer created->filled goes with better R and with MORE drift (rho +0.47).
## Caveats (adversarial)
1. L_brk is NOT latency. ORB orders are preplaced at ~09:35:02 and rest until the breakout fires; created->filled (median 83 s, max
   56 min) is mostly waiting for price. Quick fills = orders marketable immediately (price already above the level) = the fills that
   chase; slow fills = true resting breakouts. (a)/(b) therefore measure "marketable-at-submit vs resting", a selection variable,
   not execution speed. Do not read it as a latency effect either way.
2. L_eng for live is an offset proxy (created - 09:35:00; median 4.3 s, p75 41 s, max 421 s); the long tail is a later decision time,
   not a slow pipe. No live SUBMIT LATENCY lines exist (hence the 26-49 s claim is unverified here). Paper L_eng is 0.7-8.6 s and
   includes Alpaca paper's ~7 s OMS queue (created is stamped before the queue? -> not checked).
3. entry_quote_ask timestamp is not documented in the schema (assumed decision-time quote); drift vs range_high is in the CSV
   (drift_rh_bps) but not tested. R uses fill - stop (planned risk would differ ~ a few %); 1 fill has |R|>5 (kept).
4. Tercile T3 is carried by few big winners (sem .14, no tail test run); 4 sequential cells here, ~12 tests, no multiplicity
   correction; programme count 1,700+. 53 days clustering handled, but symbols repeat across days (not clustered).
5. Not independently rebuilt yet (second agent: use fills_1700.csv, then rebuild from trades.db + broker_ts.csv).
