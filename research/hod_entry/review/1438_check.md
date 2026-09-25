# Adversarial check — cell 1,438 (causal arming), 2026-09-25
Read-only rebuild; scripts + outputs in the session scratchpad (`chk1438.py`, `chk1427.py`, `decomp.py`).
No research file modified. **Verdict: the negative STANDS. The rule is not rescued by any live-knowable cut.**
## 1. Level
* All 9,911 fills: arming re-run from the chosen bars reproduces level and stop 9,911 / 9,911.
* The 7,983 fills on SIP bars match the bars_sip.db running max of closed highs before the fill minute exactly (0 below, 0 above).
  1,928 fills use cache.db because bars_sip.db has no bars for that symbol-day. Their cache.db completeness is P50 0.99 / P10 0.64,
  so a few may carry a stale level. That cohort is the POSITIVE one (+0.155 R), so any error there flatters the rule.
* Random 300: 244 match SIP, 0 disagree, 56 have no SIP bars. cache.db alone agrees with the used level on 159; 94 sit below SIP.
* 1,427 levels, rebuilt independently (TRAIN-H2+VAL): fills 2,576 / 2,608 match true HOD. Of 5,590 no-fills, 3,632 sat BELOW the
  true HOD, all equal to the cache.db max of 1–5 bars. In 2,865 of those 3,632 the "crossing" print was itself below the true HOD:
  1,427 was not even looking at a break. 20 examples: `review/1438_check_examples.txt`. For example:
  INBX 2026-03-02 11:14, level 72.48 (1 cache bar) vs true 76.98, cross print 76.98, ask 77.65; LITE 2026-01-09 10:00, 346.83 vs 360.17,
  print 359.85; CPS 2025-10-20 13:25, 32.77 vs 34.61, print 34.65, ask 34.71.
## 2. Fill rule (all 9,911; random 200 identical)
* fill == prevailing NBBO ask at the first print ≥ trigger: 9,911/9,911. ask ≤ limit: 9,911. Print after the arming-bar close: 9,911.
  No earlier crossing print in the window: 0 violations. Quote age at the print: P50 3 ms, P90 4.4 s, 72 fills > 60 s.
* The BT is optimistic, not pessimistic. 761 fills (7.7 %) are at an ask BELOW the trigger. Live VECO today is a real example:
  the tape ask was 48.86, the broker filled at 48.93, a 0.28 R gap on R $0.25. 63 % of trigger prints are odd lots (1,442 covered this).
* Cost defect, same in 1,427: the entry is filled AT the ask and then charged a half-spread again (mean 0.10 R).
  Adding it back gives −0.108 (TRAIN-H2) / −0.120 (VAL). Raw R before any cost is +0.020 / +0.015. So the negative is not a cost artefact.
## 3. Denominator
* The 82 % is fills / symbol-days with ≥ 1 armed bar whose NEXT bar's minute high reaches the trigger (TRAIN-H2 4,398/5,205, VAL 5,513/6,715).
  Each day gets every crossing armed bar as a chance.
* 1,427's 31.6 % (2,608/8,248) is fills / B0 signal bars, one break bar each, with sparse cache.db levels. 3,632 no-fills there were
  ask-far-above-a-stale-level, so the two rates are not comparable.
* Live 9/25: 30 ledger arms (29 `LIVE ARMED` since 17:53 UTC); 2 crossed (VECO, CDNA) and both filled, i.e. 2/2 of crossed.
  That is consistent with 82 %. The 28 NO_CROSS arms never enter the BT denominator. Live was also capped: max_concurrent 2 logged
  "cap reached, arm stays tape-only" 76 times and 24 kill-rail blocks. So "29 armed / 2 fills" is not a fill rate.
## 4. Decomposition (1,438 fills; mean net R, day-clustered t; TRAIN-H2 | VAL)
| cohort | TRAIN-H2 | VAL |
|---|---|---|
| both fill (1,427 also filled) | 1,114 +0.242 (t 2.8) | 1,388 +0.276 (t 4.0) |
| B0 day, 1,427 no-fill, 1,427 level wrong | 1,110 −0.794 | 1,523 −0.771 |
| B0 day, 1,427 no-fill, level right | 608 +0.002 | 821 +0.027 |
| not a B0 signal day | 1,559 −0.194 | 1,774 −0.259 |
Overlap reproduced (both +0.242/+0.276; 1,438-only −0.36/−0.39).
* Causal features, TRAIN quintiles read on VAL: distance from open, rv at j, ask distance, time of day, R %, range-to-j, arm index.
  Every bin is negative on both splits. Best bins: R % ≥ 2.41 −0.15/−0.06; dist_open < 5.2 % −0.11/−0.15; ask < 2.5 bps −0.18/−0.12;
  first armed cross −0.18/−0.17. No causal sub-rule isolates the positive cohort. Seven features × 5 bins = 35 report-only cells.
* **Tracked hypothesis (coordinator).** "Tracked" = cache.db holds ≥ 90 % of the minutes before the fill bar.
  Tracked: +0.229 (t 2.8) / +0.263 (t 3.8). Untracked: −0.564 / −0.613. Difference t +16.7 / +20.0.
  At 12/4 slots: 39.0 / 44.6 fills/wk, +0.197 / +0.176. **It clears +0.15 and t ≥ 2 on both, but it is NOT causal:**
  - No live writer: intraday_bars_1min is written only by backtest cache builders (batch_backtest find_big_movers =
    full-day (high−low)/low ≥ 10 % priced on the close; orb_backtest; macd_wave_backtest). The live scanner writes none,
    and scan_results is EMPTY for the whole window, so "qualified before the break" cannot be tested from the DB.
  - Outcome leak: tracked AND full-day range < 10 % gives −0.324 / −0.341; tracked AND range ≥ 10 % gives +0.292 / +0.365.
    Range ≥ 10 % alone: +0.13 / +0.19 against −0.75 / −0.77.
  - The live-knowable analogues are flat or negative. Range through bar j ≥ 10 %: −0.18 / −0.29. Gap ≥ 3 % at the open: −0.25 / −0.13.
  - Residual: within range ≥ 10 %, tracked still beats untracked (+0.29 against −0.24). That likely reflects the builder's other
    close-priced terms. It is untested and non-causal until a live-scanner membership record exists.
## 5. Live defects seen while checking (not the cell)
* ERROR "safety-net SL submit failed after a LIVE fill" (available 0) for VECO and CDNA at 18:17 and 18:32 UTC: fills without a broker stop.
* `logs/hod_live_parity_ledger.csv` holds 20 test rows (ZZZ/VECO/CDNA @ 11.00, 15:05–15:24 ET). Tests write into the production ledger.
## Verdict
The negative stands. Levels and fills reproduce exactly. The BT's biases (ask below the trigger, a stale level on cache-only days)
flatter the rule. The double-charged entry half-spread explains 0.10 R, not the sign: raw R is about 0.
1,427's edge is the tracked / level-correct cohort, and that cohort is selected on the day's outcome through the cache.db build.
A test of real scanner membership needs a point-in-time watchlist log. None exists for 2025-07..2026-05.
