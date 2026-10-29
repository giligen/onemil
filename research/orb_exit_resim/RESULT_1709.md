# RESULT 1,709 (a mechanism, b money) -- ORB exits re-simulated at the live / tape fill (built 10/29; script resim_1709.py, tables_1709{a,b}.md, per-fill resim_1709a.csv / resim_1709b.csv)
NOT YET REBUILT independently (PREREG rebuild step pending) -> not reportable to the owner until the Sonnet rebuild agrees (stop share 2 pp, mean R 0.02, exit reason 95 %).
## Implementation (deviations from the PREREG text are flagged D)
Rules from trading/orb_winner_stack.floored_stop + orb_touchgo_filter (Rule M 0.5 close-pos at fill-bar close; Rule D 0.75R revert -> -0.5R), lock 1.75R/0.5R in range_size units, stop checked before arming within a bar,
stops fill at min(stop, bar open) x (1-35 bps), tag/force exits at the bar close, no cost on them (cost10 sensitivity = 10 bps). Entry bar stops only if its CLOSE <= stop ('ebstop' = its LOW can).
D1 force close = 15:45 ET (live ledger 15:45:01, orb.yaml, BT); PREREG says 15:55 -> 'fc1555' column. D2 ATR floor only where pattern_data stores atr14 (8 of 123 live rows; live floor began 8/25; ledger stop == sim stop on 100 % of
the 123, so the stored-ATR rule reproduces the live stops). Replay rows have no stored ATR -> stop = range_low for ALL 520 (floor share 0 %). D3 add-on at 1R and the 40 %@3R scale-out (both ON in orb.yaml) are NOT simulated.
D4 bars: bars_sip.db first; for b also research/day_breadth/y2024/bars.db (read-only, 2024H2 existing store); nothing fetched. R for b = pnl/375 (cell 1,708 convention, true shares = _rp_position/entry).
## 1,709a (n 123 live, 5/19-9/23, all with bars_sip bars; the 14 paper fills after 9/28 excluded, 0 fallback)
| book | stop share | mean R [day-boot 95 % CI] | lock | tag | force |
|---|---|---|---|---|---|
| actual ledger | 39.0 % | -0.109 [-0.306,+0.122] | 9.8 | 30.9 | 20.3 |
| (i) BT entry rh x 1.003, main | 34.1 % | -0.135 [-0.369,+0.129] | 9.8 | 30.1 | 26.0 |
| (ii) live fill, main | 35.0 % | -0.111 [-0.360,+0.173] | 9.8 | 29.3 | 26.0 |
| (ii) ebstop / fc1555 / cost10 | 35.8 / 35.8 / 35.0 % | -0.123 / -0.120 / -0.129 | | | |
Ledger excess over the published BT 28.4 %: +10.6 pp (123-fill basis; 41.6 % / +13 pp on all 137). (ii)-(i) = +0.8 pp = 8 % of that excess. The re-simulated stop share at the BT entry is already 34 % (not 28 %).
Confusion (rows ledger, cols sim ii): force 23/0/0/2 (fc/lock/stop/tag); lock 0/11/0/1; stop 1/0/40/7; tag_bb 8/1/3/26. Agreement 81.3 % (100/123). R is 0.002 R from the ledger mean (-0.111 vs -0.109).
Hand-check (ledger vs sim ii, exit UTC / price): WAY 131 ledger tag_bb 13:47:16 @19.997 vs sim force_close (the engine saw the fill 468 s late, so its Rule M fired late: not reproducible from bars);
LMRI 134 ledger force_close 19:45:01 @7.010 vs sim tag_bb 13:39 (live Rule M did not fire); YANG 139 stop_loss 13:54:17 @29.493 vs sim stop 13:54 @29.387 (sim slip 35 bps vs actual 2 bps);
TRT 148 lock_stop 18:09:44 @16.288 vs sim lock 18:09 @15.949; RUN 257 stop_loss_market_fallback 17:46:41 @15.255 vs sim stop 17:45 @15.236. Stops, times and reasons match; prices differ by the slip model.
## 1,709b (replay population, filled rows at delay 0 s / 30 s; bars coverage by regime)
Coverage (delay 0): in 2025H1 42/42, in 2025H2+2026 172/172, oor 2024H2 34/34 (y2024 store), oor 2023-01..24H1 0/53 (NO local minute bars: orb_2023/bars.db is a 0-byte file, no fetch allowed).
Rail: 2023-24H1 VOID (0 %); pooled out-of-regime 34/87 = 39 % < 80 % -> VOID. In-regime and 2024H2 pass the rail (miss gap 0 pp).
| slice (delay 0, main) | n | mean R375 [day-boot CI] | ex-top5 % | +3R cap | stop share | fills/wk |
|---|---|---|---|---|---|---|
| out-of-regime 2024H2 (only covered oor) | 34 | +0.133 [-0.116,+0.494] | -0.065 | +0.087 | 29.4 % | 1.47 |
| out-of-regime 2023-01..24H1 | 0 | uncovered | | | | 0.73 (replay-filled) |
| in 2025H1 (TRAIN half 1) | 42 | +0.257 [+0.007,+0.528] | +0.079 | +0.257 | 35.7 % | 1.78 |
| in 2025H2+2026 (TRAIN half 2) | 172 | +0.112 [-0.015,+0.260] | -0.034 | +0.075 | 27.3 % | 2.70 |
| in-regime pooled | 214 | +0.140 [+0.024,+0.275] | -0.006 | +0.111 | 29.0 % | 2.39 |
Delay 30 s: oor 2024H2 +0.111 [-0.184,+0.561] n 24; in pooled +0.110 [-0.011,+0.248] n 154. Sensitivities (fc1555 / cost10 / ebstop) move the pooled in-regime mean by at most 0.005 and 2024H2 by 0.03.
Reference: the 1,708 BT-exit pnl on the same 248 covered rows = +0.135 R vs re-sim +0.139 -> re-simulating exits changes nothing; the tape entry, not the exit rule, sets the edge.
## Pass bar, line by line
H1: (ii)-(i) explains >= 50 % of the stop excess -> FAIL (8 %). (ii) reproduces the actual reason on >= 70 % -> PASS (81.3 %). H1 FAILS.
H2 (out-of-regime, delay 0): mean R >= +0.10 -> point +0.133 on 2024H2 only, PASS on the point estimate but the slice is 39 % of the oor population (rail VOID); CI lower > 0 -> FAIL (-0.116);
same sign in both TRAIN halves -> PASS (+0.257, +0.112); ex-top-5 % > 0 -> FAIL (-0.065; in-regime -0.006); >= 3 replay-filled fills/week -> FAIL (0.73-1.47 oor, 2.39 in-regime pooled, 2.70 best half).
H2 FAILS (3 criteria fail + availability VOID) -> per the PREREG decision: ORB is shelved 2026-11-13 (owner call; this file is a build output, not yet independently rebuilt).
## Adversarial caveats
1. Out-of-regime evidence is 34 fills of 2024H2 (CI [-0.12,+0.49]); the 2023-01..24H1 half has no bars, so a void rail means H2 could not pass even with a good number; this is a coverage failure, not a measured null for 2023-24H1.
2. Tail dependence: ex-top-5 % is <= 0 in every pooled slice and the +3R cap removes 0.03-0.05 R; the edge is a few winners.
3. Add-on and scale-out (ON live) not simulated; the ledger R includes them, so a vs ledger is like-for-like only on the stop/lock/tag classes. The live engine's tick-level Rule M timing and late fill detection are not reproducible from minute bars (confusion: 8 tag_bb -> sim force_close).
4. Exit costs: only stops carry 35 bps; tag/force exits are free (live tag_bb fills ~69 bps worse than target per orb_machine_rules) -> the b means are optimistic; cost10 shows only -0.005 but the tag slip is larger.
5. Replay rows use range_low stops (no stored ATR) and the BT's own range_high; the BT books' stop floor status is unknown. R denominator 375 is a fixed convention, not each fill's own risk (Rown column in the CSV).
6. Daily-bar-only ATR was deliberately not used (PREREG: stored value or range_low). Cell count: +2 (a, b) + 3 sensitivity reads.
