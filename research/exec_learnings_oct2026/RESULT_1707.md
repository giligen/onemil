# RESULT 1,707 - marketable-at-submit rule on the ORB tick-replay population (10/23)
Verdict: **FAIL** (PASS bar applied literally; amendment buckets descriptive). Script `score_1707.py`, per-fill table `fills_1707.csv`.
Method: replay engine of cell 1,426 (`research/orb_latency_bt/replay.py::replay_one`, chase rule) at submit = 09:35:05 ET
(delay 5 s, the same instant for every name - the engine's per-name decision time is not recoverable from the ticks).
MKT = last XNAS print at/before 09:35:05 >= range_high (trigger); RESTING = otherwise. R = replay P&L / 375, replay-filled fills only.
## Coverage (no new data)
638 BT-entered fills; 21 no tick parquet (uncovered, 3.3 %); 617 replayed: 270 filled, 140 skipped by chase guard, 207 never reached trigger.
Availability rail of 1,426: usable 96.4 %, winner/loser gap 0.3 pp. **223/617 have NO print in the 09:34:50-09:35:05 lookback**
(last unknown -> counted RESTING). 7 fills have t* < submit but last < trigger (discordant, counted RESTING).
## Cohorts (day-clustered 95 % CI | ex-top-5 % | +3R cap | stop share | wrapper share)
| segment | cohort | n | mean R | CI | ex5% | cap3 | stop | wrap |
|---|---|---|---|---|---|---|---|---|
| in-regime 2025-01..26-09 | MKT | 5 | -0.138 | [-0.234,-0.042] | -0.161 | -0.138 | 40% | 80% |
| in-regime | RESTING | 185 | +0.138 | [+0.021,+0.254] | +0.005 | +0.118 | 32% | 58% |
| 2025H1 | RESTING (MKT n 0) | 35 | +0.264 | [+0.014,+0.514] | +0.144 | +0.264 | 46% | 37% |
| 2025H2+2026 | MKT / RESTING | 5 / 150 | -0.138 / +0.109 | [-0.022,+0.239] (REST) | -0.024 (REST) | +0.084 | 29% | 63% |
| out 2023-24 | MKT | 4 | +0.057 | [-0.471,+0.585] | -0.252 | +0.057 | 25% | 25% |
| out 2023-24 | RESTING | 76 | -0.001 | [-0.131,+0.130] | -0.119 | -0.001 | 51% | 4% |
| out 2023-01..24H1 / 24H2 | RESTING | 46 / 30 | -0.091 / +0.137 | [-0.185,+0.003] / [-0.150,+0.424] | -0.154 / -0.042 | | 57% / 43% | |
| all | RESTING | 261 | +0.098 | [+0.006,+0.189] | -0.031 | +0.083 | 38% | 42% |
Sensitivity +10 / +30 bps above range_high: MKT n = 0 at both (the engine's 30 bps chase guard already skips every such fill; the
140 guard-skips are the marketable-and-too-far cohort). Cohort split on all 454 covered BT-entered in-regime fills with BT P&L (not replay-filled):
MKT n 37 -0.024 R (CI [-0.117,+0.069]), RESTING n 417 +0.122 R (ex5% +0.019); out of regime MKT n 15 +0.160 (CI [-0.32,+0.64]), RESTING n 148 +0.006.
## PASS-bar criteria
1. RESTING >= +0.10 R, same sign in both TRAIN halves: in-regime +0.138; 2025H1 +0.264, 2025H2+26 +0.109 -> **pass**.
2. ... AND out of regime: RESTING -0.001 R (< +0.10) -> **FAIL**.
3. MKT <= 0 in both regimes: in -0.138 (n 5) ok; out +0.057 (n 4) -> **FAIL** (and n is far too small to say anything: 9 MKT fills in total).
4. Ex-top-5 % of RESTING > 0: in-regime +0.005 (pass, marginal), all -0.031, out -0.119 -> **FAIL** overall.
5. Fills/week if RESTING-only (cadence bar >= 3/wk): replay-filled RESTING 2.06/wk (90 wk), MKT 0.06/wk; BT-entered basis 4.63/wk -> **FAIL on the replay basis** (pass on the BT basis, which includes fills the tape could not obtain).
=> On FAIL the live tercile split (1,700) is recorded as selection noise as far as this test can see; no rule ships; engine untouched.
## Buckets, time submit -> fill (descriptive, decides nothing), all replay-filled | in-regime | out
<= 5 s: n 44 -0.081 | n 27 -0.018 | n 17 -0.182. 5-34 s: n 85 +0.231 | n 65 +0.234 | n 20 +0.221. 34 s-4 min: n 121 +0.099 | n 85 +0.141 | n 36 +0.001.
> 4 min: n 20 -0.152 | n 13 -0.142 | n 7 -0.169. Stop share per bucket in-regime: 18 % / 34 % / 35 % / 31 % (BT overall 34 %).
## Adversarial caveats
* The marketable cohort is nearly empty (9 fills; 5 in-regime) because the replay's own chase guard removes marketable fills before they are scored; the split cannot confirm or refute a "chase" effect. The <= 5 s bucket (n 44) is the better proxy and is -0.08 R (CI [-0.18,+0.02]), not significantly negative.
* 36 % of fills have no print in the lookback: MKT is under-identified (a quote-based last would change it); engine decision time not modelled (all names at 09:35:05).
* Exit reasons / stop share are the BT's own (only the entry price is re-simulated), so the stop-share mechanism of 1,702 (live 41.6 % vs BT 28.4 %) is NOT testable here: replay stop share is 32 % in-regime and not higher in the early bucket.
* Out-of-regime failure rests on n 76-80 with CI spanning +/-0.13; sub-period out-of-regime halves have opposite signs (-0.09 / +0.14). Wrapper mix differs sharply by regime (58-63 % in, 4 % out) - a composition confound. In-regime RESTING is tail-carried (ex-5 % +0.005, cap3 +0.118 survives).
* Cell count: 1,707 plus the 3 threshold variants and ~25 segment reads here; none corrected for multiplicity.
