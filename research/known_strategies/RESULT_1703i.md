# RESULT 1703i: Stack weight optimization

## GREF reference (w=0)
GREF ratio: 0.7575 (expected ≈ 0.75)
GREF max DD: -0.3825
H1 ratio (2017-21): 0.6501
H2 ratio (2022-26): 2.0435

## Component cells (w > 0)
mom_20 w=10%: ratio 0.8562 (+0.0987), DD -0.3851, comp_worst10 -0.1372 — FAIL
mom_20 w=20%: ratio 0.9337 (+0.1762), DD -0.3930, comp_worst10 -0.1376 — FAIL
mom_20 w=30%: ratio 0.9668 (+0.2093), DD -0.4151, comp_worst10 -0.1380 — FAIL
sma_100 w=10%: ratio 0.8862 (+0.1287), DD -0.3782, comp_worst10 -0.1372 — FAIL
sma_100 w=20%: ratio 0.9806 (+0.2231), DD -0.3855, comp_worst10 -0.1376 — FAIL
sma_100 w=30%: ratio 1.0637 (+0.3063), DD -0.3928, comp_worst10 -0.1380 — FAIL

## Shared tail (10 worst weeks)
See 1703i_worst_weeks.csv

## Verdict: NO CELLS PASS
BTC trend is a personal-holding matter; close the stack question.
## Reviewer correction (Fable): the agent's `comp_ret` column equals `sleeve_ret` in 7 of 10 rows — it measured the sleeve, not the component.
Recomputed from the BTC closes (REREAD_1703i_tail.py, 1703i_worst_weeks_corrected.txt): component mean in the sleeve's 10 worst weeks
= −4.2 % (mom_20), −5.3 % (sma_100), BTC hold −11.6 %. Still < 0 → every cell FAILS the shared-tail criterion; verdict unchanged.
Shape: in 6 of the 10 weeks the trend rule was in cash (≈ 0); in 2021-02-26 and 2021-05-14 it lost 17 % / 14–18 % alongside the sleeve
(risk-off weeks hit momentum stocks and BTC together). Ratio gains (+0.10…+0.31) and the flat max DD are not disputed.
