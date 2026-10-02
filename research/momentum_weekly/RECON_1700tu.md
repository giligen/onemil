# RECON_1700tu: first build (A: 1700u_gate_guarded.py) vs rebuild (B: REBUILD_1700tu.py), guarded sleeve, $50K, Mondays 2017-02-06..2026-09-28
Files: RECON_1700tu_dump.py (A/B weekly equity, holdings, weights, in/out prices, cost per name -> RECON_1700tu_{A,B}_{plain,guarded,gated}_{weeks,hold}.csv),
RECON_1700tu_compare.py (-> RECON_1700tu_weeks.csv, _compare.log), RECON_1700tu_corrected.py (-> _corrected.csv), RECON_1700tu_lastweek.py, _diag.py. Neither build modified.
## Price basis: IDENTICAL (not a cause)
10,056 common name-weeks: 0 entry-price mismatches (Monday open, same split-adjusted panel; TSLA 148.20 on 2020-08-31 in both), 2 exit mismatches (both delisted names,
below). Weights equal to 5 decimals; turnover 44.6 %/wk in both; both charge the Monday re-equalisation of kept names. Mean stock return 0.676 %/wk in both.
## Gap (common window to 2026-09-21, Monday-open marks): A 29.29 % / $591,250 vs B 28.67 % / $564,611 = 0.48 %/yr log
| cause | %/yr | whose treatment is right |
|---|---|---|
| cost rule (A: rate from the TRADE-day bar, proxy clipped so rate <= 15 bp; B: signal-Friday proxy, unclipped, cap 20 bp, floor 5 bp) | 0.40 (drag A 3.26, B 3.66) | neither validated (band, not NBBO); B's 20 bp cap binds on 66 % of name-weeks, A's 15 bp ceiling has no basis -> B conservative |
| name selection (4 weeks; history gate, + delist price) | 0.08 | A (spec and LIVE: close 273 rows back exists = 274 bars; B admits 273 bars) |
| price basis | 0.00 | same |
Cost gap by row: name-weeks with rate_B > 15 bp (66 %) +0.63 %/yr; the rest -0.23 %/yr (B's Friday proxy is lower than the Monday range on quiet names). Exits 1.49 vs 1.54 %/yr.
Top 10 |diff| weeks explain 21.5 % of the gap (the 4 name-selection weeks 2020-08-24, 2020-10-26, 2018-10-29, 2018-12-17 = 1.29 log pts of the 4.59); the steady remainder is 0.38 %/yr = cost, 0.0073 %/wk.
| week | net A / B % | cause |
|---|---|---|
| 2020-08-24 (ends at the 2020-08-31 open) | -1.28 / -1.75 | B holds DKNG, LVGO (exactly 273 bars, p=272), A holds FSLY, JD: history-gate off-by-one |
| 2020-10-26 | -6.00 / -6.37 | B PTON (273 bars) vs A PLUG: same gate; LVGO exit 140.45 vs 139.77 (delist) |
| 2018-10-29 | +2.98 / +3.28 | B ROKU (273 bars) vs A ESRX: same gate (B better here) |
| 2018-12-17 | -7.35 / -7.51 | same 20 names; ESRX delisted (Cigna): A exit = last OPEN carried 95.57, B = last CLOSE 92.33; B right |
| 2026-01-26, 2021-11-29, 2025-03-03 | diff 0.03-0.04 | identical names and prices; pure cost (0.014-0.03 pt) |
2020-08-31 week itself: A -9.954 / B -9.965, identical names/prices (AAPL not held, TSLA split-adjusted in both): the 0.01 pt is cost. The first >1 % separation is NOT the split; it is
the 2020-08-24 name-selection week (-0.47 pt) plus the steady cost drift.
## Bridge of reported headlines, guarded CAGR: B 28.34 -> A 29.34
B reported 28.34 (B starts 5 flat-cash weeks early 2017-01-03 and its curve stops 2026-09-21, one rebalance short) -> aligned window 2017-02-06..09-28: 28.80 (+0.46: +0.33 start, +0.13 last week)
-> history gate 274 bars (A/live): 28.72 (-0.07; in the last week B's 273-bar name was a +0.55 % winner, early weeks it lost: a wash) -> cost rule A: 29.24 (+0.52) -> residual +0.10
(delist ffill of the last open, B's guard window one bar short: cb[p]-cb[p-272] covers 272 bars vs A's 273; float32; not traced further) -> A 29.34.
## Corrected headlines (B engine + 274-bar gate + last week + aligned window; max DD on daily close marks; Monday-mark DD in _corrected.csv)
| book | conservative cost (B rule, recommended) | A cost rule (A's own reference) | reported A / B |
|---|---|---|---|
| plain | 26.59 % / -42.9 % / $484,857 | 27.10 % / -42.4 % / $503,989 | ~27.2 -41.6 / 26.24 -42.9 |
| guarded | 28.72 % / -38.1 % / $569,447 | 29.24 % / -38.0 % / $591,891 | 29.34 -38.3 $596,394 / 28.34 -38.1 $564,611 |
| guarded + half gate | 29.99 % / -36.7 % / $626,278 | 30.60 % / -36.7 % / $655,118 | 30.67 -37.1 $658,510 / 29.66 -36.7 $623,605 |
Gate: A 143 half weeks, B 142 (one week, not traced); costs per trade are the band model, so CAGR carries +-0.5 pt/yr cost-model uncertainty until measured NBBO replaces it.
## Which build is right for a live account sending market orders at Monday's open and re-equalising 20 names
Prices/weights/re-equalisation: both (identical). History gate and guard window: A (live code trading/momentum_sleeve.py HIST_MIN_ROWS_BACK=273, GUARD_LOOKBACK=273 matches A; no live defect found).
Delist exit: B (last close) is closer to a real exit than A's carried last open. Cost: use B's rule (conservative) as the headline, A's as the upper bound; neither is a measured cost.
B's reported curve also omitted the last rebalance and began in cash 5 weeks early: report windows must be aligned.
