# RESULT 1,457-1,465 -- perfect-foresight ceiling, causal big-day predictors, cost

| cell | holdout | n_kept | n_dropped | kept_mean | dropped_mean | delta_R | t_kept | ex_top5 | fills_wk | winner_capped_mean | kept_mean_flat30 | null_pctile | passes_bar | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1457 | TRAIN-H2 | 2503 | 1895 | 0.1725 | -0.7835 | 0.9560 | 3.2029 | 0.0774 | 45.0741 | 0.1725 | 0.1938 | 100.0000 | False |  |
| 1457 | VAL | 3101 | 2412 | 0.1726 | -0.7855 | 0.9581 | 3.6173 | 0.0779 | 49.3636 | 0.1726 | 0.1903 | 100.0000 | False |  |
| 1458 | TRAIN-H2 | 2209 | 2189 | -0.2861 | -0.1922 | -0.0939 | -5.0526 | -0.4048 | 41.8148 | -0.2861 | -0.3016 | 1.7000 | False |  |
| 1458 | VAL | 2990 | 2523 | -0.1969 | -0.3055 | 0.1087 | -4.1108 | -0.3114 | 46.1364 | -0.1969 | -0.2229 | 99.6000 | False |  |
| 1459 | TRAIN-H2 | 3928 | 470 | -0.2391 | -0.2417 | 0.0026 | -5.3988 | -0.3554 | 49.5926 | -0.2391 | -0.2393 | 51.2000 | False |  |
| 1459 | VAL | 5017 | 496 | -0.2385 | -0.3289 | 0.0904 | -5.5731 | -0.3547 | 49.3636 | -0.2385 | -0.2422 | 91.1000 | False |  |
| 1460 | TRAIN-H2 | 3326 | 1072 | -0.2327 | -0.2600 | 0.0272 | -5.1922 | -0.3488 | 48.8148 | -0.2327 | -0.2256 | 70.7000 | False |  |
| 1460 | VAL | 4306 | 1207 | -0.2274 | -0.3151 | 0.0877 | -5.1341 | -0.3428 | 47.9091 | -0.2274 | -0.2240 | 95.5000 | False |  |
| 1461 | TRAIN-H2 | 23 | 4375 | -0.0691 | -0.2403 | 0.1712 | -0.2167 | -0.1617 | 0.8519 | -0.0691 | -0.0826 | 70.8000 | False |  |
| 1461 | VAL | 25 | 5488 | 0.1391 | -0.2483 | 0.3874 | 0.5295 | 0.0630 | 1.1364 | 0.1391 | 0.1295 | 90.3000 | False |  |
| 1462 | TRAIN-H2 | 777 | 3621 | -0.1499 | -0.2586 | 0.1087 | -1.9810 | -0.2624 | 23.7778 | -0.1499 | -0.2027 | 95.8000 | False |  |
| 1462 | VAL | 935 | 4578 | -0.1114 | -0.2742 | 0.1628 | -1.7829 | -0.2221 | 31.3182 | -0.1114 | -0.1811 | 100.0000 | False |  |
| 1465 | TRAIN-H2 | 23 | 4375 | -0.0526 | -0.2306 | 0.1779 | -0.1668 | -0.1445 | 0.8519 | -0.0526 | -0.0826 | 70.8000 | False | one VAL read, entry component 1461 |
| 1465 | VAL | 25 | 5488 | 0.1391 | -0.2483 | 0.3874 | 0.5295 | 0.0630 | 1.1364 | 0.1391 | 0.1295 | 90.3000 | False | one VAL read, entry component 1461 |
| 1464 | TRAIN-H2 | 4398 | 0 | -0.0989 | -0.2394 | 0.1405 | 9.3265 | -0.2087 | 50.6296 | -0.0989 | NaN | NaN | True |  |
| 1464 | VAL | 5513 | 0 | -0.1199 | -0.2466 | 0.1267 | 10.7470 | -0.2306 | 50.2273 | -0.1199 | NaN | NaN | True |  |

## 1,463 cost table (stop-limit reexecution, bps unless noted)

| variant | holdout | n_resolved | n_no_fill_tail | n_unresolved_total | slip_mean | slip_median | slip_p90 | no_fill_mean_slip | base_stop_slip_mean | book_net_R | book_net_R_before |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20bps | TRAIN-H2 | 964 | 38 | 3741 | 8.847 | 8.259 | 18.660 | 81.527 | 35.893 | -0.231 | -0.239 |
| 20bps | VAL | 1070 | 0 | 3741 | 7.804 | 8.407 | 17.107 | NaN | 34.838 | -0.247 | -0.247 |
| 50bps | TRAIN-H2 | 1587 | 35 | 2421 | 20.896 | 17.923 | 44.077 | 130.904 | 35.893 | -0.230 | -0.239 |
| 50bps | VAL | 1767 | 0 | 2421 | 17.734 | 15.335 | 39.530 | NaN | 34.838 | -0.247 | -0.247 |

1,463 ship bar (mean slip lower by >=10bps AND no-fill-tail mean slip <=100bps, both holdouts): {'20bps': True, '50bps': False}

Coverage: pm_dollar_vol=100%, atr14_pct=99%, prior_range_pct=100%, n_articles=2%, full_day_range_pct=100%.
Kill switch (1,457 VAL kept mean < +0.15): not triggered.
Caveats: TEST is sealed and was not read. 1,463 re-executes the cached bid-250ms measurement with zero new network cost when it already resolves the fill; the remainder needed a fresh Alpaca SIP tape re-fetch (cell 1,443's own fetch_window), bounded at 150 new fetches per variant per this run's time/token budget -- unresolved rows keep the baseline net_R_corr and are counted, never silently dropped (see n_unresolved_total). 1,461's universe is the ORB scanner's own scanned symbol-days (research/scripts/orb_pm_news_nightly_append.py:main()), not the HOD-break universe -- low coverage is expected and reported, not treated as a defect. 1,464 rescales the ORIGINAL tape-measured slip bps onto the new R rather than re-measuring the tape at the new stop price (no new fetch); this is an approximation, disclosed here.

## Judge's note (2026-09-26 06:45 UTC)
* Entry predictors 1,458–1,462: FAIL (all negative on VAL); 1,461 VOID (news file covers 2 % of the book); 1,465 n 25.
* Ceiling 1,457: +0.17 R on VAL with perfect foresight of a ≥ 10 % range day — the kill switch (< +0.15) did not fire,
  but the number says the entry-predictor route is dead at the current execution: a predictor would have to be nearly
  perfect to reach the bar. The adequacy critic notes a higher look-ahead bound exists (the MACD-rule cohort, +0.30/+0.41,
  adds close ≥ $10 and prior-day volume ≥ 1M) and that no causal price/liquidity floor was tested in this round; the
  1,445 liquidity cells (dollar volume through j, bar density) were negative, prior-day volume was not tested.
* 1,464 (R floor 2.5 % of price): PASSES its bar — paired ΔR +0.14 / +0.13, t 9.3 / 10.7; the book goes −0.24 → −0.12 R.
  Not yet independently rebuilt (the rebuilder skipped it by instruction) and not refuted: round 3 does both before it
  counts.
* 1,463 (stop-limit 20 bps): the reported PASS is on a resolved subset selected by construction (stops whose cached
  250 ms bid already cleared the limit resolve for free; 3,741 of 5,775 stops unresolved) — biased low; redone in round 3
  on a random sample with fresh tape.
* Verdict: no positive book on this population; two execution mechanisms with real paired lifts, to be verified and
  composed once (round 3, `PREREG_1466.md`).
