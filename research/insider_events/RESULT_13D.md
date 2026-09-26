# RESULT — cell I4, initial Schedule 13D filings (Builder B)

| split | arm | n_trades | mean_net_% | t_day_clust | months_pos_% | sig/wk | ex_top5_% | book_%/mo | book_maxDD_% | null_pctile | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN | all | 951 | +0.74 | 0.92 | 56.2 | 3.04 | -2.08 | +1.75 | -71.8 | 20.9 | NO |
| VAL | all | 286 | -1.81 | -0.98 | 44.0 | 2.75 | -5.38 | -1.36 | -46.7 | 3.6 | NO |
| 2018-23 | surv_with_delisted | 1007 | -0.42 | -0.47 | — | — | — | — | — | — | n/a |
| 2018-23 | surv_ex_delisted | 1006 | -0.39 | -0.43 | — | — | — | — | — | — | n/a |

I4 FAILS the frozen bar on both splits: TRAIN misses mean (+0.74% < +0.8%) and t (0.92 < 2.5); VAL flips sign
(-1.81%), fails every VAL gate including the count-matched null (percentile 3.6, i.e. worse than 96% of random
eligible-symbol draws on the same dates) and book max-DD (-46.7% vs a -15% cap). Survivorship arm carries almost no
signal (1/1007 trades on a name that stops pricing before panel end) so it cannot rescue or explain the failure.
Root cause not diagnosed here — deferred to the negative-result adequacy review this repo requires before closing
a cell (per CLAUDE.md, a null is a claim about the test first). No independent reimplementation has been run yet;
this number is NOT cleared for the owner until that check and the causality/price-scale/fill-realism passes run.
