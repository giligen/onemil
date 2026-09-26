# RESULT — cell 1,443: stop slippage measured on the tape (WEEKEND_QUEUE.md row 10)

Report-only cell: no pass bar. Base fills = `causal_arming_causal.csv` (cell 1,438, correct levels);
the 1,427 "E1 fills" are VOID (sparse levels) so 1,438's fills replace them. **TEST was never run for
1,438, so TEST is not read here** — only TRAIN-H2 and VAL. Coverage: 4879/7096 tape windows measured
(68.8%); gaps are `no_valid_quote` (1154), `no_print_le_stop` (999 — stop never touched by a print in
that minute, e.g. gap-through or quote-only triggers), `no_tape` (64); zero `fetch_error` (no rate
limiting). 76 `stop_bar` rows are flagged `fill_bar_approx` (fill instant not stored in this CSV).

## Slip table — stop and EOD exits, per holdout (bps)

| holdout | kind | n measured / requested | mean | median | p75 | p90 | share >30bps | share >100bps | mean slip (R) |
|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | stop | 1831 / 2565 (71.4%) | 35.9 | 23.7 | 46.4 | 82.6 | 41.7% | 6.3% | 0.209 |
| TRAIN-H2 | eod | 332 / 567 (58.6%) | 11.5 | 10.3 | 25.4 | 39.5 | — | — | — |
| VAL | stop | 2291 / 3210 (71.4%) | 34.8 | 22.4 | 47.2 | 81.9 | 40.3% | 6.5% | 0.199 |
| VAL | eod | 425 / 754 (56.4%) | 9.7 | 6.5 | 22.0 | 46.7 | — | — | — |

## Restatement table — cell × holdout: net R before / after the measured stop slip

1,438 uses its OWN per-trade measured slip (charged only on stop/stop_bar exits); 1439/1428/1441 reuse
the 1,438 HOLDOUT-MEAN bps applied to their own stop exits (no separate tape fetch, per spec). "Before"
and "after" are the mean net R over the FULL fill population (target/eod exits included, unchanged) —
only the stop/stop_bar subset is adjusted.

| cell | holdout | n stops | mean slip (bps) | net R before | net R after |
|---|---|---|---|---|---|
| 1438 | TRAIN-H2 | 2565 (1831 measured) | 35.9 | -0.208 | -0.295 |
| 1438 | VAL | 3210 (2291 measured) | 34.8 | -0.224 | -0.307 |
| 1439 | TRAIN-H2 | 2702 | 35.9 (1438's mean) | -0.129 | -0.263 |
| 1439 | VAL | 3618 | 34.8 (1438's mean) | -0.279 | -0.414 |
| 1428 | TRAIN-H2 | 209 | 35.9 (1438's mean) | -0.067 | -0.185 |
| 1428 | VAL | 244 | 34.8 (1438's mean) | +0.109 | +0.002 |
| 1441 | TRAIN-H2 | 74 | 35.9 (1438's mean) | +0.016 | -0.089 |
| 1441 | VAL | 139 | 34.8 (1438's mean) | -0.168 | -0.286 |

1,428's fills CSV: no separate fills file exists for that cell — `cell_1428_causal.csv` (same schema)
was used as the fills source; not skipped.

## Live comparison, corrected

The 9/25 live day had **no stop exit** to compare (VECO closed at TARGET +$5.5, CDNA at the 15:55 EOD
exit −$88). The only live-tape number available is ENTRY slippage: VECO +14.3 bps, CDNA −2.0 bps
(n=2, `logs/hod_live_parity_ledger.csv` column `slippage_vs_tape_bps`). No live stop-exit slippage
exists yet to validate the measured-tape numbers above against.

## Verdict

* VAL mean measured stop slip = 34.8 bps, TRAIN-H2 = 35.9 bps — both **below the 40 bps size gate**:
  PASS, no block on a size increase from this gate alone.
* Every restated cell's net R gets WORSE under measured slip (by 0.09–0.16 R depending on cell/holdout)
  — the flat 30 bps variant cell 1430 already used was, if anything, optimistic vs the measured tape.
* Coverage is 68.8% with real (not fabricated) gaps split three ways above; this is a measurement, not
  a closure — the ~31% unmeasured rows are left at their as-recorded net_R, not imputed.
