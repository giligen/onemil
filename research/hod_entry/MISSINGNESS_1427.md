# Cell 1,427 — unusable-signal classification (TRAIN-H2 / VAL)

683 unusable (`status=='no_tape'`) signals reclassified by proximate cause, using cached
`sip_cache/{day}.pkl.gz` tapes + `data/cache.db intraday_bars_1min`. No rule code touched.
Coverage/gap figures reproduce `REPORT_1427.md` exactly (3208/3503 = 91.6%, 4357/4745 = 91.8%;
missing win/lose = 4.0/8.9 pp TRAIN-H2, 3.8/9.4 pp VAL — losers go missing ~2.3x more often
than winners; population base winner rate ≈ 35% both cohorts).

| reason | TRAIN-H2 n | VAL n | win share (base 35%) |
|---|---|---|---|
| no valid quote at the fill-instant print (`prevailing_quote`=None) | 229 | 320 | 18% |
| no valid quote anywhere in the window (`qv` empty) | 44 | 40 | 0% |
| no trades at all in [S-60s,S) | 21 | 26 | 12–29% |
| no prior RTH bars → level NaN | 1 | 2 | — |
| trades+quotes both present, no print ≥ trigger | 0 | 0 | n/a (this path resolves to `nofill`, i.e. usable, not unusable — never occurs) |

Every bucket is loser-skewed vs the 35% base rate — the miss-rate asymmetry is real, not one
outlier cause. `no_quote_in_window`: of 84, only 16 had the 1-min bar high confirm a break
(bar_high≥level) — most are quiet names where neither ticks nor quotes moved. `no_trades_in_window`
(5 live-requeried at S-120..S+60): all 5 returned prints just outside the 60s bar (1–68 trades in
the wider window) — genuine sparse-tape illiquidity, not a window bug.

**Diagnosis:** mechanical, partially fixable. The dominant cause (549/683, 80%) is `no_quote_at_fill_instant`
— a real trigger print exists but the nearest prevailing quote is missing *because the fetch window
only looks back to S-65s*; for thin names the last true quote update can be older than that, so
`prevailing_quote` wrongly returns None instead of the actual prevailing NBBO. This is a window
truncation bug in the fetch, not a tape gap — widening the quote lookback (fetch quotes from well
before S-65s, independent of the trade window) should recover a material share of these. The
remaining ~20% (`no_quote_in_window`, `no_trades_in_window`) is genuine low-liquidity tape sparsity
concentrated in failed/losing setups, consistent with the loser-skew being partly a real property
of the underlying names, not purely an artifact.
