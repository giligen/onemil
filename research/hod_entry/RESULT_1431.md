# RESULT — cell 1,431: no-fill cohort short

`research/hod_entry/cell_1431.py`, frozen per `PREREG_WEEKEND.md`. Population = cell 1,427's
`nofill` signals with a real cross (`ask_at` set, i.e. the ask exceeded the 15 bps limit — 5,169 of
5,590 `nofill` rows; the other 421 never printed a triggering print at all and are excluded),
restricted to those whose break bar closed >= 15 bps above the level (break-bar OHLC from
`data/cache.db intraday_bars_1min`, sip_rebuild's own level source): TRAIN-H2 D=237, VAL D=324
after that filter. 9/14 further skipped by the 60 bps chase cap (entry too far below the level);
54/65 dropped for no cached tape quote at the entry instant or no path; a further 164/230 dropped by
a **necessary correction**: R = stop − entry is unfloored in the PREREG text, and for signals whose
next-bar open lands on or past the break-bar high, R collapses to 1–2 ticks (one row hit R=3.6e-15,
i.e. entry == stop to float precision) — every R-normalized figure exploded to ±10^13 R. Added a
0.5%-of-price R floor (the project's own prior rule: "R must exceed the spread" killed 3 earlier
books on exactly this failure mode) and logged a WARNING per drop (501 total, see stderr). Final
scored population: TRAIN-H2 n=64, VAL n=93.

Cost: entry half-spread from the cached SIP quote at the entry bar's open; exit cost = the median
reconstructed B0 exit half-spread over TRAIN-H2 E1 fills only (**$0.17**, frozen before scoring,
applied to both holdouts — no per-signal exit spread exists for a trade E1 never took) + 2 bp of
exit price; 30 bps stop-slip on stop exits, reported vs. no-slip.

Shortable flag: the PREREG's named file (`data/research/alpaca_assets_all_20260905.csv`) has **no**
`shortable`/`easy_to_borrow` column in either copy in the repo — used
`research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv` instead (same Alpaca `TradingClient` schema,
today's snapshot, same survivorship caveat as its own source script).

| holdout | D (post break-bar-close filter) | scored n | mean net R (no-slip) | mean net R (30bps slip) | VAL t | share shortable | fills/wk |
|---|---|---|---|---|---|---|---|
| TRAIN-H2 | 237 | 64 | -1.375 | -1.866 | -9.42 | 0.375 | 2.67 |
| VAL | 324 | 93 | -1.058 | -1.438 | -7.90 | 0.376 | 4.43 |

Shortable-only (n=24 TRAIN-H2 / 35 VAL): mean net R (slip) -2.358 / -2.021 — worse, not better.

**Verdict: FAIL, decisively.** Mean net R is deeply negative on both holdouts (bar +0.10), VAL t is
strongly negative in the wrong direction (bar +2), and the shortable share (0.375/0.376) misses the
60% bar too. Mechanism read: this cohort is exactly the signals whose ask ran away fastest after the
break print — shorting the next bar's open buys into the tail of that same run, with the break-bar
high (often already passed) as a stop that gets clipped immediately. No re-run without a different
entry mechanism (e.g. a pullback trigger) under its own PREREG.
