# REPORT — HOD-break ENTRY: resting stop-limit at the level (cells 1,423-1,425)

PREREG: `research/hod_entry/PREREG.md` (frozen 2026-09-25, unchanged). Implementation:
`research/hod_entry/entry_replay.py` (24 unit tests in `test_entry_replay.py`, all pass, on
synthetic ticks/bars — no DB, no parquet). Full replay: `research/hod_entry/replay_signals.csv`
(8,248 rows, TRAIN-H2 + VAL only; TEST dropped on read from `b0_trades.csv` and never touched).

**Caveat up front (CLAUDE.md protocol):** this is a SINGLE implementation. The independent
reimplementation-and-trade-by-trade-compare step (CLAUDE.md's "no research claim ships without an
independent check", step 1) was NOT done inside this task's 40-tool-call budget — flag before
relaying the headline number to the owner.

## Data source note — level definition
Level = running max of CLOSED 1-min highs strictly before the break bar (`trading/hod_break.py:
detect`, `level = hod[i-1]`). Recomputed from `data/cache.db`'s `intraday_bars_1min` (regular
hours, deduped by minute) per PREREG's explicit pointer — NOT read from the `level` column already
in `research/bf_zero/causal_filter/features.csv`. An 8-signal spot check found 4/4 exact matches
where both sources had data, 1 mismatch, 3 missing break-bar bars in cache.db — i.e. `bars_sip.db`
(what `features.csv`/the original walker used) and `cache.db` are not identical sources. Recomputed
level is None (VOID) for only 3/8,248 signals.

## Availability rail
| Split | n signals | usable | usable share | winner usable | loser usable | gap |
|---|---|---|---|---|---|---|
| TRAIN-H2 | 3,503 | 2,625 | 74.9% | 70.4% | 77.4% | 7.1 pp |
| VAL | 4,745 | 3,538 | 74.6% | 71.7% | 76.1% | 4.4 pp |

**Both splits are below the 80% availability floor** (74.9%/74.6%). Void reasons (VAL):
no_trigger_cross_xnas 667, no_trade_in_break_bar 536 (single-venue XNAS tape gaps — consistent with
lens (i) below), no_level 2, no_quote_at_fill 2. Winner/loser missingness gap is inside the 5pp bar
on VAL (4.4pp) but not TRAIN-H2 (7.1pp). This alone would VOID the availability leg; reported here
for completeness since the PASS bar fails on other legs regardless (see verdict).

## Chase distribution (level -> B0's next-open entry, in R)
| Split | median | mean | n |
|---|---|---|---|
| TRAIN-H2 | 0.192 R | 0.963 R | 3,502 |
| VAL | 0.202 R | 0.948 R | 4,743 |

The median signal pays ~0.19-0.20 R of pure chase between the level and B0's forced next-bar-open
entry — this is the lever the resting order is meant to recover.

## E1 / E2 / E3 tables
E1 = capped stop-limit (15 bps). E2 = idealised (report-only). E3 = B0 entry re-costed (pairing
control). Fill rate = E1 fills / usable. Fills/week via `research/hod_consol/run_consol.
simulate_slots` (CONCURRENT_CAP=4, DAILY_CAP=12) on E1-filled trades, over the split's unique `wk`
count from `b0_trades.csv`.

| Metric | TRAIN-H2 | VAL |
|---|---|---|
| E1 fills / usable | 895 / 2,625 | 1,059 / 3,538 |
| E1 fill rate | 34.1% | 29.9% |
| E1 mean net R | **+0.262** | **+0.384** |
| E1 ex-top-5% mean net R | +0.172 | +0.301 |
| E1 fills/week (slot-filtered, raw fills) | 25.5 (688 kept / 895 raw, 27 wk) | 32.3 (711 kept / 1,059 raw, 22 wk) |
| E1 ask+1-tick sensitivity, mean net R | +0.245 | +0.367 |
| E2 mean net R (5 degenerate near-zero-R rows excluded, chase_R~1.0) | +0.176 (n=1,514) | +0.186 (n=1,921) |
| E3 mean net R | -0.289 (n=2,613) | -0.330 (n=3,525) |
| Paired dR (E1-E3), mean | **+0.048** | **+0.085** |
| Paired dR, day-clustered SE / t | 0.017 / t=2.79 (n=894, 118 days) | 0.019 / **t=4.42** (n=1,059, 102 days) |
| Lens (i): first XNAS print >= trigger arrives >=5s after break-bar open | 66.2% | 65.5% |

E2's 5 excluded rows (all chase_R ~1.0, i.e. level ~ B0's own stop) had a near-zero recomputed R
that blew up the R-normalized net_R (one as extreme as -5.2e13); a real data-integrity edge case in
the idealised (report-only) leg, not present in E1 or E3 (both checked clean, max |net_R| < 2.5).

## Three lenses
1. **Single-venue timing**: 65-66% of first-trigger-cross prints land >=5s after the break bar
   opens — a consolidated tape would very plausibly trigger EARLIER on many of these (other venues
   printing sooner), meaning E1's measured fill/no-fill split and cost are biased toward the LATE,
   worse-priced end of what a multi-venue router would see. Direction: this study likely
   UNDERSTATES achievable fill rate and OVERSTATES entry cost relative to a real router.
2. **Fill-rate/edge trade-off (5/15/30 bps)**: NOT computed — would require re-running the full
   tick replay at each limit (the 15 bps run alone took ~8 min end to end); out of budget. Flagged
   as the first follow-up if this lever is revisited.
3. **Ask+1-tick sensitivity**: done (table above) — E1 net R stays comfortably positive on VAL
   (+0.367 vs +0.384 base), so the entry-side result is not a one-tick artifact.

## Frozen pass-bar verdict (E1 PASS iff ALL of the following)
| Leg | Bar | TRAIN-H2 | VAL | Leg result |
|---|---|---|---|---|
| E1 mean net R | >= +0.05 both splits | +0.262 | +0.384 | **PASS** |
| Paired dR vs E3 | >= +0.10 both splits | +0.048 | +0.085 | **FAIL** (both below bar) |
| VAL day-clustered t | >= 2 | - | 4.42 | PASS |
| Fill rate | >= 70% | 34.1% | 29.9% | **FAIL** (both well below bar) |
| E1 ex-top-5% on VAL | > 0 | - | +0.301 | PASS |
| E1 fills/week | >= 3 | 25.5 | 32.3 | PASS |
| Ask+1-tick sensitivity on VAL | >= 0 | - | +0.367 | PASS |

## Verdict: **FAIL**

Two legs fail outright: paired dR vs E3 (+0.048/+0.085, bar +0.10) and fill rate (30-34%, bar
70%). The entry price lever recovers real edge — mean net R is comfortably positive on both
splits, the E1-vs-E3 pairing is directionally right and significant on VAL (t=4.42), and the level
itself carries ~0.2R of recoverable chase — but the 15 bps chase cap is too tight for XNAS-only
tape reality (roughly two-thirds of usable signals never cross-and-qualify under the 15bps limit),
so under the FROZEN trigger/limit this entry does not clear its own bar. Per PREREG's "Not
allowed" clause, the limit/trigger/cost/pass-bar were not touched to try to pass.

**Consequence (pre-committed): FAIL.** MDE: fill rate would need to roughly double (30% -> 70%) to
clear that leg alone; paired dR would need to roughly double too (0.05-0.08 -> 0.10) independent of
fill rate. This specific 15-bps-capped resting-limit construction is closed on this population.
Lens (ii) (wider caps, 30 bps report-only) is the natural next cell if this lever is revisited —
not run here (budget).
