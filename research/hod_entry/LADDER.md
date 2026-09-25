# HOD fill-filter ladder — cells 1,428–1,437, pre-registered 2026-09-25 ~17:15 UTC, BEFORE cell 1,427's answer

Every rung below runs ONLY if cell 1,427 (the resting 15 bps stop-limit's fills as the book, confirmed on the sealed
TEST with the consolidated tape) PASSES. Order = expected value per dollar of work. Each rung is one cell with its own
pass bar, scored on TRAIN-H2 and VAL first, TEST read once only if VAL passes. The "E1 book" below always means the
frozen rule of `PREREG_1427.md` (trigger level + $0.01, limit level × 1.0015, fill at the NBBO ask at the first
consolidated print ≥ trigger, B0 stop / 2 R target / 15:55 exit, measured half-spread cost). Nothing in that rule
changes in any rung; rungs ADD to it or apply it elsewhere.

| cell | rung | what changes | pass bar (all: day-clustered t ≥ 2 on VAL, ex-top-5 % > 0, ≥ 3 fills/wk) |
|---|---|---|---|
| 1,428 | **Same order, gapper universe** | Apply the E1 rule to HOD breaks in the ORB candidate universe (gap ≥ 5 %, open $3–30, prior vol ≥ 500K) instead of the $20+/ADV100K universe; signals built by the HOD-break spec on that population, 2025-01..2026-05 | E1-fill mean net R ≥ +0.15 on TRAIN-H2 and VAL (small caps cost more; the bar is higher) |
| 1,429 | **Limit width as size** | Report-only 5 / 15 / 30 bps in 1,427 → here a rule: fills whose ask-at-fill is within 5 bps of the level get 1.5× risk, 5–15 bps 1×, and (new) 15–30 bps 0.5×, all from the SAME resting order (a single stop-limit at 30 bps, sized after the fill is known is NOT obtainable — so the size is set from the pre-fill quote: the ask distance the instant BEFORE the trigger) | book R per unit risk ≥ the flat E1 book's by +0.05 on both holdouts; worst week not worse |
| 1,430 | **Exits re-tuned on a winning book** | The exit lab's 35 exit cells were scored on a −0.3 R population; rerun the 6 best-mechanism cells (breakeven lock, ORB-style lock, scale-out 50 % @ +2 R, time-stop 90 min, VWAP-loss exit, EOD hold-to-15:55 vs 14:30) PAIRED on the E1 fills | ΔR ≥ +0.05 on both holdouts vs the B0 exit |
| 1,431 | **No-fill cohort as the short** | The ~70 % that never fill are the losing side (B0 −0.3 R). New definition: short at the next bar's open when the break bar closed above the level but the E1 order did NOT fill AND the close is ≥ 15 bps above the level; stop = break bar high + 1 tick; target 2 R; borrow-checked (`shortable` from the asset dump) | mean net R ≥ +0.10 on both holdouts; ≥ 60 % of names shortable |
| 1,432 | **ORB with the same fill discipline** | ORB's buy-stop at the range high with the live 50 bps chase cap → replay ORB fills (the cell-1,426 tick data) with a 15 bps limit instead; compare books | out-of-sample (2023-24 + 2025H2-26) mean R ≥ +0.10 and ≥ +0.05 over the 50 bps book |
| 1,433 | **Bull flag with the same fill discipline** | BF's buy-stop at the flag high (live: stop-limit, limit +0.5 %) → 15 bps limit, on the P1 Stage-2 trades 2025-26 with SIP ticks | mean R ≥ +0.20 on the fills and fill rate ≥ 40 % |
| 1,434 | **Second break of the day** | After a filled E1 that stopped out, re-arm the same order at the new HOD once (max 2 entries per name-day) | the re-entry cohort's mean net R ≥ 0 and the day book's R/day ≥ the single-entry book's |
| 1,435 | **Pre-market-high break, same order** | The exit lab's other frame (`PREREG_PMH.md`): PMH level (pre-market vol ≥ 50K), the E1 order at PMH + $0.01, before 11:30 | mean net R ≥ +0.10 on both holdouts |
| 1,436 | **Time-of-day and day-of-week as a reported table, NOT a filter** | Report-only on the E1 book: R by entry-hour bucket and weekday | no pass bar (multiplicity guard: no filter is adopted from this table without its own cell) |
| 1,437 | **Live parity instrument** | The dry run logs, per signal, the resting order's fill/no-fill and the NBBO ask at the first cross, so the live fill rate and fill quality are compared daily with TEST's | 10 sessions: live fill rate within 10 pp of TEST's and fills' mean net R same-signed |

## Not allowed
Running any rung before 1,427 PASSES; changing the E1 rule inside a rung; adopting a filter from 1,436; reading TEST
for any rung before its VAL bar is met. Programme count after this file: 1,437.
