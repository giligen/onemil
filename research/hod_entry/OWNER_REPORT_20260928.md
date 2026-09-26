# Owner report — HOD-break weekend, for Monday 2026-09-28

Written 2026-09-26 by the main session (judge) from results that each passed an independent build or an adversarial
check. Programme count on the HOD line: 1,444 cells. Every number below is net of measured half-spread and B0 exit cost.

## 1. Findings (VAL holdout unless stated; t = day-clustered)

| cell | what | n | mean net R | t | ex-top-5 % | fills/wk | verdict |
|---|---|---|---|---|---|---|---|
| 1,438 | the LIVE rule, correct levels (bars_sip.db), adversarially checked | 5,513 | −0.22 (−0.11 after the double-charged half-spread is removed; raw ≈ 0) | −8 | — | 45 | FAIL |
| 1,427 | "fill filter" +0.33 R on TEST | — | — | — | — | — | VOID: levels from sparse cache.db |
| 1,444 | arm only scanner-qualified names (the +0.25 R cohort restated) | 3,344 | +0.26 in the proxy cohort | 3.8 | — | 45 | NOT RUNNABLE: the cohort is a full-day-range look-ahead from the backtest cache build; no point-in-time scanner record exists |
| 1,429 | size 1.5× when the pre-trigger ask is within 5 bps | 1,443 | ΔR +0.017 | 4.4 | — | — | FAIL (bar +0.05) |
| 1,430 | six exit variants, paired | 1,433–1,443 | ΔR −0.04 to −0.17 | −1.5 to −8.6 | — | — | FAIL ×6 |
| 1,431 | short the no-fill breaks | 93 | −1.06 | −7.9 | — | 4.4 | FAIL (population also built on the void levels) |
| 1,442 | tape-triggered override vs broker round-lot rule | 879 | ΔR −0.016 | −5.1 | — | — | FAIL: waiting 300 ms loses more fills than it improves |
| 1,440 | stop floor 0.8 % / cap 3 % | 5,513 | ΔR −0.0005 / −0.007 | −1.0 / −2.5 | — | — | FAIL: floor is a no-op, cap hurts |
| 1,439 | low-of-day mirror SHORT (non-SSR, shortable) | 3,344 | −0.29 | −4.8 | −0.41 | 43 | FAIL (first run +0.6 R was VOID: population look-ahead, fixed and rerun) |
| 1,428 | same order on the gapper universe (gap ≥ 5 %, $3–30, PIT) | 444 | +0.11 | 0.5 | +0.01 | 17 | FAIL (bar +0.15, t 2); TRAIN-H2 −0.07 |
| 1,441 | prior-day-high as the level | 239 | −0.17 | −3.4 | −0.28 | 10 | FAIL (population = day after a mover day, 74 % had no prior-session bars) |
| 1,443 | stop slippage measured on the tape | 2,291 stops | mean 35 bps, median 22, p90 82 | | | | report-only: below the 40 bps size gate; every book worse than the flat 30 bps assumption |
| LIVE 9/25 | real orders 13:38–15:00 ET at $50 risk | 2 fills / 28 armed no-cross | VECO +$5.5 (target), CDNA −$88 (15:55 exit) | | | | entry vs tape: +14.3 bps (16 s after the cross), −2.0 bps (43 s); no stop exit yet |

### 1a. Stop slippage (cell 1,443, measured on the tape: NBBO bid 250 ms after the first print at or below the stop)

| holdout | stops measured | mean | median | p90 | 1,438 net R before → with measured slip |
|---|---|---|---|---|---|
| TRAIN-H2 | 1,831 / 2,565 | 35.9 bps | — | — | −0.208 → −0.29 |
| VAL | 2,291 / 3,210 | 34.8 bps | 22.4 bps | 81.9 bps | −0.224 → −0.307 |

Below the 40 bps size gate, so the gate itself does not block size — moot, since no book passed. Measurement coverage
69 % (1,154 windows without a valid quote, 999 where the tape never printed at or below the bar-low stop). Every cell
restated with the measured slip is worse than under the flat 30 bps assumption. Live comparator: none yet (no stop
exit on 9/25; entry slippage vs tape VECO +14.3 bps, CDNA −2.0 bps). Corrected live expectation for the HOD rule =
−0.22 R (1,438 −0.31 with measured slip, +0.10 for the double-charged half-spread).

**Reading.** Long or short, $20+ or gapper universe, running high or prior-day high, any exit, any stop, any size rule:
the resting stop-limit at the break earns raw ≈ 0 and loses the spread. The two positive numbers of the week (+0.33 R
TEST in 1,427, +0.6 R in 1,439's first run) were both data-selection artifacts caught by review, not by statistics —
which is why the check protocol exists. Minimum detectable effect at the live cadence (≈ 40 fills/wk, σ ≈ 1.2 R): a
month of dry data resolves ±0.15 R at t 2; nothing smaller is worth a live dollar.

## 2. What changes Monday (all shipped, tested, pushed; rollback = `git revert <hash>` + restart)

| change | evidence | commit |
|---|---|---|
| HOD `dry_run: true` — zero real orders; resting-order parity ledger keeps recording | 1,438 −0.11 R after correction, t −8 | config (backup in scratchpad); rollback = `dry_run: false` on the owner's word |
| TP + safety-net stop as ONE OCO after a fill | Alpaca rejected the second sell on VECO/CDNA (fills without a broker stop for minutes) | 19dd6bc |
| exit bookkeeping: live fills register a Position; TP-leg / StopMonitor exits close the DB row and drop the watch | rows 380/381 stayed open on 9/25 | 4403999 |
| sizing: risk, notional and liquidity caps in one helper | CDNA $3,708 on a $2,000 cap; VECO 11 sh | 5463098 |
| boot reconcile AFTER the session roll | 17:53 boot orphaned HUM/LABX | a2534aa |
| no Telegram per arm; harmless data errors at WARNING | Telegram 429; owner: no harmless errors | 7e605ab, 8df3df5, ebb6876, 18fafd9 |
| HOD state files isolated from the test suite | tests wrote fixture rows into the live parity ledger | 48ba482 |
| EOD report journal count inside journalctl, 90 s | "journal check timed out" on the real 9/25 report | 5a4c5ec |
| research never in RTH, one process, ≤ 2 workers | scanner cycle overruns 75–78 s every 5 min from 15:00 UTC | operating rule (no code) |

## 3. Money view (what survives = nothing on this population)
At the corrected live expectation (−0.22 R net with measured stop slip, ≈ 40 fills/wk under the 4/12 caps): $50 risk
≈ −$1.9K/month, $100 ≈ −$3.8K, $375 ≈ −$14K. There is no path on HOD-break as defined that recovers the $15K; the honest statement is
that the recovery has to come from a different signal. BF: paused (no out-of-sample edge, ~0.8 trades/wk). ORB: closed
as a money book (tape replay +0.03 R/fill out of sample at zero latency, ≈ $11/trade at $375).

## 4. Questions for the owner (only what is genuinely yours)
1. **Budget for point-in-time minute bars.** Every look-ahead this week came from a bar source built by a backtest.
   A full PIT 1-minute set (Databento, ≈ $0.0004/symbol-day) for the 2025-07..2026-05 universes is ≈ $100–150. Approve
   or not; without it every new population starts with a fetch that costs a weekend day.
2. **HOD dry run as a free forward instrument, or stop.** Keeping it costs nothing and yields the only point-in-time
   test of "scanner-qualified names" (cell 1,444's untestable hypothesis) after ≈ 20 sessions. Stopping it frees the
   node and the attention. Default if no answer: keep it dry, with the new ledger column.
3. **The short side.** 1,439 says no for this population; if you want the short frame kept alive it needs a different
   signal, and it interacts with your own manual shorts on the shared account (borrow, SSR days). Your call whether
   the account should carry automated shorts at all.

## 5. Recommendations (priority order)
1. **Monday runs dry, and the next research dollar goes to a NEW signal with gross edge first**, not another HOD-break
   variant: 1,444 cells, 0 pass, every executable entry −0.1 to −0.3 R; the exit lab (35 cells), OFI (3), rank (4),
   fill-filter (18) and this weekend (10) all land inside ±0.05 R of each other. Number it rests on: raw R ≈ 0 on 9,911
   fills with correct levels; −0.22 R net with measured stop slip.
2. **Add `scanner_qualified_at_arm` to the dry ledger (Monday engineering item, ~40 lines)** so the one untested
   hypothesis is measured on point-in-time records; decision rule pre-committed: qualified-cohort ≥ +0.15 R, t ≥ 2 on
   ≥ 100 dry fills → a $50 exploration run. Number it rests on: +0.26 R in the (non-causal) proxy cohort, t 3.8.
3. **Buy the PIT minute bars (question 1)** before any new population study; every VOID this week traced to a bar
   source that selected on the outcome. Number it rests on: 2 of 2 positive results this week were that artifact.

## Revival addendum (2026-09-26, your instruction: "find the way to revive it")
Three rounds, 23 cells (1,445–1,467), every one with an independent rebuild and refuters (`PREREG_1445/1457/1466.md`,
`RESULT_*.md`, `review/1445_*.md`):

| round | what | result |
|---|---|---|
| 1 | the live scanner's own predicate replayed at the arm bar, multi-day highs, liquidity at arm; builder-terms placebo | all FAIL; the placebo failed too → the +0.24/+0.28 R cohort was traced to cache-builder selection on the FULL-DAY range (foresight through the bar cache); levels reliable; cost reconciled: R is 1.6 % of price, so 35 bps stop slip = 0.21 R per stop; the exact cohort under full cost = +0.23 / +0.06 R |
| 2 | perfect-foresight ceiling; pre-market volume, ATR, prior-day range, news; R floor 2.5 %; stop-limit exit | ceiling +0.17 R VAL (a predictor would have to be near-perfect); every causal predictor negative; R floor +0.13 R paired lift; stop-limit needed an unbiased measure |
| 3 | verification + one joint cell | stop-limit VERIFIED (slip 35 → ≈ 13 bps); R floor NOT verified (aggregate agrees, rows do not; the lift sits in 5 % of fills = rescued noise stop-outs); joint +0.05 R t 1.1 FAIL; whole book with both fixes −0.05 / −0.08 FAIL |

**Answer:** there is no way to revive it on this population. The number you were shown was day-range foresight; under
honest execution the entry carries no information a live order can use, and fixing execution recovers about 0.17 R of
the 0.25 R cost but not the book. What survives and ships to the dry run on Monday, all behind flags with tests:
the stop-limit exit (worth ≈ 20 bps per stop on ANY book), the R-floor and stop-limit counterfactual columns, and the
scanner-qualified column. Money view: unchanged. Recommendations: #1 stands (new signal, gross edge first); #2 becomes
the stop-limit exit rehearsal (it applies to every book you run); #3 the free forward columns.
