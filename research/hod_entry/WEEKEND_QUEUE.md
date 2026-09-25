# Weekend research queue 2026-09-26/27 (owner 9/25: "run them all through this weekend 1 by 1; low-end agents run, Fable reviews and judges; by Monday we optimize")

One cell at a time (2-CPU node). Every cell: PREREG frozen by the main session before any number, Sonnet (Haiku for
mechanical steps) runs, main session reviews adversarially, VAL first, the sealed TEST read ONCE only on a VAL pass,
result appended to `research/hod_entry/WEEKEND_RESULTS.md`. Base rule = `PREREG_1427.md` E1 (resting stop-limit at
level + $0.01, limit 15 bps, NBBO ask at the first cross, B0 stop / 2 R / 15:55, measured half-spread); once cell
1,438 reports, its causal-arming version replaces it as the base for every later cell.

| # | cell | what | data | pass bar |
|---|---|---|---|---|
| 0 | live sizing fix (Monday, code) | `trading/hod_break.py::resting_order_qty` must enforce ALL of: qty ≤ risk_usd / (trigger − stop); qty × trigger ≤ max_notional_usd (bypassed 9/25: CDNA $3,708 on a $2,000 cap); and a liquidity cap that is not the prior bar's 5 % (VECO 11 shares on a $50 risk) — use min(25 % of the prior bar's volume, displayed ask size × 3) with a WARNING when it binds; test each cap binding alone; parity test with the backtest's size assumptions reported (the BT has no notional cap) | code | tests + rehearsal before Monday's boot |
| 0b | live exit bookkeeping (Monday, code) | after a broker TP/SL leg fills, the engine must detect it (poll the leg ids it stores), close the StopMonitor watch, and write the exit to trades.db (VECO 9/25: TP filled at the broker, engine logged nothing, row 380 stayed open, phantom watch survived until a restart; CDNA row 381 open after the StopMonitor stop exit) | code | tests: leg fill → watch removed + row closed; StopMonitor exit → row closed |
| 1 | 1,438 | causal arming (the live rule) | bars_sip.db + SIP ticks | PREREG_1438 |
| 2 | 1,430 | exits on the winning fills: time stop 90 min, breakeven lock at +1 R, ORB-style lock, 50 % scale-out at +2 R, VWAP-loss exit, 14:30 vs 15:55 close — PAIRED on the fills | paths.parquet + fills | ΔR ≥ +0.05 on TRAIN-H2 and VAL, VAL t ≥ 2 |
| 3 | 1,429 | fill-quality sizing: risk multiplier from the pre-trigger ask distance (≤ 5 bps 1.5×, 5–15 bps 1×) and from the trigger print size (round lot vs odd lot, report-only) | SIP quotes at the cross | book R per unit risk ≥ flat + 0.05 on both holdouts, worst week not worse |
| 4 | 1,431 | no-fill cohort SHORT (break bar closed ≥ 15 bps above the level, E1 did not fill): short at the next open, stop = break-bar high + 1 tick, 2 R target, shortable names only | b0 population + tape | mean net R ≥ +0.10 both holdouts, VAL t ≥ 2, ≥ 60 % shortable |
| 5 | **DONE 21:31 UTC — FAIL (`RESULT_1439.md`): primary non-SSR shortable book −0.15 / −0.29 R (t −2.3 / −4.8); first run VOID (population look-ahead), fixed + rerun** — 1,439 (NEW) | low-of-day mirror: the HOD-break spec mirrored (≥ 5 % below the open, 5-bar consolidation within 4 % of the running LOD, rv band), resting SELL stop-limit at LOD − $0.01, limit LOD × 0.9985, fill at the NBBO bid at the first print ≤ trigger, stop = consolidation high, 2 R target, 15:55 cover; shortable + borrow-fee-aware names | bars_sip.db + SIP ticks (new fetch) | fill mean net R ≥ +0.10 both holdouts, VAL t ≥ 2, ex-top-5 % > 0, coverage ≥ 80 %, gap ≤ 5 pp, ≥ 3 fills/wk |
| 6 | **DONE 23:40 UTC — FAIL (`RESULT_1428.md`): VAL +0.11 R t 0.5 (bar +0.15, t 2), TRAIN-H2 −0.07 R, stop-slip variant +0.02; 7,487 PIT symbol-days, coverage 100 %** — 1,428 | gapper universe (gap ≥ 5 %, open $3–30, prior vol ≥ 500K), same order | orb universes' minute bars + SIP ticks (new fetch) | fill mean net R ≥ +0.15 both holdouts |
| 7 | 1,435 / 1,432 / 1,441 (NEW) | other levels with the same order: pre-market high; opening-range high (ORB tick data exists); prior-day high | existing bars/ticks (+ PMH pre-market bars) | mean net R ≥ +0.10 both holdouts each |
| 8 | **DONE 20:20 UTC — FAIL (`RESULT_1440.md`): floor 0.8 % binds on 0.02 % of fills (ΔR 0); floor+cap 3 % touches 10 % and LOWERS the book, VAL ΔR −0.007 t −2.5, TRAIN same sign** — 1,440 (NEW) | stop distance: floor / cap the consolidation-low stop at 0.8 % / 3 % of price; report cost in R and fill count | fills | ΔR ≥ +0.05 both holdouts |
| 9 | 1,442 (NEW) | tape-triggered override: the same E1 fills re-priced two ways, paired per fill — (a) the broker's rule: fill at the ask at the first ROUND-LOT print ≥ trigger (Alpaca's NBBO-filtered trigger, 9/25 VECO: +16 s, +14 bps), (b) our override: fill at the ask 300 ms after the first print of ANY size ≥ trigger, both capped at the 15 bps limit; report the fill-rate and mean-R difference, and the outcome of odd-lot-led crosses vs round-lot-led crosses | SIP ticks already fetched | override ΔR ≥ +0.03 per fill on both holdouts with no loss of fill rate → ship the 1-second override Monday; else keep the resting order as is |
| 9b | **DONE 20:05 UTC — FAIL / not runnable (`RESULT_1444.md`): the +0.25 R cohort is a full-day-range look-ahead from the cache build; no point-in-time scanner record; forward instrument = `scanner_qualified_at_arm` column in the dry ledger (Monday engineering item)** — 1,444 (NEW, `PREREG_1444.md`) | resting entry ONLY on names the live scanner had already qualified before the break (the causal restatement of what separated the +0.25 R cohort from the −0.36 R cohort in 1,438) | 1,438's fills + scan_results | VAL ≥ +0.15 R, t ≥ 2, ex-top-5 % > 0, ≥ 3 fills/wk; look-ahead placebo must not beat it |
| 10 | 1,443 (NEW) | stop slippage measured on the tape: for every E1 fill that exited by stop (TRAIN-H2, VAL, TEST), fetch SIP quotes for the stop minute and take the NBBO bid 250 ms after the first print ≤ stop; slip = (stop − bid)/stop in bps; also the 15:55 exit vs the 15:55 bid; charge the measured per-trade slip and re-report every cell's net R | Alpaca SIP quotes (new fetch, ~2,600 windows, free) | report-only for the number; the live book's expectation is restated with it; if the measured mean slip > 40 bps no size increase until live stops confirm |

Programme count after this file: 1,443. Not allowed: reordering to chase a good number; changing E1 constants; reading
TEST for any cell before its VAL bar; more than one heavy job at a time while the trader runs.

## Final item — OWNER REPORT (`research/hod_entry/OWNER_REPORT_20260928.md`, ≤ 1 page, written last, Sunday evening)
1. **Findings**: one table — every cell run this weekend (n, mean net R, t, ex-top-5 %, fills/wk, verdict) + the live
   ledger from 9/25 (fills, tape-vs-broker bps, stop slip) in one row each.
2. **What changes Monday** (config and code, each with its evidence line and rollback): sizing caps, exit bookkeeping,
   OCO, any cell that passed.
3. **Money view**: expected monthly $ at $50 / $100 / $375 risk under the numbers that survived, worst month beside,
   the review's caveats that still apply, and where the $15K recovery stands on that path.
4. **Questions for the owner**: only decisions that are genuinely his (size step, budget for data, short-side borrow
   costs, anything touching his manual account use). No menus.
5. **Recommendations**: the next three actions in priority order, each with the number it rests on.
Think as an owner throughout the weekend: every fill review ends in a queue item, every failed cell states its MDE, and
nothing reaches this report without its independent check.
