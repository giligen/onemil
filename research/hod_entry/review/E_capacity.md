# LENS E — Capacity and Operations (adversarial review of cell 1,427 / PREREG_1438)

Reviewer: independent Sonnet pass, budget 40 tool calls. Source data: `research/hod_entry/sip_rebuild_test.csv`
(the SIP TEST cohort behind REPORT_1427.md's headline, 3,521 signals / 972 fills, `status=='fill'`, `fill`=ask paid,
`R`=$ risk/share = entry−stop), `data/cache.db` (`daily_bars` for ADV20, `intraday_bars_1min` for the entry-bar
traded volume — read-only URI), `research/hod_entry/replay_signals.csv` (single-venue XNAS reference, has
`trigger_lag_s`), `trading/hod_break.py`, `trading/hod_break_engine.py`. 922/972 TEST fills had both ADV20 (≥10
trailing days) and an entry-bar volume row; 50 dropped (new listings / bar gaps) — not weighted further.

**Bottom line: the R/fill number is real but not the deployable number at $375 or $1,000 risk. The position itself
is frequently a large share of the one minute it needs to fill in, which the backtest never charges for. At $100
risk the book is capacity-clean. The engine cannot place this order at all today — `resting_stop_limit` is
hard-DRY, and its own dry-run fill logic is explicitly not tape-accurate (bar-close resolution, not tick).**

## 1. Position size vs. the name's liquidity

`shares = floor(risk_usd / R)` (`hod_break.py::shares_for`, unmodified by any liquidity check — grepped both files,
zero hits for a size cap, only a UNIVERSE-entry ADV20 floor of 100k shares/day). Against the 922 usable TEST fills:

| risk/trade | shares (med/p90/max) | $ position (med/p90/max) | % of ADV20 (med/p90) | **% of the entry MINUTE's volume (med/p90/max)** | fills >1% of minute vol | fills >5% of minute vol |
|---|---|---|---|---|---|---|
| $100 | 129 / 285 / 526 | $4,836 / $8,412 / $15,116 | 0.004% / 0.029% | 0.71% / 10.4% / 221% | **43.8%** | 16.9% |
| $375 | 487 / 1,071 / 1,973 | $18,227 / $31,607 / $56,751 | 0.013% / 0.110% | 2.68% / 39.2% / 829% | **71.0%** | 37.4% |
| $1,000 | 1,298 / 2,857 / 5,263 | $48,694 / $84,299 / $151,423 | 0.036% / 0.292% | 7.15% / 104.4% / 2,210% | **84.3%** | 57.4% |

ADV20 (median 3.1M shares/day, min 100k floor) is a non-issue at every tier — this is a full-day average and the
book trades a burst. **The binding constraint is the entry bar's own printed volume**, because the fill model is
"ask at the first print ≥ trigger inside the break bar" — i.e. the whole position must clear in the remaining
seconds of a single 1-minute bar (median lag from bar-open to cross, from the XNAS reference file, is 21.5 s; 6% of
crosses happen in the bar's last 5 s, leaving almost no fill window). At $1,000 risk the median fill already asks
for 7% of that minute's ENTIRE printed volume (all trades, all sizes, all participants) and the max case is 22×
the minute's volume — i.e. impossible as modeled; the backtest's constant-ask fill price cannot survive contact
with an order that size. At $100 risk it is real but bounded: 44% of fills exceed 1% of the minute, 17% exceed 5%.

**Gap in the data**: the cached SIP quotes (`research/hod_entry/sip_rebuild.py::fetch_tape`) keep only `bp`/`ap`
(bid/ask price) from the Alpaca quote payload — `bs`/`ap` displayed SIZE was never pulled into the cache. There is
no way, from what's on disk, to check the position against the DISPLAYED ask size at the fill instant (what the
task asked for directly); the entry-minute total print volume above is the best available proxy and likely
OVERSTATES fillable size (displayed ask size at one instant is a fraction of a full minute's trades).

## 2. Buying power vs. the $65K account at 4× intraday ($260K)

Using each fill's own dollar size (not a hypothetical), 4 concurrent positions at the sample's own distribution:

| risk/trade | 4× median | 4× p90 | 4× max (worst 4 fills stacked) |
|---|---|---|---|
| $100 | $19,346 | $33,647 | $60,463 |
| $375 | $72,908 | $126,428 | $227,003 |
| $1,000 | $194,774 | **$337,197** | **$605,694** |

$100 and $375 stay under $260K even in the worst-observed single-fill stack. **At $1,000 risk the p90 case alone
($337K) exceeds the account's full 4× intraday buying power**, and the tail case is 2.3× over it — a day where the
book's own 4-concurrent slot cap would need orders rejected or undersized by the broker's margin engine, not by
`max_concurrent=4`. This risk is asymmetric with the account: the owner's own manual positions share the same
buying power and are off-limits to touch (`feedback_owner_manual_trades_untouchable`), so a margin-constrained day
would either throttle HOD's own fills or, worse, interact with his book. Not modeled by REPORT_1427 at all.

## 3. Impact-adjusted monthly P&L

REPORT_1427's TEST book: mean net R = +0.330 (raw, unfilled excluded), 34.2 fills/week after slots, no-fill cohort
mean = −0.571R. Naive linear scaling (34.2 fills/wk × 4.33 wk/mo × mean R × risk$) gives $5,589 / $20,959 / $55,891
per month at $100 / $375 / $1,000 — this is REPORT_1427's implicit capacity assumption and it is wrong at scale.
Re-scored downgrading any fill whose modeled share count exceeds X% of its own entry-minute volume to the no-fill
cohort's −0.571R (a hard floor, not a slippage model — the real number is probably between this and the raw one):

| risk/trade | thresh >5% of min-vol → no-fill | thresh >10% | thresh >20% | raw (unadjusted) |
|---|---|---|---|---|
| $100 | $3,876/mo (17% of fills downgraded) | $4,564/mo (10%) | $5,097/mo (6%) | $5,589/mo |
| $375 | $4,769/mo (37%) | $9,953/mo (26%) | $14,422/mo (17%) | $20,959/mo |
| $1,000 | **−$10,078/mo** (57%) | **$3,156/mo** (44%) | $19,460/mo (30%) | $55,891/mo |

At $100 risk the claim survives capacity (edge shrinks ~10–30% depending on threshold, stays clearly positive). At
$375 it roughly halves. **At $1,000 risk, under any threshold tighter than "a fifth of the minute's volume," the
book is flat-to-negative** — the +0.330R headline is earned disproportionately by the fills that need the fewest
shares (small $ positions on liquid names), and dollar-scaling the same trade population up erases exactly the
fills that carried the edge while keeping the ones that couldn't have filled as modeled. This is the tail-dependence
warning from CLAUDE.md §"No research claim ships" applied to size, not to trade count.

## 4. Operational latency of the resting order

Two different latencies, easy to conflate:
* **Arm-to-cross** (how long the order has existed before price reaches it): by construction the arm is set at the
  PRIOR bar's close, before the break bar even opens (PREREG_1438's causal fix) — from the XNAS reference file,
  median 21.5 s / mean 24.1 s elapse from bar-open to the actual cross (n=1,954 XNAS fills), max 60 s. This part is
  causal and fine.
* **Cross-to-resolution** (how long before the LIVE engine even checks whether it filled) — this is where the dry
  implementation diverges from the backtest's model. `hod_break_engine.py::_evaluate_resting` only walks bar `j`
  once bar `j` is a CLOSED bar the engine has ingested (`for j in range(cand.resting_scanned_idx+1, n)`); it does
  not react to a live trade print mid-bar. Its own docstring: *"DRY ONLY... The live engine has no trade-print
  stream (only closed 1-min bars), so a cross is resolved with the bar's high plus the CURRENT quote... logged
  WARNING (not tape-accurate)."* Concretely: a cross that happens 5 s into the bar (18% of the XNAS sample) is not
  even checked until the bar closes ~55 s later, and the "fill" price used is whatever the NBBO ask is AT THAT LATER
  MOMENT — not the ask that was actually available at the cross. The backtest's edge is built entirely on "ask at
  the first print ≥ trigger" (sub-second); the dry engine as coded today cannot produce that number even in
  simulation, only an approximation with up to ~60 s of drift on both timing and price. The 10-session dry run this
  cell's PASS unlocks will NOT be tape-accurate by the engine's own logging — its fill rate/mean-R comparison
  against TEST's 27.8%/+0.330R needs that caveat attached before it's read as confirmatory.

## 5. What the engine already supports vs. what must be built

Already built and reusable:
* `trading/hod_break.py::shares_for`, `arm_state`, `resting_entry_fill` — the pure sizing/arming/fill functions,
  shared by research and live (parity by construction, per CLAUDE.md's "ONE spec" rule).
* `entry_mode='resting_stop_limit'` config flag and the bar-by-bar arm/resolve loop (`_evaluate_resting`), wired
  into the candidate state machine (`resting_arm`, `resting_scanned_idx`, `resting_filled`).
* A broker-side stop-limit primitive already exists and is used elsewhere: `submit_stop_limit_order` (grepped in
  `trading/order_executor.py:537/556`, `trading/ignition_prestage.py:1186`) — a "simple stop-limit order, no bracket
  legs," so the broker call this book would need is not new code.
* The dry ledger (`_append_dry_ledger`, `logs/hod_dry_entry_ledger.csv`) and `scripts/hod_dry_ledger.py` status.

Must be built before this can submit a real order (none of this exists today — grepped `_evaluate_resting` and the
order-submission path at line ~800; the two are entirely disjoint code paths, and the resting path returns after
logging, never falls into `_submit`/`submit_bracket_order`):
* An order-lifecycle manager for a GENUINELY resting broker order: submit at arm, cancel/replace every bar the level
  moves or arming is lost (the spec says re-price every close), cancel on fill or day-end — none of this exists;
  today's `next_open` path submits exactly once, after the fact, which is a different lifecycle entirely.
* OR a real-time trade-print stream (Alpaca websocket trades channel) so a cross can be detected and acted on
  mid-bar instead of at the next bar close — the codebase's own comment says this does not exist for this engine.
* A liquidity/impact cap on `shares_for`'s output — nothing today checks position size against ADV20, the entry
  bar's volume, or displayed size; §1/§3 above show this is not optional at $375+ risk.
* A per-book buying-power guard aware of the shared account (owner's manual positions) — §2 shows $1,000 risk can
  need more than the account's 4× intraday capacity on its own, before his positions are counted at all.

## Threat to the claim

**Major, not fatal.** The R/fill edge (mechanism: fill-conditioning on a tight ask) is not undermined by this lens —
§1–3 attack SIZE, not the sign of the edge, and at $100 risk (the level CLAUDE.md's other live books are currently
capped to, "every trade risks ≤ $150") the capacity haircut is real but the book stays clearly positive. What's
wrong with the claim as staged for launch: (a) REPORT_1427/PREREG_1438 report a fill-weighted R with no size
attached, and the monthly-P&L extrapolation implied by "34 fills/week" scales that R linearly — it doesn't, past
~$375/trade; (b) the mechanism that must fire live (a broker-resting or tick-reactive order) is NOT what's built —
`resting_stop_limit` is dry-only by a hardcoded guard, and its own fill-resolution is bar-close-lagged and
explicitly flagged not tape-accurate; (c) no displayed-size data was even cached to check this properly, so §1's
numbers are a lower-bound proxy, not a measurement. None of this closes the cell — it sets the scale: ship the dry
run and the eventual live proposal at $100 risk only, and do not read the 10-session dry-run fill rate/mean-R as
tape-accurate confirmation of TEST until the resolution-timing gap in §4 is fixed or explicitly bounded.
