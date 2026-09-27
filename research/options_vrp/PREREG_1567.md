# PREREG — cells 1,567–1,590: DEFINED-RISK PREMIUM SELLING on SPY — the put credit spread ladder

FROZEN 2026-09-27 10:20 UTC before any number. Programme count: 1,566 → 1,590 (24 pre-registered variants, selected on
TRAIN, one VAL read). Owner 9/27: "I'm all in on options with the right risk."

## The risk, defined first (the owner's constraint; everything else follows)
* Risk budget B = 10 % of account equity (≈ $6,500 at $65K), the MAXIMUM loss across every open position at once. It
  is a construction, not a forecast: each position is a put credit spread whose worst case is (width − credit) × 100
  per contract, contracts are sized so the sum of worst cases of all open spreads never exceeds B, and nothing else
  is ever open. The worst month is −B by definition, and it needs SPY below every open short strike at expiry.
* No naked options, no calls, no rolling into a loss, no averaging down, no second underlying. SPY only (penny-wide
  quotes out of the money, no index-option tiers, American exercise handled by closing before 21 DTE).
* Mechanism on record: the variance risk premium — index put insurance is overpriced on average (CBOE PUT index ≈ +6 %/yr
  over T-bills, `research/passive_income/E1_vrp_options.md`); a spread keeps a fraction of that premium in exchange for
  the cap. Expected order of magnitude, disclosed before the test: +5–10 % of the deployed budget per 6-week cycle
  ≈ $200–450 a month on B, with occasional −B months (SPY −8 % or worse inside 45 days: 2024-08, 2025-04, 2020-03).

## Rules (the ladder)
* Cadence: every Monday at 10:00 ET (the first session of the week if Monday is closed) open ONE bull put spread on
  SPY with the expiration nearest 45 calendar days (any listed expiry 38–52 DTE, nearest to 45). About six spreads are
  open at any time.
* Strikes: sell the put whose Black–Scholes delta is nearest the target Δ ∈ {0.15, 0.20, 0.30} (delta from the 10:00
  mid, SPY 10:00 price, DTE, r = 3-month T-bill (constant 4.5 % if no series), q = 1.3 %); buy the put W ∈ {$5, $10}
  below it. Fill = the spread's mid at 10:00–10:05 ET minute bars minus $0.03 per leg (the SPY OTM put quote is
  $0.01–0.05 wide; disclosed as the slippage standard), regulatory fees $0.03/contract, commission $0 (Alpaca).
* Size: contracts = floor((B / 6) / ((W − credit) × 100)); zero if that is 0. Six ladders × (B / 6) = B.
* Management M ∈ {A: close at 50 % of the credit (buy back at mid + $0.03/leg) or at 21 DTE, whichever first, and a
  STOP when the spread's mark reaches 2 × credit (close at mid + $0.03/leg next minute bar after the daily close that
  triggered it → conservative: use the next session's 10:00 price); B: hold to expiry, settle at the intrinsic value
  from the SPY 16:00 price}. Early assignment: a short put in the money at 21 DTE is closed by rule A; under B the
  intrinsic settlement is the same economics; dividend ex-dates (quarterly) are report-only flags.
* IV gate G ∈ {none; sell only when the 45-DTE ATM implied vol ≥ 15 % (from the chain at 10:00), else skip the week}.
Cells: Δ (3) × W (2) × M (2) × G (2) = 24, cells 1,567–1,590. Report-only comparison lines: the naked short put at the
same Δ (the PUT-index equivalent, no cap, sized to the same premium) and buy-and-hold SPY over the same windows.

## Data and samples
Alpaca historical option bars (daily and minute) and the chain snapshots — available from 2024-01 (verified 9/27:
SPY250321P00540000 251 daily bars from 2024-01-23). Contracts pulled: every SPY put with strike within 20 % below spot and
DTE ≤ 60 on each Monday (resumable cache `research/options_vrp/opt_cache/`, LOST count, completeness gate). SPY prices:
Alpaca minute/daily bars. TRAIN = entries 2024-02-05..2025-06-30 (includes 2024-08-05 and 2025-04), VAL = entries
2025-07-07..2026-08-17 (last cycle can expire by 2026-09-25). No sealed TEST (disclosed: the forward PAPER run is the
test — Alpaca paper accounts carry options).

## Report per cell, per split
Cycles n, weeks skipped by the gate, mean net P&L per cycle in $ and in % of the cycle's own risk, monthly P&L series
on B (calendar months), mean monthly return on B, monthly Sharpe (annualised), green-month share, worst month (must be
≥ −B by construction — assert it), max drawdown on B, the share of P&L from the best 5 % of cycles, the per-cycle win
rate, mean holding days, the count of stop exits / 21-DTE exits / 50 %-profit exits, the 2024-08 and 2025-04 months
in isolation, and the two report-only comparison lines.

## Selection and pass bar (frozen)
On TRAIN: among the 24 cells pick the one with the highest monthly Sharpe subject to ≥ 12 cycles and green months
≥ 55 %. Read VAL for that ONE cell: mean monthly return on B ≥ +4 % (≈ $260/month at B = $6.5K), monthly Sharpe ≥ 1.0,
green months ≥ 60 %, ex-top-5 % of cycles positive, TRAIN same sign, worst month ≥ −B (assert), max drawdown ≤ 1.5 B,
and the cell's VAL beats buy-and-hold SPY's return on the same capital at risk on a per-unit-of-drawdown basis (report
both). The full VAL table is written for the record, labelled unselected. A neighbour check: the selected cell's Δ- and
W-neighbours on VAL same-signed.

## Independent check and consequences
Rebuild from this prose on the same cache (cycle-set Jaccard ≥ 0.99 on (entry date, strikes, expiry); P&L within $5 per
cycle on ≥ 99 %). Refuters: obtainability (the $0.03/leg standard vs the actual 10:00 quotes where minute quotes exist;
stop fills after a gap; assignment; the 2024-08-05 open), look-ahead (strike selection from the 10:00 mid only; the gate
from the chain at 10:00; the management trigger from the daily close acted on the NEXT session), data (missing bars on
illiquid strikes — a cycle whose legs lack bars is VOID and counted), statistics (24 cells selected on TRAIN, the
neighbour check, month concentration, the two spike months, the naked comparison). PASS → an options engine
(`trading/options_spread_engine.py`: Monday 10:00 job, multi-leg order, the guardrail ledger, the budget assertion
before every order, Telegram lines) on the PAPER account for 6 weeks (real quotes, zero capital), then live at B on the
owner's word. FAIL → defined-risk selling closes at this size with the table on record; the owner's "all in" then means
the undefined-risk version, which this desk will not run without a separate PREREG and his written acceptance of −20 %
months.

## Owner actions needed
Enable options trading on the Alpaca account at Level 3 (spreads) — the account currently shows no options level.
Confirm or change B = 10 % of equity.

## Not allowed
Adding strikes, widths, cadences or underlyings after seeing TRAIN; selecting on VAL; rolling rules; any position whose
worst case is not counted in B.

## Amendment 2 (2026-09-27 15:15 UTC, after the FAIL of cells 1,567–1,590; before any v2 number) — quotes, not prints
The v1 test priced legs from trade prints in a 5-minute window and VOIDed pairs without a print (which removed the
April-2025 losses from the selected cell). v2 (cells 1,591–1,598): the same ladder with (1) leg prices from OPRA
QUOTES — the NBBO mid at 10:00:00–10:00:30 ET on the entry Monday for both legs (fill = mid − $0.02/leg, the measured
half-spread if the quote is wider, i.e. sell at the bid, buy at the ask when |ask − bid| > $0.04), daily marks from the
15:59 ET NBBO mid of each open leg, exits at the next session's 10:00 NBBO (management A) or intrinsic settlement (B);
(2) a pair is never VOID because of a missing print — if a leg's quote is missing the cycle is VOID and counted, and the
VOID share per cell is a reported rail (> 10 % → the cell is VOID); (3) months without an exit count as $0; (4) the grid
reduced to what the premium supports after slippage: Δ ∈ {0.20, 0.30}, W = $10, M ∈ {A, B}, G ∈ {none, IV ≥ 15 %} = 8
cells; selection on TRAIN by monthly Sharpe with ≥ 12 cycles and ≥ 55 % green months (calendar months, zeros
included), one VAL read, the same pass bar; (5) the paper run may start in parallel once the account has Level 3 — it
is the forward test regardless of which cell v2 selects (the engine runs the TRAIN-selected cell).
Amendment 2a (15:30 UTC): OPRA quotes are NOT available on the data plan (GET /v1beta1/options/quotes → 404; trades →
200). v2 therefore prices each leg from TICK TRADES: entry = the last trade of each leg in 10:00:00–10:00:30 ET on the
entry Monday (a leg with no trade in 10:00:00–10:05:00 → the cycle is VOID and counted; the VOID rail stays at 10 %),
fill = that price − $0.03/leg for the sold leg and + $0.03 for the bought leg; sensitivity rails at $0.05 and $0.10 per
leg reported beside; daily marks from the cached option daily closes (liquid 20–30-delta legs); management-A exits at
the last trade in 10:00:00–10:00:30 of the next session (same VOID rule → exit at the daily open as the fallback,
counted). Everything else as Amendment 2.
