# Passive income: the judge's plan (2026-09-25)

Inputs: route reports E1–E5 in this folder, plus two checks the judge ran this session (scripts are in the
session scratchpad, not committed): ORB backtest vs live, paired by (symbol, day), and monthly correlations
between sleeves. Tags: **[E#]** = route report, **[J]** = judge's own computation today, **[PUB]** = published.
Nothing here is committed and no live change has been made. Every live step below needs the owner's word.

## 0. The answer in six lines

1. **$2–3K a month of constant, passive income is not attainable on $65K.** It needs 37–55 %/yr with no
   drawdown. No documented strategy does that, and nothing in 9 months of our own work does either.
2. **Constant and passive, attainable today: about $200–255/month** from T-bills (BIL/BOXX), with no red months.
3. **Every "income" product that pays more is short-crash beta in a different wrapper.** That covers put-writing,
   covered-call ETFs, short VIX, and funding carry. Its worst month wipes out 5–25 months of its income. On worst
   month per dollar of income, **plain equity beta (SPY/IVV) matches or beats all of them**, with 100 years of
   evidence and nothing to build.
4. **ORB is the only line that could add $0.5–1K/month, but it has never made money live.** It lost **−$5,281
   on 123 live fills** [J]. Over the same months the current-config backtest made +$3,100. The fills that live
   and the backtest shared match well. The gap came from live taking different trades (only 29 % pick parity).
   ORB earns a place in the plan only after it proves pick parity live (Experiment 2).
5. **HOD-break is not the edge.** Across 15,656 live-config signals it earned about −0.04 R gross, and every
   order-flow bucket was about 0 gross. The forward dry run lost −0.454 R per trade. It stays as a free dry-run
   ledger with a pre-committed trigger to reopen (§6), and gets no capital.
6. **What really raises the ceiling is capital, not research.** Every $10K added is +$38/month at the floor.
   $2K/month passive needs roughly $240K (paying out the full 10 % expected return) to $400–480K (a sustainable
   5–6 % payout).

## 1. Ranking: expected income per dollar of worst plausible monthly loss

Excess means income above T-bills. Capital sitting in T-bills earns about $37 per $10K per month, so a risky
sleeve has to beat that. The ratio is excess income divided by the worst plausible month. Evidence grades:
A = decades of published history plus our own data; B = our own backtest with an independent check; B− = a
single-pass backtest or a proxy.

| # | Route | Sleeve / unit | Expected $/mo | Excess over T-bills | Worst plausible month | Ratio | Evidence | Implementable | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **T-bills / BOXX** (E4) | per $10K | $36–39 | 0 (this is the floor) | ≈ −$1 (BIL worst −0.01 %; BOXX never red) | unbounded | A | now | **DO: floor** |
| 2 | **ORB, R = $375** (E5 + [J]) | intraday overlay, uses buying power, not capital | backtest: mean $838, median $574 (out-of-sample, 15 mo). **Live: −$1,056 average over 5 months** | same as expected | backtest −$763; **live −$3,187** (Jun-26) | backtest 1.1; planning ≈ 0.09 at $290/−$3.2K; live < 0 | B (backtest rebuilt independently; live contradicts it) | now (running) | **KEEP at R = $375, count $0 until Exp 2 passes** |
| 3 | **Equity beta, SPY/IVV** (E4 table) | per $10K | ≈ $83 (10 %/yr over the long run) | ≈ $45 | −$1,700 to −$2,200 (Oct-2008 −17 %, Oct-1987 −22 %); in-sample −$924 | 0.021–0.026 (long run); 0.074 in E4's 2022–26 sample | A | now | **DO: small sleeve** |
| 4 | Covered-call ETFs (JEPI/QYLD/XYLD) (E4) | per $10K | $55–72 total return (**the distribution is not the return**) | $20–35 | −$640 to −$890 in-sample (all Sep-22); about −$1,300 to −$1,700 in a 2008/2020-type month | 0.012–0.027 | A−/B | now | dominated by #3: same crash, less upside |
| 5 | Put-write / credit spreads (E1) | per $10K notional | PUT Index excess over T-bills ≈ 6 %/yr → about $50 | ≈ $50 | −$2,100 to −$2,900 (PUT Q1-2020 −20.7 % / −28.9 % peak-to-trough) | 0.017–0.024 | B (published index) | after options Level 3 approval plus new code; at $65K only spreads fit | dominated by #3: same risk, needs approval and a build |
| 6 | Short vol (E2): **long SVXY** | per $10K | $146 in-sample (2019–26), about $73 with a 50 % forward haircut | $110 → ≈ $36 | −$3,900 in-sample (Mar-20); about −$5,000 if Feb-2018 repeats at −0.5× | 0.028 in-sample → ≈ 0.007 forward | B− (one 7.7-yr draw that misses 2018) | now | dominated by #3 |
| 6b | Short vol (E2): short UVXY, 15 % cap | $9,762 | $578 | ≈ $540 | **−$15,217** (−156 % of the allocation); weekly rebalancing still −$14,648 [J] | 0.036 excess (0.038 gross, identical to SVXY long's 0.038) ⇒ **it is simply about 4× SVXY** | B− | after margin/locate checks | worse version of #6: unbounded loss, recall risk, buy-in risk |
| 7 | Leveraged-ETF pairs (E2) | SPXL/SPXS, $65K gross short (the only pair shortable today) | $36; median −$95; 39 % green months | ≈ $36 | −$2,563 | 0.014 | B− | after setup | **DEAD**: no income; the other 3 pairs are not shortable |
| 8 | Crypto funding carry (E3) | $30K | normal $85 = **3.4 %/yr, below T-bills** | **−$25 to −$33** | −$15K venue/liquidation event | < 0 | B− (Binance proxy, not the US venue) | new venue plus new cross-venue code | **DEAD**: T-bills beat it on the same capital |
| — | HOD-break | — | gross ≈ 0; forward dry run −0.454 R per trade | < 0 | — | — | A (own data, 1,250+ cells) | dry run only | free dry-run ledger (§6) |

**Where the judge corrected the route reports** (their caveats, read adversarially):

- **E1.** Its "8.3 % premium matches the PUT Index's 9.4 %" cross-check compares two different things. The 8.3 %
  is gross premium before any payouts. The PUT Index's 9.4 % is net of payouts and includes about 3 %/yr of
  T-bill interest. E1 also sizes its spread worst case on notional ($32.5K × −20 to −29 %). A spread's real
  crash loss is close to its full width, so the true worst case depends on how the spreads are sized. Treat
  E1's income figures as upper bounds.
- **E2.** Weekly rebalancing tames the SOXL/SOXS tail but **not UVXY's**: the worst calendar month is −$14,648
  weekly vs −$15,217 monthly [J]. UVXY short and SVXY long have **the same mean/worst ratio (0.038)**, so the
  short adds borrow, recall and unbounded loss for no extra edge. The sample starts in 2019 and misses Feb-2018
  (XIV was terminated).
- **E3.** At 1× the capital sits in two places (the spot position and the perp collateral), so the "normal"
  $85/mo on $30K is 3.4 %/yr. That is below the 4.3–4.7 % that BIL pays on the same money, before any venue risk.
- **E4.** The direction is correct. Its capital table pays out the full 8–12 % expected return, with no allowance
  for inflation or a smoothing buffer. A sustainable payout is about 5–6 %, which means roughly 1.6–2× the capital
  E4 lists.
- **E5.**
  - (a) Its share formula gives 2.72× live sizing, so the capacity wall sits nearer R ≈ $1,000 than R ≈ $375.
  - (b) Its gross 0.286 R (proxy sizing) vs the book's 0.105 R is never reconciled, so the capacity table is
    directional only.
  - (c) The "extra 72 bps latency" conflicts with its own −14 bps model-vs-fill figure. On the 45 fills that
    live and the backtest share, live trails the backtest by **−0.12 pp per trade on average (median +0.08 pp)
    with 91 % sign agreement** [J]. Execution matches.
  - (d) **It never looked at live P&L.**
  - (e) Its first experiment (the add-on pools) was **already closed on 9/24**. Both pools are dry-only; p30 made
    −0.21 R per fill out of regime (`research/orb_seed_wide/PREREG_LIVE_UNION.md`).
  - (f) Without the top 5 % of fills the book averages **+0.006 R per fill** [J]. By CLAUDE.md rule 5 it is shaped
    like a lottery ticket. The top single fill is 10 % of the total.

## 2. The honest ceiling on $65K

| Tier | $/month | Worst month | What it takes |
|---|---|---|---|
| Constant and passive | **$200–255** | ≈ $0 | all T-bills. Falls about $54/mo for every 1 pp the Fed cuts. |
| Passive and near-constant (book core, §3) | **≈ $285 expected** | about −$2,000 | T-bills plus a $10K beta sleeve |
| Plus ORB, **if** the backtest holds live | **+$575 to +$840** (backtest median/mean at R = $375) | −$763 in the backtest, −$3,187 live so far | Exp 2 passes. Then the above-water ramp takes 6–12 months. |
| ORB's ceiling, even if proven | about $0.7–1.5K | about −$3–6K | R ≈ $750–1,500 before market impact eats the edge (E5, corrected for (a)) |
| **Not attainable** | $1K+ constant with no red month; $2–3K passive | — | 18–55 %/yr at zero drawdown does not exist. The "income" products that look like it are short-crash beta. |

**Plainly:** a realistic good year for this book is **$6–13K** (about $500–1,100/mo on average), lumpy, with one or
two red months. The part that is truly constant and passive is **about $250/month**. The honest route to
multi-$1,000s is growing the capital, plus ORB if it proves itself live.

## 3. The proposed book (three sleeves, $65,083)

| Sleeve | Capital | Calm month | Normal month | Stressed month | Worst plausible | What runs it |
|---|---|---|---|---|---|---|
| **A. Floor + payout buffer**: BIL (BOXX only if tax status favours it) | **$55,000** | +$205 | +$205 | +$200 (T-bills rise in a crisis) | ≈ $0 | monthly systemd timer, **unattended** |
| **B. Equity beta**: IVV/VOO (not SPY if the owner trades SPY by hand, so positions never mix) | **$10,000** | +$150 to +$300 | +$80 (long-run mean) | −$800 to −$1,200 | **−$2,200** | same monthly timer; rebalance band ±10 %; **unattended** |
| **C. ORB at R = $375**: the existing engine, no config change | $0 of capital (intraday buying power) | +$1,600 (backtest P75) | **$0 for planning**; +$575 if the backtest holds | −$495 (backtest P10) | **−$3,200** (the live Jun-26 month) | onemil-trader; automated but **not passive** (weekly refit and alarm triage) |
| **Book** | $65,083 | ≈ **+$2,000** | ≈ **+$285** (plan) / +$860 (if ORB's backtest holds) | ≈ −$1,300 | **≈ −$5,200** (B and C at their worst in the same month, 8 % of equity) | |

- **Correlation.**
  - A has about zero correlation with both B and C.
  - B vs C: **−0.10** over 21 months [J] (ORB backtest vs SPY monthly). That is not distinguishable from 0.
  - ORB vs a short-vol sleeve: **−0.44** [J]. ORB's best backtest months were the volatility spikes (Mar-25,
    Apr-25, Mar-26). It looks like a long-volatility book. This is a 21-month backtest pattern, a hypothesis
    rather than a claim, and it is one more reason not to add short vol next to it.
- **Payout rule, which is what makes the income constant.** The owner withdraws a fixed **$250/month**, funded
  from A. Gains and losses in B and C flow into A's buffer. A is 18 years of payouts deep, so no red month can
  interrupt the payout. The stack cannot move money out of Alpaca. The timer only makes sure the cash for the
  payout is ready.
- **BF** stays at its $150 cap under its own ramp. It counts as $0 and is not an income sleeve. **Not in the book:**
  options, UVXY/SVXY, leveraged-ETF pairs, crypto carry, covered-call ETFs (reasons in §1).
- **A diversifier to check later:** managed futures (DBMF, +21.5 % in 2022 while SPY fell 18.6 % [E4]) could take
  half of B. It is not assumed until it has its own check.

## 4. First two experiments (PREREG: every number fixed before any data comes in)

### Exp 1: Passive core live (sleeves A + B). Cost $0 (≈ 1 bp spread). About 1 week to build and rehearse, then 30 days to score.

- **Desk pre-checks** (1 day, all must pass):
  - Alpaca's cash interest on this account. If idle cash already earns at least BIL's yield, A becomes "hold cash".
  - Alpaca's maintenance margin for BIL and IVV.
  - Intraday buying power with A + B held must be ≥ 2× the peak concurrent ORB + BF notional over the last 60
    sessions, plus the owner's largest manual position in the last 60 days.
  - Confirm the owner has not traded BIL or IVV in `trades`/positions history.
- **Build.** One module plus a monthly timer. Orders are tagged by client_order_id. StopMonitor, BF, ORB, HOD
  and the reconcile logic must treat these positions as their own sleeve, never as owner positions and never as
  positions to exit. Unit, integration and system tests. A weekend boot rehearsal on the exact ExecStart.
  Then a **$1,000 live probe** before full size.
- **Pass after 30 days, all of:**
  - 0 orders from any book against A/B positions.
  - 0 "unknown position" or owner-interference ERROR/WARNING lines.
  - 0 ORB/BF rejections for buying power (baseline: the prior 30 days).
  - A's accrual ≥ 0.33 % (≥ 4.0 %/yr) and A's worst day ≥ −0.05 %.
  - The daily brief reports sleeve P&L separately from trading P&L.
- **Kill.** Any interference incident: liquidate A/B to cash the same day, revert, and document.

### Exp 2: ORB live pick parity and live edge (the only growth lever). Cost $0 extra (R stays $375, under the existing ramp). Runs to 40 live fills or 2026-12-31, whichever comes first.

- **Why.** Execution already matches on shared fills. The live loss came from **different trades**: 109 of the
  154 backtest fills since May were never taken live, and the 78 live fills the backtest did not have lost
  −$2,252 [J]. The pause, the corpse-gate defect and the 3-vs-8-slot and catalyst-veto history explain part of
  this, not all of it.
- **Step 0 (desk, 1 day).** Split the −$5,281 by config era: catalyst veto on/off, slots, corpse-gate window,
  pause. Record how much of it was a known defect and how much is unexplained.
- **Measured daily from 2026-09-21 on:** backtest picks on the entered-inclusive features at the live config,
  compared with live orders and fills, by (symbol, day).
- **Pass (all of):**
  - (P1) pick parity: ≥ 80 % of backtest fills have a live fill, and ≥ 80 % of live fills are backtest fills.
  - (P2) execution: mean live − backtest pnl_pct on paired fills ≥ −0.20 pp (today −0.12 pp).
  - (P3) at 40 fills: live mean R > 0, realized stage P&L ≥ 0 (the above-water rule), and no week ≤ −4 R.
- **Kill.** Live mean R ≤ −0.15 at 40 fills, or pick parity < 60 % after 20 backtest fills without a fixable
  defect. Either one pauses ORB, and it is recorded as a live null with its MDE.
- **Power, stated in advance.** SD is 0.607 R per fill [J], so SE at 40 fills is 0.096 R and the MDE is about
  0.27 R. **40 fills is a parity and sanity gate, not proof of edge.** Telling +0.105 R apart from 0 takes
  about 262 fills: about 12 months in a hot tape, years in a cold one.
- **Pass** means ORB enters the book at its backtest median, and the ramp may advance under its existing rules.
- **Fail** means sleeve C is removed, and the book is A + B, about $285/month.

## 5. What the last 9 months proved (so the owner is not asked to learn it again)

- **Measured honestly, small-cap intraday momentum carries almost no edge.** "Honestly" means a point-in-time
  universe, causal features, obtainable fills and measured per-trade NBBO cost. The programme ran about 1,420
  pre-registered cells:
  - HOD-break: gross ≈ 0 on 12–16K signals; the order-flow lift turned out to be cost.
  - Ignition: nothing found starting from zero.
  - Index ORB, overnight SPY/QQQ/TQQQ, post-earnings drift: all fail.
  - Bull flag: +0.10 R on 26 out-of-regime trades, carried by one trade; its book halves under measured cost.
- **Cost decides.** A band cost table turned a published +0.3 R book into −0.62 R. R must be well above about
  0.5 % of price.
- **One line survived: ORB.** It earns about +0.1 R per fill in and out of regime; the regime changes how often
  it trades, not the per-fill result. It is carried by its tail (+0.006 R without the top 5 %), limited in
  capacity, and **−$5.3K live so far**.
- **The checking process is not the problem.** Independent rebuild, causality trace, price-scale check, fill
  realism, tail test and cell count are what caught five false positives. Their honest output is mostly "no".
- **Passive, constant income is a payout rule applied to capital × return, not a property of a strategy.** On
  $65K the return side is capped by what exists in the world. The remaining levers are capital, and an ORB that
  proves itself live.

## 6. Closed here, and what would reopen each (pre-committed)

| Route | Reopen only if |
|---|---|
| HOD-break | the free dry-run ledger (`scripts/hod_dry_ledger.py`) shows ≥ +0.15 R net per trade over ≥ 150 booked trades (the old causal-filter ship bar); then a new PREREG, never capital first |
| Crypto carry (E3) | trailing 2-quarter BTC/ETH funding ≥ 15 %/yr (≈ 7.5 %/yr on capital at 1×, about 3 pp over T-bills) **and** a US perp venue with 12+ months of stress history |
| Options VRP (E1) | the account reaches ≥ $250K, so cash-secured index puts fit without leverage, **and** it still beats sleeve B's ratio after costs |
| Short vol / leveraged-ETF pairs (E2) | never as income. Same beta as sleeve B with a worse tail. |
| Research spend | no new intraday signal mining until Exp 2 resolves. Any new line needs a published mechanism plus capacity above $65K. |

Evidence files: `E1_vrp_options.md` … `E5_orb_capacity.md`, `E2_*.csv` (this folder); `research/thermo/book_2025_26.csv`;
`data/trades.db` (`strategy='orb'`, 123 filled rows, read-only); `research/orb_seed_wide/PREREG_LIVE_UNION.md`;
`docs/cadence_bar.md`.
