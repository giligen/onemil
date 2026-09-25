# Route E1 — Volatility Risk Premium via Index Options (SPY/QQQ/XSP)

Written 2026-09-25. Skeptic pass on selling index puts / spreads for "passive constant income" on the
$65,083 live account. Every number below is tagged with its source: **[PUB]** published index/paper,
**[LIVE]** pulled from our own Alpaca account/data just now, **[CALC]** arithmetic on the two.

## 1. The mechanism
Sell (write) out-of-the-money SPX/SPY/QQQ puts, or buy-cap them into a put credit spread / iron condor,
monthly or weekly. This harvests the **variance risk premium (VRP)**: index implied vol trades above
realized vol on average, more so for OTM puts (the post-1987 "skew" — crash insurance is priced rich).
It is a genuine, heavily replicated risk premium, not an inefficiency — which also means it is not free:
the premium is compensation for occasionally paying out big on a crash.

## 2. What the published record says
- **CBOE PUT Index** (sells 1-month ATM-ish SPX put, 100% cash-secured, monthly roll), Jul 1986–Aug 2023:
  **+9.40%/yr annualized vs S&P 500 +9.91%/yr**, stdev **10.26% vs 15.38%**, max drawdown **-32.66% vs
  -50.96%**. [PUB — CBOE/Ennis Knupp study, cdn.cboe.com/resources/education/research_publications/PUTIndexEnnisKnupp.pdf;
  corroborated by en.wikipedia.org/wiki/CBOE_S%26P_500_PutWrite_Index]
- **Q1 2020 crash window**: PUT Index **-28.92%** peak-to-trough (Feb 19–Mar 23 2020, 5 weeks), **-20.68%**
  for the full quarter. [PUB — Gateway/CBOE Active Index-PutWrite Composite Commentary Q1 2020,
  gia.com/wp-content/uploads/2022/03/Active-Index-PutWrite-Composite-Commentary-Q1-2020.pdf]. This is the
  real cost of "one bad month" on a fully-collateralized program — not a tail scenario, it already happened.
- **CNDR** (SPX iron condor, short ~20-delta / long ~5-delta wings, monthly, DEFINED risk) worst
  peak-to-trough **monthly** drawdown **~19%**, vs **~47%** for BXM (covered call) and **~51%** for the
  S&P 500 itself. [PUB — CBOE Benchmark Indices fact sheet, cdn.cboe.com/resources/indices/documents/benchmarks-fact-sheet.pdf].
  I could not find a CBOE-published long-run *return* series for CNDR as prominent as PUT/BXM's in the
  time budget — flagging that gap rather than guessing a number.
- **Aug 2024**: VIX spiked to levels last seen 2021, but reverted to a mid-teens handle within ~2 weeks —
  a real stress episode, not obviously a single "worst month" in the PUT/CNDR series. [PUB — CBOE Index
  Insights, Aug 2024, cboe.com/insights/posts/index-insights-august-2024]
- **Academic point** (VRP/tail literature): short-vol P&L is structurally **negatively skewed**; the
  Sharpe ratio of put-selling/short-vol strategies is explained almost entirely by their negative skew, not
  by a separate "edge" term — i.e., the return **is** the tail-risk compensation, and in the worst historical
  episodes put sellers have lost multiples of a normal month's premium in one print. [PUB — arXiv:1409.7720
  "Risk Premia: Asymmetric Tail Risks and Excess Returns"; Quantpedia "Volatility Risk Premium Effect",
  quantpedia.com/strategies/volatility-risk-premium-effect]

## 3. What our own account and data say right now (live probe, 2026-09-25 ~11:52 UTC)
- **Account**: equity **$65,083.14**, cash **$65,083.14** (flat, no open positions),
  `options_approved_level = None`. [LIVE — `TradingClient.get_account()`]. **We are not approved for any
  options level.** Per Alpaca support docs, approval is a real application (up to 2 initial submissions,
  60-day wait to reapply if declined) — this alone means E1 cannot place a single live contract today.
  [PUB — alpaca.markets/support/what-option-levels-or-tiers-do-you-provide]
- **Level required**: cash-secured puts = **Level 2**; put spreads / iron condors = **Level 3** (higher bar).
  [PUB — alpaca.markets/learn/level-3-options-trading, docs.alpaca.markets/us/docs/options-level-3-trading]
- **Live quotes**: SPY **$770.55**, QQQ **$746.65**. [LIVE — `StockHistoricalDataClient.get_stock_latest_trade`]
- **Real SPY put chain**, expiry 2026-11-20 (56 DTE — the nearest clean monthly; Oct 16 is only 21 DTE, so
  there is no exact 30-45 DTE monthly on the board right now): **30-delta strike = $750**, mid premium
  **$9.565** (bid 9.52 / ask 9.61) → **$956.50 collected per contract** against **$75,000 of strike notional
  secured**. [LIVE — `OptionHistoricalDataClient.get_option_chain`]. That's 1.28%/56-day-cycle on secured
  notional → **[CALC] ≈ 8.3%/yr run-rate on that leg alone**, which lines up with the PUT Index's realized
  9.4%/yr average (a decent cross-check that the live quote isn't an outlier).
- **Capital-sizing problem, found directly from these prices, not assumed**: one cash-secured SPY put ties
  up **~$75,000** — *more than the entire account*. I also pulled **XSP** (Alpaca's Mini-SPX index option,
  went live on Alpaca's Trading API this month) expecting it to be 1/10 the size like SPX/10 implies — it
  is **not** meaningfully smaller: XSP ≈ SPX/10 ≈ SPY's price (strikes clustering $730-810 against SPY at
  $770.55), so a single XSP put also secures **~$75-80K**. [LIVE — `get_option_chain("XSP", ...)`, 319
  contracts returned, strikes confirm]. **Conclusion: "size to ~50% of $65K notional" ($32.5K) is not
  achievable with a single naked cash-secured put on SPY, QQQ, or XSP — one contract already over-collateralizes
  past 100% of equity.** A true cash-secured program on this account is capped at **zero contracts** at
  today's prices unless real margin/leverage is used instead of full cash security — which turns this from
  "sell insurance, collect premium" into "sell insurance, levered," a materially different and riskier
  product than the PUT Index return series describes.
- **What would actually fit $65K**: (a) **defined-risk spreads** (put credit spread / iron condor) where
  margin = spread width, e.g. a $5-wide SPY put spread ≈ $500/contract — lets you size to 50% notional
  properly and hard-caps the loss, but needs Level 3 (harder approval) and collects only a fraction of the
  naked premium; or (b) a cheaper single underlying (e.g. IWM ~$240/share, ~$24K/contract) — departs from
  the SPY/QQQ mandate and swaps large-cap index risk for small-cap-index risk.
- **Cost/liquidity**: SPY/QQQ/XSP quotes are penny-to-nickel wide even far OTM (e.g. $9.52/$9.61 on the
  750 put ≈ 0.7% of premium) — nothing like the small-cap slippage that killed ORB/HOD/BF ideas. Alpaca
  options fee is **$0.65/contract** plus pass-through regulatory fees (ORF **$0.02295**/contract,
  TAF **$0.00329**/contract, sells only). [PUB — files.alpaca.markets/disclosures/BrokFeeSched.pdf,
  alpaca.markets/support/regulatory-fees]. On a ~$956 premium that's **[CALC] ~0.07% total cost — cost is
  not the obstacle here.**
- **New-product caveat**: SPX/XSP/VIX/DJX index options only moved from paper to live on Alpaca's Trading
  API **this month** (Sept 2026). [PUB — alpaca.markets/blog/alpaca-launches-index-options-via-trading-api].
  Our data snapshot also returned `greeks=None` for every XSP quote (present and populated for SPY/QQQ
  equity options) — live delta-targeting on XSP needs a workaround (compute delta locally, or use SPY as
  the practical underlying) until/unless the data plan populates index-option greeks.

## 4. Verdict on "passive" and "constant"
- **Not passive** in a set-and-forget sense: monthly/weekly strike-and-expiry selection, order placement,
  early-assignment monitoring (SPY/QQQ are American-style; XSP/SPX are European and cash-settled, avoiding
  assignment — a real point in their favor *if* the capital problem above gets solved), and roll/close
  logic before expiry are a recurring build, not a flag flip. It can run unattended once built (this stack
  already runs via systemd), but it is new code, not a config change.
- **Not constant**: by construction, and confirmed in every cited index, this is many small green
  months and rare-but-real double-digit-percent red months — PUT Index -20.7% for Q1 2020, CNDR's own
  worst *monthly* drawdown -19% even with defined risk. This is a positive-skew-in-*frequency*,
  negative-skew-in-*size* income stream — the opposite shape from what "constant" usually means.
- The premium is real and one of the most replicated risk premia in finance, but it is **compensation for
  crash risk, not alpha** — it will not decouple from a market crash. It behaves like a leveraged-beta
  product wearing an "income" label.

## 5. Numbers the owner asked for (calm / normal / stressed, this $65K account)
Base rate: the live 30-delta/56-day SPY quote (1.28%/cycle ≈ 0.83%/30-day-equivalent on secured notional),
cross-checked against the PUT Index's 37-year realized average (9.4%/yr on 100% notional).

- **Naked cash-secured puts at "50% notional" ($32.5K), as specified in the brief: not currently
  achievable.** One contract on SPY/QQQ/XSP secures $75-80K — above total equity. This route is closed
  until the account is much bigger, or Level-3 spreads are used instead.
- **If run as defined-risk put spreads at ~50% notional ($32.5K in spread-width margin)**: gross premium
  collected is a fraction of the naked number (BXM-vs-CNDR gap suggests roughly 1/3–1/2), so realistically
  **[CALC] ~+$100 to +$170/month calm**, less in normal/stressed vol, with loss capped near the spread
  width per cycle rather than open-ended.
- **Calm/normal months, naked (only meaningful if capital problem is solved — e.g. bigger account or
  smaller underlying)**: **[CALC] roughly +$200 to +$450/month** on $32.5K notional, extrapolated from the
  real $956/contract/56-day SPY quote and cross-checked against the PUT Index's long-run average.
- **Worst plausible month, naked, at 50% notional**: **[CALC] approximately -$6,500 to -$9,500 in a single
  month** (scaling the PUT Index's realized -20.7% Q1-2020 quarter and -28.9% peak-to-trough to $32.5K) —
  **15-25+ months of calm income erased in one event.** This is the number that matters most here.
- **worst_month_usd (headline, full $65K equity, the size the account is actually forced into if it trades
  naked SPY/QQQ/XSP puts at all — see §3): approximately -$13,000 to -$21,000** in a Q1-2020-style month.

## 6. Recommendation
1. File the options application (Level 2 minimum, Level 3 if spreads are wanted) today — it is the actual
   bottleneck, costs $0, takes days-to-weeks. **setup_needed**, not a research question.
2. While waiting: paper-trade (Alpaca paper accounts get Level 3 automatically) one full SPY 30-delta
   put-credit-spread cycle end-to-end through this stack — chain selection, order, monitor, close/roll —
   to prove the automation. $0 cost, one ~30-45 day cycle to first real read.
3. **Do not go live naked/undefined-risk on this account size** — the capital math in §3 rules it out
   structurally, not as a matter of risk appetite. If/when going live, go live with **defined-risk spreads**,
   sized so the worst-case loss per cycle fits the cadence bar's weekly-P10 tolerance, at real minimum size
   — consistent with the project's "live exploration tier" rule (positive point estimate + mechanism +
   bounded downside + resolves in a quarter → minimum size live; frequency, not confidence, is the gate).
4. Be honest with the owner: this is a well-known, 40-year-old public risk premium, already fully priced
   and published (CBOE PUT/CNDR indices) — it is not proprietary edge, it is beta with a different shape.
   It can plausibly add low-hundreds-of-dollars/month at a size this account can actually field, with real
   five-figure-dollar downside months. "Passive" needs a caveat (real but automatable maintenance burden);
   "constant" is the wrong word — "less lumpy than momentum, still lumpy, and occasionally very red" is
   the accurate one.

## Sources
- CBOE PUT Index study (Ennis Knupp): https://cdn.cboe.com/resources/education/research_publications/PUTIndexEnnisKnupp.pdf
- CBOE S&P 500 PutWrite Index (Wikipedia): https://en.wikipedia.org/wiki/CBOE_S%26P_500_PutWrite_Index
- Gateway Active Index-PutWrite Composite Commentary, Q1 2020: https://www.gia.com/wp-content/uploads/2022/03/Active-Index-PutWrite-Composite-Commentary-Q1-2020.pdf
- CBOE Benchmark Indices fact sheet (BXM/CNDR/PUT drawdowns): https://cdn.cboe.com/resources/indices/documents/benchmarks-fact-sheet.pdf
- CBOE Index Insights, August 2024: https://www.cboe.com/insights/posts/index-insights-august-2024
- "Risk Premia: Asymmetric Tail Risks and Excess Returns" (arXiv:1409.7720): https://arxiv.org/pdf/1409.7720
- Quantpedia, Volatility Risk Premium Effect: https://quantpedia.com/strategies/volatility-risk-premium-effect
- Alpaca options levels/tiers: https://alpaca.markets/support/what-option-levels-or-tiers-do-you-provide
- Alpaca Level 3 options: https://alpaca.markets/learn/level-3-options-trading
- Alpaca brokerage fee schedule: https://files.alpaca.markets/disclosures/BrokFeeSched.pdf
- Alpaca regulatory fees: https://alpaca.markets/support/regulatory-fees
- Alpaca index options live launch (Sept 2026): https://alpaca.markets/blog/alpaca-launches-index-options-via-trading-api/
- LIVE: this account's `TradingClient.get_account()`, `get_stock_latest_trade()`, `get_option_chain()` — probed 2026-09-25 ~11:52-11:56 UTC, scripts left in the scratchpad (not committed).
