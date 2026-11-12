# HOD — the WIDE external idea search (owner 2026-11-12: "go wide, search for ideas"; steer: "HOD related if possible due to its great frequency")

Brief: `research/hod_entry/IDEAS_WIDE_BRIEF_20261112.md`. Not a repeat of the 60 in `IDEAS_HOD_20261112.md` (referenced
below as "H#n" where an idea extends one of them). Every idea below uses the HOD event stream (~93 arms/day, ~50
fills/week, breaks, retests, failures, the armed-but-unfilled set) as its **trigger, universe or counterparty**. The
stream's FREQUENCY is the asset; each Mapping line says what share of it the idea uses.

Reference numbers used in every cost line: R = stop distance ≥ 1.5 % of price ⇒ 15–40 bps round trip = 0.10–0.27 R;
measured gross +0.075 R/fill, cost ≈ 0.15 R (chase 8 bps, stop slip 13–35 bps on 41 % stop-outs, spread).
Prior: STRONG = peer-reviewed or replicated; MEDIUM = practitioner/working paper with numbers; WEAK = anecdote.
Source marker: [s] = found/read in this session's web search; [m] = cited from memory, not re-fetched — verify before a PREREG.
Caveat that applies to every "new tape data" idea: 2,400+ cells say this population carries no per-fill information
in every feature tried; a new-data split is only worth a cell if the data is genuinely new (sub-second, venue/condition
flags, short-supply), and it inherits the multiplicity count.

---

## A. Be the counterparty to the HOD chaser (liquidity provision on the event)

**A1. Overreaction fade ladder above the break** — MEDIUM
Source: Zawadowski, Andor & Kertész, "Short-term market reaction after extreme price changes of liquid stocks", Quant. Finance 2006 [s].
Mechanism: after an extreme 15-min move, behavioural over-shoot ≈ 1 % reverts within 10–60 min; the HOD chaser buys the over-shoot.
Evidence: NYSE/Nasdaq liquid stocks, significant ~1 % reversal after both up- and down-extremes; volume and spread normalise in 30–60 min.
Mapping: every FILLED break (≈ 50/wk): rest sell limits at HOD +1/+2 % on ETB names, cover at 30 min; minute paths of 15,656 signals give the over-shoot distribution free. Must reconcile with 1,736 (placebo shorts ≈ −0.6 R under tight stops — use a time exit, not a 1.5 % stop).
Cost: passive entry earns the spread; 1 % target on a time exit ≈ 0.6 R vs 0.1 R cost — survives IF the over-shoot exists in small caps.

**A2. Price-pressure bid under the failure flush** — STRONG
Source: Hendershott & Menkveld, "Price pressures", JFE 2014 [s]; Nagel, "Evaporating liquidity", RFS 2012 [s].
Mechanism: an intermediary absorbing a liquidation shades price to unload inventory; the transitory component reverts with a half-life.
Evidence: NYSE, mean price pressure 49 bps, half-life 0.92 days; smallest quintile 118 bps with half-life 2.11 days.
Mapping: the FAILED/UNFILLED breaks (the −0.57 R set, ~40–50 % of arms) + the 41 % stop-outs: a resting bid at failure level −1 ATR(5m), hold 1–2 days. Distinct from H#10 (VWAP reclaim = momentum re-entry); this is buying the dump that our own stops feed.
Cost: 118 bps reversion vs 20–40 bps RT on a multi-day hold ⇒ survives on paper; overnight gap risk is the bill.

**A3. VIX-scaled liquidity-provision sleeves** — STRONG
Source: Nagel 2012 RFS [s].
Mechanism: returns to supplying liquidity (short-term reversal) are priced by intermediaries' risk capacity, so they rise with VIX.
Evidence: reversal-strategy returns highly predictable by VIX; conditional Sharpe spikes in turmoil; intraday reversals carry most of it.
Mapping: size H#1/H#2/A1/A2 (the provision legs on the stream) by VIX quintile — NOT a filter on the long break (the SPY/vol-regime filter on the break is closed).
Cost: neutral; changes when the provision legs carry size.

**A4. Maker-rebate routing for the passive legs** — STRONG (fee schedules are rules, not estimates)
Source: SEC Rule 610 access-fee cap (30 mils/share) and exchange maker-taker schedules [m]; Battalio, Corwin & Jennings, "Can brokers have it all?", JF 2016 [m].
Mechanism: exchanges pay ~20–32 mils/share to resting liquidity; on a $5 stock that is 4–6 bps PER SIDE, i.e. 0.03–0.04 R on a 1.5 % stop.
Evidence: published fee schedules; BCJ show the routing choice moves limit-order fill quality materially.
Mapping: every resting leg the HOD book could have (H#1 coil bid, H#2 retest bid, H#4 exit into strength, A1, A2, the 2 R target) ≈ 100 % of fills × 1–2 legs. Alpaca pays no rebates (PFOF model) — needs a direct-access broker (IBKR Pro tiered / Lightspeed / DAS); the break entry then pays taker fees, so route only passive legs there.
Cost: turns 0.03–0.08 R/fill of cost into income on the passive side; a measured number, not an edge claim.

**A5. Route resting HOD bids to low-rebate / inverted venues for fill quality** — STRONG
Source: Battalio, Corwin & Jennings, JF 2016 [m].
Mechanism: high-rebate venues have the longest queues, so a resting order fills last and mostly when informed flow runs it over (adverse selection); inverted venues fill first.
Evidence: BCJ find limit orders routed for rebates get lower fill rates and worse post-fill markouts.
Mapping: the retest/coil bids at the HOD level (H#1/H#2) — the "dip limit selects broken setups" risk (memory 9/18) is exactly queue-position adverse selection; venue choice is the lever. Needs a broker with venue routing.
Cost: improves the conditional-on-fill markout; size unknown until measured on our arms.

**A6. IEX D-Limit / D-Peg for the resting legs** — MEDIUM
Source: IEX "Discretionary Limit" product page and IEX performance notes 2020–22 [s].
Mechanism: IEX's crumbling-quote signal reprices resting D-Limit orders before a stale-quote pick-off — the exact moment a break runs through a bid/offer.
Evidence: IEX reports higher at-or-inside-NBBO D-Peg share since D-Limit launched (vendor data).
Mapping: resting offers into the break (H#4/H#6) and retest bids; Alpaca has no venue routing → IBKR (IEX routing supported) needed.
Cost: reduces pick-off on the passive legs; vendor evidence only.

**A7. LULD-band take-profit — sell into the band-stress pop** — MEDIUM
Source: SEC DERA "LULD pilot" white paper [s]; Alphanume halt study (535 bursts) [s]; LULD/magnet literature (mixed) [s].
Mechanism: a break that runs toward the upper LULD band either pauses or snaps back; 37 % of Tier-1 straddle/limit states reverse to the pre-limit price within 1 min after the resume; ≥ 4 % band-stress bursts reverse 65 %.
Evidence: SEC staff paper (full LULD sample); Alphanume: 2.5–4 % bursts continue 56 %, ≥ 4 % reverse 65 %.
Mapping: every OPEN HOD position whose 2 R target lies above the live upper band (band computable from the 5-min reference price): rest a sell at band −1 tick. Not one of the 35 R-based exit cells — the level is mechanical.
Cost: passive exit, earns spread; trims the tail only if halts don't continue — measure on fills that reached the band.

## B. Microstructure at the HOD level (new tape data, free from SIP)

**B1. ISO share at the break print** — STRONG source / MEDIUM fit
Source: Chakravarty, Jain, Upson & Wood, "Clean Sweep: informed trading through intermarket sweep orders", JFQA 2012 [s].
Mechanism: ISOs are how impatient, informed institutions take liquidity across venues; ISO prints carry a disproportionate information share.
Evidence: ISOs = 46 % of trades / 41 % of volume; significantly larger information share; post-trade returns point to institutions.
Mapping: SIP trade condition "F" (intermarket sweep) is in the Alpaca trade stream: ISO share of the volume that crosses the HOD in the break minute; every fill (100 % of stream). New DATA, not a re-cut of OFI.
Cost: a split, not a trade; worth it only if ISO-led breaks clear +0.15 R gross.

**B2. Sub-penny (retail) share at the break** — STRONG
Source: Boehmer, Jones, Zhang & Zhang, "Tracking retail investor activity", JF 2021 [s]; Barber, Huang, Odean & Schwarz, "Attention-induced trading and returns: Robinhood", JF 2022 [s].
Mechanism: wholesaler-internalised retail prints carry sub-penny prices; attention-driven retail herding predicts negative returns in heavy-retail names.
Evidence: retail imbalance +10 bps/week in general, but −14.8 %/yr long-short among heavy-retail stocks; top 0.5 % Robinhood-bought names ≈ −5 % next month.
Mapping: TRF prints (exchange "D") with fractional-cent prices in the Alpaca trade stream at the break minute; every fill. Retail-led breaks = the chaser crowd → candidate side for A1/A2; institution-led (B1) = the long side.
Cost: as B1.

**B3. Queue imbalance at the level to time the entry** — STRONG
Source: Gould & Bonart, "Queue imbalance as a one-tick-ahead price predictor", 2016 [s].
Mechanism: in large-tick books the bid/ask queue ratio predicts the next mid move; most $2–20 small caps are one-tick-spread, i.e. large-tick.
Evidence: Nasdaq, 10 stocks, 1 year; strong out-of-sample for large-tick names, weak for small-tick.
Mapping: replace the blind stop-limit with "lift the HOD ask only when bid-queue/ask-queue > k" — an EXECUTION change on every fill, aimed at the 8 bps chase and the tick-back after the print. Needs Alpaca quote stream (free) at the arm.
Cost: targets ~0.05 R of measured chase; costs fill rate (unfilled breaks are −0.57 R — losing them is protective).

**B4. Micro-price referenced entry cap** — STRONG
Source: Stoikov, "The micro-price: a high-frequency estimator of future prices", Quant. Finance 2018 [m].
Mechanism: the size-weighted fair price sits between bid and ask; a fixed "+15 bps over the level" cap overpays when the micro-price is already at the ask and underpays otherwise.
Evidence: micro-price beats mid/weighted-mid as a short-horizon predictor on liquid US stocks.
Mapping: cap = f(micro-price at trigger) on every arm; same quote stream as B3.
Cost: a pure chase-reduction lever on 100 % of fills.

**B5. Odd-lot quotes inside the NBBO for our odd-lot orders** — STRONG
Source: Bartlett, McCrary & O'Hara, "The market inside the market: odd-lot quotes", 2022 [m]; O'Hara, Yao & Ye, "What's not there: odd lots and market data", JF 2014 [m].
Mechanism: odd-lot quotes are invisible in the round-lot NBBO yet often better; on > $50 names our $2K-notional orders ARE odd lots and can fill there.
Evidence: odd-lot quotes improve on the NBBO a large share of the time for higher-priced stocks.
Mapping: the $30–50+ slice of the stream; also a COST-MEASUREMENT correction — our NBBO half-spread may overstate what an odd-lot order pays.
Cost: lowers measured cost on the high-price slice; zero effect on $2–10 names.

**B6. Own-trigger marketable IOC instead of a broker-held stop** — MEDIUM
Source: broker order-handling docs (Alpaca stop orders are simulated/triggered by the broker, verify) [m]; our own 7-s paper queue and 8-bps chase (memory 10/17, 10/30).
Mechanism: a stop held at the broker fires on the broker's feed and queue; our websocket sees the same print; triggering ourselves with a marketable limit IOC at level + cap removes one hop of latency.
Evidence: anecdote + our measured paper latency; live latency 0.01 s.
Mapping: every arm (93/day); compare live chase per fill by trigger mode.
Cost: a fraction of 8 bps per fill — but on 100 % of fills.

**B7. Round-number geometry of the HOD** — STRONG (FX) / MEDIUM (equities)
Source: Osler, "Currency orders and exchange-rate dynamics", JF 2003 and "Stop-loss orders and price cascades", JIMF 2005 [s]; Osler, "Support for resistance", FRBNY EPR 2000 [s].
Mechanism: take-profit orders cluster ON round numbers (reversal), stop-loss orders cluster just BEYOND them (cascade); a HOD just under $X.00 meets a take-profit wall, a HOD just over it rides the stops.
Evidence: FX order books — trends accelerate past stop clusters; published S/R levels predict intraday trend interruptions for ≥ 5 days.
Mapping: arm level = max(HOD, round number) + 1 tick when the HOD sits within k bps under a $0.50/$1.00 mark; every arm (cheap, no new data). Changes which level we arm, not a post-hoc filter.
Cost: no extra cost; possibly lower chase (cascade does the work).

**B8. Trade-arrival intensity (self-excitation) before the break** — MEDIUM
Source: Bacry, Mastromatteo & Muzy, "Hawkes processes in finance", 2015 [m].
Mechanism: order flow is self-exciting; an accelerating print rate into the level signals a cascade (stops + chasers) vs a lone print.
Evidence: Hawkes branching ratios near criticality in equity/futures order flow.
Mapping: print-count acceleration in the 30 s before the touch from the Alpaca trade stream; every arm. Differs from H#37 (trade SIGN) — this is RATE.
Cost: a split; same filter caveat as B1.

## C. Short-supply and regulation mechanics on HOD names

**C1. SSR-active day HOD breaks** — STRONG
Source: OFR WP 23-08 / Management Science 2024, "Are short-selling restrictions effective?" [s]; EFMA 2018 short-sale circuit breaker paper [s].
Mechanism: Rule 201 forces shorts to post above the bid on the trigger day and the next; the resting short supply at the HOD shrinks.
Evidence: regression discontinuity: 8 % lower short volume, +35 bps daily return, no reversal after the restriction lifts.
Mapping: HOD breaks on names whose SSR is active (fell ≥ 10 % vs prior close yesterday or earlier today; computable from daily bars / Nasdaq SSR file) — a universe split by an exogenous rule, ~5–10 % of the stream on gapper days.
Cost: +35 bps/day ≈ 0.23 R on a 1.5 % stop, before any break edge.

**C2. Short-supply exhaustion intraday (shortable shares → 0)** — MEDIUM
Source: Schultz, "What makes short selling risky: other short sellers" [s]; squeeze literature, JFQA 2022 "Short squeezes and their consequences" [s]; IBKR public shortable-shares/fee file (updated intraday, verify) [m].
Mechanism: when locates run out as price approaches the HOD, no new shorts can lean on the level; recall risk peaks on high-volume days.
Evidence: when fees > 10 %, a lender recall squeeze ≈ once every 25 days; recall risk highest on high-volume days.
Mapping: poll available-shares for the armed names (~93/day); break with exhausted supply vs plentiful supply. Also the honest kill switch for any failed-break short (H#7).
Cost: split, free data.

**C3. End-of-day short-covering on high-fee breakers** — STRONG
Source: Bogousslavsky, "The cross-section of intraday and overnight returns", JFE 2021 [s].
Mechanism: lending fees and margin are charged overnight, so arbitrage shorts cover into the close; overpriced (heavily-shorted) names rise at the end of the day.
Evidence: mispricing factor positive through the day, negative in the last 30 min; IVOL/size anomalies realised in the last half-hour or overnight.
Mapping: every HOD position still open at 15:25 (≈ all non-stopped fills): hold to 15:55 only if borrow fee/utilisation is high, else flat at 15:25 (see D1 for the other half). A time-of-day exit rule from a mechanism, not an R-target cell.
Cost: removes 30 min of exposure; no extra trades.

**C4. Dilution overhang — the hidden seller at the HOD** — MEDIUM (practitioner)
Source: dilutionwatch.com ATM guide, Alphanume ATM explainer (2025–26) [s]; baby-shelf rule (I.B.6 of Form S-3) [s].
Mechanism: an issuer with an active ATM sells into strength through its sales agent; offers reload at the same level without a visible seller of size.
Evidence: practitioner pattern; no peer-reviewed intraday number found.
Mapping: tag every armed name with "effective S-3 + 424B5/sales-agreement 8-K in last 90 days" (EDGAR tooling on disk). Expect lower break-hold and a higher failure rate → the failed-break side's universe.
Cost: free tag; the claim is about which side, not cost.

**C5. Prior-day short-sale ratio as the side selector** — STRONG source / WEAK fit
Source: Diether, Lee & Werner, "Short-sale strategies and return predictability", RFS 2009 (WP "Can short-sellers predict returns? Daily evidence") [s]; Hu, Jones, Zhang & Zhang, JFE 2025 [s].
Mechanism: short sellers are informed; heavy shorting predicts lower returns days ahead; intraday shorting near the open predicts most.
Evidence: daily NYSE/Nasdaq Reg SHO data; shorting flow at open/mid/close all negatively predict returns.
Mapping: FINRA daily short volume (free) for every armed name; decides which side (break long vs failure short) gets the name — a population split, filter caveat applies.
Cost: free data; a daily-horizon effect, small per intraday fill.

## D. Close-of-day flows meeting the day's HOD names

**D1. Cross-sectional end-of-day reversal of the day's winners** — STRONG
Source: Baltussen, Da & Soebhag, "End-of-day reversal", 2024 [s]; "End-of-day momentum in the cross-section and option hedging", EFMA 2021 [s].
Mechanism: individual stocks' early-day returns NEGATIVELY predict their last-half-hour return in the cross-section (liquidity/hedging), unlike the index.
Evidence: US cross-section, last 30 min; hedging pressure documented as a driver.
Mapping: the day's HOD breakers ARE the early-day winners: (a) flatten every open HOD position at 15:25–15:30 instead of 15:55; (b) a 15:30→MOC short sleeve on ETB breakers still near their HOD (pair with C3: exclude high-fee names). Uses every fill held late + all breakers.
Cost: 30-min hold on a small cap ≈ 10–20 bps RT; effect size needs measuring on our names before believing.

**D2. Single-stock leveraged-ETF rebalancing on HOD names** — STRONG mechanism / small share
Source: Bessembinder, "Rebalancing, frictional costs, and returns to levered single-stock ETFs", SSRN 2025 [s]; "Of seesaws and swings: the market-wide impact of levered ETF rebalancing", EFMA 2024 [s].
Mechanism: a 2× long ETF must buy (L²−L)·AUM·r of the underlying near the close on an up day — a forced, predictable buyer.
Evidence: rebalancing costs 0.20–0.42 %/month to the ETFs (the flow is real and sized); literature links rebalancing demand to late-day returns.
Mapping: HOD breakers with a listed 2× single-stock ETF (many mid-cap momentum names now have one): compute required $ rebalance / remaining-day ADV at 15:00; hold to close when it is large.
Cost: zero extra trades; a hold-or-flatten rule on a slice of the stream.

**D3. Strike pin at the HOD (positive gamma / OPEX)** — STRONG
Source: Ni, Pearson & Poteshman, "Stock price clustering on option expiration dates", JFE 2005 [m]; Ni, Pearson, Poteshman & White, "Does option trading have a pervasive impact on underlying stock prices?", RFS 2021 [s].
Mechanism: with positive net dealer gamma, delta-hedging sells rallies and buys dips — reversal and pinning at big-OI strikes; negative gamma does the opposite (H#42 is the negative side only).
Evidence: closes cluster at strikes on expiration dates; aggregate gamma sign flips intraday momentum vs reversal.
Mapping: arms whose HOD sits just under a large-OI call strike on a positive-gamma day (esp. Fridays) → skip or fade; OCC/CBOE OI free. Only the optionable slice (~56 % have a chain).
Cost: split; free data.

**D4. Market-gamma hold-to-close switch** — STRONG source / WEAK fit
Source: Baltussen, Da, Lammers & Martens, "Hedging demand and market intraday momentum", JFE 2021 [s]; Gao, Han, Li & Zhou, "Market intraday momentum", JFE 2018 [s].
Mechanism: short-gamma dealers and LETF rebalancers trade with the day's move into the close.
Evidence: 60+ futures 1974–2020; rest-of-day return predicts last 30 min.
Mapping: hold HOD positions to 15:55 only on days with large index short-gamma/LETF flow. Our own SPY/QQQ timing test was reversed below cost (cells 1,714–1,716), so WEAK; listed for completeness.
Cost: no trades added.

## E. Horizon changes triggered by the HOD event

**E1. High-volume return premium — 20-day hold of HOD-day names** — STRONG
Source: Gervais, Kaniel & Mingelgrin, "The high-volume return premium", JF 2001 (replicated cross-country) [s].
Mechanism: a volume shock raises visibility → more future buyers; strongest when volume is extreme but the return is NOT.
Evidence: unusually high 1-day/1-week volume predicts appreciation over the next month, across sizes; not explained by announcements or risk.
Mapping: every HOD-break day is an RVOL shock (1–5×); the ARMED set with MODEST day return (not the gappers) feeds a monthly sleeve; daily bars on disk, PIT universe on disk. Different trigger from the failed 52-week-high sleeve.
Cost: month horizon, ~20–40 bps RT vs a monthly premium — survives if it exists in this universe; check vs MOM sleeve overlap.

**E2. Same-half-hour periodicity** — STRONG source / WEAK cost
Source: Heston, Korajczyk & Sadka, "Intraday patterns in the cross-section of stock returns", JF 2010 [s].
Mechanism: recurring institutional order-splitting at the same time of day; a stock's return in half-hour k continues at daily lags in the same half-hour for ≥ 40 days.
Evidence: 13 half-hour intervals, continuation at multiples of a day, strongest first/last half-hour.
Mapping: each HOD break stamps a half-hour; buy that name at the start of the same half-hour on days +1..+5. The stream multiplies into 5× trade count.
Cost: HKS effects are a few bps; small-cap spreads likely eat it — test gross first, expect a fail on cost.

**E3. Day-2 intraday fade of yesterday's breakers** — STRONG source / MEDIUM fit
Source: Lou, Polk & Skouras, "A tug of war: overnight versus intraday expected returns", JFE 2019 [s].
Mechanism: momentum accrues overnight (institutional clientele), the intraday clientele trades against it; returns flip sign between the legs.
Evidence: all momentum profits earned overnight; intraday leg reverses; ~2 %/month overnight-minus-intraday spread variation.
Mapping: yesterday's HOD breakers (≈ 10/day): short day-2 open-to-close on ETB names that fail to take yesterday's HOD in the first 30 min (the "prior-day high" long failed 9/26 — this is the other side). Daily stop, not a 1.5 % stop (1,736 lesson).
Cost: one round trip per name-day; borrow; needs ≥ 50 bps gross.

**E4. Attention-herding fade over the following week** — STRONG
Source: Barber, Huang, Odean & Schwarz, JF 2022 [s].
Mechanism: attention-induced retail buying overprices; extreme herding events reverse over days–weeks.
Evidence: top 0.5 % Robinhood-bought names ≈ −5 % over the next month, ≈ −9 % for extreme herding.
Mapping: HOD breakers that topped the day's % gainers with B2 retail share high → a weekly short basket (ETB, capped), formed at the close. Distinct from the intraday failed-break short.
Cost: weekly horizon makes 40 bps irrelevant vs −5 %; borrow fees on these names are the real cost.

**E5. News vs no-news split for any post-day hold** — STRONG
Source: Chan, "Stock price reaction to news and no-news: drift and reversal after headlines", JFE 2003 [s].
Mechanism: under-reaction to public news (drift) vs over-reaction to moves without news (reversal).
Evidence: 1980–2000 headlines; bad-news drift strong, no-news extreme moves reverse.
Mapping: tag every HOD-break day by "headline present" (binary, not the closed LLM quality score); use ONLY as the side selector for E1/E3/E4 multi-day sleeves.
Cost: free (news tag already exists).

**E6. Volatility-contraction days (NR7/ID-NR4) as the expansion sleeve** — MEDIUM
Source: Crabel, "Day trading with short-term price patterns and opening range breakout", 1990 [s].
Mechanism: range contraction precedes expansion (volatility clustering/mean reversion of range).
Evidence: practitioner: narrow ranges precede trend days ~68 %; breakouts from bottom-quartile ranges move ~40 % more.
Mapping: HOD breaks on names whose prior day was NR7 — a pre-open universe flag; filter caveat applies.
Cost: unchanged per fill; claim is larger MFE.

**E7. Compound-extreme HOD (also a multi-day/52-week extreme) — multi-day hold** — STRONG source / WEAK fit
Source: Huddart, Lang & Yetman, "Volume and price patterns around a stock's 52-week highs and lows", Mgmt Sci 2009 [s].
Mechanism: anchoring/limited attention — volume spikes and small-investor buyer-initiated trades rise after the past range is crossed; returns are reliably positive afterwards.
Evidence: large-sample US; volume spike + positive subsequent returns after either extreme.
Mapping: the slice of HOD breaks that also clear a 20/52-week high; multi-day hold. The intraday prior-day-high cell and the 52-week sleeve both failed — only the event-day multi-day version is untested.
Cost: multi-day; cost small.

## F. Options and volatility around the HOD stream (optionable slice only — 1,738 found 22 % with prints)

**F1. Option legs only intraday — never overnight** — STRONG
Source: Muravyev & Ni, "Why do option returns change sign from day to night?", JFE 2020 [s].
Mechanism: option prices ignore that volatility is much higher intraday than overnight; long options earn intraday, lose overnight.
Evidence: S&P index options +0.3 %/day intraday vs −1 %/day overnight; same pattern in equity options, all maturities/moneyness.
Mapping: a design rule for H#12/H#13 on the liquid/optionable slice: buy at the break, sell by the close.
Cost: option spreads are the binding cost — check on the 22 % with prints.

**F2. Sell the overnight option leg on HOD names** — STRONG source / MEDIUM fit
Source: Muravyev & Ni 2020 [s].
Mechanism: the mirror of F1 — overnight option returns are strongly negative.
Evidence: as F1.
Mapping: on HOD breakers with liquid chains, sell defined-risk premium (call spreads/iron flies) at 15:50, buy back at the open; L3 approved.
Cost: two option round trips per night on wide spreads — likely fails on small caps, passable on the mid-cap slice.

**F3. Sell lottery premium on the HOD crowd's names** — STRONG
Source: Boyer & Vorkink, "Stock options as lotteries", JF 2014 [s].
Mechanism: skewness-seeking buyers overpay for calls on lottery-like stocks; HOD-break days manufacture exactly that demand.
Evidence: options on high ex-ante-skew stocks underperform by 10–50 %/week across sorts; calls on most-lottery names −10–20 %/month vs least.
Mapping: names with ≥ 2 HOD breaks in a day and a listed chain → next-morning short OTM call spreads (defined risk) for 1–5 days.
Cost: wide option spreads; defined-risk spreads give up part of the premium; gap-up risk = the tail.

**F4. Call skew as a free crowding gauge** — MEDIUM
Source: Kris Abdelmessih (Moontower), "when vertical spreads are traps", "a sense of proportion around skew" [s].
Mechanism: when call skew rips (meme rallies) the vertical prices the stock LOWER; the market tells you the break is a lottery.
Evidence: practitioner reasoning with worked examples (GME).
Mapping: for optionable armed names, call-skew slope at the arm as a side selector (break long vs F3/failure side). Optionable slice only.
Cost: free read off the chain.

## G. Diagnostics, risk and sizing that use the whole stream

**G1. Variance-ratio diagnostic decides stop-book vs provision-book** — STRONG
Source: Kaminski & Lo, "When do stop-loss rules stop losses?", JFM 2014 [s].
Mechanism: under a random walk a stop always lowers expected return; stops add value only if post-entry returns have momentum.
Evidence: theory + 1950–2004 data (50–100 bps/month in stop-out periods when momentum exists).
Mapping: compute the post-break variance ratio / autocorrelation on all 15,656 per-minute signal paths (on disk, zero cost). VR < 1 (mean reversion) predicts exactly what we measured (no-stop +0.2 R paired, stop slip, 41 % stop-outs) and says: move capital from break-and-stop designs to A1/A2/H#1/H#2 provision designs. One analysis, programme-wide decision.
Cost: none.

**G2. Own-variance (volatility-managed) book sizing** — MEDIUM
Source: Moreira & Muir, "Volatility-managed portfolios", JF 2017 [s]; critique Cederburg, O'Doherty, Wang & Yan, JFE 2020 [s].
Mechanism: book volatility varies more than book expected return, so scaling by inverse trailing variance raises Sharpe.
Evidence: 50–100 % Sharpe gains on factors; the critique finds most gains do not survive out of sample.
Mapping: scale the HOD book's daily risk by 1/var(trailing 20-day daily R) — different from H#33 (fill-rate). Only after the book has a positive sign.
Cost: none; a variance tool, not an edge.

**G3. Intraday beta hedge with IWM** — MEDIUM
Source: standard; day-clustered SE requirement (CLAUDE.md §7) [m].
Mechanism: HOD fills cluster on risk-on days; a residual book removes the shared market leg that makes days, not trades, the unit.
Evidence: our own day-clustering (t falls sharply clustered).
Mapping: short IWM against open HOD notional × beta (one liquid instrument, 1 bp). 100 % of fills.
Cost: ~1–2 bps on the hedge leg; reduces day variance, adds no edge.

**G4. 1/N risk across ALL arms instead of first-come slots** — STRONG source / MEDIUM fit
Source: DeMiguel, Garlappi & Uppal, "Optimal versus naive diversification", RFS 2009 [m].
Mechanism: with ~93 arms/day and no per-fill information, equal tiny risk on every arm beats concentrating in the first 14.
Evidence: 1/N beats optimised allocations out of sample when estimation error dominates (it does here).
Mapping: risk/arm = daily risk budget / expected fills; uses 100 % of the stream. Slot cells moved the MEAN ±0.04 R — this targets day-variance and the ramp's ability to resolve the sign.
Cost: more fills × smaller size — Alpaca has no per-trade commission; odd-lot fills (B5) help.

## H. Behavioural counterparties and the human crowd at the HOD

**H1. Public-alert crowd arrival** — WEAK
Source: Warrior Trading/Ross Cameron (audited $20M+, 71 % of 936 trades in one challenge) [s]; social-attention literature [s].
Mechanism: large chat rooms alert the same small-cap HOD breaks; thousands of followers arrive seconds-to-minutes later — a predictable buyer queue.
Evidence: anecdote + one trader's audit; WSB studies find mixed intraday effects.
Mapping: timestamp public alerts (where scrapable) against our arms; measure post-alert flow; if real, our early fill sells into their arrival (H#4 limit).
Cost: no extra cost; legality fine (public info), ethics fine (we are the earlier buyer, not a manipulator).

**H2. Micro-pullback re-entry after the break** — WEAK
Source: Warrior Trading momentum strategy pages [s]; SMB Capital playbook/second-day/VWAP sessions [s].
Mechanism: the first 1-min pullback after a break shakes out chasers; the first candle to make a new high after it is the professional's entry with a tick-tight stop.
Evidence: practitioner, no published statistics.
Mapping: filled breaks that pull back 1–3 bars; entry on the first new 1-min high; re-entry was toxic on every book — needs its own PREREG.
Cost: tight stop ⇒ cost ratio worse; WEAK.

**H3. Pre-market rotation ratio** — MEDIUM (practitioner with counts)
Source: SmallCapLab gap-up fade/runner research (3,000+ gap-ups, 3,400+ runners since 2022) [s].
Mechanism: pre-market volume / float measures how much supply already changed hands; low rotation = a gap with no buyers behind it.
Evidence: practitioner thresholds (≥ 0.3× float = gap-and-go probable, ≤ 0.1× = trap).
Mapping: pre-open tag on every gapper in the arm universe; side selector between break long and failure side. Filter caveat.
Cost: free (Alpaca extended-hours bars).

**H4. Promotion/pump names: hold intraday, never overnight** — MEDIUM
Source: Renault, "Market manipulation and suspicious stock recommendations on social media", AMF/EFMA 2017 [s].
Mechanism: promoted small caps show short-term continuation then reversal as promoters distribute.
Evidence: tweet-driven pumps: price impact, volume, short-term continuation, longer-term reversal (stronger when promoter accounts tweet).
Mapping: HOD breaks on names in a promotion burst (the stupid-money scraper on this node) → long intraday with a hard 15:30 flat, short-side candidate at day 2–5 (E4).
Cost: intraday unchanged; the multi-day short pays borrow.

**H5. Day-trader base rate: be the house** — STRONG
Source: Barber, Lee, Liu & Odean, Taiwan day traders (1992–2006) [s]; "Day traders lose money and keep trading" [s].
Mechanism: < 1 % of day traders are predictably profitable; the median HOD chaser loses ~24 bps/day net; profit accrues to whoever supplies liquidity and charges for information.
Evidence: complete Taiwan market records.
Mapping: the programme-level argument for A1–A7 and I1: the 50/wk breaks are a stream of losing-on-average counterparties; we have been joining them.
Cost: n/a — strategic framing with a strong base rate.

## I. Monetise the stream itself, not only the trade

**I1. Sell the HOD event stream (alerts / data)** — WEAK (business)
Source: scanner vendors (Trade Ideas, Benzinga Pro, Warrior scanners) sell HOD/momentum alerts on subscription [m, pricing unverified].
Mechanism: retail day traders (H5) pay for exactly this event stream; revenue does not depend on our per-fill sign.
Evidence: an existing paid market; no numbers fetched.
Mapping: the arm/break/retest/failure stream with context (H#25 Telegram list is the MVP). Needs owner GO; regulatory check (publisher exemption).
Cost: none per trade; the frequency is the product.

**I2. Theme cluster → buy the liquid leader** — MEDIUM
Source: Cohen & Frazzini, "Economic links and predictable returns", JF 2008 [m]; "Decoding momentum spillover effects", JFQA [s].
Mechanism: attention/information diffuses slowly from the names where it shows first to linked names.
Evidence: customer-supplier and industry spillovers predict returns (monthly); intraday version unknown.
Mapping: when ≥ 3 HOD breaks cluster in a theme within 30 min, buy the theme's liquid leader/ETF (cost 1–3 bps) for the rest of the day. Extends H#24 (MOM rotation) to an intraday trade with liquid cost.
Cost: liquid leg ⇒ survives easily if any spillover exists.

**I3. Owner-position alert on failed HOD in his names** — WEAK
Source: our own −0.57 R unfilled-break number [m].
Mechanism: a failed break in a name the owner holds is a measured negative event.
Evidence: internal.
Mapping: report-only Telegram when an owner-held name prints a failed HOD (owner positions are never touched).
Cost: zero.

## J. Markets where the identical HOD mechanics transfer

**J1. 2× single-stock ETF as the instrument** — WEAK
Source: Bessembinder 2025 [s]; "The rise of single-stock ETFs and more volatile stock prices" (FMA) [s].
Mechanism: the LETF's market maker quotes off the underlying; same R at half the notional; no borrow on the short side via the inverse ETF.
Evidence: LETF frictions 0.55–1.04 %/month — they are costly to HOLD, not necessarily to trade intraday.
Mapping: HOD breakers with a 2×/inverse LETF; trade the ETF on the underlying's signal; inverse ETF for failure shorts without locates.
Cost: compare LETF spread vs underlying per name — often worse.

**J2. Crypto perpetuals: funding rate as the counterparty gauge** — MEDIUM
Source: "Intraday return predictability in the cryptocurrency markets: momentum, reversal, or both", N. Am. J. Econ. & Finance 2022 [s]; crypto Donchian fee-sensitivity tests [s].
Mechanism: a high positive funding rate means longs are crowded and paying; 24-h-high breaks then fail (forced long unwinds), negative funding breaks squeeze.
Evidence: crypto shows both intraday momentum and reversal regimes; Donchian edges die between 10 and 50 bps cost.
Mapping: extends H#17 (crypto 24-h high): condition side on funding (free API). Same engine, 24/7 frequency.
Cost: perp taker ~4–5 bps; survives where equities do not.

**J3. FX session-high breaks at round numbers** — STRONG source / WEAK access
Source: Osler 2003/2005 [s].
Mechanism: the best-documented stop-cascade-at-level market; the HOD mechanics are literally Osler's.
Evidence: order-book data from a major dealing bank; cascades past stop clusters.
Mapping: needs an FX broker (not Alpaca); same arm/stop engine on EUR/USD, USD/JPY session highs with B7 geometry.
Cost: 0.5–1 bp spread ⇒ cost is a non-issue; edge size unknown.

**J4. Overnight-session (Blue Ocean) high breaks** — WEAK
Source: Eaton, Shkilko & Werner, "Nocturnal trading" (AFA) [s]; Lim, "Overnight adverse selection: Blue Ocean ATS vs Nasdaq", SSRN 2026 [s].
Mechanism: a ~90 %-retail venue with ~10 market makers; price discovery share ~9 %; cost premium is structural, not informational.
Evidence: as cited; no breakout study.
Mapping: our day-session HOD names that trade overnight; their overnight-session high as the level. Broker access needed.
Cost: wide spreads overnight; likely fails on cost.

**J5. China limit-up "board hitting"** — STRONG source / no access
Source: Wang et al., "Post-hit dynamics of price limit hits in the Chinese stock markets", Physica A 2017 [s]; Chen, Gao, He, Jiang & Xiong, "Daily price limits and destructive market behavior", JoE (NBER w24014) [s].
Mechanism: closing limit-up opens higher next day (79.4 %, +2.43 % overnight); large investors buy at the limit and sell next day, then prices reverse.
Evidence: full A-share sample.
Mapping: not accessible; the US analog (close-at-high overnight) is closed (tail-carried). Listed only to record the mechanism.
Cost: n/a.

**J6. Commodity-futures session-high breaks after contraction** — WEAK
Source: Crabel 1990 [s].
Mechanism: as E6, on 23-h futures with one tick spreads.
Evidence: practitioner.
Mapping: index-futures ORB/intraday momentum is closed (cells 1,714–1,716, index ORB 18 cells); commodities untested; needs a futures account.
Cost: 1 tick ≈ 1–3 bps — survives if gross exists.

---

## Count by family
A 7 · B 8 · C 5 · D 4 · E 7 · F 4 · G 4 · H 5 · I 3 · J 6 = **53 ideas**, every one triggered by, drawn from, or
trading against the HOD event stream.

## Top 10 — ranked by (share of the ~50/week used) × mechanism strength × fit to our data × cost survival
1. **G1 variance-ratio diagnostic** — uses all 15,656 paths at zero cost; Kaminski-Lo says it decides whether ANY stop-based HOD design can win or the money is in provision (A1/A2/H#1/H#2).
2. **A4 maker-rebate routing (+A5 venue choice)** — every passive leg on every fill; 4–6 bps/side on a $5 name is a rule, not an estimate; needs a direct-access broker.
3. **B3 + B4 queue-imbalance / micro-price entry** — 100 % of fills, attacks the measured 8 bps chase with free quote data; strong microstructure evidence for large-tick names (most of our universe).
4. **D1 + C3 end-of-day treatment (EOD reversal vs short-covering split)** — every position held past 15:25; two JFE/JF-grade mechanisms with opposite signs that our borrow-fee tag separates.
5. **A2 price-pressure bid under the failure flush** — uses the failed/unfilled half of the arms (the −0.57 R set) plus our own stop-outs; 118 bps small-cap pressure with a 2-day half-life clears cost.
6. **B1 + B2 ISO and sub-penny tags at the break** — every fill, free SIP condition/price fields, genuinely new data (institution vs retail-herd at the level), strong peer-reviewed sources.
7. **B7 round-number arming** — every arm, no data cost, Osler's stop/take-profit clustering maps one-to-one onto a level-break engine.
8. **E1 high-volume return premium sleeve** — every HOD day is a volume shock; ~50 names/week feed a monthly book where 40 bps is noise; replicated JF result.
9. **A7 LULD-band take-profit** — every open position whose target sits past the band; mechanical level, passive exit, SEC-documented reversal rates.
10. **C1 SSR-active breaks** — only ~5–10 % of the stream, but an exogenous regulatory shock with RD evidence of +35 bps/day ≈ 0.23 R before any break edge.

Not in the top 10 and why: E2/F2/J4 fail the cost line on small caps; F-family is capped at the optionable 22–56 %;
J-family needs new accounts; H1/H2/I1/I3 are anecdote-grade; D4/C5/E7 overlap closed tests.
