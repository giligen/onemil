# Route E3 — Crypto Basis / Funding-Rate Arbitrage

**Status: research only, NOT live. Verdict: real but small, not "passive," not "constant."**
Date: 2026-09-25. Author: agent run under `research/passive_income/`.

## 1. Mechanism

Long spot BTC/ETH (Alpaca crypto, cash, no leverage) + short an equal-notional perpetual future on
another venue. Delta ≈ 0 (price risk cancels). P&L = funding payments received from perp shorts when
funding is positive, minus fees, minus funding paid when funding is negative, minus any basis
gap at entry/exit.

## 2. Data pulled (own computation, not a report I'm relaying)

Source: Binance USDT-margined perpetual funding-rate history, public REST
(`fapi.binance.com/fapi/v1/fundingRate`), paginated 2023-01-01T00:00 → 2026-09-25T08:00, **4,091
funding events each for BTCUSDT and ETHUSDT** (8h interval, 3/day, 1,364 days). Cross-checked the
endpoint against Bybit's public funding endpoint (`api.bybit.com/v5/market/funding/history`) on a
3-row spot sample — same sign, same order of magnitude, not a full reconciliation. Raw JSON and the
computation script are in the scratchpad (not committed; too large/ephemeral for the repo).

**This is Binance's index/funding formula, not Kraken Derivatives US's or Coinbase Financial
Markets's** (the two venues actually reachable from the US — see §4). Those products are new
(launched Jun–Aug 2026) and may compute funding on a different interval or index. Binance/Bybit
history tells you how the *crypto basis market* has behaved, not what a US-regulated account would
have realized to the decimal. Flagging this per the "read your own report as an adversary" rule.

### BTC (BTCUSDT), n=4,091, 2023-01-01 → 2026-09-25
- Overall annualized funding (mean rate × 3 × 365): **+7.25%/yr**
- Negative-funding days: 148 / 1,364 = **10.85%** of days
- Longest negative stretch: **11 consecutive days, 2026-04-11 → 2026-04-21**
- By quarter (annualized %): 2023 Q1 8.2, Q2 6.3, Q3 4.8, Q4 12.2 | 2024 Q1 22.3, Q2 9.3, Q3 3.5, Q4
  12.7 | 2025 Q1 5.3, Q2 3.5, Q3 7.0, Q4 4.7 | **2026 Q1 1.2, Q2 1.0**, Q3 6.7 (partial, 260 events)

### ETH (ETHUSDT), n=4,091, same window
- Overall annualized funding: **+7.37%/yr**
- Negative-funding days: 155 / 1,364 = **11.36%**
- Longest negative stretch: **9 consecutive days, 2023-08-18 → 2023-08-26**
- By quarter (annualized %): 2023 Q1 9.1, Q2 5.4, Q3 4.4, Q4 14.1 | 2024 Q1 22.9, Q2 10.3, Q3 4.8, Q4
  14.0 | 2025 Q1 5.1, Q2 4.3, Q3 6.0, Q4 4.3 | **2026 Q1 0.04, Q2 0.8**, Q3 4.8 (partial)

**Read:** the edge is real and persistent (only ~11% of days negative, longest bad stretch 9–11
days out of 1,364 — never a multi-month dead zone) but it is a *regime*, not a constant: quarterly
annualized yield ranged 0.04%–22.9%, and the two most recent full quarters (2026 Q1–Q2) are the
weakest in the whole sample for both coins. There is no guarantee 2026 Q1/Q2-style compression
doesn't persist or extend.

## 3. Costs (fees, sourced)

- **Alpaca crypto spot: 0.25% taker** per trade, volume-tiered lower at scale
  ([Alpaca crypto fee docs](https://docs.alpaca.markets/us/docs/crypto-fees)). One-time cost per
  entry/exit, not recurring while the position is held.
- **Perp leg, US-legal venue (Kraken Derivatives US):** retail futures fee schedule starts **0.02%
  maker / 0.05% taker**, better at volume tiers
  ([Kraken fee schedule](https://support.kraken.com/articles/360048917612-fee-schedule)); Kraken's
  CFTC-regulated US product (via Bitnomial/NinjaTrader Clearing) instead charges a flat **$0.15/contract/side**
  all-in — check which fee table actually applies to the regulated US product before sizing, the
  two schedules aren't the same thing and search results don't disambiguate cleanly.
- **Round-trip cost estimate on $15K notional per leg:** ~$37.50 (Alpaca) + ~$7.50 (Kraken taker) ≈
  $45 to open, ~$45 to close ≈ **$90, or 0.3% of $30K capital, one-time** if the position is held for
  months and not churned. Trivial next to the funding numbers above — **fees are not the binding
  constraint here; funding-regime compression and counterparty/liquidation risk are.**

## 4. Venue access — this is the load-bearing finding, not a footnote

**US person:** Binance and Bybit are **both fully closed to US persons** (DOJ/OFAC settlement for
Binance; Bybit added the US to its restricted-jurisdiction list). Until May–June 2026 there was **no
legal US retail perp venue at all**. That changed: CFTC approved the first US-regulated perpetual
futures May 29 2026; **Kraken launched CFTC-regulated perps June 15 2026** (BTC, ETH, SOL, XRP, LINK,
DOGE, LTC, AVAX, via Kraken Pro / Kraken Derivatives US, cleared through Bitnomial); **Coinbase
Financial Markets** offers a CFTC-regulated perpetual-style product too, BTC/ETH, **capped at 10x
leverage**. These are the only two routes if the Alpaca account is a US person — both brand-new
(3–4 months of live history as of this report), meaning no track record for uptime, spread quality,
or how their funding mechanism behaves in stress.

**EU person:** Binance has **withdrawn its EU MiCA application and is winding down EU service**
(deadline was 2026-07-01, no extension). **Bybit EU (Austria) holds a MiCA CASP license** — but MiCA
covers spot/custody; **derivatives (perpetual futures) fall under MiFID II, a separate license**, and
Bybit's own CEO has said MiCA alone isn't sufficient to operate profitably in the EU. Whether Bybit EU
actually offers perpetuals to *retail* EU clients, at what leverage cap, needs to be confirmed
directly with Bybit EU before relying on it — the search results describe the licensing gap, not a
confirmed retail perp product. Historically EU retail crypto-derivative leverage has been
capped hard by ESMA-style rules; that would materially cut the achievable notional per dollar of
margin versus a US Kraken/Coinbase account.

**Bottom line: this is not "flip a flag."** The existing OneMil stack has an Alpaca integration and
nothing else. Route E3 needs (a) a brand-new brokerage account at Kraken Derivatives US or Coinbase
Financial Markets, its own approval process, its own API keys/secrets, (b) a new automation module
(margin monitoring, funding sweep, delta rehedge) that doesn't exist in this codebase today, and
(c) for an EU person, a live confirmation call/email with Bybit EU on retail derivatives eligibility
before assuming the venue is even usable.

## 5. Operational risks

- **Liquidation despite being "hedged."** The spot leg and the short-perp leg sit on two different
  venues with two different collateral pools. A fast BTC/ETH rally can liquidate the perp leg on
  margin before the (paper) gain on the spot leg is realized or transferable — the position is
  economically hedged but not margin-fungible across venues. Mitigation is low leverage (≤2–3x) and
  a standing cash buffer, which lowers the effective yield on capital.
- **Basis blowout at entry/exit.** Spot and perp prices can gap apart under stress; entering/exiting
  at a bad basis is a one-off cost on top of the funding P&L, worse in fast markets — exactly when
  you'd want to be exiting.
- **Counterparty/exchange failure.** The FTX collapse (Nov 2022) is the standing precedent: traders
  using FTX as the short venue lost the entire perp-leg collateral while the spot leg was unaffected
  in value but left completely unhedged. On this structure that is a **near-total loss of the
  perp-leg allocation** (illustratively ~$15K of the $30K if split 50/50), not a bad month — a tail,
  not a distribution.
- **Not actually passive.** Automated rebalancing between two venues (margin top-ups, delta
  rehedging as spot/perp values drift, funding sweep) is required infrastructure, not a nice-to-have
  — without it the position silently drifts un-hedged or gets liquidated unattended. This is new code
  and a new monitored service, i.e. daily-babysitting risk during the build/shakeout period even if
  the steady state is closer to hands-off.
- **New-venue risk specific to 2026:** both US-legal products are 3–4 months old. No history of how
  they behave in a real stress event.

## 6. Realistic net yield, $30K deployed (illustrative split: $15K spot + $15K perp notional,
matched 1:1, i.e. ~1x leverage on the short leg — deliberately conservative to keep liquidation risk
low; a buffer beyond this reduces yield further and is recommended in practice)

Using the blended BTC/ETH funding series above, one-time fees amortized to ~0 over a multi-month hold:

| Regime | Basis | Annualized funding | Gross $/mo on $15K notional | Net (after ~$90 one-time fees, amortized) |
|---|---|---|---|---|
| **Calm/hot** (best quarters, e.g. 2024 Q1) | ~22% blended | ~22%/yr | ~+$275/mo | ~+$270/mo |
| **Normal** (full-sample mean) | ~7.3% blended | ~7.3%/yr | ~+$91/mo | ~+$85/mo |
| **Stressed** (2026 Q1–Q2-style compression, or a negative stretch) | ~0–(-3%) blended | ~0 to -3%/yr | ~$0 to -$37/mo | ~$0 to -$45/mo |

**Worst plausible month, in dollars, on this $30K:** funding-only downside in a bad month is small
(roughly -$40 to -$150, bounded by exchange funding-rate caps). The real worst case is **not a
funding month at all** — it's the tail: a missed margin call in a violent rally or a venue failure,
which can cost **most or all of the ~$15K perp-leg collateral** in a single event. That is a standing
risk that sits alongside the monthly funding P&L, not inside its distribution — report it separately,
per the project's tail-dependence rule.

## 7. Skew and tail (explicit, per project rule)

Distribution is **negatively skewed**: many small positive months (funding carry), one rare large
negative event (liquidation or counterparty failure) that dwarfs months of carry. This is
structurally the same shape as short-vol/short-gamma strategies — steady collection, rare blowup —
and should be sized and reported as such, never averaged into a smooth expected return.

## 8. Verdict on "constant-ness"

**Not constant.** Quarterly annualized yield swung 0.04%–22.9% over 15 quarters on both coins, and
the two most recent quarters (2026 Q1–Q2) are the weakest on record. It is **frequent** (only ~11% of
days negative, no dead stretch over 11 days in 3.75 years) — closer to the project's own "frequency
over confidence" bar than most of what's been tried — but the dollar income scales with a regime the
strategy doesn't control, and the tail risk (liquidation, counterparty failure) is not visible in any
of the monthly numbers above. **Not passive** in the "no daily babysitting" sense until a real
cross-venue automation module is built and proven; until then it needs at least a daily margin/health
check, which is exactly the babysitting the owner wants to avoid.

## 9. Recommended first experiment (minimum size, per the project's live-exploration-tier rule)

1. Open a Kraken Derivatives US (or Coinbase Financial Markets) account — approval, not code;
   budget 1-2 weeks.
2. Deploy **$2,000–3,000 total** (not $30K) in the 1x-matched structure above for one full quarter,
   manually rebalanced (no new automation yet) — the goal is to observe real fills, real funding
   receipts/payments, and real margin behavior on the *actual* regulated-US product, since the
   Binance/Bybit history above is a proxy, not a promise.
3. Cost: ~$10-15 in one-time fees on $2-3K, ~3 months of manual checking (10 min/day), $0 in data
   spend (public REST is free).
4. Gate to scale: if realized funding over that quarter is directionally consistent with the
   Binance-proxy series (same sign, same rough magnitude) and no margin/liquidation surprise
   occurred, THEN scope the automation module and consider the $30K deployment.
