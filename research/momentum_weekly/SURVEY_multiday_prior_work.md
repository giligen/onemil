# Survey: Multiday Strategies Already Tested — OneMil Research

## Prior Work Summary

**Count: 74 cells across 11 studies tested, verdict: 0 passing per the pass bars.**

### Studies with Most Positive Results

1. **research/fomc_drift/RESULT_1702.md** — Pre-FOMC drift (10 cells, 1,702)
   - 2 of 10 pass on close-to-close, QQQ W1/W2: +24.5 / +26.0 bps (t 2.49 / 2.58)
   - Verdict: NOT RECOMMENDED; resolution time ~14 years at 8 events/yr; fails sleeve rule (W4 2012–26 needs t ≥ 1.5, measured t = 0.88)

2. **research/insider_events/RESULT.md** — Insider Form 4 / 13D purchases (4 cells, I1–I4)
   - I3 (opportunistic ≥ $25K) closest: VAL +0.60 % per 20 sess, t 1.28; hits MDE wall (~1 %/month at book size)
   - Verdict: FAIL; survivorship arm void (delisted names never backfilled)

3. **research/multiday_catalyst/REPORT.md** — Catalyst-day continuation 3-day hold (2 cells, 1,285/1,286)
   - Point estimate negative in both TRAIN (−0.061R) and VAL (−0.191R)
   - Verdict: REFUTED at VAL; cadence bar fails all 7 criteria

---

## All Tested Ideas (Row: idea, cell#, universe/window, verdict ≤12 words)

| Idea | Cell(s) | Universe / Window | Verdict |
|---|---|---|---|
| Post-earnings drift (SUE-PEAD family) | F1 | US common, $5+, $1M ADV20$, 60-session hold, 2016–23 | 0 of 11 cells clear, closed |
| Short-interest reversal (Boehmer–Huszár) | A2 | US 8-K filers, FINRA short-int sort, 2018–23 | long leg negative; A2 −65.7 bps, t −2.05 |
| Dividend month premium | A4 | US common, predicted dividend payers, monthly, 2016–23 | −0.6 bps (t −0.09), MDE 13 bps |
| Net-share-issuance sort | A3 | US common, monthly NSI sort, 2016–23 | 0 of 2 cells clear G1/G2 |
| 52-week-high tilt | F5 | US common, 6-month hold 52wk high, 2016–23 | 0 of 2 cells clear |
| 12-1 / 6-1 momentum (12-month hold) | F3 | US common, momentum deciles, 1-month hold, 2016–23 | 0 of 6 cells clear, t max 1.45 |
| Weekly industry-adjusted reversal | F4 | US common, 1-week industry residual, 2016–23 | 0 of 1 cell clear, lottery artifact |
| Overnight effect on index ETFs | 1347–1350 | SPY/QQQ/TQQQ/UPRO overnight, 2016–24 | FAIL; VAL t 0.32–1.75; MDD −27% |
| Post-8-K item-2.02 drift | 1633–1636 | US 8-K filers, 20–40–60-session hold, 2023–24 | FAIL; 0 of 4 cells clear, ex-top-5 % negative |
| Pre-announcement run-up (8-K history) | 1646–1647 | US common, E−5→E+1 days, TRAIN 2020–22, VAL 2023–24 | FAIL; mean +10 bps, t 0.62, ex-top-5 % −71 bps |
| Overnight new-high long | 1550–1551 | US common $5+, $10M ADV20$, new-52w-high vol-gated overnight, 2019–24 | FAIL; ex-top-5 % −29 to −47 bps in every year |
| Catalyst-day continuation (3-day) | 1285–1286 | US common, post-catalyst long 3-day, 2025–26 | REFUTED; TRAIN −0.061R, VAL −0.191R |
| Insider Form 4 purchases | I1–I4 | US common, insider trades, 20-session hold, 2016–23 | FAIL; I3 closest at +0.6 %, t 1.28 |
| Pre-FOMC drift (W1, W2, W3, W4) | 1702 (10 cells) | SPY/QQQ/IWM, 1–4 weeks pre-FOMC, 1994–2024 | 2 of 10 pass (QQQ W1/W2); not recommended |
| Gap continuation (overnight) | K1 | US common $5+, $10M ADV20$, overnight gap 1–10 day hold, 2019–25 | FAIL; −136.4 bps, t −3.9 |
| 52-week-high breakout (volume gated) | K2 | US common, 52w high, vol-ratio gated, 1–20 day hold, 2019–25 | FAIL; −82.2 bps, t −6.9 |
| Short-term reversal (5-day, intraday) | K3 | US common, 5-day return, 1–20 day hold, 2019–25 | FAIL; −28.5 bps; corrected one positive cell |
| Overnight continuation (20-day MA) | K4 | US common, overnight return, 1–20 day hold, 2019–25 | FAIL; −79.0 bps, t −35.1 (ITCH bias adjusted) |
| Uptrend pullback (SMA5 entry) | K5 | US common, SMA5 entry, 5–50 day hold, 2019–25 | FAIL; −59.6 bps, t −9.3 |

---

## Ideas Never Tested Here

- **Earnings date (absolute day)** — flag on earnings calendar, e.g. straddle the release or sell volatility
- **EDGAR 8-K item 5.02 / 3.02** — specific item types (debt, material agreements) not tested as single signals
- **Buyback announcements** — open-market repurchase 10b5-1 continuation (insider exit vs. repurchase parity untested)
- **Spin-off announcement** — pre-spin drift, post-spin underperformance
- **Index add/remove** — S&P 500 / Russell reconstitution effect
- **Dividend yield cross-section** — high-yield (5%+) long hold (only "predicted dividend month" in A4)
- **Sector rotation** — tactical seasonal within month/quarter (no sector logic tested)
- **Low-volatility anomaly** — long low-vol hold (only ad-hoc regimes tested on intraday books)
- **New 52-week high (standalone)** — not gated on volume ratio (tested volume-gated only; non-gated gate in K2 failed)
- **Seasonal (turn-of-month, day-of-week)** — systematic seasonal patterns not pre-registered
- **Pairs / relative value** — cross-correlated long/short within a pair or sector
- **Market regime** — VIX-conditional hold entry (intraday books have regime, multiday none)

---

## Standing Rules from Closures

1. **Multi-day means 1–20 session holds on decile-cut universe** — the edge must exceed the MDE band at $66K book
2. **Corporate-action contamination is load-bearing** — back-adjust prices, gate on raw close, key split-control on feature window
3. **Tail-dependence kills books** — every closure shows ex-top-5 % ≤ 0 or mixed sign across years (tail lottery, not edge)
4. **Intraday drift is often misattributed** — overnight mean includes the ITCH open bias (+20 bps on buy-side deciles); intraday/overnight decomposition is not free
5. **Long-only announcement premium does not replicate** — short-interest long (Boehmer–Huszár), earnings drift, pre-announcement all fail; the literature priors are out of sample
