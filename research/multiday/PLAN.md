# Multi-day program — pre-registered 2026-09-18 21:40 UTC (owner: "Go deep on multiday. Get what data you need. Make it work.")

## Why the previous multi-day tests do not settle the question
Stage K / N2 / R_daily tested FIVE home-made long-only rules at 1–10 day holds on Nasdaq-listed names, charged a quoted
half-spread + 5 bps per side on auction fills, and had no earnings calendar. The published multi-day effects are none of
those things: they are (a) event-anchored (earnings), (b) long-SHORT cross-sectional portfolios, (c) held 20–250 sessions,
(d) executed in the opening/closing auctions where a resting order pays no spread (PLAN §1 item 4 of the audit program:
"auction-executed trades must NOT be charged a quoted spread" — the K family was over-charged on every trade). This
program tests the published constructions faithfully FIRST, then the executable long-only / small-account variants as
separate cells, and reports the gap between them. "Does the anomaly exist" and "can this account harvest it" are two
different questions and get two different answers.

## Data (all free; purchases only if the plan below cannot be met, priced first)
- **Prices**: Alpaca SIP daily bars 2016-01-04 → now, `adjustment='all'` (split+dividend adjusted; verified 9/18: AAPL
  2016-01-04 raw 105.35 / adjusted 23.69), for every currently listed US common stock and ETF (`get_all_tradeable_assets`).
  Store raw AND adjusted; features on adjusted, fills and share counts on raw. Opens/closes are the official consolidated
  session prints (the auction fill the engine would get with `opg`/`cls`).
- **Survivorship** (the known hole: Alpaca lists only current names): quantify it on two overlaps — Databento
  `pit_definition` 2024-07→now (all exchanges, `research/scripts/pit_listings.py`) and XNAS.ITCH daily 2018→2024
  (Nasdaq, delisted included, on disk under N3/R_daily). Report the delisting rate per year and re-run every survivor cell
  on the point-in-time Nasdaq panel as the survivorship check. A long-only result that does not survive the PIT re-run is
  survivorship.
- **Earnings calendar**: SEC EDGAR submissions API — 8-K filings with Item 2.02 (Results of Operations) = the earnings
  release, timestamped by `acceptanceDateTime` (point-in-time; verified 9/18: AAPL 45 of 103 8-Ks). Event day = the first
  session whose close is after the acceptance time.
- **Earnings surprise without a consensus vendor**: XBRL `companyconcept` `EarningsPerShareDiluted` quarterly facts with
  their `filed` dates (verified: AAPL 174 facts 2008→2026). SUE per Bernard–Thomas 1989 / Foster–Olsen–Shevlin 1984:
  seasonal random walk with drift, (EPS_q − EPS_{q−4} − drift) / σ of the last 8 seasonal differences, computable ONLY from
  facts filed before the event (availability audit on `filed`). Secondary surprise measure that needs no accounting data:
  the 2-day announcement return (Chan–Jegadeesh–Lakonishok 1996) — the "earnings announcement return" drift.
- **Industry** for the industry-adjusted reversal: SIC code from the EDGAR submissions record (2-digit).
- Test tickers excluded; names absent from `daily_bars`/Alpaca assets excluded; the raw close ≥ $5 gate on RAW prices.

## Families (the published constructions; each with its citation, exactly as published, then the executable variant)
- **F1 PEAD** (Bernard–Thomas 1989; Livnat–Mendenhall 2006 for the modern decay): sort into SUE deciles at each event, long
  D10 short D1, enter at the NEXT open after the 8-K acceptance, hold 60 sessions (also 20, 40). Executable variant: long
  D10 only, 20 names max, MOO entry, MOC exit.
- **F2 Earnings-announcement-return drift** (CJL 1996): sort on the 2-day announcement return, same holds, L-S and long-only.
- **F3 Cross-sectional momentum 12-1** (Jegadeesh–Titman 1993; Asness–Moskowitz–Pedersen 2013 for the modern universe):
  monthly rebalance, skip the last month, deciles, L-S; executable: long D10 only, 20 names, and the 6-1 variant. Report
  the 2016–2026 momentum crashes explicitly (Daniel–Moskowitz 2016) — the drawdown IS the finding for a compounding account.
- **F4 Short-term reversal, industry-adjusted** (Da–Liu–Schaumburg 2014 — the part that survives costs): weekly, L-S on
  the residual of the 1-week return vs the 2-digit-SIC industry mean; executable long-only leg reported separately.
- **F5 52-week-high** (George–Hwang 2004): monthly, L-S on nearness to the 52-week high; long-only leg.
- **F6 Overnight-vs-intraday cross-section** (Lou–Polk–Skouras 2019): long past overnight winners / short past intraday
  winners, monthly; executable: hold overnight only (MOC buy, MOO sell) on the long leg.
## AMENDED 2026-09-18 21:05 UTC by `research/multiday/LIT_REVIEW.md`, BEFORE any family was scored
The adversarial review (≈45 verified citations) found two defects in the grid above: it enumerated **22** cells while
claiming 26 (a wrong permutation denominator), and it omitted the crux column — the LONG-LEG SHARE, which the
literature supplies for only one of the six. It also re-ranked the families on post-publication decay. The amended
grid, budget unchanged at **26 cells**:

| family | cells | change and the citation that decides it |
|---|---|---|
| F1 PEAD/SUE | 6 → **2** | Martineau 2022 CFR 11(3-4): no significant drift for all-but-microcap stocks **after 2006**. Demoted to a declared null-replication, not a candidate. |
| **F2 announcement-return drift** | 6, **RUN FIRST** | Martineau explicitly does not test this measure; CJL 1996 stands and no verified post-2015 re-test exists either way. Now the highest-prior family. |
| F3 12-1 momentum | 4 → **6** | Israel–Moskowitz 2013 JFE 108(2): the LONG leg is ≈ 50% of momentum profits and there is no reliable size relation; Fama–French 2008: pervasive in big caps. +2 residual-momentum cells (Blitz–Huij–Martens). |
| F4 industry-adj. reversal | 2 → **1** long-only | Novy-Marx–Velikov: few >50%-turnover strategies survive costs; the published construction purges news with analyst revisions we do not have. Venue corrected to Mgmt Sci 60(3) 658-674 (2014). |
| F5 52-week high | **2**, January split out | George–Hwang: 0.45%/mo raw vs 1.23% ex-January; George–Hwang–Li 2018 JFE 128(1): q-factors *explain* price-to-high. Re-labelled a factor tilt, not an anomaly. |
| F6 overnight vs intraday | 2 → **1** measurement cell | Haghani–Ragulin–Dewey: 1 bp round-trip removes ~5 pts/yr from a 38%-gross overnight L-S; matches our own `overnight_auction.md` (+25/+15/**−12** bps) and M29 (+7.4/+6.7/**−13.4**). |
| **A1 earnings-announcement premium** | **2** | Johnson–So 2018 JAR: the pre-announcement bias REVERSES — buy before / sell at the event, the opposite trade to F1. The only addition whose PUBLISHED form is long-only. Same 8-K table, zero new data. |
| **A2 low short interest, liquid names** | **2** | Boehmer–Huszár–Jordan 2010 JFE 96(1): the significant abnormal return is on the **LONG** side and "often larger" than the short side, in liquid names. Free FINRA semi-monthly files — key on the DISSEMINATION date or it is a look-ahead. |
| **A3 net share issuance** | **2** | Pontiff–Woodgate 2008; FF2008 pervasive in big caps; Goto et al. survive costs. One extra EDGAR `companyconcept`. Lowest turnover in the review. |
| **A4 dividend-month premium** | **2** | Hartzmark–Solomon 2013; one extra Alpaca corporate-action type. Ranked last: **no post-2016 replication found** — run only if A1–A3 leave budget. |

**Rejected additions, with the citation that kills each** (do not re-propose without new evidence): index addition
(Greenwood–Sammon, 7.6% → 0.8%); pre-FOMC drift (Kurov et al. 2021, "essentially disappeared after 2015");
TSMOM / ETF trend (Huang et al. 2020 JFE, "little evidence… in- and out-of-sample"); betting-against-beta long leg
($1.05 per $1 sits in the bottom 1% of cap); failures-to-deliver (short side); analyst-revision drift (no vendor);
seasonality as a standalone family.

**Three columns are now MANDATORY on every cell** and a cell without them is not reportable: (1) the **long-leg share
of the L-S spread** — we can only trade the long leg, so a family whose profit lives in the short leg is dead for this
account whatever its t-stat; (2) the **break-even cost** (Stage O's rule: the charge at which the cell reaches zero,
stated in bps and as a multiple of the honest auction cost); (3) the **ex-January result**, because F5 and the
small-cap families are substantially a January effect.

## AMENDED AGAIN 2026-09-18 21:55 UTC — the FREQUENCY FLOOR (owner: "looking for 10+ trades a wk imo")
A fourth mandatory column, and it is a GATE, not a statistic: **expected trades per week at a $50–66K book**. The
owner's floor is **≥ 10/week**, and the reason is not impatience — it is resolution time. At 3 trades/month a book
cannot distinguish +0.2R from 0 inside a year; at 10+/week a quarter does it. Frequency is what converts a small
honest edge into both money and evidence.

This program's whole history is the two failure modes of that trade-off: **edge without frequency** (ORB, a real
edge at ~2.5 trades/week; BF P1 at 0.65/week — neither can ever resolve or compound fast enough on its own) and
**frequency without edge** (HOD-break at 25–30/week and −0.043R gross). The target is the intersection, and every
cell is now scored on it.

**Expected frequency per family** (stated BEFORE the runs, from universe size × event rate × the book's slots):
| family | mechanism of frequency | expected trades/wk | vs the ≥10 floor |
|---|---|---|---|
| F2 announcement-return drift | ~8,000 names × quarterly earnings ≈ 600 events/wk market-wide; top-decile + liquidity filter | **10–40** | **PASS** |
| A1 earnings-announcement premium | same event table, opposite side of the event | **10–40** | **PASS** |
| F4 industry-adj. reversal | weekly rebalance, 10–20 positions | **10–20** | **PASS** |
| A2 low short interest | FINRA semi-monthly dissemination → 2 rebalances/mo | 3–8 | FAIL as a standalone; keep as an overlay/filter on a passing family |
| F3 12-1 momentum | monthly rebalance, 10–20 positions | 3–5 | FAIL standalone |
| F5 52-week high | monthly rebalance | 3–5 | FAIL standalone |
| A3 net share issuance | annual/quarterly signal, monthly rebalance | 1–3 | FAIL standalone |
| F6 overnight | measurement cell only | n/a | n/a |
| F1 PEAD | declared null-replication | n/a | n/a |
| A4 dividend-month | monthly | 2–4 | FAIL standalone |

### SOFTENED 2026-09-18 22:05 UTC (owner: "ok ok we can do less don't constraint it can be an additive strat")
**The ≥10/week floor is a REPORTED COLUMN and a ranking factor — NOT a gate. No family is deleted for frequency.**
A low-frequency book qualifies on ADDITIVITY instead, and the resolution problem is solved at the PORTFOLIO level,
not per strategy: four uncorrelated sleeves at 3/week each give a 12/week portfolio that resolves inside a quarter
even though no single sleeve could.

**Additivity test, replacing the frequency gate — a slow family qualifies if:**
1. **It does not compete with ORB for the binding resource.** ORB's constraint is intraday slots and same-morning
   buying power at 09:35. A multi-day book that enters at the OPEN or CLOSE auction and holds overnight uses
   different capital at a different time of day — genuinely additive. State the overlap explicitly per family.
2. **Return correlation with the live ORB book is low** — compute it on overlapping months and report it; a sleeve
   that merely re-expresses ORB's exposure adds variance, not diversification.
3. **The COMBINED book** (ORB + the sleeve) is what must clear the exploration tier's condition 4, and the combined
   trades/week is what gets reported alongside the family's own.
4. Capacity is real at our size: $ per position at 1% of ADV, and the sleeve must not need more capital than the
   account has spare after ORB's per-position caps.

**Do NOT import Stage I's "stacking hurts" conclusion here without re-testing.** That result came from stacking
SAME-DAY INTRADAY books competing for the same slots, capital and 09:35 attention. Auction-executed multi-day
sleeves are a different structure and the interaction must be measured, not assumed — in either direction.

**Revised order of execution — fast families first because they can produce a live-testable book THIS quarter, but
every family is run: F2 → A1 → F4 → F3 → F5 → A2 → A3 → F1 (null-replication) → F6 → A4.**

## Splits, costs, gates
- **Splits**: TRAIN 2016-01→2021-12 (6 yr), VAL 2022-01→2023-12 (2 yr), TEST 2024-01→2026-09 (sealed; opened once behind
  `FREEZE.md` for G2 survivors). Power statement per split BEFORE returns are looked at (MDE in bps/month of L-S spread).
- **Costs, auction-faithful**: entries/exits at the official open/close via `opg`/`cls` → no quoted spread; charge SEC fee
  + FINRA TAF on sells (~0.4 bps), a market-impact allowance = 10 bps × (order $ / 1% of ADV$) capped at 1% ADV, and
  borrow cost on the short leg = the general-collateral 0.3%/yr plus a flag: a short is allowed only if today's Alpaca
  asset record is `easy_to_borrow` (survivorship caveat stated). Secondary cost arm: 5 bps/side flat.
- **Gates** (PLAN §1 of the audit program): G1 t ≥ 2 on TRAIN; G2 VAL same sign and ≥ 55% of months positive; TEST once.
  Tail tests (ex-top-5%, winners capped at +3× the cell's median win); permutation p across the 26 cells; availability
  audit on every field (`filed`/`acceptanceDateTime` ≤ decision time); the PIT Nasdaq re-run for every survivor; price-scale
  check raw vs adjusted on 200 keys; cell count reported (26 here; 70 cumulative on the multi-day line incl. K/N2/R_daily).
- **Executability** is scored, not assumed: Alpaca `opg`/`cls` TIF exist in the SDK and are UNUSED in this repo (zero
  `TimeInForce.OPG/CLS`); shorting needs margin + ETB. A survivor needs a daily-order engine — a build of ~2 weeks, listed
  in the report with its parts.

## Deliverables
`research/multiday/DATA.md` (panel sizes, survivorship rates, EDGAR coverage share of the universe, EPS-fact coverage,
the availability audit), `research/multiday/REPORT.md` (one page: the 26-cell table per split with L-S and long-only side
by side, MDE, drawdowns and worst month for every long-only survivor at a $50K book, the PIT re-run, the verdict in the
phrasing rule, and for any survivor the exact rule in prose for an independent rebuild + the engine build list), 3 lines in
`research/fuckup_audit/LOG.md`. Order: data → F1/F2 (event-anchored, highest prior) → F3 → F4/F5/F6.
