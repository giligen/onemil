# Pre-registration — F4 (1 cell) + F3 (6 cells), written 2026-09-18 BEFORE any return was computed

Panel: `data/panel_f3f4.npz` (all 4,870 `kind=='common'` Alpaca names + SPY, 2016-01-04 → 2026-09-18;
built by `build_panel_f3f4.py`). Splits TRAIN 2016-01→2021-12 / VAL 2022-01→2023-12 / TEST 2024-01→2026-09
**SEALED** (opened once, only behind a written `FREEZE.md` naming G2 survivors).

## 0. The ADV fix, done before any scoring (REPORT_F2_A1.md §9 defect 6)
`ADV20$` is rebuilt on the **RAW** panel (`vwap_raw × volume_raw`). The F2/A1 panel built it on the
split+dividend-ADJUSTED pull; dollar volume is split-invariant but the dividend factor scales price down
without touching volume, and that factor is a function of every dividend paid between the bar and today.
The liquidity gate was therefore mildly forward-looking. Effect measured on 200 random symbol-sessions and
on the whole membership grid, reported in `REPORT_F4_F3.md` §0 and `data/adv_fix_check.json`.
**Every family from here uses the RAW gate.**

## Universe (both families), at the decision close `t`
`kind=='common'` ∩ RAW close ≥ $5 at `t` ∩ **ADV20$(raw) ≥ $1M** ending at `t` ∩ the signal's formation
prices all finite. Features on the ADJUSTED panel; the $5 gate, share counts and ADV on RAW. `$10M ADV`
is a reported ARM, not a scored cell. SPY = the market.

## Execution convention (binding, unchanged from F2/A1)
**Close-to-close, `cls` both legs, one-session skip between signal and entry** — a `cls` order must rest
before the close it fills, so a signal computed from close(t) can only be filled at close(t+1). No opening
auction anywhere (Goyal–Jegadeesh–Wu JFQA 2026). No quoted spread on an auction cross.
Costs: SEC+TAF **0.4 bps on sells**; impact = `10 bps × (order$ / 1% of ADV20$)` both sides, order$ capped
at 1% of ADV; borrow **0.3%/yr** on any short leg, gated on today's `easy_to_borrow` (a disclosed
look-ahead). **Order$ scales with the ACTUAL position count** (REPORT_F2_A1.md §9 defect 3):
`order$ = $66,000 / n_positions in that leg on that rebalance`, not a fixed $3,300. Secondary arm: 5 bps/side.
**Costs are charged to the book only, never to the benchmark** (defect 4) — the benchmark series is gross.

## Split convention (adopted from REPORT_F2_A1.md §8b)
**Sealed**: a trade whose EXIT falls outside the split is not in that split. VAL is reported BOTH ways
(sealed and complete), because in F2/A1 the straddlers averaged +1,875 bps and flattered VAL with
TEST-window prices.

## F4 — weekly industry-adjusted short-term reversal. **1 long-only cell.**
Da–Liu–Schaumburg (Mgmt Sci 60(3) 658-674, 2014). Prior DOWNGRADED by AMENDMENT 3(a):
Chen–Cohen–Liang–Sun (JEF 82, 101608, 2025) find weekly US reversal only conditional on **high MAX**
(1.66%/wk vs 0.65%/wk low-MAX) and only in high retail-order-imbalance periods; SEF 2023 reports the
unconditional US effect dead since 2000.
- Week-end session `W` = the last session of each ISO week. Signal at close(W):
  `r1w = close_adj[W]/close_adj[W−5] − 1`; industry mean = the equal-weighted mean of `r1w` over
  eligible names with the same **2-digit SIC**, requiring ≥ 5 such names; names with no SIC are DROPPED,
  never bucketed as "unknown" (DATA.md §6). Signal = `r1w − industry_mean`.
- Decile against the eligible universe of that week (the reference population, stated).
- **Cell `F4-LO`**: long the BOTTOM decile of the residual (the losers, which reverse).
  Entry close(W+1), exit close(W_next+1) — a one-week hold, non-overlapping.
- Benchmark: the equal-weighted eligible universe **excluding the book's decile** (the clean contrast
  pre-registered by REPORT_F2_A1.md §9 defect 5), gross of cost.
- **Mandatory by AMENDMENT 3(a), and pre-committed as to sign**: (i) the result split by **MAX quintile**
  of the entry universe (`MAX` = the largest daily return in the 21 sessions ending at W) — *if the profit
  is concentrated in the top MAX quintile that is a **NEGATIVE** result for this account*, because those
  are the lottery names our floors exclude and where costs are worst; (ii) the **share of P&L from names
  below a $10 raw close**; (iii) the **break-even cost at its ~100%/week turnover**, stated in bps per
  round trip and annualised, against Detzel–Novy-Marx–Velikov (JF 78(3) 1743-1775, 2023).

## F3 — 12-1 momentum + residual momentum. **6 cells.**
Jegadeesh–Titman 1993; Israel–Moskowitz (JFE 108(2), 2013) — the long leg is ≈50% of momentum profits and
there is **no reliable size relation**; Fama–French 2008 — pervasive in big caps; Jensen–Kelly–Pedersen
(JF 2023) — replicates out of sample across 93 countries; Daniel–Moskowitz (JFE 122(2), 2016) — crashes.
- Month-end session `M_k` = the last session of calendar month `k`. Monthly rebalance, one-month hold,
  non-overlapping. Entry close(M_k+1), exit close(M_{k+1}+1).
- **12-1**: cumulative adjusted return over months `k−11 … k−1` (11 months, the most recent month skipped)
  = `close[M_{k−1}]/close[M_{k−12}] − 1`.
- **6-1**: `close[M_{k−1}]/close[M_{k−6}] − 1`.
- **Residual momentum** (Blitz–Huij–Martens JEF 18(3), 2011; Blitz–Hanauer–Vidojevic IREF 69, 2020):
  monthly returns regressed on the **market** return over a rolling 36-month window ending at month `k`
  (≥ 24 valid observations required); signal = mean of the residuals of months `k−11 … k−1` divided by
  their standard deviation. **Declared deviation:** the published construction uses Fama–French **three**
  factors; we have no book-to-market and no shares outstanding, so SMB and HML cannot be built from this
  panel and the residual is a **market-model (CAPM) residual**, i.e. *idiosyncratic* momentum w.r.t. the
  market only. This is a weaker purge of the dynamic beta that Daniel–Moskowitz blame for the crashes, so
  the residual cells' crash-protection claim is tested in the weaker form. Evidence base is 2020.
- Deciles against the eligible universe at `M_k` (the reference population, stated).
- Cells: `F3-LS-12-1` (D10−D1), `F3-LO-12-1` (D10 long only), `F3-LS-6-1`, `F3-LO-6-1`,
  `F3-RES-LS`, `F3-RES-LO`.
- Benchmark for the long-only cells: the equal-weighted eligible universe **excluding D10**, gross of cost.
  L-S cells are self-benchmarked.
- **Momentum-crash drawdowns are a mandatory column**: the maximum drawdown of the monthly compounded
  equity per split and over TRAIN+VAL, plus the 2020-03→2020-06 and 2022 windows named explicitly. For a
  compounding account the drawdown IS the finding (Daniel–Moskowitz 2016).

## Mandatory columns on every one of the 7 cells
1. **As-is / ex-top-1% / ex-top-5%** — the portfolio REBUILT with those trades removed and the monthly
   series re-estimated (REPORT_F2_A1.md §9 defect 2: in F2/A1 roughly 1% of trades WERE the entire effect).
2. **Long-leg share** of the L−S spread (undefined, and reported as such, where the spread ≤ 0).
3. **Break-even cost** in bps round trip and as a multiple of the honest auction cost.
4. **Ex-January.**
5. **Trades/week at the $66K / 20-slot book, and additivity vs the live ORB book.**
6. **Newey–West t** (lag = the hold length in months, minimum 1) alongside the raw t, and the **MDE**
   (2 × SE of the monthly mean) in bps/month and in $/month at book size.
7. **Survivorship statement per cell.** The panel is 100% survivors (F2/A1: 3,806 of 3,807 scored symbols
   still quoted in 2026; `delisted_names.parquet` has ZERO ticker overlap with `universe.parquet` and
   those names were never priced; the "missing price → 0% return" rule lets an in-sample delisting break
   even instead of going to zero). This biases long-only UP and **hits F3 and F4 hardest** — F3's
   formation window is a year long and its D1 decile is exactly the delisting cohort. **The GROSS
   per-trade column is therefore NOT quotable** unless the PIT re-run below is completed for that cell.
8. **PIT survivorship re-run** on the delisting-inclusive Nasdaq tape
   (`research/fuckup_audit/N_databento/N3/xnas_daily.parquet` + `R_daily/xnas_daily_2024H1.parquet`,
   2018-05 → 2024-06), with a volume-confirmed split detector because that tape is unadjusted. Run for
   every cell whose as-is point estimate is positive on TRAIN.

## Gates
G1 TRAIN: t ≥ 2 on the benchmark-adjusted monthly series. G2 VAL: same sign AND ≥ 55% of months positive.
TEST once, only for G2 survivors named in `FREEZE.md`.
Permutation p: block bootstrap (block 3 months, 5,000 resamples) on the TRAIN excess, Sidak-adjusted
across the **7** cells of this stage.
**Cumulative multi-day scored-cell count: K 20 + N2 4 + R_daily 20 + F2/A1 8 + 7 here = 59.**

## Calibration prior (pre-committed, unchanged)
Chen & Velikov: 204 published anomalies net **~4 bps/month**, strongest ~10 bps before impact. On $66K
that is **$25–65/month**. Anything materially larger is a **leakage suspect first, a discovery second**.
AMENDMENT 3(e) adds, for F4 specifically: our ≥$5 / ≥$1M floors delete exactly the lottery corner where
the live published effect lives, so F4 is *expected* to measure materially less than 1.66%/week.

## Phrasing rule
No cell may be reported as "no edge exists". Only: "no edge detectable in THIS universe, at THIS horizon,
at THIS book size, over THIS window, at THIS cost", with the MDE stated alongside.
