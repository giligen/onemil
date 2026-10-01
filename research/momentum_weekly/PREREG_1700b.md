# PREREG — cell 1,700b: weekly momentum sleeve, literature-definition universe fix

Motivation: cell 1,700 (`RESULT_1700.md`) found the naive top-N-by-trailing-return universe is dominated by
micro-cap hype names and at least one outright warrant (QBTS+) — not a sign/alignment bug, a universe-definition
gap. This cell fixes the universe to the literature's definition (CRSP-style domestic common stock, a size/liquidity
floor as a market-cap proxy, standard portfolio widths) BEFORE re-running, per the multiplicity rule (no filter
changes after seeing numbers).

## Universe recipe (pre-declared mechanism, not a fit)
1. Same point-in-time panel as 1700 (Databento EQUS.SUMMARY parquet, data/research/databento, delisted included,
   zero fetches). Drop test tickers `^Z[A-Z]ZZT$` and null symbols.
2. **Common-stock filter (the fix):** join each (symbol, bar-month) to Databento's point-in-time definitions feed
   (`data/research/databento/pit_definition/def_YYYYMM.parquet`, field `security_type`), matched by calendar month
   so a reclass is knowable only from that month forward. Keep **`security_type == 'C'`** only. Decoded empirically
   against known tickers before freezing (AAPL/QBTS=C; SPY/QQQ/IWM/TQQQ/leveraged single-stock ETFs=Q=ETF;
   QBTS+=W=warrant; ABR-D/ACGLN=P=preferred; AACBU/AACT==U=unit; AACBR=R=rights; AB/ARLP/BBU (MLPs)=L;
   BPT/CRT/PBT (royalty trusts)=V; ACN/ABEV/AEG (foreign ordinary/ADR)=O/A; small mixed REIT/closed-end-fund
   bucket=S). This is the literature's CRSP-10/11-style domestic-ordinary-common convention: ETFs, preferreds,
   warrants, units, rights, LPs, trusts and foreign ordinary/ADR shares are excluded as a class, not by proxy.
3. **SPAC-name filter (the fix, since a SPAC's own common unit is still `security_type=='C'`):** also exclude any
   symbol whose `name` field in `data/research/orb_asset_class_map_20260711.csv` contains "acquisition"
   (case-insensitive) — the standard blank-check-company name marker. Symbols absent from that map cannot be
   name-checked (undercount only; state the gap, do not patch it after seeing results).
4. Size/liquidity floor as a market-cap proxy (**no market-cap field is on disk**): close ≥ $10 AND 20-trading-day
   average dollar volume (adv20) ≥ $20M, both measured at the signal date, same mechanism as 1700 with new
   thresholds.
5. ETFs are now excluded entirely by the `security_type` filter (step 2) — the old `orb_asset_class_map` "wrapper"
   tag (which mis-tagged plain SPY/IWM as wrapper, per 1700's own caveat) is NOT reused for exclusion, only its
   `name` column is reused for step 3. SPY is pulled before all filtering, benchmark-only, never a candidate.

## Signal and portfolios (fixed)
Signal = 12-1 only (M2: `close[t-21]/close[t-252]-1`, trading-day lookback, measured at the signal date = Friday
close for weekly books, last trading day of the prior month for the monthly book). Four books, all equal-weight,
all on the SAME filtered universe:
  a. **Decile-weekly**: top 10% of that week's eligible pool by signal (N_t = round(0.10 × pool size), floor 1).
  b. **Top50-weekly**: fixed N=50.
  c. **Top20-weekly**: fixed N=20.
  d. **Decile-monthly** (the literature's reference cadence): same decile rule, rebalanced on the first trading
     day of each month instead of each Monday, held one month.
Weekly books rebalance Monday open, held one week (as 1700); unchanged names pay no turnover cost. Costs: 5bps/side
+ half the daily high-low/close spread proxy (capped 20bps) on every traded dollar — identical to 1700.

## Windows (identical to 1700)
2025-07-01..2026-09-30 (panel starts 2024-07-01, 252-day lookback binds first availability to ~2025-07). Halves:
halfA 2025-07-01..2025-12-31, halfB 2026-01-01..2026-09-30. The monthly book will have very few halfA observations
(~6 months) — flagged as thin by construction, not patched.

## Reads (per book × window, both halves and whole) — identical list to 1700
Weekly/monthly return series; annualised return, volatility, Sharpe (annualised to the book's own frequency), max
drawdown, worst/best period, green-share; turnover and cost drag %/yr; beta and alpha vs SPY, alpha vs the
equal-weight eligible universe; count-matched null (1,000 draws from the SAME filtered pool each period, count-
matched on the period's realised N) → percentile of realised annualised return and Sharpe; realised median pool
size (the "universe size after filters") and realised median N for the decile books (sanity check against the
~100–200 expectation); concentration/capital note at $65K as in 1700.

## Pass bar (identical to 1700 — phase 1, a sleeve on its own bar)
Annualised alpha vs SPY ≥ +8%/yr AND weekly(or monthly)-Sharpe ≥ 1.0 on BOTH halves, null percentile(return) ≥ 95
both, cost drag < 4%/yr, max drawdown ≤ 20%. Pass → independent rebuild from prose, then paper sleeve at $20K.
Fail with a positive whole-window point estimate → phase 2 (free long history) is worth running; fail with a
non-positive point estimate → no phase-2 case on this read.

## Multiplicity
4 books × 3 windows = 12 reads + nulls (down from 1700's 27 — the variant and N sweep is retired by this cell;
only the pre-declared universe fix and the four portfolio widths above are tested). Not allowed after seeing
numbers: changing the security_type set, the SPAC-name pattern, the $10/$20M floor, N widths, costs, or windows.

## Output
`RESULT_1700b.md` (≤90 lines), `1700b_reads.csv`, `1700b_weekly.csv` (per-period detail, weekly+monthly rows),
`1700b_momentum.py` (copied from `1700_momentum.py`, original left unchanged), `1700b_momentum.log`. Agent returns
≤150 words.
