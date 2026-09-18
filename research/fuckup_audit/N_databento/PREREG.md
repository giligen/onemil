# Stage N — Databento pulls, pre-registered 2026-09-18 (before any pull)

Key: `DATABENTO_API_KEY` in `.env` (verified 9/18; never printed). Datasets available: EQUS.SUMMARY (daily, 2024-07→),
EQUS.MINI (mbp-1/tbbo/1-min, 2023-03→, multi-venue subset), XNAS.ITCH (everything, 2018-05→, Nasdaq only).
Priced 9/18: EQUS.SUMMARY daily 2024H2 $2.22; XNAS.ITCH daily ALL 2018-05→2025-01 $23.09; mbp-1 one symbol × 15 min ≈ $0.01.

## N1 — order flow at the ORB breakout minute (the CKS plan, `project_hod_break_ofi_filter_plan`)
**Population**: the 13,033 entered-inclusive ORB candidates (`research/fuckup_audit/D1_orb/candidates_dump.csv`,
7,402 fills + 5,631 modeled non-fills, 2025-01-02 → 2026-09-16). The book under test: `book_n8_q1on.csv` (8 slots,
215 picks) and the same pipeline re-run with the new feature — never the fills alone (that was the D1 leak).
**Data**: EQUS.MINI `tbbo` (every trade with the BBO at that instant) 09:29:30–09:46 ET per candidate symbol-day
(the 5-min range + the breakout window). tbbo gives both features from one pull. Budget ≤ $150; stop if the
priced cost exceeds it (`metadata.get_cost` first).
**Features (declared, all live-computable from Alpaca trade+quote streams)**:
- `ofi_range`: Cont–Kukanov–Stoikov order-flow imbalance summed over 09:30–09:35 from successive BBO updates,
  divided by the mean displayed depth (bid_sz+ask_sz)/2 over the window (depth-normalized).
- `tai_range`: trade aggressor imbalance 09:30–09:35 = (buy-initiated $ − sell-initiated $)/(total $), tick rule
  against the prevailing BBO (trade ≥ ask → buy, ≤ bid → sell, else midpoint sign).
- `ofi_break60`, `tai_break60`: the same over the 60 s AFTER the first trade above range_high (the breakout
  instant). Usable live only as a time-stop/hold rule (fill happens inside that minute) — evaluated separately.
- `spread_at_break_bps`: NBBO spread at the breakout instant (an availability-safe version of the spread gate).
Anything computed after the fill minute is NOT an entry feature.
**Cells**: 5 features × 2 uses (add to the composite z-score as an 8th feature with TRAIN-fit params; veto bottom
quintile post-ranking, no refill) = 10, plus the 2 break60 features as a 10-minute hold rule (exit at fill+10 min
if imbalance ≤ 0 and P&L < +0.25R) = 2 → **12 cells**. Splits TRAIN 2025 / VAL 2026-01..05 / TEST 2026-06..09;
gates PLAN §1 (G1 t ≥ 2 TRAIN, G2 VAL sign + ≥ 55% weeks green, TEST once); tail tests (ex-top-5%, +3R cap);
permutation p across the 12 cells; availability audit (the feature must be computable at 09:35:00 from data
timestamped ≤ 09:35:00 — assert on the raw timestamps); price-scale check (Databento raw vs Alpaca bars on 200
symbol-days: the breakout price must agree within 0.1%).
**Coverage caveat**: EQUS.MINI is a venue subset (1–10% of SIP volume on thin names — REPORT §6a of bf_zero). Report
the share of candidates with < 20 trades in the window; a feature that is empty on the thin half is not a feature.
**Deliverable**: `N1/REPORT.md` one page: per-cell TRAIN/VAL/TEST R/pick, the 8-slot book $ at stage size with and
without, worst month, MDD, the coverage share, the cell count. Survivor → independent rebuild before any engine work.

## N2 — daily history for the multi-day family K2 (52-week-high breakout on volume)
EQUS.SUMMARY ohlcv-1d ALL_SYMBOLS 2024-07-01 → 2024-12-31 ($2.22) appended to
`data/research/databento/equs_daily_2025_2026.parquet` → K2 signals computable from 2025-07 (252-day lookback).
Then XNAS.ITCH ohlcv-1d ALL 2018-05 → 2024-06 ($23) for the full lookback on Nasdaq names only — pulled ONLY if
K2 on the half-window clears G1. Cells: K/PREREG.md as declared (K2 × 2 holds × 2 books = 4). No new cells.

## N3 — single-stock simulator calibration (published stocks-in-play ORB, 2018-05 → 2023-12)
XNAS.ITCH daily (from N2) → the paper's universe rule (top-20 relative volume at the open, price ≥ $5) → 1-min bars
for those ~20 names/day (≈ 28K symbol-days, priced before the pull, budget ≤ $60). Run OUR simulator with OUR cost
contract on THEIR rule in THEIR period. Pass = same sign and same order of magnitude of R/trade per year as
published. Fail = the simulator or the cost contract is wrong and every summer null is suspect. One cell, no gate.

Order: N1 (money on the live book) → N2 → N3. Node rule: one heavy python at a time, `nice -n 10`, memory-capped.
