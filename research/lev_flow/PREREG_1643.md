# PREREG — cells 1,643–1,645: LEVERAGED-ETF REBALANCING FLOW INTO THE CLOSE (last-half-hour momentum on big-move days)

FROZEN 2026-09-29 05:20 UTC before any number. Programme count: 1,642 → 1,645. Free queue #1 (`research/ideas_web/
RANKED_20260928.md`; idea 3 of `research/IDEAS_20260928.md`). No paid data.

## Mechanism (documented: Cheng & Madhavan 2009; Tuzun 2013; Shum, Hejazi, Haryanto & Rodier 2016)
A daily-rebalanced L× fund must trade (L² − L) × AUM × r_t of its underlying at the close in the DIRECTION of the day's
move — long and inverse funds both buy after an up day and sell after a down day. The flow is known by mid-afternoon
(r_t is observable, AUM is slow) and is executed in the last 30 minutes, increasingly through the closing auction.
Prediction: on large-move days the underlying continues in the day's direction over the last half hour, and part of
it reverts at the next open. The flow is largest where leveraged AUM is large relative to the underlying's liquidity:
semiconductors (SOXL/SOXS), Nasdaq-100 (TQQQ/SQQQ), Russell 2000 (TNA/TZA), financials (FAS/FAZ), energy (ERX/ERY),
gold miners (NUGT/DUST), biotech (LABU/LABD), long Treasuries (TMF/TMV), S&P 500 (UPRO/SPXU/SPXL/SPXS).

## Data (free)
Alpaca minute bars (adjustment all) 2016-01-04 → 2026-09-04 for the underlying ETFs SMH, QQQ, IWM, XLF, XLE, GDX, XBI,
TLT, SPY; Alpaca daily bars for the official close and next open. Early-close sessions (13:00 ET) excluded; FOMC
decision days flagged (a report line with and without them — the 14:00 move inflates |r| for a different reason).
Splits: TRAIN 2016–2020, VAL 2021–2023, TEST 2024-01..2026-09 sealed (one read for the single best passing cell).
MDE printed with the bar: with ≈ 400 VAL events at a 30-minute SD of ≈ 40 bps the SE is ≈ 2 bps, so a t ≥ 2.5 bar
detects ≈ 5 bps — reachable; if the realised n or SD differ, recompute and print before reading the verdict.

## Signal and trades
r_t = the underlying's return from the prior official close to the 15:30 ET minute bar's close (the last price known
at 15:31:00). The trade fires when |r_t| ≥ 1.0 % (pre-declared; the |r| tercile table among firing days is the
mechanism check — the effect must rise with |r|). Direction = sign(r_t).
* 1,643 LONG-SIDE, into the close: on up days buy at the 15:31 bar's OPEN (a marketable order in a liquid ETF; 1 bp
  half-spread for SPY/QQQ/IWM/TLT, 2 bps for the sector ETFs), exit with an MOC at the official close (0.5 bp).
* 1,644 report-only: the same entry held to the next open (the reversal reading), and the 15:45 entry variant.
* 1,645 SHORT-SIDE mirror on down days (short the ETF at 15:31, cover MOC; ETFs are easy-to-borrow, borrow ignored
  intraday; SSR flagged on the day it applies).
Report per cell and split: n events, events/week (pooled across underlyings), mean net bps, day-clustered t (events on
the same day across underlyings are one cluster), ex-top-5 % / ex-top-1 %, winner-capped +50 bps, the |r| tercile
table, per-underlying and per-year tables, the SPY-only line (the most arbitraged), FOMC-excluded line, the worst
day, and the MDE line.

## Pass bar (frozen; VAL 2021–2023, per cell)
Mean net ≥ +5 bps per event, day-clustered t ≥ 2.5, ex-top-5 % > 0, ≥ 3 events/week pooled, TRAIN same sign t ≥ 1,
the |r| tercile table monotone on both halves, and positive in at least 5 of the 9 underlyings (not one name's story).

## Independent check and consequences
Rebuild from the prose (event set Jaccard ≥ 0.98, bps within 1); refuters: the bar-timestamp convention (the 15:30
bar closes at 15:31:00 — nothing from a later bar in the signal; the entry is the 15:31 bar's open), the official
close vs the 16:00 bar (MOC fills at the daily close), early closes, FOMC days, split/dividend adjustment, the
cross-underlying clustering, tails. PASS → paper on the ORB paper account for 4 weeks (the MOC leg exists in
`trading/eod_exit.py`), then live at $10K notional per event on the owner's word (the loss is bounded by a 30-minute
move: ≈ $100 on a bad day). FAIL → closed with the tercile table on record.

## Not allowed
Tuning the 1.0 % threshold, the 15:30 decision time, the exit or the underlying list after a number.
