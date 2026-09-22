# PREREG + SPEC — Short the failed HOD break. Cells 1,357–1,358 (+ placebo diagnostics)

Owner 2026-09-21/22. Fact on the book: ~60 % of HOD breaks fail (stop rate 41–45 % + fades), the base rate is
stable across 2025 halves and 2026, and no causal feature predicts which (cells 1,343–1,355). The failed break is
therefore the reliable event; this pass takes it as the trade. The implementing agent reads ONLY this file, the
schema of `bars_sip.db`, the header of `features.csv`, and `cells.py` (grep/offset) for the split and R helpers.

## Population (causal)
Every HOD-break signal row of `features.csv` (day, symbol, entry_m, the break level L = the long `entry` price,
the long `stop`). Shortable proxy (borrow/locate is not in the data): keep rows with price ≥ $5 and adv20 ≥ 1M
shares; report the share excluded. SSR proxy: exclude rows whose price at the signal minute is ≤ 90 % of the prior
close (uptick rule in force); report the share.

## Signal, entry, stop, exit (frozen)
* **Failure**: the first 1-minute bar with `close < L` within K = 10 minutes after the long signal minute
  (bars strictly after entry_m). No failure within K → no trade.
* **Entry**: SHORT at the OPEN of the bar after the failing bar (obtainable; never the level itself).
* **Stop**: the post-break high H = max(high) over the bars from entry_m through the failing bar, touched →
  fill at H; if a bar opens above H, fill at that bar's open (gap-through). R = H − entry (must be ≥ 0.3 % of price,
  else no trade; report the share).
* **Exit**: cell 1,357 — cover at the OPEN of the 15:55 bar. Cell 1,358 — cover at entry − 2R when a bar's low ≤
  entry − 2R (fill at the target; if a bar opens below it, at that open), else 15:55 open.
* **Cost**: the study's own per-signal NBBO half-spread where available (`fetch_nbbo` output; use the signal
  minute's half-spread as the proxy for the short minute) charged on both legs, plus 2 bp per side of stop/target
  slippage; report gross and net. Coverage < 80 % → cost from the minute-of-day table is NOT allowed; VOID instead.
* One short per symbol-day (the first failure). No refill, no re-entry.

## Placebo decomposition (diagnostics, reported beside the cells)
D1 universe bound: the same short rule fired at the SAME name-day at a random minute in 10:00–14:00 (seeded), stop =
the prior 10-minute high, same exits. D3: the same rule on the same day for a random OTHER symbol of the
population. The cell's edge must exceed D1 and D3 by ≥ +0.10 R on VAL, else it is "shorting gappers", not the signal.

## Splits, statistics, report (`FAILED_BREAK_SHORT_REPORT.md`, trades CSVs)
TRAIN 2025 (halves H1/H2), VAL 2026-01..05, TEST sealed. Per cell per split: n, trades/week, gross and net mean R,
sd, iid and day-clustered t, win rate, ex-top-5 %, capped-at-5R, share stopped / target / eod / gap-through,
the placebo lines, cadence block on VAL (`scripts/cadence_bar.py`), and the week-by-week VAL P&L at R = $100.

## Pass bar (all)
1. net mean R ≥ +0.10 on TRAIN and VAL, VAL day-clustered t ≥ 2;  2. both TRAIN halves > 0;  3. ex-top-5 % > 0 on
both splits;  4. exceeds D1 and D3 by ≥ +0.10 R on VAL;  5. ≥ 3 trades/week on VAL after the shortable and SSR
proxies;  6. cadence C3/C4 on VAL.
Pass → independent rebuild (Haiku) → live PREREG for a DRY short book (`[HOD-S DRY] WOULD SHORT`, zero orders,
shortable/ETB check at signal time logged). Fail → report the MDE and the placebo split; the HOD loop moves to
the order-flow-imbalance filter (Databento, owner spend).

## Not allowed
K other than 10 as a cell; other stops; any long-side change; scoring TEST. Programme count 1,358 (+ 4 placebo
diagnostics, not cells).
