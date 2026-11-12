# PREREG 1,735 — the stop IS the cost: no-stop / soft-stop / dead-trade variants of the HOD-break long (owner 2026-11-12 "do all tonight")

FROZEN 2026-11-12 19:15 UTC before any number. Programme count on the HOD line: 1,734 → 1,735 (one cell, five
pre-committed variants, no sweep). Rules: no git, no service action, no orders, read-only on everything except
`research/hod_entry/1735_*`; NEVER open `research/bf_zero/bars_sip.db` (heavy-DB blackout until 20:05 UTC and this
cell does not need it). Never Read a file > 300 lines in full. Budget ≤ 45 calls. This task IS the owner's request.

## Why
Live measurement (cells 1,721–1,722): the HOD long earns +0.075 R GROSS at its levels and pays ≈ 0.15 R: stop slip
13–35 bps on 41 % stop-outs plus the entry chase. The exit lab (35 cells) moved exits within ±0.04 R but every variant
KEPT a hard stop. This cell removes or softens the stop and pays for the risk with SIZE, which the lab never did.
First grep `research/hod_exit_lab/PREREG.md` and `REPORT.md` for "time" / "no stop" / "MTM": if any variant below was
already scored there, say so in the RESULT beside the new number (disclosure, not a reason to skip).

## Data (cached, no DB)
`research/hod_exit_lab/paths.parquet` (day, symbol, m, hhmm, o, h, l, c, vwap, spy_close, minutes_since_entry — the
1-minute path of every signal from its entry minute to 15:55) and `signals.parquet` (day, symbol, entry_m, entry,
stop, target, R, r_pct, price, half_spread, spread_mean, split, half, wk; 15,656 rows: TRAIN 7,390 / VAL 4,745 /
TEST 3,521 — TEST stays SEALED, never scored). B0 reference = `research/hod_exit_lab/b0_trades.csv` (net_R per
signal). Reproduce B0 first from the paths with the exit-lab fill physics (stop fills at the stop, gap-through at the
bar open, target only if h ≥ target, both-touched = stop, else the 15:55 open; cost = half_spread on both legs + 2 bp
per side) and show the TRAIN mean matches b0_trades within 0.01 R before any variant is scored.

## Variants (all long at `entry`, all exit at the 15:55 open if nothing else fires; R units = the B0 R of the same
signal, i.e. baseline-R, so every ΔR is comparable; size factor s multiplies the P&L AND the cost)
* **V1 no-stop, half size** (s = 0.5): no stop, no target, 15:55 exit.
* **V2 no-stop + target, half size** (s = 0.5): no stop; sell at `target` if h ≥ target; else 15:55.
* **V3 soft stop, full size** (s = 1): no hard stop; exit at the NEXT bar's open after the first 1-min CLOSE ≤ entry − 2 R
  (a close, not a touch — intrabar sweeps cannot stop it); target 2 R kept as in B0.
* **V4 dead-trade time stop** (s = 1): B0 exactly, plus: if at minutes_since_entry = 20 the close < entry + 0.3 R, exit
  at the open of the next bar.
* **V5 V4 at 60 minutes** (s = 1): same rule at minute 60.
Costs: V1/V2 pay half_spread × 2 + 2 bp × 2 like B0 (times s); V3 soft-stop exits at the open (no slip beyond the
spread model); V4/V5 time exits likewise. State the cost in baseline-R per variant.

## Scoring (per variant; TRAIN H1, TRAIN H2, TRAIN, VAL; never TEST)
Paired ΔR vs B0 per signal: mean, day-clustered t (cluster = day), iid t, ex-top-5 % of ΔR (drop the 5 % largest
positive ΔR), winner-capped (R capped at +3 before differencing). The variant's OWN book: mean net R, day-t,
ex-top-5 %, stop-out share, green-week share, weekly P10, max drawdown in R, fills/wk unchanged. Placebo: 300
shuffles of the (day) label → percentile of the observed ΔR on VAL.

## Pass bar (pre-committed)
A variant PASSES only if: paired ΔR ≥ +0.05 R with day-t ≥ 2.0 on TRAIN AND on VAL; ex-top-5 % ΔR ≥ 0 on both; both
TRAIN halves have the same sign of ΔR; placebo percentile ≥ 95 on VAL; AND the variant's own VAL mean net R ≥ +0.05
with day-t ≥ 2 (a less-negative book is not a book). A pass unseals nothing — TEST is scored only after an
independent rebuild agrees, by a later agent, never in this cell.

## Output
`research/hod_entry/1735_score.py` (end-to-end, prints every table), `1735_per_signal.csv` (day, symbol, split, half,
b0_net_R, v1..v5 net_R in baseline-R, exit reason per variant), `1735_summary.csv` (variant × read × the metrics
above), `RESULT_1735.md` ≤ 350 words: the B0 reproduction check, the tables, PASS/FAIL per variant against the bar,
and the caveats (cost model, s = 0.5 halves the dollar book, soft stop obtainability at the next open). Return ≤ 120
words: per variant VAL paired ΔR, day-t, own mean R, PASS/FAIL.

## Independent rebuild (second agent, has not read 1735_score.py)
From this prose: recompute V1 and V4 per signal on TRAIN and VAL from paths.parquet + signals.parquet; compare to
`1735_per_signal.csv` on (day, symbol) to 0.001 R; mismatches > 0 rows = the number does not reach the owner.
Write `REBUILD_1735.md`.
