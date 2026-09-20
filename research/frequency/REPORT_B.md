# REPORT_B — Cell 1,297: bull flag re-entry at recalibrated cost

**Verdict: NO-GO / UNTESTABLE as specified — BT_ALLOW_REENTRY has zero
effect at Stage-2.** No trades were added; the pass bar cannot be evaluated
because the added set is empty.

## What happened
Two Stage-2 runs against the read-only cache
(`data/bull_flag_cache_causal_full_20260905.csv`, `BT_ALLOW_REENTRY=0` vs
`=1`, 2025-01-01..2026-05-31, capital $50K / risk $2K) produced **byte-
identical output**: 47 trades, $142,723.30, `r0.equals(r1) == True`. Zero
added `(symbol, date, entry_time_et)` keys.

## Root cause (code-verified)
`BT_ALLOW_REENTRY` only sets `runner.early_exit_after_trade = False` on a
`BacktestRunner` created inside the per-symbol-day SIMULATION loop
(`batch_backtest.py:1357` parallel-worker path; `:1928` cache-build branch).
Both sites execute only when Stage-1 walks bars for a symbol-day, i.e.
during `--build-cache`. Stage-2-from-cache (`BT_CACHE_PATH_OVERRIDE`) never
constructs that runner — it reads a finished per-trade CSV and applies
filters. The env var is a **cache-build-time flag**, not query-time; this
cell's "Stage-2 only, read-only cache" constraint makes it structurally
unobservable without a `--build-cache` run (prohibited here).
Corroborating: the cache already has 897 raw rows / 892 unique symbol-dates
— 4 symbol-dates carry 2 legs (e.g. KXIN 2025-12-29, both survive Stage-2
today, independent of the env var). Multi-leg days already flow through
Stage 2 whenever the cache happens to contain them; `BT_ALLOW_REENTRY`
controls nothing on top of that at query time.

## What was still run (per the letter of the task)
The reentry-on Stage-2 book was re-priced at cost setting M — entry exactly
via `planned_entry * (1+0.0023235)` (M's 23.235 bps target, matches
`research/exec_cost/recal.py`'s BF-M entry target); exit via a **flat-rate
proxy** (recal.py needs real per-trade quote spreads, `sp_e`/`sp_x`, which
do not exist in the Stage-2 CSV — `CSV_HEADERS` has none). Proxy: recover
50% of the assumed 0.3% stop-exit slippage table for `stop`-type exits, 0
elsewhere. **This is the recalibrated BASELINE book (no added set exists to
report — it is identical to the reentry-off book).** Total at cost M:
$162,287 vs $142,723 raw (cost recal alone helps, unrelated to re-entry).

Cadence (`scripts/cadence_bar.py`, recalibrated book, no reentry effect):
```
TRAIN: C1 fail(8.0/18.4wk) C2 pass(bleed P90 6.56R) C3 fail(P10 -0.94R,MDD 4.72R)
       C4 pass(green 70%>50%) C5 fail(0.60/wk) C7 fail(4 cycles)
VAL:   C1 fail(8.0/8.0wk) C2 fail(bleed P90 -4.08R) C3 fail(P10 -1.38R,MDD 6.83R)
       C4 pass(green 73%>50%) C5 fail(0.68/wk) C7 fail(1 cycle)
```
3/6 fail TRAIN, 4/6 fail VAL even before re-entry — 47 trades/17mo
(~0.65/wk) is thin regardless of the cost question.

## Pass-bar check
Added-set net R/trade: **N/A — 0 added trades, both splits.** All four
pre-committed bar items fail by construction (no added set -> can't clear
+0.10R; stacked $ unchanged; MDD unchanged; trades/week unchanged, not up
30%). **NO-GO.**

## Caveat
Code-verified structural finding, not a P&L claim. Exercising
`BT_ALLOW_REENTRY` for real needs a `--build-cache` run (owner sign-off,
never-overwrite-cache rule) plus a real quote-spread exit-cost method
(not the flat-rate proxy above) before any $ number is trusted.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PKSsd3LjBkXYzPECLNYR5W
