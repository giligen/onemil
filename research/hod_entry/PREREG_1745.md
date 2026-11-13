# PREREG 1,745 — the HOD-break spec, unchanged, on the LIQUID universe (synthesis D2: the population, not the signal)

FROZEN 2026-11-13 08:30 UTC before any number. Programme count on the HOD line: 1,744 → 1,745. Two agents: (A)
fetch, (B) score. Rules: no git / service / orders; read-only on every existing store; write only
`research/hod_entry/1745_*`, `research/hod_entry/1745_bars/` (parquet, scratch — not committed) and `RESULT_1745.md`;
`df` ≥ 5 GB before the fetch; atomic writes, resume; ≤ 180 requests/min; `nice -n 19`; background scripts with
log polling; never Read a file > 300 lines in full; budget ≤ 60 calls per agent; never print keys. This task IS the
owner's request ("deduce new strats" → `SYNTHESIS_20261113.md` D2); do not pivot.

## Why
The HOD book's gross at the level (+0.075 R live, ≈ +0.1 R in the ORB book in and out of regime) is paid away by a
round trip of 0.3–0.4 % of price on names where R ≈ 1.5 %. On the liquid top of the market the same geometry at
R ≥ 1 % of price costs 0.02–0.05 %. This is the only remaining idea that can change the SIGN of the book rather
than its cost. ONE spec: `trading/hod_break.detect` / `arm_state` / `HodBreakParams` defaults with the live floor
`min_r_pct = 1.5`, max_per_day 8, max_concurrent 4 — nothing re-tuned.

## Universe (every condition knowable at 09:30 ET of the day)
Top 300 US common stocks by 20-day average dollar volume as of the prior close, recomputed monthly on the first
trading day from `data/cache.db daily_bars` (read-only `file:...?mode=ro`); exclude ETFs/ETNs/wrappers
(`trading/bf_universe_filter.is_bf_eligible` or the asset `class`/name rule it uses), test tickers `^Z[A-Z]ZZT$`,
price < $5. Period 2026-01-02 … 2026-11-12 (≈ 215 days). Splits: TRAIN = Jan–Jun, VAL = Jul–Nov (no TEST — the
cell is a go/no-go for a PAPER instrument, which is the real out-of-sample).

## Fetch (agent A, `1745_fetch.py`)
Alpaca `GET /v2/stocks/bars?symbols=…&timeframe=1Min&feed=sip&limit=10000` (data.alpaca.markets, the live key's
data entitlement; adjustment raw), 09:30–16:00 ET per day, batched by symbol groups; one parquet per day in
`1745_bars/`; completeness gate per `feedback_fetch_completeness_gate`: a LOST count per day, VOID the day if < 95 %
of the universe has ≥ 300 bars. ADV20 per (symbol, day) from `daily_bars`. Log `DONE days=… symbols=… lost=…`.

## Score (agent B, `1745_score.py`)
Per symbol-day: run the spec exactly as `research/hod_entry/causal_arming.py` does for the paper book (arm on the
consolidation, resting stop-limit at the level with the 15 bps cap, fill when the bar's high crosses — report the
obtainability share where the next bar's open is above the limit; stop = consolidation low with the 1.5 % floor;
target 2 R; stop-limit exit 20 bps under; flat 15:55). Cost: entry = half the measured bar-level spread proxy is NOT
acceptable — use the NBBO half-spread from Alpaca `/v2/stocks/quotes` at the fill minute for a 400-fill sample to
calibrate, then the per-name median half-spread by minute-of-day; charge entry chase 8 bps (the live measurement)
and stop slip 13 bps (the live stop-limit measurement); report the book at 0 cost beside it. Books: uncapped and
capped (8/day, 4 concurrent). Report per split: n, fills/wk, r_pct median, mean net R, day-clustered t, ex-top-5 %,
winner-capped +3 R, stop share, green-week share, weekly P10, max drawdown, and the cadence-bar scorer
(`scripts/cadence_bar.py`) on the capped book. Placebo: same geometry from a random minute after the arm (300 draws).

## Pass bar (pre-committed)
Capped book: mean net R ≥ +0.10 with day-t ≥ 2.0 on TRAIN AND VAL, same sign both halves of TRAIN, ex-top-5 % ≥ 0,
placebo ≥ 95, ≥ 3 fills/wk, and the cadence bar's weekly P10 ≥ −2 R. Pass → independent rebuild from this prose
(population AND score), then a PAPER book spec on the HOD paper account with its own universe file — never live
from a cell. Fail → the liquid population is closed with the MDE stated; the HOD line is then closed at the engine
level (synthesis).

## Output
`1745_fetch.py`, `1745_universe.csv` (month, symbol), `1745_fetch.log`, `1745_score.py`, `1745_fills.csv`,
`1745_summary.csv`, `RESULT_1745.md` ≤ 400 words. Agent A returns ≤ 80 words (days, symbols, lost, PID/ETA); agent B
≤ 120 words (n, fills/wk, mean R / day-t per split, capped book, cadence line, PASS/FAIL).
