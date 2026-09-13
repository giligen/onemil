# Ignition from zero — pre-registered research design (2026-09-13, before any run)

Owner: "I'm sure ignition has an edge. Go back to zero, 15–20 hypotheses on signals, type of stocks, similar stocks, news, whatever. Find a real valid strategy based on the principle of ignition; can be different. No bias, no look-ahead, no god mode. Verify, scrutinize."

## The principle
A stock that accelerates sharply intraday from a flat open, on a reason, continues far enough to pay a structural stop. Everything below is a candidate answer to *which* accelerations, *when*, and *held how*. The event is the +X% cross from the RTH open on a 1-minute bar; the stop is the structural low; that skeleton stays.

## What we already know (facts, not hypotheses)
- Bare +10% cross, hold to close: per-trade R ≈ 0 in every era; 3% of trades carry everything.
- Sibling-cohort confirmation is the only cell with positive ex-tail expectancy in all three eras, and it lost −$51K over Jun 1–Sep 11 2026 (356 trades, WR 41%). Regime-dependent with no in-advance signal found yet.
- Pre-trigger news: no lift to the average trade, 1.5× monster rate.
- Tightening the stop (0.5R/0.75R) loses dollars and makes the book tail-only. Hold-to-close gives the intraday move back; partial@+1R/BE is the only exit positive ex-tail in all eras.

## Data (all point-in-time; nothing computed from anything after the trigger minute)
- 1-min bars 2025-01-02 → 2026-09-11: `data/cache.db` + Databento PIT top-up (`research/ignition_capcheck/topup.db`, delisted names included → survivorship fixed).
- Daily bars (prior days only), premarket 1-min bars where cached, SPY 1-min bars.
- News: Alpaca news API backfill, window prev-day 15:00 ET → trigger minute; 2026 done (`research/ignition_news/news_2026.csv`, headlines kept), 2025 being fetched now.
- Sibling/anchor map: `trading/orb_asset_class.py` (wrappers ↔ underlyings), plus a same-sector map if obtainable offline.
- Float / shares outstanding: universe snapshots (patchy) — hypotheses that need it are marked and reported with coverage.

## Splits — fixed now, never moved
- TRAIN: 2025-01-02 → 2025-12-31 (thresholds, if any, are chosen here ONLY).
- VALIDATE: 2026-01-01 → 2026-05-31.
- TEST: 2026-06-01 → 2026-09-11 (the quarter that killed the current design; the one that matters).
A hypothesis is a finding only if the sign of its effect holds in TRAIN and VALIDATE independently, and the TEST quarter is positive with the tail removed. Anything tuned after seeing TEST is discarded. 20 hypotheses at p=0.05 → ~1 false positive expected; we do not ship a lone survivor without a mechanism.

## Metrics (every table, every era, every hypothesis)
Per-trade R with the tail removed (tail = trade P&L ≥ 2× its $ risk), WR, weeks green, P&L at $500 risk with the model's participation cap, worst week, n. Never an annual average alone; always the week-by-week of TEST.

## The hypotheses (H1–H20; entry event = +X% cross unless stated; exit = partial 50%@+1R/BE, remainder lock, unless the hypothesis is about the exit)
Velocity / timing
- H1 Speed of ignition: minutes from open to the cross ≤ 15 vs > 15 — faster igniters continue more.
- H2 Cross level: +5 / +7 / +10 / +15% — is 10% the right event, or a late one?
- H3 First-pullback entry: enter on the first higher low after the cross instead of at the cross (avoid the immediate fade).
- H4 Trigger-bar quality: close in the top third of the trigger bar's range vs bottom third (the ORB "touch-go" idea).
- H5 Ignition volume: trigger-bar volume vs the day's prior per-minute average ≥ 3× vs less.
Stock type
- H6 Price band: $2–5 / $5–10 / $10–20 / >20.
- H7 Float (where known): < 20M vs ≥ 20M.
- H8 Prior-day range: day-2 continuation (prev range ≥ 10%) vs fresh (the ORB PDR mechanism).
- H9 Distance from the 20-day high at the cross: breakout into new highs vs bounce inside the range.
- H10 Wrapper vs common stock (2x ETFs ignite with their underlying).
Similar stocks
- H11 Sibling cohort size ≥ 2 (the known cell) — re-measured under the new splits as the control.
- H12 Sympathy lag: when a cohort leader ignites, enter the *lagging* sibling at its own cross within 30 min.
- H13 Theme heat: number of names igniting in the same 30 minutes ≥ 3 (market-wide momentum morning) vs isolated.
News / catalyst
- H14 Headline class (offline keyword classes, no LLM): offering/dilution/reverse-split vs contract/FDA/earnings/M&A vs none — dilution headlines fade, others continue.
- H15 News recency: article ≤ 60 min before the trigger vs premarket vs none.
- H16 Premarket dollar volume ≥ $1M (intent before the open) vs less.
Market context
- H17 SPY 5-min return at the trigger ≥ 0 vs < 0; SPY 3-day range (calm vs violent tape).
- H18 Time of day: crosses 9:35–9:50 / 9:50–10:10 / 10:10–10:30 / 10:30–11:30 (the window itself is a hypothesis).
Holding
- H19 Structure trail: trail the stop under each new 1-min higher low after +1R instead of the fixed lock.
- H20 Time stop keyed to velocity: fast igniters exit at +60 min, slow ones hold — the hold-vs-time rule as a function of H1.

## Procedure
1. Build ONE candidate table: every symbol-day with a +5% cross in 9:35–11:30 ET (superset of every H2 level), all H-features at the trigger minute, all exit variants re-simulated on bars. Single ulimit-capped process, resumable, verified against capsim on the +10% subset (must match to 4 decimals).
2. Score each hypothesis on TRAIN; keep those with a sign; confirm on VALIDATE; only then look at TEST, once.
3. Combine at most 3 surviving features into a candidate book; re-run it as ONE rule set on TEST with week-by-week.
4. Write REPORT.md with every table, the failures included, and a shippable spec or the sentence "no valid strategy found".

Telegram status hourly while it runs; questions for the owner recorded in memory, not blocking.
