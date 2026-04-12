# MACD Wave Pipeline Latency Audit

Scope: latency from "bar closes at Alpaca" to "order arrives at Alpaca" for MACD wave entries.

## Current hot path (serialized)

```
bar closes at T
  ↓ 0–60s   main-loop sleep (macd_wave.py:274, time_mod.sleep(60 - elapsed))
  ↓ 300–500 ms   scan_for_movers: batched get_latest_trades for N=200-chunked universe
  ↓              (≤5 chunks for full universe)
  ↓ for each crossed_stock (serial, usually 1–5):
  ↓   200–400 ms get_1min_bars(sym)         — macd_wave_engine.py:491
  ↓   < 20 ms    MACD compute (pandas EWM)  — macd_wave_engine.py:501–505
  ↓   100–250 ms _get_smart_limit_price(sym) → get_latest_quote
  ↓              — macd_wave_engine.py:545 → :723
  ↓ 300–700 ms   submit_bracket_order       — macd_wave_engine.py:672
  ↓              (only if entering)
order at Alpaca
```

### Measured components

| Stage | Typical | File:line |
|-------|---------|-----------|
| Polling wake-up lag | 0–60 s, avg **~30 s** | macd_wave.py:274 |
| Batched trade fetch (full universe) | 300–500 ms | alpaca_client.py:get_latest_trades |
| Per-symbol bar fetch | 200–400 ms | alpaca_client.py:get_1min_bars |
| Per-symbol quote fetch | 100–250 ms | alpaca_client.py:get_latest_quote |
| Order submit | 300–700 ms | alpaca_client.py:954 submit_bracket_order |
| Fill detection (polled) | 0–60 s | check_exits → get_order (per cycle) |

**Critical-path total** for an entry signal, 3 crossed stocks being watched (1 enters):
- Best: 0 s polling + 400 ms scan + 3×(200+150) ms fetch/quote + 500 ms submit ≈ **1.95 s**
- Worst: 60 s polling + 500 ms scan + 5×(400+250) ms fetch/quote + 700 ms submit ≈ **64.5 s**
- Avg: **~33 s**, almost entirely owned by polling lag

The current instrumentation (Migration 10 + `analyze_slippage.py`) will confirm these numbers empirically once production runs with it.

## Concurrency / latency opportunities (ranked)

### Tier 1 — biggest wins

#### T1.1  WebSocket 1-min bar feed instead of 60 s polling
- **Target**: the 30 s average polling lag.
- **Effort**: medium. `StopMonitor` already runs an Alpaca `StockDataStream` WebSocket (trading/stop_monitor.py). Reuse the pattern: subscribe to 1-min bars for all `crossed_stocks`, push bar-close events to a `queue.Queue`, drain on the main loop or react immediately.
- **Key issue**: bars arrive from Alpaca ~2–5 s after the minute closes (consolidation). That's still 25+ s faster than polling's 30 s average.
- **Win**: ~25–28 s reduction in average critical-path latency. Biggest lever.

#### T1.2  Parallel bar fetch via existing batch API
- **Target**: the per-symbol 200–400 ms × N serial fetches in `check_entries`.
- **Effort**: low. `alpaca_client.get_1min_bars_multi(symbols, lookback_minutes)` **already exists** at alpaca_client.py:841 (one REST roundtrip for all symbols). Replace the loop at macd_wave_engine.py:491 with a single batch call before iterating.
- **Win**: ~(N − 1) × 300 ms. For 3 crossed stocks = ~0.6 s; for 5 = ~1.2 s.
- **Safety**: no concurrency hazard — it's one REST call, computation remains sequential.

### Tier 2 — moderate wins

#### T2.1  Parallel order submission when ≥2 symbols confirm same cycle
- **Target**: serial 500 ms × N submit round-trips when multiple signals fire.
- **Effort**: low–medium. `ThreadPoolExecutor` over the confirmed entries. Important pattern:
  - Workers call `submit_bracket_order` and return result dicts
  - Main thread applies `self.open_positions[sym] = OpenPosition(...)` and DB writes serially from the returned results
  - This isolates the one thing that needs to be race-free (engine state) while still parallelizing the slow thing (REST round-trips)
- **Capacity check caveat**: `active_count >= self.max_concurrent` must be recomputed *before* dispatching to account for in-flight workers — reserve slots up front.
- **Win**: 500–1500 ms on multi-entry cycles. Low on most days (usually 0–1 entry/cycle).

#### T2.2  Parallel quote fetch for crossed_stocks
- **Target**: serial 100–250 ms × N quote fetches (one per watched stock).
- **Effort**: low. `ThreadPoolExecutor` on `_get_smart_limit_price`. No batch API exists for quotes (checked Alpaca SDK), so it has to be thread-parallel REST.
- **Win**: 300–800 ms on days with 3–5 crossed stocks.
- **Caveat**: Alpaca may rate-limit parallel requests. Use a small pool (4–8 workers) and honor 429s.

### Tier 3 — smaller wins

#### T3.1  TradeUpdateStream for fill events (replace get_order polling)
- **Target**: fill-detection latency (currently 0–60 s) and per-position `get_order` calls every cycle.
- **Effort**: medium. Alpaca's `TradingStream` pushes order state changes. Subscribe once at start; maintain a dict `{order_id → last_status}` populated by the stream callback.
- **Win**: fill detection goes from avg 30 s to <1 s. Frees main loop from N × get_order calls per cycle.
- **Indirect win**: tighter `order_filled_at` data in the new Migration 10 columns → better `submit_to_fill_ms` signal.

#### T3.2  DB writes on a worker thread
- **Target**: 10–50 ms sync SQLite writes in the entry hot path.
- **Effort**: low. Queue writes to a daemon thread; main loop fires and forgets.
- **Win**: small (10–50 ms). Worth doing only if T1.1/T1.2 land first and we want to shave the remainder.

#### T3.3  Speculative quote pre-fetch at pos_count=2
- **Target**: 100–250 ms quote RTT on the critical path when a signal confirms at bar 3.
- **Effort**: medium. Pre-fetch quote for symbols at bar 2; use within a short freshness window (≤2 s) on bar 3.
- **Win**: 100–250 ms on a subset of signals. Adds API traffic on non-entering symbols.
- **Reject unless**: instrumentation (T1.1/T1.2 deployed) shows the quote RTT is the remaining dominant latency.

## Safety pattern for all concurrent changes

Engine state that must not race: `open_positions`, `crossed_stocks`, `invalidated`, `trades_today`, `daily_pnl`, DB writes.

**Pattern (Option A — recommended)**: fire parallel I/O from workers → workers return result dicts → main thread applies state mutations serially. This keeps the concurrency surface to "I/O only" and preserves today's single-threaded reasoning about engine state.

**Anti-pattern**: locking the whole engine. Simple but serializes the thing we're trying to parallelize.

## Recommended implementation order

| Step | Change | Est. effort | Expected wall-clock win | Risk |
|------|--------|-------------|-------------------------|------|
| 1 | **T1.2** batch bar fetch | 0.5 day | 0.6–1.2 s | very low |
| 2 | **T1.1** WebSocket bars | 2–3 days | 25–28 s (avg) | medium — WS infra, reconnect handling |
| 3 | **T3.1** TradeUpdateStream | 1–2 days | fill detection 30 s → <1 s | medium |
| 4 | **T2.1** parallel submit | 1 day | 0.5–1.5 s on multi-entry cycles | low (needs capacity-reservation care) |
| 5 | (measure, decide) | — | — | — |

After steps 1 + 2, critical path for a new entry drops from ~33 s avg to ~3–5 s. That directly reduces `drift_bar_to_ask_bps` (the new Migration 10 column), which is the price drift component of slippage — exactly what the dev data can't currently show.

## What I'm deliberately NOT doing yet

- Changing `confirm_bars` or triggering on partial bars (the :30 idea). Orthogonal to plumbing and can't be backtested without sub-minute data.
- Rewriting the main loop as asyncio. Large refactor for modest gain once T1.1 lands. Most of the remaining latency after WebSocket bars is external (Alpaca REST RTT) — threads handle that fine.
- Caching quotes across cycles. Quote freshness matters; a cached quote is worse than a fresh one on momentum stocks.

## Open question for before we start coding

The 60-second polling loop also powers `scan_for_movers` (threshold crossing detection on the whole universe). T1.1 only covers bars for `crossed_stocks` (a small subset). We'd still need the 60 s scan to detect NEW crosses — but that's a different latency budget (a new cross doesn't have a pending signal yet; 60 s lag here is tolerable).

So **T1.1 is scoped to: WebSocket bars for crossed_stocks only**. Scan loop continues at 60 s. This is the right split — keeps the change small and attacks the hot path only.
