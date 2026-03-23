# OneMil - Day Trading System

Real-time stock scanner + automated trading system targeting Ross Cameron's momentum day trading strategy.

## Goals

1. Real-time stock scanner (gap ups, high relative volume, low float, $2-$30)
2. Automated paper trading via Alpaca
3. Go live

## Architecture

```
main.py                         CLI entry point (scanner + trading engine)
backtest.py                     Single-symbol backtesting CLI
batch_backtest.py               Batch backtest (universe scan → CSV report)

data_sources/
  alpaca_client.py              Alpaca API client (market data + trading)
  float_provider.py             Float share data via Yahoo Finance
  news_provider.py              News & sentiment analysis

scanner/
  realtime_scanner.py           Real-time stock screening
  criteria.py                   Filtering logic (price, float, volume, etc.)

trading/
  pattern_detector.py           Bull flag pattern detection (1-min bars)
  trade_planner.py              Trade plan creation (entry/stop/target/sizing)
  trading_engine.py             Main trading orchestration
  order_executor.py             Order submission/management
  position_manager.py           Position tracking
  stop_monitor.py               WebSocket stop monitoring + spread-based exits
  market_regime.py              SPY volatility + trend regime filter
  exhaustion_signals.py         Exhaustion exit signal detection (shared backtest/prod)

persistence/
  database.py                   SQLite database layer

batch/
  universe_builder.py           Stock universe construction

monitoring/
  logger.py                     Logging setup (UTF-8 for Windows)

notifications/
  telegram_notifier.py          Telegram alerts
  telegram_error_handler.py     Error notification
```

## Trading Strategy

Bull flag momentum pattern on 1-minute bars:

1. **Pole**: 3+ consecutive green candles with 3%+ gain
2. **Flag**: 2-5 red/doji candles, max 50% retracement, declining volume
3. **Breakout**: Close above flag high with 1.5x+ volume expansion

### Trade Plan Rules

- **Entry**: Buy-stop at breakout level (flag high), limit 2% above
- **Stop**: Flag low region, min 1% / max 4% risk, min $0.09 stop distance (tick-noise filter)
- **Target**: 2.5:1 R:R (overridden by trailing stop when enabled)
- **Position size**: fixed_risk mode, $2,000 risk per trade, max 10,000 shares
- **Trailing stop**: 1R below highest high, activates at +2R from entry
- **Exhaustion exit**: At +3R, sell 50% into strength, tighten trail to 0.5R on remainder
- **Self-managed stops**: WebSocket tick-by-tick monitoring, spread-based limit sells
- **Last entry time**: 11:00 ET (11:xx+ entries are net losers)
- **One trade per symbol per day**

### MACD Zone Filter

Scales position risk based on MACD histogram strength at entry (as % of price). U-shape: strong negative AND strong positive MACD are the best setups; near-zero is the worst.

| Zone | MACD % Range | Multiplier | Rationale |
|------|-------------|------------|-----------|
| Strong Negative | < -0.5% | 1.5x | Deep pullback = loaded spring |
| Normal | -0.5% to -0.2%, +0.1% to +0.5% | 1.0x | Standard risk |
| **Dead Zone** | **-0.2% to +0.1%** | **SKIP** | 30% WR, negative avg P&L |
| Strong Positive | > +0.5% | 1.5x | Momentum confirmation |

Uses previous day's bars for MACD warm-up (avoids cold-start on early-morning setups).

### Exhaustion Exit

Sells 50% of position into strength when exhaustion signals fire at +3R profit, then tightens trailing stop to 0.5R on the remainder. Reduces slippage by selling while buyers are still aggressive.

Active signals: **climax candle** (2x avg body + 2x avg volume = blow-off top) and **shooting star** (upper wick > 2x body, close in bottom 40%).

**Production implementation**: TradingEngine polls 1-min bars every 60s, runs signal detection from shared `trading/exhaustion_signals.py` module, executes partial sell via `StopMonitor.execute_partial_exit()` with fill confirmation + safety-net SL qty update.

### Spread-Based Exit Pricing

StopMonitor uses real-time NBBO quotes for exit limit pricing instead of fixed offsets:

| Spread | Pricing | Method |
|--------|---------|--------|
| < $0.05 (tight) | Midpoint of bid/ask | Saves $0.02-$0.24/share vs fixed offset |
| $0.05-$0.15 (medium) | Bid + $0.01 | Fast fill, minimal give |
| > $0.15 (wide) | Bid | Take what's available |

Falls back to fixed offset (`max($0.03, price × 0.5%)`) if quote fetch fails. 30-second fill timeout: if limit order unfilled, cancel and market-sell via `close_position()`.

### Minimum Stop Distance Filter

Rejects setups where the stop distance (entry - stop_loss) is less than $0.12. These are penny-wide stops on low-priced stocks where the breakout barely triggers the buy-stop then immediately reverses — the stop is within tick noise, not at a meaningful technical level.

### Market Regime Filter

Blocks or tightens entries when SPY indicates a hostile regime. Four independent signals:

1. **High volatility + downtrend**: SPY 5-day avg daily range > 1.5% AND close below SMA(50) → **blocks all entries**
2. **Declining SMA50 (dead-cat-bounce filter)**: SPY SMA(50) 5-day slope < -0.5 → **blocks all entries**. Even when price crosses above a falling SMA50, momentum breakouts fail because the underlying macro trend is still weakening. Threshold -0.5 balances DD reduction (-$20.9K best) with trade opportunity.
3. **Euphoria filter**: SPY up/down volume ratio > 1.2 AND RSI(14) > 60 → **blocks all entries**. When broad market has bullish volume dominance + overbought RSI, momentum breakouts get crowded — FOMO buyers dump, stops get tagged. Root cause of the 12-loss consecutive streak.
4. **Thin liquidity (H5 OR filter)**: SPY T-1 volume / SMA20(volume) < 0.70 → **tightens breakout volume requirement** from 1.5x to 2.0x.

**Lookahead prevention**: All indicators use data strictly *before* trade date. SMA50 slope uses yesterday's value (known before market open). Live system uses `date.today()` at 9:30 AM, so T-1 = yesterday's settled close.

| Config Parameter | Default | Description |
|-----------------|---------|-------------|
| `market_regime_enabled` | `true` | Enable/disable regime filter |
| `market_regime_vol_threshold` | `1.5` | SPY vol threshold (%) |
| `market_regime_sma_period` | `50` | SMA lookback period |
| `market_regime_min_spy_volume_ratio` | `0.70` | Min SPY volume ratio (T-1 vol / SMA20) — thin liquidity threshold |
| `market_regime_thin_liquidity_breakout_vol_ratio` | `2.0` | Min breakout volume ratio on thin liquidity days |
| `max_trades_per_day` | `5` | Max entries per day |

## Backtesting

### Single Symbol Backtest

```bash
python backtest.py PLYX 2026-03-13           # run backtest
python backtest.py PLYX 2026-03-13 --verbose # with debug logging
```

### Batch Backtest

Scans the full stock universe for 10%+ intraday movers, runs backtests on each qualifying (symbol, date) pair, and produces a CSV report. All API data (daily bars + 1-min intraday bars) is cached to SQLite so subsequent runs are instant.

```bash
python batch_backtest.py                                        # March 2026 (default)
python batch_backtest.py --start 2026-02-01 --end 2026-03-13   # Feb+Mar 2026
python batch_backtest.py --output my_results.csv --verbose      # custom output + debug
```

**Output**: Trade-level CSV with columns: `symbol, date, entry_time_et, entry_price, stop_loss, target, shares, exit_time_et, exit_price, exit_reason, pnl, pnl_pct`

### 15-Month Baseline — Jan 2025 to Mar 2026

Current production config with all filters active: 0.3% exit slippage, trailing stops, exhaustion exits, MACD zones, regime filter with SMA50 slope, min stop distance $0.09, last entry 11:00 ET.

| Metric | Value |
|--------|-------|
| **Trades** | 253 |
| **Win Rate** | 43.9% |
| **Total P&L** | **$198,494** |
| **Avg Win** | $5,200 |
| **Avg Loss** | -$2,650 |
| **W/L Ratio** | 1.96 |
| **Profit Factor** | 1.53 |
| **Sharpe (annualized)** | 3.69 |
| **Max Drawdown** | -$20,932 |
| **Losing Months** | 2 |

Saved as `backtest_baseline_aligned.csv`.

### Month-by-Month Breakdown

| Month | Trades | WR | P&L | Cum P&L |
|-------|--------|----|-----|---------|
| 2025-01 | 37 | 48.6% | +$52,583 | $52,583 |
| 2025-02 | 13 | 38.5% | +$6,218 | $58,801 |
| 2025-03 | — | — | $0 (regime blocked) | $58,801 |
| 2025-04 | — | — | $0 (regime blocked) | $58,801 |
| 2025-05 | 12 | 33.3% | +$2,856 | $61,657 |
| 2025-06 | 38 | 60.5% | +$84,681 | $146,337 |
| 2025-07 | 33 | 42.4% | +$34,189 | $180,526 |
| 2025-08 | 23 | 43.5% | +$2,291 | $182,817 |
| 2025-09 | 24 | 29.2% | +$83 | $182,900 |
| 2025-10 | 48 | 39.6% | +$15,452 | $198,352 |
| 2025-11 | 12 | 41.7% | +$6,786 | $205,139 |
| 2025-12 | 28 | 35.7% | +$11,389 | $216,528 |
| 2026-01 | 48 | 37.5% | +$21,294 | $237,822 |
| 2026-02 | 14 | 50.0% | +$28,111 | $265,933 |
| 2026-03 | 10 | 40.0% | +$6,894 | $272,827 |

## Configuration

Trading parameters are configured in `config.yaml` and `.env` (API keys).

## Testing

```bash
pytest tests/ -v          # full suite
pytest tests/ -q          # quick summary
```

## Future Tasks

- **Implement min_stop_distance $0.09 in production**: Add config param and filter in both backtest and production trade planner
- **Implement SMA50 slope regime filter**: Add slope check to MarketRegimeFilter — block when SMA50 5-day slope < 0
- **Run fresh baseline backtest** after implementing above two filters to confirm $289K / Sharpe 2.86 / DD -$26K
- **News-based late-day entry filter**: Evaluate using news sentiment to allow selective entries after 11:00 ET for catalyst-driven stocks
