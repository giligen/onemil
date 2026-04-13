# OneMil - Day Trading System

Real-time stock scanner + automated trading system targeting Ross Cameron's momentum day trading strategy.

## Goals

1. Real-time stock scanner (gap ups, high relative volume, low float, $2-$30)
2. Automated paper trading via Alpaca
3. Go live

## Architecture

```
=== Strategy 1: Bull Flag (main service) ===
main.py                         CLI entry point (scanner + trading engine)
backtest.py                     Single-symbol backtesting CLI
batch_backtest.py               Batch backtest (universe scan → CSV report)
config.yaml                     Bull flag configuration

=== Strategy 2: MACD Wave (separate service) ===
macd_wave.py                    Standalone service entry point
macd_wave_backtest.py           MACD wave backtesting CLI
macd_wave.yaml                  MACD wave configuration
trading/macd_wave_engine.py     Core engine (mover detection, MACD, entry/exit)

=== Shared Infrastructure ===
data_sources/
  alpaca_client.py              Alpaca API client (market data + trading)
  float_provider.py             Float share data via Yahoo Finance
  news_provider.py              News & sentiment analysis

scanner/
  realtime_scanner.py           Real-time stock screening (bull flag)
  criteria.py                   Filtering logic (price, float, volume, etc.)

trading/
  pattern_detector.py           Bull flag pattern detection (1-min bars)
  trade_planner.py              Trade plan creation (entry/stop/target/sizing)
  trading_engine.py             Bull flag trading orchestration
  order_executor.py             Order submission/management
  position_manager.py           Position tracking
  stop_monitor.py               WebSocket stop monitoring + spread-based exits
  market_regime.py              SPY volatility + trend regime filter
  exhaustion_signals.py         Exhaustion exit signal detection (shared backtest/prod)

persistence/
  database.py                   SQLite database layer (shared, strategy column)

batch/
  universe_builder.py           Stock universe construction (bull flag)

monitoring/
  logger.py                     Logging setup (UTF-8 for Windows)

notifications/
  telegram_notifier.py          Telegram alerts (shared, [Bull Flag] / [MACD Wave] prefix)
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
- **Position size**: fixed_risk mode, $2,000 risk per trade, max 15,000 shares
- **Trailing stop**: 1R below highest high, activates at +1.5R from entry
- **Exhaustion exit**: At +3R, sell 50% into strength, tighten trail to 0.5R on remainder
- **Self-managed stops**: WebSocket tick-by-tick monitoring, spread-based limit sells
- **Last entry time**: 10:30 ET (10:30-11:00 bucket is breakeven PF 1.01)

#### Trail activation tradeoff (validated Q1 2026)

Current config is **activate_at_r=1.5, trail_r=1.0**. Prior default was 2.0/1.0. Q1 2026 comparison (apples-to-apples, same code, only trail differs):

| Metric | a=2.0 t=1.0 (prior) | a=1.5 t=1.0 (current) |
|--------|--------------------|-----------------------|
| Q1 2026 P&L | -$9,919 | -$586 (**+$9,333**) |
| Win rate | 30.9% | 33.3% |
| Feb 2026 loss | -$35,484 | -$18,843 (**+$16,641**) |
| Max DD | $46,846 | $34,759 (**-$12,087**) |
| Profit factor | 0.92 | 0.99 |

Earlier activation (1.5R) locks in profits on marginal winners that peak between +1.5R and +2.0R then reverse — OPTX on 2026-04-13 peaked at exactly +1.97R ($0.01 below 2.0R trigger) then stopped out. The 1.5R trigger captures that class of trade.

**Tradeoff**: sacrifices some extension on multi-R winners that would run past 2.0R under the old config. Net on Q1 2026 is strongly positive; full 16-month validation pending proper rebuild.
- **One trade per symbol per day**

### P&L Layers (execution order)

The $446K 15-month result depends on 7 layers applied in order. BT is the golden source; PROD must match exactly.

#### Layer 1: MACD Dead Zone — block flat-momentum trades

Scales position risk based on MACD histogram strength at entry (as % of price). U-shape: strong negative AND strong positive MACD are the best setups; near-zero is the worst.

| Zone | MACD % Range | Multiplier | Rationale |
|------|-------------|------------|-----------|
| Strong Negative | < -0.5% | 1.5x | Deep pullback = loaded spring |
| Normal | -0.5% to -0.2%, +0.1% to +0.5% | 1.0x | Standard risk |
| **Dead Zone** | **-0.2% to +0.1%** | **SKIP** | 30% WR, negative avg P&L |
| Strong Positive | > +0.5% | 1.5x | Momentum confirmation |

Uses previous day's bars for MACD warm-up (avoids cold-start on early-morning setups). Config: `macd_zones.enabled: true`.

#### Layer 2: News Kill Rules — block no-news trades in loser segments (+$158K)

If stock has a **real catalyst** (FDA, earnings, contract, M&A, analyst, product, SEC filing) the trade always passes. If NO catalyst, these segments are killed:

| Rule | Condition | WR | Impact |
|------|-----------|-----|--------|
| 1 | avg_daily_vol >= 3M + no news | 22% | -$101K eliminated |
| 2 | price < $3 + no news | 20% | -$11K eliminated |
| 3 | float >= 30M + no news | 19% | -$14K eliminated |
| 4 | $5-12 + pole 8-15% + no news | 27% | -$32K eliminated |

Config: `news_kill_rules.enabled: true`. News fetched from Alpaca News API, classified via regex (realtime) and `news_history` table (backtest).

#### Layer 3: Conviction Scoring — scale position by setup quality

5 pattern-only rules (no news in scoring) that scale position size 0.25x to 3.0x:

| Rule | Condition | Score |
|------|-----------|-------|
| 1 | Pole gain 4.5-9% (sweet spot) | +0.3 |
| 2 | Flag tightness < 30% / > 50% | +0.3 / -0.3 |
| 3 | Pole/flag vol ratio > 1.7x | +0.3 |
| 4 | SPY 3d range > 1.2% / < 0.8% | +0.3 / -0.5 |
| 5 | Retracement < 30% | +0.2 |

Combined with risk tier multiplier, capped at 3.0x. Applied via `create_plan(risk_multiplier=combined_mult)`. Config: `conviction_scoring.enabled: true`.

#### Layer 4: Post-Fill Exit — SPY calm + weak breakout volume

After buy-stop fills, if SPY 3-day range < 0.8% (calm market) AND breakout bar volume < 1.0x flag average → **immediate close**. Catches weak breakouts that rely on broad-market momentum that isn't there.

#### Layer 5: Risk Tiers — 2x on $10-15 stocks

Price-based risk scaling validated on 15 months:

| Tier | Price Range | Volume Range | Multiplier |
|------|------------|--------------|------------|
| 1 | $10-15 | 500K-5M | **2.0x** (PF 2.16, $3.2K/trade) |
| 2 | $15-23 | 500K-5M | 1.0x (PF 0.65 at 3x, kept at 1x) |

Config: `risk_tiers.enabled: true`.

#### Layer 6: Gap-Over Rejection — >2% fill above breakout

If the fill price gaps more than 2% above the breakout level, the trade is immediately closed. These are chasing entries with 23% WR — net losers. Hardcoded 2% threshold.

#### Layer 7: Standard Filters

| Filter | Value | Source |
|--------|-------|--------|
| Min price | $2.00 | config |
| Min stop distance | $0.09 | config |
| Last entry time | 10:30 ET | config |
| Max shares | 15,000 | config |
| Daily loss limit | -$5,000 | PositionManager |
| Max concurrent | 3 | PositionManager |
| Max trades/day | 5 | config |

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

## Strategy 2: MACD Wave

Separate service targeting medium-cap stocks ($15-30) with strong intraday momentum. Enters on MACD histogram confirmation after +10% move, exits on histogram flip.

### Backtest Results (Jan 2025 — Mar 2026, 15 months)

| Metric | Value |
|--------|-------|
| **Trades** | 61 |
| **Win Rate** | 44.3% |
| **Total P&L** | **+$122,935** |
| **Avg Win** | +$5,533 (+13.8%) |
| **Avg Loss** | -$778 (-2.0%) |
| **Profit Factor** | 5.65 |
| **Sharpe** | 5.49 |
| **Max Drawdown** | -$6,978 |
| **Position Size** | $50,000 |

### Entry Rules

1. **Universe**: All US equities $15-30, volume > 1M/day (built pre-market)
2. **Trigger**: Stock crosses +10% from open within 3 minutes
3. **Volume filter**: Cumulative volume at cross < 300K (avoids crowded trades)
4. **MACD confirmation**: 3 consecutive positive histogram bars, histogram ≥ 0.5% of price
5. **Entry**: Limit buy at ask + 0.1%

### Exit Rules

- **MACD flip**: Histogram turns negative → limit sell at bid
- **Hard stop**: 2% below entry → market sell
- **Force close**: 15:45 ET → market sell
- **One trade per symbol per day**

### Running the MACD Wave Service

```bash
python macd_wave.py                  # Live paper trading
python macd_wave.py --dry-run        # Monitor only, no orders
python macd_wave.py --verbose        # Debug logging
python macd_wave.py --skip-wait      # Skip pre-market wait (testing)
```

### MACD Wave Backtest

```bash
python macd_wave_backtest.py                                    # March 2026 (default)
python macd_wave_backtest.py --start 2025-01-01 --end 2026-03-27  # Full 15 months
python macd_wave_backtest.py --cross-time 3 --macd-min 0.5 --max-price 30 --max-vol 300000
python macd_wave_backtest.py --no-slippage                      # Compare without slippage
python macd_wave_backtest.py --w1-scout --w1-min 5 --max-waves 3  # W1 scout mode
```

## Running Both Services

Both strategies run as **systemd services** — auto-start on boot, auto-restart on failure. They share the same Alpaca paper account and SQLite database but do not interfere with each other.

### Service Management

```bash
# Status
sudo systemctl status onemil-trader          # Bull flag
sudo systemctl status onemil-macd-wave       # MACD wave

# Start / Stop / Restart
sudo systemctl start onemil-trader
sudo systemctl start onemil-macd-wave
sudo systemctl restart onemil-trader
sudo systemctl restart onemil-macd-wave
sudo systemctl stop onemil-trader
sudo systemctl stop onemil-macd-wave

# View logs
journalctl -u onemil-trader -f              # Bull flag (live)
journalctl -u onemil-macd-wave -f           # MACD wave (live)
tail -f logs/onemil.log                      # Bull flag file log
tail -f logs/macd_wave.log                   # MACD wave file log

# Check both running
ps aux | grep 'main.py\|macd_wave' | grep -v grep
```

### Systemd Service Files

```
/etc/systemd/system/onemil-trader.service       # Bull flag
/etc/systemd/system/onemil-macd-wave.service    # MACD wave
```

Both auto-restart on failure (30s delay), use `.env` for API keys, 2GB memory limit.

### Pre-Market Universe Builder (Bull Flag)

The bull flag scanner uses a pre-built stock universe. Run the batch builder nightly:

```bash
python main.py --batch    # Fetches assets, filters by price/float, caches volume profiles
```

The MACD wave service builds its own universe from Alpaca snapshots at 8:30 AM ET each day — no pre-build needed.

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

Current production config with all 7 layers active: MACD dead zone, news kill rules, conviction scoring, post-fill exit, risk tiers, gap-over rejection, standard filters. 0.5% entry slippage, 0.3% exit slippage, trailing stops, exhaustion exits.

| Metric | Value |
|--------|-------|
| **Total P&L** | **$446K** |
| **Losing Months** | 3 of 15 |
| **Key Layers** | MACD dead zone, news kill (+$158K), conviction scoring, risk tiers (2x on $10-15) |

## Configuration

Trading parameters are configured in `config.yaml` and `.env` (API keys).

## Testing

```bash
pytest tests/ -v          # full suite
pytest tests/ -q          # quick summary
```

## Future Tasks

- **BuyMonitor Phase 2**: Replace buy-stop orders with SIP WebSocket limit buys for tighter entry slippage (data collection active via Phase 1 quote monitoring)
- **MACD Wave W1 scout mode**: Paper-trade W1, only enter W2-3 if W1 >= 5% (tested: 62% WR on W2-3 but low trade count)
- **L2 data evaluation**: Assess Level 2 order book data for entry timing optimization
- **Combined P&L dashboard**: Unified daily report across both strategies
- **News-based late-day entry filter**: Evaluate using news sentiment to allow selective entries after 11:00 ET for catalyst-driven stocks
