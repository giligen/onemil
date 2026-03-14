# OneMil - Day Trading System

Real-time stock scanner + automated trading system targeting Ross Cameron's momentum day trading strategy.

## Goals

1. Real-time stock scanner (gap ups, high relative volume, low float, $2-$20)
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

- **Entry**: Breakout level (flag high)
- **Stop**: Flag low - $0.01, capped at $0.20 from entry (Ross's rule), minimum $0.05 risk
- **Target**: 2:1 R:R
- **Position size**: floor($500 / entry), max 1000 shares
- **One trade per symbol per day**

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

### Backtest Filter Analysis (Feb+Mar 2026, 48 trades)

| Filter Config                | Trades | Win Rate | P&L     | Avg Win | Avg Loss |
|------------------------------|--------|----------|---------|---------|----------|
| No filters                   | 48     | 54.2%    | $4,420  | $330    | -$189    |
| Skip midday only (DEFAULT)   | 32     | 62.5%    | $4,241  | $333    | -$202    |
| Price >= $5 only             | 29     | 65.5%    | $3,778  | $267    | -$129    |
| Price >= $5 + skip midday    | 21     | 71.4%    | $3,353  | $278    | -$136    |

**Key findings:**
- **Midday entries (11:30-14:00 ET)**: 37.5% WR — the strategy's worst time window
- **Sub-$5 stocks**: 36.8% WR but avg win is $502 (high risk, high reward)
- **Default**: Skip midday only — retains 96% of PnL while boosting WR from 54% to 63%

### Relative Volume (rvol) Analysis — 15-Month Comparison (Jan 2025 - Mar 2026)

Three rvol approaches were compared across 764 stock-day pairs:

| Metric | no_rvol (disabled) | cumulative 5x (Ross) | bucket 5x (fixed) |
|--------|-------------------|-----------------------|-------------------|
| **Trades** | 764 | 586 | 331 |
| **Total P&L** | **$247,088** | $170,066 | $103,717 |
| **P&L/trade** | $323 | $290 | $313 |
| **Win Rate** | 39.4% | 39.2% | 39.3% |
| **Profit Factor** | **14.08** | 8.10 | 5.60 |
| **Sharpe (monthly)** | **3.72** | 2.89 | 2.01 |
| **Winning Months** | 12/15 | 13/15 | 9/15 |
| **Max Drawdown** | $18,888 | $23,956 | $15,331 |

**Key findings:**
- **rvol filtering hurts performance** — both rvol modes cut profitable trades without improving win rate (~39% across all modes)
- **no_rvol dominates** on Sharpe (3.72), total P&L ($247K), and profit factor (14.08)
- **bucket rvol is worst** — only 9/15 winning months, a timezone mismatch bug in the scanner made this effectively disabled in production anyway
- **Decision**: rvol filter disabled in production (`relative_volume_min: 0` in config.yaml)

Month-by-month breakdown:

| Month | no_rvol | cumulative | bucket |
|-------|---------|-----------|--------|
| 2025-01 | 32 trades +$17,091 | 24 trades +$9,442 | 12 trades -$526 |
| 2025-02 | 33 trades +$28,241 | 30 trades +$28,177 | 11 trades +$10,724 |
| 2025-03 | 27 trades -$5,437 | 20 trades -$7,007 | 13 trades -$5,007 |
| 2025-04 | 47 trades -$8,793 | 38 trades -$16,949 | 23 trades -$8,516 |
| 2025-05 | 48 trades -$4,657 | 35 trades +$159 | 19 trades -$1,807 |
| 2025-06 | 69 trades +$41,001 | 58 trades +$39,059 | 40 trades +$31,292 |
| 2025-07 | 62 trades +$37,293 | 55 trades +$22,148 | 41 trades +$27,614 |
| 2025-08 | 47 trades +$10,420 | 35 trades +$6,993 | 27 trades +$2,857 |
| 2025-09 | 55 trades +$17,353 | 44 trades +$3,866 | 25 trades +$4,371 |
| 2025-10 | 85 trades +$42,307 | 63 trades +$26,576 | 52 trades +$24,232 |
| 2025-11 | 42 trades +$20,838 | 32 trades +$14,376 | 12 trades -$128 |
| 2025-12 | 51 trades +$15,078 | 41 trades +$12,813 | 16 trades +$12,595 |
| 2026-01 | 73 trades +$11,197 | 63 trades +$10,681 | 22 trades -$6,570 |
| 2026-02 | 44 trades +$13,090 | 23 trades +$7,730 | 4 trades +$4,798 |
| 2026-03 | 49 trades +$12,067 | 25 trades +$12,002 | 14 trades +$7,788 |

### BacktestRunner Parameters

| Parameter     | Default      | Description                                |
|---------------|-------------|---------------------------------------------|
| `min_price`   | 0.0         | Minimum entry price filter                  |
| `skip_midday` | True        | Skip 11:30-14:00 ET entries (dead zone)     |
| `rvol_mode`   | 'cumulative'| Rvol approach: 'cumulative' or 'bucket'     |

```python
# Override defaults:
runner = BacktestRunner(min_price=5.0, skip_midday=True)  # conservative
runner = BacktestRunner(min_price=0.0, skip_midday=False) # no filters
runner = BacktestRunner(rvol_mode='bucket')                # bucket rvol mode
```

### Rvol Comparison Script

Runs the full 3-way rvol comparison across 15 months:

```bash
python compare_rvol_modes.py    # takes ~40 minutes
```

## Configuration

Trading parameters are configured in `config.yaml` and `.env` (API keys).

## Testing

```bash
pytest tests/ -v          # full suite
pytest tests/ -q          # quick summary
```

## Future Tasks

- **Target strategy optimization**: Currently using fixed 2:1 R:R target. Evaluate pole height projection as an alternative or combined target strategy (e.g., partial exit at 2:1, trail remainder toward pole projection). Backtesting showed pole height targets ($7.12) can be too greedy — missing exits that 2:1 ($6.98) would have captured profitably.
- **Partial profit exit**: Sell half position at +1R, trail remainder with breakeven stop. Requires position splitting + order replacement — deferred for separate implementation.
