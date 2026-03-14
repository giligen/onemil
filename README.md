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

### BacktestRunner Parameters

| Parameter     | Default | Description                                |
|---------------|---------|--------------------------------------------|
| `min_price`   | 0.0     | Minimum entry price filter                 |
| `skip_midday` | True    | Skip 11:30-14:00 ET entries (dead zone)    |

```python
# Override defaults:
runner = BacktestRunner(min_price=5.0, skip_midday=True)  # conservative
runner = BacktestRunner(min_price=0.0, skip_midday=False) # no filters
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
- **Midday filter in production**: Add `_is_midday()` check to `PositionManager.can_open_position()` — see design below.
