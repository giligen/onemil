# CRITICAL - ALWAYS READ FIRST
Since you have a memory of a chicken, you MUST stop every 10min and re-read CLAUDE.md AND DO THIS before every 10th prompt -- this is a must!!!

# Project: OneMil - Day Trading System

Real-time stock scanner + automated trading system targeting Ross Cameron's momentum day trading strategy.

## Goals
1. Real-time stock scanner (gap ups, high relative volume, low float, $2-$20)
2. Automated paper trading via Alpaca
3. Go live

# CRITICAL: Running Long Commands
* **NEVER pipe long-running commands through `| tail`, `| head`, `| grep`** — this buffers ALL output and you see NOTHING until the process finishes. Run commands directly and let output stream.
* **BAD**: `python batch_backtest.py --build-cache 2>&1 | tail -10` (buffered, blind for 30+ min)
* **GOOD**: `python batch_backtest.py --build-cache` (output streams in real-time)
* For background tasks, use `run_in_background=true` without piping
* **NEVER overwrite or delete cache files (cache.db, CSV caches) without explicit user permission**

# Code Quality
* When writing code you should behave as if you are Linus Torvalds -- partitioning, modular code, reusable code, extract common pieces to accessors, use meaningful names like Linus would
* TDD approach -- never assume that what you wrote will work. use a tdd approach to test it
* Code coverage MUST trend at ~90%, must! don't tell me you are done before code-coverage is complete
* Always validate everything you wrote via running the specific unit-test that is located in the appropriate tests directory
* Always validate everything you wrote via running a system test to ensure you didn't break anything and that the implemented functionality works
* Use Verbose/Debug flags for extra logging for the sake of debugging. Don't guess issues, find in the logging the root-cause
* Always solve the root cause, never apply work-arounds
* Instead of writing 10s of bespoke scripts, strive to use the main code models with specific flags
* Always push to github
* Always keep dependency installation file up-to-date

# Integration Testing for Multi-Component Flows

**MANDATORY**: When implementing features that involve data flowing through multiple components, you MUST write integration tests that validate the FULL end-to-end flow.

## When Integration Tests Are Required

Integration tests are REQUIRED for any flow involving:
- Database operations (save -> retrieve -> use)
- Data transformations across boundaries (JSON <-> dict, serialization/deserialization)
- Multi-step processes (input -> processing -> output)
- External API calls -> internal processing -> storage
- Configuration/state that affects multiple components

## What Integration Tests Must Cover

Integration tests MUST validate:
1. **Data integrity through the entire pipeline** - verify data format at EACH boundary
2. **Type transformations** - if data is serialized/deserialized, test both directions
3. **Edge cases** - empty data, missing fields, null values
4. **Actual component interactions** - use REAL instances, not mocks
5. **The complete flow** - from entry point to final destination

## Coverage Rule

- Unit tests: Test components in isolation (mocked dependencies)
- Integration tests: Test components working together (real dependencies)
- System tests: Test entire system end-to-end (real environment)

**You are NOT done** until you have all three levels for complex flows.

# CRITICAL: Testing & Deployment Protocol

**NEVER commit changes to production without testing first!**

## Required Testing Steps for External API Changes:
1. **Research API documentation** - Don't assume capabilities, verify them
2. **Create separate test file** - Build/test in isolation
3. **Test with real API** - Unit tests with mocks are NOT sufficient for API integration
4. **Verify success** - Check logs for actual success, not just "no errors"
5. **System test** - Run full cycle
6. **Monitor for errors** - Grep output for ERROR/exception before committing
7. **Only then commit** - If all tests pass

## Post-Incident Protocol:
1. **Immediately revert** broken code to restore production
2. **Document** what broke and why
3. **Commit revert** with clear explanation
4. **Don't rush the fix** - Take time to do it properly with testing

**Remember: Breaking production wastes more time than proper testing takes!**

# Code Standards

* Add docstrings to all functions
* Keep MD files up-to-date per model
* Keep Readme.md file up-to-date
* Unicode and emojis are supported (logging handlers must use UTF-8 encoding on Windows)
* Code should be verbose enough to show progress throughout long processes
* Use git source control and push to master every time I confirm things are working well
* Use descriptive meaningful names always
* No tests should ever be failing, always fix the core issue and don't work around it
* All Errors must be reported, e.g., missing API Keys and execution should break
* Always document latest architecture in readme.md and keep it up-to-date
* **CRITICAL: All fallback code paths MUST log ERROR or WARNING** - Silent failures hide bugs. Every fallback (try/except, if/else with defaults, .get() with fallback values) MUST explain WHY it triggered via logger.error() or logger.warning()
* **NEVER leave broken unit tests** - Even if the test was broken by someone else's code, fix it. Zero failing tests is mandatory. Every session should end with all tests passing.

# System-in-dev

* Assume DB might be locked by other process that are running in parallel to you
* When building batch processors always make them verbose to show progress
* Unit tests -> Mock external APIs
* Integration tests -> Use real APIs (or testnet/paper)
* Production code -> Never includes mock logic, always real implementations

# CRITICAL: MagicMock() MUST Use spec= Parameter

**The Problem**: `MagicMock()` without `spec=` HIDES bugs by returning `MagicMock` for ANY attribute access, even non-existent ones.

## MANDATORY RULES FOR MagicMock():

### 1. ALWAYS Use spec= for Domain Classes
```python
# BAD - Hides AttributeErrors
executor = MagicMock()

# GOOD - Catches interface violations
executor = MagicMock(spec=AlpacaExecutor)
```

### 2. Use AsyncMock for Async Classes
```python
# BAD
notifier = MagicMock()

# GOOD
notifier = AsyncMock(spec=TelegramNotifier)
```

### 3. External SDK Objects Are OK Without spec=
```python
# OK - External SDK objects, not our domain
mock_order = MagicMock()  # Alpaca Order object, OK without spec
```

### 4. Use conftest.py Fixtures
Pre-configure fixtures with spec= in `tests/conftest.py`.

# Two Trading Services

## Service 1: Bull Flag (`onemil-trader`)
```bash
sudo systemctl status onemil-trader      # Check status
sudo systemctl restart onemil-trader     # Restart
sudo systemctl stop onemil-trader        # Stop
journalctl -u onemil-trader -f           # Live logs
```
- Systemd service: `/etc/systemd/system/onemil-trader.service`
- Runs: `python main.py --scan --trade --verbose`
- Auto-restarts on failure (30s delay)
- Config: `config.yaml`
- Logs: `logs/onemil.log`
- Universe: pre-built via `python main.py --rebuild-universe` (nightly cron)

## Service 2: MACD Wave (`onemil-macd-wave`)
```bash
sudo systemctl status onemil-macd-wave   # Check status
sudo systemctl restart onemil-macd-wave  # Restart
sudo systemctl stop onemil-macd-wave     # Stop
journalctl -u onemil-macd-wave -f        # Live logs
```
- Systemd service: `/etc/systemd/system/onemil-macd-wave.service`
- Runs: `python macd_wave.py`
- Auto-restarts on failure (30s delay)
- Config: `macd_wave.yaml` (validated filters: $15-30, cross<3m, MACD≥0.5%, vol<300K, 2% stop)
- Logs: `logs/macd_wave.log`
- Telegram: messages prefixed with `[MACD Wave]`
- DB: trades table has `strategy` column ('bull_flag' or 'macd_wave')
- Universe: self-built at 8:30 AM ET each day from Alpaca snapshots (no pre-build needed)

# Running Backtests

## Bull Flag Backtests

### Single symbol
```bash
python backtest.py PLYX 2026-03-13 --verbose
```

### Batch backtest (all movers in a date range)
```bash
python batch_backtest.py --start 2026-02-01 --end 2026-03-13 --verbose
```
- Data is cached in SQLite (daily bars + 1-min bars) — first run fetches from Alpaca API, subsequent runs are instant
- Output: CSV file with trade-level results
- Default filters: skip midday (11:30-14:00 ET) entries

### Backtest defaults
- `BacktestRunner(min_price=0.0, skip_midday=True)` — skip midday is the only default filter
- To override: pass `min_price=5.0` or `skip_midday=False` to `BacktestRunner`

## MACD Wave Backtests

```bash
# Default: March 2026
python macd_wave_backtest.py

# Full 15-month validation
python macd_wave_backtest.py --start 2025-01-01 --end 2026-03-27

# With winning filters (these are the defaults in macd_wave.yaml)
python macd_wave_backtest.py --cross-time 3 --macd-min 0.5 --max-price 30 --max-vol 300000

# Without slippage for comparison
python macd_wave_backtest.py --no-slippage

# W1 scout mode (paper W1, trade W2+)
python macd_wave_backtest.py --w1-scout --w1-min 5 --max-waves 3
```
- Daily bars cached in `daily_bars` table (first run ~5min for full universe, subsequent instant)
- 1-min bars cached in `intraday_bars_1min` table
- All filter params configurable via CLI or `macd_wave.yaml`

# Backtest Learnings & Anti-Patterns

## Python output buffering
When running backtests via `python3 -c "..."`, print() output is BUFFERED until script exits. Use `sys.stdout.flush()` after each print, or run as a script file instead of inline. This has wasted time repeatedly.

## Gap threshold doesn't matter for bull flags
Tested 3%, 5%, 8%, 10% intraday range thresholds — all produce identical results (254-257 trades, same P&L). The bull flag pattern detector (min_pole_gain_pct=3%) is the real filter, not the daily bar screen.

## Key backtest numbers (Jan-Mar 2026)
- **Bull flag**: 254 trades, 33.5% WR, PF 1.13, +$28K, DD -$30K
- **MACD wave**: 61 trades, 44% WR, PF 5.65, +$123K, DD -$7K (15 months)
- **Combined**: +$330K over 15 months, 14/15 green months

## Overfitting warning
MACD wave filters (cross<3m, MACD≥0.5%, vol<300K, $15-30) were optimized on the same 15-month dataset used for validation. NO out-of-sample test performed. Expect 40-60% of backtest P&L in production.

## Slippage reality vs model
- Backtest: 0.1% entry, 0.1-0.3% exit
- Real production (13 trades): 0.54% avg entry slippage
- Entry slippage on thin stocks is 5x worse than modeled

## MACD wave P&L is outlier-dependent
Top 3 trades (FLYE, BTTC, KELYB) = 62% of total MACD wave P&L. Miss one big winner in a quarter and results look very different.

# Interactive Sessions
* I'm here for you to answer questions and clarify ambiguous points/logic
* **Bug Prevention Protocol**:
  - Whenever there's a bug, write BOTH unit tests AND integration tests
  - Unit test: Isolate the specific component that failed
  - Integration test: Validate the full data flow that exposed the bug
  - This ensures bugs can NEVER happen again at any level
