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

# Three Trading Strategies (one systemd service: `onemil-trader`)

Bull flag, MACD wave, and ORB all run as modules inside `main.py` under the
`onemil-trader` service. Each is toggled via CLI flags: `--flag`, `--macd`, `--orb`.

## Strategy 1: Bull Flag (`onemil-trader`)
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

### Feature flag: two-tier filter (added 2026-04-17, shipping ON)
- Config key: `trading.bull_flag.two_tier_filter.enabled`
- Default: `true` (shipping ON). Set to `false` to revert to A_f6 behavior (byte-identical: $54,572.15 / 83 trades, verified).
- When `true`: BT projects **+$10,455 on 2025 OOS and +$5,439 on Q1 2026 vs A_f6**
- Shared module: `trading/two_tier_filter.py` (imported by both BT Stage-2 and live engine → parity by construction)
- Enable: flip flag in `config.yaml`, run `python -c "from config import Config; print(Config().two_tier_filter_cfg)"` to verify, then `sudo systemctl restart onemil-trader`
- Monitor: `journalctl -u onemil-trader | grep "TWO-TIER FILTER"` — shows rejected Extras with reason (`extras_macd_surgical_drop` or `extras_composite_below_threshold`)
- Rollback: flip flag to `false` + restart (zero state to unwind — pure gate flip)
- Full details in README.md "Two-Tier Filter" section
- **Dormant companion change**: `BT_ALLOW_REENTRY=1` env var enables multi-trade-per-symbol-per-day in the backtest. Empirically −$1,299/yr — DO NOT enable in prod.

### Feature flag: V-reversal conviction bonus (Experiment V, added 2026-04-17, shipping ON)
- Config key: `trading.conviction_scoring.v_reversal_bonus.enabled`
- Default: `true` (shipping ON). Set to `false` to revert to V2_clean 7-rule baseline.
- When `true`: Rule 9 adds `bonus` (default 0.4) to raw conviction for gap-down V-reversal setups (gap<0 + intraday_range≥20% + pole_gain≥5%). Final score still clamped to [0.25, 3.0] — max sizing unchanged.
- BT lift: 2025 +$4,396 (+6.4%), Q1 2026 +$2,284 (+19%). Stacks on TTF+D.
- Shared between BT (`backtest.py`) and live (`trading/trading_engine.py`) conviction functions — parity by construction.
- Live: no cache rebuild, just restart trader. Enable via `Config().v_reversal_bonus_cfg`.
- Monitor: `journalctl -u onemil-trader | grep "v_reversal"` in log breakdown when conv trade fires.

### Feature flag: marginal-conviction defensive scaling (Experiment H, added 2026-04-17, research artifact)
- Config key: `trading.conviction_scoring.marginal_scaling.enabled`
- Default: `false` — mixed BT signal. 2025 V+H: −$4,892 (hurts); Q1 2026 V+H: +$851 (helps). Net **−$4,041 across both periods**, so NOT shipping on.
- Keep in codebase for future regime-aware activation (bucket is net-loser in some periods, net-winner in others).
- When `true`: trades with conv in `[min_threshold, upper_bound)` have SIZING scaled by `scale_factor` (default 0.5). Stored conviction_mult unchanged (Stage-2 filters see raw).
- Live: `journalctl -u onemil-trader | grep "MARGINAL CONV SCALE"`
- Rollback: flip flag to `false`, restart.

### Feature flag: volume-confirmed trail exit (Experiment D, added 2026-04-17, shipping ON)
- Config key: `trading.trailing_stop.vol_confirmed_exit.enabled`
- Default: `true` (shipping ON). Set to `false` to revert to naive trail behavior.
- When `true` (stacked on top of TTF-on): BT projects **additional +$3,764 on 2025 and +$1,836 on Q1 2026** (Pareto improvement — same trades, same DD, bigger avg win)
- Shared module: `trading/trail_vol_guard.py` (single helper used by BT simulator + live StopMonitor both tick and poll paths)
- Logic: trail-stop triggering bar must have volume >= `min_vol_ratio × flag_avg_volume` to fire. Low-vol drift-downs are skipped. Hard stop (pre-trailing) always fires.
- Enable: flip flag in `config.yaml`, verify via `python -c "from config import Config; print(Config().vol_confirmed_trail_cfg)"`, then `sudo systemctl restart onemil-trader`
- Monitor: `journalctl -u onemil-trader | grep "VOL-CONF SKIP"` — shows each skipped trail exit with bar volume vs threshold
- Rollback: flip flag to `false` + restart (pure config flip)
- Full details in README.md "Volume-Confirmed Trail Exit" section

### Per-tier MACD scaling + V-rev bump (S2-max, shipped 2026-04-18, **default-on, no flag**)
- Config: `trading.macd_zones.extras_tier.{strong_pos,strong_neg,normal}_multiplier` + bumped `strong_pos/neg_multiplier: 1.5→1.8` (A-tier) + `v_reversal_bonus.bonus: 0.4→1.0`
- Ships with hardcoded values (no feature flag — per-tier analysis is the production baseline going forward)
- BT: **+28.7% lift on 2025+Q1 2026** (baseline $81,911 → $105,420). Per-quarter breakdown: all 5 quarters positive. HOQ1 holdout +$4,737 (+36.0%).
- Mechanism: A-tier (≥20% intraday) stays close to current behavior; Extras-tier (10-20%) amps strong MACD 2.0x and SKIPS MACD-neutral trades (the −$14,734 landmine bucket).
- Shared classifier: `trading/two_tier_filter.py::classify_tier` (same as TTF). Both BT `backtest.py:_get_macd_zone_multiplier` and PROD `trading/trading_engine.py:_get_macd_zone_multiplier` take `intraday_change_pct` kwarg.
- Parity: `tests/test_bt_prod_parity.py` (11 tests), `tests/test_per_tier_macd_zones.py` (19 tests). 1217 total tests pass.
- Monitor: `journalctl -u onemil-trader | grep "tier="` — shows per-trade tier classification + applied multiplier.
- Rollback: `git revert` the ship commit (single commit flips all 6 yaml values + 2 function signatures back). Or manual YAML revert of `strong_pos/neg_multiplier` to 1.5, `v_reversal_bonus.bonus` to 0.4, delete `extras_tier` block.
- Full details in README.md "Per-tier MACD zone scaling (S2-max)" section.

### Feature flag: regime-aware sizing (Phase 1.4b, shipped 2026-04-18, **default-on**)
- Config key: `trading.regime_sizing.enabled`
- Default: `true` (shipping ON). Set to `false` to revert to pre-regime S2-max behavior (byte-identical via `_get_regime_for_date` short-circuit when disabled).
- Classifies each trading day as A/B/C1/C2 from SPY T-1 features (vol_20_ann, above_sma_50, sma_50_slope_10d). Applies per-regime mult on top of conviction × macd_zone.
  - **A** (Clean Bull: above SMA, vol<22%) → 1.25×
  - **B** (Volatile: vol≥22%) → 1.00×
  - **C1** (True Defensive: below SMA, slope≤+0.15%) → 1.50×
  - **C2** (Shallow-dip-in-uptrend: below SMA, slope>+0.15%) → **0.00× (skip)**
- BT: **+$28,470 on Jan 2025 → Apr 17 2026 (+34% lift)**. Feb 2026 drawdown flips −$1,159 → +$1,570. Full monthly breakdown: `research/scripts/monthly_regime_report.py`.
- All 3 CV splits positive (TRAIN +$6K, VAL +$6.1K, HOQ1 +$15.2K). MDD unchanged ($18.5K — Apr 2025 DD was all B-regime, mult=1.0).
- Shared module: `trading/regime_helpers.py` — imported by both `backtest.py` (`_get_regime_for_date` + sizing stack) and `trading/trading_engine.py` (`_get_today_regime` + sizing stack). Parity by construction; enforced by `tests/test_regime_sizing_parity.py` (23 tests).
- PROD classifier runs once per ET date at first trade attempt — fetches ~100 calendar days of SPY daily bars via `alpaca.get_daily_bars_range(['SPY'], today-100, today-1)`, classifies last row. Cached per-day; error path caches `'unknown'` (mult 1.0, no trade effect).
- Monitor: `journalctl -u onemil-trader -f | grep REGIME` — one line per day ("REGIME today=YYYY-MM-DD classified as X") + one per trade that scales ("SYM: REGIME C1 mult=1.50 → shares A→B") or skips ("SYM: REGIME C2 skip — no trade").
- Rollback: flip `trading.regime_sizing.enabled: false` + `sudo systemctl restart onemil-trader` (pure config flip, zero state to unwind).
- Known cost: Jan 2025 lost $2,534 because 21/24 trades were C2-skipped on profitable days — C1/C2 threshold is a global optimum; accepts individual-month variance.

## Strategy 2: MACD Wave (in `onemil-trader`)
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
- DB: trades table has `strategy` column ('bull_flag', 'macd_wave', or 'orb')
- Universe: self-built at 8:30 AM ET each day from Alpaca snapshots (no pre-build needed)

## Strategy 3: ORB — Opening Range Breakout (added 2026-04-19, OFF by default)

```bash
sudo systemctl restart onemil-trader           # Must restart after config change
journalctl -u onemil-trader | grep "\[ORB\]"   # Monitor ORB-specific logs
```

Runs as a module inside `onemil-trader`. Fires at 9:35 ET on gap-up stocks that
break above their first 5-min opening range high. Validated full-timeline on
Jan'25-Apr'26 with the shipped static_lock_1R exit (`study_orb_pipeline_static_lock.py`):
**$+342,565 P&L, $-18,126 max DD (trough 2025-11-13), Calmar 18.90x, 1,001 trades,
daily WR 56.6%, only 1 red month (Aug 2025 at $-9,288)**.

⚠️  IMPORTANT: earlier docs cited `$+239,853 / Calmar 15.68x` — those came from
scripts that read `orb_features_*.csv::pnl` directly. That CSV was generated with
fixed +2R target / -1R stop exits, NOT the shipped `static_lock_1R`. The numbers
above are production-parity. Use the `*_static_lock*.py` scripts for any new
ORB analysis; the older ones have warning headers pointing to the shipped variants.

**Enable**:
1. Set `ALPACA_ORB_API_KEY` + `ALPACA_ORB_API_SECRET` in `.env` (separate paper account in Phase 1)
2. Flip `strategy.enabled: true` in `orb.yaml`
3. Add `--orb` CLI flag to the service command (systemd unit file)
4. Restart service

**Architecture**:
- Separate paper `AlpacaClient` for order execution (Phase 1)
- SHARED `StopMonitor` routes exit orders to ORB via `alpaca_clients_by_strategy={'orb': orb_paper_client}` — uses main-account WebSocket for market data (free, account-agnostic) but ORB-account client for order submission
- Separate `OrderStreamWatcher` for ORB's order events
- Shared DB with `strategy='orb'` tag
- `[ORB]` Telegram prefix

**Entry mechanics**:
- Pre-placed **stop-limit buy** at `range_high × (1 + 30bps)` at 9:35 ET, auto-cancel after 60min
- 7-feature composite z-score filter (threshold ≥ 0.0, TRAIN-fit params in `orb.yaml`)
- Q4-preferred ranking, then composite DESC
- Family + super-group dedup (14 families, 91 symbols, `lev_short`/`lev_long` super-groups)
- Max 4 concurrent positions, per-pos cap $25K ($100K budget / 4)
- Risk-parity sizing: $3K risk/trade, applied adaptive quintile mult (Q5 capped at 1.5x — anti-overfit)
- Spread gate: skip entries with spread > 150bps + Telegram warning

**Exit mechanics**:
- Initial stop: `range_low`
- **Static lock**: after price touches +1.5R, stop moves to +1R forever (no trailing). StopMonitor has new `lock_arm_at_r` + `lock_stop_r` fields on WatchEntry.
- No fixed target: hold until stop/lock hit OR 15:45 ET force close

**Feature flag + rollback**:
- Master kill switch: `strategy.enabled` in `orb.yaml` (default `false`)
- Runtime disable: flip to `false` + restart. Existing positions force-close on shutdown. Bull flag + MACD wave unaffected (separate accounts/tags).

**Monitoring**:
- `journalctl -u onemil-trader | grep '\[ORB\]'` for entries/exits/skips
- `journalctl -u onemil-trader | grep 'LOCK ARMED'` for lock-state transitions
- DB queries: `db.get_open_trades(today, strategy='orb')`

**Rollout phases (as of 2026-04-19)**:
- Phase 0: code merged, `enabled: false` — no trades
- Phase 1: paper account, `enabled: true`, risk=$3K — 1 week monitor
- Phase 2: live small, risk=$1K — 1 week monitor
- Phase 3: live full, risk=$3K

**Do NOT**:
- Enable with `ALPACA_ORB_API_KEY` empty — main.py will warn + disable
- Refit z-score / quintile / adaptive params without running `study_orb_refit.py` first (quarterly cadence)
- Remove Q5 cap from `orb.yaml::adaptive_mults.Q5: 1.5` — it's the anti-overfit guard

# Running Backtests

## Bull Flag Backtests

### Single symbol
```bash
python backtest.py PLYX 2026-03-13 --verbose
```

### Batch backtest — TWO-STAGE WORKFLOW (CRITICAL)

**The backtest is a two-stage process. You MUST run both stages and ONLY report Stage 2 numbers.**

#### Stage 1: Build cache (broad, 10% threshold)
```bash
# --build-cache auto-enables --monthly chunking
python batch_backtest.py --start 2026-01-01 --end 2026-03-31 --build-cache
```
- Finds ALL movers with 10%+ intraday range
- Stores raw unfiltered trades in `data/bull_flag_cache_e50_x30.csv`
- These numbers are RAW/UNFILTERED — **NEVER report these as backtest results**

#### Stage 2: Run filtered backtest (production-matched, DEFAULT)
```bash
# Default behavior — reads from cache, applies ALL production filters from config.yaml
python batch_backtest.py --start 2026-01-01 --end 2026-03-31
```
- Reads from cache, applies: 20% threshold, 200K volume, leveraged ETF filter, max 3 concurrent, $5K daily loss limit, risk tiers
- These numbers match production behavior — **THIS is the real backtest result**
- Takes <1 second (reads from cache), so there is ZERO reason to skip this step

**NEVER report Stage 1 numbers as results. ALWAYS run Stage 2 after Stage 1.**
**If your numbers don't match what the user expects, question YOUR methodology first, not the user's memory.**

### Single symbol
```bash
python backtest.py PLYX 2026-03-13 --verbose
```

### Backtest defaults
- `BacktestRunner(min_price=0.0, skip_midday=True)` — skip midday is the only default filter
- To override: pass `min_price=5.0` or `skip_midday=False` to `BacktestRunner`
- Data is cached in SQLite (daily bars + 1-min bars) — first run fetches from Alpaca API, subsequent runs are instant

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
Changing the intraday range threshold on the cache-build step (3%, 5%, 8%, 10%) produces essentially identical results. The bull flag pattern detector (min_pole_gain_pct) is the real filter, not the daily bar screen — don't waste time re-tuning the cache threshold.

## Overfitting warning
MACD wave filters were originally tuned on the same 15-month dataset used for validation, with no out-of-sample split. Expect real-world P&L to be a significant haircut off the backtest. When proposing filter changes, push for walk-forward validation (train on one split, test on another).

## Slippage reality vs model
Bull flag entry slippage on thin stocks runs multiples of the backtest model. When recalibrating, source numbers from the `trades` DB (entry/exit quote telemetry) rather than re-quoting memorized figures — they drift as trades accumulate. Authoritative numbers live in README.md.

## MACD wave P&L is outlier-dependent
A small number of top trades can drive the majority of total MACD wave P&L. Miss one big winner in a quarter and results look very different — be careful quoting blended P&L without checking the contribution distribution.

# Interactive Sessions
* I'm here for you to answer questions and clarify ambiguous points/logic
* **Bug Prevention Protocol**:
  - Whenever there's a bug, write BOTH unit tests AND integration tests
  - Unit test: Isolate the specific component that failed
  - Integration test: Validate the full data flow that exposed the bug
  - This ensures bugs can NEVER happen again at any level
