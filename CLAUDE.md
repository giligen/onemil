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

### `min_pole_candles` tested 3 → 2 — **NOT SHIPPED** (2026-05-15)
- Motivation: live this week 3,234 "Pole too short (2 candles, need 3+)" rejections. AIIO 2026-05-13 rejected as 2-candle pole at 14:14, ran +52% intraday. Same pattern killed SMX/MASK/KPTI live wins.
- **2025 OOS (12 months)** with pole=2: **+$63,172 (+18.3%)**, WR 46.8%→47.9%, 267→338 trades. 8 better months, 4 worse.
- **2026 OOS Jan-Apr (4 months)** per-month head-to-head: **3 of 4 months WORSE under pole=2**. Jan −$20K, Feb −$7K, Mar −$11K, Apr +$22K. Net 4mo: **−$16,007**.
- **Combined 16 months**: net +$47K (+5%). Annualized ~$35K/yr — but 3-of-4 recent months negative + WR drop in 3/4 months reads as overfit to 2025 regime, not durable signal. Same "marginal positive with bad recency" pattern that justified rejecting the earlier `max_pullback 5→10` change.
- Trade-level diff (2025): 79 new (symbol, date) added in pole=2, WR 48.1%. Top winners MSW +$2.2K / RYET +$1.3K / BQ +$1.2K (broad signal in 2025 only).
- Other knobs tested and **rejected** in the same sweep: `max_retracement_pct 50→70` (−$24K 2025 alone), `max_green_in_flag 2→3` (−$119K!), `min_breakout_volume_ratio 1.5→1.0` (BT-inert), `min_pole_gain_pct 3→2` (+$28K but dominated by pole=2).
- **Kept as research infrastructure**: env-var overrides in `trading/pattern_detector.py` (`BF_MIN_POLE_CANDLES`, `BF_MAX_PULLBACK_CANDLES`, `BF_FVRR_STRICT`, `BF_MIN_POLE_GAIN_PCT`, `BF_MAX_RETRACEMENT_PCT`, `BF_MAX_GREEN_IN_FLAG`, `BF_MIN_BREAKOUT_VOLUME_RATIO`) — default OFF, no behavior change without explicit env var. Also `fvrr_strict` constructor flag (default True, matches existing FVRR-on behavior).
- Reconsider when: 2026 Q2+ accumulates 3+ months of live data that flips supportive, OR a regime model can predict which months favor 2-candle vs 3-candle poles.
- Sweep artifacts: `scripts/study_bf_wide_sweep_2025.sh`, `scripts/diff_a_pole2_trades.py`.

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
- Spread gate: skip entries with spread > 300bps + Telegram warning (loosened from 150 on 2026-07-04 — the 150 gate skipped monsters BKKT/XNDU; NEVER tighten below 150 without rereading research/orb_spread_gate_verdict.md: 100-150bps is the richest per-trade bucket)

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

**Rollout phases — see `docs/orb_rollout_plan.md` for the live cushion-gated ramp**:
- Pre-Stage-0 LIVE (data collection): $15K budget / $500 risk / -$750 daily loss — half-size live to capture real fill quality vs paper. Hard stop -$3K cushion. Use `scripts/orb_pre0_daily.py` for daily monitoring.
- Stage 0: $30K budget / $1K risk / -$1.5K daily — formal live launch
- Stages 1-4: $50K → $174K (full DTBP). Cushion + days-in-stage gated.

**Q1 filter (shipped 2026-04-25, default ON)**: drops bottom-quintile candidates at ranking time. BT lift +$8,556 OOS (no DD increase). Config: `orb.yaml::filter.skip_q1: true` (also via `ORB_SKIP_Q1=0` to disable in BT). Validated by `study_orb_q1q2_filter.py`. Slot mechanics: filter never refills slots (Q1 is last-priority — see `check_q1_refill_potential.py`). Monitor: `journalctl -u onemil-trader | grep "Q1 filter"`.

**Touchgo filter (Rule M + Rule D) (shipped 2026-05-16, default ON)**: two post-fill exit rules that catch failed breakouts within the first 1-2 minutes of trade life.
- **Rule M**: at the close of the breakout bar (the bar that triggered our stop-limit BUY), if its close was in the bottom half of its high-low range (`bb_close_pos < 0.5`), exit at next bar open. Catches "touch and go" failed breakouts.
- **Rule D**: at the close of the first post-entry bar, if the bar's low went ≥0.75R below entry (R = range_high - range_low), exit at entry - 0.5R. Catches fast reversal patterns.
- **BT validation**: walk-forward Jan'25-May'26 (924 trades, 8/11 OOS months helped, +$27K OOS lift, +$26K full-timeline pipeline-integrated lift, **WR 47.8% → 52.1%, negative months 4 → 2**). Threshold 0.5/0.75 stable across all rolling training windows.
- **Shared module**: `trading/orb_touchgo_filter.py` (imported by both BT `study_orb_pipeline_static_lock.py` and live `trading/orb_engine.py` — parity by construction; enforced by `tests/test_orb_touchgo_parity.py`).
- **Live wiring**: `_evaluate_touchgo` called from `_ingest_bars` on every bar event; on fire, calls `stop_monitor.force_exit(symbol, reason='tag_bb'/'tag_b1', limit_price=...)` (new public method on StopMonitor) which routes through the same exit machinery as autonomous stops. Sends `[ORB] TAG_BB/TAG_B1 EXIT` Telegram message with bb_close_pos or b1_revert_R, exit price, and saved-vs-full-stop estimate.
- **Config**: `orb.yaml::filter.touchgo.{enabled,rule_m.{enabled,threshold},rule_d.{enabled,revert_R,exit_R}}`. Env-var overrides: `ORB_TOUCHGO_ENABLED=0` (master), `ORB_TOUCHGO_RULE_M_THRESH`, `ORB_TOUCHGO_RULE_D_R`, `ORB_TOUCHGO_RULE_D_EXIT_R`.
- **Monitor**: `journalctl -u onemil-trader | grep -E "TAG_BB|TAG_B1|touchgo"`. Expect ~3 firings/day (BT prevalence 26% of fills × ~12 daily entries).
- **Rollback**: `filter.touchgo.enabled: false` + `sudo systemctl restart onemil-trader` (zero-state — filter only fires within first 2min post-fill).

**News-gated PM sizing mult (shipped 2026-07-10, live 2026-07-13, default ON)**: the premarket dollar-volume boost is gated on pre-market NEWS presence (Alpaca/Benzinga, window prev-day 15:00 ET → fetch time ~9:31).
- **Semantics**: PM$ > $5.82M cut AND has_news **AND identified COMMON STOCK** → **2.0×**; everything else → 1.0×. PM$ high without news → 1.0 (flat bucket); news without PM$ → 1.0 (headline nobody trades is a dud). Fail-open both channels: fetch failure → no boost, loud WARNING.
- **Asset-class rule (2026-07-11, deliberate)**: 45% of the universe are leveraged wrappers (2x/inverse single-stock ETFs). They have no company events; every news window tested fails for them (same-morning underlying news = crowding, NEGATIVE all 3 eras). The boost requires positive stock identification via `trading/orb_asset_class.py` (lev-family sets → 33K offline map `data/research/orb_asset_class_map_20260711.csv` → `get_asset_name` API → unknown never boosts). Cost vs the accidental gate: $5.6K/18mo; buys immunity to vendor tagging changes. Full rule book: `research/orb_machine_rules.md`. Do NOT map wrapper news to underlyings or industries (REFUTED); do NOT extend the window to prev-day session news (NO-SHIP, recency-dead).
- **Cannot kill/delay trades (2026-07-10 edge-case audit)**: the whole PM/news stack can only size a trade 1.0-2.0× — no path skips/vetoes/zeroes one (min mult 1.0; shares can't floor to 0 at our price band). News fetch: 8s hard timeout, 0 timeout-retries, 0 rate-limit-retries (`NEWS_API_TIMEOUT`; the default 90s×2 + 429-backoff ladder ≈3min was unacceptable in the entry window), failure poisons the day's flags (one attempt, never re-blocks a tick). 9:33 upgrade-only second pass covers Benzinga indexing lag (a systematically missed newsy flag would ship the worst grid row: −$62K/18mo — the EoD news-drift check is the tripwire for residual lag).
- **Evidence** (research/orb_news_catalyst_jul2026.md): news×PM$ combo cell +$1,580/+$1,569/+$935 per trade per era — strongest era-consistent separator since PM$ itself; monster rate 28/15/13% vs 6-8% rest; zero lookahead (all articles ≤9:30 ET). Pipeline: TOT $250,276→$301,518, all eras +, MDD improves −$18.8K→−$18.2K. Known texture: lift is monster-concentrated (top-5 = all of it) and combo big-loser rate rises era-over-era (3→8→10%) — expect a slow bleed punctuated by rare large wins; judge on monsters-included windows only.
- **Do NOT** add an LLM/keyword catalyst-quality filter for longs: REFUTED — recap-only articles ("20 stocks moving premarket") perform equal to real catalysts and hold AMCI +$23K / BNAI +$13.6K. (Opposite of stupid-money's short-divergence use case.)
- **Shared helper**: `trading/orb_pm_mult.py::pm_size_multiplier` (BT `study_orb_pipeline_static_lock.py` + live `orb_engine._get_pm_mult` — parity by construction; BT news source: `data/research/orb_news_catalyst_*.csv`, regen via `research/scripts/orb_news_backfill.py`).
- **Config**: `orb.yaml::sizing.pm_dollar_vol_mult.{high_mult: 1.0, high_mult_news: 2.0, news_gate: true}`. Env: `ORB_PM_NEWS_GATE=0` (no news fetch, everything above cut at high_mult), `ORB_PM_MULT=0` (whole mult off).
- **Rollback to pre-gate legacy**: `news_gate: false` + `high_mult: 1.5` + restart (zero state).
- **Monitor**: `journalctl -u onemil-trader | grep -E "PM MULT|NEWS prefetch"`. EoD: the daily green check now prints per-trade sizing attribution (quintile × pm_mult × news flag), HARD-fails the day on recorded-vs-recomputed pm_mult drift, soft-flags live-vs-EoD news drift, and tracks Q2/Q3 vs Q4/Q5 vs news-boosted cumulative P&L since 2026-07-13 (`scripts/report_common.py::sizing_attribution`).

**Catalyst-required veto (shipped 2026-07-18, live 2026-07-20, default ON)**: every ORB entry needs a CATALYST — own-ticker premarket news OR complex confirmation (≥2 same-morning candidates sharing the underlying anchor: a stock + its wrappers, or sibling wrappers of one underlying). Newsless-and-alone picks are vetoed POST-ranking, slot consumed, NO refill (refill re-tested toxic: MDD +42%).
- **Evidence** (live-parity resim): book $293,568→$253-257K (−$36K, owner-approved budget), MDD −$16.3K→−$14.0K, worst month −$10.3K→−$7.8K, July-26 bleed −62%, trades −67%. Cost era-consistent. Disclosed trade-off: negative months 6→9 (all shallow). Newsless-alone universe cohort NEGATIVE all 3 eras; complex-confirmed newsless positive all 3.
- **Shared helper**: `trading/orb_catalyst_veto.py` + `orb_asset_class.underlying_anchor` (BT pipeline + live engine — parity by construction). Fail-open on unknown news (fetch failure never vetoes).
- **Config**: `orb.yaml::filter.catalyst_veto.{enabled,min_cohort}`. Env: `ORB_CATALYST_VETO=0`.
- **Monitor**: `journalctl -u onemil-trader | grep "CATALYST VETO"`; pattern_data records anchor + anchor_cohort per trade.
- **Rollback**: `enabled: false` + restart (zero state). Tests: `tests/test_orb_catalyst_veto.py` (22).

**PDR veto — prev-day-range (shipped 2026-07-04, default ON)**: skips selected picks whose PREVIOUS day's range was quiet (`prev_day_range_pct <= 8.0`). ORB monetizes continuation — "day-2 of the fireworks, not day-1"; quiet-prev-day gappers are fresh pops that mean-revert.
- **NO-REFILL invariant**: applied POST-ranking inside the submit loop — a vetoed pick's slot stays EMPTY. The refill form was tested and is TOXIC (2025H2 → ~$0, MDD −$29K→−$50K; same failure mode as the refuted ETF exclusion). Never "improve" this by backfilling.
- **BT evidence** (defended replica, Jan'25–Jul'26): TOT $155K→$210K (+35%), MDD −$29.3K→−$20.1K, WR 35.8→40.2%, trades/day 3.3→1.6, all 3 eras positive (25H1/25H2/2026), monotone across thresholds 6–10, ALL top-10 giants kept.
- **Shared helper**: `trading/orb_pdr_veto.py` (imported by live `orb_engine._pdr_veto_reject` + BT `study_orb_pipeline_static_lock.py` — parity by construction). Feature def matches `study_orb_features.py:287`.
- **Config**: `orb.yaml::filter.prev_day_range_veto.{enabled,min_prev_day_range_pct}`. Env: `ORB_PDR_VETO=0` (disable), `ORB_PDR_VETO_MIN_PCT` (threshold override).
- **Monitor**: `journalctl -u onemil-trader | grep "PDR VETO"` — one line per vetoed pick (or per fail-open on missing prev-day data).
- **Rollback**: flip `enabled: false` + restart (zero state). Tests: `tests/test_orb_pdr_veto.py` (27).

**Touchgo breakout-bar re-keying + late-fill guard (shipped 2026-06-04, default ON)**: fixes a BT↔LIVE parity gap discovered comparing paper(dev) vs live(prod).
- **Bug**: live keyed Rule M/D to the minute of the actual *fill* (`breakout_bar_ts = minute(fill)`), but BT keys to the *market breakout bar* (first 1-min bar with `high > range_high`). When a stop-limit fill lagged the breakout, live evaluated a different bar → **23% of live fills (7/31, May 19–Jun 3) flipped the `tag_bb` decision**, skewed toward spurious early exits.
- **Fix**: live now captures the market breakout bar during the pending phase (`_ensure_breakout_bar_ts` from `_ingest_bars`) via the shared `trading.orb_touchgo_filter.find_breakout_bar_ts` (BT calls the same helper — parity by construction). Robust to late fills (captured while the bar is still in the streamed window).
- **Late-fill guard**: if the fill lagged the breakout bar by > `max_breakout_age_min` (default 15), touchgo is skipped — a stale entry (e.g. ASTN 2026-06-03 filled 34min late) is no longer a clean ORB and gets no retroactive tag exit.
- **Counterfactual**: on the 33-trade live sample the fix nets **+$251.8** (7 flipped trades −$223 → +$29; e.g. re-enables the failed-breakout cut on LMRI −3.84%→tag_bb, drops spurious cuts on IHRT/PURR). Directionally restores the BT-validated +$27K touchgo edge; small sample, not an annual projection.
- **Config**: `orb.yaml::filter.touchgo.{breakout_bar_source: market|fill, max_breakout_age_min: 15}`. Env: `ORB_TOUCHGO_BREAKOUT_BAR_SOURCE`, `ORB_TOUCHGO_MAX_BREAKOUT_AGE_MIN`.
- **Rollback**: `filter.touchgo.breakout_bar_source: fill` + restart (restores legacy fill-bar behaviour, zero state). Audit scripts: `scripts/audit_touchgo_breakout_bar_gap.py`, `scripts/audit_touchgo_fix_pnl_delta.py`.
- **Tests**: `tests/test_orb_touchgo_parity.py` (helper unit + BT/live parity), `tests/test_orb_engine.py::TestTouchgoBreakoutBarReKey` (capture, fire-on-breakout-bar, late-fill guard, legacy mode).

**ORB diagnostic scripts** (in `scripts/`):
- `orb_ramp_check.py` — current stage + advancement eligibility (cushion + days)
- `orb_pre0_daily.py` — Pre-Stage-0 daily monitor (cushion, slippage vs BT, promotion eligibility, demotion triggers). Refuses to run if orb.yaml ≠ Pre-0 spec unless `--launch-date` passed.
- `analyze_orb_slippage.py` — per-trade entry/exit slippage vs BT 30/10 bps
- `investigate_composite_drift.py` — diff live `ORB SCORED` log vs BT features CSV to find feature responsible for any composite drift

**ORB research summary**: `docs/orb_research_apr_2026.md` (50+ exit/add-to-winners variants tested April 2026 — V0 confirmed Pareto-frontier; only Q1 filter shipped; bull-flag-gated add-to-winners parked due to small ~$10K/yr lift).

**orb.yaml** is gitignored (instance-specific config). Use `orb.yaml.template` as base for new node setup: `cp orb.yaml.template orb.yaml`.

**Do NOT**:
- Enable with `ALPACA_ORB_API_KEY` empty — main.py will warn + disable
- Refit z-score / quintile / adaptive params AT ALL without a walk-forward harness proving the new fit OOS.
  **The quarterly-refit cadence is CANCELLED (2026-07-03 audit)**: quarterly refits tested $34–47K WORSE
  OOS than the frozen H1-2025 fit across 4 quarters (static +$126.8K vs expanding-refit +$79.5K vs
  rolling-12mo +$92.7K) — refits chase the recent regime and get whipsawed. The frozen fit is an
  accidental regularizer. See research/money_machine_audit_jul2026.md #4 +
  research/scripts/orb_pipeline_replica.py (the harness).
- Remove Q5 cap from `orb.yaml::adaptive_mults.Q5: 1.5` — it's the anti-overfit guard
- Disable Q1 filter (`filter.skip_q1`) without revisiting `docs/orb_research_apr_2026.md` first
- Skip Pre-Stage-0 LIVE phase before formal Stage 0 — paper data has structural limits (synthetic fills don't capture real venue queue)

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

#### Stage 2: Run filtered backtest (RELATIVE tool — NOT a P&L forecast)
```bash
# Default behavior — reads from cache, applies production filters from config.yaml
python batch_backtest.py --start 2026-01-01 --end 2026-03-31
```
- Reads from cache, applies: 20% threshold, 200K volume, leveraged ETF filter, max 3 concurrent, $5K daily loss limit, risk tiers, **buying-power ceiling** (`bt_buying_power_usd`, added 2026-05-14)
- Takes <1 second (reads from cache), so there is ZERO reason to skip this step

**Stage 2 is a RELATIVE tool, not a P&L forecast.** It models *some* of the
live sizing/filter stack but NOT all of it. Parity status:
- **Regime sizing (A/B/C1/C2)**: MODELED as of 2026-07-04 (assumption-ledger
  fix) — day-level multiplier + C2-day skip via `trading/regime_helpers`,
  applied before the daily-loss accumulator like live. `BT_REGIME_SIZING=0`
  restores pre-fix behavior for old relative comparisons (verified
  byte-identical: $31,864.54/74 trades on 2025-01→2026-07-02).
- **UD scaling** (SPY up/down-volume euphoria guard): still live-only.
- Plus the 6 structural BT/LIVE drift sources in the
  `project_bull_flag_drift_findings` memory (20% threshold mismatch,
  scan_results bug, entry latency, exit divergence, …).

→ Stage 2 is valid for **feature A/B comparisons** ("does filter X help vs
not-X?") — the unmodeled layers cancel in the diff. It is **NOT** valid as
an absolute P&L projection. The only honest P&L forecast is accumulated
**LIVE** data. Do not tell the user "BT says we'll make $X".

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
