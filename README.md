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

7 pattern-only rules (no news in scoring) that scale position size 0.25x to 3.0x:

| Rule | Condition | Score |
|------|-----------|-------|
| 1 | Pole gain 4.5-9% (sweet spot) | +0.3 |
| 2 | Flag tightness < 30% / > 50% | +0.3 / -0.3 |
| 3 | Pole/flag vol ratio > 1.7x | +0.3 |
| 4 | SPY 3d range > 1.2% / < 0.8% | +0.3 / -0.5 |
| 5 | Retracement < 30% | +0.2 |
| 7 | VWAP distance >= 2% (extension above VWAP) | +0.2 |
| 8 | Gap fading (gap-up >=2% but breakout below open) | -0.3 |

(Rule 6 reserved — `daily_range_pct` was rejected as look-ahead: uses whole-day high/low which isn't knowable at setup detection.)

Combined with risk tier multiplier, capped at 3.0x. Applied via `create_plan(risk_multiplier=combined_mult)`. Config: `conviction_scoring.enabled: true`.

**Conviction also acts as a hard filter** when `conviction_scoring.min_threshold > 0`. Setups scoring below the threshold are skipped entirely (no order placed).

V2_clean (shipped 2026-04-15) added Rules 7+8 and raised the threshold from 1.2 → 1.4. Walk-forward OOS (test only, 3 chronological splits):

| Split | V0 (5 rules @ 1.2) test P&L | **V2_clean (7 rules @ 1.4)** | Δ |
|-------|------------------------------|-------------------------------|---|
| A: H1'25 → H2'25-Apr'26 | +$165,688 | **+$205,874** | **+$40,186** |
| B: Y2025 → Q1+Apr'26 | +$34,080 | **+$53,475** | **+$19,395** |
| C: Jan-Sep'25 → Oct'25-Apr'26 | +$102,104 | **+$127,887** | **+$25,783** |
| **Mean Δ** | | | **+$28,455** |
| **Worst Δ** | | | **+$19,395** |

**Canonical 16mo BT** (measured, post-ship on rebuilt cache): $338K → **$372K (+$34K, +10.1%)**, 161 → 145 trades, WR 54% → **56.6%**. The walk-forward rescaling heuristic estimated $390K; real BT came in $18K lower due to integer-share rounding and max-concurrent-3 reshuffling — lift is smaller than estimated but still directionally positive on every metric (fewer trades, higher WR, more P&L). Each skip is logged with per-rule breakdown. **Threshold is coupled to the 7 rules — re-validate if rules change.**

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

### Market Regime Filter (Vol-Only Mode)

**Primary block**: SPY 5-day avg daily range > **5.0%** → blocks all entries. Catches extreme regime chaos where momentum breakouts fail.

**Validated on 16mo (Jan 2025 – Apr 2026):**

| Metric | Filter OFF | Filter ON (vol>5%) |
|--------|-----------|---------------------|
| Total P&L | $347,276 | **$368,294** (+$21K) |
| Max DD | $51,889 | **$36,273** (-30%) |
| Win rate | 42.3% | 43.7% |
| Red months | 3 | 3 |
| Days blocked | — | 9 of ~350 (2.5%) |

**Why vol-only (not vol AND downtrend)**: Prior AND logic kept the filter dormant on Feb 2026 (flat SPY but extreme intraday chop) and blocked Mar 2025 winners (SPY down but low vol). Historical chaos happens in BOTH directions — vol alone catches it cleanly.

**Threshold 5.0% specifically**: There's a dead zone between 3% and 5% in SPY 5-day vol (either calm <2% or chaos >5%). Threshold 5.0% activates only during genuine regime breaks (Feb 2026 early-month spike to 18-19%, 1 Apr 2025 day). 14 of 16 months completely untouched.

**Secondary filters** (all independent, all default **disabled**):
- `sma_slope_filter`: block when SPY SMA(50) slope < -0.5 (dead-cat-bounce catcher)
- `euphoria_filter`: block when SPY UD ratio > 1.2 AND RSI(14) > 60 (FOMO top)
- `thin_liquidity_breakout_vol_ratio`: tightens breakout vol to 2.0x on thin SPY days (not a block — just raises the bar)

**Lookahead prevention**: All indicators use data strictly *before* trade date. T-1 close + 5-day vol window ending T-1.

| Config Parameter | Default | Description |
|-----------------|---------|-------------|
| `market_regime_enabled` | `true` | Enable filter (vol-only mode) |
| `market_regime_vol_threshold` | `5.0` | SPY 5d avg range % — blocks when exceeded |
| `market_regime_sma_period` | `50` | SMA lookback (used by sma_slope_filter only) |
| `max_trades_per_day` | `5` | Max entries per day |

### Two-Tier Filter (2026-04-17) — feature-flagged, default OFF

Tests whether lowering the scanner threshold from 20% to 10% can beat A_f6 by catching the same A-tier setups earlier plus a filtered subset of 10-19% "Extras" setups. Lives in `trading/two_tier_filter.py`, shared by BT Stage-2 and the live engine for BT↔live parity.

**Tiers (by `max(gap_pct, range_pct)` at/before entry bar):**

| Tier | Range | Treatment |
|---|---|---|
| **A** | ≥ 20% | Unfiltered — same setups A_f6 already takes |
| **Extras** | 10-19% | Surgical drop + composite z-score gate |
| **edge** | < 10% | Unfiltered — small residual near qualification boundary |

**Extras gate (both conditions must pass):**

1. **Surgical drop**: reject if `macd_zone_mult < 1.25` (one joint cell — Extras × MACD 1.0 — had +0.01 Kelly on TRAIN and −0.13 on VAL; empirically noise)
2. **Composite score**: 4-feature signed z-score average over `conviction_mult`, `qf_vwap_dist_pct`, `qf_fill_vwap_dist_pct`, `entry_minute` (all `sign=-1`, lower raw is better). Reject if average z < −0.50.

Frozen-fit params from Jan–Jul 2025 Extras subset (n=83); locked in `config.yaml`.

**Backtest results with flag enabled:**

| | A_f6 (reference) | **O + TTF** | Δ |
|---|---:|---:|---:|
| 2025 trades | 83 | **135** | +52 |
| 2025 P&L | +$54,572 | **+$65,027** | **+$10,455 (+19%)** |
| 2025 WR | 60.2% | 58.5% | −1.7pt |
| 2025 DD | $2,502 | ~$4,700 | +$2.2K |
| Q1 2026 P&L | +$4,495 | **+$9,934** | **+$5,439 (+121%)** |
| Q1 2026 WR | — | 47.8% | — |
| **Combined (2025 + Q1 2026)** | **+$59,067** | **+$74,961** | **+$15,894 (+27%)** |

**Parity verified:** with flag `false`, A_f6 produces byte-identical output ($54,572.15 to the cent, 83/83 trade match).

**Config block** (under `trading.bull_flag`):
```yaml
two_tier_filter:
  enabled: false                   # flip to true to activate
  extras_lower: 10.0
  a_tier_lower: 20.0
  drop_extras_macd_below: 1.25
  composite_threshold: -0.50
  composite_features:              # frozen z-score params (TRAIN fit)
    conviction_mult:       {mean: 1.789,   std: 0.284,  sign: -1}
    qf_vwap_dist_pct:      {mean: 4.218,   std: 2.239,  sign: -1}
    qf_fill_vwap_dist_pct: {mean: 4.604,   std: 2.274,  sign: -1}
    entry_minute:          {mean: 603.614, std: 20.195, sign: -1}
```

**Running backtests with the filter:**

```bash
# With flag OFF (current behavior — matches A_f6)
python batch_backtest.py --start 2025-01-01 --end 2025-12-31

# With flag ON (experiment) — write a one-off config, don't edit prod
cp config.yaml /tmp/config_ttf_on.yaml
sed -i 's/enabled: false  *# flip to true.*$/enabled: true/' /tmp/config_ttf_on.yaml
python batch_backtest.py --config /tmp/config_ttf_on.yaml --start 2025-01-01 --end 2025-12-31

# Expected with flag ON:
#  2025:   135 tr, WR 58.5%, P&L +$65,027
#  Q1 26:   46 tr, WR 47.8%, P&L +$9,934
```

**Deploying to live prod:**

```bash
# Edit the real prod config
vi config.yaml   # set trading.bull_flag.two_tier_filter.enabled: true

# Validate config loads
python -c "from config import Config; print(Config().two_tier_filter_cfg)"
# → {'enabled': True, 'extras_lower': 10.0, 'a_tier_lower': 20.0, ...}

# Restart service
sudo systemctl restart onemil-trader

# Monitor rejections
journalctl -u onemil-trader -f | grep "TWO-TIER FILTER"
# → "...: TWO-TIER FILTER SKIP (tier=E, max_intraday=14.3%, macd_mult=1.00): extras_macd_surgical_drop"

# Rollback (zero state to unwind — pure flag flip)
vi config.yaml    # flip enabled: true → false
sudo systemctl restart onemil-trader
```

**Tests:**

```bash
pytest tests/test_two_tier_filter.py -v              # 37 unit tests (classifier, scorer, bar replay)
pytest tests/test_two_tier_filter_integration.py -v  # 13 BT-live parity tests
pytest tests/                                         # full suite (1,142 passing)
```

**Key implementation notes:**

- `intraday_change_at_entry` is computed at setup-fire time in `backtest.py` and persisted as a cache column. Any backtest that runs pre-deploy code on a post-deploy cache will see the new column; old caches (missing column) still work via `.get('', '')` fallback (trades classify as edge → passthrough).
- Live engine reads `self._qualified_max_intraday[symbol]` populated by the scanner via the `max_intraday_change_pct` kwarg on `on_stock_qualified()`. Scanner maintains running max per symbol.
- When `macd_zones.enabled=false`, the engine passes `macd_zone_mult=None` to the gate, which skips the surgical drop (signal unavailable) and evaluates only the composite.
- Frozen composite params live in YAML — retune requires re-fitting on TRAIN and updating the YAML values. Do NOT refit per-backtest (look-ahead).

**Out of scope (tested but NOT shipping this PR):**

- **Re-entry** (multiple trades per symbol per day) — empirically −$1,299 over 58 trades. Env-gated via `BT_ALLOW_REENTRY=1`, dormant in code.
- **Multiplier re-tuning** (Kelly v3 capping to 2.0×) — reserved for when base risk scales to $1,000+ (current $200 base, rubric is P&L-max as-is).

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

### V4 Conviction Sizing (shipped 2026-04-14)

Position size is scaled by a 2-rule conviction score at entry time. Formula lives in `trading/macd_conviction.py` — imported by BOTH the BT and PROD engine so they can't drift.

| Rule | Top tier (+0.4) | Second (+0.2) | Third (+0.1) | Bottom (0.0) |
|---|---|---|---|---|
| `cross_time_min` (minutes open→+10% cross) | ≤3 | ≤5 | ≤7 | >7 |
| `vol_at_cross` (cumulative shares at cross) | ≤27K | ≤79K | ≤165K | >165K |

Score = `1.0 + cross_contrib + vol_contrib`, clamped to `[0.5, 2.0]`. Range on current rules: **1.0 → 1.8**. Position: `shares = int(position_size × score / entry_price)`. Hard cap via `sizing.conviction_sizing.max_position_size_usd` in yaml (default $90K).

**Walk-forward evidence** (`analysis_results/macd_conviction_step2_*.md`):

| Split | Test ΔP&L (V4 vs flat) |
|---|---|
| A — Train H1'25 → Test Jul'25-Mar'26 | +$33,851 |
| B — Train Jan-Dec'25 → Test Q1'26 | +$10,305 |
| C — Train Jan-Sep'25 → Test Oct'25-Mar'26 | +$29,668 |
| **Mean** | **+$24,608** (worst +$10,305 — robust across all 3 splits) |

**Canonical 15mo BT**: baseline $109K / DD -$9.5K → V4 sized **+$163K / DD -$13K** (+49.5% P&L, same 551 trades).

**Filter approach was tested and rejected.** Dropping low-conviction trades fails OOS (mean -$153, worst -$5.9K — the low-conv bucket still has positive EV in test). Used as sizing multiplier only.

Toggle via `sizing.conviction_sizing.enabled` in `macd_wave.yaml`. Every entry logs its breakdown: `[macd_wave] SYM: ... CONVICTION 1.80 (cross=+0.4 vol=+0.4; pos=$90,000)`.

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
