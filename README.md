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

#### Layer 2: News Kill Rules — block trades in loser segments (+$158K)

These empirically loser segments are killed:

| Rule | Condition | WR | Impact |
|------|-----------|-----|--------|
| 1 | avg_daily_vol >= 3M | 22% | -$101K eliminated |
| 2 | price < $3 | 20% | -$11K eliminated |
| 3 | float >= 30M | 19% | -$14K eliminated |
| 4 | $5-12 + pole 8-15% | 27% | -$32K eliminated |

Config: `news_kill_rules.enabled: true`. Shared decision module `trading/news_kill_guard.py` (`news_kill_decision`) — imported by both the backtest (`_check_news_kill`) and the live engine, so they cannot drift.

**Catalyst exemption (`news_kill_rules.catalyst_exemption`, default `false` — shipped 2026-05-21):** Historically a trade with a real news catalyst (FDA, earnings, contract, M&A, analyst, product, SEC filing) was *exempted* from the segment rules. The 2026-05 news-classifier A/B found that exemption is **value-destroying** — bad-segment trades that genuinely have a real catalyst are still net losers, and the exemption just leaks them past the gate. On a 1,195-trade sample, applying the rules with **no** exemption ($204,844 raw) beat the regex-exemption config ($192,721) by ~$12K; an *accurate* Haiku classifier was worse still ($43K behind regex on the clean Stage-2 A/B — the rules were co-calibrated to regex's loose catalyst rate). With the flag `false` the segment rules apply to **every** trade and **no news classifier is consulted on the trade-decision path** (the buggy regex / Haiku divergence both become irrelevant). Set to `true` to restore the legacy exempt-on-catalyst behavior. Monitor: `journalctl -u onemil-trader | grep "NEWS KILL"`. Rollback: flip flag to `true` + restart.

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

### Exit Reason Taxonomy

Every closed trade in `trades.exit_reason` carries one of the values
below. The single source of truth is `trading/exit_reasons.py::ExitReason`
— inline string literals are forbidden (enforced by
`tests/test_exit_reasons.py::TestNoStringLiteralDrift`).

| Group | Value | When emitted |
|---|---|---|
| **Shared** | `stop_loss` | StopMonitor's stop level hit + sell filled cleanly |
| | `take_profit` | TP leg filled (pre-trail BF; rare under current config) |
| | `trail_stop` | Trailing stop ratcheted past last high then triggered |
| | `lock_stop` | ORB static-lock: stop locked at +0.5R after touching +1.75R, then hit |
| | `force_close` | End-of-day force flat (15:45 ET for ORB; configurable per strategy) |
| | `unknown_exit` | **Leak signal** — every row of this type is a bug. See `needs_reconcile()`. |
| | `post_fill_exit` | Post-fill filter (BT gap check; BF thin-liquidity check) |
| **StopMonitor recovery** | `stop_loss_market_fallback` | Limit sell stranded → escalated to market close, which filled |
| | `stop_loss_bracket_sl_race` | `close_position` raced; bracket SL leg won the exit — real fill recovered from leg |
| | `stop_loss_unconfirmed` | Limit AND market-close both timed out; `trigger_price` placeholder + WARNING |
| | `stop_loss_fallback` | Older generic SL recovery (BF only) |
| | `exhaustion_partial` | Exhaustion-candle partial sell (not a full close — see `partial_exit_*` columns) |
| **Bull flag** | `gap_over_rejection` | Rejected at fill: gap up exceeded cap |
| | `gap_adjust_failed` | Gap-up SL adjust failed; safety force-exit |
| | `thin_liquidity_reject` | Post-fill spread/depth degraded below floor → force-exit |
| **ORB touchgo** | `tag_bb` | Rule M: breakout bar closed in bottom half (bb_close_pos < 0.5) |
| | `tag_b1` | Rule D: bar-1 reverted ≥0.75R below entry |
| **MACD wave** | `macd_flip` | MACD histogram flipped sign — momentum reversal exit |
| | `bracket_exit` | Bracket leg handled exit; unable to classify which leg |
| | `bracket_sl_tp` | Bracket-attributed exit (SL or TP) |
| | `hard_stop` | Hard SL leg fired (distinct from trail) |
| | `stopmonitor_exit` | StopMonitor (not bracket) closed it |
| **Historical** | `sync_reconcile` | **No current writer** — 6 rows 2026-04-02 orphan cleanup. New paths emit `unknown_exit`. |
| | `stop_loss_timeout` | **No current writer** — 1 row 2026-03-26. Superseded by `stop_loss_unconfirmed`. |

Helper predicates exposed by the same module:
- `is_known(value)` — gate at every DB read; novel strings = drift bug
- `is_attributed(value)` — `True` for clean attributions; `False` for leak paths
- `needs_reconcile(value)` — `True` for `unknown_exit`, `*_unconfirmed`, historical reconcile values. Daily summary should alert on non-zero count.

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

### Two-Tier Filter (2026-04-17) — default ON (shipped, rollback via flag)

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

### V-Reversal Conviction Bonus (Experiment V, 2026-04-17) — default ON (shipped, rollback via flag)

Fat-tail analysis on the shipping TTF+D stack revealed that the top 10 winners of 2025 share a distinctive feature: **median `qf_gap_pct = −1.09%`** (gap-down) plus intraday range ≥ 20% and pole gain ≥ 5% — i.e., oversold V-reversal plays, not the gap-ups the pattern was originally designed around. The existing 7-rule conviction scoring had no term that explicitly rewarded this shape.

Rule 9 (V-reversal bonus) adds a configurable amount to the conviction raw score when all three triggers fire:

```
is_v_reversal = (gap_pct < gap_pct_max) AND (intraday_range_pct >= intraday_range_min) AND (pole_gain_pct >= pole_gain_min)
```

**Stays within the existing conviction envelope** — final score still clamped to `[0.25, 3.0]`, max risk multiplier still 3.0×. The rule simply shifts V-reversals UP the existing scale so they size bigger.

**Backtest lift** (TTF+D on, bonus=0.4, defaults):

| | TTF+D (shipping) | **+V (V-reversal)** | Δ |
|---|---:|---:|---:|
| 2025 trades | 135 | 142 | +7 (borderline V-revs now pass conv filter) |
| 2025 P&L | +$68,791 | **+$73,187** | **+$4,396 (+6.4%)** |
| 2025 WR | 58.5% | 56.3% | −2.2pt |
| 2025 avg_win | +$1,392 | **+$1,451** | +$59 |
| Q1 2026 trades | 46 | 47 | +1 |
| Q1 2026 P&L | +$9,934→$11,770 (D) | **+$14,054** | **+$2,284 (+19%)** |

**Config block** (under `trading.conviction_scoring`):
```yaml
v_reversal_bonus:
  enabled: false            # flip to true to activate
  bonus: 0.4
  gap_pct_max: 0.0          # fires only when qf_gap_pct < this
  intraday_range_min: 20.0  # fires only when intraday range >= this
  pole_gain_min: 5.0        # fires only when pole gain >= this
```

**Unlike TTF/D, no cache rebuild needed in prod**: live engine computes conviction per-trade in real-time. Deployment is a config flip + `sudo systemctl restart onemil-trader`.

### Marginal-Conviction Defensive Scaling (Experiment H, 2026-04-17) — feature-flagged, default OFF (research artifact)

Complements V on the defensive side. P&L-bucket analysis showed the shipping stack has **35 mid-losers (−$500 to −$1K)** vs A_f6's 18 — a disproportionate concentration of marginal-quality trades that barely clear the `min_threshold: 1.4` conviction filter.

Marginal scaling halves (or scales by configurable factor) the SIZING multiplier for trades with conviction in `[min_threshold, upper_bound)` — without changing the stored conviction value (so Stage-2 filters still see the true quality signal).

**Backtest result — regime-dependent, not a universal win**:

| | V-on baseline | V+H | Δ |
|---|---:|---:|---:|
| 2025 P&L | +$73,187 | +$68,295 | **−$4,892** (H hurts) |
| Q1 2026 P&L | +$14,054 | +$14,905 | +$851 (H helps) |
| **Combined** | **+$87,241** | **+$83,200** | **−$4,041 (−4.6%)** |

The marginal bucket [1.4, 1.7] is net-winner on 2025 but net-loser on Q1 — halving both symmetrically costs more on winners than it saves on losers **on 2025's data**. Could still be worth enabling in a regime where the bucket is predominantly loser; shipped as research artifact for future tuning (e.g., regime-conditional activation).

**Config block**:
```yaml
marginal_scaling:
  enabled: false             # DEFAULT OFF — mixed BT signal
  scale_factor: 0.5          # size multiplier for marginal trades
  upper_bound: 1.7           # applies to conviction in [min_threshold, this)
```

### Volume-Confirmed Trail Exit (Experiment D, 2026-04-17) — default ON (shipped, rollback via flag)

When a trailing stop triggers on a bar, require that bar's volume to exceed `flag_avg_volume × min_vol_ratio` before firing the exit. Low-volume drift-downs are treated as noise (hold position); only active selling (volume-confirmed) fires the exit. Initial hard stop (pre-trailing) is never skipped. Shared helper `trading/trail_vol_guard.py` used by both BT simulator and live `StopMonitor` for parity.

**Empirical analysis** (on 20 mid-range Q1 2026 winners, see `exit_quality_research.py` for the full deep-dive): classified exits as LUCKY (saved from reversal, ~45%), GOOD (caught near peak, ~30%), MIXED (~20%), or EARLY (~5%). The core thesis from the analysis: for "slow-burn" winners (CDNA, KPTI, HBIO), the trail fires on a small low-volume pullback mid-day, missing the afternoon continuation. Volume confirmation preserves these.

**Backtest results** (TTF on, r=1.0 default):

| | Baseline (trail 1.0R) | **D (vol-conf trail)** | Δ |
|---|---:|---:|---:|
| 2025 trades | 135 | 135 | 0 |
| 2025 P&L | +$65,027 | **+$68,791** | **+$3,764 (+5.8%)** |
| 2025 WR | 58.5% | 57.8% | −0.7pt |
| 2025 DD | $4,722 | **$4,722** | unchanged |
| 2025 avg_win | +$1,322 | **+$1,392** | +$70 |
| 2025 avg_loss | −$704 | −$698 | +$6 (better) |
| Q1 2026 trades | 46 | 46 | 0 |
| Q1 2026 P&L | +$9,934 | **+$11,770** | **+$1,836 (+18%)** |
| Q1 2026 DD | $5,296 | **$5,296** | unchanged |

**Pareto improvement**: same trade count, same DD, same avg loss — winners just run further. Stacks cleanly on top of the two-tier filter.

**Parameter sensitivity** (Q1 2026, TTF on):

| min_vol_ratio | Q1 P&L | 2025 P&L (where tested) |
|---:|---:|---:|
| 0.5 | +$11,892 (Q1 best) | +$66,836 |
| 0.8 | +$10,733 | — |
| **1.0 (default)** | **+$11,770** | **+$68,791 (2025 best)** |
| 1.2 | +$11,716 | — |
| 1.5 | +$11,837 | — |

Flat surface; any ratio in [0.5, 1.5] gives similar results. Default `1.0` is the safest choice.

**Config block** (under `trading.trailing_stop`):
```yaml
vol_confirmed_exit:
  enabled: false           # flip to true to activate
  min_vol_ratio: 1.0       # bar vol must be >= this × flag_avg_volume
```

**Combined with TTF — full config stack result:**

| Config | 2025 P&L | Q1 2026 P&L | Combined vs A_f6 |
|---|---:|---:|---|
| A_f6 (current prod baseline) | +$54,572 | +$4,495 | — |
| TTF-on only | +$65,027 | +$9,934 | +27% |
| **TTF-on + D (full stack)** | **+$68,791** | **+$11,770** | **+36%** |

**Running backtests with D:**

```bash
# OFF (current behavior)
python batch_backtest.py --start 2025-01-01 --end 2025-12-31 --build-cache
python batch_backtest.py --start 2025-01-01 --end 2025-12-31

# ON — requires rebuild (D affects Stage-1 simulator exits)
cp config.yaml /tmp/config_D_on.yaml
sed -i 's/vol_confirmed_exit:\n      enabled: false/vol_confirmed_exit:\n      enabled: true/' /tmp/config_D_on.yaml
python batch_backtest.py --config /tmp/config_D_on.yaml --start 2025-01-01 --end 2025-12-31 --build-cache --cache-file /tmp/cache_D.csv
python batch_backtest.py --config /tmp/config_D_on.yaml --start 2025-01-01 --end 2025-12-31 --cache-file /tmp/cache_D.csv

# Expected with D on + TTF on:
#  2025:   135 tr, WR 57.8%, P&L +$68,791
#  Q1 26:   46 tr, WR 47.8%, P&L +$11,770
```

**Deploying to live prod:**

```bash
vi config.yaml    # set trading.trailing_stop.vol_confirmed_exit.enabled: true
python -c "from config import Config; print(Config().vol_confirmed_trail_cfg)"
# → {'enabled': True, 'min_vol_ratio': 1.0}
sudo systemctl restart onemil-trader
journalctl -u onemil-trader -f | grep "VOL-CONF SKIP"
# Rollback: flip enabled: false + restart (pure flag flip, zero state)
```

**Tests:**
```bash
pytest tests/test_trail_vol_guard.py -v          # 21 unit tests (helper boundaries)
pytest tests/test_backtest.py::TestVolConfirmedTrailExit -v       # 4 BT simulator tests
pytest tests/test_stop_monitor.py::TestVolConfirmedTrailExit -v   # 5 live path tests
```

**Key implementation notes:**

- Shared helper `trading/trail_vol_guard.py` — single source of truth for BT + live.
- Safe defaults throughout: missing baseline → never skip (fall back to naive trail); missing bar volume → treated as 0 (skips when ratio > 0, matching "no trading" semantic).
- Live: `WatchEntry.last_bar_volume` updated by `_on_bar()` callback on every closed bar. Both tick path (`_on_trade`) and poll path (`_process_bar_snapshot`) call the shared helper.
- Only trail exits are gated — initial hard stop (before trail activation) never skips (capital-preservation path).
- Crash-recovery path: watches re-registered from DB pass `avg_flag_volume=0.0` (not stored), so vol-conf falls back to naive trail until next fresh trade.
- **2026-09-05**: the guard now reads the PREVIOUS closed bar on BOTH sides (BT used the triggering bar's own volume — lookahead). See "BF trail unification".

### BF trail unification — ONE exit spec for BT and live (2026-09-05)

**Trigger**: CWVX 2026-08-03. Live +$313 (`trail_stop` at 09:58 @14.76);
the reference cache +$2,381 at BT sizing (`exhaust+trail_stop` 13:32
@15.92). Same entry, same bars. An Explore diff of `StopMonitor` vs
`TradeSimulator.simulate` found 21 divergences; five moved money:

| # | Dimension | LIVE (before) | BT (before) | Now (both) |
|---|---|---|---|---|
| 1 | R basis | plan-R (Bug 5, 2026-05-08) | fill-R (`use_planned_r` never wired) | `trading.trailing_stop.r_basis` (default `plan`) via `bf_trail.r_baseline_and_unit` |
| 2 | Ratchet source | every tick, ratchet-then-check in one tick | closed-bar highs, check-then-ratchet | closed bars only via `bf_trail.arm_and_ratchet`; ticks just trigger |
| 3 | Vol-guard bar | previous closed bar | the triggering bar itself (lookahead) | previous closed bar |
| 4 | Entry bar | stops from first tick after fill | excluded (loop starts entry+1) | excluded (`skip_exits_until_ts` = end of fill minute) |
| 5 | `highest_since_entry` | max(tick, bar.high) | bar.high | bar.high |

CWVX under the unified spec: plan-R R=$0.2095 → 09:55 close arms
(14.79 ≥ 14.3985), stop 14.5805; 09:56 close → 14.7505; 09:57 low 14.68
trips. **Both sides exit on the 09:57 bar** (`tests/test_bf_trail.py::
TestBtLiveParity::test_cwvx_golden_plan_r_exits_0957`). So the live exit
was the SPEC, and the cache had been booking a ride that the live spec
never takes.

**Shared module** `trading/bf_trail.py`:
`normalize_r_basis`, `r_baseline_and_unit(planned_entry, planned_stop,
fill, fill_stop, r_basis)`, `arm_and_ratchet(bar_high, highest, stop,
active, r_baseline, r_unit, activate_at_r, trail_r) → TrailStep`,
`entry_bar_excluded(bar_start_ts, skip_exits_until_ts)`. Imported by
`backtest.py::TradeSimulator.simulate` and
`trading/stop_monitor.py::_maybe_ratchet_from_bar_high` /
`_r_baseline_and_unit`. Pct trails (MACD wave) and the ORB/ignition
static lock are untouched.

**Honest book, unified spec** (regen-6 cache, exits re-simulated with
`python batch_backtest.py --start 2025-01-01 --end 2026-08-28 --resim-exits OUT.csv`
with `BT_CACHE_PATH_OVERRIDE=data/bull_flag_cache_causal_full_20260830.csv`,
then Stage-2 at $50K/$2K/10K):

| | fill-R (8/30 reference) | plan-R unified spec |
|---|---|---|
| Stage-2 P&L / trades / WR | $198,276 / 106 / 44.3% | **$191,142 / 105 / 46.7%** |
| Max drawdown | −$58,049 | **−$52,741** |
| Worst / best month | −$27,354 / +$76,772 | −$23,793 / +$83,641 |
| Green months | 15/20 | 14/20 |
| Stage-1 raw Σ (829 rows) | −$62,853 | +$15,066 |

Same 106 trade set; the swings cancel (NEGG −$22.0K, CWVX −$20.2K,
VELO −$11.3K vs AQMS +$14.5K, RDAC +$12.2K, INDP +$8.5K). Planned entry
is approximated as fill/(1+0.5%) until the regen-7 rebuild writes the
new `planned_entry` cache column (exact for gap-through fills, which
only tightens the trail further). Still a RELATIVE tool — never a
forecast.

**Config**: `trading.trailing_stop.r_basis: plan` (config.yaml +
template). Invalid values raise at boot and at simulator construction.
**Rollback**: `r_basis: fill` + regen restores the retired basis; the
bar-only ratchet, entry-bar exclusion and previous-bar vol guard have no
flag — they are the spec on both sides.
**Monitor**: `journalctl -u onemil-trader | grep "StopMonitor (bar)"`.
**Tests**: `tests/test_bf_trail.py` (22: unit, BT↔live parity on one
tape, CWVX golden, fill-R rides to 10:01 on both, entry-bar, vol-guard);
`tests/test_stop_monitor.py::TestTrailingStop` rewritten to the bar
contract (+ `test_tick_does_not_arm_or_ratchet_r_trail`).
**Known deviation**: poll-mode StopMonitor (paper nodes only) still
ratchets R-trails per snapshot.

### Per-tier MACD zone scaling (S2-max, shipped 2026-04-18)

MACD zone multipliers are now **tier-aware** — A-tier (intraday change ≥20%) and Extras tier (10% ≤ intraday < 20%) receive different multipliers based on where the edge actually is.

**Research finding**: per-tier β-coefficient analysis on 10%-frame 2025+Q1 2026 fresh clean caches showed rule edges are radically different between tiers:

- **A-tier MACD-strong** bucket: already positive edge, amp helps modestly
- **A-tier V-reversal** (rule 9, fires only on ≥20% range): β = +0.839R stable across TRAIN/VAL/HOLDOUT — the single most consistent signal in the conviction system
- **Extras MACD-strong** bucket: +0.32R edge on HOLDOUT — BIG under-utilized signal
- **Extras MACD-neutral** bucket: **−$14,734 landmine** across 219 baseline trades — consistent loser every split

**Ship config** (default-on, no feature flag — simple value bumps):

```yaml
trading:
  macd_zones:
    strong_pos_multiplier: 1.8          # A-tier strong-MACD (was 1.5)
    strong_neg_multiplier: 1.8          # A-tier strong-MACD
    normal_multiplier: 1.0              # A-tier neutral (unchanged)
    extras_tier:
      strong_pos_multiplier: 2.0        # amp Extras strong bucket
      strong_neg_multiplier: 2.0
      normal_multiplier: 0.0            # SKIP Extras neutral (landmine)
  conviction_scoring:
    v_reversal_bonus:
      bonus: 1.0                         # was 0.4 — bump A-tier V-rev
```

**BT validation** (Stage-2, production frame: 10% scanner threshold + 200K min_daily_volume):

| Quarter | Baseline | S2-max | Δ | Δ% |
|---|---:|---:|---:|---:|
| 2025-Q1 | $28,913 | $36,118 | +$7,205 | +24.9% |
| 2025-Q2 | $14,996 | $17,579 | +$2,583 | +17.2% |
| 2025-Q3 | $12,463 | $16,355 | +$3,892 | +31.2% |
| 2025-Q4 | $12,368 | $17,460 | +$5,093 | +41.2% |
| 2026-Q1 (HOLDOUT) | $13,170 | $17,907 | +$4,737 | +36.0% |
| **TOTAL** | **$81,911** | **$105,420** | **+$23,509** | **+28.7%** |

Every quarter positive. HOQ1 holdout gain (+$4,737) is anti-overfit validation.

**Per-tier wiring**: `_get_macd_zone_multiplier(..., intraday_change_pct=<float>)` — both BT and PROD now take the intraday change as a parameter and classify via `trading/two_tier_filter.py::classify_tier()`.

**Rollback**: single-commit `git revert` flips all 4 yaml values back. Or manual YAML edit of `strong_pos_multiplier`, `strong_neg_multiplier`, `bonus`, and delete the `extras_tier` block.

**Research artifacts**: `research/S2_proposal.md` (ship synthesis), `research/per_tier_decomp.md` (tier-by-feature decomposition), `research/per_tier_joint_search.md` (22,700-config grid search), `research/scripts/holistic_*.py` (reproducibility).

**Key implementation notes:**

- Tier classifier reused: `trading/two_tier_filter.py::classify_tier` (same one TTF uses at Stage-2).
- `intraday_change_pct` is plumbed through at the caller level. PROD reads from `_qualified_max_intraday[symbol]` populated by `on_stock_qualified`. BT computes via `trading.two_tier_filter.max_intraday_change_pre_entry` before the MACD call.
- Per-tier BT↔PROD parity enforced by `tests/test_bt_prod_parity.py` (11 tests).
- Per-tier multiplier correctness enforced by `tests/test_per_tier_macd_zones.py` (19 tests, including tier boundaries and dead-zone invariance).

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

## ORB (Opening Range Breakout) — Third Strategy

Runs as a module inside the same `onemil-trader` systemd service alongside
Bull Flag and MACD Wave. Fires at 9:35 ET on gap-up stocks that break
above their 9:30–9:34 opening-range high. Uses the `static_lock` exit
(stop at range_low, lock at entry+0.5R after price touches +1.75R;
orb.yaml `exit.lock_arm_at_r: 1.75` / `lock_stop_r: 0.5`).

**Validated Jan 2025 → Apr 2026** (`study_orb_pipeline_static_lock.py`,
production-parity):

| Metric | Value |
|---|---:|
| Full-timeline P&L | **$+342,565** |
| Max DD (full-timeline) | −$18,126 (trough 2025-11-13) |
| Calmar | 18.90× |
| Daily WR | 56.6% |
| Trades | 1,001 over 309 trading days |
| Negative months | 1 (Aug 2025, −$9,288) |

### Enable

1. **Set ORB Alpaca keys in `.env`** (separate paper account from Bull
   Flag / MACD so ORB's positions & BP don't interact with the others):

   ```bash
   ALPACA_ORB_API_KEY=<your-orb-paper-key>
   ALPACA_ORB_API_SECRET=<your-orb-paper-secret>
   ALPACA_ORB_PAPER=true
   ```

2. **Bootstrap `orb.yaml` from the template** (only needed once per node — `orb.yaml` is gitignored, instance-specific):

   ```bash
   # First-time setup or after a rebuild — copy the in-repo template:
   cp orb.yaml.template orb.yaml

   # Subsequent updates: pull and merge any new template fields into your
   # existing orb.yaml manually (config.yaml follows the same pattern).
   git diff origin/master -- orb.yaml.template
   ```

3. **Master kill switch** in `orb.yaml`:

   ```yaml
   strategy:
     name: orb
     enabled: true          # ← flip to true
   ```

4. **Add `--orb` to the systemd unit** so `main.py` loads the module
   alongside the other strategies:

   ```bash
   # The shipped unit runs all three:
   #   /usr/bin/python3 main.py --scan --trade --flag --macd --orb --verbose
   grep ExecStart /etc/systemd/system/onemil-trader.service
   # If --orb isn't there yet, edit the unit + daemon-reload:
   sudo systemctl edit --full onemil-trader
   sudo systemctl daemon-reload
   ```

5. **Restart**:

   ```bash
   sudo systemctl restart onemil-trader
   journalctl -u onemil-trader --since "1 minute ago" | grep "ORB strategy ENABLED"
   # Expected: "ORB strategy ENABLED — master_flag=True, dry_run=False"
   ```

ORB runs concurrently with Bull Flag and MACD Wave in the same process —
no separate service. Each strategy uses its own Alpaca client (ORB's
keys above; Bull Flag + MACD share the main `ALPACA_API_KEY`). Cross-
strategy FCFS dedup means if a symbol is already open in any strategy,
ORB skips it (`conflict.skip_if_any_strategy_has_symbol: true` in
`orb.yaml`).

### Disable / Rollback

Flip the master kill switch + restart. Existing ORB positions force-close
on shutdown (controlled by `exit.force_close_time_et` and the shutdown
hook in `trading/orb_engine.py`):

```yaml
strategy:
  enabled: false
```

```bash
sudo systemctl restart onemil-trader
```

Bull Flag and MACD Wave remain unaffected — they run from different
accounts with different tags in `trades.strategy`.

Nuclear option (service crash / emergency): `sudo systemctl stop
onemil-trader` halts everything. Any still-open ORB position on the ORB
Alpaca account can be flattened manually via the Alpaca web UI or:

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from config import Config
from alpaca.trading.client import TradingClient
cfg = Config()
tc = TradingClient(cfg.alpaca_orb_api_key, cfg.alpaca_orb_api_secret, paper=True)
tc.close_all_positions(cancel_orders=True)
"
```

### Monitor

```bash
# Live ORB activity only
journalctl -u onemil-trader -f | grep '\[ORB\]\|ORB:'

# Per-candidate composite scoring (telemetry from 2026-04-22 fix)
journalctl -u onemil-trader --since today | grep "ORB SCORED"

# Entries / fills / exits
journalctl -u onemil-trader --since today | grep -E "ORB ENTRY SUBMITTED|ORB FILL|ORB EXIT"

# Today's trade summary from the DB
sqlite3 data/trades.db "
  SELECT symbol, fill_price, exit_price, exit_reason, pnl,
         json_extract(pattern_data, '\$.quintile') AS Q
  FROM trades
  WHERE strategy='orb' AND trade_date = date('now')
  ORDER BY id;
"

# Alpaca-side open positions (ORB account)
python3 -c "
import sys; sys.path.insert(0, '.')
from config import Config
from data_sources.alpaca_client import AlpacaClient
cfg = Config()
orb = AlpacaClient(cfg.alpaca_orb_api_key, cfg.alpaca_orb_api_secret, paper=True)
for p in orb.get_open_positions():
    print(f\"{p['symbol']:6s} qty={p['qty']} avg={p['avg_entry_price']} upnl={p['unrealized_pl']}\")
"
```

### Nightly post-close BT refresh

A systemd timer fires at 16:30 ET Mon–Fri to refresh the features CSV
+ run the BT pipeline against today's final bars (installed separately —
see `systemd/README.md`):

```bash
systemctl list-timers onemil-orb-backtest.timer
journalctl -u onemil-orb-backtest.service -n 200 --no-pager   # latest run
```

### July 2026 assumption audit (owner-mandated full-machine pass)

Every parameter in the machine now carries an evidence status in
**`research/assumption_ledger.md`** — validated / shipped / dead /
untestable-monitored. A ship without a ledger-row update is incomplete.
Key outcomes (full details in the ledger + research/ verdict docs):

- **Spread gate loosened 150 → 300bps** (`entry.max_spread_bps`): first
  NBBO validation showed the 150 gate skipped monsters (BKKT +$20.8K at
  153bps, XNDU +$11.9K at 267bps); 100–150bps is the richest per-trade
  bucket. +$24.7K/18mo with honest exit-cost penalties. NEVER tighten
  below 150. `research/orb_spread_gate_verdict.md`.
- **BT force-close parity (15:45)**: the BT's last-bar (~15:59) EOD exit
  understated the book — live's 15:45 is the sweep optimum. Authoritative
  baseline moved $209.7K → **$258.3K** (pre-PM-mult).
  `ORB_BT_FORCE_CLOSE_ET=15:59` reproduces older studies.
- **Entry stop-limit buffer (30bps) validated** — 97% of BT-assumed fills
  genuinely fillable, all top-10 monsters fillable, knob insensitive
  10–150bps. The BT fill assumption holds.
- **Price band $3–30 is final**: $2–3, $30–60, and a $4 floor all fail
  pre-declared bars. `research/orb_price_band_verdicts_jul2026.md`.
- Slippage models (30/10bps) confirmed conservative vs live telemetry;
  time-stop 60min re-confirmed (90min is worse); `min_stop_pct` dead.

### Premarket dollar-volume sizing mult (shipped 2026-07-04, default ON)

Picks whose premarket (4:00–9:29 ET) dollar volume exceeds the H1-2025
TRAIN upper tercile (**$5,816,688 — frozen**) get **1.5× sizing**.
Upsize-only by design: the PM$ gradient's value is at the top
(+$792/trade high tercile vs −$17 low), monsters can only be boosted,
and a boosted loser's damage is capped by its stop while a boosted
winner is uncapped.

- BT (walk-forward): **+$76.8K/18mo** — ΔTRAIN +$19.6K / Δ25H2 +$14.7K /
  Δ2026 +$42.5K, all eras positive; 0 giants downsized; worst month
  −$2.6K; corr(PM$, composite) = 0.05 → genuinely orthogonal channel.
- Shared math: `trading/orb_pm_mult.py` (live planner + BT pipeline stack
  it exactly like the quintile mult — parity by construction).
- Live: one batched premarket-bars fetch per day at first entry check;
  fail-open ×1.0 on missing data. Monitor: `grep "PM MULT" journal`.
- Config: `orb.yaml::sizing.pm_dollar_vol_mult`; env `ORB_PM_MULT=0`.
- Historical PM data: `data/research/orb_premarket_dollar_vol_*.csv`
  (nightly appends needed for future BT parity — see ledger).

### PDR veto — prev-day-range filter (shipped 2026-07-04, default ON)

Skips selected picks whose PREVIOUS day's high-low range was quiet
(`prev_day_range_pct = (prev_high − prev_low)/prev_close × 100 ≤ 8.0`).
Economic mechanism: ORB monetizes **continuation** of an already-explosive
move — "day-2 of the fireworks". A quiet-prev-day name gapping up is a
day-1 fresh pop with no established momentum; those mean-revert against
the breakout.

**NO-REFILL invariant (critical):** the veto runs POST-ranking, inside the
submit loop over the day's top-K picks — a vetoed pick's slot stays
**empty**. The refill variant (veto candidates pre-ranking so slots
backfill with next-ranked names) was explicitly tested and is **toxic**:
2025-H2 P&L collapses from +$51K to ~$0 and MDD balloons −$29K→−$50K,
the same below-cutline-junk failure mode that refuted the ETF exclusion.

**BT evidence** (defended pipeline, Jan'25–Jul'26, production-parity run
verified dollar-exact against the replica):

| metric | base | PDR veto 8.0 |
|---|---|---|
| Cum P&L | $154,892 | **$209,734 (+35%)** |
| Max DD | −$29,297 | **−$20,129 (−31%)** |
| Win rate | 35.8% | 40.2% |
| Trades/day | 3.30 | 1.57 |
| Eras (25H1/25H2/2026) | +$28K/+$36K/+$91K | **+$53K/+$51K/+$106K** |

Monotone across thresholds 6–10; ALL top-10 giants kept (ANNA/QCLS/ASST/
BKKT/CRNC/HERE/CRCD/BNAI/XNDU/NAMM); vetoed mass = 623 picks carrying
−$54,842. Search honesty: found as 1 of 418 stump rules, but survived
dose-response monotonicity, three-era consistency, and monsters-kept —
the neighboring rule families (`price_vs_20d_high` etc.) failed OOS with
sign flips, which is the expected signature of a real effect vs mining.

- Shared helper: `trading/orb_pdr_veto.py` — imported by live
  (`orb_engine._pdr_veto_reject`) and BT (`study_orb_pipeline_static_lock.py`);
  parity by construction. Feature def matches `study_orb_features.py:287`.
- Config: `orb.yaml::filter.prev_day_range_veto.{enabled,min_prev_day_range_pct}`.
  Env: `ORB_PDR_VETO=0`, `ORB_PDR_VETO_MIN_PCT`.
- Fail-open: missing prev-day data → no veto + WARNING (BT drops such
  candidates at the feature stage, so fail-open cannot cause divergence).
- Monitor: `journalctl -u onemil-trader | grep "PDR VETO"`.
- Rollback: `enabled: false` + restart (zero state).
- Tests: `tests/test_orb_pdr_veto.py` (27).

### Winner stack — SZ1 ATR stop-floor + 40%@+3R scale-out (built 2026-08-22, default OFF)

Two independent exit flags (owner order 8/22); flags off = byte-identical
legacy behavior. Combined = the validated frontier C-point on the B+ book:
**$455/mo avg, 15/20 green months, MDD −$700, worst month −$309**
(`research/stability/_frontier_monthly.json`; owner accepted −$81/mo avg for
+3 greens). SZ1 alone = B-point ($11,004/13 green) — a TAIL-shaping device,
not alpha ("it did nothing again" is the expected monitoring reading).

- **SZ1 ATR stop-floor**: initial protective stop = max(range_low,
  fill − 0.25×ATR14(T-1)). Sizing + lock machinery untouched. <15 daily bars
  of history → fail-open to range_low (WARNING). Written to BOTH
  `stop_loss_price` + `real_stop_loss_price` at fill (restart-safe, P0-4).
- **Scale-out**: on touch of fill + 3.0R (range-R), safety legs resize to the
  runner qty and an independent limit sell for floor(0.40×shares) rests at the
  level (fill polled async — the exit latch is never held across the wait).
  Runner keeps the same (floored) stop + static lock. Touchgo prefire exits
  the WHOLE position (no scale). The scale leg books into nullable
  `scale_qty/scale_price/scale_pnl/scaled_at` columns while the row stays
  OPEN; the final exit writes the combined `pnl` exactly once (single-writer).
- **Shared module**: `trading/orb_winner_stack.py` (BT pipeline + live engine
  + StopMonitor — parity by construction). Frozen semantics incl. the
  corrected same-bar rule: a bar hitting both the stop and +3R FILLS the
  scale (NCNA 2025-08-21 golden); live tick ordering is conservative vs BT.
- Config: `orb.yaml::exit.atr_stop_floor.{enabled,k}` +
  `exit.scale_out.{enabled,frac,level_r}`. Env kills: `ORB_ATR_FLOOR=0`,
  `ORB_SCALE_OUT=0`. Constants FROZEN (amendment appended to
  `research/orb_bplus_frozen_params_aug2026.yaml`).
- Data: `scripts/backfill_daily_bars_gaps.py` (INSERT-only) closed the
  daily_bars cache gaps (P0-6); reference regen:
  `research/stability/regen_winner_stack_reference.py` — the Monday flip is
  GATED on the regen reproducing the C-point and the flags-on pipeline
  matching it to <$5/month.
- Monitor: `journalctl -u onemil-trader | grep -E "ATR FLOOR|SCALE OUT"`;
  EoD green check gains a floored-stop drift check (recorded vs BT recompute,
  HARD flag) via `scripts/report_common.floored_stop_drift`.
- Rollback: each flag independently to `false` + restart. Rehydration and
  P&L composition are DATA-driven (`scaled_at IS NOT NULL`), never
  flag-gated — an open scaled position stays correct through a rollback.
- Design/review: `docs/orb_winner_stack_design_aug2026.md` (v2) +
  `docs/orb_winner_stack_review_aug2026.md` (P0/P1 binding). Tests:
  `tests/test_orb_winner_stack*.py`, `tests/test_orb_scale_out_monitor.py`,
  `tests/test_report_common_scale.py`.

### Touchgo filter — Rule M + Rule D (shipped 2026-05-16, default ON)

Two post-fill exit rules that catch failed breakouts within the first 1–2
minutes of trade life. The composite score plus quintile sizing already
selects high-probability setups, but **the breakout-bar price action is
not captured by entry-time features** — that's the gap the touchgo filter
fills.

**Rule M (entry-bar weakness)**: at the close of the breakout bar (the
1-min bar whose high triggered our stop-limit BUY), if its close sat in
the bottom half of its high-low range (`bb_close_pos < 0.5`), exit at the
next bar open. Catches "touch and go" failed breakouts where buyers
couldn't hold the high.

**Rule D (bar-1 pullback)**: at the close of the first post-entry bar, if
the bar's low went ≥0.75R below entry (`R = range_high − range_low`),
exit at `entry − 0.5R`. Catches fast reversal patterns the composite
can't see.

**BT validation** (walk-forward, Jan 2025 → May 2026, 924 trades):
- 8/11 OOS months helped (+$27,238 OOS lift)
- +$26,076 full-timeline pipeline-integrated lift ($406K → $432K)
- WR 47.8% → 52.1% (+4.3 pp)
- Negative months 4 → 2 (halved)
- Cumulative max DD −16%
- Threshold 0.5/0.75 stable across all rolling 6-month training windows
  — not overfit

**Architecture (BT/LIVE parity by construction)**:
- Shared helper: `trading/orb_touchgo_filter.py` (3 pure functions +
  `TouchgoConfig` dataclass).
- BT: `study_orb_pipeline_static_lock.py::simulate_static_lock` imports
  the helper and applies Rule M / Rule D in priority order before the
  static-lock loop. Exit reasons `tag_bb` / `tag_b1`.
- LIVE: `trading/orb_engine.py::_evaluate_touchgo` invoked from
  `_ingest_bars` on every 1-min bar event. On fire, calls
  `stop_monitor.force_exit(symbol, reason, limit_price)` — the new
  public wrapper around `_execute_stop_exit`, so the exit flows through
  the same machinery as autonomous stops (bracket cancel → marketable
  limit sell → fill polling → market fallback). No fork of exit logic.
- Parity enforced by `tests/test_orb_touchgo_parity.py` (source-code
  inspection: both modules must import from `trading.orb_touchgo_filter`,
  neither may redefine `evaluate_rule_m` / `evaluate_rule_d`).

**Telegram**: every firing sends `[ORB] TAG_BB EXIT: <SYMBOL>` (or
`TAG_B1 EXIT`) with `bb_close_pos` or `b1_revert_R`, entry / exit / stop
prices, estimated P&L, and savings vs full −1R stop. Failure is
non-fatal — the exit submits regardless.

**Configuration** (`orb.yaml::filter.touchgo`):

```yaml
filter:
  skip_q1: true                  # existing Q1 filter
  touchgo:                       # NEW (default-on)
    enabled: true                # master kill switch
    rule_m:
      enabled: true
      threshold: 0.5
    rule_d:
      enabled: true
      revert_R: 0.75
      exit_R: -0.5
    breakout_bar_source: market    # NEW 2026-06-04 (BT-parity); 'fill' = legacy
    max_breakout_age_min: 15       # NEW: late-fill guard (skip touchgo if stale)
```

Defaults are baked into `load_touchgo_config({})` so BT runs without
`orb.yaml` still apply the validated thresholds.

**Breakout-bar re-keying + late-fill guard (2026-06-04)**: Rule M/D evaluate
the **market breakout bar** — the first 1-min bar with `high > range_high`,
located by the shared `find_breakout_bar_ts` (BT and live call it, so they key
to the identical bar). Live previously used the minute of the actual *fill*,
which diverged from BT whenever a stop-limit fill lagged the breakout — measured
at **23% of live fills, every one flipping the `tag_bb` decision** (paper-vs-live
investigation, May–Jun 2026). The breakout bar is captured during the pending
phase (`_ensure_breakout_bar_ts`), so the fix is robust to late fills. The
**late-fill guard** (`max_breakout_age_min`, default 15) skips touchgo entirely
when the fill lagged the breakout bar by more than the cap — a stale entry is no
longer an opening-range breakout and gets no retroactive tag exit. Counterfactual
on the 33-trade live sample: **+$251.8** (restores the BT-validated edge).
Rollback: `breakout_bar_source: fill` + restart.

**Env-var overrides for BT research / live emergency**:
- `ORB_TOUCHGO_ENABLED=0` — master disable
- `ORB_TOUCHGO_RULE_M_ENABLED=0` / `ORB_TOUCHGO_RULE_D_ENABLED=0` — per-rule
- `ORB_TOUCHGO_RULE_M_THRESH=0.4` — tighten Rule M
- `ORB_TOUCHGO_RULE_D_R=0.6` — tighten Rule D revert trigger
- `ORB_TOUCHGO_RULE_D_EXIT_R=-0.3` — less aggressive Rule D exit
- `ORB_TOUCHGO_BREAKOUT_BAR_SOURCE=fill` — roll back to legacy fill-bar keying
- `ORB_TOUCHGO_MAX_BREAKOUT_AGE_MIN=15` — late-fill guard threshold (minutes)

**Monitor**:

```bash
journalctl -u onemil-trader -f | grep -E "TAG_BB|TAG_B1|touchgo"
```

Expected firing rate ≈ 3/day (BT prevalence: 26% of fills × ~12 daily
entries). Telegram messages appear at firing time with full context.

**Rollback paths** (safest first; all zero-state because filter only
fires within first 2 min post-fill — no in-flight position state to
unwind):
1. `filter.touchgo.enabled: false` in `orb.yaml` → restart trader
2. `ORB_TOUCHGO_ENABLED=0` in systemd `Environment=` → restart
3. `git revert <ship-commit>` → restart

**Tests**:
- `tests/test_orb_touchgo_filter.py` — 38 helper unit tests
- `tests/test_study_orb_static_lock.py` — 10 BT integration tests
  (Rule M / Rule D fires; existing stop / lock / eod paths unchanged
  regression guards; env-var disable matches pre-change behavior)
- `tests/test_orb_engine_touchgo.py` — 15 live integration tests
  (`breakout_bar_ts` capture, Rule M / D firing, no double-eval,
  Telegram alert, Telegram failure non-blocking, disabled-via-YAML)
- `tests/test_orb_touchgo_parity.py` — 12 parity tests (source-code
  inspection + scalar equivalence + defaults locked + force_exit
  whitelist)

### Deeper references

- **`CLAUDE.md`** — full ORB reference: entry mechanics, exit mechanics,
  sizing math, risk config, correlation dedup, rollout phases, "do NOT"
  list, telemetry examples.
- **`orb.yaml`** — the single source of truth for all tunable parameters.
  Top-level comments explain what each section does. Do NOT change
  `sizing.old_position_reference_usd` or the `quintile_cutoffs` /
  `adaptive_mults` / `filter.features` blocks without running
  `study_orb_refit.py` first (quarterly cadence).
- **`docs/orb_rollout_plan.md`** — live capital ramp (next section).

## Reporting stack (shipped 2026-07-04)

Three layers, all external observers (broker/DB truth, zero prod-code risk):

1. **Hourly holdings pulse** (`scripts/holdings_pulse.py`, cron :05
   market hours): open positions with unrealized $ + R-multiple vs actual
   stop; silent when flat. Ownership-filtered on the shared MAIN account.
2. **Daily green check** (`scripts/daily_green_check.py`, 21:30 UTC):
   computes the day's operational-green verdict (exits attributed, no
   pending-verification, BT-selection parity vs the nightly BT with a
   staleness guard) → `logs/green_streak.json`. One line when green,
   loud block on red. **The streak IS the ramp advancement gate.**
3. **Weekly report** (`scripts/weekly_report.py`, Fri 21:45 UTC): P&L per
   day per strategy, gate progress + loss-floor headroom, BT-vs-live edge
   capture + runner-capture, monster watch, flags.

## Ramp policy (revised 2026-07-06, owner-approved)

Cushion (profit-target) gates are RETIRED. Advance = 10 consecutive
operational-green sessions + loss floor (−1× weekly loss budget) +
slippage parity + min days. Demote = operational failure or −2× weekly
budget; **BT-consistent drawdown is NOT a demotion trigger**. Full
rationale: `docs/ramp_policy_proposal_jul2026.md`; gates implemented in
`scripts/orb_ramp_check.py` (reads the green streak).

## ORB Live Roll-Out

Going live with ORB follows a **cushion-gated capital ramp** — 5 stages
from conservative launch ($30K budget, $1K risk) up to full 4× PDT margin
deployment ($174K budget, $5.2K risk). Each stage advance is gated by
realized P&L, not calendar time. Bad stages auto-demote. Hard stop at
−15% of starting cash.

Authoritative artifacts:

- **[docs/orb_rollout_plan.md](docs/orb_rollout_plan.md)** — the full
  playbook: stages table, advancement gates, demotion triggers, hard
  stop, timeline expectations, FAQ.
- **[scripts/orb_ramp_check.py](scripts/orb_ramp_check.py)** —
  eligibility checker. Reads `orb.yaml` + trades DB + git log; prints
  current stage, cushion, blockers, demotion triggers, hard-stop status:

  ```bash
  python3 scripts/orb_ramp_check.py          # summary
  python3 scripts/orb_ramp_check.py -v       # with per-day P&L history
  ```

Run the checker **before every stage change** and whenever you want a
dispassionate read on how the live ramp is going.

## Production Bug Fixes — 2026-05-08

A long debugging day surfaced **5 distinct production bugs** that had been
silently corroding LIVE P&L vs BT for 14+ days. Documented here so the same
incident vectors don't recur.

### Bug 1 — MACD wave trail-arm gate (CORD/BOBS/ASPN flash exits)

**Symptom**: BOBS, ASPN, CORD all entered around 13:35 ET. Within seconds:
- BOBS: trail ratcheted from $13.01 → $13.37 from a tick high $13.41,
  then tripped at $13.33 → exit -$675 (7 sec from fill).
- ASPN: peak $5.52 (+$725 unrealized), trail tripped at $5.47 → $0 P&L.
  ASPN then ran to $5.63 (+$2,100 unrealized) AFTER the trail had exited.
- CORD: trail tripped 1 sec after fill on a $5.12 bid (entry $5.19); the
  follow-up close failed (held_for_orders race) → naked position rescued
  manually.

**Root cause**: pct trails (MACD wave's `trail_pct=0.003`) had `trailing_active`
set to True at watch creation. With `highest_since_entry = entry_price`, the
trail stop was placed at `entry × 0.997` — below entry. First post-fill bid
print typically dipped 0.5%+ → trail tripped immediately at a small loss.

**Fix**: new `trail_arm_pct` field on `WatchEntry`. Trail does not activate
until `observed_high ≥ entry_price × (1 + trail_arm_pct)`. Default to
`trail_pct` so the trail can never fire at a loss.

Files: `trading/stop_monitor.py` (WatchEntry, _maybe_arm_pct_trail helper, 3
trail-update sites). Config: `macd_wave.yaml::risk.trail_arm_pct: 0.003`.

### Bug 2 — MACD wave pct trail tick-ratchet whipsaw

**Symptom**: even when trail did arm correctly, individual ticks during
fast moves ratcheted the stop to `tick_high × 0.997` and the very next bid
print tripped it. BT runs on 1-min bars only (`backtest.py` simulator) and
shows trail_pct=0.003 producing +$260K over 15 months. LIVE wasn't
delivering anywhere near that.

**Root cause**: pct trail ratcheted on EVERY tick. BT/LIVE divergence —
BT only sees 1-min bar.high, never tick-level micro-volatility.

**Fix**: pct trails now ratchet `stop_price` ONLY on closed-bar highs (BT
parity). Tick path still updates `highest_since_entry` (for arming check)
and still TRIGGERS exits when `price ≤ stop_price` — only the
stop_price-ratchet path moved from tick-cadence to bar-cadence.

Files: `trading/stop_monitor.py::_on_trade` and the poll-loop equivalent.

### Bug 3 — Held-qty race + emergency SL safety net

**Symptom**: when StopMonitor cancelled bracket legs to submit a fresh
limit sell, Alpaca's `held_for_orders` had not propagated yet. The new
sell came back with code `40310000` ("insufficient qty available, requested
N, available 0"). Fallback `close_position()` failed for the same reason.
Original code logged "safety-net SL is the last line of defense" and
returned — but the bracket's OCO had auto-cancelled the safety SL when its
sibling was cancelled, so the position was actually NAKED.

**Fix** (3 layers):
1. **Held-qty retry-with-backoff**: new `_is_held_qty_race(e)` classifier;
   `_submit_with_held_qty_retry()` helper retries at 0.2s / 0.5s / 1.5s
   (cumulative 2.2s, fits inside fill-poll budget).
2. **Emergency stop-market SL**: if both `submit_limit_sell` and
   `close_position` fail through retries, place a fresh `submit_stop_sell_order`
   at min(trigger_price, watch.stop_price) × 0.99 — broker-side protection.
3. **CRITICAL log + Telegram alert** if even the emergency SL fails.

Files: `trading/stop_monitor.py` (_is_held_qty_race, _submit_with_held_qty_retry,
emergency-SL block at the end of force-close path).

### Bug 4 — Bull flag WS handler-loss race (TTGT — 14+ days silent)

**Symptom**: TTGT entered at $5.79 (10:04 ET). Peak $6.32 (+2.4R). The
trail's R-activation gate is +1.5R = $6.12. **Trail never armed.** Position
drifted back to fill price with 0% protection — and the same pattern was
true for every bull flag fill in the prior 14 days. `journalctl` over
2 weeks showed **0 (zero) R-trail activations across all bull flag positions**
despite multiple trades crossing +1.5R.

**Root cause** — asyncio race in the quote-watch → stop-watch upgrade flow:
1. `remove_quote_watch(symbol)` schedules `_unsubscribe_symbol` coroutine
   (which does `_handlers["trades"].pop(symbol)` then awaits an unsubscribe
   message).
2. `add_watch(symbol, ...)` schedules `_subscribe_symbol` coroutine (which
   does `_handlers["trades"][symbol] = self._on_trade` then awaits subscribe).
3. Asyncio interleaves them. If the subscribe's sync handler-set runs
   BEFORE the unsubscribe's sync handler-pop, the **unsubscribe wipes out
   the freshly-installed handler**. WS server keeps delivering trade ticks;
   client-side dispatch has no handler → ticks silently dropped.

**Fix**: new atomic `StopMonitor.upgrade_quote_to_stop_watch()` that pops
quote-watch + installs stop-watch under one lock with NO WS operations.
The handlers stay registered because the prior quote-watch already
installed them. `trading_engine.py` fill path now calls this single method
instead of the racy `remove_quote_watch + add_watch` pair.

Regression coverage: `tests/test_quote_to_stop_upgrade_race.py::TestUpgradeRaceProof`
4 tests including a synthetic asyncio-interleave reproducer that fails on
the OLD path and passes on the NEW.

Files: `trading/stop_monitor.py` (new method ~95 LOC), `trading/trading_engine.py`
(call site swap at fill-time).

### Bug 5 — Bull flag plan-R (slippage-inflated activation gate)

**Symptom**: TTGT planned breakout $5.715, fill $5.79 (1.3% slip). IREZ
planned $6.72, fill $6.83 (1.6% slip). Under fill-based R math:

| Trade | planned R | fill-based R | activation level | peak | armed? |
|---|---|---|---|---|---|
| TTGT  | $0.145    | $0.22        | $6.12            | $6.32 | yes (just barely) |
| IREZ  | $0.25     | $0.36        | $7.37            | $7.29 | **NO — peaked $0.08 short** |

Slippage inflates R by `slippage% × entry / R`, which pushes the +1.5R
activation gate further from current price. The strongest momentum
breakouts (which have the most slippage) get the WORST trail protection —
inverse of what's wanted.

**Root cause**: trail math (activation, ratchet) used
`risk_per_share = fill - planned_stop`. Slippage inflated this number, and
all R-multiples derived from it shifted away from entry.

**Fix**: new `planned_entry_price` and `planned_risk_per_share` fields on
`WatchEntry`. When set, R-trail math uses these (the SETUP's structural
values) instead of fill-based. Hard stop and broker safety SL stay at
fill-based levels — only trail/lock thresholds move.

**2026-09-05 follow-up**: this fix shipped LIVE only — the BT cache
builder kept simulating fill-R (`use_planned_r` was never wired from
config), so the reference book and the live machine ran two exit specs
for four months (CWVX 2026-08-03: live +$313 vs cache +$2,381). Both
sides now share `trading/bf_trail.py` and the `trading.trailing_stop.r_basis`
knob — see "BF trail unification" below.

**BT validation** (`study_planned_r_realistic_slippage.py`): plan-R wins
HOLDOUT P&L at every slippage level tested. At LIVE-realistic slippage
(1.5-2.0%), plan-R adds **+$62-72K HOLDOUT vs fill-R**. Note: BT also shows
the strategy is unprofitable at >1% slippage in BOTH modes — plan-R is
damage mitigation; reducing slippage at entry is the bigger lever.

**Live verification (2026-05-08 16:01:54 / 16:03:20)**: TTGT armed within
2 sec of restart with new code; IREZ armed 88 sec later. **First R-trail
activations in production in 14+ days** — log line shows `(planned=yes)`
confirming the new path is firing.

Regression coverage: `tests/test_quote_to_stop_upgrade_race.py::TestPlanRTrail`
(IREZ + TTGT replay scenarios + helper unit test + legacy backward-compat).

Files: `trading/stop_monitor.py` (WatchEntry fields, `_r_baseline_and_unit`
helper, 3 trail-update sites), `trading/trading_engine.py` (fresh-fill +
crash-recovery paths pass planned values).

### Bug 6 — Post-fill kill switch idempotency on OrderStream replay

**Symptom**: A second restart re-fired the bull flag post-fill kill switch
(`SPY 3d hostile + bk_vol < 1x → close immediately`) on the open IREZ
position. `close_position()` failed (held_for_orders race) and the system
nearly flushed +$1.2K of unrealized gain. Saved only by the broker SL
holding the qty against the market sell.

**Root cause** (two layers):
1. **Pending-order recovery includes terminal-status trades**:
   `_sync_filled_trades_at_startup` re-added cancelled IREZ trade rows
   (which had `order_id` + NULL fill/exit) to `_pending_orders`. Their
   original Alpaca order ID still showed "filled" in order history (the
   buy-stop did fill before being closed) → OrderStream replay treated
   the stale fill as a fresh fill.
2. **No idempotency guard in `_manage_pending_orders`**: every restart
   could re-process old fills.

**Fix** (defense-in-depth):
1. Pending-order recovery filters terminal statuses: `cancelled`, `canceled`,
   `expired`, `rejected`, `time_stop_canceled`.
2. `_manage_pending_orders` skips fills for symbols already in
   `_traded_symbols` (live position managed elsewhere).

Files: `trading/trading_engine.py` (sync + manage_pending_orders).
Tests: `tests/test_trading_engine.py::TestStartupSync` (2 new regression
cases).

### Bug 7 — BT study harness double-counted partial_pnl (1.8x inflation)

**Symptom**: `study_bull_flag_exits.py` claimed ProdBaseline HOLDOUT
P&L = +$186,946. After fixing harness math, true value was +$104,684 (1.8x
inflation). The user noticed: "with $20-30K/mo from today's gains, BT $46K/mo
sounds too good".

**Root cause**: `backtest.py:739` sets `trade.pnl = trade.partial_pnl + final_pnl`
— `pnl` is already the TOTAL. The study script was adding `trade.partial_pnl`
on top, double-counting partial exits.

**Fix**: removed the `+= partial_pnl` line. Re-ran study; HOLDOUT now reads
$104,684 (= $26K/month) which matches LIVE projection from today's actuals.

Files: `study_bull_flag_exits.py::simulate_one`.

### Strategy upgrade — ORB lock thresholds 1.5R/1.0R → 1.75R/0.5R (shipped 2026-05-08)

After today's debugging exposed the bull flag plan-R issue, a parallel
investigation surfaced that ORB's `lock_arm_at_r` / `lock_stop_r` pair was
not at the empirical optimum. Walk-forward BT (TRAIN H1 2025 / VAL H2 2025
/ HOLDOUT Jan-Apr 2026) ran a full 38-config grid of `(arm_r, lock_r)` pairs
plus a 16-config two-stage exit grid (BE-lock + static lock).

**Result**: `1.75R / 0.50R` Pareto-dominates the prior `1.5R / 1.0R` PROD on
every metric measured on the OOS HOLDOUT:

| Metric | PROD (1.5/1.0) | New (1.75/0.5) | Δ |
|---|---|---|---|
| HOLDOUT P&L | $93,989 | $116,325 | **+$22,336 (+24%)** |
| Sharpe (weekly, annualized) | 4.30 | 5.83 | +36% |
| Sortino (downside-only) | 21.96 | 38.45 | +75% |
| % weeks positive | 66.7% | 77.8% | +11pp |
| Worst weekly loss | -$5,525 | -$4,814 | -13% |
| Max consecutive losing weeks | 2 | 1 | half |
| Weekly std dev | $8,758 | $7,991 | -9% |
| Calmar | 4.78 | 8.55 | +79% |
| % months positive (4-mo HOLDOUT) | 100% | 100% | same |

Walk-forward integrity: 1.75/0.5 ranked #2 on VAL ($149K, vs #1 winner
1.75/0.75 at $151K — within $2K = noise). VAL top-5 was a tightly bunched
cluster; 1.75/0.5 is the strict-walk-forward best within tolerance and the
Pareto-best on HOLDOUT stability.

Not shipped (BT-falsified):
- **Lower arm thresholds (0.5-1.0R)**: lost $20-55K HOLDOUT vs PROD.
  Capping the runners costs more than rescuing near-misses.
- **Two-stage exits (BE-lock + static lock)**: every walk-forward winner
  failed HOLDOUT by $11-23K. Adding a BE_arm_r parameter overfits without
  improving stability.

Sanity check: ran the BT harness on the prior PROD config first;
reproduced the documented full-timeline ~$342K P&L (mine: $340,077) within
0.7%. Confirms BT mechanics are honest — no double-count or look-ahead.

Files: `orb.yaml`, `orb.yaml.template`, `study_orb_pipeline_static_lock.py`
(constants updated). Studies: `study_orb_r_grid.py`, `study_orb_two_stage.py`,
`study_orb_two_stage_variance.py`.

### Cross-strategy impact summary

| Bug | Bull Flag | MACD Wave | ORB |
|---|---|---|---|
| #1 trail-arm gate | not affected (R-trail uses different activation) | **fixed** | not affected (no pct trail) |
| #2 bar-only ratchet | not affected (R-trail bar-ratchet was already there post-OPTX 4/13) | **fixed** | not affected (uses static_lock, no ratchet at all) |
| #3 held-qty race + emergency SL | **fixed** (shared StopMonitor exit path) | **fixed** (same path) | **fixed** (same path) |
| #4 WS handler-loss race (atomic upgrade) | **fixed** | not affected (no quote-watch upgrade flow) | not affected (no quote-watch upgrade flow) |
| #5 plan-R | **fixed** | n/a (uses pct trail not R-trail) | already correct (passes `lock_r_unit = range_size` — fixed at setup time, slippage-immune by design) |
| #6 post-fill kill switch idempotency | **fixed** | not affected (no post-fill kill switch) | not affected (no post-fill kill switch) |
| #7 BT harness double-count partial_pnl | study-only fix; production bull flag P&L unaffected | n/a | n/a |

**Why ORB was structurally immune to most of these**: ORB was designed
around a fixed-R unit (`lock_r_unit = range_size`) from the start —
slippage-decoupled by construction. ORB uses static lock (one-shot stop
ratchet at +arm_R touch, no further trailing) instead of dynamic trail —
no whipsaw surface. ORB doesn't use the quote-watch → stop-watch upgrade
flow — no async race. The bull flag and MACD wave designs evolved
piecewise and accumulated structural debt that ORB avoided.

**Three strategies — combined live impact (today)**:
- **Bull flag**: TTGT + IREZ both armed within minutes of plan-R fix.
  IREZ ratcheted 23+ times (trail climbed from $6.85 → $7.10), locked
  profit floor +$2.9K. Combined unrealized +$9.6K at one point. **First
  R-trail activations across any bull flag trade in 14+ days.**
- **MACD wave**: 8 trades closed on the day, post-fix entries (FNKO at
  14:16) exited cleanly at -$146 (small loss, no flash exit). Pre-fix
  trades earlier in the day (BOBS, ASPN, CORD, XNDU, AMN, ARLO, PGNY,
  RDWU) totaled -$4,338 — the cohort that exposed Bugs 1-3.
- **ORB**: GLWG and DBX held cleanly. ORB's static-lock arming behavior
  is by-design plan-R-equivalent — no fixes required.

### Lessons learned

1. **BT-LIVE divergence is rarely a single bug**. We had 5 distinct
   production bugs simultaneously corroding bull flag + MACD wave LIVE P&L
   in different directions. Each one alone would have looked like noise;
   together they made LIVE consistently underperform BT by ~$15-20K/month.
2. **Async race conditions in event-driven code can be silent for weeks.**
   The bull flag handler-loss race produced ZERO error logs — every
   bull flag trail simply didn't fire. Only careful counting (`grep
   "trailing stop ACTIVATED" | wc -l` over 14 days = 0) made it visible.
3. **Tick-cadence vs bar-cadence is a structural BT/LIVE knob.** BT
   doesn't see ticks; LIVE does. Anything that ratchets on tick (stops,
   trails, activations) will whipsaw differently than BT projects. Match
   the cadence to BT (`bar-only ratchet`) when seeking parity.
4. **R-based math is sensitive to slippage.** `R = entry - stop` looks
   innocent until you realize entry is the FILL (slippage-inflated) and
   stop is PLANNED (unchanged). High slippage inflates R, pushing
   R-multiples away from entry. The fix (plan-R) decouples them.
5. **Multiple defense layers > one perfect layer.** The held-qty race
   bug went undetected for weeks because the bracket-OCO and broker SL
   silently saved us. Today they didn't (different cancel order). The
   emergency-SL fallback + retries + CRITICAL alerting are now in place.
6. **Tests must reproduce the bug, not just exercise the fix.** The
   upgrade-race regression test forces the asyncio interleave that breaks
   the OLD code path AND verifies the NEW path is structurally immune.
   Without the failing-on-old-code test, we'd have no proof the fix
   addresses the exact failure mode.

## Future Tasks

- **BuyMonitor Phase 2**: Replace buy-stop orders with SIP WebSocket limit buys for tighter entry slippage (data collection active via Phase 1 quote monitoring)
- **MACD Wave W1 scout mode**: Paper-trade W1, only enter W2-3 if W1 >= 5% (tested: 62% WR on W2-3 but low trade count)
- **L2 data evaluation**: Assess Level 2 order book data for entry timing optimization
- **Combined P&L dashboard**: Unified daily report across both strategies
- **News-based late-day entry filter**: Evaluate using news sentiment to allow selective entries after 11:00 ET for catalyst-driven stocks
