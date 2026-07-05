"""
Application configuration module.

Loads environment variables from .env file and YAML config,
providing typed access to configuration values throughout the application.
Singleton pattern ensures one config instance per process.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv
import yaml

logger = logging.getLogger(__name__)

# Singleton instance
_config_instance: Optional['Config'] = None

# Module-level override for yaml path. Set via Config.set_config_path() by
# CLI tools (batch_backtest, variant_runner) that want to load an alternate
# config without touching config.yaml in the project root. Inherited by
# multiprocessing fork() workers so subprocess backtest runs use the same
# override without needing to thread the path through every call site.
#
# NOT an env var — explicit module state. Tools should pass --config on the
# command line and call Config.set_config_path(args.config) once, up front.
_config_path_override: Optional[Path] = None


class Config:
    """Application configuration loaded from environment variables and YAML."""

    REQUIRED_KEYS = [
        "ALPACA_API_KEY",
        "ALPACA_API_SECRET",
    ]

    def __init__(self, env_path: Optional[str] = None, yaml_path: Optional[str] = None):
        """
        Initialize configuration from .env and config.yaml.

        Args:
            env_path: Path to .env file. Defaults to .env in project root.
            yaml_path: Path to config.yaml file. Defaults to config.yaml in project root.

        Raises:
            FileNotFoundError: If .env file is missing
            ValueError: If required environment variables are missing
        """
        project_root = Path(__file__).parent

        # Load .env
        if env_path is None:
            env_path = project_root / ".env"

        if not Path(env_path).exists():
            logger.error(f"Environment file not found: {env_path}")
            raise FileNotFoundError(f"Environment file not found: {env_path}")

        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")

        # Load YAML config. Precedence: explicit yaml_path arg > module override
        # (set via Config.set_config_path) > _ONEMIL_CFG env (worker inheritance)
        # > default project_root/config.yaml.
        if yaml_path is None:
            yaml_path = Config._resolve_config_path()

        self._yaml: Dict[str, Any] = {}
        if Path(yaml_path).exists():
            with open(yaml_path, 'r') as f:
                self._yaml = yaml.safe_load(f) or {}
            logger.info(f"Loaded YAML config from {yaml_path}")
        else:
            logger.warning(f"YAML config not found: {yaml_path}, using defaults")

        self._validate_required_keys()

    def _validate_required_keys(self) -> None:
        """Validate that all required configuration keys are present."""
        missing = [key for key in self.REQUIRED_KEYS if not os.getenv(key)]
        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            raise ValueError(f"Missing required environment variables: {missing}")

    @staticmethod
    def set_config_path(path: Optional[str]) -> None:
        """Set the override config.yaml path for this process and its workers.

        Used by CLI tools (batch_backtest, variant_runner) to redirect all
        Config loads to an alternate YAML file without mutating config.yaml.

        Sets BOTH module state AND an env var. The env var is required so
        that multiprocessing `forkserver` workers (which start cleanly and
        do not inherit module state) still pick up the override via
        os.environ inheritance. Pass None to clear both.

        User-facing interface is still the --config CLI argument; the env
        var is an internal implementation detail for worker inheritance.
        """
        global _config_path_override
        if path:
            resolved = Path(path)
            _config_path_override = resolved
            os.environ["_ONEMIL_CFG"] = str(resolved)
        else:
            _config_path_override = None
            os.environ.pop("_ONEMIL_CFG", None)

    @staticmethod
    def _resolve_config_path() -> Path:
        """Return the active config path. Module state wins, then env, then default."""
        if _config_path_override is not None:
            return _config_path_override
        env_path = os.environ.get("_ONEMIL_CFG")
        if env_path:
            return Path(env_path)
        return Path(__file__).parent / "config.yaml"

    @staticmethod
    def _load_yaml_only() -> dict:
        """Load config.yaml without .env (for backtest use — avoids env pollution).

        Resolves path via module override > env var > default. Env var is set
        internally by set_config_path so multiprocessing workers (forkserver)
        pick up the same override.
        """
        yaml_path = Config._resolve_config_path()
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _get_yaml(self, *keys, default=None):
        """Traverse nested YAML keys, returning default if any key is missing."""
        node = self._yaml
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    # =========================================================================
    # API Keys
    # =========================================================================

    @property
    def alpaca_api_key(self) -> str:
        """Alpaca API key."""
        return os.getenv("ALPACA_API_KEY", "")

    @property
    def alpaca_api_secret(self) -> str:
        """Alpaca API secret."""
        return os.getenv("ALPACA_API_SECRET", "")

    @property
    def alpaca_paper(self) -> bool:
        """Whether to use Alpaca paper trading. Default True for safety."""
        return os.getenv("ALPACA_PAPER", "true").lower() in ("true", "1", "yes")

    @property
    def alpaca_base_url(self) -> str:
        """Alpaca API base URL."""
        return os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")

    # =========================================================================
    # ORB Paper Account (Phase 1 paper validation — separate from main/live)
    # =========================================================================

    @property
    def alpaca_orb_api_key(self) -> str:
        """Alpaca API key for ORB paper account (Phase 1). Empty until creds provided."""
        return os.getenv("ALPACA_ORB_API_KEY", "")

    @property
    def alpaca_orb_api_secret(self) -> str:
        """Alpaca API secret for ORB paper account (Phase 1)."""
        return os.getenv("ALPACA_ORB_API_SECRET", "")

    @property
    def alpaca_orb_paper(self) -> bool:
        """Whether the ORB account is paper (default True for Phase 1)."""
        return os.getenv("ALPACA_ORB_PAPER", "true").lower() in ("true", "1", "yes")

    # =========================================================================
    # Bull Flag Paper Account (created 2026-05-11 — isolates BF from main acct)
    # =========================================================================

    @property
    def alpaca_bf_api_key(self) -> str:
        """Alpaca API key for bull flag paper account. Empty → fall back to main."""
        return os.getenv("ALPACA_BF_API_KEY", "")

    @property
    def alpaca_bf_api_secret(self) -> str:
        """Alpaca API secret for bull flag paper account."""
        return os.getenv("ALPACA_BF_API_SECRET", "")

    @property
    def alpaca_bf_paper(self) -> bool:
        """Whether the bull flag account is paper (default True)."""
        return os.getenv("ALPACA_BF_PAPER", "true").lower() in ("true", "1", "yes")

    @property
    def anthropic_api_key(self) -> str:
        """Anthropic API key (optional — enables LLM news analysis)."""
        return os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def news_analyzer_model(self) -> str:
        """Model ID for LLM news analyzer."""
        return self._get_yaml("news_analyzer", "model", default="claude-haiku-4-5-20251001")

    # =========================================================================
    # Logging
    # =========================================================================

    @property
    def log_level(self) -> str:
        """Logging level from YAML config."""
        return self._get_yaml("log_level", default="INFO")

    # =========================================================================
    # Scanner Thresholds
    # =========================================================================

    @property
    def price_min(self) -> float:
        """Minimum stock price for universe."""
        return float(self._get_yaml("scanner", "price_min", default=2.0))

    @property
    def price_max(self) -> float:
        """Maximum stock price for universe."""
        return float(self._get_yaml("scanner", "price_max", default=30.0))  # 2026-07-05: 20->30

    @property
    def float_max(self) -> int:
        """Maximum float shares for universe."""
        return int(self._get_yaml("scanner", "float_max", default=10_000_000))

    @property
    def gap_pct_min(self) -> float:
        """Minimum pre-market gap percentage."""
        return float(self._get_yaml("scanner", "gap_pct_min", default=2.0))

    @property
    def intraday_change_pct_min(self) -> float:
        """Minimum intraday price change percentage."""
        return float(self._get_yaml("scanner", "intraday_change_pct_min", default=10.0))

    @property
    def relative_volume_min(self) -> float:
        """Minimum relative volume (bucket vol / avg)."""
        return float(self._get_yaml("scanner", "relative_volume_min", default=5.0))

    @property
    def require_news(self) -> bool:
        """Whether to require news catalyst for qualification."""
        return bool(self._get_yaml("scanner", "require_news", default=False))

    @property
    def min_dollar_volume(self) -> float:
        """Minimum daily dollar volume to filter untradeable micro-caps."""
        return float(self._get_yaml("scanner", "min_dollar_volume", default=0))

    @property
    def volume_profile_days(self) -> int:
        """Number of days for volume profile calculation."""
        return int(self._get_yaml("scanner", "volume_profile_days", default=50))

    # =========================================================================
    # Timing
    # =========================================================================

    @property
    def premarket_poll_interval(self) -> int:
        """Pre-market polling interval in seconds."""
        return int(self._get_yaml("timing", "premarket_poll_interval", default=60))

    @property
    def intraday_scan_interval(self) -> int:
        """Intraday scan interval in minutes."""
        return int(self._get_yaml("timing", "intraday_scan_interval", default=15))

    @property
    def premarket_start(self) -> str:
        """Pre-market start time (ET)."""
        return self._get_yaml("timing", "premarket_start", default="04:00")

    @property
    def market_open(self) -> str:
        """Market open time (ET)."""
        return self._get_yaml("timing", "market_open", default="09:30")

    @property
    def market_close(self) -> str:
        """Market close time (ET)."""
        return self._get_yaml("timing", "market_close", default="16:00")

    # =========================================================================
    # Database
    # =========================================================================

    @property
    def db_path(self) -> str:
        """Legacy single SQLite database file path."""
        return self._get_yaml("database", "path", default=None)

    @property
    def cache_db_path(self) -> str:
        """Cache database path (bars, universe, profiles)."""
        return self._get_yaml("database", "cache_path", default=None)

    @property
    def trades_db_path(self) -> str:
        """Trades database path (trades, scan results, summaries)."""
        return self._get_yaml("database", "trades_path", default=None)

    # =========================================================================
    # Float Cache
    # =========================================================================

    @property
    def float_cache_refresh_days(self) -> int:
        """Days before float data is considered stale."""
        return int(self._get_yaml("float_cache", "refresh_days", default=7))

    # =========================================================================
    # Telegram Notifications
    # =========================================================================

    @property
    def telegram_bot_token(self) -> str:
        """Telegram bot token from .env."""
        return os.getenv("TELEGRAM_BOT_TOKEN", "")

    @property
    def telegram_chat_id(self) -> str:
        """Telegram chat ID from .env."""
        return os.getenv("TELEGRAM_CHAT_ID", "")

    @property
    def telegram_enabled(self) -> bool:
        """Whether Telegram notifications are enabled."""
        return bool(self._get_yaml("notifications", "telegram", "enabled", default=True))

    @property
    def telegram_send_on_startup(self) -> bool:
        """Send notification on scanner startup."""
        return bool(self._get_yaml("notifications", "telegram", "send_on_startup", default=True))

    @property
    def telegram_send_on_qualified(self) -> bool:
        """Send notification when stock qualifies."""
        return bool(self._get_yaml("notifications", "telegram", "send_on_qualified", default=True))

    @property
    def telegram_send_on_pattern(self) -> bool:
        """Send notification when pattern detected."""
        return bool(self._get_yaml("notifications", "telegram", "send_on_pattern", default=True))

    @property
    def telegram_send_on_trade(self) -> bool:
        """Send notification when trade submitted."""
        return bool(self._get_yaml("notifications", "telegram", "send_on_trade", default=True))

    @property
    def telegram_send_on_close(self) -> bool:
        """Send notification when position closed."""
        return bool(self._get_yaml("notifications", "telegram", "send_on_close", default=True))

    @property
    def telegram_send_daily_report(self) -> bool:
        """Send end-of-day report."""
        return bool(self._get_yaml("notifications", "telegram", "send_daily_report", default=True))

    # =========================================================================
    # Trading
    # =========================================================================

    @property
    def trading_enabled(self) -> bool:
        """Master kill switch for automated trading."""
        return bool(self._get_yaml("trading", "enabled", default=False))

    @property
    def capital(self) -> float:
        """Trading capital allocation (reference value for scaling)."""
        return float(self._get_yaml("trading", "capital", default=50000))

    @property
    def position_size_dollars(self) -> float:
        """Dollar amount per position."""
        return float(self._get_yaml("trading", "position_size_dollars", default=500))

    @property
    def max_shares(self) -> int:
        """Maximum shares per position."""
        return int(self._get_yaml("trading", "max_shares", default=1000))

    @property
    def max_positions(self) -> int:
        """Maximum concurrent positions."""
        return int(self._get_yaml("trading", "max_positions", default=3))

    @property
    def daily_loss_limit(self) -> float:
        """Daily loss limit in dollars (negative value)."""
        return float(self._get_yaml("trading", "daily_loss_limit", default=-100.0))

    @property
    def max_risk_per_share(self) -> float:
        """Maximum risk per share in dollars (Ross's 20-cent rule)."""
        return float(self._get_yaml("trading", "max_risk_per_share", default=0.20))

    @property
    def min_risk_per_share(self) -> float:
        """Minimum risk per share in dollars — rejects noise stops."""
        return float(self._get_yaml("trading", "min_risk_per_share", default=0.02))

    @property
    def min_risk_reward(self) -> float:
        """Minimum risk/reward ratio."""
        return float(self._get_yaml("trading", "min_risk_reward", default=2.0))

    @property
    def pattern_poll_interval(self) -> int:
        """Pattern detection polling interval in seconds."""
        return int(self._get_yaml("trading", "pattern_poll_interval", default=60))

    @property
    def stop_trading_before_close_min(self) -> int:
        """Minutes before close to stop opening new positions."""
        return int(self._get_yaml("trading", "stop_trading_before_close_min", default=15))

    @property
    def min_pole_candles(self) -> int:
        """Minimum consecutive green candles for bull flag pole."""
        return int(self._get_yaml("trading", "bull_flag", "min_pole_candles", default=3))

    @property
    def min_pole_gain_pct(self) -> float:
        """Minimum pole gain percentage."""
        return float(self._get_yaml("trading", "bull_flag", "min_pole_gain_pct", default=3.0))

    @property
    def max_retracement_pct(self) -> float:
        """Maximum pullback retracement as percentage of pole height."""
        return float(self._get_yaml("trading", "bull_flag", "max_retracement_pct", default=50.0))

    @property
    def max_pullback_candles(self) -> int:
        """Maximum pullback candles before pattern is rejected."""
        return int(self._get_yaml("trading", "bull_flag", "max_pullback_candles", default=10))  # 2026-07-05: 5->10

    @property
    def min_breakout_volume_ratio(self) -> float:
        """Minimum breakout volume relative to pullback average."""
        return float(self._get_yaml("trading", "bull_flag", "min_breakout_volume_ratio", default=1.5))

    @property
    def sizing_mode(self) -> str:
        """Position sizing mode: 'fixed_investment' or 'fixed_risk'."""
        return str(self._get_yaml("trading", "sizing_mode", default="fixed_investment"))

    @property
    def risk_per_trade(self) -> float:
        """Dollar risk budget per trade (fixed_risk mode)."""
        return float(self._get_yaml("trading", "risk_per_trade", default=500.0))

    @property
    def min_risk_pct(self) -> Optional[float]:
        """Min risk as fraction of entry price (e.g., 0.01 = 1%)."""
        val = self._get_yaml("trading", "min_risk_pct", default=None)
        return float(val) if val is not None else None

    @property
    def max_risk_pct(self) -> Optional[float]:
        """Max risk as fraction of entry price (e.g., 0.05 = 5%)."""
        val = self._get_yaml("trading", "max_risk_pct", default=None)
        return float(val) if val is not None else None

    @property
    def max_green_in_flag(self) -> int:
        """Max green candles tolerated inside bull flag pullback."""
        return int(self._get_yaml("trading", "bull_flag", "max_green_in_flag", default=1))

    @property
    def require_macd_positive(self) -> bool:
        """Whether bull flag detector requires positive MACD."""
        return bool(self._get_yaml("trading", "bull_flag", "require_macd_positive", default=False))

    @property
    def max_pole_bars(self) -> int:
        """Max bull flag pole length filter (cache-delta units; 0 = disabled).

        Compared as `(pole_candle_count - 1) > max_pole_bars`, so 3 means
        reject 5+ candle poles. Off-by-one matches `qf_pole_bars` cache
        convention for BT↔live parity.
        """
        return int(self._get_yaml("trading", "bull_flag", "max_pole_bars", default=0))

    # =========================================================================
    # Two-Tier Filter (10% scanner quality gate)
    # =========================================================================

    @property
    def two_tier_filter_enabled(self) -> bool:
        """Whether the two-tier filter is active (BT + live)."""
        return bool(self._get_yaml("trading", "bull_flag", "two_tier_filter",
                                   "enabled", default=False))

    @property
    def v_reversal_bonus_cfg(self) -> dict:
        """Experiment V — V-reversal conviction bonus.

        Rule 9 adds `bonus` to the raw conviction score when a setup matches
        the V-reversal pattern (gap_pct < gap_pct_max, intraday_range >=
        intraday_range_min, pole_gain >= pole_gain_min). Disabled by default.
        """
        cfg = self._get_yaml("trading", "conviction_scoring",
                             "v_reversal_bonus", default={}) or {}
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "bonus": float(cfg.get("bonus", 0.4)),
            "gap_pct_max": float(cfg.get("gap_pct_max", 0.0)),
            "intraday_range_min": float(cfg.get("intraday_range_min", 20.0)),
            "pole_gain_min": float(cfg.get("pole_gain_min", 5.0)),
        }

    @property
    def conviction_marginal_scaling_cfg(self) -> dict:
        """Experiment H — marginal-conviction defensive scaling.

        After min_threshold filter passes, trades with conviction in
        [min_threshold, upper_bound) get their SIZING multiplier scaled by
        scale_factor. Disabled by default (scale_factor effectively 1.0).
        """
        cfg = self._get_yaml("trading", "conviction_scoring",
                             "marginal_scaling", default={}) or {}
        enabled = bool(cfg.get("enabled", False))
        raw_factor = float(cfg.get("scale_factor", 0.5))
        return {
            "enabled": enabled,
            # When disabled, expose factor=1.0 so downstream sizing is a no-op.
            "scale_factor": raw_factor if enabled else 1.0,
            "upper_bound": float(cfg.get("upper_bound", 1.7)),
        }

    @property
    def vol_confirmed_trail_cfg(self) -> dict:
        """Volume-confirmed trail exit config (Experiment D).

        Returns dict with {enabled, min_vol_ratio}. Default disabled. Loaded
        from `trading.trailing_stop.vol_confirmed_exit`. Safe for use as
        kwargs into the shared helper / TradeSimulator / StopMonitor.
        """
        cfg = self._get_yaml("trading", "trailing_stop", "vol_confirmed_exit",
                             default={}) or {}
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "min_vol_ratio": float(cfg.get("min_vol_ratio", 1.0)),
        }

    @property
    def marketable_limit_fallback_cfg(self) -> dict:
        """Buy stop-limit rejection workaround (IREZ+TTGT post-mortem
        2026-05-08; extended 2026-05-14 after KPTI/TRT).

        Alpaca LIVE rejects a buy stop-limit whenever stop_price <= current
        ASK — the order is immediately marketable, not a real stop. (Paper
        Alpaca does not enforce this; that's the parity gap that lost TTGT,
        KPTI, TRT and ~24 other prod orders.) The earlier revision of this
        fix checked the BID, which is the wrong side of the spread — it only
        caught the "whole spread above stop" case and let the straddle case
        through. Two sub-cases, both gated by `enabled`:

          bid >= stop        → breakout fully confirmed → marketable LIMIT
                               buy at limit_price.
          bid < stop <= ask  → spread straddles the breakout level; a native
                               stop is rejected but the breakout is NOT yet
                               confirmed by trades → re-bump the stop to
                               ask + rebump_buffer (still a real stop — only
                               fills on a genuine upward print), provided the
                               bumped stop stays <= limit_price. If the ask
                               has already run past limit_price the breakout
                               is too extended — skip (don't chase).

        See docs/irez_ttgt_paper_vs_prod_divergence.md.
        """
        cfg = self._get_yaml("trading", "marketable_limit_fallback",
                             default={}) or {}
        return {
            "enabled": bool(cfg.get("enabled", True)),
            # Min 0.02 so the bumped stop is strictly > ask after 2-dp
            # rounding (Alpaca requires a buy stop strictly above market).
            "rebump_buffer": max(0.02, float(cfg.get("rebump_buffer", 0.02))),
        }

    @property
    def post_fill_gate_cfg(self) -> dict:
        """Post-fill gate config (IREZ post-mortem 2026-05-08).

        Kills positions immediately after fill if SPY 3d range AND
        breakout volume are BOTH below their thresholds.

        Default thresholds tightened to 0.5 / 0.5 (V1) on 2026-05-08.
        16-mo BT (Jan 2025-May 2026, 310 trades) showed the prior 0.8/1.0
        thresholds destroyed $24,643 of net P&L via a non-discriminating
        winner/loser kill — outcome distribution was 5 winners / 4 losers
        out of 9 kills (~50/50). Walk-forward sign-flipped: gate net-helped
        in train (Jan-Sep 2025: +$8,917) and net-hurt in test (Oct 2025-
        May 2026: -$33,560). At 0.5/0.5 zero kills fired on the 16-mo
        dataset — equivalent to disabled in observable behavior, but
        retains a defense for severely hostile market events.

        See `docs/post_fill_gate_variant_analysis.md` for the full memo.
        """
        cfg = self._get_yaml("trading", "post_fill_gate", default={}) or {}
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "spy_3d_threshold": float(cfg.get("spy_3d_threshold", 0.5)),
            "bk_ratio_threshold": float(cfg.get("bk_ratio_threshold", 0.5)),
        }

    @property
    def regime_sizing_cfg(self) -> dict:
        """Regime-aware sizing config (Phase 1.4b ship, 2026-04-18).

        Classifies each trading day A/B/C1/C2 from SPY T-1 features, then
        multiplies final shares by the per-regime multiplier (stacking on
        top of conviction × macd_zone). Default disabled. Loaded from
        `trading.regime_sizing`. Passed to `trading.regime_helpers`.

        Returns {enabled, vol_threshold_pct, slope_threshold_pct,
        multipliers: {A, B, C1, C2}}. Unknown regime or flag-off →
        multiplier 1.0 downstream (no boost, no skip).
        """
        cfg = self._get_yaml("trading", "regime_sizing", default={}) or {}
        mults_raw = cfg.get("multipliers") or {}
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "vol_threshold_pct": float(cfg.get("vol_threshold_pct", 22.0)),
            "slope_threshold_pct": float(cfg.get("slope_threshold_pct", 0.15)),
            "multipliers": {
                "A":  float(mults_raw.get("A",  1.0)),
                "B":  float(mults_raw.get("B",  1.0)),
                "C1": float(mults_raw.get("C1", 1.0)),
                "C2": float(mults_raw.get("C2", 1.0)),
            },
        }

    @property
    def orphan_reconciler_cfg(self):
        """Build ReconcilerConfig from `orphan_reconciler` block.

        All keys optional — defaults come from the dataclass. Returns the
        dataclass instance ready to pass to reconcile_strategy_orphans.

        Shipping defaults (set in config.yaml):
          auto_close_enabled: true  — flip to false for observe-mode
          max_closes_per_hour: 3    — blast-radius cap if something breaks
          alert_cooldown_minutes: 60 — Telegram dedup per (strategy, symbol)
          lookback_days: 14          — covers the 10-day SMU case
        """
        from trading.orphan_reconciler import ReconcilerConfig
        raw = self._get_yaml("orphan_reconciler", default={}) or {}
        defaults = ReconcilerConfig()
        return ReconcilerConfig(
            auto_close_enabled=bool(raw.get(
                "auto_close_enabled", defaults.auto_close_enabled)),
            lookback_days=int(raw.get(
                "lookback_days", defaults.lookback_days)),
            avg_entry_match_pct=float(raw.get(
                "avg_entry_match_pct", defaults.avg_entry_match_pct)),
            avg_entry_match_abs_min=float(raw.get(
                "avg_entry_match_abs_min", defaults.avg_entry_match_abs_min)),
            max_closes_per_hour=int(raw.get(
                "max_closes_per_hour", defaults.max_closes_per_hour)),
            alert_cooldown_minutes=int(raw.get(
                "alert_cooldown_minutes", defaults.alert_cooldown_minutes)),
        )

    @property
    def two_tier_filter_cfg(self) -> dict:
        """Full two-tier filter config dict (threshold, cap, feature params).

        Returns a dict with keys: enabled, extras_lower, a_tier_lower,
        drop_extras_macd_below, composite_threshold, composite_features.
        Suitable to pass directly to `trading.two_tier_filter.should_keep`.
        """
        cfg = self._get_yaml("trading", "bull_flag", "two_tier_filter", default={}) or {}
        # Provide sane fallbacks so callers can rely on keys being present.
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "extras_lower": float(cfg.get("extras_lower", 10.0)),
            "a_tier_lower": float(cfg.get("a_tier_lower", 20.0)),
            "drop_extras_macd_below": cfg.get("drop_extras_macd_below"),
            "composite_threshold": cfg.get("composite_threshold"),
            "composite_features": cfg.get("composite_features") or {},
        }

    # =========================================================================
    # MACD Zone Filter
    # =========================================================================

    @property
    def macd_zones_enabled(self) -> bool:
        """Whether MACD zone risk scaling is active."""
        return bool(self._get_yaml("trading", "macd_zones", "enabled", default=False))

    @property
    def macd_dead_zone_min_pct(self) -> float:
        """Lower edge of MACD dead zone (% of price)."""
        return float(self._get_yaml("trading", "macd_zones", "dead_zone_min_pct", default=-0.2))

    @property
    def macd_dead_zone_max_pct(self) -> float:
        """Upper edge of MACD dead zone (% of price)."""
        return float(self._get_yaml("trading", "macd_zones", "dead_zone_max_pct", default=0.1))

    @property
    def macd_strong_neg_threshold_pct(self) -> float:
        """MACD% below this = strong negative zone."""
        return float(self._get_yaml("trading", "macd_zones", "strong_neg_threshold_pct", default=-0.5))

    @property
    def macd_strong_neg_multiplier(self) -> float:
        """Risk multiplier for strong negative MACD zone."""
        return float(self._get_yaml("trading", "macd_zones", "strong_neg_multiplier", default=1.25))

    @property
    def macd_strong_pos_threshold_pct(self) -> float:
        """MACD% above this = strong positive zone."""
        return float(self._get_yaml("trading", "macd_zones", "strong_pos_threshold_pct", default=0.5))

    @property
    def macd_strong_pos_multiplier(self) -> float:
        """Risk multiplier for strong positive MACD zone."""
        return float(self._get_yaml("trading", "macd_zones", "strong_pos_multiplier", default=1.5))

    @property
    def macd_normal_multiplier(self) -> float:
        """Risk multiplier for normal MACD zones (between dead and strong)."""
        return float(self._get_yaml("trading", "macd_zones", "normal_multiplier", default=1.0))

    @property
    def max_consecutive_losses(self) -> int:
        """Stop trading for the day after N consecutive losses."""
        return int(self._get_yaml("trading", "max_consecutive_losses", default=2))

    @property
    def market_regime_enabled(self) -> bool:
        """Whether the SPY market regime filter is active."""
        return bool(self._get_yaml("trading", "market_regime", "enabled", default=True))

    @property
    def market_regime_vol_threshold(self) -> float:
        """5-day avg daily range % threshold for volatility regime filter."""
        return float(self._get_yaml("trading", "market_regime", "vol_threshold", default=1.5))

    @property
    def market_regime_sma_period(self) -> int:
        """SMA period for trend detection in regime filter."""
        return int(self._get_yaml("trading", "market_regime", "sma_period", default=50))

    @property
    def market_regime_min_spy_volume_ratio(self) -> float:
        """SPY volume ratio threshold — below this = thin liquidity day."""
        return float(self._get_yaml("trading", "market_regime", "min_spy_volume_ratio", default=0.70))

    @property
    def market_regime_thin_liquidity_breakout_vol_ratio(self) -> float:
        """Stricter breakout volume ratio required on thin liquidity days."""
        return float(self._get_yaml("trading", "market_regime", "thin_liquidity_breakout_vol_ratio", default=2.0))

    @property
    def max_trades_per_day(self) -> int:
        """Maximum trades allowed per day."""
        return int(self._get_yaml("trading", "max_trades_per_day", default=5))

    @property
    def setup_expiry_bars(self) -> int:
        """Cancel pending buy-stop after this many bars."""
        return int(self._get_yaml("trading", "setup_expiry_bars", default=10))

    # =========================================================================
    # Self-Managed Stops
    # =========================================================================

    @property
    def self_managed_stops_enabled(self) -> bool:
        """Whether self-managed stop monitoring is active."""
        return bool(self._get_yaml("trading", "self_managed_stops", "enabled", default=False))

    @property
    def safety_net_sl_pct(self) -> float:
        """Safety-net stop-loss percentage (wide SL on Alpaca for crash protection)."""
        return float(self._get_yaml("trading", "self_managed_stops", "safety_net_sl_pct", default=0.05))

    @property
    def marketable_limit_offset(self) -> float:
        """Minimum dollar offset below price for marketable limit sell."""
        return float(self._get_yaml("trading", "self_managed_stops", "marketable_limit_offset", default=0.03))

    @property
    def marketable_limit_offset_pct(self) -> float:
        """Percentage offset below price for marketable limit sell."""
        return float(self._get_yaml("trading", "self_managed_stops", "marketable_limit_offset_pct", default=0.005))

    @property
    def exit_min_offset(self) -> float:
        """Minimum floor (dollars) on the spread-aware exit limit buffer.

        Used by StopMonitor.compute_limit_price when an ask is supplied
        (stop_bid / stop_bid_rest paths). Prevents the buffer collapsing
        to $0 on penny-spread NBBOs. FABC 2026-06-09: default $0.01.
        """
        return float(self._get_yaml(
            "trading", "self_managed_stops", "exit_min_offset", default=0.01))

    @property
    def exit_spread_offset_factor(self) -> float:
        """Fraction of the bid-ask spread used as the exit-limit buffer.

        FABC 2026-06-09: default 0.30. A 3¢ spread → ~1¢ buffer (vs the
        old fixed $0.03 that placed the limit a full spread below bid).
        """
        return float(self._get_yaml(
            "trading", "self_managed_stops", "exit_spread_offset_factor", default=0.30))

    # =========================================================================
    # Trailing Stop
    # =========================================================================

    @property
    def trailing_stop_enabled(self) -> bool:
        """Whether trailing stop replaces fixed TP."""
        return bool(self._get_yaml("trading", "trailing_stop", "enabled", default=False))

    @property
    def trailing_stop_r(self) -> float:
        """Trail distance in R units below highest high since entry."""
        return float(self._get_yaml("trading", "trailing_stop", "trail_r", default=1.0))

    @property
    def trailing_activate_at_r(self) -> float:
        """Activate trailing stop after price reaches +NR from entry."""
        return float(self._get_yaml("trading", "trailing_stop", "activate_at_r", default=2.0))


def get_config(env_path: Optional[str] = None, yaml_path: Optional[str] = None) -> Config:
    """
    Get or create the singleton Config instance.

    Args:
        env_path: Path to .env file (only used on first call)
        yaml_path: Path to config.yaml (only used on first call)

    Returns:
        Config singleton instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(env_path=env_path, yaml_path=yaml_path)
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing)."""
    global _config_instance
    _config_instance = None
