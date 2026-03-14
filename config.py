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

        # Load YAML config
        if yaml_path is None:
            yaml_path = project_root / "config.yaml"

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
    def _load_yaml_only() -> dict:
        """Load config.yaml without .env (for backtest use — avoids env pollution)."""
        project_root = Path(__file__).parent
        yaml_path = project_root / "config.yaml"
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
    def alpaca_base_url(self) -> str:
        """Alpaca API base URL."""
        return os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")

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
        return float(self._get_yaml("scanner", "price_max", default=20.0))

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
        """SQLite database file path."""
        return self._get_yaml("database", "path", default="data/onemil.db")

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
        return int(self._get_yaml("trading", "bull_flag", "max_pullback_candles", default=5))

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
    def require_macd_positive(self) -> bool:
        """Whether bull flag detector requires positive MACD."""
        return bool(self._get_yaml("trading", "bull_flag", "require_macd_positive", default=False))

    @property
    def circuit_breaker_dd(self) -> float:
        """Drawdown threshold to trigger circuit breaker (dollars)."""
        return float(self._get_yaml("trading", "circuit_breaker_dd", default=1500.0))

    @property
    def circuit_breaker_pause(self) -> int:
        """Number of trades to skip when circuit breaker triggers."""
        return int(self._get_yaml("trading", "circuit_breaker_pause", default=1))

    @property
    def market_regime_enabled(self) -> bool:
        """Whether the SPY market regime filter is active."""
        return bool(self._get_yaml("trading", "market_regime", "enabled", default=True))

    @property
    def market_regime_spy_5d_return_min(self) -> float:
        """Minimum SPY 5-day return percentage to allow trading."""
        return float(self._get_yaml("trading", "market_regime", "spy_5d_return_min", default=-2.0))

    @property
    def setup_expiry_bars(self) -> int:
        """Cancel pending buy-stop after this many bars."""
        return int(self._get_yaml("trading", "setup_expiry_bars", default=10))


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
