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
