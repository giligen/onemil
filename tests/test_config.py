"""
Unit tests for config.Config class.

Tests loading from .env and config.yaml, validation of required keys,
YAML defaults, typed property access, and singleton management.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from config import Config, get_config, reset_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the config singleton before and after every test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture()
def env_file(tmp_path):
    """Create a minimal valid .env file and return its path."""
    env = tmp_path / ".env"
    env.write_text(
        "ALPACA_API_KEY=test-key\n"
        "ALPACA_API_SECRET=test-secret\n"
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2\n"
    )
    return str(env)


@pytest.fixture()
def yaml_file(tmp_path):
    """Create a config.yaml with custom values and return its path."""
    import yaml

    cfg = {
        "log_level": "DEBUG",
        "scanner": {
            "price_min": 3.0,
            "price_max": 15.0,
            "float_max": 5_000_000,
            "gap_pct_min": 4.0,
            "intraday_change_pct_min": 12.0,
            "relative_volume_min": 8.0,
            "volume_profile_days": 30,
        },
        "timing": {
            "premarket_poll_interval": 120,
            "intraday_scan_interval": 5,
            "premarket_start": "05:00",
            "market_open": "09:30",
            "market_close": "15:30",
        },
        "database": {"path": "data/custom.db"},
        "float_cache": {"refresh_days": 14},
    }
    yaml_path = tmp_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f)
    return str(yaml_path)


# ---------------------------------------------------------------------------
# Loading / Validation
# ---------------------------------------------------------------------------

class TestConfigLoading:
    """Tests for Config initialisation and validation."""

    def test_load_valid_config(self, env_file, yaml_file):
        """Config loads successfully when both .env and config.yaml are valid."""
        cfg = Config(env_path=env_file, yaml_path=yaml_file)
        assert cfg.alpaca_api_key == "test-key"
        assert cfg.alpaca_api_secret == "test-secret"

    def test_missing_env_raises_file_not_found(self, tmp_path):
        """FileNotFoundError is raised when .env does not exist."""
        missing = str(tmp_path / "nonexistent.env")
        with pytest.raises(FileNotFoundError, match="Environment file not found"):
            Config(env_path=missing)

    def test_missing_required_env_vars_raises_value_error(self, tmp_path):
        """ValueError is raised when required env vars are absent."""
        env = tmp_path / ".env"
        env.write_text("SOME_OTHER_VAR=value\n")

        # Clear any previously loaded env vars so they don't leak
        with patch.dict(os.environ, {}, clear=True):
            # Preserve PATH so subprocess-based loaders still work
            os.environ["PATH"] = ""
            with pytest.raises(ValueError, match="Missing required environment variables"):
                Config(env_path=str(env))

    def test_yaml_defaults_when_no_yaml(self, env_file, tmp_path):
        """All YAML-backed properties return defaults when config.yaml is absent."""
        missing_yaml = str(tmp_path / "no_such.yaml")
        cfg = Config(env_path=env_file, yaml_path=missing_yaml)

        assert cfg.log_level == "INFO"
        assert cfg.price_min == 2.0
        assert cfg.price_max == 20.0
        assert cfg.float_max == 10_000_000
        assert cfg.gap_pct_min == 2.0
        assert cfg.intraday_change_pct_min == 10.0
        assert cfg.relative_volume_min == 5.0
        assert cfg.volume_profile_days == 50
        assert cfg.premarket_poll_interval == 60
        assert cfg.intraday_scan_interval == 15
        assert cfg.premarket_start == "04:00"
        assert cfg.market_open == "09:30"
        assert cfg.market_close == "16:00"
        assert cfg.db_path == "data/onemil.db"
        assert cfg.float_cache_refresh_days == 7

        # Trading defaults
        assert cfg.trading_enabled is False
        assert cfg.position_size_dollars == 500
        assert cfg.max_shares == 1000
        assert cfg.max_positions == 3
        assert cfg.daily_loss_limit == -100.0
        assert cfg.max_risk_per_share == 0.20
        assert cfg.min_risk_reward == 2.0
        assert cfg.pattern_poll_interval == 60
        assert cfg.stop_trading_before_close_min == 15
        assert cfg.min_pole_candles == 3
        assert cfg.min_pole_gain_pct == 3.0
        assert cfg.max_retracement_pct == 50.0
        assert cfg.max_pullback_candles == 5
        assert cfg.min_breakout_volume_ratio == 1.5

        # Telegram defaults
        assert cfg.telegram_enabled is True
        assert cfg.telegram_send_on_startup is True
        assert cfg.telegram_send_on_qualified is True
        assert cfg.telegram_send_on_pattern is True
        assert cfg.telegram_send_on_trade is True
        assert cfg.telegram_send_on_close is True
        assert cfg.telegram_send_daily_report is True


# ---------------------------------------------------------------------------
# Property types / values
# ---------------------------------------------------------------------------

class TestConfigProperties:
    """Verify every property returns the correct type and custom YAML value."""

    def test_api_properties_are_strings(self, env_file, yaml_file):
        """API key/secret/url properties are strings."""
        cfg = Config(env_path=env_file, yaml_path=yaml_file)
        assert isinstance(cfg.alpaca_api_key, str)
        assert isinstance(cfg.alpaca_api_secret, str)
        assert isinstance(cfg.alpaca_base_url, str)

    def test_scanner_properties_types(self, env_file, yaml_file):
        """Scanner properties have correct numeric types."""
        cfg = Config(env_path=env_file, yaml_path=yaml_file)
        assert isinstance(cfg.price_min, float)
        assert isinstance(cfg.price_max, float)
        assert isinstance(cfg.float_max, int)
        assert isinstance(cfg.gap_pct_min, float)
        assert isinstance(cfg.intraday_change_pct_min, float)
        assert isinstance(cfg.relative_volume_min, float)
        assert isinstance(cfg.volume_profile_days, int)

    def test_timing_properties_types(self, env_file, yaml_file):
        """Timing properties have correct types (int for intervals, str for times)."""
        cfg = Config(env_path=env_file, yaml_path=yaml_file)
        assert isinstance(cfg.premarket_poll_interval, int)
        assert isinstance(cfg.intraday_scan_interval, int)
        assert isinstance(cfg.premarket_start, str)
        assert isinstance(cfg.market_open, str)
        assert isinstance(cfg.market_close, str)

    def test_custom_yaml_values_loaded(self, env_file, yaml_file):
        """Custom YAML values override defaults."""
        cfg = Config(env_path=env_file, yaml_path=yaml_file)
        assert cfg.log_level == "DEBUG"
        assert cfg.price_min == 3.0
        assert cfg.price_max == 15.0
        assert cfg.float_max == 5_000_000
        assert cfg.gap_pct_min == 4.0
        assert cfg.intraday_change_pct_min == 12.0
        assert cfg.relative_volume_min == 8.0
        assert cfg.volume_profile_days == 30
        assert cfg.premarket_poll_interval == 120
        assert cfg.intraday_scan_interval == 5
        assert cfg.premarket_start == "05:00"
        assert cfg.market_close == "15:30"
        assert cfg.db_path == "data/custom.db"
        assert cfg.float_cache_refresh_days == 14

    def test_db_path_type(self, env_file, yaml_file):
        """db_path is a string."""
        cfg = Config(env_path=env_file, yaml_path=yaml_file)
        assert isinstance(cfg.db_path, str)

    def test_float_cache_refresh_days_type(self, env_file, yaml_file):
        """float_cache_refresh_days is an int."""
        cfg = Config(env_path=env_file, yaml_path=yaml_file)
        assert isinstance(cfg.float_cache_refresh_days, int)


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

class TestSingleton:
    """Tests for get_config() / reset_config() singleton management."""

    def test_get_config_returns_same_instance(self, env_file, yaml_file):
        """Consecutive get_config() calls return the same object."""
        first = get_config(env_path=env_file, yaml_path=yaml_file)
        second = get_config()
        assert first is second

    def test_reset_config_clears_singleton(self, env_file, yaml_file):
        """reset_config() forces a fresh instance on next get_config()."""
        first = get_config(env_path=env_file, yaml_path=yaml_file)
        reset_config()
        second = get_config(env_path=env_file, yaml_path=yaml_file)
        assert first is not second

    def test_get_config_passes_paths_only_on_creation(self, env_file, yaml_file):
        """Paths are only used on the first call; subsequent calls ignore them."""
        cfg = get_config(env_path=env_file, yaml_path=yaml_file)
        # Second call with different (bogus) paths should still return the cached instance
        same_cfg = get_config(env_path="/bogus/.env", yaml_path="/bogus/config.yaml")
        assert cfg is same_cfg


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

class TestGetYamlHelper:
    """Tests for the _get_yaml nested key traversal."""

    def test_deeply_nested_key(self, env_file, yaml_file):
        """_get_yaml traverses nested dicts correctly."""
        cfg = Config(env_path=env_file, yaml_path=yaml_file)
        assert cfg._get_yaml("scanner", "price_min") == 3.0

    def test_missing_intermediate_key_returns_default(self, env_file, yaml_file):
        """_get_yaml returns the default when an intermediate key is absent."""
        cfg = Config(env_path=env_file, yaml_path=yaml_file)
        assert cfg._get_yaml("nonexistent", "sub", default="fallback") == "fallback"

    def test_non_dict_node_returns_default(self, env_file):
        """_get_yaml returns default when a non-dict node is traversed."""
        cfg = Config(env_path=env_file, yaml_path="/nonexistent.yaml")
        cfg._yaml = {"scanner": "not_a_dict"}
        assert cfg._get_yaml("scanner", "price_min", default=99) == 99
