"""Pin that Config.orphan_reconciler_cfg parses the yaml correctly,
and that main.py wires it into all three engines so the kill switches
actually do something."""
from pathlib import Path

import yaml

from trading.orphan_reconciler import ReconcilerConfig


ROOT = Path(__file__).parent.parent


def test_default_cfg_is_safe_active():
    """When the yaml block is absent, defaults are the active-mode
    config (auto_close_enabled=True, max 3 closes/hr)."""
    cfg = ReconcilerConfig()
    assert cfg.auto_close_enabled is True
    assert cfg.max_closes_per_hour == 3
    assert cfg.lookback_days >= 7  # covers the SMU 10-day case


def test_shipping_config_yaml_has_reconciler_block():
    """config.yaml must carry the orphan_reconciler block — flipping the
    kill switch is meant to be a one-line yaml edit, not a code change."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert "orphan_reconciler" in cfg
    rc = cfg["orphan_reconciler"]
    assert isinstance(rc.get("auto_close_enabled"), bool)
    assert isinstance(rc.get("max_closes_per_hour"), int)
    assert isinstance(rc.get("lookback_days"), int)


def test_config_property_returns_dataclass():
    """The Config.orphan_reconciler_cfg property returns a real
    ReconcilerConfig (not a dict), ready to pass to
    reconcile_strategy_orphans without further conversion."""
    from config import Config
    cfg = Config()
    rc = cfg.orphan_reconciler_cfg
    assert isinstance(rc, ReconcilerConfig)
    # Shipping defaults from config.yaml as of 2026-06-05
    assert rc.auto_close_enabled is True
    assert rc.lookback_days == 14


def test_main_wires_engines_with_reconciler_cfg():
    """Source inspection: each engine constructor in main.py is followed
    by `engine.orphan_reconciler_cfg = config.orphan_reconciler_cfg`
    (or equivalent), so the engine's sync path picks up yaml settings."""
    main_src = (ROOT / "main.py").read_text()
    # Three engines, three settings — one per strategy.
    assert main_src.count("orphan_reconciler_cfg = config.orphan_reconciler_cfg") >= 3
