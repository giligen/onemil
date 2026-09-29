"""BT/live parity for every ORB filter/veto switch (2026-09-29 BEZ fix).

The 2026-09-28 disagreement: live ran with `filter.catalyst_veto.enabled:
false` in orb.yaml (owner GO 9/19), but study_orb_pipeline_static_lock.py's
nightly book only read the ORB_CATALYST_VETO env var — hard-coded default
'1' (ON) whenever the var was unset — and never looked at orb.yaml. The
nightly book kept vetoing (dropped BEZ, slot #2) after live had turned the
veto off. Fixed in `load_bt_config()`: every boolean filter/veto switch now
reads the SAME orb.yaml key trading/orb_engine.py reads, with the SAME
default, and the matching ORB_<X> env var overrides ONLY when explicitly set
(`_env_bool`, mirroring the engine's `if env is not None: override` pattern).

This test builds the LIVE engine's flag set and the PIPELINE's flag set from
the SAME orb.yaml dict and asserts they agree for every switch, across
several yaml permutations, plus dedicated tests for the env-override
semantics and the literal 9/28 regression.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine
from trading.stop_monitor import StopMonitor
from tests.conftest import load_orb_yaml_pinned
from study_orb_pipeline_static_lock import load_bt_config

ROOT = Path(__file__).parent.parent

# (engine attribute, bt_cfg key) for every boolean filter/veto switch that
# BOTH the live engine and the nightly pipeline must read from the SAME
# orb.yaml key with the SAME default.
BOOL_SWITCH_PAIRS = [
    ('catalyst_veto_enabled', 'catalyst_enabled'),
    ('pdr_veto_enabled', 'pdr_enabled'),
    ('g1_veto_enabled', 'g1_enabled'),
    ('range_size_veto_enabled', 'rs_enabled'),
    ('skip_q1', 'skip_q1'),
    ('pm_news_gate', 'pm_news_gate'),
    ('dedup_by_family', 'dedup_by_family'),
    ('dedup_by_super_group', 'dedup_by_super_group'),
]
# (engine attribute, bt_cfg key) for the numeric thresholds that ride along
# with a subset of the switches above.
VALUE_PAIRS = [
    ('catalyst_min_cohort', 'catalyst_min_cohort'),
    ('pdr_veto_min_pct', 'pdr_min'),
    ('range_size_veto_min_pct', 'rs_min'),
]

ENV_VARS = ('ORB_CATALYST_VETO', 'ORB_PDR_VETO', 'ORB_PDR_VETO_MIN_PCT',
            'ORB_G1_VETO', 'ORB_RANGE_SIZE_VETO',
            'ORB_RANGE_SIZE_VETO_MIN_PCT', 'ORB_SKIP_Q1', 'ORB_PM_NEWS_GATE')


def _clear_override_env(monkeypatch):
    """A stray exported ORB_CATALYST_VETO=1 from a research shell must not
    silently make this test pass — every override env var is cleared."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _make_engine(cfg):
    al = MagicMock(spec=AlpacaClient)
    al.get_account_info.return_value = {
        'equity': 10000.0, 'multiplier': 1.0, 'daytrade_count': 0,
        'buying_power': 10000.0}
    d = tempfile.mkdtemp()
    db = Database(trades_path=os.path.join(d, 'trades.db'),
                  cache_path=os.path.join(d, 'cache.db'))
    return ORBEngine(alpaca_client=al, db=db,
                      stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


def _make_bt_cfg(cfg, tmp_path, name):
    """Write `cfg` to a throwaway yaml file and run it through the SAME
    load_bt_config() the nightly pipeline uses, so both sides of the parity
    check are built from one identical in-memory dict."""
    yaml_path = tmp_path / name
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    return load_bt_config(str(yaml_path))


@pytest.mark.parametrize("overrides", [
    {},  # today's production orb.yaml (test-neutral pins only)
    {"filter.catalyst_veto.enabled": True},
    {"filter.catalyst_veto.enabled": False},
    {"filter.prev_day_range_veto.enabled": False},
    {"filter.g1_veto.enabled": False},
    {"filter.range_size_veto.enabled": True},
    {"filter.range_size_veto.enabled": False},
    {"filter.skip_q1": False},
    {"sizing.pm_dollar_vol_mult.news_gate": False},
    {"dedup.by_family": False},
    {"dedup.by_super_group": False},
], ids=lambda o: (str(o) if o else "production-pinned"))
def test_engine_and_pipeline_flags_agree(overrides, tmp_path, monkeypatch):
    """With no ORB_<X> env var set (the normal service/cron invocation),
    every filter/veto switch must resolve to the SAME value in the live
    engine and in the pipeline's load_bt_config(), for a range of yaml
    permutations — not just today's values."""
    _clear_override_env(monkeypatch)
    cfg = load_orb_yaml_pinned(**overrides)
    engine = _make_engine(cfg)
    bt_cfg = _make_bt_cfg(cfg, tmp_path, "orb_parity.yaml")

    for engine_attr, bt_key in BOOL_SWITCH_PAIRS:
        ev, bv = getattr(engine, engine_attr), bt_cfg[bt_key]
        assert ev == bv, (
            f"parity break: engine.{engine_attr}={ev} != "
            f"bt_cfg['{bt_key}']={bv} for overrides={overrides}")
    for engine_attr, bt_key in VALUE_PAIRS:
        ev, bv = getattr(engine, engine_attr), bt_cfg[bt_key]
        assert ev == pytest.approx(bv), (
            f"parity break: engine.{engine_attr}={ev} != "
            f"bt_cfg['{bt_key}']={bv} for overrides={overrides}")


def test_bez_20260928_regression(tmp_path, monkeypatch):
    """The literal 9/28 disagreement: production orb.yaml has catalyst_veto
    OFF. With no env var set, the pipeline must NOT veto on catalyst —
    BEZ (or any newsless-and-alone pick) must survive that gate."""
    _clear_override_env(monkeypatch)
    cfg = load_orb_yaml_pinned()  # production orb.yaml, unmodified filter section
    assert cfg['filter']['catalyst_veto']['enabled'] is False, (
        "this regression test assumes today's orb.yaml still ships "
        "catalyst_veto OFF (owner GO 9/19) — if the owner re-enabled it, "
        "update this assertion rather than deleting the test")
    bt_cfg = _make_bt_cfg(cfg, tmp_path, "orb_bez.yaml")
    assert bt_cfg['catalyst_enabled'] is False


@pytest.mark.parametrize("env_value,expected", [
    ('0', False), ('false', False), ('no', False), ('off', False),
    ('1', True), ('true', True), ('yes', True), ('on', True),
])
def test_catalyst_veto_env_overrides_in_either_direction(
        env_value, expected, tmp_path, monkeypatch):
    """ORB_CATALYST_VETO, once explicitly set, wins over orb.yaml either way
    (matches trading/orb_engine.py's own override semantics)."""
    _clear_override_env(monkeypatch)
    monkeypatch.setenv('ORB_CATALYST_VETO', env_value)
    for yaml_val in (True, False):
        cfg = load_orb_yaml_pinned(**{"filter.catalyst_veto.enabled": yaml_val})
        bt_cfg = _make_bt_cfg(cfg, tmp_path,
                               f"orb_cv_{yaml_val}_{env_value}.yaml")
        assert bt_cfg['catalyst_enabled'] is expected, (
            f"ORB_CATALYST_VETO={env_value!r} should force "
            f"{expected} regardless of yaml={yaml_val}")


def test_catalyst_veto_env_unset_follows_yaml(tmp_path, monkeypatch):
    """With ORB_CATALYST_VETO unset, the pipeline must follow orb.yaml, not
    a hard-coded default — the exact defect this file guards against."""
    _clear_override_env(monkeypatch)
    for yaml_val in (True, False):
        cfg = load_orb_yaml_pinned(**{"filter.catalyst_veto.enabled": yaml_val})
        bt_cfg = _make_bt_cfg(cfg, tmp_path, f"orb_cv_unset_{yaml_val}.yaml")
        assert bt_cfg['catalyst_enabled'] is yaml_val, (
            f"ORB_CATALYST_VETO unset must follow orb.yaml={yaml_val}, "
            f"got {bt_cfg['catalyst_enabled']}")
