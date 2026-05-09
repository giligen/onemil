"""BT↔PROD parity tests for regime-aware sizing (Phase 1.4b ship).

Enforces the same guarantees we have for the two-tier filter and per-tier
MACD zones: backtest and live engine must derive identical regime labels +
multipliers from identical inputs, achieved by both importing from the
shared `trading.regime_helpers` module.

Mirrors the structure of tests/test_bt_prod_parity.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestRegimeHelperImportedByBoth:
    """Both BT and PROD must call into trading.regime_helpers — not a
    privately-forked copy of the classifier in either file."""

    def test_bt_imports_regime_helpers(self):
        src = open('backtest.py').read()
        assert 'from trading.regime_helpers import' in src, (
            "backtest.py must import from trading.regime_helpers")
        assert 'build_regime_lookup' in src, (
            "backtest.py must call build_regime_lookup for BT path")
        assert 'get_regime_multiplier' in src, (
            "backtest.py must use get_regime_multiplier at sizing site")

    def test_prod_imports_regime_helpers(self):
        src = open('trading/trading_engine.py').read()
        assert 'from trading.regime_helpers import' in src, (
            "trading_engine.py must import from trading.regime_helpers")
        assert 'classify_regime' in src, (
            "trading_engine.py must call classify_regime for today's label")
        assert 'compute_regime_features' in src, (
            "trading_engine.py must compute features the same way BT does")
        assert 'get_regime_multiplier' in src, (
            "trading_engine.py must use get_regime_multiplier at sizing site")


class TestConfigLoadedIdentically:
    """BT and PROD must read the SAME config keys with the SAME defaults."""

    def test_shipped_config_matches_expected_defaults(self, tmp_path, monkeypatch):
        """The SHIPPED config (config.yaml.template) has the Grid-winner mults.

        Local `config.yaml` is per-instance and may be partially edited (e.g.,
        dev's was missing the regime_sizing block entirely as of 2026-05-08).
        This test enforces that the *template* — what new deployments get —
        ships the validated values. We construct a Config pointed at the
        template so we don't depend on any one instance's local edits.
        """
        from config import Config
        import shutil
        target = tmp_path / "config.yaml"
        shutil.copy("config.yaml.template", target)
        monkeypatch.chdir(tmp_path)
        cfg = Config().regime_sizing_cfg
        assert cfg['enabled'] is True
        assert cfg['vol_threshold_pct'] == 22.0
        assert cfg['slope_threshold_pct'] == 0.15
        assert cfg['multipliers'] == {'A': 1.25, 'B': 1.0, 'C1': 1.5, 'C2': 0.0}

    def test_bt_loads_shipped_multipliers(self, tmp_path, monkeypatch):
        """Same template-based check, exercised through BacktestRunner."""
        import shutil, os
        # Copy entire repo skeleton for BacktestRunner — keep it minimal:
        # the runner only needs config.yaml in cwd to read its sizing block.
        target = tmp_path / "config.yaml"
        shutil.copy("config.yaml.template", target)
        # BacktestRunner also reads orb.yaml and macd_wave.yaml at import time.
        for opt in ("orb.yaml.template", "macd_wave.yaml.template"):
            if os.path.exists(opt):
                shutil.copy(opt, tmp_path / opt.replace(".template", ""))
        monkeypatch.chdir(tmp_path)
        from backtest import BacktestRunner
        r = BacktestRunner()
        assert r.regime_sizing_enabled is True
        assert r.regime_vol_threshold == 22.0
        assert r.regime_slope_threshold == 0.15
        assert r.regime_multipliers == {'A': 1.25, 'B': 1.0, 'C1': 1.5, 'C2': 0.0}

    def test_prod_loads_same_keys_as_bt(self):
        """Can't instantiate TradingEngine (needs API deps) — verify via
        source inspection that the same config keys are read."""
        src = open('trading/trading_engine.py').read()
        # Every key the BT reads must appear in PROD's config load
        for key in ('regime_sizing', 'vol_threshold_pct',
                    'slope_threshold_pct', 'multipliers'):
            assert key in src, f"PROD must read '{key}' from config"


class TestClassifierDeterminism:
    """Identical inputs → identical output, regardless of caller."""

    @pytest.mark.parametrize("vol, above, slope, expected", [
        # Clean bull
        (10.0, True,  0.5,  'A'),
        (15.0, True,  -0.3, 'A'),
        # Volatile override
        (22.0, True,  0.5,  'B'),
        (30.0, False, -2.0, 'B'),
        # True defensive
        (10.0, False, -0.5, 'C1'),
        (15.0, False, 0.15, 'C1'),
        (15.0, False, 0.0,  'C1'),
        # Shallow-dip-in-uptrend
        (10.0, False, 0.30, 'C2'),
        (15.0, False, 0.60, 'C2'),
        # Unknown on missing inputs
        (None, True,  0.3,  'unknown'),
        (15.0, None,  0.3,  'unknown'),
    ])
    def test_classifier_boundaries(self, vol, above, slope, expected):
        from trading.regime_helpers import classify_regime
        assert classify_regime(vol, above, slope) == expected

    def test_multiplier_lookup_matches_ship_config(self):
        from trading.regime_helpers import get_regime_multiplier
        ship = {'A': 1.25, 'B': 1.0, 'C1': 1.5, 'C2': 0.0}
        assert get_regime_multiplier('A',  ship) == 1.25
        assert get_regime_multiplier('B',  ship) == 1.0
        assert get_regime_multiplier('C1', ship) == 1.5
        assert get_regime_multiplier('C2', ship) == 0.0
        # Unknown / disabled fall through to 1.0 (no-op, safe default)
        assert get_regime_multiplier('unknown',  ship) == 1.0
        assert get_regime_multiplier('disabled', ship) == 1.0


class TestFeatureComputationParity:
    """The feature-engineering step must be byte-identical for BT and PROD.
    Both call `compute_regime_features` — we assert its output is stable."""

    def test_feature_output_is_reproducible(self):
        from trading.regime_helpers import compute_regime_features
        dates = pd.date_range('2025-01-01', periods=80, freq='B')
        # Realistic-ish SPY: 500 → 570 with a wobble in the middle
        closes = np.concatenate([
            np.linspace(500, 540, 40),
            np.linspace(540, 525, 10),  # pullback
            np.linspace(525, 570, 30),
        ])
        df = pd.DataFrame({'bar_date': dates, 'close': closes})

        f1 = compute_regime_features(df)
        f2 = compute_regime_features(df)  # second call, same input
        pd.testing.assert_frame_equal(f1, f2)

    def test_bt_lookup_and_prod_on_demand_agree_on_last_row(self):
        """Synthesize the same SPY history, verify BT's lookup-by-date
        matches PROD's classify-on-last-row result."""
        from trading.regime_helpers import (
            build_regime_lookup, compute_regime_features, classify_regime)
        dates = pd.date_range('2025-01-01', periods=80, freq='B')
        closes = np.linspace(500, 580, 80)
        df = pd.DataFrame({'bar_date': dates, 'close': closes})

        # BT path: build lookup, ask for day = dates[-1]
        lookup = build_regime_lookup(df)
        bt_regime = lookup[dates[-1].strftime('%Y-%m-%d')]

        # PROD path: compute features on everything-except-last-day,
        # classify from last row (yesterday's close).
        df_up_to_yesterday = df.iloc[:-1].copy()
        feats = compute_regime_features(df_up_to_yesterday)
        last = feats.iloc[-1]
        prod_regime = classify_regime(
            float(last['vol_20_ann']) if not pd.isna(last['vol_20_ann']) else None,
            None if pd.isna(last['above_sma_50']) else bool(last['above_sma_50']),
            float(last['sma_50_slope_10d']) if not pd.isna(last['sma_50_slope_10d']) else None,
        )
        assert bt_regime == prod_regime, (
            f"BT lookup returned {bt_regime} but PROD on-demand got {prod_regime}")


class TestSizingSiteParity:
    """Both BT and PROD apply the regime multiplier at the SAME logical
    point (right after MACD zone scaling) using the SAME cap formula:
        effective_max = max_shares × macd_zone × regime_mult
    Source-level check, since we can't run the hot loop in unit tests."""

    def test_bt_sizing_site_stacks_on_macd_zone(self):
        src = open('backtest.py').read()
        # The regime sizing call-site must come after the MACD-zone
        # assignment block — regime_mult stacks on the applied macd zone.
        needle_regime = 'get_regime_multiplier(_regime, self.regime_multipliers)'
        needle_macd = '_applied_macd_zone = zone_mult'
        assert src.find(needle_macd) < src.find(needle_regime), (
            "BT regime sizing must follow the MACD-zone assignment block")
        # The cap formula must include _applied_macd_zone
        assert 'self.planner.max_shares * _applied_macd_zone * _regime_mult' in src, (
            "BT must stack: max_shares × macd_zone × regime_mult")
        # The skip path must use continue (scan loop)
        assert 'REGIME' in src and 'continue' in src

    def test_prod_sizing_site_stacks_on_macd_zone(self):
        src = open('trading/trading_engine.py').read()
        # Same stacking formula in PROD
        assert 'self.planner.max_shares * _applied_macd_zone * _regime_mult' in src, (
            "PROD must stack: max_shares × macd_zone × regime_mult")
        # The skip path must use return None (_validate_and_size exit)
        assert 'REGIME' in src and 'return None' in src

    def test_both_paths_use_same_multiplier_helper(self):
        bt = open('backtest.py').read()
        prod = open('trading/trading_engine.py').read()
        # Exact function name parity
        assert 'get_regime_multiplier(_regime, self.regime_multipliers)' in bt
        assert 'get_regime_multiplier(_regime, self.regime_multipliers)' in prod


class TestFlagOffIsNoOp:
    """With enabled: false, the feature must have ZERO code-path effect."""

    def test_bt_disabled_returns_disabled_label(self):
        from backtest import BacktestRunner
        r = BacktestRunner()
        r.regime_sizing_enabled = False
        # Should short-circuit without touching DB / lookup
        assert r._get_regime_for_date('2026-02-24') == 'disabled'
        # Disabled label → mult 1.0 → no size change downstream
        from trading.regime_helpers import get_regime_multiplier
        assert get_regime_multiplier('disabled', r.regime_multipliers) == 1.0
