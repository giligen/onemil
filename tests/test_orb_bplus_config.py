"""B+ RESTART 2026-08-15 config integrity + BT-pipeline drift (review P1-4).

Pins the frozen B+ parameter set in orb.yaml + orb.yaml.template (verbatim from
research/orb_bplus_frozen_params_aug2026.yaml) and asserts the BT pipeline's
load_bt_config() reads the SAME sizing/threshold/veto constants live trades —
so the nightly BT book can never be a different-config ledger.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent


@pytest.fixture(params=['orb.yaml', 'orb.yaml.template'])
def cfg(request):
    return yaml.safe_load(open(ROOT / request.param))


class TestFrozenValues:
    def test_strategy_enabled_is_explicit_bool(self, cfg, request):
        # Shipped FALSE until the 8/16 rehearsal passed; the 8/17 GO
        # session flipped the INSTANCE yaml to True (owner GO 8/15).
        # The template must still ship disabled for new-node setup.
        if 'template' in str(request.node.callspec.id):
            assert cfg['strategy']['enabled'] is False
        else:
            assert isinstance(cfg['strategy']['enabled'], bool)

    def test_threshold(self, cfg):
        assert cfg['filter']['threshold'] == 0.012081536791

    def test_quintile_cutoffs(self, cfg):
        assert cfg['quintile_cutoffs'] == pytest.approx(
            [0.107384942420, 0.194230490420, 0.287618573598, 0.397035743689])

    def test_zparams(self, cfg):
        f = cfg['filter']['features']
        assert f['gap_pct']['mean'] == 133.273200161459
        assert f['gap_pct']['std'] == 1515.341490804868
        assert f['range_total_volume']['mean'] == 971724.893776823999
        assert f['range_close_position']['sign'] == 1
        assert f['prev_day_close_position']['mean'] == 0.498689946893

    def test_adaptive_mults_uniform_1(self, cfg):
        assert cfg['adaptive_mults'] == {'Q1': 1.0, 'Q2': 1.0, 'Q3': 1.0,
                                         'Q4': 1.0, 'Q5': 1.0}

    def test_pdr_veto_11(self, cfg):
        assert cfg['filter']['prev_day_range_veto']['min_prev_day_range_pct'] == 11.0

    def test_g1_veto(self, cfg):
        g = cfg['filter']['g1_veto']
        assert g['enabled'] is True
        assert g['return_volatility_20d_min'] == 7.106
        assert g['prev_day_range_pct_min'] == 9.226

    def test_catalyst_veto_unchanged(self, cfg):
        assert cfg['filter']['catalyst_veto']['enabled'] is True
        assert cfg['filter']['catalyst_veto']['min_cohort'] == 2

    def test_pm_mult_off(self, cfg):
        assert cfg['sizing']['pm_dollar_vol_mult']['enabled'] is False

    def test_sizing(self, cfg):
        assert cfg['sizing']['account_budget_usd'] == 10000
        assert cfg['sizing']['max_concurrent'] == 3
        assert cfg['sizing']['risk_per_trade_usd'] == 375

    def test_per_pos_cap_derives_to_3333(self, cfg):
        cap = cfg['sizing']['account_budget_usd'] / cfg['sizing']['max_concurrent']
        assert cap == pytest.approx(3333.33, abs=0.01)

    def test_static_lock(self, cfg):
        assert cfg['exit']['lock_arm_at_r'] == 1.75
        assert cfg['exit']['lock_stop_r'] == 0.5

    def test_skip_q1(self, cfg):
        assert cfg['filter']['skip_q1'] is True

    def test_kill_rails(self, cfg):
        kr = cfg['risk']['kill_rails']
        assert kr['enabled'] is True
        assert kr['daily_usd'] == -500
        assert kr['weekly_usd'] == -750
        assert kr['month_abandon_usd'] == -1500

    def test_pdt_guard(self, cfg):
        p = cfg['risk']['pdt_guard']
        assert p['enabled'] is True
        assert p['max_daytrades_5d'] == 3
        assert p['equity_threshold_usd'] == 25000

    def test_bt_book_path(self, cfg):
        assert cfg['backtest']['nightly_book_csv'] == \
            'analysis_results/orb_bplus_book.csv'


class TestPipelineConfigDrift:
    """load_bt_config() must equal the live orb.yaml — the nightly BT book
    simulates the same config live trades (no $100K/N4/pdr8 drift)."""

    def test_pipeline_reads_orb_yaml(self):
        from study_orb_pipeline_static_lock import load_bt_config
        c = load_bt_config(str(ROOT / 'orb.yaml'))
        live = yaml.safe_load(open(ROOT / 'orb.yaml'))
        assert c['account'] == live['sizing']['account_budget_usd']
        assert c['n'] == live['sizing']['max_concurrent']
        assert c['risk'] == live['sizing']['risk_per_trade_usd']
        assert c['threshold'] == live['filter']['threshold']
        assert c['pdr_min'] == \
            live['filter']['prev_day_range_veto']['min_prev_day_range_pct']
        assert c['g1_rv20_min'] == \
            live['filter']['g1_veto']['return_volatility_20d_min']
        assert c['g1_pdr_min'] == \
            live['filter']['g1_veto']['prev_day_range_pct_min']
        assert c['g1_enabled'] is True
        assert c['pm_enabled'] is False   # B+ disables PM in BT too

    def test_pipeline_book_path_matches_report_common(self):
        """P1-1: the file the pipeline WRITES == the file report_common READS."""
        from study_orb_pipeline_static_lock import load_bt_config
        import scripts.report_common as rc
        pipeline_out = load_bt_config(str(ROOT / 'orb.yaml'))['book_csv']
        report_reads = rc.bt_book_csv_path()
        assert Path(pipeline_out).resolve() == Path(report_reads).resolve()
