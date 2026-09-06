"""trading/orb_experimental_rules — the 2026-09-05 ORB signal study hooks.

Pins: flags default OFF (any_on False with a clean env), each mask is
fail-open on NaN, the ranking key orders RVOL descending with NaN last,
midpoint-kill fires only below the midpoint before +0.5R, re-arm is once and
only inside the window after a tag/mid-kill exit.
"""
import math

import pandas as pd
import pytest

from trading import orb_experimental_rules as X


def test_flags_default_off(monkeypatch):
    for k in ('ORB_EXP_RVOL_VETO', 'ORB_EXP_RVOL_RANK', 'ORB_EXP_RCP_GATE', 'ORB_EXP_RCP_FORM',
              'ORB_EXP_RATR_MIN', 'ORB_EXP_RATR_MAX', 'ORB_EXP_MID_KILL', 'ORB_EXP_REARM'):
        monkeypatch.delenv(k, raising=False)
    f = X.load_flags()
    assert not f.any_on
    assert f.rcp_form == 'green'


def test_flags_parse(monkeypatch):
    monkeypatch.setenv('ORB_EXP_RVOL_VETO', '1.5')
    monkeypatch.setenv('ORB_EXP_RCP_GATE', 'post')
    monkeypatch.setenv('ORB_EXP_RCP_FORM', 'upper')
    monkeypatch.setenv('ORB_EXP_MID_KILL', '1')
    f = X.load_flags()
    assert f.any_on and f.rvol_veto == 1.5 and f.rcp_gate == 'post' and f.rcp_form == 'upper' and f.mid_kill


def test_bad_gate_value_raises(monkeypatch):
    monkeypatch.setenv('ORB_EXP_RCP_GATE', 'sideways')
    with pytest.raises(ValueError):
        X.load_flags()


def test_rvol_mask_fail_open_and_threshold():
    s = pd.Series([0.4, 1.0, 2.5, float('nan')])
    assert X.rvol_keep_mask(s, 1.0).tolist() == [False, True, True, True]


def test_rvol_rank_key_desc_nan_last():
    s = pd.Series([1.0, 3.0, float('nan'), 2.0])
    order = s.index[X.rvol_rank_key(s).argsort()].tolist()
    assert order == [1, 3, 0, 2]


def test_range_direction_masks():
    df = pd.DataFrame({'range_return_pct': [1.2, -0.5, 0.0, float('nan')],
                       'range_close_position': [0.9, 0.2, 0.5, float('nan')]})
    assert X.range_direction_keep_mask(df, 'green').tolist() == [True, False, False, True]
    assert X.range_direction_keep_mask(df, 'upper').tolist() == [True, False, True, True]
    with pytest.raises(ValueError):
        X.range_direction_keep_mask(df, 'sideways')


def test_ratr_mask_and_tiers():
    s = pd.Series([0.1, 0.45, 0.9, float('nan')])
    assert X.ratr_keep_mask(s, 0.3, None).tolist() == [False, True, True, True]
    assert X.ratr_keep_mask(s, None, 0.6).tolist() == [True, True, False, True]
    assert [X.ratr_tier(v) for v in s] == ['narrow', 'normal', 'wide', 'unknown']


def test_midpoint_kill_semantics():
    rh, rl, e = 10.0, 9.0, 10.03          # R = 1.0, mid = 9.5, +0.5R = 10.53
    assert X.midpoint_kill_fires(9.40, rh, rl, e, max_high_since_entry=10.20)     # below mid, never +0.5R
    assert not X.midpoint_kill_fires(9.60, rh, rl, e, max_high_since_entry=10.20) # above mid
    assert not X.midpoint_kill_fires(9.40, rh, rl, e, max_high_since_entry=10.60) # +0.5R was touched


def test_rearm_rules():
    assert X.rearm_allowed('tag_bb', 5, 10, rearmed_already=False)
    assert X.rearm_allowed('mid_kill', 5, 10, rearmed_already=False)
    assert not X.rearm_allowed('stop', 5, 10, rearmed_already=False)      # not an early tag exit
    assert not X.rearm_allowed('tag_bb', 12, 10, rearmed_already=False)   # window closed
    assert not X.rearm_allowed('tag_bb', 5, 10, rearmed_already=True)     # once only


class TestFeatureVeto:
    """V1 (2026-09-06): post-selection, no-refill vetoes on existing features."""

    def test_parse(self):
        from trading.orb_experimental_rules import parse_feature_vetoes
        assert parse_feature_vetoes('') == ()
        assert parse_feature_vetoes('range_size_pct<=2.221; spy_3d_range_pct>=1.484') == (
            ('range_size_pct', '<=', 2.221), ('spy_3d_range_pct', '>=', 1.484))
        import pytest
        with pytest.raises(ValueError):
            parse_feature_vetoes('range_size_pct=2')

    def test_keep_mask_any_rule_fires_nan_kept(self):
        import pandas as pd
        from trading.orb_experimental_rules import feature_veto_keep_mask, parse_feature_vetoes
        df = pd.DataFrame({'range_size_pct': [1.0, 3.0, 3.0, None],
                           'spy_3d_range_pct': [0.5, 2.0, 0.5, 0.5]})
        rules = parse_feature_vetoes('range_size_pct<=2.221;spy_3d_range_pct>=1.484')
        keep = feature_veto_keep_mask(df, rules)
        assert keep.tolist() == [False, False, True, True]

    def test_unknown_column_is_loud(self):
        import pandas as pd, pytest
        from trading.orb_experimental_rules import feature_veto_keep_mask
        with pytest.raises(KeyError):
            feature_veto_keep_mask(pd.DataFrame({'a': [1]}), (('nope', '<=', 1.0),))

    def test_flags_env(self, monkeypatch):
        from trading.orb_experimental_rules import load_flags
        monkeypatch.setenv('ORB_EXP_FEAT_VETO', 'range_size_pct<=2.221')
        fl = load_flags()
        assert fl.feat_veto == (('range_size_pct', '<=', 2.221),) and fl.any_on
        assert 'feat_veto=range_size_pct<=2.221' in fl.describe()
        monkeypatch.delenv('ORB_EXP_FEAT_VETO')
        assert load_flags().feat_veto == ()
