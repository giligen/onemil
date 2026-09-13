"""HOD-break BT/live parity — one spec, no forks, constants locked (tests/test_orb_touchgo_parity.py style)."""
from pathlib import Path

from trading.hod_break import HodBreakParams, VP_FRACTION, VP_CHECKPOINTS

REPO = Path(__file__).parent.parent


def test_live_engine_imports_the_shared_spec():
    src = (REPO / 'trading' / 'hod_break_engine.py').read_text(encoding='utf-8')
    assert 'from trading.hod_break import' in src and 'detect' in src and 'shares_for' in src


def test_backtest_imports_the_shared_spec():
    src = (REPO / 'research' / 'bf_zero' / 'spec_sim.py').read_text(encoding='utf-8')
    assert 'from trading.hod_break import' in src and 'simulate' in src


def test_no_private_detector_in_the_engine():
    """The engine must not re-implement consolidation/break logic."""
    src = (REPO / 'trading' / 'hod_break_engine.py').read_text(encoding='utf-8')
    assert 'maximum.accumulate' not in src and 'rolling(' not in src


def test_defaults_locked_to_the_study():
    p = HodBreakParams()
    assert (p.consol_bars, p.consol_pct, p.min_dist_open_pct) == (5, 0.04, 5.0)
    assert (p.rv_lo, p.rv_hi, p.min_r_pct) == (1.0, 5.0, 1.0)
    assert (p.cap, p.target_r, p.max_per_day, p.max_concurrent) == (0.006, 2.0, 8, 4)
    assert (p.last_entry_minute, p.flat_minute) == (930, 955)


def test_config_defaults_match_the_spec_defaults():
    """config.py's hod_break_cfg params must equal HodBreakParams() when the yaml block is absent."""
    from config import Config
    cfg = Config.__dict__['hod_break_cfg'].fget.__doc__  # exists
    import types
    c = Config.__new__(Config); c._get_yaml = lambda *a, **k: {}
    params = Config.hod_break_cfg.fget(c)['params']
    assert HodBreakParams(**params) == HodBreakParams()


def test_volume_profile_constants_are_monotone_and_complete():
    assert list(VP_CHECKPOINTS) == sorted(VP_CHECKPOINTS)
    fr = [VP_FRACTION[c] for c in VP_CHECKPOINTS]
    assert fr == sorted(fr) and 0 < fr[0] < fr[-1] < 1
