"""trading/hod_break.py — the shared HOD-break spec (detection, capped fill, exits, sizing)."""
import numpy as np
import pytest

from trading import hod_break as hb
from trading.hod_break import HodBreakParams, detect, entry_fill, walk_exit, simulate, consolidation_low, rv_profile, shares_for


def bars(prices, minute0=570):
    """Build o/h/l/c/v/m arrays from a list of (o, h, l, c, v) tuples, one bar per minute."""
    o = np.array([b[0] for b in prices], float); h = np.array([b[1] for b in prices], float)
    l = np.array([b[2] for b in prices], float); c = np.array([b[3] for b in prices], float)
    v = np.array([b[4] for b in prices], float); m = np.arange(minute0, minute0 + len(prices))
    return o, h, l, c, v, m


def drive_then_consolidate(n_consol=5, break_high=11.2):
    """open 10 → drives to 11 (10% above open) → n_consol bars holding 10.7-11.0 → break bar to break_high."""
    tape = [(10.0, 10.3, 9.9, 10.3, 5000), (10.3, 10.7, 10.2, 10.7, 6000), (10.7, 11.0, 10.6, 10.95, 8000)]
    tape += [(10.9, 10.98, 10.7, 10.85, 3000)] * n_consol
    tape += [(10.9, break_high, 10.85, 11.1, 9000)]          # the break bar (high reaches the HOD 11.0)
    tape += [(11.05, 11.1, 11.0, 11.05, 4000)]                # next bar: open 11.05 <= 11.0 x 1.006 = 11.066 → fill
    return tape


class TestRvProfile:
    def test_fraction_steps_with_the_clock(self):
        assert hb.profile_fraction(575) == hb.VP_FRACTION[575]
        assert hb.profile_fraction(599) == hb.VP_FRACTION[585]
        assert hb.profile_fraction(900) == hb.VP_FRACTION[900]
        assert hb.profile_fraction(950) == hb.VP_FRACTION[900]

    def test_rv_is_cum_volume_over_expected_pace(self):
        assert rv_profile(20_000, 1_000_000, 575) == pytest.approx(20_000 / (1_000_000 * 0.02))

    def test_missing_adv_is_nan_never_a_pass(self):
        assert np.isnan(rv_profile(1000, 0, 575))
        assert np.isnan(rv_profile(1000, None, 575))


class TestConsolidation:
    def test_needs_k_bars_within_pct_of_hod(self):
        o, h, l, c, v, m = bars(drive_then_consolidate())
        assert consolidation_low(l, h, 7, HodBreakParams()) == pytest.approx(10.7)

    def test_rejects_when_a_bar_dips_too_far(self):
        tape = drive_then_consolidate(); tape[5] = (10.9, 10.98, 10.4, 10.85, 3000)     # 10.4 < 11.0 x 0.96
        o, h, l, c, v, m = bars(tape)
        assert consolidation_low(l, h, 7, HodBreakParams()) is None

    def test_too_early_is_none(self):
        o, h, l, c, v, m = bars(drive_then_consolidate())
        assert consolidation_low(l, h, 2, HodBreakParams()) is None


class TestDetect:
    def test_first_break_after_consolidation(self):
        o, h, l, c, v, m = bars(drive_then_consolidate())
        sig = detect(o, h, l, v, m, adv20=1_000_000)
        assert sig is not None and sig.bar_idx == 8 and sig.level == pytest.approx(11.0) and sig.stop == pytest.approx(10.7)
        assert sig.dist_open_pct == pytest.approx(10.0)

    def test_causal_floor_blocks_small_moves(self):
        o, h, l, c, v, m = bars(drive_then_consolidate())
        assert detect(o, h, l, v, m, adv20=1_000_000, p=HodBreakParams(min_dist_open_pct=12.0)) is None

    def test_rv_band_blocks_quiet_and_frantic(self):
        o, h, l, c, v, m = bars(drive_then_consolidate())
        assert detect(o, h, l, v, m, adv20=200_000_000) is None       # rv ≈ 0 → below band
        assert detect(o, h, l, v, m, adv20=100) is None               # rv huge → above band

    def test_no_entry_after_last_entry_minute(self):
        o, h, l, c, v, m = bars(drive_then_consolidate(), minute0=925)
        assert detect(o, h, l, v, m, adv20=1_000_000) is None

    def test_start_idx_lets_live_resume_on_new_bars(self):
        o, h, l, c, v, m = bars(drive_then_consolidate())
        assert detect(o, h, l, v, m, adv20=1_000_000, start_idx=9) is None
        assert detect(o, h, l, v, m, adv20=1_000_000, start_idx=8).bar_idx == 8


class TestFillAndExit:
    def test_capped_fill_next_open(self):
        p = HodBreakParams()
        assert entry_fill(11.05, 11.0, p) == pytest.approx(11.05)
        assert entry_fill(11.07, 11.0, p) is None                    # above 11.066 → no chase

    def test_stop_gap_through_and_target_on_close(self):
        p = HodBreakParams()
        o, h, l, c, v, m = bars([(10, 10, 10, 10, 1)] * 3 + [(9.5, 9.6, 9.4, 9.5, 1)])
        k, px, why = walk_exit(o, h, l, c, m, 0, 10.0, 9.8, 10.4, p)
        assert (k, why) == (3, 'stop') and px == pytest.approx(9.5 * 0.999)
        o, h, l, c, v, m = bars([(10, 10, 10, 10, 1), (10.1, 10.6, 10.0, 10.3, 1), (10.3, 10.5, 10.2, 10.45, 1)])
        k, px, why = walk_exit(o, h, l, c, m, 0, 10.0, 9.8, 10.4, p)
        assert (k, why, px) == (2, 'target', 10.4)                  # bar 1 wicked through 10.4 but closed below: no fill

    def test_flat_at_1555(self):
        p = HodBreakParams()
        o, h, l, c, v, m = bars([(10, 10, 10, 10, 1)] * 3, minute0=953)
        k, px, why = walk_exit(o, h, l, c, m, 0, 10.0, 9.8, 10.4, p)
        assert (k, why) == (2, 'eod')

    def test_full_trade(self):
        tape = drive_then_consolidate() + [(11.1, 11.9, 11.05, 11.85, 5000)]      # closes above target 11.05 + 2 x 0.35 = 11.75
        o, h, l, c, v, m = bars(tape)
        t = simulate(o, h, l, c, v, m, adv20=1_000_000)
        assert t is not None and t.entry == pytest.approx(11.05) and t.stop == pytest.approx(10.7)
        assert t.target == pytest.approx(11.75) and t.reason == 'target' and t.rr == pytest.approx(2.0)

    def test_min_r_pct_rejects_noise_stops(self):
        tape = drive_then_consolidate()
        o, h, l, c, v, m = bars(tape)
        assert simulate(o, h, l, c, v, m, adv20=1_000_000, p=HodBreakParams(min_r_pct=5.0)) is None


def test_shares_for():
    assert shares_for(100.0, 11.05, 10.7) == int(100 / 0.35)
    assert shares_for(100.0, 10.0, 10.0) == 0
