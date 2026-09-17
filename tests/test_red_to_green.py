"""trading/red_to_green.py — the F6-PDR spec against hand-built tapes, plus the live-call parity (start_idx)."""
import numpy as np
import pytest

from trading.red_to_green import RedToGreenParams, detect, eligible, level_for, prior_day_range_pct, r_ok, entry_fill

P = RedToGreenParams()
PRIOR_CLOSE = 10.00
PDR_OK = 12.0


def tape(highs, lows, opens=None):
    """Bars from 09:30; volumes 1; minutes 570+."""
    n = len(highs)
    opens = opens or [lows[0]] * n
    return (np.array(opens, float), np.array(highs, float), np.array(lows, float), np.ones(n), np.arange(570, 570 + n))


def test_prior_day_range_pct():
    assert prior_day_range_pct(11.0, 10.0) == pytest.approx(10.0)
    assert prior_day_range_pct(None, 10.0) is None
    assert prior_day_range_pct(9.0, 10.0) is None
    assert prior_day_range_pct(10.0, 0.0) is None


def test_eligible_requires_gap_down_and_pdr():
    assert eligible(9.5, PRIOR_CLOSE, PDR_OK)
    assert not eligible(10.5, PRIOR_CLOSE, PDR_OK)          # opened above prior close
    assert not eligible(9.5, PRIOR_CLOSE, 7.9)              # prior day too quiet
    assert not eligible(9.5, PRIOR_CLOSE, None)
    assert not eligible(9.5, 0.0, PDR_OK)


def test_level_is_prior_close_plus_buffer():
    assert level_for(PRIOR_CLOSE) == pytest.approx(10.03)


def test_no_signal_when_not_eligible():
    o, h, l, v, m = tape([10.2, 10.5], [9.5, 9.6], opens=[10.5, 10.5])   # opened above prior close
    assert detect(o, h, l, v, m, 1e6, PRIOR_CLOSE, PDR_OK) is None


def test_floor_must_hold_on_bars_strictly_before_the_signal_bar():
    # bar0 range 9.50-9.60 (1.05%): floor fails at bar1 even though bar1 reaches the level
    # bar1 range 9.40-10.10: floor on bars 0..1 = (10.10-9.40)/9.40 = 7.4% -> bar2 can signal
    o, h, l, v, m = tape([9.60, 10.10, 10.20], [9.50, 9.40, 9.90], opens=[9.5, 9.5, 10.0])
    sig = detect(o, h, l, v, m, 1e6, PRIOR_CLOSE, PDR_OK)
    assert sig is not None and sig.bar_idx == 2
    assert sig.level == pytest.approx(10.03)
    assert sig.stop == pytest.approx(9.40)                   # lowest low from 09:30 through the signal bar
    assert sig.dist_open_pct == pytest.approx((10.03 / 9.5 - 1) * 100)


def test_signal_is_the_first_bar_reaching_the_level_after_the_floor():
    o, h, l, v, m = tape([10.00, 9.90, 9.95, 10.05, 10.50], [9.40, 9.50, 9.60, 9.80, 10.0], opens=[9.5] * 5)
    sig = detect(o, h, l, v, m, 1e6, PRIOR_CLOSE, PDR_OK)
    assert sig.bar_idx == 3 and sig.stop == pytest.approx(9.40)


def test_last_entry_minute_blocks_late_signals():
    o, h, l, v, m = tape([10.00, 10.50], [9.40, 9.50], opens=[9.5, 9.5])
    m = np.array([840, 841])
    assert detect(o, h, l, v, m, 1e6, PRIOR_CLOSE, PDR_OK) is None


def test_stop_at_or_above_level_is_skipped():
    p = RedToGreenParams(range_floor_pct=0.0)
    o, h, l, v, m = tape([10.05, 10.20], [10.04, 10.10], opens=[9.9, 10.1])   # lows above the level: no risk defined
    assert detect(o, h, l, v, m, 1e6, PRIOR_CLOSE, PDR_OK, p) is None


def test_live_incremental_calls_agree_with_the_backtest_call():
    o, h, l, v, m = tape([10.00, 9.90, 9.95, 10.05, 10.50], [9.40, 9.50, 9.60, 9.80, 10.0], opens=[9.5] * 5)
    full = detect(o, h, l, v, m, 1e6, PRIOR_CLOSE, PDR_OK)
    seen = None
    for k in range(1, len(h) + 1):                          # the engine sees the tape one closed bar at a time
        s = detect(o[:k], h[:k], l[:k], v[:k], m[:k], 1e6, PRIOR_CLOSE, PDR_OK, start_idx=k - 1)
        if s is not None and seen is None:
            seen = s
    assert seen == full


def test_fill_and_r_floor_are_the_shared_rules():
    assert entry_fill(10.05, 10.03, P) == pytest.approx(10.05)
    assert entry_fill(10.10, 10.03, P) is None               # above the 0.6% cap: no chase
    assert r_ok(10.05, 9.40)
    assert not r_ok(10.05, 10.00)                             # 0.5% < min_r_pct
