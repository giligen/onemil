"""ORB budget ramp checker (owner adopted 2026-09-07): pure gate logic.

Gate-2 of docs/scaling_plan_2026.md adds the ex-monster and BT-band columns
and the parity FREEZE; they are pinned at the bottom of this file.
"""
import importlib.util, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location('orb_ramp_check', os.path.join(ROOT, 'scripts', 'orb_ramp_check.py'))
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)

from trading import ramp_bt_band as bb          # noqa: E402
from trading import ramp_freeze as rf           # noqa: E402

# Synthetic BT reference (R = pnl_pct / range_size_pct in the real book):
# mostly -1R with a fat right tail, like the honest ORB books.
BT_R = [-1.0] * 6 + [0.5, 1.0, 4.0, 7.0]

SESS = [f'2026-09-{d:02d}' for d in (8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 28)]   # 15 sessions


def F(day, pnl, slip_bps=20.0, total_risk=100.0):
    trig = 10.0
    return {'trade_date': day, 'symbol': 'X', 'pnl': pnl, 'entry_price': trig,
            'fill_price': trig * (1 + slip_bps / 1e4), 'total_risk': total_risk,
            'exited_at': f'{day}T15:00:00'}


def stats(fills, sessions=15, parity=None, budget=10000, limit=-750,
          bt_r=BT_R, **kw):
    """compute_stats with the BT band scored (an unscored band blocks ADVANCE)."""
    return m.compute_stats(fills, budget, limit, sessions, SESS[:sessions],
                           parity or {}, bt_r=bt_r, **kw)


def test_stage_lookup():
    assert m.stage_for_budget(10000)['name'] == 'S0' and m.next_stage(m.stage_for_budget(10000))['budget'] == 30000
    assert m.next_stage(m.stage_for_budget(100000)) is None


def test_advance_all_gates():
    fills = [F(d, 60) for d in SESS[:8]]
    assert m.verdict(stats(fills)) == 'ADVANCE'


def test_hold_on_each_missing_gate():
    fills = [F(d, 60) for d in SESS[:8]]
    assert m.verdict(stats(fills[:7])) == 'HOLD'                                   # fills
    assert m.verdict(stats(fills, sessions=14)) == 'HOLD'                          # sessions
    assert m.verdict(stats(fills[:-1] + [F(SESS[7], -500)])) == 'HOLD'             # P&L <= 0
    assert m.verdict(stats(fills, parity={SESS[3]: ['BT picks never ordered live: [ABC]']})) == 'HOLD'
    assert m.verdict(stats([F(d, 60, slip_bps=55.0) for d in SESS[:8]])) == 'HOLD'  # slippage 55 > 40
    limit_hit = fills + [F(SESS[10], -800)]                                         # within last 10 sessions
    s = stats(limit_hit); assert s.limit_hits == 1 and m.verdict(s) == 'HOLD'


def test_parity_outside_stage_ignored():
    fills = [F(d, 60) for d in SESS[:8]]
    assert m.verdict(stats(fills, parity={'2026-08-01': ['fill-parity: BT FILLED but live never']})) == 'ADVANCE'


def test_demote_and_pause():
    assert m.verdict(stats([F(d, -130) for d in SESS[:5]])) == 'DEMOTE'            # -6.5% and streak 5
    assert m.verdict(stats([F(d, -50) for d in SESS[:5]])) == 'DEMOTE'             # streak 5 alone
    two_hits = [F(SESS[0], -800), F(SESS[5], -800)]
    assert m.verdict(stats(two_hits)) == 'PAUSE'                                   # -16% of budget
    slip = [F(d, 30, slip_bps=70.0) for d in SESS[:3]]
    assert stats(slip).slip_2x_fills == 3 and m.verdict(stats(slip)) == 'DEMOTE'


def test_units_hold_across_stages():
    s0 = [F(d, 60) for d in SESS[:8]]; s1 = [F(d, 180) for d in SESS[:8]]
    assert m.verdict(stats(s0)) == m.verdict(m.compute_stats(s1, 30000, -1500, 15, SESS, {}, bt_r=BT_R)) == 'ADVANCE'


# ---------------------------------------------------------------------------
# Gate-2 additions (docs/scaling_plan_2026.md items 3 and 4) + Gate-1 freeze
# ---------------------------------------------------------------------------

def test_ex_monster_is_stage_pnl_minus_the_best_fill():
    fills = [F(SESS[0], -100), F(SESS[1], -100), F(SESS[2], 900)]
    s = stats(fills)
    assert s.pnl == 700 and s.best_fill_pnl == 900 and s.pnl_ex_monster == -200
    assert not s.above_water_ex_monster
    assert round(s.pnl_ex_monster_pct, 3) == -2.0


def test_one_monster_carrying_the_stage_blocks_advance():
    pnls = [900, -40, -40, -40, -40, 5, -40, -40]         # +$665, streak 4
    fills = [F(d, p) for d, p in zip(SESS, pnls)]
    s = stats(fills)
    assert s.pnl > 0 and s.losing_streak < 5
    assert not s.above_water_ex_monster and m.verdict(s) == 'HOLD'


def test_live_r_uses_the_fills_own_dollar_risk():
    """R = pnl / total_risk — the same 1R the BT normalizes by."""
    s = stats([F(SESS[0], 60, total_risk=100.0), F(SESS[1], -100, total_risk=100.0)])
    assert round(s.live_mean_r, 3) == -0.2
    # a fill without a usable risk figure is dropped from R, not counted as 0
    s2 = stats([F(SESS[0], 60, total_risk=100.0), F(SESS[1], 999, total_risk=0)])
    assert round(s2.live_mean_r, 3) == 0.6 and s2.band.n == 1


def test_band_in_band_advances_below_p10_does_not():
    fills = [F(d, 60) for d in SESS[:8]]
    assert stats(fills).band_status == bb.IN_BAND
    tight = stats(fills, bt_r=[1.5, 2.0] * 20)           # BT never below +1.5R
    assert tight.band_status in (bb.BELOW_P10, bb.BELOW_P5)
    assert m.verdict(tight) != 'ADVANCE'


def test_below_p5_after_eight_fills_demotes():
    pnls = [-30, 10, -30, 10, -30, 10, -30, 10]          # flat, streak 1
    fills = [F(d, p) for d, p in zip(SESS, pnls)]
    s = stats(fills, bt_r=[2.0] * 40)
    assert s.band_status == bb.BELOW_P5 and m.verdict(s) == 'DEMOTE'
    s7 = stats(fills[:7], bt_r=[2.0] * 40)
    assert s7.band_status == bb.BELOW_P5 and m.verdict(s7) == 'HOLD'


def test_above_p90_blocks_advance():
    fills = [F(d, 300) for d in SESS[:8]]                 # +3R live
    s = stats(fills, bt_r=[-1.0] * 9 + [1.0])
    assert s.band_status == bb.ABOVE_P90 and m.verdict(s) == 'HOLD'


def test_missing_bt_reference_blocks_advance():
    s = stats([F(d, 60) for d in SESS[:8]], bt_r=[])
    assert s.band_status == bb.NO_DATA and m.verdict(s) == 'HOLD'


def test_freeze_blocks_advance_regardless_of_pnl():
    fills = [F(d, 60) for d in SESS[:8]]
    assert m.verdict(stats(fills, frozen=True)) == 'HOLD'
    assert m.verdict(stats(fills)) == 'ADVANCE'


def test_frozen_sessions_excluded_from_the_stage_clock(tmp_path):
    p = tmp_path / 'freeze.json'
    rf.set_freeze('orb', 'mult drift', day=SESS[2], path=p, notify=False)
    rf.clear_freeze('orb', 'fixed', path=p, day=SESS[3])
    live = rf.unfrozen_sessions('orb', SESS, p)
    assert SESS[2] not in live and len(live) == len(SESS) - 1
    s = stats([F(d, 60) for d in SESS[:8]], sessions=len(live),
              frozen_sessions=len(SESS) - len(live))
    assert s.sessions == 14 and s.frozen_sessions == 1
    assert m.verdict(s) == 'HOLD'                        # 14 < 15 sessions


def test_reference_follows_the_catalyst_veto_state():
    assert bb.orb_reference(False).path == bb.ORB_REF_VETO_OFF
    assert bb.orb_reference(True).path == bb.ORB_REF_VETO_ON
