"""BF P1 ramp checker: pure gate logic (docs/bf_p1_ramp.md + scaling_plan_2026 Gate 2)."""
import importlib.util, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location('bf_ramp_check', os.path.join(ROOT, 'scripts', 'bf_ramp_check.py'))
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)

from trading import ramp_bt_band as bb          # noqa: E402
from trading import ramp_freeze as rf           # noqa: E402

# Synthetic BT reference (R = pnl/$2K in the real VOL_OFF book): 7 full losers
# and a fat right tail, so a +2R live sample of 8 is IN-BAND and a flat one is
# not yet below p5. The real distribution is loaded by main().
BT_R = [-1.0] * 7 + [2.0, 8.0, 16.0]


def cs(trades, base, sessions, bt_r=BT_R, **kw):
    """compute_stats with the BT band scored (an unscored band blocks ADVANCE)."""
    return m.compute_stats(trades, base, sessions, bt_r=bt_r, **kw)


def T(day, pnl, flag=0):
    return {'trade_date': day, 'symbol': 'X', 'pnl': pnl, 'exited_at': f'{day}T15:00:00', 'exit_pending_verification': flag}


def test_stage_lookup():
    assert m.stage_for_risk(150)['name'] == 'L0' and m.next_stage(m.stage_for_risk(150))['risk'] == 400
    assert m.next_stage(m.stage_for_risk(2000)) is None
    assert m.stage_for_risk(60)['name'].startswith('custom')


def test_advance_needs_positive_eight_trades_fifteen_sessions():
    base = 150
    wins = [T(f'2026-09-{d:02d}', 300) for d in (8, 9, 10, 14, 15, 16, 17, 18)]   # 8 trades, +16u
    s = cs(wins, base, sessions=15)
    assert m.verdict(s) == 'ADVANCE'
    assert m.verdict(cs(wins, base, sessions=14)) == 'HOLD'          # sessions
    small = [T(f'2026-09-{d:02d}', 40) for d in (8, 9, 10, 14, 15, 16, 17)]      # 7 trades, +1.9u: not 8, not the early read
    assert m.verdict(cs(small, base, sessions=15)) == 'HOLD'


def test_early_read_six_trades_plus_four_u():
    base = 150
    six = [T(f'2026-09-{d:02d}', 150) for d in (8, 9, 10, 14, 15, 16)]            # +6u on 6 trades
    assert m.verdict(cs(six, base, sessions=15)) == 'ADVANCE'
    weak = [T(f'2026-09-{d:02d}', 50) for d in (8, 9, 10, 14, 15, 16)]            # +2u on 6 trades
    assert m.verdict(cs(weak, base, sessions=15)) == 'HOLD'


def test_negative_stage_never_advances():
    base = 150
    tr = [T(f'2026-09-{d:02d}', 300) for d in (8, 9, 10, 14, 15, 16, 17)] + [T('2026-09-18', -2200)]
    s = cs(tr, base, sessions=20)
    assert s.pnl < 0 and m.verdict(s) == 'HOLD'


def test_demote_on_minus_six_u_or_streak_or_weekly_rail():
    base = 150
    losers = [T(f'2026-09-{d:02d}', -200) for d in (8, 9, 10, 14, 15)]            # -6.7u, streak 5
    assert m.verdict(cs(losers, base, sessions=6)) == 'DEMOTE'
    streak = [T(f'2026-09-{d:02d}', -100) for d in (8, 9, 10, 14, 15)]            # -3.3u but 5 in a row
    assert m.verdict(cs(streak, base, sessions=6)) == 'DEMOTE'
    week = [T('2026-09-08', -600), T('2026-09-09', -500)]                          # -1100 in one week = -7.3u
    s = cs(week, base, sessions=2)
    assert s.weekly_rail_hit and m.verdict(s) == 'DEMOTE'


def test_pause_on_minus_eight_u():
    assert m.verdict(cs([T('2026-09-08', -700), T('2026-09-15', -600)], 150, sessions=6)) == 'PAUSE'


def test_parity_flag_and_daily_rail_block_advance():
    base = 150
    wins = [T(f'2026-09-{d:02d}', 300) for d in (8, 9, 10, 14, 15, 16, 17, 18)]
    flagged = wins[:-1] + [T('2026-09-18', 300, flag=1)]
    assert m.verdict(cs(flagged, base, sessions=15)) == 'HOLD'
    railed = wins + [T('2026-09-21', -800)]                                        # one -5.3u day, still +10.7u
    s = cs(railed, base, sessions=16)
    assert s.daily_rail_hits == 1 and m.verdict(s) == 'HOLD'


def test_units_scale_with_stage():
    """Same trades in u give the same verdict at L0 and L2."""
    l0 = [T(f'2026-09-{d:02d}', 150 * 2) for d in (8, 9, 10, 14, 15, 16, 17, 18)]
    l2 = [T(f'2026-09-{d:02d}', 1000 * 2) for d in (8, 9, 10, 14, 15, 16, 17, 18)]
    assert m.verdict(cs(l0, 150, 15)) == m.verdict(cs(l2, 1000, 15)) == 'ADVANCE'


# ---------------------------------------------------------------------------
# Gate-2 additions (docs/scaling_plan_2026.md items 3 and 4) + Gate-1 freeze
# ---------------------------------------------------------------------------

def wins8(pnl=300):
    return [T(f'2026-09-{d:02d}', pnl) for d in (8, 9, 10, 14, 15, 16, 17, 18)]


def test_ex_monster_is_stage_pnl_minus_the_best_trade():
    tr = [T('2026-09-08', -100), T('2026-09-09', -100), T('2026-09-10', 900)]
    s = cs(tr, 150, sessions=15)
    assert s.pnl == 700 and s.best_trade_pnl == 900
    assert s.pnl_ex_monster == -200 and not s.above_water_ex_monster
    assert round(s.pnl_ex_monster_u, 3) == round(-200 / 150, 3)


def test_one_monster_carrying_the_stage_blocks_advance():
    """+8R on one trade, seven small losers: positive stage, not a stage."""
    pnls = [1200, -60, -60, -60, -60, 5, -60, -60, -60]   # streak 4, +5.2u
    tr = [T(f'2026-09-{d:02d}', p)
          for d, p in zip((8, 9, 10, 11, 14, 15, 16, 17, 18), pnls)]
    s = cs(tr, 150, sessions=20)
    assert s.pnl > 0 and s.losing_streak < 5 and not s.above_water_ex_monster
    assert s.band_status == bb.IN_BAND and m.verdict(s) == 'HOLD'


def test_band_status_and_gates():
    base = 150
    # IN-BAND (+2R x 8) advances; a flat book is below p10 and must not
    below = [T(f'2026-09-{d:02d}', 150) for d in (8, 9, 10, 14, 15, 16, 17, 18)]
    assert cs(wins8(), base, sessions=15).band_status == bb.IN_BAND
    # +1R live against a backtest that never does worse than +1.5R
    s_below = cs(below, base, sessions=15, bt_r=[1.5, 2.0] * 20)
    assert s_below.band_status in (bb.BELOW_P10, bb.BELOW_P5)
    assert m.verdict(s_below) != 'ADVANCE'


def test_below_p5_after_eight_trades_demotes():
    pnls = [-30, 10, -30, 10, -30, 10, -30, 10]          # flat-ish, streak 1
    flat = [T(f'2026-09-{d:02d}', p)
            for d, p in zip((8, 9, 10, 11, 14, 15, 16, 17), pnls)]
    tight = [2.0] * 40                     # BT says +2R every time
    s = cs(flat, 150, sessions=20, bt_r=tight)
    assert s.band_status == bb.BELOW_P5 and m.verdict(s) == 'DEMOTE'
    # under 8 trades the same read is only a HOLD (n too small to conclude)
    s7 = cs(flat[:7], 150, sessions=20, bt_r=tight)
    assert s7.band_status == bb.BELOW_P5 and m.verdict(s7) == 'HOLD'


def test_above_p90_blocks_advance():
    """A live book beating its own backtest is a leak or a bug before it is luck."""
    s = cs(wins8(), 150, sessions=15, bt_r=[-1.0] * 9 + [1.0])
    assert s.band_status == bb.ABOVE_P90 and m.verdict(s) == 'HOLD'


def test_missing_bt_reference_blocks_advance():
    s = cs(wins8(), 150, sessions=15, bt_r=[])
    assert s.band_status == bb.NO_DATA and m.verdict(s) == 'HOLD'


def test_freeze_blocks_advance_regardless_of_pnl():
    s = cs(wins8(), 150, sessions=15, frozen=True)
    assert s.band_status == bb.IN_BAND and s.above_water_ex_monster
    assert m.verdict(s) == 'HOLD'
    assert m.verdict(cs(wins8(), 150, sessions=15, frozen=False)) == 'ADVANCE'


def test_frozen_sessions_do_not_count_toward_the_minimum(tmp_path, monkeypatch):
    p = tmp_path / 'freeze.json'
    rf.set_freeze('bf', 'breach', day='2026-09-16', path=p, notify=False)
    rf.clear_freeze('bf', 'fixed', path=p, day='2026-09-17')
    sessions = m.session_dates('2026-09-14', today=__import__('datetime').date(2026, 9, 18))
    assert sessions == ['2026-09-14', '2026-09-15', '2026-09-16', '2026-09-17', '2026-09-18']
    live = rf.unfrozen_sessions('bf', sessions, p)
    assert len(live) == 4                              # the frozen day is gone
    s = cs(wins8(), 150, sessions=len(live), frozen_sessions=1)
    assert s.sessions == 4 and s.frozen_sessions == 1


def test_session_count_matches_session_dates():
    assert m.session_count('2026-09-14') == len(m.session_dates('2026-09-14'))
