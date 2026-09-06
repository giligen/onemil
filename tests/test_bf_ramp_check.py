"""BF P1 ramp checker: pure gate logic (docs/bf_p1_ramp.md)."""
import importlib.util, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('bf_ramp_check', os.path.join(ROOT, 'scripts', 'bf_ramp_check.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def T(day, pnl, flag=0):
    return {'trade_date': day, 'symbol': 'X', 'pnl': pnl, 'exited_at': f'{day}T15:00:00', 'exit_pending_verification': flag}


def test_stage_lookup():
    assert m.stage_for_risk(150)['name'] == 'L0' and m.next_stage(m.stage_for_risk(150))['risk'] == 400
    assert m.next_stage(m.stage_for_risk(2000)) is None
    assert m.stage_for_risk(60)['name'].startswith('custom')


def test_advance_needs_positive_eight_trades_fifteen_sessions():
    base = 150
    wins = [T(f'2026-09-{d:02d}', 300) for d in (8, 9, 10, 14, 15, 16, 17, 18)]   # 8 trades, +16u
    s = m.compute_stats(wins, base, sessions=15)
    assert m.verdict(s) == 'ADVANCE'
    assert m.verdict(m.compute_stats(wins, base, sessions=14)) == 'HOLD'          # sessions
    small = [T(f'2026-09-{d:02d}', 40) for d in (8, 9, 10, 14, 15, 16, 17)]      # 7 trades, +1.9u: not 8, not the early read
    assert m.verdict(m.compute_stats(small, base, sessions=15)) == 'HOLD'


def test_early_read_six_trades_plus_four_u():
    base = 150
    six = [T(f'2026-09-{d:02d}', 150) for d in (8, 9, 10, 14, 15, 16)]            # +6u on 6 trades
    assert m.verdict(m.compute_stats(six, base, sessions=15)) == 'ADVANCE'
    weak = [T(f'2026-09-{d:02d}', 50) for d in (8, 9, 10, 14, 15, 16)]            # +2u on 6 trades
    assert m.verdict(m.compute_stats(weak, base, sessions=15)) == 'HOLD'


def test_negative_stage_never_advances():
    base = 150
    tr = [T(f'2026-09-{d:02d}', 300) for d in (8, 9, 10, 14, 15, 16, 17)] + [T('2026-09-18', -2200)]
    s = m.compute_stats(tr, base, sessions=20)
    assert s.pnl < 0 and m.verdict(s) == 'HOLD'


def test_demote_on_minus_six_u_or_streak_or_weekly_rail():
    base = 150
    losers = [T(f'2026-09-{d:02d}', -200) for d in (8, 9, 10, 14, 15)]            # -6.7u, streak 5
    assert m.verdict(m.compute_stats(losers, base, sessions=6)) == 'DEMOTE'
    streak = [T(f'2026-09-{d:02d}', -100) for d in (8, 9, 10, 14, 15)]            # -3.3u but 5 in a row
    assert m.verdict(m.compute_stats(streak, base, sessions=6)) == 'DEMOTE'
    week = [T('2026-09-08', -600), T('2026-09-09', -500)]                          # -1100 in one week = -7.3u
    s = m.compute_stats(week, base, sessions=2)
    assert s.weekly_rail_hit and m.verdict(s) == 'DEMOTE'


def test_pause_on_minus_eight_u():
    assert m.verdict(m.compute_stats([T('2026-09-08', -700), T('2026-09-15', -600)], 150, sessions=6)) == 'PAUSE'


def test_parity_flag_and_daily_rail_block_advance():
    base = 150
    wins = [T(f'2026-09-{d:02d}', 300) for d in (8, 9, 10, 14, 15, 16, 17, 18)]
    flagged = wins[:-1] + [T('2026-09-18', 300, flag=1)]
    assert m.verdict(m.compute_stats(flagged, base, sessions=15)) == 'HOLD'
    railed = wins + [T('2026-09-21', -800)]                                        # one -5.3u day, still +10.7u
    s = m.compute_stats(railed, base, sessions=16)
    assert s.daily_rail_hits == 1 and m.verdict(s) == 'HOLD'


def test_units_scale_with_stage():
    """Same trades in u give the same verdict at L0 and L2."""
    l0 = [T(f'2026-09-{d:02d}', 150 * 2) for d in (8, 9, 10, 14, 15, 16, 17, 18)]
    l2 = [T(f'2026-09-{d:02d}', 1000 * 2) for d in (8, 9, 10, 14, 15, 16, 17, 18)]
    assert m.verdict(m.compute_stats(l0, 150, 15)) == m.verdict(m.compute_stats(l2, 1000, 15)) == 'ADVANCE'
