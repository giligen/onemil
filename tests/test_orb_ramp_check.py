"""ORB budget ramp checker (owner adopted 2026-09-07): pure gate logic."""
import importlib.util, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('orb_ramp_check', os.path.join(ROOT, 'scripts', 'orb_ramp_check.py'))
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)

SESS = [f'2026-09-{d:02d}' for d in (8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 28)]   # 15 sessions


def F(day, pnl, slip_bps=20.0):
    trig = 10.0
    return {'trade_date': day, 'symbol': 'X', 'pnl': pnl, 'entry_price': trig,
            'fill_price': trig * (1 + slip_bps / 1e4), 'exited_at': f'{day}T15:00:00'}


def stats(fills, sessions=15, parity=None, budget=10000, limit=-750):
    return m.compute_stats(fills, budget, limit, sessions, SESS[:sessions], parity or {})


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
    assert m.verdict(stats(s0)) == m.verdict(m.compute_stats(s1, 30000, -1500, 15, SESS, {})) == 'ADVANCE'
