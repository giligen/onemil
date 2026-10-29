"""Tests for the momentum sleeve's LIVE path on the shared live account (docs/momentum_sleeve_live_spec_20261017.md).

Unit (pure helpers in trading/momentum_live.py) + integration (scripts/momentum_sleeve.run against a fake broker
with real state/ledger files in tmp_path): paper default unchanged, isolation of the owner's positions, collisions,
sell caps, buying-power refusal, drawdown pause, dry run writes nothing, MOM-LIVE line, ``moml-`` order ids.
"""
import importlib.util
import json
import os
import sys
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_sources.alpaca_client import AlpacaClient  # noqa: E402
from notifications.telegram_notifier import TelegramNotifier  # noqa: E402
from trading import momentum_live as ml  # noqa: E402
from trading import momentum_sleeve as ms  # noqa: E402

_spec = importlib.util.spec_from_file_location('momentum_sleeve_runner_live',
                                               os.path.join(ROOT, 'scripts', 'momentum_sleeve.py'))
runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = runner
_spec.loader.exec_module(runner)

sys.path.insert(0, os.path.join(ROOT, 'tests'))
from test_momentum_sleeve import make_panel  # noqa: E402

NOW = datetime(2026, 10, 5, 14, 0, tzinfo=ZoneInfo('UTC'))
PX = 50.0


class FakeBroker:
    """The trading-client surface the sleeve touches: orders fill at PX, positions/account are settable."""

    def __init__(self, positions=None, buying_power=1e6, account_number='LIVE1234'):
        self.positions = list(positions or [])
        self.orders = {}
        self.buying_power = buying_power
        self.account_number = account_number

    def submit_order(self, req):
        """Record the order as filled at PX."""
        qty = float(req.qty) if req.qty is not None else float(req.notional) / PX
        o = SimpleNamespace(client_order_id=req.client_order_id, symbol=req.symbol, status='filled',
                            filled_qty=qty, filled_avg_price=PX, side=str(req.side), qty=req.qty,
                            notional=req.notional)
        self.orders[req.client_order_id] = o
        return o

    def get_order_by_client_id(self, coid):
        """404 for an unknown id."""
        if coid not in self.orders:
            raise Exception('404 order not found')
        return self.orders[coid]

    def get_all_positions(self):
        return self.positions

    def get_account(self):
        return SimpleNamespace(account_number=self.account_number, buying_power=str(self.buying_power),
                               equity='65000')

    def get_asset(self, sym):
        return SimpleNamespace(fractionable=True)


def pos(sym, qty, px=PX, last=PX):
    """A broker position object."""
    return SimpleNamespace(symbol=sym, qty=str(qty), current_price=str(px), lastday_price=str(last))


def live_client(broker=None, paper=False):
    """MagicMock(spec=AlpacaClient) wired to a FakeBroker."""
    b = broker or FakeBroker()
    c = MagicMock(spec=AlpacaClient)
    c.is_paper = paper
    c.trading_client = b
    c._call_with_timeout = MagicMock(side_effect=RuntimeError('no data in tests'))
    c.get_asset_tolerant.side_effect = lambda s, **k: b.get_asset(s)
    cal = [{'date': d.date(), 'open': None, 'close': None} for d in pd.bdate_range('2024-01-01', '2026-12-31')]
    c.get_market_calendar.side_effect = lambda s, e: [x for x in cal if s <= x['date'] <= e]
    return c, b


@pytest.fixture
def world(monkeypatch, tmp_path):
    """Patched LIVE file paths + a 25-name panel; returns (args factory, asof)."""
    for name in ('LIVE_STATE_PATH', 'LIVE_LEDGER_PATH', 'LIVE_WEEKLY_PATH', 'LIVE_SHADOW_GATE_PATH'):
        monkeypatch.setattr(ml, name, str(tmp_path / 'live' / name.lower()))
    specs = {f'S{i:02d}': (50, 0.001 * (i + 1), 1e7, 0.01) for i in range(25)}
    specs['SPY'] = (400, 0.0005, 1e8, 0.005)
    panel, dates = make_panel(specs)
    a = pd.DataFrame({'symbol': [f'S{i:02d}' for i in range(25)], 'name': ['Co'] * 25})
    monkeypatch.setattr(runner, 'fetch_assets', lambda client: a)
    monkeypatch.setattr(runner, 'DATA_DIR', str(tmp_path))
    monkeypatch.setattr(runner, 'shadow_gate_info', lambda asof: None)
    monkeypatch.setattr(ms, 'ADV_CUTOFF', 0.0)
    asof = dates[-1].date()
    panel.to_parquet(runner.cache_file(asof), index=False)
    runner.write_completeness(asof, 26, panel, [])

    def args(submit=False, **kw):
        base = dict(force=True, submit=submit, asof=str(asof), n=20, equity_start=10000.0, skip_fetch=True,
                    gate='half', live=True, cadence_warn_weeks=8)
        base.update(kw)
        return SimpleNamespace(**base)
    yield args, asof, tmp_path
    try:
        os.remove(runner.cache_file(asof))
    except OSError:
        pass


def live_files(tmp_path):
    """Every file a LIVE run may write."""
    return [p for p in (ml.LIVE_STATE_PATH, ml.LIVE_LEDGER_PATH, ml.LIVE_WEEKLY_PATH, ml.LIVE_SHADOW_GATE_PATH)
            if os.path.exists(p)]


# ------------------------------------------------------------------ paper default unchanged / live paths
def test_paper_default_paths_and_flags_unchanged(monkeypatch):
    p = runner.mode_paths(False)
    assert p == {'state': runner.STATE_PATH, 'ledger': runner.LEDGER_PATH, 'weekly': runner.WEEKLY_PATH,
                 'gate': runner.SHADOW_GATE_PATH}
    assert runner.STATE_PATH.endswith('data/momentum_sleeve/state.json')
    assert runner.client_order_id(date(2026, 10, 5), 'AAA', 'buy') == 'mom-20261005-AAA-b'
    ns = runner.build_arg_parser().parse_args([])
    assert ns.live is False and ns.equity_start is None and ns.cadence_warn_weeks == 8
    paper = MagicMock(spec=AlpacaClient)
    paper.is_paper = True
    paper.trading_client = SimpleNamespace(_base_url='https://paper-api.alpaca.markets')
    runner.assert_paper_account(paper)            # the PA assertion stays
    paper.trading_client = SimpleNamespace(_base_url='https://api.alpaca.markets')
    with pytest.raises(RuntimeError):
        runner.assert_paper_account(paper)


def test_live_paths_are_separate_files():
    p = runner.mode_paths(True)
    assert p['state'].endswith('data/momentum_sleeve/state_live.json')
    assert p['ledger'].endswith('logs/momentum_sleeve_live_ledger.csv')
    assert p['weekly'].endswith('logs/momentum_sleeve_live_weekly.csv')
    assert p['gate'].endswith('logs/momentum_sleeve_live_shadow_gate.csv')
    assert set(p.values()).isdisjoint(runner.mode_paths(False).values())


def test_live_requires_explicit_equity_start(monkeypatch, caplog):
    monkeypatch.setattr(sys, 'argv', ['momentum_sleeve.py', '--live'])
    monkeypatch.setattr(runner, 'Config', lambda: MagicMock())
    monkeypatch.setattr(runner, 'AlpacaClient', lambda *a, **k: pytest.fail('no client without --equity-start'))
    assert runner.main() == 1 and '--equity-start' in caplog.text


def test_non_pa_assertion():
    c, b = live_client()
    runner.assert_live_account(c)                          # LIVE1234 passes
    b.account_number = 'PA3XYZ'
    with pytest.raises(RuntimeError, match='paper'):
        runner.assert_live_account(c)
    c2, _ = live_client(paper=True)
    with pytest.raises(RuntimeError):
        runner.assert_live_account(c2)


# ------------------------------------------------------------------ isolation
def test_broker_positions_outside_state_are_invisible(caplog):
    allq = {'OWN1': 10.0, 'MINE': 5.0, 'OWN2': 3.0}
    view, ignored = ml.sleeve_view(allq, ['MINE'])
    assert view == {'MINE': 5.0} and ignored == 2
    c, b = live_client(FakeBroker([pos('OWN1', 10, 999), pos('MINE', 5, 60)]))
    assert runner.broker_marks(c, only={'MINE'}) == {'MINE': 60.0}
    state = {'cash': 0.0, 'positions': {'MINE': 5.0}, 'equity_start': 10000.0}
    orders = [{'symbol': 'OWN1', 'side': 'sell', 'notional': 500.0, 'full_exit': True},
              {'symbol': 'MINE', 'side': 'sell', 'notional': 250.0, 'full_exit': True}]
    with caplog.at_level('INFO'):
        fills = runner.execute_live(c, orders, {'OWN1': PX, 'MINE': PX}, state, date(2026, 10, 5))
    assert [f['symbol'] for f in fills] == ['MINE']                      # the owner's OWN1 is never sold
    assert not any('OWN1' in k for k in b.orders)


def test_dry_run_reports_count_only(world, capsys, caplog):
    args, asof, tmp = world
    c, b = live_client(FakeBroker([pos('OWNZ', 100, 400), pos('OWNY', 5)]))
    with caplog.at_level('DEBUG'):
        rc = runner.run(args(), c, None, NOW)
    out = capsys.readouterr().out
    assert rc == 0 and '2 non-sleeve positions ignored' in out and 'LIVE PLAN' in out
    assert 'buying power' in out and 'TOP:' in out and 'BUY' in out
    assert 'OWNZ' not in out + caplog.text and 'OWNY' not in out + caplog.text


def test_dry_run_writes_nothing(world):
    args, asof, tmp = world
    c, b = live_client()
    assert runner.run(args(), c, None, NOW) == 0
    assert live_files(tmp) == [] and not os.path.exists(os.path.dirname(ml.LIVE_STATE_PATH))
    assert b.orders == {}


def test_buy_collision_skipped_with_message(world):
    args, asof, tmp = world
    c, b = live_client(FakeBroker([pos('S24', 7), pos('S23', 2)]))      # held manually, top-ranked, not in state
    notifier = MagicMock(spec=TelegramNotifier)
    assert runner.run(args(submit=True), c, notifier, NOW) == 0
    buys = [k for k in b.orders if k.endswith('-b') or '-b-' in k]
    assert not any('S24' in k or 'S23' in k for k in buys) and len(buys) == 18
    msg = notifier.send_message_sync.call_args[0][0]
    assert msg.startswith('[MOM LIVE]')
    assert 'COLLISION S24: held manually — skipped' in msg and 'COLLISION S23: held manually — skipped' in msg
    state = json.load(open(ml.LIVE_STATE_PATH))
    assert 'S24' not in state['positions'] and len(state['positions']) == 18
    assert state['cash'] == pytest.approx(10000 - 18 * 500, abs=1.0)   # the two collided slots stay cash
    assert all(k.startswith('moml-') for k in b.orders)


def test_sell_capped_at_state_qty_and_reduced_to_broker(caplog):
    orders = [{'symbol': 'AAA', 'side': 'sell', 'notional': 1000.0, 'full_exit': True},
              {'symbol': 'BBB', 'side': 'sell', 'notional': 1000.0, 'full_exit': True}]
    state = {'cash': 0.0, 'positions': {'AAA': 10.0, 'BBB': 10.0}}
    c, b = live_client(FakeBroker([pos('AAA', 15), pos('BBB', 4)]))     # AAA: owner added 5; BBB: only 4 there
    notes = []
    with caplog.at_level('WARNING'):
        fills = runner.execute_live(c, orders, {'AAA': PX, 'BBB': PX}, state, date(2026, 10, 5), notes=notes)
    q = {f['symbol']: f['qty'] for f in fills}
    assert q == {'AAA': 10.0, 'BBB': 4.0}
    assert 'more than the sleeve state' in caplog.text and 'less than the sleeve state' in caplog.text
    assert len(notes) == 2 and state['positions'] == {}                 # BBB state qty set to 4 then sold to 0
    assert state['cash'] == pytest.approx(14 * PX)


def test_buying_power_refusal_submits_no_buys(world, caplog):
    args, asof, tmp = world
    c, b = live_client(FakeBroker(buying_power=100.0))
    state = {'cash': 10000.0, 'positions': {}, 'equity_start': 10000.0}
    notifier = MagicMock(spec=TelegramNotifier)
    with caplog.at_level('ERROR'):
        rc = runner.run(args(submit=True), c, notifier, NOW)
    assert rc == 0 and b.orders == {}
    assert 'buying power' in caplog.text and 'NO buys' in caplog.text
    assert 'NO buys' in notifier.send_message_sync.call_args[0][0]
    st = json.load(open(ml.LIVE_STATE_PATH))
    assert st['positions'] == {} and st['cash'] == 10000.0


def test_drawdown_rail_liquidates_pauses_and_refuses_next_run(world, capsys):
    args, asof, tmp = world
    os.makedirs(os.path.dirname(ml.LIVE_STATE_PATH))
    json.dump({'cash': 0.0, 'positions': {'S00': 50.0, 'S01': 50.0}, 'peak_equity': 10000.0,
               'equity_start': 10000.0, 'last_rebalance': None}, open(ml.LIVE_STATE_PATH, 'w'))   # 5000 <= 7500
    c, b = live_client()
    notifier = MagicMock(spec=TelegramNotifier)
    assert runner.run(args(submit=True), c, notifier, NOW) == 0
    st = json.load(open(ml.LIVE_STATE_PATH))
    assert st['paused'] is True and 'drawdown rail' in st['paused_reason'] and st['positions'] == {}
    assert all('-s' in k for k in b.orders)                                  # sells only, no buys
    assert 'PAUSED' in notifier.send_message_sync.call_args[0][0]
    n_orders = len(b.orders)
    assert runner.run(args(submit=True), c, notifier, NOW) == 3             # refuses until the flag is cleared
    assert len(b.orders) == n_orders and 'PAUSED' in capsys.readouterr().out
    assert 'state_live.json' in ml.PAUSE_CLEAR_CMD and 'paused' in ml.PAUSE_CLEAR_CMD


def test_full_live_rotation_books_state_files_and_coids(world):
    args, asof, tmp = world
    c, b = live_client(FakeBroker([pos('OWNQ', 3, 300)]))
    notifier = MagicMock(spec=TelegramNotifier)
    assert runner.run(args(submit=True), c, notifier, NOW) == 0
    st = json.load(open(ml.LIVE_STATE_PATH))
    assert len(st['positions']) == 20 and 'OWNQ' not in st['positions']
    assert st['cash'] == pytest.approx(0.0, abs=1.0) and st['equity_start'] == 10000.0
    assert all(k.startswith('moml-20261005-') for k in b.orders)
    w = ml.read_weekly(ml.LIVE_WEEKLY_PATH)
    assert len(w) == 1 and w[0]['weeks_since_strong'] == '0' and 'week_r' in w[0]
    assert len(ml.read_weekly(ml.LIVE_LEDGER_PATH)) == 20
    assert 'OWNQ' not in open(ml.LIVE_LEDGER_PATH).read()
    assert notifier.send_message_sync.call_args[0][0].startswith('[MOM LIVE]')


# ------------------------------------------------------------------ pure helpers
def test_cadence_and_week_r():
    rows = [{'date': '2026-09-07', 'equity': '10600'}, {'date': '2026-09-14', 'equity': '10550'}]
    assert [round(r, 1) for _, r in ml.week_r_series(rows, 10000)] == [6.0, -0.5]       # R = $100
    cur, weeks = ml.cadence(rows, 10000, date(2026, 10, 12), new_equity=10560)
    assert cur == pytest.approx(0.1) and weeks == 5                                       # last strong 9/7
    _, w0 = ml.cadence([], 10000, date(2026, 10, 12))
    assert w0 == 0
    cur, weeks = ml.cadence(rows, 10000, date(2026, 10, 12), new_equity=11600)
    assert cur == pytest.approx(10.5) and weeks == 0


def test_apply_fills_state_tracked_cash():
    st = {'cash': 1000.0, 'positions': {'A': 2.0}}
    ml.apply_fills(st, [{'symbol': 'A', 'side': 'sell', 'qty': 2.0, 'avg_price': 50.0},
                        {'symbol': 'B', 'side': 'buy', 'qty': 4.0, 'avg_price': 25.0}])
    assert st == {'cash': 1000.0, 'positions': {'B': 4.0}}


# ------------------------------------------------------------------ reporting
def test_mom_live_line_reads_only_sleeve_names(tmp_path):
    state = {'cash': 100.0, 'positions': {'AAA': 10.0}, 'equity_start': 1000.0, 'peak_equity': 1100.0}
    ps = [pos('AAA', 10, 105, 100), pos('OWNER', 5000, 9999, 1)]
    marks, prev = ml.sleeve_marks(ps, state)
    assert marks == {'AAA': 105.0} and prev == {'AAA': 100.0}
    line = ml.mom_live_line(state, marks, prev, [{'date': '2026-10-02', 'equity': '1000'}], date(2026, 10, 7))
    assert line.startswith('MOM-LIVE:') and 'day $+50' in line and 'week $+150' in line
    assert 'since start $+150' in line and 'slice $1,150' in line and 'DD from peak 0.0' in line
    assert 'OWNER' not in line and '9999' not in line and '65000' not in line


def test_hourly_collect_filters_and_absent_state_is_omitted(tmp_path):
    import hourly_summary as hs
    c, b = live_client(FakeBroker([pos('AAA', 10, 105, 100), pos('OWNER', 5, 9999, 1)]))
    assert hs.collect_mom_live(c, date(2026, 10, 7), state_path=str(tmp_path / 'none.json')) is None
    sp = tmp_path / 'state_live.json'
    sp.write_text(json.dumps({'cash': 0.0, 'positions': {'AAA': 10.0}, 'equity_start': 1000.0}))
    d = hs.collect_mom_live(c, date(2026, 10, 7), state_path=str(sp), weekly_path=str(tmp_path / 'w.csv'))
    assert d['marks'] == {'AAA': 105.0} and 'OWNER' not in json.dumps(d, default=str)


def test_eod_mom_live_lines(tmp_path):
    import eod_sections as es
    assert es.mom_live_lines('2026-10-07', client=MagicMock(), state_path=str(tmp_path / 'none.json')) == []
    sp = tmp_path / 's.json'
    sp.write_text(json.dumps({'cash': 100.0, 'positions': {'AAA': 10.0}, 'equity_start': 1000.0}))
    c, b = live_client(FakeBroker([pos('AAA', 10, 105, 100), pos('OWNER', 5, 9999, 1)]))
    lines = es.mom_live_lines('2026-10-07', client=c, state_path=str(sp), weekly_path=str(tmp_path / 'w.csv'))
    assert len(lines) == 1 and lines[0].startswith('MOM-LIVE:') and 'OWNER' not in lines[0]
    c.trading_client.get_all_positions = MagicMock(side_effect=RuntimeError('boom'))
    assert 'NO-DATA' in es.mom_live_lines('2026-10-07', client=c, state_path=str(sp))[0]


def test_hourly_message_includes_mom_live_line_only_when_state_exists():
    import hourly_summary as hs
    now = datetime(2026, 10, 7, 11, 0, tzinfo=ZoneInfo('US/Eastern'))
    live = {'account': {'equity': 65000.0, 'last_equity': 64900.0}}
    base = {'ORB': None, 'HOD': None, 'MOM': None, 'LIVE': live}
    assert 'MOM-LIVE' not in hs.build_message(now, dict(base), {})            # absent state: omitted, not NO-DATA
    ml_data = {'state': {'cash': 100.0, 'positions': {'AAA': 10.0}, 'equity_start': 1000.0}, 'marks': {'AAA': 105.0},
               'prev': {'AAA': 100.0}, 'weekly': [], 'day': now.date()}
    msg = hs.build_message(now, {**base, 'MOM-LIVE': ml_data}, {})
    assert '<b>MOM-LIVE:' in msg and 'slice $1,150' in msg and '65,000' not in msg.split('MOM-LIVE')[1]
    assert '<b>MOM-LIVE</b> n/a' in hs.build_message(now, {**base, 'MOM-LIVE': None}, {'MOM-LIVE': 'api error'})
