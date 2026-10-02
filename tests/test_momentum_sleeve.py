"""Tests for trading/momentum_sleeve.py (pure selection) and scripts/momentum_sleeve.py (runner).

Includes a PARITY test against research/momentum_weekly/recon/A_holdings.csv (skipped if absent).
"""
import importlib.util
import os
import sys
from datetime import date, datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_sources.alpaca_client import AlpacaClient  # noqa: E402
from trading import momentum_sleeve as ms  # noqa: E402

spec = importlib.util.spec_from_file_location('momentum_sleeve_runner',
                                              os.path.join(ROOT, 'scripts', 'momentum_sleeve.py'))
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

PANEL = os.path.join(ROOT, 'research', 'momentum_weekly', 'panel_2016_2026.parquet')
HOLD = os.path.join(ROOT, 'research', 'momentum_weekly', 'recon', 'A_holdings.csv')
ASSETS = os.path.join(ROOT, 'research', 'momentum_weekly', '1700c_assets.csv')


def make_panel(specs, n_days=300):
    """specs: {symbol: (start_close, daily_drift, volume, noise)} -> long panel of n_days rows."""
    dates = pd.bdate_range('2024-01-01', periods=n_days)
    rng = np.random.default_rng(1)
    rows = []
    for sym, (c0, drift, vol, noise) in specs.items():
        px = c0 * np.cumprod(1 + drift + noise * rng.standard_normal(n_days))
        rows.append(pd.DataFrame({'symbol': sym, 'bar_date': dates, 'close': px.astype('float32'),
                                  'volume': float(vol)}))
    return pd.concat(rows, ignore_index=True), dates


def assets(**names):
    return pd.DataFrame({'symbol': list(names), 'name': list(names.values())})


# ------------------------------------------------------------------ pure functions

def test_name_exclusion_word_boundary():
    a = assets(NFLX='NETFLIX INC', SPYX='SPDR S&P 500 ETF TRUST', BOGUS='Foo Warrants', ZAZZT='Test Inc',
               ZVZZT='NASDAQ TEST STOCK', PREF='Bar Preferred Stock', OKAY='Unitedhealth Group')
    ex = ms.excluded_symbols(a)
    assert 'NFLX' not in ex          # 'etf' substring inside NETFLIX must NOT match
    assert 'OKAY' not in ex          # 'unit' inside UNITEDHEALTH must not match
    assert {'SPYX', 'BOGUS', 'PREF', 'ZVZZT'} <= ex


def test_signal_formula_and_history_gate():
    panel, dates = make_panel({'AAA': (50, 0.002, 1e7, 0.01), 'SHORT': (50, 0.002, 1e7, 0.01)})
    panel = panel[~((panel.symbol == 'SHORT') & (panel.bar_date < dates[100]))]   # only 200 rows
    asof = dates[-1]
    f = ms.risk_adjusted_momentum(panel, asof)
    s = panel[panel.symbol == 'AAA'].sort_values('bar_date')['close'].astype('float64').to_numpy()
    mom = s[-1 - 21] / s[-1 - 252] - 1
    rets = pd.Series(panel[panel.symbol == 'AAA'].sort_values('bar_date')['close'].pct_change().to_numpy())
    vol = rets.iloc[-252:].std()
    assert f.loc['AAA', 'sig_12_1'] == pytest.approx(mom, rel=1e-6)
    assert f.loc['AAA', 'signal'] == pytest.approx(mom / vol, rel=1e-4)
    assert bool(f.loc['AAA', 'history_ok']) and not bool(f.loc['SHORT', 'history_ok'])


def test_eligible_universe_filters():
    panel, dates = make_panel({'BIG': (50, 0.002, 1e7, 0.01),      # $500M/day
                               'CHEAP': (5, 0.002, 1e9, 0.01),     # price < 10
                               'THIN': (50, 0.002, 1e5, 0.01),     # adv < 200M
                               'ETFX': (50, 0.002, 1e7, 0.01),     # excluded by name
                               'ZAZZT': (50, 0.002, 1e7, 0.01)})   # placeholder, name ok
    a = assets(BIG='Big Co', CHEAP='Cheap Co', THIN='Thin Co', ETFX='Some ETF', ZAZZT='Test')
    assert 'BIG' in ms.eligible_universe(panel, dates[-1], a)
    got = set(ms.eligible_universe(panel, dates[-1], a))
    assert 'CHEAP' not in got and 'THIN' not in got and 'ETFX' not in got


def test_symbol_without_asof_bar_is_dropped():
    panel, dates = make_panel({'BIG': (50, 0.002, 1e7, 0.01), 'STALE': (50, 0.002, 1e7, 0.01)})
    panel = panel[~((panel.symbol == 'STALE') & (panel.bar_date == dates[-1]))]
    f = ms.risk_adjusted_momentum(panel, dates[-1])
    assert 'STALE' not in f.index and 'BIG' in f.index


def test_select_top_order_ties_and_short_pool(caplog):
    s = pd.Series({'B': 1.0, 'A': 1.0, 'C': 2.0, 'D': np.nan})
    assert ms.select_top(s, 2) == ['C', 'A']       # tie broken by symbol
    with caplog.at_level('WARNING'):
        assert len(ms.select_top(s, 5)) == 3
    assert 'only 3 eligible' in caplog.text


def test_rebalance_resets_drift_and_sells_first():
    cur = {'A': 1500.0, 'B': 500.0, 'LEAVER': 1000.0, 'KEEP': 1000.0}
    tgt = {'A': 1000.0, 'B': 1000.0, 'KEEP': 1000.0, 'NEW': 1000.0}
    o = ms.rebalance_orders(cur, tgt)
    sides = [x['side'] for x in o]
    assert sides == sorted(sides, key=lambda s: s != 'sell')   # all sells before buys
    d = {x['symbol']: x for x in o}
    assert d['LEAVER']['full_exit'] and d['LEAVER']['notional'] == 1000.0
    assert d['A']['side'] == 'sell' and d['A']['notional'] == pytest.approx(500.0)
    assert d['B']['side'] == 'buy' and d['NEW']['side'] == 'buy'
    assert 'KEEP' not in d


def test_rebalance_min_trade_threshold_and_weights():
    assert ms.rebalance_orders({'A': 1002.0}, {'A': 1000.0}, min_trade_usd=5.0) == []
    assert ms.target_weights(['A', 'B'], 20) == {'A': 0.05, 'B': 0.05}
    assert ms.target_dollars(['A'], 20000.0, 20) == {'A': 1000.0}


# ------------------------------------------------------------------ parity vs build A

@pytest.mark.skipif(not (os.path.exists(PANEL) and os.path.exists(HOLD) and os.path.exists(ASSETS)),
                    reason='research panel / build A holdings not present')
@pytest.mark.parametrize('target', ['2021-02-08', '2024-06-03', '2026-09-21'])
def test_parity_with_build_a(target):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    hold = pd.read_csv(HOLD)
    dates = sorted(hold.rebalance_date.unique())
    reb = min(dates, key=lambda d: abs(pd.Timestamp(d) - pd.Timestamp(target)))
    reb_ts = pd.Timestamp(reb)
    lo, hi = reb_ts - pd.Timedelta(days=460), reb_ts
    tbl = pq.read_table(PANEL, columns=['symbol', 'bar_date', 'close', 'volume'],
                        filters=[('bar_date', '>=', lo.to_pydatetime()), ('bar_date', '<', hi.to_pydatetime())])
    panel = tbl.to_pandas()
    panel['close'] = panel['close'].astype('float32')
    asof = panel.loc[panel.symbol == 'SPY', 'bar_date'].max()      # prior trading day
    assets_df = pd.read_csv(ASSETS, dtype={'symbol': str, 'name': str})
    sel = ms.select_top(ms.risk_adjusted_momentum(panel, asof).loc[
        ms.eligible_universe(panel, asof, assets_df), 'signal'], 20)
    expected = set(hold.loc[hold.rebalance_date == reb, 'symbol'])
    print(f'PARITY {reb}: {len(set(sel) & expected)}/20')
    assert len(set(sel) & expected) >= 19      # a float32/tie difference may swap one name at most


# ------------------------------------------------------------------ runner

def test_client_order_id_and_calendar_helpers():
    assert runner.client_order_id(date(2026, 10, 5), 'AAPL', 'sell') == 'mom-20261005-AAPL-s'
    assert runner.client_order_id(date(2026, 10, 5), 'AAPL', 'buy') == 'mom-20261005-AAPL-b'
    with pytest.raises(ValueError):
        runner.client_order_id(date(2026, 10, 5), 'AAPL', 'x')
    assert runner.is_rebalance_session(date(2026, 10, 6), [date(2026, 10, 6), date(2026, 10, 7)])  # Mon holiday
    assert not runner.is_rebalance_session(date(2026, 10, 7), [date(2026, 10, 5), date(2026, 10, 7)])
    et = ZoneInfo('US/Eastern')
    with pytest.raises(RuntimeError):
        runner.check_submit_window(datetime(2026, 10, 5, 9, 0, tzinfo=et), True)
    with pytest.raises(RuntimeError):
        runner.check_submit_window(datetime(2026, 10, 5, 10, 0, tzinfo=et), False)
    runner.check_submit_window(datetime(2026, 10, 5, 9, 45, tzinfo=et), True)


def fake_client(paper=True, existing=None):
    c = MagicMock(spec=AlpacaClient)
    c.is_paper = paper
    c.trading_client = MagicMock()
    c.trading_client._base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
    existing = existing or {}

    def by_coid(coid):
        if coid in existing:
            return existing[coid]
        raise Exception('404 order not found')
    c.trading_client.get_order_by_client_id.side_effect = by_coid
    c.trading_client.get_all_positions.return_value = []
    return c


def test_assert_paper_refuses_live():
    with pytest.raises(RuntimeError):
        runner.assert_paper_account(fake_client(paper=False))


def test_submit_market_is_idempotent_and_dry_submits_nothing():
    o = MagicMock()
    c = fake_client(existing={'mom-20261005-AAA-b': o})
    assert runner.submit_market(c, 'AAA', 'buy', 'mom-20261005-AAA-b', notional=1000) is o
    c.trading_client.submit_order.assert_not_called()
    runner.submit_market(c, 'BBB', 'buy', 'mom-20261005-BBB-b', notional=1000)
    req = c.trading_client.submit_order.call_args[0][0]
    assert req.client_order_id == 'mom-20261005-BBB-b' and req.notional == 1000


def test_execute_sells_before_buys_and_updates_state(monkeypatch):
    c = fake_client()
    c.trading_client.get_all_positions.return_value = [MagicMock(symbol='OLD', qty='10')]
    calls = []

    def sub(req):
        calls.append((str(req.side.value), req.symbol))
        return MagicMock()
    c.trading_client.submit_order.side_effect = sub

    def fake_poll(client, coids, **k):
        return {x: {'status': 'filled', 'qty': 10.0 if x.endswith('-s') else 5.0, 'avg': 100.0} for x in coids}
    monkeypatch.setattr(runner, 'poll_fills', fake_poll)
    state = {'cash': 0.0, 'positions': {'OLD': 10.0}}
    orders = [{'symbol': 'NEW', 'side': 'buy', 'notional': 500.0, 'full_exit': False},
              {'symbol': 'OLD', 'side': 'sell', 'notional': 1000.0, 'full_exit': True}]
    orders = sorted(orders, key=lambda o: o['side'] != 'sell')
    fills = runner.execute(c, orders, {'OLD': 100.0, 'NEW': 100.0}, state, date(2026, 10, 5))
    assert [x[0] for x in calls] == ['sell', 'buy']
    # execute() no longer mutates the state: the caller resyncs it from the broker (sync_state_from_broker)
    assert state == {'cash': 0.0, 'positions': {'OLD': 10.0}}
    assert len(fills) == 2 and {f['side'] for f in fills} == {'sell', 'buy'}


def test_full_exit_sells_the_brokers_exact_quantity():
    """2026-10-02: state held 2.365105 (rounded up), the broker 2.365104893 -> the sell was rejected 40310000.
    A full exit must request exactly the broker's quantity, never the rounded state quantity."""
    c = fake_client()
    c.trading_client.get_all_positions.return_value = [MagicMock(symbol='MPC', qty='2.365104893')]
    sent = []
    c.trading_client.submit_order.side_effect = lambda req: sent.append(req) or MagicMock()
    c.trading_client.get_order_by_client_id.side_effect = Exception('404 not found')
    import scripts.momentum_sleeve as r2
    orig = r2.poll_fills
    r2.poll_fills = lambda client, coids, **k: {x: {'status': 'filled', 'qty': 2.365104893, 'avg': 420.0} for x in coids}
    try:
        r2.execute(c, [{'symbol': 'MPC', 'side': 'sell', 'notional': 936.4, 'full_exit': True}],
                   {'MPC': 420.0}, {'cash': 0.0, 'positions': {'MPC': 2.365105}}, date(2026, 10, 5))
    finally:
        r2.poll_fills = orig
    assert float(sent[0].qty) == pytest.approx(2.365104893, abs=1e-12)
    assert float(sent[0].qty) <= 2.365104893


def test_floor_qty_never_rounds_up():
    assert runner.floor_qty(2.3651048939) == 2.365104893
    assert runner.floor_qty(0.4925391999) == 0.492539199


def test_rejected_order_does_not_abort_the_run(caplog):
    """One broker rejection is logged ERROR and the remaining orders still go out."""
    import logging
    c = fake_client()
    c.trading_client.get_all_positions.return_value = [MagicMock(symbol='AAA', qty='1.0'), MagicMock(symbol='BBB', qty='1.0')]
    c.trading_client.get_order_by_client_id.side_effect = Exception('404 not found')
    sent = []

    def sub(req):
        if req.symbol == 'AAA':
            raise RuntimeError('insufficient qty available')
        sent.append(req.symbol); return MagicMock()
    c.trading_client.submit_order.side_effect = sub
    import scripts.momentum_sleeve as r2
    orig = r2.poll_fills
    r2.poll_fills = lambda client, coids, **k: {x: {'status': 'filled', 'qty': 1.0, 'avg': 10.0} for x in coids}
    try:
        with caplog.at_level(logging.ERROR):
            fills = r2.execute(c, [{'symbol': 'AAA', 'side': 'sell', 'notional': 10.0, 'full_exit': True},
                                   {'symbol': 'BBB', 'side': 'sell', 'notional': 10.0, 'full_exit': True}],
                               {'AAA': 10.0, 'BBB': 10.0}, {'cash': 0.0, 'positions': {}}, date(2026, 10, 5))
    finally:
        r2.poll_fills = orig
    assert sent == ['BBB'] and len(fills) == 1 and 'REJECTED' in caplog.text


def test_sync_state_from_broker_rebuilds_positions_and_cash():
    """Positions come from the broker (dust dropped); cash = start - filled mom- buys + filled mom- sells."""
    c = fake_client()
    c.trading_client.get_all_positions.return_value = [MagicMock(symbol='AAA', qty='3.5'),
                                                       MagicMock(symbol='DUST', qty='0.00000005')]
    def order(coid, side, qty, avg, i):
        o = MagicMock(); o.id = f'id{i}'; o.client_order_id = coid; o.filled_qty = qty; o.filled_avg_price = avg
        o.side = MagicMock(value=side); return o
    c.trading_client.get_orders.return_value = [order('mom-20261002-AAA-b', 'buy', '5', '100', 1),
                                                order('mom-20261002-AAA-s-f1', 'sell', '1.5', '110', 2),
                                                order('orb-xyz', 'buy', '9', '50', 3)]
    state = {'cash': 123.0, 'positions': {'OLD': 1.0}}
    runner.sync_state_from_broker(c, state, 20000.0)
    assert state['positions'] == {'AAA': 3.5}
    assert state['cash'] == pytest.approx(20000.0 - 500.0 + 165.0)


def test_run_dry_run_submits_nothing(monkeypatch, capsys, tmp_path):
    panel, dates = make_panel({f'S{i:02d}': (50, 0.001 * (i + 1), 1e7, 0.01) for i in range(25)})
    a = pd.DataFrame({'symbol': [f'S{i:02d}' for i in range(25)], 'name': ['Co'] * 25})
    c = fake_client()
    cal = [{'date': d.date(), 'open': None, 'close': None} for d in pd.bdate_range('2024-01-01', '2026-12-31')]
    c.get_market_calendar.side_effect = lambda s, e: [x for x in cal if s <= x['date'] <= e]
    monkeypatch.setattr(runner, 'fetch_assets', lambda client: a)
    monkeypatch.setattr(runner, 'fetch_panel', lambda client, syms, asof: (panel, []))
    monkeypatch.setattr(runner, 'write_cache_atomic', lambda p, d: 'x')
    monkeypatch.setattr(runner, 'STATE_PATH', str(tmp_path / 'state.json'))
    args = MagicMock(force=True, submit=False, asof=str(dates[-1].date()), n=20, equity_start=20000.0,
                     skip_fetch=False, gate='half')
    monkeypatch.setattr(ms, 'ADV_CUTOFF', 0.0)
    rc = runner.run(args, c, None, datetime(2026, 10, 5, 14, 0, tzinfo=ZoneInfo('UTC')))
    out = capsys.readouterr().out
    assert rc == 0 and 'TOP:' in out and 'BUY' in out
    c.trading_client.submit_order.assert_not_called()


def test_run_submit_refuses_outside_window(monkeypatch):
    c = fake_client()
    cal = [{'date': d.date(), 'open': None, 'close': None} for d in pd.bdate_range('2026-09-01', '2026-12-31')]
    c.get_market_calendar.side_effect = lambda s, e: [x for x in cal if s <= x['date'] <= e]
    args = MagicMock(force=True, submit=True, asof=None, n=20, equity_start=20000.0, skip_fetch=True)
    with pytest.raises(RuntimeError):
        runner.run(args, c, None, datetime(2026, 10, 5, 22, 0, tzinfo=ZoneInfo('UTC')))   # 18:00 ET
    c.trading_client.submit_order.assert_not_called()


def test_non_rebalance_day_is_noop(capsys):
    c = fake_client()
    cal = [{'date': d.date(), 'open': None, 'close': None} for d in pd.bdate_range('2026-09-01', '2026-12-31')]
    c.get_market_calendar.side_effect = lambda s, e: [x for x in cal if s <= x['date'] <= e]
    args = MagicMock(force=False, submit=False, asof=None, n=20, equity_start=20000.0, skip_fetch=True)
    assert runner.run(args, c, None, datetime(2026, 10, 7, 15, 0, tzinfo=ZoneInfo('UTC'))) == 0
    assert 'not a rebalance session' in capsys.readouterr().out


class TestBrokerMarks:
    """Post-trade equity is marked at the broker's current prices, not the signal-date closes."""

    def test_marks_come_from_broker_positions(self):
        import scripts.momentum_sleeve as runner
        from unittest.mock import MagicMock
        from data_sources.alpaca_client import AlpacaClient
        client = MagicMock(spec=AlpacaClient)
        client.trading_client = MagicMock()
        pos = MagicMock(); pos.symbol = 'AAA'; pos.current_price = '12.5'
        client.trading_client.get_all_positions.return_value = [pos]
        assert runner.broker_marks(client) == {'AAA': 12.5}

    def test_marks_failure_falls_back_with_warning(self, caplog):
        import logging
        import scripts.momentum_sleeve as runner
        from unittest.mock import MagicMock
        from data_sources.alpaca_client import AlpacaClient
        client = MagicMock(spec=AlpacaClient)
        client.trading_client = MagicMock()
        client.trading_client.get_all_positions.side_effect = RuntimeError('boom')
        with caplog.at_level(logging.WARNING):
            assert runner.broker_marks(client) == {}
        assert 'broker marks unavailable' in caplog.text


def test_forced_run_gets_its_own_client_order_ids():
    """A deliberate second run on the same day must not collide with the first run's order ids."""
    import scripts.momentum_sleeve as runner
    from datetime import date as _d
    plain = runner.client_order_id(_d(2026, 10, 2), 'MU', 'buy')
    forced = runner.client_order_id(_d(2026, 10, 2), 'MU', 'buy', 'f162500')
    assert plain == 'mom-20261002-MU-b' and forced == 'mom-20261002-MU-b-f162500' and plain != forced


def test_build_plan_sizes_on_current_marks_not_signal_closes():
    """A held name that moved since the signal date is trimmed/topped to 1/N at the CURRENT price."""
    panel, dates = make_panel({f'S{i:02d}': (50, 0.001 * (i + 1), 1e7, 0.01) for i in range(25)})
    a = pd.DataFrame({'symbol': [f'S{i:02d}' for i in range(25)], 'name': ['Co'] * 25})
    asof = dates[-1].date()
    sel, feat, *_ = runner.build_plan(panel, a, asof, 20, {'cash': 20000.0, 'positions': {}})
    held = sel[0]; close = float(feat.loc[held, 'close'])
    state = {'cash': 0.0, 'positions': {held: 1000.0 / close}}
    _, _, eq0, _, orders0, _ = runner.build_plan(panel, a, asof, 20, dict(state))
    _, _, eq1, _, orders1, _ = runner.build_plan(panel, a, asof, 20, dict(state), marks={held: close * 2})
    assert eq0 == pytest.approx(1000.0) and eq1 == pytest.approx(2000.0)
    o1 = [o for o in orders1 if o['symbol'] == held][0]
    assert o1['side'] == 'sell' and o1['notional'] == pytest.approx(2000.0 - 2000.0 / 20)


# ------------------------------------------------------------------ hygiene guard (spec 2026-10-02 A)

def guard_panel(closes, dates=None, symbol='XYZ'):
    """One-symbol long panel from a list of closes on consecutive business days (or the given dates)."""
    dates = pd.bdate_range('2024-01-01', periods=len(closes)) if dates is None else pd.DatetimeIndex(dates)
    return pd.DataFrame({'symbol': symbol, 'bar_date': dates, 'close': np.asarray(closes, dtype='float32'),
                         'volume': 1e6}), dates


def test_guard_constants_match_the_backtest():
    assert (ms.GUARD_MOVE_UP, ms.GUARD_MOVE_DOWN, ms.GUARD_MAX_GAP_DAYS) == (2.0, -0.75, 10)


def test_guard_up_move_over_200pct_is_out_and_just_under_is_in():
    closes = [100.0] * 300
    closes[200:] = [301.0] * 100                       # +201 %
    panel, dates = guard_panel(closes)
    assert ms.hygiene_ineligible(panel, dates[-1]) == {'XYZ': 'move'}
    closes[200:] = [299.0] * 100                       # +199 %
    panel, dates = guard_panel(closes)
    assert ms.hygiene_ineligible(panel, dates[-1]) == {}


def test_guard_down_move_75pct_is_out():
    closes = [100.0] * 300
    closes[250:] = [24.0] * 50                         # -76 %
    panel, dates = guard_panel(closes)
    assert ms.hygiene_ineligible(panel, dates[-1]) == {'XYZ': 'move'}
    closes[250:] = [26.0] * 50                         # -74 %
    panel, dates = guard_panel(closes)
    assert ms.hygiene_ineligible(panel, dates[-1]) == {}


def test_guard_gap_over_10_calendar_days_is_out_and_holiday_gap_is_in():
    d = list(pd.bdate_range('2024-01-01', periods=299))
    gapped = d[:150] + [x + pd.Timedelta(days=11) for x in d[150:]]       # 11-day hole between two bars
    panel, dates = guard_panel([100.0] * 299, gapped)
    assert ms.hygiene_ineligible(panel, dates[-1]) == {'XYZ': 'gap'}
    holiday = d[:150] + [x + pd.Timedelta(days=4) for x in d[150:]]       # 4-day long weekend
    panel, dates = guard_panel([100.0] * 299, holiday)
    assert ms.hygiene_ineligible(panel, dates[-1]) == {}


def test_guard_event_outside_the_273_bar_lookback_is_in():
    closes = [100.0] * 400
    closes[100:] = [400.0] * 300                       # the +300 % move is bar 100
    panel, dates = guard_panel(closes)
    assert ms.hygiene_ineligible(panel, dates[100 + 272]) == {'XYZ': 'move'}   # 272 bars back: inside
    assert ms.hygiene_ineligible(panel, dates[100 + 273]) == {}                # 273 bars back: out


def test_guard_empty_and_single_row_do_not_raise():
    empty = pd.DataFrame({'symbol': [], 'bar_date': pd.to_datetime([]), 'close': [], 'volume': []})
    assert ms.hygiene_ineligible(empty, pd.Timestamp('2024-06-03')) == {}
    one, dates = guard_panel([100.0])
    assert ms.hygiene_ineligible(one, dates[0]) == {}


def test_guard_ignores_bars_after_asof():
    closes = [100.0] * 300
    closes[299] = 500.0
    panel, dates = guard_panel(closes)
    assert ms.hygiene_ineligible(panel, dates[298]) == {}
    assert ms.hygiene_ineligible(panel, dates[299]) == {'XYZ': 'move'}


def test_eligible_universe_removes_guarded_names_before_ranking():
    specs = {'AAA': (50.0, 0.001, 1e7, 0.01), 'BBB': (50.0, 0.002, 1e7, 0.01)}
    panel, dates = make_panel(specs)
    panel.loc[(panel.symbol == 'BBB') & (panel.bar_date == dates[290]), 'close'] *= np.float32(10.0)
    names = assets(AAA='AAA INC', BBB='BBB INC')
    assert 'BBB' in ms.eligible_universe(panel, dates[-1], names, guard=False)
    assert ms.eligible_universe(panel, dates[-1], names) == ['AAA']
    events = ms.hygiene_events(panel, dates[-1])
    assert events['BBB'][0] == 'move' and events['BBB'][1] == dates[291].date()   # latest of the +900 % bar and the -90 % reversal bar


G_HOLD = os.path.join(ROOT, 'research', 'momentum_weekly', 'recon', 'G_holdings.csv')


@pytest.mark.skipif(not (os.path.exists(PANEL) and os.path.exists(G_HOLD) and os.path.exists(ASSETS)),
                    reason='research panel / guarded BT holdings not present')
@pytest.mark.parametrize('target', ['2021-02-08', '2025-12-29', '2026-06-29'])
def test_parity_with_guarded_backtest(target):
    import pyarrow.parquet as pq
    hold = pd.read_csv(G_HOLD)
    sub = hold[hold.rebalance_date == target]
    reb_ts = pd.Timestamp(target)
    lo = reb_ts - pd.Timedelta(days=460)
    tbl = pq.read_table(PANEL, columns=['symbol', 'bar_date', 'close', 'volume'],
                        filters=[('bar_date', '>=', lo.to_pydatetime()), ('bar_date', '<', reb_ts.to_pydatetime())])
    panel = tbl.to_pandas()
    panel['close'] = panel['close'].astype('float32')
    asof = pd.Timestamp(sub.signal_date.iloc[0])
    assets_df = pd.read_csv(ASSETS, dtype={'symbol': str, 'name': str})
    sel = ms.select_top(ms.risk_adjusted_momentum(panel, asof).loc[
        ms.eligible_universe(panel, asof, assets_df), 'signal'], 20)
    print(f'GUARD PARITY {target}: {len(set(sel) & set(sub.symbol))}/20')
    assert set(sel) == set(sub.symbol) and len(sel) == 20


# ------------------------------------------------------------------ shadow term-structure gate (spec B)

def vix_series(n=400, seed=3):
    idx = pd.bdate_range('2023-01-02', periods=n)
    rng = np.random.default_rng(seed)
    v3m = pd.Series(20 + rng.standard_normal(n), index=idx)
    vix = v3m * (0.9 + 0.1 * rng.random(n))
    return vix, v3m


def test_term_structure_gate_percentile_and_gate_on():
    vix, v3m = vix_series()
    asof = vix.index[-1]
    vix.iloc[-1] = v3m.iloc[-1] * 0.5                  # deeply in contango: lowest ratio of the window
    out = ms.term_structure_gate(vix, v3m, asof)
    assert out['gate_on'] is True and out['percentile'] == pytest.approx(0.0)
    vix.iloc[-1] = v3m.iloc[-1] * 1.5                  # backwardation: highest ratio
    out = ms.term_structure_gate(vix, v3m, asof)
    assert out['gate_on'] is False and out['percentile'] == pytest.approx(1.0)
    assert out['ratio'] == pytest.approx(1.5)


def test_term_structure_gate_matches_the_backtest_states():
    """PARITY with 1700u (w252, p30): the gate state at six signal Fridays, on a saved slice of the CBOE closes."""
    fixture = os.path.join(ROOT, 'research', 'momentum_weekly', 'recon', 'vix_fixture.csv')
    if not os.path.exists(fixture):
        pytest.skip('vix fixture absent')
    d = pd.read_csv(fixture, index_col=0, parse_dates=True)
    expected = {'2026-08-21': True, '2026-08-28': True, '2026-09-04': True,
                '2026-09-11': False, '2026-09-18': True, '2026-09-25': True}
    got = {day: ms.term_structure_gate(d['VIX'], d['VIX3M'], day)['gate_on'] for day in expected}
    assert got == expected


def test_term_structure_gate_never_uses_data_after_asof():
    vix, v3m = vix_series()
    asof = vix.index[300]
    base = ms.term_structure_gate(vix.iloc[:301], v3m.iloc[:301], asof)
    assert ms.term_structure_gate(vix, v3m, asof) == base
    vix2 = vix.copy()
    vix2.iloc[301:] = 99.0
    assert ms.term_structure_gate(vix2, v3m, asof) == base


def test_term_structure_gate_short_history_returns_none(caplog):
    vix, v3m = vix_series(n=100)
    with caplog.at_level('WARNING'):
        assert ms.term_structure_gate(vix, v3m, vix.index[-1]) is None
    assert 'history' in caplog.text


def test_gate_does_not_change_the_order_list():
    """Shadow flag: build_plan takes no gate input, so the orders are identical with the gate ON and OFF."""
    import inspect
    assert 'gate' not in ' '.join(inspect.signature(runner.build_plan).parameters)
    specs = {f'S{i}': (50.0, 0.001 * (i + 1), 1e7, 0.01) for i in range(30)}
    panel, dates = make_panel(specs)
    names = assets(**{s: f'{s} INC' for s in specs})
    state = {'cash': 20000.0, 'positions': {}, 'peak_equity': 20000.0}
    vix, v3m = vix_series()
    plans = []
    for scale in (0.5, 1.5):                           # gate ON / OFF inputs
        v = vix.copy()
        v.iloc[-1] = v3m.iloc[-1] * scale
        ms.term_structure_gate(v, v3m, v.index[-1])
        plans.append(runner.build_plan(panel, names, dates[-1], 20, dict(state, positions={})))
    assert plans[0][4] == plans[1][4] and plans[0][0] == plans[1][0]


def test_load_gate_inputs_failure_is_na_with_warning(monkeypatch, caplog):
    def boom(url, timeout=60):
        raise OSError('network down')
    monkeypatch.setattr(runner, 'http_get_text', boom)
    with caplog.at_level('WARNING'):
        info = runner.shadow_gate_info(pd.Timestamp('2026-10-02'))
    assert info is None and 'gate' in caplog.text.lower()
    assert runner.gate_label(None) == 'gate n/a'


def test_gate_label_and_csv_row(tmp_path, monkeypatch):
    info = dict(date=date(2026, 10, 2), vix=16.0, vix3m=18.0, ratio=16 / 18, percentile=0.21, gate_on=True)
    assert runner.gate_label(info) == 'gate ON (p21)'
    assert runner.gate_label(dict(info, percentile=0.55, gate_on=False)) == 'gate OFF (p55)'
    path = tmp_path / 'shadow.csv'
    monkeypatch.setattr(runner, 'SHADOW_GATE_PATH', str(path))
    runner.append_shadow_gate(date(2026, 10, 5), info, 20123.4)
    row = pd.read_csv(path).iloc[0]
    assert list(pd.read_csv(path).columns) == runner.SHADOW_GATE_FIELDS
    assert row['gate_on'] and row['equity'] == pytest.approx(20123.4)


def test_guard_log_line_lists_top40_removals_or_none():
    assert runner.guard_line({}, ['A', 'B']) == 'guard: none in the top 40'
    ev = {'ZZZ': ('move', date(2025, 5, 1)), 'QQQ': ('gap', date(2025, 6, 2))}
    line = runner.guard_line(ev, ['A', 'ZZZ', 'B'])
    assert 'ZZZ move 2025-05-01' in line and 'QQQ' not in line


# ------------------------------------------------------------------ half-size gate (spec C)

def _info(pct):
    """A gate-info dict at a given percentile (the shape term_structure_gate returns)."""
    return dict(date=date(2026, 10, 2), vix=16.0, vix3m=18.0, ratio=16 / 18, percentile=pct, gate_on=pct < 0.30)


def test_gate_constants():
    assert (ms.GATE_MODE_OFF, ms.GATE_MODE_SHADOW, ms.GATE_MODE_HALF) == ('off', 'shadow', 'half')
    assert ms.GATE_HALF_PCT == 0.20


def test_gate_scale_half_only_in_half_mode_below_20pct():
    assert ms.gate_scale(_info(0.19), 'half') == 0.5
    assert ms.gate_scale(_info(0.0), 'half') == 0.5
    assert ms.gate_scale(_info(0.20), 'half') == 1.0           # exactly 20 % is NOT below
    assert ms.gate_scale(_info(0.55), 'half') == 1.0
    assert ms.gate_scale(_info(0.05), 'shadow') == 1.0
    assert ms.gate_scale(_info(0.05), 'off') == 1.0


def test_gate_scale_missing_gate_is_full_size_with_warning(caplog):
    with caplog.at_level('WARNING'):
        assert ms.gate_scale(None, 'half') == 1.0
    assert 'full size' in caplog.text.lower()


def test_target_dollars_scale_multiplies_every_target():
    full = ms.target_dollars(['A', 'B'], 1000.0, 20)
    half = ms.target_dollars(['A', 'B'], 1000.0, 20, scale=0.5)
    assert full == {'A': 50.0, 'B': 50.0} and half == {'A': 25.0, 'B': 25.0}
    assert ms.target_dollars(['A'], 1000.0, 20) == {'A': 50.0}


def _half_world():
    specs = {f'S{i}': (50.0, 0.001 * (i + 1), 1e7, 0.01) for i in range(30)}
    panel, dates = make_panel(specs)
    return panel, assets(**{s: f'{s} INC' for s in specs}), dates[-1]


def test_mode_off_and_shadow_never_change_the_order_list():
    panel, names, asof = _half_world()
    state = {'cash': 20000.0, 'positions': {}, 'peak_equity': 20000.0}
    base = runner.build_plan(panel, names, asof, 20, dict(state, positions={}))
    for mode in ('off', 'shadow'):
        s = ms.gate_scale(_info(0.01), mode)
        plan = runner.build_plan(panel, names, asof, 20, dict(state, positions={}), scale=s)
        assert plan[4] == base[4] and plan[3] == base[3]


def test_gated_week_sells_to_half_and_ungated_week_buys_back_with_cash_unchanged():
    panel, names, asof = _half_world()
    state = {'cash': 20000.0, 'positions': {}, 'peak_equity': 20000.0}
    sel, feat, eq, tg, orders, prices = runner.build_plan(panel, names, asof, 20, dict(state, positions={}))
    held = {s: 1000.0 / prices[s] for s in sel}
    full_state = {'cash': 0.0, 'positions': held}
    _, _, eq_h, tg_h, ord_h, _ = runner.build_plan(panel, names, asof, 20, dict(full_state), scale=0.5)
    assert eq_h == pytest.approx(20000.0)
    assert all(v == pytest.approx(500.0) for v in tg_h.values())
    assert len(ord_h) == 20 and all(o['side'] == 'sell' and o['notional'] == pytest.approx(500.0) for o in ord_h)
    half_state = {'cash': 10000.0, 'positions': {s: 500.0 / prices[s] for s in sel}}
    _, _, eq_b, tg_b, ord_b, _ = runner.build_plan(panel, names, asof, 20, dict(half_state), scale=1.0)
    assert eq_b == pytest.approx(20000.0)                      # cash accounting does not depend on the mode
    assert all(o['side'] == 'buy' and o['notional'] == pytest.approx(500.0) for o in ord_b)


def test_size_label_gate_label_and_csv_scale_column(tmp_path, monkeypatch):
    assert runner.size_label(1.0) == 'size 100%' and runner.size_label(0.5) == 'size 50%'
    path = tmp_path / 'shadow.csv'
    monkeypatch.setattr(runner, 'SHADOW_GATE_PATH', str(path))
    runner.append_shadow_gate(date(2026, 10, 5), _info(0.1), 20000.0, scale=0.5)
    assert pd.read_csv(path).iloc[0]['scale'] == 0.5


def test_csv_header_migration_keeps_old_rows(tmp_path, monkeypatch):
    path = tmp_path / 'shadow.csv'
    old = [f for f in runner.SHADOW_GATE_FIELDS if f != 'scale']
    path.write_text(','.join(old) + '\n' + ','.join(['1'] * len(old)) + '\n')
    monkeypatch.setattr(runner, 'SHADOW_GATE_PATH', str(path))
    runner.append_shadow_gate(date(2026, 10, 5), _info(0.1), 20000.0, scale=0.5)
    df = pd.read_csv(path)
    assert list(df.columns) == runner.SHADOW_GATE_FIELDS and len(df) == 2
    assert pd.isna(df.iloc[0]['scale']) and df.iloc[1]['scale'] == 0.5


def test_gate_cli_default_is_half():
    assert runner.build_arg_parser().parse_args([]).gate == 'half'
    assert runner.build_arg_parser().parse_args(['--gate', 'shadow']).gate == 'shadow'


H_GATE = os.path.join(ROOT, 'research', 'momentum_weekly', 'recon', 'H_gate.csv')


@pytest.mark.skipif(not os.path.exists(H_GATE), reason='half-gate BT dump not present')
def test_parity_with_half_gate_backtest():
    """Per Monday: the BT's gate state (p20 half cell) and per-name weight vs gate_scale + target_dollars."""
    h = pd.read_csv(H_GATE)
    assert h.rebalance_date.nunique() == 3 and h.groupby('rebalance_date').gated.first().sum() == 2
    for d, sub in h.groupby('rebalance_date'):
        pct = float(sub.percentile.iloc[0])
        sc = ms.gate_scale(_info(pct), 'half')
        tgt = ms.target_dollars(list(sub.symbol), 1.0, 20, scale=sc)
        assert (sc == 0.5) == bool(sub.gated.iloc[0]), d
        assert len(tgt) == 20 and all(v == pytest.approx(w) for v, w in zip(tgt.values(), sub.weight)), d
        print(f'HALF-GATE PARITY {d}: gated={bool(sub.gated.iloc[0])} p={pct:.3f} weight={sub.weight.iloc[0]}')
