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
    assert 'OLD' not in state['positions'] and state['positions']['NEW'] == 5.0
    assert state['cash'] == pytest.approx(1000.0 - 500.0)
    assert len(fills) == 2


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
                     skip_fetch=False)
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
