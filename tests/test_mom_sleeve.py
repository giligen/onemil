"""Unit tests for the weekly momentum paper sleeve (scripts/mom_sleeve.py, trading/mom_sleeve_select.py)."""
import importlib.util
import os
import re
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from data_sources.alpaca_client import AlpacaClient
from trading import mom_sleeve_select as mss

_spec = importlib.util.spec_from_file_location(
    'mom_sleeve', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'mom_sleeve.py'))
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)


def make_client(paper=True, url='https://paper-api.alpaca.markets'):
    """AlpacaClient-spec'd mock with a plain trading_client (instance attribute, outside the class spec)."""
    c = MagicMock(spec=AlpacaClient)
    c.is_paper = paper
    c.trading_client = MagicMock()
    c.trading_client._base_url = url
    return c


def fixture_panel(n_sym=30, n_days=330, seed=7):
    """Random-walk panel, large volume so adv20 clears $200M, one stale symbol and one cheap symbol."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end='2026-09-30', periods=n_days)
    rows = []
    for i in range(n_sym):
        r = rng.normal(0.0006 * (i % 7), 0.02 + 0.002 * (i % 5), n_days)
        close = 40 * np.exp(np.cumsum(r)) if i != 3 else 5 * np.exp(np.cumsum(r) * 0.1)
        d = days[:-3] if i == 5 else days
        for dt, c in zip(d, close[:len(d)]):
            rows.append((f'S{i:02d}', dt, c, c * 1.01, c * 0.99, c, 6e6))
    return pd.DataFrame(rows, columns=mss.BAR_COLS), days[-1]


def reference_sigv2(bars, signal_date):
    """Verbatim panel math of research/momentum_weekly/1700j_frontier.py (module scope there)."""
    raw = bars.copy()
    for c in ('open', 'high', 'low', 'close'):
        raw[c] = raw[c].astype('float32')
    raw['volume'] = raw['volume'].astype('float32')
    raw = raw.drop_duplicates(subset=['symbol', 'bar_date'], keep='last')
    raw = raw[~((raw.open <= 0) | (raw.high <= 0) | (raw.low <= 0) | (raw.close <= 0))].reset_index(drop=True)
    panel = raw.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    g = panel.groupby('symbol', sort=False, observed=True)
    panel['adv20'] = (panel.close * panel.volume).groupby(panel.symbol, observed=True).rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    c21, c252, c273 = g['close'].shift(21), g['close'].shift(252), g['close'].shift(273)
    panel['sig12_1'] = c21 / c252 - 1
    panel['ret1d'] = g['close'].pct_change()
    panel['vol252'] = panel.groupby('symbol', sort=False, observed=True)['ret1d'].rolling(252, min_periods=252).std().reset_index(level=0, drop=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        panel['sigV2'] = np.where(panel.vol252 > 0, panel.sig12_1 / panel.vol252, np.nan)
    panel['ok'] = c273.notna()
    sr = panel[(panel.bar_date == signal_date) & (panel.close >= 10.0) & panel.ok & (panel.adv20 >= 200_000_000.0)
               & panel.sigV2.notna()]
    return sr.nlargest(40, 'sigV2')[['symbol', 'sigV2']].reset_index(drop=True)


def test_selection_parity_with_reference():
    """Shared module == reference panel math on the same fixture: same symbols, same order, same signal."""
    bars, sd = fixture_panel()
    ref = reference_sigv2(bars, sd)
    got = mss.rank_universe(mss.compute_signal_table(bars, sd))
    assert len(ref) > 10
    assert list(got.symbol) == list(ref.symbol)
    np.testing.assert_allclose(got.sigV2.values, ref.sigV2.values, rtol=1e-9)
    assert 'S03' not in set(got.symbol) and 'S05' not in set(got.symbol)   # cheap and stale names out


def test_name_exclusions_word_boundary():
    """Reference regex: ETF/fund/trust/warrant/unit/preferred/right words excluded; NFLX-style substrings kept."""
    assert mss.is_excluded('SPY', 'SPDR S&P 500 ETF Trust')
    assert mss.is_excluded('XYZW', 'Acme Warrants')
    assert mss.is_excluded('ZVZZT', 'NASDAQ test stock')
    assert mss.is_excluded('0029900E0', 'placeholder')
    assert not mss.is_excluded('NFLX', 'Netflix, Inc. Common Stock')
    assert not mss.is_excluded('BRKR', 'Bright Horizons Inc')


def test_pick_top_skips_ineligible_and_takes_next_rank(caplog):
    """A non-fractionable pick is skipped with a WARNING and the next rank is taken."""
    ranked = pd.DataFrame({'symbol': [f'A{i}' for i in range(25)], 'sigV2': np.arange(25, 0, -1.0),
                           'close': 50.0, 'rank': np.arange(1, 26)})
    with caplog.at_level('WARNING'):
        picks, skipped = mss.pick_top(ranked, lambda s: (s != 'A1', 'not fractionable'))
    assert [p['symbol'] for p in picks][:2] == ['A0', 'A2'] and len(picks) == 20
    assert skipped[0]['symbol'] == 'A1' and 'not fractionable' in caplog.text


def test_build_orders_equal_weight_sells_first_and_owned_only():
    """Reset to value/20: leaver sold in full, kept names trimmed/topped up, entrants bought; QQQ never touched."""
    picks = [dict(symbol=f'P{i}', rank=i + 1, signal=1.0, prior_close=10.0) for i in range(20)]
    state = {'positions': {'P0': {'qty': 200.0, 'cost': 1}, 'P1': {'qty': 10.0, 'cost': 1}, 'OLD': {'qty': 100.0, 'cost': 1}}}
    prices = {f'P{i}': 10.0 for i in range(20)} | {'OLD': 10.0, 'QQQ': 700.0, 'SPY': 600.0}
    orders, value = ms.build_orders(state, picks, prices, date(2026, 10, 5))
    assert value == pytest.approx(2000 + 100 + 1000)
    target = value / 20
    sides = [o['side'] for o in orders]
    assert sides == sorted(sides, key=lambda s: s != 'sell')            # all sells precede all buys
    by = {o['symbol']: o for o in orders}
    assert by['OLD']['side'] == 'sell' and by['OLD']['qty'] == 100.0
    assert by['P0']['side'] == 'sell' and by['P0']['qty'] == pytest.approx((2000 - target) / 10)
    assert by['P1']['side'] == 'buy' and by['P1']['notional'] == pytest.approx(target - 100)
    assert by['P2']['note'] == 'entrant' and by['P2']['notional'] == pytest.approx(target)
    assert 'QQQ' not in by and by['P2']['coid'] == 'mom-20261005-P2-buy'


def test_inception_value_is_notional():
    """No positions: the sleeve value is the inception notional, every pick gets notional/20."""
    picks = [dict(symbol=f'P{i}', rank=i + 1, signal=1.0, prior_close=10.0) for i in range(20)]
    orders, value = ms.build_orders({'positions': {}}, picks, {f'P{i}': 10.0 for i in range(20)}, date(2026, 10, 5))
    assert value == 20000.0 and len(orders) == 20 and all(o['notional'] == 1000.0 for o in orders)


def test_paper_assertion_refuses_live():
    """is_paper False or a non-paper base URL both refuse."""
    with pytest.raises(RuntimeError):
        ms.assert_paper_account(make_client(paper=False))
    with pytest.raises(RuntimeError):
        ms.assert_paper_account(make_client(url='https://api.alpaca.markets'))
    ms.assert_paper_account(make_client())


def test_holiday_monday_rule():
    """Monday holiday: Tuesday is the rebalance day, Wednesday is not; normal week: Monday."""
    c = make_client()
    c.get_market_calendar.return_value = [{'date': date(2026, 9, d)} for d in (8, 9, 10, 11)]   # Labor Day week
    assert ms.is_rebalance_day(date(2026, 9, 8), c) and not ms.is_rebalance_day(date(2026, 9, 9), c)
    c.get_market_calendar.return_value = [{'date': date(2026, 10, d)} for d in (5, 6, 7, 8, 9)]
    assert ms.is_rebalance_day(date(2026, 10, 5), c) and not ms.is_rebalance_day(date(2026, 10, 6), c)


def _broker(c, fill_px=10.0):
    """Wire a fake broker onto c.trading_client; returns the ordered submit log."""
    submitted, log = {}, []

    def order(req_coid):
        r = submitted[req_coid]
        q = float(getattr(r, 'qty', None) or (r.notional / fill_px))
        return SimpleNamespace(id=f'id-{req_coid}', client_order_id=req_coid, status='filled', filled_qty=str(q),
                               filled_avg_price=str(fill_px))

    def get_by_coid(coid):
        if coid not in submitted:
            raise Exception('404 order not found')
        return order(coid)

    def submit(req):
        submitted[req.client_order_id] = req
        log.append(req.client_order_id)
        return order(req.client_order_id)

    c.trading_client.get_order_by_client_id.side_effect = get_by_coid
    c.trading_client.submit_order.side_effect = submit
    return log


def test_execute_sells_before_buys_and_idempotent(tmp_path, monkeypatch):
    """Sells submitted before buys; state/ledger updated; a second pass submits nothing and adds no rows."""
    monkeypatch.setattr(ms, 'official_opens', lambda c, s, d: {x: 10.0 for x in s})
    c = make_client()
    log = _broker(c)
    state = {'positions': {'OLD': {'qty': 100.0, 'cost': 900.0}}}
    orders = [dict(symbol='NEW', side='buy', qty='', notional=1000.0, rank=1, signal=2.0, prior_close=10.0,
                   note='entrant', coid='mom-20261005-NEW-buy'),
              dict(symbol='OLD', side='sell', qty=100.0, notional=1000.0, rank='', signal='', prior_close='',
                   note='left_top20', coid='mom-20261005-OLD-sell')]
    sp, lp = str(tmp_path / 's.json'), str(tmp_path / 'l.csv')
    ms.execute_orders(c, state, orders, date(2026, 10, 5), lp, sp, fill_timeout=1, poll=0)
    assert log == ['mom-20261005-OLD-sell', 'mom-20261005-NEW-buy']
    assert set(state['positions']) == {'NEW'} and state['positions']['NEW']['qty'] == pytest.approx(100.0)
    rows = ms.read_ledger(lp)
    assert len(rows) == 2 and rows[0]['slippage_bps'] == '0.0'
    ms.execute_orders(c, state, orders, date(2026, 10, 5), lp, sp, fill_timeout=1, poll=0)
    assert log == ['mom-20261005-OLD-sell', 'mom-20261005-NEW-buy'] and len(ms.read_ledger(lp)) == 2


def test_never_sells_unowned_qqq(tmp_path):
    """The account holds QQQ; the sleeve plan never contains it and the broker never receives a QQQ order."""
    c = make_client()
    log = _broker(c)
    c.get_open_positions.return_value = [{'symbol': 'QQQ', 'qty': 27}]
    picks = [dict(symbol=f'P{i}', rank=i + 1, signal=1.0, prior_close=10.0) for i in range(20)]
    orders, _ = ms.build_orders({'positions': {'P0': {'qty': 100.0, 'cost': 1}}}, picks,
                                {f'P{i}': 10.0 for i in range(20)} | {'QQQ': 700.0}, date(2026, 10, 5))
    assert all(o['symbol'] != 'QQQ' for o in orders) and not any('QQQ' in x for x in log)


def test_completeness_gate_aborts_without_orders(monkeypatch, tmp_path):
    """Universe < 90 % of last week's size -> MomAbort, nothing submitted, state untouched."""
    c = make_client()
    c.get_market_calendar.return_value = [{'date': date(2026, 10, d)} for d in (1, 2)] + [{'date': date(2026, 10, 5)}]
    roster = pd.DataFrame({'symbol': ['AAA'], 'name': ['Aaa Corp'], 'exchange': ['NASDAQ'], 'tradable': [True],
                           'fractionable': [True]})
    monkeypatch.setattr(ms.msd, 'fetch_asset_roster', lambda cl: roster)
    monkeypatch.setattr(ms.msd, 'fetch_bars', lambda *a, **k: (pd.DataFrame(columns=mss.BAR_COLS).assign(symbol=['AAA']), {}))
    ranked = pd.DataFrame({'symbol': ['AAA'], 'sigV2': [1.0], 'close': [50.0], 'rank': [1]})
    ranked.attrs['universe_size'] = 500
    monkeypatch.setattr(ms.mss, 'compute_signal_table', lambda *a, **k: pd.DataFrame({'symbol': ['AAA']}))
    monkeypatch.setattr(ms.mss, 'rank_universe', lambda *a, **k: ranked)
    with pytest.raises(ms.MomAbort, match='completeness'):
        ms.select_picks(c, {'positions': {}, 'universe_size': 1000}, date(2026, 10, 5))
    c.trading_client.submit_order.assert_not_called()


def _plan_json(plan_dir, day='2026-10-05', n=20):
    """Write a plan file like --plan does."""
    import json
    picks = [dict(symbol=f'P{i}', rank=i + 1, signal=1.0, prior_close=10.0) for i in range(n)]
    doc = dict(date=day, signal_date='2026-10-02', universe_size=600, fetch_coverage=0.99, picks=picks, skipped=[])
    with open(os.path.join(plan_dir, 'mom_sleeve_plan_20261005.json'), 'w') as f:
        json.dump(doc, f)


def _week_client():
    c = make_client()
    c.get_market_calendar.return_value = [{'date': date(2026, 10, d)} for d in (5, 6, 7, 8, 9)]
    return c


def test_execute_noop_outside_window_and_non_rebalance_day(tmp_path):
    """Execute outside 09:40-09:55 ET, or on a non-first session, is a no-op that touches nothing."""
    c = _week_client()
    kw = dict(state_path=str(tmp_path / 's.json'), ledger_path=str(tmp_path / 'l.csv'), plan_dir=str(tmp_path))
    assert 'outside' in ms.run_execute(c, None, datetime(2026, 10, 5, 13, 35, tzinfo=timezone.utc), **kw)  # 09:35 ET
    assert 'not the first' in ms.run_execute(c, None, datetime(2026, 10, 6, 13, 45, tzinfo=timezone.utc), **kw)
    c.trading_client.submit_order.assert_not_called()
    assert ms.in_window(datetime(2026, 10, 5, 9, 40, tzinfo=ms.ET)) and not ms.in_window(datetime(2026, 10, 5, 9, 36, tzinfo=ms.ET))


def test_execute_requires_todays_plan(tmp_path):
    """Missing plan -> MomAbort, no orders; a plan dated another day -> MomAbort, no orders."""
    c = _week_client()
    kw = dict(state_path=str(tmp_path / 's.json'), ledger_path=str(tmp_path / 'l.csv'), plan_dir=str(tmp_path))
    now = datetime(2026, 10, 5, 13, 45, tzinfo=timezone.utc)
    with pytest.raises(ms.MomAbort, match='no plan file'):
        ms.run_execute(c, None, now, **kw)
    _plan_json(str(tmp_path), day='2026-09-28')
    with pytest.raises(ms.MomAbort, match='dated'):
        ms.run_execute(c, None, now, **kw)
    c.trading_client.submit_order.assert_not_called()


def test_execute_performs_no_bar_fetch_and_trades(tmp_path, monkeypatch):
    """With a plan present, execute submits 20 buys and never touches the roster/bar fetch."""
    c = _week_client()
    log = _broker(c)
    c.get_latest_trades.return_value = {f'P{i}': {'price': 10.0} for i in range(20)} | {'SPY': {'price': 600.0}}
    monkeypatch.setattr(ms, 'official_opens', lambda cl, s, d: {})
    for name in ('fetch_asset_roster', 'fetch_bars'):
        monkeypatch.setattr(ms.msd, name, MagicMock(side_effect=AssertionError('data fetch in --execute')))
    _plan_json(str(tmp_path))
    kw = dict(state_path=str(tmp_path / 's.json'), ledger_path=str(tmp_path / 'l.csv'), plan_dir=str(tmp_path))
    ms.run_execute(c, None, datetime(2026, 10, 5, 13, 45, tzinfo=timezone.utc), fill_timeout=1, **kw)
    assert len(log) == 20 and all(x.endswith('-buy') for x in log)
    c.get_daily_bars.assert_not_called()
    dry = ms.run_execute(c, None, datetime(2026, 10, 5, 13, 45, tzinfo=timezone.utc), dry_run=True, **kw)
    assert 'Orders' in dry and len(log) == 20


def test_plan_refused_after_cutoff_and_writes_json(tmp_path, monkeypatch):
    """--plan after 09:25 ET refuses; before it, writes the plan JSON and no order is sent."""
    c = _week_client()
    picks = [dict(symbol=f'P{i}', rank=i + 1, signal=1.0, prior_close=10.0) for i in range(20)]
    monkeypatch.setattr(ms, 'select_picks', lambda cl, st, td: (picks, [], 600, date(2026, 10, 2), 0.99))
    sp = str(tmp_path / 's.json')
    assert 'refused' in ms.run_plan(c, datetime(2026, 10, 5, 13, 30, tzinfo=timezone.utc), state_path=sp, plan_dir=str(tmp_path))
    assert not os.path.exists(ms.plan_path(str(tmp_path), date(2026, 10, 5)))
    ms.run_plan(c, datetime(2026, 10, 5, 12, 5, tzinfo=timezone.utc), state_path=sp, plan_dir=str(tmp_path))
    assert ms.load_plan(str(tmp_path), date(2026, 10, 5))['universe_size'] == 600
    c.trading_client.submit_order.assert_not_called()


def test_liquidate_sells_only_state_positions(tmp_path):
    """--liquidate sells exactly the state-file symbols (never QQQ); --dry-run submits nothing."""
    c = make_client()
    log = _broker(c)
    sp = str(tmp_path / 's.json')
    ms.save_state({'positions': {'AAA': {'qty': 3.5, 'cost': 30.0}, 'BBB': {'qty': 10.0, 'cost': 90.0}}}, sp)
    now = datetime(2026, 10, 6, 15, 0, tzinfo=timezone.utc)
    assert 'AAA' in ms.run_liquidate(c, None, now, dry_run=True, state_path=sp, ledger_path=str(tmp_path / 'l.csv'))
    assert log == []
    ms.run_liquidate(c, None, now, state_path=sp, ledger_path=str(tmp_path / 'l.csv'), fill_timeout=1)
    assert log == ['momliq-20261006-AAA-sell', 'momliq-20261006-BBB-sell']
    assert ms.load_state(sp)['positions'] == {}


def test_apply_fill_partial_sell_keeps_cost_proportional():
    """A trim reduces qty and cost proportionally; a full sell deletes the position."""
    st = {'positions': {'A': {'qty': 10.0, 'cost': 100.0}}}
    ms.apply_fill(st, {'symbol': 'A', 'side': 'sell'}, 4.0, 12.0)
    assert st['positions']['A'] == {'qty': 6.0, 'cost': pytest.approx(60.0)}
    ms.apply_fill(st, {'symbol': 'A', 'side': 'sell'}, 6.0, 12.0)
    assert 'A' not in st['positions']
