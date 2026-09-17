"""D3 FIX 2 — the sliced, re-priced exit ladder.

`research/fuckup_audit/D3_exec/REPORT.md` §M2/§M3. `compute_limit_price`
returned `bid - max($0.01, 0.30 x spread)` for the WHOLE quantity in ONE
order. In 9 of the 11 audited events that order was larger than the
displayed bid:

    EEIQ 46.9x   RBNE 14.2x   IRE 7.1x   EHGO 4.2x   SMCX 3.5x
    TJGC 2.8x    FJET 2.4x    HCAI 2.1x  RCAT 1.7x

A one-cent concession buys the top of book and nothing else. When the
order stalled, `_escalate_to_market_close` called `close_position` — an
unpriced market sell — into the hole the first slice had just made. On
RBNE 2026-07-16 that was 2,841 shares against a 200-share bid and 46.8%
of the minute's volume: -$304.91 vs the bid, **-1.30R on a trade whose
entire planned risk was $234**.

The ladder slices to the displayed book, prices each rung at
`bid - max(tick, cross_factor x spread)`, RE-PRICES via
`replace_order_limit_price` as the bid moves, and escalates to a market
close only after `max_rounds` or `hard_deadline_s`.

**DEFAULT OFF.** `enabled: false` must be the pre-2026-09-17 path byte
for byte — that is the rollback contract and it has its own test below.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaClient
from trading.exit_reasons import ExitBranch
from trading.stop_monitor import StopMonitor


LADDER_ON = {
    'enabled': True, 'slice_to_bid_size': True, 'min_slice': 100,
    'cross_factor': 0.25, 'reprice_after_s': 0.15, 'max_rounds': 3,
    'hard_deadline_s': 2.0,
}


@pytest.fixture
def client():
    c = MagicMock(spec=AlpacaClient)
    c.cancel_order.return_value = True
    c.close_position.return_value = {'id': 'mkt-1', 'status': 'accepted'}
    c.get_open_positions.return_value = []
    c.trading_client = MagicMock()
    c.trading_client.get_orders.return_value = []
    c.get_latest_quote.return_value = {
        'bid_price': 4.68, 'ask_price': 4.74,
        'bid_size': 200, 'ask_size': 200}
    return c


def _monitor(client, ladder=None) -> StopMonitor:
    m = StopMonitor(api_key='k', api_secret='s', alpaca_client=client,
                    exit_ladder=ladder)
    m._STOP_EXIT_FILL_TIMEOUT_S = 0.2
    m._MARKET_CLOSE_FILL_TIMEOUT_S = 0.2
    m._STOP_EXIT_POLL_INTERVAL_S = 0.05
    return m


def _rbne_watch(monitor, shares=2841, bid=4.68, ask=4.74, bid_size=200):
    """RBNE 2026-07-16: 2,841 sh, bid 4.68 x 200, 128 bps spread."""
    monitor.add_watch('RBNE', 4.685, shares, 'tp-1', '')
    with monitor._watch_lock:
        w = monitor._watches['RBNE']
        w.latest_bid, w.latest_ask = bid, ask
        w.latest_bid_size, w.latest_ask_size = bid_size, bid_size
        w.latest_quote_ts = time.time()
    return w


class _Book:
    """A scripted order book: each submitted order fills `fill_qty` shares
    at its own limit, and the bid can be walked between rungs."""

    def __init__(self, client, fill_qty=200):
        self.client = client
        self.fill_qty = fill_qty
        self.orders = {}
        self.n = 0
        client.submit_limit_sell_order.side_effect = self._submit
        client.replace_order_limit_price.side_effect = self._replace
        client.get_order.side_effect = self._get

    def _submit(self, symbol, qty, limit_price, **kw):
        self.n += 1
        oid = f'lad-{self.n}'
        fq = min(int(qty), self.fill_qty)
        self.orders[oid] = {
            'id': oid, 'status': 'filled' if fq >= qty else 'partially_filled',
            'filled_qty': fq, 'filled_avg_price': limit_price, 'qty': qty}
        return {'id': oid, 'status': 'accepted'}

    def _replace(self, order_id, new_limit_price):
        self.n += 1
        oid = f'lad-{self.n}'
        old = self.orders[order_id]
        self.orders[oid] = dict(old, id=oid, filled_avg_price=new_limit_price)
        return {'id': oid, 'status': 'accepted'}

    def _get(self, order_id):
        return self.orders.get(
            order_id, {'id': order_id, 'status': 'filled',
                       'filled_qty': 0, 'filled_avg_price': None})


# ---------------------------------------------------------------------------
# the pure helpers
# ---------------------------------------------------------------------------

class TestLadderSlicing:

    def test_slices_to_displayed_bid_size(self):
        """RBNE: qty 2,841 against a 200-share bid -> the first order is
        200 shares, not 2,841."""
        assert StopMonitor.ladder_slice_qty(2841, 200, True, 100) == 200

    def test_min_slice_floors_a_one_share_bid(self):
        """Otherwise 2,841 shares becomes 2,841 orders."""
        assert StopMonitor.ladder_slice_qty(2841, 1, True, 100) == 100

    def test_never_exceeds_the_remainder(self):
        assert StopMonitor.ladder_slice_qty(150, 5000, True, 100) == 150

    def test_disabled_slicing_sends_the_whole_order(self):
        assert StopMonitor.ladder_slice_qty(2841, 200, False, 100) == 2841

    def test_zero_remaining_is_zero(self):
        assert StopMonitor.ladder_slice_qty(0, 200, True, 100) == 0

    def test_missing_bid_size_falls_back_to_min_slice(self):
        assert StopMonitor.ladder_slice_qty(2841, 0, True, 100) == 100


class TestLadderPricing:

    def test_wide_spread_concedes_a_quarter_of_it(self):
        """RBNE's 6c spread -> 1.5c under the bid, not 1c."""
        assert StopMonitor.ladder_limit_price(4.68, 4.74, 0.25) == pytest.approx(
            4.66, abs=0.005)

    def test_penny_spread_gives_up_one_tick(self):
        assert StopMonitor.ladder_limit_price(15.29, 15.30, 0.25) == 15.28

    def test_never_prices_above_the_bid(self):
        """The EEIQ midpoint defect (§M5): bid 7.67 / ask 7.72 must not
        produce 7.70."""
        px = StopMonitor.ladder_limit_price(7.67, 7.72, 0.25)
        assert px <= 7.67 and px != 7.70

    def test_missing_or_inverted_ask_degrades_to_one_tick(self):
        assert StopMonitor.ladder_limit_price(4.68, 0.0, 0.25) == 4.67
        assert StopMonitor.ladder_limit_price(4.68, 4.60, 0.25) == 4.67

    def test_floored_at_a_cent(self):
        assert StopMonitor.ladder_limit_price(0.01, 0.02, 0.25) == 0.01


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------

class TestLadderExecution:

    @pytest.mark.asyncio
    async def test_first_order_is_the_bid_size_not_the_position(self, client):
        m = _monitor(client, LADDER_ON)
        book = _Book(client, fill_qty=200)
        w = _rbne_watch(m)
        await m._execute_stop_exit('RBNE', 4.68, w, exit_reason='stop_loss')
        first = client.submit_limit_sell_order.call_args_list[0].kwargs
        assert first['qty'] == 200, "2,841 sh must not be shown at once"
        assert first['limit_price'] == pytest.approx(4.66, abs=0.005)

    @pytest.mark.asyncio
    async def test_reprices_after_reprice_after_s(self, client):
        """The bid falls 4.68 -> 4.60 while a rung is resting: the order is
        REPLACED at the new bid minus the same offset — never cancelled and
        re-queued, which would give up its place in line."""
        m = _monitor(client, LADDER_ON)
        w = _rbne_watch(m)
        n = {'i': 0, 'polls': 0}

        client.submit_limit_sell_order.side_effect = (
            lambda symbol, qty, limit_price, **kw: {
                'id': f'lad-{n["i"]}', 'status': 'accepted'})

        def _get(order_id):
            # The tape moves under the resting rung, deterministically on
            # the first poll rather than on a wall-clock race.
            n['polls'] += 1
            if n['polls'] == 1:
                with m._watch_lock:
                    m._watches['RBNE'].latest_bid = 4.60
                    m._watches['RBNE'].latest_ask = 4.66
            return {'id': order_id, 'status': 'new', 'filled_qty': 0,
                    'filled_avg_price': None}
        client.get_order.side_effect = _get
        client.replace_order_limit_price.side_effect = (
            lambda order_id, new_limit_price: {'id': 'lad-rep',
                                               'status': 'accepted'})

        await m._execute_stop_exit('RBNE', 4.68, w, exit_reason='stop_loss')

        assert client.replace_order_limit_price.called, (
            "a resting rung must be re-priced to the new bid")
        new_px = client.replace_order_limit_price.call_args.args[1]
        assert new_px == pytest.approx(4.58, abs=0.005)   # 4.60 - 25% of 6c

    @pytest.mark.asyncio
    async def test_market_escalation_only_after_hard_deadline(self, client):
        """`close_position` must not fire on the first unfilled poll — the
        whole point is that the market order is the LAST resort."""
        m = _monitor(client, dict(LADDER_ON, max_rounds=3,
                                  hard_deadline_s=2.0))
        w = _rbne_watch(m)
        fired_at = {}
        t0 = time.time()

        client.submit_limit_sell_order.side_effect = (
            lambda symbol, qty, limit_price, **kw: {'id': 'lad-x',
                                                    'status': 'accepted'})
        client.get_order.side_effect = lambda oid: (
            {'id': oid, 'status': 'filled', 'filled_qty': 2841,
             'filled_avg_price': 4.50} if oid == 'mkt-1' else
            {'id': oid, 'status': 'new', 'filled_qty': 0,
             'filled_avg_price': None})

        def _close(symbol):
            fired_at['t'] = time.time() - t0
            return {'id': 'mkt-1', 'status': 'accepted'}
        client.close_position.side_effect = _close

        await m._execute_stop_exit('RBNE', 4.68, w, exit_reason='stop_loss')
        assert 't' in fired_at, "the remainder must still get out"
        assert fired_at['t'] >= 3 * LADDER_ON['reprice_after_s'], (
            f"escalated after only {fired_at['t']:.2f}s — the ladder must "
            f"work its rounds first")

    @pytest.mark.asyncio
    async def test_completed_ladder_is_branch_limit_with_blended_price(
            self, client):
        """Three 947-share rungs at falling prices -> the qty-weighted
        blend, branch `limit`, and no market order at all."""
        m = _monitor(client, dict(LADDER_ON, min_slice=947, max_rounds=3))
        w = _rbne_watch(m, shares=2841, bid_size=947)
        prices = [4.66, 4.64, 4.62]
        n = {'i': 0}

        def _submit(symbol, qty, limit_price, **kw):
            oid = f'lad-{n["i"]}'
            m._pending = (oid, qty, prices[n['i']])
            n['i'] += 1
            return {'id': oid, 'status': 'accepted'}

        def _get(order_id):
            oid, qty, px = m._pending
            return {'id': order_id, 'status': 'filled', 'filled_qty': qty,
                    'filled_avg_price': px}
        client.submit_limit_sell_order.side_effect = _submit
        client.get_order.side_effect = _get

        await m._execute_stop_exit('RBNE', 4.68, w, exit_reason='trail_stop')
        ev = m.drain_exit_events()[0]
        assert client.close_position.call_count == 0
        assert ev.exit_branch == ExitBranch.LIMIT.value
        assert ev.exit_reason == 'trail_stop'
        assert ev.shares == 2841
        assert ev.exit_price == pytest.approx(sum(prices) / 3, abs=0.001)

    @pytest.mark.asyncio
    async def test_blend_includes_the_escalated_remainder(self, client):
        """One rung fills at 4.66, the rest is market-closed at 4.50 — the
        booked price is the blend, not 4.50 on every share (the §M2
        accounting defect)."""
        m = _monitor(client, dict(LADDER_ON, max_rounds=1))
        w = _rbne_watch(m)
        client.submit_limit_sell_order.side_effect = (
            lambda symbol, qty, limit_price, **kw: {'id': 'lad-1',
                                                    'status': 'accepted'})
        client.get_order.side_effect = lambda oid: (
            {'id': oid, 'status': 'filled', 'filled_qty': 2641,
             'filled_avg_price': 4.50} if oid == 'mkt-1' else
            {'id': oid, 'status': 'filled', 'filled_qty': 200,
             'filled_avg_price': 4.66})

        await m._execute_stop_exit('RBNE', 4.68, w, exit_reason='stop_loss')
        ev = m.drain_exit_events()[0]
        expected = (200 * 4.66 + 2641 * 4.50) / 2841
        assert ev.exit_price == pytest.approx(expected, abs=0.001)
        assert ev.exit_price > 4.50
        assert ev.exit_branch == ExitBranch.MARKET_FALLBACK.value
        assert ev.exit_reason == 'stop_loss'

    @pytest.mark.asyncio
    async def test_no_bid_bails_out_to_escalation(self, client):
        """A missing quote must not produce an order at a made-up price."""
        m = _monitor(client, LADDER_ON)
        m.add_watch('RBNE', 4.685, 2841, 'tp-1', '')
        with m._watch_lock:
            w = m._watches['RBNE']
            w.latest_bid = w.latest_ask = 0.0
        client.get_order.side_effect = lambda oid: {
            'id': oid, 'status': 'filled', 'filled_qty': 2841,
            'filled_avg_price': 4.50}
        # The REST quote path still prices the legacy limit; the ladder
        # itself has no cached bid and must not submit.
        with MagicMock():
            pass
        client.get_latest_quote.side_effect = Exception('no quote')
        await m._execute_stop_exit('RBNE', 4.68, w, exit_reason='stop_loss')
        assert client.submit_limit_sell_order.call_count == 0
        assert client.close_position.call_count == 1


# ---------------------------------------------------------------------------
# THE ROLLBACK CONTRACT
# ---------------------------------------------------------------------------

class TestLadderDisabledIsByteIdentical:
    """`enabled: false` must reproduce the pre-2026-09-17 call sequence
    exactly. This is what makes the flag a rollback rather than a hope."""

    def _sequence(self, ladder):
        c = MagicMock(spec=AlpacaClient)
        c.cancel_order.return_value = True
        c.close_position.return_value = {'id': 'mkt-1', 'status': 'accepted'}
        c.get_open_positions.return_value = []
        c.trading_client = MagicMock()
        c.trading_client.get_orders.return_value = []
        c.submit_limit_sell_order.return_value = {'id': 'lmt-1',
                                                  'status': 'accepted'}
        c.get_order.return_value = {'id': 'lmt-1', 'status': 'filled',
                                    'filled_qty': 2841,
                                    'filled_avg_price': 4.66}
        m = _monitor(c, ladder)
        w = _rbne_watch(m)
        asyncio.run(m._execute_stop_exit('RBNE', 4.68, w,
                                         exit_reason='stop_loss'))
        return c, m.drain_exit_events()

    def test_ladder_disabled_is_byte_identical_to_today(self):
        c, events = self._sequence(None)          # no config at all
        c2, events2 = self._sequence({'enabled': False})

        for client in (c, c2):
            # ONE order, the FULL quantity, priced by compute_limit_price
            # (30% of spread), and no replace anywhere.
            assert client.submit_limit_sell_order.call_count == 1
            kw = client.submit_limit_sell_order.call_args.kwargs
            assert kw['qty'] == 2841
            assert kw['limit_price'] == pytest.approx(4.66, abs=0.005)
            assert client.replace_order_limit_price.call_count == 0
            assert client.close_position.call_count == 0

        assert len(events) == len(events2) == 1
        a, b = events[0], events2[0]
        for field in ('exit_price', 'shares', 'exit_reason', 'exit_branch',
                      'order_id', 'exit_limit_price', 'confirmed'):
            assert getattr(a, field) == getattr(b, field), field
        assert a.exit_branch == ExitBranch.LIMIT.value

    def test_default_construction_has_the_ladder_off(self):
        m = StopMonitor(api_key='k', api_secret='s',
                        alpaca_client=MagicMock(spec=AlpacaClient))
        assert m._exit_ladder['enabled'] is False

    def test_config_default_is_off(self):
        from config import Config
        assert Config().exit_ladder_cfg['enabled'] is False

    def test_config_defaults_match_the_report(self):
        from config import Config
        cfg = Config().exit_ladder_cfg
        assert cfg['cross_factor'] == 0.25
        assert cfg['reprice_after_s'] == 2.0
        assert cfg['max_rounds'] == 3
        assert cfg['hard_deadline_s'] == 10.0

    @pytest.mark.asyncio
    async def test_disabled_ladder_returns_none_immediately(self, client):
        m = _monitor(client, {'enabled': False})
        w = _rbne_watch(m)
        assert await m._execute_exit_ladder(
            asyncio.get_event_loop(), client, 'RBNE', w, 2841, 4.68) is None
        assert client.submit_limit_sell_order.call_count == 0
