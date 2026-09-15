"""FakeAlpacaBroker — a stateful stand-in for `data_sources.alpaca_client.AlpacaClient` with Alpaca ORDER semantics,
for lifecycle tests that must run on every commit without a broker.

Modeled (each one bit the HOD-break engine or its ancestors at least once):
- a bracket = parent limit BUY + two `held` sell legs (take-profit limit, stop-loss stop); the legs go live on the
  parent's fill; canceling the parent kills the held legs;
- a marketable limit fills IMMEDIATELY (buy at the ask, sell at the bid); a stop sell elects when the bid <= stop;
- OCO: a leg that fills completely cancels its sibling; a PARTIAL leg fill reduces the sibling's qty;
- `replace_order_limit_price` creates a NEW order id — the old order becomes `replaced` with `replaced_by`;
- `cancel_order` returns False when the order is already filled / unknown (the client swallows 404/422);
  `defer_cancel(order_id)` makes the next cancel answer `pending_cancel` (the order can still fill);
- a sell larger than the shares available (position minus shares reserved by live sell orders) is REJECTED while a
  long exists; a sell with NO position opens a SHORT (negative qty) — the double-sell failure mode;
- `partial_fill(order_id, qty, price=None)` fills part of a working order on demand; `tick(bid, ask)` moves the
  market and settles every working order; `auto_fill=False` lets orders rest (close orders that do not fill).
- dict shapes mirror the real client exactly (get_order has NO top-level limit_price; the raw SDK-like order from
  `trading_client.get_order_by_id` does), so a test written on the fake runs unchanged on paper.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, time as dtime, timezone
from types import SimpleNamespace as NS
from typing import Dict, List, Optional

from data_sources.alpaca_client import AlpacaAPIError

WORKING = ('new', 'accepted', 'partially_filled', 'pending_cancel')   # can fill
OPEN = WORKING + ('held',)                                             # listed by get_open_orders


class FakeAlpacaBroker:
    """See the module docstring. `calls` records every client-surface call as (method, kwargs)."""
    is_paper = True

    def __init__(self, bid: float = 15.00, ask: float = 15.01, positions: Optional[Dict[str, int]] = None):
        self.bid = float(bid); self.ask = float(ask)
        self.orders: Dict[str, dict] = {}
        self.positions: Dict[str, int] = dict(positions or {})
        self.auto_fill = True
        self._deferred_cancel: set = set()
        self.calls: List[tuple] = []
        self.trading_client = NS(_base_url='https://paper-api.alpaca.markets', get_order_by_id=self._raw_order, submit_order=self._submit_sdk,
                                 get_account=lambda: NS(status=NS(value='ACTIVE'), equity='100000', account_number='FAKE'),
                                 get_clock=lambda: NS(is_open=True))

    # ------------------------------------------------------------------ test controls
    def tick(self, bid: Optional[float] = None, ask: Optional[float] = None) -> None:
        """Move the market and settle every working order against it."""
        if bid is not None: self.bid = float(bid)
        if ask is not None: self.ask = float(ask)
        self._settle(force=True)

    def partial_fill(self, order_id: str, qty: int, price: Optional[float] = None) -> None:
        """Fill `qty` shares of a working order at `price` (default: its limit/stop, else the touch)."""
        o = self.orders[order_id]
        assert o['status'] in WORKING, f'{order_id} is {o["status"]}'
        px = price if price is not None else (o['limit_price'] or o['stop_price'] or (self.bid if o['side'] == 'sell' else self.ask))
        self._fill(o, qty, float(px))

    def defer_cancel(self, order_id: str) -> None:
        """The next cancel of this order answers `pending_cancel` instead of canceling (it may still fill)."""
        self._deferred_cancel.add(order_id)

    def sells_for(self, symbol: str) -> List[dict]:
        return [o for o in self.orders.values() if o['symbol'] == symbol and o['side'] == 'sell' and o['order_class'] == 'simple']

    # ------------------------------------------------------------------ internals
    def _new(self, **kw) -> dict:
        o = dict(id=str(uuid.uuid4()), client_order_id='', symbol='', side='buy', type='limit', order_class='simple', qty=0, limit_price=None, stop_price=None,
                 status='new', filled_qty=0, filled_notional=0.0, legs=[], parent=None, sibling=None, replaced_by='', replaces='',
                 submitted_at=datetime.now(timezone.utc), filled_at=None)
        o.update(kw); self.orders[o['id']] = o; return o

    @staticmethod
    def _avg(o: dict) -> Optional[float]:
        return round(o['filled_notional'] / o['filled_qty'], 4) if o['filled_qty'] else None

    def _fill(self, o: dict, qty: int, px: float) -> None:
        qty = min(int(qty), o['qty'] - o['filled_qty'])
        if qty <= 0: return
        o['filled_qty'] += qty; o['filled_notional'] += qty * px
        self.positions[o['symbol']] = self.positions.get(o['symbol'], 0) + (qty if o['side'] == 'buy' else -qty)
        sib = self.orders.get(o['sibling']) if o['sibling'] else None
        if o['filled_qty'] >= o['qty']:
            o['status'] = 'filled'; o['filled_at'] = datetime.now(timezone.utc)
            for lid in o['legs']:
                if self.orders[lid]['status'] == 'held': self.orders[lid]['status'] = 'new'
            if sib and sib['status'] in OPEN: sib['status'] = 'canceled'          # OCO
        else:
            o['status'] = 'partially_filled'
            if sib and sib['status'] in OPEN: sib['qty'] = max(0, sib['qty'] - qty)   # Alpaca trims the OCO sibling on a partial

    def _settle(self, force: bool = False) -> None:
        if not (self.auto_fill or force): return
        for o in list(self.orders.values()):
            if o['status'] not in WORKING: continue
            rem = o['qty'] - o['filled_qty']
            if o['type'] == 'market': self._fill(o, rem, self.ask if o['side'] == 'buy' else self.bid)
            elif o['side'] == 'buy' and o['limit_price'] is not None and o['limit_price'] >= self.ask: self._fill(o, rem, self.ask)
            elif o['side'] == 'sell' and o['type'] == 'limit' and o['limit_price'] <= self.bid: self._fill(o, rem, self.bid)
            elif o['side'] == 'sell' and o['type'] == 'stop' and o['stop_price'] >= self.bid: self._fill(o, rem, self.bid)

    def _available(self, symbol: str) -> int:
        reserved = sum(o['qty'] - o['filled_qty'] for o in self.orders.values() if o['symbol'] == symbol and o['side'] == 'sell' and o['status'] in WORKING)
        return self.positions.get(symbol, 0) - reserved

    def _reject_if_oversold(self, symbol: str, qty: int) -> None:
        if self.positions.get(symbol, 0) > 0 and self._available(symbol) < qty:
            raise AlpacaAPIError(f'insufficient qty available for order (requested: {qty}, available: {self._available(symbol)})')

    def _leg_dict(self, lid: str) -> dict:
        l = self.orders[lid]
        return {'id': lid, 'side': l['side'], 'type': l['type'], 'stop_price': l['stop_price'], 'limit_price': l['limit_price'],
                'filled_avg_price': self._avg(l), 'status': l['status']}

    def _order_dict(self, o: dict) -> dict:
        """The real `AlpacaClient.get_order` shape (no top-level limit/stop price)."""
        return {'id': o['id'], 'status': o['status'], 'symbol': o['symbol'], 'qty': o['qty'], 'filled_qty': o['filled_qty'], 'filled_avg_price': self._avg(o),
                'replaced_by': o['replaced_by'], 'client_order_id': o['client_order_id'], 'side': o['side'], 'type': o['type'],
                'legs': [self._leg_dict(l) for l in o['legs']]}

    def _raw_order(self, order_id: str, filter=None):
        self._settle(); o = self.orders[order_id]
        return NS(id=o['id'], client_order_id=o['client_order_id'], symbol=o['symbol'], status=NS(value=o['status']), side=NS(value=o['side']), type=NS(value=o['type']),
                  qty=o['qty'], filled_qty=o['filled_qty'], filled_avg_price=self._avg(o), limit_price=o['limit_price'], stop_price=o['stop_price'],
                  replaced_by=o['replaced_by'] or None, replaces=o['replaces'] or None, submitted_at=o['submitted_at'], filled_at=o['filled_at'],
                  legs=[NS(**self._leg_dict(l)) for l in o['legs']])

    def _submit_sdk(self, req):
        """`trading_client.submit_order(MarketOrderRequest)` — the tests' own flattening path."""
        qty = int(req.qty); side = req.side.value
        if side == 'sell': self._reject_if_oversold(req.symbol, qty)
        o = self._new(symbol=req.symbol, qty=qty, side=side, type='market', client_order_id=getattr(req, 'client_order_id', '') or '')
        self._settle(); return NS(id=o['id'], status=NS(value=o['status']))

    # ------------------------------------------------------------------ AlpacaClient surface
    def get_latest_quote(self, symbol: str, feed=None) -> dict:
        self.calls.append(('get_latest_quote', {'symbol': symbol}))
        return {'bid_price': self.bid, 'ask_price': self.ask, 'bid_size': 100, 'ask_size': 100, 'timestamp': datetime.now(timezone.utc).isoformat()}

    def submit_bracket_order(self, symbol: str, qty: int, side: str, limit_price: float, tp_price: float, sl_price: float, client_order_id: Optional[str] = None) -> dict:
        self.calls.append(('submit_bracket_order', dict(symbol=symbol, qty=qty, side=side, limit_price=limit_price, tp_price=tp_price, sl_price=sl_price, client_order_id=client_order_id)))
        p = self._new(symbol=symbol, qty=int(qty), side=side, limit_price=round(limit_price, 2), order_class='bracket', client_order_id=client_order_id or '')
        tp = self._new(symbol=symbol, qty=int(qty), side='sell', limit_price=round(tp_price, 2), status='held', parent=p['id'], order_class='bracket')
        sl = self._new(symbol=symbol, qty=int(qty), side='sell', type='stop', stop_price=round(sl_price, 2), status='held', parent=p['id'], order_class='bracket')
        tp['sibling'] = sl['id']; sl['sibling'] = tp['id']; p['legs'] = [tp['id'], sl['id']]
        self._settle()
        d = self._order_dict(p); d.update({'qty': qty, 'side': side, 'limit_price': limit_price}); return d

    def submit_limit_sell_order(self, symbol: str, qty: int, limit_price: float, client_order_id: Optional[str] = None) -> dict:
        self.calls.append(('submit_limit_sell_order', dict(symbol=symbol, qty=qty, limit_price=limit_price, client_order_id=client_order_id)))
        self._reject_if_oversold(symbol, int(qty))
        o = self._new(symbol=symbol, qty=int(qty), side='sell', limit_price=round(limit_price, 2), client_order_id=client_order_id or '')
        self._settle()
        return {'id': o['id'], 'status': o['status'], 'symbol': symbol, 'qty': qty, 'limit_price': limit_price}

    def get_order(self, order_id: str) -> dict:
        self._settle()
        if order_id not in self.orders: raise AlpacaAPIError(f'order {order_id} not found')
        return self._order_dict(self.orders[order_id])

    def cancel_order(self, order_id: str) -> bool:
        self.calls.append(('cancel_order', {'order_id': order_id}))
        o = self.orders.get(order_id)
        if o is None or o['status'] not in OPEN: return False                      # 404 / 422: the client returns False
        if order_id in self._deferred_cancel:
            self._deferred_cancel.discard(order_id); o['status'] = 'pending_cancel'; return True
        o['status'] = 'canceled'
        for lid in o['legs']:                                                       # a canceled parent takes its held legs …
            leg = self.orders[lid]
            if leg['status'] != 'held': continue
            if o['filled_qty'] > 0:                                                 # … unless part of it filled: the legs go live for the filled qty
                leg['status'] = 'new'; leg['qty'] = o['filled_qty']
            else:
                leg['status'] = 'canceled'
        sib = self.orders.get(o['sibling']) if o['sibling'] else None
        if sib and sib['status'] in OPEN: sib['status'] = 'canceled'                # OCO
        return True

    def replace_order_limit_price(self, order_id: str, new_limit_price: float) -> dict:
        self.calls.append(('replace_order_limit_price', {'order_id': order_id, 'new_limit_price': new_limit_price}))
        o = self.orders.get(order_id)
        if o is None or o['status'] not in OPEN: raise AlpacaAPIError(f'order {order_id} is not replaceable')
        n = self._new(symbol=o['symbol'], qty=o['qty'] - o['filled_qty'], side=o['side'], type=o['type'], order_class=o['order_class'], limit_price=round(new_limit_price, 2),
                      status=o['status'], parent=o['parent'], sibling=o['sibling'], replaces=o['id'], client_order_id=o['client_order_id'])
        if o['sibling'] and o['sibling'] in self.orders: self.orders[o['sibling']]['sibling'] = n['id']
        if o['parent'] and o['parent'] in self.orders:
            p = self.orders[o['parent']]; p['legs'] = [n['id'] if l == o['id'] else l for l in p['legs']]
        o['status'] = 'replaced'; o['replaced_by'] = n['id']
        self._settle()
        return {'id': n['id'], 'status': n['status']}

    def get_open_orders(self) -> List[dict]:
        self._settle()
        return [{'id': o['id'], 'client_order_id': o['client_order_id'], 'symbol': o['symbol'], 'status': o['status'], 'side': o['side'], 'qty': float(o['qty']),
                 'filled_qty': float(o['filled_qty']), 'filled_avg_price': self._avg(o), 'stop_price': o['stop_price'], 'limit_price': o['limit_price']}
                for o in self.orders.values() if o['status'] in OPEN]

    def get_open_positions(self) -> List[dict]:
        self._settle()
        return [{'symbol': s, 'qty': q, 'side': 'long' if q > 0 else 'short', 'avg_entry_price': 0.0, 'market_value': 0.0, 'unrealized_pl': 0.0, 'unrealized_plpc': 0.0}
                for s, q in self.positions.items() if q]

    def get_market_calendar(self, start: date, end: date) -> List[dict]:
        return [{'date': start, 'open': dtime(9, 30), 'close': dtime(16, 0)}]

    def get_1min_bars_multi(self, symbols, lookback_minutes=None, **kw) -> dict:
        return {}

    def close_position(self, symbol: str) -> dict:
        self.calls.append(('close_position', {'symbol': symbol}))
        raise AssertionError(f'close_position({symbol}) must never be called by the HOD-break engine')
