"""HOD-break ORDER LIFECYCLE — the same test bodies against two brokers:

* marker `lifecycle` (always runs, seconds): `tests.fakes.fake_alpaca_broker.FakeAlpacaBroker`, Alpaca order semantics in
  process (bracket legs, OCO, replace → new id, partial fills, rejects, shorts on a double sell);
* marker `integration` (skips unless a PAPER key is proven): Alpaca PAPER via the ORB credentials.

    ulimit -v 2500000 && python3 -m pytest tests/integration/test_hod_break_paper_orders.py -q -m lifecycle -p no:cacheprovider
    ulimit -v 2500000 && python3 -m pytest tests/integration/test_hod_break_paper_orders.py -q -x -m integration -p no:cacheprovider -s

Paper proof before ANY order: `ALPACA_ORB_API_KEY/SECRET` present, key prefix `PK` (live keys are `AK…`), `ALPACA_ORB_PAPER`
true, `AlpacaClient.is_paper`, the SDK base URL is the paper endpoint, `/v2/account` answers on paper-api, market open.
Anything else → SKIP (never fail, never live). Real `persistence.database.Database` on a temp sqlite, real engine, entries
driven through `HodBreakEngine._try_enter` with a signal priced off the broker's NBBO. Every test is self-sufficient; the
paper fixture snapshots the symbol's qty / open orders before and cancels our orders + flattens the delta after. Engine
contradictions with the broker truth are collected in FINDINGS and asserted by the LAST test (so `-x` runs every stage);
paper findings are appended to research/bf_zero/parity_review/hod_paper_orders.log.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pytest

REPO = Path(__file__).resolve().parents[2]
LOG_PATH = REPO / 'research' / 'bf_zero' / 'parity_review' / 'hod_paper_orders.log'
SYMBOL = os.environ.get('HOD_PAPER_TEST_SYMBOL', 'SOFI')
RISK_USD = 5.0
MAX_NOTIONAL_USD = 200.0
FILL_WAIT_S = 30.0
LIVE_ORDER_STATUSES = ('new', 'accepted', 'held', 'pending_new', 'partially_filled', 'accepted_for_bidding')
CANCELED_STATUSES = ('canceled', 'cancelled', 'expired', 'replaced')

FINDINGS: List[str] = []
_PAPER: Dict = {}


def finding(msg: str) -> None:
    """Record an engine-vs-broker contradiction (asserted by the last test, logged at teardown)."""
    FINDINGS.append(msg); print(f"\n[FINDING] {msg}")


def note(msg: str) -> None:
    print(f"\n[broker] {msg}")


# ---------------------------------------------------------------------------- paper proof
def _paper_client_or_skip():
    """The ONLY credential path: the ORB keys, and only if every paper proof passes. Never `cfg.alpaca_api_key`."""
    if 'client' in _PAPER:
        return _PAPER['client']
    from dotenv import load_dotenv
    load_dotenv(REPO / '.env')
    key = os.environ.get('ALPACA_ORB_API_KEY', ''); sec = os.environ.get('ALPACA_ORB_API_SECRET', '')
    if not key or not sec:
        pytest.skip('ALPACA_ORB_API_KEY / ALPACA_ORB_API_SECRET missing')
    if not key.startswith('PK'):
        pytest.skip(f"ALPACA_ORB_API_KEY prefix {key[:2]!r} is not a paper key (paper keys start with PK) — refusing to trade")
    if os.environ.get('ALPACA_ORB_PAPER', 'true').lower() not in ('true', '1', 'yes'):
        pytest.skip('ALPACA_ORB_PAPER is not true — the ORB account is not declared paper')
    from data_sources.alpaca_client import AlpacaClient
    client = AlpacaClient(key, sec, paper=True)
    assert client.is_paper, 'AlpacaClient(paper=True) must report is_paper'
    base = str(getattr(client.trading_client, '_base_url', ''))
    if 'paper' not in base.lower():
        pytest.skip(f'trading client base URL {base!r} is not the paper endpoint')
    try:
        acct = client.trading_client.get_account()
    except Exception as e:                                  # a live key answers 401 on paper-api
        pytest.skip(f'paper endpoint rejected the ORB key ({e}) — cannot prove a paper account')
    if str(getattr(acct, 'status', '')).lower() not in ('active', 'accountstatus.active'):
        pytest.skip(f'paper account status {acct.status} is not ACTIVE')
    if not client.trading_client.get_clock().is_open:
        pytest.skip('market closed')
    note(f'PAPER PROVEN: base={base} account={getattr(acct, "account_number", "?")} equity={acct.equity}')
    _PAPER['client'] = client
    return client


# ---------------------------------------------------------------------------- broker helpers (both brokers)
def _symbol_qty(client, symbol: str) -> int:
    for p in client.get_open_positions() or []:
        if p.get('symbol') == symbol:
            return int(p.get('qty') or 0)
    return 0


def _open_order_ids(client, symbol: str) -> set:
    return {o['id'] for o in (client.get_open_orders() or []) if o.get('symbol') == symbol}


def _raw_order(client, order_id: str):
    """The SDK order object (carries `limit_price`, `replaced_by`, `filled_at`, which the client's dict omits)."""
    from alpaca.trading.requests import GetOrderByIdRequest
    return client.trading_client.get_order_by_id(order_id, filter=GetOrderByIdRequest(nested=True))


def _live_leg_id(client, leg_id: Optional[str]) -> Optional[str]:
    """Follow `replaced_by` to the order that is actually working at the broker."""
    seen = 0
    while leg_id and seen < 5:
        o = _raw_order(client, leg_id)
        if str(o.status.value) != 'replaced' or not o.replaced_by:
            return leg_id
        leg_id = str(o.replaced_by); seen += 1
    return leg_id


def _market_order(client, symbol: str, qty: int, side: str) -> str:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY if side == 'buy' else OrderSide.SELL, time_in_force=TimeInForce.DAY)
    return str(client.trading_client.submit_order(req).id)


def _wait(pred, timeout_s: float, every_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        if pred():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(every_s)


# ---------------------------------------------------------------------------- fixtures: two brokers, one contract
@pytest.fixture
def fake_broker():
    from tests.fakes.fake_alpaca_broker import FakeAlpacaBroker
    client = FakeAlpacaBroker(bid=15.00, ask=15.01)
    state = {'client': client, 'fake': True, 'pre_qty': 0, 'pre_orders': set(), 'orders': [], 'timeout_s': 1.0}
    yield state
    q = _symbol_qty(client, SYMBOL)
    if q < 0:
        finding(f'fake broker left SHORT {SYMBOL} qty {q}')


@pytest.fixture
def paper_broker():
    """Paper client + pre-test snapshot; teardown cancels our orders and flattens the qty delta (paper only)."""
    client = _paper_client_or_skip()
    state: Dict = {'client': client, 'fake': False, 'pre_qty': _symbol_qty(client, SYMBOL), 'pre_orders': _open_order_ids(client, SYMBOL), 'orders': [], 'timeout_s': 5.0}
    note(f'{SYMBOL} pre-test broker qty={state["pre_qty"]} pre-existing open orders={len(state["pre_orders"])}')
    yield state
    try:
        for oid in _open_order_ids(client, SYMBOL) - state['pre_orders']:
            try: client.cancel_order(oid); note(f'cleanup: canceled {oid}')
            except Exception as e: note(f'cleanup: cancel {oid} failed: {e}')
        _wait(lambda: not (_open_order_ids(client, SYMBOL) - state['pre_orders']), 10)
        delta = _symbol_qty(client, SYMBOL) - state['pre_qty']
        if delta:
            oid = _market_order(client, SYMBOL, abs(delta), 'sell' if delta > 0 else 'buy')
            note(f'cleanup: flattening delta {delta:+d} via market {oid}')
            _wait(lambda: _symbol_qty(client, SYMBOL) == state['pre_qty'], 20)
        note(f'cleanup: {SYMBOL} qty now {_symbol_qty(client, SYMBOL)} (pre-test {state["pre_qty"]})')
    finally:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, 'a') as f:
            f.write(f"\n=== {datetime.now(timezone.utc).isoformat()} PAPER {SYMBOL} orders={state['orders']}\n")
            f.write('\n'.join(FINDINGS or ['no findings']) + '\n')


@pytest.fixture(params=[pytest.param('fake', marks=pytest.mark.lifecycle), pytest.param('paper', marks=pytest.mark.integration)])
def broker(request):
    """The lifecycle tests run once per broker: the fake on every commit, paper when a PK key is proven."""
    return request.getfixturevalue('fake_broker' if request.param == 'fake' else 'paper_broker')


@pytest.fixture
def db(tmp_path):
    """A REAL Database on a temp sqlite — never data/*.db."""
    from persistence.database import Database
    p = tmp_path / 'trades.db'
    d = Database(db_path=str(p))
    assert Path(d._trades_path).resolve() == p.resolve()
    return d


def cfg(tmp_dir: str, **over) -> dict:
    base = {'enabled': True, 'dry_run': False, 'risk_usd': RISK_USD, 'daily_kill_usd': -600.0, 'weekly_kill_usd': -1500.0,
            'max_notional_usd': MAX_NOTIONAL_USD, 'min_price': 1.0, 'min_adv20': 100_000.0, 'max_spread_bps': 100.0,
            'order_timeout_s': FILL_WAIT_S, 'stream_universe': False, 'stream_list_dir': tmp_dir,
            'params': {'consol_bars': 5, 'consol_pct': 0.04, 'min_dist_open_pct': 5.0, 'rv_lo': 1.0, 'rv_hi': 5.0, 'min_r_pct': 1.0,
                       'cap': 0.006, 'target_r': 2.0, 'max_per_day': 12, 'max_concurrent': 4, 'last_entry_minute': 840, 'flat_minute': 955}}
    base.update(over); return base


def make_engine(db, broker, tmp_path, **over):
    from trading.hod_break_engine import HodBreakEngine
    e = HodBreakEngine(broker['client'], db, stop_monitor=None, notifier=None, cfg=cfg(str(tmp_path), **over), order_stream=None)
    e._roll_session()
    assert e.dry_run is False and e.enabled
    return e


def db_row(db, trade_id: int) -> dict:
    con = sqlite3.connect(str(db._trades_path)); con.row_factory = sqlite3.Row
    try:
        r = con.execute('select * from trades where id=?', (trade_id,)).fetchone(); return dict(r) if r else {}
    finally:
        con.close()


# ---------------------------------------------------------------------------- entry helpers (the engine's own path)
def submit_entry(engine, broker, *, stale_quote=None, ask_shade=0.0):
    """Drive `_try_enter` with a signal priced off the broker's NBBO. Marketable by default (limit >= ask);
    `ask_shade` makes the engine believe the ask is that much lower (a one-tick-stale quote) so its expected fill differs
    from the real one and `_anchor_target_to_fill` has real work; `stale_quote=(bid, ask)` makes the engine believe a
    quote AT its own cap while the real market is higher (the no-chase test)."""
    from trading.hod_break import HodBreakSignal
    from trading.hod_break_engine import Candidate
    client = broker['client']; q = client.get_latest_quote(SYMBOL); bid = float(q['bid_price']); ask = float(q['ask_price'])
    assert bid > 0 and ask > 0, q
    if stale_quote is None:
        level = round(ask - 0.03, 2)
        if round(level * (1 + engine.params.cap), 2) < ask:
            level = round(ask - 0.01, 2)
        stop = round(ask * 0.98, 2)
        if ask_shade:
            engine._quote = lambda symbol, tries=3: (round(bid - ask_shade, 2), round(ask - ask_shade, 2))
    else:
        level = round(stale_quote[1] / (1 + engine.params.cap), 2); stop = round(level * 0.98, 2)
        limit = round(level * (1 + engine.params.cap), 2)
        engine._quote = lambda symbol, tries=3: (round(limit - 0.01, 2), limit)
    sig = HodBreakSignal(bar_idx=0, level=level, stop=stop, dist_open_pct=7.0, rv_profile=2.0)
    cand = Candidate(symbol=SYMBOL, day_open=round(level / 1.07, 2), adv20=50_000_000.0)
    t0 = time.monotonic()
    engine._try_enter(cand, sig, cand.day_open)
    assert cand.rejected_reason == 'ordered', f'entry rejected: {cand.rejected_reason} (bid {bid} ask {ask} level {level} stop {stop})'
    pos = engine.positions[SYMBOL]
    broker['orders'] += [x for x in (pos.order_id, pos.tp_leg_id, pos.sl_leg_id) if x]
    note(f'ENTRY {SYMBOL} order={pos.order_id} coid={pos.client_order_id} tp={pos.tp_leg_id} sl={pos.sl_leg_id} limit={pos.limit_price} stop={pos.stop} '
         f'target={pos.target} x{pos.shares} (bid {bid} ask {ask}) submit {time.monotonic() - t0:.2f}s')
    return pos, bid, ask


def wait_fill(engine, broker, pos):
    filled = _wait(lambda: (engine._process_pending_fills(), SYMBOL in engine.positions and engine.positions[SYMBOL].status == 'open')[1], FILL_WAIT_S)
    if not filled:
        try: broker['client'].cancel_order(pos.order_id)
        finally: pytest.fail(f'{SYMBOL} entry {pos.order_id} not filled within {FILL_WAIT_S:.0f}s (limit {pos.limit_price})')
    raw = _raw_order(broker['client'], pos.order_id)
    lat = (raw.filled_at - raw.submitted_at).total_seconds() if raw.filled_at and raw.submitted_at else float('nan')
    note(f'FILLED {SYMBOL} x{pos.shares} @ {pos.fill_price} broker submit→fill {lat:.3f}s, engine saw it after {(pos.filled_at - pos.submitted_at).total_seconds():.1f}s')
    return pos


def enter_and_fill(engine, broker, **kw):
    pos, _, _ = submit_entry(engine, broker, **kw)
    return wait_fill(engine, broker, pos)


def expected_target(pos) -> float:
    return round(pos.fill_price + 2.0 * (pos.fill_price - pos.stop), 2)


def assert_legs_dead(client, pos, close_id: Optional[str]):
    for name, leg in (('tp', pos.tp_leg_id), ('sl', pos.sl_leg_id)):
        live = _live_leg_id(client, leg)
        if not live: continue
        if not _wait(lambda: client.get_order(live)['status'].lower() in CANCELED_STATUSES + ('filled',), 5):
            finding(f'{name} leg left WORKING after the flat: engine id {leg} → live id {live} status {client.get_order(live)["status"]} (close order {close_id})')
            try: client.cancel_order(live)
            except Exception: pass
        else:
            note(f'{name} leg {leg} (live {live}) → {client.get_order(live)["status"]}')


def force_close_and_verify(engine, broker, db, pos):
    """Engine flat; returns the close order id. Asserts the broker truth; engine contradictions go to FINDINGS."""
    client = broker['client']; qty_before = _symbol_qty(client, SYMBOL); expect_qty = pos.open_qty
    n = engine.force_close_all()
    assert n == 1 and pos.close_order_id, f'force_close_all submitted {n} close order(s); close_order_id={pos.close_order_id}'
    broker['orders'].append(pos.close_order_id)
    close = client.get_order(pos.close_order_id)
    assert close['side'] == 'sell' and close['type'] == 'limit' and close['qty'] == expect_qty, close
    assert close['client_order_id'].startswith('hod-fc-'), close['client_order_id']
    assert pos.pattern_data.get('close_order_id') == pos.close_order_id and pos.pattern_data.get('close_submitted_at'), pos.pattern_data
    assert_legs_dead(client, pos, pos.close_order_id)
    gone = _wait(lambda: (engine.check_exits(), SYMBOL not in engine.positions)[1], FILL_WAIT_S)
    assert gone, f'close order {pos.close_order_id} did not fill within {FILL_WAIT_S:.0f}s: {client.get_order(pos.close_order_id)}'
    row = db_row(db, pos.trade_id)
    assert row['order_status'] == 'closed' and row['exit_reason'] == 'eod' and row['exit_price'] and row['pnl'] is not None, row
    assert _wait(lambda: _symbol_qty(client, SYMBOL) == qty_before - expect_qty, 10), f'broker qty {_symbol_qty(client, SYMBOL)} != {qty_before - expect_qty}'
    note(f'CLOSED {SYMBOL} eod @ {row["exit_price"]} pnl {row["pnl"]:+.2f} close order {pos.close_order_id}')
    return pos.close_order_id


# ---------------------------------------------------------------------------- the lifecycle (both brokers)
def test_1_entry_fill_and_tp_reanchor(broker, db, tmp_path, monkeypatch):
    client = broker['client']; engine = make_engine(db, broker, tmp_path)
    replaces: List = []
    orig = client.replace_order_limit_price
    monkeypatch.setattr(client, 'replace_order_limit_price', lambda oid, px: replaces.append((oid, px, orig(oid, px))) or replaces[-1][2])
    pos, bid, ask = submit_entry(engine, broker, ask_shade=0.01)     # engine expects a fill 1c under the real ask → the re-anchor must move the TP
    parent = client.get_order(pos.order_id)
    assert pos.client_order_id and parent['client_order_id'] == pos.client_order_id, (parent.get('client_order_id'), pos.client_order_id)
    legs = parent['legs']; assert len(legs) == 2, parent
    tp = next(l for l in legs if l['limit_price'] is not None and l['stop_price'] is None); sl = next(l for l in legs if l['stop_price'] is not None)
    assert pos.tp_leg_id == tp['id'] and pos.sl_leg_id == sl['id'], (pos.tp_leg_id, pos.sl_leg_id, legs)
    assert tp['limit_price'] == pytest.approx(pos.target) and sl['stop_price'] == pytest.approx(pos.stop)
    wait_fill(engine, broker, pos)
    assert pos.fill_price and pos.filled_at and pos.status == 'open'
    row = db_row(db, pos.trade_id)
    assert row['order_status'] == 'filled' and row['fill_price'] == pytest.approx(pos.fill_price) and row['filled_at'] and row['filled_qty'] == pos.shares, row
    assert row['take_profit_price'] == pytest.approx(pos.target), row
    t2 = expected_target(pos)
    note(f'target: bracket submitted {tp["limit_price"]} → spec on fill {t2}; engine replace calls: {[(o, p, r.get("id")) for o, p, r in replaces]}')
    if not replaces:
        note('engine re-anchor was a no-op (fill == expected ask): the replace path was NOT exercised this run')
        assert abs(t2 - tp['limit_price']) < 0.01
    original_tp = tp['id']
    live_tp = _live_leg_id(client, original_tp)
    raw = _raw_order(client, live_tp)
    assert float(raw.limit_price) == pytest.approx(t2) and str(raw.status.value).lower() in LIVE_ORDER_STATUSES, (live_tp, raw.limit_price, raw.status)
    assert pos.target == pytest.approx(t2)
    pd_ = json.loads(db_row(db, pos.trade_id)['pattern_data'])
    if pos.tp_leg_id != live_tp or pd_.get('tp_leg_id') != live_tp:
        finding(f'after replace_order_limit_price the working TP is {live_tp} (original {original_tp}, status {client.get_order(original_tp)["status"]}) '
                f'but the engine holds tp_leg_id={pos.tp_leg_id} and pattern_data tp_leg_id={pd_.get("tp_leg_id")}')
    else:
        note(f'engine tracks the working TP leg {live_tp} (original {original_tp}) in memory and pattern_data')
    force_close_and_verify(engine, broker, db, pos)


def test_2_force_close_never_uses_close_position(broker, db, tmp_path, monkeypatch):
    client = broker['client']; engine = make_engine(db, broker, tmp_path)
    pos = enter_and_fill(engine, broker)
    def boom(symbol): raise AssertionError(f'force_close_all called close_position({symbol})')
    monkeypatch.setattr(client, 'close_position', boom)
    force_close_and_verify(engine, broker, db, pos)


def test_3_no_chase_cancel_after_timeout(broker, db, tmp_path):
    client = broker['client']; timeout = broker['timeout_s']; engine = make_engine(db, broker, tmp_path, order_timeout_s=timeout)
    q = client.get_latest_quote(SYMBOL); bid = float(q['bid_price'])
    stale = round(bid * 0.98, 2)
    pos, _, _ = submit_entry(engine, broker, stale_quote=(stale, stale))      # limit ≈ 2% under the real bid: cannot fill
    assert pos.limit_price <= round(bid * 0.985, 2), (pos.limit_price, bid)
    t0 = time.monotonic(); engine._process_pending_fills()
    assert SYMBOL in engine.positions and engine.positions[SYMBOL].status == 'pending', 'canceled before the timeout'
    time.sleep(timeout + 0.2)
    dropped = _wait(lambda: (engine._process_pending_fills(), SYMBOL not in engine.positions)[1], 15)
    st = client.get_order(pos.order_id)
    if st['filled_qty'] > 0:                                                    # the market fell 2% inside the window: not the scenario, clean up
        engine._process_pending_fills()
        if SYMBOL in engine.positions and engine.positions[SYMBOL].status == 'open':
            force_close_and_verify(engine, broker, db, engine.positions[SYMBOL])
        pytest.skip(f'stale-limit order filled ({st}) — not the no-chase scenario; position closed')
    assert dropped, f'pending entry not dropped after {time.monotonic() - t0:.0f}s: {st}'
    assert st['status'].lower() in CANCELED_STATUSES, st
    for leg in (pos.tp_leg_id, pos.sl_leg_id):                                  # a canceled parent takes its held legs
        assert client.get_order(leg)['status'].lower() in CANCELED_STATUSES, client.get_order(leg)
    row = db_row(db, pos.trade_id)
    assert row['order_status'] == 'time_stop_canceled', row
    assert SYMBOL not in engine.entered_today and SYMBOL in engine.seen_today
    note(f'NO-CHASE: {pos.order_id} canceled {time.monotonic() - t0:.1f}s after submit (timeout {timeout:.0f}s), DB time_stop_canceled')


def test_4_restart_resync_and_close(broker, db, tmp_path):
    client = broker['client']; a = make_engine(db, broker, tmp_path)
    pos = enter_and_fill(a, broker, ask_shade=0.01)                             # replaced TP: the restart must rehydrate the WORKING id
    b = make_engine(db, broker, tmp_path)
    n = b.sync_positions()
    assert n == 1 and SYMBOL in b.positions, (n, list(b.positions))
    p2 = b.positions[SYMBOL]
    assert p2.status == 'open' and p2.trade_id == pos.trade_id and p2.shares == pos.shares and p2.fill_price == pytest.approx(pos.fill_price)
    assert p2.tp_leg_id == pos.tp_leg_id and p2.sl_leg_id == pos.sl_leg_id and p2.client_order_id == pos.client_order_id, (p2.tp_leg_id, p2.sl_leg_id)
    assert db_row(db, pos.trade_id)['order_status'] == 'filled', 'sync marked the position exit_pending_verification despite the broker holding it'
    live_tp = _live_leg_id(client, p2.tp_leg_id)
    if live_tp != p2.tp_leg_id:
        finding(f'sync_positions rehydrated tp_leg_id={p2.tp_leg_id} from pattern_data but the working TP is {live_tp}')
    note(f'RESYNC ok: trade {p2.trade_id} x{p2.shares} @ {p2.fill_price} tp={p2.tp_leg_id} sl={p2.sl_leg_id}')
    force_close_and_verify(b, broker, db, p2)


def test_5_tp_fills_in_the_flat_race(broker, db, tmp_path):
    """(a) The TP leg fills between the last poll and the 15:55 flat → no sell, exit booked 'target', no short."""
    client = broker['client']; engine = make_engine(db, broker, tmp_path)
    pos = enter_and_fill(engine, broker)
    pre = _symbol_qty(client, SYMBOL) - pos.shares
    live_tp = _live_leg_id(client, pos.tp_leg_id)
    if broker['fake']:
        client.tick(bid=pos.target + 0.02, ask=pos.target + 0.03)              # the market runs through the target: the TP leg fills
    else:
        q = client.get_latest_quote(SYMBOL); marketable = round(float(q['bid_price']) * 0.995, 2)
        client.replace_order_limit_price(live_tp, marketable)                  # make the real TP leg marketable: it fills at the broker
    assert _wait(lambda: _symbol_qty(client, SYMBOL) == pre, 20), 'TP leg did not fill'
    sells_before = len(client.sells_for(SYMBOL)) if broker['fake'] else len(_open_order_ids(client, SYMBOL))
    n = engine.force_close_all()                                               # the engine has NOT polled since the fill
    pos_gone = SYMBOL not in engine.positions
    row = db_row(db, pos.trade_id); qty = _symbol_qty(client, SYMBOL)
    sells_after = len(client.sells_for(SYMBOL)) if broker['fake'] else len(_open_order_ids(client, SYMBOL))
    if n or pos.close_order_id or sells_after > sells_before:
        finding(f'RACE: TP filled before the flat but force_close_all submitted sell {pos.close_order_id} (n={n}); broker qty {qty} (pre {pre})')
    if qty < pre:
        finding(f'RACE produced a SHORT: {SYMBOL} qty {qty} < {pre}')
    assert pos_gone and row['order_status'] == 'closed' and row['exit_reason'] == 'target' and row['pnl'] is not None, (pos_gone, row)
    assert row['exit_price'] == pytest.approx(float(_raw_order(client, _live_leg_id(client, live_tp)).filled_avg_price))
    note(f'race: TP fill booked as target @ {row["exit_price"]} pnl {row["pnl"]:+.2f}; force_close_all returned {n}, qty {qty} (pre {pre})')


# ---------------------------------------------------------------------------- fake-only lifecycle cases (need market/fill control)
@pytest.mark.lifecycle
def test_b_partial_tp_then_stop_books_by_quantity(fake_broker, db, tmp_path):
    """(b) 30% of the position leaves at the target, the rest at the stop → pnl by quantity, exit_reason 'stop+partial'."""
    client = fake_broker['client']; engine = make_engine(db, fake_broker, tmp_path)
    pos = enter_and_fill(engine, fake_broker)
    part = int(round(0.3 * pos.shares)); rest = pos.shares - part
    client.partial_fill(pos.tp_leg_id, part)                                   # TP prints 4 of 13 at its limit
    assert engine.check_exits() == [] and SYMBOL in engine.positions
    assert pos.closed_qty == part and pos.open_qty == rest and pos.pattern_data['partial_exit'] == 'target'
    assert client.get_order(pos.sl_leg_id)['qty'] == rest                      # the OCO sibling was trimmed
    client.tick(bid=pos.stop - 0.05, ask=pos.stop - 0.04)                       # the stop elects for the remainder
    assert engine.check_exits() == [SYMBOL] and SYMBOL not in engine.positions
    row = db_row(db, pos.trade_id)
    tp_px = float(_raw_order(client, pos.tp_leg_id).filled_avg_price); sl_px = float(_raw_order(client, pos.sl_leg_id).filled_avg_price)
    expected_pnl = part * (tp_px - pos.fill_price) + rest * (sl_px - pos.fill_price)
    assert row['exit_reason'] == 'stop+partial' and row['order_status'] == 'closed', row
    assert row['pnl'] == pytest.approx(expected_pnl, abs=1e-6) and row['exit_price'] == pytest.approx((part * tp_px + rest * sl_px) / pos.shares)
    assert client.get_order(pos.tp_leg_id)['status'] in CANCELED_STATUSES       # nothing keeps selling
    assert _symbol_qty(client, SYMBOL) == 0
    note(f'partial: {part} @ {tp_px} target + {rest} @ {sl_px} stop → pnl {row["pnl"]:+.2f} ({row["exit_reason"]})')


@pytest.mark.lifecycle
def test_c_close_order_partial_resubmits_the_remainder(fake_broker, db, tmp_path):
    """(c) The 15:55 close order fills 5 of 13 and stalls → after FC_RESUBMIT_S the engine re-submits exactly 8."""
    client = fake_broker['client']; engine = make_engine(db, fake_broker, tmp_path)
    pos = enter_and_fill(engine, fake_broker)
    client.auto_fill = False                                                   # the close order will rest
    assert engine.force_close_all() == 1
    first = pos.close_order_id; assert client.get_order(first)['status'] == 'new'
    client.partial_fill(first, 5)
    assert engine.force_close_all() == 0                                        # inside FC_RESUBMIT_S: still working, nothing re-sent
    assert pos.closed_qty == 5 and pos.open_qty == pos.shares - 5 and SYMBOL in engine.positions
    pos.close_submitted_at -= timedelta(seconds=engine.FC_RESUBMIT_S + 1)       # the clock passes the re-submit window
    assert engine.force_close_all() == 1
    second = pos.close_order_id; assert second != first
    assert client.get_order(first)['status'] == 'canceled' and client.get_order(first)['filled_qty'] == 5
    assert client.get_order(second)['qty'] == pos.shares - 5 and client.get_order(second)['client_order_id'].startswith('hod-fc-')
    assert pos.fc_attempts == 2 and pos.pattern_data['closed_qty'] == 5 and pos.pattern_data['close_order_id'] == second
    client.auto_fill = True; client.tick()                                      # the remainder fills
    assert engine.check_exits() == [SYMBOL]
    row = db_row(db, pos.trade_id)
    p1 = float(_raw_order(client, first).filled_avg_price); p2 = float(_raw_order(client, second).filled_avg_price)
    assert row['exit_reason'] == 'eod+partial' and row['pnl'] == pytest.approx(5 * (p1 - pos.fill_price) + (pos.shares - 5) * (p2 - pos.fill_price), abs=1e-6)
    assert _symbol_qty(client, SYMBOL) == 0 and len(client.sells_for(SYMBOL)) == 2
    note(f'close partial: {first} 5 @ {p1} then {second} {pos.shares - 5} @ {p2} → {row["exit_reason"]} pnl {row["pnl"]:+.2f}')


@pytest.mark.lifecycle
def test_d_restart_with_working_close_order_sells_once(fake_broker, db, tmp_path):
    """(d) Restart while the close order is working: the new engine adopts it from pattern_data — no second sell."""
    client = fake_broker['client']; a = make_engine(db, fake_broker, tmp_path)
    pos = enter_and_fill(a, fake_broker)
    client.auto_fill = False
    assert a.force_close_all() == 1
    close_id = pos.close_order_id
    b = make_engine(db, fake_broker, tmp_path)
    assert b.sync_positions() == 1
    p2 = b.positions[SYMBOL]
    assert p2.close_order_id == close_id and p2.close_submitted_at is not None and p2.status == 'open'
    assert b.force_close_all() == 0 and p2.close_order_id == close_id            # adopted, still working: nothing re-sent
    assert len(client.sells_for(SYMBOL)) == 1
    assert not [c for c in client.calls if c[0] == 'submit_limit_sell_order' and c[1]['client_order_id'] != pos.pattern_data['close_client_order_id']]
    client.auto_fill = True; client.tick()
    assert b.check_exits() == [SYMBOL]
    row = db_row(db, pos.trade_id)
    assert row['exit_reason'] == 'eod' and row['order_status'] == 'closed' and _symbol_qty(client, SYMBOL) == 0
    note(f'restart: close order {close_id} adopted by the new engine, one sell total, exit eod @ {row["exit_price"]}')


@pytest.mark.lifecycle
def test_e_pending_cancel_then_fill_is_tracked(fake_broker, db, tmp_path):
    """(e) The no-chase cancel answers pending_cancel and the order then fills → the position is tracked, not dropped."""
    client = fake_broker['client']; engine = make_engine(db, fake_broker, tmp_path, order_timeout_s=0.0)
    client.auto_fill = False
    pos, _, _ = submit_entry(engine, fake_broker)
    client.defer_cancel(pos.order_id)
    engine._process_pending_fills()                                            # timeout → cancel → broker says pending_cancel
    assert client.get_order(pos.order_id)['status'] == 'pending_cancel'
    assert SYMBOL in engine.positions and engine.positions[SYMBOL].status == 'pending', 'engine assumed the cancel'
    assert db_row(db, pos.trade_id)['order_status'] == 'pending_new'
    client.auto_fill = True; client.tick()                                      # the pending-cancel order fills anyway
    engine._process_pending_fills()
    assert engine.positions[SYMBOL].status == 'open' and pos.fill_price == pytest.approx(client.ask) and pos.shares == client.get_order(pos.order_id)['filled_qty']
    row = db_row(db, pos.trade_id)
    assert row['order_status'] == 'filled' and row['fill_price'] == pytest.approx(pos.fill_price)
    for leg in (pos.tp_leg_id, pos.sl_leg_id):
        assert client.get_order(leg)['status'] == 'new'
    assert SYMBOL in engine.entered_today
    force_close_and_verify(engine, fake_broker, db, pos)


@pytest.mark.lifecycle
@pytest.mark.integration
def test_z_engine_invariants_hold():
    """Every contradiction the broker truth exposed above (empty = the engine tracks its own orders)."""
    assert not FINDINGS, '\n'.join(FINDINGS)
