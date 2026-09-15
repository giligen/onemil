#!/usr/bin/env python3
"""Targeted engine-state probes (read-only, mocks): (1) outage mark defeated by the m[0]==09:30 shortcut when the
REST re-backfill fails; (2) kill-rail block is not sticky → a LATER break is ordered; (3) scan-admitted candidate
with no 09:30 print vs the day_open sanity check."""
import logging, os, sqlite3, sys, tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; sys.path.insert(0, ROOT); os.chdir(ROOT); logging.disable(logging.CRITICAL)
from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.stop_monitor import StopMonitor
from trading.hod_break import HodBreakParams, detect, OPEN_MINUTE
from trading.hod_break_engine import HodBreakEngine, Candidate
sys.path.insert(0, f'{ROOT}/research/bf_zero/parity_review')
from diff_detect_incremental import gen_day, arrays, bar_dict, CFG, P, BASE

def world(sym, adv20, last_close):
    d = tempfile.mkdtemp(); trades = os.path.join(d, 't.db'); con = sqlite3.connect(trades)
    con.execute("create table trades (id integer primary key, strategy text, trade_date text, pnl real, symbol text, order_status text)"); con.commit(); con.close()
    alp = MagicMock(spec=AlpacaClient); db = MagicMock(spec=Database); sm = MagicMock(spec=StopMonitor); sm.polling_mode = False
    db._trades_path = trades; db.get_active_universe.return_value = []; db.get_open_trades.return_value = []; db.save_trade.return_value = 1
    alp.submit_bracket_order.side_effect = lambda **kw: {'id': 'o', 'status': 'accepted', 'legs': []}
    alp.get_latest_quote.side_effect = lambda s: {'bid_price': round(last_close[0] - 0.01, 2), 'ask_price': round(last_close[0] + 0.01, 2)}
    e = HodBreakEngine(alp, db, sm, cfg=CFG); e.session_date = '2026-09-15'
    return e, alp, sm, trades

def find_day(seed0, need_two=False):
    rng = np.random.default_rng(seed0)
    while True:
        s = int(rng.integers(0, 2**31)); bars, pm, post, adv = gen_day(np.random.default_rng(s))
        o, h, l, c, v, m = arrays(bars); sig = detect(o, h, l, v, m, adv, P)
        if sig is None or sig.bar_idx + 8 >= len(bars) or bars[0][0] != OPEN_MINUTE: continue
        if need_two:
            sig2 = detect(o, h, l, v, m, adv, P, start_idx=sig.bar_idx + 1)
            if sig2 is None: continue
            return s, bars, adv, sig, sig2
        return s, bars, adv, sig, None

# ---------- probe 1: outage → REST fails → the next streamed bar re-enables evaluation on a day with a hole
s, bars, adv, sig, _ = find_day(11)
last = [bars[0][4]]; e, alp, sm, _ = world('P1', adv, last); e.candidates['P1'] = Candidate('P1', 0.0, adv, subscribed=True, backfill_ok=True)
clock = [OPEN_MINUTE]
with patch.object(HodBreakEngine, '_minute_of_day', side_effect=lambda: clock[0]):
    j = sig.bar_idx - 3                                        # the hole: bars j..sig.bar_idx-1 never stream (outage)
    for b in bars[:j]: clock[0] = b[0] + 1; last[0] = b[4]; e._on_bar_close('P1', bar_dict(b)); e.drain_bar_events()
    sm.ws_generation = 1; e._ws_gen = 0                        # reconnect happened during the hole
    alp.get_1min_bars_multi.side_effect = lambda syms, lookback_minutes=30: {}     # REST fails (or returns nothing) this tick
    clock[0] = bars[sig.bar_idx][0]
    e.process_tick()                                            # _check_stream_outage marks; _backfill gets nothing
    c = e.candidates['P1']; print('after outage tick: backfill_ok', c.backfill_ok, 'tries', c.backfill_tries)
    b = bars[sig.bar_idx]; clock[0] = b[0] + 1; last[0] = b[4]; e._on_bar_close('P1', bar_dict(b)); e.drain_bar_events()   # the break bar streams (post-reconnect)
    print('after next streamed bar: backfill_ok', c.backfill_ok, '| rejected', c.rejected_reason, '| ordered', alp.submit_bracket_order.called,
          '| bars held', c.n_bars, 'of', sig.bar_idx + 1, '| in retry list next tick:', (c.subscribed and not c.backfill_ok and c.rejected_reason is None))
    print('  -> PROBE1', 'DEFECT: evaluated with a hole, outage mark lost' if c.backfill_ok else 'ok', f'(day seed {s}, hole bars {j}..{sig.bar_idx-1})')

# ---------- probe 2: kill rail blocks the FIRST break (no rejected_reason) → rails lift → the SECOND break is ordered
s, bars, adv, sig, sig2 = find_day(23, need_two=True)
last = [bars[0][4]]; e, alp, sm, trades = world('P2', adv, last); e.candidates['P2'] = Candidate('P2', 0.0, adv, subscribed=True, backfill_ok=True)
clock = [OPEN_MINUTE]
con = sqlite3.connect(trades); con.execute("insert into trades (strategy, trade_date, pnl, symbol, order_status) values ('hod_break','2026-09-15',-5000,'X','closed')"); con.commit(); con.close()
e.daily_kill_usd = -600.0
with patch.object(HodBreakEngine, '_minute_of_day', side_effect=lambda: clock[0]):
    for b in bars[:sig.bar_idx + 1]: clock[0] = b[0] + 1; last[0] = b[4]; e._on_bar_close('P2', bar_dict(b)); e.drain_bar_events()
    c = e.candidates['P2']; print('first break under kill rail: ordered', alp.submit_bracket_order.called, '| rejected_reason', c.rejected_reason, '| next_idx', c.next_idx)
    con = sqlite3.connect(trades); con.execute("update trades set pnl=+100"); con.commit(); con.close()        # a winner lifts the rail
    for b in bars[sig.bar_idx + 1:sig2.bar_idx + 1]: clock[0] = b[0] + 1; last[0] = b[4]; e._on_bar_close('P2', bar_dict(b)); e.drain_bar_events()
    print('second break after the rail lifts: ordered', alp.submit_bracket_order.called, '| rejected', c.rejected_reason, f'(spec first break bar {sig.bar_idx}, second {sig2.bar_idx}; day seed {s})')
    print('  -> PROBE2', 'DEFECT: a break the spec never trades is ordered' if alp.submit_bracket_order.called else 'ok')

# ---------- probe 3: scan-admitted candidate (day_open from the scanner) whose first RTH bar is 09:31 (no 09:30 print)
s, bars, adv, sig, _ = find_day(37)
bars = [b for b in bars if b[0] != OPEN_MINUTE]                # no 09:30 print
o, h, l, c_, v, m = arrays(bars); sig = detect(o, h, l, v, m, adv, P)
last = [bars[0][4]]; e, alp, sm, _ = world('P3', adv, last)
e.candidates['P3'] = Candidate('P3', day_open=bars[0][1], adv20=adv, subscribed=True, backfill_ok=False)   # scanner passes the daily open = first trade
clock = [OPEN_MINUTE]
with patch.object(HodBreakEngine, '_minute_of_day', side_effect=lambda: clock[0]):
    df = pd.DataFrame([bar_dict(b) for b in bars[:5]]); alp.get_1min_bars_multi.side_effect = lambda syms, lookback_minutes=30: {s_: df for s_ in syms}
    clock[0] = bars[4][0] + 1; e._backfill([e.candidates['P3']]); c = e.candidates['P3']
    print('scan-admitted, first bar 09:31, day_open == that bar open: backfill_ok', c.backfill_ok, 'tries', c.backfill_tries)
    for b in bars[5:(sig.bar_idx + 1) if sig else 40]: clock[0] = b[0] + 1; last[0] = b[4]; e._on_bar_close('P3', bar_dict(b)); e.drain_bar_events()
    print('  streamed to the break: ordered', alp.submit_bracket_order.called, '| rejected', c.rejected_reason, '| backfill_ok', c.backfill_ok)
    # variant: the scanner's daily open differs from the 09:31 bar open by 2 cents (opening print on another venue / odd-lot rules)
    e2, alp2, sm2, _ = world('P4', adv, last); e2.candidates['P4'] = Candidate('P4', day_open=round(bars[0][1] + 0.02, 2), adv20=adv, subscribed=True, backfill_ok=False)
    alp2.get_1min_bars_multi.side_effect = lambda syms, lookback_minutes=30: {s_: df for s_ in syms}
    clock[0] = bars[4][0] + 1
    for _ in range(3): e2._backfill([e2.candidates['P4']])
    c4 = e2.candidates['P4']; print('  day_open off by 2c: backfill_ok', c4.backfill_ok, 'tries', c4.backfill_tries, 'REST calls', alp2.get_1min_bars_multi.call_count, '-> never evaluated, one REST call per tick')
