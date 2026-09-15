#!/usr/bin/env python3
"""Randomized differential test: HOD-break SPEC (whole-day `simulate`/`detect`) vs the LIVE ENGINE fed one closed bar
at a time through `_on_bar_close` + `drain_bar_events` (the drain path), with the replay-test mocks.

Per synthetic symbol-day the EXPECTED order is the spec's first `detect` signal that passes simulate's fill/r_min
gates with the engine's ask (= break-bar close + 0.01) as the entry; the ACTUAL order is what `submit_bracket_order`
received (limit = level x 1.006, sl = stop) plus the pattern_data the engine saved (level, consol_low, rv_profile).

Delivery scenarios: A in-order stream (with random duplicate re-delivery); B two batches (REST backfill of the first
k bars via `_backfill`, then stream); C one random early bar arrives LATE via `_on_bar_close` after later bars.
Premarket bars (minute < 570) and a post-close bar are interleaved and must be ignored.

Usage: python research/bf_zero/parity_review/diff_detect_incremental.py [n_days] [seed]
"""
import logging
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
logging.disable(logging.CRITICAL)

from data_sources.alpaca_client import AlpacaClient          # noqa: E402
from persistence.database import Database                    # noqa: E402
from trading.stop_monitor import StopMonitor                 # noqa: E402
from trading.hod_break import HodBreakParams, detect, simulate, OPEN_MINUTE  # noqa: E402
from trading.hod_break_engine import HodBreakEngine, Candidate  # noqa: E402

BASE = datetime(2026, 9, 15, 13, 30, tzinfo=timezone.utc)      # 09:30 ET on an EDT day
P = HodBreakParams(max_per_day=10 ** 6, max_concurrent=10 ** 6)  # isolate detection from the book caps
CFG = {'enabled': True, 'dry_run': False, 'risk_usd': 100.0, 'daily_kill_usd': -1e9, 'weekly_kill_usd': -1e9, 'max_notional_usd': 1e12,
       'min_price': 1.0, 'min_adv20': 1.0, 'max_spread_bps': 1e9, 'order_timeout_s': 20.0, 'stream_universe': False,
       'params': {k: getattr(P, k) for k in P.__dataclass_fields__}}


# ----------------------------------------------------------------------------- synthetic days
def gen_day(rng):
    """RTH bars as list of (minute, o, h, l, c, v) sorted by minute, plus premarket noise bars and adv20."""
    n_slots = 390
    p_miss = rng.choice([0.0, 0.03, 0.15])
    have = rng.random(n_slots) >= p_miss
    if rng.random() < 0.10: have[0] = False                   # no 09:30 print
    end = int(rng.integers(60, 391)); have[end:] = False      # day may end early (halt/delist); >= 60 minutes
    minutes = np.flatnonzero(have) + OPEN_MINUTE
    n = len(minutes)
    px = float(rng.uniform(1.5, 60.0))
    # regime per bar: drive (trend up), consolidate (tiny range), break (spike), random walk
    n_drive = int(rng.integers(3, 16)); vol_rw = rng.choice([0.002, 0.004, 0.008])
    bars = []; prev_c = px; hod = -1.0
    i = 0
    while i < n:
        if i < n_drive:
            r = abs(rng.normal(0.006, 0.004)); wick = 0.002
            seg = 1
        elif rng.random() < 0.35:                              # consolidation 5-9 bars just under the running HOD
            seg = int(rng.integers(5, 10)); r = None
        else:
            seg = int(rng.integers(1, 6)); r = None
        for k in range(seg):
            if i >= n: break
            o = prev_c * (1 + (rng.normal(0, 0.001) if rng.random() < 0.3 else 0.0))
            if i < n_drive:
                c = o * (1 + abs(rng.normal(0.006, 0.004))); h = max(o, c) * (1 + abs(rng.normal(0, 0.002))); l = min(o, c) * (1 - abs(rng.normal(0, 0.002)))
            elif r is None and seg >= 5:
                lo_band = hod * (1 - rng.uniform(0.005, 0.035)) if hod > 0 else o * 0.99
                c = float(rng.uniform(lo_band, hod * 0.999 if hod > 0 else o)); o = float(np.clip(o, lo_band, hod * 0.999 if hod > 0 else o))
                h = max(o, c) * (1 + abs(rng.normal(0, 0.0005))); l = min(o, c) * (1 - abs(rng.normal(0, 0.0005)))
                if hod > 0: h = min(h, hod * 0.9999)
                l = max(l, lo_band * 0.995)
            else:
                spike = rng.random() < 0.15
                c = o * (1 + rng.normal(0.004 if spike else 0.0, vol_rw)); h = max(o, c) * (1 + abs(rng.normal(0, vol_rw))); l = min(o, c) * (1 - abs(rng.normal(0, vol_rw)))
                if spike and hod > 0 and rng.random() < 0.7: h = max(h, hod * (1 + rng.uniform(-0.001, 0.004)))
            o, h, l, c = round(o, 2), round(h, 2), round(l, 2), round(c, 2)
            h = max(h, o, c); l = min(l, o, c); l = max(l, 0.01)
            v = float(int(rng.lognormal(9.5 if i < 5 else 8.5, 0.8)))
            bars.append((int(minutes[i]), o, h, l, c, v)); prev_c = c; hod = max(hod, h); i += 1
    tot = sum(b[5] for b in bars)
    adv20 = tot / (rng.uniform(0.2, 3.0) * 0.4)                 # so rv at a mid-morning break lands around [0.3, 6)
    pm = [(int(m), round(px * rng.uniform(0.9, 1.3), 2), round(px * 1.35, 2), round(px * 0.85, 2), round(px, 2), float(int(rng.lognormal(10, 1))))
          for m in rng.choice(np.arange(240, OPEN_MINUTE), size=int(rng.integers(0, 6)), replace=False)]
    post = [(960 + int(rng.integers(0, 60)), round(px, 2), round(px * 1.5, 2), round(px * 0.5, 2), round(px, 2), 1e7)] if rng.random() < 0.3 else []
    return bars, pm, post, float(adv20)


def arrays(bars):
    a = np.array([b[1:] for b in bars], dtype=float); m = np.array([b[0] for b in bars], dtype=int)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], m


def bar_dict(b):
    m, o, h, l, c, v = b
    return {'timestamp': BASE + timedelta(minutes=m - OPEN_MINUTE), 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}


def expected_order(bars, adv20):
    """The spec's first signal + simulate's fill/r_min gates with the engine's ask as the entry (engine rounding)."""
    o, h, l, c, v, m = arrays(bars)
    sig = detect(o, h, l, v, m, adv20, P)
    if sig is None:
        return None, None
    ask = round(c[sig.bar_idx] + 0.01, 2)
    limit = round(sig.level * (1.0 + P.cap), 2); stop = round(sig.stop, 2)
    if stop >= limit or ask > limit:
        return sig, None
    r = ask - stop
    if r <= 0 or r / ask * 100.0 < P.min_r_pct:
        return sig, None
    return sig, {'bar_idx': sig.bar_idx, 'minute': int(m[sig.bar_idx]), 'level': sig.level, 'stop': sig.stop, 'rv': sig.rv_profile, 'limit': limit, 'sl': stop}


# ----------------------------------------------------------------------------- engine harness
def make_engine(tmpdir, sym, adv20, last_close, clock):
    trades = os.path.join(tmpdir, f'{sym}.db'); con = sqlite3.connect(trades)
    con.execute("create table trades (id integer primary key, strategy text, trade_date text, pnl real, symbol text, order_status text)"); con.commit(); con.close()
    alp = MagicMock(spec=AlpacaClient); db = MagicMock(spec=Database); sm = MagicMock(spec=StopMonitor); sm.polling_mode = False
    db._trades_path = trades; db.get_active_universe.return_value = [{'symbol': sym, 'avg_volume_daily': adv20}]; db.get_open_trades.return_value = []
    saved = []
    db.save_trade.side_effect = lambda rec: saved.append(rec) or 1
    alp.submit_bracket_order.side_effect = lambda **kw: {'id': 'o-' + kw['symbol'], 'status': 'accepted', 'legs': [{'id': 'tp', 'limit_price': kw['tp_price'], 'stop_price': None}, {'id': 'sl', 'limit_price': None, 'stop_price': kw['sl_price']}]}
    alp.get_order.return_value = {'status': 'accepted', 'filled_qty': 0}; alp.get_open_positions.return_value = []; alp.cancel_order.return_value = True
    alp.get_latest_quote.side_effect = lambda s: {'bid_price': round(last_close[0] - 0.01, 2), 'ask_price': round(last_close[0] + 0.01, 2)}
    e = HodBreakEngine(alp, db, sm, cfg=CFG); e.session_date = '2026-09-15'
    e.candidates[sym] = Candidate(symbol=sym, day_open=0.0, adv20=adv20, subscribed=True, backfill_ok=True)   # the streamed-universe path
    return e, alp, saved


def feed(e, sym, b, last_close, clock):
    last_close[0] = b[4]; clock[0] = b[0] + 1
    e._on_bar_close(sym, bar_dict(b)); e.drain_bar_events()


def run_engine(bars, pm, post, adv20, scenario, rng, tmpdir, day_no=0, late_target=None):
    sym = 'ZZ%06d' % day_no; last_close = [bars[0][4]]; clock = [OPEN_MINUTE]
    e, alp, saved = make_engine(tmpdir, sym, adv20, last_close, clock)
    log = []
    with patch.object(HodBreakEngine, '_minute_of_day', side_effect=lambda: clock[0]):
        order = list(bars)
        noise = list(pm) + list(post)
        if scenario == 'A':
            for i, b in enumerate(order):
                if noise and rng.random() < 0.2: feed(e, sym, noise.pop(), last_close, clock)
                feed(e, sym, b, last_close, clock); log.append(b[0])
                if rng.random() < 0.05: e._on_bar_close(sym, bar_dict(b)); e.drain_bar_events(); log.append(('dup', b[0]))
        elif scenario == 'B':
            k = int(rng.integers(1, len(order)))                   # first k bars via REST backfill
            e.candidates[sym].backfill_ok = False
            df = pd.DataFrame([bar_dict(b) for b in order[:k]] + [bar_dict(b) for b in pm])
            alp.get_1min_bars_multi.side_effect = lambda syms, lookback_minutes=30: {s: df for s in syms}
            last_close[0] = order[k - 1][4]; clock[0] = order[k - 1][0] + 1
            e._backfill([e.candidates[sym]]); log.append(('backfill', k))
            for b in order[k:]: feed(e, sym, b, last_close, clock); log.append(b[0])
        elif scenario == 'C':
            # one early bar arrives late: targeted at the bars that matter (break bar, consolidation bars, the HOD bar) 70% of the time
            sig_idx = late_target
            if sig_idx is not None and rng.random() < 0.7:
                j = int(np.clip(sig_idx - int(rng.integers(0, 8)), 0, len(order) - 3))
            else:
                j = int(rng.integers(0, max(1, len(order) - 3)))
            d = int(rng.integers(1, 4))
            seq = order[:j] + order[j + 1:j + 1 + d] + [order[j]] + order[j + 1 + d:]
            for b in seq:
                feed(e, sym, b, last_close, clock); log.append(b[0])
            log.append(('late', j, d))
        elif scenario == 'D':
            # a bar past last_entry_minute arrives BEFORE an earlier bar (e.g. the break bar): 931 then 930
            cut = [i for i, b in enumerate(order) if b[0] > P.last_entry_minute]
            if len(cut) >= 1 and cut[0] >= 2:
                j = cut[0] - 1 if late_target is None else int(np.clip(late_target, 1, cut[0] - 1))
                seq = order[:j] + [order[cut[0]]] + [order[j]] + order[j + 1:cut[0]] + order[cut[0] + 1:]
            else:
                seq = order
            for b in seq: feed(e, sym, b, last_close, clock); log.append(b[0])
            log.append(('post_cut_first', len(cut)))
        elif scenario == 'E':
            # stream hole: bar j never streams; d bars later a WS reconnect is detected and the REST re-backfill (whole day so far) repairs it
            j = int(np.clip((late_target if late_target is not None else int(rng.integers(0, len(order) - 3))) - int(rng.integers(0, 8)), 0, len(order) - 3))
            d = int(rng.integers(1, 4))
            for b in order[:j] + order[j + 1:j + 1 + d]: feed(e, sym, b, last_close, clock); log.append(b[0])
            cand = e.candidates[sym]; cand.backfill_ok = False                       # what _check_stream_outage does
            df = pd.DataFrame([bar_dict(b) for b in order[:j + 1 + d]])
            alp.get_1min_bars_multi.side_effect = lambda syms, lookback_minutes=30: {s: df for s in syms}
            e._backfill([cand]); log.append(('hole', j, d))
            for b in order[j + 1 + d:]: feed(e, sym, b, last_close, clock); log.append(b[0])
        for b in noise: feed(e, sym, b, last_close, clock)
    calls = alp.submit_bracket_order.call_args_list
    if not calls:
        return None, e.candidates[sym].rejected_reason, log, sym
    kw = calls[0].kwargs; pdat = __import__('json').loads(saved[0]['pattern_data']) if saved else {}
    return {'limit': kw['limit_price'], 'sl': kw['sl_price'], 'level': pdat.get('level'), 'stop': pdat.get('consol_low'), 'rv': pdat.get('rv_profile'), 'n_orders': len(calls)}, e.candidates[sym].rejected_reason, log, sym


def main():
    n_days = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260915
    rng = np.random.default_rng(seed)
    tmpdir = tempfile.mkdtemp(prefix='hod_diff_')
    stats = {'days': 0, 'spec_signal': 0, 'expected_order': 0, 'match': 0, 'expected_stale': 0}
    findings = []
    for d in range(n_days):
        day_seed = int(rng.integers(0, 2 ** 31)); drng = np.random.default_rng(day_seed)
        bars, pm, post, adv20 = gen_day(drng)
        u = drng.random()
        scenario = 'A' if u < 0.55 else ('B' if u < 0.75 else ('C' if u < 0.85 else ('D' if u < 0.92 else 'E')))
        sig, exp = expected_order(bars, adv20)
        stats['days'] += 1; stats['spec_signal'] += sig is not None; stats['expected_order'] += exp is not None
        try:
            got, rej, log, sym = run_engine(bars, pm, post, adv20, scenario, drng, tmpdir, day_no=d, late_target=None if sig is None else sig.bar_idx)
        except Exception as ex:
            findings.append(dict(kind='ENGINE_RAISED', scenario=scenario, day_seed=day_seed, err=repr(ex))); continue
        # scenario B: a spec signal strictly inside the backfill batch is the documented stale_break (not a mismatch)
        if scenario in ('B', 'E') and sig is not None:
            t = [x for x in log if isinstance(x, tuple) and x[0] in ('backfill', 'hole')][0]
            k = t[1] if t[0] == 'backfill' else t[1] + 1 + t[2]          # bars present after the (re)backfill
            if sig.bar_idx < k - 1:
                stats['expected_stale'] += 1
                if got is not None or rej != 'stale_break':
                    findings.append(dict(kind='STALE_EXPECTED_BUT', scenario=scenario, day_seed=day_seed, got=got, rej=rej, k=k, sig_bar=sig.bar_idx))
                continue
        if exp is None and got is None:
            stats['match'] += 1; continue
        if exp is not None and got is not None and abs(got['limit'] - exp['limit']) < 1e-9 and abs(got['sl'] - exp['sl']) < 1e-9 \
                and abs(got['level'] - exp['level']) < 1e-9 and abs(got['stop'] - exp['stop']) < 1e-9 and abs(got['rv'] - exp['rv']) < 1e-9 and got['n_orders'] == 1:
            stats['match'] += 1; continue
        findings.append(dict(kind='MISMATCH', scenario=scenario, day_seed=day_seed, expected=exp, got=got, rej=rej, n_bars=len(bars),
                             sig_bar=None if sig is None else sig.bar_idx, sig_minute=None if sig is None else int(bars[sig.bar_idx][0]),
                             tail=[x for x in log if isinstance(x, tuple)][-3:], adv20=adv20))
    print(f'seed {seed} | {stats}')
    by = {}
    for f in findings: by[(f['kind'], f['scenario'])] = by.get((f['kind'], f['scenario']), 0) + 1
    print('findings by (kind, scenario):', by)
    for f in findings[:40]:
        print(f)
    out = os.path.join(ROOT, 'research/bf_zero/parity_review/diff_findings.csv')
    pd.DataFrame(findings).to_csv(out, index=False); print('written', out, len(findings))
    return findings


if __name__ == '__main__':
    fs = main()
    sys.exit(1 if fs else 0)
