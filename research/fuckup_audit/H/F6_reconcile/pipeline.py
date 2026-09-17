#!/usr/bin/env python3
"""ONE parameterised F6 red-to-green pipeline. Every convention that differs between

  A = H/F6/f6_pdr_book.py over C/pop_c.csv (B/build_candidates4.py -> bf_zero/build_candidates.py, BFZ_SLIP=0)
  B = H/F6_rebuild/{prefilter,scan,book}.py
  E = the LIVE engine: trading/red_to_green.py + trading/hod_break.{entry_fill,walk_exit,run_book}

is a switch, so a single flip can be attributed.  Pure functions + two read-only sqlite handles.
"""
import os, sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

ROOT = '/home/ec2-user/onemil'
ET = ZoneInfo('America/New_York'); UTC = ZoneInfo('UTC')
OPEN_M, RTH_HI, FLAT_M = 570, 959, 955
STOP_SLIP = 0.999

# --------------------------------------------------------------- the switches
A_CFG = dict(bars='cache_first', level_mult=1.000, floor_den='o0', stop='excl', prev='panel',
             price_on='level', day_open5=False, min_bars=10, late_on='fill', late_m=841,
             scan='break_then_floor')
B_CFG = dict(bars='sip_first', level_mult=1.003, floor_den='runlo', stop='incl', prev='uni_pref',
             price_on='entry', day_open5=True, min_bars=0, late_on='fill', late_m=841,
             scan='break_then_floor')
E_CFG = dict(bars='cache_first', level_mult=1.003, floor_den='runlo', stop='incl', prev='panel',
             price_on='entry', day_open5=False, min_bars=0, late_on='signal', late_m=840,
             scan='floor_and_break')

_off = {}


def et_off(day):
    if day not in _off:
        d = datetime.fromisoformat(day + 'T12:00:00').replace(tzinfo=UTC)
        _off[day] = -int(d.astimezone(ET).utcoffset().total_seconds() // 60)
    return _off[day]


class Bars:
    """Both stores, read-only. `get(sym, day, order)` -> (minutes, o,h,l,c,v, source)."""

    def __init__(self):
        self.cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
        self.sip = sqlite3.connect(f'file:{ROOT}/research/bf_zero/bars_sip.db?mode=ro', uri=True, timeout=120)

    def _cache(self, sym, day):
        return self.cache.execute('select timestamp,open,high,low,close,volume from intraday_bars_1min '
                                  'where symbol=? and bar_date=?', (sym, day)).fetchall()

    def _sip(self, sym, day):
        return self.sip.execute('select t,o,h,l,c,v from bars where symbol=? and day=?', (sym, day)).fetchall()

    def raw(self, sym, day, store):
        rows = self._cache(sym, day) if store == 'cache' else self._sip(sym, day)
        off = et_off(day)
        seen = {}
        for t, o, h, l, c, v in rows:
            m = int(t[11:13]) * 60 + int(t[14:16]) - off
            if OPEN_M <= m <= RTH_HI and m not in seen:
                seen[m] = (float(o), float(h), float(l), float(c), float(v))
        ms = sorted(seen)
        if not ms:
            return None
        arr = np.array([seen[m] for m in ms], dtype=float)
        return np.array(ms, dtype=int), arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

    def get(self, sym, day, order):
        first, second = ('cache', 'sip') if order == 'cache_first' else ('sip', 'cache')
        r = self.raw(sym, day, first)
        if r is not None:
            return r + (first,)
        r = self.raw(sym, day, second)
        if r is not None:
            return r + (second,)
        return None


def walk(o, h, l, c, m, fi, entry, stop, mode):
    """Exit walk from the bar AFTER the fill bar. mode: hold | r2 | partial.
    Semantics of trading.hod_break.walk_exit (identical to A's walk_2r and B's walk_exits):
    eod (m>=955, fill at open) beats stop (l<=stop, fill min(stop,open)*0.999) beats target (c>=target, fill target).
    Returns (exit_m, exit_type, grossR, legs)."""
    R = entry - stop
    tgt = entry + 2.0 * R
    n = len(o)
    if mode == 'partial':
        gross = 0.0; part = False; cur = stop; legs = []
        for k in range(fi + 1, n):
            if m[k] >= FLAT_M:
                w = 0.5 if part else 1.0
                gross += w * (o[k] - entry) / R; legs.append((w, 'eod'))
                return int(m[k]), ('pp+eod' if part else 'eod'), gross, legs
            if l[k] <= cur:
                px = min(cur, o[k]) * STOP_SLIP
                w = 0.5 if part else 1.0
                gross += w * (px - entry) / R; legs.append((w, 'stop'))
                return int(m[k]), ('pp+stop' if part else 'stop'), gross, legs
            if (not part) and c[k] >= tgt:
                gross += 0.5 * (tgt - entry) / R; legs.append((0.5, 'target'))
                part = True; cur = entry
        if n > fi + 1:
            w = 0.5 if part else 1.0
            gross += w * (c[-1] - entry) / R; legs.append((w, 'eod'))
            return int(m[-1]), ('pp+eod' if part else 'eod'), gross, legs
        return int(m[fi]), 'eod', 0.0, [(1.0, 'eod')]
    for k in range(fi + 1, n):
        if m[k] >= FLAT_M:
            return int(m[k]), 'eod', (o[k] - entry) / R, [(1.0, 'eod')]
        if l[k] <= stop:
            px = min(stop, o[k]) * STOP_SLIP
            return int(m[k]), 'stop', (px - entry) / R, [(1.0, 'stop')]
        if mode == 'r2' and c[k] >= tgt:
            return int(m[k]), 'target', (tgt - entry) / R, [(1.0, 'target')]
    if n > fi + 1:
        return int(m[-1]), 'eod', (c[-1] - entry) / R, [(1.0, 'eod')]
    return int(m[fi]), 'eod', 0.0, [(1.0, 'eod')]


def run_day(bars_api, sym, day, prev, cfg, day_open=None, cap=0.006, pdr_min=8.0,
            floor_pct=5.0, min_r_pct=1.0, min_price=5.0):
    """prev = dict(close, high, low) already resolved for cfg['prev'].
    Returns dict(ok=bool, reason=str, ...fields). reason is the FIRST gate that stopped it."""
    out = dict(sym=sym, day=day, ok=False, reason='', src='')
    pc, ph, pl = prev.get('close'), prev.get('high'), prev.get('low')
    if pc is None or pc != pc or pc <= 0:
        out['reason'] = 'no_prev'; return out
    if pl is None or pl != pl or pl <= 0 or ph is None or ph != ph:
        out['reason'] = 'no_prev'; return out
    pdr = (ph - pl) / pl * 100.0
    out['pdr'] = pdr
    if pdr < pdr_min:
        out['reason'] = 'pdr'; return out
    if cfg['day_open5'] and day_open is not None and day_open < 5.0:
        out['reason'] = 'day_open5'; return out
    got = bars_api.get(sym, day, cfg['bars'])
    if got is None:
        out['reason'] = 'no_bars'; return out
    m, o, h, l, c, v, src = got
    out['src'] = src
    if len(m) < max(cfg['min_bars'], 2):
        out['reason'] = 'few_bars'; return out
    o0 = o[0]
    out['o0'] = o0; out['first_m'] = int(m[0])
    if not (o0 < pc):
        out['reason'] = 'not_red'; return out
    level = pc * cfg['level_mult']
    out['level'] = level
    run_hi = np.maximum.accumulate(h); run_lo = np.minimum.accumulate(l)

    def floor_ok(i):
        if i < 1:
            return False
        den = o0 if cfg['floor_den'] == 'o0' else run_lo[i - 1]
        return den > 0 and (run_hi[i - 1] - run_lo[i - 1]) / den * 100.0 >= floor_pct

    si = None
    if cfg['scan'] == 'break_then_floor':
        for i in range(1, len(m)):
            if h[i] >= level:
                si = i; break
        if si is None:
            out['reason'] = 'no_signal'; return out
        if not floor_ok(si):
            out['reason'] = 'floor'; return out
    else:                                              # engine: first bar where the floor holds AND the level breaks
        for i in range(1, len(m)):
            if cfg['late_on'] == 'signal' and m[i] > cfg['late_m']:
                out['reason'] = 'late'; return out
            if not floor_ok(i):
                continue
            if h[i] < level:
                continue
            st = run_lo[i] if cfg['stop'] == 'incl' else run_lo[i - 1]
            if st >= level:
                continue
            si = i; break
        if si is None:
            out['reason'] = 'no_signal'; return out
    out['sig_m'] = int(m[si])
    if cfg['late_on'] == 'signal' and m[si] > cfg['late_m']:
        out['reason'] = 'late'; return out
    stop = run_lo[si] if cfg['stop'] == 'incl' else run_lo[si - 1]
    out['stop'] = float(stop)
    if stop >= level:
        out['reason'] = 'stop_above'; return out
    if si + 1 >= len(m):
        out['reason'] = 'no_next'; return out
    fi = si + 1
    entry = float(o[fi])
    out['entry'] = entry; out['entry_m'] = int(m[fi])
    if cfg['late_on'] == 'fill' and m[fi] > cfg['late_m']:
        out['reason'] = 'late'; return out
    if entry > level * (1.0 + cap):
        out['reason'] = 'cap'; return out
    px_gate = level if cfg['price_on'] == 'level' else entry
    if px_gate < min_price:
        out['reason'] = 'price'; return out
    R = entry - stop
    if R <= 0 or R < min_r_pct / 100.0 * entry:
        out['reason'] = 'r_small'; return out
    out['R'] = R; out['r_pct'] = R / entry * 100.0
    for mode in ('hold', 'r2', 'partial'):
        em, et, gr, legs = walk(o, h, l, c, m, fi, entry, stop, mode)
        out[f'{mode}_exit_m'] = em; out[f'{mode}_exit_type'] = et; out[f'{mode}_grossR'] = gr
        out[f'{mode}_legs'] = ';'.join('%g:%s' % (w, t) for w, t in legs)
    out['ok'] = True; out['reason'] = 'taken'
    return out
