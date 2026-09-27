"""cell_1591.py -- v2 SCORE+FETCH stage for PREREG_1567 (Amendment 2 / 2a): cells 1,591-1,598.

Amendment 2a replaced Amendment 2's OPRA-quote plan (404 on this data plan) with TICK TRADES:
every leg's entry/exit fill is the last trade in a real 5-minute market window, never a print
plucked without a VOID check and never a daily-bar approximation of an intraday fill. This module
reuses cell_1567's Black-Scholes/IV math and its Cache (grid, spy bars, option daily bars,
option 10:00 entry-minute bars) UNCHANGED -- those are the same spec, the same cache, the same
"ONE helper" the project's rules require -- and only replaces what Amendment 2/2a actually changed:
strike SELECTION still uses the cached entry-minute price (a delta estimate, not a fill), but every
dollar in every P&L number comes from a tick trade fetched here, cached under
opt_cache/ticks/ticks_state.db (resumable, symbol+date keyed, LOST/VOID counted).

Cells (Delta outer, then Management, then Gate; W=$10 fixed):
  1591 D=0.20 M=A G=0   1592 D=0.20 M=A G=1   1593 D=0.20 M=B G=0   1594 D=0.20 M=B G=1
  1595 D=0.30 M=A G=0   1596 D=0.30 M=A G=1   1597 D=0.30 M=B G=0   1598 D=0.30 M=B G=1

Fill rule (Amendment 2a): entry = last trade of each leg in 10:00:00-10:00:30 ET on the entry
Monday, fallback to the last trade in the wider 10:00:00-10:05:00 ET window (WARNING-logged) if the
narrow window is empty; NO trade anywhere in the 5-minute window -> the CYCLE is VOID (counted
against the 10% VOID rail, never imputed). Priced short leg = trade - $0.03 (sold), long leg =
trade + $0.03 (bought); sensitivity rails at $0.05 and $0.10/leg computed alongside, never
substituted for the headline $0.03 line.

Management A defers EVERY active exit (50% credit, stop @ 2x credit, 21-DTE) to the NEXT SESSION's
10:00-10:00:30 tick trade once a daily-close mark (option_daily) trips the rule that day -- you
cannot execute at an as-of daily close, only at a real quote the next time the market opens, and
Amendment 2a's tick window is a morning window, not a close. If a leg has no exit-session tick, the
fallback is that session's daily OPEN (Amendment 2a, exits only) -- WARNING-logged, counted, NOT
void (this differs from the entry rule on purpose: the amendment says so explicitly). Management B
never touches ticks: it holds to expiry and settles on the intrinsic value at the SPY 16:00 price,
exactly as cell_1567's Management B does.
"""
import argparse
import datetime as dt
import logging
import math
import os
import sqlite3
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, 'opt_cache')
TICK_DIR = os.path.join(CACHE_DIR, 'ticks')
TICK_DB = os.path.join(TICK_DIR, 'ticks_state.db')
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from cell_1567 import (  # noqa: E402  -- ONE spec: reuse the frozen BS/IV math and the Cache
    Cache, bs_put_delta, bs_put_price, implied_vol_put, precompute_monday, select_strikes,
    size_position, R_RATE, Q_RATE, DTE_HARD_EXIT, PROFIT_TARGET_FRAC, STOP_MULT,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('cell_1591')

ET = ZoneInfo('America/New_York')
UTC = dt.timezone.utc

# --------------------------------------------------------------------------- constants (PREREG v2)
EQUITY = 65_000.0
B_FRAC = 0.10
B = EQUITY * B_FRAC                 # $6,500 risk budget, fixed by construction
N_LADDER = 6
WIDTH = 10.0
DELTAS = [0.20, 0.30]
MANAGEMENTS = ['A', 'B']
GATES = [0, 1]
IV_GATE_MIN = 0.15
ENTRY_SLIP = 0.03                    # Amendment 2a headline: $0.03/leg
SLIP_RAILS = [0.05, 0.10]            # Amendment 2a sensitivity rails
VOID_RAIL_PCT = 0.10                 # > 10% VOID cycles -> the cell is VOID (Amendment 2)

TRAIN_START, TRAIN_END = '2024-02-05', '2025-06-30'
VAL_START, VAL_END = '2025-07-07', '2026-08-17'

CELLS = [
    {'cell': 1591 + i, 'delta': d, 'mgmt': m, 'gate': g}
    for i, (d, m, g) in enumerate((d, m, g) for d in DELTAS for m in MANAGEMENTS for g in GATES)
]


# --------------------------------------------------------------------------- tick cache (resumable)

def init_tick_db(con):
    con.executescript('''
        CREATE TABLE IF NOT EXISTS ticks (
            symbol TEXT, date TEXT, status TEXT,
            price_30s REAL, time_30s TEXT,
            price_5m REAL, time_5m TEXT,
            n_trades INTEGER, fetched_at TEXT,
            PRIMARY KEY(symbol, date)
        );
    ''')
    con.commit()


def _window_utc(day):
    """(start_utc, end_utc) for the 10:00:00-10:05:00 ET wall-clock window on `day` (DST-correct)."""
    d = dt.date.fromisoformat(day)
    s = dt.datetime(d.year, d.month, d.day, 10, 0, 0, tzinfo=ET)
    e = dt.datetime(d.year, d.month, d.day, 10, 5, 0, tzinfo=ET)
    return s.astimezone(UTC), e.astimezone(UTC)


def fetch_tick(con, client, symbol, date, warn_counter, pause_s=0.35, max_retries=3):
    """Fetches (if not already cached) trades for `symbol` on `date` in the 10:00:00-10:05:00 ET
    window and caches the priced result. Returns True iff a live API call was made (for throttling
    by the caller), False if served entirely from cache."""
    cur = con.execute('SELECT 1 FROM ticks WHERE symbol=? AND date=?', (symbol, date))
    if cur.fetchone() is not None:
        return False
    from alpaca.data.requests import OptionTradesRequest
    s_utc, e_utc = _window_utc(date)
    trades = None
    for attempt in range(1, max_retries + 1):
        try:
            req = OptionTradesRequest(symbol_or_symbols=[symbol], start=s_utc, end=e_utc)
            res = client.get_option_trades(req)
            trades = res.data.get(symbol, [])
            break
        except Exception as e:
            wait = 1.5 * attempt
            log.warning('tick fetch %s %s attempt %d/%d failed: %s (retry %.1fs)',
                        symbol, date, attempt, max_retries, e, wait)
            time.sleep(wait)
    now = dt.datetime.now(UTC).isoformat()
    if trades is None:
        log.error('tick fetch %s %s FAILED after %d retries -- cached as error/VOID', symbol, date, max_retries)
        con.execute('INSERT OR REPLACE INTO ticks VALUES (?,?,?,?,?,?,?,?,?)',
                    (symbol, date, 'error', None, None, None, None, 0, now))
        con.commit()
        warn_counter['tick_fetch_error'] = warn_counter.get('tick_fetch_error', 0) + 1
        time.sleep(pause_s)
        return True
    trades = sorted(trades, key=lambda t: t.timestamp)
    price_30s = time_30s = price_5m = time_5m = None
    narrow_cutoff = s_utc + dt.timedelta(seconds=30)
    narrow = [t for t in trades if t.timestamp <= narrow_cutoff]
    if narrow:
        last = narrow[-1]
        price_30s, time_30s = float(last.price), last.timestamp.isoformat()
    if trades:
        last_any = trades[-1]
        price_5m, time_5m = float(last_any.price), last_any.timestamp.isoformat()
    if price_30s is not None:
        status = 'ok'
    elif price_5m is not None:
        status = 'fallback_5m'
        warn_counter['tick_fallback_5m_window'] = warn_counter.get('tick_fallback_5m_window', 0) + 1
        log.warning('no trade in first 30s for %s on %s; falling back to last trade in the wider '
                    '10:00-10:05 window (%.2f @ %s)', symbol, date, price_5m, time_5m)
    else:
        status = 'void'
        warn_counter['tick_void_no_trade'] = warn_counter.get('tick_void_no_trade', 0) + 1
        log.warning('NO trade at all for %s on %s in 10:00:00-10:05:00 ET -- VOID leg', symbol, date)
    con.execute('INSERT OR REPLACE INTO ticks VALUES (?,?,?,?,?,?,?,?,?)',
                (symbol, date, status, price_30s, time_30s, price_5m, time_5m, len(trades), now))
    con.commit()
    time.sleep(pause_s)
    return True


def get_tick_price(con, symbol, date):
    """Cached tick price for (symbol, date): price_30s preferred, else the price_5m fallback,
    else VOID. Returns (price_or_None, void_bool, used_fallback_bool)."""
    cur = con.execute('SELECT status, price_30s, price_5m FROM ticks WHERE symbol=? AND date=?',
                       (symbol, date))
    row = cur.fetchone()
    if row is None:
        return None, True, False
    status, p30, p5m = row
    if p30 is not None:
        return float(p30), False, False
    if p5m is not None:
        return float(p5m), False, True
    return None, True, False


# --------------------------------------------------------------------------- phase 1: what to fetch

def strike_shift_diagnostic(cache, monday, mkt, target_delta, entry):
    """Counts how often the missing-print rule shifted the short-strike pick: True if the strike
    ONE DOLLAR closer to spot than the one actually selected exists in the grid but has no
    entry-minute print at all (put delta is monotonic in strike, so that neighbour would have had
    a smaller |delta - target| gap had it been quotable)."""
    if entry is None:
        return False
    strikes = mkt['strikes']
    neighbour = strikes[strikes['strike'] == entry['short_strike'] + 1.0]
    if neighbour.empty:
        return False
    sym = neighbour['symbol'].iloc[0]
    have_row = cache.opt_minute_entry[(cache.opt_minute_entry['symbol'] == sym) &
                                       (cache.opt_minute_entry['monday'] == monday)]
    return have_row.empty  # neighbour exists in the grid, but no print at all -> selection was shifted


def build_required_entries(cache, mondays_ctx, warn_counter):
    """Phase 1: strike selection for every (monday, delta) via the cached entry-minute price
    (Amendment: selection stays print-based, only FILLS move to ticks). Returns
    {(monday, delta): entry_dict_or_None} and a set of (symbol, date) tick fetches needed for entry."""
    entry_by_md = {}
    entry_fetches = set()
    n_shift = 0
    for monday, mkt in mondays_ctx:
        if mkt is None:
            continue
        for delta_t in DELTAS:
            entry = select_strikes(cache, monday, mkt, delta_t, WIDTH, warn_counter)
            entry_by_md[(monday, delta_t)] = entry
            if entry is None:
                continue
            if strike_shift_diagnostic(cache, monday, mkt, delta_t, entry):
                n_shift += 1
            entry_fetches.add((entry['short_symbol'], monday))
            entry_fetches.add((entry['long_symbol'], monday))
    warn_counter['strike_shift_due_to_missing_print'] = n_shift
    return entry_by_md, entry_fetches


def entry_credit_from_ticks(con, entry, monday, slip):
    """(short - slip) - (long + slip) from cached tick prices; None if either leg is VOID."""
    s_px, s_void, _ = get_tick_price(con, entry['short_symbol'], monday)
    l_px, l_void, _ = get_tick_price(con, entry['long_symbol'], monday)
    if s_void or l_void:
        return None, None, None
    return (s_px - slip) - (l_px + slip), s_px, l_px


def find_trigger_day(cache, entry, monday, net_credit):
    """Walks option_daily closes forward from `monday` (mirrors cell_1567.run_cycle's management-A
    loop) and returns (trigger_day, reason) the FIRST session whose daily-close mark trips 50% credit,
    2x-credit stop, or the 21-DTE hard exit -- or (None, 'expiry') if none trips before expiry.
    Every active exit (not just the stop) is priced the NEXT session under Amendment 2a, since a
    tick fill can never be dated to an as-of daily close."""
    short_sym, long_sym, expiry = entry['short_symbol'], entry['long_symbol'], entry['expiry']
    business_days = pd.bdate_range(monday, expiry)
    if len(business_days) < 2:
        return None, 'expiry'
    for d in business_days[1:]:
        day = d.strftime('%Y-%m-%d')
        if day >= expiry:
            break
        s_close = cache.daily_asof(short_sym, day)
        l_close = cache.daily_asof(long_sym, day)
        if s_close is None or l_close is None:
            return None, 'void_daily_mark'
        mark = s_close - l_close
        if mark >= STOP_MULT * net_credit and net_credit > 0:
            return day, 'stop'
        if mark <= PROFIT_TARGET_FRAC * net_credit:
            return day, 'profit_50'
        if (pd.Timestamp(expiry) - pd.Timestamp(day)).days <= DTE_HARD_EXIT:
            return day, 'dte21'
    return None, 'expiry'


def next_session(cache, after_day, expiry):
    """First business day strictly after `after_day`, at or before `expiry`."""
    s = cache._daily_by_symbol  # any populated series' calendar works; use SPY daily instead
    days = cache.spy_daily[(cache.spy_daily['day'] > after_day) & (cache.spy_daily['day'] <= expiry)]
    if days.empty:
        return None
    return days['day'].iloc[0]


def resolve_exits(cache, con, entry_by_md, warn_counter):
    """Post entry-tick-fetch: computes each (monday, delta)'s net credit from cached entry ticks,
    walks daily marks to the trigger day, and resolves the next-session exit date. Returns
    {(monday, delta): dict} with keys short_symbol/long_symbol/expiry/trigger_day/reason/
    exit_session/net_credit_<rail> for rail in (0.03, 0.05, 0.10), or None (VOID: entry ticks
    missing or no daily marks available for the walk)."""
    resolved = {}
    exit_fetches = set()
    for (monday, delta_t), entry in entry_by_md.items():
        if entry is None:
            resolved[(monday, delta_t)] = None
            continue
        credits = {}
        void_entry = False
        for slip in [ENTRY_SLIP] + SLIP_RAILS:
            credit, s_px, l_px = entry_credit_from_ticks(con, entry, monday, slip)
            if credit is None:
                void_entry = True
                break
            credits[slip] = credit
        if void_entry:
            warn_counter['void_entry_tick_missing'] = warn_counter.get('void_entry_tick_missing', 0) + 1
            resolved[(monday, delta_t)] = None
            continue
        trigger_day, reason = find_trigger_day(cache, entry, monday, credits[ENTRY_SLIP])
        if reason == 'void_daily_mark':
            warn_counter['void_no_daily_mark_for_walk'] = warn_counter.get('void_no_daily_mark_for_walk', 0) + 1
            resolved[(monday, delta_t)] = None
            continue
        exit_session = None
        if trigger_day is not None:
            exit_session = next_session(cache, trigger_day, entry['expiry'])
            if exit_session is None:
                exit_session = entry['expiry']
            exit_fetches.add((entry['short_symbol'], exit_session))
            exit_fetches.add((entry['long_symbol'], exit_session))
        resolved[(monday, delta_t)] = {
            **entry, 'credits': credits, 'trigger_day': trigger_day, 'reason': reason,
            'exit_session': exit_session,
        }
    return resolved, exit_fetches


# --------------------------------------------------------------------------- phase 2: cycle pricing

def price_exit(cache, con, resolved, slip, warn_counter):
    """Buy-back cost for an M=A active exit at resolved['exit_session'], from tick trades with a
    daily-OPEN fallback (Amendment 2a, exits only -- NOT void). Returns (cost, used_fallback:bool)."""
    short_sym, long_sym, sess = resolved['short_symbol'], resolved['long_symbol'], resolved['exit_session']
    s_px, s_void, s_fb = get_tick_price(con, short_sym, sess)
    l_px, l_void, l_fb = get_tick_price(con, long_sym, sess)
    used_fallback = s_fb or l_fb
    if s_void:
        s_px = cache.daily_asof(short_sym, sess) or cache.daily_next_open(short_sym, sess)
        used_fallback = True
        warn_counter['exit_fallback_daily_open'] = warn_counter.get('exit_fallback_daily_open', 0) + 1
        log.warning('exit tick VOID for short %s on %s -- falling back to the daily bar (Amendment 2a)',
                    short_sym, sess)
    if l_void:
        l_px = cache.daily_asof(long_sym, sess) or cache.daily_next_open(long_sym, sess)
        used_fallback = True
        warn_counter['exit_fallback_daily_open'] = warn_counter.get('exit_fallback_daily_open', 0) + 1
        log.warning('exit tick VOID for long %s on %s -- falling back to the daily bar (Amendment 2a)',
                    long_sym, sess)
    if s_px is None or l_px is None:
        return None, used_fallback
    return (s_px + slip) - (l_px - slip), used_fallback


def run_cycle_v2(cache, con, resolved, mgmt, warn_counter):
    """Returns dict of {slip: pnl_per_share} for slip in (0.03,0.05,0.10), plus exit_date/reason,
    or None if VOID (an exit tick AND its daily-bar fallback are both unavailable)."""
    if mgmt == 'B':
        S_T = cache.spy_16_close(resolved['expiry'])
        if S_T is None:
            warn_counter['void_no_expiry_spot'] = warn_counter.get('void_no_expiry_spot', 0) + 1
            return None
        intrinsic_short = max(0.0, resolved['short_strike'] - S_T)
        intrinsic_long = max(0.0, resolved['long_strike'] - S_T)
        settlement = intrinsic_short - intrinsic_long
        pnl = {slip: resolved['credits'][slip] - settlement for slip in [ENTRY_SLIP] + SLIP_RAILS}
        return {'pnl': pnl, 'exit_date': resolved['expiry'], 'exit_reason': 'expiry', 'closed_actively': False}
    # mgmt == 'A'
    if resolved['trigger_day'] is None:
        S_T = cache.spy_16_close(resolved['expiry'])
        if S_T is None:
            warn_counter['void_no_expiry_spot'] = warn_counter.get('void_no_expiry_spot', 0) + 1
            return None
        intrinsic_short = max(0.0, resolved['short_strike'] - S_T)
        intrinsic_long = max(0.0, resolved['long_strike'] - S_T)
        settlement = intrinsic_short - intrinsic_long
        pnl = {slip: resolved['credits'][slip] - settlement for slip in [ENTRY_SLIP] + SLIP_RAILS}
        return {'pnl': pnl, 'exit_date': resolved['expiry'], 'exit_reason': 'expiry', 'closed_actively': False}
    pnl = {}
    for slip in [ENTRY_SLIP] + SLIP_RAILS:
        cost, _ = price_exit(cache, con, resolved, slip, warn_counter)
        if cost is None:
            return None
        pnl[slip] = resolved['credits'][slip] - cost
    return {'pnl': pnl, 'exit_date': resolved['exit_session'], 'exit_reason': resolved['reason'],
            'closed_actively': True}


def run_cell(cache, con, cell_def, mondays_ctx, resolved_by_md, warn_counter):
    delta_t, mgmt, gate = cell_def['delta'], cell_def['mgmt'], cell_def['gate']
    open_positions = []
    cycles = []
    n_skipped_gate = n_void = n_zero_size = 0
    for monday, mkt in mondays_ctx:
        open_positions = [p for p in open_positions if p[0] > monday]
        if mkt is None:
            n_void += 1
            continue
        if gate == 1 and mkt['iv_atm'] < IV_GATE_MIN:
            n_skipped_gate += 1
            continue
        resolved = resolved_by_md.get((monday, delta_t))
        if resolved is None:
            n_void += 1
            continue
        net_credit = resolved['credits'][ENTRY_SLIP]
        reserved = sum(p[1] for p in open_positions)
        remaining = max(B - reserved, 0.0)
        alloc = min(B / N_LADDER, remaining)
        contracts, worst_per_contract = size_position(alloc, WIDTH, net_credit)
        if contracts <= 0:
            n_zero_size += 1
            continue
        worst_case = worst_per_contract * contracts
        assert reserved + worst_case <= B + 1e-6, (
            f'BUDGET ASSERTION VIOLATED: reserved={reserved} + worst_case={worst_case} > B={B}')
        result = run_cycle_v2(cache, con, resolved, mgmt, warn_counter)
        if result is None:
            n_void += 1
            continue
        open_positions.append((result['exit_date'], worst_case))
        fee_leg = 0.03
        fee_entry = fee_leg * 2 * contracts
        fee_exit = fee_leg * 2 * contracts if result['closed_actively'] else 0.0
        holding_days = (pd.Timestamp(result['exit_date']) - pd.Timestamp(monday)).days
        naked_pnl_per_share = naked_short_pnl(cache, con, resolved, monday, resolved, mgmt, warn_counter)
        naked_pnl_usd = (naked_pnl_per_share * 100.0 * contracts - fee_leg * contracts *
                          (2 if result['closed_actively'] else 1)) if naked_pnl_per_share is not None else None
        row = {
            'cell': cell_def['cell'], 'split': None, 'entry_date': monday, 'expiry': resolved['expiry'],
            'short_strike': resolved['short_strike'], 'long_strike': resolved['long_strike'],
            'contracts': contracts, 'credit': net_credit, 'exit_date': result['exit_date'],
            'exit_reason': result['exit_reason'], 'void_reason': '',
            'pnl_usd': result['pnl'][ENTRY_SLIP] * 100.0 * contracts - fee_entry - fee_exit,
            'pnl_usd_slip005': result['pnl'][0.05] * 100.0 * contracts - fee_entry - fee_exit,
            'pnl_usd_slip010': result['pnl'][0.10] * 100.0 * contracts - fee_entry - fee_exit,
            'worst_case_usd': worst_case, 'holding_days': holding_days,
            'naked_pnl_usd': naked_pnl_usd,
        }
        cycles.append(row)
    return cycles, {'n_skipped_gate': n_skipped_gate, 'n_void': n_void, 'n_zero_size': n_zero_size}


# --------------------------------------------------------------------------- reporting

def monthly_series(cycles_df, col='pnl_usd', start=None, end=None):
    """Calendar-month P&L series over [start,end] with $0 for months that had no exit at all
    (Amendment 2, point 3)."""
    if start is None or end is None:
        if cycles_df.empty:
            return pd.Series(dtype=float)
        start, end = cycles_df['exit_date'].min(), cycles_df['exit_date'].max()
    months = pd.period_range(pd.Period(start, 'M'), pd.Period(end, 'M'), freq='M')
    if cycles_df.empty:
        return pd.Series(0.0, index=months)
    s = cycles_df.copy()
    s['month'] = pd.to_datetime(s['exit_date']).dt.to_period('M')
    grp = s.groupby('month')[col].sum()
    return grp.reindex(months, fill_value=0.0).sort_index()


def cell_stats(cycles_df, split_name, start, end):
    monthly = monthly_series(cycles_df, 'pnl_usd', start, end)
    n_cycles = int(len(cycles_df))
    void_cycles = int((cycles_df['void_reason'] != '').sum()) if not cycles_df.empty else 0
    if n_cycles == 0:
        return {'split': split_name, 'n_cycles': 0, 'void_cycles': void_cycles, 'void_share': float('nan'),
                'skipped_weeks': 0, 'mean_monthly_ret_on_B': float('nan'), 'monthly_sharpe': float('nan'),
                'green_months': 0, 'n_months': int(len(monthly)), 'green_month_share': float('nan'),
                'worst_month_usd': 0.0, 'worst_month_ge_negB': True,
                'max_dd_usd': 0.0, 'max_dd_le_1p5B': True, 'win_rate': float('nan'),
                'top5pct_cycles_share': float('nan'), 'ex_top5pct_pnl_usd': 0.0,
                'exit_mix': {}, 'aug2024_pnl_usd': 0.0, 'apr2025_pnl_usd': 0.0,
                'ret_at_slip_005': float('nan'), 'ret_at_slip_010': float('nan'),
                'naked_comparison_pnl_usd': float('nan'), 'mean_cycle_pnl_usd': float('nan'),
                'passes_bar': False}
    monthly_ret = monthly / B
    mean_monthly_ret = float(monthly_ret.mean())
    sharpe = (float(monthly_ret.mean() / monthly_ret.std(ddof=1) * math.sqrt(12))
              if len(monthly_ret) > 1 and monthly_ret.std(ddof=1) > 0 else float('nan'))
    green_share = float((monthly > 0).mean())
    worst_month = float(monthly.min())
    cum = monthly.cumsum()
    dd = (cum - cum.cummax()).min()
    max_dd = float(-dd) if pd.notna(dd) else 0.0
    pnl_sorted = cycles_df['pnl_usd'].sort_values(ascending=False)
    top5_n = max(1, int(math.ceil(0.05 * len(pnl_sorted))))
    top5_share = float(pnl_sorted.iloc[:top5_n].sum() / pnl_sorted.sum()) if pnl_sorted.sum() != 0 else float('nan')
    ex_top5_pnl = float(pnl_sorted.iloc[top5_n:].sum())
    win_rate = float((cycles_df['pnl_usd'] > 0).mean())
    exit_mix = cycles_df['exit_reason'].value_counts().to_dict()
    aug24 = float(monthly_series(cycles_df, 'pnl_usd', start, end).get(pd.Period('2024-08'), 0.0))
    apr25 = float(monthly_series(cycles_df, 'pnl_usd', start, end).get(pd.Period('2025-04'), 0.0))
    ret005 = float(monthly_series(cycles_df, 'pnl_usd_slip005', start, end).mean() / B)
    ret010 = float(monthly_series(cycles_df, 'pnl_usd_slip010', start, end).mean() / B)
    void_share = void_cycles / n_cycles if n_cycles else 0.0
    naked_valid = cycles_df['naked_pnl_usd'].dropna()
    naked_sum = float(naked_valid.sum()) if not naked_valid.empty else float('nan')
    return {
        'split': split_name, 'n_cycles': n_cycles, 'void_cycles': void_cycles, 'void_share': void_share,
        'mean_monthly_ret_on_B': mean_monthly_ret, 'monthly_sharpe': sharpe,
        'green_months': int((monthly > 0).sum()), 'n_months': int(len(monthly)),
        'green_month_share': green_share, 'worst_month_usd': worst_month,
        'worst_month_ge_negB': bool(worst_month >= -B - 1e-6), 'max_dd_usd': max_dd,
        'max_dd_le_1p5B': bool(max_dd <= 1.5 * B),
        'top5pct_cycles_share': top5_share, 'ex_top5pct_pnl_usd': ex_top5_pnl,
        'win_rate': win_rate, 'exit_mix': exit_mix,
        'aug2024_pnl_usd': aug24, 'apr2025_pnl_usd': apr25,
        'ret_at_slip_005': ret005, 'ret_at_slip_010': ret010,
        'mean_cycle_pnl_usd': float(cycles_df['pnl_usd'].mean()),
        'naked_comparison_pnl_usd': naked_sum,
    }


def spy_buy_hold(spy_daily, start, end):
    sub = spy_daily[(spy_daily['day'] >= start) & (spy_daily['day'] <= end)].sort_values('day')
    if len(sub) < 2:
        return {'return_pct': float('nan'), 'pnl_on_B_usd': float('nan'), 'dd_pct': float('nan')}
    ret = sub['c'].iloc[-1] / sub['c'].iloc[0] - 1.0
    cum = sub['c'] / sub['c'].iloc[0]
    dd = float((cum - cum.cummax()).min())
    return {'return_pct': float(ret), 'pnl_on_B_usd': float(ret * B), 'dd_pct': dd}


def naked_short_pnl(cache, con, entry, monday, resolved, mgmt, warn_counter):
    """Report-only undefined-risk comparison: short leg alone, same entry Monday and same
    resolved exit session/reason as the spread, priced from the SAME cached ticks (no new fetch)."""
    short_sym = entry['short_symbol']
    s_entry_px, void, _ = get_tick_price(con, short_sym, monday)
    if void:
        return None
    short_credit = s_entry_px - ENTRY_SLIP
    if mgmt == 'B' or resolved['trigger_day'] is None:
        S_T = cache.spy_16_close(resolved['expiry'])
        if S_T is None:
            return None
        return short_credit - max(0.0, entry['short_strike'] - S_T)
    s_exit_px, s_void, _ = get_tick_price(con, short_sym, resolved['exit_session'])
    if s_void:
        s_exit_px = cache.daily_asof(short_sym, resolved['exit_session'])
        if s_exit_px is None:
            return None
    return short_credit - (s_exit_px + ENTRY_SLIP)


# --------------------------------------------------------------------------- driver

def get_client():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, '.env'))
    from config import Config
    cfg = Config()
    if not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
        log.error('missing Alpaca API credentials (ALPACA_API_KEY/ALPACA_API_SECRET) -- cannot fetch ticks')
        return None
    from alpaca.data.historical.option import OptionHistoricalDataClient
    return OptionHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke-test', action='store_true', help='first 8 entry Mondays only')
    ap.add_argument('--no-fetch', action='store_true', help='score only from whatever ticks are cached')
    ap.add_argument('--out-dir', default=HERE)
    args = ap.parse_args()

    os.makedirs(TICK_DIR, exist_ok=True)
    cache = Cache()
    mondays = sorted(cache.grid['monday'].unique())
    if args.smoke_test:
        mondays = mondays[:8]
    warn_counter = {}
    mkt_by_monday = {}
    for m in mondays:
        mkt_by_monday[m] = precompute_monday(cache, m, warn_counter)
    mondays_ctx = [(m, mkt_by_monday[m]) for m in mondays]
    n_void_mondays = sum(1 for _, v in mondays_ctx if v is None)
    log.info('Mondays with no usable expiry: %d/%d', n_void_mondays, len(mondays))

    entry_by_md, entry_fetches = build_required_entries(cache, mondays_ctx, warn_counter)
    log.info('Entry ticks needed: %d (symbol, date) pairs across %d (monday,delta) selections',
             len(entry_fetches), sum(1 for v in entry_by_md.values() if v is not None))

    con = sqlite3.connect(TICK_DB)
    init_tick_db(con)

    client = None if args.no_fetch else get_client()
    if client is not None:
        for i, (sym, date) in enumerate(sorted(entry_fetches)):
            fetch_tick(con, client, sym, date, warn_counter)
            if (i + 1) % 50 == 0:
                log.info('  entry ticks: %d/%d', i + 1, len(entry_fetches))

    resolved_by_md, exit_fetches = resolve_exits(cache, con, entry_by_md, warn_counter)
    log.info('Exit ticks needed (Management A only): %d (symbol, date) pairs', len(exit_fetches))
    if client is not None:
        for i, (sym, date) in enumerate(sorted(exit_fetches)):
            fetch_tick(con, client, sym, date, warn_counter)
            if (i + 1) % 50 == 0:
                log.info('  exit ticks: %d/%d', i + 1, len(exit_fetches))

    all_cycles = []
    all_rows = []
    for cell_def in CELLS:
        cycles, counts = run_cell(cache, con, cell_def, mondays_ctx, resolved_by_md, warn_counter)
        for c in cycles:
            c2 = dict(c)
            c2['delta'] = cell_def['delta']; c2['mgmt'] = cell_def['mgmt']; c2['gate'] = cell_def['gate']
            all_cycles.append(c2)
        df = pd.DataFrame(cycles)
        for split_name, (s, e) in [('TRAIN', (TRAIN_START, TRAIN_END)), ('VAL', (VAL_START, VAL_END))]:
            split_df = df[(df['entry_date'] >= s) & (df['entry_date'] <= e)] if not df.empty else df
            stats = cell_stats(split_df, split_name, s, e)
            stats.update({'cell': cell_def['cell'], 'delta': cell_def['delta'], 'mgmt': cell_def['mgmt'],
                          'gate': cell_def['gate'], **counts, 'spy_bh': spy_buy_hold(cache.spy_daily, s, e)})
            all_rows.append(stats)
        log.info('Cell %d (D=%.2f M=%s G=%d): %d cycles, %d void, %d gate-skip, %d zero-size',
                  cell_def['cell'], cell_def['delta'], cell_def['mgmt'], cell_def['gate'],
                  len(cycles), counts['n_void'], counts['n_skipped_gate'], counts['n_zero_size'])

    for k, v in warn_counter.items():
        log.warning('%s: %d', k, v)

    cycles_df = pd.DataFrame(all_cycles)
    cycles_df.to_csv(os.path.join(args.out_dir, 'cell_1591_cycles.csv'), index=False)
    log.info('Wrote %d cycle rows', len(cycles_df))

    monthly_rows = []
    for cell_def in CELLS:
        cdf = cycles_df[cycles_df['cell'] == cell_def['cell']] if not cycles_df.empty else cycles_df
        for split_name, (s, e) in [('TRAIN', (TRAIN_START, TRAIN_END)), ('VAL', (VAL_START, VAL_END))]:
            sdf = cdf[(cdf['entry_date'] >= s) & (cdf['entry_date'] <= e)] if not cdf.empty else cdf
            ms = monthly_series(sdf, 'pnl_usd', s, e)
            for month, pnl in ms.items():
                monthly_rows.append({'cell': cell_def['cell'], 'split': split_name, 'month': str(month),
                                      'pnl_usd': pnl, 'return_on_B': pnl / B})
    pd.DataFrame(monthly_rows).to_csv(os.path.join(args.out_dir, 'cell_1591_monthly.csv'), index=False)

    rows_df = pd.DataFrame(all_rows)
    write_result_md(rows_df, cache, args.out_dir, warn_counter, n_void_mondays, len(mondays))
    con.close()
    log.info('Done.')


def write_result_md(rows_df, cache, out_dir, warn_counter, n_void_mondays, n_mondays):
    train = rows_df[rows_df['split'] == 'TRAIN'].copy()
    eligible = train[(train['n_cycles'] >= 12) & (train['green_month_share'] >= 0.55)]
    lines = ['# RESULT_1591 -- v2 quote-era put-credit-spread ladder (Amendment 2/2a, tick-priced)\n']
    lines.append(f'Equity fixed at ${EQUITY:,.0f}; B = {B_FRAC:.0%} = ${B:,.2f}. {n_mondays} entry Mondays, '
                 f'{n_void_mondays} with no usable listed expiry.\n')
    lines.append('## TRAIN selection (highest monthly Sharpe, n_cycles>=12, green_months>=55%)\n')
    selected_cell = None
    if eligible.empty:
        lines.append('**No cell cleared the TRAIN eligibility bar.**\n')
    else:
        best = eligible.sort_values('monthly_sharpe', ascending=False).iloc[0]
        selected_cell = int(best['cell'])
        lines.append(f"Selected cell **{selected_cell}** (Delta={best['delta']}, M={best['mgmt']}, "
                      f"G={best['gate']}): TRAIN monthly Sharpe {best['monthly_sharpe']:.2f}, "
                      f"n_cycles {int(best['n_cycles'])}, green months {best['green_month_share']:.0%}, "
                      f"VOID share {best.get('void_share', float('nan')):.1%}.\n")
    cols = ['cell', 'delta', 'mgmt', 'gate', 'n_cycles', 'void_cycles', 'mean_monthly_ret_on_B',
            'monthly_sharpe', 'green_month_share', 'worst_month_usd', 'max_dd_usd', 'win_rate']
    lines.append('\n## Full TRAIN table (8 cells)\n')
    lines.append(train[cols].sort_values('monthly_sharpe', ascending=False).to_markdown(index=False))
    if selected_cell is not None:
        val = rows_df[(rows_df['split'] == 'VAL') & (rows_df['cell'] == selected_cell)]
        if not val.empty:
            v = val.iloc[0]
            lines.append('\n\n## VAL read of the selected cell\n')
            lines.append(f"n_cycles={int(v['n_cycles'])}, VOID share={v.get('void_share', float('nan')):.1%}, "
                          f"mean monthly return on B={v['mean_monthly_ret_on_B']:.2%}, "
                          f"monthly Sharpe={v['monthly_sharpe']:.2f}, green months={v['green_month_share']:.0%}, "
                          f"ex-top5% cycle PnL=${v['ex_top5pct_pnl_usd']:,.0f}, worst month=${v['worst_month_usd']:,.0f} "
                          f"(>=-B: {v['worst_month_ge_negB']}), max DD=${v['max_dd_usd']:,.0f} "
                          f"(<=1.5B: {v['max_dd_le_1p5B']}), slip $0.05/leg monthly ret={v['ret_at_slip_005']:.2%}, "
                          f"slip $0.10/leg monthly ret={v['ret_at_slip_010']:.2%}.\n")
            spy_bh = v['spy_bh']
            lines.append(f"SPY buy-and-hold over the same VAL window: return {spy_bh['return_pct']:.2%}, "
                          f"${spy_bh['pnl_on_B_usd']:,.0f} on B, max DD {spy_bh['dd_pct']:.2%}.\n")
            pass_checklist = {
                'mean monthly return >= 4%': v['mean_monthly_ret_on_B'] >= 0.04,
                'monthly Sharpe >= 1.0': v['monthly_sharpe'] >= 1.0,
                'green months >= 60%': v['green_month_share'] >= 0.60,
                'ex-top5% cycles positive': v['ex_top5pct_pnl_usd'] > 0,
                'worst month >= -B': v['worst_month_ge_negB'],
                'max DD <= 1.5B': v['max_dd_le_1p5B'],
                'VOID share <= 10%': v.get('void_share', 1.0) <= VOID_RAIL_PCT,
            }
            lines.append('\nPass-bar checklist: ' + ', '.join(
                f'{k}={"PASS" if ok else "FAIL"}' for k, ok in pass_checklist.items()) + '\n')
            lines.append(f"\n**Overall: {'PASS' if all(pass_checklist.values()) else 'FAIL'}**\n")
    lines.append('\n\n## Full VAL table (8 cells)\n')
    val_all = rows_df[rows_df['split'] == 'VAL']
    lines.append(val_all[cols].sort_values('monthly_sharpe', ascending=False).to_markdown(index=False))
    lines.append('\n\n## Amendment 2a caveats carried into this result\n')
    lines.append('* Strike SELECTION still uses the cached 10:00 entry-minute trade price (a delta '
                 'estimate), never a tick -- only the fill and every P&L dollar are tick-priced.\n')
    lines.append('* Entry: a leg with no trade in 10:00:00-10:05:00 ET -> the CYCLE is VOID (counted). '
                 'Exit (Management A only): a leg with no trade in that window on the exit session falls '
                 'back to the session\'s daily OPEN (counted, NOT void) -- Amendment 2a states this '
                 'asymmetry explicitly.\n')
    lines.append(f'* WARNING/fallback/VOID counters: {warn_counter}\n')
    with open(os.path.join(out_dir, 'RESULT_1591.md'), 'w') as f:
        f.write('\n'.join(lines))
    log.info('Wrote RESULT_1591.md')


if __name__ == '__main__':
    sys.exit(main())
