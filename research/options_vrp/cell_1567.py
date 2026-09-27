"""cell_1567.py — SCORE stage for PREREG_1567: defined-risk put-credit-spread ladder on SPY.

Builds all 24 cells (Delta in {0.15,0.20,0.30}) x (Width in {$5,$10}) x (Management in {A,B})
x (IV Gate in {off,on}) from the cache under research/options_vrp/opt_cache/ (see FETCH_1567.md).

Documented approximations (see RESULT_1567.md caveats, carried from FETCH_1567.md's "say-so"
scope reduction):
  * "mid" price for strike/IV selection and for entry/exit FILLS at the entry Monday is the mean
    trade close of the option_minute_entry bars inside the 10:00-10:05 ET sub-window (falling back
    to the nearest available bar in the fetched 09:55-10:10 ET window if 10:00-10:05 itself has no
    bar) -- Alpaca option bars are trade OHLCV, not quotes, so there is no true bid/ask mid in the
    cache; this trade-price proxy is the best available and is WARNING-logged whenever the fallback
    fires.
  * The daily MANAGEMENT mark (50%-credit / 2x-stop / 21-DTE checks on every session after entry)
    uses the option's daily CLOSE (as-of, forward-filled across bar-sparse gaps -- median 8 daily
    bars per contract lifetime) because minute bars were fetched ONLY for the entry Monday, per
    FETCH_1567.md's disclosed scope reduction. The STOP's exit fill approximates the PREREG's "next
    session's 10:00" with that next session's daily OPEN (closer to a 10:00 fill than a close).
A cycle whose short or long leg lacks the bars this stage needs (entry mid, or every daily mark
between entry and exit) is VOID and counted, never imputed.
"""
import argparse
import datetime as dt
import logging
import math
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, 'opt_cache')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('cell_1567')

# --------------------------------------------------------------------------- constants (PREREG)
EQUITY = 65_000.0                 # fixed for the whole backtest (PREREG: "state it")
B_FRAC = 0.10
B = EQUITY * B_FRAC                # $6,500 risk budget
N_LADDER = 6                        # six weekly rungs -> B/6 per rung
R_RATE = 0.045                      # constant 4.5% (PREREG: 3-mo T-bill, constant if no series)
Q_RATE = 0.013                      # 1.3%
LEG_SLIPPAGE = 0.03                 # $/share per leg, always against the position
FEE_PER_CONTRACT_LEG = 0.03         # regulatory fee, $/contract/leg/transaction
DELTAS = [0.15, 0.20, 0.30]
WIDTHS = [5.0, 10.0]
MANAGEMENTS = ['A', 'B']
GATES = [0, 1]
IV_GATE_MIN = 0.15
DTE_LO, DTE_HI, DTE_TARGET = 38, 52, 45
PROFIT_TARGET_FRAC = 0.50
STOP_MULT = 2.0
DTE_HARD_EXIT = 21

TRAIN_START, TRAIN_END = '2024-02-05', '2025-06-30'
VAL_START, VAL_END = '2025-07-07', '2026-08-17'

CELLS = [
    {'cell': 1567 + i, 'delta': d, 'width': w, 'mgmt': m, 'gate': g}
    for i, (d, w, m, g) in enumerate(
        (d, w, m, g) for d in DELTAS for w in WIDTHS for m in MANAGEMENTS for g in GATES
    )
]


# --------------------------------------------------------------------------- Black-Scholes (put)

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put_price(S, K, T, r, q, sigma):
    """European put, continuous dividend yield q. T in years, sigma annualised vol."""
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def bs_put_delta(S, K, T, r, q, sigma):
    """Signed put delta (negative). T in years."""
    if T <= 0:
        return -1.0 if S < K else 0.0
    if sigma <= 0:
        sigma = 1e-6
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return -math.exp(-q * T) * _norm_cdf(-d1)


def implied_vol_put(price, S, K, T, r, q, lo=1e-4, hi=4.0, tol=1e-4, max_iter=80):
    """Bisection IV solve for a European put. Returns None if price is outside no-arbitrage
    bounds (bad/stale trade print) or T<=0."""
    if T <= 0 or price is None or not np.isfinite(price):
        return None
    lower_bound = max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    upper_bound = K * math.exp(-r * T)
    eps = 1e-6
    if price < lower_bound - eps or price > upper_bound + eps:
        return None  # bad print -- caller logs a WARNING and excludes the strike
    price = min(max(price, lower_bound), upper_bound)
    f_lo = bs_put_price(S, K, T, r, q, lo) - price
    f_hi = bs_put_price(S, K, T, r, q, hi) - price
    if f_lo > 0:
        return lo
    if f_hi < 0:
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_put_price(S, K, T, r, q, mid) - price
        if abs(f_mid) < tol:
            return mid
        if f_mid > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- data loading

class Cache:
    """Holds every table SCORE needs, loaded once."""

    def __init__(self, cache_dir=CACHE_DIR):
        log.info('Loading cache from %s', cache_dir)
        self.spy_daily = pd.read_parquet(os.path.join(cache_dir, 'spy_daily.parquet'))
        self.spy_minute = pd.read_parquet(os.path.join(cache_dir, 'spy_minute.parquet'))
        self.opt_daily = pd.read_parquet(os.path.join(cache_dir, 'option_daily.parquet'))
        self.opt_minute_entry = pd.read_parquet(os.path.join(cache_dir, 'option_minute_entry.parquet'))
        con = sqlite3.connect(os.path.join(cache_dir, 'state.db'))
        self.grid_ref = pd.read_sql_query('SELECT symbol, monday FROM grid_ref', con)
        self.contracts = pd.read_sql_query('SELECT symbol, expiry, strike FROM contracts', con)
        con.close()
        self.grid = self.grid_ref.merge(self.contracts, on='symbol', how='left')
        self.spy_minute['t'] = pd.to_datetime(self.spy_minute['t'], utc=True)
        self.opt_minute_entry['t'] = pd.to_datetime(self.opt_minute_entry['t'], utc=True)
        self.opt_minute_entry['t_et'] = self.opt_minute_entry['t'].dt.tz_convert('America/New_York')
        # per-symbol sorted daily series for as-of lookups
        self._daily_by_symbol = {
            sym: g.sort_values('day').reset_index(drop=True)
            for sym, g in self.opt_daily.groupby('symbol')
        }
        # (symbol, monday) -> entry-minute rows, indexed once (entry_mid is called ~1e5 times;
        # a fresh boolean-mask scan of all 27,721 rows per call does not scale).
        self._entry_by_symbol_monday = {
            key: g for key, g in self.opt_minute_entry.groupby(['symbol', 'monday'])
        }
        log.info('Cache loaded: %d grid rows, %d option-daily rows, %d entry-minute rows',
                  len(self.grid), len(self.opt_daily), len(self.opt_minute_entry))

    def spot_at_10(self, monday):
        """SPY price nearest the 10:00 ET minute bar on `monday`."""
        day_bars = self.spy_minute[self.spy_minute['day'] == monday]
        if day_bars.empty:
            return None
        target = pd.Timestamp(monday + ' 15:00:00', tz='UTC')  # 10:00 ET ~ 15:00 UTC (DST-approx)
        # exact: convert each bar's ET wall time and pick nearest to 10:00
        et = day_bars['t'].dt.tz_convert('America/New_York')
        target_minutes = 10 * 60
        mins = et.dt.hour * 60 + et.dt.minute
        idx = (mins - target_minutes).abs().idxmin()
        return float(day_bars.loc[idx, 'c'])

    def spy_16_close(self, day):
        """SPY price nearest 16:00 ET on `day` (expiry settlement mark)."""
        day_bars = self.spy_minute[self.spy_minute['day'] == day]
        if day_bars.empty:
            row = self.spy_daily[self.spy_daily['day'] == day]
            return float(row['c'].iloc[0]) if not row.empty else None
        et = day_bars['t'].dt.tz_convert('America/New_York')
        mins = et.dt.hour * 60 + et.dt.minute
        idx = (mins - 16 * 60).abs().idxmin()
        return float(day_bars.loc[idx, 'c'])

    def entry_mid(self, symbol, monday, warn_counter=None):
        """Mean trade close in the 10:00-10:05 ET sub-window on `monday`; falls back to the
        nearest bar in the fetched 09:55-10:10 ET window. Returns None if the symbol has no
        entry-minute bar at all that Monday (leg VOID)."""
        rows = self._entry_by_symbol_monday.get((symbol, monday))
        if rows is None or rows.empty:
            return None
        mins = rows['t_et'].dt.hour * 60 + rows['t_et'].dt.minute
        core = rows[(mins >= 10 * 60) & (mins < 10 * 60 + 5)]
        if not core.empty:
            return float(core['c'].mean())
        if warn_counter is not None:
            warn_counter['entry_mid_fallback'] = warn_counter.get('entry_mid_fallback', 0) + 1
        idx = (mins - 10 * 60).abs().idxmin()
        return float(rows.loc[idx, 'c'])

    def daily_asof(self, symbol, as_of_day):
        """Last known daily close for `symbol` at or before `as_of_day`. None if no bar exists
        on/before that date (leg VOID for management)."""
        s = self._daily_by_symbol.get(symbol)
        if s is None or s.empty:
            return None
        sub = s[s['day'] <= as_of_day]
        if sub.empty:
            return None
        return float(sub['c'].iloc[-1])

    def daily_next_open(self, symbol, after_day):
        """First daily OPEN strictly after `after_day` (stand-in for 'next session's 10:00')."""
        s = self._daily_by_symbol.get(symbol)
        if s is None or s.empty:
            return None
        sub = s[s['day'] > after_day]
        if sub.empty:
            return None
        return float(sub['o'].iloc[0])


# --------------------------------------------------------------------------- per-Monday precompute

def precompute_monday(cache, monday, warn_counter):
    """Chooses the expiry (listed, nearest 45 DTE among 38-52) and the ATM 45-DTE IV gate value
    for one entry Monday. Returns dict or None (VOID: no usable expiry)."""
    spot = cache.spot_at_10(monday)
    if spot is None:
        return None
    grid_m = cache.grid[cache.grid['monday'] == monday]
    if grid_m.empty:
        return None
    grid_m = grid_m.assign(dte=(pd.to_datetime(grid_m['expiry']) - pd.Timestamp(monday)).dt.days)
    grid_m = grid_m[(grid_m['dte'] >= DTE_LO) & (grid_m['dte'] <= DTE_HI)]
    if grid_m.empty:
        return None
    # a 'usable' expiry: at least one strike near spot has an entry-minute mid
    best = None
    for expiry, exp_grp in grid_m.groupby('expiry'):
        dte = int(exp_grp['dte'].iloc[0])
        strikes = exp_grp.sort_values('strike')
        atm_row = strikes.iloc[(strikes['strike'] - spot).abs().values.argmin()]
        atm_mid = cache.entry_mid(atm_row['symbol'], monday, warn_counter)
        if atm_mid is None:
            continue
        rank = abs(dte - DTE_TARGET)
        if best is None or rank < best['rank']:
            iv_atm = implied_vol_put(atm_mid, spot, atm_row['strike'], dte / 365.0, R_RATE, Q_RATE)
            if iv_atm is None:
                warn_counter['bad_iv_print'] = warn_counter.get('bad_iv_print', 0) + 1
                continue
            best = {'expiry': expiry, 'dte': dte, 'rank': rank, 'iv_atm': iv_atm,
                    'strikes': strikes, 'spot': spot}
    if best is None:
        return None
    return best


def select_strikes(cache, monday, mkt, target_delta, width, warn_counter):
    """Nearest-BS-delta short strike at `mkt['expiry']`, long strike `width` below. Returns dict
    with entry mids and net credit, or None (VOID: a leg lacks an entry mid or no strike has a
    solvable IV within target range)."""
    spot, dte, expiry = mkt['spot'], mkt['dte'], mkt['expiry']
    T = dte / 365.0
    strikes = mkt['strikes'].sort_values('strike')
    best_strike = None
    best_gap = None
    candidates = []
    for _, row in strikes.iterrows():
        if row['strike'] > spot:
            continue  # OTM puts only
        mid = cache.entry_mid(row['symbol'], monday, warn_counter)
        if mid is None or mid <= 0:
            continue
        iv = implied_vol_put(mid, spot, row['strike'], T, R_RATE, Q_RATE)
        if iv is None:
            warn_counter['bad_iv_print'] = warn_counter.get('bad_iv_print', 0) + 1
            continue
        delta = bs_put_delta(spot, row['strike'], T, R_RATE, Q_RATE, iv)
        gap = abs(abs(delta) - target_delta)
        candidates.append((row['strike'], row['symbol'], mid, iv, delta, gap))
        if best_gap is None or gap < best_gap:
            best_gap, best_strike = gap, (row['strike'], row['symbol'], mid, iv, delta)
    if best_strike is None:
        return None
    short_strike, short_symbol, short_mid, short_iv, short_delta = best_strike
    long_strike = short_strike - width
    long_row = strikes[strikes['strike'] == long_strike]
    if long_row.empty:
        return None
    long_symbol = long_row['symbol'].iloc[0]
    long_mid = cache.entry_mid(long_symbol, monday, warn_counter)
    if long_mid is None or long_mid <= 0:
        return None
    net_credit = (short_mid - LEG_SLIPPAGE) - (long_mid + LEG_SLIPPAGE)
    return {
        'expiry': expiry, 'dte': dte,
        'short_strike': short_strike, 'short_symbol': short_symbol, 'short_mid': short_mid,
        'short_delta': short_delta,
        'long_strike': long_strike, 'long_symbol': long_symbol, 'long_mid': long_mid,
        'net_credit': net_credit,
    }


# --------------------------------------------------------------------------- position lifecycle

def run_cycle(cache, entry, monday, width, mgmt, contracts, warn_counter):
    """Walks the daily marks forward from entry to expiry/close under management `mgmt`.
    Returns (exit_date, exit_reason, pnl_per_share, closed_actively:bool) or None if a leg's
    daily series can't support management (VOID)."""
    short_sym, long_sym = entry['short_symbol'], entry['long_symbol']
    expiry = entry['expiry']
    net_credit = entry['net_credit']
    business_days = pd.bdate_range(monday, expiry)
    if len(business_days) < 2:
        return None
    if mgmt == 'B':
        # Management B never manages intraday/daily -- it holds to expiry unconditionally, so it
        # needs no daily marks at all (falls straight through to intrinsic settlement below).
        pass
    else:
        stop_pending_from = None
        for d in business_days[1:]:
            day = d.strftime('%Y-%m-%d')
            if day >= expiry:
                break
            if stop_pending_from is not None:
                s_open = cache.daily_next_open(short_sym, stop_pending_from)
                l_open = cache.daily_next_open(long_sym, stop_pending_from)
                if s_open is None or l_open is None:
                    warn_counter['void_stop_no_next_open'] = warn_counter.get('void_stop_no_next_open', 0) + 1
                    return None
                exit_cost = (s_open + LEG_SLIPPAGE) - (l_open - LEG_SLIPPAGE)
                return day, 'stop', net_credit - exit_cost, True
            s_close = cache.daily_asof(short_sym, day)
            l_close = cache.daily_asof(long_sym, day)
            if s_close is None or l_close is None:
                warn_counter['void_missing_daily_mark'] = warn_counter.get('void_missing_daily_mark', 0) + 1
                return None
            mark = s_close - l_close  # raw cost to close today
            if mark >= STOP_MULT * net_credit and net_credit > 0:
                stop_pending_from = day
                continue
            if mark <= PROFIT_TARGET_FRAC * net_credit:
                exit_cost = (s_close + LEG_SLIPPAGE) - (l_close - LEG_SLIPPAGE)
                return day, 'profit_50', net_credit - exit_cost, True
            dte_left = (pd.Timestamp(expiry) - pd.Timestamp(day)).days
            if dte_left <= DTE_HARD_EXIT:
                exit_cost = (s_close + LEG_SLIPPAGE) - (l_close - LEG_SLIPPAGE)
                return day, 'dte21', net_credit - exit_cost, True
    # reached expiry without an active exit -> intrinsic settlement
    S_T = cache.spy_16_close(expiry)
    if S_T is None:
        warn_counter['void_no_expiry_spot'] = warn_counter.get('void_no_expiry_spot', 0) + 1
        return None
    intrinsic_short = max(0.0, entry['short_strike'] - S_T)
    intrinsic_long = max(0.0, entry['long_strike'] - S_T)
    settlement = intrinsic_short - intrinsic_long
    return expiry, 'expiry', net_credit - settlement, False


def size_position(alloc_dollars, width, net_credit):
    """floor((alloc)/((W-credit)*100)); 0 if that's 0 (PREREG sizing rule)."""
    worst_per_contract = (width - net_credit) * 100.0
    if worst_per_contract <= 0:
        return 0, worst_per_contract
    contracts = math.floor(alloc_dollars / worst_per_contract)
    return max(contracts, 0), worst_per_contract


# --------------------------------------------------------------------------- one cell

def build_entry_cache(cache, mondays_ctx, warn_counter):
    """Precomputes select_strikes() once per (monday, delta, width) -- selection depends only on
    those two knobs, not on management or the IV gate, so this is shared by all 4 (mgmt x gate)
    cells per (delta, width) instead of recomputed 4x."""
    entry_cache = {}
    for monday, mkt in mondays_ctx:
        if mkt is None:
            continue
        for delta_t in DELTAS:
            for width in WIDTHS:
                entry_cache[(monday, delta_t, width)] = select_strikes(cache, monday, mkt, delta_t, width, warn_counter)
    return entry_cache


def run_cell(cache, cell_def, mondays_ctx, entry_cache, warn_counter):
    """Simulates one (delta, width, mgmt, gate) cell across every entry Monday, tracking the
    running open-risk budget so every open obeys the sum-of-worst-cases <= B assertion."""
    delta_t, width, mgmt, gate = cell_def['delta'], cell_def['width'], cell_def['mgmt'], cell_def['gate']
    open_positions = []  # list of (exit_date, worst_case_dollars)
    cycles = []
    n_skipped_gate = 0
    n_void = 0
    n_zero_size = 0
    for monday, mkt in mondays_ctx:
        exit_dates_remaining = [p for p in open_positions if p[0] > monday]
        open_positions = exit_dates_remaining
        if mkt is None:
            n_void += 1
            continue
        if gate == 1 and mkt['iv_atm'] < IV_GATE_MIN:
            n_skipped_gate += 1
            continue
        entry = entry_cache.get((monday, delta_t, width))
        if entry is None:
            n_void += 1
            continue
        reserved = sum(p[1] for p in open_positions)
        remaining = max(B - reserved, 0.0)
        alloc = min(B / N_LADDER, remaining)
        contracts, worst_per_contract = size_position(alloc, width, entry['net_credit'])
        if contracts <= 0:
            n_zero_size += 1
            continue
        worst_case = worst_per_contract * contracts
        assert reserved + worst_case <= B + 1e-6, (
            f'BUDGET ASSERTION VIOLATED: reserved={reserved} + worst_case={worst_case} > B={B}')
        result = run_cycle(cache, entry, monday, width, mgmt, contracts, warn_counter)
        if result is None:
            n_void += 1
            continue
        exit_date, exit_reason, pnl_per_share, closed_actively = result
        open_positions.append((exit_date, worst_case))
        fee_entry = FEE_PER_CONTRACT_LEG * 2 * contracts
        fee_exit = FEE_PER_CONTRACT_LEG * 2 * contracts if closed_actively else 0.0
        pnl_usd = pnl_per_share * 100.0 * contracts - fee_entry - fee_exit
        # naked short-put comparison (report-only): same contracts, short leg only
        naked_entry_credit = entry['short_mid'] - LEG_SLIPPAGE
        naked_pnl_per_share, naked_exit_reason = naked_leg_pnl(cache, entry, monday, mgmt, warn_counter)
        naked_pnl_usd = None
        if naked_pnl_per_share is not None:
            naked_fee = FEE_PER_CONTRACT_LEG * contracts * (2 if naked_exit_reason != 'expiry' else 1)
            naked_pnl_usd = naked_pnl_per_share * 100.0 * contracts - naked_fee
        holding_days = (pd.Timestamp(exit_date) - pd.Timestamp(monday)).days
        cycles.append({
            'cell': cell_def['cell'], 'entry_date': monday, 'expiry': entry['expiry'],
            'short_strike': entry['short_strike'], 'long_strike': entry['long_strike'],
            'contracts': contracts, 'credit': entry['net_credit'], 'exit_date': exit_date,
            'exit_reason': exit_reason, 'pnl_usd': pnl_usd, 'worst_case_usd': worst_case,
            'holding_days': holding_days, 'naked_pnl_usd': naked_pnl_usd,
        })
    return cycles, {'n_skipped_gate': n_skipped_gate, 'n_void': n_void, 'n_zero_size': n_zero_size}


def naked_leg_pnl(cache, entry, monday, mgmt, warn_counter):
    """Report-only: the short leg alone (undefined risk), same management timing rule, same
    contracts as the spread. Returns (pnl_per_share, exit_reason) or (None, None) if VOID."""
    short_sym = entry['short_symbol']
    expiry = entry['expiry']
    net_credit = entry['short_mid'] - LEG_SLIPPAGE
    business_days = pd.bdate_range(monday, expiry)
    if len(business_days) < 2:
        return None, None
    stop_pending_from = None
    for d in business_days[1:]:
        day = d.strftime('%Y-%m-%d')
        if day >= expiry:
            break
        if stop_pending_from is not None:
            s_open = cache.daily_next_open(short_sym, stop_pending_from)
            if s_open is None:
                return None, None
            return net_credit - (s_open + LEG_SLIPPAGE), 'stop'
        s_close = cache.daily_asof(short_sym, day)
        if s_close is None:
            return None, None
        if mgmt == 'A':
            if s_close >= STOP_MULT * net_credit and net_credit > 0:
                stop_pending_from = day
                continue
            if s_close <= PROFIT_TARGET_FRAC * net_credit:
                return net_credit - (s_close + LEG_SLIPPAGE), 'profit_50'
            if (pd.Timestamp(expiry) - pd.Timestamp(day)).days <= DTE_HARD_EXIT:
                return net_credit - (s_close + LEG_SLIPPAGE), 'dte21'
    S_T = cache.spy_16_close(expiry)
    if S_T is None:
        return None, None
    return net_credit - max(0.0, entry['short_strike'] - S_T), 'expiry'


# --------------------------------------------------------------------------- reporting

def monthly_series(cycles_df, col='pnl_usd', date_col='exit_date'):
    if cycles_df.empty:
        return pd.Series(dtype=float)
    s = cycles_df.copy()
    s['month'] = pd.to_datetime(s[date_col]).dt.to_period('M')
    return s.groupby('month')[col].sum().sort_index()


def cell_stats(cycles_df, split_name, spy_daily):
    if cycles_df.empty:
        return {'split': split_name, 'n_cycles': 0}
    monthly = monthly_series(cycles_df)
    monthly_ret = monthly / B
    mean_monthly_ret = float(monthly_ret.mean())
    sharpe = float(monthly_ret.mean() / monthly_ret.std(ddof=1) * math.sqrt(12)) if len(monthly_ret) > 1 and monthly_ret.std(ddof=1) > 0 else float('nan')
    green_share = float((monthly > 0).mean())
    worst_month = float(monthly.min())
    cum = monthly.cumsum()
    running_max = cum.cummax()
    dd = (cum - running_max).min()
    max_dd = float(-dd) if pd.notna(dd) else 0.0
    pnl_sorted = cycles_df['pnl_usd'].sort_values(ascending=False)
    top5_n = max(1, int(math.ceil(0.05 * len(pnl_sorted))))
    top5_share = float(pnl_sorted.iloc[:top5_n].sum() / pnl_sorted.sum()) if pnl_sorted.sum() != 0 else float('nan')
    ex_top5_pnl = float(pnl_sorted.iloc[top5_n:].sum())
    win_rate = float((cycles_df['pnl_usd'] > 0).mean())
    mean_holding = float(cycles_df['holding_days'].mean())
    exit_mix = cycles_df['exit_reason'].value_counts().to_dict()
    aug24 = cycles_df[pd.to_datetime(cycles_df['exit_date']).dt.to_period('M') == pd.Period('2024-08')]['pnl_usd'].sum()
    apr25 = cycles_df[pd.to_datetime(cycles_df['exit_date']).dt.to_period('M') == pd.Period('2025-04')]['pnl_usd'].sum()
    mean_cycle_pnl = float(cycles_df['pnl_usd'].mean())
    mean_cycle_ret_on_risk = float((cycles_df['pnl_usd'] / cycles_df['worst_case_usd']).mean())
    naked_valid = cycles_df['naked_pnl_usd'].dropna()
    naked_sum = float(naked_valid.sum()) if not naked_valid.empty else float('nan')
    return {
        'split': split_name, 'n_cycles': int(len(cycles_df)),
        'mean_cycle_pnl_usd': mean_cycle_pnl, 'mean_cycle_ret_on_risk': mean_cycle_ret_on_risk,
        'mean_monthly_ret_on_B': mean_monthly_ret, 'monthly_sharpe': sharpe,
        'green_months': int((monthly > 0).sum()), 'n_months': int(len(monthly)),
        'green_month_share': green_share, 'worst_month_usd': worst_month,
        'worst_month_ge_negB': bool(worst_month >= -B - 1e-6),
        'max_dd_usd': max_dd, 'top5pct_cycles_share': top5_share, 'ex_top5pct_pnl_usd': ex_top5_pnl,
        'win_rate': win_rate, 'mean_holding_days': mean_holding, 'exit_mix': exit_mix,
        'aug2024_pnl_usd': float(aug24), 'apr2025_pnl_usd': float(apr25),
        'naked_comparison_pnl_usd': naked_sum,
    }


def spy_buy_hold(spy_daily, start, end):
    sub = spy_daily[(spy_daily['day'] >= start) & (spy_daily['day'] <= end)].sort_values('day')
    if len(sub) < 2:
        return {'return_pct': float('nan'), 'pnl_on_B_usd': float('nan')}
    ret = sub['c'].iloc[-1] / sub['c'].iloc[0] - 1.0
    return {'return_pct': float(ret), 'pnl_on_B_usd': float(ret * B)}


# --------------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke-test', action='store_true', help='first 10 Mondays only, all 24 cells')
    ap.add_argument('--out-dir', default=HERE)
    args = ap.parse_args()

    cache = Cache()
    mondays = sorted(cache.grid['monday'].unique())
    if args.smoke_test:
        mondays = mondays[:10]
    log.info('Precomputing expiry/gate choice for %d entry Mondays', len(mondays))
    warn_counter = {}
    mkt_by_monday = {}
    for i, monday in enumerate(mondays):
        mkt_by_monday[monday] = precompute_monday(cache, monday, warn_counter)
        if (i + 1) % 20 == 0:
            log.info('  precomputed %d/%d Mondays', i + 1, len(mondays))
    mondays_ctx = [(m, mkt_by_monday[m]) for m in mondays]
    n_void_mondays = sum(1 for _, v in mondays_ctx if v is None)
    log.info('Mondays with no usable expiry (VOID before any cell): %d/%d', n_void_mondays, len(mondays))

    log.info('Precomputing strike selection per (Monday, Delta, W)')
    entry_cache = build_entry_cache(cache, mondays_ctx, warn_counter)

    all_cycles = []
    all_rows = []
    for cell_def in CELLS:
        cycles, counts = run_cell(cache, cell_def, mondays_ctx, entry_cache, warn_counter)
        for c in cycles:
            c2 = dict(c)
            c2['delta'] = cell_def['delta']; c2['width'] = cell_def['width']
            c2['mgmt'] = cell_def['mgmt']; c2['gate'] = cell_def['gate']
            all_cycles.append(c2)
        df = pd.DataFrame(cycles)
        for split_name, (s, e) in [('TRAIN', (TRAIN_START, TRAIN_END)), ('VAL', (VAL_START, VAL_END)),
                                    ('FULL', (mondays[0], mondays[-1]))]:
            split_df = df[(df['entry_date'] >= s) & (df['entry_date'] <= e)] if not df.empty else df
            stats = cell_stats(split_df, split_name, cache.spy_daily)
            stats.update({'cell': cell_def['cell'], 'delta': cell_def['delta'], 'width': cell_def['width'],
                          'mgmt': cell_def['mgmt'], 'gate': cell_def['gate'], **counts,
                          'spy_bh': spy_buy_hold(cache.spy_daily, s, e)})
            all_rows.append(stats)
        log.info('Cell %d (Delta=%.2f W=%d M=%s G=%d): %d cycles, %d void, %d gate-skip, %d zero-size',
                  cell_def['cell'], cell_def['delta'], cell_def['width'], cell_def['mgmt'], cell_def['gate'],
                  len(cycles), counts['n_void'], counts['n_skipped_gate'], counts['n_zero_size'])

    log.info('WARNING counters (fallbacks/void reasons): %s', warn_counter)
    for k, v in warn_counter.items():
        log.warning('%s: %d', k, v)

    cycles_df = pd.DataFrame(all_cycles)
    cycles_df.to_csv(os.path.join(args.out_dir, 'cell_1567_cycles.csv'), index=False)
    log.info('Wrote %d cycle rows to cell_1567_cycles.csv', len(cycles_df))

    monthly_rows = []
    for cell_def in CELLS:
        cdf = cycles_df[cycles_df['cell'] == cell_def['cell']]
        if cdf.empty:
            continue
        ms = monthly_series(cdf)
        for month, pnl in ms.items():
            monthly_rows.append({'cell': cell_def['cell'], 'month': str(month), 'pnl_usd': pnl,
                                  'return_on_B': pnl / B})
    pd.DataFrame(monthly_rows).to_csv(os.path.join(args.out_dir, 'cell_1567_monthly.csv'), index=False)

    rows_df = pd.DataFrame(all_rows)
    rows_df.to_json(os.path.join(args.out_dir, 'cell_1567_rows.json'), orient='records', indent=2)

    write_result_md(rows_df, args.out_dir, warn_counter, n_void_mondays, len(mondays))
    log.info('Done.')


def write_result_md(rows_df, out_dir, warn_counter, n_void_mondays, n_mondays):
    train = rows_df[rows_df['split'] == 'TRAIN'].copy()
    eligible = train[(train['n_cycles'] >= 12) & (train['green_month_share'] >= 0.55)]
    lines = ['# RESULT_1567 — defined-risk put-credit-spread ladder on SPY\n']
    lines.append(f'Equity fixed at ${EQUITY:,.0f} for the whole backtest; B = {B_FRAC:.0%} = ${B:,.2f}. '
                 f'{n_mondays} entry Mondays, {n_void_mondays} with no usable listed expiry (VOID before any cell).\n')
    lines.append('## TRAIN selection (highest monthly Sharpe, n_cycles>=12, green_months>=55%)\n')
    if eligible.empty:
        lines.append('**No cell cleared the TRAIN eligibility bar (>=12 cycles and >=55% green months).**\n')
        selected_cell = None
    else:
        best = eligible.sort_values('monthly_sharpe', ascending=False).iloc[0]
        selected_cell = int(best['cell'])
        lines.append(f"Selected cell **{selected_cell}** (Delta={best['delta']}, W={best['width']}, "
                      f"M={best['mgmt']}, G={best['gate']}): TRAIN monthly Sharpe {best['monthly_sharpe']:.2f}, "
                      f"n_cycles {int(best['n_cycles'])}, green months {best['green_month_share']:.0%}.\n")
    lines.append('\n## Full TRAIN table (all 24 cells)\n')
    cols = ['cell', 'delta', 'width', 'mgmt', 'gate', 'n_cycles', 'mean_monthly_ret_on_B',
            'monthly_sharpe', 'green_month_share', 'worst_month_usd', 'max_dd_usd', 'win_rate']
    lines.append(train[cols].sort_values('monthly_sharpe', ascending=False).to_markdown(index=False))
    if selected_cell is not None:
        val = rows_df[(rows_df['split'] == 'VAL') & (rows_df['cell'] == selected_cell)]
        if not val.empty:
            v = val.iloc[0]
            lines.append('\n\n## VAL read of the selected cell\n')
            lines.append(f"n_cycles={int(v['n_cycles'])}, mean monthly return on B={v['mean_monthly_ret_on_B']:.2%}, "
                          f"monthly Sharpe={v['monthly_sharpe']:.2f}, green months={v['green_month_share']:.0%}, "
                          f"ex-top5% cycle PnL=${v['ex_top5pct_pnl_usd']:,.0f}, worst month=${v['worst_month_usd']:,.0f} "
                          f"(>= -B: {v['worst_month_ge_negB']}), max DD=${v['max_dd_usd']:,.0f} "
                          f"(<=1.5B={1.5*B:,.0f}: {v['max_dd_usd'] <= 1.5*B}).\n")
            spy_bh = v['spy_bh']
            lines.append(f"SPY buy-and-hold over the same VAL window: return {spy_bh['return_pct']:.2%}, "
                          f"${spy_bh['pnl_on_B_usd']:,.0f} on B.\n")
            pass_checklist = {
                'mean monthly return >= 4%': v['mean_monthly_ret_on_B'] >= 0.04,
                'monthly Sharpe >= 1.0': v['monthly_sharpe'] >= 1.0,
                'green months >= 60%': v['green_month_share'] >= 0.60,
                'ex-top5% cycles positive': v['ex_top5pct_pnl_usd'] > 0,
                'worst month >= -B': v['worst_month_ge_negB'],
                'max DD <= 1.5B': v['max_dd_usd'] <= 1.5 * B,
            }
            lines.append('\nPass-bar checklist: ' + ', '.join(f'{k}={"PASS" if ok else "FAIL"}' for k, ok in pass_checklist.items()) + '\n')
            lines.append(f"\n**Overall: {'PASS' if all(pass_checklist.values()) else 'FAIL'}** "
                         "(neighbour check and TRAIN-same-sign must also be read from the full VAL table below before shipping).\n")
    lines.append('\n\n## Full VAL table (all 24 cells, unselected ones labelled for the record)\n')
    val_all = rows_df[rows_df['split'] == 'VAL']
    lines.append(val_all[cols].sort_values('monthly_sharpe', ascending=False).to_markdown(index=False))
    lines.append('\n\n## Data caveats (carried from FETCH_1567.md)\n')
    lines.append('* Option bars are trade OHLCV, not quotes: "mid" is the mean trade close in the 10:00-10:05 '
                 'ET sub-window (entry) and the as-of daily close (ongoing management), both documented approximations.\n')
    lines.append('* Management marks use daily closes, not intraday quotes (minute bars were fetched for the '
                 'entry Monday only, per FETCH_1567.md\'s scope reduction); the STOP exit uses the next session\'s '
                 'daily OPEN as a stand-in for "next session\'s 10:00".\n')
    lines.append(f'* WARNING/fallback counters: {warn_counter}\n')
    with open(os.path.join(out_dir, 'RESULT_1567.md'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
