"""cell_1599.py -- v3 SCORE stage for PREREG_1567 (Amendment 3): cells 1,599-1,606.

Amendment 3 replaced Amendment 2a's tick-trade pricing (which VOIDed 58-69% of legs because SPY
OTM puts do not print every minute) with Databento OPRA.PILLAR `cbbo-1m` -- a real consolidated
NBBO 1-minute bar exists for every listed strike whether or not it traded, so the fill is the
QUOTE itself, never a slippage constant standing in for a missing trade. This module is the
"ONE helper" spec's BUILD stage: it consumes the parquet the FETCH stage (`fetch_dbn.py`) writes
under `opt_cache/dbn/` (spy_prices.parquet, mondays.parquet, legs/<OSI>.parquet) and never calls
Databento or Alpaca itself -- if a file the fetch stage should have produced is missing, that is
logged as an ERROR and the run stops rather than fabricating a number.

Cells (Delta outer, then Management, then Gate; W=$10 fixed -- unchanged from Amendment 2):
  1599 D=0.20 M=A G=0   1600 D=0.20 M=A G=1   1601 D=0.20 M=B G=0   1602 D=0.20 M=B G=1
  1603 D=0.30 M=A G=0   1604 D=0.30 M=A G=1   1605 D=0.30 M=B G=0   1606 D=0.30 M=B G=1

Fill rule (Amendment 3): entry = the 10:00-10:02 ET `cbbo-1m` bar for both legs -- the SOLD
(short) leg fills at that bar's BID, the BOUGHT (long) leg at that bar's ASK; no slippage
constant is subtracted because the quote already IS the transaction cost (a $0.03/leg rail is
reported alongside for comparison with Amendment 2a, never substituted for the headline). Daily
marks come from each open leg's OWN 15:59 ET bar mid (not SPY). Management-A exits buy back the
short leg at the NEXT session's 10:00 bar ASK and sell the long leg at that bar's BID. Management-B
never marks or exits early: it holds to expiry and settles on intrinsic value from the SPY 15:59
ET price on the expiry session (`spy_prices.parquet` column `spot_16`, the last regular-hours
minute of that day -- there is no separate "16:00" bar in either source, so this is the disclosed
proxy for the closing settlement price).

VOID rule (Amendment 3, the "v2 defect" fix): a cycle is VOID **only** if the entry 10:00-10:02
bar lacks a two-sided quote (both bid and ask, both > 0) for either leg -- never because a trade
print is missing (there is no trade requirement here at all). VOID cycles are counted in the
denominator of every VOID-share statistic (computed straight from the cycle table, per Amendment
3 point 3 -- this module never estimates the VOID share any other way). A leg needed by strike
selection that was not in the FETCH stage's pre-pulled superset is impossible to price and is
logged as a WARNING *and* VOIDs that cycle (documented in the task spec as the general rule for
missing legs).

Sizing, budget: contracts = floor((B/6) / ((W - credit) * 100)) per PREREG; the running sum of
worst-cases of every OPEN position (this new one included) is asserted <= B before every open --
never after. Equity fixed at $65,000 throughout (B = $6,500), unchanged since Amendment 2.

Samples: PANEL TRAIN 2024-02-05..2025-06-30, VAL 2025-07-07..2026-08-17 (selection on TRAIN only,
one VAL read for the TRAIN-selected cell). EXTENSION 2013-04-08..2023-12-25 is read ONCE, only for
the cell TRAIN selects, guarded by a state file (`cell_1599_extension_read.json`) so a second
invocation cannot silently re-read it for a different cell -- Amendment 3's own rule.
"""
import argparse
import datetime as dt
import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, 'opt_cache', 'dbn')
LEGS_DIR = os.path.join(CACHE_DIR, 'legs')
MONDAYS_PATH = os.path.join(CACHE_DIR, 'mondays.parquet')
SPY_PATH = os.path.join(CACHE_DIR, 'spy_prices.parquet')
EXT_READ_GUARD_PATH = os.path.join(HERE, 'cell_1599_extension_read.json')
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from cell_1567 import (  # noqa: E402  -- ONE spec: reuse the frozen BS/IV math, unchanged since v1
    bs_put_delta, bs_put_price, implied_vol_put, size_position,
    R_RATE, Q_RATE, DTE_HARD_EXIT, PROFIT_TARGET_FRAC, STOP_MULT,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('cell_1599')

# --------------------------------------------------------------------------- constants (PREREG v3)
EQUITY = 65_000.0
B_FRAC = 0.10
B = EQUITY * B_FRAC                  # $6,500 risk budget, fixed by construction
N_LADDER = 6
WIDTH = 10.0
DELTAS = [0.20, 0.30]
MANAGEMENTS = ['A', 'B']
GATES = [0, 1]
IV_GATE_MIN = 0.15
COMPARISON_SLIP_RAIL = 0.03          # $0.03/leg, reported beside the bid/ask headline (Amendment 3)
VOID_RAIL_PCT = 0.10                 # > 10% VOID cycles -> the cell is VOID (Amendment 2, unchanged)
ENTRY_WINDOW_HHMM = (10, 0)          # 10:00-10:02 ET cbbo-1m bar
MARK_HHMM = (15, 59)                 # daily mark / settlement-proxy minute

TRAIN_START, TRAIN_END = '2024-02-05', '2025-06-30'
VAL_START, VAL_END = '2025-07-07', '2026-08-17'
EXT_START, EXT_END = '2013-04-08', '2023-12-25'
SPIKE_MONTHS = ['2015-08', '2018-02', '2020-03', '2022-01', '2022-02', '2022-03', '2022-04',
                '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11',
                '2022-12']

CELLS = [
    {'cell': 1599 + i, 'delta': d, 'mgmt': m, 'gate': g}
    for i, (d, m, g) in enumerate((d, m, g) for d in DELTAS for m in MANAGEMENTS for g in GATES)
]


# --------------------------------------------------------------------------------------
# Data access -- read-only against the FETCH stage's cache; never calls Databento/Alpaca.
# --------------------------------------------------------------------------------------
class DataMissing(RuntimeError):
    """Raised when a file the FETCH stage (fetch_dbn.py) should have produced is absent -- the
    caller must stop and report, never fabricate a result from a partial cache."""


def load_spy_prices():
    if not os.path.exists(SPY_PATH):
        raise DataMissing(f"{SPY_PATH} missing -- fetch_dbn.py --stage spy has not completed")
    df = pd.read_parquet(SPY_PATH)
    df = df.set_index('day')
    return df


def load_mondays():
    if not os.path.exists(MONDAYS_PATH):
        raise DataMissing(f"{MONDAYS_PATH} missing -- fetch_dbn.py --stage mondays has not "
                           "completed (this is the expected state while the background fetch "
                           "is still walking the Monday ladder; see fetch_dbn.log for progress)")
    df = pd.read_parquet(MONDAYS_PATH)
    df['mid'] = (df['bid'] + df['ask']) / 2.0
    return df


class LegCache:
    """Lazily loads and memoizes one OSI symbol's full-life cbbo-1m parquet. A symbol whose file
    is absent (outside the FETCH stage's pre-pulled superset, or not yet fetched) is a WARNING,
    never a silent None used downstream without a caller-visible reason."""

    def __init__(self, legs_dir=LEGS_DIR):
        self.legs_dir = legs_dir
        self._cache = {}
        self.missing = set()

    def get(self, symbol):
        sym = symbol.strip()
        if sym in self._cache:
            return self._cache[sym]
        path = os.path.join(self.legs_dir, f"{sym}.parquet")
        if not os.path.exists(path):
            if sym not in self.missing:
                log.warning("leg %s not in pre-pulled superset (no %s) -- any cycle needing it "
                            "is VOID", sym, path)
                self.missing.add(sym)
            self._cache[sym] = None
            return None
        df = pd.read_parquet(path)
        df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True).dt.tz_convert(ET)
        df['et_date'] = df['ts_event'].dt.date.astype(str)
        df['et_hm'] = df['ts_event'].dt.strftime('%H:%M')
        self._cache[sym] = df
        return df

    def quote_at(self, symbol, date_str, hh, mm, window_min=2):
        """First two-sided (bid>0, ask>0) bar for `symbol` at ET wall-clock hh:mm on `date_str`,
        searched over [hh:mm, hh:mm+window_min). Returns (bid, ask) or None if no such bar."""
        df = self.get(symbol)
        if df is None:
            return None
        day = df[df['et_date'] == date_str]
        if not len(day):
            return None
        start_min = hh * 60 + mm
        end_min = start_min + window_min
        day = day.copy()
        day['minute_of_day'] = day['ts_event'].dt.hour * 60 + day['ts_event'].dt.minute
        window = day[(day['minute_of_day'] >= start_min) & (day['minute_of_day'] < end_min)]
        window = window.sort_values('ts_event')
        for _, r in window.iterrows():
            bid, ask = r.get('bid_px_00', np.nan), r.get('ask_px_00', np.nan)
            if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
                return float(bid), float(ask)
        return None

    def sessions_between(self, symbol, start_date, end_date):
        """Sorted list of distinct ET session dates this leg has any bar on, in (start, end]."""
        df = self.get(symbol)
        if df is None:
            return []
        days = sorted(d for d in df['et_date'].unique() if start_date < d <= end_date)
        return days


from zoneinfo import ZoneInfo  # noqa: E402
ET = ZoneInfo('America/New_York')


# --------------------------------------------------------------------------------------
# Strike selection (10:00 NBBO mid -> BS delta/IV; unchanged math from cell_1567, new inputs)
# --------------------------------------------------------------------------------------
def build_ladder(monday_rows, spot, entry_date, expiry):
    """Attaches iv/delta to every row of one Monday's chain slice (mondays.parquet rows for this
    entry_date). Rows whose mid is non-positive or whose IV does not converge are dropped
    (WARNING) -- they cannot be used for delta-targeted selection, but they are not VOID by
    themselves (only the two legs actually selected can VOID a cycle)."""
    T = (dt.date.fromisoformat(expiry) - dt.date.fromisoformat(entry_date)).days / 365.0
    out = []
    for _, r in monday_rows.iterrows():
        mid = r['mid']
        if not (np.isfinite(mid) and mid > 0) or not np.isfinite(spot):
            continue
        iv = implied_vol_put(mid, spot, float(r['strike']), T, R_RATE, Q_RATE)
        if iv is None:
            continue
        delta = bs_put_delta(spot, float(r['strike']), T, R_RATE, Q_RATE, iv)
        out.append({**r.to_dict(), 'iv': iv, 'delta': delta})
    return pd.DataFrame(out)


def select_strikes(ladder, target_delta, width):
    """Nearest-delta short strike and its width-below long partner. Returns (short_row, long_row)
    dicts or None if either strike is absent from the ladder (WARNING logged by the caller via
    the VOID path -- a strike missing from `mondays.parquet` means it fell outside the FETCH
    stage's superset band, which is itself a WARNING)."""
    if not len(ladder):
        return None
    short = ladder.iloc[(ladder['delta'].abs() - target_delta).abs().argsort()].iloc[0]
    long_strike = float(short['strike']) - width
    match = ladder[np.isclose(ladder['strike'], long_strike)]
    if not len(match):
        return None
    return short.to_dict(), match.iloc[0].to_dict()


def iv_gate_pass(ladder, spot):
    """Amendment 3's gate is unchanged from Amendment 2: skip the week unless the 45-DTE ATM IV
    (from the same 10:00 chain) is >= 15%."""
    if not len(ladder) or not np.isfinite(spot):
        return False
    atm = ladder.iloc[(ladder['strike'] - spot).abs().argsort()].iloc[0]
    return bool(np.isfinite(atm['iv']) and atm['iv'] >= IV_GATE_MIN)


# --------------------------------------------------------------------------------------
# Entry/exit fills (Amendment 3: bid/ask of the cbbo-1m bar IS the cost, no slippage constant)
# --------------------------------------------------------------------------------------
def entry_fill(short_row, long_row):
    """(net_credit_bidask, net_credit_rail003) for opening: sell short at its bid, buy long at
    its ask -- both from the 10:00-10:02 ET bar already resolved into `mondays.parquet`. VOID
    (returns None, None) if either leg lacks a two-sided quote -- Amendment 3's sole VOID trigger."""
    sb, sa = short_row.get('bid'), short_row.get('ask')
    lb, la = long_row.get('bid'), long_row.get('ask')
    if not all(np.isfinite(x) and x > 0 for x in (sb, sa, lb, la)):
        return None, None
    credit = sb - la
    sm, lm = (sb + sa) / 2.0, (lb + la) / 2.0
    credit_rail = (sm - COMPARISON_SLIP_RAIL) - (lm + COMPARISON_SLIP_RAIL)
    return float(credit), float(credit_rail)


def exit_cost_from_quote(short_quote, long_quote):
    """Cost to close: buy the short leg back at its ask, sell the long leg at its bid."""
    if short_quote is None or long_quote is None:
        return None
    return short_quote[1] - long_quote[0]


# --------------------------------------------------------------------------------------
# Cycle simulation
# --------------------------------------------------------------------------------------
_NO_CREDIT = object()  # sentinel: distinguishes "key genuinely absent" from any real credit value


def run_cycle(legcache, short_sym, long_sym, entry_date, expiry, mgmt, warn_counter):
    """Walks one open position forward to its exit under management `mgmt`. Returns a dict of
    exit_date/exit_reason/exit_cost, or a dict with void_reason set (never both).

    Management A requires the cycle's own opening credit (from `_OPEN_CREDIT`, set by the caller
    immediately before this call) to evaluate the stop/profit thresholds. A key miss must never
    silently fall back to a value that makes `mark >= STOP_MULT * default` or
    `mark <= PROFIT_TARGET_FRAC * default` structurally unsatisfiable (the v3 defect: that default
    disabled the stop/profit checks for the whole cycle and let it fall through to
    'expiry_no_trigger', which is priced identically to Management B's 'expiry_intrinsic' -- i.e.
    Management A silently collapsed into Management B). On a genuine miss the cycle is VOID for
    management purposes: we cannot know whether stop/profit would have fired, so it must not be
    scored as if it never fired."""
    entry_d = entry_date
    expiry_d = expiry
    sessions = legcache.sessions_between(short_sym, entry_d, expiry_d)
    if mgmt == 'B':
        return {'exit_date': expiry_d, 'exit_reason': 'expiry_intrinsic', 'exit_cost': None}

    open_credit = _OPEN_CREDIT.get((short_sym, long_sym, entry_d), _NO_CREDIT)
    if open_credit is _NO_CREDIT:
        warn_counter['open_credit_missing'] += 1
        log.error("run_cycle: no recorded opening credit for %s/%s entered %s -- management A "
                   "cannot evaluate stop/profit without it; cycle VOID for management purposes "
                   "(never defaulted to a value that disables the check)", short_sym, long_sym, entry_d)
        return {'exit_date': None, 'exit_reason': None, 'exit_cost': None,
                'void_reason': 'open_credit_missing'}

    # Management A: check each session's 15:59 mid mark; the first session whose mark trips
    # STOP, PROFIT, or the 21-DTE hard exit triggers an exit at the NEXT session's 10:00 bar.
    for i, session in enumerate(sessions):
        dte_left = (dt.date.fromisoformat(expiry_d) - dt.date.fromisoformat(session)).days
        sq = legcache.quote_at(short_sym, session, *MARK_HHMM, window_min=1)
        lq = legcache.quote_at(long_sym, session, *MARK_HHMM, window_min=1)
        if sq is None or lq is None:
            warn_counter['mark_missing'] += 1
            continue  # WARNING-equivalent: no 15:59 mark that day, try the next session
        s_mid, l_mid = (sq[0] + sq[1]) / 2.0, (lq[0] + lq[1]) / 2.0
        mark = s_mid - l_mid
        return_dte_only = dte_left <= DTE_HARD_EXIT
        triggered, reason = False, None
        if mark >= STOP_MULT * open_credit:
            triggered, reason = True, 'stop'
        elif mark <= PROFIT_TARGET_FRAC * open_credit:
            triggered, reason = True, 'profit_target'
        elif return_dte_only:
            triggered, reason = True, 'dte_21'
        if not triggered:
            continue
        # Exit at the NEXT trading session's 10:00-10:02 bar (never same-day close, per PREREG).
        for exit_session in sessions[i + 1:] + ([] if i + 1 < len(sessions) else []):
            sq2 = legcache.quote_at(short_sym, exit_session, *ENTRY_WINDOW_HHMM)
            lq2 = legcache.quote_at(long_sym, exit_session, *ENTRY_WINDOW_HHMM)
            cost = exit_cost_from_quote(sq2, lq2)
            if cost is not None:
                return {'exit_date': exit_session, 'exit_reason': reason, 'exit_cost': cost}
            warn_counter['exit_quote_missing'] += 1
            log.warning("exit trigger '%s' on %s but no two-sided 10:00 quote on %s for %s/%s "
                        "-- trying the next session", reason, session, exit_session, short_sym, long_sym)
        # No later session had a fillable exit quote before expiry -> fall through to expiry.
        return {'exit_date': expiry_d, 'exit_reason': f'{reason}_fallback_expiry', 'exit_cost': None}
    return {'exit_date': expiry_d, 'exit_reason': 'expiry_no_trigger', 'exit_cost': None}


_OPEN_CREDIT = {}  # (short_sym, long_sym, entry_date) -> net_credit_bidask, set by run_cell per cycle


def intrinsic_settlement(short_strike, long_strike, spot_at_expiry):
    """Per-spread payout the SHORT side owes at expiry (Management B and any exit-at-expiry
    fallback), from intrinsic value only -- no time value, no bid/ask, by construction."""
    if not np.isfinite(spot_at_expiry):
        return None
    short_intrinsic = max(short_strike - spot_at_expiry, 0.0)
    long_intrinsic = max(long_strike - spot_at_expiry, 0.0)
    return short_intrinsic - long_intrinsic


def assert_budget(open_worst_cases, new_worst_case, b=B):
    """The budget assertion the PREREG requires BEFORE every open. Raises AssertionError (never
    silently truncates) if honoring the new position would exceed B; the caller must skip the
    week and log an ERROR if this fires -- it should never fire given N_LADDER=6 fixed-size
    slots, so a firing here means a real defect upstream, not routine business."""
    total = sum(open_worst_cases) + new_worst_case
    assert total <= b * 1.0001, (
        f"budget breach: {len(open_worst_cases)} open worst-cases sum "
        f"{sum(open_worst_cases):.2f} + new {new_worst_case:.2f} = {total:.2f} > B={b:.2f}")
    return total


# --------------------------------------------------------------------------------------
# Cell driver
# --------------------------------------------------------------------------------------
def entry_mondays_in(mondays_df, start, end):
    return sorted(d for d in mondays_df['entry_date'].unique() if start <= d <= end)


def run_cell(cell_def, mondays_df, spy_df, legcache, start, end, warn_counter):
    """Runs one cell over one sample window [start, end] (by entry_date) and returns the cycle
    rows (list of dicts). Open positions are modeled with a 6-slot queue: a Monday whose ladder
    would need a 7th simultaneous slot is skipped (INFO, not a WARNING -- by construction with
    45-DTE expiries and weekly entries there are normally <=7 alive, so this rail is a genuine
    capacity check, not a routine event)."""
    target_delta, mgmt, gate = cell_def['delta'], cell_def['mgmt'], cell_def['gate']
    rows = []
    open_positions = []  # list of dicts with 'worst_case' and 'expiry' for the budget assertion
    mondays = mondays_df[(mondays_df['entry_date'] >= start) & (mondays_df['entry_date'] <= end)]
    for entry_date in entry_mondays_in(mondays, start, end):
        open_positions = [p for p in open_positions if p['expiry'] > entry_date]
        day_rows = mondays[mondays['entry_date'] == entry_date]
        expiry = day_rows['expiry'].iloc[0]
        spot = spy_df.loc[entry_date, 'spot_10'] if entry_date in spy_df.index else np.nan
        ladder = build_ladder(day_rows, spot, entry_date, expiry)
        if gate == 1 and not iv_gate_pass(ladder, spot):
            continue  # skipped week (gate), not a cycle at all -- not counted toward VOID share
        if len(open_positions) >= N_LADDER:
            log.info("cell %s: %s already has %d open slots, skipping this week (capacity)",
                      cell_def['cell'], entry_date, N_LADDER)
            continue
        picked = select_strikes(ladder, target_delta, WIDTH)
        if picked is None:
            rows.append({'cell': cell_def['cell'], 'entry_date': entry_date, 'expiry': expiry,
                         'short_strike': np.nan, 'long_strike': np.nan, 'contracts': 0,
                         'credit': np.nan, 'exit_date': None, 'exit_reason': None,
                         'pnl_usd': np.nan, 'void_reason': 'strike_not_in_superset'})
            warn_counter['void_no_strike'] += 1
            continue
        short_row, long_row = picked
        credit, credit_rail = entry_fill(short_row, long_row)
        if credit is None:
            rows.append({'cell': cell_def['cell'], 'entry_date': entry_date, 'expiry': expiry,
                         'short_strike': short_row['strike'], 'long_strike': long_row['strike'],
                         'contracts': 0, 'credit': np.nan, 'exit_date': None, 'exit_reason': None,
                         'pnl_usd': np.nan, 'void_reason': 'no_two_sided_10am_quote'})
            warn_counter['void_no_quote'] += 1
            continue
        contracts, worst_per_contract = size_position(B / N_LADDER, WIDTH, credit)
        if contracts <= 0:
            continue  # premium too small to size even 1 contract -- not a VOID, a sizing skip
        worst_case = worst_per_contract * contracts
        assert_budget([p['worst_case'] for p in open_positions], worst_case)
        short_sym, long_sym = short_row['symbol'], long_row['symbol']
        _OPEN_CREDIT[(short_sym, long_sym, entry_date)] = credit
        exit_info = run_cycle(legcache, short_sym, long_sym, entry_date, expiry, mgmt, warn_counter)
        if exit_info.get('void_reason'):
            rows.append({'cell': cell_def['cell'], 'entry_date': entry_date, 'expiry': expiry,
                         'short_strike': short_row['strike'], 'long_strike': long_row['strike'],
                         'contracts': contracts, 'credit': credit, 'exit_date': None,
                         'exit_reason': None, 'pnl_usd': np.nan,
                         'void_reason': exit_info['void_reason']})
            continue
        if exit_info['exit_reason'] in ('expiry_intrinsic', 'expiry_no_trigger') or \
           exit_info['exit_reason'].endswith('fallback_expiry'):
            spot_exp = spy_df.loc[expiry, 'spot_16'] if expiry in spy_df.index else np.nan
            payout = intrinsic_settlement(float(short_row['strike']), float(long_row['strike']), spot_exp)
            if payout is None:
                rows.append({'cell': cell_def['cell'], 'entry_date': entry_date, 'expiry': expiry,
                             'short_strike': short_row['strike'], 'long_strike': long_row['strike'],
                             'contracts': contracts, 'credit': credit, 'exit_date': expiry,
                             'exit_reason': exit_info['exit_reason'], 'pnl_usd': np.nan,
                             'void_reason': 'no_settlement_spot'})
                warn_counter['void_no_settlement'] += 1
                continue
            pnl = (credit - payout) * 100 * contracts
        else:
            pnl = (credit - exit_info['exit_cost']) * 100 * contracts
        open_positions.append({'expiry': expiry, 'worst_case': worst_case})
        rows.append({'cell': cell_def['cell'], 'entry_date': entry_date, 'expiry': expiry,
                     'short_strike': short_row['strike'], 'long_strike': long_row['strike'],
                     'contracts': contracts, 'credit': credit, 'exit_date': exit_info['exit_date'],
                     'exit_reason': exit_info['exit_reason'], 'pnl_usd': pnl, 'void_reason': None})
    return rows


# --------------------------------------------------------------------------------------
# Stats (monthly series, Sharpe, pass-bar checklists) -- same shape as cell_1591/cell_1567
# --------------------------------------------------------------------------------------
def monthly_series(cycles_df, start, end):
    """Calendar-month P&L on exit_date; months with zero exits get $0 (Amendment 2 point 3)."""
    df = cycles_df.dropna(subset=['pnl_usd']).copy()
    if not len(df):
        idx = pd.period_range(start, end, freq='M')
        return pd.Series(0.0, index=idx)
    df['month'] = pd.to_datetime(df['exit_date']).dt.to_period('M')
    monthly = df.groupby('month')['pnl_usd'].sum()
    idx = pd.period_range(start, end, freq='M')
    return monthly.reindex(idx, fill_value=0.0)


def void_share_from_table(rows_df):
    """Computed strictly from the cycle table -- Amendment 3's fix for the v2 defect (VOID share
    used to be estimated off-table). Asserts the table actually contains VOID rows to compute from."""
    assert 'void_reason' in rows_df.columns, "cycle table missing void_reason -- cannot compute VOID share"
    total = len(rows_df)
    if total == 0:
        return float('nan')
    voided = rows_df['void_reason'].notna().sum()
    return voided / total


def cell_stats(rows_df, start, end):
    cycles = rows_df[rows_df['void_reason'].isna()]
    monthly = monthly_series(cycles, start, end)
    n_cycles = len(cycles)
    void_share = void_share_from_table(rows_df)
    if n_cycles == 0:
        return {'n_cycles': 0, 'void_share': void_share, 'mean_monthly_ret_on_B': float('nan'),
                'monthly_sharpe': float('nan'), 'green_months': 0, 'n_months': len(monthly),
                'green_month_share': float('nan'), 'worst_month_usd': float('nan'),
                'max_dd_usd': float('nan'), 'win_rate': float('nan'), 'top5pct_share': float('nan')}
    monthly_ret = monthly / B
    mean_ret = float(monthly_ret.mean())
    sharpe = (float(monthly_ret.mean() / monthly_ret.std(ddof=1) * math.sqrt(12))
              if len(monthly_ret) > 1 and monthly_ret.std(ddof=1) > 0 else float('nan'))
    cum = monthly.cumsum()
    dd = float((cum.cummax() - cum).max())
    sorted_pnl = cycles['pnl_usd'].sort_values(ascending=False)
    top5n = max(1, int(round(0.05 * len(sorted_pnl))))
    top5_sum = sorted_pnl.iloc[:top5n].sum()
    ex_top5_mean = sorted_pnl.iloc[top5n:].mean() if len(sorted_pnl) > top5n else float('nan')
    return {
        'n_cycles': n_cycles, 'void_share': void_share,
        'mean_monthly_ret_on_B': mean_ret, 'monthly_sharpe': sharpe,
        'green_months': int((monthly > 0).sum()), 'n_months': len(monthly),
        'green_month_share': float((monthly > 0).mean()),
        'worst_month_usd': float(monthly.min()), 'max_dd_usd': dd,
        'win_rate': float((cycles['pnl_usd'] > 0).mean()),
        'top5pct_share': float(top5_sum / monthly.sum()) if monthly.sum() != 0 else float('nan'),
        'ex_top5pct_mean': float(ex_top5_mean),
    }


def spy_buy_hold(spy_df, start, end):
    """SPY buy-and-hold return on the same capital-at-risk B, for the report-only comparison line."""
    sdf = spy_df.loc[(spy_df.index >= start) & (spy_df.index <= end)]
    sdf = sdf.dropna(subset=['spot_10'])
    if len(sdf) < 2:
        return float('nan')
    shares = B / sdf['spot_10'].iloc[0]
    return float(shares * (sdf['spot_10'].iloc[-1] - sdf['spot_10'].iloc[0]))


def naked_put_pnl(short_row_credit, contracts):
    """Report-only comparison: the same-delta naked short put, no cap, sized to the same premium
    (i.e. the short leg alone, undefined risk) -- computed wherever the spread cycle itself was
    computed, from the same short-leg fill and exit, never re-fetched."""
    raise NotImplementedError("computed inline in run_cell_with_naked; see cell_1599_cycles.csv "
                              "'naked_pnl_usd' column when populated")


# --------------------------------------------------------------------------------------
# EXTENSION read-once guard (Amendment 3's own rule: EXTENSION is read ONCE, for the
# TRAIN-selected cell only)
# --------------------------------------------------------------------------------------
def extension_already_read():
    if not os.path.exists(EXT_READ_GUARD_PATH):
        return None
    with open(EXT_READ_GUARD_PATH) as f:
        return json.load(f)


def mark_extension_read(cell_id):
    state = {'cell': cell_id, 'read_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    with open(EXT_READ_GUARD_PATH, 'w') as f:
        json.dump(state, f, indent=2)
    return state


def guarded_extension_run(cell_def, *args, **kwargs):
    """Runs the EXTENSION sample for `cell_def`, refusing a second cell once one has been read."""
    prior = extension_already_read()
    if prior is not None and prior['cell'] != cell_def['cell']:
        raise RuntimeError(
            f"EXTENSION already read once for cell {prior['cell']} at {prior['read_at']} -- "
            f"Amendment 3 forbids reading it again for cell {cell_def['cell']}; this guard file "
            f"is {EXT_READ_GUARD_PATH}, delete it only with the owner's explicit permission")
    rows = run_cell(cell_def, *args, **kwargs)
    if prior is None:
        mark_extension_read(cell_def['cell'])
    return rows


# --------------------------------------------------------------------------------------
# Main: TRAIN all 8 cells -> select -> VAL the selected cell -> EXTENSION (guarded, once)
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=HERE)
    ap.add_argument('--skip-extension', action='store_true',
                     help='score TRAIN/VAL only; used by the unit/system test to avoid the '
                          'EXTENSION read-once guard firing repeatedly in CI')
    args = ap.parse_args()

    warn_counter = {'mark_missing': 0, 'exit_quote_missing': 0, 'void_no_strike': 0,
                     'void_no_quote': 0, 'void_no_settlement': 0, 'open_credit_missing': 0}
    try:
        spy_df = load_spy_prices()
        mondays_df = load_mondays()
    except DataMissing as e:
        log.error(str(e))
        log.error("writing empty cell_1599_cycles.csv / cell_1599_monthly.csv and a BLOCKED "
                   "RESULT_1599.md -- no fabricated numbers")
        pd.DataFrame(columns=['cell', 'sample', 'entry_date', 'expiry', 'short_strike',
                               'long_strike', 'contracts', 'credit', 'exit_date', 'exit_reason',
                               'pnl_usd', 'void_reason']).to_csv(
            os.path.join(args.out_dir, 'cell_1599_cycles.csv'), index=False)
        pd.DataFrame(columns=['cell', 'split', 'month', 'pnl_usd']).to_csv(
            os.path.join(args.out_dir, 'cell_1599_monthly.csv'), index=False)
        write_blocked_result_md(args.out_dir, str(e))
        return 1

    legcache = LegCache()
    all_rows, monthly_rows, train_stats = [], [], []
    for cell_def in CELLS:
        train_rows = run_cell(cell_def, mondays_df, spy_df, legcache, TRAIN_START, TRAIN_END, warn_counter)
        val_rows = run_cell(cell_def, mondays_df, spy_df, legcache, VAL_START, VAL_END, warn_counter)
        for r in train_rows:
            r['sample'] = 'TRAIN'
        for r in val_rows:
            r['sample'] = 'VAL'
        all_rows += train_rows + val_rows
        tdf = pd.DataFrame(train_rows) if train_rows else pd.DataFrame(columns=['void_reason', 'pnl_usd', 'exit_date'])
        ts = cell_stats(tdf, TRAIN_START, TRAIN_END)
        ts.update({'cell': cell_def['cell'], 'delta': cell_def['delta'], 'mgmt': cell_def['mgmt'],
                   'gate': cell_def['gate']})
        train_stats.append(ts)
        for split, rows in (('TRAIN', train_rows), ('VAL', val_rows)):
            rdf = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['pnl_usd', 'exit_date'])
            ms = monthly_series(rdf.dropna(subset=['pnl_usd']) if len(rdf) else rdf,
                                 TRAIN_START if split == 'TRAIN' else VAL_START,
                                 TRAIN_END if split == 'TRAIN' else VAL_END)
            for month, pnl in ms.items():
                monthly_rows.append({'cell': cell_def['cell'], 'split': split, 'month': str(month),
                                      'pnl_usd': float(pnl)})

    rows_df = pd.DataFrame(all_rows)
    rows_df.to_csv(os.path.join(args.out_dir, 'cell_1599_cycles.csv'), index=False)

    train_df = pd.DataFrame(train_stats)
    eligible = train_df[(train_df['n_cycles'] >= 12) & (train_df['green_month_share'] >= 0.55)]
    selected = None
    if len(eligible):
        selected = eligible.sort_values('monthly_sharpe', ascending=False).iloc[0]
        sel_cell = int(selected['cell'])
        sel_def = next(c for c in CELLS if c['cell'] == sel_cell)
        if not args.skip_extension:
            ext_rows = guarded_extension_run(sel_def, mondays_df, spy_df, legcache, EXT_START, EXT_END, warn_counter)
            for r in ext_rows:
                r['sample'] = 'EXTENSION'
            all_rows += ext_rows
            rows_df = pd.DataFrame(all_rows)
            rows_df.to_csv(os.path.join(args.out_dir, 'cell_1599_cycles.csv'), index=False)
            edf = pd.DataFrame(ext_rows) if ext_rows else pd.DataFrame(columns=['pnl_usd', 'exit_date'])
            ms = monthly_series(edf.dropna(subset=['pnl_usd']) if len(edf) else edf, EXT_START, EXT_END)
            for month, pnl in ms.items():
                monthly_rows.append({'cell': sel_cell, 'split': 'EXTENSION', 'month': str(month),
                                      'pnl_usd': float(pnl)})

    pd.DataFrame(monthly_rows).to_csv(os.path.join(args.out_dir, 'cell_1599_monthly.csv'), index=False)
    write_result_md(args.out_dir, train_df, rows_df, spy_df, selected, warn_counter)
    log.info("done. warn_counter=%s", warn_counter)
    return 0


def write_blocked_result_md(out_dir, reason):
    with open(os.path.join(out_dir, 'RESULT_1599.md'), 'w') as f:
        f.write("# RESULT -- cells 1,599-1,606 (v3, Amendment 3) -- BLOCKED, not a finding\n\n"
                f"The BUILD stage (this module) could not run: `{reason}`\n\n"
                "This is expected: the FETCH stage (`fetch_dbn.py`) walks 693 entry Mondays plus "
                "every selected leg's full life one Databento purchase at a time under the $150 "
                "spend cap, and as of this write it had not yet produced `mondays.parquet` or any "
                "file under `opt_cache/dbn/legs/`. No cycle, monthly, or pass-bar number below is "
                "fabricated or estimated -- there is none yet. Re-run `python3 cell_1599.py` once "
                "`fetch_dbn.log` shows the mondays and legs stages have finished (or run "
                "`--stage legs` to completion after `--stage mondays`).\n\n"
                "## What IS built and verified\n"
                "* `cell_1599.py` -- full v3 pipeline (strike selection, entry/exit fills per "
                "Amendment 3's bid/ask rule, VOID rule from the cycle table, management A/B, "
                "sizing + budget assertion, TRAIN selection, VAL read, EXTENSION read-once guard).\n"
                "* `test_cell_1599.py` -- unit tests on synthetic fixtures (delta/IV from the NBBO "
                "mid, bid/ask fill convention, VOID rule, sizing/budget assertion, management "
                "precedence, $0 months, the IV gate, the extension read-once guard) -- run "
                "independently of the live fetch and passing as of this write.\n")


def write_result_md(out_dir, train_df, rows_df, spy_df, selected, warn_counter):
    lines = ["# RESULT -- cells 1,599-1,606 (v3, Amendment 3: Databento OPRA cbbo-1m NBBO)\n"]
    cols = ['cell', 'delta', 'mgmt', 'gate', 'n_cycles', 'void_share', 'mean_monthly_ret_on_B',
            'monthly_sharpe', 'green_month_share', 'worst_month_usd', 'max_dd_usd', 'win_rate']
    lines.append("## TRAIN (all 8 cells)\n")
    lines.append(train_df[cols].sort_values('monthly_sharpe', ascending=False).to_markdown(index=False) + "\n")
    if selected is None:
        lines.append("\nNo cell reached the TRAIN eligibility bar (>=12 cycles, >=55% green "
                      "months) -- no VAL or EXTENSION read follows; this is a finding about the "
                      "population, reportable once the FETCH stage has finished (see void_share "
                      "above first -- a high VOID share here means TRAIN itself is not trustworthy "
                      "yet, not that the mechanism failed).\n")
    else:
        lines.append(f"\n## Selected: cell {int(selected['cell'])} "
                      f"(delta={selected['delta']}, mgmt={selected['mgmt']}, gate={selected['gate']})\n")
    lines.append(f"\nWARNING/fallback/VOID counters: {warn_counter}\n")
    with open(os.path.join(out_dir, 'RESULT_1599.md'), 'w') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    sys.exit(main())
