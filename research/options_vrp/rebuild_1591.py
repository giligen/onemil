"""Independent rebuild of PREREG_1567 cells 1,591-1,598 (Amendment 2 + 2a) from prose only.

Does NOT read cell_1591.py, test_cell_1591.py, cell_1591_cycles.csv, cell_1591_monthly.csv,
RESULT_1591.md, or the v1 cell_1567.py. Built from PREREG_1567.md + Amendment 2 + Amendment 2a
text and the shared caches under opt_cache/ (spy_daily.parquet, spy_minute.parquet,
option_daily.parquet, option_minute_entry.parquet, manifest.parquet) plus a NEW tick-trade fetch
into opt_cache/ticks_rebuild/ for the specific legs/dates this rebuild needs (Amendment 2a: OPRA
quotes are 404 on this data plan; price legs from tick trades).

Disclosed methodology / approximations (say-so clause, mirrors FETCH_1567.md's own disclosures):
  D1. Strike SELECTION (which strike is nearest target delta) uses the entry-minute BAR close
      nearest 10:00 ET (already cached in option_minute_entry.parquet, aggregated from trades in
      09:55-10:10) as the "10:00 price" input to the IV bisection -- not a fresh tick fetch across
      the whole strike grid every Monday (that would be ~20 strikes x 133 Mondays x 2 legs of tick
      fetching just to pick a strike, an order of magnitude more requests than pricing the two
      legs actually traded). The ACTUAL FILL price used for every dollar of P&L is a real tick
      trade (Amendment 2a), fetched fresh into ticks_rebuild/ for exactly the legs selected.
  D2. The IV gate (45-DTE ATM implied vol >= 15% at 10:00) uses the same entry-minute bar close
      for the same reason -- it is a go/no-go filter on the week, not a fill price.
  D3. Management-A daily marks (to find the trigger day for 50%-credit / 2x-stop / 21-DTE) use
      the cached option DAILY closes (Amendment 2a: "daily marks from the cached option daily
      closes"); only the ACTUAL EXIT tick trade (next session, 10:00:00-10:00:30) is freshly
      fetched, for the specific (symbol, date) pairs the daily-close trigger logic identifies.
  D4. Monthly P&L is realized-basis: a cycle's full P&L is booked to the calendar month of its
      EXIT (management-A close, or expiry under B). Calendar months with zero exits in the split's
      own [min exit month, max exit month] range are included at $0 (Amendment 2a).
  D5. Regulatory fees: $0.03/contract charged once at entry and once at exit (two transactions),
      per PREREG "regulatory fees $0.03/contract, commission $0".
"""
import argparse
import datetime as dt
import logging
import os
import sqlite3
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

CACHE_DIR = os.path.join(HERE, 'opt_cache')
TICK_DIR = os.path.join(CACHE_DIR, 'ticks_rebuild')
os.makedirs(TICK_DIR, exist_ok=True)
TICK_DB = os.path.join(TICK_DIR, 'ticks_rebuild.db')

ET = ZoneInfo('America/New_York')
UTC = dt.timezone.utc

R = 0.045
Q = 0.013
W = 10.0
B = 0.10 * 65000.0          # risk budget, $6,500
SLOT = B / 6.0               # per-slot budget
FEE_PER_CONTRACT = 0.03
DELTAS = [0.20, 0.30]
MGMTS = ['A', 'B']
GATES = ['none', 'ivgate']
TRAIN_START, TRAIN_END = '2024-02-05', '2025-06-30'
VAL_START, VAL_END = '2025-07-07', '2026-08-17'
BATCH = 15
PAUSE_S = 0.35

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('rebuild_1591')


# --------------------------------------------------------------------------- BS math

def bs_put_price(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_put_delta(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return -1.0 if K > S else 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return -np.exp(-q * T) * norm.cdf(-d1)


def implied_vol_put(price, S, K, T, r, q):
    """Bisect (Brent) for sigma given an observed put price. None if unsolvable (price outside
    the no-arbitrage band, or degenerate T)."""
    if T <= 0 or price <= 0:
        return None
    intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    upper_bound = K * np.exp(-r * T)
    if price <= intrinsic or price >= upper_bound:
        return None
    try:
        f = lambda sig: bs_put_price(S, K, T, r, q, sig) - price
        return brentq(f, 1e-4, 6.0, xtol=1e-6, maxiter=100)
    except Exception:
        return None


def delta_from_price(price, S, K, T, r, q):
    """|delta| of a put given its observed 10:00 price, via IV bisection. None if unsolvable."""
    iv = implied_vol_put(price, S, K, T, r, q)
    if iv is None:
        return None
    return abs(bs_put_delta(S, K, T, r, q, iv))


# --------------------------------------------------------------------------- cache loading

def load_caches():
    log.info('loading opt_cache parquet files...')
    spy_daily = pd.read_parquet(os.path.join(CACHE_DIR, 'spy_daily.parquet'))
    spy_minute = pd.read_parquet(os.path.join(CACHE_DIR, 'spy_minute.parquet'))
    opt_daily = pd.read_parquet(os.path.join(CACHE_DIR, 'option_daily.parquet'))
    opt_entry = pd.read_parquet(os.path.join(CACHE_DIR, 'option_minute_entry.parquet'))
    manifest = pd.read_parquet(os.path.join(CACHE_DIR, 'manifest.parquet'))
    spy_daily['day'] = pd.to_datetime(spy_daily['day'])
    opt_entry['t'] = pd.to_datetime(opt_entry['t'], utc=True)
    spy_minute['t'] = pd.to_datetime(spy_minute['t'], utc=True)
    opt_daily['day'] = pd.to_datetime(opt_daily['day'])
    manifest_expiry = dict(zip(manifest['symbol'], pd.to_datetime(manifest['expiry'])))
    manifest_strike = dict(zip(manifest['symbol'], manifest['strike']))
    log.info('caches loaded: spy_daily=%d spy_minute=%d opt_daily=%d opt_entry=%d manifest=%d',
              len(spy_daily), len(spy_minute), len(opt_daily), len(opt_entry), len(manifest))
    return spy_daily, spy_minute, opt_daily, opt_entry, manifest_expiry, manifest_strike


def spy_price_near_10(spy_minute, day_str):
    d0 = dt.date.fromisoformat(day_str)
    target_et = dt.datetime(d0.year, d0.month, d0.day, 10, 0, tzinfo=ET)
    target_utc = pd.Timestamp(target_et.astimezone(UTC))
    day_lo = pd.Timestamp(dt.datetime(d0.year, d0.month, d0.day, 0, 0, tzinfo=ET).astimezone(UTC))
    day_hi = day_lo + pd.Timedelta(days=1)
    rows = spy_minute[(spy_minute['t'] >= day_lo) & (spy_minute['t'] < day_hi)]
    if rows.empty:
        return None
    idx = (rows['t'] - target_utc).abs().idxmin()
    return float(rows.loc[idx, 'c'])


def spy_price_near_16(spy_minute, spy_daily, day_str):
    d0 = dt.date.fromisoformat(day_str)
    target_et = dt.datetime(d0.year, d0.month, d0.day, 15, 59, tzinfo=ET)
    target_utc = pd.Timestamp(target_et.astimezone(UTC))
    day_lo = pd.Timestamp(dt.datetime(d0.year, d0.month, d0.day, 0, 0, tzinfo=ET).astimezone(UTC))
    day_hi = day_lo + pd.Timedelta(days=1)
    rows = spy_minute[(spy_minute['t'] >= day_lo) & (spy_minute['t'] < day_hi)]
    if not rows.empty:
        idx = (rows['t'] - target_utc).abs().idxmin()
        return float(rows.loc[idx, 'c'])
    dd = spy_daily[spy_daily['day'] == pd.Timestamp(day_str)]
    if not dd.empty:
        return float(dd.iloc[0]['c'])
    return None


def entry_price_near_10(opt_entry_day, symbol):
    rows = opt_entry_day[opt_entry_day['symbol'] == symbol]
    if rows.empty:
        return None
    d0 = rows['monday'].iloc[0]
    y, m, d = [int(x) for x in d0.split('-')]
    target_et = dt.datetime(y, m, d, 10, 0, tzinfo=ET)
    target_utc = pd.Timestamp(target_et.astimezone(UTC))
    idx = (rows['t'] - target_utc).abs().idxmin()
    return float(rows.loc[idx, 'c'])


# --------------------------------------------------------------------------- selection phase

def select_legs(spy_minute, opt_entry, manifest_expiry, manifest_strike):
    """For every entry Monday in the cache, pick the nearest-45-DTE expiry, then for each target
    delta pick the strike whose BS |delta| (IV from the entry-minute bar close nearest 10:00) is
    closest, and the long leg W=$10 below it. Also computes the ATM-45DTE gate IV.
    Returns a DataFrame: monday, expiry, dte, delta_target, short_symbol, short_strike,
    long_symbol, long_strike, short_price10, long_price10, gate_iv (or NaN)."""
    mondays = sorted(opt_entry['monday'].unique())
    rows = []
    for monday in mondays:
        spot = spy_price_near_10(spy_minute, monday)
        if spot is None:
            log.warning('select_legs: no SPY 10:00 price for monday=%s, skipping', monday)
            continue
        day_entries = opt_entry[opt_entry['monday'] == monday].copy()
        day_entries['expiry'] = day_entries['symbol'].map(manifest_expiry)
        day_entries['strike'] = day_entries['symbol'].map(manifest_strike)
        day_entries = day_entries.dropna(subset=['expiry', 'strike'])
        d0 = dt.date.fromisoformat(monday)
        day_entries['dte'] = (day_entries['expiry'] - pd.Timestamp(d0)).dt.days
        expiries = sorted(day_entries['expiry'].unique())
        if not expiries:
            log.warning('select_legs: no expiries with entry-minute data for monday=%s', monday)
            continue
        target_expiry = min(expiries, key=lambda e: abs((e - pd.Timestamp(d0)).days - 45))
        dte = (target_expiry - pd.Timestamp(d0)).days
        T = dte / 365.0
        chain = day_entries[day_entries['expiry'] == target_expiry].drop_duplicates('symbol')
        # per-strike "10:00 price" and delta
        priced = []
        for _, r_ in chain.iterrows():
            p10 = entry_price_near_10(chain, r_['symbol'])
            if p10 is None or p10 <= 0:
                continue
            delta = delta_from_price(p10, spot, r_['strike'], T, R, Q)
            priced.append((r_['symbol'], r_['strike'], p10, delta))
        priced_df = pd.DataFrame(priced, columns=['symbol', 'strike', 'price10', 'delta'])
        priced_df = priced_df.dropna(subset=['delta'])
        if priced_df.empty:
            log.warning('select_legs: no strike with a solvable delta for monday=%s expiry=%s',
                        monday, target_expiry.date())
            continue
        # IV gate: ATM strike (nearest spot) at a 45-DTE-ish expiry (use the same target expiry,
        # the nearest listed to 45 DTE, per Amendment: "45-DTE ATM implied vol at 10:00")
        atm_row = priced_df.iloc[(priced_df['strike'] - spot).abs().argsort()[:1]]
        atm_iv = None
        if not atm_row.empty:
            atm_iv = implied_vol_put(float(atm_row['price10'].iloc[0]), spot,
                                      float(atm_row['strike'].iloc[0]), T, R, Q)
        strikes_have_price = set(chain['strike'])
        for dtarget in DELTAS:
            cand = priced_df.copy()
            cand['derr'] = (cand['delta'] - dtarget).abs()
            cand = cand.sort_values('derr')
            short_row = None
            for _, cr in cand.iterrows():
                long_strike = cr['strike'] - W
                if long_strike in strikes_have_price:
                    short_row = cr
                    break
            if short_row is None:
                log.warning('select_legs: monday=%s delta=%.2f no strike has a priced long leg %s below it',
                            monday, dtarget, W)
                continue
            long_strike = short_row['strike'] - W
            long_sym_rows = chain[chain['strike'] == long_strike]
            if long_sym_rows.empty:
                continue
            long_symbol = long_sym_rows['symbol'].iloc[0]
            long_p10 = entry_price_near_10(chain, long_symbol)
            rows.append(dict(
                monday=monday, expiry=target_expiry.date().isoformat(), dte=dte,
                delta_target=dtarget, short_symbol=short_row['symbol'], short_strike=short_row['strike'],
                short_delta=short_row['delta'], long_symbol=long_symbol, long_strike=long_strike,
                short_price10=short_row['price10'], long_price10=long_p10, gate_iv=atm_iv))
    out = pd.DataFrame(rows)
    log.info('select_legs: %d (monday, delta_target) legs selected out of %d mondays', len(out), len(mondays))
    return out


# --------------------------------------------------------------------------- management-A trigger day

def trading_days_between(spy_daily, start_str, end_str):
    days = spy_daily[(spy_daily['day'] >= pd.Timestamp(start_str)) & (spy_daily['day'] <= pd.Timestamp(end_str))]
    return sorted(days['day'].dt.date.tolist())


def next_trading_day(spy_daily, day):
    days = sorted(spy_daily['day'].dt.date.tolist())
    for d in days:
        if d > day:
            return d
    return None


def find_mgmtA_trigger(opt_daily, spy_daily, short_sym, long_sym, monday_str, expiry_str, credit_proxy):
    """Returns (trigger_reason, trigger_day date, exit_day date) for management A, using cached
    option DAILY closes as the daily mark (Amendment 2a D3). exit_day = next trading session
    after trigger_day (conservative: the actual close happens at that NEXT session's 10:00)."""
    entry_d = dt.date.fromisoformat(monday_str)
    expiry_d = dt.date.fromisoformat(expiry_str)
    cutoff_21dte = expiry_d - dt.timedelta(days=21)
    short_c = opt_daily[opt_daily['symbol'] == short_sym].set_index('day')['c']
    long_c = opt_daily[opt_daily['symbol'] == long_sym].set_index('day')['c']
    days = trading_days_between(spy_daily, (entry_d + dt.timedelta(days=1)).isoformat(), expiry_d.isoformat())
    last_short, last_long = None, None
    for d in days:
        ts = pd.Timestamp(d)
        if ts in short_c.index:
            last_short = float(short_c.loc[ts])
        if ts in long_c.index:
            last_long = float(long_c.loc[ts])
        if last_short is None or last_long is None:
            continue
        spread_val = last_short - last_long
        if spread_val <= 0.5 * credit_proxy:
            reason = 'profit'
        elif spread_val >= 2.0 * credit_proxy:
            reason = 'stop'
        elif d >= cutoff_21dte:
            reason = '21dte'
        else:
            continue
        exit_day = next_trading_day(spy_daily, d)
        if exit_day is None or exit_day > expiry_d:
            exit_day = expiry_d
        return reason, d, exit_day
    # fallback: never triggered by daily marks (missing bars) -> mandatory 21-DTE close anyway
    exit_day = next_trading_day(spy_daily, cutoff_21dte) or expiry_d
    if exit_day > expiry_d:
        exit_day = expiry_d
    return '21dte_fallback', cutoff_21dte, exit_day


def build_fetch_plan(legs, opt_daily, spy_daily):
    """Every (date, symbol) pair whose real tick trade this rebuild needs: entry legs (both
    deltas, both management arms use the SAME entry fill) and management-A exit legs (computed
    per delta_target row, reused for the A cell of both gate settings)."""
    plan = {}  # date_str -> set(symbols)
    legs = legs.copy()
    legs['mgmtA_reason'] = None
    legs['mgmtA_trigger_day'] = None
    legs['mgmtA_exit_day'] = None
    for i, row in legs.iterrows():
        plan.setdefault(row['monday'], set()).update([row['short_symbol'], row['long_symbol']])
        credit_proxy = row['short_price10'] - row['long_price10']
        reason, trig, exitd = find_mgmtA_trigger(opt_daily, spy_daily, row['short_symbol'], row['long_symbol'],
                                                   row['monday'], row['expiry'], credit_proxy)
        legs.at[i, 'mgmtA_reason'] = reason
        legs.at[i, 'mgmtA_trigger_day'] = trig.isoformat()
        legs.at[i, 'mgmtA_exit_day'] = exitd.isoformat()
        plan.setdefault(exitd.isoformat(), set()).update([row['short_symbol'], row['long_symbol']])
    log.info('build_fetch_plan: %d distinct dates need tick trades', len(plan))
    return legs, plan


# --------------------------------------------------------------------------- tick fetch (Amendment 2a)

def init_tick_db(con):
    con.executescript('''
        CREATE TABLE IF NOT EXISTS trades (symbol TEXT, day TEXT, t TEXT, price REAL, size REAL,
            PRIMARY KEY(symbol, day, t));
        CREATE TABLE IF NOT EXISTS fetch_log (symbol TEXT, day TEXT, status TEXT, n_trades INTEGER,
            fetched_at TEXT, PRIMARY KEY(symbol, day));
    ''')
    con.commit()


def et_window_utc(day, sh, sm, ss, eh, em, es):
    s = dt.datetime(day.year, day.month, day.day, sh, sm, ss, tzinfo=ET)
    e = dt.datetime(day.year, day.month, day.day, eh, em, es, tzinfo=ET)
    return s.astimezone(UTC), e.astimezone(UTC)


def fetch_ticks(plan, max_retries=3):
    """Fetch tick trades for every (date, symbol) in `plan` into TICK_DB, resumable. Window per
    Amendment 2a: 10:00:00-10:00:30 primary, extended to 10:00:00-10:05:00 if no trade -- we
    always fetch the full 10:00:00-10:05:00 window in one request (cheaper) and pick the
    scoring-time window logic from the cached trades."""
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionTradesRequest
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, '.env'))
    key = os.environ.get('ALPACA_API_KEY')
    secret = os.environ.get('ALPACA_API_SECRET')
    if not key or not secret:
        log.error('fetch_ticks: ALPACA_API_KEY/ALPACA_API_SECRET missing -- cannot fetch ticks')
        raise SystemExit(1)
    client = OptionHistoricalDataClient(key, secret)
    con = sqlite3.connect(TICK_DB)
    init_tick_db(con)
    dates = sorted(plan.keys())
    n_ok = n_absent = n_err = n_skip = 0
    total_pairs = sum(len(v) for v in plan.values())
    log.info('fetch_ticks: %d dates, %d (date,symbol) pairs total', len(dates), total_pairs)
    done_pairs = 0
    for date_str in dates:
        day = dt.date.fromisoformat(date_str)
        symbols = sorted(plan[date_str])
        todo = [s for s in symbols if not con.execute(
            'SELECT 1 FROM fetch_log WHERE symbol=? AND day=?', (s, date_str)).fetchone()]
        done_pairs += len(symbols) - len(todo)
        if not todo:
            n_skip += len(symbols)
            continue
        s_utc, e_utc = et_window_utc(day, 10, 0, 0, 10, 5, 0)
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            req = OptionTradesRequest(symbol_or_symbols=chunk, start=s_utc, end=e_utc, limit=1000)
            data = None
            for attempt in range(1, max_retries + 1):
                try:
                    resp = client.get_option_trades(req)
                    data = resp.data if hasattr(resp, 'data') else resp
                    break
                except Exception as e:
                    log.warning('fetch_ticks %s batch@%d attempt %d/%d failed: %s', date_str, i, attempt,
                                max_retries, e)
                    time.sleep(1.5 * attempt)
            now = dt.datetime.utcnow().isoformat()
            if data is None:
                for s in chunk:
                    con.execute('INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,?)',
                                (s, date_str, 'error', 0, now))
                    n_err += 1
            else:
                for s in chunk:
                    trades = data.get(s, []) if isinstance(data, dict) else []
                    if trades:
                        for tr in trades:
                            ts = tr.timestamp if tr.timestamp.tzinfo else tr.timestamp.replace(tzinfo=UTC)
                            con.execute('INSERT OR IGNORE INTO trades VALUES (?,?,?,?,?)',
                                        (s, date_str, ts.astimezone(UTC).isoformat(), float(tr.price),
                                         float(tr.size)))
                        con.execute('INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,?)',
                                    (s, date_str, 'ok', len(trades), now))
                        n_ok += 1
                    else:
                        con.execute('INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,?)',
                                    (s, date_str, 'absent', 0, now))
                        n_absent += 1
            con.commit()
            done_pairs += len(chunk)
            time.sleep(PAUSE_S)
        pct_err = 100.0 * n_err / max(n_ok + n_absent + n_err, 1)
        log.info('fetch_ticks progress: date=%s cumulative=%d/%d ok=%d absent=%d err=%d skip=%d err%%=%.2f',
                  date_str, done_pairs, total_pairs, n_ok, n_absent, n_err, n_skip, pct_err)
        if pct_err > 3.0 and (n_ok + n_absent + n_err) > 50:
            log.error('fetch_ticks COMPLETENESS GATE: err%% %.2f > 3.0 at date=%s -- STOPPING', pct_err, date_str)
            break
    log.info('fetch_ticks DONE: ok=%d absent=%d err=%d skip=%d (LOST=%d)', n_ok, n_absent, n_err, n_skip,
              n_err)
    con.close()


# --------------------------------------------------------------------------- scoring

def last_trade_in_window(con, symbol, date_str, sh, sm, ss, eh, em, es):
    day = dt.date.fromisoformat(date_str)
    s_utc, e_utc = et_window_utc(day, sh, sm, ss, eh, em, es)
    rows = con.execute('SELECT t, price FROM trades WHERE symbol=? AND day=? ORDER BY t', (symbol, date_str)).fetchall()
    in_win = [(pd.Timestamp(t), p) for t, p in rows if s_utc <= pd.Timestamp(t).to_pydatetime() <= e_utc]
    if not in_win:
        return None
    return in_win[-1][1]


def fill_entry(con, symbol, date_str, side):
    """side: 'sell' (short leg) or 'buy' (long leg). Amendment 2a: last trade 10:00:00-10:00:30,
    fallback 10:00:00-10:05:00; VOID (None) if neither has a trade."""
    p = last_trade_in_window(con, symbol, date_str, 10, 0, 0, 10, 0, 30)
    if p is None:
        p = last_trade_in_window(con, symbol, date_str, 10, 0, 0, 10, 5, 0)
    if p is None:
        return None
    return p - 0.03 if side == 'sell' else p + 0.03


def fill_exit_A(con, symbol, date_str, side, opt_daily):
    """Exit fill for management A: last trade 10:00:00-10:00:30 of the exit session; fallback =
    the daily OPEN of that session (Amendment 2a: 'exit at the daily open as the fallback,
    counted'). side: 'buy' (buying back the short) or 'sell' (selling the long)."""
    p = last_trade_in_window(con, symbol, date_str, 10, 0, 0, 10, 0, 30)
    if p is None:
        row = opt_daily[(opt_daily['symbol'] == symbol) & (opt_daily['day'] == pd.Timestamp(date_str))]
        if not row.empty:
            p = float(row.iloc[0]['o'])
    if p is None:
        return None
    return p + 0.03 if side == 'buy' else p - 0.03


def score_cell(legs, con, opt_daily, spy_minute, spy_daily, delta_target, mgmt, gate):
    """Build the cycle table for one of the 8 cells and return (cycles_df,)."""
    sub = legs[legs['delta_target'] == delta_target].copy()
    if gate == 'ivgate':
        sub = sub[sub['gate_iv'] >= 0.15]
    rows = []
    for _, r in sub.iterrows():
        fill_short_in = fill_entry(con, r['short_symbol'], r['monday'], 'sell')
        fill_long_in = fill_entry(con, r['long_symbol'], r['monday'], 'buy')
        void = fill_short_in is None or fill_long_in is None
        credit = None if void else (fill_short_in - fill_long_in)
        contracts = 0
        if not void and credit is not None and (W - credit) > 0:
            contracts = int(np.floor((SLOT) / ((W - credit) * 100)))
        skip = (contracts == 0)
        pnl = None
        exit_day = None
        exit_reason = None
        if void:
            exit_day = r['mgmtA_exit_day'] if mgmt == 'A' else r['expiry']
            exit_reason = 'VOID_ENTRY'
        elif skip:
            exit_reason = 'SKIP_ZERO_CONTRACTS'
        else:
            if mgmt == 'A':
                exit_day = r['mgmtA_exit_day']
                exit_reason = r['mgmtA_reason']
                fill_short_out = fill_exit_A(con, r['short_symbol'], exit_day, 'buy', opt_daily)
                fill_long_out = fill_exit_A(con, r['long_symbol'], exit_day, 'sell', opt_daily)
                if fill_short_out is None or fill_long_out is None:
                    void = True
                    exit_reason = 'VOID_EXIT'
                else:
                    exit_debit = fill_short_out - fill_long_out
                    pnl = (credit - exit_debit) * 100 * contracts - FEE_PER_CONTRACT * contracts * 2
            else:  # mgmt == 'B': hold to expiry, intrinsic settlement
                exit_day = r['expiry']
                exit_reason = 'EXPIRY_INTRINSIC'
                s_exp = spy_price_near_16(spy_minute, spy_daily, r['expiry'])
                if s_exp is None:
                    void = True
                    exit_reason = 'VOID_NO_SPY_AT_EXPIRY'
                else:
                    intrinsic_diff = max(r['short_strike'] - s_exp, 0.0) - max(r['long_strike'] - s_exp, 0.0)
                    pnl = (credit - intrinsic_diff) * 100 * contracts - FEE_PER_CONTRACT * contracts * 2
        rows.append(dict(monday=r['monday'], expiry=r['expiry'], delta_target=delta_target, mgmt=mgmt,
                          gate=gate, short_symbol=r['short_symbol'], long_symbol=r['long_symbol'],
                          credit=credit, contracts=contracts, void=void, skip=skip,
                          exit_day=exit_day, exit_reason=exit_reason, pnl=pnl))
    return pd.DataFrame(rows)


def monthly_series(cycles, start_str, end_str):
    """Calendar-month P&L series for the cycles whose MONDAY (entry) falls in [start,end], booked
    to the month of each cycle's own exit_day, zero-filled across the split's own exit-month span
    (Amendment 2a D4)."""
    c = cycles[(cycles['monday'] >= start_str) & (cycles['monday'] <= end_str)].copy()
    realized = c[c['pnl'].notna()]
    if realized.empty:
        return pd.Series(dtype=float), c
    realized = realized.copy()
    realized['exit_month'] = pd.to_datetime(realized['exit_day']).dt.to_period('M')
    monthly = realized.groupby('exit_month')['pnl'].sum()
    full_range = pd.period_range(monthly.index.min(), monthly.index.max(), freq='M')
    monthly = monthly.reindex(full_range, fill_value=0.0)
    return monthly, c


def cell_stats(monthly, c, split_name):
    n_cycles = len(c)
    n_void = int(c['void'].sum())
    n_skip = int((c['skip'] & ~c['void']).sum())
    n_realized = int(c['pnl'].notna().sum())
    void_share = n_void / n_cycles if n_cycles else np.nan
    ret = monthly / B
    mean_ret = ret.mean() if len(ret) else np.nan
    sharpe = (ret.mean() / ret.std(ddof=1) * np.sqrt(12)) if len(ret) > 1 and ret.std(ddof=1) > 0 else np.nan
    green_share = (monthly > 0).mean() if len(monthly) else np.nan
    worst_month = monthly.min() if len(monthly) else np.nan
    cum = monthly.cumsum()
    max_dd = (cum.cummax() - cum).max() if len(cum) else np.nan
    top5 = max(1, int(np.ceil(0.05 * len(monthly))))
    ex_top5_mean = monthly.sort_values()[:-top5].mean() if len(monthly) > top5 else np.nan
    return dict(split=split_name, n_cycles=n_cycles, n_void=n_void, n_skip=n_skip, n_realized=n_realized,
                void_share=void_share, n_months=len(monthly), mean_monthly_ret_on_B=mean_ret,
                monthly_sharpe=sharpe, green_month_share=green_share, worst_month=worst_month,
                max_drawdown=max_dd, ex_top5_months_mean=ex_top5_mean,
                worst_month_ge_negB=bool(worst_month >= -B) if pd.notna(worst_month) else None)


# --------------------------------------------------------------------------- main

LEGS_PATH = os.path.join(HERE, 'rebuild_1591_legs.parquet')


def cmd_plan():
    spy_daily, spy_minute, opt_daily, opt_entry, manifest_expiry, manifest_strike = load_caches()
    legs = select_legs(spy_minute, opt_entry, manifest_expiry, manifest_strike)
    legs, plan = build_fetch_plan(legs, opt_daily, spy_daily)
    legs.to_parquet(LEGS_PATH)
    import json
    with open(os.path.join(HERE, 'rebuild_1591_fetch_plan.json'), 'w') as f:
        json.dump({k: sorted(v) for k, v in plan.items()}, f)
    log.info('cmd_plan: wrote %s (%d legs) and fetch_plan.json (%d dates)', LEGS_PATH, len(legs), len(plan))


def cmd_fetch():
    import json
    with open(os.path.join(HERE, 'rebuild_1591_fetch_plan.json')) as f:
        plan = {k: set(v) for k, v in json.load(f).items()}
    fetch_ticks(plan)


def cmd_score():
    legs = pd.read_parquet(LEGS_PATH)
    spy_daily, spy_minute, opt_daily, opt_entry, manifest_expiry, manifest_strike = load_caches()
    con = sqlite3.connect(TICK_DB)
    init_tick_db(con)

    all_cycles = []
    cell_id = {}
    n = 1591
    train_rows, val_rows = [], []
    cells_meta = []
    for dtarget in DELTAS:
        for mgmt in MGMTS:
            for gate in GATES:
                cyc = score_cell(legs, con, opt_daily, spy_minute, spy_daily, dtarget, mgmt, gate)
                cyc['cell'] = n
                all_cycles.append(cyc)
                monthly_tr, c_tr = monthly_series(cyc, TRAIN_START, TRAIN_END)
                monthly_val, c_val = monthly_series(cyc, VAL_START, VAL_END)
                st_tr = cell_stats(monthly_tr, c_tr, 'TRAIN')
                st_val = cell_stats(monthly_val, c_val, 'VAL')
                st_tr.update(cell=n, delta_target=dtarget, mgmt=mgmt, gate=gate)
                st_val.update(cell=n, delta_target=dtarget, mgmt=mgmt, gate=gate)
                train_rows.append(st_tr)
                val_rows.append(st_val)
                cells_meta.append((n, dtarget, mgmt, gate))
                n += 1
    cycles_df = pd.concat(all_cycles, ignore_index=True)
    cycles_df.to_csv(os.path.join(HERE, 'rebuild_1591_cycles.csv'), index=False)
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    # TRAIN selection: highest monthly Sharpe s.t. n_cycles>=12 and green_month_share>=0.55
    eligible = train_df[(train_df['n_cycles'] >= 12) & (train_df['green_month_share'] >= 0.55)]
    if eligible.empty:
        log.error('TRAIN selection: NO cell clears n_cycles>=12 and green_month_share>=0.55')
        selected = None
    else:
        selected = eligible.sort_values('monthly_sharpe', ascending=False).iloc[0]
    log.info('TRAIN table:\n%s', train_df[['cell', 'delta_target', 'mgmt', 'gate', 'n_cycles',
              'green_month_share', 'monthly_sharpe', 'mean_monthly_ret_on_B']].to_string())
    log.info('VAL table:\n%s', val_df[['cell', 'delta_target', 'mgmt', 'gate', 'n_cycles',
              'green_month_share', 'monthly_sharpe', 'mean_monthly_ret_on_B', 'worst_month']].to_string())

    out = dict(train_table=train_df, val_table=val_df, selected=selected, cycles=cycles_df)
    import pickle
    with open(os.path.join(HERE, 'rebuild_1591_state.pkl'), 'wb') as f:
        pickle.dump(out, f)
    log.info('cmd_score DONE. selected=%s', None if selected is None else int(selected['cell']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['plan', 'fetch', 'score'])
    args = ap.parse_args()
    if args.mode == 'plan':
        cmd_plan()
    elif args.mode == 'fetch':
        cmd_fetch()
    elif args.mode == 'score':
        cmd_score()
