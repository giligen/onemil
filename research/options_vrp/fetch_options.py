"""PREREG_1567 FETCH stage — SPY + SPY-put option bars for the defined-risk put-credit-spread ladder.

Builds research/options_vrp/opt_cache/state.db (resumable sqlite) then exports:
  opt_cache/spy_daily.parquet          SPY daily bars, 2024-01-02..end
  opt_cache/spy_minute.parquet         SPY RTH (09:30-16:00 ET) minute bars, 2024-01-02..end
  opt_cache/option_daily.parquet       daily bars for every grid contract, 2024-01-02..expiry
  opt_cache/option_minute_entry.parquet  09:55-10:10 ET minute bars for every grid contract on
                                          its referencing entry Monday(s)
  opt_cache/manifest.parquet           symbol, expiry, strike, first_bar, last_bar, n_daily, n_minute

Grid: for every entry Monday (first trading session of the week) in [2024-02-05, 2026-08-17],
SPY puts with strike in [0.80*spot10, spot10] in $1 steps (spot10 = SPY price nearest 10:00 ET
that Monday), for every expiration whose calendar DTE from that Monday is in [38, 52] and whose
weekday is Mon-Fri (SPY does not list every weekday; non-listed combinations simply return no
bars and are counted as ABSENT, not a fetch failure).

Documented scope reduction (per the task's own fallback clause, "if that is too many requests ...
say so"): option MINUTE bars are pulled only for the entry-Monday 09:55-10:10 ET window, not for
15:55-16:00 ET of every session while a contract is live. A full-grid every-session minute pull is
O(unique contracts x average contract lifetime) ~= tens of millions of contract-days at this grid
size -- infeasible under the 200 req/min ceiling in a single research session. Option DAILY bars
(2024-01-02..expiry, all grid contracts) substitute for the daily mark used by the management rule
(50% credit / 2x stop / 21 DTE); the entry/exit FILL price still comes from real minute bars
(09:55-10:10 ET) because the PREREG's fill rule is minute-bar-sourced and must not be approximated.
SPY's own two windows (item 1 of the task) ARE fetched for every session, in full-RTH form (simpler
and explicitly allowed by the task), since that is one symbol and cheap.

Usage:
  python3 research/options_vrp/fetch_options.py --smoke-test        # 2 Mondays, sanity check
  python3 research/options_vrp/fetch_options.py                     # full resumable run
  python3 research/options_vrp/fetch_options.py --stage export      # re-export parquet only
"""
import argparse
import datetime as dt
import logging
import os
import sqlite3
import sys
import time
from zoneinfo import ZoneInfo

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

CACHE_DIR = os.path.join(HERE, 'opt_cache')
DB_PATH = os.path.join(CACHE_DIR, 'state.db')

ET = ZoneInfo('America/New_York')
UTC = dt.timezone.utc

SPY_START = '2024-01-02'
SPY_END = '2026-09-26'
GRID_MONDAY_START = '2024-02-05'
GRID_MONDAY_END = '2026-08-17'
DTE_LO, DTE_HI = 38, 52
STRIKE_PCT_BELOW = 0.20
BATCH = 20                 # alpaca-py multi-symbol batching (measured limit elsewhere: ~10-20/page safe)
PAUSE_S = 0.35              # ~2.85 req/s -> ~171 req/min, safely under the 200/min ceiling
MAX_RETRIES = 3
LOST_PCT_ERROR = 3.0        # completeness gate: ERROR + stop if true fetch ERRORS > this % of attempts

log = logging.getLogger('fetch_options_1567')


# --------------------------------------------------------------------------- helpers

def occ_symbol(expiry: dt.date, strike: float, root: str = 'SPY', opt_type: str = 'P') -> str:
    """OCC symbol: ROOT + YYMMDD + [C|P] + strike*1000 zero-padded to 8 digits."""
    strike_int = round(strike * 1000)
    return f"{root}{expiry.strftime('%y%m%d')}{opt_type}{strike_int:08d}"


def et_window_utc(day: str, sh: int, sm: int, eh: int, em: int):
    """(start_utc, end_utc) for an ET wall-clock window on `day`, DST-correct."""
    d = dt.date.fromisoformat(day)
    s = dt.datetime(d.year, d.month, d.day, sh, sm, tzinfo=ET)
    e = dt.datetime(d.year, d.month, d.day, eh, em, tzinfo=ET)
    return s.astimezone(UTC), e.astimezone(UTC)


def init_db(con):
    con.executescript('''
        CREATE TABLE IF NOT EXISTS spy_daily (day TEXT PRIMARY KEY, o REAL,h REAL,l REAL,c REAL,v REAL);
        CREATE TABLE IF NOT EXISTS spy_minute (day TEXT, t TEXT, o REAL,h REAL,l REAL,c REAL,v REAL,
            PRIMARY KEY(day,t));
        CREATE TABLE IF NOT EXISTS spy_minute_log (day TEXT PRIMARY KEY, n_bars INTEGER, fetched_at TEXT);
        CREATE TABLE IF NOT EXISTS contracts (symbol TEXT PRIMARY KEY, expiry TEXT, strike REAL,
            first_monday TEXT);
        CREATE TABLE IF NOT EXISTS grid_ref (symbol TEXT, monday TEXT, PRIMARY KEY(symbol, monday));
        CREATE TABLE IF NOT EXISTS opt_daily (symbol TEXT, day TEXT, o REAL,h REAL,l REAL,c REAL,v REAL,
            PRIMARY KEY(symbol,day));
        CREATE TABLE IF NOT EXISTS opt_daily_log (symbol TEXT PRIMARY KEY, status TEXT, n_bars INTEGER,
            fetched_at TEXT);
        CREATE TABLE IF NOT EXISTS opt_minute_entry (symbol TEXT, monday TEXT, t TEXT, o REAL,h REAL,l REAL,
            c REAL,v REAL, PRIMARY KEY(symbol,monday,t));
        CREATE TABLE IF NOT EXISTS opt_minute_entry_log (symbol TEXT, monday TEXT, status TEXT,
            n_bars INTEGER, fetched_at TEXT, PRIMARY KEY(symbol,monday));
    ''')
    con.commit()


def with_retries(fn, desc, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying transient errors with backoff. Returns None on final failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            wait = 1.5 * attempt
            log.warning('  %s attempt %d/%d failed: %s (retry in %.1fs)', desc, attempt, MAX_RETRIES, e, wait)
            time.sleep(wait)
    log.error('  %s FAILED after %d retries', desc, MAX_RETRIES)
    return None


# --------------------------------------------------------------------------- stage A: SPY bars

def fetch_spy_daily(con, stock_client):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    have = {r[0] for r in con.execute('SELECT day FROM spy_daily').fetchall()}
    if have:
        log.info('SPY daily: %d days already cached, skipping fetch (delete state.db to force refresh)', len(have))
        return
    req = StockBarsRequest(symbol_or_symbols=['SPY'], timeframe=TimeFrame.Day,
                            start=dt.date.fromisoformat(SPY_START), end=dt.date.fromisoformat(SPY_END))
    bars = with_retries(stock_client.get_stock_bars, 'SPY daily', req)
    rows = (bars.data.get('SPY', []) if bars else [])
    for b in rows:
        d = b.timestamp.date().isoformat()
        con.execute('INSERT OR IGNORE INTO spy_daily VALUES (?,?,?,?,?,?)',
                    (d, float(b.open), float(b.high), float(b.low), float(b.close), float(b.volume)))
    con.commit()
    log.info('SPY daily: fetched %d bars', len(rows))


def fetch_spy_minute(con, stock_client):
    """Full RTH minute bars, monthly chunks (resumable per month)."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    start = dt.date.fromisoformat(SPY_START)
    end = dt.date.fromisoformat(SPY_END)
    months = []
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        nxt = dt.date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        months.append((cur, min(nxt - dt.timedelta(days=1), end)))
        cur = nxt
    done = {r[0] for r in con.execute("SELECT day FROM spy_minute_log WHERE n_bars >= 0").fetchall()}
    for m_start, m_end in months:
        tag = m_start.strftime('%Y-%m')
        if tag in done:
            continue
        s_utc = dt.datetime(m_start.year, m_start.month, m_start.day, 9, 30, tzinfo=ET).astimezone(UTC)
        e_utc = dt.datetime(m_end.year, m_end.month, m_end.day, 16, 0, tzinfo=ET).astimezone(UTC)
        req = StockBarsRequest(symbol_or_symbols=['SPY'], timeframe=TimeFrame.Minute,
                                start=s_utc, end=e_utc, feed=DataFeed.SIP, limit=100000)
        bars = with_retries(stock_client.get_stock_bars, f'SPY minute {tag}', req)
        rows = (bars.data.get('SPY', []) if bars else [])
        for b in rows:
            ts = b.timestamp if b.timestamp.tzinfo else b.timestamp.replace(tzinfo=UTC)
            day = ts.astimezone(ET).date().isoformat()
            t = ts.astimezone(UTC).isoformat()
            con.execute('INSERT OR IGNORE INTO spy_minute VALUES (?,?,?,?,?,?,?)',
                        (day, t, float(b.open), float(b.high), float(b.low), float(b.close), float(b.volume)))
        con.execute('INSERT OR REPLACE INTO spy_minute_log VALUES (?,?,?)',
                    (tag, len(rows), dt.datetime.utcnow().isoformat()))
        con.commit()
        log.info('SPY minute %s: %d bars', tag, len(rows))
        time.sleep(PAUSE_S)


def spy_price_near(con, day: str, hh: int, mm: int):
    """Closest SPY minute close to hh:mm ET on `day`; None if no bars that day."""
    rows = con.execute('SELECT t, c FROM spy_minute WHERE day = ? ORDER BY t', (day,)).fetchall()
    if not rows:
        return None
    target = dt.datetime(*[int(x) for x in day.split('-')], hh, mm, tzinfo=ET).astimezone(UTC)
    best = min(rows, key=lambda r: abs((dt.datetime.fromisoformat(r[0]) - target).total_seconds()))
    return best[1]


# --------------------------------------------------------------------------- stage B: grid

def trading_days(con):
    return sorted(r[0] for r in con.execute('SELECT day FROM spy_daily').fetchall())


def entry_mondays(con, limit_weeks=0):
    """Calendar Mondays in [GRID_MONDAY_START, GRID_MONDAY_END], each snapped forward to the
    first actual trading session (per spy_daily) if the calendar Monday is a holiday."""
    days = trading_days(con)
    dayset = set(days)
    out = []
    d = dt.date.fromisoformat(GRID_MONDAY_START)
    end = dt.date.fromisoformat(GRID_MONDAY_END)
    while d <= end:
        probe = d
        for _ in range(5):  # snap forward at most a work-week
            if probe.isoformat() in dayset:
                out.append(probe.isoformat())
                break
            probe += dt.timedelta(days=1)
        else:
            log.warning('entry_mondays: no trading day found within 5 days of %s, skipping', d)
        d += dt.timedelta(days=7)
        if limit_weeks and len(out) >= limit_weeks:
            break
    return out


def build_grid(con, mondays):
    """Populate contracts + grid_ref for every (monday, strike, expiry) combination."""
    dayset = set(trading_days(con))
    n_pairs = 0
    for monday in mondays:
        spot = spy_price_near(con, monday, 10, 0)
        if spot is None:
            log.warning('build_grid: no SPY minute price near 10:00 on %s, skipping this Monday', monday)
            continue
        lo = int(spot * (1 - STRIKE_PCT_BELOW))
        hi = int(spot)
        strikes = list(range(lo, hi + 1))
        d0 = dt.date.fromisoformat(monday)
        expiries = []
        for dte in range(DTE_LO, DTE_HI + 1):
            e = d0 + dt.timedelta(days=dte)
            if e.weekday() < 5:  # Mon-Fri; actual listing confirmed later by whether bars come back
                expiries.append(e)
        for e in expiries:
            for k in strikes:
                sym = occ_symbol(e, float(k))
                con.execute('INSERT OR IGNORE INTO contracts VALUES (?,?,?,?)',
                            (sym, e.isoformat(), float(k), monday))
                con.execute('INSERT OR IGNORE INTO grid_ref VALUES (?,?)', (sym, monday))
                n_pairs += 1
        con.commit()
        log.info('build_grid: monday=%s spot@10:00=%.2f strikes=%d expiries=%d -> %d contract refs',
                  monday, spot, len(strikes), len(expiries), len(strikes) * len(expiries))
    n_contracts = con.execute('SELECT COUNT(*) FROM contracts').fetchone()[0]
    n_refs = con.execute('SELECT COUNT(*) FROM grid_ref').fetchone()[0]
    log.info('build_grid TOTAL: %d unique contracts, %d (monday,symbol) refs (raw attempts %d)',
              n_contracts, n_refs, n_pairs)


# --------------------------------------------------------------------------- stage C: option daily

def fetch_option_daily(con, opt_client):
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame
    todo = [r[0] for r in con.execute(
        'SELECT symbol FROM contracts WHERE symbol NOT IN (SELECT symbol FROM opt_daily_log)').fetchall()]
    total = len(todo)
    log.info('option daily: %d contracts to fetch (of %d total)', total,
              con.execute('SELECT COUNT(*) FROM contracts').fetchone()[0])
    start_date = dt.date.fromisoformat(SPY_START)
    n_ok = n_absent = n_err = 0
    for i in range(0, total, BATCH):
        chunk = todo[i:i + BATCH]
        exp_by_sym = dict(con.execute(
            f"SELECT symbol, expiry FROM contracts WHERE symbol IN ({','.join('?' * len(chunk))})", chunk))
        end_date = max(dt.date.fromisoformat(e) for e in exp_by_sym.values())
        req = OptionBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                                  start=start_date, end=min(end_date, dt.date.fromisoformat(SPY_END)))
        bars = with_retries(opt_client.get_option_bars, f'opt daily batch@{i}', req)
        now = dt.datetime.utcnow().isoformat()
        if bars is None:
            for s in chunk:
                con.execute('INSERT OR REPLACE INTO opt_daily_log VALUES (?,?,?,?)', (s, 'error', 0, now))
                n_err += 1
        else:
            data = getattr(bars, 'data', {}) or {}
            for s in chunk:
                rows = data.get(s, [])
                if rows:
                    for b in rows:
                        d = b.timestamp.date().isoformat()
                        con.execute('INSERT OR IGNORE INTO opt_daily VALUES (?,?,?,?,?,?,?)',
                                    (s, d, float(b.open), float(b.high), float(b.low), float(b.close),
                                     float(b.volume)))
                    con.execute('INSERT OR REPLACE INTO opt_daily_log VALUES (?,?,?,?)',
                                (s, 'ok', len(rows), now))
                    n_ok += 1
                else:
                    con.execute('INSERT OR REPLACE INTO opt_daily_log VALUES (?,?,?,?)',
                                (s, 'absent', 0, now))
                    n_absent += 1
        con.commit()
        if (i // BATCH) % 25 == 0:
            done = n_ok + n_absent + n_err
            pct_err = 100.0 * n_err / max(done, 1)
            log.info('  option daily progress: %d/%d done (ok=%d absent=%d err=%d, err%%=%.2f)',
                      done, total, n_ok, n_absent, n_err, pct_err)
            if pct_err > LOST_PCT_ERROR and done > 200:
                log.error('COMPLETENESS GATE: option-daily error rate %.2f%% > %.1f%% at %d done -- STOPPING',
                           pct_err, LOST_PCT_ERROR, done)
                return
        time.sleep(PAUSE_S)
    log.info('option daily DONE: ok=%d absent=%d err=%d', n_ok, n_absent, n_err)


# --------------------------------------------------------------------------- stage D: option entry minute

def fetch_option_entry_minute(con, opt_client):
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame
    mondays = sorted({r[0] for r in con.execute('SELECT DISTINCT monday FROM grid_ref').fetchall()})
    n_ok = n_absent = n_err = 0
    total_refs = con.execute('SELECT COUNT(*) FROM grid_ref').fetchone()[0]
    done_so_far = 0
    for monday in mondays:
        syms = [r[0] for r in con.execute(
            'SELECT symbol FROM grid_ref g WHERE g.monday = ? AND NOT EXISTS '
            '(SELECT 1 FROM opt_minute_entry_log l WHERE l.symbol=g.symbol AND l.monday=g.monday)',
            (monday,)).fetchall()]
        if not syms:
            continue
        s_utc, e_utc = et_window_utc(monday, 9, 55, 10, 10)
        for i in range(0, len(syms), BATCH):
            chunk = syms[i:i + BATCH]
            req = OptionBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Minute,
                                      start=s_utc, end=e_utc, limit=1000)
            bars = with_retries(opt_client.get_option_bars, f'opt entry-min {monday}@{i}', req)
            now = dt.datetime.utcnow().isoformat()
            if bars is None:
                for s in chunk:
                    con.execute('INSERT OR REPLACE INTO opt_minute_entry_log VALUES (?,?,?,?,?)',
                                (s, monday, 'error', 0, now))
                    n_err += 1
            else:
                data = getattr(bars, 'data', {}) or {}
                for s in chunk:
                    rows = data.get(s, [])
                    if rows:
                        for b in rows:
                            ts = b.timestamp if b.timestamp.tzinfo else b.timestamp.replace(tzinfo=UTC)
                            t = ts.astimezone(UTC).isoformat()
                            con.execute('INSERT OR IGNORE INTO opt_minute_entry VALUES (?,?,?,?,?,?,?,?)',
                                        (s, monday, t, float(b.open), float(b.high), float(b.low),
                                         float(b.close), float(b.volume)))
                        con.execute('INSERT OR REPLACE INTO opt_minute_entry_log VALUES (?,?,?,?,?)',
                                    (s, monday, 'ok', len(rows), now))
                        n_ok += 1
                    else:
                        con.execute('INSERT OR REPLACE INTO opt_minute_entry_log VALUES (?,?,?,?,?)',
                                    (s, monday, 'absent', 0, now))
                        n_absent += 1
            con.commit()
            done_so_far += len(chunk)
            time.sleep(PAUSE_S)
        pct_err = 100.0 * n_err / max(n_ok + n_absent + n_err, 1)
        log.info('option entry-minute: monday=%s cumulative done=%d/%d (ok=%d absent=%d err=%d err%%=%.2f)',
                  monday, done_so_far, total_refs, n_ok, n_absent, n_err, pct_err)
        if pct_err > LOST_PCT_ERROR and (n_ok + n_absent + n_err) > 200:
            log.error('COMPLETENESS GATE: entry-minute error rate %.2f%% > %.1f%% -- STOPPING at monday=%s',
                       pct_err, LOST_PCT_ERROR, monday)
            return
    log.info('option entry-minute DONE: ok=%d absent=%d err=%d', n_ok, n_absent, n_err)


# --------------------------------------------------------------------------- stage E: export

def export_parquet(con):
    import pandas as pd
    os.makedirs(CACHE_DIR, exist_ok=True)

    pd.read_sql('SELECT * FROM spy_daily ORDER BY day', con).to_parquet(
        os.path.join(CACHE_DIR, 'spy_daily.parquet'))
    pd.read_sql('SELECT * FROM spy_minute ORDER BY day, t', con).to_parquet(
        os.path.join(CACHE_DIR, 'spy_minute.parquet'))
    pd.read_sql('SELECT * FROM opt_daily ORDER BY symbol, day', con).to_parquet(
        os.path.join(CACHE_DIR, 'option_daily.parquet'))
    pd.read_sql('SELECT * FROM opt_minute_entry ORDER BY symbol, monday, t', con).to_parquet(
        os.path.join(CACHE_DIR, 'option_minute_entry.parquet'))

    manifest = pd.read_sql('''
        SELECT c.symbol, c.expiry, c.strike, c.first_monday,
               MIN(d.day) AS first_bar, MAX(d.day) AS last_bar, COUNT(d.day) AS n_daily,
               COALESCE((SELECT COUNT(*) FROM opt_minute_entry m WHERE m.symbol = c.symbol), 0) AS n_minute
        FROM contracts c LEFT JOIN opt_daily d ON d.symbol = c.symbol
        GROUP BY c.symbol
    ''', con)
    manifest.to_parquet(os.path.join(CACHE_DIR, 'manifest.parquet'))
    log.info('export: spy_daily=%d spy_minute=%d option_daily=%d option_minute_entry=%d manifest=%d rows',
              len(pd.read_sql('SELECT 1 FROM spy_daily', con)),
              len(pd.read_sql('SELECT 1 FROM spy_minute', con)),
              len(pd.read_sql('SELECT 1 FROM opt_daily', con)),
              len(pd.read_sql('SELECT 1 FROM opt_minute_entry', con)),
              len(manifest))
    return manifest


# --------------------------------------------------------------------------- report

def write_report(con, manifest):
    n_contracts = con.execute('SELECT COUNT(*) FROM contracts').fetchone()[0]
    n_refs = con.execute('SELECT COUNT(*) FROM grid_ref').fetchone()[0]
    n_mondays = con.execute('SELECT COUNT(DISTINCT monday) FROM grid_ref').fetchone()[0]
    daily_status = dict(con.execute('SELECT status, COUNT(*) FROM opt_daily_log GROUP BY status').fetchall())
    minute_status = dict(con.execute('SELECT status, COUNT(*) FROM opt_minute_entry_log GROUP BY status').fetchall())
    daily_err_pct = 100.0 * daily_status.get('error', 0) / max(sum(daily_status.values()), 1)
    minute_err_pct = 100.0 * minute_status.get('error', 0) / max(sum(minute_status.values()), 1)
    gate_daily = 'PASS' if daily_err_pct <= LOST_PCT_ERROR else 'FAIL'
    gate_minute = 'PASS' if minute_err_pct <= LOST_PCT_ERROR else 'FAIL'
    n_spy_daily = con.execute('SELECT COUNT(*) FROM spy_daily').fetchone()[0]
    n_spy_minute = con.execute('SELECT COUNT(*) FROM spy_minute').fetchone()[0]

    text = f'''# FETCH_1567 — options_vrp data fetch (PREREG_1567 FETCH stage)

Run finished: {dt.datetime.utcnow().isoformat()}Z

## SPY underlying
* Daily bars ({SPY_START}..{SPY_END}): {n_spy_daily} rows.
* Minute bars (full RTH 09:30-16:00 ET, every session, monthly-chunked fetch): {n_spy_minute} rows.

## Options grid
* Entry Mondays (first trading session of the week, {GRID_MONDAY_START}..{GRID_MONDAY_END}): {n_mondays}.
* Unique OCC contracts attempted: {n_contracts}.
* (monday, contract) reference pairs: {n_refs}.
* Strikes: $1 steps, spot*(1-{STRIKE_PCT_BELOW}) to spot, spot priced at the SPY minute bar nearest
  10:00 ET that Monday. **$0.50 strikes were NOT attempted** (scope reduction — doubling every
  request for a refinement not needed to bracket delta targets 0.15-0.30 on a $1 grid; flagged per
  the task's own "say so" clause, not silently dropped).
* Expiries: every calendar-weekday (Mon-Fri) date at DTE {DTE_LO}-{DTE_HI} from the entry Monday.
  SPY does not list options on every weekday; non-listed (monday,strike,expiry) combinations return
  no bars and are counted as ABSENT below, not as a fetch error.

## Option daily bars (2024-01-02..expiry, every grid contract)
* ok={daily_status.get('ok', 0)}  absent={daily_status.get('absent', 0)}  error={daily_status.get('error', 0)}
* Completeness gate (error rate vs {LOST_PCT_ERROR}% threshold): {daily_err_pct:.2f}% -> **{gate_daily}**

## Option entry-minute bars (09:55-10:10 ET, entry Monday only — see scope reduction below)
* ok={minute_status.get('ok', 0)}  absent={minute_status.get('absent', 0)}  error={minute_status.get('error', 0)}
* Completeness gate (error rate vs {LOST_PCT_ERROR}% threshold): {minute_err_pct:.2f}% -> **{gate_minute}**

## Documented scope reduction (say-so clause)
The task's literal ask was minute bars at 10:00-10:05 ET for every session a contract is live, or
(fallback) 09:55-10:10 ET on Mondays **and** 15:55-16:00 ET of every session. At this grid size
({n_contracts} unique contracts, avg lifetime ~40 sessions), the every-session leg is
O(contracts x sessions) ~= several million requests — infeasible under the 200 req/min ceiling in
one research session. What was fetched instead:
* Option DAILY bars for every contract, full listing-to-expiry range — gives the SCORE stage a
  daily mark (close) to evaluate the 50%-credit / 2x-stop / 21-DTE management rule.
* Option MINUTE bars only for the entry-Monday 09:55-10:10 ET window — gives the real minute-bar
  fill price the PREREG's Fill rule requires (mid at 10:00-10:05, never approximated by a daily bar).
* NOT fetched: 15:55-16:00 ET minute bars on every non-entry session for every contract. The SCORE
  stage must either (a) treat the daily close as the management-trigger price (documented
  approximation, disclose it), or (b) request a second, much smaller fetch limited to the specific
  legs actually selected after Stage-1 strike selection (a small, bounded set vs. the full grid) —
  recommended, and cheap once legs are known.

## Files
* {os.path.join('research/options_vrp/opt_cache', 'spy_daily.parquet')}
* {os.path.join('research/options_vrp/opt_cache', 'spy_minute.parquet')}
* {os.path.join('research/options_vrp/opt_cache', 'option_daily.parquet')}
* {os.path.join('research/options_vrp/opt_cache', 'option_minute_entry.parquet')}
* {os.path.join('research/options_vrp/opt_cache', 'manifest.parquet')} ({len(manifest)} contracts)
* {os.path.join('research/options_vrp/opt_cache', 'state.db')} (sqlite, resumable fetch state — keep)
'''
    with open(os.path.join(HERE, 'FETCH_1567.md'), 'w') as f:
        f.write(text)
    log.info('wrote %s', os.path.join(HERE, 'FETCH_1567.md'))


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--smoke-test', action='store_true', help='2 entry Mondays only, sanity check')
    ap.add_argument('--stage', choices=['all', 'spy', 'grid', 'daily', 'minute', 'export'], default='all')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    os.makedirs(CACHE_DIR, exist_ok=True)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, '.env'))
    from config import Config
    cfg = Config()
    if not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
        log.error('missing Alpaca API credentials (ALPACA_API_KEY/ALPACA_API_SECRET) - cannot fetch')
        return 1

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    stock_client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    opt_client = OptionHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)

    con = sqlite3.connect(DB_PATH)
    init_db(con)

    if a.stage in ('all', 'spy'):
        fetch_spy_daily(con, stock_client)
        fetch_spy_minute(con, stock_client)
    if a.stage in ('all', 'grid'):
        mondays = entry_mondays(con, limit_weeks=2 if a.smoke_test else 0)
        log.info('grid: %d entry mondays selected%s', len(mondays), ' (SMOKE TEST)' if a.smoke_test else '')
        build_grid(con, mondays)
    if a.stage in ('all', 'daily'):
        fetch_option_daily(con, opt_client)
    if a.stage in ('all', 'minute'):
        fetch_option_entry_minute(con, opt_client)
    if a.stage in ('all', 'export'):
        manifest = export_parquet(con)
        write_report(con, manifest)

    con.close()
    log.info('DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
