#!/usr/bin/env python3
"""Forward ledger (FORWARD_SPEC.md): the FROZEN cell-1,412 rule applied to one session, forward,
with the identical mechanism as TEST (PREREG_1412.md, test_1412.py). No orders, no engine change,
no edits to any existing research cache -- data/cache.db, bars_sip.db and bars_rth.db (PMH causal)
are read-only here; the only DB this script ever writes is its own forward_bars.db.

Pipeline per date D:
  1. Universe = study_orb_broad.load_broad_universe patched to the wide-seed constants (MIN_GAP_PCT
     3.0, MAX_OPEN_PRICE 50.0; MIN_OPEN_PRICE 3 / MIN_PREV_DAY_VOL 500K already match the defaults),
     DATE_START=DATE_END=D so the query touches one day -- byte-identical to pm_candidates.csv
     (research/orb_seed_wide/build_wide_features.py). D's daily_bars row must exist (else ERROR).
  2. RTH 1-min SIP bars: bars_sip.db -> bars_rth.db (PMH causal, read-only) -> forward_bars.db (own,
     resumable). Missing symbol-days are fetched from Alpaca SIP (pattern verbatim from
     research/hod_pmh_causal/fetch_rth.py) and appended ONLY to forward_bars.db.
  3. Base signal (run_consol.find_signal/build_signal), C1 walk (run_consol.walk + run_consol.fill_c1)
     -> net_R_proxy (proxy cost, PREREG's 15bp half-spread both legs + 2bp/side). BR per minute
     (breadth.symbol_minute_flags) attached at signal_m (causal: closes <= signal_m). kept =
     BR(signal_m) >= EDGE (0.6115, PREREG_1412.md, never re-tuned here); order_in_day among kept,
     ranked by entry_m.
  4. Measured cost: Alpaca SIP NBBO quote in force at the entry bar's open and at the exit bar's
     timestamp for every kept trade (last quote at-or-before, 300s lookback -- pattern verbatim from
     research/fuckup_audit/O_halt/REVIVE/fetch_exit_nbbo.py), half-spread charged per leg from that
     leg's OWN measured spread + the same 2bp/side slip term as run_consol.cost_net -> net_R_measured.
     Coverage (share of kept trades with both quotes found) is logged.
  5. Placebo: one seeded random-minute long per kept trade, same name-day, drawn from the day's OWN
     risk-on minutes (BR(m) >= EDGE -- this population's own definition of risk-on, PREREG_1412.md),
     stop = min low of the prior 20 bars, 1.5% cost floor (<=5 redraws, run_consol.PREREG constants),
     C1 exit (run_consol.fill_c1) -> placebo_R.
  6. Append to forward_ledger.csv / forward_summary.csv. Idempotent: re-running a date replaces its
     rows in both files.
  7. Telegram summary line via scripts/send_telegram_alert.py, unless --no-telegram.

Usage:
  python3 research/day_breadth/forward_ledger.py --date 2026-09-22
  python3 research/day_breadth/forward_ledger.py --dates-from 2026-09-16 --dates-to 2026-09-18 --no-telegram
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import subprocess
import sys
import time
from datetime import date as _date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
HERE = f'{ROOT}/research/day_breadth'
sys.path.insert(0, f'{ROOT}/research/hod_consol')
sys.path.insert(0, HERE)
import run_consol as rc          # noqa: E402 -- find_signal, build_signal, walk, fill_c1, simulate_slots, cost_net, W
import breadth as B              # noqa: E402 -- symbol_minute_flags, M0/M1/NM
import study_orb_broad as broad  # noqa: E402 -- load_broad_universe, patched constants
from score_cells import SLIP_BP  # noqa: E402 -- same 2bp/side slip term as run_consol.cost_net

log = logging.getLogger('forward_ledger')

EDGE = 0.6115                                          # cell 1,412 frozen (PREREG_1412.md)
WIDE_MIN_GAP_PCT, WIDE_MAX_OPEN_PRICE = 3.0, 50.0      # research/orb_seed_wide/build_wide_features.py
FORWARD_BARS_DB = f'{HERE}/forward_bars.db'
LEDGER_CSV = f'{HERE}/forward_ledger.csv'
SUMMARY_CSV = f'{HERE}/forward_summary.csv'
BATCH = 100                                            # symbols/Alpaca request, same as fetch_rth.py
NBBO_LOOKBACK_S = 300                                  # same lookback as fetch_exit_nbbo.py
LEDGER_COLS = ['date', 'symbol', 'signal_m', 'entry_m', 'exit_m', 'why', 'R', 'r_pct', 'BR', 'kept',
               'order_in_day', 'net_R_proxy', 'net_R_measured', 'placebo_R']
SUMMARY_COLS = ['date', 'n_universe', 'n_signals', 'n_kept', 'kept_R_sum_proxy', 'kept_R_sum_measured',
                'first4_R_sum', 'later_R_sum', 'placebo_mean']

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
  symbol TEXT, day TEXT, t TEXT, o REAL, h REAL, l REAL, c REAL, v REAL,
  PRIMARY KEY (symbol, day, t)
);
CREATE INDEX IF NOT EXISTS idx_bars_day on bars(day);
"""


# ---------------------------------------------------------------------------------------------
# Universe (data/cache.db::daily_bars via the patched study_orb_broad -- byte-identical constants
# to pm_candidates.csv's wide seed; research/orb_seed_wide/build_wide_features.py)
# ---------------------------------------------------------------------------------------------

def daily_bar_exists(day: str, db_path: str = None) -> bool:
    con = sqlite3.connect(db_path or broad.CACHE_DB)
    n = con.execute('SELECT COUNT(*) FROM daily_bars WHERE bar_date = ?', (day,)).fetchone()[0]
    con.close()
    return n > 0


def universe_for_day(day: str, db_path: str = None) -> list:
    """Gapper universe for ONE day. Mutates study_orb_broad's module constants (the same pattern
    build_wide_features.py uses) -- these are read as globals inside load_broad_universe's query at
    CALL time, so the patch takes effect; db_path is passed explicitly since load_broad_universe's
    own default argument is bound at import time and would NOT see a later CACHE_DB patch."""
    broad.MIN_GAP_PCT = WIDE_MIN_GAP_PCT
    broad.MAX_OPEN_PRICE = WIDE_MAX_OPEN_PRICE
    broad.DATE_START = day
    broad.DATE_END = day
    uni = broad.load_broad_universe(db_path=db_path or broad.CACHE_DB)
    return sorted(set(uni.get(day, [])))


# ---------------------------------------------------------------------------------------------
# RTH bars: bars_sip.db -> bars_rth.db (PMH causal, read-only) -> forward_bars.db (own, resumable)
# fetch pattern verbatim from research/hod_pmh_causal/fetch_rth.py
# ---------------------------------------------------------------------------------------------

def init_forward_db():
    con = sqlite3.connect(FORWARD_BARS_DB)
    con.executescript(SCHEMA)
    con.commit()
    return con


def _covered_symbols(con, day):
    return set(r[0] for r in con.execute('SELECT DISTINCT symbol FROM bars WHERE day=?', (day,)))


def fetch_missing_bars(symbols: list, day: str) -> None:
    """Fetch (symbol, day) pairs not already in bars_sip.db / bars_rth.db / forward_bars.db from
    Alpaca SIP into forward_bars.db ONLY (bars_sip.db / bars_rth.db are never written here).
    Resumable: a symbol already present in forward_bars.db is not re-fetched."""
    sip = sqlite3.connect(rc.BARS_SIP_DB)
    have_sip = _covered_symbols(sip, day)
    sip.close()
    have_rth = set()
    if os.path.exists(rc.BARS_RTH_DB):
        rth = sqlite3.connect(rc.BARS_RTH_DB)
        have_rth = _covered_symbols(rth, day)
        rth.close()
    fcon = init_forward_db()
    have_fwd = _covered_symbols(fcon, day)
    todo = sorted(set(symbols) - have_sip - have_rth - have_fwd)
    if not todo:
        fcon.close()
        log.info('[bars] %s: all %d universe symbols already cached (sip=%d rth=%d fwd=%d)',
                  day, len(symbols), len(have_sip), len(have_rth), len(have_fwd))
        return
    now = int(datetime.now(timezone.utc).strftime('%H%M'))
    if rc.BLOCK_START <= now < rc.BLOCK_END:
        log.error('[bars] %s: UTC %04d is inside the heavy-DB blackout [%d,%d) -- refusing to fetch '
                  '%d missing symbols; re-run outside the window', day, now, rc.BLOCK_START,
                  rc.BLOCK_END, len(todo))
        fcon.close()
        raise RuntimeError('fetch attempted inside the heavy-DB blackout window')
    from config import Config
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    cfg = Config()
    if not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
        log.error('[bars] ALPACA API key/secret empty -- cannot fetch %d missing symbol-days for %s',
                  len(todo), day)
        fcon.close()
        raise RuntimeError('missing Alpaca credentials')
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(hours=8)
    end = start + timedelta(hours=16)
    fetched = 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        for attempt in range(3):
            try:
                req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Minute,
                                        start=start, end=end, feed='sip')
                df = client.get_stock_bars(req).df
                if df is not None and not df.empty:
                    df = df.reset_index()
                    out = pd.DataFrame({
                        'symbol': df['symbol'], 'day': day,
                        't': pd.to_datetime(df['timestamp'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
                        'o': df['open'], 'h': df['high'], 'l': df['low'], 'c': df['close'], 'v': df['volume']})
                    out.to_sql('bars', fcon, if_exists='append', index=False)
                    fetched += len(out)
                break
            except Exception as e:  # noqa: BLE001
                log.warning('[bars] %s chunk %d attempt %d failed: %s', day, i // BATCH, attempt + 1, e)
                time.sleep(2 * (attempt + 1))
        else:
            log.error('[bars] %s chunk %d SKIPPED after 3 attempts', day, i // BATCH)
    fcon.commit()
    fcon.close()
    log.info('[bars] %s: fetched %d bar rows for %d/%d missing symbols', day, fetched, len(todo), len(symbols))


def fetch_day_bars(sip_con, rth_con, fwd_con, symbol, day):
    """bars_sip.db -> bars_rth.db -> forward_bars.db (own). Same parse as run_consol.fetch_day_bars_dual."""
    d, src = rc.fetch_day_bars_dual(sip_con, rth_con, symbol, day)
    if d is not None:
        return d, src
    cur = fwd_con.execute('SELECT t,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY t', (symbol, day))
    rows = cur.fetchall()
    if not rows:
        return None, None
    d = pd.DataFrame(rows, columns=['t', 'o', 'h', 'l', 'c', 'v'])
    ts = pd.to_datetime(d.t, utc=True).dt.tz_convert('America/New_York')
    d['m'] = ts.dt.hour * 60 + ts.dt.minute
    d = d[(d.m >= rc.W.OPEN_M) & (d.m <= rc.W.CLOSE_M)].sort_values('m').drop_duplicates('m').reset_index(drop=True)
    return (d if len(d) else None), 'forward'


# ---------------------------------------------------------------------------------------------
# BR (breadth.py's causal per-minute flags) for the universe on day D
# ---------------------------------------------------------------------------------------------

def day_breadth(bars_by_symbol: dict) -> pd.DataFrame:
    """BR(m) = share of the universe trading above its 09:30 open, minute by minute (causal)."""
    acc = np.zeros((2, B.NM), dtype=np.int64)
    for d in bars_by_symbol.values():
        has, above, _, _ = B.symbol_minute_flags(d)
        acc[0] += has
        acc[1] += above
    with np.errstate(divide='ignore', invalid='ignore'):
        br = acc[1] / acc[0]
    return pd.DataFrame(dict(m=np.arange(B.M0, B.M1 + 1), n_bar=acc[0], n_above=acc[1], BR=br))


def apply_kept_rule(w: pd.DataFrame, edge: float = EDGE) -> pd.DataFrame:
    """kept = BR(signal_m) >= edge (cell 1,412, PREREG_1412.md). NaN BR -> not kept (defensive)."""
    w = w.copy()
    w['kept'] = w.BR >= edge
    w['order_in_day'] = np.nan
    kept_idx = w[w.kept].sort_values('entry_m').index
    w.loc[kept_idx, 'order_in_day'] = range(1, len(kept_idx) + 1)
    return w


# ---------------------------------------------------------------------------------------------
# Measured NBBO cost (pattern verbatim from research/fuckup_audit/O_halt/REVIVE/fetch_exit_nbbo.py)
# ---------------------------------------------------------------------------------------------

def _quote_at_or_before(client, symbol, when):
    from alpaca.data.requests import StockQuotesRequest
    from alpaca.data.enums import DataFeed
    try:
        q = client.get_stock_quotes(StockQuotesRequest(
            symbol_or_symbols=symbol,
            start=(when - pd.Timedelta(seconds=NBBO_LOOKBACK_S)).to_pydatetime(),
            end=(when + pd.Timedelta(seconds=1)).to_pydatetime(),
            feed=DataFeed.SIP))
        qs = q.data.get(symbol, []) if hasattr(q, 'data') else q.get(symbol, [])
        rows = [(pd.Timestamp(x.timestamp), float(x.bid_price), float(x.ask_price)) for x in qs
                if float(x.bid_price) > 0 and float(x.ask_price) >= float(x.bid_price)]
        prev = None
        for ts, bid, ask in rows:
            if ts <= when:
                prev = (bid, ask)
            else:
                break
        return prev
    except Exception as e:  # noqa: BLE001
        log.warning('[nbbo] %s @ %s failed: %s', symbol, when, e)
        return None


def measured_cost(client, kept: pd.DataFrame, bars_by_symbol: dict) -> pd.Series:
    """Half-spread PER LEG from that leg's OWN measured NBBO (entry bar's open, exit bar's
    timestamp) + the same 2bp/side slip term as run_consol.cost_net -> net_R_measured."""
    out = {}
    n_ok = 0
    for r in kept.itertuples():
        d = bars_by_symbol[r.symbol]
        entry_row, exit_row = d[d.m == r.entry_m], d[d.m == r.exit_m]
        if entry_row.empty or exit_row.empty:
            out[r.Index] = np.nan
            continue
        qe = _quote_at_or_before(client, r.symbol, pd.Timestamp(entry_row.t.iloc[0]))
        qx = _quote_at_or_before(client, r.symbol, pd.Timestamp(exit_row.t.iloc[0]))
        time.sleep(0.03)
        if qe is None or qx is None:
            out[r.Index] = np.nan
            continue
        spread_entry, spread_exit = qe[1] - qe[0], qx[1] - qx[0]
        slip_R = SLIP_BP * (r.entry + r.exit_price) / r.R
        cost_R = 0.5 * spread_entry / r.R + 0.5 * spread_exit / r.R + slip_R
        out[r.Index] = (r.exit_price - r.entry) / r.R - cost_R
        n_ok += 1
    log.info('[nbbo] measured cost coverage: %d/%d kept trades (%.0f%%)', n_ok, len(kept),
              100.0 * n_ok / len(kept) if len(kept) else 0.0)
    return pd.Series(out)


# ---------------------------------------------------------------------------------------------
# Placebo: one seeded random-minute long per kept trade, drawn from the day's OWN risk-on minutes
# ---------------------------------------------------------------------------------------------

def placebo_riskon(kept: pd.DataFrame, bars_by_symbol: dict, br: pd.DataFrame) -> pd.Series:
    """Same name-day, seeded random minute drawn from {m in [SCAN_START_M,SCAN_END_M]: BR(m) >=
    EDGE} (this population's own definition of risk-on), stop = min low of the prior 20 bars, 1.5%
    cost floor (<=5 redraws, run_consol.PLACEBO_MAX_REDRAWS), C1 exit (run_consol.fill_c1). One
    draw per kept trade, seeded on (day, symbol, entry_m) for idempotent re-runs."""
    riskon = br.loc[(br.m >= rc.SCAN_START_M) & (br.m <= rc.SCAN_END_M) & (br.BR >= EDGE), 'm'].values
    out = {}
    for r in kept.itertuples():
        d = bars_by_symbol[r.symbol]
        val = np.nan
        if len(riskon):
            rng = np.random.default_rng(abs(hash((r.day, r.symbol, int(r.entry_m)))) % (2 ** 32))
            for _ in range(rc.PLACEBO_MAX_REDRAWS + 1):
                m0 = int(rng.choice(riskon))
                entry_row = d[d.m == m0 + 1]
                prior = d[(d.m >= m0 - rc.BASE_WINDOW + 1) & (d.m <= m0)]
                if entry_row.empty or len(prior) < rc.BASE_WINDOW:
                    continue
                entry, stop = float(entry_row.o.iloc[0]), float(prior.l.min())
                ok, R = rc._floor_ok(entry, stop)
                if not ok:
                    continue
                after = d[(d.m > m0 + 1) & (d.m <= rc.EOD_M)]
                if after.empty:
                    continue
                res = rc.fill_c1(entry, stop, R, after.itertuples())
                if res is None:
                    continue
                _, exit_px, _ = res
                val = (exit_px - entry) / R
                break
        out[r.Index] = val
    return pd.Series(out)


# ---------------------------------------------------------------------------------------------
# Ledger I/O -- idempotent (re-running a date replaces its rows)
# ---------------------------------------------------------------------------------------------

def _replace_date_rows(path: str, day: str, new_rows: pd.DataFrame) -> None:
    if os.path.exists(path):
        old = pd.read_csv(path, keep_default_na=True)
        old = old[old.date != day]
        out = pd.concat([old, new_rows], ignore_index=True)
    else:
        out = new_rows
    out.to_csv(path, index=False)


def _telegram(line: str) -> None:
    try:
        subprocess.run([sys.executable, f'{ROOT}/scripts/send_telegram_alert.py', line],
                        timeout=30, check=False)
    except Exception as e:  # noqa: BLE001
        log.warning('[telegram] send failed: %s', e)


# ---------------------------------------------------------------------------------------------
# One day, end to end
# ---------------------------------------------------------------------------------------------

def process_day(day: str, send_telegram: bool = True) -> bool:
    log.info('=== %s ===', day)
    if not daily_bar_exists(day):
        log.error('[universe] daily_bars has no row for %s -- the 10:30 UTC batch has not updated '
                  'it yet; not proceeding', day)
        return False
    symbols = universe_for_day(day)
    log.info('[universe] %s: %d symbols (gap>=%.1f%%, open $3-%.0f, prevvol>=500K)',
              day, len(symbols), WIDE_MIN_GAP_PCT, WIDE_MAX_OPEN_PRICE)
    if not symbols:
        log.warning('[universe] %s: zero gapper candidates', day)

    if symbols:
        fetch_missing_bars(symbols, day)

    sip_con = sqlite3.connect(rc.BARS_SIP_DB)
    rth_con = sqlite3.connect(rc.BARS_RTH_DB) if os.path.exists(rc.BARS_RTH_DB) else None
    fwd_con = sqlite3.connect(FORWARD_BARS_DB)
    bars_by_symbol, sig_rows, path_frames = {}, [], []
    for sym in symbols:
        d, _src = fetch_day_bars(sip_con, rth_con, fwd_con, sym, day)
        if d is None or len(d) < 5:
            continue
        bars_by_symbol[sym] = d
        sigbar, _reason = rc.find_signal(d)
        if sigbar is None:
            continue
        sig, _reason2 = rc.build_signal(d, sigbar, sym, day)
        if sig is None:
            continue
        sig_rows.append(sig)
        p = d.copy()
        p['day'], p['symbol'] = day, sym
        path_frames.append(p[['day', 'symbol', 'm', 'o', 'h', 'l', 'c', 't']])
    sip_con.close()
    if rth_con is not None:
        rth_con.close()
    fwd_con.close()
    log.info('[bars] %s: %d/%d symbols with bars, %d base signals', day, len(bars_by_symbol),
              len(symbols), len(sig_rows))

    if not sig_rows:
        log.warning('[signals] %s: zero base signals', day)
        _replace_date_rows(LEDGER_CSV, day, pd.DataFrame(columns=LEDGER_COLS))
        _replace_date_rows(SUMMARY_CSV, day, pd.DataFrame([dict(
            date=day, n_universe=len(symbols), n_signals=0, n_kept=0, kept_R_sum_proxy=0.0,
            kept_R_sum_measured=0.0, first4_R_sum=0.0, later_R_sum=0.0, placebo_mean=float('nan'))]))
        if send_telegram:
            _telegram(f'[FWD 1412] {day}: kept 0, R proxy/measured n/a, first-4 vs later n/a, placebo n/a')
        return True

    sig = pd.DataFrame(sig_rows)
    sig['split'] = 'FWD'
    sig['half'] = None
    sig['wk'] = rc.week_monday(day)
    paths = pd.concat(path_frames, ignore_index=True)
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()

    w = rc.walk(sig, idx, rc.fill_c1)
    if w.empty:
        log.warning('[walk] %s: zero trades walked (all signals dropped -- missing path data)', day)
        _replace_date_rows(LEDGER_CSV, day, pd.DataFrame(columns=LEDGER_COLS))
        _replace_date_rows(SUMMARY_CSV, day, pd.DataFrame([dict(
            date=day, n_universe=len(symbols), n_signals=len(sig), n_kept=0, kept_R_sum_proxy=0.0,
            kept_R_sum_measured=0.0, first4_R_sum=0.0, later_R_sum=0.0, placebo_mean=float('nan'))]))
        if send_telegram:
            _telegram(f'[FWD 1412] {day}: kept 0 (zero trades walked)')
        return True
    w = w.rename(columns={'net_R': 'net_R_proxy'})
    w = w.merge(sig[['day', 'symbol', 'signal_m', 'r_pct']], on=['day', 'symbol', 'signal_m'], how='left')

    br = day_breadth(bars_by_symbol)
    w['dm'] = w.signal_m.clip(B.M0, B.M1).astype(int)
    w = w.merge(br[['m', 'BR']], left_on='dm', right_on='m', how='left')
    miss = w.BR.isna().mean()
    if miss > 0:
        log.warning('[BR] %s: %.1f%% of signals have no BR value (unexpected intra-day)', day, miss * 100)

    w = apply_kept_rule(w)
    kept = w[w.kept].copy()
    w['net_R_measured'] = np.nan
    w['placebo_R'] = np.nan
    if len(kept):
        now = int(datetime.now(timezone.utc).strftime('%H%M'))
        if rc.BLOCK_START <= now < rc.BLOCK_END:
            log.error('[nbbo] %s: UTC %04d inside the heavy-DB blackout -- measured cost/placebo '
                      'skipped this run, re-run outside the window', day, now)
        else:
            from config import Config
            cfg = Config()
            if not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
                log.error('[nbbo] ALPACA API key/secret empty -- measured cost cannot be computed for %s', day)
            else:
                from alpaca.data.historical import StockHistoricalDataClient
                client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
                w.loc[kept.index, 'net_R_measured'] = measured_cost(client, kept, bars_by_symbol)
                w.loc[kept.index, 'placebo_R'] = placebo_riskon(kept, bars_by_symbol, br)

    w['date'] = w['day']
    ledger = w[LEDGER_COLS].copy()
    k = w[w.kept]
    first4, later = k[k.order_in_day <= 4], k[k.order_in_day > 4]
    summary = pd.DataFrame([dict(
        date=day, n_universe=len(symbols), n_signals=len(w), n_kept=len(k),
        kept_R_sum_proxy=float(k.net_R_proxy.sum()), kept_R_sum_measured=float(k.net_R_measured.sum()),
        first4_R_sum=float(first4.net_R_proxy.sum()), later_R_sum=float(later.net_R_proxy.sum()),
        placebo_mean=float(k.placebo_R.mean()) if len(k) else float('nan'))])

    _replace_date_rows(LEDGER_CSV, day, ledger)
    _replace_date_rows(SUMMARY_CSV, day, summary)
    log.info('[done] %s: n_signals=%d n_kept=%d R_proxy=%.2f R_measured=%.2f first4=%.2f later=%.2f '
              'placebo=%.3f', day, len(w), len(k), k.net_R_proxy.sum(), k.net_R_measured.sum(),
              first4.net_R_proxy.sum(), later.net_R_proxy.sum(),
              k.placebo_R.mean() if len(k) else float('nan'))
    if send_telegram:
        if len(k):
            _telegram(f'[FWD 1412] {day}: kept {len(k)}, R proxy {k.net_R_proxy.sum():+.2f} / '
                      f'measured {k.net_R_measured.sum():+.2f}, first-4 {first4.net_R_proxy.sum():+.2f} '
                      f'vs later {later.net_R_proxy.sum():+.2f}, placebo {k.placebo_R.mean():+.3f}')
        else:
            _telegram(f'[FWD 1412] {day}: kept 0')
    return True


def business_days(d1: str, d2: str) -> list:
    a, b = _date.fromisoformat(d1), _date.fromisoformat(d2)
    out = []
    while a <= b:
        if a.weekday() < 5:
            out.append(a.isoformat())
        a += timedelta(days=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--date')
    ap.add_argument('--dates-from')
    ap.add_argument('--dates-to')
    ap.add_argument('--no-telegram', action='store_true')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    if a.date:
        days = [a.date]
    elif a.dates_from and a.dates_to:
        days = business_days(a.dates_from, a.dates_to)
    else:
        ap.error('--date or --dates-from/--dates-to required')
        return 2
    ok = True
    for d in days:
        try:
            ok = process_day(d, send_telegram=not a.no_telegram) and ok
        except Exception as e:  # noqa: BLE001
            log.error('[FATAL] %s: %s', d, e)
            ok = False
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
