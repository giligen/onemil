"""Cell 1,427 — INDEPENDENT rebuild of the E1 resting buy-stop-limit on the Alpaca SIP consolidated tape.

Written from the prose of research/hod_entry/PREREG_1427.md ONLY (the first implementation
entry_replay.py was not read). The rule, literally:

  * S        = entry_m x 60 ET seconds (entry_m = the B0 entry bar); break bar = [S-60 s, S).
  * level    = running max of CLOSED 1-minute RTH highs (09:30 onward) before the break bar
               (the HOD-break spec: hod[i-1] over bars[:i]), from data/cache.db intraday_bars_1min.
  * trigger  = level + 0.01 ; limit = level x 1.0015 (15 bps).
  * fill     = the prevailing SIP NBBO ask at the FIRST consolidated print >= trigger inside the break
               bar, if that ask <= limit; otherwise NO FILL (no position).
  * after the fill, a print <= stop inside the rest of the break bar = stopped at the stop.
  * then B0's path rules on research/hod_exit_lab/paths.parquet from the entry bar on (gap-through
    at the open, low <= stop -> stop, high >= target -> target, stop-before-target on a bar that
    touches both, exit at the 15:55 bar open); target = fill + 2 R with R = fill - stop.
  * cost     = half the NBBO spread at the fill instant (entry leg) + B0's exit-leg rule
               (half the B0 per-signal measured spread + 2 bp of the exit price).
  * tape availability rail: a signal is USABLE iff the break bar has >= 1 print and a valid NBBO
    quote exists in [S-65 s, S); a triggering print with no prevailing quote is also unusable.

Usage:
  python research/hod_entry/sip_rebuild.py --part val     # TRAIN-H2 + VAL, writes the agreement check
  python research/hod_entry/sip_rebuild.py --part test    # TEST, refuses unless the agreement PASSED
"""
import argparse
import datetime as dt
import gzip
import os
import pickle
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
BOOK_CSV = os.path.join(ROOT, 'research/hod_exit_lab/b0_trades.csv')
PATHS_PARQUET = os.path.join(ROOT, 'research/hod_exit_lab/paths.parquet')
CACHE_DB_URI = 'file:' + os.path.join(ROOT, 'data/cache.db') + '?mode=ro'
CACHE_DIR = os.path.join(HERE, 'sip_cache')
REPORT_MD = os.path.join(HERE, 'REPORT_1427.md')
OUT_CSV = os.path.join(HERE, 'sip_rebuild_{part}.csv')

ET = ZoneInfo('America/New_York')
OPEN_M, EOD_M = 570, 955
TICK = 0.01
LIMIT_BPS = 0.0015
QUOTE_LOOKBACK_S = 5
QUOTE_HISTORY_S = 900     # prevailing-quote history: last valid NBBO in [S-900 s, S-65 s), else back to 09:30
TARGET_R = 2.0
SLIP_BP = 0.0002
# Cell 1,423 VAL reference numbers and the frozen agreement bands (PREREG_1427.md).
REF_VAL_MEAN_R, REF_VAL_FILL_RATE = 0.384, 0.299
BAND_R, BAND_FILL = 0.05, 0.10
AGREEMENT_PASS_TOKEN = 'AGREEMENT VERDICT: PASS'

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'research/hod_consol'))
sys.path.insert(0, os.path.join(ROOT, 'research/hod_exit_lab'))


def log(msg):
    """Timestamped, flushed progress line."""
    print(f'[{dt.datetime.now():%H:%M:%S}] {msg}', flush=True)


# --------------------------------------------------------------------------------------------- time
def et_ns(day, seconds):
    """UTC epoch nanoseconds of `seconds` after ET midnight on `day` (YYYY-MM-DD)."""
    d = dt.date.fromisoformat(day)
    base = dt.datetime(d.year, d.month, d.day, tzinfo=ET)
    return int((base + dt.timedelta(seconds=float(seconds))).timestamp() * 1e9)


def ns_to_et_minutes(ns, day):
    """Fractional ET minutes-since-midnight of a UTC-ns timestamp on `day`."""
    return (ns - et_ns(day, 0)) / 60e9


# --------------------------------------------------------------------------------------------- book
def load_book(part):
    """The B0 book rows for a part: 'val' = TRAIN-H2 + VAL, 'test' = TEST only."""
    b = pd.read_csv(BOOK_CSV, dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
    if part == 'val':
        b = b[((b.split == 'TRAIN') & (b.half == 'H2')) | (b.split == 'VAL')]
    elif part == 'test':
        b = b[b.split == 'TEST']
    else:
        raise ValueError(f'unknown part {part!r}')
    b = b.reset_index(drop=True).copy()
    b['break_m'] = b.entry_m - 1
    b['S_ns'] = [et_ns(d, m * 60) for d, m in zip(b.day, b.entry_m)]
    # B0 exit-leg half-spread ($) recovered from B0's own cost: cost_R*R = spread + 2bp*(entry+exit)
    b['b0_exit_half'] = 0.5 * (b.cost_R * b.R - SLIP_BP * (b.entry + b.exit_price))
    log(f'[book] part={part}: {len(b)} signals, splits={b.groupby(["split", "half"]).size().to_dict()}')
    return b


def hod_level(bars, break_m):
    """Running max of closed RTH 1-min highs strictly before the break bar (09:30 <= m < break_m).
    `bars` has integer ET minute column `m` and `high`. NaN if no bar qualifies."""
    sel = bars[(bars.m >= OPEN_M) & (bars.m < break_m)]
    return float(sel.high.max()) if len(sel) else float('nan')


def load_levels(book):
    """Level per signal from data/cache.db intraday_bars_1min (read-only). Also records the break
    bar's minute high for a consistency count (high < level means the minute bars disagree)."""
    con = sqlite3.connect(CACHE_DB_URI, uri=True)
    levels, bb_high = [], []
    cache = {}
    for i, r in enumerate(book.itertuples()):
        key = (r.symbol, r.day)
        if key not in cache:
            q = pd.read_sql_query('SELECT timestamp, high FROM intraday_bars_1min WHERE symbol=? AND bar_date=?',
                                  con, params=key)
            ts = pd.to_datetime(q.timestamp, utc=True).dt.tz_convert(ET)
            q['m'] = ts.dt.hour * 60 + ts.dt.minute
            cache = {key: q}                       # one (symbol, day) at a time — book is sorted by day
        q = cache[key]
        levels.append(hod_level(q, r.break_m))
        hb = q.high[q.m == r.break_m]
        bb_high.append(float(hb.iloc[0]) if len(hb) else float('nan'))
        if (i + 1) % 2000 == 0:
            log(f'[levels] {i + 1}/{len(book)}')
    con.close()
    book['level'] = levels
    book['bb_high_1m'] = bb_high
    n_nan = int(book.level.isna().sum())
    if n_nan:
        log(f'[levels] WARNING: {n_nan} signals have no RTH minute bars before the break bar — level NaN, '
            f'they cannot be simulated and are counted as unusable')
    n_dis = int((book.bb_high_1m < book.level).sum())
    log(f'[levels] done; break-bar minute high < level on {n_dis} signals (minute-bar disagreement count)')
    return book


# --------------------------------------------------------------------------------------------- tape
_tls = threading.local()


def _client():
    """Per-thread alpaca-py StockHistoricalDataClient (raw_data=True for speed)."""
    if not hasattr(_tls, 'client'):
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, '.env'))
        import config
        from alpaca.data.historical import StockHistoricalDataClient
        c = config.Config()
        if not c.alpaca_api_key or not c.alpaca_api_secret:
            raise RuntimeError('ALPACA api key/secret missing — cannot fetch the SIP tape')
        _tls.client = StockHistoricalDataClient(c.alpaca_api_key, c.alpaca_api_secret, raw_data=True)
    return _tls.client


def _records(raw, symbol, kind):
    """Pull the record list out of an alpaca-py raw response (dict keyed by symbol or by kind)."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        if symbol in raw:
            return raw[symbol] or []
        if kind in raw and isinstance(raw[kind], dict):
            return raw[kind].get(symbol) or []
    return []


def fetch_tape(symbol, S_ns, retries=6):
    """SIP trades and quotes for [S-65 s, S). Returns (trades df[ts, price, size], quotes df[ts, bid, ask]).
    Retries with exponential backoff; raises after `retries` failures (the caller counts it LOST)."""
    from alpaca.data.requests import StockTradesRequest, StockQuotesRequest
    start = pd.Timestamp(S_ns - (60 + QUOTE_LOOKBACK_S) * 10**9, unit='ns', tz='UTC').to_pydatetime()
    end = pd.Timestamp(S_ns, unit='ns', tz='UTC').to_pydatetime()
    for attempt in range(retries):
        try:
            cl = _client()
            tr = _records(cl.get_stock_trades(StockTradesRequest(symbol_or_symbols=symbol, start=start, end=end,
                                                                 feed='sip')), symbol, 'trades')
            qu = _records(cl.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=symbol, start=start, end=end,
                                                                 feed='sip')), symbol, 'quotes')
            t = pd.DataFrame({'ts': pd.to_datetime([x['t'] for x in tr], utc=True).asi8 if tr else [],
                              'price': [float(x['p']) for x in tr], 'size': [float(x.get('s', 0)) for x in tr]})
            q = pd.DataFrame({'ts': pd.to_datetime([x['t'] for x in qu], utc=True).asi8 if qu else [],
                              'bid': [float(x.get('bp', 0)) for x in qu], 'ask': [float(x.get('ap', 0)) for x in qu]})
            return t.astype({'ts': 'int64'}), q.astype({'ts': 'int64'})
        except Exception as e:                                   # noqa: BLE001 — network/rate limit
            wait = min(60, 2 ** attempt)
            log(f'[fetch] WARNING {symbol} {S_ns}: {type(e).__name__}: {e} — retry {attempt + 1}/{retries} in {wait}s')
            time.sleep(wait)
    raise RuntimeError(f'fetch failed after {retries} attempts: {symbol} {S_ns}')


def sig_key(symbol, break_m):
    """Cache key of one signal inside its day file."""
    return f'{symbol}|{int(break_m)}'


def fetch_all(book, workers):
    """Resumable per-day cache under sip_cache/{day}.pkl.gz: {key: (trades, quotes)}. Days are
    fetched with a thread pool across the day's signals; a day file is rewritten with whatever
    succeeded, so a re-run fetches only what is missing. Returns {day: {key: (t, q)}} and LOST count."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tapes, lost, n_req, t0 = {}, 0, 0, time.time()
    days = sorted(book.day.unique())
    for di, day in enumerate(days):
        path = os.path.join(CACHE_DIR, f'{day}.pkl.gz')
        have = {}
        if os.path.exists(path):
            with gzip.open(path, 'rb') as f:
                have = pickle.load(f)
        g = book[book.day == day]
        todo = [(r.symbol, r.break_m, r.S_ns) for r in g.itertuples() if sig_key(r.symbol, r.break_m) not in have]
        if todo:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(fetch_tape, s, S): sig_key(s, bm) for s, bm, S in todo}
                for fu in as_completed(futs):
                    try:
                        have[futs[fu]] = fu.result()
                        n_req += 2
                    except Exception as e:                       # noqa: BLE001
                        lost += 1
                        log(f'[fetch] ERROR {day} {futs[fu]}: {e} — LOST (re-run to resume)')
            tmp = path + '.tmp'
            with gzip.open(tmp, 'wb') as f:
                pickle.dump(have, f)
            os.replace(tmp, path)
        tapes[day] = have
        if (di + 1) % 10 == 0 or di == len(days) - 1:
            el = time.time() - t0
            log(f'[fetch] day {di + 1}/{len(days)} ({day}) | requests this run {n_req} '
                f'({n_req / max(el, 1e-9) * 60:.0f}/min) | LOST {lost}')
    return tapes, lost


def _valid(q):
    """Rows of a quotes frame that are a usable NBBO (bid>0, ask>0, not crossed)."""
    return q[(q.ask > 0) & (q.bid > 0) & (q.ask >= q.bid)]


def needs_quote_history(quotes, S_ns):
    """True iff the cached window has no valid quote at or before the break-bar start (S-60 s): only then can
    the prevailing NBBO of a break-bar print predate the window. Rule-independent (a data-completeness test)."""
    return not len(_valid(quotes)[lambda d: d.ts <= S_ns - 60 * 10**9])


def _fetch_quotes(symbol, start_ns, end_ns, retries=6):
    """Valid SIP quotes in [start_ns, end_ns) as df[ts, bid, ask], with exponential-backoff retries."""
    from alpaca.data.requests import StockQuotesRequest
    start = pd.Timestamp(start_ns, unit='ns', tz='UTC').to_pydatetime()
    end = pd.Timestamp(end_ns, unit='ns', tz='UTC').to_pydatetime()
    for attempt in range(retries):
        try:
            qu = _records(_client().get_stock_quotes(StockQuotesRequest(symbol_or_symbols=symbol, start=start,
                                                                        end=end, feed='sip')), symbol, 'quotes')
            q = pd.DataFrame({'ts': pd.to_datetime([x['t'] for x in qu], utc=True).asi8 if qu else [],
                              'bid': [float(x.get('bp', 0)) for x in qu], 'ask': [float(x.get('ap', 0)) for x in qu]})
            q = _valid(q.astype({'ts': 'int64'}))
            return q[q.ts < end_ns]
        except Exception as e:                                   # noqa: BLE001 — network/rate limit
            wait = min(60, 2 ** attempt)
            log(f'[qhist] WARNING {symbol}: {type(e).__name__}: {e} — retry {attempt + 1}/{retries} in {wait}s')
            time.sleep(wait)
    raise RuntimeError(f'quote-history fetch failed after {retries} attempts: {symbol} {start_ns}')


def fetch_quote_history(symbol, day, S_ns):
    """The last valid NBBO strictly before the cached window (S-65 s): first from [S-900 s, S-65 s), and if
    that is empty from [09:30 ET, S-900 s). Returns a 0- or 1-row df[ts, bid, ask]."""
    w_end = S_ns - (60 + QUOTE_LOOKBACK_S) * 10**9
    opn = et_ns(day, OPEN_M * 60)
    q = _fetch_quotes(symbol, max(opn, S_ns - QUOTE_HISTORY_S * 10**9), w_end)
    if not len(q) and S_ns - QUOTE_HISTORY_S * 10**9 > opn:
        q = _fetch_quotes(symbol, opn, S_ns - QUOTE_HISTORY_S * 10**9)
    return q.sort_values('ts', kind='stable').tail(1).reset_index(drop=True)


def extend_quotes(book, tapes, workers):
    """Prepend the prevailing-quote history to every signal whose cached window lacks a valid quote at/before
    S-60 s. Resumable cache sip_cache/qhist_{day}.pkl.gz: {key: 0/1-row df}. Mutates `tapes`; returns LOST."""
    lost, n_need, n_found = 0, 0, 0
    for day in sorted(book.day.unique()):
        path = os.path.join(CACHE_DIR, f'qhist_{day}.pkl.gz')
        have = {}
        if os.path.exists(path):
            with gzip.open(path, 'rb') as f:
                have = pickle.load(f)
        need = []
        for r in book[book.day == day].itertuples():
            k = sig_key(r.symbol, r.break_m)
            tq = tapes.get(day, {}).get(k)
            if tq is not None and needs_quote_history(tq[1], r.S_ns):
                need.append((r.symbol, r.S_ns, k))
        todo = [x for x in need if x[2] not in have]
        if todo:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(fetch_quote_history, s, day, S): k for s, S, k in todo}
                for fu in as_completed(futs):
                    try:
                        have[futs[fu]] = fu.result()
                    except Exception as e:                       # noqa: BLE001
                        lost += 1
                        log(f'[qhist] ERROR {day} {futs[fu]}: {e} — LOST (re-run to resume)')
            tmp = path + '.tmp'
            with gzip.open(tmp, 'wb') as f:
                pickle.dump(have, f)
            os.replace(tmp, path)
        for s, S, k in need:
            if k in have and len(have[k]):
                t, q = tapes[day][k]
                tapes[day][k] = (t, pd.concat([have[k], q], ignore_index=True).sort_values('ts', kind='stable'))
                n_found += 1
        n_need += len(need)
    log(f'[qhist] {n_need} signals lacked a quote at/before S-60 s; prevailing quote found for {n_found}; '
        f'LOST {lost}')
    return lost


# --------------------------------------------------------------------------------------------- rule
def tape_usable(trades, quotes, S_ns):
    """Availability rail: >= 1 print inside [S-60 s, S) and >= 1 valid quote (bid>0, ask>0) in the window."""
    bb = trades[(trades.ts >= S_ns - 60 * 10**9) & (trades.ts < S_ns)]
    qv = quotes[(quotes.ask > 0) & (quotes.bid > 0) & (quotes.ts < S_ns)]
    return len(bb) > 0 and len(qv) > 0


def prevailing_quote(quotes, ts):
    """Last valid quote (bid>0, ask>0, ask>=bid) with timestamp <= ts, or None."""
    q = quotes[(quotes.ts <= ts) & (quotes.ask > 0) & (quotes.bid > 0) & (quotes.ask >= quotes.bid)]
    if not len(q):
        return None
    last = q.iloc[int(np.argmax(q.ts.values))] if not q.ts.is_monotonic_increasing else q.iloc[-1]
    return float(last.bid), float(last.ask)


def simulate_entry(trades, quotes, S_ns, level, stop, limit_bps=LIMIT_BPS, extra_ticks=0):
    """E1 entry inside the break bar. Returns dict(status in {'no_tape','nofill','fill'}, fill, fill_ts,
    half_spread, stopped_bb). `extra_ticks` adds ticks to the fill price (report-only 'ask + 1 tick')."""
    out = dict(status='no_tape', fill=np.nan, fill_ts=np.nan, half_spread=np.nan, stopped_bb=False, ask_at=np.nan)
    if not np.isfinite(level) or not tape_usable(trades, quotes, S_ns):
        return out
    trigger, limit = round(level + TICK, 6), level * (1.0 + limit_bps)
    bb = trades[(trades.ts >= S_ns - 60 * 10**9) & (trades.ts < S_ns)].sort_values('ts', kind='stable')
    hit = bb[bb.price >= trigger - 1e-9]
    if not len(hit):
        out['status'] = 'nofill'
        return out
    t_hit = int(hit.ts.iloc[0])
    pq = prevailing_quote(quotes, t_hit)
    if pq is None:
        return out                                               # triggering print, no quote -> unusable
    bid, ask = pq
    out['ask_at'] = ask
    if ask > limit + 1e-9:
        out['status'] = 'nofill'
        return out
    out.update(status='fill', fill=ask + extra_ticks * TICK, fill_ts=t_hit, half_spread=0.5 * (ask - bid))
    after = bb[bb.ts > t_hit]
    out['stopped_bb'] = bool((after.price <= stop + 1e-9).any())
    return out


def walk_path(fill, stop, target, path):
    """B0 path physics from the entry bar on (the E1 position is already open). `path` rows m >= entry_m,
    sorted. Gap-through at the open, low<=stop -> stop, high>=target -> target (stop first on a bar
    touching both), 15:55 bar -> its open. Returns (exit_m, exit_price, why)."""
    for row in path.itertuples():
        if row.m >= EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.l <= stop:
            return int(row.m), float(row.o if row.o <= stop else stop), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
    last = path.iloc[-1]
    log(f'[walk] WARNING: path ended before 15:55 (last m={int(last.m)}) — exit at its close')
    return int(last.m), float(last.c), 'eod_fallback'


def trade_result(fill, stop, half_entry, exit_px, exit_half):
    """(raw R, cost R, net R, R$) in the NEW R = fill - stop. Entry: half-spread at the fill instant;
    exit: B0's leg rule = half-spread + 2 bp of the exit price."""
    R = fill - stop
    raw = (exit_px - fill) / R
    cost = (half_entry + exit_half + SLIP_BP * exit_px) / R
    return raw, cost, raw - cost, R


def simulate_book(book, tapes, paths_idx, limit_bps=LIMIT_BPS, extra_ticks=0):
    """Run E1 on every signal of `book`. Returns a DataFrame with status, fill, exit, net_R per signal."""
    rows, n_fb = [], 0
    for r in book.itertuples():
        t, q = tapes.get(r.day, {}).get(sig_key(r.symbol, r.break_m), (None, None))
        base = dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, split=r.split, half=r.half, wk=r.wk,
                    b0_net_R=r.net_R, level=r.level)
        if t is None:
            rows.append({**base, 'status': 'lost'})
            continue
        e = simulate_entry(t, q, r.S_ns, r.level, r.stop, limit_bps, extra_ticks)
        base.update(status=e['status'], ask_at=e['ask_at'])
        if e['status'] != 'fill':
            rows.append(base)
            continue
        fill = e['fill']
        exit_half = r.b0_exit_half
        if not np.isfinite(exit_half):
            n_fb += 1
            exit_half = e['half_spread']
        target = fill + TARGET_R * (fill - r.stop)
        fill_min = ns_to_et_minutes(e['fill_ts'], r.day)
        if e['stopped_bb']:
            exit_m, exit_px, why = fill_min, float(r.stop), 'stop_bb'
        else:
            p = paths_idx.get((r.day, r.symbol))
            p = p[(p.m >= r.entry_m) & (p.m <= EOD_M)] if p is not None else None
            if p is None or not len(p):
                log(f'[sim] ERROR: no path for {r.day} {r.symbol} — counted LOST')
                rows.append({**base, 'status': 'lost'})
                continue
            exit_m, exit_px, why = walk_path(fill, r.stop, target, p)
        raw, cost, net, R = trade_result(fill, r.stop, e['half_spread'], exit_px, exit_half)
        rows.append({**base, 'fill': fill, 'fill_min': fill_min, 'exit_m': exit_m, 'exit_price': exit_px,
                     'why': why, 'R': R, 'raw_R': raw, 'cost_R': cost, 'net_R': net})
    if n_fb:
        log(f'[sim] WARNING: {n_fb} fills had no B0 exit spread (B0 cost NaN) — exit half-spread = the '
            f'fill-instant half-spread')
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- score
def score(res, book_part):
    """Stats of one split: coverage, missingness gap, fill rate, mean/t/ex-top-5 of fills, fills/week
    after slots (first 12/day, 4 concurrent), no-fill cohort B0 outcome."""
    from score_cells import day_clustered_t, ex_top5
    from run_consol import simulate_slots
    n = len(res)
    usable = res.status.isin(['fill', 'nofill'])
    win = res.b0_net_R > 0
    lose = res.b0_net_R <= 0
    miss_w = 1 - usable[win].mean() if win.any() else np.nan
    miss_l = 1 - usable[lose].mean() if lose.any() else np.nan
    f = res[res.status == 'fill'].copy()
    nf = res[res.status == 'nofill']
    t, nd = day_clustered_t(f.net_R, f.day) if len(f) > 1 else (np.nan, 0)
    slots = f.assign(entry_m=f.fill_min).reset_index(drop=True)
    keep = simulate_slots(slots) if len(slots) else pd.Series(dtype=bool)
    weeks = book_part.wk.nunique()
    kept = slots[keep] if len(slots) else slots
    return dict(n=n, usable=int(usable.sum()), coverage=usable.mean(), lost=int((res.status == 'lost').sum()),
                miss_win=miss_w, miss_lose=miss_l, miss_gap=abs(miss_w - miss_l),
                fills=len(f), fill_rate=len(f) / max(int(usable.sum()), 1),
                mean_R=f.net_R.mean(), t=t, ndays=nd, extop5=ex_top5(f.net_R) if len(f) else np.nan,
                stop_bb=int((f.why == 'stop_bb').sum()) if len(f) else 0,
                slot_fills=int(len(kept)), weeks=weeks, fills_wk=len(kept) / max(weeks, 1),
                slot_mean_R=kept.net_R.mean() if len(kept) else np.nan,
                nofill_n=len(nf), nofill_b0=nf.b0_net_R.mean(), fill_b0=f.b0_net_R.mean())


def fmt_row(name, s):
    """One markdown table row of a score dict."""
    return (f'| {name} | {s["n"]} | {s["usable"]} ({s["coverage"]*100:.1f} %) | {s["lost"]} | '
            f'{s["miss_win"]*100:.1f} / {s["miss_lose"]*100:.1f} (gap {s["miss_gap"]*100:.1f} pp) | '
            f'{s["fills"]} ({s["fill_rate"]*100:.1f} %) | {s["mean_R"]:+.3f} | {s["t"]:.2f} ({s["ndays"]} d) | '
            f'{s["extop5"]:+.3f} | {s["stop_bb"]} | {s["fills_wk"]:.1f} ({s["slot_mean_R"]:+.3f}) | '
            f'{s["nofill_b0"]:+.3f} (n {s["nofill_n"]}) vs fills {s["fill_b0"]:+.3f} |')


TABLE_HEAD = ('| cohort | signals | usable (coverage) | lost | missing win / lose | fills (rate of usable) | '
              'mean net R | day-clust t | ex-top-5 % | stopped in break bar | fills/wk after slots (mean R) | '
              'no-fill cohort B0 net R |\n|' + '---|' * 12)


def load_paths_idx(book):
    """Paths for the book's (day, symbol) keys as a dict of per-key frames sorted by m."""
    keys = set(zip(book.day, book.symbol))
    p = pd.read_parquet(PATHS_PARQUET, columns=['day', 'symbol', 'm', 'o', 'h', 'l', 'c'])
    p = p[p.day.isin(set(book.day))]
    idx = {k: g.sort_values('m') for k, g in p.groupby(['day', 'symbol']) if k in keys}
    miss = len(keys - set(idx))
    log(f'[paths] {len(idx)} (day,symbol) paths loaded; missing {miss}' + (' — WARNING' if miss else ''))
    return idx


def run(part, workers, quote_history=True):
    """Full pipeline for one part, writes the per-signal CSV and the report section."""
    book = load_levels(load_book(part))
    tapes, lost = fetch_all(book, workers)
    if lost:
        log(f'[fetch] retry pass for {lost} LOST signals')
        tapes, lost = fetch_all(book, workers)
    if lost:
        log(f'[fetch] ERROR: {lost} signals still LOST after the retry pass — they count as unusable')
    if quote_history:
        qlost = extend_quotes(book, tapes, workers)
        if qlost:
            log(f'[qhist] ERROR: {qlost} quote-history fetches LOST — re-run to resume')
    paths_idx = load_paths_idx(book)
    variants = [('E1 15 bps (frozen)', LIMIT_BPS, 0), ('limit 5 bps', 0.0005, 0), ('limit 30 bps', 0.0030, 0),
                ('15 bps, fill ask + 1 tick', LIMIT_BPS, 1)]
    results = {}
    for name, lb, xt in variants:
        log(f'[sim] {part}: {name}')
        results[name] = simulate_book(book, tapes, paths_idx, lb, xt)
    results['E1 15 bps (frozen)'].to_csv(OUT_CSV.format(part=part), index=False)
    cohorts = [('TRAIN-H2', lambda d: (d.split == 'TRAIN')), ('VAL', lambda d: d.split == 'VAL')] if part == 'val' \
        else [('TEST', lambda d: d.split == 'TEST')]
    scores = {}
    for name, res in results.items():
        for cn, sel in cohorts:
            scores[(name, cn)] = score(res[sel(res)], book[sel(book)])
    return scores, lost


def write_val_report(scores):
    """Write REPORT_1427.md with the agreement check (VAL vs cell 1,423) and the TRAIN-H2/VAL tables."""
    v = scores[('E1 15 bps (frozen)', 'VAL')]
    dR, dF = v['mean_R'] - REF_VAL_MEAN_R, v['fill_rate'] - REF_VAL_FILL_RATE
    ok = abs(dR) <= BAND_R and abs(dF) <= BAND_FILL
    lines = ['# REPORT — cell 1,427: independent SIP rebuild of E1 (PREREG_1427.md)', '',
             f'Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC} by research/hod_entry/sip_rebuild.py '
             '(written from the PREREG prose only; entry_replay.py / REPORT.md / hod_ofi/pipeline.py not read).', '',
             '## Step 2 — agreement check vs cell 1,423 VAL (frozen bands)', '',
             f'* VAL E1 mean net R (fills): **{v["mean_R"]:+.3f}** vs +0.384 → diff {dR:+.3f} (band ±0.05) '
             f'→ {"IN" if abs(dR) <= BAND_R else "OUT"}',
             f'* VAL fill rate (fills / usable tape): **{v["fill_rate"]*100:.1f} %** vs 29.9 % → diff '
             f'{dF*100:+.1f} pp (band ±10 pp) → {"IN" if abs(dF) <= BAND_FILL else "OUT"}',
             f'* {AGREEMENT_PASS_TOKEN if ok else "AGREEMENT VERDICT: FAIL — TEST NOT OPENED"}', '',
             '## TRAIN-H2 / VAL tables (SIP consolidated tape)', '', TABLE_HEAD]
    for (name, cn), s in scores.items():
        lines.append(fmt_row(f'{name} — {cn}', s))
    lines += ['', 'Cost: entry = half NBBO spread at the fill instant; exit = B0 per-signal half-spread + 2 bp. '
              'Fill rate = fills / usable-tape signals. Slots = run_consol.simulate_slots (first 12/day, 4 '
              'concurrent) ordered by fill minute.', '']
    with open(REPORT_MD, 'w') as f:
        f.write('\n'.join(lines))
    log(f'[report] VAL mean {v["mean_R"]:+.3f} (diff {dR:+.3f}), fill rate {v["fill_rate"]*100:.1f} % '
        f'(diff {dF*100:+.1f} pp) → {"PASS" if ok else "FAIL"}; wrote {REPORT_MD}')
    return ok


def append_test_report(scores):
    """Append the TEST pass-bar verdict leg by leg plus the report-only tables."""
    s = scores[('E1 15 bps (frozen)', 'TEST')]
    legs = [('mean net R ≥ +0.10', s['mean_R'] >= 0.10, f'{s["mean_R"]:+.3f}'),
            ('day-clustered t ≥ 2', s['t'] >= 2, f'{s["t"]:.2f} ({s["ndays"]} days)'),
            ('ex-top-5 % > 0', s['extop5'] > 0, f'{s["extop5"]:+.3f}'),
            ('coverage ≥ 80 %', s['coverage'] >= 0.80, f'{s["coverage"]*100:.1f} %'),
            ('winner/loser missingness gap ≤ 5 pp', s['miss_gap'] <= 0.05, f'{s["miss_gap"]*100:.1f} pp'),
            ('≥ 3 fills/week after slots', s['fills_wk'] >= 3, f'{s["fills_wk"]:.1f}'),
            ('TEST-opening condition', True, 'met by PREREG amendment 2026-09-25 (judged; SIP build is the reference)')]
    ok = all(x[1] for x in legs)
    lines = ['', '## Step 3 — TEST (sealed, read once)', '', '| leg | value | pass |', '|---|---|---|']
    lines += [f'| {n} | {v} | {"PASS" if p else "FAIL"} |' for n, p, v in legs]
    lines += ['', f'**TEST VERDICT: {"PASS" if ok else "FAIL"}**', '', TABLE_HEAD]
    for (name, cn), sc in scores.items():
        lines.append(fmt_row(f'{name} — {cn}', sc))
    with open(REPORT_MD, 'a') as f:
        f.write('\n'.join(lines) + '\n')
    log(f'[report] TEST {"PASS" if ok else "FAIL"}: ' + '; '.join(f'{n}={v}' for n, _, v in legs))
    return ok


def append_rerun_report(old_csv, new_csv):
    """Append the quote-history re-run section: coverage, missingness gap, E1 stats per split, and the same
    stats restricted to signals already usable in the S-65 s run (must be identical). No agreement token is
    written — opening TEST is the coordinator's decision."""
    old = pd.read_csv(old_csv, dtype={'day': str, 'symbol': str})
    new = pd.read_csv(new_csv, dtype={'day': str, 'symbol': str})
    key = ['day', 'symbol', 'entry_m']
    old_u = old[old.status.isin(['fill', 'nofill'])][key]
    lines = ['', '## Step 2c — re-run with the prevailing-quote history (quotes back to S-900 s, else 09:30)', '',
             'Only the quote fetch changed (`extend_quotes`: a signal whose cached window has no valid NBBO at or '
             'before S-60 s gets the last valid quote in [S-900 s, S-65 s), else in [09:30, S-900 s)). No rule '
             'constant or fill logic changed. TEST not opened.', '', TABLE_HEAD]
    checks = []
    for cn, sp in [('TRAIN-H2', 'TRAIN'), ('VAL', 'VAL')]:
        n = new[new.split == sp]
        lines.append(fmt_row(f'E1 15 bps, quote history — {cn} (all signals)', score(n, n)))
        nr = n.merge(old_u, on=key)
        orr = old[old.split == sp].merge(old_u, on=key)
        s_new, s_old = score(nr, n), score(orr, old[old.split == sp])
        lines.append(fmt_row(f'same, restricted to the {len(nr)} previously-usable — {cn}', s_new))
        lines.append(fmt_row(f'S-65 s run, same {len(orr)} signals — {cn}', s_old))
        same = (nr.sort_values(key).status.values == orr.sort_values(key).status.values).all() and \
            np.allclose(nr.sort_values(key).net_R.fillna(0).values, orr.sort_values(key).net_R.fillna(0).values)
        checks.append(f'{cn}: previously-usable signals identical per signal (status and net R): '
                      f'{"YES" if same else "NO — INVESTIGATE"}')
        if sp == 'VAL':
            v = score(n, n)
            checks.append(f'VAL vs cell 1,423 bands (informational): mean {v["mean_R"]:+.3f} vs +0.384 '
                          f'(diff {v["mean_R"] - REF_VAL_MEAN_R:+.3f}), fill rate {v["fill_rate"]*100:.1f} % vs 29.9 % '
                          f'(diff {(v["fill_rate"] - REF_VAL_FILL_RATE)*100:+.1f} pp)')
    lines += [''] + [f'* {c}' for c in checks] + ['']
    with open(REPORT_MD, 'a') as f:
        f.write('\n'.join(lines) + '\n')
    for c in checks:
        log(f'[rerun] {c}')


def main(argv=None):
    """CLI entry: --part val|test, --workers N."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--part', choices=['val', 'test'], required=True)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--rerun-append', metavar='OLD_CSV', default=None,
                    help='val only: append a re-run section comparing against OLD_CSV instead of rewriting the report')
    a = ap.parse_args(argv)
    if a.part == 'test':
        prereg = open(os.path.join(HERE, 'PREREG_1427.md')).read()
        judged = 'TEST is opened ONCE with the SIP build' in ' '.join(prereg.split())      # judge amendment 2026-09-25
        if not judged and (not os.path.exists(REPORT_MD) or AGREEMENT_PASS_TOKEN not in open(REPORT_MD).read()):
            log('[test] ERROR: the VAL agreement has not PASSED and no judge amendment — TEST stays sealed')
            return 2
        if '## Step 3' in open(REPORT_MD).read():
            log('[test] ERROR: TEST was already read once — refusing a second read')
            return 2
    scores, _ = run(a.part, a.workers)
    if a.part == 'val' and a.rerun_append:
        append_rerun_report(a.rerun_append, OUT_CSV.format(part='val'))
        return 0
    if a.part == 'val':
        return 0 if write_val_report(scores) else 1
    append_test_report(scores)
    return 0


if __name__ == '__main__':
    sys.exit(main())
