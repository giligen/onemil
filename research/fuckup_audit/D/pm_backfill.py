#!/usr/bin/env python3
"""Stage D1 — premarket (04:00-09:29 ET) dollar volume per candidate symbol-day.

WHY (PLAN.md H4): the ORB book's strongest validated separator is
"premarket NEWS present AND premarket DOLLAR VOLUME above a cut"
(research/orb_machine_rules.md, cut $5,816,688). Stage D needs that feature
for the F6/F8 candidate table. research/bf_zero/bars_sip.db holds 04:00-09:29
bars for only part of the keys (it was fetched 04:00-20:00 ET, but only for
the symbol-days the parity review needed); the rest are backfilled from
Alpaca REST (MARKET DATA ONLY — no trading call is ever made).

KEYS: the union of (day, symbol) in research/fuckup_audit/D/table.csv
      (already exactly the F6/F8 entry_m >= 600 candidate set, 61,902 keys).

CONTRACT
  src='sip_store'  the store already had >= 1 bar with t < 09:30 ET -> computed
                   from the store, no API call.
  src='alpaca'     fetched from Alpaca (SIP, 1-min, adjustment raw, 04:00-09:29 ET),
                   >= 1 premarket bar returned.
  src='none'       Alpaca returned NOTHING for the window. A genuine
                   no-premarket-trades day. n_pm_bars = 0, pm_* = NULL.
                   This is DIFFERENT from a fetch failure: failures are logged in
                   fetch_log(status='error') and are NOT written to pm, so they are
                   retried (once at the end, then on any later resume).

  pm_dollar_vol = sum(close * volume) over premarket bars. The live helper
  trading/orb_pm_mult.compute_pm_dollar_vol prefers the bar's vwap when present
  and falls back to close; bars_sip.db carries no vwap column, so CLOSE is used
  for BOTH sources here to keep the two provenances comparable (per-minute
  difference is immaterial against a $5.8M cut, as that helper's docstring says).
  pm_vwap = pm_dollar_vol / pm_volume (NULL when pm_volume == 0).

DST: 09:30 ET is 13:30Z in summer and 14:30Z in winter. Every boundary is built
with zoneinfo('America/New_York') per day, never with a fixed offset. The store's
t column is ISO UTC ('2025-06-02T13:30:00+00:00', fixed 25-char format), so a
lexicographic string range is exact.

RESUMABLE: research/fuckup_audit/D/pm_backfill_state.json records the days already
done per phase; pm_bars.db rows are the real state (PRIMARY KEY (symbol, day)).
Re-running picks up where it stopped.

RUN (detached, per PLAN.md §1 — one nice'd, memory-capped process):
  setsid nohup bash -c "cd /home/ec2-user/onemil; ulimit -v 1000000; nice -n 10 \
    python3 research/fuckup_audit/D/pm_backfill.py > research/fuckup_audit/D/pm_backfill.log 2>&1; \
    echo EXIT=\$? >> research/fuckup_audit/D/pm_backfill.log" >/dev/null 2>&1 </dev/null &

Writes ONLY research/fuckup_audit/D/{pm_bars.db, pm_backfill_state.json,
pm_summary.md, pm_backfill.log}. bars_sip.db is opened read-only.
"""
import csv
import json
import os
import random
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import logging
logging.basicConfig(level=logging.ERROR)

ET = ZoneInfo('America/New_York')
D = f'{ROOT}/research/fuckup_audit/D'
KEYS_CSV = f'{D}/table.csv'
SIP_STORE = f'{ROOT}/research/bf_zero/bars_sip.db'
OUT_DB = f'{D}/pm_bars.db'
STATE = f'{D}/pm_backfill_state.json'
SUMMARY = f'{D}/pm_summary.md'

FETCH_BATCH = 200            # symbols per Alpaca request
STORE_CHUNK = 400            # symbols per sqlite IN(...) query
MIN_REQ_INTERVAL = 0.34      # <= ~176 req/min, under Alpaca's 200/min
PM_CUT_USD = 5_816_688.0     # trading/orb_pm_mult.DEFAULT_HIGH_CUT_USD
SYM_OK = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')

_last_req = [0.0]


def log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


# ---------------------------------------------------------------- state / db

def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {'store_days': [], 'alpaca_days': [], 'retry_done': False}


def save_state(st):
    tmp = STATE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(st, f)
    os.replace(tmp, STATE)


def open_out():
    con = sqlite3.connect(OUT_DB, timeout=120)
    con.execute("""create table if not exists pm (
        symbol TEXT, day TEXT, n_pm_bars INT, pm_volume REAL, pm_dollar_vol REAL,
        pm_high REAL, pm_low REAL, pm_last REAL, pm_vwap REAL, src TEXT,
        PRIMARY KEY (symbol, day))""")
    con.execute("""create table if not exists fetch_log (
        symbol TEXT, day TEXT, status TEXT, detail TEXT, ts TEXT,
        PRIMARY KEY (symbol, day))""")
    con.execute("create index if not exists idx_pm_day on pm(day)")
    con.commit()
    return con


# ---------------------------------------------------------------- boundaries

def pm_window_utc(day):
    """(start, end) ISO-UTC strings for 04:00:00 and 09:30:00 ET on `day`.

    DST-correct by construction (zoneinfo, never a fixed offset).
    Returns strings in the store's exact format so a lexicographic
    string range on `t` is an exact time range.
    """
    d = datetime.strptime(day, '%Y-%m-%d')
    s = datetime(d.year, d.month, d.day, 4, 0, tzinfo=ET).astimezone(timezone.utc)
    e = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET).astimezone(timezone.utc)
    return s.isoformat(), e.isoformat(), s, e


def agg(bars):
    """bars: list of (t, h, l, c, v) sorted by t -> the pm row fields."""
    n = len(bars)
    if n == 0:
        return None
    vol = sum(b[4] for b in bars)
    dol = sum(b[3] * b[4] for b in bars)
    hi = max(b[1] for b in bars)
    lo = min(b[2] for b in bars)
    last = bars[-1][3]
    vwap = (dol / vol) if vol > 0 else None
    return (n, vol, dol, hi, lo, last, vwap)


# ---------------------------------------------------------------- phase 1: store

def phase_store(keys_by_day, con, st):
    """Compute pm from bars_sip.db for every key that already has a pre-09:30 bar.

    Returns {day: [symbols still to fetch]}.
    """
    src = sqlite3.connect(f'file:{SIP_STORE}?mode=ro', uri=True, timeout=120)
    done_days = set(st['store_days'])
    have = set(r[0] for r in con.execute("select symbol || '|' || day from pm").fetchall())
    todo_by_day = {}
    total_keys = sum(len(v) for v in keys_by_day.values())
    days = sorted(keys_by_day)
    t0 = time.time()
    seen = 0
    n_store = 0
    for i, day in enumerate(days):
        syms = keys_by_day[day]
        seen += len(syms)
        if day in done_days:
            # rebuild the todo list from what is missing in pm
            todo = [s for s in syms if f'{s}|{day}' not in have]
            if todo:
                todo_by_day[day] = todo
            continue
        s_iso, e_iso, _, _ = pm_window_utc(day)
        found = {}
        for j in range(0, len(syms), STORE_CHUNK):
            chunk = syms[j:j + STORE_CHUNK]
            q = ("select symbol, t, h, l, c, v from bars where day=? and t>=? and t<? "
                 f"and symbol in ({','.join('?' * len(chunk))}) order by symbol, t")
            for sym, t, h, l, c, v in src.execute(q, [day, s_iso, e_iso] + chunk):
                found.setdefault(sym, []).append((t, h, l, c, v))
        rows = []
        for sym, bars in found.items():
            a = agg(bars)
            if a:
                rows.append((sym, day, *a, 'sip_store'))
        if rows:
            con.executemany("insert or replace into pm values (?,?,?,?,?,?,?,?,?,?)", rows)
            con.commit()
            n_store += len(rows)
            have.update(f'{r[0]}|{day}' for r in rows)
        todo = [s for s in syms if s not in found and f'{s}|{day}' not in have]
        if todo:
            todo_by_day[day] = todo
        st['store_days'].append(day)
        if i % 25 == 0 or i == len(days) - 1:
            save_state(st)
        if seen % 500 < len(syms) or i == len(days) - 1:
            el = time.time() - t0
            rate = seen / el if el else 0
            eta = (total_keys - seen) / rate / 60 if rate else 0
            log(f'[store] day {i + 1}/{len(days)} {day} | keys {seen}/{total_keys} '
                f'| from store {n_store} | {el / 60:.1f} min elapsed, ETA {eta:.1f} min')
    save_state(st)
    src.close()
    n_todo = sum(len(v) for v in todo_by_day.values())
    log(f'[store] DONE: {n_store} keys served by bars_sip.db, {n_todo} keys need Alpaca '
        f'over {len(todo_by_day)} days')
    return todo_by_day


# ---------------------------------------------------------------- alpaca

def alpaca_client():
    from config import Config
    from data_sources.alpaca_client import AlpacaClient
    cfg = Config()
    return AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)


def throttle():
    dt = time.time() - _last_req[0]
    if dt < MIN_REQ_INTERVAL:
        time.sleep(MIN_REQ_INTERVAL - dt)
    _last_req[0] = time.time()


def fetch_pm_day(client, day, symbols):
    """{symbol: [(t, h, l, c, v)]} for 04:00-09:29:59 ET on `day`. MARKET DATA ONLY.

    Raises on unrecoverable failure so the caller can log the keys as errors.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed, Adjustment
    _, _, start, end = pm_window_utc(day)
    end = end - timedelta(seconds=1)          # 09:29:59 ET — never an RTH bar
    out = {}
    for i in range(0, len(symbols), FETCH_BATCH):
        chunk = [s for s in symbols[i:i + FETCH_BATCH] if SYM_OK.match(s)]
        raw = {}
        for k, wait in enumerate((0, 5, 20, 60, 120)):
            if wait:
                time.sleep(wait)
            if not chunk:
                break
            throttle()
            req = StockBarsRequest(symbol_or_symbols=chunk,
                                   timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                   start=start, end=end,
                                   feed=DataFeed.SIP, adjustment=Adjustment.RAW)
            try:
                raw = client._to_dict(client.data_client.get_stock_bars(req))
                break
            except Exception as e:
                msg = str(e)
                m = re.search(r'invalid symbol: ([A-Z0-9.\-]+)', msg)
                if m and m.group(1) in chunk:   # one bad symbol 400s the batch
                    log(f'  {day}: Alpaca rejects {m.group(1)} — dropped')
                    chunk = [s for s in chunk if s != m.group(1)]
                    continue
                if '429' in msg or 'rate limit' in msg.lower():
                    log(f'  {day} chunk {i // FETCH_BATCH}: 429 — backing off 30s')
                    time.sleep(30)
                    continue
                log(f'  {day} chunk {i // FETCH_BATCH} try {k + 1} failed: {msg[:200]}')
                if k == 4:
                    raise
        for s in chunk:
            bars = raw.get(s) or []
            rows = []
            for b in bars:
                t = b.timestamp.astimezone(ET)
                if (t.hour, t.minute) >= (9, 30) or t.hour < 4:
                    continue                    # belt and braces on the window
                rows.append((b.timestamp.astimezone(timezone.utc).isoformat(),
                             float(b.high), float(b.low), float(b.close), float(b.volume)))
            out[s] = rows
    return out


def phase_alpaca(todo_by_day, con, st, tag='alpaca'):
    done_days = set(st['alpaca_days']) if tag == 'alpaca' else set()
    days = sorted(todo_by_day)
    total = sum(len(v) for v in todo_by_day.values())
    if not total:
        log(f'[{tag}] nothing to fetch')
        return
    client = alpaca_client()
    t0 = time.time()
    seen = n_hit = n_none = n_err = 0
    for i, day in enumerate(days):
        syms = sorted(todo_by_day[day])
        seen += len(syms)
        if day in done_days:
            continue
        now = datetime.now(timezone.utc).isoformat()
        try:
            got = fetch_pm_day(client, day, syms)
        except Exception as e:
            n_err += len(syms)
            con.executemany("insert or replace into fetch_log values (?,?,?,?,?)",
                            [(s, day, 'error', str(e)[:300], now) for s in syms])
            con.commit()
            log(f'[{tag}] {day}: FETCH FAILED for {len(syms)} keys: {str(e)[:200]}')
            continue
        rows = []
        for s in syms:
            if s not in got:                     # symbol never made it into a request
                n_err += 1
                con.execute("insert or replace into fetch_log values (?,?,?,?,?)",
                            (s, day, 'error', 'symbol rejected/not requested', now))
                continue
            a = agg(got[s])
            if a:
                rows.append((s, day, *a, 'alpaca'))
                n_hit += 1
            else:
                rows.append((s, day, 0, None, None, None, None, None, None, 'none'))
                n_none += 1
        if rows:
            con.executemany("insert or replace into pm values (?,?,?,?,?,?,?,?,?,?)", rows)
            # a key that now has a pm row is no longer a failure
            con.execute("delete from fetch_log where day=? and symbol in "
                        f"({','.join('?' * len(rows))})", [day] + [r[0] for r in rows])
        con.commit()
        if tag == 'alpaca':
            st['alpaca_days'].append(day)
            if i % 10 == 0:
                save_state(st)
        if seen % 500 < len(syms) or i == len(days) - 1:
            el = time.time() - t0
            rate = seen / el if el else 0
            eta = (total - seen) / rate / 60 if rate else 0
            log(f'[{tag}] day {i + 1}/{len(days)} {day} | keys {seen}/{total} '
                f'| alpaca {n_hit} none {n_none} err {n_err} '
                f'| {el / 60:.1f} min elapsed, ETA {eta:.1f} min')
    if tag == 'alpaca':
        save_state(st)
    log(f'[{tag}] DONE: alpaca {n_hit}, none {n_none}, errors {n_err}')


# ---------------------------------------------------------------- summary

def quantiles(vals, qs):
    v = sorted(vals)
    out = []
    for q in qs:
        if not v:
            out.append(float('nan'))
            continue
        k = (len(v) - 1) * q
        f = int(k)
        c = min(f + 1, len(v) - 1)
        out.append(v[f] + (v[c] - v[f]) * (k - f))
    return out


def spot_check(con, n=20):
    """n keys served by the store, re-fetched fresh from Alpaca — bar-exact?"""
    keys = con.execute("select symbol, day from pm where src='sip_store'").fetchall()
    random.seed(7)
    random.shuffle(keys)
    keys = keys[:n]
    try:
        client = alpaca_client()
    except Exception as e:
        return [], f'spot check skipped: Alpaca client unavailable ({e})'
    src = sqlite3.connect(f'file:{SIP_STORE}?mode=ro', uri=True, timeout=120)
    out = []
    for sym, day in keys:
        s_iso, e_iso, _, _ = pm_window_utc(day)
        st_rows = src.execute("select t, h, l, c, v from bars where symbol=? and day=? "
                              "and t>=? and t<? order by t", (sym, day, s_iso, e_iso)).fetchall()
        try:
            got = fetch_pm_day(client, day, [sym]).get(sym, [])
        except Exception as e:
            out.append((sym, day, len(st_rows), -1, 'FETCH FAILED ' + str(e)[:60]))
            continue
        a_st, a_al = agg(st_rows), agg(got)
        if a_al is None:
            out.append((sym, day, len(st_rows), 0, 'alpaca returned nothing'))
            continue
        same_n = a_st[0] == a_al[0]
        # bar-exact: same timestamps and same OHLCV-relevant fields
        d_st = {r[0]: r[1:] for r in st_rows}
        d_al = {r[0]: r[1:] for r in got}
        exact = d_st.keys() == d_al.keys() and all(
            all(abs(x - y) <= 1e-6 * max(1.0, abs(x)) for x, y in zip(d_st[k], d_al[k]))
            for k in d_st)
        dv = (a_al[2] - a_st[2]) / a_st[2] * 100 if a_st[2] else 0.0
        out.append((sym, day, a_st[0], a_al[0],
                    'BAR-EXACT' if exact else
                    (f'same n, fields differ (pm$ {dv:+.3f}%)' if same_n
                     else f'n differs (pm$ {dv:+.3f}%)')))
    src.close()
    return out, None


def write_summary(con):
    by_src = dict(con.execute("select src, count(*) from pm group by src").fetchall())
    total = sum(by_src.values())
    errs = con.execute("select count(*) from fetch_log where status='error'").fetchall()[0][0]
    dv = [r[0] for r in con.execute(
        "select pm_dollar_vol from pm where pm_dollar_vol is not null").fetchall()]
    qs = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    qv = quantiles(dv, qs)
    above = sum(1 for x in dv if x > PM_CUT_USD)
    # by src, for the two provenances separately
    per_src = {}
    for s in ('sip_store', 'alpaca'):
        v = [r[0] for r in con.execute(
            "select pm_dollar_vol from pm where src=? and pm_dollar_vol is not null", (s,)).fetchall()]
        per_src[s] = (len(v), sum(1 for x in v if x > PM_CUT_USD),
                      quantiles(v, [0.5, 0.9])) if v else (0, 0, [float('nan')] * 2)
    spot, spot_err = spot_check(con)

    L = []
    L.append('# Stage D1 — premarket dollar volume per candidate symbol-day\n')
    L.append(f'Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")} by '
             '`research/fuckup_audit/D/pm_backfill.py`.\n')
    L.append('**What this is.** One row per (day, symbol) of the Stage-D candidate table '
             '(`D/table.csv`, F6/F8, entry_m >= 600): premarket 04:00-09:29 ET aggregates '
             'from the 1-min SIP tape. `pm_dollar_vol = sum(close x volume)` (the live helper '
             '`trading/orb_pm_mult.compute_pm_dollar_vol` prefers bar vwap and falls back to '
             'close; `bars_sip.db` carries no vwap column, so CLOSE is used for BOTH '
             'provenances here so they are comparable). 09:30 ET boundaries are built per day '
             'with `zoneinfo`, so DST is exact.\n')
    L.append('## Coverage\n')
    L.append('| src | meaning | keys | share |')
    L.append('|---|---|---:|---:|')
    meaning = {'sip_store': 'already in research/bf_zero/bars_sip.db (no API call)',
               'alpaca': 'fetched from Alpaca SIP, >= 1 premarket bar',
               'none': 'Alpaca returned nothing — genuine no-premarket-trades day'}
    for s in ('sip_store', 'alpaca', 'none'):
        n = by_src.get(s, 0)
        L.append(f'| `{s}` | {meaning[s]} | {n:,} | {n / total * 100:.1f}% |' if total else '')
    L.append(f'| **total** | rows in `pm` | **{total:,}** | |')
    L.append('')
    L.append(f'Unresolved fetch failures still in `fetch_log(status=\'error\')`: **{errs:,}** '
             '(these have NO `pm` row; re-running the script retries them).\n')
    L.append('## pm_dollar_vol distribution (keys with >= 1 premarket bar)\n')
    L.append(f'n = {len(dv):,}\n')
    L.append('| quantile | pm_dollar_vol ($) |')
    L.append('|---|---:|')
    for q, v in zip(qs, qv):
        L.append(f'| p{q * 100:g} | {v:,.0f} |')
    L.append('')
    L.append(f'**Share above the ORB cut ($5,816,688): {above:,} / {len(dv):,} = '
             f'{above / len(dv) * 100:.2f}%** of keys with premarket trades '
             f'({above / total * 100:.2f}% of all {total:,} keys).\n')
    L.append('| src | n with pm bars | above cut | share | median | p90 |')
    L.append('|---|---:|---:|---:|---:|---:|')
    for s in ('sip_store', 'alpaca'):
        n, a, m = per_src[s]
        L.append(f'| `{s}` | {n:,} | {a:,} | {a / n * 100:.2f}% | {m[0]:,.0f} | {m[1]:,.0f} |'
                 if n else f'| `{s}` | 0 | 0 | — | — | — |')
    L.append('')
    L.append('## Spot check — 20 store-served keys re-fetched fresh from Alpaca\n')
    if spot_err:
        L.append(spot_err + '\n')
    else:
        L.append('| symbol | day | n bars (store) | n bars (alpaca) | verdict |')
        L.append('|---|---|---:|---:|---|')
        for sym, day, n_st, n_al, verdict in spot:
            L.append(f'| {sym} | {day} | {n_st} | {n_al} | {verdict} |')
        ok = sum(1 for r in spot if r[4] == 'BAR-EXACT')
        L.append('')
        L.append(f'**{ok}/{len(spot)} bar-exact** (identical timestamp set and identical '
                 'h/l/c/v to 1e-6 relative).\n')
    L.append('## Caveats\n')
    L.append('- `src=\'none\'` is "Alpaca served no 04:00-09:29 ET bar for this symbol-day". '
             'For a thin small-cap that is the normal case, not an error; a fetch error is '
             'recorded separately and never becomes a `none`.\n')
    L.append('- Keys already in the store were NOT re-fetched, so their tape is whatever '
             '`bars_sip.db` holds (SIP, per the 2026-09-15 parity review). The spot check above '
             'is the evidence for treating the two provenances as one series.\n')
    L.append('- Bars are `adjustment=raw`. A split between the bar date and today makes '
             'pm_high/low/last raw-price, consistent with the intraday tape used elsewhere in '
             'this tree, but NOT comparable to an adjusted daily file (PLAN.md §1 price-scale '
             'rule).\n')
    with open(SUMMARY, 'w') as f:
        f.write('\n'.join(L) + '\n')
    log(f'wrote {SUMMARY}')


# ---------------------------------------------------------------- main

def load_keys():
    keys_by_day = {}
    seen = set()
    with open(KEYS_CSV, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            if row['fam'] not in ('F6', 'F8'):
                continue
            try:
                if int(row['entry_m']) < 600:
                    continue
            except (ValueError, KeyError):
                continue
            k = (row['day'], row['symbol'])
            if k in seen:
                continue
            seen.add(k)
            keys_by_day.setdefault(row['day'], []).append(row['symbol'])
    for d in keys_by_day:
        keys_by_day[d].sort()
    log(f'keys: {len(seen):,} (day, symbol) over {len(keys_by_day)} days')
    return keys_by_day


def main():
    t0 = time.time()
    keys_by_day = load_keys()
    st = load_state()
    con = open_out()

    todo = phase_store(keys_by_day, con, st)
    phase_alpaca(todo, con, st, tag='alpaca')

    # retry every key still logged as an error, once
    errs = con.execute("select symbol, day from fetch_log where status='error'").fetchall()
    if errs:
        log(f'[retry] {len(errs)} keys failed — one more pass')
        rt = {}
        for s, d in errs:
            rt.setdefault(d, []).append(s)
        phase_alpaca(rt, con, st, tag='retry')
        st['retry_done'] = True
        save_state(st)

    write_summary(con)
    con.close()
    log(f'ALL DONE in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
