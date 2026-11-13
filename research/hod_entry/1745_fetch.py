"""PREREG 1,745 fetch (agent A): 1-min SIP bars for the monthly top-300 liquid US common stocks, 2026-01-02 .. 2026-11-12.

Universe (knowable at 09:30 ET of the day): on the first trading day of each month, rank symbols of `data/cache.db`
daily_bars (read-only) by the mean dollar volume (close*volume) of the 20 sessions strictly BEFORE that day, keep
price (prior close) >= $5, drop symbols the BF/live predicate `trading.bf_universe_filter.is_bf_eligible` rejects
(ETF/ETN/wrapper/warrant/unit/preferred/right by NAME), drop symbols with NO known asset name (strict: unknown names
could be ETFs), drop test tickers ^Z[A-Z]ZZT$, take the top 300.  Written to 1745_universe.csv (month, symbol, adv20, price).

Bars: Alpaca GET /v2/stocks/bars (data.alpaca.markets), timeframe 1Min, feed sip, adjustment raw, 09:30..16:00 ET,
limit 10000 per page, symbols batched; pages followed with next_page_token.  One parquet per day in 1745_bars/
(columns symbol, t (UTC), o h l c v n vw), atomic write (tmp + rename).  Resume: days whose parquet exists are skipped.
Completeness gate (feedback_fetch_completeness_gate): per day, LOST = universe symbols with < 300 bars; a day with
< 95 % of the universe at >= 300 bars is VOID: logged ERROR, NO parquet written (so a rerun retries it).
Rate: strict token spacing 0.34 s (<= 176 requests/min).  One writer only.  Read-only on every existing store.
Usage: nice -n 19 python3 1745_fetch.py [--only-day YYYY-MM-DD] [--build-universe-only]
"""
import argparse, csv, os, re, shutil, sqlite3, sys, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
BARS_DIR = os.path.join(HERE, '1745_bars')
UNIVERSE = os.path.join(HERE, '1745_universe.csv')
URL = 'https://data.alpaca.markets/v2/stocks/bars'
ET = ZoneInfo('America/New_York')
START_DAY, END_DAY = '2026-01-02', '2026-11-12'
TOP_N, MIN_PRICE, ADV_WINDOW = 300, 5.0, 20
BATCH = 300                 # symbols per request group (pages follow next_page_token)
MIN_BARS, MIN_OK_SHARE = 300, 0.95
MIN_FREE_GB = 5.0
MIN_INTERVAL = 0.34
TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')
# The live BF predicate drops only LEVERAGED wrappers; PREREG 1,745 also excludes plain ETFs/ETNs/grantor trusts/funds,
# so the same name dump is screened with these fund markers (not 'SHARES' alone: ADRs are 'Depositary Shares').
FUND_NAME = re.compile(r'\bETF\b|\bETN\b|\bETFS\b|EXCHANGE[- ]TRADED|\bFUND\b|ISHARES|SPDR|VANGUARD|INVESCO|PROSHARES|'
                       r'GLOBAL X|VANECK|ROUNDHILL|KRANESHARES|SCHWAB U\.?S|ARK |GOLD TRUST|SILVER TRUST|BITCOIN TRUST|'
                       r'ETHEREUM TRUST|PHYSICAL|MINISHARES|GOLD SHARES|QQQ TRUST|SERIES 1\b|INDEX|SELECT SECTOR|'
                       r'GRAYSCALE|WISDOMTREE|DIREXION|T-BILL|TREASURY|PORTFOLIO', re.I)
STATS = {'net': 0, 'r429': 0}
_last = [0.0]


def free_gb():
    """Free GB on the filesystem holding the bars."""
    return shutil.disk_usage(HERE).free / 1e9


def headers():
    """Auth headers (never printed)."""
    load_dotenv(os.path.join(ROOT, '.env'))
    return {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'], 'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']}


def db():
    """Read-only handle on data/cache.db."""
    return sqlite3.connect('file:' + os.path.join(ROOT, 'data', 'cache.db') + '?mode=ro', uri=True)


def trading_days():
    """SPY trading dates within the period from daily_bars."""
    c = db()
    try:
        return [r[0] for r in c.execute("select bar_date from daily_bars where symbol='SPY' and bar_date>=? and bar_date<=? "
                                        "order by bar_date", (START_DAY, END_DAY))]
    finally:
        c.close()


def month_starts(days):
    """First trading day of each month present in `days`."""
    seen, out = set(), []
    for d in days:
        if d[:7] not in seen:
            seen.add(d[:7]); out.append(d)
    return out


def build_universe(days):
    """Monthly top-N by ADV20 (prior 20 sessions), price >= MIN_PRICE, eligible common stock only. Returns DataFrame."""
    from trading.bf_universe_filter import is_bf_eligible, load_names
    names = load_names()
    c = db()
    rows = []
    try:
        for first in month_starts(days):
            prior = [r[0] for r in c.execute("select distinct bar_date from daily_bars where symbol='SPY' and bar_date<? "
                                             "order by bar_date desc limit ?", (first, ADV_WINDOW))]
            lo = prior[-1]
            df = pd.read_sql_query("select symbol, bar_date, close, volume from daily_bars where bar_date>=? and bar_date<?",
                                   c, params=(lo, first))
            df['dv'] = df.close * df.volume
            g = df.groupby('symbol').agg(adv20=('dv', 'mean'), n=('dv', 'size'))
            last = df.sort_values('bar_date').groupby('symbol').close.last().rename('price')
            g = g.join(last)
            g = g[(g.n >= ADV_WINDOW - 2) & (g.price >= MIN_PRICE)].sort_values('adv20', ascending=False)
            kept, no_name, bad = [], 0, 0
            for sym, r in g.iterrows():
                if TEST_TICKER.match(sym) or not re.match(r'^[A-Z]{1,5}$', sym):
                    bad += 1; continue
                if sym not in names:
                    no_name += 1; continue
                if not is_bf_eligible(sym, names) or FUND_NAME.search(names[sym]):
                    bad += 1; continue
                kept.append((first[:7], sym, round(float(r.adv20), 0), round(float(r.price), 2)))
                if len(kept) >= TOP_N:
                    break
            print(f'universe {first[:7]} first_day={first} window={lo}..{prior[0]} kept={len(kept)} '
                  f'dropped_ineligible={bad} dropped_noname={no_name}', flush=True)
            rows += kept
    finally:
        c.close()
    out = pd.DataFrame(rows, columns=['month', 'symbol', 'adv20', 'price'])
    tmp = UNIVERSE + '.tmp'
    out.to_csv(tmp, index=False)
    os.replace(tmp, UNIVERSE)
    return out


def utc_iso(day, hh, mm):
    """ET wall time of `day` as an RFC3339 UTC string."""
    dt = datetime.fromisoformat(day).replace(hour=hh, minute=mm, tzinfo=ET)
    return dt.astimezone(ZoneInfo('UTC')).strftime('%Y-%m-%dT%H:%M:%SZ')


def get(params):
    """One GET with strict spacing and 429/5xx back-off; any other failure raises (no silent fallback)."""
    delay = 2.0
    for _ in range(8):
        wait = MIN_INTERVAL - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        STATS['net'] += 1
        r = requests.get(URL, headers=headers(), params=params, timeout=90)
        if r.status_code == 429 or r.status_code >= 500:
            STATS['r429'] += 1
            print(f'WARNING HTTP {r.status_code}; backoff {delay}s', flush=True)
            time.sleep(delay); delay = min(delay * 2, 60); continue
        if r.status_code != 200:
            raise RuntimeError(f'HTTP {r.status_code} {r.text[:200]}')
        return r.json()
    raise RuntimeError('gave up after retries')


def fetch_day(day, symbols):
    """All 1-min SIP bars 09:30..16:00 ET for `symbols` on `day` -> DataFrame."""
    recs = []
    for i in range(0, len(symbols), BATCH):
        tok = None
        while True:
            p = dict(symbols=','.join(symbols[i:i + BATCH]), timeframe='1Min', start=utc_iso(day, 9, 30), end=utc_iso(day, 16, 0),
                     limit=10000, feed='sip', adjustment='raw', sort='asc')
            if tok:
                p['page_token'] = tok
            j = get(p)
            for sym, bl in (j.get('bars') or {}).items():
                for b in bl:
                    recs.append((sym, b['t'], b['o'], b['h'], b['l'], b['c'], b['v'], b.get('n', 0), b.get('vw', 0.0)))
            tok = j.get('next_page_token')
            if not tok:
                break
    df = pd.DataFrame(recs, columns=['symbol', 't', 'o', 'h', 'l', 'c', 'v', 'n', 'vw'])
    df['t'] = pd.to_datetime(df.t, utc=True)
    return df


def gate(df, symbols):
    """(ok, lost, void): ok = symbols with >= MIN_BARS bars; lost = the rest; void if ok share < MIN_OK_SHARE."""
    counts = df.groupby('symbol').size() if len(df) else pd.Series(dtype=int)
    ok = int(sum(counts.get(s, 0) >= MIN_BARS for s in symbols))
    return ok, len(symbols) - ok, ok < MIN_OK_SHARE * len(symbols)


def main():
    """Build the universe, then fetch every day with resume, gate, progress and disk guard."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--only-day', default='')
    ap.add_argument('--build-universe-only', action='store_true')
    a = ap.parse_args()
    if free_gb() < MIN_FREE_GB:
        print(f'ERROR df {free_gb():.1f} GB < {MIN_FREE_GB}; refusing to start', flush=True); sys.exit(2)
    os.makedirs(BARS_DIR, exist_ok=True)
    days = trading_days()
    uni = build_universe(days) if (a.build_universe_only or not os.path.exists(UNIVERSE)) else pd.read_csv(UNIVERSE)
    print(f'universe rows={len(uni)} per month: {uni.groupby("month").size().to_dict()}', flush=True)
    if a.build_universe_only:
        return
    todo = [a.only_day] if a.only_day else days
    by_month = {m: sorted(g.symbol) for m, g in uni.groupby('month')}
    t0, n_done, tot_ok, tot_lost, voids, syms_seen = time.time(), 0, 0, 0, 0, set()
    for day in todo:
        path = os.path.join(BARS_DIR, f'{day}.parquet')
        if os.path.exists(path):
            continue
        if free_gb() < MIN_FREE_GB:
            print(f'ERROR free disk {free_gb():.1f} GB < {MIN_FREE_GB}; stopping (resume later)', flush=True); sys.exit(4)
        syms = by_month[day[:7]]
        df = fetch_day(day, syms)
        ok, lost, void = gate(df, syms)
        tot_ok += ok; tot_lost += lost; syms_seen.update(syms)
        n_done += 1
        if void:
            voids += 1
            print(f'ERROR VOID day={day} ok={ok}/{len(syms)} lost={lost}: no parquet written (rerun retries)', flush=True)
        else:
            tmp = path + '.tmp'
            df.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        left = len([d for d in todo if not os.path.exists(os.path.join(BARS_DIR, f'{d}.parquet'))])
        eta = (time.time() - t0) / n_done * left / 60
        print(f'day={day} symbols_ok={ok} lost={lost} rows={len(df)} requests={STATS["net"]} r429={STATS["r429"]} '
              f'free={free_gb():.0f}GB eta={eta:.0f}min', flush=True)
    print(f'DONE days={n_done} symbols={len(syms_seen)} lost={tot_lost} void_days={voids} requests={STATS["net"]}', flush=True)


if __name__ == '__main__':
    main()
