"""PREREG 1,741 fetch: pre-break option-flow aggregates per (symbol, day) for the HOD-break signals (TRAIN + VAL only).

Per symbol-day (de-duplicated on (symbol, day); one row each in 1741_flow.csv):
  1. Chain: contracts on the underlying with expiration in (day, day+45] and strike within +-20 % of `entry`, calls and
     puts, status inactive (expired) + active (if any could still be live). Empty chain -> covered=False, no more calls.
  2. Bars: 1-min option TRADE bars (Alpaca historical, NOT quotes) for those contracts, in batches of BATCH symbols:
       - today: 09:30 ET .. minute label entry_m - 2 INCLUSIVE.  Convention: `entry_m` is the stored entry minute, the
         BREAK bar is entry_m - 1 (as in cell 1,740), so the last fully pre-break 1-min bar is the one labelled
         entry_m - 2 (it covers [entry_m-2, entry_m-1)).  Nothing from the break bar or later is read.
       - prior trading day: full session 09:30..16:00 ET (prior day = previous SPY row of data/cache.db daily_bars).
  3. Row: covered, n_contracts, call_vol, put_vol (contracts), call_notional, put_notional (sum v*c*100, c = bar close),
     n_prints (sum of bar trade counts `n`), prior_day_vol (contracts, full session), first_print_minute (ET minute of
     day of the earliest pre-break bar, blank if none).

Reuses 1738_fetch.py helpers (headers, cached_get, paged) with CACHE pointed at 1741_cache/.  Atomic cache writes live in
cached_get (tmp + os.replace); csv rows are appended one at a time and flushed; resume skips (symbol, day) already in the
csv.  One writer only.  Rate: one request per 0.34 s (<= ~176/min, a strict token spacing).  Read-only endpoints only.
Usage: nice -n 19 python3 1741_fetch.py [--limit N] [--out PATH]
"""
import argparse, bisect, csv, importlib.util, os, shutil, sqlite3, sys, time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHE = os.path.join(HERE, '1741_cache')
FLOW = os.path.join(HERE, '1741_flow.csv')
LAB = os.path.join(HERE, '..', 'hod_exit_lab')
BATCH = 100            # contracts per bars request
MIN_FREE_GB = 5.0
COLS = ['day', 'symbol', 'covered', 'n_contracts', 'call_vol', 'put_vol', 'call_notional', 'put_notional',
        'n_prints', 'prior_day_vol', 'first_print_minute']
ET = ZoneInfo('America/New_York')

_spec = importlib.util.spec_from_file_location('f1738', os.path.join(HERE, '1738_fetch.py'))
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
F.CACHE = CACHE
F.MIN_INTERVAL = 0.34


def free_gb():
    """Free GB on the filesystem holding the cache."""
    return shutil.disk_usage(HERE).free / 1e9


def trading_days():
    """Sorted SPY trading dates from cache.db daily_bars (read-only)."""
    c = sqlite3.connect('file:' + os.path.join(ROOT, 'data', 'cache.db') + '?mode=ro', uri=True)
    try:
        return [r[0] for r in c.execute("select bar_date from daily_bars where symbol='SPY' order by bar_date")]
    finally:
        c.close()


def population():
    """TRAIN+VAL signals de-duplicated on (symbol, day), sorted by day then symbol."""
    s = pd.read_parquet(os.path.join(LAB, 'signals.parquet'))
    s = s[s.split.isin(['TRAIN', 'VAL'])][['day', 'symbol', 'entry_m', 'entry']]
    s = s.drop_duplicates(['symbol', 'day'])
    return s.sort_values(['day', 'symbol']).reset_index(drop=True)


def chain(sym, day, entry):
    """Contracts (calls+puts) expiring in (day, day+45], strike within +-20 % of entry; slim dicts."""
    d = date.fromisoformat(day)
    lo, hi = repr(round(entry * 0.8, 4)), repr(round(entry * 1.2, 4))
    out = []
    for status in ('inactive', 'active'):
        if status == 'active' and d + timedelta(days=45) < date.today() - timedelta(days=1):
            continue
        out += F.paged(F.CONTRACTS_URL, dict(underlying_symbols=sym, expiration_date_gte=(d + timedelta(days=1)).isoformat(),
                       expiration_date_lte=(d + timedelta(days=45)).isoformat(), status=status,
                       strike_price_gte=lo, strike_price_lte=hi, limit=10000), 'option_contracts')
    seen, res = set(), []
    for c in out:
        if c.get('underlying_symbol') == sym and c.get('root_symbol') == sym and c['symbol'] not in seen:
            seen.add(c['symbol']); res.append(dict(symbol=c['symbol'], type=c['type']))
    return res


def bars(contracts, day, start_m, end_m):
    """Merged {contract: [bars]} for batches of contracts, labels start_m..end_m inclusive (ET minutes of day)."""
    res = {}
    syms = sorted(c['symbol'] for c in contracts)
    for i in range(0, len(syms), BATCH):
        part = F.paged(F.BARS_URL, dict(symbols=','.join(syms[i:i + BATCH]), timeframe='1Min', start=F.utc_iso(day, start_m),
                                        end=F.utc_iso(day, end_m), limit=10000), 'bars') or {}
        for k, v in part.items():
            res.setdefault(k, []).extend(v)
    return res


def row_for(r, prev_day):
    """Build the flow row for one population row r (day, symbol, entry_m, entry)."""
    base = dict(day=r.day, symbol=r.symbol, covered=False, n_contracts=0, call_vol=0, put_vol=0, call_notional=0.0,
                put_notional=0.0, n_prints=0, prior_day_vol=0, first_print_minute='')
    cs = chain(r.symbol, r.day, float(r.entry))
    if not cs:
        return base
    typ = {c['symbol']: c['type'] for c in cs}
    base.update(covered=True, n_contracts=len(cs))
    last_m = int(r.entry_m) - 2                      # last pre-break bar label (break bar = entry_m - 1)
    first = None
    if last_m >= 570:
        for k, bl in bars(cs, r.day, 570, last_m).items():
            for b in bl:
                m = datetime.fromisoformat(b['t'].replace('Z', '+00:00')).astimezone(ET)
                mm = m.hour * 60 + m.minute
                if mm > last_m:
                    continue                          # defensive: never count the break bar or later
                v, notional = int(b['v']), b['v'] * b['c'] * 100.0
                kind = 'call_' if typ.get(k) == 'call' else 'put_'
                base[kind + 'vol'] += v; base[kind + 'notional'] += notional; base['n_prints'] += int(b.get('n', 1))
                first = mm if first is None else min(first, mm)
    base['first_print_minute'] = '' if first is None else first
    if prev_day:
        for k, bl in bars(cs, prev_day, 570, 960).items():
            base['prior_day_vol'] += sum(int(b['v']) for b in bl)
    return base


def done_keys():
    """(symbol, day) pairs already in the csv (resume)."""
    if not os.path.exists(FLOW):
        return set()
    d = pd.read_csv(FLOW, usecols=['day', 'symbol'], dtype=str)
    return set(zip(d.symbol, d.day))


def main():
    """Fetch loop with resume, progress lines, disk guard and a consecutive-failure abort."""
    global FLOW
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--out', default=FLOW)
    a = ap.parse_args()
    FLOW = a.out
    os.makedirs(CACHE, exist_ok=True)
    if free_gb() < MIN_FREE_GB:
        print(f'ERROR df {free_gb():.1f} GB < {MIN_FREE_GB}; refusing to start', flush=True); sys.exit(2)
    pop = population()
    days = trading_days()
    skip = done_keys()
    todo = [r for r in pop.itertuples() if (r.symbol, r.day) not in skip]
    if a.limit:
        todo = todo[:a.limit]
    print(f'population {len(pop)} symbol-days, already done {len(skip)}, to do {len(todo)}, free {free_gb():.1f} GB', flush=True)
    new = not os.path.exists(FLOW)
    fails, t0, n = 0, time.time(), 0
    with open(FLOW, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        for r in todo:
            i = bisect.bisect_left(days, r.day)
            prev = days[i - 1] if i > 0 else None
            try:
                row = row_for(r, prev)
                fails = 0
            except Exception as e:                    # logged, skipped (stays missing from csv so a rerun retries it)
                fails += 1
                print(f'ERROR {r.symbol} {r.day}: {type(e).__name__} {str(e)[:200]}', flush=True)
                if fails >= 20:
                    print('ERROR 20 consecutive failures; aborting', flush=True); sys.exit(3)
                continue
            w.writerow({k: row[k] for k in COLS}); fh.flush()
            n += 1
            if n % 200 == 0 or n == 1:
                if free_gb() < MIN_FREE_GB:
                    print(f'ERROR free disk {free_gb():.1f} GB < {MIN_FREE_GB}; stopping (resume later)', flush=True); sys.exit(4)
                el = time.time() - t0
                eta = el / n * (len(todo) - n) / 60
                print(f'{n}/{len(todo)} {r.day} {r.symbol} covered={row["covered"]} requests={F.STATS["net"]} cached={F.STATS["cached"]} '
                      f'r429={F.STATS["r429"]} free={free_gb():.0f}GB eta={eta:.0f}min', flush=True)
    print(f'DONE n_symbol_days={n} n_requests={F.STATS["net"]}', flush=True)


if __name__ == '__main__':
    main()
