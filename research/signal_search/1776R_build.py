"""Independent rebuild of cell 1,776 primary (gap top-10, L1 pre-market $vol top tercile, short MOO -> 10:30 ask)."""
import os, sys, json, time, sqlite3, threading
import numpy as np, pandas as pd, requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
os.nice(19)
ROOT = '/home/ec2-user/onemil'; OUT = f'{ROOT}/research/signal_search'
CACHE = f'{OUT}/1776R_cache/quotes.jsonl'
ET = ZoneInfo('America/New_York')
FEES, LOCATE, BORROW = 4.0, 9.0, 27.8
load_dotenv(f'{ROOT}/.env', override=True)
KEY, SEC = os.environ['ALPACA_HOD_API_KEY'], os.environ['ALPACA_HOD_API_SECRET']

def load_universe():
    """Population + daily_bars open/prev close, gap filter, top-10 by gap, L1."""
    p = pd.read_csv(f'{ROOT}/research/hod_entry/cell_1428_population.csv')
    db = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True)
    d = pd.read_sql("select symbol,bar_date as day,open,close from daily_bars where bar_date>='2025-06-01'", db)
    spy = pd.read_sql("select bar_date from daily_bars where symbol='SPY' order by bar_date", db).bar_date.tolist()
    d = d.sort_values(['symbol', 'day']); d['pclose'] = d.groupby('symbol').close.shift(1)
    p = p.merge(d[['symbol', 'day', 'open', 'pclose']], on=['day', 'symbol'], how='left')
    p['gap'] = p.open / p.pclose - 1
    n0 = len(p); p = p[~(p.gap > 2.0)]; print('dropped gap>200% or ...', n0 - len(p), flush=True)
    p = p[p.open.notna() & p.gap.notna()]
    l1 = {}
    for day in p.day.unique():
        f = f'{ROOT}/research/hod_entry/1762_cache/premarket/{day}.parquet'
        if not os.path.exists(f): continue
        b = pd.read_parquet(f)
        t = pd.to_datetime(b.t, utc=True).dt.tz_convert(ET)
        end = t + pd.Timedelta(minutes=1); m = t.dt.normalize()
        ok = (t >= m + pd.Timedelta(hours=4)) & (end <= m + pd.Timedelta(hours=9, minutes=25))
        b = b[ok.values]; b = b.assign(dv=b.c * b.v)
        for s, v in b.groupby('symbol').dv.sum().items(): l1[(day, s)] = v
    p['L1'] = [l1.get((a, b), np.nan) for a, b in zip(p.day, p.symbol)]
    return p, spy

def key(day, sym): return f'{day}|{sym}'
class Fetcher:
    """Resumable rate-limited Alpaca sip last-quote-at-or-before-10:30 ET fetcher."""
    def __init__(s):
        s.done = {}; s.lock = threading.Lock(); s.last = 0.0
        if os.path.exists(CACHE):
            for l in open(CACHE):
                try: r = json.loads(l); s.done[r['k']] = r
                except Exception: pass
        s.f = open(CACHE, 'a')
    def one(s, day, sym):
        k = key(day, sym)
        if k in s.done: return
        t0 = datetime.fromisoformat(day).replace(hour=10, minute=29, second=30, tzinfo=ET).astimezone(ZoneInfo('UTC'))
        t1 = datetime.fromisoformat(day).replace(hour=10, minute=30, second=0, tzinfo=ET).astimezone(ZoneInfo('UTC'))
        params = dict(start=t0.strftime('%Y-%m-%dT%H:%M:%SZ'), end=t1.strftime('%Y-%m-%dT%H:%M:%S.000001Z'),
                      limit=1, sort='desc', feed='sip')
        for att in range(6):
            with s.lock:
                w = 0.33 - (time.time() - s.last)
                if w > 0: time.sleep(w)
                s.last = time.time()
            try:
                r = requests.get(f'https://data.alpaca.markets/v2/stocks/{sym}/quotes', params=params,
                                 headers={'APCA-API-KEY-ID': KEY, 'APCA-API-SECRET-KEY': SEC}, timeout=30)
            except Exception as e:
                print('WARN req', sym, e, flush=True); time.sleep(2); continue
            if r.status_code == 429: time.sleep(5); continue
            if r.status_code != 200:
                print('WARN', r.status_code, sym, day, r.text[:80], flush=True); time.sleep(2); continue
            q = r.json().get('quotes') or []
            rec = {'k': k, 'ap': q[0]['ap'] if q else None, 'bp': q[0]['bp'] if q else None, 't': q[0]['t'] if q else None}
            with s.lock: s.done[k] = rec; s.f.write(json.dumps(rec) + '\n'); s.f.flush()
            return
        print('ERROR gave up', day, sym, flush=True)

def price(p, fx):
    """Attach ask / r_bp / net_bp; count dropped quotes."""
    ap, bp = [], []
    for d, s in zip(p.day, p.symbol):
        r = fx.done.get(key(d, s)); ap.append(r['ap'] if r else np.nan); bp.append(r['bp'] if r else np.nan)
    p = p.assign(ask=ap, bid=bp)
    mid = (p.ask + p.bid) / 2; hs = (p.ask - p.bid) / 2 / mid * 1e4
    bad = ~((p.ask > 0) & (p.bid > 0)) | (hs > 1000) | (mid < 0.5 * p.open) | (mid > 2 * p.open)
    p['valid'] = ~bad & p.ask.notna()
    p['r_bp'] = (p.open / p.ask - 1) * 1e4; p['net'] = p.r_bp - FEES - LOCATE - BORROW
    return p

def stats(t, label):
    """Primary stats for a trades frame with day, net."""
    t = t[t.valid]; n = len(t)
    if n < 5: return f'{label}: n={n}'
    m = t.net.mean(); e = t.net - m
    se = np.sqrt((e.groupby(t.day).sum() ** 2).sum()) / n
    s = t.net.sort_values(); k = max(1, int(np.ceil(0.05 * n)))
    ex = s.iloc[:n - k].mean(); worst = s.iloc[:k].mean() / 100
    nd = t.groupby('day').size()
    wk = pd.to_datetime(t.day).dt.to_period('W'); w = (300 * t.net / 1e4 / 150).groupby(wk.values).sum()
    return (f'{label}: n={n} mean_net={m:+.1f}bp day-t={m/se:.2f} ex-top5%={ex:+.1f} worst5%={worst:.1f}% '
            f'wkP10={w.quantile(.1):.2f}R names/day med={nd.median():.1f} mean={nd.mean():.1f}')

def boot(vals, n, reps=200, seed=1):
    """p95 of count-matched bootstrap means."""
    rng = np.random.default_rng(seed); v = np.asarray(vals)
    return float(np.percentile([rng.choice(v, n, replace=False if len(v) >= n else True).mean() for _ in range(reps)], 95))

if __name__ == '__main__':
    p, spy = load_universe()
    p = p.sort_values(['day', 'gap'], ascending=[True, False])
    tr = p[p.split == 'TRAIN']; q = tr.L1.dropna()
    lo, hi = q.quantile(2 / 3), None
    print('L1 rows without premarket:', int(p.L1.isna().sum()), 'of', len(p), ' TRAIN T3 cutoff', lo, flush=True)
    top = p.groupby('day').head(10).copy()
    nol1 = int(top.L1.isna().sum()); print('top10 rows without L1 (excluded):', nol1, flush=True)
    book = top[top.L1.notna() & (top.L1 >= lo)].copy()
    # placebo ii: same symbol, 5 trading days later (VAL book)
    idx = {d: i for i, d in enumerate(spy)}; od = pd.read_csv  # noqa
    db = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True)
    dbar = pd.read_sql("select symbol,bar_date as day,open from daily_bars where bar_date>='2025-06-01'", db)
    om = {(a, b): c for a, b, c in zip(dbar.symbol, dbar.day, dbar.open)}
    vb = book[book.split == 'VAL'].copy()
    vb['day5'] = [spy[idx[d] + 5] if d in idx and idx[d] + 5 < len(spy) else None for d in vb.day]
    pl2 = vb[vb.day5.notna()].copy(); pl2['day0'] = pl2.day; pl2['day'] = pl2.day5
    pl2['open'] = [om.get((s, d), np.nan) for s, d in zip(pl2.symbol, pl2.day)]; pl2 = pl2[pl2.open.notna()]
    # placebo i pool: VAL days, same tercile, not in book
    bk = set(zip(book.day, book.symbol))
    pool = p[(p.split == 'VAL') & p.L1.notna() & (p.L1 >= lo) & ~pd.Series([(a, b) in bk for a, b in zip(p.day, p.symbol)], index=p.index)]
    pool = pool[pool.day.isin(vb.day.unique())]
    print('book', len(book), 'VAL', len(vb), 'placebo ii', len(pl2), 'pool i', len(pool), flush=True)
    pool = pool.sample(n=min(1500, len(pool)), random_state=7)
    fx = Fetcher()
    jobs = list(zip(book.day, book.symbol)) + list(zip(pl2.day, pl2.symbol)) + list(zip(pool.day, pool.symbol))
    jobs = [j for j in dict.fromkeys(jobs) if key(*j) not in fx.done]
    print('to fetch', len(jobs), flush=True)
    with ThreadPoolExecutor(4) as ex:
        for i, _ in enumerate(ex.map(lambda j: fx.one(*j), jobs)):
            if i % 200 == 0: print('fetched', i, flush=True)
    B = price(book, fx); P2 = price(pl2, fx); P1 = price(pool, fx)
    for nm, d in (('book', B), ('placebo2', P2), ('placebo1', P1)):
        print(nm, 'rows', len(d), 'valid', int(d.valid.sum()), 'dropped/no-quote', int((~d.valid).sum()), flush=True)
    B.to_parquet(f'{OUT}/1776R_events.parquet')
    V = B[B.split == 'VAL']; T = B[B.split == 'TRAIN']
    lines = [stats(V, 'VAL'), stats(T, 'TRAIN'),
             stats(T[T.day < '2025-10-01'], 'TRAIN H1 (Jul-Sep)'), stats(T[T.day >= '2025-10-01'], 'TRAIN H2 (Oct-Dec)')]
    v = V[V.valid]
    lines.append(stats(P2, 'PLACEBO ii (+5d same symbol)') + f"  boot p95 (n={len(v)}) = {boot(P2[P2.valid].net, len(v)):+.1f}")
    lines.append(f'PLACEBO i pool valid n={int(P1.valid.sum())} mean={P1[P1.valid].net.mean():+.1f} boot p95 (n={len(v)}) = {boot(P1[P1.valid].net, len(v)):+.1f}')
    # compare
    ev = pd.read_parquet(f'{OUT}/1776_events.parquet')
    print('their cols', list(ev.columns), flush=True)
    print(ev.L1_T.value_counts(dropna=False).to_dict(), ev.r_ask1030.describe().to_dict(), flush=True)
    ev = ev[(ev.split == 'VAL') & ev.r_ask1030.notna()]
    ev = ev[ev.L1_T.astype(str).isin(['2', '2.0', 'T3', 'top', '3', '3.0', 'High', 'high'])]
    ev = ev.assign(day=ev.day.astype(str).str[:10], net=ev.r_ask1030 - FEES - LOCATE - BORROW)[['day', 'symbol', 'net']]
    m = v[['day', 'symbol', 'net']].merge(ev, on=['day', 'symbol'], how='outer', suffixes=('_me', '_th'), indicator=True)
    both = m[m._merge == 'both']
    lines.append(f"COMPARE matched={len(both)} only-mine={(m._merge=='left_only').sum()} only-theirs={(m._merge=='right_only').sum()} "
                 f"mean|net diff|={np.abs(both.net_me-both.net_th).mean():.2f}bp theirs-rows={len(ev)} theirs mean net={ev.net.mean():+.1f}")
    open(f'{OUT}/1776R_compare.md', 'w').write('# 1776R rebuild\n\n' + '\n\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
