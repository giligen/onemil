"""Cell 1,748 fetch.  Stages: bars | events | news | cost.   Resumable, atomic, keys never printed.
bars  : raw + all-adjusted daily bars for every eligible common-stock candidate (rank 1,001-3,000 is decided later, point in time).
events: PIT universe at t-1 (rank 1,001-3,000 by 20d median $ volume, px>=3, dv20>=2M) and day-t return <= -8 % -> 1748_events_raw.parquet.
news  : one Alpaca news query per event DAY (all event symbols of that day, window 16:00 ET t-1 .. 16:00 ET t), cached per day.
cost  : measured NBBO half-spread 09:30-09:35 and 15:55-16:00 on day t+1 for a 400-event VAL no-news sample.
Reuses helpers of 1746_fetch.py."""
import importlib.util, json, os, shutil, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('f46', os.path.join(HERE, '1746_fetch.py'))
f46 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f46)
CACHE = os.path.join(HERE, '1748_cache'); os.makedirs(CACHE, exist_ok=True)
NEWS_DIR = os.path.join(CACHE, 'news'); os.makedirs(NEWS_DIR, exist_ok=True)
NEWS_URL = 'https://data.alpaca.markets/v1beta1/news'
QUOTES_URL = 'https://data.alpaca.markets/v2/stocks/quotes'
ET, UTC = ZoneInfo('America/New_York'), ZoneInfo('UTC')
log = f46.log
THRESH = -0.08
RANK_LO, RANK_HI, MIN_PX, MIN_DV = 1001, 3000, 3.0, 2e6
_last = [0.0]


def pace():
    """Global 0.36 s spacing (<= 166 req/min)."""
    w = float(os.environ.get('PACE', '0.36')) - (time.time() - _last[0])
    if w > 0:
        time.sleep(w)
    _last[0] = time.time()


def utc(day, hh, mm):
    """ET wall time on `day` -> RFC3339 UTC."""
    return datetime.fromisoformat(day).replace(hour=hh, minute=mm, tzinfo=ET).astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def stage_bars():
    """Raw + all-adjusted bars for all candidates, batched, resumable; logs LOST count."""
    syms = sorted(f46.candidate_names())
    bdir = os.path.join(CACHE, 'bars_parts'); os.makedirs(bdir, exist_ok=True)
    B = 100
    batches = [syms[i:i + B] for i in range(0, len(syms), B)]
    off = int(os.environ.get('BARS_OFFSET', '0'))
    order = [(i + off) % len(batches) for i in range(len(batches))]
    for i in order:
        b = batches[i]
        for adj in ('raw', 'all'):
            part = os.path.join(bdir, f'{adj}_{i:03d}.parquet')
            if os.path.exists(part):
                continue
            if shutil.disk_usage(HERE).free / 1e9 < f46.MIN_FREE_GB:
                log('ERROR disk < 5 GB'); sys.exit(3)
            df = f46.get_bars(b, adj, f46.BARS_START, f46.BARS_END)
            df.to_parquet(part + '.tmp'); os.replace(part + '.tmp', part)
            log(f'bars {adj} batch {i + 1}/{len(batches)} rows {len(df)} got {df.symbol.nunique()}/{len(b)}')
    for adj in ('raw', 'all'):
        parts = [pd.read_parquet(os.path.join(bdir, f)) for f in sorted(os.listdir(bdir)) if f.startswith(adj + '_') and f.endswith('.parquet')]
        d = pd.concat(parts, ignore_index=True)
        d.to_parquet(os.path.join(CACHE, f'bars_{adj}.parquet.tmp')); os.replace(os.path.join(CACHE, f'bars_{adj}.parquet.tmp'), os.path.join(CACHE, f'bars_{adj}.parquet'))
        log(f'bars {adj}: {len(d)} rows, {d.symbol.nunique()} symbols of {len(syms)} (LOST {len(syms) - d.symbol.nunique()})')


def load_panels():
    """(days, dict of date x symbol frames) restricted to SPY trading days."""
    days = pd.read_csv(os.path.join(f46.CACHE, 'trading_days.csv')).date.tolist()
    out = {}
    for adj in ('raw', 'all'):
        d = pd.read_parquet(os.path.join(CACHE, f'bars_{adj}.parquet'))
        for col in 'ohclv':
            out[f'{col}_{adj}'] = d.pivot_table(index='date', columns='symbol', values=col, aggfunc='last')
    idx = sorted(set(out['c_raw'].index) | set(days))
    for k in out:
        out[k] = out[k].reindex(idx)
    return idx, out


def stage_events():
    """PIT rank + day-t drop <= -8 %.  Writes 1748_events_raw.parquet and the daily universe-size line."""
    idx, P = load_panels()
    dv = (P['c_raw'] * P['v_raw'])
    dv20 = dv.shift(1).rolling(20, min_periods=15).median()      # median over t-20..t-1
    px1 = P['c_raw'].shift(1)
    rank = dv20.rank(axis=1, ascending=False, method='first')
    ret = P['c_all'] / P['c_all'].shift(1) - 1
    ret_raw = P['c_raw'] / P['c_raw'].shift(1) - 1
    spy = ret['SPY'] if 'SPY' in ret else None
    rows = []
    days = pd.read_csv(os.path.join(f46.CACHE, 'trading_days.csv')).date.tolist()
    dpos = {d: i for i, d in enumerate(days)}
    for t in days[22:-3]:
        r = ret.loc[t]
        m = (r <= THRESH) & (rank.loc[t] >= RANK_LO) & (rank.loc[t] <= RANK_HI) & (px1.loc[t] >= MIN_PX) & (dv20.loc[t] >= MIN_DV)
        for s in r.index[m.fillna(False)]:
            i = dpos[t]
            t1, t3 = days[i + 1], days[i + 3]
            rows.append(dict(date=t, symbol=s, ret=r[s], ret_raw=ret_raw.loc[t, s], rank=int(rank.loc[t, s]), dv20=dv20.loc[t, s], px_prev=px1.loc[t, s],
                             c_t=P['c_all'].loc[t, s], o_1=P['o_all'].loc[t1, s], c_1=P['c_all'].loc[t1, s], c_3=P['c_all'].loc[t3, s],
                             h_t=P['h_all'].loc[t, s], l_t=P['l_all'].loc[t, s], date1=t1,
                             prev_date=days[i - 1],
                             atr14=float(np.nanmean([(P['h_all'][s].iloc[j] - P['l_all'][s].iloc[j]) for j in range(P['h_all'].index.get_loc(t) - 14, P['h_all'].index.get_loc(t))])) if s in P['h_all'] else np.nan))
    ev = pd.DataFrame(rows)
    log(f'raw events {len(ev)}; years {ev.date.str[:4].value_counts().sort_index().to_dict()}')
    ev.to_parquet(os.path.join(HERE, '1748_events_raw.parquet'))
    # placebo candidates: open(t'+1)->close(t'+1) returns on days t+5..t+60 for the same name
    pl = {}
    for e in ev.itertuples():
        i = dpos[e.date]
        cand = []
        for j in range(i + 5, min(i + 61, len(days) - 1)):
            o, c, r_ = P['o_all'].loc[days[j + 1], e.symbol], P['c_all'].loc[days[j + 1], e.symbol], ret.loc[days[j], e.symbol]
            if pd.notna(o) and pd.notna(c) and o > 0 and pd.notna(r_) and r_ > THRESH:
                cand.append(c / o - 1)
        pl[f'{e.date}|{e.symbol}'] = cand
    pd.to_pickle(pl, os.path.join(CACHE, 'placebo.pkl'))
    # market (SPY) returns and 20d realised vol for the stress terciles
    if spy is not None:
        s = pd.DataFrame({'spy_ret': spy, 'rv20': spy.rolling(20).std() * np.sqrt(252) * 100})
        s.to_parquet(os.path.join(CACHE, 'spy_rv.parquet'))
    log('events stage done')


def news_day(day, prev_day, syms):
    """Query news for `syms` in [16:00 ET prev_day, 16:00 ET day]; returns {sym: n_items}."""
    cnt = {s: 0 for s in syms}
    for k in range(0, len(syms), 20):
        chunk = syms[k:k + 20]
        tok, pages = None, 0
        while True:
            p = {'symbols': ','.join(chunk), 'start': utc(prev_day, 16, 0), 'end': utc(day, 16, 0), 'limit': 50}
            if tok:
                p['page_token'] = tok
            delay = 2.0
            for _ in range(8):
                pace()
                r = requests.get(NEWS_URL, headers=f46.headers(), params=p, timeout=90)
                if r.status_code == 429 or r.status_code >= 500:
                    log(f'WARNING news HTTP {r.status_code}; backoff {delay}s'); time.sleep(delay); delay = min(delay * 2, 60); continue
                if r.status_code != 200:
                    raise RuntimeError(f'news HTTP {r.status_code} {r.text[:200]}')
                break
            else:
                raise RuntimeError('news gave up')
            j = r.json()
            for it in j.get('news', []):
                for s in it.get('symbols', []):
                    if s in cnt:
                        cnt[s] += 1
            tok = j.get('next_page_token'); pages += 1
            if not tok:
                break
            if pages >= 30:
                log(f'WARNING news pagination capped at 30 pages {day} {chunk[:3]}'); break
    return cnt


def stage_news():
    """One cached JSON per event day."""
    ev = pd.read_parquet(os.path.join(HERE, '1748_events_raw.parquet'))
    g = ev.groupby('date')
    n_days = g.ngroups
    for i, (day, grp) in enumerate(g):
        path = os.path.join(NEWS_DIR, day + '.json')
        if i % int(os.environ.get('NEWS_MOD', '1')) != int(os.environ.get('NEWS_REM', '0')) or os.path.exists(path):
            continue
        cnt = news_day(day, grp.prev_date.iloc[0], sorted(grp.symbol.unique()))
        f46.atomic_write(path, json.dumps(cnt), 'w')
        if i % 50 == 0:
            log(f'news day {i + 1}/{n_days} {day} syms {len(cnt)} with_news {sum(v > 0 for v in cnt.values())}')
        if shutil.disk_usage(HERE).free / 1e9 < f46.MIN_FREE_GB:
            log('ERROR disk < 5 GB'); sys.exit(3)
    if os.environ.get('NEWS_MOD', '1') != '1':
        log('partial news worker done'); return
    rows = []
    for f in sorted(os.listdir(NEWS_DIR)):
        if f.endswith('.json'):
            for sym, n in json.load(open(os.path.join(NEWS_DIR, f))).items():
                rows.append((f[:-5], sym, n))
    nd = pd.DataFrame(rows, columns=['date', 'symbol', 'n_news'])
    m = ev.merge(nd, on=['date', 'symbol'], how='left')
    log(f'WARNING events without a news answer: {int(m.n_news.isna().sum())}' if m.n_news.isna().any() else 'all events have a news answer')
    m['news'] = (m.n_news.fillna(0) > 0).astype(int)
    m.to_parquet(os.path.join(HERE, '1748_events.parquet'))
    log(f'news stage done: events {len(m)}, with >=1 item {m.news.mean():.3f}')


def hs_one(hdr, sym, day, hh, mm, hh2, mm2):
    """(n quotes, median half-spread bp) for the quote window."""
    p = {'symbols': sym, 'start': utc(day, hh, mm), 'end': utc(day, hh2, mm2), 'limit': 1000, 'feed': 'sip'}
    for _ in range(6):
        pace()
        r = requests.get(QUOTES_URL, headers=hdr, params=p, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            log(f'WARNING quotes HTTP {r.status_code}'); time.sleep(3); continue
        if r.status_code != 200:
            log(f'ERROR quotes HTTP {r.status_code} {sym} {day}'); return 0, None
        qs = (r.json().get('quotes') or {}).get(sym, [])
        v = [((q['ap'] - q['bp']) / 2 / ((q['ap'] + q['bp']) / 2) * 1e4) for q in qs if q['bp'] > 0 and q['ap'] > q['bp'] and (q['ap'] - q['bp']) / q['bp'] < 0.2]
        return len(qs), (float(pd.Series(v).median()) if v else None)
    log(f'ERROR quotes gave up {sym} {day}'); return 0, None


def stage_cost():
    """400-event VAL no-news sample (seed 1748): entry 09:30-09:35, exit 15:55-16:00 on t+1."""
    ev = pd.read_parquet(os.path.join(HERE, '1748_events.parquet'))
    d = ev[(ev.date >= '2021-01-01') & (ev.date <= '2023-12-31') & (ev.news == 0)]
    s = d.sample(min(400, len(d)), random_state=1748).sort_values('date')
    hdr = f46.headers(); rows = []
    for i, r in enumerate(s.itertuples()):
        ne, he = hs_one(hdr, r.symbol, r.date1, 9, 30, 9, 35)
        nx, hx = hs_one(hdr, r.symbol, r.date1, 15, 55, 16, 0)
        rows.append(dict(symbol=r.symbol, date=r.date, date1=r.date1, rank=r.rank, entry_n=ne, entry_hs_bp=he, exit_n=nx, exit_hs_bp=hx))
        if i % 50 == 0:
            log(f'cost {i}/{len(s)} {rows[-1]}')
    o = pd.DataFrame(rows); o.to_csv(os.path.join(HERE, '1748_cost.csv'), index=False)
    log(f'cost: entry {o.entry_hs_bp.notna().sum()}/{len(o)} mean {o.entry_hs_bp.mean():.1f} med {o.entry_hs_bp.median():.1f}; exit {o.exit_hs_bp.notna().sum()}/{len(o)} mean {o.exit_hs_bp.mean():.1f} med {o.exit_hs_bp.median():.1f}')


if __name__ == '__main__':
    for st in sys.argv[1:]:
        {'bars': stage_bars, 'events': stage_events, 'news': stage_news, 'cost': stage_cost}[st]()
