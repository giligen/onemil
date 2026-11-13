"""Cell 1,770 quote fetch (PREREG_1770, frozen 2026-11-13 21:20 UTC): per-trade measured NBBO for the 1,769 primary trades
(E1 short at 09:35, stop = first-5-min high +50 bp, cover 10:30).  Sample = ALL VAL 2021-23 primary trades + 1,000 random
TRAIN primary trades (seed 1770).  Windows (America/New_York, 60 s): entry 09:35:00-09:35:59 (last valid quote), cover
10:30:00-10:30:59 (last valid quote), and for stop-hit trades the minute whose bar high first reached the stop (first valid
quote of the minute, first valid quote with ask >= stop level, last valid quote).  SENSITIVITY windows (added, not in the PREREG):
09:34:55-09:34:59 / 10:29:55-10:29:59 = the last quote BEFORE the 09:35 / 10:30 instant at which the 1769 bar price is struck.  Only reduced rows are kept; per-day
checkpoint into 1770_cache/<day>.parquet; resumes.  TEST (2024+) is never requested.
Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/signal_search/1770_fetch_quotes.py
"""
import os, time
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '1770_cache')
BARS = os.path.join(HERE, '1769_cache')
SAMPLE = os.path.join(HERE, '1770_sample.parquet')
URL = 'https://data.alpaca.markets/v2/stocks/quotes'
NY = ZoneInfo('America/New_York')
BUDGET_S, MIN_DT, SLIP_BP, N_TRAIN, SEED = 540, 0.21, 50.0, 1000, 1770
ENTRY_STAMP, COVER_STAMP = 574, 629          # 1769: bar stamped 09:34 closes at 09:35; bar stamped 10:29 closes at 10:30


def log(m):
    print(m, flush=True)


def keys():
    """Alpaca headers from .env (never printed)."""
    load_dotenv('/home/ec2-user/onemil/.env')
    k = os.getenv('ALPACA_HOD_API_KEY') or os.getenv('ALPACA_API_KEY')
    s = os.getenv('ALPACA_HOD_API_SECRET') or os.getenv('ALPACA_API_SECRET')
    if not k or not s:
        raise SystemExit('ERROR: Alpaca keys missing in .env')
    return {'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s}


def stamp_to_rfc(day, minute_of_day, add_s=0):
    """RFC-3339 for `day` at minute-of-day (NY) + add_s seconds, DST-correct."""
    d = datetime.fromisoformat(f'{day}T00:00:00').replace(tzinfo=NY) + timedelta(minutes=int(minute_of_day), seconds=add_s)
    return d.isoformat()


def stop_info(day, symbol):
    """Re-derive the 1769 stop level / stop-bar stamp / entry price from the cached minute bars (same rule as 1769_score)."""
    b = pd.read_parquet(os.path.join(BARS, f'{day}.parquet'))
    g = b[b.symbol == symbol]
    et = pd.to_datetime(g.t, utc=True).dt.tz_convert(NY)
    m = (et.dt.hour * 60 + et.dt.minute).to_numpy(); h, c, o = g.h.to_numpy(), g.c.to_numpy(), g.o.to_numpy()
    ent = np.flatnonzero((m >= 570) & (m <= ENTRY_STAMP))
    if not len(ent):
        return None
    H = float(h[ent].max()); trig = H * (1 + SLIP_BP / 1e4)
    win = np.flatnonzero((m > ENTRY_STAMP) & (m <= COVER_STAMP))
    hit = win[h[win] >= trig] if len(win) else win
    if len(hit):
        j = hit[0]; return dict(H=H, trig=trig, stop_re=1, stop_m=int(m[j]), model_fill=max(trig, float(o[j])), entry_bar=float(c[ent[-1]]))
    k = np.flatnonzero(m <= COVER_STAMP)
    return dict(H=H, trig=trig, stop_re=0, stop_m=-1, model_fill=float(c[k[-1]]) if len(k) else np.nan, entry_bar=float(c[ent[-1]]))


def build_sample():
    """All VAL primary trades + 1,000 seeded TRAIN primary trades, annotated with stop level / stop minute; write parquet."""
    parts = []
    for split in ('VAL', 'TRAIN'):
        t = pd.read_parquet(os.path.join(HERE, f'1769_trades_{split}.parquet'))
        t = t[(t.cover == '1030') & (t.conv == 'prereg')].reset_index(drop=True)
        if split == 'TRAIN':
            t = t.sample(n=N_TRAIN, random_state=SEED).sort_values(['day', 'symbol']).reset_index(drop=True)
        t['split'] = split; parts.append(t)
    S = pd.concat(parts, ignore_index=True)
    info = [stop_info(d, s) for d, s in zip(S.day, S.symbol)]
    assert all(i is not None for i in info), 'missing bars for a 1769 primary trade'
    S = pd.concat([S, pd.DataFrame(info)], axis=1)
    bad = int((S.stop_re != S['stop']).sum()); assert bad == 0, f'{bad} stop flags differ from 1769_trades'
    S.to_parquet(SAMPLE); log(f'sample written: {len(S)} trades ({S.split.value_counts().to_dict()}), stop-hit {S["stop"].mean():.1%}')
    return S


def get(sess, params, last):
    """One GET with <=5 req/s pacing and retry on 429/5xx."""
    for attempt in range(6):
        w = MIN_DT - (time.time() - last[0])
        if w > 0:
            time.sleep(w)
        last[0] = time.time()
        r = sess.get(URL, params=params, timeout=90)
        if r.status_code == 200:
            return r.json()
        log(f'WARNING: HTTP {r.status_code} attempt {attempt}: {r.text[:200]}')
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 + 3 * attempt); continue
        raise RuntimeError(f'HTTP {r.status_code}: {r.text[:200]}')
    raise RuntimeError('retries exhausted')


def window(sess, last, day, minute, symbols, trig=None, off_s=0, span_s=59):
    """Quotes in [minute:00, minute:59] ET for `symbols` -> dict symbol -> reduced row (last valid, first valid, first ask>=trig).
    Valid quote: bid > 0, ask > bid (crossed / locked / zero quotes are skipped and counted)."""
    out = {}
    for i in range(0, len(symbols), 50):
        chunk = symbols[i:i + 50]
        p = dict(symbols=','.join(chunk), start=stamp_to_rfc(day, minute, off_s), end=stamp_to_rfc(day, minute, off_s + span_s), feed='sip', limit=10000)
        while True:
            j = get(sess, p, last)
            for s, qs in (j.get('quotes') or {}).items():
                r = out.setdefault(s, dict(n=0, n_bad=0, last=None, first=None, trig_q=None))
                for q in qs:
                    if not (q['bp'] > 0 and q['ap'] > q['bp']):
                        r['n_bad'] += 1; continue
                    r['n'] += 1; v = (q['t'], q['bp'], q['ap'])
                    r['last'] = v                                   # pages arrive in time order per symbol
                    if r['first'] is None:
                        r['first'] = v
                    if trig is not None and r['trig_q'] is None and q['ap'] >= trig[s]:
                        r['trig_q'] = v
            if not j.get('next_page_token'):
                break
            p['page_token'] = j['next_page_token']
    return out


def fetch_day(sess, last, day, S):
    """All required windows for one day -> DataFrame of reduced quote rows."""
    D = S[S.day == day]; syms = sorted(D.symbol.unique()); rows = []
    for kind, minute, off, span in (('entry', 575, 0, 59), ('cover', 630, 0, 59), ('entry_pre', 574, 55, 4), ('cover_pre', 629, 55, 4)):
        res = window(sess, last, day, minute, syms, off_s=off, span_s=span)
        for s in syms:
            r = res.get(s)
            if r and r['last']:
                rows.append(dict(day=day, symbol=s, kind=kind, n=r['n'], n_bad=r['n_bad'], t=r['last'][0], bid=r['last'][1], ask=r['last'][2]))
            else:
                rows.append(dict(day=day, symbol=s, kind=kind, n=0, n_bad=(r or {}).get('n_bad', 0)))
    St = D[D['stop'] == 1]
    for m, g in St.groupby('stop_m'):
        trig = dict(zip(g.symbol, g.trig)); ss = sorted(trig)
        res = window(sess, last, day, int(m), ss, trig)
        for s in ss:
            r = res.get(s)
            if r and r['last']:
                tq = r['trig_q']
                rows.append(dict(day=day, symbol=s, kind='stop', n=r['n'], n_bad=r['n_bad'], t=r['first'][0], bid=r['first'][1], ask=r['first'][2],
                                 t_trig=tq[0] if tq else None, ask_trig=tq[2] if tq else np.nan, bid_trig=tq[1] if tq else np.nan,
                                 t_last=r['last'][0], ask_last=r['last'][2]))
            else:
                rows.append(dict(day=day, symbol=s, kind='stop', n=0, n_bad=(r or {}).get('n_bad', 0)))
    return pd.DataFrame(rows)


def main():
    """Build the sample if absent, fetch each missing day until done / budget spent, then assemble 1770_quotes.parquet."""
    t0 = time.time(); os.makedirs(CACHE, exist_ok=True)
    S = pd.read_parquet(SAMPLE) if os.path.exists(SAMPLE) else build_sample()
    assert S.day.max() < '2024-01-01', 'TEST leaked'
    days = sorted(S.day.unique()); todo = [d for d in days if not os.path.exists(os.path.join(CACHE, f'{d}.parquet'))]
    log(f'sample days {len(days)}, trades {len(S)}, to fetch {len(todo)}')
    sess = requests.Session(); sess.headers.update(keys()); last = [0.0]
    for k, d in enumerate(todo):
        if time.time() - t0 > BUDGET_S:
            log(f'budget spent at day {k}/{len(todo)}; rerun to resume'); return
        df = fetch_day(sess, last, d, S)
        tmp = os.path.join(CACHE, f'{d}.parquet.tmp'); df.to_parquet(tmp); os.replace(tmp, os.path.join(CACHE, f'{d}.parquet'))
        if k % 25 == 0:
            ok = (df.n > 0).mean() if len(df) else float('nan')
            log(f'{d}: {len(df)} rows, with quotes {ok:.0%} ({k + 1}/{len(todo)}, {time.time() - t0:.0f}s)')
    allq = pd.concat([pd.read_parquet(os.path.join(CACHE, f'{d}.parquet')) for d in days], ignore_index=True)
    allq.to_parquet(os.path.join(HERE, '1770_quotes.parquet')); log(f'fetch complete: {len(allq)} quote rows -> 1770_quotes.parquet')


if __name__ == '__main__':
    main()
