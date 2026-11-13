"""PREREG 1,747 cost: measured NBBO half-spread at 15:55-16:00 ET (first 1,000 quotes from 15:55:00, feed sip) on the
entry day (ex-1) and the exit day (ex) for a 400-event sample (liquid tier, VAL, seed 1747).  Median over quotes of
(ask-bid)/2/mid in bp, 0<bid<ask, spread<5 %.  Resumable per row.  Output 1747_cost.csv.  Keys never printed."""
import os, time
import pandas as pd
import requests
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
URL = 'https://data.alpaca.markets/v2/stocks/quotes'
ET, UTC = ZoneInfo('America/New_York'), ZoneInfo('UTC')
OUT = os.path.join(HERE, '1747_cost.csv')


def utc(day, hh, mm):
    """ET wall time -> RFC3339 UTC."""
    return datetime.fromisoformat(day).replace(hour=hh, minute=mm, tzinfo=ET).astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def one(hdr, sym, day):
    """(n_quotes, median half-spread bp) for sym on day 15:55-16:00 ET."""
    p = {'symbols': sym, 'start': utc(day, 15, 55), 'end': utc(day, 16, 0), 'limit': 1000, 'feed': 'sip'}
    delay = 2.0
    for _ in range(6):
        r = requests.get(URL, headers=hdr, params=p, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            print(f'WARNING HTTP {r.status_code} {sym} {day}', flush=True); time.sleep(delay); delay *= 2; continue
        if r.status_code != 200:
            print(f'ERROR HTTP {r.status_code} {sym} {day} {r.text[:120]}', flush=True); return 0, None
        qs = (r.json().get('quotes') or {}).get(sym, [])
        v = [((q['ap'] - q['bp']) / 2 / ((q['ap'] + q['bp']) / 2) * 1e4) for q in qs
             if q['bp'] > 0 and q['ap'] > q['bp'] and (q['ap'] - q['bp']) / q['bp'] < 0.05]
        return len(qs), (float(pd.Series(v).median()) if v else None)
    print(f'ERROR gave up {sym} {day}', flush=True)
    return 0, None


def main():
    load_dotenv(os.path.join(ROOT, '.env'))
    hdr = {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'], 'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']}
    d = pd.read_parquet(os.path.join(HERE, '1747_scored.parquet'))
    d = d[(d.ex_date >= '2021-01-01') & (d.ex_date <= '2023-12-31') & (d.dv20 >= 20e6)]
    s = d.sample(400, random_state=1747).sort_values('ex_date')
    days = pd.read_csv(os.path.join(HERE, '1746_cache', 'trading_days.csv')).date.tolist()
    prev = {x: days[i - 1] for i, x in enumerate(days) if i}
    rows, last = [], 0.0
    for i, r in enumerate(s.itertuples()):
        res = {'symbol': r.symbol, 'ex_date': r.ex_date}
        for tag, day in (('entry', prev[r.ex_date]), ('exit', r.ex_date)):
            w = 0.34 - (time.time() - last)
            if w > 0:
                time.sleep(w)
            last = time.time()
            n, hs = one(hdr, r.symbol, day)
            res[f'{tag}_n'] = n; res[f'{tag}_hs_bp'] = hs
        rows.append(res)
        if i % 50 == 0:
            print(f'{i}/{len(s)} {r.symbol} {r.ex_date} {res}', flush=True)
    o = pd.DataFrame(rows)
    o.to_csv(OUT + '.tmp', index=False); os.replace(OUT + '.tmp', OUT)
    print(f"cost: entry {o.entry_hs_bp.notna().sum()}/{len(o)} mean {o.entry_hs_bp.mean():.2f} median {o.entry_hs_bp.median():.2f}; "
          f"exit {o.exit_hs_bp.notna().sum()}/{len(o)} mean {o.exit_hs_bp.mean():.2f} median {o.exit_hs_bp.median():.2f} bp", flush=True)


if __name__ == '__main__':
    main()
