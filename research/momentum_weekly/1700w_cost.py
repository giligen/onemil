"""1700w_cost.py STAGE (fetch|analyse|restate): PREREG_1700w. Measured NBBO half-spread at 09:45/09:31/10:00 ET on 80 sampled rebalance Mondays
for every name the guarded reference traded (RECON_1700tu_A_guarded_hold.csv: entries kept=0, re-equalised kept=1, exits kept=-1).
fetch: resumable (1700w_raw.csv, one row per name-week-instant); analyse: 1700w_spreads.csv + parquet + stats; restate: three books at measured cost.
Run: bash scripts/research_run.sh -m 2500M python3 research/momentum_weekly/1700w_cost.py STAGE > 1700w_STAGE.log 2>&1"""
import os, sys, time, inspect
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path('/home/ec2-user/onemil/research/momentum_weekly'); NY = ZoneInfo('America/New_York')
RAW = HERE / '1700w_raw.csv'; P = lambda *a: print(*a, flush=True)
INSTANTS = [(9, 45), (9, 31), (10, 0)]; NSAMP = 80


def sample_weeks():
    """Evenly spaced 80 rebalance dates from 2020-01 on (every k-th, fixed by index) and their traded names."""
    h = pd.read_csv(HERE / 'RECON_1700tu_A_guarded_hold.csv'); h = h[h['date'] >= '2020-01-01']
    dates = sorted(h['date'].unique()); k = len(dates) / NSAMP
    pick = [dates[int(i * k)] for i in range(NSAMP)]
    h = h[h['date'].isin(pick)].copy(); h['traded'] = (h['cost'] / h['rate'].where(h['rate'] > 0)).fillna(0.0)
    return h


def last_quote(c, sym, t, StockQuotesRequest, DataFeed):
    """Last quote at or before t: windows 1s,5s,30s,120s; limit 10000; a window that hits the limit is truncated at its START so is skipped."""
    for w in (1, 5, 30, 120):
        for attempt in range(4):
            try:
                r = c.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=sym, start=t - timedelta(seconds=w), end=t, feed=DataFeed.SIP, limit=10000))
                q = r.data.get(sym, []); break
            except Exception as e:
                msg = str(e)
                if 'invalid symbol' in msg.lower() or '404' in msg: return None, 'invalid:' + msg[:60]
                time.sleep(2 * (attempt + 1)); q = None
        if q is None: return None, 'error'
        if q and len(q) < 10000: x = q[-1]; return (x.timestamp, x.bid_price, x.ask_price, x.bid_size, x.ask_size, w), 'ok'
        if q: return None, 'truncated'
    return None, 'empty'


def fetch():
    from dotenv import load_dotenv; load_dotenv('/home/ec2-user/onemil/.env')
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockQuotesRequest
    from alpaca.data.enums import DataFeed
    c = StockHistoricalDataClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_API_SECRET'])
    h = sample_weeks(); done = set()
    if RAW.exists():
        r = pd.read_csv(RAW); done = set(zip(r['date'], r['sym'], r['hh'], r['mm']))
    else: RAW.write_text('date,sym,hh,mm,status,qts,bid,ask,bsz,asz,win\n')
    todo = [(d, s, hh, mm) for d, s in zip(h['date'], h['sym']) for hh, mm in INSTANTS if (d, s, hh, mm) not in done]
    P('name-weeks', len(h), 'requests todo', len(todo)); n = 0
    with open(RAW, 'a') as f:
        for d, s, hh, mm in todo:
            y, m, dd = map(int, d.split('-')); t = datetime(y, m, dd, hh, mm, tzinfo=NY)
            t0 = time.time(); q, st = last_quote(c, s, t, StockQuotesRequest, DataFeed)
            f.write(f"{d},{s},{hh},{mm},{st}," + (f"{q[0]},{q[1]},{q[2]},{q[3]},{q[4]},{q[5]}" if q else ",,,,,") + '\n'); f.flush(); n += 1
            if n % 100 == 0: P('done', n, 'of', len(todo), d)
            time.sleep(max(0, 0.34 - (time.time() - t0)))
    P('FETCH DONE')


def spreads():
    """Join raw quotes to name-week table; half-spread bp; drop crossed/locked."""
    h = sample_weeks(); r = pd.read_csv(RAW)
    r['hs_bp'] = (r['ask'] - r['bid']) / 2 / ((r['ask'] + r['bid']) / 2) * 1e4
    r.loc[(r['status'] != 'ok') | (r['ask'] <= r['bid']) | (r['bid'] <= 0), 'hs_bp'] = np.nan
    w = r.pivot_table(index=['date', 'sym'], columns='hh', values='hs_bp', aggfunc='first'); w.columns = [f'hs{int(c)}' for c in w.columns]
    ok945 = r[(r.hh == 9) & (r.mm == 45)].set_index(['date', 'sym']); ok931 = r[(r.hh == 9) & (r.mm == 31)].set_index(['date', 'sym'])
    out = h.set_index(['date', 'sym'])[['kept', 'traded', 'w']].join(ok945['hs_bp'].rename('hs945')).join(ok931['hs_bp'].rename('hs931'))
    out['status945'] = ok945['status']; out['crossed945'] = ok945['bid'].notna() & out['hs945'].isna() & (ok945['status'] == 'ok')
    out['hs1000'] = r[(r.hh == 10)].set_index(['date', 'sym'])['hs_bp']
    out['side'] = out['kept'].map({0: 'entry', 1: 'reequalise', -1: 'exit'}); out['year'] = pd.to_datetime(out.index.get_level_values('date')).year
    out = out.reset_index(); out.to_csv(HERE / '1700w_spreads.csv', index=False)
    try: r.to_parquet(HERE / '1700w_quotes.parquet')
    except Exception as e: P('parquet failed (WARN)', e)
    return out, r


def analyse():
    out, r = spreads(); n = len(out); ok = out['hs945'].notna()
    P('earliest date served ok:', r[r.status == 'ok']['date'].min(), 'Mondays', out['date'].nunique())
    P('name-weeks requested', n, 'found@0945', int(ok.sum()), 'LOST', int((~ok).sum()), 'coverage %.1f%%' % (100 * ok.mean()), 'crossed/locked', int(out['crossed945'].sum()))
    P('status counts 0945', out['status945'].value_counts().to_dict())
    for c in ('hs945', 'hs931', 'hs1000'):
        x = out[c].dropna(); P(c, 'n', len(x), 'median %.2f mean %.2f P90 %.2f share>10bp %.3f' % (x.median(), x.mean(), x.quantile(.9), (x > 10).mean()))
    P('by year 0945 mean'); P(out.groupby('year')['hs945'].agg(['count', 'mean', 'median']).round(2).to_string())
    P('by year 0931 mean'); P(out.groupby('year')['hs931'].agg(['count', 'mean']).round(2).to_string())
    P('by side'); P(out.groupby('side')['hs945'].agg(['count', 'mean', 'median']).round(2).to_string())
    out['sz'] = pd.qcut(out['traded'].where(out['traded'] > 0), 3, labels=['small', 'mid', 'large'], duplicates='drop')
    P('by size tercile'); P(out.groupby('sz', observed=True)['hs945'].agg(['count', 'mean']).round(2).to_string())
    wt = out[ok & (out['traded'] > 0)]; P('notional-weighted mean 0945 %.2f bp' % np.average(wt['hs945'], weights=wt['traded']))
    ye = out.groupby('year')['hs945'].mean(); P('year rates (bp incl +1):', (ye + 1).round(2).to_dict())


def restate():
    """Three books at per-year measured cost (year mean 0945 half-spread + 1bp; years w/o sample = earliest sampled year mean x1.5)."""
    sys.path.insert(0, str(HERE)); import REBUILD_1700tu as RB
    out = pd.read_csv(HERE / '1700w_spreads.csv'); ye = out.groupby('year')['hs945'].mean() / 1e4; y0 = ye.index.min()
    rate_y = lambda y: (ye[y] if y in ye.index else ye[y0] * 1.5) + 1e-4
    P('rates by year', {y: round(rate_y(y) * 1e4, 2) for y in range(2017, 2027)})
    s = inspect.getsource(RB.features).replace('pos >= 272', 'pos >= 272').replace('p + 1 < 273', 'p < 272'); ns = dict(RB.__dict__); exec(s, ns)
    data = RB.load(); cal = pd.DatetimeIndex(data['SPY'][0]); rebs = RB.rebal_dates(cal)
    ts = np.array([cal[cal.searchsorted(r) - 1] for r in rebs], dtype='datetime64[ns]'); pct = RB.gate_series(); half = []
    for i in range(len(rebs)):
        p = pct.loc[:pd.Timestamp(ts[i])].iloc[-1] if pd.Timestamp(ts[i]) >= pct.index[0] else np.nan; half.append(bool(p < 0.20) if np.isfinite(p) else False)
    feats = ns['features'](data, cal, ts, None); plain, guarded = [], []
    for i, rr in enumerate(feats):
        if len(rr) < RB.TOPN: plain.append(None); guarded.append(None); continue
        plain.append([x[0] for x in sorted(rr, key=lambda x: -x[1])[:RB.TOPN]]); guarded.append([x[0] for x in sorted([x for x in rr if not x[2]], key=lambda x: -x[1])[:RB.TOPN]])
    rows = []
    for book, pk, hf in (('plain', plain, [False] * len(rebs)), ('guarded', guarded, [False] * len(rebs)), ('gated', guarded, half)):
        cash, hold, wk, daily = RB.CAP0, {}, [], []
        for i, reb in enumerate(rebs):
            nxt = rebs[i + 1] if i + 1 < len(rebs) else None; tg = pk[i]; E = cash + sum(hold.values())
            if tg is None: wk.append((reb, E)); hold = {}; continue
            w = 1.0 / (2 * RB.TOPN if hf[i] else RB.TOPN); rt = rate_y(pd.Timestamp(reb).year); cost = 0.0
            for s_ in set(hold) | set(tg): cost += abs((w * E if s_ in tg else 0.0) - hold.get(s_, 0.0)) * rt
            wk.append((reb, E - cost))
            if nxt is None: break
            cash = E - cost - w * E * len(tg); qty = {s_: (w * E) / RB.px(data, s_, reb, True) for s_ in tg}
            for dd in cal[(cal >= reb) & (cal < nxt)]: daily.append((dd, cash + sum(qty[s_] * RB.px(data, s_, dd, False) for s_ in qty)))
            hold = {s_: qty[s_] * RB.px(data, s_, nxt, True) for s_ in qty}
        wk = pd.Series(dict(wk)); wk = wk[wk.index >= pd.Timestamp('2017-02-06')]; dl = pd.Series(dict(daily)); dl = dl[dl.index >= pd.Timestamp('2017-02-06')]
        yrs = (wk.index[-1] - wk.index[0]).days / 365.25
        rows.append(dict(book=book, cagr=(wk.iloc[-1] / wk.iloc[0]) ** (1 / yrs) - 1, dd_daily_close=(dl / dl.cummax() - 1).min(), end=wk.iloc[-1])); P(rows[-1])
    pd.DataFrame(rows).to_csv(HERE / '1700w_restated.csv', index=False); P('RESTATE DONE')


{'fetch': fetch, 'analyse': analyse, 'restate': restate}[sys.argv[1]]()
