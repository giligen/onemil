"""Refuter-1 probe for cell 1,599 (v3): buys the 10:00-10:02 ET cbbo-1m SPY.OPT window on spike /
reference sessions, caches parquet, and reports quote two-sidedness, bid/ask sizes, spreads and
the cost to close a $10-wide put spread at ~20/30 delta on the ~45-DTE expiry. Every purchase is
cost-checked first and appended to spend.json (reloaded right before write) and to our own ledger."""
import os, sys, json, datetime as dt
import numpy as np, pandas as pd, databento as db
from dotenv import load_dotenv
HERE = os.path.dirname(os.path.abspath(__file__)); OV = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, OV)
from cell_1567 import implied_vol_put, bs_put_delta, R_RATE, Q_RATE
load_dotenv(os.path.join(OV, '..', '..', '.env'))
C = db.Historical(os.environ['DATABENTO_API_KEY'])
SPEND = os.path.join(OV, 'opt_cache', 'dbn', 'spend.json'); OWN = os.path.join(HERE, 'spend_own.json')
CAP = 150.0
DATES = sys.argv[1:] or ['2015-08-24', '2018-02-06', '2020-03-16', '2020-03-17', '2022-06-13', '2024-08-05', '2025-04-04', '2025-04-07', '2025-07-07']
spy = pd.read_parquet(os.path.join(OV, 'opt_cache', 'dbn', 'spy_prices.parquet')).set_index('day')
own = json.load(open(OWN)) if os.path.exists(OWN) else {'total_usd': 0.0, 'purchases': []}
out = []
for d in DATES:
    path = os.path.join(HERE, f'cbbo_{d}.parquet')
    if not os.path.exists(path):
        s = pd.Timestamp(f'{d} 10:00', tz='America/New_York').tz_convert('UTC'); e = s + pd.Timedelta(minutes=2)
        kw = dict(dataset='OPRA.PILLAR', schema='cbbo-1m', symbols=['SPY.OPT'], stype_in='parent', start=s.isoformat(), end=e.isoformat())
        cost = C.metadata.get_cost(**kw)
        st = json.load(open(SPEND))
        if st['total_usd'] + cost > CAP: print('CAP STOP', d, cost); break
        df = C.timeseries.get_range(**kw).to_df().reset_index()
        df.to_parquet(path)
        st = json.load(open(SPEND)); st['total_usd'] += cost
        st['purchases'].append({'ts': dt.datetime.utcnow().isoformat(), 'cost_usd': cost, 'label': f'refuter1_1599 cbbo-1m {d} 10:00-10:02'})
        json.dump(st, open(SPEND, 'w'), indent=2, default=str)
        own['total_usd'] += cost; own['purchases'].append({'d': d, 'cost': cost}); json.dump(own, open(OWN, 'w'), indent=2)
        print(f'bought {d} ${cost:.4f}', flush=True)
    df = pd.read_parquet(path)
    sym = df['symbol'].str.strip()
    df['right'] = sym.str[-9]; df['strike'] = sym.str[-8:].astype(int) / 1000.0
    df['expiry'] = pd.to_datetime(sym.str[-15:-9], format='%y%m%d')
    df['tr'] = pd.to_datetime(df['ts_recv'], utc=True).dt.tz_convert('America/New_York')
    df['te'] = pd.to_datetime(df['ts_event'], utc=True).dt.tz_convert('America/New_York')
    first = df[df['tr'].dt.strftime('%H:%M') <= '10:01'].sort_values('tr').groupby('symbol').first().reset_index()
    p = first[first.right == 'P'].copy()
    dte = (p['expiry'] - pd.Timestamp(d)).dt.days
    p = p[(dte >= 38) & (dte <= 52)]
    if not len(p): out.append({'day': d, 'note': 'no 38-52 DTE expiry'}); continue
    exp = p.loc[(dte[p.index] - 45).abs().idxmin(), 'expiry']; p = p[p.expiry == exp].copy()
    spot = spy.loc[d, 'spot_10'] if d in spy.index else np.nan
    p['two'] = (p.bid_px_00 > 0) & (p.ask_px_00 > 0)
    p['mid'] = (p.bid_px_00 + p.ask_px_00) / 2
    stale = (p['tr'] - p['te']).dt.total_seconds()
    row = {'day': d, 'spot10': spot, 'expiry': str(exp.date()), 'n_puts': len(p), 'two_sided_share': round(p.two.mean(), 3),
           'ts_recv_first': str(p['tr'].min().time()), 'stale_s_median': float(stale.median()), 'stale_s_p90': float(stale.quantile(.9))}
    if np.isfinite(spot):
        T = (exp - pd.Timestamp(d)).days / 365.0
        p['iv'] = [implied_vol_put(m, spot, k, T, R_RATE, Q_RATE) if m > 0 else None for m, k in zip(p.mid, p.strike)]
        p = p[p.iv.notna()].copy(); p['delta'] = [bs_put_delta(spot, k, T, R_RATE, Q_RATE, v) for k, v in zip(p.strike, p.iv)]
        for td in (0.20, 0.30):
            sh = p.iloc[(p.delta.abs() - td).abs().argsort()].iloc[0]; lg = p[np.isclose(p.strike, sh.strike - 10)]
            if not len(lg): row[f'd{int(td*100)}'] = 'no partner'; continue
            lg = lg.iloc[0]
            open_bidask = sh.bid_px_00 - lg.ask_px_00; open_mid = sh.mid - lg.mid
            close_bidask = sh.ask_px_00 - lg.bid_px_00
            row[f'd{int(td*100)}'] = dict(K=sh.strike, credit_bidask=round(open_bidask, 3), credit_mid=round(open_mid, 3),
                                         roundtrip_spread_cost=round(close_bidask - open_bidask, 3),
                                         sh_sz=(int(sh.bid_sz_00), int(sh.ask_sz_00)), lg_sz=(int(lg.bid_sz_00), int(lg.ask_sz_00)),
                                         sh_stale_s=float((sh.tr - sh.te).total_seconds()), lg_stale_s=float((lg.tr - lg.te).total_seconds()))
    out.append(row); print(row, flush=True)
json.dump(out, open(os.path.join(HERE, 'probe_out.json'), 'w'), indent=2, default=str)
print('own spend', own['total_usd'])
