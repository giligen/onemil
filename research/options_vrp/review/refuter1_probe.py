"""Refuter-1 probe for PREREG_1567 (obtainability / look-ahead / data lens). Read-only on the cache."""
import sys, os, math, pickle
import pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cell_1567 as C

cache = C.Cache()
mondays = sorted(cache.grid['monday'].unique())
wc = {}
rows = []
mkts = {}
for m in mondays:
    mkt = C.precompute_monday(cache, m, wc)
    mkts[m] = mkt
    r = {'monday': m}
    if mkt is None:
        r['status'] = 'VOID_monday'; rows.append(r); continue
    r.update(spot=mkt['spot'], expiry=mkt['expiry'], dte=mkt['dte'], iv_atm=mkt['iv_atm'])
    for d, w in [(0.15, 10.0), (0.30, 5.0), (0.20, 5.0)]:
        e = C.select_strikes(cache, m, mkt, d, w, wc)
        key = f'd{int(d*100)}w{int(w)}'
        if e is None:
            # diagnose: short found? long strike in grid? long printed?
            strikes = mkt['strikes']
            r[key] = 'VOID'
            continue
        r[key] = 'ok'
        r[key + '_short'] = e['short_strike']; r[key + '_long'] = e['long_strike']
        r[key + '_delta'] = e['short_delta']; r[key + '_credit'] = e['net_credit']
        r[key + '_otm'] = 1 - e['short_strike'] / mkt['spot']
        r[key + '_long_in_grid'] = bool((strikes := mkt['strikes'])['strike'].eq(e['long_strike']).any())
    rows.append(r)
df = pd.DataFrame(rows)
# SPY outcome over the nearest-45-DTE window for every Monday
sd = cache.spy_daily.sort_values('day').reset_index(drop=True)
def fwd(m, exp):
    sub = sd[(sd['day'] > m) & (sd['day'] <= exp)]
    s0 = sd[sd['day'] == m]['c']
    if sub.empty or s0.empty: return np.nan, np.nan
    return sub['c'].iloc[-1] / s0.iloc[0] - 1, sub['l'].min() / s0.iloc[0] - 1
df[['fwd_ret', 'fwd_minlow']] = df.apply(lambda r: pd.Series(fwd(r['monday'], r['expiry'])) if isinstance(r.get('expiry'), str) else pd.Series([np.nan, np.nan]), axis=1)
df.to_csv(os.path.join(HERE, 'refuter1_mondays.csv'), index=False)
print('warn', wc)
pd.set_option('display.width', 250)
print(df[['monday', 'spot', 'expiry', 'dte', 'iv_atm', 'd15w10', 'd15w10_delta', 'd15w10_otm', 'd15w10_credit', 'd30w5', 'd30w5_delta', 'fwd_ret', 'fwd_minlow']].to_string())
