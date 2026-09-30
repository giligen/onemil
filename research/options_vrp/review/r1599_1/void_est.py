"""Refuter-1: estimates the v3 builder's entry-VOID share on the PANEL Mondays by replaying cell_1599's
select_strikes/entry_fill rules (nearest-45 expiry in [38,52], strikes in [0.8*spot, spot], nearest-delta
short, exact K-10 partner, two-sided quotes) on the rebuild's independently-fetched 10:00 snapshots."""
import sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from cell_1567 import implied_vol_put, bs_put_delta, R_RATE, Q_RATE, size_position
spy = pd.read_parquet('opt_cache/dbn/spy_prices.parquet').set_index('day')
rows = []
for f in sorted(glob.glob('opt_cache/dbn/snapshots/*.parquet')):
    d = os.path.basename(f)[:10]; s = pd.read_parquet(f); spot = spy.loc[d, 'spot_10'] if d in spy.index else np.nan
    p = s[s.right == 'P'].copy(); p['exp'] = pd.to_datetime(p.expiry); dte = (p.exp - pd.Timestamp(d)).dt.days
    c = sorted(set(zip(p.exp[(dte >= 38) & (dte <= 52)], dte[(dte >= 38) & (dte <= 52)])), key=lambda x: abs(x[1] - 45))
    if not c or not np.isfinite(spot): rows.append({'day': d, 'reason': 'no_expiry_or_spot'}); continue
    e = c[0][0]; T = (e - pd.Timestamp(d)).days / 365; ch = p[(p.exp == e) & (p.strike >= .8 * spot) & (p.strike <= spot)].copy()
    ch['mid'] = (ch.bid + ch.ask) / 2
    ch = ch[ch.mid > 0]; ch['iv'] = [implied_vol_put(m, spot, k, T, R_RATE, Q_RATE) for m, k in zip(ch.mid, ch.strike)]
    ch = ch[ch.iv.notna()]; ch['delta'] = [bs_put_delta(spot, k, T, R_RATE, Q_RATE, v) for k, v in zip(ch.strike, ch.iv)]
    for td in (0.2, 0.3):
        r = {'day': d, 'td': td, 'expiry': str(e.date()), 'n_strikes': len(ch)}
        if not len(ch): r['reason'] = 'empty_ladder'; rows.append(r); continue
        sh = ch.iloc[(ch.delta.abs() - td).abs().argsort()].iloc[0]; lg = ch[np.isclose(ch.strike, sh.strike - 10)]
        r.update(K=sh.strike, dlt=round(sh.delta, 3))
        if not len(lg): r['reason'] = 'no_partner'; rows.append(r); continue
        lg = lg.iloc[0]
        if not (sh.bid > 0 and sh.ask > 0 and lg.bid > 0 and lg.ask > 0): r['reason'] = 'one_sided'; rows.append(r); continue
        cr = sh.bid - lg.ask; n, w = size_position(6500 / 6, 10.0, cr)
        r.update(credit=round(cr, 3), credit_mid=round((sh.bid + sh.ask - lg.bid - lg.ask) / 2, 3), contracts=n,
                 reason='ok' if n > 0 else 'sizing_skip'); rows.append(r)
df = pd.DataFrame(rows); df.to_csv('review/r1599_1/void_est.csv', index=False)
print(df.groupby(['td', 'reason']).size()); print('days', df.day.nunique(), df.day.min(), df.day.max())
ok = df[df.reason == 'ok']; print(ok.groupby('td')[['credit', 'credit_mid', 'contracts']].describe().T)
print(df[df.reason != 'ok'].head(12).to_string())
