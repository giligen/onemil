"""Independent re-read of cell 1,703g 2x/no-stop from the PREREG prose, plus the executable version on REAL BITX closes."""
import pandas as pd, numpy as np, yfinance as yf
o = pd.read_parquet('research/known_strategies/1703g_btc_ohlc.parquet')
if 'date' in o.columns: o = o.set_index('date')
o.index = pd.to_datetime(o.index).tz_localize(None); o = o.sort_index()
px = o.close; pos = ((px.shift(1) / px.shift(21) - 1) > 0).astype(float)      # signal at close t-1, held over day t
sw = pos.diff().abs().fillna(0)
def book(L, er):
    r = px.pct_change()
    return pos * (L * r - (er + (L - 1) * 0.05) / 365) - 0.001 * sw
def st(r, a, z):
    r = r[a:z].dropna(); e = (1 + r).cumprod(); y = len(r) / 365.25
    return '%.1f%% / %.1f%%' % (100 * (e.iloc[-1] ** (1 / y) - 1), 100 * (e / e.cummax() - 1).min())
for L, er in [(1, .0025), (2, .0185)]:
    b = book(L, er); print('sim L=%d' % L, '2018-26', st(b, '2018-01-01', '2026-09-30'), '| 2022-26', st(b, '2022-01-01', '2026-09-30'), flush=True)
# executable: real BITX / IBIT closes, signal from BTC UTC close of the prior calendar day, position over the next session
y = yf.download(['BITX', 'IBIT', 'BTC-USD'], start='2023-06-01', end='2026-10-02', auto_adjust=True, progress=False)['Close']
y.index = pd.to_datetime(y.index).tz_localize(None)
sess = y[['BITX', 'IBIT']].dropna(how='all')
sig = pos.reindex(sess.index, method='ffill')           # signal known at the prior UTC close
rb = sess.BITX.pct_change(); ri = sess.IBIT.pct_change(); rbtc = y['BTC-USD'].reindex(sess.index).pct_change()
swb = sig.diff().abs().fillna(0)
rows = {'BITX hold': rb, 'BITX trend': sig * rb - 0.001 * swb, 'IBIT hold': ri, 'IBIT trend': sig * ri - 0.001 * swb, 'BTC hold (sessions)': rbtc}
a0 = sess.BITX.first_valid_index(); i0 = sess.IBIT.first_valid_index()
print('\nREAL ETFs, session days, signal = BTC 20-day at prior UTC close')
for k, r in rows.items():
    a = i0 if 'IBIT' in k else a0
    rr = r[a:].dropna(); e = (1 + rr).cumprod(); yrs = (rr.index[-1] - rr.index[0]).days / 365.25
    print('%-20s from %s  CAGR %6.1f%%  maxDD %6.1f%%  x%.2f' % (k, a.date(), 100 * (e.iloc[-1] ** (1 / yrs) - 1), 100 * (e / e.cummax() - 1).min(), e.iloc[-1]), flush=True)
m = pd.DataFrame({k: (1 + r['2025-04-01':]).resample('ME').prod() - 1 for k, r in rows.items() if 'BTC' not in k})
m.index = m.index.strftime('%Y-%m'); print('\n' + (m * 100).round(1).to_string())
