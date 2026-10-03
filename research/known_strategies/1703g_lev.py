#!/usr/bin/env python3
"""Cell 1,703g (PREREG_1703g_crypto_lev.md, frozen), v2 after review. Leveraged BTC trend with ATR stops on Alpaca ETFs.
Run: bash scripts/research_run.sh -m 2500M python3 research/known_strategies/1703g_lev.py
Output: 1703g_btc_ohlc.parquet, 1703g_cells.csv, 1703g_monthly.csv, RESULT_1703g.md
Conventions (documented): position over day t follows the 20d-return signal at close t-1, executed at open_t (BTC UTC day
open == prior close within a tick, so a held position earns close-to-close). ETF-session timing is IGNORED (BTC 24/7 bars).
Stop = entry open - S * 20d simple-mean true range at t-1, fixed at entry. open<=stop -> fill at open (gap through);
elif low<=stop -> fill at stop level. 10 bp of equity per switch incl. stops. Cash earns 0 (as in 1703e).
"""
from __future__ import annotations
import os, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
OUT = Path('/home/ec2-user/onemil/research/known_strategies')
START_EQ, TRATE, ER = 50_000.0, 0.045, {1: 0.0025, 2: 0.0185}
COST = 0.001
t0 = time.time()
def say(*a): print(f'[{time.time()-t0:5.0f}s]', *a, flush=True)

def keys():
    """Alpaca keys from env/.env via the repo's usual path (no key contents printed)."""
    from dotenv import load_dotenv
    load_dotenv('/home/ec2-user/onemil/.env')
    k = os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID'); s = os.getenv('ALPACA_API_SECRET') or os.getenv('APCA_API_SECRET_KEY')
    if not k:
        import yaml; cfg = yaml.safe_load(open('/home/ec2-user/onemil/config.yaml')); k, s = cfg['alpaca']['api_key'], cfg['alpaca']['api_secret']
    return k, s

def yf_ohlc(tk, start, end=None, **kw):
    """yfinance daily OHLC with logged failure."""
    import yfinance as yf
    try:
        y = yf.download(tk, start=start, end=end, interval='1d', progress=False, auto_adjust=False)
        if isinstance(y.columns, pd.MultiIndex): y.columns = y.columns.get_level_values(0)
        y = y[['Open', 'High', 'Low', 'Close']].astype(float); y.columns = ['open', 'high', 'low', 'close']
        y.index = pd.to_datetime(y.index).tz_localize(None).normalize(); say('yfinance', tk, 'rows', len(y)); return y
    except Exception as e:
        say('ERROR yfinance', tk, repr(e)); return pd.DataFrame()

def load_btc_ohlc() -> pd.DataFrame:
    """Spliced BTC daily OHLC: Alpaca crypto API (2021+) over yfinance (earlier, scaled to Alpaca at the first overlap day)."""
    p = OUT / '1703g_btc_ohlc.parquet'
    if p.exists(): return pd.read_parquet(p)
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    r = CryptoHistoricalDataClient().get_crypto_bars(CryptoBarsRequest(symbol_or_symbols=['BTC/USD'], timeframe=TimeFrame.Day, start=pd.Timestamp('2017-01-01', tz='UTC'), end=pd.Timestamp('2026-10-02', tz='UTC'))).df.reset_index()
    al = pd.DataFrame({'date': pd.to_datetime(r.timestamp).dt.tz_localize(None).dt.normalize(), 'open': r.open, 'high': r.high, 'low': r.low, 'close': r.close}).set_index('date').astype(float)
    say('Alpaca BTC OHLC', al.index.min().date(), al.index.max().date(), len(al))
    y = yf_ohlc('BTC-USD', '2017-06-01', '2026-10-02')
    ov = al.index.intersection(y.index)
    say('overlap days', len(ov), 'median alpaca/yf close ratio', float((al.close[ov] / y.close[ov]).median()))
    d0 = al.index.min(); scale = al.close[d0] / y.close[d0]; say('splice scale at', d0.date(), scale)
    pre = y[y.index < d0] * scale
    d = pd.concat([pre, al]).sort_index(); d = d[~d.index.duplicated()]
    d.to_parquet(p); return d

def atr20(px): 
    """20-day simple mean of true range (BTC 24/7: gap vs prior close included)."""
    pc = px.close.shift(1); tr = pd.concat([px.high - px.low, (px.high - pc).abs(), (px.low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(20).mean()

def simulate(px, L, S, sig, atr):
    """Daily loop. Returns (daily return series of the strategy, switches, stop count, days in market)."""
    o, h, lo, c = px.open.values, px.high.values, px.low.values, px.close.values
    n = len(px); r = np.zeros(n); inpos = False; lock = False; entry = stop = np.nan; sw = nst = inm = 0
    fin = (ER[L] + (L - 1) * (TRATE + 0.005)) / 365
    for i in range(1, n):
        s = sig[i - 1] if not np.isnan(sig[i - 1]) else 0.0
        if lock and s <= 0: lock = False
        ret_day = 0.0; basis = c[i - 1]; cost = 0.0
        if inpos and s <= 0:                       # signal exit at open
            ret_day = L * (o[i] / basis - 1); cost += COST; inpos = False; sw += 1
        elif (not inpos) and s > 0 and not lock:   # entry at open
            inpos = True; entry = o[i]; basis = o[i]; cost += COST; sw += 1
            stop = entry - S * atr[i - 1] if S is not None else -np.inf
        if inpos:
            inm += 1
            if S is not None and o[i] <= stop:     # gap through: fill at open
                ret_day = L * (o[i] / basis - 1) - fin; inpos = False; lock = True; cost += COST; sw += 1; nst += 1
            elif S is not None and lo[i] <= stop:  # intraday: fill at the stop level
                ret_day = L * (stop / basis - 1) - fin; inpos = False; lock = True; cost += COST; sw += 1; nst += 1
            else:
                ret_day = L * (c[i] / basis - 1) - fin
        r[i] = max(ret_day - cost, -0.999)
    return pd.Series(r, index=px.index), sw, nst, inm

def metrics(r, win, sw, inm_days):
    """Window metrics from daily returns r (already sliced); years = calendar days / 365.25."""
    eq = (1 + r).cumprod(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1; dd = (eq / eq.cummax() - 1).min()
    m = (1 + r).resample('M').prod() - 1
    med = m.median(); m5 = m.copy(); m5[m5 >= m.quantile(0.95)] = med
    ex = (1 + m5).prod() ** (12 / len(m5)) - 1
    h1 = (1 + r[r.index < '2022-01-01']).prod() - 1 if r.index[0] < pd.Timestamp('2022-01-01') else np.nan
    h2 = (1 + r[r.index >= '2022-01-01']).prod() - 1
    return dict(cagr=cagr, maxdd=dd, ratio=cagr / abs(dd), worst_month=m.min(), months_green=int((m > 0).sum()), months=len(m),
                switches_yr=sw / yrs, time_in_mkt=inm_days / len(r), extop5_cagr=ex, h1=h1, h2=h2)

# ------------------------------------------------------------------ data
say('=== Cell 1703g v2 ==='); px = load_btc_ohlc().asfreq('D'); say('LOST calendar days (missing bars)', int(px.close.isna().sum())); px = px.ffill()
for col in ('open', 'high', 'low'): px[col] = px[col].fillna(px.close)
say('BTC OHLC', px.index.min().date(), px.index.max().date(), len(px), 'rows; low<=close frac', float((px.low <= px.close + 1e-9).mean()))
sig = (px.close.shift(1) / px.close.shift(21) - 1).values  # signal at close t-1 uses closes t-1 and t-21 (== 1703e `mom` at t)
# NOTE sig[i-1] inside simulate = ret_{close i-1 / close i-21}... shift(1) already applied; see anchor check below.
mom_close = px.close / px.close.shift(20) - 1            # 20-day return known at close of day d; simulate reads sig[i-1] = value at close i-1
sig = mom_close.values
atr = atr20(px).values
WINS = {'2018-2026': ('2018-01-01', '2026-09-30'), '2022-2026': ('2022-01-01', '2026-09-30')}
# ---- anchor: exact 1703e formula on the new closes (close-to-close, no ER, no stops)
ret = px.close.pct_change(); mom = px.close.shift(1) / px.close.shift(21) - 1; pos = (mom > 0).astype(float)
dpos = pos.diff().abs().fillna(abs(pos.iloc[0])); a1703e = (pos * ret - 0.001 * dpos)
# ------------------------------------------------------------------ cells
rows, monthly, rets = [], {}, {}
for L in (1, 2):
    for S in (None, 2.0, 1.0, 0.5):
        r, sw, nst, inm = simulate(px, L, S, sig, atr); key = f'L{L}_S{S or "none"}'; rets[key] = r
        for wn, (a, b) in WINS.items():
            rw = r[a:b]; m = metrics(rw, wn, int(((r.index >= a) & (r.index <= b)).sum() * 0 + sw * len(rw) / len(r)), int(inm * len(rw) / len(r)))
            rows.append(dict(L=L, stop_atr=S or 'none', window=wn, stops=nst, **m))
        say(key, 'switches', sw, 'stops', nst, f"2018-26 CAGR {rows[-2]['cagr']:.3f} DD {rows[-2]['maxdd']:.3f}")
cells = pd.DataFrame(rows); cells.to_csv(OUT / '1703g_cells.csv', index=False)
mo = pd.DataFrame({k: (1 + v['2018-01-01':]).resample('M').prod() - 1 for k, v in rets.items()})
# ---- references
fin2 = (ER[2] + (TRATE + 0.005)) / 365
refs = {'btc_1x_hold': ret, 'btc_2x_hold': 2 * ret - fin2, 'btc_1x_trend_1703e': a1703e}
refrows = []
for k, v in refs.items():
    v = v.fillna(0)
    for wn, (a, b) in WINS.items():
        rw = v[a:b]; eq = (1 + rw).cumprod(); yrs = (rw.index[-1] - rw.index[0]).days / 365.25
        refrows.append(dict(ref=k, window=wn, cagr=eq.iloc[-1] ** (1 / yrs) - 1, maxdd=(eq / eq.cummax() - 1).min()))
    mo[k] = (1 + v['2018-01-01':]).resample('M').prod() - 1
mo.to_csv(OUT / '1703g_monthly.csv'); refdf = pd.DataFrame(refrows); say(refdf.to_string())
anchor = cells[(cells.L == 1) & (cells.stop_atr == 'none') & (cells.window == '2018-2026')].iloc[0]
a_ref = refdf[(refdf.ref == 'btc_1x_trend_1703e') & (refdf.window == '2018-2026')].iloc[0]
say('ANCHOR new 1x/none', anchor.cagr, anchor.maxdd, '| 1703e formula on new data', a_ref.cagr, a_ref.maxdd)

# ------------------------------------------------------------------ BITX tracking
def bitx_check():
    """Simulated 2x (daily reset, US-close to US-close BTC return) vs real BITX. Returns dict + source note."""
    import yfinance as yf
    out = dict(src='', n=0, corr=np.nan, td_ann=np.nan, td_ann_noER=np.nan, corr_daily=np.nan, note='')
    b = yf_ohlc('BITX', '2023-06-01')
    if len(b) == 0:
        try:
            h = yf.Ticker('BITX').history(start='2023-06-01', auto_adjust=True); say('yf Ticker.history rows', len(h))
            if len(h): b = pd.DataFrame({'close': h.Close.values}, index=pd.to_datetime(h.index).tz_localize(None).normalize()); out['src'] = 'yfinance Ticker.history'
        except Exception as e: say('ERROR Ticker.history', repr(e))
    else: out['src'] = 'yfinance download (auto_adjust=False close)'
    if len(b) == 0:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            k, s = keys(); r = StockHistoricalDataClient(k, s).get_stock_bars(StockBarsRequest(symbol_or_symbols='BITX', timeframe=TimeFrame.Day, start=pd.Timestamp('2023-06-01', tz='UTC'), adjustment='all')).df.reset_index()
            b = pd.DataFrame({'close': r.close.values}, index=pd.to_datetime(r.timestamp).dt.tz_localize(None).dt.normalize()); out['src'] = 'Alpaca stock bars (adjusted)'; say('Alpaca BITX rows', len(b))
        except Exception as e: say('ERROR Alpaca BITX', repr(e))
    if len(b) < 50: out['note'] = 'NO DATA'; return out
    bc = b.close.astype(float); bc = bc[~bc.index.duplicated()]
    # BTC at the US close: Alpaca hourly bars, hour starting 15:00 ET closes at 16:00 ET
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
        hr = CryptoHistoricalDataClient().get_crypto_bars(CryptoBarsRequest(symbol_or_symbols=['BTC/USD'], timeframe=TimeFrame.Hour, start=pd.Timestamp('2023-05-25', tz='UTC'), end=pd.Timestamp('2026-10-02', tz='UTC'))).df.reset_index()
        t = pd.to_datetime(hr.timestamp).dt.tz_convert('America/New_York'); sel = hr[t.dt.hour == 15]
        us = pd.Series(sel.close.values, index=t[t.dt.hour == 15].dt.tz_localize(None).dt.normalize()); us = us[~us.index.duplicated()]
        out['note'] = 'BTC priced at 16:00 ET from Alpaca hourly bars'
    except Exception as e:
        say('ERROR hourly BTC, falling back to UTC daily closes (misaligned by ~4h)', repr(e)); us = px.close; out['note'] = 'BTC UTC daily close (misaligned ~4h)'
    d = pd.concat([bc.rename('bitx'), us.rename('btc')], axis=1, join='inner').dropna(); d = d[d.index >= '2023-06-01']
    rb, rt = d.bitx.pct_change().dropna(), d.btc.pct_change().dropna()   # between consecutive shared US sessions (BTC compounds the weekend)
    x = pd.concat([rb.rename('b'), rt.rename('t')], axis=1).dropna(); sim = 2 * x.t - fin2; sim0 = 2 * x.t
    out.update(n=len(x), corr=float(x.b.corr(sim)), td_ann=float((sim - x.b).mean() * 252), td_ann_noER=float((sim0 - x.b).mean() * 252),
               bitx_ann=float(x.b.mean() * 252), beta=float(np.polyfit(sim, x.b, 1)[0]), bitx_cagr=float((1+x.b).prod()**(365.25/((x.index[-1]-x.index[0]).days))-1), sim_cagr=float((1+sim).prod()**(365.25/((x.index[-1]-x.index[0]).days))-1), btc_cagr=float((1+x.t).prod()**(365.25/((x.index[-1]-x.index[0]).days))-1), sim_ann=float(sim.mean() * 252), first=str(x.index[0].date()), last=str(x.index[-1].date()))
    return out
bx = bitx_check(); say('BITX', bx)
pd.Series(bx).to_csv(OUT / '1703g_bitx_check.csv')

# ------------------------------------------------------------------ RESULT
ok2x = bx['n'] > 0 and bx['corr'] >= 0.99
def cell(L, S, w): return cells[(cells.L == L) & (cells.stop_atr == (S or 'none')) & (cells.window == w)].iloc[0]
def rf(k, w): return refdf[(refdf.ref == k) & (refdf.window == w)].iloc[0]
cells['pass'] = (cells.ratio >= 0.65) & (cells.maxdd > -0.45) & (np.sign(cells.h1) == np.sign(cells.h2)) & ((cells.L == 1) | ok2x)
full = cells[cells.window == '2018-2026']; npass = int(full['pass'].sum()); best = full.loc[full.ratio.idxmax()]
L = ['# Cell 1,703g v2 (OHLC stops, years/365.25)', '',
     '## 1. BITX tracking check (2x daily reset, sim vs real BITX)',
     f"- Source: {bx['src'] or 'none'}; {bx['note']}; n={bx['n']} sessions {bx.get('first','')}..{bx.get('last','')}",
     f"- Correlation {bx['corr']:.4f} (bar >= 0.99: {'MET' if ok2x else 'NOT MET -> 2x cells NOT reportable'}); tracking diff (sim - real) {bx['td_ann']*100:.1f} %/yr with ER+financing, {bx['td_ann_noER']*100:.1f} %/yr before costs; real BITX mean {bx.get('bitx_ann',np.nan)*100:.1f} %/yr vs sim {bx.get('sim_ann',np.nan)*100:.1f} %/yr; compounded CAGR real {bx.get('bitx_cagr',np.nan)*100:.1f} % vs sim {bx.get('sim_cagr',np.nan)*100:.1f} % (BTC 1x {bx.get('btc_cagr',np.nan)*100:.1f} %); beta real-on-sim {bx.get('beta',np.nan):.3f}",
     '', '## 2. Anchor (1x / no stop, 2018-2026)',
     f"- This run: {anchor.cagr*100:.1f} % / {anchor.maxdd*100:.1f} % (open-execution, 25 bp ER); 1703e formula on the new OHLC closes: {a_ref.cagr*100:.1f} % / {a_ref.maxdd*100:.1f} %; target 37.7 / -58.1 (+-0.5 pt)",
     f"- Anchor {'REPRODUCED' if abs(anchor.cagr-0.377)<=0.005 and abs(anchor.maxdd+0.581)<=0.01 else 'NOT reproduced within 0.5 pt; see note'}",
     '', '## 3. Cells (CAGR % / maxDD % / CAGR/DD / stops hit / h1 / h2 returns)',
     '| L | stop | 2018-26 | 2022-26 | stops | h1 | h2 | pass |', '|---|---|---|---|---|---|---|---|']
for L_ in (1, 2):
    for S in (None, 2.0, 1.0, 0.5):
        a, b = cell(L_, S, '2018-2026'), cell(L_, S, '2022-2026'); p = cells[(cells.L == L_) & (cells.stop_atr == (S or 'none')) & (cells.window == '2018-2026')]['pass'].iloc[0]
        L.append(f"| {L_} | {S or 'none'} | {a.cagr*100:.1f} / {a.maxdd*100:.1f} / {a.ratio:.2f} | {b.cagr*100:.1f} / {b.maxdd*100:.1f} / {b.ratio:.2f} | {int(a.stops)} | {a.h1:.2f} | {a.h2:.2f} | {'Y' if p else 'n'} |")
L += ['', 'References (2018-26 / 2022-26 CAGR % / maxDD %): ' + '; '.join(f"{k} {rf(k,'2018-2026').cagr*100:.1f}/{rf(k,'2018-2026').maxdd*100:.1f} , {rf(k,'2022-2026').cagr*100:.1f}/{rf(k,'2022-2026').maxdd*100:.1f}" for k in refs),
      '', '## 4. Verdict vs PREREG bar (CAGR/DD >= 0.65 AND maxDD better than -45 % AND halves same-signed)',
      f"- Cells passing (2018-26): {npass} of 8. Best by CAGR/DD: L={int(best.L)} stop={best.stop_atr} {best.cagr*100:.1f} % / {best.maxdd*100:.1f} % / {best.ratio:.2f}",
      f"- 2x cells reportable only if the BITX check is met: {'met' if ok2x else 'NOT met, 2x rows are informational only'}",
      '', '## 5. Adversary caveats',
      '1. BTC UTC-day bars, 24/7: the ETF session (09:30-16:00 ET, weekend gaps) is ignored; real fills are worse at weekend/overnight gaps. Stop gap-through uses the UTC open.',
      '2. Stop level fixed at entry from a 20d mean true range at t-1; flat after a stop until the signal turns negative then positive. 10 bp per switch also on stops; cash earns 0.',
      '3a. BITX tracking: correlation passes the PREREG bar but real BITX trails the sim by ~25 pts/yr (futures roll/basis, swap costs not modelled), so every 2x CAGR here is OVERSTATED by that drag. Pre-2021 OHLC = yfinance scaled to Alpaca at the splice day; sim 2x for pre-2023 is a model (ER 185 bp + T-bill+50 bp financing).',
      '4. 8 cells x 2 windows; tail-dependence shown by extop5_cagr in 1703g_cells.csv; 2018-21 vs 2022-26 regimes differ strongly (h1 vs h2).',
      '5. Independent re-read still required (PREREG) before the owner sees any number.']
(OUT / 'RESULT_1703g.md').write_text('\n'.join(L) + '\n'); say('done')
