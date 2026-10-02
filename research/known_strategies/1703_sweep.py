#!/usr/bin/env python3
"""Cells 1,703a-e (PREREG_1703_sweep.md, frozen). One process: panel loaded once for a, c, d; b from panel ETFs; e from crypto bars.
Run: bash scripts/research_run.sh -m 4000M python3 research/known_strategies/1703_sweep.py > research/known_strategies/1703.log 2>&1
Panel load / U2 rule / hygiene guard / cost model copied from research/momentum_weekly/1700s_lowvix.py (not modified)."""
from __future__ import annotations
import re, sys, time, warnings, gc
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
warnings.filterwarnings('ignore')
MW = Path('/home/ec2-user/onemil/research/momentum_weekly'); OUT = Path('/home/ec2-user/onemil/research/known_strategies')
PRICE_MIN, ADV_CUT, START_EQ = 10.0, 200_000_000.0, 50_000.0
WIN_START, WIN_END = pd.Timestamp('2017-01-01'), pd.Timestamp('2026-09-30')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|\bPREFERRED\b|\bRIGHTS?\b', re.I)
FLAT_C = 0.00105   # cell c: measured 10.5 bp per traded dollar (1,700w)
FLAT_ETF = 0.0002  # cell b: 2 bp per traded dollar on liquid ETFs (assumption, stated)
t0 = time.time()
def say(*a): print(f'[{time.time()-t0:5.0f}s]', *a, flush=True)

# ------------------------------------------------------------------ panel load
cols = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']
tbl = pq.read_table(MW / 'panel_2016_2026.parquet', columns=cols, read_dictionary=['symbol'])
tbl = tbl.cast(pa.schema([('symbol', tbl.schema.field('symbol').type), ('bar_date', tbl.schema.field('bar_date').type)] + [(c, pa.float32()) for c in cols[2:]]))
raw = tbl.to_pandas(self_destruct=True); del tbl
raw['bar_date'] = pd.to_datetime(raw['bar_date'])
_c = raw.symbol.values.codes; _cats = raw.symbol.values.categories; _d = raw.bar_date.values
if not bool(np.all((_c[1:] > _c[:-1]) | ((_c[1:] == _c[:-1]) & (_d[1:] >= _d[:-1])))): say('raw not sorted -- STOP'); raise SystemExit(4)
ETFS = ['SPY', 'EFA', 'SHY', 'IEF']
missing = [e for e in ETFS if e not in set(_cats)]
if missing: say('ETF MISSING FROM PANEL', missing); raise SystemExit(1)
etf = {}
for e in ETFS:
    m = (raw.symbol == e).values
    s = raw.loc[m, ['bar_date', 'open', 'close']].drop_duplicates('bar_date', keep='last').set_index('bar_date').sort_index(); etf[e] = s
tdays = pd.DatetimeIndex(sorted(etf['SPY'].index)); didx = {d: i for i, d in enumerate(tdays)}
say('ETFs in panel:', {e: (str(v.index.min().date()), str(v.index.max().date()), len(v)) for e, v in etf.items()})
assets = pd.read_csv(MW / '1700c_assets.csv', dtype={'symbol': str, 'name': str}); assets['name'] = assets['name'].fillna('')
excl = set(assets.loc[assets.name.str.contains(NAME_RE), 'symbol']) | {x for x in _cats if TEST_RE.match(x)}
bad_cat = np.array([(x in excl) or x in ETFS for x in _cats])
dup_next = np.append((_c[1:] == _c[:-1]) & (_d[1:] == _d[:-1]), False)
o_, h_, l_, c_, v_ = (raw[k].values for k in ('open', 'high', 'low', 'close', 'volume'))
keep = ~dup_next & ~((o_ <= 0) | (h_ <= 0) | (l_ <= 0) | (c_ <= 0)) & ~bad_cat[_c]
panel = pd.DataFrame({'symbol': pd.Categorical.from_codes(_c[keep], categories=_cats), 'bar_date': _d[keep], 'open': o_[keep], 'close': c_[keep]})
panel['symbol'] = panel.symbol.cat.remove_unused_categories()
hi = h_[keep].copy()
spread_ = (((h_[keep] - l_[keep]) / c_[keep]).clip(min=0) * 0.1).clip(max=0.002).astype(np.float32)
dv = (c_[keep] * v_[keep]).astype(np.float64); cl = c_[keep]
del raw, _c, _d, o_, h_, l_, c_, v_, keep, dup_next; gc.collect()
panel['spread'] = spread_; del spread_
codes = panel.symbol.values.codes; dts = panel.bar_date.values; N = len(panel)
say('panel rows after filters', N)

# ------------------------------------------------------------------ calendar: weekly Mondays, monthly firsts, signal dates = prior session
per = pd.Series(tdays).dt.to_period('W'); first = pd.Series(tdays).groupby(per).min().sort_index()
ent_w = [d for d in first if WIN_START <= d <= WIN_END]
mper = pd.Series(tdays).dt.to_period('M'); firstm = pd.Series(tdays).groupby(mper).min().sort_index()
ent_m = [d for d in firstm if WIN_START <= d <= WIN_END]
prior = {d: tdays[didx[d] - 1] for d in set(ent_w) | set(ent_m)}
sigdates = np.array(sorted(set(prior.values())), dtype='datetime64[ns]')
say('weekly rebalances', len(ent_w), 'monthly', len(ent_m))

# ------------------------------------------------------------------ per-symbol signals, kept only at signal dates (data through that close)
rows = []
bounds = np.flatnonzero(np.diff(codes)) + 1; starts = np.r_[0, bounds]; ends = np.r_[bounds, N]
for a, b in zip(starts, ends):
    sm = np.isin(dts[a:b], sigdates)
    if not sm.any(): continue
    c = pd.Series(cl[a:b].astype(np.float64)); adv = pd.Series(dv[a:b]).rolling(20, min_periods=20).mean().values
    vol = c.pct_change().rolling(252, min_periods=252).std().values
    sig12 = (c / c.shift(252) - 1).values; ret5 = (c / c.shift(5) - 1).values
    h52 = pd.Series(hi[a:b].astype(np.float64)).rolling(252, min_periods=252).max().values; ratio = c.values / h52
    ok = c.shift(273).notna().values
    cc = c.values; mv = np.zeros(b - a); mv[1:] = cc[1:] / cc[:-1] - 1; gp = np.zeros(b - a, bool); gp[1:] = np.diff(dts[a:b]).astype('timedelta64[D]').astype(np.int64) > 10
    ea = pd.Series(((mv > 2.0) | (mv < -0.75)).astype(np.float64)).rolling(273, min_periods=1).max().values > 0
    eb = pd.Series(gp.astype(np.float64)).rolling(273, min_periods=1).max().values > 0
    g = (ea | eb)
    ix = np.flatnonzero(sm)
    rows.append(pd.DataFrame({'code': codes[a], 'date': dts[a:b][ix], 'close': cc[ix], 'adv': adv[ix], 'vol': vol[ix], 'sig12': sig12[ix], 'ret5': ret5[ix], 'ratio': ratio[ix], 'ok': ok[ix], 'guard_bad': g[ix]}))
SG = pd.concat(rows, ignore_index=True); del rows, dv, cl, hi; gc.collect()
SG['symbol'] = [_cats_i for _cats_i in panel.symbol.cat.categories[SG.code.values]] if False else panel.symbol.cat.categories[SG.code.values].astype(str)
say('signal rows', len(SG))
elig = SG[(SG.close >= PRICE_MIN) & (SG.adv >= ADV_CUT) & ~SG.guard_bad]
u2 = elig[elig.ok]
def top(df, col, asc, n=20, req=None):
    out = {}
    for d, sub in df.groupby('date'):
        s = sub[sub[col].notna() & np.isfinite(sub[col])]
        if req is not None: s = s[req(s)]
        out[pd.Timestamp(d)] = s.sort_values([col, 'symbol'], ascending=[asc, True]).head(n).symbol.tolist()
    return out
rk_a = top(u2, 'ratio', False)                                  # a: closest to 52w high
rk_c = top(elig, 'ret5', True)                                  # c: 20 worst prior-week returns
rk_d = top(u2, 'vol', True, req=lambda s: s.sig12 > 0)          # d: 20 lowest 252d vol with positive 12m return
say('ranked a/c/d dates', len(rk_a), len(rk_c), len(rk_d))
cand = sorted({s for r in (rk_a, rk_c, rk_d) for v in r.values() for s in v}); cidx = {s: i for i, s in enumerate(cand)}
sub = panel[panel.symbol.isin(cand)]
def piv(col): return sub.pivot_table(index='bar_date', columns='symbol', values=col, aggfunc='last', observed=True).reindex(index=tdays, columns=cand)
O = piv('open').ffill().fillna(0.0).values.astype(np.float64)
RATE_SP = np.minimum(0.0005 + 0.5 * np.nan_to_num(piv('spread').values.astype(np.float64), nan=0.002), 0.002)
del panel, sub; gc.collect()
say('candidates', len(cand), O.shape)

# ------------------------------------------------------------------ engine (daily mark at the open, rebalance at the open, 1/N reset; cost on traded notional)
def simulate(O, RATE, rk, ents, T1, n):
    """rk: {signal date -> ordered symbol idx list}; ents: rebalance dates; equity marked at each open. Returns dict(E, dates, cost_f, trade_f, nrebal)."""
    ns = O.shape[1]; sh = np.zeros(ns); cash = START_EQ
    reb = {didx[d]: [x for x in rk[prior[d]][:n]] for d in ents if prior[d] in rk}
    T0 = min(reb); nd = T1 - T0 + 1; E = np.zeros(nd); cost_f = trade_f = 0.0
    for k in range(nd):
        ti = T0 + k; o = O[ti]; pre = cash + sh @ o; tr = cs = 0.0
        if ti in reb:
            top_ = [i for i in reb[ti] if o[i] > 0]; tgt = np.zeros(ns)
            if top_: tgt[top_] = pre / len(top_)
            delta = tgt - sh * o
            for i in np.where(delta < -1e-9)[0]:
                v = -delta[i]; c = RATE[ti, i] * v; cash += v - c; sh[i] -= v / o[i]; tr += v; cs += c
                if sh[i] <= 1e-12: sh[i] = 0
            for i in np.where(delta > 1e-9)[0]:
                v = delta[i]; c = RATE[ti, i] * v; cash -= v + c; sh[i] += v / o[i]; tr += v; cs += c
        E[k] = cash + sh @ o
        if pre > 0: cost_f += cs / pre; trade_f += tr / pre
    return dict(E=pd.Series(E, index=tdays[T0:T1 + 1]), cost_f=cost_f, trade_f=trade_f, nrebal=len(reb))
T1 = len(tdays) - 1
mk = lambda r: {d: [cidx[s] for s in v] for d, v in r.items()}
res = {}
res['a'] = simulate(O, RATE_SP, mk(rk_a), ent_w, T1, 20); say('a done', res['a']['nrebal'])
res['c'] = simulate(O, np.full_like(RATE_SP, FLAT_C), mk(rk_c), ent_w, T1, 20); say('c done', res['c']['nrebal'])
res['d'] = simulate(O, RATE_SP, mk(rk_d), ent_m, T1, 20); say('d done', res['d']['nrebal'])

# ------------------------------------------------------------------ b: dual momentum on panel ETFs (signal = prior close; trade next open)
Ec = pd.DataFrame({e: etf[e]['close'] for e in ETFS}).reindex(tdays); Eo = pd.DataFrame({e: etf[e]['open'] for e in ETFS}).reindex(tdays)
r12 = Ec / Ec.shift(252) - 1
holdings = ['SPY', 'EFA', 'IEF']; Ob = Eo[holdings].values.astype(np.float64)
rk_b = {}
for d in ent_m:
    p = prior[d]
    if np.isnan(r12.loc[p, ['SPY', 'EFA', 'SHY']]).any(): continue
    w = 'SPY' if r12.loc[p, 'SPY'] >= r12.loc[p, 'EFA'] else 'EFA'
    rk_b[p] = [holdings.index(w if r12.loc[p, w] > r12.loc[p, 'SHY'] else 'IEF')]
res['b'] = simulate(Ob, np.full_like(Ob, FLAT_ETF), rk_b, ent_m, T1, 1); say('b done', res['b']['nrebal'])
b_hold = pd.Series({d: holdings[rk_b[prior[d]][0]] for d in ent_m if prior[d] in rk_b})

# ------------------------------------------------------------------ e: crypto
cp = OUT / '1703e_crypto.parquet'
def fetch_crypto():
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    import yfinance as yf
    r = CryptoHistoricalDataClient().get_crypto_bars(CryptoBarsRequest(symbol_or_symbols=['BTC/USD', 'ETH/USD'], timeframe=TimeFrame.Day, start=pd.Timestamp('2017-01-01', tz='UTC'), end=pd.Timestamp('2026-10-01', tz='UTC'))).df.reset_index()
    al = pd.DataFrame({'symbol': r.symbol.str[:3], 'date': pd.to_datetime(r.timestamp).dt.tz_localize(None).dt.normalize(), 'close': r.close.astype(float), 'source': 'alpaca'})
    for s, tk in (('BTC', 'BTC-USD'), ('ETH', 'ETH-USD')):
        say('Alpaca', s, 'earliest', al[al.symbol == s].date.min().date(), 'days', int((al.symbol == s).sum()))
    parts = [al]
    for s, tk in (('BTC', 'BTC-USD'), ('ETH', 'ETH-USD')):
        y = yf.download(tk, start='2017-06-01', end='2021-02-01', interval='1d', progress=False, auto_adjust=False)
        y = y['Close'].squeeze() if 'Close' in y else y.iloc[:, 0]
        yd = pd.DataFrame({'symbol': s, 'date': pd.to_datetime(y.index).tz_localize(None).normalize(), 'close': y.values.astype(float), 'source': 'yfinance'})
        ov = al[al.symbol == s].merge(yd, on='date', suffixes=('_a', '_y'))
        say('price-scale check', s, 'overlap days', len(ov), 'median alpaca/yfinance close ratio', float((ov.close_a / ov.close_y).median()) if len(ov) else 'NA')
        parts.append(yd[yd.date < al[al.symbol == s].date.min()])
    d = pd.concat(parts).sort_values(['symbol', 'date']).drop_duplicates(['symbol', 'date']); d.to_parquet(cp); return d
cr = pd.read_parquet(cp) if cp.exists() else fetch_crypto()
say('crypto rows', cr.groupby(['symbol', 'source']).date.agg(['min', 'max', 'count']).to_dict('index'))
px = cr.pivot(index='date', columns='symbol', values='close').sort_index()
full = pd.date_range(px.index.min(), px.index.max(), freq='D'); lost = {s: int(px[s].reindex(full).isna().sum()) for s in px}
say('LOST calendar days (missing bars) per symbol:', lost); px = px.reindex(full).ffill()
ret = px.pct_change(); mom = px.shift(1) / px.shift(21) - 1; pos = (mom > 0).astype(float)       # position over day t from the signal at close t-1
dpos = pos.diff().abs().fillna(pos.iloc[0].abs())
leg = pos * ret - 0.001 * dpos
pr = leg.mean(axis=1)
pr = pr[pr.index >= '2018-01-01']; Ee = START_EQ * (1 + pr).cumprod()
e_turn = float(dpos.loc[pr.index].mean(axis=1).sum()); e_cost = e_turn * 0.001
res['e'] = dict(E=Ee, cost_f=e_cost, trade_f=e_turn, nrebal=int((dpos.sum(axis=1) > 0).sum()))
e_alwayslong = START_EQ * (1 + ret.mean(axis=1)[ret.index >= '2018-01-01']).cumprod()
say('e done; buy&hold BTC/ETH 50/50 CAGR for reference')

# ------------------------------------------------------------------ metrics
cur = pd.read_csv(MW / '1700u_curves_daily.csv', index_col=0, parse_dates=True)
G = cur['guard'].dropna(); say('GREF = 1700u_curves_daily.csv column guard (daily equity, sleeve with hygiene guard, 1700t/u); span', G.index.min().date(), G.index.max().date())
spyE = etf['SPY']['open'].reindex(tdays)
def cagr(E): 
    y = (E.index[-1] - E.index[0]).days / 365.25; return (E.iloc[-1] / E.iloc[0]) ** (1 / y) - 1
def mdd(E): return float((E / E.cummax() - 1).min())
def yearly(E): return (E.groupby(E.index.year).last() / E.groupby(E.index.year).last().shift(1).fillna(E.iloc[0])) - 1
def wk(E): return E.resample('W-FRI').last().dropna().pct_change().dropna()
def eqw(r): return (1 + r).cumprod()
def wstats(r):
    E = eqw(r); y = len(r) / 52.0; c = E.iloc[-1] ** (1 / y) - 1; d = mdd(E); return c, d, (c / abs(d) if d else np.nan)
def episodes(E, k=3):
    dd = E / E.cummax() - 1; eps = []; i = 0; idx = E.index
    peak = 0
    while peak < len(E) - 1:
        # next peak-to-recovery segment
        j = peak + 1
        while j < len(E) and E.iloc[j] < E.iloc[peak]: j += 1
        seg = dd.iloc[peak:j]
        if len(seg) > 1: eps.append((float(seg.min()), idx[peak], seg.idxmin()))
        peak = j if j < len(E) else len(E) - 1
        if j >= len(E): break
    return sorted(eps)[:k]
EPS = episodes(G); say('GREF 3 deepest episodes', [(round(d, 3), str(p.date()), str(t.date())) for d, p, t in EPS])
GW = wk(G)
out = []
sh_row = lambda E: None
for k, name in (('a', '52w-high momentum'), ('b', 'dual momentum'), ('c', 'weekly ST reversal'), ('d', 'low-vol'), ('e', 'crypto TSM')):
    r = res[k]; E = r['E']; E = E[E.index <= WIN_END]; ys = (E.index[-1] - E.index[0]).days / 365.25
    dr = E.pct_change().dropna(); ann = 365 if k == 'e' else 252
    yy = yearly(E); sy = yearly(spyE.reindex(E.index).ffill().dropna()) if k != 'e' else yearly(spyE.reindex(pd.date_range(E.index[0], E.index[-1])).ffill().dropna())
    beats = int((yy.reindex(sy.index) > sy).sum()); nyr = int(len(yy))
    h1 = E[E.index < '2022-01-01']; h2 = E[E.index >= '2022-01-01']
    cw = wk(E); j = pd.concat([GW.rename('g'), cw.rename('c')], axis=1).dropna()
    corr = float(j.g.corr(j.c)); wm = pd.Series(False, index=j.index)
    for _, p, t in EPS: wm |= (j.index > p) & (j.index <= t + pd.Timedelta(days=6))
    corr_ep = float(j.g[wm].corr(j.c[wm])) if wm.sum() > 5 else np.nan
    g_c, g_d, g_r = wstats(j.g); h_c, h_d, h_r = wstats(0.5 * j.g + 0.5 * j.c); s_c, s_d, s_r = wstats(j.g + 0.5 * j.c)
    row = dict(cell='1703' + k, strategy=name, start=str(E.index[0].date()), end=str(E.index[-1].date()), CAGR=cagr(E), maxDD=mdd(E), CAGR_over_DD=cagr(E) / abs(mdd(E)),
               sharpe=float(dr.mean() / dr.std() * np.sqrt(ann)), worst_year=float(yy.min()), yrs_beat_SPY=f'{beats}/{nyr}', turnover_per_yr=r['trade_f'] / ys, cost_drag_per_yr=r['cost_f'] / ys,
               H1_CAGR=cagr(h1) if len(h1) > 30 else np.nan, H2_CAGR=cagr(h2), corr_GREF=corr, corr_GREF_deep_episodes=corr_ep, n_ep_weeks=int(wm.sum()), n_weeks=len(j),
               GREF_w_CAGR=g_c, GREF_w_DD=g_d, GREF_w_ratio=g_r, fifty_CAGR=h_c, fifty_DD=h_d, fifty_ratio=h_r, stack_CAGR=s_c, stack_DD=s_d, stack_ratio=s_r, end_equity=float(E.iloc[-1]))
    row['P_standalone'] = bool(row['CAGR'] >= 0.10 and row['maxDD'] > -0.30); row['P_corr'] = bool(corr <= 0.5); row['P_ratio'] = bool(h_r >= 0.85)
    row['P_halves'] = bool(row['H1_CAGR'] > 0 and row['H2_CAGR'] > 0); row['PASS'] = all(row[x] for x in ('P_standalone', 'P_corr', 'P_ratio', 'P_halves'))
    row['replaces_GREF'] = bool(row['CAGR'] > 0.287 and row['maxDD'] > -0.381)
    out.append(row); say(k, {x: (round(v, 4) if isinstance(v, float) else v) for x, v in row.items()})
df = pd.DataFrame(out); df.to_csv(OUT / '1703_cells.csv', index=False)
say('GREF full-window daily CAGR/DD', cagr(G), mdd(G), ' SPY', cagr(spyE.reindex(G.index).dropna()), mdd(spyE.reindex(G.index).dropna()))
say('e buy&hold 50/50 BTC/ETH CAGR', cagr(e_alwayslong), 'maxDD', mdd(e_alwayslong))
say('b holdings share', b_hold.value_counts().to_dict())

# ------------------------------------------------------------------ causality / shift test, one date per cell (recompute signal from data strictly before D)
D = pd.Timestamp('2023-06-05'); pD = prior[D]; say('SHIFT TEST date', D.date(), 'signal date', pD.date(), '(< D: ', pD < D, ')')
for k, rk_ in (('a', rk_a), ('c', rk_c), ('d', rk_d)):
    s0 = rk_[pD][0]; h = pq.read_table(MW / 'panel_2016_2026.parquet', columns=['symbol', 'bar_date', 'high', 'close'], filters=[('symbol', '=', s0)]).to_pandas()
    h['bar_date'] = pd.to_datetime(h.bar_date); h = h[h.bar_date < D].sort_values('bar_date'); c = h.close.astype(float).reset_index(drop=True)
    stored = SG[(SG.symbol == s0) & (SG.date == pD)].iloc[0]
    if k == 'a': mine, st = c.iloc[-1] / h.high.astype(float).iloc[-252:].max(), stored.ratio
    elif k == 'c': mine, st = c.iloc[-1] / c.iloc[-6] - 1, stored.ret5
    else: mine, st = c.pct_change().iloc[-252:].std(), stored.vol
    say('shift', k, s0, 'recomputed', round(float(mine), 6), 'stored', round(float(st), 6), 'match', bool(abs(mine - st) < 1e-4), '| last row used', h.bar_date.iloc[-1].date())
pb = rk_b[prior[ent_m[ent_m.index(pd.Timestamp('2023-06-01'))]]] if pd.Timestamp('2023-06-01') in ent_m else None
say('shift b: holdings decided from closes < rebalance date; e: pos[t] uses close[t-1]/close[t-21] (shift(1) in code); same-day unshifted version e CAGR for contrast:',
    cagr(START_EQ * (1 + (((px / px.shift(20) - 1) > 0).astype(float) * ret).mean(axis=1).loc['2018-01-01':]).cumprod()))
say('DONE')
