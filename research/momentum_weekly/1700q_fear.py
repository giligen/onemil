"""Cell 1,700q (PREREG_1700q.md, FROZEN): do fear gauges (VIX family, credit, breadth, F&G proxy) predict the next week?
Sleeve engine = the verbatim prefix of 1700o_hedge.py (loader + simulate + REF check), exec'd so there is ONE copy of it.
All predictors use data through the prior trading day's close (the Friday close); target week = Monday open -> next Monday open."""
from __future__ import annotations
import io, re, sys, urllib.request, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')
HERE = Path('/home/ec2-user/onemil/research/momentum_weekly')
src = (HERE / '1700o_hedge.py').read_text().splitlines()
cut = next(i for i, l in enumerate(src) if 'REF MISMATCH' in l) + 1
exec(compile('\n'.join(src[:cut]).replace('1700o.log', '1700q_engine.log'), '1700o_prefix', 'exec'), globals())
for _n in ('g', 'ETF', 'assets', 'sr'): globals().pop(_n, None)
import gc as _gc, ctypes as _ct; _gc.collect(); _ct.CDLL('libc.so.6').malloc_trim(0)
print('RSS MB', int(open('/proc/self/statm').read().split()[1]) * 4096 // 2**20, flush=True)
log.info('%s REF reproduced (the prefix exits with code 2 otherwise)', el())
def say(*a): print(*a, flush=True)
LOST = []; COV = {}

# ---------------------------------------------------------------- free data
def get(url, tries=3):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 research'}), timeout=60).read().decode()
        except Exception as e: say('fetch retry', url, k, e)
    LOST.append(url); say('ERROR LOST', url); return None
def cboe(name):
    t = get(f'https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv')
    if t is None: return pd.Series(dtype=float, name=name)
    d = pd.read_csv(io.StringIO(t)); d['DATE'] = pd.to_datetime(d['DATE'], format='%m/%d/%Y')
    return d.set_index('DATE')[name if name == 'VVIX' else 'CLOSE'].astype(float).rename(name)
V = {n: cboe(n) for n in ('VIX', 'VIX3M', 'VIX9D', 'VVIX')}
for n, s in V.items(): say(n, len(s), s.index.min().date() if len(s) else None, s.index.max().date() if len(s) else None)
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
etf = pq.read_table(HERE / 'panel_2016_2026.parquet', columns=['symbol', 'bar_date', 'close'], filters=[('symbol', 'in', ['SPY', 'TLT', 'HYG', 'IEF'])]).to_pandas()
etf['bar_date'] = pd.to_datetime(etf.bar_date); etf = etf.drop_duplicates(['symbol', 'bar_date'], keep='last')
PX = etf.pivot(index='bar_date', columns='symbol', values='close').sort_index()
say('panel ETF closes', {c: int(PX[c].notna().sum()) for c in PX})
import yfinance as yf
yfs = yf.download('SPY', start='1993-01-01', end='2026-10-02', auto_adjust=True, progress=False)
yfs.columns = [c[0] if isinstance(c, tuple) else c for c in yfs.columns]; yfs.index = pd.to_datetime(yfs.index).tz_localize(None)
j = pd.concat([yfs['Close'].pct_change(), PX['SPY'].pct_change()], axis=1, keys=['yf', 'pn']).dropna()
j = j[j.index >= '2016-06-01']; scale_corr = j.corr().iloc[0, 1]
say(f'price-scale check yfinance SPY vs panel SPY daily-return corr {scale_corr:.5f} on {len(j)} days (need >= 0.999)')
jj = j.assign(d=(j.yf - j.pn).abs()).sort_values('d', ascending=False); say('worst SPY return mismatches', jj.head(6).round(4).to_dict('index'))
rob = jj.iloc[10:].drop(columns='d').corr().iloc[0, 1]; say(f'corr excluding 10 worst days {rob:.5f}')
SCALE_NOTE = f'corr {scale_corr:.5f}, ex-10-worst {rob:.5f}'

# U2 breadth from the panel: point-in-time eligible names (close >= 10, adv20 >= 200M), excl wrappers/tests as the sleeve does
_syms = sorted(set(_keep) - set(excl) - {'SPY'})
import ctypes; gc.collect(); ctypes.CDLL('libc.so.6').malloc_trim(0)
BR = pd.DataFrame(0.0, index=tdays, columns=['n50', 'above', 'n252', 'hi', 'lo'])
def load_rows():
    """Stream the panel row group by row group, keep only the breadth universe as compact arrays (code, day index, close, volume)."""
    pf = pq.ParquetFile(HERE / 'panel_2016_2026.parquet'); cmap = {x: i for i, x in enumerate(_syms)}; out = []
    for g_ in range(pf.metadata.num_row_groups):
        d = pf.read_row_group(g_, columns=['symbol', 'bar_date', 'close', 'volume']).to_pandas()
        d = d[d.symbol.isin(cmap)]
        if len(d): out.append((d.symbol.map(cmap).to_numpy(np.int32), tdays.get_indexer(pd.to_datetime(d.bar_date)).astype(np.int32), d.close.to_numpy(np.float32), d.volume.to_numpy(np.float32)))
    return [np.concatenate(x) for x in zip(*out)]
ROWS = load_rows(); say('breadth rows', len(ROWS[0]))
def breadth_chunk(lo_, hi_, upto=None):
    """Point-in-time eligibility (close >= 10, adv20 >= 200M); counts for P7/P8 over symbol codes [lo_, hi_); trailing windows only; rows after `upto` dropped (shift test)."""
    codes, ri, cl, vo = ROWS
    m = (codes >= lo_) & (codes < hi_) & (ri >= 0) & (ri <= (len(tdays) - 1 if upto is None else tdays.get_loc(upto)))
    Cn = np.full((len(tdays), hi_ - lo_), np.nan, np.float32); Vn = Cn.copy(); Cn[ri[m], codes[m] - lo_] = cl[m]; Vn[ri[m], codes[m] - lo_] = vo[m]
    Cc = pd.DataFrame(Cn, index=tdays); adv = (Cc * pd.DataFrame(Vn, index=tdays)).rolling(20, min_periods=20).mean()
    E_ = ((Cc >= PRICE_MIN) & (adv >= ADV_CUT)).values
    r50 = Cc.rolling(50, min_periods=50).mean().values; rmx = Cc.rolling(252, min_periods=252).max().values; rmn = Cc.rolling(252, min_periods=252).min().values
    v50 = E_ & ~np.isnan(r50); v252 = E_ & ~np.isnan(rmx)   # denominators count only names with a full trailing window
    return np.c_[v50.sum(1), (v50 & (Cn > r50)).sum(1), v252.sum(1), (v252 & (Cn >= rmx)).sum(1), (v252 & (Cn <= rmn)).sum(1)].astype(float)
BR = pd.DataFrame(0.0, index=tdays, columns=['n50', 'above', 'n252', 'hi', 'lo'])
for i in range(0, len(_syms), 500):
    BR.iloc[:, :] += breadth_chunk(i, min(i + 500, len(_syms))); say('breadth chunk', i, 'of', len(_syms))
BRD = pd.DataFrame({'P7': BR.above / BR.n50.replace(0, np.nan), 'P8': (BR.hi - BR.lo) / BR.n252.replace(0, np.nan)})
say('breadth: eligible/day median', int(BR.n252.median()), 'P7 first valid', BRD.P7.first_valid_index(), 'P8 first valid', BRD.P8.first_valid_index())

def prank(s, w=252):
    """trailing percentile rank of the last value within the trailing w-day window (incl. itself); min 126 obs."""
    return s.rolling(w, min_periods=126).apply(lambda a: (a[:-1] < a[-1]).mean() + 0.5 * (a[:-1] == a[-1]).mean() if len(a) > 1 else np.nan, raw=True)

def daily_preds(V, PX, BRD):
    """All 11 predictors, daily, each using only data up to and including that day."""
    D = pd.DataFrame(index=tdays)
    vx = V['VIX'].reindex(tdays).ffill(); v3 = V['VIX3M'].reindex(tdays).ffill(); v9 = V['VIX9D'].reindex(tdays).ffill(); vv = V['VVIX'].reindex(tdays).ffill()
    px = PX.reindex(tdays)
    D['P1'] = prank(vx); D['P2'] = vx / vx.shift(5) - 1; D['P3'] = vx / v3; D['P4'] = prank(vv); D['P5'] = v9 / vx
    D['P6'] = px.HYG.pct_change(20) - px.IEF.pct_change(20)
    D['P7'], D['P8'] = BRD.P7.reindex(tdays), BRD.P8.reindex(tdays)
    D['P9'] = px.SPY.pct_change(5); D['P10'] = px.SPY.pct_change(20) - px.TLT.pct_change(20)
    mom = px.SPY / px.SPY.rolling(125, min_periods=125).mean() - 1
    D['P11'] = pd.concat([prank(mom), prank(D.P8), prank(D.P7), prank(D.P6), 1 - D.P1, prank(D.P10)], axis=1).mean(axis=1, skipna=False)
    return D
D = daily_preds(V, PX, BRD); say('daily predictors built', D.notna().mean().round(3).to_dict())

# ---------------------------------------------------------------- weekly table (2017-01..2026-09)
rows = []
for i in range(len(rebal_dates) - 1):
    d0, d1 = rebal_dates[i], rebal_dates[i + 1]
    if (d1 - d0).days > 9: continue
    pdt = prior[d0]; k0, k1 = didx[d0] - T0, didx[d1] - T0
    r = dict(week=d0, sig_date=pdt, T1=spy_o[didx[d1]] / spy_o[didx[d0]] - 1, T2=E_REF[k1] / E_REF[k0] - 1)
    r['T3'] = r['T2'] - r['T1']; r.update(D.loc[pdt].to_dict()); rows.append(r)
W = pd.DataFrame(rows); PN = [f'P{i}' for i in range(1, 12)]
COV['weeks_total'] = len(W)
for p in PN: COV[p] = float(W[p].notna().mean()); say(f'coverage {p}: {COV[p]:.3f}  ({int(W[p].isna().sum())} weeks missing)')
say('P12 CNN Fear&Greed: not reachable free (CNN endpoint 418 bot-block) -> not run; coverage 0 %')

# ---------------------------------------------------------------- long window (1993..2026): SPY open/close from yfinance, VIX from CBOE
yd = yfs[['Open', 'Close']].dropna(); vxl = V['VIX'].reindex(yd.index).ffill()
L = pd.DataFrame(index=yd.index); L['P1'] = prank(vxl); L['P2'] = vxl / vxl.shift(5) - 1; L['P9'] = yd.Close.pct_change(5)
wk = list(pd.Series(yd.index).groupby(pd.Series(yd.index).dt.to_period('W')).min())
lrows = []
for a, b in zip(wk[:-1], wk[1:]):
    ia = yd.index.get_loc(a)
    if (b - a).days > 9 or ia == 0: continue
    lrows.append(dict(week=a, sig_date=yd.index[ia - 1], T1=yd.Open[b] / yd.Open[a] - 1, **L.iloc[ia - 1].to_dict()))
WL = pd.DataFrame(lrows); say('long window weeks', len(WL), WL.week.min().date(), WL.week.max().date())

# ---------------------------------------------------------------- shift (look-ahead) test: delete everything after a Friday, predictor must not change
rng = np.random.default_rng(7); chk = W.sig_date.iloc[rng.choice(len(W), 12, replace=False)]; bad = 0
for dt in chk:
    Vt = {n: s[s.index <= dt] for n, s in V.items()}; Pt = PX[PX.index <= dt]
    Dt = daily_preds(Vt, Pt, BRD[BRD.index <= dt])
    a = Dt.loc[dt]; b = D.loc[dt]; diff = ((a - b).abs() > 1e-9) & ~(a.isna() & b.isna())
    if diff.any(): bad += 1; say('LOOK-AHEAD at', dt.date(), list(diff[diff].index))
say(f'SHIFT TEST: {len(chk)} random Fridays, predictors recomputed on data truncated at that Friday: {bad} mismatches')
for dt in chk.iloc[:4]:  # breadth itself: rebuild chunk 0 from data truncated at the Friday, compare to the full-data row
    a = breadth_chunk(0, 500, upto=dt)[tdays.get_loc(dt)]; bfull = breadth_chunk(0, 500)[tdays.get_loc(dt)]
    if not np.allclose(a, bfull): bad += 1; say('BREADTH LOOK-AHEAD', dt.date(), a, bfull)
say(f'SHIFT TEST incl. breadth recompute on 4 Fridays: total mismatches {bad}')
SHIFT_BAD = bad

# ---------------------------------------------------------------- reads
def qread(df, p, t, half_cut):
    d = df[[p, t, 'week']].dropna()
    if len(d) < 100: return None
    d['q'] = pd.qcut(d[p].rank(method='first'), 5, labels=False)
    def spread(x):
        top, bot = x[x.q == 4][t], x[x.q == 0][t]
        se = np.sqrt(top.var(ddof=1) / len(top) + bot.var(ddof=1) / len(bot)); sp = top.mean() - bot.mean()
        return sp, sp / se, 2.8 * se
    sp, tt, mde = spread(d); g = d.groupby('q')[t]; qm = g.mean().values; qu = g.apply(lambda s: (s > 0).mean()).values
    h1 = d[d.week < half_cut]; h2 = d[d.week >= half_cut]
    s1 = spread(h1)[0] if len(h1) > 50 else np.nan; s2 = spread(h2)[0] if len(h2) > 50 else np.nan
    lo, hi = d[t].quantile([.05, .95]); dx = d[(d[t] >= lo) & (d[t] <= hi)]; sx = spread(dx)[0]
    rho = stats.spearmanr(d[p], d[t]); n = len(d); tr = rho.statistic * np.sqrt((n - 2) / (1 - rho.statistic ** 2))
    mono = stats.spearmanr(range(5), qm).statistic
    ok = abs(tt) >= 3.2 and np.sign(s1) == np.sign(sp) == np.sign(s2) and np.sign(sx) == np.sign(sp) and abs(mono) >= 0.8
    return dict(pred=p, target=t, n=n, up_share=(d[t] > 0).mean(), q1_mean=qm[0], q5_mean=qm[4], q1_up=qu[0], q5_up=qu[4], spread=sp, t=tt, mde=mde,
                spearman=rho.statistic, spearman_t=tr, h1_spread=s1, h2_spread=s2, ex5_spread=sx, mono=mono, passes=bool(ok),
                qmeans=' '.join(f'{x*100:.2f}' for x in qm))
cells = []
for t in ('T1', 'T2', 'T3'):
    for p in PN:
        r = qread(W, p, t, pd.Timestamp('2022-01-01')); r and cells.append(dict(r, window='2017-2026'))
for p in ('P1', 'P2', 'P9'):
    r = qread(WL, p, 'T1', pd.Timestamp('2010-01-01')); r and cells.append(dict(r, window='1993-2026'))
C = pd.DataFrame(cells); C.to_csv(HERE / '1700q_cells.csv', index=False)
say(f'cells {len(C)}, passes {int(C.passes.sum())}'); say(C[['window', 'pred', 'target', 'n', 'spread', 't', 'mde', 'h1_spread', 'h2_spread', 'ex5_spread', 'mono', 'passes']].round(4).to_string())
b = W.T1.gt(0).mean(); say(f'BASE RATE up weeks 2017-26: T1 {b:.3f}, T2 {W.T2.gt(0).mean():.3f}, T3 {W.T3.gt(0).mean():.3f}; long window T1 {WL.T1.gt(0).mean():.3f} n={len(WL)}')

# ---------------------------------------------------------------- walk-forward logistic model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
M = W.dropna(subset=PN + ['T1', 'T2']).copy(); M['yr'] = M.week.dt.year; say('model weeks', len(M), M.week.min().date())
MODEL = {}
for t in ('T1', 'T2'):
    out = []
    for Y in range(2020, 2027):
        tr_, te = M[M.yr < Y], M[M.yr == Y]
        if len(te) == 0 or len(tr_) < 100: continue
        sc = StandardScaler().fit(tr_[PN]); lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(tr_[PN]), (tr_[t] > 0).astype(int))
        te = te.assign(p=lr.predict_proba(sc.transform(te[PN]))[:, 1]); out.append(te)
    O = pd.concat(out); O['pred_up'] = O.p > 0.5; O['hit'] = O.pred_up == (O[t] > 0); base = (O[t] > 0).mean(); n = len(O)
    z = (O.hit.mean() - base) / np.sqrt(base * (1 - base) / n)
    up, dn = O[O.pred_up][t], O[~O.pred_up][t]
    tt = stats.ttest_ind(up, dn, equal_var=False).statistic if len(dn) > 2 else np.nan
    O['dec'] = pd.qcut(O.p.rank(method='first'), 10, labels=False)
    cal = O.groupby('dec').agg(p=('p', 'mean'), actual_up=(t, lambda s: (s > 0).mean()), ret=(t, 'mean'))
    ok = (O.hit.mean() - base >= 0.04) and z >= 2 and len(dn) > 0 and dn.mean() < 0
    MODEL[t] = dict(n=n, hit=O.hit.mean(), base=base, z=z, n_up=len(up), n_dn=len(dn), mean_up=up.mean(), mean_dn=dn.mean() if len(dn) else np.nan, t=tt, passes=bool(ok), cal=cal)
    say(t, {k: v for k, v in MODEL[t].items() if k != 'cal'}); say(cal.round(4).to_string())
pd.DataFrame({k: {a: b_ for a, b_ in v.items() if a != 'cal'} for k, v in MODEL.items()}).to_csv(HERE / '1700q_model.csv')
W.to_parquet(HERE / '1700q_series.parquet')
import json; json.dump(dict(cov=COV, lost=LOST, shift_bad=SHIFT_BAD, scale_corr=scale_corr, ref=[ref_cagr, ref_dd, float(E_REF[-1])]), open(HERE / '1700q_meta.json', 'w'))
say('DONE', el())
