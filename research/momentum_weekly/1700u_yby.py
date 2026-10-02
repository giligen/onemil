#!/usr/bin/env python3
"""Cell 1,700u: term-structure gate on the GUARDED sleeve (20 cells + SPY 2008-2016). Reuses the 1700s panel/engine code verbatim (flat script, so copied head).
Run: bash scripts/research_run.sh -m 2500M python3 research/momentum_weekly/1700s_lowvix.py > research/momentum_weekly/1700s.out"""
from __future__ import annotations
import logging, re, sys, time, itertools
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('/home/ec2-user/onemil/research/momentum_weekly')
logging.basicConfig(filename=str(OUT / '1700u.log'), filemode='w', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700u'); log.addHandler(logging.StreamHandler(sys.stdout))
PRICE_MIN, ADV_CUT, START_EQ = 10.0, 200_000_000.0, 50_000.0
WIN_START, WIN_END = pd.Timestamp('2017-01-01'), pd.Timestamp('2026-09-30')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|\bPREFERRED\b|\bRIGHTS?\b', re.I)
MAXN = 40
t0 = time.time()
def el(): return f'{time.time()-t0:5.0f}s'

# ---------------------------------------------------------------- panel + signals (as 1700p)
cols = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']
import pyarrow.parquet as pq, pyarrow as pa
tbl = pq.read_table(OUT / 'panel_2016_2026.parquet', columns=cols, read_dictionary=['symbol'])
tbl = tbl.cast(pa.schema([('symbol', tbl.schema.field('symbol').type), ('bar_date', tbl.schema.field('bar_date').type)] + [(c, pa.float32()) for c in ('open', 'high', 'low', 'close', 'volume')]))
raw = tbl.to_pandas(self_destruct=True); del tbl
raw['bar_date'] = pd.to_datetime(raw['bar_date'])
# lean load: ONE boolean mask (dedup + positive prices + ETF/test-name exclusion + SPY removal) applied to column arrays; same row set as 1700p
_c = raw.symbol.values.codes; _cats = raw.symbol.values.categories; _d = raw.bar_date.values
_sorted = bool(np.all((_c[1:] > _c[:-1]) | ((_c[1:] == _c[:-1]) & (_d[1:] >= _d[:-1]))))
log.info('raw rows %d, already sorted by (symbol codes, date): %s', len(raw), _sorted)
if not _sorted: log.error('raw not sorted -- STOP'); raise SystemExit(4)
spy_df = raw.loc[raw.symbol == 'SPY', ['bar_date', 'open']].sort_values('bar_date').drop_duplicates('bar_date', keep='last').reset_index(drop=True)
if spy_df.empty: log.error('SPY missing'); raise SystemExit(1)
tdays = pd.DatetimeIndex(sorted(spy_df.bar_date.unique())); didx = {d: i for i, d in enumerate(tdays)}
assets = pd.read_csv(OUT / '1700c_assets.csv', dtype={'symbol': str, 'name': str}); assets['name'] = assets['name'].fillna('')
excl = set(assets.loc[assets.name.str.contains(NAME_RE), 'symbol']) | {x for x in _cats if TEST_RE.match(x)}
bad_cat = np.array([(x in excl) or x == 'SPY' for x in _cats])
dup_next = np.append((_c[1:] == _c[:-1]) & (_d[1:] == _d[:-1]), False)
o_, h_, l_, c_, v_ = (raw[k].values for k in ('open', 'high', 'low', 'close', 'volume'))
keep = ~dup_next & ~((o_ <= 0) | (h_ <= 0) | (l_ <= 0) | (c_ <= 0)) & ~bad_cat[_c]
panel = pd.DataFrame({'symbol': pd.Categorical.from_codes(_c[keep], categories=_cats), 'bar_date': _d[keep], 'open': o_[keep], 'close': c_[keep]})
panel['symbol'] = panel.symbol.cat.remove_unused_categories()
spread_ = (((h_[keep] - l_[keep]) / c_[keep]).clip(min=0) * 0.1).clip(max=0.002).astype(np.float32)
dv = (c_[keep] * v_[keep]).astype(np.float64); cl = c_[keep]
del raw, _c, _d, o_, h_, l_, c_, v_, keep, dup_next
import gc; gc.collect()
panel['spread'] = spread_; del spread_
codes = panel.symbol.values.codes
dts = panel.bar_date.values; gtype = np.zeros(len(panel), np.int8)
N = len(panel); adv20 = np.full(N, np.nan); sigV2 = np.full(N, np.nan); okv = np.zeros(N, bool)
bounds = np.flatnonzero(np.diff(codes)) + 1; starts = np.r_[0, bounds]; ends = np.r_[bounds, N]
for a, b in zip(starts, ends):
    c = pd.Series(cl[a:b]); adv20[a:b] = pd.Series(dv[a:b]).rolling(20, min_periods=20).mean().values
    c21, c252, c273 = c.shift(21), c.shift(252), c.shift(273)
    ret = c.pct_change(); vol = ret.rolling(252, min_periods=252).std().values
    sig = (c21 / c252 - 1).values
    with np.errstate(divide='ignore', invalid='ignore'): sigV2[a:b] = np.where(vol > 0, sig / vol, np.nan)
    okv[a:b] = c273.notna().values
    cc = cl[a:b].astype(np.float64); mv = np.zeros(b - a); mv[1:] = cc[1:] / cc[:-1] - 1; gp = np.zeros(b - a, bool); gp[1:] = np.diff(dts[a:b]).astype('timedelta64[D]').astype(np.int64) > 10
    ea = pd.Series(((mv > 2.0) | (mv < -0.75)).astype(np.float64)).rolling(273, min_periods=1).max().values > 0
    eb = pd.Series(gp.astype(np.float64)).rolling(273, min_periods=1).max().values > 0
    gtype[a:b] = ea.astype(np.int8) + 2 * eb.astype(np.int8)
panel['adv20'] = adv20; panel['sigV2'] = sigV2; panel['ok'] = okv; panel['gtype'] = gtype; del dv, cl, adv20, sigV2, okv
log.info('%s signals built (%d rows)', el(), len(panel))

per = pd.Series(tdays).dt.to_period('W'); first = pd.Series(tdays).groupby(per).min().sort_index()
ent = first[(first >= WIN_START) & (first <= WIN_END)].reset_index(drop=True)
prior = {d: tdays[didx[d] - 1] for d in ent if didx[d] > 0}
wk_days = pd.Series(tdays).groupby(per).apply(list)
DAYMAP = {w: {} for w in range(5)}
for d in ent:
    days = [x for x in wk_days[d.to_period('W')] if x >= d]
    for w in range(5):
        c = [x for x in days if x.weekday() >= w]
        if c: DAYMAP[w][c[0]] = d
        elif w >= 3 and len(days) >= 2: DAYMAP[w][days[-1]] = d
extra = {x for dct in DAYMAP.values() for x in dct if didx[x] > 0}
prior.update({x: tdays[didx[x] - 1] for x in extra})
sigdates = set(prior.values())
# U2 eligibility (price, history, ADV, sigV2 defined) -- ALL eligible rows kept (REF uses the top 40 by sigV2; the residual cells rank everything eligible)
sr = panel.loc[panel.bar_date.isin(sigdates) & (panel.close >= PRICE_MIN) & panel.ok & (panel.adv20 >= ADV_CUT) & panel.sigV2.notna(), ['symbol', 'bar_date', 'sigV2', 'gtype']]
sr['symbol'] = sr['symbol'].astype(str)
ranked_syms = {d: sub.nlargest(MAXN, 'sigV2')['symbol'].tolist() for d, sub in sr.groupby('bar_date', observed=True)}
ranked_g = {d: sub[sub.gtype == 0].nlargest(MAXN, 'sigV2')['symbol'].tolist() for d, sub in sr.groupby('bar_date', observed=True)}
GUARD_SET = {d: {s: {1: 'move', 2: 'gap', 3: 'move+gap'}[g] for s, g in zip(sub.symbol, sub.gtype) if g > 0} for d, sub in sr.groupby('bar_date', observed=True)}
log.info('guard flags: %d of %d eligible signal rows', int((sr.gtype > 0).sum()), len(sr))
cand = sorted({s for v in ranked_syms.values() for s in v} | {s for v in ranked_g.values() for s in v}); cidx = {s: i for i, s in enumerate(cand)}
sub = panel[panel.symbol.isin(cand)]
def piv(col): return sub.pivot_table(index='bar_date', columns='symbol', values=col, aggfunc='last', observed=True).reindex(index=tdays, columns=cand)
O_raw, Cdf, SP = piv('open'), piv('close'), piv('spread')
O = O_raw.ffill().fillna(0.0).values.astype(np.float64)
C = Cdf.values.astype(np.float64)
RATE = np.minimum(0.0005 + 0.5 * np.nan_to_num(SP.values.astype(np.float64), nan=0.002), 0.002)
spy_o = spy_df.set_index('bar_date')['open'].reindex(tdays).ffill().values.astype(np.float64)
del panel, sub, sr, O_raw, SP, Cdf
log.info('%s candidates %d (eligible union), pivots %s', el(), len(cand), O.shape)
rebal_dates = [d for d in ent if d in prior and prior[d] in ranked_syms]
rebal = {didx[d]: [cidx[s] for s in ranked_syms[prior[d]]] for d in rebal_dates}
T0, T1 = didx[rebal_dates[0]], didx[rebal_dates[-1]]
log.info('%s rebalances %d, %s..%s', el(), len(rebal), rebal_dates[0].date(), rebal_dates[-1].date())

# ================================================================ 1700s / 1700t tail (gate + guard on the 1700r/1700p daily engine)
import io, urllib.request, warnings
from scipy import stats
warnings.filterwarnings('ignore')
def say(*a): print(*a, flush=True)
GATED_SCALE_NOTE = 'gated week: book sold at Monday open (engine cost), cash 0, re-bought at next ungated Monday open; half = 1/(2N) per name'

def simulate(rank_map, n=20, scale=None):
    """One cell on the daily engine (1700r/1700p, Monday-open weekly 1/N reset). scale[ti] in {1, 0.5, 0} multiplies the target book on that
    reset date (0 = all cash, 0.5 = every name at 1/(2N)); trades and costs are charged by the engine on the traded notional exactly as REF."""
    ns = len(cand); sh = np.zeros(ns); cash = START_EQ; nd = T1 - T0 + 1; E = np.zeros(nd); cost_f = trade_f = 0.0
    rk = {didx[x]: [cidx[s] for s in rank_map[prior[x]]] for x in DAYMAP[0] if prior[x] in rank_map and didx[x] <= T1}
    for k in range(nd):
        ti = T0 + k; o = O[ti]; pre = cash + sh @ o; tr = cs = 0.0
        if ti in rk:
            eq = cash + sh @ o; s = 1.0 if scale is None else scale.get(ti, 1.0)
            top = [i for i in rk[ti][:n] if o[i] > 0]
            tgt = np.zeros(ns)
            if s > 0: tgt[top] = eq * s / n
            delta = tgt - sh * o
            for i in np.where(delta < -1e-9)[0]:
                v = -delta[i]; c = RATE[ti, i] * v; cash += v - c; sh[i] -= v / o[i]; tr += v; cs += c
                if sh[i] <= 1e-12: sh[i] = 0
            for i in np.where(delta > 1e-9)[0]:
                v = delta[i]; c = RATE[ti, i] * v; cash -= v + c; sh[i] += v / o[i]; tr += v; cs += c
        E[k] = cash + sh @ o
        cost_f += cs / pre if pre > 0 else 0; trade_f += tr / pre if pre > 0 else 0
    return dict(E=E, cost_f=cost_f, trade_f=trade_f)

def dd_series(E): return E / np.maximum.accumulate(E) - 1
dates = tdays[T0:T1 + 1]; yrs = (dates[-1] - dates[0]).days / 365.25
spyE = spy_o[T0:T1 + 1] / spy_o[T0] * START_EQ
def yearly(E):
    s = pd.Series(pd.Series(E).pct_change().fillna(0).values, index=dates); return (1 + s).groupby(s.index.year).prod() - 1
spy_y = yearly(spyE)
H2 = pd.Timestamp('2022-01-01'); m1 = np.asarray(dates < H2); m2 = ~m1
def stats_of(E, m=None):
    """CAGR, max DD, ratio on the whole path or a boolean-mask half (rebased)."""
    x = E if m is None else E[m]; d = dates if m is None else dates[m]; y = (d[-1] - d[0]).days / 365.25
    c = (x[-1] / x[0]) ** (1 / y) - 1; dd = dd_series(x).min(); return c, dd, c / abs(dd)
widx = np.array([didx[d] - T0 for d in rebal_dates]); wd = pd.DatetimeIndex(rebal_dates)
def weekly(E): return pd.Series(E[widx]).pct_change().dropna().values
wdates = wd[1:]                                             # wk[j] = return of the week starting at rebal_dates[j]  (index j of wd[:-1])
wk_start = wd[:-1]

# ---------------------------------------------------------------- GREF repro gate (guarded reference)
t_g = simulate(ranked_g); E_REF = t_g['E']; c0, d0, r0 = stats_of(E_REF)
say(f'GREF CAGR {c0:.4%} maxDD {d0:.3%} end ${E_REF[-1]:,.0f}'); log.info('GREF CAGR %.4f maxDD %.4f end %.0f', c0, d0, E_REF[-1])
if abs(c0 - 0.2934) > 0.0005 or abs(d0 + 0.383) > 0.0006 or abs(E_REF[-1] - 596394) > 0.002 * 596394:
    log.error('GREF DOES NOT REPRODUCE -- STOP'); (OUT / 'RESULT_1700u.md').write_text(f'GREF DOES NOT REPRODUCE: CAGR {c0:.4f} DD {d0:.4f} end {E_REF[-1]:.0f}\n'); raise SystemExit(2)
ref_h1, ref_h2 = stats_of(E_REF, m1), stats_of(E_REF, m2); wk_ref = weekly(E_REF)
# ---------------------------------------------------------------- gauges (CBOE daily history; percentiles recomputed here)
def get(url, tries=3):
    for k in range(tries):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 research'}), timeout=60).read().decode()
        except Exception as e: log.warning('fetch retry %s %d %s', url, k, e)
    log.error('LOST %s', url); raise SystemExit(5)
def cboe(name):
    d = pd.read_csv(io.StringIO(get(f'https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv'))); d['DATE'] = pd.to_datetime(d['DATE'], format='%m/%d/%Y')
    return d.set_index('DATE')['CLOSE'].astype(float).rename(name)
VIX, VIX3 = cboe('VIX'), cboe('VIX3M')
say('VIX', len(VIX), VIX.index.min().date(), VIX.index.max().date(), 'VIX3M', len(VIX3), VIX3.index.min().date())
def prank(s, w):
    """percentile of the last value inside the trailing w observations ending that day (strictly-less + half ties, as 1700q), min 126 obs; nothing later is used."""
    return s.rolling(w, min_periods=126).apply(lambda a: (a[:-1] < a[-1]).mean() + 0.5 * (a[:-1] == a[-1]).mean() if len(a) > 1 else np.nan, raw=True)
def gauge(vix, vix3, which): return vix if which == 'VIX' else (vix / vix3.reindex(vix.index)).dropna()
PCT = {}
for gname in ('VIX', 'VIXratio'):
    s = gauge(VIX, VIX3, gname)
    for w in (252, 504): PCT[(gname, w)] = prank(s, w)
PCTd = {k: v.reindex(tdays).ffill() for k, v in PCT.items()}
# shift test: truncate the data after a Friday, recompute, the percentile (hence next week's gate for every threshold) must be unchanged
rng = np.random.default_rng(11); sigs = [prior[d] for d in rebal_dates]; chk = [sigs[i] for i in rng.choice(len(sigs), 10, replace=False)]; bad = 0
for dt in chk:
    for gname in ('VIX', 'VIXratio'):
        s = gauge(VIX[VIX.index <= dt], VIX3[VIX3.index <= dt], gname)
        for w in (252, 504):
            a = prank(s.iloc[-(w + 5):], w).iloc[-1]; b = PCT[(gname, w)].loc[dt]
            if not ((np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12): bad += 1
            for thr in (10, 15, 20, 25, 30):
                if ((a < thr / 100) if not np.isnan(a) else False) != ((b < thr / 100) if not np.isnan(b) else False): bad += 1
say(f'SHIFT TEST: {len(chk)} Fridays x 4 gauges x 5 thresholds, data truncated at the Friday: {bad} mismatches')


# ---------------------------------------------------------------- year-by-year / 2026 month-by-month dump (read-only)
def gate_curve(w, thr, act='cash'):
    """Equity curve of the guarded sleeve with the VIX-ratio gate (window w, threshold thr %)."""
    pv = np.array([PCTd[('VIXratio', w)].loc[prior[d]] for d in rebal_dates])
    gated = np.nan_to_num(pv, nan=1.0) < thr / 100
    sc = {didx[d]: (0.0 if act == 'cash' else 0.5) for d, g in zip(rebal_dates, gated) if g}
    return simulate(ranked_g, scale=sc)['E'], gated
curves = {'SPY': spyE, 'plain': simulate(ranked_syms)['E'], 'guard': E_REF}
G = {}
for nm, (w, thr) in {'gate_p30': (252, 30), 'gate_p10': (252, 10), 'gate_w504p30': (504, 30)}.items():
    curves[nm], G[nm] = gate_curve(w, thr)
D = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in curves.items()}, index=pd.DatetimeIndex(dates))
D.to_csv(OUT / '1700u_curves_daily.csv')
ye = D.groupby(D.index.year).last(); yb = pd.concat([D.iloc[[0]], ye.iloc[:-1]]).values
Y = pd.DataFrame(ye.values / yb - 1, index=ye.index, columns=D.columns)
say('YEARLY %'); say((Y * 100).round(1).to_string())
say('END $'); say(D.iloc[-1].round(0).to_string())
say('MAXDD %'); say(((D / D.cummax() - 1).min() * 100).round(1).to_string())
m = D[D.index >= '2025-12-01']; me = m.groupby([m.index.year, m.index.month]).last(); M = me.pct_change().dropna()
say('2026 MONTHLY %'); say((M * 100).round(1).to_string())
gw = pd.Series(G['gate_p30'], index=pd.DatetimeIndex(rebal_dates)); g26 = gw[gw.index >= '2026-01-01']
say('gate_p30 weeks in cash per 2026 month:'); say(g26.groupby(g26.index.month).agg(['sum', 'count']).to_string())
say('gate_p30 last 6 weeks:', [(str(d.date()), bool(v)) for d, v in gw.iloc[-6:].items()])
