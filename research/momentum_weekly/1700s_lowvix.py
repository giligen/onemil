#!/usr/bin/env python3
"""Cells 1,700s (low-VIX gate, 40 cells) AND 1,700t (data-hygiene guard, REF vs REF+guard) in ONE process / one panel load (the 1700t guard is a mode of this script).
Run: bash scripts/research_run.sh -m 2500M python3 research/momentum_weekly/1700s_lowvix.py > research/momentum_weekly/1700s.out"""
from __future__ import annotations
import logging, re, sys, time, itertools
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('/home/ec2-user/onemil/research/momentum_weekly')
logging.basicConfig(filename=str(OUT / '1700s.log'), filemode='w', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700s'); log.addHandler(logging.StreamHandler(sys.stdout))
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

# ---------------------------------------------------------------- REF + repro gate
t_ref = simulate(ranked_syms); E_REF = t_ref['E']
c0, d0, r0 = stats_of(E_REF)
say(f'REF CAGR {c0:.4%} maxDD {d0:.3%} end ${E_REF[-1]:,.0f}'); log.info('REF CAGR %.4f maxDD %.4f end %.0f', c0, d0, E_REF[-1])
if abs(c0 - 0.2718) > 0.0005 or abs(d0 + 0.4450) > 0.0006 or abs(E_REF[-1] - 507823) > 0.002 * 507823:
    log.error('REF DOES NOT REPRODUCE -- STOP'); (OUT / 'RESULT_1700s.md').write_text(f'REF DOES NOT REPRODUCE: CAGR {c0:.4f} DD {d0:.4f} end {E_REF[-1]:.0f}\n'); raise SystemExit(2)
ref_h1, ref_h2 = stats_of(E_REF, m1), stats_of(E_REF, m2)
wk_ref = weekly(E_REF)

# ---------------------------------------------------------------- 1700t: guard (REF vs REF + guard), same process, same panel
t_g = simulate(ranked_g); E_G = t_g['E']; cg, dg, rg = stats_of(E_G)
say(f'REF+guard CAGR {cg:.4%} maxDD {dg:.3%} end ${E_G[-1]:,.0f}')
rem = []
for d in rebal_dates:
    pdt = prior[d]; g = set(ranked_g.get(pdt, []))
    for r_, s in enumerate(ranked_syms[pdt][:MAXN]):
        if s in GUARD_SET.get(pdt, ()): rem.append(dict(rebal_date=d.date(), signal_date=pdt.date(), symbol=s, ref_rank=r_ + 1, held_in_ref=r_ < 20, guard_type=GUARD_SET[pdt][s]))
rem = pd.DataFrame(rem); rem.to_csv(OUT / '1700t_removed.csv', index=False)
nw_held = int(rem.held_in_ref.sum()) if len(rem) else 0
yg, yr_ = yearly(E_G), yearly(E_REF)
say(f'guard: name-weeks removed from REF top-20 {nw_held}, from top-40 {len(rem)}')
rem_names = rem[rem.held_in_ref].groupby('symbol').size().sort_values(ascending=False) if len(rem) else pd.Series(dtype=int)

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

# ---------------------------------------------------------------- the 40 cells
def tstat(x): return float(np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and np.std(x) > 0 else np.nan
runmax = np.maximum.accumulate(E_REF); eps = []; i = 0
while i < len(E_REF):
    if E_REF[i] < runmax[i]:
        pk = int(np.where(E_REF[:i] == runmax[i])[0][-1]) if i > 0 else 0; j = i
        while j < len(E_REF) and E_REF[j] < runmax[i]: j += 1
        tr_i = pk + int(np.argmin(E_REF[pk:j])); eps.append((pk, tr_i, j if j < len(E_REF) else None, E_REF[tr_i] / E_REF[pk] - 1)); i = j
    else: i += 1
eps = sorted(eps, key=lambda x: x[3])[:5]
def ep_depths(Ec): return [(Ec[pk:(rc if rc else len(Ec))] / np.maximum.accumulate(Ec[pk:(rc if rc else len(Ec))]) - 1).min() for pk, tr, rc, _ in eps]
rows, cellE, cellG = [], {}, {}
for gname in ('VIX', 'VIXratio'):
    for w in (252, 504):
        pv = np.array([PCTd[(gname, w)].loc[prior[d]] for d in rebal_dates])
        for thr in (10, 15, 20, 25, 30):
            gated = np.nan_to_num(pv, nan=1.0) < thr / 100                      # NaN percentile -> not gated
            for act in ('cash', 'half'):
                sc = {didx[d]: (0.0 if act == 'cash' else 0.5) for d, g in zip(rebal_dates, gated) if g}
                name = f'{gname}|w{w}|p{thr}|{act}'; r = simulate(ranked_syms, scale=sc); Ec = r['E']; cellE[name] = Ec
                c, dd, ra = stats_of(Ec); c1, d1, ra1 = stats_of(Ec, m1); c2, d2, ra2 = stats_of(Ec, m2)
                gw = gated[:-1]; ref_g = wk_ref[gw]; h1m = gw & np.asarray(wk_start < H2); h2m = gw & np.asarray(wk_start >= H2)
                m_all = float(ref_g.mean()) if len(ref_g) else np.nan; m1_ = float(wk_ref[h1m].mean()) if h1m.any() else np.nan; m2_ = float(wk_ref[h2m].mean()) if h2m.any() else np.nan
                ex5 = float(np.sort(ref_g)[int(np.ceil(0.05 * len(ref_g))):].mean()) if len(ref_g) > 20 else np.nan
                spells = int(((gated[1:] & ~gated[:-1]).sum()) + gated[0]); wk = weekly(Ec); dif = wk - wk_ref; kp = dif <= np.quantile(dif, 0.95)
                gy = pd.Series(gated, index=wd).groupby(wd.year).agg(['sum', 'mean']); tops = gy['sum'] / max(gy['sum'].sum(), 1); y = yearly(Ec)
                row = dict(cell=name, gauge=gname, window=w, thr=thr, action=act, cagr=c, max_dd=dd, ratio=ra, end_usd=Ec[-1], ratio_h1=ra1, ratio_h2=ra2, ref_ratio_h1=ref_h1[2], ref_ratio_h2=ref_h2[2],
                           gated_share=float(gated.mean()), n_gated=int(gated.sum()), spells=spells, gmean=m_all, gmean_h1=m1_, gmean_h2=m2_, gmean_ex5=ex5,
                           paired_mean=float(dif.mean()), paired_t=tstat(dif), paired_ex_top5=float(dif[kp].mean()), years_beat_spy=int((y > spy_y).sum()), n_years=len(y), worst_year=float(y.min()),
                           top_year=int(tops.idxmax()), top_year_share=float(tops.max()))
                for k_, dp in enumerate(ep_depths(Ec), 1): row[f'ep{k_}_depth'] = dp
                row['improve'] = bool(c >= c0 and dd - d0 >= 0.03); row['ratio_both'] = bool(ra1 > ref_h1[2] and ra2 > ref_h2[2])
                row['neg_both'] = bool(m1_ < 0 and m2_ < 0 and ex5 <= 0) if not (np.isnan(m1_) or np.isnan(m2_) or np.isnan(ex5)) else False
                rows.append(row); cellG[name] = tops
                say(f'{name}: CAGR {c:.2%} DD {dd:.1%} ratio {ra:.2f} gated {gated.mean():.0%} improve={row["improve"]} ratioBoth={row["ratio_both"]} negBoth={row["neg_both"]}')
cdf = pd.DataFrame(rows); cdf.to_csv(OUT / '1700s_cells.csv', index=False)
n_imp, n_rb, n_nb = int(cdf.improve.sum()), int(cdf.ratio_both.sum()), int(cdf.neg_both.sum())
order = cdf.sort_values('ratio', ascending=False).reset_index(drop=True); med = order.iloc[20]        # 21st of 40 by ratio (upper-middle)
med_gy = pd.Series(cellG[med.cell]); gated_med = np.nan_to_num(np.array([PCTd[(med.gauge, int(med.window))].loc[prior[d]] for d in rebal_dates]), nan=1.0) < med.thr / 100
frac_y = pd.Series(gated_med, index=wd).groupby(wd.year).mean()
yr_ok = bool(med.top_year_share <= 0.40)
verdict = 'REAL' if (n_imp >= 30 and n_rb >= 30 and n_nb >= 30 and yr_ok) else ('partial' if n_imp >= 15 else 'FAIL')
say(f'FAMILY: improve {n_imp}/40, ratio both halves {n_rb}/40, neg gated mean both halves + ex5 {n_nb}/40, median cell {med.cell}, top-year share {med.top_year_share:.0%}, VERDICT {verdict}')

# ---------------------------------------------------------------- SPY 1993-2026 buy-and-hold, VIX-level gate (mechanism read, not a pass condition)
spyrows = []
try:
    import yfinance as yf
    yfs = yf.download('SPY', start='1993-01-01', end='2026-10-02', auto_adjust=True, progress=False)
    yfs.columns = [c[0] if isinstance(c, tuple) else c for c in yfs.columns]; yfs.index = pd.to_datetime(yfs.index).tz_localize(None); yd = yfs[['Open']].dropna()
    wkf = list(pd.Series(yd.index).groupby(pd.Series(yd.index).dt.to_period('W')).min()); lr = []
    for a_, b_ in zip(wkf[:-1], wkf[1:]):
        ia = yd.index.get_loc(a_)
        if (b_ - a_).days > 9 or ia == 0: continue
        lr.append(dict(week=a_, sig=yd.index[ia - 1], r=yd.Open[b_] / yd.Open[a_] - 1))
    WL = pd.DataFrame(lr)
    for w in (252, 504):
        pv = PCT[('VIX', w)].reindex(yd.index).ffill().reindex(WL.sig).values
        for thr in (10, 15, 20, 25, 30):
            g = np.nan_to_num(pv, nan=1.0) < thr / 100; ok = ~np.isnan(pv); a_, b_ = WL.r[g & ok], WL.r[~g & ok]; h = WL.week < pd.Timestamp('2010-01-01')
            spyrows.append(dict(window=w, thr=thr, n_gated=int(g.sum()), gated_mean=a_.mean(), ungated_mean=b_.mean(), diff_t=float(stats.ttest_ind(a_, b_, equal_var=False).statistic),
                                gated_h1=WL.r[g & ok & h].mean(), gated_h2=WL.r[g & ok & ~h].mean(), ungated_h1=WL.r[~g & ok & h].mean(), ungated_h2=WL.r[~g & ok & ~h].mean()))
    pd.DataFrame(spyrows).to_csv(OUT / '1700s_spy_long.csv', index=False)
except Exception as e: log.error('SPY long-window mechanism check FAILED: %s', e)
SP = pd.DataFrame(spyrows)

# ---------------------------------------------------------------- reports
L = ['# RESULT 1,700s -- low-VIX gate on the momentum sleeve, 40-cell family (PREREG_1700s.md, FROZEN)', '',
     f'REF reproduced BEFORE any cell was read: CAGR {c0:.2%} / max DD {d0:.1%} / end ${E_REF[-1]:,.0f} (target 27.18 / -44.5 / 507,823). 2017-01..2026-09, $50K, 1700j/1700r daily engine. {GATED_SCALE_NOTE}.',
     f'Gauges recomputed from CBOE daily VIX / VIX3M (percentile in trailing window ending the Friday). Shift test: {len(chk)} Fridays x 4 gauges x 5 thresholds, data deleted after the Friday: {bad} mismatches.',
     f'Halves 2017-21 / 2022-26 ratio (CAGR/|DD|) REF {ref_h1[2]:.2f} / {ref_h2[2]:.2f}. Columns: gate cell | CAGR | maxDD | ratio | end $ | gated % | spells | REF mean wk ret in gated weeks all/H1/H2/ex-worst5% (%) | impr | ratio both halves',
     '| cell | CAGR | DD | ratio | end $ | gated | sp | gated-wk mean all/H1/H2/ex5 | I | R |', '|---|---|---|---|---|---|---|---|---|---|']
for _, x in cdf.iterrows():
    L.append(f"| {x.cell} | {x.cagr:.1%} | {x.max_dd:.1%} | {x.ratio:.2f} | {x.end_usd/1e3:,.0f}K | {x.gated_share:.0%} | {x.spells} | {x.gmean*100:+.2f}/{x.gmean_h1*100:+.2f}/{x.gmean_h2*100:+.2f}/{x.gmean_ex5*100:+.2f} | {'Y' if x.improve else '.'} | {'Y' if x.ratio_both else '.'} |")
L += ['', f'FAMILY (need 30 of 40 each): improve (CAGR >= REF AND DD better by >= 3 pts) {n_imp}; ratio beats REF in BOTH halves {n_rb}; gated-week mean < 0 in both halves AND <= 0 ex worst 5% {n_nb}; median cell top-year share {med.top_year_share:.0%} (<= 40% needed: {"ok" if yr_ok else "FAIL"}).',
      f'VERDICT: {verdict}. Median cell (21st of 40 by ratio): {med.cell} CAGR {med.cagr:.2%} / DD {med.max_dd:.1%} / end ${med.end_usd:,.0f} / gated {med.gated_share:.0%} of weeks in {med.spells} spells; ratio {med.ratio:.2f} vs REF {r0:.2f}; paired weekly diff {med.paired_mean*100:+.3f}% (t {med.paired_t:.1f}, ex-top5% {med.paired_ex_top5*100:+.3f}%); yrs>SPY {med.years_beat_spy}/{med.n_years}, worst yr {med.worst_year:.1%}.',
      'Median cell by-year (gated share of that year | share of all gated weeks): ' + ' '.join(f"{y_}: {frac_y[y_]:.0%}|{med_gy.get(y_, 0):.0%}" for y_ in frac_y.index),
      'SPY buy-and-hold 1993-2026, VIX-level gate (window/thr: gated-week mean vs ungated %, t; H1<2010 / H2 gated mean):']
for _, x in SP.iterrows(): L.append(f"  w{int(x.window)} p{int(x.thr)}: n {int(x.n_gated)} gated {x.gated_mean*100:+.2f} vs {x.ungated_mean*100:+.2f} t {x.diff_t:+.1f}; H1/H2 gated {x.gated_h1*100:+.2f}/{x.gated_h2*100:+.2f}")
L += ['Adversary caveats: (1) the lead was read off 1,700q after seeing it (36 tests); the 40 cells are neighbours of one lead, not independent; family count on this line +40. (2) Costs use the engine band model, not NBBO; a gated week sells and re-buys the whole book (turnover cost is in).',
      '(3) 504-day windows start 2017 with partial history only where VIX history allows (CBOE since 1990, so full). (4) Gated-week means are REF weekly returns (counterfactual), whole-sample selection; ex-worst-5% guards the tail only. (5) Two half-splits only; a gate with ~10-30% weeks is a handful of spells, so one or two episodes can decide a cell.']
(OUT / 'RESULT_1700s.md').write_text('\n'.join(L) + '\n')
nm = ', '.join(f'{k} x{v}' for k, v in rem_names.head(15).items())
T = ['# RESULT 1,700t -- data-hygiene guard on the sleeve signal (PREREG_1700t.md, FROZEN)', '',
     f'Guard (ineligible if inside the 273-bar lookback: a one-day close move > +200% or < -75%, or a gap > 10 calendar days between consecutive bars), applied in the eligibility step BEFORE ranking. Same process/panel as 1700s; REF reproduced first.',
     f'REF        CAGR {c0:.2%} / max DD {d0:.1%} / end ${E_REF[-1]:,.0f}', f'REF+guard  CAGR {cg:.2%} / max DD {dg:.1%} / end ${E_G[-1]:,.0f}',
     f'Move: CAGR {100*(cg-c0):+.2f} pt, max DD {100*(dg-d0):+.2f} pt.  Rule: < 1.0 pt CAGR and < 2.0 pt DD -> adopt + re-baseline; larger -> report names, adopt anyway, restate headlines.',
     f'Name-weeks removed: {nw_held} of REF top-20 holdings ({len(rem)} of top-40 ranks); by type: ' + (str(rem.guard_type.value_counts().to_dict()) if len(rem) else 'none'),
     f'Names removed from holdings (count of weeks): {nm}', f'Full list (date, symbol, rank, type): 1700t_removed.csv.',
     'By year REF -> guard: ' + ' '.join(f'{y_}: {yr_[y_]:.1%}->{yg[y_]:.1%}' for y_ in yr_.index),
     f"WOLF held in REF? {'yes' if len(rem) and ((rem.symbol == 'WOLF') & rem.held_in_ref).any() else 'no (not in removed holdings)'}.",
     'Caveat: (b) fires on any >10-day listing gap incl. real halts; (a) removes real moves (AMC, ABVX-type) with the fake ones -- hygiene, not an edge claim; one cell.']
(OUT / 'RESULT_1700t.md').write_text('\n'.join(T) + '\n'); log.info('DONE %s', el()); say('DONE')
