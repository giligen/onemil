#!/usr/bin/env python3
"""Cell 1,700r: residual (beta-adjusted) momentum (PREREG_1700r.md, FROZEN). Panel load + 1700p daily engine copied; ONLY the ranking score changes.
Run: bash scripts/research_run.sh -m 2500M python3 research/momentum_weekly/1700r_residual.py > research/momentum_weekly/1700r.out"""
from __future__ import annotations
import logging, re, sys, time, itertools
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('/home/ec2-user/onemil/research/momentum_weekly')
logging.basicConfig(filename=str(OUT / '1700r.log'), filemode='w', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700r'); log.addHandler(logging.StreamHandler(sys.stdout))
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
# factor closes (SPY, QQQ, IWM): small filtered parquet read
FACT = {}
_f = pq.read_table(OUT / 'panel_2016_2026.parquet', columns=['symbol', 'bar_date', 'close'], filters=[('symbol', 'in', ['SPY', 'QQQ', 'IWM'])]).to_pandas()
_f['bar_date'] = pd.to_datetime(_f['bar_date']); _f['symbol'] = _f['symbol'].astype(str)
for f in ('SPY', 'QQQ', 'IWM'):
    s_ = _f.loc[_f.symbol == f].drop_duplicates('bar_date').set_index('bar_date')['close'].reindex(tdays)
    if s_.isna().any(): log.error('factor %s has %d missing closes', f, int(s_.isna().sum()))
    FACT[f] = s_.values.astype(np.float64)
del _f
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
N = len(panel); adv20 = np.full(N, np.nan); sigV2 = np.full(N, np.nan); okv = np.zeros(N, bool)
bounds = np.flatnonzero(np.diff(codes)) + 1; starts = np.r_[0, bounds]; ends = np.r_[bounds, N]
for a, b in zip(starts, ends):
    c = pd.Series(cl[a:b]); adv20[a:b] = pd.Series(dv[a:b]).rolling(20, min_periods=20).mean().values
    c21, c252, c273 = c.shift(21), c.shift(252), c.shift(273)
    ret = c.pct_change(); vol = ret.rolling(252, min_periods=252).std().values
    sig = (c21 / c252 - 1).values
    with np.errstate(divide='ignore', invalid='ignore'): sigV2[a:b] = np.where(vol > 0, sig / vol, np.nan)
    okv[a:b] = c273.notna().values
panel['adv20'] = adv20; panel['sigV2'] = sigV2; panel['ok'] = okv; del dv, cl, adv20, sigV2, okv
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
sr = panel.loc[panel.bar_date.isin(sigdates) & (panel.close >= PRICE_MIN) & panel.ok & (panel.adv20 >= ADV_CUT) & panel.sigV2.notna(), ['symbol', 'bar_date', 'sigV2']]
sr['symbol'] = sr['symbol'].astype(str)
ranked_syms = {d: sub.nlargest(MAXN, 'sigV2')['symbol'].tolist() for d, sub in sr.groupby('bar_date', observed=True)}
elig_syms = {d: sub['symbol'].tolist() for d, sub in sr.groupby('bar_date', observed=True)}
cand = sorted({s for v in elig_syms.values() for s in v} | {'NVDA', 'KO'}); cidx = {s: i for i, s in enumerate(cand)}
sub = panel[panel.symbol.isin(cand)]
def piv(col): return sub.pivot_table(index='bar_date', columns='symbol', values=col, aggfunc='last', observed=True).reindex(index=tdays, columns=cand)
O_raw, Cdf, SP = piv('open'), piv('close'), piv('spread')
O = O_raw.ffill().fillna(0.0).values.astype(np.float64)
C = Cdf.values.astype(np.float64)
RATE = np.minimum(0.0005 + 0.5 * np.nan_to_num(SP.values.astype(np.float64), nan=0.002), 0.002)
spy_o = spy_df.set_index('bar_date')['open'].reindex(tdays).ffill().values.astype(np.float64)
del panel, sub, sr, O_raw, SP, Cdf
elig_idx = {d: np.array([cidx[s] for s in v]) for d, v in elig_syms.items()}
log.info('%s candidates %d (eligible union), pivots %s', el(), len(cand), O.shape)
rebal_dates = [d for d in ent if d in prior and prior[d] in ranked_syms]
rebal = {didx[d]: [cidx[s] for s in ranked_syms[prior[d]]] for d in rebal_dates}
T0, T1 = didx[rebal_dates[0]], didx[rebal_dates[-1]]
log.info('%s rebalances %d, %s..%s', el(), len(rebal), rebal_dates[0].date(), rebal_dates[-1].date())

# ---------------------------------------------------------------- simulation (1700p engine, g=None path; n and the rank map parameterised -- nothing else changed)
def simulate(rank_map, n=20, g=None, replace=True, barweeks=0, day=0, tm='open'):
    """One cell. rank_map[signal_date] = ranked candidate indices; n names, weekly 1/N reset of ALL names at Monday open. Returns E, cost_f, trade_f."""
    ns = len(cand); sh = np.zeros(ns); cash = START_EQ
    nd = T1 - T0 + 1; E = np.zeros(nd); cost_f = trade_f = 0.0
    rk = {didx[x]: rank_map[prior[x]] for x in DAYMAP[day] if prior[x] in rank_map and didx[x] <= T1}
    open_rk = rk
    for k in range(nd):
        ti = T0 + k; o = O[ti]; pre = cash + sh @ o; tr = cs = 0.0
        is_reset = ti in open_rk
        def reset(price, rkl):
            nonlocal cash, tr, cs
            eq = cash + sh @ price
            top = [i for i in rkl[:n] if price[i] > 0]
            tgt = np.zeros(ns); tgt[top] = eq / n; delta = tgt - sh * price
            for i in np.where(delta < -1e-9)[0]:
                v = -delta[i]; c = RATE[ti, i] * v; cash += v - c; sh[i] -= v / price[i]; tr += v; cs += c
                if sh[i] <= 1e-12: sh[i] = 0
            for i in np.where(delta > 1e-9)[0]:
                v = delta[i]; c = RATE[ti, i] * v; cash -= v + c; sh[i] += v / price[i]; tr += v; cs += c
        if is_reset: reset(o, open_rk[ti])
        E[k] = cash + sh @ o
        cost_f += cs / pre if pre > 0 else 0; trade_f += tr / pre if pre > 0 else 0
    return dict(E=E, cost_f=cost_f, trade_f=trade_f)

def dd_series(E): m = np.maximum.accumulate(E); return E / m - 1
dates = tdays[T0:T1 + 1]; yrs = (dates[-1] - dates[0]).days / 365.25
def cg(E): return (E[-1] / START_EQ) ** (1 / yrs) - 1

# ---------------------------------------------------------------- REF reproduction GATE (before anything else)
ref_map = {d: [cidx[s] for s in v] for d, v in ranked_syms.items()}
res = {'REF': simulate(ref_map, 20)}; E = res['REF']['E']
log.info('%s REF end $%.0f CAGR %.2f%% maxDD %.2f%%', el(), E[-1], 100 * cg(E), 100 * dd_series(E).min())
if abs(100 * cg(E) - 27.18) > .2 or abs(100 * dd_series(E).min() + 44.5) > .2 or abs(E[-1] - 507823) > 0.01 * 507823:
    log.error('REF DOES NOT REPRODUCE -- STOP'); raise SystemExit(2)
log.info('REPRO REF OK (27.18 / -44.5 / 507,823 within tolerance)')

# ---------------------------------------------------------------- rolling-beta residual scores
T = len(tdays); ns_all = len(cand) + 1                     # last column = SPY itself (sanity)
Cx = np.full((T, ns_all), np.nan); Cx[:, :len(cand)] = C; Cx[:, -1] = FACT['SPY']
Rm = np.full((T, ns_all), np.nan)
Rm[1:] = np.where(np.isfinite(Cx[1:]) & np.isfinite(Cx[:-1]), Cx[1:] / np.where(Cx[:-1] > 0, Cx[:-1], np.nan) - 1, np.nan)
Mk = np.isfinite(Rm); R0 = np.where(Mk, Rm, 0.0); Mf = Mk.astype(np.float64); del Rm, Cx
Fr = {f: np.r_[np.nan, FACT[f][1:] / FACT[f][:-1] - 1] for f in FACT}; Fr = {f: np.nan_to_num(v) for f, v in Fr.items()}
log.info('%s returns built; max |r| on masked entries %.2f', el(), float(np.abs(R0).max()))
FSETS = {'SPY': ['SPY'], '3F': ['SPY', 'QQQ', 'IWM']}
LBS = {252: 252, 126: 126}; WS = (252, 504)

def window_sums(A, s_rows):
    """Sum of A over the 4 windows (per signal row s): beta windows [s-W, s-1] (W 252/504); score windows [s-L, s-21] (L 252/126). Returns dict key -> (nS, ns)."""
    cs = np.zeros((A.shape[0] + 1, A.shape[1])); np.cumsum(A, axis=0, out=cs[1:]); out = {}
    for W in WS: out[('b', W)] = cs[s_rows] - cs[np.maximum(s_rows - W, 0)]
    for L in LBS: out[('s', L)] = cs[np.maximum(s_rows - 20, 0)] - cs[np.maximum(s_rows - L, 0)]
    return out

def compute_scores(R0_, Mf_, X, s_rows, fs_name):
    """Residual-momentum scores for signal rows s_rows from data rows <= s_rows only. Returns {(lb, W): (nS, ns) array}, betas {W: (nS, ns, k)}."""
    k = X.shape[1]; Mw = window_sums(Mf_, s_rows); Sr = window_sums(R0_, s_rows); Srr = window_sums(R0_ ** 2, s_rows)
    SFr = [window_sums(X[:, a:a + 1] * R0_, s_rows) for a in range(k)]; SF = [window_sums(X[:, a:a + 1] * Mf_, s_rows) for a in range(k)]
    SFF = {(a, b): window_sums((X[:, a] * X[:, b])[:, None] * Mf_, s_rows) for a in range(k) for b in range(a, k)}
    betas = {}
    for W in WS:
        key = ('b', W); cnt = Mw[key]
        with np.errstate(divide='ignore', invalid='ignore'):
            mx = [SF[a][key] / cnt for a in range(k)]; mr = Sr[key] / cnt
            Cxx = np.zeros(cnt.shape + (k, k)); Cxr = np.zeros(cnt.shape + (k,))
            for a in range(k):
                Cxr[..., a] = SFr[a][key] / cnt - mx[a] * mr
                for b in range(a, k): Cxx[..., a, b] = Cxx[..., b, a] = SFF[(a, b)][key] / cnt - mx[a] * mx[b]
        bad = ~np.isfinite(Cxx).all(axis=(-1, -2)) | ~np.isfinite(Cxr).all(axis=-1)
        Cxx[bad] = np.eye(k); Cxr[bad] = 0
        Cxx = Cxx + 1e-12 * np.eye(k)
        bt = np.linalg.solve(Cxx.reshape(-1, k, k), Cxr.reshape(-1, k, 1)).reshape(cnt.shape + (k,))
        need = 0.8 * np.minimum(W, s_rows)[:, None]
        bad |= cnt < need
        bt[bad] = np.nan; betas[W] = bt
    scores = {}
    for L in LBS:
        key = ('s', L); n_ = Mw[key]
        for W in WS:
            bt = betas[W]
            sm = Sr[key].copy(); sq = Srr[key].copy()
            for a in range(k): sm = sm - bt[..., a] * SF[a][key]; sq = sq - 2 * bt[..., a] * SFr[a][key]
            for a in range(k):
                for b in range(k): sq = sq + bt[..., a] * bt[..., b] * SFF[(min(a, b), max(a, b))][key]
            with np.errstate(divide='ignore', invalid='ignore'):
                var = (sq - sm ** 2 / n_) / (n_ - 1); sd = np.sqrt(np.where(var > 0, var, np.nan)); sc = sm / sd
            sc[(n_ < 20) | ~np.isfinite(bt).all(axis=-1) | (sd < 1e-9)] = np.nan
            scores[(L, W)] = sc
        log.info('%s %s lookback %d scores done', el(), fs_name, L)
    return scores, betas

sig_list = sorted(set(prior[d] for d in rebal_dates if prior[d] in elig_idx))
s_rows = np.array([didx[d] for d in sig_list]); san_date = pd.Timestamp('2024-12-31'); s_all = np.unique(np.r_[s_rows, didx[san_date]])
row_of = {int(s): j for j, s in enumerate(s_all)}
rank_maps = {}; beta_spy_W252 = None; checks = {}
for fs_name, fl in FSETS.items():
    X = np.column_stack([Fr[f] for f in fl]); sc, bt = compute_scores(R0, Mf, X, s_all, fs_name)
    if fs_name == 'SPY':
        j = row_of[didx[san_date]]
        for sym in ('NVDA', 'KO'): checks[f'beta_{sym}'] = tuple(float(bt[W][j, cidx[sym], 0]) for W in WS)
        checks['spy_resid'] = tuple(float(np.nan_to_num(sc[(252, W)][j, -1], nan=np.nan)) for W in WS)
        # SPY own residual sum (unnormalised): beta ~1 => ~0
        b_spy = bt[252][j, -1, 0]; checks['spy_beta_self'] = float(b_spy)
    for (L, W), S in sc.items():
        for nn in (20, 30):
            mp = {}; short = 0
            for d in sig_list:
                ids = elig_idx[d]; v = S[row_of[didx[d]], ids]; okm = np.isfinite(v)
                order = np.argsort(-v[okm], kind='stable')[:MAXN]; mp[d] = ids[okm][order].tolist()
                if len(mp[d]) < nn: short += 1
            rank_maps[(fs_name, L, W, nn)] = mp
        log.info('%s %s L%d W%d: rebalances with fewer than 30 scored names: %d of %d', el(), fs_name, L, W, short, len(sig_list))
    if fs_name == '3F':      # causality: recompute one signal date with ALL later data deleted
        d_chk = [d for d in sig_list if d >= pd.Timestamp('2022-06-01')][0]; s_ = didx[d_chk]
        sc2, _ = compute_scores(R0[:s_ + 1], Mf[:s_ + 1], X[:s_ + 1], np.array([s_]), '3F-truncated')
        ok_all = True
        for key_, S in sc.items():
            ids = elig_idx[d_chk]; a_ = S[row_of[s_], ids]; b_ = sc2[key_][0, ids]
            ra = ids[np.isfinite(a_)][np.argsort(-a_[np.isfinite(a_)], kind='stable')][:MAXN]; rb = ids[np.isfinite(b_)][np.argsort(-b_[np.isfinite(b_)], kind='stable')][:MAXN]
            ok_all &= bool(np.array_equal(ra, rb)) and bool(np.allclose(a_[np.isfinite(a_)], b_[np.isfinite(b_)], rtol=1e-9, atol=0))
        checks['causal'] = (str(d_chk.date()), ok_all)
    del sc, bt
for k_, v in checks.items(): log.info('SANITY %s = %s', k_, v)
if not checks['causal'][1]: log.error('CAUSALITY CHECK FAILED -- STOP'); raise SystemExit(3)
del R0, Mf

# ---------------------------------------------------------------- run the 16 cells
cells = [(fs, L, W, nn) for fs in FSETS for L in (252, 126) for nn in (20, 30) for W in WS]
def cname(c): return f'{c[0]}_{ {252: "12-1", 126: "6-1"}[c[1]] }_N{c[3]}_W{c[2]}'
for c in cells:
    res[cname(c)] = simulate(rank_maps[c], c[3]); Ec = res[cname(c)]['E']
    log.info('%s %s end $%.0f CAGR %.2f%% maxDD %.1f%%', el(), cname(c), Ec[-1], 100 * cg(Ec), 100 * dd_series(Ec).min())

# ---------------------------------------------------------------- reads (as 1700l)
spyE = spy_o[T0:T1 + 1] / spy_o[T0] * START_EQ
def yearly(E_):
    s = pd.Series(pd.Series(E_).pct_change().fillna(0).values, index=dates); return (1 + s).groupby(s.index.year).prod() - 1
spy_y = yearly(spyE); mstarts = [i for i in range(len(dates)) if i == 0 or dates[i].month != dates[i - 1].month]
def roll5(E_):
    w = b = 0
    for i in mstarts:
        j = dates.searchsorted(dates[i] + pd.DateOffset(years=5))
        if j >= len(dates): continue
        w += 1; b += (E_[j] / E_[i]) > (spyE[j] / spyE[i])
    return b / max(w, 1), w
E = res['REF']['E']; D = dd_series(E); runmax = np.maximum.accumulate(E); eps = []; i = 0
while i < len(E):
    if D[i] < 0:
        pk = int(np.where(E[:i] == runmax[i])[0][-1]) if i > 0 else 0
        j = i
        while j < len(E) and E[j] < runmax[i]: j += 1
        tr_i = pk + int(np.argmin(E[pk:j])); eps.append((pk, tr_i, j if j < len(E) else None, E[tr_i] / E[pk] - 1)); i = j
    else: i += 1
eps = sorted(eps, key=lambda x: x[3])[:5]
def ep_depths(Ec): return [(Ec[pk:(rc if rc else len(Ec))] / np.maximum.accumulate(Ec[pk:(rc if rc else len(Ec))]) - 1).min() for pk, tr, rc, _ in eps]
widx = np.array([didx[d] - T0 for d in rebal_dates]); wdates = pd.DatetimeIndex(rebal_dates)[1:]
wk_ref = pd.Series(E[widx]).pct_change().dropna().values
wk_spy = pd.Series(spy_o[T0 + widx]).pct_change().dropna().values
H = [(pd.Timestamp('2017-01-01'), pd.Timestamp('2022-01-01')), (pd.Timestamp('2022-01-01'), pd.Timestamp('2027-01-01'))]
def half_ratio(Ec, lo, hi):
    m = (dates >= lo) & (dates < hi); s = Ec[m]; y = (dates[m][-1] - dates[m][0]).days / 365.25
    return ((s[-1] / s[0]) ** (1 / y) - 1) / abs(dd_series(s).min())
ref_h = [half_ratio(E, *h) for h in H]; ref_cagr, ref_dd = cg(E), D.min(); ref_ratio = ref_cagr / abs(ref_dd)
def alpha_beta(wk):
    Xm = np.column_stack([np.ones(len(wk)), wk_spy]); coef, *_ = np.linalg.lstsq(Xm, wk, rcond=None); r = wk - Xm @ coef
    s2 = r @ r / (len(wk) - 2); cov = s2 * np.linalg.inv(Xm.T @ Xm); return coef[1], coef[0] * 52, coef[0] / np.sqrt(cov[0, 0])
def overlap(c):
    """Mean over rebalances of |cell names ∩ REF top-20| / N_cell."""
    a = []
    for d in rebal_dates:
        sd = prior[d]
        if sd in rank_maps[c] and sd in ref_map: A = set(rank_maps[c][sd][:c[3]]); a.append(len(A & set(ref_map[sd][:20])) / max(len(A), 1))
    return float(np.mean(a))
out = []
for name, r in res.items():
    Ec = r['E']; cagr = cg(Ec); mdd = dd_series(Ec).min(); y = yearly(Ec); wk = pd.Series(Ec[widx]).pct_change().dropna().values
    r5, nw = roll5(Ec); dif = wk - wk_ref; keepm = dif <= np.quantile(dif, 0.95); sd_ = dif.std(ddof=1); bta, alp, alt = alpha_beta(wk)
    row = dict(cell=name, cagr=cagr, max_dd=mdd, ratio=cagr / abs(mdd), end_usd=Ec[-1], sharpe=wk.mean() / wk.std(ddof=1) * np.sqrt(52), worst_year=y.min(), worst_year_label=int(y.idxmin()),
               years_beat_spy=int((y > spy_y).sum()), n_years=len(y), roll5_share=r5, beta_spy=bta, alpha_ann=alp, alpha_t=alt, corr_ref=float(np.corrcoef(wk, wk_ref)[0, 1]),
               overlap_ref=1.0 if name == 'REF' else overlap([c for c in cells if cname(c) == name][0]), turnover_oneway_per_yr=r['trade_f'] / 2 / yrs, cost_drag_per_yr=r['cost_f'] / yrs,
               paired_mean=dif.mean(), paired_t=dif.mean() / (sd_ / np.sqrt(len(dif))) if sd_ > 0 else np.nan, paired_ex_top5=dif[keepm].mean(),
               half1_ratio=half_ratio(Ec, *H[0]), half2_ratio=half_ratio(Ec, *H[1]))
    for k_, dpt in enumerate(ep_depths(Ec), 1): row[f'ep{k_}_depth'] = dpt
    row['improves'] = bool(name != 'REF' and mdd >= ref_dd + 0.08 and cagr >= 0.22 and row['ratio'] >= ref_ratio + 0.15)
    row['halves_beat'] = bool(name != 'REF' and row['half1_ratio'] > ref_h[0] and row['half2_ratio'] > ref_h[1])
    row['eps_cut'] = int(sum(row[f'ep{k_}_depth'] > ep_depths(E)[k_ - 1] for k_ in range(1, 6))) if name != 'REF' else 0
    out.append(row)
cdf = pd.DataFrame(out); cdf.to_csv(OUT / '1700r_cells.csv', index=False)
f16 = cdf[cdf.cell != 'REF']; n_imp = int(f16.improves.sum()); n_half = int(f16.halves_beat.sum())
med = f16.sort_values('ratio').iloc[len(f16) // 2]; n_cut = int(med.eps_cut)
real = n_imp >= 12 and n_half >= 12 and n_cut >= 3
verdict = 'REAL' if real else ('PARTIAL' if 6 <= n_imp <= 11 else 'FAIL')
# 50/50 blend of the median cell with REF (daily-rebalanced from daily equity returns)
Em = res[med.cell]['E']; rb = 0.5 * np.r_[0, np.diff(E) / E[:-1]] + 0.5 * np.r_[0, np.diff(Em) / Em[:-1]]; Eb = START_EQ * np.cumprod(1 + rb)
log.info('%s verdict %s improves %d halves %d median %s eps_cut %d; blend %.2f%% / %.1f%% / %.0f', el(), verdict, n_imp, n_half, med.cell, n_cut, 100 * cg(Eb), 100 * dd_series(Eb).min(), Eb[-1])

L = ['# RESULT 1,700r -- residual (beta-adjusted) momentum (PREREG_1700r.md, frozen)', '',
     f'REF reproduces: CAGR {ref_cagr:.2%} / max DD {ref_dd:.1%} / end ${E[-1]:,.0f} (target 27.18% / -44.5% / $507,823). Engine = 1700p daily engine copied (ONLY the ranking score changes); sim {dates[0].date()}..{dates[-1].date()}, $50K, band cost.',
     f'Sanity 2024-12-31 (SPY-only beta, W252/W504): NVDA {checks["beta_NVDA"][0]:.2f}/{checks["beta_NVDA"][1]:.2f}, KO {checks["beta_KO"][0]:.2f}/{checks["beta_KO"][1]:.2f}; SPY own beta {checks["spy_beta_self"]:.4f}, SPY residual score {checks["spy_resid"][0]} (0/0 noise expected).',
     f'Causality: scores for signal date {checks["causal"][0]} recomputed with all later data deleted -> identical ranks and scores (all 8 score matrices): {checks["causal"][1]}.', '',
     '| cell | CAGR | max DD | ratio | end $ | Sharpe | beta | alpha/yr (t) | corr REF | overlap | halves r/REF | ep1..5 | improves |', '|' + '---|' * 13]
for _, x in cdf.iterrows():
    L.append(f"| {x['cell']} | {x.cagr:.1%} | {x.max_dd:.1%} | {x.ratio:.2f} | {x.end_usd:,.0f} | {x.sharpe:.2f} | {x.beta_spy:.2f} | {x.alpha_ann:+.1%} ({x.alpha_t:.1f}) | {x.corr_ref:.2f} | {x.overlap_ref:.0%} | "
             f"{x.half1_ratio/ref_h[0]:.2f}/{x.half2_ratio/ref_h[1]:.2f} | " + ' '.join(f"{x[f'ep{k}_depth']:.0%}" for k in range(1, 6)) + f" | {'YES' if x.improves else ('-' if x.cell != 'REF' else 'ref')} |")
L += ['', f'Pass rule (DD >= 8 pts better AND CAGR >= 22% AND ratio >= REF {ref_ratio:.2f}+0.15): improves {n_imp}/16 (need 12); ratio beats REF in BOTH halves {n_half}/16 (need 12); median cell by ratio {med.cell} cuts {n_cut}/5 episodes (need 3).',
      f'## Family verdict: **{verdict}**', '',
      f'Median cell {med.cell}: {med.cagr:.1%} / {med.max_dd:.1%} / ${med.end_usd:,.0f}, beta {med.beta_spy:.2f}, alpha {med.alpha_ann:+.1%}/yr (t {med.alpha_t:.1f}), corr with REF {med.corr_ref:.2f}, holdings overlap {med.overlap_ref:.0%}, paired weekly diff vs REF {med.paired_mean*100:+.3f}% (t {med.paired_t:.1f}, ex-top-5% {med.paired_ex_top5*100:+.3f}%).',
      f'50/50 blend median+REF (daily-rebalanced, informational): {cg(Eb):.1%} / {dd_series(Eb).min():.1%} / ${Eb[-1]:,.0f}. ' + ('Goes to an independent rebuild before any paper flag.' if real else 'No recommendation (family not REAL).'), '',
      '## Caveats (adversary)',
      '- Cost model is the 1700c band (not measured NBBO); residual names may differ in liquidity from REF, so cost parity is assumed, not shown.',
      '- W504 early-sample: the panel starts 2016-01, so in 2017 the 504-day window is truncated; presence rule applied to min(W, days available) -- W504 cells differ from W252 partly by sample at the start.',
      '- One beta set per name-day applied to the whole score window (in-sample for the 12-1 window when W covers it); betas are noisy for names with <1 yr of history.',
      '- U2 eligibility and the 40-name pre-cut are NOT applied to residual cells (they rank all eligible names, REF ranks the top-40 by sigV2 then takes 20): the candidate pool differs, not only the score.',
      '- Close prices are as stored in the panel (adjustment status inherited from 1700j); no per-trade price-scale check was run on the residual cells; max |daily return| logged in 1700r.log.',
      '- Overlap = mean share of the cell\'s N names also in REF top-20. Blend is informational only. Multiplicity: +16 cells on this line; single sample, 2017-2026 bull-heavy, halves are not independent of the theme.']
(OUT / 'RESULT_1700r.md').write_text('\n'.join(L) + '\n'); log.info('DONE %s', el())
