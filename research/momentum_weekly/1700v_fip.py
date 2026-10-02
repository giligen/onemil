#!/usr/bin/env python3
"""Cell 1,700v: information-discreteness (continuous-path momentum) selection family, 16 cells, on the guarded sleeve GREF; one process / one panel load; engine, loader and guard copied from 1700s_lowvix.py.
Run: bash scripts/research_run.sh -m 2500M python3 research/momentum_weekly/1700v_fip.py > research/momentum_weekly/1700v.out"""
from __future__ import annotations
import logging, re, sys, time, itertools
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('/home/ec2-user/onemil/research/momentum_weekly')
logging.basicConfig(filename=str(OUT / '1700v.log'), filemode='w', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700v'); log.addHandler(logging.StreamHandler(sys.stdout))
PRICE_MIN, ADV_CUT, START_EQ = 10.0, 200_000_000.0, 50_000.0
WIN_START, WIN_END = pd.Timestamp('2017-01-01'), pd.Timestamp('2026-09-30')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|\bPREFERRED\b|\bRIGHTS?\b', re.I)
MAXN = 40
t0 = time.time()
def el(): return f'{time.time()-t0:5.0f}s'

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
spread_ = (((h_[keep] - l_[keep]) / c_[keep]).clip(min=0) * 0.1).clip(max=0.002).astype(np.float32)
dv = (c_[keep] * v_[keep]).astype(np.float64); cl = c_[keep]
del raw, _c, _d, o_, h_, l_, c_, v_, keep, dup_next
import gc; gc.collect()
panel['spread'] = spread_; del spread_

def build_signals(codes, dts, cl, dv):
    """Per-symbol backward-looking signals: adv20, sigV2, ok, gtype (guard) and information discreteness ID for the 12-1 (252..21) and 6-1 (126..21) windows.
    ID = sign(window return) * (share of down days - share of up days) over the window's close-to-close returns; zero-return days count in neither share
    but stay in the denominator. Uses only rows <= t (the Friday close)."""
    N = len(codes); adv20 = np.full(N, np.nan); sigV2 = np.full(N, np.nan); okv = np.zeros(N, bool); gtype = np.zeros(N, np.int8)
    idA = np.full(N, np.nan, np.float32); idB = np.full(N, np.nan, np.float32)
    bounds = np.flatnonzero(np.diff(codes)) + 1; starts = np.r_[0, bounds]; ends = np.r_[bounds, N]
    for a, b in zip(starts, ends):
        c = pd.Series(cl[a:b]); adv20[a:b] = pd.Series(dv[a:b]).rolling(20, min_periods=20).mean().values
        c21, c252, c273 = c.shift(21), c.shift(252), c.shift(273)
        ret = c.pct_change(); vol = ret.rolling(252, min_periods=252).std().values
        sig = (c21 / c252 - 1).values
        with np.errstate(divide='ignore', invalid='ignore'): sigV2[a:b] = np.where(vol > 0, sig / vol, np.nan)
        okv[a:b] = c273.notna().values
        cc = cl[a:b].astype(np.float64); m = b - a; mv = np.zeros(m); mv[1:] = cc[1:] / cc[:-1] - 1; gp = np.zeros(m, bool); gp[1:] = np.diff(dts[a:b]).astype('timedelta64[D]').astype(np.int64) > 10
        ea = pd.Series(((mv > 2.0) | (mv < -0.75)).astype(np.float64)).rolling(273, min_periods=1).max().values > 0
        eb = pd.Series(gp.astype(np.float64)).rolling(273, min_periods=1).max().values > 0
        gtype[a:b] = ea.astype(np.int8) + 2 * eb.astype(np.int8)
        cu_up = np.cumsum(mv > 0); cu_dn = np.cumsum(mv < 0)      # mv[0]=0 -> counted nowhere
        for out, L in ((idA, 252), (idB, 126)):
            if m > L:
                t_ = np.arange(L, m); s_ = t_ - L; e_ = t_ - 21; n_ = e_ - s_
                up = (cu_up[e_] - cu_up[s_]) / n_; dn = (cu_dn[e_] - cu_dn[s_]) / n_; wr = cc[e_] / cc[s_] - 1
                out[a + L:b] = (np.sign(wr) * (dn - up)).astype(np.float32)
    return adv20, sigV2, okv, gtype, idA, idB
panel['symbol'] = panel.symbol.cat.remove_unused_categories()
codes = panel.symbol.values.codes; dts = panel.bar_date.values
adv20, sigV2, okv, gtype, idA, idB = build_signals(codes, dts, cl, dv)
panel['adv20'] = adv20; panel['sigV2'] = sigV2; panel['ok'] = okv; panel['gtype'] = gtype; panel['idA'] = idA; panel['idB'] = idB; del adv20, sigV2, okv, gtype, idA, idB
log.info('%s signals built (%d rows)', el(), len(panel))
import warnings; warnings.filterwarnings('ignore')
from scipy import stats
def say(*a): print(*a, flush=True)

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
sr = panel.loc[panel.bar_date.isin(sigdates) & (panel.close >= PRICE_MIN) & panel.ok & (panel.adv20 >= ADV_CUT) & panel.sigV2.notna(), ['symbol', 'bar_date', 'sigV2', 'gtype', 'idA', 'idB']]
sr['symbol'] = sr['symbol'].astype(str)

# ---- sanity prints: ID of AMC and NVDA
for sym in ('AMC', 'NVDA'):
    for dd_ in ('2021-06-04', '2024-06-07'):
        r = panel[(panel.symbol == sym) & (panel.bar_date == pd.Timestamp(dd_))]
        say(f'ID {sym} {dd_}: ' + (f'idA(252..21) {float(r.idA.iloc[0]):+.4f} idB(126..21) {float(r.idB.iloc[0]):+.4f} guard_type {int(r.gtype.iloc[0])}' if len(r) else 'no bar'))

WINC = {252: 'idA', 126: 'idB'}
def pick(sub, sel, wcol):
    """One signal date's holdings list. sub = eligible GUARDED rows (gtype 0) with sigV2 (the sleeve score) and ID. A: drop names above the cross-sectional
    ID quantile of the whole eligible guarded universe, rank the rest by score. B: take the K best by score, order them by ID ascending (lowest ID first)."""
    sub = sub[sub[wcol].notna()]
    if sel in ('A50', 'A67'):
        thr = sub[wcol].quantile(0.5 if sel == 'A50' else 2 / 3); return sub[sub[wcol] <= thr].nlargest(MAXN, 'sigV2')['symbol'].tolist()
    k = 40 if sel == 'B40' else 60
    top = sub.nlargest(k, 'sigV2').sort_values([wcol, 'sigV2'], ascending=[True, False]); return top['symbol'].tolist()[:MAXN]
SELS = [(s, w) for s in ('A50', 'A67', 'B40', 'B60') for w in (252, 126)]
sg = sr[sr.gtype == 0]
ranked_g = {d: sub.nlargest(MAXN, 'sigV2')['symbol'].tolist() for d, sub in sg.groupby('bar_date', observed=True)}
RM = {(s, w): {d: pick(sub, s, WINC[w]) for d, sub in sg.groupby('bar_date', observed=True)} for s, w in SELS}
log.info('%s selections built', el())

CATS = panel.symbol.cat.categories
cand = sorted({s for v in ranked_g.values() for s in v} | {s for m_ in RM.values() for v in m_.values() for s in v}); cidx = {s: i for i, s in enumerate(cand)}
sub = panel[panel.symbol.isin(cand)]
def piv(col): return sub.pivot_table(index='bar_date', columns='symbol', values=col, aggfunc='last', observed=True).reindex(index=tdays, columns=cand)
O_raw, Cdf, SP = piv('open'), piv('close'), piv('spread')
O = O_raw.ffill().fillna(0.0).values.astype(np.float64)
RATE = np.minimum(0.0005 + 0.5 * np.nan_to_num(SP.values.astype(np.float64), nan=0.002), 0.002)
spy_o = spy_df.set_index('bar_date')['open'].reindex(tdays).ffill().values.astype(np.float64)
del panel, sub, O_raw, SP, Cdf; gc.collect()
log.info('%s candidates %d, pivots %s', el(), len(cand), O.shape)
# ---- causality check: delete all data after one Friday, rebuild signals, selections must be identical
_dates_sorted = sorted(d for d in ranked_g if d >= pd.Timestamp('2021-01-01')); D = pd.Timestamp('2021-06-04') if pd.Timestamp('2021-06-04') in ranked_g else _dates_sorted[0]
msk = dts <= np.datetime64(D)
a2, s2, o2, g2, iA2, iB2 = build_signals(codes[msk], dts[msk], cl[msk], dv[msk])
kd = dts[msk] == np.datetime64(D); df2 = pd.DataFrame({'symbol': CATS[codes[msk][kd]].astype(str), 'bar_date': dts[msk][kd], 'close': cl[msk][kd], 'adv20': a2[kd], 'sigV2': s2[kd], 'ok': o2[kd], 'gtype': g2[kd], 'idA': iA2[kd], 'idB': iB2[kd]})
df2 = df2[(df2.bar_date == D) & (df2.close >= PRICE_MIN) & df2.ok & (df2.adv20 >= ADV_CUT) & df2.sigV2.notna() & (df2.gtype == 0)]
CAUS_BAD = [f'{s}|{w}' for s, w in SELS if pick(df2, s, WINC[w]) != RM[(s, w)][D]]
CAUS = f'causality: all data after {D.date()} deleted, signals+selections rebuilt for that date: {len(SELS) - len(CAUS_BAD)}/{len(SELS)} cells identical holdings (mismatch: {CAUS_BAD or "none"}); eligible n={len(df2)}'
say(CAUS); del a2, s2, o2, g2, iA2, iB2, df2, msk

del dv, cl, codes, dts; gc.collect()
rebal_dates = [d for d in ent if d in prior and prior[d] in ranked_g]
T0, T1 = didx[rebal_dates[0]], didx[rebal_dates[-1]]

def simulate(rank_map, n=20):
    """Monday-open weekly 1/N reset on the daily engine (identical to 1700s simulate with scale=None); returns equity, cost and turnover fractions, and the held set per rebalance."""
    ns = len(cand); sh = np.zeros(ns); cash = START_EQ; nd = T1 - T0 + 1; E = np.zeros(nd); cost_f = trade_f = 0.0; held = {}
    rk = {didx[x]: [cidx[s] for s in rank_map[prior[x]]] for x in DAYMAP[0] if prior[x] in rank_map and didx[x] <= T1}
    for k in range(nd):
        ti = T0 + k; o = O[ti]; pre = cash + sh @ o; tr = cs = 0.0
        if ti in rk:
            eq = cash + sh @ o
            top = [i for i in rk[ti][:n] if o[i] > 0]; held[ti] = set(top)
            tgt = np.zeros(ns); tgt[top] = eq / n
            delta = tgt - sh * o
            for i in np.where(delta < -1e-9)[0]:
                v = -delta[i]; c = RATE[ti, i] * v; cash += v - c; sh[i] -= v / o[i]; tr += v; cs += c
                if sh[i] <= 1e-12: sh[i] = 0
            for i in np.where(delta > 1e-9)[0]:
                v = delta[i]; c = RATE[ti, i] * v; cash -= v + c; sh[i] += v / o[i]; tr += v; cs += c
        E[k] = cash + sh @ o
        cost_f += cs / pre if pre > 0 else 0; trade_f += tr / pre if pre > 0 else 0
    return dict(E=E, cost_f=cost_f, trade_f=trade_f, held=held)
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
def tstat(x): return float(np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and np.std(x) > 0 else np.nan

# ---- GREF repro gate
tG = simulate(ranked_g); E_G = tG['E']; c0, d0, r0 = stats_of(E_G)
say(f'GREF CAGR {c0:.4%} maxDD {d0:.3%} end ${E_G[-1]:,.0f}')
GREF_OK = abs(c0 - 0.2934) < 0.0005 and abs(d0 + 0.383) < 0.0006 and abs(E_G[-1] - 596394) < 0.002 * 596394
if not GREF_OK:
    (OUT / 'RESULT_1700v.md').write_text(f'GREF DOES NOT REPRODUCE: CAGR {c0:.4f} DD {d0:.4f} end {E_G[-1]:.0f} (target 29.34 / -38.3 / 596,394). STOP, no cell read.\n'); log.error('GREF NOT REPRODUCED'); raise SystemExit(2)
g_h1, g_h2 = stats_of(E_G, m1), stats_of(E_G, m2); wk_g = weekly(E_G); yG = yearly(E_G)
spy_wk = pd.Series(spyE[widx]).pct_change().dropna().values
# GREF's three deepest episodes
runmax = np.maximum.accumulate(E_G); eps = []; i = 0
while i < len(E_G):
    if E_G[i] < runmax[i]:
        pk = int(np.where(E_G[:i] == runmax[i])[0][-1]) if i > 0 else 0; j = i
        while j < len(E_G) and E_G[j] < runmax[i]: j += 1
        tr_i = pk + int(np.argmin(E_G[pk:j])); eps.append((pk, tr_i, j if j < len(E_G) else None, E_G[tr_i] / E_G[pk] - 1)); i = j
    else: i += 1
eps = sorted(eps, key=lambda x: x[3])[:3]
def ep_depths(Ec): return [(Ec[pk:(rc if rc else len(Ec))] / np.maximum.accumulate(Ec[pk:(rc if rc else len(Ec))]) - 1).min() for pk, tr, rc, _ in eps]
ep_g = ep_depths(E_G); ep_lab = [f'{dates[pk].date()}..{dates[tr].date()}' for pk, tr, rc, _ in eps]
say('GREF episodes', list(zip(ep_lab, [round(x, 3) for x in ep_g])))

def regress(wk):
    """OLS of weekly cell return on weekly SPY open-to-open return: beta, alpha (weekly), alpha t."""
    X = np.c_[np.ones(len(spy_wk)), spy_wk]; b = np.linalg.lstsq(X, wk, rcond=None)[0]; res = wk - X @ b
    se = np.sqrt(res @ res / (len(wk) - 2) * np.linalg.inv(X.T @ X)[0, 0]); return b[1], b[0], b[0] / se
rows, CE = [], {}
for sel, win in SELS:
    for n in (20, 30):
        name = f'{sel}|w{win}|N{n}'; r = simulate(RM[(sel, win)], n); Ec = r['E']; CE[name] = Ec
        c, dd, ra = stats_of(Ec); c1, d1, ra1 = stats_of(Ec, m1); c2, d2, ra2 = stats_of(Ec, m2)
        wk = weekly(Ec); dif = wk - wk_g; kp = dif <= np.quantile(dif, 0.95); y = yearly(Ec); beta, alpha, at = regress(wk)
        gh = tG['held'] if n == 20 else simulate(ranked_g, n)['held']; ov = float(np.mean([len(r['held'][t] & gh[t]) / n for t in r['held'] if t in gh]))
        row = dict(cell=name, selection=sel, window=win, N=n, cagr=c, max_dd=dd, ratio=ra, end_usd=Ec[-1], sharpe=float(wk.mean() / wk.std(ddof=1) * np.sqrt(52)), worst_year=float(y.min()), years_beat_spy=int((y > spy_y).sum()), n_years=len(y),
                   ratio_h1=ra1, ratio_h2=ra2, gref_ratio_h1=g_h1[2], gref_ratio_h2=g_h2[2], beta=beta, alpha_wk=alpha, alpha_t=at, wk_corr_gref=float(np.corrcoef(wk, wk_g)[0, 1]), overlap_gref=ov,
                   turnover_per_yr=r['trade_f'] / yrs, cost_drag_per_yr=r['cost_f'] / yrs, paired_mean=float(dif.mean()), paired_t=tstat(dif), paired_ex_top5=float(dif[kp].mean()))
        for k_, dp in enumerate(ep_depths(Ec), 1): row[f'ep{k_}_depth'] = dp; row[f'ep{k_}_gref'] = ep_g[k_ - 1]
        row['improve'] = bool(ra >= r0 + 0.10 and c >= 0.25); row['ratio_both'] = bool(ra1 > g_h1[2] and ra2 > g_h2[2]); row['paired_ok'] = bool(row['paired_ex_top5'] >= 0)
        rows.append(row); say(f'{name}: CAGR {c:.2%} DD {dd:.1%} ratio {ra:.2f} end ${Ec[-1]:,.0f} ov {ov:.2f} improve={row["improve"]} both={row["ratio_both"]} pairedOK={row["paired_ok"]}')
cdf = pd.DataFrame(rows); cdf.to_csv(OUT / '1700v_cells.csv', index=False)
n_imp, n_rb, n_pp = int(cdf.improve.sum()), int(cdf.ratio_both.sum()), int(cdf.paired_ok.sum())
order = cdf.sort_values('ratio', ascending=False).reset_index(drop=True); med = order.iloc[8]       # 9th of 16 by ratio (upper-middle)
n_ep = int(sum(1 for k_ in (1, 2, 3) if med[f'ep{k_}_depth'] - med[f'ep{k_}_gref'] >= 0.03))
verdict = 'REAL' if (n_imp >= 12 and n_rb >= 12 and n_pp >= 8 and n_ep >= 2) else ('partial' if n_imp >= 6 else 'FAIL')
axis = cdf.groupby('selection').improve.sum().to_dict(), cdf.groupby('window').improve.sum().to_dict(), cdf.groupby('N').improve.sum().to_dict()
say(f'FAMILY improve {n_imp}/16 both-halves {n_rb}/16 paired ex5>=0 {n_pp}/16 median-cell episodes shallower>=3pt {n_ep}/3 VERDICT {verdict}; improve by selection/window/N {axis}')
yM = yearly(CE[med.cell])
L = ['# RESULT 1,700v -- information-discreteness (continuous-path) selection on the guarded sleeve (PREREG_1700v.md, FROZEN)', '',
     f'GREF reproduced BEFORE any cell was read: CAGR {c0:.2%} / max DD {d0:.1%} / end ${E_G[-1]:,.0f} (target 29.34 / -38.3 / 596,394); ratio {r0:.2f}, halves {g_h1[2]:.2f}/{g_h2[2]:.2f}. 2017-01..2026-09, $50K, 1700s engine, guard ON.',
     'ID = sign(window ret) x (share down days - share up days) of close-to-close returns, window rows t-252..t-21 (231 returns) or t-126..t-21 (105); zero-return days in neither share, kept in the denominator; t = Friday close before the Monday rebalance.',
     'A50/A67: eligible guarded universe, drop ID above the cross-sectional median / 2/3 quantile, rank the rest by sleeve score. B40/B60: K best by score, hold the N lowest ID.',
     CAUS, 'Sanity ID prints (AMC, NVDA on 2021-06-04, 2024-06-07): see 1700v.out.',
     '| cell | CAGR | DD | ratio | end $K | h1/h2 ratio | beta | ov | corr | pairedEx5 | I | R | P |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for _, x in cdf.iterrows():
    L.append(f"| {x.cell} | {x.cagr:.1%} | {x.max_dd:.1%} | {x.ratio:.2f} | {x.end_usd/1e3:,.0f} | {x.ratio_h1:.2f}/{x.ratio_h2:.2f} | {x.beta:.2f} | {x.overlap_gref:.2f} | {x.wk_corr_gref:.2f} | {x.paired_ex_top5*100:+.3f}% | {'Y' if x.improve else '.'} | {'Y' if x.ratio_both else '.'} | {'Y' if x.paired_ok else '.'} |")
L += ['', f'COUNTS: improve (ratio >= GREF {r0:.2f}+0.10 AND CAGR >= 25%) {n_imp}/16 (need 12); ratio beats GREF in both halves {n_rb}/16 (need 12); paired weekly diff >= 0 ex-top-5% {n_pp}/16 (need 8); median cell shallower by >= 3 pts in {n_ep}/3 GREF episodes (need 2).',
      f'VERDICT: {verdict}. Improve by selection {axis[0]}, window {axis[1]}, N {axis[2]}.',
      f'Median cell (9th of 16 by ratio): {med.cell} CAGR {med.cagr:.2%} / DD {med.max_dd:.1%} / end ${med.end_usd:,.0f} / ratio {med.ratio:.2f} / Sharpe {med.sharpe:.2f} / beta {med.beta:.2f} (alpha t {med.alpha_t:.1f}) / overlap with GREF {med.overlap_gref:.0%} / weekly corr {med.wk_corr_gref:.2f} / turnover {med.turnover_per_yr:.1f}x, cost {med.cost_drag_per_yr*100:.2f}%/yr / paired {med.paired_mean*100:+.3f}%/wk (t {med.paired_t:.1f}, ex-top5% {med.paired_ex_top5*100:+.3f}%) / worst yr {med.worst_year:.1%} / yrs>SPY {med.years_beat_spy}/{med.n_years}.',
      'GREF three deepest episodes (GREF depth -> median cell depth): ' + '; '.join(f'{ep_lab[k]} {ep_g[k]:.1%} -> {med[f"ep{k+1}_depth"]:.1%}' for k in range(3)),
      'Median cell by year (GREF | cell | SPY): ' + ' '.join(f'{y_}: {yG[y_]:.0%}|{yM[y_]:.0%}|{spy_y[y_]:.0%}' for y_ in yM.index),
      'Adversary caveats: (1) 16 neighbours of one published idea, not independent; family count +16; all share GREF engine, band-model costs (not NBBO) and the guard, whose own selection (273-bar lookback) was whole-sample chosen. (2) 2020 crash is the shared floor; ID cannot see it. (3) A-cells use a cross-sectional ID quantile over the whole eligible universe (~ price>=10, ADV>=200M); B-cells a rank-within-top-K re-order, so the N lowest-ID rule is also a K-N cut. (4) Two half-splits only; paired ex-top-5% trims the best weeks only. (5) Daily bars: zero-return days include illiquid unchanged closes; ID denominators keep them. (6) N=30 cells are compared with GREF N=20 (paired/overlap use GREF at the cell N for overlap only).']
(OUT / 'RESULT_1700v.md').write_text('\n'.join(L) + '\n'); log.info('DONE %s', el()); say('DONE')
