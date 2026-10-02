#!/usr/bin/env python3
"""Cell 1,700k (diversification caps, PREREG_1700k) built on the reconciled 1,700j engine. Original doc: Cell 1,700j: drawdown frontier of the reconciled momentum sleeve (V2 risk-adjusted top-N, weekly).
PREREG_1700j.md + Amendment 1. Daily-resolution simulation (open-to-open valuation, Monday rebalance to equal
weight with delta-trade costs, cost rate = 5 bps + half the (H-L)/C*0.1 proxy, total capped 20 bps). Name stops
read DAILY CLOSES vs the highest close since entry and execute at the NEXT day's open (cost charged); a name
stopped at a Monday open is not re-bought at that same rebalance. Stopped cash earns 0 until the next rebalance.
Vol target: exposure = min(1, target / (63d realised vol of the book's unscaled daily return)), set at each
Monday. D1: exposure 0.5 at a Monday when equity is >15% below its own high. C1: kept names neither sold nor
re-bought (entrants get equity/N). Episode 'cut' (pre-committed here, PREREG silent): a cell's depth over the
reference episode window is >= 25% shallower (relative) than the reference's depth.
Copied panel/signal code from 1700g_vol.py (module-scope grid there, so no import)."""
from __future__ import annotations
import logging, re, sys, time, traceback
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('/home/ec2-user/onemil/research/momentum_weekly')
logging.basicConfig(filename=str(OUT / '1700k.log'), filemode='w', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700k'); log.addHandler(logging.StreamHandler(sys.stdout))
PRICE_MIN, ADV_CUT, START_EQ = 10.0, 200_000_000.0, 50_000.0
WIN_START, WIN_END = pd.Timestamp('2017-01-01'), pd.Timestamp('2026-09-30')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|\bPREFERRED\b|\bRIGHTS?\b', re.I)
MAXN = 100
t0 = time.time()
sys.excepthook = lambda *a: log.error('ERROR %s', ''.join(traceback.format_exception(*a)))
def el(): return f'{time.time()-t0:5.0f}s'

# ---------------------------------------------------------------- panel + signals (as 1700g)
cols = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']
raw = pd.read_parquet(OUT / 'panel_2016_2026.parquet', columns=cols)
raw['symbol'] = raw['symbol'].astype('category')
for c in ('open', 'high', 'low', 'close'): raw[c] = raw[c].astype('float32')
raw['volume'] = raw['volume'].astype('float32')
raw['bar_date'] = pd.to_datetime(raw['bar_date'])
raw = raw.drop_duplicates(subset=['symbol', 'bar_date'], keep='last')
raw = raw[~((raw.open <= 0) | (raw.high <= 0) | (raw.low <= 0) | (raw.close <= 0))].reset_index(drop=True)
spy_df = raw.loc[raw.symbol == 'SPY', ['bar_date', 'open', 'close']].sort_values('bar_date').reset_index(drop=True)
if spy_df.empty: log.error('SPY missing'); raise SystemExit(1)
tdays = pd.DatetimeIndex(sorted(spy_df.bar_date.unique())); didx = {d: i for i, d in enumerate(tdays)}
assets = pd.read_csv(OUT / '1700c_assets.csv', dtype={'symbol': str, 'name': str}); assets['name'] = assets['name'].fillna('')
excl = set(assets.loc[assets.name.str.contains(NAME_RE), 'symbol']) | {s for s in raw.symbol.unique() if TEST_RE.match(s)}
panel = raw[(raw.symbol != 'SPY') & ~raw.symbol.isin(excl)].copy(); del raw
panel['symbol'] = panel.symbol.cat.remove_unused_categories()
panel = panel.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
g = panel.groupby('symbol', sort=False, observed=True)
panel['adv20'] = (panel.close * panel.volume).groupby(panel.symbol, observed=True).rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
panel['spread'] = (((panel.high - panel.low) / panel.close).clip(lower=0) * 0.1).clip(upper=0.002)
c21, c252, c273 = g['close'].shift(21), g['close'].shift(252), g['close'].shift(273)
panel['sig12_1'] = c21 / c252 - 1
panel['ret1d'] = g['close'].pct_change()
panel['vol252'] = panel.groupby('symbol', sort=False, observed=True)['ret1d'].rolling(252, min_periods=252).std().reset_index(level=0, drop=True)
with np.errstate(divide='ignore', invalid='ignore'):
    panel['sigV2'] = np.where(panel.vol252 > 0, panel.sig12_1 / panel.vol252, np.nan)
panel['ok'] = c273.notna(); del c21, c252, c273
log.info('%s signals built (%d rows)', el(), len(panel))

# weekly calendar (first trading day of each week), prior-day signal
per = pd.Series(tdays).dt.to_period('W'); first = pd.Series(tdays).groupby(per).min().sort_index()
ent = first[(first >= WIN_START) & (first <= WIN_END)].reset_index(drop=True)
prior = {d: tdays[didx[d] - 1] for d in ent if didx[d] > 0}
sigdates = set(prior.values())
sr = panel.loc[panel.bar_date.isin(sigdates) & (panel.close >= PRICE_MIN) & panel.ok & (panel.adv20 >= ADV_CUT) & panel.sigV2.notna(),
               ['symbol', 'bar_date', 'sigV2']]
ranked_syms = {d: sub.nlargest(MAXN, 'sigV2')['symbol'].astype(str).tolist() for d, sub in sr.groupby('bar_date', observed=True)}
cand = sorted({s for v in ranked_syms.values() for s in v}); cidx = {s: i for i, s in enumerate(cand)}
sub = panel[panel.symbol.isin(cand)]
def piv(col): return sub.pivot_table(index='bar_date', columns='symbol', values=col, aggfunc='last', observed=True).reindex(index=tdays, columns=cand)
O_raw, C, SP = piv('open'), piv('close'), piv('spread')
O = O_raw.ffill().fillna(0.0).values.astype(np.float64)
C = C.values.astype(np.float64)
RATE = np.minimum(0.0005 + 0.5 * np.nan_to_num(SP.values.astype(np.float64), nan=0.002), 0.002)
spy_o = spy_df.set_index('bar_date')['open'].reindex(tdays).ffill().values.astype(np.float64)
del panel, sub, sr, O_raw, SP
log.info('%s candidates %d, panel pivots %s', el(), len(cand), O.shape)
rebal_dates = [d for d in ent if d in prior and prior[d] in ranked_syms]
rebal = {didx[d]: [cidx[s] for s in ranked_syms[prior[d]]] for d in rebal_dates}
T0, T1 = didx[rebal_dates[0]], didx[rebal_dates[-1]]
log.info('%s rebalances %d, %s..%s', el(), len(rebal), rebal_dates[0].date(), rebal_dates[-1].date())

def simulate(n=20, stop=None, vt=None, dd=False, drift=False, keep=False, rb=None):
    """Daily simulation of one cell. Returns dict of equity series, costs, turnover, reduced weeks, shares history."""
    ns = len(cand); sh = np.zeros(ns); cash = START_EQ; hc = np.full(ns, np.nan); pend = np.zeros(ns, bool)
    nd = T1 - T0 + 1; E = np.zeros(nd); G = []; e_cur = 1.0; peak = START_EQ
    cost_sum = trade_sum = 0.0; red = 0; SH = np.zeros((nd, ns), np.float32) if keep else None
    prev_post = START_EQ; cost_f = 0.0; trade_f = 0.0
    for k in range(nd):
        ti = T0 + k; o = O[ti]
        pre = cash + sh @ o
        if k > 0: G.append((pre / prev_post - 1) / max(e_cur, 1e-9))
        blocked = set(); tr = 0.0; cs = 0.0
        if pend.any():
            for i in np.where(pend)[0]:
                v = sh[i] * o[i]; c = RATE[ti, i] * v; cash += v - c; sh[i] = 0; hc[i] = np.nan; blocked.add(i); tr += v; cs += c
            pend[:] = False
        if ti in rebal:
            eq = cash + sh @ o
            e = 1.0
            if vt and len(G) >= 63:
                vol = np.std(G[-63:], ddof=1) * np.sqrt(252); e = min(1.0, vt / vol) if vol > 0 else 1.0
            if dd and eq / peak - 1 < -0.15: e = 0.5
            e_cur = e; red += e < 1.0
            top = [i for i in (rb or rebal)[ti][:n] if i not in blocked and o[i] > 0]
            tgt = np.zeros(ns); tgt[top] = e * eq / n
            if drift:
                for i in np.where((sh > 0) & (tgt == 0))[0]:
                    v = sh[i] * o[i]; c = RATE[ti, i] * v; cash += v - c; sh[i] = 0; hc[i] = np.nan; tr += v; cs += c
                new = [i for i in top if sh[i] == 0]
                want = sum(tgt[i] for i in new)
                scale = min(1.0, max(cash, 0) / want) if want > 0 else 1.0
                for i in new:
                    v = tgt[i] * scale; c = RATE[ti, i] * v; cash -= v + c; sh[i] = v / o[i]; hc[i] = np.nan; tr += v; cs += c
            else:
                delta = tgt - sh * o
                for i in np.where(delta < -1e-9)[0]:
                    v = -delta[i]; c = RATE[ti, i] * v; cash += v - c; sh[i] -= v / o[i]; tr += v; cs += c
                    if sh[i] <= 1e-12: sh[i] = 0; hc[i] = np.nan
                for i in np.where(delta > 1e-9)[0]:
                    v = delta[i]; c = RATE[ti, i] * v; cash -= v + c
                    if sh[i] == 0: hc[i] = np.nan
                    sh[i] += v / o[i]; tr += v; cs += c
        post = cash + sh @ o; E[k] = post; prev_post = post; peak = max(peak, post)
        cost_f += cs / pre if pre > 0 else 0; trade_f += tr / pre if pre > 0 else 0; cost_sum += cs; trade_sum += tr
        if keep: SH[k] = sh
        if stop:
            held = sh > 0; c = C[ti]
            hc = np.where(held, np.fmax(hc, c), np.nan)
            pend |= held & np.isfinite(c) & (c <= (1 - stop) * hc)
    return dict(E=E, cost_f=cost_f, trade_f=trade_f, red=red, SH=SH)

def dd_series(E): m = np.maximum.accumulate(E); return E / m - 1
dates = tdays[T0:T1 + 1]; yrs = (dates[-1] - dates[0]).days / 365.25
spyE = spy_o[T0:T1 + 1] / spy_o[T0] * START_EQ
def yearly(E):
    r = pd.Series(E).pct_change().fillna(0).values; s = pd.Series(r, index=dates)
    return (1 + s).groupby(s.index.year).prod() - 1
spy_y = yearly(spyE)
mstarts = [i for i in range(len(dates)) if i == 0 or dates[i].month != dates[i - 1].month]
def roll5(E):
    w = b = 0
    for i in mstarts:
        j = dates.searchsorted(dates[i] + pd.DateOffset(years=5))
        if j >= len(dates): continue
        w += 1; b += (E[j] / E[i]) > (spyE[j] / spyE[i])
    return b / max(w, 1), w


# ---------------------------------------------------------------- 1700k selectors
spy_c = spy_df.set_index('bar_date')['close'].reindex(tdays).ffill().values.astype(np.float64)
RET = np.full_like(C, np.nan); RET[1:] = C[1:] / C[:-1] - 1
SPYR = np.full(len(tdays), np.nan); SPYR[1:] = spy_c[1:] / spy_c[:-1] - 1
ADR_RE = re.compile(r'(?<!\w)(?:ADR|ADS|N\.V\.|S\.A\.)(?!\w)|\b(?:Depositary|Limited|plc)\b|\bHoldings? Ltd\b', re.I)
adr_names = set(assets.loc[assets.name.str.contains(ADR_RE), 'symbol'])
CORR_W, BETA_W, MINV = 63, 252, 40
stat = dict(unknown_corr=0, unknown_beta=0, short_weeks=0)
info = {}   # ti -> (ids, corr matrix 100x100, valid mask, beta array)
def week_info(ti):
    """Point-in-time corr (63d) and beta (252d) of the ranked list for the rebalance at day index ti (data through ti-1)."""
    ids = rebal[ti]; pi = ti - 1
    X = RET[pi - CORR_W + 1:pi + 1][:, ids]; ok = np.isfinite(X)
    known = ok.sum(0) >= MINV
    corr = pd.DataFrame(X).corr(min_periods=MINV).values
    Y = RET[pi - BETA_W + 1:pi + 1][:, ids]; s = SPYR[pi - BETA_W + 1:pi + 1][:, None]
    v = np.isfinite(Y) & np.isfinite(s); n = v.sum(0)
    Yz = np.where(v, Y, 0.0); Sz = np.where(v, s, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        my = Yz.sum(0) / n; ms = Sz.sum(0) / n
        cov = ((Yz - my) * (Sz - ms) * v).sum(0); var = (((Sz - ms) ** 2) * v).sum(0)
        beta = np.where(n >= MINV, cov / var, np.nan)
    return ids, corr, known, beta
def place(order, known, corr, caps_fn, n=20):
    """Walk names in rank order (known first, then unknown); caps_fn(pos, chosen)->bool admits. Returns chosen positions."""
    chosen = []
    for pos in [p for p in order if known[p]] + [p for p in order if not known[p]]:
        if len(chosen) >= n: break
        if not known[pos]: stat['unknown_corr'] += 1
        if caps_fn(pos, chosen): chosen.append(pos)
    if len(chosen) < n: stat['short_weeks'] += 1
    return chosen
def select(kind, ti):
    """Selection for one cell at rebalance ti; returns list of candidate indices."""
    ids, corr, known, beta = info[ti]; order = list(range(len(ids))); ids = np.array(ids)
    if kind in ('K5', 'K6'):
        stat['unknown_beta'] += int(np.isnan(beta).sum())
        order = [p for p in order if not (np.isfinite(beta[p]) and beta[p] > 2.0)]
    if kind == 'K7':
        order = [p for p in order if cand[ids[p]] not in adr_names]
        return [ids[p] for p in order[:20]]
    if kind in ('K1', 'K2', 'K6'):
        thr = 0.60 if kind == 'K2' else 0.70
        def fn(pos, ch): return not any(np.isfinite(corr[pos, q]) and corr[pos, q] > thr for q in ch)
        return [ids[p] for p in place(order, known, corr, fn)]
    if kind in ('K3', 'K4', 'K8', 'K5'):
        if kind == 'K5': return [ids[p] for p in order[:20]]
        cap = 3 if kind == 'K4' else 4
        D = 1 - np.nan_to_num(corr, nan=0.0); D[~known, :] = 1; D[:, ~known] = 1; np.fill_diagonal(D, 0); D = np.clip((D + D.T) / 2, 0, 2)
        lab = fcluster(linkage(squareform(D, checks=False), 'average'), 0.5, 'distance')
        cnt = {}
        def fn(pos, ch):
            if cnt.get(lab[pos], 0) >= cap: return False
            cnt[lab[pos]] = cnt.get(lab[pos], 0) + 1; return True
        return [ids[p] for p in place(order, known, corr, fn)]
    return [ids[p] for p in order[:20]]
for ti in rebal: info[ti] = week_info(ti)
log.info('%s week_info built for %d rebalances', el(), len(info))

# ---------------------------------------------------------------- cells
def dd_series(E): m = np.maximum.accumulate(E); return E / m - 1
dates = tdays[T0:T1 + 1]; yrs = (dates[-1] - dates[0]).days / 365.25
spyE = spy_o[T0:T1 + 1] / spy_o[T0] * START_EQ
def yearly(E):
    r = pd.Series(E).pct_change().fillna(0).values; s = pd.Series(r, index=dates)
    return (1 + s).groupby(s.index.year).prod() - 1
spy_y = yearly(spyE)
mstarts = [i for i in range(len(dates)) if i == 0 or dates[i].month != dates[i - 1].month]
def roll5(E):
    w = b = 0
    for i in mstarts:
        j = dates.searchsorted(dates[i] + pd.DateOffset(years=5))
        if j >= len(dates): continue
        w += 1; b += (E[j] / E[i]) > (spyE[j] / spyE[i])
    return b / max(w, 1), w
cells = {'REF': ('REF', {}), 'K1_corr70': ('K1', {}), 'K2_corr60': ('K2', {}), 'K3_clu4': ('K3', {}), 'K4_clu3': ('K4', {}),
         'K5_beta2': ('K5', {}), 'K6_K1K5': ('K6', {}), 'K7_noADR': ('K7', {}), 'K8_K3T1': ('K3', dict(stop=.15))}
res = {}; meancorr = {}; cellstat = {}
for name, (kind, kw) in cells.items():
    stat.update(unknown_corr=0, unknown_beta=0, short_weeks=0)
    rb = {ti: select(kind, ti) for ti in rebal}
    cs = []
    for ti, sel in rb.items():
        ids, corr, known, beta = info[ti]; pos = [ids.index(i) for i in sel]
        if len(pos) > 1:
            m = corr[np.ix_(pos, pos)]; cs.append(np.nanmean(m[~np.eye(len(pos), dtype=bool)]))
    meancorr[name] = float(np.nanmean(cs)); cellstat[name] = dict(stat)
    r = simulate(rb=rb, **kw); res[name] = r
    log.info('%s %s end $%.0f maxDD %.1f%% meancorr %.2f stat %s', el(), name, r['E'][-1], 100 * dd_series(r['E']).min(), meancorr[name], stat)

E = res['REF']['E']; D = dd_series(E); runmax = np.maximum.accumulate(E); eps = []; i = 0
while i < len(E):
    if D[i] < 0:
        pk = int(np.where(E[:i] == runmax[i])[0][-1]) if i > 0 else 0
        j = i
        while j < len(E) and E[j] < runmax[i]: j += 1
        tr_i = pk + int(np.argmin(E[pk:j])); eps.append((pk, tr_i, j if j < len(E) else None, E[tr_i] / E[pk] - 1)); i = j
    else: i += 1
eps = sorted(eps, key=lambda x: x[3])[:5]
def ep_depths(Ec):
    out = []
    for pk, tr, rc, _ in eps:
        w = Ec[pk:(rc if rc else len(Ec))]; out.append((w / np.maximum.accumulate(w) - 1).min())
    return out
ref_cagr = (E[-1] / START_EQ) ** (1 / yrs) - 1; ref_dd = D.min(); ref_ep = ep_depths(E); rebal_set = set(rebal_dates); out = []
for name, r in res.items():
    Ec = r['E']; cagr = (Ec[-1] / START_EQ) ** (1 / yrs) - 1; mdd = dd_series(Ec).min(); y = yearly(Ec)
    epd = ep_depths(Ec); cutl = [(a - b) >= 0.25 * abs(b) for a, b in zip(epd, ref_ep)]; r5, nw = roll5(Ec)
    row = dict(cell=name, cagr=cagr, max_dd=mdd, end_usd=Ec[-1], worst_year=y.min(), worst_year_label=int(y.idxmin()), y2020=y.get(2020, np.nan),
               years_beat_spy=int((y > spy_y).sum()), n_years=len(y), roll5_share=r5, roll5_windows=nw, mean_pair_corr=meancorr[name],
               turnover_oneway_per_yr=r['trade_f'] / 2 / yrs, cost_drag_per_yr=r['cost_f'] / yrs, eps_cut=sum(cutl), eps_cut_which=''.join(str(k + 1) for k, c in enumerate(cutl) if c),
               unknown_corr_names=cellstat[name]['unknown_corr'], unknown_beta_names=cellstat[name]['unknown_beta'], weeks_under20=cellstat[name]['short_weeks'])
    for k, dpt in enumerate(epd, 1): row[f'ep{k}_depth'] = dpt
    row['pass'] = bool(name != 'REF' and (mdd - ref_dd) >= 0.10 and (ref_cagr - cagr) <= 0.05 and r5 >= 0.90 and sum(cutl) >= 3)
    out.append(row)
cdf = pd.DataFrame(out); cdf.to_csv(OUT / '1700k_cells.csv', index=False)
L = ['# RESULT 1,700k -- diversification caps on the momentum sleeve (PREREG_1700k, 8 cells + REF)', '',
     f'Engine = 1700j reconciled book (daily sim {dates[0].date()}..{dates[-1].date()}, $50K, open-to-open, Monday equal weight /20, delta costs, band cost not NBBO). '
     f'Corr 63d / beta 252d daily close-to-close through the prior trading day; names with <{MINV} valid days = unknown, placed after all known names. '
     f'Clustering: scipy average linkage on 1-corr (unknown names = singleton). Caps that cannot reach 20 names hold fewer (rest cash). SPY {(spyE[-1]/START_EQ)**(1/yrs)-1:.1%} / {dd_series(spyE).min():.1%}. '
     f'Ref episodes (peak->trough): ' + '; '.join(f'{dates[a].date()}..{dates[b].date()}' for a, b, _, _ in eps) + '.', '',
     '| cell | CAGR | max DD | end $ | ep1..5 depth | worst yr | yrs>SPY | roll5y | pair corr | turn/yr | cost/yr | cut eps | <20 wks | unk | PASS |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
def epstr(x): return ' '.join('%.0f%%' % (100 * x['ep%d_depth' % k]) for k in range(1, 6))
for _, x in cdf.iterrows():
    L.append(f"| {x['cell']} | {x['cagr']:.1%} | {x['max_dd']:.1%} | {x['end_usd']:,.0f} | {epstr(x)} | {x['worst_year']:.1%} ({x['worst_year_label']}) | {x['years_beat_spy']}/{x['n_years']} | {x['roll5_share']:.0%} | {x['mean_pair_corr']:.2f} | {x['turnover_oneway_per_yr']:.1f}x | {x['cost_drag_per_yr']:.2%} | {x['eps_cut_which'] or '-'} | {x['weeks_under20']} | {x['unknown_corr_names']}/{x['unknown_beta_names']} | {'PASS' if x['pass'] else 'no'} |")
nc = cdf[cdf.cell != 'REF'].copy(); nc['cpd'] = nc.cagr / nc.max_dd.abs(); best = nc.sort_values('cpd', ascending=False).iloc[0]
k7 = cdf[cdf.cell == 'K7_noADR'].iloc[0]; rf = cdf[cdf.cell == 'REF'].iloc[0]
L += ['', f"Pass list: {', '.join(cdf.loc[cdf['pass'], 'cell']) or 'NONE'}. Best CAGR/|DD|: {best['cell']} ({best['cagr']:.1%} / {best['max_dd']:.1%}); REF {rf['cagr']:.1%} / {rf['max_dd']:.1%}, CAGR/|DD| {rf['cagr']/abs(rf['max_dd']):.2f}.",
      f"K7 2020 return {k7['y2020']:.1%} vs REF {rf['y2020']:.1%} (cost of the ADR exclusion in the 2020 China-ADR year). ADR-named symbols in assets: {len(adr_names)}.",
      '', 'EPISODES_PLACEHOLDER']
(OUT / 'RESULT_1700k.md').write_text('\n'.join(L) + '\n')
log.info('DONE %s', el())
