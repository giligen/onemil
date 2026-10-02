#!/usr/bin/env python3
"""Cell 1,700j: drawdown frontier of the reconciled momentum sleeve (V2 risk-adjusted top-N, weekly).
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
import logging, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('/home/ec2-user/onemil/research/momentum_weekly')
logging.basicConfig(filename=str(OUT / '1700j.log'), filemode='w', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700j'); log.addHandler(logging.StreamHandler(sys.stdout))
PRICE_MIN, ADV_CUT, START_EQ = 10.0, 200_000_000.0, 50_000.0
WIN_START, WIN_END = pd.Timestamp('2017-01-01'), pd.Timestamp('2026-09-30')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|\bPREFERRED\b|\bRIGHTS?\b', re.I)
MAXN = 40
t0 = time.time()
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
spy_df = raw.loc[raw.symbol == 'SPY', ['bar_date', 'open']].sort_values('bar_date').reset_index(drop=True)
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

def simulate(n=20, stop=None, vt=None, dd=False, drift=False, keep=False):
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
            top = [i for i in rebal[ti][:n] if i not in blocked and o[i] > 0]
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

cells = {'REF': {}, 'T1_15': dict(stop=.15), 'T2_20': dict(stop=.20), 'T3_25': dict(stop=.25), 'S1_30': dict(vt=.30), 'S2_35': dict(vt=.35),
         'D1': dict(dd=True), 'N1_30': dict(n=30), 'N2_40': dict(n=40), 'C1_drift': dict(drift=True),
         'J1_T2S1': dict(stop=.20, vt=.30), 'J2_T2N1': dict(stop=.20, n=30), 'J3_T2D1': dict(stop=.20, dd=True),
         'J4_T2S1N1': dict(stop=.20, vt=.30, n=30)}
res = {}
for name, kw in cells.items():
    r = simulate(keep=(name == 'REF'), **kw); res[name] = r
    log.info('%s %s end $%.0f maxDD %.1f%%', el(), name, r['E'][-1], 100 * dd_series(r['E']).min())

# ---------------------------------------------------------------- Step 1 anatomy on REF
E = res['REF']['E']; D = dd_series(E); runmax = np.maximum.accumulate(E)
eps = []; i = 0
while i < len(E):
    if D[i] < 0:
        pk = int(np.where(E[:i] == runmax[i])[0][-1]) if i > 0 else 0
        j = i
        while j < len(E) and E[j] < runmax[i]: j += 1
        seg = slice(pk, j); tr_i = pk + int(np.argmin(E[seg])); eps.append((pk, tr_i, j if j < len(E) else None, E[tr_i] / E[pk] - 1)); i = j
    else: i += 1
eps = sorted(eps, key=lambda x: x[3])[:5]
SH = res['REF']['SH']; rows = []
for rk, (pk, tr, rc, depth) in enumerate(eps, 1):
    held = np.where(SH[pk] > 0)[0]; wts = SH[pk][held] * O[T0 + pk][held]; top = held[np.argsort(-wts)][:8]
    pnl = SH[pk:tr].astype(np.float64) * (O[T0 + pk + 1:T0 + tr + 1] - O[T0 + pk:T0 + tr]); tot = pnl.sum(); own = pnl[:, held].sum()
    spy_dd = (spy_o[T0 + pk:T0 + (rc if rc else len(E))].min() / spy_o[T0 + pk]) - 1
    rows.append(dict(rank=rk, peak=dates[pk].date(), trough=dates[tr].date(), recovery=dates[rc].date() if rc else 'none',
                     depth=depth, days_peak_trough=(dates[tr] - dates[pk]).days, days_to_recover=(dates[rc] - dates[pk]).days if rc else np.nan,
                     spy_dd_same_dates=spy_dd, loss_pct_by_peak_names=own / tot if tot else np.nan, loss_pct_by_rotation=1 - own / tot if tot else np.nan,
                     holdings_at_peak=' '.join(cand[i] for i in top), n_held=len(held)))
epdf = pd.DataFrame(rows); epdf.to_csv(OUT / '1700j_episodes.csv', index=False)
log.info('%s episodes written', el())

# ---------------------------------------------------------------- Step 2 cells table
ref_cagr = (E[-1] / START_EQ) ** (1 / yrs) - 1; ref_dd = D.min()
def ep_depths(Ec):
    out = []
    for pk, tr, rc, _ in eps:
        w = Ec[pk:(rc if rc else len(Ec))]; out.append((w / np.maximum.accumulate(w) - 1).min())
    return out
ref_ep = ep_depths(E); out = []
for name, r in res.items():
    Ec = r['E']; cagr = (Ec[-1] / START_EQ) ** (1 / yrs) - 1; mdd = dd_series(Ec).min(); y = yearly(Ec); wk = pd.Series(Ec[::1]); 
    wkr = pd.Series(Ec, index=dates)[[d in rebal_set for d in dates]].pct_change().dropna().values if (rebal_set := set(rebal_dates)) else None
    sharpe = wkr.mean() / wkr.std(ddof=1) * np.sqrt(52)
    epd = ep_depths(Ec); cut = sum((a - b) >= 0.25 * abs(b) for a, b in zip(epd, ref_ep))
    r5, nw = roll5(Ec)
    row = dict(cell=name, cagr=cagr, max_dd=mdd, end_usd=Ec[-1], worst_year=y.min(), worst_year_label=int(y.idxmin()),
               years_beat_spy=int((y > spy_y).sum()), n_years=len(y), roll5_share=r5, roll5_windows=nw, sharpe=sharpe,
               turnover_oneway_per_yr=r['trade_f'] / 2 / yrs, cost_drag_per_yr=r['cost_f'] / yrs, weeks_reduced=r['red'], eps_cut=cut)
    for k, dpt in enumerate(epd, 1): row[f'ep{k}_depth'] = dpt
    row['pass'] = bool(name != 'REF' and (mdd - ref_dd) >= 0.10 and (ref_cagr - cagr) <= 0.05 and r5 >= 0.90 and cut >= 3)
    out.append(row)
cdf = pd.DataFrame(out); cdf.to_csv(OUT / '1700j_cells.csv', index=False)

L = ['# RESULT 1,700j -- drawdown frontier of the reconciled momentum sleeve (V2 top-N weekly)', '',
     f'PREREG_1700j.md + Amendment 1. Daily sim {dates[0].date()}..{dates[-1].date()} ({yrs:.2f} y), $50K start, open-to-open, Monday equal-weight '
     'delta-trade costs (5 bps + half H-L proxy, cap 20 bps), stops on closes executed next open. Band-based cost, NOT measured NBBO; panel survivorship/adjusted-price '
     f'caveats of RECON_1700_sleeve apply. SPY: CAGR {(spyE[-1]/START_EQ)**(1/yrs)-1:.1%}, max DD {dd_series(spyE).min():.1%}. REF should reproduce ~26.6% / -42% (weekly basis).', '',
     '## Step 1 -- five deepest REF drawdowns', '', '| # | peak | trough | recovered | depth | SPY dd | loss by peak names / rotation | holdings at peak (top 8 by weight) |', '|---|---|---|---|---|---|---|---|']
for _, x in epdf.iterrows():
    L.append(f"| {x['rank']} | {x['peak']} | {x['trough']} | {x['recovery']} | {x['depth']:.1%} | {x['spy_dd_same_dates']:.1%} | {x['loss_pct_by_peak_names']:.0%} / {x['loss_pct_by_rotation']:.0%} | {x['holdings_at_peak']} |")
L += ['', 'Loss split = P&L peak->trough of names held at the peak vs names entered after (rotation). Themes: read from the holdings column.', '',
      '## Step 2 -- frontier sorted by max DD (pass = DD >=10 pts better, CAGR <=5 pts lower, rolling-5y >=90%, >=3 of 5 episodes cut by >=25% relative)', '',
      '| cell | CAGR | max DD | end $ | worst yr | yrs>SPY | roll5y | Sharpe | turn/yr | cost/yr | wks red. | ep1..5 depth | cut | PASS |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for _, x in cdf.sort_values('max_dd', ascending=False).iterrows():
    eps_s = ' '.join(f"{x[f'ep{k}_depth']:.0%}" for k in range(1, 6))
    L.append(f"| {x['cell']} | {x['cagr']:.1%} | {x['max_dd']:.1%} | {x['end_usd']:,.0f} | {x['worst_year']:.1%} ({x['worst_year_label']}) | {x['years_beat_spy']}/{x['n_years']} | {x['roll5_share']:.0%} ({x['roll5_windows']}) | {x['sharpe']:.2f} | {x['turnover_oneway_per_yr']:.1f}x | {x['cost_drag_per_yr']:.2%} | {x['weeks_reduced']} | {eps_s} | {x['eps_cut']}/5 | {'PASS' if x['pass'] else 'no'} |")
nc = cdf[cdf.cell != 'REF'].copy(); nc['cpd'] = nc.cagr / nc.max_dd.abs(); best = nc.sort_values('cpd', ascending=False).iloc[0]
L += ['', f"Pass list: {', '.join(cdf.loc[cdf['pass'], 'cell']) or 'NONE'}. Best CAGR per |max DD|: {best['cell']} ({best['cagr']:.1%} / {best['max_dd']:.1%}); REF = {cdf.cell.eq('REF').pipe(lambda m: cdf[m].cagr.iloc[0]):.1%} / {ref_dd:.1%}.",
      'Cells: 14 + REF, frozen grid, nothing added after numbers. Stop re-entry rule (blocked at the rebalance where the sale executes) and the 25%-relative "cut" definition were fixed in the script docstring before any read.']
(OUT / 'RESULT_1700j.md').write_text('\n'.join(L) + '\n')
log.info('DONE %s', el())
