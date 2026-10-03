#!/usr/bin/env python3
"""Cell 1,700x: short overlay (SPY beta / bottom-20 losers) on the GUARDED sleeve (4 cells). Reuses the 1700s panel/engine code verbatim (flat script, so copied head).
Run: bash scripts/research_run.sh -m 2500M python3 research/momentum_weekly/1700x_hedge.py"""
from __future__ import annotations
import logging, re, sys, time, itertools
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('/home/ec2-user/onemil/research/momentum_weekly')
logging.basicConfig(filename=str(OUT / '1700x.log'), filemode='w', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700x'); log.addHandler(logging.StreamHandler(sys.stdout))
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
ranked_l = {d: sub[sub.gtype == 0].nsmallest(20, 'sigV2')['symbol'].tolist() for d, sub in sr.groupby('bar_date', observed=True)}
GUARD_SET = {d: {s: {1: 'move', 2: 'gap', 3: 'move+gap'}[g] for s, g in zip(sub.symbol, sub.gtype) if g > 0} for d, sub in sr.groupby('bar_date', observed=True)}
log.info('guard flags: %d of %d eligible signal rows', int((sr.gtype > 0).sum()), len(sr))
cand = sorted({s for v in ranked_syms.values() for s in v} | {s for v in ranked_g.values() for s in v} | {s for v in ranked_l.values() for s in v}); cidx = {s: i for i, s in enumerate(cand)}
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
    log.error('GREF DOES NOT REPRODUCE -- STOP'); (OUT / 'RESULT_1700x.md').write_text(f'GREF DOES NOT REPRODUCE: CAGR {c0:.4f} DD {d0:.4f} end {E_REF[-1]:.0f}\n'); raise SystemExit(2)
ref_h1, ref_h2 = stats_of(E_REF, m1), stats_of(E_REF, m2); wk_ref = weekly(E_REF)

# ================================================================ overlay engine (1700x)
BORROW = {'beta': 0.005, 'loser': 0.03}
def simulate_overlay(kind, h, n_short=20):
    """Long leg identical to simulate(ranked_g); plus a short book reset at each Monday open to h x long equity.
    kind 'beta': short SPY (2 bp per traded $); 'loser': equal-weight short of ranked_l[:20] (band RATE). Proceeds earn nothing; borrow accrued daily
    on the short book's market value; mark at daily open like the engine. Returns combined equity, long-only equity, cost and borrow dollars."""
    ns = len(cand); sh = np.zeros(ns); cash = START_EQ; nd = T1 - T0 + 1; E = np.zeros(nd); EL = np.zeros(nd)
    ss = np.zeros(ns); sspy = 0.0; scash = 0.0; scost = sborrow = 0.0
    rk = {didx[x]: [cidx[s] for s in ranked_g[prior[x]]] for x in DAYMAP[0] if prior[x] in ranked_g and didx[x] <= T1}
    rl = {didx[x]: [cidx[s] for s in ranked_l[prior[x]]] for x in DAYMAP[0] if prior[x] in ranked_l and didx[x] <= T1}
    for k in range(nd):
        ti = T0 + k; o = O[ti]
        if ti in rk:
            eq = cash + sh @ o; top = [i for i in rk[ti][:20] if o[i] > 0]; tgt = np.zeros(ns); tgt[top] = eq / 20; delta = tgt - sh * o
            for i in np.where(delta < -1e-9)[0]:
                v = -delta[i]; c = RATE[ti, i] * v; cash += v - c; sh[i] -= v / o[i]
                if sh[i] <= 1e-12: sh[i] = 0
            for i in np.where(delta > 1e-9)[0]:
                v = delta[i]; c = RATE[ti, i] * v; cash -= v + c; sh[i] += v / o[i]
            target_short = h * (cash + sh @ o)                      # long equity AFTER its own costs, at the Monday open
            if kind == 'beta':
                tv = target_short; cur = sspy * spy_o[ti]; d = tv - cur; c = 0.0002 * abs(d); scash += d - c; sspy += d / spy_o[ti]; scost += c
            else:
                shorts = [i for i in rl.get(ti, [])[:n_short] if o[i] > 0]; tgt = np.zeros(ns)
                if shorts: tgt[shorts] = target_short / len(shorts)
                dl = tgt - ss * o
                for i in np.where(np.abs(dl) > 1e-9)[0]:
                    v = abs(dl[i]); c = RATE[ti, i] * v; scost += c; scash += (v if dl[i] > 0 else -v) - c; ss[i] += dl[i] / o[i]   # dl>0 = sell more short
                    if ss[i] <= 1e-12: ss[i] = 0
        sval = sspy * spy_o[ti] if kind == 'beta' else ss @ o
        if k > 0: b = BORROW[kind] / 252 * sval; scash -= b; sborrow += b
        EL[k] = cash + sh @ o; E[k] = EL[k] + scash - sval
    return dict(E=E, EL=EL, cost=scost, borrow=sborrow)

# long-leg reproduction against the stored guard curve BEFORE any overlay
cur = pd.read_csv(OUT / '1700u_curves_daily.csv', index_col=0, parse_dates=True)['guard']
r0_ = simulate_overlay('beta', 0.0); EL0 = r0_['EL']
say(f'long-leg engine end {EL0[-1]:,.0f} vs simulate {E_REF[-1]:,.0f} (max abs diff {np.abs(EL0-E_REF).max():.6f}); stored guard curve end {cur.iloc[-1]:,.0f} DD {dd_series(cur.values).min():.4%} vs {d0:.4%}')
if abs(EL0[-1] / E_REF[-1] - 1) > 0.005 or abs(cur.iloc[-1] / E_REF[-1] - 1) > 0.005 or abs(dd_series(cur.values).min() - d0) > 0.005 * abs(d0):
    log.error('LONG LEG DOES NOT REPRODUCE -- STOP'); (OUT / 'RESULT_1700x.md').write_text('LONG LEG DOES NOT REPRODUCE\n'); raise SystemExit(3)
runmax = np.maximum.accumulate(E_REF); eps = []; i = 0
while i < len(E_REF):
    if E_REF[i] < runmax[i]:
        pk = int(np.where(E_REF[:i] == runmax[i])[0][-1]) if i > 0 else 0; j = i
        while j < len(E_REF) and E_REF[j] < runmax[i]: j += 1
        tr_i = pk + int(np.argmin(E_REF[pk:j])); eps.append((pk, tr_i, j if j < len(E_REF) else None, E_REF[tr_i] / E_REF[pk] - 1)); i = j
    else: i += 1
eps = sorted(eps, key=lambda x: x[3])[:3]
def ep_depths(Ec): return [(Ec[pk:(rc if rc else len(Ec))] / np.maximum.accumulate(Ec[pk:(rc if rc else len(Ec))]) - 1).min() for pk, tr, rc, _ in eps]
ref_ep = ep_depths(E_REF); say('GREF episodes', [(str(dates[a].date()), str(dates[b].date()), round(dp, 4)) for a, b, _, dp in eps])
worst10 = np.argsort(wk_ref)[:10]; wk_ref_worst = wk_ref.min()
def row_of(Ec): 
    w = weekly(Ec); c, dd, ra = stats_of(Ec); return c, dd, ra, w
rows = []; curves = {'GREF': E_REF}; res = {}
for kind in ('beta', 'loser'):
    for h in (0.25, 0.50):
        name = f'X-{kind}|h{int(h*100)}'; r = simulate_overlay(kind, h); Ec = r['E']; res[name] = r; curves[name] = Ec
        c, dd, ra, w = row_of(Ec); _, _, ra1 = stats_of(Ec, m1); _, _, ra2 = stats_of(Ec, m2)
        ov = Ec - r['EL']; ovw = (ov[widx][1:] - ov[widx][:-1]) / E_REF[widx][:-1]
        cost_yr = (r['cost'] + r['borrow']) / yrs / float(np.mean(Ec))
        ok = (dd - d0 >= 0.08) and (ra >= r0 + 0.10) and (ra1 >= ref_h1[2]) and (ra2 >= ref_h2[2]) and (w.min() >= wk_ref_worst - 0.02)
        row = dict(cell=name, h=h, cagr=c, maxdd=dd, ratio=ra, worst_week=w.min(), weekly_p10=float(np.quantile(w, 0.10)), green_share=float((w > 0).mean()))
        for k_, dp in enumerate(ep_depths(Ec), 1): row[f'ep{k_}'] = dp
        row.update(overlay_mean_in_gref_worst10=float(ovw[worst10].mean()), h1_ratio=ra1, h2_ratio=ra2, overlay_cost_yr=cost_yr, pass_=bool(ok),
                   overlay_worst_week=float(ovw.min()), end_usd=Ec[-1], overlay_cost_usd=r['cost'], overlay_borrow_usd=r['borrow'])
        rows.append(row); say(f'{name}: CAGR {c:.2%} DD {dd:.1%} ratio {ra:.2f} (GREF {c0:.2%}/{d0:.1%}/{r0:.2f}) h1 {ra1:.2f} h2 {ra2:.2f} worst wk {w.min():.2%} ovl worst10 {ovw[worst10].mean():.2%} ovl worst wk {ovw.min():.2%} cost+borrow/yr {cost_yr:.2%} pass={ok}')
cdf = pd.DataFrame(rows).rename(columns={'pass_': 'pass'}); cdf.to_csv(OUT / '1700x_cells.csv', index=False)
pd.DataFrame(curves, index=dates).to_csv(OUT / '1700x_curves_daily.csv')
# momentum-crash months
mon = {k: pd.Series(v, index=dates).resample('ME').last().pct_change() for k, v in curves.items() if k in ('GREF', 'X-loser|h25', 'X-loser|h50')}
mt = pd.DataFrame(mon); mt = pd.concat([mt['2020-03':'2020-07'], mt['2026-03':'2026-05']]); say((mt * 100).round(1).to_string())
L = ['# RESULT 1,700x -- short overlay on the guarded sleeve (PREREG_1700x.md, FROZEN)', '',
     f'Long-leg reproduction: engine end ${EL0[-1]:,.0f} = simulate(ranked_g) ${E_REF[-1]:,.0f}; stored 1700u guard curve end ${cur.iloc[-1]:,.0f}, DD {dd_series(cur.values).min():.2%} vs {d0:.2%}. PASS (<0.5 %).',
     f'GREF: CAGR {c0:.2%} / DD {d0:.1%} / ratio {r0:.2f}; halves {ref_h1[2]:.2f} / {ref_h2[2]:.2f}; worst week {wk_ref_worst:.2%}; episodes ' + '; '.join(f'{dates[a].date()}..{dates[b].date()} {dp:.1%}' for a, b, _, dp in eps) + '.', '',
     '| cell | CAGR | DD | ratio | h1 | h2 | worst wk | P10 | green | ep1/2/3 | ovl in GREF worst10 | ovl worst wk | cost+borrow /yr | pass |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for _, x in cdf.iterrows():
    L.append(f"| {x.cell} | {x.cagr:.1%} | {x.maxdd:.1%} | {x.ratio:.2f} | {x.h1_ratio:.2f} | {x.h2_ratio:.2f} | {x.worst_week:.1%} | {x.weekly_p10:.1%} | {x.green_share:.0%} | {x.ep1:.0%}/{x.ep2:.0%}/{x.ep3:.0%} | {x.overlay_mean_in_gref_worst10:+.2%} | {x.overlay_worst_week:+.1%} | {x.overlay_cost_yr:.2%} | {'Y' if x['pass'] else 'N'} |")
L += ['', 'Monthly returns % (GREF | loser h25 | loser h50):']
for d_, x in mt.iterrows(): L.append(f"  {d_:%Y-%m}: {x['GREF']*100:+.1f} | {x['X-loser|h25']*100:+.1f} | {x['X-loser|h50']*100:+.1f}")
L += ['', 'Rule: DD better >= 8 pt AND ratio >= GREF+0.10 AND both halves >= GREF half AND worst week no worse than 2 pt. ' + ('RECOMMENDED: ' + ', '.join(cdf[cdf['pass']].cell) if cdf['pass'].any() else 'NO cell passes: short overlay CLOSED as a DD repair at these ratios.'),
      'Adversary caveats: (1) delisted losers carry a stale ffilled open (no delisting squeeze/halt gap); short squeeze, recall and hard-to-borrow costs beyond 3 %/yr are not modelled. (2) Marks at daily opens, borrow on 252-day year. (3) Costs are the band model, not NBBO; the book is re-sized (delta) each Monday. (4) 4 cells, one window, Oct-2026 knowledge of the crash months. (5) Proceeds earn 0, no margin interest/limits modelled.']
(OUT / 'RESULT_1700x.md').write_text('\n'.join(L) + '\n'); log.info('DONE %s', el()); say('DONE')
