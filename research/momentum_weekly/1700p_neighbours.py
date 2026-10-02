#!/usr/bin/env python3
"""Cell 1,700p: neighbours of the mid-week leads (PREREG_1700p.md, FROZEN). Reuses the 1700l lean loader and daily engine, parameterised."""
from __future__ import annotations
import logging, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('/home/ec2-user/onemil/research/momentum_weekly')
logging.basicConfig(filename=str(OUT / '1700p.log'), filemode='w', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700p'); log.addHandler(logging.StreamHandler(sys.stdout))
PRICE_MIN, ADV_CUT, START_EQ = 10.0, 200_000_000.0, 50_000.0
WIN_START, WIN_END = pd.Timestamp('2017-01-01'), pd.Timestamp('2026-09-30')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|\bPREFERRED\b|\bRIGHTS?\b', re.I)
MAXN = 40
t0 = time.time()
def el(): return f'{time.time()-t0:5.0f}s'

# ---------------------------------------------------------------- panel + signals (as 1700g)
cols = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']
import pyarrow.parquet as pq, pyarrow as pa
tbl = pq.read_table(OUT / 'panel_2016_2026.parquet', columns=cols, read_dictionary=['symbol'])
tbl = tbl.cast(pa.schema([('symbol', tbl.schema.field('symbol').type), ('bar_date', tbl.schema.field('bar_date').type)] + [(c, pa.float32()) for c in ('open', 'high', 'low', 'close', 'volume')]))
raw = tbl.to_pandas(self_destruct=True); del tbl
raw['bar_date'] = pd.to_datetime(raw['bar_date'])
raw = raw.sort_values(['symbol', 'bar_date'], kind='stable').reset_index(drop=True)
dup_next = ((raw.symbol.values.codes[1:] == raw.symbol.values.codes[:-1]) & (raw.bar_date.values[1:] == raw.bar_date.values[:-1]))
raw = raw[~np.append(dup_next, False)]
raw = raw[~((raw.open <= 0) | (raw.high <= 0) | (raw.low <= 0) | (raw.close <= 0))].reset_index(drop=True)
spy_df = raw.loc[raw.symbol == 'SPY', ['bar_date', 'open']].sort_values('bar_date').reset_index(drop=True)
if spy_df.empty: log.error('SPY missing'); raise SystemExit(1)
tdays = pd.DatetimeIndex(sorted(spy_df.bar_date.unique())); didx = {d: i for i, d in enumerate(tdays)}
assets = pd.read_csv(OUT / '1700c_assets.csv', dtype={'symbol': str, 'name': str}); assets['name'] = assets['name'].fillna('')
excl = set(assets.loc[assets.name.str.contains(NAME_RE), 'symbol']) | {s for s in raw.symbol.unique() if TEST_RE.match(s)}
panel = raw[(raw.symbol != 'SPY') & ~raw.symbol.isin(excl)].reset_index(drop=True); del raw
panel['symbol'] = panel.symbol.cat.remove_unused_categories()
panel['spread'] = (((panel.high - panel.low) / panel.close).clip(lower=0) * 0.1).clip(upper=0.002)
dv = (panel.close * panel.volume).values; cl = panel.close.values; codes = panel.symbol.values.codes
panel = panel.drop(columns=['high', 'low', 'volume'])
N = len(panel); adv20 = np.full(N, np.nan); sigV2 = np.full(N, np.nan); okv = np.zeros(N, bool)
bounds = np.flatnonzero(np.diff(codes)) + 1; starts = np.r_[0, bounds]; ends = np.r_[bounds, N]
for a, b in zip(starts, ends):                       # per-symbol pandas rolling (same semantics as the grouped version, low memory)
    c = pd.Series(cl[a:b]); adv20[a:b] = pd.Series(dv[a:b]).rolling(20, min_periods=20).mean().values
    c21, c252, c273 = c.shift(21), c.shift(252), c.shift(273)
    ret = c.pct_change(); vol = ret.rolling(252, min_periods=252).std().values
    sig = (c21 / c252 - 1).values
    with np.errstate(divide='ignore', invalid='ignore'): sigV2[a:b] = np.where(vol > 0, sig / vol, np.nan)
    okv[a:b] = c273.notna().values
panel['adv20'] = adv20; panel['sigV2'] = sigV2; panel['ok'] = okv; del dv, cl, adv20, sigV2, okv
log.info('%s signals built (%d rows)', el(), len(panel))

# weekly calendar: decision day per (week, weekday target); signal = prior trading day's close
per = pd.Series(tdays).dt.to_period('W'); first = pd.Series(tdays).groupby(per).min().sort_index()
ent = first[(first >= WIN_START) & (first <= WIN_END)].reset_index(drop=True)
prior = {d: tdays[didx[d] - 1] for d in ent if didx[d] > 0}
wk_days = pd.Series(tdays).groupby(per).apply(list)
DAYMAP = {w: {} for w in range(5)}          # DAYMAP[target weekday][decision day] = Monday (week key)
for d in ent:
    days = [x for x in wk_days[d.to_period('W')] if x >= d]
    for w in range(5):
        c = [x for x in days if x.weekday() >= w]
        if c: DAYMAP[w][c[0]] = d
        elif w >= 3 and len(days) >= 2: DAYMAP[w][days[-1]] = d
extra = {x for dct in DAYMAP.values() for x in dct if didx[x] > 0}
prior.update({x: tdays[didx[x] - 1] for x in extra})
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

# ---------------------------------------------------------------- simulation (1700l engine, parameterised)
def simulate(g=None, replace=True, barweeks=0, day=0, tm='open', keep=False):
    """One cell. g = gap-down exit threshold (None = off); replace = vacated slot refilled by next-ranked name the same morning (else cash to Monday);
    barweeks = 0 re-entry allowed at next Monday reset, 4 = barred 28 calendar days; day/tm = rebalance weekday (0=Mon) and open/close.
    Returns E (daily open-valued equity), cost, turnover, exits list (name, ti)."""
    n = 20; ns = len(cand); sh = np.zeros(ns); cash = START_EQ; hc = None
    nd = T1 - T0 + 1; E = np.zeros(nd); cost_f = trade_f = 0.0; exits = []; bar = {}
    rk = {didx[x]: [cidx[s] for s in ranked_syms[prior[x]]] for x in DAYMAP[day] if prior[x] in ranked_syms and didx[x] <= T1}
    open_rk, close_rk = (rk, {}) if tm == 'open' else ({}, rk)
    fill = g is not None and replace; cur_rank = []; sold_wk = set()
    for k in range(nd):
        ti = T0 + k; o = O[ti]; pre = cash + sh @ o; tr = cs = 0.0
        if ti in rebal: cur_rank = rebal[ti]; sold_wk = set()
        is_reset = ti in open_rk

        def sell(i, price):
            nonlocal cash, tr, cs
            v = sh[i] * price[i]; c = RATE[ti, i] * v; cash += v - c; sh[i] = 0; tr += v; cs += c; return v - c
        def buy_value(i, amount, price):
            nonlocal cash, tr, cs
            v = amount / (1 + RATE[ti, i]); c = RATE[ti, i] * v; cash -= v + c; sh[i] += v / price[i]; tr += v; cs += c
        def do_replace(proceeds):
            for i in cur_rank:
                if sh[i] == 0 and i not in sold_wk and bar.get(i, -1) < ti and o[i] > 0 and not (C[ti - 1, i] > 0 and o[i] <= (1 - g) * C[ti - 1, i]):
                    buy_value(i, proceeds, o); return
        if g is not None and not is_reset and ti not in close_rk:
            for i in np.where((sh > 0) & (C[ti - 1] > 0) & (o <= (1 - g) * C[ti - 1]))[0]:
                p = sell(i, o); sold_wk.add(i); exits.append((int(i), ti))
                if barweeks: bar[int(i)] = ti + 20
                if replace: do_replace(p)
        def reset(price, rkl, excl):
            nonlocal cash, tr, cs
            eq = cash + sh @ price
            if fill: top = [i for i in rkl if i not in excl and price[i] > 0][:n]
            else: top = [i for i in rkl[:n] if i not in excl and price[i] > 0]
            tgt = np.zeros(ns); tgt[top] = eq / n; delta = tgt - sh * price
            for i in np.where(delta < -1e-9)[0]:
                v = -delta[i]; c = RATE[ti, i] * v; cash += v - c; sh[i] -= v / price[i]; tr += v; cs += c
                if sh[i] <= 1e-12: sh[i] = 0
            for i in np.where(delta > 1e-9)[0]:
                v = delta[i]; c = RATE[ti, i] * v; cash -= v + c; sh[i] += v / price[i]; tr += v; cs += c
        barred = {i for i, t in bar.items() if t >= ti}
        if is_reset: reset(o, open_rk[ti], barred)
        E[k] = cash + sh @ o
        cost_f += cs / pre if pre > 0 else 0; trade_f += tr / pre if pre > 0 else 0
        if ti in close_rk:
            pre_c = cash + sh @ CX[ti]; tr = cs = 0.0; reset(CX[ti], close_rk[ti], barred)
            cost_f += cs / pre_c if pre_c > 0 else 0; trade_f += tr / pre_c if pre_c > 0 else 0
    return dict(E=E, cost_f=cost_f, trade_f=trade_f, exits=exits)

def dd_series(E): m = np.maximum.accumulate(E); return E / m - 1
dates = tdays[T0:T1 + 1]; yrs = (dates[-1] - dates[0]).days / 365.25
def cg(E): return (E[-1] / START_EQ) ** (1 / yrs) - 1
CX = np.where(np.isnan(C), O, C)

cells = {'REF': {}}
for g_ in (.04, .06, .08, .10, .12):
    for rp in (True, False):
        for bw in (0, 4): cells[f'G_g{int(g_*100)}_{"repl" if rp else "cash"}_{"allow" if bw == 0 else "bar4"}'] = dict(g=g_, replace=rp, barweeks=bw)
DN = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
for dy in range(5):
    for t_ in ('open', 'close'): cells[f'D_{DN[dy]}_{t_}'] = dict(day=dy, tm=t_)
res = {}
for name, kw in cells.items():
    res[name] = simulate(**kw); E = res[name]['E']
    log.info('%s %s end $%.0f CAGR %.2f%% maxDD %.1f%% exits %d', el(), name, E[-1], 100 * cg(E), 100 * dd_series(E).min(), len(res[name]['exits']))
def chk(name, c, d, e):
    E = res[name]['E']; ok = abs(100 * cg(E) - c) <= .2 and abs(100 * dd_series(E).min() - d) <= .2 and abs(E[-1] - e) <= .01 * e
    log.info('REPRO %s %s (CAGR %.2f DD %.2f end %.0f vs %.2f %.2f %.0f)', name, 'OK' if ok else 'FAIL', 100 * cg(E), 100 * dd_series(E).min(), E[-1], c, d, e); return ok
oks = [chk('REF', 27.18, -44.5, 507823), chk('G_g8_repl_allow', 28.5, -40.1, 562131), chk('D_Wed_open', 29.3, -41.1, 596242), chk('D_Mon_close', 27.2, -43.6, 510091)]
if not all(oks): log.error('REPRODUCTION FAILED -- STOP'); raise SystemExit(2)

# ---------------------------------------------------------------- reads
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
def half_ratio(Ec, lo, hi):
    m = (dates >= lo) & (dates < hi); s = Ec[m]; y = (dates[m][-1] - dates[m][0]).days / 365.25
    return ((s[-1] / s[0]) ** (1 / y) - 1) / abs(dd_series(s).min())
H = [(pd.Timestamp('2017-01-01'), pd.Timestamp('2022-01-01')), (pd.Timestamp('2022-01-01'), pd.Timestamp('2027-01-01'))]
ref_h = [half_ratio(E, *h) for h in H]; ref_cagr, ref_dd = cg(E), D.min()
out = []
for name, r in res.items():
    Ec = r['E']; cagr = cg(Ec); mdd = dd_series(Ec).min(); wk = pd.Series(Ec[widx]).pct_change().dropna().values
    dif = wk - wk_ref; keepm = dif <= np.quantile(dif, 0.95); sd = dif.std(ddof=1)
    row = dict(cell=name, family=name[0], cagr=cagr, max_dd=mdd, ratio=cagr / abs(mdd), end_usd=Ec[-1], sharpe=wk.mean() / wk.std(ddof=1) * np.sqrt(52),
               paired_mean=dif.mean(), paired_t=dif.mean() / (sd / np.sqrt(len(dif))) if sd > 0 else np.nan, paired_ex_top5=dif[keepm].mean(),
               half1_ratio=half_ratio(Ec, *H[0]), half2_ratio=half_ratio(Ec, *H[1]), half1_ref=ref_h[0], half2_ref=ref_h[1],
               turnover_oneway_per_yr=r['trade_f'] / 2 / yrs, cost_drag_per_yr=r['cost_f'] / yrs, n_exits=len(r['exits']))
    for k_, dpt in enumerate(ep_depths(Ec), 1): row[f'ep{k_}_depth'] = dpt
    ex = r['exits']; lo1 = lo4 = n1 = n4 = 0
    for i_, ti in ex:
        if ti + 20 < len(O) and O[ti, i_] > 0 and O[ti + 5, i_] > 0: n1 += 1; lo1 += O[ti + 5, i_] < O[ti, i_]
        if ti + 20 < len(O) and O[ti, i_] > 0 and O[ti + 20, i_] > 0: n4 += 1; lo4 += O[ti + 20, i_] < O[ti, i_]
    row['lower_1w'] = lo1 / n1 if n1 else np.nan; row['lower_4w'] = lo4 / n4 if n4 else np.nan
    row['better_both'] = bool(name != 'REF' and cagr > ref_cagr and mdd > ref_dd)
    row['halves_better'] = bool(name != 'REF' and row['half1_ratio'] > ref_h[0] and row['half2_ratio'] > ref_h[1])
    out.append(row)
cdf = pd.DataFrame(out); cdf.to_csv(OUT / '1700p_cells.csv', index=False)

def verdict(fam):
    f = cdf[(cdf.family == fam) & (cdf.cell != 'REF')]; nn = len(f)
    a = int(f.better_both.sum()); b = int(f.halves_better.sum())
    ok = a >= 0.75 * nn and b >= 0.75 * nn
    c = None
    if fam == 'G':
        c = float((f.lower_4w * f.n_exits).sum() / f.n_exits.sum()); ok = ok and c >= 0.55
    med = f.sort_values('ratio').iloc[len(f) // 2]
    return nn, a, b, c, ok, med
vG, vD = verdict('G'), verdict('D')
def fmt(x):
    ep = ' '.join(f"{x[f'ep{k}_depth']:.0%}" for k in range(1, 6))
    return (f"| {x['cell']} | {x.cagr:.1%} | {x.max_dd:.1%} | {x.ratio:.2f} | {x.end_usd:,.0f} | {x.sharpe:.2f} | {ep} | {x.half1_ratio/x.half1_ref:.2f}/{x.half2_ratio/x.half2_ref:.2f} | "
            f"{x.paired_mean*100:+.3f}% ({x.paired_t:.1f}) {x.paired_ex_top5*100:+.3f}% |")
L = ['# RESULT 1,700p -- are the mid-week leads real? (PREREG_1700p.md, frozen)', '',
     'Engine = 1700l (band cost, daily opens). Repro: REF, G_g8_repl_allow=M5, D_Wed_open=M4b, D_Mon_close=M4a all within 0.2 pt (see 1700p.log). Halves = ratio vs REF ratio (>1 improves). Paired = weekly diff vs REF mean (t) ex-top-5%.', '']
hdrG = '| cell | CAGR | max DD | ratio | end $ | Sharpe | ep1..5 | halves ratio/REF | paired mean (t) ex5% | exits | lower 1w/4w |'
hdrD = hdrG.replace(' | exits | lower 1w/4w |', ' |')
L += ['## Family G (gap-down exit, 20 cells)', '', hdrG, '|' + '---|' * 11]
for _, x in cdf[cdf.family.isin(['R', 'G'])].iterrows(): L.append(fmt(x) + f" {x.n_exits} | {x.lower_1w:.0%}/{x.lower_4w:.0%} |")
L += ['', '## Family D (rebalance day x time, 10 cells; Mon open = REF)', '', hdrD, '|' + '---|' * 9]
for _, x in cdf[cdf.family == 'D'].iterrows(): L.append(fmt(x))
for fam, v in (('G', vG), ('D', vD)):
    nn, a, b, c, ok, med = v
    L += ['', f"Family {fam}: both CAGR and DD better {a}/{nn} (need {int(np.ceil(.75*nn))}); both halves' ratio better {b}/{nn}" + (f"; sold names lower 4w {c:.1%} of exits (need 55%)" if c is not None else '') +
          f" -> **{'REAL' if ok else 'FAVOURABLE DRAW'}**. Median cell by ratio: {med['cell']} {med.cagr:.1%} / {med.max_dd:.1%} / ${med.end_usd:,.0f}."]
if vG[4] and vD[4]:
    mg, md = vG[5], vD[5]; kw = dict(cells[mg['cell']]); kw.update(cells[md['cell']]); rj = simulate(**kw); Ej = rj['E']
    L += ['', f"Joint ({mg['cell']} + {md['cell']}): {cg(Ej):.1%} / {dd_series(Ej).min():.1%} / ${Ej[-1]:,.0f}."]; log.info('joint run')
else: L += ['', 'Joint cell not run (not both families real).']
(OUT / 'RESULT_1700p.md').write_text('\n'.join(L) + '\n'); log.info('DONE %s', el())
