#!/usr/bin/env python3
"""Cell 1,700l: mid-week actions on the reconciled momentum sleeve (PREREG_1700l.md, FROZEN). Cells M1,M1b,M2,M3a,M3b,M4a,M4b,M4c,M5,M6
on the 1700j daily engine (panel/signals/cost copied from 1700j_frontier.py). Conventions fixed here (PREREG silent): a stop/event/gap
executes at the open; on a Monday reset day pending stops are executed and the name is blocked from re-buy at that reset (as 1700j);
M1/M5/M3b fill vacated slots with the next-ranked name from Monday's top-40; M3a leaves the slot in cash until next Monday; M6 adds once per name
per week, funded pro rata from the other names (delta costs both legs); M4 trades use the prior trading day's signal (REF convention) at the
open (M4b) or at the close (M4a Monday, M4c Friday); all cells valued at daily opens, weekly returns sampled at Monday opens."""
from __future__ import annotations
import logging, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('/home/ec2-user/onemil/research/momentum_weekly')
logging.basicConfig(filename=str(OUT / '1700l.log'), filemode='w', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700l'); log.addHandler(logging.StreamHandler(sys.stdout))
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
tbl = pq.read_table(OUT / 'panel_2016_2026.parquet', columns=cols)
tbl = tbl.cast(pa.schema([('symbol', tbl.schema.field('symbol').type), ('bar_date', tbl.schema.field('bar_date').type)] + [(c, pa.float32()) for c in ('open', 'high', 'low', 'close', 'volume')]))
raw = tbl.to_pandas(self_destruct=True); del tbl
raw['symbol'] = raw['symbol'].astype('category'); raw['bar_date'] = pd.to_datetime(raw['bar_date'])
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

# weekly calendar (first trading day of each week), prior-day signal
per = pd.Series(tdays).dt.to_period('W'); first = pd.Series(tdays).groupby(per).min().sort_index()
ent = first[(first >= WIN_START) & (first <= WIN_END)].reset_index(drop=True)
prior = {d: tdays[didx[d] - 1] for d in ent if didx[d] > 0}
# extra decision days per week for M2 (Thu), M4b (Wed), M4c (Fri)
wk_days = pd.Series(tdays).groupby(per).apply(list)
WED, THU, FRI = {}, {}, {}
for d in ent:
    days = [x for x in wk_days[d.to_period('W')] if x >= d]
    if len(days) < 2: continue
    w = [x for x in days[1:] if x.weekday() >= 2]; th = [x for x in days[1:] if x.weekday() >= 3]
    if w: WED[w[0]] = d
    if th: THU[th[0]] = d
    FRI[days[-1]] = d
extra = {x for dct in (WED, THU, FRI) for x in dct if didx[x] > 0}
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


# ---------------------------------------------------------------- earnings calendar (EDGAR 8-K item 2.02)
nd_all = len(tdays); EVM = np.zeros((nd_all, len(cand)), bool); EV_FILING = {}
parts = []
for ch in pd.read_csv('/home/ec2-user/onemil/research/edgar_desk/events_raw.csv', usecols=['symbol', 'form', 'filing_date', 'acceptance_datetime', 'items'],
                      dtype=str, chunksize=500_000):
    ch = ch[ch.form.isin(['8-K', '8-K/A']) & ch['items'].fillna('').str.contains(r'(?:^|;)2\.02(?:;|$)') & (ch.filing_date >= '2016-06-01')]
    parts.append(ch[['symbol', 'filing_date', 'acceptance_datetime']])
ev = pd.concat(parts, ignore_index=True); log.info('%s 2.02 events %d, first filing %s', el(), len(ev), ev.filing_date.min())
acc = pd.to_datetime(ev.acceptance_datetime, utc=True).dt.tz_convert('America/New_York')
pre_open = (acc.dt.hour * 60 + acc.dt.minute) <= 9 * 60 + 15
base = acc.dt.tz_localize(None).dt.normalize()
tds = tdays.values
sess = np.where(pre_open, np.searchsorted(tds, base.values, 'left'), np.searchsorted(tds, base.values, 'right'))
ev['sess'] = sess; ev = ev[(ev.sess < nd_all) & ev.symbol.isin(cidx)]
for s, k in zip(ev.symbol, ev.sess): EVM[k, cidx[s]] = True
EVDATES = {s: np.sort(tdays.values[sub_.sess.values]) for s, sub_ in ev.groupby('symbol')}
log.info('%s event matrix %d (name,session) pairs on candidates', el(), int(EVM.sum()))
CX = np.where(np.isnan(C), O, C)
Cdf = pd.DataFrame(C); NEWH = (Cdf > Cdf.shift(1).rolling(20, min_periods=20).max()).values
GAP_THR = 0.08

# ---------------------------------------------------------------- generalized simulation
def simulate(mode='REF', stop=None, keep=False):
    """One cell on the daily engine. mode in REF,M1,M2,M3a,M3b,M4a,M4b,M4c,M5,M6 (stop = M1 trailing-close threshold)."""
    n = 20; ns = len(cand); sh = np.zeros(ns); cash = START_EQ; hc = np.full(ns, np.nan); pend = np.zeros(ns, bool)
    nd = T1 - T0 + 1; E = np.zeros(nd); cost_f = trade_f = 0.0; SH = np.zeros((nd, ns), np.float32) if keep else None
    open_rk, close_rk = {}, {}
    if mode == 'M4a': close_rk = dict(rebal)
    elif mode == 'M4b': open_rk = {didx[x]: [cidx[s] for s in ranked_syms[prior[x]]] for x in WED if prior[x] in ranked_syms and didx[x] <= T1}
    elif mode == 'M4c': close_rk = {didx[x]: [cidx[s] for s in ranked_syms[prior[x]]] for x in FRI if prior[x] in ranked_syms and didx[x] <= T1}
    else: open_rk = dict(rebal)
    if mode == 'M2': open_rk.update({didx[x]: [cidx[s] for s in ranked_syms[prior[x]]] for x in THU if prior[x] in ranked_syms and didx[x] <= T1})
    fill = mode in ('M1', 'M5', 'M3b'); cur_rank = []; sold_wk = set(); added_wk = set(); mons = sorted(rebal)
    for k in range(nd):
        ti = T0 + k; o = O[ti]; pre = cash + sh @ o; tr = cs = 0.0
        if ti in rebal: cur_rank = rebal[ti]; sold_wk = set(); added_wk = set()
        is_reset = ti in open_rk; blocked = set()

        def sell(i, price):
            nonlocal cash, tr, cs
            v = sh[i] * price[i]; c = RATE[ti, i] * v; cash += v - c; sh[i] = 0; hc[i] = np.nan; tr += v; cs += c; return v - c
        def buy_value(i, amount, price):
            nonlocal cash, tr, cs
            v = amount / (1 + RATE[ti, i]); c = RATE[ti, i] * v; cash -= v + c; sh[i] += v / price[i]; tr += v; cs += c
        def replace(proceeds, gapchk):
            for i in cur_rank:
                if sh[i] == 0 and i not in sold_wk and o[i] > 0 and not (gapchk and C[ti - 1, i] > 0 and o[i] <= (1 - GAP_THR) * C[ti - 1, i]):
                    buy_value(i, proceeds, o); hc[i] = np.nan; return
        if mode in ('M1',) and pend.any():                       # executed stops
            for i in np.where(pend)[0]:
                p = sell(i, o); sold_wk.add(i); blocked.add(i)
                if not is_reset: replace(p, False)
            pend[:] = False
        if mode == 'M5' and not is_reset:                        # gap-down exits
            for i in np.where((sh > 0) & (C[ti - 1] > 0) & (o <= (1 - GAP_THR) * C[ti - 1]))[0]:
                p = sell(i, o); sold_wk.add(i); replace(p, True)
        if mode == 'M3a' and not is_reset:                       # sell at event-session open, cash until Monday
            for i in np.where((sh > 0) & EVM[ti])[0]: sell(i, o)
        if mode == 'M6' and not is_reset:                        # press the winners
            A = [i for i in np.where((sh > 0) & NEWH[ti - 1])[0] if i not in added_wk]
            if A:
                eq = cash + sh @ o; others = [i for i in np.where(sh > 0)[0] if i not in A]; ov = sum(sh[i] * o[i] for i in others); tot = 0.025 * eq * len(A)
                if others and ov > tot:
                    f = tot / ov
                    for i in others:
                        v = sh[i] * o[i] * f; c = RATE[ti, i] * v; cash += v - c; sh[i] -= v / o[i]; tr += v; cs += c
                    for i in A: buy_value(i, tot / len(A) * 1.0, o); added_wk.add(i)
        def reset(price, rk, blk, excl):
            nonlocal cash, tr, cs
            eq = cash + sh @ price
            if fill: top = [i for i in rk if i not in blk and i not in excl and price[i] > 0][:n]
            else: top = [i for i in rk[:n] if i not in blk and i not in excl and price[i] > 0]
            tgt = np.zeros(ns); tgt[top] = eq / n; delta = tgt - sh * price
            for i in np.where(delta < -1e-9)[0]:
                v = -delta[i]; c = RATE[ti, i] * v; cash += v - c; sh[i] -= v / price[i]; tr += v; cs += c
                if sh[i] <= 1e-12: sh[i] = 0; hc[i] = np.nan
            for i in np.where(delta > 1e-9)[0]:
                v = delta[i]; c = RATE[ti, i] * v; cash -= v + c
                if sh[i] == 0: hc[i] = np.nan
                sh[i] += v / price[i]; tr += v; cs += c
        def excl_set():
            if mode == 'M3a': return set(np.where(EVM[ti])[0])
            if mode == 'M3b':
                nxt = next((m for m in mons if m > ti), ti + 5); w = EVM[ti:nxt].any(axis=0)
                return {int(i) for i in np.where(w & (sh == 0))[0]}
            return set()
        if is_reset: reset(o, open_rk[ti], blocked, excl_set())
        post = cash + sh @ o; E[k] = post
        cost_f += cs / pre if pre > 0 else 0; trade_f += tr / pre if pre > 0 else 0
        if ti in close_rk:                                        # close-time rebalance, valued at the close
            pre_c = cash + sh @ CX[ti]; tr = cs = 0.0; reset(CX[ti], close_rk[ti], set(), set())
            cost_f += cs / pre_c if pre_c > 0 else 0; trade_f += tr / pre_c if pre_c > 0 else 0
        if keep: SH[k] = sh
        held = sh > 0; c = C[ti]; hc = np.where(held, np.fmax(hc, c), np.nan)
        if stop: pend |= held & np.isfinite(c) & (c <= (1 - stop) * hc)
    return dict(E=E, cost_f=cost_f, trade_f=trade_f, SH=SH)

def dd_series(E): m = np.maximum.accumulate(E); return E / m - 1
dates = tdays[T0:T1 + 1]; yrs = (dates[-1] - dates[0]).days / 365.25
spyE = spy_o[T0:T1 + 1] / spy_o[T0] * START_EQ
def yearly(E):
    s = pd.Series(pd.Series(E).pct_change().fillna(0).values, index=dates); return (1 + s).groupby(s.index.year).prod() - 1
spy_y = yearly(spyE); mstarts = [i for i in range(len(dates)) if i == 0 or dates[i].month != dates[i - 1].month]
def roll5(E):
    w = b = 0
    for i in mstarts:
        j = dates.searchsorted(dates[i] + pd.DateOffset(years=5))
        if j >= len(dates): continue
        w += 1; b += (E[j] / E[i]) > (spyE[j] / spyE[i])
    return b / max(w, 1), w

cells = {'REF': dict(), 'M1': dict(mode='M1', stop=.15), 'M1b': dict(mode='M1', stop=.20), 'M2': dict(mode='M2'), 'M3a': dict(mode='M3a'),
         'M3b': dict(mode='M3b'), 'M4a': dict(mode='M4a'), 'M4b': dict(mode='M4b'), 'M4c': dict(mode='M4c'), 'M5': dict(mode='M5'), 'M6': dict(mode='M6')}
res = {}
for name, kw in cells.items():
    res[name] = simulate(keep=(name == 'REF'), **kw); E = res[name]['E']
    log.info('%s %s end $%.0f CAGR %.2f%% maxDD %.1f%%', el(), name, E[-1], 100 * ((E[-1] / START_EQ) ** (1 / yrs) - 1), 100 * dd_series(E).min())
    if name == 'REF':
        c_ = (E[-1] / START_EQ) ** (1 / yrs) - 1
        if abs(c_ - 0.2718) > 0.002 or abs(dd_series(E).min() + 0.4450) > 0.002 or abs(E[-1] - 507823) > 0.01 * 507823:
            log.error('REF DOES NOT REPRODUCE (CAGR %.4f maxDD %.4f end %.0f) -- STOP', c_, dd_series(E).min(), E[-1]); raise SystemExit(2)
        log.info('REF reproduces 1700j')

# ---------------------------------------------------------------- M3 coverage on REF holdings
SHr = res['REF']['SH']; cov_n = cov_k = cov_n19 = cov_k19 = 0
for d in rebal_dates:
    k = didx[d] - T0; held = np.where(SHr[k] > 0)[0]
    for i in held:
        sn = cand[i]; arr = EVDATES.get(sn); has = False
        if arr is not None:
            lo = np.datetime64(d - pd.Timedelta(days=100)); has = bool(((arr >= lo) & (arr < np.datetime64(d))).any())
        cov_n += 1; cov_k += has
        if d >= pd.Timestamp('2019-07-01'): cov_n19 += 1; cov_k19 += has
coverage = cov_k / cov_n; cov19 = cov_k19 / max(cov_n19, 1)
log.info('%s M3 coverage all %.1f%% (%d name-weeks), from 2019-07 %.1f%%', el(), 100 * coverage, cov_n, 100 * cov19)

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
widx = np.array([didx[d] - T0 for d in rebal_dates]); wd = pd.DatetimeIndex(rebal_dates)
wk_ref = pd.Series(E[widx]).pct_change().dropna().values; wdates = wd[1:]
ref_cagr = (E[-1] / START_EQ) ** (1 / yrs) - 1; ref_dd = D.min(); ref_ratio = ref_cagr / abs(ref_dd); out = []
for name, r in res.items():
    Ec = r['E']; cagr = (Ec[-1] / START_EQ) ** (1 / yrs) - 1; mdd = dd_series(Ec).min(); y = yearly(Ec); wk = pd.Series(Ec[widx]).pct_change().dropna().values
    r5, nw = roll5(Ec); dif = wk - wk_ref; keepm = dif <= np.quantile(dif, 0.95)
    h1 = dif[wdates < pd.Timestamp('2022-01-01')].mean(); h2 = dif[wdates >= pd.Timestamp('2022-01-01')].mean()
    t = dif.mean() / (dif.std(ddof=1) / np.sqrt(len(dif))) if dif.std() > 0 else np.nan
    row = dict(cell=name, cagr=cagr, max_dd=mdd, ratio=cagr / abs(mdd), end_usd=Ec[-1], sharpe=wk.mean() / wk.std(ddof=1) * np.sqrt(52), worst_year=y.min(),
               worst_year_label=int(y.idxmin()), years_beat_spy=int((y > spy_y).sum()), n_years=len(y), roll5_share=r5, turnover_oneway_per_yr=r['trade_f'] / 2 / yrs,
               cost_drag_per_yr=r['cost_f'] / yrs, paired_mean=dif.mean(), paired_t=t, paired_ex_top5=dif[keepm].mean(), half1=h1, half2=h2)
    for k_, dpt in enumerate(ep_depths(Ec), 1): row[f'ep{k_}_depth'] = dpt
    row['pass'] = bool(name != 'REF' and row['ratio'] - ref_ratio >= 0.10 and cagr >= 0.25 and row['paired_ex_top5'] >= 0 and h1 > 0 and h2 > 0)
    out.append(row)
cdf = pd.DataFrame(out); cdf.to_csv(OUT / '1700l_cells.csv', index=False)
equiv = [x.cell for _, x in cdf.iterrows() if x.cell in ('M4a', 'M4b', 'M4c') and abs(x.cagr - ref_cagr) <= .01 and abs(x.max_dd - ref_dd) <= .01]
L = ['# RESULT 1,700l -- mid-week actions on the reconciled momentum sleeve (PREREG_1700l.md, frozen)', '',
     f'Daily sim {dates[0].date()}..{dates[-1].date()} ({yrs:.2f} y), $50K, 1700j engine/costs (band-based, not NBBO). REF reproduces 1700j (27.2%/-44.5%/$507,823). '
     'Paired = weekly return (Monday opens) minus REF; ex-top-5% drops the best 5% of weekly differences; halves 2017-21 / 2022-26 both must be > 0.', '',
     '| cell | CAGR | max DD | CAGR/DD | end $ | Sharpe | worst yr | yrs>SPY | roll5y | turn/yr | cost/yr | ep1..5 depth | paired mean/wk (t) | ex-top5 | halves | PASS |', '|' + '---|' * 16]
for _, x in cdf.iterrows():
    L.append(f"| {x['cell']} | {x['cagr']:.1%} | {x['max_dd']:.1%} | {x['ratio']:.2f} | {x['end_usd']:,.0f} | {x['sharpe']:.2f} | {x['worst_year']:.1%} ({x['worst_year_label']}) | {x['years_beat_spy']}/{x['n_years']} | {x['roll5_share']:.0%} | {x['turnover_oneway_per_yr']:.1f}x | {x['cost_drag_per_yr']:.2%} | "
             + ' '.join(f"{x[f'ep{k}_depth']:.0%}" for k in range(1, 6)) + f" | {x['paired_mean']*100:+.3f}% ({x['paired_t']:.1f}) | {x['paired_ex_top5']*100:+.3f}% | {x['half1']*100:+.3f}/{x['half2']*100:+.3f} | {'PASS' if x['pass'] else 'no'} |")
L += ['', f"Pass list: {', '.join(cdf.loc[cdf['pass'], 'cell']) or 'NONE'} (pass = ratio >= REF {ref_ratio:.2f}+0.10, CAGR >= 25%, ex-top5 paired >= 0, both halves > 0).",
      f"Timing cells equivalent to Monday open (within 1 pt CAGR and DD): {', '.join(equiv) or 'none'}. M3 coverage (held name-weeks with a 2.02 event in prior 100 days): {coverage:.1%} over all {cov_n} name-weeks "
      f"({cov19:.1%} from 2019-07; events_raw starts 2019) -> M3 {'VOID (<80%)' if coverage < 0.8 else 'valid'}.",
      'Conventions (PREREG silent, fixed before numbers): stops/events/gaps at the open; Monday pending stops executed + blocked from re-buy; M1/M5/M3b fill with next-ranked of Monday top-40; M3a slot in cash to Monday; M6 once per name per week, funded pro rata; M4 uses prior-day signal.']
(OUT / 'RESULT_1700l.md').write_text('\n'.join(L) + '\n'); log.info('DONE %s', el())
