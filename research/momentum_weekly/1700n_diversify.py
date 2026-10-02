"""Cell 1,700n (PREREG_1700n.md, FROZEN): ETF momentum diversifier (A1-A3), blends with the sleeve, short-SPY hedges (B1-B5).
Engine/loader/REF copied verbatim from 1700j_frontier.py (daily open-to-open, Monday rebalance, band costs)."""
from __future__ import annotations
import logging, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('/home/ec2-user/onemil/research/momentum_weekly')
logging.basicConfig(filename=str(OUT / '1700n.log'), filemode='w', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700n'); log.addHandler(logging.StreamHandler(sys.stdout))
PRICE_MIN, ADV_CUT, START_EQ = 10.0, 200_000_000.0, 50_000.0
WIN_START, WIN_END = pd.Timestamp('2017-01-01'), pd.Timestamp('2026-09-30')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|\bPREFERRED\b|\bRIGHTS?\b', re.I)
MAXN = 40
ETFS = 'SPY QQQ IWM EFA EEM VNQ TLT IEF SHY LQD HYG TIP GLD SLV DBC USO XLE XLF XLK XLV XLI XLP XLY XLU XLB'.split()
t0 = time.time()
def el(): return f'{time.time()-t0:5.0f}s'

# ---------------------------------------------------------------- panel + signals (as 1700g)
cols = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc, gc
_t = pq.read_table(OUT / 'panel_2016_2026.parquet', columns=cols, read_dictionary=['symbol'])
_t = _t.set_column(_t.schema.get_field_index('volume'), 'volume', pc.cast(_t['volume'], pa.float32()))
for _c in ('open', 'high', 'low', 'close'): _t = _t.set_column(_t.schema.get_field_index(_c), _c, pc.cast(_t[_c], pa.float32()))
_dv = pc.multiply(pc.cast(_t['close'], pa.float64()), pc.cast(_t['volume'], pa.float64()))
_g = pa.table({'symbol': _t['symbol'].cast(pa.string()), 'mc': _t['close'], 'mdv': _dv}).group_by('symbol').aggregate([('mc', 'max'), ('mdv', 'max')]).to_pandas()
# necessary conditions for ever entering the sleeve (price >= 10 and adv20 >= 200M at some date) + the ETF list + SPY: identical sleeve, far less memory
_keep = set(_g.loc[(_g.mc_max >= PRICE_MIN) & (_g.mdv_max >= ADV_CUT), 'symbol']) | set(ETFS)
log.info('prefilter keeps %d of %d symbols', len(_keep), len(_g))
_t = _t.filter(pc.is_in(_t['symbol'].cast(pa.string()), value_set=pa.array(sorted(_keep))))
raw = _t.to_pandas(); del _t, _dv, _g; gc.collect()
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
ETF = raw[raw.symbol.isin(ETFS)].copy(); ETF['symbol'] = ETF.symbol.astype(str)
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
# ================================================================ 1700n
log.info('%s REF simulate', el())
ref = simulate(); E_REF = ref['E']
ref_cagr = (E_REF[-1] / START_EQ) ** (1 / yrs) - 1; ref_dd = dd_series(E_REF).min()
log.info('REF check: CAGR %.2f%% maxDD %.2f%% end $%.0f', 100 * ref_cagr, 100 * ref_dd, E_REF[-1])
if abs(ref_cagr - 0.272) > 0.002 or abs(ref_dd + 0.445) > 0.002 or abs(E_REF[-1] - 507823) > 0.01 * 507823:
    log.error('REF MISMATCH: expected 27.2%% / -44.5%% / 507,823'); raise SystemExit(2)
ND = len(dates); TD = tdays[T0:T1 + 1]
# ---- ETF panel
ETF['bar_date'] = pd.to_datetime(ETF.bar_date)
def epiv(col): return ETF.pivot_table(index='bar_date', columns='symbol', values=col, aggfunc='last').reindex(index=tdays, columns=ETFS)
EO, EC, EH, EL = epiv('open'), epiv('close'), epiv('high'), epiv('low')
for s in ETFS:
    fv = EC[s].first_valid_index(); log.info('ETF %s %s first %s', s, 'OK' if fv is not None else 'MISSING', fv.date() if fv is not None else '-')
ret = EC.pct_change(fill_method=None)
SIG = (EC.shift(21) / EC.shift(252) - 1)
RAW12 = SIG.copy()
SIGV = SIG / ret.rolling(252, min_periods=252).std()
ESP = (((EH - EL) / EC).clip(lower=0) * 0.1).clip(upper=0.002)
ERATE = np.minimum(0.0002 + 0.5 * ESP.fillna(0.002).values, 0.002)
EOv = EO.ffill().values.astype(np.float64); iSHY = ETFS.index('SHY')
def etf_book(topn, freq):
    """ETF top-N by 12-1/vol252 (prior-day signal); slot whose 12-1 <= SHY's goes to SHY; equal-weight slots; rebalance at the open."""
    dts = list(range(T0, T1 + 1))
    if freq == 'W': rb = {didx[d] for d in rebal_dates}
    else:
        rb = {T0}; 
        for ti in range(T0 + 1, T1 + 1):
            if tdays[ti].month != tdays[ti - 1].month: rb.add(ti)
    sh = np.zeros(len(ETFS)); cash = START_EQ; E = np.zeros(ND); cost = 0.0; trade = 0.0; hold = []
    for k, ti in enumerate(dts):
        o = EOv[ti]
        if ti in rb:
            sv = SIGV.values[ti - 1]; r12 = RAW12.values[ti - 1]; ok = np.isfinite(sv) & np.isfinite(r12) & (o > 0)
            idx = np.where(ok)[0]; top = idx[np.argsort(-sv[idx])][:topn]
            tgt = np.zeros(len(ETFS)); shy12 = r12[iSHY]
            for i in top:
                j = iSHY if (not np.isfinite(shy12) or r12[i] <= shy12) else i
                tgt[j] += 1.0 / topn
            if len(top) < topn: tgt[iSHY] += (topn - len(top)) / topn
            eq = cash + sh @ o; tgt = tgt * eq; delta = tgt - sh * o
            for i in np.where(np.abs(delta) > 1e-9)[0]:
                v = abs(delta[i]); c = ERATE[ti, i] * v; cost += c; trade += v
                cash += (v if delta[i] < 0 else -v) - c; sh[i] += delta[i] / o[i]
        E[k] = cash + sh @ o
    return dict(E=E, cost_f=cost / START_EQ, trade_f=trade / START_EQ)
def blend(Es, Ed, w_s):
    """Monthly rebalance to w_s sleeve / 1-w_s diversifier on daily open-to-open stream returns; 2 bps on traded amount."""
    rs = np.r_[0, Es[1:] / Es[:-1] - 1]; rd = np.r_[0, Ed[1:] / Ed[:-1] - 1]
    a, b = START_EQ * w_s, START_EQ * (1 - w_s); E = np.zeros(ND)
    for k in range(ND):
        a *= 1 + rs[k]; b *= 1 + rd[k]
        if k == 0 or TD[k].month != TD[k - 1].month:
            tot = a + b; t = tot * w_s; tradev = abs(a - t) * 2; tot -= 0.0002 * tradev; a, b = tot * w_s, tot * (1 - w_s)
        E[k] = a + b
    return E
spyc = EC['SPY'].ffill().values
spy_sma = pd.Series(spyc).rolling(200, min_periods=200).mean().values
rS = np.r_[0, E_REF[1:] / E_REF[:-1] - 1]; rM = np.r_[0, spy_o[T0 + 1:T1 + 1] / spy_o[T0:T1] - 1]
def hedge(kind):
    """Short SPY overlay on the sleeve: fraction h_k (decided from info up to the prior close) of equity, daily rebalanced; borrow 0.5%/yr; cash 0."""
    h = np.zeros(ND)
    sret = pd.Series(rS); mret = pd.Series(rM)
    if kind == 'B1': h[:] = .3
    elif kind == 'B2': h[:] = .5
    elif kind == 'B3':
        beta = (sret.rolling(126, min_periods=63).cov(mret) / mret.rolling(126, min_periods=63).var()).shift(1).fillna(0).values
        h = 0.5 * np.clip(beta, 0, 2)
    elif kind == 'B4':
        cond = np.array([np.isfinite(spy_sma[T0 + k - 1]) and spyc[T0 + k - 1] < spy_sma[T0 + k - 1] if k > 0 else False for k in range(ND)]); h = np.where(cond, .5, 0.)
    elif kind == 'B5':
        v = sret.rolling(63).std(); med = v.rolling(756, min_periods=252).median(); h = ((v > med).shift(1).fillna(False)).astype(float).values * .5
    r = rS - h * rM - h * 0.005 / 252
    return START_EQ * np.cumprod(1 + np.r_[0, r[1:]]), h
# ---- metrics
epr = pd.read_csv(OUT / '1700j_episodes.csv'); dpos = {d.date(): i for i, d in enumerate(dates)}
EP = [(dpos[pd.Timestamp(p).date()], dpos[pd.Timestamp(t).date()], (dpos[pd.Timestamp(r).date()] if r != 'none' else None)) for p, t, r in zip(epr.peak, epr.trough, epr.recovery)]
wkmask = np.array([d in set(rebal_dates) for d in dates])
def wk(Ec): return pd.Series(Ec[wkmask]).pct_change().dropna().values
wkS = wk(E_REF); wkmask_idx = np.where(wkmask)[0]
def inside(Ec):
    sel = np.zeros(len(wkmask_idx) - 1, bool)
    for pk, tr, rc in EP: sel |= (wkmask_idx[1:] > pk) & (wkmask_idx[1:] <= tr)
    return sel
INS = inside(E_REF)
def ep_depths(Ec): return [(lambda w: (w / np.maximum.accumulate(w) - 1).min())(Ec[pk:(rc if rc else len(Ec))]) for pk, tr, rc in EP]
ref_ep = ep_depths(E_REF)
def cagr_of(a, b, Ec): return (Ec[b] / Ec[a]) ** (365.25 / (dates[b] - dates[a]).days) - 1
h1 = int(np.where(dates.year <= 2021)[0][-1]); 
def metrics(name, part, Ec, kind, extra=None):
    cagr = (Ec[-1] / START_EQ) ** (1 / yrs) - 1; mdd = dd_series(Ec).min(); y = yearly(Ec); w = wk(Ec); epd = ep_depths(Ec)
    cut = sum((a - b) >= 0.25 * abs(b) for a, b in zip(epd, ref_ep)); r5, _ = roll5(Ec)
    c1, c2 = cagr_of(0, h1, Ec), cagr_of(h1, len(Ec) - 1, Ec)
    row = dict(cell=name, part=part, cagr=cagr, max_dd=mdd, ratio=cagr / abs(mdd), end_usd=Ec[-1], sharpe=w.mean() / w.std(ddof=1) * np.sqrt(52),
               worst_year=y.min(), worst_year_label=int(y.idxmin()), years_beat_spy=int((y > spy_y).sum()), n_years=len(y), roll5_share=r5,
               corr_all=np.corrcoef(w, wkS)[0, 1], corr_in_eps=np.corrcoef(w[INS], wkS[INS])[0, 1], cagr_H1=c1, cagr_H2=c2, eps_cut=cut,
               ref_ratio_gain=cagr / abs(mdd) - ref_cagr / abs(ref_dd))
    for k, d in enumerate(epd, 1): row[f'ep{k}_depth'] = d
    for k, (pk, tr, rc) in enumerate(EP, 1): row[f'ep{k}_ret_peak_trough'] = Ec[tr] / Ec[pk] - 1
    row['pass'] = ('n/a' if kind == 'A' else bool((mdd - ref_dd) >= 0.10 and cagr >= 0.20 and row['ref_ratio_gain'] >= 0.15 and cut >= 3 and c1 >= 0.15 and c2 >= 0.15))
    return row
rows = [metrics('REF', 'ref', E_REF, 'R')]; STREAM = {}
for nm, (n, f) in dict(A1=(3, 'M'), A2=(3, 'W'), A3=(5, 'M')).items():
    r = etf_book(n, f); STREAM[nm] = r['E']; row = metrics(nm, 'A_standalone', r['E'], 'A')
    row['turn_per_yr'] = r['trade_f'] / 2 / yrs; row['cost_per_yr'] = r['cost_f'] / yrs; rows.append(row)
    log.info('%s %s CAGR %.1f%% DD %.1f%% end %.0f', el(), nm, 100 * row['cagr'], 100 * row['max_dd'], r['E'][-1])
for nm in ('A1', 'A2', 'A3'):
    for ws in (.7, .5, .3):
        Eb = blend(E_REF, STREAM[nm], ws); rows.append(metrics(f'{nm}_{int(ws*100)}/{int(100-ws*100)}', 'A_blend', Eb, 'B'))
for nm in ('B1', 'B2', 'B3', 'B4', 'B5'):
    Eh, h = hedge(nm); row = metrics(nm, 'B_hedge', Eh, 'B'); row['mean_short'] = float(h.mean()); rows.append(row)
cdf = pd.DataFrame(rows); cdf.to_csv(OUT / '1700n_cells.csv', index=False)
spyE_ = spyE; sp = metrics('SPY', 'bench', spyE_, 'R')
log.info('SPY: CAGR %.1f%% DD %.1f%%', 100 * sp['cagr'], 100 * sp['max_dd'])
def tbl(sub, passcol=True):
    L = ['| cell | CAGR | maxDD | ratio | end $ | Sharpe | worst yr | yrs>SPY | roll5y | corr all/eps | H1/H2 CAGR | ep1..5 depth | PASS |', '|' + '---|' * 13]
    for _, x in sub.iterrows():
        L.append(f"| {x['cell']} | {x['cagr']:.1%} | {x['max_dd']:.1%} | {x['ratio']:.2f} | {x['end_usd']:,.0f} | {x['sharpe']:.2f} | {x['worst_year']:.0%} ({x['worst_year_label']}) | {x['years_beat_spy']}/{x['n_years']} | {x['roll5_share']:.0%} | {x['corr_all']:.2f}/{x['corr_in_eps']:.2f} | {x['cagr_H1']:.0%}/{x['cagr_H2']:.0%} | " + ' '.join(f"{x[f'ep{k}_depth']:.0%}" for k in range(1, 6)) + f" | {x['pass']} |")
    return L
sa = cdf[cdf.part == 'A_standalone']
L = ['# RESULT 1,700n -- ETF diversifier and short-SPY hedge for the momentum sleeve (PREREG_1700n.md)', '',
     f'Daily sim {dates[0].date()}..{dates[-1].date()} ({yrs:.2f} y), $50K, same engine/costs as 1700j (band cost, not NBBO). REF check passed: {ref_cagr:.1%} / {ref_dd:.1%} / ${E_REF[-1]:,.0f}. SPY: {sp["cagr"]:.1%} / {sp["max_dd"]:.1%}. '
     'Ratio = CAGR/|maxDD|. ep depth = depth inside each 1700j episode window (peak to recovery). Pass: DD +10 pts, CAGR>=20%, ratio +0.15, >=3/5 episodes cut (>=25% shallower), both halves CAGR>=15%.',
     'Impl notes: B5 median = trailing 756d of the 63d vol, min 252d (no hedge before); hedge/blend daily open-to-open; B3 beta clipped 0..2; blend rebalance cost 2 bps of traded; SHY used for the absolute filter.', '',
     '## Stand-alone diversifier (PASS n/a)', ''] + tbl(pd.concat([cdf[cdf.cell == 'REF'], sa]))
L += ['', 'Diversifier return peak->trough inside each REF episode (ep1..5): ' + '; '.join(f"{x['cell']}: " + ' '.join(f"{x[f'ep{k}_ret_peak_trough']:+.1%}" for k in range(1, 6)) for _, x in sa.iterrows()),
      'Sleeve (REF) peak->trough: ' + ' '.join(f"{(E_REF[tr]/E_REF[pk]-1):+.1%}" for pk, tr, rc in EP),
      'ETF presence: ' + ', '.join(f"{s} {EC[s].first_valid_index().date()}" if EC[s].first_valid_index() is not None else f"{s} MISSING" for s in ETFS) + '. Cost/turnover: ' + '; '.join(f"{x['cell']} {x['turn_per_yr']:.1f}x/yr, {x['cost_per_yr']:.2%}/yr" for _, x in sa.iterrows()), '',
      '## Blends (sleeve/diversifier)', ''] + tbl(cdf[cdf.part == 'A_blend'])
L += ['', '## Hedges (short SPY)', ''] + tbl(cdf[cdf.part == 'B_hedge'])
L += ['', 'Mean short fraction: ' + ', '.join(f"{x['cell']} {x['mean_short']:.2f}" for _, x in cdf[cdf.part == 'B_hedge'].iterrows()), '']
cand_ = cdf[cdf.part.isin(['A_blend', 'B_hedge'])]
L += [f"Pass list: {', '.join(cand_.loc[cand_['pass'] == True, 'cell']) or 'NONE'}."]
for part, lab in (('A_blend', 'Blends'), ('B_hedge', 'Hedges')):
    s = cdf[(cdf.part == part) & (cdf.max_dd >= -0.30)]
    L.append(f"Best end $ with max DD <= -30% ({lab}): " + (f"{s.sort_values('end_usd').iloc[-1]['cell']} CAGR {s.sort_values('end_usd').iloc[-1]['cagr']:.1%}, DD {s.sort_values('end_usd').iloc[-1]['max_dd']:.1%}, ${s.sort_values('end_usd').iloc[-1]['end_usd']:,.0f}" if len(s) else 'none reaches -30%'))
L.append('Cells: 3 + 9 + 5 = 17 frozen; nothing added after numbers.')
(OUT / 'RESULT_1700n.md').write_text('\n'.join(L) + '\n')
log.info('DONE %s', el())
