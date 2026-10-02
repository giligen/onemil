"""Cell 1,700n (PREREG_1700n.md, FROZEN): ETF momentum diversifier (A1-A3), blends with the sleeve, short-SPY hedges (B1-B5).
Engine/loader/REF copied verbatim from 1700j_frontier.py (daily open-to-open, Monday rebalance, band costs)."""
from __future__ import annotations
import logging, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('/home/ec2-user/onemil/research/momentum_weekly')
logging.basicConfig(filename=str(OUT / '1700o.log'), filemode='w', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700o'); log.addHandler(logging.StreamHandler(sys.stdout))
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

# ================================================================ 1700o: conditional hedge, 32 cells
import itertools
spyc = EC['SPY'].ffill().values; qqc = EC['QQQ'].ffill().values
rS = np.r_[0, E_REF[1:] / E_REF[:-1] - 1]
rM_ = {'SPY': np.r_[0, spy_o[T0 + 1:T1 + 1] / spy_o[T0:T1] - 1],
       'QQQ': np.r_[0, EOv[T0 + 1:T1 + 1, ETFS.index('QQQ')] / EOv[T0:T1, ETFS.index('QQQ')] - 1]}
nall = len(tdays)
def trend_state(rule):
    """Boolean per trading day: SPY downtrend as known at that day's close (SPY always the trend instrument)."""
    c = pd.Series(spyc)
    if rule == '150d': return (c < c.rolling(150, min_periods=150).mean()).values
    if rule == '200d': return (c < c.rolling(200, min_periods=200).mean()).values
    if rule == '252r': return (c / c.shift(252) - 1 < 0).values
    if rule == '10mo':
        me = np.array([i + 1 < nall and tdays[i + 1].month != tdays[i].month for i in range(nall)])
        mc = pd.Series(spyc[me], index=np.where(me)[0]); ms = mc.rolling(10, min_periods=10).mean()
        flag = pd.Series(np.nan, index=range(nall)); flag.loc[mc.index] = (mc < ms).astype(float).values
        return flag.ffill().fillna(0).astype(bool).values
epr = pd.read_csv(OUT / '1700j_episodes.csv'); dpos = {d.date(): i for i, d in enumerate(dates)}
EP = [(dpos[pd.Timestamp(p).date()], dpos[pd.Timestamp(t).date()], (dpos[pd.Timestamp(r).date()] if r != 'none' else None)) for p, t, r in zip(epr.peak, epr.trough, epr.recovery)]
def ep_depths(Ec): return [(lambda w: (w / np.maximum.accumulate(w) - 1).min())(Ec[pk:(rc if rc else len(Ec))]) for pk, tr, rc in EP]
ref_ep = ep_depths(E_REF)
h1 = int(np.where(dates.year <= 2021)[0][-1])
RULES = ['150d', '200d', '10mo', '252r']; RATIOS = [.25, .5, .75, 1.0]; INSTR = ['SPY', 'QQQ']
def run_cell(rule, ratio, ins):
    st = trend_state(rule); cond = np.array([bool(st[T0 + k - 1]) if k > 0 else False for k in range(ND)])
    h = np.where(cond, ratio, 0.); rM = rM_[ins]
    leg = -h * rM - h * 0.005 / 252; r = rS + leg
    Ec = START_EQ * np.cumprod(1 + np.r_[0, r[1:]])
    prev = np.r_[START_EQ, Ec[:-1]]; legusd = prev * leg
    spells = []; k = 0
    while k < ND:
        if h[k] > 0:
            j = k
            while j + 1 < ND and h[j + 1] > 0: j += 1
            spells.append((dates[k].date(), dates[j].date(), float(legusd[k:j + 1].sum()))); k = j + 1
        else: k += 1
    return Ec, h, spells
def half_ratio(Ec, a, b):
    s = Ec[a:b + 1]; cg = (s[-1] / s[0]) ** (365.25 / (dates[b] - dates[a]).days) - 1
    return cg / abs(dd_series(s).min())
ref_half = (half_ratio(E_REF, 0, h1), half_ratio(E_REF, h1, ND - 1))
ref_ratio = ref_cagr / abs(ref_dd); rows = []; SP_ = {}
for rule, ratio, ins in itertools.product(RULES, RATIOS, INSTR):
    Ec, h, spells = run_cell(rule, ratio, ins); cagr = (Ec[-1] / START_EQ) ** (1 / yrs) - 1; mdd = dd_series(Ec).min()
    hr = (half_ratio(Ec, 0, h1), half_ratio(Ec, h1, ND - 1)); ed = ep_depths(Ec)
    row = dict(rule=rule, ratio=ratio, instr=ins, cagr=cagr, max_dd=mdd, ratio_cd=cagr / abs(mdd), end_usd=Ec[-1], pct_days_hedged=float((h > 0).mean()),
               n_spells=len(spells), spells_won=sum(s[2] > 0 for s in spells), both_improve=bool(cagr > ref_cagr and mdd > ref_dd),
               half1_gain=hr[0] - ref_half[0], half2_gain=hr[1] - ref_half[1], halves_ok=bool(hr[0] > ref_half[0] and hr[1] > ref_half[1]))
    for i, d in enumerate(ed, 1): row[f'ep{i}_depth'] = d
    rows.append(row); SP_[(rule, ratio, ins)] = spells
    log.info('%s %s %.2f %s: CAGR %.2f%% DD %.2f%% end %.0f spells %d won %d', el(), rule, ratio, ins, 100 * cagr, 100 * mdd, Ec[-1], len(spells), row['spells_won'])
df = pd.DataFrame(rows); df.to_csv(OUT / '1700o_cells.csv', index=False)
b4 = df[(df.rule == '200d') & (df.ratio == .5) & (df.instr == 'SPY')].iloc[0]
log.info('REPRO REF %.2f%% %.2f%% %.0f | B4 cell %.2f%% %.2f%% %.0f', 100 * ref_cagr, 100 * ref_dd, E_REF[-1], 100 * b4.cagr, 100 * b4.max_dd, b4.end_usd)
ok_b4 = abs(b4.cagr - .289) <= .002 and abs(b4.max_dd + .403) <= .002 and abs(b4.end_usd - 578749) <= .01 * 578749
nb = int(df.both_improve.sum()); tot_sp = int(df.n_spells.sum()); won = int(df.spells_won.sum()); share = won / max(tot_sp, 1)
nh = int(df.halves_ok.sum()); real = nb >= 24 and share >= .6 and nh >= 24
med = df.sort_values('ratio_cd').iloc[len(df) // 2 - 1:len(df) // 2 + 1][['cagr', 'max_dd', 'end_usd']].mean()
sp0 = SP_[('200d', .5, 'SPY')]
L = ['# RESULT 1,700o -- is the conditional market hedge real? (PREREG_1700o.md, FROZEN)', '',
     f'REF {100*ref_cagr:.2f}% / {100*ref_dd:.2f}% / ${E_REF[-1]:,.0f} (reproduces 27.18/-44.50/507,823). B4 cell (200d,50%,SPY) {100*b4.cagr:.2f}% / {100*b4.max_dd:.2f}% / ${b4.end_usd:,.0f} (1700n B4 28.9/-40.3/578,749): {"OK" if ok_b4 else "MISMATCH"}.', '',
     '| trend | hedge | instr | CAGR | maxDD | ratio | end $ | %days hedged | spells | spells won | ep1..5 depth | half1/half2 ratio gain |', '|' + '---|' * 12]
for _, r in df.iterrows():
    L.append(f"| {r.rule} | {int(r.ratio*100)}% | {r.instr} | {100*r.cagr:.1f}% | {100*r.max_dd:.1f}% | {r.ratio_cd:.2f} | {r.end_usd:,.0f} | {100*r.pct_days_hedged:.1f}% | {r.n_spells} | {r.spells_won} | "
             + '/'.join(f'{100*r[f"ep{i}_depth"]:.0f}' for i in range(1, 6)) + f" | {r.half1_gain:+.2f}/{r.half2_gain:+.2f} |")
L += ['', f'REF ep depths: ' + '/'.join(f'{100*x:.0f}' for x in ref_ep) + f'; REF half ratios {ref_half[0]:.2f}/{ref_half[1]:.2f}; REF ratio {ref_ratio:.2f}.', '',
      f'1. Cells improving BOTH CAGR and max DD vs REF: {nb} of 32 (need >= 24).',
      f'2. Hedge-leg spell win share: {won}/{tot_sp} = {100*share:.0f}% (need >= 60%).',
      f'3. Cells improving the ratio in BOTH halves: {nh} of 32 (PREREG says "both halves improve"; applied at the same >= 24 bar).',
      f'VERDICT: {"REAL" if real else "FAVOURABLE DRAW"}', f'Median-ratio neighbour (mean of the two middle cells): CAGR {100*med.cagr:.1f}% / maxDD {100*med.max_dd:.1f}% / end ${med.end_usd:,.0f}.', '',
      'Spells of 200d/50%/SPY (start, end, hedge P&L $): ' + '; '.join(f'{a}..{b} {p:+,.0f}' for a, b, p in sp0), '',
      'Caveats: daily open-to-open, hedge costed at borrow 0.5%/yr only (as 1,700n), trend rules read on prior close; one 10-year sample, 5 episodes; neighbours are highly correlated (same SPY drawdowns), so 32 cells are far fewer than 32 independent looks.']
(OUT / 'RESULT_1700o.md').write_text('\n'.join(L) + '\n'); log.info('spells list: %s', sp0); log.info('DONE ok_b4=%s nb=%d share=%.2f nh=%d real=%s', ok_b4, nb, share, nh, real)
