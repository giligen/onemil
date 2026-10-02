#!/usr/bin/env python3
"""Independent rebuild of cells 1,703a (52-week-high momentum, forms a1/a2) and 1,703b (dual momentum) from the PREREG prose.
Run: bash scripts/research_run.sh -m 4000M python3 research/known_strategies/REBUILD_1703ab.py > research/known_strategies/REBUILD_1703ab.log
Universe/hygiene/cost conventions follow research/momentum_weekly/1700s_lowvix.py (U2: close>=10, ADV20>=200M, >=273 bars, name exclusions, gtype==0)."""
from __future__ import annotations
import re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

MW = Path('/home/ec2-user/onemil/research/momentum_weekly'); OUTD = Path('/home/ec2-user/onemil/research/known_strategies')
START_EQ, PRICE_MIN, ADV_CUT = 50_000.0, 10.0, 200_000_000.0
WIN_START, WIN_END = pd.Timestamp('2017-01-01'), pd.Timestamp('2026-09-30')
TEST_RE = re.compile(r'^Z[A-Z]ZZT$')
NAME_RE = re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|\bPREFERRED\b|\bRIGHTS?\b', re.I)
t0 = time.time()
def say(*a): print(f'[{time.time()-t0:5.0f}s]', *a, flush=True)

# ------------------------------------------------------------------ load
tbl = pq.read_table(MW / 'panel_2016_2026.parquet', columns=['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume'], read_dictionary=['symbol'])
tbl = tbl.cast(pa.schema([('symbol', tbl.schema.field('symbol').type), ('bar_date', tbl.schema.field('bar_date').type)] + [(c, pa.float32()) for c in ('open', 'high', 'low', 'close', 'volume')]))
raw = tbl.to_pandas(self_destruct=True); del tbl
raw['bar_date'] = pd.to_datetime(raw['bar_date'])
codes0 = raw.symbol.values.codes; cats = raw.symbol.values.categories; d0 = raw.bar_date.values
if not bool(np.all((codes0[1:] > codes0[:-1]) | ((codes0[1:] == codes0[:-1]) & (d0[1:] >= d0[:-1])))): say('raw not sorted'); raise SystemExit(4)
say('rows', len(raw))
# ETF series for cell b (taken before any universe mask)
ETF = {}
for s in ('SPY', 'EFA', 'SHY', 'IEF'):
    x = raw.loc[raw.symbol == s, ['bar_date', 'open', 'close']].drop_duplicates('bar_date', keep='last').set_index('bar_date').sort_index()
    ETF[s] = x.astype(np.float64); say(s, len(x), x.index[0].date(), x.index[-1].date())
tdays = ETF['SPY'].index; didx = {d: i for i, d in enumerate(tdays)}
assets = pd.read_csv(MW / '1700c_assets.csv', dtype={'symbol': str, 'name': str}); assets['name'] = assets['name'].fillna('')
excl = set(assets.loc[assets.name.str.contains(NAME_RE), 'symbol']) | {x for x in cats if TEST_RE.match(x)}
bad = np.array([(x in excl) or x == 'SPY' for x in cats])
dup_next = np.append((codes0[1:] == codes0[:-1]) & (d0[1:] == d0[:-1]), False)
o_, h_, l_, c_, v_ = (raw[k].values for k in ('open', 'high', 'low', 'close', 'volume'))
keep = ~dup_next & ~((o_ <= 0) | (h_ <= 0) | (l_ <= 0) | (c_ <= 0)) & ~bad[codes0]
CODE = codes0[keep].copy(); DT = d0[keep].copy(); OP = o_[keep].astype(np.float64); HI = h_[keep].astype(np.float64); CL = c_[keep].astype(np.float64)
SPREAD = (((h_[keep] - l_[keep]) / c_[keep]).clip(min=0) * 0.1).clip(max=0.002).astype(np.float64); DV = c_[keep].astype(np.float64) * v_[keep].astype(np.float64)
del raw, codes0, d0, o_, h_, l_, c_, v_, keep, dup_next
import gc; gc.collect(); say('universe rows', len(CODE))

# ------------------------------------------------------------------ signal dates
per = pd.Series(tdays).dt.to_period('W'); first = pd.Series(tdays).groupby(per).min().sort_index()
ent = [d for d in first if WIN_START <= d <= WIN_END]                      # Monday-open (first session of the week) trade dates
mper = pd.Series(tdays).dt.to_period('M'); mlast = pd.Series(tdays).groupby(mper).max().sort_index()   # last session of each month
SIG_A1 = {d: tdays[didx[d] - 1] for d in ent}                              # signal = prior session close
T0, T1 = didx[ent[0]], didx[ent[-1]]
month_sig = [s for s in mlast if didx[s] + 1 < len(tdays) and T0 <= didx[s] + 1 <= T1]   # signal on month-end, trade next session open
SIG_A2 = {s: tdays[didx[s] + 1] for s in month_sig}
sig_all = np.array(sorted(set(SIG_A1.values()) | set(SIG_A2.keys())), dtype='datetime64[ns]')
say('weeks', len(ent), 'months', len(month_sig), 'window', ent[0].date(), ent[-1].date())

# ------------------------------------------------------------------ features at signal dates (causal rolling per symbol)
def features(upto=None):
    """Per-symbol rolling features, returned only at signal dates. If upto is given, ALL rows after that date are deleted first (shift test)."""
    m = slice(None) if upto is None else slice(0, int(np.searchsorted(DT, np.datetime64(upto), side='right')) if False else None)
    sel = np.ones(len(CODE), bool) if upto is None else (DT <= np.datetime64(upto))
    code, dt, cl, hi, dv = CODE[sel], DT[sel], CL[sel], HI[sel], DV[sel]
    bnd = np.flatnonzero(np.diff(code)) + 1; st = np.r_[0, bnd]; en = np.r_[bnd, len(code)]
    rows = []
    for a, b in zip(st, en):
        d = dt[a:b]; sm = np.isin(d, sig_all)
        if not sm.any(): continue
        c = cl[a:b]; n = b - a; s = pd.Series(c)
        adv = pd.Series(dv[a:b]).rolling(20, min_periods=20).mean().values
        hmax = pd.Series(hi[a:b]).rolling(252, min_periods=252).max().values
        r126 = np.full(n, np.nan); r126[126:] = c[126:] / c[:-126] - 1
        mv = np.zeros(n); mv[1:] = c[1:] / c[:-1] - 1; gp = np.zeros(n, bool); gp[1:] = np.diff(d).astype('timedelta64[D]').astype(np.int64) > 10
        ea = pd.Series(((mv > 2.0) | (mv < -0.75)).astype(np.float64)).rolling(273, min_periods=1).max().values > 0
        eb = pd.Series(gp.astype(np.float64)).rolling(273, min_periods=1).max().values > 0
        nb = np.arange(1, n + 1)
        with np.errstate(divide='ignore', invalid='ignore'): ratio = c / hmax
        ok = (c >= PRICE_MIN) & (adv >= ADV_CUT) & (nb >= 273) & ~ea & ~eb & np.isfinite(ratio) & np.isfinite(r126)
        idx = np.flatnonzero(sm & ok)
        if len(idx): rows.append(pd.DataFrame({'code': code[a], 'date': d[idx], 'ratio': np.round(ratio[idx], 9), 'r126': r126[idx]}))
    f = pd.concat(rows, ignore_index=True); f['symbol'] = [cats[i] for i in f.code]
    return f
F = features(); say('features', len(F))
G = {pd.Timestamp(k): v for k, v in F.groupby('date')}
def top(date, n):
    """Top n by ratio desc, ties (ratio==1) by 126-session return desc."""
    g = G[pd.Timestamp(date)].sort_values(['ratio', 'r126'], ascending=False, kind='mergesort'); return g.head(n)
def tiecount(date): return int((G[pd.Timestamp(date)].ratio >= 1.0).sum())

# ------------------------------------------------------------------ price matrices for candidates
A1 = {d: top(SIG_A1[d], 20) for d in ent if SIG_A1[d] in G}
A2 = {SIG_A2[s]: top(s, 30) for s in month_sig if s in G}
cand = sorted({c for t in list(A1.values()) + list(A2.values()) for c in t.code}); cpos = {c: i for i, c in enumerate(cand)}
nd = len(tdays); O = np.full((nd, len(cand)), np.nan); RATE = np.full((nd, len(cand)), 0.002)
cm = np.isin(CODE, cand); ci = np.flatnonzero(cm)
ti = np.searchsorted(tdays.values, DT[ci]); ok_ = (ti < nd) & (tdays.values[np.minimum(ti, nd - 1)] == DT[ci])
cc = np.array([cpos[c] for c in CODE[ci]]); O[ti[ok_], cc[ok_]] = OP[ci][ok_]; RATE[ti[ok_], cc[ok_]] = np.minimum(0.0005 + 0.5 * SPREAD[ci][ok_], 0.002)
O = pd.DataFrame(O).ffill().fillna(0.0).values
say('candidates', len(cand))

# ------------------------------------------------------------------ engines
def sim_a1():
    """Every first session of the week: reset the 20 names to equity/20 at the open (delta trades), RATE cost per traded dollar."""
    sh = np.zeros(len(cand)); cash = START_EQ; E = np.zeros(T1 - T0 + 1); trade = 0.0; tr_yr = {}
    for k in range(len(E)):
        t = T0 + k; o = O[t]
        if tdays[t] in A1:
            eq = cash + sh @ o; tgt = np.zeros(len(cand)); tp = [cpos[c] for c in A1[tdays[t]].code if o[cpos[c]] > 0]; tgt[tp] = eq / 20
            dl = tgt - sh * o
            for i in np.where(dl < -1e-9)[0]:
                v = -dl[i]; cash += v * (1 - RATE[t, i]); sh[i] -= v / o[i]; trade += v; tr_yr[tdays[t].year] = tr_yr.get(tdays[t].year, 0) + v / eq
            for i in np.where(dl > 1e-9)[0]:
                v = dl[i]; cash -= v * (1 + RATE[t, i]); sh[i] += v / o[i]; trade += v; tr_yr[tdays[t].year] = tr_yr.get(tdays[t].year, 0) + v / eq
            sh[sh < 1e-12] = 0
        E[k] = cash + sh @ o
    return E, tr_yr

def sim_a2():
    """Month-end signal, next-session open: expiring tranche (opened 6 months ago) sold; its proceeds (or cash/empty-count while ramping) buy the top 30 equal-weight."""
    tr = [np.zeros(len(cand)) for _ in range(6)]; filled = [False] * 6; cash = START_EQ; E = np.zeros(T1 - T0 + 1); m = 0; tr_yr = {}
    for k in range(len(E)):
        t = T0 + k; o = O[t]
        if tdays[t] in A2:
            j = m % 6; m += 1; eq = cash + sum(x @ o for x in tr)
            v = tr[j] * o; sell = v.sum(); cash += (v * (1 - RATE[t])).sum(); tr[j] = np.zeros(len(cand))
            budget = cash / (6 - sum(filled)) if not filled[j] else sell * 1.0
            if filled[j]: budget = min(cash, sell)
            tp = [cpos[c] for c in A2[tdays[t]].code if o[cpos[c]] > 0]
            for i in tp:
                vv = budget / len(tp) / (1 + RATE[t, i]); tr[j][i] = vv / o[i]; cash -= vv * (1 + RATE[t, i])
            filled[j] = True; tr_yr[tdays[t].year] = tr_yr.get(tdays[t].year, 0) + (sell + budget) / eq
        E[k] = cash + sum(x @ o for x in tr)
    return E, tr_yr

def b_decision(closes, sdate):
    """Dual momentum decision at a month-end session from closes (a dict of Series indexed by date); only data <= sdate used."""
    r = {}
    for s in ('SPY', 'EFA', 'SHY'):
        c = closes[s][:sdate]
        if len(c) < 253: return None
        r[s] = c.iloc[-1] / c.iloc[-253] - 1                           # close / close 252 sessions back - 1
    w = 'SPY' if r['SPY'] >= r['EFA'] else 'EFA'
    return (w if r[w] > r['SHY'] else 'IEF'), r
CLS = {s: ETF[s].close for s in ETF}

def sim_b():
    """Month-end decision, next session open trade, 2 bp per side; marks at the open."""
    cash = START_EQ; hold = None; sh = 0.0; E = np.zeros(T1 - T0 + 1); log = []; trade_yr = {}
    sigmap = {tdays[didx[s] + 1]: s for s in mlast if didx[s] + 1 < len(tdays)}
    for k in range(len(E)):
        t = T0 + k; d = tdays[t]
        if d in sigmap:
            dec = b_decision(CLS, sigmap[d])
            if dec is not None:
                new, r = dec; eq = cash + (sh * ETF[hold].open.iloc[t] if hold else 0)
                log.append((sigmap[d], d, r['SPY'], r['EFA'], r['SHY'], new, eq))
                if new != hold:
                    if hold: cash += sh * ETF[hold].open.iloc[t] * (1 - 0.0002); sh = 0.0
                    px = ETF[new].open.iloc[t]; sh = cash / (1 + 0.0002) / px; cash = 0.0; hold = new
                    trade_yr[d.year] = trade_yr.get(d.year, 0) + 2.0
        E[k] = cash + (sh * ETF[hold].open.iloc[t] if hold else 0)
    return E, trade_yr, log

# ------------------------------------------------------------------ reads
dates = tdays[T0:T1 + 1]; yrs = (dates[-1] - dates[0]).days / 365.25
def reads(E, tr_yr, name):
    r = pd.Series(E).pct_change().fillna(0).values; cagr = (E[-1] / E[0]) ** (1 / yrs) - 1; dd = (E / np.maximum.accumulate(E) - 1).min()
    s = pd.Series(r, index=dates); by = (1 + s).groupby(s.index.year).prod() - 1
    out = dict(cell=name, cagr=cagr, maxdd=dd, end=E[-1], sharpe=r.mean() / r.std() * np.sqrt(252), worst_year=by.min(), turnover_per_yr=sum(tr_yr.values()) / yrs)
    say(name, {k: (round(v, 4) if not isinstance(v, str) else v) for k, v in out.items()}); return out, by
Ea1, tya1 = sim_a1(); Ea2, tya2 = sim_a2(); Eb, tyb, blog = sim_b()
SPYE = ETF['SPY'].open.values[T0:T1 + 1] / ETF['SPY'].open.values[T0] * START_EQ
ra1, ya1 = reads(Ea1, tya1, 'a1'); ra2, ya2 = reads(Ea2, tya2, 'a2'); rb, yb = reads(Eb, tyb, 'b'); rs, ys = reads(SPYE, {}, 'SPY_buyhold')
yt = pd.DataFrame({'a1': ya1, 'a2': ya2, 'b': yb, 'SPY': ys}); yt.index.name = 'year'; yt.to_csv(OUTD / 'REBUILD_1703ab_by_year.csv', float_format='%.4f'); say('\n' + yt.round(3).to_string())
pd.DataFrame([ra1, ra2, rb, rs]).to_csv(OUTD / 'REBUILD_1703ab_reads.csv', index=False)

# ------------------------------------------------------------------ ties, top-20 prints
tie1 = [tiecount(SIG_A1[d]) for d in ent if SIG_A1[d] in G]; tie2 = [tiecount(s) for s in month_sig if s in G]
say('TIES at 1.0 a1 weekly: median', np.median(tie1), 'mean', np.mean(tie1), 'max', max(tie1), 'share of weeks with >=20 ties', np.mean(np.array(tie1) >= 20))
say('TIES at 1.0 a2 monthly: median', np.median(tie2))
for d in ('2021-02-01', '2024-06-03', '2026-09-21'):
    d = pd.Timestamp(d); say('TOP20', d.date(), 'signal', SIG_A1[d].date(), 'ties', tiecount(SIG_A1[d]), ' '.join(f'{r.symbol}({r.ratio:.3f})' for r in top(SIG_A1[d], 20).itertuples()))

# ------------------------------------------------------------------ b log + hand-checks
bl = pd.DataFrame(blog, columns=['signal', 'trade', 'r_SPY', 'r_EFA', 'r_SHY', 'hold', 'equity']); bl['month_ret'] = bl['equity'].shift(-1) / bl['equity'] - 1
say('B holdings per month:\n' + bl.drop(columns='equity').round(4).to_string())
say('B hold counts', bl.hold.value_counts().to_dict())
for s in bl.signal:
    if (s.year == 2020 and s.month in (3, 4)) or s.year == 2022:
        row = bl[bl.signal == s].iloc[0]; i = CLS['SPY'][:s].shape[0] - 1
        say('HAND', s.date(), {k: (round(CLS[k][:s].iloc[-1], 2), round(CLS[k][:s].iloc[-253], 2), round(CLS[k][:s].iloc[-1] / CLS[k][:s].iloc[-253] - 1, 4)) for k in ('SPY', 'EFA', 'SHY')}, '->', row.hold)

# ------------------------------------------------------------------ shift tests (delete all later rows, recompute, compare)
def shift_a(sig_date, n, tag):
    Fs = features(upto=sig_date); g = Fs[Fs.date == pd.Timestamp(sig_date)].sort_values(['ratio', 'r126'], ascending=False, kind='mergesort').head(n)
    same = g.symbol.tolist() == top(sig_date, n).symbol.tolist(); say('SHIFT', tag, sig_date, 'identical holdings:', same); return same
s1 = shift_a(SIG_A1[pd.Timestamp('2021-02-01')], 20, 'a1'); s2 = shift_a(mlast[mlast.dt.strftime('%Y-%m') == '2022-06'].iloc[0], 30, 'a2')
sd = mlast[mlast.dt.strftime('%Y-%m') == '2022-06'].iloc[0]; full = b_decision(CLS, sd)
tr_cl = {k: v[:sd] for k, v in CLS.items()}; trunc = b_decision(tr_cl, sd); s3 = full[0] == trunc[0] and full[1] == trunc[1]
say('SHIFT b', sd.date(), 'identical:', s3, full[0])
with open(OUTD / 'REBUILD_1703ab_shift.txt', 'w') as fh: fh.write(f'a1 {s1}\na2 {s2}\nb {s3}\n')
bl.to_csv(OUTD / 'REBUILD_1703ab_b_months.csv', index=False)
say('done')
