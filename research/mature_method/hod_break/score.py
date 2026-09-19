#!/usr/bin/env python3
"""mature_method / HOD-break — steps 1-9 of RUNBOOK.md, cells exactly as declared in PREREG.md.

TEST is sealed: no TEST number is computed unless FREEZE.md exists AND --test is passed.
Read-only on every DB. One process.
"""
import os, sqlite3, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book                       # noqa: E402
from research.scripts.pit_listings import is_test_ticker     # noqa: E402

D = f'{ROOT}/research/mature_method/hod_break'
RATIO = {'stop': 0.875, 'eod': 0.412, 'target': 0.0}         # the score4 contract (CAUSAL_FILTER §6)
PB_EDGES = [0, 5, 10, 20, 30, 50, 100, 1e9]
PB_LAB = ['<$5', '$5-10', '$10-20', '$20-30', '$30-50', '$50-100', '$100+']
HB_EDGES = [569, 585, 600, 660, 780, 960]
HB_LAB = ['0930-0945', '0945-1000', '1000-1100', '1100-1300', '1300+']
CAP = 0.006
RISK = 100.0                                                  # live risk_usd
EARLY_CLOSE = {'2025-07-03', '2025-11-28', '2025-12-24'}
SPLIT_RANGE = {'TRAIN': ('2025-01-02', '2025-12-31'), 'VAL': ('2026-01-01', '2026-05-31'),
               'TEST': ('2026-06-01', '2026-09-11')}
NW = {s: len(pd.period_range(a, b, freq='W-FRI')) for s, (a, b) in SPLIT_RANGE.items()}
ALL_WEEKS = {s: [str(p) for p in pd.period_range(a, b, freq='W-FRI')] for s, (a, b) in SPLIT_RANGE.items()}
WITH_TEST = '--test' in sys.argv and os.path.exists(f'{D}/FREEZE.md')
SPLITS = ('TRAIN', 'VAL', 'TEST') if WITH_TEST else ('TRAIN', 'VAL')
IMPUTE = {}
IMPUTE_GLOBAL = np.nan
BAND = {}
BAND_GLOBAL = np.nan


def split_of(day):
    return np.where(day < '2026-01-01', 'TRAIN', np.where(day < '2026-06-01', 'VAL', 'TEST'))


# ------------------------------------------------------------------ step 1: reproduction gate
def repro():
    """Reproduce REPORT §6a's LIVE-CONFIG book from spec_trades.csv, to the printed digit."""
    T = pd.read_csv(f'{ROOT}/research/bf_zero/spec_trades.csv', dtype={'symbol': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    T['split'] = split_of(T.day.values)
    T['wk'] = pd.to_datetime(T.day).dt.to_period('W-FRI').astype(str)
    TL = T[(T.entry_m <= 841) & (T.price >= 20.0)]
    rows = [(r.day, int(r.entry_m), int(r.exit_m), r.symbol, r.Index) for r in TL.itertuples()]
    bk = TL.loc[[t[4] for t in run_book(rows, 12, 4)]]
    print('\n== STEP 1 reproduction gate: REPORT §6a LIVE-CONFIG book (12/day, 4 conc, >=$20, <=14:00) ==')
    for s in ('TRAIN', 'VAL', 'TEST'):
        d = bk[bk.split == s]
        w = d.groupby('wk').rr.sum().reindex(ALL_WEEKS[s]).fillna(0.0)
        print(f'  {s:5s} {len(d) / NW[s]:5.1f}/wk  meanR {d.rr.mean():+.3f}  weekly {w.mean():+.1f}R  '
              f'green {(w > 0).sum()}/{NW[s]}  worst {w.min():+.1f}')
    T2 = T[~T.day.isin(EARLY_CLOSE)]
    TL2 = T2[(T2.entry_m <= 841) & (T2.price >= 20.0)]
    r2 = [(r.day, int(r.entry_m), int(r.exit_m), r.symbol, r.Index) for r in TL2.itertuples()]
    b2 = TL2.loc[[t[4] for t in run_book(r2, 12, 4)]]
    d2 = b2[b2.split == 'TRAIN']
    print(f'  (with the 3 early-close days REMOVED -- what §6a\'s prose claims -- TRAIN is '
          f'{len(d2) / NW["TRAIN"]:.1f}/wk meanR {d2.rr.mean():+.4f}; §6a\'s printed 44.1/-0.030 is the '
          f'INCLUDED variant, so its code did not apply cut (c). VAL/TEST are unaffected.)')
    print('  reference §6a: TRAIN 44.1/wk -0.030 -1.3 24/53 -17.5 | VAL 45.0 -0.006 -0.3 9/23 -16.2 | '
          'TEST 43.7 +0.016 +0.7 9/15 -17.5')
    return bk


# ------------------------------------------------------------------ load
def daily_bars_symbols():
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    s = pd.read_sql('select distinct symbol from daily_bars', con).symbol.astype(str)
    con.close()
    return set(s)


def load_breaks():
    bk = pd.read_csv(f'{D}/breaks.csv', dtype={'symbol': str, 'day': str, 'why_b': str, 'why_l': str},
                     keep_default_na=False, na_values=[''])
    n0 = len(bk); bk = bk[~bk.day.isin(EARLY_CLOSE)]
    n1 = len(bk); bk = bk[~bk.symbol.map(is_test_ticker)]
    n2 = len(bk); bk = bk[bk.symbol.isin(daily_bars_symbols())]
    n3 = len(bk)
    bk = bk.sort_values(['day', 'symbol', 'entry_m'], kind='mergesort').reset_index(drop=True)
    bk['split'] = split_of(bk.day.values)
    bk['wk'] = pd.to_datetime(bk.day).dt.to_period('W-FRI').astype(str)
    print(f'\nmembership: break rows {n0} -> minus early-close {n1} -> minus test tickers {n2} '
          f'-> in daily_bars {n3}', flush=True)
    nb = pd.read_csv(f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv',
                     dtype={'symbol': str, 'day': str}, keep_default_na=False,
                     na_values=['']).drop_duplicates(['day', 'symbol', 'entry_m'])
    # EXACT (day, symbol, minute) match only: the NBBO was measured at that signal's own minute.
    # A cell whose first-qualifying break lands on another minute gets the declared imputation.
    bk = bk.merge(nb[['day', 'symbol', 'entry_m', 'spread_mean', 'ask_dec', 'bid_dec', 'n_sig']],
                  on=['day', 'symbol', 'entry_m'], how='left')
    return bk


def build_impute(bk):
    """The imputation model for signals outside the measured set: median spread-% by
    (price band x hour band) of the MEASURED sample. Declared in PREREG §1."""
    global IMPUTE, IMPUTE_GLOBAL
    m = bk[bk.spread_mean.notna() & (bk.next_open > 0)].copy()
    m['sp_pct'] = m.spread_mean / m.next_open * 100
    m['pb'] = pd.cut(m.next_open, PB_EDGES, labels=PB_LAB)
    m['hb'] = pd.cut(m.entry_m, HB_EDGES, labels=HB_LAB)
    IMPUTE = m.groupby(['pb', 'hb'], observed=True).sp_pct.median().to_dict()
    IMPUTE_GLOBAL = float(m.sp_pct.median())
    global BAND, BAND_GLOBAL
    cc = pd.read_csv(f'{ROOT}/research/lit_review_2026/cost_curve.csv', dtype={'symbol': str, 'day': str},
                     keep_default_na=False, na_values=[''])
    cc = cc[(cc.n_q > 0) & (cc.price > 0)].copy()
    cc['sp_pct'] = cc.spread / cc.price * 100
    cc['pb'] = pd.cut(cc.price, PB_EDGES, labels=PB_LAB); cc['hb'] = pd.cut(cc.entry_m, HB_EDGES, labels=HB_LAB)
    BAND = cc.groupby(['pb', 'hb'], observed=True).sp_pct.mean().to_dict()
    BAND_GLOBAL = float(cc.sp_pct.mean())
    print(f'cost model: measured NBBO on {len(m)} of {len(bk)} break rows '
          f'({len(m) / len(bk):.1%}); imputation cells {len(IMPUTE)}, global median {IMPUTE_GLOBAL:.3f}%')


def attach_cost(x, tag):
    x = x.copy()
    x['price'] = x.next_open
    x['r_pct'] = x[f'r_pct_{tag}']; x['rr'] = x[f'rr_{tag}']; x['why'] = x[f'why_{tag}']
    x['exit_m'] = x[f'exit_m_{tag}']; x['stop'] = x[f'stop_{tag}']
    pb = pd.cut(x.price, PB_EDGES, labels=PB_LAB); hb = pd.cut(x.entry_m, HB_EDGES, labels=HB_LAB)
    sp = x.spread_mean / x.price * 100
    x['imputed'] = sp.isna()
    imp = pd.Series([IMPUTE.get((p, h), IMPUTE_GLOBAL) for p, h in zip(pb, hb)], index=x.index)
    bnd = pd.Series([BAND.get((p, h), BAND_GLOBAL) for p, h in zip(pb, hb)], index=x.index)
    x['sp_pct'] = sp.fillna(imp).fillna(IMPUTE_GLOBAL)              # PREREG arm
    x['sp_band'] = sp.fillna(bnd).fillna(BAND_GLOBAL)               # conservative arm (band table)
    ratio = x.why.map(RATIO).fillna(0.875)
    half = 0.5 * x.sp_pct / x.r_pct.clip(lower=0.05)
    x['net'] = x.rr - half - half * ratio
    hb2 = 0.5 * x.sp_band / x.r_pct.clip(lower=0.05)
    x['netb'] = x.rr - hb2 - hb2 * ratio
    x['notional'] = RISK / (x.price - x.stop).clip(lower=1e-6) * x.price
    x['obtainable'] = np.where(x.ask_dec.notna(), x.ask_dec <= x.level * (1 + CAP) * (1 + 1e-9), True)
    return x


# ------------------------------------------------------------------ the cell machine
def signals(bk, tag='b', dist_min=5.0, rv=(1.0, 5.0), last_m=840, min_price=20.0, r_min=1.0):
    """detect() + simulate() semantics: the FIRST break passing the DETECTION gates per symbol-day,
    then the level/fill/r_min gates -- which KILL the symbol-day (the engine retires the candidate)
    rather than promoting a later break."""
    d = bk[bk[f'stop_{tag}'].notna() & (bk.dist_open_pct >= dist_min) & (bk.entry_m <= last_m + 1)]
    if rv is not None:
        d = d[(d.rv_profile >= rv[0]) & (d.rv_profile < rv[1])]
    d = d.drop_duplicates(['day', 'symbol'], keep='first')   # bk is pre-sorted by (day, symbol, entry_m)
    return d[(d.fill_capped == 1) & (d[f'r_pct_{tag}'] >= r_min) & (d.next_open >= min_price)]


def apply_book(s, nday, nconc):
    if not len(s):
        return s.assign(pnl=pd.Series(dtype=float))
    rows = [(r.day, int(r.entry_m), int(r.exit_m), r.symbol, r.Index) for r in s.itertuples()]
    b = s.loc[[t[4] for t in run_book(rows, nday, nconc)]].copy()
    b['pnl'] = b.net * RISK
    return b


def cell(bk, tag='b', dist_min=5.0, rv=(1.0, 5.0), last_m=840, min_price=20.0, r_min=1.0,
         nday=12, nconc=4, max_bps=100.0, max_frac_r=0.15, obtain=True, veto_d5=False,
         return_pre=False):
    s = attach_cost(signals(bk, tag, dist_min, rv, last_m, min_price, r_min), tag)
    if veto_d5:
        s = s[~((s.entry_m >= 615) & (s.entry_m < 624))]
    if max_bps:
        s = s[(s.sp_pct * 100) <= max_bps]
    if max_frac_r:
        s = s[(s.sp_pct / s.r_pct.clip(lower=0.05)) <= max_frac_r]
    if obtain:
        s = s[s.obtainable.astype(bool)]
    b = apply_book(s, nday, nconc)
    return (b, s) if return_pre else b


# ------------------------------------------------------------------ reporting
def week_stats(b, split):
    wk = ALL_WEEKS[split]; d = b[b.split == split]
    w = d.groupby('wk').pnl.sum().reindex(wk).fillna(0.0)
    traded = d.groupby('wk').size().reindex(wk).fillna(0)
    streak = mx = 0
    for v in (w < 0).values:
        streak = streak + 1 if v else 0
        mx = max(mx, streak)
    cum = w.cumsum(); mo = d.groupby(d.day.str[:7]).pnl.sum()
    return dict(n=len(d), per_wk=len(d) / NW[split],
                net=float(d.net.mean()) if len(d) else np.nan,
                netb=float(d.netb.mean()) if len(d) else np.nan,
                gross=float(d.rr.mean()) if len(d) else np.nan,
                t=float(d.net.mean() / (d.net.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 else np.nan,
                green=float((w > 0).mean() * 100), flat=float((traded == 0).mean() * 100),
                redstreak=mx, worst=float(w.min()), best=float(w.max()), total=float(w.sum()),
                mdd=float((cum - cum.cummax()).min()), wk_mean=float(w.mean()),
                green_mo=float((mo > 0).mean() * 100) if len(mo) else np.nan,
                ex5=float(d.net[d.net <= d.net.quantile(0.95)].mean()) if len(d) > 20 else np.nan,
                imp=float(d.imputed.mean() * 100) if len(d) else np.nan)


def null_band(b, split, draws=2000, seed=7):
    d = b[b.split == split]
    if len(d) < 5:
        return (np.nan,) * 4
    wk = ALL_WEEKS[split]
    cnt = d.groupby('wk').size().reindex(wk).fillna(0).astype(int).values
    pnl = d.pnl.values.copy()
    obs = float((d.groupby('wk').pnl.sum().reindex(wk).fillna(0.0) > 0).mean() * 100)
    rng = np.random.default_rng(seed); edges = np.cumsum(cnt)[:-1]
    out = np.empty(draws)
    for i in range(draws):
        p = rng.permutation(pnl)
        out[i] = np.mean([s.sum() > 0 for s in np.split(p, edges)]) * 100
    return obs, float(out.mean()), float(np.percentile(out, 5)), float(np.percentile(out, 95))


def sep_row(kept, rej, label, pos=''):
    o = {'gate': label, 'pos': pos}
    for s in ('POOL',) + SPLITS:
        k = kept if s == 'POOL' else kept[kept.split == s]
        r = rej if s == 'POOL' else rej[rej.split == s]
        if len(k) < 3 or len(r) < 3:
            o[s] = None; continue
        dm = float(k.net.mean() - r.net.mean())
        se = float(np.sqrt(k.net.var(ddof=1) / len(k) + r.net.var(ddof=1) / len(r)))
        o[s] = dict(d=dm, t=dm / se if se else np.nan, nk=len(k), nr=len(r),
                    mk=float(k.net.mean()), mr=float(r.net.mean()),
                    g=float(k.rr.mean() - r.rr.mean()))
    o['notional_kept'] = float(kept.notional.median()) if len(kept) else np.nan
    o['notional_rej'] = float(rej.notional.median()) if len(rej) else np.nan
    return o


def fmt_sep(o):
    p = o['POOL']
    s = f"| {o['gate']:<34s} | {o['pos']:<3s} |"
    if p is None:
        return s + ' n/a |'
    s += f" {p['g']:+.3f} | {p['d']:+.3f} | {p['t']:+.2f} | {p['nk']:5d} | {p['nr']:5d} | {p['mk']:+.3f} | {p['mr']:+.3f} |"
    for sp in SPLITS:
        q = o[sp]
        s += f" {q['d']:+.3f}" if q else ' n/a'
        s += ' |'
    s += f" {o['notional_kept']:,.0f} | {o['notional_rej']:,.0f} |"
    return s


def fmt_cell(name, b, extra=''):
    out = []
    for s in SPLITS:
        w = week_stats(b, s)
        out.append(f"| {name:<28s} | {s:5s} | {w['n']:5d} | {w['per_wk']:5.1f} | {w['gross']:+.3f} | "
                   f"{w['net']:+.3f} | {w['netb']:+.3f} | {w['t']:+5.2f} | {w['green']:5.1f} | {w['flat']:5.1f} | "
                   f"{w['redstreak']:2d} | {w['worst']:+8.0f} | {w['total']:+9.0f} | {w['mdd']:+8.0f} | "
                   f"{w['green_mo']:5.1f} | {w['ex5']:+.3f} | {w['imp']:4.0f} |{extra}")
    return '\n'.join(out)


# ------------------------------------------------------------------ driver
HDR_CELL = ('| cell                         | split |     n |  /wk |  grossR |    netR |   netB |     t | green |  flat | '
            'rs |    worst |     total |      mdd | grnMo |    ex5% | imp% |')
SEP_HDR = ('| gate                               | pos | grossD |  netD |     t |  n_kp |  n_rj |  meanK |  meanR |'
           + ''.join(f' {s[:5]} |' for s in SPLITS) + ' notl_kp | notl_rj |')


SHIPPED = dict(tag='b', dist_min=5.0, rv=(1.0, 5.0), last_m=840, min_price=20.0, r_min=1.0,
               max_bps=100.0, max_frac_r=0.15, obtain=True)


def stack(bk, **over):
    """The shipped stack with overrides -> the PRE-BOOK signal set (every gate but the book's slots)."""
    kw = {**SHIPPED, **over}
    s = attach_cost(signals(bk, kw['tag'], kw['dist_min'], kw['rv'], kw['last_m'], kw['min_price'],
                            kw['r_min']), kw['tag'])
    if kw['max_bps']:
        s = s[(s.sp_pct * 100) <= kw['max_bps']]
    if kw['max_frac_r']:
        s = s[(s.sp_pct / s.r_pct.clip(lower=0.05)) <= kw['max_frac_r']]
    if kw['obtain']:
        s = s[s.obtainable.astype(bool)]
    return s


def gate_map(bk):
    """LEAVE-ONE-OUT separation: for each gate, the whole shipped stack MINUS that gate is the
    population, and the gate splits it. That is 'at its cascade position' -- every other live rule
    is in force, so the number is what removing this one gate would do."""
    print('\n== STEP 5 gate-separation map (leave-one-out on the shipped stack; GROSS and NET R) ==')
    print(SEP_HDR)
    rows = []
    base = stack(bk)
    for lo in (7.0, 10.0, 15.0):
        rows.append(sep_row(base[base.dist_open_pct >= lo], base[base.dist_open_pct < lo],
                            f'Ga dist>={lo:.0f}% (ladder, in-stack)', '1'))
    p = stack(bk, rv=None)
    inb = (p.rv_profile >= 1) & (p.rv_profile < 5)
    rows.append(sep_row(p[inb], p[~inb], 'Gb rv in [1,5)', '2'))
    rows.append(sep_row(p[inb], p[p.rv_profile < 1], 'Gb   .. vs rv<1 only', '2'))
    rows.append(sep_row(p[inb], p[p.rv_profile >= 5], 'Gb   .. vs rv>=5 only', '2'))
    pl = stack(bk, tag='l')
    rows.append(sep_row(pl[pl.stop_b.notna()], pl[pl.stop_b.isna()], 'Gc consol K5/4% (loose stop both)', '3'))
    q = stack(bk, min_price=0.0)
    for f in (5.0, 10.0, 20.0, 50.0):
        rows.append(sep_row(q[q.price >= f], q[q.price < f], f'Gd price>=${f:.0f}', '4'))
    r = stack(bk, r_min=0.0)
    rows.append(sep_row(r[r.r_pct >= 1.0], r[r.r_pct < 1.0], 'Ge stop >=1% of entry', '5'))
    f6 = stack(bk, max_frac_r=0.0)
    fr = f6.sp_pct / f6.r_pct.clip(lower=0.05)
    rows.append(sep_row(f6[fr <= 0.15], f6[fr > 0.15], 'Gf spread <=15% of R', '6'))
    g7 = stack(bk, max_bps=0.0)
    rows.append(sep_row(g7[g7.sp_pct * 100 <= 100], g7[g7.sp_pct * 100 > 100], 'Gg spread <=100 bps', '7'))
    h8 = stack(bk, last_m=930)
    rows.append(sep_row(h8[h8.entry_m <= 841], h8[h8.entry_m > 841], 'Gh entry <=14:00', '8'))
    ob = stack(bk, obtain=False)
    rows.append(sep_row(ob[ob.obtainable.astype(bool)], ob[~ob.obtainable.astype(bool)],
                        'Gk obtainable (quoted ask<=cap)', '8b'))
    for lbl, nd, nc, pos in (('Gi 12/day (conc unbound)', 12, 9999, '9'),
                             ('Gj 4 concurrent (day unbound)', 9999, 4, '10'),
                             ('Gi+Gj 12/day & 4 concurrent', 12, 4, '9+10')):
        tk = apply_book(base, nd, nc)
        rows.append(sep_row(base.loc[tk.index], base.drop(index=tk.index), lbl, pos))
    for o in rows:
        print(fmt_sep(o))
    return rows


CELLS = {
    'B0 shipped':              dict(),
    'F-b rv band OFF':         dict(rv=None),
    'F-c consolidation loose': dict(tag='l'),
    'F-d price floor $5':      dict(min_price=5.0),
    'F-e r_min OFF':           dict(r_min=0.0),
    'F-f 15%-of-R cap OFF':    dict(max_frac_r=0.0),
    'F-g 100bps ceiling OFF':  dict(max_bps=0.0),
    'F-h last entry 15:30':    dict(last_m=930),
    'F-i 20/day':              dict(nday=20),
    'F-j 8 concurrent':        dict(nconc=8),
    'T1 dist >=10%':           dict(dist_min=10.0),
    'T2 rv in [1,3)':          dict(rv=(1.0, 3.0)),
    'T3 price >=$50':          dict(min_price=50.0),
    'T4 r_min >=2%':           dict(r_min=2.0),
    'T5 spread <=8% of R':     dict(max_frac_r=0.08),
    'C1 both spread gates OFF': dict(max_bps=0.0, max_frac_r=0.0),
    'C2 rv OFF + r_min OFF':   dict(rv=None, r_min=0.0),
    'C3 price $5 + rv OFF':    dict(min_price=5.0, rv=None),
    'C4 slots 20/8 + 15:30':   dict(nday=20, nconc=8, last_m=930),
    'C5 structural ceiling':   dict(tag='l', rv=None, r_min=0.0, min_price=5.0, max_bps=0.0,
                                    max_frac_r=0.0, last_m=930, nday=20, nconc=8),
    'C6 T1+T2 (tight)':        dict(dist_min=10.0, rv=(1.0, 3.0)),
    'C7 B0 + D5 veto':         dict(veto_d5=True),
}



# ------------------------------------------------------------------ steps 2, 3, 4
def step2_gross(bk):
    """GROSS R (no cost) on the honest population and on the shipped book, with the MDE."""
    print('\n== STEP 2 gross before net (NO cost) ==')
    s0 = attach_cost(signals(bk), 'b')
    b0 = apply_book(s0.assign(net=s0.rr), 12, 4)
    for lbl, d in (('all live-config signals', s0), ('the shipped 12/4 book', b0)):
        for sp in SPLITS:
            x = d[d.split == sp]
            if len(x) < 3:
                continue
            se = x.rr.std(ddof=1) / np.sqrt(len(x))
            print(f'  {lbl:<24s} {sp:5s} n {len(x):5d} ({len(x)/NW[sp]:4.1f}/wk)  grossR {x.rr.mean():+.4f} '
                  f'+/- {se:.4f}  t {x.rr.mean()/se:+5.2f}  MDE80 {2.8*se:.3f}R  WR {(x.rr>0).mean()*100:4.1f}%')
    return b0


def step3_cost(bk):
    """MEASURED NBBO vs the band constant, on the shipped book's own signals."""
    print('\n== STEP 3 measured cost vs the band table ==')
    s0 = attach_cost(signals(bk), 'b')
    m = s0[s0.spread_mean.notna()]
    print(f'  measured coverage on B0 signals: {len(m)}/{len(s0)} = {len(m)/len(s0):.1%}')
    print(f'  measured spread: median {m.sp_pct.median()*100:.0f} bps, mean {m.sp_pct.mean()*100:.0f} bps, '
          f'p90 {np.percentile(m.sp_pct,90)*100:.0f} bps')
    try:
        cc = pd.read_csv(f'{ROOT}/research/lit_review_2026/cost_curve.csv', dtype={'symbol': str, 'day': str},
                         keep_default_na=False, na_values=[''])
        cc = cc[(cc.n_q > 0) & (cc.price > 0)].copy()
        cc['sp_pct'] = cc.spread / cc.price * 100
        cc['pb'] = pd.cut(cc.price, PB_EDGES, labels=PB_LAB); cc['hb'] = pd.cut(cc.entry_m, HB_EDGES, labels=HB_LAB)
        band = cc.groupby(['pb', 'hb'], observed=True).sp_pct.mean()
        glob = float(cc[cc.price >= 20].sp_pct.mean())
        pb = pd.cut(m.price, PB_EDGES, labels=PB_LAB); hb = pd.cut(m.entry_m, HB_EDGES, labels=HB_LAB)
        bb = pd.Series([band.get((p, h), glob) for p, h in zip(pb, hb)], index=m.index)
        print(f'  band constant on the SAME signals: median {bb.median()*100:.0f} bps, mean {bb.mean()*100:.0f} bps'
              f'  -> the band OVER-charges by {bb.mean()/m.sp_pct.mean():.2f}x (mean) / '
              f'{bb.median()/m.sp_pct.median():.2f}x (median)')
        ratio = m.why.map(RATIO).fillna(0.875)
        for tag, sp in (('measured', m.sp_pct), ('band', bb)):
            half = 0.5 * sp / m.r_pct.clip(lower=0.05)
            print(f'  cost charged per trade, {tag:8s}: {float((half + half*ratio).mean()):+.4f} R')
    except Exception as e:
        print(f'  band arm unavailable: {type(e).__name__} {e}')


def step4_fill(bk):
    """The engine's real fill model and the UNFILLED counterfactual."""
    print('\n== STEP 4 capped-limit fill and the unfilled counterfactual ==')
    d = bk[bk.stop_b.notna() & (bk.dist_open_pct >= 5.0) & (bk.entry_m <= 841)]
    d = d[(d.rv_profile >= 1) & (d.rv_profile < 5)].drop_duplicates(['day', 'symbol'], keep='first')
    d = d[(d.r_pct_b >= 1.0) & (d.next_open >= 20.0)]
    d = attach_cost(d, 'b')
    f, u = d[d.fill_capped == 1], d[d.fill_capped == 0]
    print(f'  first-qualifying signals {len(d)} | filled at the cap {len(f)} ({len(f)/len(d):.1%}) | '
          f'no fill {len(u)} ({len(u)/len(d):.1%})')
    for lbl, x in (('FILLED (paid <= level x 1.006)', f), ('UNFILLED counterfactual (paid the open)', u)):
        for sp in SPLITS:
            y = x[x.split == sp]
            if len(y) < 3:
                continue
            print(f'  {lbl:<40s} {sp:5s} n {len(y):5d}  grossR {y.rr.mean():+.3f}  netR {y.net.mean():+.3f}  '
                  f'WR {(y.rr>0).mean()*100:4.1f}%  median gap {(y.next_open/y.level-1).median()*1e4:+.0f} bps')
    gap = (u.next_open / u.level - 1) * 1e4
    print(f'  unfilled overshoot above the level: median {gap.median():.0f} bps, p90 {np.percentile(gap,90):.0f} bps')
    print('  CLASSIFICATION: a chase guard if the unfilled population is WORSE than the filled one '
          '(the fill is a better price on the same setup); a dip-buy if it is BETTER.')
    # obtainability from the measured NBBO ask at the decision instant
    q = f[f.ask_dec.notna()]
    if len(q):
        ob = q.ask_dec <= q.level * (1 + CAP)
        print(f'  measured-NBBO obtainability on filled rows: {ob.mean():.1%} of {len(q)} quoted '
              f'(the bar open says fill, the quoted ask says {1-ob.mean():.1%} would not have)')


def main():
    repro()
    bk = load_breaks()
    build_impute(bk)

    # --- independent-rebuild check: the base cell derived from breaks.csv vs spec_trades.csv
    s0 = signals(bk, last_m=930, min_price=0.0, r_min=1.0)
    T = pd.read_csv(f'{ROOT}/research/bf_zero/spec_trades.csv', dtype={'symbol': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    T = T[~T.day.isin(EARLY_CLOSE) & ~T.symbol.map(is_test_ticker)]
    m = s0.merge(T[['day', 'symbol', 'entry_m', 'rr']], on=['day', 'symbol'], suffixes=('', '_spec'))
    same_m = (m.entry_m == m.entry_m_spec)
    print(f'\n== independent rebuild vs spec_trades.csv: {len(m)} shared symbol-days, '
          f'same entry minute {same_m.mean():.4%}, max |d rr| '
          f'{np.abs(m.rr_b[same_m] - m.rr[same_m]).max():.2e} ==')
    print(f'   pass signals {len(s0)} vs spec (post-membership) {len(T)}')

    step2_gross(bk)
    step3_cost(bk)
    step4_fill(bk)
    gate_map(bk)

    print('\n== STEP 6/7 frequency frontier, ranked on % GREEN WEEKS (dollars at $100 risk) ==')
    print(HDR_CELL)
    books = {}
    for name, kw in CELLS.items():
        b = cell(bk, **kw)
        books[name] = b
        print(fmt_cell(name, b))
    pd.to_pickle({k: v[['day', 'symbol', 'split', 'wk', 'entry_m', 'net', 'netb', 'rr', 'pnl', 'price',
                        'r_pct', 'why', 'imputed']] for k, v in books.items()}, f'{D}/books.pkl')

    print('\n== STEP 9 count-matched permutation null (2,000 draws, per-week pick count fixed) ==')
    print('| cell                         | split | obs green% | null mean | null p5 | null p95 | inside |')
    for name, b in books.items():
        for s in SPLITS:
            o, mu, p5, p95 = null_band(b, s)
            if o != o:
                continue
            print(f'| {name:<28s} | {s:5s} | {o:9.1f}  | {mu:8.1f}  | {p5:6.1f}  | {p95:7.1f}  | '
                  f'{"yes" if p5 <= o <= p95 else "NO":6s} |')

    print('\n== STEP 8 weekly dollar table, B0 and the best frontier cells, VAL (2026-01..05) ==')
    for name in ('B0 shipped', 'F-d price floor $5', 'C5 structural ceiling', 'C7 B0 + D5 veto'):
        b = books[name]
        d = b[b.split == 'VAL']
        w = d.groupby('wk').pnl.sum().reindex(ALL_WEEKS['VAL']).fillna(0.0)
        n = d.groupby('wk').size().reindex(ALL_WEEKS['VAL']).fillna(0).astype(int)
        print(f'\n{name}: ' + ' '.join(f'{int(v):+d}({c})' for v, c in zip(w.values, n.values)))

    print('\n== availability audit (share non-null on the B0 pre-book signal set) ==')
    _, pre = cell(bk, return_pre=True)
    for c in ('dist_open_pct', 'rv_profile', 'adv20', 'stop_b', 'r_pct_b', 'next_open', 'level',
              'spread_mean', 'ask_dec'):
        print(f'  {c:<16s} {pre[c].notna().mean():7.2%}')
    print(f'  measured-NBBO share of B0 book rows: {1 - books["B0 shipped"].imputed.mean():.2%}')
    # --- the live dry run, 9/14-9/18, scored the same way (forward, not a cell)
    try:
        f = pd.read_csv(f'{ROOT}/research/bf_zero/causal_filter/forward_book.csv', dtype={'symbol': str})
        print(f'\n== forward dry run (4 live sessions, the engine\'s own book, measured spreads) ==')
        print(f'  booked {len(f)} trades over {f.day.nunique()} sessions | grossR {f.rr.mean():+.3f} | '
              f'netR {f.net.mean():+.3f} | total {f.net.sum()*RISK:+,.0f} $ at $100 risk | '
              f'WR {(f.rr>0).mean()*100:.1f}% | green sessions {(f.groupby("day").net.sum()>0).sum()}/{f.day.nunique()}')
        v = f[(f.em >= 615) & (f.em < 624)]
        print(f'  the declared D5 window holds {len(v)} of them, {v.net.sum():+.2f} R -> the veto would have '
              f'moved the forward book by {-v.net.sum()*RISK:+,.0f} $')
    except Exception as e:
        print(f'  forward book unavailable: {type(e).__name__} {e}')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
