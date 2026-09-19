#!/usr/bin/env python3
"""hod_fresh — the scorer.  Cells exactly as declared in PREREG.md (commit b1e5470).

TEST is sealed: no TEST number unless FREEZE.md exists AND --test is passed.
Read-only on every DB.  One process.
"""
import os, sqlite3, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S                                              # noqa: E402
import score2 as S2                                            # noqa: E402
from research.scripts.pit_listings import is_test_ticker       # noqa: E402

D = f'{ROOT}/research/mature_method/hod_fresh'
RD = dict(dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
WITH_TEST = '--test' in sys.argv and os.path.exists(f'{D}/FREEZE.md')
S.SPLITS = ('TRAIN', 'VAL', 'TEST') if WITH_TEST else ('TRAIN', 'VAL')
S2.SPLITS = S.SPLITS
SPLITS = S.SPLITS
EARLY_CLOSE = {'2025-07-03', '2025-11-28', '2025-12-24'}
STOPS = ('n5', 'n3', 'b', 'bb', 'a05', 'a10')
STOP_LABEL = {'n5': 'last-5-bar low (the B2 stop)', 'n3': 'last-3-bar low',
              'b': 'shipped consolidation low K5/X4%', 'bb': "the breakout bar's own low",
              'a05': 'entry - 0.5 x ATR20d', 'a10': 'entry - 1.0 x ATR20d'}
RUNGS = ('base', 'le3', 'le5', 'le8', 'le12', 'ge20')
RUNG_LABEL = {'base': 'no age condition (= B2)', 'le3': 'consol_bars <= 3', 'le5': 'consol_bars <= 5',
              'le8': 'consol_bars <= 8', 'le12': 'consol_bars <= 12', 'ge20': 'consol_bars >= 20 [CONTROL]'}

HDR = ('| cell                         | split | n     | /wk   | grossR | netR   | net(b) |  t    | '
       'green | flat  | rs | worst $  | total $   | MDD $    | gr mo | ex5    | imp |')
SEP = '|' + '|'.join(['-' * 6] * 17) + '|'
BOOKS, CELLS = {}, []


# ---------------------------------------------------------------- statistics
def clustered_t(d, col='net'):
    """Cluster-robust t of the mean, clusters = trading DAYS (hod_preopen_regime §4's rule)."""
    if len(d) < 3:
        return np.nan
    x = d[col].values.astype(float)
    mu = x.mean()
    r = x - mu
    g = pd.Series(r).groupby(d.day.values).sum().values
    se = np.sqrt((g ** 2).sum()) / len(x)
    return float(mu / se) if se > 0 else np.nan


def halves(b):
    """(H1-2025, H2-2025, VAL) gross means + the pre-committed same-sign verdict."""
    tr = b[b.split == 'TRAIN']
    h1 = tr[tr.day < '2025-07-01']; h2 = tr[tr.day >= '2025-07-01']; va = b[b.split == 'VAL']
    g = [float(x.rr.mean()) if len(x) else np.nan for x in (h1, h2, va)]
    n = [len(x) for x in (h1, h2, va)]
    ok = all(v == v and v > 0 for v in g)
    return g, n, ok


def mde(d, col='net'):
    return 2.80 * float(d[col].std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan


def show(name, b, note=''):
    BOOKS[name] = b
    print(S.fmt_cell(name, b))
    g, n, ok = halves(b)
    for sp in SPLITS:
        w = S.week_stats(b, sp)
        d = b[b.split == sp]
        w.update(cell=name, split=sp, tc=clustered_t(d), cost=float((d.rr - d.net).mean()) if len(d) else np.nan,
                 r_pct_med=float(d.r_pct.median()) if len(d) else np.nan,
                 big_range=float((d.day_range_pct >= 10).mean() * 100) if len(d) else np.nan,
                 h1=g[0], h2=g[1], vgross=g[2], half_ok=ok, mde=mde(d), note=note)
        CELLS.append(w)
    print(f'    halves  H1 {g[0]:+.3f} (n {n[0]}) | H2 {g[1]:+.3f} (n {n[1]}) | VAL {g[2]:+.3f} '
          f'(n {n[2]})  -> same-signed POSITIVE in all three: {ok}')
    for sp in SPLITS:
        d = b[b.split == sp]
        if len(d) > 2:
            print(f'    {sp:5s} iid t {d.net.mean()/(d.net.std(ddof=1)/np.sqrt(len(d))):+.2f}  '
                  f'CLUSTERED t {clustered_t(d):+.2f}  cost/R {float((d.rr-d.net).mean()):.4f}  '
                  f'medianR% {float(d.r_pct.median()):.2f}  >=10%-range days {float((d.day_range_pct>=10).mean()):.0%}  '
                  f'MDE {mde(d):.3f}')
    return b


# ---------------------------------------------------------------- populations
def sig_set4(p, rung='base', stop='n5', min_price=20.0, r_min=1.0, max_bps=100.0,
             max_frac_r=0.15, obtain=True, last_m=840):
    """PREREG §1's live cascade on the hod_fresh pass, for one (admission rung x stop design)."""
    d = p[(p[f'first_{rung}'] == 1) & (p.entry_m <= last_m + 1)]
    d = d[d[f'r_pct_{stop}'].notna() & (d.fill_capped == 1) &
          (d[f'r_pct_{stop}'] >= r_min) & (d.next_open >= min_price)]
    x = S.attach_cost(d, stop)
    if max_bps:
        x = x[(x.sp_pct * 100) <= max_bps]
    if max_frac_r:
        x = x[(x.sp_pct / x.r_pct.clip(lower=0.05)) <= max_frac_r]
    if obtain:
        x = x[x.obtainable.astype(bool)]
    return x


def book(p, **kw):
    return S.apply_book(sig_set4(p, **kw), 12, 4)


def pick(rows, label):
    """The PREREG §2 selector: eligible = >=10/wk on BOTH splits AND gross positive in H1, H2, VAL;
    rank by min(green% TRAIN, green% VAL), tie-break min(TRAIN gross, VAL gross).  If none eligible,
    fall back to max min(TRAIN gross, VAL gross) at >=10/wk and SAY SO."""
    tab = []
    for nm in rows:
        b = BOOKS[nm]
        wt, wv = S.week_stats(b, 'TRAIN'), S.week_stats(b, 'VAL')
        g, _n, ok = halves(b)
        tab.append(dict(cell=nm, wk_t=wt['per_wk'], wk_v=wv['per_wk'], grn_t=wt['green'],
                        grn_v=wv['green'], gr_t=wt['gross'], gr_v=wv['gross'], half_ok=ok))
    t = pd.DataFrame(tab)
    freq = t[(t.wk_t >= 10) & (t.wk_v >= 10)]
    el = freq[freq.half_ok]
    print(f'\n  -- SELECTOR ({label}) -- at >=10/wk both splits: {list(freq.cell)}; '
          f'of those half-consistent-positive: {list(el.cell)}')
    if len(el):
        el = el.assign(k=el[['grn_t', 'grn_v']].min(axis=1), k2=el[['gr_t', 'gr_v']].min(axis=1))
        w = el.sort_values(['k', 'k2'], ascending=False).iloc[0]
        print(f'  -- SELECTED {w.cell} (eligible)')
        return str(w.cell), True
    src = freq if len(freq) else t
    src = src.assign(k2=src[['gr_t', 'gr_v']].min(axis=1))
    w = src.sort_values('k2', ascending=False).iloc[0]
    print(f'  -- NO ELIGIBLE CELL (eligibility failed). Carrying {w.cell} for completeness only — '
          f'a cell carried this way can never clear a bar.')
    return str(w.cell), False


def main():
    print('== hod_fresh — the fresh-high admission + stop redesign ==', flush=True)
    print(f'   splits scored: {SPLITS}   (TEST sealed: {not WITH_TEST})\n', flush=True)

    # ================= STEP 1 — REPRODUCTION GATE ==========================================
    print('== STEP 1 — REPRODUCTION GATE ==')
    pop = S2.load_pop()
    S.build_impute(pop)                       # ONE impute model, built exactly as score2 does
    b0 = S.apply_book(S2.sig_set(pop, **S2.BASES['B0']), 12, 4)
    print('\nR1  B0 shipped (ref 1,688/-0.027/-0.088/41.5%/-14,835 | 820/+0.016/-0.050/43.5%/-4,128)')
    print(HDR + '\n' + SEP)
    print(S.fmt_cell('R1 B0 shipped', b0))

    path = pd.read_csv(f'{ROOT}/research/mature_method/hod_losers/path.csv', **RD)
    cb = path[path.tag == 'B0'].drop_duplicates(['day', 'symbol', 'entry_m'])[
        ['day', 'symbol', 'entry_m', 'consol_bars', 'touch_n']]
    s0 = S2.sig_set(pop, **S2.BASES['B0']).merge(cb, on=['day', 'symbol', 'entry_m'], how='left')
    p13 = S.apply_book(s0[s0.consol_bars < 8], 12, 4)
    print('\nR2  P13 = B0 ^ consol_bars<8 (ref 297/5.6/-0.038/-0.104/35.8%/-3,099 | '
          '219/9.5/+0.207/+0.136/52.2%/+2,977)')
    print(S.fmt_cell('R2 P13 consol_bars<8', p13))

    # ---- the hod_fresh pass ---------------------------------------------------------------
    print('\nloading sig3.csv ...', flush=True)
    p = pd.read_csv(f'{D}/sig3.csv', **RD)
    n0 = len(p)
    p = p[~p.day.isin(EARLY_CLOSE)]
    p = p[~p.symbol.map(is_test_ticker)]
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    dbs = set(pd.read_sql('select distinct symbol from daily_bars', con).symbol.astype(str))
    con.close()
    p = p[p.symbol.isin(dbs)].reset_index(drop=True)
    print(f'membership: {n0} -> {len(p)} rows, {p.day.nunique()} days')
    p['split'] = S.split_of(p.day.values)
    p['wk'] = pd.to_datetime(p.day).dt.to_period('W-FRI').astype(str)
    nb = pd.read_csv(f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv', **RD
                     ).drop_duplicates(['day', 'symbol', 'entry_m'])
    p = p.merge(nb[['day', 'symbol', 'entry_m', 'spread_mean', 'ask_dec', 'bid_dec', 'n_sig']],
                on=['day', 'symbol', 'entry_m'], how='left')
    # day-level: SPY 09:30->09:35 (D2, known 09:35:00) and the >=10%-range-day cohort marker
    df = pd.read_csv(f'{ROOT}/research/mature_method/hod_preopen_regime/day_fields.csv',
                     dtype={'day': str}, keep_default_na=False, na_values=[''])
    p = p.merge(df[['day', 'spy_r5_pct']], on='day', how='left')
    u = pd.read_csv(f'{ROOT}/research/bf_zero/universe.csv',
                    usecols=['symbol', 'bar_date', 'open', 'high', 'low'],
                    dtype={'symbol': str, 'bar_date': str}, keep_default_na=False, na_values=[''])
    u = u.rename(columns={'bar_date': 'day'})
    for k in ('open', 'high', 'low'):
        u[k] = pd.to_numeric(u[k], errors='coerce')
    u['day_range_pct'] = (u.high - u.low) / u.open * 100.0
    p = p.merge(u[['symbol', 'day', 'day_range_pct']].drop_duplicates(['symbol', 'day']),
                on=['symbol', 'day'], how='left')
    p = p.sort_values(['day', 'symbol', 'entry_m'], kind='mergesort').reset_index(drop=True)

    print('\nR3  hod_fresh base rung x n5 stop == B2 (ref 1,622/-0.039/-0.107/32.1%/-17,346 | '
          '706/+0.083/+0.013/43.5%/+893)')
    r3 = show('R3 base x n5 (= B2)', book(p, rung='base', stop='n5'))

    # ---- availability audit on the two NEW decision fields ---------------------------------
    print('\n== AVAILABILITY AUDIT (the two new decision fields, on the base pre-book set) ==')
    base_pre = sig_set4(p, rung='base', stop='n5')
    for f in ('consol_bars', 'atr20d'):
        v = pd.to_numeric(base_pre[f], errors='coerce')
        win = base_pre.rr > 0
        print(f'  {f:14s} cov {v.notna().mean():6.1%}  miss win {v[win].isna().mean():6.1%}  '
              f'miss loss {v[~win].isna().mean():6.1%}  '
              f'{"<-- OUTCOME-DEPENDENT" if abs(v[win].isna().mean()-v[~win].isna().mean())>0.05 else ""}')
    print(f'  consol_bars is computed from bars STRICTLY BEFORE the break bar; atr20d from the 20 '
          f'DAILY sessions strictly before the day.')
    print(f'  consol_bars distribution (base pre-book): ' +
          ' '.join(f'p{q}={base_pre.consol_bars.quantile(q/100):.0f}' for q in (10, 25, 50, 75, 90)))

    # ================= FAMILY A — the fresh-high admission ladder ===========================
    print('\n== FAMILY A — ADMISSION LADDER (scan rule: KEEP-SCANNING; stop = last-5-bar low) ==')
    print(HDR + '\n' + SEP)
    A = []
    for r in RUNGS:
        nm = f'A {r} [{RUNG_LABEL[r]}]'
        show(nm, book(p, rung=r, stop='n5'))
        A.append(nm)
    best_a, elig_a = pick(A, 'family A')
    rung_a = best_a.split()[1]

    # ================= FAMILY B — stop design ==============================================
    print(f'\n== FAMILY B — STOP DESIGN on the selected rung ({rung_a}) ==')
    print('| stop                              | split | n   | /wk  | med R% of px | cost/R  | '
          'grossR  | netR    | green | total $ |')
    B = []
    adm = p[(p[f'first_{rung_a}'] == 1) & (p.entry_m <= 841)]
    for st in STOPS:
        nm = f'B {st} [{STOP_LABEL[st]}] x {rung_a}'
        b = book(p, rung=rung_a, stop=st)
        BOOKS[nm] = b
        n_adm = len(adm)
        n_r = int((adm[f'r_pct_{st}'].notna() & (adm[f'r_pct_{st}'] >= 1.0)).sum())
        n_pre = len(sig_set4(p, rung=rung_a, stop=st))
        print(f'|   cascade: admitted {n_adm} -> stop computable & r>=1% {n_r} '
              f'-> after fill-cap/price/cost/obtainable {n_pre}')
        for sp in SPLITS:
            w = S.week_stats(b, sp); d = b[b.split == sp]
            print(f'| {STOP_LABEL[st]:<33s} | {sp:5s} | {w["n"]:4d} | {w["per_wk"]:4.1f} | '
                  f'{float(d.r_pct.median()) if len(d) else np.nan:12.2f} | '
                  f'{float((d.rr-d.net).mean()) if len(d) else np.nan:7.4f} | {w["gross"]:+7.3f} | '
                  f'{w["net"]:+7.3f} | {w["green"]:5.1f} | {w["total"]:+8.0f} |')
        B.append(nm)
    print('\n' + HDR + '\n' + SEP)
    for nm in B:
        show(nm, BOOKS[nm])
    best_b, elig_b = pick(B, 'family B')
    stop_b = best_b.split()[1]

    # ================= FAMILY C — the D2 day gate ==========================================
    print(f'\n== FAMILY C — D2 (SPY 09:30->09:35, known 09:35:00) on {rung_a} x {stop_b} ==')
    print(HDR + '\n' + SEP)
    C = []
    base_ab = sig_set4(p, rung=rung_a, stop=stop_b)
    for lab, thr in (('C1 spy_r5>0', 0.0), ('C2 spy_r5>=+0.2%', 0.2), ('C3 spy_r5>=+0.4%', 0.4)):
        m = base_ab.spy_r5_pct > thr if thr == 0.0 else base_ab.spy_r5_pct >= thr
        show(lab, S.apply_book(base_ab[m.fillna(False)], 12, 4))
        C.append(lab)
    best_c, elig_c = pick(C, 'family C')

    # ================= FAMILY D — the two declared interactions =============================
    print(f'\n== FAMILY D — the two declared interactions ==')
    print(HDR + '\n' + SEP)
    show('D1 A x B x rv>=5', S.apply_book(base_ab[base_ab.rv_profile >= 5.0], 12, 4))
    mC = (base_ab.spy_r5_pct > 0.0) if best_c.startswith('C1') else (
        base_ab.spy_r5_pct >= (0.2 if best_c.startswith('C2') else 0.4))
    abc = base_ab[mC.fillna(False)]
    show('D2 A x B x C x spread<=8%R',
         S.apply_book(abc[(abc.sp_pct / abc.r_pct.clip(lower=0.05)) <= 0.08], 12, 4))

    # ================= diagnostics ==========================================================
    print('\n== UNFILLED COUNTERFACTUAL (selected admission rung, before the cap gate) ==')
    raw = p[(p[f'first_{rung_a}'] == 1) & (p.entry_m <= 841) & p[f'r_pct_{stop_b}'].notna() &
            (p[f'r_pct_{stop_b}'] >= 1.0) & (p.next_open >= 20.0)]
    for sp in SPLITS:
        d = raw[raw.split == sp]
        f_ = d[d.fill_capped == 1]; nf = d[d.fill_capped == 0]
        print(f'  {sp:5s} filled {len(f_)} ({len(f_)/max(len(d),1):.1%}) gross {f_[f"rr_{stop_b}"].mean():+.3f} | '
              f'UNFILLED {len(nf)} gross-if-paid-anyway {nf[f"rr_{stop_b}"].mean():+.3f}')

    print('\n== COUNT-MATCHED PERMUTATION NULL (2,000 draws, per-week pick count fixed) ==')
    print('| cell | split | observed green % | null mean | [p5, p95] | outside? |')
    nulls = []
    for nm, b in BOOKS.items():
        for sp in SPLITS:
            obs, mu, p5, p95 = S.null_band(b, sp)
            if obs != obs:
                continue
            o = 'ABOVE' if obs > p95 else ('below' if obs < p5 else 'inside')
            nulls.append(dict(cell=nm, split=sp, obs=obs, mu=mu, p5=p5, p95=p95, outside=o))
            print(f'| {nm} | {sp} | {obs:.1f} | {mu:.1f} | [{p5:.1f}, {p95:.1f}] | {o} |')
    pd.DataFrame(nulls).to_csv(f'{D}/nulls.csv', index=False)
    pd.DataFrame(CELLS).to_csv(f'{D}/cells.csv', index=False)

    # ================= the bars ============================================================
    print('\n== BOTH BARS ==')
    cf = pd.DataFrame(CELLS)
    cf = cf[~cf.cell.str.startswith('R3')]
    g1 = cf[(cf.split == 'TRAIN') & (cf.net > 0) & (cf.t >= 2.0) & (cf.tc >= 2.0) &
            (cf.per_wk >= 10) & (cf.gross >= 0.25)]
    print(f'G1 (TRAIN net>0, iid t>=2, CLUSTERED t>=2, >=10/wk, gross>=+0.25R): {len(g1)} of '
          f'{cf[cf.split=="TRAIN"].cell.nunique()} cells pass -> {list(g1.cell)}')
    tr = cf[cf.split == 'TRAIN']
    print(f'   best TRAIN net R {tr.net.max():+.4f} ({tr.loc[tr.net.idxmax(),"cell"]}); '
          f'best TRAIN gross {tr.gross.max():+.4f} ({tr.loc[tr.gross.idxmax(),"cell"]}); '
          f'best TRAIN clustered t {tr.tc.max():+.2f}')
    ship = []
    for nm in cf.cell.unique():
        t_ = cf[(cf.cell == nm) & (cf.split == 'TRAIN')].iloc[0]
        v_ = cf[(cf.cell == nm) & (cf.split == 'VAL')].iloc[0]
        if (t_.gross >= 0.25 and v_.gross >= 0.25 and min(t_.per_wk, v_.per_wk) >= 10 and
                v_.green >= 50 and t_.total > 0 and v_.total > 0 and t_.tc >= 2.0):
            ship.append(nm)
    print(f"SHIP BAR (gross>=+0.25R both splits, >=10/wk, VAL green>=50%, $>0 both, clustered t>=2 "
          f"TRAIN): {len(ship)} pass -> {ship}")
    print('\n== MDE (80% power, two-sided 5%) ==')
    for nm in [f'A {rung_a} [{RUNG_LABEL[rung_a]}]', best_b] + ([best_c] if best_c in BOOKS else []):
        for sp in SPLITS:
            d = BOOKS[nm][BOOKS[nm].split == sp]
            print(f'  {nm:<46s} {sp:5s} n {len(d):5d}  MDE {mde(d):.3f} R')
    for sp in SPLITS:
        d = base_pre[base_pre.split == sp]
        print(f'  {"pre-book base (rung=base, n5)":<46s} {sp:5s} n {len(d):5d}  MDE {mde(d, "rr"):.3f} R')
    print('\nDONE')


if __name__ == '__main__':
    main()
