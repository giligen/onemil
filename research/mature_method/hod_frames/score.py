#!/usr/bin/env python3
"""hod_frames — the scorer.  Cells exactly as declared in PREREG.md (commit ecf0682, before any
cell was scored).  TEST is sealed: no TEST number unless FREEZE.md exists AND --test is passed.
Read-only on every DB.  One process."""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S            # noqa: E402
import score2 as S2          # noqa: E402
from research.scripts.pit_listings import is_test_ticker   # noqa: E402

D = f'{ROOT}/research/mature_method/hod_frames'
RD = dict(dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
WITH_TEST = '--test' in sys.argv and os.path.exists(f'{D}/FREEZE.md')
S.SPLITS = S2.SPLITS = SPLITS = ('TRAIN', 'VAL', 'TEST') if WITH_TEST else ('TRAIN', 'VAL')
TS, XS = (660, 720, 780), (6.0, 7.0, 8.0)
TLAB = {660: '11:00', 720: '12:00', 780: '13:00'}
HDR = ('| cell                              | split | n     | /wk   | grossR | netR   |  t    | tc    |'
       ' green | rs | worst $  | total $   | MDD $    | ex5    | imp |')
SEP = '|' + '|'.join(['-' * 6] * 15) + '|'
CELLS, BOOKS = [], {}


def clustered_t(d, col='net'):
    if len(d) < 3:
        return np.nan
    x = d[col].values.astype(float); mu = x.mean()
    g = pd.Series(x - mu).groupby(d.day.values).sum().values
    se = np.sqrt((g ** 2).sum()) / len(x)
    return float(mu / se) if se > 0 else np.nan


def halves(b):
    tr = b[b.split == 'TRAIN']
    parts = (tr[tr.day < '2025-07-01'], tr[tr.day >= '2025-07-01'], b[b.split == 'VAL'])
    g = [float(x.rr.mean()) if len(x) else np.nan for x in parts]
    return g, [len(x) for x in parts], all(v == v and v > 0 for v in g)


def fmt(name, b):
    out = [HDR, SEP] if not CELLS else []
    for sp in SPLITS:
        w = S.week_stats(b, sp); d = b[b.split == sp]
        out.append(f'| {name:<33s} | {sp:5s} | {w["n"]:5d} | {w["per_wk"]:5.1f} | {w["gross"]:+.3f} | '
                   f'{w["net"]:+.3f} | {w["t"]:+5.2f} | {clustered_t(d):+5.2f} | {w["green"]:5.1f} | '
                   f'{w["redstreak"]:2d} | {w["worst"]:8.0f} | {w["total"]:9.0f} | {w["mdd"]:8.0f} | '
                   f'{w["ex5"]:+.3f} | {w["imp"]:3.0f} |')
    print('\n'.join(out), flush=True)


def show(name, b, note=''):
    BOOKS[name] = b
    fmt(name, b)
    g, n, ok = halves(b)
    for sp in SPLITS:
        w = S.week_stats(b, sp); d = b[b.split == sp]
        w.update(cell=name, split=sp, tc=clustered_t(d),
                 cost=float((d.rr - d.net).mean()) if len(d) else np.nan,
                 h1=g[0], h2=g[1], vgross=g[2], half_ok=ok, note=note)
        CELLS.append(w)
    print(f'    halves H1 {g[0]:+.3f} (n {n[0]}) | H2 {g[1]:+.3f} (n {n[1]}) | VAL {g[2]:+.3f} '
          f'(n {n[2]}) -> same-signed POSITIVE in all three: {ok}', flush=True)
    return b


# ------------------------------------------------------------------ Frame 1 cost / cascade
def attach_short(x, tgt):
    """The programme's cost model, applied to the SHORT leg: price = the short fill, R = the short's
    own R, the same half-spread-in + RATIO*half-out contract."""
    x = x.copy()
    x['price'] = x.short_px; x['r_pct'] = x.r_pct_s; x['stop'] = x.short_stop
    x['rr'] = x[f'rr_{tgt}']; x['why'] = x[f'why_{tgt}']; x['exit_m'] = x[f'exit_m_{tgt}']
    pb = pd.cut(x.price, S.PB_EDGES, labels=S.PB_LAB)
    hb = pd.cut(x.short_m, S.HB_EDGES, labels=S.HB_LAB)
    sp = x.spread_mean / x.price * 100
    x['imputed'] = sp.isna()
    imp = pd.Series([S.IMPUTE.get((p, h), S.IMPUTE_GLOBAL) for p, h in zip(pb, hb)], index=x.index)
    x['sp_pct'] = sp.fillna(imp).fillna(S.IMPUTE_GLOBAL)
    ratio = x.why.map(S.RATIO).fillna(0.875)
    half = 0.5 * x.sp_pct / x.r_pct.clip(lower=0.05)
    x['net'] = x.rr - half - half * ratio
    x['netb'] = x['net']
    return x


def short_set(sh, trig, sv, tgt, r_min=1.0, min_price=20.0, max_bps=100.0, max_frac_r=0.15,
              obtain=True):
    d = sh[(sh.trig == trig) & (sh.stop_var == sv) & sh[f'rr_{tgt}'].notna() &
           (sh.r_pct_s >= r_min) & (sh.short_px >= min_price)]
    x = attach_short(d, tgt)
    if max_bps:
        x = x[(x.sp_pct * 100) <= max_bps]
    if max_frac_r:
        x = x[(x.sp_pct / x.r_pct.clip(lower=0.05)) <= max_frac_r]
    if obtain:
        x = x[x.obtainable == 1]
    return x


def bookit(s, nday=12, nconc=4):
    if not len(s):
        return s.assign(pnl=pd.Series(dtype=float))
    t = s.rename(columns={'entry_m': 'long_entry_m', 'short_m': 'entry_m'})
    return S.apply_book(t, nday, nconc)


# ------------------------------------------------------------------ main
def main():
    print('== hod_frames — three new frames on HOD-break ==')
    print(f'   splits scored: {SPLITS}   (TEST sealed: {not WITH_TEST})\n', flush=True)

    pop = S2.load_pop(); S.build_impute(pop)
    sig = S2.sig_set(pop, **S2.BASES['B2'])
    b2 = S.apply_book(sig, 12, 4)
    print('== STEP 1 — REPRODUCTION GATE (ref B2: 1,622/30.6/-0.039/-0.107/32.1%/-17,346 | '
          '706/30.7/+0.083/+0.013/43.5%/+893) ==')
    show('R1 B2 shipped (long)', b2)

    df = pd.read_csv(f'{ROOT}/research/mature_method/hod_preopen_regime/day_fields.csv',
                     dtype={'day': str}, keep_default_na=False, na_values=[''])[['day', 'spy_r5_pct']]
    rng = pd.read_csv(f'{D}/range.csv', **RD)
    bf = pd.read_csv(f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv',
                     dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    BORROW = set(bf[(bf.shortable.astype(str) == 'True') &
                    (bf.easy_to_borrow.astype(str) == 'True')].symbol)
    nb = pd.read_csv(f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv', **RD
                     ).drop_duplicates(['day', 'symbol', 'entry_m'])

    # ================= FRAME 1 — the short book =========================================
    print('\n\n================ FRAME 1 — SHORT the failed break ================')
    sh = pd.read_csv(f'{D}/short.csv', **RD)
    n0 = len(sh)
    sh = sh[~sh.day.isin(S.EARLY_CLOSE)]
    sh = sh[~sh.symbol.map(is_test_ticker)]
    sh['split'] = S.split_of(sh.day.values)
    sh['wk'] = pd.to_datetime(sh.day).dt.to_period('W-FRI').astype(str)
    sh = sh.merge(nb[['day', 'symbol', 'entry_m', 'spread_mean']].rename(
        columns={'entry_m': 'short_m'}), on=['day', 'symbol', 'short_m'], how='left')
    sh = sh.merge(df, on='day', how='left')
    sh['borrow'] = sh.symbol.isin(BORROW)
    sh = sh.sort_values(['day', 'symbol', 'short_m'], kind='mergesort').reset_index(drop=True)
    print(f'short rows {n0} -> {len(sh)} after membership; trigger mix '
          f'{sh.groupby(["trig"]).size().to_dict()}')
    u = sh.drop_duplicates(['day', 'symbol', 'entry_m', 'trig'])
    print(f'BORROW: tradeable share of short SIGNALS (shortable AND easy_to_borrow, TODAY\'s Alpaca '
          f'snapshot) = {u.borrow.mean():.1%}  ({u.borrow.sum()} of {len(u)} signals, '
          f'{u[u.borrow].symbol.nunique()} of {u.symbol.nunique()} symbols)')
    print(f'OBTAINABLE (next open >= trigger close x 0.994): {u.obtainable.mean():.1%}')
    # the unfilled counterfactual, on the fb x mfe x tls design
    cf = sh[(sh.trig == 'fb') & (sh.stop_var == 'mfe') & sh.rr_tls.notna()]
    for sp in SPLITS:
        d = cf[cf.split == sp]
        print(f'  unfilled counterfactual {sp}: obtainable n {int((d.obtainable==1).sum())} '
              f'grossR {d[d.obtainable==1].rr_tls.mean():+.3f} | NOT obtainable n '
              f'{int((d.obtainable==0).sum())} grossR {d[d.obtainable==0].rr_tls.mean():+.3f}')

    long_keys = set(zip(b2.day, b2.symbol, b2.entry_m))
    sh['on_booked_long'] = [k in long_keys for k in zip(sh.day, sh.symbol, sh.entry_m)]

    print('\n-- (a) reversal on a long the book TOOK, + (b) the failed break on it --')
    for trig in ('a01', 'a00', 'am1', 'fb'):
        for tgt in ('tls', 't2'):
            s = short_set(sh[sh.on_booked_long], trig, 'mfe', tgt)
            show(f'F1{"b" if trig == "fb" else "a"} {trig}/mfe/{tgt}', bookit(s))
    print('\n-- (c) the PURE short book, whole B2 population --')
    for sv in ('mfe', 'h05', 'h10'):
        for tgt in ('t1', 't2', 'tls'):
            show(f'F1c fb/{sv}/{tgt}', bookit(short_set(sh, 'fb', sv, tgt)))
    print('\n-- (d) (c) on SPY-DOWN days (spy_r5_pct < 0, known 09:35:00) --')
    for sv in ('mfe', 'h05', 'h10'):
        for tgt in ('t1', 't2', 'tls'):
            show(f'F1d fb/{sv}/{tgt} SPYdn',
                 bookit(short_set(sh[sh.spy_r5_pct < 0], 'fb', sv, tgt)))
    print('\n-- (c) on the BORROWABLE subset only --')
    for sv in ('mfe', 'h05', 'h10'):
        for tgt in ('t1', 't2', 'tls'):
            show(f'F1c-bor fb/{sv}/{tgt}', bookit(short_set(sh[sh.borrow], 'fb', sv, tgt)))

    # ================= FRAME 2 — the noon conditional-mover book =======================
    print('\n\n================ FRAME 2 — the noon conditional-mover book ================')
    R = rng.set_index(['day', 'symbol'])
    print('\nConditional hit rate P(EOD RTH range >= 10% | range >= X at T) on the signal population')
    print('| T | X | signals | P(EOD>=10%) | base rate | med EOD range |')
    print('|---|---|---|---|---|---|')
    F2POP = {}
    for T in TS:
        d = pop[(pop.stop_n.notna()) & (pop.dist_open_pct >= 5.0) & (pop.entry_m > T) &
                (pop.entry_m <= 841)].drop_duplicates(['day', 'symbol'], keep='first')
        d = d[(d.fill_capped == 1) & (d.r_pct_n >= 1.0) & (d.next_open >= 20.0)]
        x = S.attach_cost(d, 'n')
        x = x[((x.sp_pct * 100) <= 100.0) & ((x.sp_pct / x.r_pct.clip(lower=0.05)) <= 0.15) &
              x.obtainable.astype(bool)]
        x = x.join(R[[f'rng_{T}', 'rng_day']], on=['day', 'symbol'])
        x = x[x.split.isin(SPLITS)]
        F2POP[T] = x
        base = float((x.rng_day >= 10).mean())
        for X in XS:
            q = x[x[f'rng_{T}'] >= X]
            hr = float((q.rng_day >= 10).mean()) if len(q) else np.nan
            print(f'| {TLAB[T]} | {X:.0f}% | {len(q)} | **{hr:.1%}** | {base:.1%} | '
                  f'{q.rng_day.median():.1f}% |')
    print('\n-- the 9 cells --')
    for T in TS:
        for X in XS:
            x = F2POP[T]
            show(f'F2 T{TLAB[T]} X{X:.0f}%', S.apply_book(x[x[f'rng_{T}'] >= X], 12, 4))

    # ================= FRAME 3 — size by regime ========================================
    print('\n\n================ FRAME 3 — size by regime, frequency preserved ================')
    b = b2.join(df.set_index('day'), on='day')
    b = b.join(R[['rng_720']], on=['day', 'symbol'])
    SIZERS = {'S1 1.5up/0.5dn': (1.5, 0.5, 'spy'), 'S2 2.0up/0.5dn': (2.0, 0.5, 'spy'),
              'S3 2.0up/0.0dn [ctrl]': (2.0, 0.0, 'spy'), 'S4 rng12>=7% 2.0/0.5': (2.0, 0.5, 'rng')}
    print('| cell | split | n | /wk | grossR (must equal base) | weekly $ | green % | worst $ | '
          'total $ | MDD $ | rs |')
    print('|' + '|'.join(['---'] * 11) + '|')
    base_gross = {sp: float(b2[b2.split == sp].rr.mean()) for sp in SPLITS}
    F3 = {}
    for nm, (mu, md, key) in SIZERS.items():
        z = b.copy()
        k = (z.spy_r5_pct > 0) if key == 'spy' else (z.rng_720 >= 7.0)
        z['pnl'] = z.net * S.RISK * np.where(k.fillna(False), mu, md)
        F3[nm] = z
        for sp in SPLITS:
            w = S.week_stats(z, sp)
            print(f'| {nm} | {sp} | {w["n"]} | {w["per_wk"]:.1f} | {w["gross"]:+.4f} '
                  f'(base {base_gross[sp]:+.4f}) | {w["total"]/S.NW[sp]:+.0f} | {w["green"]:.1f} | '
                  f'{w["worst"]:.0f} | {w["total"]:.0f} | {w["mdd"]:.0f} | {w["redstreak"]} |')
    for sp in SPLITS:
        w = S.week_stats(b2, sp)
        print(f'| BASE 1.0x | {sp} | {w["n"]} | {w["per_wk"]:.1f} | {w["gross"]:+.4f} | '
              f'{w["total"]/S.NW[sp]:+.0f} | {w["green"]:.1f} | {w["worst"]:.0f} | {w["total"]:.0f} '
              f'| {w["mdd"]:.0f} | {w["redstreak"]} |')

    print('\n-- Frame 3 null: the sizing KEY shuffled across days, pick set fixed (2,000 draws) --')
    rgen = np.random.default_rng(11)
    for nm, (mu, md, key) in SIZERS.items():
        z = F3[nm]
        for sp in SPLITS:
            d = z[z.split == sp]
            if not len(d):
                continue
            dk = d.groupby('day').apply(
                lambda g: bool((g.spy_r5_pct > 0).iloc[0]) if key == 'spy'
                else bool((g.rng_720 >= 7.0).mean() >= 0.5), include_groups=False)
            days = dk.index.values; kv = dk.values
            obsw = S.week_stats(d, sp)
            gt, gr = [], []
            for _ in range(2000):
                mp = dict(zip(days, rgen.permutation(kv)))
                p = d.net * S.RISK * np.where(d.day.map(mp).fillna(False).values, mu, md)
                w = p.groupby(d.wk).sum().reindex(S.ALL_WEEKS[sp]).fillna(0.0)
                gt.append(w.sum()); gr.append((w > 0).mean() * 100)
            print(f'  {nm:24s} {sp:5s} total$ obs {obsw["total"]:+8.0f} null mean {np.mean(gt):+8.0f} '
                  f'[{np.percentile(gt,5):+.0f}, {np.percentile(gt,95):+.0f}]   green% obs '
                  f'{obsw["green"]:.1f} null {np.mean(gr):.1f} [{np.percentile(gr,5):.1f}, '
                  f'{np.percentile(gr,95):.1f}]', flush=True)

    # ================= nulls + bars ====================================================
    print('\n\n== count-matched permutation null on green weeks (Frames 1 and 2) ==')
    print('| cell | split | observed green % | null mean | [p5, p95] | outside? |')
    print('|---|---|---|---|---|---|')
    nulls = []
    for nm, bb in BOOKS.items():
        for sp in SPLITS:
            obs, mu_, p5, p95 = S.null_band(bb, sp)
            o = 'ABOVE' if obs == obs and obs > p95 else ('below' if obs == obs and obs < p5 else 'inside')
            print(f'| {nm} | {sp} | {obs:.1f} | {mu_:.1f} | [{p5:.1f}, {p95:.1f}] | {o} |')
            nulls.append(dict(cell=nm, split=sp, obs=obs, mu=mu_, p5=p5, p95=p95, outside=o))
    pd.DataFrame(nulls).to_csv(f'{D}/nulls.csv', index=False)
    cf = pd.DataFrame(CELLS); cf.to_csv(f'{D}/cells.csv', index=False)

    print('\n== BOTH BARS ==')
    g1 = cf[(cf.split == 'TRAIN') & (cf.net > 0) & (cf.t >= 2.0) & (cf.tc >= 2.0) & (cf.per_wk >= 10)]
    print(f'G1 (TRAIN net>0, iid t>=2, CLUSTERED t>=2, >=10/wk): {len(g1)} of '
          f'{cf[cf.split=="TRAIN"].cell.nunique()} cells -> {list(g1.cell)}')
    ship = []
    for nm in cf.cell.unique():
        t_ = cf[(cf.cell == nm) & (cf.split == 'TRAIN')].iloc[0]
        v_ = cf[(cf.cell == nm) & (cf.split == 'VAL')].iloc[0]
        if (t_.total > 0 and v_.total > 0 and t_.green >= 50 and v_.green >= 50 and
                min(t_.per_wk, v_.per_wk) >= 10 and t_.tc >= 2.0 and t_.half_ok):
            ship.append(nm)
    print(f'LIVE-EXPLORATION BAR ($>0 both, green>=50% both, >=10/wk both, clustered t>=2 TRAIN, '
          f'halves same-signed): {len(ship)} -> {ship}')
    print('\nMDE (80% power, per trade, net) per cell:')
    for nm, bb in BOOKS.items():
        row = ' '.join(f'{sp} {2.80*bb[bb.split==sp].net.std(ddof=1)/np.sqrt(max(len(bb[bb.split==sp]),1)):.3f}'
                       for sp in SPLITS if len(bb[bb.split == sp]) > 2)
        print(f'  {nm:34s} {row}')


if __name__ == '__main__':
    main()
