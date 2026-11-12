"""Cell 1,742 (PREREG_1742) — anatomy of the strict M1 retest-bid fills + two causal variants (V1 hold-gated bid, V2 retest-and-reclaim).
Reuses 1737_score walkers (run_bid / walk_long / stats / pool_bid / sample_starts) UNCHANGED; conventions of PREREG_1740 (stop = min low bars j-5..j-1
floored at level*0.985, target fill+2R, costs 2 bp / half_spread+2 bp, stop slip 13 bp). Capped book 8/day, 4 concurrent. Placebo 300 draws.
Usage: python3 1742_score.py [arms_tag]  ('_b1' = corrected level file 1740_arms_b1.csv, the default; '' = literal-level 1740_arms.csv)."""
import sys, sqlite3, importlib.util, math
import numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location('s1737', '/home/ec2-user/onemil/research/hod_entry/1737_score.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
W = S.walker; ROOT = S.ROOT; OUT = S.OUT
atag = sys.argv[1] if len(sys.argv) > 1 else '_b1'
KS = (3, 5, 10)
W.wait_for_db_window()
arms = pd.read_csv(f'{OUT}/1740_arms{atag}.csv', dtype={'symbol': str, 'day': str})
arms = arms[arms.level.notna()].reset_index(drop=True)
sig = pd.read_parquet(f'{ROOT}/research/hod_exit_lab/signals.parquet')[['day', 'symbol', 'half_spread']].drop_duplicates(['day', 'symbol'])
arms = arms.merge(sig, on=['day', 'symbol'], how='left')
arms['matched_spread'] = arms.half_spread.notna()
arms['dec'] = pd.qcut(arms.level, 10, labels=False, duplicates='drop')
med = arms[arms.matched_spread].groupby('dec').half_spread.median()
arms.loc[~arms.matched_spread, 'half_spread'] = arms.loc[~arms.matched_spread, 'dec'].map(med)
b0 = pd.read_csv(f'{ROOT}/research/hod_exit_lab/b0_trades.csv').drop_duplicates(['day', 'symbol'])[['day', 'symbol', 'net_R', 'entry']].rename(columns={'net_R': 'b0_net_R', 'entry': 'b0_entry'})
arms = arms.merge(b0, on=['day', 'symbol'], how='left')


def v2_run(A, s0, lvl, stop, hs):
    """V2 retest-and-reclaim from bar s0: touch = first bar >= s0 with low < lvl (before 15:30); buy at the CLOSE of the first later bar with
    close >= lvl within 15 bars of the touch; entry cost hs + 2 bp; stop `stop`; target fill + 2R. Returns the walk_long dict (+fill_i = signal bar)."""
    h, l, c, m = A['h'], A['l'], A['c'], A['m']; n = len(m)
    T = next((k for k in range(s0, n) if m[k] < S.LAST_M and l[k] < lvl), None)
    if T is None: return dict(filled=False)
    R = next((k for k in range(T + 1, min(T + 16, n)) if m[k] < S.LAST_M and c[k] >= lvl), None)
    if R is None or R + 1 >= n: return dict(filled=False)
    entry = c[R] * (1 + S.SLIP) + hs
    r = S.walk_long(A, R + 1, entry, stop, None, hs, target_from=R + 1)
    if r.get('filled'): r['fill_i'] = R
    return r


def pool_v2(A, starts, ref_close, lvl, stop, hs):
    """V2 placebo pool: same geometry (level, stop, hs) scaled by close[s-1]/ref_close from each random start s."""
    out = []
    for s in starts:
        sc = A['c'][s - 1] / ref_close
        r = v2_run(A, s, lvl * sc, stop * sc, hs * sc)
        out.append(r['net_R'] if r.get('filled') else np.nan)
    return np.array(out)


def features(A, j, C, lvl, hs, a):
    """Anatomy features for one arm (ex-post w.r.t. the fill for retest_depth / hold_k — hypotheses only)."""
    h, l, c, m = A['h'], A['l'], A['c'], A['m']; n = len(m)
    T = next((k for k in range(C + 1, n) if m[k] < S.LAST_M and l[k] < lvl), None)
    hold_k = (T - C - 1) if T is not None else 999
    depth = (lvl - min(l[T:T + 3])) / lvl if T is not None else np.nan
    lo5 = max(j - 6, 0); pr = (max(h[lo5:j - 1]) - min(l[lo5:j - 1])) / lvl if j - 1 > lo5 else np.nan
    return dict(hold_k=hold_k, retest_depth=depth, chase=(a.b0_entry - lvl) / lvl if pd.notna(a.b0_entry) else np.nan, half_spread_f=hs,
                c_mod=m[C], bars_since_open=m[C] - 570, prior5_range=pr)


con = sqlite3.connect('file:' + ROOT + '/research/bf_zero/bars_sip.db?mode=ro', uri=True)
rng = np.random.default_rng(1742)
rows, pools, feats, nobars, noC = [], {}, [], 0, 0


def add(var, a, C, m, hs, stop_p, r, ordered):
    """Append one order row."""
    f = bool(r.get('filled'))
    rows.append(dict(var=var, day=a.day, symbol=a.symbol, entry_m=a.entry_m, level=a.level, split=a.split, half=a.half, half_spread=hs, b0_net_R=a.b0_net_R,
                     c_m=m[C], stop=stop_p, ordered=ordered, filled=f, fill_m=m[r['fill_i']] if f else np.nan, exit_m=m[r['exit_i']] if f else np.nan,
                     **{k: r.get(k, np.nan) for k in ('R', 'r_pct', 'exit_px', 'why', 'net_R')}))
    return len(rows) - 1


for ix, a in enumerate(arms.itertuples()):
    d = W.fetch_day_bars(con, a.symbol, a.day)
    if d is None: nobars += 1; continue
    A = S.arrays(d); h, l, c, m = A['h'], A['l'], A['c'], A['m']; n = len(m)
    idx = [k for k in range(n) if m[k] >= a.entry_m]
    if not idx: nobars += 1; continue
    j = idx[0]; lvl = a.level; hs = a.half_spread
    stop_p = min(min(l[max(j - 1 - 4, 0):max(j - 1, 0) + 1]), lvl * (1 - S.FLOOR))
    C = next((k for k in range(j, n) if m[k] < S.LAST_M and c[k] >= lvl), None)
    if C is None or C + 1 >= n: noC += 1; continue
    feats.append(dict(day=a.day, symbol=a.symbol, split=a.split, half=a.half, level=lvl, b0_net_R=a.b0_net_R, **features(A, j, C, lvl, hs, a)))
    # M1 (k=0) and V1 gated bids, strict + thru10
    for k in (0,) + KS:
        gate_ok = C + k + 1 < n and (k == 0 or min(l[C + 1:C + k + 1]) >= lvl) and all(c[q] < lvl * 1.02 for q in range(C + 1, C + k + 1))
        name = 'M1' if k == 0 else f'V1k{k}'
        for thr, vn in ((0.0, 'strict'), (0.001, 'thru10')):
            S.THRU = thr
            r = S.run_bid(A, C + k + 1, lvl, None, lvl * 1.02, stop_p, None, hs) if gate_ok else dict(filled=False)
            i = add(f'{name}_{vn}', a, C, m, hs, stop_p, r, gate_ok)
            if vn == 'strict' and k > 0 and gate_ok:
                st = S.sample_starts(rng, A, C + k + 2)
                pools[i] = S.pool_bid(A, st, c[C + k], lvl, None, lvl * 1.02, stop_p, None, hs) if st else np.array([])
    S.THRU = 0.0
    r = v2_run(A, C + 1, lvl, stop_p, hs)
    i = add('V2', a, C, m, hs, stop_p, r, True)
    st = S.sample_starts(rng, A, C + 2)
    pools[i] = pool_v2(A, st, c[C], lvl, stop_p, hs) if st else np.array([])
    if ix % 500 == 0: W.log(f'{ix}/{len(arms)} rows {len(rows)}')
o = pd.DataFrame(rows); o.to_csv(f'{OUT}/1742_orders{atag}.csv', index=False)
F = pd.DataFrame(feats); M1 = o[o['var'] == 'M1_strict'].set_index(['day', 'symbol'])
F['m1_filled'] = F.set_index(['day', 'symbol']).index.map(M1.filled).values
F['m1_net_R'] = F.set_index(['day', 'symbol']).index.map(M1.net_R).values
F.to_csv(f'{OUT}/1742_features{atag}.csv', index=False)
W.log(f'orders+features written; nobars={nobars} noC={noC}; scoring')


def capped(fl):
    """First-come capped book: <= 8 fills/day, <= 4 concurrent."""
    keep = []
    for day, g in fl.sort_values(['day', 'fill_m', 'symbol']).groupby('day'):
        acc = []
        for x in g.itertuples():
            if len(acc) >= 8: break
            if sum(1 for e in acc if e.exit_m > x.fill_m) >= 4: continue
            acc.append(x); keep.append(x.Index)
    return fl.loc[keep]


def groups(df):
    """Scoring groups present in df."""
    g = {}
    t = df[df.split == 'TRAIN']
    g['TRAIN_H1'] = t[t.half == 'H1']; g['TRAIN_H2'] = t[t.half == 'H2']; g['TRAIN'] = t; g['VAL'] = df[df.split == 'VAL']
    return g


def diff_t(r_hi, d_hi, r_lo, d_lo):
    """Day-clustered t of mean(hi) - mean(lo) via per-trade influence functions."""
    n1, n0 = len(r_hi), len(r_lo)
    if n1 < 5 or n0 < 5: return np.nan, np.nan
    m1, m0 = r_hi.mean(), r_lo.mean()
    psi = pd.concat([pd.Series((r_hi - m1) / n1, index=d_hi), pd.Series(-(r_lo - m0) / n0, index=d_lo)])
    se = math.sqrt((psi.groupby(level=0).sum() ** 2).sum())
    return m1 - m0, ((m1 - m0) / se if se > 0 else np.nan)


# ---- Part 1 anatomy
FEATS = ['hold_k', 'retest_depth', 'chase', 'half_spread_f', 'c_mod', 'bars_since_open', 'prior5_range']
an = []
for tgt, sel_fn, col in (('M1', lambda g: g[g.m1_filled == True], 'm1_net_R'), ('B0', lambda g: g, 'b0_net_R')):
    for gname in ('TRAIN', 'VAL'):
        g = F[F.split == gname]; g = sel_fn(g)
        for f in FEATS:
            x = g[[f, col, 'day']].dropna()
            if len(x) < 100: continue
            q = pd.qcut(x[f].rank(method='first'), 5, labels=False)
            means = [x[col][q == i].mean() for i in range(5)]; ns = [int((q == i).sum()) for i in range(5)]
            sp, t = diff_t(x[col][q == 4].values, x.day[q == 4].values, x[col][q == 0].values, x.day[q == 0].values)
            an.append(dict(target=tgt, split=gname, feature=f, **{f'q{i + 1}_mean': means[i] for i in range(5)}, **{f'q{i + 1}_n': ns[i] for i in range(5)},
                           q1_lo=x[f][q == 0].min(), q5_hi=x[f][q == 4].max(), spread_q5_q1=sp, day_t=t))
A_ = pd.DataFrame(an); A_.to_csv(f'{OUT}/1742_anatomy{atag}.csv', index=False)
for tgt in ('M1', 'B0'):
    p = A_[A_.target == tgt].pivot(index='feature', columns='split', values=['spread_q5_q1', 'day_t'])
    p['agree'] = np.sign(p[('spread_q5_q1', 'TRAIN')]) == np.sign(p[('spread_q5_q1', 'VAL')])
    p['absVALt'] = p[('day_t', 'VAL')].abs()
    W.log(f'ANATOMY {tgt} (ranked by |VAL t|)\n' + p.sort_values('absVALt', ascending=False).round(3).to_string())

# ---- Part 2
prng = np.random.default_rng(17420)
out = []
armsF = arms.copy()
for v in o['var'].unique():
    ov = o[o['var'] == v]
    for gname, sel in groups(ov).items():
        fl = sel[sel.filled]
        arms_g = groups(armsF)[gname]
        nweeks = pd.to_datetime(arms_g.day).dt.strftime('%G-%V').nunique()
        st = S.stats(fl, nweeks) if len(fl) else dict(n=0)
        r = np.sort(fl.net_R.values); k1 = max(int(math.ceil(0.01 * len(r))), 1)
        st.update(var=v, group=gname, n_arms=len(arms_g), n_c_arms=len(sel), n_orders=int(sel.ordered.sum()), n_filled=len(fl),
                  fill_share_orders=len(fl) / max(int(sel.ordered.sum()), 1), fill_share_arms=len(fl) / max(len(arms_g), 1), nweeks=nweeks,
                  ex_top1=r[:-k1].mean() if len(r) > k1 else np.nan, b0_filled=fl.b0_net_R.mean(), b0_unfilled=sel[~sel.filled].b0_net_R.mean())
        ids = [i for i in sel.index if i in pools and len(pools[i])]
        if len(fl) > 5 and ids:
            L = np.array([len(pools[i]) for i in ids]); P = np.full((len(ids), L.max()), np.nan)
            for r_, i in enumerate(ids): P[r_, :L[r_]] = pools[i]
            pick = (prng.random((S.N_PLACEBO, len(ids))) * L).astype(int)
            with np.errstate(all='ignore'): draws = np.nanmean(P[np.arange(len(ids)), pick], axis=1)
            st.update(placebo_pctile=(draws < fl.net_R.mean()).mean() * 100, placebo_mean=np.nanmean(draws))
        if len(fl) > 5:
            cst = S.stats(capped(fl), nweeks)
            st.update({'cap_' + k: val for k, val in cst.items() if k in ('n', 'mean_R', 'day_t', 'fills_per_wk', 'ex_top5')})
        out.append(st)
        W.log(f"{v:13s} {gname:8s} ord={st['n_orders']} fill={len(fl)} mean={st.get('mean_R', float('nan')):.3f} t={st.get('day_t', float('nan')):.2f} "
              f"ex5={st.get('ex_top5', float('nan')):.3f} cap_n={st.get('cap_n')} cap_mean={st.get('cap_mean_R', float('nan')):.3f} pct={st.get('placebo_pctile', float('nan'))}")
pd.DataFrame(out).to_csv(f'{OUT}/1742_summary{atag}.csv', index=False)
W.log('DONE')
