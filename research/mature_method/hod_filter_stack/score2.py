#!/usr/bin/env python3
"""hod_filter_stack — stages 2/4/5/6/8: base populations, feature separation, rule cells,
day cells, nulls. Cells exactly as declared in PREREG.md.

TEST is sealed: no TEST number is computed unless FREEZE.md exists AND --test is passed.
Read-only on every DB. One process.
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
import score as S                                    # noqa: E402  (cost model + book + week stats)

D = f'{ROOT}/research/mature_method/hod_filter_stack'
RD = dict(dtype={'symbol': str, 'day': str, 'why_b': str, 'why_l': str, 'why_n': str},
          keep_default_na=False, na_values=[''])
WITH_TEST = '--test' in sys.argv and os.path.exists(f'{D}/FREEZE.md')
S.SPLITS = ('TRAIN', 'VAL', 'TEST') if WITH_TEST else ('TRAIN', 'VAL')
SPLITS = S.SPLITS

NEW_FEATS = ['breadth_min', 'breadth_15m', 'breadth_day', 'cohort_rank_dist', 'rs_vs_cohort',
             'spy_dist_hod_pct', 'spy_ret_30m', 'spy_ret_open_sig',
             'ret_open_1000', 'ret_1000_sig', 'ret_open_sig', 'slope5', 'slope_prior10',
             'slope_accel', 'vol3_over_10', 'vwap_dist_pct', 'range_pos', 'atr14_pct',
             'hod_age_bars', 'prev_close_pos', 'dow', 'spy_vol20', 'breadth_by_1000',
             'sym_prior_n', 'sym_prior_meanR', 'news_recency_min', 'news_n']
OLD_FEATS = ['entry_m', 'rv_profile', 'dist_open_pct', 'gap_pct', 'prev_range_pct',
             'dist_20d_high_pct', 'is_wrapper', 'has_news', 'above_vwap', 'n_prior',
             'spy_5m_ret', 'spy_range3', 'bar_vol']
OLD_MERGED = ['bar_vol_x', 'drive_min', 'rv_clock', 'coh_by_t']   # from causal_filter/features.csv


# ------------------------------------------------------------------ population
def load_pop():
    p = pd.read_csv(f'{D}/pop.csv', **RD)
    dc = pd.read_csv(f'{D}/day_ctx.csv', dtype={'day': str}, keep_default_na=False, na_values=[''])
    p = p.merge(dc, on='day', how='left')
    p['split'] = S.split_of(p.day.values)
    p['wk'] = pd.to_datetime(p.day).dt.to_period('W-FRI').astype(str)
    nb = pd.read_csv(f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv',
                     dtype={'symbol': str, 'day': str}, keep_default_na=False,
                     na_values=['']).drop_duplicates(['day', 'symbol', 'entry_m'])
    p = p.merge(nb[['day', 'symbol', 'entry_m', 'spread_mean', 'ask_dec', 'bid_dec', 'n_sig']],
                on=['day', 'symbol', 'entry_m'], how='left')
    p = p.sort_values(['day', 'symbol', 'entry_m'], kind='mergesort').reset_index(drop=True)
    return p


def sig_set(p, tag='b', band=1, min_price=20.0, r_min=1.0, max_bps=100.0, max_frac_r=0.15,
            obtain=True, last_m=840):
    """The pre-book signal set of a base: first-qualifying break for (tag, band), then the gates
    that KILL the symbol-day (fill cap, r_min, price floor), then the cost gates."""
    d = p[(p[f'first_{tag}{band}'] == 1) & (p.entry_m <= last_m + 1)]
    d = d[d[f'r_pct_{tag}'].notna() & (d.fill_capped == 1) &
          (d[f'r_pct_{tag}'] >= r_min) & (d.next_open >= min_price)]
    x = S.attach_cost(d, tag)
    if max_bps:
        x = x[(x.sp_pct * 100) <= max_bps]
    if max_frac_r:
        x = x[(x.sp_pct / x.r_pct.clip(lower=0.05)) <= max_frac_r]
    if obtain:
        x = x[x.obtainable.astype(bool)]
    return x


BASES = {'B0': dict(tag='b', band=1),
         'B1L': dict(tag='l', band=1),
         'B1': dict(tag='n', band=1),
         'B2': dict(tag='n', band=0),
         'B3': dict(tag='n', band=0, max_frac_r=0.08)}

HDR = ('| cell                         | split | n     | /wk   | grossR | netR   | net(b) |  t    | '
       'green | flat  | rs | worst $  | total $   | MDD $    | gr mo | ex5    | imp |')
SEP = '|' + '|'.join(['-' * 6] * 17) + '|'


BOOKS = {}


def show(name, b, out=None):
    BOOKS[name] = b
    print(S.fmt_cell(name, b))
    if out is not None:
        for sp in SPLITS:
            w = S.week_stats(b, sp)
            w.update(cell=name, split=sp)
            out.append(w)


# ------------------------------------------------------------------ separation table
def tercile_table(x, feats, out_csv):
    """TRAIN-cut terciles per feature; gross-R spread best-minus-worst, n each side, sign
    consistency across the two halves of TRAIN, and the VAL read of the TRAIN-best tercile."""
    tr = x[x.split == 'TRAIN']
    tr_h1 = tr[tr.day < '2025-07-01']; tr_h2 = tr[tr.day >= '2025-07-01']
    va = x[x.split == 'VAL']
    rows = []
    for f in feats:
        v = pd.to_numeric(x[f], errors='coerce')
        cov = float(v.notna().mean())
        tv = pd.to_numeric(tr[f], errors='coerce')
        if tv.notna().sum() < 900 or tv.nunique() < 3:
            rows.append(dict(feat=f, cov=cov, note='insufficient'))
            continue
        q = tv.quantile([1 / 3, 2 / 3]).values
        if q[0] == q[1]:
            q = np.unique(tv.quantile([0.25, 0.5, 0.75]).values)[:2]
            if len(q) < 2 or q[0] == q[1]:
                rows.append(dict(feat=f, cov=cov, note='degenerate'))
                continue

        def terc(d):
            vv = pd.to_numeric(d[f], errors='coerce')
            return np.where(vv <= q[0], 0, np.where(vv <= q[1], 1, 2)), vv.notna().values

        t_tr, ok_tr = terc(tr)
        g = [tr.rr.values[ok_tr & (t_tr == k)] for k in (0, 1, 2)]
        n = [len(z) for z in g]
        mu = [float(np.mean(z)) if len(z) else np.nan for z in g]
        best = int(np.nanargmax(mu)); worst = int(np.nanargmin(mu))
        spread = mu[best] - mu[worst]
        half_ok = True
        for hh in (tr_h1, tr_h2):
            th, okh = terc(hh)
            gb = hh.rr.values[okh & (th == best)]; gw = hh.rr.values[okh & (th == worst)]
            if len(gb) < 30 or len(gw) < 30 or (np.mean(gb) - np.mean(gw)) <= 0:
                half_ok = False
        t_va, ok_va = terc(va)
        gv = [va.rr.values[ok_va & (t_va == k)] for k in (0, 1, 2)]
        nv = [len(z) for z in gv]
        muv = [float(np.mean(z)) if len(z) else np.nan for z in gv]
        se = np.sqrt(np.var(g[best], ddof=1) / max(n[best], 2) + np.var(g[worst], ddof=1) / max(n[worst], 2))
        # winsorised twin (1%/99% of the whole TRAIN rr) -- a spread that lives in the tails shows here
        lo, hi = np.percentile(tr.rr.values, [1, 99])
        wz = [float(np.mean(np.clip(z, lo, hi))) if len(z) else np.nan for z in g]
        wspread = wz[best] - wz[worst]
        rows.append(dict(feat=f, cov=cov, q1=q[0], q2=q[1], n0=n[0], n1=n[1], n2=n[2],
                         mu0=mu[0], mu1=mu[1], mu2=mu[2], best=best, worst=worst, spread=spread,
                         t=spread / se if se else np.nan, half_consistent=half_ok,
                         val_best=muv[best], val_worst=muv[worst], val_spread=muv[best] - muv[worst],
                         nv_best=nv[best], mu_best=mu[best], mu_worst=mu[worst], wspread=wspread,
                         selectable=bool(spread >= 0.20 and min(n[best], n[worst]) >= 300 and half_ok)))
    t = pd.DataFrame(rows)
    t.to_csv(out_csv, index=False)
    return t


def main():
    print('== hod_filter_stack — stages 2/4/5/6/8 ==', flush=True)
    p = load_pop()
    print(f'pop {len(p)} rows | days {p.day.nunique()} | nbbo matched {p.spread_mean.notna().mean():.1%}')
    S.build_impute(p)
    cells = []

    # ---------------- stage 2: base populations -------------------------------------------
    print('\n== STAGE 2 — BASE POPULATIONS ==\n' + HDR + '\n' + SEP)
    pre = {}
    for nm, kw in BASES.items():
        s = sig_set(p, **kw)
        pre[nm] = s
        show(nm, S.apply_book(s, 12, 4), cells)
    for nm, s in pre.items():
        for sp in SPLITS:
            d = s[s.split == sp]
            print(f'  pre-book {nm:4s} {sp:5s} n={len(d):6d} grossR {d.rr.mean():+.4f} '
                  f'+/- {d.rr.std(ddof=1)/np.sqrt(max(len(d),1)):.4f}')

    B2 = pre['B2']
    B2.to_pickle(f'{D}/b2.pkl')

    # ---------------- availability audit --------------------------------------------------
    print('\n== AVAILABILITY AUDIT (B2 pre-book) ==')
    av = []
    for f in NEW_FEATS + OLD_FEATS:
        if f not in B2.columns:
            av.append(dict(feat=f, note='ABSENT')); print(f'  {f:22s} ABSENT'); continue
        v = pd.to_numeric(B2[f], errors='coerce')
        win = B2.rr > 0
        mw = float(v[win].isna().mean()); ml = float(v[~win].isna().mean())
        av.append(dict(feat=f, cov=float(v.notna().mean()), miss_win=mw, miss_loss=ml,
                       drop=bool(abs(mw - ml) > 0.05)))
        flag = '  <-- DROP (outcome-dependent missingness)' if abs(mw - ml) > 0.05 else ''
        print(f'  {f:22s} cov {v.notna().mean():6.1%}  miss win {mw:6.1%} loss {ml:6.1%}{flag}')
    pd.DataFrame(av).to_csv(f'{D}/availability.csv', index=False)
    dropped = {r['feat'] for r in av if r.get('drop') or r.get('note') == 'ABSENT'}
    feats_new = [f for f in NEW_FEATS if f not in dropped]
    feats_old = [f for f in OLD_FEATS if f not in dropped]

    # ---------------- stage 4a: separation table ------------------------------------------
    print('\n== STAGE 4 — TERCILE SEPARATION ON B2 (TRAIN edges) ==')
    t = tercile_table(B2, feats_new + feats_old, f'{D}/separation.csv')
    t['fam'] = np.where(t.feat.isin(feats_new), 'NEW', 'OLD')
    tt = t[t.spread.notna()].sort_values('spread', ascending=False)
    print('| feature              | fam | cov  | n best | n worst | mu best | mu worst | TRAIN spread | '
          'winsor | t     | halves | VAL spread | selectable |')
    for r in tt.itertuples():
        nb_ = [r.n0, r.n1, r.n2][int(r.best)]; nw_ = [r.n0, r.n1, r.n2][int(r.worst)]
        print(f'| {r.feat:<20s} | {r.fam:3s} | {r.cov:4.0%} | {int(nb_):6d} | {int(nw_):7d} | '
              f'{r.mu_best:+7.3f} | {r.mu_worst:+8.3f} | {r.spread:+12.3f} | {r.wspread:+6.3f} | '
              f'{r.t:+5.2f} | {str(r.half_consistent):6s} | {r.val_spread:+10.3f} | '
              f'{str(r.selectable):10s} |')

    # ---- the apples-to-apples table: NEW vs the FOUR merged OLD features, on B0 where the
    # ---- causal-filter feature file is ~complete -----------------------------------------
    print('\n== STAGE 4 — THE SAME TABLE ON B0 (where the 4 merged OLD features are complete) ==')
    t0 = tercile_table(pre['B0'], feats_new + feats_old + OLD_MERGED, f'{D}/separation_b0.csv')
    t0['fam'] = np.where(t0.feat.isin(OLD_MERGED + feats_old), 'OLD', 'NEW')
    tt0 = t0[t0.spread.notna()].sort_values('spread', ascending=False)
    print('| feature              | fam | cov  | n best | n worst | mu best | mu worst | TRAIN spread | '
          'winsor | t     | halves | VAL spread | selectable |')
    for r in tt0.itertuples():
        nb_ = [r.n0, r.n1, r.n2][int(r.best)]; nw_ = [r.n0, r.n1, r.n2][int(r.worst)]
        print(f'| {r.feat:<20s} | {r.fam:3s} | {r.cov:4.0%} | {int(nb_):6d} | {int(nw_):7d} | '
              f'{r.mu_best:+7.3f} | {r.mu_worst:+8.3f} | {r.spread:+12.3f} | {r.wspread:+6.3f} | '
              f'{r.t:+5.2f} | {str(r.half_consistent):6s} | {r.val_spread:+10.3f} | '
              f'{str(r.selectable):10s} |')

    # ---------------- stage 4b: rule cells ------------------------------------------------
    sel = tt[tt.selectable].head(5)
    print(f'\nselected by the pre-committed rule (spread>=0.20R, n>=300 each side, same sign in both '
          f'TRAIN halves, max 5): {list(sel.feat)}')
    print('\n== STAGE 4 — RULE CELLS ==\n' + HDR + '\n' + SEP)
    masks = {}
    for i, r in enumerate(sel.itertuples(), 1):
        v = pd.to_numeric(B2[r.feat], errors='coerce')
        terc = np.where(v <= r.q1, 0, np.where(v <= r.q2, 1, 2))
        mask = pd.Series((terc == int(r.best)) & v.notna().values, index=B2.index)
        masks[r.feat] = mask
        show(f'R{i} {r.feat} T{int(r.best)}', S.apply_book(B2[mask], 12, 4), cells)
    fl = list(masks)
    if len(fl) >= 2:
        show('R-AND2', S.apply_book(B2[masks[fl[0]] & masks[fl[1]]], 12, 4), cells)
    if len(fl) >= 3:
        show('R-AND3', S.apply_book(B2[masks[fl[0]] & masks[fl[1]] & masks[fl[2]]], 12, 4), cells)

    # ---------------- the decile middle-bucket veto ---------------------------------------
    print('\n== STAGE 4 — DECILE VETO CELLS (the veto CAUSAL_FILTER forbade itself) ==')
    tr = B2[B2.split == 'TRAIN']
    veto_feats = list(tt.head(3).feat) if len(tt) >= 3 else list(tt.feat)
    for i, f in enumerate(veto_feats, 1):
        v = pd.to_numeric(tr[f], errors='coerce')
        if v.notna().sum() < 1000:
            print(f'  V{i} {f}: too few rows'); continue
        ed = np.unique(v.quantile(np.arange(0, 1.001, 0.1)).values)
        if len(ed) < 4:
            print(f'  V{i} {f}: degenerate deciles'); continue
        dec = pd.cut(v, ed, labels=False, include_lowest=True)
        mu = tr.rr.groupby(dec).mean()
        n = tr.rr.groupby(dec).size()
        wd = int(mu.idxmin())
        print(f'  V{i} {f}: worst TRAIN decile #{wd} edges [{ed[wd]:.4g}, {ed[wd+1]:.4g}) '
              f'-> {mu[wd]:+.3f} R on {n[wd]} rows; all deciles ' +
              ' '.join(f'{x:+.2f}' for x in mu.values))
        va = pd.to_numeric(B2[f], errors='coerce')
        bad = (va >= ed[wd]) & (va < ed[wd + 1]) if wd < len(ed) - 2 else (va >= ed[wd])
        show(f'V{i} veto {f} d{wd}', S.apply_book(B2[~bad.fillna(False)], 12, 4), cells)

    # ---------------- stage 6: day-level cells --------------------------------------------
    print('\n== STAGE 6 — DAY-LEVEL CELLS ==')
    dctx = B2.drop_duplicates('day')[['day', 'split', 'breadth_by_1000', 'spy_ret_0930_1000',
                                      'spy_vol20', 'dow']].set_index('day')
    trd = dctx[dctx.split == 'TRAIN']
    bq = trd.breadth_by_1000.quantile([1 / 3, 2 / 3]).values
    vq = trd.spy_vol20.quantile([1 / 3, 2 / 3]).values
    dw_tr = B2[B2.split == 'TRAIN'].groupby('dow').rr.agg(['mean', 'size'])
    worst_dow = int(dw_tr['mean'].idxmin())
    print(f'  TRAIN breadth_by_1000 terciles {bq.round(1)} | spy_vol20 terciles {vq.round(2)} | '
          f'weekday TRAIN gross ' + ' '.join(f'{d}:{m:+.3f}' for d, m in dw_tr['mean'].items()) +
          f' -> worst weekday {worst_dow}')
    bcol = B2.day.map(dctx.breadth_by_1000); vcol = B2.day.map(dctx.spy_vol20)
    scol = B2.day.map(dctx.spy_ret_0930_1000)
    day_masks = {
        'D-a breadth TOP tercile': bcol > bq[1],
        'D-b breadth BOT tercile': bcol <= bq[0],
        'D-c SPY 0930-1000 up': scol > 0,
        'D-d spy_vol20 BOT tercile': vcol <= vq[0],
        'D-e spy_vol20 TOP tercile': vcol > vq[1],
        f'D-f skip weekday {worst_dow}': B2.dow != worst_dow,
        'D-g SPY up AND breadth>=mid': (scol > 0) & (bcol > bq[0]),
    }
    print(HDR + '\n' + SEP)
    dres = {}
    for nm, mk in day_masks.items():
        b = S.apply_book(B2[mk.fillna(False)], 12, 4)
        dres[nm] = S.week_stats(b, 'TRAIN')['green']
        show(nm, b, cells)
    top2 = sorted(dres, key=dres.get, reverse=True)[:2]
    show(f'D-h AND({top2[0]} , {top2[1]})',
         S.apply_book(B2[(day_masks[top2[0]] & day_masks[top2[1]]).fillna(False)], 12, 4), cells)

    # ---------------- stage 8: nulls ------------------------------------------------------
    cf = pd.DataFrame(cells)
    cf.to_csv(f'{D}/cells.csv', index=False)
    print('\n== STAGE 8 — COUNT-MATCHED PERMUTATION NULL (2,000 draws) ==')
    print('| cell | split | observed green % | null mean | [p5, p95] | outside? |')
    nulls = []
    todo = [c for c in cf.cell.unique()]
    for nm in todo:
        for sp in SPLITS:
            key = (cf.cell == nm) & (cf.split == sp)
            if not key.any():
                continue
            b = BOOKS.get(nm)
            if b is None:
                continue
            obs, mu, p5, p95 = S.null_band(b, sp)
            out = 'ABOVE' if obs > p95 else ('below' if obs < p5 else 'inside')
            nulls.append(dict(cell=nm, split=sp, obs=obs, mu=mu, p5=p5, p95=p95, outside=out))
            print(f'| {nm} | {sp} | {obs:.1f} | {mu:.1f} | [{p5:.1f}, {p95:.1f}] | {out} |')
    pd.DataFrame(nulls).to_csv(f'{D}/nulls.csv', index=False)
    print('\nDONE')


if __name__ == '__main__':
    main()
