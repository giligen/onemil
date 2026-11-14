"""Cell 1,779 GAPSHORT frequency frontier: marginal-row reads for rungs R1-R6 (spec: PREREG_1779.md)."""
import os, sys, json, importlib.util
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
os.nice(19)
OUT = '/home/ec2-user/onemil/research/signal_search'
spec = importlib.util.spec_from_file_location('b1776', f'{OUT}/1776R_build.py')
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)   # reuse load_universe/Fetcher/boot (1776R)
COST = B.FEES + B.LOCATE + B.BORROW
RUNGS = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']

def log(*a):
    """Flushed progress line."""
    print(*a, flush=True)

def build_pop():
    """Population with gap rank, L1 tercile (TRAIN top-10 cutoffs), 10:30 price, close, 1774 quotes."""
    p, spy = B.load_universe()
    p = p.sort_values(['day', 'gap'], ascending=[True, False]).copy()
    p['rank'] = p.groupby('day').cumcount() + 1
    e73 = pd.read_parquet(f'{OUT}/1773_events.parquet')
    e73['day'] = e73.day.astype(str).str[:10]
    p = p.merge(e73[['day', 'symbol', 'p1030', 'close']], on=['day', 'symbol'], how='left')
    e74 = pd.read_parquet(f'{OUT}/1774_events.parquet'); e74['day'] = e74.day.astype(str).str[:10]
    q74 = {(d, s): (a, b, a11) for d, s, a, b, a11 in zip(e74.day, e74.symbol, e74.ask1030, e74.bid1030, e74.ask1100)}
    tr10 = p[(p.split == 'TRAIN') & (p['rank'] <= 10)].L1.dropna()
    trall = p[p.split == 'TRAIN'].L1.dropna()
    log('T3 cutoff top10-TRAIN', tr10.quantile(2/3), 'all-TRAIN', trall.quantile(2/3), '(1776R reported 7,224,163)')
    hi = trall.quantile(2/3) if abs(trall.quantile(2/3) - 7224163) < abs(tr10.quantile(2/3) - 7224163) else tr10.quantile(2/3)
    ref = trall if hi == trall.quantile(2/3) else tr10
    lo = ref.quantile(1/3)
    log(f'USED cutoffs: T3>= {hi:,.0f}  T2>= {lo:,.0f} (ref={"all-TRAIN" if ref is trall else "top10-TRAIN"})')
    p['T'] = np.where(p.L1.isna(), 0, np.where(p.L1 >= hi, 3, np.where(p.L1 >= lo, 2, 1)))
    p['base'] = (p['rank'] <= 10) & (p['T'] == 3)
    return p, spy, q74, (lo, hi)

def rung_sets(p):
    """Boolean marginal masks per rung."""
    ext = (p.p1030 >= 1.05 * p.open) & ~p.base
    return {'R1': (p['rank'] <= 10) & (p['T'] == 2), 'R2': (p['rank'] <= 10) & (p['T'] == 1),
            'R3': (p['rank'].between(11, 15)) & (p['T'] == 3), 'R4': (p['rank'].between(11, 15)) & (p['T'] == 2),
            'R5': ext, 'R6': ext & p['T'].isin([2, 3])}

def quote(fx, q74, d, s):
    """(ask, bid, ask1100, source) from jsonl cache first, then 1774 columns, else None."""
    r = fx.done.get(B.key(d, s))
    if r is not None: return (r['ap'] or np.nan, r['bp'] or np.nan, np.nan, 'q')
    if (d, s) in q74:
        a, b, a11 = q74[(d, s)]; return (a, b, a11, '74')
    return None

def price_rows(df, fx, q74, short_pm):
    """Attach ask/bid/valid/net. short_pm=False: R1-4 open/ask; True: R5/6 bid*0.999/close."""
    A, Bd, A11, S = [], [], [], []
    for d, s in zip(df.day, df.symbol):
        q = quote(fx, q74, d, s) or (np.nan, np.nan, np.nan, 'none')
        A.append(q[0]); Bd.append(q[1]); A11.append(q[2]); S.append(q[3])
    df = df.assign(ask=A, bid=Bd, ask1100=A11, qsrc=S)
    mid = (df.ask + df.bid) / 2; hs = (df.ask - df.bid) / 2 / mid * 1e4
    bad = ~((df.ask > 0) & (df.bid > 0)) | (hs > 1000) | (mid < 0.5 * df.open) | (mid > 2 * df.open)
    df['valid'] = ~bad
    if short_pm:
        df['net'] = (df.bid * 0.999 / df.close - 1) * 1e4 - COST
        df['net1100'] = (df.bid * 0.999 / df.ask1100 - 1) * 1e4 - 4 - 9 - 27.8 * 0 - 27.8  # same stack, cover at ask1100
        df.loc[df.close.isna() | df.bid.isna(), 'valid'] = False
    else:
        df['net'] = (df.open / df.ask - 1) * 1e4 - COST
    return df

def wkR(t):
    """Weekly R sum ($300 notional, R=$150) for valid rows, keyed by ISO week."""
    t = t[t.valid]
    return (300 * t.net / 1e4 / 150).groupby(pd.to_datetime(t.day).dt.to_period('W').values).sum()

def summarize(m, base, rung, split, extra=''):
    """One summary row for marginal rows m (priced), the base book and rung label."""
    mv = m[m.valid]; n = len(mv)
    allwk = pd.Series(1, index=pd.to_datetime(base.day).dt.to_period('W').unique())
    wkc = pd.to_datetime(m.day).dt.to_period('W').value_counts().reindex(allwk.index, fill_value=0)
    row = dict(rung=rung, split=split, n_rows=len(m), n_valid=n, coverage=(n / len(m)) if len(m) else np.nan,
               names_wk_med=wkc.median(), names_wk_p10=wkc.quantile(.1), names_wk_mean=wkc.mean())
    if n >= 5:
        mean = mv.net.mean(); e = mv.net - mean
        se = np.sqrt((e.groupby(mv.day).sum() ** 2).sum()) / n
        s = mv.net.sort_values(); k = max(1, int(np.ceil(.05 * n)))
        w = wkR(m).reindex(allwk.index, fill_value=0)
        comb = wkR(pd.concat([base, m])).reindex(allwk.index, fill_value=0)
        bd = base[base.valid].groupby('day').size().reindex(sorted(set(base.day) | set(m.day)), fill_value=0)
        md = mv.groupby('day').size().reindex(bd.index, fill_value=0)
        row.update(mean_net=mean, day_t=mean / se if se > 0 else np.nan, ex_top5=s.iloc[:n - k].mean(),
                   worst5_pct=s.iloc[:k].mean() / 100, wkP10_rung=w.quantile(.1), wkP10_comb=comb.quantile(.1),
                   wk_mean_comb=comb.mean(), names_day_after_med=(bd + md).median(), names_day_after_mean=(bd + md).mean())
    # missingness gap: winners (close<open) vs losers
    if len(m):
        win = (m.close < m.open); miss = ~m.valid
        row['miss_gap_pp'] = 100 * abs(miss[win].mean() - miss[~win].mean()) if win.any() and (~win).any() else np.nan
    return row

def fetch_jobs(fx, jobs, label):
    """Fetch every missing (day,sym) quote with 4 threads; progress every 200."""
    jobs = [j for j in dict.fromkeys(jobs) if B.key(*j) not in fx.done]
    log(f'[{label}] to fetch {len(jobs)}')
    with ThreadPoolExecutor(4) as ex:
        for i, _ in enumerate(ex.map(lambda j: fx.one(*j), jobs)):
            if i % 200 == 0: log(f'[{label}] fetched {i}/{len(jobs)}')

def placebos(p, spy, q74, fx, rung, m, base, short_pm, om, rng, sel_days):
    """Placebo (i) same-day non-base non-rung quoted rows, (ii) same symbol +-5 days; returns dict of p95s."""
    v = m[m.valid]; n = len(v); out = {}
    excl = set(zip(base.day, base.symbol)) | set(zip(m.day, m.symbol))
    pool = p[(p.split == 'VAL') & p.day.isin(sel_days) & p.L1.notna() & ~pd.Series([(a, b) in excl for a, b in zip(p.day, p.symbol)], index=p.index)]
    pool = price_rows(pool, fx, q74, short_pm); pool = pool[pool.valid]
    out['pl1_n'] = len(pool)
    out['pl1_p95'] = B.boot(pool.net, n, 200, 1) if len(pool) >= 20 and n else np.nan
    idx = {d: i for i, d in enumerate(spy)}; rows = []
    for d, s, o, c in zip(m.day, m.symbol, m.open, m.close):
        for sh in (5, -5):
            j = idx.get(d, -999) + sh
            if 0 <= j < len(spy):
                d2 = spy[j]; o2 = om.get((s, d2))
                if o2 is not None and not np.isnan(o2): rows.append((d2, s, o2, om_close.get((s, d2), np.nan)))
    pl = pd.DataFrame(rows, columns=['day', 'symbol', 'open', 'close'])
    out['pl2_rows'] = len(pl)
    if len(pl):
        pl = price_rows(pl, fx, q74, short_pm); pl = pl[pl.valid]
        out['pl2_n'] = len(pl); out['pl2_p95'] = B.boot(pl.net, n, 200, 2) if len(pl) >= 20 and n else np.nan
    return out, pool

if __name__ == '__main__':
    p, spy, q74, (lo, hi) = build_pop()
    sets = rung_sets(p)
    fx = B.Fetcher()
    # sanity: 1774 vs jsonl quote agreement
    ag = [(fx.done[B.key(d, s)]['ap'], q74[(d, s)][0]) for (d, s) in q74 if B.key(d, s) in fx.done and fx.done[B.key(d, s)]['ap'] and q74[(d, s)][0] == q74[(d, s)][0]]
    ag = np.array(ag, float); log('1774 vs jsonl ask agreement: n', len(ag), 'exact', float((np.abs(ag[:, 0] - ag[:, 1]) < 1e-9).mean()) if len(ag) else None)
    base = B.pd.read_parquet(f'{OUT}/1776R_events.parquet'); log('base rows', len(base), 'mine', int(p.base.sum()))
    # fetch R1-R4 then R5/R6 (rows with neither cache nor 1774 quote)
    need = lambda r: [(d, s) for d, s in zip(p[sets[r]].day, p[sets[r]].symbol) if quote(fx, q74, d, s) is None]
    for grp in (['R1', 'R2', 'R3', 'R4'], ['R5', 'R6']):
        j = sum([need(r) for r in grp], []); log(grp, 'unique missing', len(set(j)))
        if '--nofetch' not in sys.argv: fetch_jobs(fx, j, ','.join(grp))
    db = B.sqlite3.connect(f'file:{B.ROOT}/data/cache.db?mode=ro', uri=True)
    dbar = pd.read_sql("select symbol,bar_date as day,open,close from daily_bars where bar_date>='2025-05-01'", db)
    om = {(a, b): c for a, b, c in zip(dbar.symbol, dbar.day, dbar.open)}
    om_close = {(a, b): c for a, b, c in zip(dbar.symbol, dbar.day, dbar.close)}
    basep = price_rows(p[p.base], fx, q74, False)
    rng = np.random.default_rng(3); summ, marg = [], []
    for r in RUNGS:
        sp = r in ('R5', 'R6'); m = price_rows(p[sets[r]], fx, q74, sp); m['rung'] = r; marg.append(m)
        for nm, sel in (('TRAIN', m.split == 'TRAIN'), ('TRAIN_H1', (m.split == 'TRAIN') & (m.day < '2025-10-01')),
                        ('TRAIN_H2', (m.split == 'TRAIN') & (m.day >= '2025-10-01')), ('VAL', m.split == 'VAL')):
            bsel = basep[(basep.split == ('VAL' if nm == 'VAL' else 'TRAIN'))]
            if nm.startswith('TRAIN_H'): bsel = bsel[bsel.day < '2025-10-01'] if nm == 'TRAIN_H1' else bsel[bsel.day >= '2025-10-01']
            summ.append(summarize(m[sel], bsel, r, nm))
        if sp:
            mv = m[(m.split == 'VAL') & m.valid & m.net1100.notna() & m.ask1100.notna()]
            log(f'{r} secondary cover@ask1100 VAL n={len(mv)} mean={mv.net1100.mean():+.1f}' if len(mv) else f'{r} secondary: no rows')
            summ[-1]['net1100_mean'] = mv.net1100.mean() if len(mv) else np.nan; summ[-1]['n1100'] = len(mv)
    S = pd.DataFrame(summ); S.to_csv(f'{OUT}/1779_summary.csv', index=False)
    log(S[S.split == 'VAL'].T.to_string())
    # placebos for VAL mean > 0
    pc = []
    mg = pd.concat(marg); mg.to_parquet(f'{OUT}/1779_marginal.parquet')
    for r in RUNGS:
        v = S[(S.rung == r) & (S.split == 'VAL')].iloc[0]
        if not (v.get('mean_net', -1) > 0): continue
        m = mg[(mg.rung == r) & (mg.split == 'VAL')]; sp = r in ('R5', 'R6')
        sel_days = sorted(m.day.unique())
        # fetch placebo quotes: cap pool sample at 600 rows/rung, plus +-5d own quotes
        excl = set(zip(basep.day, basep.symbol)) | set(zip(m.day, m.symbol))
        cand = p[(p.split == 'VAL') & p.day.isin(sel_days) & p.L1.notna()]
        cand = cand[[(a, b) not in excl for a, b in zip(cand.day, cand.symbol)]]
        cand = cand.sample(n=min(600, len(cand)), random_state=5)
        jobs = [(d, s) for d, s in zip(cand.day, cand.symbol) if quote(fx, q74, d, s) is None]
        idx = {d: i for i, d in enumerate(spy)}
        for d, s in zip(m.day, m.symbol):
            for sh in (5, -5):
                j = idx.get(d, -999) + sh
                if 0 <= j < len(spy) and (s, spy[j]) in om and quote(fx, q74, spy[j], s) is None: jobs.append((spy[j], s))
        fetch_jobs(fx, jobs, f'placebo {r}')
        p_small = p[p.index.isin(cand.index)]
        o, _ = placebos(p_small, spy, q74, fx, r, m, basep, sp, om, rng, sel_days); o['rung'] = r; pc.append(o); log(r, o)
    pd.DataFrame(pc).to_csv(f'{OUT}/1779_placebo.csv', index=False)
    log('DONE')
