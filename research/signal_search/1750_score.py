"""Cell 1,750 scorer (PREREG_1750): opportunistic insider purchases (Form 4), entry open f+1, exit close f+5 / f+20.

Universe = rank 1-3,000 by 20-day median RAW dollar volume THROUGH the filing day f (known before the f+1 open), raw close(f) >= $3.
Excess = event return (adjusted close(f+h) / adjusted open(f+1) - 1) - mean of the same universe (or rank bucket) on day f.
Windows touching a >10 % adjusted/raw ratio jump in (f, f+h] are dropped (split).  TEST (f >= 2024) hard-excluded.
Reads: ALL / OPP (any opportunistic owner) / ROUTINE (qualifying, no opportunistic owner) / CLUSTER (>= 2 distinct owners within 5 trading
days, event = day of the second owner's filing, deduped 5 days) / OPP_CLUSTER.  Cost = measured NBBO half-spread by rank bucket + 2 bp/leg.
Usage: nice -n 19 python3 1750_score.py
"""
import importlib.util, os, sys, time
os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location('s49', os.path.join(HERE, '1749_score.py'))
s49 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s49)
nw_t = s49.nw_t
SEAL = '2024-01-01'
SPLITS = {'TRAIN': ('2016-01-01', '2020-12-31'), 'T_H1_16_18': ('2016-01-01', '2018-12-31'),
          'T_H2_19_20': ('2019-01-01', '2020-12-31'), 'VAL': ('2021-01-01', '2023-12-31')}
BUCKETS = {'r1_1000': (1, 1000), 'r1001_2000': (1001, 2000), 'r2001_3000': (2001, 3000)}
EXTRA_BP = 2.0
MIN_U = 100
t0 = time.time()


def log(m):
    print(f'[{time.time() - t0:6.0f}s] {m}', flush=True)


def costs():
    """Round-trip bp by rank bucket from measured half-spread tables (entry at the open, exit at the close) + 2 bp per leg."""
    c46 = pd.read_csv(os.path.join(HERE, '1746_cost.csv')).half_spread_bp.dropna().mean()
    c47 = pd.read_csv(os.path.join(HERE, '1747_cost.csv')).exit_hs_bp.dropna().mean()
    c48 = pd.read_csv(os.path.join(HERE, '1748_cost.csv'))
    out = {'r1_1000': c46 + c47 + 2 * EXTRA_BP}
    for b, (lo, hi) in list(BUCKETS.items())[1:]:
        s = c48[(c48['rank'] >= lo) & (c48['rank'] <= hi)]
        out[b] = s.entry_hs_bp.mean() + s.exit_hs_bp.mean() + 2 * EXTRA_BP
    return out


def rss():
    """Current and peak resident set size in MB (peak = ru_maxrss)."""
    import resource
    cur = int(open('/proc/self/statm').read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**20
    return f'rss {cur:.0f} MB peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB'


CUT = '2024-02-29'   # bars past this are never needed: events f < 2024-01 and the longest window is f+20 trading days (TEST stays sealed)


def _keys(path):
    """Row-aligned (symbol codes, symbol list, date codes, date list) of a bars file, dictionary-encoded so no 12M-object string column is built."""
    import pyarrow.parquet as pq
    out = []
    for c in ('symbol', 'date'):
        col = pq.read_table(path, columns=[c]).column(c).combine_chunks().dictionary_encode()
        out.append((col.indices.to_numpy(zero_copy_only=False).astype(np.int32), np.array(col.dictionary.to_pylist())))
    return out[0][0], out[0][1], out[1][0], out[1][1]


def _vals(path, col):
    """One float column of a bars file as float64 (read alone to keep the peak small)."""
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=[col]).column(col).to_numpy().astype(np.float64)


def load_lean(cache_dir, extra_dir):
    """Build (T x S) float32 matrices o, c, craw, dv over cache + extra symbols, one column at a time, dates <= CUT.  Extras whose symbol
    is already in the cache are skipped; extras are appended after the cache symbols (rank-only, never in the benchmark)."""
    srcs = []
    for tag, d, na, nr in (('cache', cache_dir, 'bars_all.parquet', 'bars_raw.parquet'), ('extra', extra_dir, 'bars_extra_all.parquet', 'bars_extra_raw.parquet')):
        pa_, pr_ = os.path.join(d, na), os.path.join(d, nr)
        if not (os.path.exists(pa_) and os.path.exists(pr_)):
            log(f'WARNING {tag} bars missing on disk ({d}): skipped'); continue
        srcs.append((tag, pa_, pr_))
    keys = {}
    for tag, pa_, pr_ in srcs:
        keys[tag] = (_keys(pa_), _keys(pr_)); log(f'{tag} keys loaded; {rss()}')
    (sc, sl, dc, dl), _ = keys['cache']
    dates = np.array(sorted(d for d in set(dl) if d <= CUT))
    csyms = np.array(sorted(set(sl)))
    ex_syms = np.array([])
    if 'extra' in keys:
        (esc, esl, _, _), (erc, erl, _, _) = keys['extra']
        ex_syms = np.array(sorted((set(esl) & set(erl)) - set(csyms))); log(f'extra symbols with bars {len(ex_syms)}')
    syms = np.concatenate([csyms, ex_syms]); T, S = len(dates), len(syms)
    di = {d: i for i, d in enumerate(dates)}; si = {s: i for i, s in enumerate(syms)}
    log(f'matrix shape {T} x {S} ({len(csyms)} cache + {len(ex_syms)} extra); {rss()}')
    def maps(k):
        (c_, l_, d_, dl_) = k
        sm = np.array([si.get(s, -1) for s in l_], np.int32)[c_]; dm = np.array([di.get(d, -1) for d in dl_], np.int32)[d_]
        return sm, dm
    M = {n: np.full((T, S), np.nan, np.float32) for n in ('o', 'c', 'craw', 'dv')}
    for tag, pa_, pr_ in srcs:
        for which, path, kk in (('adj', pa_, keys[tag][0]), ('raw', pr_, keys[tag][1])):
            sm, dm = maps(kk); ok = (sm >= 0) & (dm >= 0); r, cidx = dm[ok], sm[ok]; del sm, dm
            if which == 'adj':
                for n, col in (('o', 'o'), ('c', 'c')):
                    M[n][r, cidx] = _vals(path, col)[ok]
            else:
                c = _vals(path, 'c')[ok]; M['craw'][r, cidx] = c
                M['dv'][r, cidx] = c * _vals(path, 'v')[ok]; del c
            del ok, r, cidx
            log(f'{tag} {which} pivoted; {rss()}')
    return dates, syms, len(csyms), M


def build_events(dates, syms):
    """Symbol-day events with read flags; map symbols to bar columns."""
    ev = pd.read_parquet(os.path.join(HERE, '1750_events.parquet'))
    ev = ev[ev.f < SEAL].copy()
    sset = set(syms)
    def m(s):
        for c in (s, s.replace('-', '.'), s.replace('.', '-')):
            if c in sset:
                return c
        return None
    mp = {s: m(s) for s in ev.symbol.unique()}
    ev['bsym'] = ev.symbol.map(mp)
    log(f'owner-events (pre-2024) {len(ev)}; symbols mapped to bars: {ev.bsym.notna().mean():.3f} of owner-events')
    ev['fd'] = ev.f.dt.strftime('%Y-%m-%d')
    di = np.searchsorted(dates, ev.fd.to_numpy())                  # first trading day >= f
    ev['t'] = np.where((di < len(dates)) & (dates[np.minimum(di, len(dates) - 1)] == ev.fd.to_numpy()), di, di - 1)   # last close known
    ev['t'] = np.where(ev.t < 0, 0, ev.t)
    ev['opp'] = ~ev.routine
    return ev


def day_reads(ev):
    """Collapse to symbol-day events; compute cluster flag over trading-day index t."""
    g = ev.groupby(['bsym', 't'], as_index=False).agg(f=('f', 'min'), n_own=('RPTOWNERCIK', 'nunique'), opp=('opp', 'max'),
                                                      value=('value', 'sum'))
    owners = ev.groupby(['bsym', 't']).RPTOWNERCIK.agg(lambda s: frozenset(s))
    g['own'] = [owners[(b, t)] for b, t in zip(g.bsym, g.t)]
    g = g.sort_values(['bsym', 't']).reset_index(drop=True)
    clus = np.zeros(len(g), bool)
    for b, idx in g.groupby('bsym').indices.items():
        ts = g.t.to_numpy()[idx]; ow = g.own.to_numpy()[idx]
        last_c = -99
        for k in range(len(idx)):
            multi = len(ow[k]) >= 2
            prior = set()
            for j in range(k - 1, -1, -1):
                if ts[k] - ts[j] > 4:
                    break
                prior |= ow[j]
            if (multi or (prior - ow[k]) or (len(ow[k] | prior) >= 2 and prior)) and len(ow[k] | prior) >= 2:
                if ts[k] - last_c > 4:
                    clus[idx[k]] = True
                    last_c = ts[k]
    g['cluster'] = clus
    return g.drop(columns='own')


def main():
    dates, syms, nC, M = load_lean(os.path.join(HERE, '1748_cache'), os.path.join(HERE, '1750_cache'))
    T, S = M['c'].shape
    isC = (np.arange(S) < nC)[None, :]
    adv = np.empty((T, S), np.float32)                                      # 20-day median raw dollar volume through day t, in column blocks
    for j in range(0, S, 1500):
        adv[:, j:j + 1500] = pd.DataFrame(M['dv'][:, j:j + 1500]).rolling(20, min_periods=15).median().to_numpy(np.float32)
    del M['dv']; log(f'adv ready; {rss()}')
    rank = np.full((T, S), np.nan, np.float32)
    for t0_ in range(0, T, 200):                                            # cache names ranked per day in row blocks; extras ranked against the cache distribution
        blk = pd.DataFrame(adv[t0_:t0_ + 200, :nC]).rank(axis=1, ascending=False, method='first').to_numpy(np.float32)
        rank[t0_:t0_ + 200, :nC] = blk
    for t in range(T):
        a = np.sort(adv[t, :nC][~np.isnan(adv[t, :nC])]); x = adv[t, nC:]; ok = ~np.isnan(x)
        rank[t, nC:][ok] = len(a) - np.searchsorted(a, x[ok], side='right') + 1
    del adv; log(f'rank ready; {rss()}')
    craw = M['craw']
    univ = (rank <= 3000) & (craw >= 3)
    ratio = M['c'] / craw
    jump = np.zeros((T, S), np.int32); jump[1:] = np.abs(ratio[1:] / ratio[:-1] - 1) > 0.10
    del ratio
    cs = np.cumsum(jump, axis=0, dtype=np.int32); del jump
    o1 = np.full((T, S), np.nan, np.float32); o1[:-1] = M['o'][1:]
    R = {}
    for h in (5, 20):
        ch = np.full((T, S), np.nan, np.float32); ch[:-h] = M['c'][h:]
        r = ch / o1 - 1; del ch
        bad = np.zeros((T, S), bool); bad[:-h] = (cs[h:] - cs[:-h]) > 0
        r[bad] = np.nan; R[h] = r; del bad
    del cs, o1; log(f'matrices ready; {rss()}')
    # universe mean per day: whole universe and each rank bucket
    U, N = {}, {}
    for h in (5, 20):
        for bn, (lo, hi) in {'all': (1, 3000), **BUCKETS}.items():
            m = isC & univ & (rank >= lo) & (rank <= hi) & ~np.isnan(R[h])     # benchmark = cache names only (as 1749)
            N[bn, h] = m.sum(1)
            U[bn, h] = np.where(N[bn, h] > 0, np.where(m, R[h], np.float32(0)).sum(1, dtype=np.float64) / np.maximum(N[bn, h], 1), np.nan); del m
    log(f'universe means ready; {rss()}')
    # events
    ev = build_events(dates, syms)
    ev['ci'] = ev.bsym.map({s: i for i, s in enumerate(syms)})
    has = ev.ci.notna()
    log(f'coverage: owner-events {len(ev)}; no bars {int((~has).sum())} ({(~has).mean():.3f})')
    e = ev[has].copy(); e['ci'] = e.ci.astype(int)
    e['inU'] = univ[e.t.to_numpy(), e.ci.to_numpy()]; e['rk'] = rank[e.t.to_numpy(), e.ci.to_numpy()]
    log(f'in bars: {len(e)}; in universe (rank<=3000, px>=3) at f: {e.inU.mean():.3f}; by year in-universe share: '
        + str(e.groupby(e.f.dt.year).inU.mean().round(3).to_dict()))
    cov = pd.DataFrame({'all_owner_events': ev.groupby(ev.f.dt.year).size(), 'with_bars': e.groupby(e.f.dt.year).size(),
                        'in_universe': e[e.inU].groupby(e.f.dt.year).size()})
    log('coverage by year:\n' + cov.to_string())
    e = e[e.inU]
    reads_in = day_reads(e)
    reads_in['ci'] = reads_in.bsym.map({s: i for i, s in enumerate(syms)}).astype(int)
    reads_in['rk'] = rank[reads_in.t.to_numpy(), reads_in.ci.to_numpy()]
    for h in (5, 20):
        reads_in[f'r{h}'] = R[h][reads_in.t.to_numpy(), reads_in.ci.to_numpy()]
    reads_in['fd'] = dates[reads_in.t.to_numpy()]
    reads_in['bucket'] = pd.cut(reads_in.rk, [0, 1000, 2000, 3000], labels=list(BUCKETS))
    reads_in.to_parquet(os.path.join(HERE, '1750_symdays.parquet'))
    log(f'symbol-day events in universe {len(reads_in)}; OPP {int(reads_in.opp.sum())} ROUTINE {int((~reads_in.opp).sum())} CLUSTER {int(reads_in.cluster.sum())}')
    cost = costs(); log(f'round-trip cost bp by bucket: { {k: round(v, 1) for k, v in cost.items()} }')
    reads_in['cost'] = reads_in.bucket.map(cost).astype(float)
    READS = {'ALL': lambda r: r.opp | ~r.opp, 'OPP': lambda r: r.opp, 'ROUTINE': lambda r: ~r.opp,
             'CLUSTER': lambda r: r.cluster, 'OPP_CLUSTER': lambda r: r.cluster & r.opp}
    summ, yearly = [], []
    for scope in ['all'] + list(BUCKETS):
        for h in (5, 20):
            rc = f'r{h}'
            u = pd.Series(U[scope, h], index=dates)[N[scope, h] >= MIN_U]
            base = reads_in if scope == 'all' else reads_in[reads_in.bucket == scope]
            base = base[base[rc].notna() & base.fd.isin(u.index)]
            for split, (a, b) in SPLITS.items():
                ndays = int(((u.index >= a) & (u.index <= b)).sum())
                for rn, f in READS.items():
                    x = base[f(base) & (base.fd >= a) & (base.fd <= b)].copy()
                    if len(x) < 10:
                        continue
                    x['U'] = u.reindex(x.fd).to_numpy(); x['ex'] = x[rc] - x.U
                    daily = x.groupby('fd').ex.mean()
                    if len(daily) < 15:
                        continue
                    m, t, ti, se = nw_t(daily.to_numpy())
                    srt = x.ex.sort_values(ascending=False); cut = int(np.ceil(0.05 * len(x)))
                    mon = daily.groupby(daily.index.str[:7]).mean()
                    summ.append(dict(scope=scope, h=h, split=split, read=rn, n=len(x), per_wk=len(x) / (ndays / 5), days=len(daily),
                                     excess_bp=m * 1e4, nw_t=t, iid_t=ti, ev_mean_bp=x.ex.mean() * 1e4,
                                     ex_top5_bp=srt.iloc[cut:].mean() * 1e4, cap20_bp=((x[rc].clip(upper=0.20) - x.U).mean()) * 1e4,
                                     green_months=(mon > 0).mean(), mde_bp=2.8 * se * 1e4, cost_bp=x.cost.mean(),
                                     net_bp=x.ex.mean() * 1e4 - x.cost.mean(), max_r=x[rc].max()))
                    if split == 'VAL' or split == 'TRAIN':
                        pass
            for rn, f in READS.items():
                x = base[f(base) & (base.fd < SEAL)].copy()
                x['ex'] = x[rc] - u.reindex(x.fd).to_numpy()
                for y, xx in x.groupby(x.fd.str[:4]):
                    d = xx.groupby('fd').ex.mean()
                    yearly.append(dict(scope=scope, h=h, read=rn, year=y, n=len(xx), excess_bp=d.mean() * 1e4,
                                       nw_t=nw_t(d.to_numpy())[1] if len(d) > 15 else np.nan))
    S_ = pd.DataFrame(summ); Y = pd.DataFrame(yearly)
    # OPP - ROUTINE (event-level Welch)
    diffs = []
    for scope in ['all']:
        for h in (5, 20):
            rc = f'r{h}'; u = pd.Series(U[scope, h], index=dates)[N[scope, h] >= MIN_U]
            base = reads_in[reads_in[rc].notna() & reads_in.fd.isin(u.index)].copy(); base['ex'] = base[rc] - u.reindex(base.fd).to_numpy()
            for split in ('TRAIN', 'VAL'):
                a, b = SPLITS[split]; x = base[(base.fd >= a) & (base.fd <= b)]
                o, r = x[x.opp].ex, x[~x.opp].ex
                se = np.sqrt(o.var() / len(o) + r.var() / max(len(r), 2)) if len(r) > 2 else np.nan
                diffs.append(dict(h=h, split=split, n_opp=len(o), n_rout=len(r), opp_bp=o.mean() * 1e4, rout_bp=r.mean() * 1e4,
                                  diff_bp=(o.mean() - r.mean()) * 1e4, welch_t=(o.mean() - r.mean()) / se if se == se else np.nan))
    D = pd.DataFrame(diffs)
    S_.to_csv(os.path.join(HERE, '1750_summary.csv'), index=False); Y.to_csv(os.path.join(HERE, '1750_yearly.csv'), index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 600)
    log('SUMMARY scope=all h=5'); print(S_[(S_.scope == 'all') & (S_.h == 5)].round(2).to_string(), flush=True)
    log('SUMMARY scope=all h=20'); print(S_[(S_.scope == 'all') & (S_.h == 20)].round(2).to_string(), flush=True)
    log('SUMMARY buckets OPP h=5 TRAIN/VAL'); print(S_[(S_.scope != 'all') & (S_.h == 5) & (S_.read == 'OPP') & S_.split.isin(['TRAIN', 'VAL'])].round(2).to_string(), flush=True)
    log('YEARLY scope=all h=5'); print(Y[(Y.scope == 'all') & (Y.h == 5)].round(2).to_string(), flush=True)
    log('OPP vs ROUTINE'); print(D.round(2).to_string(), flush=True)
    P = S_[(S_.scope == 'all') & (S_.h == 5) & (S_.read == 'OPP')].set_index('split')
    Yp = Y[(Y.scope == 'all') & (Y.h == 5) & (Y.read == 'OPP')].set_index('year')
    d5 = D[D.h == 5].set_index('split'); v = P.loc['VAL']
    kills = {'VAL gross<=30bp': v.excess_bp <= 30, 'VAL |NWt|<2': abs(v.nw_t) < 2,
             'TRAIN halves opposite sign': np.sign(P.loc['T_H1_16_18'].excess_bp) != np.sign(P.loc['T_H2_19_20'].excess_bp),
             'VAL ex-top5<=0': v.ex_top5_bp <= 0, '2022 negative': Yp.loc['2022'].excess_bp < 0, 'VAL net<=0': v.net_bp <= 0,
             'OPP not above ROUTINE on TRAIN': d5.loc['TRAIN'].diff_bp <= 0, 'OPP not above ROUTINE on VAL': d5.loc['VAL'].diff_bp <= 0,
             '<3 events/wk (VAL)': v.per_wk < 3}
    log('KILL LIST (primary OPP h5 all-universe): ' + '; '.join(f'{k}={bool(x)}' for k, x in kills.items()))
    log('VERDICT: ' + ('KILL' if any(bool(x) for x in kills.values()) else 'SURVIVE'))


if __name__ == '__main__':
    main()
