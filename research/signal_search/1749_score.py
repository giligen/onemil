"""Cell 1,749 scorer (PREREG_1749): volume shock WITHOUT a price move (Gervais-Kaniel-Mingelgrin 2001), daily resolution.

Signal on day t: volume_t is the highest or 2nd highest of the stock's own 50 bars (t-49..t, adjusted volume);
LOW = lowest or 2nd lowest.  npm (no price move) = |close-to-close ret_t| <= 1.5 x median |ret| of the 49 prior days.
Entry open t+1, exit close t+5 (primary) / t+20.  Excess = leg mean - tier-universe mean on the same signal day.
Tiers are point-in-time from t-1 (20d median raw $ volume, raw px): liquid > $100M px>=5; mid $10-100M px>=5;
rank 1,001-3,000 (px>=3, dv20>=2M).  Bars: 1748_cache (superset of 1746).  TEST (signal date >= 2024) hard-excluded.
Windows touching a >10 % adjusted/raw price-ratio jump (split) between t-50 and t+h are dropped (volume scale).
Usage: nice -n 19 python3 1749_score.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SEAL = '2024-01-01'
SPLITS = {'TRAIN': ('2016-01-01', '2020-12-31'), 'T_H1_16_18': ('2016-01-01', '2018-12-31'),
          'T_H2_19_20': ('2019-01-01', '2020-12-31'), 'VAL': ('2021-01-01', '2023-12-31')}
W, K_NPM, MIN_U = 50, 1.5, 50
TIERS = ('liquid', 'mid', 'rank1k3k')
EXTRA_BP = 2.0
t0 = time.time()


def log(m):
    print(f'[{time.time() - t0:6.0f}s] {m}', flush=True)


def nw_t(x, lag=5):
    """Newey-West (Bartlett) mean, t_nw, t_iid, se_nw of series x."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n, m = len(x), x.mean()
    e = x - m
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = np.sqrt(v / n)
    return m, m / se, m / (e.std(ddof=1) / np.sqrt(n)), se


def load_matrices():
    """Pivot raw+adjusted bars to (date x symbol) matrices."""
    cache = os.path.join(HERE, '1748_cache')
    raw = pd.read_parquet(os.path.join(cache, 'bars_raw.parquet'))
    adj = pd.read_parquet(os.path.join(cache, 'bars_all.parquet'))
    log(f'bars raw {raw.shape} adj {adj.shape}')
    dates = sorted(adj.date.unique()); syms = sorted(adj.symbol.unique())
    di = {d: i for i, d in enumerate(dates)}; si = {s: i for i, s in enumerate(syms)}

    def piv(df, col):
        m = np.full((len(dates), len(syms)), np.nan, np.float32)
        m[df.date.map(di).to_numpy(), df.symbol.map(si).to_numpy()] = df[col].to_numpy()
        return m
    M = {'o': piv(adj, 'o'), 'c': piv(adj, 'c'), 'v': piv(adj, 'v')}
    raw['dv'] = raw.c * raw.v
    M['craw'] = piv(raw, 'c'); M['dv'] = piv(raw, 'dv')
    del raw, adj
    return np.array(dates), np.array(syms), M


def derive(dates, M):
    """PIT tiers, returns, split flags, volume-shock flags."""
    T, S = M['c'].shape
    c = M['c']
    ret = np.full((T, S), np.nan, np.float32); ret[1:] = c[1:] / c[:-1] - 1
    dv = pd.DataFrame(M['dv'])
    adv = dv.shift(1).rolling(20, min_periods=15).median().to_numpy()      # known at t-1
    pxp = np.full((T, S), np.nan, np.float32); pxp[1:] = M['craw'][:-1]
    rank = pd.DataFrame(adv).rank(axis=1, ascending=False, method='first').to_numpy()
    tier = {'liquid': (adv > 100e6) & (pxp >= 5), 'mid': (adv > 10e6) & (adv <= 100e6) & (pxp >= 5),
            'rank1k3k': (rank >= 1001) & (rank <= 3000) & (pxp >= 3) & (adv >= 2e6)}
    # split / scale-break flag: adjusted/raw ratio jump > 10 %
    ratio = c / M['craw']
    jump = np.zeros((T, S), np.int32); jump[1:] = (np.abs(ratio[1:] / ratio[:-1] - 1) > 0.10)
    jump = np.nan_to_num(jump)
    cs = np.cumsum(jump, axis=0)
    # forward returns
    fw = {}
    o1 = np.full((T, S), np.nan, np.float32); o1[:-1] = M['o'][1:]
    for h in (5, 20):
        ch = np.full((T, S), np.nan, np.float32); ch[:-h] = c[h:]
        fw[h] = ch / o1 - 1
    # bad window: jump in (t-W, t+h]
    bad = {}
    for h in (5, 20):
        b = np.zeros((T, S), bool)
        for t in range(T):
            lo, hi = max(t - W, 0), min(t + h, T - 1)
            b[t] = (cs[hi] - cs[lo]) > 0
        bad[h] = b
    return ret, tier, fw, bad


def shock_flags(v, ret):
    """HIGH/LOW (top/bottom-2 of 50 own volumes) and npm flags, chunked over dates."""
    T, S = v.shape
    hi = np.zeros((T, S), bool); lo = np.zeros((T, S), bool); npm = np.zeros((T, S), bool); okm = np.zeros((T, S), bool)
    absr = np.abs(ret)
    CH = 60
    for a in range(W - 1, T, CH):
        b = min(a + CH, T)
        win = sliding_window_view(v[a - W + 1:b], W, axis=0)       # (b-a, S, W)
        full = ~np.isnan(win).any(-1)
        cur = win[..., -1]
        gt = (win > cur[..., None]).sum(-1); lt = (win < cur[..., None]).sum(-1)
        hi[a:b] = full & (gt <= 1); lo[a:b] = full & (lt <= 1); okm[a:b] = full
        rw = sliding_window_view(absr[a - W + 1:b], W, axis=0)[..., :-1]   # 49 prior days
        med = np.nanmedian(np.where(np.isnan(rw).sum(-1)[..., None] > 5, np.nan, rw), axis=-1)
        npm[a:b] = full & (absr[a:b] <= K_NPM * med)
        log(f'  shock flags through {dates_g[b - 1]}')
    return hi, lo, npm, okm


def main():
    global dates_g
    dates, syms, M = load_matrices(); dates_g = dates
    ret, tier, fw, bad = derive(dates, M)
    log('derived')
    hi, lo, npm, okm = shock_flags(M['v'], ret)
    valid_t = dates < SEAL
    R = {h: np.where(bad[h], np.nan, fw[h]) for h in (5, 20)}
    # daily universe mean/count per tier (valid = ok window, ret defined)
    base = okm & ~np.isnan(ret)
    U, N = {}, {}
    for tn in TIERS:
        for h in (5, 20):
            m = tier[tn] & base & ~np.isnan(R[h])
            N[tn, h] = m.sum(1)
            U[tn, h] = np.where(N[tn, h] > 0, np.nansum(np.where(m, R[h], 0), 1) / np.maximum(N[tn, h], 1), np.nan)
    # signal rows
    sig = (hi | lo) & base & (tier['liquid'] | tier['mid'] | tier['rank1k3k'])
    sig[~valid_t] = False
    ti, si = np.nonzero(sig)
    rows = pd.DataFrame({'date': dates[ti], 'symbol': syms[si], 'high': hi[ti, si], 'low': lo[ti, si], 'npm': npm[ti, si],
                         'ret_t': ret[ti, si], 'r5': R[5][ti, si], 'r20': R[20][ti, si]})
    for tn in TIERS:
        rows[tn] = tier[tn][ti, si]
    rows.to_parquet(os.path.join(HERE, '1749_signals.parquet'))
    log(f'signal rows {len(rows)}')
    Udf = {}
    for tn in TIERS:
        for h in (5, 20):
            Udf[tn, h] = pd.DataFrame({'U': U[tn, h], 'N': N[tn, h]}, index=dates)
    # cost tables
    c46 = pd.read_csv(os.path.join(HERE, '1746_cost.csv')).half_spread_bp.dropna()
    c48 = pd.read_csv(os.path.join(HERE, '1748_cost.csv'))
    cost_rt = {'liquid': 2 * c46.mean() + 2 * EXTRA_BP,
               'mid': c48.entry_hs_bp.mean() + c48.exit_hs_bp.mean() + 2 * EXTRA_BP,
               'rank1k3k': c48.entry_hs_bp.mean() + c48.exit_hs_bp.mean() + 2 * EXTRA_BP}
    log(f'round-trip cost bp: {cost_rt} (liquid hs mean {c46.mean():.1f} med {c46.median():.1f}; 1748 entry {c48.entry_hs_bp.mean():.1f} exit {c48.exit_hs_bp.mean():.1f})')
    summ, yearly = [], []
    SETS = {'HIGH_npm': lambda r: r.high & r.npm, 'HIGH_all': lambda r: r.high, 'HIGH_ext': lambda r: r.high & ~r.npm,
            'LOW_all': lambda r: r.low, 'LOW_npm': lambda r: r.low & r.npm}
    for tn in TIERS:
        for h in (5, 20):
            rc = f'r{h}'
            u = Udf[tn, h]; u = u[(u.N >= MIN_U)]
            rt = rows[rows[tn] & rows[rc].notna() & rows.date.isin(u.index)]
            daily = {}
            for sn, f in SETS.items():
                x = rt[f(rt)]
                g = x.groupby('date')[rc].agg(['mean', 'size'])
                daily[sn] = g
            for split, (a, b) in SPLITS.items():
                def sl(s):
                    return s[(s.index >= a) & (s.index <= b)]
                ud = sl(u.U)
                nd = len(sl(u))
                for sn in SETS:
                    g = sl(daily[sn]['mean']); n = sl(daily[sn]['size'])
                    ex = g - ud.reindex(g.index)
                    if len(ex) < 30:
                        continue
                    m, t, ti_, se = nw_t(ex)
                    x = rt[SETS[sn](rt) & (rt.date >= a) & (rt.date <= b)].copy()
                    x['U'] = u.U.reindex(x.date).to_numpy(); x['nd'] = x.groupby('date')[rc].transform('size')
                    contrib = (x[rc] - x.U) / x.nd / nd
                    cut = int(np.ceil(0.05 * len(x)))
                    ex5 = contrib.sort_values(ascending=False).iloc[cut:].sum()
                    capc = ((x[rc].clip(upper=0.10) - x.U) / x.nd / nd).sum()
                    mon = ex.groupby(ex.index.str[:7]).mean()
                    summ.append(dict(tier=tn, h=h, split=split, set=sn, days=len(ex), sig_per_wk=len(x) / (nd / 5), mean_n_day=n.mean(),
                                     excess_bp=m * 1e4, nw_t=t, iid_t=ti_, ex_top5_bp=ex5 * 1e4, cap10_bp=capc * 1e4,
                                     green_months=(mon > 0).mean(), mde_bp=2.8 * se * 1e4, net_bp=m * 1e4 - cost_rt[tn],
                                     raw_mean_bp=g.mean() * 1e4))
                # differences
                for nm, (s1, s2) in {'HIGH_npm-LOW_all': ('HIGH_npm', 'LOW_all'), 'HIGH_all-LOW_all': ('HIGH_all', 'LOW_all'),
                                     'HIGH_npm-HIGH_ext': ('HIGH_npm', 'HIGH_ext')}.items():
                    d = (sl(daily[s1]['mean']) - sl(daily[s2]['mean'])).dropna()
                    if len(d) < 30:
                        continue
                    m, t, ti_, se = nw_t(d)
                    summ.append(dict(tier=tn, h=h, split=split, set=nm, days=len(d), excess_bp=m * 1e4, nw_t=t, iid_t=ti_,
                                     mde_bp=2.8 * se * 1e4))
            for sn in ('HIGH_npm', 'HIGH_all'):
                g = daily[sn]['mean']; ex = (g - u.U.reindex(g.index)).dropna()
                for y, xx in ex.groupby(ex.index.str[:4]):
                    yearly.append(dict(tier=tn, h=h, set=sn, year=y, excess_bp=xx.mean() * 1e4, nw_t=nw_t(xx)[1] if len(xx) > 20 else np.nan, days=len(xx)))
    S = pd.DataFrame(summ); Y = pd.DataFrame(yearly)
    S.to_csv(os.path.join(HERE, '1749_summary.csv'), index=False); Y.to_csv(os.path.join(HERE, '1749_yearly.csv'), index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 500)
    log('SUMMARY (h=5, all tiers)')
    print(S[S.h == 5].round(2).to_string(), flush=True)
    log('SUMMARY (h=20, HIGH_npm only)')
    print(S[(S.h == 20) & (S.set == 'HIGH_npm')].round(2).to_string(), flush=True)
    log('YEARLY (h=5)')
    print(Y[Y.h == 5].round(2).to_string(), flush=True)
    log(f'extremes: r5 min {rows.r5.min():.3f} max {rows.r5.max():.3f}; r20 max {rows.r20.max():.3f}')
    # KILL list on the primary
    P = S[(S.tier == 'liquid') & (S.h == 5) & (S.set == 'HIGH_npm')].set_index('split')
    Yp = Y[(Y.tier == 'liquid') & (Y.h == 5) & (Y.set == 'HIGH_npm')].set_index('year')
    D = S[(S.tier == 'liquid') & (S.h == 5) & (S.set == 'HIGH_npm-HIGH_ext')].set_index('split')
    v = P.loc['VAL']
    kills = {'VAL gross<=15bp': v.excess_bp <= 15, 'VAL |NWt|<2': abs(v.nw_t) < 2,
             'TRAIN halves opposite sign': np.sign(P.loc['T_H1_16_18'].excess_bp) != np.sign(P.loc['T_H2_19_20'].excess_bp),
             'VAL ex-top5<=0': v.ex_top5_bp <= 0, '2022 negative': Yp.loc['2022'].excess_bp < 0, 'VAL net<=0': v.net_bp <= 0,
             'npm not above ext on TRAIN': D.loc['TRAIN'].excess_bp <= 0, 'npm not above ext on VAL': D.loc['VAL'].excess_bp <= 0,
             '<3 sig/wk': v.sig_per_wk < 3}
    log('KILL LIST (primary liquid HIGH_npm h5): ' + '; '.join(f'{k}={bool(x)}' for k, x in kills.items()))
    log('VERDICT: ' + ('KILL' if any(bool(x) for x in kills.values()) else 'SURVIVE'))


if __name__ == '__main__':
    main()
