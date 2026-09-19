#!/usr/bin/env python3
"""hod_filter_stack — stage 3: assemble the scored population.

sig2.csv (the bar pass) + blite.csv (the raw-break stream) + SPY + daily context + news
-> pop.csv, one row per first-qualifying break per declared combo, with every PREREG §3 feature.

Read-only on every DB. Checkpointed: writes pop.csv and availability.csv and exits.
"""
import os, sqlite3, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
from research.scripts.pit_listings import is_test_ticker   # noqa: E402

D = f'{ROOT}/research/mature_method/hod_filter_stack'
CF = f'{ROOT}/research/bf_zero/causal_filter'
EARLY_CLOSE = {'2025-07-03', '2025-11-28', '2025-12-24'}
RD = dict(dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- daily context
def daily_ctx(symbols):
    """prev-day context + 20d high, from the FULL daily_bars table (not the screened universe
    file -- the screen is only ~1,560 names a day, so prev-day fields taken from it are missing for
    ~22% of signals and that missingness correlates with the outcome).  Only the symbols this study
    actually scores are loaded."""
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=180)
    syms = sorted(symbols)
    parts = []
    for i in range(0, len(syms), 900):
        ch = syms[i:i + 900]
        parts.append(pd.read_sql(
            'select symbol, bar_date as day, open, high, low, close from daily_bars '
            f"where symbol in ({','.join('?' * len(ch))})", con, params=ch))
    con.close()
    u = pd.concat(parts, ignore_index=True)
    for k in ('open', 'high', 'low', 'close'):
        u[k] = pd.to_numeric(u[k], errors='coerce')
    u = u[u.close > 0].sort_values(['symbol', 'day'], kind='mergesort')
    g = u.groupby('symbol', sort=False)
    u['prev_close'] = g.close.shift(1)
    u['prev_high'] = g.high.shift(1)
    u['prev_low'] = g.low.shift(1)
    u['high20'] = g.high.shift(1).rolling(20, min_periods=10).max().reset_index(level=0, drop=True)
    u['gap_pct'] = (u.open / u.prev_close - 1.0) * 100.0
    rng = u.prev_high - u.prev_low
    u['prev_range_pct'] = rng / u.prev_close * 100.0
    u['prev_close_pos'] = ((u.prev_close - u.prev_low) / rng).where(rng > 0)
    return u[['symbol', 'day', 'gap_pct', 'prev_range_pct', 'prev_close_pos', 'high20']]


# ---------------------------------------------------------------- SPY
def spy_ctx():
    s = pd.read_csv(f'{CF}/spy_1min.csv')
    s = s[(s.m >= 570) & (s.m < 960)].sort_values(['bar_date', 'm'])
    s['hod'] = s.groupby('bar_date').close.cummax()
    s['spy_dist_hod_pct'] = (s.close / s.hod - 1.0) * 100.0
    op = s.groupby('bar_date').close.transform('first')
    s['spy_ret_open_sig'] = (s.close / op - 1.0) * 100.0
    g = s.groupby('bar_date').close
    s['spy_ret_30m'] = (s.close / g.shift(30) - 1.0) * 100.0
    s['spy_5m_ret'] = (s.close / g.shift(5) - 1.0) * 100.0
    s['spy_range3'] = ((g.rolling(3, min_periods=3).max() - g.rolling(3, min_periods=3).min())
                       .reset_index(level=0, drop=True) / s.close * 100.0)
    per_min = s[['bar_date', 'm', 'spy_dist_hod_pct', 'spy_ret_open_sig', 'spy_ret_30m',
                 'spy_5m_ret', 'spy_range3']].rename(columns={'bar_date': 'day', 'm': 'break_m'})
    # day level: SPY 09:30->10:00, and SPY vs its prior close at 10:00
    d1000 = s[s.m == 600][['bar_date', 'close']].rename(columns={'bar_date': 'day', 'close': 'c1000'})
    dop = s.groupby('bar_date').close.first().rename('c0930').reset_index().rename(columns={'bar_date': 'day'})
    dd = d1000.merge(dop, on='day', how='outer')
    dd['spy_ret_0930_1000'] = (dd.c1000 / dd.c0930 - 1.0) * 100.0
    # SPY 20d realised vol (T-1) from daily_bars  -- the VIX proxy
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    sd = pd.read_sql("select bar_date as day, close from daily_bars where symbol='SPY' order by bar_date", con)
    con.close()
    sd['r'] = np.log(sd.close).diff()
    sd['spy_vol20'] = sd.r.rolling(20).std().shift(1) * np.sqrt(252) * 100.0
    dd = dd.merge(sd[['day', 'spy_vol20']], on='day', how='left')
    return per_min, dd[['day', 'spy_ret_0930_1000', 'spy_vol20']]


# ---------------------------------------------------------------- breadth / cohort
def breadth(sig):
    """From the RAW break stream, deduped to the FIRST break per (day, symbol) so every count is a
    DISTINCT-SYMBOL count. Everything is strictly causal: `_15m` and `_day` use minutes STRICTLY
    before the signal's break bar; `_min` is the same minute (bars close simultaneously) and is
    flagged as the marginal one."""
    b = pd.read_csv(f'{D}/blite.csv', **RD)
    b = b.sort_values(['day', 'break_m', 'symbol'], kind='mergesort').drop_duplicates(['day', 'symbol'])
    out = {}
    day_breadth = {}
    for day, gg in b.groupby('day', sort=False):
        bm = gg.break_m.values.astype(np.int32)
        dist = gg.dist_open_pct.values.astype(np.float64)
        ret = gg.ret_open_sig.values.astype(np.float64)
        day_breadth[day] = int((bm <= 600).sum())
        out[day] = (bm, dist, ret)
    cols = {k: np.full(len(sig), np.nan) for k in
            ('breadth_min', 'breadth_15m', 'breadth_day', 'cohort_rank_dist', 'rs_vs_cohort')}
    days = sig.day.values; ms = sig.break_m.values.astype(np.int32)
    dsig = sig.dist_open_pct.values.astype(np.float64); rsig = sig.ret_open_sig.values.astype(np.float64)
    for day, gi in pd.Series(np.arange(len(sig))).groupby(pd.Series(days)):
        if day not in out:
            continue
        bm, dist, ret = out[day]
        for i in gi.values:
            m = ms[i]
            pre = bm < m
            cols['breadth_min'][i] = int((bm == m).sum())
            cols['breadth_15m'][i] = int(((bm >= m - 15) & pre).sum())
            cols['breadth_day'][i] = int(pre.sum())
            coh = bm <= m
            nc = int(coh.sum())
            if nc >= 2:
                cols['cohort_rank_dist'][i] = float((dist[coh] < dsig[i]).sum()) / (nc - 1)
                cols['rs_vs_cohort'][i] = (rsig[i] - float(np.nanmean(ret[coh]))) * 100.0
    for k, v in cols.items():
        sig[k] = v
    return sig, pd.DataFrame({'day': list(day_breadth), 'breadth_by_1000': list(day_breadth.values())})


# ---------------------------------------------------------------- news
def news_ctx(sig):
    n = pd.read_csv(f'{CF}/news.csv', **RD)
    n = n[['symbol', 'day', 'n_articles', 'latest']].drop_duplicates(['symbol', 'day'])
    n['news_covered'] = 1
    s = sig.merge(n, on=['symbol', 'day'], how='left')
    lt = pd.to_datetime(s.latest, errors='coerce', utc=True).dt.tz_convert('America/New_York')
    sig_t = pd.to_datetime(s.day) + pd.to_timedelta(s.break_m, unit='m')
    rec = (sig_t - lt.dt.tz_localize(None)).dt.total_seconds() / 60.0
    s['news_recency_min'] = rec.where(s.n_articles.fillna(0) > 0)
    s['news_n'] = s.n_articles
    s['has_news'] = (s.n_articles.fillna(0) > 0).astype(float).where(s.news_covered == 1)
    return s.drop(columns=['latest', 'n_articles'])


# ---------------------------------------------------------------- main
def main():
    log('loading sig2 ...')
    sig = pd.read_csv(f'{D}/sig2.csv', **RD)
    n0 = len(sig)
    sig = sig[~sig.day.isin(EARLY_CLOSE)]
    sig = sig[~sig.symbol.map(is_test_ticker)]
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    dbs = set(pd.read_sql('select distinct symbol from daily_bars', con).symbol.astype(str))
    con.close()
    sig = sig[sig.symbol.isin(dbs)].reset_index(drop=True)
    log(f'membership: {n0} -> {len(sig)}')

    log('daily context ...')
    sig = sig.merge(daily_ctx(set(sig.symbol.unique())), on=['symbol', 'day'], how='left')
    sig['dist_20d_high_pct'] = (sig.level / sig.high20 - 1.0) * 100.0
    sig['dow'] = pd.to_datetime(sig.day).dt.dayofweek

    log('spy ...')
    per_min, per_day = spy_ctx()
    sig = sig.merge(per_min, on=['day', 'break_m'], how='left')

    log('breadth / cohort ...')
    sig, dbr = breadth(sig)
    per_day = per_day.merge(dbr, on='day', how='outer')

    log('news ...')
    sig = news_ctx(sig)

    log('derived ...')
    # the four OLD features this study cannot rebuild from bars alone (they need the per-symbol
    # prior-20-session volume profile / the pullback anatomy) are MERGED from the causal-filter
    # study's own feature file on the exact (day, symbol, entry_m) key. Coverage is the B0 signal
    # set, so the old-vs-new comparison is run on B0 where it is ~complete, and the coverage is
    # printed wherever a B2 number uses them.
    cf = pd.read_csv(f'{CF}/features.csv', **RD)[
        ['day', 'symbol', 'entry_m', 'bar_vol_x', 'drive_min', 'rv_clock', 'coh_by_t']
    ].drop_duplicates(['day', 'symbol', 'entry_m'])
    sig = sig.merge(cf, on=['day', 'symbol', 'entry_m'], how='left')
    sig['above_vwap'] = (sig.vwap_dist_pct > 0).astype(float)
    sig['n_prior'] = sig.n_break
    from trading.orb_asset_class import _lev_family_symbols, load_class_map, WRAPPER
    cmap = load_class_map()
    lev = set(_lev_family_symbols()) | {s for s, k in cmap.items() if k == WRAPPER}
    sig['is_wrapper'] = sig.symbol.isin(lev).astype(float)
    # the symbol's OWN prior HOD-break history -- strictly prior days, expanding
    s = sig[sig.first_n0 == 1].sort_values(['symbol', 'day'], kind='mergesort')
    g = s.groupby('symbol', sort=False)
    s['sym_prior_n'] = g.cumcount()
    s['sym_prior_meanR'] = (g.rr_n.apply(lambda x: x.shift(1).expanding().mean())
                            .reset_index(level=0, drop=True))
    sig = sig.merge(s[['day', 'symbol', 'break_m', 'sym_prior_n', 'sym_prior_meanR']],
                    on=['day', 'symbol', 'break_m'], how='left')
    # "no prior history" is a state that is FULLY KNOWN at the decision instant, so it is encoded
    # explicitly rather than left missing (leaving it missing made the field look outcome-dependent).
    sig['sym_prior_n'] = sig.sym_prior_n.fillna(0.0)
    sig['sym_prior_meanR'] = sig.sym_prior_meanR.where(sig.sym_prior_n > 0, 0.0)

    per_day.to_csv(f'{D}/day_ctx.csv', index=False)
    sig.to_csv(f'{D}/pop.csv', index=False)
    log(f'pop.csv {len(sig)} rows, {sig.shape[1]} cols; day_ctx.csv {len(per_day)} days')


if __name__ == '__main__':
    main()
