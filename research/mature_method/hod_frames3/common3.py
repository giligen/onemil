#!/usr/bin/env python3
"""hod_frames3 — shared loaders and the cell machine.  ONE definition for F11, F10 and F12."""
import os, sqlite3, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S            # noqa: E402
import score2 as S2          # noqa: E402
from research.scripts.pit_listings import is_test_ticker   # noqa: E402

D = f'{ROOT}/research/mature_method/hod_frames3'
RD = dict(dtype={'symbol': str, 'day': str, 'why_n': str}, keep_default_na=False, na_values=[''])
BR_COLS = ['day', 'symbol', 'entry_m', 'break_m', 'level', 'open_px', 'next_open', 'dist_open_pct',
           'rv_profile', 'adv20', 'cumv', 'cum_dollar', 'bar_vol', 'n_break', 'n_prior',
           'fill_capped', 'stop_n', 'r_pct_n', 'rr_n', 'why_n', 'exit_m_n', 'hod_age_bars',
           'rng_sig', 'rng_day']


def clustered_t(d, col='net'):
    """Cluster-robust t of the mean, clusters = trading DAYS (hod_preopen_regime §4's rule)."""
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


def sigset(d, min_price=20.0, r_min=1.0, max_bps=100.0, max_frac_r=0.15, last_m=840):
    """The shipped pre-book cascade (identical to hod_frames2/score.sigset)."""
    d = d[(d.entry_m <= last_m + 1) & d.r_pct_n.notna() & (d.fill_capped == 1) &
          (d.r_pct_n >= r_min) & (d.next_open >= min_price)]
    x = S.attach_cost(d, 'n')
    x = x[(x.sp_pct * 100) <= max_bps]
    x = x[(x.sp_pct / x.r_pct.clip(lower=0.05)) <= max_frac_r]
    return x[x.obtainable.astype(bool)]


def daily_ctx():
    """ADV$ and the 20-prior-session median daily range %, from universe.csv — strictly prior."""
    u = pd.read_csv(f'{ROOT}/research/bf_zero/universe.csv',
                    usecols=['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume'],
                    dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    u = u.rename(columns={'bar_date': 'day'})
    for k in ('open', 'high', 'low', 'close', 'volume'):
        u[k] = pd.to_numeric(u[k], errors='coerce')
    u = u.sort_values(['symbol', 'day'], kind='mergesort')
    u['dv'] = u.close * u.volume
    u['rngp'] = (u.high - u.low) / u.open.replace(0, np.nan) * 100
    g = u.groupby('symbol', sort=False)
    u['adv_dollar'] = g.dv.transform(lambda s: s.rolling(20, min_periods=15).mean().shift(1))
    u['med_rng'] = g.rngp.transform(lambda s: s.rolling(20, min_periods=15).median().shift(1))
    u['day_range_pct'] = u.rngp
    return u[['day', 'symbol', 'adv_dollar', 'med_rng', 'day_range_pct']]


def load_breaks(verbose=True):
    """breaks2.csv + the F11 attach passes + NBBO + day context.  ONE population for all frames."""
    br = pd.read_csv(f'{ROOT}/research/mature_method/hod_frames2/breaks2.csv',
                     usecols=BR_COLS, **RD)
    n0 = len(br)
    br = br[~br.day.isin(S.EARLY_CLOSE)]
    br = br[~br.symbol.map(lambda s: is_test_ticker(str(s)))]
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    dbs = set(pd.read_sql('select distinct symbol from daily_bars', con).symbol.astype(str))
    con.close()
    br = br[br.symbol.isin(dbs)]
    for f in ('feat3.csv', 'feat3b.csv'):
        g = pd.read_csv(f'{D}/{f}', dtype={'day': str, 'symbol': str}, keep_default_na=False,
                        na_values=['']).drop_duplicates(['day', 'symbol', 'break_m'])
        br = br.merge(g, on=['day', 'symbol', 'break_m'], how='left')
    br['split'] = S.split_of(br.day.values)
    br['wk'] = pd.to_datetime(br.day).dt.to_period('W-FRI').astype(str)
    nb = pd.read_csv(f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv',
                     dtype={'symbol': str, 'day': str}, keep_default_na=False,
                     na_values=['']).drop_duplicates(['day', 'symbol', 'entry_m'])
    for extra in (f'{D}/nbbo3.csv',):                 # the F10 dedicated fetch, when it exists
        if os.path.exists(extra):
            e = pd.read_csv(extra, dtype={'symbol': str, 'day': str}, keep_default_na=False,
                            na_values=['']).drop_duplicates(['day', 'symbol', 'entry_m'])
            nb = pd.concat([nb, e[~e.set_index(['day', 'symbol', 'entry_m']).index.isin(
                nb.set_index(['day', 'symbol', 'entry_m']).index)]], ignore_index=True)
    br = br.merge(nb[['day', 'symbol', 'entry_m', 'spread_mean', 'ask_dec', 'bid_dec']],
                  on=['day', 'symbol', 'entry_m'], how='left')
    br = br.merge(daily_ctx(), on=['day', 'symbol'], how='left')
    df = pd.read_csv(f'{ROOT}/research/mature_method/hod_preopen_regime/day_fields.csv',
                     dtype={'day': str}, keep_default_na=False, na_values=[''])
    br = br.merge(df[['day', 'spy_r5_pct']], on='day', how='left')
    br['rng_after'] = br.rng_day - br.rng_sig
    br['dollar_frac'] = br.cum_dollar / br.adv_dollar.replace(0, np.nan) * 100
    br = br.sort_values(['day', 'symbol', 'break_m'], kind='mergesort').reset_index(drop=True)
    if verbose:
        print(f'break rows {n0} -> {len(br)} after membership | symbol-days '
              f'{br.drop_duplicates(["day","symbol"]).shape[0]} | days {br.day.nunique()}',
              flush=True)
    return br


def admit(br, mask):
    """KEEP-SCANNING admission: the FIRST qualifying break of the symbol-day satisfying `mask`."""
    d = br[mask.fillna(False).values] if hasattr(mask, 'fillna') else br[mask]
    return d.drop_duplicates(['day', 'symbol'], keep='first')
