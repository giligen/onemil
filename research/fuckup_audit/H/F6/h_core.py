#!/usr/bin/env python3
"""Stage H / F6 — the shared core: population, cost contract (c), book, features.

The cost/book code is Stage C's contract as implemented in `research/fuckup_audit/B/score5.py`
(re-stated here in code; the builder `B/build_candidates4.py`'s header remains the specification):

  POPULATION  fam == 'F6', the next-open fill present and >= $5, 570 <= next_entry_m <= 841,
              the scored variant's r_pct >= 1.0 of price, and (primary) range_so_far_pct >= 5 on
              bars STRICTLY BEFORE the signal bar.
  COST        half = 0.5 * (spread_cc_bps / 100) / max(r_pct_variant, 0.05)      [R units]
              entry 0.25 x half (the next-open fill is an already-printed, ask-side price)
              exit   {stop: 0.875, lock: 0.875, eod: 0.412, target: 0.875, none: 0.875} x half
  BOOK        trading.hod_break.run_book(rows, 12, 4)
  VARIANTS    'hold'   = hold to 15:55 with the TOUCH stop      (primary)
              'stopm1' = +2R on a close, stop at level - 1%     (secondary)

Two populations are carried side by side (see `h_extract.py`):
  P = C/pop_c.csv            >=5%-range-day universe (Stage B/C/D1); F6 rows all carry the floor
  Q = E/candidates_causal.csv causal U1 u U2 universe (Stage E); NO floor -> the floor twin lives here
"""
import os, sys, sqlite3
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading.hod_break import run_book                                    # noqa: E402

H = f'{ROOT}/research/fuckup_audit/H/F6'
FA = f'{ROOT}/research/fuckup_audit'
RD = dict(keep_default_na=False, na_values=[''])

ENTRY_MULT_NEXT = 0.25
EXIT_RATIO = {'stop': 0.875, 'lock': 0.875, 'eod': 0.412, 'target': 0.875, 'none': 0.875}
VARIANTS = {                       # name -> (rr col, why col, exit-minute col, r_pct col)
    'hold':   ('next_rr_hold', 'next_why_hold', 'next_exit_m_hold', 'next_r_pct'),
    'stopm1': ('next_rr_2r_stopm1', 'next_why_2r_stopm1', 'next_exit_m_2r_stopm1', 'next_r_pct_m1'),
}
BANDS = [(570, 575, '0930_0935'), (575, 600, '0935_1000'), (600, 660, '1000_1100'),
         (660, 780, '1100_1300'), (780, 842, '1300_1401')]


def split_of(day):
    return np.where(day < '2026-01-01', 'TRAIN', np.where(day < '2026-06-01', 'VAL', 'TEST'))


def band_of(m):
    o = pd.Series('other', index=m.index)
    for a, b, nm in BANDS:
        o[(m >= a) & (m < b)] = nm
    return o


# --------------------------------------------------------------------------- feature attachment
def _news():
    fr = []
    for p in (f'{FA}/E/news_presence_e.csv', f'{FA}/D/news_presence.csv'):
        fr.append(pd.read_csv(p, usecols=['day', 'symbol', 'n_prev15_to_0930', 'fetch_ok'],
                              dtype={'day': str, 'symbol': str}, **RD))
    n = pd.concat(fr, ignore_index=True).drop_duplicates(['day', 'symbol'])
    n['has_news'] = (n.n_prev15_to_0930.fillna(0) > 0).astype(int)
    n['news_known'] = 1
    return n[['day', 'symbol', 'has_news', 'news_known']]


def _pmdb():
    con = sqlite3.connect(f'file:{FA}/D/pm_bars.db?mode=ro', uri=True)
    d = pd.read_sql('select day, symbol, pm_dollar_vol as pm_db, src as pm_src from pm', con)
    con.close()
    return d


def _dayfeat():
    d = pd.read_csv(f'{FA}/day_features.csv', dtype={'day': str}, **RD)
    return d


def _etf_state():
    """SPY/IWM cumulative return from the 09:30 open to each RTH minute — causal at that minute.

    Cached to `h_etf_state.csv` (the source table has no index on `symbol`, so each read is a full
    scan of a 2 GB file; the cache is a pure derivative of it and is rebuilt by deleting the file)."""
    cache = f'{H}/h_etf_state.csv'
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={'day': str}, **RD)
    con = sqlite3.connect(f'file:{ROOT}/research/lit_review_2026/etf_1min.db?mode=ro', uri=True)
    out = []
    for sym in ('SPY', 'IWM'):
        b = pd.read_sql("select t, o, c from bars where symbol=? order by t", con, params=(sym,))
        if not len(b):
            continue
        ts = pd.to_datetime(b.t, utc=True, format='mixed').dt.tz_convert('America/New_York')
        b['day'] = ts.dt.strftime('%Y-%m-%d')
        b['m'] = ts.dt.hour * 60 + ts.dt.minute
        b = b[(b.m >= 570) & (b.m <= 960)]
        op = b[b.m == 570].set_index('day').o
        b[f'{sym.lower()}_ret'] = (b.c / b.day.map(op) - 1.0) * 100.0
        out.append(b[['day', 'm', f'{sym.lower()}_ret']])
    con.close()
    if not out:
        return pd.DataFrame(columns=['day', 'm', 'spy_ret', 'iwm_ret'])
    e = out[0]
    for x in out[1:]:
        e = e.merge(x, on=['day', 'm'], how='outer')
    e.to_csv(cache, index=False)
    return e


def load(pop='P', with_features=True):
    """Read the F6 extract of one population, attach features, compute both variants' net R."""
    d = pd.read_csv(f'{H}/f6_{pop}.csv', dtype={'day': str, 'symbol': str, 'fam': str, 'cfg': str,
                                                'asset_class': str}, low_memory=False, **RD)
    d['split'] = split_of(d.day.values)
    d['wk'] = pd.to_datetime(d.day).dt.to_period('W-FRI').astype(str)
    d['band'] = band_of(d.next_entry_m)
    d['dow'] = pd.to_datetime(d.day).dt.dayofweek
    d['mon'] = d.day.str[:7]
    # --- net R for each variant under contract (c)
    sp = d.spread_cc_bps / 100.0
    for v, (rr, why, xm, rp) in VARIANTS.items():
        half = 0.5 * sp / d[rp].clip(lower=0.05)
        d[f'half_{v}'] = half
        d[f'net_{v}'] = d[rr] - ENTRY_MULT_NEXT * half - half * d[why].map(EXIT_RATIO).fillna(0.875)
        d[f'gross_{v}'] = d[rr]
    if not with_features:
        return d
    # --- signal-bar shape (causal: the bar that produced the signal)
    rng = (d.sig_h - d.sig_l).replace(0, np.nan)
    d['sig_close_pos'] = (d.sig_c - d.sig_l) / rng
    d['sig_body_pct'] = (d.sig_c - d.sig_o) / d.sig_o.replace(0, np.nan) * 100.0
    d['sig_range_pct'] = (d.sig_h - d.sig_l) / d.sig_c.replace(0, np.nan) * 100.0
    d['sig_dollar'] = d.sig_v * d.sig_c
    d['spread_over_r'] = (d.spread_cc_bps / 100.0) / d.next_r_pct.clip(lower=0.05)
    d['log_adv20'] = np.log10(d.adv20.clip(lower=1))
    d['log_cum_dv'] = np.log10(d.cum_dollar_vol.clip(lower=1))
    d['gap_up'] = (d.gap_pct > 0).astype(int)
    # the day's own F6 signal count STRICTLY BEFORE this signal (causal, live-computable)
    d = d.sort_values(['day', 'sig_m', 'symbol']).reset_index(drop=True)
    d['sig_seq'] = d.groupby('day').cumcount()
    # --- news (union of the two pulls)
    d = d.merge(_news(), on=['day', 'symbol'], how='left')
    if 'has_news_x' in d.columns:                       # Q already carries has_news
        d['has_news'] = d.has_news_y.fillna(d.has_news_x)
        d = d.drop(columns=['has_news_x', 'has_news_y'])
    d['news_known'] = d.news_known.fillna(0).astype(int)
    # --- premarket dollars: the file's own column, and D/pm_bars.db, kept SEPARATE (availability!)
    d = d.merge(_pmdb(), on=['day', 'symbol'], how='left')
    d['pm_any'] = d.pm_dollar_vol.where(d.pm_dollar_vol.notna(), d.pm_db)
    d['pm_known'] = d.pm_any.notna().astype(int)
    # --- day context and index state at the signal minute
    d = d.merge(_dayfeat(), on='day', how='left')
    e = _etf_state()
    d = d.merge(e.rename(columns={'m': 'sig_m'}), on=['day', 'sig_m'], how='left')
    return d


# --------------------------------------------------------------------------- book + statistics
def weeks_of(pop):
    w = pd.read_csv(f'{H}/weeks_{pop}.csv', **RD)
    return {s: sorted(w[w.split == s].wk.tolist()) for s in ('TRAIN', 'VAL', 'TEST')}


def scoreable(d, variant='hold', floor=True):
    rr, why, xm, rp = VARIANTS[variant]
    m = d[rp].notna() & (d[rp] >= 1.0) & d[rr].notna() & d[xm].notna()
    if floor:
        m &= d.range_so_far_pct >= 5
    return d[m].copy()


def book(d, split, weeks, variant='hold'):
    """run_book(12,4) on one split of an already-scoreable frame. Returns the booked trades."""
    rr, why, xm, rp = VARIANTS[variant]
    y = d[d.split == split]
    if len(y) < 40:
        return None
    rows = [(r.day, int(r.next_entry_m), int(getattr(r, xm)), r.symbol,
             float(getattr(r, f'net_{variant}')), r.Index) for r in y.itertuples()]
    bk = run_book(rows, 12, 4)
    if len(bk) < 20:
        return None
    idx = [r[5] for r in bk]
    return d.loc[idx].copy()


def stats(t, split, weeks, variant='hold'):
    """The gate's statistics on a booked frame."""
    if t is None or not len(t):
        return None
    why = VARIANTS[variant][1]
    v = t[f'net_{variant}'].values
    n = len(v)
    sd = v.std(ddof=1)
    w = t.groupby('wk')[f'net_{variant}'].sum().reindex(weeks[split]).fillna(0.0)
    s = np.sort(v)
    mix = t[why].value_counts(normalize=True)
    return dict(n=n, tpw=round(n / max(len(weeks[split]), 1), 1),
                meanR=round(float(v.mean()), 4), grossR=round(float(t[f'gross_{variant}'].mean()), 4),
                se=round(float(sd / np.sqrt(n)), 4),
                t=round(float(v.mean() / (sd / np.sqrt(n))), 2) if sd > 0 else 0.0,
                WR=round(float((v > 0).mean() * 100), 1),
                stopP=round(float(mix.get('stop', 0.0) * 100), 1),
                wkR=round(float(w.mean()), 2), wkSE=round(float(w.std(ddof=1) / np.sqrt(len(w))), 2),
                green=round(float((w > 0).mean()), 2), worstWk=round(float(w.min()), 1),
                mdd=round(float(_mdd(t, variant)), 1),
                ex1=round(float(s[:max(n - max(int(n * 0.01), 1), 1)].mean()), 4),
                ex5=round(float(s[:max(n - max(int(n * 0.05), 1), 1)].mean()), 4),
                cap3=round(float(np.minimum(v, 3.0).mean()), 4))


def _mdd(t, variant):
    """Max drawdown in R on the day-ordered equity curve of the booked trades."""
    e = t.groupby('day')[f'net_{variant}'].sum().sort_index().cumsum()
    return float((e - e.cummax()).min()) if len(e) else 0.0


def book_stats(d, split, weeks, variant='hold'):
    t = book(d, split, weeks, variant)
    return stats(t, split, weeks, variant), t
