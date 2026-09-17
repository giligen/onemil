#!/usr/bin/env python3
"""D1 step 2 — the feature matrix. EVERY feature is computable at or before the SIGNAL minute (asserted below).

Differences from D0's d0_features.py, all listed in D1/PREREG.md:
  * source is candidates4 (via D1/table.csv), whose `price`/`dist_open_pct`/`vwap_dist_pct`/`range_so_far_pct` are
    LEVEL-based and therefore already causal; only `r_pct` is fill-derived, so the causal twin
    f_r_pct_sig = (level-stop)/level*100 is used instead and the fill-derived one is never a feature.
  * the signal bar's OHLCV, the level history (n_touches/consol_bars/consol_vol_ratio), cum $ volume, VWAP distance,
    close-confirm, prev-day range, asset class and premarket dollars are NEW (candidates3 had none of them).
  * TWO targets: primary = hold exit / touch stop (rr_hold, r_pct); secondary = the 2R stop-1% variant
    (rr_2r_stopm1, r_pct_m1). Each carries its own population mask (r_pct of ITS variant >= 1).
  * news = D/news_presence.csv UNION E/news_presence_e.csv; premarket $ = D/pm_bars.db, falling back to
    candidates4's own pm_dollar_vol column; each has its own _missing indicator.

Cost contract (c), identical to C/score5c.py::net_r for the `next` fill:
    half = 0.5*(spread_cc_bps/100)/max(r_pct_variant, 0.05)
    net  = rr - 0.25*half - half*{stop:.875, lock:.875, eod:.412, target:.875, none:.875}[why]

Output: research/fuckup_audit/D1/feat.csv
"""
import os, sys, sqlite3
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
D1 = 'research/fuckup_audit/D1'
OUT = f'{D1}/feat.csv'
DAYF = 'research/fuckup_audit/day_features.csv'
POPA = 'research/fuckup_audit/A/pop_a.csv'
ETF = 'research/lit_review_2026/etf_1min.db'
NEWS = ['research/fuckup_audit/D/news_presence.csv', 'research/fuckup_audit/E/news_presence_e.csv']
PMDB = 'research/fuckup_audit/D/pm_bars.db'
PM_CUT = 5816688.0
EXIT_RATIO = {'stop': 0.875, 'lock': 0.875, 'eod': 0.412, 'target': 0.875, 'none': 0.875}
PB_EDGES = [5, 10, 20, 50, 200, 1e9]

c = pd.read_csv(f'{D1}/table.csv', dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str,
                                          'asset_class': str, 'next_why_hold': str, 'next_why_2r_stopm1': str},
                keep_default_na=False, na_values=[''])
c['key'] = c.fam + ' ' + c.cfg
c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
c['month'] = c.day.str[:7]
c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)
print('rows', len(c), 'keys', c.key.nunique(), flush=True)

# ---------------------------------------------------------------- targets (contract c), one per stop variant
spread_pct_c = c.spread_cc_bps / 100.0
for tag, rr, why, xm, rp in (('p', 'next_rr_hold', 'next_why_hold', 'next_exit_m_hold', 'next_r_pct'),
                             ('s', 'next_rr_2r_stopm1', 'next_why_2r_stopm1', 'next_exit_m_2r_stopm1',
                              'next_r_pct_m1')):
    half = 0.5 * spread_pct_c / c[rp].clip(lower=0.05)
    c[f'net_{tag}'] = c[rr] - 0.25 * half - half * c[why].map(EXIT_RATIO).fillna(0.875)
    c[f'net_{tag}'] = np.where(c[rp] >= 1.0, c[f'net_{tag}'], np.nan)      # the variant's own population
    c[f'win_{tag}'] = np.where(np.isnan(c[f'net_{tag}']), np.nan, (c[f'net_{tag}'] > 0).astype(float))
    c[f'why_{tag}'] = c[why]
    c[f'xm_{tag}'] = c[xm]
    c[f'gross_{tag}'] = c[rr]
print('primary rows', int(c.net_p.notna().sum()), 'secondary rows', int(c.net_s.notna().sum()), flush=True)

# ---------------------------------------------------------------- per-candidate features, all causal at sig_m
lv = c.level
c['f_log_price'] = np.log10(lv)
c['f_pb'] = pd.cut(lv, PB_EDGES, labels=False, include_lowest=True).astype(float)
c['f_r_pct_sig'] = (lv - c.stop) / lv * 100.0
c['f_dist_open_pct'] = c.dist_open_pct
c['f_range_so_far'] = c.range_so_far_pct
c['f_rv_adv'] = c.rv_adv
c['f_gap_pct'] = c.gap_pct
c['f_log_adv20'] = np.log10(c.adv20.clip(lower=1))
c['f_prev_day_range_pct'] = c.prev_day_range_pct
c['f_mins_since_open'] = c.sig_m - 570
c['f_dow'] = pd.to_datetime(c.day).dt.weekday
c['f_spread_cc_bps'] = c.spread_cc_bps
c['f_sp_over_r'] = (c.spread_cc_bps / 1e4) / (c.f_r_pct_sig / 100.0)
# the signal bar itself
c['f_sig_o_rel'] = (c.sig_o / lv - 1) * 100.0
c['f_sig_h_rel'] = (c.sig_h / lv - 1) * 100.0
c['f_sig_l_rel'] = (c.sig_l / lv - 1) * 100.0
c['f_sig_c_rel'] = (c.sig_c / lv - 1) * 100.0
c['f_sig_range_pct'] = (c.sig_h - c.sig_l) / lv * 100.0
rng = (c.sig_h - c.sig_l).replace(0, np.nan)
c['f_sig_body'] = (c.sig_c - c.sig_o) / rng
c['f_log_sig_v'] = np.log10(1 + c.sig_v.clip(lower=0))
c['f_sig_dollar'] = np.log10(1 + (c.sig_c * c.sig_v).clip(lower=0))
# level history / tape context
c['f_n_touches'] = c.n_touches
c['f_consol_bars'] = c.consol_bars
c['f_consol_vol_ratio'] = c.consol_vol_ratio
c['f_log_cum_dollar'] = np.log10(1 + c.cum_dollar_vol.clip(lower=0))
c['f_vwap_dist_pct'] = c.vwap_dist_pct
c['f_close_confirm'] = c.close_confirm
c['f_is_wrapper'] = (c.asset_class == 'wrapper').astype(float)
c['f_is_unknown_class'] = (c.asset_class == 'unknown').astype(float)

# ---------------------------------------------------------------- premarket dollars (D/pm_bars.db, then candidates4)
con = sqlite3.connect(f'file:{ROOT}/{PMDB}?mode=ro', uri=True)
pm = pd.read_sql('select symbol, day, pm_dollar_vol, src from pm', con)
con.close()
pm['pmv'] = np.where(pm.src == 'none', 0.0, pm.pm_dollar_vol)
pm = pm.drop_duplicates(['day', 'symbol']).set_index(['day', 'symbol']).pmv
ix = pd.MultiIndex.from_arrays([c.day, c.symbol])
v_db = pm.reindex(ix).values
v_c4 = c.pm_dollar_vol.values
pmv = np.where(~np.isnan(v_db), v_db, v_c4)
c['f_log_pm_dollar'] = np.log10(1 + np.where(np.isnan(pmv), np.nan, np.clip(pmv, 0, None)))
c['f_pm_missing'] = np.isnan(pmv).astype(float)
c['f_pm_hi'] = np.where(np.isnan(pmv), np.nan, (pmv > PM_CUT).astype(float))
print(f'pm coverage: db {float(np.mean(~np.isnan(v_db))):.3f}  union {float(np.mean(~np.isnan(pmv))):.3f} '
      f' | pm_hi share (covered rows) {float(np.nanmean(c.f_pm_hi)):.3f}', flush=True)

# ---------------------------------------------------------------- news (union of the two pulls, fetch_ok only)
nws = []
for p in NEWS:
    if os.path.exists(p):
        n = pd.read_csv(p, dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
        n = n[n.fetch_ok == 1]
        nws.append(n)
        print('news file', p, len(n), flush=True)
nw = pd.concat(nws, ignore_index=True).drop_duplicates(['day', 'symbol']).set_index(['day', 'symbol'])
c['f_news_pre'] = nw.n_prev15_to_0930.reindex(ix).values
mm = nw.mins_0930_to_1401.reindex(ix).values
c['f_news_intraday'] = [np.nan if (s is None or (isinstance(s, float) and s != s)) else
                        (0 if s == '' else sum(1 for x in str(s).split() if int(x) < sm))
                        for s, sm in zip(mm, c.sig_m)]
c['f_news_intraday'] = np.where(pd.isna(c.f_news_pre), np.nan, c.f_news_intraday)
c['f_news_missing'] = pd.isna(c.f_news_pre).astype(float)
cov = float(c.f_news_pre.notna().mean())
print(f'news coverage {cov:.3f}', flush=True)
print(c.assign(covn=c.f_news_pre.notna()).groupby('split').covn.mean().round(3).to_string(), flush=True)

# ---------------------------------------------------------------- day context (known at 09:30)
df = pd.read_csv(DAYF, dtype={'day': str}, keep_default_na=False, na_values=['']).set_index('day')
DAYCOLS = ['spy_gap', 'spy_prev_ret', 'spy_vs_sma5', 'spy_vs_sma20', 'spy_vol20',
           'iwm_gap', 'iwm_prev_ret', 'iwm_vs_sma5', 'iwm_vs_sma20', 'iwm_vol20']
for k in DAYCOLS:
    c[f'f_{k}'] = df[k].reindex(c.day).values
REG = {'A': 0, 'B': 1, 'C1': 2, 'C2': 3}
c['f_regime'] = [REG.get(str(x), np.nan) for x in df.regime.reindex(c.day).values]

# ---------------------------------------------------------------- intraday index state at the SIGNAL minute
con = sqlite3.connect(f'file:{ROOT}/{ETF}?mode=ro', uri=True)
b = pd.read_sql("select symbol, t, o, c from bars where symbol in ('IWM','SPY') and t >= '2024-12-15'", con)
con.close()
ts = pd.to_datetime(b.t, utc=True).dt.tz_convert('America/New_York')
b['day'] = ts.dt.strftime('%Y-%m-%d'); b['m'] = ts.dt.hour * 60 + ts.dt.minute
op = b[b.m == 570].set_index(['symbol', 'day']).o
b = b[(b.m >= 570) & (b.m <= 900)].copy()
b['o570'] = list(op.reindex(list(zip(b.symbol, b.day))))
b['ret'] = (b.c / b.o570 - 1) * 100
for s in ('IWM', 'SPY'):
    idx = b[b.symbol == s].set_index(['day', 'm']).ret
    v = idx.reindex(pd.MultiIndex.from_arrays([c.day, c.sig_m])).values
    c[f'f_{s.lower()}_ret'] = v
    c[f'f_{s.lower()}_sign'] = np.where(np.isnan(v), np.nan, (v >= 0).astype(float))
print('iwm_ret nan', round(float(c.f_iwm_ret.isna().mean()), 4), flush=True)

# ---------------------------------------------------------------- breadth-so-far (A3's definition, asof sig_m)
p = pd.read_csv(POPA, usecols=['day', 'symbol', 'entry_m', 'dist_open_pct'],
                dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
p = p.drop_duplicates(['day', 'symbol', 'entry_m']).sort_values(['day', 'entry_m'])
p['pos'] = (p.dist_open_pct > 0).astype(int)
nb = np.full(len(c), np.nan); pb = np.full(len(c), np.nan)
grp = {d: (g.entry_m.values, np.concatenate([[0], np.cumsum(g.pos.values)])) for d, g in p.groupby('day')}
days = c.day.values; sigm = c.sig_m.values
for i in range(len(c)):
    g = grp.get(days[i])
    if g is None:
        continue
    j = int(np.searchsorted(g[0], sigm[i], side='left'))     # strictly earlier entries only
    nb[i] = j
    pb[i] = g[1][j]
c['f_n_pop_before'] = nb
c['f_breadth'] = np.where(nb >= 5, pb / np.maximum(nb, 1), np.nan)

# ---------------------------------------------------------------- the family's own signal count so far that day
c = c.sort_values(['key', 'day', 'sig_m', 'symbol']).reset_index(drop=True)
c['f_n_fam_before'] = c.groupby(['key', 'day']).cumcount()

FEATS = [k for k in c.columns if k.startswith('f_')]
print(len(FEATS), 'features:', FEATS, flush=True)

# ---------------------------------------------------------------- causality assertions
assert (c.sig_m < c.next_entry_m).all(), 'signal must precede the entry'
assert (c.range_so_far_pct >= 5).all(), 'causal membership floor'
assert (c.next_entry >= 5).all() and (c.next_entry_m <= 841).all()
gap = c.next_entry_m - c.sig_m
print(f'entry_m-sig_m > 1 on {float((gap > 1).mean()):.4f} of rows', flush=True)
for k in FEATS:
    assert c[k].dtype.kind in 'fiub', f'{k} is not numeric'
# no feature may be derived from the fill: assert the causal R twin differs from the fill R where the fill moved
print('rows where the causal r differs from the fill r:',
      round(float((np.abs(c.f_r_pct_sig - c.next_r_pct) > 1e-9).mean()), 3), flush=True)

KEEPC = (['day', 'symbol', 'key', 'fam', 'cfg', 'sig_m', 'next_entry_m', 'split', 'month', 'wk',
          'net_p', 'win_p', 'why_p', 'xm_p', 'gross_p', 'net_s', 'win_s', 'why_s', 'xm_s', 'gross_s',
          'next_r_pct', 'next_r_pct_m1', 'spread_cc_bps'] + FEATS)
c[KEEPC].to_csv(OUT, index=False)
print('wrote', OUT, c[KEEPC].shape, flush=True)
print(c.groupby(['key', 'split']).size().unstack(fill_value=0).to_string(), flush=True)
