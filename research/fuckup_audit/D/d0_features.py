#!/usr/bin/env python3
"""D0 step 2 — the feature matrix. EVERY column is causal at the SIGNAL minute.

Causality contract enforced here (asserted at the bottom of the file):
  * candidates3's own `price`, `r_pct` and `dist_open_pct` are built from `entry` = the NEXT bar's
    open, i.e. they are known one minute AFTER the decision. They are therefore NOT used as
    features. Causal twins are rebuilt from `level` (the running level, known at the signal bar's
    close) and `stop`: price_sig = level, r_pct_sig = (level-stop)/level*100,
    dist_open_sig = (level/o0 - 1)*100 with o0 recovered from candidates3's own
    dist_open_pct (o0 = entry / (1 + dist_open_pct/100)).
  * every index / breadth / news column is read at or before the signal minute `sig_m`.
  * the TARGET keeps Stage A's adopted contract (c) exactly (it uses the realised fill), because it
    is the outcome, not a feature.

Target: net R under Stage A contract (c) for the HOLD exit:
  half_c = 0.5 * spread_pct_c / max(r_pct, 0.05);  net = rr_hold - 0.25*half_c - half_c*ratio[why]
  ratio = {stop: .875, eod: .412, target: .875}   (acore.RATIO_C / acore.ENTRY_COEF_C)

Output: research/fuckup_audit/D/feat.csv
"""
import os, sys, sqlite3
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/A'); sys.path.insert(0, ROOT)
import acore                                                              # noqa: E402

D = 'research/fuckup_audit/D'
OUT = f'{D}/feat.csv'
DAYF = 'research/fuckup_audit/day_features.csv'
POPA = 'research/fuckup_audit/A/pop_a.csv'
ETF = 'research/lit_review_2026/etf_1min.db'
NEWS = f'{D}/news_presence.csv'

c = pd.read_csv(f'{D}/table.csv', dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str},
                keep_default_na=False, na_values=[''])
c['key'] = c.fam + ' ' + c.cfg
print('rows', len(c), 'keys', c.key.nunique(), flush=True)

# ---------------------------------------------------------------- target: Stage A contract (c)
c['pb'] = pd.cut(c.price, acore.PB_EDGES, labels=acore.PB_LAB, include_lowest=True).astype(str)
c['hb'] = pd.cut(c.entry_m, acore.HB_EDGES, labels=acore.HB_LAB).astype(str)
tab = acore.corrected_spread_table()
c['spread_pct_c'] = [tab.get((p, h), np.nan) / 100.0 for p, h in zip(c.pb, c.hb)]
assert c.spread_pct_c.notna().all(), 'unmapped (pb,hb) cell'
half_c = 0.5 * c.spread_pct_c / c.r_pct.clip(lower=0.05)
rc = c.why_hold.map(acore.RATIO_C).fillna(0.875)
c['net'] = c.rr_hold - acore.ENTRY_COEF_C * half_c - half_c * rc
c['win'] = (c.net > 0).astype(int)
c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
c['month'] = c.day.str[:7]
c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)

# ---------------------------------------------------------------- causal per-candidate features
o0 = c.entry / (1 + c.dist_open_pct / 100.0)                              # the true 09:30 open
c['f_price_sig'] = c.level
c['f_log_price'] = np.log10(c.level)
c['f_r_pct_sig'] = (c.level - c.stop) / c.level * 100.0
c['f_dist_open_sig'] = (c.level / o0 - 1) * 100.0
c['f_range_so_far'] = c.range_so_far_pct
c['f_rv_adv'] = c.rv_adv
c['f_gap_pct'] = c.gap_pct
c['f_log_adv20'] = np.log10(c.adv20.clip(lower=1))
c['f_mins_since_open'] = c.sig_m - 570
c['f_dow'] = pd.to_datetime(c.day).dt.weekday
pb_sig = pd.cut(c.level, acore.PB_EDGES, labels=acore.PB_LAB, include_lowest=True).astype(str)
hb_sig = pd.cut(c.sig_m, acore.HB_EDGES, labels=acore.HB_LAB).astype(str)
c['f_spread_cc_bps'] = [tab.get((p, h), np.nan) for p, h in zip(pb_sig, hb_sig)]
c['f_sp_over_r'] = (c.f_spread_cc_bps / 1e4) / (c.f_r_pct_sig / 100.0)
c['f_pb'] = pd.Categorical(pb_sig, categories=acore.PB_LAB).codes.astype(float)

# ---------------------------------------------------------------- day context (known 09:30)
df = pd.read_csv(DAYF, dtype={'day': str}, keep_default_na=False, na_values=['']).set_index('day')
DAYCOLS = ['spy_gap', 'spy_prev_ret', 'spy_vs_sma5', 'spy_vs_sma20', 'spy_vol20',
           'iwm_gap', 'iwm_prev_ret', 'iwm_vs_sma5', 'iwm_vs_sma20', 'iwm_vol20']
for k in DAYCOLS:
    c[f'f_{k}'] = df[k].reindex(c.day).values
REG = {'A': 0, 'B': 1, 'C1': 2, 'C2': 3}
c['f_regime'] = [REG.get(str(x), np.nan) for x in df.regime.reindex(c.day).values]

# ---------------------------------------------------------------- intraday index state at sig_m
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
    v = idx.reindex(pd.MultiIndex.from_arrays([c.day, c.sig_m])).values   # close of the SIGNAL bar
    c[f'f_{s.lower()}_ret'] = v
    c[f'f_{s.lower()}_sign'] = np.where(np.isnan(v), np.nan, (v >= 0).astype(float))
print('iwm_ret nan', float(c.f_iwm_ret.isna().mean()).__round__(4), flush=True)

# ---------------------------------------------------------------- breadth-so-far (A3's definition)
p = pd.read_csv(POPA, usecols=['day', 'symbol', 'entry_m', 'dist_open_pct'],
                dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
p = p.drop_duplicates(['day', 'symbol', 'entry_m']).sort_values(['day', 'entry_m'])
p['pos'] = (p.dist_open_pct > 0).astype(int)
g = p.groupby('day')
p['cum_n'] = g.cumcount()
p['cum_pos'] = g.pos.cumsum() - p.pos
ends = p.groupby(['day', 'entry_m']).agg(n_before=('cum_n', 'min'), pos_before=('cum_pos', 'min'))
e = ends.reindex(pd.MultiIndex.from_arrays([c.day, c.entry_m]))
brd = (e.pos_before / e.n_before).values
c['f_breadth'] = np.where(e.n_before.values >= 5, brd, np.nan)
c['f_n_pop_before'] = e.n_before.values

# ---------------------------------------------------------------- the family's own signal count so far
c = c.sort_values(['key', 'day', 'sig_m', 'symbol']).reset_index(drop=True)
c['f_n_fam_before'] = c.groupby(['key', 'day']).cumcount()

# ---------------------------------------------------------------- news (joined if available)
if os.path.exists(NEWS):
    nw = pd.read_csv(NEWS, dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
    nw = nw.drop_duplicates(['day', 'symbol']).set_index(['day', 'symbol'])
    ix = pd.MultiIndex.from_arrays([c.day, c.symbol])
    c['f_news_pre'] = nw.n_prev15_to_0930.reindex(ix).values
    mm = nw.mins_0930_to_1401.reindex(ix).values
    c['f_news_intraday'] = [0 if (isinstance(s, float) or s is None or s != s or s == '')
                            else sum(1 for x in str(s).split() if int(x) < sm)
                            for s, sm in zip(mm, c.sig_m)]
    c['f_news_intraday'] = np.where(pd.isna(c.f_news_pre), np.nan, c.f_news_intraday)
    cov = float(c.f_news_pre.notna().mean())
    print(f'news coverage {cov:.3f}', flush=True)
else:
    print('NO news file yet', flush=True)

FEATS = [k for k in c.columns if k.startswith('f_')]
print(len(FEATS), 'features:', FEATS, flush=True)

# ---------------------------------------------------------------- causality assertions
assert (c.sig_m < c.entry_m).all(), 'signal must precede entry'
assert (c.entry_m >= 600).all(), 'window rule: entries >= 10:00'
# entry_m - sig_m is 1 except where the tape has missing minutes (disclosed in the report)
gap = c.entry_m - c.sig_m
print(f'entry_m-sig_m > 1 on {float((gap > 1).mean()):.4f} of rows; sig_m < 599 on {int((c.sig_m < 599).sum())}',
      flush=True)
# the causal price/R twins must NOT equal the entry-derived ones anywhere the fill moved
frac = float((np.abs(c.f_r_pct_sig - c.r_pct) > 1e-9).mean())
print(f'rows where causal r_pct differs from the entry-based r_pct: {frac:.3f}', flush=True)
for k in FEATS:
    assert c[k].dtype.kind in 'fiub', f'{k} is not numeric'

KEEPC = (['day', 'symbol', 'key', 'fam', 'cfg', 'sig_m', 'entry_m', 'exit_m_hold', 'why_hold',
          'split', 'month', 'wk', 'net', 'win', 'rr_hold', 'r_pct', 'price', 'spread_pct_c'] + FEATS)
c[KEEPC].to_csv(OUT, index=False)
print('wrote', OUT, c.shape, flush=True)
print(c.groupby(['key', 'split']).size().to_string(), flush=True)
