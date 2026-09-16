#!/usr/bin/env python3
"""Daily-bar hypothesis tests on the point-in-time panel (research/lit_review_2026/daily_panel.parquet).
Every signal uses day-t information only; the position is taken at day t's CLOSE (MOC) or at day t+1's OPEN and closed at the
next open / next close as the hypothesis says. Costs: round trip charged per trade by liquidity band. Splits fixed:
TRAIN 2025 / VAL Jan–May 2026 / TEST Jun–Sep 2026. Book: top-N names per day by the signal (N=4, the live concurrency),
equal risk. Reports mean bps per trade, t-stat, hit rate, weekly sum in R-equivalents (R = 1 trade's average |ret|),
and weeks green. Nothing is tuned: every threshold comes from the paper's spec listed in D_short_horizon_daily.md."""
import os, sys, numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
d = pd.read_parquet('research/lit_review_2026/daily_panel.parquet')
d = d[(d.close >= 5) & (d.dvol20 >= 2e6) & d.adv20.notna()].copy()
d = d[~d.symbol.astype(str).str.match(r'^Z[VWX]ZZ|^ZZ')]                                   # Nasdaq TEST symbols (ZVZZT etc.) are in the daily feed
for k in ('ret_on', 'ret_id', 'ret_cc', 'ret_on_next', 'ret_id_next', 'ret_cc_next'):        # a one-day move beyond ±50% on a $2M+/day name is a bad print or a corporate action: drop the row
    d = d[(d[k].isna()) | (d[k].abs() <= 0.5)]
d['split'] = np.where(d.bar_date < '2026-01-01', 'TRAIN', np.where(d.bar_date < '2026-06-01', 'VAL', 'TEST'))
d['wk'] = pd.to_datetime(d.bar_date).dt.to_period('W-FRI').astype(str)
def cost_bps(dv):   # round-trip cost by dollar-volume band (spread + impact, conservative)
    return np.where(dv >= 5e7, 6, np.where(dv >= 1e7, 12, np.where(dv >= 5e6, 25, 40)))
d['cost'] = cost_bps(d.dvol20.values) / 1e4
N = int(os.environ.get('TOPN', '4'))

def book(sig_rows, ret_col, name, lower_better=False):
    """sig_rows: the day-t rows that qualify with a 'score'; take top-N per day; realized = ret_col − cost."""
    out = []
    for split in ('TRAIN', 'VAL', 'TEST'):
        x = sig_rows[sig_rows.split == split].copy()
        if not len(x): out.append(dict(hyp=name, split=split, n=0)); continue
        x = x.sort_values(['bar_date', 'score'], ascending=[True, lower_better]).groupby('bar_date').head(N)
        x['net'] = x[ret_col] - x.cost
        w = x.groupby('wk').net.sum(); nw = d[d.split == split].wk.nunique()
        out.append(dict(hyp=name, split=split, n=len(x), per_day=round(len(x) / d[d.split == split].bar_date.nunique(), 1),
                        gross_bps=round(x[ret_col].mean() * 1e4, 1), median_bps=round(x[ret_col].median() * 1e4, 1), net_bps=round(x.net.mean() * 1e4, 1), t=round(x.net.mean() / (x.net.std() / np.sqrt(len(x))), 2),
                        hit=round((x.net > 0).mean() * 100, 1), wk_net_pct=round(w.sum() / nw * 100, 2), wk_green=f'{int((w > 0).sum())}/{nw}', worst_wk_pct=round(w.min() * 100, 2)))
    return out

res = []
# D1 volume-shock overnight: buy MOC the top-decile volume/ADV20 (vol_ratio) names, sell next open (ret_on_next)
s = d[d.vol_ratio.notna() & d.ret_on_next.notna()].copy(); s['score'] = s.vol_ratio
thr = s[s.split == 'TRAIN'].vol_ratio.quantile(0.9)                     # decile cut fixed on TRAIN
res += book(s[s.vol_ratio >= thr], 'ret_on_next', f'D1 volume-shock overnight (vol/ADV >= {thr:.1f})')
res += book(s[s.vol_ratio >= 3.0], 'ret_on_next', 'D1b volume >= 3x ADV overnight')
# D2 large non-news loser reversal: open→close <= −8% on >= 2x volume; buy next open, sell next close (ret_id_next); no news filter available → all
s = d[(d.ret_id <= -0.08) & (d.vol_ratio >= 2) & d.ret_id_next.notna()].copy(); s['score'] = s.ret_id
res += book(s, 'ret_id_next', 'D2 loser >= -8% id, 2x vol: next open→close', lower_better=True)
res += book(s.assign(score=s.ret_id), 'ret_cc_next', 'D2b same, next close→close', lower_better=True)
# D12 intraday-component 1-day reversal: bottom decile ret_id (residual vs market not available → raw), buy next open sell next close
s = d[d.ret_id.notna() & d.ret_id_next.notna()].copy(); s['score'] = s.ret_id
thr2 = s[s.split == 'TRAIN'].ret_id.quantile(0.1)
res += book(s[s.ret_id <= thr2], 'ret_id_next', f'D12 bottom-decile intraday ret (<= {thr2*100:.1f}%), next open→close', lower_better=True)
# D12-overnight control: bottom-decile OVERNIGHT return should NOT reverse (Barardehi et al.)
s = d[d.ret_on.notna() & d.ret_id_next.notna()].copy(); s['score'] = s.ret_on
thr3 = s[s.split == 'TRAIN'].ret_on.quantile(0.1)
res += book(s[s.ret_on <= thr3], 'ret_id_next', f'D12c control: bottom-decile OVERNIGHT ret, next open→close', lower_better=True)
# Overnight premium in the cross-section (Lou-Polk-Skouras class): hold overnight the top-decile past-overnight-return names (ret_on 20-day mean) — needs rolling; proxy with ret_on itself
s = d[d.ret_on.notna() & d.ret_on_next.notna()].copy(); s['score'] = s.ret_on
res += book(s[s.ret_on >= s[s.split == 'TRAIN'].ret_on.quantile(0.9)], 'ret_on_next', 'C-proxy: top-decile overnight ret today → hold next overnight')
# Continuation after a large up day on volume (Pritamani-Singal): close→close >= +8% on >= 2x vol; buy next open, sell next close
s = d[(d.ret_cc >= 0.08) & (d.vol_ratio >= 2) & d.ret_id_next.notna()].copy(); s['score'] = s.vol_ratio
res += book(s, 'ret_id_next', 'PS large up day 8%/2x vol: next open→close'); res += book(s, 'ret_on_next', 'PS large up day: next overnight')
# 52-week high proximity (George-Hwang, daily): close within 1% of 52w high, buy next open → next close
s = d[d.high52.notna() & (d.close >= d.high52 * 0.99) & d.ret_id_next.notna()].copy(); s['score'] = s.vol_ratio
res += book(s, 'ret_id_next', '52w-high breakout day: next open→close'); res += book(s, 'ret_on_next', '52w-high: next overnight')
R = pd.DataFrame(res); pd.set_option('display.width', 250)
print(R.to_string(index=False)); R.to_csv('research/lit_review_2026/daily_hyps_results.csv', index=False)
