#!/usr/bin/env python3
"""RUNBOOK rows 9, 14 and the row-6 refinement — all on the daily point-in-time panel.
 M16  the single-stock gap table (measurement): by gap size x dollar-volume band, P(close>open), mean/median open->close,
      P(same-day full fill of the gap), P(half fill). Nothing is selected; both years shown.
 M29  cross-sectional overnight continuation: rank on the trailing-20-day mean overnight return (known at day t's close),
      hold the top decile / top-4 close->open. Large-cap proxy = 20-day dollar volume >= $50M (6 bps round trip).
 M36r large-loser reversal SPLIT BY MARKET VOLATILITY REGIME (trailing 20-day SPY close-to-close vol tercile, causal).
Costs: RUNBOOK section 0 bands. Splits fixed. Outputs daily_queue.md."""
import os, numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
P = 'research/lit_review_2026/daily_panel.parquet'
d = pd.read_parquet(P, columns=['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'prev_close', 'volume', 'adv20', 'dvol20',
                                'ret_on', 'ret_id', 'ret_cc', 'ret_on_next', 'ret_id_next', 'vol_ratio'])
import re
bad = [c for c in d.symbol.cat.categories if re.match(r'^Z[VWX]ZZ|^ZZ', str(c))] if hasattr(d.symbol, 'cat') else []
if bad: d = d[~d.symbol.isin(bad)]
d = d[(d.close >= 5) & (d.prev_close > 0) & d.dvol20.notna()]
for k in ('ret_on', 'ret_id', 'ret_cc', 'ret_on_next', 'ret_id_next'):
    d = d[d[k].isna() | (d[k].abs() <= 0.5)]
d['split'] = np.where(d.bar_date < '2026-01-01', 'TRAIN', np.where(d.bar_date < '2026-06-01', 'VAL', 'TEST'))
d['wk'] = pd.to_datetime(d.bar_date).dt.to_period('W-FRI').astype(str)
d['cost'] = np.where(d.dvol20 >= 5e7, 6, np.where(d.dvol20 >= 1e7, 12, np.where(d.dvol20 >= 5e6, 25, 40))) / 1e4
L = []

# ---------------- M16 gap table ----------------
g = d[(d.dvol20 >= 2e6) & d.ret_on.notna() & d.ret_id.notna()].copy()
g['gapb'] = pd.cut(g.ret_on, [-1, -0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.20, 1],
                   labels=['<-20%', '-20..-10', '-10..-5', '-5..-2', '-2..+2', '+2..+5', '+5..+10', '+10..+20', '>+20%'])
g['dvb'] = pd.cut(g.dvol20, [2e6, 1e7, 5e7, 1e12], labels=['$2-10M', '$10-50M', '>$50M'])
up = g[g.ret_on > 0]; dn = g[g.ret_on < 0]
g['fill_full'] = np.where(g.ret_on > 0, g.low <= g.prev_close, g.high >= g.prev_close)
g['fill_half'] = np.where(g.ret_on > 0, g.low <= g.open - 0.5 * (g.open - g.prev_close), g.high >= g.open + 0.5 * (g.prev_close - g.open))
L += ['# RUNBOOK row 9 — M16 single-stock gap table (measurement, $5+, $2M+/day, 2025-01→2026-09)', '',
      'Columns: n, P(close>open), mean and median open→close in bps, P(gap fully filled that day), P(half filled). Split TRAIN=2025 / VAL+TEST=2026.']
for yr, mask in (('2025', g.split == 'TRAIN'), ('2026', g.split != 'TRAIN')):
    t = g[mask].groupby(['gapb', 'dvb'], observed=True).agg(n=('ret_id', 'size'), p_up=('ret_id', lambda s: round((s > 0).mean() * 100, 1)),
                                                           mean_bps=('ret_id', lambda s: round(s.mean() * 1e4)), med_bps=('ret_id', lambda s: round(s.median() * 1e4)),
                                                           fill=('fill_full', lambda s: round(s.mean() * 100, 1)), half=('fill_half', lambda s: round(s.mean() * 100, 1)))
    L += ['', f'## gap table {yr}', t.to_string()]
# the flag the runbook asks for: cells with mean open->close >= +50 bps and n >= 200 in BOTH years
a = g[g.split == 'TRAIN'].groupby(['gapb', 'dvb'], observed=True).ret_id.agg(['mean', 'size'])
b = g[g.split != 'TRAIN'].groupby(['gapb', 'dvb'], observed=True).ret_id.agg(['mean', 'size'])
j = a.join(b, lsuffix='_25', rsuffix='_26')
hit = j[(j['mean_25'] >= 0.005) & (j['mean_26'] >= 0.005) & (j['size_25'] >= 200) & (j['size_26'] >= 200)]
L += ['', f'cells with mean open→close >= +50 bps and n >= 200 in BOTH years: {len(hit)}', hit.to_string() if len(hit) else '(none)']

# ---------------- M29 cross-sectional overnight continuation ----------------
lc = d[(d.dvol20 >= 5e7) & d.ret_on.notna()].sort_values(['symbol', 'bar_date']).copy()
lc['on20'] = lc.groupby('symbol', observed=True).ret_on.transform(lambda s: s.rolling(20, min_periods=15).mean())   # known at t's close
lc = lc[lc.on20.notna() & lc.ret_on_next.notna()]
L += ['', '# RUNBOOK row 14 — M29 cross-sectional overnight continuation, large caps ($50M+/day)', '']
for sp in ('TRAIN', 'VAL', 'TEST'):
    x = lc[lc.split == sp]
    if not len(x): continue
    thr = lc[lc.split == 'TRAIN'].on20.quantile(0.9)
    dec = x[x.on20 >= thr]; top4 = x.sort_values(['bar_date', 'on20'], ascending=[True, False]).groupby('bar_date').head(4)
    for nm, y in (('top-decile', dec), ('top-4 book', top4)):
        net = y.ret_on_next - y.cost
        L.append(f"{sp:5s} {nm:11s} n {len(y):6d} gross {y.ret_on_next.mean()*1e4:+6.1f} bps median {y.ret_on_next.median()*1e4:+5.1f} net {net.mean()*1e4:+6.1f} "
                 f"t {net.mean()/(net.std()/np.sqrt(len(y))):+5.2f} hit {(net>0).mean()*100:4.1f}%")

# ---------------- M36r large-loser reversal by market-volatility regime ----------------
spy = d[d.symbol.astype(str) == 'SPY'][['bar_date', 'ret_cc']].sort_values('bar_date')
spy['mktvol'] = spy.ret_cc.rolling(20, min_periods=15).std()                      # causal: uses days <= t
reg = spy.set_index('bar_date').mktvol
lo, hi = reg[reg.index < '2026-01-01'].quantile([1/3, 2/3])
d['mktvol'] = d.bar_date.map(reg); d['regime'] = np.where(d.mktvol >= hi, 'high', np.where(d.mktvol <= lo, 'low', 'mid'))
lz = d[(d.ret_id <= -0.08) & (d.vol_ratio >= 2) & (d.dvol20 >= 2e6) & d.ret_id_next.notna()].copy()
L += ['', '# RUNBOOK row 6 refinement — M36 large-loser reversal by market-volatility regime (trailing 20-day SPY vol tercile, cut on 2025)', '']
for sp in ('TRAIN', 'VAL', 'TEST'):
    for rg in ('low', 'mid', 'high'):
        x = lz[(lz.split == sp) & (lz.regime == rg)]
        if len(x) < 30: L.append(f'{sp:5s} {rg:4s} n {len(x)} — underpowered'); continue
        x = x.sort_values(['bar_date', 'ret_id']).groupby('bar_date').head(4); net = x.ret_id_next - x.cost
        L.append(f"{sp:5s} {rg:4s} n {len(x):5d} gross {x.ret_id_next.mean()*1e4:+7.1f} bps median {x.ret_id_next.median()*1e4:+6.1f} net {net.mean()*1e4:+7.1f} "
                 f"t {net.mean()/(net.std()/np.sqrt(len(x))):+5.2f} hit {(net>0).mean()*100:4.1f}%")
open('research/lit_review_2026/daily_queue.md', 'w').write('\n'.join(L)); print('\n'.join(L)); print('DONE', flush=True)
