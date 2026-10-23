"""Cells 1,702 (exit decomposition) and 1,706 (marketable-at-submit cohort), frozen in PREREG.md. Read-only on all data."""
import sqlite3, numpy as np, pandas as pd
D = '/home/ec2-user/onemil/'; OUT = D + 'research/exec_learnings_oct2026/'
rng = np.random.default_rng(1702)
con = sqlite3.connect('file:' + D + 'data/trades.db?mode=ro', uri=True)
t = pd.read_sql("select id,trade_date date,symbol,order_id,fill_price entry,stop_loss_price stop,filled_qty,pnl,exit_reason,exit_price,"
                "filled_at,exited_at,coalesce(account,'live') account from trades where strategy='orb' and fill_price is not null and pnl is not null order by id", con)
b = pd.read_csv(OUT + 'broker_ts.csv')[['order_id', 'brk_created', 'brk_filled']]
t = t.merge(b, on='order_id', how='left')
f0 = pd.read_csv(OUT + 'fills_1700.csv')
t['risk'] = t.entry - t.stop
t['R'] = t.pnl / (t.filled_qty * t.risk)
t['L_brk'] = (pd.to_datetime(t.brk_filled, utc=True, format='ISO8601') - pd.to_datetime(t.brk_created, utc=True, format='ISO8601')).dt.total_seconds()
t = t.merge(f0[['date', 'symbol', 'drift_bps', 'L_eng']].rename(columns={'drift_bps': 'drift_bps'}) if False else
            pd.read_csv(OUT + 'fills_1700.csv')[['date', 'symbol', 'drift_bps', 'spread_bps', 'exit_slip_bps']], on=['date', 'symbol'], how='left')
assert len(t) == 137, len(t)
for c in ['filled_at', 'exited_at', 'brk_filled']: t[c] = pd.to_datetime(t[c], utc=True, format='ISO8601')
t['ent_t'] = t.brk_filled
t['min_to_exit'] = (t.exited_at - t.ent_t).dt.total_seconds() / 60
t['min_to_exit_db'] = (t.exited_at - t.filled_at).dt.total_seconds() / 60
t['fill_db_minus_brk_s'] = (t.filled_at - t.brk_filled).dt.total_seconds()
bars = sqlite3.connect('file:' + D + 'research/bf_zero/bars_sip.db?mode=ro', uri=True)
mf, ma, mf2, ma2, cov = [], [], [], [], []
for r in t.itertuples():
    e0 = r.ent_t.floor('min'); x0 = r.exited_at.floor('min')
    q = "select t,h,l from bars where symbol=? and day=? and t>=? and t<=? order by t"
    def get(lo):
        return pd.read_sql(q, bars, params=[r.symbol, r.date, lo.strftime('%Y-%m-%dT%H:%M:00+00:00'), x0.strftime('%Y-%m-%dT%H:%M:00+00:00')])
    inc = get(e0); strict = inc[inc.t > e0.strftime('%Y-%m-%dT%H:%M:00+00:00')]
    cov.append(len(inc) > 0)
    mf2.append((inc.h.max() - r.entry) / r.risk if len(inc) else np.nan); ma2.append((r.entry - inc.l.min()) / r.risk if len(inc) else np.nan)
    mf.append((strict.h.max() - r.entry) / r.risk if len(strict) else np.nan); ma.append((r.entry - strict.l.min()) / r.risk if len(strict) else np.nan)
t['MFE_R'] = mf; t['MAE_R'] = ma; t['MFE_R_incl'] = mf2; t['MAE_R_incl'] = ma2; t['bars_ok'] = cov
t['cohort5'] = np.where(t.L_brk <= 5, 'chase', 'resting')
t.drop(columns=['brk_created']).to_csv(OUT + 'fills_1702.csv', index=False)

def boot(df, fn, n=2000):
    """Day-clustered bootstrap 95% CI of fn(df)."""
    g = {d: x for d, x in df.groupby('date')}; days = list(g); out = []
    for _ in range(n):
        s = pd.concat([g[days[i]] for i in rng.integers(0, len(days), len(days))])
        v = fn(s)
        if np.isfinite(v): out.append(v)
    return np.percentile(out, [2.5, 97.5])
def ci(df, col='R'):
    lo, hi = boot(df, lambda s: s[col].mean()); return '[%.2f,%.2f]' % (lo, hi)

print('n', len(t), 'bars_ok', int(t.bars_ok.sum()), 'missing', int((~t.bars_ok).sum()))
print('missing by date:', t[~t.bars_ok].groupby('date').size().to_dict())
print('missing is paper? ', t[~t.bars_ok].account.value_counts().to_dict())
print('fill_db - brk_filled s: median %.1f p90 %.1f max %.1f; neg %d' % (t.fill_db_minus_brk_s.median(), t.fill_db_minus_brk_s.quantile(.9), t.fill_db_minus_brk_s.max(), (t.fill_db_minus_brk_s < -1).sum()))
print('\n=== 1702 by exit_reason (all 137) ===')
for k, g in list(t.groupby('exit_reason')) + [('ALL', t)]:
    h = g[g.bars_ok]
    print('%-26s n %3d  sum$ %8.0f  meanR %.3f %s  | bars n %d MFE %.2f MAE %.2f (strict) | incl-entry-bar MFE %.2f MAE %.2f | min_to_exit med %.1f p25 %.1f p75 %.1f p90 %.1f' % (
        k, len(g), g.pnl.sum(), g.R.mean(), ci(g), len(h), h.MFE_R.mean(), h.MAE_R.mean(), h.MFE_R_incl.mean(), h.MAE_R_incl.mean(),
        g.min_to_exit.median(), g.min_to_exit.quantile(.25), g.min_to_exit.quantile(.75), g.min_to_exit.quantile(.9)))
sl = t[t.exit_reason.str.startswith('stop_loss')]
print('\nstop-loss exits n', len(sl), ' within 5min of broker fill: %d (%.0f%%); within 2: %d; within 1: %d; db-fill basis within5: %d' % (
    (sl.min_to_exit <= 5).sum(), 100 * (sl.min_to_exit <= 5).mean(), (sl.min_to_exit <= 2).sum(), (sl.min_to_exit <= 1).sum(), (sl.min_to_exit_db <= 5).sum()))
print('  of all fills: stop share %.1f%%; stop within 5min share of ALL fills %.1f%%' % (100 * len(sl) / len(t), 100 * (sl.min_to_exit <= 5).sum() / len(t)))
f5 = sl[sl.min_to_exit <= 5]; s5 = sl[sl.min_to_exit > 5]
print('  stops<=5min: sumR meanR %.2f $ %.0f exit_slip_bps med %.1f | stops>5: meanR %.2f $ %.0f slip med %.1f' % (f5.R.mean(), f5.pnl.sum(), f5.exit_slip_bps.median(), s5.R.mean(), s5.pnl.sum(), s5.exit_slip_bps.median()))
print('  stops<=5min by chase/rest:', f5.groupby('cohort5').size().to_dict(), ' by account', f5.groupby('account').size().to_dict())
print('  stops<=5min: MAE strict mean %.2f; MFE strict mean %.2f (n %d)' % (f5.MAE_R.mean(), f5.MFE_R.mean(), f5.bars_ok.sum()))
print('  min_to_exit negative/NaN:', int((t.min_to_exit < 0).sum()), int(t.min_to_exit.isna().sum()))
print('  winners: share of fills with MFE>=1R strict %.2f; stops with MFE>=1R %.2f' % ((t.MFE_R >= 1).mean(), (sl.MFE_R >= 1).mean()))
bt = pd.read_csv(D + 'analysis_results/orb_bplus_book.csv'); be = bt[bt.entered == True]
print('BT book entered n', len(be), 'exit mix %', (be.exit_reason.value_counts(normalize=True) * 100).round(1).to_dict(), 'dates', be.date.min(), be.date.max())
bw = be[(be.date >= '2026-05-19') & (be.date <= '2026-09-23')]
print('BT same window n', len(bw), (bw.exit_reason.value_counts(normalize=True) * 100).round(1).to_dict())
print('live mix %', (t.exit_reason.value_counts(normalize=True) * 100).round(1).to_dict())

print('\n=== 1706 chase vs resting ===')
for th in [2, 5, 10]:
    c = np.where(t.L_brk <= th, 'chase', 'resting'); print('-- threshold %d s' % th)
    for k in ['chase', 'resting']:
        g = t[c == k]
        mix = g.exit_reason.value_counts().to_dict()
        print('  %-8s n %3d days %2d meanR %.3f %s sumR %.1f $%.0f drift med %.1f mean %.1f | %s' % (k, len(g), g.date.nunique(), g.R.mean(), ci(g), g.R.sum(), g.pnl.sum(), g.drift_bps.median(), g.drift_bps.mean(), mix))
    a = t[c == 'chase']; r_ = t[c == 'resting']
    d = a.R.mean() - r_.R.mean()
    days = t.date.unique(); g = {x: y.assign(c=c[y.index]) for x, y in t.groupby('date')}; o = []
    for _ in range(2000):
        s = pd.concat([g[days[i]] for i in rng.integers(0, len(days), len(days))])
        if (s.c == 'chase').any() and (s.c == 'resting').any(): o.append(s[s.c == 'chase'].R.mean() - s[s.c == 'resting'].R.mean())
    print('  diff chase-resting %.3f CI %s' % (d, np.percentile(o, [2.5, 97.5]).round(2)))
for acc, g0 in t.groupby('account'):
    for k in ['chase', 'resting']:
        g = g0[g0.cohort5 == k]; print('  [%s] %s n %d meanR %.3f $%.0f' % (acc, k, len(g), g.R.mean(), g.pnl.sum()))
# tail: ex top-5% by R within each cohort
for k in ['chase', 'resting']:
    g = t[t.cohort5 == k].sort_values('R'); n5 = max(1, int(round(.05 * len(g)))); print('  %s ex-top-%d R: mean %.3f ; top R values %s' % (k, n5, g.R.iloc[:-n5].mean(), g.R.iloc[-3:].round(2).tolist()))
print('  canceled-ish L_brk NaN:', int(t.L_brk.isna().sum()))
