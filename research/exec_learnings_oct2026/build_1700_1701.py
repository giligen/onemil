"""Cells 1,700 (latency -> outcome) and 1,701 (spread -> outcome), frozen in PREREG.md. Read-only on all data."""
import re, glob, sqlite3, numpy as np, pandas as pd
from scipy.stats import spearmanr

D = '/home/ec2-user/onemil/'
OUT = D + 'research/exec_learnings_oct2026/'
rng = np.random.default_rng(1700)
con = sqlite3.connect('file:' + D + 'data/trades.db?mode=ro', uri=True)
t = pd.read_sql("select id,trade_date,symbol,entry_price,fill_price,stop_loss_price,shares,filled_qty,pnl,exit_reason,"
                "exit_price,exit_trigger_price,exit_slippage,entry_quote_bid,entry_quote_ask,entry_quote_spread,"
                "drift_ask_to_fill_bps,created_at,order_submitted_at,order_filled_at,coalesce(account,'live') account,order_id,"
                "json_extract(pattern_data,'$.range_high') rh from trades where strategy='orb' and fill_price is not null "
                "and pnl is not null order by id", con)
b = pd.read_csv(OUT + 'broker_ts.csv')[['order_id', 'brk_created', 'brk_submitted', 'brk_filled']]
t = t.merge(b, on='order_id', how='left')
assert t.brk_created.notna().all()
for c in ['brk_created', 'brk_submitted', 'brk_filled']:
    t[c] = pd.to_datetime(t[c], utc=True, format='ISO8601')
t['L_brk'] = (t.brk_filled - t.brk_created).dt.total_seconds()
t['L_brk_sub'] = (t.brk_filled - t.brk_submitted).dt.total_seconds()
# engine latency: SUBMIT LATENCY lines where present (9/28+), else first-order offset from 09:35:00 ET (13:35 UTC, EDT all window)
lat = {}
for f in sorted(glob.glob(D + 'logs/session_archive/*.log')):
    for ln in open(f, errors='ignore'):
        m = re.search(r'(\d{4}-\d\d-\d\d) [\d:]+ \| .*SUBMIT LATENCY sym=(\w+) seconds=([\d.]+)', ln)
        if m: lat.setdefault((m.group(1), m.group(2)), float(m.group(3)))
off = (pd.to_datetime(t.brk_created, utc=True) - pd.to_datetime(t.trade_date + ' 13:35:00', utc=True)).dt.total_seconds()
t['L_eng'] = [lat.get((d, s), o) for d, s, o in zip(t.trade_date, t.symbol, off)]
t['L_eng_src'] = ['log' if (d, s) in lat else 'offset' for d, s in zip(t.trade_date, t.symbol)]
t['entry'] = t.fill_price
t['R'] = t.pnl / (t.filled_qty * (t.entry - t.stop_loss_price))
t['drift_bps'] = (t.fill_price - t.entry_quote_ask) / t.entry_quote_ask * 1e4
t['drift_rh_bps'] = (t.fill_price - t.rh.astype(float)) / t.rh.astype(float) * 1e4
mid = (t.entry_quote_bid + t.entry_quote_ask) / 2
t['spread_bps'] = t.entry_quote_spread / mid * 1e4
trg = t.exit_trigger_price.fillna(pd.Series(np.where(t.exit_reason == 'stop_loss', t.stop_loss_price, np.nan), index=t.index))
t['exit_slip_bps'] = (trg - t.exit_price) / trg * 1e4
t['date'] = t.trade_date
cols = ['date', 'symbol', 'L_eng', 'L_brk', 'drift_bps', 'spread_bps', 'R', 'exit_reason', 'account', 'L_brk_sub', 'L_eng_src',
        'drift_rh_bps', 'exit_slip_bps', 'pnl', 'filled_qty', 'entry', 'stop_loss_price']
t[cols].to_csv(OUT + 'fills_1700.csv', index=False)
print('fills', len(t), t.account.value_counts().to_dict(), 'R>5 or <-5:', int((t.R.abs() > 5).sum()), flush=True)

def boot(df, fn, n=2000):
    """Day-clustered bootstrap CI (2.5/97.5) of fn(df)."""
    days = df.date.unique(); g = {d: x for d, x in df.groupby('date')}; out = []
    for _ in range(n):
        s = pd.concat([g[d] for d in rng.choice(days, len(days))])
        try: out.append(fn(s))
        except Exception: pass
    out = np.array([o for o in out if np.isfinite(o)])
    return np.percentile(out, [2.5, 97.5])

def sp(df, a, c):
    d = df[[a, c]].dropna()
    return spearmanr(d[a], d[c])[0]

def report(df, tag):
    print('\n=== %s n=%d days=%d meanR=%.3f ===' % (tag, len(df), df.date.nunique(), df.R.mean()), flush=True)
    for a in ['L_brk', 'L_brk_sub', 'L_eng']:
        for c in ['R', 'drift_bps']:
            ci = boot(df, lambda s: sp(s, a, c)); p = spearmanr(df[a], df[c])
            print('(a) spearman(%s,%s)=%.3f CI[%.3f,%.3f] iid p=%.3f' % (a, c, p[0], ci[0], ci[1], p[1]), flush=True)
    q = pd.qcut(df.L_brk, 3, labels=['T1', 'T2', 'T3'], duplicates='drop'); df = df.assign(terc=q)
    print('(b) L_brk tercile edges', df.groupby('terc', observed=True).L_brk.agg(['min', 'max']).round(1).values.tolist())
    print(df.groupby('terc', observed=True).R.agg(['count', 'mean', 'sem']).round(3).to_string())
    dif = lambda s: s[s.terc == 'T1'].R.mean() - s[s.terc == 'T3'].R.mean()
    print('   T1-T3 dR=%.3f CI%s' % (dif(df), np.round(boot(df, dif), 3)), flush=True)
    lo, hi = df[df.drift_bps <= 30], df[df.drift_bps > 30]
    dd = lambda s: s[s.drift_bps <= 30].R.mean() - s[s.drift_bps > 30].R.mean()
    print('(c) drift<=30: n=%d R=%.3f | >30: n=%d R=%.3f | diff=%.3f CI%s' % (len(lo), lo.R.mean(), len(hi), hi.R.mean(), dd(df), np.round(boot(df, dd), 3)), flush=True)
    print('    median drift %.1f bps; share fills drift<=30 = %.2f' % (df.drift_bps.median(), len(lo) / len(df)))

live = t[t.account == 'live']
report(live, 'LIVE')
report(t, 'POOLED live+paper')
print('\nL_brk describe live', live.L_brk.describe().round(1).to_dict())
print('L_eng describe live', live.L_eng.describe().round(1).to_dict(), live.L_eng_src.value_counts().to_dict())
print('L_eng describe paper', t[t.account == 'paper'].L_eng.describe().round(1).to_dict())
print('spearman(L_eng,R) live=%.3f  paper-only L_brk vs R=%.3f' % (sp(live, 'L_eng', 'R'), sp(t[t.account == 'paper'], 'L_brk', 'R')))

# ---- 1701
def bucket(x): return '<50' if x < 50 else ('50-150' if x < 150 else ('150-300' if x < 300 else '>=300'))
def r1701(df, tag):
    df = df.assign(bk=df.spread_bps.map(bucket))
    print('\n=== 1701 %s n=%d spread median %.0f bps ===' % (tag, len(df), df.spread_bps.median()))
    g = df.groupby('bk').agg(n=('R', 'size'), R=('R', 'mean'), Rsem=('R', 'sem'), drift=('drift_bps', 'mean'), drift_med=('drift_bps', 'median'),
                             exitslip=('exit_slip_bps', 'mean'), pnl=('pnl', 'sum'))
    g['RT_bps'] = g.drift + g.exitslip
    print(g.round(2).to_string(), flush=True)
    for bk, x in df.groupby('bk'):
        if len(x) > 3:
            print('  bucket %s R CI day-clustered %s' % (bk, np.round(boot(x, lambda s: s.R.mean(), 1000), 3)))
    wide = df[df.spread_bps >= 150]; loss = df[df.pnl < 0].pnl.sum()
    print('  total pnl %.0f; names>=150bps n=%d pnl %.0f (%.0f%% of fills), their losing pnl %.0f of total losing %.0f (%.0f%%)' % (
        df.pnl.sum(), len(wide), wide.pnl.sum(), 100 * len(wide) / len(df), wide[wide.pnl < 0].pnl.sum(), loss, 100 * wide[wide.pnl < 0].pnl.sum() / loss))
    print('  Spearman(spread,R)=%.3f CI%s ; Spearman(spread,drift)=%.3f CI%s' % (
        sp(df, 'spread_bps', 'R'), np.round(boot(df, lambda s: sp(s, 'spread_bps', 'R'), 1000), 3),
        sp(df, 'spread_bps', 'drift_bps'), np.round(boot(df, lambda s: sp(s, 'spread_bps', 'drift_bps'), 1000), 3)))
    # half-spread cost in R at entry
    hs = (df.entry_quote_spread / 2 * df.filled_qty) / (df.filled_qty * (df.entry - df.stop_loss_price))
    print('  half-spread as R: mean %.3f; wide>=150 mean %.3f' % (hs.mean(), hs[df.spread_bps >= 150].mean()))
r1701(live, 'LIVE'); r1701(t, 'POOLED')
print('R sd live %.2f' % live.R.std(), 'exit reasons', live.exit_reason.value_counts().to_dict())
