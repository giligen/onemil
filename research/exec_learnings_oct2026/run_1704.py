"""Cell 1,704: HOD forward book (trades.db) vs all-armed population (dry_trades). Read-only on the DB."""
import sqlite3, numpy as np, pandas as pd
rng = np.random.default_rng(1704)
con = sqlite3.connect('file:/home/ec2-user/onemil/data/trades.db?mode=ro', uri=True)
START, END = '2026-09-25', '2026-10-22'
b = pd.read_sql("select id,trade_date,symbol,fill_price,stop_loss_price,shares,pnl,exit_reason,pattern_data,filled_at,exited_at from trades "
                "where strategy='hod_break' and pnl is not null and trade_date between ? and ? order by id", con, params=(START, END))
b['risk'] = b.shares * (b.fill_price - b.stop_loss_price).abs()
b['risk_src'] = np.where(b.risk > 0.5, 'fill-stop', 'fallback50')
b['risk'] = np.where(b.risk > 0.5, b.risk, 50.0)
b['R'] = b.pnl / b.risk
b['R50'] = b.pnl / 50.0
b['adopted'] = b.pattern_data.fillna('').str.contains('adopted_on_boot')
b.drop(columns=['pattern_data']).to_csv('hod_1704_book.csv', index=False)
p = pd.read_sql("select * from dry_trades where strategy='hod_break' and trade_date between ? and ? order by trade_date,entry_ts,id", con, params=(START, END))
p = p[~p.source.isin(['backfill_journal', 'replay_capped'])]
p = p[p.r_multiple.notna()].copy()
p['R'] = p.r_multiple
p['slot'] = p.groupby('trade_date').cumcount() + 1
p['slot_b'] = p.slot.clip(upper=3)
p.drop(columns=['extra_json']).to_csv('hod_1704_population.csv', index=False)

def stats(df, col='R', day='trade_date'):
    """mean, iid SE, day-clustered SE, n, n days"""
    x = df[col].values; n = len(x); m = x.mean()
    se_iid = x.std(ddof=1) / np.sqrt(n)
    g = df.assign(_e=x - m).groupby(day)._e.sum().values
    G = len(g); se_cl = np.sqrt(G / (G - 1) * (g ** 2).sum()) / n if G > 1 else np.nan
    return m, se_iid, se_cl, n, G
def line(name, df, col='R'):
    m, si, sc, n, G = stats(df, col)
    return f"{name:<28} n={n:4d} days={G:2d} mean={m:+.3f} iidSE={si:.3f} clSE={sc:.3f} CI95cl=[{m-1.96*sc:+.3f},{m+1.96*sc:+.3f}]"
def boot_diff(a, c, col='R', reps=4000):
    """day-cluster bootstrap of mean(a)-mean(c), resampling days jointly (days shared) or separately if not."""
    days = sorted(set(a.trade_date) | set(c.trade_date))
    ga = {d: x[col].values for d, x in a.groupby('trade_date')}; gc = {d: x[col].values for d, x in c.groupby('trade_date')}
    out = []
    for _ in range(reps):
        s = rng.choice(len(days), len(days))
        xa = np.concatenate([ga.get(days[i], np.array([])) for i in s]); xc = np.concatenate([gc.get(days[i], np.array([])) for i in s])
        if len(xa) and len(xc): out.append(xa.mean() - xc.mean())
    out = np.array(out); return np.percentile(out, [2.5, 97.5]), out.std()
print("== BOOK window", START, END, "rows", len(b), "ops/fallback risk rows", (b.risk_src == 'fallback50').sum(), "adopted", b.adopted.sum())
print(b.R.describe().round(3).to_dict())
print(line("BOOK R (fill-stop)", b)); print(line("BOOK R50 (pnl/50)", b, 'R50'))
print("BOOK pnl", round(b.pnl.sum(), 1), "sumR", round(b.R.sum(), 1), "sd pnl/fill", round(b.pnl.std(ddof=1), 2))
m, si, sc, n, G = stats(b); print("vs prior -0.21: z_cl=%.2f z_iid=%.2f" % ((m + .21) / sc, (m + .21) / si))
print("-- by exit reason"); 
for k, d in b.groupby('exit_reason'): print(f"{k:<24} n={len(d):3d} meanR={d.R.mean():+.3f} sumR={d.R.sum():+.1f} pnl={d.pnl.sum():+.0f} med_risk={d.risk.median():.1f}")
print("-- weekly")
b['wk'] = pd.to_datetime(b.trade_date).dt.to_period('W-FRI').astype(str)
for k, d in b.groupby('wk'): print(k, f"n={len(d)} sumR={d.R.sum():+.1f} meanR={d.R.mean():+.3f} pnl={d.pnl.sum():+.0f}")
def extop(df, col='R', f=.05):
    k = int(np.ceil(f * len(df))); d = df.sort_values(col, ascending=False).iloc[k:]; return d[col].mean(), k
print("ex-top5%% BOOK meanR %.3f (dropped %d); R50 %.3f; ex-top1%% %.3f; winsor R at +-3: %.3f" % (*extop(b), extop(b, 'R50')[0], extop(b, 'R', .01)[0], b.R.clip(-3, 3).mean()))
print("max R", b.R.max(), "min R", b.R.min(), "pnl ex-top5 %", b.sort_values('pnl', ascending=False).iloc[int(np.ceil(.05*len(b))):].pnl.sum())
print("\n== POPULATION"); print(p.groupby('source').size().to_dict())
print(line("POP all", p)); print("POP exits", p.exit_reason.value_counts().to_dict())
print("POP ex-top5%% %.3f" % extop(p)[0])
bd = set(b.trade_date); ps = p[p.trade_date.isin(bd)]
print("-- (a) same days: book days", len(bd), "pop days", ps.trade_date.nunique())
print(line("BOOK same days", b)); print(line("POP same days", ps))
ci, sd = boot_diff(b, ps); print(f"(a) book - pop = {b.R.mean()-ps.R.mean():+.3f}  day-boot CI95 [{ci[0]:+.3f},{ci[1]:+.3f}] sd {sd:.3f}")
print("book fills/day %.1f  pop arms/day %.1f ; book share of pop %.1f%%" % (len(b)/len(bd), len(ps)/len(bd), 100*len(b)/len(ps)))
print("-- (b) slot order in population")
for k, d in p.groupby('slot_b'): print(line(f"slot {k}{'+' if k==3 else ''}", d))
a12 = p[p.slot_b <= 2]; a3 = p[p.slot_b == 3]; a1 = p[p.slot == 1]
ci, sd = boot_diff(a12, a3); print(f"(b) slot1-2 minus slot3+ = {a12.R.mean()-a3.R.mean():+.3f} CI95 [{ci[0]:+.3f},{ci[1]:+.3f}]")
ci1, _ = boot_diff(a1, p[p.slot > 1]); print(f"    slot1 minus slot2+ = {a1.R.mean()-p[p.slot>1].R.mean():+.3f} CI95 [{ci1[0]:+.3f},{ci1[1]:+.3f}]")
days = sorted(p.trade_date.unique()); half = days[:len(days)//2]
for nm, hd in (('H1', p[p.trade_date.isin(half)]), ('H2', p[~p.trade_date.isin(half)])):
    x12 = hd[hd.slot_b <= 2]; x3 = hd[hd.slot_b == 3]; ci, _ = boot_diff(x12, x3)
    print(f"    {nm} days={hd.trade_date.nunique()} slot1-2 {x12.R.mean():+.3f} (n{len(x12)}) slot3+ {x3.R.mean():+.3f} (n{len(x3)}) diff {x12.R.mean()-x3.R.mean():+.3f} CI [{ci[0]:+.3f},{ci[1]:+.3f}]")
print("pop days with >=3 arms:", (p.groupby('trade_date').size() >= 3).sum(), "of", p.trade_date.nunique(), " slots 1-2 share %.1f%%" % (100*len(a12)/len(p)))
print("-- (c) last 2 weeks 10/12-10/22")
l = b[b.trade_date >= '2026-10-12']; r = b[b.trade_date < '2026-10-12']
print("last2w n", len(l), "pnl", round(l.pnl.sum(), 1), "mean/fill", round(l.pnl.mean(), 2), "book sd/fill", round(b.pnl.std(ddof=1), 2))
z0 = l.pnl.sum() / (b.pnl.std(ddof=1) * np.sqrt(len(l))); print("z vs zero-mean: %.2f" % z0)
mu = b.pnl.mean(); print("z vs whole-window mean (%.2f/fill): %.2f" % (mu, (l.pnl.mean() - mu) / (b.pnl.std(ddof=1) / np.sqrt(len(l)))))
sd_r = r.pnl.std(ddof=1); print("prior 105 fills mean %.2f; z of last2w vs prior-period mean: %.2f" % (r.pnl.mean(), (l.pnl.mean() - r.pnl.mean()) / np.sqrt(sd_r**2/len(r) + b.pnl.std(ddof=1)**2/len(l))))
dl = l.groupby('trade_date').pnl.sum(); print("day-clustered: 8 days pnl", dl.round(0).tolist(), "t vs 0 = %.2f" % (dl.mean() / (dl.std(ddof=1) / np.sqrt(len(dl)))))
print("last2w pop same days meanR %.3f n %d ; book meanR %.3f" % (p[p.trade_date >= '2026-10-12'].R.mean(), (p.trade_date >= '2026-10-12').sum(), l.R.mean()))
print("last2w ex-top-3 pnl:", round(l.sort_values('pnl', ascending=False).iloc[3:].pnl.sum(), 1), " top3:", l.pnl.nlargest(3).round(0).tolist())
