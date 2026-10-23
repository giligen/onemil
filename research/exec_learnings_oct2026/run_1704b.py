"""1,704 supplement: R50 diff and wider slot buckets (book takes ~10 fills/day)."""
import pandas as pd, numpy as np
exec(open('run_1704.py').read().split("print(\"== BOOK")[0])
ps = p[p.trade_date.isin(set(b.trade_date))]
bb = b.copy(); bb['R'] = bb.R50
ci, sd = boot_diff(bb, ps); print("R50 book-pop %+.3f CI [%+.3f,%+.3f]" % (bb.R.mean() - ps.R.mean(), *ci))
print("pop stop share %.1f%% target %.1f%% eod %.1f%% | book stop %.1f%% target %.1f%% eod %.1f%%" % (
  *(100*ps.exit_reason.value_counts(normalize=True).reindex(['stop','target','eod']).fillna(0).values), *(100*b.exit_reason.replace({'stop_loss':'stop'}).value_counts(normalize=True).reindex(['stop','target','eod']).fillna(0).values)))
p['bkt'] = pd.cut(p.slot, [0, 2, 10, 30, 1000], labels=['1-2', '3-10', '11-30', '31+'])
for k, d in p.groupby('bkt', observed=True): print(line(f'slot {k}', d))
e = p[p.slot <= 10]; l = p[p.slot > 10]; ci, _ = boot_diff(e, l); print("slot<=10 minus >10 = %+.3f CI [%+.3f,%+.3f]" % (e.R.mean() - l.R.mean(), *ci))
days = sorted(p.trade_date.unique()); h = days[:len(days)//2]
for nm, m in (('H1', p.trade_date.isin(h)), ('H2', ~p.trade_date.isin(h))):
    d = p[m]; x = d[d.slot <= 10]; y = d[d.slot > 10]; ci, _ = boot_diff(x, y); print(nm, "<=10 %+.3f (n%d) >10 %+.3f (n%d) diff %+.3f CI [%+.3f,%+.3f]" % (x.R.mean(), len(x), y.R.mean(), len(y), x.R.mean()-y.R.mean(), *ci))
print("book risk median %.1f mean %.1f; R>=5 fills: %d, their pnl %.0f" % (b.risk.median(), b.risk.mean(), (b.R>=5).sum(), b[b.R>=5].pnl.sum()))
print("corr(risk,R)=%.2f; mean R risk<20: %.3f (n%d) risk>=20: %.3f (n%d)" % (np.corrcoef(b.risk, b.R)[0,1], b[b.risk<20].R.mean(), (b.risk<20).sum(), b[b.risk>=20].R.mean(), (b.risk>=20).sum()))
