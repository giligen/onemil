"""Cell 1,700q tradable test (PREREG: only for what passes): P9 (SPY 5-day return) top-quintile flag, expanding-window threshold, sleeve to cash / half size
the following week. Uses the weekly sleeve returns (T2) saved by 1700q_fear.py (= E_REF open-to-open week returns, costs already inside).
Switch cost: 10 bp x |change in exposure| per week (1700c band mean ~5-20 bp). Weekly-compounded DD (REF on the same basis shown beside it)."""
import numpy as np, pandas as pd
W = pd.read_parquet('/home/ec2-user/onemil/research/momentum_weekly/1700q_series.parquet').sort_values('week').reset_index(drop=True)
yrs = (W.week.iloc[-1] - W.week.iloc[0]).days / 365.25
def stats_(r):
    E = 50000 * np.cumprod(1 + r); dd = (E / np.maximum.accumulate(E) - 1).min(); return (E[-1] / 50000) ** (1 / yrs) - 1, dd, E[-1]
def run(q, expo):
    thr = W.P9.expanding(min_periods=100).quantile(q).shift(1)   # threshold from PRIOR weeks only (expanding), P9 itself is Friday-known
    ex = np.where(W.P9 >= thr, expo, 1.0); ex[thr.isna().values] = 1.0
    sw = np.abs(np.diff(np.r_[1.0, ex])); r = ex * W.T2.values - 0.001 * sw
    return stats_(r) + (float((ex < 1).mean()),)
ref = stats_(W.T2.values); print('REF weekly-basis CAGR %.2f%% DD %.1f%% end $%.0f' % (100 * ref[0], 100 * ref[1], ref[2]))
rows = []
for q in (0.70, 0.75, 0.80, 0.85, 0.90):
    for name, ex in (('cash', 0.0), ('half', 0.5)):
        c, d, e, f = run(q, ex); rows.append(dict(q=q, mode=name, cagr=c, dd=d, end=e, flagged=f, both_better=bool(c > ref[0] and d > ref[1])))
R = pd.DataFrame(rows); R.to_csv('/home/ec2-user/onemil/research/momentum_weekly/1700q_trade.csv', index=False); print(R.round(4).to_string())
for mode in ('cash', 'half'):
    nb = R[(R['mode'] == mode) & (R.q != 0.80)]; print(mode, 'neighbours improving both:', int(nb.both_better.sum()), 'of', len(nb))
