"""RECON_1700tu_compare.py: weekly reconciliation of first build (A) vs rebuild (B), from RECON_1700tu_dump.py outputs.
Run: bash scripts/research_run.sh -m 2500M python3 research/momentum_weekly/RECON_1700tu_compare.py > RECON_1700tu_compare.log"""
from pathlib import Path
import numpy as np, pandas as pd
H = Path('/home/ec2-user/onemil/research/momentum_weekly')
P = lambda *a: print(*a, flush=True)


def wk(m, book):
    d = pd.read_csv(H / f'RECON_1700tu_{m}_{book}_weeks.csv', parse_dates=['date']).set_index('date')
    d['hold_ret'] = d.eq_pre.shift(-1) / d.eq_post - 1          # price return of the book held that Monday (gross of next week's cost)
    d['cost_pct'] = d.cost / d.eq_pre
    d['net'] = d.eq_post.shift(-1) / d.eq_post - 1
    return d


def stats(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, (eq / eq.cummax() - 1).min(), eq.iloc[-1]


def main():
    out = {}
    for book in ('plain', 'guarded', 'gated'):
        a, b = wk('A', book), wk('B', book)
        P(book, 'A weeks', len(a), a.index[0].date(), a.index[-1].date(), '| B weeks', len(b), b.index[0].date(), b.index[-1].date())
        idx = a.index.intersection(b.index)
        P('  date sets differ: A-only', [str(x.date()) for x in a.index.difference(b.index)][:6], 'B-only', len(b.index.difference(a.index)))
        ea, eb = a.eq_post.loc[idx], b.eq_post.loc[idx]
        for nm, e in (('A', ea), ('B', eb)):
            c, dd, end = stats(e); P(f'  {nm} common-window Monday marks: CAGR {c:.2%} DD {dd:.1%} end {end:,.0f}')
        out[book] = (a, b, idx)
    a, b, idx = out['guarded']; idx = idx[:-1]
    d = pd.DataFrame({'net_A': a.net.loc[idx], 'net_B': b.net.loc[idx], 'gross_A': a.hold_ret.loc[idx], 'gross_B': b.hold_ret.loc[idx],
                      'cost_A': a.cost_pct.loc[idx], 'cost_B': b.cost_pct.loc[idx], 'traded_A': a.traded.loc[idx] / a.eq_pre.loc[idx], 'traded_B': b.traded.loc[idx] / b.eq_pre.loc[idx]})
    d['dnet'] = d.net_A - d.net_B; d['dgross'] = d.gross_A - d.gross_B; d['dcost'] = d.cost_B - d.cost_A      # positive = A better
    d['cumA'] = a.eq_post.loc[idx] / a.eq_post.loc[idx[0]]; d['cumB'] = b.eq_post.loc[idx] / b.eq_post.loc[idx[0]]
    d.to_csv(H / 'RECON_1700tu_weeks.csv'); n = len(d); yrs = (idx[-1] - idx[0]).days / 365.25
    la, lb = np.log1p(d.net_A).sum(), np.log1p(d.net_B).sum(); gap = la - lb
    P(f'\nweeks {n}; log-growth A {la:.4f} B {lb:.4f} gap {gap:.4f} = {gap / yrs * 100:.2f} %/yr (log pts)')
    P('mean weekly net diff A-B %.4f%%  sd %.3f%%' % (d.dnet.mean() * 100, d.dnet.std() * 100))
    ld = np.log1p(d.net_A) - np.log1p(d.net_B); top = ld.abs().nlargest(10)
    P('top10 |diff| weeks:'); P((d.loc[top.index, ['net_A', 'net_B', 'gross_A', 'gross_B', 'cost_A', 'cost_B']] * 100).round(3).assign(ld=(ld[top.index] * 100).round(3)).to_string())
    P(f'top10 explain {ld[top.index].sum() / gap:.1%} of gap; remainder {(gap - ld[top.index].sum()) / yrs * 100:.2f} %/yr; remainder mean per week {(ld.drop(top.index).mean()) * 100:.4f}%')
    lg = np.log1p(d.gross_A) - np.log1p(d.gross_B); lc = np.log1p(-d.cost_A) - np.log1p(-d.cost_B)
    P(f'gap by part: gross-return {lg.sum() / yrs * 100:.2f} %/yr, cost {lc.sum() / yrs * 100:.2f} %/yr (log pts); cost drag A {d.cost_A.sum() / yrs * 100:.2f} %/yr B {d.cost_B.sum() / yrs * 100:.2f} %/yr; turnover/wk A {d.traded_A.mean():.3f} B {d.traded_B.mean():.3f}')
    by = pd.DataFrame({'gross': lg, 'cost': lc}).groupby(d.index.year).sum() * 100; by['total'] = by.gross + by.cost; P('by year (log pts A-B):'); P(by.round(2).to_string())
    big = lg.abs().nlargest(8); P('top |gross diff| weeks', [(str(i.date()), round(lg[i] * 100, 3)) for i in big.index])
    P('n weeks |gross diff|>0.05%:', int((lg.abs() > 5e-4).sum()), ' >0.2%:', int((lg.abs() > 2e-3).sum()))
    # per-name attribution
    ha = pd.read_csv(H / 'RECON_1700tu_A_guarded_hold.csv', parse_dates=['date']); hb = pd.read_csv(H / 'RECON_1700tu_B_guarded_hold.csv', parse_dates=['date'])
    ha = ha[ha.kept >= 0]; hb = hb[hb.kept >= 0]
    weeks = list(ld.abs().nlargest(5).index) + [pd.Timestamp('2020-08-31'), pd.Timestamp('2020-08-24')]
    for w in weeks:
        x = ha[ha.date == w].set_index('sym'); y = hb[hb.date == w].set_index('sym')
        j = x.join(y, lsuffix='_A', rsuffix='_B', how='outer')
        j['ret_A'] = j.p_out_A / j.p_in_A - 1; j['ret_B'] = j.p_out_B / j.p_in_B - 1
        P(f'\n== week {w.date()} A-only {sorted(set(x.index) - set(y.index))} B-only {sorted(set(y.index) - set(x.index))} net A {d.net_A.get(w, np.nan):.4%} B {d.net_B.get(w, np.nan):.4%} gross A {d.gross_A.get(w, np.nan):.4%} B {d.gross_B.get(w, np.nan):.4%}')
        j['dcontrib'] = j.w_A.fillna(0) * j.ret_A.fillna(0) - j.w_B.fillna(0) * j.ret_B.fillna(0)
        P(j.loc[j.dcontrib.abs().nlargest(6).index, ['w_A', 'w_B', 'p_in_A', 'p_in_B', 'p_out_A', 'p_out_B', 'ret_A', 'ret_B', 'cost_A', 'cost_B', 'dcontrib']].round(5).to_string())
        P('   sum w A', round(x.w.sum(), 4), 'B', round(y.w.sum(), 4), ' price in/out mismatch names:', int(((j.p_in_A - j.p_in_B).abs() > 1e-3 * j.p_in_A).sum()), int(((j.p_out_A - j.p_out_B).abs() > 1e-3 * j.p_out_A).sum()))
    # global name/price audit
    m = ha.merge(hb, on=['date', 'sym'], suffixes=('_A', '_B'))
    P('\nname-weeks A', len(ha), 'B', len(hb), 'common', len(m), ' p_in mismatch', int(((m.p_in_A - m.p_in_B).abs() > 1e-3 * m.p_in_A).sum()), ' p_out mismatch', int(((m.p_out_A - m.p_out_B).abs() > 1e-3 * m.p_out_A).sum()))
    P('weight diff in common: mean |w_A-w_B| %.5f ; rate_A mean %.5f rate_B mean %.5f ; cost-weighted: A %.5f B %.5f' % ((m.w_A - m.w_B).abs().mean(), m.rate_A.mean(), m.rate_B.mean(), m.cost_A.sum() / 1, m.cost_B.sum() / 1))
    ok = m.p_out_A.notna() & m.p_out_B.notna(); P('mean stock return A %.5f B %.5f' % ((m.p_out_A / m.p_in_A - 1)[ok].mean(), (m.p_out_B / m.p_in_B - 1)[ok].mean()))
    mm = m[(m.p_out_A - m.p_out_B).abs() > 1e-3 * m.p_out_A]; P('p_out mismatches sample:'); P(mm[['date', 'sym', 'p_in_A', 'p_out_A', 'p_out_B']].head(12).to_string())
    P('last-week: A last rebal', a.index[-1].date(), 'B last', b.index[-1].date())


main()
