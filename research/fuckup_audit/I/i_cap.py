#!/usr/bin/env python3
"""Stage I — the liquidity-capped twin (H/F6_sizing's method) for every stack, at $150 / $300 / $600 risk.

Cap: drop every POPULATION row whose participation `shares*entry / 5-minute $ volume` exceeds 1% and re-run
run_book(12,4) on the survivors, so a freed slot refills. Also the live-computable variant (four PRIOR bars x1.25).
$/month = booked net R summed per month x risk.  Frozen exit = `hold` (FREEZE.md).
Writes I/c_capacity.md, I/cap_monthly.csv (the frozen stack at $300, all three splits), I/cap_cells.csv.
"""
import numpy as np, pandas as pd
from icore import D, STACKS, load, weeks_of, dedupe, book, stats

RISKS = [150, 300, 600]
OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s); print(s, flush=True)


c = load(f'{D}/pop_i_liq.csv')
W = weeks_of(c)
c['risk_per_share'] = c.next_entry * c.next_r_pct / 100.0
say('# Stage I Part A — the liquidity-capped twin (money, not R)')
say('')
say(f'rows {len(c):,} | symbol-day found {c.sym_day_found.mean():.4f} | '
    f'fill bar present {c.fill_bar_dollar_vol.notna().mean():.4f} | '
    f'fill inside its own bar {c.fill_in_bar.mean():.4%} | 5 of 5 minutes printed {(c.n_bars_in_5 == 5).mean():.1%}')
say('')
say('| quantity | p10 | median | mean | p90 |')
say('|---|---:|---:|---:|---:|')
for lbl, col, nd in [('entry price $', c.next_entry, 2), ('R per share $', c.risk_per_share, 2),
                     ('R as % of price', c.next_r_pct, 2), ('fill-bar $ volume', c.fill_bar_dollar_vol, 0),
                     ('5-min $ volume', c.five_min_dollar_vol, 0), ('spread_cc_bps', c.spread_cc_bps, 1)]:
    say(f'| {lbl} | {col.quantile(.10):,.{nd}f} | {col.quantile(.50):,.{nd}f} | {col.mean():,.{nd}f} | '
        f'{col.quantile(.90):,.{nd}f} |')
say('')

cells = []
say('## Capped book, exit `hold`, per stack and risk level')
say('')
say('| stack | risk $ | split | pop kept @1% | n | tr/wk | mean net R | t | wkR | weeks green | worst wk | '
    'total R | $/month | uncapped $/month |')
say('|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
monthly_store = {}
for sname, fams in STACKS.items():
    dd = dedupe(c, fams)
    for R in RISKS:
        part = (R / dd.risk_per_share) * dd.next_entry / dd.five_min_dollar_vol
        keep = part <= 0.01
        surv = dd[keep]
        for sp in ('TRAIN', 'VAL', 'TEST'):
            t = book(surv[surv.split == sp], 'hold')
            tu = book(dd[dd.split == sp], 'hold')
            if len(t) < 10:
                continue
            st = stats(t, sp, W)
            nmo = t.mo.nunique()
            usd = t.net.sum() * R / max(nmo, 1)
            usd_u = tu.net.sum() * R / max(tu.mo.nunique(), 1)
            cells.append(dict(stack=sname, risk=R, kept=float(keep.mean()), usd_month=usd,
                              usd_month_uncapped=usd_u, **{k: v for k, v in st.items() if k != 'mix'}))
            say(f"| {sname} | {R} | {sp} | {keep.mean():.1%} | {st['n']} | {st['tpw']} | {st['meanR']:+.3f} | "
                f"{st['t']:.2f} | {st['wkR']:+.2f} | {st['green']:.2f} | {st['worst']:.1f} | "
                f"{t.net.sum():+.1f} | {usd:+,.0f} | {usd_u:+,.0f} |")
            if sname == 'S0' and R == 300:
                monthly_store[sp] = t
say('')
pd.DataFrame(cells).to_csv(f'{D}/cap_cells.csv', index=False)

# live-computable cap (prior four bars x1.25)
say('## The same cap made LIVE-COMPUTABLE (four PRIOR bars x1.25 — the fill bar\'s own volume is unknown when '
    'the order is sent), frozen stack S0, exit hold')
say('')
say('| risk $ | split | pop kept | n | mean net R | wkR | total R | $/month |')
say('|---:|---|---:|---:|---:|---:|---:|---:|')
dd = dedupe(c, STACKS['S0'])
prior = dd.prior4_dollar_vol.clip(lower=1.0) * 1.25
for R in RISKS:
    pc = (R / dd.risk_per_share) * dd.next_entry / prior
    surv = dd[pc <= 0.01]
    for sp in ('TRAIN', 'VAL', 'TEST'):
        t = book(surv[surv.split == sp], 'hold')
        if len(t) < 10:
            continue
        st = stats(t, sp, W)
        say(f"| {R} | {sp} | {(pc <= 0.01).mean():.1%} | {st['n']} | {st['meanR']:+.3f} | {st['wkR']:+.2f} | "
            f"{t.net.sum():+.1f} | {t.net.sum() * R / max(t.mo.nunique(), 1):+,.0f} |")
say('')

# spread x1.5 sensitivity on the frozen capped cell
say('## Sensitivity: spread x1.5 on the frozen capped cell (S0, hold, $300)')
say('')
RATIO = {'stop': 0.875, 'eod': 0.412, 'target': 0.875}
d2 = dedupe(c, STACKS['S0']).copy()
d2['net_hold'] = d2.next_rr_hold - 0.25 * (1.5 * d2.half) - (1.5 * d2.half) * d2.next_why_hold.map(RATIO).fillna(0.875)
part = (300 / d2.risk_per_share) * d2.next_entry / d2.five_min_dollar_vol
s2 = d2[part <= 0.01]
say('| split | n | mean net R x1.0 | mean net R x1.5 | $/month x1.0 | $/month x1.5 |')
say('|---|---:|---:|---:|---:|---:|')
for sp in ('TRAIN', 'VAL', 'TEST'):
    a = book(dedupe(c, STACKS['S0']).pipe(lambda x: x[(((300 / x.risk_per_share) * x.next_entry /
                                                       x.five_min_dollar_vol) <= 0.01) & (x.split == sp)]), 'hold')
    b = book(s2[s2.split == sp], 'hold')
    if len(a) < 10:
        continue
    say(f'| {sp} | {len(a)} | {a.net.mean():+.3f} | {b.net.mean():+.3f} | '
        f'{a.net.sum() * 300 / max(a.mo.nunique(), 1):+,.0f} | {b.net.sum() * 300 / max(b.mo.nunique(), 1):+,.0f} |')
say('')

# monthly $ of the frozen capped cell -> Part B input
allm = pd.concat([monthly_store[s] for s in monthly_store], ignore_index=True)
m = allm.groupby('mo').net.agg(['sum', 'count'])
m['usd'] = m['sum'] * 300
m.reset_index().to_csv(f'{D}/cap_monthly.csv', index=False)
say('## The frozen capped cell month by month (S0 = F6 alone, hold, $300 risk, 1% participation cap)')
say('')
say('| month | trades | net R | $ |')
say('|---|---:|---:|---:|')
for i, r in m.iterrows():
    say(f"| {i} | {int(r['count'])} | {r['sum']:+.1f} | {r['usd']:+,.0f} |")
say('')
say(f"total ${m.usd.sum():+,.0f} over {len(m)} months | mean ${m.usd.mean():+,.0f}/month | "
    f"months green {int((m.usd > 0).sum())}/{len(m)} | worst ${m.usd.min():+,.0f} ({m.usd.idxmin()})")
say('')
open(f'{D}/c_capacity.md', 'w').write('\n'.join(OUT) + '\n')
print('wrote', f'{D}/c_capacity.md', flush=True)
