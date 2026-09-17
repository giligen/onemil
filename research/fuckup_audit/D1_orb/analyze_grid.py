"""TASK B — summarise the slot dose-response books written by run_grid.sh."""
from __future__ import annotations
import glob, os, sys
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
from trading.orb_csv import read_orb_csv

D = 'research/fuckup_audit/D1_orb'
pd.set_option('display.width', 250)


def split_of(d):
    s = str(d)[:7]
    if s < '2026-01':
        return '2025'
    if s <= '2026-05':
        return '2026H1a'
    return '2026-06+'


def mdd(daily_pnl: pd.Series) -> float:
    cum = daily_pnl.cumsum()
    return float((cum - cum.cummax()).min())


def load(tag):
    p = f'{D}/book_{tag}.csv'
    if not os.path.exists(p):
        return None
    b = read_orb_csv(p)
    b['date'] = pd.to_datetime(b['date']).dt.strftime('%Y-%m-%d')
    b['month'] = b['date'].str[:7]
    b['split'] = b['date'].apply(split_of)
    b['fill'] = (b['entered'].astype(float) != 0)
    b['tag'] = tag
    return b


def summarise(b, label):
    daily = b.groupby('date')['_sized_pnl'].sum()
    monthly = b.groupby('month')['_sized_pnl'].sum()
    f = b[b['fill']]
    return {
        'book': label, 'picks': len(b), 'fills': int(b['fill'].sum()),
        'fill_pct': 100 * b['fill'].mean(),
        'pnl': float(b['_sized_pnl'].sum()),
        'wr_fills_pct': 100 * float((f['_sized_pnl'] > 0).mean()) if len(f) else 0.0,
        'mean_$_fill': float(f['_sized_pnl'].mean()) if len(f) else 0.0,
        'mean_$_pick': float(b['_sized_pnl'].mean()),
        'mdd': mdd(daily),
        'worst_month': float(monthly.min()),
        'red_months': int((monthly < 0).sum()),
        'n_months': int(monthly.size),
        'pnl_per_month': float(b['_sized_pnl'].sum()) / max(monthly.size, 1),
    }


tags = [f'n{n}_{q}' for n in (3, 4, 6, 8, 12) for q in ('q1on', 'q1off')]
tags += ['n4_pool2x', 'n4_poolall', 'n12_pool2x', 'n12_poolall']
books = {t: load(t) for t in tags}
books = {t: b for t, b in books.items() if b is not None}
print(f"loaded books: {list(books)}")

rows = []
for t, b in books.items():
    rows.append(summarise(b, t))
    for sp in ('2025', '2026H1a', '2026-06+'):
        sb = b[b['split'] == sp]
        if len(sb):
            rows.append(summarise(sb, f'{t} | {sp}'))
tab = pd.DataFrame(rows)
tab.to_csv(f'{D}/taskB_summary.csv', index=False)
print("\n=== TASK B — per-config, whole window and per split ===")
print(tab.to_string(index=False, float_format=lambda x: f'{x:,.1f}'))

# ---- marginal slots (books nest: greedy rank order, per-pick vetoes) ----
print("\n=== marginal P&L of the added slots (q1on) ===")
for q in ('q1on', 'q1off'):
    prev = None
    print(f"-- {q}")
    for n in (3, 4, 6, 8, 12):
        t = f'n{n}_{q}'
        b = books.get(t)
        if b is None:
            continue
        key = set(zip(b['symbol'], b['date']))
        if prev is not None:
            pk, pb, pn = prev
            add = b[~b.apply(lambda r: (r['symbol'], r['date']) in pk, axis=1)]
            miss = pk - key
            f = add[add['fill']]
            print(f"  slots {pn+1}-{n}: +{len(add)} picks (+{int(add['fill'].sum())} fills) "
                  f"${add['_sized_pnl'].sum():+,.0f}  mean/pick ${add['_sized_pnl'].mean() if len(add) else 0:+,.0f}  "
                  f"WR {100*(f['_sized_pnl']>0).mean() if len(f) else 0:.1f}%  "
                  f"| by split " + " ".join(
                      f"{sp}:{add[add['split']==sp]['_sized_pnl'].sum():+,.0f}"
                      for sp in ('2025', '2026H1a', '2026-06+'))
                  + (f"  [WARN {len(miss)} picks dropped out]" if miss else ""))
        prev = (key, b, n)

# ---- per month, the four headline configs ----
print("\n=== per month (picks / P&L) ===")
mrows = {}
for t in ('n3_q1on', 'n4_q1on', 'n6_q1on', 'n8_q1on', 'n12_q1on'):
    b = books.get(t)
    if b is None:
        continue
    g = b.groupby('month').agg(p=('_sized_pnl', 'size'), pnl=('_sized_pnl', 'sum'))
    mrows[t] = g['pnl']
    mrows[t + '_n'] = g['p']
mt = pd.DataFrame(mrows).fillna(0)
mt.to_csv(f'{D}/taskB_monthly.csv')
print(mt.to_string(float_format=lambda x: f'{x:,.0f}'))

# ---- participation / capacity ----
print("\n=== participation at stage sizing (per pick) ===")
b = books.get('n4_q1on')
if b is not None:
    b = b.copy()
    b['range_dollar'] = b['range_total_volume'].astype(float) * b['entry_price'].astype(float)
    b['day_dollar'] = b['avg_daily_volume_20d'].astype(float) * b['entry_price'].astype(float)
    b['part_range'] = 100 * b['_rp_position'].astype(float) / b['range_dollar']
    b['part_day'] = 100 * b['_rp_position'].astype(float) / b['day_dollar']
    print(b[['_rp_position', 'range_dollar', 'day_dollar', 'part_range', 'part_day']]
          .describe(percentiles=[.1, .25, .5, .75, .9])
          .to_string(float_format=lambda x: f'{x:,.3f}'))
    for k in (1, 3, 5, 10):
        pr = b['part_range'] * k
        print(f"  at {k}x stage sizing: median range-participation {pr.median():.2f}%  "
              f"p90 {pr.quantile(.9):.2f}%  picks over 5% of the 5-min range: "
              f"{int((pr > 5).sum())}/{len(b)}")
print(f"\nWrote {D}/taskB_summary.csv, {D}/taskB_monthly.csv")
