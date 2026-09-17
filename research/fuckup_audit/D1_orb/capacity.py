"""Capital actually deployed per slot count + participation at 1x/3x sizing."""
import os, sys
import pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil'); os.chdir('/home/ec2-user/onemil')
from trading.orb_csv import read_orb_csv

D = 'research/fuckup_audit/D1_orb'
ACCT = {3: 10000, 4: 13333, 6: 20000, 8: 26667, 12: 40000}
print(f"{'cfg':>10} {'picks':>6} {'fills':>6} {'pnl':>9} {'$/mo':>7} "
      f"{'budget':>7} {'%/mo':>6} {'maxday':>7} {'p95day':>7} {'meanday':>8} "
      f"{'maxcap$':>8} {'MDD':>8} {'red':>4}")
for n in (3, 4, 6, 8, 12):
    b = read_orb_csv(f'{D}/book_n{n}_q1on.csv')
    b['date'] = pd.to_datetime(b['date']).dt.strftime('%Y-%m-%d')
    per_day = b.groupby('date').size()
    daily = b.groupby('date')['_sized_pnl'].sum()
    cum = daily.cumsum()
    mdd = float((cum - cum.cummax()).min())
    monthly = b.groupby(b['date'].str[:7])['_sized_pnl'].sum()
    pnl = float(b['_sized_pnl'].sum())
    print(f"{'N='+str(n):>10} {len(b):6d} {int((b['entered'].astype(float)!=0).sum()):6d} "
          f"{pnl:9,.0f} {pnl/21:7,.0f} {ACCT[n]:7,d} {pnl/21/ACCT[n]*100:6.2f} "
          f"{per_day.max():7d} {per_day.quantile(.95):7.0f} {per_day.mean():8.2f} "
          f"{per_day.max()*3333:8,.0f} {mdd:8,.0f} {int((monthly<0).sum()):4d}")

b = read_orb_csv(f'{D}/book_n4_q1on.csv')
b['rd'] = b['range_total_volume'].astype(float) * b['entry_price'].astype(float)
for k, lbl in ((1, '$375 risk / $3.3K pos (stage)'),
               (3, '$1,125 risk / $10K pos (3x)'),
               (5, '$1,875 risk / $16.7K pos (5x)')):
    p = 100 * (b['_rp_position'].astype(float) * k) / b['rd']
    print(f"{lbl:32s} participation of the 5-min range $vol: "
          f"median {p.median():.2f}%  p75 {p.quantile(.75):.2f}%  "
          f"p90 {p.quantile(.9):.2f}%  >5% on {int((p>5).sum())}/{len(b)} picks "
          f"(>1% on {int((p>1).sum())})")
