"""daily_bars coverage + Databento-vs-Alpaca daily volume basis. Read-only."""
import os, sqlite3, sys
os.environ['OMP_NUM_THREADS'] = '1'
import pyarrow as pa
pa.set_cpu_count(1); pa.set_io_thread_count(1)
import pyarrow.parquet as pq, pandas as pd, numpy as np
os.chdir('/home/ec2-user/onemil')
S = '/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad'
c = sqlite3.connect('file:data/cache.db?mode=ro', uri=True, timeout=30)
q = """with d as (select symbol, bar_date, volume, close, row_number() over (partition by symbol order by bar_date desc) rn
 from daily_bars where bar_date >= date('now','-45 days'))
 select symbol, avg(volume) adv, count(*) cnt, max(case when rn=1 then close end) lc, max(bar_date) md from d where rn<=20 group by symbol"""
L = pd.DataFrame(c.execute(q).fetchall(), columns=['symbol', 'adv', 'cnt', 'lc', 'md'])
big = L[L.lc > 50]; print('daily_bars symbols last close>50:', len(big), '| max bar_date dist', big.md.value_counts().head(4).to_dict(), '| adv>=100K & cnt>=10:', int(((big.adv >= 1e5) & (big.cnt >= 10)).sum()))
miss = pd.DataFrame(c.execute("select a.symbol, a.close, a.volume from daily_bars a where a.bar_date='2026-09-11' and a.symbol not in (select symbol from daily_bars where bar_date='2026-09-14')").fetchall(), columns=['symbol', 'close', 'volume'])
print('415 unrefreshed (have 09-11, no 09-14): close dist', miss.close.describe()[['min', '25%', '50%', '75%', 'max']].round(2).to_dict(), '| close>=17', int((miss.close >= 17).sum()), '| vol>=100K', int((miss.volume >= 1e5).sum()))
uni = pd.DataFrame(c.execute("select symbol, price_close, avg_volume_daily from universe where active=1").fetchall(), columns=['symbol', 'pc', 'av'])
dbs = set(L.symbol); print('universe-table-only symbols (not in daily_bars last45):', int((~uni.symbol.isin(dbs)).sum()), 'of', len(uni), '| of which av>=100K', int(((~uni.symbol.isin(dbs)) & (uni.av >= 1e5)).sum()))
J = pd.read_csv(f'{S}/spec20_join.csv', dtype={'symbol': str})
t = J[J.day >= '2026-06-01']; absent = sorted(set(t.symbol) - dbs - set(uni.symbol))
print('TEST spec (level>=20) symbols:', t.symbol.nunique(), '| absent from daily_bars(45d)+universe today:', len(absent), absent[:25])
print('TEST signals on absent symbols:', int(t.symbol.isin(absent).sum()), 'meanR', round(t.rr[t.symbol.isin(absent)].mean(), 3))
print('spec level>=20 by level band: ', J.groupby(pd.cut(J.level, [20, 50, 1e6])).rr.agg(['count', 'mean']).round(3).to_dict('index'))
x = J[J.day >= '2026-01-01']; print('2026 nprev<10 signals', int((x.nprev < 10).sum()), 'meanR', round(x.rr[x.nprev < 10].mean(), 3), '| nprev 10-19', int(((x.nprev >= 10) & (x.nprev < 20)).sum()), 'meanR', round(x.rr[(x.nprev >= 10) & (x.nprev < 20)].mean(), 3))
# --- volume basis: Alpaca daily_bars vs Databento parquet, Aug 3 - Sep 4 2026, sample 400 streamed symbols
st = L[(L.adv >= 1e5) & (L.cnt >= 10) & (L.lc >= 17)]; rs = np.random.RandomState(1); syms = sorted(rs.choice(st.symbol.values, 400, replace=False))
A = pd.DataFrame(c.execute(f"select symbol, bar_date, open, close, volume from daily_bars where bar_date between '2026-08-03' and '2026-09-04' and symbol in ({','.join('?'*len(syms))})", syms).fetchall(), columns=['symbol', 'bar_date', 'a_open', 'a_close', 'a_vol'])
D = pq.read_table('data/research/databento/equs_daily_2025_2026.parquet', columns=['symbol', 'bar_date', 'open', 'close', 'volume'], filters=[('symbol', 'in', syms), ('bar_date', '>=', '2026-08-03')], use_threads=False).to_pandas()
D['bar_date'] = D.bar_date.astype(str).str[:10]; D = D[D.bar_date <= '2026-09-04'].rename(columns={'open': 'd_open', 'close': 'd_close', 'volume': 'd_vol'})
M = A.merge(D, on=['symbol', 'bar_date'], how='outer', indicator=True)
print('volume basis rows', len(M), M._merge.value_counts().to_dict())
B = M[M._merge == 'both']; r = B.a_vol / B.d_vol.replace(0, np.nan)
print('Alpaca/Databento daily volume ratio: median', round(r.median(), 4), 'p10', round(r.quantile(.1), 4), 'p90', round(r.quantile(.9), 4), '| |ratio-1|>2%:', round((abs(r - 1) > 0.02).mean() * 100, 1), '% | >5%:', round((abs(r - 1) > 0.05).mean() * 100, 1), '%')
print('close equal (1c):', round((abs(B.a_close - B.d_close) <= 0.011).mean() * 100, 1), '% | open equal (1c):', round((abs(B.a_open - B.d_open) <= 0.011).mean() * 100, 1), '%')
# per-symbol ADV20 as of 2026-09-04 both ways
g = B.groupby('symbol'); adv_a = g.a_vol.mean(); adv_d = g.d_vol.mean(); rr = (adv_a / adv_d)
print('per-symbol ADV (Aug3-Sep4) Alpaca/Databento: median', round(rr.median(), 4), 'p5', round(rr.quantile(.05), 4), 'p95', round(rr.quantile(.95), 4))
