"""Ignition capcheck — universe + coverage ADD-ON for a date window (2026-09-05).

Same prefilter as build_universe.py (open >= 1.95, gap < 5.5%, high >= open*1.09,
volume*high >= $2M) from cache.db daily_bars, plus the coverage_check.py bar
count, written to a separate file so the frozen 19-month universe files are
never rewritten. Used to extend the capsim/resting book to the live window.

Usage: python3 build_universe_addon.py START END OUT_CSV
"""
import sqlite3, sys
import pandas as pd
sys.stdout.reconfigure(line_buffering=True)
ROOT = '/home/ec2-user/onemil'
START, END, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
conn = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
lb = (pd.Timestamp(START) - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
df = pd.read_sql("select symbol, bar_date, open, high, low, close, volume from daily_bars "
                 "where bar_date >= ? and bar_date <= ? order by symbol, bar_date", conn, params=[lb, END])
df['prev_close'] = df.groupby('symbol')['close'].shift(1)
df = df[(df['bar_date'] >= START) & (df['bar_date'] <= END)]
gap = (df['open'] - df['prev_close']) / df['prev_close'] * 100.0
keep = df[(df['open'] >= 1.95) & (df['prev_close'].isna() | (gap < 5.5))
          & (df['high'] >= df['open'] * 1.09) & (df['volume'] * df['high'] >= 2_000_000)]
keep = keep[['symbol', 'bar_date', 'open', 'high', 'close', 'volume', 'prev_close']].copy()
conn.execute("ATTACH DATABASE ':memory:' AS mem")
conn.execute("CREATE TABLE mem.uni (symbol TEXT, bar_date TEXT)")
conn.executemany("INSERT INTO mem.uni VALUES (?,?)", list(zip(keep['symbol'], keep['bar_date'])))
cov = pd.read_sql("select i.symbol, i.bar_date, count(*) as nbars from intraday_bars_1min i "
                  "join mem.uni x on i.symbol = x.symbol and i.bar_date = x.bar_date "
                  "where i.bar_date >= ? and i.bar_date <= ? group by 1,2", conn, params=[START, END])
conn.close()
out = keep.merge(cov, on=['symbol', 'bar_date'], how='left')
out['nbars'] = out['nbars'].fillna(0).astype(int)
out['covered'] = out['nbars'] >= 100
out.to_csv(OUT, index=False)
print(f"{START}..{END}: {len(out):,} candidates, covered {int(out['covered'].sum()):,} "
      f"({out['covered'].mean()*100:.1f}%), zero-bar {int((out['nbars']==0).sum()):,} -> {OUT}")
print(out.groupby('bar_date')['covered'].agg(['size', 'sum']).T.to_string())
