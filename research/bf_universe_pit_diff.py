"""Bull flag — point-in-time universe survivorship diff (2026-09-05).

Applies the shared Stage-1 mover screen (trading/bf_selection.mover_day_qualifies:
max(gap-up vs prev close, in-day range) >= 10%, price band, dollar volume) to the
Databento EQUS.SUMMARY daily bars (every US equity that existed on each date) and
to cache.db daily_bars, and reports per month how many mover symbol-days the
cache-based BT could never see. Writes research/bf_universe_pit_missing.csv.
"""
import sqlite3, sys
import pandas as pd, yaml
sys.path.insert(0, '/home/ec2-user/onemil')
from trading.bf_selection import mover_day_qualifies
sys.stdout.reconfigure(line_buffering=True)
ROOT = '/home/ec2-user/onemil'
START, END = '2025-01-01', '2026-09-04'
sc = yaml.safe_load(open(f'{ROOT}/config.yaml')).get('scanner', {})
PMIN, PMAX, MINDV = float(sc.get('price_min', 0) or 0), float(sc.get('price_max', 20.0)), float(sc.get('min_dollar_volume', 0) or 0)
print(f"screen: threshold 10%, price_min {PMIN}, price_max {PMAX}, min_dollar_volume {MINDV}")

def movers(d):
    d = d.sort_values(['symbol', 'bar_date']).copy()
    d['prev_close'] = d.groupby('symbol')['close'].shift(1)
    d = d[(d['bar_date'] >= START) & (d['bar_date'] <= END) & (d['low'] > 0)]
    q = [mover_day_qualifies(high=h, low=l, prev_close=(pc if pc == pc else None), price_ref=c, volume=v,
                             threshold_pct=0.10, price_min=PMIN, price_max=PMAX, min_dollar_volume=MINDV).qualifies
         for h, l, pc, c, v in zip(d['high'], d['low'], d['prev_close'], d['close'], d['volume'])]
    return d[pd.Series(q, index=d.index)]

daily = pd.read_parquet(f'{ROOT}/data/research/databento/equs_daily_2025_2026.parquet')
daily = daily[daily['symbol'].str.match(r'^[A-Z]{1,5}(\.[A-Z])?$', na=False) & (daily['symbol'] != 'ZVZZT') & (daily['volume'] > 0)]
pit = movers(daily); del daily
print(f"PIT BF movers: {len(pit):,} symbol-days, {pit['symbol'].nunique():,} symbols")
conn = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
cache_syms = set(pd.read_sql("select distinct symbol from daily_bars", conn)['symbol'])
# cache side, month-chunked to stay under the memory cap
cb_keys = set()
for mp in pd.period_range('2025-01', '2026-09', freq='M'):
    lb = (mp.start_time - pd.Timedelta(days=10)).strftime('%Y-%m-%d'); me = mp.end_time.strftime('%Y-%m-%d')
    df = pd.read_sql("select symbol, bar_date, high, low, close, volume from daily_bars where bar_date >= ? and bar_date <= ?", conn, params=[lb, me])
    mv = movers(df); mv = mv[mv['bar_date'].str[:7] == str(mp)]
    cb_keys |= set(zip(mv['symbol'], mv['bar_date']))
conn.close()
print(f"cache-based BF movers: {len(cb_keys):,}")
pit['in_cache'] = [(s, d) in cb_keys for s, d in zip(pit['symbol'], pit['bar_date'])]
pit['sym_in_cache'] = pit['symbol'].isin(cache_syms); pit['month'] = pit['bar_date'].str[:7]
miss = pit[~pit['in_cache']]
tab = pit.groupby('month').agg(pit=('symbol', 'size'), in_cache=('in_cache', 'sum'))
tab['missing'] = tab['pit'] - tab['in_cache']
tab['missing_dead_symbol'] = miss.groupby('month')['sym_in_cache'].apply(lambda s: int((~s).sum())).reindex(tab.index).fillna(0).astype(int)
tab['missing_pct'] = (tab['missing'] / tab['pit'] * 100).round(1)
print(tab.to_string())
print(f"\nTOTAL: PIT {len(pit):,}; cache-based universe missed {len(miss):,} ({len(miss)/len(pit)*100:.1f}%), {int((~miss['sym_in_cache']).sum()):,} on symbols absent from daily_bars.")
print("top missing symbols:", miss['symbol'].value_counts().head(15).to_dict())
miss.drop(columns=['in_cache']).to_csv(f'{ROOT}/research/bf_universe_pit_missing.csv', index=False)
