"""Step 0b: one day of status (+definition) — decode actions/reasons, test symbol resolution."""
import os, sys
from dotenv import load_dotenv
load_dotenv('/home/ec2-user/onemil/.env')
import databento as db
import databento_dbn as dbn
import pandas as pd

c = db.Historical(os.environ['DATABENTO_API_KEY'])
OUT = '/home/ec2-user/onemil/research/fuckup_audit/O_halt/raw'

print("enums OK")





d = c.timeseries.get_range(dataset='XNAS.ITCH', schema='status', symbols='ALL_SYMBOLS',
                           stype_in='raw_symbol', start='2026-09-15', end='2026-09-16')
d.to_file(f'{OUT}/status_20260915.dbn.zst')
print("SYMBOL MAP available?", end=' ')
try:
    sm = d.symbology_map
    print("keys", len(sm), list(sm.items())[:3])
except Exception as e:
    print("ERR", e)

df = d.to_df(map_symbols=True)
print("rows", len(df), "cols", list(df.columns))
print(df.head(3).to_string())
if 'symbol' in df.columns:
    nn = df['symbol'].notna() & (df['symbol'].astype(str)!='') & (df['symbol'].astype(str)!='None')
    print("symbol resolved share:", round(nn.mean(),4), "distinct:", df.loc[nn,'symbol'].nunique())
print("=== action x reason counts ===")
print(df.groupby(['action','reason']).size().sort_values(ascending=False).head(30).to_string())
if 'is_trading' in df.columns:
    print(df.groupby(['action'])[['is_trading','is_quoting']].agg(['mean','size']).to_string())
print("=== ts_event hour histogram (UTC) ===")
print(df.index.hour.value_counts().sort_index().to_string())
