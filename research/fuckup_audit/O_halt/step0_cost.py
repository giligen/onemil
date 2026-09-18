"""Step 0a: price every request with metadata.get_cost BEFORE pulling anything."""
import os, sys
from dotenv import load_dotenv
load_dotenv('/home/ec2-user/onemil/.env')
import databento as db

key = os.environ.get('DATABENTO_API_KEY') or os.environ.get('DATABENTO_KEY')
assert key, "no databento key in .env"
c = db.Historical(key)

def cost(**kw):
    try:
        v = c.metadata.get_cost(**kw)
        return v
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}"

rows = []
rows.append(("status ALL 2018-05-01..2026-09-17", cost(dataset='XNAS.ITCH', schema='status',
             symbols='ALL_SYMBOLS', stype_in='raw_symbol', start='2018-05-01', end='2026-09-18')))
rows.append(("definition ALL 2018-05-01..2026-09-17", cost(dataset='XNAS.ITCH', schema='definition',
             symbols='ALL_SYMBOLS', stype_in='raw_symbol', start='2018-05-01', end='2026-09-18')))
rows.append(("status ALL 1 day 2026-09-15", cost(dataset='XNAS.ITCH', schema='status',
             symbols='ALL_SYMBOLS', stype_in='raw_symbol', start='2026-09-15', end='2026-09-16')))
rows.append(("definition ALL 1 day 2026-09-15", cost(dataset='XNAS.ITCH', schema='definition',
             symbols='ALL_SYMBOLS', stype_in='raw_symbol', start='2026-09-15', end='2026-09-16')))
rows.append(("status ALL 1 month 2026-08", cost(dataset='XNAS.ITCH', schema='status',
             symbols='ALL_SYMBOLS', stype_in='raw_symbol', start='2026-08-01', end='2026-09-01')))
rows.append(("definition ALL 1 month 2026-08", cost(dataset='XNAS.ITCH', schema='definition',
             symbols='ALL_SYMBOLS', stype_in='raw_symbol', start='2026-08-01', end='2026-09-01')))
# dataset range sanity
try:
    print("dataset range:", c.metadata.get_dataset_range(dataset='XNAS.ITCH'))
except Exception as e:
    print("range ERR", e)
for n, v in rows:
    print(f"{n:45s} {v}")
