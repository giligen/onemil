#!/usr/bin/env python3
"""frames16 ARM 1 — price every candidate pull with metadata.get_cost BEFORE pulling.

Nothing is fetched here. The printed table is the authorisation record for PREREG §cap.
"""
import os
import sys

import pandas as pd
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
load_dotenv(f'{ROOT}/.env')
import databento as db  # noqa: E402

D = f'{ROOT}/research/mature_method/frames16'
DAY = '2026-03-11'
SYMS = ['AAPL', 'FSLY', 'USAX']
ST = f'{DAY}T13:30:00+00:00'      # 09:30 ET
EN = f'{DAY}T20:00:00+00:00'      # 16:00 ET

CAND = [('EQUS.MINI', 'mbp-1'), ('EQUS.MINI', 'bbo-1s'), ('EQUS.MINI', 'tbbo'),
        ('XNAS.ITCH', 'mbp-1'), ('XNAS.ITCH', 'bbo-1s'),
        ('XNAS.BASIC', 'cmbp-1'), ('XNAS.BASIC', 'cbbo-1s'),
        ('XNYS.PILLAR', 'mbp-1'), ('XNYS.PILLAR', 'bbo-1s')]


def main():
    c = db.Historical(os.environ['DATABENTO_API_KEY'])
    rows = []
    for ds, sch in CAND:
        kw = dict(dataset=ds, symbols=SYMS, schema=sch, start=ST, end=EN, stype_in='raw_symbol')
        try:
            cost = float(c.metadata.get_cost(**kw))
            n = c.metadata.get_record_count(**kw)
        except Exception as e:
            rows.append(dict(dataset=ds, schema=sch, cost=float('nan'), records=-1, err=str(e)[:80]))
            print(f'{ds:14s} {sch:8s}  ERR {str(e)[:90]}', flush=True)
            continue
        rows.append(dict(dataset=ds, schema=sch, cost=cost, records=int(n), err=''))
        print(f'{ds:14s} {sch:8s}  ${cost:8.4f}   {int(n):>12,} records', flush=True)
    pd.DataFrame(rows).to_csv(f'{D}/price.csv', index=False)
    print(f'\nTOTAL if everything pulled: ${pd.DataFrame(rows).cost.sum():.4f}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
