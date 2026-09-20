#!/usr/bin/env python3
"""frames16 ARM 1 step 1 — pull the nine priced instruments and KEEP THE RAW ON DISK.

`hod_filter_stack/ofi.py` aggregated its quotes on the fly and never stored them, which is why this
calibration has to re-pull. Every file lands in `frames16/raw/` as a DBN store.

Priced first in `price.py` (`price.csv`): $0.4726 for all nine, inside the pass's $12 cap.
Each pull re-prices with `metadata.get_cost` immediately before it is made and refuses to exceed
the running cap.
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
RAW = f'{D}/raw'
CAP = 12.00
DAY = os.environ.get('F16_DAY', '2026-03-11')
SYMS = ['AAPL', 'FSLY', 'USAX']
ST, EN = f'{DAY}T13:30:00+00:00', f'{DAY}T20:00:00+00:00'

CAND = [('EQUS.MINI', 'mbp-1'), ('EQUS.MINI', 'bbo-1s'), ('EQUS.MINI', 'tbbo'),
        ('XNAS.ITCH', 'mbp-1'), ('XNAS.ITCH', 'bbo-1s'),
        ('XNAS.BASIC', 'cmbp-1'), ('XNAS.BASIC', 'cbbo-1s'),
        ('XNYS.PILLAR', 'mbp-1'), ('XNYS.PILLAR', 'bbo-1s')]


def main():
    os.makedirs(RAW, exist_ok=True)
    c = db.Historical(os.environ['DATABENTO_API_KEY'])
    spent = 0.0
    log = []
    for ds, sch in CAND:
        path = f'{RAW}/{ds}_{sch}_{DAY}.dbn.zst'
        kw = dict(dataset=ds, symbols=SYMS, schema=sch, start=ST, end=EN, stype_in='raw_symbol')
        cost = float(c.metadata.get_cost(**kw))
        if os.path.exists(path):
            print(f'{ds:14s} {sch:8s}  cached ({os.path.getsize(path)/1e6:.1f} MB)', flush=True)
            log.append(dict(dataset=ds, schema=sch, cost=0.0, cached=True, path=path))
            continue
        if spent + cost > CAP:
            print(f'BUDGET STOP at {ds}/{sch}: ${spent:.4f} + ${cost:.4f} > ${CAP}', flush=True)
            break
        data = c.timeseries.get_range(**kw)
        data.to_file(path)
        spent += cost
        print(f'{ds:14s} {sch:8s}  ${cost:.4f}  -> {path} ({os.path.getsize(path)/1e6:.1f} MB) '
              f'| spent ${spent:.4f}', flush=True)
        log.append(dict(dataset=ds, schema=sch, cost=cost, cached=False, path=path))
    pd.DataFrame(log).to_csv(f'{D}/pull_log.csv', index=False)
    print(f'\nARM1 PULL DONE — spent ${spent:.4f} of ${CAP:.2f}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
