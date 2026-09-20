#!/usr/bin/env python3
"""frames16 ARM 2 — re-price the lambda pull with BUCKETED windows (nothing fetched).

`price2.py` priced arm O's shape: one window per session spanning every symbol's break. Most of
that rectangle is waste — a 09:40 break and a 13:50 break on the same day force a 4-hour window on
both symbols. Bucketing the day's symbols by break minute collapses the waste.
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
FIT_MIN = 35
BUCKET = 30               # minutes of break-time grouped into one request
SAMPLE_EVERY = 6


def utc(day, minute):
    return (pd.Timestamp(day).tz_localize('America/New_York')
            + pd.Timedelta(minutes=int(minute))).tz_convert('UTC')


def buckets(g):
    g = g.copy()
    g['bk'] = (g.break_m // BUCKET).astype(int)
    for bk, gg in g.groupby('bk'):
        yield sorted(gg.symbol.unique()), int(gg.break_m.min()), int(gg.break_m.max())


def main():
    sig = pd.read_pickle(f'{ROOT}/research/mature_method/hod_filter_stack/b2.pkl')
    sig = sig[sig.split.isin(('TRAIN', 'VAL'))][['day', 'symbol', 'break_m']].drop_duplicates()
    days = sorted(sig.day.unique())
    c = db.Historical(os.environ['DATABENTO_API_KEY'])
    rows = []
    for ds, sch in (('XNAS.ITCH', 'bbo-1s'), ('XNAS.ITCH', 'mbp-1')):
        tot, n, nreq = 0.0, 0, 0
        for i, day in enumerate(days):
            if i % SAMPLE_EVERY:
                continue
            n += 1
            for syms, b0, b1 in buckets(sig[sig.day == day]):
                nreq += 1
                try:
                    tot += float(c.metadata.get_cost(
                        dataset=ds, symbols=syms, schema=sch, stype_in='raw_symbol',
                        start=utc(day, b0 - FIT_MIN).isoformat(),
                        end=utc(day, b1 + 1).isoformat()))
                except Exception as e:
                    print(f'  {day} ERR {str(e)[:60]}', flush=True)
        est = tot / max(n, 1) * len(days)
        rows.append(dict(dataset=ds, schema=sch, mode=f'bucket{BUCKET}', sessions_priced=n,
                         requests=nreq, sampled_cost=tot, est_full_cost=est,
                         est_requests_full=int(nreq / max(n, 1) * len(days))))
        print(f'{ds}/{sch} bucket{BUCKET}: {n} sessions / {nreq} requests = ${tot:.3f}  ->  '
              f'FULL B2 ({len(days)} sessions, ~{int(nreq/max(n,1)*len(days))} requests) '
              f'= ${est:.2f}', flush=True)
    pd.DataFrame(rows).to_csv(f'{D}/price_arm2_bucket.csv', index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
