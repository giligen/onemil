#!/usr/bin/env python3
"""frames16 ARM 2 — price the lambda pull over the REAL B2 day/symbol/window structure.

Nothing is fetched. PREREG §3 makes arm 2 conditional on arm 1 AND on the priced pull fitting the
remaining budget; this is that price, measured rather than extrapolated, on every candidate
instrument. Arm 1 has disqualified EQUS.MINI (§ARM 1), so the instruments priced here are the ones
that reproduce the CKS relation: XNAS.ITCH `bbo-1s` (R2 0.37-0.41) and XNAS.ITCH `mbp-1` (0.52-0.65).

Window per session: [min(break_m) - FIT_MIN, max(break_m) + 1] over that session's B2 symbols —
arm O's window widened at the front by the 30 minutes the per-name-day beta fit needs.
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
SAMPLE_EVERY = 4          # price every 4th session exactly, then scale by the session count


def utc(day, minute):
    return (pd.Timestamp(day).tz_localize('America/New_York')
            + pd.Timedelta(minutes=int(minute))).tz_convert('UTC')


def main():
    sig = pd.read_pickle(f'{ROOT}/research/mature_method/hod_filter_stack/b2.pkl')
    sig = sig[sig.split.isin(('TRAIN', 'VAL'))][['day', 'symbol', 'break_m']].drop_duplicates()
    days = sorted(sig.day.unique())
    print(f'B2 TRAIN+VAL: {len(sig):,} signals · {len(days)} sessions · '
          f'{sig.symbol.nunique():,} symbols', flush=True)
    c = db.Historical(os.environ['DATABENTO_API_KEY'])
    rows = []
    for ds, sch in (('XNAS.ITCH', 'bbo-1s'), ('XNAS.ITCH', 'mbp-1')):
        tot, n = 0.0, 0
        for i, day in enumerate(days):
            if i % SAMPLE_EVERY:
                continue
            g = sig[sig.day == day]
            kw = dict(dataset=ds, symbols=sorted(g.symbol.unique()), schema=sch,
                      start=utc(day, int(g.break_m.min()) - FIT_MIN).isoformat(),
                      end=utc(day, int(g.break_m.max()) + 1).isoformat(), stype_in='raw_symbol')
            try:
                tot += float(c.metadata.get_cost(**kw)); n += 1
            except Exception as e:
                print(f'  {day} ERR {str(e)[:70]}', flush=True)
        est = tot / max(n, 1) * len(days)
        rows.append(dict(dataset=ds, schema=sch, sessions_priced=n, sampled_cost=tot,
                         est_full_cost=est))
        print(f'{ds}/{sch}: {n} sessions priced = ${tot:.2f}  ->  ESTIMATED FULL B2 '
              f'({len(days)} sessions) = ${est:.2f}', flush=True)
    pd.DataFrame(rows).to_csv(f'{D}/price_arm2.csv', index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
