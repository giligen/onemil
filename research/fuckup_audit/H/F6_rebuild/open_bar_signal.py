"""How often does the 09:30 bar itself already reach the reclaim level?

Point-lookup of the 09:30 ET bar for every red-open candidate day, so the
'signal on the 09:30 bar' share can be reported without a full tape walk.
"""
import os
import sys
import sqlite3
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scan import utc_offset_min  # noqa: E402

sip = sqlite3.connect('file:research/bf_zero/bars_sip.db?mode=ro', uri=True)
cache = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)

pre = pd.read_csv(os.path.join(HERE, 'prefilter_all.csv'), keep_default_na=False, na_values=[''])
n = dict(rows=0, nobar=0, red=0, hi_ge_level=0, red_and_hi=0)
for r in pre.itertuples(index=False):
    off = utc_offset_min(r.bar_date)
    um = 570 + off
    ts = '%sT%02d:%02d:00+00:00' % (r.bar_date, um // 60, um % 60)
    row = sip.execute('select o,h from bars where symbol=? and day=? and t=?',
                      (r.symbol, r.bar_date, ts)).fetchone()
    if row is None:
        row = cache.execute('select open,high from intraday_bars_1min '
                            'where symbol=? and timestamp=?', (r.symbol, ts)).fetchone()
    n['rows'] += 1
    if row is None:
        n['nobar'] += 1
        continue
    o, h = row
    red = o < r.prev_close
    if red:
        n['red'] += 1
    if h >= r.level:
        n['hi_ge_level'] += 1
        if red:
            n['red_and_hi'] += 1
    if n['rows'] % 20000 == 0:
        print(n); sys.stdout.flush()
print('FINAL', n)
