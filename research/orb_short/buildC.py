#!/usr/bin/env python3
"""Cell C — chase-tolerant short entry. PREREG.md "Cell C".

Re-walks ONLY the rows sig.csv marked `gap_through` (the next open was below the
no-chase stop-LIMIT level L).  Under cell C the order is a STOP: fill = the next
bar's OPEN, capped at range_low*(1-100bps); below the cap = skip, not a chase.
Every other no-fill reason is unchanged by construction (same fill bar, same test).

Reads data/cache.db READ-ONLY.  Writes research/orb_short/sigC.csv + legsC.csv.
TEST (>= 2026-06-01) is never queried (sig.csv itself stops at 2026-05-31).
"""
import os
import sqlite3
import sys

import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/orb_short')
from build import walk                       # noqa: E402  (same exit spec)
from trading.orb_csv import read_orb_csv     # noqa: E402

D = f'{ROOT}/research/orb_short'
DB = f'file:{ROOT}/data/cache.db?mode=ro'
CAP_BPS = 100.0


def main():
    s = read_orb_csv(f'{D}/sig.csv')
    assert s.day.max() < '2026-06-01', 'TEST leaked'
    gt = s[s.nofill == 'gap_through']
    print(f'[cellC] signals {len(s)} · old fills {int((s.filled==1).sum())} '
          f'· gap_through {len(gt)}', flush=True)
    conn = sqlite3.connect(DB, uri=True)
    upd, nfill, ncap, nssr = {}, 0, 0, 0
    for k, (ds, g) in enumerate(gt.groupby('day')):
        syms = sorted(set(g.symbol))
        ph = ','.join('?' * len(syms))
        bars = pd.read_sql_query(
            f"SELECT symbol,timestamp,open,high,low,close FROM intraday_bars_1min "
            f"WHERE bar_date=? AND symbol IN ({ph}) ORDER BY symbol,timestamp",
            conn, params=[ds] + syms)
        if bars.empty:
            continue
        ts = pd.to_datetime(bars['timestamp'], utc=True, format='mixed').dt.tz_convert('America/New_York')
        bars['m'] = ts.dt.hour * 60 + ts.dt.minute
        bars = bars[(bars.m >= 575) & (bars.m <= 945)]
        by = {sym: b.reset_index(drop=True) for sym, b in bars.groupby('symbol', sort=False)}
        for r in g.itertuples():
            post = by.get(r.symbol)
            if post is None:
                continue
            j = post.index[post.m == int(r.trig_m)]
            if len(j) == 0 or int(j[0]) + 1 >= len(post):
                continue
            t = int(j[0])
            nb = post.iloc[t + 1]
            no = float(nb['open'])
            cap = r.range_low * (1 - CAP_BPS / 10000)
            if no < cap:
                ncap += 1
                continue                              # gap > 1% through = skip
            if int(r.ssr) == 1 and not (no > float(nb['low'])):
                nssr += 1
                continue
            if r.range_high - no <= 0:
                continue
            px, rsn, xi = walk(post, no, r.range_high, t + 1)
            upd[(r.day, r.symbol)] = dict(filled=1, nofill='', entry=no, exit=px, reason=rsn,
                                          entry_m=int(nb['m']), exit_m=int(post['m'].iat[xi]),
                                          R=r.range_high - no)
            nfill += 1
        if k % 60 == 0:
            print(f'  [{k}] {ds} new_fills={nfill}', flush=True)
    print(f'[cellC] gap_through -> new fills {nfill} · over-cap skips {ncap} · ssr skips {nssr}',
          flush=True)
    for col in ('filled', 'nofill', 'entry', 'exit', 'reason', 'entry_m', 'exit_m', 'R'):
        s[col] = [upd[(d, y)][col] if (d, y) in upd else v
                  for d, y, v in zip(s.day, s.symbol, s[col])]
    s['cellC_new'] = [int((d, y) in upd) for d, y in zip(s.day, s.symbol)]
    s.to_csv(f'{D}/sigC.csv', index=False)
    nf = s[s.cellC_new == 1]
    legs = pd.concat([nf[['day', 'symbol', 'entry_m']].rename(columns={'entry_m': 'm'}),
                      nf[['day', 'symbol', 'exit_m']].rename(columns={'exit_m': 'm'})])
    legs['m'] = legs.m.astype(int)
    legs.drop_duplicates().to_csv(f'{D}/legsC.csv', index=False)
    print(f'[cellC] fills now {int((s.filled==1).sum())}/{len(s)} · new legs {len(legs.drop_duplicates())}',
          flush=True)


if __name__ == '__main__':
    main()
