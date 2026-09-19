#!/usr/bin/env python3
"""hod_frames5 — PRE-PREREG sizing + parity pass.  NO CELL IS SCORED HERE.

Three things only, all of which must be known BEFORE the cells can be declared:
  1. the reproduction gate (B2 TRAIN -$17,346 / VAL +$893, and the shipped 12/4 book),
  2. the LIVE-vs-STUDY population parity check (F18 sub-question 2: what floor does the dry run
     actually admit?),
  3. the SIZE of each declared F18 population rung (how many pre-book signals, how many NBBO
     quote-minutes a measured cost would need) — the fetch budget.
No P&L of any non-reference cell is printed.
"""
import sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames5')
from common5 import ROOT, D5, S, S2, SPLITS, load_breaks4, sigset5, book_ranked, admit  # noqa: E402

pd.set_option('display.width', 200)

print('== 0. LOAD ==', flush=True)
br = load_breaks4()
S.build_impute(S2.load_pop())          # the cost model, EXACTLY as passes 3 and 4 built it
FIRST = admit(br, pd.Series(True, index=br.index))     # the first break of every symbol-day
print(f'first-break rows {len(FIRST)}', flush=True)
print(f'IMPUTE cells: {sorted({k[0] for k in S.IMPUTE})}', flush=True)

print('== 1. REPRODUCTION GATE ==', flush=True)
b2 = book_ranked(sigset5(FIRST), 12, 4)
for sp in SPLITS:
    w = S.week_stats(b2, sp)
    print(f'  B2 {sp:5s} n {w["n"]:5d} {w["per_wk"]:5.1f}/wk gross {w["gross"]:+.3f} '
          f'net {w["net"]:+.3f} green {w["green"]:.1f}% total ${w["total"]:+,.0f}', flush=True)
print('  reference (hod_frames4 §0): TRAIN 1,622 / 30.6 / -0.039 / -0.107 / 32.1% / -$17,346  |  '
      'VAL 706 / 30.7 / +0.083 / +0.013 / 43.5% / +$893', flush=True)

print('\n== 2. LIVE-vs-STUDY PARITY: what does the dry run admit? ==', flush=True)
import yaml
cfg = yaml.safe_load(open(f'{ROOT}/config.yaml'))['hod_break']
for k in ('min_price', 'universe_min_prev_close', 'max_spread_bps', 'max_spread_frac_r',
          'max_per_day', 'max_concurrent', 'last_entry_minute', 'cap', 'min_dist_open_pct',
          'rv_lo', 'rv_hi', 'min_r_pct', 'consol_bars', 'consol_pct', 'risk_usd', 'dry_run'):
    print(f'   config.yaml hod_break.{k:24s} = {cfg.get(k)}')
# the STUDY floor is on next_open; the ENGINE floor is on the break LEVEL (hod_break_engine:665)
allp = sigset5(FIRST, min_price=0.0)
lvl20 = allp[allp.level >= 20.0]
no20 = allp[allp.next_open >= 20.0]
both = set(zip(lvl20.day, lvl20.symbol)) & set(zip(no20.day, no20.symbol))
print(f'   study floor (next_open >= 20): {len(no20)} pre-book signals')
print(f'   engine floor (level    >= 20): {len(lvl20)} pre-book signals')
print(f'   overlap {len(both)}  |  in engine-not-study {len(lvl20) - len(both)}  |  '
      f'in study-not-engine {len(no20) - len(both)}')

print('\n== 3. F18 POPULATION SIZES (fetch budget) ==', flush=True)
rows = []
for floor in (5.0, 10.0, 20.0, 30.0, 50.0):
    for fr, bps in ((0.15, 100.0), (0.08, 100.0), (0.25, 100.0), (0.40, 100.0), (None, None)):
        s = sigset5(FIRST, min_price=floor, max_frac_r=fr, max_bps=bps)
        b = book_ranked(s, 12, 4)
        rows.append(dict(floor=floor, frac_r=fr, bps=bps, pre=len(s), booked=len(b),
                         imp_pre=float(s.imputed.mean() * 100) if len(s) else np.nan,
                         imp_book=float(b.imputed.mean() * 100) if len(b) else np.nan,
                         book_unmeasured=int(b.imputed.sum())))
        print(f'   floor ${floor:5.0f} frac_r {str(fr):5s} bps {str(bps):6s} -> pre-book {len(s):7d} '
              f'booked {len(b):6d}  imputed(book) {rows[-1]["imp_book"]:5.1f}%  '
              f'unmeasured booked rows {rows[-1]["book_unmeasured"]:6d}', flush=True)
pd.DataFrame(rows).to_csv(f'{D5}/popsizes.csv', index=False)

print('\n== 4. F17 HORIZON FEASIBILITY: daily-bar coverage for the booked set ==', flush=True)
import sqlite3
con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
syms = sorted(b2.symbol.unique())
q = pd.read_sql('select symbol, count(*) n, min(bar_date) a, max(bar_date) z from daily_bars '
                'group by symbol', con)
con.close()
have = set(q.symbol.astype(str))
print(f'   booked symbols {len(syms)} | in daily_bars {sum(s in have for s in syms)} '
      f'({sum(s in have for s in syms)/len(syms):.1%})')
print(f'   daily_bars span {q.a.min()} -> {q.z.max()}')
print('\nDONE (no cell scored)', flush=True)
