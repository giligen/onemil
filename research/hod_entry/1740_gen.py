"""Cell 1,740 arm generator (PREREG_1740). Population = hod_exit_lab/signals.parquet (entry_m = the BREAK minute). True level =
level_from_bars (1739_gen.py): max(h) over regular-hours 1-min bars with m < entry_m from bars_sip.db (read-only). Splits come from argv
(default TRAIN,VAL; TEST stays sealed). Writes 1740_arms[_<tag>].csv and 1740_gen_check[_<tag>].txt (sanity: break bar h >= level share,
chase distribution (B0 entry - level)/level). Usage: python3 1740_gen.py [TRAIN,VAL]"""
import sys, sqlite3
import numpy as np, pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil/research/hod_exit_lab')
import walker
ROOT = '/home/ec2-user/onemil'; OUT = ROOT + '/research/hod_entry'
splits = (sys.argv[1] if len(sys.argv) > 1 else 'TRAIN,VAL').split(',')
import os
OFF = int(os.environ.get('H1740_OFF', '0'))  # 1 = break bar is entry_m-1 (entry = open of entry_m): level excludes the break bar
tag = ('' if splits == ['TRAIN', 'VAL'] else '_' + '_'.join(s.lower() for s in splits)) + (f'_b{OFF}' if OFF else '')
walker.wait_for_db_window()
con = sqlite3.connect('file:' + ROOT + '/research/bf_zero/bars_sip.db?mode=ro', uri=True)

def level_from_bars(d, entry_m):
    """Running max of closed highs strictly before the break minute (09:30 onward); None if no prior bar. Same as 1739_gen.py."""
    p = d[d.m < entry_m]
    return float(p.h.max()) if len(p) else None

sig = pd.read_parquet(ROOT + '/research/hod_exit_lab/signals.parquet')
sig = sig[sig.split.isin(splits)].reset_index(drop=True)
rows = []
for i, r in enumerate(sig.itertuples()):
    d = walker.fetch_day_bars(con, r.symbol, r.day)
    lv, hb, hasprior = None, np.nan, False
    if d is not None:
        lv = level_from_bars(d, r.entry_m - OFF)
        b = d[d.m >= r.entry_m - OFF]
        hb = float(b.h.iloc[0]) if len(b) else np.nan
    rows.append((r.day, r.symbol, r.entry_m, lv, r.split, r.half, d is not None, r.entry, hb))
    if i % 1000 == 0: walker.log(f'gen {i}/{len(sig)}')
a = pd.DataFrame(rows, columns=['day', 'symbol', 'entry_m', 'level', 'split', 'half', 'has_bars', 'b0_entry', 'break_bar_h'])
a[['day', 'symbol', 'entry_m', 'level', 'split', 'half']].to_csv(f'{OUT}/1740_arms{tag}.csv', index=False)
ok = a[a.level.notna()]
L = [f'splits {splits}; signals {len(a)}; bars present {a.has_bars.mean():.4f}; level computed {a.level.notna().mean():.4f}',
     'level = max(h) of bars with m < entry_m from 09:30 (level_from_bars of 1739_gen.py)']
for s in splits:
    x = ok[ok.split == s]; ch = ((x.b0_entry - x.level) / x.level)
    L.append(f'{s}: n {len(x)}; share break bar h >= level {(x.break_bar_h >= x.level - 1e-9).mean():.4f}; share break bar h > level {(x.break_bar_h > x.level + 1e-9).mean():.4f}; '
             f'chase (B0 entry-level)/level median {ch.median():.4f} p10 {ch.quantile(.1):.4f} p90 {ch.quantile(.9):.4f}; share entry<level {(ch < 0).mean():.4f}')
open(f'{OUT}/1740_gen_check{tag}.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L), flush=True)
