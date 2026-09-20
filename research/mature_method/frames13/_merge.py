"""F40 — fold the 11 per-year files into ONE compact store, column by column in numpy.

`pd.concat` of 11 frames totalling 17.4M rows consolidates block by block and exceeds the node's
3 GB virtual cap; a per-column `np.concatenate` does not.  Run once.
"""
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

D13 = '/home/ec2-user/onemil/research/mature_method/frames13'
YEARS = range(2016, 2027)
COLS = {'sid': np.int32, 'd': np.int32, 'close': np.float32, 'adv20': np.float64,
        'on_pct': np.float32, 'ca': np.bool_, 'nights': np.int16}

acc = {k: [] for k in COLS}
for y in YEARS:
    t = pq.read_table(f'{D13}/on_{y}.parquet')
    for k, dt in COLS.items():
        acc[k].append(t.column(k).to_numpy(zero_copy_only=False).astype(dt))
    del t
    gc.collect()
    print(f'  {y} folded', flush=True)
# ONE .npy per column: `np.load(mmap_mode='r')` then streams from disk, so the scorer never
# materialises 17.4M rows of every column at once.  A pandas frame consolidates float blocks
# with a vstack and dies on this node.
n = 0
for k in COLS:
    a = np.concatenate(acc[k])
    n = len(a)
    np.save(f'{D13}/col_{k}.npy', a)
    acc[k] = None
    del a
    gc.collect()
    print(f'  col {k} written', flush=True)
print(f'{n:,} rows in 7 column files', flush=True)
