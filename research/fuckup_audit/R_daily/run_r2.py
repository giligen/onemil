#!/usr/bin/env python3
"""R_daily step 3 (REPORT_v2) — the SAME 20 pre-registered cells on the CORRECTED panel.

No cell is added, no cut moved, no hold changed, no gate relaxed: PREREG.md §2/§3 verbatim, the
same TRAIN/VAL/TEST splits, the same universe rule, the same cost models, the same tails,
permutation and MDE.  `run_r.py` is imported for its universe function and calendar so the two runs
differ ONLY in the panel and in the two implementation defects fixed below.

Differences from `run_r.py`, both data-integrity fixes and both reported:

1. PANEL -> `daily_panel_2018_2026_adj.parquet` (split-adjusted from Alpaca corporate actions, with
   bad-print extended-session opens blanked; `build_panel_r2.py`).

2. `simulate` counts the hold in SESSIONS, not in row offsets.  The Stage-K implementation walks
   row indices inside a symbol's block, so a symbol that stops printing and later resumes (a
   delisting with the ticker reissued, or a long halt) has its "10-session hold" span the gap:
   BOLD entered 2020-01-07 and "exited" 2024-03-28, a 1,543-calendar-day hold booked as a -76%
   trade.  28 such trades and 115 entries more than five calendar days after the signal exist in
   the original run.  PREREG says "the next trading bar's OPEN (the morning after the signal
   close)" and "the close of the hold's last bar"; this implements exactly that:
       entry bar   : must be within MAX_ENTRY_GAP sessions of the signal day, else no trade
       exit bar    : the last available bar with session index <= signal session + hold
   Everything else in `simulate` — stops, costs, columns, drop counters — is byte-identical.

    python3 run_r2.py A     # TRAIN + VAL only (TEST is never computed)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')
ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/K')
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/R_daily')
os.chdir(ROOT)

import build_k as bk        # noqa: E402
import report_k as rk       # noqa: E402
import run_r                # noqa: E402,F401  (installs the PIT universe, splits, early closes)

R = f'{ROOT}/research/fuckup_audit/R_daily'
V2 = f'{R}/v2'
MAX_ENTRY_GAP = 3           # sessions; "the morning after" tolerating a one/two-session halt

os.makedirs(V2, exist_ok=True)
os.makedirs(f'{V2}/trades', exist_ok=True)
bk.PANEL = f'{R}/daily_panel_2018_2026_adj.parquet'
bk.K = V2
rk.K = V2

N_ENTRY_GAP_DROPPED = 0
N_HOLD_TRUNCATED = 0


def simulate(fam, f, mask, hold, n_days):
    """`bk.simulate` with the hold counted in SESSIONS (see the module docstring)."""
    global N_ENTRY_GAP_DROPPED, N_HOLD_TRUNCATED
    cfg = bk.FAMILIES[fam]
    idx = np.flatnonzero(mask)
    sym, day = f['sym'], f['day']
    op, hi, lo, cl = f['open'], f['high'], f['low'], f['close']
    ends_of = {}
    for a, b in zip(f['starts'], f['ends']):
        ends_of[sym[a]] = b
    rows = []
    n_past_end = n_no_next = n_gap = n_trunc = 0
    for i in idx:
        if day[i] + hold > n_days - 1:
            n_past_end += 1
            continue
        b = ends_of[sym[i]]
        if i + 1 >= b:
            n_no_next += 1
            continue
        if day[i + 1] - day[i] > MAX_ENTRY_GAP:      # not "the morning after"
            n_gap += 1
            continue
        entry = op[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            n_no_next += 1
            continue
        if cfg['stop'] == 'gap_low':
            stop = lo[i]
        elif cfg['stop'] == 'pct7':
            stop = entry * 0.93
        elif cfg['stop'] == 'pct5':
            stop = entry * 0.95
        else:
            stop = np.nan
        # last bar whose SESSION index is within `hold` sessions of the signal
        last = i + 1
        while last + 1 < b and day[last + 1] <= day[i] + hold:
            last += 1
        full = day[last] == day[i] + hold
        if not full:
            n_trunc += 1
        exit_i, why = last, ('hold' if full else 'truncated')
        if np.isfinite(stop):
            for j in range(i + 1, last + 1):
                if cl[j] <= stop:
                    exit_i, why = j, 'stop'
                    break
        exit_px = cl[exit_i]
        if not np.isfinite(exit_px) or exit_px <= 0:
            n_no_next += 1
            continue
        dv = f['dvol20_med'][i]
        cp, cd, ca = bk.cost_rt(dv)
        gross = exit_px / entry - 1.0
        r_pct = (entry - stop) / entry if np.isfinite(stop) else np.nan
        rows.append((int(sym[i]), int(day[i]), int(day[i + 1]), int(day[exit_i]),
                     float(entry), float(exit_px), float(stop) if np.isfinite(stop) else np.nan,
                     float(r_pct) if np.isfinite(r_pct) else np.nan, why, float(gross),
                     float(gross - cp), float(gross - cd), float(gross - ca), float(dv),
                     float(f['gap'][i]), float(f['vol_ratio'][i]), float(f['ret5'][i]),
                     float(f['ret3'][i]), float(f['on20'][i]), float(f['close'][i])))
    cols = ['sym', 'sig_day', 'entry_day', 'exit_day', 'entry', 'exit', 'stop', 'r_pct', 'why',
            'gross', 'net', 'net_daily', 'net_auction', 'dvol20_med', 'gap', 'vol_ratio',
            'ret5', 'ret3', 'on20', 'sig_close']
    t = pd.DataFrame(rows, columns=cols)
    t['fam'] = fam
    t['hold'] = hold
    N_ENTRY_GAP_DROPPED += n_gap
    N_HOLD_TRUNCATED += n_trunc
    bk.log(f'  {fam} hold {hold}: {len(t):,} simulated  (dropped: {n_past_end:,} past panel end, '
           f'{n_no_next:,} no next/exit bar, {n_gap:,} entry bar more than {MAX_ENTRY_GAP} '
           f'sessions after the signal; {n_trunc:,} holds truncated at the last available bar)')
    return t, n_past_end, n_no_next


bk.simulate = simulate


# ---------------------------------------------------------------- the price gate, on RAW prices
# Back-adjustment is a look-ahead for a PRICE gate: 5,092 of the 5,889 events are REVERSE splits, so
# every pre-event price is multiplied UP and a sub-$5 penny stock that later reverse-split would
# sail through PREREG's `close >= $5`.  The gate must see the price that actually traded, so the
# panel carries `close_raw` and the universe test uses it.  (The $10M dollar-volume gate needs no
# such care: close x volume is invariant under the adjustment.)
_orig_build_features = bk.build_features


def build_features():
    import gc
    import pyarrow.parquet as pq
    f = _orig_build_features()
    parts = []
    pf = pq.ParquetFile(bk.PANEL)
    for b in pf.iter_batches(batch_size=200_000, columns=['close_raw']):
        parts.append(b.column('close_raw').to_numpy(zero_copy_only=False).astype('float32'))
        del b
    del pf
    raw = np.concatenate(parts)
    del parts
    gc.collect()
    if len(raw) != len(f['close']):
        raise RuntimeError(f'close_raw length {len(raw)} != panel rows {len(f["close"])}')
    f['close_raw'] = np.where(f['bad'], np.nan, raw).astype('float32')
    return f


bk.build_features = build_features
_pit_universe = bk.build_universe


def build_universe(f):
    u_prim, u_sec, cls_of = _pit_universe(f)
    raw_ok = np.isfinite(f['close_raw']) & (f['close_raw'] >= bk.MIN_PRICE)
    adj_only = (u_prim & ~raw_ok).sum()
    bk.log(f'price gate on the RAW traded close: {int(adj_only):,} primary-universe rows dropped '
           f'that only passed >= ${bk.MIN_PRICE:g} because back-adjustment inflated them')
    return u_prim & raw_ok, u_sec & raw_ok, cls_of


bk.build_universe = build_universe


if __name__ == '__main__':
    phase = sys.argv[1] if len(sys.argv) > 1 else 'A'
    bk.main(phase)
    rk.main(phase)
    bk.log(f'session-faithful hold: {N_ENTRY_GAP_DROPPED:,} signals dropped for a late entry bar, '
           f'{N_HOLD_TRUNCATED:,} holds truncated at the symbol\'s last available bar')
