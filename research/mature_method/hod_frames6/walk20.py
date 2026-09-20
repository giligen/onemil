#!/usr/bin/env python3
"""hod_frames6 / F20 stage 2 — THE PLACEBO BAR WALK.

For every booked `B2` trade, prices the identical bracket (same stop distance as a % of price, same
+2R target, same 15:55 walk, same stop slippage) from:

  parity  — the trade's OWN entry bar and OWN stop      -> must reproduce `rr` exactly (PREREG R3)
  arm a   — every eligible NON-break minute of the same symbol-day
  arm b   — the same minute on each of the 25 matched non-signal symbols
  arm c   — the same symbol, 15 minutes earlier
  arm d   — the matched non-signal symbols at RANDOM eligible non-break minutes (the universe bound)

Resumable per day.  Read-only on every store.  TEST is never touched (book6.csv is TRAIN+VAL only).
"""
import json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import (D6, ROOT, walk_from, bars_arrays, new_hod_mask,                  # noqa: E402
                     FIRST_ENTRY_M, LAST_ENTRY_M)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
from pass2 import load_bars                                                            # noqa: E402

SEED = 20260920
N_D = 5                      # random non-break minutes per (trade, control) pair for arm d
OUT = {k: f'{D6}/p{k}6.csv' for k in ('arity', 'a', 'b', 'c', 'd')}
HDRS = {
    'arity': ['day', 'symbol', 'entry_m', 'rr_booked', 'rr_walk'],
    'a': ['day', 'symbol', 'entry_m', 'ctrl_entry_m', 'rr', 'why'],
    'b': ['day', 'symbol', 'entry_m', 'ctrl', 'rr', 'why'],
    'c': ['day', 'symbol', 'entry_m', 'ctrl_entry_m', 'rr', 'why'],
    'd': ['day', 'symbol', 'entry_m', 'ctrl', 'ctrl_entry_m', 'rr', 'why'],
}


def eligible_entries(h, m, skip_e):
    """Entry bar indices whose DECISION bar (e-1) did not make a new high of day."""
    nh = new_hod_mask(h)
    e = np.arange(1, len(h) - 1)
    ok = (~nh[e - 1]) & (m[e] >= FIRST_ENTRY_M) & (m[e] <= LAST_ENTRY_M) & (e != skip_e)
    return e[ok]


def main():
    bk = pd.read_csv(f'{D6}/book6.csv', dtype={'day': str, 'symbol': str},
                     keep_default_na=False, na_values=[''])
    pl = pd.read_csv(f'{D6}/pool6.csv', dtype={'day': str, 'symbol': str, 'ctrl': str},
                     keep_default_na=False, na_values=[''])
    PL = {d: g for d, g in pl.groupby('day')}
    st = f'{D6}/walk20_state.json'
    done = set(json.load(open(st))['done']) if os.path.exists(st) else set()
    days = [d for d in sorted(bk.day.unique()) if d not in done]
    print(f'{len(days)} sessions to walk ({len(done)} already done)', flush=True)
    rng = np.random.default_rng(SEED)

    for nd, day in enumerate(days):
        tr = bk[bk.day == day]
        po = PL.get(day, pl.iloc[:0])
        syms = sorted(set(tr.symbol) | set(po.ctrl.astype(str)))
        bars = load_bars(day, syms)
        arr = {}
        for s, gg in bars.items():
            a = bars_arrays(gg)
            if a is not None:
                arr[s] = a
        buf = {k: [] for k in OUT}
        POD = {(r.symbol, int(r.entry_m)): [] for r in tr.itertuples()}
        for r in po.itertuples():
            POD.setdefault((r.symbol, int(r.entry_m)), []).append(str(r.ctrl))

        for r in tr.itertuples():
            A = arr.get(r.symbol)
            if A is None:
                continue
            o, h, l, c, v, m = A
            key = (day, r.symbol, int(r.entry_m))
            idx = np.where(m == int(r.entry_m))[0]
            if not len(idx):
                continue
            e0 = int(idx[0])
            # ---- parity: the trade's own bar and own stop -------------------------------
            _, _, _, rrw = walk_from(o, h, l, c, m, e0, float(r.stop))
            buf['arity'].append(key + (float(r.rr), rrw))
            rp = float(r.r_pct) / 100.0
            # ---- arm a ------------------------------------------------------------------
            for e in eligible_entries(h, m, e0):
                E = float(o[e])
                _, _, why, rr = walk_from(o, h, l, c, m, int(e), E * (1.0 - rp))
                if rr == rr:
                    buf['a'].append(key + (int(m[e]), rr, why))
            # ---- arm c ------------------------------------------------------------------
            j = np.where(m == int(r.entry_m) - 15)[0]
            if len(j) and int(j[0]) + 1 < len(o):
                E = float(o[int(j[0])])
                _, _, why, rr = walk_from(o, h, l, c, m, int(j[0]), E * (1.0 - rp))
                if rr == rr:
                    buf['c'].append(key + (int(r.entry_m) - 15, rr, why))
            # ---- arms b and d -----------------------------------------------------------
            for cs in POD.get((r.symbol, int(r.entry_m)), []):
                B = arr.get(cs)
                if B is None:
                    continue
                bo, bh, bl, bc, bv, bm = B
                k = np.where(bm == int(r.entry_m))[0]
                if len(k) and int(k[0]) + 1 < len(bo):
                    E = float(bo[int(k[0])])
                    _, _, why, rr = walk_from(bo, bh, bl, bc, bm, int(k[0]), E * (1.0 - rp))
                    if rr == rr:
                        buf['b'].append(key + (cs, rr, why))
                el = eligible_entries(bh, bm, -1)
                if len(el):
                    for e in rng.choice(el, size=min(N_D, len(el)), replace=False):
                        E = float(bo[int(e)])
                        _, _, why, rr = walk_from(bo, bh, bl, bc, bm, int(e), E * (1.0 - rp))
                        if rr == rr:
                            buf['d'].append(key + (cs, int(bm[int(e)]), rr, why))

        for k, rows in buf.items():
            if rows:
                pd.DataFrame(rows, columns=HDRS[k]).to_csv(
                    OUT[k], mode='a', header=not os.path.exists(OUT[k]), index=False)
        done.add(day)
        json.dump({'done': sorted(done)}, open(st, 'w'))
        if nd % 10 == 0 or nd == len(days) - 1:
            print(f'  [{nd + 1}/{len(days)}] {day} syms {len(arr)} '
                  f'a{len(buf["a"])} b{len(buf["b"])} c{len(buf["c"])} d{len(buf["d"])}', flush=True)
    print('WALK20 DONE', flush=True)


if __name__ == '__main__':
    main()
