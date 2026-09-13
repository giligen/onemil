#!/usr/bin/env python3
"""Ignition-from-zero — H3: first-pullback entry (DESIGN.md), the one hypothesis
that needs its own entry simulation.

Pre-registered definition (written before the run):
  * universe = the BASE book rows of score.py (level 10, gated, in BT window)
  * after the trigger bar T, wait for a PULLBACK: at least one bar whose low is
    below the previous bar's low, without the low ever reaching the original
    (pre-trigger) stop — that is the immediate-fade case and is a NO-TRADE
  * the ENTRY signal = the first bar after that pullback whose high exceeds the
    previous bar's high (the first higher-low + break, bull-flag style);
    entry = next bar open x ENTRY_SLIP; stop = the pullback low (min low since T)
  * the signal must arrive within PB_MAX_MIN minutes of T, else NO-TRADE
  * R-min gate re-applied to the new (entry, stop) — same rule as pass 1
  * exits = the same walk() as pass 1 (p1be and v0) from the entry bar
Output: pullback.csv keyed (day, symbol) with rr_pb_p1be / rr_pb_v0 / pb_status.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, f'{ROOT}/research/ignition_zero')
import build_candidates as BC  # noqa: E402  (module-level data loads; main() not run)
R = BC.R
D = 'research/ignition_zero'
PB_MAX_MIN = 20

c = pd.read_csv(f'{D}/candidates_full.csv', low_memory=False)
base = c[(c.flag_chase == 0) & (c.flag_prebars == 0) & (c.flag_rmin == 0) & (c.level == 10) & (c.in_bt_window == 1)]
print('base rows', len(base), 'days', base.day.nunique(), flush=True)
rows = []
for n, (day, sub) in enumerate(base.groupby('day')):
    B = BC.load_bars(day, sub.symbol.tolist())
    for r in sub.itertuples():
        gg = B.get(r.symbol)
        st = dict(day=day, symbol=r.symbol, pb_status='no_bars', rr_pb_p1be=np.nan, rr_pb_v0=np.nan, pb_entry_m=np.nan, pb_r_pct=np.nan)
        if gg is not None:
            rth = gg[(gg.m >= 570) & (gg.m < 960)].reset_index(drop=True)
            ti = rth.index[rth.m == r.trig_m]
            if len(ti):
                ti = ti[0]; post = rth[rth.index > ti]
                pulled = False; pb_low = np.inf; prev = rth.loc[ti]; status = 'no_signal'; sig = None
                for b in post.itertuples():
                    if b.m > r.trig_m + PB_MAX_MIN: status = 'timeout'; break
                    if b.low <= r.stop: status = 'faded_to_stop'; break
                    pb_low = min(pb_low, b.low)
                    if not pulled and b.low < prev.low: pulled = True
                    elif pulled and b.high > prev.high: sig = b; break
                    prev = b
                if sig is not None:
                    nxt = rth[rth.index > sig.Index]
                    if len(nxt):
                        nb = nxt.iloc[0]; entry = float(nb.open) * R.ENTRY_SLIP; stop = float(pb_low)
                        rp = R.r_pct_from_stop(entry, stop)
                        if stop >= entry: status = 'stop_above_entry'
                        elif rp < R.R_MIN_PCT: status = 'rmin_reject'
                        else:
                            status = 'entered'
                            walk_post = rth[rth.index > sig.Index]
                            st['rr_pb_p1be'] = BC.walk(walk_post, entry, stop, int(nb.m), mode='p1be')[0]
                            st['rr_pb_v0'] = BC.walk(walk_post, entry, stop, int(nb.m), mode='v0')[0]
                            st['pb_entry_m'] = int(nb.m); st['pb_r_pct'] = rp
                    else: status = 'no_next_bar'
                st['pb_status'] = status
        rows.append(st)
    if n % 25 == 0: print(f'{n}/{base.day.nunique()} {day} rows={len(rows)}', flush=True)
out = pd.DataFrame(rows); out.to_csv(f'{D}/pullback.csv', index=False)
print('DONE', out.pb_status.value_counts().to_dict(), flush=True)
