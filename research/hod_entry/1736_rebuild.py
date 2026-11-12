"""Independent rebuild of cell 1,736 (short the failed touch of the HOD) from PREREG prose only."""
import sqlite3, sys, random, os
STOP_HS = os.environ.get('STOP_HS','1') == '1'  # stop leg pays hs + 13bp (reading A) vs 13bp only
NEXTBAR = os.environ.get('NEXTBAR','0') == '1'  # missing bar F+1 -> use next available bar
OUT = os.environ.get('OUT','research/hod_entry/1736_rebuild_events.csv')
import numpy as np, pandas as pd
sys.path.insert(0, 'research/hod_exit_lab')
import walker

BP = 1e-4
SIG = pd.read_csv('research/hod_entry/replay_signals.csv')
SIG = SIG[SIG.level.notna() & (SIG.void_reason != 'no_level')].reset_index(drop=True)
HS = pd.read_parquet('research/hod_exit_lab/signals.parquet')[['day', 'symbol', 'half_spread']]
SIG = SIG.merge(HS, on=['day', 'symbol'], how='left')
SIG['matched_spread'] = SIG.half_spread.notna()
SIG['dec'] = pd.qcut(SIG.level, 10, labels=False, duplicates='drop')
med = SIG[SIG.matched_spread].groupby('dec').half_spread.median()
SIG['hs'] = np.where(SIG.matched_spread, SIG.half_spread, SIG.dec.map(med))
print('arms', len(SIG), 'unmatched share', 1 - SIG.matched_spread.mean(), flush=True)

bars_con = sqlite3.connect('file:research/bf_zero/bars_sip.db?mode=ro', uri=True)
cache = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)

def prior_close(sym, day):
    r = cache.execute('select close from daily_bars where symbol=? and bar_date<? order by bar_date desc limit 1', (sym, day)).fetchone()
    return r[0] if r else None

def exit_e1(b, i0, entry, stop, R, hs):
    """E1: target -2R / stop(+13bp) / 15:55 open. b = bars frame, i0 = index of bar F+1."""
    target = entry - 2 * R
    for i in range(i0, len(b)):
        m = b.m.iat[i]
        if m >= 955: break
        o, h, l = b.o.iat[i], b.h.iat[i], b.l.iat[i]
        if h >= stop:  # stop wins when both
            fill = max(o, stop) * (1 + 13 * BP) + (hs if STOP_HS else 0)
            return fill, m, 'stop'
        if l <= target:
            return target * (1 + 2 * BP) + hs, m, 'target'
    sub = b[b.m >= 955]
    if len(sub):
        return sub.o.iat[0] * (1 + 2 * BP) + hs, 955, 'time'
    return b.c.iat[-1] * (1 + 2 * BP) + hs, int(b.m.iat[-1]), 'time_lastbar'

def exit_e2(b, i0, entry, stop, R, hs, vw):
    """E2: cover at open of bar after first close >= session VWAP; else stop / 15:55."""
    reclaim = False
    for i in range(i0, len(b)):
        m = b.m.iat[i]
        if m >= 955: break
        o, h = b.o.iat[i], b.h.iat[i]
        if reclaim:
            return o * (1 + 2 * BP) + hs, m, 'reclaim'
        if h >= stop:
            return max(o, stop) * (1 + 13 * BP) + (hs if STOP_HS else 0), m, 'stop'
        if b.c.iat[i] >= vw[i]:
            reclaim = True
    sub = b[b.m >= 955]
    if len(sub):
        return sub.o.iat[0] * (1 + 2 * BP) + hs, 955, 'time'
    return b.c.iat[-1] * (1 + 2 * BP) + hs, int(b.m.iat[-1]), 'time_lastbar'

rows = []; arm_check = []
rng = random.Random(7)
chk_idx = set(rng.sample(range(len(SIG)), 20))
for k, s in SIG.iterrows():
    out = dict(day=s.day, symbol=s.symbol, entry_m=int(s.entry_m), level=s.level, split=s.split, event=0, rails=np.nan,
               e1_fill=s.e1_fill, half_spread=s.hs, matched_spread=bool(s.matched_spread), status='')
    b = walker.fetch_day_bars(bars_con, s.symbol, s.day)
    if b is None:
        out['status'] = 'nobars'; rows.append(out); continue
    if k in chk_idx:
        j = b[b.m == int(s.entry_m)]; after = b[b.m > int(s.entry_m)]
        arm_check.append((s.day, s.symbol, bool(len(j) and j.h.iat[0] < s.level), bool(len(after) and after.h.max() >= s.level)))
    L = s.level
    a = b[b.m >= int(s.entry_m) + 1].reset_index(drop=True)
    th = a.index[a.h >= L]
    if len(th) == 0:
        out['status'] = 'notouch'; rows.append(out); continue
    ti = th[0]; T = int(a.m.iat[ti]); out['touch_m'] = T
    win = a[(a.m >= T) & (a.m <= T + 3)]
    F = None; success = False
    for _, w in win.iterrows():
        if w.c >= L * 1.006:
            success = True; break
        if w.c <= L * (1 - 0.002):
            F = w; break
    if F is None:
        out['status'] = 'success' if success else 'nofail'; rows.append(out); continue
    out['event'] = 1; fm = int(F.m); out['fail_m'] = fm
    pc = prior_close(s.symbol, s.day)
    ok = (pc is not None) and F.c >= 0.90 * pc and fm <= 930
    out['rails'] = 'ok' if ok else ('late' if fm > 930 else 'ssr')
    if not ok:
        out['status'] = 'rails_fail'; rows.append(out); continue
    nxt = a[a.m == fm + 1] if not NEXTBAR else a[a.m > fm].head(1)
    if not len(nxt):
        out['status'] = 'no_entry_bar'; rows.append(out); continue
    hs = s.hs
    entry = nxt.o.iat[0] * (1 - 2 * BP) - hs
    hh = win[win.m <= fm].h.max()
    stop = hh * 1.002; R = stop - entry
    out.update(entry=entry, stop=stop, R=R, r_pct=100 * R / entry)
    if R < 0.01 * entry:
        out['status'] = 'r_too_small'; out['rails'] = 'r_too_small'; rows.append(out); continue
    i0 = int(nxt.index[0])
    cv, em, why = exit_e1(a, i0, entry, stop, R, hs)
    out.update(e1_exit=cv, e1_why=why, e1_net_R=(entry - cv) / R, e1_exit_m=em)
    # session VWAP on the FULL day frame, aligned by minute to a
    vw_full = pd.Series(walker.session_vwap(b), index=b.m.values)
    vw = vw_full.reindex(a.m.values).values
    cv2, em2, why2 = exit_e2(a, i0, entry, stop, R, hs, vw)
    out.update(e2_exit=cv2, e2_why=why2, e2_net_R=(entry - cv2) / R, e2_exit_m=em2)
    out['status'] = 'trade'; rows.append(out)
    if k % 500 == 0: print('progress', k, flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print('arm check (day,sym,barj_high<level, later high>=level):', arm_check)
print('arm check all pass:', all(x[2] and x[3] for x in arm_check), flush=True)
print(df.status.value_counts().to_dict())
