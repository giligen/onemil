"""Refuter 1 (look-ahead / obtainability) checks for cell 1,548-1,549. Read-only; imports the builder's
loaders but re-implements every trade walk independently."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil')
from research.hod_entry import cell_1548 as C
from research.hod_entry import cell_1445 as c1445

D, S, W = 0.9441, 1.9713, 120.0
df = C.load_population()
bars = C.load_bars()
fills = pd.read_csv(os.path.join(C.HERE, 'cell_1548_fills.csv'), dtype={'day': str})

def walk(entry, stop, target, path, strict_target=False, eod_at=955):
    for r in path.itertuples():
        if r.m >= eod_at:
            return r.m, r.o, 'eod'
        if r.l <= stop:
            return r.m, (r.o if r.o <= stop else stop), 'stop'
        if (r.h > target) if strict_target else (r.h >= target):
            return r.m, target, 'target'
    last = path.iloc[-1]
    return last.m, last.c, 'eod_fallback'

def cost_R(entry, stop, px, why, split, entry_cost):
    R = entry - stop
    bps = C.SLIP_STOP_BPS[split] if why == 'stop' else (C.EOD_BID_BPS[split] if why.startswith('eod') else 0.0)
    return R, ((px - entry) / R) - px * bps / 1e4 / R - entry_cost / R

# ---- 1. level / consolidation low knowable at the arm bar
lv_bad = st_fillbar_only = n_chk = 0
for r in df.itertuples():
    b = bars.get((r.symbol, r.day))
    if b is None: continue
    fm = int(np.floor(r.fill_min))
    pre = b[b.m < fm]; fb = b[b.m == fm]
    if pre.empty: continue
    n_chk += 1
    if pre.h.max() < r.level - 1e-6 and not (b[b.m < 570].empty is False):  # level above every pre-fill RTH high is fine (prior-day HOD)
        pass
    if pre.h.max() > r.level * 1.0005:  # level broken before the fill bar -> not the pre-fill HOD
        lv_bad += 1
    if len(fb) and abs(fb.l.iloc[0] - r.stop) < 1e-6 and not (np.abs(pre.l - r.stop) < 1e-6).any():
        st_fillbar_only += 1
print(f'[1] checked {n_chk}: level exceeded by a pre-fill-bar high (>0.05%): {lv_bad}; stop equals the fill-bar low only: {st_fillbar_only}')

# ---- 2. recompute a sample of builder trades exactly
rng = np.random.default_rng(1)
key = df.set_index(['day', 'symbol'])
samp = fills[(fills.status == 'filled')].sample(300, random_state=7)
mism = 0
for f in samp.itertuples():
    r = key.loc[(f.day, f.symbol)]
    b = bars[(f.symbol, f.day)]; fm = int(np.floor(r.fill_min))
    lvl = r.level; stop = lvl * (1 - S / 100); tgt = lvl * 1.05
    if f.cell == 1548:
        lim = lvl * (1 - D / 100)
        w = b[(b.m >= fm) & (b.m <= fm + W)]; hit = w[w.l < lim]
        em = hit.m.iloc[0]; ent = lim; ec = 0.0
    else:
        em = fm; ent = r.fill; ec = r.half_entry
    xm, xp, why = walk(ent, stop, tgt, b[b.m >= em])
    R, nR = cost_R(ent, stop, xp, why, f.split, ec)
    if abs(nR - f.net_R) > 1e-6 or em != f.entry_m: mism += 1
print(f'[2] 300-trade exact recompute: mismatches {mism}')

# ---- 3. obtainability variants on VAL kept (and TRAIN kept)
def run(split, kept, cell, fillbar_ok=True, strict=False, walk_from_next=False):
    sub = df[(df.split == split) & (df.hgb_kept_L3 == kept)]
    out = []; n_fb = 0; n_fb_openbelow = 0; n_same_bar_stop = 0; n_eod_after = 0
    for r in sub.itertuples():
        b = bars.get((r.symbol, r.day))
        if b is None or b.empty: continue
        fm = int(np.floor(r.fill_min)); lvl = r.level
        stop = lvl * (1 - S / 100); tgt = lvl * 1.05
        if cell == 1548:
            lim = lvl * (1 - D / 100)
            start = fm if fillbar_ok else fm + 1
            w = b[(b.m >= start) & (b.m <= fm + W)]; hit = w[w.l < lim]
            if hit.empty: continue
            em = int(hit.m.iloc[0]); ent = lim; ec = 0.0
            if em == fm:
                n_fb += 1
                if hit.o.iloc[0] < lim: n_fb_openbelow += 1
        else:
            em = fm; ent = r.fill; ec = r.half_entry
        path = b[b.m >= (em + 1 if walk_from_next else em)]
        if path.empty: continue
        xm, xp, why = walk(ent, stop, tgt, path, strict_target=strict)
        if xm == em and why == 'stop': n_same_bar_stop += 1
        R, nR = cost_R(ent, stop, xp, why, split, ec)
        out.append((r.day, nR, R / ent * 100, why))
    o = pd.DataFrame(out, columns=['day', 'net_R', 'R_pct', 'why'])
    t = c1445.day_clustered_t(o.net_R, o.day)
    return dict(n=len(o), mean_R=round(o.net_R.mean(), 4), t=round(float(t), 2),
                mean_pct=round((o.net_R * o.R_pct).mean(), 4), fillbar_fills=n_fb, fillbar_open_below=n_fb_openbelow,
                same_bar_stops=n_same_bar_stop, mix=o.why.value_counts(normalize=True).round(3).to_dict())

for split in ('TRAIN', 'VAL'):
    print(f'[3] {split} 1548 as built      ', run(split, True, 1548))
    print(f'[3] {split} 1548 no fill-bar fill', run(split, True, 1548, fillbar_ok=False))
    print(f'[3] {split} 1548 strict target  ', run(split, True, 1548, strict=True))
    print(f'[3] {split} 1549 as built      ', run(split, True, 1549))
    print(f'[3] {split} 1549 walk from fill_m+1 (upper bound: fill-bar pre-fill low ignored)', run(split, True, 1549, walk_from_next=True))
    print(f'[3] {split} 1549 strict target  ', run(split, True, 1549, strict=True))

# ---- 4. anatomy: touch bars after 15:55 among extenders counted in d/s/W
