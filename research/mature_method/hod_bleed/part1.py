#!/usr/bin/env python3
"""hod_bleed PART 1 — the path map (PREREG §3).  DESCRIPTIVE ONLY: no cell, no gate, no selection
is evaluated here.  Pure table work on walk3.py's artifacts.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_bleed')
import score as S             # noqa: E402
import score2 as S2           # noqa: E402
import core as C              # noqa: E402

D = f'{ROOT}/research/mature_method/hod_bleed'
OUT = open(f'{D}/part1.txt', 'w')
ARMS = (0.3, 0.4, 0.5, 0.6)
STOPS = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2)
MARKS = (1, 3, 5, 10, 15)


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); OUT.write(s + '\n'); OUT.flush()


def attach_sid(sg, keys):
    k = keys.copy(); sg = sg.copy()
    for d in (sg, k):
        d['_st'] = d.stop.round(6)
    return sg.merge(k[['day', 'symbol', 'entry_m', '_st', 'sid']],
                    on=['day', 'symbol', 'entry_m', '_st'], how='left')


def main():
    pop = S2.load_pop(); S.build_impute(pop)
    keys = pd.read_csv(f'{D}/keys.csv', **C.RD)
    P = C.Paths()
    p(f'paths loaded: {len(P.sig)} signals, {len(P.op):,} bars')

    SG = {}
    for nm in ('B0', 'B2'):
        sg = S2.sig_set(pop, **S2.BASES[nm])
        sg = sg[sg.split.isin(('TRAIN', 'VAL'))]
        SG[nm] = attach_sid(sg, keys).reset_index(drop=True)
    del pop

    # ---------------------------------------------------------------- parity gate
    p('\n' + '=' * 110)
    p('PARITY GATE — the re-simulated `base` vs pop.csv, on every base')
    p('=' * 110)
    cache = {}
    for nm in ('B0', 'B2'):
        sg = SG[nm]
        rr, wy, em = [], [], []
        for r in sg.itertuples():
            sid = r.sid
            if sid != sid or not P.has(int(sid)):
                rr.append(np.nan); wy.append(''); em.append(-1); continue
            sid = int(sid)
            if sid not in cache:
                cache[sid] = C.prep(P.get(sid))
            Q = cache[sid]
            k, x, w, _ = C.sim_base(Q)
            rr.append(x); wy.append(w); em.append(int(Q['m'][k]))
        sg['rr_sim'] = rr; sg['why_sim'] = wy; sg['exit_m_sim'] = em
        ok = sg.rr_sim.notna()
        p(f'  {nm}: {ok.sum()}/{len(sg)} walked   max|d rr| '
          f'{np.abs(sg.rr[ok] - sg.rr_sim[ok]).max():.3e}   why match '
          f'{(sg.why[ok] == sg.why_sim[ok]).mean():.6f}   exit-minute match '
          f'{(sg.exit_m[ok] == sg.exit_m_sim[ok]).mean():.6f}')
        SG[nm] = sg

    # the shipped books
    BK = {nm: S.apply_book(SG[nm][SG[nm].rr_sim.notna()], 12, 4) for nm in ('B0', 'B2')}
    for nm in ('B0', 'B2'):
        b = BK[nm]
        for sp in ('TRAIN', 'VAL'):
            w = S.week_stats(b, sp)
            p(f'  {nm} shipped {sp:5s} n{w["n"]:5d} {w["per_wk"]:5.1f}/wk gross {w["gross"]:+.3f} '
              f'net {w["net"]:+.3f} cost/R {w["gross"] - w["net"]:.4f} green {w["green"]:5.1f}% '
              f'${w["total"]:+,.0f}')

    b0 = BK['B0']
    p('\n' + '=' * 110)
    p('1. THE PATH, by outcome class — the shipped B0 BOOK (the population every cell acts on)')
    p('=' * 110)
    for sp in ('TRAIN', 'VAL'):
        d = b0[b0.split == sp]
        cls = np.where(d.why_sim == 'eod', 'eod', np.where(d.rr_sim > 0, 'win', 'lose'))
        p(f'\n  -- {sp} (n {len(d)}) --')
        p(f'  {"class":6s} {"n":>5s} | ' + ' '.join(f'{"cur@" + str(k):>9s}' for k in MARKS)
          + ' | ' + ' '.join(f'{"mfe@" + str(k):>9s}' for k in MARKS)
          + ' | ' + ' '.join(f'{"rtr@" + str(k):>9s}' for k in MARKS) + ' | med MFE  med MAE  hold')
        for cl in ('win', 'lose', 'eod'):
            sub = d[cls == cl]
            if not len(sub):
                continue
            cur = {k: [] for k in MARKS}; mfe = {k: [] for k in MARKS}; rtr = {k: [] for k in MARKS}
            mf, ma, hd = [], [], []
            for sid in sub.sid.astype(int):
                Q = cache[sid]; nk = Q['nk']
                ke, _, _, _ = C.sim_base(Q)
                for k in MARKS:
                    j = min(k - 1, ke)
                    cur[k].append(Q['cl'][j]); mfe[k].append(Q['runmax'][j])
                    rtr[k].append(Q['runmax'][j] - Q['cl'][j])
                mf.append(Q['runmax'][ke]); ma.append(Q['lo'][:ke + 1].min()); hd.append(ke)
            p(f'  {cl:6s} {len(sub):5d} | '
              + ' '.join(f'{np.median(cur[k]):+9.3f}' for k in MARKS) + ' | '
              + ' '.join(f'{np.median(mfe[k]):+9.3f}' for k in MARKS) + ' | '
              + ' '.join(f'{np.median(rtr[k]):+9.3f}' for k in MARKS) + ' | '
              f'{np.median(mf):+7.3f} {np.median(ma):+8.3f} {np.median(hd):5.0f}')

    p('\n' + '=' * 110)
    p('2. THE ARM LEVEL — who reaches a (shipped B0 book)')
    p('=' * 110)
    p(f'  {"split":6s} {"a":>5s} {"reach n":>8s} {"reach %":>8s} {"win%":>7s} {"lose%":>7s} '
      f'{"eod%":>7s} {"med arm min":>12s} {"lose med arm":>13s}')
    for sp in ('TRAIN', 'VAL'):
        d = b0[b0.split == sp]
        for a in ARMS:
            reach, cls_, armm, armm_l = [], [], [], []
            for sid, rr, wy in zip(d.sid.astype(int), d.rr_sim, d.why_sim):
                Q = cache[sid]; ka = C.arm_k(Q, a)
                hit = ka < C.BIG and ka <= C.sim_base(Q)[0]
                reach.append(hit)
                c_ = 'eod' if wy == 'eod' else ('win' if rr > 0 else 'lose')
                cls_.append(c_)
                if hit:
                    armm.append(ka + 1)
                    if c_ == 'lose':
                        armm_l.append(ka + 1)
            reach = np.array(reach); cls_ = np.array(cls_)
            r = cls_[reach]
            p(f'  {sp:6s} {a:5.1f} {reach.sum():8d} {reach.mean() * 100:7.1f}% '
              f'{(r == "win").mean() * 100:6.1f}% {(r == "lose").mean() * 100:6.1f}% '
              f'{(r == "eod").mean() * 100:6.1f}% {np.median(armm) if armm else np.nan:12.1f} '
              f'{np.median(armm_l) if armm_l else np.nan:13.1f}')

    p('\n' + '=' * 110)
    p('3. THE (a, s) MATRIX — of the WINNERS that reach a, what share are ejected by a stop at s')
    p('   (a winner is a trade whose SHIPPED exit is > 0 R; "ejected" = the ratchet stop fires)')
    p('=' * 110)
    for sp in ('TRAIN', 'VAL'):
        p(f'\n  -- {sp} — % of winners-that-reach-a ejected by the stop at s --')
        p(f'  {"a":>5s} {"win n":>7s} | ' + ' '.join(f'{f"s={s:+.1f}":>9s}' for s in STOPS))
        d = b0[b0.split == sp]
        for a in ARMS:
            sids = [int(x) for x, rr, wy in zip(d.sid, d.rr_sim, d.why_sim)
                    if rr > 0 and wy != 'eod' and C.arm_k(cache[int(x)], a) < C.BIG
                    and C.arm_k(cache[int(x)], a) <= C.sim_base(cache[int(x)])[0]]
            row = []
            for s in STOPS:
                ej = sum(1 for x in sids if C.sim_ratchet(cache[x], a, s)[2] == 'ratchet')
                row.append(ej / len(sids) * 100 if sids else np.nan)
            p(f'  {a:5.1f} {len(sids):7d} | ' + ' '.join(f'{v:8.1f}%' for v in row))

    p('\n' + '=' * 110)
    p('4. THE TRADE-OFF, in R per booked trade — winners lost vs losers cut, per (a, s)')
    p('   dR = [ sum(R saved on trades the ratchet improves) - sum(R given up on trades it hurts) ] / n')
    p('=' * 110)
    rows = []
    for nm in ('B0',):
        for sp in ('TRAIN', 'VAL'):
            d = BK[nm][BK[nm].split == sp]
            base = {int(x): C.sim_base(cache[int(x)])[1] for x in d.sid}
            for a in ARMS:
                for s in STOPS:
                    wl_n = wl_r = lc_n = lc_r = 0
                    for x in d.sid.astype(int):
                        b_ = base[x]; n_ = C.sim_ratchet(cache[x], a, s)[1]
                        if n_ < b_ - 1e-12:
                            wl_n += 1; wl_r += b_ - n_
                        elif n_ > b_ + 1e-12:
                            lc_n += 1; lc_r += n_ - b_
                    rows.append(dict(base=nm, split=sp, a=a, s=s, n=len(d), lost_n=wl_n,
                                     lost_R=wl_r, cut_n=lc_n, cut_R=lc_r,
                                     dR=(lc_r - wl_r) / len(d)))
    M = pd.DataFrame(rows)
    M.to_csv(f'{D}/as_matrix.csv', index=False)
    for sp in ('TRAIN', 'VAL'):
        p(f'\n  -- {sp} — dR per booked trade (gross R, before cost) --')
        p(f'  {"a":>5s} | ' + ' '.join(f'{f"s={s:+.1f}":>9s}' for s in STOPS))
        for a in ARMS:
            r = M[(M.split == sp) & (M.a == a)].set_index('s')
            p(f'  {a:5.1f} | ' + ' '.join(f'{r.dR[s]:+9.4f}' for s in STOPS))
        p(f'  -- {sp} — (trades hurt / R given up) vs (trades helped / R saved) --')
        for a in ARMS:
            r = M[(M.split == sp) & (M.a == a)].set_index('s')
            p(f'  a={a:.1f} | ' + ' '.join(
                f's{s:+.1f}: -{r.lost_n[s]:3d}/{r.lost_R[s]:6.1f}R +{r.cut_n[s]:3d}/{r.cut_R[s]:6.1f}R'
                for s in STOPS))

    p('\n  TRAIN ranking of the 24 (a, s) pairs by dR (the PREREG §4 E1 selection rule):')
    tr = M[M.split == 'TRAIN'].sort_values('dR', ascending=False)
    for i, r in enumerate(tr.itertuples(), 1):
        p(f'    {i:2d}. a={r.a:.1f} s={r.s:+.1f}  dR {r.dR:+.4f}  (hurt {r.lost_n} / -{r.lost_R:.1f}R, '
          f'helped {r.cut_n} / +{r.cut_R:.1f}R)   [VAL dR '
          f'{M[(M.split == "VAL") & (M.a == r.a) & (M.s == r.s)].dR.iloc[0]:+.4f}]')
    OUT.close()


if __name__ == '__main__':
    main()
