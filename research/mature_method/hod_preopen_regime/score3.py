#!/usr/bin/env python3
"""hod_preopen_regime — score every cell declared in PREREG.md §3. Nothing here was run before
PREREG.md was committed (fe10e89).

TEST is sealed: no TEST number unless FREEZE.md exists AND --test is passed.
Read-only on every DB. One process.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S                                     # noqa: E402
import score2 as S2                                   # noqa: E402

D = f'{ROOT}/research/mature_method/hod_preopen_regime'
WITH_TEST = '--test' in sys.argv and os.path.exists(f'{D}/FREEZE.md')
S.SPLITS = ('TRAIN', 'VAL', 'TEST') if WITH_TEST else ('TRAIN', 'VAL')
SPLITS = S.SPLITS
GATE_TIMES = [(575, '0935'), (580, '0940'), (585, '0945'), (590, '0950'),
              (600, '1000'), (615, '1015'), (630, '1030')]
LADDER = [('sgn', 0.0), ('p20', 0.2), ('p40', 0.4), ('p60', 0.6)]

HDR = ('| cell                                   | split |     n |  /wk |  grossR |    netR |     t | '
       'green |  flat | rs |   worst |    total |      mdd | grnMo |')
SEP = '|' + '|'.join(['-' * 6] * 14) + '|'
BOOKS, CELLS = {}, []


def line(name, b):
    out = []
    for sp in SPLITS:
        w = S.week_stats(b, sp)
        out.append(f"| {name:<38s} | {sp:5s} | {w['n']:5d} | {w['per_wk']:4.1f} | {w['gross']:+.3f} | "
                   f"{w['net']:+.3f} | {w['t']:+5.2f} | {w['green']:5.1f} | {w['flat']:5.1f} | "
                   f"{w['redstreak']:2d} | {w['worst']:+8.0f} | {w['total']:+9.0f} | {w['mdd']:+8.0f} | "
                   f"{w['green_mo']:5.1f} |")
        w.update(cell=name, split=sp)
        CELLS.append(w)
    return '\n'.join(out)


def show(name, b):
    BOOKS[name] = b
    print(line(name, b), flush=True)


def _cluster_var(v, days):
    """Cluster-robust variance of a sample MEAN, clusters = trading days.

    HOD-break signals cluster violently by day (on the two biggest TRAIN days B2 carries ~300
    signals each), so the iid SE of a DAY-level gate is meaningless -- it counts one day as 300
    independent observations. This is the SE every separation below is judged on.
    """
    v = np.asarray(v, float); n = len(v)
    if n < 2:
        return np.nan
    res = v - v.mean()
    s = pd.Series(res).groupby(np.asarray(days)).sum().values
    return float((s ** 2).sum()) / (n ** 2)


def sep_of(kept, rej):
    """kept-minus-rejected, on the PRE-BOOK signal sets. `t` is iid, `tc` is day-clustered."""
    o = {}
    for sp in SPLITS:
        k = kept[kept.split == sp]; r = rej[rej.split == sp]
        if len(k) < 20 or len(r) < 20:
            o[sp] = None; continue
        d = float(k.net.mean() - r.net.mean())
        se = float(np.sqrt(k.net.var(ddof=1) / len(k) + r.net.var(ddof=1) / len(r)))
        vc = _cluster_var(k.net.values, k.day.values) + _cluster_var(r.net.values, r.day.values)
        sec = float(np.sqrt(vc)) if vc == vc and vc > 0 else np.nan
        o[sp] = dict(d=d, t=d / se if se else np.nan, tc=d / sec if sec else np.nan,
                     g=float(k.rr.mean() - r.rr.mean()), nk=len(k), nr=len(r),
                     dk=int(k.day.nunique()), dr=int(r.day.nunique()))
    return o


def fmt_sep(nm, o, extra=''):
    s = f'| {nm:<34s} |'
    for sp in SPLITS:
        q = o[sp]
        s += (f" {q['g']:+.3f} | {q['d']:+.3f} | {q['t']:+6.2f} | {q['tc']:+6.2f} | {q['nk']:5d} | "
              f"{q['dk']:4d} |" if q else '   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |')
    return s + extra


# ------------------------------------------------------------------ index minute tape
def idx_tape():
    ix = pd.read_csv(f'{D}/idx_1min.csv', dtype={'symbol': str, 'day': str})
    ix = ix[(ix.m_et >= 570) & (ix.m_et < 960)].sort_values(['symbol', 'day', 'm_et'])
    per_min, per_day = {}, {}
    for sym in ('SPY', 'QQQ'):
        s = ix[ix.symbol == sym].copy()
        s['hod'] = s.groupby('day').h.cummax()
        op = s.groupby('day').o.transform('first')          # the TRUE 09:30 open
        p = sym.lower()
        s[f'{p}_disthod_pct'] = (s.c / s.hod - 1.0) * 100.0
        s[f'{p}_ret_open_m'] = (s.c / op - 1.0) * 100.0
        per_min[sym] = s[['day', 'm_et', f'{p}_disthod_pct', f'{p}_ret_open_m']].rename(
            columns={'m_et': 'break_m'})
        # the T-ladder: 09:30 open -> close of the bar ENDING at g  (bar index g-1)
        d = {}
        for g, lab in GATE_TIMES:
            c = s[s.m_et == g - 1].set_index('day').c
            d[f'{p}_ret_{lab}'] = (c / s[s.m_et == 570].set_index('day').o - 1.0) * 100.0
        # first-bar (09:30) shape
        f0 = s[s.m_et == 570].set_index('day')
        d[f'{p}_b1_ret'] = (f0.c / f0.o - 1.0) * 100.0
        d[f'{p}_b1_rng'] = (f0.h - f0.l) / f0.o * 100.0
        d[f'{p}_b1_vol'] = f0.v
        d[f'{p}_b1_pos'] = ((f0.c - f0.l) / (f0.h - f0.l).replace(0, np.nan))
        # RTH daily closes/highs/lows from the same tape -> the M family, strictly prior sessions
        dd = s.groupby('day').agg(hi=('h', 'max'), lo=('l', 'min'), cl=('c', 'last'))
        for k in (1, 2, 3, 5):
            d[f'{p}_ret_{k}d'] = (dd.cl.shift(1) / dd.cl.shift(1 + k) - 1.0) * 100.0
        r1 = dd.cl.shift(1) / dd.cl.shift(2) - 1.0
        r2 = dd.cl.shift(2) / dd.cl.shift(3) - 1.0
        d[f'{p}_path2'] = pd.Series(
            np.where((r1 > 0) & (r2 > 0), 2.0, np.where((r1 < 0) & (r2 < 0), 0.0, 1.0)),
            index=dd.index).where(r1.notna() & r2.notna())
        d[f'{p}_prev_close_pos'] = ((dd.cl - dd.lo) / (dd.hi - dd.lo).replace(0, np.nan)).shift(1)
        d[f'{p}_inside'] = ((dd.hi.shift(1) <= dd.hi.shift(2)) & (dd.lo.shift(1) >= dd.lo.shift(2))).astype(float)
        d[f'{p}_outside'] = ((dd.hi.shift(1) > dd.hi.shift(2)) & (dd.lo.shift(1) < dd.lo.shift(2))).astype(float)
        # ATR20 / mean 09:30-bar volume, both ending T-1
        d[f'{p}_atr20'] = ((dd.hi - dd.lo) / dd.cl * 100.0).rolling(20).mean().shift(1)
        d[f'{p}_b1_vol20'] = f0.v.rolling(20).mean().shift(1)
        per_day[sym] = pd.DataFrame(d)
    pm = per_min['SPY'].merge(per_min['QQQ'], on=['day', 'break_m'], how='outer')
    pd_ = per_day['SPY'].join(per_day['QQQ'], how='outer')
    return pm, pd_


def main():
    print('== hod_preopen_regime — scoring PREREG.md §3 ==', flush=True)
    p = S2.load_pop()
    S.build_impute(p)

    # ---------------- reproduction gate ---------------------------------------------------
    print('\n== REPRODUCTION GATE (must match hod_filter_stack/REPORT.md §2 to the dollar) ==')
    print(HDR + '\n' + SEP)
    pre = {nm: S2.sig_set(p, **kw) for nm, kw in S2.BASES.items() if nm in ('B0', 'B2', 'B3')}
    for nm in ('B0', 'B2', 'B3'):
        show(nm, S.apply_book(pre[nm], 12, 4))
    print('  reference §2: B0 TRAIN 1688 -0.027 -0.088 41.5 -14835 | B0 VAL 820 +0.016 -0.050 43.5 -4128')
    print('                B2 TRAIN 1622 -0.039 -0.107 32.1 -17346 | B2 VAL  706 +0.083 +0.013 43.5   +893')
    print('                B3 TRAIN 1204 -0.074 -0.114 26.4 -13723 | B3 VAL  596 +0.047 +0.005 47.8   +313')
    print(f'\n  EARLIEST DECISION: min entry_m = {int(pre["B2"].entry_m.min())} '
          f'(break bar closes at {int(pre["B2"].entry_m.min())} ET-min = '
          f'{int(pre["B2"].entry_m.min())//60:02d}:{int(pre["B2"].entry_m.min())%60:02d}); '
          f'every 09:35 field precedes it.  pre-10:00 share of B2 = {(pre["B2"].entry_m < 600).mean():.1%}')

    # ---------------- attach the fields ---------------------------------------------------
    df = pd.read_csv(f'{D}/day_fields.csv', dtype={'day': str}).set_index('day')
    per_min, per_day = idx_tape()
    for nm in pre:
        x = pre[nm].merge(per_min, on=['day', 'break_m'], how='left')
        x.index = pre[nm].index
        for c in df.columns:
            if c not in x.columns:
                x[c] = x.day.map(df[c])
        for c in per_day.columns:
            x[c] = x.day.map(per_day[c])
        pre[nm] = x
    B0, B2, B3 = pre['B0'], pre['B2'], pre['B3']
    trd = B2.drop_duplicates('day').set_index('day')
    trd = trd[trd.split == 'TRAIN']

    def q(col, qs):
        return pd.to_numeric(trd[col], errors='coerce').quantile(qs).values

    # ---------------- A: pre-09:35 day gates ----------------------------------------------
    gap_med = float(pd.to_numeric(trd.spy_gap_pct, errors='coerce').abs().median())
    r5q = q('spy_r5_pct', [2 / 3]); svq = q('spy_vol20', [1 / 3]); qvq = q('qqq_vol20', [1 / 3])
    dow_tr = B2[B2.split == 'TRAIN'].groupby('dow').rr.mean()
    worst_dow = int(dow_tr.idxmin())
    print(f'\n  TRAIN edges printed BEFORE scoring: |spy_gap| median {gap_med:.3f}% | '
          f'spy_r5 top-tercile edge {r5q[0]:+.3f}% | spy_vol20 bot edge {svq[0]:.2f} | '
          f'qqq_vol20 bot edge {qvq[0]:.2f} | weekday gross ' +
          ' '.join(f'{d}:{v:+.3f}' for d, v in dow_tr.items()) + f' -> worst {worst_dow}')

    def A_masks(x):
        return {
            'A1 spy_gap>0': x.spy_gap_pct > 0,
            'A2 |spy_gap|<=med': x.spy_gap_pct.abs() <= gap_med,
            'A3 spy_r5>0 (0935)': x.spy_r5_pct > 0,
            'A4 spy_r5 TOP terc': x.spy_r5_pct > r5q[0],
            'A5 qqq_gap>0': x.qqq_gap_pct > 0,
            'A6 qqq_r5>0 (0935)': x.qqq_r5_pct > 0,
            'A7 qqq_imb not sell-side': x.qqq_imb_side != 'A',
            'A8 qqq_imb BUY': x.qqq_imb_side == 'B',
            'A9 spy_pm_ret>0': x.spy_pm_ret_pct > 0,
            'A10 spy_vol20 BOT terc': x.spy_vol20 <= svq[0],
            'A11 qqq_vol20 BOT terc': (x.qqq_vol20 <= qvq[0]) | x.qqq_vol20.isna(),   # fail open
            f'A12 skip weekday {worst_dow}': x.dow != worst_dow,
            'A13 spy_r5>0 AND qqq_r5>0': (x.spy_r5_pct > 0) & (x.qqq_r5_pct > 0),
            'A14 gap>0 AND r5>0': (x.spy_gap_pct > 0) & (x.spy_r5_pct > 0),
        }

    seps = []
    for bn, base in (('B2', B2), ('B0', B0)):
        print(f'\n== A. PRE-09:35 DAY GATES on {bn} ==\n' + HDR + '\n' + SEP)
        for nm, mk in A_masks(base).items():
            m = mk.fillna(False)
            show(f'{nm} [{bn}]', S.apply_book(base[m], 12, 4))
            seps.append((f'{nm} [{bn}]', sep_of(base[m], base[~m])))

    # ---------------- B: first-5-minute SPY signals ---------------------------------------
    B2['b1_rng_atr'] = B2.spy_b1_rng / B2.spy_atr20
    B0['b1_rng_atr'] = B0.spy_b1_rng / B0.spy_atr20
    B2['b1_vol_x'] = B2.spy_b1_vol / B2.spy_b1_vol20
    B0['b1_vol_x'] = B0.spy_b1_vol / B0.spy_b1_vol20
    t2 = B2.drop_duplicates('day'); t2 = t2[t2.split == 'TRAIN']
    raq = t2.b1_rng_atr.quantile([1 / 3, 2 / 3]).values
    vxq = t2.b1_vol_x.quantile([1 / 3, 2 / 3]).values
    print(f'\n  TRAIN edges: b1_range/ATR20 terciles {raq.round(3)} | b1_vol/20d terciles {vxq.round(3)}')

    def F5M_masks(x):
        go = np.sign(x.spy_gap_pct) == np.sign(x.spy_r5_pct)
        return {
            'F5M-1 b1 ret>0': x.spy_b1_ret > 0,
            'F5M-2 b1rng/ATR TOP': x.b1_rng_atr > raq[1],
            'F5M-3 b1rng/ATR BOT': x.b1_rng_atr <= raq[0],
            'F5M-4 b1vol/20d TOP': x.b1_vol_x > vxq[1],
            'F5M-5 b1vol/20d BOT': x.b1_vol_x <= vxq[0],
            'F5M-6 b1 closepos>=.5': x.spy_b1_pos >= 0.5,
            'F5M-7 gap-and-GO': go,
            'F5M-8 gap-FADE': ~go,
        }

    print(f'\n== B. FIRST-5-MIN SPY SIGNALS on B2 ==\n' + HDR + '\n' + SEP)
    for nm, mk in F5M_masks(B2).items():
        m = mk.fillna(False) if hasattr(mk, 'fillna') else pd.Series(mk, index=B2.index).fillna(False)
        show(f'{nm} [B2]', S.apply_book(B2[m], 12, 4))
        seps.append((f'{nm} [B2]', sep_of(B2[m], B2[~m])))
    for nm in ('F5M-1 b1 ret>0', 'F5M-7 gap-and-GO'):
        mk = F5M_masks(B0)[nm]
        m = mk.fillna(False) if hasattr(mk, 'fillna') else pd.Series(mk, index=B0.index).fillna(False)
        show(f'{nm} [B0]', S.apply_book(B0[m], 12, 4))

    # ---------------- C: the T-ladder ------------------------------------------------------
    print('\n== C. T-LADDER — time-segmented index gates (the owner curve) ==')
    print('   rule: gate from bars closing <= g, applied ONLY to signals with entry_m >= g+1;'
          ' earlier signals are TAKEN UNGATED.')
    curve = []
    for bn, base, ladder in (('B2', B2, LADDER), ('B0', B0, LADDER[:1])):
        print(f'\n-- {bn} --\n' + HDR + '\n' + SEP)
        for idx in ('spy', 'qqq'):
            for g, lab in GATE_TIMES:
                col = f'{idx}_ret_{lab}'
                late = base.entry_m >= g + 1
                for tn, thr in ladder:
                    ok = pd.to_numeric(base[col], errors='coerce') > thr
                    m = (~late) | (late & ok.fillna(False))
                    nm = f'T {idx.upper()} {lab} {tn} [{bn}]'
                    b = S.apply_book(base[m], 12, 4)
                    show(nm, b)
                    sp = sep_of(base[late & ok.fillna(False)], base[late & ~ok.fillna(False)])
                    seps.append((nm, sp))
                    # the gate's OWN contribution: the late arm alone, gated vs ungated
                    lg = S.apply_book(base[late & ok.fillna(False)], 12, 4)
                    lu = S.apply_book(base[late], 12, 4)
                    row = dict(base=bn, idx=idx, gate=lab, thr=tn)
                    for s_ in SPLITS:
                        w = S.week_stats(b, s_)
                        row[f'{s_}_green'] = w['green']; row[f'{s_}_total'] = w['total']
                        row[f'{s_}_perwk'] = w['per_wk']
                        row[f'{s_}_sepg'] = sp[s_]['g'] if sp[s_] else np.nan
                        row[f'{s_}_sept'] = sp[s_]['t'] if sp[s_] else np.nan
                        row[f'{s_}_septc'] = sp[s_]['tc'] if sp[s_] else np.nan
                        row[f'{s_}_late_gated'] = S.week_stats(lg, s_)['total']
                        row[f'{s_}_late_ungated'] = S.week_stats(lu, s_)['total']
                    curve.append(row)
                # the pre-gate arm, reported so nothing is silently dropped
                if idx == 'spy' and bn == 'B2':
                    show(f'  (pre-gate arm, entry_m<{g+1}) [B2]', S.apply_book(base[~late], 12, 4))
    pd.DataFrame(curve).to_csv(f'{D}/curve.csv', index=False)

    # ---------------- D: rolling causal form ----------------------------------------------
    print('\n== D. ROLLING CAUSAL FORM — one rule, every signal ==')
    trs = B2[B2.split == 'TRAIN']                      # signal-level, NOT day-deduped
    hq = [float(pd.to_numeric(trs.spy_disthod_pct, errors='coerce').quantile(2 / 3))]
    hqq = [float(pd.to_numeric(trs.qqq_disthod_pct, errors='coerce').quantile(2 / 3))]
    print(f'  TRAIN top-tercile edge spy_dist_hod_pct {hq[0]:+.4f}% | qqq {hqq[0]:+.4f}%')

    def ROLL_masks(x):
        d = {}
        for idx in ('spy', 'qqq'):
            for tn, thr in LADDER:
                d[f'ROLL-{idx.upper()} ret0930_sig {tn}'] = pd.to_numeric(
                    x[f'{idx}_ret_open_m'], errors='coerce') > thr
            e = hq[0] if idx == 'spy' else hqq[0]
            v = pd.to_numeric(x[f'{idx}_disthod_pct'], errors='coerce')
            d[f'HOD-{idx.upper()} TRAIN top terc'] = v > e
            for lv in (0.10, 0.20, 0.30):
                d[f'HOD-{idx.upper()} within {lv:.2f}%'] = v >= -lv
        return d

    print(HDR + '\n' + SEP)
    for nm, mk in ROLL_masks(B2).items():
        m = mk.fillna(False)
        show(f'{nm} [B2]', S.apply_book(B2[m], 12, 4))
        seps.append((f'{nm} [B2]', sep_of(B2[m], B2[~m])))
    for nm, mk in ROLL_masks(B0).items():
        if nm.endswith('sgn') or 'top terc' in nm:
            m = mk.fillna(False)
            show(f'{nm} [B0]', S.apply_book(B0[m], 12, 4))

    # ---------------- E: multi-day SPY context --------------------------------------------
    print('\n== E. MULTI-DAY SPY CONTEXT (known 09:29) ==')

    def M_masks(x):
        d = {'M1 spy2d>0': x.spy_ret_2d > 0}
        for lv in (0.5, 1.0, 1.5):
            d[f'M1 spy2d>=+{lv}%'] = x.spy_ret_2d >= lv
        for lv in (0.5, 1.0, 1.5):
            d[f'M1 spy2d<=-{lv}%'] = x.spy_ret_2d <= -lv
        for k in (1, 3, 5):
            d[f'M2 spy{k}d>0'] = x[f'spy_ret_{k}d'] > 0
        d['M3 two-UP'] = x.spy_path2 == 2
        d['M3 two-DOWN'] = x.spy_path2 == 0
        d['M3 mixed'] = x.spy_path2 == 1
        d['M4 prev close pos>=.5'] = x.spy_prev_close_pos >= 0.5
        d['M4 prev close pos<=.5'] = x.spy_prev_close_pos <= 0.5
        d['M4 inside day'] = x.spy_inside > 0
        d['M4 outside day'] = x.spy_outside > 0
        d['M-Q qqq2d>0'] = x.qqq_ret_2d > 0
        d['M-Q two-UP'] = x.qqq_path2 == 2
        d['M-Q two-DOWN'] = x.qqq_path2 == 0
        d['M-Q mixed'] = x.qqq_path2 == 1
        return d

    print(HDR + '\n' + SEP)
    for nm, mk in M_masks(B2).items():
        m = mk.fillna(False)
        show(f'{nm} [B2]', S.apply_book(B2[m], 12, 4))
        seps.append((f'{nm} [B2]', sep_of(B2[m], B2[~m])))
    for nm in ('M1 spy2d>0', 'M3 two-UP', 'M3 two-DOWN', 'M-Q qqq2d>0'):
        m = M_masks(B0)[nm].fillna(False)
        show(f'{nm} [B0]', S.apply_book(B0[m], 12, 4))

    # ---------------- F: interactions ------------------------------------------------------
    print('\n== F. INTERACTIONS (exactly the declared five) ==')
    t2_mask = lambda x: (x.entry_m < 601) | (pd.to_numeric(x.spy_ret_1000, errors='coerce') > 0)
    print(HDR + '\n' + SEP)
    show('F2 T2(SPY 1000 sgn) x spread<=8%R [B3]', S.apply_book(B3[t2_mask(B3).fillna(False)], 12, 4))
    m3 = t2_mask(B2).fillna(False) & (B2.rv_profile >= 5)
    show('F3 T2 x rv>=5 [B2]', S.apply_book(B2[m3], 12, 4))
    print('\n  F4 (owner M5) — is the multi-day separation concentrated in the FIRST 30 MINUTES?')
    print('| M rule                         | window      | n kept | n rej | gross sep | net sep |   t   | split |')
    for nm in ('M1 spy2d>0', 'M3 two-UP', 'M4 prev close pos>=.5'):
        mk = M_masks(B2)[nm].fillna(False)
        for wlab, wm in (('0935-1000', B2.entry_m < 600), ('1000-1400', B2.entry_m >= 600)):
            k = B2[mk & wm]; r = B2[~mk & wm]
            o = sep_of(k, r)
            for sp in SPLITS:
                z = o[sp]
                if z:
                    print(f'| {nm:<30s} | {wlab:11s} | {z["nk"]:6d} | {z["nr"]:5d} | {z["g"]:+9.3f} | '
                          f'{z["d"]:+7.3f} | {z["t"]:+5.2f} | {sp} |')

    # ---------------- separation table -----------------------------------------------------
    print('\n== SEPARATION (kept-minus-rejected, PRE-BOOK). t = iid, tc = DAY-CLUSTERED (the honest one) ==')
    print('| gate                               |' + ''.join(
        f'  {s} gr |   net |     t |    tc |  n_kp | days |' for s in SPLITS))
    rows = []
    for nm, o in seps:
        print(fmt_sep(nm, o))
        r = dict(cell=nm)
        for sp in SPLITS:
            z = o[sp]
            for k in ('g', 'd', 't', 'tc', 'nk', 'dk'):
                r[f'{sp}_{k}'] = z[k] if z else np.nan
        rows.append(r)
    pd.DataFrame(rows).to_csv(f'{D}/separation.csv', index=False)
    cf = pd.DataFrame(CELLS); cf.to_csv(f'{D}/cells.csv', index=False)
    print(f'\ncells scored: {cf.cell.nunique()} unique names x {len(SPLITS)} splits = {len(cf)} rows')

    # ---------------- count-matched permutation null on EVERY cell -------------------------
    print('\n== COUNT-MATCHED PERMUTATION NULL (2,000 draws, per-week pick count fixed) ==')
    print('| cell | split | observed green % | null mean | [p5, p95] | outside? |')
    nulls = []
    for nm, b in BOOKS.items():
        for sp in SPLITS:
            obs, mu, p5, p95 = S.null_band(b, sp)
            if obs != obs:
                continue
            out = 'ABOVE' if obs > p95 else ('below' if obs < p5 else 'inside')
            nulls.append(dict(cell=nm, split=sp, obs=obs, mu=mu, p5=p5, p95=p95, outside=out))
            if out != 'inside':
                print(f'| {nm} | {sp} | {obs:.1f} | {mu:.1f} | [{p5:.1f}, {p95:.1f}] | {out} |')
    nf = pd.DataFrame(nulls)
    nf.to_csv(f'{D}/nulls.csv', index=False)
    print(f'  nulls run: {len(nf)} cell x split; outside their band: '
          f'{(nf.outside != "inside").sum()} (ABOVE {(nf.outside == "ABOVE").sum()}, '
          f'below {(nf.outside == "below").sum()})')
    both = nf.pivot(index='cell', columns='split', values='outside')
    if set(SPLITS).issubset(both.columns):
        ab = both[(both.TRAIN == 'ABOVE') & (both.VAL == 'ABOVE')]
        print(f'  cells ABOVE their null band on BOTH splits: {len(ab)}'
              + (': ' + ', '.join(ab.index) if len(ab) else ''))


if __name__ == '__main__':
    main()
