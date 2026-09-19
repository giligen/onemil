#!/usr/bin/env python3
"""hod_frames2 — THE bar pass for F5 (retest), F6 (absorption) and F9 (signal-minute cohort fields).

Emits `breaks2.csv`: ONE ROW PER QUALIFYING BREAK (not just the first) on every symbol-day that
carries at least one qualifying break in TRAIN+VAL, with the B2 stop/exit walk and every field
declared in PREREG.md §1.  All fields are computed from bars 0..i (at or before the break bar's
close); the decision is acted on at m[i+1]'s open.

TRAIN + VAL only -- TEST is sealed (`FREEZE.md`).  Read-only on every DB.  Resumable per day.
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
from trading.hod_break import HodBreakParams, profile_fraction   # noqa: E402
from pass2 import load_bars                                       # noqa: E402

D = f'{ROOT}/research/mature_method/hod_frames2'
P = HodBreakParams()
OPEN_M, EOD_M, LAST_M = 570, 955, 930
K_N, SLIP = 5, 0.001
SHELF = 0.005                      # +/- 0.5 % of the level
DAY_LO, DAY_HI = '2025-01-02', '2026-05-31'     # TRAIN+VAL -- TEST sealed

COLS = ['day', 'symbol', 'entry_m', 'break_m', 'level', 'open_px', 'next_open', 'dist_open_pct',
        'rv_profile', 'adv20', 'cumv', 'cum_dollar', 'bar_vol', 'n_break', 'n_prior', 'fill_capped',
        'stop_n', 'r_pct_n', 'rr_n', 'why_n', 'exit_m_n', 'hod_age_bars',
        'prev_stopped', 'prev_back5', 'prev_back15',
        'shelf_vol', 'shelf_bars', 'shelf_share', 'exp5_n',
        'rng_sig', 'rng_30', 'rng_first30', 'rng_day']


def vwalk(o, h, l, c, m, s0, stop, target):
    """The shipped exit walk, vectorised. Priority eod -> stop -> target, from bar s0 = i+2."""
    n = len(o)
    if s0 >= n:
        return n - 1, float(c[-1]), 'eod'
    eod = m[s0:] >= EOD_M
    hs = l[s0:] <= stop
    ht = c[s0:] >= target
    any_ = eod | hs | ht
    if not any_.any():
        return n - 1, float(c[-1]), 'eod'
    j = int(np.argmax(any_)); k = s0 + j
    if eod[j]:
        return k, float(o[k]), 'eod'
    if hs[j]:
        return k, float(min(stop, o[k]) * (1.0 - SLIP)), 'stop'
    return k, float(target), 'target'


def block_expansions(m, h):
    """Cumulative count of completed 5-minute blocks that made a NEW 5-minute high.
    Returns (bidx, exp_cum) where exp_cum[b] = expansions among blocks 0..b."""
    b = (m - OPEN_M) // 5
    nb = int(b.max()) + 1
    hi = np.full(nb, -np.inf)
    np.maximum.at(hi, b, h)
    run = -np.inf
    exp_cum = np.zeros(nb, dtype=np.int32)
    cnt = 0
    for k in range(nb):
        if hi[k] > run:
            if run > -np.inf:
                cnt += 1
            run = hi[k]
        exp_cum[k] = cnt
    return b, exp_cum


def main():
    pop = pd.read_csv(f'{ROOT}/research/mature_method/hod_filter_stack/pop.csv',
                      usecols=['day', 'symbol'], dtype={'day': str, 'symbol': str})
    pop = pop[(pop.day >= DAY_LO) & (pop.day <= DAY_HI)].drop_duplicates()
    u = pd.read_csv(f'{ROOT}/research/bf_zero/universe.csv',
                    usecols=['symbol', 'bar_date', 'adv20'], dtype={'symbol': str},
                    keep_default_na=False, na_values=[''])
    u = u.rename(columns={'bar_date': 'day'})
    u['adv20'] = pd.to_numeric(u.adv20, errors='coerce')
    need = pop.merge(u.drop_duplicates(['day', 'symbol']), on=['day', 'symbol'], how='left')
    print(f'symbol-days to walk {len(need)} over {need.day.nunique()} sessions', flush=True)

    st = f'{D}/walk_state.json'
    state = json.load(open(st)) if os.path.exists(st) else {'done': []}
    done = set(state['done'])
    days = [d for d in sorted(need.day.unique()) if d not in done]
    NE = {d: g for d, g in need.groupby('day')}
    out = f'{D}/breaks2.csv'

    for nd, day in enumerate(days):
        sub = NE[day]
        bars = load_bars(day, sub.symbol.tolist())
        rows = []
        for r in sub.itertuples():
            gg = bars.get(r.symbol)
            if gg is None:
                continue
            rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
            if len(rth) < 10:
                continue
            o, h, l, cl, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v'))
            m = rth.m.values.astype(int)
            n = len(h)
            o0 = float(o[0])
            if not (o0 > 0):
                continue
            hod = np.maximum.accumulate(h)
            run_lo = np.minimum.accumulate(l)
            cumv = np.cumsum(v)
            cumd = np.cumsum((h + l + cl) / 3.0 * v)
            adv = float(r.adv20) if r.adv20 == r.adv20 and r.adv20 and r.adv20 > 0 else np.nan
            bidx, exp_cum = block_expansions(m, h)
            i_f30 = int(np.searchsorted(m, 600, side='right')) - 1
            rng_first30 = float((hod[i_f30] - run_lo[i_f30]) / o0 * 100) if i_f30 >= 0 else np.nan
            rng_day = float((h.max() - l.min()) / o0 * 100)
            prior = []                       # (break idx, level, exit idx, why)
            nb_cand = 0
            for i in range(K_N + 1, n):
                if int(m[i]) > LAST_M:
                    break
                level = float(hod[i - 1])
                if h[i] < level or level < o0 * 1.05 or level < 1.0 or i + 1 >= n:
                    continue
                nb_cand += 1
                rv = float(cumv[i]) / (adv * profile_fraction(int(m[i]))) if adv == adv else np.nan
                if not (rv == rv and rv >= 1.0) or int(m[i + 1]) > LAST_M + 1:
                    continue
                stop = float(np.min(l[i - K_N:i]))
                nxt = float(o[i + 1])
                if stop >= nxt:
                    rp = rr = np.nan; why = ''; ex_m = -1; ek = -1
                else:
                    Rd = nxt - stop
                    ek, px, why = vwalk(o, h, l, cl, m, i + 2, stop, nxt + 2.0 * Rd)
                    rp = Rd / nxt * 100.0; rr = (px - nxt) / Rd; ex_m = int(m[ek])
                # ---- F5: the immediately preceding qualifying break ----------------------
                if prior:
                    q_i, q_lv, q_ek, q_why = prior[-1]
                    pstop = int(q_why == 'stop' and 0 <= q_ek <= i)
                    hi5 = min(q_i + 5, i); hi15 = min(q_i + 15, i)
                    pb5 = int(hi5 > q_i and bool((cl[q_i + 1:hi5 + 1] < q_lv).any()))
                    pb15 = int(hi15 > q_i and bool((cl[q_i + 1:hi15 + 1] < q_lv).any()))
                else:
                    pstop = pb5 = pb15 = 0
                # ---- F6: the shelf -------------------------------------------------------
                lo_b, hi_b = level * (1 - SHELF), level * (1 + SHELF)
                msk = (l[:i] <= hi_b) & (h[:i] >= lo_b)
                sv = float(v[:i][msk].sum()); sb = int(msk.sum())
                # ---- F9 --------------------------------------------------------------
                b_i = int(bidx[i])
                e5 = int(exp_cum[b_i - 1]) if b_i >= 1 else 0
                rng_sig = float((hod[i] - run_lo[i]) / o0 * 100)
                i30 = int(np.searchsorted(m, int(m[i]) - 30, side='right')) - 1
                rng30 = float((hod[i30] - run_lo[i30]) / o0 * 100) if i30 >= 0 else np.nan
                jh = int(np.argmax(h[:i])) if i > 0 else 0
                rows.append((day, r.symbol, int(m[i + 1]), int(m[i]), level, o0, nxt,
                             (level / o0 - 1.0) * 100.0, rv, adv, float(cumv[i]), float(cumd[i]),
                             float(v[i]), nb_cand - 1, len(prior),
                             int(nxt <= level * (1.0 + P.cap)),
                             stop, rp, rr, why, ex_m, i - jh,
                             pstop, pb5, pb15,
                             sv, sb, sv / adv * 100.0 if adv == adv else np.nan, e5,
                             rng_sig, rng30, rng_first30, rng_day))
                prior.append((i, level, ek, why))
        if rows:
            pd.DataFrame(rows, columns=COLS).to_csv(out, mode='a', header=not os.path.exists(out),
                                                    index=False)
        done.add(day)
        json.dump({'done': sorted(done)}, open(st, 'w'))
        if nd % 10 == 0 or nd == len(days) - 1:
            print(f'  [{nd + 1}/{len(days)}] {day}  rows+{len(rows)}', flush=True)
    print('PASS DONE', flush=True)


if __name__ == '__main__':
    main()
