#!/usr/bin/env python3
"""FINAL multi-day stage -- 11 cells:

  F5  52-week high (George-Hwang 2004)                  F5-LO, F5-LS      (2)
  A2  low short interest, long side (BHJ 2010 JFE 96(1))A2-LO, A2-LS      (2)
  A3  net share issuance (Pontiff-Woodgate 2008)        A3-LO, A3-LS      (2)
  A4  dividend-month premium (Hartzmark-Solomon 2013)   A4-LO, A4-LS      (2)
  F1  PEAD / SUE -- DECLARED NULL-REPLICATION           F1-LO, F1-LS      (2)
  F6  overnight vs intraday (Lou-Polk-Skouras 2019)     F6-LS             (1)

Standing corrections inherited WHOLESALE from REPORT_F2_A1.md S9 and REPORT_F4_F3.md
(not re-litigated here):
  * ADV20$ on the RAW panel                                  -> Panel.adv
  * ex-top-1% AND ex-top-5% are mandatory rebuilt columns    -> score_cell()
  * impact scales with the ACTUAL position count             -> make_trades()
  * costs are never charged to the benchmark                 -> bench series gross
  * the benchmark excludes the book's own decile             -> clean contrast
  * sealed splits (a trade whose exit leaves the split leaves it); VAL both ways
  * Newey-West t at lag = the hold length, beside the raw t  -> nw_t()
  * the GROSS/raw column is NOT quotable (100%-survivor panel)
  * a -100% delisting haircut arm on every cell              -> haircut stats

Env arms:  MD_MIN_ADV (default 1e6) | MD_FLAT_COSTS=1 (5 bps/side) | MD_ARM (out suffix)
           A2_DISS_BDAYS (default 8) | OPEN_TEST=1 (refused unless FREEZE.md exists)
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

D = '/home/ec2-user/onemil/research/multiday/data'
OUTBASE = '/home/ec2-user/onemil/research/multiday/out_final'

SPLITS = {'TRAIN': ('2016-01-01', '2021-12-31'),
          'VAL': ('2022-01-01', '2023-12-31'),
          'TEST': ('2024-01-01', '2026-09-18')}

BOOK_USD = 66_000.0
N_SLOTS = 20
MIN_RAW_CLOSE = 5.0
MIN_ADV = float(os.environ.get('MD_MIN_ADV', 1_000_000.0))
FLAT_COSTS = os.environ.get('MD_FLAT_COSTS') == '1'
ARM = os.environ.get('MD_ARM', '')
A2_DISS_BDAYS = int(os.environ.get('A2_DISS_BDAYS', 8))
SELL_FEE_BPS = 0.4
IMPACT_COEF_BPS = 10.0
FLAT_ARM_BPS = 5.0
BORROW_ANN = 0.003

FREEZE = '/home/ec2-user/onemil/research/multiday/FREEZE.md'
_OPEN = os.environ.get('OPEN_TEST') == '1' and os.path.exists(FREEZE)
SPLIT_ORDER = ('TRAIN', 'VAL', 'TEST') if _OPEN else ('TRAIN', 'VAL')
OUT = OUTBASE + ('_test' if _OPEN else '') + (('_' + ARM) if ARM else '')


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


# ------------------------------------------------------------------ panel


class Panel:
    def __init__(self):
        z = np.load(f'{D}/panel_f3f4.npz', allow_pickle=True)
        self.close = z['close_adj']
        self.raw = z['close_raw']
        self.adv = z['adv_raw']
        self.symbols = [str(s) for s in z['symbols']]
        self.sessions = pd.to_datetime([str(s) for s in z['sessions']])
        self.sic2 = np.array([str(v) for v in z['sic2']], dtype=object)
        self.etb = z['etb'].astype(bool)
        del z
        w = np.load(f'{D}/panel_final.npz')
        self.open = w['open_adj']
        self.splitcum = w['splitcum']
        del w
        self.sidx = {s: i for i, s in enumerate(self.symbols)}
        self.n_s, self.n_d = self.close.shape
        with np.errstate(invalid='ignore', divide='ignore'):
            r = self.close[:, 1:] / self.close[:, :-1] - 1.0
        self.ret = np.zeros_like(self.close)
        self.ret[:, 1:] = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        self.spy = self.sidx['SPY']
        fin = np.isfinite(self.close)
        self.last_fin = np.where(fin.any(axis=1),
                                 fin.shape[1] - 1 - np.argmax(fin[:, ::-1], axis=1), -1)
        # --- D1, found by the independent rebuild: the ADJUSTED panel contains reverse
        # splits applied FORWARD but never back-propagated (DCTH 2020-05-01 0.0999 -> 7.66,
        # PFH +3,740%, CRC +1,072%).  DATA.md gap 10 quantified the Chapter-11 ticker-recycling
        # jumps and showed the $5/$1M gates excluded them; these are a DIFFERENT population --
        # 99 of the 152 break days clear the RAW $5 gate.  The mechanism that matters for F5:
        # after an unadjusted reverse split the trailing 252-session adjusted max IS the
        # post-break level, so nearness ~ 1.0 and the name is a guaranteed decile-10 member
        # for a year.  A "tainted" flag (event .. event+252 sessions) is carried on every
        # trade so every cell can be re-scored without them.
        with np.errstate(invalid='ignore', divide='ignore'):
            rr = self.close[:, 1:] / self.close[:, :-1] - 1.0
        jump = np.zeros(self.close.shape, dtype=bool)
        jump[:, 1:] = np.isfinite(rr) & (rr > 2.0)
        nxt = np.zeros_like(jump)
        nxt[:, :-1] = jump[:, 1:]
        with np.errstate(invalid='ignore', divide='ignore'):
            rev = np.zeros(self.close.shape, dtype=bool)
            rev[:, :-1] = np.isfinite(rr) & (rr < -0.5)
        perm = jump.copy()
        perm[:, :-1] &= ~rev[:, :-1]           # a jump that REVERTS the next day is a bad print
        self.jump_events = perm
        taint = np.zeros(self.close.shape, dtype=bool)
        ii, jj = np.nonzero(perm)
        for a_, b_ in zip(ii, jj):
            taint[a_, b_:min(b_ + 253, self.close.shape[1])] = True
        self.taint = taint
        log(f'unreverted >+200% adjusted sessions: {int(perm.sum()):,} on '
            f'{int(perm.any(axis=1).sum()):,} symbols; tainted symbol-sessions '
            f'{taint.mean():.4%}')
        del rr, jump, nxt, rev, perm
        self.elig = ((np.isfinite(self.raw) & (self.raw >= MIN_RAW_CLOSE))
                     & (np.isfinite(self.adv) & (self.adv >= MIN_ADV))
                     & np.isfinite(self.close))
        mdf = pd.DataFrame({'d': self.sessions})
        mdf['ym'] = mdf['d'].dt.to_period('M')
        self.month_end = mdf.groupby('ym').tail(1).index.to_numpy()
        self.sess_np = self.sessions.values
        log(f'panel {self.n_s} symbols x {self.n_d} sessions | month-ends {len(self.month_end)}')

    def eligible(self, t):
        return self.elig[:, t]


# ------------------------------------------------------------------ costs / stats


def cost_bps(adv, order_usd, short=False, hold_days=0):
    if FLAT_COSTS:
        c = np.full(np.shape(adv), 2 * FLAT_ARM_BPS, dtype=float)
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            frac = np.minimum(order_usd / np.maximum(0.01 * adv, 1.0), 1.0)
        c = 2.0 * IMPACT_COEF_BPS * frac + SELL_FEE_BPS
    if short:
        c = c + BORROW_ANN * 1e4 * (np.asarray(hold_days, dtype=float) / 252.0)
    return c


def decile(vals, n=10):
    """Cross-sectional decile 1..n (1 = lowest). Ties broken by position (argsort, stable)."""
    order = np.argsort(np.argsort(vals, kind='stable'), kind='stable')
    return np.floor(order * n / len(vals)).astype(int) + 1


def monthly(s):
    return (1.0 + s).resample('ME').prod() - 1.0


def nw_t(x, lag):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 6:
        return np.nan
    e = x - x.mean()
    s = (e * e).sum() / n
    for L in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - L / (lag + 1.0)) * (e[L:] * e[:-L]).sum() / n
    return x.mean() / np.sqrt(s / n) if s > 0 else np.nan


def stats(m, split, hold_months=1):
    a, b = SPLITS[split]
    mm = m[(m.index >= a) & (m.index <= b)]
    mm = mm[mm != 0]
    if len(mm) < 6:
        return dict(n_months=len(mm), mean_bps=np.nan, t=np.nan, nw_t=np.nan, pct_pos=np.nan,
                    ex_jan_bps=np.nan, jan_bps=np.nan, mde_bps=np.nan, mdd_pct=np.nan,
                    worst_bps=np.nan)
    mu, sd, n = mm.mean(), mm.std(ddof=1), len(mm)
    ex = mm[mm.index.month != 1]
    jan = mm[mm.index.month == 1]
    eq = (1 + mm).cumprod()
    return dict(n_months=n, mean_bps=mu * 1e4,
                t=(mu / (sd / np.sqrt(n))) if sd > 0 else np.nan,
                nw_t=nw_t(mm.to_numpy(), max(1, int(hold_months))),
                pct_pos=100.0 * (mm > 0).mean(),
                ex_jan_bps=ex.mean() * 1e4 if len(ex) else np.nan,
                jan_bps=jan.mean() * 1e4 if len(jan) else np.nan,
                mde_bps=2.0 * sd / np.sqrt(n) * 1e4,
                mdd_pct=100.0 * (eq / eq.cummax() - 1).min(), worst_bps=mm.min() * 1e4)


def sealed(t, split):
    a, b = SPLITS[split]
    return t[(t['entry_dt'] >= a) & (t['exit_dt'] <= b)]


def complete(t, split):
    a, b = SPLITS[split]
    return t[(t['entry_dt'] >= a) & (t['entry_dt'] <= b)]


def boot_p(m, nboot=5000, block=3, seed=7):
    x = np.asarray(m, dtype=float)
    n = len(x)
    if n < 8:
        return np.nan
    rng = np.random.default_rng(seed)
    e = x - x.mean()
    nb = int(np.ceil(n / block))
    obs = abs(x.mean())
    cnt = 0
    for _ in range(nboot):
        st = rng.integers(0, max(1, n - block + 1), size=nb)
        s = np.concatenate([e[i:i + block] for i in st])[:n]
        if abs(s.mean()) >= obs:
            cnt += 1
    return (cnt + 1) / (nboot + 1)


# ------------------------------------------------------------------ trades



def concurrent_order_usd(p: Panel, a, b):
    """$ per position = BOOK_USD / the number of positions the academic column actually
    holds on that trade's entry day.

    REPORT_F2_A1.md S9 defect 3: charging every trade a $3,300 / 20-slot order size is
    fiction for an overlapping academic portfolio that holds hundreds of names at once --
    it inflates the impact charge by the overlap factor.  Counting positions open on the
    entry day gets the 6-cohort F5 book and the 60-session-overlap F1 book right, where a
    per-rebalance count would over-charge them 6x and 12x respectively.
    """
    d = np.zeros(p.n_d + 2)
    np.add.at(d, a, 1.0)
    np.add.at(d, np.minimum(b + 1, p.n_d + 1), -1.0)
    openc = np.cumsum(d)[:p.n_d]
    n = np.maximum(openc[a], 1.0)
    return BOOK_USD / n, n


def make_trades(p: Panel, sig: pd.DataFrame, anchors, sel, leg, label, hold_k=1,
                haircut=False):
    """(rebalance, symbol) selection -> trades. entry = close(anchor+1), exit = close(anchor[k+hold_k]+1).

    `haircut=True` forces a -100% gross return on any trade whose symbol stops being
    priced inside the hold (the delisting arm; the panel's default convention lets such a
    position break even, which REPORT_F2_A1 S9 defect 1 named as a survivorship inflation).
    """
    sub = sig[sel].copy()
    if not len(sub):
        return pd.DataFrame()
    k = sub['k'].to_numpy()
    if k.max() + hold_k >= len(anchors):
        sub = sub[k + hold_k < len(anchors)]
        k = sub['k'].to_numpy()
    if not len(sub):
        return pd.DataFrame()
    sub['a'] = anchors[k] + 1
    sub['b'] = anchors[k + hold_k] + 1
    sub = sub[(sub['a'] < p.n_d) & (sub['b'] < p.n_d) & (sub['b'] > sub['a'])]
    si, a, b = sub['si'].to_numpy(), sub['a'].to_numpy(), sub['b'].to_numpy()
    ca, cb = p.close[si, a], p.close[si, b]
    good = np.isfinite(ca) & np.isfinite(cb) & (ca > 0)
    sub = sub[good]
    si, a, b = si[good], a[good], b[good]
    with np.errstate(invalid='ignore', divide='ignore'):
        sub['gross'] = p.close[si, b] / p.close[si, a] - 1.0
    order_usd, npos = concurrent_order_usd(p, a, b)
    sub['n_concurrent'] = npos
    hold = (b - a).astype(float)
    sub['order_usd'] = order_usd
    sub['cost_bps'] = cost_bps(p.adv[si, a], order_usd, short=(leg < 0), hold_days=hold)
    sub['leg'] = leg
    sub['hold'] = hold
    sub['ends_mid_hold'] = p.last_fin[si] < b
    sub['tainted'] = p.taint[si, a] | np.array(
        [p.jump_events[s_, a_ + 1:b_ + 1].any() for s_, a_, b_ in zip(si, a, b)])
    sub['contrib'] = leg * sub['gross'].to_numpy()
    sub['entry_dt'] = p.sessions[a]
    sub['exit_dt'] = p.sessions[b]
    sub['label'] = label
    if haircut and sub['ends_mid_hold'].any():
        sub.loc[sub['ends_mid_hold'], 'gross'] = -1.0
    return sub


def daily_series(p: Panel, t, charge=True, haircut=False):
    """Equal-weighted overlapping daily portfolio return; costs on entry and exit days only."""
    num = np.zeros(p.n_d)
    den = np.zeros(p.n_d)
    if t is None or not len(t):
        return pd.Series(num, index=p.sessions)
    si, a, b = t['si'].to_numpy(), t['a'].to_numpy(), t['b'].to_numpy()
    leg = t['leg'].to_numpy(dtype=float)
    cost = (t['cost_bps'].to_numpy() / 1e4) if charge else np.zeros(len(t))
    emh = t['ends_mid_hold'].to_numpy() if haircut else np.zeros(len(t), dtype=bool)
    last = p.last_fin
    for i in range(len(t)):
        idx = np.arange(a[i] + 1, b[i] + 1)
        if not len(idx):
            continue
        r = p.ret[si[i], idx].astype(np.float64) * leg[i]
        if emh[i]:
            j = int(last[si[i]]) - a[i]                # first session with no price
            if 0 <= j < len(r):
                r[j:] = 0.0
                r[j] = -1.0 * leg[i]                   # the position goes to zero, not flat
        r[0] -= cost[i] * 0.5
        r[-1] -= cost[i] * 0.5
        np.add.at(num, idx, r)
        np.add.at(den, idx, 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        d = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
    return pd.Series(d, index=p.sessions)


def book_sim(p: Panel, t, rank_col, ascending=False):
    """$66K / 20 slots, first-come by entry session, ranked within the session, no refill."""
    if not len(t):
        return t
    t = t.sort_values(['a', rank_col], ascending=[True, ascending])
    free_at = np.zeros(N_SLOTS, dtype=int)
    taken = []
    for row in t.itertuples():
        j = int(np.argmin(free_at))
        if free_at[j] <= row.a:
            free_at[j] = row.b
            taken.append(row.Index)
    return t.loc[taken]


# ------------------------------------------------------------------ cell scoring


_BENCH_MEMO = {}


def bench_series(p, B, key):
    """Gross benchmark daily series, memoised -- the same benchmark frame is scored by
    several cells and it is the most expensive object in the stage."""
    if key not in _BENCH_MEMO:
        _BENCH_MEMO[key] = daily_series(p, B, charge=False)
    return _BENCH_MEMO[key]


def score_cell(p, name, long_t, short_t, bench_t, hold_months, rank_col='sig',
               ascending=False, bench_key=None):
    recs = []
    for split in SPLIT_ORDER:
        for cname, fn in (('sealed', sealed), ('complete', complete)):
            if cname == 'complete' and split == 'TRAIN':
                continue
            L = fn(long_t, split)
            B = fn(bench_t, split) if bench_t is not None and len(bench_t) else None
            S = fn(short_t, split) if short_t is not None and len(short_t) else None
            if len(L) < 20:
                continue
            ls = daily_series(p, L, charge=True)
            bs = (bench_series(p, B, (bench_key or name, split, cname)) if B is not None
                  else pd.Series(np.zeros(p.n_d), index=p.sessions))
            if S is not None and len(S):
                ss = daily_series(p, S, charge=True)
                cell = ls + ss
                exc = cell
                mL = monthly(ls)
                mS = monthly(ss)
                mB = monthly(bs)
                a0, b0 = SPLITS[split]
                mLs = mL[(mL.index >= a0) & (mL.index <= b0)]
                mSs = mS[(mS.index >= a0) & (mS.index <= b0)]
                mBs = mB[(mB.index >= a0) & (mB.index <= b0)]
                spread = mLs.mean() - (-mSs.mean())
                lls = float((mLs.mean() - mBs.mean()) / spread) if spread > 0 else None
            else:
                cell = ls
                exc = ls - bs
                lls = None
            st = stats(monthly(exc), split, hold_months)
            st_raw = stats(monthly(cell), split, hold_months)

            tail = {}
            allt = pd.concat([L] + ([S] if S is not None and len(S) else []))
            for q, lab in ((0.01, 'ex_top_1'), (0.05, 'ex_top_5')):
                thr = allt['contrib'].quantile(1 - q)
                ls2 = daily_series(p, L[L['contrib'] < thr], charge=True)
                if S is not None and len(S):
                    e2 = ls2 + daily_series(p, S[S['contrib'] < thr], charge=True)
                else:
                    e2 = ls2 - bs
                s2 = stats(monthly(e2), split, hold_months)
                tail[lab + '_bps'] = s2['mean_bps']
                tail[lab + '_t'] = s2['t']

            # SYMMETRIC tail trim (independent rebuild C4): the one-sided trim above drops
            # the top q% of the BOOK only and leaves the benchmark whole, which mechanically
            # drags a long-only excess negative.  Trimming BOTH sides at the same quantile is
            # the version that carries information about concentration rather than arithmetic.
            for q, lab in ((0.01, 'sym_top_1'), (0.05, 'sym_top_5')):
                thr = allt['contrib'].quantile(1 - q)
                ls3 = daily_series(p, L[L['contrib'] < thr], charge=True)
                if S is not None and len(S):
                    e3 = ls3 + daily_series(p, S[S['contrib'] < thr], charge=True)
                else:
                    bthr = B['contrib'].quantile(1 - q) if B is not None else np.inf
                    bs3 = daily_series(p, B[B['contrib'] < bthr], charge=False) \
                        if B is not None else bs
                    e3 = ls3 - bs3
                tail[lab + '_bps'] = stats(monthly(e3), split, hold_months)['mean_bps']

            # re-score without the unreverted-reverse-split cohort (rebuild finding D1)
            n_taint = int(L['tainted'].sum())
            if n_taint:
                lt = daily_series(p, L[~L['tainted']], charge=True)
                if S is not None and len(S):
                    et = lt + daily_series(p, S[~S['tainted']], charge=True)
                else:
                    bt = daily_series(p, B[~B['tainted']], charge=False) if B is not None else bs
                    et = lt - bt
                s_t = stats(monthly(et), split, hold_months)
                tail['ex_taint_bps'] = s_t['mean_bps']
                tail['ex_taint_t'] = s_t['t']
            else:
                tail['ex_taint_bps'] = st['mean_bps']
                tail['ex_taint_t'] = st['t']
            tail['taint_share'] = float(L['tainted'].mean())

            # -100% delisting haircut arm
            n_emh = int(L['ends_mid_hold'].sum()) + (int(S['ends_mid_hold'].sum())
                                                    if S is not None and len(S) else 0)
            if n_emh:
                lh = daily_series(p, L, charge=True, haircut=True)
                eh = (lh + daily_series(p, S, charge=True, haircut=True)) if (
                    S is not None and len(S)) else (lh - bs)
                hc = stats(monthly(eh), split, hold_months)['mean_bps']
            else:
                hc = st['mean_bps']

            bk = book_sim(p, L, rank_col, ascending=ascending)
            if len(bk) >= 20:
                bdaily = daily_series(p, bk, charge=True)
                a0, b0 = SPLITS[split]
                mb = monthly(bdaily)
                mbm = monthly(bs)
                sel = (mb.index >= a0) & (mb.index <= b0) & ((mb != 0) | (mbm != 0))
                alpha_m = (mb - mbm)[sel] * BOOK_USD
                bmu = alpha_m.mean()
                bsd = alpha_m.std(ddof=1) if len(alpha_m) > 1 else np.nan
                eq = (1 + (mb - mbm)[sel]).cumprod()
                book = dict(n=len(bk), usd_mo=bmu, raw_usd_mo=(mb[sel] * BOOK_USD).mean(),
                            t=(bmu / (bsd / np.sqrt(len(alpha_m)))) if bsd and bsd > 0 else np.nan,
                            mde_usd=2 * bsd / np.sqrt(len(alpha_m)) if bsd and bsd > 0 else np.nan,
                            pct_pos=100.0 * (alpha_m > 0).mean(), worst=alpha_m.min(),
                            mdd_usd=float((eq / eq.cummax() - 1).min() * BOOK_USD))
                wk = (pd.Timestamp(SPLITS[split][1]) - pd.Timestamp(SPLITS[split][0])).days / 7.0
                book['tr_wk'] = len(bk) / wk
                # independent rebuild C2: a month in which the 20-slot book holds NOTHING is
                # scored as a SHORT of the benchmark by the alpha basis.  Count them and
                # re-report the alpha over held months only.
                held = (mb[sel] != 0)
                book['empty_months'] = int((~held).sum())
                book['usd_mo_held'] = float(alpha_m[held.to_numpy()].mean()) if held.any() else np.nan
            else:
                book = dict(n=len(bk), usd_mo=np.nan, raw_usd_mo=np.nan, t=np.nan,
                            mde_usd=np.nan, pct_pos=np.nan, worst=np.nan, mdd_usd=np.nan,
                            tr_wk=0.0, empty_months=-1, usd_mo_held=np.nan)

            honest = float(allt['cost_bps'].mean())
            # break-even cost per round trip: the charge that zeroes the GROSS excess
            lsg = daily_series(p, L, charge=False)
            if S is not None and len(S):
                gexc = lsg + daily_series(p, S, charge=False)
            else:
                gexc = lsg - bs
            gm = stats(monthly(gexc), split, hold_months)['mean_bps']
            rt_per_month = 21.0 / max(float(L['hold'].mean()), 1.0)
            be = gm / rt_per_month if rt_per_month > 0 else np.nan

            rec = dict(cell=name, split=split, conv=cname,
                       n_trades=len(L) + (len(S) if S is not None and len(S) else 0),
                       raw_bps=st_raw['mean_bps'],
                       **{k: st[k] for k in ('mean_bps', 't', 'nw_t', 'pct_pos', 'ex_jan_bps',
                                             'jan_bps', 'mde_bps', 'mdd_pct', 'worst_bps',
                                             'n_months')},
                       **tail, lls=lls, honest_cost_bps=honest,
                       gross_excess_bps=gm, rt_per_month=rt_per_month,
                       breakeven_bps_per_rt=be,
                       breakeven_x_honest=(be / honest) if honest else np.nan,
                       ends_mid_hold=n_emh, haircut_bps=hc,
                       mean_hold_sessions=float(L['hold'].mean()),
                       book_n=book['n'], book_usd_mo=book['usd_mo'],
                       book_raw_usd_mo=book['raw_usd_mo'], book_t=book['t'],
                       book_mde_usd=book['mde_usd'], book_pct_pos=book['pct_pos'],
                       book_worst=book['worst'], book_mdd_usd=book['mdd_usd'],
                       tr_wk=book['tr_wk'], book_empty_months=book.get('empty_months', -1),
                       book_usd_mo_held=book.get('usd_mo_held', np.nan))
            if split == 'TRAIN' and cname == 'sealed':
                me = monthly(exc)
                me = me[(me.index >= SPLITS[split][0]) & (me.index <= SPLITS[split][1])]
                rec['boot_p'] = boot_p(me.replace(0, np.nan).dropna())
            recs.append(rec)
    return recs


# ------------------------------------------------------------------ F5: 52-week high


def f5_signals(p: Panel):
    """nearness = close(t) / max(close over the trailing 252 sessions ending t). Causal."""
    me = p.month_end
    rows = []
    for k in range(12, len(me) - 1):
        t = me[k]
        if t < 252:
            continue
        el = p.eligible(t)
        win = p.close[:, t - 251:t + 1]
        with np.errstate(invalid='ignore'):
            hi = np.nanmax(win, axis=1)
            near = p.close[:, t] / hi
        nobs = np.isfinite(win).sum(axis=1)
        ok = el & np.isfinite(near) & (nobs >= 200)
        idx = np.nonzero(ok)[0]
        if len(idx) < 100:
            continue
        rows.append(pd.DataFrame({'k': k, 'si': idx, 'sig': near[idx],
                                  'dec': decile(near[idx])}))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ A2: short interest


def a2_signals(p: Panel, sh: 'SharesPIT'):
    """Relative short interest at each FINRA dissemination date.

    Dissemination date is NOT in the FINRA dataset; it is derived as settlement +
    A2_DISS_BDAYS business days (FINRA Rule 4560 practice: reports are due 2 business
    days after the settlement date and are disseminated on roughly the 8th).  The decision
    close is the first session on or after that date; entry is the NEXT close.  Any error
    in the derived lag that makes it TOO SHORT would be a look-ahead, which is why the
    12-business-day arm exists.
    """
    si_df = pd.read_parquet(f'{D}/short_interest.parquet',
                            columns=['symbolCode', 'settlementDate',
                                     'currentShortPositionQuantity', 'averageDailyVolumeQuantity'])
    si_df = si_df.rename(columns={'symbolCode': 'symbol', 'settlementDate': 'settle',
                                  'currentShortPositionQuantity': 'si',
                                  'averageDailyVolumeQuantity': 'advq'})
    sset = set(p.sidx)
    si_df = si_df[si_df['symbol'].isin(sset)].copy()
    si_df['si_idx'] = si_df['symbol'].map(p.sidx).astype(int)
    settles = pd.DatetimeIndex(sorted(si_df['settle'].unique()))
    sess = p.sess_np
    amap = {}
    for st in settles:
        dd = st + pd.offsets.BDay(A2_DISS_BDAYS)
        j = int(np.searchsorted(sess, np.datetime64(dd), side='left'))
        if 0 < j < len(sess) - 2:
            amap[st] = j
    si_df['anchor'] = si_df['settle'].map(amap)
    si_df = si_df.dropna(subset=['anchor'])
    si_df['anchor'] = si_df['anchor'].astype(int)
    # one row per (symbol, settlement): FINRA can emit a row per market class
    si_df = (si_df.sort_values('si').drop_duplicates(['si_idx', 'anchor'], keep='last'))
    anchors_sorted = sorted(si_df['anchor'].unique())
    aidx = {a: i for i, a in enumerate(anchors_sorted)}
    anchors = np.array(anchors_sorted, dtype=int)
    log(f'  A2: {len(settles)} settlement dates -> {len(anchors)} dissemination anchors '
        f'({settles[0]:%Y-%m-%d} -> {settles[-1]:%Y-%m-%d}); lag {A2_DISS_BDAYS} bd')

    rows = []
    for a, cur in si_df.groupby('anchor', sort=True):
        idx = cur['si_idx'].to_numpy()
        keep = p.elig[idx, a]
        cur = cur[keep]
        if len(cur) < 100:
            continue
        idx = cur['si_idx'].to_numpy()
        shn = sh.shares_at(idx, p.sessions[a], a)
        rsi = cur['si'].to_numpy(dtype=float) / np.where(shn > 0, shn, np.nan)
        ok = np.isfinite(rsi) & (rsi > 0) & (rsi < 1.0)
        if ok.sum() < 100:
            continue
        sub = pd.DataFrame({'k': aidx[a], 'si': idx[ok], 'sig': rsi[ok]})
        sub['dec'] = decile(sub['sig'].to_numpy())
        rows.append(sub)
    return pd.concat(rows, ignore_index=True), anchors


class SharesPIT:
    """Point-in-time split-normalised share counts, indexed by panel symbol."""

    def __init__(self, p: Panel):
        uni = pd.read_parquet(f'{D}/universe.parquet', columns=['symbol', 'cik'])
        uni = uni.dropna(subset=['cik'])
        uni['si'] = uni['symbol'].map(p.sidx)
        uni = uni.dropna(subset=['si'])
        cik2si = {}
        for c, s in zip(uni['cik'].astype(int), uni['si'].astype(int)):
            cik2si.setdefault(int(c), []).append(int(s))
        f = pd.read_parquet(f'{D}/shares_facts.parquet')
        f = f[f['cik'].isin(cik2si)]
        f = f.sort_values(['cik', 'filed', 'end'])
        self.by_si = {}
        sess = p.sess_np
        for cik, g in f.groupby('cik', sort=False):
            filed = g['filed'].to_numpy('datetime64[ns]')
            end = g['end'].to_numpy('datetime64[ns]')
            val = g['val'].to_numpy(dtype=float)
            e_i = np.clip(np.searchsorted(sess, end, side='right') - 1, 0, len(sess) - 1)
            for s in cik2si[int(cik)]:
                self.by_si[s] = (filed, val, e_i)
        self.splitcum = p.splitcum
        log(f'SharesPIT: {len(self.by_si):,} panel symbols carry a share-count history')

    def shares_at(self, si_arr, asof, anchor_t):
        """Split-NORMALISED share count known at `asof`, restated to the anchor session basis."""
        out = np.full(len(si_arr), np.nan)
        a64 = np.datetime64(asof)
        for i, s in enumerate(si_arr):
            rec = self.by_si.get(int(s))
            if rec is None:
                continue
            filed, val, e_i = rec
            j = int(np.searchsorted(filed, a64, side='left')) - 1
            if j < 0:
                continue
            f_anchor = self.splitcum[int(s), anchor_t]
            f_meas = self.splitcum[int(s), e_i[j]]
            if not np.isfinite(f_anchor) or not np.isfinite(f_meas) or f_meas <= 0:
                continue
            out[i] = val[j] * (f_anchor / f_meas)
        return out

    def issuance(self, si_arr, asof, anchor_t, lookback_days=365):
        """log growth of the split-normalised share count over ~12 months, known at `asof`."""
        out = np.full(len(si_arr), np.nan)
        a64 = np.datetime64(asof)
        for i, s in enumerate(si_arr):
            rec = self.by_si.get(int(s))
            if rec is None:
                continue
            filed, val, e_i = rec
            j = int(np.searchsorted(filed, a64, side='left')) - 1
            if j < 1:
                continue
            target = filed[j] - np.timedelta64(lookback_days, 'D')
            j0 = int(np.searchsorted(filed[:j], target, side='right')) - 1
            if j0 < 0:
                j0 = 0
            # require the two observations to be 9-18 months apart
            gap = (filed[j] - filed[j0]) / np.timedelta64(1, 'D')
            if gap < 270 or gap > 550:
                continue
            f1 = self.splitcum[int(s), e_i[j]]
            f0 = self.splitcum[int(s), e_i[j0]]
            if not (np.isfinite(f1) and np.isfinite(f0)) or f0 <= 0 or f1 <= 0:
                continue
            v1 = val[j] / f1
            v0 = val[j0] / f0
            if v0 <= 0 or v1 <= 0:
                continue
            out[i] = np.log(v1 / v0)
        return out


# ------------------------------------------------------------------ A3: net share issuance


def a3_signals(p: Panel, sh: SharesPIT):
    me = p.month_end
    rows = []
    for k in range(12, len(me) - 1):
        t = me[k]
        el = p.eligible(t)
        idx = np.nonzero(el)[0]
        if len(idx) < 100:
            continue
        iss = sh.issuance(idx, p.sessions[t], t)
        ok = np.isfinite(iss)
        if ok.sum() < 100:
            continue
        sub = pd.DataFrame({'k': k, 'si': idx[ok], 'sig': iss[ok]})
        sub['dec'] = decile(sub['sig'].to_numpy())
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ A4: dividend month


def a4_signals(p: Panel, div: pd.DataFrame):
    """Predicted-dividend-month indicator, Hartzmark-Solomon 2013.

    At close(month-end k), using ONLY ex-dates strictly before that close, a stock is
    PREDICTED to pay in month k+1 if it had an ex-date in the same calendar month one
    year earlier AND has >= 2 ex-dates in the trailing 24 months.  The calendar is built
    from Alpaca's own `ex_date` field; the two-regime record-date formula is reconciled
    against it in a4_exdate_regimes() and is never used to date a trade.
    """
    me = p.month_end
    d = div[div['si'].notna()].copy()
    d['si'] = d['si'].astype(int)
    d['ym'] = d['ex_date'].dt.to_period('M')
    by = {}
    for (s, ym), g in d.groupby(['si', 'ym'], sort=False):
        by.setdefault(s, set()).add(ym)
    rows = []
    for k in range(13, len(me) - 1):
        t = me[k]
        el = p.eligible(t)
        idx = np.nonzero(el)[0]
        if len(idx) < 100:
            continue
        tgt = p.sessions[t].to_period('M') + 1          # the month we are predicting
        prev = tgt - 12
        w0 = tgt - 24
        pred = np.zeros(len(idx), dtype=bool)
        hist = np.zeros(len(idx), dtype=bool)
        for i, s in enumerate(idx):
            months = by.get(int(s))
            if not months:
                continue
            n24 = sum(1 for m in months if w0 <= m <= tgt - 1)
            hist[i] = n24 >= 2
            pred[i] = (prev in months) and hist[i]
        if pred.sum() < 10 or (hist & ~pred).sum() < 50:
            continue
        sub = pd.DataFrame({'k': k, 'si': idx, 'sig': pred.astype(float),
                            'pred': pred, 'hist': hist})
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def a4_exdate_regimes(div: pd.DataFrame):
    """The T+1 -> T+0 settlement change of 2024-05-28, and what a single-regime formula costs.

    Pre 2024-05-28 the ex-date is one business day BEFORE the record date; from
    2024-05-28 (FINRA Rule 11140(b)(1) as amended for T+1 settlement) the ex-date IS the
    record date.  Counted here against Alpaca's actual `ex_date` on every cash dividend
    that carries a record date.
    """
    d = div.dropna(subset=['ex_date', 'record_date']).copy()
    d['pred_minus1'] = d['record_date'] - pd.offsets.BDay(1)
    d['pred_same'] = d['record_date']
    d['era'] = np.where(d['ex_date'] >= '2024-05-28', 'post', 'pre')
    out = {}
    for era, g in d.groupby('era'):
        out[era] = dict(
            n=int(len(g)),
            misdated_if_always_minus1=int((g['pred_minus1'] != g['ex_date']).sum()),
            misdated_if_always_same=int((g['pred_same'] != g['ex_date']).sum()),
            misdated_two_regime=int(((g['pred_minus1'] != g['ex_date']) if era == 'pre'
                                     else (g['pred_same'] != g['ex_date'])).sum()),
        )
    # month-level consequence: how many events would land in the WRONG calendar month
    for era, g in d.groupby('era'):
        wrong_single = (g['pred_minus1'].dt.to_period('M') != g['ex_date'].dt.to_period('M'))
        out[era]['month_misassigned_if_always_minus1'] = int(wrong_single.sum())
        wrong_same = (g['pred_same'].dt.to_period('M') != g['ex_date'].dt.to_period('M'))
        out[era]['month_misassigned_if_always_same'] = int(wrong_same.sum())
    out['boundary_window'] = {}
    for lab, lo, hi in (('2024-04', '2024-04-01', '2024-04-30'),
                        ('2024-05_pre', '2024-05-01', '2024-05-27'),
                        ('2024-05_post', '2024-05-28', '2024-05-31'),
                        ('2024-06', '2024-06-01', '2024-06-30'),
                        ('2024-07', '2024-07-01', '2024-07-31')):
        g = d[(d['ex_date'] >= lo) & (d['ex_date'] <= hi)]
        out['boundary_window'][lab] = dict(
            n=int(len(g)),
            misdated_always_minus1=int((g['pred_minus1'] != g['ex_date']).sum()),
            misdated_always_same=int((g['pred_same'] != g['ex_date']).sum()))
    return out


# ------------------------------------------------------------------ F1: PEAD / SUE


def f1_signals(p: Panel):
    """SUE (Foster-Olsen-Shevlin seasonal random walk with drift) on point-in-time EPS facts."""
    ev = pd.read_parquet(f'{D}/earnings_events.parquet',
                         columns=['cik', 'symbol', 'acceptance_utc', 'event_session',
                                  'n_prior_qfacts'])
    ev = ev[ev['symbol'].isin(p.sidx)].copy()
    ev['si'] = ev['symbol'].map(p.sidx).astype(int)
    ev['acc'] = pd.to_datetime(ev['acceptance_utc'], utc=True).dt.tz_localize(None)
    ev['S'] = pd.to_datetime(ev['event_session']).map(
        {d: i for i, d in enumerate(p.sessions)})
    ev = ev.dropna(subset=['S'])
    ev['S'] = ev['S'].astype(int)
    facts = pd.read_parquet(f'{D}/eps_facts.parquet', columns=['cik', 'end', 'val', 'filed'])
    facts['end'] = pd.to_datetime(facts['end'])
    facts['filed'] = pd.to_datetime(facts['filed'])
    facts = facts.sort_values(['cik', 'end', 'filed'])
    fb = {c: (g['end'].to_numpy('datetime64[ns]'), g['filed'].to_numpy('datetime64[ns]'),
              g['val'].to_numpy(dtype=float)) for c, g in facts.groupby('cik', sort=False)}
    log(f'F1: {len(ev):,} events on {ev["si"].nunique():,} panel symbols; '
        f'{len(fb):,} CIKs with EPS facts')

    sue = np.full(len(ev), np.nan)
    ciks = ev['cik'].to_numpy()
    accs = ev['acc'].to_numpy('datetime64[ns]')
    n_ok = 0
    for i in range(len(ev)):
        rec = fb.get(ciks[i])
        if rec is None:
            continue
        end, filed, val = rec
        m = filed < accs[i]
        if m.sum() < 11:
            continue
        e, v = end[m], val[m]
        # latest filing per period end wins
        order = np.argsort(e, kind='stable')
        e, v = e[order], v[order]
        ue = np.unique(e)
        # np.unique returns the FIRST index; we want the LAST (latest filed for that end)
        idx_last = np.searchsorted(e, ue, side='right') - 1
        e, v = ue, v[idx_last]
        if len(v) < 11:
            continue
        d = v[4:] - v[:-4]
        if len(d) < 7:
            continue
        cur = d[-1]
        prior = d[-9:-1] if len(d) >= 9 else d[:-1]
        if len(prior) < 6:
            continue
        sd = prior.std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            continue
        sue[i] = (cur - prior.mean()) / sd
        n_ok += 1
    ev['sue'] = sue
    log(f'F1: SUE computable on {n_ok:,} of {len(ev):,} events ({n_ok / len(ev):.1%})')
    ev = ev.dropna(subset=['sue'])
    # eligibility at the event close
    el = p.elig[ev['si'].to_numpy(), ev['S'].to_numpy()]
    ev = ev[el]
    log(f'F1: {len(ev):,} events eligible at the event close')
    return ev.reset_index(drop=True)


def f1_causal_deciles(ev, p: Panel, window_days=250, min_win=200):
    """Decile of SUE against every PRIOR event in the trailing 250 calendar days.

    Reference population stated explicitly: prior eligible, SUE-computable events only --
    not the contemporaneous cross-section (which would be a look-ahead) and not the
    full-sample distribution.
    """
    S = ev['S'].to_numpy()
    r = ev['sue'].to_numpy()
    dates = p.sess_np
    dec = np.full(len(ev), -1, dtype=np.int64)
    order = np.argsort(S, kind='stable')
    S_o, r_o = S[order], r[order]
    uniq, starts = np.unique(S_o, return_index=True)
    ends = np.r_[starts[1:], len(S_o)]
    buf_vals, buf_sess, lo = [], [], 0
    for k, s in enumerate(uniq):
        cutoff = dates[s] - np.timedelta64(window_days, 'D')
        while lo < len(buf_sess) and dates[buf_sess[lo]] < cutoff:
            lo += 1
        win = np.array(buf_vals[lo:], dtype=np.float64) if len(buf_vals) > lo else np.empty(0)
        cur = r_o[starts[k]:ends[k]]
        if len(win) >= min_win:
            q = np.quantile(win, np.arange(1, 10) / 10.0)
            d = np.searchsorted(q, cur, side='right') + 1      # 1..10
        else:
            d = np.full(len(cur), -1, dtype=np.int64)
        dec[order[starts[k]:ends[k]]] = d
        buf_vals.extend(cur.tolist())
        buf_sess.extend([s] * len(cur))
        if lo > 50_000:
            buf_vals, buf_sess, lo = buf_vals[lo:], buf_sess[lo:], 0
    ev = ev.copy()
    ev['dec'] = dec
    return ev[ev['dec'] > 0].reset_index(drop=True)


def f1_trades(p: Panel, ev, H, leg, label, haircut=False):
    """Event-anchored trades: entry close(S+1), exit close(S+1+H)."""
    t = ev.copy()
    t['a'] = t['S'] + 1
    t['b'] = t['S'] + 1 + H
    t = t[(t['b'] < p.n_d)]
    si, a, b = t['si'].to_numpy(), t['a'].to_numpy(), t['b'].to_numpy()
    ca, cb = p.close[si, a], p.close[si, b]
    good = np.isfinite(ca) & np.isfinite(cb) & (ca > 0)
    t = t[good]
    si, a, b = si[good], a[good], b[good]
    with np.errstate(invalid='ignore', divide='ignore'):
        t['gross'] = p.close[si, b] / p.close[si, a] - 1.0
    order_usd, npos = concurrent_order_usd(p, a, b)
    t['n_concurrent'] = npos
    t['k'] = 0
    t['order_usd'] = order_usd
    t['hold'] = (b - a).astype(float)
    t['cost_bps'] = cost_bps(p.adv[si, a], order_usd, short=(leg < 0), hold_days=t['hold'])
    t['leg'] = leg
    t['ends_mid_hold'] = p.last_fin[si] < b
    t['tainted'] = p.taint[si, a] | np.array(
        [p.jump_events[s_, a_ + 1:b_ + 1].any() for s_, a_, b_ in zip(si, a, b)])
    t['contrib'] = leg * t['gross'].to_numpy()
    t['entry_dt'] = p.sessions[a]
    t['exit_dt'] = p.sessions[b]
    t['label'] = label
    t['sig'] = t['sue']
    return t


# ------------------------------------------------------------------ F6: overnight/intraday


def f6_signals(p: Panel):
    """Trailing 12-1-month cumulative OVERNIGHT and INTRADAY returns (Lou-Polk-Skouras)."""
    on = np.full(p.close.shape, np.nan, dtype=np.float32)
    intr = np.full(p.close.shape, np.nan, dtype=np.float32)
    for lo in range(0, p.n_s, 500):                      # chunked: the node has 7.8 GB
        hi = min(lo + 500, p.n_s)
        with np.errstate(invalid='ignore', divide='ignore'):
            a = p.open[lo:hi, 1:] / p.close[lo:hi, :-1] - 1.0
            b = p.close[lo:hi] / p.open[lo:hi] - 1.0
        on[lo:hi, 1:] = np.where(np.isfinite(a) & (np.abs(a) < 5.0), a, np.nan)
        intr[lo:hi] = np.where(np.isfinite(b) & (np.abs(b) < 5.0), b, np.nan)
        del a, b
    me = p.month_end
    rows = []
    for k in range(13, len(me) - 1):
        t = me[k - 1]                       # skip the most recent month (the "-1")
        t0 = me[k - 13]
        if t0 < 0:
            continue
        seg_on = on[:, t0 + 1:t + 1]
        seg_in = intr[:, t0 + 1:t + 1]
        with np.errstate(invalid='ignore'):
            con = np.nansum(np.log1p(np.where(np.isfinite(seg_on), seg_on, 0.0)), axis=1)
            cin = np.nansum(np.log1p(np.where(np.isfinite(seg_in), seg_in, 0.0)), axis=1)
        nobs = np.isfinite(seg_on).sum(axis=1)
        el = p.eligible(me[k])
        ok = el & (nobs >= 200)
        idx = np.nonzero(ok)[0]
        if len(idx) < 100:
            continue
        sub = pd.DataFrame({'k': k, 'si': idx, 'sig': con[idx], 'sig_in': cin[idx]})
        sub['dec'] = decile(sub['sig'].to_numpy())
        sub['dec_in'] = decile(sub['sig_in'].to_numpy())
        rows.append(sub)
    sig = pd.concat(rows, ignore_index=True)
    # the MEASUREMENT half: the overnight/intraday decomposition of the eligible universe
    meas = []
    for y in range(2016, 2024):
        cols = np.nonzero(np.asarray(p.sessions.year) == y)[0]
        el = np.zeros(p.n_s, dtype=bool)
        for t in cols[::21]:
            el |= p.eligible(t)
        o = on[np.ix_(el, cols)]
        i2 = intr[np.ix_(el, cols)]
        meas.append(dict(year=y, n_symbols=int(el.sum()),
                         overnight_bps_per_day=float(np.nanmean(o) * 1e4),
                         intraday_bps_per_day=float(np.nanmean(i2) * 1e4),
                         overnight_ann_pct=float(np.nanmean(o) * 252 * 100),
                         intraday_ann_pct=float(np.nanmean(i2) * 252 * 100)))
    return sig, pd.DataFrame(meas)


# ------------------------------------------------------------------ main


def main():
    os.makedirs(OUT, exist_ok=True)
    p = Panel()
    log(f'gate ADV >= ${MIN_ADV:,.0f} | flat costs {FLAT_COSTS} | A2 diss lag {A2_DISS_BDAYS} bd '
        f'| splits {SPLIT_ORDER} | out {OUT}')
    me = p.month_end
    recs = []
    extra = {}

    # ---------------- F5 ----------------
    log('F5: 52-week-high signals')
    s5 = f5_signals(p)
    log(f'  F5 rows {len(s5):,} over {s5["k"].nunique()} months')
    lo5 = make_trades(p, s5, me, (s5['dec'] == 10).to_numpy(), +1, 'F5-LO', hold_k=6)
    bn5 = make_trades(p, s5, me, (s5['dec'] != 10).to_numpy(), +1, 'F5-BN', hold_k=6)
    sh5 = make_trades(p, s5, me, (s5['dec'] == 1).to_numpy(), -1, 'F5-SH', hold_k=6)
    sh5 = sh5[p.etb[sh5['si'].to_numpy()]] if len(sh5) else sh5
    recs += score_cell(p, 'F5-LO', lo5, None, bn5, 6, bench_key='F5')
    recs += score_cell(p, 'F5-LS', lo5, sh5, bn5, 6, bench_key='F5')
    lo5.to_parquet(f'{OUT}/f5_lo_trades.parquet')

    del lo5, bn5, sh5, s5
    _BENCH_MEMO.clear()
    gc.collect()

    # ---------------- A2 ----------------
    log('A2: short-interest signals')
    sh_pit = SharesPIT(p)
    s2, anch2 = a2_signals(p, sh_pit)
    log(f'  A2 rows {len(s2):,} over {s2["k"].nunique()} dissemination dates')
    lo2 = make_trades(p, s2, anch2, (s2['dec'] == 1).to_numpy(), +1, 'A2-LO')
    bn2 = make_trades(p, s2, anch2, (s2['dec'] != 1).to_numpy(), +1, 'A2-BN')
    sh2 = make_trades(p, s2, anch2, (s2['dec'] == 10).to_numpy(), -1, 'A2-SH')
    sh2 = sh2[p.etb[sh2['si'].to_numpy()]] if len(sh2) else sh2
    recs += score_cell(p, 'A2-LO', lo2, None, bn2, 1, ascending=True, bench_key='A2')
    recs += score_cell(p, 'A2-LS', lo2, sh2, bn2, 1, ascending=True, bench_key='A2')
    # A2: the published claim is that the LONG (low-short-interest) side carries the
    # abnormal return.  Per-decile net means make that testable directly.
    a2dec = {}
    for dnum in (1, 2, 3, 9, 10):
        tt = make_trades(p, s2, anch2, (s2['dec'] == dnum).to_numpy(), +1, f'A2-D{dnum}')
        for split in SPLIT_ORDER:
            L = sealed(tt, split)
            if len(L) < 50:
                continue
            net = L['gross'] - L['cost_bps'] / 1e4
            a2dec[f'D{dnum}_{split}'] = dict(
                n=int(len(L)), mean_bps=float(net.mean() * 1e4),
                t=float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))))
        del tt
    extra['A2_deciles'] = a2dec
    extra['A2'] = dict(diss_bdays=A2_DISS_BDAYS,
                       first_settlement=str(pd.read_parquet(
                           f'{D}/short_interest.parquet',
                           columns=['settlementDate'])['settlementDate'].min().date()),
                       n_anchors=int(len(anch2)))
    lo2.to_parquet(f'{OUT}/a2_lo_trades.parquet')

    del lo2, bn2, sh2, s2
    _BENCH_MEMO.clear()
    gc.collect()

    # ---------------- A3 ----------------
    log('A3: net-share-issuance signals')
    s3 = a3_signals(p, sh_pit)
    log(f'  A3 rows {len(s3):,} over {s3["k"].nunique()} months')
    lo3 = make_trades(p, s3, me, (s3['dec'] == 1).to_numpy(), +1, 'A3-LO')
    bn3 = make_trades(p, s3, me, (s3['dec'] != 1).to_numpy(), +1, 'A3-BN')
    sh3 = make_trades(p, s3, me, (s3['dec'] == 10).to_numpy(), -1, 'A3-SH')
    sh3 = sh3[p.etb[sh3['si'].to_numpy()]] if len(sh3) else sh3
    recs += score_cell(p, 'A3-LO', lo3, None, bn3, 1, ascending=True, bench_key='A3')
    recs += score_cell(p, 'A3-LS', lo3, sh3, bn3, 1, ascending=True, bench_key='A3')
    extra['A3'] = dict(zero_issuance_share=float((np.abs(s3['sig']) < 1e-9).mean()),
                       median_issuance=float(s3['sig'].median()))
    lo3.to_parquet(f'{OUT}/a3_lo_trades.parquet')

    del lo3, bn3, sh3, s3
    _BENCH_MEMO.clear()
    gc.collect()

    # ---------------- A4 ----------------
    log('A4: dividend-month signals')
    div = pd.read_parquet(f'{D}/dividends.parquet')
    div['si'] = div['symbol'].map(p.sidx)
    extra['A4_exdate_regimes'] = a4_exdate_regimes(div)
    log('  A4 ex-date regime reconciliation: '
        + json.dumps(extra['A4_exdate_regimes'].get('post', {})))
    div_tr = div[div['ex_date'] <= SPLITS['VAL'][1]]        # TEST prices never touched
    s4 = a4_signals(p, div_tr)
    log(f'  A4 rows {len(s4):,} over {s4["k"].nunique()} months')
    lo4 = make_trades(p, s4, me, s4['pred'].to_numpy(), +1, 'A4-LO')
    # every predicted name has sig == 1.0, so the 20-slot book needs an explicit, declared
    # tiebreak (REPORT_F2_A1 S3: "the executable book's sign is set by the tiebreak").
    # Chosen BEFORE any return was read: most liquid first, the same rule F2/A1 used.
    if len(lo4):
        lo4['rank'] = p.adv[lo4['si'].to_numpy(), lo4['a'].to_numpy()]
    bn4 = make_trades(p, s4, me, (s4['hist'] & ~s4['pred']).to_numpy(), +1, 'A4-BN')
    sh4 = make_trades(p, s4, me, (s4['hist'] & ~s4['pred']).to_numpy(), -1, 'A4-SH')
    sh4 = sh4[p.etb[sh4['si'].to_numpy()]] if len(sh4) else sh4
    recs += score_cell(p, 'A4-LO', lo4, None, bn4, 1, rank_col='rank', bench_key='A4')
    recs += score_cell(p, 'A4-LS', lo4, sh4, bn4, 1, rank_col='rank', bench_key='A4')
    lo4.to_parquet(f'{OUT}/a4_lo_trades.parquet')

    del lo4, bn4, sh4, s4
    _BENCH_MEMO.clear()
    gc.collect()

    # ---------------- F1 (declared null-replication) ----------------
    log('F1: PEAD / SUE (declared null-replication)')
    ev = f1_signals(p)
    ev = f1_causal_deciles(ev, p)
    log(f'  F1 ranked events {len(ev):,}')
    lo1 = f1_trades(p, ev[ev['dec'] == 10], 60, +1, 'F1-LO-60')
    bn1 = f1_trades(p, ev[ev['dec'] != 10], 60, +1, 'F1-BN-60')
    sh1 = f1_trades(p, ev[ev['dec'] == 1], 60, -1, 'F1-SH-60')
    sh1 = sh1[p.etb[sh1['si'].to_numpy()]] if len(sh1) else sh1
    recs += score_cell(p, 'F1-LO-60', lo1, None, bn1, 3, bench_key='F1')
    recs += score_cell(p, 'F1-LS-60', lo1, sh1, bn1, 3, bench_key='F1')
    extra['F1'] = dict(n_events_ranked=int(len(ev)),
                       n_d10=int((ev['dec'] == 10).sum()), n_d1=int((ev['dec'] == 1).sum()))
    lo1.to_parquet(f'{OUT}/f1_lo_trades.parquet')

    del lo1, bn1, sh1, ev
    _BENCH_MEMO.clear()
    gc.collect()

    # ---------------- F6 (measurement cell) ----------------
    log('F6: overnight vs intraday')
    s6, meas = f6_signals(p)
    log(f'  F6 rows {len(s6):,} over {s6["k"].nunique()} months')
    lo6 = make_trades(p, s6, me, (s6['dec'] == 10).to_numpy(), +1, 'F6-LO')
    bn6 = make_trades(p, s6, me, (s6['dec'] != 10).to_numpy(), +1, 'F6-BN')
    sh6 = make_trades(p, s6, me, (s6['dec_in'] == 10).to_numpy(), -1, 'F6-SH')
    sh6 = sh6[p.etb[sh6['si'].to_numpy()]] if len(sh6) else sh6
    recs += score_cell(p, 'F6-LS', lo6, sh6, bn6, 1)
    meas.to_csv(f'{OUT}/f6_measurement.csv', index=False)
    extra['F6_measurement'] = meas.to_dict('records')
    lo6.to_parquet(f'{OUT}/f6_lo_trades.parquet')

    # universe hygiene (independent rebuild D3): `kind=='common'` carries SPAC units and
    # dual-class lines, and dotted tickers are the marker.
    dotted = [sym for sym in p.symbols if '.' in sym]
    extra['universe_hygiene'] = dict(
        n_common_plus_spy=len(p.symbols), n_dotted=len(dotted), dotted=sorted(dotted)[:30],
        n_unreverted_jump_symbols=int(p.jump_events.any(axis=1).sum()),
        n_unreverted_jump_sessions=int(p.jump_events.sum()),
        n_jump_sessions_raw_close_ge_5=int(
            (p.jump_events & (p.raw >= MIN_RAW_CLOSE)).sum()))

    df = pd.DataFrame(recs)
    df.to_csv(f'{OUT}/cells.csv', index=False)
    json.dump(extra, open(f'{OUT}/extra.json', 'w'), indent=1, default=str)
    log(f'wrote {OUT}/cells.csv ({len(df)} rows)')
    cols = ['cell', 'split', 'conv', 'n_trades', 'mean_bps', 't', 'nw_t', 'pct_pos',
            'ex_top_1_bps', 'ex_top_5_bps', 'ex_jan_bps', 'mde_bps', 'tr_wk',
            'book_usd_mo', 'book_mde_usd']
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(df[cols].round(2).to_string(index=False), flush=True)


if __name__ == '__main__':
    sys.exit(main())
