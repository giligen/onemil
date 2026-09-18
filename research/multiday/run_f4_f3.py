#!/usr/bin/env python3
"""F4 (weekly industry-adjusted reversal, 1 long-only cell) + F3 (12-1 / 6-1 / residual momentum, 6 cells).

Pre-registration: PREREG_F4_F3.md, written before any return was computed.
Standing corrections inherited from REPORT_F2_A1.md §9 are implemented here, not re-litigated:
  * ADV20$ on the RAW panel (defect 6)                  -> Panel.adv
  * ex-top-1% is a mandatory rebuilt column (defect 2)   -> tails()
  * impact scales with the ACTUAL position count (d. 3)  -> per-rebalance order$
  * costs are never charged to the benchmark (defect 4)  -> bench series gross
  * benchmark excludes the book's own decile (defect 5)  -> clean contrast
  * sealed splits: a trade whose exit leaves the split leaves the split (§8b)
  * Newey-West t at lag = the hold length                -> nw_t()

Env arms:  MD_MIN_ADV (default 1e6) · MD_FLAT_COSTS=1 (5 bps/side) · MD_ARM (output suffix)
           OPEN_TEST=1 (refused unless FREEZE.md exists)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

D = '/home/ec2-user/onemil/research/multiday/data'
OUTBASE = '/home/ec2-user/onemil/research/multiday/out_f4f3'

SPLITS = {'TRAIN': ('2016-01-01', '2021-12-31'),
          'VAL': ('2022-01-01', '2023-12-31'),
          'TEST': ('2024-01-01', '2026-09-18')}

BOOK_USD = 66_000.0
N_SLOTS = 20
POS_USD = BOOK_USD / N_SLOTS
MIN_RAW_CLOSE = 5.0
MIN_ADV = float(os.environ.get('MD_MIN_ADV', 1_000_000.0))
FLAT_COSTS = os.environ.get('MD_FLAT_COSTS') == '1'
ARM = os.environ.get('MD_ARM', '')
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
        self.adv = z['adv_raw']                 # THE FIX: raw-panel dollar volume
        self.symbols = list(z['symbols'])
        self.sessions = pd.to_datetime(list(z['sessions']))
        self.sic2 = np.array([str(v) for v in z['sic2']], dtype=object)
        self.etb = z['etb'].astype(bool)
        self.sidx = {s: i for i, s in enumerate(self.symbols)}
        self.n_s, self.n_d = self.close.shape
        prev, cur = self.close[:, :-1], self.close[:, 1:]
        with np.errstate(invalid='ignore', divide='ignore'):
            r = cur / prev - 1.0
        self.ret = np.zeros_like(self.close)
        self.ret[:, 1:] = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        self.spy = self.sidx['SPY']
        # last session with a finite adjusted close, per symbol (the delisting marker on this panel)
        fin = np.isfinite(self.close)
        self.last_fin = np.where(fin.any(axis=1), fin.shape[1] - 1 - np.argmax(fin[:, ::-1], axis=1), -1)
        # MAX = largest daily return over a trailing 21 sessions (Bali-Cakici-Whitelaw)
        self.maxret = np.full_like(self.close, np.nan)
        w = 21
        rr = np.where(np.isfinite(self.close), self.ret, np.nan)
        for t in range(w, self.n_d):
            self.maxret[:, t] = np.nanmax(rr[:, t - w + 1:t + 1], axis=1)
        mdf = pd.DataFrame({'d': self.sessions})
        mdf['ym'] = mdf['d'].dt.to_period('M')
        mdf['yw'] = mdf['d'].dt.to_period('W')
        self.month_end = mdf.groupby('ym').tail(1).index.to_numpy()
        self.week_end = mdf.groupby('yw').tail(1).index.to_numpy()
        log(f'panel {self.n_s} symbols x {self.n_d} sessions | month-ends {len(self.month_end)} '
            f'| week-ends {len(self.week_end)}')

    def eligible(self, t):
        """Universe membership at decision close t: common, RAW close >= $5, RAW ADV20$ >= gate."""
        return (np.isfinite(self.raw[:, t]) & (self.raw[:, t] >= MIN_RAW_CLOSE)
                & np.isfinite(self.adv[:, t]) & (self.adv[:, t] >= MIN_ADV)
                & np.isfinite(self.close[:, t]))


# ------------------------------------------------------------------ costs


def cost_bps(adv, order_usd, short=False, hold_days=0):
    """Round-trip auction cost. order_usd SCALES with the real position count (defect 3)."""
    if FLAT_COSTS:
        c = np.full(np.shape(adv), 2 * FLAT_ARM_BPS, dtype=float)
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            frac = np.minimum(order_usd / np.maximum(0.01 * adv, 1.0), 1.0)
        c = 2.0 * IMPACT_COEF_BPS * frac + SELL_FEE_BPS
    if short:
        c = c + BORROW_ANN * 1e4 * (hold_days / 252.0)
    return c


# ------------------------------------------------------------------ signals


def decile(vals, n=10):
    """Cross-sectional decile 1..n of `vals` (1 = lowest). Reference population = the input array."""
    order = np.argsort(np.argsort(vals))
    return np.floor(order * n / len(vals)).astype(int) + 1


def f3_signals(p: Panel, kind: str):
    """Rows: (k_month, si, signal). kind in {'m12','m6','res'}. All strictly causal at close(M_k)."""
    me = p.month_end
    n_m = len(me)
    # monthly returns for the residual regression
    mret = np.full((p.n_s, n_m), np.nan, dtype=np.float32)
    for k in range(1, n_m):
        a, b = p.close[:, me[k - 1]], p.close[:, me[k]]
        with np.errstate(invalid='ignore', divide='ignore'):
            mret[:, k] = b / a - 1.0
    rm = mret[p.spy]

    rows = []
    start = 13 if kind in ('m12', 'res') else 7
    if kind == 'res':
        start = max(start, 25)
    for k in range(start, n_m - 1):
        t = me[k]
        el = p.eligible(t)
        if kind == 'm12':
            a, b = p.close[:, me[k - 12]], p.close[:, me[k - 1]]
            with np.errstate(invalid='ignore', divide='ignore'):
                sig = b / a - 1.0
            ok = el & np.isfinite(sig)
        elif kind == 'm6':
            a, b = p.close[:, me[k - 6]], p.close[:, me[k - 1]]
            with np.errstate(invalid='ignore', divide='ignore'):
                sig = b / a - 1.0
            ok = el & np.isfinite(sig)
        else:
            lo = max(0, k - 35)
            Y = mret[:, lo:k + 1].astype(np.float64)           # window ENDS at month k (causal)
            X = rm[lo:k + 1].astype(np.float64)
            m = np.isfinite(Y) & np.isfinite(X)[None, :]
            nobs = m.sum(axis=1)
            Yz = np.where(m, Y, 0.0)
            Xz = np.where(m, X[None, :], 0.0)
            sx = Xz.sum(axis=1); sy = Yz.sum(axis=1)
            sxx = (Xz * Xz).sum(axis=1); sxy = (Xz * Yz).sum(axis=1)
            den = nobs * sxx - sx * sx
            with np.errstate(invalid='ignore', divide='ignore'):
                beta = np.where(den > 0, (nobs * sxy - sx * sy) / np.where(den == 0, 1, den), np.nan)
                alpha = (sy - beta * sx) / np.maximum(nobs, 1)
            # residuals over the formation months k-11 .. k-1
            f0, f1 = k - 11, k
            E = mret[:, f0:f1].astype(np.float64) - (alpha[:, None] + beta[:, None] * rm[f0:f1][None, :])
            E = np.where(np.isfinite(mret[:, f0:f1]), E, np.nan)
            with np.errstate(invalid='ignore', divide='ignore'):
                mu = np.nanmean(E, axis=1)
                sd = np.nanstd(E, axis=1, ddof=1)
                sig = np.where(sd > 0, mu / sd, np.nan)
            nform = np.isfinite(E).sum(axis=1)
            ok = el & np.isfinite(sig) & (nobs >= 24) & (nform >= 9)
        idx = np.nonzero(ok)[0]
        if len(idx) < 100:
            continue
        dec = decile(sig[idx])
        for i, dd, sv in zip(idx, dec, sig[idx]):
            rows.append((k, int(i), float(sv), int(dd)))
    return pd.DataFrame(rows, columns=['k', 'si', 'sig', 'dec'])


def f4_signals(p: Panel):
    """Weekly industry-adjusted 1-week reversal. Rows: (j_week, si, resid, decile, MAX, raw close)."""
    we = p.week_end
    rows = []
    for j in range(2, len(we) - 1):
        t = we[j]
        if t - 5 < 0:
            continue
        el = p.eligible(t) & np.isfinite(p.close[:, t - 5])
        with np.errstate(invalid='ignore', divide='ignore'):
            r1w = p.close[:, t] / p.close[:, t - 5] - 1.0
        has_sic = np.array([s not in ('', 'nan', 'None') for s in p.sic2])
        ok = el & np.isfinite(r1w) & has_sic
        idx = np.nonzero(ok)[0]
        if len(idx) < 100:
            continue
        sic = p.sic2[idx]
        df = pd.DataFrame({'si': idx, 'r': r1w[idx].astype(np.float64), 'sic': sic})
        grp = df.groupby('sic')['r']
        df['n_ind'] = grp.transform('size')
        df['ind_sum'] = grp.transform('sum')
        df = df[df['n_ind'] >= 5].copy()             # >=5 names in the industry, else dropped
        if len(df) < 100:
            continue
        # industry mean EXCLUDING the stock itself -- self-inclusion shrinks the residual by 1/n
        df['ind_mean'] = (df['ind_sum'] - df['r']) / (df['n_ind'] - 1)
        df['resid'] = df['r'] - df['ind_mean']
        df['dec'] = decile(df['resid'].to_numpy())
        df['j'] = j
        df['maxret'] = p.maxret[df['si'].to_numpy(), t]
        df['rawc'] = p.raw[df['si'].to_numpy(), t]
        rows.append(df[['j', 'si', 'resid', 'dec', 'maxret', 'rawc', 'r', 'ind_mean']].copy())
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ trades


def make_trades(p: Panel, sig: pd.DataFrame, anchors, sel, leg, hold_label):
    """Turn a (rebalance, symbol) selection into trades with per-rebalance-scaled costs."""
    sub = sig[sel].copy()
    if not len(sub):
        return pd.DataFrame()
    key = 'k' if 'k' in sub.columns else 'j'
    sub['a'] = anchors[sub[key].to_numpy()] + 1                     # entry = close(anchor+1)
    sub['b'] = anchors[sub[key].to_numpy() + 1] + 1                 # exit  = close(next anchor+1)
    sub = sub[(sub['a'] < p.n_d) & (sub['b'] < p.n_d) & (sub['b'] > sub['a'])]
    si = sub['si'].to_numpy(); a = sub['a'].to_numpy(); b = sub['b'].to_numpy()
    ca, cb = p.close[si, a], p.close[si, b]
    good = np.isfinite(ca) & np.isfinite(cb) & (ca > 0)
    sub = sub[good]
    si, a, b = si[good], a[good], b[good]
    with np.errstate(invalid='ignore', divide='ignore'):
        sub['gross'] = p.close[si, b] / p.close[si, a] - 1.0
    npos = sub.groupby(key)['si'].transform('size').to_numpy()
    order_usd = BOOK_USD / np.maximum(npos, 1)
    hold = (b - a).astype(float)
    sub['order_usd'] = order_usd
    sub['cost_bps'] = cost_bps(p.adv[si, a], order_usd, short=(leg < 0), hold_days=hold)
    sub['leg'] = leg
    sub['hold'] = hold
    sub['contrib'] = leg * sub['gross'].to_numpy()
    sub['ends_mid_hold'] = p.last_fin[si] < b
    sub['entry_dt'] = p.sessions[a]
    sub['exit_dt'] = p.sessions[b]
    sub['label'] = hold_label
    return sub


def daily_series(p: Panel, t: pd.DataFrame, charge=True):
    """Equal-weighted overlapping daily portfolio return; costs on the entry and exit days only."""
    n_d = p.n_d
    num = np.zeros(n_d); den = np.zeros(n_d)
    if not len(t):
        return pd.Series(num, index=p.sessions), pd.Series(den, index=p.sessions)
    si = t['si'].to_numpy(); a = t['a'].to_numpy(); b = t['b'].to_numpy()
    leg = t['leg'].to_numpy(dtype=float)
    cost = (t['cost_bps'].to_numpy() / 1e4) if charge else np.zeros(len(t))
    for i in range(len(t)):
        idx = np.arange(a[i] + 1, b[i] + 1)
        if not len(idx):
            continue
        r = p.ret[si[i], idx].astype(np.float64) * leg[i]
        r[0] -= cost[i] * 0.5
        r[-1] -= cost[i] * 0.5
        np.add.at(num, idx, r)
        np.add.at(den, idx, 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        d = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
    return pd.Series(d, index=p.sessions), pd.Series(den, index=p.sessions)


def monthly(s):
    return (1.0 + s).resample('ME').prod() - 1.0


def nw_t(x, lag):
    """Newey-West t of the mean of x with `lag` Bartlett lags."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 6:
        return np.nan
    e = x - x.mean()
    g0 = (e * e).sum() / n
    s = g0
    for L in range(1, min(lag, n - 1) + 1):
        gl = (e[L:] * e[:-L]).sum() / n
        s += 2.0 * (1.0 - L / (lag + 1.0)) * gl
    if s <= 0:
        return np.nan
    return x.mean() / np.sqrt(s / n)


def stats(m, split, hold_months=1, sealed_mask=None):
    a, b = SPLITS[split]
    mm = m[(m.index >= a) & (m.index <= b)]
    mm = mm[mm != 0]
    if len(mm) < 6:
        return dict(n_months=len(mm), mean_bps=np.nan, t=np.nan, nw_t=np.nan, pct_pos=np.nan,
                    ex_jan_bps=np.nan, mde_bps=np.nan, mdd_pct=np.nan, worst_bps=np.nan)
    mu, sd, n = mm.mean(), mm.std(ddof=1), len(mm)
    ex = mm[mm.index.month != 1]
    eq = (1 + mm).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return dict(n_months=n, mean_bps=mu * 1e4,
                t=(mu / (sd / np.sqrt(n))) if sd > 0 else np.nan,
                nw_t=nw_t(mm.to_numpy(), max(1, int(hold_months))),
                pct_pos=100.0 * (mm > 0).mean(),
                ex_jan_bps=ex.mean() * 1e4 if len(ex) else np.nan,
                mde_bps=2.0 * sd / np.sqrt(n) * 1e4,
                mdd_pct=100.0 * mdd, worst_bps=mm.min() * 1e4)


def sealed(t, split):
    """A trade whose EXIT falls outside the split is not in that split (REPORT_F2_A1 §8b)."""
    a, b = SPLITS[split]
    return t[(t['entry_dt'] >= a) & (t['exit_dt'] <= b)]


def complete(t, split):
    a, b = SPLITS[split]
    return t[(t['entry_dt'] >= a) & (t['entry_dt'] <= b)]


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


# ------------------------------------------------------------------ cell scoring


def score_cell(p, name, long_t, short_t, bench_t, hold_months, out):
    """Long-only (short_t None) or L-S cell. Returns the per-split record list."""
    recs = []
    for split in SPLIT_ORDER:
        for conv in (('sealed', sealed), ('complete', complete)):
            cname, fn = conv
            if cname == 'complete' and split == 'TRAIN':
                continue                                   # reported for VAL/TEST only
            L = fn(long_t, split)
            B = fn(bench_t, split)
            S = fn(short_t, split) if short_t is not None else None
            if len(L) < 20:
                continue
            ls, _ = daily_series(p, L, charge=True)
            bs, _ = daily_series(p, B, charge=False)       # benchmark GROSS (defect 4)
            if S is not None and len(S):
                ss, _ = daily_series(p, S, charge=True)
                cell = ls + ss                             # short leg already carries leg=-1
                exc = cell
                lls = None
                mL = monthly(ls); mS = monthly(ss)
                a, b = SPLITS[split]
                mLs = mL[(mL.index >= a) & (mL.index <= b)]; mSs = mS[(mS.index >= a) & (mS.index <= b)]
                mBs = monthly(bs); mBs = mBs[(mBs.index >= a) & (mBs.index <= b)]
                spread = mLs.mean() - (-mSs.mean())
                if spread > 0:
                    lls = float((mLs.mean() - mBs.mean()) / spread)
            else:
                cell = ls
                exc = ls - bs
                lls = None
            st = stats(monthly(exc), split, hold_months)
            st_raw = stats(monthly(cell), split, hold_months)
            # tails: rebuild the book without the top 1% / 5% CONTRIBUTORS
            tail = {}
            for q, lab in ((0.01, 'ex_top_1'), (0.05, 'ex_top_5')):
                allt = pd.concat([L] + ([S] if S is not None and len(S) else []))
                thr = allt['contrib'].quantile(1 - q)
                L2 = L[L['contrib'] < thr]
                ls2, _ = daily_series(p, L2, charge=True)
                if S is not None and len(S):
                    S2 = S[S['contrib'] < thr]
                    ss2, _ = daily_series(p, S2, charge=True)
                    e2 = ls2 + ss2
                else:
                    e2 = ls2 - bs
                s2 = stats(monthly(e2), split, hold_months)
                tail[lab + '_bps'] = s2['mean_bps']
                tail[lab + '_t'] = s2['t']
            # book (executable, 20 slots)
            bk = book_sim(p, L, 'sig' if 'sig' in L.columns else 'resid',
                          ascending=('resid' in L.columns))
            if len(bk) >= 20:
                # ALPHA basis, as in REPORT_F2_A1.md S3: the 20-slot book's own EW daily series,
                # net of cost, MINUS the gross benchmark. A raw long-only book in 2016-2021 earns
                # the market plus the panel's survivorship and is not a money estimate.
                bdaily, _ = daily_series(p, bk, charge=True)
                a0, b0 = SPLITS[split]
                mb = monthly(bdaily); mbm = monthly(bs)
                sel = (mb.index >= a0) & (mb.index <= b0) & ((mb != 0) | (mbm != 0))
                alpha_m = (mb - mbm)[sel] * BOOK_USD
                raw_m = mb[sel] * BOOK_USD
                bmu = alpha_m.mean(); bsd = alpha_m.std(ddof=1) if len(alpha_m) > 1 else np.nan
                eq = (1 + (mb - mbm)[sel]).cumprod()
                book = dict(n=len(bk), usd_mo=bmu, raw_usd_mo=raw_m.mean(),
                            t=(bmu / (bsd / np.sqrt(len(alpha_m)))) if bsd and bsd > 0 else np.nan,
                            mde_usd=2 * bsd / np.sqrt(len(alpha_m)) if bsd and bsd > 0 else np.nan,
                            pct_pos=100.0 * (alpha_m > 0).mean(), worst=alpha_m.min(),
                            mdd_usd=float((eq / eq.cummax() - 1).min() * BOOK_USD))
                wk = (pd.Timestamp(SPLITS[split][1]) - pd.Timestamp(SPLITS[split][0])).days / 7.0
                book['tr_wk'] = len(bk) / wk
            else:
                book = dict(n=0, usd_mo=np.nan, raw_usd_mo=np.nan, t=np.nan, mde_usd=np.nan,
                            pct_pos=np.nan, worst=np.nan, mdd_usd=np.nan, tr_wk=0.0)
            honest = float(pd.concat([L] + ([S] if S is not None and len(S) else []))['cost_bps'].mean())
            rec = dict(cell=name, split=split, conv=cname, n_trades=len(L) + (len(S) if S is not None else 0),
                       raw_bps=st_raw['mean_bps'], **{k: st[k] for k in
                                                      ('mean_bps', 't', 'nw_t', 'pct_pos', 'ex_jan_bps',
                                                       'mde_bps', 'mdd_pct', 'worst_bps', 'n_months')},
                       **tail, lls=lls, honest_cost_bps=honest,
                       ends_mid_hold=int(L['ends_mid_hold'].sum()),
                       book_n=book['n'], book_usd_mo=book['usd_mo'],
                       book_raw_usd_mo=book['raw_usd_mo'], book_t=book['t'],
                       book_mde_usd=book['mde_usd'], book_pct_pos=book['pct_pos'],
                       book_worst=book['worst'], book_mdd_usd=book['mdd_usd'], tr_wk=book['tr_wk'])
            if split == 'TRAIN' and cname == 'sealed':
                rec['boot_p'] = boot_p(monthly(exc)[(monthly(exc).index >= SPLITS[split][0]) &
                                                    (monthly(exc).index <= SPLITS[split][1])]
                                       .replace(0, np.nan).dropna())
            recs.append(rec)
    return recs


# ------------------------------------------------------------------ main


def main():
    os.makedirs(OUT, exist_ok=True)
    p = Panel()
    log(f'gate ADV >= ${MIN_ADV:,.0f} | flat costs {FLAT_COSTS} | splits {SPLIT_ORDER} | out {OUT}')

    me, we = p.month_end, p.week_end
    all_recs = []
    f4_extra = {}
    f3_extra = {}

    # ---------------- F4 ----------------
    log('F4: building weekly industry-adjusted reversal signals')
    s4 = f4_signals(p)
    log(f'F4 signal rows {len(s4):,} over {s4["j"].nunique()} weeks')
    lo4 = make_trades(p, s4, we, (s4['dec'] == 1).to_numpy(), +1, 'F4-LO')
    bn4 = make_trades(p, s4, we, (s4['dec'] != 1).to_numpy(), +1, 'F4-BENCH')
    sh4 = make_trades(p, s4, we, (s4['dec'] == 10).to_numpy(), -1, 'F4-SHORT')
    all_recs += score_cell(p, 'F4-LO', lo4, None, bn4, 1, OUT)
    lo4.to_parquet(f'{OUT}/f4_lo_trades.parquet')

    # F4 mandatory splits (AMENDMENT 3(a))
    for split in SPLIT_ORDER:
        L = sealed(lo4, split)
        if len(L) < 20:
            continue
        L = L.copy()
        L['net'] = L['gross'] - L['cost_bps'] / 1e4
        q = pd.qcut(L['maxret'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        g = L.groupby(q, observed=True)['net']
        f4_extra[f'max_quintile_{split}'] = {
            str(k): dict(n=int(v.size), mean_bps=float(v.mean() * 1e4),
                         t=float(v.mean() / (v.std(ddof=1) / np.sqrt(v.size))) if v.size > 2 else None,
                         pnl_share=float(v.sum() / L['net'].sum()) if L['net'].sum() != 0 else None)
            for k, v in g}
        lo10 = L['rawc'] < 10.0
        f4_extra[f'price_floor_{split}'] = dict(
            n_below_10=int(lo10.sum()), share_trades=float(lo10.mean()),
            pnl_share_below_10=float(L.loc[lo10, 'net'].sum() / L['net'].sum())
            if L['net'].sum() != 0 else None,
            mean_bps_below_10=float(L.loc[lo10, 'net'].mean() * 1e4) if lo10.any() else None,
            mean_bps_above_10=float(L.loc[~lo10, 'net'].mean() * 1e4) if (~lo10).any() else None)
        # break-even cost at ~100%/wk turnover: the per-round-trip charge that zeroes the excess
        ls, _ = daily_series(p, L, charge=False)
        bs, _ = daily_series(p, sealed(bn4, split), charge=False)
        m = monthly(ls - bs)
        a, b = SPLITS[split]
        m = m[(m.index >= a) & (m.index <= b)]
        f4_extra[f'breakeven_{split}'] = dict(
            gross_excess_bps_mo=float(m.mean() * 1e4),
            round_trips_per_month=float(52.0 / 12.0),
            breakeven_bps_per_rt=float(m.mean() * 1e4 / (52.0 / 12.0)),
            honest_cost_bps=float(L['cost_bps'].mean()),
            multiple_of_honest=float((m.mean() * 1e4 / (52.0 / 12.0)) / L['cost_bps'].mean()))
    json.dump(f4_extra, open(f'{OUT}/f4_extra.json', 'w'), indent=1, default=float)

    # ---------------- F3 ----------------
    for kind, tag in (('m12', '12-1'), ('m6', '6-1'), ('res', 'RES')):
        log(f'F3: building {tag} signals')
        s3 = f3_signals(p, kind)
        log(f'  {tag}: {len(s3):,} signal rows over {s3["k"].nunique()} months')
        lo = make_trades(p, s3, me, (s3['dec'] == 10).to_numpy(), +1, f'F3-LO-{tag}')
        bn = make_trades(p, s3, me, (s3['dec'] != 10).to_numpy(), +1, f'F3-BENCH-{tag}')
        sh = make_trades(p, s3, me, (s3['dec'] == 1).to_numpy(), -1, f'F3-SH-{tag}')
        sh = sh[p.etb[sh['si'].to_numpy()]] if len(sh) else sh    # ETB gate on the short leg
        all_recs += score_cell(p, f'F3-LO-{tag}', lo, None, bn, 1, OUT)
        all_recs += score_cell(p, f'F3-LS-{tag}', lo, sh, bn, 1, OUT)
        lo.to_parquet(f'{OUT}/f3_lo_{kind}_trades.parquet')
        sh.to_parquet(f'{OUT}/f3_sh_{kind}_trades.parquet')
        # crash windows, on the full TRAIN+VAL monthly series
        for label, tr, st in (('LO', lo, None), ('LS', lo, sh)):
            ls, _ = daily_series(p, tr, charge=True)
            if st is not None and len(st):
                ss, _ = daily_series(p, st, charge=True)
                cur = ls + ss
            else:
                bs, _ = daily_series(p, bn, charge=False)
                cur = ls - bs
            m = monthly(cur)
            m = m[(m.index >= '2016-01-01') & (m.index <= '2023-12-31')]
            m = m[m != 0]
            eq = (1 + m).cumprod()
            dd = eq / eq.cummax() - 1
            f3_extra[f'{tag}-{label}'] = dict(
                mdd_pct=float(dd.min() * 100), mdd_month=str(dd.idxmin().date()),
                crash_2020=float(((1 + m[(m.index >= '2020-03-01') & (m.index <= '2020-06-30')])
                                  .prod() - 1) * 100),
                crash_2022=float(((1 + m[(m.index >= '2022-01-01') & (m.index <= '2022-12-31')])
                                  .prod() - 1) * 100),
                worst_month_pct=float(m.min() * 100), worst_month=str(m.idxmin().date()),
                best_month_pct=float(m.max() * 100))
    json.dump(f3_extra, open(f'{OUT}/f3_extra.json', 'w'), indent=1, default=float)

    df = pd.DataFrame(all_recs)
    df.to_csv(f'{OUT}/cells.csv', index=False)
    log(f'wrote {OUT}/cells.csv  ({len(df)} rows)')
    cols = ['cell', 'split', 'conv', 'n_trades', 'mean_bps', 't', 'nw_t', 'pct_pos',
            'ex_top_1_bps', 'ex_top_5_bps', 'mde_bps', 'tr_wk', 'book_usd_mo']
    with pd.option_context('display.width', 200, 'display.max_columns', 40):
        print(df[cols].to_string(index=False), flush=True)


if __name__ == '__main__':
    sys.exit(main())
