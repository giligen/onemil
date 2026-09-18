#!/usr/bin/env python3
"""F2 (announcement-return drift, 6 cells) + A1 (announcement premium, 2 cells).

Pre-registration: research/multiday/PREREG_F2_A1.md (written before any return was computed).
Execution convention is close-to-close (`cls`), per Goyal-Jegadeesh-Wu JFQA 2026: opening auctions
are illiquid, so nothing here touches an open.

Outputs (research/multiday/out_f2a1/):
  availability.csv   -- the availability/missingness audit on every decision field
  events_f2.csv      -- one row per scored F2 event (trade-level: for tail tests + break-even)
  events_a1.csv      -- ditto for A1
  cells.csv          -- the 8-cell table, per split, with the four mandatory columns
  monthly_<cell>.csv -- the monthly portfolio series behind each cell
"""
import os, sys, json, time
import numpy as np
import pandas as pd

D = '/home/ec2-user/onemil/research/multiday/data'
OUT = '/home/ec2-user/onemil/research/multiday/out_f2a1'
os.makedirs(OUT, exist_ok=True)

SPLITS = {'TRAIN': ('2016-01-01', '2021-12-31'),
          'VAL':   ('2022-01-01', '2023-12-31'),
          'TEST':  ('2024-01-01', '2026-09-18')}

BOOK_USD = 66_000.0
N_SLOTS = 20
POS_USD = BOOK_USD / N_SLOTS            # $3,300
MIN_RAW_CLOSE = 5.0
MIN_ADV = float(os.environ.get('MD_MIN_ADV', 1_000_000.0))   # primary; $10M is a reported arm
FLAT_COSTS = os.environ.get('MD_FLAT_COSTS') == '1'   # secondary arm: 5 bps/side, flat
F2_SKIP = int(os.environ.get('MD_F2_SKIP', 1))        # 1 = obtainable (enter close(S+1)); 0 = not obtainable
ARM = os.environ.get('MD_ARM', '')
SELL_FEE_BPS = 0.4                      # SEC + FINRA TAF, sells only
IMPACT_COEF_BPS = 10.0                  # 10 bps at 1% of ADV$
RANK_WINDOW_DAYS = 250                  # causal trailing window for the decile cut
FLAT_ARM_BPS = 5.0                      # secondary cost arm, per side
CONTROL_OFFSET = 31                     # A1 placebo window: mid-quarter, no announcement due

# TEST is SEALED. It is computed only when FREEZE.md exists and OPEN_TEST=1 is passed explicitly.
FREEZE = '/home/ec2-user/onemil/research/multiday/FREEZE.md'
_OPEN = os.environ.get('OPEN_TEST') == '1' and os.path.exists(FREEZE)
SPLIT_ORDER = ('TRAIN', 'VAL', 'TEST') if _OPEN else ('TRAIN', 'VAL')
OUT = OUT + ('_test' if _OPEN else '') + (('_' + ARM) if ARM else '')
os.makedirs(OUT, exist_ok=True)


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


# ---------------------------------------------------------------- panel


class Panel:
    """Dense (symbol x session) price matrices + the index maps."""

    def __init__(self):
        z = np.load(f'{D}/panel_f2a1.npz', allow_pickle=True)
        self.close = z['close_adj']
        self.raw = z['close_raw']
        self.dvol = z['dvol']
        self.symbols = list(z['symbols'])
        self.sessions = pd.to_datetime(list(z['sessions']))
        self.sidx = {s: i for i, s in enumerate(self.symbols)}
        self.didx = {d: i for i, d in enumerate(self.sessions)}
        self.n_s, self.n_d = self.close.shape
        # daily simple returns, NaN-safe (a frozen position earns 0 that day)
        prev = self.close[:, :-1]
        cur = self.close[:, 1:]
        with np.errstate(invalid='ignore', divide='ignore'):
            r = cur / prev - 1.0
        self.ret = np.zeros_like(self.close)
        self.ret[:, 1:] = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        # ADV20$ ending at each session (trailing mean of dollar volume, min 10 obs)
        dv = np.nan_to_num(self.dvol, nan=0.0)
        ok = np.isfinite(self.dvol).astype(np.float32)
        cs = np.cumsum(dv, axis=1)
        co = np.cumsum(ok, axis=1)
        w = 20
        num = cs.copy(); den = co.copy()
        num[:, w:] = cs[:, w:] - cs[:, :-w]
        den[:, w:] = co[:, w:] - co[:, :-w]
        with np.errstate(invalid='ignore', divide='ignore'):
            self.adv = np.where(den >= 10, num / np.maximum(den, 1), np.nan).astype(np.float32)
        self.spy = self.sidx['SPY']
        log(f'panel {self.n_s} symbols x {self.n_d} sessions')


# ---------------------------------------------------------------- costs


def impact_bps(adv_usd, order_usd=POS_USD):
    """10 bps x (order$ / 1% of ADV$), the order capped at 1% of ADV -> impact capped at 10 bps."""
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = order_usd / np.maximum(0.01 * adv_usd, 1.0)
    return IMPACT_COEF_BPS * np.minimum(frac, 1.0)


def round_trip_cost_bps(adv_in, adv_out, short=False, hold_days=0):
    """Honest auction round trip: impact both sides + sell fee. Borrow only on a short leg."""
    if FLAT_COSTS:
        c = np.full(np.shape(adv_in), 2 * FLAT_ARM_BPS, dtype=float)
    else:
        c = impact_bps(adv_in) + impact_bps(adv_out) + SELL_FEE_BPS
    if short:
        c = c + 0.30 / 100.0 * 1e4 * (hold_days / 252.0)   # 0.3%/yr general collateral
    return c


# ---------------------------------------------------------------- events


def load_events(p: Panel):
    ev = pd.read_parquet(f'{D}/earnings_events.parquet',
                         columns=['symbol', 'acceptance_utc', 'acceptance_et', 'event_session',
                                  'acceptance_bucket', 'form', 'accession'])
    n0 = len(ev)
    ev = ev[ev['symbol'].isin(p.sidx)].copy()
    ev['S'] = ev['event_session'].map(p.didx)
    n_nosess = ev['S'].isna().sum()
    ev = ev.dropna(subset=['S'])
    ev['S'] = ev['S'].astype(int)
    ev['si'] = ev['symbol'].map(p.sidx).astype(int)
    # de-dup: one event per symbol per session (8-K/A amendments, multi-filing days)
    ev = ev.sort_values(['symbol', 'S', 'acceptance_utc']).drop_duplicates(['symbol', 'S'], keep='first')
    log(f'events raw {n0} -> in-panel {len(ev)} (dropped {n_nosess} with no session index)')
    return ev.reset_index(drop=True)


def availability_audit(p: Panel, ev: pd.DataFrame):
    """PLAN §1: every decision field traced to a timestamp at or before the decision close."""
    rows = []
    acc_et = pd.to_datetime(ev['acceptance_et'])
    sess = pd.to_datetime(ev['event_session'])
    # the event-session rule: the closing auction of S must be strictly after acceptance
    close_et = sess + pd.Timedelta(hours=16)
    viol = (acc_et.dt.tz_localize(None) >= close_et).sum()
    rows.append(dict(field='event_session', rule='close(S) strictly after acceptanceDateTime',
                     n=len(ev), violations=int(viol), pct_viol=100.0 * viol / len(ev)))
    prev_sess = sess.shift(0)
    # would S-1's close also have been after acceptance? (i.e. is S the FIRST such session)
    si = ev['si'].to_numpy(); S = ev['S'].to_numpy()
    for name, arr in (('close_adj[S]', p.close[si, S]),
                      ('close_adj[S-2]', p.close[si, np.maximum(S - 2, 0)]),
                      ('close_raw[S]', p.raw[si, S]),
                      ('ADV20[S]', p.adv[si, S])):
        miss = (~np.isfinite(arr)).sum()
        rows.append(dict(field=name, rule='observable at the close of S', n=len(ev),
                         violations=int(miss), pct_viol=100.0 * miss / len(ev)))
    # coverage per split and per acceptance bucket (the missingness table the standing rule requires)
    ev2 = ev.copy()
    ev2['yr'] = sess.dt.year
    cov = np.isfinite(p.close[si, S]) & np.isfinite(p.close[si, np.maximum(S - 2, 0)]) & \
          np.isfinite(p.adv[si, S]) & np.isfinite(p.raw[si, S])
    ev2['cov'] = cov
    for split, (a, b) in SPLITS.items():
        m = (sess >= a) & (sess <= b)
        sub = ev2[m.to_numpy()]
        for bucket in ['post_close', 'pre_open', 'intraday', 'ALL']:
            s2 = sub if bucket == 'ALL' else sub[sub['acceptance_bucket'] == bucket]
            if not len(s2):
                continue
            rows.append(dict(field=f'signal_complete|{split}|{bucket}', rule='all 4 price fields finite',
                             n=len(s2), violations=int((~s2['cov']).sum()),
                             pct_viol=100.0 * (~s2['cov']).mean()))
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/availability.csv', index=False)
    log('availability audit written; event_session violations = %d' % viol)
    return df


# ---------------------------------------------------------------- F2


def build_f2(p: Panel, ev: pd.DataFrame):
    si = ev['si'].to_numpy(); S = ev['S'].to_numpy()
    ok = S >= 2
    ev = ev[ok].copy(); si = si[ok]; S = S[ok]
    c_s = p.close[si, S]; c_s2 = p.close[si, S - 2]
    spy_s = p.close[p.spy, S]; spy_s2 = p.close[p.spy, S - 2]
    with np.errstate(invalid='ignore', divide='ignore'):
        r2 = c_s / c_s2 - 1.0
        r2m = spy_s / spy_s2 - 1.0
    ev['r2_abn'] = r2 - r2m
    ev['raw_close'] = p.raw[si, S]
    ev['adv'] = p.adv[si, S]
    ev['sess_date'] = p.sessions[S]
    ev = ev[np.isfinite(ev['r2_abn']) & np.isfinite(ev['raw_close']) & np.isfinite(ev['adv'])]
    log(f'F2 events with a complete signal: {len(ev)}')
    return ev.sort_values('S').reset_index(drop=True)


def causal_deciles(ev: pd.DataFrame, p: Panel):
    """Decile = percentile of r2_abn against every event with event_session < S in the trailing 250d."""
    S = ev['S'].to_numpy(); r = ev['r2_abn'].to_numpy()
    dates = p.sessions.values
    dec = np.full(len(ev), -1, dtype=np.int8)
    # group events by session index; walk sessions forward maintaining a deque of the window
    order = np.argsort(S, kind='stable')
    S_o, r_o = S[order], r[order]
    uniq, starts = np.unique(S_o, return_index=True)
    ends = np.r_[starts[1:], len(S_o)]
    buf_vals, buf_sess = [], []
    lo = 0
    for k, s in enumerate(uniq):
        cutoff = dates[s] - np.timedelta64(RANK_WINDOW_DAYS, 'D')
        while lo < len(buf_sess) and dates[buf_sess[lo]] < cutoff:
            lo += 1
        win = np.array(buf_vals[lo:], dtype=np.float64) if len(buf_vals) > lo else np.empty(0)
        cur = r_o[starts[k]:ends[k]]
        if len(win) >= 200:
            q = np.quantile(win, np.arange(1, 10) / 10.0)
            d = np.searchsorted(q, cur, side='right')   # 0..9
        else:
            d = np.full(len(cur), -1, dtype=np.int64)
        dec[order[starts[k]:ends[k]]] = d
        buf_vals.extend(cur.tolist()); buf_sess.extend([s] * len(cur))
        if lo > 50_000:                                  # compact the deque
            buf_vals = buf_vals[lo:]; buf_sess = buf_sess[lo:]; lo = 0
    ev = ev.copy(); ev['decile'] = dec
    log(f'deciles assigned; unranked (warm-up) {int((dec < 0).sum())}')
    return ev[ev['decile'] >= 0].reset_index(drop=True)


def trade_returns(p: Panel, si, a, b):
    """Gross simple return from close(a) to close(b), NaN-safe via the daily-return product."""
    out = np.empty(len(si), dtype=np.float64)
    for i in range(len(si)):
        seg = p.ret[si[i], a[i] + 1:b[i] + 1]
        out[i] = np.prod(1.0 + seg.astype(np.float64)) - 1.0
    return out


def f2_trades(p: Panel, ev: pd.DataFrame, H: int):
    """Per-event trade table: entry close(S+1), exit close(S+1+H)."""
    si = ev['si'].to_numpy(); S = ev['S'].to_numpy()
    a = S + F2_SKIP
    b = a + H
    ok = (b < p.n_d) & np.isfinite(p.close[si, np.minimum(a, p.n_d - 1)])
    t = ev[ok].copy()
    si, a, b = si[ok], a[ok], b[ok]
    t['a'] = a; t['b'] = b
    t['gross'] = trade_returns(p, si, a, b)
    t['bench'] = trade_returns(p, np.full(len(a), p.spy), a, b)   # SPY over the identical window
    t['alpha'] = t['gross'] - t['bench']
    adv_in = p.adv[si, a]; adv_out = p.adv[si, np.minimum(b, p.n_d - 1)]
    t['adv_in'] = adv_in
    t['cost_bps'] = round_trip_cost_bps(adv_in, np.where(np.isfinite(adv_out), adv_out, adv_in))
    t['cost_bps_flat'] = 2 * FLAT_ARM_BPS
    t['net'] = t['gross'] - t['cost_bps'] / 1e4
    t['entry_date'] = p.sessions[a]
    t['exit_date'] = p.sessions[b]
    t['H'] = H
    return t


# ---------------------------------------------------------------- A1


def build_a1(p: Panel, ev: pd.DataFrame):
    """Expected announcement session Ehat = snap(prior-year same-quarter session + 364d).

    Causal: the prediction is made at the prior-year event (index k), and it is only ACTED on if, five
    sessions before Ehat, exactly three of this symbol's announcements have landed since then (the three
    intervening quarters) -- i.e. the next one is genuinely due. Nothing after Ehat-5 is read.
    """
    sess_idx = p.sessions.values
    rows = []
    for sym, g in ev.groupby('symbol', sort=False):
        g = g.sort_values('S')
        S = g['S'].to_numpy(); si = int(g['si'].iloc[0])
        for k in range(len(S)):
            # causal history requirement: >= 5 announcements ALREADY filed at prediction time.
            # (An earlier draft used `range(len(S)-4)` + `len(S)>=6`, both of which condition on
            #  FUTURE events and silently delete symbols that stopped announcing -- a look-ahead
            #  and a second survivorship layer. Fixed 2026-09-18 before any number was reported.)
            if k < 4:
                continue
            d_k = sess_idx[S[k]]
            tgt = d_k + np.timedelta64(364, 'D')
            e = int(np.searchsorted(sess_idx, tgt, side='left'))
            if e >= p.n_d - 2 or e - 5 <= S[k]:
                continue
            entry = e - 5
            # causality: how many of this symbol's events landed in (S[k], entry]?
            n_between = int(((S > S[k]) & (S <= entry)).sum())
            if n_between != 3:
                continue
            nxt = S[S > entry]
            actual = int(nxt[0]) if len(nxt) else -1     # diagnostic ONLY; never a filter
            rows.append((sym, si, S[k], e, entry, actual))
    a1 = pd.DataFrame(rows, columns=['symbol', 'si', 'S_prior', 'Ehat', 'entry', 'S_actual'])
    si = a1['si'].to_numpy(); en = a1['entry'].to_numpy()
    a1['raw_close'] = p.raw[si, en]
    a1['adv'] = p.adv[si, en]
    a1['entry_date'] = p.sessions[en]
    a1['ehat_date'] = p.sessions[a1['Ehat'].to_numpy()]
    a1['gap_sessions'] = np.where(a1['S_actual'] >= 0, a1['S_actual'] - a1['Ehat'], np.nan)
    a1 = a1[np.isfinite(a1['raw_close']) & np.isfinite(a1['adv'])]
    log(f'A1 predicted windows: {len(a1)}  median |S_actual - Ehat| = '
        f'{np.nanmedian(np.abs(a1["gap_sessions"])):.1f} sessions')
    return a1.reset_index(drop=True)


def a1_trades(p: Panel, a1: pd.DataFrame, exit_off: int, label: str):
    si = a1['si'].to_numpy(); a = a1['entry'].to_numpy(); b = a1['Ehat'].to_numpy() + exit_off
    ok = (b < p.n_d) & (b > a) & np.isfinite(p.close[si, a])
    t = a1[ok].copy(); si, a, b = si[ok], a[ok], b[ok]
    t['a'] = a; t['b'] = b
    t['gross'] = trade_returns(p, si, a, b)
    # Frazzini-Lamont design: the SAME stock over an identical-length MID-QUARTER window with no
    # announcement due (31 sessions earlier). This is the announcement-vs-non-announcement control.
    ca, cb = a - CONTROL_OFFSET, b - CONTROL_OFFSET
    good = ca >= 0
    ctrl = np.full(len(a), np.nan)
    ctrl[good] = trade_returns(p, si[good], ca[good], cb[good])
    t['ctrl'] = ctrl
    t['bench'] = trade_returns(p, np.full(len(a), p.spy), a, b)
    t['alpha'] = t['gross'] - t['ctrl']          # the published A1 contrast
    t['alpha_spy'] = t['gross'] - t['bench']
    t['ca'] = ca; t['cb'] = cb
    adv_out = p.adv[si, np.minimum(b, p.n_d - 1)]
    t['adv_in'] = p.adv[si, a]
    t['cost_bps'] = round_trip_cost_bps(t['adv_in'].to_numpy(),
                                        np.where(np.isfinite(adv_out), adv_out, t['adv_in'].to_numpy()))
    t['net'] = t['gross'] - t['cost_bps'] / 1e4
    t['exit_date'] = p.sessions[b]
    t['H'] = b - a
    t['cell'] = label
    return t


# ---------------------------------------------------------------- portfolios


def daily_series(p: Panel, t: pd.DataFrame, weight_short=False, acol='a', bcol='b', charge=True):
    """Overlapping equal-weighted daily portfolio return, costs charged on the entry and exit days."""
    n_d = p.n_d
    num = np.zeros(n_d); den = np.zeros(n_d)
    si = t['si'].to_numpy(); a = t[acol].to_numpy(); b = t[bcol].to_numpy()
    cost = (t['cost_bps'].to_numpy() / 1e4) if charge else np.zeros(len(t))
    sgn = -1.0 if weight_short else 1.0
    for i in range(len(t)):
        if a[i] < 0 or b[i] <= a[i] or b[i] >= n_d:
            continue
        idx = np.arange(a[i] + 1, b[i] + 1)
        if not len(idx):
            continue
        r = p.ret[si[i], idx].astype(np.float64) * sgn
        r[0] -= cost[i] * 0.5
        r[-1] -= cost[i] * 0.5
        np.add.at(num, idx, r)
        np.add.at(den, idx, 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        d = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
    return pd.Series(d, index=p.sessions), pd.Series(den, index=p.sessions)


def monthly(s: pd.Series):
    return (1.0 + s).resample('ME').prod() - 1.0


def stats(m: pd.Series, split):
    a, b = SPLITS[split]
    mm = m[(m.index >= a) & (m.index <= b)]
    mm = mm[mm != 0]
    if len(mm) < 6:
        return dict(n_months=len(mm), mean_bps=np.nan, t=np.nan, pct_pos=np.nan,
                    ex_jan_bps=np.nan, mde_bps=np.nan, ann_pct=np.nan, mdd_pct=np.nan)
    mu = mm.mean(); sd = mm.std(ddof=1); n = len(mm)
    ex = mm[mm.index.month != 1]
    eq = (1 + mm).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return dict(n_months=n, mean_bps=mu * 1e4, t=mu / (sd / np.sqrt(n)) if sd > 0 else np.nan,
                pct_pos=100.0 * (mm > 0).mean(),
                ex_jan_bps=ex.mean() * 1e4 if len(ex) else np.nan,
                mde_bps=2.0 * sd / np.sqrt(n) * 1e4,
                ann_pct=100.0 * ((1 + mu) ** 12 - 1), mdd_pct=100.0 * mdd)


def book_sim(p: Panel, t: pd.DataFrame, rank_col: str, ascending=False):
    """$66K / 20 slots, first-come by entry session, ranked within the session, no refill."""
    t = t.sort_values(['a', rank_col], ascending=[True, ascending])
    free_at = np.zeros(N_SLOTS, dtype=int)
    taken = []
    for row in t.itertuples():
        j = int(np.argmin(free_at))
        if free_at[j] <= row.a:
            free_at[j] = row.b
            taken.append(row.Index)
    return t.loc[taken]


# ---------------------------------------------------------------- main


def main():
    p = Panel()
    ev = load_events(p)
    availability_audit(p, ev)

    f2 = build_f2(p, ev)
    f2 = f2[(f2['raw_close'] >= MIN_RAW_CLOSE) & (f2['adv'] >= MIN_ADV)]
    log(f'F2 after $5 / $1M ADV gates: {len(f2)}')
    f2 = causal_deciles(f2, p)

    a1 = build_a1(p, ev)
    a1 = a1[(a1['raw_close'] >= MIN_RAW_CLOSE) & (a1['adv'] >= MIN_ADV)]
    log(f'A1 after gates: {len(a1)}')

    cells = []
    trade_tabs = []
    series = {}
    bench_of = {}

    for H in (20, 40, 60):
        t = f2_trades(p, f2, H)
        trade_tabs.append(t.assign(fam='F2'))
        d10 = t[t['decile'] == 9]; d1 = t[t['decile'] == 0]
        s10, n10 = daily_series(p, d10)
        s1, n1 = daily_series(p, d1)
        sall, _ = daily_series(p, t)
        m10, m1, mall = monthly(s10), monthly(s1), monthly(sall)
        series[f'F2-LO-{H}'] = m10
        series[f'F2-LS-{H}'] = m10 - m1
        series[f'F2-MKT-{H}'] = mall
        series[f'F2-D1-{H}'] = m1
        # the long-only benchmark is the ALL-DECILE equal-weighted book of the same events, same hold,
        # same costs: it strips beta AND the announcer-population effect, leaving the decile signal.
        bench_of[f'F2-LO-{H}'] = f'F2-MKT-{H}'
        bench_of[f'F2-LS-{H}'] = None
        cells.append(dict(cell=f'F2-LS-{H}', fam='F2', kind='L-S', H=H,
                          t_long=d10, t_short=d1, t_all=t, alpha_col='alpha'))
        cells.append(dict(cell=f'F2-LO-{H}', fam='F2', kind='long-only', H=H,
                          t_long=d10, t_short=None, t_all=t, alpha_col='alpha'))

    for off, lab in ((-1, 'A1-a'), (0, 'A1-b'), (1, 'A1-rob')):
        t = a1_trades(p, a1, off, lab)
        trade_tabs.append(t.assign(fam='A1', decile=np.nan))
        s, _ = daily_series(p, t)
        sc, _ = daily_series(p, t, acol='ca', bcol='cb')
        series[lab] = monthly(s)
        series[lab + '-CTRL'] = monthly(sc)
        bench_of[lab] = lab + '-CTRL'
        cells.append(dict(cell=lab, fam='A1', kind='long-only', H=int(t['H'].median()),
                          t_long=t, t_short=None, t_all=t, alpha_col='alpha'))

    # ---- assemble the cell table
    rows = []
    for c in cells:
        name = c['cell']
        bn = bench_of.get(name)
        exc = series[name] - series[bn] if bn else series[name]
        for split in SPLIT_ORDER:
            a, b = SPLITS[split]
            st_raw = stats(series[name], split)
            st = stats(exc, split)                    # PRIMARY: benchmark-adjusted
            tl = c['t_long']
            msk = (tl['entry_date'] >= a) & (tl['entry_date'] <= b)
            tls = tl[msk]
            ac = c['alpha_col']
            g = tls['gross'].to_numpy(); al = tls[ac].to_numpy()
            cst = tls['cost_bps'].to_numpy()
            aln = al - cst / 1e4                      # benchmark-adjusted, net of the honest auction cost
            honest = np.nanmean(cst)
            be = np.nanmean(al) * 1e4                 # round-trip bps that zeroes the cell (on alpha)
            if np.isfinite(aln).sum() > 20:
                v = aln[np.isfinite(aln)]
                k = max(1, int(0.05 * len(v)))
                ex5 = np.sort(v)[:-k].mean()
                wins = v[v > 0]
                capv = 3 * np.median(wins) if len(wins) else np.nan
                capped = np.where(v > capv, capv, v).mean() if np.isfinite(capv) else np.nan
            else:
                ex5 = capped = np.nan
            # long-leg share of the L-S spread (only meaningful when the spread is positive)
            if c['fam'] == 'F2':
                H = c['H']
                sL = stats(series[f'F2-LO-{H}'], split)['mean_bps']
                sM = stats(series[f'F2-MKT-{H}'], split)['mean_bps']
                sD1 = stats(series[f'F2-D1-{H}'], split)['mean_bps']
                spread = sL - sD1
                lls = 100.0 * (sL - sM) / spread if spread > 0 else np.nan
            else:
                lls = 100.0        # A1's published form IS long-only; there is no short leg to miss
            bk = book_sim(p, tls, 'adv' if c['fam'] == 'A1' else 'r2_abn')
            wks = max((pd.Timestamp(b) - pd.Timestamp(a)).days / 7.0, 1)
            bk_alpha = (bk[ac].to_numpy() - bk['cost_bps'].to_numpy() / 1e4)
            rows.append(dict(cell=name, split=split, kind=c['kind'], H=c['H'],
                             n_trades=len(tls),
                             raw_bps=st_raw['mean_bps'], raw_t=st_raw['t'],
                             bench=bn or 'self (L-S)',
                             **st,
                             gross_bps_per_trade=np.nanmean(g) * 1e4,
                             alpha_bps_per_trade=be,
                             net_alpha_bps_per_trade=np.nanmean(aln) * 1e4,
                             honest_cost_bps=honest, breakeven_bps=be,
                             breakeven_x_cost=be / honest if honest else np.nan,
                             net_ex_top5_bps=ex5 * 1e4, net_wincap3x_bps=capped * 1e4,
                             long_leg_share_pct=lls,
                             book_trades=len(bk), book_trades_per_wk=len(bk) / wks,
                             book_usd_per_month=float(np.nansum(bk_alpha) * POS_USD /
                                                      max(wks / 4.345, 1)),
                             ))
    cdf = pd.DataFrame(rows)

    # ---- permutation / block-bootstrap p on the TRAIN excess, Sidak-adjusted across the 8 cells
    rng = np.random.default_rng(20260918)
    SCORED = [c['cell'] for c in cells if c['cell'] != 'A1-rob']
    pv = {}
    for c in cells:
        bn = bench_of.get(c['cell'])
        exc = series[c['cell']] - series[bn] if bn else series[c['cell']]
        a, b = SPLITS['TRAIN']
        m = exc[(exc.index >= a) & (exc.index <= b)]
        m = m[m != 0].to_numpy()
        if len(m) < 12:
            pv[c['cell']] = np.nan; continue
        obs = m.mean()
        nb = len(m) // 12
        boot = np.array([rng.choice(m, size=len(m), replace=True).mean() -
                         obs for _ in range(5000)])
        p1 = float((np.abs(boot) >= abs(obs)).mean())
        pv[c['cell']] = p1
    cdf['boot_p_train'] = cdf['cell'].map(pv)
    cdf['sidak_p_8cells'] = 1 - (1 - cdf['boot_p_train']) ** len(SCORED)

    cdf.to_csv(f'{OUT}/cells.csv', index=False)
    for k, v in series.items():
        v.to_csv(f'{OUT}/monthly_{k}.csv')
    pd.concat(trade_tabs, ignore_index=True).drop(columns=['acceptance_utc', 'acceptance_et'],
                                                  errors='ignore').to_csv(f'{OUT}/trades.csv', index=False)
    log('done -> %s/cells.csv' % OUT)
    with pd.option_context('display.width', 260, 'display.max_columns', 40):
        print(cdf[['cell', 'split', 'n_trades', 'raw_bps', 'mean_bps', 't', 'pct_pos', 'ex_jan_bps',
                   'mde_bps', 'net_alpha_bps_per_trade', 'breakeven_x_cost', 'long_leg_share_pct',
                   'book_trades_per_wk', 'book_usd_per_month', 'sidak_p_8cells']].round(2).to_string(index=False))


if __name__ == '__main__':
    sys.exit(main())
