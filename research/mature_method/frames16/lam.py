#!/usr/bin/env python3
"""frames16 ARM 2 — the LAMBDA RESIDUAL at the break (PREREG §3).

Arm 1 disqualified EQUS.MINI (its own book explains 0.6 % of its own 10-s mid moves on AAPL and it
quotes 10.9 bps where the market quotes 1.0), so this arm pulls **XNAS.ITCH `bbo-1s`**, the cheapest
instrument arm 1 showed carries the contemporaneous relation on all three liquidity tiers
(R2 0.40 / 0.37 / 0.41).  Raw DBN is KEPT under `frames16/raw2/` — arm O's mistake was discarding it.

Per name-day:
    beta_nd  fitted on 10-s windows ending in [break_m - 35, break_m - 5) minutes  (the general
             population BEFORE the consolidation; >= 30 usable windows required)
    lam      = dmid_pct(break minute) / (OFI/depth over the break minute)
    resid    = dmid_pct(break minute) - beta_nd * (OFI/depth over the break minute)
Plus the DECLARED DIAGNOSTIC: forward mid returns at 10 s / 30 s / 60 s / 5 min after the break
bar's close (decay structure, never a money horizon).

dmid is in % of mid so lambda is comparable across price levels; beta is fitted in the same units.
Resumable per day; re-prices every request and stops at the pass budget.
"""
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
load_dotenv(f'{ROOT}/.env')
import databento as db  # noqa: E402

D = f'{ROOT}/research/mature_method/frames16'
RAW2 = f'{D}/raw2'
OUT, STATE = f'{D}/lam.csv', f'{D}/lam_state.csv'
DS, SCH = 'XNAS.ITCH', 'bbo-1s'
BUDGET = float(os.environ.get('FRAMES16_ARM2_BUDGET', '10.80'))   # $12 cap - $0.4726 arm 1 - margin
FIT_LO, FIT_HI = 35, 5        # the fit window, minutes before the break
BUCKET = 30
WIN_NS = 10_000_000_000
FWD = [10, 30, 60, 300]       # seconds, the declared decay diagnostic


def utc(day, minute):
    return (pd.Timestamp(day).tz_localize('America/New_York')
            + pd.Timedelta(minutes=int(minute))).tz_convert('UTC')


def cks(bp, bs, ap, asz):
    return ((bp[1:] >= bp[:-1]) * bs[1:] - (bp[1:] <= bp[:-1]) * bs[:-1]
            - (ap[1:] <= ap[:-1]) * asz[1:] + (ap[1:] >= ap[:-1]) * asz[:-1])


def windows(g, t0_ns):
    """10-second windows for one symbol's quote sequence -> (w, ofi, depth, mid_end, n_evt)."""
    g = g.sort_values('ts', kind='mergesort')
    bp = g.bid_px_00.values.astype(float); ap = g.ask_px_00.values.astype(float)
    bs = g.bid_sz_00.values.astype(float); asz = g.ask_sz_00.values.astype(float)
    ts = g.ts.values.astype('int64')
    ok = (np.isfinite(bp) & np.isfinite(ap) & (bp > 0) & (ap > 0) & (ap > bp)
          & (bs > 0) & (asz > 0))
    bp, ap, bs, asz, ts = bp[ok], ap[ok], bs[ok], asz[ok], ts[ok]
    if len(bp) < 5:
        return None
    e = cks(bp, bs, ap, asz)
    mid = (bp + ap) / 2.0
    dep = (bs + asz) / 2.0
    w = (ts[1:] - t0_ns) // WIN_NS
    df = pd.DataFrame({'w': w, 'e': e, 'd': dep[1:], 'mid': mid[1:]})
    a = df.groupby('w').agg(ofi=('e', 'sum'), dep=('d', 'mean'), mid_end=('mid', 'last'),
                            n=('e', 'size'))
    return a


def feats(a, t0_ns, day, break_m):
    """beta on the pre-consolidation windows, then lambda / residual at the break minute."""
    b0 = (utc(day, break_m).value - t0_ns) // WIN_NS          # first window of the break minute
    b1 = b0 + 6                                                # 6 x 10 s = the break minute
    f0 = (utc(day, break_m - FIT_LO).value - t0_ns) // WIN_NS
    f1 = (utc(day, break_m - FIT_HI).value - t0_ns) // WIN_NS
    idx = a.index.values
    fit = a[(idx >= f0) & (idx < f1) & (a.dep > 0)].copy()
    if len(fit) < 30:
        return None
    fit = fit.reindex(range(int(fit.index.min()), int(fit.index.max()) + 1))
    fit['dmid'] = (fit.mid_end - fit.mid_end.shift(1)) / fit.mid_end.shift(1) * 100.0
    fit = fit[np.isfinite(fit.dmid) & np.isfinite(fit.ofi) & (fit.dep > 0)]
    if len(fit) < 30:
        return None
    x = (fit.ofi / fit.dep).values
    y = fit.dmid.values
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yh = X @ beta
    ss_res = float(((y - yh) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    brk = a[(idx >= b0) & (idx < b1)]
    pre = a[(idx < b0)]
    if not len(brk) or not len(pre):
        return None
    m_pre = float(pre.mid_end.iloc[-1]); m_brk = float(brk.mid_end.iloc[-1])
    dep = float(brk.dep.mean()); ofi = float(brk.ofi.sum())
    if not (dep > 0) or m_pre <= 0:
        return None
    flow = ofi / dep
    dmid = (m_brk - m_pre) / m_pre * 100.0
    out = dict(beta=float(beta[1]), beta_r2=float(r2), n_fit=int(len(fit)),
               ofi_break=ofi, depth_break=dep, flow_break=flow, dmid_break=dmid,
               n_upd_break=int(brk.n.sum()),
               lam=(dmid / flow if abs(flow) > 1e-6 else np.nan),
               resid=dmid - float(beta[1]) * flow)
    for s in FWD:
        j = b1 + s // 10
        nxt = a[(idx >= b1) & (idx < j)]
        out[f'fwd{s}'] = ((float(nxt.mid_end.iloc[-1]) - m_brk) / m_brk * 100.0
                          if len(nxt) else np.nan)
    return out


def main():
    os.makedirs(RAW2, exist_ok=True)
    sig = pd.read_pickle(f'{ROOT}/research/mature_method/hod_filter_stack/b2.pkl')
    sig = sig[sig.split.isin(('TRAIN', 'VAL'))][['day', 'symbol', 'break_m']].drop_duplicates()
    done = set(pd.read_csv(STATE, dtype=str).day) if os.path.exists(STATE) else set()
    spent = float(pd.read_csv(STATE).cost.sum()) if os.path.exists(STATE) else 0.0
    days = [d for d in sorted(sig.day.unique()) if d not in done]
    print(f'B2 {len(sig):,} signals · {sig.day.nunique()} sessions · todo {len(days)} · '
          f'spent so far ${spent:.3f} of ${BUDGET:.2f}', flush=True)
    c = db.Historical(os.environ['DATABENTO_API_KEY'])
    for n, day in enumerate(days):
        g = sig[sig.day == day].copy()
        g['bk'] = (g.break_m // BUCKET).astype(int)
        rows, dcost = [], 0.0
        for bk, gg in g.groupby('bk'):
            syms = sorted(gg.symbol.unique())
            b0, b1 = int(gg.break_m.min()), int(gg.break_m.max())
            st = utc(day, b0 - FIT_LO); en = utc(day, b1 + 6)
            kw = dict(dataset=DS, symbols=syms, schema=SCH, stype_in='raw_symbol',
                      start=st.isoformat(), end=en.isoformat())
            try:
                cost = float(c.metadata.get_cost(**kw))
            except Exception as e:
                print(f'  COST FAIL {day}/{bk}: {str(e)[:70]}', flush=True); continue
            if spent + dcost + cost > BUDGET:
                print(f'BUDGET STOP at {day}/{bk}: ${spent+dcost:.3f} + ${cost:.3f} > '
                      f'${BUDGET:.2f}', flush=True)
                pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT),
                                          index=False) if rows else None
                print(f'ARM2 PARTIAL — spent ${spent+dcost:.3f}', flush=True)
                return 0
            try:
                data = c.timeseries.get_range(**kw)
            except Exception as e:
                print(f'  PULL FAIL {day}/{bk}: {str(e)[:70]}', flush=True); continue
            data.to_file(f'{RAW2}/{day}_{bk}.dbn.zst')
            dcost += cost
            q = data.to_df().reset_index()
            if not len(q):
                continue
            tcol = 'ts_recv' if 'ts_recv' in q.columns else 'ts_event'
            q['ts'] = pd.to_datetime(q[tcol], utc=True).astype('int64')
            q['symbol'] = q.symbol.astype(str)
            t0 = st.value
            for sym, qq in q.groupby('symbol'):
                a = windows(qq, t0)
                if a is None:
                    continue
                for bm in gg[gg.symbol == sym].break_m.unique():
                    f = feats(a, t0, day, int(bm))
                    if f is None:
                        continue
                    f.update(day=day, symbol=sym, break_m=int(bm))
                    rows.append(f)
            del q, data
        spent += dcost
        if rows:
            pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        pd.DataFrame([dict(day=day, cost=dcost, n=len(rows))]).to_csv(
            STATE, mode='a', header=not os.path.exists(STATE), index=False)
        if n % 10 == 0:
            print(f'{n+1}/{len(days)} {day} +{len(rows)} rows  spent ${spent:.3f}', flush=True)
    print(f'ARM2 PULL DONE — spent ${spent:.3f}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
