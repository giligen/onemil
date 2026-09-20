#!/usr/bin/env python3
"""frames16 ARM 2 — does the 1-second instrument reproduce the FULL-DEPTH lambda?

PREREG §3 makes arm 2 conditional on arm 1. Arm 1 left the gate ambiguous: on the HOD-population
small cap the 1-second R2 reads 0.406 on one session and 0.070 on the other. Rather than decide the
arm on a judgement call, this measures the thing arm 2 would actually use.

For each of the two calibrated sessions and each of the three names, build lambda on the SAME
one-minute windows from `mbp-1` (the reference) and from `bbo-1s` (what arm 2 could afford), and
report the rank correlation between them, the sign-agreement rate, and the rank correlation of
lambda with the window's own return (the degenerate limit: when OFI is noise, lambda collapses to
1/OFI x dmid and its cross-section IS the return, which the programme has already scored many
times).

No new data is pulled: this reads the DBN files arm 1 already stored under `raw/`.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
import databento as db  # noqa: E402

D = f'{ROOT}/research/mature_method/frames16'
RAW = f'{D}/raw'
DAYS = ['2026-03-11', '2026-05-11']
SYMS = ['AAPL', 'FSLY', 'USAX']
WIN_NS = 60_000_000_000            # ONE MINUTE — lambda's own window in arm 2


def cks(bp, bs, ap, asz):
    return ((bp[1:] >= bp[:-1]) * bs[1:] - (bp[1:] <= bp[:-1]) * bs[:-1]
            - (ap[1:] <= ap[:-1]) * asz[1:] + (ap[1:] >= ap[:-1]) * asz[:-1])


def per_minute(ds, sch, day):
    st = db.DBNStore.from_file(f'{RAW}/{ds}_{sch}_{day}.dbn.zst')
    d = st.to_df().reset_index()
    tcol = 'ts_recv' if 'ts_recv' in d.columns else 'ts_event'
    d['ts'] = pd.to_datetime(d[tcol], utc=True).astype('int64')
    d['symbol'] = d.symbol.astype(str)
    t0 = pd.Timestamp(f'{day}T13:30:00Z').value
    t1 = pd.Timestamp(f'{day}T20:00:00Z').value
    out = []
    for sym, g in d.groupby('symbol'):
        g = g.sort_values('ts', kind='mergesort')
        bp = g.bid_px_00.values.astype(float); ap = g.ask_px_00.values.astype(float)
        bs = g.bid_sz_00.values.astype(float); asz = g.ask_sz_00.values.astype(float)
        ts = g.ts.values.astype('int64')
        ok = (np.isfinite(bp) & np.isfinite(ap) & (bp > 0) & (ap > 0) & (ap > bp)
              & (bs > 0) & (asz > 0) & (ts >= t0) & (ts < t1))
        bp, ap, bs, asz, ts = bp[ok], ap[ok], bs[ok], asz[ok], ts[ok]
        if len(bp) < 50:
            continue
        e = cks(bp, bs, ap, asz)
        mid = (bp + ap) / 2.0
        dep = (bs + asz) / 2.0
        w = (ts[1:] - t0) // WIN_NS
        a = pd.DataFrame({'w': w, 'e': e, 'd': dep[1:], 'mid': mid[1:]}).groupby('w').agg(
            ofi=('e', 'sum'), dep=('d', 'mean'), mid_end=('mid', 'last'), n=('e', 'size'))
        a = a.reindex(range(int(a.index.min()), int(a.index.max()) + 1))
        a['dmid'] = (a.mid_end - a.mid_end.shift(1)) / a.mid_end.shift(1) * 100.0
        a['flow'] = a.ofi / a.dep
        a['lam'] = np.where(np.abs(a.flow) > 1e-6, a.dmid / a.flow, np.nan)
        a['symbol'] = sym
        out.append(a.reset_index())
    del d
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    rows = []
    for day in DAYS:
        for ds, full, snap in (('XNAS.ITCH', 'mbp-1', 'bbo-1s'),
                               ('EQUS.MINI', 'mbp-1', 'bbo-1s')):
            f = per_minute(ds, full, day)
            s = per_minute(ds, snap, day)
            if not len(f) or not len(s):
                continue
            m = f.merge(s, on=['symbol', 'w'], suffixes=('_f', '_s'))
            for sym, g in m.groupby('symbol'):
                g = g[np.isfinite(g.lam_f) & np.isfinite(g.lam_s)]
                if len(g) < 50:
                    continue
                rho = float(pd.Series(g.lam_f).corr(pd.Series(g.lam_s), method='spearman'))
                sgn = float((np.sign(g.lam_f) == np.sign(g.lam_s)).mean())
                rho_ret_f = float(pd.Series(g.lam_f).corr(pd.Series(g.dmid_f), method='spearman'))
                rho_ret_s = float(pd.Series(g.lam_s).corr(pd.Series(g.dmid_s), method='spearman'))
                rho_flow = float(pd.Series(g.flow_f).corr(pd.Series(g.flow_s), method='spearman'))
                rows.append(dict(day=day, dataset=ds, symbol=sym, n=len(g), rho_lambda=rho,
                                 sign_agree=sgn, rho_flow=rho_flow,
                                 rho_lam_ret_full=rho_ret_f, rho_lam_ret_snap=rho_ret_s))
                print(f'{day} {ds:10s} {sym:5s} n={len(g):4,}  rho(lambda full, lambda 1s) '
                      f'{rho:+.3f}  sign-agree {sgn*100:5.1f} %  rho(flow full, flow 1s) '
                      f'{rho_flow:+.3f}  |  rho(lambda, return): full {rho_ret_f:+.3f} '
                      f'1s {rho_ret_s:+.3f}', flush=True)
            del f, s, m
    pd.DataFrame(rows).to_csv(f'{D}/lam_check.csv', index=False)
    print('\n[arm2 gate] a lambda that the affordable instrument cannot reproduce is not a '
          'measurement of lambda.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
