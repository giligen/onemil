#!/usr/bin/env python3
"""frames16 ARM 1 — does the instrument carry the Cont-Kukanov-Stoikov contemporaneous relation?

PREREG §2. Per (dataset, schema, symbol): 10-second non-overlapping windows over RTH,
depth-normalised OFI against the mid-price change in the same window, OLS.

    e_n = 1{Pb_n >= Pb_{n-1}} qb_n - 1{Pb_n <= Pb_{n-1}} qb_{n-1}
        - 1{Pa_n <= Pa_{n-1}} qa_n + 1{Pa_n >= Pa_{n-1}} qa_{n-1}
    OFI_w = sum e_n over w ;  D_w = mean (qb+qa)/2 over w
    dmid_w = mid(last event in w) - mid(last event in w-1)
    dmid_w = a + b * OFI_w / D_w + eps

Written from the paper's prose, not copied from `hod_filter_stack/ofi.py`; the two implementations
are compared event-for-event in §`selfcheck` below (the independent-rebuild rail).

Also reported: the number of genuine quote CHANGES each instrument delivers, and therefore the
fraction of `mbp-1` quote events a 1-second snapshot loses on the same name/session/venue.
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
DAY = os.environ.get('F16_DAY', '2026-03-11')
WIN_NS = 10_000_000_000          # 10 seconds
RTH0 = pd.Timestamp(f'{DAY}T13:30:00Z').value
RTH1 = pd.Timestamp(f'{DAY}T20:00:00Z').value

PAIRS = [('EQUS.MINI', 'mbp-1'), ('EQUS.MINI', 'bbo-1s'),
         ('XNAS.ITCH', 'mbp-1'), ('XNAS.ITCH', 'bbo-1s'),
         ('XNAS.BASIC', 'cmbp-1'), ('XNAS.BASIC', 'cbbo-1s'),
         ('XNYS.PILLAR', 'mbp-1'), ('XNYS.PILLAR', 'bbo-1s')]


def cks_increments(bp, bs, ap, asz):
    """CKS order-flow-imbalance increments from consecutive best-quote snapshots (n-1 values)."""
    up_b = bp[1:] >= bp[:-1]
    dn_b = bp[1:] <= bp[:-1]
    dn_a = ap[1:] <= ap[:-1]
    up_a = ap[1:] >= ap[:-1]
    return up_b * bs[1:] - dn_b * bs[:-1] - dn_a * asz[1:] + up_a * asz[:-1]


def load(ds, sch):
    st = db.DBNStore.from_file(f'{RAW}/{ds}_{sch}_{DAY}.dbn.zst')
    d = st.to_df()
    d = d.reset_index()
    tcol = 'ts_recv' if 'ts_recv' in d.columns else 'ts_event'
    d['ts'] = pd.to_datetime(d[tcol], utc=True).astype('int64')
    keep = ['ts', 'symbol', 'bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00', 'publisher_id']
    d = d[[c for c in keep if c in d.columns]].copy()
    d['symbol'] = d.symbol.astype(str)
    return d


def fit_one(g):
    """Return the calibration row for one (instrument, symbol) quote sequence."""
    g = g.sort_values('ts', kind='mergesort')
    bp = g.bid_px_00.values.astype(float); ap = g.ask_px_00.values.astype(float)
    bs = g.bid_sz_00.values.astype(float); asz = g.ask_sz_00.values.astype(float)
    ts = g.ts.values.astype('int64')
    n_rec = len(g)
    ok = (np.isfinite(bp) & np.isfinite(ap) & (bp > 0) & (ap > 0) & (ap > bp)
          & (bs > 0) & (asz > 0) & (ts >= RTH0) & (ts < RTH1))
    bp, ap, bs, asz, ts = bp[ok], ap[ok], bs[ok], asz[ok], ts[ok]
    if len(bp) < 50:
        return dict(n_rec=n_rec, n_ok=len(bp), n_chg=0, n_win=0, r2=np.nan, beta=np.nan,
                    t=np.nan, mean_depth=np.nan, mean_spread_bps=np.nan)
    chg = np.ones(len(bp), bool)
    chg[1:] = ((bp[1:] != bp[:-1]) | (ap[1:] != ap[:-1])
               | (bs[1:] != bs[:-1]) | (asz[1:] != asz[:-1]))
    n_chg = int(chg.sum()) - 1
    e = cks_increments(bp, bs, ap, asz)
    mid = (bp + ap) / 2.0
    depth = (bs + asz) / 2.0
    w = (ts[1:] - RTH0) // WIN_NS                 # the window each INCREMENT belongs to
    df = pd.DataFrame({'w': w, 'e': e, 'd': depth[1:], 'mid': mid[1:]})
    agg = df.groupby('w').agg(ofi=('e', 'sum'), dep=('d', 'mean'), mid_end=('mid', 'last'))
    agg = agg.reindex(range(int(agg.index.min()), int(agg.index.max()) + 1))
    agg['mid_prev'] = agg.mid_end.shift(1)
    agg['dmid'] = agg.mid_end - agg.mid_prev
    a = agg[(agg.dep > 0) & np.isfinite(agg.dmid) & np.isfinite(agg.ofi)]
    if len(a) < 30:
        return dict(n_rec=n_rec, n_ok=len(bp), n_chg=n_chg, n_win=len(a), r2=np.nan,
                    beta=np.nan, t=np.nan, mean_depth=float(np.mean(depth)),
                    mean_spread_bps=float(np.mean((ap - bp) / mid * 1e4)))
    x = (a.ofi / a.dep).values
    y = a.dmid.values
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ beta
    ss_res = float(((y - yh) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    dof = len(x) - 2
    s2 = ss_res / dof if dof > 0 else np.nan
    var_b = s2 * np.linalg.pinv(X.T @ X)[1, 1]
    tstat = float(beta[1] / np.sqrt(var_b)) if var_b > 0 else np.nan
    return dict(n_rec=n_rec, n_ok=len(bp), n_chg=n_chg, n_win=len(a), r2=float(r2),
                beta=float(beta[1]), t=tstat, mean_depth=float(np.mean(depth)),
                mean_spread_bps=float(np.mean((ap - bp) / mid * 1e4)))


def selfcheck():
    """Independent-rebuild rail: this file's increments vs `hod_filter_stack/ofi.py::cks`."""
    sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
    rng = np.random.default_rng(16)
    bp = np.round(np.cumsum(rng.normal(0, .01, 5000)) + 50, 2)
    ap = bp + np.round(rng.uniform(.01, .10, 5000), 2)
    bs = rng.integers(1, 40, 5000).astype(float)
    asz = rng.integers(1, 40, 5000).astype(float)
    mine = cks_increments(bp, bs, ap, asz)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'ofi_prior', f'{ROOT}/research/mature_method/hod_filter_stack/ofi.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules['ofi_prior'] = m
    try:
        spec.loader.exec_module(m)
    except Exception:
        pass
    theirs = m.cks(bp, bs, ap, asz)
    d = float(np.max(np.abs(mine - theirs)))
    print(f'[selfcheck] frames16 increments vs hod_filter_stack/ofi.py::cks  max|diff| = {d:.3g} '
          f'over {len(mine):,} synthetic events', flush=True)
    assert d == 0.0, 'the two CKS implementations disagree'


def main():
    selfcheck()
    rows = []
    for ds, sch in PAIRS:
        d = load(ds, sch)
        for sym, g in d.groupby('symbol'):
            r = fit_one(g)
            r.update(dataset=ds, schema=sch, symbol=sym)
            rows.append(r)
            print(f'{ds:12s} {sch:8s} {sym:5s} rec {r["n_rec"]:>9,} ok {r["n_ok"]:>9,} '
                  f'chg {r["n_chg"]:>9,} win {r["n_win"]:>5,}  R2 {r["r2"]:.4f}  '
                  f'beta {r["beta"]:+.3e} (t {r["t"]:+.1f})  depth {r["mean_depth"]:.0f} '
                  f'spr {r["mean_spread_bps"]:.1f}bps', flush=True)
        del d
    out = pd.DataFrame(rows)
    out.to_csv(f'{D}/calib_{DAY}.csv', index=False)

    print('\n== quote events lost at 1-second sampling (same name, same session, same venue) ==',
          flush=True)
    loss = []
    for ds, full, snap in (('EQUS.MINI', 'mbp-1', 'bbo-1s'), ('XNAS.ITCH', 'mbp-1', 'bbo-1s'),
                           ('XNAS.BASIC', 'cmbp-1', 'cbbo-1s'), ('XNYS.PILLAR', 'mbp-1', 'bbo-1s')):
        f = out[(out.dataset == ds) & (out.schema == full)].set_index('symbol')
        s = out[(out.dataset == ds) & (out.schema == snap)].set_index('symbol')
        for sym in f.index.intersection(s.index):
            fr = 1.0 - s.loc[sym, 'n_chg'] / max(f.loc[sym, 'n_chg'], 1)
            loss.append(dict(dataset=ds, symbol=sym, n_chg_full=int(f.loc[sym, 'n_chg']),
                             n_chg_snap=int(s.loc[sym, 'n_chg']), lost=fr,
                             r2_full=f.loc[sym, 'r2'], r2_snap=s.loc[sym, 'r2']))
            print(f'  {ds:12s} {sym:5s}  {int(f.loc[sym,"n_chg"]):>9,} -> '
                  f'{int(s.loc[sym,"n_chg"]):>7,}  lost {fr*100:5.1f} %   '
                  f'R2 {f.loc[sym,"r2"]:.3f} -> {s.loc[sym,"r2"]:.3f}', flush=True)
    pd.DataFrame(loss).to_csv(f'{D}/calib_loss_{DAY}.csv', index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
