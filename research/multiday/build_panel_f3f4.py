#!/usr/bin/env python3
"""Build the matrix panel for F3 (momentum) and F4 (weekly industry-adjusted reversal).

Differences from `build_panel.py` (the F2/A1 panel):
  1. Universe is EVERY `kind=='common'` symbol (+SPY), not only the 8-K filers -- F3/F4 are
     cross-sectional families and have no event join.
  2. **ADV20$ is built on the RAW panel.**  `run_f2_a1.py` built dollar volume as
     `vwap_adj x volume_adj` off the split+dividend adjusted pull.  Volume is split-invariant in
     dollar terms, but the DIVIDEND factor scales the price down without touching volume, and that
     factor is a function of every dividend paid between the bar and today -- i.e. the future.  The
     $1M liquidity gate was therefore mildly forward-looking (defect #6 of REPORT_F2_A1.md).
     This panel stores `dvol_raw = vwap_raw x volume_raw` (the true dollars traded) and also
     measures the size of the bug it fixes.

Writes  data/panel_f3f4.npz  and  data/adv_fix_check.json.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

D = '/home/ec2-user/onemil/research/multiday/data'
OUT = f'{D}/panel_f3f4.npz'
YEARS = list(range(2016, 2027))


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def trailing_adv(dvol, w=20, min_obs=10, chunk=400):
    """20-session trailing mean of dollar volume ENDING at each session (inclusive).

    Chunked over symbols: the node has 7.8 GB shared with a live trader, so the float64
    cumulative sums are never materialised for the whole panel at once.
    """
    out = np.empty(dvol.shape, dtype=np.float32)
    for s in range(0, dvol.shape[0], chunk):
        e = min(s + chunk, dvol.shape[0])
        blk = dvol[s:e]
        dv = np.nan_to_num(blk, nan=0.0).astype(np.float64)
        ok = np.isfinite(blk).astype(np.float64)
        cs = np.cumsum(dv, axis=1)
        co = np.cumsum(ok, axis=1)
        num = cs.copy()
        den = co.copy()
        num[:, w:] = cs[:, w:] - cs[:, :-w]
        den[:, w:] = co[:, w:] - co[:, :-w]
        with np.errstate(invalid='ignore', divide='ignore'):
            out[s:e] = np.where(den >= min_obs, num / np.maximum(den, 1), np.nan).astype(np.float32)
        del dv, ok, cs, co, num, den
    return out


def main():
    uni = pd.read_parquet(f'{D}/universe.parquet',
                          columns=['symbol', 'kind', 'sic2', 'easy_to_borrow', 'shortable', 'exchange'])
    common = uni.loc[uni['kind'] == 'common'].copy()
    target = sorted(set(common['symbol']) | {'SPY'})
    log(f'universe rows {len(uni)} | common {len(common)} | target (common+SPY) {len(target)}')

    tset = set(target)
    sidx = {s: i for i, s in enumerate(target)}

    dates = []
    for y in YEARS:
        df = pq.read_table(f'{D}/prices_by_year/all/year={y}.parquet',
                           columns=['symbol', 'date']).to_pandas()
        dates.append(df.loc[df['symbol'].astype(str) == 'SPY', 'date'].values)
        del df
    cal = np.sort(np.unique(np.concatenate(dates)))
    didx = {d: i for i, d in enumerate(cal)}
    log(f'sessions {len(cal)}  {cal[0]} -> {cal[-1]}')

    n_s, n_d = len(target), len(cal)
    close_adj = np.full((n_s, n_d), np.nan, dtype=np.float32)
    close_raw = np.full((n_s, n_d), np.nan, dtype=np.float32)
    dvol_raw = np.full((n_s, n_d), np.nan, dtype=np.float32)
    dvol_adj = np.full((n_s, n_d), np.nan, dtype=np.float32)   # the BUGGY series, kept for the audit

    for y in YEARS:
        for adj in ('all', 'raw'):
            df = pq.read_table(f'{D}/prices_by_year/{adj}/year={y}.parquet',
                               columns=['symbol', 'date', 'close', 'volume', 'vwap']).to_pandas()
            df['symbol'] = df['symbol'].astype(str)
            df = df[df['symbol'].isin(tset)]
            si = df['symbol'].map(sidx).to_numpy(dtype=np.int32)
            di = df['date'].map(didx)
            ok = di.notna().to_numpy()
            di = di.fillna(0).to_numpy(dtype=np.int32)
            si, di = si[ok], di[ok]
            c = df['close'].to_numpy(dtype=np.float32)[ok]
            dv = (df['vwap'].to_numpy(dtype=np.float64) *
                  df['volume'].to_numpy(dtype=np.float64))[ok]
            if adj == 'all':
                close_adj[si, di] = c
                dvol_adj[si, di] = dv.astype(np.float32)
            else:
                close_raw[si, di] = c
                dvol_raw[si, di] = dv.astype(np.float32)
            del df, si, di, c, dv
        log(f'  {y} loaded')

    # ---- the ADV fix, measured -------------------------------------------------
    adv_raw = trailing_adv(dvol_raw)
    adv_adj = trailing_adv(dvol_adj)

    rng = np.random.default_rng(20260918)
    fin = np.isfinite(adv_raw) & np.isfinite(adv_adj)
    ii, jj = np.nonzero(fin)
    pick = rng.choice(len(ii), size=200, replace=False)
    ki, kj = ii[pick], jj[pick]
    rel = (adv_adj[ki, kj] - adv_raw[ki, kj]) / adv_raw[ki, kj]
    keys = [{'symbol': target[int(a)], 'session': str(cal[int(b)]),
             'adv_raw': float(adv_raw[a, b]), 'adv_adj_buggy': float(adv_adj[a, b]),
             'rel_err': float(r)} for a, b, r in zip(ki, kj, rel)]

    for gate in (1e6, 1e7):
        pass
    memb = {}
    for gate, name in ((1e6, '1M'), (1e7, '10M')):
        g_raw = adv_raw >= gate
        g_adj = adv_adj >= gate
        both = fin
        memb[name] = dict(
            n_cells=int(both.sum()),
            pass_raw=int((g_raw & both).sum()),
            pass_adj=int((g_adj & both).sum()),
            only_raw=int((g_raw & ~g_adj & both).sum()),     # admitted by the FIX
            only_adj=int((g_adj & ~g_raw & both).sum()),     # admitted by the BUG
        )

    out = dict(
        n_keys=200,
        rel_err_median=float(np.median(rel)), rel_err_mean=float(rel.mean()),
        rel_err_p01=float(np.percentile(rel, 1)), rel_err_p99=float(np.percentile(rel, 99)),
        rel_err_min=float(rel.min()), rel_err_max=float(rel.max()),
        n_keys_understated=int((rel < -1e-6).sum()), n_keys_exact=int((np.abs(rel) <= 1e-6).sum()),
        n_keys_overstated=int((rel > 1e-6).sum()),
        membership=memb,
        keys=keys[:20],
    )
    with open(f'{D}/adv_fix_check.json', 'w') as f:
        json.dump(out, f, indent=1)
    log(f'ADV fix: median rel err {out["rel_err_median"]:+.4%} '
        f'p01 {out["rel_err_p01"]:+.4%} p99 {out["rel_err_p99"]:+.4%}')
    for k, v in memb.items():
        log(f'  gate ${k}: raw {v["pass_raw"]:,} vs buggy {v["pass_adj"]:,} '
            f'(+{v["only_raw"]:,} admitted by the fix / -{v["only_adj"]:,} by the bug) '
            f'of {v["n_cells"]:,} cells')

    sic2 = common.set_index('symbol')['sic2'].reindex(target)
    np.savez_compressed(
        OUT, close_adj=close_adj, close_raw=close_raw, dvol_raw=dvol_raw,
        adv_raw=adv_raw, adv_adj_buggy=adv_adj,
        symbols=np.array(target, dtype=object),
        sessions=np.array([str(d) for d in cal], dtype=object),
        sic2=np.array([('' if pd.isna(v) else str(v)) for v in sic2], dtype=object),
        etb=np.array(common.set_index('symbol')['easy_to_borrow'].reindex(target).fillna(False)
                     .to_numpy(dtype=bool)),
    )
    log(f'wrote {OUT}  close_adj coverage {np.isfinite(close_adj).mean():.3f}')


if __name__ == '__main__':
    sys.exit(main())
