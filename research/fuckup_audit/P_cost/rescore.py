#!/usr/bin/env python3
"""Stage P step 4 — re-score the honest ORB book with the MEASURED per-trade cost.

Cost is purely additive in this pipeline: selection (`_composite`, quintiles,
vetoes) never reads P&L, and `_rp_pnl = pnl * _rp_position / OLD_POS`.  So a
re-scored book is produced by writing a resim cache whose `pnl` / `pnl_pct`
carry the measured fills and running the SAME pipeline path
(`study_orb_pipeline_static_lock.py` + `ORB_BT_RESIM_CACHE`) that produced
`book_n8_q1on.csv`.  The picks are identical by construction; only the dollars
move.

Arms written (one resim cache each):
  asis      the shipped model: entry at range_high x 1.003, exit at the level
            less a flat 10 bps.  (= candidates_dump.csv, the parity control)
  measured  entry at the NBBO ASK at the fill instant (capped at the stop-limit
            price: an ask above the cap is NOT a fill -> $0, the live outcome),
            exit at the NBBO BID at the exit instant; the resting +3R scale leg
            pays nothing (a passive limit does not cross the spread).
  band      the same trades charged the BAND TABLE constant instead: entry
            0.25 x half-band-spread, exit 0.875 x (stop/lock) / 0.412 x (eod) /
            0.875 x (tag) of it, the Stage-A0 contract.

Usage: python3 rescore.py build     # write the resim caches
       python3 rescore.py run       # run the pipeline for each arm x {8,3} slots
       python3 rescore.py analyse   # the tables
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv          # noqa: E402

P = f'{ROOT}/research/fuckup_audit/P_cost'
D1 = f'{ROOT}/research/fuckup_audit/D1_orb'
def _old_pos() -> float:
    """`sizing.old_position_reference_usd` from orb.yaml — the BT share base.
    Read, never hardcoded: a wrong value rescales every measured P&L."""
    import yaml
    v = float((yaml.safe_load(open(f'{ROOT}/orb.yaml')).get('sizing') or {})
              .get('old_position_reference_usd', 50_000.0))
    return v


OLD_POS = _old_pos()
EXIT_SLIP = 10.0 / 10000.0
STOP_REASONS = {'stop', 'lock', 'scale_stop', 'scale_lock'}
SCALE_FRAC = 0.40
SCALE_LEVEL_R = 3.0
RATIO = {'stop': 0.875, 'lock': 0.875, 'scale_stop': 0.875, 'scale_lock': 0.875,
         'eod': 0.412, 'scale_eod': 0.412, 'tag_bb': 0.875, 'tag_b1': 0.875}
# 2026-09-19: corrected 0.25 -> 1.00.  The entry leg pays a FULL half-spread -- a buy that
# elects at a level lifts the offer (measured: red_to_green/REPORT.md §3,
# mature_method/entry_cost_audit/REPORT.md).  Only the `band` comparison arm uses this; the
# `measured` arm was never a contract and is unaffected.
ENTRY_COEF = 1.00
LEGACY_ENTRY_COEF = 0.25   # reproduce-only: the pre-2026-09-19 contract
ARMS = ('asis', 'zero', 'measured', 'strict', 'band')


# ---------------------------------------------------------------- build ----
def build():
    sp = pd.read_parquet(f'{P}/spreads.parquet')
    sp['key'] = sp.symbol + '|' + sp.date
    dump = read_orb_csv(f'{D1}/candidates_dump.csv')
    dump['date'] = pd.to_datetime(dump['date']).dt.strftime('%Y-%m-%d')
    dump['key'] = dump.symbol + '|' + dump.date

    s = sp.set_index('key')
    ent = dump.entered == 1
    shares = np.maximum(1, (OLD_POS / dump.entry_price).astype(int))
    # self-check: these shares must reproduce the dump's own P&L identity
    chk = np.abs(dump.pnl - dump.pnl_pct / 100 * dump.entry_price * shares)
    assert float(chk.max()) < 1e-6, (
        f'share base wrong (OLD_POS={OLD_POS}): max |pnl identity| {chk.max()}')

    # ---- measured (two entry conventions) -------------------------------
    # `measured` : fill at min(NBBO ask at the fill instant, the stop-limit
    #              cap).  An ask above the cap is treated the way the as-is
    #              book treats it -- the resting limit eventually gets its
    #              price -- so this arm isolates the SPREAD, not queue risk.
    # `strict`   : an ask above the cap is NOT a fill; the pick books $0
    #              (a slot spent, the live `time_stop_canceled` outcome).
    meas_pnl = dump.pnl.astype(float).copy()
    strict_pnl = dump.pnl.astype(float).copy()
    covered = np.zeros(len(dump), dtype=bool)
    nofill = 0

    def _leg(e_m, x_m, sh, reason, q):
        if str(reason).startswith('scale_'):
            rng = float(q.range_high) - float(q.range_low)
            spx = e_m + SCALE_LEVEL_R * rng           # resting limit, no spread
            qty = int(np.floor(SCALE_FRAC * sh))
            frac = qty / float(sh) if qty >= 1 else 0.0
            ret = frac * (spx / e_m - 1) + (1 - frac) * (x_m / e_m - 1)
            return ret * e_m * sh
        return (x_m - e_m) * sh

    for i, r in enumerate(dump.itertuples()):
        if r.entered != 1 or r.key not in s.index:
            continue
        q = s.loc[r.key]
        if not (q.cov_entry and q.cov_exit):
            continue
        ask = float(q.measured_entry)
        x_m = float(q.measured_exit)
        if not np.isfinite(ask) or not np.isfinite(x_m):
            continue
        covered[i] = True
        cap = float(r.entry_price)                    # = range_high * 1.003
        sh = int(shares.iat[i])
        meas_pnl.iat[i] = _leg(min(ask, cap), x_m, sh, r.exit_reason, q)
        if ask > cap * (1 + 1e-12):
            nofill += 1
            strict_pnl.iat[i] = 0.0
        else:
            strict_pnl.iat[i] = _leg(ask, x_m, sh, r.exit_reason, q)

    # ---- zero-cost ceiling ------------------------------------------------
    # The shipped model's ONLY costs are the 30 bps entry buffer (the fill is
    # assumed at range_high x 1.003 rather than at range_high) and the 10 bps
    # exit slip.  Both are exact multiples of a level, so the zero-cost book is
    # recoverable in closed form and bounds what any cost correction can be
    # worth on this book.  Share count held at the as-is value in every arm so
    # the arms differ only in price.
    zero_pnl = dump.pnl.astype(float).copy()
    slip = 1 - EXIT_SLIP
    for i, r in enumerate(dump.itertuples()):
        if r.entered != 1 or r.key not in s.index:
            continue
        q = s.loc[r.key]
        rh = float(q.range_high)
        E = float(r.entry_price)
        sh = int(shares.iat[i])
        if str(r.exit_reason).startswith('scale_'):
            rng = rh - float(q.range_low)
            spx = E + SCALE_LEVEL_R * rng
            qty = int(np.floor(SCALE_FRAC * sh))
            frac = qty / float(sh) if qty >= 1 else 0.0
            ret = float(r.pnl) / (E * sh)
            t_sc = frac * (spx * slip / E - 1)
            run_eff = ((ret - t_sc) / (1 - frac) + 1) * E if frac < 1 else E
            run_lvl = run_eff / slip
            spx0 = rh + SCALE_LEVEL_R * rng
            zero_pnl.iat[i] = (frac * (spx0 / rh - 1)
                               + (1 - frac) * (run_lvl / rh - 1)) * rh * sh
        else:
            lvl = (float(r.pnl) / sh + E) / slip        # the exit LEVEL
            zero_pnl.iat[i] = (lvl - rh) * sh

    # ---- band ----
    # pb/hb/r_pct for EVERY entered row (not only the fetched ones), so the
    # band arm has the same coverage as the as-is arm.
    sys.path.insert(0, f'{ROOT}/research/fuckup_audit/P_cost')
    from spreads import band_table, PB_EDGES, PB_LAB, HB_EDGES, HB_LAB
    bt = band_table().band_bps
    xt = pd.read_csv(f'{P}/exit_times.csv', keep_default_na=False, na_values=[''],
                     dtype={'symbol': str, 'date': str},
                     usecols=['symbol', 'date', 'entry_ts', 'range_high', 'range_low'])
    xt['key'] = xt.symbol + '|' + xt.date
    et = pd.to_datetime(xt.entry_ts, utc=True).dt.tz_convert('America/New_York')
    xt['entry_m'] = et.dt.hour * 60 + et.dt.minute
    j = dump.merge(xt[['key', 'entry_m', 'range_high', 'range_low']], on='key', how='left')
    j['pb'] = pd.cut(j.entry_price, PB_EDGES, labels=PB_LAB, include_lowest=True).astype(str)
    j['hb'] = pd.cut(j.entry_m, HB_EDGES, labels=HB_LAB).astype(str)
    j['r_pct'] = (j.range_high - j.range_low) / j.entry_price * 100
    band_bps = np.array([bt.get((p, h), np.nan) for p, h in zip(j.pb, j.hb)])
    rr = np.maximum(j.r_pct.values, 0.05)
    half = 0.5 * (band_bps / 100.0) / rr               # half-spread in R
    ratio = dump.exit_reason.map(RATIO).fillna(0.875).values
    gross_r = (dump.pnl.values / (dump.entry_price.values * shares.values)) / (rr / 100.0)
    band_r = gross_r - ENTRY_COEF * half - half * ratio
    band_pnl = np.where(np.isfinite(band_r),
                        band_r * (rr / 100.0) * dump.entry_price.values * shares.values,
                        dump.pnl.values)

    for arm, pnl in (('asis', dump.pnl.astype(float).values),
                     ('zero', zero_pnl.values),
                     ('measured', meas_pnl.values),
                     ('strict', strict_pnl.values),
                     ('band', band_pnl)):
        out = read_orb_csv(f'{D1}/candidates_dump.csv')
        out['pnl'] = pnl
        out['pnl_pct'] = pnl / (out.entry_price * shares.values) * 100
        out.loc[out.entered != 1, ['pnl', 'pnl_pct']] = 0.0
        out.to_csv(f'{P}/dump_{arm}.csv', index=False)
    cov = covered[ent.values].mean()
    print(f'built 3 resim caches | measured coverage on entered rows '
          f'{cov * 100:.1f}% ({int(covered.sum())}/{int(ent.sum())}) | '
          f'ask-above-cap (no fill) {nofill}', flush=True)
    pd.DataFrame(dict(key=dump.key, entered=dump.entered, covered=covered,
                      pnl_asis=dump.pnl, pnl_meas=meas_pnl,
                      pnl_strict=strict_pnl, pnl_band=band_pnl)
                 ).to_csv(f'{P}/measured_cost.csv', index=False)


# ------------------------------------------------------------------ run ----
def run():
    env0 = dict(os.environ)
    env0['ORB_BT_FEATURES_CSV'] = 'analysis_results/orb_features_20260916_2053.csv'
    env0['ORB_BT_RISK'] = '375'
    for arm in ARMS:
        for n in (8, 3):
            e = dict(env0)
            e['ORB_BT_RESIM_CACHE'] = f'{P}/dump_{arm}.csv'
            e['ORB_BT_N'] = str(n)
            e['ORB_BT_ACCOUNT'] = repr(3333.333333333333 * n)
            e['ORB_SKIP_Q1'] = '1'
            e['ORB_BT_BOOK_OUT'] = f'{P}/book_{arm}_n{n}.csv'
            e['ORB_BT_MONTHLY_OUT'] = f'{P}/monthly_{arm}_n{n}.csv'
            with open(f'{P}/log_{arm}_n{n}.txt', 'w') as fh:
                rc = subprocess.call(['nice', '-n', '10', 'python3', '-u',
                                      'study_orb_pipeline_static_lock.py'],
                                     stdout=fh, stderr=subprocess.STDOUT, env=e)
            print(f'{arm} n={n} rc={rc}', flush=True)


# -------------------------------------------------------------- analyse ----
def split_of(day):
    return 'TRAIN' if day < '2026-01-01' else ('VAL' if day < '2026-06-01' else 'TEST')


def _mdd(daily):
    """D1's definition (analyze_grid.mdd): trough of the cumulative DAILY P&L."""
    cum = daily.cumsum()
    return float((cum - cum.cummax()).min())


_RNG = None


def range_lookup():
    """(range_high - range_low) per (symbol, date) — the R denominator, taken
    from the exit-time replay (the simulator's own range), never re-derived."""
    global _RNG
    if _RNG is None:
        x = pd.read_csv(f'{P}/exit_times.csv', keep_default_na=False, na_values=[''],
                        dtype={'symbol': str, 'date': str},
                        usecols=['symbol', 'date', 'range_high', 'range_low'])
        _RNG = dict(zip(zip(x.symbol, x.date), x.range_high - x.range_low))
    return _RNG


def book_stats(path):
    b = read_orb_csv(path)
    b['date'] = pd.to_datetime(b['date']).dt.strftime('%Y-%m-%d')
    b['split'] = b.date.map(split_of)
    b['month'] = b.date.str[:7]
    rl = range_lookup()
    rng = np.array([rl.get((s, dt), np.nan) for s, dt in zip(b.symbol, b.date)])
    shares = b['_rp_position'] / b['entry_price']
    b['R'] = np.where(rng > 0, b['_sized_pnl'] / (shares * rng), 0.0)
    out = {}
    for label, g in [('ALL', b)] + list(b.groupby('split')):
        mo = g.groupby('month')._sized_pnl.sum()
        out[label] = dict(picks=len(g), fills=int((g.entered.astype(float) != 0).sum()),
                          pnl=round(float(g._sized_pnl.sum()), 0),
                          r_per_pick=round(float(g.R.mean()), 3),
                          worst_mo=round(float(mo.min()), 0) if len(mo) else np.nan,
                          red_mo=int((mo < 0).sum()),
                          mdd=round(_mdd(g.groupby('date')._sized_pnl.sum()), 0))
    return out


def analyse():
    rows = []
    for arm in ARMS:
        for n in (8, 3):
            p = f'{P}/book_{arm}_n{n}.csv'
            if not os.path.exists(p):
                continue
            st = book_stats(p)
            for k, v in st.items():
                rows.append(dict(arm=arm, slots=n, split=k, **v))
    t = pd.DataFrame(rows)
    t.to_csv(f'{P}/rescore_table.csv', index=False)
    print(t.to_string(index=False), flush=True)


if __name__ == '__main__':
    {'build': build, 'run': run, 'analyse': analyse}[sys.argv[1]]()
