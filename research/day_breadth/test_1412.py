"""Cell 1,412 (PREREG_1412.md): base entry, C1 exit, kept iff BR(signal_m) >= 0.6115 — scored ONCE on TEST.

Builds BR for TEST days with breadth.symbol_minute_flags (one pass over TEST universe symbol-days), walks the TEST
base signals with run_consol.fill_c1 on the cached paths, attaches BR at signal_m, and prints the pass-bar numbers.
Writes TEST_1412.md and trades_test_1412.csv.

Usage: python3 research/day_breadth/test_1412.py
"""
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'research/hod_consol'))
sys.path.insert(0, str(HERE))
import run_consol as rc  # noqa: E402
import breadth as B  # noqa: E402
from adversarial_read import stats  # noqa: E402

EDGE = 0.6115


def test_breadth():
    """BR per (day, minute) for TEST days only."""
    cands = [(s, d) for s, d in rc.load_candidates() if rc.split_of(d) == 'TEST']
    days = sorted({d for _, d in cands})
    acc = {d: np.zeros((2, B.NM), dtype=np.int32) for d in days}
    sip = sqlite3.connect(rc.BARS_SIP_DB, timeout=30)
    rth = sqlite3.connect(rc.BARS_RTH_DB, timeout=30)
    t0 = time.time()
    for i, (s, d) in enumerate(cands, 1):
        bars, _ = rc.fetch_day_bars_dual(sip, rth, s, d)
        if bars is None or bars.empty:
            continue
        has, above, _, _ = B.symbol_minute_flags(bars)
        acc[d][0] += has
        acc[d][1] += above
        if i % 1000 == 0 or i == len(cands):
            print(f'[pass] {i}/{len(cands)} {time.time() - t0:.0f}s', flush=True)
    rows = []
    for d in days:
        with np.errstate(divide='ignore', invalid='ignore'):
            rows.append(pd.DataFrame(dict(day=d, m=np.arange(B.M0, B.M1 + 1), BR=acc[d][1] / acc[d][0])))
    return pd.concat(rows, ignore_index=True)


def main():
    br = test_breadth()
    sig = pd.read_parquet(ROOT / 'research/hod_consol/signals.parquet')
    sig = sig[sig.split == 'TEST'].copy()
    paths = pd.read_parquet(ROOT / 'research/hod_consol/paths.parquet')
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    w = rc.walk(sig, idx, rc.fill_c1)
    w['dm'] = w.signal_m.clip(B.M0, B.M1).astype(int)
    w = w.merge(br, left_on=['day', 'dm'], right_on=['day', 'm'], how='left').dropna(subset=['BR'])
    kept, rest = w[w.BR >= EDGE], w[w.BR < EDGE]
    keep = rc.simulate_slots(kept)
    slotted = kept.loc[keep.index[keep]]
    weeks = w.wk.nunique()
    k, r_, s_ = stats(kept.net_R, kept.day), stats(rest.net_R, rest.day), stats(slotted.net_R, slotted.day)
    se = k['mean'] / k['t_cluster'] if k.get('t_cluster') else float('nan')
    lines = ['# TEST_1412.md — cell 1,412 scored once on TEST (2026-06-01 .. 2026-09-18)', '',
             f'TEST base trades walked: {len(w)} over {w.day.nunique()} days, {weeks} weeks; kept (BR ≥ {EDGE}): '
             f'{len(kept)} on {kept.day.nunique()} days', '',
             '| cohort | n | mean net R | t (day-clustered) | day-weighted mean |', '|---|---|---|---|---|']
    for name, d in (('kept (risk-on tape)', k), ('rest', r_), ('kept, slotted (12/day, 4 at once)', s_)):
        lines.append(f"| {name} | {d.get('n')} | {d.get('mean', float('nan')):+.3f} | {d.get('t_cluster', float('nan')):+.2f} "
                     f"| {d.get('day_mean', float('nan')):+.3f} |")
    fills_wk = len(slotted) / weeks if weeks else 0.0
    passed = (k['mean'] >= 0.10 and k['mean'] - r_['mean'] >= 0.10 and k['t_cluster'] >= 2 and s_['mean'] > 0
              and fills_wk >= 3)
    consistent = k['mean'] > 0 and k['mean'] - r_['mean'] >= 0.10
    lines += ['', f'kept − rest = {k["mean"] - r_["mean"]:+.3f} R; slotted fills/week {fills_wk:.1f}; '
              f'SE(kept) ≈ {se:.3f} R → MDE at t = 2 ≈ {2 * se:.3f} R', '',
              f'**PASS = {passed}**; consistent-not-proven (sign + separation hold) = {consistent}']
    (HERE / 'TEST_1412.md').write_text('\n'.join(lines) + '\n')
    w.to_csv(HERE / 'trades_test_1412.csv', index=False)
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
