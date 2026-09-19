#!/usr/bin/env python3
"""Step 1 - ONE tape walk over the honest population, emitting EVERY red-to-green signal for BOTH scan rules.

Nothing is gated here except what the scan itself needs. PDR, the price floor, the R floor, the 60 bps cap,
the last-entry cut, ADV20, the slots and the cost are ALL applied downstream in score.py, so a single walk
serves every declared cell in PREREG section 7.

Scan rules (PREREG section 2):
  S1  first-break-then-floor  - the first bar whose high reaches the level is the only candidate; if its
                                pre-signal range is below the floor (or run_lo[i] >= level) the day is dead.
                                This is what implementations A and B measured.
  S2  floor-and-break         - the first bar at which the floor holds AND the level breaks AND
                                run_lo[i] < level; floor-failing bars are SKIPPED.
                                This is trading/red_to_green.detect, i.e. the live engine.

Variants emitted (label = level_mult / floor_pct / scan):
  lvl1003_f5_S1, lvl1003_f5_S2      the declared book (B0 and 18 of the 22 cells)
  lvl1003_f8_S2, lvl1003_f10_S2     the floor ladder for S2 (S1's signal bar does not move with the floor,
                                    so F2/F3 for S1 are a downstream test on floor_val)
  lvl1000_f5_S1/S2                  cell L2 - the level A actually ran (prior close exactly)
  lvl1006_f5_S1/S2                  cell L3

Emission cut: signal minute <= 930 (15:30, the top of the declared last-entry ladder).
Exits are walked ONCE per distinct fill bar (hold / 2R / partial), identical to trading.hod_break.walk_exit.
"""
import os, sys, time
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/H/F6_reconcile')
from pipeline import Bars, walk
from trading.hod_break import rv_profile

OUT = 'research/mature_method/red_to_green'
SIG_MAX = 930
VARIANTS = [('lvl1003_f5', 1.003, 5.0, True), ('lvl1003_f8', 1.003, 8.0, False),
            ('lvl1003_f10', 1.003, 10.0, False), ('lvl1000_f5', 1.000, 5.0, True),
            ('lvl1006_f5', 1.006, 5.0, True)]
log = lambda *a: (print(*a), sys.stdout.flush())

COLS = ['day', 'symbol', 'variants', 'src', 'pdr', 'prev_close', 'adv20', 'o0', 'level', 'sig_m', 'floor_val',
        'dist_open_pct', 'rv', 'entry_m', 'entry', 'next_clock', 'stop', 'R', 'r_pct', 'over_cap_bps',
        'hold_grossR', 'hold_exit_m', 'hold_exit_type', 'hold_legs',
        'r2_grossR', 'r2_exit_m', 'r2_exit_type', 'r2_legs',
        'partial_grossR', 'partial_exit_m', 'partial_exit_type', 'partial_legs']


def scan_s1(h, run_hi, run_lo, level, floor_pct, n):
    """First bar whose high reaches the level; then the floor and the stop test decide the day."""
    for i in range(1, n):
        if h[i] >= level:
            lo_prev = run_lo[i - 1]
            if lo_prev <= 0 or (run_hi[i - 1] - lo_prev) / lo_prev * 100.0 < floor_pct:
                return None
            if run_lo[i] >= level:
                return None
            return i
    return None


def scan_s2(h, run_hi, run_lo, level, floor_pct, n):
    """First bar at which the floor holds AND the level breaks AND the running low is below the level."""
    for i in range(1, n):
        lo_prev = run_lo[i - 1]
        if lo_prev <= 0 or (run_hi[i - 1] - lo_prev) / lo_prev * 100.0 < floor_pct:
            continue
        if h[i] < level or run_lo[i] >= level:
            continue
        return i
    return None


def main():
    pop = pd.read_csv(f'{OUT}/pop.csv', dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    pop = pop[pop.prev_close.notna() & (pop.prev_close > 0) & pop.prev_low.notna() & (pop.prev_low > 0)]
    pop = pop.reset_index(drop=True)
    log('population:', len(pop))
    bars = Bars()
    out = open(f'{OUT}/cands.csv', 'w')
    out.write(','.join(COLS) + '\n')
    nrows = 0
    stats = dict(no_bars=0, few_bars=0, not_red=0, no_signal=0, no_next=0, late=0, emitted=0)
    t0 = time.time()
    for k, r in enumerate(pop.itertuples(index=False)):
        if k % 25000 == 0:
            log('  %d/%d  rows=%d  %.1f min' % (k, len(pop), nrows, (time.time() - t0) / 60))
        got = bars.get(r.symbol, r.day, 'cache_first')
        if got is None:
            stats['no_bars'] += 1; continue
        m, o, h, l, c, v, src = got
        n = len(m)
        if n < 2:
            stats['few_bars'] += 1; continue
        o0 = float(o[0])
        if not (o0 < r.prev_close):
            stats['not_red'] += 1; continue
        run_hi = np.maximum.accumulate(h); run_lo = np.minimum.accumulate(l); cumv = np.cumsum(v)
        by_sig = {}
        for lab, mult, floor_pct, do_s1 in VARIANTS:
            level = float(r.prev_close) * mult
            if do_s1:
                i1 = scan_s1(h, run_hi, run_lo, level, floor_pct, n)
                if i1 is not None:
                    by_sig.setdefault((i1, level), []).append(lab + '_S1')
            i2 = scan_s2(h, run_hi, run_lo, level, floor_pct, n)
            if i2 is not None:
                by_sig.setdefault((i2, level), []).append(lab + '_S2')
        if not by_sig:
            stats['no_signal'] += 1; continue
        walks = {}
        for (si, level), labs in by_sig.items():
            if m[si] > SIG_MAX:
                stats['late'] += 1; continue
            if si + 1 >= n:
                stats['no_next'] += 1; continue
            fi = si + 1
            entry = float(o[fi]); stop = float(run_lo[si])
            if stop <= 0 or entry <= stop:
                continue
            R = entry - stop
            if fi not in walks:
                walks[fi] = {mode: walk(o, h, l, c, m, fi, entry, stop, mode) for mode in ('hold', 'r2', 'partial')}
            w = walks[fi]
            lo_prev = run_lo[si - 1]
            fv = (run_hi[si - 1] - lo_prev) / lo_prev * 100.0 if lo_prev > 0 else 0.0
            adv = float(r.adv20) if r.adv20 == r.adv20 else 0.0
            rv = rv_profile(float(cumv[si]), adv, int(m[si])) if adv > 0 else 0.0
            row = [r.day, r.symbol, '|'.join(sorted(labs)), src,
                   '%.4f' % r.pdr, '%.4f' % r.prev_close, ('%.0f' % adv) if adv > 0 else '',
                   '%.4f' % o0, '%.4f' % level, int(m[si]), '%.3f' % fv,
                   '%.3f' % ((level / o0 - 1.0) * 100.0), '%.3f' % rv,
                   int(m[fi]), '%.4f' % entry, int(m[fi] == m[si] + 1), '%.4f' % stop,
                   '%.5f' % R, '%.4f' % (R / entry * 100.0),
                   '%.1f' % max(0.0, (entry / (level * 1.006) - 1.0) * 1e4)]
            for mode in ('hold', 'r2', 'partial'):
                em, et, gr, legs = w[mode]
                row += [int(em), et, '%.5f' % gr, ';'.join('%g:%s' % (ww, tt) for ww, tt in legs)]
            # reorder to COLS (grossR, exit_m, exit_type, legs)
            base = row[:20]
            tail = []
            for j, mode in enumerate(('hold', 'r2', 'partial')):
                em, et, gr, lg = row[20 + j * 4: 24 + j * 4]
                tail += [gr, em, et, lg]
            out.write(','.join(str(x) for x in base + tail) + '\n')
            nrows += 1; stats['emitted'] += 1
    out.close()
    log('stats:', stats)
    log('rows written:', nrows, ' %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
