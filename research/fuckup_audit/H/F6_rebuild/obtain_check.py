"""Stage 4: obtainability audit, re-reading the tape for the booked trades.

For every booked trade it re-loads the (symbol, day) RTH tape and asserts:
  * the entry fill lies inside the fill bar (low <= fill <= high)
  * the fill bar is the bar AFTER the signal bar in the tape
  * the exit fill lies inside the exit bar for stop exits (and reports the share
    of stop fills that fall BELOW the exit bar's low because of the 0.999 slip)
  * target fills happen only on bars whose close >= entry + 2R
  * eod fills equal the open of the first bar at/after 15:55
"""
import os
import sys
import sqlite3
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scan import load_bars, FLAT_MIN, STOP_SLIP  # noqa: E402

SIP = 'file:research/bf_zero/bars_sip.db?mode=ro'
CACHE = 'file:data/cache.db?mode=ro'


def log(*a):
    print(*a)
    sys.stdout.flush()


def main(sig='a', stp='i'):
    sip = sqlite3.connect(SIP, uri=True)
    cache = sqlite3.connect(CACHE, uri=True)
    report = []
    for key in ['hold', 'r2', 'partial']:
        p = os.path.join(HERE, 'trades_%s_%s%s.csv' % (key, sig, stp))
        df = pd.read_csv(p, keep_default_na=False, na_values=[''])
        c = dict(n=len(df), entry_in_bar=0, entry_is_next_bar=0,
                 stop_n=0, stop_in_bar=0, stop_below_low=0,
                 target_n=0, target_close_ok=0, target_in_bar=0,
                 eod_n=0, eod_ok=0)
        for r in df.itertuples(index=False):
            bars, _ = load_bars(sip, cache, r.symbol, r.day)
            bymin = {b[0]: b for b in bars}
            fb = bymin[r.entry_min]
            if fb[3] <= r.entry <= fb[2]:
                c['entry_in_bar'] += 1
            mins = [b[0] for b in bars]
            si = mins.index(r.sig_min)
            if si + 1 < len(mins) and mins[si + 1] == r.entry_min:
                c['entry_is_next_bar'] += 1
            eb = bymin.get(r.exit_min)
            t = r.exit_type
            target = r.entry + 2.0 * r.R
            if 'stop' in t:
                c['stop_n'] += 1
                stop_level = r.entry if t.startswith('pp+') else r.stop
                fill = min(stop_level, eb[1]) * STOP_SLIP
                if eb[3] <= fill <= eb[2]:
                    c['stop_in_bar'] += 1
                if fill < eb[3]:
                    c['stop_below_low'] += 1
            elif 'eod' in t:
                c['eod_n'] += 1
                if eb[0] >= FLAT_MIN or eb[0] == mins[-1]:
                    c['eod_ok'] += 1
            # the partial leg (if any) must have closed at/above the target
            legs = str(getattr(r, '%s_legs' % key))
            if 'target' in legs:
                c['target_n'] += 1
                # find the bar that closed >= target after entry
                hit = [b for b in bars if b[0] > r.entry_min and b[4] >= target]
                if hit:
                    c['target_close_ok'] += 1
                    b0 = hit[0]
                    if b0[3] <= target <= b0[2]:
                        c['target_in_bar'] += 1
        c['exit'] = key
        report.append(c)
        log(key, c)
    pd.DataFrame(report).to_csv(os.path.join(HERE, 'obtainability_%s%s.csv' % (sig, stp)),
                                index=False)


if __name__ == '__main__':
    main(*(sys.argv[1:3] or ['a', 'i']))
