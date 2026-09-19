#!/usr/bin/env python3
"""hod_bleed — the shared exit engine.  ONE implementation of every declared exit rule, used by
Part 1's descriptive matrix and by Part 2's cells.  Pure table work on walk3.py's artifacts.

Conventions (PREREG §2), identical on every path:
  * bar k=0 is the FILL bar (we bought its open); exits are checked from k=1, exactly as the
    shipped spec does.  The fill bar's HIGH seeds the running max (the walk2 convention).
  * arming is on CLOSED bars and takes effect on the FOLLOWING bar: the low of bar k is tested
    against the stop that was in force at the END of bar k-1.
  * a stop-type exit fills at min(stop, open) * (1 - 0.001).  A signal-type exit (peak retrace,
    volume / VWAP / MACD) fills at the NEXT bar's OPEN.  The +2R target fills at the level.
  * inside one bar the order is: 15:55 flat, stop, target -- the shipped order.
  * a signal exit and a stop exit landing on the same bar: the signal wins (it is a market order
    at that bar's open; the stop is a resting order the market has yet to reach).
"""
import glob, os
import numpy as np, pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
SLIP, EOD_M, BIG = 0.001, 955, 10 ** 9
RD = dict(dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])


def _first(mask, lo=0):
    """index of the first True at or after `lo`, or BIG."""
    if lo >= len(mask):
        return BIG
    w = np.argmax(mask[lo:])
    return (lo + int(w)) if mask[lo + w] else BIG


class Paths:
    """All walked paths in flat numpy arrays, addressed by signal id."""

    def __init__(self):
        sg = pd.read_csv(f'{D}/sigs.csv', **RD).drop_duplicates('sid')
        parts = [pd.read_parquet(f) for f in sorted(glob.glob(f'{D}/bars/*.parquet'))]
        b = pd.concat(parts, ignore_index=True).sort_values(['sid', 'k'], kind='mergesort')
        del parts
        self.sig = sg.set_index('sid')
        self.sids = b.sid.values
        n = self.sig.index.max() + 1
        starts = np.full(n, -1, np.int64); ends = np.full(n, -1, np.int64)
        chg = np.flatnonzero(np.r_[True, self.sids[1:] != self.sids[:-1]])
        for a, s in zip(chg, self.sids[chg]):
            starts[s] = a
        for a in chg[1:]:
            ends[self.sids[a - 1]] = a
        ends[self.sids[-1]] = len(self.sids)
        self.st, self.en = starts, ends
        self.m = b.m.values.astype(np.int32)
        self.op = b.op_r.values.astype(np.float64)
        self.hi = b.hi_r.values.astype(np.float64)
        self.lo = b.lo_r.values.astype(np.float64)
        self.cl = b.cl_r.values.astype(np.float64)
        self.tv = b.t_vol.values.astype(bool); self.td = b.t_down.values.astype(bool)
        self.tf = b.t_fade.values.astype(bool); self.tw = b.t_vwap.values.astype(bool)
        self.th = b.t_hist.values.astype(bool); self.tx = b.t_cross.values.astype(bool)
        self.hs = b.hit_stop.values.astype(bool); self.ht = b.hit_tgt.values.astype(bool)
        del b
        self.eord = self.sig.e_over_rd.to_dict()

    def has(self, sid):
        return 0 <= sid < len(self.st) and self.st[sid] >= 0

    def get(self, sid):
        a, z = self.st[sid], self.en[sid]
        return dict(m=self.m[a:z], op=self.op[a:z], hi=self.hi[a:z], lo=self.lo[a:z],
                    cl=self.cl[a:z], tv=self.tv[a:z], td=self.td[a:z], tf=self.tf[a:z],
                    tw=self.tw[a:z], th=self.th[a:z], tx=self.tx[a:z],
                    hs=self.hs[a:z], ht=self.ht[a:z], eord=self.eord[sid])


# ------------------------------------------------------------------ the exit engine
def prep(P):
    """Per-signal precomputation shared by every cell."""
    nk = len(P['cl'])
    P['nk'] = nk
    P['runmax'] = np.maximum.accumulate(P['hi'])
    P['k_eod'] = _first(P['m'] >= EOD_M, 1)
    P['k_tgt'] = _first(P['ht'], 1)          # exact float64 flags from the walk -> byte parity
    P['k_hard'] = _first(P['hs'], 1)
    fade3 = np.zeros(nk, bool)
    if nk >= 3:
        f = P['tf']
        fade3[2:] = f[2:] & f[1:-1] & f[:-2]
    P['ev'] = {
        'vol': P['tv'] & P['td'],           # E4a distribution selling
        'fade': fade3,                      # E4b momentum gone
        'vwap': P['tw'],                    # E4c lost session VWAP
        'hist': P['th'],                    # E4d MACD histogram negative
        'cross': P['tx'],                   # E4e MACD signal cross
    }
    two = (np.maximum.accumulate(P['ev']['vol']).astype(int)
           + np.maximum.accumulate(P['ev']['vwap']).astype(int)
           + np.maximum.accumulate(P['ev']['hist']).astype(int))
    P['ev']['two3'] = two >= 2             # E4f any two of three agree
    return P


def arm_k(P, a):
    """the first CLOSED bar whose running max reaches E + a*R (BIG if never)."""
    return _first(P['runmax'] >= a)


def _stop_fill(P, k, lvl):
    x = min(lvl, P['op'][k])
    return x - SLIP * (P['eord'] + x)


def _resolve(P, k_stop, lvl_stop, k_sig):
    """earliest of {flat, stop, target, signal}; returns (k, rr, why, cost-ratio).

    The exit-side cost ratio is the `score4` contract {stop 0.875, eod 0.412, target 0.0}; every
    exit this pass introduces is charged the marketable 0.875 (PREREG §1).  Signal wins ties.
    """
    k_eod, k_tgt = P['k_eod'], P['k_tgt']
    best = min(k_stop, k_tgt, k_eod, k_sig)
    if best >= BIG:
        k = P['nk'] - 1
        return k, float(P['cl'][k]), 'eod', 0.412
    if k_sig == best:
        return best, float(P['op'][best]), 'trigexit', 0.875
    if k_eod == best:
        return best, float(P['op'][best]), 'eod', 0.412
    if k_stop == best:
        w = 'ratchet' if lvl_stop > -1.0 else 'stop'
        return best, float(_stop_fill(P, best, lvl_stop)), w, 0.875
    return best, 2.0, 'target', 0.0


def sim_base(P):
    return _resolve(P, P['k_hard'], -1.0, BIG)


def sim_ratchet(P, a, s, n_max=None, k_sig=BIG):
    """E1/E2/E5: arm at a (optionally only if a is reached within the first n_max minutes of the
    trade, i.e. bar index <= n_max-1); the stop moves ONCE to E + s*R and never loosens."""
    ka = arm_k(P, a)
    if ka >= BIG or (n_max is not None and ka > n_max - 1):
        return _resolve(P, P['k_hard'], -1.0, k_sig)
    if P['k_hard'] <= ka:                        # the original stop is still in force through ka
        return _resolve(P, P['k_hard'], -1.0, k_sig)
    k2 = _first(P['lo'] <= s, max(ka + 1, 1))    # the ratcheted stop, effective from ka+1
    return _resolve(P, k2, s, k_sig)


def sim_peak(P, a, d):
    """E3: armed at a, exit at the NEXT bar's open when a closed bar's close <= runmax - d."""
    ka = arm_k(P, a)
    if ka >= BIG:
        return _resolve(P, P['k_hard'], -1.0, BIG)
    j = _first(P['cl'] <= P['runmax'] - d, ka)
    k_sig = (j + 1) if (j < BIG and j + 1 < P['nk']) else BIG
    return _resolve(P, P['k_hard'], -1.0, k_sig)


def trig_k(P, name, a):
    """E4: the exit bar of a signal trigger on a trade armed at a (BIG if it never fires)."""
    ka = arm_k(P, a)
    if ka >= BIG:
        return BIG
    j = _first(P['ev'][name], ka)
    return (j + 1) if (j < BIG and j + 1 < P['nk']) else BIG


def sim_trigger(P, name, a):
    return _resolve(P, P['k_hard'], -1.0, trig_k(P, name, a))


def sim_combo(P, a, s, name):
    """E5: the ratchet and the signal trigger together; whichever fires first."""
    return sim_ratchet(P, a, s, k_sig=trig_k(P, name, a))


def sim_partial(P, p, frac=0.5):
    """E6: sell `frac` at E + p*R, at the NEXT bar's open after the bar whose HIGH first reaches
    the level.  THE STOP DOES NOT MOVE -- the runner keeps the original stop, the +2R target and
    the 15:55 flat.  The trade's exit minute is the RUNNER's (the slot is held to the end)."""
    kb, rrb, whyb, ratb = sim_base(P)
    j = _first(P['runmax'] >= p)
    if j >= BIG or j >= kb or j + 1 >= P['nk']:
        return kb, rrb, whyb, ratb
    kp = j + 1
    rr = frac * float(P['op'][kp]) + (1 - frac) * rrb
    return kb, rr, f'pp+{whyb}', frac * 0.875 + (1 - frac) * ratb
