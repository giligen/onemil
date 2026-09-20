#!/usr/bin/env python3
"""frames8 / F25 — THE GEOMETRY TRANSPLANT: one vectorised multi-exit walker.

Prices ONE bracket entered at the open of bar `e` under EVERY declared geometry in a single pass
over that entry's forward bars.  The geometries are the SHIPPED exit specs of the two live books,
transplanted onto HOD-break's own population:

  X0  the shipped +2 R cap                 (must reproduce `hod_frames6/book6.csv::rr` to 1e-12)
  G1  ORB's static lock                    (arm +1.75 R -> stop +0.5 R, no target, flat 15:55)
  G2  BF's unified R-trail                 (arm +2 R, ride 1 R below the CLOSED-bar high)
  G2p G2 + the shipped 50 % @ +2 R partial (stop -> true breakeven)
  G3  the uncapped null                    (bare stop, no target, flat 15:55)
  G5/G6/G7  the target ladder UP           (+3 R / +4 R / +6 R resting limits)

Conventions shared with the whole programme and with `hod_frames6/common6.walk_from`:
entry at the OPEN of bar e; exits evaluated from bar e+1; a stop fills at
`min(stop, that bar's open) x (1 - SLIP)`; a resting target fills AT the level on a bar that CLOSES
through it; the 15:55 force close fills at that bar's open.  Read-only on every store.
"""
import numpy as np

SLIP = 0.001
EOD_M = 955                 # 15:55 ET — the HOD-break force close, kept for every transplant
ORB_ARM_R, ORB_LOCK_R = 1.75, 0.5
BF_ACT_R, BF_TRAIL_R = 2.0, 1.0
BF_PARTIAL_R, BF_PARTIAL_FRAC = 2.0, 0.5
ORB_SCALE_R, ORB_SCALE_FRAC = 3.0, 0.40
ATR_K = 0.25

GEOMS = ('X0', 'G1', 'G2', 'G2p', 'G3', 'G5', 'G6', 'G7')


# ------------------------------------------------------------------ the individual geometries
def _cap(oo, hh, ll, cc, mm, flat, E, R, stop, T):
    """A capped bracket: stop below, a resting limit at E + T x R.  Priority eod > stop > target."""
    ev = flat | (ll <= stop) | (cc >= E + T * R)
    if not ev.any():
        return (cc[-1] - E) / R, 'eod', int(mm[-1])
    j = int(np.argmax(ev))
    if flat[j]:
        return (oo[j] - E) / R, 'eod', int(mm[j])
    if ll[j] <= stop:
        px = min(stop, float(oo[j])) * (1.0 - SLIP)
        return (px - E) / R, 'stop', int(mm[j])
    return T, 'target', int(mm[j])


def _bare(oo, hh, ll, cc, mm, flat, E, R, stop):
    """The uncapped null: a bare stop ridden to the force close."""
    ev = flat | (ll <= stop)
    if not ev.any():
        return (cc[-1] - E) / R, 'eod', int(mm[-1])
    j = int(np.argmax(ev))
    if flat[j]:
        return (oo[j] - E) / R, 'eod', int(mm[j])
    px = min(stop, float(oo[j])) * (1.0 - SLIP)
    return (px - E) / R, 'stop', int(mm[j])


def _lock(oo, hh, ll, cc, mm, flat, E, R, stop, arm_r=ORB_ARM_R, lock_r=ORB_LOCK_R):
    """ORB's static lock.  Per bar: force close, then the CURRENT stop, then the arm test."""
    arm, lst = E + arm_r * R, E + lock_r * R
    ev1 = flat | (ll <= stop)
    am = hh >= arm
    k1 = int(np.argmax(ev1)) if ev1.any() else -1
    ka = int(np.argmax(am)) if am.any() else -1
    if ka < 0 or (k1 >= 0 and k1 <= ka):
        if k1 < 0:
            return (cc[-1] - E) / R, 'eod', int(mm[-1])
        if flat[k1]:
            return (oo[k1] - E) / R, 'eod', int(mm[k1])
        px = min(stop, float(oo[k1])) * (1.0 - SLIP)
        return (px - E) / R, 'stop', int(mm[k1])
    ev2 = flat | (ll <= lst)
    ev2 = ev2.copy()
    ev2[:ka + 1] = False
    if not ev2.any():
        return (cc[-1] - E) / R, 'eod', int(mm[-1])
    k2 = int(np.argmax(ev2))
    if flat[k2]:
        return (oo[k2] - E) / R, 'eod', int(mm[k2])
    px = min(lst, float(oo[k2])) * (1.0 - SLIP)
    return (px - E) / R, 'lock', int(mm[k2])


def _trail_path(hh, E, stop, base, unit, act_r, trail_r):
    """(stop_prev, armed_prev) — the stop live at each bar under the CLOSED-bar ratchet.

    `trading/bf_trail.arm_and_ratchet`'s contract: the stop a bar produces is live only from the
    NEXT bar (check-then-ratchet), so bar j is checked against the high of bars e+1..j-1.
    """
    n = len(hh)
    cm = np.maximum.accumulate(hh)
    hi_prev = np.empty(n)
    hi_prev[0] = E
    if n > 1:
        hi_prev[1:] = np.maximum(E, cm[:-1])
    armed = (hi_prev - base) / unit >= act_r
    sp = np.where(armed, np.maximum(stop, hi_prev - unit * trail_r), stop)
    return sp, armed


def _trail(oo, hh, ll, cc, mm, flat, E, R, stop, base=None, unit=None,
           act_r=BF_ACT_R, trail_r=BF_TRAIL_R, partial=False):
    """BF's unified R-trail, optionally with the shipped 50 % @ +2 R partial."""
    base = E if base is None else base
    unit = R if unit is None else unit
    sp, armed = _trail_path(hh, E, stop, base, unit, act_r, trail_r)
    ev = flat | (ll <= sp)
    if ev.any():
        j = int(np.argmax(ev))
        if flat[j]:
            rr_x, why, xm = (oo[j] - E) / R, 'eod', int(mm[j])
        else:
            px = min(float(sp[j]), float(oo[j])) * (1.0 - SLIP)
            rr_x, why, xm = (px - E) / R, ('trail_stop' if armed[j] else 'stop'), int(mm[j])
    else:
        j = len(oo) - 1
        rr_x, why, xm = (cc[-1] - E) / R, 'eod', int(mm[-1])
    if not partial:
        return rr_x, why, xm
    pm = hh >= base + BF_PARTIAL_R * unit
    if not pm.any():
        return rr_x, why, xm
    jp = int(np.argmax(pm))
    if jp >= j:                       # the stop check precedes the partial on the same bar
        return rr_x, why, xm
    prr = BF_PARTIAL_FRAC * (float(cc[jp]) - E) / R
    return prr + (1.0 - BF_PARTIAL_FRAC) * rr_x, 'pp+' + why, xm


def geoms(o, h, l, c, m, e, stop, want=GEOMS):
    """{geom_id: (rr, why, exit_m)} for ONE entry.  None when the bracket is not priceable."""
    n = len(o)
    if e < 0 or e + 1 >= n:
        return None
    E = float(o[e])
    R = E - float(stop)
    if not (R > 0) or not (E > 0):
        return None
    s0 = e + 1
    oo, hh, ll, cc, mm = o[s0:], h[s0:], l[s0:], c[s0:], m[s0:]
    flat = mm >= EOD_M
    out = {}
    for g in want:
        if g == 'X0':
            out[g] = _cap(oo, hh, ll, cc, mm, flat, E, R, stop, 2.0)
        elif g == 'G5':
            out[g] = _cap(oo, hh, ll, cc, mm, flat, E, R, stop, 3.0)
        elif g == 'G6':
            out[g] = _cap(oo, hh, ll, cc, mm, flat, E, R, stop, 4.0)
        elif g == 'G7':
            out[g] = _cap(oo, hh, ll, cc, mm, flat, E, R, stop, 6.0)
        elif g == 'G3':
            out[g] = _bare(oo, hh, ll, cc, mm, flat, E, R, stop)
        elif g == 'G1':
            out[g] = _lock(oo, hh, ll, cc, mm, flat, E, R, stop)
        elif g == 'G2':
            out[g] = _trail(oo, hh, ll, cc, mm, flat, E, R, stop)
        elif g == 'G2p':
            out[g] = _trail(oo, hh, ll, cc, mm, flat, E, R, stop, partial=True)
        else:
            raise KeyError(g)
    return out


# ------------------------------------------------------------------ the booked-only sub-arms
def sub_arms(o, h, l, c, v, m, e, stop, level=None, atr14=None):
    """{sub_id: (rr, why, exit_m)} — the four sub-arms that need a field a control does not have.

    G1a  ORB's ATR stop floor (SZ1, k = 0.25) applied to the protective stop only.
    G1b  ORB's 40 % @ +3 R scale-out on top of the static lock (frozen composition).
    G2v  BF's prev-bar volume guard on the TRAIL stop (never the hard stop).
    G2pl BF's plan-R basis: baseline = the break level, unit = level - stop.
    """
    n = len(o)
    if e < 0 or e + 1 >= n:
        return {}
    E = float(o[e])
    R = E - float(stop)
    if not (R > 0) or not (E > 0):
        return {}
    s0 = e + 1
    oo, hh, ll, cc, vv, mm = o[s0:], h[s0:], l[s0:], c[s0:], v[s0:], m[s0:]
    flat = mm >= EOD_M
    out = {}

    # --- G1a: floored stop (max(stop, E - k x ATR)), degenerate clamp as the shipped helper ---
    if atr14 is not None and np.isfinite(atr14):
        fl = max(float(stop), E - ATR_K * float(atr14))
        if fl > E * (1.0 - 1e-4):
            fl = float(stop)
        R2 = E - fl
        if R2 > 0:
            rr, why, xm = _lock(oo, hh, ll, cc, mm, flat, E, R2, fl)
            out['G1a'] = (rr * R2 / R, why, xm)      # re-expressed in the trade's OWN R unit
    # --- G1b: static lock + 40 % @ +3 R resting scale-out (frozen same-bar rule) ---
    out['G1b'] = _lock_scale(oo, hh, ll, cc, mm, flat, E, R, stop)
    # --- G2v: the R-trail with the prev-bar volume guard on the trail stop ---
    vbase = float(np.mean(v[max(0, e - 5):e])) if e >= 1 else 0.0
    out['G2v'] = _trail_volguard(oo, hh, ll, cc, vv, mm, flat, E, R, stop, vbase,
                                 float(v[e]) if e < n else 0.0)
    # --- G2pl: the R-trail on plan-R (baseline = level, unit = level - stop) ---
    if level is not None and np.isfinite(level) and float(level) - float(stop) > 0:
        out['G2pl'] = _trail(oo, hh, ll, cc, mm, flat, E, R, stop,
                             base=float(level), unit=float(level) - float(stop))
    return out


def _lock_scale(oo, hh, ll, cc, mm, flat, E, R, stop,
                arm_r=ORB_ARM_R, lock_r=ORB_LOCK_R, frac=ORB_SCALE_FRAC, level_r=ORB_SCALE_R):
    """ORB's static lock + the 40 % @ +level_r R scale-out.  Frozen composition rules:
    a same-bar stop+scale FILLS the scale; the runner keeps the SAME initial stop and re-derives
    the armed state including the scale bar."""
    arm, lst, spx = E + arm_r * R, E + lock_r * R, E + level_r * R
    stop_c, armed = stop, False
    n = len(oo)
    for i in range(n):
        if flat[i]:
            return (float(oo[i]) - E) / R, 'eod', int(mm[i])
        if ll[i] <= stop_c and hh[i] < spx:
            px = min(stop_c, float(oo[i])) * (1.0 - SLIP)
            return (px - E) / R, ('lock' if armed else 'stop'), int(mm[i])
        if hh[i] >= spx:
            leg = frac * level_r                                     # the resting limit, filled AT it
            stop2 = stop
            armed2 = bool(hh[:i + 1].max() >= arm)
            if armed2:
                stop2 = max(stop2, lst)
            for j in range(i + 1, n):
                if flat[j]:
                    return leg + (1 - frac) * (float(oo[j]) - E) / R, 'sc+eod', int(mm[j])
                if ll[j] <= stop2:
                    px = min(stop2, float(oo[j])) * (1.0 - SLIP)
                    return (leg + (1 - frac) * (px - E) / R,
                            'sc+lock' if armed2 else 'sc+stop', int(mm[j]))
                if not armed2 and hh[j] >= arm:
                    armed2 = True
                    stop2 = max(stop2, lst)
            return leg + (1 - frac) * (float(cc[-1]) - E) / R, 'sc+eod', int(mm[-1])
        if not armed and hh[i] >= arm:
            armed = True
            stop_c = max(stop_c, lst)
    return (float(cc[-1]) - E) / R, 'eod', int(mm[-1])


def _trail_volguard(oo, hh, ll, cc, vv, mm, flat, E, R, stop, vbase, v_entry,
                    act_r=BF_ACT_R, trail_r=BF_TRAIL_R, min_ratio=1.0):
    """BF's R-trail with `trading/trail_vol_guard`'s rule: a TRAIL stop fires only when the
    PREVIOUS closed bar's volume >= min_ratio x the pre-entry baseline.  The hard stop always fires."""
    hi, stop_c, active = E, stop, False
    prev_v = v_entry
    for i in range(len(oo)):
        if flat[i]:
            return (float(oo[i]) - E) / R, 'eod', int(mm[i])
        if ll[i] <= stop_c:
            if (not active) or (vbase <= 0) or (prev_v >= min_ratio * vbase):
                px = min(stop_c, float(oo[i])) * (1.0 - SLIP)
                return (px - E) / R, ('trail_stop' if active else 'stop'), int(mm[i])
        hi = max(hi, float(hh[i]))
        if not active and (hi - E) / R >= act_r:
            active = True
        if active:
            stop_c = max(stop_c, hi - R * trail_r)
        prev_v = float(vv[i])
    return (float(cc[-1]) - E) / R, 'eod', int(mm[-1])


# ------------------------------------------------------------------ the cost model (PREREG §4)
RATIO = {'target': 0.0, 'stop': 0.875, 'lock': 0.875, 'trail_stop': 0.875, 'eod': 0.412}


def exit_ratio(why):
    """The exit-leg spread multiplier for a (possibly blended) exit reason."""
    if why.startswith('pp+'):            # 50 % marketable partial + 50 % on the named leg
        return 0.5 * 0.875 + 0.5 * RATIO.get(why[3:], 0.875)
    if why.startswith('sc+'):            # 40 % resting scale-out + 60 % on the named leg
        return 0.40 * 0.0 + 0.60 * RATIO.get(why[3:], 0.875)
    return RATIO.get(why, 0.875)
