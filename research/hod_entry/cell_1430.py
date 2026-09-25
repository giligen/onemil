"""Cell 1,430 — exits on the winning fills (research/hod_entry/PREREG_WEEKEND.md, frozen
2026-09-25 18:35 UTC). Six exit variants (a)-(f), each re-walked on the cached 1-minute paths and
paired against B0's OWN exit (same entry, same stop, same fill) on the exact E1 fills of cell
1,427 (research/hod_entry/sip_rebuild_val.csv, TRAIN-H2 = split TRAIN half H2, VAL = split VAL).

Base-fill columns used (from sip_rebuild_val.csv, status == 'fill' only — "the winning fills" =
the fills, as opposed to no-fills; PREREG does not gate on the trade's own sign):
  fill (= entry price), R (= entry - B0 stop), entry_m, exit_m/exit_price/why/net_R/cost_R (B0's
  own exit, reused verbatim as the PAIRED baseline and, per PREREG's "No constant of E1 changes",
  as the entry+exit cost base for every variant).

Cost model (PREREG_WEEKEND.md lines 6-9): the E1 cost_R (measured entry half-spread + B0's own
exit half-spread, unchanged) is reused for every variant's own exit price/time -- new exit prices
are NOT re-quoted (no per-minute spread data outside the entry window exists in this cell's
inputs). On top of that unchanged constant, a NEW 30bps stop-slip charge is added when the
VARIANT's own exit reason is a stop ('stop', 'stop_bb', 'stop_runner'), reported both with and
without per PREREG.

Two variants need an explicit documented modeling choice where the PREREG prose is silent:
  (d) 50% scale-out at +2R, remainder "to the B0 rules": since +2R IS B0's target level, applying
      B0's target check to the remainder would close it on the SAME bar (degenerate — identical to
      no partial). The remainder therefore rides on the ORIGINAL STOP ONLY (target dropped once
      already touched) to EOD -- a documented choice, not a PREREG ambiguity resolved by fiat.
  (e) "after +0.5R was reached": the VWAP-close exit only arms once a bar's high >= entry + 0.5R;
      before arming, only B0's own stop/target can end the trade. Exit fires at the NEXT bar's open
      after the first close < VWAP, mirroring the house convention in
      research/hod_exit_lab/score_cells.py (x9_vwap / x_time_stop).
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

EOD_M = 955                  # 15:55 ET, per sip_rebuild.py / walker.b0_fill
CLOSE_1430_M = 870           # 14:30 ET, variant (f)
TARGET_R = 2.0                # B0's target multiple, per sip_rebuild.py TARGET_R
STOP_SLIP_BP = 0.0030         # new 30 bps stop-slip charge, PREREG_WEEKEND.md line 8


# --------------------------------------------------------------------------------------------
# Fill-price mechanics — shared primitive, identical priority to walker.b0_fill /
# score_cells.b0_style_fill: gap-through at the open, then low<=stop (both-touch counts as a stop,
# conservative), then high>=target, else the eod_m bar's open.
# --------------------------------------------------------------------------------------------

def _gap_or_touch(o, l, stop):
    """A bar whose open already gapped through `stop` fills at that open; otherwise at `stop`."""
    return o if o <= stop else stop


def b0_walk(entry, stop, target, bars, eod_m=EOD_M):
    """B0's own exit rule, walking `bars` (path rows strictly after entry_m, sorted by m, m<=955)."""
    for row in bars:
        if row.m >= eod_m:
            return int(row.m), float(row.o), 'eod'
        if row.l <= stop:
            return int(row.m), float(_gap_or_touch(row.o, row.l, stop)), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
    return None


# --------------------------------------------------------------------------------------------
# (a) time stop at 90 min after entry, exit at that bar's open
# --------------------------------------------------------------------------------------------

def variant_a_time_stop(entry, stop, target, bars, entry_m, delay_min=90, eod_m=EOD_M):
    """B0's stop/target still apply; if neither triggers by entry_m+delay_min, exit at that bar's
    open ('time_stop'), or at the session EOD if the delay would run past it ('eod')."""
    cutoff = min(entry_m + delay_min, eod_m)
    for row in bars:
        if row.m >= cutoff:
            why = 'eod' if cutoff >= eod_m else 'time_stop'
            return int(row.m), float(row.o), why
        if row.l <= stop:
            return int(row.m), float(_gap_or_touch(row.o, row.l, stop)), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
    return None


# --------------------------------------------------------------------------------------------
# (b)/(c) trigger-and-lock: stop ratchets to entry + lock_mult*R once a bar's high touches
# entry + trigger_mult*R. Lock arms and takes effect starting the NEXT bar (checked before this
# bar's own update, matching score_cells.x5_lock).
# --------------------------------------------------------------------------------------------

def _lock_walk(entry, stop0, target, R, bars, trigger_mult, lock_mult, eod_m=EOD_M):
    cur_stop = stop0
    for row in bars:
        if row.m >= eod_m:
            return int(row.m), float(row.o), 'eod'
        if row.l <= cur_stop:
            return int(row.m), float(_gap_or_touch(row.o, row.l, cur_stop)), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
        if row.h >= entry + trigger_mult * R:
            cur_stop = max(cur_stop, entry + lock_mult * R)
    return None


def variant_b_breakeven(entry, stop0, target, R, bars):
    """(b) breakeven lock: stop -> entry once a bar's high >= entry + 1R."""
    return _lock_walk(entry, stop0, target, R, bars, trigger_mult=1.0, lock_mult=0.0)


def variant_c_orb_lock(entry, stop0, target, R, bars):
    """(c) ORB-style lock: high >= entry + 1.5R -> stop -> entry + 0.5R."""
    return _lock_walk(entry, stop0, target, R, bars, trigger_mult=1.5, lock_mult=0.5)


# --------------------------------------------------------------------------------------------
# (d) 50% scale-out at +2R, remainder rides the ORIGINAL stop only (documented choice, see module
# docstring) to EOD. Returns (exit_m, blended_price, why, half1_price_or_None).
# --------------------------------------------------------------------------------------------

def variant_d_scale_out(entry, stop0, R, bars, eod_m=EOD_M):
    target = entry + TARGET_R * R
    half1_filled, half1_price = False, None
    for row in bars:
        if row.m >= eod_m:
            if half1_filled:
                return int(row.m), float(0.5 * half1_price + 0.5 * row.o), 'eod_partial', half1_price
            return int(row.m), float(row.o), 'eod', None
        if row.l <= stop0:
            px = _gap_or_touch(row.o, row.l, stop0)
            if half1_filled:
                return int(row.m), float(0.5 * half1_price + 0.5 * px), 'stop_runner', half1_price
            return int(row.m), float(px), 'stop', None
        if not half1_filled and row.h >= target:
            half1_filled, half1_price = True, target
    return None


# --------------------------------------------------------------------------------------------
# (e) exit when a bar closes below session VWAP, but only once +0.5R has been reached. Fires at
# the bar AFTER the trigger bar's own close, at that next bar's open (house convention).
# --------------------------------------------------------------------------------------------

def variant_e_vwap(entry, stop0, target, R, bars, arm_mult=0.5, eod_m=EOD_M):
    armed, trig_m = False, None
    for row in bars:
        if row.m >= eod_m:
            return int(row.m), float(row.o), 'eod'
        if row.l <= stop0:
            return int(row.m), float(_gap_or_touch(row.o, row.l, stop0)), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
        if trig_m is not None and row.m > trig_m:
            return int(row.m), float(row.o), 'vwap_exit'
        if not armed and row.h >= entry + arm_mult * R:
            armed = True
        if armed and trig_m is None and row.c < row.vwap:
            trig_m = row.m
    return None


# --------------------------------------------------------------------------------------------
# (f) close at 14:30 instead of 15:55 -- identical B0 rule, earlier EOD cutoff.
# --------------------------------------------------------------------------------------------

def variant_f_early_close(entry, stop, target, bars):
    return b0_walk(entry, stop, target, bars, eod_m=CLOSE_1430_M)


VARIANTS = ('a', 'b', 'c', 'd', 'e', 'f')


# --------------------------------------------------------------------------------------------
# Driver: load base fills + cached paths, run every variant on every fill, score paired.
# --------------------------------------------------------------------------------------------

def load_base_fills():
    """E1 fills of cell 1,427 (research/hod_entry/sip_rebuild_val.csv), status == 'fill' only,
    split into TRAIN-H2 (split TRAIN, half H2) and VAL (split VAL)."""
    p = os.path.join(HERE, 'sip_rebuild_val.csv')
    d = pd.read_csv(p, dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
    d = d[d.status == 'fill'].copy()
    d['stop0'] = d.fill - d.R
    d['target0'] = d.fill + TARGET_R * d.R
    train_h2 = d[(d.split == 'TRAIN') & (d.half == 'H2')].reset_index(drop=True)
    val = d[d.split == 'VAL'].reset_index(drop=True)
    return train_h2, val


def load_paths_index():
    """research/hod_exit_lab/paths.parquet, indexed by (day, symbol), sorted by m."""
    p = os.path.join(ROOT, 'research/hod_exit_lab/paths.parquet')
    paths = pd.read_parquet(p)
    return paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()


def _bars_after(idx, day, symbol, entry_m):
    key = (day, symbol)
    if key not in idx.index:
        return None
    g = idx.loc[[key]]
    g = g[(g.m > entry_m) & (g.m <= EOD_M)]
    return None if g.empty else g


def run_variant_on_row(letter, row, bars_df):
    """Dispatch one fill row to variant `letter`; returns (exit_m, exit_px, why) or None."""
    bars = bars_df.itertuples()
    if letter == 'a':
        return variant_a_time_stop(row.fill, row.stop0, row.target0, bars, row.entry_m)
    if letter == 'b':
        return variant_b_breakeven(row.fill, row.stop0, row.target0, row.R, bars)
    if letter == 'c':
        return variant_c_orb_lock(row.fill, row.stop0, row.target0, row.R, bars)
    if letter == 'd':
        return variant_d_scale_out(row.fill, row.stop0, row.R, bars)
    if letter == 'e':
        return variant_e_vwap(row.fill, row.stop0, row.target0, row.R, bars)
    if letter == 'f':
        return variant_f_early_close(row.fill, row.stop0, row.target0, bars)
    raise ValueError(f'unknown variant {letter!r}')


STOP_WHY = {'stop', 'stop_bb', 'stop_runner'}


def score_variant(letter, fills, idx):
    """Runs `letter` on every fill in `fills` (DataFrame with fill/R/stop0/target0/entry_m/day/
    symbol/wk/net_R/cost_R). Returns a per-fill DataFrame with net_R_slip, net_R_noslip, dR_slip,
    dR_noslip, why."""
    rows = []
    n_no_path = 0
    for r in fills.itertuples():
        bars_df = _bars_after(idx, r.day, r.symbol, r.entry_m)
        if bars_df is None:
            n_no_path += 1
            continue
        res = run_variant_on_row(letter, r, bars_df)
        if res is None:
            n_no_path += 1
            continue
        exit_m, exit_px, why = res[0], res[1], res[2]
        raw_R = (exit_px - r.fill) / r.R
        slip_R = STOP_SLIP_BP * exit_px / r.R if why in STOP_WHY else 0.0
        net_noslip = raw_R - r.cost_R
        net_slip = raw_R - r.cost_R - slip_R
        rows.append(dict(day=r.day, symbol=r.symbol, wk=r.wk, why=why,
                          net_R_noslip=net_noslip, net_R_slip=net_slip,
                          dR_noslip=net_noslip - r.net_R, dR_slip=net_slip - r.net_R,
                          b0_net_R=r.net_R))
    if n_no_path:
        import sys
        print(f'[cell_1430] WARNING: variant {letter} dropped {n_no_path}/{len(fills)} fills '
              f'(no cached path after entry_m)', file=sys.stderr)
    return pd.DataFrame(rows)


def day_clustered_t(y, day):
    """statsmodels OLS of y on an intercept, cluster-robust SE on `day`. Returns (t, n_days)."""
    y = np.asarray(y, dtype=float)
    n_days = pd.Series(day).nunique()
    if len(y) < 2 or n_days < 2:
        return float('nan'), n_days
    X = np.ones((len(y), 1))
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': np.asarray(day)})
    return float(model.tvalues[0]), n_days


def ex_top5_pct(x):
    """Mean of x after dropping the top 5% of values (ex-tail mean)."""
    x = pd.Series(x).sort_values()
    if len(x) < 20:
        cut = max(1, int(round(0.05 * len(x))))
    else:
        cut = int(round(0.05 * len(x)))
    return float(x.iloc[:len(x) - cut].mean()) if cut < len(x) else float(x.mean())


def worst_week(net_R, wk):
    """Weekly sum of net_R (R units), worst (minimum) week."""
    w = pd.Series(net_R.values, index=wk.values).groupby(level=0).sum()
    return float(w.min()) if len(w) else float('nan')


def score_split(scored, wk, base_net_R):
    """One split's summary row for a variant's scored DataFrame: mean dR, day-clustered t on dR,
    ex-top-5% of dR, worst week (slip-charged net_R) vs B0's own worst week."""
    out = {}
    out['n'] = len(scored)
    out['dR_mean_slip'] = float(scored.dR_slip.mean()) if len(scored) else float('nan')
    out['dR_mean_noslip'] = float(scored.dR_noslip.mean()) if len(scored) else float('nan')
    t, ndays = day_clustered_t(scored.dR_slip, scored.day) if len(scored) else (float('nan'), 0)
    out['t_slip'] = t
    out['t_ndays'] = ndays
    out['ex_top5_dR_slip'] = ex_top5_pct(scored.dR_slip) if len(scored) else float('nan')
    out['worst_week_variant'] = worst_week(scored.net_R_slip, scored.wk) if len(scored) else float('nan')
    out['worst_week_b0'] = worst_week(scored.b0_net_R, scored.wk) if len(scored) else float('nan')
    return out


if __name__ == '__main__':
    train_h2, val = load_base_fills()
    idx = load_paths_index()
    print(f'[cell_1430] base fills: TRAIN-H2 n={len(train_h2)} VAL n={len(val)}')
    for letter in VARIANTS:
        s_tr = score_variant(letter, train_h2, idx)
        s_va = score_variant(letter, val, idx)
        r_tr = score_split(s_tr, train_h2.wk, train_h2.net_R)
        r_va = score_split(s_va, val.wk, val.net_R)
        pass_ = (r_tr['dR_mean_slip'] >= 0.05 and r_va['dR_mean_slip'] >= 0.05
                 and (not pd.isna(r_va['t_slip'])) and r_va['t_slip'] >= 2
                 and r_va['worst_week_variant'] >= r_va['worst_week_b0'])
        print(f"({letter}) TRAIN-H2 n={r_tr['n']} dR_slip={r_tr['dR_mean_slip']:+.3f} "
              f"dR_noslip={r_tr['dR_mean_noslip']:+.3f} | "
              f"VAL n={r_va['n']} dR_slip={r_va['dR_mean_slip']:+.3f} "
              f"dR_noslip={r_va['dR_mean_noslip']:+.3f} t={r_va['t_slip']:.2f} "
              f"ex5={r_va['ex_top5_dR_slip']:+.3f} ww_var={r_va['worst_week_variant']:+.2f} "
              f"ww_b0={r_va['worst_week_b0']:+.2f} -> {'PASS' if pass_ else 'FAIL'}")
