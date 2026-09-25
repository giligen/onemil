"""Cell 1,429 — fill-quality sizing (research/hod_entry/PREREG_WEEKEND.md, frozen 2026-09-25
18:35 UTC). Sizes every E1 fill of cell 1,427 (research/hod_entry/sip_rebuild_val.csv, TRAIN-H2 +
VAL, reusing cell_1430.load_base_fills) by how close the ask was to the trigger level the instant
BEFORE the trigger print.

d = ask_at / level - 1, in bps, where `ask_at` is sip_rebuild.simulate_entry's own prevailing-quote
ask at the triggering-print timestamp (last valid NBBO with ts <= t_hit) — exactly "the ask the
instant before the trigger print" the PREREG asks for; no re-fetch of the quote cache is needed,
`ask_at` already IS that value for every fill row. Risk multiplier: 1.5x if d <= 5 bps, else 1x
(fills only exist for d <= 15 bps, sip_rebuild's own entry-limit gate — asserted below).

Score: book R per unit of risk = sum(R * mult) / sum(mult), against the flat (mult=1 everywhere)
book, on the SAME net_R the base file carries (R = net_R, already cost-charged). Reported with and
without the new 30bps stop-slip charge (PREREG_WEEKEND.md common section, lines 6-9): the charge is
added to `why in cell_1430.STOP_WHY` fills exactly as cell_1430 does, using the base file's own
`exit_price`/`R`/`why` columns (no re-walk needed — 1,429 does not touch the exit rule, only the
entry-quality weight).

Documented modeling choice (PREREG prose is silent on this): the per-WEEK "book R per unit of
risk" used for the worst-week comparison applies the SAME sum(R*mult)/sum(mult) ratio within each
week (not a raw R sum) — this is the natural per-week restriction of the cell's own headline
metric, and keeps the flat-book comparison apples-to-apples regardless of how many 1.5x fills
landed in a given week.

Report-only: the trigger print's own size (round lot >= 100 sh vs odd lot < 100 sh) and its mean
net_R (no charge, i.e. the base file's own net_R). Trigger-print size is NOT in sip_rebuild_val.csv
and is read fresh from research/hod_entry/sip_cache/{day}.pkl.gz — the same tapes sip_rebuild.py
built — by reproducing simulate_entry's own trigger-print search (S_ns via sip_rebuild.et_ns,
trigger = level + TICK, first trade in [S_ns-60s, S_ns) with price >= trigger) and reading that
print's `size` column. A day/symbol/break_m whose tape is missing from the cache is dropped from
the size-class report ONLY (never from the sizing score) and logged as a WARNING.
"""
import gzip
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cell_1430 as c1430          # noqa: E402 — reuse loader + shared stats primitives
import sip_rebuild as sr           # noqa: E402 — et_ns, sig_key, TICK

D_THRESH_BPS = 5.0       # d <= 5 bps -> 1.5x
D_MAX_BPS = 15.0         # sip_rebuild's own entry-limit gate (LIMIT_BPS=0.0015); sanity bound
MULT_TIGHT = 1.5
MULT_LOOSE = 1.0
ROUND_LOT = 100


# --------------------------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------------------------

def add_sizing(fills):
    """Adds d_bps, mult, net_R_noslip (= base net_R), net_R_slip (30bps stop-slip charged) to a
    copy of `fills`. Asserts (with a WARNING, not a hard failure) that every d_bps <= D_MAX_BPS,
    per sip_rebuild's own entry-limit gate."""
    f = fills.copy()
    f['d_bps'] = (f.ask_at / f.level - 1.0) * 1e4
    n_over = int((f.d_bps > D_MAX_BPS + 1e-6).sum())
    if n_over:
        print(f'[cell_1429] WARNING: {n_over}/{len(f)} fills have d_bps > {D_MAX_BPS} '
              f'(sip_rebuild entry-limit gate should make this impossible) — check ask_at/level', file=sys.stderr)
    f['mult'] = np.where(f.d_bps <= D_THRESH_BPS + 1e-6, MULT_TIGHT, MULT_LOOSE)   # tolerance for fp noise at the 5bps boundary
    slip_R = np.where(f.why.isin(c1430.STOP_WHY), c1430.STOP_SLIP_BP * f.exit_price / f.R, 0.0)
    f['net_R_noslip'] = f.net_R
    f['net_R_slip'] = f.net_R - slip_R
    return f


def book_r_per_risk(net_R, mult):
    """sum(net_R * mult) / sum(mult); the cell's headline metric. NaN on an empty/zero-weight book."""
    net_R = np.asarray(net_R, dtype=float)
    mult = np.asarray(mult, dtype=float)
    w = mult.sum()
    return float((net_R * mult).sum() / w) if w > 0 else float('nan')


def worst_week_ratio(net_R, mult, wk):
    """Per-week book_r_per_risk, worst (minimum) week. Empty -> NaN."""
    d = pd.DataFrame({'net_R': np.asarray(net_R, dtype=float), 'mult': np.asarray(mult, dtype=float),
                       'wk': np.asarray(wk)})
    if not len(d):
        return float('nan')
    weekly = d.groupby('wk').apply(lambda g: book_r_per_risk(g.net_R, g.mult), include_groups=False)
    return float(weekly.min()) if len(weekly) else float('nan')


def score_split(f, which):
    """One split's summary row: which in {'noslip', 'slip'} selects the net_R column. Returns dict
    with weighted/flat book R-per-risk, dR, day-clustered t of the per-fill diff R*(mult-1), worst
    weeks (weighted vs flat), n."""
    net_R = f[f'net_R_{which}']
    mult = f.mult
    weighted = book_r_per_risk(net_R, mult)
    flat = book_r_per_risk(net_R, np.ones(len(f)))
    diff = net_R * (mult - 1.0)
    t, ndays = c1430.day_clustered_t(diff, f.day) if len(f) else (float('nan'), 0)
    return dict(n=len(f), weighted_R=weighted, flat_R=flat, dR=weighted - flat,
                t_diff=t, t_ndays=ndays,
                worst_week_weighted=worst_week_ratio(net_R, mult, f.wk),
                worst_week_flat=worst_week_ratio(net_R, np.ones(len(f)), f.wk))


# --------------------------------------------------------------------------------------------
# Report-only: trigger print size class (round lot vs odd lot) and its mean R
# --------------------------------------------------------------------------------------------

def _load_day_tapes(day):
    """research/hod_entry/sip_cache/{day}.pkl.gz -> {sig_key: (trades, quotes)}, or {} if missing."""
    p = os.path.join(HERE, 'sip_cache', f'{day}.pkl.gz')
    if not os.path.exists(p):
        print(f'[cell_1429] WARNING: no sip_cache for day {day} — trigger-size class dropped for its fills',
              file=sys.stderr)
        return {}
    with gzip.open(p, 'rb') as fh:
        return pickle.load(fh)


def trigger_print_size(row, tapes_by_day):
    """Reproduces sip_rebuild.simulate_entry's own trigger-print search for one fill row; returns
    the triggering print's `size` (shares) or None if the tape/print is unavailable."""
    day_tapes = tapes_by_day.get(row.day)
    if not day_tapes:
        return None
    break_m = row.entry_m - 1
    key = sr.sig_key(row.symbol, break_m)
    if key not in day_tapes:
        return None
    trades, _quotes = day_tapes[key]
    S_ns = sr.et_ns(row.day, row.entry_m * 60)
    trigger = round(row.level + sr.TICK, 6)
    bb = trades[(trades.ts >= S_ns - 60 * 10**9) & (trades.ts < S_ns)].sort_values('ts', kind='stable')
    hit = bb[bb.price >= trigger - 1e-9]
    if not len(hit):
        return None
    return float(hit['size'].iloc[0])   # bracket access: `.size` on a DataFrame is the element count, not this column


def size_class_report(fills):
    """Per-split report: n and mean net_R (noslip) by trigger-print size class (round/odd/unknown)."""
    fills = fills.copy()
    tapes_by_day = {d: _load_day_tapes(d) for d in fills.day.unique()}
    sizes = [trigger_print_size(r, tapes_by_day) for r in fills.itertuples()]
    fills['trigger_size'] = sizes
    fills['size_class'] = np.where(fills.trigger_size.isna(), 'unknown',
                                    np.where(fills.trigger_size >= ROUND_LOT, 'round_lot', 'odd_lot'))
    n_unknown = int((fills.size_class == 'unknown').sum())
    if n_unknown:
        print(f'[cell_1429] WARNING: {n_unknown}/{len(fills)} fills have no recoverable trigger print '
              f'(missing tape/print) — excluded from the size-class report', file=sys.stderr)
    g = fills[fills.size_class != 'unknown'].groupby('size_class').net_R.agg(['count', 'mean'])
    return g


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------

def run():
    train_h2, val = c1430.load_base_fills()
    train_h2 = add_sizing(train_h2)
    val = add_sizing(val)

    rows = {}
    for split_name, f in (('TRAIN-H2', train_h2), ('VAL', val)):
        rows[split_name] = {which: score_split(f, which) for which in ('noslip', 'slip')}

    pass_ = (rows['TRAIN-H2']['slip']['dR'] >= 0.05 and rows['VAL']['slip']['dR'] >= 0.05
             and rows['VAL']['slip']['worst_week_weighted'] >= rows['VAL']['slip']['worst_week_flat'])

    print(f'[cell_1429] base fills: TRAIN-H2 n={len(train_h2)} VAL n={len(val)}')
    for split_name in ('TRAIN-H2', 'VAL'):
        for which in ('noslip', 'slip'):
            r = rows[split_name][which]
            print(f"{split_name} [{which}] n={r['n']} weighted_R={r['weighted_R']:+.4f} "
                  f"flat_R={r['flat_R']:+.4f} dR={r['dR']:+.4f} t_diff={r['t_diff']:.2f} "
                  f"(n_days={r['t_ndays']}) worst_wk_weighted={r['worst_week_weighted']:+.2f} "
                  f"worst_wk_flat={r['worst_week_flat']:+.2f}")
    print(f"[cell_1429] -> {'PASS' if pass_ else 'FAIL'}")

    size_reports = {}
    for split_name, f in (('TRAIN-H2', train_h2), ('VAL', val)):
        rep = size_class_report(f)
        size_reports[split_name] = rep
        print(f'[cell_1429] {split_name} trigger-print size class (report-only):\n{rep}')

    return rows, pass_, size_reports


if __name__ == '__main__':
    run()
