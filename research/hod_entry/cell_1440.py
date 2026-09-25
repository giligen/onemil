"""Cell 1,440 — stop distance (research/hod_entry/PREREG_WEEKEND.md, frozen 2026-09-25 18:35 UTC,
amendment 20:15 UTC base-book swap).

Base book = cell 1,438's `causal_arming_causal.csv` fills (status == 'fill', 9,911 rows; correct
levels, the live rule). Stop = consolidation low is FLOORED at 0.8% of price (variant A: floor
only) and additionally CAPPED at 3% (variant B: floor + cap). new_stop = fill x (1 - d_new); R and
qty recomputed; target = fill + 2*R_new. Only rows where the floor/cap actually change d are
re-walked (ΔR = 0 kept exactly for the rest, per the amendment):

  * fill bar = the bar with m == floor(fill_min) (fill_min is a fractional in-bar timestamp in the
    base CSV; the bar grid is integer-minute).
  * if that bar's LOW <= new_stop: stopped INSIDE the fill bar at new_stop (conservative — the
    amendment's rule, since the tick instant is not in the CSV). why = 'stop_infill', raw R = -1
    by construction (new_stop is exactly R_new below fill).
  * else: re-walk from the bar AFTER the fill bar via `sip_rebuild.walk_path(fill, new_stop,
    fill + 2*R_new, path)` — stop-first on a bar touching both, gap-through at the open, the 15:55
    bar exits at its open.
  * cost: dollar cost is held fixed at the base's cost_$ = cost_R_old * R_old; new cost_R =
    cost_$ / R_new (so the ANCHOR half-spread/exit-cost dollars don't change, only their R terms).
  * 30 bps stop-slip variant (report-only, PREREG_WEEKEND.md line 8): subtract
    0.0030 * exit_price / R on stop exits ('stop' or 'stop_bar' at base, 'stop'/'stop_infill' at
    cell), computed for BOTH the base and the cell.

Pass bar (frozen, line 56): ΔR >= +0.05 on BOTH holdouts (TRAIN-H2, VAL) AND VAL t >= 2. A lift on
this base (raw R ~ 0, see 1,438/1,442's finding) is reported as a lift, never a book — PASS ships
the mechanism to the dry run only (amendment 20:15 UTC).
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sip_rebuild as sr                                    # noqa: E402
import causal_arming as ca                                  # noqa: E402 — load_day_bars, BARS_SIP_URI
import cell_1430 as c1430                                    # noqa: E402 — day_clustered_t, ex_top5_pct

ROOT = sr.ROOT
BASE_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
OUT_CSV = os.path.join(HERE, 'cell_1440_fills.csv')
REPORT_MD = os.path.join(HERE, 'RESULT_1440.md')
WEEKEND_MD = os.path.join(HERE, 'WEEKEND_RESULTS.md')
STOP_SLIP_BP = 0.0030
FLOOR_PCT = 0.008
CAP_PCT = 0.03
STOP_WHY_BASE = {'stop', 'stop_bar'}
STOP_WHY_CELL = {'stop', 'stop_infill'}
log = sr.log


# --------------------------------------------------------------------------------------------- variant math
def new_stop_distance(d_old, cap_pct=None):
    """Floor d_old at FLOOR_PCT; if cap_pct given also cap at it (variant B)."""
    d = np.maximum(d_old, FLOOR_PCT)
    if cap_pct is not None:
        d = np.minimum(d, cap_pct)
    return d


def load_base():
    """Base = 1,438's causal_arming_causal.csv, status == 'fill'. Verifies R == fill - stop."""
    df = pd.read_csv(BASE_CSV, low_memory=False)
    fills = df[df.status == 'fill'].copy()
    fills['d_old'] = (fills.fill - fills.stop) / fills.fill
    assert np.allclose(fills.R, fills.fill - fills.stop), 'base R != fill - stop'
    log(f'[load_base] {len(fills)} fills, {fills.day.nunique()} days, '
        f'd_old floor<0.008 {(fills.d_old < FLOOR_PCT).sum()}, cap>0.03 {(fills.d_old > CAP_PCT).sum()}')
    return fills.reset_index(drop=True)


def base_stop_slip(row):
    """Base net R under the 30bps stop-slip charge (report-only)."""
    if row.why in STOP_WHY_BASE:
        return row.net_R - STOP_SLIP_BP * row.exit_price / row.R
    return row.net_R


# --------------------------------------------------------------------------------------------- one-trade recompute
def recompute_one(row, bars, d_new):
    """Recompute a single fill's outcome under a new stop distance d_new. `bars` = that
    symbol-day's minute bars (m,o,h,l,c,v) from causal_arming.load_day_bars. Returns a dict of the
    new R/raw/cost/net/exit fields, or None if bars/fill-bar are missing (logged, excluded)."""
    fill = float(row.fill)
    new_stop = fill * (1.0 - d_new)
    R_new = fill - new_stop
    if R_new <= 0:
        log(f'[recompute] WARNING {row.day} {row.symbol}: R_new <= 0 ({R_new}) — skipping')
        return None
    fill_m = int(row.fill_min)
    fb = bars[bars.m == fill_m]
    if not len(fb):
        log(f'[recompute] WARNING {row.day} {row.symbol}: no fill bar at m={fill_m} — skipping')
        return None
    fb_low = float(fb.l.iloc[0])
    cost_dollar = float(row.cost_R) * float(row.R)
    if fb_low <= new_stop + 1e-9:
        exit_m, exit_price, why = fill_m, new_stop, 'stop_infill'
    else:
        path = bars[bars.m > fill_m].sort_values('m', kind='stable')
        if not len(path):
            log(f'[recompute] WARNING {row.day} {row.symbol}: no path after fill bar m={fill_m} — skipping')
            return None
        target = fill + sr.TARGET_R * R_new
        exit_m, exit_price, why = sr.walk_path(fill, new_stop, target, path)
    raw = (exit_price - fill) / R_new
    cost_R = cost_dollar / R_new
    net = raw - cost_R
    stopslip = net - (STOP_SLIP_BP * exit_price / R_new if why in STOP_WHY_CELL else 0.0)
    return dict(stop=new_stop, R=R_new, exit_m=exit_m, exit_price=exit_price, why=why,
                raw_R=raw, cost_R=cost_R, net_R=net, net_R_stopslip=stopslip, changed=True,
                stopped_infill=(why == 'stop_infill'))


def build_variant(base, cap_pct, workers=2):
    """Build the full cell book for one variant (cap_pct=None -> A floor-only; else B floor+cap).
    Rows whose d is unchanged keep the base outcome exactly (ΔR = 0)."""
    d_new = new_stop_distance(base.d_old.values, cap_pct)
    changed_mask = ~np.isclose(d_new, base.d_old.values, atol=1e-12)
    log(f'[build_variant cap={cap_pct}] {changed_mask.sum()} / {len(base)} rows change stop distance')
    out = []
    con = sqlite3.connect(sr.CACHE_DB_URI, uri=True, timeout=120)
    sipcon = sqlite3.connect(ca.BARS_SIP_URI, uri=True, timeout=120)
    src = {}
    changed = base[changed_mask]
    for day, sub in changed.groupby('day'):
        bars_by_sym = ca.load_day_bars(con, day, sub.symbol.tolist(), sipcon, src)
        for idx, row in sub.iterrows():
            b = bars_by_sym.get(row.symbol)
            r = None
            if b is not None:
                r = recompute_one(row, b, d_new[base.index.get_loc(idx)])
            if r is None:
                log(f'[build_variant] {day} {row.symbol}: recompute failed — falling back to base outcome')
                r = dict(stop=row.stop, R=row.R, exit_m=row.exit_m, exit_price=row.exit_price, why=row.why,
                         raw_R=row.raw_R, cost_R=row.cost_R, net_R=row.net_R,
                         net_R_stopslip=base_stop_slip(row), changed=False, stopped_infill=False)
            out.append((idx, r))
    con.close(); sipcon.close()
    recompute_map = dict(out)
    rows = []
    for idx, row in base.iterrows():
        if idx in recompute_map:
            r = recompute_map[idx]
        else:
            r = dict(stop=row.stop, R=row.R, exit_m=row.exit_m, exit_price=row.exit_price, why=row.why,
                     raw_R=row.raw_R, cost_R=row.cost_R, net_R=row.net_R,
                     net_R_stopslip=base_stop_slip(row), changed=False, stopped_infill=False)
        rows.append({'day': row.day, 'symbol': row.symbol, 'split': row.split, **r})
    log(f'[build_variant cap={cap_pct}] src={src}')
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- scoring
def score_holdout(base, cell, split_name):
    """One row of the RESULT table for one (holdout, variant)."""
    b = base[base.split == split_name].reset_index(drop=True)
    c = cell[cell.split == split_name].reset_index(drop=True)
    assert len(b) == len(c) and (b.day.values == c.day.values).all() and (b.symbol.values == c.symbol.values).all()
    delta = c.net_R.values - b.net_R.values
    t, n_days = c1430.day_clustered_t(delta, b.day.values)
    ss_delta = c.net_R_stopslip.values - base_stopslip_series(b).values
    return dict(
        split=split_name, n=len(b), base_R=float(b.net_R.mean()), cell_R=float(c.net_R.mean()),
        delta_R=float(delta.mean()), t=t, n_days=n_days, ex_top5=float(c1430.ex_top5_pct(c.net_R.values)),
        changed_pct=float(c.changed.mean() * 100), stopped_infill_pct=float(c.stopped_infill.mean() * 100),
        stopslip_delta=float(ss_delta.mean()),
    )


def base_stopslip_series(b):
    return b.apply(base_stop_slip, axis=1)


# --------------------------------------------------------------------------------------------- report
def write_report(scores_a, scores_b):
    def verdict(scores, label):
        train = next(s for s in scores if s['split'] == 'TRAIN')
        val = next(s for s in scores if s['split'] == 'VAL')
        passed = train['delta_R'] >= 0.05 and val['delta_R'] >= 0.05 and val['t'] >= 2
        return passed, train, val

    def table(scores, label):
        lines = [f'### Variant {label}', '', '| holdout | n | base R | cell R | ΔR | t | ex-top-5% | changed% | stop-slip ΔR |',
                  '|---|---|---|---|---|---|---|---|---|']
        for s in scores:
            lines.append(f"| {s['split']} | {s['n']} | {s['base_R']:.4f} | {s['cell_R']:.4f} | "
                          f"{s['delta_R']:+.4f} | {s['t']:.2f} | {s['ex_top5']:.4f} | "
                          f"{s['changed_pct']:.1f}% | {s['stopslip_delta']:+.4f} |")
        return '\n'.join(lines)

    pa, tra, va = verdict(scores_a, 'A')
    pb, trb, vb = verdict(scores_b, 'B')
    body = (
        f"# Cell 1,440 — stop distance\n\nBase = 1,438 causal_arming_causal.csv fills (n={tra['n']+va['n']}).\n\n"
        f"{table(scores_a, 'A (floor 0.8% only)')}\n\n{table(scores_b, 'B (floor 0.8% + cap 3%)')}\n\n"
        f"Variant A: {'PASS' if pa else 'FAIL'} (TRAIN ΔR {tra['delta_R']:+.4f}, VAL ΔR {va['delta_R']:+.4f}, VAL t {va['t']:.2f}).\n"
        f"Variant B: {'PASS' if pb else 'FAIL'} (TRAIN ΔR {trb['delta_R']:+.4f}, VAL ΔR {vb['delta_R']:+.4f}, VAL t {vb['t']:.2f}).\n\n"
        f"Caveats: variant A touches only {tra['changed_pct']:.2f}%/{va['changed_pct']:.2f}% of fills (floor almost "
        f"never binds) — near a no-op by construction. Variant B touches ~{trb['changed_pct']:.1f}%/{vb['changed_pct']:.1f}% "
        f"(the 3% cap). {vb['stopped_infill_pct']:.1f}% of VAL variant-B recomputes are stopped inside the fill bar "
        f"(conservative, minute-low check). Per the 20:15 amendment, a PASS here is a lift on the 1,438 base "
        f"(raw R ~ 0), reported as a lift, never a book — ships to dry run only.\n"
    )
    with open(REPORT_MD, 'w') as f:
        f.write(body)
    log(f'[report] wrote {REPORT_MD}')

    weekend_block = f"\n## 1,440 — stop distance\n\n{table(scores_a, 'A (floor 0.8% only)')}\n\n{table(scores_b, 'B (floor 0.8% + cap 3%)')}\n\n" \
        f"Verdict: Variant A {'PASS' if pa else 'FAIL'} (near no-op, {tra['changed_pct']:.2f}%/{va['changed_pct']:.2f}% of fills touched). " \
        f"Variant B {'PASS' if pb else 'FAIL'} (VAL ΔR {vb['delta_R']:+.4f}, t {vb['t']:.2f} vs pass bar +0.05/t≥2). " \
        f"Lift on a raw-R~0 base — never a book (amendment 20:15 UTC).\n"
    with open(WEEKEND_MD, 'a') as f:
        f.write(weekend_block)
    log(f'[report] appended to {WEEKEND_MD}')
    return pa, pb


def main():
    base = load_base()
    cell_a = build_variant(base, cap_pct=None)
    cell_a.to_csv(OUT_CSV.replace('.csv', '_A.csv'), index=False)
    cell_b = build_variant(base, cap_pct=CAP_PCT)
    cell_b.to_csv(OUT_CSV.replace('.csv', '_B.csv'), index=False)
    scores_a = [score_holdout(base, cell_a, s) for s in ('TRAIN', 'VAL')]
    scores_b = [score_holdout(base, cell_b, s) for s in ('TRAIN', 'VAL')]
    write_report(scores_a, scores_b)
    for s in scores_a + scores_b:
        log(f"[score] {s}")


if __name__ == '__main__':
    main()
