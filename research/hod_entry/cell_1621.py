#!/usr/bin/env python3
"""Cells 1,621-1,622 -- research/hod_entry/PREREG_1617.md Frame C (FROZEN 2026-09-28).

SHORT-WINDOW CONFIRMATION: cell 1,487 (`cell_1487.py`) buys the HOD break only after it has held
WITHOUT a $0.01 dip for 15 RTH minutes -- eligible = base fills whose OWN exit is later than
fill_min+15 AND whose bars in (fill_min, fill_min+15] never printed a low <= level-0.01. That book
failed VAL (mean -0.045 R, t -0.52; RESULT_1487.md): by minute 16 the run-up is gone and R'' is
~1.5% of price with the 2R target far away. Frame C asks whether a SHORTER hold -- less run-up
given back, more of the 87-91% "dip within 15 min" cohort still excluded -- rescues the mechanism.

Rule (cells 1,621 W=3, 1,622 W=5 -- IDENTICAL to 1,487 with W substituted for the hardcoded 15):
eligible fills are base fills whose OWN exit minute is > fill_min+W AND no bar in
(fill_min, fill_min+W] has low <= level-0.01. Enter LONG at the ask of the open of minute
fill_min+W+1 (open + the base fill's recovered half_entry -- SAME recovery cell 1,487 uses,
cell_1445.corrected_cost via cell_1478.build_outcome; disclosed, not silent). stop = level-0.01.
R'' = entry-stop. target = entry + 2*R''. Walk on minute bars with sip_rebuild.walk_path from the
entry bar (inclusive). 15:55 EOD at the bid. Costs: entry half-spread via the ask; exit cost by
why -- target=0, stop/stop_bar=cell_1478's SLIP_STOP_BPS (per split), eod/eod_fallback=cell 1,443's
EOD_BPS. R'' < 0.5% of price (RFLOOR_PCT) reported but excluded from the PRIMARY book -- same floor
cell 1,487 applies. runners_lost = base fills that hit target (why=='target') at or before
fill_min+W -- these are trades this frame CANNOT enter (base already exited) even though they were
winners; reported as a cost of waiting, exactly cell 1,487's runners_lost definition with W swapped
for 15.

This script builds BOTH cells (W=3 -> 1621, W=5 -> 1622) in one pass over the same in-memory bars.
Machinery reused verbatim from cell_1487.py (its own import, unmodified): load_base (base book +
outcome_R + half_entry recovery + holdout/wk labels), load_bars (bars_fills_1478.db reader),
count_matched_null (day-stratified draws from the base book, SAME design, seed=1621 per this
PREREG's explicit pin), slot_fills_wk (slot-capped fills/week), and the cost constants
(SLIP_STOP_BPS, EOD_BPS, DIP_TICK, TARGET_R_MULT, RFLOOR_PCT). Only eligibility_and_entry is
reimplemented, parameterized on W (1,487 hardcodes WAIT_M=15/ENTRY_OFFSET_M=16 as module globals,
so it cannot be called with a different window without editing that frozen file, which this task
does not do).

Independent-check note (disclosed, not silent, same status as cell 1,487's own note): half_entry is
NOT a column of causal_arming_causal.csv -- it is recovered via cell_1445.corrected_cost, invoked
inside cell_1478.build_outcome -> cell_1457.build_base_cost, and arrives already attached to the
frame c1487.load_base() returns. Flag for the independent rebuild.

Usage:
    nice -n 19 python3 research/hod_entry/cell_1621.py [--dry-run]

Outputs: cell_1621_fills.csv (both cells, `cell` column distinguishes 1621/1622), RESULT_1621.md.
"""
import argparse
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from research.hod_entry import cell_1445 as c1445            # noqa: E402 -- day_clustered_t, ex_top5_mean
from research.hod_entry import cell_1487 as c1487            # noqa: E402 -- load_base, load_bars, count_matched_null, slot_fills_wk, cost constants
from research.hod_entry import sip_rebuild as sr              # noqa: E402 -- walk_path

DIP_TICK = c1487.DIP_TICK                          # $0.01
TARGET_R_MULT = c1487.TARGET_R_MULT                # 2.0
RFLOOR_PCT = c1487.RFLOOR_PCT                       # 0.5% of price
SLIP_STOP_BPS = c1487.SLIP_STOP_BPS                 # cell-1,478 amendment, per split
EOD_BPS = c1487.EOD_BPS                             # cell-1,443 holdout means, per split

WINDOWS = {1621: 3, 1622: 5}                        # PREREG_1617.md Frame C: W in {3, 5} RTH minutes
NULL_SEED = 1621                                    # PREREG's explicit pin for the Frame-C count-matched null
NULL_DRAWS = c1487.NULL_DRAWS

# Pass bar (frozen, PREREG_1617.md "Pass bar" section C -- "as 1,487"): VAL, primary book.
PASS_MEAN, PASS_T = 0.15, 2.5
PASS_FILLS_WK = 3.0
PASS_NULL_PCTILE = 99.0
PASS_TRAINH2_T = 1.0


def log(msg):
    """Verbose progress, flushed immediately (nohup-safe)."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ================================================================================================
# Frame C -- short-window confirmation entry (W parameterized; 1,487's eligibility_and_entry with
# WAIT_M/ENTRY_OFFSET_M swapped for W/W+1 -- everything else, including the walk and cost rules,
# byte-identical to cell_1487.py)
# ================================================================================================

def eligibility_and_entry_w(base, bars_by_sd, W, cell_id):
    """Per base fill: eligibility (no dip in (fill_min, fill_min+W], base exit_m > fill_min+W),
    then -- for eligible rows with a resolvable entry bar -- the confirmation trade's walk and R''.
    One row per base fill (eligible or not) so every reported share comes off ONE frame, exactly
    cell 1,487's contract."""
    rows = []
    n_no_bars, n_no_window_bars, n_no_entry_bar, n_bad_r2 = 0, 0, 0, 0
    for r in base.itertuples():
        bars = bars_by_sd.get((r.symbol, r.day))
        rec = dict(cell=cell_id, W=W, day=r.day, symbol=r.symbol, split=r.split, holdout=r.holdout,
                   wk=r.wk, fill_min=r.fill_min, level=r.level, base_exit_m=r.exit_m,
                   base_why=r.why, base_outcome_R=r.outcome_R,
                   entry=np.nan, exit=np.nan, net_R2=np.nan)
        if bars is None or not len(bars):
            n_no_bars += 1
            rec.update(eligible=False, entered=False, why='no_bars')
            rows.append(rec)
            continue
        win = bars[(bars.m > r.fill_min) & (bars.m <= r.fill_min + W)]
        if not len(win):
            n_no_window_bars += 1
        dip = bool((win.l <= r.level - DIP_TICK).any()) if len(win) else False
        eligible = bool((r.exit_m > r.fill_min + W) and not dip)
        rec['eligible'] = eligible
        if not eligible:
            rec.update(entered=False, why=('dip_in_window' if dip else 'base_exited_by_W'))
            rows.append(rec)
            continue

        entry_bar_m = int(np.floor(r.fill_min)) + W + 1
        eb = bars[bars.m == entry_bar_m]
        if not len(eb):
            n_no_entry_bar += 1
            rec.update(entered=False, why='no_entry_bar')
            rows.append(rec)
            continue
        entry = float(eb.o.iloc[0]) + float(r.half_entry)
        stop = r.level - DIP_TICK
        R2 = entry - stop
        if not (R2 > 0):
            n_bad_r2 += 1
            rec.update(entered=False, why='nonpositive_R2')
            rows.append(rec)
            continue
        target = entry + TARGET_R_MULT * R2
        path = bars[bars.m >= entry_bar_m].sort_values('m')
        exit_m2, exit_price2, why2 = sr.walk_path(entry, stop, target, path)
        raw_R2 = (exit_price2 - entry) / R2
        if why2 == 'target':
            cost_R2 = 0.0
        elif why2 in ('stop', 'stop_bar'):
            cost_R2 = exit_price2 * SLIP_STOP_BPS[r.split] / 1e4 / R2
        else:                                                     # eod, eod_fallback
            cost_R2 = exit_price2 * EOD_BPS[r.split] / 1e4 / R2
        net_R2 = raw_R2 - cost_R2
        rec.update(entered=True, why=why2, entry_bar_m=entry_bar_m, entry=entry, stop2=stop,
                   R2=R2, target=target, exit_m2=exit_m2, exit=exit_price2,
                   raw_R2=raw_R2, cost_R2=cost_R2, net_R2=net_R2,
                   r2_pct_price=100.0 * R2 / entry, below_floor=(R2 / entry) < RFLOOR_PCT,
                   entry_m=r.fill_min + W + 1)
        rows.append(rec)
    log(f'eligibility_and_entry_w(W={W}, cell={cell_id}): {len(rows)} base fills scored; '
        f'data-loss counts: no_bars={n_no_bars} no_window_bars={n_no_window_bars} '
        f'no_entry_bar={n_no_entry_bar} nonpositive_R2={n_bad_r2}')
    if n_no_bars:
        log(f'WARNING: {n_no_bars} fills had no bars row at all for (symbol, day) -- excluded, not imputed')
    return pd.DataFrame(rows)


def score_w(scored, base, holdout, W, seed):
    """One holdout's PRIMARY (R'' >= 0.5% floor) and ALL-eligible (floor included) books --
    cell 1,487's score_1487, minus the cache-only decoy check (not part of this PREREG's report
    list) and minus WAIT_M (parameterized on W instead)."""
    d = scored[scored.holdout == holdout]
    total = len(d)
    weeks = base[base.holdout == holdout].wk.nunique()
    eligible = d[d.eligible]
    eligible_share = len(eligible) / total if total else np.nan
    entered = d[d.entered]
    runners_lost = int(((base.holdout == holdout) & (base.why == 'target') &
                         (base.exit_m <= base.fill_min + W)).sum())
    runners_lost_share = runners_lost / total if total else np.nan
    calib_mean = float(eligible.base_outcome_R.mean()) if len(eligible) else np.nan

    out = {}
    floor_mask = entered.below_floor.astype(bool) if len(entered) else pd.Series(dtype=bool)
    for label, book in (('primary', entered[~floor_mask]), ('all_eligible', entered)):
        n = len(book)
        mean_net = float(book.net_R2.mean()) if n else np.nan
        t = c1445.day_clustered_t(book.net_R2, book.day) if n > 1 else np.nan
        extop5 = float(c1445.ex_top5_mean(book.net_R2)) if n else np.nan
        fwk = c1487.slot_fills_wk(book, weeks)
        nul = c1487.count_matched_null(base, book, holdout, seed=seed, n_draws=NULL_DRAWS)
        paired_base_mean = float(book.base_outcome_R.mean()) if n else np.nan
        delta = book.net_R2 - book.base_outcome_R if n else pd.Series(dtype=float)
        paired_delta = float(delta.mean()) if n else np.nan
        paired_delta_t = c1445.day_clustered_t(delta, book.day) if n > 1 else np.nan
        paired_delta_extop5 = float(c1445.ex_top5_mean(delta)) if n else np.nan
        r_pct_median = float(book.r2_pct_price.median()) if n else np.nan
        out[label] = dict(n=n, mean_net_R=mean_net, t=t, ex_top5=extop5, fills_wk=fwk,
                           null_pctile=nul['null_pctile'], eligible_share=eligible_share,
                           runners_lost_share=runners_lost_share, paired_base_mean=paired_base_mean,
                           paired_delta=paired_delta, paired_delta_t=paired_delta_t,
                           paired_delta_ex_top5=paired_delta_extop5,
                           calibration_base_mean_on_cohort=calib_mean, r_pct_median=r_pct_median)
    return out


def evaluate_pass(sc):
    """PREREG_1617.md pass bar C: 'as 1,487' -- mean net R'' >= 0.15, t >= 2.5, ex-top-5% > 0,
    fills/wk >= 3.0, count-matched null >= 99, TRAIN-H2 same sign t >= 1.0."""
    val, tr = sc['VAL']['primary'], sc['TRAIN-H2']['primary']
    val_pass = (val['mean_net_R'] >= PASS_MEAN and val['t'] >= PASS_T and
                val['ex_top5'] > 0 and val['fills_wk'] >= PASS_FILLS_WK and
                val['null_pctile'] >= PASS_NULL_PCTILE and
                np.sign(tr['mean_net_R']) == np.sign(val['mean_net_R']) and
                tr['t'] >= PASS_TRAINH2_T)
    tr_pass = tr['t'] >= PASS_TRAINH2_T and np.sign(tr['mean_net_R']) == np.sign(val['mean_net_R'])
    return dict(val=bool(val_pass), trainh2=bool(tr_pass))


# ================================================================================================
# Report
# ================================================================================================

def fmt_table(rows, cols):
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c)
            vals.append(f'{v:.4f}' if isinstance(v, float) else str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def write_result_md(all_sc, all_verdict, caveats, path):
    lines = ['# RESULT -- cells 1,621-1,622: short-window confirmation entry (Frame C, PREREG_1617.md)', '']
    lines.append("Same mechanism as cell 1,487 (buy after the break holds without a $0.01 dip, "
                  "stop=level-0.01, target=entry+2R''), window W swapped for 15: 1,621 W=3 min, "
                  "1,622 W=5 min.")
    lines.append('')
    vkey = {'TRAIN-H2': 'trainh2', 'VAL': 'val'}
    rows = []
    for cell_id, W in WINDOWS.items():
        sc = all_sc[cell_id]
        verdict = all_verdict[cell_id]
        for h in ('TRAIN-H2', 'VAL'):
            for label in ('primary', 'all_eligible'):
                r = dict(sc[h][label])
                r['cell'] = cell_id
                r['W'] = W
                r['holdout'] = h
                r['book'] = label
                r['passes_bar'] = verdict[vkey[h]]
                rows.append(r)
    cols = ['cell', 'W', 'holdout', 'book', 'n', 'eligible_share', 'r_pct_median',
            'runners_lost_share', 'mean_net_R', 't', 'ex_top5', 'fills_wk', 'null_pctile',
            'paired_base_mean', 'paired_delta', 'paired_delta_t', 'paired_delta_ex_top5',
            'calibration_base_mean_on_cohort', 'passes_bar']
    lines.append(fmt_table(rows, cols))
    lines.append('')
    lines.append('## Caveats')
    for c in caveats:
        lines.append(f'* {c}')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log(f'wrote {path}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true', help='200-row smoke sample, seed 1621')
    args = ap.parse_args(argv)

    t0 = time.time()
    base = c1487.load_base()
    if args.dry_run:
        base = base.sample(n=min(200, len(base)), random_state=NULL_SEED).reset_index(drop=True)
        log(f'--dry-run: subsampled to {len(base)} rows')

    log('loading minute bars (bars_fills_1478.db) -- shared across W=3 and W=5')
    bars_by_sd = c1487.load_bars(list(zip(base.symbol, base.day)))

    all_scored = {}
    all_sc = {}
    all_verdict = {}
    for cell_id, W in WINDOWS.items():
        log(f'=== cell {cell_id}: W={W} min ===')
        scored = eligibility_and_entry_w(base, bars_by_sd, W, cell_id)
        sc = {h: score_w(scored, base, h, W, seed=NULL_SEED) for h in ('TRAIN-H2', 'VAL')}
        verdict = evaluate_pass(sc)
        log(f'cell {cell_id} VAL primary: n={sc["VAL"]["primary"]["n"]} '
            f'mean_net_R={sc["VAL"]["primary"]["mean_net_R"]:.4f} '
            f't={sc["VAL"]["primary"]["t"]:.2f} fills_wk={sc["VAL"]["primary"]["fills_wk"]:.2f} '
            f'null_pctile={sc["VAL"]["primary"]["null_pctile"]:.1f} passes={verdict["val"]}')
        all_scored[cell_id] = scored
        all_sc[cell_id] = sc
        all_verdict[cell_id] = verdict

    combined = pd.concat(all_scored.values(), ignore_index=True)
    out_cols = ['cell', 'W', 'split', 'holdout', 'wk', 'day', 'symbol', 'fill_min', 'level',
                'eligible', 'entered', 'entry', 'exit', 'why', 'net_R2', 'base_outcome_R', 'R2',
                'r2_pct_price', 'below_floor', 'entry_bar_m', 'exit_m2']
    out_cols = [c for c in out_cols if c in combined.columns]
    combined[out_cols].to_csv(os.path.join(HERE, 'cell_1621_fills.csv'), index=False)
    log(f'wrote cell_1621_fills.csv: {len(combined)} rows ({len(all_scored[1621])} per cell x 2 cells)')

    n_no_bars = {cid: int((s.why == 'no_bars').sum()) for cid, s in all_scored.items()}
    n_no_entry_bar = {cid: int((s.why == 'no_entry_bar').sum()) for cid, s in all_scored.items()}
    n_bad_r2 = {cid: int((s.why == 'nonpositive_R2').sum()) for cid, s in all_scored.items()}
    caveats = [
        "half_entry source: same recovered-not-read status as cell 1,487 -- it is NOT a column of "
        "causal_arming_causal.csv; c1487.load_base() attaches it via cell_1445.corrected_cost "
        "(called inside cell_1478.build_outcome -> cell_1457.build_base_cost). Flag for the "
        "independent rebuild.",
        f"data loss cell 1621 (W=3): no_bars={n_no_bars[1621]}, no_entry_bar={n_no_entry_bar[1621]} "
        f"(halt/gap at minute fill_min+4), nonpositive_R2={n_bad_r2[1621]}.",
        f"data loss cell 1622 (W=5): no_bars={n_no_bars[1622]}, no_entry_bar={n_no_entry_bar[1622]} "
        f"(halt/gap at minute fill_min+6), nonpositive_R2={n_bad_r2[1622]}.",
        "Excluded rows are counted, never imputed -- see cell_1621_fills.csv `why` for every "
        "non-entered row's reason (no_bars / dip_in_window / base_exited_by_W / no_entry_bar / "
        "nonpositive_R2).",
        "EOD_BPS (11.5/9.7 bps) and SLIP_STOP_BPS are holdout-level EXPECTED VALUES (cells "
        "1,443/1,478), not this leg's own measured tape -- same disclosed-proxy status cell 1,487 "
        "carries.",
        "fills_wk is slot-capped (research/hod_consol.simulate_slots, 4 concurrent/12 daily default "
        "caps), not a raw count/week ratio.",
        "count-matched null uses seed=1621 for BOTH cells (this PREREG's explicit pin names only "
        "one seed for Frame C) -- a reading choice, disclosed here for the rebuild.",
        "paired_delta/_t/_ex_top5 (net_R2 - base_outcome_R, day-clustered) are ADDED beyond the "
        "PREREG's literal report list, to catch tail-driven paired lift per the paired-lift-tail-"
        "check convention -- not part of the frozen pass bar.",
        "R''-as-%-of-price rail: r_pct_median reported per book; PRIMARY excludes R'' < 0.5% of "
        "price (RFLOOR_PCT), same floor cell 1,487 applies, never moved.",
        f"runtime {time.time() - t0:.0f}s.",
    ]
    write_result_md(all_sc, all_verdict, caveats, os.path.join(HERE, 'RESULT_1621.md'))
    log('done')
    return all_sc, all_verdict


if __name__ == '__main__':
    main()
