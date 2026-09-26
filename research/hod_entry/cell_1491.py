#!/usr/bin/env python3
"""Cell 1,491-1,492 -- research/hod_entry/PREREG_1491.md (FROZEN 2026-09-26).

SHALLOW STOP: paired re-walk of the 9,911-fill base book (cell 1,438/1,478 standard, status=='fill',
TRAIN-H2 + VAL only -- TEST does not exist in these files) with the stop moved from the consolidation
low to `level * (1 - s)` for s in {0.25%, 0.50%, 0.75%}, entry UNCHANGED (= the base fill). Two target
rules per stop -> six books:
  * 1,491 SCALP: target = fill + 2 * R_s   (R_s = fill - stop_s, the book's OWN risk unit)
  * 1,492 ASYMMETRIC: target = the BASE target = fill + 2 * R_base (unchanged reward, smaller risk)

Walk semantics (sip_rebuild.walk_path, BARS ONLY -- bars_fills_1478.db, no tick/quote refetch):
  * fill bar (m == floor(fill_min)): low <= stop_s -> stop_bar, exit_price = stop_s (conservative --
    no gap-through-at-open pricing on the fill bar itself, per the PREREG's own phrasing). Base
    why=='stop_bar' rows (the ORIGINAL, deeper stop was already breached intrabar per cell 1,438's
    tick simulation) are stop_bar deterministically for every s here too: stop_s > stop_base, so a
    print at/below stop_base necessarily crossed stop_s in the same bar -- no bar data is needed to
    know this, and bars_fills_1478.db has no intrabar tick order anyway.
  * else: sip_rebuild.walk_path(fill, stop_s, target, path) on bars with m > the fill bar's own
    minute, m <= EOD_M (955) -- gap-through, stop-priority-over-target on a bar touching both, the
    15:55 bar's OPEN for 'eod', sip_rebuild's own path-ended-early 'eod_fallback' -- all unchanged.

Cost (both books' OWN R units; entry side unchanged from the base fill):
  * half_entry (HE, dollars): recovered from the base fill via cell_1445.corrected_cost -- identical
    entry, independent of s or the target rule.
  * stop / stop_bar exit: the verified stop-limit slip, PREREG_1478 amendment's blend (0.88 * the
    filled-stop holdout mean bps + 0.12 * the no-fill-tail mean bps, TRAIN-H2 / VAL), IN THE BOOK'S
    OWN R (SLIP_STOP_BPS below; same formula as cell_1478.substitute_stop_slip with R_s in place of
    the base R, per the PREREG's explicit "in units of the book's own R").
  * target exit: "target = limit" -- the exit fills exactly at the target price (already what
    walk_path returns); no further exit-side cost is charged beyond HE.
  * eod / eod_fallback exit: the 1,443 EOD holdout means (11.5 / 9.7 bps, RESULT_1443.md's "eod" row,
    TRAIN-H2 / VAL), same exit-price-scaled bps -> R conversion as the stop-limit slip.
  net_R_own = (exit_price - fill)/R_s - HE/R_s - exit_slip/R_s        (exit_slip = 0 for 'target')
  net_pct   = net_R_own * R_s / fill * 100                            (dollar P&L / fill price, in %)

Base comparison (paired, the SAME 9,911 fills): outcome_R via cell_1478.build_outcome (the amendment's
slip-substituted standard -- NOT the raw legacy net_R column in causal_arming_causal.csv);
base_pct = outcome_R * R_base / fill * 100. delta_pct = net_pct - base_pct; day-clustered t on
delta_pct (cell_1445.day_clustered_t).

Dip-depth table (report-only, no pass bar): the lowest print below the level within 15 min of the
fill, from rebuild_1481_fills.csv's `dip_low` (status=='fill', the limit-price-convention rebuild --
preferred per the PREREG over cell_1481_fills.csv), bps = (level - dip_low)/level*1e4; the base
outcome_R by bucket (<=25, 25-50, 50-75, >75 bps). Fills absent from that file (status != 'fill' there,
or the (day,symbol) missing entirely) fall back to the min LOW over bars in (fill_min, fill_min+15] on
bars_fills_1478.db -- flagged, coverage of each source reported.

Usage:
    nice -n 19 python3 research/hod_entry/cell_1491.py [--smoke]

    --smoke walks a fixed 300-row sample (seed 1491) instead of the full 9,911-row book, for fast
    pipeline verification; writes SMOKE-suffixed outputs so they never collide with a full run.

Outputs: research/hod_entry/cell_1491_fills.csv (one row per fill x book: exit_m, exit_price, why,
net_R_own, net_pct) and research/hod_entry/RESULT_1491.md.

Not allowed (PREREG): adding stops or targets to the grid; choosing s on VAL for TEST; reading TEST
more than once -- TEST does not exist in these files and is never read here.
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
sys.path.insert(0, HERE)
import sip_rebuild as sr            # noqa: E402  (EOD_M, TARGET_R, walk_path)
import cell_1445 as c1445           # noqa: E402  (load_base_book, corrected_cost, day_clustered_t, ex_top5_mean, winner_capped_mean, weeks_spanned)
import cell_1478 as c1478           # noqa: E402  (build_outcome -- the amendment-slip-substituted base standard)
import cell_1479 as c1479           # noqa: E402  (load_bars_1478, BARS_DB -- the single fresh bar store)

LOG_PATH = os.path.join(HERE, 'cell_1491.log')
OUT_CSV = os.path.join(HERE, 'cell_1491_fills.csv')
RESULT_MD = os.path.join(HERE, 'RESULT_1491.md')
DIP_CSV_1481 = os.path.join(HERE, 'rebuild_1481_fills.csv')

STOPS = [0.0025, 0.0050, 0.0075]        # s in {0.25%, 0.50%, 0.75%} of the level
RULES = ['scalp', 'asym']
HOLDOUTS = ['TRAIN-H2', 'VAL']
# PREREG_1478 amendment blend, RECOMPUTED IN THE BOOK'S OWN R (own-R units, per PREREG_1491's phrasing)
SLIP_STOP_BPS = {'TRAIN-H2': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}
EOD_BPS_1443 = {'TRAIN-H2': 11.5, 'VAL': 9.7}          # RESULT_1443.md, "eod" row, mean column
DIP_BUCKETS = [(0.0, 25.0), (25.0, 50.0), (50.0, 75.0), (75.0, np.inf)]
WINNER_CAP_R = 3.0                                      # "+3 base R", per the pass bar
SHIP_MEDIAN_PCT = 0.5                                   # R-must-exceed-spread rail
PASS_DELTA_PCT = 0.15
PASS_T = 2.5
PASS_OWN_PCT = 0.20
PASS_FILLS_WK = 3.0
PASS_TRAIN_T = 1.0


def log(msg):
    """Verbose progress, flushed immediately and appended to LOG_PATH (print() is buffered under nohup)."""
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def book_name(s, rule):
    """e.g. s=0.005, rule='scalp' -> 'SCALP_050' (050 = 0.50%, x100 to keep it an int tag)."""
    tag = f'{s * 100:04.2f}'.replace('.', '')
    return f"{'SCALP' if rule == 'scalp' else 'ASYM'}_{tag}"


# ================================================================================================
# Base book (entry unchanged) + the amendment's paired outcome_R
# ================================================================================================

def load_base():
    """The 9,911-fill base book with half_entry (entry cost, unchanged across every new book) and
    cell_1478's amendment-standard outcome_R (the paired base comparison) attached. Reuses
    cell_1457.build_base_cost (which recovers half_entry via cell_1445.corrected_cost internally)
    and cell_1478.substitute_stop_slip -- exactly cell_1478.build_outcome's Step 2, minus the
    label-building step this cell has no use for."""
    fills = c1445.load_base_book()
    enriched, _ = c1478.c1457.build_base_cost(fills)
    outcome, _, _ = c1478.substitute_stop_slip(enriched)
    fills = fills.copy()
    fills['half_entry'] = enriched['half_entry'].to_numpy()
    fills['outcome_R'] = outcome.to_numpy()
    fills['base_pct'] = fills['outcome_R'] * fills['R'] / fills['fill'] * 100.0
    n = len(fills)
    if n != 9911:
        log(f'WARNING base book has {n} rows, PREREG expects 9,911 -- proceeding, flagged')
    return fills


# ================================================================================================
# Bar-only walk (the PREREG's fill-bar rule, then sip_rebuild.walk_path from the next bar)
# ================================================================================================

def walk_one(row, bars, stop_s, target):
    """(exit_m, exit_price, why) for one fill under the shallow stop, bars-only. Returns
    (None, None, 'no_data') when bars_fills_1478.db has no usable bar for this fill."""
    m_break = int(np.floor(row.fill_min))
    if row.why == 'stop_bar':
        # the ORIGINAL (deeper) stop was already breached intrabar (cell 1,438's tick sim) -- stop_s
        # (shallower, closer to fill) was necessarily crossed too. No bar lookup needed or possible.
        return m_break, float(stop_s), 'stop_bar'
    b = bars.get(row.symbol)
    if b is None or not len(b):
        log(f'WARNING no bars_fills_1478.db bars for {row.symbol} {row.day} -- fill excluded (no_data)')
        return None, None, 'no_data'
    fb = b[b.m == m_break]
    if not len(fb):
        log(f'WARNING fill bar m={m_break} missing for {row.symbol} {row.day} -- fill excluded (no_data)')
        return None, None, 'no_data'
    if float(fb.iloc[0].l) <= stop_s:
        return m_break, float(stop_s), 'stop_bar'
    path = b[(b.m > m_break) & (b.m <= sr.EOD_M)]
    if not len(path):
        return m_break, float(row.fill), 'no_path'
    exit_m, exit_price, why = sr.walk_path(row.fill, stop_s, target, path)
    return exit_m, exit_price, why


def cost_and_net(fill, R_s, holdout, half_entry, exit_price, why):
    """(raw_R, net_R_own, net_pct) per the module docstring's cost recipe."""
    if why in ('stop', 'stop_bar'):
        exit_slip = exit_price * SLIP_STOP_BPS[holdout] / 1e4
    elif why in ('eod', 'eod_fallback'):
        exit_slip = exit_price * EOD_BPS_1443[holdout] / 1e4
    else:                                    # target, no_path
        exit_slip = 0.0
    raw_R = (exit_price - fill) / R_s
    net_R = raw_R - half_entry / R_s - exit_slip / R_s
    net_pct = net_R * R_s / fill * 100.0
    return raw_R, net_R, net_pct


def run_book(fills, bars_by_day, s, rule):
    """One (stop, target-rule) book, walked on every base fill. Returns a per-fill DataFrame."""
    rows = []
    for r in fills.itertuples():
        bars = bars_by_day.get(r.day, {})
        stop_s = r.level * (1.0 - s)
        R_s = r.fill - stop_s
        if R_s <= 0:
            log(f'WARNING {r.symbol} {r.day}: stop_s {stop_s:.4f} >= fill {r.fill:.4f} -- '
                f'excluded (non_positive_R)')
            continue
        target = r.fill + sr.TARGET_R * R_s if rule == 'scalp' else r.fill + sr.TARGET_R * r.R
        exit_m, exit_price, why = walk_one(r, bars, stop_s, target)
        if exit_m is None:
            continue
        raw_R, net_R, net_pct = cost_and_net(r.fill, R_s, r.holdout, r.half_entry, exit_price, why)
        rows.append(dict(book=book_name(s, rule), day=r.day, symbol=r.symbol, holdout=r.holdout,
                          wk=r.wk, fill=r.fill, level=r.level, stop_s=stop_s, R_s=R_s, base_R=r.R,
                          fill_min=r.fill_min, exit_m=exit_m, exit_price=exit_price, why=why,
                          raw_R=raw_R, net_R_own=net_R, net_pct=net_pct,
                          base_pct=r.base_pct, base_net_R=r.outcome_R))
    return pd.DataFrame(rows)


def preload_bars(fills):
    """{day: {symbol: bar df}} for every (day,symbol) in the base book, ONE bars_fills_1478.db
    connection, grouped per day (matches cell_1479's own query pattern)."""
    import sqlite3
    con = sqlite3.connect(f'file:{c1479.BARS_DB}?mode=ro', uri=True, timeout=120)
    out = {}
    days = sorted(fills.day.unique())
    for i, day in enumerate(days):
        syms = sorted(fills.loc[fills.day == day, 'symbol'].unique())
        out[day] = c1479.load_bars_1478(con, day, syms)
        if (i + 1) % 20 == 0 or i + 1 == len(days):
            log(f'preload_bars: {i + 1}/{len(days)} days loaded')
    con.close()
    return out


# ================================================================================================
# Scoring
# ================================================================================================

def score_book(df, holdout):
    """Every reported column for one (book, holdout): n, mean net R/pct, paired delta, day-clustered
    t, ex-top5%, fills/wk, exit mix, median R_s as % of price, winner-capped check."""
    d = df[df.holdout == holdout]
    if not len(d):
        return dict(n=0)
    delta_pct = d.net_pct - d.base_pct
    t_delta = c1445.day_clustered_t(delta_pct, d.day)
    cap_pct = WINNER_CAP_R * d.base_R / d.fill * 100.0
    winner_capped = float(np.minimum(d.net_pct, cap_pct).mean())
    r_pct_median = float((d.R_s / d.fill * 100.0).median())
    weeks = c1445.weeks_spanned(d.day)
    mix = (d.why.value_counts(normalize=True) * 100.0).round(1).to_dict()
    return dict(n=len(d), mean_net_R_own=float(d.net_R_own.mean()), mean_net_pct=float(d.net_pct.mean()),
                base_mean_pct=float(d.base_pct.mean()), delta_pct=float(delta_pct.mean()), t_delta=t_delta,
                ex_top5_pct=c1445.ex_top5_mean(d.net_pct), fills_wk=len(d) / weeks if weeks else np.nan,
                exit_mix=mix, r_pct_median=r_pct_median, winner_capped_pct=winner_capped,
                shippable=r_pct_median >= SHIP_MEDIAN_PCT)


def passes_bar_val(s_val, s_train):
    """The frozen VAL pass bar (PREREG "Pass bar" section), all six clauses."""
    if s_val.get('n', 0) == 0:
        return False
    same_sign_train = (np.sign(s_train.get('delta_pct', 0)) == np.sign(s_val['delta_pct'])
                        and s_train.get('t_delta', 0) is not None
                        and abs(s_train.get('t_delta', 0)) >= PASS_TRAIN_T)
    return bool(s_val['delta_pct'] >= PASS_DELTA_PCT and s_val['t_delta'] >= PASS_T
                and s_val['mean_net_pct'] >= PASS_OWN_PCT and s_val['ex_top5_pct'] > 0
                and s_val['fills_wk'] >= PASS_FILLS_WK and same_sign_train
                and s_val['winner_capped_pct'] > 0)


# ================================================================================================
# Dip-depth table (report-only)
# ================================================================================================

def dip_depth_table(fills):
    """Distribution of the lowest print below the level within 15 min (bps), and the base outcome_R
    by bucket. Prefers rebuild_1481_fills.csv's dip_low (limit-price convention); falls back to the
    minute bars' own lows in (fill_min, fill_min+15] for fills absent from that file."""
    dip = pd.read_csv(DIP_CSV_1481)
    dip = dip[dip.status == 'fill'][['day', 'symbol', 'level', 'dip_low']].drop_duplicates(['day', 'symbol'])
    m = fills[['day', 'symbol', 'level', 'fill_min', 'outcome_R']].merge(
        dip, on=['day', 'symbol'], suffixes=('', '_1481'), how='left')
    have = m.dip_low.notna()
    log(f'dip_depth_table: {int(have.sum())}/{len(m)} fills have a direct rebuild_1481_fills.csv dip_low '
        f'({int((~have).sum())} fall back to bars_fills_1478.db lows)')
    if (~have).any():
        con = None
        import sqlite3
        con = sqlite3.connect(f'file:{c1479.BARS_DB}?mode=ro', uri=True, timeout=120)
        missing = m[~have]
        for day, g in missing.groupby('day'):
            syms = sorted(g.symbol.unique())
            bars = c1479.load_bars_1478(con, day, syms)
            for idx, r in g.iterrows():
                b = bars.get(r.symbol)
                if b is None or not len(b):
                    continue
                lo = int(np.floor(r.fill_min))
                w = b[(b.m > lo) & (b.m <= lo + 15)]
                if len(w):
                    m.loc[idx, 'dip_low'] = float(w.l.min())
                    m.loc[idx, 'level_1481'] = r.level
        con.close()
    m['level_use'] = m['level_1481'].where(m['level_1481'].notna(), m['level'])
    m = m[m.dip_low.notna()].copy()
    m['dip_bps'] = (m.level_use - m.dip_low) / m.level_use * 1e4
    m['dip_bps'] = m['dip_bps'].clip(lower=0.0)
    rows = []
    for lo, hi in DIP_BUCKETS:
        b = m[(m.dip_bps > lo) & (m.dip_bps <= hi)] if hi != np.inf else m[m.dip_bps > lo]
        rows.append(dict(bucket=f'{int(lo)}-{"inf" if hi == np.inf else int(hi)} bps', n=len(b),
                          share_pct=100.0 * len(b) / len(m) if len(m) else np.nan,
                          base_outcome_R=float(b.outcome_R.mean()) if len(b) else np.nan))
    return pd.DataFrame(rows), len(m), len(fills)


# ================================================================================================
# Main
# ================================================================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args(argv)
    log('=== cell_1491 start ===' + (' (SMOKE)' if args.smoke else ''))
    fills = load_base()
    if args.smoke:
        fills = fills.sample(n=min(300, len(fills)), random_state=1491).reset_index(drop=True)
    log(f'walking {len(fills)} base fills x {len(STOPS)} stops x {len(RULES)} target rules on '
        f'bars_fills_1478.db...')
    bars_by_day = preload_bars(fills)

    books = []
    for s in STOPS:
        for rule in RULES:
            b = run_book(fills, bars_by_day, s, rule)
            log(f'{book_name(s, rule)}: {len(b)}/{len(fills)} fills walked')
            books.append(b)
    all_df = pd.concat(books, ignore_index=True) if books else pd.DataFrame()
    out_csv = OUT_CSV.replace('.csv', '_SMOKE.csv') if args.smoke else OUT_CSV
    all_df.to_csv(out_csv, index=False)
    log(f'wrote {out_csv} ({len(all_df)} rows)')

    scores = {}
    for s in STOPS:
        for rule in RULES:
            name = book_name(s, rule)
            b = all_df[all_df.book == name]
            scores[name] = {ho: score_book(b, ho) for ho in HOLDOUTS}
            sv, st = scores[name]['VAL'], scores[name]['TRAIN-H2']
            if sv.get('n', 0):
                pv = passes_bar_val(sv, st)
                scores[name]['VAL']['passes_bar'] = pv
                log(f'{name} VAL: n={sv["n"]} net%={sv["mean_net_pct"]:+.3f} base%={sv["base_mean_pct"]:+.3f} '
                    f'd%={sv["delta_pct"]:+.3f} t={sv["t_delta"]:.2f} fills/wk={sv["fills_wk"]:.2f} '
                    f'shippable={sv["shippable"]} PASS={pv}')

    dip_df, n_dip, n_base = dip_depth_table(fills)
    log(f'dip_depth_table: {n_dip}/{n_base} base fills covered')

    if not args.smoke:
        write_result_md(scores, dip_df, n_dip, n_base)
        log(f'wrote {RESULT_MD}')
    log('=== cell_1491 END ===')
    return all_df, scores, dip_df


def write_result_md(scores, dip_df, n_dip, n_base):
    lines = ['# RESULT -- cells 1,491-1,492: the shallow stop (bars-only re-walk, PREREG_1491.md)', '',
             'Base = the 9,911 `causal_arming_causal.csv` fills, entry unchanged; base comparison = '
             'cell_1478.build_outcome (amendment-slip-substituted). TEST does not exist in these files '
             '(never scored). Six books = 3 stops (level x (1-s), s in {0.25%,0.50%,0.75%}) x 2 targets '
             '(SCALP = 2*R_s; ASYM = the base target, unchanged reward).', '',
             '| book | holdout | n | mean_net_R_own | mean_net_%price | base_%price | delta_%price | '
             't_delta | ex_top5_% | fills/wk | R_s_%price_median | shippable | winner_capped_% | PASS |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, ho_scores in scores.items():
        for ho in HOLDOUTS:
            s = ho_scores[ho]
            if not s.get('n'):
                continue
            pv = s.get('passes_bar', '')
            lines.append(f"| {name} | {ho} | {s['n']} | {s['mean_net_R_own']:+.3f} | "
                         f"{s['mean_net_pct']:+.3f} | {s['base_mean_pct']:+.3f} | {s['delta_pct']:+.3f} | "
                         f"{s['t_delta']:.2f} | {s['ex_top5_pct']:+.3f} | {s['fills_wk']:.2f} | "
                         f"{s['r_pct_median']:.3f} | {s['shippable']} | {s['winner_capped_pct']:+.3f} | {pv} |")
    lines += ['', '## Dip-depth table (report-only, retest tape)',
              f'Coverage: {n_dip}/{n_base} base fills ({100 * n_dip / n_base:.1f}%).', '',
              '| bucket | n | share % | base outcome_R |', '|---|---|---|---|']
    for _, r in dip_df.iterrows():
        lines.append(f"| {r.bucket} | {r.n} | {r.share_pct:.1f} | {r.base_outcome_R:+.3f} |")
    with open(RESULT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
