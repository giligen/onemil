#!/usr/bin/env python3
"""Cell 1,479 -- research/hod_entry/PREREG_1478.md (FROZEN 2026-09-26) + its amendment.

PYRAMID overlay on the 9,911-fill base book (cell 1,438, status=='fill', TRAIN-H2+VAL, TEST sealed
and never read). 1/3 of the risk goes on at the break (the base fill). If, before the base trade's
own stop, a later bar's HIGH reaches fill+1R0 (R0 = fill-stop, the ORIGINAL risk unit), add the
remaining 2/3 at fill+1R0 (a fresh ask fill -- charge the fill's measured half-spread again) and
move the WHOLE position's stop to the original fill (breakeven on the first third, -1R0 on the
added two-thirds if stopped there). Target stays 2R0 from the ORIGINAL fill for the combined
position. 15:55 -> exit at that bar's open. Everything is reported in units of the ORIGINAL risk
(R0), never the post-add position's own risk.

Bar source (amendment item 1, applied here too so no cache.db/bars_sip.db store-identity leak can
enter this cell): research/hod_entry/bars_fills_1478.db ONLY -- ONE fresh Alpaca SIP 1-minute store
fetched independently of which store served the base book's own bars.

Cost model (this cell's own choice, documented because the PREREG does not spell out the exact
algebra -- see PREREG's "Interactive Sessions" rule: decide testable ambiguities with a pre-committed
rule, disclose it, never silently guess):
  * half_entry (HE) and exit_half (EH) are the REAL per-share dollar half-spreads already recovered
    for the base trade by cell_1445.corrected_cost() -- HE returned directly; EH recovered by the
    exact algebraic identity cost_R*R = HE + EH + SLIP_BP*exit_price (the same identity that built
    net_R_costfix in the first place), i.e. EH = cost_R*R - HE - SLIP_BP*exit_price. Both are
    per-SHARE price differences, not proxies -- no 50/50 approximation is needed because this run
    has direct access to cell_1445.py.
  * "Charge HE again" for the add is applied on the ADD'S OWN SHARES (2/3 N), not as a flat
    unweighted second charge on the whole position: total entry-side dollar friction for buying N
    shares via two orders (1/3 N then 2/3 N), each paying the SAME per-share HE, is HE*(1/3N) +
    HE*(2/3N) = HE*N -- identical to what a single N-share entry would cost. This is the literal,
    economically exact reading of "charge it again" (a second real crossing DOES happen, and DOES
    cost money on its own shares) without inventing an extra punitive multiplier the PREREG never
    specified. Symmetrically, EH and the flat SLIP_BP impact term apply to whatever share count is
    open at the FINAL exit (1/3 N if never added, N if added), and the stop-limit slip below applies
    the same way. Net effect: cost_R is IDENTICAL to a plain single trade's cost_R once added (no
    extra "pyramid tax"), and is exactly 1/3 of it if the add was never reached and the position
    exited at 1/3 size -- both derived from one consistent share-weighted accounting, not two
    different ad hoc formulas.
  * Stop-limit exit slip (PREREG_1478 "Base book, cost, exits": "the verified 20 bps stop-limit exit
    slip on stop exits (cell 1,463 holdout means 2.9 / 3.2 bps on filled stops; the 12% no-fill tail
    at its measured mean) -- this is now the standard exit for every HOD number"). Applied ONLY to
    this cell's own NEW stop-type exits ('stop' pre-add, 'stop_be' post-add) since bars_fills_1478.db
    has no tick/quote data to re-simulate a fresh stop-limit fill at an arbitrary NEW stop price (the
    moved-to-breakeven stop was never in cell 1,463's tape cache). The probability-weighted blend
    (RESULT_1463_unbiased.md, stop-limit-20bps row) is applied as a flat expected-value bps constant
    per holdout, weighted the same way as EH above:
      TRAIN-H2: 272/316 * 2.9 + 44/316 * 93.9 = 15.57 bps
      VAL:      286/320 * 3.2 + 34/320 * 75.8 = 10.91 bps
"""
import os
import sys
import sqlite3
import time
import argparse

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sip_rebuild as sr           # noqa: E402  (EOD_M, TICK, SLIP_BP, TARGET_R, walk_path)
import cell_1445 as c1445          # noqa: E402  (load_base_book, load_nbbo_lookup, corrected_cost, day_clustered_t, ex_top5_mean)

BARS_DB = os.path.join(HERE, 'bars_fills_1478.db')
OUT_CSV = os.path.join(HERE, 'cell_1479_fills.csv')
LOG_PATH = os.path.join(HERE, 'cell_1479.log')
ET = ZoneInfo('America/New_York')
OPEN_M = sr.OPEN_M           # 570 = 09:30 ET
EOD_M = sr.EOD_M             # 955 = 15:55 ET

# RESULT_1463_unbiased.md, stop-limit-20bps row (fill n / no-fill n out of the holdout's tape sample)
STOP_LIMIT_1463 = {
    'TRAIN-H2': dict(n_fill=272, n_nofill=44, mean_bps=2.9, nofill_mean_bps=93.9),
    'VAL':      dict(n_fill=286, n_nofill=34, mean_bps=3.2, nofill_mean_bps=75.8),
}


def log(msg):
    """Verbose progress, flushed immediately and appended to LOG_PATH (print() is buffered under nohup)."""
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def blended_stop_limit_bps(holdout):
    """Probability-weighted mean stop-limit-20bps slip for `holdout` ('TRAIN-H2' or 'VAL'), from
    RESULT_1463_unbiased.md's own fill/no-fill counts -- the PREREG's frozen exit standard."""
    d = STOP_LIMIT_1463[holdout]
    n = d['n_fill'] + d['n_nofill']
    return (d['n_fill'] * d['mean_bps'] + d['n_nofill'] * d['nofill_mean_bps']) / n


def load_base():
    """The 9,911-fill base book (cell_1445.load_base_book: causal_arming_causal.csv, status=='fill',
    TRAIN-H2+VAL, TEST excluded) with HE (half_entry) and EH (exit_half) recovered exactly, and the
    holdout label used throughout this module ('TRAIN-H2' matches load_base_book's own convention)."""
    fills = c1445.load_base_book()
    nbbo_lookup = c1445.load_nbbo_lookup()
    half_entry, net_R_costfix, nbbo_fallback = c1445.corrected_cost(fills, nbbo_lookup)
    fills = fills.copy()
    fills['half_entry'] = half_entry
    fills['exit_half'] = fills.cost_R.to_numpy() * fills.R.to_numpy() - half_entry \
        - sr.SLIP_BP * fills.exit_price.to_numpy()
    n_neg = int((fills.exit_half < 0).sum())
    if n_neg:
        log(f'load_base: {n_neg}/{len(fills)} rows have a negative recovered exit_half '
            f'(cell_1445.py: "half_entry can be negative for a small ..." -- same identity, not clipped)')
    return fills


def _to_et_minute(t_series):
    """ISO-UTC timestamp strings -> ET minute-of-day ints (America/New_York wall clock, DST-exact)."""
    ts = pd.to_datetime(t_series, utc=True).dt.tz_convert(ET)
    return ts.dt.hour * 60 + ts.dt.minute


def load_bars_1478(con, day, syms):
    """{symbol: RTH minute bars (m,o,h,l,c,v), m in [OPEN_M, EOD_M]} for one day, from the SINGLE
    fresh bars_fills_1478.db store (amendment item 1: no cache.db / bars_sip.db mixing here)."""
    if not syms:
        return {}
    q = (f"select symbol, t, o, h, l, c, v from bars where day=? and symbol in "
         f"({','.join('?' * len(syms))})")
    df = pd.read_sql(q, con, params=[day] + list(syms))
    if not len(df):
        return {}
    df['m'] = _to_et_minute(df['t'])
    df = df[(df.m >= OPEN_M) & (df.m <= EOD_M)].sort_values(['symbol', 'm'], kind='stable')
    df = df.drop_duplicates(['symbol', 'm'], keep='first')
    return {s: g[['m', 'o', 'h', 'l', 'c', 'v']].reset_index(drop=True) for s, g in df.groupby('symbol')}


def pyramid_walk(fill, stop, target, path):
    """Walk `path` (bars with m > the break bar, m <= EOD_M, sorted) applying the pyramid rule.
    Returns dict(exit_m, exit_price, why, added). `why` in {'stop','stop_be','target','eod',
    'eod_fallback'}; 'stop_be' means the combined position was stopped at breakeven AFTER the add.
    raw_R (ORIGINAL-R units) is computed by the caller from (exit_price, added), matching the
    module docstring's algebra: pre-add combined_R=(X-fill)/R0 * (1/3); post-add combined_R=
    (X-fill)/R0 - 2/3."""
    R0 = fill - stop
    add_th = fill + R0
    eps = 1e-9
    added = False
    cur_stop = stop
    for row in path.itertuples():
        m, o, h, l, c = row.m, row.o, row.h, row.l, row.c
        if m >= EOD_M:
            return dict(exit_m=int(m), exit_price=float(o), why='eod', added=added)
        if l <= cur_stop + eps:
            x = float(o) if o <= cur_stop else float(cur_stop)
            return dict(exit_m=int(m), exit_price=x, why=('stop_be' if added else 'stop'), added=added)
        if not added and h >= add_th - eps:
            added, cur_stop = True, fill                      # move stop to original fill (breakeven)
            if l <= cur_stop + eps:                            # same bar can re-trip the new stop
                x = float(o) if o <= cur_stop else float(cur_stop)
                return dict(exit_m=int(m), exit_price=x, why='stop_be', added=True)
            if h >= target - eps:
                return dict(exit_m=int(m), exit_price=float(target), why='target', added=True)
            continue
        if added and h >= target - eps:
            return dict(exit_m=int(m), exit_price=float(target), why='target', added=True)
    last = path.iloc[-1]
    return dict(exit_m=int(last.m), exit_price=float(last.c), why='eod_fallback', added=added)


def raw_R_original_units(fill, stop, mine):
    """(exit_price - fill)/R0 scaled per the pyramid's own share-weighted algebra (module docstring)."""
    R0 = fill - stop
    x = mine['exit_price']
    return (x - fill) / R0 * (1 / 3) if not mine['added'] else (x - fill) / R0 - 2 / 3


def cost_1479(row, mine):
    """(cost_R, net_R) of `mine`'s exit, in ORIGINAL-R units, per the module docstring's share-weighted
    cost model. `row` carries R, half_entry, exit_half, split/holdout from the base book."""
    R0 = row.R
    he, eh = row.half_entry, row.exit_half
    w = 1.0 if mine['added'] else (1 / 3)
    entry_cost = he * (1 / 3) + (he * (2 / 3) if mine['added'] else 0.0)     # HE*(1/3N)+HE*(2/3N) if added
    exit_cost = (eh + sr.SLIP_BP * mine['exit_price']) * w
    stop_type = mine['why'] in ('stop', 'stop_be')
    slip_extra = blended_stop_limit_bps(row.holdout) * 1e-4 * mine['exit_price'] * w if stop_type else 0.0
    cost_R = (entry_cost + exit_cost) / R0 + slip_extra / R0
    raw_R = raw_R_original_units(row.fill, row.stop, mine)
    return cost_R, raw_R - cost_R


def run_1479(fills, con, validate_n=300):
    """One row per base fill: the pyramid re-walk. Also runs a bar-loading self-check (re-walk the
    PLAIN base trade with sr.walk_path on the SAME bars_fills_1478.db bars and compare to the base
    CSV's own exit_m/exit_price/why -- validates load_bars_1478+walk_path use, NOT this cell's own
    pyramid logic, which has its own unit tests)."""
    rows = []
    n_chk = n_ok = n_bad = 0
    for day, g in fills.groupby('day'):
        syms = sorted(g.symbol.unique())
        bars = load_bars_1478(con, day, syms)
        for r in g.itertuples():
            m_break = int(np.floor(r.fill_min))
            b = bars.get(r.symbol)
            if r.why == 'stop_bar' or b is None or not len(b):
                mine = dict(exit_m=m_break, exit_price=r.stop, why='stop_bar', added=False)
            else:
                path = b[(b.m > m_break) & (b.m <= EOD_M)]
                if not len(path):
                    mine = dict(exit_m=m_break, exit_price=r.fill, why='no_path', added=False)
                else:
                    target = r.fill + sr.TARGET_R * r.R
                    mine = pyramid_walk(r.fill, r.stop, target, path)
                    if n_chk < validate_n:
                        n_chk += 1
                        bw_m, bw_px, bw_why = sr.walk_path(r.fill, r.stop, target, path)
                        ok = bw_why == r.why and abs(bw_px - r.exit_price) < 0.02
                        n_ok += int(ok)
                        n_bad += int(not ok)
            cost_R, net_R = cost_1479(r, mine)
            raw_R = raw_R_original_units(r.fill, r.stop, mine)
            rows.append(dict(day=r.day, symbol=r.symbol, split=r.split, holdout=r.holdout, wk=r.wk,
                              fill_min=r.fill_min, base_why=r.why, base_net_R=r.net_R,
                              c1479_exit_m=mine['exit_m'], c1479_exit_price=mine['exit_price'],
                              c1479_why=mine['why'], added=mine['added'],
                              c1479_raw_R=raw_R, c1479_cost_R=cost_R, c1479_net_R=net_R,
                              delta_R=net_R - r.net_R))
    log(f'bar-walk self-check (plain re-walk of the BASE trade on bars_fills_1478.db, n={n_chk}): '
        f'{n_ok} match / {n_bad} mismatch ({100 * n_ok / max(n_chk, 1):.1f}%)')
    if n_chk and n_bad / n_chk > 0.02:
        log(f'WARNING self-check mismatch rate {n_bad / n_chk:.1%} exceeds 2% -- bars_fills_1478.db '
            f'may disagree with the base book\'s own bar source; treat 1,479 numbers as provisional')
    return pd.DataFrame(rows)


def score_holdout(df, holdout):
    """n, mean net R, ΔR vs base, day-clustered t of ΔR, ex-top-5 % of the pyramid book, share added."""
    d = df[df.holdout == holdout]
    if not len(d):
        return dict(n=0)
    t_delta = c1445.day_clustered_t(d.delta_R, d.day)
    return dict(n=len(d), mean_net_R=float(d.c1479_net_R.mean()), delta_R=float(d.delta_R.mean()),
                base_mean=float(d.base_net_R.mean()), t_delta=t_delta,
                extop5=float(c1445.ex_top5_mean(d.c1479_net_R)), share_added=float(d.added.mean()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args(argv)
    log('=== cell_1479 start ===' + (' (SMOKE)' if args.smoke else ''))
    fills = load_base()
    if args.smoke:
        fills = fills.head(80).copy()
    con = sqlite3.connect(f'file:{BARS_DB}?mode=ro', uri=True, timeout=120)
    log(f'walking {len(fills)} fills on bars_fills_1478.db...')
    out = run_1479(fills, con)
    out.to_csv(OUT_CSV, index=False)
    log(f'wrote {OUT_CSV} ({len(out)} rows)')
    for sp in ('TRAIN-H2', 'VAL'):
        s = score_holdout(out, sp)
        if s.get('n'):
            log(f'{sp}: n={s["n"]} mean_net_R={s["mean_net_R"]:+.4f} base_mean={s["base_mean"]:+.4f} '
                f'delta_R={s["delta_R"]:+.4f} t={s["t_delta"]:.2f} extop5={s["extop5"]:+.4f} '
                f'share_added={s["share_added"]:.3f}')
    log('=== cell_1479 END ===')
    return out


if __name__ == '__main__':
    main()
