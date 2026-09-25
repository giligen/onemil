"""Cell 1,442 — tape-triggered override (research/hod_entry/PREREG_WEEKEND.md, frozen 2026-09-25
18:35 UTC). Re-prices every signal of the E1 population (fill AND nofill rows of cell 1,427's
sip_rebuild_val.csv, TRAIN-H2 + VAL only, TEST not read) two ways, paired:

  (a) BROKER  — fill at the NBBO ask the instant of the first ROUND-LOT (>= 100 sh) print >= the
               trigger level+tick (Alpaca's own NBBO-filtered arming rule).
  (b) OVERRIDE — fill at the NBBO ask 300 ms AFTER the first print of ANY size >= trigger (the same
               triggering print E1 itself uses) — the "tape-triggered" 1-second-family override.

Both capped at limit = level * (1 + LIMIT_BPS) (sip_rebuild's own 15 bps cap); no fill if the
prevailing ask at the decision instant is above it.

Population and R recoverability: `stop` is not a column of sip_rebuild_val.csv, but for rows with
status == 'fill' it is exactly `fill - R` (cell_1430.load_base_fills' own convention). R is
therefore only recoverable for signals E1 itself filled; the mean-R / paired scoring is restricted
to that population, matching the PREREG's own wording ("Every E1 fill re-priced two ways"). The
no-fill signals ARE re-run under both rules (same trigger search, same cap) so the FILL RATE
denominator is the full attempted population (fill + nofill), not just E1's fills — this is what
lets "no fill-rate loss" mean something (a rule that only ever loses fills relative to E1, never
rescues a nofill, cannot show a fair rate without this population). Rescued nofills are counted
(fill rate) but carry no R (no stop to compute R against) — logged, not scored.

Exit-side cost, and the exit event itself (why/exit_price), are held FIXED at the base row's own
B0-parity outcome: cell 1,430 established the house convention that no per-minute spread data
outside the entry window exists in these inputs, so exit prices are never re-quoted. The entry-side
cost (half-spread at the NEW fill instant) IS re-quoted, since that is exactly what this cell tests.
Re-walking the bar path for the small (<=15 bps) target-level shift a different fill induces is out
of scope at normal priority; fill deltas are one to two orders of magnitude below typical R (stops
are consolidation lows), so a bar-level exit change is expected to be rare — not verified here.
"""
import gzip
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cell_1429 as c1429          # noqa: E402 — reuse tape cache loader + size-class report
import cell_1430 as c1430          # noqa: E402 — reuse day_clustered_t, STOP_WHY, STOP_SLIP_BP
import sip_rebuild as sr           # noqa: E402 — et_ns, sig_key, TICK, LIMIT_BPS, prevailing_quote, simulate_entry

DELAY_NS = int(0.3 * 1e9)   # OVERRIDE's 300 ms ask lookup
ROUND_LOT = 100


# --------------------------------------------------------------------------------------------
# Fill rules
# --------------------------------------------------------------------------------------------

def find_broker(trades, quotes, S_ns, trigger, limit):
    """(a) First ROUND-LOT (>= 100 sh) print >= trigger inside the break bar; fill at the
    prevailing ask at that instant, capped. None if no such print or ask > limit."""
    bb = trades[(trades.ts >= S_ns - 60 * 10 ** 9) & (trades.ts < S_ns)].sort_values('ts', kind='stable')
    hit = bb[(bb.price >= trigger - 1e-9) & (bb['size'] >= ROUND_LOT)]
    if not len(hit):
        return None
    t_hit = int(hit.ts.iloc[0])
    pq = sr.prevailing_quote(quotes, t_hit)
    if pq is None:
        return None
    bid, ask = pq
    if ask > limit + 1e-9:
        return None
    return dict(t_hit=t_hit, bid=bid, ask=ask)


def find_override(trades, quotes, S_ns, trigger, limit):
    """(b) First print of ANY size >= trigger (same print E1 itself uses); fill at the ask
    prevailing 300 ms AFTER that print's timestamp, capped. None if no such print or ask > limit."""
    bb = trades[(trades.ts >= S_ns - 60 * 10 ** 9) & (trades.ts < S_ns)].sort_values('ts', kind='stable')
    hit = bb[bb.price >= trigger - 1e-9]
    if not len(hit):
        return None
    t_hit = int(hit.ts.iloc[0])
    pq = sr.prevailing_quote(quotes, t_hit + DELAY_NS)
    if pq is None:
        return None
    bid, ask = pq
    if ask > limit + 1e-9:
        return None
    return dict(t_hit=t_hit, bid=bid, ask=ask)


# --------------------------------------------------------------------------------------------
# Population + per-signal repricing
# --------------------------------------------------------------------------------------------

def load_base_signals():
    """fill + nofill rows of sip_rebuild_val.csv, TRAIN-H2 (split TRAIN, half H2) + VAL."""
    p = os.path.join(HERE, 'sip_rebuild_val.csv')
    d = pd.read_csv(p, dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
    d = d[d.status.isin(['fill', 'nofill'])].copy()
    train_h2 = d[(d.split == 'TRAIN') & (d.half == 'H2')].reset_index(drop=True)
    val = d[d.split == 'VAL'].reset_index(drop=True)
    return train_h2, val


def reprice_signal(row, tapes_by_day):
    """One signal -> dict with fill_a/fill_b bools + (for base_status=='fill' rows only) paired
    net_R_{noslip,slip}_{a,b}. None if the tape is unavailable for this signal."""
    day_tapes = tapes_by_day.get(row.day)
    if not day_tapes:
        return None
    break_m = row.entry_m - 1
    key = sr.sig_key(row.symbol, break_m)
    if key not in day_tapes:
        return None
    trades, quotes = day_tapes[key]
    S_ns = sr.et_ns(row.day, row.entry_m * 60)
    trigger = round(row.level + sr.TICK, 6)
    limit = row.level * (1.0 + sr.LIMIT_BPS)

    res_a = find_broker(trades, quotes, S_ns, trigger, limit)
    res_b = find_override(trades, quotes, S_ns, trigger, limit)
    out = dict(day=row.day, symbol=row.symbol, entry_m=row.entry_m, level=row.level, wk=row.wk,
               base_status=row.status, base_net_R=row.net_R if row.status == 'fill' else np.nan,
               fill_a=res_a is not None, fill_b=res_b is not None)

    if row.status == 'fill' and np.isfinite(row.R) and row.R > 0:
        stop = row.fill - row.R
        e_base = sr.simulate_entry(trades, quotes, S_ns, row.level, stop, limit_bps=sr.LIMIT_BPS)
        if e_base['status'] != 'fill' or not np.isclose(e_base['fill'], row.fill, atol=0.005):
            print(f"[cell_1442] WARNING: base-fill reproduction mismatch for {row.day} {row.symbol} "
                  f"entry_m={row.entry_m} (tape/CSV drift) — dropped from paired scoring", file=sys.stderr)
            return out
        exit_side_term = row.cost_R * row.R - e_base['half_spread']   # held fixed, house convention
        for letter, res in (('a', res_a), ('b', res_b)):
            if res is None:
                continue
            ask, bid = res['ask'], res['bid']
            R_new = ask - stop
            if R_new <= 0:
                print(f"[cell_1442] WARNING: degenerate R_new<=0 for {row.day} {row.symbol} rule {letter} "
                      f"— dropped", file=sys.stderr)
                continue
            raw_R = (row.exit_price - ask) / R_new
            half_new = 0.5 * (ask - bid)
            cost_new = (half_new + exit_side_term) / R_new
            net_noslip = raw_R - cost_new
            slip_R = c1430.STOP_SLIP_BP * row.exit_price / R_new if row.why in c1430.STOP_WHY else 0.0
            out[f'net_R_noslip_{letter}'] = net_noslip
            out[f'net_R_slip_{letter}'] = net_noslip - slip_R
    return out


def _size_class_cross(base_fill, tapes_by_day):
    """Per-signal round/odd-lot classification of the any-size trigger print (c1429's own search,
    identical to the print OVERRIDE and E1 both use), crossed with base net_R and each rule's own
    net_R_slip (report-only, PREREG line 48)."""
    f = base_fill.copy()
    f['trigger_size'] = [c1429.trigger_print_size(r, tapes_by_day) for r in f.itertuples()]
    f['size_class'] = np.where(f.trigger_size.isna(), 'unknown',
                                np.where(f.trigger_size >= ROUND_LOT, 'round_lot', 'odd_lot'))
    f = f[f.size_class != 'unknown']
    return f.groupby('size_class').agg(n=('base_net_R', 'size'), mean_base_R=('base_net_R', 'mean'),
                                        mean_R_slip_a=('net_R_slip_a', 'mean'),
                                        mean_R_slip_b=('net_R_slip_b', 'mean'))


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------

def score_split(sigs, tapes_by_day):
    """Returns (df, summary dict) for one split."""
    rows = [reprice_signal(r, tapes_by_day) for r in sigs.itertuples()]
    n_lost = sum(r is None for r in rows)
    if n_lost:
        print(f'[cell_1442] WARNING: {n_lost}/{len(rows)} signals had no cached tape — dropped', file=sys.stderr)
    df = pd.DataFrame([r for r in rows if r is not None])

    D = len(df)
    baseline_fill_rate = (df.base_status == 'fill').mean() if D else float('nan')
    fill_rate_a = df.fill_a.mean() if D else float('nan')
    fill_rate_b = df.fill_b.mean() if D else float('nan')

    base_fill = df[df.base_status == 'fill']
    resc = df[df.base_status == 'nofill']
    rescued_a, rescued_b, n_nofill = int(resc.fill_a.sum()), int(resc.fill_b.sum()), len(resc)

    summ = {}
    for letter in ('a', 'b'):
        sub = base_fill[base_fill[f'fill_{letter}'] & base_fill[f'net_R_slip_{letter}'].notna()]
        summ[letter] = dict(n=len(sub),
                             mean_R_noslip=sub[f'net_R_noslip_{letter}'].mean() if len(sub) else float('nan'),
                             mean_R_slip=sub[f'net_R_slip_{letter}'].mean() if len(sub) else float('nan'))

    common = base_fill[base_fill.fill_a & base_fill.fill_b
                        & base_fill.net_R_slip_a.notna() & base_fill.net_R_slip_b.notna()]
    diff = common.net_R_slip_b - common.net_R_slip_a
    t, ndays = c1430.day_clustered_t(diff, common.day) if len(common) else (float('nan'), 0)
    dR_ba = float(diff.mean()) if len(diff) else float('nan')

    # odd-lot-led vs round-lot-led crosses (trigger print = the same any-size print E1/override use);
    # crossed against each rule's own mean R, not just the base-row net_R
    size_rep = _size_class_cross(base_fill, tapes_by_day) if len(base_fill) else pd.DataFrame()

    return df, dict(D=D, baseline_fill_rate=baseline_fill_rate, fill_rate_a=fill_rate_a,
                     fill_rate_b=fill_rate_b, rescued_a=rescued_a, rescued_b=rescued_b,
                     n_nofill=n_nofill, a=summ['a'], b=summ['b'], dR_ba=dR_ba, t_ba=t, ndays=ndays,
                     n_common=len(common), size_rep=size_rep)


def run():
    train_h2, val = load_base_signals()
    all_days = pd.concat([train_h2.day, val.day]).unique()
    tapes_by_day = {d: c1429._load_day_tapes(d) for d in all_days}

    results = {}
    for split_name, sigs in (('TRAIN-H2', train_h2), ('VAL', val)):
        df, s = score_split(sigs, tapes_by_day)
        results[split_name] = s
        print(f"[cell_1442] {split_name} D={s['D']} baseline_fill_rate={s['baseline_fill_rate']:.3f} "
              f"fill_rate_a={s['fill_rate_a']:.3f} fill_rate_b={s['fill_rate_b']:.3f} "
              f"rescued(nofill->fill) a={s['rescued_a']}/{s['n_nofill']} b={s['rescued_b']}/{s['n_nofill']}")
        print(f"[cell_1442] {split_name} (a) BROKER n={s['a']['n']} mean_R_slip={s['a']['mean_R_slip']:+.4f} "
              f"mean_R_noslip={s['a']['mean_R_noslip']:+.4f}")
        print(f"[cell_1442] {split_name} (b) OVERRIDE n={s['b']['n']} mean_R_slip={s['b']['mean_R_slip']:+.4f} "
              f"mean_R_noslip={s['b']['mean_R_noslip']:+.4f}")
        print(f"[cell_1442] {split_name} (b)-(a) paired n={s['n_common']} dR={s['dR_ba']:+.4f} "
              f"t={s['t_ba']:.2f} (n_days={s['ndays']})")
        if len(s['size_rep']):
            print(f"[cell_1442] {split_name} trigger-print size class (report-only):\n{s['size_rep']}")

    pass_ = all(np.isfinite(results[sp]['dR_ba']) and results[sp]['dR_ba'] >= 0.03
                and results[sp]['fill_rate_b'] >= results[sp]['baseline_fill_rate'] - 1e-9
                for sp in ('TRAIN-H2', 'VAL'))
    print(f"[cell_1442] -> {'PASS' if pass_ else 'FAIL'}")
    return results, pass_


if __name__ == '__main__':
    run()
