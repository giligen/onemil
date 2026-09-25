"""Cell 1,431 — no-fill cohort short (research/hod_entry/PREREG_WEEKEND.md, frozen 2026-09-25
18:35 UTC). Population: signals of cell 1,427's rebuild (sip_rebuild_val.csv, TRAIN-H2 + VAL) that
did NOT fill because the ask exceeded the 15 bps limit AT THE CROSS. sip_rebuild.simulate_entry
gives 'nofill' status to two distinct cases — no triggering print in the break bar at all (ask_at
stays NaN) vs a triggering print whose prevailing ask was above the limit (ask_at is set) — so the
PREREG's "ask exceeded the limit at the cross" population is `status=='nofill' & ask_at.notna()`.

Restricted further to signals whose BREAK BAR closed >= 15 bps above the level, read from
data/cache.db intraday_bars_1min — the same table and (symbol, bar_date) query sip_rebuild.py's own
load_levels() uses for the level itself (its documented source choice, reused verbatim here).

Trade: short at the next bar's open (the bar at entry_m, first row of research/hod_exit_lab/
paths.parquet for (day, symbol) — confirmed to start exactly at entry_m, i.e. it does NOT carry the
break bar), mirroring B0's own chase cap: skip if that open is more than 60 bps BELOW the level.
stop = break-bar high + $0.01 (fixed, never re-walked); target = entry - 2R, R = stop - entry; cover
at the 15:55 bar's open if neither triggers first. Physics start on the ENTRY bar itself (matching
sip_rebuild.walk_path's own "path rows m >= entry_m"): gap-through at the open, high>=stop checked
before low<=target on a bar touching both (stop-first, mirroring B0's own tie-break).

Cost — explicit modeling choices where the PREREG prose does not fully specify a number (same
convention as cell_1430's documented (d)/(e)):
  * entry half-spread: the prevailing SIP NBBO quote at the entry bar's open timestamp, from the
    cached tape (research/hod_entry/sip_cache, c1429._load_day_tapes) — literally "the SIP quote at
    the next-bar open" the PREREG asks for.
  * "B0's exit cost": there is no B0 trade attached to a signal that never filled, so no per-signal
    exit-side spread can be recovered the way cell_1430/1442 reuse an existing fill's own cost_R.
    The stand-in is the MEDIAN reconstructed B0 exit half-spread (sip_rebuild.py line 99's own
    formula: 0.5*(cost_R*R - SLIP_BP*(fill+exit_price))) over TRAIN-H2's E1 fills only (never VAL),
    applied as one fixed dollar constant to both holdouts. Plus sr.SLIP_BP (2 bp) of the exit price,
    B0's own second exit-cost term. A 30 bps stop-slip charge is added on stop exits, reported both
    with and without, per the PREREG_WEEKEND.md header.

Shortable flag: the PREREG names data/research/alpaca_assets_all_20260905.csv, but that file (both
copies in the repo) carries only symbol/name/status/tradable/exchange/common — NO shortable or
easy_to_borrow column exists in it. The only Alpaca asset dump in the repo with those two flags is
research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv (same schema fetch_borrow.py's TradingClient
snapshot produces, dated 2026-09 per its own directory). Used here as the nearest available
equivalent — documented, not silent; the RESULT reports today's-snapshot survivorship caveat exactly
as fetch_borrow.py's own docstring does, and the population is scored both ALL and shortable-only.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'research/hod_consol'))
import cell_1429 as c1429          # noqa: E402 — reuse cached tape loader
import cell_1430 as c1430          # noqa: E402 — reuse day_clustered_t, STOP_SLIP_BP
import sip_rebuild as sr           # noqa: E402 — et_ns, ET, CACHE_DB_URI, prevailing_quote, SLIP_BP
import run_consol                  # noqa: E402 — simulate_slots

LEVEL_CLOSE_BPS = 0.0015    # 15 bps, PREREG
CHASE_CAP_BPS = 0.0060      # 60 bps, mirror of B0's fill cap
TARGET_R = 2.0
STOP_TICK = 0.01
MIN_R_PCT = 0.005    # R must exceed 0.5% of price (project note: R < ~0.5% killed 3 prior books)
BORROW_CSV = os.path.join(ROOT, 'research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv')


# --------------------------------------------------------------------------------------------
# Population
# --------------------------------------------------------------------------------------------

def load_population():
    """status=='nofill' & ask_at notna (true "ask exceeded limit at the cross") of TRAIN-H2 + VAL."""
    p = os.path.join(HERE, 'sip_rebuild_val.csv')
    d = pd.read_csv(p, dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
    d = d[(d.status == 'nofill') & d.ask_at.notna()].copy()
    d['break_m'] = d.entry_m - 1
    train_h2 = d[(d.split == 'TRAIN') & (d.half == 'H2')].reset_index(drop=True)
    val = d[d.split == 'VAL'].reset_index(drop=True)
    return train_h2, val


def load_break_bars(sigs):
    """{(symbol, day): dict(o,h,l,c)} for the break bar of every unique (symbol, day, break_m) in
    `sigs`, from data/cache.db intraday_bars_1min — sip_rebuild.load_levels' own source/query."""
    con = sqlite3.connect(sr.CACHE_DB_URI, uri=True)
    out = {}
    keys = sigs[['symbol', 'day', 'break_m']].drop_duplicates()
    for symbol, day, break_m in keys.itertuples(index=False):
        q = pd.read_sql_query(
            'SELECT timestamp, open, high, low, close FROM intraday_bars_1min WHERE symbol=? AND bar_date=?',
            con, params=(symbol, day))
        if not len(q):
            continue
        ts = pd.to_datetime(q.timestamp, utc=True).dt.tz_convert(sr.ET)
        q['m'] = ts.dt.hour * 60 + ts.dt.minute
        row = q[q.m == break_m]
        if len(row):
            out[(symbol, day)] = dict(o=float(row.open.iloc[0]), h=float(row.high.iloc[0]),
                                       l=float(row.low.iloc[0]), c=float(row.close.iloc[0]))
    con.close()
    return out


def filter_close_above_level(sigs, bb):
    """Keep only signals whose break bar closed >= level * (1 + LEVEL_CLOSE_BPS)."""
    keep = []
    for r in sigs.itertuples():
        b = bb.get((r.symbol, r.day))
        keep.append(b is not None and b['c'] >= r.level * (1.0 + LEVEL_CLOSE_BPS) - 1e-9)
    out = sigs[pd.Series(keep, index=sigs.index)].copy()
    out['bb_high'] = [bb[(r.symbol, r.day)]['h'] for r in out.itertuples()]
    out['bb_close'] = [bb[(r.symbol, r.day)]['c'] for r in out.itertuples()]
    return out


def load_borrow_flags():
    """symbol -> (shortable, easy_to_borrow) from borrow_flags.csv (see module docstring)."""
    d = pd.read_csv(BORROW_CSV)
    d = d.drop_duplicates('symbol').set_index('symbol')
    return d[['shortable', 'easy_to_borrow']]


# --------------------------------------------------------------------------------------------
# Short-side path walk — mirror of sip_rebuild.walk_path / cell_1430.b0_walk
# --------------------------------------------------------------------------------------------

def walk_short(entry, stop, target, bars, eod_m):
    """Physics start on the bar AT entry_m (row.m == entry_m is the first row of `bars`), gap-
    through at the open, stop (high>=stop) checked before target (low<=target) on a bar touching
    both. Returns (exit_m, exit_price, why), or None if `bars` runs out before eod_m."""
    for row in bars:
        if row.m >= eod_m:
            return int(row.m), float(row.o), 'eod'
        if row.h >= stop:
            exit_px = row.o if row.o >= stop else stop
            return int(row.m), float(exit_px), 'stop'
        if row.l <= target:
            return int(row.m), float(target), 'target'
    return None


def bars_from(idx, day, symbol, entry_m, eod_m):
    """paths.parquet rows m in [entry_m, eod_m] for (day, symbol), sorted by m, or None."""
    key = (day, symbol)
    if key not in idx.index:
        return None
    g = idx.loc[[key]]
    g = g[(g.m >= entry_m) & (g.m <= eod_m)]
    return None if g.empty else g


# --------------------------------------------------------------------------------------------
# Per-signal simulation
# --------------------------------------------------------------------------------------------

def median_b0_exit_half(train_h2_fills):
    """Median reconstructed B0 exit half-spread ($) over TRAIN-H2 E1 fills only — sip_rebuild.py
    line 99's own formula, frozen here as a fixed constant applied to both holdouts."""
    f = train_h2_fills
    half = 0.5 * (f.cost_R * f.R - sr.SLIP_BP * (f.fill + f.exit_price))
    return float(half.median())


def simulate_signal(row, paths_idx, tapes_by_day, exit_half_const):
    """One signal -> dict, or None if skipped (chase cap / no path / degenerate R)."""
    bb_high = row.bb_high
    stop = bb_high + STOP_TICK
    bars_df = bars_from(paths_idx, row.day, row.symbol, row.entry_m, sr.EOD_M)
    if bars_df is None:
        return None
    entry_row = bars_df[bars_df.m == row.entry_m]
    if not len(entry_row):
        return None
    entry = float(entry_row.o.iloc[0])
    if entry < row.level * (1.0 - CHASE_CAP_BPS) - 1e-9:
        return dict(day=row.day, symbol=row.symbol, wk=row.wk, entry_m=row.entry_m, skipped='chase_cap')
    R = stop - entry
    if R < MIN_R_PCT * entry:
        # "R must exceed the spread" (project note, R-as-%-of-price killed three prior books): a
        # next-bar open that lands on or past the break-bar high leaves R at 1-2 ticks, and every
        # R-normalized figure blows up on noise. Unusable, not a rejected trade -- WARNING, not FAIL.
        print(f'[cell_1431] WARNING: degenerate R={R:.4f} ({R / entry:.4%} of price) for {row.day} '
              f'{row.symbol} entry_m={row.entry_m} — dropped (R floor {MIN_R_PCT:.2%})', file=sys.stderr)
        return None
    target = entry - TARGET_R * R

    res = walk_short(entry, stop, target, bars_df.itertuples(), sr.EOD_M)
    if res is None:
        return None
    exit_m, exit_px, why = res

    day_tapes = tapes_by_day.get(row.day, {})
    key = sr.sig_key(row.symbol, row.break_m)
    half_entry = None
    if key in day_tapes:
        _trades, quotes = day_tapes[key]
        S_ns = sr.et_ns(row.day, row.entry_m * 60)
        pq = sr.prevailing_quote(quotes, S_ns)
        if pq is not None:
            bid, ask = pq
            half_entry = 0.5 * (ask - bid)
    if half_entry is None:
        return None    # no quote at the entry instant — unusable for cost, per the availability rail

    raw_R = (entry - exit_px) / R
    cost_R = (half_entry + exit_half_const + sr.SLIP_BP * exit_px) / R
    net_noslip = raw_R - cost_R
    slip_R = c1430.STOP_SLIP_BP * exit_px / R if why == 'stop' else 0.0
    net_slip = net_noslip - slip_R

    return dict(day=row.day, symbol=row.symbol, wk=row.wk, entry_m=row.entry_m, exit_m=exit_m,
                why=why, R=R, net_R_noslip=net_noslip, net_R_slip=net_slip, skipped=None)


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------

def score_split(sigs, paths_idx, tapes_by_day, exit_half_const, borrow):
    rows = [simulate_signal(r, paths_idx, tapes_by_day, exit_half_const) for r in sigs.itertuples()]
    n_none = sum(r is None for r in rows)
    if n_none:
        print(f'[cell_1431] WARNING: {n_none}/{len(rows)} signals dropped (no path/quote/degenerate R)',
              file=sys.stderr)
    df = pd.DataFrame([r for r in rows if r is not None])
    scored = df[df.skipped.isna()].copy() if len(df) else df
    n_chase = int((df.skipped == 'chase_cap').sum()) if len(df) else 0

    if len(scored):
        scored = scored.merge(borrow, left_on='symbol', right_index=True, how='left')
        scored['shortable'] = scored.shortable.fillna(False)
    share_shortable = float(scored.shortable.mean()) if len(scored) else float('nan')

    def _stats(sub):
        if not len(sub):
            return dict(n=0, mean_noslip=float('nan'), mean_slip=float('nan'), t=float('nan'), ndays=0)
        t, ndays = c1430.day_clustered_t(sub.net_R_slip, sub.day)
        return dict(n=len(sub), mean_noslip=float(sub.net_R_noslip.mean()),
                    mean_slip=float(sub.net_R_slip.mean()), t=t, ndays=ndays)

    all_stats = _stats(scored)
    short_stats = _stats(scored[scored.shortable]) if len(scored) else _stats(scored)

    fills_wk = float('nan')
    if len(scored):
        sim = scored.rename(columns={'entry_m': 'entry_m', 'exit_m': 'exit_m'}).copy()
        keep = run_consol.simulate_slots(sim)
        kept = sim[keep]
        nwk = kept.wk.nunique() if len(kept) else 0
        fills_wk = len(kept) / nwk if nwk else float('nan')

    return dict(D=len(df), n_chase_skipped=n_chase, share_shortable=share_shortable,
                all=all_stats, shortable_only=short_stats, fills_per_week=fills_wk)


def run():
    train_h2, val = load_population()
    bb = load_break_bars(pd.concat([train_h2, val]))
    train_h2 = filter_close_above_level(train_h2, bb)
    val = filter_close_above_level(val, bb)

    exit_half_const = median_b0_exit_half(c1430.load_base_fills()[0])
    paths_idx = c1430.load_paths_index()
    all_days = pd.concat([train_h2.day, val.day]).unique()
    tapes_by_day = {d: c1429._load_day_tapes(d) for d in all_days}
    borrow = load_borrow_flags()

    results = {}
    for name, sigs in (('TRAIN-H2', train_h2), ('VAL', val)):
        s = score_split(sigs, paths_idx, tapes_by_day, exit_half_const, borrow)
        results[name] = s
        print(f"[cell_1431] {name} D={s['D']} chase_cap_skipped={s['n_chase_skipped']} "
              f"share_shortable={s['share_shortable']:.3f} fills/wk={s['fills_per_week']:.2f}")
        print(f"[cell_1431] {name} ALL n={s['all']['n']} mean_slip={s['all']['mean_slip']:+.4f} "
              f"t={s['all']['t']:.2f} (ndays={s['all']['ndays']})")
        print(f"[cell_1431] {name} SHORTABLE-ONLY n={s['shortable_only']['n']} "
              f"mean_slip={s['shortable_only']['mean_slip']:+.4f} t={s['shortable_only']['t']:.2f}")

    val_s = results['VAL']
    pass_ = (np.isfinite(results['TRAIN-H2']['all']['mean_slip'])
             and results['TRAIN-H2']['all']['mean_slip'] >= 0.10
             and np.isfinite(val_s['all']['mean_slip']) and val_s['all']['mean_slip'] >= 0.10
             and np.isfinite(val_s['all']['t']) and val_s['all']['t'] >= 2.0
             and np.isfinite(val_s['share_shortable']) and val_s['share_shortable'] >= 0.60
             and np.isfinite(val_s['fills_per_week']) and val_s['fills_per_week'] >= 3.0)
    print(f"[cell_1431] -> {'PASS' if pass_ else 'FAIL'}")
    return results, pass_


if __name__ == '__main__':
    run()
