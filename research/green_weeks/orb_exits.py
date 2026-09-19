"""green_weeks — ORB exit cells E0..E5, one bar walk (PREREG §4).

Extends Stage M's `exit_shapes.py` (same walker, same population, same shipped
selector afterwards) with the three levers Stage M never had: a fixed target on
the whole position, a profit partial, and a trail on the remainder.

Produces one candidate dump per cell, identical in every column to Stage M's
`dump_X0.csv` except `pnl` / `pnl_pct` / `exit_reason`.  The shipped selector
then runs off each dump via `ORB_BT_RESIM_CACHE`, so selection is identical
across cells by construction and the diff is pure exit.

Cells (PREREG §4): X0 = E0 shipped · E1a-d fixed target 0.5/0.75/1.0/1.5R ·
E2a-c 50% partial at 0.5/1.0/1.5R + breakeven · E3 50%@1R + 0.5R trail, no lock ·
E4a/b time box 10/20 min at +0.5R · E5 breakeven-only at +0.75R.

Verification (PREREG §6.1):
  * the parametrised walker with every lever neutral must reproduce the SHIPPED
    `simulate_winner_stack` exit price AND reason on every fill (hard assert);
  * X0's pnl must equal Stage M's committed `dump_X0.csv` to 1e-9 on all rows.

Resources: one process, bars symbol by symbol, cache.db READ-ONLY, `ulimit -v`
set by the caller.
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta, time as _dtime

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')

import sqlite3                                        # noqa: E402
import study_orb_pipeline_static_lock as P            # noqa: E402
from trading.orb_csv import read_orb_csv              # noqa: E402
from trading.orb_touchgo_filter import (              # noqa: E402
    evaluate_rule_m, evaluate_rule_d, find_breakout_bar_ts)
from trading.orb_winner_stack import (                # noqa: E402
    floored_stop, scale_params)

D = 'research/green_weeks'
M_DIR = 'research/fuckup_audit/M'
FEATURES = 'analysis_results/orb_features_20260916_2053.csv'
D1_DUMP = 'research/fuckup_audit/M/dump_X0.csv'
CACHE_RO = 'file:data/cache.db?mode=ro'


def load_symbol_bars(conn, symbol, days):
    """{'YYYY-MM-DD' -> bars DataFrame} for one symbol, via the (symbol,
    bar_date) index — cache.db opened READ-ONLY.  Same frame shape the shipped
    walkers expect (`timestamp` tz-aware UTC, sorted)."""
    q = ("select bar_date, timestamp, open, high, low, close "
         "from intraday_bars_1min where symbol = ? and bar_date in (%s)"
         % ','.join('?' * len(days)))
    raw = conn.execute(q, (symbol, *days)).fetchall()
    if not raw:
        return {}
    d = pd.DataFrame(raw, columns=['bar_date', 'timestamp', 'open', 'high',
                                   'low', 'close'])
    d['bar_date'] = d['bar_date'].astype(str).str[:10]
    d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True)
    out = {}
    for day, g in d.groupby('bar_date'):
        out[day] = (g.drop(columns=['bar_date'])
                    .sort_values('timestamp').reset_index(drop=True))
    return out

# ---- green_weeks cells (PREREG §4).  Knobs, in order:
#   lock_enabled, ts_minutes, ts_r, be_at_r, target_r, partial_r, partial_frac,
#   trail_r
# `X0` is the SHIPPED exit and the parity reference; it is Stage M's own X0 and
# its dump is re-derived here only to prove the extended walker did not move it.
SHAPES = {
    'X0':  (True,  None, None, None, None, None, None, None),   # E0 shipped
    'E1a': (True,  None, None, None, 0.50, None, None, None),
    'E1b': (True,  None, None, None, 0.75, None, None, None),
    'E1c': (True,  None, None, None, 1.00, None, None, None),
    'E1d': (True,  None, None, None, 1.50, None, None, None),
    'E2a': (True,  None, None, None, None, 0.50, 0.50, None),
    'E2b': (True,  None, None, None, None, 1.00, 0.50, None),
    'E2c': (True,  None, None, None, None, 1.50, 0.50, None),
    'E3':  (False, None, None, None, None, 1.00, 0.50, 0.50),
    'E4a': (True,  10,   0.50, None, None, None, None, None),
    'E4b': (True,  20,   0.50, None, None, None, None, None),
    'E5':  (True,  None, None, 0.75, None, None, None, None),
}


def simulate_shape(bars, entry_price, range_high, range_low, entry_time, shares,
                   atr14, atr_floor_enabled, atr_floor_k,
                   scale_enabled, scale_frac, scale_level_r,
                   lock_enabled=True, ts_minutes=None, ts_r=None, be_at_r=None,
                   target_r=None, partial_r=None, partial_frac=None,
                   trail_r=None):
    """Parametrised copy of `simulate_winner_stack` — Stage M's three levers
    (lock / time stop / breakeven) plus green_weeks' three (fixed target,
    profit partial, trail on the remainder).

    With every lever neutral it is the shipped walker; the run asserts that
    byte-for-byte on every fill AND against Stage M's committed dump_X0.csv.

    Ordering inside a bar (declared, PREREG §5):
      1. the time stop reads that bar's OPEN (it happens before the bar's range);
      2. lock arm / breakeven arm off that bar's HIGH;
      3. trail ratchet, from CLOSED bars at or before i-1, only once the partial
         has fired;
      4. protective stop off that bar's LOW (gated by the scale touch, frozen
         same-bar rule) — a stop wins a same-bar tie against any profit leg;
      5. fixed target: fires on a bar that CLOSES at or above the level and
         fills AT the level (`trading/hod_break.walk_exit`'s convention, the
         house rule) — never a wick/touch fill, and never a fabricated bad fill
         on a spike-and-collapse bar.  The shipped 10 bps exit slip is applied
         to it like every other exit, which is conservative for a resting limit;
      6. profit partial: same trigger and fill convention, then stop -> entry;
      7. scale touch;
      8. mid-kill.

    Returns (effective_exit_price, reason).
    """
    range_size = range_high - range_low
    slip = 1 - P.EXIT_SLIP_BPS / 10000
    _et = bars['timestamp'].dt.tz_convert('America/New_York').dt.time
    _fc_h, _fc_m = (int(x) for x in P.FORCE_CLOSE_ET.split(':'))
    bars = bars[_et <= _dtime(_fc_h, _fc_m)]
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    if len(post) == 0:
        return entry_price, 'no_bars'

    # Touchgo prefires the whole position (frozen; no scale on touchgo trades).
    eb = post.iloc[0]
    fire_m, exit_m = evaluate_rule_m(float(eb['open']), float(eb['high']),
                                     float(eb['low']), float(eb['close']),
                                     P.TOUCHGO_CFG)
    if fire_m and exit_m is not None:
        return exit_m * slip, 'tag_bb'
    if len(post) >= 2:
        b1 = post.iloc[1]
        fire_d, exit_d = evaluate_rule_d(entry_price, float(b1['low']),
                                         range_size, P.TOUCHGO_CFG)
        if fire_d and exit_d is not None:
            return exit_d * slip, 'tag_b1'

    stop0 = range_low
    if atr_floor_enabled:
        stop0, floor_status = floored_stop(range_low, entry_price, atr14,
                                           atr_floor_k)
        if floor_status in ('no_atr', 'degenerate'):
            print(f"WARNING: shape floor fail-open ({floor_status}) "
                  f"entry={entry_price} atr14={atr14} — stop stays range_low")

    scale_px = None
    frac_eff = 0.0
    if scale_enabled and shares >= 1:
        px, qty = scale_params(entry_price, range_size, scale_frac,
                               scale_level_r, shares)
        if qty >= 1:
            scale_px = px
            frac_eff = qty / float(shares)

    trig = entry_price + P.LOCK_TRIGGER_R * range_size
    lock = entry_price + P.LOCK_STOP_R * range_size
    be_lvl = (entry_price + be_at_r * range_size) if be_at_r is not None else None
    tgt_lvl = (entry_price + target_r * range_size) if target_r is not None else None
    pp_lvl = (entry_price + partial_r * range_size) if partial_r is not None else None
    # pp = [fraction sold, effective fill price]; blended into every later exit
    pp = [0.0, None]

    def _fin(px, rsn):
        """Blend the partial leg (if any) into a whole-position exit."""
        if pp[1] is None:
            return px, rsn
        f = pp[0]
        ret = f * (pp[1] / entry_price - 1) + (1 - f) * (px / entry_price - 1)
        return entry_price * (1 + ret), f'pp+{rsn}'

    opens = post['open'].to_numpy(dtype=float)
    highs = post['high'].to_numpy(dtype=float)
    lows = post['low'].to_numpy(dtype=float)
    closes = post['close'].to_numpy(dtype=float)
    n = len(post)

    # index of the first bar at/after fill + ts_minutes (never the entry bar)
    ts_idx = None
    if ts_minutes is not None:
        cutoff = pd.Timestamp(entry_time) + timedelta(minutes=ts_minutes)
        mask = (post['timestamp'] >= cutoff).to_numpy()
        mask[0] = False
        hits = np.nonzero(mask)[0]
        if len(hits):
            ts_idx = int(hits[0])

    def _time_stop_fires(i):
        return (ts_idx is not None and i == ts_idx
                and (opens[i] - entry_price) / range_size < ts_r)

    # ---- phase 1: to the stop-out or the scale touch ----
    stop = stop0
    armed = False
    be_armed = False
    scale_i = None
    _mh = float(highs[0])
    for i in range(1, n):
        if _time_stop_fires(i):
            return _fin(opens[i] * slip, 'time_stop')
        if lock_enabled and not armed and highs[i] >= trig:
            armed = True
            stop = max(stop, lock)
        if be_lvl is not None and not be_armed and highs[i] >= be_lvl:
            be_armed = True
            stop = max(stop, entry_price)
        if trail_r is not None and pp[1] is not None:
            stop = max(stop, _mh - trail_r * range_size)
        if lows[i] <= stop and (scale_px is None or highs[i] < scale_px):
            return _fin(stop * slip,
                        ('lock' if armed else ('be' if be_armed else 'stop')))
        if tgt_lvl is not None and closes[i] >= tgt_lvl:
            return _fin(tgt_lvl * slip, 'target')
        if pp_lvl is not None and pp[1] is None and closes[i] >= pp_lvl:
            pp[0] = float(partial_frac)
            pp[1] = pp_lvl * slip
            stop = max(stop, entry_price)
        if scale_px is not None and highs[i] >= scale_px:
            scale_i = i
            break
        _mh = max(_mh, float(highs[i]))
        if P.EXP.mid_kill and not armed and P._mid_kill_fires(
                float(closes[i]), range_high, range_low, entry_price, _mh):
            _px = float(opens[i + 1]) if i + 1 < n else float(closes[i])
            return _fin(_px * slip, 'mid_kill')
    if scale_i is None:
        return _fin(float(closes[-1]) * slip, 'eod')

    # ---- phase 2: the runner, from the scale bar, same initial stop ----
    stop2 = stop0
    armed2 = bool(lock_enabled and scale_i >= 1
                  and highs[1:scale_i + 1].max() >= trig)
    if armed2:
        stop2 = max(stop2, lock)
    be_armed2 = bool(be_lvl is not None and scale_i >= 1
                     and highs[1:scale_i + 1].max() >= be_lvl)
    if be_armed2:
        stop2 = max(stop2, entry_price)
    # the partial (all levels < the +3R scale level) has necessarily fired by
    # the scale bar — its breakeven floor carries into the runner
    if pp[1] is not None:
        stop2 = max(stop2, entry_price)
    _mh2 = float(highs[:scale_i + 1].max())
    run_px = None
    run_rsn = 'eod'
    for i in range(scale_i, n):
        if _time_stop_fires(i):
            run_px = opens[i] * slip
            run_rsn = 'time_stop'
            break
        if lock_enabled and not armed2 and highs[i] >= trig:
            armed2 = True
            stop2 = max(stop2, lock)
        if be_lvl is not None and not be_armed2 and highs[i] >= be_lvl:
            be_armed2 = True
            stop2 = max(stop2, entry_price)
        if trail_r is not None and pp[1] is not None:
            stop2 = max(stop2, _mh2 - trail_r * range_size)
        if lows[i] <= stop2:
            run_px = stop2 * slip
            run_rsn = 'lock' if armed2 else ('be' if be_armed2 else 'stop')
            break
        if tgt_lvl is not None and closes[i] >= tgt_lvl:
            run_px = tgt_lvl * slip
            run_rsn = 'target'
            break
        _mh2 = max(_mh2, float(highs[i]))
    if run_px is None:
        run_px = float(closes[-1]) * slip
    # legs: the partial (pp), the +3R scale-out, the runner.  The scale fraction
    # is of the ORIGINAL shares (the shipped `scale_params` basis) and is capped
    # so the three legs never exceed the position.
    f_pp = pp[0] if pp[1] is not None else 0.0
    f_sc = min(frac_eff, max(0.0, 1.0 - f_pp))
    f_run = max(0.0, 1.0 - f_pp - f_sc)
    ret = ((f_pp * (pp[1] / entry_price - 1) if f_pp else 0.0)
           + f_sc * (scale_px * slip / entry_price - 1)
           + f_run * (run_px / entry_price - 1))
    _pfx = 'pp+' if f_pp else ''
    return entry_price * (1 + ret), f'{_pfx}scale_{run_rsn}'


def main():
    cfg = P.load_bt_config()
    # hard-sync the module globals the shipped walkers read at call time
    P.MIN_STOP_PCT = cfg['min_stop_pct']
    P.OLD_POS = cfg['old_pos_ref']
    P.LOCK_TRIGGER_R = cfg['lock_arm_r']
    P.LOCK_STOP_R = cfg['lock_stop_r']
    P.EXIT_SLIP_BPS = cfg['exit_slip_bps']
    P.FORCE_CLOSE_ET = cfg['force_close_et']
    assert cfg['atr_floor_enabled'] and cfg['scale_enabled'], \
        "orb.yaml winner-stack flags changed — X0 would not be the shipped exit"

    df = read_orb_csv(FEATURES)
    needed = [f for f, _ in P.FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'pnl_pct',
                                    'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])
    _smoke = os.environ.get('M_SMOKE_MONTH')          # smoke test only
    if _smoke:
        df = df[df['date'].dt.strftime('%Y-%m') == _smoke]
        print(f"SMOKE: restricted to {_smoke}")
    df = df.reset_index(drop=True)
    print(f"candidates: {len(df)}", flush=True)

    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    print(f"symbol-days: {len(pairs)}", flush=True)

    # ATR14 lookup (the shipped builder; one daily-history load per symbol)
    print("building ATR14 lookup...", flush=True)
    atr_lookup = P.build_atr14_lookup(pairs)
    n_atr = sum(1 for v in atr_lookup.values() if v is not None)
    print(f"ATR14 available on {n_atr}/{len(atr_lookup)} symbol-days", flush=True)

    shapes = list(SHAPES)
    out = {s: {'pnl': [None] * len(df), 'pnl_pct': [None] * len(df),
               'reason': [None] * len(df)} for s in shapes}

    df['_day'] = df['date'].dt.strftime('%Y-%m-%d')
    n_parity_checked = 0
    n_fills = 0
    n_nofill = 0
    n_skipped = 0

    conn = sqlite3.connect(CACHE_RO, uri=True)
    groups = df.groupby('symbol').groups
    n_sym = len(groups)
    for si, (sym, gidx) in enumerate(sorted(groups.items()), 1):
        idx = list(gidx)
        days = sorted({df.at[i, '_day'] for i in idx})
        bars_cache = load_symbol_bars(conn, sym, days)
        for i in idx:
            row = df.loc[i]
            key = (row['symbol'], row['_day'])
            bars = bars_cache.get(row['_day'])

            def _keep_recorded():
                for s in shapes:
                    out[s]['pnl'][i] = row['pnl']
                    out[s]['pnl_pct'][i] = row['pnl_pct']
                    out[s]['reason'][i] = row['exit_reason']

            if bars is None or bars.empty:
                _keep_recorded(); n_skipped += 1; continue
            if P.is_no_fill(row):
                for s in shapes:
                    out[s]['pnl'][i] = 0.0
                    out[s]['pnl_pct'][i] = 0.0
                    out[s]['reason'][i] = P.NO_FILL_REASON
                n_nofill += 1
                continue
            open_ts = P._session_open_timestamp(bars)
            if open_ts is None:
                _keep_recorded(); n_skipped += 1; continue
            range_end = open_ts + timedelta(minutes=5)
            rb = bars[(bars['timestamp'] >= open_ts) & (bars['timestamp'] < range_end)]
            if len(rb) < 5:
                _keep_recorded(); n_skipped += 1; continue
            rh = float(rb['high'].max()); rl = float(rb['low'].min())
            search = bars[(bars['timestamp'] >= range_end)
                          & (bars['timestamp'] < range_end + timedelta(minutes=60))]
            entry_ts = find_breakout_bar_ts(search, rh)
            if entry_ts is None:
                _keep_recorded(); n_skipped += 1; continue
            entry_p = float(row['entry_price'])
            shares = max(1, int(P.OLD_POS / entry_p))
            atr = atr_lookup.get(key)
            n_fills += 1

            for s in shapes:
                (lock_en, tsm, tsr, be, tgt, ppr, ppf, trl) = SHAPES[s]
                px, rsn = simulate_shape(
                    bars, entry_p, rh, rl, entry_ts, shares, atr,
                    cfg['atr_floor_enabled'], cfg['atr_floor_k'],
                    cfg['scale_enabled'], cfg['scale_frac'], cfg['scale_level_r'],
                    lock_enabled=lock_en, ts_minutes=tsm, ts_r=tsr, be_at_r=be,
                    target_r=tgt, partial_r=ppr, partial_frac=ppf, trail_r=trl)
                pnl = (px - entry_p) * shares
                out[s]['pnl'][i] = pnl
                out[s]['pnl_pct'][i] = pnl / (entry_p * shares) * 100
                out[s]['reason'][i] = rsn

            # PREREG 6.1 — the neutral-knob copy must equal the SHIPPED walker
            spx, srsn = P.simulate_winner_stack(
                bars, entry_p, rh, rl, entry_ts, shares, atr14=atr,
                atr_floor_enabled=cfg['atr_floor_enabled'],
                atr_floor_k=cfg['atr_floor_k'],
                scale_enabled=cfg['scale_enabled'],
                scale_frac=cfg['scale_frac'],
                scale_level_r=cfg['scale_level_r'])
            x0_pnl = (spx - entry_p) * shares
            assert abs(x0_pnl - out['X0']['pnl'][i]) < 1e-9 and srsn == out['X0']['reason'][i], (
                f"PARITY BREAK {key}: shipped {spx:.6f}/{srsn} vs copy "
                f"{out['X0']['pnl'][i]:.6f}/{out['X0']['reason'][i]}")
            n_parity_checked += 1
        del bars_cache
        if si % 200 == 0:
            print(f"  symbols {si}/{n_sym} ({n_fills} fills walked)", flush=True)
    conn.close()

    print(f"fills walked {n_fills} | no-fill rows {n_nofill} | "
          f"kept-recorded {n_skipped} | parity-checked {n_parity_checked}", flush=True)

    base = df.drop(columns=['_day']).copy()
    for s in shapes:
        d = base.copy()
        d['pnl'] = out[s]['pnl']
        d['pnl_pct'] = out[s]['pnl_pct']
        d['exit_reason'] = out[s]['reason']
        _pfx = 'dump_smoke_' if _smoke else 'dump_'
        d.to_csv(f'{D}/{_pfx}{s}.csv', index=False)
        print(f"wrote {D}/{_pfx}{s}.csv ({len(d)} rows) "
              f"pnl_sum={d['pnl'].sum():+,.0f}", flush=True)

    if _smoke:
        print("SMOKE run — D1 dump comparison skipped"); return
    # PREREG 6.1 — X0 must equal D1's dump to 1e-9
    d1 = read_orb_csv(D1_DUMP)   # Stage M's committed shipped-exit dump
    d1['date'] = pd.to_datetime(d1['date'])
    x0 = read_orb_csv(f'{D}/dump_X0.csv')
    x0['date'] = pd.to_datetime(x0['date'])
    m = d1[['symbol', 'date', 'pnl']].merge(
        x0[['symbol', 'date', 'pnl']], on=['symbol', 'date'], suffixes=('_d1', '_x0'))
    assert len(m) == len(d1) == len(x0), f"key mismatch {len(m)} {len(d1)} {len(x0)}"
    dmax = float((m['pnl_d1'] - m['pnl_x0']).abs().max())
    print(f"X0 vs D1 candidates_dump: max |dpnl| = {dmax:.12f} over {len(m)} rows",
          flush=True)
    assert dmax < 1e-9, "X0 does not reproduce D1's dump — STOP"
    print("PARITY OK", flush=True)


if __name__ == '__main__':
    main()
