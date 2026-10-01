"""Cell 1,691 -- parity check: LIVE add-on-pool admission code vs the idea1/P1 backtest harness.

Owner's request: "code review the live code of the new pool and ensure it aligns with the BT."
Read-only verification. Never touches config/orb.yaml/data/*.db (all DB opens are ?mode=ro) or the
live service; never git commit.

LIVE code under test (imported, not re-typed): trading/orb_addon_gates.py::evaluate_pool_gates +
PoolGateInputs, the exact functions trading/orb_engine.py::ORBEngine._run_pool_selection calls at
the 09:35 pre-placement instant for every non-production add-on pool.

BT harness under test (independently re-typed from its source + its own on-disk output, NOT
imported): research/orb_freq/1684_build.py::flag_ideas (idea1_pre daily-bar prefilter) +
research/orb_freq/1684_fastpath.py::main (the final idea1 mask, gap ceiling + the
move-to-range-high>=5% condition read off entry_price==range_high -- study_orb_features.py:435).
Ground truth for "the harness's admitted list before selection" is the actual on-disk artifact
research/orb_freq/fastpath/idea1_features.csv (written by 1684_fastpath.py BEFORE
study_orb_pipeline_static_lock.py's ranking/slot/exit chain runs -- i.e. pre-selection, as the
PREREG asked for).

Pool P1 = orb.yaml.template's `addon_gap35_range5` (universe.addon_pools.pools[2]), the owner's
"gap 3-5% plus a >=5% run by 09:35" pool. Its dict is hand-copied below VERBATIM from the template
(read-only; orb.yaml itself is never opened).

Window: 2026-06-01..2026-09-26. The on-disk harness artifact only covers 2025-01-02..2026-09-18
(the wide-seed CSVs' own span) -- the last 6 trading days of the requested window (09-19..09-26)
have NO harness admitted-list to diff against; this script computes the engine side for them
anyway and reports them separately, never silently drops them.

Usage: python3 research/orb_freq/1691_parity_check.py
"""
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
sys.path.insert(0, str(ROOT))
from trading.orb_addon_gates import GATE_KEYS, PoolGateInputs, evaluate_pool_gates  # noqa: E402

CACHE_DB = ROOT / 'data/cache.db'
BARS_SIP = ROOT / 'research/bf_zero/bars_sip.db'
OUT = ROOT / 'research/orb_freq'
HARNESS_ADMITTED_CSV = OUT / 'fastpath' / 'idea1_features.csv'

START, END = '2026-06-01', '2026-09-26'
HARNESS_COVERS_THROUGH = '2026-09-18'  # idea1_features.csv's own max date (verified below)

# orb.yaml.template lines 108-113, hand-copied verbatim (read-only; never opens orb.yaml).
POOL_CFG = {
    'name': 'addon_gap35_range5',
    'min_gap_pct': 3.0,
    'max_gap_pct': 5.0,
    'min_price': 3.0,
    'max_price': 30.0,
    'min_prev_volume': 500000,
    'min_move_to_range_high_pct': 5.0,
}


def ro(path):
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    # CLAUDE.md "System-in-dev": bars_sip.db is a shared read+append store other
    # research/dry-run processes write to concurrently -- fail after 30s of lock
    # contention rather than hang silently.
    con.execute("PRAGMA busy_timeout=30000")
    return con


def load_daily_with_prev(start, end):
    """Re-typed from research/orb_freq/pools_1684_lib.py::load_daily (read, not imported):
    daily_bars rows with prev_close/prev_volume shifted within symbol and gap_pct derived.
    Padded 20 calendar days back so `start`'s own prev_* are correct. Read-only on cache.db."""
    pad_start = (date.fromisoformat(start) - timedelta(days=20)).isoformat()
    con = ro(CACHE_DB)
    df = pd.read_sql_query(
        "SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars "
        "WHERE bar_date BETWEEN ? AND ?", con, params=[pad_start, end])
    con.close()
    df = df.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    g = df.groupby('symbol')
    df['prev_close'] = g['close'].shift(1)
    df['prev_volume'] = g['volume'].shift(1)
    df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
    df = df[(df['bar_date'] >= start) & (df['bar_date'] <= end)].reset_index(drop=True)
    return df


def harness_daily_candidates(daily):
    """Re-typed from 1684_build.py::flag_ideas' idea1_pre mask: band (today's `open` in [3,30],
    yesterday's `volume` i.e. prev_volume >= 500K) AND gap_pct in [3,5) AND not already
    production (gap_pct >= 5). `open`/`prev_close`/`prev_volume` are daily_bars columns -- the
    official session open and yesterday's official close/volume, not a live snapshot."""
    band = (daily['open'] >= 3.0) & (daily['open'] <= 30.0) & (daily['prev_volume'] >= 500_000)
    is_prod = band & (daily['gap_pct'] >= 5.0)
    idea1_pre = band & (daily['gap_pct'] >= 3.0) & (daily['gap_pct'] < 5.0) & ~is_prod
    return daily.loc[idea1_pre, ['symbol', 'bar_date', 'open', 'prev_close', 'prev_volume',
                                  'gap_pct']].copy()


def range_0930(con, symbol, day):
    """Independent reimplementation of study_orb.py::_session_open_timestamp (the bar whose ET
    wall time is EXACTLY 09:30, not merely the first bar) + study_orb_features.py::extract_features'
    range slice (that bar plus the next 4 one-minute bars, < 09:35 ET; reject if fewer than 5).
    This is also exactly what trading/orb_engine.py's RangeData construction keys off (range_open
    = bar0.open "9:30 bar open -- BT-parity" per its own comment; range_close = last bar's close;
    range_high/low = max/min across the 5 bars). Read-only on bars_sip.db."""
    rows = con.execute(
        "SELECT t, o, h, l, c, v FROM bars WHERE symbol=? AND day=? ORDER BY t",
        (symbol, day)).fetchall()
    if not rows:
        return None, 'no_bars_sip_rows'
    try:
        ts_utc = [pd.Timestamp(r[0]).tz_localize('UTC') if pd.Timestamp(r[0]).tzinfo is None
                  else pd.Timestamp(r[0]).tz_convert('UTC') for r in rows]
        ts_et = [t.tz_convert(ZoneInfo('America/New_York')) for t in ts_utc]
    except Exception as e:
        return None, f'timestamp_parse_error:{e}'
    open_idx = next((i for i, e in enumerate(ts_et) if e.hour == 9 and e.minute == 30), None)
    if open_idx is None:
        return None, 'no_0930_et_bar'
    open_ts = ts_utc[open_idx]
    rb = [r for r, t in zip(rows, ts_utc) if open_ts <= t < open_ts + timedelta(minutes=5)]
    if len(rb) < 5:
        return None, f'only_{len(rb)}_of_5_range_bars'
    highs = [r[2] for r in rb]
    lows = [r[3] for r in rb]
    return {
        'range_high': max(highs), 'range_low': min(lows),
        'range_open': rb[0][1], 'range_close': rb[-1][4],
        'range_total_volume': sum(r[5] for r in rb),
    }, 'ok'


def check_fail_closed():
    """Second check (PREREG_1691 part 2): evaluate_pool_gates must admit nothing when its inputs
    are unresolved. Exercise it on a synthetic pool_cfg that activates EVERY GATE_KEYS threshold
    (not just P1's one active gate) against an all-None PoolGateInputs()."""
    synthetic_cfg = {
        'name': 'synthetic_all_gates',
        'min_move_to_range_high_pct': 1.0,
        'min_rel_volume_0935': 1.0,
        'min_premarket_dollar_vol': 1.0,
        'require_above_vwap_0935': True,
        'max_dist_to_52wk_high_pct': 50.0,
        'min_prev_day_range_atr': 0.1,
        'require_day2_gapper': True,
    }
    admitted, gate_values = evaluate_pool_gates(synthetic_cfg, PoolGateInputs())
    checked = {k for k in GATE_KEYS if k in gate_values or
               (k == 'require_above_vwap_0935' and 'above_vwap_0935' in gate_values) or
               (k == 'require_day2_gapper' and 'is_day2_gapper' in gate_values)}
    print(f"[1691 fail-closed] synthetic all-gates cfg, all-None inputs -> "
          f"admitted={admitted} (expect False); gate_values={gate_values}")
    assert admitted is False, "FAIL-CLOSED VIOLATION: evaluate_pool_gates admitted on all-None inputs"
    # P1's actual single-gate config, same all-None inputs.
    admitted_p1, gv_p1 = evaluate_pool_gates(POOL_CFG, PoolGateInputs())
    print(f"[1691 fail-closed] P1 cfg (min_move_to_range_high_pct={POOL_CFG['min_move_to_range_high_pct']}), "
          f"all-None inputs -> admitted={admitted_p1} (expect False); gate_values={gv_p1}")
    assert admitted_p1 is False, "FAIL-CLOSED VIOLATION on P1's own config"
    return True


def main():
    print(f"[1691] === PART 2: fail-closed synthetic-input check ===", flush=True)
    check_fail_closed()

    print(f"\n[1691] === PART 1: admission parity {START}..{END}, pool={POOL_CFG['name']} ===",
          flush=True)
    harness_csv = pd.read_csv(HARNESS_ADMITTED_CSV, keep_default_na=False, na_values=[''])
    h_min, h_max = harness_csv['date'].min(), harness_csv['date'].max()
    print(f"[1691] harness ground truth {HARNESS_ADMITTED_CSV}: {len(harness_csv)} rows, "
          f"date range {h_min}..{h_max}", flush=True)
    harness_set = set(zip(harness_csv['date'], harness_csv['symbol']))

    daily = load_daily_with_prev(START, END)
    cands = harness_daily_candidates(daily)
    print(f"[1691] re-derived idea1_pre daily-bar candidates in window: {len(cands)}", flush=True)

    bars_con = ro(BARS_SIP)
    rows = []
    for _, r in cands.iterrows():
        sym, day = r['symbol'], r['bar_date']
        rd, reason = range_0930(bars_con, sym, day)
        if rd is None:
            inputs = PoolGateInputs()  # fail-closed path: unresolved input
            engine_admit, gv = evaluate_pool_gates(POOL_CFG, inputs)
            rows.append({'bar_date': day, 'symbol': sym, 'open': r['open'],
                         'prev_close': r['prev_close'], 'prev_volume': r['prev_volume'],
                         'gap_pct': r['gap_pct'], 'move_to_range_high_pct': None,
                         'engine_admit': engine_admit, 'resolve_reason': reason})
            continue
        move_pct = (rd['range_high'] - r['prev_close']) / r['prev_close'] * 100.0
        inputs = PoolGateInputs(move_to_range_high_pct=move_pct)
        engine_admit, gv = evaluate_pool_gates(POOL_CFG, inputs)
        rows.append({'bar_date': day, 'symbol': sym, 'open': r['open'],
                     'prev_close': r['prev_close'], 'prev_volume': r['prev_volume'],
                     'gap_pct': r['gap_pct'], 'move_to_range_high_pct': move_pct,
                     'engine_admit': engine_admit, 'resolve_reason': reason})
    bars_con.close()

    res = pd.DataFrame(rows)
    res['harness_admit'] = list(zip(res['bar_date'], res['symbol']))
    res['harness_admit'] = res['harness_admit'].isin(harness_set)
    res['in_harness_coverage'] = res['bar_date'] <= HARNESS_COVERS_THROUGH

    cov = res[res['in_harness_coverage']]
    out_of_cov = res[~res['in_harness_coverage']]

    both = cov[cov['engine_admit'] & cov['harness_admit']]
    engine_only = cov[cov['engine_admit'] & ~cov['harness_admit']]
    harness_only = cov[~cov['engine_admit'] & cov['harness_admit']]
    neither = cov[~cov['engine_admit'] & ~cov['harness_admit']]

    n_total = len(cov)
    n_agree = len(both) + len(neither)
    print(f"\n[1691] -- coverage window {START}..{HARNESS_COVERS_THROUGH} ({n_total} re-derived "
          f"daily-bar candidates) --", flush=True)
    print(f"[1691] agree: {n_agree}/{n_total} ({100*n_agree/max(1,n_total):.1f}%)  "
          f"[both_admit={len(both)} neither_admit={len(neither)}]")
    print(f"[1691] engine-only (engine admits, harness does not): {len(engine_only)}")
    print(f"[1691] harness-only (harness admits, engine does not): {len(harness_only)}")

    pd.set_option('display.width', 160)
    cols = ['bar_date', 'symbol', 'open', 'prev_close', 'prev_volume', 'gap_pct',
            'move_to_range_high_pct', 'resolve_reason']
    print(f"\n[1691] first 10 ENGINE-ONLY:\n{engine_only[cols].head(10).to_string(index=False)}")
    print(f"\n[1691] first 10 HARNESS-ONLY:\n{harness_only[cols].head(10).to_string(index=False)}")

    print(f"\n[1691] -- out-of-coverage tail {HARNESS_COVERS_THROUGH}..{END} "
          f"(no harness ground truth on disk) --")
    print(f"[1691] re-derived candidates: {len(out_of_cov)}, engine-admits: "
          f"{int(out_of_cov['engine_admit'].sum())}")
    print(out_of_cov[cols].to_string(index=False))

    res.to_csv(OUT / '1691_parity_rows.csv', index=False)
    print(f"\n[1691] wrote {OUT / '1691_parity_rows.csv'} ({len(res)} rows)")


if __name__ == '__main__':
    main()
