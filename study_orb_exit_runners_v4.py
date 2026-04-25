"""Round 4: Multi-signal gated runner-protection.

Round 3 found V11b (remove_stop_after_MFE5R) beats V0 by just +$1,226 — within
noise. The failure cases (like TNXP 2025-02-07, MFE +9R but round-tripped below
+1R by EOD) suggest we need additional signals to filter out "fake runners"
before applying the runner-protection.

This round tests V11b with various ENTRY-TIME gates layered on top. The gate
is computed once per trade (from features CSV + pipeline-assigned quintile),
and V11b's runner-protection logic only activates if the gate passes.

  V0                baseline
  V11b              bare MFE>=5R → remove stop (round-3 winner by $1K)
  V11b_Q5only       + quintile == Q5
  V11b_Q4Q5         + quintile in {Q4, Q5}
  V11b_SPYpos       + spy_return_5min_pct > 0 (market supporting)
  V11b_Q_and_SPY    + Q4/Q5 AND SPY positive
  V11b_VWAP         + opened above VWAP (range_vwap_distance_pct > 0)
  V11b_hivol        + prev_day_volume_vs_20d > 2.0
  V11b_no_lowgap    + gap_pct > 0 (no down-gap trades)
  V11b_multi        + combined strict gate (Q4/Q5, SPY pos, VWAP, gap>0)

Conceptually: V0 is the default exit. When a gate PASSES and MFE>=5R is also
met, switch off the +1R lock and ride to EOD. Every other trade uses V0.
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import timedelta
from typing import Callable, List, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database
from study_orb import _bars_to_df
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, fit_quintile_cutoffs, assign_quintile,
    ADAPTIVE_MULT_MIN,
)
from study_orb_correlation_filter import symbol_family, symbol_super_group


ACCOUNT = 100_000.0
N = 4
RISK = 3000.0
MIN_STOP_PCT = 1.0
OLD_POS = 50_000.0
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
EXIT_SLIP_BPS = 10.0
RUNNER_MFE_R = 5.0


def _session_open_timestamp(bars):
    if bars.empty:
        return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning):
        return morning.iloc[0]['timestamp']
    return None


def simulate_gated_v11b(bars, entry_price, range_high, range_low, entry_time,
                         gate_active: bool
                         ) -> Tuple[float, str, float]:
    """If gate_active and MFE>=5R, remove stop (V11b). Else: V0 static_lock_1R.

    When gate is INACTIVE this is bit-identical to V0.
    """
    range_size = range_high - range_low
    trigger_lvl = entry_price + 1.5 * range_size
    lock_stop = entry_price + 1.0 * range_size
    runner_trigger_abs = RUNNER_MFE_R * range_size
    stop_price = range_low
    armed = False
    runner_confirmed = False
    mfe_abs = 0.0
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        mfe_abs = max(mfe_abs, bar_high - entry_price)
        if not armed and bar_high >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop)
        if gate_active and armed and mfe_abs >= runner_trigger_abs:
            runner_confirmed = True
        if not runner_confirmed and bar_low <= stop_price:
            px = stop_price * (1 - EXIT_SLIP_BPS/10000)
            return px, 'lock' if armed else 'stop', mfe_abs / range_size
    last = post.iloc[-1]
    reason = 'eod_runner' if runner_confirmed else 'eod'
    return float(last['close']) * (1 - EXIT_SLIP_BPS/10000), reason, mfe_abs / range_size


# ---------------------------------------------------------------------------
# Gate predicates — each takes a row (pandas Series) with all features and
# returns True if gate passes (meaning: if MFE hits 5R, remove stop).
# ---------------------------------------------------------------------------

def gate_always(row) -> bool:
    return True  # equivalent to V11b bare


def gate_q5(row) -> bool:
    return row['_quintile'] == 'Q5'


def gate_q4q5(row) -> bool:
    return row['_quintile'] in ('Q4', 'Q5')


def gate_spy_pos(row) -> bool:
    return float(row.get('spy_return_5min_pct', 0) or 0) > 0


def gate_q_and_spy(row) -> bool:
    return gate_q4q5(row) and gate_spy_pos(row)


def gate_vwap(row) -> bool:
    return float(row.get('range_vwap_distance_pct', 0) or 0) > 0


def gate_hivol(row) -> bool:
    return float(row.get('prev_day_volume_vs_20d', 0) or 0) > 2.0


def gate_no_down_gap(row) -> bool:
    return float(row.get('gap_pct', 0) or 0) > 0


def gate_strict_multi(row) -> bool:
    return (gate_q4q5(row)
            and gate_spy_pos(row)
            and gate_vwap(row)
            and gate_no_down_gap(row))


GATES: List[Tuple[str, Callable]] = [
    ('V0 (baseline, gate disabled)',      lambda r: False),  # identical to V0
    ('V11b bare (no gate)',               gate_always),
    ('V11b + Q5 only',                    gate_q5),
    ('V11b + Q4/Q5',                      gate_q4q5),
    ('V11b + SPY positive',               gate_spy_pos),
    ('V11b + Q4/Q5 + SPY positive',       gate_q_and_spy),
    ('V11b + opened above VWAP',          gate_vwap),
    ('V11b + prev_day_vol > 2x',          gate_hivol),
    ('V11b + gap_pct > 0',                gate_no_down_gap),
    ('V11b + strict (Q+SPY+VWAP+gap)',    gate_strict_multi),
]


# ---------------------------------------------------------------------------
# Pipeline scaffolding
# ---------------------------------------------------------------------------

def _simulate_all(df, bars_cache, gate_fn):
    """Simulate all trades. `gate_fn` is called per-row with the full Series."""
    pnls, pcts, reasons, mfes, gate_flags = [], [], [], [], []
    for _, row in df.reset_index(drop=True).iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); mfes.append(0.0); gate_flags.append(False)
            continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); mfes.append(0.0); gate_flags.append(False)
            continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) &
                          (bars['timestamp'] < range_end)]
        if len(range_bars) < 5:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); mfes.append(0.0); gate_flags.append(False)
            continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh:
                entry_ts = b['timestamp']; break
        if entry_ts is None:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); mfes.append(0.0); gate_flags.append(False)
            continue
        gate_active = bool(gate_fn(row))
        entry_p = float(row['entry_price'])
        exit_p, reason, mfe_r = simulate_gated_v11b(
            bars, entry_p, rh, rl, entry_ts, gate_active=gate_active
        )
        shares = max(1, int(OLD_POS / entry_p))
        pnls.append((exit_p - entry_p) * shares)
        pcts.append((exit_p - entry_p) / entry_p * 100)
        reasons.append(reason); mfes.append(mfe_r); gate_flags.append(gate_active)
    out = df.reset_index(drop=True).copy()
    out['pnl'] = pnls; out['pnl_pct'] = pcts
    out['exit_reason'] = reasons; out['mfe_r'] = mfes
    out['_gate_active'] = gate_flags
    return out


def _preassign_quintile(df):
    """Assign quintile per row based on composite (independent of exit P&L).

    Pipeline logic: fit z_params on TRAIN H1 2025, compute composite, fit
    quintile cutoffs on TRAIN k-filtered subset, assign quintile to all rows.
    """
    prelim = df.copy()
    prelim_stop = prelim['range_size_pct'].clip(lower=MIN_STOP_PCT)
    prelim['_rp_position'] = (RISK / (prelim_stop / 100.0)).clip(upper=ACCOUNT/N)
    prelim['_rp_pnl'] = prelim['pnl'] * prelim['_rp_position'] / OLD_POS
    train = prelim[(prelim['date'] >= '2025-01-01') & (prelim['date'] <= '2025-06-30')]
    params = fit_z_params(train, FILTER_FEATURES)
    prelim['_composite'] = composite_score(prelim, params)
    train_k = prelim[(prelim['date'] >= '2025-01-01') &
                     (prelim['date'] <= '2025-06-30') &
                     (prelim['_composite'] >= FILTER_THRESHOLD)]
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    prelim['_quintile'] = assign_quintile(prelim['_composite'], cutoffs)
    return prelim[['symbol', 'date', '_composite', '_quintile']]


def _run_pipeline(df_with_pnl):
    df = df_with_pnl.copy()
    per_pos_cap = ACCOUNT / N
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = RISK / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    train_k = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    train_k['_quintile'] = assign_quintile(train_k['_composite'], cutoffs)
    avg = float(train_k['_rp_pnl'].mean())
    mults = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = train_k[train_k['_quintile'] == q]
        m = float(sub['_rp_pnl'].mean()) / avg if len(sub) else 1.0
        mults[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], m))
    kept = df[df['_composite'] >= FILTER_THRESHOLD].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)
    sel_rows = []
    for _, dg in kept.groupby('date'):
        d = dg.copy()
        d['_q_rank'] = d['_quintile'].map(Q_ORDER)
        d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        seen_fam = set(); seen_sup = set(); today = []
        for _, r in d.iterrows():
            fam = symbol_family(r['symbol']); sup = symbol_super_group(r['symbol'])
            if fam and fam in seen_fam: continue
            if sup and sup in seen_sup: continue
            if fam: seen_fam.add(fam)
            if sup: seen_sup.add(sup)
            today.append(r)
            if len(today) >= N: break
        sel_rows.extend(today)
    sel = pd.DataFrame(sel_rows)
    sel['_sized_pnl'] = sel.apply(lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    sel['date'] = pd.to_datetime(sel['date'])
    return sel, mults


def _metrics(sel, label):
    daily = sel.groupby('date').agg(pnl=('_sized_pnl', 'sum')).reset_index().sort_values('date').reset_index(drop=True)
    daily['cum'] = daily['pnl'].cumsum()
    peak = -1e18; mdd = 0.0
    for _, r in daily.iterrows():
        peak = max(peak, r['cum'])
        mdd = min(mdd, r['cum'] - peak)
    total_pnl = float(daily['pnl'].sum())
    calmar = total_pnl / abs(mdd) if mdd < 0 else float('inf')
    daily['month'] = daily['date'].dt.to_period('M')
    monthly = daily.groupby('month')['pnl'].sum()
    gate_hits = sel['_gate_active'].sum() if '_gate_active' in sel.columns else 0
    return {
        'label': label, 'trades': len(sel), 'pnl': total_pnl,
        'max_dd': float(mdd), 'calmar': calmar,
        'neg_months': int((monthly < 0).sum()),
        'gate_hits': int(gate_hits),
    }


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    print(f"Loading features from {csv}")
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'pnl_pct', 'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])
    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    print(f"Loading bars for {len(pairs)} pairs...")
    db = Database(db_path='data/cache.db')
    raw_bars = db.get_intraday_bars_bulk(pairs)
    db.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}

    # Pre-assign quintile per row (needed for gate evaluation)
    print("Pre-assigning quintiles...")
    q_df = _preassign_quintile(df)
    df = df.merge(q_df, on=['symbol', 'date'], how='left')

    all_m = []; all_sel = {}
    for label, gate_fn in GATES:
        print(f"\nSimulating {label}...")
        df_v = _simulate_all(df, bars_cache, gate_fn)
        sel, mults = _run_pipeline(df_v)
        # Attach per-row gate flag to selected trades
        gate_map = dict(zip(
            df_v.apply(lambda r: (r['symbol'], r['date']), axis=1),
            df_v['_gate_active']
        ))
        sel['_gate_active'] = sel.apply(
            lambda r: gate_map.get((r['symbol'], r['date']), False), axis=1
        )
        m = _metrics(sel, label)
        all_m.append(m); all_sel[label] = sel
        print(f"  P&L ${m['pnl']:+,.0f}  DD ${m['max_dd']:+,.0f}  "
              f"Calmar {m['calmar']:.2f}x  neg_mo {m['neg_months']}  "
              f"gate_hits {m['gate_hits']}/{m['trades']}")

    print(f"\n{'='*105}")
    print("  COMPARISON (full timeline Jan'25 → Apr'26)")
    print(f"{'='*105}")
    base = all_m[0]
    print(f"{'Variant':<42} {'P&L':>11} {'Δ P&L':>10} {'Max DD':>10} {'Calmar':>8} "
          f"{'Neg':>4} {'Gate/Tot':>9}")
    print('-' * 105)
    for m in all_m:
        delta = m['pnl'] - base['pnl']
        marker = ' ★' if m['pnl'] > base['pnl'] else ''
        print(f"{m['label']:<42} "
              f"${m['pnl']:>+9,.0f}  "
              f"${delta:>+8,.0f}  "
              f"${m['max_dd']:>+7,.0f}  "
              f"{m['calmar']:>6.2f}x  "
              f"{m['neg_months']:>3}   "
              f"{m['gate_hits']:>3}/{m['trades']:<4}{marker}")

    winners = [m for m in all_m[1:] if m['pnl'] > base['pnl']]  # skip V0 baseline itself
    if winners:
        winners.sort(key=lambda x: x['pnl'], reverse=True)
        print(f"\n{'='*105}\n  ★ {len(winners)} GATE(S) BEAT BASELINE\n{'='*105}")
        for m in winners:
            delta = m['pnl'] - base['pnl']
            pct = delta / abs(base['pnl']) * 100
            dd_delta = m['max_dd'] - base['max_dd']
            print(f"  {m['label']}")
            print(f"    P&L:  ${m['pnl']:+,.0f}  (Δ ${delta:+,.0f}, {pct:+.2f}%)")
            print(f"    DD:   ${m['max_dd']:+,.0f}  (Δ ${dd_delta:+,.0f} vs baseline)")
            print(f"    Calmar: {m['calmar']:.2f}x  (baseline {base['calmar']:.2f}x)")
            print(f"    Gate hits: {m['gate_hits']}/{m['trades']} "
                  f"({m['gate_hits']/m['trades']*100:.1f}% of trades)")

    pd.DataFrame(all_m).to_csv('analysis_results/orb_exit_v4_summary.csv', index=False)
    print("\nSaved analysis_results/orb_exit_v4_summary.csv")


if __name__ == '__main__':
    main()
