"""OOS walk-forward validation of the Stay-Breakeven exit rule.

Diagnostic (study_orb_pullback_oracle.py) showed aggregate always-stay +$77K
raw and ~+$110K with adaptive mults. But that was computed across full
timeline — we need to confirm the effect holds OOS.

Protocol:
  1. Fit pipeline on TRAIN (H1 2025): z-params, quintile cutoffs, mults.
  2. Three exit variants evaluated end-to-end with full sizing:
       V0  : static_lock_1R (shipped baseline)
       SB  : always-stay after +1R touch (breakeven stop + ride to EOD)
       SB_Q: SB but only for quintile Q3+ (Q1/Q2 keep V0 exit)
  3. Measure per-split: TRAIN / VAL / HOQ1+ / FULL
       - Total P&L
       - Max DD (continuous)
       - Calmar
       - Red months
       - Worst single day
  4. Ship decision rubric:
       - Both VAL and HOQ1+ show positive Δ ≥ $5K → ship SB or SB_Q
       - One split negative → investigate; consider stricter gate (Q5 only)
       - Both negative → don't ship
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

TRAIN_END = '2025-06-30'
VAL_END = '2025-12-31'


def _session_open_timestamp(bars):
    if bars.empty: return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning): return morning.iloc[0]['timestamp']
    return None


def sim_v0(bars, entry, rh, rl, et):
    rs = rh - rl
    if rs <= 0: return float(bars.iloc[-1]['close']), 'no_range'
    trig = entry + 1.5 * rs; lock = entry + 1.0 * rs
    stop = rl; armed = False
    post = bars[bars['timestamp'] >= et].reset_index(drop=True)
    for _, b in post.iloc[1:].iterrows():
        h = float(b['high']); lo = float(b['low'])
        if not armed and h >= trig: armed = True; stop = max(stop, lock)
        if lo <= stop: return stop * (1 - EXIT_SLIP_BPS/10000), ('lock' if armed else 'stop')
    return float(post.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000), 'eod'


def sim_stay_be(bars, entry, rh, rl, et):
    """Stay-Breakeven: arm at +1.5R. On pullback to +1R, DON'T exit — move stop
    to entry (breakeven). Ride to EOD. Exit early only if BE hit.
    """
    rs = rh - rl
    if rs <= 0: return float(bars.iloc[-1]['close']), 'no_range'
    trig = entry + 1.5 * rs; lock_lvl = entry + 1.0 * rs
    stop = rl; armed = False; pullback_seen = False
    post = bars[bars['timestamp'] >= et].reset_index(drop=True)
    for _, b in post.iloc[1:].iterrows():
        h = float(b['high']); lo = float(b['low'])
        if not armed and h >= trig: armed = True; stop = max(stop, lock_lvl)
        if armed and not pullback_seen and lo <= lock_lvl:
            pullback_seen = True
            stop = entry  # move to breakeven
            # continue: check same-bar BE hit (if bar went through BE too)
        # Exit check using current stop level
        if lo <= stop:
            return stop * (1 - EXIT_SLIP_BPS/10000), (
                'be' if pullback_seen else ('lock' if armed else 'stop'))
    return float(post.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000), (
        'eod_stay' if pullback_seen else ('eod_armed' if armed else 'eod'))


def sim_stay_be_q3plus(bars, entry, rh, rl, et, quintile: str):
    """SB but only for Q3, Q4, Q5. For Q1/Q2, revert to V0 static lock."""
    if quintile in ('Q1', 'Q2'):
        return sim_v0(bars, entry, rh, rl, et)
    return sim_stay_be(bars, entry, rh, rl, et)


def sim_stay_be_q4q5(bars, entry, rh, rl, et, quintile: str):
    """SB but only for Q4, Q5. Everything else keeps V0."""
    if quintile in ('Q4', 'Q5'):
        return sim_stay_be(bars, entry, rh, rl, et)
    return sim_v0(bars, entry, rh, rl, et)


def sim_stay_be_q5only(bars, entry, rh, rl, et, quintile: str):
    """SB but only for Q5 (highest composite). Everything else keeps V0."""
    if quintile == 'Q5':
        return sim_stay_be(bars, entry, rh, rl, et)
    return sim_v0(bars, entry, rh, rl, et)


def _simulate_all(df, bars_cache, exit_fn, variant_label: str,
                   quintile_map=None):
    """Apply exit_fn to every row. If variant requires quintile, pass it via quintile_map."""
    pnls, pcts, reasons = [], [], []
    for _, row in df.reset_index(drop=True).iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) &
                          (bars['timestamp'] < range_end)]
        if len(range_bars) < 5:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh: entry_ts = b['timestamp']; break
        if entry_ts is None:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); continue
        entry_p = float(row['entry_price'])
        shares = max(1, int(OLD_POS / entry_p))
        if quintile_map is not None:
            q = quintile_map.get((row['symbol'], row['date'].strftime('%Y-%m-%d')), 'Q3')
            exit_p, reason = exit_fn(bars, entry_p, rh, rl, entry_ts, q)
        else:
            exit_p, reason = exit_fn(bars, entry_p, rh, rl, entry_ts)
        pnls.append((exit_p - entry_p) * shares)
        pcts.append((exit_p - entry_p) / entry_p * 100)
        reasons.append(reason)
    out = df.reset_index(drop=True).copy()
    out['pnl'] = pnls; out['pnl_pct'] = pcts; out['exit_reason'] = reasons
    return out


def _run_pipeline(df_with_pnl):
    df = df_with_pnl.copy()
    per_pos_cap = ACCOUNT / N
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = RISK / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS

    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= TRAIN_END)]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= TRAIN_END)]
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
        dd = dg.copy()
        dd['_q_rank'] = dd['_quintile'].map(Q_ORDER)
        dd = dd.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        seen_fam = set(); seen_sup = set(); today = []
        for _, r in dd.iterrows():
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
    return sel, cutoffs, mults, params


def _metrics_for_slice(sel, lo, hi, label):
    sub = sel[(sel['date'] >= lo) & (sel['date'] <= hi)].copy()
    if len(sub) == 0:
        return {'label': label, 'pnl': 0, 'trades': 0, 'max_dd': 0,
                 'calmar': 0, 'neg_months': 0, 'worst_day': 0}
    daily = sub.groupby('date').agg(
        pnl=('_sized_pnl', 'sum')).reset_index().sort_values('date').reset_index(drop=True)
    daily['cum'] = daily['pnl'].cumsum()
    peak = -1e18; mdd = 0.0
    for _, r in daily.iterrows():
        peak = max(peak, r['cum']); mdd = min(mdd, r['cum'] - peak)
    tp = float(daily['pnl'].sum())
    calmar = tp / abs(mdd) if mdd < 0 else float('inf')
    daily['month'] = daily['date'].dt.to_period('M')
    monthly = daily.groupby('month')['pnl'].sum()
    neg = int((monthly < 0).sum())
    return {
        'label': label, 'trades': len(sub), 'pnl': tp,
        'max_dd': float(mdd), 'calmar': calmar, 'neg_months': neg,
        'worst_day': float(daily['pnl'].min()),
    }


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    print(f"Loading features from {csv}")
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'pnl_pct',
                                     'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])
    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    print(f"Loading bars for {len(pairs)} pairs...")
    db = Database(db_path='data/cache.db')
    raw_bars = db.get_intraday_bars_bulk(pairs)
    db.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}

    # First: simulate V0 to get the quintile labeling we'll need for SB_Q3
    print("Simulating V0 (baseline)...")
    df_v0 = _simulate_all(df, bars_cache, sim_v0, 'V0')
    sel_v0, cutoffs, mults, params = _run_pipeline(df_v0)
    print(f"V0 selected: {len(sel_v0)} trades")
    print(f"Mults: { {q: round(v,3) for q,v in mults.items()} }")

    # Build quintile map for all rows (not just selected)
    df_v0['_composite'] = composite_score(df_v0, params)
    df_v0['_quintile'] = assign_quintile(df_v0['_composite'], cutoffs)
    qmap = dict(zip(
        df_v0.apply(lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1),
        df_v0['_quintile']
    ))

    # Simulate variants
    print("\nSimulating SB (always-stay + breakeven)...")
    df_sb = _simulate_all(df, bars_cache, sim_stay_be, 'SB')
    sel_sb, _, _, _ = _run_pipeline(df_sb)

    print("Simulating SB_Q3+ (stay only if Q3+, else V0)...")
    df_sbq = _simulate_all(df, bars_cache, sim_stay_be_q3plus, 'SB_Q3', quintile_map=qmap)
    sel_sbq, _, _, _ = _run_pipeline(df_sbq)

    print("Simulating SB_Q4Q5 (stay only if Q4/Q5, else V0)...")
    df_sb45 = _simulate_all(df, bars_cache, sim_stay_be_q4q5, 'SB_Q4Q5', quintile_map=qmap)
    sel_sb45, _, _, _ = _run_pipeline(df_sb45)

    print("Simulating SB_Q5 (stay only if Q5, else V0)...")
    df_sb5 = _simulate_all(df, bars_cache, sim_stay_be_q5only, 'SB_Q5', quintile_map=qmap)
    sel_sb5, _, _, _ = _run_pipeline(df_sb5)

    # Per-split metrics
    slices = [
        ('TRAIN',  '2025-01-01', TRAIN_END),
        ('VAL',    '2025-07-01', VAL_END),
        ('HOQ1+',  '2026-01-01', '2030-12-31'),
        ('FULL',   '2025-01-01', '2030-12-31'),
    ]

    print(f"\n{'='*100}")
    print("  WALK-FORWARD OOS VALIDATION")
    print(f"{'='*100}")
    print(f"{'Slice':<8} {'Variant':<10} {'P&L':>12} {'Δ vs V0':>11} "
          f"{'Max DD':>10} {'Calmar':>8} {'Neg':>4} {'Worst Day':>11}")
    print('-' * 90)

    all_results = {}
    for slice_name, lo, hi in slices:
        v0_m = _metrics_for_slice(sel_v0, lo, hi, 'V0')
        sb_m = _metrics_for_slice(sel_sb, lo, hi, 'SB')
        sbq_m = _metrics_for_slice(sel_sbq, lo, hi, 'SB_Q3+')
        sb45_m = _metrics_for_slice(sel_sb45, lo, hi, 'SB_Q4Q5')
        sb5_m = _metrics_for_slice(sel_sb5, lo, hi, 'SB_Q5')
        all_results[slice_name] = (v0_m, sb_m, sbq_m, sb45_m, sb5_m)
        for m in (v0_m, sb_m, sbq_m, sb45_m, sb5_m):
            delta = m['pnl'] - v0_m['pnl']
            print(f"{slice_name:<8} {m['label']:<10} "
                  f"${m['pnl']:>+10,.0f}  "
                  f"${delta:>+9,.0f}  "
                  f"${m['max_dd']:>+7,.0f}  "
                  f"{m['calmar']:>6.2f}x  "
                  f"{m['neg_months']:>3}  "
                  f"${m['worst_day']:>+9,.0f}")
        print('-' * 90)

    # Ship decision rubric
    print(f"\n{'='*100}")
    print("  SHIP DECISION RUBRIC  —  both VAL and HOQ1+ must show Δ ≥ $5K")
    print(f"{'='*100}")
    variant_names = ['SB', 'SB_Q3+', 'SB_Q4Q5', 'SB_Q5']
    for i, name in enumerate(variant_names, start=1):
        v_val = all_results['VAL'][i]
        v_hoq = all_results['HOQ1+'][i]
        v0_val = all_results['VAL'][0]
        v0_hoq = all_results['HOQ1+'][0]
        d_val = v_val['pnl'] - v0_val['pnl']
        d_hoq = v_hoq['pnl'] - v0_hoq['pnl']
        both_pass = d_val > 5000 and d_hoq > 5000
        print(f"\n  {name}:")
        print(f"    VAL Δ:   ${d_val:>+9,.0f}  "
              f"{'✓ POS' if d_val > 5000 else '✗ NEG or <$5K'}")
        print(f"    HOQ1+ Δ: ${d_hoq:>+9,.0f}  "
              f"{'✓ POS' if d_hoq > 5000 else '✗ NEG or <$5K'}")
        print(f"    → {'✓✓ SHIP CANDIDATE' if both_pass else '⚠ FAIL'}")

    # DD check — all variants vs V0 on FULL
    print(f"\n  DD check (FULL timeline):")
    v0_full = all_results['FULL'][0]
    print(f"    V0:      ${v0_full['max_dd']:>+8,.0f}  Calmar {v0_full['calmar']:.2f}x")
    for i, name in enumerate(variant_names, start=1):
        vf = all_results['FULL'][i]
        print(f"    {name:<8} ${vf['max_dd']:>+8,.0f}  Calmar {vf['calmar']:.2f}x  "
              f"(Δ DD ${vf['max_dd']-v0_full['max_dd']:+,.0f})")

    # Exit reason breakdowns
    print(f"\n  Exit reason breakdown (FULL):")
    for label, sel in [('V0', sel_v0), ('SB', sel_sb), ('SB_Q3+', sel_sbq),
                        ('SB_Q4Q5', sel_sb45), ('SB_Q5', sel_sb5)]:
        counts = sel['exit_reason'].value_counts()
        total = len(sel)
        print(f"    {label}:")
        for r, c in counts.items():
            print(f"      {r:<15} {c:>4} ({c/total*100:5.1f}%)")

    # Tail-dependence check: is the edge concentrated in a few huge wins?
    # If removing top-N winners collapses the lift, the rule is fragile.
    print(f"\n{'='*100}")
    print("  TAIL-DEPENDENCE CHECK  —  remove top-N winning trades per variant")
    print("  If SB's lift collapses faster than V0's does, SB is fragile.")
    print(f"{'='*100}")
    for slice_name in ['VAL', 'HOQ1+', 'FULL']:
        lo = {'VAL': '2025-07-01', 'HOQ1+': '2026-01-01', 'FULL': '2025-01-01'}[slice_name]
        hi = {'VAL': VAL_END, 'HOQ1+': '2030-12-31', 'FULL': '2030-12-31'}[slice_name]

        sels = {
            'V0':      sel_v0[(sel_v0['date'] >= lo) & (sel_v0['date'] <= hi)],
            'SB':      sel_sb[(sel_sb['date'] >= lo) & (sel_sb['date'] <= hi)],
            'SB_Q3+':  sel_sbq[(sel_sbq['date'] >= lo) & (sel_sbq['date'] <= hi)],
            'SB_Q4Q5': sel_sb45[(sel_sb45['date'] >= lo) & (sel_sb45['date'] <= hi)],
            'SB_Q5':   sel_sb5[(sel_sb5['date'] >= lo) & (sel_sb5['date'] <= hi)],
        }

        print(f"\n  Slice: {slice_name}")
        hdr = '  ' + f"{'Top-N rmvd':<13}"
        for nm in sels: hdr += f" {nm:>9}"
        hdr += f"  {'Δ Q5 vs V0':>12}"
        print(hdr)
        print('  ' + '-' * 90)
        for n in [0, 1, 3, 5, 10, 20]:
            line = f"  rm top {n:<6}"
            sums = {}
            for nm, sel in sels.items():
                pnls = sel.sort_values('_sized_pnl', ascending=False)['_sized_pnl'].iloc[n:]
                sums[nm] = pnls.sum()
                line += f" ${sums[nm]:>+8,.0f}"
            delta_q5 = sums['SB_Q5'] - sums['V0']
            line += f"  ${delta_q5:>+10,.0f}"
            print(line)

    # Show the top 10 stay-winner trades specifically (what we're depending on)
    print(f"\n{'='*100}")
    print("  TOP 10 SB TRADES by sized P&L (these are what the lift depends on)")
    print(f"{'='*100}")
    top = sel_sb.sort_values('_sized_pnl', ascending=False).head(10).copy()
    # Compare each to the V0 outcome on the same (symbol, date)
    v0_map = sel_v0.set_index(['symbol', 'date'])['_sized_pnl']
    top_cmp = []
    for _, r in top.iterrows():
        k = (r['symbol'], r['date'])
        v0_pnl = v0_map.get(k, 0)
        top_cmp.append({
            'symbol': r['symbol'],
            'date': r['date'].strftime('%Y-%m-%d'),
            'quintile': r['_quintile'],
            'SB_pnl': r['_sized_pnl'],
            'V0_pnl': v0_pnl,
            'delta': r['_sized_pnl'] - v0_pnl,
            'exit_reason': r['exit_reason'],
        })
    cmp = pd.DataFrame(top_cmp)
    pd.set_option('display.width', 220)
    pd.set_option('display.float_format', '{:,.0f}'.format)
    print(cmp.to_string(index=False))

    print("\nDone.")


if __name__ == '__main__':
    main()
