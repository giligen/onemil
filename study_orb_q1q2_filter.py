"""Q1/Q2 filter — quantify the effect of skipping bottom-quintile ORB trades.

Three variants evaluated end-to-end:
  V0          : shipped baseline (take all 5 quintiles, sized adaptively).
  V0_Q3plus   : filter to Q3/Q4/Q5 only at selection time; Q1/Q2 setups dropped.
                Remaining slots in top-4 may be refilled by next-best Q3+ candidate.
  V0_Q4Q5     : even stricter — only Q4/Q5 trades.

Metrics:
  - P&L per split (TRAIN / VAL / HOQ1+ / FULL)
  - Continuous max drawdown
  - Calmar
  - Red months
  - Trade count
  - Slot utilization: how many days used 0/1/2/3/4 slots
  - Q1/Q2 contribution (what we're giving up / keeping)

Ship rubric (both VAL and HOQ1+ must improve Δ ≥ +$1K with DD no worse):
  - Q1/Q2 is already sized down 0.25-0.5x — lift will be modest at best
  - Key bet: DD improvement and/or operational simplicity
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import timedelta
from typing import Set

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
    if rs <= 0: return float(bars.iloc[-1]['close'])
    trig = entry + 1.5 * rs; lock = entry + 1.0 * rs
    stop = rl; armed = False
    post = bars[bars['timestamp'] >= et].reset_index(drop=True)
    for _, b in post.iloc[1:].iterrows():
        h = float(b['high']); lo = float(b['low'])
        if not armed and h >= trig: armed = True; stop = max(stop, lock)
        if lo <= stop: return stop * (1 - EXIT_SLIP_BPS/10000)
    return float(post.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000)


def _simulate_v0(df, bars_cache):
    pnls, pcts = [], []
    for _, row in df.reset_index(drop=True).iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct']); continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct']); continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) &
                          (bars['timestamp'] < range_end)]
        if len(range_bars) < 5:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct']); continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh: entry_ts = b['timestamp']; break
        if entry_ts is None:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct']); continue
        entry_p = float(row['entry_price'])
        shares = max(1, int(OLD_POS / entry_p))
        exit_p = sim_v0(bars, entry_p, rh, rl, entry_ts)
        pnls.append((exit_p - entry_p) * shares)
        pcts.append((exit_p - entry_p) / entry_p * 100)
    out = df.reset_index(drop=True).copy()
    out['pnl'] = pnls; out['pnl_pct'] = pcts
    return out


def _run_pipeline(df_v0, allowed_quintiles: Set[str]):
    """Standard pipeline but restrict selection to `allowed_quintiles`.

    IMPORTANT: quintile cutoffs + mults still fit on ALL TRAIN trades (not just
    the filtered ones). Otherwise we'd overfit the cutoffs to a shrunken sample.
    """
    df = df_v0.copy()
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

    # The filter step: drop setups outside allowed_quintiles BEFORE top-K selection
    kept = kept[kept['_quintile'].isin(allowed_quintiles)]

    sel_rows = []
    day_slot_counts = []  # (date, slots_used)
    for day, dg in kept.groupby('date'):
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
        day_slot_counts.append((day, len(today)))
    sel = pd.DataFrame(sel_rows)
    if len(sel) == 0:
        return sel, mults, pd.DataFrame(day_slot_counts, columns=['date', 'slots'])
    sel['_sized_pnl'] = sel.apply(lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    sel['date'] = pd.to_datetime(sel['date'])
    slots_df = pd.DataFrame(day_slot_counts, columns=['date', 'slots'])
    slots_df['date'] = pd.to_datetime(slots_df['date'])
    return sel, mults, slots_df


def _calmar(pnls):
    cum = pnls.cumsum()
    peak = -1e18; mdd = 0.0
    for v in cum:
        peak = max(peak, v); mdd = min(mdd, v - peak)
    tp = float(pnls.sum())
    return tp, float(mdd), (tp / abs(mdd) if mdd < 0 else float('inf'))


def _metrics(sel, lo, hi, label):
    sub = sel[(sel['date'] >= lo) & (sel['date'] <= hi)].copy()
    if len(sub) == 0:
        return {'label': label, 'trades': 0, 'pnl': 0, 'max_dd': 0,
                 'calmar': 0, 'neg_months': 0, 'worst_day': 0}
    daily = sub.groupby('date').agg(pnl=('_sized_pnl', 'sum')).reset_index().sort_values('date').reset_index(drop=True)
    tp, mdd, cal = _calmar(daily['pnl'])
    daily['month'] = daily['date'].dt.to_period('M')
    monthly = daily.groupby('month')['pnl'].sum()
    return {
        'label': label, 'trades': len(sub), 'pnl': tp,
        'max_dd': mdd, 'calmar': cal,
        'neg_months': int((monthly < 0).sum()),
        'worst_day': float(daily['pnl'].min()),
        'trading_days': len(daily),
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

    print("Re-simulating V0 exit for all trades...")
    df_v0 = _simulate_v0(df, bars_cache)

    variants = [
        ('V0 (all quintiles)',     {'Q1', 'Q2', 'Q3', 'Q4', 'Q5'}),
        ('V0_noQ1',                {'Q2', 'Q3', 'Q4', 'Q5'}),
        ('V0_Q3plus',              {'Q3', 'Q4', 'Q5'}),
        ('V0_Q4Q5',                {'Q4', 'Q5'}),
        ('V0_Q5only',              {'Q5'}),
    ]

    results = {}
    for label, qs in variants:
        sel, mults, slots = _run_pipeline(df_v0, qs)
        results[label] = (sel, mults, slots)
        print(f"\n{label}: {len(sel)} trades  mults="
              f"{ {q: round(v,2) for q,v in mults.items() if q in qs} }")

    # Per-split metrics
    slices = [
        ('TRAIN',  '2025-01-01', TRAIN_END),
        ('VAL',    '2025-07-01', VAL_END),
        ('HOQ1+',  '2026-01-01', '2030-12-31'),
        ('FULL',   '2025-01-01', '2030-12-31'),
    ]

    print(f"\n{'='*110}")
    print("  WALK-FORWARD METRICS (all variants use the same TRAIN-fit cutoffs + mults)")
    print(f"{'='*110}")
    print(f"{'Slice':<8} {'Variant':<22} {'Trades':>7} {'P&L':>12} {'Δ vs V0':>11} "
          f"{'Max DD':>10} {'Calmar':>8} {'Neg':>5} {'Worst Day':>11}")
    print('-' * 110)
    for slice_name, lo, hi in slices:
        v0_m = _metrics(results['V0 (all quintiles)'][0], lo, hi, 'V0')
        ms = []
        for label, _ in variants:
            ms.append(_metrics(results[label][0], lo, hi, label))
        for m in ms:
            delta = m['pnl'] - v0_m['pnl']
            print(f"{slice_name:<8} {m['label']:<22} "
                  f"{m['trades']:>7}  "
                  f"${m['pnl']:>+9,.0f}  "
                  f"${delta:>+9,.0f}  "
                  f"${m['max_dd']:>+7,.0f}  "
                  f"{m['calmar']:>6.2f}x  "
                  f"{m['neg_months']:>3}   "
                  f"${m['worst_day']:>+9,.0f}")
        print('-' * 110)

    # Slot utilization — how many slots filled per day
    print(f"\n{'='*110}")
    print("  SLOT UTILIZATION per day  (max 4 slots)")
    print(f"{'='*110}")
    for label, _ in variants:
        slots = results[label][2]
        if len(slots) == 0:
            print(f"  {label}: no trading days")
            continue
        print(f"\n  {label}:")
        for n_slots in range(5):
            count = (slots['slots'] == n_slots).sum()
            pct = count / len(slots) * 100
            print(f"    {n_slots} slot(s):  {count:>4} days ({pct:>5.1f}%)")
        avg_slots = slots['slots'].mean()
        print(f"    avg slots used per day: {avg_slots:.2f}")

    # Q1/Q2 contribution analysis — how much P&L do we lose by filtering them out?
    print(f"\n{'='*110}")
    print("  Q1/Q2 CONTRIBUTION — what we're dropping by filtering")
    print(f"{'='*110}")
    v0_sel = results['V0 (all quintiles)'][0]
    for slice_name, lo, hi in slices:
        sub = v0_sel[(v0_sel['date'] >= lo) & (v0_sel['date'] <= hi)]
        total = sub['_sized_pnl'].sum()
        print(f"\n  {slice_name}:")
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            q_sub = sub[sub['_quintile'] == q]
            if len(q_sub) == 0: continue
            q_pnl = q_sub['_sized_pnl'].sum()
            q_pct = q_pnl / total * 100 if total else 0
            print(f"    {q}: {len(q_sub):>3} trades  "
                  f"P&L ${q_pnl:>+8,.0f}  "
                  f"({q_pct:>+5.1f}% of slice)  "
                  f"avg/trade ${q_pnl/len(q_sub):>+6,.0f}")

    # Ship decision
    print(f"\n{'='*110}")
    print("  SHIP DECISION")
    print(f"{'='*110}")
    for label, _ in variants[1:]:  # skip V0 itself
        v_m_val = _metrics(results[label][0], '2025-07-01', VAL_END, label)
        v_m_hoq = _metrics(results[label][0], '2026-01-01', '2030-12-31', label)
        v0_m_val = _metrics(results['V0 (all quintiles)'][0], '2025-07-01', VAL_END, 'V0')
        v0_m_hoq = _metrics(results['V0 (all quintiles)'][0], '2026-01-01', '2030-12-31', 'V0')
        d_val = v_m_val['pnl'] - v0_m_val['pnl']
        d_hoq = v_m_hoq['pnl'] - v0_m_hoq['pnl']
        d_val_dd = v_m_val['max_dd'] - v0_m_val['max_dd']
        d_hoq_dd = v_m_hoq['max_dd'] - v0_m_hoq['max_dd']
        both_pos = d_val > 1000 and d_hoq > 1000
        dd_nonworse = d_val_dd >= -500 and d_hoq_dd >= -500
        print(f"\n  {label}:")
        print(f"    VAL   Δ P&L ${d_val:>+8,.0f}  Δ DD ${d_val_dd:>+8,.0f}")
        print(f"    HOQ1+ Δ P&L ${d_hoq:>+8,.0f}  Δ DD ${d_hoq_dd:>+8,.0f}")
        if both_pos and dd_nonworse:
            print(f"    → ✓ SHIP CANDIDATE (both OOS positive + DD not worse)")
        elif d_val + d_hoq > 0 and dd_nonworse:
            print(f"    → ⚠ marginal; net positive but one split weak")
        else:
            print(f"    → ✗ does not pass ship bar")

    print("\nDone.")


if __name__ == '__main__':
    main()
