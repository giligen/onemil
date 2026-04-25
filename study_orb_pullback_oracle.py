"""Diagnostic: what is the upper bound of edge from a "stay past +1R" decision?

At the moment V0 would exit at +1R (after arming at +1.5R), we simulate two
alternative outcomes:
  EXIT: take V0's +1R lock exit (baseline).
  STAY: move stop to BREAKEVEN (entry), ride to EOD close. Exit early if
        breakeven is hit.

For every BT trade that actually pulled back to +1R after arming, compute:
  - EXIT P&L (what V0 realized)
  - STAY P&L (what staying would have realized with BE-protected ride-to-EOD)
  - Oracle = max(EXIT, STAY) — the cap on edge from perfect classification

Aggregate:
  - % of pullback trades where STAY > EXIT (the pool of convertible wins)
  - Average $ per convertible trade
  - Total $ available across full timeline (upper bound — an oracle)
  - Pre-pullback features (peak MFE, bars since entry, etc.) to inform a
    future classifier

If fewer than ~45% of pullbacks convert, V0's exit-at-+1R is statistically
optimal and no classifier will save us. If 55%+ convert OR the winners
are much larger than the losers, a classifier is worth building.
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import timedelta
from typing import Optional, Tuple

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


def _session_open_timestamp(bars):
    if bars.empty: return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning): return morning.iloc[0]['timestamp']
    return None


def simulate_pullback_decision(bars, entry_price, range_high, range_low, entry_time
                                ) -> Optional[dict]:
    """Walk bars post-entry. Identify arm bar (+1.5R), pullback bar (+1R after arm),
    and simulate both EXIT and STAY outcomes. Return detailed dict or None if the
    trade never armed OR never pulled back (V0 would have ridden to EOD = no decision).
    """
    rs = range_high - range_low
    if rs <= 0:
        return None
    trig_lvl = entry_price + 1.5 * rs
    lock_lvl = entry_price + 1.0 * rs

    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    if len(post) < 2:
        return None

    # Walk bars: find arm, then pullback-to-+1R
    armed = False
    arm_bar_idx = None
    arm_timestamp = None
    pullback_bar_idx = None
    peak_high_pre_pullback = 0.0  # max high from entry to pullback moment
    cum_vol_pre_pullback = 0
    bars_since_entry_at_pullback = None

    walk_bars = post.iloc[1:].reset_index(drop=True)  # skip entry bar

    for i, row in walk_bars.iterrows():
        h = float(row['high']); lo = float(row['low'])
        vol = int(row.get('volume', 0) or 0)
        if h > peak_high_pre_pullback:
            peak_high_pre_pullback = h
        cum_vol_pre_pullback += vol

        if not armed and h >= trig_lvl:
            armed = True
            arm_bar_idx = i
            arm_timestamp = row['timestamp']
            # Continue: do we also pull back on the same arm bar?

        if armed and lo <= lock_lvl:
            pullback_bar_idx = i
            bars_since_entry_at_pullback = i + 1
            break

    if not armed:
        # Never armed — V0 would have stopped at range_low or EOD. Not our case.
        return {'category': 'never_armed',
                'eod_close_r': (float(post.iloc[-1]['close']) - entry_price) / rs,
                'day_max_r': (float(post['high'].max()) - entry_price) / rs}

    if pullback_bar_idx is None:
        # Armed but never pulled back to +1R — V0 rides to EOD. Not our case.
        return {'category': 'armed_no_pullback',
                'eod_close_r': (float(post.iloc[-1]['close']) - entry_price) / rs,
                'day_max_r': (float(post['high'].max()) - entry_price) / rs}

    # We have a pullback to +1R after arm. Compute EXIT and STAY outcomes.

    # EXIT = V0: exit at +1R lock price
    exit_price = lock_lvl * (1 - EXIT_SLIP_BPS/10000)

    # STAY = move stop to breakeven (entry), ride to EOD
    # From the bar AFTER pullback, walk forward. Exit at breakeven OR EOD.
    # NOTE: on the pullback bar itself, the +1R exit already happened for
    # V0. For STAY, we pretend we didn't exit — so continue from that bar.
    # But the pullback bar also had bar_low <= lock_lvl; we need to check
    # whether bar_low <= entry_price on that bar (would have hit breakeven).

    stay_exit_price = None
    stay_exit_reason = None

    # Start from pullback bar (same bar — since we're assuming we stayed,
    # the stop is now at breakeven; did that bar's low hit breakeven?)
    remaining = walk_bars.iloc[pullback_bar_idx:]
    # Track forward peak for reporting
    forward_peak_high = 0.0

    for j, row in remaining.iterrows():
        h = float(row['high']); lo = float(row['low'])
        if h > forward_peak_high:
            forward_peak_high = h
        if lo <= entry_price:
            stay_exit_price = entry_price * (1 - EXIT_SLIP_BPS/10000)
            stay_exit_reason = 'breakeven'
            break
    if stay_exit_price is None:
        stay_exit_price = float(remaining.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000)
        stay_exit_reason = 'eod'

    exit_pnl_per_share = exit_price - entry_price
    stay_pnl_per_share = stay_exit_price - entry_price

    return {
        'category': 'armed_and_pulled_back',
        'exit_pnl_per_share': exit_pnl_per_share,
        'stay_pnl_per_share': stay_pnl_per_share,
        'exit_r': exit_pnl_per_share / rs,
        'stay_r': stay_pnl_per_share / rs,
        'stay_exit_reason': stay_exit_reason,
        'stay_wins': stay_pnl_per_share > exit_pnl_per_share,
        'range_size': rs,
        'peak_mfe_pre_pullback_r': (peak_high_pre_pullback - entry_price) / rs,
        'bars_since_entry_at_pullback': bars_since_entry_at_pullback,
        'cum_vol_pre_pullback': cum_vol_pre_pullback,
        'forward_peak_r': (forward_peak_high - entry_price) / rs,
        'eod_close_r': (float(post.iloc[-1]['close']) - entry_price) / rs,
    }


def _run_pipeline(df_with_pnl):
    """Standard defended pipeline — select the same trade set V0 sees."""
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
    sel['date'] = pd.to_datetime(sel['date'])
    return sel, mults


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

    # Build selected trade set (same as V0 pipeline)
    print("Building V0 trade set...")
    sel, mults = _run_pipeline(df)
    print(f"Selected trades: {len(sel)}")

    # Replay each trade to identify pullback decision point
    print("Replaying pullback decisions...")
    results = []
    for _, row in sel.iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None: continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) &
                          (bars['timestamp'] < range_end)]
        if len(range_bars) < 5: continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh: entry_ts = b['timestamp']; break
        if entry_ts is None: continue
        entry_p = float(row['entry_price'])
        res = simulate_pullback_decision(bars, entry_p, rh, rl, entry_ts)
        if res is None: continue
        res.update({
            'symbol': row['symbol'],
            'date': row['date'].strftime('%Y-%m-%d'),
            'entry_price': entry_p,
            'quintile': row.get('_quintile', 'N/A'),
            'adaptive_mult': mults.get(row.get('_quintile', 'Q3'), 1.0),
            'rp_position': row['_rp_position'],
        })
        results.append(res)

    rdf = pd.DataFrame(results)
    print(f"\nReplayed {len(rdf)} trades")

    # Category breakdown
    print(f"\n{'='*78}")
    print(f"  Trade category breakdown (decision surface)")
    print(f"{'='*78}")
    cat_counts = rdf['category'].value_counts()
    total = len(rdf)
    for cat, count in cat_counts.items():
        print(f"  {cat:<25}: {count:>4} ({count/total*100:.1f}%)")

    # Focus on the decision trades
    dec = rdf[rdf['category'] == 'armed_and_pulled_back'].copy()
    if len(dec) == 0:
        print("\nNo decision trades! Something wrong with simulator.")
        return
    dec['stay_wins'] = dec['stay_wins'].astype(bool)

    print(f"\n{'='*78}")
    print(f"  DECISION TRADES  ({len(dec)} of {total}, {len(dec)/total*100:.1f}%)")
    print(f"  These are the trades where V0 exits at +1R — the pool where a")
    print(f"  'stay' decision matters.")
    print(f"{'='*78}")

    # Core outcome: stay_wins rate
    stay_wins_count = dec['stay_wins'].sum()
    stay_wins_pct = stay_wins_count / len(dec) * 100
    print(f"\n  Stay > Exit:     {stay_wins_count}/{len(dec)} = {stay_wins_pct:.1f}%")
    print(f"  Exit >= Stay:    {len(dec)-stay_wins_count}/{len(dec)} = {(1-stay_wins_count/len(dec))*100:.1f}%")

    # Per-trade $ economics
    dec['exit_dollar'] = dec['exit_pnl_per_share'] * dec['rp_position'] / dec['entry_price']
    dec['stay_dollar'] = dec['stay_pnl_per_share'] * dec['rp_position'] / dec['entry_price']
    dec['delta_dollar'] = dec['stay_dollar'] - dec['exit_dollar']

    print(f"\n  Per-trade $ outcomes (risk-parity sized, no quintile mult):")
    print(f"    EXIT (V0 +1R):        avg ${dec['exit_dollar'].mean():+,.0f}   "
          f"sum ${dec['exit_dollar'].sum():+,.0f}")
    print(f"    STAY (BE + EOD):      avg ${dec['stay_dollar'].mean():+,.0f}   "
          f"sum ${dec['stay_dollar'].sum():+,.0f}")
    print(f"    Δ if always STAY:     avg ${dec['delta_dollar'].mean():+,.0f}   "
          f"sum ${dec['delta_dollar'].sum():+,.0f}")

    # Oracle: max(exit, stay) per trade
    dec['oracle_dollar'] = dec[['exit_dollar', 'stay_dollar']].max(axis=1)
    oracle_lift = dec['oracle_dollar'].sum() - dec['exit_dollar'].sum()
    print(f"\n  ORACLE (perfect classifier chooses best per trade):")
    print(f"    Oracle sum:           ${dec['oracle_dollar'].sum():+,.0f}")
    print(f"    Oracle lift vs V0:    ${oracle_lift:+,.0f}")
    print(f"    (this is the upper bound; realistic classifier captures ~40-70%)")

    # Distribution of stay outcomes
    print(f"\n  Stay P&L distribution:")
    for pct, label in [(10, '10th'), (25, '25th'), (50, 'median'),
                        (75, '75th'), (90, '90th'), (99, '99th')]:
        print(f"    {label:<7}: ${dec['stay_dollar'].quantile(pct/100):>+8,.0f}")

    # Split stays into winners and losers vs exit
    wins = dec[dec['stay_wins']].copy()
    losses = dec[~dec['stay_wins']].copy()
    print(f"\n  When STAY wins ({len(wins)} trades):")
    print(f"    Avg Δ per trade:      ${wins['delta_dollar'].mean():+,.0f}")
    print(f"    Sum of Δ:             ${wins['delta_dollar'].sum():+,.0f}")
    print(f"    Forward peak R:        {wins['forward_peak_r'].mean():.2f}R avg, "
          f"{wins['forward_peak_r'].max():.1f}R max")
    print(f"  When STAY loses ({len(losses)} trades):")
    print(f"    Avg Δ per trade:      ${losses['delta_dollar'].mean():+,.0f}")
    print(f"    Sum of Δ:             ${losses['delta_dollar'].sum():+,.0f}")
    print(f"    Stay exit via BE:      "
          f"{(losses['stay_exit_reason']=='breakeven').sum()}/{len(losses)} "
          f"({(losses['stay_exit_reason']=='breakeven').mean()*100:.0f}%)")

    # Stratification by pre-pullback signals
    print(f"\n{'='*78}")
    print(f"  STAY-WIN RATE stratified by pre-pullback features")
    print(f"  (if rate > ~55%, staying with that signal is profitable)")
    print(f"{'='*78}")

    # Peak MFE pre-pullback (did trade run high before pulling back?)
    print(f"\n  By peak_mfe_pre_pullback_r:")
    for lo, hi, label in [(1.5, 2.0, '1.5-2.0R'), (2.0, 2.5, '2.0-2.5R'),
                           (2.5, 3.0, '2.5-3.0R'), (3.0, 4.0, '3.0-4.0R'),
                           (4.0, 100, '4.0R+')]:
        sub = dec[(dec['peak_mfe_pre_pullback_r'] >= lo) &
                   (dec['peak_mfe_pre_pullback_r'] < hi)]
        if not len(sub): continue
        wr = sub['stay_wins'].mean() * 100
        delta_sum = sub['delta_dollar'].sum()
        print(f"    {label:<10}: {int(sub['stay_wins'].sum()):>3}/{len(sub):<3} "
              f"({wr:.0f}%)  Δsum ${delta_sum:+,.0f}")

    # Bars since entry at pullback
    print(f"\n  By bars_since_entry_at_pullback:")
    for lo, hi, label in [(1, 2, '1 bar'), (2, 4, '2-3 bars'),
                           (4, 8, '4-7 bars'), (8, 15, '8-14 bars'),
                           (15, 100, '15+ bars')]:
        sub = dec[(dec['bars_since_entry_at_pullback'] >= lo) &
                   (dec['bars_since_entry_at_pullback'] < hi)]
        if not len(sub): continue
        wr = sub['stay_wins'].mean() * 100
        delta_sum = sub['delta_dollar'].sum()
        print(f"    {label:<10}: {int(sub['stay_wins'].sum()):>3}/{len(sub):<3} "
              f"({wr:.0f}%)  Δsum ${delta_sum:+,.0f}")

    # Quintile
    print(f"\n  By quintile:")
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = dec[dec['quintile'] == q]
        if not len(sub): continue
        wr = sub['stay_wins'].mean() * 100
        delta_sum = sub['delta_dollar'].sum()
        print(f"    {q}: {int(sub['stay_wins'].sum()):>3}/{len(sub):<3} "
              f"({wr:.0f}%)  Δsum ${delta_sum:+,.0f}")

    # Joint: peak MFE >= 2.5R AND bars >= 8 (hypothetical: "it had momentum")
    print(f"\n  Joint: peak_mfe >= 2.5R AND bars_since_entry >= 8")
    mask = (dec['peak_mfe_pre_pullback_r'] >= 2.5) & (dec['bars_since_entry_at_pullback'] >= 8)
    sub = dec[mask]
    if len(sub):
        print(f"    {int(sub['stay_wins'].sum())}/{len(sub)} "
              f"({sub['stay_wins'].mean()*100:.0f}% stay-win), "
              f"Δsum ${sub['delta_dollar'].sum():+,.0f}  "
              f"({len(sub)/len(dec)*100:.1f}% of decision trades)")

    # Save per-trade detail for future classifier
    rdf.to_csv('analysis_results/orb_pullback_oracle_trades.csv', index=False)
    print(f"\nSaved analysis_results/orb_pullback_oracle_trades.csv "
          f"({len(rdf)} rows, {len(dec)} decision trades)")


if __name__ == '__main__':
    main()
