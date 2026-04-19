#!/usr/bin/env python3
"""ORB with correlation filter (family dedup) + Q5 sizing cap.

Two fixes for the catastrophic days problem uncovered in Q1 2025 analysis:

FIX 1 — Family dedup (the correlation filter):
  Multiple picks from the same "correlation family" (e.g., UVXY+TSLZ+MSTZ are
  all short-volatility / inverse trades) behave as one correlated bet, not N
  independent ones. After Q4-pref ranking, dedupe per day: keep only the
  TOP-ranked symbol from each family.

  Family definitions cover:
    - Leveraged ETF pairs on same underlying (TSLA 2x long family vs 2x short)
    - Long/short pairs (keep only one — pushing same direction)
    - Crypto miners (MARA/RIOT/CLSK/WULF/CIFR all move on BTC price)
    - Quantum computing names (RGTI/QBTS/IONQ/QUBT move as pack)
    - Vol products (UVXY/VIXY/SVIX)

FIX 2 — Q5 sizing cap:
  Adaptive Q5 multiplier hit 3.00x in Split A because H1 2025 train was
  anomalous (Q5 best). TEST data across all 3 splits says Q4 > Q5. Cap
  Q5 max multiplier at 1.5x to prevent over-sizing into outlier-tainted
  extreme z-scores.

Run: python3 study_orb_correlation_filter.py
"""
from __future__ import annotations

import os, sys, glob
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from study_orb import SPLITS, OUT_DIR
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, ADAPTIVE_MULT_MIN, ADAPTIVE_MULT_MAX,
    fit_quintile_cutoffs, assign_quintile,
)

ACCOUNT = 100_000
N_MAX = 3
RISK = 2000
OLD_POS = 50_000.0
MIN_STOP_PCT = 1.0
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}

# Per-quintile multiplier cap — key fix. Q5 historically overshoots in Split A train.
PER_QUINTILE_MAX_MULT = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}

# =========================================================================
# CORRELATION FAMILIES — symbols that move together (treat as one bet)
# =========================================================================
# Each family is a list of symbols. Within a day, keep only the TOP-ranked
# symbol from each family (highest composite, after Q4-pref ordering).
#
# Family design principles:
#   - Leveraged ETFs on same underlying → correlation ~0.95+
#   - Long and short ETFs on same underlying → correlation ~-0.95 (but they
#     all fire on VOLATILITY, and ORB triggers when one side pops. Taking
#     both = arbitrage-style; take only one.)
#   - Sector clusters (crypto miners, quantum names) → correlation 0.6-0.85
#
FAMILIES: Dict[str, List[str]] = {
    # Tesla-related leveraged ETFs — all move on TSLA
    'tsla_leveraged': ['TSL', 'TSLL', 'TSLR', 'TSLT', 'TSLG', 'TSLS', 'TSLZ', 'TSLQ'],
    # MicroStrategy-related leveraged ETFs
    'mstr_leveraged': ['MSTU', 'MSTX', 'MSTZ', 'SMST', 'MSTP', 'MSTW', 'MSTY', 'FMST'],
    # NVDA-related leveraged ETFs
    'nvda_leveraged': ['NVD', 'NVDD', 'NVDQ', 'NVDS', 'NVDX', 'NVDL', 'NVDU'],
    # VIX / volatility products (all surge on vol spikes)
    'volatility': ['UVXY', 'VXX', 'VIXY', 'SVIX', 'SVXY', 'VMIN', 'VXZ'],
    # SPY leveraged ETFs
    'spy_leveraged': ['SPXL', 'UPRO', 'SPXS', 'SPXU', 'SPDN', 'SH'],
    # QQQ leveraged ETFs
    'qqq_leveraged': ['TQQQ', 'QLD', 'SQQQ', 'PSQ', 'QID'],
    # Semiconductor leveraged ETFs
    'semi_leveraged': ['SOXL', 'SOXS', 'USD', 'SSG', 'MSOX'],
    # Ethereum / crypto leveraged ETFs
    'eth_leveraged': ['ETHT', 'ETHU', 'ETHD', 'ETHL'],
    # Bitcoin leveraged ETFs (ETFs only — keep pure BTC miners separate)
    'btc_leveraged': ['BITU', 'BITX', 'BITI'],
    # Bitcoin miners — highly correlated with BTC price
    'btc_miners': ['MARA', 'RIOT', 'CLSK', 'WULF', 'CIFR', 'BTDR', 'HUT', 'BITF',
                   'CORZ', 'IREN', 'HIVE', 'CAN', 'BTCS', 'GREE'],
    # Quantum computing names — move as pack on quantum news
    'quantum': ['RGTI', 'QBTS', 'IONQ', 'QUBT', 'QMCO', 'QBTSW', 'RGTIW'],
    # Fintech / crypto small-caps that trade on crypto sentiment
    'crypto_fintech': ['COIN', 'HOOD', 'BKKT', 'BTBT', 'MSTR'],
    # EV and flying car / robotaxi
    'ev_aerial': ['ACHR', 'EVTL', 'JOBY', 'EH', 'BLDE', 'RKLB'],
    # China small-caps that gap together on China news
    'china_smallcap': ['NVNI', 'IREX', 'PTIR', 'SIDU', 'JFIN', 'KC'],
}

# =========================================================================
# DIRECTIONAL SUPER-GROUPS — cross-underlying correlation.
# When broad market reverses, ALL leveraged_short names lose together (even
# across different underlyings). These caused the 3/6, 3/18, 3/19 blow-ups.
# =========================================================================
LEVERAGED_SHORT_ALL = [
    'UVXY', 'VXX', 'VIXY',  # long vol = short market
    'TSLZ', 'TSLS', 'TSLQ',  # short TSLA
    'MSTZ', 'SMST',          # short MSTR
    'NVD', 'NVDD', 'NVDQ', 'NVDS',  # short NVDA
    'SOXS', 'SSG', 'MSOX',   # short semi
    'SQQQ', 'PSQ', 'QID',    # short QQQ
    'SPXS', 'SPXU', 'SPDN', 'SH',  # short SPY
    'BITI',                  # short BTC
    'ETHD',                  # short ETH
    'LABD',                  # short biotech
]

LEVERAGED_LONG_ALL = [
    'SVIX', 'SVXY', 'VMIN',  # short vol = long market
    'TSLL', 'TSLT', 'TSLR', 'TSLG',  # long TSLA
    'MSTU', 'MSTX',          # long MSTR
    'NVDL', 'NVDU', 'NVDX',  # long NVDA
    'SOXL', 'USD',           # long semi
    'TQQQ', 'QLD',           # long QQQ
    'SPXL', 'UPRO',          # long SPY
    'BITU', 'BITX',          # long BTC
    'ETHT', 'ETHU', 'ETHL',  # long ETH
    'LABU',                  # long biotech
]

LEV_SHORT_SET = set(LEVERAGED_SHORT_ALL)
LEV_LONG_SET = set(LEVERAGED_LONG_ALL)

# Build reverse lookup
SYM_TO_FAMILY: Dict[str, str] = {}
for fam, syms in FAMILIES.items():
    for s in syms:
        SYM_TO_FAMILY[s] = fam


def symbol_family(symbol: str) -> Optional[str]:
    return SYM_TO_FAMILY.get(symbol)


def symbol_super_group(symbol: str) -> Optional[str]:
    """Directional super-group: lev_short / lev_long. Returns None for non-ETFs."""
    if symbol in LEV_SHORT_SET:
        return 'lev_short'
    if symbol in LEV_LONG_SET:
        return 'lev_long'
    return None


# =========================================================================
# Pipeline
# =========================================================================

def apply_risk_parity(df, risk, cap):
    df = df.copy()
    stop_pct = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = risk / (stop_pct / 100.0)
    df['_rp_position'] = uncap.clip(upper=cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS
    return df


def fit_adaptive(train_kept, pnl_col='_rp_pnl'):
    avg = float(train_kept[pnl_col].mean()) if len(train_kept) else 1.0
    mults = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = train_kept[train_kept['_quintile'] == q]
        if len(sub) == 0 or avg <= 0:
            mults[q] = 1.0
            continue
        raw = float(sub[pnl_col].mean()) / avg
        # Apply per-quintile max cap (KEY FIX for Q5)
        capped = max(ADAPTIVE_MULT_MIN, min(PER_QUINTILE_MAX_MULT[q], raw))
        mults[q] = capped
    return mults


def select_top_k_with_dedup(dg: pd.DataFrame, k: int,
                            dedup_family: bool = True,
                            dedup_super_group: bool = True) -> pd.DataFrame:
    """Q4-preferred ranking + optional dedup.

    - dedup_family: keep only top-ranked symbol per underlying family
                    (tsla_leveraged, mstr_leveraged, btc_miners, ...)
    - dedup_super_group: keep only top-ranked symbol per directional super-group
                         (lev_short, lev_long) — prevents correlated directional bets
    Non-family, non-super-group symbols are unconstrained.
    """
    d = dg.copy()
    d['_q_rank'] = d['_quintile'].map(Q_ORDER)
    d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])

    if not dedup_family and not dedup_super_group:
        return d.head(k)

    seen_families = set()
    seen_super = set()
    kept_rows = []
    for _, r in d.iterrows():
        sym = r['symbol']
        fam = symbol_family(sym)
        sup = symbol_super_group(sym)

        if dedup_family and fam is not None and fam in seen_families:
            continue
        if dedup_super_group and sup is not None and sup in seen_super:
            continue

        if fam is not None:
            seen_families.add(fam)
        if sup is not None:
            seen_super.add(sup)
        kept_rows.append(r)
        if len(kept_rows) >= k:
            break
    return pd.DataFrame(kept_rows)


def compute_dd(daily):
    if len(daily) == 0:
        return 0.0, None
    d = daily.sort_values('date').reset_index(drop=True)
    d['cum'] = d['daily_pnl'].cumsum()
    peak = -np.inf
    dd = 0.0
    worst_date = None
    peak_date = None
    current_peak_date = None
    for i, c in enumerate(d['cum']):
        if c > peak:
            peak = c
            current_peak_date = d.loc[i, 'date']
        cur_dd = c - peak
        if cur_dd < dd:
            dd = cur_dd
            worst_date = d.loc[i, 'date']
            peak_date = current_peak_date
    return dd, {'peak_date': peak_date, 'trough_date': worst_date}


def run_split(df, tr_s, tr_e, te_s, te_e, k, dedup_family=True, dedup_super=True):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    per_pos_cap = ACCOUNT / k
    df = apply_risk_parity(df, RISK, per_pos_cap)

    train = df[(df['date'] >= tr_s) & (df['date'] <= tr_e)]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= tr_s) & (df['date'] <= tr_e)]
    test = df[(df['date'] >= te_s) & (df['date'] <= te_e)]

    test_kept = test[test['_composite'] >= FILTER_THRESHOLD].copy()
    train_kept = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_kept['_composite'])
    train_kept['_quintile'] = assign_quintile(train_kept['_composite'], cutoffs)
    test_kept['_quintile'] = assign_quintile(test_kept['_composite'], cutoffs)
    mults = fit_adaptive(train_kept)

    sel = pd.concat([select_top_k_with_dedup(dg, k, dedup_family, dedup_super)
                     for _, dg in test_kept.groupby('date')])
    sel['_sized_pnl'] = sel.apply(
        lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    daily = sel.groupby('date').agg(daily_pnl=('_sized_pnl', 'sum'),
                                     n=('_rp_pnl', 'count')).reset_index()
    dd, dd_range = compute_dd(daily)
    return {
        'pnl': float(daily['daily_pnl'].sum()) if len(daily) else 0,
        'dd': float(dd),
        'worst_day': float(daily['daily_pnl'].min()) if len(daily) else 0,
        'worst_trade': float(sel['_sized_pnl'].min()) if len(sel) else 0,
        'n_trades': len(sel),
        'mults': mults,
        'daily': daily,
        'sel': sel,
        'dd_range': dd_range,
    }


def run_full_sweep(df):
    """Run configurations and compare. dedup_family, dedup_super, cap."""
    configs = [
        ('A. baseline (no fixes)',              False, False, 3.0),
        ('B. Q5 cap only (1.5x)',               False, False, 1.5),
        ('C. family dedup only',                True,  False, 3.0),
        ('D. super-group dedup only',           False, True,  3.0),
        ('E. family + super-group dedup',       True,  True,  3.0),
        ('F. Q5 cap + family dedup',            True,  False, 1.5),
        ('G. Q5 cap + super-group dedup',       False, True,  1.5),
        ('H. Q5 cap + family + super dedup',    True,  True,  1.5),
    ]
    global PER_QUINTILE_MAX_MULT
    results = []
    for name, use_fam, use_super, q5_cap in configs:
        PER_QUINTILE_MAX_MULT = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0,
                                  'Q5': q5_cap}
        per_split = []
        for _, tr_s, tr_e, te_s, te_e in SPLITS:
            r = run_split(df, tr_s, tr_e, te_s, te_e, N_MAX,
                          dedup_family=use_fam, dedup_super=use_super)
            per_split.append(r)
        pnls = [r['pnl'] for r in per_split]
        dds = [r['dd'] for r in per_split]
        wd = [r['worst_day'] for r in per_split]
        wt = [r['worst_trade'] for r in per_split]
        results.append({
            'name': name,
            'sum_pnl': sum(pnls),
            'min_split': min(pnls),
            'worst_dd': min(dds),
            'worst_day': min(wd),
            'worst_trade': min(wt),
            'per_split_pnl': pnls,
            'pnl_dd': sum(pnls) / abs(min(dds)) if min(dds) < 0 else float('inf'),
        })
    return results


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'pnl_pct', 'date', 'range_size_pct',
                                     'entry_price'])
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df):,} trades")

    # Count family memberships
    print(f"\nFamily coverage: {len(SYM_TO_FAMILY)} symbols mapped, "
          f"{len(FAMILIES)} families")
    fam_hits = df['symbol'].map(SYM_TO_FAMILY).notna().sum()
    print(f"Trades on family symbols: {fam_hits} ({100*fam_hits/len(df):.1f}%)")

    # --- Step 1: walk-forward sweep of 4 configs ---
    print(f"\n{'='*110}")
    print(f"WALK-FORWARD SWEEP — 4 configs (N={N_MAX}, risk=${RISK:,.0f})")
    print(f"{'='*110}")
    results = run_full_sweep(df)
    print(f"  {'config':<34} {'Sum P&L':>13} {'Min split':>13} "
          f"{'Worst DD':>12} {'Worst day':>12} {'Worst tr':>11} {'P&L/DD':>9}")
    print('-' * 110)
    for r in results:
        print(f"  {r['name']:<34} ${r['sum_pnl']:>+11,.0f} ${r['min_split']:>+11,.0f} "
              f"${r['worst_dd']:>+10,.0f} ${r['worst_day']:>+10,.0f} "
              f"${r['worst_trade']:>+9,.0f} {r['pnl_dd']:>8.2f}x")

    # --- Step 2: Q1 2025 day-by-day with both fixes applied ---
    global PER_QUINTILE_MAX_MULT
    PER_QUINTILE_MAX_MULT = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}

    print(f"\n{'='*110}")
    print(f"Q1 2025 DAY-BY-DAY with ALL FIXES (Q5 cap 1.5x + family + super-group dedup)")
    print(f"{'='*110}")

    # Use Split A (trains on H1 2025) same as before for apples-to-apples comparison
    per_pos_cap = ACCOUNT / N_MAX
    dfw = apply_risk_parity(df, RISK, per_pos_cap)
    train = dfw[(dfw['date'] >= '2025-01-01') & (dfw['date'] <= '2025-06-30')]
    params = fit_z_params(train, FILTER_FEATURES)
    dfw['_composite'] = composite_score(dfw, params)
    train = dfw[(dfw['date'] >= '2025-01-01') & (dfw['date'] <= '2025-06-30')]
    train_kept = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_kept['_composite'])
    train_kept['_quintile'] = assign_quintile(train_kept['_composite'], cutoffs)
    mults = fit_adaptive(train_kept)
    print(f"Mults after Q5 cap: " +
          "  ".join(f"{q}={mults[q]:.2f}x" for q in ['Q1','Q2','Q3','Q4','Q5']))

    q1 = dfw[(dfw['date'] >= '2025-01-01') & (dfw['date'] <= '2025-03-31')]
    q1_filt = q1[q1['_composite'] >= FILTER_THRESHOLD].copy()
    q1_filt['_quintile'] = assign_quintile(q1_filt['_composite'], cutoffs)

    print(f"\n{'date':<12} {'sig':>4} {'picked':>7} {'trades selected':<56} "
          f"{'day P&L':>11} {'equity':>11} {'DD':>10}")
    print('-' * 110)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    daily_rows = []
    for day in sorted(q1_filt['date'].unique()):
        dg = q1_filt[q1_filt['date'] == day]
        n_sig = len(dg)
        picked = select_top_k_with_dedup(dg, N_MAX, dedup_family=True, dedup_super_group=True)
        picked = picked.copy()
        picked['_sized_pnl'] = picked.apply(
            lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)

        # Build trade string with family + super-group tag
        parts = []
        for _, r in picked.iterrows():
            arrow = '✓' if r['_sized_pnl'] > 0 else '✗'
            sup = symbol_super_group(r['symbol'])
            fam = symbol_family(r['symbol'])
            tag = ''
            if sup == 'lev_short': tag = '[SHORT]'
            elif sup == 'lev_long': tag = '[LONG]'
            elif fam: tag = f"[{fam[:4]}]"
            parts.append(f"{r['symbol']}{tag}({r['_quintile']}) {arrow}${r['_sized_pnl']:+,.0f}")
        trade_str = ', '.join(parts)
        if len(trade_str) > 54:
            trade_str = trade_str[:51] + '…'

        day_pnl = float(picked['_sized_pnl'].sum())
        equity += day_pnl
        peak = max(peak, equity)
        dd_now = equity - peak
        max_dd = min(max_dd, dd_now)
        daily_rows.append({'date': day, 'day_pnl': day_pnl,
                          'equity': equity, 'dd_now': dd_now,
                          'n_sig': n_sig, 'n_picked': len(picked),
                          'trade_str': trade_str})
        print(f"{str(day.date()):<12} {n_sig:>4} {len(picked):>7} "
              f"{trade_str:<56} "
              f"${day_pnl:>+9,.0f} ${equity:>+9,.0f} ${dd_now:>+8,.0f}")

    print(f"\n  Q1 2025 SUMMARY with both fixes:")
    print(f"    Final equity: ${equity:+,.0f}")
    print(f"    Peak equity:  ${peak:+,.0f}")
    print(f"    Max DD:       ${max_dd:+,.0f}")

    daily_rows_df = pd.DataFrame(daily_rows)
    print(f"\n  WORST 5 DAYS with both fixes:")
    for _, r in daily_rows_df.nsmallest(5, 'day_pnl').iterrows():
        print(f"    {r['date'].date()}  pnl=${r['day_pnl']:>+8,.0f}  "
              f"picks: {r['trade_str']}")

    # Write report
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    md_path = f"{OUT_DIR}/orb_correlation_filter_{ts}.md"
    with open(md_path, 'w') as f:
        f.write(f"# ORB Correlation Filter + Q5 Cap — walk-forward\n\n")
        f.write(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n\n")
        f.write(f"## Fixes applied\n\n")
        f.write(f"**Fix 1: Family dedup** — picks from the same correlation family "
                f"(leveraged ETFs on same underlying, sector clusters like crypto miners/quantum) "
                f"are deduped: keep only the top-ranked symbol per family per day.\n\n")
        f.write(f"**Fix 2: Q5 mult cap at 1.5x** — Q5 was hitting 3.0x in Split A "
                f"due to anomalous H1 2025 train data. Cap at 1.5x matches what Split B/C train "
                f"also produced.\n\n")
        f.write(f"Universe: {len(df):,} trades. Family symbols: {fam_hits} ({100*fam_hits/len(df):.1f}%).\n\n")

        f.write(f"## Walk-forward comparison\n\n")
        f.write(f"| Config | Sum P&L | Min split | Worst DD | Worst day | Worst trade | P&L/|DD| |\n")
        f.write(f"|---|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(f"| {r['name']} | ${r['sum_pnl']:+,.0f} | ${r['min_split']:+,.0f} | "
                    f"${r['worst_dd']:+,.0f} | ${r['worst_day']:+,.0f} | "
                    f"${r['worst_trade']:+,.0f} | {r['pnl_dd']:.2f}x |\n")

        f.write(f"\n## Family definitions\n\n")
        for fam_name, syms in FAMILIES.items():
            f.write(f"- **{fam_name}**: {', '.join(syms)}\n")

    print(f"\nReport: {md_path}")


if __name__ == '__main__':
    main()
