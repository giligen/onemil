"""L2 historical backfill + signal validation on past trades.

Phase 1 of the L2 microstructure validation plan (see
memory/project_l2_entry_system.md). Fetches Databento L2 snapshots for
every closed trade in the DB, computes microstructure signals at the
entry/exit moments, then diffs WINNERS vs LOSERS to determine if the
academic OFI / book-imbalance / spread signals translate to our
small-cap gap-up universe.

Output:
  - entry_l2_depth / exit_l2_depth columns populated in trades DB
  - analysis_results/l2_signal_validation_YYYYMMDD.md report

Usage:
  # Smoke test (1 trade, ~$0.50 cost):
  python3 scripts/l2_backfill_and_validate.py --limit 1 --backfill-only

  # Cost estimate without fetching:
  python3 scripts/l2_backfill_and_validate.py --dry-run

  # Full backfill + validation:
  python3 scripts/l2_backfill_and_validate.py

  # Validation only (after backfill is done):
  python3 scripts/l2_backfill_and_validate.py --validate-only
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env for DATABENTO_API_KEY (the snapshot function needs it).
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
except ImportError:
    # Fallback: python-dotenv not installed. Manually load .env.
    env_file = ROOT / '.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from data_sources.l2_depth import snapshot_l2_at_fill, l2_to_json


DB_PATH = str(ROOT / 'data' / 'trades.db')
ANALYSIS_DIR = ROOT / 'analysis_results'
ANALYSIS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def fetch_closed_trades_needing_l2() -> pd.DataFrame:
    """Closed trades (have pnl + filled_at) that lack L2 backfill."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT id, strategy, trade_date, symbol, pnl,
               filled_at, exited_at,
               fill_price, exit_price,
               entry_price, exit_trigger_price,
               entry_l2_depth, exit_l2_depth,
               entry_quote_spread, exit_quote_spread
        FROM trades
        WHERE pnl IS NOT NULL
          AND filled_at IS NOT NULL
          AND fill_price IS NOT NULL
        ORDER BY trade_date, symbol
    """, conn)
    conn.close()
    return df


def update_trade_l2(trade_id: int, column: str, l2_json: str) -> None:
    """Write L2 JSON to the trades table."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        f"UPDATE trades SET {column} = ? WHERE id = ?",
        (l2_json, trade_id),
    )
    conn.commit()
    conn.close()


def backfill_one_trade(row: pd.Series, force: bool = False) -> Dict:
    """Backfill entry + exit L2 for one trade. Returns status dict."""
    status = {
        'id': int(row['id']),
        'symbol': row['symbol'],
        'date': row['trade_date'],
        'entry_status': 'skipped' if row['entry_l2_depth'] and not force else 'pending',
        'exit_status': 'skipped' if row['exit_l2_depth'] and not force else 'pending',
        'entry_records': 0,
        'exit_records': 0,
    }
    # Entry
    if status['entry_status'] == 'pending':
        try:
            ts = pd.Timestamp(row['filled_at']).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            l2 = snapshot_l2_at_fill(row['symbol'], ts, window_seconds=5)
            if l2:
                update_trade_l2(int(row['id']), 'entry_l2_depth', l2_to_json(l2))
                status['entry_status'] = 'ok'
                status['entry_records'] = sum(
                    e.get('records', 0) for e in l2.get('exchanges', {}).values()
                )
            else:
                status['entry_status'] = 'no_data'
        except Exception as e:
            status['entry_status'] = f'error: {e}'
    # Exit
    if status['exit_status'] == 'pending' and row['exited_at']:
        try:
            ts = pd.Timestamp(row['exited_at']).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            l2 = snapshot_l2_at_fill(row['symbol'], ts, window_seconds=5)
            if l2:
                update_trade_l2(int(row['id']), 'exit_l2_depth', l2_to_json(l2))
                status['exit_status'] = 'ok'
                status['exit_records'] = sum(
                    e.get('records', 0) for e in l2.get('exchanges', {}).values()
                )
            else:
                status['exit_status'] = 'no_data'
        except Exception as e:
            status['exit_status'] = f'error: {e}'
    return status


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def parse_l2_json(l2_str: Optional[str]) -> Optional[Dict]:
    if not l2_str: return None
    try:
        return json.loads(l2_str)
    except Exception:
        return None


def signals_from_snapshot(l2: Dict) -> Dict[str, Optional[float]]:
    """Compute microstructure signals from one L2 snapshot."""
    if not l2:
        return {'bid_depth': None, 'ask_depth': None, 'imbalance': None,
                'top_bid_size': None, 'top_ask_size': None, 'top_imbalance': None}
    bid_depth = l2.get('combined_bid_depth', 0)
    ask_depth = l2.get('combined_ask_depth', 0)
    total = bid_depth + ask_depth
    imbalance = bid_depth / total if total > 0 else None
    # Top-of-book signals from the largest exchange's level 0
    top_bid_size = 0
    top_ask_size = 0
    for ex_data in l2.get('exchanges', {}).values():
        levels = ex_data.get('levels', [])
        if levels:
            top_bid_size += levels[0].get('bid_sz', 0)
            top_ask_size += levels[0].get('ask_sz', 0)
    top_total = top_bid_size + top_ask_size
    top_imbalance = top_bid_size / top_total if top_total > 0 else None
    return {
        'bid_depth': bid_depth,
        'ask_depth': ask_depth,
        'imbalance': imbalance,
        'top_bid_size': top_bid_size,
        'top_ask_size': top_ask_size,
        'top_imbalance': top_imbalance,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def compute_auc(scores, labels) -> float:
    """AUC of `scores` predicting binary `labels`. Returns max(AUC, 1-AUC)
    so direction doesn't matter (we report effective separation)."""
    try:
        from sklearn.metrics import roc_auc_score
        s = pd.Series(scores).dropna()
        l = pd.Series(labels)[s.index]
        if len(set(l)) < 2 or len(s) < 5:
            return 0.5
        auc = roc_auc_score(l, s)
        return max(auc, 1 - auc)
    except Exception:
        return 0.5


def welch_t(a, b) -> tuple:
    try:
        from scipy.stats import ttest_ind
        a = pd.Series(a).dropna()
        b = pd.Series(b).dropna()
        if len(a) < 3 or len(b) < 3:
            return 0.0, 1.0
        r = ttest_ind(a, b, equal_var=False, nan_policy='omit')
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return 0.0, 1.0


def validate_signals():
    """Read all trades with L2 + compute WIN vs LOSS signal diffs."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT id, strategy, trade_date, symbol, pnl,
               entry_l2_depth, exit_l2_depth,
               entry_quote_spread, exit_quote_spread,
               entry_price, fill_price
        FROM trades
        WHERE pnl IS NOT NULL
          AND entry_l2_depth IS NOT NULL
        ORDER BY trade_date
    """, conn)
    conn.close()

    if len(df) == 0:
        print("No trades with L2 data. Run backfill first.")
        return

    # Extract entry-side signals
    entry_sigs = df['entry_l2_depth'].apply(parse_l2_json).apply(signals_from_snapshot)
    sig_df = pd.DataFrame(entry_sigs.tolist())
    sig_df['pnl'] = df['pnl'].values
    sig_df['strategy'] = df['strategy'].values
    sig_df['symbol'] = df['symbol'].values
    sig_df['trade_date'] = df['trade_date'].values
    sig_df['is_winner'] = (df['pnl'] > 0).astype(int).values
    sig_df['entry_spread_pct'] = (df['entry_quote_spread'] / df['entry_price'] * 100).values

    print(f"\n{'='*78}")
    print(f"  L2 SIGNAL VALIDATION — n={len(sig_df)} trades with L2 data")
    print(f"{'='*78}")
    wins = sig_df[sig_df['is_winner'] == 1]
    losses = sig_df[sig_df['is_winner'] == 0]
    print(f"  Winners: {len(wins)}  ({len(wins)/len(sig_df)*100:.0f}%)")
    print(f"  Losers:  {len(losses)}  ({len(losses)/len(sig_df)*100:.0f}%)")

    print(f"\n  By strategy:")
    for strat in sig_df['strategy'].unique():
        sub = sig_df[sig_df['strategy'] == strat]
        wr = (sub['is_winner'].sum() / len(sub) * 100) if len(sub) else 0
        print(f"    {strat:<12}  n={len(sub):>3}  win-rate={wr:>5.1f}%")

    # Per-signal diffs
    signals_to_test = [
        ('imbalance', 'Book imbalance (bid/(bid+ask) over 10 levels)'),
        ('top_imbalance', 'Top-of-book imbalance'),
        ('bid_depth', 'Combined bid depth'),
        ('ask_depth', 'Combined ask depth'),
        ('top_bid_size', 'Top-of-book bid size'),
        ('top_ask_size', 'Top-of-book ask size'),
        ('entry_spread_pct', 'Entry spread as % of price'),
    ]
    print(f"\n{'='*78}")
    print(f"  SIGNAL ANALYSIS — WINNERS vs LOSERS  (entry-time L2)")
    print(f"{'='*78}")
    print(f"  {'Signal':<35} {'WIN mean':>12} {'LOSS mean':>12} {'AUC':>6} {'p':>7}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*6} {'-'*7}")
    for col, _label in signals_to_test:
        if col not in sig_df.columns: continue
        w_vals = wins[col].dropna()
        l_vals = losses[col].dropna()
        if len(w_vals) < 3 or len(l_vals) < 3:
            print(f"  {col:<35} (n too small)")
            continue
        wm = w_vals.mean(); lm = l_vals.mean()
        auc = compute_auc(sig_df[col].values, sig_df['is_winner'].values)
        _, p = welch_t(w_vals.values, l_vals.values)
        sig = ' ★' if auc >= 0.60 else (' (?)' if auc >= 0.55 else '')
        print(f"  {col:<35} {wm:>12,.3f} {lm:>12,.3f} {auc:>6.3f} {p:>7.3f}{sig}")

    # Save full data
    out_csv = ANALYSIS_DIR / f'l2_signal_validation_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
    sig_df.to_csv(out_csv, index=False)
    print(f"\n  Saved trade-level signals: {out_csv}")

    # Final verdict
    aucs = []
    for col, _ in signals_to_test:
        if col not in sig_df.columns: continue
        auc = compute_auc(sig_df[col].values, sig_df['is_winner'].values)
        if not pd.isna(auc): aucs.append((col, auc))
    aucs.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'='*78}")
    print(f"  VERDICT")
    print(f"{'='*78}")
    if aucs:
        best_col, best_auc = aucs[0]
        if best_auc >= 0.60:
            print(f"  ✓ Signal found: {best_col} AUC={best_auc:.3f}")
            print(f"    → Phase 1 snapshot filter ship-worthy")
            print(f"    → Phase 2 (live stream) worth further investment IF cost OK")
        elif best_auc >= 0.55:
            print(f"  ~ Marginal signal: {best_col} AUC={best_auc:.3f}")
            print(f"    → Phase 1 may give small lift; Phase 2 doesn't justify infra cost")
        else:
            print(f"  ✗ No signal: best AUC = {best_auc:.3f} ({best_col})")
            print(f"    → Snapshot L2 doesn't separate winners from losers on our universe")
            print(f"    → Don't invest in Phase 2 live infrastructure")
            print(f"    → Spread filter (already shipped on bull flag) is the only L2 win")
    if len(sig_df) < 30:
        print(f"\n  CAVEAT: n={len(sig_df)} is small. AUC confidence intervals are wide.")
        print(f"  Re-run after 50+ closed trades for statistical confidence.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Estimate cost; do not call Databento')
    parser.add_argument('--limit', type=int, default=None,
                        help='Backfill only first N trades (for smoke test)')
    parser.add_argument('--force', action='store_true',
                        help='Refetch even if L2 already populated')
    parser.add_argument('--backfill-only', action='store_true',
                        help='Skip validation step')
    parser.add_argument('--validate-only', action='store_true',
                        help='Skip backfill; run signal analysis on existing L2')
    args = parser.parse_args()

    if args.validate_only:
        validate_signals()
        return

    df = fetch_closed_trades_needing_l2()
    print(f"Closed trades total: {len(df)}")
    needs_entry = df[df['entry_l2_depth'].isnull() | args.force].shape[0]
    needs_exit = df[df['exit_l2_depth'].isnull() | args.force].shape[0]
    print(f"  Need entry L2: {needs_entry}")
    print(f"  Need exit L2:  {needs_exit}")

    # Cost estimate: assume 1 query per (trade, side, dataset) × 4 datasets
    queries = (needs_entry + needs_exit) * 4
    # Rough $/query: Databento mbp-10 small range ≈ $0.20-0.50
    est_low = queries * 0.20
    est_high = queries * 0.50
    print(f"\n  Estimated queries: {queries}")
    print(f"  Estimated cost:    ${est_low:.2f} - ${est_high:.2f}")
    print(f"  (Actual cost depends on Databento billing tier — verify after first batch)")

    if args.dry_run:
        print("\n  Dry run — no fetching.")
        return

    # Backfill
    if args.limit:
        df = df.head(args.limit)
        print(f"\n  Limit applied: {len(df)} trade(s)")

    print(f"\nBackfilling {len(df)} trade(s)...")
    statuses = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        status = backfill_one_trade(row, force=args.force)
        statuses.append(status)
        print(f"  [{i:>3}/{len(df)}] {status['symbol']:<7} {status['date']}  "
              f"entry={status['entry_status']:<10} exit={status['exit_status']:<10}  "
              f"recs (e={status['entry_records']}/x={status['exit_records']})")
        # Small pause to avoid rate-limiting on Databento side
        time.sleep(0.5)

    # Status summary
    n_entry_ok = sum(1 for s in statuses if s['entry_status'] == 'ok')
    n_entry_no = sum(1 for s in statuses if s['entry_status'] == 'no_data')
    n_entry_err = sum(1 for s in statuses if str(s['entry_status']).startswith('error'))
    n_exit_ok = sum(1 for s in statuses if s['exit_status'] == 'ok')
    n_exit_no = sum(1 for s in statuses if s['exit_status'] == 'no_data')
    n_exit_err = sum(1 for s in statuses if str(s['exit_status']).startswith('error'))
    print(f"\n  Backfill summary:")
    print(f"    Entry: ok={n_entry_ok}  no_data={n_entry_no}  errors={n_entry_err}")
    print(f"    Exit:  ok={n_exit_ok}  no_data={n_exit_no}  errors={n_exit_err}")

    if args.backfill_only:
        return

    # Validation
    print()
    validate_signals()


if __name__ == '__main__':
    main()
