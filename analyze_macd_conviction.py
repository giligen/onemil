#!/usr/bin/env python3
"""
MACD Wave Conviction Research — Step 1: Out-of-Sample Bucket Analysis.

Identifies features that discriminate winners from losers in MACD wave trades
using a strict chronological train/test split to avoid overfitting.

Usage:
    python3 analyze_macd_conviction.py \
        --csv macd_wave_results.csv \
        --train-start 2025-01-01 --train-end 2025-06-30 \
        --out analysis_results/macd_conviction_<date>.md

Step 2 (building the conviction formula) is a separate task.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


CACHE_DB = 'data/cache.db'


# ---------- Data loading and split --------------------------------------------

def load_and_split(
    csv_path: str, train_start: str, train_end: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV and split chronologically into train (inclusive of train_end)
    and test (everything after train_end)."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)

    ts = pd.Timestamp(train_start)
    te = pd.Timestamp(train_end)
    train = df[(df['date'] >= ts) & (df['date'] <= te)].copy()
    test = df[df['date'] > te].copy()

    print(f"Loaded {len(df)} trades")
    print(f"Train [{train_start} .. {train_end}]: {len(train)} trades")
    print(f"Test  ({train_end} .. ]:             {len(test)} trades")
    return train, test


# ---------- Feature enrichment ------------------------------------------------

def _prev_trading_day_close(conn, symbol: str, date: pd.Timestamp) -> Optional[float]:
    row = conn.execute(
        "SELECT close FROM daily_bars WHERE symbol=? AND bar_date < ? "
        "ORDER BY bar_date DESC LIMIT 1",
        (symbol, date.strftime('%Y-%m-%d'))
    ).fetchone()
    return float(row[0]) if row else None


def _load_intraday_bars(conn, symbol: str, date: pd.Timestamp) -> pd.DataFrame:
    """Load ALL intraday bars for a given symbol/date."""
    rows = conn.execute(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM intraday_bars_1min WHERE symbol=? AND bar_date=? "
        "ORDER BY timestamp",
        (symbol, date.strftime('%Y-%m-%d'))
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


def _regular_session_start(bars: pd.DataFrame, symbol_spy_ref: Optional[pd.Timestamp] = None) -> Optional[pd.Timestamp]:
    """Return the first regular-session bar timestamp for this date.
    Uses SPY's first bar as the anchor if provided; else falls back to the first
    bar whose volume suggests regular hours (heuristic). Pre-market bars get
    filtered out this way."""
    if symbol_spy_ref is not None:
        return symbol_spy_ref
    # Fallback: pick the first bar of day as proxy (OK if we already filtered)
    if bars.empty:
        return None
    return bars['timestamp'].iloc[0]


def enrich_features(df: pd.DataFrame, db_path: str = CACHE_DB) -> pd.DataFrame:
    """Enrich trades with intraday + SPY features. Returns df with new columns."""
    if df.empty:
        return df

    conn = sqlite3.connect(db_path)
    conn.row_factory = None

    # Build SPY daily regime features once — outer join by date
    spy_daily = pd.read_sql_query(
        "SELECT bar_date, open, high, low, close FROM daily_bars "
        "WHERE symbol='SPY' ORDER BY bar_date",
        conn,
        parse_dates=['bar_date']
    )
    # SPY 5-day vol (prev 5 days' daily range %)
    spy_daily['daily_range_pct'] = (spy_daily['high'] - spy_daily['low']) / spy_daily['low'] * 100
    spy_daily['spy_5d_vol'] = spy_daily['daily_range_pct'].rolling(5, min_periods=3).mean().shift(1)
    spy_daily['spy_prev_close'] = spy_daily['close'].shift(1)
    spy_daily['spy_gap_pct'] = (spy_daily['open'] - spy_daily['spy_prev_close']) / spy_daily['spy_prev_close'] * 100
    spy_daily['spy_sma50'] = spy_daily['close'].rolling(50, min_periods=10).mean().shift(1)
    spy_daily['spy_above_sma50'] = (spy_daily['open'] > spy_daily['spy_sma50']).astype(int)
    spy_lookup = spy_daily.set_index('bar_date')[['spy_5d_vol', 'spy_gap_pct', 'spy_above_sma50']]

    # SPY intraday open times (for regular-session anchor per date)
    spy_intraday = pd.read_sql_query(
        "SELECT bar_date, MIN(timestamp) AS first_ts FROM intraday_bars_1min "
        "WHERE symbol='SPY' GROUP BY bar_date",
        conn,
        parse_dates=['bar_date']
    )
    spy_intraday['first_ts'] = pd.to_datetime(spy_intraday['first_ts'], utc=True)
    spy_first_ts = spy_intraday.set_index('bar_date')['first_ts']

    # Per-trade enrichment
    feat_rows = []
    prev_close_cache: Dict[Tuple[str, str], Optional[float]] = {}

    for _, t in df.iterrows():
        sym = t['symbol']
        date = t['date']
        et = t['entry_time']
        feat = {'_idx': t.name}

        # SPY regime (by date)
        if date in spy_lookup.index:
            feat['spy_5d_vol'] = spy_lookup.loc[date, 'spy_5d_vol']
            feat['spy_gap_pct'] = spy_lookup.loc[date, 'spy_gap_pct']
            feat['spy_above_sma50'] = spy_lookup.loc[date, 'spy_above_sma50']
        else:
            feat['spy_5d_vol'] = np.nan
            feat['spy_gap_pct'] = np.nan
            feat['spy_above_sma50'] = np.nan

        # Prev-day close for gap calculation
        ck = (sym, date.strftime('%Y-%m-%d'))
        if ck not in prev_close_cache:
            prev_close_cache[ck] = _prev_trading_day_close(conn, sym, date)
        prev_close = prev_close_cache[ck]

        # Intraday bars for stock — use the BT's frame (ALL cached bars including
        # pre-market), matching `bars.iloc[0]['open']` as the day anchor.
        bars = _load_intraday_bars(conn, sym, date).reset_index(drop=True)
        # Keep a regular-session view for entry-bar lookup if needed
        session_start = spy_first_ts.get(date, None)

        if not bars.empty:
            # BT's day_open = first cached bar's open (may be pre-market)
            day_open = float(bars.iloc[0]['open'])
            # Gap from prev close to day open (BT's frame)
            if prev_close and prev_close > 0:
                feat['gap_pct'] = (day_open - prev_close) / prev_close * 100
            else:
                feat['gap_pct'] = np.nan

            # Pre-cross run: day_open → cross bar's close.
            # BT stores cross_time_min = si + 1, so cross bar idx = cross_time_min - 1.
            si = int(t['cross_time_min']) - 1
            if 0 <= si < len(bars):
                cross_close = float(bars.iloc[si]['close'])
                feat['pre_cross_run_pct'] = (cross_close - day_open) / day_open * 100
            else:
                feat['pre_cross_run_pct'] = np.nan

            # Find entry bar by timestamp match (full bars, pre-market included)
            eb_match = bars[bars['timestamp'] == et]
            if not eb_match.empty:
                eb = eb_match.iloc[0]
                entry_bar_idx = eb_match.index[0]

                # Entry bar vol surge
                prior5 = bars.iloc[max(0, entry_bar_idx - 5):entry_bar_idx]
                if len(prior5) > 0:
                    avg_prior_vol = prior5['volume'].mean()
                    feat['entry_bar_vol_surge'] = (
                        eb['volume'] / avg_prior_vol if avg_prior_vol > 0 else np.nan
                    )
                else:
                    feat['entry_bar_vol_surge'] = np.nan

                # Entry bar range position (close in low..high range)
                bar_range = eb['high'] - eb['low']
                if bar_range > 0:
                    feat['entry_bar_range_pos'] = (eb['close'] - eb['low']) / bar_range
                else:
                    feat['entry_bar_range_pos'] = 0.5

                # Intraday range at entry (day high/low up to and including entry bar)
                up_to_entry = bars.iloc[:entry_bar_idx + 1]
                day_hi = up_to_entry['high'].max()
                day_lo = up_to_entry['low'].min()
                feat['intraday_range_at_entry_pct'] = (
                    (day_hi - day_lo) / float(t['entry_price']) * 100 if t['entry_price'] > 0 else np.nan
                )
            else:
                feat['entry_bar_vol_surge'] = np.nan
                feat['entry_bar_range_pos'] = np.nan
                feat['intraday_range_at_entry_pct'] = np.nan
        else:
            feat['gap_pct'] = np.nan
            feat['pre_cross_run_pct'] = np.nan
            feat['entry_bar_vol_surge'] = np.nan
            feat['entry_bar_range_pos'] = np.nan
            feat['intraday_range_at_entry_pct'] = np.nan

        # Cum dollar vol at cross (proxy via vol_at_cross × entry_price)
        feat['cum_dollar_vol_at_cross'] = float(t['vol_at_cross']) * float(t['entry_price'])

        feat_rows.append(feat)

    conn.close()
    feats = pd.DataFrame(feat_rows).set_index('_idx')
    return df.join(feats)


# ---------- Bucket analysis ---------------------------------------------------

@dataclass
class BucketResult:
    edges: List[float]
    rows: List[Dict]          # per-bucket stats
    rho: Optional[float]       # spearman (bucket_idx vs avg pnl $)
    monotonic: bool            # |rho| >= 0.5 and direction consistent


def _bucket_edges_from_train(train_vals: pd.Series, n_buckets: int = 4) -> List[float]:
    """Compute quartile edges from training data only. Returns (n_buckets+1) edges."""
    clean = train_vals.dropna()
    if len(clean) < n_buckets * 2:
        return []
    qs = np.linspace(0, 1, n_buckets + 1)
    edges = clean.quantile(qs).tolist()
    # Ensure strict monotonicity (quantiles can tie on small samples)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    edges[0] = -np.inf  # include anything below lower quartile
    edges[-1] = np.inf
    return edges


def _bucket_with_edges(df: pd.DataFrame, feature: str, edges: List[float]) -> BucketResult:
    if df.empty or not edges:
        return BucketResult(edges=edges or [], rows=[], rho=None, monotonic=False)
    col = df[feature]
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (col > lo) & (col <= hi) & col.notna()
        if i == 0:
            mask = (col >= lo) & (col <= hi) & col.notna()
        sub = df[mask]
        if len(sub) == 0:
            rows.append({
                'bucket': i + 1, 'range': f"({lo:.3g}, {hi:.3g}]",
                'n': 0, 'wr_pct': np.nan, 'avg_pnl_pct': np.nan,
                'avg_pnl_dollar': np.nan, 'total_pnl': 0.0
            })
            continue
        rows.append({
            'bucket': i + 1,
            'range': f"({lo:.3g}, {hi:.3g}]",
            'n': len(sub),
            'wr_pct': (sub['pnl_dollar'] > 0).mean() * 100,
            'avg_pnl_pct': sub['pnl_pct'].mean(),
            'avg_pnl_dollar': sub['pnl_dollar'].mean(),
            'total_pnl': sub['pnl_dollar'].sum(),
        })
    # Monotonicity: spearman rho between bucket index and avg pnl $
    bucket_ids = [r['bucket'] for r in rows if r['n'] > 0]
    pnls = [r['avg_pnl_dollar'] for r in rows if r['n'] > 0]
    if len(bucket_ids) >= 3 and np.std(pnls) > 0:
        rho = np.corrcoef(bucket_ids, pnls)[0, 1]  # pearson on ranks = spearman (ranks are already 1..n)
    else:
        rho = None
    monotonic = rho is not None and abs(rho) >= 0.5
    return BucketResult(edges=edges, rows=rows, rho=rho, monotonic=monotonic)


def _verdict(train_br: BucketResult, test_br: BucketResult) -> Tuple[str, str]:
    """Return (verdict, reason) describing OOS signal strength."""
    if not train_br.rows or train_br.rho is None:
        return ("none", "insufficient data or variance on train")
    if not train_br.monotonic:
        return ("none", f"train not monotonic (ρ={train_br.rho:+.2f})")

    if not test_br.rows or test_br.rho is None:
        return ("weak", "test data missing or uniform")

    # Direction persists?
    same_dir = (train_br.rho > 0) == (test_br.rho > 0)
    if not same_dir:
        return ("weak", f"direction flips (train ρ={train_br.rho:+.2f}, test ρ={test_br.rho:+.2f})")

    # Magnitude persists?
    if abs(test_br.rho) < 0.3:
        return ("weak", f"test ρ weak ({test_br.rho:+.2f})")

    # Top/bottom EV gap
    pnls = [r['avg_pnl_dollar'] for r in test_br.rows if r['n'] > 0]
    if len(pnls) >= 2:
        ev_gap = abs(pnls[-1] - pnls[0]) if (test_br.rho > 0) else abs(pnls[0] - pnls[-1])
        if ev_gap < 100:
            return ("weak", f"test EV gap only ${ev_gap:.0f}/trade")

    return ("strong", f"train ρ={train_br.rho:+.2f}, test ρ={test_br.rho:+.2f}")


# ---------- Report writing ----------------------------------------------------

def _fmt_bucket_table(br: BucketResult, title: str) -> str:
    if not br.rows:
        return f"  _{title}: (no data)_\n"
    out = [f"  **{title}**  (ρ = {br.rho:+.2f})" if br.rho is not None else f"  **{title}**"]
    out.append("")
    out.append("  | Bucket | Range | n | WR% | Avg P&L% | Avg P&L $ | Total P&L |")
    out.append("  |---|---|---|---|---|---|---|")
    for r in br.rows:
        if r['n'] == 0:
            out.append(f"  | {r['bucket']} | {r['range']} | 0 | — | — | — | — |")
        else:
            out.append(
                f"  | {r['bucket']} | {r['range']} | {r['n']} | "
                f"{r['wr_pct']:.0f}% | {r['avg_pnl_pct']:+.2f}% | "
                f"${r['avg_pnl_dollar']:+,.0f} | ${r['total_pnl']:+,.0f} |"
            )
    out.append("")
    return "\n".join(out)


def _corr_matrix(train_df: pd.DataFrame, features: List[str]) -> str:
    if len(features) < 2:
        return "_(fewer than 2 strong features — correlation matrix not meaningful)_\n"
    data = train_df[features].dropna(how='any')
    if len(data) < 20:
        return f"_(only {len(data)} complete rows on train — matrix unreliable)_\n"
    rho = data.corr(method='spearman')
    out = ["| | " + " | ".join(features) + " |"]
    out.append("|---|" + "---|" * len(features))
    for f in features:
        row = [f]
        for g in features:
            v = rho.loc[f, g]
            row.append(f"{v:+.2f}" if f != g else "—")
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def write_report(
    out_path: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_start: str,
    train_end: str,
    results: List[Tuple[str, BucketResult, BucketResult, str, str]],
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = []
    lines.append("# MACD Wave Conviction Research — Step 1 Results\n")
    lines.append(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n")
    lines.append("## Context\n")
    lines.append(
        "Out-of-sample bucket analysis to identify features discriminating "
        "winners from losers in MACD wave trades. Train on H1'25, test on H2'25 + Q1'26.\n"
    )
    lines.append("## Setup\n")
    lines.append(f"- **Train period:** {train_start} → {train_end} ({len(train_df)} trades)")
    lines.append(f"- **Test period:** after {train_end} ({len(test_df)} trades)")
    if not train_df.empty:
        lines.append(f"- **Train P&L:** ${train_df['pnl_dollar'].sum():+,.0f}   "
                     f"WR: {(train_df['pnl_dollar'] > 0).mean() * 100:.1f}%")
    if not test_df.empty:
        lines.append(f"- **Test P&L:**  ${test_df['pnl_dollar'].sum():+,.0f}   "
                     f"WR: {(test_df['pnl_dollar'] > 0).mean() * 100:.1f}%")
    lines.append("")

    # Sort results: strong > weak > none
    rank = {"strong": 0, "weak": 1, "none": 2}
    results.sort(key=lambda r: (rank.get(r[3], 99), r[0]))

    # Summary table
    lines.append("## Summary — features ranked by OOS signal\n")
    lines.append("| Feature | Train ρ | Test ρ | Test top-bot EV gap | Verdict | Note |")
    lines.append("|---|---|---|---|---|---|")
    for name, train_br, test_br, verdict, reason in results:
        tr_rho = f"{train_br.rho:+.2f}" if train_br.rho is not None else "—"
        te_rho = f"{test_br.rho:+.2f}" if test_br.rho is not None else "—"
        ev_gap_str = "—"
        if test_br.rows:
            pnls = [r['avg_pnl_dollar'] for r in test_br.rows if r['n'] > 0]
            if len(pnls) >= 2:
                ev_gap = pnls[-1] - pnls[0]
                ev_gap_str = f"${ev_gap:+,.0f}"
        lines.append(f"| `{name}` | {tr_rho} | {te_rho} | {ev_gap_str} | **{verdict}** | {reason} |")
    lines.append("")

    # Per-feature detail
    lines.append("## Per-feature detail\n")
    for name, train_br, test_br, verdict, reason in results:
        lines.append(f"### `{name}` — **{verdict}**")
        lines.append(f"_{reason}_\n")
        if train_br.edges and len(train_br.edges) >= 2:
            edges_str = ", ".join(f"{e:.3g}" if np.isfinite(e) else "±∞" for e in train_br.edges)
            lines.append(f"  Bucket edges (from train): [{edges_str}]\n")
        lines.append(_fmt_bucket_table(train_br, "Train"))
        lines.append(_fmt_bucket_table(test_br, "Test (same edges)"))
        lines.append("")

    # Correlation matrix of strong features
    strong = [r[0] for r in results if r[3] == "strong"]
    lines.append("## Correlation matrix — strong features (Spearman, train set)\n")
    if strong:
        lines.append(_corr_matrix(train_df, strong))
    else:
        lines.append("_(no strong features found)_\n")

    # Overall conclusions
    lines.append("## Candidates for Step 2 (formula building)\n")
    if strong:
        lines.append("Features with OOS signal, in order of strength:\n")
        for name, train_br, test_br, _, _ in results:
            if _ == "weak":
                continue
            verdict = next(v for n, _, _, v, _ in results if n == name)
            if verdict == "strong":
                lines.append(f"- `{name}` (train ρ={train_br.rho:+.2f}, test ρ={test_br.rho:+.2f})")
    else:
        lines.append(
            "_No features passed the 'strong' threshold. Options:_\n"
            "- Relax thresholds (e.g., accept weak features if direction persists)\n"
            "- Add Tier 4 features (RSI, OBV, VWAP, EMA21)\n"
            "- Investigate whether the already-applied hard filters leave no gradient\n"
        )
    lines.append("")
    lines.append("**This report ends step 1.** Step 2 — building the conviction formula, "
                 "selecting thresholds, and validating across multiple walk-forward splits — "
                 "is a separate task to discuss with the user.")

    with open(out_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"\nReport written: {out_path}")


# ---------- Main --------------------------------------------------------------

FEATURES_TO_TEST = [
    # Tier 1 (from CSV)
    'cross_time_min',
    'vol_at_cross',
    'macd_hist_pct',
    'entry_price',
    # Tier 2 (enriched from intraday bars)
    'gap_pct',
    'pre_cross_run_pct',
    'entry_bar_vol_surge',
    'entry_bar_range_pos',
    'intraday_range_at_entry_pct',
    'cum_dollar_vol_at_cross',
    # Tier 3 (SPY regime)
    'spy_5d_vol',
    'spy_gap_pct',
    'spy_above_sma50',  # binary: handled separately below
]

BINARY_FEATURES = {'spy_above_sma50'}


def _binary_analysis(df: pd.DataFrame, feature: str) -> BucketResult:
    """Treat a binary feature as 2 buckets."""
    rows = []
    for val in sorted(df[feature].dropna().unique()):
        sub = df[df[feature] == val]
        rows.append({
            'bucket': int(val) + 1,
            'range': f"= {int(val)}",
            'n': len(sub),
            'wr_pct': (sub['pnl_dollar'] > 0).mean() * 100,
            'avg_pnl_pct': sub['pnl_pct'].mean(),
            'avg_pnl_dollar': sub['pnl_dollar'].mean(),
            'total_pnl': sub['pnl_dollar'].sum(),
        })
    rho = None
    if len(rows) == 2 and all(r['n'] > 0 for r in rows):
        diff = rows[1]['avg_pnl_dollar'] - rows[0]['avg_pnl_dollar']
        rho = 1.0 if diff > 0 else -1.0 if diff < 0 else 0.0
    return BucketResult(edges=[], rows=rows, rho=rho, monotonic=(rho is not None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='macd_wave_results.csv')
    parser.add_argument('--train-start', default='2025-01-01')
    parser.add_argument('--train-end', default='2025-06-30')
    parser.add_argument('--out', default=None)
    parser.add_argument('--db', default=CACHE_DB)
    args = parser.parse_args()

    out = args.out or f"analysis_results/macd_conviction_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

    train, test = load_and_split(args.csv, args.train_start, args.train_end)
    if train.empty or test.empty:
        print("ERROR: train or test split is empty")
        sys.exit(1)

    print("\nEnriching train set with intraday/SPY features...")
    train = enrich_features(train, args.db)
    print("Enriching test set...")
    test = enrich_features(test, args.db)

    print(f"\nRunning bucket analysis on {len(FEATURES_TO_TEST)} features...")
    results = []
    for f in FEATURES_TO_TEST:
        if f not in train.columns:
            print(f"  SKIP {f}: not in data")
            continue
        train_vals = train[f].dropna()
        if len(train_vals) < 20:
            print(f"  SKIP {f}: only {len(train_vals)} non-null train values")
            continue

        if f in BINARY_FEATURES:
            train_br = _binary_analysis(train, f)
            test_br = _binary_analysis(test, f)
        else:
            edges = _bucket_edges_from_train(train_vals, n_buckets=4)
            train_br = _bucket_with_edges(train, f, edges)
            test_br = _bucket_with_edges(test, f, edges)

        verdict, reason = _verdict(train_br, test_br)
        print(f"  {f:<32} train ρ={train_br.rho if train_br.rho is not None else 'n/a':<6} "
              f"test ρ={test_br.rho if test_br.rho is not None else 'n/a':<6} → {verdict}")
        results.append((f, train_br, test_br, verdict, reason))

    write_report(out, train, test, args.train_start, args.train_end, results)


if __name__ == '__main__':
    main()
