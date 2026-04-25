"""Investigate the live-vs-BT composite-score drift (~0.09 below BT, root cause unknown).

CLAUDE.md note: "live composites ran ~0.09 below BT on 4/22, cause unknown
(likely a stale T-1 daily bar or 20d-high at 9:35 ET)."

This script reconstructs both the live and BT composite for the same
(symbol, date) pair, then diffs each of the 7 input features to identify
which one is responsible for the gap.

Inputs:
  - Live: parsed from journalctl 'ORB SCORED:' log lines (each contains
    composite + 7 features as observed by the engine at 9:35 ET)
  - BT:   `analysis_results/orb_features_*.csv` (the most recent file)

Outputs per scored symbol/date:
  Feature      Live    BT     Δ (live-BT)    Notes
  gap_pct      14.5    14.5    0.0
  range_total..XXX     XXX     XXX
  ...
  -- composite XXX     XXX     XXX

The feature with the biggest absolute Δ is the prime suspect.

Usage:
    # Auto-pull last 24h of journalctl
    sudo python3 scripts/investigate_composite_drift.py
    # OR specify a log file
    python3 scripts/investigate_composite_drift.py --log path/to/log

Notes:
  - Run as root (or with sudo) so journalctl is accessible
  - If no SCORED log lines found, ORB hasn't run yet today or telemetry
    is misconfigured
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional

import pandas as pd


# Regex to parse ORB SCORED log lines from orb_engine.py:990-1004
# Format: "ORB SCORED: SYM comp=X.XXXX QX | gap=... rtv=... rabr=... rs=... p20h=... pdcp=... rcp=... | prev_close=... range_open=..."
SCORED_RE = re.compile(
    r"ORB SCORED:\s+(?P<symbol>\S+)\s+"
    r"comp=(?P<composite>-?\d+\.\d+)\s+"
    r"(?P<quintile>Q\d|--)\s+\|\s+"
    r"gap=(?P<gap_pct>-?\d+\.\d+|nan)\s+"
    r"rtv=(?P<range_total_volume>-?\d+\.?\d*|nan)\s+"
    r"rabr=(?P<range_avg_bar_range_pct>-?\d+\.\d+|nan)\s+"
    r"rs=(?P<range_size_pct>-?\d+\.\d+|nan)\s+"
    r"p20h=(?P<price_vs_20d_high_pct>-?\d+\.\d+|nan)\s+"
    r"pdcp=(?P<prev_day_close_position>-?\d+\.\d+|nan)\s+"
    r"rcp=(?P<range_close_position>-?\d+\.\d+|nan)\s+\|\s+"
    r"prev_close=(?P<prev_close>-?\d+\.\d+)\s+"
    r"range_open=(?P<range_open>-?\d+\.\d+)"
)

FILTER_FEATURES_USED = [
    'gap_pct', 'range_total_volume', 'range_avg_bar_range_pct',
    'range_size_pct', 'price_vs_20d_high_pct', 'prev_day_close_position',
    'range_close_position',
]


def parse_journalctl(log_text: str) -> List[dict]:
    """Parse SCORED lines from journalctl output."""
    out = []
    for line in log_text.splitlines():
        m = SCORED_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        # Convert numeric fields
        for k in ('composite', *FILTER_FEATURES_USED, 'prev_close', 'range_open'):
            try:
                d[k] = float(d[k])
            except (ValueError, TypeError):
                d[k] = float('nan')
        out.append(d)
    return out


def fetch_journalctl_for_date(date_str: str) -> str:
    """Fetch SCORED lines bounded to a specific ET trading day.

    Uses --since YYYY-MM-DD 08:00:00 / --until YYYY-MM-DD 17:00:00 (ET)
    via systemd's local-time interpretation (host should be UTC; we pass
    UTC bounds that span the ET trading day).
    """
    if os.geteuid() != 0:
        print("NOTE: not running as root. journalctl may return no data.",
              file=sys.stderr)
    # Trading day in ET = 9:30-16:30 ET = 13:30-20:30 UTC during EDT.
    # Widen to 12:00-22:00 UTC for slop.
    since = f"{date_str} 12:00:00 UTC"
    until = f"{date_str} 22:00:00 UTC"
    try:
        result = subprocess.run(
            ['journalctl', '-u', 'onemil-trader', '--since', since,
             '--until', until, '-o', 'cat', '--grep', 'ORB SCORED'],
            capture_output=True, text=True, timeout=60,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"WARNING: journalctl failed: {e}", file=sys.stderr)
        return ''
    except FileNotFoundError:
        print("ERROR: journalctl not found — run on the trading host", file=sys.stderr)
        sys.exit(1)


def fetch_journalctl_24h() -> str:
    """Fetch ORB SCORED lines from journalctl (last 24h).

    Requires either root or membership in the systemd-journal/adm group.
    Detect that early and surface a usable error.
    """
    if os.geteuid() != 0:
        # Non-root user — try anyway but warn first
        print("NOTE: not running as root. journalctl may return no data.",
              file=sys.stderr)
        print("      If output below is empty, retry with: sudo python3 ...",
              file=sys.stderr)
    try:
        result = subprocess.run(
            ['journalctl', '-u', 'onemil-trader', '--since', '24 hours ago',
             '-o', 'cat', '--grep', 'ORB SCORED'],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0 and result.stderr:
            print(f"WARNING: journalctl stderr: {result.stderr.strip()[:200]}",
                  file=sys.stderr)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"WARNING: journalctl failed: {e}", file=sys.stderr)
        return ''
    except FileNotFoundError:
        print("ERROR: journalctl not found — run on the trading host", file=sys.stderr)
        sys.exit(1)


def load_bt_features() -> pd.DataFrame:
    """Load most recent orb_features_*.csv (BT-side feature/composite source)."""
    candidates = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    candidates = [p for p in candidates if 'corrmatrix' not in p]
    if not candidates:
        print("ERROR: no orb_features_*.csv found", file=sys.stderr)
        sys.exit(1)
    csv = candidates[-1]
    print(f"Loading BT features from: {csv}")
    df = pd.read_csv(csv)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default=None,
                        help='Path to a saved journalctl log (else pull last 24h)')
    parser.add_argument('--date', type=str, default=None,
                        help='Filter live SCORED lines to this date (YYYY-MM-DD)')
    args = parser.parse_args()

    # 1. Get live SCORED lines
    if args.log:
        with open(args.log) as f:
            log_text = f.read()
    elif args.date:
        # Specific date requested — bound journalctl to that ET trading day
        # (9:00 ET to 16:30 ET = 13:00 UTC to 20:30 UTC during EDT;
        # widen by 1 hour each side for slop)
        log_text = fetch_journalctl_for_date(args.date)
    else:
        log_text = fetch_journalctl_24h()
    if not log_text:
        print("No journalctl output. ORB hasn't run today, or telemetry not on.")
        sys.exit(1)

    live = parse_journalctl(log_text)
    if not live:
        print(f"No 'ORB SCORED:' lines parsed from {len(log_text.splitlines())} log lines.")
        print("Either: (a) the regex needs updating for current log format,")
        print("  (b) ORB hasn't scored anything today, or")
        print("  (c) telemetry was added in a code version newer than the running service.")
        sys.exit(1)
    print(f"Parsed {len(live)} live SCORED lines")

    # 2. Load BT features
    bt = load_bt_features()
    print(f"BT features rows: {len(bt)}")

    # 3. Build comparison — match live to BT by symbol+date
    # Live doesn't directly include date, but logs are time-stamped. For
    # this script, assume all live entries are from the most recent trading
    # day (refine later via journal timestamps).
    today = args.date or pd.Timestamp.today().strftime('%Y-%m-%d')
    bt_today = bt[bt['date'] == today]
    if len(bt_today) == 0:
        print(f"\nWARNING: no BT rows for date {today}.")
        print(f"BT CSV most-recent date: {bt['date'].max()}")
        print(f"Will compare against ALL BT dates (matching by symbol). May be misleading.")
        bt_match = bt
    else:
        print(f"BT rows for {today}: {len(bt_today)}")
        bt_match = bt_today

    # 4. Per-symbol diff
    live_df = pd.DataFrame(live)
    print(f"\nLive symbols: {live_df['symbol'].unique().tolist()}")

    print(f"\n{'='*100}")
    print(f"  PER-FEATURE DIFF (live - BT) for symbols present in both")
    print(f"{'='*100}")
    for sym in sorted(live_df['symbol'].unique()):
        l_rows = live_df[live_df['symbol'] == sym]
        b_rows = bt_match[bt_match['symbol'] == sym]
        if len(b_rows) == 0:
            print(f"\n  {sym}: NO BT DATA (skipping)")
            continue
        # Use the most recent live entry for this symbol
        l = l_rows.iloc[-1]
        # Use the BT row matching today's date if available, else most recent
        b = b_rows.iloc[-1] if len(b_rows) else None
        print(f"\n  {sym}  (live ts proximate to {today})")
        print(f"    {'feature':<32} {'live':>14} {'BT':>14} {'Δ live-BT':>14}")
        print(f"    {'-'*32} {'-'*14} {'-'*14} {'-'*14}")
        for f in FILTER_FEATURES_USED + ['composite']:
            lv = l.get(f, float('nan'))
            bv = b.get(f if f != 'composite' else None, float('nan'))
            if f == 'composite':
                # BT may not have composite directly — has 'pnl' etc but not the
                # signed-z composite. Skip BT composite.
                print(f"    {f:<32} {lv:>14.4f} {'(BT computes)':>14} {'':>14}")
                continue
            try:
                lv = float(lv); bv = float(bv)
                d = lv - bv
                # Highlight if absolute difference > 1% of typical scale
                marker = ' ←' if abs(d) > 0.01 * max(abs(lv), abs(bv), 0.01) else ''
                print(f"    {f:<32} {lv:>14.4f} {bv:>14.4f} {d:>+14.4f}{marker}")
            except (ValueError, TypeError):
                print(f"    {f:<32} {lv!r:>14} {bv!r:>14} ?")

    # 5. Aggregate diffs across all matched symbols
    print(f"\n{'='*100}")
    print(f"  AGGREGATE FEATURE DIFFS (mean live - BT across matched symbols)")
    print(f"{'='*100}")
    diffs = {f: [] for f in FILTER_FEATURES_USED}
    matched = 0
    for sym in live_df['symbol'].unique():
        l_rows = live_df[live_df['symbol'] == sym]
        b_rows = bt_match[bt_match['symbol'] == sym]
        if len(b_rows) == 0: continue
        l = l_rows.iloc[-1]; b = b_rows.iloc[-1]
        matched += 1
        for f in FILTER_FEATURES_USED:
            try:
                d = float(l.get(f, float('nan'))) - float(b.get(f, float('nan')))
                if not pd.isna(d):
                    diffs[f].append(d)
            except Exception:
                pass
    if matched == 0:
        print(f"\n  No matched symbols. Cannot aggregate.")
        return
    print(f"\n  Matched symbols: {matched}")
    print(f"  {'feature':<32} {'mean Δ':>14} {'min Δ':>14} {'max Δ':>14}")
    print(f"  {'-'*32} {'-'*14} {'-'*14} {'-'*14}")
    rows = []
    for f, ds in diffs.items():
        if not ds:
            print(f"  {f:<32} {'no data':>14}")
            continue
        m, mn, mx = sum(ds)/len(ds), min(ds), max(ds)
        rows.append((f, m, mn, mx, len(ds)))
        marker = ' ←' if abs(m) > 0.01 * 5 else ''  # > 5% of avg-z scale
        print(f"  {f:<32} {m:>+14.4f} {mn:>+14.4f} {mx:>+14.4f}  (n={len(ds)}){marker}")

    print(f"\n  PRIME SUSPECTS (largest mean Δ):")
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    for f, m, mn, mx, n in rows[:3]:
        print(f"    {f}: mean Δ {m:+.4f}")

    # 6. Composite delta calc
    live_comps = live_df.groupby('symbol')['composite'].last()
    print(f"\n  Live composites (last per symbol):")
    print(live_comps.head(15).to_string())


if __name__ == '__main__':
    main()
