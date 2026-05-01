#!/usr/bin/env python3
"""ORB composite-drift daily monitor — runs after each trading day.

Reads `ORB SCORED` lines from journalctl for `today` (ET), recomputes the
composite features from cached daily/intraday bars, diffs each feature.
If any feature shows |Δ| > 0.001 (well above rounding noise), fires a
Telegram alert via the existing onemil-trader logging path.

Background
==========
2026-04-22: live composites ran ~0.09 below BT for same symbols/data — no
single failure mode identified, but the drift was sizable enough to flip
quintile assignments and change picks. That investigation led to the
diagnostic at `scripts/investigate_composite_drift.py`. This script
automates a daily run so future drift events surface immediately.

Behavior
========
- Run after market close (e.g. 16:30 ET via cron)
- If today had no ORB SCORED logs, exit silently
- If features all within ±0.001, exit with a one-line OK summary
- If any feature drifts: log to journalctl (Telegram-relayed via root
  logger) AND write a marker file to `/tmp/orb_drift_<date>.json` for
  follow-up

Usage
=====
    sudo python3 scripts/orb_drift_check_daily.py            # today (ET)
    sudo python3 scripts/orb_drift_check_daily.py 2026-04-30 # explicit date

Cron entry (suggested):
    30 16 * * 1-5  cd /home/ec2-user/onemil && \
        /usr/bin/python3 scripts/orb_drift_check_daily.py >> \
        logs/orb_drift_check.log 2>&1
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger('orb_drift_check_daily')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

ROOT = Path(__file__).parent.parent
DRIFT_THRESHOLD = 0.001  # |Δ live - BT| above this triggers alert

# Try to import zoneinfo (Python 3.9+) for ET-aware date
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
except ImportError:
    _ET = None


def today_et() -> str:
    """Return today's ET date as YYYY-MM-DD."""
    if _ET is not None:
        return datetime.now(_ET).strftime('%Y-%m-%d')
    # Fallback: UTC offset approximation
    return datetime.utcnow().strftime('%Y-%m-%d')


def run_drift_diagnostic(date_str: str) -> dict:
    """Invoke investigate_composite_drift.py and parse its output.

    Returns a dict with: matched_symbols, max_abs_drift, drifted_features.
    """
    diag = ROOT / 'scripts' / 'investigate_composite_drift.py'
    if not diag.exists():
        raise FileNotFoundError(f"Drift diagnostic script missing: {diag}")
    # Pull last 24h of journalctl for the diagnostic
    try:
        proc = subprocess.run(
            ['sudo', '/usr/bin/journalctl', '-u', 'onemil-trader',
             '--since', f'{date_str} 09:00:00', '--until', f'{date_str} 21:00:00',
             '--no-pager'],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        logger.error("journalctl timed out")
        return {'error': 'journalctl_timeout'}
    log_path = Path(f'/tmp/orb_drift_{date_str}.log')
    log_path.write_text(proc.stdout)

    try:
        out = subprocess.run(
            [sys.executable, str(diag), '--log', str(log_path), '--date', date_str],
            capture_output=True, text=True, timeout=300, cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        logger.error("drift diagnostic timed out")
        return {'error': 'diagnostic_timeout'}

    if out.returncode != 0:
        logger.error(f"drift diagnostic exit {out.returncode}: {out.stderr[:500]}")
        return {'error': 'diagnostic_failed', 'stderr': out.stderr[:500]}

    # Parse the output for the AGGREGATE FEATURE DIFFS section.
    text = out.stdout
    if 'AGGREGATE FEATURE DIFFS' not in text:
        return {'error': 'no_scored_lines', 'date': date_str}

    drifted = []
    for line in text.splitlines():
        # Lines look like:
        # "  gap_pct                                 -0.0001        -0.0004        +0.0005  (n=7)"
        parts = line.split()
        if len(parts) >= 4 and parts[0] in (
            'gap_pct', 'range_total_volume', 'range_avg_bar_range_pct',
            'range_size_pct', 'price_vs_20d_high_pct',
            'prev_day_close_position', 'range_close_position',
        ):
            try:
                mean_d = float(parts[1])
                min_d = float(parts[2])
                max_d = float(parts[3])
                worst = max(abs(min_d), abs(max_d))
                if worst > DRIFT_THRESHOLD:
                    drifted.append({
                        'feature': parts[0],
                        'mean': mean_d, 'min': min_d, 'max': max_d, 'worst': worst,
                    })
            except (ValueError, IndexError):
                continue

    return {
        'date': date_str,
        'drifted_features': drifted,
        'has_drift': len(drifted) > 0,
        'raw_output': text,
    }


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else today_et()
    logger.info(f"ORB drift check for {date_str} (ET)")
    result = run_drift_diagnostic(date_str)

    if 'error' in result:
        if result['error'] == 'no_scored_lines':
            logger.info(f"No ORB SCORED logs for {date_str} — no trading or service down. Skip.")
            return 0
        logger.error(f"Drift check failed: {result['error']}")
        # Write marker for follow-up
        marker = Path(f'/tmp/orb_drift_{date_str}.json')
        marker.write_text(json.dumps({'date': date_str, 'error': result['error']}, indent=2))
        return 2

    if not result['has_drift']:
        logger.info(
            f"ORB drift check OK for {date_str} — all features within ±{DRIFT_THRESHOLD}"
        )
        return 0

    # Drift detected
    n = len(result['drifted_features'])
    summary = '; '.join(
        f"{d['feature']}(worst={d['worst']:+.4f})"
        for d in result['drifted_features']
    )
    msg = (
        f"⚠️ ORB COMPOSITE DRIFT detected on {date_str} — "
        f"{n} feature(s): {summary}. "
        f"Picks may have diverged from BT. Run "
        f"scripts/investigate_composite_drift.py for full diff."
    )
    logger.error(msg)  # ERROR-level → relays to Telegram via root handler
    marker = Path(f'/tmp/orb_drift_{date_str}.json')
    marker.write_text(json.dumps(result, indent=2, default=str))
    return 1


if __name__ == '__main__':
    sys.exit(main())
