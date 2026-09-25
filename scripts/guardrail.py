#!/usr/bin/env python3
"""scripts/guardrail.py — CLI for the G1 cumulative-ledger auto-pause
(docs/live_guardrails_spec_20260925.md). All logic lives in
trading/live_guardrail.py; this file only wires it to production config and
data/trades.db, and gives the owner a manual clear.

Usage:
  python scripts/guardrail.py --check              # ledger + pause rules, all books; writes state on a new pause
  python scripts/guardrail.py --clear orb "reason"  # manual, logged clear
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trading import live_guardrail as gr  # noqa: E402

ORB_YAML = ROOT / 'orb.yaml'
CONFIG_YAML = ROOT / 'config.yaml'


def stage_risk_usd(book: str) -> float:
    """This book's current per-trade risk from its live config file.

    orb: orb.yaml sizing.risk_per_trade_usd. bull_flag: config.yaml
    trading.risk_per_trade. hod_break: config.yaml trading.risk_per_trade too
    (it shares the engine's base risk even though it trades zero orders) —
    used only to size its reported ledger, never to pause it.
    A missing/unreadable config logs ERROR and returns 0.0 (rule 2/3
    thresholds then collapse to 0, i.e. maximally conservative — any loss
    trips them — rather than silently skipping the check).
    """
    try:
        if book == 'orb':
            cfg = yaml.safe_load(ORB_YAML.read_text())
            return float(cfg['sizing']['risk_per_trade_usd'])
        cfg = yaml.safe_load(CONFIG_YAML.read_text())
        return float(cfg['trading']['risk_per_trade'])
    except Exception as e:  # noqa: BLE001 - a decision aid must not crash
        gr.logger.error(f"guardrail: could not read stage risk for {book} ({e}) — using $0")
        return 0.0


def band_p5(book: str, n: int) -> Optional[float]:
    """band_p5_for() wired to the config knobs the live engine is actually running."""
    if book == 'orb':
        try:
            cfg = yaml.safe_load(ORB_YAML.read_text())
            veto_on = bool(cfg.get('filter', {}).get('catalyst_veto', {}).get('enabled'))
        except Exception as e:  # noqa: BLE001
            gr.logger.error(f"guardrail: could not read orb.yaml catalyst_veto ({e}) — assuming OFF")
            veto_on = False
        return gr.band_p5_for(book, n, orb_catalyst_veto=veto_on)
    if book == 'bull_flag':
        try:
            cfg = yaml.safe_load(CONFIG_YAML.read_text())
            adv_gate = int(cfg.get('scanner', {}).get('min_daily_volume', 200_000))
        except Exception as e:  # noqa: BLE001
            gr.logger.error(f"guardrail: could not read config.yaml scanner.min_daily_volume ({e}) — using 200000")
            adv_gate = 200_000
        return gr.band_p5_for(book, n, bf_min_daily_volume=adv_gate)
    return None


def run_check(db_path: Optional[Path] = None, state_path: Optional[Path] = None,
             notify: bool = True) -> int:
    """Ledger + pause rules for every book. Returns 1 iff a NEW pause fired
    this run (else 0) — a decision aid, not a hard gate on the caller.

    db_path/state_path resolve to gr.TRADES_DB/gr.STATE_PATH at CALL time
    (not as default-arg values bound at import), so tests can monkeypatch
    those module attributes and still exercise main()."""
    db_path = db_path if db_path is not None else gr.TRADES_DB
    state_path = gr.resolve_state_path(state_path)
    any_new_pause = False
    for book in gr.BOOKS:
        risk = stage_risk_usd(book)
        stats = gr.live_record(book, stage_risk_usd=risk, db_path=db_path)
        print(stats.line())
        if book not in gr.PAUSABLE_BOOKS:
            continue
        p5 = band_p5(book, stats.trailing_40_n)
        check = gr.evaluate_pause(stats, stage_risk_usd=risk, band_p5=p5)
        already = gr.is_paused(book, path=state_path)
        if check.should_pause:
            gr.pause_book(check, stage_risk_usd=risk, path=state_path, notify=notify)
            if not already:
                any_new_pause = True
            print(f"  -> PAUSED ({check.rule}): {check.detail}")
        else:
            print(f"  -> within band (rule: {check.rule or 'none'}) — {check.detail}"
                 + (" [state says PAUSED — clear manually if resolved]" if already else ""))
    return 1 if any_new_pause else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true', help='run the ledger + pause rules for every book')
    g.add_argument('--clear', nargs=2, metavar=('BOOK', 'REASON'),
                   help='manually clear a guardrail pause, e.g. --clear orb "latency fix rehearsed"')
    ap.add_argument('--no-notify', action='store_true', help='skip the Telegram on a new pause (tests)')
    a = ap.parse_args()

    if a.check:
        return run_check(notify=not a.no_notify)

    book, reason = a.clear
    # path resolved at call time (not import-time default) so a monkeypatched
    # gr.STATE_PATH (tests) or a real one (prod) is always honoured.
    entry = gr.clear_pause(book, reason, path=gr.STATE_PATH)
    print(f"{book}: cleared by {entry['cleared_by']} at {entry['cleared_at_utc']} — {reason}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
