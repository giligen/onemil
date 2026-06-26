"""Touchgo evaluation audit log (diagnostic, append-only JSONL).

Records every live Rule M / Rule D evaluation — fire AND no-fire — so a
divergence between live's touchgo decision and the consolidated market bars
can be diagnosed after the fact. Motivated by three confirmed false-positive
exits (EIDO 6/9, OSCR 6/10, TSDD 6/23) where live fired `tag_bb` on bars
whose CONSOLIDATED bb_close_pos was ≥ 0.5 (Rule M should not have fired),
cutting real winners to ~breakeven.

The log captures the breakout-bar timestamp live keyed to, the OHLC live
computed from, and the resulting bb_close_pos. The companion script
`scripts/audit_touchgo_live_vs_consolidated.py` re-pulls Alpaca's
consolidated 1-min bars for each record and flags:
  * bb_close_pos mismatch  → live's streamed bar ≠ the consolidated bar
  * fire-decision flip      → live fired when consolidated says no (or vice versa)
  * re-keying miss          → breakout_bar_ts ≠ first consolidated bar > range_high

Design constraints:
  * NEVER raises into the trade path — every failure is swallowed + warned once.
  * Default ON (pure diagnostic, ~24 lines/day). Disable with
    `ORB_TOUCHGO_AUDIT=0`. Override path with `ORB_TOUCHGO_AUDIT_PATH`.
  * logs/ is gitignored — this is runtime data, never committed.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_lock = threading.Lock()
_warned = False  # one-shot: don't spam if the log dir is unwritable


def _enabled() -> bool:
    if os.getenv("ORB_TOUCHGO_AUDIT", "1").strip().lower() in (
        "0", "false", "no", "off", ""
    ):
        return False
    # Don't pollute the real production log from unrelated test runs. Tests that
    # specifically exercise the audit set ORB_TOUCHGO_AUDIT_PATH (a tmp path) and
    # are exempt; everything else under pytest is a no-op.
    if "PYTEST_CURRENT_TEST" in os.environ and not os.getenv("ORB_TOUCHGO_AUDIT_PATH"):
        return False
    return True


def _log_path() -> Path:
    override = os.getenv("ORB_TOUCHGO_AUDIT_PATH")
    if override:
        return Path(override)
    # repo_root/logs/touchgo_audit.jsonl  (trading/ -> repo root is parent)
    return Path(__file__).resolve().parent.parent / "logs" / "touchgo_audit.jsonl"


def record(rec: Dict[str, Any]) -> None:
    """Append one audit record as a JSON line. Best-effort, never raises.

    Args:
        rec: a flat dict of JSON-serialisable values. `schema_version` and
            `ts_utc` are added if absent (ts_utc only when a caller didn't
            supply one — callers should pass their own eval timestamp).
    """
    global _warned
    if not _enabled():
        return
    try:
        out = dict(rec)
        out.setdefault("schema_version", SCHEMA_VERSION)
        line = json.dumps(out, default=str, separators=(",", ":"))
        path = _log_path()
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:  # diagnostic must never break the trade path
        if not _warned:
            logger.warning(
                f"touchgo_audit: failed to write audit record ({exc!r}); "
                f"audit logging disabled for this run"
            )
            _warned = True
