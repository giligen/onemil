"""Point-in-time (PIT) ORB selection parameters: the z-params + quintile cutoffs in force on a session day.

Mechanism (why this module exists): `scripts/orb_weekly_refit.py` rewrites the composite's z-params and the
`quintile_cutoffs` in orb.yaml every Sunday 20:00 UTC and appends each fit to `logs/orb_refit_history.jsonl`
(`ts` = UTC `YYYYMMDD_HHMMSS`, `params`, `cutoffs`). LIVE scores each session with the set fitted before that
session. A backtest that reads the CURRENT orb.yaml once and re-scores ALL history with it (a) ranks past days with
parameters fitted on data AFTER the scored day (look-ahead) and (b) breaks live/BT parity: cell 1,733 found the
published top-8 equal to live's on 7 of 17 logged days and the PIT top-8 on 11 of 17, quintiles 91.6 % vs 100 %.
`params_for_day` is the ONE place that answers "which parameters were in force that day", shared by the BT
pipeline and the EOD replica, so both score every day exactly as live did.

Rule: an entry applies to a session day iff its refit timestamp precedes that day's 09:30 ET open. A Sunday refit
(20:00 UTC) therefore applies from Monday; a refit on the session day itself applies only if it ran before 09:30 ET
(refits run Sundays, so this is a guard, not a path). Days before the first history entry have no recorded
parameters (the jsonl stores only the NEW fit, no "previous"), so they fall back to `current` with a loud WARNING.
Missing or malformed history never silently degrades to today's set: it raises.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY = ROOT / 'logs' / 'orb_refit_history.jsonl'
_ET = ZoneInfo('America/New_York')
_OPEN_ET = time(9, 30)

# history paths for which the pre-history WARNING was already emitted in this process (one per run)
_WARNED: set = set()


@dataclass(frozen=True)
class PitParams:
    """Parameters in force on one session day. `source` is 'pit', 'pre_history' or 'current_fallback'."""
    params: Dict[str, Dict[str, float]]
    cutoffs: List[float]
    refit_ts: Optional[str]
    source: str


def _parse_ts(ts: str) -> datetime:
    """`YYYYMMDD_HHMMSS` (UTC, as written by orb_weekly_refit) -> aware UTC datetime."""
    return datetime.strptime(ts, '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)


def load_history(history_path=DEFAULT_HISTORY) -> List[dict]:
    """Read the refit history sorted by timestamp. Raises on a missing file, an empty file or any malformed line.

    Never returns a partial list: a silent skip would make a day score with the wrong week's parameters."""
    p = Path(history_path)
    if not p.exists():
        raise FileNotFoundError(f"ORB refit history missing: {p} (cannot score point-in-time; "
                                f"use an explicit --params current for a research run)")
    out = []
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            ts = _parse_ts(d['ts'])
            params = {f: {'mean': float(v['mean']), 'std': float(v['std']), 'sign': int(v['sign'])}
                      for f, v in d['params'].items()}
            cutoffs = [float(x) for x in d['cutoffs']]
            if not params or len(cutoffs) != 4:
                raise ValueError(f"empty params or {len(cutoffs)} cutoffs (need 4)")
        except Exception as e:  # noqa: BLE001 - re-raised with the line number
            raise ValueError(f"ORB refit history {p} line {i} malformed: {e!r}") from e
        out.append({'ts': d['ts'], 'dt': ts, 'params': params, 'cutoffs': cutoffs})
    if not out:
        raise ValueError(f"ORB refit history {p} has no entries")
    out.sort(key=lambda r: r['dt'])
    return out


def session_open_utc(day: date) -> datetime:
    """09:30 ET of `day` as an aware UTC datetime (DST-correct)."""
    return datetime.combine(day, _OPEN_ET, tzinfo=_ET).astimezone(timezone.utc)


def params_for_day(day: date, history_path=DEFAULT_HISTORY, current: Optional[dict] = None,
                   warn: bool = True) -> PitParams:
    """The z-params and quintile cutoffs in force for the session on `day`.

    `current` = {'params': ..., 'cutoffs': ...} (orb.yaml literals), used only for days before the first history
    entry; if it is None such a day raises. `warn=False` lets a bulk caller emit one range summary itself."""
    hist = load_history(history_path)
    day = day.date() if isinstance(day, datetime) else day
    cut = session_open_utc(day)
    prior = [h for h in hist if h['dt'] < cut]
    if prior:
        h = prior[-1]
        return PitParams(h['params'], h['cutoffs'], h['ts'], 'pit')
    if current is None:
        raise ValueError(f"{day} precedes the first ORB refit ({hist[0]['ts']}) and no `current` fallback given")
    key = str(Path(history_path))
    if warn and key not in _WARNED:
        _WARNED.add(key)
        logger.warning("ORB PIT params: %s precedes the first recorded refit %s; the history stores no "
                       "'previous' set, so days before it are scored with the CURRENT orb.yaml parameters "
                       "(look-ahead, NOT point-in-time)", day, hist[0]['ts'])
    return PitParams(current['params'], current['cutoffs'], None, 'current_fallback')
