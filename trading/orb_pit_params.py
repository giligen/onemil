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

Seed (2026-11-11): the live history starts with the 2026-09-13 refit, but the set in force BEFORE it is recoverable
from the orb.yaml snapshots the refit script takes (`research/orb_exit_resim/orb_refit_history_seed.jsonl`, same line format
plus `source`/`evidence`, every entry strictly older than the first live entry). `load_history` reads the seed first
and merges; seeded days report source 'seed'. Only days before the oldest seed entry still fall back to `current`.
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
DEFAULT_SEED = ROOT / 'research' / 'orb_exit_resim' / 'orb_refit_history_seed.jsonl'  # tracked (data/ is gitignored)
_USE_DEFAULT_SEED = object()   # sentinel: resolve DEFAULT_SEED at call time (tests may point it at None)
_ET = ZoneInfo('America/New_York')
_OPEN_ET = time(9, 30)

# history paths for which the pre-history WARNING was already emitted in this process (one per run)
_WARNED: set = set()


@dataclass(frozen=True)
class PitParams:
    """Parameters in force on one session day. `source` is 'pit' (live refit history), 'seed' (snapshot-recovered set) or 'current_fallback'."""
    params: Dict[str, Dict[str, float]]
    cutoffs: List[float]
    refit_ts: Optional[str]
    source: str


def _parse_ts(ts: str) -> datetime:
    """`YYYYMMDD_HHMMSS` (UTC, as written by orb_weekly_refit) -> aware UTC datetime."""
    return datetime.strptime(ts, '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)


def _read_entries(p: Path, source: str) -> List[dict]:
    """Parse one jsonl file into entries tagged `source`; raises ValueError naming the line on any malformed row."""
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
        out.append({'ts': d['ts'], 'dt': ts, 'params': params, 'cutoffs': cutoffs, 'source': source})
    out.sort(key=lambda r: r['dt'])
    return out


def load_history(history_path=DEFAULT_HISTORY, seed_path=_USE_DEFAULT_SEED) -> List[dict]:
    """Seed entries (older, snapshot-recovered) followed by the live refit history, sorted by timestamp.

    Raises on a missing/empty live file, any malformed line, or a seed entry not strictly older than the first live
    entry. `seed_path=None` = no seed; the default seed file missing = WARNING and no seed (an explicit path that is
    missing raises). Never returns a partial list: a silent skip would score a day with the wrong week's parameters."""
    p = Path(history_path)
    if not p.exists():
        raise FileNotFoundError(f"ORB refit history missing: {p} (cannot score point-in-time; "
                                f"use an explicit --params current for a research run)")
    live = _read_entries(p, 'pit')
    if not live:
        raise ValueError(f"ORB refit history {p} has no entries")
    default_seed = seed_path is _USE_DEFAULT_SEED
    sp = DEFAULT_SEED if default_seed else seed_path
    seed: List[dict] = []
    if sp is not None:
        sp = Path(sp)
        if sp.exists():
            seed = _read_entries(sp, 'seed')
        elif default_seed:
            logger.warning("ORB PIT params: seed file %s missing; days before %s fall back to CURRENT params "
                           "(look-ahead)", sp, live[0]['ts'])
        else:
            raise FileNotFoundError(f"ORB refit seed missing: {sp}")
    late = [h['ts'] for h in seed if h['dt'] >= live[0]['dt']]
    if late:
        raise ValueError(f"ORB refit seed {sp} entries {late} are not strictly older than the first live entry "
                         f"{live[0]['ts']} (overlap)")
    return seed + live


def session_open_utc(day: date) -> datetime:
    """09:30 ET of `day` as an aware UTC datetime (DST-correct)."""
    return datetime.combine(day, _OPEN_ET, tzinfo=_ET).astimezone(timezone.utc)


def params_for_day(day: date, history_path=DEFAULT_HISTORY, current: Optional[dict] = None,
                   warn: bool = True, seed_path=_USE_DEFAULT_SEED) -> PitParams:
    """The z-params and quintile cutoffs in force for the session on `day`.

    `current` = {'params': ..., 'cutoffs': ...} (orb.yaml literals), used only for days before the first history
    entry; if it is None such a day raises. `warn=False` lets a bulk caller emit one range summary itself."""
    hist = load_history(history_path, seed_path)
    day = day.date() if isinstance(day, datetime) else day
    cut = session_open_utc(day)
    prior = [h for h in hist if h['dt'] < cut]
    if prior:
        h = prior[-1]
        return PitParams(h['params'], h['cutoffs'], h['ts'], h['source'])
    if current is None:
        raise ValueError(f"{day} precedes the first ORB refit ({hist[0]['ts']}) and no `current` fallback given")
    key = str(Path(history_path))
    if warn and key not in _WARNED:
        _WARNED.add(key)
        logger.warning("ORB PIT params: %s precedes the first recorded refit %s; the history stores no "
                       "'previous' set (and no older snapshot seed), so days before it are scored with the CURRENT orb.yaml parameters "
                       "(look-ahead, NOT point-in-time)", day, hist[0]['ts'])
    return PitParams(current['params'], current['cutoffs'], None, 'current_fallback')
