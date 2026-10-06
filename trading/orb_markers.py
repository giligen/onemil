"""Nightly BT computed-day marker rows: ONE writer for the production and add-on pool books (spec 2026-10-06).

A marker row `(date, pool_id, candidates, picks, status, note)` is the proof that a night's BT ran for a day.
`status` is `ok` for a computed day (picks=0 is a computed day) and `failed` when the features build or the
pipeline died; `note` then carries the reason. Old files without the `status`/`note` columns read as `ok`.
The EOD report (`scripts/eod_sections.bt_marker`) turns a `failed` row into `P1 BT: FAILED (<reason>)` instead of
the ambiguous NO-DATA a missing row gives (10/5 lesson: the pipeline exited before any marker was written).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List

import pandas as pd

log = logging.getLogger(__name__)

MARKER_COLUMNS = ['date', 'pool_id', 'candidates', 'picks', 'status', 'note']
STATUS_OK = 'ok'
STATUS_FAILED = 'failed'
NOTE_MAX_CHARS = 300


def clean_reason(text: str, limit: int = NOTE_MAX_CHARS) -> str:
    """One-line, bounded rendering of a failure reason (marker notes live in a csv cell)."""
    one = ' '.join(str(text or '').split())
    return one[:limit] if len(one) > limit else one


def upsert_marker_rows(out_path: str, rows: List[Dict]) -> None:
    """Upsert marker rows keyed by (date, pool_id) into `out_path` (atomic write).

    Missing `status` defaults to ok and missing `note` to ''. Rows already in the file without those columns
    (written before 2026-10-06) are kept and read as ok. A new row for a key replaces the old one, so the latest
    run's verdict (ok or failed) is what the EOD report sees.
    """
    new = pd.DataFrame(rows)
    for col, default in (('candidates', 0), ('picks', 0), ('status', STATUS_OK), ('note', '')):
        if col not in new.columns:
            new[col] = default
    new = new[MARKER_COLUMNS]
    if os.path.exists(out_path):
        old = pd.read_csv(out_path, keep_default_na=False)
        for col, default in (('status', STATUS_OK), ('note', '')):
            if col not in old.columns:
                old[col] = default
        old['status'] = old['status'].replace('', STATUS_OK)
        keys = set(zip(new['date'].astype(str), new['pool_id'].astype(str)))
        old = old[[(a, b) not in keys for a, b in zip(old['date'].astype(str), old['pool_id'].astype(str))]]
        new = pd.concat([old[MARKER_COLUMNS], new], ignore_index=True)
    new = new.sort_values(['date', 'pool_id'])
    tmp = f"{out_path}.tmp"
    new.to_csv(tmp, index=False)
    os.replace(tmp, out_path)


def write_failed_markers(out_path: str, dates: Iterable[str], pool_id: str, reason: str) -> None:
    """Upsert a `status=failed` marker (note = reason) for every date of a run that died before computing them.

    Logs ERROR: a failed night must never be indistinguishable from "not computed yet".
    """
    days = list(dates)
    note = clean_reason(reason)
    log.error("BT FAILED marker: pool %s dates %s: %s", pool_id, days, note)
    upsert_marker_rows(out_path, [{'date': d, 'pool_id': pool_id, 'candidates': 0, 'picks': 0,
                                   'status': STATUS_FAILED, 'note': note} for d in days])
