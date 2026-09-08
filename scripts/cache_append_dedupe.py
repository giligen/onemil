#!/usr/bin/env python3
"""Drop rows from a freshly built temp cache that already exist in the
production bull-flag cache (key: symbol, date, entry_time_et).

Why (2026-09-08): the nightly roll-forward builds its gap with monthly
chunking, so a gap that starts mid-month re-walks the month's earlier days
and appended duplicates (three 9/4 rows on 9/7 — one of them, PATX, with a
DIFFERENT exit from a transient bar fetch). The production row is the
reference; a re-walk never overwrites it.

Usage: cache_append_dedupe.py PROD_CSV TMP_CSV   (rewrites TMP_CSV in place; prints counts)
"""
import sys
import pandas as pd

KEY = ['symbol', 'date', 'entry_time_et']


def dedupe_against(prod: pd.DataFrame, tmp: pd.DataFrame) -> pd.DataFrame:
    """Rows of `tmp` whose key is not already in `prod` (order preserved)."""
    if tmp.empty:
        return tmp
    have = set(map(tuple, prod[KEY].astype(str).values.tolist()))
    keep = [tuple(r) not in have for r in tmp[KEY].astype(str).values.tolist()]
    return tmp[keep]


def main() -> int:
    prod_p, tmp_p = sys.argv[1], sys.argv[2]
    rd = dict(keep_default_na=False, na_values=[''], dtype=str)
    prod = pd.read_csv(prod_p, **rd)
    tmp = pd.read_csv(tmp_p, **rd)
    out = dedupe_against(prod, tmp)
    dropped = len(tmp) - len(out)
    out.to_csv(tmp_p, index=False)
    print(f"cache_append_dedupe: {len(tmp)} built rows, {dropped} already in production (dropped), {len(out)} to append")
    return 0


if __name__ == '__main__':
    sys.exit(main())
