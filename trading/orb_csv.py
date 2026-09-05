"""ORB CSV reader — ticker-safe (2026-09-05).

pandas' default `na_values` turn the TICKERS "NA", "NAN", "NULL", "N/A" into
NaN. The entered-inclusive features regen surfaced it: one no-fill row for
Nano Labs ("NA") became a float symbol and crashed the pipeline's ATR lookup
(`sorted()` of str and float). Every ORB CSV that carries a symbol column —
features, books, news catalyst, PM-mult, goldens — must be read through
this helper: only the empty cell is missing (pandas writes NaN as '').
"""
from typing import Any

import pandas as pd


def read_orb_csv(path: str, **kwargs: Any) -> pd.DataFrame:
    """`pd.read_csv` with the default NA tokens disabled so tickers like
    "NA"/"NAN"/"NULL" survive as strings. Empty cells are still NaN."""
    kwargs.setdefault('keep_default_na', False)
    kwargs.setdefault('na_values', [''])
    return pd.read_csv(path, **kwargs)
