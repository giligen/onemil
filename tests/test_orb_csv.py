"""trading/orb_csv.read_orb_csv — tickers named NA/NAN/NULL survive (2026-09-05).

Regression for the entered-inclusive regen crash: pd.read_csv turned the
symbol "NA" (Nano Labs, a no-fill row) into NaN and the pipeline's
build_atr14_lookup died on sorted(str, float).
"""
import numpy as np
import pandas as pd
import pytest

from trading.orb_csv import read_orb_csv


@pytest.fixture
def csv_path(tmp_path):
    df = pd.DataFrame({
        'symbol': ['NA', 'NAN', 'NULL', 'N/A', 'AAPL'],
        'date': ['2025-01-02'] * 5,
        'entered': [0, 1, 1, 0, 1],
        'pnl': [0.0, np.nan, 12.5, 0.0, -3.0],
    })
    p = tmp_path / 'orb_features_test.csv'
    df.to_csv(p, index=False)
    return str(p)


def test_pandas_default_loses_tickers(csv_path):
    """Documents the trap this helper exists for: NA, NULL and N/A are
    pandas default NA tokens (uppercase NAN is not)."""
    lost = pd.read_csv(csv_path)['symbol'].isna()
    assert lost.sum() == 3
    assert not lost[[1, 4]].any()


def test_tickers_survive_and_numeric_nan_is_kept(csv_path):
    df = read_orb_csv(csv_path)
    assert df['symbol'].tolist() == ['NA', 'NAN', 'NULL', 'N/A', 'AAPL']
    assert df['symbol'].map(type).eq(str).all()
    assert df['pnl'].isna().tolist() == [False, True, False, False, False]
    assert df['entered'].dtype.kind in 'iu'


def test_kwargs_pass_through(csv_path):
    df = read_orb_csv(csv_path, usecols=['symbol', 'date'])
    assert list(df.columns) == ['symbol', 'date']
    assert df['symbol'].iloc[0] == 'NA'


def test_pipeline_sort_pattern_no_longer_crashes(csv_path):
    """The exact failure: grouping by symbol then sorted() over the keys."""
    df = read_orb_csv(csv_path)
    by_sym = {}
    for sym, day in zip(df['symbol'], df['date']):
        by_sym.setdefault(sym, set()).add(day)
    assert [s for s, _ in sorted(by_sym.items())][0] == 'AAPL'
