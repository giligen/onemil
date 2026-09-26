#!/usr/bin/env python3
"""Synthetic unit tests for cell I4 (13D): initial vs amendment, the next-session rule,
and one slot per symbol per hold. No data files or Panel() needed - pure functions only."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cell_13d import signal_session, dedupe_overlapping  # noqa: E402
from fetch_13d import parse_hit  # noqa: E402


# ------------------------------------------------------------------ initial vs amendment


def test_parse_hit_keeps_initial_13d():
    src = dict(form='SC 13D', adsh='0001-16-000001', file_date='2016-03-01',
               ciks=['0001111111', '0002222222'],
               display_names=['Acme Corp  (ACME)  (CIK 0001111111)',
                               'Some Fund LP  (CIK 0002222222)'])
    r = parse_hit(src)
    assert r is not None
    assert r['ticker_raw'] == 'ACME'
    assert r['subject_cik'] == 1111111
    assert r['accession'] == '0001-16-000001'


def test_parse_hit_drops_amendment():
    src = dict(form='SC 13D/A', adsh='0001-16-000002', file_date='2016-03-05',
               ciks=['0001111111'], display_names=['Acme Corp  (ACME)  (CIK 0001111111)'])
    assert parse_hit(src) is None


def test_parse_hit_no_ticker_still_maps_by_cik():
    src = dict(form='SC 13D', adsh='0001-16-000003', file_date='2016-04-01',
               ciks=['0003333333'], display_names=['Private Target Inc  (CIK 0003333333)'])
    r = parse_hit(src)
    assert r is not None
    assert r['ticker_raw'] is None
    assert r['subject_cik'] == 3333333


# ------------------------------------------------------------------ next-session rule


def test_signal_session_strictly_after_filing():
    sess = np.array(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
                     dtype='datetime64[D]').astype('datetime64[ns]')
    # filed exactly on a session date -> the NEXT session, never same-day
    filed = np.array(['2024-01-02'], dtype='datetime64[D]').astype('datetime64[ns]')
    S = signal_session(sess, filed)
    assert S[0] == 1 and sess[S[0]] == np.datetime64('2024-01-03')


def test_signal_session_weekend_filing_lands_on_monday():
    sess = np.array(['2024-01-05', '2024-01-08', '2024-01-09'],
                     dtype='datetime64[D]').astype('datetime64[ns]')
    # filed Saturday 1/6 (no session that day) -> first session strictly after is 1/8
    filed = np.array(['2024-01-06'], dtype='datetime64[D]').astype('datetime64[ns]')
    S = signal_session(sess, filed)
    assert sess[S[0]] == np.datetime64('2024-01-08')


def test_signal_session_beyond_panel_end_is_out_of_range():
    sess = np.array(['2024-01-02', '2024-01-03'], dtype='datetime64[D]').astype('datetime64[ns]')
    filed = np.array(['2024-01-03'], dtype='datetime64[D]').astype('datetime64[ns]')
    S = signal_session(sess, filed)
    assert S[0] == len(sess)   # caller must drop this (no data yet)


# ------------------------------------------------------------------ one slot per symbol per hold


def test_dedupe_keeps_first_signal_when_overlapping():
    df = pd.DataFrame({'symbol': ['AAA', 'AAA', 'AAA'], 'S': [10, 15, 35]})
    out = dedupe_overlapping(df, hold=20)
    # S=15 is inside [10, 30) -> dropped; S=35 is outside -> kept
    assert sorted(out['S'].tolist()) == [10, 35]


def test_dedupe_independent_across_symbols():
    df = pd.DataFrame({'symbol': ['AAA', 'BBB'], 'S': [10, 12]})
    out = dedupe_overlapping(df, hold=20)
    assert sorted(out['S'].tolist()) == [10, 12]


def test_dedupe_boundary_is_exclusive_of_extension():
    # a signal exactly at S0+hold is a NEW, independent slot (not an overlap)
    df = pd.DataFrame({'symbol': ['AAA', 'AAA'], 'S': [10, 30]})
    out = dedupe_overlapping(df, hold=20)
    assert sorted(out['S'].tolist()) == [10, 30]


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
