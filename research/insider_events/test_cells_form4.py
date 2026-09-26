#!/usr/bin/env python3
"""Unit tests for cells_form4.py / fetch_form345.py, synthetic data only (no network,
no real panel). Covers: next-session rule, cluster window, 10b5-1 exclusion, amendment
exclusion, opportunistic 12-month lookback, no-double-slot dedup."""
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cells_form4 as C
import fetch_form345 as F


class FakePanel:
    """Minimal stand-in for research/multiday/run_final.Panel exposing only the
    attributes cells_form4 needs for signal construction (no prices/returns math)."""

    def __init__(self, n_s=4, n_d=40, start='2021-01-01'):
        self.n_s, self.n_d = n_s, n_d
        self.symbols = [f'SYM{i}' for i in range(n_s)]
        self.sidx = {s: i for i, s in enumerate(self.symbols)}
        self.sessions = pd.date_range(start, periods=n_d, freq='D')
        self.sess_np = self.sessions.values
        self.elig = np.ones((n_s, n_d), dtype=bool)
        self.taint = np.zeros((n_s, n_d), dtype=bool)


@pytest.fixture
def fp():
    return FakePanel()


# ------------------------------------------------------------------ next-session rule


def test_next_session_rule_first_session_strictly_after_filing(fp):
    """Signal session S must be the first panel session strictly AFTER filing_date, even
    when filing_date exactly equals a panel session date."""
    pur = pd.DataFrame({
        'symbol': ['SYM0', 'SYM0', 'SYM0'],
        'owner_cik': ['1', '1', '1'],
        'value': [1.0, 1.0, 1.0],
        'trans_date': pd.to_datetime(['2021-01-05'] * 3),
        # exact session match, weekend/gap, and a date before the panel starts
        'filing_date': pd.to_datetime(['2021-01-05', '2021-01-10', '2020-06-01']),
    })
    out = C.load_purchases(fp, pur)
    # row0: filing_date == session index 4 (2021-01-05) -> S must be index 5 (2021-01-06)
    row0 = out[out['filing_date'] == '2021-01-05'].iloc[0]
    assert fp.sessions[row0['S']] > pd.Timestamp('2021-01-05')
    assert fp.sessions[row0['S'] - 1] <= pd.Timestamp('2021-01-05')
    # row2 (2020-06-01) is before the panel start -> dropped as out-of-range
    assert (out['filing_date'] == '2020-06-01').sum() == 0
    assert len(out) == 2


# ------------------------------------------------------------------ cluster window (I1)


def test_cluster_window_boundary_and_two_owner_requirement(fp):
    """A single owner's repeated purchases never cluster; a 2nd distinct owner within the
    10-session window (inclusive) triggers it; outside the window it does not."""
    base = pd.Timestamp('2021-01-01')
    pur = pd.DataFrame({
        'symbol': ['SYM0'] * 4,
        'owner_cik': ['A', 'A', 'B', 'C'],
        'value': [60_000.0, 10_000.0, 60_000.0, 60_000.0],
        'trans_date': [base] * 4,
        'filing_date': [base, base + pd.Timedelta(days=1),
                         base + pd.Timedelta(days=9),   # within the 10-session window of row0
                         base + pd.Timedelta(days=25)],  # far outside any window from row0/1
    })
    loaded = C.load_purchases(fp, pur).reset_index(drop=True)
    cand = C.build_i1(loaded)
    fired_S = set(zip(cand['symbol'], cand['S']))
    # row index 2 (owner B, day 9): 2 distinct owners (A,B) in its trailing window, value
    # 60k+60k=120k >= 100k -> fires
    assert (loaded.loc[2, 'symbol'], loaded.loc[2, 'S']) in fired_S
    # row index 3 (owner C, day 25): only itself in its own trailing 10-session window
    # (rows 0-2 are 16+ sessions back) -> single owner -> must not fire
    assert (loaded.loc[3, 'symbol'], loaded.loc[3, 'S']) not in fired_S
    # row index 1 (owner A again, day 1): still only owner A in window -> must not fire
    assert (loaded.loc[1, 'symbol'], loaded.loc[1, 'S']) not in fired_S


def test_cluster_requires_value_threshold(fp):
    """Two distinct owners but total value < $100K must NOT fire I1."""
    base = pd.Timestamp('2021-01-01')
    pur = pd.DataFrame({
        'symbol': ['SYM1', 'SYM1'],
        'owner_cik': ['A', 'B'],
        'value': [10_000.0, 10_000.0],
        'trans_date': [base, base],
        'filing_date': [base, base + pd.Timedelta(days=2)],
    })
    loaded = C.load_purchases(fp, pur)
    cand = C.build_i1(loaded)
    assert len(cand) == 0


# ------------------------------------------------------------------ opportunistic lookback (I3)


def test_opportunistic_12_month_lookback(fp):
    """No prior purchase (or a prior purchase > 365 days back) by the SAME owner+issuer ->
    opportunistic; a purchase within 365 days -> routine, excluded."""
    pur = pd.DataFrame({
        'symbol': ['SYM0', 'SYM0', 'SYM1', 'SYM1'],
        'si': [0, 0, 1, 1],
        'S': [5, 20, 5, 20],
        'owner_cik': ['A', 'A', 'B', 'B'],
        'value': [30_000.0, 30_000.0, 30_000.0, 30_000.0],
        'trans_date': pd.to_datetime(['2020-01-01', '2020-12-01',   # 335 days apart -> routine
                                       '2020-01-01', '2021-06-01']),  # 517 days apart -> opportunistic
        'is_officer': [False] * 4,
        'is_director': [False] * 4,
    })
    cand = C.build_i3(pur)
    # SYM0's second purchase (2020-12-01) is routine (within 12mo of 2020-01-01) -> excluded
    assert not ((cand['symbol'] == 'SYM0') & (cand['S'] == 20)).any()
    # SYM0's FIRST purchase has no history at all -> opportunistic, included
    assert ((cand['symbol'] == 'SYM0') & (cand['S'] == 5)).any()
    # SYM1's second purchase is > 365 days after the first -> opportunistic, included
    assert ((cand['symbol'] == 'SYM1') & (cand['S'] == 20)).any()


# ------------------------------------------------------------------ no-double-slot dedup


def test_no_double_slot_dedup(fp):
    """A second signal on the same symbol inside the hold window of the first is dropped;
    one that lands after the first position's hold has fully elapsed is kept."""
    cand = pd.DataFrame({
        'symbol': ['SYM0', 'SYM0', 'SYM0'],
        'si': [0, 0, 0],
        'S': [5, 15, 25],   # hold=20: 2nd signal (S=15) is inside [5,25) -> dropped;
        'value': [1.0, 1.0, 1.0],  # 3rd (S=25) is exactly at the boundary -> kept
    })
    out = C.gate_and_dedup(cand, fp, hold=20)
    assert sorted(out['S'].tolist()) == [5, 25]


def test_gate_drops_ineligible_and_tainted(fp):
    fp.elig[0, 10] = False
    fp.taint[0, 20] = True
    cand = pd.DataFrame({'symbol': ['SYM0', 'SYM0', 'SYM0'], 'si': [0, 0, 0],
                          'S': [10, 20, 30], 'value': [1.0, 1.0, 1.0]})
    out = C.gate_and_dedup(cand, fp, hold=20)
    assert out['S'].tolist() == [30]


# ------------------------------------------------------------------ fetch-level filters


@pytest.fixture
def tmp_quarter(tmp_path):
    qtr = 'test0000'
    tmpdir = str(tmp_path / qtr)
    os.makedirs(tmpdir, exist_ok=True)
    yield qtr, tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def _write_tsvs(tmpdir, trans_rows, sub_rows, own_rows):
    pd.DataFrame(trans_rows).to_csv(f'{tmpdir}/NONDERIV_TRANS.tsv', sep='\t', index=False)
    pd.DataFrame(sub_rows).to_csv(f'{tmpdir}/SUBMISSION.tsv', sep='\t', index=False)
    pd.DataFrame(own_rows).to_csv(f'{tmpdir}/REPORTINGOWNER.tsv', sep='\t', index=False)


def test_10b5_1_exclusion_and_amendment_exclusion(tmp_quarter, monkeypatch):
    qtr, tmpdir = tmp_quarter
    monkeypatch.setattr(F, 'EXTD', os.path.dirname(tmpdir))

    trans_rows = [
        dict(ACCESSION_NUMBER='ACC1', TRANS_DATE='01-JAN-2024', TRANS_CODE='P',
             TRANS_SHARES=100, TRANS_PRICEPERSHARE=10.0, TRANS_ACQUIRED_DISP_CD='A'),
        dict(ACCESSION_NUMBER='ACC2', TRANS_DATE='02-JAN-2024', TRANS_CODE='P',
             TRANS_SHARES=100, TRANS_PRICEPERSHARE=10.0, TRANS_ACQUIRED_DISP_CD='A'),
        dict(ACCESSION_NUMBER='ACC3', TRANS_DATE='03-JAN-2024', TRANS_CODE='P',
             TRANS_SHARES=100, TRANS_PRICEPERSHARE=10.0, TRANS_ACQUIRED_DISP_CD='A'),
    ]
    sub_rows = [
        # ACC1: a clean Form 4 -- should survive
        dict(ACCESSION_NUMBER='ACC1', FILING_DATE='02-JAN-2024', DOCUMENT_TYPE='4',
             ISSUERCIK='1', ISSUERTRADINGSYMBOL='AAA', AFF10B5ONE='0'),
        # ACC2: planned 10b5-1 trade -- must be excluded
        dict(ACCESSION_NUMBER='ACC2', FILING_DATE='03-JAN-2024', DOCUMENT_TYPE='4',
             ISSUERCIK='2', ISSUERTRADINGSYMBOL='BBB', AFF10B5ONE='1'),
        # ACC3: a Form 4/A amendment -- must be excluded
        dict(ACCESSION_NUMBER='ACC3', FILING_DATE='04-JAN-2024', DOCUMENT_TYPE='4/A',
             ISSUERCIK='3', ISSUERTRADINGSYMBOL='CCC', AFF10B5ONE='0'),
    ]
    own_rows = [dict(ACCESSION_NUMBER=a, RPTOWNERCIK='9', RPTOWNER_RELATIONSHIP='Director',
                      RPTOWNER_TITLE='') for a in ('ACC1', 'ACC2', 'ACC3')]
    _write_tsvs(tmpdir, trans_rows, sub_rows, own_rows)

    out = F.load_quarter(qtr)
    assert set(out['symbol']) == {'AAA'}
    assert len(out) == 1


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
