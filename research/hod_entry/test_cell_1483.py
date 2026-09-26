"""Unit tests for research/hod_entry/cell_1483.py -- pure-function pieces only (masking, PIT
runway lookup, prev-session calendar walk). No network, no Anthropic/Alpaca/SEC calls."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, '/home/ec2-user/onemil')
from research.hod_entry import cell_1483 as m


def test_mask_text_replaces_own_symbol_and_title():
    out = m.mask_text('AAPL shares rose after Apple Inc. announced a buyback', 'AAPL',
                       'Apple Inc.', ['AAPL'])
    assert 'AAPL' not in out
    assert 'Apple' not in out
    assert out.count('the company') >= 1


def test_mask_text_replaces_other_listed_symbols():
    out = m.mask_text('MSFT rallied while AAPL dipped', 'AAPL', 'Apple Inc.', ['AAPL', 'MSFT'])
    assert 'MSFT' not in out
    assert 'another company' in out


def test_mask_text_handles_empty():
    assert m.mask_text('', 'AAPL', 'Apple Inc.', ['AAPL']) == ''
    assert m.mask_text(None, 'AAPL', 'Apple Inc.', ['AAPL']) is None


def test_prev_session_walks_back_one_trading_day():
    cal = np.array(['2025-06-27', '2025-06-30', '2025-07-01', '2025-07-02'])
    assert m.prev_session('2025-07-01', cal) == '2025-06-30'
    assert m.prev_session('2025-06-30', cal) == '2025-06-27'


def test_prev_session_before_calendar_start_logs_and_falls_back(capsys):
    cal = np.array(['2025-06-30', '2025-07-01'])
    out = m.prev_session('2025-06-30', cal)
    assert out == '2025-06-30'
    captured = capsys.readouterr()
    assert 'WARNING' in captured.out


def test_latest_before_picks_most_recent_filed_before_cutoff():
    units = [
        dict(form='10-Q', filed='2025-05-01', end='2025-03-31', val=100),
        dict(form='10-Q', filed='2025-08-01', end='2025-06-30', val=200),
        dict(form='10-Q', filed='2025-11-01', end='2025-09-30', val=300),  # filed after cutoff
    ]
    best = m._latest_before(units, '2025-09-01', want_duration=False)
    assert best['val'] == 200


def test_latest_before_separates_duration_from_instant():
    units = [
        dict(form='10-Q', filed='2025-05-01', end='2025-03-31', val=100),                    # instant
        dict(form='10-Q', filed='2025-05-01', start='2025-01-01', end='2025-03-31', val=-50),  # duration
    ]
    instant = m._latest_before(units, '2025-06-01', want_duration=False)
    duration = m._latest_before(units, '2025-06-01', want_duration=True)
    assert instant['val'] == 100
    assert duration['val'] == -50


def test_latest_before_returns_none_when_nothing_qualifies():
    units = [dict(form='10-Q', filed='2025-11-01', end='2025-09-30', val=1)]
    assert m._latest_before(units, '2025-06-01', want_duration=False) is None


def test_hard_catalyst_classes_match_prereg_set():
    assert m.HARD_CLASSES == {'earnings_guidance', 'fda_clinical', 'ma_strategic',
                               'contract_product'}
    assert 'financing_dilution' not in m.HARD_CLASSES
    assert 'no_news' not in m.HARD_CLASSES


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
