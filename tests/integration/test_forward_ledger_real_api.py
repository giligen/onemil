"""Real-API tier for research/day_breadth/forward_ledger.py: one cached TEST day against the production bar DBs and
live Alpaca NBBO (41 s, rewrites the research forward_*.csv). Moved out of the unit tier 2026-10-31 (test-suite speed
spec); the unit tests stay in tests/test_forward_ledger.py."""
import sys

import pandas as pd
import pytest

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, f'{ROOT}/research/day_breadth')
sys.path.insert(0, f'{ROOT}/research/hod_consol')
import forward_ledger as fl  # noqa: E402


@pytest.mark.integration
def test_process_day_cached_test_day():
    """One cached TEST day (bars already in bars_sip.db/bars_rth.db) -- real Alpaca NBBO calls."""
    ok = fl.process_day('2026-06-01', send_telegram=False)
    assert ok is True
    summary = pd.read_csv(fl.SUMMARY_CSV)
    row = summary[summary.date == '2026-06-01']
    assert len(row) == 1
    assert row.n_signals.iloc[0] >= 1
