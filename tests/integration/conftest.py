"""tests/integration is OPT-IN: these tests reach the real Alpaca API (paper orders, live-API reads, and one of them
rewrites research/day_breadth/forward_*.csv). They must never run from a plain ``pytest tests`` — on 2026-10-31 a
suite run without ``--ignore=tests/integration`` collected them by accident. Run them deliberately:

    ONEMIL_RUN_INTEGRATION=1 python3 -m pytest tests/integration -q -p no:randomly

Without the variable every test in this directory is skipped at collection with the reason below (a skip, never a
failure, never a silent pass).
"""
import os

import pytest

_OPT_IN = "ONEMIL_RUN_INTEGRATION"


def pytest_collection_modifyitems(config, items):
    """Skip every test under tests/integration unless ONEMIL_RUN_INTEGRATION=1 is set in the environment."""
    if os.environ.get(_OPT_IN) == "1":
        return
    marker = pytest.mark.skip(reason=f"real-API integration test: set {_OPT_IN}=1 to run tests/integration deliberately")
    for item in items:
        if "tests/integration" in str(item.fspath).replace(os.sep, "/"):
            item.add_marker(marker)
