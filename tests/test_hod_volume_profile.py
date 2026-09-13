"""hod_volume_profile: Database helpers (real sqlite) + the nightly checkpoint builder."""
from datetime import datetime, timedelta, timezone

import pytest

from persistence.database import Database
from trading.hod_break import VP_CHECKPOINTS
from scripts.build_hod_volume_profile import checkpoints_for


@pytest.fixture
def db(tmp_path):
    return Database(cache_path=str(tmp_path / 'cache.db'), trades_path=str(tmp_path / 'trades.db'))


def test_checkpoints_accumulate_by_clock():
    base = datetime(2026, 9, 14, 13, 30, tzinfo=timezone.utc)          # 09:30 ET
    bars = [{'timestamp': base + timedelta(minutes=i), 'volume': 100} for i in range(390)]
    cp = checkpoints_for(bars)
    assert cp[575] == 600 and cp[585] == 1600 and cp[600] == 3100 and cp['day'] == 39000
    assert cp[900] == 33100


def test_premarket_and_afterhours_bars_are_ignored():
    base = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)           # 08:00 ET
    bars = [{'timestamp': base + timedelta(minutes=i), 'volume': 1} for i in range(90)]   # all premarket
    assert checkpoints_for(bars)['day'] == 0


def test_upsert_and_read_back_prior_days(db):
    rows = [{'symbol': 'ABC', 'bar_date': d, 'cut_minute': 575, 'cum_volume': v, 'day_volume': 10 * v}
            for d, v in (('2026-09-08', 100), ('2026-09-09', 200), ('2026-09-10', 300), ('2026-09-11', 400))]
    assert db.upsert_hod_volume_profile(rows) == 4
    assert db.get_hod_volume_profile('ABC', 575, before_date='2026-09-11', n_days=2) == [300.0, 200.0]
    assert db.get_hod_volume_profile('ABC', 575, before_date='2026-09-08') == []
    assert db.get_hod_volume_profile('ZZZ', 575, before_date='2026-09-11') == []
    db.upsert_hod_volume_profile([{'symbol': 'ABC', 'bar_date': '2026-09-10', 'cut_minute': 575, 'cum_volume': 999, 'day_volume': 1}])
    assert db.get_hod_volume_profile('ABC', 575, before_date='2026-09-11', n_days=1) == [999.0]


def test_all_checkpoints_written_per_symbol_day(db):
    rows = [{'symbol': 'ABC', 'bar_date': '2026-09-11', 'cut_minute': int(c), 'cum_volume': i, 'day_volume': 9} for i, c in enumerate(VP_CHECKPOINTS)]
    assert db.upsert_hod_volume_profile(rows) == len(VP_CHECKPOINTS)
    assert db.get_hod_volume_profile('ABC', 900, before_date='2026-09-12') == [float(len(VP_CHECKPOINTS) - 1)]
