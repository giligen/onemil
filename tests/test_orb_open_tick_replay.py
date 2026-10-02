"""scripts/orb_open_tick_replay.py opens cache.db strictly read-only."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from orb_open_tick_replay import StubAlpaca, load_snapshots, open_readonly_db  # noqa: E402


def _make_cache(path: Path) -> None:
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE daily_bars (symbol TEXT, bar_date TEXT, open REAL, high REAL, low REAL,"
              " close REAL, volume INTEGER, fetched_at TEXT)")
    c.executemany("INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?,'x')",
                  [('AAA', '2026-10-01', 9, 9, 9, 9.0, 900000), ('AAA', '2026-10-02', 10, 10, 10, 10, 1)])
    c.commit()
    c.close()


def test_replay_db_is_read_only_and_snapshots_match_shape(tmp_path):
    p = tmp_path / 'cache.db'
    _make_cache(p)
    db = open_readonly_db(p)
    with pytest.raises(sqlite3.OperationalError):
        db._cache_conn.execute("DELETE FROM daily_bars")
    snaps = load_snapshots(db, '2026-10-02', 10)
    assert snaps['AAA'] == {'open': 10.0, 'prev_close': 9.0, 'prev_volume': 900000,
                            'latest_price': 10.0, 'daily_bar_date': '2026-10-02',
                            'close': 9.0, 'volume': 900000}
    assert set(StubAlpaca(snaps).get_snapshots(['AAA', 'ZZZ'])) == {'AAA'}
