"""study_orb_broad: grouped 9:35-volume query == per-pair sums (2026-09-05)."""
import sqlite3

from study_orb_broad import et_935_cutoff_utc_str, rth_volume_by_935_for_pairs


def _db():
    conn = sqlite3.connect(':memory:')
    conn.execute("""CREATE TABLE intraday_bars_1min (
        symbol TEXT, bar_date TEXT, timestamp TEXT, open REAL, high REAL,
        low REAL, close REAL, volume INTEGER)""")
    rows = [
        # EST day (UTC-5): 9:30-9:34 ET = 14:30-14:34 UTC; 9:35 = 14:35
        ('AAA', '2025-03-05', '2025-03-05T14:30:00+00:00', 1, 1, 1, 1, 100),
        ('AAA', '2025-03-05', '2025-03-05T14:34:00+00:00', 1, 1, 1, 1, 200),
        ('AAA', '2025-03-05', '2025-03-05T14:35:00+00:00', 1, 1, 1, 1, 999),   # excluded
        ('AAA', '2025-03-05', '2025-03-05T13:59:00+00:00', 1, 1, 1, 1, 50),    # premarket, included
        # EDT day (UTC-4): 9:35 = 13:35 UTC
        ('AAA', '2025-09-10', '2025-09-10T13:34:00+00:00', 1, 1, 1, 1, 10),
        ('AAA', '2025-09-10', '2025-09-10T13:35:00+00:00', 1, 1, 1, 1, 777),   # excluded
        ('BBB', '2025-09-10', '2025-09-10T13:31:00+00:00', 1, 1, 1, 1, 5),
    ]
    conn.executemany("INSERT INTO intraday_bars_1min VALUES (?,?,?,?,?,?,?,?)", rows)
    return conn


class TestCutoff:
    def test_est_and_edt(self):
        assert et_935_cutoff_utc_str('2025-03-05') == '2025-03-05T14:35'
        assert et_935_cutoff_utc_str('2025-09-10') == '2025-09-10T13:35'


class TestGroupedQuery:
    def test_matches_per_pair_sums_and_zero_for_missing(self):
        conn = _db()
        pairs = [('AAA', '2025-03-05'), ('AAA', '2025-09-10'), ('BBB', '2025-09-10'),
                 ('CCC', '2025-09-10')]
        got = rth_volume_by_935_for_pairs(conn, pairs)
        assert got == {('AAA', '2025-03-05'): 350, ('AAA', '2025-09-10'): 10,
                       ('BBB', '2025-09-10'): 5, ('CCC', '2025-09-10'): 0}
        # identical to the legacy per-pair query
        for s, d in pairs:
            legacy = conn.execute(
                "SELECT COALESCE(SUM(volume),0) FROM intraday_bars_1min "
                "WHERE symbol=? AND bar_date=? AND timestamp < ?",
                (s, d, et_935_cutoff_utc_str(d))).fetchone()[0]
            assert got[(s, d)] == legacy

    def test_empty_pairs(self):
        assert rth_volume_by_935_for_pairs(_db(), []) == {}

    def test_repeat_call_cleans_temp_table(self):
        conn = _db()
        a = rth_volume_by_935_for_pairs(conn, [('AAA', '2025-03-05')])
        b = rth_volume_by_935_for_pairs(conn, [('AAA', '2025-03-05')])
        assert a == b == {('AAA', '2025-03-05'): 350}
