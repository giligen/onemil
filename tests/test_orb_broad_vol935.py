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


# ---- 2026-11-11: the 15K volume proxy is replaced by the bar-count (range-computability) condition ----
import study_orb_broad
from study_orb_broad import opening_range_bar_counts_for_pairs, load_broad_universe


def _bars(sym, day, hhmm_utc_list, vol=1):
    return [(sym, day, f'{day}T{t}:00+00:00', 1, 1, 1, 1, vol) for t in hhmm_utc_list]


class TestBarCountCondition:
    def test_counts_est_edt_and_ignores_premarket_and_0935(self):
        conn = sqlite3.connect(':memory:')
        conn.execute("""CREATE TABLE intraday_bars_1min (symbol TEXT, bar_date TEXT, timestamp TEXT, open REAL,
            high REAL, low REAL, close REAL, volume INTEGER)""")
        rows = (_bars('THIN', '2025-03-05', ['14:30', '14:31', '14:32', '14:33', '14:34'], vol=1)       # EST, 5 bars
                + _bars('HEAVY4', '2025-03-05', ['14:30', '14:31', '14:32', '14:33'], vol=1_000_000)   # 4 bars
                + _bars('PRE', '2025-03-05', ['13:55', '14:00', '14:29', '14:35'], vol=9)               # none in window
                + _bars('EDT', '2025-09-10', ['13:30', '13:31', '13:32', '13:33', '13:34', '13:35']))   # EDT, 5 in window
        conn.executemany("INSERT INTO intraday_bars_1min VALUES (?,?,?,?,?,?,?,?)", rows)
        got = opening_range_bar_counts_for_pairs(
            conn, [('THIN', '2025-03-05'), ('HEAVY4', '2025-03-05'), ('PRE', '2025-03-05'),
                   ('EDT', '2025-09-10'), ('NONE', '2025-09-10')], 5)
        assert got == {('THIN', '2025-03-05'): 5, ('HEAVY4', '2025-03-05'): 4, ('PRE', '2025-03-05'): 0,
                       ('EDT', '2025-09-10'): 5, ('NONE', '2025-09-10'): 0}
        assert opening_range_bar_counts_for_pairs(conn, [], 5) == {}

    def test_floor_constant_is_gone(self):
        assert not hasattr(study_orb_broad, 'MIN_935_VOL_FLOOR')


def _full_db(path):
    """Temp cache.db with daily_bars (gap-ups) + intraday bars for the real load_broad_universe."""
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE daily_bars (symbol TEXT, bar_date TEXT, open REAL, high REAL, low REAL,
        close REAL, volume INTEGER)""")
    conn.execute("""CREATE TABLE intraday_bars_1min (symbol TEXT, bar_date TEXT, timestamp TEXT, open REAL,
        high REAL, low REAL, close REAL, volume INTEGER)""")
    days = [('2025-03-04', '2025-03-05'), ('2025-09-09', '2025-09-10')]   # EST day, EDT day
    for sym in ('THIN', 'HEAVY4'):
        for prev, day in days:
            conn.execute("INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?)", (sym, prev, 10, 10, 10, 10, 2_000_000))
            conn.execute("INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?)", (sym, day, 11, 12, 11, 12, 2_000_000))
    est = ['14:30', '14:31', '14:32', '14:33', '14:34']
    edt = ['13:30', '13:31', '13:32', '13:33', '13:34']
    rows = (_bars('THIN', '2025-03-05', est, 1) + _bars('THIN', '2025-09-10', edt, 1)
            + _bars('HEAVY4', '2025-03-05', est[:4], 900_000) + _bars('HEAVY4', '2025-09-10', edt[:4], 900_000)
            + _bars('THIN', '2025-03-05', ['13:00'], 5))   # premarket bar: must not count
    conn.executemany("INSERT INTO intraday_bars_1min VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


class TestLoadBroadUniverseIntegration:
    def test_thin_five_bar_admitted_heavy_four_bar_dropped_with_reason(self, tmp_path, monkeypatch):
        db = str(tmp_path / 'cache.db')
        _full_db(db)
        monkeypatch.setattr(study_orb_broad, 'DATE_END', '2025-12-31')
        monkeypatch.delenv('ORB_RANGE_MINUTES', raising=False)
        sink = []
        uni = load_broad_universe(db_path=db, date_start='2025-01-01', drop_sink=sink)
        assert uni == {'2025-03-05': ['THIN'], '2025-09-10': ['THIN']}
        assert sorted(sink) == [('HEAVY4', '2025-03-05', 'range_not_computable'),
                                ('HEAVY4', '2025-09-10', 'range_not_computable')]
