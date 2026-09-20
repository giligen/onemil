"""Pooled ramp statistic (trading/ramp_pool.py) — frames10 F33.

The pooled z is a PRECISION gate: it can hold a stage that its own book's gate would have advanced
on noise, and it can never do anything else. The invariant tests in `TestTheInvariant` are the
reason this module is allowed to exist at all — the dry stream earns nothing, so it must never be
able to move a dollar of risk.
"""
from __future__ import annotations

import csv
import logging
import math
import sqlite3

import pytest

from trading import ramp_bt_band as bb
from trading import ramp_pool as rp


def mk(book, day, r):
    return rp.Trade(book, day, r)


def trades(book, rs, day0=1):
    """One trade per session, so the day-clustered SE has clusters to work with."""
    return [mk(book, f"2026-09-{day0 + i:02d}", r) for i, r in enumerate(rs)]


@pytest.fixture
def sds():
    return {rp.BOOK_ORB: 2.0, rp.BOOK_BF: 4.0, rp.BOOK_HOD_DRY: 1.0}


# --------------------------------------------------------------------------- the statistic
class TestPooledZ:
    def test_standardises_each_book_by_its_own_sd(self, sds):
        s = rp.pooled_z([mk(rp.BOOK_ORB, '2026-09-01', 2.0),
                         mk(rp.BOOK_BF, '2026-09-02', 4.0)], sds)
        assert s.n == 2
        assert s.z_bar == pytest.approx(1.0)          # 2/2 and 4/4 both standardise to 1.0

    def test_per_book_counts_and_dry_count(self, sds):
        s = rp.pooled_z(trades(rp.BOOK_ORB, [1, -1]) + trades(rp.BOOK_HOD_DRY, [0.5], day0=5), sds)
        assert s.per_book_n == {rp.BOOK_ORB: 2, rp.BOOK_HOD_DRY: 1}
        assert s.dry_n == 1 and s.live_n == 2

    def test_empty_pool_is_none_not_zero(self, sds):
        s = rp.pooled_z([], sds)
        assert s.z_bar is None and s.se is None and s.n == 0

    def test_missing_sd_excludes_the_book_loudly(self, caplog):
        with caplog.at_level(logging.WARNING):
            s = rp.pooled_z(trades(rp.BOOK_ORB, [1.0, 2.0]), {rp.BOOK_BF: 1.0})
        assert s.n == 0 and s.excluded and 'no usable BT SD' in s.excluded[0]
        assert 'excluded from the pool' in caplog.text

    def test_non_finite_r_is_dropped_with_an_error(self, sds, caplog):
        with caplog.at_level(logging.ERROR):
            s = rp.pooled_z([mk(rp.BOOK_ORB, '2026-09-01', float('nan')),
                             mk(rp.BOOK_ORB, '2026-09-02', 2.0)], sds)
        assert s.n == 1 and 'uncomputable R' in caplog.text

    def test_zero_or_negative_sd_is_refused(self, caplog):
        with caplog.at_level(logging.WARNING):
            s = rp.pooled_z(trades(rp.BOOK_ORB, [1.0]), {rp.BOOK_ORB: 0.0})
        assert s.n == 0

    def test_clustered_se_is_wider_than_the_naive_one_when_a_day_repeats(self, sds):
        """Same numbers, one session vs many: the day-clustered SE must not shrink with n."""
        same_day = [mk(rp.BOOK_ORB, '2026-09-01', v) for v in (2, 2, 2, -2, -2, -2)]
        spread = trades(rp.BOOK_ORB, [2, 2, 2, -2, -2, -2])
        a, b = rp.pooled_z(same_day, sds), rp.pooled_z(spread, sds)
        assert a.se is None or b.se is None or a.se >= b.se

    def test_t_and_n_days(self, sds):
        s = rp.pooled_z(trades(rp.BOOK_ORB, [2.0, 2.0, 2.0, 2.2]), sds)
        assert s.n_days == 4 and s.t is not None and s.t > 0


class TestPooledBand:
    def test_band_brackets_the_reference_mean(self, sds):
        rr = {rp.BOOK_ORB: [1.0, -1.0, 2.0, -2.0] * 25}
        band = rp.pooled_band(rr, sds, {rp.BOOK_ORB: 20})
        assert band is not None and band.p5 < band.p10 < band.p90
        assert band.n == 20 and band.n_ref == 100

    def test_dry_stream_contributes_to_the_band(self, sds):
        rr = {rp.BOOK_ORB: [1.0, -1.0] * 20, rp.BOOK_HOD_DRY: [0.5, -0.5] * 20}
        one = rp.pooled_band(rr, sds, {rp.BOOK_ORB: 10})
        two = rp.pooled_band(rr, sds, {rp.BOOK_ORB: 10, rp.BOOK_HOD_DRY: 40})
        assert two.n == 50 and one.n == 10
        # more pooled draws -> a tighter band. That is the whole point of the dry stream.
        assert (two.p90 - two.p5) < (one.p90 - one.p5)

    def test_missing_reference_is_logged_and_skipped(self, sds, caplog):
        with caplog.at_level(logging.WARNING):
            band = rp.pooled_band({rp.BOOK_ORB: [1.0, -1.0] * 10}, sds,
                                  {rp.BOOK_ORB: 5, rp.BOOK_BF: 5})
        assert band is not None and band.n == 5
        assert 'no usable BT reference' in caplog.text

    def test_no_reference_at_all_is_no_data(self, sds, caplog):
        with caplog.at_level(logging.ERROR):
            assert rp.pooled_band({}, sds, {rp.BOOK_ORB: 5}) is None
        assert 'no book contributed' in caplog.text

    def test_classify_needs_min_pooled_n(self, sds):
        rr = {rp.BOOK_ORB: [1.0, -1.0] * 20}
        band = rp.pooled_band(rr, sds, {rp.BOOK_ORB: 5})
        s = rp.pooled_z(trades(rp.BOOK_ORB, [-5.0] * 5), sds)
        assert rp.classify_pooled(s, band) == bb.NO_DATA

    def test_classify_reads_below_p5_on_a_bad_sample(self, sds):
        rr = {rp.BOOK_ORB: [1.0, -1.0] * 50}
        n = 20
        band = rp.pooled_band(rr, sds, {rp.BOOK_ORB: n})
        s = rp.pooled_z(trades(rp.BOOK_ORB, [-10.0] * n), sds)
        assert rp.classify_pooled(s, band) == bb.BELOW_P5

    def test_classify_reads_above_p90_on_a_hot_sample(self, sds):
        rr = {rp.BOOK_ORB: [1.0, -1.0] * 50}
        n = 20
        band = rp.pooled_band(rr, sds, {rp.BOOK_ORB: n})
        s = rp.pooled_z(trades(rp.BOOK_ORB, [10.0] * n), sds)
        assert rp.classify_pooled(s, band) == bb.ABOVE_P90


# --------------------------------------------------------------------------- THE INVARIANT
class TestTheInvariant:
    """A pooled gate can never turn a negative-P&L book into ADVANCE. Non-negotiable."""

    @pytest.mark.parametrize('verdict', ['HOLD', 'DEMOTE', 'PAUSE'])
    @pytest.mark.parametrize('status', [bb.BELOW_P5, bb.BELOW_P10, bb.IN_BAND,
                                        bb.ABOVE_P90, bb.NO_DATA])
    def test_never_upgrades_any_verdict(self, verdict, status):
        out, why = rp.apply_pooled_gate(verdict, status)
        assert out == verdict and why is None
        assert out != 'ADVANCE'

    @pytest.mark.parametrize('status,expected', [
        (bb.BELOW_P5, 'HOLD'), (bb.BELOW_P10, 'HOLD'),
        (bb.IN_BAND, 'ADVANCE'), (bb.ABOVE_P90, 'ADVANCE'), (bb.NO_DATA, 'ADVANCE')])
    def test_only_ever_downgrades_an_advance(self, status, expected):
        out, why = rp.apply_pooled_gate('ADVANCE', status)
        assert out == expected
        assert (why is not None) == (expected == 'HOLD')

    def test_a_losing_book_with_a_hot_pool_still_does_not_advance(self, sds):
        """The above-water rule: a hot DRY sibling cannot lift a book that is under water."""
        pool = trades(rp.BOOK_HOD_DRY, [5.0] * 30) + trades(rp.BOOK_ORB, [-3.0] * 5, day0=1)
        s = rp.pooled_z(pool, sds)
        band = rp.pooled_band({rp.BOOK_ORB: [1.0, -1.0] * 50,
                               rp.BOOK_HOD_DRY: [1.0, -1.0] * 50}, sds, s.per_book_n)
        status = rp.classify_pooled(s, band)
        assert status == bb.ABOVE_P90           # the pool looks great...
        # ...and the book's own gate (P&L <= 0) said HOLD, so the answer is still HOLD.
        assert rp.apply_pooled_gate('HOLD', status)[0] == 'HOLD'

    def test_dry_trades_are_never_in_a_pnl_clause(self, sds):
        s = rp.pooled_z(trades(rp.BOOK_HOD_DRY, [1.0] * 12), sds)
        assert s.live_n == 0 and s.dry_n == 12
        # a pool made only of paper can never advance anything
        assert rp.apply_pooled_gate('HOLD', bb.ABOVE_P90)[0] == 'HOLD'

    def test_pooled_demote_needs_n_and_below_p5(self, sds):
        few = rp.pooled_z(trades(rp.BOOK_ORB, [-1.0] * 12), sds)
        many = rp.pooled_z(trades(rp.BOOK_ORB, [-1.0] * 40), sds)
        assert rp.pooled_demote(few, bb.BELOW_P5) is False
        assert rp.pooled_demote(many, bb.BELOW_P5) is True
        assert rp.pooled_demote(many, bb.BELOW_P10) is False


# --------------------------------------------------------------------------- the line
class TestLine:
    def test_no_data_line_says_it_blocks_nothing(self, sds):
        s = rp.pooled_z(trades(rp.BOOK_ORB, [1.0]), sds)
        line = rp.pooled_line(s, None, bb.NO_DATA)
        assert 'NO-DATA' in line and 'blocks nothing' in line

    def test_line_names_the_dry_share_and_says_advisory(self, sds):
        pool = trades(rp.BOOK_ORB, [1.0] * 6) + trades(rp.BOOK_HOD_DRY, [0.5] * 8, day0=10)
        s = rp.pooled_z(pool, sds)
        band = rp.pooled_band({rp.BOOK_ORB: [1.0, -1.0] * 30,
                               rp.BOOK_HOD_DRY: [1.0, -1.0] * 30}, sds, s.per_book_n)
        line = rp.pooled_line(s, band, rp.classify_pooled(s, band))
        assert 'dry 8 of 14' in line and 'ADVISORY' in line and 'POOLED z' in line


# --------------------------------------------------------------------------- the loaders
class TestLoaders:
    def _db(self, tmp_path, rows):
        p = tmp_path / 'trades.db'
        c = sqlite3.connect(p)
        c.execute('CREATE TABLE trades (trade_date TEXT, strategy TEXT, pnl REAL, '
                  'total_risk REAL)')
        c.executemany('INSERT INTO trades VALUES (?,?,?,?)', rows)
        c.commit()
        c.close()
        return p

    def test_orb_r_is_pnl_over_total_risk(self, tmp_path):
        db = self._db(tmp_path, [('2026-09-01', 'orb', 200.0, 100.0),
                                 ('2026-09-02', 'orb', -50.0, 100.0)])
        t = rp.load_live_trades(rp.BOOK_ORB, '2026-09-01', db)
        assert [x.r for x in t] == [2.0, -0.5]

    def test_bf_r_is_pnl_over_the_stage_base(self, tmp_path):
        db = self._db(tmp_path, [('2026-09-01', 'bull_flag', 300.0, 0.0)])
        t = rp.load_live_trades(rp.BOOK_BF, '2026-09-01', db, risk_base=150.0)
        assert t[0].r == pytest.approx(2.0)

    def test_a_trade_without_a_usable_1r_is_dropped_with_an_error(self, tmp_path, caplog):
        db = self._db(tmp_path, [('2026-09-01', 'orb', 200.0, 0.0),
                                 ('2026-09-02', 'orb', 100.0, 100.0)])
        with caplog.at_level(logging.ERROR):
            t = rp.load_live_trades(rp.BOOK_ORB, '2026-09-01', db)
        assert len(t) == 1 and 'never counted as zero' in caplog.text

    def test_since_filters_and_unknown_book_errors(self, tmp_path, caplog):
        db = self._db(tmp_path, [('2026-08-01', 'orb', 100.0, 100.0),
                                 ('2026-09-02', 'orb', 100.0, 100.0)])
        assert len(rp.load_live_trades(rp.BOOK_ORB, '2026-09-01', db)) == 1
        with caplog.at_level(logging.ERROR):
            assert rp.load_live_trades('nope', '2026-09-01', db) == []
        assert 'unknown book' in caplog.text

    def test_missing_db_is_an_error_not_a_crash(self, tmp_path, caplog):
        with caplog.at_level(logging.ERROR):
            assert rp.load_live_trades(rp.BOOK_ORB, '2026-09-01', tmp_path / 'nope.db') == []
        assert 'contributes nothing' in caplog.text

    def test_dry_pool_round_trip(self, tmp_path):
        p = tmp_path / 'dry.csv'
        assert rp.append_dry_trades('2026-09-15', [('AAPL', 1.5), ('MSFT', -1.0)], p) == 2
        assert rp.append_dry_trades('2026-09-16', [('NVDA', 0.25)], p) == 1
        t = rp.load_dry_trades(p)
        assert [x.book for x in t] == [rp.BOOK_HOD_DRY] * 3
        assert [x.r for x in t] == [1.5, -1.0, 0.25]
        assert [x.day for x in rp.load_dry_trades(p, since='2026-09-16')] == ['2026-09-16']

    def test_missing_dry_file_warns_and_returns_empty(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            assert rp.load_dry_trades(tmp_path / 'nope.csv') == []
        assert 'contributes nothing' in caplog.text

    def test_bad_dry_rows_are_dropped_with_an_error(self, tmp_path, caplog):
        p = tmp_path / 'dry.csv'
        with open(p, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(rp.DRY_POOL_HEADER)
            w.writerow(['2026-09-15', 'AAA', '600', 'not-a-number'])
            w.writerow(['2026-09-15', 'BBB', '601', '1.0'])
        with caplog.at_level(logging.ERROR):
            t = rp.load_dry_trades(p)
        assert len(t) == 1 and 'without a usable R' in caplog.text

    def test_append_of_nothing_writes_nothing(self, tmp_path):
        p = tmp_path / 'dry.csv'
        assert rp.append_dry_trades('2026-09-15', [], p) == 0
        assert not p.exists()


# ------------------------------------------------------------------ frames11 F36: the producer
class TestDryPoolIdempotence:
    """The EOD check is a reporting tool the owner re-runs freely; the cron may fire twice.

    A second run of the same session must append NOTHING, or the pooled n (and therefore the band
    width the ramp gate reads) inflates on re-reads alone.
    """

    SESSION = [('AAPL', 1.5, 611), ('MSFT', -1.0, 640), ('NVDA', 0.25, 700)]

    def test_a_rerun_of_the_same_session_appends_nothing(self, tmp_path):
        p = tmp_path / 'dry.csv'
        assert rp.append_dry_trades('2026-09-21', self.SESSION, p) == 3
        before = p.read_text()
        for _ in range(3):
            assert rp.append_dry_trades('2026-09-21', self.SESSION, p) == 0
        assert p.read_text() == before
        assert len(rp.load_dry_trades(p)) == 3

    def test_the_key_is_day_symbol_and_entry_minute(self, tmp_path):
        p = tmp_path / 'dry.csv'
        assert rp.append_dry_trades('2026-09-21', [('AAPL', 1.5, 611)], p) == 1
        # same symbol, same day, DIFFERENT entry minute = a different trade (a re-break)
        assert rp.append_dry_trades('2026-09-21', [('AAPL', -0.9, 705)], p) == 1
        # same symbol and minute on another day = a different trade
        assert rp.append_dry_trades('2026-09-22', [('AAPL', 0.4, 611)], p) == 1
        # ...and the exact key again is a no-op
        assert rp.append_dry_trades('2026-09-22', [('AAPL', 0.4, 611)], p) == 0
        assert rp.dry_pool_keys(p) == {('2026-09-21', 'AAPL', '611'),
                                       ('2026-09-21', 'AAPL', '705'),
                                       ('2026-09-22', 'AAPL', '611')}

    def test_duplicates_inside_one_call_are_collapsed(self, tmp_path):
        p = tmp_path / 'dry.csv'
        assert rp.append_dry_trades('2026-09-21', [('AAPL', 1.5, 611), ('AAPL', 1.5, 611)], p) == 1

    def test_the_pooled_band_reads_the_appended_rows(self, tmp_path):
        """The whole point: what the EOD check writes is what the ramp band widens on."""
        p = tmp_path / 'dry.csv'
        rows = [(f'S{i}', 0.5 if i % 2 else -0.5, 600 + i) for i in range(14)]
        assert rp.append_dry_trades('2026-09-21', rows, p) == 14
        trades = rp.load_dry_trades(p)
        assert len(trades) == 14 and all(t.book == rp.BOOK_HOD_DRY for t in trades)
        sds, rr = rp.reference_sds_and_r([rp.BOOK_HOD_DRY])
        stat = rp.pooled_z(trades, sds)
        band = rp.pooled_band(rr, sds, stat.per_book_n)
        assert stat.n == 14 and stat.dry_n == 14 and stat.live_n == 0
        assert band is not None and band.n == 14
        assert rp.classify_pooled(stat, band) != bb.NO_DATA
        # and a re-run of the session does not move the reading
        assert rp.append_dry_trades('2026-09-21', rows, p) == 0
        assert rp.pooled_z(rp.load_dry_trades(p), sds).n == 14

    def test_the_legacy_three_column_file_still_appends_and_reads(self, tmp_path, caplog):
        p = tmp_path / 'dry.csv'
        with open(p, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(rp.DRY_POOL_HEADER_LEGACY)
            w.writerow(['2026-09-15', 'AAA', '1.0'])
        with caplog.at_level(logging.WARNING):
            assert rp.append_dry_trades('2026-09-15', [('AAA', 1.0), ('BBB', -0.5)], p) == 1
        assert 'legacy 3-column schema' in caplog.text
        assert [t.r for t in rp.load_dry_trades(p)] == [1.0, -0.5]

    def test_two_tuples_without_a_minute_still_work(self, tmp_path):
        p = tmp_path / 'dry.csv'
        assert rp.append_dry_trades('2026-09-21', [('AAPL', 1.5)], p) == 1
        assert rp.append_dry_trades('2026-09-21', [('AAPL', 1.5)], p) == 0

    def test_a_non_finite_r_is_never_written(self, tmp_path):
        p = tmp_path / 'dry.csv'
        assert rp.append_dry_trades('2026-09-21', [('AAPL', float('nan'), 611),
                                                   ('MSFT', float('inf'), 612)], p) == 0
        assert not p.exists()

    def test_keys_of_a_missing_file_are_empty(self, tmp_path):
        assert rp.dry_pool_keys(tmp_path / 'nope.csv') == set()


class TestEodCheckProducer:
    """`scripts/hod_break_eod_check.append_to_pool` — the one call that makes the pool self-feeding."""

    @staticmethod
    def _mod():
        import importlib.util
        import sys as _sys
        from pathlib import Path as _P
        root = _P(__file__).resolve().parent.parent
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location(
            'hod_break_eod_check_f36', root / 'scripts' / 'hod_break_eod_check.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # run_book output for the DRY-RUN book: (day_key, entry_m, exit_m, symbol, rr, usd)
    TAKEN = [(0, 611, 650, 'AAPL', 1.5, 150.0), (0, 640, 700, 'MSFT', -1.0, -100.0)]

    def test_dry_mode_appends_once_and_the_rerun_is_a_noop(self, tmp_path, monkeypatch):
        mod = self._mod()
        p = tmp_path / 'dry.csv'
        monkeypatch.setattr(rp, 'DRY_POOL_PATH', p)
        first = mod.append_to_pool('2026-09-21', self.TAKEN, dry_run=True, is_r2g=False)
        assert 'appended 2 of 2' in first
        again = mod.append_to_pool('2026-09-21', self.TAKEN, dry_run=True, is_r2g=False)
        assert 'appended 0 of 2' in again
        assert [t.r for t in rp.load_dry_trades(p)] == [1.5, -1.0]
        assert rp.dry_pool_keys(p) == {('2026-09-21', 'AAPL', '611'),
                                       ('2026-09-21', 'MSFT', '640')}

    def test_live_mode_writes_nothing(self, tmp_path, monkeypatch):
        mod = self._mod()
        p = tmp_path / 'dry.csv'
        monkeypatch.setattr(rp, 'DRY_POOL_PATH', p)
        assert 'nothing appended' in mod.append_to_pool('2026-09-21', self.TAKEN,
                                                        dry_run=False, is_r2g=False)
        assert not p.exists()

    def test_red_to_green_is_not_a_pooled_book(self, tmp_path, monkeypatch):
        mod = self._mod()
        p = tmp_path / 'dry.csv'
        monkeypatch.setattr(rp, 'DRY_POOL_PATH', p)
        assert 'not a pooled book' in mod.append_to_pool('2026-09-21', self.TAKEN,
                                                         dry_run=True, is_r2g=True)
        assert not p.exists()

    def test_a_broken_pool_write_never_breaks_the_eod_check(self, monkeypatch):
        mod = self._mod()

        def boom(*a, **k):
            raise OSError('disk full')

        monkeypatch.setattr(rp, 'append_dry_trades', boom)
        assert 'append FAILED' in mod.append_to_pool('2026-09-21', self.TAKEN,
                                                     dry_run=True, is_r2g=False)


# --------------------------------------------------------------------------- the real references
class TestRealReferences:
    """The SD and the band must come from the SAME distribution — one source, always."""

    def test_every_pool_book_has_a_readable_reference_and_sd(self):
        sds, rr = rp.reference_sds_and_r([rp.BOOK_ORB, rp.BOOK_BF, rp.BOOK_HOD_DRY])
        for b in (rp.BOOK_ORB, rp.BOOK_BF, rp.BOOK_HOD_DRY):
            assert b in sds and b in rr and len(rr[b]) > 20
            assert math.isfinite(sds[b]) and sds[b] > 0

    def test_sd_matches_the_distribution_the_band_is_built_from(self):
        sds, rr = rp.reference_sds_and_r([rp.BOOK_ORB])
        assert sds[rp.BOOK_ORB] == pytest.approx(bb.sd_of(rr[rp.BOOK_ORB]))

    def test_hod_reference_is_the_b2_book_train_plus_val(self):
        v = bb.load_hod_bt_r()
        assert len(v) == 2328                       # the B2 reference: 1,622 TRAIN + 706 VAL
        assert bb.sd_of(v) == pytest.approx(1.260, abs=0.005)

    def test_fallback_sds_are_close_to_the_live_ones(self):
        sds, _ = rp.reference_sds_and_r([rp.BOOK_ORB, rp.BOOK_BF, rp.BOOK_HOD_DRY])
        for b, v in sds.items():
            assert bb.BOOK_SD_FALLBACK[b] == pytest.approx(v, rel=0.001)

    def test_sd_of_needs_two_values(self):
        assert bb.sd_of([]) is None and bb.sd_of([1.0]) is None

    def test_unknown_pool_book_errors(self, caplog):
        with caplog.at_level(logging.ERROR):
            assert bb.pool_reference_r('nope') == []
        assert 'unknown book' in caplog.text

    def test_reading_runs_end_to_end_read_only(self, tmp_path):
        stat, band, status, line = rp.reading(
            {rp.BOOK_ORB: '2026-09-01', rp.BOOK_BF: '2026-09-01'},
            bf_risk_base=150.0, dry_path=tmp_path / 'none.csv')
        assert status in (bb.NO_DATA, bb.BELOW_P5, bb.BELOW_P10, bb.IN_BAND, bb.ABOVE_P90)
        assert 'POOLED z' in line and stat.n >= 0


# --------------------------------------------------------------------------- the advisory line
class TestAdvisoryLine:
    def test_returns_a_line_from_the_real_repo_state(self, tmp_path):
        line = rp.advisory_line(dry_path=tmp_path / 'none.csv')
        assert line.startswith('  POOLED z')

    def test_dry_stream_reaches_the_pool(self, tmp_path):
        p = tmp_path / 'dry.csv'
        rp.append_dry_trades('2026-09-15', [(f'S{i}', 0.5) for i in range(12)], p)
        line = rp.advisory_line(dry_path=p)
        assert 'hod_dry 12' in line or 'dry 12' in line

    def test_a_broken_resolve_degrades_to_no_data_and_never_raises(self, monkeypatch, caplog):
        from trading import ramp_stage
        monkeypatch.setattr(ramp_stage, 'resolve',
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
        with caplog.at_level(logging.ERROR):
            line = rp.advisory_line()
        assert 'NO-DATA' in line and 'blocks nothing' in line
        assert 'pooled reading unavailable' in caplog.text

    def test_a_corrupt_trades_db_is_an_error_not_a_crash(self, tmp_path, caplog):
        bad = tmp_path / 'trades.db'
        bad.write_text('not a database at all')
        with caplog.at_level(logging.ERROR):
            assert rp.load_live_trades(rp.BOOK_ORB, '2026-01-01', bad) == []
        assert 'unreadable' in caplog.text

    def test_an_unreadable_dry_path_is_an_error_not_a_crash(self, tmp_path, caplog):
        d = tmp_path / 'adir'
        d.mkdir()
        with caplog.at_level(logging.ERROR):
            assert rp.load_dry_trades(d) == []
        assert 'unreadable' in caplog.text

    def test_zero_count_books_are_skipped_in_the_band(self, sds):
        band = rp.pooled_band({rp.BOOK_ORB: [1.0, -1.0] * 20}, sds,
                              {rp.BOOK_ORB: 10, rp.BOOK_BF: 0})
        assert band is not None and band.n == 10

    def test_t_is_none_without_an_se(self, sds):
        s = rp.pooled_z([mk(rp.BOOK_ORB, '2026-09-01', 1.0)], sds)
        assert s.t is None

    def test_reference_without_a_usable_sd_is_excluded(self, monkeypatch, caplog):
        monkeypatch.setattr(bb, 'pool_reference_r', lambda b, **k: [1.0])
        with caplog.at_level(logging.WARNING):
            sds, rr = rp.reference_sds_and_r([rp.BOOK_ORB])
        assert sds == {} and rr == {} and 'no usable SD' in caplog.text
