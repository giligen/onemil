"""BT-band gate (trading/ramp_bt_band.py) — Gate-2 item 4 of the scaling plan.

The band is the backtest's own bootstrap distribution of the mean R over n
trades: the only fair bar for a live sample of n.
"""
from __future__ import annotations

import csv

import pytest

from trading import ramp_bt_band as bb


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


class TestReferences:
    def test_orb_reference_follows_the_catalyst_veto(self):
        on = bb.orb_reference(True)
        off = bb.orb_reference(False)
        assert on.path == bb.ORB_REF_VETO_ON and 'veto ON' in on.label
        assert off.path == bb.ORB_REF_VETO_OFF
        assert 'boots Monday' in off.label   # named in the checker output

    def test_bf_reference_is_the_vol_off_book(self):
        ref = bb.bf_reference()
        assert ref.path == bb.BF_REF and 'min_daily_volume 0' in ref.label

    def test_shipped_references_exist_and_parse(self):
        """The honest books must be on disk — a missing one silently kills the gate."""
        for ref, book in ((bb.orb_reference(True), 'orb'),
                          (bb.orb_reference(False), 'orb'),
                          (bb.bf_reference(), 'bf')):
            assert ref.exists, f"missing BT reference {ref.path}"
            r = bb.load_reference_r(ref, book)
            assert len(r) > 30 and all(isinstance(x, float) for x in r)

    def test_missing_reference_is_loud_and_empty(self, tmp_path, caplog):
        ref = bb.Reference(tmp_path / 'nope.csv', 'ghost')
        with caplog.at_level('ERROR'):
            assert bb.load_reference_r(ref, 'orb') == []
        assert 'BT reference missing' in caplog.text

    def test_unparseable_reference_is_loud_and_empty(self, tmp_path, caplog):
        p = tmp_path / 'bad.csv'
        p.write_bytes(b'\xff\xfe\x00broken')
        with caplog.at_level('ERROR'):
            assert bb.load_reference_r(bb.Reference(p, 'bad'), 'bf') == []
        assert 'unreadable' in caplog.text


class TestLoaders:
    def test_orb_r_is_pnl_pct_over_range_size(self, tmp_path):
        p = write_csv(tmp_path / 'orb.csv', [
            {'pnl_pct': '-4.0', 'range_size_pct': '4.0', 'entered': '1'},
            {'pnl_pct': '9.0', 'range_size_pct': '3.0', 'entered': '1'},
        ], ['pnl_pct', 'range_size_pct', 'entered'])
        assert bb.load_orb_bt_r(p) == [-1.0, 3.0]

    def test_orb_skips_no_fill_rows(self, tmp_path):
        p = write_csv(tmp_path / 'orb.csv', [
            {'pnl_pct': '0', 'range_size_pct': '4.0', 'entered': '0'},
            {'pnl_pct': '4.0', 'range_size_pct': '4.0', 'entered': '1'},
        ], ['pnl_pct', 'range_size_pct', 'entered'])
        assert bb.load_orb_bt_r(p) == [1.0]

    def test_orb_skips_unusable_rows_loudly(self, tmp_path, caplog):
        p = write_csv(tmp_path / 'orb.csv', [
            {'pnl_pct': '', 'range_size_pct': '4.0', 'entered': '1'},
            {'pnl_pct': '4.0', 'range_size_pct': '0', 'entered': '1'},
            {'pnl_pct': '4.0', 'range_size_pct': '2.0', 'entered': '1'},
        ], ['pnl_pct', 'range_size_pct', 'entered'])
        with caplog.at_level('WARNING'):
            assert bb.load_orb_bt_r(p) == [2.0]
        assert 'excluded from the BT band' in caplog.text

    def test_non_numeric_fields_are_dropped_not_crashed(self, tmp_path):
        p = write_csv(tmp_path / 'orb.csv', [
            {'pnl_pct': 'n/a', 'range_size_pct': '4.0', 'entered': '1'},
            {'pnl_pct': '4.0', 'range_size_pct': '2.0', 'entered': '1'},
        ], ['pnl_pct', 'range_size_pct', 'entered'])
        assert bb.load_orb_bt_r(p) == [2.0]
        b = write_csv(tmp_path / 'bf.csv', [{'pnl': 'oops'}, {'pnl': '1000'}],
                      ['pnl'])
        assert bb.load_bf_bt_r(b) == [0.5]

    def test_bf_r_is_pnl_over_the_2k_normalization(self, tmp_path):
        p = write_csv(tmp_path / 'bf.csv',
                      [{'pnl': '2000'}, {'pnl': '-1000'}, {'pnl': ''}],
                      ['pnl'])
        assert bb.load_bf_bt_r(p) == [1.0, -0.5]
        assert bb.load_bf_bt_r(p, risk_usd=1000.0) == [2.0, -1.0]


class TestBand:
    dist = [-1.0] * 7 + [2.0, 8.0, 16.0]

    def test_band_is_ordered_and_deterministic(self):
        a = bb.bootstrap_band(self.dist, 8)
        b = bb.bootstrap_band(self.dist, 8)
        assert (a.p5, a.p10, a.p90) == (b.p5, b.p10, b.p90)
        assert a.p5 <= a.p10 <= a.p90
        assert a.n == 8 and a.draws == bb.DRAWS and a.n_ref == 10

    def test_band_tightens_as_n_grows(self):
        small = bb.bootstrap_band(self.dist, 5)
        big = bb.bootstrap_band(self.dist, 200)
        assert (big.p90 - big.p5) < (small.p90 - small.p5)

    def test_no_reference_or_no_trades_means_no_band(self):
        assert bb.bootstrap_band([], 8) is None
        assert bb.bootstrap_band(self.dist, 0) is None

    def test_classify_boundaries(self):
        band = bb.Band(p5=-0.5, p10=-0.2, p90=1.5, n=8, draws=10, n_ref=10)
        assert bb.classify(-0.6, band) == bb.BELOW_P5
        assert bb.classify(-0.5, band) == bb.BELOW_P10     # p5 itself is not below p5
        assert bb.classify(-0.3, band) == bb.BELOW_P10
        assert bb.classify(-0.2, band) == bb.IN_BAND
        assert bb.classify(1.5, band) == bb.IN_BAND
        assert bb.classify(1.6, band) == bb.ABOVE_P90
        assert bb.classify(None, band) == bb.NO_DATA
        assert bb.classify(0.3, None) == bb.NO_DATA

    def test_band_line_states_the_rule_and_the_reference(self):
        band = bb.bootstrap_band(self.dist, 8)
        ref = bb.bf_reference()
        line = bb.band_line(bb.IN_BAND, 0.4, band, ref)
        assert 'IN-BAND' in line and 'n=8' in line and 'VOL_OFF' in line
        assert bb.BAND_RULE in line
        empty = bb.band_line(bb.NO_DATA, None, None, ref)
        assert 'NO-DATA' in empty and bb.BAND_RULE in empty
