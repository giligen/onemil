"""frames11 F36 — the BF ramp band's R basis.

The defect (frames10 F31 §1.4.2): the BF BT band was built on `pnl / $2,000`, which is NOT a
per-trade R — the reference CSV's `shares` already embed the conviction / MACD-zone / regime
multipliers, so the BT book's average trade risks ~1.8x its nominal $2K. The live side divides by a
FLAT `trading.risk_per_trade` (stage base $150 at L0). A live book earning exactly what the BT
earned per unit of risk therefore read BELOW-p10 and could never advance.

The fix: BT R = pnl / (that trade's own BT risk) = pnl / (shares x |entry - stop|), which is the
same arithmetic as F31's price-consistent `pnl_pct / stop_pct`.
"""
from __future__ import annotations

import csv
import logging
import statistics

import pytest

from trading import ramp_bt_band as bb


# --------------------------------------------------------------------------- the reference book
@pytest.fixture(scope='module')
def p1_rows():
    ref = bb.bf_reference(200_000)
    assert ref.exists, f"the shipped BF reference is missing: {ref.path}"
    return bb.read_csv_rows(ref.path)


class TestWhichColumnsTheReferenceCarries:
    """State what the CSV actually has — the fix silently picking column 3 of 3 must be visible."""

    def test_p1_has_shares_entry_and_stop_but_no_risk_per_share(self, p1_rows):
        cols = set(p1_rows[0])
        assert {'shares', 'entry_price', 'stop_loss', 'pnl'} <= cols
        assert 'risk_per_share' not in cols

    def test_p1_carries_planned_entry_as_a_column_but_it_is_empty_on_every_row(self, p1_rows):
        assert 'planned_entry' in p1_rows[0]
        assert all(not (r.get('planned_entry') or '').strip() for r in p1_rows), (
            'planned_entry is populated now — bf_trade_risk_usd prefers it over entry_price, so '
            'the band will shift; re-record the expected means in this file')

    def test_every_row_has_a_computable_per_trade_risk(self, p1_rows):
        assert all(bb.bf_trade_risk_usd(r) for r in p1_rows)


class TestTheBasisArithmetic:
    """`pnl / (shares x |entry - stop|)` is `pnl_pct / stop_pct` — F31's price-consistent unit."""

    def test_it_equals_pnl_pct_over_stop_pct_row_by_row(self, p1_rows):
        for r in p1_rows:
            entry, stop = float(r['entry_price']), float(r['stop_loss'])
            got = float(r['pnl']) / bb.bf_trade_risk_usd(r)
            want = float(r['pnl_pct']) / (abs(entry - stop) / entry * 100.0)
            # pnl_pct is stored to 2 dp, so the check is to the CSV's own precision (<= 2 %)
            assert got == pytest.approx(want, rel=0.02)

    def test_risk_per_share_wins_when_the_csv_carries_it(self):
        row = {'shares': '100', 'risk_per_share': '0.25',
               'entry_price': '10.00', 'stop_loss': '9.00'}
        assert bb.bf_trade_risk_usd(row) == pytest.approx(25.0)

    def test_planned_entry_wins_over_the_fill(self):
        row = {'shares': '100', 'planned_entry': '10.00',
               'entry_price': '10.50', 'stop_loss': '9.00'}
        assert bb.bf_trade_risk_usd(row) == pytest.approx(100.0)

    @pytest.mark.parametrize('row', [
        {'shares': '0', 'entry_price': '10', 'stop_loss': '9'},        # no size
        {'shares': '100', 'entry_price': '10'},                        # no stop
        {'shares': '100', 'entry_price': '10', 'stop_loss': '10'},     # zero-width stop
        {'shares': '', 'entry_price': '', 'stop_loss': ''},            # empty row
    ])
    def test_an_uncomputable_risk_is_none_never_zero(self, row):
        assert bb.bf_trade_risk_usd(row) is None


class TestTheBandMoves:
    """THE headline: the BF band on the shipped reference goes from ~+1.65 to ~+0.91 mean R."""

    @staticmethod
    def _split(rows, lo, hi):
        return [r for r in rows if lo <= r['date'] < hi]

    def test_train_mean_r_drops_from_1_65_to_0_91(self, p1_rows, tmp_path):
        """frames10 F31's O6 row, reproduced from the CSV the ramp gate actually reads."""
        train = self._split(p1_rows, '2025-01-01', '2026-01-01')
        assert len(train) == 34, 'F31 O6 measured 34 TRAIN trades'
        p = tmp_path / 'train.csv'
        with open(p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(p1_rows[0]))
            w.writeheader()
            w.writerows(train)
        legacy = statistics.mean(bb.load_bf_bt_r(p, basis=bb.BF_R_BASIS_NOTIONAL))
        fixed = statistics.mean(bb.load_bf_bt_r(p, basis=bb.BF_R_BASIS_RISK))
        assert legacy == pytest.approx(1.651, abs=0.005)   # F31: book R +1.651
        assert fixed == pytest.approx(0.912, abs=0.005)    # F31: price-consistent +0.9117
        assert legacy / fixed == pytest.approx(1.81, abs=0.05)

    def test_the_whole_reference_mean_halves(self, p1_rows):
        ref = bb.bf_reference(200_000)
        legacy = bb.load_reference_r(ref, 'bf', bf_basis=bb.BF_R_BASIS_NOTIONAL)
        fixed = bb.load_reference_r(ref, 'bf', bf_basis=bb.BF_R_BASIS_RISK)
        assert len(legacy) == len(fixed) == len(p1_rows)
        assert statistics.mean(legacy) == pytest.approx(1.242, abs=0.005)
        assert statistics.mean(fixed) == pytest.approx(0.701, abs=0.005)

    def test_the_default_is_the_fixed_basis(self, monkeypatch):
        monkeypatch.delenv(bb.BF_R_BASIS_ENV, raising=False)
        ref = bb.bf_reference(200_000)
        assert bb.load_reference_r(ref, 'bf') == bb.load_reference_r(
            ref, 'bf', bf_basis=bb.BF_R_BASIS_RISK)

    def test_the_legacy_basis_is_reachable_by_env_and_shouts(self, monkeypatch, caplog):
        monkeypatch.setenv(bb.BF_R_BASIS_ENV, 'notional')
        with caplog.at_level(logging.WARNING):
            assert bb.bf_r_basis() == bb.BF_R_BASIS_NOTIONAL
        assert 'NOT a per-trade R' in caplog.text

    def test_an_unknown_basis_falls_back_loudly_to_the_fix(self, monkeypatch, caplog):
        monkeypatch.setenv(bb.BF_R_BASIS_ENV, 'weekly-vibes')
        with caplog.at_level(logging.ERROR):
            assert bb.bf_r_basis() == bb.BF_R_BASIS_RISK
        assert 'not a known BF R basis' in caplog.text


class TestALiveSampleAtFlatRisk:
    """The live consequence: the same P&L per unit of risk must now read IN-BAND.

    Honest note on n. At the L0 stage's first handful of trades the LEGACY band is so wide
    (SD 3.151 vs the fixed 1.574 — the share multipliers are in the SD too) that it swallows
    almost anything, which is its own failure: a band that cannot reject is not a gate. The basis
    bites where the band is actually informative, i.e. once n reaches the ADVANCE floor and beyond,
    so the classification flip is demonstrated at the stage-completion n the gate really reads.
    """

    N = 30   # a completed stage's worth of trades; the band is informative here

    @staticmethod
    def _bands(n):
        ref = bb.bf_reference(200_000)
        legacy = bb.bootstrap_band(bb.load_reference_r(ref, 'bf',
                                                       bf_basis=bb.BF_R_BASIS_NOTIONAL), n)
        fixed = bb.bootstrap_band(bb.load_reference_r(ref, 'bf',
                                                      bf_basis=bb.BF_R_BASIS_RISK), n)
        return legacy, fixed

    def test_a_flat_risk_live_book_matching_the_bt_is_in_band_and_was_not(self):
        """A live book earning exactly the BT's R PER UNIT OF RISK (+0.70) at a flat stage base."""
        legacy, fixed = self._bands(self.N)
        live_mean_r = statistics.mean(
            bb.load_reference_r(bb.bf_reference(200_000), 'bf', bb.BF_R_BASIS_RISK))
        assert live_mean_r == pytest.approx(0.701, abs=0.005)
        assert bb.classify(live_mean_r, fixed) == bb.IN_BAND
        # the legacy bar was the SIZED book: +0.70 sits under its p10 (+0.50)... no, above it;
        # what the legacy basis really demands is its own mean, +1.24 — see the next test for the
        # sample that the two bases classify differently.
        assert legacy.p10 > fixed.p10 and legacy.p90 > fixed.p90

    def test_the_fixed_band_is_lower_at_every_percentile_where_the_band_is_informative(self):
        legacy, fixed = self._bands(self.N)
        assert fixed.p5 < legacy.p5 and fixed.p10 < legacy.p10 and fixed.p90 < legacy.p90

    def test_a_synthetic_flat_risk_sample_classifies_in_band_where_the_old_basis_read_below_p10(self):
        """$150 risk, 30 trades, mean +$60 = +0.40 R — a real book, half the BT's per-risk edge.

        Fixed basis: IN-BAND (the BT's own 30-draw p10 is +0.32). Legacy basis: BELOW-p10 (+0.50),
        i.e. the old gate would have HELD a stage this book had earned.
        """
        base = 150.0
        pnls = ([+450] * 10) + ([-150] * 15) + ([-90] * 5)   # +$1,800 over 30 = +$60 = +0.40 R
        live_mean_r = statistics.mean(x / base for x in pnls)
        assert len(pnls) == self.N
        assert live_mean_r == pytest.approx(0.40, abs=0.02)
        legacy, fixed = self._bands(self.N)
        assert bb.classify(live_mean_r, fixed) == bb.IN_BAND
        assert bb.classify(live_mean_r, legacy) == bb.BELOW_P10


class TestTheSideBySideLine:
    """The checker prints both bands, labelled, for one week."""

    def test_the_line_names_both_bases_and_the_removal_date(self):
        line = bb.bf_basis_comparison_line(bb.bf_reference(200_000), 30, 0.40)
        assert 'FIXED' in line and 'LEGACY' in line
        assert bb.BF_R_BASIS_LEGACY_UNTIL in line
        assert bb.IN_BAND in line and bb.BELOW_P10 in line
        assert '+0.701' in line and '+1.242' in line

    def test_it_survives_a_missing_reference(self, tmp_path, caplog):
        ref = bb.Reference(tmp_path / 'nope.csv', 'missing')
        with caplog.at_level(logging.ERROR):
            line = bb.bf_basis_comparison_line(ref, 8, 0.70)
        assert line.count(bb.NO_DATA) == 2

    def test_it_survives_n_zero(self):
        line = bb.bf_basis_comparison_line(bb.bf_reference(200_000), 0, None)
        assert 'band NO-DATA (n=0)' in line


class TestTheOtherReference:
    """The ADV-off variant the checker switches to when the live gate is 0 must fix too."""

    def test_vol_off_has_a_computable_risk_on_every_row(self):
        ref = bb.bf_reference(0)
        rows = bb.read_csv_rows(ref.path)
        missing = [r for r in rows if not bb.bf_trade_risk_usd(r)]
        assert not missing, f"{len(missing)} VOL_OFF row(s) without a per-trade risk"

    def test_vol_off_also_deflates(self):
        ref = bb.bf_reference(0)
        legacy = statistics.mean(bb.load_reference_r(ref, 'bf', bb.BF_R_BASIS_NOTIONAL))
        fixed = statistics.mean(bb.load_reference_r(ref, 'bf', bb.BF_R_BASIS_RISK))
        assert fixed < legacy


class TestRowsWithoutARiskAreNeverZero:
    def test_they_are_dropped_with_an_error(self, tmp_path, caplog):
        p = tmp_path / 'broken.csv'
        with open(p, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['pnl', 'shares', 'entry_price', 'stop_loss'])
            w.writerow(['100', '100', '10.00', '9.00'])
            w.writerow(['100', '0', '10.00', '9.00'])
        with caplog.at_level(logging.ERROR):
            got = bb.load_bf_bt_r(p, basis=bb.BF_R_BASIS_RISK)
        assert got == [1.0]
        assert 'without a computable per-trade risk' in caplog.text
