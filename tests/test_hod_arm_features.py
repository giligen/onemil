"""Arm-time feature columns on the HOD dry ledger (owner's ask 2026-09-29 — 'why not test everything
tomorrow': every paper arm carries its arm-time features so one live session is a forward read of every
future cut). Covers `trading.hod_break.arm_features` (pure, causal) and its header-gated wiring into
`HodBreakEngine._append_dry_ledger`, mirroring the `account` column's backward-compat rule
(tests/test_hod_resting_entry.py::TestDryLedgerWriterAccountColumn).
"""
import numpy as np
import pytest

from trading.hod_break import ARM_FEATURE_COLUMNS, arm_features
from tests.test_hod_break_engine import cfg, bars_df, admit
from tests.test_hod_resting_entry import (ADV20, mock_alpaca, mock_db, mock_sm, read_ledger,
                                           resting_engine, tape, trades_db)

# A clean 4-bar synthetic day (flat bars: high == low == close) so every ratio is hand-checkable.
# minute 570 = 09:30 ET; day_open = bar0's open = 10.0; the level (running HOD) is set at bar 1 (12.0).
BARS = [
    (10.0, 10.0, 10.0, 10.0, 1000),   # bar 0, minute 570
    (10.0, 12.0, 12.0, 12.0, 2000),   # bar 1, minute 571 -- the LEVEL bar (sets HOD 12.0)
    (12.0, 11.0, 11.0, 11.0, 3000),   # bar 2, minute 572
    (11.0, 11.5, 11.5, 11.5, 4000),   # bar 3, minute 573 -- the ARM bar (arm_idx=3)
]
ARM = dict(level=12.0, trigger=12.0, stop=10.8, limit=12.0)   # r_pct = (12.0-10.8)/12.0*100 = 10.0 exactly
ADV20_SYNTH = 100_000.0


def _arrays(bars):
    o = np.array([b[0] for b in bars], dtype=float); h = np.array([b[1] for b in bars], dtype=float)
    l = np.array([b[2] for b in bars], dtype=float); c = np.array([b[3] for b in bars], dtype=float)
    v = np.array([b[4] for b in bars], dtype=float); m = np.array([570 + i for i in range(len(bars))], dtype=int)
    return o, h, l, c, v, m


class TestArmFeaturesKnownAnswers:
    """One test per feature on the same 4-bar synthetic day (arm_idx=3) -- hand-derived expected values."""

    def _feat(self):
        o, h, l, c, v, m = _arrays(BARS)
        return arm_features(o, h, l, c, v, m, 3, ARM, ADV20_SYNTH, day_open=10.0)

    def test_r_pct(self):
        assert self._feat()['r_pct'] == pytest.approx((12.0 - 10.8) / 12.0 * 100.0)

    def test_lvl_vs_open_pct(self):
        assert self._feat()['lvl_vs_open_pct'] == pytest.approx(0.2)

    def test_lvl_vs_vwap_pct(self):
        typical = [10.0, 12.0, 11.0, 11.5]              # h == l == c on every synthetic bar
        vol = [1000, 2000, 3000, 4000]
        vwap = sum(t * w for t, w in zip(typical, vol)) / sum(vol)
        assert self._feat()['lvl_vs_vwap_pct'] == pytest.approx(12.0 / vwap - 1.0)

    def test_cumvol_to_arm(self):
        assert self._feat()['cumvol_to_arm'] == pytest.approx(10000.0)

    def test_cumvol_over_adv20(self):
        assert self._feat()['cumvol_over_adv20'] == pytest.approx(10000.0 / 100_000.0)

    def test_min_since_open(self):
        assert self._feat()['min_since_open'] == 3               # arm bar at minute 573, open at 570

    def test_level_age_min(self):
        assert self._feat()['level_age_min'] == 2                # level bar at minute 571, arm at 573

    def test_prior_crosses_today(self):
        assert self._feat()['prior_crosses_today'] == 1          # only bar1 (12.0) ties the level before the arm

    def test_arm_bar_vol_x(self):
        assert self._feat()['arm_bar_vol_x'] == pytest.approx(4000.0 / 2500.0)


class TestArmFeaturesFractionalMinuteRule:
    def test_never_reads_bars_after_arm_idx(self):
        """A bogus 5th bar appended past arm_idx=3 must not change any feature."""
        o, h, l, c, v, m = _arrays(BARS)
        base = arm_features(o, h, l, c, v, m, 3, ARM, ADV20_SYNTH, day_open=10.0)
        o5 = np.append(o, 999.0); h5 = np.append(h, 999.0); l5 = np.append(l, 999.0)
        c5 = np.append(c, 999.0); v5 = np.append(v, 999999.0); m5 = np.append(m, 9999)
        again = arm_features(o5, h5, l5, c5, v5, m5, 3, ARM, ADV20_SYNTH, day_open=10.0)
        assert again == base

    def test_raises_when_arm_idx_beyond_available_bars(self):
        o, h, l, c, v, m = _arrays(BARS)
        with pytest.raises(IndexError):
            arm_features(o, h, l, c, v, m, 10, ARM, ADV20_SYNTH, day_open=10.0)


class TestArmFeatureColumnsHeaderBackwardCompat:
    """Mirrors tests/test_hod_resting_entry.py::TestDryLedgerWriterAccountColumn."""

    def test_new_file_gets_all_columns_with_real_values(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        admit(e); cand = e.candidates['ABC']
        cand.day_open = 10.0; cand.adv20 = ADV20_SYNTH
        for i, (o, h, l, c, v) in enumerate(BARS):
            cand.set_bar(570 + i, o, h, l, c, v)
        arm = dict(ARM, idx=3, arm_ts='2026-09-29T09:33:00')
        e._append_dry_ledger(cand, arm, e._et_now(), ask=12.0, filled=True, fill_px=12.0, tape_accurate=True)
        rows = read_ledger(e.dry_ledger_path)
        assert len(rows) == 1
        for col in ARM_FEATURE_COLUMNS:
            assert col in rows[0]
        assert float(rows[0]['r_pct']) == pytest.approx(10.0)
        assert float(rows[0]['lvl_vs_open_pct']) == pytest.approx(0.2)
        assert rows[0]['min_since_open'] == '3'
        assert rows[0]['level_age_min'] == '2'
        assert rows[0]['prior_crosses_today'] == '1'

    def test_existing_header_without_the_new_columns_is_never_rewritten(
            self, mock_alpaca, mock_db, mock_sm, tmp_path, caplog):
        path = tmp_path / 'hod_dry_entry_ledger.csv'
        old_header = ('date,symbol,arm_ts,cross_ts,level,trigger,limit,ask,filled,fill_px,stop,target,'
                      'tape_accurate,live,account')
        path.write_text(old_header + '\n')
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        admit(e); cand = e.candidates['ABC']
        cand.day_open = 10.0; cand.adv20 = ADV20_SYNTH
        for i, (o, h, l, c, v) in enumerate(BARS):
            cand.set_bar(570 + i, o, h, l, c, v)
        arm = dict(ARM, idx=3, arm_ts='2026-09-29T09:33:00')
        e._append_dry_ledger(cand, arm, e._et_now(), ask=12.0, filled=True, fill_px=12.0, tape_accurate=True)
        with open(path) as fh:
            lines = fh.read().strip('\n').split('\n')
        assert lines[0] == old_header
        assert len(lines[1].split(',')) == len(old_header.split(','))
        for col in ARM_FEATURE_COLUMNS:
            assert col not in lines[0]


class TestFeatureFailureFallback:
    def test_missing_bars_leaves_columns_empty_and_warns_once_per_symbol_day(
            self, mock_alpaca, mock_db, mock_sm, tmp_path, caplog):
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        admit(e); cand = e.candidates['ABC']          # no bars ingested -> _rth_arrays(cand) is None
        arm = dict(ARM, idx=3, arm_ts='2026-09-29T09:33:00')
        with caplog.at_level('WARNING'):
            e._append_dry_ledger(cand, arm, e._et_now(), ask=12.0, filled=True, fill_px=12.0, tape_accurate=True)
            e._append_dry_ledger(cand, arm, e._et_now(), ask=12.0, filled=True, fill_px=12.0, tape_accurate=True)
        rows = read_ledger(e.dry_ledger_path)
        assert rows[0]['r_pct'] == '' and rows[1]['r_pct'] == ''
        warnings = [r for r in caplog.records if 'arm-time feature computation failed' in r.message]
        assert len(warnings) == 1                      # ONE warning per symbol-day, not per row


class TestArmFeaturesIntegrationThroughEngine:
    """A real dry cross through the full engine (arm_state -> resolve -> _append_dry_ledger), asserting the
    written row's feature values against independently-derived expectations from tape()'s own bar data."""

    def test_dry_cross_row_carries_correct_arm_time_features(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        admit(e); e._ingest_bars('ABC', bars_df(tape()))
        rows = read_ledger(e.dry_ledger_path)
        assert len(rows) == 1 and rows[0]['filled'] == '1'
        row = rows[0]
        t = tape()[:8]                                    # bars 0..7 -- arm_idx=7 is the arm this fill resolves
        opens, highs, lows, closes, vols = zip(*t)
        day_open = opens[0]
        level, trigger, stop = 11.0, 11.01, 10.7            # trading.hod_break.arm_state's own values at j=7
        assert float(row['r_pct']) == pytest.approx((trigger - stop) / trigger * 100.0)
        assert float(row['lvl_vs_open_pct']) == pytest.approx(level / day_open - 1.0)
        typical = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
        vwap = sum(t_ * v for t_, v in zip(typical, vols)) / sum(vols)
        assert float(row['lvl_vs_vwap_pct']) == pytest.approx(level / vwap - 1.0)
        assert float(row['cumvol_to_arm']) == pytest.approx(float(sum(vols)))
        assert float(row['cumvol_over_adv20']) == pytest.approx(sum(vols) / ADV20)
        assert row['min_since_open'] == '7'                 # arm bar at minute 570+7
        assert row['level_age_min'] == '5'                   # level set at bar idx2, arm at bar idx7 -> 5 min
        assert row['prior_crosses_today'] == '1'             # only bar idx2 (11.0) ties the level before the arm
        assert float(row['arm_bar_vol_x']) == pytest.approx(vols[7] / (sum(vols) / len(vols)))
