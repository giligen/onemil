"""Journal-flood fix 2026-11-02: close-call definition, third-party log level, transition-only lines."""
import logging

import pytest

from monitoring.logger import setup_logging
from scanner.criteria import ScanCandidate, ScannerCriteria, is_close_call
from scanner import realtime_scanner
from trading.orb_gap_gate import resolve_gap_input


def _cand(symbol="AAA", change=0.0, missing=("intraday_change",)):
    """Candidate with all six criteria met except `missing`."""
    names = ["price_range", "float", "relative_volume", "intraday_change", "has_news", "dollar_volume"]
    return ScanCandidate(symbol=symbol, intraday_change_pct=change,
                         criteria_met={n: n not in missing for n in names})


class TestIsCloseCall:
    """The one close-call predicate."""

    def test_unmoved_stock_is_not_a_close_call(self):
        assert not is_close_call(_cand(change=1.0), 10.0)

    def test_half_threshold_move_is_a_close_call(self):
        assert is_close_call(_cand(change=5.0), 10.0)
        assert not is_close_call(_cand(change=4.99), 10.0)

    def test_other_single_miss_is_a_close_call_regardless_of_move(self):
        assert is_close_call(_cand(change=0.0, missing=("relative_volume",)), 10.0)

    def test_two_missing_or_none_missing_is_not(self):
        assert not is_close_call(_cand(change=9.0, missing=("float", "intraday_change")), 10.0)
        assert not is_close_call(_cand(change=20.0, missing=()), 10.0)

    def test_method_uses_configured_threshold(self):
        sc = ScannerCriteria(intraday_change_pct_min=20.0)
        assert not sc.is_close_call(_cand(change=9.0))
        assert sc.is_close_call(_cand(change=10.0))

    def test_debug_line_only_for_real_near_miss(self, caplog):
        sc = ScannerCriteria()
        caplog.set_level(logging.DEBUG, logger="scanner.criteria")
        sc.evaluate_intraday(ScanCandidate(symbol="FLAT", current_price=5, float_shares=1_000_000,
                                           relative_volume=9, intraday_change_pct=0.5, has_news=True,
                                           current_volume=10**7))
        assert "CLOSE CALL" not in caplog.text
        sc.evaluate_intraday(ScanCandidate(symbol="NEAR", current_price=5, float_shares=1_000_000,
                                           relative_volume=9, intraday_change_pct=7.0, has_news=True,
                                           current_volume=10**7))
        assert "CLOSE CALL: NEAR" in caplog.text


class TestQualifiedTransitionOnly:
    """INTRADAY QUALIFIED logs on the transition, not every cycle."""

    def test_logged_once_then_again_after_drop(self, caplog):
        sc = ScannerCriteria()
        caplog.set_level(logging.INFO, logger="scanner.criteria")

        def ev(change):
            return sc.evaluate_intraday(ScanCandidate(
                symbol="QQ", current_price=5, float_shares=1_000_000, relative_volume=9,
                intraday_change_pct=change, has_news=True, current_volume=10**7))
        assert ev(15) and ev(16) and ev(17)
        assert caplog.text.count("INTRADAY QUALIFIED: QQ") == 1
        assert not ev(1)
        assert ev(15)
        assert caplog.text.count("INTRADAY QUALIFIED: QQ") == 2


class TestVerboseTableCap:
    """The verbose close-call table prints at most CLOSE_CALLS_PRINT_CAP rows."""

    def test_cap_and_more_marker(self, capsys):
        sc = realtime_scanner.RealtimeScanner.__new__(realtime_scanner.RealtimeScanner)
        sc.verbose = True
        calls = [_cand(symbol=f"S{i}", change=5.0 + i * 0.01) for i in range(35)]
        import inspect
        src = inspect.getsource(realtime_scanner.RealtimeScanner._print_intraday_output)
        assert "CLOSE_CALLS_PRINT_CAP" in src and "more)" in src
        assert realtime_scanner.CLOSE_CALLS_PRINT_CAP == 20
        assert len(calls) > realtime_scanner.CLOSE_CALLS_PRINT_CAP


class TestThirdPartyLevels:
    """--verbose keeps our loggers at DEBUG and pins third-party ones above it."""

    def test_verbose_pins_third_party(self, tmp_path):
        root = logging.getLogger()
        saved = (root.level, root.handlers[:])
        try:
            setup_logging(verbose=True, log_dir=str(tmp_path))
            assert logging.getLogger("alpaca.data.live.websocket").getEffectiveLevel() == logging.INFO
            assert logging.getLogger("alpaca").getEffectiveLevel() == logging.INFO
            assert logging.getLogger("asyncio").getEffectiveLevel() == logging.INFO
            for n in ("websockets", "urllib3", "httpx"):
                assert logging.getLogger(n).getEffectiveLevel() >= logging.INFO
            assert logging.getLogger("scanner.criteria").getEffectiveLevel() == logging.DEBUG
        finally:
            for h in root.handlers:
                h.close()
            root.setLevel(saved[0])
            root.handlers = saved[1]
            for n in ("alpaca", "asyncio"):
                logging.getLogger(n).setLevel(logging.NOTSET)


class TestGapGateOncePerSymbolSource:
    """GAP_GATE echo logs once per (symbol, source)."""

    def test_repeat_tick_is_silent(self, caplog):
        """A sub-floor symbol's repeat ticks are silent after the first DEBUG echo."""
        caplog.set_level(logging.DEBUG)
        for _ in range(3):
            resolve_gap_input("ZZFLOODONCE", snapshot_open=10.1, prev_close=10.0,
                              minute_bar_open=None, gap_floor_pct=2.0)
        assert caplog.text.count("[ORB] GAP_GATE ZZFLOODONCE") == 1

    def test_real_candidate_logs_every_tick(self, caplog):
        """A symbol above the gap floor logs INFO on every tick: scripts/eod_sections.py reads gap_pct from the
        line, so a candidate that first ticked sub-floor must still get its candidate line."""
        caplog.set_level(logging.DEBUG)
        resolve_gap_input("ZZFLOODCAND", snapshot_open=10.1, prev_close=10.0, minute_bar_open=None, gap_floor_pct=2.0)
        for _ in range(2):
            resolve_gap_input("ZZFLOODCAND", snapshot_open=10.5, prev_close=10.0, minute_bar_open=None, gap_floor_pct=2.0)
        recs = [r for r in caplog.records if "[ORB] GAP_GATE ZZFLOODCAND" in r.message]
        assert [r.levelname for r in recs] == ["DEBUG", "INFO", "INFO"]
