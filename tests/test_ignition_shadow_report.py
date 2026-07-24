"""Ignition S1 daily report — the go/no-go instrument gets its own tests
(2026-07-19 review round 3: the script that produces the week-1 verdict
had ZERO coverage; a resim bug would silently corrupt the S3 decision)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
import ignition_shadow_report as rep


def _bars(rows):
    """rows: list of (m, open, high, low, close)."""
    return pd.DataFrame(
        [{'m': m, 'open': o, 'high': h, 'low': lo, 'close': c,
          'volume': 10000} for m, o, h, lo, c in rows])


ENTRY, STOP = 10.0, 9.0   # R = $1


class TestResimExit:
    def test_stop_hit_before_arm(self):
        b = _bars([(601, 9.9, 10.0, 8.9, 9.0)])
        rr, reason = rep.resim_exit(b, ENTRY, STOP, 600)
        assert reason == 'stop'
        assert rr == pytest.approx((STOP * 0.999 - ENTRY) / 1.0)

    def test_arm_then_lock(self):
        # bar1 arms (high >= entry + 1.75R), bar2 dips to the +0.5R lock
        b = _bars([(601, 10.5, 11.8, 10.4, 11.7),
                   (602, 11.0, 11.1, 10.4, 10.5)])
        rr, reason = rep.resim_exit(b, ENTRY, STOP, 600)
        assert reason == 'lock'
        assert rr == pytest.approx((10.5 * 0.999 - ENTRY) / 1.0)

    def test_gap_open_below_stop_fills_at_open_not_stop(self):
        """Gap-down realism: a bar OPENING below the stop cannot fill AT
        the stop — pre-fix the resim credited stop-price fills on gaps,
        overstating P&L exactly where monster-tail books hurt most."""
        b = _bars([(601, 8.0, 8.2, 7.9, 8.1)])   # opens $1 below stop
        rr, reason = rep.resim_exit(b, ENTRY, STOP, 600)
        assert reason == 'stop'
        assert rr == pytest.approx((8.0 * 0.999 - ENTRY) / 1.0)

    def test_eod_force_close_at_1545(self):
        b = _bars([(601, 10.1, 10.3, 10.0, 10.2),
                   (945, 10.4, 10.5, 10.3, 10.4)])
        rr, reason = rep.resim_exit(b, ENTRY, STOP, 600)
        assert reason == 'eod'
        assert rr == pytest.approx((10.4 - ENTRY) / 1.0)

    def test_entry_bar_excluded(self):
        # the trigger bar itself (m == entry_min) lows below stop — must
        # NOT count (entry happened during/after it); next bar survives
        b = _bars([(600, 9.5, 10.1, 8.5, 10.0),
                   (601, 10.1, 10.2, 10.0, 10.1)])
        rr, reason = rep.resim_exit(b, ENTRY, STOP, 600)
        assert reason == 'eod'   # ran out of bars -> last close
        assert rr == pytest.approx((10.1 - ENTRY) / 1.0)

    def test_no_post_bars(self):
        b = _bars([(599, 9.9, 10.0, 9.8, 9.9)])
        rr, reason = rep.resim_exit(b, ENTRY, STOP, 600)
        assert (rr, reason) == (0.0, 'none')

    def test_survives_to_last_close_when_no_exit(self):
        b = _bars([(601, 10.1, 10.4, 10.05, 10.3)])
        rr, reason = rep.resim_exit(b, ENTRY, STOP, 600)
        assert reason == 'eod'
        assert rr == pytest.approx((10.3 - ENTRY) / 1.0)


def _run_main(monkeypatch, tmp_path, day, recs, bars_fn=None,
              no_telegram=False):
    """Drive main() end-to-end: journal in tmp logs, day_bars + telegram
    monkeypatched. Returns (exit_code, printed_msg, sent_messages)."""
    (tmp_path / 'logs').mkdir(exist_ok=True)
    if recs is not None:
        p = tmp_path / 'logs' / f'ignition_shadow_{day}.jsonl'
        p.write_text('\n'.join(json.dumps(r) for r in recs) + '\n')
    sent = []
    monkeypatch.setattr(rep.rc, 'ROOT', tmp_path)
    monkeypatch.setattr(rep.rc, 'send_telegram', lambda m: sent.append(m))
    monkeypatch.setattr(
        rep, 'day_bars',
        bars_fn or (lambda sym, d: _bars([(601, 10.1, 10.3, 10.0, 10.2),
                                          (945, 10.4, 10.5, 10.3, 10.4)])))
    argv = ['ignition_shadow_report.py', '--date', day]
    if no_telegram:
        argv.append('--no-telegram')
    monkeypatch.setattr(sys, 'argv', argv)
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = rep.main()
    return code, buf.getvalue(), sent


def _trig(sym='IGNI', **kw):
    r = {'ts_utc': 'x', 'symbol': sym, 'day': '2026-07-20',
         'minute_et': 600, 'minute_final_et': 600,
         'intraday_change_pct': 15.0, 'gap_pct': 12.0, 'price': 10.0,
         'has_news': True, 'verdict': 'SHADOW_TRIGGER',
         'catalyst': 'news', 'spread_bps': 40.0, 'latency_s': 30.0,
         'bars_fetch_s': 1.0, 'hypo_entry': 10.0, 'hypo_stop': 9.0,
         'hypo_position_usd': 25000.0}
    r.update(kw)
    return r


class TestMainEndToEnd:
    def test_trigger_day_message_and_telegram(self, monkeypatch, tmp_path):
        code, out, sent = _run_main(
            monkeypatch, tmp_path, '2026-07-20', [_trig()])
        assert code == 0
        assert len(sent) == 1
        assert 'IGNI' in sent[0] and '1 trigger' in sent[0]
        assert '✓trigger rate' in sent[0]
        assert '✓spread median' in sent[0]
        assert '✓latency p90' in sent[0]

    def test_no_journal_still_telegrams(self, monkeypatch, tmp_path):
        """Silent-death visibility: a dead shadow must produce an alert,
        not a silent skip (pre-fix: print-only, no Telegram all week)."""
        code, out, sent = _run_main(
            monkeypatch, tmp_path, '2026-07-20', None)
        assert code == 0
        assert len(sent) == 1
        assert 'NO journal' in sent[0]

    def test_chase_violation_excluded_from_tally(self, monkeypatch, tmp_path):
        """Pre-guard journals (7/20-24): the report reclassifies entries
        past 1.155x open instead of tallying a trade the BT refuses."""
        recs = [_trig(day_open=8.0, hypo_entry=10.0)]   # 1.25x open
        code, out, sent = _run_main(
            monkeypatch, tmp_path, '2026-07-20', recs)
        assert 'chase-skip' in sent[0]
        assert '0 trigger' not in sent[0]   # still counted as a trigger
        assert '$+0' in sent[0]             # but zero P&L tallied

    def test_inverted_quote_skips_resim(self, monkeypatch, tmp_path):
        code, out, sent = _run_main(
            monkeypatch, tmp_path, '2026-07-20',
            [_trig(hypo_entry=8.5, hypo_stop=9.0)])
        assert code == 0
        assert 'SKIPPED resim' in sent[0]
        assert 'bad quote' in sent[0]

    def test_pass_bar_boundaries_inclusive(self, monkeypatch, tmp_path):
        # spread median/p90 exactly 60 and latency exactly 90 -> pass
        # (bars are <=, not <)
        recs = [_trig(spread_bps=60.0, latency_s=90.0)]
        code, out, sent = _run_main(
            monkeypatch, tmp_path, '2026-07-20', recs)
        assert '✗' not in sent[0].split('checks:')[1].split('skips:')[0]

    def test_zero_triggers_flags_orange(self, monkeypatch, tmp_path):
        recs = [{'symbol': 'AAA', 'day': '2026-07-20', 'minute_et': 600,
                 'verdict': 'skip_no_catalyst', 'spread_bps': 30.0}]
        code, out, sent = _run_main(
            monkeypatch, tmp_path, '2026-07-20', recs)
        assert sent[0].startswith('🟠')
        assert '✗trigger rate' in sent[0]
        assert 'skip_no_catalyst' in sent[0]

    def test_resim_failure_reported_not_raised(self, monkeypatch, tmp_path):
        def _boom(sym, d):
            raise RuntimeError('api down')
        code, out, sent = _run_main(
            monkeypatch, tmp_path, '2026-07-20', [_trig()], bars_fn=_boom)
        assert code == 0
        assert 'resim failed' in sent[0]
