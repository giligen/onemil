"""Unit tests for scripts/bf_decision_parity.py (BF live-vs-BT parity, Stage 1).

Covers the pure classification/comparison functions plus the file-backed
loaders against synthetic fixtures in tmp_path — no live DB, no production
cache, no API, no subprocess.
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bf_decision_parity as bp  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic trade builders
# ---------------------------------------------------------------------------

def bt_trade(symbol, pnl, exit_reason='stop', entry=10.0):
    """Minimal Stage-2 BT trade dict as run_stage2_backtest returns it."""
    return {'symbol': symbol, 'entry_time_et': '09:44:00',
            'entry_price': entry, 'exit_time_et': '10:00:00',
            'exit_price': entry * (1 + pnl / 10000.0),
            'exit_reason': exit_reason, 'pnl': pnl, 'shares': 1000}


def live_row(symbol, pnl=None, exit_reason=None, status='filled',
             entry=10.0, fill=None):
    """Minimal trades.db row dict as load_live_bf_rows returns it."""
    filled = status == 'filled'
    return {'symbol': symbol, 'order_status': status, 'entry_price': entry,
            'fill_price': (fill if fill is not None else entry) if filled
            else None,
            'exit_price': entry if pnl is not None else None,
            'exit_reason': exit_reason, 'pnl': pnl, 'shares': 1000,
            'filled_qty': 1000 if filled else None}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestPureHelpers:
    def test_pnl_sign(self):
        assert bp.pnl_sign(220.83) == 1
        assert bp.pnl_sign(-469.48) == -1
        assert bp.pnl_sign(0.0) == 0
        assert bp.pnl_sign(None) is None

    def test_entry_delta_bps(self):
        # BEZ 8/6: BT entry 8.92, live fill 8.89 → about -33.6 bps
        d = bp.entry_delta_bps(8.92, 8.89)
        assert d == pytest.approx(-33.63, abs=0.05)
        assert bp.entry_delta_bps(None, 8.89) is None
        assert bp.entry_delta_bps(8.92, None) is None
        assert bp.entry_delta_bps(0, 8.89) is None  # zero guard

    def test_overall_status(self):
        assert bp.overall_status(False, '2026-08-07', 0) == 'AGREE'
        assert bp.overall_status(False, '2026-08-07', 3) == 'DIVERGE(3)'
        assert bp.overall_status(True, '2026-08-07', 0) == \
            'BT_STALE (cache ends 2026-08-07)'
        # Stale ALWAYS wins — even with divergences recorded
        assert bp.overall_status(True, '2026-08-07', 5).startswith('BT_STALE')
        assert bp.overall_status(True, None, 0) == \
            'BT_STALE (cache ends NEVER)'


# ---------------------------------------------------------------------------
# BOTH comparison
# ---------------------------------------------------------------------------

class TestCompareBoth:
    def test_sign_flip_and_exit_reason_divergence(self):
        """The BEZ 8/6 class: BT stop loss vs live trail-stop win."""
        cmp = bp.compare_both(
            bt_trade('BEZ', pnl=-469.48, exit_reason='stop', entry=8.92),
            live_row('BEZ', pnl=220.83, exit_reason='trail_stop',
                     entry=8.879, fill=8.89))
        assert cmp['divergent'] is True
        assert cmp['pnl_sign_match'] is False
        assert cmp['exit_reason_match'] is False
        assert cmp['pnl_delta_usd'] == pytest.approx(220.83 - (-469.48))
        assert cmp['entry_delta_bps'] == pytest.approx(-33.63, abs=0.05)
        assert any('SIGN FLIP' in r for r in cmp['divergence_reasons'])
        assert any('trail_stop' in r for r in cmp['divergence_reasons'])

    def test_clean_agreement(self):
        cmp = bp.compare_both(
            bt_trade('CION', pnl=136.83, exit_reason='force_close',
                     entry=7.20),
            live_row('CION', pnl=94.82, exit_reason='force_close',
                     entry=7.17))
        assert cmp['divergent'] is False
        assert cmp['pnl_sign_match'] is True
        assert cmp['exit_reason_match'] is True
        assert cmp['divergence_reasons'] == []

    def test_live_never_filled_is_divergent(self):
        """Live placement cancelled while BT simulated a trade (SPHL 8/6)."""
        cmp = bp.compare_both(
            bt_trade('SPHL', pnl=-118.28, exit_reason='stop', entry=3.22),
            live_row('SPHL', status='cancelled', entry=3.2))
        assert cmp['divergent'] is True
        assert cmp['live_filled'] is False
        assert any('never filled' in r for r in cmp['divergence_reasons'])
        # No pnl/exit comparison possible — must be None, not False
        assert cmp['pnl_sign_match'] is None
        assert cmp['exit_reason_match'] is None


# ---------------------------------------------------------------------------
# Day-level classification
# ---------------------------------------------------------------------------

class TestClassifyDay:
    def test_both_with_sign_flip(self):
        cls = bp.classify_day(
            bt_trades=[bt_trade('BEZ', -469.48, 'stop', 8.92)],
            live_rows=[live_row('BEZ', 220.83, 'trail_stop', entry=8.89)],
            stage1_day_symbols={'BEZ'}, bt_stale=False)
        assert len(cls['both']) == 1
        assert cls['bt_only'] == [] and cls['live_only'] == []
        assert cls['n_divergent'] == 2  # sign flip + exit reason
        assert any('BEZ' in d and 'SIGN FLIP' in d
                   for d in cls['divergences'])

    def test_bt_only(self):
        cls = bp.classify_day(
            bt_trades=[bt_trade('OESX', -564.25, 'stop', 19.81)],
            live_rows=[], stage1_day_symbols={'OESX'}, bt_stale=False)
        assert len(cls['bt_only']) == 1
        assert cls['bt_only'][0]['label'] == bp.LABEL_BT_ONLY
        assert cls['n_divergent'] == 1
        assert 'OESX' in cls['divergences'][0]

    def test_live_only_off_universe_vs_in_universe(self):
        cls = bp.classify_day(
            bt_trades=[],
            live_rows=[live_row('LUNL', -29.44, 'thin_liquidity_reject'),
                       live_row('REAL', 50.0, 'trail_stop')],
            stage1_day_symbols={'REAL'},  # REAL had a raw Stage-1 row
            bt_stale=False)
        by_sym = {c['symbol']: c for c in cls['live_only']}
        assert by_sym['LUNL']['label'] == bp.LABEL_OFF_UNIVERSE
        assert by_sym['REAL']['label'] == bp.LABEL_IN_UNIVERSE
        assert cls['n_divergent'] == 2

    def test_stale_day_uses_common_stock_lookup(self):
        """On stale days universe attribution falls back to the production
        common-stock classifier: leveraged wrappers still get nailed."""
        lookup = {'LUNL': False, 'MYST': None}.get
        cls = bp.classify_day(
            bt_trades=[],
            live_rows=[live_row('LUNL', -29.44, 'thin_liquidity_reject'),
                       live_row('MYST', 10.0, 'trail_stop')],
            stage1_day_symbols=None, bt_stale=True,
            is_common_lookup=lookup)
        by_sym = {c['symbol']: c for c in cls['live_only']}
        assert by_sym['LUNL']['label'] == bp.LABEL_OFF_UNIVERSE_LEV
        assert by_sym['MYST']['label'] == bp.LABEL_UNIVERSE_UNKNOWN
        # Stale day charges NO divergences — status is BT_STALE, not DIVERGE
        assert cls['n_divergent'] == 0

    def test_stale_status_never_agree(self):
        assert bp.overall_status(True, '2026-08-07', 0) != 'AGREE'

    def test_empty_day_both_sides_is_agree_zero(self):
        cls = bp.classify_day(bt_trades=[], live_rows=[],
                              stage1_day_symbols=set(), bt_stale=False)
        assert cls['n_divergent'] == 0
        assert bp.overall_status(False, '2026-08-07', cls['n_divergent']) \
            == 'AGREE'

    def test_multiple_live_rows_prefers_filled(self):
        cls = bp.classify_day(
            bt_trades=[bt_trade('CION', 136.83, 'force_close', 7.20)],
            live_rows=[live_row('CION', status='cancelled', entry=7.15),
                       live_row('CION', 94.82, 'force_close', entry=7.17)],
            stage1_day_symbols={'CION'}, bt_stale=False)
        assert len(cls['both']) == 1
        assert cls['both'][0]['live_filled'] is True
        assert cls['both'][0]['divergent'] is False


# ---------------------------------------------------------------------------
# File-backed loaders (tmp_path fixtures)
# ---------------------------------------------------------------------------

CACHE_HEADER = ['symbol', 'date', 'entry_time_et', 'entry_price',
                'stop_loss', 'target', 'shares', 'exit_time_et',
                'exit_price', 'exit_reason', 'pnl', 'pnl_pct']


def write_cache(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(CACHE_HEADER)
        for sym, day in rows:
            w.writerow([sym, day, '09:44:00', '5.00', '4.90', '5.20',
                        '1000', '10:00:00', '5.10', 'trail_stop',
                        '100.00', '2.00'])


class TestScanCache:
    def test_max_date_and_day_symbols(self, tmp_path):
        cache = tmp_path / 'cache.csv'
        write_cache(cache, [('AAA', '2026-08-05'), ('BBB', '2026-08-06'),
                            ('CCC', '2026-08-06'), ('DDD', '2026-08-07')])
        max_date, day_syms = bp.scan_cache(cache, '2026-08-06')
        assert max_date == '2026-08-07'
        assert day_syms == {'BBB', 'CCC'}

    def test_stale_detection(self, tmp_path):
        cache = tmp_path / 'cache.csv'
        write_cache(cache, [('AAA', '2026-08-07')])
        max_date, day_syms = bp.scan_cache(cache, '2026-08-13')
        assert '2026-08-13' > max_date  # → caller must report BT_STALE
        assert day_syms == set()

    def test_empty_cache_is_stale(self, tmp_path):
        cache = tmp_path / 'cache.csv'
        write_cache(cache, [])
        max_date, day_syms = bp.scan_cache(cache, '2026-08-06')
        assert max_date is None
        assert bp.overall_status(True, max_date, 0) == \
            'BT_STALE (cache ends NEVER)'


class TestLoadLiveRows:
    def _make_db(self, tmp_path, rows):
        db = tmp_path / 'trades.db'
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE trades (trade_date TEXT, symbol TEXT, "
            "strategy TEXT, order_status TEXT, entry_price REAL, "
            "fill_price REAL, exit_price REAL, exit_reason TEXT, pnl REAL, "
            "shares INTEGER, filled_qty INTEGER)")
        conn.executemany(
            "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        conn.close()
        return db

    def test_filters_by_date_and_strategy_all_statuses(self, tmp_path):
        db = self._make_db(tmp_path, [
            ('2026-08-06', 'BEZ', 'bull_flag', 'filled', 8.879, 8.89,
             9.06, 'trail_stop', 220.83, 1299, 1299),
            ('2026-08-06', 'SPHL', 'bull_flag', 'cancelled', 3.2, None,
             None, None, None, 948, None),
            ('2026-08-06', 'XORB', 'orb', 'filled', 5.0, 5.0,
             5.5, 'static_lock', 500.0, 1000, 1000),      # other strategy
            ('2026-08-05', 'OLD', 'bull_flag', 'filled', 5.0, 5.0,
             5.5, 'trail_stop', 100.0, 1000, 1000),       # other day
        ])
        rows = bp.load_live_bf_rows('2026-08-06', db_path=db)
        assert sorted(r['symbol'] for r in rows) == ['BEZ', 'SPHL']
        sphl = next(r for r in rows if r['symbol'] == 'SPHL')
        assert sphl['order_status'] == 'cancelled'  # placements included
        assert sphl['fill_price'] is None

    def test_empty_day(self, tmp_path):
        db = self._make_db(tmp_path, [])
        assert bp.load_live_bf_rows('2026-08-06', db_path=db) == []


# ---------------------------------------------------------------------------
# Rendering (status must be loud in every channel)
# ---------------------------------------------------------------------------

class TestRendering:
    def _report(self, status='DIVERGE(1)', bt_stale=False, **cls):
        base = {'day': '2026-08-06', 'status': status, 'bt_stale': bt_stale,
                'cache_path': 'data/x.csv', 'cache_max_date': '2026-08-07',
                'n_bt_trades': 1, 'n_live_rows': 1, 'bt_trades': [],
                'stage1_day_symbols': [], 'both': [], 'bt_only': [],
                'live_only': [], 'divergences': [], 'n_divergent': 0,
                'generated_at_utc': 'now'}
        base.update(cls)
        return base

    def test_telegram_has_prefix_status_and_divergence_lines(self):
        rep = self._report(divergences=['BEZ: pnl SIGN FLIP: BT $-469.48 '
                                        'vs live $+220.83'],
                           n_divergent=1)
        msg = bp.format_telegram(rep)
        assert msg.startswith('[BF PARITY] 2026-08-06: DIVERGE(1)')
        assert 'SIGN FLIP' in msg

    def test_telegram_stale_mentions_cron(self):
        rep = self._report(
            status='BT_STALE (cache ends 2026-08-07)', bt_stale=True,
            live_only=[{'symbol': 'LUNL',
                        'label': bp.LABEL_OFF_UNIVERSE_LEV,
                        'live_status': 'filled', 'live_pnl': -29.44,
                        'live_exit_reason': 'thin_liquidity_reject',
                        'classification': 'LIVE_ONLY'}])
        msg = bp.format_telegram(rep)
        assert 'BT_STALE' in msg
        assert 'LUNL' in msg
        assert 'cron' in msg

    def test_summary_renders_all_classes(self):
        rep = self._report(
            both=[bp.compare_both(
                bt_trade('BEZ', -469.48, 'stop', 8.92),
                live_row('BEZ', 220.83, 'trail_stop', entry=8.89))],
            bt_only=[{'symbol': 'OESX', 'classification': 'BT_ONLY',
                      'label': bp.LABEL_BT_ONLY, 'bt_pnl': -564.25,
                      'bt_exit_reason': 'stop'}],
            live_only=[{'symbol': 'LUNL', 'classification': 'LIVE_ONLY',
                        'label': bp.LABEL_OFF_UNIVERSE,
                        'live_status': 'filled', 'live_pnl': -29.44,
                        'live_exit_reason': 'thin_liquidity_reject'}])
        out = bp.format_summary(rep)
        assert 'BOTH  BEZ' in out and 'BT_ONLY   OESX' in out \
            and 'LIVE_ONLY LUNL' in out
        assert 'SIGN FLIP' in out
