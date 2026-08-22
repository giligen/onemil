"""Winner-stack goldens against REAL cache data (validation gate 3.3).

Requires data/cache.db (post scripts/backfill_daily_bars_gaps.py) and the
regenerated reference (research/stability/regen_winner_stack_reference.py).
Skipped cleanly when the artifacts are absent (CI without the cache).

Goldens (docs/orb_winner_stack_review_aug2026.md):
  * NCNA 2025-08-21 both-hit bar: scale FILLS at +3R, runner lock-stops the
    same bar => +$158-class under C (P0-1 corrected same-bar reading).
  * ATR boundary at exactly 14/15 prior bars (P0-6.1 frozen rule) — BTQ
    2025-10-16 sits at exactly 14 post-backfill: frozen rule fail-opens.
  * The 6 cache-gap symbols (SMST, ARQQ, BTQ, RGTZ, FJET, PS): SMST/ARQQ
    now have real ATR (floors UNBOUND — range_low already tighter, zero
    P&L impact); the other 4 are genuine new listings => no_atr fail-open.
"""
import json
import sqlite3
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'data' / 'cache.db'
BOOK = ROOT / 'analysis_results' / 'orb_bplus_book.csv'
REGEN = ROOT / 'research' / 'stability' / 'winner_stack_regen_reference.json'

pytestmark = pytest.mark.skipif(
    not (CACHE.exists() and BOOK.exists()),
    reason='cache.db / book CSV not available on this node')

from trading.orb_winner_stack import atr14_t1, floored_stop  # noqa: E402


def _atr_frozen(sym, day):
    con = sqlite3.connect(CACHE)
    d = pd.read_sql_query(
        "SELECT bar_date, high, low, close FROM daily_bars "
        "WHERE symbol=? AND bar_date<? ORDER BY bar_date", con,
        params=(sym, day))
    con.close()
    return atr14_t1(d.tail(40)), len(d)


def _load_day_bars(sym, day):
    con = sqlite3.connect(CACHE)
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM intraday_bars_1min WHERE symbol=? AND bar_date=? "
        "ORDER BY timestamp", con, params=(sym, day))
    con.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


class TestNcnaBothHitGolden:
    def test_ncna_scale_fills_runner_lock_stops(self):
        """The 1-of-81 both-hit trade: pipeline walk must bank 40% at +3R
        and lock-stop the runner — ≈ +$158 at book sizing (vs ≈ +$50
        stopped-first — the backwards reading the review corrected)."""
        from study_orb_pipeline_static_lock import simulate_winner_stack
        book = pd.read_csv(BOOK)
        row = book[(book['symbol'] == 'NCNA')
                   & (book['date'] == '2025-08-21')]
        assert len(row) == 1, 'NCNA 2025-08-21 missing from the book'
        row = row.iloc[0]
        bars = _load_day_bars('NCNA', '2025-08-21')
        assert not bars.empty
        et = bars['timestamp'].dt.tz_convert('America/New_York')
        open_ts = bars.loc[(et.dt.hour == 9) & (et.dt.minute == 30),
                           'timestamp'].iloc[0]
        range_end = open_ts + timedelta(minutes=5)
        rbars = bars[(bars['timestamp'] >= open_ts)
                     & (bars['timestamp'] < range_end)]
        rh, rl = float(rbars['high'].max()), float(rbars['low'].min())
        from trading.orb_touchgo_filter import find_breakout_bar_ts
        search = bars[(bars['timestamp'] >= range_end)
                      & (bars['timestamp'] < range_end
                         + timedelta(minutes=60))]
        entry_ts = find_breakout_bar_ts(search, rh)
        entry = float(row['entry_price'])
        atr, _ = _atr_frozen('NCNA', '2025-08-21')
        shares = max(1, int(50_000 / entry))
        exit_p, reason = simulate_winner_stack(
            bars, entry, rh, rl, entry_ts, shares,
            atr14=atr, atr_floor_enabled=True, scale_enabled=True)
        assert reason == 'scale_lock'
        pnl = (exit_p - entry) * shares * float(row['_rp_position']) / 50_000
        # regen reference: +$157.93 (harness-exact frac; pipeline uses the
        # effective int-floored fraction — sub-dollar difference)
        assert pnl == pytest.approx(157.93, abs=1.5)


class TestAtrBoundaryGolden:
    def test_btq_exactly_14_bars_fail_open(self):
        """BTQ 2025-10-16: exactly 14 prior bars post-backfill — the frozen
        rule (>=15) fail-opens where the frontier's 13-TR variant would
        have floored. This is the P0-6.1 boundary, now on real data."""
        atr, n = _atr_frozen('BTQ', '2025-10-16')
        assert n == 14
        assert atr is None

    def test_btq_15th_bar_would_floor(self):
        """One session later BTQ has 15 prior bars -> ATR becomes real."""
        atr, n = _atr_frozen('BTQ', '2025-10-17')
        if n < 15:
            pytest.skip('cache lacks the 15th BTQ bar')
        assert atr is not None and atr > 0


class TestBackfilledSymbolFloors:
    @pytest.mark.parametrize('sym,day,expect_atr', [
        ('SMST', '2025-01-07', True),    # real history existed — backfilled
        ('ARQQ', '2025-01-16', True),    # real history existed — backfilled
        ('BTQ', '2025-10-03', False),    # listed ~2025-09 — genuine no-history
        ('RGTZ', '2025-10-16', False),
        ('FJET', '2025-12-24', False),
        ('PS', '2026-05-01', False),
    ])
    def test_floor_availability(self, sym, day, expect_atr):
        atr, _n = _atr_frozen(sym, day)
        if expect_atr:
            assert atr is not None, (
                f'{sym} {day}: backfill should have restored ATR history')
        else:
            assert atr is None, (
                f'{sym} {day}: genuine new listing must stay fail-open')

    def test_smst_arqq_floors_unbound(self):
        """The two backfilled trades' floors are UNBOUND (range_low already
        tighter) — the regenerated C-point equals the validated artifact
        because the backfill has ZERO P&L effect on the book."""
        book = pd.read_csv(BOOK)
        for sym, day in (('SMST', '2025-01-07'), ('ARQQ', '2025-01-16')):
            row = book[(book['symbol'] == sym) & (book['date'] == day)]
            if row.empty:
                pytest.skip(f'{sym} {day} not in current book')
            atr, _ = _atr_frozen(sym, day)
            entry = float(row.iloc[0]['entry_price'])
            # range_low from the regen per-trade record via physics is not
            # in the book CSV; recompute from bars
            bars = _load_day_bars(sym, day)
            et = bars['timestamp'].dt.tz_convert('America/New_York')
            open_ts = bars.loc[(et.dt.hour == 9) & (et.dt.minute == 30),
                               'timestamp'].iloc[0]
            rbars = bars[(bars['timestamp'] >= open_ts)
                         & (bars['timestamp'] < open_ts
                            + timedelta(minutes=5))]
            rl = float(rbars['low'].min())
            stop, status = floored_stop(rl, entry, atr, 0.25)
            assert status == 'unbound'
            assert stop == pytest.approx(rl)


class TestRegenReferenceArtifact:
    @pytest.mark.skipif(not REGEN.exists(),
                        reason='regen reference not generated yet')
    def test_regen_matches_validated_frontier(self):
        """The regenerated (post-backfill, frozen-ATR-rule) targets must
        reproduce the validated frontier numbers: C ≈ $9,092 / 15 green,
        B ≈ $11,004 / 13 green, base repro ≈ the book."""
        ref = json.load(open(REGEN))
        assert ref['totals']['c'] == pytest.approx(9092.44, abs=5)
        assert ref['greens']['c'] == 15
        assert ref['totals']['b'] == pytest.approx(11003.60, abs=5)
        assert ref['greens']['b'] == 13
        assert ref['totals']['base'] == pytest.approx(
            ref['totals']['base_book'], abs=5)
        assert ref['reason_mix_c'] == {
            'stop': 25, 'scale_sz1': 20, 'tag_bb': 19, 'lock': 8,
            'eod': 7, 'tag_b1': 2}
