"""
Unit tests for trading.exit_reasons.

Two contracts to defend:

  1. **Backward compat**: every string value that has EVER been written to
     trades.exit_reason in production must remain a defined ExitReason
     member. Renaming a member without a backfill would break analytics
     SQL (`GROUP BY exit_reason`) and silently mis-attribute trades.

  2. **No drift**: every inline `exit_reason='...'` (or `'exit_reason':
     '...'`) literal across the BF / ORB / MACD / BT / stop-monitor code
     paths must resolve to an ExitReason member string. Otherwise a typo
     in a new branch (`stop-loss` instead of `stop_loss`) silently writes
     a novel value that's invisible to the daily summary.

The drift test parses each .py file with ast and walks the tree —
catches both kwargs (`exit_reason='x'`) and dict literals
(`{'exit_reason': 'x'}`) without false positives on dunders or comments.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from trading.exit_reasons import (
    ExitReason,
    is_attributed,
    is_historical_only,
    is_known,
    needs_reconcile,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Enum membership + helpers
# ---------------------------------------------------------------------------


class TestEnumMembership:
    """Every observed-in-prod exit_reason must be a defined member."""

    # Snapshot of `SELECT exit_reason, COUNT(*) FROM trades` on
    # data/trades.db as of 2026-06-12. Update when introducing a new
    # value (and only after adding it to the enum).
    OBSERVED_IN_PROD = (
        'stop_loss', 'tag_bb', 'force_close', 'trail_stop',
        'sync_reconcile', 'lock_stop', 'unknown_exit',
        'macd_flip', 'thin_liquidity_reject', 'take_profit',
        'stop_loss_timeout', 'stop_loss_bracket_sl_race',
    )

    @pytest.mark.parametrize('value', OBSERVED_IN_PROD)
    def test_prod_value_is_defined(self, value):
        assert is_known(value), (
            f"{value!r} exists in production DB but is NOT in ExitReason. "
            f"Renaming would break analytics; add a member if needed."
        )

    def test_no_member_string_collision(self):
        """Two members must not share the same .value (would break
        analytics aggregation)."""
        values = [m.value for m in ExitReason]
        assert len(values) == len(set(values)), (
            f"Duplicate member values in ExitReason: {values}"
        )

    def test_all_members_are_lowercase_snake(self):
        """All values are lowercase snake_case (matches existing DB rows
        and the rest of the codebase)."""
        for m in ExitReason:
            assert m.value == m.value.lower(), m.value
            assert ' ' not in m.value, m.value


class TestCategorizationHelpers:
    def test_is_known_returns_false_for_none_and_empty(self):
        assert is_known(None) is False
        assert is_known('') is False
        assert is_known('made_up_reason_2027') is False

    def test_is_attributed_excludes_leak_paths(self):
        assert is_attributed(ExitReason.STOP_LOSS.value) is True
        assert is_attributed(ExitReason.TRAIL_STOP.value) is True
        assert is_attributed(ExitReason.FORCE_CLOSE.value) is True
        # Leak paths — analytics must NOT count these as clean attributions
        assert is_attributed(ExitReason.UNKNOWN_EXIT.value) is False
        assert is_attributed(ExitReason.STOP_LOSS_UNCONFIRMED.value) is False
        assert is_attributed(ExitReason.SYNC_RECONCILE.value) is False
        assert is_attributed(ExitReason.STOP_LOSS_TIMEOUT.value) is False

    def test_needs_reconcile_flags_leaks_and_history(self):
        assert needs_reconcile(ExitReason.UNKNOWN_EXIT.value) is True
        assert needs_reconcile(ExitReason.STOP_LOSS_UNCONFIRMED.value) is True
        assert needs_reconcile(ExitReason.SYNC_RECONCILE.value) is True
        assert needs_reconcile(ExitReason.STOP_LOSS_TIMEOUT.value) is True
        assert needs_reconcile(ExitReason.STOP_LOSS.value) is False
        assert needs_reconcile(ExitReason.FORCE_CLOSE.value) is False
        assert needs_reconcile(None) is False
        assert needs_reconcile('bogus') is False

    def test_historical_only_is_a_subset_of_needs_reconcile(self):
        for m in ExitReason:
            if is_historical_only(m.value):
                assert needs_reconcile(m.value), (
                    f"{m.value}: historical implies needs_reconcile"
                )

    def test_str_enum_subclass_str_compat(self):
        """A member should be usable wherever a str is — important for
        legacy assertions like `exit_reason == 'stop_loss'` to keep
        passing after the refactor."""
        assert ExitReason.STOP_LOSS == 'stop_loss'
        assert {'exit_reason': ExitReason.STOP_LOSS.value} == {
            'exit_reason': 'stop_loss'
        }


# ---------------------------------------------------------------------------
# AST drift prevention
# ---------------------------------------------------------------------------


# Files that emit exit_reason at runtime. Adding a new emitter file
# without adding it here is a deliberate choice — but the new file
# should still go through ExitReason; this list keeps the scan finite.
EXIT_REASON_EMITTER_FILES = (
    'trading/orb_engine.py',
    'trading/trading_engine.py',
    'trading/stop_monitor.py',
    'trading/macd_wave_engine.py',
    'backtest.py',
)


def _find_exit_reason_string_literals(path: Path) -> list[tuple[int, str]]:
    """Walk the AST of `path` and return [(lineno, value), ...] for
    every place where an inline string literal is assigned to an
    `exit_reason` kwarg or dict key. Ignores:
      * `partial_exit_reason` (separate column, different taxonomy)
      * docstrings + comments (AST doesn't surface them at these nodes)
      * variable assignment like `exit_reason = ExitReason.X.value`
        (those are ast.Attribute, not ast.Constant)
    """
    tree = ast.parse(path.read_text())
    findings: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        # kwarg form: foo(exit_reason='stop_loss')
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if (kw.arg == 'exit_reason'
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)):
                    findings.append((kw.value.lineno, kw.value.value))

        # dict literal form: {'exit_reason': 'stop_loss'}
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant)
                        and k.value == 'exit_reason'
                        and isinstance(v, ast.Constant)
                        and isinstance(v.value, str)):
                    findings.append((v.lineno, v.value))

    return findings


class TestNoStringLiteralDrift:
    """Every inline exit_reason='...' or 'exit_reason': '...' literal in
    emitter files must match an ExitReason member. If a new contributor
    adds a typo'd literal (`'stop-loss'` or `'stoploss'`), this test
    catches it before it lands in production DB."""

    @pytest.mark.parametrize('rel', EXIT_REASON_EMITTER_FILES)
    def test_emitter_file_has_no_unknown_literal(self, rel):
        path = REPO_ROOT / rel
        assert path.exists(), f"missing emitter file: {rel}"
        findings = _find_exit_reason_string_literals(path)
        unknown = [(ln, v) for ln, v in findings if not is_known(v)]
        assert not unknown, (
            f"{rel} contains inline exit_reason literal(s) not defined in "
            f"trading/exit_reasons.py::ExitReason. Add the member there "
            f"first (or use ExitReason.X.value at the call site).\n"
            f"  Offending lines: {unknown}"
        )


# ---------------------------------------------------------------------------
# D3 FIX 6 — the exit_branch / exit_reason split
# ---------------------------------------------------------------------------
#
# `research/fuckup_audit/D3_exec/REPORT.md` §M6: `_execute_stop_exit`
# overwrote `exit_reason` with 'stop_loss_market_fallback' for ANY watch
# that escalated to a market close, so the bucket contained FJET
# 2026-06-12 (a trailing-stop exit on a WINNER, +$122.84), SMCX (another
# trail) and HCAI 2026-09-11 (an ignition EOD force-flat) alongside three
# real stops. Every number that grouped on exit_reason was wrong.


class TestExitBranchCatalog:
    """The new `trades.exit_branch` vocabulary."""

    def test_the_four_report_branches_exist(self):
        from trading.exit_reasons import ExitBranch
        for expected in ('limit', 'market_fallback', 'sl_leg_race',
                         'last_resort'):
            assert expected in {b.value for b in ExitBranch}

    def test_sl_leg_branch_exists_for_fix_1(self):
        """FIX 1's designed exit: the broker SL leg is repriced and fills
        it, so there is no naked window at all."""
        from trading.exit_reasons import ExitBranch
        assert ExitBranch.SL_LEG.value == 'sl_leg'

    def test_no_branch_value_collision(self):
        from trading.exit_reasons import ExitBranch
        values = [b.value for b in ExitBranch]
        assert len(values) == len(set(values))

    def test_branch_values_are_lowercase_snake(self):
        from trading.exit_reasons import ExitBranch
        for b in ExitBranch:
            assert b.value == b.value.lower()
            assert ' ' not in b.value

    def test_branch_and_reason_namespaces_are_disjoint(self):
        """A branch string must never be mistakable for a reason string —
        they live in different columns and mean different things."""
        from trading.exit_reasons import ExitBranch
        assert not ({b.value for b in ExitBranch}
                    & {r.value for r in ExitReason})

    def test_is_known_branch(self):
        from trading.exit_reasons import ExitBranch, is_known_branch
        for b in ExitBranch:
            assert is_known_branch(b.value) is True
        assert is_known_branch('market_close') is False   # internal tag
        assert is_known_branch('stop_loss') is False      # that's a reason

    def test_null_branch_is_not_drift(self):
        """Historic rows (and non-StopMonitor writers) carry NULL —
        legitimate, not a violation to alert on."""
        from trading.exit_reasons import is_known_branch
        assert is_known_branch(None) is False
        assert is_known_branch('') is False


class TestEscalationTagMapping:
    """StopMonitor's internal BRANCH_* tags must each map onto a defined
    ExitBranch — otherwise an escalation writes a branch nothing can
    group on."""

    def test_every_escalation_tag_maps_to_a_known_branch(self):
        from trading.exit_reasons import is_known_branch
        from trading.stop_monitor import StopMonitor
        tags = (StopMonitor.BRANCH_LIMIT_RACE, StopMonitor.BRANCH_MARKET_CLOSE,
                StopMonitor.BRANCH_SL_LEG_RACE, StopMonitor.BRANCH_LAST_RESORT)
        for tag in tags:
            mapped = StopMonitor._EXIT_BRANCH_BY_ESCALATION.get(tag)
            assert is_known_branch(mapped), f"{tag} -> {mapped!r}"

    def test_limit_race_is_a_limit_fill_not_a_fallback(self):
        """The limit filled while we were cancelling it — the limit did
        the work, so the branch histogram must count it as `limit`."""
        from trading.exit_reasons import ExitBranch
        from trading.stop_monitor import StopMonitor
        assert (StopMonitor._EXIT_BRANCH_BY_ESCALATION[
            StopMonitor.BRANCH_LIMIT_RACE] == ExitBranch.LIMIT.value)


class TestBuildExitUpdateCarriesBranch:
    """Every engine (BF / ORB / MACD / ignition) drains through
    build_exit_update, so the branch reaches the DB by construction."""

    def _ev(self, **kw):
        from trading.stop_monitor import StopExitEvent
        d = dict(symbol='X', stop_price=10.0, exit_price=9.5, shares=100,
                 order_id='id-1', exit_reason='trail_stop')
        d.update(kw)
        return StopExitEvent(**d)

    def test_branch_written_when_set(self):
        from trading.exit_reasons import ExitBranch
        from trading.stop_monitor import build_exit_update
        out = build_exit_update(
            self._ev(exit_branch=ExitBranch.MARKET_FALLBACK.value))
        assert out['exit_branch'] == 'market_fallback'
        assert out['exit_reason'] == 'trail_stop'

    def test_branch_is_none_when_unset(self):
        from trading.stop_monitor import build_exit_update
        out = build_exit_update(self._ev())
        assert out['exit_branch'] is None

    def test_unconfirmed_payload_still_carries_branch(self):
        """The forensics value of the branch is highest precisely on the
        rows we could not confirm."""
        from trading.exit_reasons import ExitBranch
        from trading.stop_monitor import build_exit_update
        out = build_exit_update(self._ev(
            confirmed=False, exit_reason='stop_loss_unconfirmed',
            exit_branch=ExitBranch.LAST_RESORT.value))
        assert out['exit_branch'] == 'last_resort'
        assert out['order_status'] == 'exit_pending_verification'
        assert 'exit_price' not in out

    def test_default_event_has_empty_branch(self):
        ev = self._ev()
        assert ev.exit_branch == ''


class TestExitBranchColumnMigration:
    """Migration 15 — idempotent ADD COLUMN, historic strings untouched."""

    def _db(self, tmp_path):
        from persistence.database import Database
        p = str(tmp_path / 'trades.db')
        return Database(db_path=p, cache_path=str(tmp_path / 'cache.db'),
                        trades_path=p)

    def test_column_exists_after_open(self, tmp_path):
        db = self._db(tmp_path)
        cols = [r[1] for r in db._trades_conn.execute(
            'PRAGMA table_info(trades)').fetchall()]
        assert 'exit_branch' in cols

    def test_migration_is_idempotent(self, tmp_path):
        db = self._db(tmp_path)
        db._migrate()          # second run must not raise / duplicate
        db._migrate()
        cols = [r[1] for r in db._trades_conn.execute(
            'PRAGMA table_info(trades)').fetchall()]
        assert cols.count('exit_branch') == 1

    def test_historic_rows_keep_their_reason_and_get_null_branch(self, tmp_path):
        """A pre-split row must read back byte-identical, with a NULL
        branch — no backfill, no rewrite of a load-bearing string."""
        db = self._db(tmp_path)
        db._trades_conn.execute("""
            INSERT INTO trades (
                trade_date, symbol, side, entry_price, stop_loss_price,
                take_profit_price, shares, risk_per_share, total_risk,
                risk_reward_ratio, order_status, fill_price, exit_price,
                exit_reason, strategy, created_at, updated_at
            ) VALUES ('2026-06-12', 'FJET', 'buy', 6.52, 6.52, 8, 2415, 0.2,
                      500, 2, 'closed', 6.52, 6.70,
                      'stop_loss_market_fallback', 'bull_flag',
                      '2026-06-12T00:00:00', '2026-06-12T00:00:00')
        """)
        db._trades_conn.commit()
        db._migrate()
        row = db._trades_conn.execute(
            "SELECT exit_reason, exit_branch FROM trades "
            "WHERE symbol='FJET'").fetchone()
        assert row[0] == 'stop_loss_market_fallback'
        assert row[1] is None

    def test_update_trade_writes_the_branch(self, tmp_path):
        db = self._db(tmp_path)
        db._trades_conn.execute("""
            INSERT INTO trades (
                trade_date, symbol, side, entry_price, stop_loss_price,
                take_profit_price, shares, risk_per_share, total_risk,
                risk_reward_ratio, order_status, fill_price, strategy,
                created_at, updated_at
            ) VALUES ('2026-09-17', 'TST', 'buy', 10, 9, 12, 100, 1, 100, 2,
                      'filled', 10.0, 'bull_flag',
                      '2026-09-17T00:00:00', '2026-09-17T00:00:00')
        """)
        db._trades_conn.commit()
        tid = db._trades_conn.execute(
            "SELECT id FROM trades WHERE symbol='TST'").fetchone()[0]
        db.update_trade(tid, {'exit_reason': 'trail_stop',
                              'exit_branch': 'market_fallback'})
        row = db._trades_conn.execute(
            "SELECT exit_reason, exit_branch FROM trades "
            "WHERE id=?", (tid,)).fetchone()
        assert row[0] == 'trail_stop'
        assert row[1] == 'market_fallback'


# ---------------------------------------------------------------------------
# FIX 6 — the two events the report names, replayed through _execute_stop_exit
# ---------------------------------------------------------------------------

import asyncio      # noqa: E402
import time as _time  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from data_sources.alpaca_client import AlpacaClient  # noqa: E402
from trading.exit_reasons import ExitBranch  # noqa: E402
from trading.stop_monitor import StopMonitor  # noqa: E402


@pytest.fixture
def escalating_client():
    """An Alpaca client whose limit sell NEVER fills and whose market
    close does — the exact shape of FJET / SMCX / HCAI."""
    c = MagicMock(spec=AlpacaClient)
    c.cancel_order.return_value = True
    c.submit_limit_sell_order.return_value = {'id': 'lmt-1', 'status': 'new'}
    c.close_position.return_value = {'id': 'mkt-1', 'status': 'accepted'}
    c.get_open_positions.return_value = []
    c.trading_client = MagicMock()
    c.trading_client.get_orders.return_value = []

    def _get_order(order_id):
        if order_id == 'lmt-1':
            return {'id': order_id, 'status': 'new',
                    'filled_avg_price': None, 'filled_qty': 0}
        return {'id': order_id, 'status': 'filled',
                'filled_avg_price': 6.70, 'filled_qty': 2415}
    c.get_order.side_effect = _get_order
    return c


@pytest.fixture
def escalating_monitor(escalating_client):
    m = StopMonitor(api_key='k', api_secret='s',
                    alpaca_client=escalating_client)
    m._STOP_EXIT_FILL_TIMEOUT_S = 0.2
    m._MARKET_CLOSE_FILL_TIMEOUT_S = 0.2
    m._STOP_EXIT_POLL_INTERVAL_S = 0.05
    return m


def _armed_watch(monitor, symbol, stop, shares, bid, ask):
    monitor.add_watch(symbol, stop, shares, 'tp-1', 'sl-1')
    with monitor._watch_lock:
        w = monitor._watches[symbol]
        w.latest_bid, w.latest_ask = bid, ask
        w.latest_bid_size, w.latest_ask_size = 200, 200
        w.latest_quote_ts = _time.time()
    return w


class TestEscalationPreservesReason:

    def test_escalation_preserves_trail_reason(self, escalating_monitor,
                                               escalating_client):
        """FJET 2026-06-12: a TRAILING stop fired on a WINNER (trigger
        $6.75 vs stop $6.5207, booked P&L +$122.84); the limit hung and
        close_position finished it. Pre-fix the row said
        `stop_loss_market_fallback`, so a winning trail exit was counted
        in the stop-loss book."""
        m, w = escalating_monitor, None
        w = _armed_watch(m, 'FJET', 6.5207, 2415, 6.75, 6.77)
        asyncio.run(m._execute_stop_exit(
            'FJET', 6.75, w, exit_reason='trail_stop'))

        events = m.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == 'trail_stop'
        assert events[0].exit_branch == ExitBranch.MARKET_FALLBACK.value
        assert escalating_client.close_position.call_count == 1

    def test_force_exit_reason_survives_escalation(self, escalating_monitor):
        """HCAI 2026-09-11: the journal says
        `force_exit(HCAI, stage_force_flat)` — an ignition EOD flat, not a
        stop at all. It was booked `stop_loss_market_fallback`."""
        m = escalating_monitor
        w = _armed_watch(m, 'HCAI', 2.1002, 421, 2.17, 2.21)
        asyncio.run(m._execute_stop_exit(
            'HCAI', 2.1002, w, exit_reason='stage_force_flat'))

        events = m.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == 'stage_force_flat'
        assert events[0].exit_branch == ExitBranch.MARKET_FALLBACK.value

    def test_clean_limit_fill_is_branch_limit(self, escalating_monitor,
                                              escalating_client):
        """The intended path — reason preserved, branch `limit`."""
        escalating_client.get_order.side_effect = lambda oid: {
            'id': oid, 'status': 'filled',
            'filled_avg_price': 4.66, 'filled_qty': 500}
        m = escalating_monitor
        w = _armed_watch(m, 'RBNE', 4.685, 500, 4.68, 4.74)
        asyncio.run(m._execute_stop_exit('RBNE', 4.68, w,
                                         exit_reason='stop_loss'))
        events = m.drain_exit_events()
        assert events[0].exit_reason == 'stop_loss'
        assert events[0].exit_branch == ExitBranch.LIMIT.value
        assert escalating_client.close_position.call_count == 0

    def test_last_resort_branch_is_unconfirmed(self, escalating_monitor,
                                               escalating_client):
        """Nothing confirmed a fill → branch last_resort, confirmed False,
        and the reason stays the reconciler's signal."""
        escalating_client.get_order.side_effect = lambda oid: {
            'id': oid, 'status': 'new',
            'filled_avg_price': None, 'filled_qty': 0}
        m = escalating_monitor
        w = _armed_watch(m, 'PLYX', 4.29, 500, 4.26, 4.27)
        asyncio.run(m._execute_stop_exit('PLYX', 4.25, w,
                                         exit_reason='trail_stop'))
        ev = m.drain_exit_events()[0]
        assert ev.exit_branch == ExitBranch.LAST_RESORT.value
        assert ev.exit_reason == 'stop_loss_unconfirmed'
        assert ev.confirmed is False

    def test_sl_leg_race_branch(self, escalating_monitor, escalating_client):
        """The broker's SL leg executed the sale — a different exit, not
        just a different route; the reason records that."""
        def _get_order(oid):
            if oid == 'sl-1':
                return {'id': 'sl-1', 'status': 'filled',
                        'filled_avg_price': 2.70, 'filled_qty': 500}
            return {'id': oid, 'status': 'new',
                    'filled_avg_price': None, 'filled_qty': 0}
        escalating_client.get_order.side_effect = _get_order
        escalating_client.close_position.side_effect = Exception(
            '40410000: position not found for PLYX')
        m = escalating_monitor
        w = _armed_watch(m, 'PLYX', 3.00, 500, 2.99, 3.00)
        asyncio.run(m._execute_stop_exit('PLYX', 2.98, w,
                                         exit_reason='trail_stop'))
        ev = m.drain_exit_events()[0]
        assert ev.exit_branch == ExitBranch.SL_LEG_RACE.value
        assert ev.exit_reason == 'stop_loss_bracket_sl_race'
        assert ev.order_id == 'sl-1'
