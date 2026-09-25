"""docs/hod_live_resting_orders_spec_20260925.md item 5 — tests for the LIVE resting stop-limit order path
(entry_mode='resting_stop_limit', dry_run=False): `HodBreakEngine._arm_live_order/_cancel_live_order/
_sweep_live_cutoffs/_poll_live_fills/_on_live_fill/_append_live_parity_row/_persist_live_orders/
_reconcile_live_orders_on_boot`, `trading.hod_break.resting_order_qty`, and the `hod_break`-pausable guardrail
wiring (trading/live_guardrail.py, scripts/guardrail.py). All collaborators are MagicMock(spec=...); the DRY
tape side (_evaluate_resting/_on_trade_print) is untouched by this file — see tests/test_hod_resting_entry.py.
"""
import csv
import json
from datetime import datetime, timezone

import pytest

from trading.hod_break import resting_order_qty
from trading.hod_break_engine import Candidate, HodBreakEngine
from trading import live_guardrail as gr
from tests.test_hod_break_engine import cfg, bars_df, admit
from tests.test_hod_resting_entry import tape
from tests.test_live_guardrail import _stats


def live_engine(alpaca, db, sm, stream, tmp_path, dry_run=False, **over):
    """A resting_stop_limit engine (dry_run=False by default — the REAL live-order path is active)."""
    c = cfg(dry_run=dry_run, entry_mode='resting_stop_limit', **over)
    c.setdefault('live_parity_ledger_path', str(tmp_path / 'hod_live_parity_ledger.csv'))
    c.setdefault('live_orders_state_path', str(tmp_path / 'hod_live_resting_orders_state.json'))
    c.setdefault('dry_ledger_path', str(tmp_path / 'hod_dry_entry_ledger.csv'))
    e = HodBreakEngine(alpaca, db, sm, cfg=c, order_stream=stream)
    e._roll_session()
    # Same live_since back-date as tests/test_hod_resting_entry.py's resting_engine(): these tests feed bars
    # straight to _ingest_bars, never through _on_bar_close, so without this every synthetic bar is (correctly,
    # 2026-09-25 fix) treated as pre-live backfill and skipped for arming.
    e.live_since = e._bar_close_et(0)
    return e


def read_ledger(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


# a stable arm dict for direct method-level tests; bar_volume 1e6 keeps the 5%-of-volume cap from binding
# unless a test wants it to (matches tape()'s consolidation bar volume 3000, used in the qty tests below).
BIG_VOL_ARM = dict(level=11.0, trigger=11.01, limit=11.0165, stop=10.6, idx=6, arm_ts='2026-09-28T09:36:00', bar_volume=1_000_000.0)


# --------------------------------------------------------------------------------------------- item 1: qty formula
class TestRestingOrderQty:
    def test_uncapped_matches_the_risk_over_r_formula(self):
        arm = dict(trigger=11.01, stop=10.6, bar_volume=1_000_000.0)
        assert resting_order_qty(100.0, arm) == int(100.0 / 0.41)   # 243, far under 5% of 1e6

    def test_capped_at_5pct_of_the_prior_bars_volume(self):
        arm = dict(trigger=11.01, stop=10.6, bar_volume=3000.0)     # tape()'s consolidation-bar volume
        assert resting_order_qty(100.0, arm) == 150                  # 5% of 3000 binds under 243


# --------------------------------------------------------------------------------------------- item 1: arm -> ONE order
class TestArmPlacesLiveOrder:
    def test_arm_places_exactly_one_stop_limit_with_right_prices_qty_tif_and_coid(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, risk_usd=100.0)
        admit(e)
        e._ingest_bars('ABC', bars_df(tape()[:7]))                   # bars 0..6: consolidation completes at j=6, no cross yet
        assert hod_live_alpaca.submit_stop_limit_order.call_count == 1
        kw = hod_live_alpaca.submit_stop_limit_order.call_args.kwargs
        assert kw['symbol'] == 'ABC' and kw['side'] == 'buy'
        assert kw['stop_price'] == pytest.approx(11.01)               # level 11.00 + 0.01
        assert kw['limit_price'] == pytest.approx(11.0 * 1.0015)
        assert kw['qty'] == 150                                        # 5%-of-bar-volume cap binds (bar_volume 3000)
        assert kw['client_order_id'].startswith('hod-rest-ABC-')
        # time-in-force is DAY inside AlpacaClient.submit_stop_limit_order itself — the engine passes no tif kwarg
        assert 'time_in_force' not in kw
        lo = e.candidates['ABC'].live_order
        assert lo['order_id'] == f"broker-{kw['client_order_id']}"


# --------------------------------------------------------------------------------------------- item 2: level change
class TestLevelChange:
    def test_level_change_cancels_and_places_a_brand_new_order_never_replace(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        admit(e)
        cand = e.candidates['ABC']
        e._arm_live_order(cand, dict(BIG_VOL_ARM))
        first_id = cand.live_order['order_id']
        e._arm_live_order(cand, dict(BIG_VOL_ARM, level=11.5, trigger=11.51, limit=11.5173, stop=10.9))
        assert hod_live_alpaca.cancel_order.call_args_list[0].args[0] == first_id
        assert hod_live_alpaca.submit_stop_limit_order.call_count == 2
        assert cand.live_order['order_id'] != first_id
        assert cand.live_order['trigger'] == pytest.approx(11.51)
        assert not hod_live_alpaca.replace_order_stop_price.called
        assert not hod_live_alpaca.replace_order_limit_price.called
        assert not hod_live_alpaca.replace_order_qty.called

    def test_unchanged_level_does_not_re_place(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        admit(e)
        cand = e.candidates['ABC']
        e._arm_live_order(cand, dict(BIG_VOL_ARM))
        e._arm_live_order(cand, dict(BIG_VOL_ARM))                    # same trigger/limit
        assert hod_live_alpaca.submit_stop_limit_order.call_count == 1
        assert not hod_live_alpaca.cancel_order.called


# --------------------------------------------------------------------------------------------- item 3: disarm cancels
class TestDisarmCancels:
    def test_disarm_cancels_and_writes_a_parity_row(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        admit(e)
        cand = e.candidates['ABC']
        e._arm_live_order(cand, dict(BIG_VOL_ARM))
        order_id = cand.live_order['order_id']
        e._cancel_live_order(cand, 'arm_lost')
        hod_live_alpaca.cancel_order.assert_called_once_with(order_id)
        assert cand.live_order is None and 'ABC' not in e._live_cap_slots
        rows = read_ledger(e.live_parity_ledger_path)
        assert len(rows) == 1 and rows[0]['broker_status'] == 'cancelled' and rows[0]['reason'] == 'arm_lost'


# --------------------------------------------------------------------------------------------- item 4: fill registers StopMonitor
class TestFillRegistersStopMonitor:
    def test_fill_and_partial_top_up_update_the_cumulative_qty(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        admit(e)
        cand = e.candidates['ABC']
        e._arm_live_order(cand, dict(BIG_VOL_ARM))
        coid = cand.live_order['coid']

        hod_live_stream.snapshot_by_client_prefix.return_value = {
            coid: {'status': 'partially_filled', 'filled_qty': 50, 'filled_avg_price': 11.02, 'client_order_id': coid}}
        e._poll_live_fills()
        hod_live_sm.add_watch.assert_called_once()
        k1 = hod_live_sm.add_watch.call_args.kwargs
        assert k1['shares'] == 50 and k1['stop_price'] == pytest.approx(10.6) and k1['strategy'] == 'hod_break'
        assert cand.live_order['booked_qty'] == 50
        assert cand.live_filled is False                              # still resting for the remainder

        hod_live_stream.snapshot_by_client_prefix.return_value = {
            coid: {'status': 'filled', 'filled_qty': 150, 'filled_avg_price': 11.02, 'client_order_id': coid}}
        e._poll_live_fills()
        assert hod_live_sm.add_watch.call_count == 2
        k2 = hod_live_sm.add_watch.call_args.kwargs
        assert k2['shares'] == 150                                     # cumulative, not the increment
        assert hod_live_alpaca.cancel_order.call_count >= 2             # stale TP+SL legs from the partial were cancelled
        assert cand.live_filled is True and cand.live_order is None
        assert 'ABC' in e.entered_today


# --------------------------------------------------------------------------------------- item 4b: OCO safety net (2026-09-25 VECO fix)
class TestOcoSafetyNet:
    def test_fill_places_one_oco_order_with_right_prices_qty_and_stores_both_leg_ids(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        admit(e)
        cand = e.candidates['ABC']
        e._arm_live_order(cand, dict(BIG_VOL_ARM))
        coid = cand.live_order['coid']
        hod_live_stream.snapshot_by_client_prefix.return_value = {
            coid: {'status': 'filled', 'filled_qty': 150, 'filled_avg_price': 11.02, 'client_order_id': coid}}
        e._poll_live_fills()
        assert hod_live_alpaca.submit_oco_sell_order.call_count == 1
        assert not hod_live_alpaca.submit_limit_sell_order.called
        assert not hod_live_alpaca.submit_stop_sell_order.called
        kw = hod_live_alpaca.submit_oco_sell_order.call_args.kwargs
        assert kw['symbol'] == 'ABC' and kw['qty'] == 150
        assert kw['limit_price'] == pytest.approx(11.02 + e.params.target_r * (11.02 - BIG_VOL_ARM['stop']))
        assert kw['stop_price'] == pytest.approx(round(BIG_VOL_ARM['stop'] * (1.0 - e.SAFETY_NET_PCT), 2))
        watch = hod_live_sm.add_watch.call_args.kwargs
        assert watch['tp_leg_id'] == 'tp-1' and watch['sl_leg_id'] == 'sl-1'

    def test_oco_submit_failure_falls_back_to_stop_only_safety_net(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        admit(e)
        cand = e.candidates['ABC']
        e._arm_live_order(cand, dict(BIG_VOL_ARM))
        coid = cand.live_order['coid']
        hod_live_alpaca.submit_oco_sell_order.side_effect = Exception('order_class oco not supported')
        hod_live_stream.snapshot_by_client_prefix.return_value = {
            coid: {'status': 'filled', 'filled_qty': 150, 'filled_avg_price': 11.02, 'client_order_id': coid}}
        e._poll_live_fills()
        hod_live_alpaca.submit_stop_sell_order.assert_called_once()
        sk = hod_live_alpaca.submit_stop_sell_order.call_args.kwargs
        assert sk['symbol'] == 'ABC' and sk['qty'] == 150
        assert sk['stop_price'] == pytest.approx(round(BIG_VOL_ARM['stop'] * (1.0 - e.SAFETY_NET_PCT), 2))
        watch = hod_live_sm.add_watch.call_args.kwargs
        assert watch['sl_leg_id'] == 'sl-1' and watch['tp_leg_id'] == ''    # no TP leg — stop-only fallback
        # position is never left unprotected: the fallback stop is always placed
        assert cand.live_order is None and 'ABC' in e.entered_today

    def test_partial_fill_top_up_cancels_the_old_oco_and_places_a_new_one_at_cumulative_qty(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        admit(e)
        cand = e.candidates['ABC']
        e._arm_live_order(cand, dict(BIG_VOL_ARM))
        coid = cand.live_order['coid']
        hod_live_stream.snapshot_by_client_prefix.return_value = {
            coid: {'status': 'partially_filled', 'filled_qty': 50, 'filled_avg_price': 11.02, 'client_order_id': coid}}
        e._poll_live_fills()
        assert hod_live_alpaca.submit_oco_sell_order.call_count == 1
        assert not hod_live_alpaca.cancel_order.called                 # nothing to cancel on the first fill

        hod_live_alpaca.submit_oco_sell_order.return_value = {
            'id': 'oco-2', 'status': 'accepted',
            'legs': [{'id': 'tp-2', 'type': 'limit'}, {'id': 'sl-2', 'type': 'stop'}]}
        hod_live_stream.snapshot_by_client_prefix.return_value = {
            coid: {'status': 'filled', 'filled_qty': 150, 'filled_avg_price': 11.02, 'client_order_id': coid}}
        e._poll_live_fills()
        assert hod_live_alpaca.submit_oco_sell_order.call_count == 2
        assert hod_live_alpaca.cancel_order.call_args_list[0].args[0] == 'tp-1'   # stale OCO legs cancelled first
        assert hod_live_alpaca.cancel_order.call_args_list[1].args[0] == 'sl-1'
        second_kw = hod_live_alpaca.submit_oco_sell_order.call_args.kwargs
        assert second_kw['qty'] == 150                                  # cumulative, not the increment
        watch = hod_live_sm.add_watch.call_args.kwargs
        assert watch['tp_leg_id'] == 'tp-2' and watch['sl_leg_id'] == 'sl-2'


# --------------------------------------------------------------------------------------------- item 5: one fill per symbol-day
class TestOneFillPerSymbolDay:
    def test_second_cross_after_a_live_fill_places_nothing(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        admit(e)
        cand = e.candidates['ABC']
        cand.live_filled = True                                       # broker already confirmed a fill today
        from tests.test_hod_break import bars as bars_arrays
        o, h, l, c, v, m = bars_arrays(tape() + [(11.05, 11.3, 11.0, 11.2, 2000)])
        e._evaluate_resting(cand, o, h, l, v, m, e.params)
        assert not hod_live_alpaca.submit_stop_limit_order.called
        assert cand.live_order is None


# --------------------------------------------------------------------------------------------- item 6: resting cap is
# SEPARATE from max_concurrent/max_per_day (9/25 fix: with ~30 armed names only the first 2 to arm got a real
# order when a resting slot also spent a max_concurrent slot — trading/hod_break_engine.py::_arm_live_order).
class TestRestingCapSeparateFromFillCaps:
    def test_12_armed_names_get_resting_orders_the_13th_is_tape_only(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path,
                        params=dict(cfg()['params'], max_concurrent=20, max_per_day=20, max_resting=12))
        admit(e)
        cands = [e.candidates['ABC']] + [Candidate(symbol=f'S{i}', day_open=10.0, adv20=1_000_000.0) for i in range(1, 13)]
        for c in cands[1:]:
            e.candidates[c.symbol] = c
        for c in cands:
            e._arm_live_order(c, dict(BIG_VOL_ARM))
        assert hod_live_alpaca.submit_stop_limit_order.call_count == 12
        assert sum(1 for c in cands if c.live_order is not None) == 12
        assert sum(1 for c in cands if c.live_order is None) == 1           # the 13th stays tape-only

    def test_low_max_concurrent_or_max_per_day_does_NOT_block_a_second_resting_order(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        # max_concurrent=1 and max_per_day=1 with ZERO fills today: both FILLED-position caps read 0, so
        # a second resting order is still allowed — only max_resting governs resting-order count now.
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path,
                        params=dict(cfg()['params'], max_concurrent=1, max_per_day=1, max_resting=12))
        admit(e)
        cand1 = e.candidates['ABC']
        cand2 = Candidate(symbol='DEF', day_open=10.0, adv20=1_000_000.0)
        e.candidates['DEF'] = cand2
        e._arm_live_order(cand1, dict(BIG_VOL_ARM))
        e._arm_live_order(cand2, dict(BIG_VOL_ARM, level=12.0, trigger=12.01, limit=12.0173, stop=11.6))
        assert hod_live_alpaca.submit_stop_limit_order.call_count == 2
        assert cand1.live_order is not None and cand2.live_order is not None


# --------------------------------------------------------------------------------------------- item (2): a fill that
# reaches a FILLED-position cap sweeps every other resting order immediately.
class TestFillCapCancelsResting:
    def test_fill_reaching_max_concurrent_cancels_the_other_resting_orders(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path,
                        params=dict(cfg()['params'], max_concurrent=1, max_per_day=8, max_resting=12))
        admit(e)
        cand1 = e.candidates['ABC']
        cand2 = Candidate(symbol='DEF', day_open=10.0, adv20=1_000_000.0)
        e.candidates['DEF'] = cand2
        e._arm_live_order(cand1, dict(BIG_VOL_ARM))
        e._arm_live_order(cand2, dict(BIG_VOL_ARM, level=12.0, trigger=12.01, limit=12.0173, stop=11.6))
        assert hod_live_alpaca.submit_stop_limit_order.call_count == 2      # both armed — resting cap is separate

        coid = cand1.live_order['coid']
        hod_live_db.get_open_trades.return_value = [{'symbol': 'ABC'}]      # DB now shows ABC open, post-fill
        hod_live_stream.snapshot_by_client_prefix.return_value = {
            coid: {'status': 'filled', 'filled_qty': 150, 'filled_avg_price': 11.02, 'client_order_id': coid}}
        with caplog.at_level('INFO'):
            e._poll_live_fills()
        assert cand2.live_order is None                                      # swept immediately
        assert any('fill cap reached' in r.message and 'cancelled 1 resting orders' in r.message for r in caplog.records)

    def test_fill_reaching_max_per_day_cancels_resting_and_stops_new_arms(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path,
                        params=dict(cfg()['params'], max_concurrent=8, max_per_day=1, max_resting=12))
        admit(e)
        cand1 = e.candidates['ABC']
        cand2 = Candidate(symbol='DEF', day_open=10.0, adv20=1_000_000.0)
        e.candidates['DEF'] = cand2
        e._arm_live_order(cand1, dict(BIG_VOL_ARM))
        e._arm_live_order(cand2, dict(BIG_VOL_ARM, level=12.0, trigger=12.01, limit=12.0173, stop=11.6))
        assert hod_live_alpaca.submit_stop_limit_order.call_count == 2

        coid = cand1.live_order['coid']
        hod_live_stream.snapshot_by_client_prefix.return_value = {
            coid: {'status': 'filled', 'filled_qty': 150, 'filled_avg_price': 11.02, 'client_order_id': coid}}
        with caplog.at_level('INFO'):
            e._poll_live_fills()
        assert cand2.live_order is None
        assert any('day cap reached' in r.message for r in caplog.records)

        cand3 = Candidate(symbol='GHI', day_open=10.0, adv20=1_000_000.0)
        e.candidates['GHI'] = cand3
        e._arm_live_order(cand3, dict(BIG_VOL_ARM, level=13.0, trigger=13.01, limit=13.0195, stop=12.6))
        assert cand3.live_order is None                                      # per-day cap now blocks all new arming
        assert hod_live_alpaca.submit_stop_limit_order.call_count == 2       # no third order placed


# --------------------------------------------------------------------------------------------- item (3): the noisy
# per-bar 'LIVE cap reached' line is deduped to once per symbol per session.
class TestCapLogDedup:
    def test_cap_reached_logs_once_per_symbol_per_session(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path,
                        params=dict(cfg()['params'], max_concurrent=1, max_per_day=8, max_resting=12))
        admit(e)
        cand = e.candidates['ABC']
        hod_live_db.get_open_trades.return_value = [{'symbol': 'XYZ'}]       # already at max_concurrent=1
        with caplog.at_level('INFO'):
            e._arm_live_order(cand, dict(BIG_VOL_ARM))
            e._arm_live_order(cand, dict(BIG_VOL_ARM))
        assert not hod_live_alpaca.submit_stop_limit_order.called
        cap_logs = [r for r in caplog.records if 'LIVE cap reached' in r.message and r.levelname == 'INFO']
        assert len(cap_logs) == 1
        assert not any(r.levelname == 'WARNING' and 'LIVE cap reached' in r.message for r in caplog.records)


# --------------------------------------------------------------------------------------------- owner 9/25: buying-power
# notional guard (max_resting is a safety ceiling only — the backtest never capped resting-order COUNT).
class TestBuyingPowerGuard:
    def test_resting_order_skipped_when_notional_would_exceed_25pct_of_buying_power(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        hod_live_alpaca.get_buying_power.return_value = 1000.0        # 25% = $250; one order's notional ~= $2,677
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path,
                        params=dict(cfg()['params'], max_concurrent=20, max_per_day=20, max_resting=20))
        admit(e)
        cand = e.candidates['ABC']
        with caplog.at_level('INFO'):
            e._arm_live_order(cand, dict(BIG_VOL_ARM))
        assert not hod_live_alpaca.submit_stop_limit_order.called
        assert cand.live_order is None
        assert any('buying-power guard' in r.message for r in caplog.records)

    def test_second_order_skipped_once_running_notional_would_cross_the_threshold(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        hod_live_alpaca.get_buying_power.return_value = 12_000.0      # 25% = $3,000: room for one ~$2,677 order, not two
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path,
                        params=dict(cfg()['params'], max_concurrent=20, max_per_day=20, max_resting=20))
        admit(e)
        cand1 = e.candidates['ABC']
        cand2 = Candidate(symbol='DEF', day_open=10.0, adv20=1_000_000.0)
        e.candidates['DEF'] = cand2
        e._arm_live_order(cand1, dict(BIG_VOL_ARM))
        e._arm_live_order(cand2, dict(BIG_VOL_ARM, level=12.0, trigger=12.01, limit=12.0173, stop=11.6))
        assert cand1.live_order is not None
        assert cand2.live_order is None
        assert hod_live_alpaca.submit_stop_limit_order.call_count == 1

    def test_buying_power_fetch_failure_with_no_cache_fails_closed(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        hod_live_alpaca.get_buying_power.side_effect = Exception('boom')
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path,
                        params=dict(cfg()['params'], max_concurrent=20, max_per_day=20, max_resting=20))
        admit(e)
        cand = e.candidates['ABC']
        with caplog.at_level('ERROR'):
            e._arm_live_order(cand, dict(BIG_VOL_ARM))
        assert not hod_live_alpaca.submit_stop_limit_order.called
        assert cand.live_order is None
        assert any('buying-power fetch failed' in r.message for r in caplog.records)


# --------------------------------------------------------------------------------------------- item 7: cutoff sweeps
class TestCutoffSweeps:
    def test_last_entry_minute_and_flat_minute_sweep_all_resting_orders(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        admit(e)
        cand1 = e.candidates['ABC']
        cand2 = Candidate(symbol='DEF', day_open=10.0, adv20=1_000_000.0)
        e.candidates['DEF'] = cand2
        e._arm_live_order(cand1, dict(BIG_VOL_ARM))
        e._arm_live_order(cand2, dict(BIG_VOL_ARM, level=12.0, trigger=12.01, limit=12.0173, stop=11.6))
        assert hod_live_alpaca.submit_stop_limit_order.call_count == 2

        e._minute_of_day = lambda: e.params.last_entry_minute
        e._sweep_live_cutoffs()
        assert cand1.live_order is None and cand2.live_order is None
        assert hod_live_alpaca.cancel_order.call_count == 2

        e._minute_of_day = lambda: e.params.flat_minute                 # nothing left resting - must not double-cancel
        e._sweep_live_cutoffs()
        assert hod_live_alpaca.cancel_order.call_count == 2


# --------------------------------------------------------------------------------------------- item 8: boot reconciliation
class TestBootReconciliation:
    def test_adopts_only_still_open_persisted_ids_and_never_cancels_an_unknown_order(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        state_path = tmp_path / 'state.json'
        persisted = {
            'ABC': dict(order_id='open-1', coid='hod-rest-ABC-x', level=11.0, trigger=11.01, limit=11.0165,
                       stop=10.6, qty=100, booked_qty=0, arm_ts='t', tp_leg_id='', sl_leg_id='', trade_id=None),
            'DEF': dict(order_id='gone-1', coid='hod-rest-DEF-x', level=12.0, trigger=12.01, limit=12.0173,
                       stop=11.6, qty=50, booked_qty=0, arm_ts='t', tp_leg_id='', sl_leg_id='', trade_id=None),
        }
        state_path.write_text(json.dumps(persisted))
        hod_live_alpaca.get_open_orders.return_value = [{'id': 'open-1'}, {'id': 'owner-manual-1'}]
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path,
                        live_orders_state_path=str(state_path))
        e._reconcile_live_orders_on_boot()
        assert e.candidates['ABC'].live_order['order_id'] == 'open-1'
        assert 'ABC' in e._live_cap_slots
        assert 'DEF' not in e.candidates or e.candidates['DEF'].live_order is None
        assert not hod_live_alpaca.cancel_order.called                  # adopts/drops only, never cancels


# --------------------------------------------------------------------------------------------- item 9: parity ledger rows
class TestParityLedgerTapeColumns:
    def test_fill_no_fill_no_cross_rows_with_tape_columns_populated_only_from_a_print_watch_cross(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        admit(e)
        cand = e.candidates['ABC']
        lo = dict(order_id='o1', coid='hod-rest-ABC-x', level=11.0, trigger=11.01, limit=11.0165, stop=10.6,
                  qty=100, booked_qty=0, arm_ts='t')

        # FILL: the print watch saw the cross and it filled.
        cand.resting_filled = True
        cand.tape_cross = {'ts': datetime(2026, 9, 28, 13, 36, tzinfo=timezone.utc), 'print': 11.02, 'ask': 11.015, 'nbbo_ok': True}
        e._append_live_parity_row(cand, lo, broker_status='filled',
                                  broker_fill_ts=datetime(2026, 9, 28, 13, 36, 5, tzinfo=timezone.utc),
                                  broker_fill_px=11.02, broker_fill_qty=100)

        # NO_FILL: the print watch saw a cross, but the ask was above the limit (no chase).
        cand.resting_filled = False
        cand.resting_tape_cross_idx = 6
        cand.tape_cross = {'ts': datetime(2026, 9, 28, 13, 37, tzinfo=timezone.utc), 'print': 11.02, 'ask': 11.03, 'nbbo_ok': True}
        e._append_live_parity_row(cand, lo, broker_status='cancelled', reason='last_entry_minute')

        # NO_CROSS: never crossed — nothing for the print watch to see.
        cand.resting_tape_cross_idx = None
        cand.tape_cross = None
        e._append_live_parity_row(cand, lo, broker_status='cancelled', reason='flat_minute')

        rows = read_ledger(e.live_parity_ledger_path)
        assert [r['tape_expected'] for r in rows] == ['FILL', 'NO_FILL', 'NO_CROSS']
        assert rows[0]['tape_ask'] == '11.0150' and rows[0]['tape_print'] == '11.0200' and rows[0]['trigger_print_nbbo_ok'] == '1'
        assert float(rows[0]['slippage_vs_tape_bps']) == pytest.approx((11.02 - 11.015) / 11.015 * 1e4, rel=1e-3)
        assert rows[1]['tape_ask'] == '11.0300' and rows[1]['slippage_vs_tape_bps'] == ''
        assert rows[2]['tape_ask'] == '' and rows[2]['tape_print'] == '' and rows[2]['trigger_print_nbbo_ok'] == ''


# --------------------------------------------------------------------------------------------- item 10: dry_run places nothing
class TestDryRunPlacesNothing:
    def test_dry_run_never_calls_the_live_order_surface(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, dry_run=True)
        admit(e)
        e._ingest_bars('ABC', bars_df(tape()))                          # full tape: tape side arms AND fills
        assert not hod_live_alpaca.submit_stop_limit_order.called
        assert not hod_live_alpaca.cancel_order.called
        assert e.candidates['ABC'].live_order is None
        assert e.candidates['ABC'].resting_filled is True                # tape path unaffected — byte-identical


# --------------------------------------------------------------------------------------------- item 11: guardrail pausable
class TestGuardrailPausesHodBreak:
    def test_hod_break_is_pausable_and_scales_to_its_own_risk_usd(self):
        assert 'hod_break' in gr.PAUSABLE_BOOKS
        # -3 x risk x SESSION_MULT['hod_break'] with risk_usd=50 (the first live week's config) = -600
        stats = _stats(book='hod_break', trailing_20_session_usd=-600.0)
        check = gr.evaluate_pause(stats, stage_risk_usd=50.0, band_p5=None)
        assert check.should_pause and check.rule == gr.RULE_TRAILING_20_SESSION

        above_band = _stats(book='hod_break', trailing_20_session_usd=-599.0)
        check2 = gr.evaluate_pause(above_band, stage_risk_usd=50.0, band_p5=None)
        assert not check2.should_pause
