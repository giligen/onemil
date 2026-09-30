"""HOD-break failure-short — pure order-mechanics unit tests (trading/hod_failure_short.py).
Rails, sizing, stop/target prices, caps and the day kill. No I/O, no engine, no broker."""
import pytest

from trading.hod_failure_short import (
    FailureShortConfig, fresh_short_qty, ledger_row, protective_stop_price, rails_reason,
    resulting_short_qty, reversal_sell_qty, should_evaluate_signal, should_submit,
    short_pattern_data, target_buy_price,
)


def cfg(**over) -> FailureShortConfig:
    return FailureShortConfig(**over)


class TestSizing:
    def test_reversal_sell_qty_is_double_the_long(self):
        assert reversal_sell_qty(100) == 200
        assert reversal_sell_qty(37) == 74

    def test_resulting_short_qty_equals_long_shares(self):
        assert resulting_short_qty(100) == 100
        assert resulting_short_qty(37) == 37

    def test_reversal_then_resulting_are_consistent(self):
        long_shares = 58
        assert reversal_sell_qty(long_shares) // 2 == resulting_short_qty(long_shares)

    def test_fresh_short_qty_risk_based(self):
        # risk $150, entry 10.00, stop 10.50 -> risk/sh 0.50 -> 300 sh
        assert fresh_short_qty(150.0, 10.00, 10.50) == 300

    def test_fresh_short_qty_non_positive_risk_returns_zero(self):
        assert fresh_short_qty(150.0, 10.50, 10.00) == 0   # stop below entry: not a valid short stop
        assert fresh_short_qty(150.0, 10.00, 10.00) == 0   # zero risk/share


class TestPrices:
    def test_protective_stop_is_day_high_plus_one_cent(self):
        assert protective_stop_price(12.34) == 12.35

    def test_protective_stop_rounds_to_cents(self):
        assert protective_stop_price(12.3449) == 12.35  # 12.3449 + 0.01 = 12.3549 -> round to 12.35

    def test_target_is_the_longs_stop_level(self):
        assert target_buy_price(9.87) == 9.87

    def test_target_rounds_to_cents(self):
        assert target_buy_price(9.8765) == 9.88


class TestRails:
    def base_kwargs(self, **over):
        k = dict(shortable=True, easy_to_borrow=True, require_etb=True, price=20.0, prior_close=20.0,
                 open_count=0, submitted_today=0, day_realized_r=0.0, cfg=cfg())
        k.update(over); return k

    def test_all_rails_pass(self):
        assert rails_reason(**self.base_kwargs()) is None

    def test_not_shortable_blocks(self):
        assert rails_reason(**self.base_kwargs(shortable=False)) == 'not_shortable'

    def test_not_etb_blocks_when_required(self):
        assert rails_reason(**self.base_kwargs(easy_to_borrow=False)) == 'not_etb'

    def test_not_etb_allowed_when_require_etb_false(self):
        assert rails_reason(**self.base_kwargs(easy_to_borrow=False, require_etb=False)) is None

    def test_ssr_unknown_prior_close_fails_closed(self):
        assert rails_reason(**self.base_kwargs(prior_close=0.0)) == 'ssr_unknown_prior_close'
        assert rails_reason(**self.base_kwargs(prior_close=-1.0)) == 'ssr_unknown_prior_close'

    def test_ssr_blocked_below_90pct_of_prior_close(self):
        # prior close 20.00 -> SSR floor 18.00
        assert rails_reason(**self.base_kwargs(price=17.99, prior_close=20.0)) == 'ssr_blocked'
        assert rails_reason(**self.base_kwargs(price=18.00, prior_close=20.0)) is None   # exactly at the floor passes

    def test_max_concurrent_cap(self):
        c = cfg(max_concurrent=3)
        assert rails_reason(**self.base_kwargs(open_count=3, cfg=c)) == 'max_concurrent'
        assert rails_reason(**self.base_kwargs(open_count=2, cfg=c)) is None

    def test_max_per_day_cap(self):
        c = cfg(max_per_day=8)
        assert rails_reason(**self.base_kwargs(submitted_today=8, cfg=c)) == 'max_per_day'
        assert rails_reason(**self.base_kwargs(submitted_today=7, cfg=c)) is None

    def test_day_kill_blocks(self):
        c = cfg(day_kill_r=-5.0)
        assert rails_reason(**self.base_kwargs(day_realized_r=-5.0, cfg=c)) == 'day_kill'
        assert rails_reason(**self.base_kwargs(day_realized_r=-5.1, cfg=c)) == 'day_kill'
        assert rails_reason(**self.base_kwargs(day_realized_r=-4.9, cfg=c)) is None

    def test_rail_order_borrow_before_ssr(self):
        # not shortable AND SSR-blocked -> reports not_shortable first (evidence: caller order)
        assert rails_reason(**self.base_kwargs(shortable=False, price=1.0, prior_close=20.0)) == 'not_shortable'


class TestBarTiming:
    def test_should_evaluate_signal_exact_bar(self):
        assert should_evaluate_signal(current_minute=601, fill_minute=600) is True

    def test_should_evaluate_signal_not_yet(self):
        assert should_evaluate_signal(current_minute=600, fill_minute=600) is False

    def test_should_evaluate_signal_late_poll_still_fires(self):
        assert should_evaluate_signal(current_minute=605, fill_minute=600) is True

    def test_should_submit_default_offset_two(self):
        assert should_submit(current_minute=602, fill_minute=600, entry_bar_offset=2) is True
        assert should_submit(current_minute=601, fill_minute=600, entry_bar_offset=2) is False

    def test_should_submit_custom_offset(self):
        assert should_submit(current_minute=603, fill_minute=600, entry_bar_offset=3) is True
        assert should_submit(current_minute=602, fill_minute=600, entry_bar_offset=3) is False


class TestLedgerAndPatternData:
    def test_ledger_row_passed_true_when_go(self):
        row = ledger_row(ts='t', date='2026-09-30', symbol='ABC', fill_bar=601, p=0.75, cfg=cfg(tau=0.6),
                          rails=None, would_be_qty=200, would_be_stop=12.35, would_be_target=11.80)
        assert row['passed'] is True and row['p'] == 0.75 and row['rails_reason'] == ''

    def test_ledger_row_passed_false_on_low_p(self):
        row = ledger_row(ts='t', date='2026-09-30', symbol='ABC', fill_bar=601, p=0.5, cfg=cfg(tau=0.6),
                          rails=None, would_be_qty=200, would_be_stop=12.35, would_be_target=11.80)
        assert row['passed'] is False

    def test_ledger_row_passed_false_when_p_none(self):
        row = ledger_row(ts='t', date='2026-09-30', symbol='ABC', fill_bar=601, p=None, cfg=cfg(tau=0.6),
                          rails=None, would_be_qty=200, would_be_stop=12.35, would_be_target=11.80)
        assert row['passed'] is False and row['p'] == ''

    def test_ledger_row_passed_false_when_rails_block(self):
        row = ledger_row(ts='t', date='2026-09-30', symbol='ABC', fill_bar=601, p=0.9, cfg=cfg(tau=0.6),
                          rails='not_shortable', would_be_qty=200, would_be_stop=12.35, would_be_target=11.80)
        assert row['passed'] is False and row['rails_reason'] == 'not_shortable'

    def test_short_pattern_data_shape(self):
        d = short_pattern_data(long_trade_id=42, p=0.71, tp_leg_id='tp1', sl_leg_id='sl1', target=11.80, stop=12.35)
        assert d['mechanism'] == 'failure_short' and d['reversed_long_trade_id'] == 42
        assert d['tp_leg_id'] == 'tp1' and d['sl_leg_id'] == 'sl1' and d['target'] == 11.80 and d['stop'] == 12.35


class TestConfigDefaults:
    def test_defaults_are_fail_safe(self):
        c = FailureShortConfig()
        assert c.enabled is False and c.telemetry_only is True and c.allow_fresh_short is False
