"""FC DB SYNC prices an added-to position off the BLENDED cost basis (2026-11-24).

RAMZ 11/24: base 216 @ 23.066991, add 216 @ 24.007454 (blended 23.5372225 in pattern_data.add_on.add_avg_px),
force-closed 432 @ 24.02. The sync booked (24.02 - 23.066991) * 432 = +411.70 instead of +208.58; the same on
NVDQ, MUD and MUZ (+$687 overstated in total). The row's fill_price stays the base fill (rehydrate contract, D3 11/13),
so the blended basis and the add-be leg's P&L must come from pattern_data.add_on.
"""
import json
import logging
from unittest.mock import MagicMock

import pytest

from trading.exit_reasons import ExitReason
from tests.test_orb_winner_stack_engine import _skeleton_engine, _scaled_db_row, _db_updates, TestSyncDbAfterFc


def _ramz_row(add_on=None, pattern_data_raw=None):
    row = _scaled_db_row()
    row.update({'id': 882, 'symbol': 'RAMZ', 'fill_price': 23.066991, 'entry_price': 23.5372225,
                'shares': 432, 'filled_qty': 432, 'exit_price': None,
                'scale_qty': None, 'scale_price': None, 'scale_pnl': None, 'scaled_at': None})
    pdata = {'range_high': 23.0, 'range_low': 22.0}
    if add_on is not None:
        pdata['add_on'] = add_on
    row['pattern_data'] = pattern_data_raw if pattern_data_raw is not None else json.dumps(pdata)
    return row


def _engine_with(row, sell_price, sell_qty):
    eng = _skeleton_engine()
    eng.db.get_open_trades.return_value = [row]
    eng.alpaca.trading_client.get_orders.return_value = [
        TestSyncDbAfterFc._sell_order(TestSyncDbAfterFc(), price=sell_price, qty=sell_qty)]
    eng.check_exits = MagicMock(return_value=[])
    return eng


class TestFcSyncAddOnBasis:
    def test_ramz_replay_uses_blended_basis(self):
        row = _ramz_row(add_on={'add_qty': 216, 'add_avg_px': 23.5372225,
                                'adds': [{'at_r': 1.0, 'px': 24.007454, 'qty': 216}]})
        eng = _engine_with(row, 24.02, 432)
        eng._sync_db_after_fc(['RAMZ'])
        upd = _db_updates(eng.db)[-1]
        assert upd['pnl'] == pytest.approx((24.02 - 23.5372225) * 432, abs=0.01)   # +208.58, not +411.70
        assert upd['pnl_pct'] == pytest.approx(upd['pnl'] / (23.5372225 * 432) * 100, abs=1e-6)
        assert upd['exit_reason'] == ExitReason.FORCE_CLOSE.value

    def test_add_be_pnl_composes_once(self):
        row = _ramz_row(add_on={'add_avg_px': 23.5372225, 'add_be_pnl': 12.5})
        eng = _engine_with(row, 24.02, 432)
        eng._sync_db_after_fc(['RAMZ'])
        assert _db_updates(eng.db)[-1]['pnl'] == pytest.approx((24.02 - 23.5372225) * 432 + 12.5, abs=0.01)

    def test_row_without_add_on_prices_off_the_fill(self):
        row = _ramz_row(add_on=None)
        eng = _engine_with(row, 24.02, 432)
        eng._sync_db_after_fc(['RAMZ'])
        assert _db_updates(eng.db)[-1]['pnl'] == pytest.approx((24.02 - 23.066991) * 432, abs=0.01)

    def test_unreadable_pattern_data_warns_and_prices_off_the_fill(self, caplog):
        row = _ramz_row(pattern_data_raw='{not json')
        eng = _engine_with(row, 24.02, 432)
        with caplog.at_level(logging.WARNING):
            eng._sync_db_after_fc(['RAMZ'])
        assert _db_updates(eng.db)[-1]['pnl'] == pytest.approx((24.02 - 23.066991) * 432, abs=0.01)
        assert any('pattern_data unreadable' in r.getMessage() for r in caplog.records)

    def test_scaled_row_still_composes_scale_pnl(self):
        """The existing winner-stack contract is untouched: scale_pnl composes once at final close."""
        row = _scaled_db_row()
        row['exit_price'] = None
        eng = _engine_with(row, 11.0, 600)
        eng._sync_db_after_fc(['TEST'])
        assert _db_updates(eng.db)[-1]['pnl'] == pytest.approx((11.0 - 10.0) * 600 + 1200.0)

    def test_add_on_pdata_helper(self):
        from trading.orb_engine import _add_on_pdata
        assert _add_on_pdata({'pattern_data': None}) == {}
        assert _add_on_pdata({'pattern_data': {'add_on': {'add_avg_px': 1.5}}}) == {'add_avg_px': 1.5}
        assert _add_on_pdata({'pattern_data': json.dumps({'x': 1})}) == {}
