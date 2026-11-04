"""Integration (real Database on a tempdir): a partial fill that later completes must leave ONE trades row with
shares == filled_qty == cumulative broker fill (2026-11-04 XPEL: shares stayed 21 while the broker held 34).
Spec: docs/hod_partial_fill_shares_spec_20261104.md."""
import os
import sqlite3
import tempfile

from persistence.database import Database
from tests.test_hod_break_engine import admit
from tests.test_hod_live_resting import BIG_VOL_ARM, live_engine


def test_partial_then_filled_leaves_one_row_with_shares_34_filled_qty_34(
        hod_live_alpaca, hod_live_sm, hod_live_stream, tmp_path):
    d = tempfile.mkdtemp()
    db = Database(trades_path=os.path.join(d, 'trades.db'), cache_path=os.path.join(d, 'cache.db'))
    e = live_engine(hod_live_alpaca, db, hod_live_sm, hod_live_stream, tmp_path)
    e._adv_map = {'ABC': 1_000_000.0}          # the real cache.db has no daily_bars here: seed the ADV gate directly
    admit(e)
    cand = e.candidates['ABC']
    e._arm_live_order(cand, dict(BIG_VOL_ARM))
    coid = cand.live_order['coid']
    for status, qty in (('partially_filled', 21), ('filled', 34)):
        hod_live_stream.snapshot_by_client_prefix.return_value = {
            coid: {'status': status, 'filled_qty': qty, 'filled_avg_price': 11.02, 'client_order_id': coid}}
        e._poll_live_fills()
    con = sqlite3.connect(os.path.join(d, 'trades.db'))
    rows = con.execute("SELECT shares, filled_qty FROM trades WHERE symbol='ABC'").fetchall()
    con.close()
    assert rows == [(34, 34)]
