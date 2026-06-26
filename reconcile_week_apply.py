"""Apply the two verified reconciliation fixes (FABC #196, GLXG #210).

Re-queries Alpaca for the exact fills (single source of truth), verifies the
target row still matches what the audit saw (db_id + symbol + current pnl),
then UPDATEs. Idempotent-ish: if a row already matches the broker truth it's
left untouched. Backup must exist before running (caller's responsibility).
"""
from __future__ import annotations

import sqlite3
import sys

from reconcile_week_audit import true_fills

# (db_id, symbol, date, strategy, expected_current_pnl, new_exit_reason_or_None)
TARGETS = [
    (196, 'FABC', '2026-06-09', 'orb', -560.82, None),          # keep 'stop_loss'
    (210, 'GLXG', '2026-06-11', 'bull_flag', 0.0, 'trail_stop'),
]

conn = sqlite3.connect('data/trades.db', timeout=30.0)
conn.execute('PRAGMA busy_timeout=30000')
cur = conn.cursor()

for tid, sym, day, strat, exp_pnl, new_reason in TARGETS:
    cur.execute("SELECT symbol, shares, filled_qty, fill_price, exit_price, "
                "pnl, pnl_pct, exit_reason FROM trades WHERE id=?", (tid,))
    row = cur.fetchone()
    if row is None:
        print(f"db_id={tid}: NOT FOUND — skip"); continue
    (db_sym, db_sh, db_fq, db_fp, db_xp, db_pnl, db_pp, db_xr) = row
    if db_sym != sym:
        print(f"db_id={tid}: symbol mismatch ({db_sym}!={sym}) — ABORT row"); continue
    if abs((db_pnl or 0.0) - exp_pnl) > 0.01:
        print(f"db_id={tid} {sym}: current pnl ${db_pnl} != expected ${exp_pnl} "
              f"(already fixed?) — skip"); continue

    bq, bn, sq, sn, n = true_fills(strat, sym, day)
    if bq <= 0 or sq <= 0:
        print(f"db_id={tid} {sym}: not round-tripped on broker (buy={bq} sell={sq}) — skip"); continue
    shares = int(round(bq))
    fill_avg = round(bn / bq, 6)
    exit_avg = round(sn / sq, 6)
    true_pnl = round(sn - bn, 2)
    pnl_pct = round(true_pnl / (fill_avg * shares) * 100, 6) if fill_avg and shares else 0.0

    upd = {
        'shares': shares,
        'filled_qty': shares,
        'fill_price': fill_avg,
        'exit_price': exit_avg,
        'pnl': true_pnl,
        'pnl_pct': pnl_pct,
    }
    if new_reason:
        upd['exit_reason'] = new_reason

    set_sql = ', '.join(f"{k}=:{k}" for k in upd)
    upd['id'] = tid
    cur.execute(f"UPDATE trades SET {set_sql} WHERE id=:id", upd)

    print(f"db_id={tid} {sym} {day}:")
    print(f"    shares   {db_sh} -> {shares}")
    print(f"    fill     ${db_fp} -> ${fill_avg}")
    print(f"    exit     ${db_xp} -> ${exit_avg}")
    print(f"    pnl      ${db_pnl:+.2f} -> ${true_pnl:+.2f}")
    print(f"    pnl_pct  {db_pp} -> {pnl_pct}")
    if new_reason:
        print(f"    reason   {db_xr} -> {new_reason}")

conn.commit()
conn.close()
print("\nCommitted.")
