"""Write-transaction serialization in Database (2026-11-19 shared-connection commit race).

Two ORB preplace threads shared one sqlite connection; B's commit() swallowed A's INSERT and A's
commit() raised 'cannot commit - no transaction is active'. Every write now runs under `_write`.
"""
import ast
import threading
from pathlib import Path

import pytest

from persistence.database import Database

DB_SRC = Path(__file__).resolve().parent.parent / "persistence" / "database.py"
_WRITE_PREFIXES = ("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "ALTER", "DROP")


def _record(i: int) -> dict:
    """Minimal valid trades row with a unique order id."""
    return {
        'trade_date': '2026-11-19', 'symbol': f'T{i}', 'side': 'buy', 'entry_price': 10.0,
        'stop_loss_price': 9.0, 'take_profit_price': 12.0, 'shares': 10, 'risk_per_share': 1.0,
        'total_risk': 10.0, 'risk_reward_ratio': 2.0, 'order_id': f'oid-{i}',
        'order_status': 'pending_new', 'strategy': 'orb', 'account': 'paper',
        'fill_price': None, 'filled_at': None, 'exit_price': None, 'exit_reason': None,
        'exited_at': None, 'pnl': None, 'pnl_pct': None, 'pattern_data': None,
    }


@pytest.fixture()
def db(tmp_path):
    """Temp-file Database torn down after the test."""
    database = Database(db_path=str(tmp_path / "wl.db"))
    yield database
    database.close()


def _first_str(node: ast.AST):
    """Leading constant text of a str / f-string node, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
        return node.values[0].value
    return None


def test_every_write_is_inside_write_lock():
    """Exactly one .commit() (inside _write); every write statement lexically inside `with self._write`."""
    tree = ast.parse(DB_SRC.read_text())
    parents = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[c] = p

    def ancestors(n):
        while n in parents:
            n = parents[n]
            yield n

    def in_write_with(n):
        for a in ancestors(n):
            if isinstance(a, ast.With):
                for item in a.items:
                    e = item.context_expr
                    if (isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute)
                            and e.func.attr == '_write'):
                        return True
        return False

    commits, bad = [], []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
            continue
        if n.func.attr == 'commit':
            commits.append(n)
        elif n.func.attr in ('execute', 'executemany', 'executescript') and n.args:
            text = _first_str(n.args[0])
            if text is not None and text.strip().upper().startswith(_WRITE_PREFIXES):
                if not in_write_with(n):
                    bad.append(n.lineno)
    assert len(commits) == 1, f"commit() sites: {[c.lineno for c in commits]}"
    fn = next(a for a in ancestors(commits[0]) if isinstance(a, ast.FunctionDef))
    assert fn.name == '_write'
    assert not bad, f"write statements outside `with self._write` at lines {bad}"


def test_write_blocks_concurrent_writer(db):
    """B's save_trade must wait while A holds an open write transaction (no interleaved commit)."""
    a_in_tx, go, errors = threading.Event(), threading.Event(), []

    def writer_a():
        try:
            with db._write(db._trades_conn) as c:
                c.execute(
                    "INSERT INTO trades (trade_date, symbol, side, entry_price, stop_loss_price, take_profit_price, shares, risk_per_share, total_risk, risk_reward_ratio, order_id, "
                    "order_status, strategy, account, created_at, updated_at) VALUES "
                    "('2026-11-19','AAA','buy',1.0,0.9,1.2,1,0.1,0.1,2.0,'oid-A','pending_new','orb','paper',"
                    "datetime('now'),datetime('now'))")
                a_in_tx.set()
                go.wait(5)
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    def writer_b():
        try:
            db.save_trade(_record(2))
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    ta = threading.Thread(target=writer_a)
    ta.start()
    assert a_in_tx.wait(5)
    tb = threading.Thread(target=writer_b)
    tb.start()
    tb.join(0.3)
    assert tb.is_alive(), "B was not blocked by A's open write transaction"
    go.set()
    ta.join(5)
    tb.join(5)
    assert not errors, errors
    rows = db._trades_conn.execute("SELECT id FROM trades ORDER BY id").fetchall()
    assert len(rows) == 2 and rows[0][0] != rows[1][0]


def test_write_rolls_back_on_exception(db):
    """An exception inside _write rolls back and re-raises."""
    with pytest.raises(RuntimeError):
        with db._write(db._trades_conn) as c:
            c.execute("INSERT INTO trades (trade_date, symbol, side, entry_price, stop_loss_price, take_profit_price, shares, risk_per_share, total_risk, risk_reward_ratio, order_id, "
                      "order_status, strategy, account, created_at, updated_at) VALUES "
                      "('2026-11-19','ZZZ','buy',1.0,0.9,1.2,1,0.1,0.1,2.0,'oid-Z','x','orb','paper',"
                      "datetime('now'),datetime('now'))")
            raise RuntimeError("boom")
    assert db._trades_conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0


def test_concurrent_save_trade_brute_force(db):
    """8 threads x 25 save_trade -> 200 rows, no exception."""
    errors, barrier = [], threading.Barrier(8)

    def worker(w):
        barrier.wait()
        for k in range(25):
            try:
                db.save_trade(_record(w * 100 + k))
            except Exception as e:
                errors.append(e)

    ts = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    [t.start() for t in ts]
    [t.join(60) for t in ts]
    assert not errors, errors[:3]
    assert db._trades_conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 200
