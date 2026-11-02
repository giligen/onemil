"""Pass 4: the session schema template must be indistinguishable from the real constructor on a fresh path."""
import sqlite3
from pathlib import Path

from persistence.database import Database
from tests import conftest as cf


def _dump(path):
    """Every schema object (type, name, table, sql) plus the pragmas a constructor sets, sorted."""
    con = sqlite3.connect(str(path))
    try:
        rows = sorted(con.execute("SELECT type, name, tbl_name, sql FROM sqlite_master").fetchall(), key=repr)
        pragmas = (con.execute("PRAGMA journal_mode").fetchone()[0], con.execute("PRAGMA user_version").fetchone()[0])
        counts = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                  for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        return rows, pragmas, counts
    finally:
        con.close()


def _real_fresh(path):
    """A database built by the REAL constructor, no template."""
    inst = object.__new__(Database)
    cf._REAL_DB_INIT(inst, str(path))
    inst.close()
    return path


def test_template_copy_has_the_identical_schema_pragmas_and_no_rows(tmp_path):
    fresh = _dump(_real_fresh(tmp_path / "real.db"))
    db = Database(str(tmp_path / "via_template.db"))      # conftest's patched __init__ copies the template first
    db.close()
    via = _dump(tmp_path / "via_template.db")
    assert via == fresh
    assert len(fresh[0]) > 10 and all(n == 0 for n in fresh[2].values())


def test_existing_file_is_never_overwritten_by_the_template(tmp_path):
    p = tmp_path / "old.db"
    con = sqlite3.connect(str(p)); con.execute("CREATE TABLE marker(x)"); con.execute("INSERT INTO marker VALUES (7)")
    con.commit(); con.close()
    Database(str(p)).close()
    con = sqlite3.connect(str(p))
    assert con.execute("SELECT x FROM marker").fetchone()[0] == 7
    con.close()


def test_split_databases_are_not_templated(tmp_path):
    a, b = tmp_path / "cache.db", tmp_path / "trades.db"
    Database(cache_path=str(a), trades_path=str(b)).close()

    assert a.exists() and b.exists() and a != Path(cf._DB_TEMPLATE["path"])


def test_migration_still_runs_on_an_old_schema_file(tmp_path):
    """An existing file that lacks a migrated column (built by the real constructor, column dropped) is upgraded."""
    p = _real_fresh(tmp_path / "old_trades.db")
    con = sqlite3.connect(str(p))
    con.execute("ALTER TABLE trades DROP COLUMN filled_qty"); con.commit(); con.close()
    Database(str(p)).close()
    con = sqlite3.connect(str(p))
    cols = [r[1] for r in con.execute("PRAGMA table_info(trades)")]
    con.close()
    assert "filled_qty" in cols
