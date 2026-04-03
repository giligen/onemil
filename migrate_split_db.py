"""
One-time migration: split onemil.db into cache.db + trades.db.

Usage:
    python migrate_split_db.py [--source data/onemil.db]

Creates:
    data/cache.db  — universe, volume_profiles, daily_bars, intraday_bars_1min, news_cache
    data/trades.db — trades, scan_results, daily_trading_summary

The original onemil.db is kept as backup (not deleted).
"""
import argparse
import os
import sqlite3
import sys
import time


CACHE_TABLES = ['universe', 'volume_profiles', 'daily_bars', 'intraday_bars_1min', 'news_cache']
TRADES_TABLES = ['trades', 'scan_results', 'daily_trading_summary']


def migrate(source_path: str, cache_path: str = "data/cache.db", trades_path: str = "data/trades.db"):
    """Split source DB into cache and trades databases."""
    if not os.path.exists(source_path):
        print(f"ERROR: Source DB not found: {source_path}")
        sys.exit(1)

    if os.path.exists(cache_path) or os.path.exists(trades_path):
        print(f"ERROR: Target DBs already exist. Remove them first:")
        if os.path.exists(cache_path):
            print(f"  {cache_path} ({os.path.getsize(cache_path) / 1e9:.1f} GB)")
        if os.path.exists(trades_path):
            print(f"  {trades_path} ({os.path.getsize(trades_path) / 1e9:.1f} GB)")
        sys.exit(1)

    source_size = os.path.getsize(source_path) / 1e9
    print(f"Source: {source_path} ({source_size:.1f} GB)")
    print(f"Cache target: {cache_path}")
    print(f"Trades target: {trades_path}")
    print()

    src = sqlite3.connect(source_path)

    # Get table list from source
    tables = [r[0] for r in src.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    print(f"Tables in source: {tables}")

    # Copy cache tables
    print(f"\nCopying cache tables to {cache_path}...")
    t0 = time.time()
    src.execute(f"ATTACH DATABASE '{cache_path}' AS cache_db")
    for table in CACHE_TABLES:
        if table not in tables:
            print(f"  {table}: SKIP (not in source)")
            continue
        # Create table in target with same schema
        schema = src.execute(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'"
        ).fetchone()[0]
        src.execute(f"CREATE TABLE IF NOT EXISTS cache_db.{table} AS SELECT * FROM main.{table} WHERE 0")
        # Actually copy with schema
        src.execute(f"DROP TABLE IF EXISTS cache_db.{table}")
        src.execute(schema.replace(f"CREATE TABLE {table}", f"CREATE TABLE cache_db.{table}"))
        count = src.execute(f"INSERT INTO cache_db.{table} SELECT * FROM main.{table}").rowcount
        src.commit()
        print(f"  {table}: {count:,} rows")

    # Copy indexes for cache tables
    for row in src.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
    ).fetchall():
        sql = row[0]
        # Check if this index belongs to a cache table
        for table in CACHE_TABLES:
            if f"ON {table}" in sql or f"ON {table}(" in sql:
                try:
                    src.execute(sql.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS cache_db.idx_", 1)
                                if "cache_db." not in sql
                                else sql)
                except Exception:
                    pass  # Index may already exist
    src.execute("DETACH DATABASE cache_db")
    elapsed = time.time() - t0
    cache_size = os.path.getsize(cache_path) / 1e9
    print(f"  Done in {elapsed:.1f}s ({cache_size:.2f} GB)")

    # Copy trades tables
    print(f"\nCopying trades tables to {trades_path}...")
    t0 = time.time()
    src.execute(f"ATTACH DATABASE '{trades_path}' AS trades_db")
    for table in TRADES_TABLES:
        if table not in tables:
            print(f"  {table}: SKIP (not in source)")
            continue
        schema = src.execute(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'"
        ).fetchone()[0]
        src.execute(f"DROP TABLE IF EXISTS trades_db.{table}")
        src.execute(schema.replace(f"CREATE TABLE {table}", f"CREATE TABLE trades_db.{table}"))
        count = src.execute(f"INSERT INTO trades_db.{table} SELECT * FROM main.{table}").rowcount
        src.commit()
        print(f"  {table}: {count:,} rows")

    # Copy indexes for trades tables
    for row in src.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
    ).fetchall():
        sql = row[0]
        for table in TRADES_TABLES:
            if f"ON {table}" in sql or f"ON {table}(" in sql:
                try:
                    src.execute(sql.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS trades_db.idx_", 1)
                                if "trades_db." not in sql
                                else sql)
                except Exception:
                    pass
    src.execute("DETACH DATABASE trades_db")
    elapsed = time.time() - t0
    trades_size = os.path.getsize(trades_path) / 1e6
    print(f"  Done in {elapsed:.1f}s ({trades_size:.1f} MB)")

    src.close()

    print(f"\n{'='*50}")
    print(f"Migration complete!")
    print(f"  Cache:  {cache_path} ({os.path.getsize(cache_path)/1e9:.2f} GB)")
    print(f"  Trades: {trades_path} ({os.path.getsize(trades_path)/1e6:.1f} MB)")
    print(f"  Source: {source_path} (kept as backup)")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split onemil.db into cache.db + trades.db")
    parser.add_argument("--source", default="data/onemil.db", help="Source DB path")
    parser.add_argument("--cache", default="data/cache.db", help="Cache DB output path")
    parser.add_argument("--trades", default="data/trades.db", help="Trades DB output path")
    args = parser.parse_args()
    migrate(args.source, args.cache, args.trades)
