#!/usr/bin/env python3
"""
Phase 1 of the news-classifier A/B — precompute catalyst verdicts.

For every candidate (symbol, date) in the bull-flag Stage-1 cache, fetch that
day's news ONCE and classify it three ways:

  regex          — BacktestRunner._classify_headline (the BT today)
  haiku          — LLMNewsAnalyzer + production SYSTEM_PROMPT, temp=0 (LIVE)
  haiku_revised  — LLMNewsAnalyzer + REVISED_SYSTEM_PROMPT, temp=0

Verdicts are stored in data/news_ab.db. The backtest then reads them via the
BT_NEWS_CLASSIFIER env var, so the A/B runs themselves cost ZERO API calls.

Cost is bounded by the cache size (~1,195 rows): ~$6-8 of Haiku, one time.
Idempotent + resumable — already-done rows are skipped; safe to Ctrl-C.

Usage:
  python scripts/precompute_news_ab.py --limit 15     # smoke test first
  python scripts/precompute_news_ab.py                # full cache
  python scripts/precompute_news_ab.py --reclassify   # re-Haiku cached articles
"""
import argparse
import csv
import os
import sqlite3
import sys
import time

# Resolve project root from this script's location so relative data/ paths and
# imports work regardless of the caller's cwd (same idiom on EC2 and local).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

CACHE_CSV = "data/bull_flag_cache_e50_x30.csv"
UNIVERSE_DB = "data/cache.db"
HAIKU_COST_PER_CALL = 0.0009   # ~570 in @ $1/M + ~60 out @ $5/M, Haiku 4.5


def load_env() -> None:
    """Best-effort load of .env into os.environ (does not overwrite existing)."""
    if os.path.exists('.env'):
        for line in open('.env'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k, v.strip())


def get_keys() -> tuple:
    """Return (anthropic_key, alpaca_key, alpaca_secret) or exit fatally.

    CLAUDE.md: missing API keys must break execution loudly, never silently.
    """
    load_env()
    ak = os.environ.get('ANTHROPIC_API_KEY')
    alp_k = os.environ.get('ALPACA_API_KEY') or os.environ.get('APCA_API_KEY_ID')
    alp_s = os.environ.get('ALPACA_API_SECRET') or os.environ.get('APCA_API_SECRET_KEY')
    missing = [n for n, v in [('ANTHROPIC_API_KEY', ak),
                              ('ALPACA_API_KEY', alp_k),
                              ('ALPACA_API_SECRET', alp_s)] if not v]
    if missing:
        sys.exit(f"FATAL: missing required API key(s): {', '.join(missing)}")
    return ak, alp_k, alp_s


def load_cache_rows(cache_csv: str) -> list:
    """Return the de-duplicated [(symbol, date), ...] candidate set."""
    if not os.path.exists(cache_csv):
        sys.exit(f"FATAL: cache CSV not found: {cache_csv}")
    seen, rows = set(), []
    with open(cache_csv) as f:
        for r in csv.DictReader(f):
            key = (r['symbol'], r['date'])
            if key not in seen:
                seen.add(key)
                rows.append(key)
    return rows


def load_floats(universe_db: str) -> dict:
    """Return {symbol: float_shares} from the cache.db universe table."""
    floats = {}
    try:
        conn = sqlite3.connect(universe_db)
        for sym, fl in conn.execute(
                "SELECT symbol, float_shares FROM universe "
                "WHERE float_shares IS NOT NULL"):
            floats[sym] = int(fl) if fl else 0
        conn.close()
    except Exception as e:
        print(f"WARNING: float lookup failed ({e}) — Haiku runs without "
              f"float context", flush=True)
    return floats


def regex_verdict(articles: list, classify_fn, real_cats) -> tuple:
    """Regex classification of a day's articles: (has_catalyst, category)."""
    cat = 'NO_NEWS'
    for art in articles:
        hl = (art.get('headline') or '').strip()
        if not hl:
            continue
        c = classify_fn(hl)
        if c in real_cats:
            return True, c
        cat = c
    return False, cat


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute news-classifier A/B verdicts")
    ap.add_argument('--limit', type=int, default=0,
                    help="process only the first N rows (smoke test)")
    ap.add_argument('--reclassify', action='store_true',
                    help="re-run Haiku on cached articles (after a prompt change)")
    ap.add_argument('--cache', default=CACHE_CSV, help="Stage-1 cache CSV path")
    args = ap.parse_args()

    ak, alp_k, alp_s = get_keys()

    import anthropic
    from backtest import BacktestRunner
    from data_sources.news_provider import LLMNewsAnalyzer
    from trading.news_ab import (fetch_alpaca_news, classify_day_catalyst,
                                 NewsABStore, REVISED_SYSTEM_PROMPT)

    real_cats = BacktestRunner._REAL_CATS
    regex_fn = BacktestRunner._classify_headline
    client = anthropic.Anthropic(api_key=ak)
    an_cur = LLMNewsAnalyzer(client)                                  # LIVE prompt
    an_rev = LLMNewsAnalyzer(client, system_prompt=REVISED_SYSTEM_PROMPT)
    store = NewsABStore()
    floats = load_floats(UNIVERSE_DB)

    rows = load_cache_rows(args.cache)
    if args.limit:
        rows = rows[:args.limit]
    total = len(rows)

    print("=" * 70)
    print(f"NEWS A/B PRECOMPUTE — {total} candidate (symbol, date) rows")
    print(f"  classifiers: regex, haiku, haiku_revised   db: {store.db_path}")
    print(f"  mode: {'RECLASSIFY (cached articles)' if args.reclassify else 'fetch + classify'}")
    print("=" * 70, flush=True)

    done = skipped = haiku_calls = 0
    t0 = time.time()
    for i, (sym, date) in enumerate(rows, 1):
        if not args.reclassify and store.is_done(sym, date):
            skipped += 1
            continue

        articles = store.get_articles(sym, date) if args.reclassify else None
        if not articles:
            articles = fetch_alpaca_news(sym, date, alp_k, alp_s)
            store.save_articles(sym, date, articles)

        ctx = {'float_shares': floats[sym]} if floats.get(sym) else None

        rx_cat, rx_category = regex_verdict(articles, regex_fn, real_cats)

        an_cur._cache.clear()
        an_rev._cache.clear()
        h_cat, h_category, top_hl, n1 = classify_day_catalyst(
            articles, sym, an_cur, real_cats, ctx)
        hr_cat, hr_category, _, n2 = classify_day_catalyst(
            articles, sym, an_rev, real_cats, ctx)
        haiku_calls += n1 + n2

        store.upsert(sym, date, n_articles=len(articles),
                     regex_catalyst=rx_cat, haiku_catalyst=h_cat,
                     haiku_revised_catalyst=hr_cat,
                     regex_category=rx_category, haiku_category=h_category,
                     haiku_revised_category=hr_category, top_headline=top_hl)
        done += 1

        if i % 25 == 0 or i == total:
            el = time.time() - t0
            rate = i / el if el else 0
            eta = (total - i) / rate / 60 if rate else 0
            print(f"  [{i}/{total}] done={done} skipped={skipped}  "
                  f"haiku_calls={haiku_calls} (~${haiku_calls * HAIKU_COST_PER_CALL:.2f})  "
                  f"{rate:.1f} rows/s  ETA {eta:.1f}m", flush=True)

    store.close()
    print("=" * 70)
    print(f"DONE — {done} classified, {skipped} already cached, "
          f"{haiku_calls} Haiku calls (~${haiku_calls * HAIKU_COST_PER_CALL:.2f})")
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()
