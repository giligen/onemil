"""
News-classifier A/B infrastructure for the bull-flag backtest.

The backtest's news-kill gate decides whether a no-catalyst trade in a loser
segment is killed. Today that catalyst decision is made by a regex classifier
(`BacktestRunner._classify_headline`); production LIVE uses Claude Haiku
(`LLMNewsAnalyzer`). This module lets the backtest compare classifiers on the
SAME candidate trades:

  regex          — BacktestRunner._classify_headline (BT today)
  haiku          — LLMNewsAnalyzer + production SYSTEM_PROMPT, temp=0 (LIVE)
  haiku_revised  — LLMNewsAnalyzer + REVISED_SYSTEM_PROMPT, temp=0
                   (adds "a patent APPLICATION is never a catalyst" — the
                   step-0 QUCY experiment's A3/A4 variant)

Workflow:
  Phase 1  scripts/precompute_news_ab.py — fetch news + classify every
           candidate (symbol, date) once; store verdicts in data/news_ab.db.
  Phase 2  backtest.py reads the precomputed verdict (BT_NEWS_CLASSIFIER env),
           so the A/B runs cost ZERO API calls and differ only by column.

Import-safe for backtest.py: this module imports nothing from backtest → no
import cycle.
"""
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from data_sources.news_provider import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_DB = "data/news_ab.db"

# ── Revised prompt — the step-0 experiment's A3/A4 variant ────────────────────
# Built by patching the production SYSTEM_PROMPT so the two stay in lock-step
# except for the one audited diff. The asserts fail LOUD if a future
# SYSTEM_PROMPT edit breaks a patch target — a silent no-op would quietly
# corrupt the A/B (arm B would become identical to arm A).
_PL_OLD = "  PRODUCT_LAUNCH — new product, expansion, patent, initiative\n"
_PL_NEW = ("  PRODUCT_LAUNCH — new commercial product/service launch, major "
           "expansion, granted patent tied to a named product\n")
_R4_OLD = ("4. Multiple similar PRs from same company in short period = pump "
           "cadence = GARBAGE_RECAP")
_R4_NEW = (_R4_OLD +
           "\n5. A PATENT APPLICATION (provisional OR non-provisional; verbs "
           "'files'/'filed'/'applies for') is NEVER a catalyst -> OTHER. "
           "Filing != grant != revenue. Only an ALREADY-GRANTED patent tied "
           "to a named product or licensing deal = PRODUCT_LAUNCH.\n"
           "6. If genuinely uncertain between a catalyst category and OTHER, "
           "choose OTHER (catalyst=false). Never guess upward.")

assert _PL_OLD in SYSTEM_PROMPT, "news_ab: revised-prompt patch target _PL_OLD not found"
assert _R4_OLD in SYSTEM_PROMPT, "news_ab: revised-prompt patch target _R4_OLD not found"
REVISED_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(_PL_OLD, _PL_NEW).replace(_R4_OLD, _R4_NEW)
assert REVISED_SYSTEM_PROMPT != SYSTEM_PROMPT, "news_ab: revised prompt identical to base"

# Classifier names accepted by BT_NEWS_CLASSIFIER and stored as columns.
CLASSIFIERS = ('regex', 'haiku', 'haiku_revised')
_VERDICT_COL = {
    'regex': 'regex_catalyst',
    'haiku': 'haiku_catalyst',
    'haiku_revised': 'haiku_revised_catalyst',
}


def fetch_alpaca_news(symbol: str, trade_date: str, api_key: str,
                      api_secret: str, limit: int = 10) -> List[Dict]:
    """Fetch news articles for `symbol` published before the trade decision.

    Time-bounded prev-day 21:00 UTC -> trade-day 20:00 UTC (= prev 16:00 ET ->
    15:00 ET) — the SAME window as backtest.py::_fetch_news_for_date, so there
    is no look-ahead bias.

    Returns raw article dicts (headline, summary, created_at, source). Returns
    [] on any error, logged at WARNING — never a silent failure.
    """
    import requests
    prev = (datetime.strptime(trade_date, '%Y-%m-%d')
            - timedelta(days=1)).strftime('%Y-%m-%d')
    start = f"{prev}T21:00:00Z"
    end = f"{trade_date}T20:00:00Z"
    url = ("https://data.alpaca.markets/v1beta1/news?"
           f"symbols={symbol}&start={start}&end={end}&limit={limit}&sort=desc")
    headers = {'APCA-API-KEY-ID': api_key, 'APCA-API-SECRET-KEY': api_secret}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json().get('news', []) or []
    except Exception as e:
        logger.warning(f"{symbol} {trade_date}: Alpaca news fetch failed: {e}")
        return []


def classify_day_catalyst(articles: List[Dict], symbol: str, analyzer,
                          real_cats, stock_context: Optional[Dict] = None
                          ) -> Tuple[bool, str, str, int]:
    """Classify a day's articles with an LLMNewsAnalyzer.

    Mirrors backtest._has_real_catalyst semantics: a day "has a catalyst" if
    ANY article is flagged catalyst=True with a category in `real_cats`.
    Early-exits on the first such article to minimise Haiku calls.

    Returns (has_catalyst, category, top_headline, n_llm_calls).
    """
    best_cat, top_hl = 'NO_NEWS', ''
    n_calls = 0
    for art in articles:
        hl = (art.get('headline') or '').strip()
        if not hl:
            continue
        if not top_hl:
            top_hl = hl
        cat_bool, category, _reason = analyzer.classify(
            art, symbol=symbol, stock_context=stock_context)
        n_calls += 1
        if cat_bool and category in real_cats:
            return True, category, hl, n_calls
        best_cat = category
    return False, best_cat, top_hl, n_calls


class NewsABStore:
    """Read/write access to data/news_ab.db — precomputed news-classifier A/B
    verdicts keyed (symbol, trade_date).

    Safe for concurrent backtest workers: 30s busy-timeout + WAL journal so
    readers never block writers and writers wait on the lock rather than
    erroring (CLAUDE.md: assume the DB may be locked by a parallel process).
    """

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, timeout=30.0)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._ensure_schema(self._conn)
        return self._conn

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        """Create the news_ab + news_ab_articles tables if absent."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_ab (
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                n_articles INTEGER,
                regex_catalyst INTEGER,
                haiku_catalyst INTEGER,
                haiku_revised_catalyst INTEGER,
                regex_category TEXT,
                haiku_category TEXT,
                haiku_revised_category TEXT,
                top_headline TEXT,
                classified_at TEXT,
                PRIMARY KEY (symbol, trade_date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_ab_articles (
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                idx INTEGER NOT NULL,
                article_time TEXT,
                headline TEXT,
                summary TEXT,
                source TEXT,
                PRIMARY KEY (symbol, trade_date, idx)
            )
        """)
        conn.commit()

    def get_verdict(self, symbol: str, trade_date: str,
                    classifier: str) -> Optional[bool]:
        """Return the precomputed catalyst bool for one classifier.

        Returns None if (symbol, trade_date) was never precomputed — the caller
        must treat that as an error (re-run the precompute), not as 'no news'.
        """
        col = _VERDICT_COL.get(classifier)
        if col is None:
            raise ValueError(f"unknown classifier '{classifier}'")
        row = self._connect().execute(
            f"SELECT {col} FROM news_ab WHERE symbol=? AND trade_date=?",
            (symbol, trade_date)).fetchone()
        if row is None or row[0] is None:
            return None
        return bool(row[0])

    def is_done(self, symbol: str, trade_date: str) -> bool:
        """True if all three verdicts are already stored for (symbol, date)."""
        row = self._connect().execute(
            "SELECT regex_catalyst, haiku_catalyst, haiku_revised_catalyst "
            "FROM news_ab WHERE symbol=? AND trade_date=?",
            (symbol, trade_date)).fetchone()
        return row is not None and all(v is not None for v in row)

    def get_articles(self, symbol: str, trade_date: str) -> List[Dict]:
        """Return cached raw articles for (symbol, date), ordered. [] if none."""
        rows = self._connect().execute(
            "SELECT article_time, headline, summary, source FROM news_ab_articles "
            "WHERE symbol=? AND trade_date=? ORDER BY idx",
            (symbol, trade_date)).fetchall()
        return [{'created_at': r[0], 'headline': r[1], 'summary': r[2],
                 'source': r[3]} for r in rows]

    def save_articles(self, symbol: str, trade_date: str,
                      articles: List[Dict]) -> None:
        """Persist raw fetched articles so a re-classify needs no Alpaca call."""
        conn = self._connect()
        for idx, art in enumerate(articles):
            conn.execute(
                "INSERT OR REPLACE INTO news_ab_articles "
                "(symbol, trade_date, idx, article_time, headline, summary, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (symbol, trade_date, idx,
                 (art.get('created_at') or '')[:30],
                 (art.get('headline') or '')[:500],
                 (art.get('summary') or '')[:1000],
                 (art.get('source') or '')[:50]))
        conn.commit()

    def upsert(self, symbol: str, trade_date: str, n_articles: int,
               regex_catalyst: bool, haiku_catalyst: bool,
               haiku_revised_catalyst: bool, regex_category: str,
               haiku_category: str, haiku_revised_category: str,
               top_headline: str) -> None:
        """Write/replace the full A/B verdict row for (symbol, date)."""
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO news_ab "
            "(symbol, trade_date, n_articles, regex_catalyst, haiku_catalyst, "
            " haiku_revised_catalyst, regex_category, haiku_category, "
            " haiku_revised_category, top_headline, classified_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (symbol, trade_date, n_articles,
             int(regex_catalyst), int(haiku_catalyst), int(haiku_revised_catalyst),
             regex_category, haiku_category, haiku_revised_category,
             (top_headline or '')[:300],
             datetime.now(timezone.utc).isoformat(timespec='seconds')))
        conn.commit()

    def all_rows(self) -> List[sqlite3.Row]:
        """Return every news_ab row (for --reclassify and analysis)."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM news_ab").fetchall()
        conn.row_factory = None
        return rows

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
