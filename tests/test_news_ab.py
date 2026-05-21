"""
Tests for the news-classifier A/B infrastructure (trading/news_ab.py) and the
backtest BT_NEWS_CLASSIFIER router.

Unit:        REVISED_SYSTEM_PROMPT, fetch_alpaca_news, classify_day_catalyst,
             NewsABStore.
Integration: precompute-style flow (mocked Alpaca + mocked Haiku) → store →
             backtest router reads the precomputed verdict end-to-end.
"""
import pytest
from unittest.mock import MagicMock

from data_sources.news_provider import SYSTEM_PROMPT
from trading.news_ab import (
    REVISED_SYSTEM_PROMPT, fetch_alpaca_news, classify_day_catalyst, NewsABStore,
)

# Catalyst categories — mirrors BacktestRunner._REAL_CATS.
REAL_CATS = {'FDA_CLINICAL', 'EARNINGS', 'CONTRACT_DEAL', 'CONTRACT', 'MA',
             'ANALYST', 'PRODUCT', 'PRODUCT_LAUNCH', 'MGMT', 'MANAGEMENT',
             'SEC_FILING', 'CRYPTO'}


def _json_resp(payload):
    """Fake requests.Response with .json() + a no-op .raise_for_status()."""
    r = MagicMock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


class _FakeAnalyzer:
    """LLMNewsAnalyzer stand-in — returns a queued (catalyst, category, reason)."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def classify(self, article, symbol=None, stock_context=None):
        self.calls.append((article.get('headline'), symbol))
        return self._results.pop(0)


# ============================================================================
# REVISED_SYSTEM_PROMPT
# ============================================================================

class TestRevisedPrompt:
    def test_differs_from_production_prompt(self):
        assert REVISED_SYSTEM_PROMPT != SYSTEM_PROMPT

    def test_adds_patent_application_rule(self):
        assert "PATENT APPLICATION" in REVISED_SYSTEM_PROMPT
        assert "NEVER a catalyst" in REVISED_SYSTEM_PROMPT

    def test_keeps_shared_json_instruction(self):
        assert "Reply with ONLY a JSON object" in REVISED_SYSTEM_PROMPT

    def test_importing_news_ab_does_not_mutate_production_prompt(self):
        assert "Reply with ONLY a JSON object" in SYSTEM_PROMPT
        assert "PATENT APPLICATION" not in SYSTEM_PROMPT


# ============================================================================
# fetch_alpaca_news
# ============================================================================

class TestFetchAlpacaNews:
    def test_returns_articles_and_passes_keys(self, monkeypatch):
        captured = {}

        def fake_get(url, headers=None, timeout=None):
            captured['url'] = url
            captured['headers'] = headers
            return _json_resp({'news': [{'headline': 'A'}, {'headline': 'B'}]})

        monkeypatch.setattr('requests.get', fake_get)
        out = fetch_alpaca_news('AAPL', '2026-01-15', 'mykey', 'mysecret')

        assert len(out) == 2
        assert 'symbols=AAPL' in captured['url']
        assert captured['headers']['APCA-API-KEY-ID'] == 'mykey'
        assert captured['headers']['APCA-API-SECRET-KEY'] == 'mysecret'

    def test_time_window_has_no_lookahead(self, monkeypatch):
        captured = {}

        def fake_get(url, headers=None, timeout=None):
            captured['url'] = url
            return _json_resp({'news': []})

        monkeypatch.setattr('requests.get', fake_get)
        fetch_alpaca_news('X', '2026-01-15', 'k', 's')
        # prev-day 21:00 UTC → trade-day 20:00 UTC — never reads later news.
        assert 'start=2026-01-14T21:00:00Z' in captured['url']
        assert 'end=2026-01-15T20:00:00Z' in captured['url']

    def test_network_error_returns_empty(self, monkeypatch):
        def boom(*a, **k):
            raise Exception("network down")

        monkeypatch.setattr('requests.get', boom)
        assert fetch_alpaca_news('X', '2026-01-15', 'k', 's') == []

    def test_missing_news_key_returns_empty(self, monkeypatch):
        monkeypatch.setattr('requests.get', lambda *a, **k: _json_resp({}))
        assert fetch_alpaca_news('X', '2026-01-15', 'k', 's') == []


# ============================================================================
# classify_day_catalyst
# ============================================================================

class TestClassifyDayCatalyst:
    def test_early_exit_on_first_catalyst(self):
        an = _FakeAnalyzer([(True, 'EARNINGS', 'beat')])
        arts = [{'headline': 'Q3 beat'}, {'headline': 'other'}]
        has, cat, hl, n = classify_day_catalyst(arts, 'SYM', an, REAL_CATS)
        assert has is True and cat == 'EARNINGS' and hl == 'Q3 beat'
        assert n == 1  # stopped after the first article — minimises Haiku calls

    def test_no_catalyst_classifies_all(self):
        an = _FakeAnalyzer([(False, 'GARBAGE_RECAP', 'r'), (False, 'OTHER', 'r')])
        arts = [{'headline': 'h1'}, {'headline': 'h2'}]
        has, cat, hl, n = classify_day_catalyst(arts, 'SYM', an, REAL_CATS)
        assert has is False and n == 2

    def test_catalyst_true_but_non_real_category_is_not_a_catalyst(self):
        an = _FakeAnalyzer([(True, 'OTHER', 'r')])
        has, cat, hl, n = classify_day_catalyst(
            [{'headline': 'h'}], 'SYM', an, REAL_CATS)
        assert has is False

    def test_skips_empty_headlines(self):
        an = _FakeAnalyzer([(True, 'MA', 'r')])
        arts = [{'headline': ''}, {'headline': '   '}, {'headline': 'real'}]
        has, cat, hl, n = classify_day_catalyst(arts, 'SYM', an, REAL_CATS)
        assert has is True and hl == 'real' and n == 1

    def test_empty_article_list(self):
        has, cat, hl, n = classify_day_catalyst([], 'SYM', _FakeAnalyzer([]), REAL_CATS)
        assert has is False and n == 0 and hl == ''

    def test_passes_symbol_and_context_to_analyzer(self):
        an = _FakeAnalyzer([(False, 'OTHER', 'r')])
        classify_day_catalyst([{'headline': 'h'}], 'TSLA', an, REAL_CATS,
                              stock_context={'float_shares': 1_000_000})
        assert an.calls == [('h', 'TSLA')]


# ============================================================================
# NewsABStore
# ============================================================================

class TestNewsABStore:
    def _store(self, tmp_path):
        return NewsABStore(str(tmp_path / 'news_ab.db'))

    def test_upsert_and_get_verdict(self, tmp_path):
        s = self._store(tmp_path)
        s.upsert('AAA', '2026-01-02', n_articles=3,
                 regex_catalyst=False, haiku_catalyst=True,
                 haiku_revised_catalyst=False, regex_category='OTHER',
                 haiku_category='PRODUCT_LAUNCH', haiku_revised_category='OTHER',
                 top_headline='Patent filed')
        assert s.get_verdict('AAA', '2026-01-02', 'regex') is False
        assert s.get_verdict('AAA', '2026-01-02', 'haiku') is True
        assert s.get_verdict('AAA', '2026-01-02', 'haiku_revised') is False
        s.close()

    def test_get_verdict_missing_returns_none(self, tmp_path):
        s = self._store(tmp_path)
        assert s.get_verdict('NOPE', '2026-01-02', 'haiku') is None
        s.close()

    def test_get_verdict_unknown_classifier_raises(self, tmp_path):
        s = self._store(tmp_path)
        with pytest.raises(ValueError):
            s.get_verdict('AAA', '2026-01-02', 'bogus')
        s.close()

    def test_is_done(self, tmp_path):
        s = self._store(tmp_path)
        assert s.is_done('AAA', '2026-01-02') is False
        s.upsert('AAA', '2026-01-02', 1, False, False, False,
                 'OTHER', 'OTHER', 'OTHER', 'h')
        assert s.is_done('AAA', '2026-01-02') is True
        s.close()

    def test_upsert_replaces_existing_row(self, tmp_path):
        s = self._store(tmp_path)
        s.upsert('AAA', '2026-01-02', 1, False, False, False,
                 'OTHER', 'OTHER', 'OTHER', 'h')
        s.upsert('AAA', '2026-01-02', 2, True, True, True,
                 'MA', 'MA', 'MA', 'h2')
        assert s.get_verdict('AAA', '2026-01-02', 'regex') is True
        s.close()

    def test_save_and_get_articles(self, tmp_path):
        s = self._store(tmp_path)
        arts = [{'headline': 'h1', 'summary': 's1', 'created_at': 't1',
                 'source': 'src'},
                {'headline': 'h2', 'summary': 's2', 'created_at': 't2',
                 'source': 'src'}]
        s.save_articles('AAA', '2026-01-02', arts)
        got = s.get_articles('AAA', '2026-01-02')
        assert len(got) == 2
        assert got[0]['headline'] == 'h1' and got[1]['summary'] == 's2'
        s.close()

    def test_get_articles_missing_returns_empty(self, tmp_path):
        s = self._store(tmp_path)
        assert s.get_articles('NOPE', '2026-01-02') == []
        s.close()

    def test_persists_across_connections(self, tmp_path):
        path = str(tmp_path / 'news_ab.db')
        s1 = NewsABStore(path)
        s1.upsert('AAA', '2026-01-02', 1, True, False, True,
                  'MA', 'OTHER', 'MA', 'h')
        s1.close()
        s2 = NewsABStore(path)
        assert s2.get_verdict('AAA', '2026-01-02', 'regex') is True
        s2.close()


# ============================================================================
# Integration — precompute flow + backtest BT_NEWS_CLASSIFIER router
# ============================================================================

class TestBacktestRouter:
    def test_no_env_var_uses_production_path(self, monkeypatch):
        monkeypatch.delenv('BT_NEWS_CLASSIFIER', raising=False)
        from backtest import BacktestRunner
        assert BacktestRunner()._news_ab_mode is None

    def test_invalid_classifier_raises(self, monkeypatch):
        monkeypatch.setenv('BT_NEWS_CLASSIFIER', 'gpt9')
        from backtest import BacktestRunner
        with pytest.raises(ValueError):
            BacktestRunner()

    def test_router_reads_precomputed_haiku_verdict(self, monkeypatch, tmp_path):
        monkeypatch.setenv('BT_NEWS_CLASSIFIER', 'haiku')
        from backtest import BacktestRunner
        runner = BacktestRunner()
        store = NewsABStore(str(tmp_path / 'news_ab.db'))
        store.upsert('AAA', '2026-01-02', 1, regex_catalyst=False,
                     haiku_catalyst=True, haiku_revised_catalyst=False,
                     regex_category='OTHER', haiku_category='MA',
                     haiku_revised_category='OTHER', top_headline='h')
        runner._news_ab_store = store
        assert runner._has_real_catalyst('AAA', '2026-01-02') is True

    def test_router_haiku_revised_column_isolated(self, monkeypatch, tmp_path):
        monkeypatch.setenv('BT_NEWS_CLASSIFIER', 'haiku_revised')
        from backtest import BacktestRunner
        runner = BacktestRunner()
        store = NewsABStore(str(tmp_path / 'news_ab.db'))
        # regex + haiku say catalyst; haiku_revised says no (patent-app rule).
        store.upsert('AAA', '2026-01-02', 1, regex_catalyst=True,
                     haiku_catalyst=True, haiku_revised_catalyst=False,
                     regex_category='PRODUCT', haiku_category='PRODUCT_LAUNCH',
                     haiku_revised_category='OTHER', top_headline='Patent filed')
        runner._news_ab_store = store
        assert runner._has_real_catalyst('AAA', '2026-01-02') is False

    def test_router_falls_back_to_regex_when_not_precomputed(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv('BT_NEWS_CLASSIFIER', 'haiku')
        from backtest import BacktestRunner
        runner = BacktestRunner()
        runner._news_ab_store = NewsABStore(str(tmp_path / 'empty.db'))
        monkeypatch.setattr(runner, '_has_real_catalyst_regex',
                            lambda s, d: 'REGEX_FALLBACK')
        assert runner._has_real_catalyst('ZZZ', '2026-01-02') == 'REGEX_FALLBACK'

    def test_precompute_flow_end_to_end(self, monkeypatch, tmp_path):
        """fetch (mock Alpaca) → classify (mock Haiku) → store → router reads."""
        monkeypatch.setattr('requests.get', lambda *a, **k: _json_resp(
            {'news': [{'headline': 'Acme wins $50M defense contract',
                       'summary': 'big deal'}]}))
        articles = fetch_alpaca_news('ACME', '2026-01-02', 'k', 's')
        assert len(articles) == 1

        an = _FakeAnalyzer([(True, 'CONTRACT_DEAL', 'real contract')])
        has, cat, hl, n = classify_day_catalyst(articles, 'ACME', an, REAL_CATS)
        assert has is True

        db = str(tmp_path / 'news_ab.db')
        store = NewsABStore(db)
        store.save_articles('ACME', '2026-01-02', articles)
        store.upsert('ACME', '2026-01-02', len(articles), regex_catalyst=False,
                     haiku_catalyst=has, haiku_revised_catalyst=has,
                     regex_category='OTHER', haiku_category=cat,
                     haiku_revised_category=cat, top_headline=hl)
        store.close()

        monkeypatch.setenv('BT_NEWS_CLASSIFIER', 'haiku')
        from backtest import BacktestRunner
        runner = BacktestRunner()
        runner._news_ab_store = NewsABStore(db)
        assert runner._has_real_catalyst('ACME', '2026-01-02') is True
