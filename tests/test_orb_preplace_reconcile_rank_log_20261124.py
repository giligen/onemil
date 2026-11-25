"""ORB preplace reconcile logs the WHOLE final ranking (quintile, composite), including the names past the
top-K (2026-11-24 SKDD: preplaced Q4, re-ranked Q3 on its final range, 9th behind a rangeless stand-in; the
top-K-only line made the cancel look unexplained)."""
from __future__ import annotations

import logging
import re

from tests.test_orb_preplace_budget_20261006 import (  # noqa: F401  (fixture re-export)
    eng, _go_final, _preplace)

CANCELED = {'status': 'canceled', 'filled_qty': 0}
PREFIX = '[ORB PREPLACE] reconcile ranking (Q, comp):'


def _ranking_record(caplog):
    recs = [r for r in caplog.records if r.getMessage().startswith(PREFIX)]
    assert len(recs) == 1, [r.getMessage() for r in caplog.records if 'reconcile' in r.getMessage()]
    return recs[0].getMessage()


def test_every_scored_name_appears_with_quintile_and_composite(eng, caplog):
    eng.alpaca.get_order.return_value = CANCELED
    _preplace(eng, ok=('JAGX',))
    _go_final(eng)
    scored = eng._rank_production_final()
    assert scored, "harness must rank at least one name"
    with caplog.at_level(logging.INFO):
        eng._reconcile_preplaced()
    msg = _ranking_record(caplog)
    assert f"beyond top-{eng.max_concurrent}:" in msg
    for c in scored:
        assert re.search(rf"{c.symbol}\(Q[1-5],-?\d+\.\d{{4}}\)", msg), (c.symbol, msg)


def test_names_past_top_k_land_in_the_beyond_part(eng, caplog):
    eng.alpaca.get_order.return_value = CANCELED
    _preplace(eng, ok=('JAGX',))
    _go_final(eng)
    eng.max_concurrent = 1                       # force a long tail past the top-K
    scored = eng._rank_production_final()
    assert len(scored) >= 2
    with caplog.at_level(logging.INFO):
        eng._reconcile_preplaced()
    msg = _ranking_record(caplog)
    head, tail = msg.split('| beyond top-1:')
    assert scored[0].symbol in head
    for c in scored[1:]:
        assert c.symbol in tail and c.symbol not in head.split(':', 1)[1], (c.symbol, msg)


def test_old_top_k_line_unchanged_for_the_parsers(eng, caplog):
    eng.alpaca.get_order.return_value = CANCELED
    _preplace(eng, ok=('JAGX',))
    _go_final(eng)
    with caplog.at_level(logging.INFO):
        eng._reconcile_preplaced()
    old = [r.getMessage() for r in caplog.records
           if r.getMessage().startswith('[ORB PREPLACE] reconcile final top-')]
    assert len(old) == 1 and '| preplaced: [' in old[0]
