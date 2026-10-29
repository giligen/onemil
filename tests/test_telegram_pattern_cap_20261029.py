"""Telegram pattern cap (docs/exit_coverage_invariant_spec_20261029.md Part 2)."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from notifications.telegram_notifier import TelegramNotifier, message_pattern


@pytest.fixture
def n():
    t = TelegramNotifier(bot_token='tok', chat_id='1', enabled=True)
    t._post_message = AsyncMock(return_value=True)
    t._rate_limit_max = 1000
    return t


def _send(n, text, now):
    """Drive send_message with a controlled clock."""
    import notifications.telegram_notifier as mod
    real = mod.time.monotonic
    mod.time.monotonic = lambda: now
    try:
        return asyncio.run(n.send_message(text))
    finally:
        mod.time.monotonic = real


def sent(n):
    return [c.args[0] for c in n._post_message.await_args_list]


def test_pattern_key_normalises_digits_hex_dollars_uuid():
    a = message_pattern('LABX 423 sh $12.50 order 3f2a9c1d77ab')
    b = message_pattern('LABX 17 sh $1,200 order 99ffeedd0011')
    assert a == b == 'LABX # sh # order #'
    assert message_pattern('x 3f2a9c1d-7a1b-4c2d-8e3f-0123456789ab') == message_pattern('x 00000000-0000-0000-0000-000000000001')


def test_fourth_same_pattern_in_ten_minutes_suppressed(n):
    for i in range(6):
        _send(n, f'dead leg retry {i}', now=100 + i * 70)   # distinct text, same pattern, 70 s apart
    assert len(sent(n)) == 3


def test_differing_ids_same_pattern(n):
    ids = ['3f2a9c1d77ab', '99ffeedd0011', 'abcdef012345', '0123456789ab']
    for i, oid in enumerate(ids):
        _send(n, f'scale-out failed order {oid}', now=100 + i * 70)
    assert len(sent(n)) == 3


def test_summary_sent_once_after_window_rolls(n):
    for i in range(6):
        _send(n, f'dead leg retry {i}', now=100 + i * 70)       # 3 sent, 3 suppressed
    _send(n, 'dead leg retry 99', now=100 + 650)                 # first send aged out (>600 s)
    out = sent(n)
    summaries = [t for t in out if t.startswith('⏸')]
    assert len(summaries) == 1 and 'suppressed 3 further "dead leg retry #"' in summaries[0]
    assert out[-1] == 'dead leg retry 99' and out[-2] == summaries[0]
    _send(n, 'dead leg retry 100', now=100 + 700)
    assert len([t for t in sent(n) if t.startswith('⏸')]) == 1


@pytest.mark.parametrize('word', ['EXIT COVERAGE', 'NAKED', 'EMERGENCY', 'CRITICAL', '[MOM LIVE]', 'FILL'])
def test_protected_words_bypass(n, word):
    for i in range(8):
        _send(n, f'{word} event {i}', now=100 + i * 70)
    assert len(sent(n)) == 8


def test_window_reset_allows_three_more(n):
    for i in range(4):
        _send(n, f'poll failed {i}', now=100 + i * 10)
    assert len(sent(n)) == 3
    for i in range(3):
        _send(n, f'poll failed {i + 10}', now=1000 + i * 10)
    assert len(sent(n)) == 3 + 1 + 3     # +1 summary line for the earlier suppressed one


def test_pattern_flood_does_not_consume_rate_budget(n):
    n._rate_limit_max = 5
    for i in range(30):
        _send(n, f'flood {i}', now=100 + i * 0.1)
    assert _send(n, 'something else entirely', now=104) is True
