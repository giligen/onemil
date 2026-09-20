"""The `--book {hod_break,red_to_green}` option on the HOD-break reporting scripts (2026-09-17).

Covers, with mocked journal text and small synthetic bars:
  * `scripts/book_spec.py` — argv parsing, the two books' identity, the red-to-green spec (detect/simulate) and the
    HOD book's delegation to `trading.hod_break.simulate` (the default path must stay the SAME function);
  * `scripts/hod_break_eod_check.py` — the per-book journal regexes, the strategy-scoped trades-DB read, the
    red-to-green band table, and a GOLDEN of the default book's 16 measurable rows (the unchanged-output test);
  * `scripts/hod_break_miss_audit.py` — the per-book journal tag and the per-book mover screen;
  * `scripts/hod_break_deadman_flat.py` — a red-to-green row is flattened with an `r2g-dm-` client id and the
    HOD-break rows are never read.
"""
import importlib.util
import json
import sqlite3
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))

from trading.hod_break import HodBreakParams, simulate as hod_simulate  # noqa: E402
from trading.red_to_green import RedToGreenParams                       # noqa: E402

import book_spec                                                        # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


eod = _load('hod_break_eod_check')
audit = _load('hod_break_miss_audit')
deadman = _load('hod_break_deadman_flat')

HOD_CFG = {'book': 'hod_break', 'dry_run': True, 'risk_usd': 100.0, 'min_price': 20.0, 'min_adv20': 100_000.0,
           'universe_min_prev_close': 17.0, 'params': {'max_per_day': 12, 'last_entry_minute': 840}}
R2G_CFG = {'book': 'red_to_green', 'dry_run': True, 'risk_usd': 100.0, 'min_price': 5.0, 'min_adv20': 100_000.0,
           'universe_min_prev_close': 5.0, 'params': {'pdr_min_pct': 8.0, 'range_floor_pct': 5.0, 'level_buffer': 0.003,
                                                      'cap': 0.006, 'min_r_pct': 1.0, 'target_r': 2.0, 'max_per_day': 12,
                                                      'max_concurrent': 4, 'last_entry_minute': 840, 'flat_minute': 955}}
PREV = {'AAA': (10.00, 11.00, 10.00),        # prior close 10.00, prior-day range 10% -> eligible
        'QUIET': (10.00, 10.30, 10.00),      # prior-day range 3% -> never streamed, never a candidate
        'LOW': (2.00, 2.40, 2.00)}           # eligible on range, below every price floor


def r2g_book(prev=None):
    return book_spec.load_book('red_to_green', cfg=R2G_CFG, prev_day=prev if prev is not None else PREV)


def hod_book():
    return book_spec.load_book('hod_break', cfg=HOD_CFG)


# --------------------------------------------------------------------------- book_spec: argv + identity
def test_book_from_argv_default_and_values():
    argv = ['2026-09-17']
    assert book_spec.book_from_argv(argv) == 'hod_break' and argv == ['2026-09-17']
    argv = ['2026-09-17', '--book', 'red_to_green']
    assert book_spec.book_from_argv(argv) == 'red_to_green' and argv == ['2026-09-17']   # the day stays positional
    argv = ['--book', 'hod_break', '--floor', '20']
    assert book_spec.book_from_argv(argv) == 'hod_break' and argv == ['--floor', '20']


def test_book_from_argv_rejects_unknown_and_missing():
    with pytest.raises(SystemExit):
        book_spec.book_from_argv(['--book', 'nonsense'])
    with pytest.raises(SystemExit):
        book_spec.book_from_argv(['--book'])
    with pytest.raises(ValueError):
        book_spec.load_book('nonsense', cfg={})


def test_identity_per_book():
    h, r = hod_book(), r2g_book()
    assert (h.tag, h.strategy, h.coid_prefix, h.is_r2g) == ('HOD', 'hod_break', 'hod', False)
    assert (r.tag, r.strategy, r.coid_prefix, r.is_r2g) == ('R2G', 'red_to_green', 'r2g', True)
    assert isinstance(h.params, HodBreakParams) and isinstance(r.params, RedToGreenParams)
    assert r.params.pdr_min_pct == 8.0 and r.params.target_r == 2.0 and h.params.max_per_day == 12


def test_journal_line_selects_only_its_own_book():
    h, r = hod_book(), r2g_book()
    hod_line = 'host x[1]: [HOD DRY] WOULD BUY AAA level 25.00'
    r2g_line = 'host x[1]: [R2G DRY] WOULD BUY AAA level 10.03'
    err = 'host x[1]: ERROR trading.red_to_green boom'
    assert h.journal_line(hod_line) and not h.journal_line(r2g_line)
    assert r.journal_line(r2g_line) and not r.journal_line(hod_line)
    assert r.journal_line(err) and not h.journal_line(err)


def test_prior_and_level_and_screen():
    r = r2g_book()
    assert r.prior('AAA') == (10.0, pytest.approx(10.0))
    assert r.level('AAA') == pytest.approx(10.03)
    assert r.eligible_prior_day('AAA') and not r.eligible_prior_day('QUIET') and not r.eligible_prior_day('MISSING')
    assert hod_book().level('AAA') is None and hod_book().eligible_prior_day('QUIET')     # no prior-day rule in the HOD book
    adv = {'AAA': 500_000, 'QUIET': 500_000, 'LOW': 500_000, 'THIN': 10_000}
    last = {'AAA': 9.5, 'QUIET': 9.5, 'LOW': 2.0, 'THIN': 40.0}
    assert r.screen_universe(adv, last) == {'AAA'}                    # QUIET fails pdr, LOW fails prev close, THIN fails ADV
    assert hod_book().screen_universe(adv, last) == set()             # nothing clears prev close >= 17 AND ADV >= 100K here


# --------------------------------------------------------------------------- book_spec: the red-to-green spec
def _r2g_tape(bar2_low=9.90, bar3_open=10.05, last_close=12.20, minutes=None):
    """AAA: opens 9.50 under the 10.00 prior close, dips to 9.00 (range 6.1% before the signal bar), bar 2 reaches
    the 10.03 level, bar 3 opens 10.05 (<= the 10.09 cap) -> fill 10.05, stop 9.00, R 1.05, target 12.15."""
    o = [9.50, 9.55, 9.95, bar3_open, 11.00]
    h = [9.55, 9.55, 10.05, 10.50, 12.30]
    l = [9.50, 9.00, bar2_low, 10.00, 10.10]
    c = [9.52, 9.05, 10.00, 10.40, last_close]
    v = [1000.0] * 5
    m = minutes or [570, 571, 572, 573, 574]
    return (np.array(o), np.array(h), np.array(l), np.array(c), np.array(v), np.array(m))


def test_r2g_detect_level_stop_and_bar():
    o, h, l, c, v, m = _r2g_tape()
    sig = r2g_book().detect('AAA', o, h, l, v, m, 500_000.0)
    assert sig.bar_idx == 2 and sig.level == pytest.approx(10.03) and sig.stop == pytest.approx(9.00)
    assert sig.dist_open_pct == pytest.approx((10.03 / 9.50 - 1) * 100)


def test_r2g_simulate_target_exit():
    o, h, l, c, v, m = _r2g_tape()
    t = r2g_book().simulate('AAA', o, h, l, c, v, m, 500_000.0)
    assert t.entry_idx == 3 and t.entry == pytest.approx(10.05) and t.stop == pytest.approx(9.00)
    assert t.r_per_share == pytest.approx(1.05) and t.target == pytest.approx(12.15)
    assert t.reason == 'target' and t.rr == pytest.approx(2.0) and t.exit_idx == 4


def test_r2g_no_signal_when_preconditions_fail():
    b = r2g_book()
    o, h, l, c, v, m = _r2g_tape()
    assert b.detect('QUIET', o, h, l, v, m, 500_000.0) is None          # prior-day range 3% < 8%
    assert b.detect('MISSING', o, h, l, v, m, 500_000.0) is None        # no prior day at all
    up = (np.array([10.50, 9.55, 9.95, 10.05, 11.00]), h, l, c, v, m)   # opens ABOVE the prior close: not red-to-green
    assert b.detect('AAA', up[0], h, l, v, m, 500_000.0) is None
    # the level is reached at bar 1, before the day has ranged 5%: the book does NOT take it, and it never comes back
    early = (np.array([9.80, 9.84, 9.95, 9.85, 9.80]), np.array([9.85, 10.05, 9.90, 9.88, 9.87]),
             np.array([9.78, 9.70, 9.60, 9.55, 9.50]), np.array([9.82, 9.88, 9.80, 9.70, 9.60]))
    assert b.detect('AAA', early[0], early[1], early[2], v, m, 500_000.0) is None
    late = [841, 842, 843, 844, 845]                                    # every bar past the 14:00 cutoff
    assert b.detect('AAA', o, h, l, v, np.array(late), 500_000.0) is None


def test_r2g_no_chase_and_r_floor():
    b = r2g_book()
    o, h, l, c, v, m = _r2g_tape(bar3_open=10.20)                       # next open above level x 1.006 = 10.09
    assert b.detect('AAA', o, h, l, v, m, 500_000.0) is not None and b.simulate('AAA', o, h, l, c, v, m, 500_000.0) is None
    o, h, l, c, v, m = _r2g_tape(bar2_low=9.00)                         # same tape; now tighten the stop below min_r_pct
    tight = book_spec.load_book('red_to_green', cfg={**R2G_CFG, 'params': {**R2G_CFG['params'], 'min_r_pct': 50.0}}, prev_day=PREV)
    assert tight.simulate('AAA', o, h, l, c, v, m, 500_000.0) is None


def test_hod_simulate_delegates_unchanged():
    """The default book must be the SAME function the scripts always called."""
    rng = np.random.default_rng(7)
    base = 30.0 * (1 + np.cumsum(rng.normal(0, 0.002, 60)))
    o = base; h = base * 1.004; l = base * 0.996; c = base; v = np.full(60, 50_000.0)
    m = np.arange(570, 630)
    h[25:] *= 1.08                                                       # a late run so a break exists
    b = hod_book()
    assert b.simulate('X', o, h, l, c, v, m, 3_000_000.0) == hod_simulate(o, h, l, c, v, m, 3_000_000.0, b.params)
    assert b.detect('X', o, h, l, v, m, 3_000_000.0) == __import__('trading.hod_break', fromlist=['detect']).detect(o, h, l, v, m, 3_000_000.0, b.params)


# --------------------------------------------------------------------------- the EOD check
DAY = '2026-09-17'
TS = f'{DAY}T14:31:02+0000 host onemil-trader[1]: {DAY} 14:31:02 | INFO | trading.hod_break_engine:1 | '
R2G_LINES = [
    TS + '[R2G DRY] WOULD BUY AAA level 10.03 limit 10.09 stop 9.00 target 12.15 R 1.05 (10.4%) x95 | +5.6% from open, rv 1.8, spread 12 bps, ask 10.05, prior close 10.00, pdr 10.0%',
    TS + '[R2G] BBB: ask 40.40 above cap 40.18 — NO CHASE, skip',
    TS + '[HOD DRY] WOULD BUY ZZZ level 25.00 limit 25.15 stop 24.50 target 26.20 R 0.60 (2.4%) x166 | +6.1% from open, rv 1.8, spread 12 bps, ask 25.10',
]


def test_eod_regexes_are_per_book():
    m = eod.rx_dry('R2G').search(R2G_LINES[0])
    assert m.group(1) == 'AAA' and float(m.group(2)) == 10.03 and float(m.group(4)) == 9.00 and int(m.group(8)) == 95
    assert eod.rx_dry('HOD').search(R2G_LINES[0]) is None                # the HOD regex must not eat an R2G line
    assert eod.rx_dry('HOD').search(R2G_LINES[2]).group(1) == 'ZZZ'
    assert eod.rx_reject('R2G').search(R2G_LINES[1]).group(1) == 'BBB'
    assert eod.rx_reject('HOD').search(R2G_LINES[1]) is None
    assert eod.RX_DRY.pattern == eod.rx_dry('HOD').pattern and eod.RX_BUY.pattern == eod.rx_buy('HOD').pattern


@pytest.fixture
def two_book_db(tmp_path):
    p = tmp_path / 'trades.db'; con = sqlite3.connect(p)
    con.execute('create table trades (id integer primary key, strategy text, trade_date text, symbol text, fill_price real, entry_price real, '
                'stop_loss_price real, exit_price real, exit_reason text, pnl real, shares integer, filled_qty integer, filled_at text, pattern_data text, order_status text)')
    con.execute("insert into trades(strategy, trade_date, symbol, fill_price, stop_loss_price, exit_price, exit_reason, pnl, shares, filled_qty, filled_at, pattern_data, order_status) "
                "values ('hod_break', ?, 'ZZZ', 25.10, 24.50, 26.30, 'target', 199.2, 166, 166, ?, '{}', 'closed')", (DAY, f'{DAY}T14:31:03+00:00'))
    con.execute("insert into trades(strategy, trade_date, symbol, fill_price, stop_loss_price, exit_price, exit_reason, pnl, shares, filled_qty, filled_at, pattern_data, order_status) "
                "values ('red_to_green', ?, 'AAA', 10.05, 9.00, 12.15, 'target', 199.5, 95, 95, ?, '{}', 'closed')", (DAY, f'{DAY}T14:31:03+00:00'))
    con.commit(); con.close(); return p


def test_closed_trades_is_scoped_to_the_book(two_book_db):
    assert [r['symbol'] for r in eod.closed_trades(two_book_db, DAY, DAY)] == ['ZZZ']              # default = hod_break
    assert [r['symbol'] for r in eod.closed_trades(two_book_db, DAY, DAY, 'red_to_green')] == ['AAA']


def test_r2g_measurables_use_the_r2g_strategy_rows(two_book_db):
    rows = {r[0]: r for r in eod.live_measurables(R2G_LINES, DAY, False, two_book_db, {}, 955, since=DAY, strategy='red_to_green', price_floor=5.0)}
    assert rows[1][1] == 'gate pass rate, $5+ signals' and rows[15][1] == 'spread at decision, passing $5+'
    assert rows[3][2].startswith('1 today')                              # only the red_to_green row is counted
    assert rows[11][2] == '+2.000R over 1 trades'                        # (12.15 - 10.05) / (10.05 - 9.00)
    assert '--book red_to_green' in rows[14][2]


def test_r2g_bands_replace_hod_bands_and_drop_their_verdicts(two_book_db):
    rows = eod.live_measurables(R2G_LINES, DAY, False, two_book_db, {}, 955, since=DAY, strategy='red_to_green', price_floor=5.0)
    banded = {r[0]: r for r in eod.r2g_bands(rows)}
    raw = {r[0]: r for r in rows}
    for no in (1, 3, 7, 8, 11, 12, 13, 15, 16):
        assert banded[no][3] == eod.R2G_BANDS[no] and banded[no][4] == 'n/a'
    for no in eod.R2G_KEEP_VERDICT:                                      # mechanical rows keep the engine's verdict
        assert banded[no][3] == raw[no][3] and banded[no][4] == raw[no][4]


HOD_GOLDEN = [                                                           # the DEFAULT book's rows must not move
    (1, 'gate pass rate, $20+ signals', 'no signal reached the gate', 'WATCH n<20'),
    (2, 'signals/day reaching the gates, $20+', '1 today', 'WATCH <5'),
    (14, 'miss rate vs spec', '0 MISSED lines today · full audit: scripts/hod_break_miss_audit.py', 'OK'),
    (15, 'spread at decision, passing $20+', 'no passing signal', 'n/a'),
    (16, 'r_min reject rate, $20+ signals', '0/1 = 0%', 'OK'),
]


def test_default_book_output_unchanged(two_book_db):
    """Unchanged-output test for the default path: same labels, same values, same verdicts as before the option."""
    lines = [TS + '[HOD] BBB: ask 40.40 above cap 40.18 — NO CHASE, skip']
    rows = {r[0]: r for r in eod.live_measurables(lines, DAY, True, two_book_db, {}, 955, since=DAY)}
    for no, lab, val, verdict in HOD_GOLDEN:
        assert (rows[no][1], rows[no][2], rows[no][4]) == (lab, val, verdict)
    assert all(rows[k][2] == 'n/a dry' for k in range(3, 14))
    assert len(rows) == 16


# --------------------------------------------------------------------------- the miss audit
def test_journal_events_reads_only_its_own_tag(monkeypatch):
    out = ('2026-09-17T14:00:00+0000 h x[1]: [R2G] AAA: level 10.03 below the $5 floor — skip\n'
           '2026-09-17T14:02:00+0000 h x[1]: [R2G] BBB: MISSED the spec\'s break at bar 12 (level 40.00, now 3 bars old)\n'
           '2026-09-17T13:31:00+0000 h x[1]: [R2G] streaming 338 universe symbols (prev close >= 5.00)\n')

    def fake_run(cmd, *a, **k):
        assert '[R2G' in cmd[-1]                                        # the grep is built from the book's tag
        return types.SimpleNamespace(stdout=out)
    monkeypatch.setattr(audit.subprocess, 'run', fake_run)
    admitted, evaluated, stale, stream_start = audit.journal_events(tag='R2G')
    assert list(evaluated) == ['AAA'] and list(stale) == ['BBB'] and stream_start == 9 * 60 + 31


def test_movers_screen_per_book():
    day = {'AAA': {'open': 9.50, 'high': 10.20, 'low': 9.00, 'close': 10.10},    # gap down, level reached, range 13%
           'QUIET': {'open': 9.50, 'high': 10.20, 'low': 9.00, 'close': 10.10},  # prior-day range 3%: not in the book
           'NOGAP': {'open': 10.50, 'high': 11.20, 'low': 10.00, 'close': 11.00},
           'LOW': {'open': 1.80, 'high': 2.10, 'low': 1.70, 'close': 2.00}}
    prev = {**PREV, 'NOGAP': (10.00, 11.00, 10.00)}
    assert audit.movers_for(r2g_book(prev), day, 5.0) == ['AAA']                 # NOGAP opens above the prior close, LOW under the floor
    # the HOD screen is the day-running distance from the open, with no prior-day input at all
    assert sorted(audit.movers_for(hod_book(), day, 5.0)) == ['AAA', 'NOGAP', 'QUIET']   # LOW is under the $5 floor
    hod_day = {'X': {'open': 10.0, 'high': 10.6, 'low': 9.9}, 'Y': {'open': 10.0, 'high': 10.2, 'low': 9.9}}
    assert audit.movers_for(hod_book(), hod_day, 5.0) == ['X']


# --------------------------------------------------------------------------- the dead-man flat
class _FakeAlpaca:
    def __init__(self, *a, **k): self.sold = []
    def get_market_calendar(self, a, b): return []
    def get_open_positions(self): return [{'symbol': 'AAA', 'qty': '95'}]
    def get_latest_quote(self, s): return {'bid_price': 11.0, 'ask_price': 11.05}
    def submit_limit_sell_order(self, sym, qty, px, client_order_id=None):
        self.sold.append((sym, qty, px, client_order_id)); return {'id': 'ord-1'}
    def cancel_order(self, oid): raise AssertionError('no legs to cancel')
    def get_order(self, oid): raise AssertionError('no legs to read')


class _FakeDb:
    def __init__(self, rows): self.rows = rows; self.asked = []; self.updates = []
    def get_open_trades(self, day, strategy=None):
        self.asked.append(strategy); return [r for r in self.rows if r['strategy'] == strategy]
    def update_trade(self, tid, kv): self.updates.append((tid, kv))


def test_deadman_flattens_the_chosen_book_only(monkeypatch):
    rows = [{'id': 1, 'strategy': 'red_to_green', 'symbol': 'AAA', 'shares': 95, 'fill_price': 10.05,
             'order_status': 'filled', 'pattern_data': json.dumps({})},
            {'id': 2, 'strategy': 'hod_break', 'symbol': 'ZZZ', 'shares': 10, 'fill_price': 25.0,
             'order_status': 'filled', 'pattern_data': json.dumps({})}]
    db = _FakeDb(rows); alp = _FakeAlpaca()
    monkeypatch.setattr(deadman, 'Database', lambda *a, **k: db)
    monkeypatch.setattr(deadman, 'AlpacaClient', lambda *a, **k: alp)
    monkeypatch.setattr(deadman, 'Config', lambda: types.SimpleNamespace(alpaca_api_key='k', alpaca_api_secret='s', alpaca_paper=True))
    monkeypatch.setattr(deadman.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(stdout='inactive'))
    monkeypatch.setattr(sys, 'argv', ['deadman', '--force', '--book', 'red_to_green'])
    assert deadman.main() == 0
    assert db.asked == ['red_to_green']                                   # the HOD-break rows are never even read
    sym, qty, px, coid = alp.sold[0]
    assert (sym, qty) == ('AAA', 95) and coid.startswith('r2g-dm-AAA-') and len(alp.sold) == 1
    assert db.updates[0][1]['exit_reason'] == 'deadman_flat'


def test_deadman_defaults_to_hod_break(monkeypatch):
    db = _FakeDb([]); alp = _FakeAlpaca()
    monkeypatch.setattr(deadman, 'Database', lambda *a, **k: db)
    monkeypatch.setattr(deadman, 'AlpacaClient', lambda *a, **k: alp)
    monkeypatch.setattr(deadman, 'Config', lambda: types.SimpleNamespace(alpaca_api_key='k', alpaca_api_secret='s', alpaca_paper=True))
    monkeypatch.setattr(deadman.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(stdout='inactive'))
    monkeypatch.setattr(sys, 'argv', ['deadman', '--force'])
    assert deadman.main() == 0 and db.asked == ['hod_break'] and alp.sold == []


def test_tags_and_last_bar_signal_and_real_config(tmp_path):
    h, r = hod_book(), r2g_book()
    assert (h.dry_tag, h.live_tag) == ('[HOD DRY]', '[HOD]') and (r.dry_tag, r.live_tag) == ('[R2G DRY]', '[R2G]')
    o, h_, l, c, v, m = _r2g_tape()
    # the signal lands on the LAST bar: there is no next open to fill at, so the spec has no trade
    assert r.simulate('AAA', o[:3], h_[:3], l[:3], c[:3], v[:3], m[:3], 500_000.0) is None
    # the real config.yaml blocks: both books load and carry their shipped knobs
    live_r = book_spec.load_book('red_to_green', prev_day={})
    assert live_r.strategy == 'red_to_green' and live_r.params.max_per_day == 12 and live_r.cfg['min_price'] == 5.0
    assert book_spec.load_book().strategy == 'hod_break'


def test_load_prev_day_reads_daily_bars(tmp_path):
    """The loader is the engine's own — a temp cache.db with one daily row comes back as (close, high, low)."""
    p = tmp_path / 'cache.db'; con = sqlite3.connect(p)
    con.execute('create table daily_bars (symbol text, bar_date text, open real, high real, low real, close real, volume real)')
    import datetime as dt
    from trading.hod_break_engine import ET
    # The loader's "today" is the ET date, not the box's UTC date — between 20:00 and 24:00 ET the
    # two differ and a UTC-derived yesterday IS the ET today, which the loader excludes by design.
    yday = (dt.datetime.now(dt.timezone.utc).astimezone(ET).date() - dt.timedelta(days=1)).isoformat()
    con.execute('insert into daily_bars values (?,?,?,?,?,?,?)', ('AAA', yday, 10.5, 11.0, 10.0, 10.0, 1e6))
    con.commit(); con.close()
    assert book_spec.load_prev_day(p)['AAA'] == (10.0, 11.0, 10.0)
