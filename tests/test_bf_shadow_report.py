"""BF P1 shadow report: pure parsers over journal lines + DB rows."""
import importlib.util, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('bf_shadow_report', os.path.join(ROOT, 'scripts', 'bf_shadow_report.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

LINES = [
    "Sep 08 09:41:02 host python[1]: 2026-09-08 09:41:02 [INFO] trading.trading_engine: ABCD: Conviction 2.10x (pole=6.1%, ...)",
    "Sep 08 09:41:02 host python[1]: [INFO] trading.trading_engine: ABCD: VWAP GATE [SHADOW] would skip — vwap_dist -1.20% <= +0.00% (breakout at/below VWAP)",
    "Sep 08 09:41:02 host python[1]: [INFO] trading.trading_engine: ABCD: REGIME C1 mult=1.50 → shares 100 → 150",
    "Sep 08 09:52:10 host python[1]: [INFO] trading.trading_engine: EFGH: Conviction 1.90x (...)",
    "Sep 08 09:52:10 host python[1]: [INFO] trading.trading_engine: EFGH: CONSISTENCY RULES [SHADOW] would skip — pole 4.2% < 5%; price $22.10 > $20",
    "Sep 08 10:03:00 host python[1]: [INFO] trading.trading_engine: IJKL: CONVICTION SKIP: 1.20 < 1.80 (...)",
    "Sep 08 10:20:00 host python[1]: [INFO] trading.trading_engine: MNOP: Conviction 2.30x (...)",
    "Sep 08 10:31:00 host python[1]: [INFO] trading.trading_engine: MNOP: PROFIT PARTIAL [SHADOW] would sell 50% — closed-bar high $5.40 reached +2.0R (no order; shadow window)",
    "Sep 08 10:31:01 host python[1]: [ERROR] trading.bf_profit_partial: something broke",
]


def test_parse_decisions():
    d = mod.parse_shadow_lines(LINES)
    assert d['ABCD']['seen'] and 'below VWAP' in d['ABCD']['gate_skip'] and d['ABCD']['regime_mult'] == 1.5
    assert d['EFGH']['rules_skip'].startswith('pole 4.2%') and d['EFGH']['gate_skip'] is None
    assert d['IJKL']['seen'] and d['IJKL']['gate_skip'] is None
    assert d['MNOP']['partial'] is True and d['MNOP']['regime_mult'] == 1.0
    assert len(d['__errors__']['lines']) == 1


def test_p1_counterfactual():
    d = mod.parse_shadow_lines(LINES); d.pop('__errors__')
    trades = [
        {'symbol': 'ABCD', 'strategy': 'bull_flag', 'pnl': 300.0},   # would-skip → 0
        {'symbol': 'MNOP', 'strategy': 'bull_flag', 'pnl': -90.0},   # kept, partial would fire
        {'symbol': 'QRST', 'strategy': 'bull_flag', 'pnl': 150.0},   # no lines: kept at 1.0x
        {'symbol': 'ORBX', 'strategy': 'orb', 'pnl': 999.0},         # not BF
    ]
    rows = {r['symbol']: r for r in mod.p1_counterfactual(d, trades)}
    assert set(rows) == {'ABCD', 'MNOP', 'QRST'}
    assert rows['ABCD']['p1_skip'] and rows['ABCD']['p1_pnl'] == 0.0
    assert rows['MNOP']['partial_would_fire'] and rows['MNOP']['p1_pnl'] == -90.0
    assert rows['QRST']['p1_pnl'] == 150.0 and not rows['QRST']['seen']


def test_regime_divides_out():
    d = mod.parse_shadow_lines(["x: WXYZ: Conviction 2.0x", "x: WXYZ: REGIME A mult=1.25 → shares 80 → 100"])
    d.pop('__errors__')
    rows = mod.p1_counterfactual(d, [{'symbol': 'WXYZ', 'strategy': 'bull_flag', 'pnl': 250.0}])
    assert rows[0]['p1_pnl'] == 200.0
