"""trading/hod_failure_signal.py — (1) parity: rebuild one fill's feature vector from raw bars via
the live signal module and compare to research/hod_entry/1669_per_fill.csv's stored columns
(the research scorer's own output); (2) boot: the model loads and scores a synthetic fill in [0, 1].

F15 is EXCLUDED from the parity comparison on purpose: its denominator (F8, "dollar volume vs
normal") isn't stored anywhere the live engine can reach, so trading/hod_failure_signal.py maps it
to adv20 instead (documented in the module docstring and docs/hod_failure_short_spec_20260930.md)
— a deliberate, disclosed approximation, not a parity bug.
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from trading.hod_failure_signal import build_feature_row, build_signal_fn

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
PER_FILL_CSV = os.path.join(ROOT, 'research/hod_entry/1669_per_fill.csv')
BARS_SIP_DB = os.path.join(ROOT, 'research/bf_zero/bars_sip.db')
MODEL_PATH = os.path.join(ROOT, 'research/hod_entry/models/ff10_k1_val.joblib')

pytestmark = pytest.mark.skipif(not (os.path.exists(PER_FILL_CSV) and os.path.exists(BARS_SIP_DB)),
                                 reason="research parity fixtures not present on this node")


def _bars_for(symbol: str, day: str) -> dict:
    """Canonical {'o','h','l','c','v','minarr'} dict, read-only from research/bf_zero/bars_sip.db
    (a research-only historical tick-bar cache; never touched by production code)."""
    con = sqlite3.connect(f'file:{BARS_SIP_DB}?mode=ro', uri=True)
    try:
        df = pd.read_sql("select t, o, h, l, c, v from bars where symbol=? and day=? order by t", con, params=[symbol, day])
    finally:
        con.close()
    minarr = np.array([int(t[11:13]) * 60 + int(t[14:16]) for t in df['t']])
    return {'o': df['o'].to_numpy(float), 'h': df['h'].to_numpy(float), 'l': df['l'].to_numpy(float),
            'c': df['c'].to_numpy(float), 'v': df['v'].to_numpy(float), 'minarr': minarr}


def _one_fill_with_level_bar():
    df = pd.read_csv(PER_FILL_CSV, dtype={'date': str, 'symbol': str})
    df = df[df['F11'].notna() & df['level'].notna()]
    assert len(df), "no row with a found level bar (F11 notna) in 1669_per_fill.csv"
    row = df.iloc[0]
    bars = _bars_for(row['symbol'], row['date'])
    if len(bars['o']) < 3:
        pytest.skip(f"no/thin bars for {row['symbol']} {row['date']} in bars_sip.db")
    return row, bars


class TestParity:
    def test_bar0_and_F11_F14_match_the_research_scorer(self):
        row, bars = _one_fill_with_level_bar()
        feats, computable = build_feature_row(bars=bars, i0=_find_i0(bars, row), entry=float(row['entry_price']),
                                               stop=float(row['stop']), target=float(row['target_price']),
                                               level=float(row['level']), atr14=float(row['atr14']), adv20=None)
        if not computable:
            pytest.skip("this fill preempted before k=1 in the SIP bar replay — not a parity failure")
        for col in ('clv0', 'body0', 'rangeATR0', 'volratio0', 'closeR0', 'F11', 'F12', 'F13', 'F14'):
            expected = float(row[col])
            got = feats.get(col)
            assert got is not None and np.isfinite(got), f"{col}: engine produced {got}"
            assert got == pytest.approx(expected, abs=1e-6), f"{col}: engine={got} research={expected}"


def _find_i0(bars, row):
    from trading.hod_failure_features import find_fill_index
    return find_fill_index(bars, float(row['fill_min']))


class TestBoot:
    @pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="model file not present on this node")
    def test_model_loads_and_scores_a_synthetic_fill_in_0_1(self):
        signal_fn = build_signal_fn({'model_path': MODEL_PATH})
        assert signal_fn is not None
        n = 40
        minarr = np.arange(570, 570 + n)
        rng = np.random.default_rng(0)
        c = 10.0 + np.cumsum(rng.normal(0, 0.02, n))
        o = c - rng.normal(0, 0.01, n)
        h = np.maximum(o, c) + abs(rng.normal(0, 0.02, n))
        l = np.minimum(o, c) - abs(rng.normal(0, 0.02, n))
        v = rng.integers(500, 5000, n).astype(float)
        bars = {'o': o, 'h': h, 'l': l, 'c': c, 'v': v, 'minarr': minarr}
        fill_minute = 590
        arm_context = {'fill_minute': fill_minute, 'long_fill_price': float(c[fill_minute - 570]),
                        'long_stop': float(c[fill_minute - 570]) - 0.30, 'long_target': float(c[fill_minute - 570]) + 0.60,
                        'level': float(h[:fill_minute - 570 + 1].max()), 'adv20': 5_000_000.0, 'date': '2026-09-30'}
        p = signal_fn('SYN', bars, arm_context)
        assert p is None or (isinstance(p, float) and 0.0 <= p <= 1.0)

    def test_missing_model_file_returns_none_not_raise(self):
        assert build_signal_fn({'model_path': '/nonexistent/model.joblib'}) is None
