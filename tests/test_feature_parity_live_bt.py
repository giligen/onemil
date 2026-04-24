"""Parity test: live ORBEngine._compute_features + composite_score ==
BT study_orb_features.extract_features + composite_score, given the
same synthetic bars.

This would have caught the 2026-04-22 → 2026-04-23 cache-pollution bug
(BMNZ live 0.35 Q4 vs BT 0.48 Q5) at commit time rather than via
live-trading telemetry. The two paths must agree when fed identical
data; any divergence is either:
  (a) a bug in one of the feature-extraction functions, OR
  (b) stale/different input — which the parity test doesn't cover
      directly but makes mistake-at-the-data-layer obvious.
"""
from __future__ import annotations

from datetime import datetime, date, timedelta, timezone
from unittest.mock import MagicMock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, RangeData, CandidateState
from trading.orb_filter import load_feature_params, composite_score as prod_composite
from trading.stop_monitor import StopMonitor

from study_orb_filter import (
    FILTER_FEATURES, fit_z_params, composite_score as bt_composite,
)
from study_orb_features import extract_features


# ---------------------------------------------------------------------------
# Fixtures — shared synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def orb_cfg():
    yaml_path = Path(__file__).parent.parent / 'orb.yaml'
    with open(yaml_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope='module')
def z_params_prod(orb_cfg):
    return load_feature_params(orb_cfg['filter'])


@pytest.fixture
def engine(orb_cfg):
    orb_cfg['strategy']['enabled'] = True
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.get_open_positions.return_value = []
    alpaca.get_account_info.return_value = {'buying_power': 500_000.0}
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 1
    db.get_open_trades.return_value = []
    db.get_trades_by_date.return_value = []
    db.get_daily_bars_cached.return_value = {}
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    return ORBEngine(alpaca_client=alpaca, db=db, stop_monitor=sm, config=orb_cfg)


def _mk_range_bars(
    open_p: float, highs, lows=None, volumes=None,
    start_utc=None,
) -> pd.DataFrame:
    """Build a 5-bar DataFrame covering 9:30–9:34 ET."""
    if start_utc is None:
        start_utc = datetime(2026, 4, 22, 13, 30, tzinfo=timezone.utc)
    rows = []
    for i, h in enumerate(highs):
        ts = start_utc.replace(minute=30 + i)
        lo = lows[i] if lows else open_p
        vol = volumes[i] if volumes else 10_000
        rows.append({
            'timestamp': ts,
            'open': open_p if i == 0 else highs[i - 1],
            'high': h, 'low': lo, 'close': h, 'volume': vol,
        })
    return pd.DataFrame(rows)


def _mk_daily_history(
    sym: str, prev_days: list, bars_start: date = None,
) -> pd.DataFrame:
    """Build a daily-bars DataFrame for `study_orb_features.extract_features`.
    prev_days: list of (open, high, low, close, volume) tuples, oldest-first.
    The last entry is T-1. Dates span backward from `bars_start` (default
    2026-04-21)."""
    if bars_start is None:
        bars_start = date(2026, 4, 21)
    rows = []
    for i, (o, h, l, c, v) in enumerate(prev_days):
        d = bars_start - timedelta(days=len(prev_days) - 1 - i)
        rows.append({'symbol': sym, 'bar_date': pd.Timestamp(d),
                     'open': o, 'high': h, 'low': l, 'close': c,
                     'volume': v})
    return pd.DataFrame(rows)


def _ctx_for_live(prev_day, daily_20d_high):
    """Synthesize the `providers` dict that live's _compute_features accepts."""
    o, h, l, c, v = prev_day
    return {
        'prev_day_bar': {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v},
        'daily_stats_20d': {'high_20d': daily_20d_high, 'volume_20d': 0.0},
    }


# ---------------------------------------------------------------------------
# Scenarios — realistic, edge-case, pathological
# ---------------------------------------------------------------------------

# (name, bars_df_kwargs, prev_day, 20_prev_days_list, event_date)
SCENARIOS = [
    (
        'bmnz_like_small_gap',
        dict(open_p=14.23, highs=[14.34, 14.25, 14.28, 14.32, 14.30],
             lows=[14.05, 14.05, 14.08, 14.10, 14.12],
             volumes=[9000, 8000, 7000, 10000, 10194]),
        # Prev day 4/22 final: open 14.12, high 14.31, low 13.24, close 13.43
        (14.12, 14.31, 13.24, 13.43, 4_150_038),
        # 20 prior days — just give varied highs so max is recoverable
        [(10 + i*0.1, 11 + i*0.1, 9 + i*0.1, 10.5 + i*0.1, 100_000) for i in range(20)],
        date(2026, 4, 23),
    ),
    (
        'high_price_tight_range',
        dict(open_p=25.07, highs=[25.00, 24.95, 24.80, 24.88, 24.90],
             lows=[23.76, 23.80, 23.76, 23.85, 23.90],
             volumes=[50000, 30000, 25000, 40000, 75000]),
        (19.85, 20.10, 19.40, 19.85, 500_000),
        [(18 + i*0.2, 19 + i*0.2, 17 + i*0.2, 18.5 + i*0.2, 200_000) for i in range(20)],
        date(2026, 4, 23),
    ),
    (
        'low_price_big_range',
        dict(open_p=3.00, highs=[3.15, 3.12, 3.18, 3.14, 3.16],
             lows=[2.88, 2.90, 2.92, 2.94, 2.95],
             volumes=[100000, 80000, 50000, 90000, 120000]),
        (2.60, 2.75, 2.45, 2.60, 500_000),
        [(2 + i*0.05, 2.5 + i*0.05, 1.8 + i*0.05, 2.3 + i*0.05, 300_000) for i in range(20)],
        date(2026, 4, 23),
    ),
    (
        'zero_prior_gap',
        # Today's open identical to yesterday's close — gap_pct=0
        dict(open_p=10.00, highs=[10.20, 10.15, 10.18, 10.22, 10.19],
             lows=[9.95, 9.98, 10.00, 10.03, 10.05],
             volumes=[20000, 18000, 15000, 22000, 25000]),
        (10.00, 10.50, 9.50, 10.00, 1_000_000),
        [(9 + i*0.1, 10 + i*0.1, 8 + i*0.1, 9.5 + i*0.1, 500_000) for i in range(20)],
        date(2026, 4, 23),
    ),
    (
        'degenerate_flat_range',
        # All 5 bars identical price — minimum range_size
        dict(open_p=5.00, highs=[5.00, 5.00, 5.00, 5.00, 5.00],
             lows=[4.99, 4.99, 4.99, 4.99, 4.99],
             volumes=[5000, 5000, 5000, 5000, 5000]),
        (4.80, 5.10, 4.60, 4.80, 200_000),
        [(4 + i*0.05, 5 + i*0.05, 3.5 + i*0.05, 4.5 + i*0.05, 200_000) for i in range(20)],
        date(2026, 4, 23),
    ),
]


# ---------------------------------------------------------------------------
# The parity test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, bars_kwargs, prev_day, hist_20d, evt_date", SCENARIOS)
def test_live_vs_bt_features_match(
    engine, z_params_prod, name, bars_kwargs, prev_day, hist_20d, evt_date,
):
    """For identical input (bars + prev-day + 20d history), live and BT
    must produce the same 7 filter features and the same composite
    score. Any divergence here is a real logic bug in one of the paths."""
    bars = _mk_range_bars(**bars_kwargs)

    # ---- LIVE path ----
    # _ingest_bars populates cand.range_data from the 5-bar DataFrame.
    sym = 'TEST'
    engine.build_universe(source_loader=lambda: [sym])
    engine._ingest_bars(sym, bars)
    cand = engine.candidates[sym]
    assert cand.range_data is not None, "_ingest_bars failed to populate range_data"

    # Live's _get_feature_context in production does:
    #   bars_list = db.get_daily_bars_cached(...)   # returns all prior days
    #   window = bars_list[-20:]                    # last 20 INCLUDING T-1
    #   high_20d = max(b['high'] for b in window)
    # BT's extract_features does:
    #   prior_df = sym_daily[bar_date < dtx].tail(20)  # last 20 INCLUDING T-1
    #   high_20d = prior_df['high'].max()
    # So for parity, compute high_20d from the last 20 of (hist + prev_day).
    full_history = hist_20d + [prev_day]
    last_20 = full_history[-20:]
    high_20d = max(h for _, h, _, _, _ in last_20)

    live_feats = engine._compute_features(
        cand,
        prev_day_bar=_ctx_for_live(prev_day, high_20d)['prev_day_bar'],
        daily_stats_20d=_ctx_for_live(prev_day, high_20d)['daily_stats_20d'],
    )

    # ---- BT path ----
    # extract_features takes bars_df + daily_by_sym + spy_intraday + spy_daily.
    # SPY inputs don't affect the 7 filter features we care about; feed empty.
    daily_by_sym = {sym: _mk_daily_history(sym, full_history,
                                            bars_start=evt_date - timedelta(days=1))}
    spy_intraday_empty = pd.DataFrame(columns=['timestamp', 'open', 'high',
                                               'low', 'close', 'volume'])
    spy_intraday_empty['timestamp'] = pd.to_datetime(spy_intraday_empty['timestamp'], utc=True)
    spy_daily_empty = pd.DataFrame(columns=['symbol', 'bar_date', 'open',
                                            'high', 'low', 'close', 'volume'])
    spy_daily_empty['bar_date'] = pd.to_datetime(spy_daily_empty['bar_date'])

    bt_feats = extract_features(
        bars_df=bars, symbol=sym, date_str=evt_date.isoformat(),
        daily_by_sym=daily_by_sym,
        spy_intraday=spy_intraday_empty,
        spy_daily=spy_daily_empty,
    )
    assert bt_feats is not None, f"BT extract_features returned None for {name}"

    # ---- Compare the 7 filter features ----
    filter_keys = [k for k, _ in FILTER_FEATURES]
    for key in filter_keys:
        live_v = live_feats.get(key)
        bt_v = bt_feats.get(key)
        assert live_v is not None, f"[{name}] live missing {key}"
        assert bt_v is not None, f"[{name}] BT missing {key}"
        assert live_v == pytest.approx(bt_v, rel=1e-6, abs=1e-9), (
            f"[{name}] feature {key}: live={live_v} bt={bt_v} (Δ={live_v-bt_v:+.6e})"
        )

    # ---- Compare composite via both composite_score implementations ----
    live_comp = prod_composite(live_feats, z_params_prod)
    # BT's composite_score takes a DataFrame; wrap the feats row.
    bt_comp_df = pd.DataFrame([bt_feats])
    z_params_bt = {
        k: {'sign': p.sign, 'mean': p.mean, 'std': p.std}
        for k, p in z_params_prod.items()
    }
    bt_comp = float(bt_composite(bt_comp_df, z_params_bt).iloc[0])

    assert live_comp == pytest.approx(bt_comp, rel=1e-6, abs=1e-9), (
        f"[{name}] composite: live={live_comp:.6f} bt={bt_comp:.6f} "
        f"(Δ={live_comp-bt_comp:+.6e})"
    )


def test_bmnz_real_world_recovery(engine, z_params_prod):
    """Re-derive the 2026-04-23 BMNZ composite from real-world inputs and
    confirm both paths land on the correct ~0.48 Q5 answer. This is the
    trade that inspired the whole parity test suite.

    Numbers: prev_close=$13.43 (true 4/22 close, NOT the polluted $13.66),
    gap=5.957%, composite should sit in the high-0.4 range.
    """
    bars = _mk_range_bars(
        open_p=14.23,
        highs=[14.335, 14.25, 14.28, 14.32, 14.30],
        lows=[14.05, 14.05, 14.08, 14.10, 14.12],
        volumes=[9000, 8000, 7000, 10000, 10194],  # sums to 44194
    )
    # Real daily context for BMNZ circa 4/22
    prev_day = (14.12, 14.31, 13.24, 13.43, 4_150_038)
    hist = [(10 + i*0.2, 11 + i*0.2, 9 + i*0.2, 10.5 + i*0.2, 100_000)
            for i in range(20)]

    engine.build_universe(source_loader=lambda: ['BMNZ'])
    engine._ingest_bars('BMNZ', bars)
    cand = engine.candidates['BMNZ']
    high_20d = max(h for _, h, _, _, _ in hist)
    live_feats = engine._compute_features(
        cand,
        prev_day_bar={'open': 14.12, 'high': 14.31, 'low': 13.24,
                      'close': 13.43, 'volume': 4_150_038},
        daily_stats_20d={'high_20d': high_20d, 'volume_20d': 0.0},
    )
    live_comp = prod_composite(live_feats, z_params_prod)
    # gap_pct with real prev_close should be 5.957% (NOT the polluted 4.173%).
    assert live_feats['gap_pct'] == pytest.approx(5.957, abs=0.01), (
        f"Real-world gap_pct should match BT's 5.957 when fed the TRUE 4/22 "
        f"close. Got {live_feats['gap_pct']}. If this fails, either the "
        f"parity logic regressed OR the cache is polluted again."
    )
    # Composite should be in Q5 territory (>= 0.4081 per orb.yaml cutoffs).
    assert live_comp >= 0.40, (
        f"Composite {live_comp:.3f} should sit in Q4/Q5 for a clean gap-up "
        f"breakout. Below 0.35 would indicate prev_close pollution."
    )
