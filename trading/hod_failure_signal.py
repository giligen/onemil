"""Pluggable P(FF10) signal for the HOD-break failure-short overlay (order mechanics live in
trading/hod_failure_short.py + trading/hod_break_engine.py). `build_signal_fn(cfg)` loads the
FF10-at-k=1 model ONCE at boot and returns a `signal_fn(symbol, bars, arm_context) -> float | None`
closure that scores it from the engine's OWN live inputs via trading/hod_failure_features.py's
`k1_features` -- the SAME function research/hod_entry/1675_forward.py (the research scorer) calls,
so the research verdict and the live signal are the same arithmetic over different bar sources
(parity by construction, CLAUDE.md "ONE spec").

F11-F15 (research/hod_entry/1667_sweep.py, function compute_intraday_features) and atr14
(research/hod_entry/1675_forward.py::atr14_causal) are PORTED here verbatim (same query, same
arithmetic) because the research versions read research/bf_zero/bars_sip.db, a research-only
historical tick-bar cache the live engine has no access to -- this module computes the identical
formulas from the engine's own live bars / ADV map / data/cache.db daily_bars instead. `F8`
(F15's dollar-volume denominator) is not stored anywhere the live engine can reach; mapped to the
engine's own adv20 (documented assumption -- docs/hod_failure_short_spec_20260930.md).
"""
import json
import logging
import sqlite3
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd

from trading.hod_break import OPEN_MINUTE
from trading.hod_failure_features import find_break_bar, find_fill_index, k1_features, vwap_through

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = 'research/hod_entry/models/ff10_k1_val.joblib'
DEFAULT_FEATURES_PATH = 'research/hod_entry/models/ff10_k1_features.json'
DEFAULT_CACHE_DB = 'data/cache.db'
NAN_FEATURE_ERROR_MAX = 20     # > this many NaN columns out of 152 -> signal returns None, logged ERROR once/day
# spy_bars is never wired into this engine (no SPY bar feed) -- cS5_spyret_1 is therefore
# structurally NaN on every single evaluation, not a data gap. Logged once at boot, not per-call.
STRUCTURALLY_NEVER_COMPUTABLE = ('cS5_spyret_1',)


def atr14_causal(cache_db_path: str, symbol: str, day: str, _cache: dict) -> Optional[float]:
    """Standard 14-session ATR (simple mean of True Range), strictly BEFORE `day`. Ported verbatim
    from research/hod_entry/1675_forward.py::atr14_causal (same query, same arithmetic) against
    `data/cache.db daily_bars`, READ-ONLY (this module never writes to it). `_cache` memoizes one
    lookup per (symbol, day) for the process lifetime -- callers pass a dict they own."""
    key = (cache_db_path, symbol, day)
    if key in _cache:
        return _cache[key]
    try:
        con = sqlite3.connect(f'file:{cache_db_path}?mode=ro', uri=True, timeout=30)
        try:
            d = pd.read_sql(
                "select bar_date, high, low, close from daily_bars where symbol=? and bar_date<? "
                "order by bar_date desc limit 30", con, params=[symbol, day])
        finally:
            con.close()
    except Exception as e:
        logger.error(f"FAILURE-SHORT SIGNAL {symbol}: atr14 read from {cache_db_path} failed: {e}")
        _cache[key] = None
        return None
    if len(d) < 5:
        _cache[key] = None
        return None
    d = d.sort_values('bar_date')
    prev_close = d['close'].shift(1)
    tr = np.maximum(d['high'] - d['low'], np.maximum((d['high'] - prev_close).abs(), (d['low'] - prev_close).abs()))
    tr = tr.dropna().tail(14)
    out = float(tr.mean()) if len(tr) else None
    _cache[key] = out
    return out


def compute_f11_f15(bars: dict, fill_min: int, level: float, atr14: Optional[float], adv20: Optional[float]) -> dict:
    """F11-F15, same formulas as research/hod_entry/1667_sweep.py's intraday sweep:
    F11 = level/vwap-1, F12 = level/day_open-1, F13 = (day_high-day_low)/atr14, F14 = level age
    (minutes), F15 = dollar-volume-through-level / F8 -- all measured through the LEVEL bar (the
    breakout bar), never through the fill bar. F8 := adv20 here (see module docstring)."""
    out = {'F11': np.nan, 'F12': np.nan, 'F13': np.nan, 'F14': np.nan, 'F15': np.nan}
    level_idx = find_break_bar(bars, fill_min, level)
    if level_idx is None:
        return out
    lb_min = int(bars['minarr'][level_idx])
    vsum = float(bars['v'][:level_idx + 1].sum())
    vwap = vwap_through(bars, level_idx)
    if vwap and vsum > 0:
        out['F11'] = level / vwap - 1
        dvol = float((bars['c'][:level_idx + 1] * bars['v'][:level_idx + 1]).sum())
        if adv20 and adv20 > 0:
            out['F15'] = dvol / adv20
    day_open = float(bars['o'][0])
    if day_open:
        out['F12'] = level / day_open - 1
    day_high = float(bars['h'][:level_idx + 1].max())
    day_low = float(bars['l'][:level_idx + 1].min())
    if atr14 and atr14 > 0:
        out['F13'] = (day_high - day_low) / atr14
    out['F14'] = fill_min - lb_min
    return out


def build_feature_row(*, bars: dict, i0: int, entry: float, stop: float, target: float, level: float,
                       atr14: Optional[float], adv20: Optional[float]) -> tuple:
    """Assembles the k=1 arm_ctx (r_pct, atr14_pct, F11-F15, minutes_since_open) exactly as
    research/hod_entry/1675_forward.py::simulate_short does, then calls
    trading.hod_failure_features.k1_features. `fill_min` for both F11-F15 and k1_features is the
    FILL BAR's own minute (bars['minarr'][i0]), matching the research scorer -- never the raw
    (possibly fractional/off-bar) fill instant. Returns (feats: dict, computable: bool)."""
    fill_min = int(bars['minarr'][i0])
    break_idx = find_break_bar(bars, fill_min, level)
    break_bar_v = float(bars['v'][break_idx]) if break_idx is not None else np.nan
    atr14_pct = (atr14 / entry * 100) if (atr14 and entry) else np.nan
    arm_ctx = {'r_pct': (entry - stop) / entry * 100 if entry else np.nan, 'atr14_pct': atr14_pct,
               'minutes_since_open': fill_min - OPEN_MINUTE}
    arm_ctx.update(compute_f11_f15(bars, fill_min, level, atr14, adv20))
    return k1_features(bars, i0, entry, stop, target, level, atr14, break_bar_v, arm_ctx, spy_bars=None, fill_min=fill_min)


def build_signal_fn(cfg: dict) -> Optional[Callable[[str, dict, dict], Optional[float]]]:
    """Loads the FF10-k1 model + its ordered feature list ONCE (boot) and returns
    `signal_fn(symbol, bars, arm_context) -> float | None` for trading/hod_break_engine.py.

    `bars` must be the canonical {'o','h','l','c','v','minarr'} dict (project convention, index 0 =
    first RTH bar of the DAY) through the signal bar inclusive. `arm_context` must carry
    `fill_minute`, `long_fill_price`, `long_stop`, and SHOULD carry `long_target`, `level`, `adv20`,
    `date` (YYYY-MM-DD, for the atr14 cache-db lookup and the once-per-day NaN-storm log).

    Returns None (never raises) if the model/feature file cannot be loaded at boot -- the overlay
    then has no signal and every evaluation logs 'no signal_fn wired'
    (trading/hod_break_engine.py::_fs_evaluate_signal)."""
    model_path = str(cfg.get('model_path') or DEFAULT_MODEL_PATH)
    features_path = str(cfg.get('features_path') or DEFAULT_FEATURES_PATH)
    cache_db_path = str(cfg.get('cache_db_path') or DEFAULT_CACHE_DB)
    try:
        model = joblib.load(model_path)
        feat_cols = json.load(open(features_path))['feature_cols']
    except Exception as e:
        logger.error(f"FAILURE-SHORT SIGNAL: failed to load model={model_path} features={features_path}: {e} — signal_fn is None")
        return None
    logger.info(f"FAILURE-SHORT SIGNAL loaded {model_path} ({len(feat_cols)} feature columns)")
    structural = [c for c in STRUCTURALLY_NEVER_COMPUTABLE if c in feat_cols]
    if structural:
        logger.warning(f"FAILURE-SHORT SIGNAL: {len(structural)}/{len(feat_cols)} feature columns are "
                        f"structurally never computable by this engine (no SPY bar feed wired): {structural}")
    atr_cache: dict = {}
    nan_error_logged_for_day: list = [None]   # mutable cell closed over -- ERROR at most once/day

    def signal_fn(symbol: str, bars: dict, arm_context: dict) -> Optional[float]:
        try:
            i0 = find_fill_index(bars, arm_context['fill_minute'])
            if i0 is None:
                logger.warning(f"FAILURE-SHORT SIGNAL {symbol}: fill bar not found in the bar window — not computable")
                return None
            entry = float(arm_context['long_fill_price']); stop = float(arm_context['long_stop'])
            target = float(arm_context.get('long_target', entry))
            level = arm_context.get('level')
            level = float(level) if level is not None else float('nan')
            day = arm_context.get('date') or ''
            atr14 = atr14_causal(cache_db_path, symbol, day, atr_cache) if day else None
            adv20 = arm_context.get('adv20')
            feats, computable = build_feature_row(bars=bars, i0=i0, entry=entry, stop=stop, target=target,
                                                   level=level, atr14=atr14, adv20=adv20)
            if not computable:
                logger.info(f"FAILURE-SHORT SIGNAL {symbol}: not computable at k=1 (preempted stop/target/eod before fill+1)")
                return None
            row = {c: feats.get(c, np.nan) for c in feat_cols}
            n_nan = sum(1 for v in row.values() if v is None or (isinstance(v, float) and np.isnan(v)))
            if n_nan > NAN_FEATURE_ERROR_MAX:
                if nan_error_logged_for_day[0] != day:
                    logger.error(f"FAILURE-SHORT SIGNAL {symbol}: {n_nan}/{len(feat_cols)} feature columns NaN "
                                 f"(> {NAN_FEATURE_ERROR_MAX}) — signal not trustworthy, returning None")
                    nan_error_logged_for_day[0] = day
                return None
            df = pd.DataFrame([row], columns=feat_cols).astype(float)
            return float(model.predict_proba(df)[:, 1][0])
        except Exception as e:
            logger.error(f"FAILURE-SHORT SIGNAL {symbol}: signal_fn failed ({e}) — treating as not computable")
            return None

    return signal_fn
