"""Two-tier filter for 10%-threshold bull flag signal quality.

Shared between backtest Stage-2 filter and live trading engine. Both callers
import the same functions to guarantee identical accept/reject decisions.

Tiers:
    A    — max_intraday_change_pct_at_entry >= a_tier_lower (default 20%).
           Unfiltered. Matches what the 20%-threshold scanner would have seen.
    E    — a_tier_lower > change >= extras_lower (default 10-19%). Only 10%
           scanner sees these; empirically noisy. Gated by two filters:
               (i)  surgical drop:  reject if macd_zone_mult < drop_extras_macd_below
               (ii) composite score: reject if z-score < composite_threshold
    edge — change < extras_lower (<10%). Unfiltered (small subset that just
           barely missed qualification; empirically fine).

The composite score is the mean of SIGNED z-scores over N features. Each
feature contributes `sign * (value - mean) / std` where mean/std/sign are
frozen TRAIN-fit parameters (live config). Higher score = better predicted
outcome. Extras below `composite_threshold` are rejected.

Origin of frozen params: empirical fit on Jan-Jul 2025 Extras subset (O_f6
cache, n=83). See composite_score.py / phase_final.py for the fit harness.
"""
from __future__ import annotations
from typing import Callable, Dict, Optional


# Tier labels
TIER_A = "A"
TIER_EXTRAS = "E"
TIER_EDGE = "edge"


def classify_tier(
    max_intraday_change_pct: float,
    a_tier_lower: float = 20.0,
    extras_lower: float = 10.0,
) -> str:
    """Classify trade by max intraday % change reached before entry.

    Boundary: >=a_tier_lower -> A, >=extras_lower -> E, else edge.
    """
    if max_intraday_change_pct is None:
        return TIER_EDGE
    if max_intraday_change_pct >= a_tier_lower:
        return TIER_A
    if max_intraday_change_pct >= extras_lower:
        return TIER_EXTRAS
    return TIER_EDGE


def composite_score(
    features: Dict[str, float],
    params: Dict[str, Dict[str, float]],
) -> Optional[float]:
    """Mean of signed z-scores across features that have non-None values.

    Args:
        features: dict of {feature_name: value} from the trade.
        params: dict of {feature_name: {"mean": m, "std": s, "sign": +-1}}.
                Only feature_names present in BOTH are used.

    Returns None if no feature overlaps (caller decides default behavior).
    """
    total = 0.0
    count = 0
    for name, p in params.items():
        v = features.get(name)
        if v is None:
            continue
        std = float(p.get("std", 1.0) or 1.0)
        mean = float(p.get("mean", 0.0))
        sign = int(p.get("sign", 1))   # cast in case YAML loads it as str
        z = (v - mean) / std
        total += sign * z
        count += 1
    if count == 0:
        return None
    return total / count


def should_keep(
    tier: str,
    macd_zone_mult: float,
    features: Dict[str, float],
    cfg: Dict,
) -> tuple[bool, str]:
    """Return (keep, reason). Reason is empty string when kept.

    `cfg` keys (typically loaded from YAML under trading.bull_flag.two_tier_filter):
        enabled: bool
        drop_extras_macd_below: float (e.g. 1.25)
        composite_threshold: float (e.g. -0.50)
        composite_features: dict of {name: {mean, std, sign}}
    """
    if not cfg.get("enabled", False):
        return True, ""
    if tier == TIER_A or tier == TIER_EDGE:
        return True, ""
    if tier != TIER_EXTRAS:
        return True, ""

    # Extras tier — two gates.
    # Surgical drop only fires when MACD signal is KNOWN. macd_zone_mult=None
    # means caller (e.g. live engine with macd_zones disabled) couldn't compute
    # it — in that case skip the surgical check and rely only on composite.
    # Setting macd_zone_mult explicitly to 0.0 or low values DOES trigger the
    # drop (that's the "MACD dead zone" case from the empirical data).
    #
    # NOTE (S2-max ship, 2026-04-18): this gate is now largely redundant for
    # Extras trades because per-tier MACD zone mults are {dead=0.0, normal=0.0,
    # strong=2.0}. All Extras normal/dead trades are rejected at
    # `_get_macd_zone_multiplier` before reaching TTF; surviving trades have
    # macd_zone_mult=2.0 > drop_below(1.25), so they pass. Kept as defensive
    # safety net in case extras_tier config is removed (fallback to A-tier
    # macd mults = {1.0, 1.8}, where normal=1.0 < 1.25 still triggers drop).
    drop_below = cfg.get("drop_extras_macd_below")
    if drop_below is not None and macd_zone_mult is not None:
        if macd_zone_mult < float(drop_below):
            return False, "extras_macd_surgical_drop"

    threshold = cfg.get("composite_threshold")
    params = cfg.get("composite_features") or {}
    if threshold is not None and params:
        score = composite_score(features, params)
        if score is None:
            # Missing all feature inputs — conservative: reject with explicit reason
            return False, "extras_composite_no_features"
        if score < float(threshold):
            return False, "extras_composite_below_threshold"

    return True, ""


def max_intraday_change_pre_entry(
    bars,
    prev_close: Optional[float],
    entry_ts_utc: str,
) -> Optional[float]:
    """Replay pre-entry bars (1-min) tracking max(gap_pct, range_pct).

    Mirrors scanner/realtime_scanner.py:610-614 qualification logic.
    `bars` is an iterable of (timestamp_str, open, high, low, close) tuples
    sorted by timestamp. Returns the MAX value of max(gap_pct, range_pct)
    over all bars with timestamp < entry_ts_utc. Returns None if no bars.

    gap_pct = (close - prev_close) / prev_close * 100 (skipped if prev_close None)
    range_pct = (day_high - day_low) / day_low * 100

    This is the canonical feature used for tier classification.
    """
    day_high = None
    day_low = None
    max_qual = None
    saw_bar = False
    for ts, _o, h, l, c in bars:
        if ts >= entry_ts_utc:
            break
        saw_bar = True
        if h is not None and (day_high is None or h > day_high):
            day_high = h
        if l is not None and l > 0 and (day_low is None or l < day_low):
            day_low = l
        # Both day_high and day_low must be known-good to compute range_pct.
        if day_high is not None and day_low is not None and day_low > 0:
            range_pct = (day_high - day_low) / day_low * 100
        else:
            range_pct = 0.0
        if prev_close and prev_close > 0 and c is not None:
            gap_pct = (c - prev_close) / prev_close * 100
        else:
            gap_pct = float("-inf")
        qual = max(gap_pct, range_pct)
        if max_qual is None or qual > max_qual:
            max_qual = qual
    if not saw_bar:
        return None
    return max_qual


def build_features_from_trade(trade: Dict) -> Dict[str, float]:
    """Extract the 4 default composite features from a trade dict / cache row.

    Feature names match keys in `composite_features` config. Missing values
    return None — composite_score() ignores them, but for score-scale
    consistency the fallback rule below keeps all 4 features populated in
    the normal case:
        qf_fill_vwap_dist_pct  -> falls back to qf_vwap_dist_pct when empty.
    Rationale: the fill VWAP differs from setup VWAP by 1-2 bars (r>0.95
    empirically). In post-fill-exit BT rows it's never populated; in the
    live engine we don't know the fill bar yet. Using the setup-bar value
    in those cases preserves the 4-feature normalization scale used to fit
    the frozen z-score params.
    """
    def _f(v):
        try:
            return float(v) if v not in (None, "", "None") else None
        except (ValueError, TypeError):
            return None

    entry_time_et = trade.get("entry_time_et") or ""
    entry_minute = None
    if entry_time_et:
        try:
            hh, mm = entry_time_et.split(":")[:2]
            entry_minute = int(hh) * 60 + int(mm)
        except (ValueError, IndexError):
            entry_minute = None

    setup_vwap_dist = _f(trade.get("qf_vwap_dist_pct"))
    fill_vwap_dist = _f(trade.get("qf_fill_vwap_dist_pct"))
    if fill_vwap_dist is None and setup_vwap_dist is not None:
        fill_vwap_dist = setup_vwap_dist  # preserve 4-feature scale

    return {
        "conviction_mult": _f(trade.get("conviction_mult")),
        "qf_vwap_dist_pct": setup_vwap_dist,
        "qf_fill_vwap_dist_pct": fill_vwap_dist,
        "entry_minute": float(entry_minute) if entry_minute is not None else None,
    }
