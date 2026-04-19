"""ORB composite z-score filter — production module.

Loads TRAIN-fit params from orb.yaml and provides pure functions to:
  - compute the composite z-score for a single candidate
  - assign a quintile bucket (Q1..Q5) given the cutoffs

Mirrors the validated research implementation in study_orb_filter.py.
No look-ahead: all inputs are features known AT 9:35 ET (end of range window).

Shared by both PROD (trading/orb_engine.py) and BT (future parity harness).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FeatureParam:
    """Z-score params for one feature, fit on TRAIN data."""
    sign: int      # +1 (higher is better) or -1 (lower is better)
    mean: float
    std: float     # MUST be > 0; loader enforces


def load_feature_params(orb_yaml_filter: dict) -> Dict[str, FeatureParam]:
    """Parse the `filter.features` section of orb.yaml into a dict of FeatureParam.

    Args:
        orb_yaml_filter: the `filter` sub-dict from orb.yaml (must have 'features' key).

    Returns:
        dict mapping feature name → FeatureParam. Ordering preserved from yaml.

    Raises:
        ValueError if any feature has std <= 0 or sign not in {-1, +1}.
    """
    features_raw = orb_yaml_filter.get('features', {})
    if not features_raw:
        raise ValueError("orb.yaml filter.features is empty")
    out: Dict[str, FeatureParam] = {}
    for name, spec in features_raw.items():
        sign = int(spec.get('sign', 0))
        mean = float(spec.get('mean', 0.0))
        std = float(spec.get('std', 0.0))
        if sign not in (-1, 1):
            raise ValueError(f"Feature '{name}' has invalid sign={sign} (must be -1 or +1)")
        if std <= 0:
            raise ValueError(f"Feature '{name}' has std={std} (must be > 0)")
        out[name] = FeatureParam(sign=sign, mean=mean, std=std)
    return out


def composite_score(features: Dict[str, float],
                    params: Dict[str, FeatureParam]) -> Optional[float]:
    """Compute the signed-z-score composite for one candidate.

    Formula: composite = mean over features of (sign * (value - mean) / std)

    Args:
        features: dict of feature_name → raw value (e.g., {'gap_pct': 10.5, ...})
        params: loaded via load_feature_params.

    Returns:
        composite score (higher = better), or None if any required feature is
        missing / NaN. Conservative: rather than invent a value, we reject the
        candidate entirely.
    """
    if not params:
        raise ValueError("composite_score: empty params dict")
    total = 0.0
    for feat_name, p in params.items():
        raw = features.get(feat_name)
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            return None
        z = (float(raw) - p.mean) / p.std
        total += z * p.sign
    return total / len(params)


def assign_quintile(score: float, cutoffs: List[float]) -> str:
    """Bucket a composite score into Q1..Q5 given 4 ascending cutoffs.

    Cutoffs split the train-kept composite distribution into 5 equal-size buckets.
    Q1 = below cutoffs[0]; Q5 = >= cutoffs[3]. Between cutoffs are Q2..Q4.

    Args:
        score: composite z-score (as returned by composite_score).
        cutoffs: list of 4 floats in ascending order.

    Returns:
        One of 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'.

    Raises:
        ValueError if cutoffs length != 4 or not ascending.
    """
    if len(cutoffs) != 4:
        raise ValueError(f"cutoffs must be length 4, got {len(cutoffs)}")
    if cutoffs != sorted(cutoffs):
        raise ValueError(f"cutoffs must be ascending: {cutoffs}")
    for i, c in enumerate(cutoffs):
        if score < c:
            return f"Q{i+1}"
    return "Q5"
