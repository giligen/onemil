"""Generic per-pool admission gates for ORB add-on pools (PREREG_LIVE_UNION.md,
owner GO 2026-09-21; gate extension owner GO 2026-10-01).

Every add-on pool (`orb.yaml::universe.addon_pools.pools[]`) already has its
own gap/price membership bounds, checked at universe-build time (the 09:30-ish
snapshot tick) in `ORBEngine.build_orb_universe_from_snapshots`. This module
adds a SECOND, OPTIONAL layer of gates evaluated LATER, at the 09:35
pre-placement instant once the 5-minute opening range has closed — the same
instant `ORBEngine._run_pool_selection` re-validates the phantom gap and
scores candidates. Only data closed by 09:35 is used (CLAUDE.md causality
rule: membership/admission must be knowable at the decision bar).

Every gate defaults to None/absent in a pool's YAML dict == INACTIVE. A pool
that sets no gate keys behaves exactly as before this module existed —
byte-identical (tests/test_orb_addon_gates.py::TestByteIdentical). A pool
sets only the gates it wants; an unset gate never vetoes a candidate.

Mechanism, not accident (CLAUDE.md "ONE spec ... mechanism + evidence +
explicit code"): every input is named, computed exactly one way by the
caller, and handed in already-resolved. A gate configured against an input
that could not be resolved (missing daily-bar history, no premarket prints,
etc.) FAILS CLOSED and logs — it never silently admits on missing data.

No separate "ADV map" module exists in this codebase. `min_rel_volume_0935`
reuses the 20-day average volume already fetched into `daily_stats_20d` by
`ORBEngine._get_feature_context` (the same provider `_compute_features`
uses) and the SAME range-bar volume aggregation (`range_total_volume`) the
gap gate's universe build already produces — no new data source.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Regular session length in minutes (09:30-16:00 ET) — scales a full-day ADV20
# down to a 5-minute-bar expectation for min_rel_volume_0935. 390 = 6.5h * 60.
SESSION_MINUTES = 390.0
RANGE_MINUTES = 5.0

# require_day2_gapper's threshold is fixed by spec ("yesterday's gap >= 10%"),
# not configurable per pool.
DAY2_GAPPER_MIN_PCT = 10.0

# Pool-membership keys (checked in ORBEngine.build_orb_universe_from_snapshots,
# NOT by evaluate_pool_gates below) plus min_prev_volume, which is a per-pool
# OVERRIDE of the global universe.min_prev_volume checked at that same
# membership instant — it is resolved too early (before the range closes) to
# be a 09:35 gate, so it is read directly off the pool dict by the engine, not
# through evaluate_pool_gates. Listed here anyway so warn_unknown_pool_keys
# recognizes it.
POOL_MEMBERSHIP_KEYS = {
    'name', 'min_gap_pct', 'max_gap_pct', 'min_price', 'max_price',
    'min_prev_volume', 'pool_id',
}
# The 09:35-instant gate keys evaluate_pool_gates reads.
GATE_KEYS = {
    'min_move_to_range_high_pct',
    'min_rel_volume_0935',
    'min_premarket_dollar_vol',
    'require_above_vwap_0935',
    'max_dist_to_52wk_high_pct',
    'min_prev_day_range_atr',
    'require_day2_gapper',
}
KNOWN_POOL_KEYS = POOL_MEMBERSHIP_KEYS | GATE_KEYS


def warn_unknown_pool_keys(pool_cfg: Dict) -> None:
    """Log WARNING for any key in a pool's YAML dict that neither the
    membership check nor evaluate_pool_gates reads.

    The 9/25 lesson (docs/live_guardrails_spec_20260925.md / CLAUDE.md
    "Live launch lessons 9/25"): a config key absent from the code that
    reads it is silently dropped, never an error. A typo'd gate name must
    not look like an inactive gate — it must be LOUD at startup.
    """
    unknown = set(pool_cfg.keys()) - KNOWN_POOL_KEYS
    if unknown:
        logger.warning(
            "ORB ADDON POOL %r: unrecognized config key(s) %s — NOT read by "
            "any membership check or gate, will be silently ignored (typo? "
            "known keys: %s)",
            pool_cfg.get('name', '?'), sorted(unknown), sorted(KNOWN_POOL_KEYS),
        )


def pool_id_for(pool_cfg: Dict) -> str:
    """Short pool identifier for client_order_id / log / Telegram tagging and
    pattern_data/ledger persistence (owner 2026-10-01, "be clear on its
    trades"). Config key `pool_id`, defaulting to the pool's `name`
    truncated to 12 chars — resolved HERE ONCE so the client_order_id
    prefix, the `[ORB <pool_id>]` log/Telegram tag, pattern_data.pool_id and
    the dry-ledger pool_id column all agree by construction (CLAUDE.md "ONE
    spec ... mechanism + evidence + explicit code"). Callers special-case
    the production book directly — 'production' is never passed in here.
    """
    return str(pool_cfg.get('pool_id') or pool_cfg.get('name', ''))[:12]


@dataclass
class PoolGateInputs:
    """Every value an add-on pool gate can be evaluated against, resolved
    ONCE by the caller (`ORBEngine._build_pool_gate_inputs`) at the 09:35
    pre-placement instant, from data closed by then.

    A None field means that input could not be resolved (e.g. no daily-bar
    history, no premarket prints). A gate configured against a None input
    FAILS CLOSED — see evaluate_pool_gates.
    """
    move_to_range_high_pct: Optional[float] = None   # (range_high - prev_close) / prev_close * 100
    rel_volume_0935: Optional[float] = None           # range_total_volume / (ADV20-scaled 5-min expectation)
    premarket_dollar_vol: Optional[float] = None      # same source as sizing.pm_dollar_vol_mult
    above_vwap_0935: Optional[bool] = None            # range_close > 5-min typical-price VWAP proxy
    range_close_top_half: Optional[bool] = None       # range_close_position >= 0.5
    dist_to_52wk_high_pct: Optional[float] = None     # (high_52wk - ref_open) / high_52wk * 100; <=0 = at/above
    prev_day_range_atr: Optional[float] = None        # (prev_high - prev_low) / atr14_t1
    is_day2_gapper: Optional[bool] = None             # T-1's OWN gap (T-1 open vs T-2 close) >= DAY2_GAPPER_MIN_PCT


def evaluate_pool_gates(pool_cfg: Dict, inputs: PoolGateInputs
                         ) -> Tuple[bool, Dict[str, object]]:
    """Admit-or-reject one candidate into one add-on pool's OPTIONAL gates.

    Generic by construction: every gate is a (config key -> input field)
    pair checked by one of the two helpers below; adding a future gate is
    adding one more call, never a new branch shape. A pool with none of
    GATE_KEYS set returns (True, {}) for every candidate — byte-identical to
    pre-gate behaviour.

    Returns:
        (admitted, gate_values) — gate_values holds ONLY the gates this pool
        has configured (inactive gates are absent, not None-valued) together
        with the resolved input that was actually checked. The caller
        persists this dict verbatim as the dry/paper ledger row's pool_gates
        column and pattern_data.pool_gates — telemetry for the forward read
        (PREREG_LIVE_UNION.md).
    """
    gate_values: Dict[str, object] = {}
    admitted = True
    pool_name = pool_cfg.get('name', '?')

    def check_min(gate_key: str, cfg_key: str, value: Optional[float]) -> None:
        nonlocal admitted
        threshold = pool_cfg.get(cfg_key)
        if threshold is None:
            return  # gate inactive
        gate_values[gate_key] = value
        if value is None or value < float(threshold):
            admitted = False
            logger.info(
                "ORB ADDON GATE reject pool=%s key=%s value=%s threshold>=%s",
                pool_name, gate_key, value, threshold)

    def check_max(gate_key: str, cfg_key: str, value: Optional[float]) -> None:
        nonlocal admitted
        threshold = pool_cfg.get(cfg_key)
        if threshold is None:
            return  # gate inactive
        gate_values[gate_key] = value
        if value is None or value > float(threshold):
            admitted = False
            logger.info(
                "ORB ADDON GATE reject pool=%s key=%s value=%s threshold<=%s",
                pool_name, gate_key, value, threshold)

    check_min('move_to_range_high_pct', 'min_move_to_range_high_pct',
              inputs.move_to_range_high_pct)
    check_min('rel_volume_0935', 'min_rel_volume_0935', inputs.rel_volume_0935)
    check_min('premarket_dollar_vol', 'min_premarket_dollar_vol',
              inputs.premarket_dollar_vol)
    check_max('dist_to_52wk_high_pct', 'max_dist_to_52wk_high_pct',
              inputs.dist_to_52wk_high_pct)
    check_min('prev_day_range_atr', 'min_prev_day_range_atr',
              inputs.prev_day_range_atr)

    # Combined gate: "above VWAP AND the range closing in its top half" is
    # ONE config switch (require_above_vwap_0935) — both must hold.
    if pool_cfg.get('require_above_vwap_0935'):
        gate_values['above_vwap_0935'] = inputs.above_vwap_0935
        gate_values['range_close_top_half'] = inputs.range_close_top_half
        if not (inputs.above_vwap_0935 and inputs.range_close_top_half):
            admitted = False
            logger.info(
                "ORB ADDON GATE reject pool=%s key=require_above_vwap_0935 "
                "above_vwap=%s top_half=%s",
                pool_name, inputs.above_vwap_0935, inputs.range_close_top_half)

    if pool_cfg.get('require_day2_gapper'):
        gate_values['is_day2_gapper'] = inputs.is_day2_gapper
        if not inputs.is_day2_gapper:
            admitted = False
            logger.info(
                "ORB ADDON GATE reject pool=%s key=require_day2_gapper "
                "is_day2_gapper=%s (threshold %.1f%%)",
                pool_name, inputs.is_day2_gapper, DAY2_GAPPER_MIN_PCT)

    return admitted, gate_values
