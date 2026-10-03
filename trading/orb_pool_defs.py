"""ORB add-on pool definitions and membership: the ONE reader shared by the live engine and the nightly BT.

`orb.yaml::universe.addon_pools.pools[]` defines each pool's membership bounds (gap / price / prev-volume). The engine's
universe build (`ORBEngine.build_orb_universe_from_snapshots`) and the nightly backtest's pool universe both call
`pool_matches` / `pool_bounds`, so the thresholds exist in exactly one place (spec 2026-10-03, parity by construction).
"""
import logging
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_YAML = 'orb.yaml'


def parse_addon_pools(universe_cfg: Dict) -> Dict:
    """Return {'enabled', 'dry_run', 'pools'} from the `universe` config dict (engine defaults: off, dry)."""
    addon_cfg = (universe_cfg or {}).get('addon_pools', {}) or {}
    return {
        'enabled': bool(addon_cfg.get('enabled', False)),
        'dry_run': bool(addon_cfg.get('dry_run', True)),
        'pools': list(addon_cfg.get('pools', []) or []),
    }


def load_addon_pools(yaml_path: str = DEFAULT_YAML) -> Dict:
    """Load orb.yaml and return `parse_addon_pools` of its `universe` block plus the production bounds."""
    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    uni = cfg.get('universe', {}) or {}
    out = parse_addon_pools(uni)
    out['production'] = {
        'min_gap_pct': float(uni.get('min_gap_pct', 5.0)),
        'min_price': float(uni.get('min_price', 3.0)),
        'max_price': float(uni.get('max_price', 30.0)),
        'min_prev_volume': float(uni.get('min_prev_volume', 500_000)),
    }
    return out


def pool_bounds(pool: Dict, default_min_prev_volume: float) -> Dict[str, float]:
    """Membership bounds of one pool dict, with the engine's defaults for absent keys."""
    return {
        'min_price': float(pool.get('min_price', 0.0)),
        'max_price': float(pool.get('max_price', float('inf'))),
        'min_gap_pct': float(pool.get('min_gap_pct', 0.0)),
        'max_gap_pct': float(pool.get('max_gap_pct', float('inf'))),
        'min_prev_volume': float(pool.get('min_prev_volume', default_min_prev_volume)),
    }


def pool_matches(pool: Dict, open_price: float, gap_pct: float, prev_volume: float,
                 default_min_prev_volume: float) -> bool:
    """True iff (open, gap, prev volume) is inside the pool's membership bounds (inclusive, as the engine)."""
    b = pool_bounds(pool, default_min_prev_volume)
    return (b['min_price'] <= open_price <= b['max_price']
            and b['min_gap_pct'] <= gap_pct <= b['max_gap_pct']
            and prev_volume >= b['min_prev_volume'])


def find_pool(pools: List[Dict], name: str) -> Optional[Dict]:
    """The pool dict whose `name` or `pool_id` equals `name`, else None."""
    for p in pools:
        if name in (p.get('name'), p.get('pool_id')):
            return p
    return None
