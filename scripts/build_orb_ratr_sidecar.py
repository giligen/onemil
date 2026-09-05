"""ORB signal study C5 — range width vs ATR14 sidecar (2026-09-05).

ratr = (range_size_pct/100 × range_high) / ATR14(T−1), range_high recovered
from the order level (entry_price = range_high × 1.003 in the features CSV),
ATR14 from the SAME shared helper the winner stack uses (atr14_t1 via
study_orb_pipeline_static_lock.build_atr14_lookup, cache daily_bars). Tiers:
narrow < 0.3, normal 0.3–0.6, wide > 0.6 (TradingStats).

Usage: python3 scripts/build_orb_ratr_sidecar.py FEATURES_CSV OUT_CSV
"""
import sys

import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
sys.stdout.reconfigure(line_buffering=True)
from study_orb_pipeline_static_lock import build_atr14_lookup  # noqa: E402
from trading.orb_csv import read_orb_csv  # noqa: E402
from trading.orb_experimental_rules import ratr_tier  # noqa: E402

ENTRY_OVER_RANGE_HIGH = 1.003


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    f = read_orb_csv(src, usecols=['symbol', 'date', 'entry_price', 'range_size_pct'])
    f['date'] = pd.to_datetime(f['date']).dt.strftime('%Y-%m-%d')
    pairs = list(zip(f['symbol'], f['date']))
    print(f"ATR14 lookup for {len(pairs):,} pairs...")
    atr = build_atr14_lookup(pairs)
    rh = f['entry_price'] / ENTRY_OVER_RANGE_HIGH
    rng = f['range_size_pct'] / 100.0 * rh
    a = pd.Series([atr.get(k) for k in pairs], index=f.index, dtype='float64')
    f['ratr'] = rng / a
    f['ratr_tier'] = f['ratr'].map(ratr_tier)
    out = f[['symbol', 'date', 'ratr', 'ratr_tier']]
    out.to_csv(dst, index=False)
    print(f"sidecar -> {dst}: {len(out):,} keys | ratr available {int(out['ratr'].notna().sum()):,} | "
          f"tiers {out['ratr_tier'].value_counts().to_dict()} | median {out['ratr'].median():.2f}")


if __name__ == '__main__':
    main()
