"""PREREG_1675 population step: run the cell-1,438 causal-arming pipeline UNCHANGED, over the sealed
forward window, into forward_2026q3/ only. Never touches the original TRAIN/VAL outputs.

Population window: 2026-06-01 .. 2026-09-04 (research/bf_zero/universe.csv's own max date is
2026-09-04 as of this run; 2026-09-05..2026-09-26 is NOT covered because the universe.csv builder
script could not be located within budget -- see RESULT_1675.md caveat. This is the SAME universe.csv
file used by every other HOD cell, just sliced to a later date range: not a different population.

Everything else (population gates, arming rule, resting stop-limit fill logic, cost model, NBBO
fallback to fill-instant half-spread) is byte-identical to research/hod_entry/causal_arming.py --
only SPLITS/OUT_CSV/REPORT_MD/CACHE_DIR are monkeypatched so outputs land under forward_2026q3/ and
TRAIN/VAL-specific reporting (which needs the 1,427 comparison file, not applicable to a sealed
forward-only book) is skipped in favor of a minimal coverage summary.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HOD = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(HOD))
sys.path.insert(0, HOD)
sys.path.insert(0, ROOT)

import causal_arming as ca  # noqa: E402
import sip_rebuild as sr  # noqa: E402

FWD_START, FWD_END = '2026-06-01', '2026-09-04'

# All population rows land in the 'VAL' bucket (TRAIN window inverted so nothing qualifies as TRAIN).
ca.SPLITS = {'TRAIN': (FWD_START, '2020-01-01'), 'VAL': (FWD_START, FWD_END)}
ca.OUT_CSV = os.path.join(HERE, 'causal_arming_{variant}.csv')
sr.CACHE_DIR = os.path.join(HERE, 'sip_cache')
os.makedirs(sr.CACHE_DIR, exist_ok=True)


def main():
    workers = int(os.environ.get('FWD_WORKERS', '10'))
    sr.log(f'[fwd] population window {FWD_START}..{FWD_END} | universe.csv (unchanged) | workers={workers}')
    res, n_nobars = ca.run(workers)
    fills = res[(res.variant == 'causal') & (res.status == 'fill')]
    cov = res[res.variant == 'causal'].status.value_counts().to_dict()
    sr.log(f'[fwd] DONE rows={len(res)} causal_fills={len(fills)} n_nobars={n_nobars} status_counts={cov}')
    summary_path = os.path.join(HERE, 'population_summary.txt')
    with open(summary_path, 'w') as fh:
        fh.write(f'window {FWD_START}..{FWD_END}\n')
        fh.write(f'candidate symbol-days (causal superset, both variants counted once via causal): '
                 f'{len(res[res.variant == "causal"])}\n')
        fh.write(f'fills: {len(fills)}\n')
        fh.write(f'status_counts: {cov}\n')
        fh.write(f'n_nobars (superset symbol-days with < K+2 minute bars in cache.db+bars_sip.db): {n_nobars}\n')
    sr.log(f'[fwd] summary written {summary_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
