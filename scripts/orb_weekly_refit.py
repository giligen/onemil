#!/usr/bin/env python3
"""ORB weekly SELECTION refit — owner decision 2026-09-08 ("so 26 wks").

Re-fits the composite's z-params (`filter.features.<f>.{mean,std}`) and the
`quintile_cutoffs` in orb.yaml on the trailing 26 weeks of ORB candidates
(entered-inclusive features CSV — every morning candidate, not just fills).
The adaptive_mults are NEVER touched: on the honest book, refitting the
sizing multipliers is the whipsaw (one ANNA fill sized 3x carried the whole
"expanding" gain); refitting the selection is a plateau across windows
>= 20w and refit days Mon/Wed/Fri (research/orb_refit_walkforward/REPORT.md:
26w $7,588 vs frozen $5,669, MDD -223 vs -551, red months 4 vs 6).

The fit is byte-for-byte the walk-forward harness's fit (same functions:
study_orb_pipeline_static_lock.fit_z_params / composite_score,
study_orb_sizing.fit_quintile_cutoffs) so the live parameters ARE the
parameters the backtest validated.

Writes in place (comments, signs and every other key preserved), backs the
yaml up first, logs the fit to logs/orb_refit_history.jsonl. The engine and
the nightly pipeline read the literals at their next start (Monday 12:30 boot).

Usage:
  python scripts/orb_weekly_refit.py --dry-run          # print what would change
  python scripts/orb_weekly_refit.py                    # write (Sunday cron)
  python scripts/orb_weekly_refit.py --as-of 2026-09-07 # window end for parity checks
Refuses (exit 2) when the window has < MIN_ROWS candidates, a std is ~0, or
the features CSV is older than MAX_AGE_DAYS.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

WINDOW_WEEKS = 26
MIN_ROWS = 500
MAX_AGE_DAYS = 4
ORB_YAML = ROOT / 'orb.yaml'
HISTORY = ROOT / 'logs' / 'orb_refit_history.jsonl'


def latest_features_csv() -> str:
    paths = sorted(p for p in glob.glob('analysis_results/orb_features_*.csv') if 'corrmatrix' not in p)
    if not paths:
        raise SystemExit("FATAL: no analysis_results/orb_features_*.csv")
    return paths[-1]


def fit_window(df: pd.DataFrame, as_of: date, weeks: int = WINDOW_WEEKS) -> Tuple[Dict, List[float], Dict]:
    """The harness's fit on [as_of - weeks, as_of). Returns (params, cutoffs, meta)."""
    import study_orb_pipeline_static_lock as P
    from study_orb_sizing import fit_quintile_cutoffs
    import yaml
    thresh = float(yaml.safe_load(open(ORB_YAML))['filter']['threshold'])
    d = df.copy(); d['date'] = pd.to_datetime(d['date'])
    lo = pd.Timestamp(as_of) - pd.Timedelta(weeks=weeks)
    train = d[(d['date'] >= lo) & (d['date'] < pd.Timestamp(as_of))]
    if len(train) < MIN_ROWS:
        raise SystemExit(f"REFUSE: only {len(train)} candidates in the {weeks}-week window (< {MIN_ROWS})")
    params = P.fit_z_params(train, P.FILTER_FEATURES)
    for f, p in params.items():
        if p['std'] <= 1e-6:
            raise SystemExit(f"REFUSE: std ~0 for {f}")
    comp = P.composite_score(train, params)
    kept = comp[comp >= thresh]
    if len(kept) < 25:
        raise SystemExit(f"REFUSE: only {len(kept)} rows above threshold in the window")
    cutoffs = [float(x) for x in fit_quintile_cutoffs(kept)]
    meta = dict(window_start=str(lo.date()), window_end=str(as_of), n_rows=int(len(train)),
                n_above_threshold=int(len(kept)), threshold=thresh)
    return params, cutoffs, meta


def rewrite_yaml(text: str, params: Dict, cutoffs: List[float]) -> str:
    """In-place edit of mean/std under filter.features.<f> and the quintile_cutoffs line.
    Everything else (comments, sign lines, other keys) is preserved byte-for-byte."""
    out = text
    for f, p in params.items():
        # locate the feature block: '    <f>:' then its 'mean:' and 'std:' lines
        m = re.search(rf"^(    {re.escape(f)}:\n)((?:      .*\n){{1,4}})", out, flags=re.M)
        if not m:
            raise SystemExit(f"REFUSE: feature block for {f} not found in orb.yaml")
        block = m.group(2)
        block2 = re.sub(r"^(      mean: )[-0-9.eE+]+", rf"\g<1>{p['mean']:.12f}", block, count=1, flags=re.M)
        block2 = re.sub(r"^(      std: )[-0-9.eE+]+", rf"\g<1>{p['std']:.12f}", block2, count=1, flags=re.M)
        if block2 == block and ('mean:' not in block or 'std:' not in block):
            raise SystemExit(f"REFUSE: mean/std lines for {f} not found")
        out = out[:m.start(2)] + block2 + out[m.end(2):]
    cut_line = "quintile_cutoffs: [" + ", ".join(f"{c:.12f}" for c in cutoffs) + "]"
    out, n = re.subn(r"^quintile_cutoffs: \[[^\]]*\]", cut_line, out, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("REFUSE: quintile_cutoffs line not found")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--as-of', default=None, help='window end (exclusive), default today')
    ap.add_argument('--features', default=None)
    a = ap.parse_args()
    as_of = date.fromisoformat(a.as_of) if a.as_of else date.today()
    csv = a.features or latest_features_csv()
    age = (datetime.now(timezone.utc) - datetime.fromtimestamp(os.path.getmtime(csv), timezone.utc)).days
    if age > MAX_AGE_DAYS and not a.dry_run:
        raise SystemExit(f"REFUSE: {csv} is {age} days old (> {MAX_AGE_DAYS}) — nightly features stale")
    from trading.orb_csv import read_orb_csv
    df = read_orb_csv(csv)
    params, cutoffs, meta = fit_window(df, as_of)
    text = ORB_YAML.read_text()
    new_text = rewrite_yaml(text, params, cutoffs)
    import yaml
    old = yaml.safe_load(text); new = yaml.safe_load(new_text)
    assert old['adaptive_mults'] == new['adaptive_mults'], "mults must never change"
    print(f"ORB weekly refit — window {meta['window_start']} → {meta['window_end']} ({meta['n_rows']} candidates, "
          f"{meta['n_above_threshold']} above threshold) from {csv}")
    for f, p in params.items():
        o = old['filter']['features'][f]
        print(f"  {f:26s} mean {o['mean']:.4f} → {p['mean']:.4f} | std {o['std']:.4f} → {p['std']:.4f}")
    print(f"  cutoffs {[round(c, 4) for c in old['quintile_cutoffs']]} → {[round(c, 4) for c in cutoffs]}")
    if a.dry_run:
        print("DRY RUN — nothing written")
        return 0
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    shutil.copy2(ORB_YAML, ORB_YAML.with_name(f'orb.yaml.bak.refit_{ts}'))
    ORB_YAML.write_text(new_text)
    HISTORY.parent.mkdir(exist_ok=True)
    with open(HISTORY, 'a') as fh:
        fh.write(json.dumps(dict(ts=ts, **meta, features_csv=csv, params=params, cutoffs=cutoffs)) + '\n')
    print(f"WRITTEN orb.yaml (backup orb.yaml.bak.refit_{ts}); history → {HISTORY}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
