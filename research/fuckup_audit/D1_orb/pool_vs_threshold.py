"""Candidate-pool size as a function of the composite threshold / Q1 switch.

Used only to CHOOSE the two thresholds that give ~2x and ~3x the shipped B+
post-filter pool, so the 'wider pool' cells in Task B are declared in advance.
"""
import os
import sys
import pandas as pd
import yaml
sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
from trading.orb_csv import read_orb_csv
from study_orb_filter import FILTER_FEATURES, composite_score
from study_orb_sizing import assign_quintile

cfg = yaml.safe_load(open('orb.yaml'))
filt = cfg['filter']
Z = {f: {'mean': float(filt['features'][f]['mean']),
         'std': float(filt['features'][f]['std']),
         'sign': int(filt['features'][f]['sign'])} for f, _ in FILTER_FEATURES}
THR = float(filt['threshold'])
CUT = [float(x) for x in cfg['quintile_cutoffs']]

f = read_orb_csv('analysis_results/orb_features_20260916_2053.csv')
f = f.dropna(subset=[x for x, _ in FILTER_FEATURES]).copy()
f['_c'] = composite_score(f, Z)
f['_q'] = assign_quintile(f['_c'], CUT)
base = f[(f['_c'] >= THR) & (f['_q'] != 'Q1')]
print(f"candidates with features: {len(f)}")
print(f"B+ pool (composite >= {THR}, Q1 dropped): {len(base)}")
print(f"Q1-off pool (composite >= {THR}):        "
      f"{int((f['_c'] >= THR).sum())}  "
      f"= {(f['_c'] >= THR).sum()/len(base):.2f}x")
print("\nthreshold -> pool (Q1 still dropped / Q1 kept)")
for t in [THR, 0.0, -0.05, -0.1, -0.15, -0.2, -0.3, -0.4, -0.5, -0.7, -1.0, -99]:
    a = f[f['_c'] >= t]
    b = a[a['_q'] != 'Q1']
    print(f"  {t:>8.4f}: Q1-off {len(b):5d} ({len(b)/len(base):4.2f}x)   "
          f"Q1-on {len(a):5d} ({len(a)/len(base):4.2f}x)")
