#!/usr/bin/env python3
"""ORB V1 veto study runner — DESIGN.md is the contract. Baseline + each veto
through the honest B+ pipeline via the exit-resim dump; outputs stay in this
directory. Prints the pass-rule table and writes summary.csv."""
import os, subprocess, sys
import pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.dirname(os.path.abspath(__file__))
FEATURES = 'analysis_results/orb_features_20260905_1940.csv'
DUMP = 'research/orb_veto_study/candidates_static_lock_dump.csv'   # produced by a full bar walk with ORB_BT_DUMP_CANDIDATES (static-lock exits)
RUNS = {
    'baseline': '',
    'V1a_range_size': 'range_size_pct<=2.221',
    'V1b_adr20': 'avg_daily_range_pct_20d<=5.01',
    'V1c_spy3d': 'spy_3d_range_pct>=1.484',
    'V1d_bar_range': 'range_avg_bar_range_pct<=0.943',
    'V1e_retvol20': 'return_volatility_20d<=3.798',
    'V1all_abc': 'range_size_pct<=2.221;avg_daily_range_pct_20d<=5.01;spy_3d_range_pct>=1.484',
}

def run(name, veto):
    if not os.path.exists(os.path.join(ROOT, DUMP)):
        sys.exit(f"FATAL: {DUMP} missing — run the full walk first: ORB_BT_DUMP_CANDIDATES={DUMP}")
    env = dict(os.environ, ORB_BT_FEATURES_CSV=FEATURES, ORB_BT_RESIM_CACHE=DUMP,
               ORB_BT_BOOK_OUT=f'{OUT}/{name}_book.csv', ORB_BT_MONTHLY_OUT=f'{OUT}/{name}_monthly.csv')
    if veto: env['ORB_EXP_FEAT_VETO'] = veto
    else: env.pop('ORB_EXP_FEAT_VETO', None)
    r = subprocess.run([sys.executable, 'study_orb_pipeline_static_lock.py'], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    open(f'{OUT}/{name}.log', 'w').write(r.stdout + '\n--- STDERR ---\n' + r.stderr)
    if r.returncode != 0:
        print(f'{name}: FAILED rc={r.returncode} — see {name}.log', flush=True); return None
    b = pd.read_csv(f'{OUT}/{name}_book.csv'); m = pd.read_csv(f'{OUT}/{name}_monthly.csv')
    b['date'] = pd.to_datetime(b['date'])
    pcol = '_sized_pnl' if '_sized_pnl' in b.columns else 'pnl'   # stage-sized, same scale as the monthly table
    era = lambda a, z: float(b[(b.date >= a) & (b.date <= z)][pcol].sum())
    cum = m.pnl.cumsum(); mdd = float((cum - cum.cummax()).min())
    s = dict(run=name, picks=len(b), entered=int(b.entered.sum()) if 'entered' in b else None,
             total=round(m.pnl.sum()), mdd=round(mdd), red=int((m.pnl < 0).sum()), months=len(m),
             e25H1=round(era('2025-01-01', '2025-06-30')), e25H2=round(era('2025-07-01', '2025-12-31')),
             e2026=round(era('2026-01-01', '2026-12-31')), worst_mo=round(m.pnl.min()))
    print(s, flush=True); return s

rows = [r for r in (run(n, v) for n, v in RUNS.items()) if r]
S = pd.DataFrame(rows).set_index('run')
base = S.loc['baseline']
def verdict(r):
    if r.name == 'baseline': return ''
    ok = (r.total >= base.total and r.mdd >= base.mdd and r.red <= base.red
          and all(r[e] >= base[e] - 100 for e in ('e25H1', 'e25H2', 'e2026')))
    return 'KEEP' if ok else 'REJECT'
S['verdict'] = [verdict(r) for _, r in S.iterrows()]
S.to_csv(f'{OUT}/summary.csv')
print('\n' + S.to_string(), flush=True)
