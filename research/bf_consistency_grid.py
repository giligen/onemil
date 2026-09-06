"""BF consistency grid (2026-09-06, owner: "shaving P&L for WR and consistency
is the right move"). Exit profiles re-simulated on the regen-7 cache with the
rich-master pattern fields (faithful to Stage-1 on 705/745 rows), then Stage-2
with the SAME scratch config. Scored on consistency metrics.

Pre-committed selection rule: 2026 P&L >= 0 AND green months >= 70% AND worst
month >= -$8K AND MDD >= -$20K (at $2K risk); among those, highest
mean-month / worst-month ratio; total P&L only breaks ties. Baseline = V0.
"""
import copy, os, re, subprocess, sys
import pandas as pd, yaml
sys.path.insert(0, '/home/ec2-user/onemil'); sys.stdout.reconfigure(line_buffering=True)
from trading.orb_csv import read_orb_csv
ROOT = '/home/ec2-user/onemil'; SP = '/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad'
CACHE = 'data/bull_flag_cache_causal_full_20260905.csv'; RICH = 'backtest_results/backtest_full_2025_01_to_2026_09.csv'
OUT = f'{ROOT}/research/bf_consistency'; os.makedirs(OUT, exist_ok=True)
base = yaml.safe_load(open(f'{ROOT}/config.yaml'))
def setp(cfg, path, val):
    d = cfg
    for k in path[:-1]: d = d.setdefault(k, {})
    d[path[-1]] = val
T = ['trading']
V = {
 'V0_asis': [],
 'V1_partial1R_be': [(T+['partial_profit','enabled'], True), (T+['partial_profit','r_multiple'], 1.0), (T+['partial_profit','fraction'], 0.5), (T+['trailing_stop','breakeven_at_r'], 1.0)],
 'V2_partial1.5R_be': [(T+['partial_profit','enabled'], True), (T+['partial_profit','r_multiple'], 1.5), (T+['partial_profit','fraction'], 0.5), (T+['trailing_stop','breakeven_at_r'], 1.5)],
 'V3_partial1R_act1.5_trail0.75': [(T+['partial_profit','enabled'], True), (T+['partial_profit','r_multiple'], 1.0), (T+['partial_profit','fraction'], 0.5), (T+['trailing_stop','activate_at_r'], 1.5), (T+['trailing_stop','trail_r'], 0.75)],
 'V4_exhaust2R': [(T+['exhaustion_exit','min_profit_r'], 2.0)],
 'V5_nopop10': [(T+['no_pop_exit','enabled'], True), (T+['no_pop_exit','bars'], 10), (T+['no_pop_exit','min_pct'], 0.005)],
 'V6_partial1R_be_nopop10': [(T+['partial_profit','enabled'], True), (T+['partial_profit','r_multiple'], 1.0), (T+['partial_profit','fraction'], 0.5), (T+['trailing_stop','breakeven_at_r'], 1.0), (T+['no_pop_exit','enabled'], True), (T+['no_pop_exit','bars'], 10), (T+['no_pop_exit','min_pct'], 0.005)],
 'V7_act1.5_trail1': [(T+['trailing_stop','activate_at_r'], 1.5)],
 'V8_be1R': [(T+['trailing_stop','breakeven_at_r'], 1.0)],
}
rows = []
for name, changes in V.items():
    cfg = copy.deepcopy(base)
    for p, v in changes: setp(cfg, p, v)
    cp = f'{SP}/cfg_bfc_{name}.yaml'; yaml.safe_dump(cfg, open(cp, 'w'), sort_keys=False)
    resim = f'{OUT}/resim_{name}.csv'
    env = dict(os.environ, BT_CACHE_PATH_OVERRIDE=CACHE)
    r1 = subprocess.run(['/usr/bin/python3', 'batch_backtest.py', '--config', cp, '--resim-exits', resim, '--resim-rich', RICH, '--start', '2025-01-01', '--end', '2026-09-04'], env=env, capture_output=True, text=True, cwd=ROOT)
    if not os.path.exists(resim):
        print(name, 'RESIM FAILED', r1.stdout[-400:], r1.stderr[-400:]); continue
    env2 = dict(os.environ, BT_CACHE_PATH_OVERRIDE=resim)
    r2 = subprocess.run(['/usr/bin/python3', 'batch_backtest.py', '--config', cp, '--start', '2025-01-01', '--end', '2026-09-04', '--capital', '50000', '--risk', '2000', '--max-shares', '10000'], env=env2, capture_output=True, text=True, cwd=ROOT)
    b = read_orb_csv(f'{ROOT}/backtest_results_march_2026.csv'); b['date'] = b['date'].astype(str).str[:10]
    b.to_csv(f'{OUT}/stage2_{name}.csv', index=False)
    d = b.groupby('date').pnl.sum().sort_index(); c = d.cumsum(); m = b.groupby(b.date.str[:7]).pnl.sum()
    top5 = b.nlargest(5, 'pnl').pnl.sum()
    rows.append({'variant': name, 'trades': len(b), 'WR%': round((b.pnl > 0).mean() * 100, 1), 'total': round(b.pnl.sum()), '2025': round(m[m.index < '2026'].sum()), '2026': round(m[m.index >= '2026'].sum()),
                 'green_mo': f"{int((m > 0).sum())}/{len(m)}", 'mean_mo': round(m.mean()), 'median_mo': round(m.median()), 'worst_mo': round(m.min()), 'mdd': round(float((c - c.cummax()).min())), 'top5_share%': round(top5 / b.pnl.sum() * 100) if b.pnl.sum() > 0 else None,
                 'avg_win': round(b[b.pnl > 0].pnl.mean()), 'avg_loss': round(b[b.pnl <= 0].pnl.mean())})
    print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/grid_summary.csv', index=False)
print("\n" + df.to_string(index=False))
ok = df[(df['2026'] >= 0) & (df.green_mo.str.split('/').str[0].astype(int) / df.green_mo.str.split('/').str[1].astype(int) >= 0.7) & (df.worst_mo >= -8000) & (df.mdd >= -20000)]
print("\nPASS the pre-committed consistency bar:", ok.variant.tolist() or 'none')
print("GRID DONE")
