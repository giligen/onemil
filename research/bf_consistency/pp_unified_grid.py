"""Profit partial on the UNIFIED spec (trading/bf_profit_partial.py, plan-R,
remainder keeps the real trail/exhaustion/vol-guard) — resim on regen-7 with the
rich master, then Stage-2. Compare with the legacy-branch sweep."""
import copy, os, subprocess, sys, yaml
sys.path.insert(0,'/home/ec2-user/onemil'); sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
from trading.orb_csv import read_orb_csv
ROOT='/home/ec2-user/onemil'; SP='/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad'; OUT=f'{ROOT}/research/bf_consistency'
base = yaml.safe_load(open(f'{ROOT}/config.yaml'))
rows=[]
for r in (1.5, 2.0, 2.5):
    for frac in (0.5, 0.67):
        name=f'PPU_r{r}_f{frac}'
        cfg=copy.deepcopy(base); t=cfg['trading']
        t['profit_partial']={'enabled':True,'r_multiple':r,'fraction':frac,'move_to_breakeven':True}
        t.pop('partial_profit', None)
        cp=f'{SP}/cfg_ppu_{name}.yaml'; yaml.safe_dump(cfg, open(cp,'w'), sort_keys=False)
        resim=f'{OUT}/resim_{name}.csv'
        subprocess.run(['/usr/bin/python3','batch_backtest.py','--config',cp,'--resim-exits',resim,'--resim-rich','backtest_results/backtest_full_2025_01_to_2026_09.csv','--start','2025-01-01','--end','2026-09-04'], env=dict(os.environ, BT_CACHE_PATH_OVERRIDE='data/bull_flag_cache_causal_full_20260905.csv'), capture_output=True, text=True, cwd=ROOT)
        subprocess.run(['/usr/bin/python3','batch_backtest.py','--config',cp,'--start','2025-01-01','--end','2026-09-04','--capital','50000','--risk','2000','--max-shares','10000'], env=dict(os.environ, BT_CACHE_PATH_OVERRIDE=resim), capture_output=True, text=True, cwd=ROOT)
        b=read_orb_csv(f'{ROOT}/backtest_results_march_2026.csv'); b['date']=b['date'].astype(str).str[:10]
        b.to_csv(f'{OUT}/stage2_{name}.csv', index=False)
        d=b.groupby('date').pnl.sum().sort_index(); c=d.cumsum(); m=b.groupby(b.date.str[:7]).pnl.sum()
        rows.append({'r':r,'frac':frac,'trades':len(b),'WR%':round((b.pnl>0).mean()*100,1),'total':round(b.pnl.sum()),'2025':round(m[m.index<'2026'].sum()),'2026':round(m[m.index>='2026'].sum()),'green':f"{int((m>0).sum())}/{len(m)}",'worst_mo':round(m.min()),'mdd':round(float((c-c.cummax()).min())),'top5%':round(b.nlargest(5,'pnl').pnl.sum()/b.pnl.sum()*100) if b.pnl.sum()>0 else None,'pp_fired':int(b.exit_reason.astype(str).str.startswith('pp+').sum())})
        print(rows[-1])
df=pd.DataFrame(rows); df.to_csv(f'{OUT}/pp_unified_grid.csv', index=False); print(df.to_string(index=False)); print('PPU DONE')
