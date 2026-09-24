#!/bin/bash
# Repair of cell 1,417's run (2026-09-24): re-run Stage 1 for the two failed months SEQUENTIALLY into a side cache,
# merge with the good months, run Stage 2 on the merged cache (PREREG risk_cap pin active), score. Detached.
set -uo pipefail
cd /home/ec2-user/onemil
D=research/bf_2024
LOG=$D/repair.log
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }
h=$(date -u +%H%M); if [ "$h" -ge 1325 ] && [ "$h" -lt 2005 ]; then log "REFUSED inside market hours"; exit 1; fi
log "REPAIR START: Stage 1 2024-08-01..2024-09-30, sequential months, side cache"
BF24_STAGE=1 BF24_SEQ_MONTHS=1 BF24_START=2024-08-01 BF24_END=2024-09-30 \
  BF24_CACHE=$D/cache_2024_augsep.csv BF24_OUT_MONTHLY=$D/backtest_results_2024_augsep \
  python3 $D/run_bf_2024.py >> "$LOG" 2>&1
log "STAGE 1 exit=$?"
python3 - >> "$LOG" 2>&1 <<'PY'
import pandas as pd
D='research/bf_2024'
a=pd.read_csv(f'{D}/cache_2024.csv', keep_default_na=False, na_values=[''])
b=pd.read_csv(f'{D}/cache_2024_augsep.csv', keep_default_na=False, na_values=[''])
m=pd.concat([a,b], ignore_index=True).drop_duplicates(['symbol','date','entry_time_et'])
m=m.sort_values(['date','entry_time_et','symbol']); m.to_csv(f'{D}/cache_2024_full.csv', index=False)
print('MERGE', len(a), '+', len(b), '->', len(m), 'rows; months', sorted(m.date.astype(str).str[:7].unique()))
PY
log "MERGE exit=$?"
BF24_STAGE=2 BF24_CACHE=$D/cache_2024_full.csv BF24_STAGE2_OUT=$D/stage2_2024_full.csv \
  BF24_OUT_MONTHLY=$D/backtest_results_2024_stage2 python3 $D/run_bf_2024.py >> "$LOG" 2>&1
log "STAGE 2 exit=$?"
BF24_STAGE2_OUT=$D/stage2_2024_full.csv python3 $D/score.py >> "$LOG" 2>&1
log "SCORE exit=$?"
log "REPAIR DONE"
