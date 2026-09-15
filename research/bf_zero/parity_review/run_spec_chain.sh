#!/bin/bash
# parity review 2026-09-15: after the SIP refetch — verify 20 keys, wait for an idle spec_sim, back up + prune the
# affected days, re-run spec_sim on the SIP store, then capacity_8a. Logs under parity_review/.
cd /home/ec2-user/onemil
PR=research/bf_zero/parity_review; L=$PR/spec_chain.log; : > $L
until grep -q "FETCH_EXIT" $PR/refetch.log; do sleep 30; done
echo "$(date -u +%T) fetch finished: $(grep FETCH_EXIT $PR/refetch.log)" >> $L
ulimit -v 3000000
nice -n 10 python3 research/bf_zero/refetch_thin_tape.py --verify 20 >> $L 2>&1; echo "VERIFY_EXIT=$?" >> $L
while pgrep -f "python3 spec_sim.py" > /dev/null; do sleep 30; done
echo "$(date -u +%T) no spec_sim running — pruning affected days" >> $L
nice -n 10 python3 - >> $L 2>&1 <<'PY'
import json, sqlite3, shutil, pandas as pd
D = 'research/bf_zero'
shutil.copy(f'{D}/spec_trades.csv', f'{D}/spec_trades.pre_refetch.csv'); shutil.copy(f'{D}/spec_state.json', f'{D}/spec_state.pre_refetch.json')
con = sqlite3.connect(f'file:{D}/bars_sip.db?mode=ro', uri=True)
days = {r[0] for r in con.execute("select distinct day from fetch_log where n_bars > 0")}
# FULL re-simulation: spec_sim now seeds from the whole causal superset (not the thin-tape F5 rows), so every day changes
st = json.load(open(f'{D}/spec_state.json')); before = len(st['done']); st['done'] = []
json.dump(st, open(f'{D}/spec_state.json', 'w'))
T = pd.read_csv(f'{D}/spec_trades.csv', dtype=str, keep_default_na=False); n0 = len(T); T = T.iloc[0:0]; T.to_csv(f'{D}/spec_trades.csv', index=False)
print(f'refetched days {len(days)} | state done {before} -> 0 (full re-run) | spec_trades rows {n0} -> 0 (backup kept)', flush=True)
PY
echo "$(date -u +%T) spec_sim start (BFZ_SIP_STORE)" >> $L
BFZ_SIP_STORE=/home/ec2-user/onemil/research/bf_zero/bars_sip.db nice -n 10 python3 research/bf_zero/spec_sim.py > $PR/spec_sim_sip.log 2>&1; echo "SPEC_EXIT=$?" >> $L
nice -n 10 python3 research/bf_zero/capacity_8a.py > $PR/capacity_8a_sip.log 2>&1; echo "CAP_EXIT=$?" >> $L
echo "$(date -u +%T) CHAIN DONE" >> $L
