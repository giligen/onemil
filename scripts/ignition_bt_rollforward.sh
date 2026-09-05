#!/bin/bash
# Ignition BT roll-forward (2026-09-05, owner: "don't you have a cron that
# compares BT to live?"). The 19-month capsim/resting study was frozen at
# 2026-08-14; live started 8/21 with no BT twin. This keeps the BT current
# every night so the EOD dive can match every live trade to its BT twin.
#
# Chain (all resumable per day via state files; frozen 19-month artifacts
# are never rewritten — everything lands in *_LIVEWIN* files):
#   1. candidate universe + bar coverage 2026-08-15..today (cache.db)
#   2. Databento 1-min top-up for uncovered symbol-days (cents/day)
#   3. capsim LIVEWIN (chase-entry baseline)  4. annotate (anchor/cohort)
#   5. resting_sim LIVEWIN (pre-staged fill model)  6. live-vs-shadow-vs-BT table
set -u
cd /home/ec2-user/onemil || exit 1
ulimit -v 3500000
D=research/ignition_capcheck
START=2026-08-15
END=$(date -u +%F)
LOG=logs/ignition_bt_rollforward.log
{
  echo "=== $(date -u +%FT%TZ) roll-forward $START..$END ==="
  /usr/bin/python3 $D/build_universe_addon.py "$START" "$END" $D/universe_live_window.csv || exit 2
  /usr/bin/python3 - <<'EOF' || exit 3
import pandas as pd
u = pd.read_csv('research/ignition_capcheck/universe_live_window.csv')
u[~u.covered][['symbol', 'bar_date']].to_csv('research/ignition_capcheck/missing_keys_live_window.csv', index=False)
print('uncovered keys:', int((~u.covered).sum()))
EOF
  /usr/bin/python3 $D/fetch_missing_databento.py "$END" $D/missing_keys_live_window.csv $D/topup.db $D/fetch_databento_state_live_window.json || exit 4
  ( cd $D && CAPSIM_UNIVERSE=universe_live_window.csv /usr/bin/python3 capsim.py LIVEWIN "$START" "$END" ) || exit 5
  ( cd $D && /usr/bin/python3 annotate_trades.py trades_LIVEWIN.csv trades_LIVEWIN_annotated.csv ) || exit 6
  ( cd $D && RESTING_TRADES=trades_LIVEWIN_annotated.csv RESTING_UNIVERSE=universe_live_window.csv /usr/bin/python3 resting_sim.py LIVEWIN "$START" "$END" ) || exit 7
  /usr/bin/python3 $D/live_window_compare.py 2026-08-21 "$END" || exit 8
  echo "=== done $(date -u +%FT%TZ) ==="
} >> "$LOG" 2>&1
