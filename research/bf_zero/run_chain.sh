#!/bin/bash
# bf_zero orchestrator (2026-09-13): wait for the Databento fetch and the first scan pass, stay out of the
# 20:30-23:45 UTC nightly-cron window (memory), then retry-pass the fetched pairs, pass 2, score, Telegram.
cd /home/ec2-user/onemil
D=research/bf_zero
log() { echo "$(date -u +%FT%TZ) $*" >> $D/chain.log; }
log "chain start"
until grep -q "FETCH_EXIT=" $D/fetch.log && grep -q "BUILD_EXIT=" $D/build.log; do sleep 120; done
log "fetch: $(grep FETCH_EXIT $D/fetch.log) | scan: $(grep BUILD_EXIT $D/build.log) | rows $(wc -l < $D/candidates.csv)"
H=$(date -u +%H%M)
if [ "$H" -ge 2030 ] && [ "$H" -lt 2345 ]; then
  log "inside the nightly window, sleeping until 23:45 UTC"
  while [ "$(date -u +%H%M)" -lt 2345 ]; do sleep 120; done
fi
log "retry pass start ($(wc -l < $D/coverage_missing.csv) missing pairs)"
( ulimit -v 4500000; BFZ_RETRY=1 nice -n 10 /usr/bin/python3 $D/build_candidates.py >> $D/build_retry.log 2>&1 ); log "retry exit $?"
log "pass2 start"; ( ulimit -v 5000000; nice -n 10 /usr/bin/python3 $D/pass2.py > $D/pass2.log 2>&1 ); log "pass2 exit $?"
log "score start"; ( ulimit -v 5000000; nice -n 10 /usr/bin/python3 $D/score.py > $D/score.log 2>&1 ); log "score exit $?"
log "slip sensitivity skipped in chain (needs a rescan; run on the surviving configs only)"
SUMMARY="$(grep -E '^candidates|STEP 1|STEPS 2-4|CANDIDATES' $D/score.log | head -6)"
/usr/bin/python3 scripts/send_telegram_alert.py "BF FROM ZERO — scoring finished (research/bf_zero/score.log)
${SUMMARY}
Report + tables follow in the morning brief." || true
log "chain done"
