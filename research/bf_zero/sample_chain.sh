#!/bin/bash
# after the unbiased 10%-symbol all-days fetch: scan those symbol-days with NO range gate, then compare F5 on the sample
cd /home/ec2-user/onemil; D=research/bf_zero
log() { echo "$(date -u +%FT%TZ) $*" >> $D/sample_chain.log; }
until grep -q "FETCH_EXIT=" $D/sample_fetch.log; do sleep 120; done
log "sample fetch: $(grep FETCH_EXIT $D/sample_fetch.log)"
sed -e "s#STATE, OUT, MISS = f'{D}/build_state.json', f'{D}/candidates.csv', f'{D}/coverage_missing.csv'#STATE, OUT, MISS = f'{D}/sample_state.json', f'{D}/sample_candidates.csv', f'{D}/sample_missing.csv'#" \
    -e "s#VPROF = f'{D}/volume_profile.csv'#VPROF = f'{D}/sample_volume_profile.csv'#" \
    -e "s#uni = pd.read_csv(f'{D}/universe.csv'#uni = pd.read_csv(f'{D}/sample_universe.csv'#" $D/build_candidates.py > $D/sample_build.py
( ulimit -v 4500000; nice -n 10 /usr/bin/python3 $D/sample_build.py > $D/sample_build.log 2>&1 ); log "sample scan exit $?"
log "done"
