#!/bin/bash
# bf_zero2: fetch every universe symbol-day (open >= 5 first) not yet in bars_sip.db from Alpaca SIP; stops itself if disk < 4 GB.
cd /home/ec2-user/onemil; L=research/bf_zero2/fetch_universe.log
ulimit -v 3000000
( while true; do free_kb=$(df --output=avail -k /home/ec2-user | tail -1); if [ "$free_kb" -lt 4000000 ]; then echo "$(date -u +%T) DISK LOW ($free_kb KB) — stopping the fetch" >> $L; pkill -f "refetch_thin_tape.py --fetch --keys"; break; fi; sleep 120; done ) &
nice -n 10 python3 research/bf_zero/refetch_thin_tape.py --fetch --keys research/bf_zero2/universe_keys_nocache.csv --src universe --min-open 5 >> $L 2>&1; echo "FETCH5_EXIT=$?" >> $L
echo "$(date -u +%T) disk after: $(df -h /home/ec2-user | tail -1)" >> $L
