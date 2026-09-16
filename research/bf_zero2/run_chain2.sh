#!/bin/bash
cd /home/ec2-user/onemil; L=research/bf_zero2/chain2.log; : > $L; ulimit -v 5500000
echo "$(date -u +%T) pass 2b start" >> $L
nice -n 10 python3 research/bf_zero2/pass2b.py > research/bf_zero2/pass2.log 2>&1; echo "PASS2_EXIT=$?" >> $L
echo "$(date -u +%T) score start" >> $L
nice -n 10 python3 research/bf_zero2/score2.py > research/bf_zero2/score.log 2>&1; echo "SCORE_EXIT=$?" >> $L
echo "$(date -u +%T) CHAIN2 DONE" >> $L
