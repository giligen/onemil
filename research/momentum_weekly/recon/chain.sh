cd /home/ec2-user/onemil
P=research/momentum_weekly/recon
nice -n 15 python3 -u $P/A_dump.py > $P/A.out 2>&1; echo "A rc=$?" >> $P/chain.log
run(){ B_TAG=$1 B_FLAGS=$2 nice -n 15 python3 -u $P/B_dump.py > $P/B_$1.out 2>&1; echo "B $1 rc=$?" >> $P/chain.log; }
run base ""
run eqw EQW
run aexcl AEXCL,NODOTS
run acost ACOST
run clean CLEAN
run all AEXCL,NODOTS,ACOST,CLEAN,EQW
echo DONE >> $P/chain.log
