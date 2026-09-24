#!/bin/bash
# Data prep for the 2023-2024H1 out-of-regime ORB test: consolidated daily bars + universe, then minute bars.
cd /home/ec2-user/onemil
python3 research/orb_2023/build_universe.py > research/orb_2023/universe.log 2>&1 && \
python3 research/orb_2023/fetch_minutes.py > research/orb_2023/fetch.log 2>&1
echo "PREP DONE rc=$?" >> research/orb_2023/universe.log
