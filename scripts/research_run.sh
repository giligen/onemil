#!/bin/bash
# Run a research job inside a hard resource cage so it can never starve the live trading service.
#   bash scripts/research_run.sh [-m 3G] <command ...>
# The job runs in its own systemd scope with a memory ceiling (default 3G; the kernel kills the JOB, not the
# box, if it exceeds it — no swap storm), low CPU and IO weight (the trader always wins contention), nice 19.
# Why: 2026-10-02 the instance froze when research jobs took 5.7 GB next to the service on 8 GB / 2 vCPU.
# Use this for EVERY research/backfill/pytest run during market hours (12:30-20:10 UTC).
set -u
MEM="3G"
if [ "${1:-}" = "-m" ]; then MEM="$2"; shift 2; fi
if [ $# -eq 0 ]; then echo "usage: $0 [-m 3G] <command ...>" >&2; exit 2; fi
exec sudo -n systemd-run --scope --quiet --collect \
    --uid="$(id -un)" --gid="$(id -gn)" \
    -p MemoryMax="$MEM" -p MemorySwapMax=512M -p CPUWeight=10 -p IOWeight=10 \
    --setenv=HOME="$HOME" --setenv=PATH="$PATH" \
    nice -n 19 "$@"
